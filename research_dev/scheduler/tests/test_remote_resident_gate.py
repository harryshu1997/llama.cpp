"""The phone-owner preflight never claims a completed physical gate."""

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from dataclasses import replace

from research_dev.scheduler.adapters import (
    LlamaCppCompletionPayload, LlamaCppHttpClient, PhysicalAdapterError,
)
from research_dev.scheduler.campaigns.burstgpt import remote_resident_gate as gate
from research_dev.scheduler._internal.route_generation.costing_parameters import RouteParameterMixin
from research_dev.scheduler._internal.route_generation.common import RouteGenerationError
from research_dev.scheduler._internal.route_generation.feasibility import resident_phone_workspace
from research_dev.scheduler.adapters.ffn_shards import FfnShardIndex, FfnShardRecord
from research_dev.scheduler.configuration.campaign import FixedPhoneResidencyConfiguration


class MultiSessionOwnerTests(unittest.TestCase):
    def fixture(self):
        from test_remote_resident_launch import MANIFEST
        records = tuple(FfnShardRecord(
            session_hint=f"HTP{index}", remote_path=f"/phone/HTP{index}.ffn.gguf",
            shard_sha256="sha256:" + str(index + 1) * 64, parent_sha256=MANIFEST.artifact_sha256,
            layer_mask=0xFF << (8 * index), columns=MANIFEST.feed_forward_length,
            n_ff=MANIFEST.feed_forward_length, shard_bytes=2_831_157_504, weight_type="F16",
        ) for index in range(3))
        fixed = FixedPhoneResidencyConfiguration("three-session-test", tuple(
            (row.session_hint, row.parent_sha256, row.layer_mask, row.columns) for row in records))
        args = SimpleNamespace(
            session_id="HTP0", remote_layer_mask="0xffffff", owner="phone:session://phone/HTP0",
            fixed_phone_residency_json=json.dumps(fixed.to_json()), fallback_mode="teardown",
        )
        index = FfnShardIndex(MANIFEST.artifact_sha256, records, "sha256:" + "a" * 64)
        return args, MANIFEST, fixed, index

    def test_three_session_assignment_resolves_each_existing_shard(self):
        args, manifest, expected, index = self.fixture()
        fixed, records = gate._phone_owner_assignment(args, manifest, index)
        self.assertEqual(fixed, expected)
        self.assertEqual(tuple(records), ("HTP0", "HTP1", "HTP2"))
        self.assertEqual(sum(row.layer_mask for row in records.values()), 0xFFFFFF)

    def test_legacy_one_session_assignment_is_unchanged(self):
        args, manifest, _, index = self.fixture()
        args.fixed_phone_residency_json, args.remote_layer_mask = None, "0xff"
        fixed, records = gate._phone_owner_assignment(args, manifest, index)
        self.assertEqual(fixed.assignments, (("HTP0", manifest.artifact_sha256, 0xFF,
                                             manifest.feed_forward_length),))
        self.assertEqual(tuple(records), ("HTP0",))

    def test_missing_or_incompatible_shard_fails_before_preparation(self):
        args, manifest, _, index = self.fixture()
        for changed in (None, replace(index, records=index.records[:2]),
                        replace(index, parent_sha256="sha256:" + "b" * 64),
                        replace(index, records=(replace(index.records[0], weight_type="Q8_0"),
                                               *index.records[1:]))):
            with self.subTest(index=changed), self.assertRaises(RuntimeError):
                gate._phone_owner_assignment(args, manifest, changed)
        args.remote_layer_mask = "0xffff"
        with self.assertRaisesRegex(RuntimeError, "differs from omitted layers"):
            gate._phone_owner_assignment(args, manifest, index)

    def test_packed_assignment_and_anchor_endpoint_must_match(self):
        args, _, fixed, _ = self.fixture()
        shards = tuple(SimpleNamespace(session_id=session, artifact_sha256=artifact,
            layer_mask=mask, maximum_columns=columns, endpoint=f"session://phone/{session}")
            for session, artifact, mask, columns in fixed.assignments)
        gate._validate_phone_owner_layout(args, fixed, SimpleNamespace(shards=shards))
        with self.assertRaisesRegex(RuntimeError, "packed phone owners differ"):
            gate._validate_phone_owner_layout(args, fixed, SimpleNamespace(shards=shards[:2]))
        shards[0].endpoint = "session://other/HTP0"
        with self.assertRaisesRegex(RuntimeError, "packed session endpoint"):
            gate._validate_phone_owner_layout(args, fixed, SimpleNamespace(shards=shards))

    def accounting(self):
        from research_dev.scheduler._internal.runtime_resources import RuntimeRemoteResidentOmissionProof
        args, manifest, fixed, _ = self.fixture()
        shards = [{"session_id": session, "artifact_sha256": artifact, "layer_mask": mask,
                   "session_generation": index + 1, "resident_bytes": sum(
                       manifest.tensor_by_id[key].nbytes for key in gate.remote_resident_tensor_ids(mask))}
                  for index, (session, artifact, mask, _columns) in enumerate(fixed.assignments)]
        omitted = sum(row["resident_bytes"] for row in shards)
        memory = {"vm_rss_bytes": 16_000_000_000, "process_vram_bytes": 12_000_000_000,
                  "host_memory": {"MemAvailable": 10_000_000_000},
                  "gpu": {"memory_free_bytes": 3_000_000_000}}
        record = {"owner": {"kind": "phone"}, "arms": {"full": {"memory": memory},
            "reduced": {"memory": {**memory, "vm_rss_bytes": memory["vm_rss_bytes"] - omitted},
                        "ready_identity": {"phone_shards": shards},
                        "phone_workspace_reserved_bytes": 805_306_368}}}
        proof = RuntimeRemoteResidentOmissionProof(0xFFFFFF, omitted, omitted - 4096, "validated")
        return record, manifest, proof, omitted, 0xFFFFFF, args, "cuda0-vram"

    def test_accounting_uses_each_physical_session_without_global_generation_assumption(self):
        arguments = self.accounting()
        result = gate._accounting(*arguments)
        self.assertEqual(result["phone_weights_bytes_by_session"], {
            "HTP0": 2_831_155_200, "HTP1": 2_831_155_200, "HTP2": 2_831_155_200})
        self.assertEqual(result["reclaimed_bytes"], 8_493_465_600)
        self.assertEqual(result["phone_workspace_bytes"], 805_306_368)

    def test_accounting_rejects_incomplete_or_mismatched_physical_owners(self):
        for change in ({"session_generation": 0}, {"resident_bytes": 1},
                       {"artifact_sha256": "sha256:" + "b" * 64}, {"session_id": "HTP1"}):
            arguments = self.accounting()
            arguments[0]["arms"]["reduced"]["ready_identity"]["phone_shards"][0].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, "memory identity differs"):
                gate._accounting(*arguments)
        arguments = self.accounting()
        arguments[0]["arms"]["reduced"]["ready_identity"]["phone_shards"].pop()
        with self.assertRaisesRegex(RuntimeError, "memory coverage differs"):
            gate._accounting(*arguments)


class OutputComparisonTests(unittest.TestCase):
    def rows(self, tokens=(7, 8), error=None):
        return [{"request_id": "test", "tokens": list(tokens),
                 "output_tokens": 2, "error": error}]

    def test_semantic_mode_records_but_does_not_reject_token_difference(self):
        full, reduced = self.rows(), self.rows((7, 9))
        result = gate._completion_gate(full, reduced, 1, output_comparison="semantic-sanity")
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["exact_token_match_required"])
        self.assertEqual(result["token_agreement"]["identical_requests"], 0)
        self.assertEqual(result["token_agreement"]["pairs"][0]["first_divergence_index"], 1)
        self.assertEqual(gate._completion_gate(full, reduced, 1)["status"], "FAIL")

    def test_semantic_mode_keeps_failure_and_completion_checks(self):
        for rows in (self.rows(error="semantic quality failed"), self.rows((7,)), [], self.rows(())):
            with self.subTest(rows=rows):
                result = gate._completion_gate(self.rows(), rows, 1, output_comparison="semantic-sanity")
                self.assertEqual(result["status"], "FAIL")
        self.assertEqual(gate._completion_gate([], [], 0, output_comparison="semantic-sanity")["status"], "FAIL")

    def test_invalid_policy_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "output comparison policy is invalid"):
            gate._completion_gate(self.rows(), self.rows(), 1, output_comparison="ignore-all")


class LongContextGateTests(unittest.TestCase):
    def document(self, tokens, context=8192, minimum=4096):
        response = Mock(status=200)
        response.read.return_value = json.dumps({"tokens": tokens}).encode()
        connection = Mock()
        connection.getresponse.return_value = response
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text("A frozen document.")
            with patch.object(gate.http.client, "HTTPConnection", return_value=connection):
                rows = gate._document_rows("http://127.0.0.1:8080", path, context, 64, 2, minimum)
        connection.close.assert_called_once()
        return rows

    def test_document_tokens_are_identical_across_requests_and_not_truncated(self):
        tokens = [12] * 7000
        rows = self.document(tokens)
        self.assertEqual([row["prompt_tokens"] for row in rows], [tokens, tokens])
        self.assertEqual([row["input_tokens"] for row in rows], [7000, 7000])
        self.assertEqual([row["output_tokens"] for row in rows], [64, 64])

    def test_document_shape_fails_closed(self):
        for tokens, reason in (([1] * 100, "too short"), ([1] * 8128, "without truncation"),
                               ([True] * 7000, "invalid")):
            with self.subTest(reason=reason), self.assertRaisesRegex(RuntimeError, reason):
                self.document(tokens)

    def test_complete_prompt_requires_untruncated_terminal_counts(self):
        final = {"stop": True, "truncated": False, "timings": {"prompt_n": 7000, "predicted_n": 64}}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "response.raw"
            payload = SimpleNamespace(stream_path=path, input_tokens=7000, output_tokens=64)
            path.write_text("data: " + json.dumps(final) + "\n")
            self.assertEqual(gate._context_completion(payload, {})["terminal"], final)
            for changed in ({**final, "truncated": True}, {**final, "truncated": None},
                            {**final, "timings": {"prompt_n": 2048, "predicted_n": 64}}):
                path.write_text("data: " + json.dumps(changed) + "\n")
                with self.subTest(final=changed), self.assertRaises(RuntimeError):
                    gate._context_completion(payload, {})


class DiagnosticLogprobTests(unittest.TestCase):
    def payload(self, directory, count=0):
        return LlamaCppCompletionPayload(
            request_id="numeric", expected_model_alias="synthetic", input_tokens=1,
            output_tokens=1, prompt_tokens=(1,), seed=42,
            stream_path=Path(directory) / "stream.raw", on_first_token=lambda _at: None,
            quality_mode="accounting-only", diagnostic_top_logprobs=count,
        )

    def wire_request(self, count):
        chunks = [
            {"content": "word", "tokens": [7], "tokens_predicted": 1,
             "id_slot": 0, "stop": False},
            {"content": "", "tokens": [], "id_slot": 0, "stop": True,
             "model": "synthetic", "timings": {
                 "prompt_n": 1, "predicted_n": 1, "prompt_ms": 1, "predicted_ms": 1}},
        ]
        stream = io.BytesIO(b"".join(b"data: " + json.dumps(row).encode() + b"\n\n" for row in chunks))
        response = SimpleNamespace(status=200, readline=stream.readline)
        connection = Mock()
        connection.getresponse.return_value = response
        client = LlamaCppHttpClient(slots_probe=lambda *_args: [
            {"id": 0, "id_task": 9, "is_processing": True},
        ])
        with TemporaryDirectory() as directory, patch(
            "research_dev.scheduler.adapters.http_backend.http.client.HTTPConnection",
            return_value=connection,
        ):
            result = client.complete("http://127.0.0.1:1234", self.payload(directory, count), lambda: None)
            self.assertEqual(result["tokens"], [7])
        return json.loads(connection.request.call_args.kwargs["body"])

    def test_default_request_body_is_unchanged(self):
        self.assertEqual(self.wire_request(0), {
            "cache_prompt": False, "ignore_eos": True, "n_predict": 1,
            "prompt": [1], "return_tokens": True, "seed": 42,
            "stream": True, "temperature": 0.0,
        })

    def test_diagnostics_only_add_pre_sampling_observations(self):
        plain = self.wire_request(0)
        self.assertEqual(self.wire_request(32), {
            **plain, "n_probs": 32, "post_sampling_probs": False,
        })

    def test_invalid_diagnostic_counts_fail_before_execution(self):
        with TemporaryDirectory() as directory:
            for count in (-1, 257, True, "32"):
                with self.subTest(count=count), self.assertRaisesRegex(
                    PhysicalAdapterError, "diagnostic logprob count is invalid",
                ):
                    self.payload(directory, count)

    def test_gate_forwards_observations_without_changing_prompt_or_seed(self):
        client = Mock()
        client.complete.return_value = {"tokens": [7]}
        row = {"request_index": 1, "input_tokens": 1, "output_tokens": 1, "prompt_tokens": [1]}
        with TemporaryDirectory() as directory, patch.object(gate, "_energy", return_value={}):
            gate._run_requests(client, "http://127.0.0.1:1234", [row], "synthetic",
                               Path(directory), "full", object(), diagnostic_top_logprobs=32)
        payload = client.complete.call_args.args[1]
        self.assertEqual(payload.diagnostic_top_logprobs, 32)
        self.assertEqual((payload.prompt_tokens, payload.output_tokens, payload.seed), ((1,), 1, 42))


class ResidentPrefillCapacityTests(unittest.TestCase):
    def transport(self, declared=None):
        parameters = {"phone_device_id": "phone", "ffn_n_embd": 3840,
                      "ubatch_size": 512, "parallel": 1}
        if declared is not None:
            parameters["ffn_max_tokens"] = declared
        compiler = SimpleNamespace(_transport_adapter_parameters=Mock(return_value={}))
        RouteParameterMixin._candidate_transport_parameters(
            compiler, None, (), None, SimpleNamespace(assistance_phase="decode"), parameters,
        )
        return parameters, compiler._transport_adapter_parameters.call_args.args

    def test_explicit_resident_batch_capacity_survives_decode_materialization(self):
        parameters, arguments = self.transport(512)
        self.assertEqual(parameters["ffn_max_tokens"], 512)
        self.assertEqual(arguments[2], 3840 * 512 * 2)

    def test_ordinary_decode_capacity_is_unchanged(self):
        parameters, arguments = self.transport()
        self.assertEqual(parameters["ffn_max_tokens"], 1)
        self.assertEqual(arguments[2], 3840 * 2)

    def test_invalid_explicit_capacity_fails_before_transport(self):
        for value in (0, -1, 513, True, "512"):
            with self.subTest(value=value), self.assertRaises(RouteGenerationError):
                self.transport(value)

    def test_resident_prefill_workspace_is_reserved_without_changing_other_devices(self):
        from test_remote_resident_launch import MANIFEST
        capability = SimpleNamespace(workspace_bytes_per_token=4096)
        parameters = {"phone_device_id": "phone", "ffn_max_tokens": 512}
        expected = 513 * MANIFEST.embedding_length * 2 + 4096 * 512
        self.assertEqual(resident_phone_workspace(MANIFEST, capability, parameters, "phone", 1), expected)
        self.assertEqual(resident_phone_workspace(MANIFEST, capability, parameters, "cpu", 1), 1)
        self.assertEqual(resident_phone_workspace(MANIFEST, capability, parameters, "phone", expected * 2), expected * 2)

    def test_prefill_preparation_without_matching_transport_remains_ineligible(self):
        import test_offline_phone_residency as fixture
        from research_dev.scheduler import UnifiedScheduler, UnifiedScheduleError
        fixture.OfflinePhoneResidencyTests.setUpClass()
        source, requests, snapshot = fixture.OfflinePhoneResidencyTests.replay_runtime(load_evidence=False)
        catalog = source._runtime_capabilities
        catalog = replace(catalog, composite_executors=tuple(
            replace(row, adapter_parameters={**row.adapter_parameters,
                    "ffn_max_tokens": row.adapter_parameters["ubatch_size"],
                    "memory_workspace_minimum_bytes:op15-phone": 805306368})
            if row.assisted_operator_kind == "ffn" else row
            for row in catalog.composite_executors
        ))
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce", maximum_phone_sessions=1)
        scheduler.register_runtime_capabilities(catalog)
        for manifest in source._runtime_manifests.values():
            scheduler.register_model_manifest(manifest)
        with self.assertRaisesRegex(UnifiedScheduleError, "offline phone residency demand is unavailable"):
            scheduler.plan_offline_phone_residency(
                requests, snapshot=snapshot, observed_at_us=snapshot.captured_at_us,
            )


class PhoneOwnerPreflightTests(unittest.TestCase):
    def test_completed_arm_is_persisted_before_the_next_arm(self):
        with TemporaryDirectory() as directory:
            record = {"status": "RUNNING"}
            result = {"requests": [{"error": None}], "memory": {"vm_rss_bytes": 123}}
            gate._record_arm(record, Path(directory), "full", result)
            record["status"] = "FAILED"
            self.assertEqual(record["arms"]["full"], result)
            self.assertEqual(json.loads((Path(directory) / "full-ARM.json").read_text()), result)
            with self.assertRaisesRegex(RuntimeError, "artifact already exists"):
                gate._record_arm(record, Path(directory), "full", result)

    def _preflight(self, directory, error=None):
        args = SimpleNamespace(
            output=Path(directory), usb_qualification_identity=Path("identity.json"),
            server=Path("server"), transport_host_dependency=(("llama", Path("libllama.so")),),
        )
        identity = SimpleNamespace(
            software_identity=(
                {"host_binary_sha256": "old", "host_dependency_sha256:llama": "old-lib"}
                if error is not None else
                {"host_binary_sha256": "actual", "host_dependency_sha256:llama": "actual"}
            ),
            to_json=lambda: {"identity": "original"},
        )
        with (
            patch.object(gate, "load_transport_qualification_identity", return_value=identity),
            patch.object(gate, "_sha256", return_value="actual"),
            patch.object(gate.runner, "_direct_phone_configuration"),
            patch.object(gate, "DirectPhoneFfnSession") as phone,
        ):
            if error is not None:
                phone.return_value.preflight.side_effect = error
            else:
                phone.return_value.preflight.return_value.to_json.return_value = {"verified": True}
            result = gate._phone_owner_preflight(
                args, object(), {}, SimpleNamespace(to_json=lambda: {"gpu_layers": 23}), {}
            )
            phone.return_value.start.assert_not_called()
            phone.return_value.abort.assert_not_called()
        saved = json.loads((args.output / "PHONE_OWNER_PREFLIGHT.json").read_text())
        self.assertEqual(saved, result)
        return result

    def test_stale_identity_is_blocked_and_differences_are_preserved(self):
        with TemporaryDirectory() as directory:
            result = self._preflight(directory, PhysicalAdapterError("qualification software identity differs"))
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("qualification software identity differs", result["error"])
        self.assertEqual(result["host_software_mismatches"]["host_dependency_sha256:llama"],
                         {"expected": "old-lib", "observed": "actual"})
        self.assertFalse(result["physical_inference_started"])
        self.assertEqual(result["gates"], {})

    def test_success_is_only_a_preflight_pass(self):
        with TemporaryDirectory() as directory:
            result = self._preflight(directory)
        self.assertEqual(result["status"], "PREFLIGHT_PASS")
        self.assertFalse(result["physical_inference_started"])
        self.assertEqual(result["gates"], {})


if __name__ == "__main__":
    unittest.main()
