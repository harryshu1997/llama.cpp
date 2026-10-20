"""Remote-resident owner arm with a static decode co-helper (Phase 2, target B; opt-in).

The OnePlus owns the remote-resident layers (prefill and decode, whole ubatches); a Pixel serves
decode of its own desktop-resident layers under its own row cap (``HELPER<k>_MAX_TOKENS``), driven
by one static policy that never names a remote-resident layer. The proof accepts both phones'
calls, and a lost helper is never masked out of a remote parent's server.
"""
from __future__ import annotations

from dataclasses import replace
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler.adapters import (  # noqa: E402
    CatalogMaterializationError,
    PhysicalAdapterError,
    llama_server_launch_contract,
    materialize_remote_resident_owner,
)
from research_dev.scheduler.adapters.llama_server import ManagedLlamaServer  # noqa: E402
from research_dev.scheduler.adapters.phone_helpers import (  # noqa: E402
    PhoneHelperBinding,
    helper_server_environment,
    helper_transport_contracts,
)
from research_dev.scheduler.adapters.ticket import (  # noqa: E402
    remote_resident_co_helper_policy,
    static_decode_policy,
    validate_physical_execution_command,
)
from research_dev.scheduler.config import (  # noqa: E402
    CAMPAIGN_MANIFEST_SCHEMA,
    CampaignManifest,
    RemoteResidentOwnerConfiguration,
    SchedulerConfigurationError,
)
from research_dev.scheduler._internal.plan_contracts.co_helpers import (  # noqa: E402
    PHONE_CO_HELPERS_PARAMETER,
    PHONE_HELPERS_PARAMETER,
    phone_helper_rows,
    phone_helpers_support,
)
from research_dev.scheduler._internal.plan_contracts.common import RuntimePlanError  # noqa: E402
from research_dev.scheduler._internal.route_generation.common import RouteGenerationError  # noqa: E402
from research_dev.scheduler._internal.route_generation.remote_resident import (  # noqa: E402
    REMOTE_RESIDENT_CO_HELPER_DECODE_PARAMETER,
    REMOTE_RESIDENT_FFN_PARAMETER,
    parse_remote_resident_declaration,
    remote_resident_co_helper_parameters,
)

try:
    from . import test_remote_resident_co_helpers as co_fixture
    from . import test_remote_resident_launch as launch_fixture
    from . import test_remote_resident_materialization as owner
    from . import two_phone_harness as two_phone
except ImportError:
    import test_remote_resident_co_helpers as co_fixture
    import test_remote_resident_launch as launch_fixture
    import test_remote_resident_materialization as owner
    import two_phone_harness as two_phone

OWNER_MASK = co_fixture.MASK_0_7
PIXEL_MASK = co_fixture.PIXEL_MASK_8_15
DECODE = {REMOTE_RESIDENT_CO_HELPER_DECODE_PARAMETER: 1}


def pixel(max_tokens: int = 4):
    return co_fixture.co_helpers(max_tokens=max_tokens)


class BindingTests(unittest.TestCase):
    def test_a_capped_co_helper_row_carries_its_own_batch(self) -> None:
        declaration = pixel()
        plain = phone_helper_rows(declaration.phone_helpers("op15-phone", OWNER_MASK))
        self.assertNotIn("max_tokens", plain[1])
        capped = phone_helper_rows(declaration.phone_helpers("op15-phone", OWNER_MASK, maximum_tokens=512))
        self.assertEqual((capped[0].get("max_tokens"), capped[1]["max_tokens"]), (None, 4))
        same = phone_helper_rows(pixel(512).phone_helpers("op15-phone", OWNER_MASK, maximum_tokens=512))
        self.assertNotIn("max_tokens", same[1])
        with self.assertRaises(RuntimePlanError):
            declaration.phone_helpers("op15-phone", OWNER_MASK, maximum_tokens=2)
        live = declaration.phone_helpers("op15-phone", OWNER_MASK, maximum_tokens=512)
        self.assertTrue(phone_helpers_support(live, live))
        self.assertFalse(phone_helpers_support(declaration.phone_helpers("op15-phone", OWNER_MASK), live))
        for row in ({"max_tokens": 0}, {"max_tokens": "4"}):
            rows = json.loads(live)
            rows[1].update(row)
            with self.subTest(row=row), self.assertRaises(RuntimePlanError):
                phone_helper_rows(json.dumps(rows))
        rows = json.loads(live)
        rows[0]["max_tokens"] = 4
        with self.assertRaises(RuntimePlanError):
            phone_helper_rows(json.dumps(rows))

    def test_the_server_environment_caps_only_that_helper(self) -> None:
        owner_row = PhoneHelperBinding(device_id="op15-phone", serial="OP15", layer_mask=OWNER_MASK, label="op15")
        pixel_row = PhoneHelperBinding(device_id="pixel-phone", serial="PIXEL10", layer_mask=PIXEL_MASK,
                                       label="pixel", transport_parameters=co_fixture.PIXEL_TRANSPORT,
                                       max_tokens=4)
        self.assertEqual(PhoneHelperBinding.from_json(pixel_row.to_json()), pixel_row)
        self.assertNotIn("max_tokens", owner_row.to_json())
        shared = {"S41_SERVER_FFN_LAYER_MASK": str(OWNER_MASK | PIXEL_MASK), "S41_SERVER_FFN_MAX_TOKENS": "512",
                  "S41_SERVER_FFN_RUNTIME_CONTROL": "1",
                  "S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK": str(OWNER_MASK)}
        rows = (owner_row, pixel_row)
        contracts = helper_transport_contracts(rows, launch_fixture.parameters())
        environment = helper_server_environment(rows, shared, contracts)
        self.assertEqual(environment["S41_SERVER_FFN_HELPER1_MAX_TOKENS"], "4")
        self.assertNotIn("S41_SERVER_FFN_HELPER0_MAX_TOKENS", environment)
        self.assertEqual(environment["S41_SERVER_FFN_MAX_TOKENS"], "512")
        for changes in ({"S41_SERVER_FFN_RUNTIME_CONTROL": "0"}, {"S41_SERVER_FFN_MAX_TOKENS": "4"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(PhysicalAdapterError, "capped phone helper"):
                helper_server_environment(rows, {**shared, **changes}, contracts)
        with self.assertRaisesRegex(PhysicalAdapterError, "capped phone helper"):
            helper_server_environment((replace(pixel_row, transport_parameters={}),),
                                      {**shared, "S41_SERVER_FFN_LAYER_MASK": str(PIXEL_MASK)},
                                      contracts[:1])


class RouteGenerationTests(unittest.TestCase):
    def test_decode_co_helpers_may_cap_their_rows_only_with_the_opt_in(self) -> None:
        manifest = co_fixture.quantized_manifest("Q8_0")
        with self.assertRaisesRegex(RouteGenerationError, "batch capacity"):
            co_fixture.compiler(manifest, "q8_0", helpers=pixel())._remote_resident_group(
                manifest, co_fixture.route_fixture.pattern())
        route = co_fixture.compiler(manifest, "q8_0", helpers=pixel(), **DECODE)
        self.assertEqual(route._remote_resident_group(manifest, co_fixture.route_fixture.pattern()).dtype, "q8_0")
        coordinator = next(iter(route.catalog.composite_executor_by_id.values()))
        parameters = dict(coordinator.adapter_parameters)
        remote_resident_co_helper_parameters(parameters)
        rows = phone_helper_rows(parameters[PHONE_HELPERS_PARAMETER])
        self.assertEqual([(row["device_id"], row["layer_mask"], row.get("max_tokens")) for row in rows],
                         [("op15-phone", OWNER_MASK, None), ("pixel-phone", PIXEL_MASK, 4)])
        self.assertEqual(parameters["ffn_resident_layer_mask"], OWNER_MASK | PIXEL_MASK)
        with self.assertRaisesRegex(RouteGenerationError, "batch capacity"):
            co_fixture.compiler(manifest, "q8_0", helpers=pixel(1024), **DECODE)._remote_resident_group(
                manifest, co_fixture.route_fixture.pattern())


def decode_command(*, decode: bool = True, max_tokens: int = 4, operator_plan_resources: bool = True):
    """A bound remote parent (owner HTP0: layers 0-7) with the Pixel decode co-helper on 8-15."""
    declaration = pixel(max_tokens)
    command = co_fixture.two_phone_command(helpers=declaration.phone_helpers(
        "op15-phone", OWNER_MASK, **({"maximum_tokens": 512} if decode else {})))
    parameters = {**command.adapter_parameters, PHONE_CO_HELPERS_PARAMETER: declaration.encode(),
                  **(DECODE if decode else {})}
    plan = dict(command.operator_plan)
    if operator_plan_resources:
        plan["resource_ids"] = ["desktop-cpu", "op15-htp", "pixel-adb"]
    return replace(command, adapter_parameters=parameters, operator_plan=plan)


class StaticPolicyAndLaunchTests(unittest.TestCase):
    def test_static_policy_names_the_co_helper_layers_only(self) -> None:
        command = decode_command()
        validate_physical_execution_command(command)
        policy = remote_resident_co_helper_policy(command)
        self.assertIs(type(static_decode_policy(command)), type(policy))
        self.assertEqual(static_decode_policy(command).policy_hash, policy.policy_hash)
        self.assertEqual((policy.layer_mask, policy.columns, policy.split_fraction_ppm),
                         (PIXEL_MASK, launch_fixture.MANIFEST.feed_forward_length, 1_000_000))
        self.assertFalse(policy.layer_mask & OWNER_MASK)
        self.assertEqual(policy.device_layer_masks, (("pixel-phone", PIXEL_MASK),))
        self.assertIsNone(remote_resident_co_helper_policy(decode_command(decode=False, max_tokens=512)))
        self.assertIsNone(static_decode_policy(decode_command(decode=False, max_tokens=512)))
        with self.assertRaisesRegex(PhysicalAdapterError, "policy is absent"):
            remote_resident_co_helper_policy(decode_command(operator_plan_resources=False))
        environment = llama_server_launch_contract(command, launch_fixture.MANIFEST).ffn_environment
        self.assertEqual(environment["S41_SERVER_FFN_HELPER1_MAX_TOKENS"], "4")
        self.assertEqual(environment["S41_SERVER_FFN_HELPER0_REMOTE_RESIDENT_LAYER_MASK"], str(OWNER_MASK))
        self.assertNotIn("S41_SERVER_FFN_HELPER0_MAX_TOKENS", environment)
        self.assertEqual(environment["S41_SERVER_FFN_MAX_TOKENS"], "512")


class ProofTests(unittest.TestCase):
    N_FF = launch_fixture.MANIFEST.feed_forward_length
    ROW = launch_fixture.MANIFEST.embedding_length * 2

    def run_proof(self, command, lines, *, applied_token=1, output_tokens=3):
        contract = llama_server_launch_contract(command, launch_fixture.MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            server = ManagedLlamaServer(("server",), {}, Path(directory), "reduced", contract)
            server.process = SimpleNamespace(poll=lambda: None)
            marker = server.begin_execution(command, launch_fixture.MANIFEST)
            server.stderr_lines.extend(lines)
            policy = static_decode_policy(command)
            ack = None if policy is None else {
                "applied_token_index": applied_token, "plan_generation": 1, "policy_hash": policy.policy_hash,
                "request_id": command.request_id, "slot_id": 0}
            return server.finish_execution(marker, command, launch_fixture.MANIFEST,
                                           output_tokens=output_tokens, static_control_ack=ack)

    def call(self, request_id, layer, tokens):
        return (f"S41SERVERFFNCALL request={request_id} layer={layer} tokens={tokens} "
                f"columns={self.N_FF} payload_bytes={self.ROW * tokens}")

    def lines(self, *, pixel_tokens=1, pixel_steps=2, pixel_layers=range(8, 16)):
        owner = [self.call(1 + step * 8 + layer, layer, 1) for step in range(3) for layer in range(8)]
        helper = [self.call((1 << 24) + 1 + step * 8 + index, layer, pixel_tokens)
                  for step in range(pixel_steps) for index, layer in enumerate(pixel_layers)]
        return owner + helper

    def test_owner_and_decode_co_helper_calls_are_both_proven(self) -> None:
        proof = self.run_proof(decode_command(), self.lines())
        self.assertEqual(proof.phone_call_count, 24 + 16)
        sessions = {row.session_id: row for row in proof.phone_calls_by_session}
        self.assertEqual(set(sessions), {"HTP0", "PIXEL0"})
        self.assertEqual((sessions["HTP0"].layer_mask, sessions["PIXEL0"].layer_mask), (OWNER_MASK, PIXEL_MASK))
        self.assertEqual(sessions["PIXEL0"].rows, 16)
        self.assertEqual(proof.static_policy_applied_token_index, 1)

    def test_calls_outside_the_policy_or_the_cap_fail_closed(self) -> None:
        cases = (
            ("differ from the operator plan", self.lines(pixel_tokens=5)),
            ("differ from the operator plan", self.lines() + [self.call((1 << 24) + 99, 20, 1)]),
            ("coverage is incomplete", self.lines(pixel_steps=1)),
            ("outside its owning phone's ids", self.lines()[:-1] + [self.call(9999, 15, 1)]),
        )
        for message, lines in cases:
            with self.subTest(message=message), self.assertRaisesRegex(PhysicalAdapterError, message):
                self.run_proof(decode_command(), lines)

    def test_without_the_opt_in_co_helper_calls_stay_refused(self) -> None:
        command = decode_command(decode=False, max_tokens=512)
        owner_only = self.lines(pixel_steps=0)
        self.assertEqual(self.run_proof(command, owner_only).phone_call_count, 24)
        with self.assertRaisesRegex(PhysicalAdapterError, "differ from the operator plan"):
            self.run_proof(command, self.lines())

    def test_a_policy_applied_at_the_last_token_needs_no_co_helper_call(self) -> None:
        proof = self.run_proof(decode_command(), self.lines(pixel_steps=0), applied_token=3)
        self.assertEqual({row.session_id for row in proof.phone_calls_by_session}, {"HTP0"})

    def test_a_one_token_request_may_end_before_the_policy_applies(self) -> None:
        owner_calls = [self.call(1 + layer, layer, 1) for layer in range(8)]
        command = decode_command()
        contract = llama_server_launch_contract(command, launch_fixture.MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            server = ManagedLlamaServer(("server",), {}, Path(directory), "reduced", contract)
            server.process = SimpleNamespace(poll=lambda: None)
            marker = server.begin_execution(command, launch_fixture.MANIFEST)
            server.stderr_lines.extend(owner_calls)
            proof = server.finish_execution(marker, command, launch_fixture.MANIFEST, output_tokens=1)
            self.assertIsNone(proof.static_policy_hash)
            with self.assertRaisesRegex(PhysicalAdapterError, "acknowledgement is absent"):
                marker = server.begin_execution(command, launch_fixture.MANIFEST)
                server.stderr_lines.extend(owner_calls + owner_calls)
                server.finish_execution(marker, command, launch_fixture.MANIFEST, output_tokens=2)


class MaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        scope = two_phone.TemporaryModel()
        self.model = scope.__enter__()
        self.addCleanup(scope.__exit__)
        self.harness = owner.QuantizedOwnerTests()
        self.harness.model = self.model

    def test_the_co_helper_joins_the_parent_only_with_the_opt_in(self) -> None:
        catalog = self.harness.catalog(declaration=two_phone.co_helpers())
        index = owner.shard_index(self.model, {"HTP0": 0b1111}, weight_type="Q8_0")
        with self.assertRaisesRegex(CatalogMaterializationError, "co-helper"):
            materialize_remote_resident_owner(catalog, self.model, {"HTP0": 0b1111}, index)
        catalog = materialize_remote_resident_owner(
            catalog, self.model, {"HTP0": 0b1111}, index, co_helper_decode=True)
        remote = catalog.composite_executor_by_id["physical:two:desktop:remote-owner"]
        self.assertEqual(remote.participant_device_ids[-2:], (two_phone.OP15, two_phone.PIXEL))
        self.assertTrue({"pixel-adb", "pixel-gpu"} <= set(remote.resource_ids))
        self.assertEqual(remote.adapter_parameters[REMOTE_RESIDENT_CO_HELPER_DECODE_PARAMETER], 1)
        self.assertIn(PHONE_CO_HELPERS_PARAMETER, remote.adapter_parameters)
        group = parse_remote_resident_declaration(remote.adapter_parameters[REMOTE_RESIDENT_FFN_PARAMETER])
        self.assertFalse(group.layer_mask & two_phone.PIXEL_MASK)
        with self.assertRaisesRegex(CatalogMaterializationError, "decode co-helpers need"):
            materialize_remote_resident_owner(
                self.harness.catalog(), self.model, {"HTP0": 0b1111}, index, co_helper_decode=True)
        with self.assertRaisesRegex(CatalogMaterializationError, "decode co-helpers need"):
            materialize_remote_resident_owner(
                self.harness.catalog(declaration=two_phone.co_helpers()), self.model, {"HTP0": 0b111111},
                owner.shard_index(self.model, {"HTP0": 0b111111}, weight_type="Q8_0"), co_helper_decode=True)

    def test_the_generated_remote_route_binds_the_capped_co_helper(self) -> None:
        from research_dev.scheduler import UnifiedScheduler

        # the Pixel takes two decode rows while the owner serves the whole 4-row ubatch
        catalog = materialize_remote_resident_owner(
            self.harness.catalog(declaration=two_phone.co_helpers(max_tokens=2)), self.model, {"HTP0": 0b1111},
            owner.shard_index(self.model, {"HTP0": 0b1111}, weight_type="Q8_0"), co_helper_decode=True)
        scheduler = UnifiedScheduler.for_runtime_discovery("enforce")
        scheduler.register_runtime_capabilities(catalog)
        scheduler.register_model_manifest(self.model)
        candidates = scheduler.generate_automated_candidates(
            two_phone.request("remote-owner"), self.model.model_id, two_phone.snapshot(self.model, catalog))
        candidate, = (row for row in candidates.candidates
                      if row.binding.executor_id == "physical:two:desktop:remote-owner")
        rows = phone_helper_rows(candidate.plan.adapter_parameters[PHONE_HELPERS_PARAMETER])
        self.assertEqual([(row["device_id"], row.get("max_tokens")) for row in rows],
                         [(two_phone.OP15, None), (two_phone.PIXEL, 2)])
        self.assertIn(two_phone.PIXEL, candidate.device_ids)
        # owners are not loaded yet; the transport and the co-helper's batch are accepted
        self.assertNotIn("TRANSPORT_PROFILE_INCOMPLETE", candidate.rejection_reasons)
        self.assertTrue(any(reason.startswith("REMOTE_RESIDENT_OWNER_NOT_READY:HTP0")
                            for reason in candidate.rejection_reasons))


class ConfigurationAndMaskOutTests(unittest.TestCase):
    ROW = {"schema": CAMPAIGN_MANIFEST_SCHEMA, "campaign_id": "remote-owner-b",
           "rig_manifest_path": "rig.json", "models_manifest_path": "models.json",
           "evidence_manifest_path": "evidence.json", "selection_mode": "energy-aware",
           "maximum_latency_ppm": 1_100_000, "energy_attribution_kind": "diagnostic",
           "trace": {"large_requests_path": "large.json", "overlay_requests_path": "overlay.json",
                     "trace_manifest_path": "trace.json"},
           "fixed_phone_residency": {"mode": "fixed", "reference_id": "q32", "assignments": [
               {"session_id": "HTP0", "artifact_sha256": "sha256:" + "a" * 64, "layer_mask": 127,
                "maximum_columns": 25600}]}}

    def test_static_co_helpers_is_an_explicit_opt_in(self) -> None:
        owner_row = {"model_id": "qwen3-32b-q8_0", "static_co_helpers": True}
        configured = CampaignManifest.from_json({**self.ROW, "remote_resident_owner": owner_row}, Path("/inputs"))
        self.assertTrue(configured.remote_resident_owner.static_co_helpers)
        self.assertEqual(CampaignManifest.from_json(configured.to_json(), Path("/inputs")), configured)
        self.assertNotIn("static_co_helpers", RemoteResidentOwnerConfiguration("m").to_json())
        with self.assertRaises(SchedulerConfigurationError):
            RemoteResidentOwnerConfiguration("m", static_co_helpers=1)
        mask_out = {"drop_recovery": True, "helper_loss_recovery": "mask_out"}
        with self.assertRaisesRegex(SchedulerConfigurationError, "mask out"):
            CampaignManifest.from_json({**self.ROW, "remote_resident_owner": owner_row,
                                        "elastic_phones": mask_out}, Path("/inputs"))

    def test_the_model_declares_co_helper_shards_exactly_with_the_opt_in(self) -> None:
        from test_two_phone_helpers import CampaignConfigurationTests as Fixture, two_phone_rig_json
        from research_dev.scheduler.config import load_scheduler_configuration

        case = Fixture()
        case.setUp()
        self.addCleanup(case.tearDown)
        shards = {"pixel10pro-phone": {"index_path": "pixel/FFN_SHARDS.json", "directory": "/data/local/tmp/q32"}}
        path = case._write(two_phone_rig_json(), shards)
        models_path = case.root / "models.json"
        models = json.loads(models_path.read_text())
        models["models"][0].update(phone_ffn_shard_index_path="FFN_SHARDS.json",
                                   phone_ffn_shard_directory="/data/local/tmp/q32")
        models_path.write_text(json.dumps(models))
        campaign = json.loads(path.read_text())
        campaign["fixed_phone_residency"] = self.ROW["fixed_phone_residency"]
        for static, loads in ((True, True), (False, False)):
            campaign["remote_resident_owner"] = {"model_id": "qwen3-14b-q4km-dequant-f16",
                                                 **({"static_co_helpers": True} if static else {})}
            path.write_text(json.dumps(campaign))
            with self.subTest(static=static):
                if loads:
                    configuration = load_scheduler_configuration(path, environ={})
                    self.assertTrue(configuration.campaign.remote_resident_owner.static_co_helpers)
                else:
                    with self.assertRaisesRegex(SchedulerConfigurationError, "static_co_helpers"):
                        load_scheduler_configuration(path, environ={})

    def test_a_remote_parent_never_masks_out_a_lost_helper(self) -> None:
        import test_elastic_s2a_mask_out as s2a

        case = s2a.RigMaskOutTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.mark("t0:attempt:0")
        case.lose()
        command = SimpleNamespace(
            ticket_id="t0:attempt:0", executor_id=s2a.DESKTOP, request_id="t0",
            execution_contract=SimpleNamespace(execution_mode="desktop", remote_resident_ffn=object()))
        classification = case.rig._classify_execution_failure(command, s2a.cut(), started_ns=time.monotonic_ns())
        self.assertIsNone(classification.masked_executor_id)
        refused = [row for row in case.rig.helper_mask_events if row["kind"] == "HELPER_MASK_OUT_REFUSED"]
        self.assertEqual([row["reason"] for row in refused], ["REMOTE_RESIDENT_PARENT"])


if __name__ == "__main__":
    unittest.main()
