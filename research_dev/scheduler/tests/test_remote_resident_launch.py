"""Physical launch of a reduced desktop parent: command validation, server environment, proof."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import ModelManifest  # noqa: E402
from research_dev.scheduler.adapters import (  # noqa: E402
    ManagedLlamaServer,
    PhysicalAdapterError,
    PhysicalExecutionCommand,
    PhysicalParticipantCommand,
    llama_server_launch_contract,
)
from research_dev.scheduler.adapters.ticket import (  # noqa: E402
    validate_physical_execution_command,
)
from research_dev.scheduler._internal.runtime_plan import (  # noqa: E402
    RuntimeExecutionContract,
    RuntimeRemoteResidentFfn,
    RuntimeRemoteResidentSession,
    remote_resident_tensor_ids,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ModelManifest.from_json(json.loads(
    (ROOT / "campaigns/burstgpt/data/GEMMA_MANIFEST.json").read_text(encoding="ascii")
))
MASK_0_7 = 0xFF
PLAN_SHA256 = "sha256:" + "2" * 64
PLACEMENT_SHA256 = "sha256:" + "3" * 64


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def omitted_bytes() -> int:
    tensor_by_id = MANIFEST.tensor_by_id
    return sum(tensor_by_id[name].nbytes for name in remote_resident_tensor_ids(MASK_0_7))


def group(generation: int = 5) -> RuntimeRemoteResidentFfn:
    return RuntimeRemoteResidentFfn(
        parent_artifact_sha256=MANIFEST.artifact_sha256,
        layer_mask=MASK_0_7,
        dtype="f16",
        omitted_bytes=omitted_bytes(),
        tensor_ids=remote_resident_tensor_ids(MASK_0_7),
        shard_index_sha256=_sha("index"),
        sessions=(RuntimeRemoteResidentSession(
            session_id="HTP0",
            endpoint="session://op15/HTP0",
            layer_mask=MASK_0_7,
            shard_sha256=_sha("shard"),
            resident_geometry_sha256=_sha("geometry"),
            resident_bytes=omitted_bytes(),
            remote_path="/data/local/tmp/gemma24/HTP0.ffn.gguf",
            session_generation=generation,
            operator_plan_sha256=_sha("resident-operator-plan"),
        ),),
    )


def parameters(**overrides) -> dict:
    values = {
        "batch_size": 2048,
        "context_size": 2560,
        "cpu_device_id": "desktop-cpu",
        "ffn_activation": "geglu",
        "ffn_assistance_phase": "decode",
        "ffn_max_tokens": 512,
        "ffn_n_embd": MANIFEST.embedding_length,
        "ffn_resident_columns": MANIFEST.feed_forward_length,
        "ffn_resident_layer_mask": (1 << 24) - 1,
        "ffn_runtime_control_protocol": "decode-boundary-v1",
        "ffn_timeout_ms": 5000,
        "ffn_transport": "functionfs-usb",
        "gpu_device_id": "desktop-cuda",
        "gpu_layers": 23,
        "model_alias": "gemma-4-12b-it-q4_0",
        "parallel": 1,
        "phone_device_id": "op15-phone",
        "ubatch_size": 512,
        "usb_allocator": "devmem",
        "usb_batch_plan": "split-row",
        "usb_full_duplex": 1,
        "usb_max_payload_bytes": 4_194_304,
        "usb_product_id": 0x5678,
        "usb_queue_depth": 2,
        "usb_slot_safety_bytes": 65536,
        "usb_split_h2d": 0,
        "usb_transport_generation": "functionfs-v9",
        "usb_transport_profile_id": "op15-functionfs-profile-v1",
        "usb_vendor_id": 0x1234,
        "usbfs_available_bytes": 16_777_216,
    }
    values.update(overrides)
    return values


def desktop_operator_rows(gpu_layers: int) -> list[dict]:
    first_gpu_layer = MANIFEST.block_count - gpu_layers
    rows = []
    for operator in sorted(MANIFEST.operators, key=lambda row: row.operator_id):
        layer = int(operator.layer_id.split(":")[1]) if operator.layer_id.startswith("layer:") else None
        device = "desktop-cuda" if layer is not None and layer >= first_gpu_layer else "desktop-cpu"
        rows.append({
            "device_ids": [device],
            "operator_id": operator.operator_id,
            "operator_kind": operator.kind,
            "split_axis": "none",
            "split_fraction_ppm": 0,
        })
    return rows


def command(contract: RuntimeExecutionContract, params: dict, *, layout_generation=1) -> PhysicalExecutionCommand:
    operator_plan = {
        "desktop_placement_sha256": PLACEMENT_SHA256,
        "execution_contract": contract.to_json(),
        "operators": desktop_operator_rows(23),
        "plan_sha256": PLAN_SHA256,
        "route_id": "auto:coordinated:physical:hot:desktop-remote:residency:hot",
    }
    return PhysicalExecutionCommand(
        ticket_id="request-1:attempt:0",
        request_id="request-1",
        model_id=MANIFEST.model_id,
        artifact_sha256=MANIFEST.artifact_sha256,
        route_id=operator_plan["route_id"],
        executor_id="physical:hot:desktop-remote",
        endpoint="http://127.0.0.1:18573",
        operator_plan_protocol="llama-server-http-v1",
        operator_plan_sha256=PLAN_SHA256,
        planned_start_us=10,
        planned_finish_us=20,
        planned_finish_upper_us=30,
        operator_plan=operator_plan,
        participants=(PhysicalParticipantCommand(
            executor_id="physical:hot:desktop-remote",
            device_id="desktop-cpu",
            endpoint="http://127.0.0.1:18573",
            backend="llama-server-cuda-cpu",
            resource_ids=("desktop-cpu",),
        ),),
        leases=(),
        memory_reservations=(),
        transitions=(),
        execution_contract=contract,
        adapter_parameters=params,
        phone_layout_generation=layout_generation,
    )


class RemoteResidentCommandValidationTests(unittest.TestCase):
    def test_bound_remote_parent_is_valid(self) -> None:
        validate_physical_execution_command(
            command(RuntimeExecutionContract.desktop(group()), parameters())
        )

    def test_unbound_generation_missing_layout_or_mask_fail_closed(self) -> None:
        with self.assertRaisesRegex(PhysicalAdapterError, "lack physical generations"):
            validate_physical_execution_command(
                command(RuntimeExecutionContract.desktop(group(generation=0)), parameters())
            )
        with self.assertRaisesRegex(PhysicalAdapterError, "phone layout generation"):
            validate_physical_execution_command(
                command(RuntimeExecutionContract.desktop(group()), parameters(), layout_generation=None)
            )
        with self.assertRaisesRegex(PhysicalAdapterError, "exceed the phone resident layer mask"):
            validate_physical_execution_command(
                command(RuntimeExecutionContract.desktop(group()), parameters(ffn_resident_layer_mask=0x0F))
            )
        plain = parameters()
        plain.pop("phone_device_id")
        with self.assertRaisesRegex(PhysicalAdapterError, "phone runtime parameter"):
            validate_physical_execution_command(
                command(RuntimeExecutionContract.desktop(group()), plain)
            )

    def test_plain_desktop_command_still_rejects_a_phone_endpoint(self) -> None:
        with self.assertRaisesRegex(PhysicalAdapterError, "carries a phone endpoint"):
            validate_physical_execution_command(
                command(RuntimeExecutionContract.desktop(), parameters(), layout_generation=None)
            )


class RemoteResidentLaunchContractTests(unittest.TestCase):
    def test_environment_pins_remote_layers_and_whole_ubatch(self) -> None:
        contract = llama_server_launch_contract(
            command(RuntimeExecutionContract.desktop(group()), parameters()), MANIFEST
        )
        env = contract.ffn_environment
        self.assertEqual(env["S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK"], str(MASK_0_7))
        self.assertEqual(env["S41_SERVER_FFN_LAYER_MASK"], str((1 << 24) - 1))
        self.assertEqual(env["S41_SERVER_FFN_COLUMNS"], str(MANIFEST.feed_forward_length))
        self.assertEqual(env["S41_SERVER_FFN_MAX_TOKENS"], "512")
        self.assertEqual(env["S41_SERVER_FFN_RUNTIME_CONTROL"], "1")
        self.assertEqual(env["S41_SERVER_FFN_SHARDS"], f"HTP0@session://op15/HTP0:{MASK_0_7}")
        self.assertEqual(env["S41_SERVER_FFN_TRANSPORT"], "functionfs-usb")
        self.assertEqual(env["S41_SERVER_FFN_ARTIFACT_SHA256"], MANIFEST.artifact_sha256)
        self.assertEqual(contract.phone_device_id, "op15-phone")
        self.assertEqual(contract.gpu_layers, 23)
        self.assertEqual(contract.ubatch_size, 512)

    def test_shape_mismatches_fail_closed(self) -> None:
        for message, overrides in (
            ("shape differs", {"ffn_max_tokens": 64}),
            ("shape differs", {"ffn_resident_columns": 8192}),
            ("shape differs", {"ffn_resident_layer_mask": 0x0F}),
            ("decode-boundary", {"ffn_assistance_phase": "all"}),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(PhysicalAdapterError, message):
                    llama_server_launch_contract(
                        command(RuntimeExecutionContract.desktop(group()), parameters(**overrides)),
                        MANIFEST,
                    )
        with self.assertRaisesRegex(PhysicalAdapterError, "bound session generations"):
            llama_server_launch_contract(
                command(RuntimeExecutionContract.desktop(group(generation=0)), parameters()), MANIFEST
            )

    def test_reduced_and_full_parents_never_share_a_server(self) -> None:
        from research_dev.scheduler.adapters.llama_server import (
            _launch_contract_supports_execution,
        )
        reduced = llama_server_launch_contract(
            command(RuntimeExecutionContract.desktop(group()), parameters()), MANIFEST
        )
        full_parameters = parameters()
        for name in list(full_parameters):
            if name.startswith(("ffn_", "usb", "phone_")):
                full_parameters.pop(name)
        full = llama_server_launch_contract(
            command(RuntimeExecutionContract.desktop(), full_parameters, layout_generation=None), MANIFEST
        )
        self.assertFalse(_launch_contract_supports_execution(reduced, full))
        self.assertFalse(_launch_contract_supports_execution(full, reduced))
        self.assertTrue(_launch_contract_supports_execution(reduced, reduced))


class RemoteResidentProofTests(unittest.TestCase):
    def _remote_server(self, root: Path):
        execution = command(RuntimeExecutionContract.desktop(group()), parameters())
        server = ManagedLlamaServer(
            ("server",), {}, root, "reduced", llama_server_launch_contract(execution, MANIFEST)
        )
        server.process = SimpleNamespace(poll=lambda: None)
        return execution, server, server.begin_execution(execution, MANIFEST)

    def _append_calls(self, server, *, columns=None, skipped_layer=None):
        columns = MANIFEST.feed_forward_length if columns is None else columns
        for token in range(2):
            for layer in range(8):
                if layer != skipped_layer:
                    server.stderr_lines.append(
                        f"S41SERVERFFNCALL request={1 + token * 8 + layer} layer={layer} "
                        f"tokens=1 columns={columns} payload_bytes={MANIFEST.embedding_length * 2}"
                    )

    def test_remote_parent_requires_phone_execution_proof(self) -> None:
        execution = command(RuntimeExecutionContract.desktop(group()), parameters())
        contract = llama_server_launch_contract(execution, MANIFEST)
        with tempfile.TemporaryDirectory() as directory:
            server = ManagedLlamaServer(("server",), {}, Path(directory), "reduced", contract)
            server.process = SimpleNamespace(poll=lambda: None)
            marker = server.begin_execution(execution, MANIFEST)
            self.assertIsNotNone(marker.phone_contract)
            self.assertEqual(marker.phone_contract.layer_mask, MASK_0_7)
            with self.assertRaisesRegex(PhysicalAdapterError, "phone calls differ"):
                server.finish_execution(marker, execution, MANIFEST, output_tokens=2)

    def test_remote_proof_binds_loaded_session_not_desktop_operator_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            execution, server, marker = self._remote_server(Path(directory))
            self._append_calls(server)
            proof = server.finish_execution(marker, execution, MANIFEST, output_tokens=2)
            self.assertEqual(proof.phone_call_count, 16)
            self.assertEqual(proof.operator_plan_sha256, PLAN_SHA256)
            shard, = proof.phone_calls_by_session
            owner, = execution.execution_contract.remote_resident_ffn.sessions
            self.assertEqual(shard.operator_plan_sha256, owner.operator_plan_sha256)
            self.assertNotEqual(shard.operator_plan_sha256, proof.operator_plan_sha256)
            self.assertEqual(shard.session_generation, 5)
            self.assertEqual(shard.artifact_sha256, MANIFEST.artifact_sha256)
            self.assertEqual(shard.endpoint, owner.endpoint)
            self.assertEqual(shard.resident_geometry_sha256, owner.resident_geometry_sha256)
            self.assertEqual(shard.layer_mask, MASK_0_7)
            self.assertEqual(shard.rows, 16)

    def test_remote_proof_rejects_partial_columns_and_missing_layer(self) -> None:
        for options, message in (
            ({"columns": MANIFEST.feed_forward_length // 2}, "phone calls differ"),
            ({"skipped_layer": 3}, "coverage is incomplete"),
        ):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                execution, server, marker = self._remote_server(Path(directory))
                self._append_calls(server, **options)
                with self.assertRaisesRegex(PhysicalAdapterError, message):
                    server.finish_execution(marker, execution, MANIFEST, output_tokens=2)

    def test_remote_owner_operator_binding_is_required(self) -> None:
        remote = group()
        remote = replace(remote, sessions=(replace(remote.sessions[0], operator_plan_sha256=None),))
        with self.assertRaisesRegex(PhysicalAdapterError, "lack physical operator plans"):
            validate_physical_execution_command(command(RuntimeExecutionContract.desktop(remote), parameters()))

    def test_remote_proof_rejects_rebinding_after_execution_begins(self) -> None:
        for values in (
            {"session_generation": 6},
            {"operator_plan_sha256": _sha("another-loaded-plan")},
            {"shard_sha256": _sha("another-shard")},
        ):
            with self.subTest(values=values), tempfile.TemporaryDirectory() as directory:
                execution, server, marker = self._remote_server(Path(directory))
                self._append_calls(server)
                remote = execution.execution_contract.remote_resident_ffn
                changed = replace(remote, sessions=(replace(remote.sessions[0], **values),))
                changed_command = replace(execution, execution_contract=RuntimeExecutionContract.desktop(changed))
                with self.assertRaisesRegex(PhysicalAdapterError, "owner identity changed"):
                    server.finish_execution(marker, changed_command, MANIFEST, output_tokens=2)

    def test_managed_server_exposes_a_single_proof(self) -> None:
        contract = llama_server_launch_contract(
            command(RuntimeExecutionContract.desktop(group()), parameters()), MANIFEST
        )
        with tempfile.TemporaryDirectory() as directory:
            server = ManagedLlamaServer(("server",), {}, Path(directory), "reduced", contract)
            server.process = SimpleNamespace(poll=lambda: None)
            self.assertIsNone(server.remote_resident_proof())
            server.stderr_lines.extend([
                "S41SERVERFFN ready host=usb port=0 columns=15360",
                f"S41SERVERFFN remote_resident mask={MASK_0_7} omitted_bytes={omitted_bytes()} "
                f"unmapped_bytes={omitted_bytes() - 8192} warmup=validated",
            ])
            proof = server.remote_resident_proof()
            self.assertEqual(proof.layer_mask, MASK_0_7)
            self.assertEqual(proof.omitted_bytes, omitted_bytes())
            self.assertEqual(proof.warmup, "validated")
            server.stderr_lines.append(server.stderr_lines[-1])
            with self.assertRaisesRegex(PhysicalAdapterError, "several"):
                server.remote_resident_proof()


if __name__ == "__main__":
    unittest.main()
