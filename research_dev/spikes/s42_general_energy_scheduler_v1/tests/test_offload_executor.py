#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[4]
S42_ROOT = REPO_ROOT / "research_dev/spikes/s42_general_energy_scheduler_v1"
PLANNER_PATH = S42_ROOT / "physical_ab_v1/plan_unified_burstgpt.py"
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    BackendArtifact,
    CohortPlanner,
    ExecutionPlanError,
    OperatorOffloadContract,
    OperatorSplitError,
    OperatorSplitPolicy,
    PhoneBackendConfig,
    PhoneTransportContract,
    QualityClass,
    load_certified_cohort,
    load_execution_plan,
    write_execution_plan,
)


def i3_split() -> OperatorSplitPolicy:
    return OperatorSplitPolicy.from_table(
        policy_id="i3-hidden-wait",
        operator_family="gemma4-dense-ffn-swiglu",
        layer_ids=range(48),
        n_embd=3840,
        eligible_columns=11136,
        max_tokens=512,
        column_quantum=2048,
        alternate_columns=(9664,),
        io_type="f16",
        weight_layout="view_safe_dense_ffn",
        table="1:9664,3:8192,8:4096,128:8192,512:11136",
    )


def artifacts() -> tuple[BackendArtifact, ...]:
    roles = (
        "bridge",
        "cold_model",
        "cold_server",
        "hot_model",
        "hot_server",
        "phone_model",
        "phone_session",
        "phone_worker",
        "restore_usb",
    )
    return tuple(
        BackendArtifact(role, f"/tmp/{role}", "a" * 64) for role in roles
    )


class OperatorSplitTests(unittest.TestCase):
    def test_i3_shape_table_and_work_accounting(self) -> None:
        split = i3_split()
        self.assertEqual(
            [split.columns_for(tokens) for tokens in (1, 2, 4, 9, 512)],
            [9664, 8192, 4096, 8192, 11136],
        )
        invocation = split.invocation(47, 512)
        self.assertEqual(invocation.activation_bytes, 3_932_160)
        self.assertEqual(
            invocation.phone_macs,
            3 * 3840 * 11136 * 512,
        )
        summary = split.summarize(((1, 2), (8, 3)))
        self.assertEqual(summary.calls, 5)
        self.assertEqual(summary.token_rows, 26)
        self.assertEqual(summary.upload_bytes, 26 * 3840 * 2)

    def test_unqualified_width_is_rejected(self) -> None:
        with self.assertRaisesRegex(OperatorSplitError, "column quantum"):
            OperatorSplitPolicy.from_table(
                policy_id="bad",
                operator_family="dense-ffn",
                layer_ids=(0,),
                n_embd=128,
                eligible_columns=1024,
                max_tokens=8,
                column_quantum=512,
                alternate_columns=(),
                io_type="f16",
                weight_layout="view-safe",
                table="8:700",
            )


class PhoneOffloadTests(unittest.TestCase):
    def contract(self) -> OperatorOffloadContract:
        split = i3_split()
        transport = PhoneTransportContract(
            transport_id="op15-dmabuf",
            protocol="s41-ffn-split-flex-v2",
            host_endpoint="libusb-bulk",
            phone_endpoint="functionfs-dmabuf",
            allocator="devmem",
            io_type="f16",
            payload_offset_bytes=128,
            max_payload_bytes=3_932_160,
            queue_depth=1,
            usb_speed_mbps=5000,
            usb_vendor_product="18d1:2d00",
            phone_resource_id="op15-htp",
            transport_resource_id="op15-usb",
            usb_root_resource_id="usb-root-0",
            bridge_residency_id="bridge:" + "a" * 64,
            worker_residency_id="worker:" + "a" * 64,
            reset_generation=0,
            max_reset_recoveries=0,
        )
        backend = PhoneBackendConfig(
            backend_id="op15",
            phone_serial="phone",
            adb_port=5037,
            compute_backend="HTP0",
            layer_spec="0-47",
            bridge_bind="127.0.0.1",
            bridge_port=25660,
            server_timeout_ms=35000,
            session_timeout_s=1800,
            max_requests=120000,
            artifacts=tuple(
                item for item in artifacts()
                if item.role in {
                    "bridge",
                    "cold_model",
                    "phone_model",
                    "phone_session",
                    "phone_worker",
                    "restore_usb",
                }
            ),
        )
        return OperatorOffloadContract(
            route_id="offload",
            host_resource_id="cpu-cold",
            split=split,
            transport=transport,
            backend=backend,
        )

    def test_contract_owns_server_phone_and_transport_configuration(self) -> None:
        contract = self.contract()
        invocation = contract.invocation(0, 512)
        self.assertEqual(invocation.transfer.payload_bytes, 3_932_160)
        self.assertEqual(invocation.transfer.upload_wire_bytes, 3_932_288)
        self.assertEqual(
            invocation.required_resource_ids,
            ("cpu-cold", "op15-htp", "op15-usb", "usb-root-0"),
        )
        self.assertEqual(
            contract.server_environment()["S41_SERVER_FFN_POLICY"],
            contract.split.table,
        )
        self.assertEqual(
            contract.server_environment()["S41_SERVER_FFN_LAYER_MASK"],
            "0x0000ffffffffffff",
        )
        self.assertEqual(
            contract.server_environment()["S41_SERVER_FFN_ACTIVATION"],
            "geglu",
        )
        swiglu = replace(
            contract,
            split=replace(contract.split, activation="swiglu"),
        )
        self.assertEqual(
            swiglu.server_environment()["S41_SERVER_FFN_ACTIVATION"],
            "swiglu",
        )
        self.assertEqual(contract.bridge_command()[-1], "devmem")
        self.assertEqual(
            contract.phone_environment()["S41_FFN_STAGED_DMABUF"], "0"
        )
        staged = replace(
            contract,
            transport=replace(contract.transport, allocator="malloc-split"),
        )
        self.assertEqual(
            staged.phone_environment()["S41_FFN_STAGED_DMABUF"], "1"
        )


class CertifiedCohortTests(unittest.TestCase):
    def profile(self):
        trace = (
            S42_ROOT.parent
            / "s41_gemma_qwen_continuous_baseline/tp_operator_split_v1"
            / "burstgpt_gpu_cpu_op15_v1/REQUESTS_SEMANTIC_SOURCE.jsonl"
        )
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        route_root = S42_ROOT / "runtime_routes_v1"
        return load_certified_cohort(
            compiled_path=route_root / "COMPILED_4060TI_OP15_I3_ROUTES_V1.json",
            epoch_path=route_root / "I3_CERTIFIED_EPOCH_4060TI_OP15_V1.json",
            runtime_contracts_path=(
                route_root / "I3_RUNTIME_GATE_CONTRACTS_4060TI_OP15_V1.json"
            ),
            workload_id="burstgpt-source-two-model-v1",
            unit_id="burstgpt-source-74-v1",
            member_request_ids=[row["event_id"] for row in rows],
            quality_requirement=QualityClass.SEMANTIC,
            boundary_id="fleet-v3",
        )

    def test_energy_policy_selects_exact_cohort_offload(self) -> None:
        profile = self.profile()
        control = CohortPlanner().schedule(profile.candidates, "control")
        enforce = CohortPlanner().schedule(profile.candidates, "enforce")
        self.assertEqual(control.route_id, "i3-cold-cpu-control-v1")
        self.assertEqual(enforce.route_id, "i3-cold-cpu-op15-ffn-v1")
        self.assertEqual(enforce.reason, "VERIFIED_COHORT_ENERGY_SAVING")
        self.assertLess(enforce.energy_upper_uj, control.energy_mean_uj)


class ExecutionPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location(
            "plan_unified_burstgpt", PLANNER_PATH
        )
        assert spec is not None and spec.loader is not None
        cls.planner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.planner)

    def test_burstgpt_plan_round_trip_and_tamper_rejection(self) -> None:
        trace = (
            S42_ROOT.parent
            / "s41_gemma_qwen_continuous_baseline/tp_operator_split_v1"
            / "burstgpt_gpu_cpu_op15_v1/REQUESTS_SEMANTIC_SOURCE.jsonl"
        )
        route_root = S42_ROOT / "runtime_routes_v1"
        rows = artifacts()
        plan = self.planner.build_plan(
            trace_path=trace,
            compiled_path=route_root / "COMPILED_4060TI_OP15_I3_ROUTES_V1.json",
            epoch_path=route_root / "I3_CERTIFIED_EPOCH_4060TI_OP15_V1.json",
            contracts_path=(
                route_root / "I3_RUNTIME_GATE_CONTRACTS_4060TI_OP15_V1.json"
            ),
            mode="enforce",
            artifact_paths={item.role: item.path for item in rows},
            artifact_hashes={item.role: item.sha256 for item in rows},
            cuda_lib_dir="/tmp/cuda",
            cold_lib_dir="/tmp/cold",
            adb_port=5037,
            phone_serial="phone",
        )
        self.assertEqual(plan.execution_mode, "op15")
        self.assertIsNotNone(plan.offload)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "plan.json"
            write_execution_plan(path, plan)
            self.assertEqual(load_execution_plan(path), plan)
            value = json.loads(path.read_text())
            value["runtime_bindings"]["hot_parallel"] = 99
            path.write_text(json.dumps(value), encoding="ascii")
            with self.assertRaisesRegex(ExecutionPlanError, "hash mismatch"):
                load_execution_plan(path)


if __name__ == "__main__":
    unittest.main()
