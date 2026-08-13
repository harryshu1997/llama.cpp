#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]

from research_dev.scheduler import (  # noqa: E402
    BackendArtifact,
    CapacityError,
    CapacityPlanner,
    CohortDecision,
    DeviceMemoryCapacity,
    ExecutionPlanError,
    LayerPlacementCandidate,
    LayerPlacementContract,
    OffloadError,
    OperatorOffloadContract,
    OperatorSplitPolicy,
    PhoneBackendConfig,
    PhoneTransportContract,
    build_execution_plan,
    load_execution_plan,
    write_execution_plan,
)


MODEL_HASH = "sha256:" + "a" * 64


def measured_split() -> LayerPlacementCandidate:
    return LayerPlacementCandidate(
        candidate_id="gemma-f16-cpu23-cuda25-v1",
        model_id="gemma-f16-proxy",
        model_sha256=MODEL_HASH,
        total_layers=48,
        cpu_prefix_layers=23,
        gpu_suffix_layers=25,
        runtime_gpu_layers=25,
        gpu_weight_bytes=12_914_765_332,
        gpu_kv_bytes=1_694_498_816,
        gpu_compute_bytes=290_455_552,
        cpu_weight_bytes=12_914_744_361,
        cpu_kv_bytes=1_694_498_816,
        cpu_compute_bytes=0,
        latency_us=779_833_719,
        latency_sample_count=3,
        status="measured",
        placement_verified=True,
        evidence_ids=("fp16-control-r1", "fp16-control-r2", "fp16-control-r3"),
    )


def full_gpu() -> LayerPlacementCandidate:
    return LayerPlacementCandidate(
        candidate_id="gemma-f16-cuda-full-v1",
        model_id="gemma-f16-proxy",
        model_sha256=MODEL_HASH,
        total_layers=48,
        cpu_prefix_layers=0,
        gpu_suffix_layers=48,
        runtime_gpu_layers=49,
        gpu_weight_bytes=23_832_065_056,
        gpu_kv_bytes=3_388_997_632,
        gpu_compute_bytes=290_455_552,
        cpu_weight_bytes=0,
        cpu_kv_bytes=0,
        cpu_compute_bytes=0,
        latency_us=None,
        latency_sample_count=0,
        status="estimated",
        placement_verified=False,
        evidence_ids=("model-file-size",),
    )


def capacities() -> tuple[DeviceMemoryCapacity, DeviceMemoryCapacity]:
    return (
        DeviceMemoryCapacity("cuda0-vram", 17_175_674_880, 0, 536_870_912),
        DeviceMemoryCapacity("host-ram", 32_862_289_920, 0, 2_147_483_648),
    )


def artifacts() -> tuple[BackendArtifact, ...]:
    roles = (
        "bridge",
        "cold_model",
        "cold_server",
        "phone_model",
        "phone_session",
        "phone_worker",
        "restore_usb",
    )
    return tuple(
        BackendArtifact(role, f"/tmp/{role}", "b" * 64) for role in roles
    )


def offload(layer_ids=range(23)) -> OperatorOffloadContract:
    rows = artifacts()
    split = OperatorSplitPolicy.from_table(
        policy_id="f16-small-m-v1",
        operator_family="dense-ffn",
        layer_ids=layer_ids,
        n_embd=128,
        eligible_columns=512,
        max_tokens=16,
        column_quantum=256,
        alternate_columns=(256,),
        io_type="f16",
        weight_layout="view_safe_dense_ffn",
        table="16:256",
    )
    transport = PhoneTransportContract(
        transport_id="op15-dmabuf",
        protocol="ffn-split",
        host_endpoint="libusb-bulk",
        phone_endpoint="functionfs-dmabuf",
        allocator="malloc-split",
        io_type="f16",
        payload_offset_bytes=128,
        max_payload_bytes=4096,
        queue_depth=1,
        usb_speed_mbps=5000,
        usb_vendor_product="18d1:2d00",
        phone_resource_id="op15-htp",
        transport_resource_id="op15-usb",
        usb_root_resource_id="usb-root-0",
        bridge_residency_id="bridge:" + "b" * 64,
        worker_residency_id="worker:" + "b" * 64,
        reset_generation=0,
        max_reset_recoveries=0,
    )
    layer_tuple = tuple(layer_ids)
    backend = PhoneBackendConfig(
        backend_id="op15",
        phone_serial="phone",
        adb_port=5037,
        compute_backend="HTP0",
        layer_spec=f"{layer_tuple[0]}-{layer_tuple[-1]}",
        bridge_bind="127.0.0.1",
        bridge_port=25660,
        server_timeout_ms=35000,
        session_timeout_s=1800,
        max_requests=120000,
        artifacts=tuple(
            row for row in rows
            if row.role != "cold_server"
        ),
        resident_weight_bytes=3_255_880_909,
        resident_weight_budget_bytes=3_463_438_336,
    )
    return OperatorOffloadContract(
        route_id="gpu-cpu-op15",
        host_resource_id="cpu-prefix",
        split=split,
        transport=transport,
        backend=backend,
    )


class CapacityPlannerTests(unittest.TestCase):
    def test_full_gpu_is_rejected_and_measured_overflow_split_is_selected(self) -> None:
        gpu, cpu = capacities()
        placement = CapacityPlanner(minimum_samples=3).plan(
            route_id="gpu-cpu-op15",
            candidates=(full_gpu(), measured_split()),
            gpu_capacity=gpu,
            cpu_capacity=cpu,
        )
        self.assertEqual(
            placement.selected.candidate_id,
            "gemma-f16-cpu23-cuda25-v1",
        )
        self.assertEqual(placement.cpu_layer_spec, "0-22")
        self.assertEqual(placement.gpu_layer_spec, "23-47")
        self.assertIn(
            ("gemma-f16-cuda-full-v1", "GPU_CAPACITY"),
            placement.rejected,
        )
        self.assertEqual(
            LayerPlacementContract.from_json(placement.to_json()), placement
        )

    def test_no_capacity_fit_fails_closed(self) -> None:
        gpu, cpu = capacities()
        with self.assertRaisesRegex(CapacityError, "no verified"):
            CapacityPlanner().plan(
                route_id="gpu-cpu-op15",
                candidates=(full_gpu(),),
                gpu_capacity=gpu,
                cpu_capacity=cpu,
            )

    def test_phone_resident_weights_must_fit_budget(self) -> None:
        value = offload().backend.to_json()
        value["resident_weight_bytes"] = (
            value["resident_weight_budget_bytes"] + 1
        )
        with self.assertRaisesRegex(OffloadError, "exceed memory budget"):
            PhoneBackendConfig.from_json(value)


class CapacityExecutionPlanTests(unittest.TestCase):
    def placement(self) -> LayerPlacementContract:
        gpu, cpu = capacities()
        return CapacityPlanner(minimum_samples=3).plan(
            route_id="gpu-cpu-op15",
            candidates=(full_gpu(), measured_split()),
            gpu_capacity=gpu,
            cpu_capacity=cpu,
        )

    @staticmethod
    def decision() -> CohortDecision:
        return CohortDecision(
            profile_id="overflow-profile-v1",
            unit_id="cold-17",
            work_set_hash="sha256:" + "c" * 64,
            mode="shadow",
            route_id="gpu-cpu-op15",
            fallback_route_id="gpu-cpu",
            reason="SHADOW_FASTEST_MEASURED_COHORT",
            latency_mean_us=712_861_046,
            latency_upper_us=712_861_046,
            energy_mean_uj=None,
            energy_upper_uj=None,
            runtime_gate=None,
            rejected=(),
        )

    def plan(self, split: OperatorOffloadContract):
        return build_execution_plan(
            plan_id="overflow-treatment-v1",
            epoch_key="sha256:" + "d" * 64,
            decision=self.decision(),
            execution_mode="cuda-cpu-op15",
            trace_sha256="sha256:" + "e" * 64,
            request_count=17,
            input_tokens=11_476,
            output_tokens=6_919,
            model_hashes={"cold": MODEL_HASH},
            artifacts=artifacts(),
            runtime_bindings={"n_gpu_layers": 25},
            expected_work={"phone_calls": 27_485},
            offload=split,
            layer_placement=self.placement(),
            evidence_ids=("fp16-control-r3", "fp16-op15-r1"),
        )

    def test_plan_round_trip_binds_all_three_devices(self) -> None:
        plan = self.plan(offload())
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "plan.json"
            write_execution_plan(path, plan)
            self.assertEqual(load_execution_plan(path), plan)
            value = json.loads(path.read_text(encoding="ascii"))
            value["layer_placement"]["selected"]["runtime_gpu_layers"] = 24
            path.write_text(json.dumps(value), encoding="ascii")
            with self.assertRaises((CapacityError, ExecutionPlanError)):
                load_execution_plan(path)

    def test_phone_split_cannot_include_cuda_suffix(self) -> None:
        with self.assertRaisesRegex(
            ExecutionPlanError, "non-CPU-resident layer"
        ):
            self.plan(offload((23,)))


if __name__ == "__main__":
    unittest.main()
