#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
sys.path[:0] = [str(REPO_ROOT), str(ROOT)]
SCRIPT = ROOT / "small_model_phone_v1/profile_selector.py"
SPEC = importlib.util.spec_from_file_location("small_model_profile_selector", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SELECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SELECTOR
SPEC.loader.exec_module(SELECTOR)

from research_dev.scheduler import (  # noqa: E402
    RuntimeSnapshot,
    UnifiedScheduler,
    load_lifecycle_profile_set,
)


EPOCH_KEY = "sha256:28e0863362abdcfcf861ee92ee9d98b76701319171943f9e0833e68a6bc19b80"
MODEL_RESIDENCY = "sha256:f046db1dc724cf4f6f0a0c5917e922823b73eb1d27b8f9a9c2797f7866974804"
RPC_RESIDENCY = "bge-embedding-rpc-session-v1"


def runtime_snapshot(
    *,
    model_resident: bool = True,
    phone_ready: bool = True,
    phone_temperature_millic: int = 35000,
) -> RuntimeSnapshot:
    def state(
        residency_ids: list[str],
        *,
        temperature_millic: int | None,
    ) -> dict[str, object]:
        return {
            "circuit_open": False,
            "contention_bucket": "qualified",
            "failure_count": 0,
            "generation": 1,
            "heartbeat_age_us": 100,
            "ready": True,
            "reset_generation": 0,
            "residency_ids": residency_ids,
            "slowdown_ppm": 1000000,
            "temperature_millic": temperature_millic,
            "thermal_bucket": "nominal",
        }

    value = {
        "cancellation_generation": 0,
        "captured_at_us": 100,
        "epoch_key": EPOCH_KEY,
        "generation": 1,
        "resources": {
            "op15-adreno": state(
                [MODEL_RESIDENCY] if model_resident else [],
                temperature_millic=phone_temperature_millic,
            ),
            "usb-token-rpc": state(
                [RPC_RESIDENCY],
                temperature_millic=None,
            ),
        },
        "schema": "s42-runtime-snapshot-v1",
        "snapshot_id": "bge-op15-snapshot-1",
        "valid_until_us": 200,
    }
    value["resources"]["op15-adreno"]["ready"] = phone_ready
    return RuntimeSnapshot.from_json(value)


def schedule(
    receipt: object | None,
    groups: int,
    *,
    snapshot: RuntimeSnapshot | None = None,
    deadline_us: int = 1_000_000_000,
):
    _, profile_set = load_lifecycle_profile_set(
        "bge-small-cuda-lifecycle-test",
        "cuda0",
        SELECTOR.PROFILE_PATHS,
        SELECTOR.CUDA_EPOCH_OPEN,
        frozenset({SELECTOR.CUDA_EPOCH_REUSED}),
    )
    scheduler = UnifiedScheduler(
        (),
        "enforce",
        lifecycle_profiles=(profile_set,),
        runtime_snapshot=snapshot,
    )
    scheduler.update_lifecycle("cuda0", receipt)
    request = SELECTOR.make_request("bge-request", groups, 0, deadline_us)
    return scheduler.profile_for(request.workload_id), scheduler.schedule(
        request, runtime_now_us=150
    )


class SmallModelPhoneProfileTests(unittest.TestCase):
    def test_profiles_have_valid_hashes_and_expected_epoch_metadata(self) -> None:
        opened = SELECTOR.load_selected_profile(None)
        reused = SELECTOR.load_selected_profile(SELECTOR.CudaEpochReceipt(
            SELECTOR.CUDA_EPOCH_REUSED,
            "tail-charge-1",
        ))
        self.assertEqual(opened.epoch_state, SELECTOR.CUDA_EPOCH_OPEN)
        self.assertEqual(reused.epoch_state, SELECTOR.CUDA_EPOCH_REUSED)
        self.assertEqual(opened.raw["cuda_epoch"]["tail_charge_uj"], 313014997)
        self.assertEqual(reused.raw["cuda_epoch"]["tail_charge_uj"], 0)

    def test_unknown_epoch_fails_closed_to_epoch_open_profile(self) -> None:
        selected = SELECTOR.load_selected_profile(SELECTOR.CudaEpochReceipt(
            SELECTOR.CUDA_EPOCH_UNKNOWN,
        ))
        self.assertEqual(selected.epoch_state, SELECTOR.CUDA_EPOCH_OPEN)

    def test_reused_epoch_requires_a_tail_charge_receipt(self) -> None:
        with self.assertRaises(SELECTOR.ProfileSelectionError):
            SELECTOR.load_selected_profile(SELECTOR.CudaEpochReceipt(
                SELECTOR.CUDA_EPOCH_REUSED,
            ))

    def test_open_epoch_cannot_reuse_a_tail_charge_receipt(self) -> None:
        with self.assertRaises(SELECTOR.ProfileSelectionError):
            SELECTOR.load_selected_profile(SELECTOR.CudaEpochReceipt(
                SELECTOR.CUDA_EPOCH_OPEN,
                "tail-charge-1",
            ))

    def test_request_helper_binds_exact_batch32_geometry(self) -> None:
        request = SELECTOR.make_request("r0", 2, 10, 1_000_000)
        self.assertEqual(request.input_tokens, 2176)
        self.assertEqual(request.features["batch32_groups"], 2)
        self.assertEqual(request.workload_id, SELECTOR.WORKLOAD_ID)
        self.assertEqual(request.semantics.kv_owner, "none")
        self.assertFalse(request.semantics.full_logits_required)
        with self.assertRaises(SELECTOR.ProfileSelectionError):
            SELECTOR.make_request("r1", 0, 0, 1_000_000)

    def test_epoch_open_selects_phone_through_conservative_threshold(self) -> None:
        receipt = SELECTOR.CudaEpochReceipt(SELECTOR.CUDA_EPOCH_OPEN)
        _, below = schedule(receipt, 106, snapshot=runtime_snapshot())
        _, above = schedule(receipt, 107, snapshot=runtime_snapshot())
        self.assertEqual(below.route_id, "phone-adreno")
        self.assertEqual(below.reason, "VERIFIED_ENERGY_SAVING")
        self.assertEqual(above.route_id, "desktop-cuda")
        self.assertIn(("phone-adreno", "ENERGY_MARGIN"), above.rejected)

    def test_mean_energy_crossover_is_between_119_and_120_batches(self) -> None:
        selected = SELECTOR.load_selected_profile(None)
        routes = {route.route_id: route for route in selected.profile.routes}
        for groups, favored in ((119, "phone-adreno"), (120, "desktop-cuda")):
            request = SELECTOR.make_request(
                f"mean-{groups}", groups, 0, 1_000_000_000
            )
            costs = {
                route_id: route.energy.predict_uj(
                    request,
                    route.latency.predict_us(request),
                )
                for route_id, route in routes.items()
            }
            self.assertEqual(min(costs, key=costs.get), favored)

    def test_tail_reused_profile_selects_cuda(self) -> None:
        receipt = SELECTOR.CudaEpochReceipt(
            SELECTOR.CUDA_EPOCH_REUSED,
            "tail-charge-1",
        )
        _, decision = schedule(receipt, 1, snapshot=runtime_snapshot())
        self.assertEqual(decision.route_id, "desktop-cuda")
        self.assertIn(("phone-adreno", "ENERGY_MARGIN"), decision.rejected)

    def test_cuda_tail_is_present_only_in_epoch_open_profile(self) -> None:
        opened = SELECTOR.load_selected_profile(None)
        reused = SELECTOR.load_selected_profile(SELECTOR.CudaEpochReceipt(
            SELECTOR.CUDA_EPOCH_REUSED,
            "tail-charge-1",
        ))
        request = SELECTOR.make_request("tail", 1, 0, 1_000_000)
        open_cuda = next(route for route in opened.profile.routes if route.baseline)
        reused_cuda = next(route for route in reused.profile.routes if route.baseline)
        open_energy = open_cuda.energy.predict_uj(request, 20304)
        reused_energy = reused_cuda.energy.predict_uj(request, 20304)
        assert open_energy is not None and reused_energy is not None
        self.assertEqual(open_energy - reused_energy, 313014997)

    def test_relaxed_latency_still_obeys_request_deadline(self) -> None:
        receipt = SELECTOR.CudaEpochReceipt(SELECTOR.CUDA_EPOCH_OPEN)
        _, decision = schedule(
            receipt,
            1,
            snapshot=runtime_snapshot(),
            deadline_us=100000,
        )
        self.assertEqual(decision.route_id, "desktop-cuda")
        self.assertIn(("phone-adreno", "SLO_INFEASIBLE"), decision.rejected)

    def test_missing_runtime_snapshot_falls_back_to_cuda(self) -> None:
        receipt = SELECTOR.CudaEpochReceipt(SELECTOR.CUDA_EPOCH_OPEN)
        _, decision = schedule(receipt, 1)
        self.assertEqual(decision.route_id, "desktop-cuda")
        self.assertIn(
            ("phone-adreno", "RUNTIME_SNAPSHOT_MISSING"),
            decision.rejected,
        )

    def test_missing_phone_residency_falls_back_to_cuda(self) -> None:
        receipt = SELECTOR.CudaEpochReceipt(SELECTOR.CUDA_EPOCH_OPEN)
        _, decision = schedule(
            receipt,
            1,
            snapshot=runtime_snapshot(model_resident=False),
        )
        self.assertEqual(decision.route_id, "desktop-cuda")
        self.assertIn(
            ("phone-adreno", "RUNTIME_RESIDENCY_MISSING"),
            decision.rejected,
        )

    def test_phone_thermal_limit_falls_back_to_cuda(self) -> None:
        receipt = SELECTOR.CudaEpochReceipt(SELECTOR.CUDA_EPOCH_OPEN)
        _, decision = schedule(
            receipt,
            1,
            snapshot=runtime_snapshot(phone_temperature_millic=45001),
        )
        self.assertEqual(decision.route_id, "desktop-cuda")
        self.assertIn(
            ("phone-adreno", "RUNTIME_THERMAL_LIMIT"),
            decision.rejected,
        )

    def test_compact_physical_result_matches_scheduler_threshold(self) -> None:
        path = (
            SELECTOR.RESULT_ROOT / "BGE_BATCH32_RESULT.json"
        )
        result = json.loads(path.read_text(encoding="ascii"))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["scheduler_thresholds"][
                "max_phone_batch32_groups_cuda_epoch_open"
            ],
            106,
        )

    def test_profiles_are_bound_to_compact_and_raw_evidence(self) -> None:
        compact = SELECTOR.RESULT_ROOT / "BGE_BATCH32_RESULT.json"
        manifest = SELECTOR.RESULT_ROOT / "RAW_EVIDENCE_SHA256SUMS"
        compact_id = "sha256:" + hashlib.sha256(compact.read_bytes()).hexdigest()
        manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.assertEqual(
            compact_id,
            "sha256:ddb84fa2a28b63970ffcbbb029838bf6fb9f6d581e5b26b75b322ecd280444aa",
        )
        self.assertEqual(
            manifest_hash,
            "a1297a9b0a537541cbe102f020ac73fbce9f0006c449dcc960a007a721288f51",
        )
        for receipt in (
            None,
            SELECTOR.CudaEpochReceipt(
                SELECTOR.CUDA_EPOCH_REUSED,
                "tail-charge-1",
            ),
        ):
            selected = SELECTOR.load_selected_profile(receipt)
            self.assertTrue(all(
                compact_id in route.evidence_ids
                for route in selected.profile.routes
            ))


if __name__ == "__main__":
    unittest.main()
