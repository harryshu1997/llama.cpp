#!/usr/bin/env python3

from __future__ import annotations

import json
import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
sys.path[:0] = [str(REPO_ROOT), str(ROOT)]

from research_dev.scheduler import (  # noqa: E402
    RequestSemantics,
    RouteRuntimeContract,
    RuntimeGateError,
    RuntimeSnapshot,
    evaluate_runtime_gate,
)
from runtime_snapshot_probe import (  # noqa: E402
    parse_nvidia_thermal,
    parse_sha256sum,
    parse_thermal_service,
)
from runtime_gate_audit import audit  # noqa: E402


EPOCH = "sha256:" + "a" * 64


def contract_value() -> dict[str, object]:
    return {
        "schema": "s42-route-runtime-contract-v1",
        "epoch_key": EPOCH,
        "failure_mode": "fallback_before_dispatch",
        "resources": {
            "phone": {
                "heartbeat_max_age_us": 1000,
                "max_temperature_millic": 45000,
                "allowed_thermal_buckets": ["nominal"],
                "allowed_contention_buckets": ["exclusive"],
                "max_slowdown_ppm": 1050000,
                "max_failure_count": 0,
                "reset_generation": 0,
                "required_residency_ids": ["gemma-ffn"],
            }
        },
        "semantics": {
            "kv_owner": "llama_context",
            "kv_migration": False,
            "context_shift": True,
            "full_logits": True,
            "grammar": True,
            "sampler_locations": ["desktop"],
            "speculative_decode": False,
            "cancellation_modes": ["before_dispatch", "cooperative"],
        },
    }


def snapshot_value() -> dict[str, object]:
    return {
        "schema": "s42-runtime-snapshot-v1",
        "snapshot_id": "snapshot-1",
        "generation": 1,
        "epoch_key": EPOCH,
        "captured_at_us": 100,
        "valid_until_us": 200,
        "cancellation_generation": 0,
        "resources": {
            "phone": {
                "ready": True,
                "generation": 1,
                "heartbeat_age_us": 100,
                "temperature_millic": 40000,
                "thermal_bucket": "nominal",
                "contention_bucket": "exclusive",
                "slowdown_ppm": 1020000,
                "failure_count": 0,
                "circuit_open": False,
                "reset_generation": 0,
                "residency_ids": ["gemma-ffn"],
            }
        },
    }


def evaluate(
    contract_row: dict[str, object],
    snapshot_row: dict[str, object] | None,
    semantics: RequestSemantics | None = None,
    now_us: int | None = 150,
):
    return evaluate_runtime_gate(
        RouteRuntimeContract.from_json(contract_row),
        semantics or RequestSemantics(),
        None if snapshot_row is None else RuntimeSnapshot.from_json(snapshot_row),
        now_us,
    )


class RuntimeGateTests(unittest.TestCase):
    def test_idle_physical_audit_is_hash_bound_and_fail_closed(self) -> None:
        snapshot_path = ROOT / "RUNTIME_GATE_IDLE_SNAPSHOT_4060TI_OP15_V1.json"
        audit_path = ROOT / "RUNTIME_GATE_IDLE_AUDIT_4060TI_OP15_V1.json"
        snapshot_raw = snapshot_path.read_bytes()
        snapshot = json.loads(snapshot_raw)
        audit_row = json.loads(audit_path.read_text(encoding="ascii"))
        RuntimeSnapshot.from_json(snapshot)
        self.assertEqual(
            audit_row["snapshot"]["file_sha256"],
            "sha256:" + hashlib.sha256(snapshot_raw).hexdigest(),
        )
        self.assertTrue(
            all(not row["admitted"] for row in audit_row["routes"].values())
        )
        self.assertEqual(snapshot["evidence"]["phone_thermal"]["status"], 0)
        self.assertFalse(snapshot["evidence"]["desktop_topology_qualified"])

    def test_parse_android_thermal_status_and_current_temperatures(self) -> None:
        output = (
            "Thermal Status: 0\n"
            "Cached temperatures:\n"
            "  Temperature{mValue=99.0, mType=0, mName=old, mStatus=0}\n"
            "Current temperatures from HAL:\n"
            "  Temperature{mValue=31.4, mType=1, mName=GPU0, mStatus=0}\n"
            "  Temperature{mValue=0.0, mType=6, mName=vbat, mStatus=6}\n"
        )
        value = parse_thermal_service(output)
        self.assertEqual(value["bucket"], "android-none")
        self.assertEqual(value["max_current_millic"], 31400)

    def test_parse_nvidia_thermal_reasons(self) -> None:
        output = (
            "HW Thermal Slowdown : Not Active\n"
            "SW Thermal Slowdown : Not Active\n"
        )
        self.assertEqual(parse_nvidia_thermal(output), "not-throttled")
        self.assertEqual(
            parse_nvidia_thermal(output.replace("Not Active", "Active", 1)),
            "throttled",
        )

    def test_parse_sha256sum_fails_closed(self) -> None:
        digest = "a" * 64
        self.assertEqual(parse_sha256sum(f"{digest}  file\n"), digest)
        self.assertIsNone(parse_sha256sum("not-a-hash file\n"))

    def test_i3_contract_artifact_matches_compiled_routes(self) -> None:
        contract_path = (
            ROOT
            / "runtime_routes_v1"
            / "I3_RUNTIME_GATE_CONTRACTS_4060TI_OP15_V1.json"
        )
        routes_path = (
            ROOT
            / "runtime_routes_v1"
            / "COMPILED_4060TI_OP15_I3_ROUTES_V1.json"
        )
        contracts = json.loads(contract_path.read_text(encoding="ascii"))
        routes = json.loads(routes_path.read_text(encoding="ascii"))
        self.assertEqual(
            set(contracts["routes"]),
            {route["route_id"] for route in routes["compiled_routes"]},
        )
        self.assertFalse(contracts["qualification"]["activation_ready"])
        for value in contracts["routes"].values():
            contract = RouteRuntimeContract.from_json(value)
            self.assertEqual(contract.epoch_key, routes["epoch_key"])

    def test_i3_contracts_admit_only_a_fully_qualified_snapshot(self) -> None:
        path = (
            ROOT
            / "runtime_routes_v1"
            / "I3_RUNTIME_GATE_CONTRACTS_4060TI_OP15_V1.json"
        )
        document = json.loads(path.read_text(encoding="ascii"))
        requirements: dict[str, object] = {}
        for raw_contract in document["routes"].values():
            contract = RouteRuntimeContract.from_json(raw_contract)
            requirements.update(contract.resources)
        resources = {}
        for resource_id, requirement in requirements.items():
            resources[resource_id] = {
                "ready": True,
                "generation": 1,
                "heartbeat_age_us": (
                    None if requirement.heartbeat_max_age_us is None else 0
                ),
                "temperature_millic": None,
                "thermal_bucket": requirement.allowed_thermal_buckets[0],
                "contention_bucket": requirement.allowed_contention_buckets[0],
                "slowdown_ppm": 1000000,
                "failure_count": 0,
                "circuit_open": False,
                "reset_generation": requirement.reset_generation,
                "residency_ids": list(requirement.required_residency_ids),
            }
        snapshot = {
            "schema": "s42-runtime-snapshot-v1",
            "snapshot_id": "qualified-i3",
            "generation": 1,
            "epoch_key": document["epoch_key"],
            "captured_at_us": 100,
            "valid_until_us": 200,
            "cancellation_generation": 0,
            "resources": resources,
        }
        result = audit(
            document,
            EPOCH,
            snapshot,
            EPOCH,
            RequestSemantics(),
            150,
        )
        self.assertTrue(all(row["admitted"] for row in result["routes"].values()))
        shifted = audit(
            document,
            EPOCH,
            snapshot,
            EPOCH,
            RequestSemantics(context_shift_required=True),
            150,
        )
        self.assertTrue(
            all(
                row["reason"] == "CONTEXT_SHIFT_UNSUPPORTED"
                for row in shifted["routes"].values()
            )
        )

    def test_all_runtime_and_semantic_gates_pass(self) -> None:
        result = evaluate(contract_value(), snapshot_value())
        self.assertTrue(result.admitted)
        self.assertEqual(result.reason, "RUNTIME_GATES_PASS")
        self.assertEqual(result.checked_resources, ("phone",))

    def test_snapshot_and_epoch_are_mandatory(self) -> None:
        self.assertEqual(
            evaluate(contract_value(), None).reason,
            "RUNTIME_SNAPSHOT_MISSING",
        )
        changed = snapshot_value()
        changed["epoch_key"] = "sha256:" + "b" * 64
        self.assertEqual(
            evaluate(contract_value(), changed).reason,
            "RUNTIME_EPOCH_MISMATCH",
        )
        self.assertEqual(
            evaluate(contract_value(), snapshot_value(), now_us=201).reason,
            "RUNTIME_SNAPSHOT_EXPIRED",
        )

    def test_thermal_and_contention_limits_fail_closed(self) -> None:
        thermal = snapshot_value()
        thermal["resources"]["phone"]["temperature_millic"] = 45001
        self.assertEqual(
            evaluate(contract_value(), thermal).reason,
            "RUNTIME_THERMAL_LIMIT",
        )
        contention = snapshot_value()
        contention["resources"]["phone"]["slowdown_ppm"] = 1050001
        self.assertEqual(
            evaluate(contract_value(), contention).reason,
            "RUNTIME_CONTENTION_LIMIT",
        )

    def test_heartbeat_failure_reset_and_residency_fail_closed(self) -> None:
        cases = (
            ("heartbeat_age_us", 1001, "RUNTIME_HEARTBEAT_STALE"),
            ("failure_count", 1, "RUNTIME_FAILURE_LIMIT"),
            ("reset_generation", 1, "RUNTIME_RESET_GENERATION"),
            ("residency_ids", [], "RUNTIME_RESIDENCY_MISSING"),
            ("circuit_open", True, "RUNTIME_CIRCUIT_OPEN"),
        )
        for key, value, reason in cases:
            with self.subTest(key=key):
                snapshot = snapshot_value()
                snapshot["resources"]["phone"][key] = value
                self.assertEqual(evaluate(contract_value(), snapshot).reason, reason)

    def test_kv_sampler_and_speculative_semantics_fail_closed(self) -> None:
        cases = (
            (
                RequestSemantics(kv_migration_required=True),
                "KV_MIGRATION_UNSUPPORTED",
            ),
            (
                RequestSemantics(sampler_location="phone"),
                "SAMPLER_LOCATION_UNSUPPORTED",
            ),
            (
                RequestSemantics(speculative_decode=True),
                "SPECULATIVE_DECODE_UNSUPPORTED",
            ),
        )
        for semantics, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(
                    evaluate(contract_value(), snapshot_value(), semantics).reason,
                    reason,
                )

    def test_cancelled_request_is_never_admitted(self) -> None:
        result = evaluate(
            contract_value(),
            snapshot_value(),
            RequestSemantics(cancelled=True),
        )
        self.assertEqual(result.reason, "REQUEST_CANCELLED")

    def test_contract_rejects_duplicate_capabilities(self) -> None:
        value = contract_value()
        value["semantics"]["sampler_locations"] = ["desktop", "desktop"]
        with self.assertRaisesRegex(RuntimeGateError, "duplicates"):
            RouteRuntimeContract.from_json(value)


if __name__ == "__main__":
    unittest.main()
