#!/usr/bin/env python3

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from campaign_reduce import (  # noqa: E402
    expected_primary,
    metrics_from_reduced,
    reduce_campaign,
    reduce_physical_campaign,
)
from campaign_plan import CAMPAIGN_TOOL_PATHS, build_campaign  # noqa: E402
from evidence_common import (  # noqa: E402
    EvidenceError,
    canonical_bytes,
    digest_bytes,
    digest_file,
)


def metrics(seed: int, phone: bool = False) -> dict:
    result = {
        "cleanup_count": 1,
        "cleanup_p50_ns": 10 + seed,
        "cleanup_p95_ns": 11 + seed,
        "completed_request_count": 70,
        "completion_latency_p50_ns": 100 + seed,
        "completion_latency_p95_ns": 200 + seed,
        "completion_latency_p99_ns": 300 + seed,
        "controller_process_cpu_utilization_milli_pct_p50": 120 + seed,
        "controller_process_swap_growth_bytes": 0,
        "cpu_utilization_milli_pct_p50": 220 + seed,
        "discard_count": 0,
        "discard_p50_ns": None,
        "discard_p95_ns": None,
        "drain_count": 1,
        "drain_p50_ns": 20 + seed,
        "drain_p95_ns": 21 + seed,
        "gpu_energy_scope": "SELECTED_GPU_BOARD_DEVELOPMENT_ONLY",
        "load_count": 1,
        "load_p50_ns": 30 + seed,
        "load_p95_ns": 31 + seed,
        "maximum_global_token_publication_gap_ns": 400 + seed,
        "maximum_model_publication_gap_ns": 500 + seed,
        "minimum_system_mem_available_bytes": 600 + seed,
        "ownership_commit_count": 1,
        "ownership_commit_latency_p95_ns": 40 + seed,
        "peak_controller_process_rss_bytes": 650 + seed,
        "peak_gpu_memory_used_bytes": 700 + seed,
        "queue_p50_ns": 800 + seed,
        "queue_p95_ns": 900 + seed,
        "queue_p99_ns": 1000 + seed,
        "replay_count": 1,
        "replay_p50_ns": 50 + seed,
        "replay_p95_ns": 51 + seed,
        "selected_gpu_board_energy_nj": 1100 + seed,
        "slo_goodput_milli_rps": 1200 + seed,
        "slo_met_count": 65,
        "stranded_request_count": 4,
        "system_swap_growth_bytes": 0,
        "tokens_per_second_milli": 1300 + seed,
        "ttft_p50_ns": 1400 + seed,
        "ttft_p95_ns": 1500 + seed,
        "ttft_p99_ns": 1600 + seed,
        "unload_count": 1,
        "unload_p50_ns": 60 + seed,
        "unload_p95_ns": 61 + seed,
    }
    for phone_name in ("op12", "op15"):
        result.update({
            f"{phone_name}_all_interface_rx_bytes":
                1700 + seed if phone else None,
            f"{phone_name}_all_interface_tx_bytes":
                1800 + seed if phone else None,
            f"{phone_name}_minimum_available_bytes":
                1900 + seed if phone else None,
            f"{phone_name}_swap_growth_bytes": 0 if phone else None,
            f"{phone_name}_thermal_max_millic":
                2000 + seed if phone else None,
        })
    return result


def phone_summary(seed: int = 0) -> dict:
    return {
        "interface_byte_deltas": {
            name: {
                "wlan0": {
                    "rx_bytes": 1700 + seed,
                    "tx_bytes": 1800 + seed,
                },
            }
            for name in ("op12", "op15")
        },
        "memory_min_bytes": {
            name: 1900 + seed for name in ("op12", "op15")
        },
        "status": "PASS",
        "swap_growth_bytes": {
            name: 0 for name in ("op12", "op15")
        },
        "thermal_max_millic": {
            name: 2000 + seed for name in ("op12", "op15")
        },
    }


def validation_for_manifest(path: Path) -> dict:
    value = __import__("json").loads(path.read_text(encoding="ascii"))
    is_phone = value["mode"] in {
        "T1_PHONE_WARM_TIER",
        "T2_PHONE_NO_PROMOTION",
    }
    return {
        "performance_claim_authorized": True,
        "phone_observer_summary": phone_summary() if is_phone else None,
        "status": "S40_RUN_MANIFEST_V5_VALID",
    }


def fixture() -> list[dict]:
    rows = []
    for index, expected in enumerate(expected_primary(1)):
        rows.append({
            **expected,
            "manifest_sha256": f"{index + 1:064x}",
            "metrics": metrics(
                index,
                expected["mode"] in {
                    "T1_PHONE_WARM_TIER",
                    "T2_PHONE_NO_PROMOTION",
                },
            ),
            "performance_claim_authorized": True,
            "run_id": f"run-{index}",
        })
    return rows


def reduced(seed: int = 0) -> dict:
    row = metrics(seed)
    energy = {
        "controller_process_cpu_utilization_milli_pct_p50":
            row["controller_process_cpu_utilization_milli_pct_p50"],
        "controller_process_swap_growth_bytes":
            row["controller_process_swap_growth_bytes"],
        "cpu_utilization_milli_pct_p50":
            row["cpu_utilization_milli_pct_p50"],
        "energy_claim_authorized": False,
        "gpu_energy_nj": row["selected_gpu_board_energy_nj"],
        "gpu_energy_scope": row["gpu_energy_scope"],
        "minimum_system_mem_available_bytes":
            row["minimum_system_mem_available_bytes"],
        "peak_controller_process_rss_bytes":
            row["peak_controller_process_rss_bytes"],
        "peak_gpu_memory_used_bytes": row["peak_gpu_memory_used_bytes"],
        "system_swap_growth_bytes": row["system_swap_growth_bytes"],
    }
    result = {
        name: value
        for name, value in row.items()
        if name not in {
            "gpu_energy_scope",
            "minimum_system_mem_available_bytes",
            "peak_gpu_memory_used_bytes",
            "selected_gpu_board_energy_nj",
            "system_swap_growth_bytes",
        }
    }
    result["phase_duration_ns"] = {
        phase: {
            "count": row[f"{phase}_count"],
            "p50": row[f"{phase}_p50_ns"],
            "p95": row[f"{phase}_p95_ns"],
        }
        for phase in (
            "cleanup", "discard", "drain", "load", "replay", "unload")
    }
    result["energy"] = energy
    result["verdict"] = "PASS"
    return result


SOFTWARE_LOCK = {
    **{
        f"{name}_sha256": digest_file(path)
        for name, path in CAMPAIGN_TOOL_PATHS.items()
    },
    "controller_binary_sha256": "1" * 64,
    "evidence_bundle_manifest_sha256": "2" * 64,
    "executor_bundle_manifest_sha256": "3" * 64,
    "ldd_sha256": "6" * 64,
    "native_bench_binary_sha256": "4" * 64,
    "nvidia_smi_sha256": "5" * 64,
    "python_sha256": "7" * 64,
    "schema": "s40-primary-software-lock-v2",
}


def write_artifact(root: Path, role: str, raw: bytes, index: int) -> dict:
    suffix = ".jsonl" if role in {
        "controller_events", "resource_samples"} else ".json"
    path = root / f"{index:02d}-{role}{suffix}"
    path.write_bytes(raw)
    return {
        "bytes": len(raw),
        "format": "JSONL" if suffix == ".jsonl" else "JSON",
        "path": path.name,
        "record_count": 1 if suffix == ".jsonl" else None,
        "role": role,
        "sha256": digest_bytes(raw),
    }


def physical_fixture(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    requests_raw = b'{"request_id":"fixture"}\n'
    requests = root / "requests.jsonl"
    requests.write_bytes(requests_raw)
    contract = root / "contract.json"
    contract.write_bytes(canonical_bytes({
        "workload": {
            "requests_path": requests.name,
            "requests_sha256": digest_bytes(requests_raw),
        },
    }))
    campaign = build_campaign(
        1,
        campaign_id="physical-test",
        experiment_contract_sha256=digest_bytes(contract.read_bytes()),
        software_lock=SOFTWARE_LOCK,
    )
    campaign_path = root / "campaign.json"
    campaign_path.write_bytes(canonical_bytes(campaign))
    campaign_sha256 = digest_bytes(campaign_path.read_bytes())
    manifests = []
    for index, row in enumerate(campaign["primary"]):
        run_root = root / f"run-{index:02d}"
        run_root.mkdir()
        launch = canonical_bytes({
            "run_id": row["run_id"],
            "started_ns": 1000 + index * 100,
            "stopped_ns": 1050 + index * 100,
        })
        artifacts = [
            write_artifact(
                run_root, "controller_events", b"{}\n", 0),
            write_artifact(
                run_root, "resource_samples", b"{}\n", 1),
            write_artifact(run_root, "trace_start", b"{}\n", 2),
            write_artifact(run_root, "controller_launch", launch, 3),
        ]
        devices = [{
            "boot_id": "boot-fixture",
            "device_role": "GPU",
            "stable_id": "gpu-fixture",
        }]
        if row["mode"] == "C2_GPU_PLUS_CPU_WARM_EXECUTOR":
            devices.append({
                "boot_id": "boot-fixture",
                "device_role": "CPU",
                "stable_id": "cpu-fixture",
            })
        if row["mode"] in {
                "T1_PHONE_WARM_TIER", "T2_PHONE_NO_PROMOTION"}:
            devices.extend([
                {
                    "boot_id": "boot-op12",
                    "device_role": "OP12",
                    "stable_id": "serial-op12",
                },
                {
                    "boot_id": "boot-op15",
                    "device_role": "OP15",
                    "stable_id": "serial-op15",
                },
            ])
        manifest = {
            "artifacts": artifacts,
            "cache_regime": row["cache_regime"],
            "campaign_binding": {
                "campaign_id": campaign["campaign_id"],
                "campaign_sha256": campaign_sha256,
                "order": row["order"],
                "phase": row["phase"],
            },
            "development": False,
            "devices": devices,
            "mode": row["mode"],
            "repeat_index": row["repeat_index"],
            "run_id": row["run_id"],
        }
        path = run_root / "manifest.json"
        path.write_bytes(canonical_bytes(manifest))
        manifests.append(path)
    return contract, campaign_path, manifests


class CampaignReduceTests(unittest.TestCase):
    def test_complete_primary_campaign_is_reduced(self):
        result = reduce_campaign(
            fixture(),
            campaign_sha256="a" * 64,
            t2_repetitions=1,
        )
        self.assertEqual(
            result["status"],
            "MECHANICS_ONLY_COMPLETE_WITH_SERVICE_FAILURES",
        )
        self.assertEqual(result["primary_run_count"], 13)
        primary = [
            row for row in result["cells"]
            if row["mode"] != "T2_PHONE_NO_PROMOTION"
        ]
        self.assertTrue(all(row["repetitions"] == 3 for row in primary))
        self.assertTrue(
            all(
                row["service_status"] == "INCLUDES_STRANDED_REQUESTS"
                for row in primary
            )
        )
        self.assertTrue(
            all(
                row["selected_gpu_board_energy_ratio_milli"] is None
                and not row["service_comparable"]
                for row in result["comparisons_against_c1_warm"]
            )
        )
        self.assertFalse(result["energy_claim_authorized"])
        self.assertEqual(
            result["evidence_scope"],
            "MECHANICS_ONLY_CALLER_SUPPLIED",
        )

    def test_missing_or_reordered_run_is_rejected(self):
        rows = fixture()
        with self.assertRaisesRegex(EvidenceError, "run count"):
            reduce_campaign(
                rows[:-1],
                campaign_sha256="a" * 64,
                t2_repetitions=1,
            )
        rows = fixture()
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(EvidenceError, "prospective order"):
            reduce_campaign(
                rows,
                campaign_sha256="a" * 64,
                t2_repetitions=1,
            )

    def test_duplicate_manifest_or_run_is_rejected(self):
        rows = fixture()
        rows[1]["manifest_sha256"] = rows[0]["manifest_sha256"]
        with self.assertRaisesRegex(EvidenceError, "duplicate manifest"):
            reduce_campaign(
                rows,
                campaign_sha256="a" * 64,
                t2_repetitions=1,
            )
        rows = fixture()
        rows[1]["run_id"] = rows[0]["run_id"]
        with self.assertRaisesRegex(EvidenceError, "duplicate run"):
            reduce_campaign(
                rows,
                campaign_sha256="a" * 64,
                t2_repetitions=1,
            )

    def test_unauthorized_performance_is_rejected(self):
        rows = fixture()
        rows[0]["performance_claim_authorized"] = False
        with self.assertRaisesRegex(EvidenceError, "not authorized"):
            reduce_campaign(
                rows,
                campaign_sha256="a" * 64,
                t2_repetitions=1,
            )

    def test_terminal_and_slo_counts_are_fail_closed(self):
        rows = fixture()
        rows[0]["metrics"]["stranded_request_count"] = 3
        with self.assertRaisesRegex(EvidenceError, "conservation"):
            reduce_campaign(
                rows,
                campaign_sha256="a" * 64,
                t2_repetitions=1,
            )
        rows = fixture()
        rows[0]["metrics"]["slo_met_count"] = 71
        with self.assertRaisesRegex(EvidenceError, "SLO count"):
            reduce_campaign(
                rows,
                campaign_sha256="a" * 64,
                t2_repetitions=1,
            )

    def test_metric_and_scope_mutations_are_rejected(self):
        rows = fixture()
        del rows[0]["metrics"]["queue_p99_ns"]
        with self.assertRaisesRegex(EvidenceError, "metric key"):
            reduce_campaign(
                rows,
                campaign_sha256="a" * 64,
                t2_repetitions=1,
            )
        rows = fixture()
        rows[0]["metrics"]["gpu_energy_scope"] = "SERVER_WALL"
        with self.assertRaisesRegex(EvidenceError, "energy scope"):
            reduce_campaign(
                rows,
                campaign_sha256="a" * 64,
                t2_repetitions=1,
            )

    def test_phase_counts_and_phone_metric_applicability_are_fail_closed(self):
        rows = fixture()
        rows[0]["metrics"]["drain_count"] = 0
        with self.assertRaisesRegex(EvidenceError, "expected null"):
            reduce_campaign(
                rows,
                campaign_sha256="a" * 64,
                t2_repetitions=1,
            )

        rows = fixture()
        phone_row = next(
            row for row in rows
            if row["mode"] == "T1_PHONE_WARM_TIER")
        phone_row["metrics"]["op12_minimum_available_bytes"] = None
        with self.assertRaisesRegex(
                EvidenceError, "all-null or all-integer"):
            reduce_campaign(
                rows,
                campaign_sha256="a" * 64,
                t2_repetitions=1,
            )

        rows = fixture()
        rows[0]["metrics"]["op12_minimum_available_bytes"] = 1
        for name in (
                "op12_all_interface_rx_bytes",
                "op12_all_interface_tx_bytes",
                "op12_swap_growth_bytes",
                "op12_thermal_max_millic",
                "op15_all_interface_rx_bytes",
                "op15_all_interface_tx_bytes",
                "op15_minimum_available_bytes",
                "op15_swap_growth_bytes",
                "op15_thermal_max_millic"):
            rows[0]["metrics"][name] = 1
        with self.assertRaisesRegex(
                EvidenceError, "phone metric applicability"):
            reduce_campaign(
                rows,
                campaign_sha256="a" * 64,
                t2_repetitions=1,
            )

    def test_float_and_bool_metrics_are_rejected(self):
        for value in (1.0, True):
            with self.subTest(value=value):
                rows = fixture()
                rows[0]["metrics"]["queue_p50_ns"] = value
                with self.assertRaisesRegex(EvidenceError, "integer"):
                    reduce_campaign(
                        rows,
                        campaign_sha256="a" * 64,
                        t2_repetitions=1,
                    )

    def test_all_stranded_run_remains_reportable(self):
        rows = fixture()
        rows[0]["metrics"]["completed_request_count"] = 0
        rows[0]["metrics"]["stranded_request_count"] = 74
        rows[0]["metrics"]["slo_met_count"] = 0
        for name in (
                "completion_latency_p50_ns",
                "completion_latency_p95_ns",
                "completion_latency_p99_ns",
                "queue_p50_ns",
                "queue_p95_ns",
                "queue_p99_ns",
                "ttft_p50_ns",
                "ttft_p95_ns",
                "ttft_p99_ns"):
            rows[0]["metrics"][name] = None
        result = reduce_campaign(
            rows,
            campaign_sha256="a" * 64,
            t2_repetitions=1,
        )
        cell = next(
            row for row in result["cells"]
            if row["mode"] == rows[0]["mode"]
            and row["cache_regime"] == rows[0]["cache_regime"]
        )
        self.assertIsNone(cell["medians"]["queue_p50_ns"])
        self.assertEqual(
            cell["service_status"],
            "INCLUDES_ZERO_COMPLETION_RUN",
        )

    def test_all_stranded_campaign_is_never_labeled_pass(self):
        rows = fixture()
        for row in rows:
            row["metrics"]["completed_request_count"] = 0
            row["metrics"]["stranded_request_count"] = 74
            row["metrics"]["slo_met_count"] = 0
            for name in (
                    "completion_latency_p50_ns",
                    "completion_latency_p95_ns",
                    "completion_latency_p99_ns",
                    "queue_p50_ns",
                    "queue_p95_ns",
                    "queue_p99_ns",
                    "ttft_p50_ns",
                    "ttft_p95_ns",
                    "ttft_p99_ns"):
                row["metrics"][name] = None
        result = reduce_campaign(
            rows,
            campaign_sha256="a" * 64,
            t2_repetitions=1,
        )
        self.assertEqual(
            result["status"],
            "MECHANICS_ONLY_COMPLETE_WITH_SERVICE_FAILURES",
        )
        self.assertTrue(
            all(
                row["service_status"] == "INCLUDES_ZERO_COMPLETION_RUN"
                for row in result["cells"]
            )
        )

    def test_energy_ratio_requires_equal_completed_work(self):
        rows = fixture()
        for row in rows:
            row["metrics"]["completed_request_count"] = 74
            row["metrics"]["stranded_request_count"] = 0
            row["metrics"]["slo_met_count"] = 74
        result = reduce_campaign(
            rows,
            campaign_sha256="a" * 64,
            t2_repetitions=1,
        )
        self.assertEqual(result["status"], "MECHANICS_ONLY_COMPLETE")
        self.assertTrue(
            all(
                row["service_comparable"]
                and row["selected_gpu_board_energy_ratio_milli"] is not None
                for row in result["comparisons_against_c1_warm"]
            )
        )

    def test_latency_presence_must_match_completion_count(self):
        rows = fixture()
        rows[0]["metrics"]["queue_p50_ns"] = None
        with self.assertRaisesRegex(EvidenceError, "integer"):
            reduce_campaign(
                rows,
                campaign_sha256="a" * 64,
                t2_repetitions=1,
            )
        rows = fixture()
        rows[0]["metrics"]["completed_request_count"] = 0
        rows[0]["metrics"]["stranded_request_count"] = 74
        rows[0]["metrics"]["slo_met_count"] = 0
        with self.assertRaisesRegex(EvidenceError, "expected null"):
            reduce_campaign(
                rows,
                campaign_sha256="a" * 64,
                t2_repetitions=1,
            )

    def test_input_is_not_mutated(self):
        rows = fixture()
        before = copy.deepcopy(rows)
        reduce_campaign(
            rows,
            campaign_sha256="a" * 64,
            t2_repetitions=1,
        )
        self.assertEqual(rows, before)

    def test_metrics_are_derived_from_reduced_evidence(self):
        self.assertEqual(metrics_from_reduced(reduced()), metrics(0))

    def test_reduced_verdict_and_energy_scope_are_fail_closed(self):
        value = reduced()
        value["verdict"] = "FAIL_EXECUTOR"
        with self.assertRaisesRegex(EvidenceError, "fail-closed"):
            metrics_from_reduced(value)
        value = reduced()
        value["energy"]["energy_claim_authorized"] = True
        with self.assertRaisesRegex(EvidenceError, "energy scope"):
            metrics_from_reduced(value)

    def test_physical_campaign_reopens_raw_manifests_and_ledgers(self):
        with tempfile.TemporaryDirectory(
                prefix="s40_campaign_") as directory:
            contract, campaign_path, manifests = physical_fixture(
                Path(directory))
            calls = []

            def validate(path, _contract):
                value = __import__("json").loads(
                    path.read_text(encoding="ascii"))
                return {
                    "performance_claim_authorized": True,
                    "phone_observer_summary": (
                        phone_summary()
                        if value["mode"] in {
                            "T1_PHONE_WARM_TIER",
                            "T2_PHONE_NO_PROMOTION",
                        } else None
                    ),
                    "status": "S40_RUN_MANIFEST_V5_VALID",
                    "run_id": value["run_id"],
                }

            def reduce_raw(events, requests, resources, trace):
                calls.append((events, requests, resources, trace))
                self.assertNotIn(
                    "run-", str(events),
                    "reducer must consume private raw snapshots",
                )
                return reduced(len(calls))

            with patch(
                    "campaign_reduce.validate_run_manifest",
                    side_effect=validate), patch(
                        "campaign_reduce.reduce_paths",
                        side_effect=reduce_raw):
                result = reduce_physical_campaign(
                    campaign_path, manifests, contract)
            self.assertEqual(result["status"],
                             "MEASURED_COMPLETE_WITH_SERVICE_FAILURES")
            self.assertEqual(
                result["evidence_scope"],
                "REVALIDATED_PHYSICAL_MANIFESTS",
            )
            self.assertEqual(len(calls), 13)
            phone_cell = next(
                row for row in result["cells"]
                if row["mode"] == "T1_PHONE_WARM_TIER")
            self.assertEqual(
                phone_cell["medians"]["op12_all_interface_rx_bytes"],
                1700,
            )
            self.assertEqual(
                phone_cell["medians"]["op15_thermal_max_millic"],
                2000,
            )

    def test_physical_campaign_rejects_fabricated_summary_and_raw_mutation(self):
        with tempfile.TemporaryDirectory(
                prefix="s40_campaign_") as directory:
            contract, campaign_path, manifests = physical_fixture(
                Path(directory))
            manifest = __import__("json").loads(
                manifests[0].read_text(encoding="ascii"))
            manifest["metrics"] = metrics(0)
            manifests[0].write_bytes(canonical_bytes(manifest))
            with patch(
                    "campaign_reduce.validate_run_manifest",
                    side_effect=lambda path, _contract:
                    validation_for_manifest(path)), patch(
                        "campaign_reduce.reduce_paths",
                        return_value=reduced(99)):
                result = reduce_physical_campaign(
                    campaign_path, manifests, contract)
            first_cell = next(
                row for row in result["cells"]
                if row["mode"] == "C1_GPU_ONLY_OPTIMIZED"
                and row["cache_regime"] == "WARM_HOST_CACHE"
            )
            self.assertNotEqual(
                first_cell["medians"]["queue_p50_ns"],
                manifest["metrics"]["queue_p50_ns"],
            )

            contract, campaign_path, manifests = physical_fixture(
                Path(directory) / "second")
            first = __import__("json").loads(
                manifests[0].read_text(encoding="ascii"))
            event = manifests[0].parent / next(
                row["path"] for row in first["artifacts"]
                if row["role"] == "controller_events")
            event.write_bytes(b'{"mutated":true}\n')
            with patch(
                    "campaign_reduce.validate_run_manifest",
                    return_value={
                        "performance_claim_authorized": True,
                        "status": "S40_RUN_MANIFEST_V5_VALID",
                    }):
                with self.assertRaisesRegex(EvidenceError, "byte binding"):
                    reduce_physical_campaign(
                        campaign_path, manifests, contract)

    def test_physical_campaign_rejects_reorder_overlap_and_legacy_status(self):
        with tempfile.TemporaryDirectory(
                prefix="s40_campaign_") as directory:
            contract, campaign_path, manifests = physical_fixture(
                Path(directory))
            with patch(
                    "campaign_reduce.validate_run_manifest",
                    side_effect=lambda path, _contract:
                    validation_for_manifest(path)), patch(
                        "campaign_reduce.reduce_paths",
                        return_value=reduced()):
                with self.assertRaisesRegex(
                        EvidenceError, "prospective campaign row"):
                    reduce_physical_campaign(
                        campaign_path,
                        [manifests[1], manifests[0], *manifests[2:]],
                        contract,
                    )

            launch_record = next(
                row for row in __import__("json").loads(
                    manifests[1].read_text(encoding="ascii"))["artifacts"]
                if row["role"] == "controller_launch")
            launch_path = manifests[1].parent / launch_record["path"]
            launch = __import__("json").loads(
                launch_path.read_text(encoding="ascii"))
            launch["started_ns"] = 1040
            launch_path.write_bytes(canonical_bytes(launch))
            manifest = __import__("json").loads(
                manifests[1].read_text(encoding="ascii"))
            target = next(
                row for row in manifest["artifacts"]
                if row["role"] == "controller_launch")
            target["bytes"] = launch_path.stat().st_size
            target["sha256"] = digest_bytes(launch_path.read_bytes())
            manifests[1].write_bytes(canonical_bytes(manifest))
            with patch(
                    "campaign_reduce.validate_run_manifest",
                    return_value={
                        "performance_claim_authorized": True,
                        "status": "S40_RUN_MANIFEST_V5_VALID",
                    }), patch(
                        "campaign_reduce.reduce_paths",
                        return_value=reduced()):
                with self.assertRaisesRegex(
                        EvidenceError, "overlap or are reordered"):
                    reduce_physical_campaign(
                        campaign_path, manifests, contract)

            with patch(
                    "campaign_reduce.validate_run_manifest",
                    return_value={
                        "performance_claim_authorized": True,
                        "status": "S40_RUN_MANIFEST_V3_VALID",
                    }):
                with self.assertRaisesRegex(
                        EvidenceError, "not authorized"):
                    reduce_physical_campaign(
                        campaign_path, manifests, contract)

    def test_physical_campaign_rejects_cross_run_phone_reboot(self):
        with tempfile.TemporaryDirectory(
                prefix="s40_campaign_") as directory:
            contract, campaign_path, manifests = physical_fixture(
                Path(directory))
            phone_manifests = [
                path for path in manifests
                if __import__("json").loads(
                    path.read_text(encoding="ascii"))["mode"]
                in {"T1_PHONE_WARM_TIER", "T2_PHONE_NO_PROMOTION"}
            ]
            target = phone_manifests[1]
            value = __import__("json").loads(
                target.read_text(encoding="ascii"))
            op12 = next(
                device for device in value["devices"]
                if device["device_role"] == "OP12")
            op12["boot_id"] = "boot-op12-restarted"
            target.write_bytes(canonical_bytes(value))

            with patch(
                    "campaign_reduce.validate_run_manifest",
                    side_effect=lambda path, _contract:
                    validation_for_manifest(path)), patch(
                        "campaign_reduce.reduce_paths",
                        return_value=reduced()):
                with self.assertRaisesRegex(
                        EvidenceError,
                        "device identity or boot changed for OP12"):
                    reduce_physical_campaign(
                        campaign_path, manifests, contract)


if __name__ == "__main__":
    unittest.main()
