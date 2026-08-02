#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
S41 = HERE.parent
sys.path.insert(0, str(S41))

import reduce_server_results as reducer  # noqa: E402
import render_server_graphs as graphs  # noqa: E402


MODELS = ("gemma-4-12b-it-q8_0", "qwen3-14b-q4_k_m")


def write_json(path: Path, value):
    path.write_bytes(reducer.canonical_bytes(value))


def write_jsonl(path: Path, rows):
    path.write_bytes(b"".join(reducer.canonical_bytes(row) for row in rows))


def write_manifest(root: Path):
    lines = []
    for path in sorted(item for item in root.iterdir() if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}\n")
    (root / "SHA256SUMS.txt").write_text("".join(lines), encoding="ascii")


def raw_run(
    root: Path,
    completed_count: int = 4,
    energy_end_ns: int = 5_000_000_000,
    publish: bool = True,
    cpu: bool = True,
):
    start_ns = 1_000_000_000
    end_ns = 5_000_000_000
    events = [{
        "kind": "replay_start",
        "t_ns": start_ns,
    }]
    for index in range(4):
        model_id = MODELS[index % 2]
        scheduled_ns = start_ns + index * 500_000_000
        events.append({
            "kind": "request_arrival",
            "model_id": model_id,
            "request_index": index,
            "scheduled_t_ns": scheduled_ns,
            "slo_us": 30_000_000,
            "t_ns": scheduled_ns,
        })
        if index < completed_count:
            events.append({
                "completion_ns": scheduled_ns + 300_000_000,
                "first_token_ns": scheduled_ns + 100_000_000,
                "kind": "request_complete",
                "model_id": model_id,
                "request_index": index,
                "scheduled_arrival_ns": scheduled_ns,
                "slo_us": 30_000_000,
                "tokens": list(range(index * 8, index * 8 + 8)),
            })
    events.extend([
        {
            "intent_index": 0,
            "kind": "switch_started",
            "scheduled_t_ns": 2_000_000_000,
            "t_ns": 2_010_000_000,
            "to_model_id": MODELS[1],
        },
        {
            "kind": "model_load_start",
            "label": "switch-00",
            "model_id": MODELS[1],
            "t_ns": 2_050_000_000,
        },
        {
            "kind": "model_ready",
            "label": "switch-00",
            "model_id": MODELS[1],
            "t_ns": 2_400_000_000,
        },
    ])
    if publish:
        events.append({
            "intent_index": 0,
            "kind": "model_published",
            "publication_gap_ns": 500_000_000,
            "published_ns": 2_500_000_000,
            "t_ns": 2_500_000_000,
        })
    events.append({"kind": "replay_end", "t_ns": end_ns})
    samples = []
    for index, t_ns in enumerate(range(start_ns, end_ns + 1, 500_000_000)):
        row = {
            "gpu_power_instant_mw": 100_000,
            "process_rss_bytes": 2_000_000_000 + index * 10_000_000,
            "system_mem_available_bytes": 30_000_000_000 - index * 10_000_000,
            "t_ns": t_ns,
        }
        if cpu:
            row["server_cpu_utilization_milli_pct"] = 25_000 + index * 100
        samples.append(row)
    energy_nj = 100_000 * (energy_end_ns - start_ns) // 1000
    report = {
        "energy": {
            "energy_nj": energy_nj,
            "window_end_ns": energy_end_ns,
            "window_start_ns": start_ns,
        },
        "paid_end_ns": end_ns,
        "paid_start_ns": start_ns,
        "schema": "s39-cp0d-replay-v1",
        "status": (
            "RAW_DESKTOP_REPLAY_PASS_ANALYSIS_PENDING"
            if completed_count == 4 else "FAIL_STRANDED_REQUESTS"
        ),
    }
    write_jsonl(root / "events.jsonl", events)
    write_jsonl(root / "resource_samples.jsonl", samples)
    write_json(root / "replay.json", report)
    write_manifest(root)


def reduce_fixture(
    root: Path,
    label: str,
    mode: str,
    repeat: int,
    cache: str,
):
    return reducer.reduce_run(
        root, label, mode, repeat, cache, MODELS, 1_000_000_000)


def summary(runs):
    return {
        "energy_claim": (
            "Selected GPU board only; not server-wall or "
            "total-system energy."
        ),
        "models": [
            {"id": MODELS[0], "label": "Gemma 4 12B Q8_0"},
            {"id": MODELS[1], "label": "Qwen3 14B Q4_K_M"},
        ],
        "runs": runs,
        "schema": reducer.SCHEMA,
    }


class ServerGraphTests(unittest.TestCase):
    def test_raw_s39_run_reduces_and_renders_svg_png(self):
        with tempfile.TemporaryDirectory(prefix="s41_graph_") as directory:
            root = Path(directory)
            run_root = root / "warm"
            run_root.mkdir()
            raw_run(run_root)
            run = reduce_fixture(
                run_root, "warm r0", "C1_GPU_SWITCH_WARM", 0,
                "WARM_HOST_CACHE")
            self.assertEqual(run["completed_output_tokens"], 32)
            self.assertEqual(run["slo_goodput_milli_rps"], 1000)
            self.assertEqual(
                run["model_throughput_milli_tps"],
                {MODELS[0]: 4000, MODELS[1]: 4000},
            )
            self.assertEqual(
                run["gpu_energy_scope"], reducer.COMPLETE_ENERGY_SCOPE)
            summary_path = root / "summary.json"
            write_json(summary_path, summary([run]))
            outputs = graphs.render(summary_path, root / "graphs")
            self.assertEqual(len(outputs), 10)
            for path in outputs:
                self.assertGreater(path.stat().st_size, 1000)
                if path.suffix == ".png":
                    self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))

    def test_cold_stranding_is_visible_and_energy_is_excluded(self):
        with tempfile.TemporaryDirectory(prefix="s41_graph_") as directory:
            root = Path(directory)
            warm_root = root / "warm"
            cold_root = root / "cold"
            warm_root.mkdir()
            cold_root.mkdir()
            raw_run(warm_root)
            raw_run(
                cold_root, completed_count=2,
                energy_end_ns=4_000_000_000, publish=False)
            warm = reduce_fixture(
                warm_root, "warm r0", "C1_GPU_SWITCH_WARM", 0,
                "WARM_HOST_CACHE")
            cold = reduce_fixture(
                cold_root, "cold r0", "C1_GPU_SWITCH_COLD", 0,
                "COLD_NVME")
            self.assertEqual(cold["stranded_request_count"], 2)
            self.assertTrue(cold["publication_gap_censored"])
            self.assertIsNone(cold["maximum_publication_gap_ns"])
            self.assertEqual(
                cold["gpu_energy_scope"], reducer.PREFIX_ENERGY_SCOPE)
            value = summary([warm, cold])
            comparison = graphs.throughput_comparison_svg(value)
            energy = graphs.energy_comparison_svg(value)
            latency = graphs.latency_comparison_svg(value)
            self.assertIn("2 stranded", comparison)
            self.assertIn("partial run excluded", energy)
            self.assertIn("not server-wall or total-system energy", energy)
            self.assertIn("censored", latency)

    def test_completed_run_missing_tail_metric_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s41_graph_") as directory:
            run_root = Path(directory)
            raw_run(run_root)
            run = reduce_fixture(
                run_root, "warm", "C1_GPU_SWITCH_WARM", 0,
                "WARM_HOST_CACHE")
            value = summary([run])
            value["runs"][0]["ttft_p95_ns"] = None
            with self.assertRaisesRegex(
                    graphs.GraphError, "lacks tail latency"):
                graphs.validate_summary(value)

    def test_unknown_energy_scope_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="s41_graph_") as directory:
            run_root = Path(directory)
            raw_run(run_root)
            run = reduce_fixture(
                run_root, "warm", "C1_GPU_SWITCH_WARM", 0,
                "WARM_HOST_CACHE")
            value = summary([copy.deepcopy(run)])
            value["runs"][0]["gpu_energy_scope"] = "TOTAL_SYSTEM_ENERGY"
            with self.assertRaisesRegex(
                    graphs.GraphError, "energy scope/eligibility"):
                graphs.validate_summary(value)

    def test_manifest_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s41_graph_") as directory:
            run_root = Path(directory)
            raw_run(run_root)
            with (run_root / "events.jsonl").open("ab") as stream:
                stream.write(b"{}\n")
            with self.assertRaisesRegex(
                    reducer.ReductionError, "digest mismatch"):
                reduce_fixture(
                    run_root, "warm", "C1_GPU_SWITCH_WARM", 0,
                    "WARM_HOST_CACHE")

    def test_failed_s39_prefix_without_terminal_report_is_supported(self):
        with tempfile.TemporaryDirectory(prefix="s41_graph_") as directory:
            run_root = Path(directory)
            raw_run(run_root, completed_count=2)
            (run_root / "replay.json").unlink()
            events = [
                row for row in reducer.read_jsonl(run_root / "events.jsonl")
                if row.get("kind") != "replay_end"
            ]
            write_jsonl(run_root / "events.jsonl", events)
            (run_root / "SHA256SUMS.txt").unlink()
            write_manifest(run_root)
            run = reduce_fixture(
                run_root, "cold", "C1_GPU_SWITCH_COLD", 0, "COLD_NVME")
            self.assertEqual(run["stranded_request_count"], 2)
            self.assertTrue(run["energy_is_lower_bound"])
            self.assertEqual(run["verdict"], "FAIL_STRANDED_REQUESTS")

    def test_completed_dual_swap_failure_is_labeled_and_mode_bound(self):
        with tempfile.TemporaryDirectory(prefix="s41_graph_") as directory:
            run_root = Path(directory)
            raw_run(run_root)
            (run_root / "replay.json").unlink()
            events = [
                row for row in reducer.read_jsonl(run_root / "events.jsonl")
                if row.get("kind") not in {
                    "model_load_start",
                    "model_ready",
                    "model_published",
                    "switch_started",
                }
            ]
            write_jsonl(run_root / "events.jsonl", events)
            write_json(run_root / "failure.json", {
                "error": "dual replay grew swap",
                "repeat_index": 0,
                "schema": "s41-dual-server-failure-v1",
                "status": "S41_GPU_CPU_DUAL_READY_FAILED",
            })
            (run_root / "SHA256SUMS.txt").unlink()
            write_manifest(run_root)
            run = reduce_fixture(
                run_root, "dual", "C2_GEMMA_GPU_QWEN_CPU", 0,
                "WARM_HOST_CACHE")
            self.assertEqual(run["verdict"], "RESOURCE_FAIL_SWAP_GROWTH")
            value = summary([run])
            comparison = graphs.throughput_comparison_svg(value)
            self.assertIn("swap gate failed", comparison)
            timeline = graphs.timeline_svg(value, run)
            self.assertIn(
                "INVALID CONTROL: zero-swap gate failed", timeline)
            with self.assertRaisesRegex(
                    reducer.ReductionError, "lacks terminal report"):
                reduce_fixture(
                    run_root, "wrong mode", "C1_GPU_SWITCH_WARM", 0,
                    "WARM_HOST_CACHE")


if __name__ == "__main__":
    unittest.main()
