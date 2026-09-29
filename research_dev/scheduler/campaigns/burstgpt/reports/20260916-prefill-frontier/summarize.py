"""Summarize the bounded document gates without claiming matched fleet savings."""

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
REMOTE = {
    "physical-v2": "/mnt/storage/s42-prefill-affinity-20260916-v2-nuxeVS",
    "physical-v3": "/mnt/storage/s42-context16384-20260916-v3-86SDY3",
    "physical-v4": "/mnt/storage/s42-context32768-20260916-v4-9bEpE6",
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(name):
    root = ROOT / name
    gate_path = root / "gate-run-v1/REMOTE_RESIDENT_GATE.json"
    audit = read(root / "AUDIT.json")
    gate = read(gate_path)
    assert audit["status"] == gate["status"] == "PASS"
    assert audit["gate_result_sha256"] == digest(gate_path).removeprefix("sha256:")
    assert read(root / "run-launch.json")["postflight"] == "PASS"
    assert audit["terminal_status"] == audit["reset_recoveries"] == 0
    request, = audit["requests"]
    assert digest(root / "PROMPT.txt") == audit["prompt_sha256"]
    prompt = (root / "PROMPT.txt").read_text()
    expected_key = re.search(r"archive key is ([A-Z0-9-]+)\.", prompt).group(1)
    samples = read(root / "gate-run-v1/HOST_SAMPLES.json")["rows"]
    arms = {}
    for name_in_gate in ("full", "reduced"):
        arm = gate["arms"][name_in_gate]
        measurement, = arm["requests"]
        timing = measurement["context_completion"]["terminal"]["timings"]
        assert timing["prompt_n"] == audit["input_tokens"]
        assert timing["predicted_n"] == 64
        assert timing["cache_n"] == 0
        memory = arm["memory"]
        live = [row for row in samples if any(
            process["pid"] == memory["pid"]
            for process in row.get("host_activity", {}).get("processes", []))]
        assert live
        arms[name_in_gate] = {
            "output_diagnostic": {
                "required_archive_key": expected_key,
                "archive_key_present": expected_key in measurement["context_completion"]["output_text"],
                "output_text": measurement["context_completion"]["output_text"],
                "gate_semantic_sanity": measurement["context_completion"]["output_quality"],
                "task_accuracy_qualified": False,
            },
            "prefill_s": timing["prompt_ms"] / 1000,
            "decode_s": timing["predicted_ms"] / 1000,
            "request_interval_s": measurement["duration_us"] / 1e6,
            "request_interval_includes_desktop_launch": name_in_gate == "reduced",
            "server_energy_j": measurement["energy"]["server_compute_device_energy_j"],
            "cpu_energy_j": measurement["energy"]["cpu_package_energy_j"],
            "gpu_energy_j": measurement["energy"]["gpu_board_energy_j"],
            "rss_snapshot_bytes": memory["vm_rss_bytes"],
            "rss_high_water_bytes": memory["vm_hwm_bytes"],
            "process_vram_snapshot_bytes": memory["process_vram_bytes"],
            "observed_device_vram_peak_bytes": max(row["gpu"]["memory_used_bytes"] for row in live),
            "kv_buffer_lines": [line for line in memory["kv_lines"] if "buffer size" in line],
        }
    sessions = {key: {
        "generation": value["shard"]["session_generation"],
        "load_count": value["load_count"],
        "resident_bytes": value["shard"]["resident_bytes"],
        "phone_load_phases_us": value["phase_intervals"],
        "host_ready_since_preparation_start_s": value["scheduler_receipt_ready_since_first_stage_start_us"] / 1e6,
        "host_verified_since_preparation_start_s": value["scheduler_verified_since_first_stage_start_us"] / 1e6,
        "request_calls": next(proof["calls"] for proof in request["session_proofs"]
                              if proof["session_id"] == key),
    } for key, value in audit["sessions"].items()}
    assert all(value["generation"] == value["load_count"] == 1 for value in sessions.values())
    prep_s = audit["phone_preparation_total_us"] / 1e6
    prep_energy = audit["preparation_energy"]["server_compute_device_energy_j"]
    sensitivity = [{
        "assumed_phone_active_power_w": power,
        "full_request_server_plus_idle_phone_j": arms["full"]["server_energy_j"] + 0.875 * arms["full"]["request_interval_s"],
        "reduced_request_server_plus_active_phone_j": arms["reduced"]["server_energy_j"] + power * arms["reduced"]["request_interval_s"],
        "separate_phone_preparation_server_plus_active_phone_j": prep_energy + power * prep_s,
    } for power in (3, 4.5, 6)]
    return {
        "artifact_directory": REMOTE[name], "context_size": audit["context_size"],
        "input_tokens": audit["input_tokens"], "output_tokens": 64,
        "status": audit["status"], "arms": arms, "sessions": sessions,
        "phone_preparation_s": prep_s, "phone_preparation_server_energy_j": prep_energy,
        "phone_calls": audit["phone_call_count_requests"],
        "phone_prefill_calls": request["phone_prefill_calls"],
        "phone_decode_calls": request["phone_decode_calls"],
        "omission": audit["gates"]["B_memory"]["proof"],
        "omitted_pages_vma_overlap_bytes": audit["gates"]["B_memory"]["omitted_pages_vma_overlap_bytes"],
        "phone_weight_bytes": sum(row["resident_bytes"] for row in sessions.values()),
        "phone_workspace_reservation_bytes": gate["accounting"]["phone_workspace_bytes"],
        "token_identical": request["tokens_identical"],
        "power_sensitivity_diagnostic_only": sensitivity,
        "hashes": {"gate": digest(gate_path), "audit": digest(root / "AUDIT.json"),
                   "router_log": digest(root / "router.log"), "prompt": audit["prompt_sha256"],
                   "assignment": audit["assignment_sha256"],
                   "desktop_parent": audit["desktop_parent_placement_sha256"],
                   "native": audit["native_hashes"], "server": audit["server_binary_sha256"]},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "s42-bounded-context-summary-v2", "max_tested_context": 32768,
        "runs": [summarize(name) for name in REMOTE],
        "limitations": [
            "Both layouts pass the same contexts; no higher maximum-context ceiling is established.",
            "RSS snapshots and high-water marks are separate; device VRAM peak includes other processes.",
            "Phone workspace is reserved, not a measured phone peak.",
            "Full request excludes desktop load; reduced request includes it. Cleanup is not included.",
            "Power sensitivity charges active power over each entire reduced interval, not measured phone power.",
            "Full request phone idle power is assumed 0.875 W. Phase figures are diagnostic, not matched savings.",
            "Output semantic-sanity is not long-document task-accuracy qualification.",
            "At 32k both arms omit the required archive key and emit repetitive channel markers; useful 32k output fails.",
            "One request per arm and context; performance variability is not characterized.",
        ],
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "runs": len(result["runs"]),
                      "summary_sha256": digest(args.output)}))
