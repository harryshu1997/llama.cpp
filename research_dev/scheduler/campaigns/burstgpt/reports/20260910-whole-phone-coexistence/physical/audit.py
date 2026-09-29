"""Persist the bounded device gate's measurements and post-cleanup state."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


SOURCES = (
    "adapters/android_llama_server.py", "adapters/heterogeneous_rig.py", "adapters/probes.py",
    "adapters/catalog_materialization.py", "adapters/native/android_ncm_adb_control.sh",
    "configuration/rig.py", "campaigns/burstgpt/arguments.py", "campaigns/burstgpt/catalog.py",
    "campaigns/burstgpt/launch.py", "campaigns/burstgpt/runner.py", "tests/test_android_ncm_control.py",
    "_internal/capability_contracts/executors.py",
)


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    attempt = root / "whole-attempt-4"
    read = lambda name: json.loads((attempt / name).read_text())
    result, cleanup, execution = read("RESULT.json"), read("CLEANUP.json"), read("WHOLE_EXECUTION.json")
    assert result["status"] == cleanup["status"] == "PASS"
    assert not execution["recoveries"]
    sources = root / "SOURCE_GATE"
    sources.mkdir()
    source_hashes = {}
    for name in SOURCES:
        path = args.deployment / "research_dev/scheduler" / name
        target = sources / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[name] = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
    write_new(root / "SOURCE_GATE.json", source_hashes)
    commands = []
    for command in (
        ["/usr/bin/adb", "-P", "5037", "-s", "3C15AU002CL00000", "shell", "ps -A"],
        ["/usr/bin/adb", "-P", "5037", "-s", "3C15AU002CL00000", "shell",
         "getprop sys.usb.config; uname -r; cat /sys/kernel/btf/vmlinux | sha256sum"],
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader"],
        ["ps", "-p", "6871", "-o", "pid,comm,args"],
    ):
        started = time.time_ns()
        value = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
        commands.append({"command": command, "started_epoch_ns": started, "finished_epoch_ns": time.time_ns(),
                         "returncode": value.returncode, "stdout": value.stdout, "stderr": value.stderr})
    remaining = [line for line in commands[0]["stdout"].splitlines() if "llama-" in line or "ffn-split" in line]
    assert not remaining, remaining
    write_new(root / "POST_GATE_DEVICE_AUDIT.json", commands)
    preparation = read("PREPARATION.json")["plan"]["stages"][0]
    samples = read("WHOLE_ACTIVE_TELEMETRY.json") + read("WHOLE_TELEMETRY.json")
    allocations = [row for sample in samples for row in sample.get("resident_allocations", []) if row.get("valid")]
    terminal = next(row for row in cleanup["direct_phone_receipts"] if "launch" in row)
    weight, = terminal["launch"]["weight_sources"]
    summary = {
        "status": "PASS", "scope": "Control, telemetry, whole-phone inference with one idle READY HTP shard",
        "simultaneous_htp_compute_proven": False, "whole_gpu_peak_memory_qualified": False,
        "output_tokens": 32, "input_tokens": 915,
        "service_time_us": execution["finished_us"] - execution["started_us"],
        "htp_load_time_us": weight["load_finished_epoch_us"] - weight["load_started_epoch_us"],
        "first_session_ready_from_preload_epoch_us": preparation["verified_at_us"],
        "htp_session": weight["session_id"], "htp_generation": weight["session_generation"],
        "htp_resident_bytes": result["session_state"]["phone_shards"][0]["resident_bytes"],
        "shard_file_bytes": weight["bytes_loaded"], "shard_sha256": weight["source_sha256"],
        "shard_index_sha256": weight["index_sha256"], "parent_sha256": weight["parent_artifact_sha256"],
        "shard_loads": sum(result["session_state"]["load_count_by_session"].values()),
        "reloads_during_whole_request": 0, "usb_reset_recoveries": terminal["terminal"]["reset_recoveries"],
        "htp_calls": sum(row["calls"] for row in terminal["terminal"]["session_proofs"]),
        "allocation_samples": len(allocations),
        "allocation_sample_span_us": (max(row["captured_at_ns"] for row in allocations)
                                      - min(row["captured_at_ns"] for row in allocations)) // 1000,
        "maximum_allocation_age_us": max(row["age_us"] for row in allocations),
        "pss_min_bytes": min(row["allocated_bytes"] for row in allocations),
        "pss_max_bytes": max(row["allocated_bytes"] for row in allocations),
        "whole_endpoint_generations": sorted({row["generation"] for row in allocations}),
        "whole_process_identity": allocations[-1]["process_identity"],
        "physical_execution_proof": execution["physical_execution_proof"],
        "execution_energy": execution["energy"], "cleanup": "PASS", "remaining_owned_phone_workers": remaining,
        "notes": ["PSS is partial process residency, not all OpenCL allocations or a qualified peak.",
                  "The 3 GB whole-service peak is a conservative declaration, not a measurement.",
                  "No baseline or savings comparison; no trace was run.",
                  "The whole endpoint's current capability still reserves the shared HTP resource."]
    }
    write_new(root / "SUMMARY.json", summary)
    hashes = {str(path.relative_to(root)): "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(root.rglob("*")) if path.is_file()}
    write_new(root / "ARTIFACT_HASHES.json", hashes)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
