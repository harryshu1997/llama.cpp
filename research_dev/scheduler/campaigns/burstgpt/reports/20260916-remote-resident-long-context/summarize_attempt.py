"""Audit the failed long-context attempt without counting warmup as completion."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.attempt / "gate-run-v1"
    result = read(run / "REMOTE_RESIDENT_GATE.json")
    document = read(run / "DOCUMENT_REQUESTS.json")
    preparation = read(run / "phone/PREPARATION.json")
    ready = read(run / "phone/READY.json")
    terminal = read(run / "phone/TERMINAL.json")
    native, = [row["terminal"] for row in terminal["phone_receipts"] if "terminal" in row]
    full, = result["arms"]["full"]["requests"]
    completion = full["context_completion"]
    assert result["status"] == "FAILED" and not terminal["execution_proofs"]
    assert full["error"] is None and completion["terminal"]["truncated"] is False
    assert completion["terminal"]["timings"]["prompt_n"] == len(document["rows"][0]["prompt_tokens"])
    assert completion["terminal"]["timings"]["predicted_n"] == len(full["tokens"]) == 64
    origin = preparation["plan"]["stages"][0]["started_at_us"]
    sessions = []
    for stage in preparation["plan"]["stages"]:
        sid = stage["selected_session_id"]
        physical, = [row for row in ready["phone_shards"] if row["session_id"] == sid]
        proof, = [row for row in native["session_proofs"] if row["session_id"] == sid]
        for key in ("artifact_sha256", "resident_geometry_sha256", "operator_plan_sha256", "session_generation"):
            assert physical[key] == proof[key]
        sessions.append({
            "session_id": sid, "generation": physical["session_generation"],
            "resident_bytes": physical["resident_bytes"], "loads": ready["load_count_by_session"][sid],
            "stage_started_us": stage["started_at_us"], "receipt_ready_us": stage["ready_at_us"],
            "scheduler_verified_us": stage["verified_at_us"],
            "stage_to_verified_s": (stage["verified_at_us"] - stage["started_at_us"]) / 1e6,
            "first_stage_to_verified_s": (stage["verified_at_us"] - origin) / 1e6,
            "warmup_calls": proof["calls"],
        })
    prep_s = (preparation["finished_ns"] - preparation["started_ns"]) / 1e9
    summary = {
        "schema": "s42-long-context-attempt-audit-v1", "status": "FAILED_NATIVE_PREFILL",
        "context_size": document["context_size"], "prompt_sha256": document["prompt_sha256"],
        "desktop_request": {"input_tokens": full["input_tokens"], "output_tokens": full["output_tokens"],
            "native_timings": completion["terminal"]["timings"], "duration_us": full["duration_us"],
            "truncated": False, "energy": full["energy"], "output_text": completion["output_text"]},
        "sessions": sessions, "phone_preparation_s": prep_s,
        "phone_preparation_server_energy": preparation["energy"],
        "phone_preparation_estimated_fleet_j": {str(power):
            preparation["energy"]["server_compute_device_energy_j"] + power * prep_s for power in (3, 4.5, 6)},
        "long_document_phone_output_tokens": 0, "successful_long_document_execution_proofs": 0,
        "warmup_calls_total": native["requests"], "terminal_status": native["status"],
        "reset_recoveries": native["reset_recoveries"], "usb_restoration": terminal["usb_restoration"],
        "postflight": read(args.attempt / "run-launch.json"),
        "native_error": "FFN split runtime context differs from tensor rows",
        "observed_server_batch": 2048, "configured_native_microbatch": 512,
        "diagnosis": "Runtime request rows are assigned at server batch scope, then llama_decode splits into smaller tensor microbatches",
        "limitations": ["No successful phone long-context completion or savings claim",
                        "Warmup calls are not long-document calls",
                        "No new maximum-context capacity proof", "Phone preparation power is assumed"],
        "hashes": {name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in (
            "REMOTE_RESIDENT_GATE.json", "phone/PREPARATION.json", "phone/READY.json", "phone/TERMINAL.json")},
    }
    with args.output.open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: summary[key] for key in ("status", "sessions", "phone_preparation_s")}))


if __name__ == "__main__":
    main()
