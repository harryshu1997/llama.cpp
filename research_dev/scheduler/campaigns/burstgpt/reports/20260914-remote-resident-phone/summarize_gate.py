"""Summarize immutable relocation evidence without changing the gate verdict."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
run = ROOT / "gate-run-v4"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


result = read(run / "REMOTE_RESIDENT_GATE.json")
preparation = read(run / "phone/PREPARATION.json")
ready = read(run / "phone/READY.json")
terminal = read(run / "phone/TERMINAL.json")
native, = [row for row in terminal["phone_receipts"] if "terminal" in row]
requests = []
lease_tokens = set()
for full, reduced in zip(result["arms"]["full"]["requests"], result["arms"]["reduced"]["requests"], strict=True):
    command = read(run / "phone" / (reduced["request_id"] + "-command.json"))
    proof = reduced["proof"]
    current_leases = {row["token"] for row in command["leases"]}
    assert current_leases and not current_leases & lease_tokens
    lease_tokens.update(current_leases)
    assert proof["ticket_id"] == command["ticket_id"]
    assert proof["artifact_sha256"] == result["artifact_sha256"]
    assert proof["operator_plan_sha256"] == command["operator_plan_sha256"]
    assert proof["phone_call_count"] > 0
    owners = command["execution_contract"]["remote_resident_ffn"]["sessions"]
    for owner in owners:
        shard, = [row for row in ready["phone_shards"] if row["session_id"] == owner["session_id"]]
        assert owner["session_generation"] == shard["session_generation"] > 0
        assert owner["resident_geometry_sha256"] == shard["resident_geometry_sha256"]
        assert owner["operator_plan_sha256"] == shard["operator_plan_sha256"]
    requests.append({
        "request_index": reduced["request_index"], "output_tokens": reduced["output_tokens"],
        "full_duration_us": full["duration_us"], "reduced_duration_us": reduced["duration_us"],
        "reduced_duration_scope": "adapter execution including any desktop transition; phone preload separate",
        "phone_calls": proof["phone_call_count"], "ticket_id": command["ticket_id"],
        "fresh_lease_tokens": sorted(current_leases), "session_proofs": proof["phone_calls_by_session"],
        "tokens_identical": full["tokens"] == reduced["tokens"],
    })
sessions = {}
for shard in ready["phone_shards"]:
    sid = shard["session_id"]
    events = [row for row in terminal["session_events"] if row["session_id"] == sid]
    phases = {row["phase"]: row for row in events}
    intervals = {}
    for name, start, end in (
        ("weight_read", "WEIGHT_READ_BEGIN", "WEIGHT_READ_READY"),
        ("htp_init", "HTP_INIT_BEGIN", "HTP_INIT_READY"),
        ("weight_upload", "WEIGHT_UPLOAD_BEGIN", "WEIGHT_UPLOAD_READY"),
        ("load_authorized_to_ready", "LOAD_AUTHORIZED", "READY"),
    ):
        intervals[name + "_us"] = phases[end]["monotonic_us"] - phases[start]["monotonic_us"]
    sessions[sid] = {"shard": shard, "phase_intervals": intervals, "events": events,
                     "load_count": ready["load_count_by_session"][sid]}
summary = {
    "schema": "s42-remote-phone-relocation-summary-v1", "status": result["status"],
    "gates": result["gates"], "requests": requests, "sessions": sessions,
    "phone_preparation_total_us": (preparation["finished_ns"] - preparation["started_ns"]) // 1000,
    "preparation_energy": preparation["energy"],
    "preparation_receipts": preparation["receipts"],
    "phone_call_count_requests": sum(row["phone_calls"] for row in requests),
    "phone_call_count_terminal_including_warmup": native["terminal"]["requests"],
    "reset_recoveries": native["terminal"]["reset_recoveries"],
    "terminal_status": native["terminal"]["status"],
    "usb_restoration": terminal["usb_restoration"],
    "weight_sources": native["launch"]["weight_sources"],
    "native_hashes": native["launch"]["remote_hashes"],
    "server_binary_sha256": result["runtime_binary_sha256"],
    "server_libraries_sha256": result["runtime_libraries_sha256"],
    "source_archive_sha256": digest(ROOT / "source-gate-run-v4.tar.gz"),
    "gate_result_sha256": digest(run / "REMOTE_RESIDENT_GATE.json"),
    "transport_identity": native["launch"]["qualification_identity_sha256"],
    "limits": ["Exact token agreement failed for request 50 at token 37; verdict is unchanged",
               "No per-layer error or logit-margin measurement; numerical cause is not established",
               "Host RAM relocation proved, not additional usable KV capacity or energy savings",
               "In-flight HTP cancellation and phone-owner recovery were not tested"],
}
with (ROOT / "GATE_V4_SUMMARY.json").open("x") as stream:
    json.dump(summary, stream, sort_keys=True, indent=2)
    stream.write("\n")
print(json.dumps({"status": summary["status"], "request_phone_calls": summary["phone_call_count_requests"],
                  "preparation_us": summary["phone_preparation_total_us"],
                  "sessions": {key: value["phase_intervals"] for key, value in sessions.items()},
                  "gate_result_sha256": summary["gate_result_sha256"]}, sort_keys=True))
