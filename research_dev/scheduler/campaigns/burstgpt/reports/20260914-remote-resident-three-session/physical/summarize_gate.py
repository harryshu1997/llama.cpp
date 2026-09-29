"""Audit bounded relocation artifacts without modifying their verdict or accounting."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    result = read(run / "REMOTE_RESIDENT_GATE.json")
    preparation = read(run / "phone/PREPARATION.json")
    ready = read(run / "phone/READY.json")
    terminal = read(run / "phone/TERMINAL.json")
    assignment = read(run / "REMOTE_OWNER_ASSIGNMENT.json")
    native, = [row for row in terminal["phone_receipts"] if "terminal" in row]
    expected = {row["session_id"]: row for row in assignment["assignments"]}
    physical = {row["session_id"]: row for row in ready["phone_shards"]}
    assert set(expected) == set(physical) == set(ready["load_count_by_session"])
    assert all(value == 1 for value in ready["load_count_by_session"].values())
    requests, tokens_seen, tickets = [], set(), set()
    for full, reduced in zip(result["arms"]["full"]["requests"],
                             result["arms"]["reduced"]["requests"], strict=True):
        command = read(run / "phone" / (reduced["request_id"] + "-command.json"))
        proof = reduced["proof"]
        tokens = {row["token"] for row in command["leases"]}
        assert tokens and not tokens & tokens_seen
        tokens_seen.update(tokens)
        assert proof["ticket_id"] == command["ticket_id"] not in tickets
        tickets.add(command["ticket_id"])
        assert proof["artifact_sha256"] == result["artifact_sha256"]
        assert proof["operator_plan_sha256"] == command["operator_plan_sha256"]
        owners = {row["session_id"]: row for row in
                  command["execution_contract"]["remote_resident_ffn"]["sessions"]}
        calls = {row["session_id"]: row for row in proof["phone_calls_by_session"]}
        assert set(owners) == set(calls) == set(expected)
        for sid, owner in owners.items():
            for field in ("endpoint", "layer_mask", "session_generation",
                          "resident_geometry_sha256", "operator_plan_sha256"):
                assert owner[field] == physical[sid][field] == calls[sid][field], (sid, field)
            assert owner["session_generation"] > 0 and calls[sid]["calls"] > 0
            assert owner["shard_sha256"] == assignment["shards"][sid]["shard_sha256"]
            assert physical[sid]["artifact_sha256"] == expected[sid]["artifact_sha256"]
        requests.append({
            "request_index": reduced["request_index"], "output_tokens": reduced["output_tokens"],
            "full_duration_us": full["duration_us"], "reduced_duration_us": reduced["duration_us"],
            "phone_calls": proof["phone_call_count"], "ticket_id": command["ticket_id"],
            "fresh_lease_tokens": sorted(tokens), "session_proofs": proof["phone_calls_by_session"],
            "tokens_identical": full["tokens"] == reduced["tokens"],
            "includes_desktop_transition": bool(command["transitions"]),
        })
    sessions = {}
    origin = preparation["plan"]["stages"][0]["started_at_us"]
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
            intervals[name + "_us"] = (None if start not in phases or end not in phases else
                phases[end]["monotonic_us"] - phases[start]["monotonic_us"])
        stage, = [row for row in preparation["plan"]["stages"] if row["selected_session_id"] == sid]
        sessions[sid] = {"shard": shard, "phase_intervals": intervals, "events": events,
            "scheduler_load_to_publication_us": stage["ready_at_us"] - stage["started_at_us"],
            "publication_since_first_stage_start_us": stage["ready_at_us"] - origin,
            "load_count": ready["load_count_by_session"][sid]}
    assert native["terminal"]["status"] == 0 and native["terminal"]["reset_recoveries"] == 0
    summary = {
        "schema": "s42-multi-session-relocation-summary-v1", "status": result["status"],
        "gates": result["gates"], "requests": requests, "sessions": sessions,
        "assignment_sha256": assignment["assignment_sha256"],
        "desktop_parent_placement_sha256": assignment["desktop_parent_placement_sha256"],
        "phone_preparation_total_us": (preparation["finished_ns"] - preparation["started_ns"]) // 1000,
        "preparation_energy": preparation["energy"], "preparation_receipts": preparation["receipts"],
        "phone_call_count_requests": sum(row["phone_calls"] for row in requests),
        "phone_call_count_terminal_including_warmup": native["terminal"]["requests"],
        "reset_recoveries": native["terminal"]["reset_recoveries"],
        "terminal_status": native["terminal"]["status"],
        "usb_restoration": terminal["usb_restoration"], "weight_sources": ready["weight_sources"],
        "native_hashes": native["launch"]["remote_hashes"],
        "server_binary_sha256": result["runtime_binary_sha256"],
        "server_libraries_sha256": result["runtime_libraries_sha256"],
        "gate_result_sha256": digest(run / "REMOTE_RESIDENT_GATE.json"),
        "transport_identity": native["launch"]["qualification_identity_sha256"],
        "limits": ["Memory/execution gate, not a larger-context or end-to-end savings proof",
                   "First reduced request includes desktop loading; later request intervals do not",
                   "Phone power assumed; prep server energy measured separately",
                   "Publication offsets use the first scheduler stage start, not a cross-device clock subtraction",
                   "No in-flight phone owner loss or recovery injection"],
    }
    with args.output.open("x") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: summary[key] for key in
                      ("status", "phone_call_count_requests", "phone_preparation_total_us", "gate_result_sha256")}))


if __name__ == "__main__":
    main()
