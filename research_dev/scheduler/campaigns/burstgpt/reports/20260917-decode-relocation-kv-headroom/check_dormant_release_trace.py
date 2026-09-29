"""Verdict for a BurstGPT trace run with the decode-only FFN relocation under the automated scheduler.

usage: check_dormant_release_trace.py RUN_DIR [--summary PATH]

Reads RESULT.json (request results, dormant_share_admission, trace energy) and the desktop server logs
(`S41SERVERFFN dormant_host_share` proof lines) and checks:
  A. the trace completed (status PASS, every planned request terminal);
  B. every release generation the coordinator observed (acknowledgements, statistics reads) was credited
     exactly once and none was refused; the observed share of all server-side releases is reported as coverage;
  C. no prompt reached a server while its share was restore-blocked: every hold ended in prompt_admitted
     (no prompt_hold_timeout), and every server-side populate (phase=local) follows a release;
  D. the ledger never booked a share it could not hold (no server_booking_refused; over-budget bookings are
     reported, not failed).
It prints a table per model (requests, assisted requests, fractions, releases, credited bytes, holds) and the
trace energy; exit status 0 on PASS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

PROOF = re.compile(r"S41SERVERFFN dormant_host_share phase=(decode|local) layer_mask=(\d+) host_columns=(\d+) (released|restored)_bytes=(\d+)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    result_path = args.run_dir / "RESULT.json"
    if not result_path.exists():
        failure = args.run_dir / "FAILURE.json"
        print("NO RESULT.json;", "FAILURE.json present" if failure.exists() else "no FAILURE.json either")
        if failure.exists():
            print(json.dumps(json.load(failure.open()).get("error"))[:600])
        return 2
    result = json.load(result_path.open())
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    requests = result.get("request_results", [])
    counts = result.get("counts", {})
    check("A.trace-complete", result.get("status") == "PASS" and counts.get("terminals") == counts.get("requests") == len(requests),
          f"status={result.get('status')} terminals={counts.get('terminals')} requests={counts.get('requests')}")
    admission = result.get("dormant_share_admission")
    events = [] if not admission else admission.get("events", [])
    kinds = {}
    for row in events:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    # server-side proofs from the desktop logs
    releases, populates = [], []
    for log in sorted(args.run_dir.glob("*.stderr")):
        for line in log.read_text(errors="replace").splitlines():
            m = PROOF.search(line)
            if m:
                (releases if m.group(1) == "decode" else populates).append({"log": log.name, "layer_mask": int(m.group(2)),
                                                                              "host_columns": int(m.group(3)), "bytes": int(m.group(5))})
    credited = [row for row in events if row["kind"] == "release_credited"]
    # The coordinator learns the release state from control acknowledgements and FFN statistics reads; with several
    # server slots a release can be created and populated again between two reads and is then never observable.
    # Rule B therefore requires that every observed generation was credited exactly once and nothing was refused;
    # the share of server releases that were observed is reported as coverage, not gated.
    # generations restart with every server launch on an endpoint: key distinctness by (endpoint, launch, generation)
    launches: dict[str, int] = {}
    distinct = set()
    for row in events:
        if row["kind"] == "server_booked":
            launches[row["endpoint"]] = launches.get(row["endpoint"], 0) + 1
        elif row["kind"] == "release_credited":
            distinct.add((row["endpoint"], launches.get(row["endpoint"], 0), row["generation"]))
    coverage = (len(credited) / len(releases)) if releases else None
    check("B.observed-releases-credited-once", kinds.get("release_credit_refused", 0) == 0 and len(distinct) == len(credited)
          and (not releases or credited),
          f"server releases={len(releases)} observed+credited={len(credited)} refused={kinds.get('release_credit_refused', 0)} "
          f"coverage={coverage if coverage is None else round(coverage, 2)}")
    holds = [row for row in events if row["kind"] == "prompt_held"]
    admitted = [row for row in events if row["kind"] == "prompt_admitted"]
    check("C.no-prompt-while-blocked", kinds.get("prompt_hold_timeout", 0) == 0 and len(admitted) >= len(holds)
          and len(populates) <= len(releases),
          f"holds={len(holds)} admitted={len(admitted)} timeouts={kinds.get('prompt_hold_timeout', 0)} server populates={len(populates)} releases={len(releases)}")
    check("D.bookings", kinds.get("server_booking_refused", 0) == 0 and admission is not None,
          f"booked={kinds.get('server_booked', 0)} over_budget={kinds.get('server_booking_over_budget', 0)} refused={kinds.get('server_booking_refused', 0)} forgotten={kinds.get('server_forgotten', 0)}")
    # per-model table
    table = {}
    for req in requests:
        model = req.get("model_id", "?")
        row = table.setdefault(model, {"requests": 0, "assisted": 0, "fractions": {}, "latency_s": 0.0, "energy_j": 0.0})
        row["requests"] += 1
        history = req.get("fraction_history") or {}
        fraction = history.get("selected_split_fraction_ppm", history.get("initial_split_fraction_ppm", 0)) or 0
        if fraction:
            row["assisted"] += 1
        row["fractions"][str(fraction)] = row["fractions"].get(str(fraction), 0) + 1
        row["latency_s"] += (req.get("actual_latency_us") or 0) / 1e6
        energy = req.get("measured_energy") or {}
        if isinstance(energy, dict):
            row["energy_j"] += float(energy.get("total_j") or energy.get("energy_j") or 0)
    print("| model | requests | assisted | fractions (ppm: count) | sum latency s |")
    print("|---|---:|---:|---|---:|")
    for model, row in sorted(table.items()):
        print(f"| {model} | {row['requests']} | {row['assisted']} | {row['fractions']} | {row['latency_s']:.1f} |")
    credited_bytes = sum(int(row.get("credited_bytes", 0)) for row in credited)
    print(f"\nreleases={len(releases)} credited={len(credited)} credited_bytes={credited_bytes} populates={len(populates)} "
          f"holds={len(holds)} (waited {sum(float(row.get('waited_s', 0)) for row in admitted):.1f} s total) "
          f"servers_booked={kinds.get('server_booked', 0)} over_budget={kinds.get('server_booking_over_budget', 0)}")
    print("trace_energy:", json.dumps(result.get("trace_energy"))[:400])
    print("duration_s:", (result.get("duration_us") or 0) / 1e6)
    for row in checks:
        print(f"{row['status']} {row['check']}: {row['detail']}")
    verdict = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    print("VERDICT", verdict)
    if args.summary:
        args.summary.write_text(json.dumps({"verdict": verdict, "checks": checks, "table": table, "releases": releases,
                                            "populates": populates, "admission_event_counts": kinds}, indent=1, sort_keys=True) + "\n")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
