"""Summarize a decode-only relocation gate directory: timings, proofs, residency and memory per arm."""

import argparse
import json
from pathlib import Path
import statistics


def load_jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = json.loads((args.gate / "RESULT.json").read_text())
    spec = json.loads((args.gate / "SPEC.json").read_text())
    memory = load_jsonl(args.gate / "SERVER_MEMORY.jsonl")
    expected = result["expected_release_bytes"]
    arms = {}
    for run in result["runs"]:
        arms.setdefault(run["arm"], []).append(run)

    def phase_memory(run, start_ns, end_ns, key):
        values = [m[key] for m in memory if start_ns <= m["observed_ns"] <= end_ns and m.get(key) is not None]
        return (min(values), max(values)) if values else (None, None)

    summary = {"expected_release_bytes": expected, "share_range_count": result["share_range_count"],
               "sessions": result.get("sessions"), "prompt_tokens": spec["prompt"]["input_tokens"],
               "arms": {}, "proof_lines": result.get("proof_lines"), "recovery": result.get("recovery")}
    print(f"expected page-exact release {expected/1e9:.3f} GB over {result['share_range_count']} ranges; "
          f"prompt {spec['prompt']['input_tokens']} tokens; sessions {result.get('sessions')}")
    print(f"{'arm':>16} {'req':>3} {'prompt ms':>10} {'decode ms/tok':>14} {'file RSS prefill max GB':>24} {'file RSS decode min GB':>23} {'MemAvail decode max GB':>23} {'share resident decode':>22} {'phone calls':>11} {'err':>4}")
    for arm, runs in arms.items():
        rows = []
        for run in runs:
            first = run.get("first_token_ns") or run["finished_ns"]
            pre_rss = phase_memory(run, run["started_ns"], first, "model_file_rss")[1]
            dec_rss = phase_memory(run, first, run["finished_ns"], "model_file_rss")[0]
            avail = None
            values = [m["meminfo"]["MemAvailable"] for m in memory if first <= m["observed_ns"] <= run["finished_ns"] and m.get("meminfo")]
            if values:
                avail = max(values)
            def meminfo_range(start_ns, end_ns, key):
                vals = [m["meminfo"][key] for m in memory if start_ns <= m["observed_ns"] <= end_ns and m.get("meminfo") and key in m["meminfo"]]
                return (min(vals), max(vals)) if vals else (None, None)
            def cgroup_range(start_ns, end_ns):
                rows_ = [m["cgroup"] for m in memory if start_ns <= m["observed_ns"] <= end_ns and isinstance(m.get("cgroup"), dict) and "memory.current" in m["cgroup"]]
                if not rows_:
                    return None
                current = [r["memory.current"] for r in rows_]
                events = [r.get("memory.events", {}) for r in rows_]
                return {"memory_current_max": max(current), "memory_current_min": min(current),
                        "memory_max": rows_[0].get("memory.max"),
                        "max_events_delta": (events[-1].get("max", 0) - events[0].get("max", 0)) if events else None,
                        "oom_events_delta": (events[-1].get("oom", 0) - events[0].get("oom", 0)) if events else None,
                        "pgmajfault_delta": ((rows_[-1].get("memory.stat", {}).get("pgmajfault", 0) - rows_[0].get("memory.stat", {}).get("pgmajfault", 0)) if rows_[0].get("memory.stat") else None)}
            system = {"cached_prefill": meminfo_range(run["started_ns"], first, "Cached"),
                      "cached_decode": meminfo_range(first, run["finished_ns"], "Cached"),
                      "memfree_prefill": meminfo_range(run["started_ns"], first, "MemFree"),
                      "memfree_decode": meminfo_range(first, run["finished_ns"], "MemFree"),
                      "cgroup_prefill": cgroup_range(run["started_ns"], first),
                      "cgroup_decode": cgroup_range(first, run["finished_ns"])}
            decode_samples = [s for s in run.get("residency_samples", []) if ":decode" in s.get("tag", "")]
            prefill_samples = [s for s in run.get("residency_samples", []) if ":prefill" in s.get("tag", "")]
            row = {"request_id": run["request_id"], "prompt_ms": run["prompt_ms"], "decode_ms_per_token": run["decode_ms_per_token"],
                   "output_tokens": run["output_tokens"], "error": run.get("error"),
                   "model_file_rss_prefill_max": pre_rss, "model_file_rss_decode_min": dec_rss,
                   "mem_available_decode_max": avail,
                   "share_resident_fraction_decode": [s["resident_fraction"] for s in decode_samples],
                   "share_resident_fraction_prefill": [s["resident_fraction"] for s in prefill_samples],
                   "phone_calls_mid_decode": run.get("phone_calls_mid_decode"),
                   "energy_j": (run.get("energy") or {}).get("server_compute_device_energy_j"),
                   "system_memory": system}
            rows.append(row)
            cg = system["cgroup_decode"]
            if cg:
                print(f"{'':>16} cgroup decode: current {cg['memory_current_min']/1e9:.2f}-{cg['memory_current_max']/1e9:.2f} GB of max "
                      f"{(cg['memory_max']/1e9 if isinstance(cg['memory_max'], int) else cg['memory_max'])} | max-events +{cg['max_events_delta']} "
                      f"oom +{cg['oom_events_delta']} majfault +{cg['pgmajfault_delta']}")
            cf, cd = system["cached_prefill"], system["cached_decode"]
            ff, fd_ = system["memfree_prefill"], system["memfree_decode"]
            if cf[0] is not None and cd[0] is not None:
                print(f"{'':>16} system Cached prefill {cf[0]/1e9:.2f}-{cf[1]/1e9:.2f} / decode {cd[0]/1e9:.2f}-{cd[1]/1e9:.2f} GB; "
                      f"MemFree prefill {ff[0]/1e9:.2f}-{ff[1]/1e9:.2f} / decode {fd_[0]/1e9:.2f}-{fd_[1]/1e9:.2f} GB")
            fr = row["share_resident_fraction_decode"]
            print(f"{arm:>16} {run['request_id'][-2:]:>3} {row['prompt_ms']:>10.0f} "
                  f"{(row['decode_ms_per_token'] or 0):>14.1f} "
                  f"{(pre_rss or 0)/1e9:>24.2f} {(dec_rss or 0)/1e9:>23.2f} {(avail or 0)/1e9:>23.2f} "
                  f"{('/'.join(f'{v:.3f}' for v in fr) if fr else '-'):>22} {str(row['phone_calls_mid_decode']):>11} "
                  f"{'yes' if row['error'] else '-':>4}")
        decode = [r["decode_ms_per_token"] for r in rows if r["decode_ms_per_token"]]
        summary["arms"][arm] = {"requests": rows, "decode_ms_per_token_mean": statistics.mean(decode) if decode else None,
                                "prompt_ms": [r["prompt_ms"] for r in rows]}
    for kind, lines in (result.get("proof_lines") or {}).items():
        print("proof lines", kind, [(l.get("phase"), l.get("released_bytes", l.get("restored_bytes")), l.get("elapsed_us")) for l in lines])
    idle = [s for s in result.get("residency_samples", []) if "idle" in s.get("tag", "")]
    print("idle residency samples:", [(s["tag"], round(s["resident_fraction"], 4) if s.get("resident_fraction") is not None else None, round(s["model_file_rss"]/1e9, 2)) for s in idle])
    if result.get("recovery"):
        rec = result["recovery"]
        print("recovery: victim error:", (rec["victim"]["error"] or "")[:120], "| follow-up error:", rec["local_follow_up"]["error"],
              "| follow-up decode ms/tok:", rec["local_follow_up"]["decode_ms_per_token"], "| after-loss share resident:",
              rec["after_owner_loss"]["resident_fraction"] if rec.get("after_owner_loss") else None)
    if args.out:
        args.out.write_text(json.dumps({"schema": "s42-decode-only-relocation-gate-summary-v1", **summary}, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
