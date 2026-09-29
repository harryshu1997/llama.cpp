"""Merge the capped desktop-only arm (v2) with the capped dormant arm (v3): timings and cgroup pressure per phase."""

import argparse
import json
from pathlib import Path


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def phase_rows(memory, start_ns, end_ns):
    return [m for m in memory if start_ns <= m["observed_ns"] <= end_ns]


def cgroup_summary(rows):
    cg = [r["cgroup"] for r in rows if isinstance(r.get("cgroup"), dict) and isinstance(r["cgroup"].get("memory.current"), int)]
    if not cg:
        return None
    events = [c.get("memory.events", {}) for c in cg]
    stats = [c.get("memory.stat", {}) for c in cg]
    return {"memory_current_min": min(c["memory.current"] for c in cg), "memory_current_max": max(c["memory.current"] for c in cg),
            "memory_max": cg[0].get("memory.max"), "max_events_delta": events[-1].get("max", 0) - events[0].get("max", 0),
            "pgmajfault_delta": stats[-1].get("pgmajfault", 0) - stats[0].get("pgmajfault", 0),
            "anon_max": max(s.get("anon", 0) for s in stats), "file_max": max(s.get("file", 0) for s in stats)}


def describe(label, run, memory, proofs=None):
    first = run["first_token_ns"] or run["finished_ns"]
    pre = phase_rows(memory, run["started_ns"], first)
    dec = phase_rows(memory, first, run["finished_ns"])
    row = {"arm": label, "request_id": run["request_id"], "prompt_ms": run["prompt_ms"], "decode_ms_per_token": run["decode_ms_per_token"],
           "output_tokens": run["output_tokens"], "error": run.get("error"),
           "model_file_rss_decode_min": min((m["model_file_rss"] for m in dec), default=None),
           "model_file_rss_prefill_max": max((m["model_file_rss"] for m in pre), default=None),
           "cgroup_prefill": cgroup_summary(pre), "cgroup_decode": cgroup_summary(dec),
           "share_resident_decode": [s["resident_fraction"] for s in run.get("residency_samples", []) if ":decode" in s.get("tag", "")]}
    cg = row["cgroup_decode"] or {}
    print(f"{label:>22} {run['request_id']:>12} prefill {run['prompt_ms']/1000:>7.1f} s  decode {(run['decode_ms_per_token'] or 0):>8.1f} ms/tok  "
          f"file RSS decode min {(row['model_file_rss_decode_min'] or 0)/1e9:>5.2f} GB  cgroup decode current {cg.get('memory_current_min', 0)/1e9:.2f}-{cg.get('memory_current_max', 0)/1e9:.2f} GB "
          f"max-events +{cg.get('max_events_delta')} majfault +{cg.get('pgmajfault_delta')}  share resident {row['share_resident_decode']}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plain_dir", type=Path)
    parser.add_argument("dormant_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    plain_run = json.loads((args.plain_dir / "EXECUTION-plain-00.json").read_text())
    plain_mem = load_jsonl(args.plain_dir / "SERVER_MEMORY.jsonl")
    dormant = json.loads((args.dormant_dir / "RESULT.json").read_text())
    dormant_mem = load_jsonl(args.dormant_dir / "SERVER_MEMORY.jsonl")
    spec = json.loads((args.dormant_dir / "SPEC.json").read_text())
    print(f"cap: memory.max {spec['cgroup_at_start'].get('memory.max')} swap.max {spec['cgroup_at_start'].get('memory.swap.max')}; prompt {spec['prompt']['input_tokens']} tokens; release expected {dormant['expected_release_bytes']}")
    rows = [describe("desktop only (capped)", plain_run, plain_mem)]
    for run in dormant["runs"]:
        rows.append(describe("dormant 3 sessions (capped)", run, dormant_mem))
    proofs = dormant.get("proof_lines", {})
    for kind, lines in proofs.items():
        print("proof lines", kind, [(l.get("phase"), l.get("released_bytes", l.get("restored_bytes")), l.get("elapsed_us")) for l in lines])
    drops = [e for e in dormant.get("events", []) if e.get("kind") == "model_cache_drop"]
    print("cache drops:", [(d["label"], round(d["cached_before"]/1e9, 2), round(d["cached_after"]/1e9, 2)) for d in drops])
    if args.out:
        args.out.write_text(json.dumps({"schema": "s42-decode-only-relocation-capped-summary-v1", "cap": spec["cgroup_at_start"],
                                        "rows": rows, "proof_lines": proofs, "cache_drops": drops}, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
