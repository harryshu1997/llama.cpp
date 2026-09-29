"""Summarize a decode split-ratio sweep directory: per-fraction decode time, energy and server RSS."""

import argparse
import json
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    runs = []
    for path in sorted(args.sweep.glob("EXECUTION-*.json")):
        row = json.loads(path.read_text())
        execution = row.get("execution", {})
        tokens = row.get("tokens") or []
        predicted_ms = float(execution.get("predicted_ms") or row.get("predicted_ms") or 0.0)
        prompt_ms = float(execution.get("prompt_ms") or row.get("prompt_ms") or 0.0)
        runs.append({
            "fraction_percent": row["fraction_percent"], "sequence": row["sequence"], "repetition": row["repetition"],
            "request_id": row["request_id"], "started_ns": row["started_ns"], "finished_ns": row["finished_ns"],
            "first_token_ns": row.get("first_token_ns"), "wall_s": row["wall_s"], "prompt_ms": prompt_ms,
            "predicted_ms": predicted_ms, "output_tokens": len(tokens),
            "decode_ms_per_token": predicted_ms / len(tokens) if tokens else None,
            "tokens": tokens, "controls": row.get("controls", []),
        })
    memory = []
    memory_path = args.sweep / "SERVER_MEMORY.jsonl"
    if memory_path.exists():
        for line in memory_path.read_text().splitlines():
            if line.strip():
                memory.append(json.loads(line))
    result_path = args.sweep / "RESULT.json"
    energies = {}
    if result_path.exists():
        for row in json.loads(result_path.read_text())["runs"]:
            energies[row["request_id"]] = row.get("energy")
    summary = {}
    by_fraction = {}
    for run in runs:
        by_fraction.setdefault(run["fraction_percent"], []).append(run)
    plain_tokens = None
    for fraction in sorted(by_fraction):
        rows = by_fraction[fraction]
        decode = [r["decode_ms_per_token"] for r in rows if r["decode_ms_per_token"]]
        rss = []
        for r in rows:
            samples = [m["bytes"].get("VmRSS", 0) for m in memory
                       if r["first_token_ns"] and r["first_token_ns"] <= m["observed_ns"] <= r["finished_ns"]]
            if samples:
                rss.append(max(samples))
        energy = [energies[r["request_id"]]["server_compute_device_energy_j"] for r in rows
                  if energies.get(r["request_id"])]
        agree = None
        if fraction == 0:
            plain_tokens = rows[0]["tokens"]
        elif plain_tokens:
            agree = [sum(1 for a, b in zip(plain_tokens, r["tokens"]) if a == b) for r in rows]
        summary[str(fraction)] = {
            "runs": len(rows),
            "decode_ms_per_token": {"mean": statistics.mean(decode) if decode else None,
                                    "values": decode},
            "prompt_ms": [r["prompt_ms"] for r in rows],
            "wall_s": [round(r["wall_s"], 3) for r in rows],
            "server_energy_j": energy,
            "decode_rss_max_bytes": rss,
            "tokens_agreeing_with_plain_first_run": agree,
            "output_tokens": [r["output_tokens"] for r in rows],
        }
    plain = summary.get("0", {}).get("decode_ms_per_token", {}).get("mean")
    for fraction, row in summary.items():
        mean = row["decode_ms_per_token"]["mean"]
        row["decode_speedup_vs_plain_percent"] = None if not (plain and mean) else round((plain - mean) / plain * 100, 2)
    print(f"{'phone %':>8} {'decode ms/tok':>14} {'vs plain':>9} {'prompt ms':>10} {'energy J':>9} {'RSS GB':>7} {'agree':>6}")
    for fraction in sorted(summary, key=int):
        row = summary[fraction]
        mean = row["decode_ms_per_token"]["mean"]
        e = row["server_energy_j"]
        rss = row["decode_rss_max_bytes"]
        print(f"{fraction:>8} {mean if mean is None else round(mean, 1):>14} "
              f"{row['decode_speedup_vs_plain_percent'] if row['decode_speedup_vs_plain_percent'] is not None else '-':>9} "
              f"{round(statistics.mean(row['prompt_ms'])) if row['prompt_ms'] else '-':>10} "
              f"{round(statistics.mean(e)) if e else '-':>9} {round(max(rss) / 1e9, 2) if rss else '-':>7} "
              f"{'/'.join(str(v) for v in row['tokens_agreeing_with_plain_first_run']) if row['tokens_agreeing_with_plain_first_run'] else '-':>6}")
    if args.out:
        args.out.write_text(json.dumps({"schema": "s42-decode-split-sweep-summary-v1", "summary": summary,
                                        "runs": [{k: v for k, v in r.items() if k != "tokens"} for r in runs]},
                                       indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
