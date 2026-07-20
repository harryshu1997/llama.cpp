#!/usr/bin/env python3
"""Assemble batch_atlas.json from the measured atlas_rows.jsonl.

Applies the frozen correctness gate: a phone route row at batch B is exact only
if its generated token ids equal the SAME-batch CUDA_R0 control row's token ids
(CUDA_R0 is its own control). Assigns verdicts, computes the per-route
throughput knee, and emits the deliverable atlas plus a human summary.

Row verdicts (PLAN.md section 6):
  ELIGIBLE        support+!oom+correct+placement, eligible_level=="eligible"
  ELIGIBLE_SCREEN support+!oom+correct+placement, screen level only
  INELIGIBLE      fails support/memory/correctness/placement
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE / "results" / "atlas_rows.jsonl"
OUT = HERE / "batch_atlas.json"


def load_rows(tag: str = "screen") -> list[dict]:
    rows = []
    for line in ROWS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("tag") == tag:
            rows.append(r)
    return rows


def latest_by_cell(rows: list[dict]) -> dict:
    """Keep the last row per (route, batch) (later runs supersede)."""
    out = {}
    for r in rows:
        out[(r["route"], r["batch"])] = r
    return out


def build(screen_rows: list[dict], eligible_rows: list[dict]) -> dict:
    cells = latest_by_cell(screen_rows)
    # eligible-level rows (7 exch + 3 reps) UPGRADE a cell to eligible only when
    # they themselves passed. A flaky eligible-run hang (e.g. OP12) must not
    # downgrade a batch that already certified at screen level.
    for (k, r) in latest_by_cell(eligible_rows).items():
        if r.get("support"):
            cells[k] = r
        elif k not in cells:
            cells[k] = r   # only-evidence-is-a-failed-eligible-run -> record the fail

    cuda_tokens = {b: cells[("CUDA_R0", b)]["correctness"]["route_token_ids"]
                   for (route, b) in cells if route == "CUDA_R0"
                   and cells[("CUDA_R0", b)]["support"]}

    atlas_rows = []
    for (route, batch), r in sorted(cells.items()):
        support = r["support"]
        oom = r["oom"]
        placement = r["placement"]
        route_tokens = r["correctness"]["route_token_ids"]
        control_tokens = cuda_tokens.get(batch, []) if route != "CUDA_R0" else route_tokens
        exact = bool(route_tokens) and route_tokens == control_tokens
        plc_ok = placement["scheduled_placement_ok"] and placement["missing_buffer"] == 0 \
            and placement["cpu_get_rows_only"]
        gpu1_busy = r.get("gpu1", {}).get("busy", False)
        level = "eligible" if r["process_evidence"].get("eligible_level") == "eligible" \
            else ("eligible" if r.get("tag") == "eligible" else "screen")

        reason = None
        if not support:
            reason = "oom" if oom else "unsupported"
        elif oom:
            reason = "oom"
        elif not exact:
            reason = "inexact_tokens"
        elif not plc_ok:
            reason = "placement_fail"
        elif gpu1_busy:
            reason = "gpu1_busy"

        if reason is not None:
            verdict = "INELIGIBLE"
        elif level == "eligible":
            verdict = "ELIGIBLE"
        else:
            verdict = "ELIGIBLE_SCREEN"

        atlas_rows.append({
            "schema": "s19-batch-atlas-row-v1",
            "route": route, "batch": batch,
            "device": r["device"], "backend": r["backend"],
            "layer_range": r["layer_range"], "host_layer_start": r["host_layer_start"],
            "model_hashes": r["model_hashes"], "binary_hashes": r["binary_hashes"],
            "context_envelope": r["context_envelope"],
            "support": support, "memory_ok": r["memory_ok"], "oom": oom,
            "selected_gpu_peak_mib": r["selected_gpu_peak_mib"],
            "correctness": {"control_token_ids": control_tokens,
                            "route_token_ids": route_tokens, "exact_match": exact},
            "placement": placement,
            "p50_us": int(r["latency_us"]["p50"]),
            "p95_us": int(r["latency_us"]["p95"]),
            "p99_us": int(r["latency_us"]["p99"]),
            "latency_us": r["latency_us"],
            "throughput_tok_s": r["throughput_tok_s"],
            "thermal": r["thermal"],
            "process_evidence": r["process_evidence"],
            "gpu1": r.get("gpu1"),
            "eligible_level": level,
            "verdict": verdict,
            "ineligible_reason": reason,
        })

    # per-route throughput knee: max throughput among ELIGIBLE/ELIGIBLE_SCREEN rows,
    # then the knee is the smallest batch reaching >=95 percent of that peak.
    summary = {}
    for route in ("CUDA_R0", "OP15_R1", "OP12_R1"):
        elig = [r for r in atlas_rows if r["route"] == route
                and r["verdict"] in ("ELIGIBLE", "ELIGIBLE_SCREEN")]
        if not elig:
            summary[route] = {"eligible_batches": [], "knee": None}
            continue
        peak = max(r["throughput_tok_s"] for r in elig)
        knee = min((r["batch"] for r in sorted(elig, key=lambda x: x["batch"])
                    if r["throughput_tok_s"] >= 0.95 * peak), default=None)
        elig_batches = sorted(r["batch"] for r in elig)
        adjacent = []
        if knee is not None:
            i = elig_batches.index(knee)
            for j in (i - 1, i, i + 1):
                if 0 <= j < len(elig_batches):
                    adjacent.append(elig_batches[j])
        summary[route] = {
            "eligible_batches": elig_batches,
            "peak_throughput_tok_s": round(peak, 1),
            "knee_batch": knee,
            "knee_and_adjacent": sorted(set(adjacent)),
            "throughput_by_batch": {r["batch"]: round(r["throughput_tok_s"], 1) for r in elig},
            "p50_ms_by_batch": {r["batch"]: round(r["p50_us"] / 1000, 1) for r in elig},
            "peak_gpu_mib_by_batch": {r["batch"]: r["selected_gpu_peak_mib"] for r in elig},
        }

    return {"schema": "s19-batch-atlas-v1", "rows": atlas_rows, "summary": summary}


def main() -> int:
    screen = load_rows("screen")
    eligible = load_rows("eligible")
    atlas = build(screen, eligible)
    OUT.write_text(json.dumps(atlas, indent=1))
    print(f"wrote {OUT} with {len(atlas['rows'])} rows")
    for route, s in atlas["summary"].items():
        print(f"{route}: eligible={s['eligible_batches']} knee={s.get('knee_batch')} "
              f"peak={s.get('peak_throughput_tok_s')}tok/s adj={s.get('knee_and_adjacent')}")
    inelig = [(r["route"], r["batch"], r["ineligible_reason"]) for r in atlas["rows"]
              if r["verdict"] == "INELIGIBLE"]
    if inelig:
        print("INELIGIBLE:", inelig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
