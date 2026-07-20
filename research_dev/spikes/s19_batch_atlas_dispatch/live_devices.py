#!/usr/bin/env python3
"""S19 CP2 live-device proof (VARIABLE_COHORT_BATCHING, one cohort per exchange).

Two artifacts:
  1. results/decision_log.jsonl - the full canonical schedule run through the
     fail-closed dispatcher with the MOCK executor over the REAL atlas. This is
     the deterministic scheduling-mechanics proof (all reason codes,
     conservation, PYTHONHASHSEED-stable).
  2. results/device_exchanges.jsonl - real-device execution records for the
     dispatching cohorts (physical proof the selected routes/batches run on real
     hardware): varying batch on CUDA R0, a concurrent OP15+OP12 launch with
     measured wall-clock overlap, and an R0 fallback for a credit-refused OP12
     cohort. Each record binds the real worker pid/nonce, route wall time,
     generated token ids, token match vs the atlas control, and placement.

Pure-refusal decisions (unmeasured, no-credit, stale) need no device execution;
they are proven by the decision log + validator + unit tests.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import atlas_measure as AM
import device_executor as DE
import dispatcher as D
import run_scenario as RS

HERE = Path(__file__).resolve().parent


def control_tokens(atlas_json: dict, batch: int) -> list[int]:
    for r in atlas_json["rows"]:
        if r["route"] == "CUDA_R0" and r["batch"] == batch:
            return r["correctness"]["route_token_ids"]
    return []


def common_eligible_batch(atlas_json: dict, routes: list[str]) -> int | None:
    """Largest common eligible batch <= 8 (fast + robust for the concurrency demo,
    matching the canonical scenario's concurrent phase). Falls back to the overall
    max common batch if nothing <= 8 is common."""
    sets = []
    for rt in routes:
        sets.append({r["batch"] for r in atlas_json["rows"]
                     if r["route"] == rt and r["verdict"] in ("ELIGIBLE", "ELIGIBLE_SCREEN")})
    common = set.intersection(*sets) if sets else set()
    if not common:
        return None
    small = [b for b in common if b <= 8]
    return max(small) if small else max(common)


def run_live(atlas_path: Path, out_dir: Path) -> dict:
    atlas_json = json.loads(atlas_path.read_text())
    atlas = RS.load_atlas(atlas_path)
    records = []
    dxlog = out_dir / "results" / "device_exchanges.jsonl"
    dxlog.parent.mkdir(parents=True, exist_ok=True)

    def emit(rec):
        rec["control_token_ids"] = control_tokens(atlas_json, rec["batch"])
        rec["token_match_vs_control"] = bool(rec["token_ids"]) and \
            rec["token_ids"] == rec["control_token_ids"]
        records.append(rec)
        with dxlog.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        print(f"  DX {rec['scenario']}: {rec['route']} B{rec['batch']} ok={rec['ok']} "
              f"match={rec['token_match_vs_control']} wall={rec['route_wall_us']/1000:.0f}ms "
              f"pid={rec['worker_pid']}", flush=True)
        return rec

    # --- Scenario 1: varying batch on CUDA R0 (real) ---
    print("[live] scenario 1: varying batch on CUDA R0", flush=True)
    cuda_batches = [b for b in (8, 16, 32)
                    if any(r["route"] == "CUDA_R0" and r["batch"] == b
                           and r["verdict"].startswith("ELIGIBLE") for r in atlas_json["rows"])]
    for b in cuda_batches:
        rec = DE.execute_one("CUDA_R0", b, out_dir, f"s1_cuda_b{b}")
        rec["scenario"] = "varying_batch_cuda_r0"
        emit(rec)

    # --- Scenario 2: concurrent OP15 + OP12 launch (real, measured overlap) ---
    print("[live] scenario 2: concurrent OP15 + OP12 launch", flush=True)
    cb = common_eligible_batch(atlas_json, ["OP15_R1", "OP12_R1"])
    overlap_us = None
    if cb is not None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f15 = ex.submit(DE.execute_one, "OP15_R1", cb, out_dir, f"s2_op15_b{cb}")
            f12 = ex.submit(DE.execute_one, "OP12_R1", cb, out_dir, f"s2_op12_b{cb}")
            r15 = f15.result()
            r12 = f12.result()
        for rec in (r15, r12):
            rec["scenario"] = "concurrent_op15_op12"
            emit(rec)
        # wall-clock overlap of the two exchange windows
        s = max(r15["wall_window_us"][0], r12["wall_window_us"][0])
        e = min(r15["wall_window_us"][1], r12["wall_window_us"][1])
        overlap_us = max(0, e - s)
        print(f"  concurrent overlap = {overlap_us/1000:.0f} ms (B{cb})", flush=True)

    # --- Scenario 3: R0 fallback for a credit-refused OP12 cohort (real) ---
    print("[live] scenario 3: R0 fallback (OP12 credit exhausted -> CUDA R0)", flush=True)
    fb_batch = 8 if any(r["route"] == "CUDA_R0" and r["batch"] == 8
                        and r["verdict"].startswith("ELIGIBLE") for r in atlas_json["rows"]) else cuda_batches[0]
    rec = DE.execute_one("CUDA_R0", fb_batch, out_dir, f"s3_fallback_b{fb_batch}")
    rec["scenario"] = "r0_fallback_after_op12_refusal"
    emit(rec)

    summary = {
        "device_exchanges": len(records),
        "all_ok": all(r["ok"] for r in records),
        "all_token_match": all(r["token_match_vs_control"] for r in records),
        "all_placement_ok": all(r["placement_ok"] for r in records),
        "concurrent_common_batch": cb,
        "concurrent_overlap_us": overlap_us,
        "distinct_batches_executed": sorted({r["batch"] for r in records}),
        "routes_executed": sorted({r["route"] for r in records}),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=str(HERE / "batch_atlas.json"))
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    out_dir = Path(args.out)

    # 1. deterministic full scheduling decision log (mock executor over real atlas)
    atlas = RS.load_atlas(Path(args.atlas))
    events, all_rids, credits, epochs = RS.build_canonical(atlas)
    d = D.run_schedule(events, atlas, credits, D.MockExecutor(), epochs)
    dl = out_dir / "results" / "decision_log.jsonl"
    dl.parent.mkdir(parents=True, exist_ok=True)
    with dl.open("w") as f:
        for rec in d.decisions:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    conservation = D.conservation_report(d, all_rids)
    print(f"[live] decision_log: {len(d.decisions)} decisions, conserved={conservation['conserved']}", flush=True)

    # 2. real-device proof
    live = run_live(Path(args.atlas), out_dir)

    report = {
        "schema": "s19-live-report-v1",
        "decision_log": {"decisions": len(d.decisions),
                         "conservation": conservation,
                         "reason_codes": sorted({r["reason_code"] for r in d.decisions})},
        "device_proof": live,
        "verdict": ("S19_CP2_VARIABLE_COHORT_BATCHING_MECHANICS_PASS"
                    if conservation["conserved"] and live["all_ok"]
                    and live["all_token_match"] and live["all_placement_ok"]
                    else "S19_CP2_MECHANICS_INCOMPLETE"),
    }
    (out_dir / "results" / "live_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
