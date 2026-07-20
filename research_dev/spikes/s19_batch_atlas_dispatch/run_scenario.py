#!/usr/bin/env python3
"""S19 CP2 canonical arrival-varying scenario driver.

Builds the frozen deterministic scenario (PLAN.md section 12) over a measured
atlas and runs the fail-closed dispatcher. With --executor mock it is fully
deterministic (used for the PYTHONHASHSEED determinism test and unit checks);
with --executor device it drives the real persistent workers (device_executor).

Writes decision_log.jsonl (sort_keys, deterministic) and prints a summary of the
required reason codes present plus the conservation report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import dispatcher as D

HERE = Path(__file__).resolve().parent


def load_atlas(path: Path) -> D.Atlas:
    data = json.loads(path.read_text())
    rows = []
    for r in data["rows"]:
        rows.append(D.AtlasRow(
            route=r["route"], batch=r["batch"], device=r.get("device", ""),
            p50_us=int(r["p50_us"]), p95_us=int(r["p95_us"]),
            verdict=r["verdict"], urgent_control=r.get("urgent_control", False)))
    return D.Atlas(rows)


def compat_for_phone(target_route: str, avail: list[str], salt: str) -> str:
    """Deterministically find a compat string that route-selects target_route."""
    idx_want = avail.index(target_route)
    for i in range(10000):
        c = f"{salt}-{i}"
        idx = int(hashlib.sha256(c.encode()).hexdigest(), 16) % len(avail)
        if idx == idx_want:
            return c
    raise RuntimeError("no compat found")


def build_canonical(atlas: D.Atlas):
    """Return (events, all_rids, credits, epochs). Deterministic.

    Forces every required behavior (PLAN section 12). Phone routing is by a
    stable hash of the compatibility key over the full PHONE_ROUTES list, which
    matches the dispatcher's route selection when both phones have eligible rows.
    """
    c15 = compat_for_phone("OP15_R1", list(D.PHONE_ROUTES), "lane15")
    c15b = compat_for_phone("OP15_R1", list(D.PHONE_ROUTES), "lane15b")
    c15c = compat_for_phone("OP15_R1", list(D.PHONE_ROUTES), "lane15c")
    c15d = compat_for_phone("OP15_R1", list(D.PHONE_ROUTES), "lane15d")
    c15e = compat_for_phone("OP15_R1", list(D.PHONE_ROUTES), "lane15e")
    c12 = compat_for_phone("OP12_R1", list(D.PHONE_ROUTES), "lane12")
    c12b = compat_for_phone("OP12_R1", list(D.PHONE_ROUTES), "lane12b")
    MS = 1000
    SEC = 1_000_000
    far = 120 * SEC

    events = []
    rid = 0

    def arrive(t, n, compat, priority="low", deadline=None, urgent=False):
        nonlocal rid
        dl = deadline if deadline is not None else t + far
        for _ in range(n):
            rid += 1
            events.append({"kind": "arrival", "now_us": t,
                           "request": D.Request(rid, t, dl, priority, compat, urgent)})

    def tick(t, drain=False):
        events.append({"kind": "tick", "now_us": t, "drain": drain})

    # Phase A: concurrent OP15 + OP12 launches, both B8.
    arrive(0, 8, c15)
    arrive(0, 8, c12)                               # OP12 use 1 of 2
    tick(50 * MS)

    # Phase B: wait_for_batch (3 < smallest batch) then a larger cohort on OP15.
    arrive(200 * MS, 3, c15b)
    tick(250 * MS)
    arrive(300 * MS, 13, c15b)                      # total 16 -> split into measured
    tick(350 * MS)

    # Phase C: split_microbatch (20 -> largest useful + remainder) on OP15.
    arrive(400 * MS, 20, c15c)
    tick(450 * MS)

    # Phase H/I: larger low-priority cohorts exercise the mid/large measured
    # batches (different selected sizes over time), each on a fresh OP15 compat.
    arrive(1000 * MS, 32, c15d)                     # -> B32
    tick(1050 * MS)
    arrive(1100 * MS, 48, c15e)                     # -> B48
    tick(1150 * MS)

    # Phase J: a non-urgent high-priority cohort stays on CUDA R0 at a mid batch.
    arrive(1200 * MS, 24, "high-2", priority="high")   # -> CUDA B24
    tick(1250 * MS)

    # Phase D: exhaust OP12 credit (use 2), then a 3rd OP12 group is refused -> R0.
    arrive(1300 * MS, 8, c12)                       # OP12 use 2 of 2 (credit -> 0)
    tick(1350 * MS)
    arrive(1400 * MS, 8, c12b)                      # no_phone_credit -> r0_fallback
    tick(1450 * MS)

    # Phase E: deadline_release below preferred (tight SLO forces an R0 B8 now).
    arrive(1500 * MS, 8, "urgent-low", deadline=1500 * MS + 20 * MS)
    tick(1510 * MS)

    # Phase F: high-priority urgent work stays on R0.
    arrive(1600 * MS, 8, "high-1", priority="high",
           deadline=1600 * MS + 20 * MS, urgent=True)
    tick(1610 * MS)

    # Phase G: a sub-minimum-batch tail left pending -> fail-closed terminal at
    # shutdown (conservation: it reaches a terminal outcome, is not lost).
    arrive(1700 * MS, 2, "tail-1")
    tick(1750 * MS)

    # shutdown drains everything and verifies conservation
    events.append({"kind": "shutdown", "now_us": 1800 * MS})

    all_rids = list(range(1, rid + 1))
    credits = _initial_credits()
    epochs = {
        "CUDA_R0": D.Epochs(1, 1, 1),
        "OP15_R1": D.Epochs(1, 1, 1),
        "OP12_R1": D.Epochs(1, 1, 1),
        "urgent": D.Epochs(1, 1, 1),
    }
    return events, all_rids, credits, epochs


def _initial_credits() -> "D.Credits":
    return D.Credits(
        free_kv={"CUDA0": 4096, "3C15AU002CL00000": 512, "5ae7a43d": 512},
        phone_lane={"OP15_R1": 100, "OP12_R1": 2},   # OP12 hard credit of 2 (S18)
        usb={"OP15_R1": 100, "OP12_R1": 100},
        cuda_tail={"OP15_R1": 100, "OP12_R1": 100})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--executor", choices=["mock", "device"], default="mock")
    ap.add_argument("--out", default=str(HERE / "results" / "decision_log.jsonl"))
    ap.add_argument("--credits-out", default=None)
    ap.add_argument("--requests-out", default=None)
    args = ap.parse_args()

    atlas = load_atlas(Path(args.atlas))
    events, all_rids, credits, epochs = build_canonical(atlas)

    if args.executor == "mock":
        executor = D.MockExecutor()
    else:
        import device_executor
        executor = device_executor.DeviceExecutor()

    d = D.run_schedule(events, atlas, credits, executor, epochs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in d.decisions:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    if args.credits_out:
        Path(args.credits_out).write_text(json.dumps(_initial_credits().snapshot()))
    if args.requests_out:
        Path(args.requests_out).write_text(json.dumps(all_rids))

    report = D.conservation_report(d, all_rids)
    reasons = sorted({rec["reason_code"] for rec in d.decisions})
    required = {"form_largest_useful", "split_microbatch", "wait_for_batch",
                "deadline_release", "no_phone_credit", "r0_fallback",
                "conserve_shutdown"}
    batches = sorted({rec["selected_batch"] for rec in d.decisions
                      if rec["selected_batch"] is not None and rec["outcome"] == "dispatched"})
    dispatched = [rec for rec in d.decisions if rec["outcome"] == "dispatched"]
    concurrent = _concurrent_phone_launches(dispatched)
    summary = {
        "decisions": len(d.decisions),
        "conservation": report,
        "reason_codes_present": reasons,
        "required_reason_codes_present": sorted(required & set(reasons)),
        "required_reason_codes_missing": sorted(required - set(reasons)),
        "distinct_dispatched_batches": batches,
        "concurrent_op15_op12_epochs": concurrent,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["conserved"] and not (required - set(reasons)) else 1


def _concurrent_phone_launches(dispatched: list[dict]) -> list[int]:
    """now_us values where both an OP15 and an OP12 cohort dispatched."""
    by_time: dict[int, set] = {}
    for rec in dispatched:
        if rec["selected_route"] in D.PHONE_ROUTES:
            by_time.setdefault(rec["now_us"], set()).add(rec["selected_route"])
    return sorted(t for t, s in by_time.items() if s == set(D.PHONE_ROUTES))


if __name__ == "__main__":
    sys.exit(main())
