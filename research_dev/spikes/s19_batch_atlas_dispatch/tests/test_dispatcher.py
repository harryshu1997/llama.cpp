#!/usr/bin/env python3
"""S19 CP2 unit + adversarial tests for the fail-closed dispatcher.

Covers every required test in PLAN.md section 11. Runs standalone:
  python3 tests/test_dispatcher.py
Exit 0 iff all pass. No pytest dependency, no device.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import dispatcher as D  # noqa: E402
import validate_dispatch as V  # noqa: E402

SEC = 1_000_000
MS = 1000


def mk_atlas(spec: dict) -> D.Atlas:
    rows = []
    for route, entries in spec.items():
        for (batch, p50, p95, verdict, urgent) in entries:
            rows.append(D.AtlasRow(route, batch, "dev", p50, p95, verdict, urgent))
    return D.Atlas(rows)


def base_credits(**over) -> D.Credits:
    c = D.Credits(
        free_kv={"CUDA0": 4096, "3C15AU002CL00000": 512, "5ae7a43d": 512},
        phone_lane={"OP15_R1": 100, "OP12_R1": 100},
        usb={"OP15_R1": 100, "OP12_R1": 100},
        cuda_tail={"OP15_R1": 100, "OP12_R1": 100})
    for k, v in over.items():
        getattr(c, k).update(v)
    return c


def base_epochs():
    return {r: D.Epochs(1, 1, 1) for r in ("CUDA_R0", "OP15_R1", "OP12_R1", "urgent")}


def req(rid, t, dl, priority="low", compat="c", urgent=False):
    return D.Request(rid, t, dl, priority, compat, urgent)


def reasons(d):
    return {rec["reason_code"] for rec in d.decisions}


def dispatched(d):
    return [rec for rec in d.decisions if rec["outcome"] == "dispatched"]


# ---------------------------------------------------------------------------
def test_unmeasured_batch_rejected():
    atlas = mk_atlas({"OP15_R1": [(8, 2_000_000, 2_100_000, "ELIGIBLE", False)]})
    d = D.Dispatcher(atlas, base_credits(), D.MockExecutor(), base_epochs())
    # feasibility fails closed for an unmeasured batch
    assert d._phone_feasible("OP15_R1", 16) == "unmeasured_batch"
    assert not atlas.has("OP15_R1", 16)
    assert 16 not in atlas.batches("OP15_R1")
    # a full run never dispatches a batch absent from the atlas
    ev = [{"kind": "arrival", "now_us": 0, "request": req(i, 0, 60 * SEC, compat="c")}
          for i in range(1, 9)]
    ev.append({"kind": "tick", "now_us": MS})
    ev.append({"kind": "shutdown", "now_us": 2 * MS})
    dd = D.run_schedule(ev, atlas, base_credits(), D.MockExecutor(), base_epochs())
    for rec in dispatched(dd):
        assert atlas.has(rec["selected_route"], rec["selected_batch"]), rec
    # validator rejects a forged log that dispatched an unmeasured batch
    forged = _one_dispatch("OP15_R1", 99, [1], "form_largest_useful")
    sel = {"OP15_R1": {8}}
    try:
        V.validate([forged], sel, _init_credits_dict(), [1])
        raise AssertionError("validator accepted unmeasured batch")
    except V.ValidationError as e:
        assert "unmeasured" in str(e)


def test_insufficient_memory_rejected():
    atlas = mk_atlas({"OP15_R1": [(8, 2_000_000, 2_100_000, "ELIGIBLE", False),
                                  (16, 2_200_000, 2_300_000, "ELIGIBLE", False)]})
    cr = base_credits(free_kv={"3C15AU002CL00000": 4})   # < 8
    ev = [{"kind": "arrival", "now_us": 0, "request": req(i, 0, 60 * SEC, compat="c")}
          for i in range(1, 17)]
    ev.append({"kind": "tick", "now_us": MS})
    ev.append({"kind": "shutdown", "now_us": 2 * MS})
    d = D.run_schedule(ev, atlas, cr, D.MockExecutor(), base_epochs())
    assert "insufficient_memory" in reasons(d)
    # no phone dispatch happened (free_kv too small)
    assert not any(r["selected_route"] == "OP15_R1" for r in dispatched(d))


def test_no_downstream_credit_rejected():
    atlas = mk_atlas({"OP15_R1": [(8, 2_000_000, 2_100_000, "ELIGIBLE", False)]})
    cr = base_credits(cuda_tail={"OP15_R1": 0})
    d = D.Dispatcher(atlas, cr, D.MockExecutor(), base_epochs())
    assert d._phone_feasible("OP15_R1", 8) == "no_downstream_credit"
    ev = [{"kind": "arrival", "now_us": 0, "request": req(i, 0, 60 * SEC, compat="c")}
          for i in range(1, 9)]
    ev.append({"kind": "tick", "now_us": MS})
    ev.append({"kind": "shutdown", "now_us": 2 * MS})
    dd = D.run_schedule(ev, atlas, base_credits(cuda_tail={"OP15_R1": 0}),
                        D.MockExecutor(), base_epochs())
    assert "no_downstream_credit" in reasons(dd)
    assert not any(r["selected_route"] == "OP15_R1" for r in dispatched(dd))


def test_b1_b2_only_urgent():
    # (a) non-urgent never selects B1/B2 even if an urgent-control row exists
    atlas = mk_atlas({"OP15_R1": [(8, 2_000_000, 2_100_000, "ELIGIBLE", False)],
                      "urgent": [(2, 300_000, 320_000, "ELIGIBLE", True)],
                      "CUDA_R0": [(8, 400_000, 420_000, "ELIGIBLE", False)]})
    ev = [{"kind": "arrival", "now_us": 0, "request": req(i, 0, 60 * SEC, compat="c")}
          for i in range(1, 9)]
    ev.append({"kind": "tick", "now_us": MS})
    ev.append({"kind": "shutdown", "now_us": 2 * MS})
    d = D.run_schedule(ev, atlas, base_credits(), D.MockExecutor(), base_epochs())
    assert not any(r["selected_batch"] in (1, 2) for r in dispatched(d))
    # (b) urgent with a certified B2 control -> B2 used with urgent reason.
    # Deadline is urgent (well within urgent_slack) but still long enough that the
    # B2 control's conservative finish (~384 ms) fits; otherwise fail-closed refuses.
    ev2 = [{"kind": "arrival", "now_us": 0,
            "request": req(i, 0, 600 * MS, urgent=True, compat="c")} for i in (1, 2)]
    ev2.append({"kind": "tick", "now_us": 10 * MS})
    ev2.append({"kind": "shutdown", "now_us": 700 * MS})
    d2 = D.run_schedule(ev2, atlas, base_credits(), D.MockExecutor(), base_epochs())
    b2 = [r for r in dispatched(d2) if r["selected_batch"] == 2]
    assert b2 and all(r["reason_code"] == "urgent_small_batch" for r in b2)
    # (c) validator rejects B2 dispatched outside the urgent branch
    forged = _one_dispatch("urgent", 2, [1], "form_largest_useful")
    try:
        V.validate([forged], {"urgent": {2}}, _init_credits_dict(), [1])
        raise AssertionError("validator accepted B2 outside urgent")
    except V.ValidationError as e:
        assert "urgent" in str(e)


def test_earliest_slo_forces_release():
    atlas = mk_atlas({"CUDA_R0": [(8, 400_000, 420_000, "ELIGIBLE", False),
                                  (16, 500_000, 520_000, "ELIGIBLE", False)]})
    # 8 requests, deadline 20 ms away -> no batch finishes in time -> forced release
    ev = [{"kind": "arrival", "now_us": 0,
           "request": req(i, 0, 20 * MS, compat="c")} for i in range(1, 9)]
    ev.append({"kind": "tick", "now_us": 10 * MS})
    ev.append({"kind": "shutdown", "now_us": 30 * MS})
    d = D.run_schedule(ev, atlas, base_credits(), D.MockExecutor(), base_epochs())
    assert "deadline_release" in reasons(d)


def test_conservation_no_loss_dup_double():
    atlas = _canonical_atlas()
    import run_scenario as RS
    events, all_rids, credits, epochs = RS.build_canonical(atlas)
    d = D.run_schedule(events, atlas, credits, D.MockExecutor(), epochs)
    rep = D.conservation_report(d, all_rids)
    assert rep["conserved"], rep
    assert not rep["missing"] and not rep["extra"] and not rep["duplicates"]
    # every request appears in exactly one committing cohort
    seen = {}
    for rec in d.decisions:
        if rec["outcome"] in ("dispatched", "terminal_failure", "rejected") and rec["cohort_request_ids"]:
            for rid in rec["cohort_request_ids"]:
                seen[rid] = seen.get(rid, 0) + 1
    assert all(c == 1 for c in seen.values()), [k for k, c in seen.items() if c > 1]
    assert set(seen) == set(all_rids)


def test_stale_epoch_rejected():
    atlas = mk_atlas({"OP15_R1": [(8, 2_000_000, 2_100_000, "ELIGIBLE", False)]})
    committed = base_epochs()
    live = base_epochs()
    live["OP15_R1"] = D.Epochs(2, 1, 1)   # worker generation advanced -> committed stale
    d = D.Dispatcher(atlas, base_credits(), D.MockExecutor(), committed, live_epochs=live)
    for i in range(1, 9):
        d.on_arrival(req(i, 0, 60 * SEC, compat="c"), 0)
    d.form_cohorts(MS)
    d.shutdown(2 * MS)
    assert "stale_epoch" in reasons(d)
    assert not any(r["selected_route"] == "OP15_R1" for r in dispatched(d))
    rep = D.conservation_report(d, list(range(1, 9)))
    assert rep["conserved"], rep


def test_deterministic_across_hashseed():
    atlas_path = HERE / "fixture_atlas.json"
    outs = []
    for seed in ("0", "1", "777"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        with tempfile.NamedTemporaryFile("r", suffix=".jsonl", delete=False) as tf:
            out = tf.name
        subprocess.run([sys.executable, str(ROOT / "run_scenario.py"),
                        "--atlas", str(atlas_path), "--executor", "mock", "--out", out],
                       check=True, capture_output=True, env=env)
        outs.append(Path(out).read_text())
    assert outs[0] == outs[1] == outs[2], "decision log differs across PYTHONHASHSEED"


def test_worker_failure_fallback_or_terminal():
    # (a) fallback: OP15 fails, CUDA B8 available -> fell_back then R0 dispatch
    atlas = mk_atlas({"OP15_R1": [(8, 2_000_000, 2_100_000, "ELIGIBLE", False)],
                      "CUDA_R0": [(8, 400_000, 420_000, "ELIGIBLE", False)]})
    ex = D.MockExecutor(fail_on=lambda route, b, ids: route == "OP15_R1")
    ev = [{"kind": "arrival", "now_us": 0, "request": req(i, 0, 60 * SEC, compat="c")}
          for i in range(1, 9)]
    ev.append({"kind": "tick", "now_us": MS})
    ev.append({"kind": "shutdown", "now_us": 2 * MS})
    d = D.run_schedule(ev, atlas, base_credits(), ex, base_epochs())
    assert "worker_failure" in reasons(d)
    # never a fabricated success on the failed route
    assert not any(r["selected_route"] == "OP15_R1" for r in dispatched(d))
    # requests completed via R0 fallback
    rep = D.conservation_report(d, list(range(1, 9)))
    assert rep["conserved"] and rep["by_outcome"].get("completed") == 8, rep
    # (b) terminal: OP15 fails, no CUDA atlas -> terminal_failure, no fake success
    atlas2 = mk_atlas({"OP15_R1": [(8, 2_000_000, 2_100_000, "ELIGIBLE", False)]})
    ex2 = D.MockExecutor(fail_on=lambda route, b, ids: True)
    d2 = D.run_schedule(ev, atlas2, base_credits(), ex2, base_epochs())
    assert "worker_failure" in reasons(d2)
    assert not dispatched(d2)  # zero fabricated successes
    rep2 = D.conservation_report(d2, list(range(1, 9)))
    assert rep2["conserved"], rep2
    assert rep2["by_outcome"].get("terminal_failure") == 8, rep2


# ---------------------------------------------------------------------------
def _one_dispatch(route, batch, cohort, reason):
    return {
        "schema": "s19-dispatch-decision-v1", "epoch": 0, "now_us": 0,
        "event": "form_cohort", "ready_before": sorted(cohort),
        "compatibility_key": "c", "priority": "low", "earliest_slo_us": 10 ** 9,
        "selected_route": route, "selected_batch": batch,
        "cohort_request_ids": sorted(cohort), "microbatch_plan": [batch],
        "reason_code": reason,
        "credits_after": _init_credits_dict(), "epochs": {"route": 1, "residency": 1, "session": 1},
        "outcome": "dispatched", "executed": None,
    }


def _init_credits_dict():
    return {"free_kv": {"CUDA0": 4096, "3C15AU002CL00000": 512, "5ae7a43d": 512},
            "phone_lane": {"OP15_R1": 100, "OP12_R1": 100},
            "usb": {"OP15_R1": 100, "OP12_R1": 100},
            "cuda_tail": {"OP15_R1": 100, "OP12_R1": 100}}


def _canonical_atlas():
    return D.Atlas([
        D.AtlasRow("CUDA_R0", b, "g", p, int(p * 1.05), "ELIGIBLE", False)
        for b, p in [(4, 317000), (8, 329000), (16, 365000), (24, 415000),
                     (32, 457000), (48, 502000), (64, 585000)]
    ] + [
        D.AtlasRow("OP15_R1", b, "o15", p, int(p * 1.05), "ELIGIBLE", False)
        for b, p in [(4, 1388000), (8, 1988000), (16, 2243000), (24, 2476000),
                     (32, 2735000), (48, 3133000), (64, 3586000)]
    ] + [
        D.AtlasRow("OP12_R1", b, "o12", p, int(p * 1.05), "ELIGIBLE", False)
        for b, p in [(4, 4200000), (8, 5400000), (16, 7000000), (32, 9148000)]
    ])


TESTS = [
    ("unmeasured_batch_rejected", test_unmeasured_batch_rejected),
    ("insufficient_memory_rejected", test_insufficient_memory_rejected),
    ("no_downstream_credit_rejected", test_no_downstream_credit_rejected),
    ("b1_b2_only_urgent", test_b1_b2_only_urgent),
    ("earliest_slo_forces_release", test_earliest_slo_forces_release),
    ("conservation_no_loss_dup_double", test_conservation_no_loss_dup_double),
    ("stale_epoch_rejected", test_stale_epoch_rejected),
    ("deterministic_across_hashseed", test_deterministic_across_hashseed),
    ("worker_failure_fallback_or_terminal", test_worker_failure_fallback_or_terminal),
]


def main() -> int:
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
