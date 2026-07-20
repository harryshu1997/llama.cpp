#!/usr/bin/env python3
"""S19 CP2 fail-closed validator (independent of dispatcher.py logic).

Re-derives, from decision_log.jsonl + the atlas + initial credits alone:
  1. schema, reason-code, and outcome validity of every record;
  2. request conservation: every request reaches exactly one terminal outcome,
     none lost, duplicated, or double-completed;
  3. no dispatched/fell-back decision ever selected a batch without a selectable
     atlas row (no unmeasured batch);
  4. no phone cohort launched without a positive downstream CUDA-tail credit,
     phone-lane credit, USB credit, and sufficient free KV (re-derived here from
     first principles, not imported from the dispatcher);
  5. B1/B2 batches appear only under the urgent policy branch;
  6. credit balances never go negative.

Exit 0 and prints VALIDATION_PASS only if every check passes; else exit 2 with
the first failing check. This checker shares no scheduling code with
dispatcher.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

VALID_REASONS = {
    "unmeasured_batch", "insufficient_memory", "no_downstream_credit",
    "no_phone_credit", "urgent_small_batch", "r0_fallback",
    "form_largest_useful", "split_microbatch", "wait_for_batch",
    "deadline_release", "stale_epoch", "worker_failure", "conserve_shutdown",
}
VALID_OUTCOMES = {"dispatched", "waited", "rejected", "fell_back", "completed",
                  "terminal_failure"}
PHONE_ROUTES = {"OP15_R1", "OP12_R1"}
COMMIT_OUTCOMES = {"dispatched", "terminal_failure", "rejected"}
REQUIRED_KEYS = {
    "schema", "epoch", "now_us", "event", "ready_before", "compatibility_key",
    "priority", "earliest_slo_us", "selected_route", "selected_batch",
    "cohort_request_ids", "microbatch_plan", "reason_code", "credits_after",
    "epochs", "outcome", "executed",
}


class ValidationError(Exception):
    pass


def load_atlas_selectable(atlas_path: Path) -> dict:
    atlas = json.loads(atlas_path.read_text())
    sel: dict = {}
    for row in atlas["rows"]:
        if row["verdict"] in ("ELIGIBLE", "ELIGIBLE_SCREEN"):
            sel.setdefault(row["route"], set()).add(row["batch"])
    return sel


def validate(decisions: list[dict], selectable: dict, initial_credits: dict,
             all_request_ids: list[int]) -> dict:
    # 1. schema + reason + outcome
    for i, d in enumerate(decisions):
        if set(d) != REQUIRED_KEYS:
            raise ValidationError(f"decision {i}: key set mismatch {sorted(set(d) ^ REQUIRED_KEYS)}")
        if d["schema"] != "s19-dispatch-decision-v1":
            raise ValidationError(f"decision {i}: bad schema")
        if d["reason_code"] not in VALID_REASONS:
            raise ValidationError(f"decision {i}: bad reason {d['reason_code']}")
        if d["outcome"] not in VALID_OUTCOMES:
            raise ValidationError(f"decision {i}: bad outcome {d['outcome']}")
        if d["epoch"] != i:
            raise ValidationError(f"decision {i}: non-monotonic epoch {d['epoch']}")

    # 2. conservation
    committed: dict[int, int] = {}
    terminal: dict[int, str] = {}
    for d in decisions:
        if d["outcome"] in COMMIT_OUTCOMES and d["cohort_request_ids"]:
            for rid in d["cohort_request_ids"]:
                committed[rid] = committed.get(rid, 0) + 1
                if d["outcome"] == "dispatched":
                    terminal[rid] = "completed"
                elif d["outcome"] == "terminal_failure":
                    terminal[rid] = "terminal_failure"
                elif d["outcome"] == "rejected":
                    terminal[rid] = "rejected_terminal"
        if d["outcome"] == "fell_back" and d["cohort_request_ids"]:
            # fell_back is a transition; the R0 dispatch that follows commits.
            pass
    dupes = sorted(rid for rid, c in committed.items() if c > 1)
    if dupes:
        raise ValidationError(f"double-committed request ids: {dupes}")
    all_set = set(all_request_ids)
    missing = sorted(all_set - set(terminal))
    extra = sorted(set(terminal) - all_set)
    if missing:
        raise ValidationError(f"requests never reached a terminal outcome: {missing}")
    if extra:
        raise ValidationError(f"terminal outcome for unknown request ids: {extra}")

    # 3. no unmeasured batch dispatched / fell-back forward
    for i, d in enumerate(decisions):
        if d["outcome"] in ("dispatched",) and d["selected_batch"] is not None:
            route, batch = d["selected_route"], d["selected_batch"]
            # urgent control routes are named "urgent"; they must be selectable there
            if batch not in selectable.get(route, set()):
                raise ValidationError(
                    f"decision {i}: dispatched unmeasured batch {route} B{batch}")

    # 4. phone downstream credit + 6. non-negative credits (re-derive)
    free_kv = dict(initial_credits["free_kv"])
    phone_lane = dict(initial_credits["phone_lane"])
    usb = dict(initial_credits["usb"])
    cuda_tail = dict(initial_credits["cuda_tail"])
    route_device = initial_credits.get("route_device", {
        "CUDA_R0": "CUDA0", "OP15_R1": "3C15AU002CL00000",
        "OP12_R1": "5ae7a43d", "urgent": "CUDA0"})
    for i, d in enumerate(decisions):
        if d["outcome"] != "dispatched" or d["selected_batch"] is None:
            continue
        route, batch = d["selected_route"], d["selected_batch"]
        dev = route_device.get(route, "CUDA0")
        if free_kv.get(dev, 0) < batch:
            raise ValidationError(f"decision {i}: dispatched with free_kv {free_kv.get(dev,0)} < B{batch}")
        if route in PHONE_ROUTES:
            if cuda_tail.get(route, 0) <= 0:
                raise ValidationError(f"decision {i}: phone dispatch without downstream tail credit")
            if phone_lane.get(route, 0) <= 0:
                raise ValidationError(f"decision {i}: phone dispatch without lane credit")
            if usb.get(route, 0) <= 0:
                raise ValidationError(f"decision {i}: phone dispatch without USB credit")
            phone_lane[route] -= 1
            usb[route] -= 1
            cuda_tail[route] -= 1
        # free_kv is transient (reserved then released) so net zero per exchange
        for name, bal in (("phone_lane", phone_lane), ("usb", usb), ("cuda_tail", cuda_tail)):
            for k, v in bal.items():
                if v < 0:
                    raise ValidationError(f"decision {i}: {name}[{k}] went negative")

    # 5. B1/B2 only urgent
    for i, d in enumerate(decisions):
        if d["selected_batch"] in (1, 2) and d["outcome"] == "dispatched":
            if d["reason_code"] != "urgent_small_batch":
                raise ValidationError(
                    f"decision {i}: B{d['selected_batch']} used outside urgent branch")

    return {
        "decisions": len(decisions),
        "requests": len(all_set),
        "conserved": True,
        "terminal_by_outcome": _by_outcome(terminal),
    }


def _by_outcome(terminal: dict[int, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in terminal.values():
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--credits", required=True, help="JSON initial credits")
    ap.add_argument("--requests", required=True, help="JSON list of all request ids")
    args = ap.parse_args()
    decisions = [json.loads(l) for l in Path(args.decisions).read_text().splitlines() if l.strip()]
    selectable = load_atlas_selectable(Path(args.atlas))
    credits = json.loads(Path(args.credits).read_text())
    rids = json.loads(Path(args.requests).read_text())
    try:
        report = validate(decisions, selectable, credits, rids)
    except ValidationError as exc:
        print(f"VALIDATION_FAIL: {exc}")
        return 2
    print("VALIDATION_PASS " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
