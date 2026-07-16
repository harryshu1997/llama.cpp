#!/usr/bin/env python3
"""Emit a feasible-but-SUBOPTIMAL certificate for the frozen transition fixture.

The schedule is legal and has identical zero-miss/zero-lateness outcomes, but it is
the earliest/left placement (162000000 nJ) rather than the optimal delayed placement
(147250000 nJ). It carries the same complete-search marker as the real optimum and a
valid reseal, so it can only be rejected by an INDEPENDENT optimality proof -- never
by treating a signed completeness assertion as proof.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "oracle")]
import exact  # noqa: E402


def build():
    inst = exact.load_strict(str(ROOT / "fixtures" / "transition_delay_counterexample.json"))
    req_by_id, node_by_id = exact.validate_instance(inst)
    actions = [
        {"id": "a00", "device": "SERVER", "members": ["n0"],
         "start_us": 50, "finish_us": 150},
        {"id": "a01", "device": "SERVER", "members": ["n1"],
         "start_us": 1000, "finish_us": 1100},
    ]
    result = exact.evaluate(inst, req_by_id, node_by_id, actions)
    if result is None:
        raise SystemExit("suboptimal schedule is unexpectedly infeasible")
    optimum = exact.solve(inst)
    cert = {
        "schema_version": 2,
        "instance_id": inst["instance_id"],
        "instance_sha256": exact.digest(inst),
        "actions": actions,
        "request_outcomes": result["request_outcomes"],
        "activation_peak_bytes": result["activation_peak_bytes"],
        "energy": result["energy"],
        "objective": result["objective"],
        # borrowed completeness assertion from the genuine optimum
        "search": copy.deepcopy(optimum["search"]),
    }
    cert["certificate_sha256"] = exact.digest(cert)
    if cert["energy"]["total_nj"] != 162000000:
        raise SystemExit(f"expected the 162000000 nJ earliest schedule, got "
                         f"{cert['energy']['total_nj']}")
    return inst, cert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    _inst, cert = build()
    with open(args.out, "w", encoding="ascii") as handle:
        handle.write(json.dumps(cert, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
