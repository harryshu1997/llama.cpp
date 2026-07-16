#!/usr/bin/env python3
"""Differential harness: exact oracle vs the independent optimum reference.

Runs a deterministic slice of the frozen generated corpus (tests/gen_cases.py) and
compares the COMPLETE lexicographic objective of oracle/exact.py against
checker/reference.py, which has separate configuration, start-time, legality, and
objective enumeration and shares no oracle imports.

It also emits a digest over every (seed, objective, certificate_sha256) so separate
processes and PYTHONHASHSEED values can be compared byte for byte.

Exit codes: 0 all compared cases agree; 2 any disagreement or unexpected error.

Usage: differential.py --start 0 --count 250 [--quiet]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "oracle"), str(ROOT / "checker"), str(ROOT / "tests")]

import exact          # noqa: E402
import reference      # noqa: E402
import checker        # noqa: E402
import gen_cases      # noqa: E402


def run(start, count):
    digest = hashlib.sha256()
    compared = 0
    agreed_infeasible = 0
    skipped_domain = 0
    mismatches = []
    for seed in range(start, start + count):
        inst = gen_cases.make_case(seed)

        try:
            cert = exact.solve(inst)
            oracle_obj = tuple(cert["objective"])
            oracle_status = "ok"
            oracle_err = None
        except RuntimeError as exc:
            cert = None
            oracle_obj = None
            oracle_err = str(exc)
            if oracle_err == "no feasible complete schedule":
                oracle_status = "infeasible"
            elif (oracle_err.startswith("temporal exact domain is limited") or
                  oracle_err.startswith("temporal start-time domain exceeds") or
                  oracle_err.startswith("foundation exact solver is limited")):
                oracle_status = "domain"
            else:
                oracle_status = "error"

        try:
            ref_obj = tuple(reference.optimum(inst))
            ref_status = "ok"
            ref_err = None
        except reference.ReferenceOutOfDomain as exc:
            ref_obj = None
            ref_status = "domain"
            ref_err = str(exc)
        except RuntimeError as exc:
            ref_obj = None
            ref_err = str(exc)
            if ref_err.startswith("no legal schedule exists"):
                ref_status = "infeasible"
            else:
                ref_status = "error"

        if "error" in (oracle_status, ref_status):
            mismatches.append({"seed": seed, "instance_id": inst["instance_id"],
                               "oracle_status": oracle_status,
                               "oracle": oracle_obj, "oracle_err": oracle_err,
                               "reference_status": ref_status,
                               "reference": ref_obj, "reference_err": ref_err})
            continue

        if "domain" in (oracle_status, ref_status):
            skipped_domain += 1
            continue

        if oracle_status == ref_status == "infeasible":
            # both say there is no legal schedule
            agreed_infeasible += 1
            digest.update(f"{seed}:INFEASIBLE\n".encode("ascii"))
            continue

        if oracle_status != "ok" or ref_status != "ok":
            mismatches.append({"seed": seed, "instance_id": inst["instance_id"],
                               "oracle_status": oracle_status,
                               "oracle": oracle_obj, "oracle_err": oracle_err,
                               "reference_status": ref_status,
                               "reference": ref_obj, "reference_err": ref_err})
            continue

        if oracle_obj != ref_obj:
            mismatches.append({"seed": seed, "instance_id": inst["instance_id"],
                               "oracle": list(oracle_obj), "reference": list(ref_obj)})
            continue

        # the oracle's own certificate must also survive the independent checker
        failures = checker.check(inst, cert, require_complete=False)
        if failures:
            mismatches.append({"seed": seed, "instance_id": inst["instance_id"],
                               "checker_failures": failures[:3]})
            continue

        compared += 1
        digest.update(
            f"{seed}:{list(oracle_obj)}:{cert['certificate_sha256']}\n".encode("ascii"))

    return {
        "start": start,
        "count": count,
        "compared": compared,
        "agreed_infeasible": agreed_infeasible,
        "skipped_out_of_domain": skipped_domain,
        "mismatches": mismatches,
        "digest": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.count <= 0 or args.start < 0:
        print("DIFF_FAIL: invalid slice", file=sys.stderr)
        return 2
    result = run(args.start, args.count)
    if result["mismatches"]:
        for row in result["mismatches"][:10]:
            print(f"DIFF_FAIL: {json.dumps(row, sort_keys=True)}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(json.dumps({key: value for key, value in result.items()
                          if key != "mismatches"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
