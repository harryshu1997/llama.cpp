#!/usr/bin/env python3
"""CP-C negative tests: the 3-device acceptance gate must REJECT a malformed run.

Covers the required failure modes: missing OP12 certificate, failed host process,
wrong layer range, CPU fallback (undeclared CPU op), and incomplete result rows.
Pure unit tests on pipe3_device.build_checks / validate_stage_cert -- no devices.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipe3_device as p3

K1, K2 = 8, 12
REF = [1, 2, 3, 4, 5, 6, 7, 8]


def good_cert(start, end):
    return {
        "status": "SCHEDULED_PLACEMENT_OK", "missing_buffer_compute_nodes": 0,
        "layer_start": start, "layer_end": end,
        "compute_by_buffer_type": {"HTP0": 3712, "CPU": 16},
        "compute_by_op_and_buffer": {"MUL_MAT": {"HTP0": 100}, "GET_ROWS": {"CPU": 16}},
    }


def good_rows():
    return [{"token_ids": REF, "stage_a_us": 100, "host_us": 50, "request_wall_us": 200}]


CASES = []


def case(name, host_rc, rows, certA, certB, expect_certified):
    checks = p3.build_checks(host_rc, rows, certA, certB, K1, K2, REF)
    certified = all(checks.values())
    ok = certified == expect_certified
    CASES.append((name, ok, certified, checks))
    return ok


def main():
    # Positive control.
    case("valid_run", 0, good_rows(), good_cert(0, K1), good_cert(K1, K2), True)
    case("missing_op12_cert", 0, good_rows(), good_cert(0, K1), None, False)
    case("failed_host_rc", 3, good_rows(), good_cert(0, K1), good_cert(K1, K2), False)
    case("wrong_range_op12", 0, good_rows(), good_cert(0, K1), good_cert(K1, 18), False)

    bad_fallback = good_cert(0, K1)
    bad_fallback["compute_by_op_and_buffer"] = {"MUL_MAT": {"CPU": 5}, "GET_ROWS": {"CPU": 16}}
    case("cpu_fallback_mulmat", 0, good_rows(), bad_fallback, good_cert(K1, K2), False)
    case("incomplete_rows", 0, [], good_cert(0, K1), good_cert(K1, K2), False)

    bad_buf = good_cert(K1, K2)
    bad_buf["missing_buffer_compute_nodes"] = 3
    case("missing_buffer_node", 0, good_rows(), good_cert(0, K1), bad_buf, False)
    case(
        "token_mismatch",
        0,
        [{"token_ids": [9, 9, 9], "stage_a_us": 1, "host_us": 1, "request_wall_us": 1}],
        good_cert(0, K1),
        good_cert(K1, K2),
        False,
    )

    allcpu = good_cert(K1, K2)
    allcpu["compute_by_buffer_type"] = {"CPU": 3712}
    case("htp_zero_compute", 0, good_rows(), good_cert(0, K1), allcpu, False)

    declared_ok = not p3.validate_stage_cert(good_cert(0, K1), 0, K1)
    n_pass = sum(1 for _, ok, *_ in CASES if ok) + (1 if declared_ok else 0)
    n_total = len(CASES) + 1
    for name, ok, certified, checks in CASES:
        if not ok:
            print(f"  [XFAIL-BROKEN] {name}: certified={certified} checks={checks}")
    print(f"declared GET_ROWS@CPU accepted: {declared_ok}")
    print(f"\n{n_pass}/{n_total} negative-gate tests pass")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
