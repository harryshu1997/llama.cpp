#!/usr/bin/env python3
"""Independent checker for E2A. Shares NO code with src/.

CP4 requires a checker that shares no resolver or aggregate decision code. This
file therefore imports nothing from `src/`: not the resolver, not the aggregate,
not e2a_canon, not E2's comparator. It re-implements strict loading, the
canonical form, the SHA-256 record digest, the all-pairs sum, and the
conservative arithmetic from scratch, against CONTRACT.md rather than against
the other implementation.

That independence is the entire value. If both implementations are wrong in the
same way, agreement proves nothing -- so nothing here is copied, and the two are
checked against each other on a corpus by tests/test_e2a.py. A disagreement is a
finding, not a merge conflict to be smoothed away.

The checker is deliberately SIMPLER than the resolver: it verifies arithmetic
inside an AggregateComparison against the per-pair contributions that record
itself carries. It does not re-resolve artifacts. So it answers exactly one
question -- "given these contributions, are these sums and arithmetic decisions
internally consistent?" -- and it is not, and must not be read as, a second
opinion on whether the contributions describe reality. The resolver owns that,
and this file cannot substitute for it.

WHY THIS FILE CANNOT PRINT A PHYSICAL LABEL
-------------------------------------------
It used to. `main()` ran check_aggregate() and printed whatever label came back,
exiting 0. Since check_aggregate() derives the label from the record's OWN
declared anchor_kind and its OWN declared contributions, a red team wrote twenty
lines of JSON by hand -- no bundle, no anchor, no artifacts, no privileges -- and
got `SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED` and exit 0 out of the very artifact
a reviewer runs.

That is E2's worst bug, exactly: a component that could be HANDED a conclusion. It
survived here because the file is honest about its scope in prose and the CLI
ignored the prose. Internal consistency is not evidence, and a checker that says
"PASS" cannot be trusted to be read alongside its own docstring.

So the arithmetic result and the physical label are separate things.
check_aggregate() never derives or returns a physical label. When internal
arithmetic would otherwise reach a physical-claim branch it refuses with E_LABEL.
The CLI reports only ARITHMETIC_CONSISTENT_ONLY, never a physical label, and exits
non-zero. Only the resolver, which re-resolves every artifact against an anchored
plan, can produce a label -- and on this host it refuses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

MAX_INT = 2 ** 53 - 1

LABEL_GPU = "GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED"
LABEL_SERVER = "SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED"
LABEL_FAIL = "RELIEF_FAIL"
LABEL_INVALID = "MEASUREMENT_INVALID"
SCOPE_LABEL = {"GPU_BOARD": LABEL_GPU, "SERVER_WALL": LABEL_SERVER}

# Independently restated from ANCHOR_AUDIT.md, not imported from src/anchors.py.
# A producer-selected inclusion proof shows one leaf, not every plan registered
# for an experiment identity. No currently modelled anchor proves exclusivity.
ENUMERABLE_ANCHORS = frozenset()
INDEPENDENT_ANCHORS = frozenset({"RFC3161_TSA", "TRANSPARENCY_LOG_INCLUSION"})

FORBIDDEN_NEEDLE = "SYSTEMENERGYSAVING"


class CheckError(ValueError):
    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def fail(code, message):
    raise CheckError(code, message)


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            fail("E_DUPLICATE_KEY", f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject(value):
    fail("E_SCHEMA", f"non-finite JSON constant: {value}")


def loads_strict(text):
    return json.loads(text, object_pairs_hook=_unique, parse_constant=_reject)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def record_digest(record):
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return digest(body)


def is_int(value):
    """type() identity, not isinstance. bool subclasses int; a float compares
    equal to one. Either would satisfy every check below."""
    return type(value) is int


def checked(value, what):
    if not is_int(value):
        fail("E_TYPE", f"{what} is {type(value).__name__} {value!r}, not an int")
    if value < 0:
        fail("E_TYPE", f"{what} is negative: {value}")
    if value > MAX_INT:
        fail("E_OVERFLOW", f"{what} exceeds {MAX_INT}")
    return value


def checked_signed(value, what):
    if not is_int(value):
        fail("E_TYPE", f"{what} is {type(value).__name__} {value!r}, not an int")
    if value < -MAX_INT or value > MAX_INT:
        fail("E_OVERFLOW", f"{what} is outside [-{MAX_INT}, {MAX_INT}]")
    return value


def normalized_text(document):
    text = canonical(document).decode("ascii").upper()
    return "".join(ch for ch in text if ch.isalnum())


def sum_pairs(contributions):
    """Elementwise integer sums. No quadrature, no mean, no trimming."""
    if not isinstance(contributions, list) or not contributions:
        fail("E_PAIRS", "no pair contributions")
    c = u_c = t = u_t = 0
    for index, item in enumerate(contributions):
        for field in ("control_energy_nj", "control_uncertainty_nj",
                      "treatment_energy_nj", "treatment_uncertainty_nj"):
            if field not in item:
                fail("E_PAIRS", f"pair {index} has no {field}")
        c += checked(item["control_energy_nj"], f"pair {index} control energy")
        u_c += checked(item["control_uncertainty_nj"],
                       f"pair {index} control uncertainty")
        t += checked(item["treatment_energy_nj"],
                     f"pair {index} treatment energy")
        u_t += checked(item["treatment_uncertainty_nj"],
                       f"pair {index} treatment uncertainty")
    for name, value in (("C", c), ("Uc", u_c), ("T", t), ("Ut", u_t)):
        if value > MAX_INT:
            fail("E_OVERFLOW", f"{name} sums to {value}, above {MAX_INT}")
    return c, u_c, t, u_t


def decide(c, u_c, t, u_t):
    """control_lower = C-Uc ; treatment_upper = T+Ut ; integer cross multiply."""
    control_lower = c - u_c
    treatment_upper = t + u_t
    if control_lower <= 0:
        fail("E_UNCERTAINTY",
             f"control lower bound is {control_lower}; an instrument whose error "
             f"swamps the signal cannot establish relief")
    relief = treatment_upper < control_lower
    # t <= 0.9 * c  <=>  t*10 <= c*9. Integer only: no division, no float.
    meets_ten = treatment_upper * 10 <= control_lower * 9
    return control_lower, treatment_upper, relief, meets_ten


def check_aggregate(record):
    """Verify AggregateComparison arithmetic, never a physical claim label."""
    if not isinstance(record, dict):
        fail("E_SCHEMA", "aggregate is not an object")
    if record.get("kind") != "AggregateComparison":
        fail("E_SCHEMA", f"unexpected kind {record.get('kind')!r}")
    if record.get("aggregate_method") != "SUM_ALL_PAIRS_V1":
        fail("E_METHOD", "aggregate does not declare SUM_ALL_PAIRS_V1")
    if FORBIDDEN_NEEDLE in normalized_text(record):
        fail("E_SYSTEM_CLAIM", "aggregate expresses a total-system claim")
    if record.get("record_sha256") != record_digest(record):
        fail("E_DIGEST", "aggregate digest does not cover its body")

    # This checker receives a self-authored aggregate, not resolved evidence.
    # Reject a supplied physical conclusion before another policy failure can
    # make label injection appear harmless.
    if record.get("result_label") in (LABEL_GPU, LABEL_SERVER):
        fail("E_LABEL", "physical claim requires resolved external evidence")

    contributions = record.get("pair_contributions")
    n_pairs = record.get("n_pairs")
    checked(n_pairs, "n_pairs")
    if not isinstance(contributions, list) or len(contributions) != n_pairs:
        fail("E_PAIRS",
             f"aggregate declares {n_pairs} pairs but carries "
             f"{len(contributions) if isinstance(contributions, list) else '?'} "
             f"contributions; a partial aggregate is not an all-pairs sum")
    if n_pairs < 8:
        fail("E_PAIRS", f"{n_pairs} pairs is below MIN_PAIRS=8")
    indices = []
    for index, item in enumerate(contributions):
        if not isinstance(item, dict):
            fail("E_PAIRS", f"pair {index} is not an object")
        pair_index = item.get("pair_index")
        checked(pair_index, f"pair {index} pair_index")
        indices.append(pair_index)
    if indices != list(range(n_pairs)):
        fail("E_PAIRS",
             f"pair_index values must be exactly 0..{n_pairs - 1}; a gap can hide "
             f"a dropped attempt")

    c, u_c, t, u_t = sum_pairs(contributions)
    for field, value in (("control_energy_sum_nj", c),
                         ("control_uncertainty_sum_nj", u_c),
                         ("treatment_energy_sum_nj", t),
                         ("treatment_uncertainty_sum_nj", u_t)):
        actual = checked(record.get(field), field)
        if actual != value:
            fail("E_SUM",
                 f"aggregate {field}={actual!r} but the contributions "
                 f"sum to {value}")
    control_lower, treatment_upper, relief, meets_ten = decide(c, u_c, t, u_t)
    if checked(record.get("control_lower_nj"),
               "control_lower_nj") != control_lower:
        fail("E_SUM", "control_lower_nj is not C - Uc")
    if checked(record.get("treatment_upper_nj"),
               "treatment_upper_nj") != treatment_upper:
        fail("E_SUM", "treatment_upper_nj is not T + Ut")
    if checked_signed(record.get("conservative_margin_nj"),
                      "conservative_margin_nj") != control_lower - treatment_upper:
        fail("E_SUM", "conservative_margin_nj is not control_lower - "
                      "treatment_upper")
    if type(record.get("conservative_relief")) is not bool:
        fail("E_TYPE", "conservative_relief is not a bool")
    if type(record.get("meets_ten_percent_gate")) is not bool:
        fail("E_TYPE", "meets_ten_percent_gate is not a bool")
    if record["conservative_relief"] != (relief and meets_ten):
        fail("E_DECISION",
             f"aggregate claims conservative_relief="
             f"{record['conservative_relief']} but the arithmetic gives "
             f"{relief and meets_ten}")
    if record["meets_ten_percent_gate"] != meets_ten:
        fail("E_DECISION", "meets_ten_percent_gate disagrees with the arithmetic")

    scope = record.get("scope")
    if scope not in SCOPE_LABEL:
        fail("E_SCOPE", f"unknown scope {scope!r}")
    delta = c - t
    if checked_signed(record.get("boundary_delta_nj"),
                      "boundary_delta_nj") != delta:
        fail("E_SUM", "boundary_delta_nj is not C - T")
    # A GPU board sensor cannot see how CPU, DRAM, fans or PSU losses moved, so a
    # board delta is not a server delta and cannot fund a phone budget.
    expected_server = delta if scope == "SERVER_WALL" else None
    actual_server = record.get("server_wall_delta_nj")
    if actual_server is not None:
        actual_server = checked_signed(actual_server, "server_wall_delta_nj")
    if actual_server != expected_server:
        fail("E_SCOPE",
             "server_wall_delta_nj must be null unless the scope is SERVER_WALL; "
             "a GPU board delta is not a server delta")
    actual_budget = record.get("phone_plus_external_break_even_budget_nj")
    if actual_budget is not None:
        actual_budget = checked_signed(
            actual_budget, "phone_plus_external_break_even_budget_nj")
    if actual_budget != expected_server:
        fail("E_SCOPE", "the break-even budget is only defined at SERVER_WALL")

    anchor_kind = record.get("anchor_kind")
    enumerable = anchor_kind in ENUMERABLE_ANCHORS
    independent = anchor_kind in INDEPENDENT_ANCHORS
    expected_property = ("ORDERING_AND_ENUMERABLE" if enumerable and independent
                         else "ORDERING_ONLY" if independent
                         else "NONE")
    if record.get("anchor_property") != expected_property:
        fail("E_ANCHOR_PROPERTY",
             f"aggregate declares anchor_property "
             f"{record.get('anchor_property')!r} but {anchor_kind} is "
             f"{expected_property}")
    if expected_property != "ORDERING_AND_ENUMERABLE":
        expected_label, expected_reason = LABEL_INVALID, "ANCHOR_NOT_ENUMERABLE"
    elif not (relief and meets_ten):
        expected_label, expected_reason = LABEL_FAIL, "NO_CONSERVATIVE_RELIEF"
    else:
        # Unreachable with the current capability table. Keep the fail-closed
        # branch for a future independently implemented enumerable mechanism.
        fail("E_LABEL", "physical claim requires resolved external evidence")
    if record.get("result_label") != expected_label:
        fail("E_LABEL",
             "aggregate result_label disagrees with arithmetic-only status")
    if record.get("reason_code") != expected_reason:
        fail("E_LABEL",
             "aggregate reason_code disagrees with arithmetic-only status")
    return expected_label, expected_reason


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Independent E2A aggregate arithmetic checker. Verifies that "
                    "a record's sums and decision follow from the contributions "
                    "it carries. NOT a measurement verdict: it "
                    "cannot see whether those contributions describe anything "
                    "real, and never emits a physical label.")
    parser.add_argument("--aggregate", required=True)
    args = parser.parse_args(argv)
    try:
        with open(args.aggregate, encoding="ascii") as handle:
            record = loads_strict(handle.read())
    except CheckError as exc:
        print(f"E2A_CHECK_FAIL: {exc.code}", file=sys.stderr)
        return 2
    except (OSError, ValueError, UnicodeDecodeError):
        print("E2A_CHECK_FAIL: E_LOAD", file=sys.stderr)
        return 2
    try:
        label, reason = check_aggregate(record)
    except CheckError as exc:
        print(f"E2A_CHECK_FAIL: {exc.code}", file=sys.stderr)
        return 1

    # The label is DELIBERATELY not reported here, and the exit code is
    # deliberately non-zero. A record can be perfectly self-consistent and
    # entirely fabricated: this file never resolved an artifact, never checked an
    # anchor, and never saw a plan. Printing "SERVER_RELIEF_PASS" on that basis
    # is precisely the fail-open a red team demonstrated with twenty lines of
    # hand-written JSON.
    print(json.dumps({
        "result": "ARITHMETIC_CONSISTENT_ONLY",
        "note": "internal consistency only; no artifact, anchor or plan was "
                "resolved. A physical label can come only from the resolver "
                "(src/aggregate.py evaluate), against an anchored plan.",
    }, sort_keys=True))
    _ = label
    _ = reason
    return 1


if __name__ == "__main__":
    sys.exit(main())
