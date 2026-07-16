#!/usr/bin/env python3
"""SUM_ALL_PAIRS_V1: the all-pairs aggregate evaluator. Standard library only.

Implements CONTRACT.md section 6.

    C  = sum(control_energy_nj)        Uc = sum(control_uncertainty_nj)
    T  = sum(treatment_energy_nj)      Ut = sum(treatment_uncertainty_nj)
    control_lower   = C - Uc
    treatment_upper = T + Ut
    relief          iff treatment_upper < control_lower
    ten percent     iff treatment_upper * 10 <= control_lower * 9

Every planned pair contributes. There is no trimming, no median, no retry
selection, no favourable-pair filter, and no partial aggregation. A missing,
failed, incorrect, or undrained attempt invalidates the ENTIRE result rather than
being dropped from it -- the whole point of an all-pairs rule is that the cohort
cannot be edited after the fact.

WHAT THIS MODULE REFUSES TO BE HANDED
-------------------------------------
No function here takes a label, a reason, a pair selection, a precomputed sum, or
a boolean. E2's worst bug was a function that accepted a `label` argument and
checked only that it was a member of the allowed set: importing the module and
calling it stamped a sealed SERVER_RELIEF_PASS onto junk whose treatment burned
1e15 nJ more. That function was this one's caller. Derive, never accept.

WHY THE ENERGIES ARE RE-INTEGRATED HERE
---------------------------------------
The per-pair energies are recomputed from the raw artifact bytes through E2's
frozen validator, not read from any MatchedComparison. `canon.seal()` is a public
function: a sealed comparison record proves only that its own body is internally
consistent, never that its numbers describe the bytes they claim to. A signed
number is never proof of itself.

WHY UNCERTAINTY IS SUMMED ELEMENTWISE
-------------------------------------
Uc and Ut are plain integer sums. Not quadrature, not isqrt(sum of squares), not
a standard error, not divided by N or by sqrt(N). This is the single
highest-motive bug in the whole checkpoint: at N=8, quadrature shrinks the
uncertainty by ~2.83x and is very often the only way relief appears at all. It
would also be wrong. NVML's +/-5 W is a VENDOR-STATED SYSTEMATIC BOUND on each
reading, not a random error with a mean of zero. Systematic error does not average
down over repetitions; if the sensor reads 3 W high, it reads 3 W high in all
eight pairs. Treating it as noise would be borrowing precision the instrument
never had.
"""

from __future__ import annotations

import pathlib

import anchors
import e2a_canon as canon
import resolver
from e2a_e2 import comparator as e2_comparator
from e2a_e2 import integrator as e2_integrator

LABEL_GPU = "GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED"
LABEL_SERVER = "SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED"
LABEL_FAIL = "RELIEF_FAIL"
LABEL_INVALID = "MEASUREMENT_INVALID"

SCOPE_LABEL = {"GPU_BOARD": LABEL_GPU, "SERVER_WALL": LABEL_SERVER}

# E2's forbidden-label guard, restated. E2A must never be able to express a
# total-system claim: phone, USB, charger and external-device energy are UNKNOWN
# and out of scope permanently. A positive server-side result is a BREAK-EVEN
# BUDGET, not a saving.
FORBIDDEN_LABEL = "SYSTEM_ENERGY_SAVING"
FORBIDDEN_NEEDLE = "SYSTEMENERGYSAVING"

# The canonical JSON contract caps integers at 2^53-1 so a JSON consumer cannot
# silently lose precision. Python ints do not overflow, so this is a CONTRACT
# bound rather than a machine one -- which is exactly why it has to be checked
# explicitly: nothing will crash to tell us.
MAX_INT = canon.MAX_INT


class AggregateError(ValueError):
    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def _fail(code, message):
    raise AggregateError(code, message)


def _normalized_text(document):
    text = canon.canonical(document).decode("ascii").upper()
    return "".join(ch for ch in text if ch.isalnum())


def _checked(value, what):
    """Integer type gate at the point of use, before any comparison or sum.

    Per RECORD, not on the total. sum([True, 900000.0, 3]) returns a clean int
    and the poison is already inside by the time anyone looks at the result.
    """
    if not canon.is_int(value):
        _fail("E_TYPE",
              f"{what} is {type(value).__name__} {value!r}; this aggregate is "
              f"integer-only, and a float compares equal to an integer while a "
              f"bool IS an integer, so either would pass every check downstream")
    if value < 0:
        _fail("E_TYPE", f"{what} is negative: {value}")
    if value > MAX_INT:
        _fail("E_OVERFLOW",
              f"{what} is {value}, above the canonical integer bound {MAX_INT}; "
              f"beyond this a JSON consumer loses precision silently")
    return value


def sum_all_pairs(contributions):
    """The frozen elementwise integer sum over EVERY contribution.

    Pure arithmetic plus its own type gate. It deliberately does NOT decide
    anything and does NOT emit a label: it is separated from the decision so the
    frozen sum can be tested on small hand-checkable inputs, exactly as E2 split
    integrate() from gate_window(). Calling it directly buys nothing -- it
    returns four integers, and only evaluate() can turn integers into a verdict.
    """
    if not isinstance(contributions, list) or not contributions:
        _fail("E_PAIRS", "an aggregate needs at least one pair contribution")
    totals = {"control_energy_sum_nj": 0, "control_uncertainty_sum_nj": 0,
              "treatment_energy_sum_nj": 0, "treatment_uncertainty_sum_nj": 0}
    field_map = (
        ("control_energy_nj", "control_energy_sum_nj"),
        ("control_uncertainty_nj", "control_uncertainty_sum_nj"),
        ("treatment_energy_nj", "treatment_energy_sum_nj"),
        ("treatment_uncertainty_nj", "treatment_uncertainty_sum_nj"),
    )
    for index, item in enumerate(contributions):
        for source, target in field_map:
            value = item.get(source, canon.MISSING)
            if value is canon.MISSING:
                _fail("E_UNCERTAINTY" if "uncertainty" in source else "E_PAIRS",
                      f"pair {index} has no {source}; uncertainty is never "
                      f"optional and never dropped from the decision")
            totals[target] += _checked(value, f"pair {index}.{source}")
    for key, value in totals.items():
        if value > MAX_INT:
            _fail("E_OVERFLOW",
                  f"{key} sums to {value}, above the canonical integer bound "
                  f"{MAX_INT}")
    return totals


def decide_aggregate(totals, scope):
    """The conservative decision over the sums, via E2's frozen arithmetic.

    Routed through E2's decide() rather than reimplemented inline. Two reasons:
    the pair-level and aggregate-level rules must be the same rule, and an inline
    re-derivation is where `c * 9 // 10` or `t < c * 0.9` creeps in and quietly
    moves the boundary.
    """
    control = {"energy_nj": totals["control_energy_sum_nj"],
               "uncertainty_nj": totals["control_uncertainty_sum_nj"]}
    treatment = {"energy_nj": totals["treatment_energy_sum_nj"],
                 "uncertainty_nj": totals["treatment_uncertainty_sum_nj"]}
    try:
        return e2_comparator.decide(control, treatment, scope)
    except e2_comparator.ComparisonError as exc:
        _fail(exc.code, str(exc).split(": ", 1)[1])


def resolve_bundle(bundle_path):
    """Resolve one bundle under a single immutable artifact view."""
    with resolver.resolution_context():
        return _resolve_bundle_in_context(bundle_path)


def _resolve_bundle_in_context(bundle_path):
    """Resolve EVERYTHING the plan declares. Raises unless the bundle is whole.

    Returns the resolved bundle. There is no partial success: if any planned
    slot, record, anchor, request, or action fails to resolve, this raises and
    the aggregate is never reached. CP2's rule -- stop before aggregation if
    anything is unresolved -- is enforced by control flow, not by a flag.
    """
    trusted_root = resolver.derive_trusted_root(bundle_path)
    _resolved, data = resolver.read_once(
        pathlib.Path(bundle_path).name,
        canon.sha256_bytes(pathlib.Path(bundle_path).read_bytes()),
        trusted_root)
    index = resolver.parse_once(data, "bundle index")
    required = {"schema", "plan_path", "plan_sha256", "plan_anchor_path",
                "plan_anchor_sha256", "ledger_path", "ledger_sha256",
                "ledger_close_path", "ledger_close_sha256", "manifest_path",
                "manifest_sha256", "slots"}
    if not isinstance(index, dict) or set(index) != required:
        _fail("E_SCHEMA", "bundle index does not match the e2a.bundle.v2 contract")
    if index["schema"] != "e2a.bundle.v2":
        _fail("E_SCHEMA", f"unknown bundle schema {index['schema']!r}")

    _r, plan_data = resolver.read_once(index["plan_path"], index["plan_sha256"],
                                       trusted_root)
    plan = resolver.parse_once(plan_data, "plan")
    resolver.resolve_plan(plan, trusted_root)

    _r, anchor_data = resolver.read_once(index["plan_anchor_path"],
                                         index["plan_anchor_sha256"],
                                         trusted_root)
    plan_anchor = resolver.parse_once(anchor_data, "plan anchor")
    resolver.bind_plan_anchor(plan_anchor, plan)

    _r, ledger_data = resolver.read_once(index["ledger_path"],
                                         index["ledger_sha256"], trusted_root)
    ledger = resolver.parse_once(ledger_data, "ledger")
    terminal = resolver.resolve_ledger(ledger, plan, plan_anchor)

    _r, manifest_data = resolver.read_once(index["manifest_path"],
                                           index["manifest_sha256"],
                                           trusted_root)
    manifest = resolver.parse_once(manifest_data, "request set manifest")
    resolver.resolve_manifest(manifest, plan)

    contributions, resolved = _resolve_slots(index, plan, terminal, manifest,
                                             trusted_root)

    # THE ANCHOR GATE, applied LAST of the resolution steps and before any label.
    #
    # The ordering is deliberate and was changed after an earlier revision put it
    # first. Failing fast on anchors was cheaper, but it MASKED every structural
    # check behind it: at CLI level a broken ledger, a truncated cohort, a forged
    # energy and a missing slot all reported E_ANCHOR_UNENUMERABLE, so no negative
    # could distinguish a working check from a deleted one. Those checks only
    # become load-bearing on the day an enumerable anchor exists -- which is
    # precisely the day nobody would discover they had rotted.
    #
    # Running it here also sharpens the finding. When this raises, it raises about
    # a bundle that is otherwise impeccable: every slot resolved, every energy
    # re-integrated, every request matched. The refusal is not "your evidence is
    # broken". It is "your evidence is fine and still cannot support this claim,
    # because the cohort was never externally committed".
    _r, close_data = resolver.read_once(index["ledger_close_path"],
                                        index["ledger_close_sha256"],
                                        trusted_root)
    close = resolver.parse_once(close_data, "ledger close")
    resolver.bind_close_anchor(close, plan, plan_anchor, ledger)
    resolver.resolve_plan_anchor(plan_anchor, plan, trusted_root)
    resolver.resolve_close_anchor(close, plan, plan_anchor, ledger, trusted_root)

    # The anchor gate is host-level: no bundle can fix it. Provenance is
    # bundle-level. Reporting the unfixable blocker first tells a reader that
    # producing better evidence would not help, which is the more useful fact.
    check_provenance(resolved)
    return {"plan": plan, "plan_anchor": plan_anchor, "ledger": ledger,
            "ledger_close": close, "manifest": manifest,
            "contributions": contributions, "trusted_root": trusted_root}


PLAN_TIMELINE_FIELDS = (
    "workload_digest", "trace_digest", "slo_policy_digest",
    "source_revision", "build_revision", "scope", "instrument_kind",
    "instrument_identity", "device_identity",
)

PLAN_TIMELINE_SET_FIELDS = ("board_uuids", "included_rails", "excluded_rails")


def _bind_timeline_to_plan(timeline, plan, plan_slot, slot_index):
    for field in PLAN_TIMELINE_FIELDS:
        if timeline[field] != plan[field]:
            _fail("E_PLAN_TIMELINE_BINDING",
                  f"slot {slot_index} timeline {field}={timeline[field]!r}, "
                  f"plan pins {plan[field]!r}")
    for field in PLAN_TIMELINE_SET_FIELDS:
        if set(timeline[field]) != set(plan[field]):
            _fail("E_PLAN_TIMELINE_BINDING",
                  f"slot {slot_index} timeline {field} differs from the plan")
    expected_policy = (plan["policy_digest_control"]
                       if plan_slot["role"] ==
                       "OPTIMIZED_SERVER_ONLY_CONTROL"
                       else plan["policy_digest_treatment"])
    if timeline["policy_digest"] != expected_policy:
        _fail("E_PLAN_TIMELINE_BINDING",
              f"slot {slot_index} ran a policy the plan did not pin")


def _check_global_slots(resolved):
    windows = []
    unique_fields = {
        "record_sha256": set(),
        "raw_artifact_sha256": set(),
        "execution_artifact_sha256": set(),
        "paid_payload_sha256": set(),
    }
    for slot_index in sorted(resolved):
        timeline = resolved[slot_index][0]
        windows.append((timeline["window_start_us"], timeline["window_end_us"],
                        slot_index))
        for field, seen in unique_fields.items():
            value = timeline[field]
            if value in seen:
                _fail("E_EVIDENCE_REUSE",
                      f"slot {slot_index} reuses timeline {field}={value}")
            seen.add(value)
    windows.sort()
    for previous, current in zip(windows, windows[1:]):
        if current[0] < previous[1]:
            _fail("E_WINDOW_OVERLAP",
                  f"slots {previous[2]} and {current[2]} overlap in time")


def _resolve_slots(index, plan, terminal, manifest, trusted_root):
    """Resolve all 2N slots into per-pair contributions.

    NOTE the absence of a try/except around the per-slot work. A refusal must
    never become an omission: `except: continue` inside an all-pairs loop turns
    every failure into a silently smaller, and systematically more favourable,
    cohort.
    """
    slots = {item["slot_index"]: item for item in index["slots"]}
    if set(slots) != set(range(2 * plan["n_pairs"])):
        _fail("E_SLOT_UNCOVERED",
              f"the bundle supplies slots {sorted(slots)[:4]}... but the plan "
              f"declares {2 * plan['n_pairs']}; every planned slot must resolve")

    resolved = {}
    for slot_index in sorted(slots):
        entry = slots[slot_index]
        plan_slot = plan["slots"][slot_index]
        owner_prefix = ("slot", slot_index, plan_slot["run_nonce"])
        resolver.claim_evidence_path(
            entry["timeline_path"], owner_prefix + ("timeline",), trusted_root)
        _r, tl_data = resolver.read_once(entry["timeline_path"],
                                         entry["timeline_sha256"], trusted_root)
        timeline = resolver.parse_once(tl_data, f"timeline slot {slot_index}")

        for kind in ("raw", "execution"):
            path_field = f"{kind}_artifact_path"
            digest_field = f"{kind}_artifact_sha256"
            resolver.claim_evidence_path(
                timeline[path_field], owner_prefix + (kind,), trusted_root)
            resolver.read_once(timeline[path_field], timeline[digest_field],
                               trusted_root)

        # E2's frozen timeline validation: re-integrates the raw artifact,
        # re-derives every declared statistic, and rejects on any mismatch.
        failures = e2_comparator.validate_timeline(timeline, trusted_root)
        if failures:
            _fail("E_TIMELINE_INVALID",
                  f"slot {slot_index} timeline {timeline.get('timeline_id')!r}: "
                  f"{failures[0]}")
        if timeline["run_nonce"] != plan_slot["run_nonce"]:
            _fail("E_SLOT_BINDING",
                  f"slot {slot_index} ran nonce {timeline['run_nonce']!r} but the "
                  f"anchored plan pinned {plan_slot['run_nonce']!r}")
        if timeline["role"] != plan_slot["role"]:
            _fail("E_ORDER",
                  f"slot {slot_index} ran role {timeline['role']} but the plan "
                  f"pinned {plan_slot['role']}")
        _bind_timeline_to_plan(timeline, plan, plan_slot, slot_index)
        ledger_group = terminal[slot_index]
        ledger_entry = ledger_group["end"]
        if ledger_entry["timeline_record_sha256"] != timeline["record_sha256"]:
            _fail("E_SLOT_BINDING",
                  f"slot {slot_index} ledger binds timeline record "
                  f"{ledger_entry['timeline_record_sha256']} but the resolved "
                  f"timeline digests to {timeline['record_sha256']}")

        resolver.claim_evidence_path(
            entry["lifecycle_path"], owner_prefix + ("lifecycle",), trusted_root)
        _r, lc_data = resolver.read_once(entry["lifecycle_path"],
                                         entry["lifecycle_sha256"], trusted_root)
        lifecycle = resolver.parse_once(lc_data, f"lifecycle slot {slot_index}")
        if ledger_entry["lifecycle_record_sha256"] != \
                lifecycle.get("record_sha256"):
            _fail("E_SLOT_BINDING",
                  f"slot {slot_index} ledger binds another lifecycle record")

        resolver.claim_evidence_path(
            entry["outcomes_path"], owner_prefix + ("outcomes",), trusted_root)
        _r, ro_data = resolver.read_once(entry["outcomes_path"],
                                         entry["outcomes_sha256"], trusted_root)
        outcomes = resolver.parse_once(ro_data, f"outcomes slot {slot_index}")
        if ledger_entry["request_outcome_record_sha256"] != \
                outcomes.get("record_sha256"):
            _fail("E_SLOT_BINDING",
                  f"slot {slot_index} ledger binds another outcome record")
        resolved_outcomes = resolver.resolve_requests(
            manifest, outcomes, timeline, trusted_root,
            f"outcomes slot {slot_index}", plan_slot["pair_index"])
        resolver.resolve_lifecycle(
            lifecycle, timeline, plan_slot, resolved_outcomes)
        ledger_start = ledger_group["start"]
        if ledger_start["clock_epoch_id"] != timeline["clock_epoch_id"] or \
                ledger_entry["clock_epoch_id"] != timeline["clock_epoch_id"]:
            _fail("E_CLOCK_EPOCH",
                  f"slot {slot_index} ledger and timeline use different clocks")
        expected_times = {
            "window_start_us": timeline["window_start_us"],
            "window_end_us": timeline["window_end_us"],
            "drain_acknowledged_us": lifecycle["drain_acknowledged_us"],
        }
        for field, expected in expected_times.items():
            if ledger_entry[field] != expected:
                _fail("E_LEDGER_TIME_BINDING",
                      f"slot {slot_index} ledger {field}="
                      f"{ledger_entry[field]}, expected {expected}")
        if ledger_start["window_start_us"] != timeline["window_start_us"]:
            _fail("E_LEDGER_TIME_BINDING",
                  f"slot {slot_index} ledger START does not bind window start")
        resolved[slot_index] = (timeline, resolved_outcomes, lifecycle)

    _check_global_slots(resolved)

    contributions = []
    for pair_index in range(plan["n_pairs"]):
        a, b = resolved[2 * pair_index], resolved[2 * pair_index + 1]
        by_role = {}
        for timeline, outcomes, lifecycle in (a, b):
            by_role[timeline["role"]] = (timeline, outcomes, lifecycle)
        if set(by_role) != {"OPTIMIZED_SERVER_ONLY_CONTROL", "Q_PIM_TREATMENT"}:
            _fail("E_ROLE", f"pair {pair_index} is not one control and one "
                            f"treatment")
        control, c_out, _cl = by_role["OPTIMIZED_SERVER_ONLY_CONTROL"]
        treatment, t_out, _tl = by_role["Q_PIM_TREATMENT"]

        # Work is resolved BEFORE energy is looked at. Order is deliberate: a
        # number computed over an unvalidated cohort has already misled anyone
        # who stopped reading at the number.
        resolver.check_same_work(c_out, t_out, manifest)

        failures = e2_comparator.check_match(control, treatment)
        if failures:
            _fail("E_NOT_MATCHED", f"pair {pair_index}: {failures[0]}")

        # first_executed_role is DERIVED from the validated windows, never read
        # from a record. A self-declared field describing the producer's own
        # ordering discipline is worth nothing.
        first = (control["role"]
                 if control["window_start_us"] <= treatment["window_start_us"]
                 else treatment["role"])
        expected_first = ("OPTIMIZED_SERVER_ONLY_CONTROL" if pair_index % 2 == 0
                          else "Q_PIM_TREATMENT")
        if first != expected_first:
            _fail("E_ORDER",
                  f"pair {pair_index} actually ran {first} first, but the ABBA "
                  f"rotation the plan committed to requires {expected_first}; a "
                  f"warm/cold bias would be indistinguishable from a policy "
                  f"effect")
        contributions.append({
            "pair_index": pair_index,
            "control_timeline_id": control["timeline_id"],
            "control_record_sha256": control["record_sha256"],
            "treatment_timeline_id": treatment["timeline_id"],
            "treatment_record_sha256": treatment["record_sha256"],
            "control_energy_nj": control["energy_nj"],
            "control_uncertainty_nj": control["uncertainty_nj"],
            "treatment_energy_nj": treatment["energy_nj"],
            "treatment_uncertainty_nj": treatment["uncertainty_nj"],
            "first_executed_role": first,
        })
    if len(contributions) != plan["n_pairs"]:
        _fail("E_PAIRS",
              f"aggregated {len(contributions)} pairs but the plan declares "
              f"{plan['n_pairs']}; a partial aggregate is not an all-pairs sum")
    return contributions, resolved


def check_provenance(resolved):
    """Synthetic evidence exercises mechanics and carries no physical result.

    A POLICY gate, kept with the anchor gate rather than buried in the slot loop:
    both answer "what can this evidence support?", as opposed to the structural
    checks, which answer "is this evidence internally sound?". Mixing the two
    orders made a synthetic bundle report a provenance refusal before the
    structural work had finished, which hid whether the structural work was
    right.
    """
    for slot_index in sorted(resolved):
        timeline = resolved[slot_index][0]
        if timeline["provenance"] == "SYNTHETIC" or \
                timeline["instrument_kind"] == "SYNTHETIC":
            _fail("E_PROVENANCE",
                  f"slot {slot_index} ({timeline['timeline_id']}) is synthetic "
                  f"({timeline['provenance']}/{timeline['instrument_kind']}); "
                  f"synthetic timelines exercise mechanics and can never carry a "
                  f"physical result")
    return True


def derive_label(bundle, totals, detail):
    """Derive the label from resolved evidence. There is no label parameter."""
    plan = bundle["plan"]
    scope = plan["scope"]
    prop = anchors.anchor_property(bundle["plan_anchor"]["anchor_kind"])
    if prop != anchors.REQUIRED_PROPERTY:
        return LABEL_INVALID, "ANCHOR_NOT_ENUMERABLE"
    if not detail["relief"]:
        return LABEL_FAIL, "NO_CONSERVATIVE_RELIEF"
    return SCOPE_LABEL[scope], "ALL_PAIRS_CONSERVATIVE_RELIEF"


def evaluate(bundle_path):
    """Resolve, sum, decide, and seal. The only entry point that emits a label.

    On this host it cannot return anything but MEASUREMENT_INVALID, because
    resolve_bundle raises at the anchor gate long before the arithmetic. The
    arithmetic below is nonetheless real and tested: when an enumerable anchor
    exists, this is the code that will run.
    """
    bundle = resolve_bundle(bundle_path)
    plan = bundle["plan"]
    totals = sum_all_pairs(bundle["contributions"])
    detail = decide_aggregate(totals, plan["scope"])
    label, reason = derive_label(bundle, totals, detail)

    record = {
        "schema_version": 2,
        "kind": "AggregateComparison",
        "aggregate_id": "agg." + canon.digest({
            "plan": plan["record_sha256"],
            "ledger_head": bundle["ledger"]["head_sha256"],
        }),
        "chain_id": plan["chain_id"],
        "set_id": plan["set_id"],
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["record_sha256"],
        "plan_anchor_record_sha256": bundle["plan_anchor"]["record_sha256"],
        "ledger_close_record_sha256": bundle["ledger_close"]["record_sha256"],
        "ledger_head_sha256": bundle["ledger"]["head_sha256"],
        "request_set_manifest_sha256": bundle["manifest"]["record_sha256"],
        "aggregate_method": "SUM_ALL_PAIRS_V1",
        "anchor_kind": bundle["plan_anchor"]["anchor_kind"],
        "anchor_property": anchors.anchor_property(
            bundle["plan_anchor"]["anchor_kind"]),
        "scope": plan["scope"],
        "instrument_kind": plan["instrument_kind"],
        "n_pairs": plan["n_pairs"],
        "pair_contributions": bundle["contributions"],
        "control_energy_sum_nj": totals["control_energy_sum_nj"],
        "control_uncertainty_sum_nj": totals["control_uncertainty_sum_nj"],
        "treatment_energy_sum_nj": totals["treatment_energy_sum_nj"],
        "treatment_uncertainty_sum_nj": totals["treatment_uncertainty_sum_nj"],
        "control_lower_nj": detail["control_lower_nj"],
        "treatment_upper_nj": detail["treatment_upper_nj"],
        "conservative_margin_nj": detail["conservative_margin_nj"],
        "boundary_delta_nj": detail["boundary_delta_nj"],
        "server_wall_delta_nj": detail["server_wall_delta_nj"],
        "phone_plus_external_break_even_budget_nj":
            detail["phone_plus_external_break_even_budget_nj"],
        "conservative_relief": bool(detail["relief"]),
        "meets_ten_percent_gate": bool(detail["meets_ten_percent_gate"]),
        "result_label": label,
        "reason_code": reason,
        "record_sha256": "",
    }
    canon.seal(record)
    if FORBIDDEN_NEEDLE in _normalized_text(record):
        _fail("E_SYSTEM_CLAIM",
              f"aggregate mentions {FORBIDDEN_LABEL}; E2A cannot express a "
              f"total-system claim -- phone, USB, charger and external-device "
              f"energy are unknown, so a server-side result is a break-even "
              f"budget and never a saving")
    return record


# E2's guard keys on {RealizedTimeline, MatchedComparison, RepetitionSet}. An
# AggregateComparison is not in that set, so calling E2's function on one was a
# NO-OP -- the guard against re-entering the additive solver did nothing for the
# most aggregated record E2A produces. That is the exact bug E2's own docstring
# says it fixed ("an earlier revision keyed on RealizedTimeline alone, so a
# MatchedComparison or a whole RepetitionSet -- which are MORE aggregated, not
# less -- passed straight through"), recurring one kind later.
E2A_AGGREGATE_KINDS = frozenset({"AggregateComparison"})


def assert_not_additive_input(record):
    """CONTRACT.md section 0, made executable for E2A's own record kind.

    A BOUNDARY guard, to be called by anything about to hand a record to E1's
    additive per-device solver. It is deliberately NOT called by evaluate() on
    its own output: "assert my aggregate is not an additive input" is either a
    no-op or an unconditional self-refusal, and it was the former -- which is how
    the missing kind went unnoticed. A tripwire is worth something only where the
    wire is, and the wire is E1's front door.

    E2A never calls E1's solver, so this exists for the code that one day might.
    """
    if isinstance(record, dict) and record.get("kind") in E2A_AGGREGATE_KINDS:
        _fail("E_ADDITIVE_REUSE",
              f"AggregateComparison {record.get('aggregate_id')!r} is a SUM over "
              f"aggregate boundary measurements and must never enter E1's "
              f"additive per-device solver; doing so would double-count shared "
              f"power and attribute it to individual routes")
    # Defer to E2's guard for the kinds it owns, so there is one rule per kind
    # rather than two that can drift.
    return e2_comparator.assert_not_additive_input(record)


def validate_aggregate(record, bundle_path):
    """Re-derive an AggregateComparison and compare it field by field.

    This is what makes the label unfakeable rather than merely typed: a
    hand-written record with result_label SERVER_RELIEF_PASS is schema-valid and
    digest-valid, and dies here because the re-derived label disagrees.
    """
    resolver.gate_record("aggregate", record, "aggregate")
    if record["aggregate_method"] != "SUM_ALL_PAIRS_V1":
        _fail("E_METHOD", "aggregate does not declare SUM_ALL_PAIRS_V1")
    expected = evaluate(bundle_path)
    for field in sorted(expected):
        if field == "record_sha256":
            continue
        if record.get(field, canon.MISSING) != expected[field]:
            _fail("E_INCOHERENT",
                  f"aggregate {field}={record.get(field)!r} does not match the "
                  f"re-derived value {expected[field]!r}")
    if record["record_sha256"] != expected["record_sha256"]:
        _fail("E_DIGEST", "aggregate digest does not match the re-derived record")
    return True
