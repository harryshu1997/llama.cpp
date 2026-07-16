#!/usr/bin/env python3
"""Frozen energy-boundary semantics (EVIDENCE_CONTRACT.md section 4).

The three measurement boundaries are NOT interchangeable, and this module is the
one place that decides what a set of bound power records is allowed to claim.

    GPU_BOARD   GPU board rails only
    SERVER_WALL host/server wall
    TOTAL_WALL  synchronized complete boundary

E1 authorizes no physical claim from any of them. Its solver adds per-route and
per-device terms, while wall measurements are aggregate timelines. The scope
ranking and result labels below are frozen for a later typed matched-comparison
record; PHYSICAL_CLAIMS_ENABLED remains false until that record exists.

Nothing here measures anything. The equations below are frozen for a later,
separately authorized checkpoint:

    delta_E_server = E_server_control - E_server_qpim
    phone_plus_external_break_even_budget = delta_E_server

A positive delta_E_server is NOT a total-system energy saving. It is a budget:
the excluded phone, USB, charger, and relay energy may consume at most that much
before the system stops breaking even. Those terms are unmeasured, so the sign of
the total remains unknown.
"""

from __future__ import annotations

CLAIM_NONE = "NONE_MECHANICS_ONLY"
CLAIM_GPU_BOARD = "GPU_BOARD_RELIEF"
CLAIM_SERVER = "SERVER_RELIEF"
CLAIM_SYSTEM = "SYSTEM_ENERGY_SAVING"
PHYSICAL_CLAIMS_ENABLED = False
MAX_INT = 2 ** 53 - 1

# Strictly increasing strength. A claim is permitted only if every bound power
# record supports at least it.
CLAIM_RANK = {CLAIM_NONE: 0, CLAIM_GPU_BOARD: 1, CLAIM_SERVER: 2, CLAIM_SYSTEM: 3}

SCOPE_MAX_CLAIM = {
    "GPU_BOARD": CLAIM_GPU_BOARD,
    "SERVER_WALL": CLAIM_SERVER,
    # TOTAL_WALL is intentionally not bindable to the additive per-device
    # solver. A whole-wall meter produces one timeline, not one additive power
    # record per device. SYSTEM_ENERGY_SAVING requires a separate matched
    # control/treatment comparison record in a later gate.
    "TOTAL_WALL": CLAIM_NONE,
}

RESULT_GPU_BOARD_BLOCKED = "GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED"
RESULT_SERVER_BLOCKED = "SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED"
RESULT_RELIEF_FAIL = "RELIEF_FAIL"
RESULT_MEASUREMENT_INVALID = "MEASUREMENT_INVALID"


def delta_e_server(e_server_control, e_server_qpim):
    """Frozen equation. Integer nJ in, integer nJ out. Sign is meaningful."""
    if type(e_server_control) is not int or type(e_server_qpim) is not int:
        raise ValueError("server energies must be integers in nJ")
    return e_server_control - e_server_qpim


def phone_plus_external_break_even_budget(e_server_control, e_server_qpim):
    """The frozen budget identity. Deliberately equal to delta_E_server."""
    return delta_e_server(e_server_control, e_server_qpim)


def classify_energy_claim(inst, bundle, records, bound_targets):
    """Strongest claim the bound power evidence can support. Fails closed.

    Returns (claim, reasons). A MECHANICS_ONLY bundle can never support any
    physical claim, no matter how well formed it is.
    """
    reasons = []
    if bundle.get("provenance") != "MEASURED" or \
            inst["evidence"]["scope"] != "MEASURED":
        return CLAIM_NONE, ["bundle or instance is MECHANICS_ONLY; no physical "
                            "energy claim is authorized"]
    if not PHYSICAL_CLAIMS_ENABLED:
        return CLAIM_NONE, [
            "E1 physical claims are disabled: additive solver terms cannot "
            "represent aggregate wall timelines, and no typed matched "
            "control/treatment comparison exists"]

    # Range over EVERY bound record that contributes an energy term, not only the
    # PowerProfiles. A BoundaryProfile's energy_nj flows into energy.total_nj just
    # as device power does, so a GPU-board-scoped transfer energy inside a total
    # claim is exactly the boundary confusion this module exists to prevent.
    scopes = set()
    for target, binding in sorted(bound_targets.items()):
        record = records.get(binding.get("record_id"))
        if record is None:
            continue
        if record.get("kind") == "PowerProfile":
            scope = record.get("scope")
        elif record.get("kind") == "BoundaryProfile" and \
                target.endswith("extra_energy_nj"):
            scope = record.get("energy_scope")
            if scope == "NONE":
                return CLAIM_NONE, [
                    f"{target} takes energy from boundary {record.get('record_id')} "
                    f"measured at no boundary; an unmeasured energy term cannot "
                    f"support any claim"]
        else:
            continue
        if scope not in SCOPE_MAX_CLAIM:
            return CLAIM_NONE, [f"record {record.get('record_id')} has unknown energy "
                                f"scope {scope!r}"]
        scopes.add(scope)

    if not scopes:
        return CLAIM_NONE, ["no power evidence is bound"]

    claim = min(scopes, key=lambda s: CLAIM_RANK[SCOPE_MAX_CLAIM[s]])
    claim = SCOPE_MAX_CLAIM[claim]

    if claim == CLAIM_SYSTEM:
        # A total-system claim needs every device accounted for, including every
        # phone. An unmeasured phone is UNKNOWN, and UNKNOWN is not zero.
        for name, device in sorted(inst["devices"].items()):
            binding = bound_targets.get(f"devices.{name}.active_mw")
            if binding is None:
                return CLAIM_NONE, [f"device {name} has no bound power evidence"]
            record = records.get(binding.get("record_id"))
            if record is None or record.get("status") != "PASS":
                return CLAIM_NONE, [f"device {name} has no PASS power evidence"]
            if record.get("scope") != "TOTAL_WALL":
                reasons.append(
                    f"device {name} is measured at {record.get('scope')}, so the "
                    f"boundary is not complete")
                return CLAIM_SERVER if record.get("scope") == "SERVER_WALL" \
                    else CLAIM_GPU_BOARD, reasons
    return claim, reasons


def validate_relief_comparison(control, treatment):
    """Frozen validity contract for a later control/treatment energy comparison.

    Not wired to any measurement in this checkpoint. It exists so the semantics
    are frozen now rather than negotiated when results are on the table.
    """
    failures = []
    required = {"completed_work", "slo_outcomes", "window_us", "scope"}
    scopes = {"GPU_BOARD", "SERVER_WALL", "TOTAL_WALL"}
    for label, record in (("control", control), ("treatment", treatment)):
        if not isinstance(record, dict):
            failures.append(f"E_SCHEMA: {label} comparison is not an object")
            continue
        extra = sorted(set(record) - required)
        missing = sorted(required - set(record))
        if missing:
            failures.append(f"E_SCHEMA: {label} comparison is missing {missing}")
        if extra:
            failures.append(f"E_SCHEMA: {label} comparison has unknown fields {extra}")
        for field in ("completed_work", "window_us"):
            value = record.get(field)
            if type(value) is not int or value <= 0 or value > MAX_INT:
                failures.append(
                    f"E_SCHEMA: {label}.{field} must be a positive in-range integer")
        if record.get("scope") not in scopes:
            failures.append(
                f"E_SCOPE: {label} comparison has invalid scope "
                f"{record.get('scope')!r}")
        outcomes = record.get("slo_outcomes")
        outcome_fields = {"met", "tardy", "rejected"}
        if not isinstance(outcomes, dict) or set(outcomes) != outcome_fields or \
                any(type(value) is not int or value < 0 or value > MAX_INT
                    for value in outcomes.values()):
            failures.append(
                f"E_SCHEMA: {label}.slo_outcomes must contain exactly met, tardy, "
                f"and rejected non-negative integer counts")
        elif type(record.get("completed_work")) is int and \
                outcomes["met"] + outcomes["tardy"] != record["completed_work"]:
            failures.append(
                f"E_WORK_MISMATCH: {label} completed_work does not equal "
                f"met+tardy outcomes")
    if failures:
        return failures
    if control.get("completed_work") != treatment.get("completed_work"):
        failures.append(
            f"E_WORK_MISMATCH: control completed {control.get('completed_work')!r} "
            f"but treatment completed {treatment.get('completed_work')!r}; energy "
            f"per unit of different work is not a comparison")
    if control.get("slo_outcomes") != treatment.get("slo_outcomes"):
        failures.append(
            f"E_SLO_MISMATCH: control SLO outcomes {control.get('slo_outcomes')!r} "
            f"differ from treatment {treatment.get('slo_outcomes')!r}; buying energy "
            f"with missed deadlines is not relief")
    if control.get("window_us") != treatment.get("window_us"):
        failures.append(
            f"E_WINDOW_MISMATCH: control window {control.get('window_us')!r} us "
            f"differs from treatment {treatment.get('window_us')!r} us")
    if control.get("scope") != treatment.get("scope"):
        failures.append(
            f"E_SCOPE: control boundary {control.get('scope')!r} differs from "
            f"treatment {treatment.get('scope')!r}; the two are not the same "
            f"measurement")
    return failures


def relief_label(claim, delta_nj, valid):
    """Map a claim plus a measured delta onto exactly one frozen result label."""
    if type(valid) is not bool or not valid:
        return RESULT_MEASUREMENT_INVALID
    if type(delta_nj) is not int:
        return RESULT_MEASUREMENT_INVALID
    if delta_nj < -MAX_INT or delta_nj > MAX_INT:
        return RESULT_MEASUREMENT_INVALID
    if delta_nj <= 0:
        return RESULT_RELIEF_FAIL
    if claim == CLAIM_GPU_BOARD:
        return RESULT_GPU_BOARD_BLOCKED
    if claim == CLAIM_SERVER:
        return RESULT_SERVER_BLOCKED
    if claim == CLAIM_SYSTEM:
        return RESULT_MEASUREMENT_INVALID
    return RESULT_MEASUREMENT_INVALID
