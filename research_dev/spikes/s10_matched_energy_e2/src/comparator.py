#!/usr/bin/env python3
"""Strict E2 validator and matched-timeline comparator. Standard library only.

Implements CONTRACT.md. Fails closed everywhere: there is no default, no
best-effort, and no warning-and-continue.

E2 imports NO E1 decision logic. It reuses only canonical JSON and strict loading
through e2_canon. An aggregate timeline must never re-enter the additive solver
(CONTRACT.md section 0); `assert_not_additive_input` exists so that rule is
executable rather than aspirational.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import e2_canon as canon
import integrator

SYSTEM_PYTHON = pathlib.Path("/usr/bin/python3")
SCHEMA_WORKER = pathlib.Path(__file__).with_name("e2_schema_gate.py")

MIN_PAIRS = integrator.MIN_PAIRS

LABEL_GPU = "GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED"
LABEL_SERVER = "SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED"
LABEL_FAIL = "RELIEF_FAIL"
LABEL_INVALID = "MEASUREMENT_INVALID"
# MatchedComparison v1 is pair-local and diagnostic-only. Physical labels belong
# to a future aggregate-result version that resolves every predeclared run.
ALLOWED_LABELS = frozenset({LABEL_INVALID})

# The label E2 must never be able to emit. Kept as a constant purely so the guard
# that hunts for it is greppable and testable.
FORBIDDEN_LABEL = "SYSTEM_ENERGY_SAVING"
# The needle, normalized the same way the haystack is (see _normalized_text): the
# separators are stripped from BOTH sides so a split across fields still matches.
FORBIDDEN_NEEDLE = "SYSTEMENERGYSAVING"

SCOPE_LABEL = {"GPU_BOARD": LABEL_GPU, "SERVER_WALL": LABEL_SERVER}


class ComparisonError(ValueError):
    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def _fail(failures, code, message):
    failures.append(f"{code}: {message}")


def _normalized_text(document):
    """Canonical bytes, uppercased, with every non-alphanumeric run removed.

    Collapsing separators means SYSTEM_ENERGY_SAVING cannot be smuggled past the
    scan by splitting it across two fields ("SYSTEM_ENERGY" + "_SAVING"), by
    changing case, or by inserting punctuation between the words.
    """
    text = canon.canonical(document).decode("ascii").upper()
    return "".join(ch for ch in text if ch.isalnum())


# ---------------------------------------------------------------------------
# schema gate
# ---------------------------------------------------------------------------

def schema_errors(kind, document):
    """Run the draft-2020 schema under an isolated system Python.

    Refuses if the engine is unavailable: an absent schema engine is not a pass.
    """
    if not SYSTEM_PYTHON.exists():
        raise ComparisonError("E_SCHEMA",
                              f"system python {SYSTEM_PYTHON} is unavailable; the "
                              f"schema gate cannot be skipped")
    try:
        result = subprocess.run(
            [str(SYSTEM_PYTHON), "-I", str(SCHEMA_WORKER)],
            input=json.dumps({"kind": kind, "document": document}),
            capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ComparisonError("E_SCHEMA", f"schema worker failed: {exc}")
    if result.returncode != 0 or not result.stdout.strip():
        raise ComparisonError("E_SCHEMA",
                              f"schema worker error: "
                              f"{(result.stdout or result.stderr).strip()[:200]}")
    payload = json.loads(result.stdout)
    if "engine_error" in payload:
        raise ComparisonError("E_SCHEMA", f"schema engine: {payload['engine_error']}")
    return [f"{err['path']}: {err['message']}" for err in payload["errors"]]


# ---------------------------------------------------------------------------
# timeline validation
# ---------------------------------------------------------------------------

RAW_BOUND_FIELDS = (
    "raw_artifact_id", "execution_artifact_id", "run_nonce", "timeline_id",
    "role", "provenance", "instrument_kind", "instrument_identity",
    "device_identity", "clock_epoch_id", "scope", "synchronization",
    "board_uuids", "included_rails", "excluded_rails", "source_revision",
    "build_revision",
)


def parse_raw_artifact(data):
    """Parse the EXACT bytes that were hashed. Strict; no float, no dup keys.

    Takes bytes, never a path: re-opening the path would reintroduce the
    time-of-check/time-of-use gap that read_verified_artifact closes.

    `pstates` is REQUIRED and must be one status per sample. An earlier revision
    read an optional scalar `pstate`, which made the status-change check dead code
    across the whole suite: the fixtures wrote `"pstate": "P0"` (a scalar), the
    real A6000 trace spans P0/P2/P3/P8, and a producer could simply omit the field
    to skip the check. A gate the producer can opt out of is not a gate.
    """
    try:
        document = canon.loads_strict(data.decode("ascii"))
    except UnicodeDecodeError as exc:
        raise integrator.TimelineError("E_SCHEMA", f"raw artifact is not ASCII: {exc}")
    if not isinstance(document, dict):
        raise integrator.TimelineError("E_SCHEMA", "raw artifact is not an object")
    required = {"schema", "pstates", "samples", *RAW_BOUND_FIELDS}
    if set(document) != required:
        raise integrator.TimelineError(
            "E_SCHEMA",
            "raw artifact fields do not match the e2.raw.v2 contract")
    if document["schema"] != "e2.raw.v2":
        raise integrator.TimelineError(
            "E_SCHEMA", f"raw artifact schema {document['schema']!r} is not e2.raw.v2")
    samples = document["samples"]
    pstates = document["pstates"]
    if not isinstance(pstates, list):
        raise integrator.TimelineError(
            "E_SCHEMA", "pstates must be a list with one entry per sample")
    if not isinstance(samples, list) or len(pstates) != len(samples):
        raise integrator.TimelineError(
            "E_SCHEMA",
            f"pstates has {len(pstates)} entries for {len(samples) if isinstance(samples, list) else '?'} "
            f"samples; a device status must be recorded for every sample")
    for index, state in enumerate(pstates):
        if not isinstance(state, str) or not state or len(state) > 64 or \
                not state.isascii() or any(ord(ch) < 33 or ord(ch) > 126
                                           for ch in state):
            raise integrator.TimelineError(
                "E_SCHEMA", f"pstates[{index}] must be a nonempty printable ASCII "
                "string of at most 64 bytes")
    return document


def parse_wall_capability_proof(data):
    """Parse the hashed capability proof; declarations are not self-proving."""
    try:
        document = canon.loads_strict(data.decode("ascii"))
    except UnicodeDecodeError as exc:
        raise integrator.TimelineError(
            "E_SCHEMA", f"wall capability proof is not ASCII: {exc}")
    required = {
        "schema", "capability_id", "coverage_proof_artifact_id", "provenance",
        "instrument_kind", "instrument_identity", "covers_cpu", "covers_dram",
        "covers_gpu", "covers_storage", "covers_psu_losses", "covers_fans",
        "uncertainty_floor_mw",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise integrator.TimelineError(
            "E_SCHEMA", "wall capability proof fields do not match the frozen v1 "
            "contract")
    if document["schema"] != "e2.wall-capability-proof.v1":
        raise integrator.TimelineError(
            "E_SCHEMA", f"unknown wall capability proof schema "
            f"{document['schema']!r}")
    coverage_fields = ("covers_cpu", "covers_dram", "covers_gpu",
                       "covers_storage", "covers_psu_losses", "covers_fans")
    if any(type(document[field]) is not bool for field in coverage_fields):
        raise integrator.TimelineError(
            "E_SCHEMA", "wall capability coverage fields must be booleans")
    try:
        canon.check_int(document["uncertainty_floor_mw"],
                        "wall capability proof uncertainty_floor_mw")
    except ValueError as exc:
        raise integrator.TimelineError("E_SCHEMA", str(exc))
    if document["uncertainty_floor_mw"] <= 0:
        raise integrator.TimelineError(
            "E_UNCERTAINTY", "wall capability uncertainty floor must be positive")
    return document


EXECUTION_BOUND_FIELDS = (
    "execution_artifact_id", "raw_artifact_id", "raw_artifact_sha256",
    "run_nonce", "provenance", "paid_payload_sha256", "timeline_id", "role",
    "policy_digest",
    "workload_digest", "trace_digest", "slo_policy_digest",
    "route_schedule_digest", "offered_work", "completed_work", "outcomes",
    "clock_epoch_id", "marker_start_us", "marker_end_us", "source_revision",
    "build_revision", "device_identity", "status", "reason_code",
)


def parse_execution_artifact(data):
    """Parse scheduler/run evidence that binds work, outcomes, and markers."""
    try:
        document = canon.loads_strict(data.decode("ascii"))
    except UnicodeDecodeError as exc:
        raise integrator.TimelineError(
            "E_SCHEMA", f"execution artifact is not ASCII: {exc}")
    required = {"schema", *EXECUTION_BOUND_FIELDS}
    if not isinstance(document, dict) or set(document) != required:
        raise integrator.TimelineError(
            "E_SCHEMA", "execution artifact fields do not match e2.execution.v2")
    if document["schema"] != "e2.execution.v2":
        raise integrator.TimelineError(
            "E_SCHEMA", f"unknown execution artifact schema {document['schema']!r}")
    try:
        canon.check_integers(document, "execution artifact")
    except ValueError as exc:
        raise integrator.TimelineError("E_SCHEMA", str(exc))
    return document


def paid_payload_digest(samples, pstates, window_start_us, window_end_us):
    """Digest only the status/power intervals whose energy is paid."""
    if len(samples) != len(pstates):
        raise integrator.TimelineError(
            "E_SCHEMA", "samples and pstates have different lengths")
    intervals = []
    for index in range(len(samples) - 1):
        left = max(samples[index][0], window_start_us)
        right = min(samples[index + 1][0], window_end_us)
        if left < right:
            power, state = samples[index][1], pstates[index]
            if intervals and intervals[-1][1] == left and \
                    intervals[-1][2] == power and intervals[-1][3] == state:
                intervals[-1][1] = right
            else:
                intervals.append([left, right, power, state])
    if not intervals:
        raise integrator.TimelineError("E_WINDOW", "paid window has no sample interval")
    return canon.digest({"schema": "e2.paid-payload.v1", "intervals": intervals})


def validate_timeline(record, trusted_root):
    """Full validation of one RealizedTimeline. Returns a list of E_* failures.

    Recomputation is MANDATORY, never opt-in. Every derived statistic
    (energy_nj, independent_updates, max_gap_us, sample_count) is re-derived from
    the artifact's real bytes and must equal what the record declares.

    An earlier revision of this file took `samples=None` and skipped
    recomputation unless a caller opted in -- which no caller, including the CLI,
    ever did. A record could then declare any energy it liked and validate
    cleanly on the strength of a correct artifact hash. That is E1's lesson
    restated: a signed number is never proof of itself. The hash proves the bytes
    were not edited; only re-integrating them proves the record describes them.
    """
    failures = []
    try:
        errors = schema_errors("timeline", record)
    except ComparisonError as exc:
        return [str(exc)]
    if errors:
        return [f"E_SCHEMA: {err}" for err in errors[:8]]

    # The type gate BEFORE any comparison. A float would satisfy every equality
    # and every ordering test below.
    try:
        canon.check_integers(record, "timeline")
    except ValueError as exc:
        return [f"E_SCHEMA: {exc}"]

    if record.get("record_sha256") != canon.record_digest(record):
        _fail(failures, "E_DIGEST",
              f"timeline {record['timeline_id']} digest does not cover its body")

    # A timeline that reports its own run as failed or ineligible is not evidence.
    # The schema permits FAILED/INELIGIBLE so a set can RECORD an attempt that went
    # wrong (CONTRACT.md section 6); it must never reach the decision.
    if record["status"] != "OK":
        _fail(failures, "E_PAIRS",
              f"timeline {record['timeline_id']} has status {record['status']} "
              f"(reason {record['reason_code']!r}); a run that reports itself "
              f"failed or ineligible cannot support a comparison")

    # Never let the forbidden label exist anywhere, under any key. Normalized
    # before scanning: a contiguous case-sensitive substring test is evaded by
    # splitting the phrase across two fields or by changing case. This is
    # defence in depth -- the closed enums are what actually block the label.
    if FORBIDDEN_NEEDLE in _normalized_text(record):
        _fail(failures, "E_SYSTEM_CLAIM",
              f"timeline {record['timeline_id']} mentions {FORBIDDEN_LABEL}; E2 "
              f"cannot express a total-system claim")

    kind = record["instrument_kind"]
    scope = record["scope"]
    try:
        integrator.check_scope(kind, scope)
    except integrator.TimelineError as exc:
        _fail(failures, exc.code, str(exc).split(": ", 1)[1])

    if scope == "GPU_BOARD" and not record["board_uuids"]:
        _fail(failures, "E_SCOPE",
              "a GPU_BOARD timeline must name the board(s) it measured; this host "
              "has more than one board")
    if scope == "SERVER_WALL" and record["board_uuids"]:
        _fail(failures, "E_SCOPE",
              "a SERVER_WALL timeline must not be keyed to GPU boards")

    overlap = sorted(set(record["included_rails"]) & set(record["excluded_rails"]))
    if overlap:
        _fail(failures, "E_SCOPE", f"rails {overlap} are both included and excluded")

    outcomes = record["outcomes"]
    total = sum(outcomes[k] for k in ("met", "tardy", "rejected", "canceled"))
    if total != record["offered_work"]:
        _fail(failures, "E_WORK_NOT_CLOSED",
              f"timeline {record['timeline_id']} closed {total} of "
              f"{record['offered_work']} offered units; work left in flight at the "
              f"window edge is unpaid energy")
    if record["completed_work"] != outcomes["met"] + outcomes["tardy"]:
        _fail(failures, "E_WORK_MISMATCH",
              f"completed_work {record['completed_work']} != met+tardy "
              f"{outcomes['met'] + outcomes['tardy']}")

    if record["marker_end_us"] <= record["marker_start_us"]:
        _fail(failures, "E_WINDOW", "end marker is not after the start marker")
    if record["window_start_us"] < record["marker_start_us"] or \
            record["window_end_us"] > record["marker_end_us"]:
        _fail(failures, "E_WINDOW",
              "the effective window is not contained by the execution markers")
    if record["window_start_us"] != record["marker_start_us"] or \
            record["window_end_us"] != record["marker_end_us"]:
        _fail(failures, "E_WINDOW",
              "E2 v1 measures the complete execution-marker interval; selecting a "
              "favorable subwindow after the run is not admissible")

    window_us = record["window_end_us"] - record["window_start_us"]
    if window_us < integrator.MIN_WINDOW_US:
        _fail(failures, "E_WINDOW",
              f"window {window_us} us is below MIN_WINDOW_US="
              f"{integrator.MIN_WINDOW_US}")

    if record["independent_updates"] < integrator.MIN_INDEPENDENT_UPDATES:
        _fail(failures, "E_UPDATES",
              f"timeline {record['timeline_id']} has "
              f"{record['independent_updates']} independent sensor updates "
              f"({record['sample_count']} rows) against "
              f"MIN_INDEPENDENT_UPDATES={integrator.MIN_INDEPENDENT_UPDATES}; "
              f"oversampling a slow sensor does not create information")
    if record["max_gap_us"] > integrator.MAX_SAMPLE_GAP_US:
        _fail(failures, "E_GAP",
              f"max gap {record['max_gap_us']} us exceeds "
              f"{integrator.MAX_SAMPLE_GAP_US} us")

    measured_units = max(1, len(record["board_uuids"])) \
        if scope == "GPU_BOARD" else 1
    floor = integrator.uncertainty_nj_floor(kind, window_us, measured_units)
    if record["uncertainty_nj"] < floor:
        _fail(failures, "E_UNCERTAINTY",
              f"timeline {record['timeline_id']} declares uncertainty "
              f"{record['uncertainty_nj']} nJ, below the {kind} floor of {floor} nJ "
              f"over {window_us} us; an instrument is not more accurate than its "
              f"vendor says")

    if record["synchronization"] == "NONE":
        _fail(failures, "E_CLOCK_EPOCH",
              "an unsynchronized timeline cannot be matched to another")

    # The normalizer is pinned to the frozen rule, not merely declared. An
    # earlier revision required this field in the schema and then checked it
    # nowhere, so a timeline could announce any normalizer and still be compared
    # against one produced by the frozen zero-order hold.
    if record["normalizer_digest"] != NORMALIZER_DIGEST:
        _fail(failures, "E_SCHEMA",
              f"timeline {record['timeline_id']} declares normalizer "
              f"{record['normalizer_digest']} but the frozen rule "
              f"(e2.zoh.left_edge.v1) is {NORMALIZER_DIGEST}; a timeline "
              f"normalized by another rule is not comparable to one normalized by "
              f"this one")

    # Artifact: resolve inside the trusted root, read it ONCE, hash those exact
    # bytes, and re-derive every declared statistic from the same buffer.
    try:
        _resolved, data = integrator.read_verified_artifact(
            record["raw_artifact_path"], record["raw_artifact_sha256"],
            trusted_root)
        raw = parse_raw_artifact(data)
        samples, pstates = raw["samples"], raw["pstates"]
    except integrator.TimelineError as exc:
        _fail(failures, exc.code, str(exc).split(": ", 1)[1])
        return failures
    except (ValueError, OSError) as exc:
        _fail(failures, "E_SCHEMA", f"raw artifact is unreadable: {exc}")
        return failures

    try:
        _resolved, execution_data = integrator.read_verified_artifact(
            record["execution_artifact_path"],
            record["execution_artifact_sha256"], trusted_root)
        execution = parse_execution_artifact(execution_data)
    except integrator.TimelineError as exc:
        _fail(failures, exc.code, str(exc).split(": ", 1)[1])
        return failures
    except (ValueError, OSError) as exc:
        _fail(failures, "E_SCHEMA", f"execution artifact is unreadable: {exc}")
        return failures

    for field in RAW_BOUND_FIELDS:
        if raw[field] != record[field]:
            _fail(failures, "E_RAW_BINDING",
                  f"timeline {field}={record[field]!r} does not match its hashed "
                  f"raw artifact value {raw[field]!r}")

    for field in EXECUTION_BOUND_FIELDS:
        if execution[field] != record[field]:
            _fail(failures, "E_EXECUTION_BINDING",
                  f"timeline {field}={record[field]!r} does not match its hashed "
                  f"execution artifact value {execution[field]!r}")

    failures.extend(_verify_recomputation(record, samples, pstates))
    return failures


def _verify_recomputation(record, samples, pstates):
    """Re-derive every reported statistic from the raw samples. Trust nothing.

    Quality is measured over the SAME interval the energy is taken from. Counting
    changes over the whole artifact let an adversary pad busy samples outside the
    paid window: they cost zero energy and bought unlimited `independent_updates`
    for a window that still held only 57 real changes.
    """
    failures = []
    start, end = record["window_start_us"], record["window_end_us"]
    try:
        normalized = integrator.normalize_samples(samples)
        energy = integrator.integrate(normalized, start, end)
        updates, gap = integrator.window_quality(normalized, start, end)
        payload_digest = paid_payload_digest(normalized, pstates, start, end)
    except integrator.TimelineError as exc:
        return [str(exc)]

    if len(normalized) != record["sample_count"]:
        _fail(failures, "E_SCHEMA",
              f"record claims {record['sample_count']} samples, artifact has "
              f"{len(normalized)}")
    if updates != record["independent_updates"]:
        _fail(failures, "E_UPDATES",
              f"record claims {record['independent_updates']} independent updates, "
              f"the artifact yields {updates} inside the paid window "
              f"[{start}, {end}] ({len(normalized)} rows total); samples outside "
              f"the window contribute no energy and buy no quality")
    if updates < integrator.MIN_INDEPENDENT_UPDATES:
        _fail(failures, "E_UPDATES",
              f"the window contains {updates} independent sensor updates against "
              f"MIN_INDEPENDENT_UPDATES={integrator.MIN_INDEPENDENT_UPDATES}; "
              f"oversampling a slow sensor does not create information")
    if gap != record["max_gap_us"]:
        _fail(failures, "E_GAP",
              f"record claims max gap {record['max_gap_us']} us, the artifact "
              f"yields {gap} us inside the window")
    if gap > integrator.MAX_SAMPLE_GAP_US:
        _fail(failures, "E_GAP",
              f"largest in-window gap {gap} us exceeds MAX_SAMPLE_GAP_US="
              f"{integrator.MAX_SAMPLE_GAP_US}")
    if energy != record["energy_nj"]:
        _fail(failures, "E_SCHEMA",
              f"record claims {record['energy_nj']} nJ, the frozen zero-order-hold "
              f"integral of the artifact is {energy} nJ")
    if payload_digest != record["paid_payload_sha256"]:
        _fail(failures, "E_RAW_BINDING",
              f"record paid_payload_sha256={record['paid_payload_sha256']} but "
              f"the paid sample/status intervals yield {payload_digest}")

    # A status applies to the same left-edge hold interval as its power sample.
    # The sample exactly at window_end has no paid interval and is excluded.
    window_states = {
        pstates[index]
        for index in range(len(normalized) - 1)
        if max(normalized[index][0], start) <
           min(normalized[index + 1][0], end)
    }
    if len(window_states) > 1:
        _fail(failures, "E_STATUS_CHANGE",
              f"the device changed status {sorted(window_states)} inside the paid "
              f"window; one window cannot span them")
    return failures


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

# The identity of the frozen integration rule (CONTRACT.md section 3). A timeline
# normalized by some other rule is not comparable to one normalized by this one,
# so the digest is pinned rather than merely recorded.
NORMALIZER_DIGEST = canon.digest({"normalizer": "e2.zoh.left_edge.v1"})

MATCH_FIELDS = (
    ("workload_digest", "E_WORKLOAD_MISMATCH"),
    ("trace_digest", "E_WORKLOAD_MISMATCH"),
    ("slo_policy_digest", "E_SLO_MISMATCH"),
    # route_schedule_digest is deliberately absent: control and treatment run
    # different schedules by definition (that is the treatment), so requiring
    # equality would be incoherent. It is recorded provenance, not a match key.
    ("normalizer_digest", "E_SCHEMA"),
    ("offered_work", "E_WORK_MISMATCH"),
    ("completed_work", "E_WORK_MISMATCH"),
    ("scope", "E_SCOPE_MISMATCH"),
    ("instrument_kind", "E_SCOPE_MISMATCH"),
    ("instrument_identity", "E_SCOPE_MISMATCH"),
    ("device_identity", "E_DEVICE_MISMATCH"),
    ("source_revision", "E_BUILD_MISMATCH"),
    ("build_revision", "E_BUILD_MISMATCH"),
    ("synchronization", "E_CLOCK_EPOCH"),
    ("clock_epoch_id", "E_CLOCK_EPOCH"),
)


def check_match(control, treatment):
    """Every matching requirement from CONTRACT.md section 4."""
    failures = []
    for field, code in MATCH_FIELDS:
        if control[field] != treatment[field]:
            _fail(failures, code,
                  f"control {field}={control[field]!r} but treatment "
                  f"{field}={treatment[field]!r}; E2 never compares across this")
    if control["outcomes"] != treatment["outcomes"]:
        _fail(failures, "E_SLO_MISMATCH",
              f"terminal outcomes differ: control {control['outcomes']} vs "
              f"treatment {treatment['outcomes']}; buying energy with a different "
              f"SLO result is not relief")
    for field in ("included_rails", "excluded_rails", "board_uuids"):
        if set(control[field]) != set(treatment[field]):
            _fail(failures, "E_SCOPE_MISMATCH",
                  f"{field} differ: control {sorted(control[field])} vs treatment "
                  f"{sorted(treatment[field])}")
    if control["role"] != "OPTIMIZED_SERVER_ONLY_CONTROL" or \
            treatment["role"] != "Q_PIM_TREATMENT":
        _fail(failures, "E_ROLE",
              f"roles must be exactly one control and one treatment, got "
              f"{control['role']} and {treatment['role']}")
    if control["policy_digest"] == treatment["policy_digest"]:
        _fail(failures, "E_SAME_POLICY",
              "control and treatment share a policy digest; comparing a policy "
              "against itself measures noise")
    cw = control["window_end_us"] - control["window_start_us"]
    tw = treatment["window_end_us"] - treatment["window_start_us"]
    if abs(cw - tw) > integrator.WINDOW_TOLERANCE_US:
        _fail(failures, "E_WINDOW_MISMATCH",
              f"effective windows differ by {abs(cw - tw)} us, beyond "
              f"WINDOW_TOLERANCE_US={integrator.WINDOW_TOLERANCE_US}")
    if control["timeline_id"] == treatment["timeline_id"]:
        _fail(failures, "E_DUPLICATE_RUN",
              "control and treatment are the same run")
    if control["run_nonce"] == treatment["run_nonce"]:
        _fail(failures, "E_DUPLICATE_RUN",
              "control and treatment share a run nonce")
    if control["paid_payload_sha256"] == treatment["paid_payload_sha256"]:
        _fail(failures, "E_DUPLICATE_RUN",
              "control and treatment reuse the same paid power/status payload; "
              "formatting or out-of-window padding cannot turn one observation "
              "into two independent runs")
    for field in ("raw_artifact_id", "raw_artifact_path", "raw_artifact_sha256",
                  "execution_artifact_id", "execution_artifact_path",
                  "execution_artifact_sha256"):
        if control[field] == treatment[field]:
            _fail(failures, "E_DUPLICATE_RUN",
                  f"control and treatment alias the same {field}; one power trace "
                  f"cannot serve as two independent executions")
    return failures


def decide(control, treatment, scope):
    """The conservative decision. Integer-only, no division (CONTRACT.md 5).

    The type gate runs HERE too, not only at load: decide() is importable and a
    float energy would compare equal to an int and silently poison the cross
    multiplication. Fail closed at the point of arithmetic.
    """
    if scope not in SCOPE_LABEL:
        raise ComparisonError("E_SCOPE", f"unknown decision scope {scope!r}")
    for label, record in (("control", control), ("treatment", treatment)):
        for field in ("energy_nj", "uncertainty_nj"):
            value = record.get(field, canon.MISSING)
            if value is canon.MISSING:
                raise ComparisonError(
                    "E_UNCERTAINTY" if field == "uncertainty_nj" else "E_SCHEMA",
                    f"{label} has no {field}; uncertainty is never optional and "
                    f"never dropped from the decision")
            if not canon.is_int(value):
                raise ComparisonError(
                    "E_SCHEMA",
                    f"{label}.{field} is {type(value).__name__} {value!r}; this "
                    f"decision is integer-only, and a float compares equal to an "
                    f"integer while bypassing the type gate")
            if value < 0:
                raise ComparisonError(
                    "E_SCHEMA", f"{label}.{field} is negative: {value}")
    control_lower = control["energy_nj"] - control["uncertainty_nj"]
    treatment_upper = treatment["energy_nj"] + treatment["uncertainty_nj"]
    if control_lower <= 0:
        # A control whose lower bound is at or below zero carries uncertainty at
        # least as large as its own energy. No relief can be established against
        # it, and the cross multiplication below would compare against a
        # non-positive budget.
        raise ComparisonError(
            "E_UNCERTAINTY",
            f"control lower bound is {control_lower} nJ (energy "
            f"{control['energy_nj']} nJ, uncertainty {control['uncertainty_nj']} "
            f"nJ); an instrument whose error swamps the signal cannot establish "
            f"relief")
    margin = control_lower - treatment_upper
    # 10 percent gate by integer cross multiplication:
    #   treatment_upper <= 0.9 * control_lower   <=>   t*10 <= c*9
    meets_ten = (treatment_upper * 10) <= (control_lower * 9)
    conservative_relief = treatment_upper < control_lower
    delta = control["energy_nj"] - treatment["energy_nj"]
    server_delta = delta if scope == "SERVER_WALL" else None
    return {
        "control_lower_nj": control_lower,
        "treatment_upper_nj": treatment_upper,
        "conservative_margin_nj": margin,
        "boundary_delta_nj": delta,
        "server_wall_delta_nj": server_delta,
        "phone_plus_external_break_even_budget_nj": server_delta,
        "meets_ten_percent_gate": meets_ten,
        "relief": conservative_relief and meets_ten,
    }


def compare(control, treatment, trusted_root, wall_capability=None):
    """Validate, match, and decide. Returns (label, reason, detail, failures).

    There is no way to ask for a comparison WITHOUT artifact recomputation: the
    samples are always loaded from the trusted root and re-integrated.
    """
    failures = []
    for record in (control, treatment):
        failures.extend(validate_timeline(record, trusted_root))
    if failures:
        return LABEL_INVALID, "TIMELINE_INVALID", {}, failures

    failures.extend(check_match(control, treatment))
    if failures:
        return LABEL_INVALID, "NOT_MATCHED", {}, failures

    # Synthetic evidence proves mechanics and nothing physical, ever.
    if control["provenance"] == "SYNTHETIC" or treatment["provenance"] == "SYNTHETIC" \
            or control["instrument_kind"] == "SYNTHETIC":
        detail = decide(control, treatment, control["scope"])
        return (LABEL_INVALID, "SYNTHETIC_NO_PHYSICAL_CLAIM", detail,
                ["E_PROVENANCE: synthetic timelines exercise mechanics and emit no "
                 "physical result"])

    scope = control["scope"]
    if scope == "SERVER_WALL":
        cap_failures = []
        for timeline in (control, treatment):
            cap_failures.extend(
                check_wall_capability(wall_capability, timeline, trusted_root))
        if cap_failures:
            return LABEL_INVALID, "INCOMPLETE_WALL", {}, cap_failures

    detail = decide(control, treatment, scope)
    return (LABEL_INVALID, "PAIR_ONLY_NO_AGGREGATE_CLAIM", detail,
            ["E_PAIRS: one control/treatment pair is diagnostic only; a physical "
             "label requires the complete, predeclared repetition set and the "
             "frozen all-pairs aggregate"])


def check_wall_capability(capability, timeline, trusted_root):
    """SERVER_WALL needs demonstrated complete coverage, not a name."""
    failures = []
    if capability is None:
        return ["E_INCOMPLETE_WALL: a SERVER_WALL claim requires a "
                "ServerWallCapability record demonstrating complete coverage; none "
                "was supplied"]
    try:
        errors = schema_errors("capability", capability)
    except ComparisonError as exc:
        return [str(exc)]
    if errors:
        return [f"E_SCHEMA: {err}" for err in errors[:8]]
    try:
        canon.check_int(capability["uncertainty_floor_mw"],
                        "wall capability uncertainty_floor_mw")
    except ValueError as exc:
        return [f"E_SCHEMA: {exc}"]
    if capability.get("record_sha256") != canon.record_digest(capability):
        _fail(failures, "E_DIGEST", "capability digest does not cover its body")
    if capability["status"] != "OK":
        _fail(failures, "E_INCOMPLETE_WALL",
              f"capability {capability['capability_id']} is "
              f"{capability['status']} ({capability['reason_code']})")
    if capability["uncertainty_floor_mw"] <= 0:
        _fail(failures, "E_UNCERTAINTY",
              "wall capability uncertainty floor must be positive")
    if capability["instrument_kind"] != timeline["instrument_kind"]:
        _fail(failures, "E_SCOPE",
              f"capability describes {capability['instrument_kind']} but the "
              f"timeline used {timeline['instrument_kind']}")
    if capability["instrument_identity"] != timeline["instrument_identity"]:
        _fail(failures, "E_SCOPE",
              "capability describes a different instrument instance")
    if capability["provenance"] != timeline["provenance"]:
        _fail(failures, "E_PROVENANCE",
              f"{capability['provenance']} capability evidence cannot authorize a "
              f"{timeline['provenance']} timeline")
    missing = [field for field in ("covers_cpu", "covers_dram", "covers_gpu",
                                   "covers_storage", "covers_psu_losses",
                                   "covers_fans")
               if capability.get(field) is not True]
    if missing:
        _fail(failures, "E_INCOMPLETE_WALL",
              f"capability {capability['capability_id']} does not cover {missing}; "
              f"a partial boundary is not a server wall, whatever it is called")

    try:
        _resolved, data = integrator.read_verified_artifact(
            capability["coverage_proof_artifact_path"],
            capability["coverage_proof_sha256"], trusted_root)
        proof = parse_wall_capability_proof(data)
    except integrator.TimelineError as exc:
        _fail(failures, exc.code, str(exc).split(": ", 1)[1])
        return failures
    except (ValueError, OSError) as exc:
        _fail(failures, "E_SCHEMA", f"wall capability proof is unreadable: {exc}")
        return failures

    proof_fields = (
        "capability_id", "coverage_proof_artifact_id", "provenance",
        "instrument_kind", "instrument_identity", "covers_cpu", "covers_dram",
        "covers_gpu", "covers_storage", "covers_psu_losses", "covers_fans",
        "uncertainty_floor_mw",
    )
    for field in proof_fields:
        if capability[field] != proof[field]:
            _fail(failures, "E_INCOMPLETE_WALL",
                  f"capability {field}={capability[field]!r} does not match its "
                  f"hashed proof value {proof[field]!r}")

    window_us = timeline["window_end_us"] - timeline["window_start_us"]
    capability_floor_nj = capability["uncertainty_floor_mw"] * window_us
    if timeline["uncertainty_nj"] < capability_floor_nj:
        _fail(failures, "E_UNCERTAINTY",
              f"timeline uncertainty {timeline['uncertainty_nj']} nJ is below the "
              f"capability floor {capability_floor_nj} nJ")
    if capability["provenance"] == "MEASURED":
        _fail(failures, "E_CAPABILITY_UNCERTIFIED",
              "ServerWallCapability v1 is a hashed declaration, not calibration "
              "or acquisition evidence; measured wall claims remain blocked until "
              "a versioned proof binds topology, validity, calibration, and raw "
              "acquisition artifacts")
    return failures


def assert_not_additive_input(record):
    """CONTRACT.md section 0, made executable.

    An aggregate boundary timeline must never be fed back into E1's additive
    per-device solver: doing so double-counts shared power and attributes it to
    individual routes. This is the guard that says so out loud.
    """
    # Section 0 says "an E2 timeline OR COMPARISON". An earlier revision keyed on
    # RealizedTimeline alone, so a MatchedComparison or a whole RepetitionSet --
    # which are MORE aggregated, not less -- passed straight through.
    aggregate_kinds = {"RealizedTimeline", "MatchedComparison", "RepetitionSet"}
    if isinstance(record, dict) and record.get("kind") in aggregate_kinds:
        raise ComparisonError(
            "E_ADDITIVE_REUSE",
            f"{record['kind']} {record.get('timeline_id') or record.get('comparison_id') or record.get('set_id')!r} "
            f"is an AGGREGATE boundary measurement and must never enter the "
            f"additive per-device solver; E2 compares realized timelines post hoc "
            f"instead")
    return record


# ---------------------------------------------------------------------------
# repetition set
# ---------------------------------------------------------------------------

def validate_repetition_set(record):
    """Every attempted repetition present, balanced rotation, no reused run."""
    failures = []
    try:
        errors = schema_errors("repetition_set", record)
    except ComparisonError as exc:
        return [str(exc)]
    if errors:
        return [f"E_SCHEMA: {err}" for err in errors[:8]]
    try:
        canon.check_integers(record, "repetition_set")
    except ValueError as exc:
        return [f"E_SCHEMA: {exc}"]
    if record.get("record_sha256") != canon.record_digest(record):
        _fail(failures, "E_DIGEST", "repetition set digest does not cover its body")
    try:
        integrator.check_scope(record["instrument_kind"], record["scope"])
    except integrator.TimelineError as exc:
        _fail(failures, exc.code, str(exc).split(": ", 1)[1])

    pairs = record["pairs"]
    if record["attempted_pairs"] != len(pairs):
        _fail(failures, "E_PAIRS",
              f"{record['attempted_pairs']} pairs were attempted but {len(pairs)} "
              f"are listed; every attempt must be represented, including failures, "
              f"so a dropped unfavorable run is a diff rather than a silence")

    ok_pairs = [p for p in pairs if p["status"] == "OK"]
    if len(ok_pairs) < MIN_PAIRS:
        _fail(failures, "E_PAIRS",
              f"{len(ok_pairs)} OK pairs is below MIN_PAIRS={MIN_PAIRS}")
    failed_pairs = [p["pair_index"] for p in pairs if p["status"] == "FAILED"]
    if failed_pairs:
        _fail(failures, "E_PAIRS",
              f"measurement pairs {failed_pairs} failed; failed attempts are "
              f"recorded but cannot be excluded from a physical claim")

    seen = {}
    seen_digests = {}
    for pair in pairs:
        if pair["control_record_sha256"] == pair["treatment_record_sha256"]:
            _fail(failures, "E_DUPLICATE_RUN",
                  f"pair {pair['pair_index']} binds the same timeline record digest "
                  f"for control and treatment")
        for role in ("control", "treatment"):
            role_key = f"{role}_timeline_id"
            digest_key = f"{role}_record_sha256"
            run = pair[role_key]
            run_digest = pair[digest_key]
            if run in seen:
                _fail(failures, "E_DUPLICATE_RUN",
                      f"run {run} appears in pair {seen[run]} and pair "
                      f"{pair['pair_index']}; reusing one run across pairs fakes "
                      f"independent repetitions")
            seen[run] = pair["pair_index"]
            if run_digest in seen_digests:
                _fail(failures, "E_DUPLICATE_RUN",
                      f"record digest {run_digest} appears as {run} and "
                      f"{seen_digests[run_digest]}; changing a timeline ID does "
                      "not create an independent run")
            else:
                seen_digests[run_digest] = run

    indices = [p["pair_index"] for p in pairs]
    expected_indices = list(range(len(pairs)))
    if indices != expected_indices:
        _fail(failures, "E_PAIRS",
              f"pair_index values must be exactly {expected_indices}, got {indices}; "
              f"gaps can hide failed attempts")

    warmup_ids = [item["timeline_id"] for item in record["warmup_timelines"]]
    if len(set(warmup_ids)) != len(warmup_ids):
        _fail(failures, "E_DUPLICATE_RUN", "a warmup timeline is listed twice")
    overlap = sorted(set(warmup_ids) & set(seen))
    if overlap:
        _fail(failures, "E_DUPLICATE_RUN",
              f"warmup timelines {overlap} are reused as measured repetitions; "
              f"warmups cannot contribute to the aggregate")
    for item in record["warmup_timelines"]:
        run_digest = item["record_sha256"]
        if run_digest in seen_digests:
            _fail(failures, "E_DUPLICATE_RUN",
                  f"warmup {item['timeline_id']} reuses measured record digest "
                  f"{run_digest}; changing its ID does not make it independent")
        elif run_digest in {other["record_sha256"]
                            for other in record["warmup_timelines"]
                            if other is not item}:
            _fail(failures, "E_DUPLICATE_RUN",
                  f"warmup record digest {run_digest} is listed more than once")

    # Balanced rotation: the declared order must alternate which role ran first,
    # and the executed order must match it. Post-selecting the order after seeing
    # results is the easiest way to manufacture a win.
    if record["order_digest"] != canon.digest(record["declared_order"]):
        _fail(failures, "E_ORDER",
              "order_digest does not cover declared_order; the execution order "
              "must be fixed before the runs, not chosen after")
    firsts = [p["first_executed_role"] for p in pairs]
    controls_first = sum(1 for f in firsts if f == "OPTIMIZED_SERVER_ONLY_CONTROL")
    treatments_first = len(firsts) - controls_first
    if pairs and abs(controls_first - treatments_first) > 1:
        _fail(failures, "E_ORDER",
              f"execution order is unbalanced: control ran first {controls_first} "
              f"times and treatment {treatments_first}; a warm/cold bias would be "
              f"indistinguishable from a policy effect")
    expected = []
    for index in range(len(pairs)):
        first = ("OPTIMIZED_SERVER_ONLY_CONTROL" if index % 2 == 0
                 else "Q_PIM_TREATMENT")
        second = ("Q_PIM_TREATMENT" if index % 2 == 0
                  else "OPTIMIZED_SERVER_ONLY_CONTROL")
        expected.extend([first, second])
    if len(record["declared_order"]) != len(expected):
        _fail(failures, "E_ORDER",
              f"declared_order has {len(record['declared_order'])} roles for "
              f"{len(pairs)} pairs; it must contain exactly {len(expected)}")
    elif pairs and record["declared_order"] != expected:
        _fail(failures, "E_ORDER",
              "declared_order is not the frozen ABBA rotation")
    for index, pair in enumerate(pairs):
        want = ("OPTIMIZED_SERVER_ONLY_CONTROL" if index % 2 == 0
                else "Q_PIM_TREATMENT")
        if pair["first_executed_role"] != want:
            _fail(failures, "E_ORDER",
                  f"pair {pair['pair_index']} ran {pair['first_executed_role']} "
                  f"first but the declared rotation says {want}")
    if record["aggregate_result_label"] != LABEL_INVALID:
        _fail(failures, "E_PAIRS",
              "the all-pairs aggregate evaluator is not implemented; a repetition "
              "set cannot self-declare a physical result")
    return failures


def validate_comparison(record, control, treatment, repetition_set, trusted_root,
                        wall_capability=None):
    """Resolve and recompute a diagnostic-only MatchedComparison v1 record."""
    failures = []
    try:
        errors = schema_errors("comparison", record)
    except ComparisonError as exc:
        return [str(exc)]
    if errors:
        return [f"E_SCHEMA: {err}" for err in errors[:8]]
    try:
        canon.check_integers(
            record, "comparison",
            frozenset({"comparison.meets_ten_percent_gate"}))
    except ValueError as exc:
        return [f"E_SCHEMA: {exc}"]
    if record.get("record_sha256") != canon.record_digest(record):
        _fail(failures, "E_DIGEST", "comparison digest does not cover its body")
    if record.get("result_label") != LABEL_INVALID:
        _fail(failures, "E_PAIRS",
              "MatchedComparison v1 is diagnostic-only and cannot carry a "
              "physical result label")
    if FORBIDDEN_NEEDLE in _normalized_text(record):
        _fail(failures, "E_SYSTEM_CLAIM",
              f"comparison mentions forbidden label {FORBIDDEN_LABEL}")

    try:
        label, reason, detail, source_failures = compare(
            control, treatment, trusted_root, wall_capability)
    except ComparisonError as exc:
        return [str(exc)]
    except (KeyError, TypeError, ValueError) as exc:
        return [f"E_SCHEMA: comparison source validation failed: {exc}"]
    if reason in {"TIMELINE_INVALID", "NOT_MATCHED", "INCOMPLETE_WALL"}:
        _fail(failures, "E_BINDING",
              f"comparison source records are not eligible: {reason}")
        failures.extend(source_failures[:4])
        return failures

    set_failures = validate_repetition_set(repetition_set)
    if set_failures:
        _fail(failures, "E_BINDING", "comparison repetition set is invalid")
        failures.extend(set_failures[:4])
        return failures
    pair = next((item for item in repetition_set["pairs"]
                 if item["control_timeline_id"] == control["timeline_id"]
                 and item["control_record_sha256"] == control["record_sha256"]
                 and item["treatment_timeline_id"] == treatment["timeline_id"]
                 and item["treatment_record_sha256"] == treatment["record_sha256"]),
                None)
    if pair is None:
        _fail(failures, "E_BINDING",
              "comparison sources are not an exact pair in the repetition set")
        return failures

    expected_id = "cmp." + canon.digest({
        "set": repetition_set["record_sha256"],
        "pair_index": pair["pair_index"],
        "control": control["record_sha256"],
        "treatment": treatment["record_sha256"],
    })
    expected_matched = {
        "workload_digest": control["workload_digest"],
        "trace_digest": control["trace_digest"],
        "slo_policy_digest": control["slo_policy_digest"],
        "offered_work": control["offered_work"],
        "completed_work": control["completed_work"],
        "outcomes": dict(control["outcomes"]),
        "included_rails": sorted(control["included_rails"]),
        "excluded_rails": sorted(control["excluded_rails"]),
        "board_uuids": sorted(control["board_uuids"]),
        "clock_epoch_id": control["clock_epoch_id"],
        "effective_window_us": control["window_end_us"] - control["window_start_us"],
    }
    expected = {
        "comparison_id": expected_id,
        "control_timeline_id": control["timeline_id"],
        "control_record_sha256": control["record_sha256"],
        "treatment_timeline_id": treatment["timeline_id"],
        "treatment_record_sha256": treatment["record_sha256"],
        "repetition_set_digest": repetition_set["record_sha256"],
        "order_digest": repetition_set["order_digest"],
        "scope": control["scope"],
        "instrument_kind": control["instrument_kind"],
        "matched_on": expected_matched,
        "control_energy_nj": control["energy_nj"],
        "control_uncertainty_nj": control["uncertainty_nj"],
        "treatment_energy_nj": treatment["energy_nj"],
        "treatment_uncertainty_nj": treatment["uncertainty_nj"],
        "control_lower_nj": detail["control_lower_nj"],
        "treatment_upper_nj": detail["treatment_upper_nj"],
        "conservative_margin_nj": detail["conservative_margin_nj"],
        "boundary_delta_nj": detail["boundary_delta_nj"],
        "server_wall_delta_nj": detail["server_wall_delta_nj"],
        "phone_plus_external_break_even_budget_nj":
            detail["phone_plus_external_break_even_budget_nj"],
        "meets_ten_percent_gate": detail["meets_ten_percent_gate"],
        "result_label": label,
        "reason_code": reason,
    }
    if repetition_set["scope"] != control["scope"] or \
            repetition_set["instrument_kind"] != control["instrument_kind"]:
        _fail(failures, "E_BINDING",
              "repetition-set scope/instrument do not match comparison sources")
    for field, expected_value in expected.items():
        if record[field] != expected_value:
            _fail(failures, "E_INCOHERENT",
                  f"comparison {field}={record[field]!r} does not match the "
                  f"resolved value {expected_value!r}")
    return failures


def build_comparison(control, treatment, trusted_root, repetition_set,
                     wall_capability=None):
    """Validate the evidence and emit a sealed MatchedComparison.

    The label is DERIVED here, never accepted from the caller.

    An earlier revision took `label`, `reason`, and `detail` as arguments and
    checked only that the label was one of the four allowed strings. Nothing tied
    it to the evidence, so importing this module and calling this function stamped
    a digest-valid, schema-valid `SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED` onto
    unvalidated junk whose treatment burned 1e15 nJ MORE -- on a host whose audit
    says no wall instrument exists. `compare()`'s discipline was the only thing
    holding, and it was bypassed by not calling it.

    That mattered because the next checkpoint's aggregate evaluator will call this
    function. It now cannot be handed a conclusion; it can only reach one.
    """
    label, reason, detail, failures = compare(control, treatment, trusted_root,
                                              wall_capability)
    if reason in {"TIMELINE_INVALID", "NOT_MATCHED", "INCOMPLETE_WALL"}:
        raise ComparisonError(reason, "; ".join(failures[:4]))
    if label != LABEL_INVALID:
        raise ComparisonError(
            "E_PAIRS", "MatchedComparison v1 cannot carry a physical result")

    set_failures = validate_repetition_set(repetition_set)
    if set_failures:
        raise ComparisonError("E_PAIRS", "; ".join(set_failures[:4]))
    if repetition_set["scope"] != control["scope"] or \
            repetition_set["instrument_kind"] != control["instrument_kind"]:
        raise ComparisonError(
            "E_BINDING",
            "repetition-set scope/instrument do not match the bound timelines")
    pair = next((item for item in repetition_set["pairs"]
                 if item["control_timeline_id"] == control["timeline_id"]
                 and item["control_record_sha256"] == control["record_sha256"]
                 and item["treatment_timeline_id"] == treatment["timeline_id"]
                 and item["treatment_record_sha256"] == treatment["record_sha256"]),
                None)
    if pair is None:
        raise ComparisonError(
            "E_BINDING",
            "control/treatment IDs and record digests are not an exact pair in "
            "the validated repetition set")

    record = {
        "schema_version": 1,
        "kind": "MatchedComparison",
        "comparison_id": "cmp." + canon.digest({
            "set": repetition_set["record_sha256"],
            "pair_index": pair["pair_index"],
            "control": control["record_sha256"],
            "treatment": treatment["record_sha256"],
        }),
        "control_timeline_id": control["timeline_id"],
        "control_record_sha256": control["record_sha256"],
        "treatment_timeline_id": treatment["timeline_id"],
        "treatment_record_sha256": treatment["record_sha256"],
        "repetition_set_digest": repetition_set["record_sha256"],
        "order_digest": repetition_set["order_digest"],
        "scope": control["scope"],
        "instrument_kind": control["instrument_kind"],
        "matched_on": {
            "workload_digest": control["workload_digest"],
            "trace_digest": control["trace_digest"],
            "slo_policy_digest": control["slo_policy_digest"],
            "offered_work": control["offered_work"],
            "completed_work": control["completed_work"],
            "outcomes": dict(control["outcomes"]),
            "included_rails": sorted(control["included_rails"]),
            "excluded_rails": sorted(control["excluded_rails"]),
            "board_uuids": sorted(control["board_uuids"]),
            "clock_epoch_id": control["clock_epoch_id"],
            "effective_window_us": control["window_end_us"] - control["window_start_us"],
        },
        "control_energy_nj": control["energy_nj"],
        "control_uncertainty_nj": control["uncertainty_nj"],
        "treatment_energy_nj": treatment["energy_nj"],
        "treatment_uncertainty_nj": treatment["uncertainty_nj"],
        "control_lower_nj": detail.get("control_lower_nj", 0),
        "treatment_upper_nj": detail.get("treatment_upper_nj", 0),
        "conservative_margin_nj": detail.get("conservative_margin_nj", 0),
        "boundary_delta_nj": detail.get("boundary_delta_nj", 0),
        "server_wall_delta_nj": detail.get("server_wall_delta_nj"),
        "phone_plus_external_break_even_budget_nj":
            detail.get("phone_plus_external_break_even_budget_nj"),
        "meets_ten_percent_gate": bool(detail.get("meets_ten_percent_gate", False)),
        "result_label": label,
        "reason_code": reason,
        "record_sha256": "",
    }
    canon.seal(record)
    record_failures = validate_comparison(
        record, control, treatment, repetition_set, trusted_root,
        wall_capability)
    if record_failures:
        raise ComparisonError("E_SCHEMA", "; ".join(record_failures[:4]))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="E2 matched timeline comparator")
    parser.add_argument("--control", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--trusted-root", required=True)
    parser.add_argument("--capability")
    parser.add_argument("--out")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        control = canon.load_strict(args.control)
        treatment = canon.load_strict(args.treatment)
        capability = canon.load_strict(args.capability) if args.capability else None
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        print(f"E2_FAIL: cannot load input: {exc}", file=sys.stderr)
        return 2

    try:
        label, reason, detail, failures = compare(
            control, treatment, args.trusted_root, wall_capability=capability)
    except ComparisonError as exc:
        print(f"E2_FAIL: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError) as exc:
        print(f"E2_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    for failure in failures:
        print(f"E2_FAIL: {failure}", file=sys.stderr)
    if not args.quiet:
        print(json.dumps({"result_label": label, "reason_code": reason,
                          "detail": detail}, sort_keys=True))
    if args.out:
        print("E2_FAIL: E_PAIRS: --out is disabled until a validated repetition "
              "set and all-pairs aggregate can supply real set/order digests",
              file=sys.stderr)
        return 1
    return 0 if label in (LABEL_GPU, LABEL_SERVER, LABEL_FAIL) else 1


if __name__ == "__main__":
    sys.exit(main())
