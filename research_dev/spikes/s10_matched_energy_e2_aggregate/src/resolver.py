#!/usr/bin/env python3
"""One trusted-root, read-once resolver for E2A. Standard library only.

Implements CONTRACT.md sections 2-4. Fails closed everywhere.

DESIGN RULE, stated once and enforced throughout: this module RESOLVES and
VALIDATES. It never decides anything about relief and it never accepts a
conclusion. No function here takes a label, a count, a verdict, or a `drained`
flag. Everything is derived from bytes that were read exactly once.

The resolver's job is to make "unresolved" impossible to confuse with "fine". If
any planned slot, record, anchor, request, or action does not resolve, the caller
gets an exception, not a shorter list.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import pathlib
import subprocess

import anchors
import e2a_canon as canon
from e2a_e2 import integrator as e2_integrator

SYSTEM_PYTHON = pathlib.Path("/usr/bin/python3")
SCHEMA_WORKER = pathlib.Path(__file__).with_name("e2a_schema_gate.py")

# E2's frozen gates, restated by digest rather than re-declared. The plan must pin
# these exact values; a plan that pins different ones has retuned the gate after
# the fact. See CONTRACT.md section 5.
GATE_CONSTANTS = {
    "MIN_PAIRS": 8,
    "MIN_INDEPENDENT_UPDATES": 100,
    "MAX_SAMPLE_GAP_US": 250000,
    "MIN_WINDOW_US": 1000000,
    "WINDOW_TOLERANCE_US": 50000,
    "MAX_SAMPLES": 1000000,
    "RELIEF_NUMERATOR": 9,
    "RELIEF_DENOMINATOR": 10,
    "MIN_PREFIX_TOKENS": 32,
    "MAX_LENGTH_TOLERANCE_TOKENS": 4,
    "MIN_SLO_MET_PPM": 1000000,
    "NORMALIZER": "e2.zoh.left_edge.v1",
    "AGGREGATE_METHOD": "SUM_ALL_PAIRS_V1",
}
GATE_CONSTANTS_DIGEST = canon.digest(GATE_CONSTANTS)

MIN_PAIRS = GATE_CONSTANTS["MIN_PAIRS"]
ACTIVE_SCHEMA_VERSION = 2

# A ledger entry status that is anything but OK poisons the whole set. INELIGIBLE
# is listed explicitly because it is the drop channel nobody guards: it is neither
# OK nor FAILED, so a `status == "FAILED"` check misses it entirely and an
# unfavourable run can be relabelled INELIGIBLE and vanish.
POISON_STATUSES = frozenset({"FAILED", "CANCELED", "CRASHED", "INELIGIBLE",
                             "ABORTED_BY_GATE"})

ZERO_DIGEST = "0" * 64

# Until SLO policy is a typed, resolved record, every request must meet its
# predeclared deadline. This prevents a comparison with identical all-tardy
# roles from passing merely because both sides failed in the same way.
MIN_PREFIX_TOKENS = GATE_CONSTANTS["MIN_PREFIX_TOKENS"]
MAX_LENGTH_TOLERANCE_TOKENS = \
    GATE_CONSTANTS["MAX_LENGTH_TOLERANCE_TOKENS"]
MIN_SLO_MET_PPM = GATE_CONSTANTS["MIN_SLO_MET_PPM"]

# One resolution consumes one immutable view of every canonical path. Repeated
# reads return the original buffer. A second identity may not claim the same
# path, even when the bytes happen to match.
_RESOLUTION_CONTEXT = contextvars.ContextVar("e2a_resolution_context",
                                              default=None)

# Which actions must lie inside the paid window? ALL of them, unless explicitly
# exempted here -- and nothing is exempted.
#
# An earlier revision listed the energy-bearing kinds instead, which made the
# window check a NAME gate over 7 of the schema's 12 kinds: QUEUE_SUBMIT, CANCEL,
# LEASE_ACQUIRE, LEASE_RELEASE and QUEUE_DRAIN were unchecked, so relabelling a
# PREFETCH as a QUEUE_SUBMIT moved weight staging outside the window for free.
# The set of things that cost energy is not knowable from a label the producer
# chooses, so the default is "charged" and the exemption list carries the burden
# of proof. It is empty on purpose: if a kind is ever added here, that addition
# is the claim that it consumes no energy, and it should be argued in review.
FREE_ACTIONS = frozenset()

# Actions that must be complete before the window closes.
TERMINAL_STATES = frozenset({"COMPLETE", "CANCELED", "FAILED"})


class ResolveError(ValueError):
    """A resolution defect. Always fatal; never downgraded to a warning."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def _fail(code, message):
    raise ResolveError(code, message)


_RESOLVED_REQUESTS_CAPABILITY = object()


class _ResolvedRequests:
    """Request evidence issued only after resolve_requests validates it."""

    __slots__ = ("record", "token_ids", "timing", "_capability")

    def __init__(self, record, token_ids, timing, capability):
        if capability is not _RESOLVED_REQUESTS_CAPABILITY:
            raise TypeError("resolved request evidence is resolver-issued")
        self.record = record
        self.token_ids = token_ids
        self.timing = timing
        self._capability = capability


def _require_resolved_requests(value, what):
    if type(value) is not _ResolvedRequests or \
            value._capability is not _RESOLVED_REQUESTS_CAPABILITY:
        _fail("E_UNRESOLVED_WORK",
              f"{what} work must be the resolver-issued object returned by "
              "resolve_requests")
    return value


# ---------------------------------------------------------------------------
# trusted root and read-once IO
# ---------------------------------------------------------------------------

def derive_trusted_root(bundle_path):
    """The trusted root is DERIVED from the bundle's own location.

    Never a parameter. A caller-supplied root lets the producer point the
    resolver at a directory it staged for the occasion, which turns every
    path check below into decoration.
    """
    resolved = pathlib.Path(bundle_path).resolve(strict=True)
    if not resolved.is_file():
        _fail("E_ROOT_NOT_DERIVED", f"{bundle_path} is not a regular file")
    return resolved.parent


def resolve_path(path_text, trusted_root):
    """Resolve a relative path inside the trusted root. No traversal, no links."""
    trusted_root = pathlib.Path(trusted_root).resolve()
    if not isinstance(path_text, str) or not path_text:
        _fail("E_PATH", "artifact path must be a non-empty string")
    candidate = pathlib.Path(path_text)
    if candidate.is_absolute():
        _fail("E_PATH", f"artifact path {path_text!r} is absolute")
    if ".." in candidate.parts:
        _fail("E_PATH", f"artifact path {path_text!r} traverses upward")
    probe = trusted_root
    for part in candidate.parts:
        probe = probe / part
        if probe.is_symlink():
            _fail("E_PATH",
                  f"artifact path component {probe.name!r} is a symlink; its "
                  f"target can be repointed after validation")
    try:
        resolved = (trusted_root / candidate).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _fail("E_MISSING", f"artifact {path_text!r} is unreadable: {exc}")
    if not str(resolved).startswith(str(trusted_root) + os.sep):
        _fail("E_PATH", f"artifact {path_text!r} resolves outside the trusted root")
    if not resolved.is_file():
        _fail("E_MISSING", f"artifact {path_text!r} is not a regular file")
    return resolved


@contextlib.contextmanager
def resolution_context():
    """Create one read-once and path-ownership scope for a bundle."""
    state = {"reads": {}, "owners": {}}
    token = _RESOLUTION_CONTEXT.set(state)
    try:
        yield state
    finally:
        _RESOLUTION_CONTEXT.reset(token)


def claim_evidence_path(path_text, owner, trusted_root):
    """Give one canonical evidence path to exactly one run-specific owner."""
    state = _RESOLUTION_CONTEXT.get()
    if state is None:
        return resolve_path(path_text, trusted_root)
    resolved = resolve_path(path_text, trusted_root)
    key = str(resolved)
    previous = state["owners"].get(key)
    if previous is not None and previous != owner:
        _fail("E_EVIDENCE_PATH_REUSE",
              f"canonical path {path_text!r} is claimed by both {previous!r} "
              f"and {owner!r}")
    state["owners"][key] = owner
    return resolved


def read_once(path_text, declared_sha256, trusted_root):
    """Open ONCE, hash and return the same buffer.

    E2's lesson: hashing a path and then re-opening it to parse it is a
    time-of-check/time-of-use gap the red team won 74 times out of 400 with no
    privileges. There must be exactly one set of bytes, and the hash and the
    parse must both come from it. O_NOFOLLOW closes the last-component swap;
    st_nlink == 1 refuses a hardlinked alias that can be replaced under us.
    """
    resolved = resolve_path(path_text, trusted_root)
    state = _RESOLUTION_CONTEXT.get()
    key = str(resolved)
    if state is not None and key in state["reads"]:
        previous_digest, data = state["reads"][key]
        if previous_digest != declared_sha256:
            _fail("E_PATH_DIGEST_CONFLICT",
                  f"canonical path {path_text!r} was first bound to "
                  f"{previous_digest} and is now declared as {declared_sha256}")
        return resolved, data
    try:
        fd = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        _fail("E_MISSING", f"artifact {path_text!r} is unopenable: {exc}")
    try:
        info = os.fstat(fd)
        if info.st_nlink != 1:
            _fail("E_HARDLINK",
                  f"artifact {path_text!r} has {info.st_nlink} links; an aliased "
                  f"file can be replaced through another name after validation")
        with os.fdopen(fd, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        try:
            os.close(fd)
        except OSError:
            pass
        _fail("E_MISSING", f"artifact {path_text!r} is unreadable: {exc}")
    actual = canon.sha256_bytes(data)
    if actual != declared_sha256:
        _fail("E_HASH",
              f"artifact {path_text!r} hashes to {actual} but the record declares "
              f"{declared_sha256}")
    if state is not None:
        state["reads"][key] = (declared_sha256, data)
    return resolved, data


def parse_once(data, what):
    """Parse the exact bytes that were hashed. Strict: no floats, no dup keys."""
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        _fail("E_SCHEMA", f"{what} is not ASCII: {exc}")
    try:
        return canon.loads_strict(text)
    except canon.DuplicateKeyError as exc:
        _fail("E_DUPLICATE_KEY", f"{what} has duplicate JSON keys: {exc}")
    except ValueError as exc:
        _fail("E_SCHEMA", f"{what} is not strict JSON: {exc}")


# ---------------------------------------------------------------------------
# schema gate
# ---------------------------------------------------------------------------

# The subprocess environment is scrubbed to an allowlist. Inheriting the parent
# env hands LD_PRELOAD and PYTHONWARNINGS to a process whose whole job is to be
# uncorrupted.
SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "HOME")


def schema_errors(kind, document):
    """Run the draft-2020 schema under an isolated system Python.

    An absent schema engine is not a pass. It raises.
    """
    if not SYSTEM_PYTHON.exists():
        _fail("E_SCHEMA",
              f"system python {SYSTEM_PYTHON} is unavailable; the schema gate "
              f"cannot be skipped")
    env = {key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ}
    env["PATH"] = env.get("PATH", "/usr/bin:/bin")
    try:
        result = subprocess.run(
            [str(SYSTEM_PYTHON), "-I", str(SCHEMA_WORKER)],
            input=json.dumps({"kind": kind, "document": document}),
            capture_output=True, text=True, timeout=120, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        _fail("E_SCHEMA", f"schema worker failed: {exc}")
    if result.returncode != 0 or not result.stdout.strip():
        _fail("E_SCHEMA",
              f"schema worker error: "
              f"{(result.stdout or result.stderr).strip()[:200]}")
    payload = json.loads(result.stdout)
    if "engine_error" in payload:
        _fail("E_SCHEMA", f"schema engine: {payload['engine_error']}")
    return [f"{err['path']}: {err['message']}" for err in payload["errors"]]


# The ONLY paths where a genuine boolean is permitted. The type gate rejects
# bools everywhere else because bool subclasses int, so True reads as a count of
# one. These two fields are real booleans, and they are enumerated by exact path
# rather than by name so a bool cannot appear anywhere else and claim to be one
# of them.
#
# This map exists because the gate was strictly-correct and therefore useless:
# gate_record refused every bool, so validate_aggregate raised E_TYPE on its OWN
# sealed output and could never once have run -- the function documented as "what
# makes the label unfakeable" was dead on arrival, and no test called it, so 135
# green tests said nothing about it. A gate that rejects the thing it is meant to
# protect is not strict; it is absent.
ALLOWED_BOOL_PATHS = frozenset({
    "aggregate.conservative_relief",
    "aggregate.meets_ten_percent_gate",
})


def gate_record(kind, record, what):
    """Schema, then the integer type gate, then the self-digest. In that order.

    The type gate must run before ANY comparison downstream: a float compares
    equal to an int and satisfies every ordering test, and bool is a subclass of
    int. JSON Schema cannot catch either, because draft6+ defines `integer` as
    any number with a zero fractional part -- 900000.0 IS an integer to the
    schema engine.
    """
    if not isinstance(record, dict) or \
            record.get("schema_version") != ACTIVE_SCHEMA_VERSION:
        _fail("E_SCHEMA_VERSION",
              f"{what} must use E2A schema_version {ACTIVE_SCHEMA_VERSION}; "
              "v1 is retained only as historical unsafe evidence")
    errors = schema_errors(kind, record)
    if errors:
        _fail("E_SCHEMA", f"{what}: {errors[0]}")
    try:
        canon.check_integers(record, what, ALLOWED_BOOL_PATHS)
    except ValueError as exc:
        _fail("E_TYPE", str(exc))
    if record.get("record_sha256") != canon.record_digest(record):
        _fail("E_DIGEST", f"{what} digest does not cover its body")
    return record


def _check_bools(record, fields, what):
    """type(x) is bool, explicitly. `x == True` is satisfied by 1 and by 1.0."""
    for field in fields:
        value = record.get(field, canon.MISSING)
        if value is canon.MISSING:
            _fail("E_TYPE", f"{what} has no {field}")
        if type(value) is not bool:
            _fail("E_TYPE",
                  f"{what}.{field} is {type(value).__name__} {value!r}, not a "
                  f"bool; 1 and 1.0 both compare equal to True")


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------

def _resolve_anchor_common(receipt, kind, plan, trusted_root, expect_digest,
                           what):
    """Validate bindings, invoke the verifier, then apply policy gates."""
    if receipt is None:
        _fail("E_ANCHOR_MISSING",
              f"{what} is absent; SUM_ALL_PAIRS_V1 sums a cohort that must have "
              f"been fixed before the results were seen, and an unanchored plan "
              f"is a plan that can be written afterwards")
    anchor_kind = receipt["anchor_kind"]
    if receipt["status"] != "OK":
        _fail("E_ANCHOR_STATUS",
              f"{what} reports status {receipt['status']} "
              f"({receipt['reason_code']!r})")
    if receipt["verifier_kind"] == "NONE":
        _fail("E_ANCHOR_VERIFY", f"{what} names no verifier")
    if receipt["token_hash_algorithm"] != "sha256":
        _fail("E_ANCHOR_VERIFY",
              f"{what} uses {receipt['token_hash_algorithm']}; E2A record "
              "digests and message imprints are SHA-256")
    if receipt["trust_root_pin_location"] != "E2A_FROZEN_CONSTANT":
        _fail("E_ANCHOR_TRUST_ROOT",
              f"{what} takes its trust root from "
              f"{receipt['trust_root_pin_location']}; the authority must be a "
              f"frozen code constant, because a producer that chooses the "
              f"authority can be the authority")
    try:
        anchors.check_trust_root(receipt)
    except anchors.AnchorError as exc:
        _fail(exc.code, str(exc).split(": ", 1)[1])

    # The imprint must equal a digest we RECOMPUTE, never one the record declares.
    if receipt["anchored_digest"] != expect_digest:
        _fail("E_ANCHOR_BINDING",
              f"{what} anchors {receipt['anchored_digest']} but the recomputed "
              f"digest of the thing it claims to anchor is {expect_digest}")
    if receipt["message_imprint_sha256"] != expect_digest:
        _fail("E_ANCHOR_IMPRINT",
              f"{what} message imprint does not cover the recomputed digest")
    if receipt["set_id"] != plan["set_id"]:
        _fail("E_ANCHOR_BINDING", f"{what} names a different set_id")
    if receipt["chain_id"] != plan["chain_id"]:
        _fail("E_ANCHOR_BINDING", f"{what} names a different chain_id")
    _resolved, token = read_once(receipt["token_der_path"],
                                 receipt["token_der_sha256"], trusted_root)
    if not token:
        _fail("E_ANCHOR_VERIFY", f"{what} token is empty")

    try:
        verified = anchors.verify_token(receipt, token)
    except anchors.AnchorError as exc:
        _fail(exc.code, str(exc).split(": ", 1)[1])
    for field in sorted(anchors.VERIFIED_TOKEN_FIELDS):
        if verified[field] != receipt[field]:
            _fail("E_ANCHOR_VERIFY",
                  f"{what}.{field}={receipt[field]!r} disagrees with the "
                  f"authenticated token value {verified[field]!r}")
    try:
        anchors.check_precedence_supported(receipt, verified)
    except anchors.AnchorError as exc:
        _fail(exc.code, str(exc).split(": ", 1)[1])
    if receipt["provenance"] == "SYNTHETIC":
        _fail("E_ANCHOR_PROVENANCE",
              f"{what} is SYNTHETIC; synthetic evidence exercises mechanics and "
              f"can never anchor a physical claim")
    try:
        prop = anchors.check_anchor_kind(anchor_kind)
    except anchors.AnchorError as exc:
        _fail(exc.code, str(exc).split(": ", 1)[1])
    return prop


def bind_plan_anchor(receipt, plan):
    """Schema-check and bind a plan receipt without applying anchor policy."""
    if receipt is None:
        _fail("E_ANCHOR_MISSING", "plan anchor receipt is absent")
    gate_record("plan_anchor", receipt, "plan anchor receipt")
    if receipt["plan_id"] != plan["plan_id"]:
        _fail("E_ANCHOR_BINDING", "plan anchor names a different plan_id")
    if receipt["chain_id"] != plan["chain_id"] or \
            receipt["set_id"] != plan["set_id"]:
        _fail("E_ANCHOR_BINDING", "plan anchor names another chain or set")
    if receipt["anchored_digest"] != plan["record_sha256"] or \
            receipt["message_imprint_sha256"] != plan["record_sha256"]:
        _fail("E_ANCHOR_BINDING", "plan anchor does not bind the resolved plan")
    return receipt


def resolve_plan_anchor(receipt, plan, trusted_root):
    """Resolve a plan receipt and bind every plan-specific identity field."""
    bind_plan_anchor(receipt, plan)
    return _resolve_anchor_common(
        receipt, "plan_anchor", plan, trusted_root, plan["record_sha256"],
        "plan anchor receipt")


def bind_close_anchor(receipt, plan, plan_anchor, ledger):
    """Schema-check and bind a close receipt without applying anchor policy."""
    if receipt is None:
        _fail("E_ANCHOR_MISSING", "ledger close receipt is absent")
    gate_record("ledger_close", receipt, "ledger close receipt")
    bindings = {
        "plan_anchor_id": plan_anchor["anchor_id"],
        "plan_anchor_record_sha256": plan_anchor["record_sha256"],
        "attempt_ledger_sha256": ledger["record_sha256"],
        "ledger_head_sha256": ledger["head_sha256"],
    }
    for field, expected in bindings.items():
        if receipt[field] != expected:
            _fail("E_ANCHOR_BINDING",
                  f"ledger close {field}={receipt[field]!r}, expected "
                  f"{expected!r}")
    if receipt["anchor_kind"] != plan_anchor["anchor_kind"]:
        _fail("E_ANCHOR_BINDING",
              "plan and close receipts use different anchor mechanisms")
    if receipt["anchor_time_utc_us"] < plan_anchor["anchor_time_utc_us"]:
        _fail("E_ANCHOR_ORDER", "ledger close predates the plan anchor")
    if receipt["anchored_digest"] != ledger["record_sha256"] or \
            receipt["message_imprint_sha256"] != ledger["record_sha256"]:
        _fail("E_ANCHOR_BINDING",
              "ledger close does not anchor the complete ledger record")
    return receipt


def resolve_close_anchor(receipt, plan, plan_anchor, ledger, trusted_root):
    """Resolve a close receipt over the complete ledger record."""
    bind_close_anchor(receipt, plan, plan_anchor, ledger)
    return _resolve_anchor_common(
        receipt, "ledger_close", plan, trusted_root,
        ledger["record_sha256"], "ledger close receipt")


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def resolve_plan(plan, trusted_root):
    """Validate the PreRunPlan's internal structure. No ranges, no retries."""
    gate_record("plan", plan, "plan")
    if plan["aggregate_method"] != GATE_CONSTANTS["AGGREGATE_METHOD"]:
        _fail("E_PLAN", "plan does not declare SUM_ALL_PAIRS_V1")
    if plan["gate_constants_digest"] != GATE_CONSTANTS_DIGEST:
        _fail("E_GATE_RETUNED",
              f"plan pins gate constants {plan['gate_constants_digest']} but the "
              f"frozen constants digest to {GATE_CONSTANTS_DIGEST}; a plan that "
              f"pins different thresholds has retuned the gate")
    n_pairs = plan["n_pairs"]
    if n_pairs < MIN_PAIRS:
        _fail("E_PAIRS", f"plan declares {n_pairs} pairs, below MIN_PAIRS="
                         f"{MIN_PAIRS}")
    slots = plan["slots"]
    if len(slots) != 2 * n_pairs:
        _fail("E_PLAN",
              f"plan declares {n_pairs} pairs but {len(slots)} slots; a set has "
              f"exactly 2N measured slots")

    # The ABBA rotation, checked against the DECLARED pair count rather than
    # against len(slots). E2's repetition-set check derived the expected order
    # from len(pairs), which makes it a tautology: drop a pair and the expected
    # order shrinks to match.
    expected_order = []
    for index in range(n_pairs):
        first = ("OPTIMIZED_SERVER_ONLY_CONTROL" if index % 2 == 0
                 else "Q_PIM_TREATMENT")
        second = ("Q_PIM_TREATMENT" if index % 2 == 0
                  else "OPTIMIZED_SERVER_ONLY_CONTROL")
        expected_order.extend([first, second])
    if plan["declared_order"] != expected_order:
        _fail("E_ORDER", "plan declared_order is not the frozen ABBA rotation")
    if plan["order_digest"] != canon.digest(plan["declared_order"]):
        _fail("E_ORDER", "plan order_digest does not cover declared_order")

    seen_nonces = set()
    for index, slot in enumerate(slots):
        if slot["slot_index"] != index:
            _fail("E_PLAN",
                  f"slot_index values must be contiguous 0..{len(slots) - 1}; "
                  f"a gap can hide a planned attempt")
        if slot["pair_index"] != index // 2:
            _fail("E_PLAN", f"slot {index} names pair {slot['pair_index']}")
        if slot["ordinal_in_pair"] != index % 2:
            _fail("E_PLAN", f"slot {index} has wrong ordinal_in_pair")
        if slot["role"] != expected_order[index]:
            _fail("E_ORDER",
                  f"slot {index} role {slot['role']} contradicts the ABBA "
                  f"rotation, which requires {expected_order[index]}")
        if slot["run_nonce"] in seen_nonces:
            _fail("E_PLAN",
                  f"run_nonce {slot['run_nonce']!r} is reused across slots; one "
                  f"nonce cannot identify two runs")
        seen_nonces.add(slot["run_nonce"])

    if plan["policy_digest_control"] == plan["policy_digest_treatment"]:
        _fail("E_SAME_POLICY",
              "plan gives control and treatment the same policy digest; "
              "comparing a policy against itself measures noise")

    # The typed instrument gate, applied to the PLAN and not only to each
    # timeline. The plan is the anchored commitment: if it can pair NVML_BOARD
    # with SERVER_WALL, the whole set is designed around a promoted scope and
    # every later per-timeline refusal arrives after the experiment is spent.
    # Capability is typed, and relabelling the instrument string grants nothing.
    try:
        e2_integrator.check_scope(plan["instrument_kind"], plan["scope"])
    except e2_integrator.TimelineError as exc:
        _fail(exc.code, str(exc).split(": ", 1)[1])
    overlap = sorted(set(plan["included_rails"]) & set(plan["excluded_rails"]))
    if overlap:
        _fail("E_SCOPE", f"rails {overlap} are both included and excluded")
    if plan["scope"] == "GPU_BOARD" and not plan["board_uuids"]:
        _fail("E_SCOPE",
              "a GPU_BOARD plan must name the board(s) it will measure; this "
              "host has more than one board")
    return plan


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------

def entry_body(entry):
    return {key: value for key, value in entry.items() if key != "entry_sha256"}


def resolve_ledger(ledger, plan, plan_anchor=None):
    """Validate the exact one-attempt state machine for every planned slot."""
    gate_record("ledger", ledger, "ledger")
    if ledger["set_id"] != plan["set_id"]:
        _fail("E_LEDGER_BINDING", "ledger names a different set_id than the plan")
    if ledger["chain_id"] != plan["chain_id"]:
        _fail("E_LEDGER_BINDING", "ledger names a different chain_id")
    if ledger["plan_sha256"] != plan["record_sha256"]:
        _fail("E_LEDGER_BINDING",
              f"ledger binds plan {ledger['plan_sha256']} but the resolved plan "
              f"digests to {plan['record_sha256']}")

    entries = ledger["entries"]
    expected_count = 2 + 3 * len(plan["slots"])
    if len(entries) != expected_count:
        code = ("E_SLOT_UNCOVERED" if len(entries) < expected_count
                else "E_SLOT_DOUBLE_TERMINAL")
        _fail(code,
              f"ledger has {len(entries)} entries; the exact state machine for "
              f"{len(plan['slots'])} slots requires {expected_count}")
    previous = ZERO_DIGEST
    previous_time = None
    clock_epoch = None
    for index, entry in enumerate(entries):
        if entry["seq"] != index:
            _fail("E_LEDGER_SEQ",
                  f"entry {index} declares seq {entry['seq']}; sequence numbers "
                  f"must be contiguous from 0 so a removal is a gap")
        if entry["prev_entry_sha256"] != previous:
            _fail("E_LEDGER_CHAIN",
                  f"entry {index} chains to {entry['prev_entry_sha256']} but the "
                  f"previous entry hashes to {previous}")
        recomputed = canon.digest(entry_body(entry))
        if entry["entry_sha256"] != recomputed:
            _fail("E_LEDGER_CHAIN",
                  f"entry {index} declares digest {entry['entry_sha256']} but its "
                  f"body hashes to {recomputed}")
        previous = entry["entry_sha256"]
        if clock_epoch is None:
            clock_epoch = entry["clock_epoch_id"]
        elif entry["clock_epoch_id"] != clock_epoch:
            _fail("E_CLOCK_EPOCH",
                  f"ledger entry {index} changes clock epoch from {clock_epoch} "
                  f"to {entry['clock_epoch_id']}")
        if previous_time is not None and entry["monotonic_us"] < previous_time:
            _fail("E_LEDGER_TIME",
                  f"ledger entry {index} moves backward from {previous_time} to "
                  f"{entry['monotonic_us']} us")
        previous_time = entry["monotonic_us"]
        if entry["attempt_ordinal"] != 0:
            _fail("E_UNDECLARED_RETRY",
                  f"ledger entry {index} has attempt_ordinal "
                  f"{entry['attempt_ordinal']}; the plan permits one attempt")
        if entry["status"] != "OK":
            _fail("E_NON_OK_ATTEMPT",
                  f"ledger entry {index} has status {entry['status']} "
                  f"({entry['reason_code']!r})")
    if ledger["head_sha256"] != previous:
        _fail("E_LEDGER_HEAD",
              f"ledger head {ledger['head_sha256']} is not the last entry digest "
              f"{previous}")

    null_evidence = {
        "timeline_id": None,
        "timeline_record_sha256": None,
        "lifecycle_record_sha256": None,
        "request_outcome_record_sha256": None,
    }
    null_times = {
        "window_start_us": None,
        "window_end_us": None,
        "drain_acknowledged_us": None,
    }
    first = entries[0]
    if first["entry_kind"] != "PLAN_ANCHOR" or first["slot_index"] != -1 or \
            first["run_nonce"] is not None:
        _fail("E_LEDGER_STATE", "ledger must start with one global PLAN_ANCHOR")
    for field, expected in null_evidence.items():
        if first[field] != expected:
            _fail("E_LEDGER_STATE", f"PLAN_ANCHOR must have null {field}")
    for field, expected in null_times.items():
        if first[field] != expected:
            _fail("E_LEDGER_STATE", f"PLAN_ANCHOR must have null {field}")
    if first["plan_anchor_record_sha256"] is None:
        _fail("E_LEDGER_BINDING", "PLAN_ANCHOR does not bind its receipt")
    if plan_anchor is not None and first["plan_anchor_record_sha256"] != \
            plan_anchor["record_sha256"]:
        _fail("E_LEDGER_BINDING", "ledger binds a different plan anchor receipt")

    terminal = {}
    cursor = 1
    for slot_index, plan_slot in enumerate(plan["slots"]):
        group = entries[cursor:cursor + 3]
        cursor += 3
        expected_kinds = ("SLOT_OPEN", "SLOT_ATTEMPT_START",
                          "SLOT_ATTEMPT_END")
        if tuple(entry["entry_kind"] for entry in group) != expected_kinds:
            _fail("E_LEDGER_STATE",
                  f"slot {slot_index} must be OPEN -> START -> END")
        for entry in group:
            if entry["slot_index"] != slot_index:
                _fail("E_LEDGER_BINDING",
                      f"ledger processes slot {entry['slot_index']} where "
                      f"slot {slot_index} is required")
            if entry["run_nonce"] != plan_slot["run_nonce"]:
                _fail("E_LEDGER_BINDING",
                      f"slot {slot_index} ledger nonce does not match the plan")
            if entry["plan_anchor_record_sha256"] is not None:
                _fail("E_LEDGER_STATE", "slot events must not carry plan anchor")
        for entry in group[:2]:
            for field, expected in null_evidence.items():
                if field == "timeline_id":
                    expected = None
                if entry[field] != expected:
                    _fail("E_LEDGER_STATE",
                          f"{entry['entry_kind']} must have null {field}")
        for field, expected in null_times.items():
            if group[0][field] != expected:
                _fail("E_LEDGER_STATE", f"SLOT_OPEN must have null {field}")
        start_entry = group[1]
        if start_entry["window_start_us"] is None or \
                start_entry["window_end_us"] is not None or \
                start_entry["drain_acknowledged_us"] is not None:
            _fail("E_LEDGER_TIME_BINDING",
                  f"slot {slot_index} START must bind only window_start_us")
        end_entry = group[2]
        required_end = ("timeline_id", "timeline_record_sha256",
                        "lifecycle_record_sha256",
                        "request_outcome_record_sha256")
        if any(end_entry[field] is None for field in required_end):
            _fail("E_LEDGER_BINDING",
                  f"slot {slot_index} END does not bind all evidence records")
        if any(end_entry[field] is None for field in null_times):
            _fail("E_LEDGER_TIME_BINDING",
                  f"slot {slot_index} END does not bind window and drain times")
        if start_entry["monotonic_us"] != start_entry["window_start_us"] or \
                end_entry["monotonic_us"] != end_entry["window_end_us"] or \
                start_entry["window_start_us"] != end_entry["window_start_us"]:
            _fail("E_LEDGER_TIME_BINDING",
                  f"slot {slot_index} START/END timestamps do not bind one window")
        if not (end_entry["window_start_us"] <=
                end_entry["drain_acknowledged_us"] <=
                end_entry["window_end_us"]):
            _fail("E_LEDGER_TIME_BINDING",
                  f"slot {slot_index} drain lies outside its ledger window")
        terminal[slot_index] = {
            "open": group[0], "start": start_entry, "end": end_entry,
        }

    seal = entries[-1]
    if seal["entry_kind"] != "SET_SEAL" or seal["slot_index"] != -1 or \
            seal["run_nonce"] is not None or \
            seal["plan_anchor_record_sha256"] is not None:
        _fail("E_LEDGER_STATE", "ledger must end with one global SET_SEAL")
    for field, expected in null_evidence.items():
        if seal[field] != expected:
            _fail("E_LEDGER_STATE", f"SET_SEAL must have null {field}")
    for field, expected in null_times.items():
        if seal[field] != expected:
            _fail("E_LEDGER_STATE", f"SET_SEAL must have null {field}")
    return terminal


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------

def lifecycle_state_digest(actions, when):
    """Derive the boundary state immediately before `when`."""
    active_actions = sorted(
        action["action_id"] for action in actions
        if action["enqueue_us"] < when < action["ack_us"])
    active_leases = set()
    for action in sorted(actions, key=lambda item: (item["ack_us"],
                                                    item["action_id"])):
        if action["ack_us"] >= when:
            continue
        if action["action_kind"] == "LEASE_ACQUIRE":
            active_leases.add(action["lease_id"])
        elif action["action_kind"] == "LEASE_RELEASE":
            active_leases.discard(action["lease_id"])
    return canon.digest({
        "schema": "e2a.lifecycle-state.v2",
        "active_actions": active_actions,
        "active_leases": sorted(active_leases),
    })


def resolve_lifecycle(record, timeline, plan_slot=None, resolved_outputs=None):
    """Every action bound, closed, and inside the paid window.

    Why this is an ENERGY check and not bookkeeping: energy is integrated over
    [window_start, window_end]. Any work that runs outside that interval is real,
    is caused by the run, and is charged to nobody. A treatment that moves its
    weight staging into a PREFETCH before the window, or lets an async D2H land
    after it, or defers CLEANUP past the end marker, spends real joules that the
    integral never sees. That is not a smaller energy bill; it is a smaller
    window.
    """
    resolved_outputs = _require_resolved_requests(
        resolved_outputs, "lifecycle")
    gate_record("lifecycle", record, "lifecycle")
    if record["timeline_id"] != timeline["timeline_id"]:
        _fail("E_LIFECYCLE_BINDING", "lifecycle names a different timeline")
    if record["run_nonce"] != timeline["run_nonce"]:
        _fail("E_LIFECYCLE_BINDING", "lifecycle names a different run_nonce")
    if record["clock_epoch_id"] != timeline["clock_epoch_id"]:
        _fail("E_CLOCK_EPOCH",
              "lifecycle timestamps are on a different clock epoch than the "
              "timeline window; they cannot be compared")
    start = timeline["window_start_us"]
    end = timeline["window_end_us"]
    if record["window_start_us"] != start or record["window_end_us"] != end:
        _fail("E_LIFECYCLE_BINDING",
              "lifecycle window does not match the timeline window")

    actions = record["actions"]
    by_id = {}
    for action in actions:
        what = f"action {action['action_id']}"
        if action["action_id"] in by_id:
            _fail("E_LIFECYCLE_ORDER", f"duplicate {what}")
        by_id[action["action_id"]] = action
        for field in ("enqueue_us", "start_us", "end_us", "ack_us"):
            if not canon.is_int(action[field]):
                _fail("E_TYPE", f"{what}.{field} is not an integer")
        if not (action["enqueue_us"] <= action["start_us"] <= action["end_us"]
                <= action["ack_us"]):
            _fail("E_LIFECYCLE_ORDER",
                  f"{what} timestamps are not ordered "
                  f"enqueue<=start<=end<=ack")
        if action["state"] != "COMPLETE":
            _fail("E_LIFECYCLE_OPEN",
                  f"{what} ended in {action['state']}; a failed or canceled action "
                  "cannot support a complete measurement")
        if action["action_kind"] not in FREE_ACTIONS:
            if action["start_us"] < start:
                _fail("E_ACTION_UNBOUND",
                      f"{what} ({action['action_kind']}) starts at "
                      f"{action['start_us']} us, before the paid window opens at "
                      f"{start} us; work staged outside the window burns real "
                      f"energy that the integral never charges")
            if action["ack_us"] > end:
                _fail("E_DEFERRED_CLEANUP",
                      f"{what} ({action['action_kind']}) is acknowledged at "
                      f"{action['ack_us']} us, after the paid window closes at "
                      f"{end} us; deferring work past the end marker moves its "
                      f"energy off the bill")

    for action in actions:
        parent_id = action["parent_action_id"]
        if parent_id is None:
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            _fail("E_LIFECYCLE_ORDER",
                  f"action {action['action_id']} names missing parent {parent_id}")
        if parent["ack_us"] > action["start_us"]:
            _fail("E_LIFECYCLE_ORDER",
                  f"action {action['action_id']} starts before parent {parent_id} "
                  "is acknowledged")

    required_kinds = {"EXEC", "RESULT_EMIT"}
    kinds = {action["action_kind"] for action in actions}
    missing_kinds = sorted(required_kinds - kinds)
    if missing_kinds:
        _fail("E_LIFECYCLE_CAUSAL",
              f"lifecycle has no required actions {missing_kinds}")

    timing = resolved_outputs.timing
    expected_requests = set(timing)
    if expected_requests:
        for kind in ("EXEC", "RESULT_EMIT"):
            covered = set()
            for action in actions:
                if action["action_kind"] == kind:
                    covered.update(action["request_ids"])
            if covered != expected_requests:
                _fail("E_LIFECYCLE_CAUSAL",
                      f"{kind} covers requests {sorted(covered)}, expected "
                      f"{sorted(expected_requests)}")

        submitted = set()
        for action in actions:
            if action["action_kind"] == "QUEUE_SUBMIT":
                submitted.update(action["request_ids"])
        if submitted != expected_requests:
            _fail("E_ARRIVAL_BINDING",
                  f"QUEUE_SUBMIT covers requests {sorted(submitted)}, expected "
                  f"{sorted(expected_requests)}")

        # Arrival and completion are parsed from run-specific output artifacts,
        # not copied from the lifecycle. Each request must be submitted exactly
        # at its replay arrival, executed through its final token, and emitted
        # only after both output completion and the covering EXEC ACK.
        for request_id in sorted(expected_requests):
            submits = [action for action in actions
                       if action["action_kind"] == "QUEUE_SUBMIT" and
                       request_id in action["request_ids"]]
            if len(submits) != 1:
                _fail("E_ARRIVAL_BINDING",
                      f"request {request_id} has {len(submits)} QUEUE_SUBMIT "
                      "actions; exactly one must bind its arrival")
            arrival = timing[request_id]["arrival_us"]
            submit = submits[0]
            if submit["enqueue_us"] != arrival or submit["start_us"] != arrival:
                _fail("E_ARRIVAL_BINDING",
                      f"request {request_id} arrival is {arrival} us but "
                      f"QUEUE_SUBMIT starts at {submit['start_us']} us")
            if submit["ack_us"] > timing[request_id]["dispatched_us"]:
                _fail("E_LIFECYCLE_CAUSAL",
                      f"request {request_id} dispatch precedes QUEUE_SUBMIT ACK")

            executions = [action for action in actions
                          if action["action_kind"] == "EXEC" and
                          request_id in action["request_ids"]]
            covering_exec = [action for action in executions
                             if action["start_us"] <=
                             timing[request_id]["dispatched_us"] and
                             action["end_us"] >=
                             timing[request_id]["last_token_us"]]
            if not covering_exec:
                _fail("E_LIFECYCLE_CAUSAL",
                      f"request {request_id} has no EXEC covering dispatch "
                      "through output completion")

            emits = [action for action in actions
                     if action["action_kind"] == "RESULT_EMIT" and
                     request_id in action["request_ids"]]
            if len(emits) != 1:
                _fail("E_LIFECYCLE_CAUSAL",
                      f"request {request_id} has {len(emits)} RESULT_EMIT "
                      "actions; exactly one is required")
            earliest_emit = max(
                [timing[request_id]["last_token_us"]] +
                [action["ack_us"] for action in covering_exec])
            if emits[0]["start_us"] < earliest_emit:
                _fail("E_LIFECYCLE_CAUSAL",
                      f"request {request_id} RESULT_EMIT starts at "
                      f"{emits[0]['start_us']} us before causal boundary "
                      f"{earliest_emit} us")

    # Leases must be balanced. A lease held past the window can pin a clock or a
    # residency that the next run inherits for free.
    open_leases = {}
    lease_intervals = []
    lease_actions = sorted(
        (action for action in actions
         if action["action_kind"] in ("LEASE_ACQUIRE", "LEASE_RELEASE")),
        key=lambda action: (action["start_us"], action["ack_us"],
                            action["action_id"]))
    for action in lease_actions:
        if action["action_kind"] == "LEASE_ACQUIRE":
            if action["lease_id"] is None:
                _fail("E_LEASE_OPEN", "LEASE_ACQUIRE names no lease_id")
            if action["lease_id"] in open_leases:
                _fail("E_LEASE_OPEN",
                      f"lease {action['lease_id']!r} acquired twice")
            open_leases[action["lease_id"]] = action
        elif action["action_kind"] == "LEASE_RELEASE":
            if action["lease_id"] not in open_leases:
                _fail("E_LEASE_OPEN",
                      f"lease {action['lease_id']!r} released without an acquire")
            acquire = open_leases[action["lease_id"]]
            if acquire["ack_us"] > action["start_us"]:
                _fail("E_LEASE_ORDER",
                      f"lease {action['lease_id']!r} release starts before its "
                      "acquire is acknowledged")
            lease_intervals.append((acquire, action))
            del open_leases[action["lease_id"]]
    if open_leases:
        _fail("E_LEASE_OPEN",
              f"leases {sorted(open_leases)} are still held at the end of the "
              f"run")

    # Pairing alone is not ownership. Every action that can consume queue,
    # residency, backend, or result resources must start after a lease is
    # acquired and finish before that same lease begins release. QUEUE_DRAIN is
    # the post-release proof that no work remains, so it is the only non-lease
    # action outside this coverage requirement.
    lease_boundary_kinds = {
        "LEASE_ACQUIRE", "LEASE_RELEASE", "QUEUE_DRAIN",
    }
    for work in actions:
        if work["action_kind"] in lease_boundary_kinds:
            continue
        covered = any(
            acquire["ack_us"] <= work["start_us"] and
            work["ack_us"] <= release["start_us"]
            for acquire, release in lease_intervals)
        if not covered:
            _fail("E_LEASE_COVERAGE",
                  f"{work['action_kind']} action {work['action_id']!r} is not "
                  "fully covered by one acquired lease")

    # Drain must be ACKNOWLEDGED, and the acknowledgement recomputed rather than
    # asserted. A `drained: true` field read with .get() is dead for False, 0,
    # 0.0, and omission alike.
    drains = [a for a in actions if a["action_kind"] == "QUEUE_DRAIN"]
    if len(drains) != 1:
        _fail("E_DRAIN_UNACKED",
              f"the run declares {len(drains)} QUEUE_DRAIN actions; exactly one "
              "must close the paid window")
    last_drain = drains[0]["ack_us"]
    if last_drain != max(action["ack_us"] for action in actions):
        _fail("E_DRAIN_UNACKED", "QUEUE_DRAIN is not the last acknowledged action")
    if record["drain_acknowledged_us"] != last_drain:
        _fail("E_DRAIN_UNACKED",
              f"record declares drain acknowledged at "
              f"{record['drain_acknowledged_us']} us but the last QUEUE_DRAIN is "
              f"acknowledged at {last_drain} us")
    outstanding = outstanding_at(record["actions"], end)
    if outstanding:
        _fail("E_LIFECYCLE_OPEN",
              f"{len(outstanding)} actions {sorted(outstanding)[:4]} are still "
              f"outstanding at window_end {end} us")
    derived_entry = lifecycle_state_digest(actions, start)
    derived_exit = lifecycle_state_digest(actions, end)
    if record["entry_state_digest"] != derived_entry or \
            record["exit_state_digest"] != derived_exit:
        _fail("E_LIFECYCLE_STATE",
              "lifecycle boundary state digests do not match the action replay")
    clean_state = lifecycle_state_digest([], start)
    if derived_entry != clean_state or derived_exit != clean_state:
        _fail("E_LIFECYCLE_STATE",
              "lifecycle does not enter and exit with clean action/lease state")
    if plan_slot is not None:
        if record["entry_state_digest"] != \
                plan_slot["expected_entry_state_digest"] or \
                record["exit_state_digest"] != \
                plan_slot["expected_exit_state_digest"]:
            _fail("E_LIFECYCLE_STATE",
                  "lifecycle boundary state differs from the anchored plan")
    return record


def outstanding_at(actions, when):
    """Recompute what is in flight at a timestamp. Never read a declared count."""
    return [a["action_id"] for a in actions
            if a["enqueue_us"] <= when < a["ack_us"]]


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------

def resolve_manifest(manifest, plan):
    """Bind the request set and its derived work vector to the plan."""
    gate_record("request_set", manifest, "request set manifest")
    if manifest["record_sha256"] != plan["request_set_manifest_sha256"]:
        _fail("E_REQUEST_BINDING", "plan pins a different request set manifest")
    if manifest["set_id"] != plan["set_id"]:
        _fail("E_REQUEST_BINDING", "manifest names a different set_id")
    if manifest["sampler_digest"] != plan["sampler_digest"]:
        _fail("E_REQUEST_BINDING", "manifest sampler differs from the plan")

    seen = set()
    prompt_tokens = 0
    max_tokens = 0
    for entry in manifest["entries"]:
        rid = entry["request_id"]
        if rid in seen:
            _fail("E_REQUEST_DUPLICATE", f"manifest repeats request {rid}")
        seen.add(rid)
        body = {key: value for key, value in entry.items()
                if key != "entry_sha256"}
        if entry["entry_sha256"] != canon.digest(body):
            _fail("E_DIGEST", f"manifest entry {rid} digest does not cover it")
        if entry["model_digest"] != plan["model_digest"]:
            _fail("E_MODEL_MISMATCH", f"manifest request {rid} changes the model")
        if entry["tokenizer_digest"] != plan["tokenizer_digest"]:
            _fail("E_REQUEST_BINDING",
                  f"manifest request {rid} changes the tokenizer")
        if entry["min_prefix_tokens"] < MIN_PREFIX_TOKENS or \
                entry["min_prefix_tokens"] > entry["max_tokens"]:
            _fail("E_WORK_GATE",
                  f"manifest request {rid} min_prefix_tokens="
                  f"{entry['min_prefix_tokens']} is outside the frozen range "
                  f"[{MIN_PREFIX_TOKENS}, {entry['max_tokens']}]")
        if entry["length_tolerance_tokens"] > \
                MAX_LENGTH_TOLERANCE_TOKENS:
            _fail("E_WORK_GATE",
                  f"manifest request {rid} length_tolerance_tokens="
                  f"{entry['length_tolerance_tokens']} exceeds frozen maximum "
                  f"{MAX_LENGTH_TOLERANCE_TOKENS}")
        prompt_tokens += entry["prompt_tokens"]
        max_tokens += entry["max_tokens"]
    expected = {
        "requests": len(manifest["entries"]),
        "prompt_tokens": prompt_tokens,
        "max_generated_tokens": max_tokens,
    }
    if manifest["work_vector"] != expected:
        _fail("E_REQUEST_BINDING",
              f"manifest work_vector is {manifest['work_vector']}, expected "
              f"{expected}")
    return manifest


OUTPUT_FIELDS = frozenset({
    "schema", "timeline_id", "run_nonce", "request_id", "model_digest",
    "tokenizer_digest", "sampling_mode", "seed", "arrival_us", "dispatched_us",
    "token_events", "stop_reason",
})


def resolve_output_artifact(item, entry, timeline, trusted_root, what):
    """Parse and recompute the realized output from the read-once bytes."""
    owner = ("output", timeline["run_nonce"], item["request_id"])
    claim_evidence_path(item["output_artifact_path"], owner, trusted_root)
    _path, data = read_once(item["output_artifact_path"],
                            item["output_artifact_sha256"], trusted_root)
    output = parse_once(data, f"{what} output {item['request_id']}")
    if not isinstance(output, dict) or set(output) != OUTPUT_FIELDS:
        _fail("E_OUTPUT_FORMAT", "output artifact does not match e2a.output.v2")
    if output["schema"] != "e2a.output.v2":
        _fail("E_OUTPUT_FORMAT", f"unknown output schema {output['schema']!r}")
    try:
        canon.check_integers(output, f"{what} output {item['request_id']}")
    except ValueError as exc:
        _fail("E_TYPE", str(exc))
    # Manifest arrivals are replay-relative. The run's monotonic arrival is the
    # paid-window origin plus that frozen offset.
    expected_arrival = timeline["window_start_us"] + entry["arrival_us"]
    bindings = {
        "timeline_id": timeline["timeline_id"],
        "run_nonce": timeline["run_nonce"],
        "request_id": item["request_id"],
        "model_digest": entry["model_digest"],
        "tokenizer_digest": entry["tokenizer_digest"],
        "sampling_mode": entry["sampling_mode"],
        "seed": entry["seed"],
        "arrival_us": expected_arrival,
        "dispatched_us": item["dispatched_us"],
        "stop_reason": item["stop_reason"],
    }
    for field, expected in bindings.items():
        if output[field] != expected:
            _fail("E_OUTPUT_BINDING",
                  f"output {field}={output[field]!r}, expected {expected!r}")

    events = output["token_events"]
    if not isinstance(events, list):
        _fail("E_OUTPUT_FORMAT", "output token_events is not an array")
    token_ids = []
    previous_time = output["dispatched_us"]
    for index, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != {"token_id", "emitted_us"}:
            _fail("E_OUTPUT_FORMAT", f"token event {index} has wrong fields")
        if not canon.is_int(event["token_id"]) or event["token_id"] < 0 or \
                not canon.is_int(event["emitted_us"]):
            _fail("E_TYPE", f"token event {index} is not integer-valued")
        if event["emitted_us"] < previous_time:
            _fail("E_OUTPUT_ORDER", f"token event {index} moves backward in time")
        previous_time = event["emitted_us"]
        token_ids.append(event["token_id"])
    if len(token_ids) != item["realized_output_tokens"]:
        _fail("E_OUTPUT_COUNT",
              f"output has {len(token_ids)} tokens, record declares "
              f"{item['realized_output_tokens']}")
    if canon.digest(token_ids) != item["output_token_ids_sha256"]:
        _fail("E_OUTPUT_DIGEST", "token ID digest does not cover parsed tokens")
    expected_last = events[-1]["emitted_us"] if events else output["dispatched_us"]
    if expected_last != item["last_token_us"]:
        _fail("E_OUTPUT_ORDER",
              f"parsed last token is at {expected_last}, record declares "
              f"{item['last_token_us']}")
    return {
        "token_ids": token_ids,
        "arrival_us": output["arrival_us"],
        "dispatched_us": output["dispatched_us"],
        "last_token_us": expected_last,
    }


def resolve_requests(manifest, outcome_record, timeline, trusted_root, what,
                     expected_pair_index=None):
    """Per-request identical work, resolved BEFORE any energy is examined.

    Order matters and is not stylistic. If energy is compared first and work
    second, then every arithmetic result the reader has already seen was computed
    over an unvalidated cohort, and a reviewer who stops reading at the number
    has been misled. Work is a PRECONDITION of the comparison, not a companion
    check to it.

    HONEST LIMIT, stated here because it is load-bearing: the treatment runs on
    different hardware by construction, so bit-identical outputs are NOT
    expectable and this resolver never demands them. What it demands is that both
    roles ran the SAME REQUEST -- same input, same tokenizer, same model digest,
    same seed, same decode parameters, same stop set, same SLO -- and that each
    realized output is bound to a hashed artifact whose length is within the
    predeclared tolerance and whose stop reason agrees. That catches shorter
    generations, truncation, model swaps, seed drift, and substitution. It does
    NOT prove the outputs are semantically equivalent, and no field here should
    be read as proving that. See CONTRACT.md section 9.
    """
    gate_record("request_outcomes", outcome_record, what)
    if outcome_record["timeline_id"] != timeline["timeline_id"]:
        _fail("E_REQUEST_BINDING", f"{what} names a different timeline")
    if outcome_record["run_nonce"] != timeline["run_nonce"]:
        _fail("E_REQUEST_BINDING", f"{what} names a different run_nonce")
    if outcome_record["manifest_sha256"] != manifest["record_sha256"]:
        _fail("E_REQUEST_BINDING",
              f"{what} binds manifest {outcome_record['manifest_sha256']} but the "
              f"resolved manifest digests to {manifest['record_sha256']}")
    if outcome_record["set_id"] != manifest["set_id"]:
        _fail("E_REQUEST_BINDING", f"{what} names a different set_id")
    if outcome_record["role"] != timeline["role"]:
        _fail("E_REQUEST_BINDING", f"{what} names a different role")
    if expected_pair_index is not None and \
            outcome_record["pair_index"] != expected_pair_index:
        _fail("E_REQUEST_BINDING", f"{what} names a different pair_index")

    entries = {e["request_id"]: e for e in manifest["entries"]}
    outcomes = outcome_record["outcomes"]
    seen = set()
    token_ids_by_request = {}
    timing_by_request = {}
    derived_outcomes = {"met": 0, "tardy": 0, "rejected": 0, "canceled": 0}
    for item in outcomes:
        rid = item["request_id"]
        if rid in seen:
            _fail("E_REQUEST_DUPLICATE", f"{what} reports request {rid} twice")
        seen.add(rid)
        entry = entries.get(rid)
        if entry is None:
            _fail("E_REQUEST_SUBSTITUTED",
                  f"{what} reports request {rid!r}, which the predeclared "
                  f"manifest does not contain; swapping a request for another "
                  f"keeps the count identical while changing the work")
        if item["manifest_entry_sha256"] != entry["entry_sha256"]:
            _fail("E_REQUEST_SUBSTITUTED",
                  f"{what} request {rid} binds manifest entry "
                  f"{item['manifest_entry_sha256']} but the manifest entry hashes "
                  f"to {entry['entry_sha256']}")
        if item["realized_model_digest"] != entry["model_digest"]:
            _fail("E_MODEL_MISMATCH",
                  f"{what} request {rid} ran model "
                  f"{item['realized_model_digest']} but the plan pinned "
                  f"{entry['model_digest']}; a smaller model is cheaper and is "
                  f"not the same work")
        if item["weights_transform_id"] != "IDENTITY":
            _fail("E_MODEL_TRANSFORM_UNCERTIFIED",
                  f"{what} request {rid} ran weights transformed by "
                  f"{item['weights_transform_id']}; E2A has no evidence that a "
                  f"weight transform preserves the computation, so it refuses "
                  f"rather than assuming")
        if item["realized_seed"] != entry["seed"]:
            _fail("E_SEED_MISMATCH",
                  f"{what} request {rid} ran seed {item['realized_seed']}, "
                  f"planned {entry['seed']}")
        if item["realized_sampling_mode"] != entry["sampling_mode"]:
            _fail("E_SAMPLING_MISMATCH",
                  f"{what} request {rid} sampling mode drifted")
        if entry["sampling_mode"] != "GREEDY":
            _fail("E_SAMPLING_REFUSED",
                  f"{what} request {rid} is SAMPLED; E2A can only certify "
                  f"correspondence between two runs under greedy decoding, so it "
                  f"refuses to compare sampled work rather than pretending a "
                  f"distribution match is a work match")
        if item["realized_output_tokens"] > entry["max_tokens"]:
            _fail("E_OUTPUT_OVERRUN",
                  f"{what} request {rid} produced "
                  f"{item['realized_output_tokens']} tokens above the planned cap "
                  f"{entry['max_tokens']}")
        if item["stop_reason"] == "MAX_TOKENS" and \
                item["realized_output_tokens"] != entry["max_tokens"]:
            _fail("E_OUTPUT_TRUNCATED",
                  f"{what} request {rid} claims it stopped at the token cap but "
                  f"produced {item['realized_output_tokens']} of "
                  f"{entry['max_tokens']}")
        if item["terminal_outcome"] not in ("met", "tardy"):
            _fail("E_REQUEST_OUTCOME",
                  f"{what} request {rid} terminated {item['terminal_outcome']}; a "
                  f"rejected or canceled request is cheaper and is not the same "
                  f"work")
        expected_arrival = timeline["window_start_us"] + entry["arrival_us"]
        if item["arrival_us"] != expected_arrival:
            _fail("E_ARRIVAL_BINDING",
                  f"{what} request {rid} arrived at {item['arrival_us']} us, "
                  f"expected replay arrival {expected_arrival} us")
        if not (item["arrival_us"] <= item["dispatched_us"] <=
                item["last_token_us"]):
            _fail("E_REQUEST_ORDER", f"{what} request {rid} ends before it starts")
        if item["arrival_us"] < timeline["window_start_us"] or \
                item["last_token_us"] > timeline["window_end_us"]:
            _fail("E_REQUEST_OUT_OF_WINDOW",
                  f"{what} request {rid} runs outside the paid window "
                  f"[{timeline['window_start_us']}, {timeline['window_end_us']}]")
        derived_terminal = ("met" if item["last_token_us"] -
                            item["arrival_us"] <= entry["slo_deadline_us"]
                            else "tardy")
        if item["terminal_outcome"] != derived_terminal:
            _fail("E_SLO_OUTCOME",
                  f"{what} request {rid} declares {item['terminal_outcome']}, "
                  f"but its measured latency derives {derived_terminal}")
        certificate = item["certificate"]
        if certificate["certificate_kind"] == "NONE" or \
                certificate["referee_verdict"] == "NOT_RUN":
            _fail("E_CERT_MISSING",
                  f"{what} request {rid} carries no correctness certificate; an "
                  f"output nobody checked is not evidence that the work was done")
        if certificate["referee_verdict"] != "AGREE":
            _fail("E_CERT_FAIL",
                  f"{what} request {rid} certificate verdict "
                  f"{certificate['referee_verdict']}")
        if certificate["prefix_agreement_tokens"] < entry["min_prefix_tokens"]:
            _fail("E_CERT_FAIL",
                  f"{what} request {rid} agrees on only "
                  f"{certificate['prefix_agreement_tokens']} tokens, below the "
                  f"predeclared minimum {entry['min_prefix_tokens']}")
        resolved_output = resolve_output_artifact(
            item, entry, timeline, trusted_root, what)
        token_ids_by_request[rid] = resolved_output["token_ids"]
        timing_by_request[rid] = {
            "arrival_us": resolved_output["arrival_us"],
            "dispatched_us": resolved_output["dispatched_us"],
            "last_token_us": resolved_output["last_token_us"],
        }
        derived_outcomes[derived_terminal] += 1

    missing = sorted(set(entries) - seen)
    if missing:
        _fail("E_REQUEST_MISSING",
              f"{what} does not report requests {missing[:4]}; a request that is "
              f"planned and not run is work that was not done")
    if timeline["offered_work"] != len(entries) or \
            timeline["completed_work"] != len(outcomes) or \
            timeline["outcomes"] != derived_outcomes:
        _fail("E_REQUEST_ACCOUNTING",
              "timeline work and outcome counters do not match resolved requests")
    met_ppm = derived_outcomes["met"] * 1000000 // len(entries)
    if met_ppm < MIN_SLO_MET_PPM:
        _fail("E_SLO_COHORT",
              f"{what} has {met_ppm} SLO-met ppm, below frozen minimum "
              f"{MIN_SLO_MET_PPM}; an opaque SLO policy cannot authorize a "
              "low-quality physical comparison")
    return _ResolvedRequests(
        outcome_record, token_ids_by_request, timing_by_request,
        _RESOLVED_REQUESTS_CAPABILITY)


def check_same_work(control_outcomes, treatment_outcomes, manifest):
    """Per-request equivalence between the two roles. Runs before any energy."""
    for role, value in (("control", control_outcomes),
                        ("treatment", treatment_outcomes)):
        _require_resolved_requests(value, role)
    control_record = control_outcomes.record
    treatment_record = treatment_outcomes.record
    control_tokens = control_outcomes.token_ids
    treatment_tokens = treatment_outcomes.token_ids
    control = {i["request_id"]: i for i in control_record["outcomes"]}
    treatment = {i["request_id"]: i for i in treatment_record["outcomes"]}
    if set(control) != set(treatment):
        _fail("E_REQUEST_MISSING",
              "control and treatment did not run the same request IDs")
    entries = {e["request_id"]: e for e in manifest["entries"]}
    for rid in sorted(control):
        c, t = control[rid], treatment[rid]
        entry = entries[rid]
        if c["terminal_outcome"] != t["terminal_outcome"]:
            _fail("E_OUTCOME_MISMATCH",
                  f"request {rid} terminated {c['terminal_outcome']} for control "
                  f"and {t['terminal_outcome']} for treatment; buying energy with "
                  f"a different SLO result is not relief")
        if c["stop_reason"] != t["stop_reason"]:
            _fail("E_STOP_REASON_MISMATCH",
                  f"request {rid} stopped for {c['stop_reason']} in control and "
                  f"{t['stop_reason']} in treatment")
        delta = abs(c["realized_output_tokens"] - t["realized_output_tokens"])
        if delta > entry["length_tolerance_tokens"]:
            _fail("E_OUTPUT_SHORT",
                  f"request {rid} produced {c['realized_output_tokens']} tokens "
                  f"in control and {t['realized_output_tokens']} in treatment, a "
                  f"difference of {delta} above the predeclared tolerance "
                  f"{entry['length_tolerance_tokens']}; generating less text is "
                  f"the cheapest way to look faster")
        if c["realized_model_digest"] != t["realized_model_digest"]:
            _fail("E_MODEL_MISMATCH", f"request {rid} ran different models")
        if c["realized_seed"] != t["realized_seed"]:
            _fail("E_SEED_MISMATCH", f"request {rid} ran different seeds")
        if control_tokens is not None and treatment_tokens is not None:
            c_tokens = control_tokens[rid]
            t_tokens = treatment_tokens[rid]
            prefix = 0
            for left, right in zip(c_tokens, t_tokens):
                if left != right:
                    break
                prefix += 1
            first_divergence = (-1 if c_tokens == t_tokens else prefix)
            verdict = "AGREE" if prefix >= entry["min_prefix_tokens"] else "DIVERGE"
            for role, item in (("control", c), ("treatment", t)):
                certificate = item["certificate"]
                expected = {
                    "certificate_kind": "GREEDY_PREFIX_AGREEMENT",
                    "prefix_agreement_tokens": prefix,
                    "first_divergence_index": first_divergence,
                    "referee_verdict": verdict,
                }
                if certificate != expected:
                    _fail("E_CERT_FAIL",
                          f"request {rid} {role} certificate {certificate} does "
                          f"not match parsed outputs {expected}")
            if verdict != "AGREE":
                _fail("E_CERT_FAIL",
                      f"request {rid} parsed outputs agree for only {prefix} "
                      f"tokens, below {entry['min_prefix_tokens']}")
    return True
