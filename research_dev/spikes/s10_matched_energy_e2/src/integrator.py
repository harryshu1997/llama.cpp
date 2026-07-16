#!/usr/bin/env python3
"""Deterministic integer timeline integrator for E2. Standard library only.

Implements CONTRACT.md section 3: left-edge zero-order hold, integer-exact.

    energy_nj = sum(power_mw[i] * (timestamp_us[i+1] - timestamp_us[i]))

No floats, no division, no interpolation, no idle-baseline subtraction. Gross
energy only, over a window that must be BRACKETED by real samples on both sides.

This module does not decide anything about relief. It normalizes, gates sample
quality, and integrates. The comparator decides.
"""

from __future__ import annotations

import pathlib

import e2_canon as canon

# Frozen sample-quality gates. Set BEFORE any measurement exists so they cannot be
# tuned to admit a favoured artifact. See CONTRACT.md section 7.
MIN_INDEPENDENT_UPDATES = 100
MAX_SAMPLE_GAP_US = 250_000
MIN_WINDOW_US = 1_000_000
MAX_SAMPLES = 1_000_000
WINDOW_TOLERANCE_US = 50_000
MIN_PAIRS = 8

# Vendor-stated accuracy floors, in mW. NVML power.draw is documented "accurate to
# within +/- 5 watts" (see INSTRUMENT_AUDIT.md section 1). An NVML timeline that
# claims better accuracy than the vendor does is not more precise, it is wrong.
INSTRUMENT_UNCERTAINTY_FLOOR_MW = {
    "NVML_BOARD": 5000,
    "RAPL_PACKAGE": 0,
    "EXTERNAL_WALL_METER": 0,
    "BMC_INPUT_POWER": 0,
    "SYNTHETIC": 0,
}

# Typed capability -> the ONE scope it may claim. A free-form instrument label
# grants nothing (CONTRACT.md section 2).
INSTRUMENT_ALLOWED_SCOPE = {
    "NVML_BOARD": {"GPU_BOARD"},
    "RAPL_PACKAGE": set(),          # component counter; never a wall, never a board
    "EXTERNAL_WALL_METER": {"SERVER_WALL"},
    "BMC_INPUT_POWER": {"SERVER_WALL"},
    "SYNTHETIC": {"GPU_BOARD", "SERVER_WALL"},   # mechanics only; no physical label
}


class TimelineError(ValueError):
    """A timeline defect. Always fatal; never downgraded to a warning."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code


def _fail(code, message):
    raise TimelineError(code, message)


def normalize_samples(raw):
    """Validate and normalize a raw sample list into integer (us, mW) pairs.

    `raw` is a list of two-element sequences. Every value must be a true int:
    the type gate runs BEFORE any ordering comparison, because a float compares
    equal to an int and would slip past the monotonicity and gap checks.
    """
    if not isinstance(raw, list):
        _fail("E_SCHEMA", "samples must be an array")
    if len(raw) < 2:
        _fail("E_SCHEMA", f"a timeline needs at least 2 samples, got {len(raw)}")
    if len(raw) > MAX_SAMPLES:
        _fail("E_SCHEMA", f"{len(raw)} samples exceeds MAX_SAMPLES={MAX_SAMPLES}")

    out = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            _fail("E_SCHEMA", f"sample {index} is not a [timestamp_us, power_mw] pair")
        timestamp, power = item
        try:
            canon.check_integers(list(item), f"samples[{index}]")
        except ValueError as exc:
            _fail("E_SCHEMA", str(exc))
        if timestamp < 0:
            _fail("E_SCHEMA", f"sample {index} has negative timestamp {timestamp}")
        if power < 0:
            _fail("E_SCHEMA", f"sample {index} has negative power {power} mW")
        out.append((timestamp, power))

    for index in range(1, len(out)):
        previous, current = out[index - 1][0], out[index][0]
        if current == previous:
            _fail("E_NONMONOTONIC",
                  f"samples {index - 1} and {index} share timestamp {current} us; "
                  f"a duplicated timestamp has no defined hold interval")
        if current < previous:
            _fail("E_NONMONOTONIC",
                  f"sample {index} timestamp {current} us goes backwards from "
                  f"{previous} us; a counter reset or a reordered log is not a "
                  f"timeline")
    return out


def window_slice(samples, window_start_us, window_end_us):
    """The samples whose hold intervals intersect the window, plus its brackets.

    Load-bearing for quality gating. Energy is integrated over the WINDOW, but an
    earlier revision counted value changes over the WHOLE ARTIFACT. Samples
    outside the window contribute zero energy, so padding there was free: an
    adversary appended busy samples outside the paid window and bought unlimited
    `independent_updates` for a window that still contained only 57 real changes.
    Quality must be measured over the same interval the energy is taken from.
    """
    first = 0
    for index in range(len(samples)):
        if samples[index][0] <= window_start_us:
            first = index
        else:
            break
    last = len(samples) - 1
    for index in range(len(samples) - 1, -1, -1):
        if samples[index][0] >= window_end_us:
            last = index
        else:
            break
    return samples[first:last + 1]


def window_quality(samples, window_start_us, window_end_us):
    """Return value changes and the largest observation gap inside the window.

    Samples at the end marker only bracket the integral; their new value applies
    after the paid interval and cannot buy measurement quality. Samples before the
    start marker establish the held value but likewise cannot count as updates.
    """
    if window_end_us <= window_start_us:
        _fail("E_WINDOW", "quality window is empty or inverted")
    if window_start_us < samples[0][0] or window_end_us > samples[-1][0]:
        _fail("E_UNBRACKETED", "quality window is not bracketed by samples")

    bracketed = window_slice(samples, window_start_us, window_end_us)
    gap = max(bracketed[index][0] - bracketed[index - 1][0]
              for index in range(1, len(bracketed)))
    updates = 0
    previous_power = next(
        power for timestamp, power in reversed(samples)
        if timestamp <= window_start_us)
    for timestamp, power in samples:
        if timestamp <= window_start_us:
            continue
        if timestamp >= window_end_us:
            break
        if power != previous_power:
            updates += 1
        previous_power = power
    return updates, gap


def independent_updates(samples):
    """Count CHANGES in the sensor value, not rows.

    Load-bearing. NVML on Ampere returns a 1-second average, so a 10 Hz poll
    yields ~10 identical rows per real observation. Counting rows would let a
    sampler manufacture arbitrary "sample counts" from a slow sensor. See
    INSTRUMENT_AUDIT.md: the existing A6000 trace has 322 rows and 57 changes.

    This is a conservative quality gate, not proof of statistical independence.
    """
    return sum(1 for i in range(1, len(samples))
               if samples[i][1] != samples[i - 1][1])


def max_gap_us(samples):
    return max(samples[i][0] - samples[i - 1][0] for i in range(1, len(samples)))


def integrate(samples, window_start_us, window_end_us):
    """Left-edge zero-order-hold integral over a bracketed window. Integer-exact.

    Returns energy in nJ (mW * us). Frozen in CONTRACT.md section 3.

    This function is PURE ARITHMETIC plus its own preconditions (a non-empty
    window, bracketed by real samples). It deliberately does NOT enforce
    MIN_WINDOW_US: "this window is too short to be a credible measurement" is a
    policy gate, applied by gate_window/validate_timeline. Mixing the two would
    make the frozen integration rule untestable on small hand-checkable inputs
    and would conflate a wrong integral with an inadmissible one.
    """
    canon.check_integers([window_start_us, window_end_us], "window")
    if window_end_us <= window_start_us:
        _fail("E_WINDOW",
              f"window [{window_start_us}, {window_end_us}] is empty or inverted")
    if window_start_us < samples[0][0]:
        _fail("E_UNBRACKETED",
              f"window starts at {window_start_us} us, before the first sample at "
              f"{samples[0][0]} us; extrapolating before the first observation "
              f"invents power that was never measured")
    if window_end_us > samples[-1][0]:
        _fail("E_UNBRACKETED",
              f"window ends at {window_end_us} us, after the last sample at "
              f"{samples[-1][0]} us; extrapolating past the last observation "
              f"invents power that was never measured")

    energy_nj = 0
    for index in range(len(samples) - 1):
        start, power = samples[index]
        end = samples[index + 1][0]
        # Clip the hold interval to the window. Integer arithmetic throughout.
        lo = max(start, window_start_us)
        hi = min(end, window_end_us)
        if hi > lo:
            energy_nj += power * (hi - lo)
    return energy_nj


def gate_window(window_start_us, window_end_us):
    """The window POLICY gate, kept separate from the integration arithmetic."""
    window_us = window_end_us - window_start_us
    if window_us < MIN_WINDOW_US:
        _fail("E_WINDOW",
              f"window is {window_us} us, below MIN_WINDOW_US={MIN_WINDOW_US}; a "
              f"window shorter than the sensor's own averaging period cannot "
              f"resolve a policy difference")
    return window_us


def gate_samples(samples, instrument_kind):
    """Apply the frozen sample-quality gates. Raises on any failure."""
    updates = independent_updates(samples)
    if updates < MIN_INDEPENDENT_UPDATES:
        _fail("E_UPDATES",
              f"timeline has {updates} independent sensor updates ({len(samples)} "
              f"rows) but MIN_INDEPENDENT_UPDATES={MIN_INDEPENDENT_UPDATES}; "
              f"oversampling a slow sensor does not create information")
    gap = max_gap_us(samples)
    if gap > MAX_SAMPLE_GAP_US:
        _fail("E_GAP",
              f"largest sample gap is {gap} us, above MAX_SAMPLE_GAP_US="
              f"{MAX_SAMPLE_GAP_US}; energy across an unobserved gap is a guess")
    if instrument_kind not in INSTRUMENT_ALLOWED_SCOPE:
        _fail("E_SCHEMA", f"unknown instrument_kind {instrument_kind!r}")
    return updates, gap


def check_scope(instrument_kind, scope):
    """Typed capability, not a name. Raises E_SCOPE on any promotion attempt."""
    allowed = INSTRUMENT_ALLOWED_SCOPE.get(instrument_kind)
    if allowed is None:
        _fail("E_SCHEMA", f"unknown instrument_kind {instrument_kind!r}")
    if not allowed:
        _fail("E_SCOPE",
              f"instrument_kind {instrument_kind} is a component counter and can "
              f"support no measurement scope; it cannot be promoted to {scope}")
    if scope not in allowed:
        _fail("E_SCOPE",
              f"instrument_kind {instrument_kind} may only claim scope "
              f"{sorted(allowed)}, not {scope!r}; capability is typed, and "
              f"relabelling the instrument string grants nothing")


def uncertainty_nj_floor(instrument_kind, window_us, measured_units=1):
    """Minimum worst-case uncertainty over the window and measured units."""
    if type(measured_units) is not int or measured_units <= 0:
        _fail("E_UNCERTAINTY",
              f"measured_units must be a positive integer, got {measured_units!r}")
    floor_mw = INSTRUMENT_UNCERTAINTY_FLOOR_MW.get(instrument_kind, 0)
    return floor_mw * window_us * measured_units


def resolve_artifact(path_text, trusted_root):
    """Resolve a raw artifact path inside a trusted root. No traversal, no symlink.

    Rejects absolute paths, `..` traversal, and symlinks. A symlink is refused
    even when it currently points inside the root, because the target can be
    repointed after validation.
    """
    trusted_root = pathlib.Path(trusted_root).resolve()
    if not isinstance(path_text, str) or not path_text:
        _fail("E_ARTIFACT_PATH", "artifact path must be a non-empty string")
    candidate = pathlib.Path(path_text)
    if candidate.is_absolute():
        _fail("E_ARTIFACT_PATH", f"artifact path {path_text!r} is absolute")
    if ".." in candidate.parts:
        _fail("E_ARTIFACT_PATH",
              f"artifact path {path_text!r} traverses upward")
    target = trusted_root / candidate
    # Check every component for a symlink before resolving, so a link cannot be
    # swapped for a directory after the check.
    probe = trusted_root
    for part in candidate.parts:
        probe = probe / part
        if probe.is_symlink():
            _fail("E_ARTIFACT_PATH",
                  f"artifact path component {probe.name!r} is a symlink; its "
                  f"target can be repointed after validation")
    try:
        resolved = target.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _fail("E_ARTIFACT_MISSING", f"artifact {path_text!r} is unreadable: {exc}")
    if not str(resolved).startswith(str(trusted_root) + "/"):
        _fail("E_ARTIFACT_PATH",
              f"artifact {path_text!r} resolves outside the trusted root")
    if not resolved.is_file():
        _fail("E_ARTIFACT_MISSING", f"artifact {path_text!r} is not a regular file")
    return resolved


def read_verified_artifact(path_text, declared_sha256, trusted_root):
    """Read the artifact ONCE and return the exact bytes that were hashed.

    Returns (resolved_path, data). The caller MUST parse the returned bytes, not
    re-open the path.

    An earlier revision hashed the path in one open and then re-opened the same
    path to parse it. That is a time-of-check/time-of-use gap: two opens can see
    two different files, and an adversary who swaps the file between them gets a
    record whose declared energy describes bytes other than the ones it pins. The
    red team won that race 74 times out of 400 with no privileges. Reading once
    and passing the buffer forward closes it: there is only one set of bytes, and
    both the hash and the integral are taken from it.
    """
    resolved = resolve_artifact(path_text, trusted_root)
    try:
        with open(resolved, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        _fail("E_ARTIFACT_MISSING", f"artifact {path_text!r} is unreadable: {exc}")
    actual = canon.sha256_bytes(data)
    if actual != declared_sha256:
        _fail("E_ARTIFACT_HASH",
              f"artifact {path_text!r} hashes to {actual} but the record declares "
              f"{declared_sha256}")
    return resolved, data
