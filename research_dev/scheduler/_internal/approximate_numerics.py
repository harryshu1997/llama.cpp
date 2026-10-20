"""Approximate-numerics qualification for quantized phone helpers (WS17, opt-in evidence class).

The exact gates (byte-identical numerical rows, server token identity) assume the phone reproduces the
desktop's arithmetic. A quantized HTP kernel cannot: the tiled HVX path quantizes activations with one
f16 scale per 128 values (the desktop CPU uses one per 32) and HMX rounds tiles to f16. Exact gates
therefore fail by construction for quantized helpers, and free-running greedy sequences are the wrong
metric too: one early flip changes every later token (S33 reached 43-59 % exact-sequence agreement).

This module defines a two-part receipt instead. It never replaces an exact gate: a helper without the
``approximate-numerics-v1`` policy keeps every exact gate unchanged.

1. **Row screen** (per call, real activations). The server's ``S41SERVERFFNROW stage=returned`` lines carry the
   relative L2 distance of each phone FFN row to the desktop's local shadow of the same row
   (``S41_SERVER_FFN_ROW_DIAGNOSTIC_STEPS``). Every value must be finite, the median <= 1e-2 and the p99 <= 3e-2
   (WS8 FEASIBILITY section 5). Rationale: f16 phone rows sit at ~1e-3 to 3e-3 and a Q8_0 slice measured
   4.7e-3 to 5.0e-3 with argmax exact (S41); per-128 activation scales add roughly 0.4-1 % per element, so a
   correct Q4_0 row should stay well under 1e-2 and a wrong or swapped row (relative L2 ~1) stands out.
2. **Teacher-forced top-1 agreement** (end to end, logits). On the reference (desktop-only) greedy path of each
   prompt, the candidate's top-1 token is read at every position with the reference prefix forced, so a
   flip cannot cascade. The disagreement rate D is compared with the A/A rate A of the desktop against
   itself on the same artifact (e.g. ``-ngl 0`` versus the qualified GPU placement: CPU and CUDA kernels
   differ in the same way the phone and the CPU do). Gate: D <= max(absolute_floor, factor * A + slack),
   defaults 1 %, 2 and 0.5 percentage points. The floor keeps a near-zero A from making the gate a coin
   flip on a handful of near ties; the factor bounds the phone to the desktop's own format noise.
   Disagreements at near ties (reference top-2 log-prob gap below ``near_tie_gap``, default 0.1 nats) are
   reported separately; they are expected and not an error signal by themselves.

Task quality (the WS4 GSM8K non-inferiority test) stays the outcome measure; this receipt only qualifies
the numerics of the helper before a campaign uses it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Iterable, Mapping, Sequence

POLICY_ID = "approximate-numerics-v1"
RECEIPT_SCHEMA = "ws17-approximate-numerics-receipt-v1"
PATH_SCHEMA = "ws17-teacher-forced-path-v1"
ROW_LINE = re.compile(r"^S41SERVERFFNROW stage=returned .*?\blayer=(-?\d+)\b.*?\blocal_rel_l2=(\S+)")


class ApproximateNumericsError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ApproximateNumericsError(message)


@dataclass(frozen=True)
class ApproximateNumericsPolicy:
    row_rel_l2_median_max: float = 1e-2
    row_rel_l2_p99_max: float = 3e-2
    min_row_samples: int = 256
    min_positions: int = 256
    absolute_floor: float = 0.01
    aa_factor: float = 2.0
    aa_slack: float = 0.005
    near_tie_gap: float = 0.1

    def __post_init__(self) -> None:
        for name in ("row_rel_l2_median_max", "row_rel_l2_p99_max", "absolute_floor", "aa_factor", "aa_slack",
                     "near_tie_gap"):
            value = getattr(self, name)
            _require(type(value) is float and math.isfinite(value) and value >= 0.0, name + " must be a float >= 0")
        _require(self.row_rel_l2_median_max <= self.row_rel_l2_p99_max, "median bound above the p99 bound")
        for name in ("min_row_samples", "min_positions"):
            _require(type(getattr(self, name)) is int and getattr(self, name) >= 1, name + " must be an int >= 1")

    @classmethod
    def from_json(cls, value: Mapping[str, object] | None) -> "ApproximateNumericsPolicy":
        if value is None:
            return cls()
        raw = dict(value)
        _require(raw.pop("policy", POLICY_ID) == POLICY_ID, "unknown approximate-numerics policy")
        known = set(cls.__dataclass_fields__)
        _require(set(raw) <= known, "unknown approximate-numerics policy fields: " + ",".join(sorted(set(raw) - known)))
        return cls(**raw)

    def to_json(self) -> dict[str, object]:
        return {"policy": POLICY_ID, **asdict(self)}


def nearest_rank(values: Sequence[float], percent: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "no values")
    rank = max(1, math.ceil(percent / 100.0 * len(ordered)))
    return ordered[rank - 1]


def parse_row_diagnostics(lines: Iterable[str]) -> list[dict[str, object]]:
    """``S41SERVERFFNROW stage=returned`` rows with their relative L2 (``inf``/``nan`` kept as non-finite)."""
    rows = []
    for line in lines:
        match = ROW_LINE.match(line.strip())
        if not match:
            continue
        raw = match.group(2).lower()
        value = float("inf") if raw in ("inf", "+inf") else float("nan") if raw in ("nan", "-nan") else float(raw)
        if value == -1.0:          # the returned row had no local shadow (diagnostic window exhausted)
            continue
        rows.append({"layer": int(match.group(1)), "rel_l2": value})
    return rows


def row_screen(rows: Sequence[Mapping[str, object]], policy: ApproximateNumericsPolicy) -> dict[str, object]:
    values = [float(row["rel_l2"]) for row in rows]
    non_finite = sum(1 for value in values if not math.isfinite(value))
    finite = [value for value in values if math.isfinite(value)]
    result: dict[str, object] = {"samples": len(values), "non_finite": non_finite}
    if finite:
        by_layer: dict[int, list[float]] = {}
        for row in rows:
            if math.isfinite(float(row["rel_l2"])):
                by_layer.setdefault(int(row["layer"]), []).append(float(row["rel_l2"]))
        result.update(median=nearest_rank(finite, 50), p99=nearest_rank(finite, 99), max=max(finite),
                      worst_layer_median=max((nearest_rank(v, 50), layer) for layer, v in by_layer.items())[1])
    reasons = []
    if len(values) < policy.min_row_samples:
        reasons.append(f"only {len(values)} row samples (< {policy.min_row_samples})")
    if non_finite:
        reasons.append(f"{non_finite} non-finite rows")
    if finite and result["median"] > policy.row_rel_l2_median_max:
        reasons.append(f"median rel-L2 {result['median']:.3e} > {policy.row_rel_l2_median_max:.1e}")
    if finite and result["p99"] > policy.row_rel_l2_p99_max:
        reasons.append(f"p99 rel-L2 {result['p99']:.3e} > {policy.row_rel_l2_p99_max:.1e}")
    result["status"] = "PASS" if not reasons and finite else "FAIL"
    result["reasons"] = reasons or ([] if finite else ["no finite rows"])
    return result


def _check_path(path: Mapping[str, object], name: str) -> list[dict[str, object]]:
    _require(path.get("schema") == PATH_SCHEMA, name + " is not a teacher-forced path")
    prompts = path.get("prompts")
    _require(type(prompts) is list and prompts, name + " has no prompts")
    return prompts


def disagreement(reference: Mapping[str, object], candidate: Mapping[str, object],
                 policy: ApproximateNumericsPolicy) -> dict[str, object]:
    """Per-position top-1 of ``candidate`` against ``reference``'s greedy token, on the same forced prefixes.

    Path rows: ``{"prompt_id", "positions": [{"token": int, "top": [[id, logprob], ...]}]}``; the reference
    ``token`` is its greedy choice, the candidate's ``top[0]`` its argmax with the reference prefix forced.
    """
    ref_prompts = {row["prompt_id"]: row for row in _check_path(reference, "reference")}
    cand_prompts = {row["prompt_id"]: row for row in _check_path(candidate, "candidate")}
    _require(set(ref_prompts) == set(cand_prompts), "reference and candidate cover different prompts")
    digest = reference.get("path_sha256")
    _require(digest is None or candidate.get("forced_reference_sha256") == digest,
             "candidate was forced along another path")
    positions = disagreements = near_ties = non_finite = 0
    first = []
    for prompt_id in sorted(ref_prompts):
        ref_positions = ref_prompts[prompt_id]["positions"]
        cand_positions = cand_prompts[prompt_id]["positions"]
        _require(len(ref_positions) == len(cand_positions), f"prompt {prompt_id}: position counts differ")
        for index, (ref, cand) in enumerate(zip(ref_positions, cand_positions)):
            positions += 1
            ref_top, cand_top = ref["top"], cand["top"]
            values = [lp for _, lp in ref_top] + [lp for _, lp in cand_top]
            if not all(type(v) in (int, float) and math.isfinite(v) for v in values) or not cand_top:
                non_finite += 1
                continue
            if int(cand_top[0][0]) != int(ref["token"]):
                disagreements += 1
                gap = (ref_top[0][1] - ref_top[1][1]) if len(ref_top) > 1 else float("inf")
                if gap < policy.near_tie_gap:
                    near_ties += 1
                if len(first) < 16:
                    first.append({"prompt_id": prompt_id, "position": index, "reference": int(ref["token"]),
                                  "candidate": int(cand_top[0][0]), "reference_top2_gap": gap})
    return {"positions": positions, "disagreements": disagreements, "near_tie_disagreements": near_ties,
            "non_finite": non_finite, "rate": (disagreements / positions) if positions else float("nan"),
            "first_disagreements": first}


def evaluate(*, reference: Mapping[str, object], candidate: Mapping[str, object],
             aa_candidate: Mapping[str, object], rows: Sequence[Mapping[str, object]],
             policy: ApproximateNumericsPolicy | None = None, identity: Mapping[str, object] | None = None
             ) -> dict[str, object]:
    """The receipt: PASS only when the row screen and the teacher-forced agreement both pass."""
    policy = policy or ApproximateNumericsPolicy()
    screen = row_screen(rows, policy)
    phone = disagreement(reference, candidate, policy)
    aa = disagreement(reference, aa_candidate, policy)
    bound = max(policy.absolute_floor, policy.aa_factor * aa["rate"] + policy.aa_slack) \
        if math.isfinite(aa["rate"]) else float("nan")
    reasons = []
    if phone["positions"] < policy.min_positions:
        reasons.append(f"only {phone['positions']} forced positions (< {policy.min_positions})")
    if phone["non_finite"] or aa["non_finite"]:
        reasons.append("non-finite log-probabilities")
    if not math.isfinite(bound) or not phone["rate"] <= bound:
        reasons.append(f"top-1 disagreement {phone['rate']:.4f} > bound {bound:.4f} (A/A {aa['rate']:.4f})")
    agreement = {"status": "FAIL" if reasons else "PASS", "bound": bound, "phone": phone, "aa": aa,
                 "reasons": reasons}
    status = "PASS" if screen["status"] == "PASS" and agreement["status"] == "PASS" else "FAIL"
    return {"schema": RECEIPT_SCHEMA, "status": status, "policy": policy.to_json(), "row_screen": screen,
            "teacher_forced": agreement, "identity": dict(identity or {})}
