"""Per-model matrix precision policy for quantized artifacts (WS17, opt-in).

The frozen campaigns require every 2-D tensor of the Qwen and Gemma proxies to be F16
(``preflight._precision_check(..., "F16")``). A quantized artifact needs a different rule:

* ``allowed``: the quantization types any 2-D tensor may have (e.g. ``Q4_0`` plus ``Q6_K`` for the output
  head of a plain ``llama-quantize Q4_0`` file, or ``Q4_K``/``Q6_K`` for a Q4_K_M file);
* ``ffn`` (optional): the single type every ``ffn_gate``/``ffn_up``/``ffn_down`` tensor must have. Phone FFN shards
  need it: the shard writer and the worker require gate/up/down and all layers to share one type
  (``native/ffn_shard_gguf.py``, ``ffn-split-worker.cpp``). An imatrix-quantized Q4_0 file keeps the first
  n_layer/8 ``ffn_down`` tensors in Q4_1 and fails this rule, which is the point.

A 2-D F32 tensor is refused unless ``F32`` is listed. When a model carries no ``matrix_precision`` the
preflight keeps its frozen F16 rule, so existing campaigns are byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

SCHEMA = "ws17-matrix-precision-v1"
KNOWN_TYPES = frozenset((
    "F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
    "Q8_K", "IQ4_NL", "IQ4_XS", "MXFP4",
))
FFN_MATRIX_MARKERS = (".ffn_gate.", ".ffn_up.", ".ffn_down.")


class MatrixPrecisionError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatrixPrecisionError(message)


@dataclass(frozen=True)
class MatrixPrecisionSpec:
    allowed: tuple[str, ...]
    ffn: str | None = None

    def __post_init__(self) -> None:
        _require(type(self.allowed) is tuple and len(self.allowed) > 0, "matrix precision needs allowed types")
        _require(len(set(self.allowed)) == len(self.allowed), "matrix precision allowed types repeat")
        _require(all(type(name) is str and name in KNOWN_TYPES for name in self.allowed),
                 "matrix precision names an unknown type")
        _require(self.ffn is None or (type(self.ffn) is str and self.ffn in self.allowed),
                 "matrix precision ffn type must be one of the allowed types")

    @classmethod
    def from_json(cls, value: object) -> "MatrixPrecisionSpec":
        _require(type(value) is dict and set(value) <= {"allowed", "ffn"} and "allowed" in value,
                 "matrix precision is {allowed: [...], ffn?: TYPE}")
        allowed = value["allowed"]
        _require(type(allowed) is list, "matrix precision allowed types must be a list")
        return cls(tuple(allowed), value.get("ffn"))

    def to_json(self) -> dict[str, object]:
        return {"allowed": list(self.allowed), **({"ffn": self.ffn} if self.ffn is not None else {})}


def is_ffn_matrix(tensor_id: str) -> bool:
    return any(marker in tensor_id for marker in FFN_MATRIX_MARKERS)


def evaluate(tensors: Iterable[object], spec: MatrixPrecisionSpec) -> tuple[bool, str]:
    """``tensors`` rows carry ``tensor_id``, ``quantization`` and ``shape`` (``ModelManifest.tensors``)."""
    matrix = [row for row in tensors if len(row.shape) >= 2]
    invalid = sorted(row.tensor_id for row in matrix if row.quantization not in spec.allowed)
    ffn = [row for row in matrix if is_ffn_matrix(row.tensor_id)]
    ffn_types = sorted({row.quantization for row in ffn})
    ffn_invalid = [] if spec.ffn is None else sorted(row.tensor_id for row in ffn if row.quantization != spec.ffn)
    counts: dict[str, int] = {}
    for row in matrix:
        counts[row.quantization] = counts.get(row.quantization, 0) + 1
    ok = bool(matrix) and not invalid and not ffn_invalid and (spec.ffn is None or bool(ffn))
    detail = (str(len(matrix)) + " matrix tensors " + ",".join(f"{k}={v}" for k, v in sorted(counts.items()))
              + "; allowed=" + ",".join(spec.allowed)
              + "; ffn=" + (spec.ffn or "any") + " (found " + (",".join(ffn_types) or "none") + ")"
              + "; invalid=" + ",".join((invalid + ffn_invalid)[:8]))
    return ok, detail


def parse_cli(value: str) -> MatrixPrecisionSpec:
    """argparse ``type=``: a JSON object."""
    import json

    try:
        return MatrixPrecisionSpec.from_json(json.loads(value))
    except (ValueError, MatrixPrecisionError) as error:
        import argparse

        raise argparse.ArgumentTypeError(str(error)) from error


def spec_or_none(value: Mapping[str, object] | None) -> MatrixPrecisionSpec | None:
    return None if value is None else MatrixPrecisionSpec.from_json(dict(value))


CALIBRATION_PRIOR_SCHEMA = "ws17-calibration-prior-v1"


@dataclass(frozen=True)
class CalibrationPrior:
    """Opt-in (WS17): derive a new artifact's route profiles from its role's existing calibration runs.

    The catalog fits each role's latency model from the 74-request S41/S42 calibration runs (role ``hot``/``cold``)
    and attaches it to the model's artifact. A new artifact of the same role (e.g. Qwen3-32B Q4_0 for the hot
    role) has no runs of its own, so the fitted service coefficients are multiplied by measured service-time
    ratios (``desktop_service_scale_ppm`` for the desktop route, ``assisted_service_scale_ppm`` for the phone route;
    1,000,000 = unchanged), taken from a rig receipt whose sha256 is bound into the route evidence. With
    ``phone_columns: "model"`` a full-replacement calibration decision is rescaled to the new model's FFN width
    (n_embd must still match). The profiles stay SHADOW priors; the runtime's adaptive stores refine them.
    """

    desktop_service_scale_ppm: int
    assisted_service_scale_ppm: int
    receipt_sha256: str
    phone_columns: str = "recorded"

    def __post_init__(self) -> None:
        for name in ("desktop_service_scale_ppm", "assisted_service_scale_ppm"):
            value = getattr(self, name)
            _require(type(value) is int and 10_000 <= value <= 100_000_000, name + " must be 10,000-100,000,000 ppm")
        _require(type(self.receipt_sha256) is str and self.receipt_sha256.startswith("sha256:")
                 and len(self.receipt_sha256) == 71, "calibration prior receipt must be a sha256 digest")
        _require(self.phone_columns in ("recorded", "model"), "calibration prior phone_columns is recorded|model")

    @classmethod
    def from_json(cls, value: object) -> "CalibrationPrior":
        _require(type(value) is dict and value.get("schema") == CALIBRATION_PRIOR_SCHEMA,
                 "calibration prior schema")
        fields = {"desktop_service_scale_ppm", "assisted_service_scale_ppm", "receipt_sha256", "phone_columns"}
        _require(set(value) - {"schema"} <= fields, "calibration prior has unknown fields")
        return cls(**{key: value[key] for key in fields if key in value})

    def to_json(self) -> dict[str, object]:
        return {"schema": CALIBRATION_PRIOR_SCHEMA, "desktop_service_scale_ppm": self.desktop_service_scale_ppm,
                "assisted_service_scale_ppm": self.assisted_service_scale_ppm,
                "receipt_sha256": self.receipt_sha256, "phone_columns": self.phone_columns}

    def scale(self, arm_is_desktop: bool) -> float:
        return (self.desktop_service_scale_ppm if arm_is_desktop else self.assisted_service_scale_ppm) / 1_000_000

    def rescale_decision(self, decision: Mapping[str, object], feed_forward_length: int) -> dict[str, object]:
        row = dict(decision)
        if self.phone_columns == "model" and row.get("execution_mode") == "full_replacement":
            row["phone_columns"] = feed_forward_length
        return row
