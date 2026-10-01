"""Measured FFN layer placement: which device computes the FFN of each CPU-resident layer (WS11).

The desktop keeps a model's first layers on the CPU (llama-server ``-ngl``); helper phones can compute the
FFN of any of those layers instead, one synchronous call per layer and decode step. This module decides,
per model, which helper (or the desktop CPU itself) owns each such layer so that the predicted system
energy per decoded token is minimal, under

* per-device memory capacity in the device's own shard format (bytes per layer differ: an f16 shard,
  the Pixel's packed Q4_K/Q6_K copy, a Q4_0 copy ...), optionally packed into per-session limits;
* a latency bound (the decode step may be at most ``latency_ppm`` of the desktop-only step);
* per-device availability (absent, quarantined, thermally excluded) and an optional busy cap per token
  (a thermal envelope);
* optionally, a column split of every helper layer (one fraction per model server: the server supports
  exactly "host prefix + one helper suffix" per layer with a single shared width).

Energy model (per decode step at ``rows`` rows; the chain is synchronous, so step time is additive)::

    E_step = sum_l [ P_state(l) * t_l + e_call(l) + P_phone_idle * t_l ] + placement-invariant work

with ``P_state`` = desktop host power while its CPU streams an FFN layer (``ffn``) or while it waits on a
helper call (``wait``), ``t_l`` the layer's time on its owner (CPU FFN time or the helper's measured
round trip), ``e_call`` the helper's energy per call above idle, ``P_phone_idle`` the idle power of the
helper phones that stay on regardless. Everything the placement cannot change (attention, GPU layers,
the head) is a constant and drops out of the comparison. The objective is the expected energy per
decoded token over a measured mix of step row counts.

Every cost comes from :class:`LayerPlacementProfile`: measured rows (server ``S41SERVERFFNSHAPE``
summaries, phone meters, desktop timings) first, then scaled measurements (same device and format,
other model: compute scales with weight bytes; other row count: the device's measured row factor),
then priors -- each estimate carries its provenance, and a plan lists every prior it used.

The solver is exact (dynamic programming over classes of interchangeable layers, with dominance
pruning) whenever the state space is small, which covers every real instance seen so far; otherwise
greedy by marginal energy saving per byte with a local-improvement pass and an a-posteriori gap bound
(a relaxation that drops the one-device-per-layer coupling). Pure functions, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import itertools
import json
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence


LAYER_PLACEMENT_SCHEMA = "research-scheduler-layer-placement-v1"
LAYER_PLACEMENT_PROFILE_SCHEMA = "research-scheduler-layer-placement-profile-v1"
DESKTOP_CPU = "desktop-cpu"
PROVENANCES = ("measured", "derived", "scaled", "prior")
RESIDENCIES = ("co-resident", "per-model")
_MAX_LAYERS = 64

# ggml block geometry (elements per block, bytes per block) of the weight types FFN shards use.
GGML_BLOCKS = MappingProxyType({
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q4_0": (32, 18), "Q4_1": (32, 20), "Q8_0": (32, 34),
    "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "IQ4_NL": (32, 18),
})


class LayerPlacementError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LayerPlacementError(message)


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
             and value >= minimum, name + " must be a finite number >= " + str(minimum))
    return float(value)


def _int(value: object, name: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, name + " must be an integer >= " + str(minimum))
    return value


def layer_mask(layers: Iterable[int]) -> int:
    mask = 0
    for layer in layers:
        _require(type(layer) is int and 0 <= layer < _MAX_LAYERS, "layer index out of 0..63")
        mask |= 1 << layer
    return mask


def mask_layers(mask: int) -> tuple[int, ...]:
    return tuple(index for index in range(_MAX_LAYERS) if mask >> index & 1)


def layer_spec(layers: Iterable[int]) -> str:
    """``18-24`` / ``0-5,8`` spelling (empty string for no layers)."""
    ordered = sorted(set(layers))
    spans: list[list[int]] = []
    for layer in ordered:
        if spans and spans[-1][1] == layer - 1:
            spans[-1][1] = layer
        else:
            spans.append([layer, layer])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def tensor_bytes(weight_type: str, rows: int, columns: int) -> int:
    """Bytes of a (rows x columns) ggml matrix whose rows are ``columns`` long."""
    _require(weight_type in GGML_BLOCKS, "unknown ggml weight type " + str(weight_type))
    block, size = GGML_BLOCKS[weight_type]
    _require(columns % block == 0, f"{columns} columns do not align to {weight_type} blocks of {block}")
    return rows * columns // block * size


def ffn_layer_bytes(n_embd: int, n_ff: int, gate: str, up: str | None = None, down: str | None = None) -> int:
    """Bytes of one layer's gate/up (n_ff rows of n_embd) and down (n_embd rows of n_ff) matrices."""
    up = gate if up is None else up
    down = gate if down is None else down
    return (tensor_bytes(gate, n_ff, n_embd) + tensor_bytes(up, n_ff, n_embd)
            + tensor_bytes(down, n_embd, n_ff))


# --------------------------------------------------------------------------------------------------
# estimates and the measured profile store
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """One cost number and where it came from (``measured`` > ``derived`` > ``scaled`` > ``prior``)."""

    value: float
    provenance: str
    evidence: tuple[str, ...] = ()
    samples: float = 0.0      # calls behind a measured value (0 = not counted)

    def __post_init__(self) -> None:
        _require(self.provenance in PROVENANCES, "unknown provenance " + str(self.provenance))
        _number(self.value, "estimate value")


def _weakest(*provenances: str) -> str:
    return max(provenances, key=PROVENANCES.index)


@dataclass(frozen=True)
class DesktopPower:
    """Host power (CPU package + GPU board) per decode state, watts."""

    ffn_w: float
    wait_w: float
    other_w: float = 0.0
    provenance: str = "prior"
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("ffn_w", "wait_w", "other_w"):
            _number(getattr(self, name), "desktop power " + name)
        _require(self.provenance in PROVENANCES, "unknown desktop power provenance")

    def to_json(self) -> dict[str, object]:
        return {"ffn_w": self.ffn_w, "wait_w": self.wait_w, "other_w": self.other_w,
                "provenance": self.provenance, "evidence": list(self.evidence)}

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "DesktopPower":
        return cls(_number(value["ffn_w"], "ffn_w"), _number(value["wait_w"], "wait_w"),
                   _number(value.get("other_w", 0.0), "other_w"), str(value.get("provenance", "prior")),
                   tuple(str(item) for item in value.get("evidence", ())))


@dataclass(frozen=True)
class PowerOperatingPoint:
    """One measured decode operating point: seconds per step in each state and host energy per step."""

    ffn_s: float
    wait_s: float
    other_s: float
    energy_j: float
    label: str = ""


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (small dense systems)."""
    size = len(vector)
    rows = [list(row) + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda index: abs(rows[index][column]))
        _require(abs(rows[pivot][column]) > 1e-12, "desktop power fit is singular")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for index in range(size):
            if index != column:
                factor = rows[index][column] / rows[column][column]
                rows[index] = [a - factor * b for a, b in zip(rows[index], rows[column])]
    return [rows[index][size] / rows[index][index] for index in range(size)]


def fit_desktop_power(points: Sequence[PowerOperatingPoint], *, evidence: Sequence[str] = ()) -> DesktopPower:
    """Least-squares host power per state from measured operating points (at least three).

    Each point says how long one step spends with the CPU streaming FFN weights, waiting on a helper,
    and doing everything else, and how much host energy the step took. The three powers are the
    least-squares solution of ``ffn_s * P_ffn + wait_s * P_wait + other_s * P_other = energy_j``.
    """
    _require(len(points) >= 3, "desktop power fit needs at least three operating points")
    columns = [[p.ffn_s, p.wait_s, p.other_s] for p in points]
    normal = [[sum(row[i] * row[j] for row in columns) for j in range(3)] for i in range(3)]
    right = [sum(row[i] * p.energy_j for row, p in zip(columns, points)) for i in range(3)]
    ffn_w, wait_w, other_w = _solve_linear(normal, right)
    _require(min(ffn_w, wait_w, other_w) >= 0, "desktop power fit gave a negative state power")
    return DesktopPower(ffn_w, wait_w, other_w, "derived", tuple(evidence) or tuple(p.label for p in points))


@dataclass(frozen=True)
class CallKey:
    device_id: str
    model_id: str
    shard_format: str
    rows: int
    transport: str


@dataclass(frozen=True)
class CallStats:
    """Call-weighted means of one (device, model, format, rows, transport) bucket."""

    calls: float
    rpc_ms: float
    compute_ms: float
    energy_j: float | None = None
    energy_calls: float = 0.0
    provenance: str = "measured"
    evidence: tuple[str, ...] = ()

    def merged(self, calls: float, rpc_ms: float, compute_ms: float, evidence: str, *, keep: float = 1.0) -> "CallStats":
        old = self.calls * keep
        total = old + calls
        return replace(self, calls=total, rpc_ms=(self.rpc_ms * old + rpc_ms * calls) / total,
                       compute_ms=(self.compute_ms * old + compute_ms * calls) / total,
                       energy_calls=self.energy_calls * keep, provenance="measured",
                       evidence=_bounded_evidence(self.evidence, evidence))


def _bounded_evidence(old: Sequence[str], new: str | None, limit: int = 16) -> tuple[str, ...]:
    rows = [row for row in old if row != new] + ([new] if new else [])
    return tuple(rows[-limit:])


@dataclass(frozen=True)
class DeviceCostPriors:
    """What a device costs when nothing about a (model, format) is measured on it."""

    bytes_per_s: Mapping[str, float]                    # shard format -> compute weight-stream rate
    overhead_ms: Mapping[str, Mapping[int, float]]      # transport -> rows -> non-compute ms per call
    rows_factor: Mapping[int, float]                    # compute(rows) / compute(1)
    active_marginal_w: float                            # power above idle while computing
    idle_w: float
    provenance: Mapping[str, str] = MappingProxyType({})  # what each prior is based on

    def overhead(self, transport: str, rows: int) -> float | None:
        table = self.overhead_ms.get(transport)
        if not table:
            return None
        if rows in table:
            return table[rows]
        nearest = min(table, key=lambda value: (abs(value - rows), value))
        return table[nearest]

    def factor(self, rows: int) -> float:
        return _table_factor(self.rows_factor, rows)

    def to_json(self) -> dict[str, object]:
        return {
            "active_marginal_w": self.active_marginal_w,
            "bytes_per_s": dict(sorted(self.bytes_per_s.items())),
            "idle_w": self.idle_w,
            "overhead_ms": {transport: {str(rows): value for rows, value in sorted(table.items())}
                            for transport, table in sorted(self.overhead_ms.items())},
            "provenance": dict(sorted(self.provenance.items())),
            "rows_factor": {str(rows): value for rows, value in sorted(self.rows_factor.items())},
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "DeviceCostPriors":
        return cls(
            bytes_per_s=MappingProxyType({str(k): _number(v, "bytes_per_s", minimum=1.0)
                                          for k, v in dict(value.get("bytes_per_s", {})).items()}),
            overhead_ms=MappingProxyType({
                str(transport): MappingProxyType({int(rows): _number(ms, "overhead_ms") for rows, ms in table.items()})
                for transport, table in dict(value.get("overhead_ms", {})).items()
            }),
            rows_factor=MappingProxyType({int(k): _number(v, "rows_factor", minimum=1e-6)
                                          for k, v in dict(value.get("rows_factor", {"1": 1.0})).items()}),
            active_marginal_w=_number(value.get("active_marginal_w", 0.0), "active_marginal_w"),
            idle_w=_number(value.get("idle_w", 0.0), "idle_w"),
            provenance=MappingProxyType({str(k): str(v) for k, v in dict(value.get("provenance", {})).items()}),
        )


@dataclass(frozen=True)
class CallCost:
    rpc_ms: Estimate
    compute_ms: Estimate
    energy_j: Estimate

    @property
    def provenance(self) -> str:
        return _weakest(self.rpc_ms.provenance, self.compute_ms.provenance, self.energy_j.provenance)


@dataclass
class LayerPlacementProfile:
    """Measured per-device costs plus the priors used when a measurement is missing.

    Mutable on purpose: the online path (`observe_*`) folds new server summaries and meter readings in;
    the planner reads a frozen :meth:`snapshot`. ``keep`` < 1 forgets older calls geometrically per
    observation batch (drift tracking); 1.0 keeps a plain call-weighted mean.
    """

    desktop_power: DesktopPower
    desktop_ffn_ms: dict[tuple[str, int], Estimate] = field(default_factory=dict)  # (model, rows) -> ms/layer
    desktop_bytes_per_s: float = 28e9
    desktop_rows_factor: Mapping[int, float] = field(default_factory=lambda: MappingProxyType({1: 1.0}))
    device_priors: dict[str, DeviceCostPriors] = field(default_factory=dict)
    calls: dict[CallKey, CallStats] = field(default_factory=dict)
    energy_per_call: dict[tuple[str, str, int], Estimate] = field(default_factory=dict)  # (device, model, rows)
    keep: float = 1.0
    revision: int = 0

    # -- online updates --------------------------------------------------------------------------
    def observe_call_summary(self, device_id: str, model_id: str, shard_format: str, transport: str, rows: int,
                             calls: int, rpc_ms: float, compute_ms: float, evidence: str = "") -> None:
        """Fold one server summary bucket (e.g. a parsed ``S41SERVERFFNSHAPE`` row) into the store."""
        _require(calls > 0 and rows >= 1, "call summary needs calls > 0 and rows >= 1")
        _number(rpc_ms, "rpc_ms")
        _number(compute_ms, "compute_ms")
        key = CallKey(device_id, model_id, shard_format, rows, transport)
        old = self.calls.get(key)
        if old is None:
            self.calls[key] = CallStats(float(calls), float(rpc_ms), float(compute_ms),
                                        evidence=(evidence,) if evidence else ())
        else:
            self.calls[key] = old.merged(float(calls), float(rpc_ms), float(compute_ms), evidence, keep=self.keep)
        self.revision += 1

    def observe_call_energy(self, device_id: str, model_id: str, rows: int, joules_above_idle: float,
                            calls: int, evidence: str = "") -> None:
        """Phone meter: energy above idle over a window divided over the calls in it."""
        _require(calls > 0, "call energy needs calls > 0")
        value = _number(joules_above_idle, "joules_above_idle") / calls
        self.energy_per_call[(device_id, model_id, rows)] = Estimate(value, "measured", (evidence,) if evidence else ())
        self.revision += 1

    def observe_desktop_layer(self, model_id: str, rows: int, ms: float, evidence: str = "",
                              provenance: str = "measured") -> None:
        self.desktop_ffn_ms[(model_id, rows)] = Estimate(_number(ms, "desktop layer ms"), provenance,
                                                          (evidence,) if evidence else ())
        self.revision += 1

    def observe_desktop_power(self, power: DesktopPower) -> None:
        self.desktop_power = power
        self.revision += 1

    # -- lookups ---------------------------------------------------------------------------------
    def desktop_layer_ms(self, model_id: str, rows: int, layer_bytes: int) -> Estimate:
        exact = self.desktop_ffn_ms.get((model_id, rows))
        if exact is not None:
            return exact
        base = self.desktop_ffn_ms.get((model_id, 1))
        factor = _table_factor(self.desktop_rows_factor, rows)
        if base is not None:
            return Estimate(base.value * factor, _weakest(base.provenance, "scaled"), base.evidence)
        return Estimate(layer_bytes / self.desktop_bytes_per_s * 1000.0 * factor, "prior",
                        ("desktop_bytes_per_s",))

    def call_cost(self, device_id: str, model_id: str, shard_format: str, transport: str, rows: int,
                  layer_bytes: int, model_bytes: Mapping[str, Mapping[str, int]] | None = None) -> CallCost:
        """rpc / compute ms and call energy of one full-width layer call.

        ``model_bytes`` (model -> format -> bytes per layer) lets a measurement of another model on the same
        device and format scale by weight bytes (the kernels stream weights; the transport cost stays).
        """
        priors = self.device_priors.get(device_id)
        exact = self.calls.get(CallKey(device_id, model_id, shard_format, rows, transport))
        if exact is not None:
            compute = Estimate(exact.compute_ms, exact.provenance, exact.evidence, exact.calls)
            rpc = Estimate(exact.rpc_ms, exact.provenance, exact.evidence, exact.calls)
        else:
            compute, overhead = self._scaled_compute(device_id, model_id, shard_format, transport, rows,
                                                     layer_bytes, model_bytes or {}, priors)
            rpc = Estimate(compute.value + overhead.value, _weakest(compute.provenance, overhead.provenance),
                           compute.evidence + overhead.evidence)
        energy = self._call_energy(device_id, model_id, rows, compute, priors)
        return CallCost(rpc, compute, energy)

    def _scaled_compute(self, device_id, model_id, shard_format, transport, rows, layer_bytes, model_bytes, priors):
        # 1. same device/model/format/rows on another transport: compute carries over, overhead changes
        same = [stats for key, stats in self.calls.items()
                if (key.device_id, key.model_id, key.shard_format, key.rows) == (device_id, model_id, shard_format, rows)]
        overhead = self._overhead(device_id, transport, rows, priors)
        if same:
            stats = max(same, key=lambda row: row.calls)
            return Estimate(stats.compute_ms, "scaled", stats.evidence), overhead
        # 2. same device/format, other model (and/or other rows): scale compute by bytes and row factor
        candidates = []
        for key, stats in self.calls.items():
            if key.device_id != device_id or key.shard_format != shard_format:
                continue
            other_bytes = model_bytes.get(key.model_id, {}).get(shard_format)
            if not other_bytes:
                continue
            factor = self._rows_factor(device_id, rows, priors) / self._rows_factor(device_id, key.rows, priors)
            candidates.append((key.rows != rows, -stats.calls, stats.compute_ms * layer_bytes / other_bytes * factor,
                               stats.evidence))
        if candidates:
            candidates.sort(key=lambda row: (row[0], row[1]))
            _, _, value, evidence = candidates[0]
            return Estimate(value, "scaled", evidence), overhead
        # 3. prior rate of the device for this format
        _require(priors is not None and shard_format in priors.bytes_per_s,
                 f"no measurement or prior for {device_id} format {shard_format}")
        value = layer_bytes / priors.bytes_per_s[shard_format] * 1000.0 * priors.factor(rows)
        return Estimate(value, "prior", ("prior:" + device_id + ":" + shard_format,)), overhead

    def _overhead(self, device_id, transport, rows, priors) -> Estimate:
        measured = [stats for key, stats in self.calls.items()
                    if key.device_id == device_id and key.transport == transport and key.rows == rows]
        if measured:
            total = sum(stats.calls for stats in measured)
            value = sum((stats.rpc_ms - stats.compute_ms) * stats.calls for stats in measured) / total
            return Estimate(max(0.0, value), "scaled", ("overhead:" + device_id + ":" + transport,))
        prior = None if priors is None else priors.overhead(transport, rows)
        _require(prior is not None, f"no transport overhead for {device_id} over {transport}")
        return Estimate(prior, "prior", ("prior:" + device_id + ":" + transport + ":overhead",))

    def _rows_factor(self, device_id, rows, priors) -> float:
        if rows == 1:
            return 1.0
        pairs = {}
        for key, stats in self.calls.items():
            if key.device_id == device_id and key.rows in (1, rows):
                pairs.setdefault((key.model_id, key.shard_format, key.transport), {})[key.rows] = stats
        ratios = [(row[1].calls, row[rows].compute_ms / row[1].compute_ms)
                  for row in pairs.values() if 1 in row and rows in row and row[1].compute_ms > 0]
        if ratios:
            total = sum(weight for weight, _ in ratios)
            return sum(weight * ratio for weight, ratio in ratios) / total
        return 1.0 if priors is None else priors.factor(rows)

    def _call_energy(self, device_id, model_id, rows, compute: Estimate, priors) -> Estimate:
        exact = self.energy_per_call.get((device_id, model_id, rows))
        if exact is not None:
            return exact
        measured = [(key, value) for key, value in self.energy_per_call.items() if key[0] == device_id]
        if measured:
            # energy per compute-ms of the same device (other model or rows), times this compute time
            rates = []
            for (device, model, other_rows), value in measured:
                cost = [stats for key, stats in self.calls.items()
                        if (key.device_id, key.model_id, key.rows) == (device, model, other_rows)]
                if cost:
                    stats = max(cost, key=lambda row: row.calls)
                    if stats.compute_ms > 0:
                        rates.append(value.value / stats.compute_ms)
            if rates:
                return Estimate(sum(rates) / len(rates) * compute.value, _weakest("scaled", compute.provenance),
                                ("energy-rate:" + device_id,))
        _require(priors is not None, f"no energy measurement or prior for {device_id}")
        return Estimate(priors.active_marginal_w * compute.value / 1000.0, "prior",
                        ("prior:" + device_id + ":active_marginal_w",))

    def idle_w(self, device_id: str) -> float:
        priors = self.device_priors.get(device_id)
        return 0.0 if priors is None else priors.idle_w

    # -- persistence -----------------------------------------------------------------------------
    def to_json(self) -> dict[str, object]:
        return {
            "schema": LAYER_PLACEMENT_PROFILE_SCHEMA,
            "revision": self.revision,
            "keep": self.keep,
            "desktop": {
                "power": self.desktop_power.to_json(),
                "bytes_per_s": self.desktop_bytes_per_s,
                "rows_factor": {str(k): v for k, v in sorted(self.desktop_rows_factor.items())},
                "ffn_layer_ms": [
                    {"model_id": model, "rows": rows, "ms": value.value, "provenance": value.provenance,
                     "evidence": list(value.evidence)}
                    for (model, rows), value in sorted(self.desktop_ffn_ms.items())
                ],
            },
            "devices": {device: priors.to_json() for device, priors in sorted(self.device_priors.items())},
            "calls": [
                {"device_id": key.device_id, "model_id": key.model_id, "format": key.shard_format,
                 "rows": key.rows, "transport": key.transport, "calls": stats.calls, "rpc_ms": stats.rpc_ms,
                 "compute_ms": stats.compute_ms, "provenance": stats.provenance, "evidence": list(stats.evidence)}
                for key, stats in sorted(self.calls.items(), key=lambda item: (
                    item[0].device_id, item[0].model_id, item[0].shard_format, item[0].rows, item[0].transport))
            ],
            "call_energy": [
                {"device_id": device, "model_id": model, "rows": rows, "energy_j": value.value,
                 "provenance": value.provenance, "evidence": list(value.evidence)}
                for (device, model, rows), value in sorted(self.energy_per_call.items())
            ],
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "LayerPlacementProfile":
        _require(value.get("schema") == LAYER_PLACEMENT_PROFILE_SCHEMA, "not a layer placement profile")
        desktop = dict(value["desktop"])
        profile = cls(
            desktop_power=DesktopPower.from_json(desktop["power"]),
            desktop_bytes_per_s=_number(desktop.get("bytes_per_s", 28e9), "desktop bytes_per_s", minimum=1.0),
            desktop_rows_factor=MappingProxyType({int(k): _number(v, "desktop rows_factor", minimum=1e-6)
                                                  for k, v in dict(desktop.get("rows_factor", {"1": 1.0})).items()}),
            device_priors={str(k): DeviceCostPriors.from_json(v) for k, v in dict(value.get("devices", {})).items()},
            keep=_number(value.get("keep", 1.0), "keep", minimum=0.0),
            revision=int(value.get("revision", 0)),
        )
        _require(profile.keep <= 1.0, "profile keep must be in [0, 1]")
        for row in desktop.get("ffn_layer_ms", ()):
            profile.desktop_ffn_ms[(str(row["model_id"]), int(row["rows"]))] = Estimate(
                _number(row["ms"], "desktop ms"), str(row.get("provenance", "measured")),
                tuple(str(item) for item in row.get("evidence", ())))
        for row in value.get("calls", ()):
            key = CallKey(str(row["device_id"]), str(row["model_id"]), str(row["format"]), int(row["rows"]),
                          str(row["transport"]))
            profile.calls[key] = CallStats(
                _number(row["calls"], "calls", minimum=1e-9), _number(row["rpc_ms"], "rpc_ms"),
                _number(row["compute_ms"], "compute_ms"), provenance=str(row.get("provenance", "measured")),
                evidence=tuple(str(item) for item in row.get("evidence", ())))
        for row in value.get("call_energy", ()):
            profile.energy_per_call[(str(row["device_id"]), str(row["model_id"]), int(row["rows"]))] = Estimate(
                _number(row["energy_j"], "energy_j"), str(row.get("provenance", "measured")),
                tuple(str(item) for item in row.get("evidence", ())))
        return profile

    def copy(self) -> "LayerPlacementProfile":
        return LayerPlacementProfile.from_json(json.loads(json.dumps(self.to_json())))


def _table_factor(table: Mapping[int, float], rows: int) -> float:
    """Row factor at ``rows``: linear interpolation, linear extrapolation above the largest known row."""
    if rows in table:
        return table[rows]
    known = sorted(table)
    if not known:
        return 1.0
    if rows < known[0]:
        return table[known[0]]
    if rows > known[-1]:
        if len(known) == 1:
            return table[known[-1]] * rows / known[-1]
        a, b = known[-2], known[-1]
        slope = (table[b] - table[a]) / (b - a)
        return max(table[b], table[b] + slope * (rows - b))
    lower = max(value for value in known if value <= rows)
    upper = min(value for value in known if value >= rows)
    weight = (rows - lower) / (upper - lower)
    return table[lower] * (1 - weight) + table[upper] * weight


# --------------------------------------------------------------------------------------------------
# the placement problem
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelLayers:
    """A model's CPU-resident layers and its FFN bytes per layer in each shard format."""

    model_id: str
    cpu_layers: tuple[int, ...]
    layer_bytes: Mapping[str, int]                       # shard format -> bytes per layer
    layer_bytes_overrides: Mapping[str, Mapping[int, int]] = MappingProxyType({})  # format -> layer -> bytes
    n_embd: int = 0
    n_ff: int = 0
    column_quantum: int = 0          # server column quantum (0 = no column split candidates)
    other_step_ms: Mapping[int, float] = MappingProxyType({})  # rows -> placement-invariant step ms
    host_format: str = "f16"         # what the desktop CPU streams for this model

    def __post_init__(self) -> None:
        _require(bool(self.model_id), "model id is empty")
        _require(len(set(self.cpu_layers)) == len(self.cpu_layers) and all(
            type(layer) is int and 0 <= layer < _MAX_LAYERS for layer in self.cpu_layers),
            self.model_id + " CPU layers must be distinct indices 0..63")
        _require(bool(self.layer_bytes) and all(type(v) is int and v > 0 for v in self.layer_bytes.values()),
                 self.model_id + " needs positive bytes per layer for each format")
        _require(self.host_format in self.layer_bytes, self.model_id + " has no bytes for its host format")

    def bytes_for(self, shard_format: str, layer: int) -> int:
        override = self.layer_bytes_overrides.get(shard_format, {})
        if layer in override:
            return override[layer]
        _require(shard_format in self.layer_bytes, f"{self.model_id} has no {shard_format} layer size")
        return self.layer_bytes[shard_format]


@dataclass(frozen=True)
class HelperDevice:
    """A helper that can own CPU-resident FFN layers.

    ``formats`` maps a model to the shard format the device executes for it (absent model = the device
    cannot serve it). ``residency`` ``per-model``: the device re-provisions its shards when the desktop
    switches models (capacity applies to each model separately, e.g. the OP15's HTP sessions);
    ``co-resident``: shards of every model stay resident together (capacity is shared, e.g. the Pixel).
    ``allowed_layers`` (model -> mask) restricts which layers may be placed there at all (e.g. only
    stored and qualified shards for an executable plan); absent = any CPU layer.
    """

    device_id: str
    formats: Mapping[str, str]
    capacity_bytes: int
    transport: str
    sessions: int = 1
    session_limit_bytes: int = 0
    residency: str = "co-resident"
    available: bool = True
    unavailable_reason: str | None = None
    max_busy_ms_per_token: float | None = None
    allowed_layers: Mapping[str, int] | None = None
    busy_envelope_ms: float | None = None   # busy per step the device has sustained in measured operation
    stored_layers: Mapping[str, int] = MappingProxyType({})
    qualified_layers: Mapping[str, int] = MappingProxyType({})

    def __post_init__(self) -> None:
        _require(bool(self.device_id) and self.device_id != DESKTOP_CPU, "helper device id is invalid")
        _require(self.residency in RESIDENCIES, "unknown residency " + str(self.residency))
        _require(type(self.capacity_bytes) is int and self.capacity_bytes >= 0, "capacity must be bytes")
        _require(type(self.sessions) is int and self.sessions >= 1, "sessions must be >= 1")
        _require(type(self.session_limit_bytes) is int and self.session_limit_bytes >= 0,
                 "session limit must be bytes")
        _require(self.max_busy_ms_per_token is None or self.max_busy_ms_per_token >= 0, "busy cap must be >= 0")

    def serves(self, model_id: str) -> bool:
        return self.available and model_id in self.formats

    def may_hold(self, model_id: str, layer: int) -> bool:
        if not self.serves(model_id):
            return False
        if self.allowed_layers is None:
            return True
        return bool(self.allowed_layers.get(model_id, 0) >> layer & 1)


@dataclass(frozen=True)
class PlacementProblem:
    models: tuple[ModelLayers, ...]
    devices: tuple[HelperDevice, ...]
    profile: LayerPlacementProfile
    rows_mix: Mapping[str, Mapping[int, float]] = MappingProxyType({})   # model -> rows -> share of steps
    latency_ppm: int | None = None
    column_fractions: Mapping[str, tuple[float, ...]] = MappingProxyType({})  # model -> candidate fractions
    current: Mapping[str, Mapping[int, str]] = MappingProxyType({})  # model -> layer -> owner device id
    # risk premium on an option's energy by the weakest provenance behind it (ppm); a measured bucket with
    # fewer than ``confident_calls`` calls counts as ``scaled``. Empty = plain expected value.
    uncertainty_ppm: Mapping[str, int] = MappingProxyType({})
    confident_calls: int = 0

    def __post_init__(self) -> None:
        ids = [device.device_id for device in self.devices]
        _require(len(set(ids)) == len(ids), "duplicate helper device ids")
        names = [model.model_id for model in self.models]
        _require(len(set(names)) == len(names), "duplicate model ids")
        _require(self.latency_ppm is None or (type(self.latency_ppm) is int and self.latency_ppm >= 1_000_000),
                 "latency_ppm must be an integer >= 1000000")
        _require(set(self.uncertainty_ppm) <= set(PROVENANCES) and all(
            type(value) is int and value >= 0 for value in self.uncertainty_ppm.values()),
            "uncertainty_ppm maps provenances to integer ppm >= 0")
        _require(type(self.confident_calls) is int and self.confident_calls >= 0, "confident_calls must be >= 0")

    def penalty(self, estimates: Iterable[Estimate]) -> float:
        """Risk factor (>= 1) of an option priced from ``estimates``."""
        if not self.uncertainty_ppm:
            return 1.0
        worst = 0
        for estimate in estimates:
            provenance = estimate.provenance
            if provenance == "measured" and 0 < estimate.samples < self.confident_calls:
                provenance = "scaled"
            worst = max(worst, self.uncertainty_ppm.get(provenance, 0))
        return 1.0 + worst / 1e6

    def mix(self, model_id: str) -> Mapping[int, float]:
        table = self.rows_mix.get(model_id) or {1: 1.0}
        total = sum(table.values())
        _require(total > 0 and all(rows >= 1 and share >= 0 for rows, share in table.items()),
                 model_id + " rows mix is invalid")
        return {rows: share / total for rows, share in sorted(table.items()) if share > 0}

    def device(self, device_id: str) -> HelperDevice:
        for device in self.devices:
            if device.device_id == device_id:
                return device
        raise LayerPlacementError("unknown helper device " + device_id)


# --------------------------------------------------------------------------------------------------
# per-layer option costs
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionCost:
    """One layer on one owner, averaged over the model's rows mix.

    ``energy_j`` is the objective per step (expected energy times the problem's risk factor),
    ``expected_j`` the plain expected energy, ``ms_by_rows`` the chain time it adds per step at each row
    count, ``busy_ms`` the owner's busy time per step at the lowest row count (thermal envelope).
    """

    owner: str
    energy_j: float
    ms_by_rows: Mapping[int, float]
    busy_ms: float
    bytes: int
    provenance: str
    priors: tuple[str, ...]
    expected_j: float = 0.0


def _option_costs(problem: PlacementProblem, model: ModelLayers, layer: int, fraction: float) -> tuple[OptionCost, ...]:
    profile = problem.profile
    power = profile.desktop_power
    mix = problem.mix(model.model_id)
    idle_w = sum(profile.idle_w(device.device_id) for device in problem.devices if device.available)
    host_bytes = model.bytes_for(model.host_format, layer)
    model_bytes = {row.model_id: dict(row.layer_bytes) for row in problem.models}
    cpu_ms = {rows: profile.desktop_layer_ms(model.model_id, rows, host_bytes) for rows in mix}
    cpu_energy = sum(share * (power.ffn_w + idle_w) * cpu_ms[rows].value / 1000.0 for rows, share in mix.items())
    cpu_provenance = _weakest(*(row.provenance for row in cpu_ms.values()))
    options = [OptionCost(
        DESKTOP_CPU, cpu_energy * problem.penalty(cpu_ms.values()), {rows: cpu_ms[rows].value for rows in mix}, 0.0,
        0, cpu_provenance,
        tuple(sorted({f"{DESKTOP_CPU}:{model.model_id}:ffn_ms"} if cpu_provenance == "prior" else set())),
        cpu_energy,
    )]
    for device in problem.devices:
        if not device.may_hold(model.model_id, layer):
            continue
        shard_format = device.formats[model.model_id]
        size = model.bytes_for(shard_format, layer)
        energy = 0.0
        ms_by_rows = {}
        busy_by_rows = {}
        provenance = "measured"
        priors = set()
        estimates = []
        for rows, share in mix.items():
            cost = profile.call_cost(device.device_id, model.model_id, shard_format, device.transport, rows, size,
                                     model_bytes)
            compute = cost.compute_ms.value * fraction
            overhead = max(0.0, cost.rpc_ms.value - cost.compute_ms.value)
            helper_ms = overhead + compute
            host_share_ms = cpu_ms[rows].value * (1.0 - fraction)
            step_energy = ((power.ffn_w + idle_w) * host_share_ms
                           + (power.wait_w + idle_w) * max(0.0, helper_ms - host_share_ms)) / 1000.0
            step_energy += cost.energy_j.value * fraction
            energy += share * step_energy
            ms_by_rows[rows] = max(helper_ms, host_share_ms)
            busy_by_rows[rows] = helper_ms
            provenance = _weakest(provenance, cost.provenance, cpu_ms[rows].provenance if fraction < 1 else "measured")
            estimates.extend((cost.rpc_ms, cost.compute_ms, cost.energy_j))
            if fraction < 1:
                estimates.append(cpu_ms[rows])
            for name, estimate in (("rpc_ms", cost.rpc_ms), ("compute_ms", cost.compute_ms),
                                   ("energy_j", cost.energy_j)):
                if estimate.provenance == "prior":
                    priors.add(f"{device.device_id}:{model.model_id}:{shard_format}:rows{rows}:{name}")
        busy = busy_by_rows[min(busy_by_rows)]
        options.append(OptionCost(device.device_id, energy * problem.penalty(estimates), ms_by_rows, busy, size,
                                  provenance, tuple(sorted(priors)), energy))
    return tuple(options)


# --------------------------------------------------------------------------------------------------
# solvers
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Item:
    model_id: str
    layer: int
    options: tuple[OptionCost, ...]

    @property
    def cpu(self) -> OptionCost:
        return self.options[0]


@dataclass(frozen=True)
class _Constraints:
    """Resource dimensions every solver checks."""

    capacity: Mapping[tuple[str, str], int]          # (device, scope) -> bytes; scope = model or "*"
    session_slots: Mapping[tuple[str, str], tuple[int, int]]  # (device, scope) -> (sessions, limit)
    busy_cap: Mapping[str, float]                    # device -> ms per step (one model's step)
    latency_budget: Mapping[tuple[str, int], float]  # (model, rows) -> max FFN-chain ms


def _scope(device: HelperDevice, model_id: str) -> str:
    return model_id if device.residency == "per-model" else "*"


def _constraints(problem: PlacementProblem, items: Sequence[_Item]) -> _Constraints:
    capacity = {}
    slots = {}
    for device in problem.devices:
        for model in problem.models:
            if device.serves(model.model_id):
                key = (device.device_id, _scope(device, model.model_id))
                capacity[key] = device.capacity_bytes
                if device.session_limit_bytes:
                    slots[key] = (device.sessions, device.session_limit_bytes)
    busy = {device.device_id: device.max_busy_ms_per_token for device in problem.devices
            if device.max_busy_ms_per_token is not None}
    latency = {}
    if problem.latency_ppm is not None:
        factor = problem.latency_ppm / 1_000_000
        for model in problem.models:
            for rows in problem.mix(model.model_id):
                cpu_chain = sum(item.cpu.ms_by_rows[rows] for item in items if item.model_id == model.model_id)
                other = model.other_step_ms.get(rows, 0.0)
                latency[(model.model_id, rows)] = factor * (cpu_chain + other) - other
    return _Constraints(MappingProxyType(capacity), MappingProxyType(slots), MappingProxyType(busy),
                        MappingProxyType(latency))


def _sessions_fit(sizes: Sequence[int], sessions: int, limit: int) -> bool:
    """First-fit decreasing into ``sessions`` bins of ``limit`` bytes (exact for equal sizes)."""
    bins = [0] * sessions
    for size in sorted(sizes, reverse=True):
        for index in range(sessions):
            if bins[index] + size <= limit:
                bins[index] += size
                break
        else:
            return False
    return True


def _usage_ok(sizes_by_scope: Mapping[tuple[str, str], Sequence[int]], limits: _Constraints) -> bool:
    for scope, sizes in sizes_by_scope.items():
        if sum(sizes) > limits.capacity.get(scope, 0):
            return False
        if scope in limits.session_slots and not _sessions_fit(sizes, *limits.session_slots[scope]):
            return False
    return True


def _feasible(problem: PlacementProblem, items: Sequence[_Item], choice: Sequence[int], limits: _Constraints) -> bool:
    used: dict[tuple[str, str], list[int]] = {}
    busy: dict[tuple[str, str], float] = {}
    chain: dict[tuple[str, int], float] = {}
    for item, index in zip(items, choice):
        option = item.options[index]
        for rows, ms in option.ms_by_rows.items():
            chain[(item.model_id, rows)] = chain.get((item.model_id, rows), 0.0) + ms
        if option.owner == DESKTOP_CPU:
            continue
        device = problem.device(option.owner)
        used.setdefault((option.owner, _scope(device, item.model_id)), []).append(option.bytes)
        busy[(option.owner, item.model_id)] = busy.get((option.owner, item.model_id), 0.0) + option.busy_ms
    if not _usage_ok(used, limits):
        return False
    if any(value > limits.busy_cap[device] + 1e-9 for (device, _), value in busy.items() if device in limits.busy_cap):
        return False
    return all(chain.get(key, 0.0) <= budget + 1e-9 for key, budget in limits.latency_budget.items())


def _classes(items: Sequence[_Item]) -> list[list[int]]:
    """Indices of interchangeable items (same model, same option costs and sizes)."""
    groups: dict[tuple, list[int]] = {}
    for index, item in enumerate(items):
        signature = (item.model_id, tuple(
            (option.owner, round(option.energy_j, 12),
             tuple(sorted((k, round(v, 9)) for k, v in option.ms_by_rows.items())),
             round(option.busy_ms, 9), option.bytes) for option in item.options))
        groups.setdefault(signature, []).append(index)
    return list(groups.values())


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def _advance(problem, key, item: _Item, split) -> tuple:
    sizes, busy, chain = (dict(key[0]), dict(key[1]), dict(key[2]))
    for n, option in zip(split, item.options):
        if n == 0:
            continue
        for rows, ms in option.ms_by_rows.items():
            chain[rows] = round(chain.get(rows, 0.0) + n * ms, 9)
        if option.owner == DESKTOP_CPU:
            continue
        device = problem.device(option.owner)
        scope = (option.owner, _scope(device, item.model_id))
        sizes[scope] = tuple(sorted(sizes.get(scope, ()) + (option.bytes,) * n))
        busy[option.owner] = round(busy.get(option.owner, 0.0) + n * option.busy_ms, 9)
    return (tuple(sorted(sizes.items())), tuple(sorted(busy.items())), tuple(sorted(chain.items())))


def _state_ok(key, limits: _Constraints) -> bool:
    if not _usage_ok(dict(key[0]), limits):
        return False
    return all(busy <= limits.busy_cap[device] + 1e-9 for device, busy in key[1] if device in limits.busy_cap)


def _shared_usage(key) -> tuple:
    """The part of a model's usage other models compete for: co-resident ("*") scopes."""
    return tuple((scope, sizes) for scope, sizes in key[0] if scope[1] == "*")


def _dominates(a: tuple, b: tuple, limits: _Constraints) -> bool:
    """Usage ``a`` leaves at least as much room as ``b`` in every shared scope (byte totals; scopes with a
    session limit only compare equal size multisets, since packing is not monotone in totals)."""
    left, right = dict(a), dict(b)
    for scope in set(left) | set(right):
        x, y = left.get(scope, ()), right.get(scope, ())
        if scope in limits.session_slots:
            if tuple(sorted(x)) != tuple(sorted(y)) and sum(x) > 0:
                return False
        elif sum(x) > sum(y):
            return False
    return True


def _pareto(front: dict, limits: _Constraints, *, limit: int = 4000) -> dict:
    """Drop dominated usages (quadratic, so only for fronts up to ``limit`` entries)."""
    if len(front) > limit:
        return front
    ordered = sorted(front.items(), key=lambda row: row[1][0])
    kept: list = []
    for usage, value in ordered:
        if any(_dominates(other, usage, limits) for other, _ in kept):
            continue
        kept.append((usage, value))
    return dict(kept)


def _solve_exact(problem: PlacementProblem, items: Sequence[_Item], limits: _Constraints,
                 max_states: int) -> list[int] | None:
    """Exact minimum: DP per model over classes of interchangeable layers, then a DP over models.

    Within a model every constraint is a sum over its layers bounded above (capacity per scope, session
    packing on the size multiset, busy per device, chain per row count) and the objective is additive, so
    partial assignments with the same usage vector are interchangeable and only the cheapest is kept.
    Models interact only through co-resident ("*") capacity scopes: each model contributes a Pareto front
    of (shared usage -> min energy) and the fronts are combined with the same state rule. Returns None
    when any state set exceeds ``max_states``.
    """
    order = []
    for item in items:
        if item.model_id not in order:
            order.append(item.model_id)
    fronts = []
    for model_id in order:
        members_all = [index for index, item in enumerate(items) if item.model_id == model_id]
        sub = [items[index] for index in members_all]
        classes = _classes(sub)
        states: dict[tuple, tuple[float, tuple]] = {((), (), ()): (0.0, ())}
        for members in classes:
            item = sub[members[0]]
            next_states: dict[tuple, tuple[float, tuple]] = {}
            for split in _compositions(len(members), len(item.options)):
                delta = sum(n * option.energy_j for n, option in zip(split, item.options))
                for key, (energy, picks) in states.items():
                    new_key = _advance(problem, key, item, split)
                    if not _state_ok(new_key, limits):
                        continue
                    value = energy + delta
                    old = next_states.get(new_key)
                    if old is None or value < old[0] - 1e-12:
                        next_states[new_key] = (value, picks + (split,))
                    if len(next_states) > max_states:
                        return None
            states = next_states
            if not states:
                return None
        front: dict[tuple, tuple[float, tuple]] = {}
        for key, (energy, picks) in states.items():
            if any(chain > limits.latency_budget.get((model_id, rows), math.inf) + 1e-9 for rows, chain in key[2]):
                continue
            usage = _shared_usage(key)
            old = front.get(usage)
            if old is None or energy < old[0] - 1e-12:
                front[usage] = (energy, picks)
        if not front:
            return None
        if len(order) == 1:
            # a single model: nothing to combine, the cheapest feasible final state is the optimum
            usage, best = min(front.items(), key=lambda row: row[1][0])
            front = {usage: best}
        fronts.append((members_all, sub, classes, _pareto(front, limits)))
    combined: dict[tuple, tuple[float, tuple]] = {(): (0.0, ())}
    for position, (_, _, _, front) in enumerate(fronts):
        next_combined: dict[tuple, tuple[float, tuple]] = {}
        for usage, (energy, chosen) in combined.items():
            for model_usage, (model_energy, picks) in front.items():
                merged = dict(usage)
                for scope, sizes in model_usage:
                    merged[scope] = tuple(sorted(merged.get(scope, ()) + sizes))
                if not _usage_ok(merged, limits):
                    continue
                key = tuple(sorted(merged.items()))
                value = energy + model_energy
                old = next_combined.get(key)
                if old is None or value < old[0] - 1e-12:
                    next_combined[key] = (value, chosen + (picks,))
                if len(next_combined) > max_states:
                    return None
        combined = _pareto(next_combined, limits)
        if not combined:
            return None
    _, chosen = min(combined.values(), key=lambda row: row[0])
    choice = [0] * len(items)
    for (members_all, sub, classes, _), picks in zip(fronts, chosen):
        for members, split in zip(classes, picks):
            cursor = 0
            for option_index, n in enumerate(split):
                for member in members[cursor:cursor + n]:
                    choice[members_all[member]] = option_index
                cursor += n
    return choice


def _energy(items: Sequence[_Item], choice: Sequence[int]) -> float:
    return sum(item.options[index].energy_j for item, index in zip(items, choice))


def _solve_greedy(problem: PlacementProblem, items: Sequence[_Item], limits: _Constraints) -> list[int]:
    """Largest energy saving per stored byte first, then single moves and pairwise swaps until stable."""
    choice = [0] * len(items)
    candidates = []
    for item_index, item in enumerate(items):
        for option_index, option in enumerate(item.options[1:], start=1):
            saving = item.cpu.energy_j - option.energy_j
            if saving > 0:
                candidates.append((-saving / max(option.bytes, 1), -saving, option.owner, item.layer,
                                   item_index, option_index))
    candidates.sort()
    for *_, item_index, option_index in candidates:
        if choice[item_index] != 0:
            continue
        choice[item_index] = option_index
        if not _feasible(problem, items, choice, limits):
            choice[item_index] = 0
    for _ in range(50):
        improved = False
        for item_index, item in enumerate(items):
            here = item.options[choice[item_index]].energy_j
            for option_index, option in enumerate(item.options):
                if option.energy_j < here - 1e-12:
                    old = choice[item_index]
                    choice[item_index] = option_index
                    if _feasible(problem, items, choice, limits):
                        here, improved = option.energy_j, True
                    else:
                        choice[item_index] = old
        for first, second in itertools.combinations(range(len(items)), 2):
            a, b = items[first], items[second]
            owner_a, owner_b = a.options[choice[first]].owner, b.options[choice[second]].owner
            if owner_a == owner_b:
                continue
            index_a = next((i for i, o in enumerate(a.options) if o.owner == owner_b), None)
            index_b = next((i for i, o in enumerate(b.options) if o.owner == owner_a), None)
            if index_a is None or index_b is None:
                continue
            delta = (a.options[index_a].energy_j + b.options[index_b].energy_j
                     - a.options[choice[first]].energy_j - b.options[choice[second]].energy_j)
            if delta < -1e-12:
                old = (choice[first], choice[second])
                choice[first], choice[second] = index_a, index_b
                if _feasible(problem, items, choice, limits):
                    improved = True
                else:
                    choice[first], choice[second] = old
        if not improved:
            break
    return choice


def _saving_upper_bound(problem: PlacementProblem, items: Sequence[_Item], limits: _Constraints) -> float:
    """Upper bound on the energy saving (vs all-CPU) of any feasible placement.

    min of (a) every layer on its best device ignoring all limits and (b) a fractional knapsack per
    capacity scope over the savings of the layers it may hold (drops the one-owner-per-layer coupling,
    busy caps and latency, so it never cuts a feasible placement).
    """
    unconstrained = sum(max(0.0, max(item.cpu.energy_j - option.energy_j for option in item.options))
                        for item in items)
    per_scope: dict[tuple[str, str], list[tuple[float, int]]] = {}
    for item in items:
        for option in item.options[1:]:
            saving = item.cpu.energy_j - option.energy_j
            if saving <= 0:
                continue
            device = problem.device(option.owner)
            per_scope.setdefault((option.owner, _scope(device, item.model_id)), []).append((saving, option.bytes))
    knapsack = 0.0
    for scope, rows in per_scope.items():
        room = limits.capacity.get(scope, 0)
        if scope in limits.session_slots:
            sessions, limit = limits.session_slots[scope]
            room = min(room, sessions * limit)
        for saving, size in sorted(rows, key=lambda row: -row[0] / max(row[1], 1)):
            if room <= 0:
                break
            take = min(1.0, room / max(size, 1))
            knapsack += take * saving
            room -= take * size
    return min(unconstrained, knapsack)


# --------------------------------------------------------------------------------------------------
# layer index choice and the plan
# --------------------------------------------------------------------------------------------------


def _choose_layers(problem: PlacementProblem, items: Sequence[_Item], choice: Sequence[int]) -> list[int]:
    """Re-map counts to concrete layers: keep current owners, then stored shards, then contiguity.

    The solver only fixes how many interchangeable layers each owner gets; which ones is free. Prefer
    layers the owner already serves, then layers whose shards it already stores, then layers adjacent to
    what it keeps (sessions hold contiguous ranges), then low indices for devices that start low.
    """
    result = list(choice)
    for members in _classes(items):
        item = items[members[0]]
        counts = {index: 0 for index in range(len(item.options))}
        for member in members:
            counts[choice[member]] += 1
        remaining = sorted(members, key=lambda member: items[member].layer)
        current = problem.current.get(item.model_id, {})
        order = sorted(range(1, len(item.options)), key=lambda index: (
            min((layer for layer, owner in current.items() if owner == item.options[index].owner), default=_MAX_LAYERS),
            index))
        for option_index in order:
            need = counts[option_index]
            if not need:
                continue
            owner = item.options[option_index].owner
            device = problem.device(owner)
            kept = [items[m].layer for m in remaining if current.get(items[m].layer) == owner]
            anchor = (min(kept) + max(kept)) / 2 if kept else None

            def score(member: int) -> tuple:
                layer = items[member].layer
                stored = bool(device.stored_layers.get(item.model_id, 0) >> layer & 1)
                qualified = bool(device.qualified_layers.get(item.model_id, 0) >> layer & 1)
                distance = abs(layer - anchor) if anchor is not None else 0
                return (current.get(layer) != owner, not stored, not qualified, distance, layer)

            picked = sorted(remaining, key=score)[:need]
            for member in picked:
                result[member] = option_index
            remaining = [member for member in remaining if member not in picked]
        for member in remaining:
            result[member] = 0
    return result


def _stability(problem: PlacementProblem, item: _Item, owner: str) -> int:
    """0 = the layer stays with its current owner, 1 = it moves to a device that stores its shard, 2 = other."""
    if problem.current.get(item.model_id, {}).get(item.layer, DESKTOP_CPU) == owner:
        return 0
    if owner != DESKTOP_CPU and problem.device(owner).stored_layers.get(item.model_id, 0) >> item.layer & 1:
        return 1
    return 2


def _contiguity(items: Sequence[_Item], choice: Sequence[int]) -> int:
    """Number of owner changes along each model's layer order (fewer = contiguous ranges)."""
    breaks = 0
    previous: dict[str, str] = {}
    for item, index in sorted(zip(items, choice), key=lambda row: (row[0].model_id, row[0].layer)):
        owner = item.options[index].owner
        if item.model_id in previous and previous[item.model_id] != owner:
            breaks += 1
        previous[item.model_id] = owner
    return breaks


def _ordered_fill(problem: PlacementProblem, items: Sequence[_Item], choice: Sequence[int], *,
                  by_class: bool = True) -> list[int]:
    """Same owner counts (per class of interchangeable layers when ``by_class``), as contiguous ranges.

    Owners are ordered by the lowest layer they own now (owners without current layers last, the desktop
    CPU last among equals); walking each model's layers in order, a layer goes to the first owner that
    still needs one (of its class). With ``by_class`` the counts per (owner, class) -- hence energy and
    every limit -- are kept; without it only counts per owner are, and the caller re-checks both.
    """
    result = list(choice)
    class_of = {}
    for class_index, members in enumerate(_classes(items)):
        for member in members:
            class_of[member] = class_index if by_class else 0
    for model_id in sorted({item.model_id for item in items}):
        members = sorted((index for index, item in enumerate(items) if item.model_id == model_id),
                         key=lambda index: items[index].layer)
        quota: dict[tuple[str, int], int] = {}
        for index in members:
            key = (items[index].options[choice[index]].owner, class_of[index])
            quota[key] = quota.get(key, 0) + 1
        current = problem.current.get(model_id, {})
        owners = sorted({owner for owner, _ in quota}, key=lambda owner: (
            min((layer for layer, value in current.items() if value == owner), default=_MAX_LAYERS),
            owner == DESKTOP_CPU, owner))
        for index in members:
            item = items[index]
            for owner in owners:
                key = (owner, class_of[index])
                if quota.get(key, 0) > 0:
                    quota[key] -= 1
                    result[index] = next(i for i, option in enumerate(item.options) if option.owner == owner)
                    break
    return result


def _prefer_current(problem: PlacementProblem, items: Sequence[_Item], choice: list[int],
                    limits: _Constraints) -> list[int]:
    """Among equal-energy placements, keep current owners, then stored shards, then contiguous ranges.

    The DP fixes per-class counts; when two classes tie (e.g. layers that differ only in the packed size
    on one helper) it may split owners across classes arbitrarily. Pairwise owner swaps that do not raise
    the objective and keep every limit are applied while they lower (moves, contiguity breaks).
    """
    def score(current_choice):
        return (sum(_stability(problem, item, item.options[index].owner) for item, index in zip(items, current_choice)),
                _contiguity(items, current_choice))

    best = score(choice)
    energy = _energy(items, choice)
    for by_class in (False, True):
        filled = _ordered_fill(problem, items, choice, by_class=by_class)
        if (filled != choice and score(filled) < best and _energy(items, filled) <= energy + 1e-9
                and _feasible(problem, items, filled, limits)):
            choice, best = filled, score(filled)
    for _ in range(20):
        improved = False
        for first, second in itertools.combinations(range(len(items)), 2):
            a, b = items[first], items[second]
            if a.model_id != b.model_id:
                continue
            owner_a, owner_b = a.options[choice[first]].owner, b.options[choice[second]].owner
            if owner_a == owner_b:
                continue
            index_a = next((i for i, o in enumerate(a.options) if o.owner == owner_b), None)
            index_b = next((i for i, o in enumerate(b.options) if o.owner == owner_a), None)
            if index_a is None or index_b is None:
                continue
            delta = (a.options[index_a].energy_j + b.options[index_b].energy_j
                     - a.options[choice[first]].energy_j - b.options[choice[second]].energy_j)
            if delta > 1e-9:
                continue
            old = (choice[first], choice[second])
            choice[first], choice[second] = index_a, index_b
            candidate = score(choice)
            if candidate < best and _feasible(problem, items, choice, limits):
                best, improved = candidate, True
            else:
                choice[first], choice[second] = old
        if not improved:
            break
    return choice


@dataclass(frozen=True)
class ModelPlacement:
    model_id: str
    owners: Mapping[int, str]                   # CPU layer -> owner (DESKTOP_CPU or a helper)
    column_fraction: float
    energy_j_per_token: float
    cpu_only_energy_j_per_token: float
    current_energy_j_per_token: float | None
    chain_ms_by_rows: Mapping[int, float]
    cpu_only_chain_ms_by_rows: Mapping[int, float]
    tokens_per_step: float
    objective_j_per_token: float = 0.0       # expected energy with the problem's risk premiums

    def mask(self, owner: str) -> int:
        return layer_mask(layer for layer, device in self.owners.items() if device == owner)

    @property
    def helper_masks(self) -> dict[str, int]:
        owners = sorted({device for device in self.owners.values() if device != DESKTOP_CPU})
        return {device: self.mask(device) for device in owners}

    def to_json(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "owners": {device: {"layers": layer_spec(mask_layers(mask)), "layer_mask": f"{mask:016x}",
                                "count": bin(mask).count("1")}
                       for device, mask in sorted({**self.helper_masks, DESKTOP_CPU: self.mask(DESKTOP_CPU)}.items())},
            "column_fraction": self.column_fraction,
            "energy_j_per_token": round(self.energy_j_per_token, 6),
            "objective_j_per_token": round(self.objective_j_per_token, 6),
            "cpu_only_energy_j_per_token": round(self.cpu_only_energy_j_per_token, 6),
            "current_energy_j_per_token": (None if self.current_energy_j_per_token is None
                                           else round(self.current_energy_j_per_token, 6)),
            "saving_vs_cpu_j_per_token": round(self.cpu_only_energy_j_per_token - self.energy_j_per_token, 6),
            "saving_vs_current_j_per_token": (None if self.current_energy_j_per_token is None else
                                              round(self.current_energy_j_per_token - self.energy_j_per_token, 6)),
            "ffn_chain_ms_by_rows": {str(k): round(v, 4) for k, v in sorted(self.chain_ms_by_rows.items())},
            "cpu_only_ffn_chain_ms_by_rows": {str(k): round(v, 4) for k, v in sorted(self.cpu_only_chain_ms_by_rows.items())},
            "tokens_per_step": round(self.tokens_per_step, 6),
        }


@dataclass(frozen=True)
class PlacementPlan:
    """``gap_bound_j_per_step``: upper bound on (this plan - optimum) of the objective, joules per decode
    step summed over the models (0 for an exact solve)."""

    models: Mapping[str, ModelPlacement]
    solver: str
    gap_bound_j_per_step: float
    priors_used: tuple[str, ...]
    device_usage: Mapping[str, Mapping[str, int]]   # device -> scope -> bytes
    device_busy_ms: Mapping[str, Mapping[str, float]]   # device -> model -> ms per step at the lowest rows
    infeasible_current: tuple[str, ...] = ()

    def owner(self, model_id: str, layer: int) -> str:
        return self.models[model_id].owners.get(layer, DESKTOP_CPU)

    def helper_masks(self, model_id: str) -> dict[str, int]:
        return self.models[model_id].helper_masks

    def to_json(self) -> dict[str, object]:
        return {
            "schema": LAYER_PLACEMENT_SCHEMA,
            "solver": self.solver,
            "gap_bound_j_per_step": round(self.gap_bound_j_per_step, 6),
            "priors_used": list(self.priors_used),
            "device_usage_bytes": {device: dict(sorted(scopes.items())) for device, scopes in sorted(self.device_usage.items())},
            "device_busy_ms_per_step": {device: {model: round(ms, 4) for model, ms in sorted(rows.items())}
                                        for device, rows in sorted(self.device_busy_ms.items())},
            "infeasible_current": list(self.infeasible_current),
            "models": {name: placement.to_json() for name, placement in sorted(self.models.items())},
        }


def _items_for(problem: PlacementProblem, model: ModelLayers, fraction: float) -> list[_Item]:
    return [_Item(model.model_id, layer, _option_costs(problem, model, layer, fraction))
            for layer in sorted(model.cpu_layers)]


def _tokens_per_step(problem: PlacementProblem, model_id: str) -> float:
    return sum(rows * share for rows, share in problem.mix(model_id).items())


def evaluate_owners(problem: PlacementProblem, owners: Mapping[str, Mapping[int, str]],
                    fractions: Mapping[str, float] | None = None, *, objective: bool = False) -> dict[str, float]:
    """Predicted energy per token of a given placement (model -> layer -> owner), all models.

    ``objective`` prices with the problem's risk premiums (what the solver minimizes) instead of the
    plain expected energy."""
    result = {}
    for model in problem.models:
        fraction = (fractions or {}).get(model.model_id, 1.0)
        items = _items_for(_unrestricted(problem), model, fraction)
        total = 0.0
        for item in items:
            owner = owners.get(model.model_id, {}).get(item.layer, DESKTOP_CPU)
            option = next((option for option in item.options if option.owner == owner), None)
            _require(option is not None, f"{model.model_id} layer {item.layer} cannot run on {owner}")
            total += option.energy_j if objective else option.expected_j
        result[model.model_id] = total / _tokens_per_step(problem, model.model_id)
    return result


def _unrestricted(problem: PlacementProblem) -> PlacementProblem:
    """Same problem without allowed-layer restrictions or availability (to price any given placement)."""
    return replace(problem, devices=tuple(replace(device, allowed_layers=None, available=True)
                                          for device in problem.devices))


def _fractions(problem: PlacementProblem, model: ModelLayers) -> tuple[float, ...]:
    candidates = problem.column_fractions.get(model.model_id) or (1.0,)
    for value in candidates:
        _require(0.0 < value <= 1.0, model.model_id + " column fractions must be in (0, 1]")
    return tuple(sorted(set(candidates), reverse=True))


def solve_placement(problem: PlacementProblem, *, method: str = "auto", max_states: int = 200_000) -> PlacementPlan:
    """Minimum predicted energy per token placement of every model's CPU layers.

    ``method``: ``exact`` (fails if the DP state space is too large), ``greedy``, or ``auto`` (exact when
    it fits in ``max_states``, else greedy). Column fractions (one per model server) are searched
    jointly: models that share a co-resident device's capacity are solved together.
    """
    _require(method in ("auto", "exact", "greedy"), "unknown placement method " + method)
    fraction_options = [_fractions(problem, model) for model in problem.models]
    best = None
    for fractions in itertools.product(*fraction_options):
        by_model = {model.model_id: fraction for model, fraction in zip(problem.models, fractions)}
        plan = _solve_fixed_fractions(problem, by_model, method, max_states)
        if plan is None:
            continue
        total = sum(row.objective_j_per_token for row in plan.models.values())
        if best is None or total < best[0] - 1e-12:
            best = (total, plan)
    _require(best is not None, "no feasible placement (even all-CPU breaks the latency bound?)")
    return best[1]


def _solve_fixed_fractions(problem: PlacementProblem, fractions: Mapping[str, float], method: str,
                           max_states: int) -> PlacementPlan | None:
    items: list[_Item] = []
    for model in problem.models:
        items.extend(_items_for(problem, model, fractions[model.model_id]))
    limits = _constraints(problem, items)
    cpu_choice = [0] * len(items)
    if not _feasible(problem, items, cpu_choice, limits):
        # all-CPU always meets capacity/busy; it can only miss a latency bound tighter than 1x
        return None
    choice = None
    solver = "greedy"
    if method in ("auto", "exact"):
        choice = _solve_exact(problem, items, limits, max_states)
        if choice is not None:
            solver = "exact"
        elif method == "exact":
            raise LayerPlacementError("exact placement state space exceeds max_states")
    if choice is None:
        choice = _solve_greedy(problem, items, limits)
    choice = _choose_layers(problem, items, choice)
    _require(_feasible(problem, items, choice, limits), "internal: chosen layers break a limit")
    choice = _prefer_current(problem, items, choice, limits)
    saving = _energy(items, cpu_choice) - _energy(items, choice)
    gap = 0.0 if solver == "exact" else max(0.0, _saving_upper_bound(problem, items, limits) - saving)
    models = {}
    priors = set()
    usage: dict[str, dict[str, int]] = {}
    busy: dict[str, dict[str, float]] = {}
    infeasible = []
    current_prices = {}
    if problem.current:
        # the static placement as it would actually run now: layers of unavailable owners on the host
        available = {device.device_id for device in problem.devices if device.available}
        effective = {model: {layer: (owner if owner in available else DESKTOP_CPU) for layer, owner in owners.items()}
                     for model, owners in problem.current.items()}
        try:
            current_prices = evaluate_owners(problem, effective, fractions)
        except LayerPlacementError as error:
            infeasible.append(str(error))
    for model in problem.models:
        member_indices = [index for index, item in enumerate(items) if item.model_id == model.model_id]
        tokens = _tokens_per_step(problem, model.model_id)
        owners = {}
        chain: dict[int, float] = {}
        cpu_chain: dict[int, float] = {}
        energy = 0.0
        objective = 0.0
        cpu_energy = 0.0
        for index in member_indices:
            item = items[index]
            option = item.options[choice[index]]
            owners[item.layer] = option.owner
            energy += option.expected_j
            objective += option.energy_j
            cpu_energy += item.cpu.expected_j
            for rows, ms in option.ms_by_rows.items():
                chain[rows] = chain.get(rows, 0.0) + ms
            for rows, ms in item.cpu.ms_by_rows.items():
                cpu_chain[rows] = cpu_chain.get(rows, 0.0) + ms
            if option.owner != DESKTOP_CPU:
                priors.update(option.priors)
                device = problem.device(option.owner)
                scope = _scope(device, model.model_id)
                usage.setdefault(option.owner, {})
                usage[option.owner][scope] = usage[option.owner].get(scope, 0) + option.bytes
                busy.setdefault(option.owner, {})
                busy[option.owner][model.model_id] = busy[option.owner].get(model.model_id, 0.0) + option.busy_ms
            elif option.priors:
                priors.update(option.priors)
        models[model.model_id] = ModelPlacement(
            model.model_id, MappingProxyType(owners), fractions[model.model_id], energy / tokens, cpu_energy / tokens,
            current_prices.get(model.model_id), MappingProxyType(chain), MappingProxyType(cpu_chain), tokens,
            objective / tokens)
    return PlacementPlan(MappingProxyType(models), solver, gap, tuple(sorted(priors)),
                         MappingProxyType({k: MappingProxyType(v) for k, v in usage.items()}),
                         MappingProxyType({k: MappingProxyType(v) for k, v in busy.items()}), tuple(infeasible))


def with_envelope_caps(problem: PlacementProblem, growth_ppm: int) -> PlacementProblem:
    """Safe exploration: a device may be asked for at most its measured busy envelope per step times
    ``1 + growth_ppm / 1e6`` (devices without a measured envelope keep their configured cap)."""
    _require(type(growth_ppm) is int and growth_ppm >= 0, "growth_ppm must be an integer >= 0")
    devices = []
    for device in problem.devices:
        if device.busy_envelope_ms is None:
            devices.append(device)
            continue
        cap = device.busy_envelope_ms * (1 + growth_ppm / 1e6)
        if device.max_busy_ms_per_token is not None:
            cap = min(cap, device.max_busy_ms_per_token)
        devices.append(replace(device, max_busy_ms_per_token=cap))
    return replace(problem, devices=tuple(devices))


def restrict_to_executable(problem: PlacementProblem) -> PlacementProblem:
    """Each device may only own layers whose shards it stores AND whose evidence qualified them."""
    return replace(problem, devices=tuple(
        replace(device, allowed_layers=MappingProxyType({
            model: device.stored_layers.get(model, 0) & device.qualified_layers.get(model, 0)
            for model in device.formats}))
        for device in problem.devices))


@dataclass(frozen=True)
class ProvisioningNeed:
    device_id: str
    model_id: str
    layers: tuple[int, ...]
    bytes: int
    needs_shard: bool
    needs_qualification: bool

    def to_json(self) -> dict[str, object]:
        return {"device_id": self.device_id, "model_id": self.model_id, "layers": layer_spec(self.layers),
                "bytes": self.bytes, "needs_shard": self.needs_shard,
                "needs_qualification": self.needs_qualification}


def provisioning_needs(problem: PlacementProblem, plan: PlacementPlan) -> tuple[ProvisioningNeed, ...]:
    """What the plan needs that the devices do not have: shards to push and evidence to collect."""
    needs = []
    by_model = {model.model_id: model for model in problem.models}
    for model_id, placement in sorted(plan.models.items()):
        for device_id, mask in sorted(placement.helper_masks.items()):
            device = problem.device(device_id)
            missing_shard = mask & ~device.stored_layers.get(model_id, 0)
            missing_evidence = mask & ~device.qualified_layers.get(model_id, 0)
            missing = missing_shard | missing_evidence
            if not missing:
                continue
            layers = mask_layers(missing)
            shard_format = device.formats[model_id]
            size = sum(by_model[model_id].bytes_for(shard_format, layer) for layer in mask_layers(missing_shard))
            needs.append(ProvisioningNeed(device_id, model_id, layers, size, bool(missing_shard),
                                          bool(missing_evidence)))
    return tuple(needs)


def placement_report(problem: PlacementProblem, *, method: str = "auto") -> dict[str, object]:
    """Ideal plan, executable plan (stored + qualified shards only) and what separates them."""
    ideal = solve_placement(problem, method=method)
    executable = solve_placement(restrict_to_executable(problem), method=method)
    return {
        "schema": LAYER_PLACEMENT_SCHEMA + "-report",
        "ideal": ideal.to_json(),
        "executable": executable.to_json(),
        "provisioning_needs": [row.to_json() for row in provisioning_needs(problem, ideal)],
        "ideal_vs_executable_j_per_token": {
            model: round(executable.models[model].energy_j_per_token - ideal.models[model].energy_j_per_token, 6)
            for model in sorted(ideal.models)
        },
    }
