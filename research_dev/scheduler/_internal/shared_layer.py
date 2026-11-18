"""Shared FFN layers: the measured fit and the static share plan (portable path).

A host and one helper phone compute disjoint column shares of the same FFN layer at the same time. With
the helper holding the share ``f`` of the columns, the layer takes::

    T(f) = max(c_host + (1 - f) * t_host,  t_link + f * t_phone)        (f > 0)
    T(0) = t_host

``t_host`` is the layer on the host alone, ``c_host`` the fixed cost of sharing it, ``t_link`` the call's
fixed cost on the link and ``t_phone`` the whole layer on the phone. Both sides finish together at
``f* = (t_host + c_host - t_link) / (t_host + t_phone)``.

``fit_shared_layer`` recovers the four numbers from the server's own ``S41SERVERFFNSHAPE`` summary rows
of a share sweep (one row per column count: host share time, phone round trip). ``bracketed_deltas``
compares every shared leg with its two neighbouring host-only legs, so that host drift during the sweep
does not enter the saving. ``plan_static_share`` then picks the column count for a phone of limited
memory, and ``fold_fit_into_profile`` writes the fit into a :class:`LayerPlacementProfile` in the form
the layer-placement planner prices (its option cost is the same ``max`` without ``c_host``).

Pure functions, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import statistics
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .layer_placement import (
    CallKey,
    CallStats,
    DeviceCostPriors,
    LayerPlacementError,
    LayerPlacementProfile,
    ffn_layer_bytes,
)


SHARED_LAYER_FIT_SCHEMA = "research-scheduler-shared-layer-fit-v1"
STATIC_SHARE_PLAN_SCHEMA = "research-scheduler-static-share-plan-v1"
BINDING_CONSTRAINTS = ("optimum", "session", "memory", "not-beneficial")
# The fit separates neighbouring shares only down to about one sixteenth of the layer: near the optimum
# the measured layer time is the mean of a maximum of two noisy times and exceeds the model by 0.1-0.3 ms.
DEFAULT_SHARE_STEPS = 16
MINIMUM_FIT_POINTS = 3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LayerPlacementError(message)


def _least_squares(points: Sequence[tuple[float, float]], name: str) -> tuple[float, float]:
    """(intercept, slope) of y = intercept + slope * x."""
    _require(len(points) >= MINIMUM_FIT_POINTS, f"{name} fit needs at least {MINIMUM_FIT_POINTS} shares")
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    spread = sum((x - mean_x) ** 2 for x, _ in points)
    _require(spread > 0.0, name + " fit needs distinct shares")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / spread
    return mean_y - slope * mean_x, slope


@dataclass(frozen=True)
class SharePoint:
    """One measured column share of a sweep (call-weighted over the server rows of that width)."""

    columns: int
    share: float
    calls: int
    layer_ms: float       # wall time of the shared layer (host share and phone call overlapped)
    host_ms: float        # the host's own share
    rpc_ms: float         # the phone call as the host sees it
    compute_ms: float     # the phone's compute inside the call

    def to_json(self) -> dict[str, object]:
        return {"calls": self.calls, "columns": self.columns, "compute_ms": self.compute_ms,
                "host_ms": self.host_ms, "layer_ms": self.layer_ms, "rpc_ms": self.rpc_ms, "share": self.share}

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "SharePoint":
        return cls(int(value["columns"]), float(value["share"]), int(value["calls"]), float(value["layer_ms"]),
                   float(value["host_ms"]), float(value["rpc_ms"]), float(value["compute_ms"]))


@dataclass(frozen=True)
class SharedLayerFit:
    rows: int
    n_ff: int
    t_host_ms: float
    c_host_ms: float
    t_link_ms: float
    t_phone_ms: float
    points: tuple[SharePoint, ...] = ()

    def __post_init__(self) -> None:
        _require(type(self.rows) is int and self.rows >= 1, "fit rows must be >= 1")
        _require(type(self.n_ff) is int and self.n_ff > 0, "fit n_ff must be positive")
        _require(self.t_host_ms > 0.0 and self.t_phone_ms > 0.0, "fit needs positive host and phone layer times")
        _require(self.t_link_ms >= 0.0, "fit link time is negative")

    def host_ms(self, share: float) -> float:
        """The host's part of a layer whose helper holds ``share`` of the columns."""
        if share <= 0.0:
            return self.t_host_ms
        return 0.0 if share >= 1.0 else self.c_host_ms + (1.0 - share) * self.t_host_ms

    def phone_ms(self, share: float) -> float:
        return 0.0 if share <= 0.0 else self.t_link_ms + share * self.t_phone_ms

    def layer_ms(self, share: float) -> float:
        return max(self.host_ms(share), self.phone_ms(share))

    def gain_ms(self, share: float) -> float:
        """Layer time saved against the host alone (negative = the share is slower)."""
        return self.t_host_ms - self.layer_ms(share)

    @property
    def f_star(self) -> float:
        """Share at which the host's part and the phone call end together, inside [0, 1]."""
        return min(1.0, max(0.0, (self.t_host_ms + self.c_host_ms - self.t_link_ms)
                            / (self.t_host_ms + self.t_phone_ms)))

    @property
    def f_plain(self) -> float:
        """The same share without the fixed cost of sharing (the two-endpoint form)."""
        return min(1.0, max(0.0, (self.t_host_ms - self.t_link_ms) / (self.t_host_ms + self.t_phone_ms)))

    def to_json(self) -> dict[str, object]:
        return {
            "schema": SHARED_LAYER_FIT_SCHEMA,
            "rows": self.rows,
            "n_ff": self.n_ff,
            "t_host_ms": self.t_host_ms,
            "c_host_ms": self.c_host_ms,
            "t_link_ms": self.t_link_ms,
            "t_phone_ms": self.t_phone_ms,
            "f_star": self.f_star,
            "t_star_ms": self.layer_ms(self.f_star),
            "points": [point.to_json() for point in self.points],
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "SharedLayerFit":
        _require(value.get("schema") == SHARED_LAYER_FIT_SCHEMA, "not a shared-layer fit")
        return cls(int(value["rows"]), int(value["n_ff"]), float(value["t_host_ms"]), float(value["c_host_ms"]),
                   float(value["t_link_ms"]), float(value["t_phone_ms"]),
                   tuple(SharePoint.from_json(row) for row in value.get("points", ())))


def share_points(shape_rows: Iterable[Mapping[str, object]], *, n_ff: int, rows: int = 1) -> tuple[SharePoint, ...]:
    """Parsed ``S41SERVERFFNSHAPE`` rows -> one point per column count at ``rows`` rows per call."""
    buckets: dict[int, list[Mapping[str, object]]] = {}
    for row in shape_rows:
        if int(row.get("tokens", 0)) != rows or int(row.get("calls", 0)) <= 0:
            continue
        columns = int(row["columns"])
        _require(0 < columns <= n_ff, f"shape row has {columns} columns outside 1..{n_ff}")
        for name in ("host_mean_ms", "rpc_mean_ms", "compute_mean_ms", "overlap_mean_ms"):
            _require(name in row, "shape row lacks " + name)
        buckets.setdefault(columns, []).append(row)
    points = []
    for columns in sorted(buckets):
        group = buckets[columns]
        calls = sum(int(row["calls"]) for row in group)

        def mean(name: str) -> float:
            return sum(float(row[name]) * int(row["calls"]) for row in group) / calls

        points.append(SharePoint(columns, columns / n_ff, calls, mean("overlap_mean_ms"), mean("host_mean_ms"),
                                 mean("rpc_mean_ms"), mean("compute_mean_ms")))
    return tuple(points)


def fit_shared_layer(shape_rows: Iterable[Mapping[str, object]], *, n_ff: int, rows: int = 1) -> SharedLayerFit:
    """Least-squares fit of the shared-layer model to a share sweep.

    The phone call is fitted over every share (``rpc = t_link + f * t_phone``); the host's part over the
    shares that leave it work (``host = c_host + (1 - f) * t_host``, f < 1). Fails closed on fewer than
    three usable shares per side or on a fit without positive layer times.
    """
    _require(type(n_ff) is int and n_ff > 0, "n_ff must be positive")
    points = share_points(shape_rows, n_ff=n_ff, rows=rows)
    t_link, t_phone = _least_squares([(point.share, point.rpc_ms) for point in points], "phone call")
    c_host, t_host = _least_squares([(1.0 - point.share, point.host_ms) for point in points if point.share < 1.0],
                                    "host share")
    _require(t_host > 0.0 and t_phone > 0.0, "sweep does not show the layer time growing with the share")
    _require(t_link >= 0.0, "sweep gives a negative link time")
    return SharedLayerFit(rows, n_ff, t_host, c_host, t_link, t_phone, points)


# --------------------------------------------------------------------------------------------------
# shared legs against their neighbouring host-only legs
# --------------------------------------------------------------------------------------------------

LEG_FIELDS = ("columns", "share", "ms_token", "host_j_token", "total_j_token", "cpu_w", "gpu_w", "phone_w")


def bracketed_deltas(legs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Every shared leg against the mean of the nearest host-only leg before and after it.

    A leg is ``{"columns", "share", "ms_token", "host_j_token", "total_j_token", "cpu_w", "gpu_w",
    "phone_w"}`` in run order; ``columns`` 0 marks a host-only leg. ``phone_w`` may be None (unmetered):
    the phone then adds nothing to the leg's energy and ``phone_metered`` is False.
    """
    out = []
    for index, leg in enumerate(legs):
        if not leg["columns"]:
            continue
        before = next((legs[j] for j in range(index - 1, -1, -1) if not legs[j]["columns"]), None)
        after = next((legs[j] for j in range(index + 1, len(legs)) if not legs[j]["columns"]), None)
        near = [row for row in (before, after) if row is not None]
        _require(bool(near), "a shared leg has no host-only leg to compare with")

        def mean(function) -> float:
            return statistics.fmean(function(row) for row in near)

        def phone_j(row: Mapping[str, object]) -> float:
            return (float(row["phone_w"]) if row.get("phone_w") is not None else 0.0) * float(row["ms_token"]) * 1e-3

        base_ms = mean(lambda row: float(row["ms_token"]))
        base_j = mean(lambda row: float(row["total_j_token"]))
        base_host_j = mean(lambda row: float(row["host_j_token"]))
        ms = float(leg["ms_token"])
        out.append({
            "share": float(leg["share"]), "columns": int(leg["columns"]), "brackets": len(near),
            "ms": ms, "j": float(leg["total_j_token"]), "base_ms": base_ms, "base_j": base_j,
            "d_time": ms / base_ms - 1.0, "d_energy": float(leg["total_j_token"]) / base_j - 1.0,
            "phone_w": leg.get("phone_w"), "phone_metered": leg.get("phone_w") is not None,
            "cpu_w": float(leg["cpu_w"]), "gpu_w": float(leg["gpu_w"]),
            "base_cpu_w": mean(lambda row: float(row["cpu_w"])), "base_gpu_w": mean(lambda row: float(row["gpu_w"])),
            "host_dj": base_host_j - float(leg["host_j_token"]), "dt_ms": base_ms - ms,
            "cpu_j": float(leg["cpu_w"]) * ms * 1e-3, "gpu_j": float(leg["gpu_w"]) * ms * 1e-3,
            "phone_j": phone_j(leg), "host_j": float(leg["host_j_token"]),
            "base_cpu_j": mean(lambda row: float(row["cpu_w"]) * float(row["ms_token"]) * 1e-3),
            "base_gpu_j": mean(lambda row: float(row["gpu_w"]) * float(row["ms_token"]) * 1e-3),
            "base_host_j": base_host_j, "base_phone_j": mean(phone_j),
        })
    return out


def pool_deltas(deltas: Iterable[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    """Per share over all legs: time and energy change (mean, range), the alone and shared token, and the
    marginal host power of the time the share removed (``P_F``: host joules saved per second saved)."""
    groups: dict[float, list[Mapping[str, object]]] = {}
    for row in deltas:
        groups.setdefault(round(float(row["share"]), 4), []).append(row)
    out: dict[str, dict[str, object]] = {}
    for share in sorted(groups):
        rows = groups[share]

        def mean(key: str) -> float:
            return statistics.fmean(float(row[key]) for row in rows)

        time_saving = [-float(row["d_time"]) * 100.0 for row in rows]
        energy_saving = [(1.0 - (float(row["host_j"]) + float(row["phone_j"])) / float(row["base_host_j"])) * 100.0
                         for row in rows]
        marginal = [float(row["host_dj"]) / (float(row["dt_ms"]) * 1e-3) for row in rows if float(row["dt_ms"]) > 0.0]
        out[str(share)] = {
            "n_legs": len(rows),
            "alone": {"cpu_j": mean("base_cpu_j"), "gpu_j": mean("base_gpu_j"), "total_j": mean("base_host_j"),
                      "ms": mean("base_ms")},
            "shared": {"cpu_j": mean("cpu_j"), "gpu_j": mean("gpu_j"), "phone_j": mean("phone_j"),
                       "total_j": mean("host_j") + mean("phone_j"), "ms": mean("ms")},
            "time_saving_mean": statistics.fmean(time_saving), "time_saving_min": min(time_saving),
            "time_saving_max": max(time_saving),
            "energy_saving_mean": statistics.fmean(energy_saving), "energy_saving_min": min(energy_saving),
            "energy_saving_max": max(energy_saving),
            "marginal_host_w": statistics.fmean(marginal) if marginal else None,
            "phone_metered": all(bool(row["phone_metered"]) for row in rows),
        }
    return out


# --------------------------------------------------------------------------------------------------
# the static share for one phone
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PhoneCapacity:
    """What a phone can keep resident: bytes in total, sessions, and bytes per session (0 = no limit)."""

    usable_bytes: int
    sessions: int = 1
    session_limit_bytes: int = 0

    def __post_init__(self) -> None:
        _require(type(self.usable_bytes) is int and self.usable_bytes >= 0, "usable bytes must be an integer >= 0")
        _require(type(self.sessions) is int and self.sessions >= 1, "sessions must be >= 1")
        _require(type(self.session_limit_bytes) is int and self.session_limit_bytes >= 0,
                 "session limit must be an integer >= 0")

    def layers_that_fit(self, layer_bytes: int, wanted: int) -> tuple[int, str]:
        """(layers, what bounds them) for shards of ``layer_bytes`` per layer."""
        by_memory = self.usable_bytes // layer_bytes
        per_session = by_memory if not self.session_limit_bytes else self.session_limit_bytes // layer_bytes
        by_sessions = self.sessions * per_session
        layers = min(wanted, by_memory, by_sessions)
        if layers == wanted:
            return layers, "optimum"
        return layers, "memory" if by_memory <= by_sessions else "session"


@dataclass(frozen=True)
class StaticSharePlan:
    columns: int
    share: float
    layers: tuple[int, ...]
    sessions: tuple[tuple[int, ...], ...]      # the layers of each phone session, contiguous
    layer_bytes: int
    resident_bytes: int
    predicted_layer_ms: float
    predicted_token_ms_saved: float
    binding_constraint: str
    alternatives: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        _require(self.binding_constraint in BINDING_CONSTRAINTS, "unknown binding constraint")

    @property
    def beneficial(self) -> bool:
        return self.binding_constraint != "not-beneficial"

    def to_json(self) -> dict[str, object]:
        return {
            "schema": STATIC_SHARE_PLAN_SCHEMA,
            "columns": self.columns,
            "share": self.share,
            "layers": list(self.layers),
            "sessions": [list(row) for row in self.sessions],
            "layer_bytes": self.layer_bytes,
            "resident_bytes": self.resident_bytes,
            "predicted_layer_ms": self.predicted_layer_ms,
            "predicted_token_ms_saved": self.predicted_token_ms_saved,
            "binding_constraint": self.binding_constraint,
            "alternatives": [dict(row) for row in self.alternatives],
        }

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "StaticSharePlan":
        _require(value.get("schema") == STATIC_SHARE_PLAN_SCHEMA, "not a static share plan")
        return cls(int(value["columns"]), float(value["share"]), tuple(int(x) for x in value["layers"]),
                   tuple(tuple(int(x) for x in row) for row in value["sessions"]), int(value["layer_bytes"]),
                   int(value["resident_bytes"]), float(value["predicted_layer_ms"]),
                   float(value["predicted_token_ms_saved"]), str(value["binding_constraint"]),
                   tuple(MappingProxyType(dict(row)) for row in value.get("alternatives", ())))


def _session_layout(layers: Sequence[int], layer_bytes: int, capacity: PhoneCapacity) -> tuple[tuple[int, ...], ...]:
    """Contiguous, balanced blocks over the fewest sessions that hold the layers."""
    if not layers:
        return ()
    per_session = len(layers) if not capacity.session_limit_bytes else capacity.session_limit_bytes // layer_bytes
    count = -(-len(layers) // per_session)
    base, extra = divmod(len(layers), count)
    out, start = [], 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        out.append(tuple(layers[start:start + size]))
        start += size
    return tuple(out)


def plan_static_share(fit: SharedLayerFit, *, cpu_layers: Sequence[int], n_embd: int, column_quantum: int,
                      weight_type: str, capacity: PhoneCapacity, share_steps: int = DEFAULT_SHARE_STEPS,
                      stored_columns: int | None = None) -> StaticSharePlan:
    """The column share one phone should hold of every CPU-resident FFN layer it can keep.

    Candidates are the multiples of ``n_ff / share_steps`` (rounded down to the worker's column quantum),
    up to ``stored_columns`` when the shards already on the phone bound them. Each candidate keeps as many
    of the lowest CPU layers as the phone's memory and sessions allow; the plan maximises
    ``layers x gain per layer`` and prefers fewer resident bytes on a tie. When memory binds, a layer
    subset at a larger share usually beats a smaller share on every layer, because the fixed cost of a
    call is paid per layer. A sweep whose best candidate saves nothing gives ``not-beneficial``.
    """
    layers_wanted = tuple(sorted(cpu_layers))
    _require(len(set(layers_wanted)) == len(layers_wanted) and bool(layers_wanted), "CPU layers must be distinct")
    _require(type(column_quantum) is int and column_quantum >= 1, "column quantum must be >= 1")
    _require(type(share_steps) is int and share_steps >= 2, "share steps must be >= 2")
    limit = fit.n_ff if stored_columns is None else stored_columns
    _require(0 < limit <= fit.n_ff, "stored columns must be inside 1..n_ff")
    candidates = sorted({(fit.n_ff * step // share_steps) // column_quantum * column_quantum
                         for step in range(1, share_steps + 1)})
    rows = []
    for columns in candidates:
        if columns <= 0 or columns > limit:
            continue
        share = columns / fit.n_ff
        layer_bytes = ffn_layer_bytes(n_embd, columns, weight_type)
        count, bound = capacity.layers_that_fit(layer_bytes, len(layers_wanted))
        gain = fit.gain_ms(share)
        rows.append({"columns": columns, "share": share, "layers": count, "layer_bytes": layer_bytes,
                     "resident_bytes": count * layer_bytes, "layer_ms": fit.layer_ms(share),
                     "gain_ms_per_layer": gain, "token_ms_saved": count * gain, "bound": bound})
    _require(bool(rows), "no column share is available below the stored columns")
    best = max(rows, key=lambda row: (row["token_ms_saved"], -row["resident_bytes"]))
    alternatives = tuple(MappingProxyType(row) for row in rows)
    if best["token_ms_saved"] <= 0.0 or best["layers"] == 0:
        return StaticSharePlan(0, 0.0, (), (), 0, 0, fit.t_host_ms, 0.0, "not-beneficial", alternatives)
    unbounded = max(rows, key=lambda row: (len(layers_wanted) * row["gain_ms_per_layer"], -row["layer_bytes"]))
    if best["columns"] == unbounded["columns"] and best["layers"] == len(layers_wanted):
        constraint = "optimum"
    else:
        # what stopped the phone from holding the unbounded optimum on every layer
        constraint = unbounded["bound"] if unbounded["bound"] != "optimum" else best["bound"]
    chosen = layers_wanted[:best["layers"]]
    return StaticSharePlan(
        columns=best["columns"], share=best["share"], layers=chosen,
        sessions=_session_layout(chosen, best["layer_bytes"], capacity), layer_bytes=best["layer_bytes"],
        resident_bytes=best["resident_bytes"], predicted_layer_ms=best["layer_ms"],
        predicted_token_ms_saved=best["token_ms_saved"], binding_constraint=constraint,
        alternatives=alternatives)


# --------------------------------------------------------------------------------------------------
# the fit in the planner's own profile
# --------------------------------------------------------------------------------------------------


def fold_fit_into_profile(profile: LayerPlacementProfile, fit: SharedLayerFit, *, model_id: str, device_id: str,
                          shard_format: str, transport: str, layer_bytes: int, evidence: str = "",
                          host_ffn_w: float | None = None, phone_active_w: float | None = None,
                          phone_idle_w: float = 0.0) -> None:
    """Write a fit into ``profile`` so that the layer-placement planner prices this pair from it.

    The planner's option cost is ``max((1 - f) * cpu_ms, overhead + f * compute)``, so the host layer goes
    in as ``t_host`` and the phone's full-width call as ``rpc = t_link + t_phone`` with ``compute =
    t_phone``. ``c_host`` has no slot there (the planner's layer is about 3 % too fast near the optimum);
    it stays in the fit. ``layer_bytes`` is the full-width layer in ``shard_format``.
    """
    _require(type(layer_bytes) is int and layer_bytes > 0, "layer bytes must be positive")
    calls = float(sum(point.calls for point in fit.points)) or 1.0
    note = (evidence,) if evidence else ()
    profile.observe_desktop_layer(model_id, fit.rows, fit.t_host_ms, evidence, provenance="derived")
    profile.calls[CallKey(device_id, model_id, shard_format, fit.rows, transport)] = CallStats(
        calls, fit.t_link_ms + fit.t_phone_ms, fit.t_phone_ms, provenance="derived", evidence=note)
    if host_ffn_w is not None:
        profile.desktop_power = replace(profile.desktop_power, ffn_w=float(host_ffn_w), provenance="derived",
                                        evidence=note)
    old = profile.device_priors.get(device_id)
    rate = layer_bytes / (fit.t_phone_ms / 1000.0)
    overhead = {name: dict(table) for name, table in (old.overhead_ms.items() if old else ())}
    overhead.setdefault(transport, {})[fit.rows] = fit.t_link_ms
    marginal = old.active_marginal_w if old else 0.0
    if phone_active_w is not None:
        marginal = max(0.0, float(phone_active_w) - float(phone_idle_w))
    profile.device_priors[device_id] = DeviceCostPriors(
        bytes_per_s=MappingProxyType({**(dict(old.bytes_per_s) if old else {}), shard_format: rate}),
        overhead_ms=MappingProxyType({name: MappingProxyType(table) for name, table in overhead.items()}),
        rows_factor=old.rows_factor if old else MappingProxyType({1: 1.0}),
        active_marginal_w=marginal,
        idle_w=float(phone_idle_w) if phone_active_w is not None or old is None else old.idle_w,
        provenance=MappingProxyType({**(dict(old.provenance) if old else {}),
                                     "shared_layer_fit": evidence or "sweep"}),
    )
    profile.revision += 1
