"""New resources and new models for measured layer placement (WS11).

``probe_plan`` / ``profile_from_probe``: a new helper runs a short capability probe -- per-layer FFN calls
at rows 1, 2 and 4 on a synthetic or real shard of a known size, timed by the worker (compute) and the
client (round trip), optionally with a phone meter window -- and the result becomes measured profile
rows plus the device's priors (weight-stream rate per format, row factor, transport overhead, energy per
compute-ms). From then on the planner prices that device like any other.

``onboard_model``: fail-closed check that a new model can be split at all and onto which devices:
the desktop FFN split exists only for dense layers of the architectures llama-server builds with
``build_dense_ffn_split``; every candidate layer needs gate/up/down of one uniform shape; a device
needs a shard format it can execute for the model's weight types, block-aligned widths, and room for at
least one layer. Every refusal names its reason.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from .layer_placement import (
    DeviceCostPriors,
    GGML_BLOCKS,
    HelperDevice,
    LayerPlacementError,
    LayerPlacementProfile,
    ModelLayers,
    ffn_layer_bytes,
)


# llama-server builds the FFN split (build_dense_ffn_split) for these architectures' dense FFN layers
# (src/models/qwen3.cpp:194, llama.cpp:194, gemma4.cpp:467); the worker's activation per architecture.
SPLIT_ARCHITECTURES = MappingProxyType({"qwen3": "swiglu", "llama": "swiglu", "gemma4": "geglu"})

# What each helper kind executes: shard format -> (weight types accepted, source requirement).
#   "same"   the shard holds the host GGUF's own bytes (ffn_shard_gguf.py slices them)
#   "origin" the shard holds the quantized origin's bytes; the host f16 must be its exact dequantization
DEVICE_FORMATS = MappingProxyType({
    "htp": MappingProxyType({"f16": (("F16",), "same"), "q4_0": (("Q4_0",), "same"), "q8_0": (("Q8_0",), "same")}),
    "cpu-worker": MappingProxyType({"f16": (("F16",), "same"), "q4_0": (("Q4_0",), "same"),
                                    "q8_0": (("Q8_0",), "same")}),
    "pixel-packed": MappingProxyType({"q4k-packed": (("Q4_K", "Q6_K"), "origin"),
                                      "q4_0-packed": (("Q4_0",), "origin"), "f16": (("F16",), "same")}),
})
# Formats whose execution is not the host's f16 arithmetic -> quantized-numerics evidence class (WS8);
# the packed formats compute on the exact origin weights with a residual correction (rel-L2 + tokens gate).
APPROXIMATE_FORMATS = frozenset({"q4_0", "q8_0"})
PROBE_ROWS = (1, 2, 4)


@dataclass(frozen=True)
class ProbePlan:
    device_id: str
    shard_format: str
    transport: str
    layers: int
    layer_bytes: int
    rows: tuple[int, ...]
    calls_per_rows: int
    warmup_calls: int
    cadence_ms: float
    meter: bool

    def to_json(self) -> dict[str, object]:
        return {"device_id": self.device_id, "format": self.shard_format, "transport": self.transport,
                "layers": self.layers, "layer_bytes": self.layer_bytes, "rows": list(self.rows),
                "calls_per_rows": self.calls_per_rows, "warmup_calls": self.warmup_calls,
                "cadence_ms": self.cadence_ms, "meter": self.meter,
                "total_calls": len(self.rows) * (self.calls_per_rows + self.warmup_calls)}


def probe_plan(device_id: str, shard_format: str, transport: str, layer_bytes: int, *, layers: int = 2,
               calls_per_rows: int = 96, warmup_calls: int = 12, cadence_ms: float = 6.0,
               rows: Sequence[int] = PROBE_ROWS, meter: bool = True) -> ProbePlan:
    """Calls to run on a new helper: ``layers`` resident layers, each row count at server-like cadence.

    96 measured calls per row count keep the standard error of a ~10 ms mean near 1 %; the warm-up calls
    absorb DVFS ramp and first-touch costs; ``cadence_ms`` spaces calls like a decode step does.
    """
    if layer_bytes <= 0 or layers < 1 or calls_per_rows < 24 or not rows or min(rows) < 1:
        raise LayerPlacementError("probe plan needs positive bytes, >= 1 layer, >= 24 calls per row count")
    return ProbePlan(device_id, shard_format, transport, layers, layer_bytes, tuple(sorted(set(rows))),
                     calls_per_rows, warmup_calls, cadence_ms, meter)


@dataclass(frozen=True)
class ProbeResult:
    device_id: str
    calls: int
    by_rows: Mapping[int, Mapping[str, float]]
    priors: DeviceCostPriors
    energy_j_per_call: Mapping[int, float]

    def to_json(self) -> dict[str, object]:
        return {"device_id": self.device_id, "calls": self.calls,
                "by_rows": {str(k): dict(v) for k, v in sorted(self.by_rows.items())},
                "priors": self.priors.to_json(),
                "energy_j_per_call": {str(k): v for k, v in sorted(self.energy_j_per_call.items())}}


def profile_from_probe(profile: LayerPlacementProfile, plan: ProbePlan, model_id: str,
                       calls: Sequence[Mapping[str, object]], *, meter_j_above_idle: float | None = None,
                       idle_w: float = 0.0, evidence: str = "") -> ProbeResult:
    """Fold probe calls into ``profile`` and register the device's priors.

    ``calls`` rows carry ``rows``, ``compute_us``, ``rpc_us`` (or ``overhead_us``) and optionally ``step``
    (negative = warm-up, skipped). ``meter_j_above_idle`` is the phone meter's energy above its idle
    baseline over the measured calls; it is spread over calls in proportion to compute time.
    """
    measured = [row for row in calls if int(row.get("step", 0)) >= 0]
    by_rows: dict[int, dict[str, float]] = {}
    for rows in plan.rows:
        group = [row for row in measured if int(row["rows"]) == rows]
        if len(group) < 24:
            raise LayerPlacementError(f"probe has {len(group)} measured calls at rows {rows} (< 24)")
        compute = sum(float(row["compute_us"]) for row in group) / len(group) / 1000.0
        if "rpc_us" in group[0]:
            rpc = sum(float(row["rpc_us"]) for row in group) / len(group) / 1000.0
        else:
            rpc = compute + sum(float(row["overhead_us"]) for row in group) / len(group) / 1000.0
        if compute <= 0 or rpc < compute:
            raise LayerPlacementError(f"probe timings at rows {rows} are inconsistent")
        by_rows[rows] = {"calls": float(len(group)), "compute_ms": compute, "rpc_ms": rpc}
        profile.observe_call_summary(plan.device_id, model_id, plan.shard_format, plan.transport, rows, len(group),
                                     rpc, compute, evidence)
    base = by_rows[min(by_rows)]
    energy = {}
    active_w = 0.0
    if meter_j_above_idle is not None:
        total_compute_s = sum(row["compute_ms"] * row["calls"] for row in by_rows.values()) / 1000.0
        if total_compute_s <= 0 or meter_j_above_idle < 0:
            raise LayerPlacementError("probe meter window is inconsistent")
        active_w = meter_j_above_idle / total_compute_s
        for rows, row in by_rows.items():
            energy[rows] = active_w * row["compute_ms"] / 1000.0
            profile.observe_call_energy(plan.device_id, model_id, rows, energy[rows] * row["calls"],
                                        int(row["calls"]), evidence)
    priors = DeviceCostPriors(
        bytes_per_s=MappingProxyType({plan.shard_format: plan.layer_bytes / (base["compute_ms"] / 1000.0)}),
        overhead_ms=MappingProxyType({plan.transport: MappingProxyType(
            {rows: row["rpc_ms"] - row["compute_ms"] for rows, row in by_rows.items()})}),
        rows_factor=MappingProxyType({rows: row["compute_ms"] / base["compute_ms"] for rows, row in by_rows.items()}),
        active_marginal_w=active_w, idle_w=idle_w,
        provenance=MappingProxyType({"all": "probe:" + (evidence or plan.device_id)}),
    )
    old = profile.device_priors.get(plan.device_id)
    if old is not None:
        # keep the other formats' and transports' priors the device already had
        priors = DeviceCostPriors(
            bytes_per_s=MappingProxyType({**old.bytes_per_s, **priors.bytes_per_s}),
            overhead_ms=MappingProxyType({**old.overhead_ms, **priors.overhead_ms}),
            rows_factor=priors.rows_factor,
            active_marginal_w=active_w or old.active_marginal_w, idle_w=idle_w or old.idle_w,
            provenance=MappingProxyType({**old.provenance, **priors.provenance}),
        )
    profile.device_priors[plan.device_id] = priors
    profile.revision += 1
    return ProbeResult(plan.device_id, sum(int(row["calls"]) for row in by_rows.values()),
                       MappingProxyType({k: MappingProxyType(v) for k, v in by_rows.items()}), priors,
                       MappingProxyType(energy))


# --------------------------------------------------------------------------------------------------
# model onboarding
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceVerdict:
    device_id: str
    shard_format: str | None
    max_layers: int
    reasons: tuple[str, ...]
    approximate_numerics: bool = False

    @property
    def ok(self) -> bool:
        return self.shard_format is not None and self.max_layers > 0 and not self.reasons

    def to_json(self) -> dict[str, object]:
        return {"device_id": self.device_id, "format": self.shard_format, "max_layers": self.max_layers,
                "ok": self.ok, "reasons": list(self.reasons), "approximate_numerics": self.approximate_numerics}


@dataclass(frozen=True)
class OnboardingVerdict:
    model_id: str
    splittable: bool
    reasons: tuple[str, ...]
    cpu_layers: tuple[int, ...]
    devices: tuple[DeviceVerdict, ...]
    model: ModelLayers | None

    @property
    def placeable(self) -> bool:
        return self.splittable and any(row.ok for row in self.devices)

    def to_json(self) -> dict[str, object]:
        return {"model_id": self.model_id, "splittable": self.splittable, "placeable": self.placeable,
                "reasons": list(self.reasons), "cpu_layers": list(self.cpu_layers),
                "devices": [row.to_json() for row in self.devices]}


@dataclass(frozen=True)
class DeviceCapability:
    """What a helper can execute: its kind (``htp``, ``cpu-worker``, ``pixel-packed``), preferred formats
    in order, capacity for this model's shards, sessions and the per-session limit."""

    device_id: str
    kind: str
    preferred_formats: tuple[str, ...]
    capacity_bytes: int
    transport: str
    sessions: int = 1
    session_limit_bytes: int = 0
    residency: str = "co-resident"
    allow_approximate: bool = False


def onboard_model(metadata: Mapping[str, object], *, n_gpu_layers: int, devices: Sequence[DeviceCapability],
                  origin_metadata: Mapping[str, object] | None = None,
                  origin_dequantizes_exactly: bool | None = None) -> OnboardingVerdict:
    """Can a new model be placed, and on which device with which shard format?

    ``metadata``: ``model_id``, ``architecture``, ``block_count``, ``n_embd``, ``n_ff``, and per-layer
    ``layers: [{index, gate:{type, shape}, up:{...}, down:{...}, moe: bool}]`` (the CLI reads it from the
    GGUF). ``n_gpu_layers`` is the server's ``-ngl`` (llama.cpp offloads the last ``n_gpu_layers - 1``
    blocks plus the output layer when ``-ngl`` <= ``block_count``). ``origin_metadata`` is the quantized
    file a packed format would copy from; ``origin_dequantizes_exactly`` the result of the exact
    dequantization check (prepare_pixel_packed_weights / pixel_packed_shard verify).
    """
    reasons = []
    model_id = str(metadata.get("model_id", ""))
    architecture = str(metadata.get("architecture", ""))
    if architecture not in SPLIT_ARCHITECTURES:
        reasons.append(f"ARCHITECTURE_UNSUPPORTED: {architecture!r} has no build_dense_ffn_split "
                       f"(supported: {', '.join(sorted(SPLIT_ARCHITECTURES))})")
    block_count = int(metadata.get("block_count", 0))
    n_embd, n_ff = int(metadata.get("n_embd", 0)), int(metadata.get("n_ff", 0))
    if block_count <= 0 or n_embd <= 0 or n_ff <= 0:
        reasons.append("GEOMETRY_MISSING: block_count, n_embd and n_ff are required")
    first_gpu = max(block_count + 1 - n_gpu_layers, 0) if n_gpu_layers > 0 else block_count
    cpu_layers = tuple(range(min(first_gpu, block_count)))
    if any(layer >= 64 for layer in cpu_layers):
        reasons.append("LAYER_MASK_LIMIT: helper layer masks are 64-bit; CPU layers beyond 63 cannot be placed")
        cpu_layers = tuple(layer for layer in cpu_layers if layer < 64)
    if not cpu_layers:
        reasons.append("NO_CPU_LAYERS: the whole model is GPU-resident at this -ngl; nothing to place")
    layers = {int(row["index"]): row for row in metadata.get("layers", ())}
    types: set[tuple[str, str, str]] = set()
    for layer in cpu_layers:
        row = layers.get(layer)
        if row is None:
            reasons.append(f"FFN_MISSING: layer {layer} has no ffn_gate/up/down")
            continue
        if row.get("moe"):
            reasons.append(f"MOE_LAYER: layer {layer} is a mixture-of-experts layer (no dense split)")
            continue
        shapes = (tuple(row["gate"]["shape"]), tuple(row["up"]["shape"]), tuple(row["down"]["shape"]))
        if shapes != ((n_embd, n_ff), (n_embd, n_ff), (n_ff, n_embd)):
            reasons.append(f"FFN_SHAPE: layer {layer} gate/up/down {shapes} differ from ({n_embd}x{n_ff})")
            continue
        types.add((row["gate"]["type"], row["up"]["type"], row["down"]["type"]))
    splittable = not reasons
    host_types = {item for row in types for item in row}
    verdicts = []
    model = None
    if splittable:
        sizes: dict[str, int] = {}
        overrides: dict[str, dict[int, int]] = {}
        for capability in devices:
            verdicts.append(_device_verdict(capability, host_types, cpu_layers, layers, n_embd, n_ff,
                                            origin_metadata, origin_dequantizes_exactly, sizes, overrides))
        host_format = "f16" if host_types == {"F16"} else "host"
        if host_format == "host":
            sizes["host"] = max(_layer_bytes_of(layers[layer]) for layer in cpu_layers)
        else:
            sizes["f16"] = ffn_layer_bytes(n_embd, n_ff, "F16")
        model = ModelLayers(model_id=model_id, cpu_layers=cpu_layers, layer_bytes=MappingProxyType(dict(sizes)),
                            layer_bytes_overrides=MappingProxyType({k: MappingProxyType(v) for k, v in overrides.items()}),
                            n_embd=n_embd, n_ff=n_ff, host_format=host_format)
    return OnboardingVerdict(model_id, splittable, tuple(reasons), cpu_layers, tuple(verdicts), model)


def _layer_bytes_of(row: Mapping[str, object]) -> int:
    total = 0
    for name in ("gate", "up", "down"):
        weight_type = row[name]["type"]
        ne0, ne1 = row[name]["shape"]
        block, size = GGML_BLOCKS[weight_type]
        total += ne1 * (ne0 // block) * size
    return total


def _device_verdict(capability: DeviceCapability, host_types, cpu_layers, layers, n_embd, n_ff, origin_metadata,
                    origin_exact, sizes, overrides) -> DeviceVerdict:
    table = DEVICE_FORMATS.get(capability.kind)
    if table is None:
        return DeviceVerdict(capability.device_id, None, 0, ("DEVICE_KIND_UNKNOWN: " + capability.kind,))
    refusals = []
    for shard_format in capability.preferred_formats:
        if shard_format not in table:
            refusals.append(f"{shard_format}: not executable on {capability.kind}")
            continue
        accepted, source = table[shard_format]
        if source == "same":
            if not host_types <= set(accepted):
                refusals.append(f"{shard_format}: host weights {sorted(host_types)} are not {list(accepted)}")
                continue
            per_layer = {layer: _layer_bytes_of(layers[layer]) for layer in cpu_layers}
        else:
            origin_layers = {int(row["index"]): row for row in (origin_metadata or {}).get("layers", ())}
            if not origin_layers:
                refusals.append(f"{shard_format}: needs the quantized origin file's metadata")
                continue
            origin_types = {origin_layers[layer][name]["type"] for layer in cpu_layers if layer in origin_layers
                            for name in ("gate", "up", "down")}
            if any(layer not in origin_layers for layer in cpu_layers) or not origin_types <= set(accepted):
                refusals.append(f"{shard_format}: origin weights {sorted(origin_types)} are not {list(accepted)}")
                continue
            if origin_exact is not True:
                refusals.append(f"{shard_format}: the host f16 is not proven to be the exact dequantization of "
                                "the origin (run the exact-dequant check)")
                continue
            per_layer = {layer: _layer_bytes_of(origin_layers[layer]) for layer in cpu_layers}
        if shard_format in APPROXIMATE_FORMATS and not capability.allow_approximate:
            refusals.append(f"{shard_format}: quantized execution is approximate numerics (needs the quantized "
                            "evidence class)")
            continue
        block = max(GGML_BLOCKS[t][0] for t in (accepted if source == "origin" else host_types))
        if n_embd % block or n_ff % block:
            refusals.append(f"{shard_format}: widths {n_embd}/{n_ff} are not multiples of block {block}")
            continue
        largest = max(per_layer.values())
        limit = capability.capacity_bytes
        if capability.session_limit_bytes:
            per_session = capability.session_limit_bytes // largest
            max_layers = min(per_session * capability.sessions, limit // largest)
        else:
            max_layers = limit // largest
        max_layers = min(max_layers, len(cpu_layers))
        if max_layers <= 0:
            refusals.append(f"{shard_format}: one layer ({largest} B) does not fit {capability.device_id}")
            continue
        common = max(set(per_layer.values()), key=list(per_layer.values()).count)
        sizes[shard_format] = common
        odd = {layer: size for layer, size in per_layer.items() if size != common}
        if odd:
            overrides[shard_format] = odd
        return DeviceVerdict(capability.device_id, shard_format, max_layers, (), shard_format in APPROXIMATE_FORMATS)
    return DeviceVerdict(capability.device_id, None, 0, tuple(refusals) or ("NO_FORMAT",))


def helper_device_for(verdict: DeviceVerdict, capability: DeviceCapability, model_id: str,
                      base: HelperDevice | None = None) -> HelperDevice:
    """Add an onboarded model's format to a helper's placement spec (fails closed on a refused verdict)."""
    if not verdict.ok:
        raise LayerPlacementError(f"{capability.device_id} cannot serve {model_id}: {'; '.join(verdict.reasons)}")
    if base is None:
        return HelperDevice(capability.device_id, MappingProxyType({model_id: verdict.shard_format}),
                            capability.capacity_bytes, capability.transport, capability.sessions,
                            capability.session_limit_bytes, capability.residency)
    formats = dict(base.formats)
    formats[model_id] = verdict.shard_format
    return replace(base, formats=MappingProxyType(formats))


__all__ = [
    "APPROXIMATE_FORMATS", "DEVICE_FORMATS", "DeviceCapability", "DeviceVerdict", "OnboardingVerdict",
    "PROBE_ROWS", "ProbePlan", "ProbeResult", "SPLIT_ARCHITECTURES", "helper_device_for", "onboard_model",
    "probe_plan", "profile_from_probe",
]
