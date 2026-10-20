"""Operator-group placement (WS14): which weight-streaming operator groups of the CPU-resident layers to offload.

The WS11 planner places the FFN of every CPU-resident layer. This module widens each layer's choice to the
operator groups a helper could run, priced with the same measured profile, and the output head:

* ``ffn``: the layer's FFN (WS11; the only kernel with a qualified path today).
* ``attn_proj`` (Q/K/V/O weights). The per-layer call boundary is "after attention(l), before attention(l+1)",
  so the projection unit charged to layer ``l`` is O(l) plus Q/K/V(l+1) when ``l+1`` is CPU-resident too
  (Q/K/V of the model's first CPU layer has no preceding call and stays on the desktop CPU, a constant).

  - ``fused``: the unit runs inside the FFN call of layer ``l`` on the device that owns that FFN at full
    width (attention output in -> O -> residual -> norm -> FFN -> residual -> norm -> Q/K/V out); no round
    trip is added, only compute and the larger reply.
  - ``separate``: the unit runs on any capable device as its own calls (O after attention, Q/K/V before the
    next attention: two calls per layer when both exist), each paying that device's per-call overhead; the
    FFN may stay where it is.

* ``attn_kv``: attention over the KV cache of layer ``l``. Bytes per decoded row = context x KV bytes per
  token (sliding-window layers cap the context at the window). Only the ``fused`` form is modelled: the
  device that runs the layer's fused projections also keeps its KV and runs its attention, so a layer stays
  one call. It is charged with KV memory for every concurrently active request (``slots`` x context, a
  capacity entry that shares the device's memory but not its HTP sessions), the one-time shipping of each
  prompt's KV after prefill (link bulk rate, desktop waiting power), and a recovery penalty (a lost device
  loses its KV: re-prefill and re-ship that layer's share, at a prior loss rate).
* ``output_head``: on the GPU today; decided per model after the layers, against the remaining capacity and
  latency slack (it is one item per model and couples nothing else), including the logits' return transfer.

Every device option is a :class:`~layer_placement.OptionCost` with a ``variant`` and, for separate calls on
another device and for KV memory, ``parts``; the WS11 solvers (exact DP over classes, greedy with a gap
bound) solve the widened items unchanged. With FFN-only kernels the ``""`` options equal WS11's exactly plus
the desktop's projection and KV time, which every option of a layer then carries -- so the FFN-only plan here
is WS11's plan.

Kernels are capabilities: ``HelperDevice.kernels`` (``None`` = ``{"ffn"}``) for what has a qualified path
now; a report asks the same question with hypothetical kernels to say what would pay if built. Nothing here
names a device.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import itertools
import json
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .layer_placement import (
    DESKTOP_CPU,
    GGML_BLOCKS,
    KV_SUFFIX,
    Estimate,
    LayerPlacementError,
    LayerPlacementProfile,
    ModelLayers,
    OptionCost,
    PlacementProblem,
    _Item,
    _choose_layers,
    _constraints,
    _energy,
    _feasible,
    _option_costs,
    _prefer_current,
    _saving_upper_bound,
    _scope,
    _solve_exact,
    _solve_greedy,
    _table_factor,
    _usage_scope,
    _usages,
    _usage_ok,
    _weakest,
    layer_spec,
    restrict_to_executable,
    tensor_bytes,
)


OPGROUP_SCHEMA = "research-scheduler-opgroup-placement-v1"
GEOMETRY_SCHEMA = "research-scheduler-model-geometry-v1"
OP_KERNELS = ("ffn", "attn_proj:fused", "attn_proj:separate", "attn_kv:fused", "output_head")
ALL_KERNELS = frozenset(OP_KERNELS)
FFN_ONLY = frozenset({"ffn"})
# WS20: the runtime capability "two helpers compute disjoint column shares of ONE layer's FFN at the same time
# and the server sums their partial outputs". It is a kernel-like capability per device (both devices of a pair
# need it) and never part of OP_KERNELS / ALL_KERNELS, so every report and plan without it is unchanged.
PARALLEL_KERNEL = "ffn:parallel"
PARALLEL_VARIANT = "parallel"
KV_CACHE_BYTES_PER_ELEMENT = 2      # f16 KV cache (llama.cpp default cache type)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LayerPlacementError(message)


# --------------------------------------------------------------------------------------------------
# model geometry from GGUF headers
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerGeometry:
    """One block's attention geometry: output widths of the Q/K/V projections, bytes of each projection
    matrix per format, and KV-cache bytes per token (``window`` = sliding window, None = global)."""

    index: int
    q_out: int
    k_out: int
    v_out: int                                   # 0 = no V projection (V derived from K, cached separately)
    o_in: int
    proj_bytes: Mapping[str, Mapping[str, int]]  # format -> {"q", "k", "v", "o"} -> bytes
    kv_bytes_per_token: int
    window: int | None = None

    def bytes_of(self, shard_format: str, names: Iterable[str]) -> int:
        _require(shard_format in self.proj_bytes, f"layer {self.index} has no {shard_format} projection sizes")
        table = self.proj_bytes[shard_format]
        return sum(table.get(name, 0) for name in names)

    def to_json(self) -> dict[str, object]:
        return {"index": self.index, "q_out": self.q_out, "k_out": self.k_out, "v_out": self.v_out,
                "o_in": self.o_in, "kv_bytes_per_token": self.kv_bytes_per_token, "window": self.window,
                "proj_bytes": {fmt: dict(sorted(table.items())) for fmt, table in sorted(self.proj_bytes.items())}}

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "LayerGeometry":
        return cls(int(value["index"]), int(value["q_out"]), int(value["k_out"]), int(value["v_out"]),
                   int(value["o_in"]),
                   MappingProxyType({str(fmt): MappingProxyType({str(k): int(v) for k, v in dict(table).items()})
                                     for fmt, table in dict(value["proj_bytes"]).items()}),
                   int(value["kv_bytes_per_token"]), None if value.get("window") is None else int(value["window"]))


@dataclass(frozen=True)
class ModelGeometry:
    model_id: str
    architecture: str
    n_embd: int
    n_vocab: int
    layers: tuple[LayerGeometry, ...]
    head_bytes: Mapping[str, int]               # format -> bytes of the output head
    head_tied: bool = False                     # the head reuses the token embedding
    sources: tuple[str, ...] = ()

    def layer(self, index: int) -> LayerGeometry:
        _require(0 <= index < len(self.layers), f"{self.model_id} has no layer {index}")
        return self.layers[index]

    def to_json(self) -> dict[str, object]:
        return {"schema": GEOMETRY_SCHEMA, "model_id": self.model_id, "architecture": self.architecture,
                "n_embd": self.n_embd, "n_vocab": self.n_vocab, "head_tied": self.head_tied,
                "head_bytes": dict(sorted(self.head_bytes.items())), "sources": list(self.sources),
                "layers": [layer.to_json() for layer in self.layers]}

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "ModelGeometry":
        _require(value.get("schema") == GEOMETRY_SCHEMA, "not a model geometry")
        return cls(str(value["model_id"]), str(value["architecture"]), int(value["n_embd"]), int(value["n_vocab"]),
                   tuple(LayerGeometry.from_json(row) for row in value["layers"]),
                   MappingProxyType({str(k): int(v) for k, v in dict(value["head_bytes"]).items()}),
                   bool(value.get("head_tied", False)), tuple(str(item) for item in value.get("sources", ())))


def model_geometry(model_id: str, header: Mapping[str, object],
                   formats: Mapping[str, Mapping[str, object]], *, sources: Sequence[str] = ()) -> ModelGeometry:
    """Geometry of a model from a parsed GGUF header (see :func:`read_gguf_header`).

    ``header`` is the model's own file (its tensor shapes define the widths); ``formats`` maps every shard
    format to the header whose tensor TYPES that format stores (``f16`` -> the f16 file itself, a packed
    format -> the quantized origin whose blocks it repacks: same bytes per matrix).
    """
    fields = dict(header["fields"])
    tensors = dict(header["tensors"])
    architecture = str(header["architecture"])
    block_count = int(fields[f"{architecture}.block_count"])
    n_embd = int(fields[f"{architecture}.embedding_length"])
    window = fields.get(f"{architecture}.attention.sliding_window")
    pattern = fields.get(f"{architecture}.attention.sliding_window_pattern")
    layers = []
    for index in range(block_count):
        def shape(name):
            row = tensors.get(f"blk.{index}.{name}.weight")
            return None if row is None else row
        q, k, v, o = shape("attn_q"), shape("attn_k"), shape("attn_v"), shape("attn_output")
        _require(q is not None and k is not None and o is not None, f"{model_id} layer {index} lacks attention tensors")
        q_out, k_out = int(q["shape"][1]), int(k["shape"][1])
        v_out = 0 if v is None else int(v["shape"][1])
        o_in = int(o["shape"][0])
        proj = {}
        for shard_format, source in formats.items():
            source_tensors = dict(source["tensors"])
            table = {}
            for name, key in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output")):
                row = source_tensors.get(f"blk.{index}.{key}.weight")
                if row is None:
                    continue
                columns, rows = int(row["shape"][0]), int(row["shape"][1])
                table[name] = tensor_bytes(str(row["type"]), rows, columns)
            proj[shard_format] = MappingProxyType(table)
        # V is cached even when it is derived from K (gemma4: V = rms_norm(K), stored in the V cache)
        kv_per_token = (k_out + (v_out or k_out)) * KV_CACHE_BYTES_PER_ELEMENT
        swa = bool(pattern[index]) if isinstance(pattern, list) and index < len(pattern) else False
        layers.append(LayerGeometry(index, q_out, k_out, v_out, o_in, MappingProxyType(proj), kv_per_token,
                                    int(window) if swa and window else None))
    head = tensors.get("output.weight")
    tied = head is None
    head_name = "token_embd.weight" if tied else "output.weight"
    head = tensors[head_name]
    n_vocab = int(head["shape"][1])
    head_bytes = {}
    for shard_format, source in formats.items():
        row = dict(source["tensors"]).get(head_name)
        if row is not None:
            head_bytes[shard_format] = tensor_bytes(str(row["type"]), int(row["shape"][1]), int(row["shape"][0]))
    return ModelGeometry(model_id, architecture, n_embd, n_vocab, tuple(layers), MappingProxyType(head_bytes), tied,
                         tuple(sources))


def read_gguf_header(path, gguf_module) -> dict[str, object]:
    """Header of a GGUF (fields + tensor names, types and shapes; no tensor data) with gguf-py's reader."""
    reader = gguf_module.GGUFReader(str(path))
    fields = {}
    for name, value in reader.fields.items():
        if name.startswith("tokenizer.ggml.") and name not in ("tokenizer.ggml.model",):
            continue
        try:
            fields[name] = value.contents()
        except Exception:   # noqa: BLE001 - a field gguf-py cannot decode is not geometry
            continue
    architecture = str(fields["general.architecture"])
    tensors = {tensor.name: {"type": gguf_module.GGMLQuantizationType(tensor.tensor_type).name,
                             "shape": [int(dim) for dim in tensor.shape]}
               for tensor in reader.tensors}
    return {"architecture": architecture, "fields": fields, "tensors": tensors}


# --------------------------------------------------------------------------------------------------
# inputs that are not in the WS11 profile
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextStats:
    """Decode-context distribution of one model's requests (from a trace)."""

    mean_decode_context: float     # tokens attended per decoded token, averaged over decode steps
    max_context: int               # largest prompt + output of one request
    prompt_tokens_per_decoded: float   # prompt tokens per decoded token (KV shipping per decoded token)
    requests: int = 0
    decode_tokens: int = 0

    @classmethod
    def from_requests(cls, rows: Iterable[tuple[int, int]]) -> "ContextStats":
        """``rows`` = (prompt tokens, output tokens) per request."""
        rows = [(int(p), int(n)) for p, n in rows if int(n) > 0]
        _require(bool(rows), "context stats need at least one request with output")
        steps = sum(n for _, n in rows)
        attended = sum(p * n + n * (n - 1) / 2 for p, n in rows)
        return cls(attended / steps, max(p + n for p, n in rows), sum(p for p, _ in rows) / steps, len(rows), steps)

    @classmethod
    def constant(cls, context: int, prompt_tokens_per_decoded: float = 0.0) -> "ContextStats":
        return cls(float(context), int(context), prompt_tokens_per_decoded)

    def to_json(self) -> dict[str, object]:
        return {"mean_decode_context": round(self.mean_decode_context, 3), "max_context": self.max_context,
                "prompt_tokens_per_decoded": round(self.prompt_tokens_per_decoded, 4), "requests": self.requests,
                "decode_tokens": self.decode_tokens}


@dataclass(frozen=True)
class OpGroupInputs:
    """Everything operator groups need beyond the WS11 profile and inventory, each with its provenance."""

    geometry: Mapping[str, ModelGeometry]
    contexts: Mapping[str, ContextStats]
    slots: Mapping[str, int]                          # model -> concurrently active requests (KV residency)
    link_bytes_per_s: Mapping[str, float]             # transport -> bulk bytes/s (KV shipping, larger replies)
    gpu_head_bytes_per_s: float = 288e9               # prior: RTX 4060 Ti peak memory bandwidth
    gpu_active_w: float = 34.0                        # prior: GPU board active / loaded-idle (09-27 audit)
    gpu_loaded_idle_w: float = 27.6
    host_format: str = "f16"
    loss_rate_per_h: float = 1.0                      # prior: helper losses per device-hour (KV recovery)
    prefill_j_per_token: float = 0.376                # prior: ~74 W host at ~197 prompt tok/s (Gemma stream)
    provenance: Mapping[str, str] = field(default_factory=dict)

    # WS20 parallel column split (opt-in; absent = no parallel options, every plan unchanged):
    #   concurrency_penalty_ms   added to the pair time (synchronizing two helper calls on the host);
    #   concurrency_degradation  slowdown of each share's compute when both links/buses are busy at once (>= 1);
    #   concurrency_provenance   "measured" once the WS20 microbenchmark (tools/parallel_split_microbench.py) has
    #                            been ingested, else "prior": an executable plan never contains a parallel split
    #                            priced from a prior (it may only propose a probe).
    parallel_split: bool = False
    concurrency_penalty_ms: float | None = None
    concurrency_degradation: float = 1.0
    concurrency_provenance: str = "prior"
    def __post_init__(self) -> None:
        for model, value in self.slots.items():
            _require(type(value) is int and value >= 1, f"{model} slots must be >= 1")
        for name in ("gpu_head_bytes_per_s", "gpu_active_w", "gpu_loaded_idle_w", "prefill_j_per_token"):
            _require(getattr(self, name) > 0, name + " must be > 0")
        _require(self.loss_rate_per_h >= 0, "loss rate must be >= 0")
        _require(all(value > 0 for value in self.link_bytes_per_s.values()), "link bytes/s must be > 0")
        _require(type(self.parallel_split) is bool, "parallel_split must be a bool")
        _require(self.concurrency_penalty_ms is None or self.concurrency_penalty_ms >= 0,
                 "concurrency_penalty_ms must be >= 0")
        _require(self.concurrency_degradation >= 1.0, "concurrency_degradation must be >= 1 (slowdown factor)")
        _require(self.concurrency_provenance in ("measured", "prior"), "concurrency_provenance is measured or prior")


def device_kernels(device) -> frozenset[str]:
    return FFN_ONLY if device.kernels is None else device.kernels


def with_kernels(problem: PlacementProblem, kernels: Mapping[str, frozenset[str]] | frozenset[str]) -> PlacementProblem:
    """Same problem with every device's kernels replaced (one set for all devices, or per device id)."""
    def pick(device):
        value = kernels.get(device.device_id, device_kernels(device)) if isinstance(kernels, Mapping) else kernels
        return frozenset(value) | FFN_ONLY
    return replace(problem, devices=tuple(replace(device, kernels=pick(device)) for device in problem.devices))


# --------------------------------------------------------------------------------------------------
# per-layer options
# --------------------------------------------------------------------------------------------------


def _proj_unit(model: ModelLayers, geometry: ModelGeometry, layer: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(projections of THIS layer in the unit, projections of the NEXT layer in the unit): O(l), Q/K/V(l+1)."""
    following = ("q", "k", "v") if (layer + 1) in model.cpu_layers else ()
    return ("o",), following


def _unit_bytes(geometry: ModelGeometry, layer: int, shard_format: str, own: Sequence[str], nxt: Sequence[str]) -> int:
    size = geometry.layer(layer).bytes_of(shard_format, own)
    if nxt:
        size += geometry.layer(layer + 1).bytes_of(shard_format, nxt)
    return size


def _kv_context(stats: ContextStats, layer_geometry: LayerGeometry, *, peak: bool = False) -> float:
    context = stats.max_context if peak else stats.mean_decode_context
    return min(context, layer_geometry.window) if layer_geometry.window else context


@dataclass(frozen=True)
class _LayerCosts:
    """Host-side times of one layer's non-FFN groups (ms per step at each row count)."""

    proj_ms: Mapping[int, float]
    kv_ms: Mapping[int, float]
    proj_host_bytes: int
    kv_bytes_by_rows: Mapping[int, float]


def _host_costs(problem: PlacementProblem, inputs: OpGroupInputs, model: ModelLayers, layer: int) -> _LayerCosts:
    profile = problem.profile
    geometry = inputs.geometry[model.model_id]
    own, nxt = _proj_unit(model, geometry, layer)
    host_bytes = _unit_bytes(geometry, layer, inputs.host_format, own, nxt)
    rate = profile.desktop_bytes_per_s
    stats = inputs.contexts[model.model_id]
    kv_row = _kv_context(stats, geometry.layer(layer)) * geometry.layer(layer).kv_bytes_per_token
    proj_ms, kv_ms, kv_bytes = {}, {}, {}
    for rows in problem.mix(model.model_id):
        proj_ms[rows] = host_bytes / rate * 1000.0 * _table_factor(profile.desktop_rows_factor, rows)
        kv_bytes[rows] = kv_row * rows                       # every row reads its own context: no amortization
        kv_ms[rows] = kv_bytes[rows] / rate * 1000.0
    return _LayerCosts(MappingProxyType(proj_ms), MappingProxyType(kv_ms), host_bytes, MappingProxyType(kv_bytes))


def _shipping_and_recovery_j(problem: PlacementProblem, inputs: OpGroupInputs, model: ModelLayers, layer: int,
                             transport: str, rows: int, step_s: float) -> float:
    """Energy per step of keeping layer ``layer``'s KV on a device: prompt-KV shipping amortized per decoded
    row, plus the expected recovery (re-prefill this layer's share and re-ship its KV) per step."""
    geometry = inputs.geometry[model.model_id]
    stats = inputs.contexts[model.model_id]
    layer_geometry = geometry.layer(layer)
    power = problem.profile.desktop_power
    link = inputs.link_bytes_per_s[transport]
    prompt_kv = stats.prompt_tokens_per_decoded * layer_geometry.kv_bytes_per_token
    shipping = rows * prompt_kv / link * power.wait_w
    active_tokens = inputs.slots[model.model_id] * stats.mean_decode_context
    recover = (active_tokens * inputs.prefill_j_per_token / max(1, len(geometry.layers))
               + active_tokens * layer_geometry.kv_bytes_per_token / link * power.wait_w)
    return shipping + inputs.loss_rate_per_h / 3600.0 * step_s * recover


def _model_step_s(problem: PlacementProblem, inputs: OpGroupInputs, model: ModelLayers) -> float:
    """Desktop-only step time at one row (placement-invariant part + every CPU layer's host work)."""
    other = model.other_step_ms.get(1, 0.0)
    total = other
    for layer in model.cpu_layers:
        total += problem.profile.desktop_layer_ms(model.model_id, 1, model.bytes_for(model.host_format, layer)).value
    return total / 1000.0


def opgroup_options(problem: PlacementProblem, inputs: OpGroupInputs, model: ModelLayers, layer: int,
                    fraction: float, *, prune: bool = True) -> tuple[OptionCost, ...]:
    """Every way to run layer ``layer``'s FFN, projection unit and KV attention (index 0 = all on the CPU)."""
    profile = problem.profile
    power = profile.desktop_power
    mix = problem.mix(model.model_id)
    idle_w = sum(profile.idle_w(device.device_id) for device in problem.devices if device.available)
    geometry = inputs.geometry[model.model_id]
    host = _host_costs(problem, inputs, model, layer)
    own, nxt = _proj_unit(model, geometry, layer)
    model_bytes = {row.model_id: dict(row.layer_bytes) for row in problem.models}
    step_s = _model_step_s(problem, inputs, model)
    hot = (power.ffn_w + idle_w) / 1000.0         # J per host-streaming ms
    wait = (power.wait_w + idle_w) / 1000.0       # J per waiting ms

    def extra_host(rows, *, proj: bool, kv: bool) -> float:
        return (host.proj_ms[rows] if proj else 0.0) + (host.kv_ms[rows] if kv else 0.0)

    def add_host(option: OptionCost, *, proj: bool, kv: bool, variant: str = "", parts=()) -> OptionCost:
        """WS11's option plus the host still streaming the projection unit / reading the KV."""
        extra = {rows: extra_host(rows, proj=proj, kv=kv) for rows in mix}
        energy = sum(share * hot * extra[rows] for rows, share in mix.items())
        return replace(option, energy_j=option.energy_j + energy * (option.energy_j / option.expected_j
                                                                   if option.expected_j else 1.0),
                       expected_j=option.expected_j + energy,
                       ms_by_rows=MappingProxyType({rows: option.ms_by_rows[rows] + extra[rows] for rows in mix}),
                       variant=variant, parts=tuple(parts))

    base = _option_costs(problem, model, layer, fraction)
    options: list[OptionCost] = [add_host(base[0], proj=True, kv=True)]
    calls = {}      # device -> rows -> CallCost (full width) for building the other variants
    for option in base[1:]:
        options.append(add_host(option, proj=True, kv=True))

    # per-device costs: the device's FFN call at full width and its compute rate per stored byte
    for device in problem.devices:
        if not device.may_hold(model.model_id, layer):
            continue
        fmt = device.formats[model.model_id]
        size = model.bytes_for(fmt, layer)
        calls[device.device_id] = {rows: profile.call_cost(device.device_id, model.model_id, fmt, device.transport,
                                                           rows, size, model_bytes) for rows in mix}

    def device_rate(device_id, rows):
        """(ms per stored byte at ``rows``, J per compute-ms, overhead ms, estimates) of a device for this model."""
        device = problem.device(device_id)
        fmt = device.formats[model.model_id]
        cost = calls[device_id][rows]
        size = model.bytes_for(fmt, layer)
        per_byte = cost.compute_ms.value / size
        per_ms = cost.energy_j.value / cost.compute_ms.value if cost.compute_ms.value > 0 else 0.0
        overhead = max(0.0, cost.rpc_ms.value - cost.compute_ms.value)
        return per_byte, per_ms, overhead, (cost.rpc_ms, cost.compute_ms, cost.energy_j)

    def priors_of(device_id, estimates):
        return {f"{device_id}:{model.model_id}:{name}" for name, estimate in estimates if estimate.provenance == "prior"}

    reply_extra = 0
    if nxt:
        next_geometry = geometry.layer(layer + 1)
        reply_extra = (next_geometry.q_out + next_geometry.k_out + max(next_geometry.v_out, 0)) * 2   # f16 per row
    # a fused call receives the attention output (o_in wide) instead of the hidden state (n_embd wide)
    request_extra = max(0, geometry.layer(layer).o_in - geometry.n_embd) * 2

    # fused projections (and fused KV) on the device that owns this layer's FFN at full width
    if fraction == 1.0:
        for device in problem.devices:
            kernels = device_kernels(device)
            if device.device_id not in calls or "attn_proj:fused" not in kernels \
                    or not device.kernel_may_run("attn_proj:fused", model.model_id, layer):
                continue
            fmt = device.formats[model.model_id]
            if fmt not in geometry.layer(layer).proj_bytes:
                continue
            ffn_size = model.bytes_for(fmt, layer)
            proj_size = _unit_bytes(geometry, layer, fmt, own, nxt)
            link = inputs.link_bytes_per_s.get(device.transport)
            _require(link is not None, f"no link rate for transport {device.transport}")
            kv_size = int(math.ceil(inputs.slots[model.model_id]
                                    * _kv_context(inputs.contexts[model.model_id], geometry.layer(layer), peak=True)
                                    * geometry.layer(layer).kv_bytes_per_token))
            for with_kv in (False, True):
                if with_kv and ("attn_kv:fused" not in kernels
                                or not device.kernel_may_run("attn_kv:fused", model.model_id, layer)):
                    continue
                energy = 0.0
                ms_by_rows, busy_by_rows = {}, {}
                estimates, priors = [], set()
                for rows, share in mix.items():
                    per_byte, per_ms, overhead, est = device_rate(device.device_id, rows)
                    compute = calls[device.device_id][rows].compute_ms.value + proj_size * per_byte
                    # WS15: a measured fused call of this exact unit replaces the scaled compute and link time
                    measured = None if with_kv else profile.fused_call(
                        device.device_id, model.model_id, fmt + ("+next" if nxt else ""), device.transport, rows)
                    if measured is not None:
                        call_ms = measured.rpc_ms
                        step = wait * call_ms + per_ms * measured.compute_ms + hot * host.kv_ms[rows]
                        energy += share * step
                        ms_by_rows[rows] = call_ms + host.kv_ms[rows]
                        busy_by_rows[rows] = call_ms
                        estimates.extend(est)
                        priors |= priors_of(device.device_id, zip(("rpc_ms", "compute_ms", "energy_j"), est))
                        continue
                    if with_kv:
                        # KV bytes do not amortize over rows: each row reads its own context at the 1-row rate
                        per_byte_1, _, _, _ = device_rate(device.device_id, 1)
                        compute += host.kv_bytes_by_rows[rows] * per_byte_1
                    if with_kv:
                        # attention runs on the device: it receives this layer's Q/K/V (conservatively always:
                        # they are already there only when the previous unit was fused on the same device)
                        own_qkv = (geometry.layer(layer).q_out + geometry.layer(layer).k_out
                                   + max(geometry.layer(layer).v_out, 0)) * 2
                        transfer = rows * (reply_extra + own_qkv) / link * 1000.0
                    else:
                        transfer = rows * (reply_extra + request_extra) / link * 1000.0
                    call_ms = overhead + compute + transfer
                    step = wait * call_ms + per_ms * compute
                    if not with_kv:
                        step += hot * host.kv_ms[rows]
                    else:
                        step += _shipping_and_recovery_j(problem, inputs, model, layer, device.transport, rows, step_s)
                    energy += share * step
                    ms_by_rows[rows] = call_ms + (0.0 if with_kv else host.kv_ms[rows])
                    busy_by_rows[rows] = call_ms
                    estimates.extend(est)
                    priors |= priors_of(device.device_id, zip(("rpc_ms", "compute_ms", "energy_j"), est))
                parts = ((device.device_id, kv_size, 0.0, "kv"),) if with_kv else ()
                options.append(OptionCost(
                    device.device_id, energy * problem.penalty(estimates), MappingProxyType(ms_by_rows),
                    busy_by_rows[min(busy_by_rows)], ffn_size + proj_size,
                    _weakest("scaled", *(row.provenance for row in estimates)), tuple(sorted(priors)), energy,
                    "proj-fused+kv" if with_kv else "proj-fused", parts))

    # separate projection calls on any capable device; the FFN stays on the CPU or on its own device
    separate = [device for device in problem.devices if "attn_proj:separate" in device_kernels(device)
                and device.device_id in calls and device.formats[model.model_id] in geometry.layer(layer).proj_bytes]
    if separate:
        first_options = list(options[:1 + len(base) - 1])     # the CPU option and the WS11 FFN options
        for helper in separate:
            fmt = helper.formats[model.model_id]
            proj_size = _unit_bytes(geometry, layer, fmt, own, nxt)
            link = inputs.link_bytes_per_s.get(helper.transport)
            _require(link is not None, f"no link rate for transport {helper.transport}")
            sep_ms, sep_j, estimates, priors = {}, {}, [], set()
            for rows in mix:
                per_byte, per_ms, overhead, est = device_rate(helper.device_id, rows)
                o_bytes = geometry.layer(layer).bytes_of(fmt, own)
                calls_ms = overhead + o_bytes * per_byte + rows * (geometry.layer(layer).o_in + geometry.n_embd) * 2 / link * 1000.0
                compute = o_bytes * per_byte
                if nxt:
                    qkv_bytes = geometry.layer(layer + 1).bytes_of(fmt, nxt)
                    calls_ms += overhead + qkv_bytes * per_byte + rows * (geometry.n_embd + reply_extra // 2) * 2 / link * 1000.0
                    compute += qkv_bytes * per_byte
                sep_ms[rows] = calls_ms
                sep_j[rows] = wait * calls_ms + per_ms * compute
                estimates.extend(est)
                priors |= priors_of(helper.device_id, zip(("rpc_ms", "compute_ms", "energy_j"), est))
            for ffn in first_options:
                if ffn.variant:
                    continue
                remove = {rows: host.proj_ms[rows] for rows in mix}
                energy = sum(share * (sep_j[rows] - hot * remove[rows]) for rows, share in mix.items())
                ms = {rows: ffn.ms_by_rows[rows] - remove[rows] + sep_ms[rows] for rows in mix}
                busy = sep_ms[min(sep_ms)]
                expected = ffn.expected_j + energy
                factor = max(ffn.energy_j / ffn.expected_j if ffn.expected_j else 1.0, problem.penalty(estimates))
                if ffn.owner == helper.device_id:
                    option = replace(ffn, energy_j=expected * factor,
                                     expected_j=expected, ms_by_rows=MappingProxyType(ms),
                                     busy_ms=ffn.busy_ms + busy, bytes=ffn.bytes + proj_size,
                                     variant="proj-separate", priors=tuple(sorted(set(ffn.priors) | priors)),
                                     provenance=_weakest(ffn.provenance, "scaled"))
                else:
                    option = replace(ffn, energy_j=expected * factor,
                                     expected_j=expected, ms_by_rows=MappingProxyType(ms),
                                     variant="proj-separate", parts=((helper.device_id, proj_size, busy, "w"),),
                                     priors=tuple(sorted(set(ffn.priors) | priors)),
                                     provenance=_weakest(ffn.provenance, "scaled"))
                options.append(option)
    if fraction == 1.0 and inputs.parallel_split:
        options.extend(_parallel_options(problem, inputs, model, layer, calls, device_rate, priors_of, add_host))
    if not prune or all(not option.variant and not option.parts for option in options):
        return tuple(options)
    return tuple(_prune(problem, options))


def _executable_mode(problem: PlacementProblem) -> bool:
    """True for a problem narrowed by ``restrict_to_executable`` (per-kernel allowed layers are set)."""
    return any(device.kernel_allowed_layers is not None for device in problem.devices)


def parallel_shares(problem: PlacementProblem, model: ModelLayers, layer: int, pair: tuple[str, str],
                    calls, degradation: float) -> tuple[int, int] | None:
    """Quantum-aligned column shares (columns of ``pair[0]``, columns of ``pair[1]``) that balance the two
    devices' call times at one row (the dominant row count), or None when no aligned split exists."""
    quantum = model.column_quantum
    if quantum <= 0 or model.n_ff <= 0 or model.n_ff % quantum or model.n_ff // quantum < 2:
        return None
    rates = []
    for device_id in pair:
        device = problem.device(device_id)
        fmt = device.formats[model.model_id]
        size = model.bytes_for(fmt, layer)
        cost = calls[device_id][1]
        overhead = max(0.0, cost.rpc_ms.value - cost.compute_ms.value)
        rates.append((cost.compute_ms.value / model.n_ff * degradation, overhead))   # ms per column, overhead
    best = None
    for columns in range(quantum, model.n_ff, quantum):
        a = rates[0][1] + rates[0][0] * columns
        b = rates[1][1] + rates[1][0] * (model.n_ff - columns)
        key = (max(a, b), abs(2 * columns - model.n_ff), columns)
        if best is None or key < best[0]:
            best = (key, (columns, model.n_ff - columns))
    return None if best is None else best[1]


def _parallel_options(problem: PlacementProblem, inputs: OpGroupInputs, model: ModelLayers, layer: int,
                      calls, device_rate, priors_of, add_host) -> list[OptionCost]:
    """WS20: two helpers each own a disjoint column share of this layer's FFN and compute at the same time; the
    desktop waits for the slower share plus a synchronization penalty, each device pays its share's compute
    energy and stores its share. Needs ``ffn:parallel`` (both devices), a column quantum, and -- in an executable
    problem -- a MEASURED concurrency model and both shares qualified for the kernel on this layer."""
    profile = problem.profile
    power = profile.desktop_power
    mix = problem.mix(model.model_id)
    idle_w = sum(profile.idle_w(device.device_id) for device in problem.devices if device.available)
    wait = (power.wait_w + idle_w) / 1000.0
    executable = _executable_mode(problem)
    if executable and inputs.concurrency_provenance != "measured":
        return []
    penalty_ms = inputs.concurrency_penalty_ms if inputs.concurrency_penalty_ms is not None else 1.0
    penalty_estimate = Estimate(penalty_ms, inputs.concurrency_provenance, ("concurrency",), 0.0)
    capable = [device for device in problem.devices
               if device.device_id in calls and PARALLEL_KERNEL in device_kernels(device)
               and device.kernel_may_run(PARALLEL_KERNEL, model.model_id, layer)]
    options: list[OptionCost] = []
    for index, first in enumerate(capable):
        for second in capable[index + 1:]:
            pair = (first.device_id, second.device_id)
            shares = parallel_shares(problem, model, layer, pair, calls, inputs.concurrency_degradation)
            if shares is None:
                continue
            energy = 0.0
            ms_by_rows, busy_by_rows, bytes_by_device = {}, {}, {}
            estimates, priors = [penalty_estimate], set()
            if inputs.concurrency_provenance == "prior":
                priors.add("concurrency:penalty_ms")
            for rows, share in mix.items():
                step, pair_ms = 0.0, 0.0
                for device, columns in zip((first, second), shares):
                    fmt = device.formats[model.model_id]
                    size = model.bytes_for(fmt, layer)
                    per_byte, per_ms, overhead, est = device_rate(device.device_id, rows)
                    compute = size * (columns / model.n_ff) * per_byte * inputs.concurrency_degradation
                    link = inputs.link_bytes_per_s.get(device.transport)
                    _require(link is not None, f"no link rate for transport {device.transport}")
                    transfer = rows * 2 * model.n_embd * 2 / link * 1000.0     # activations in, partial out
                    call_ms = overhead + compute + transfer
                    pair_ms = max(pair_ms, call_ms)
                    step += per_ms * compute
                    busy_by_rows.setdefault(device.device_id, {})[rows] = call_ms
                    bytes_by_device[device.device_id] = int(math.ceil(size * columns / model.n_ff))
                    estimates.extend(est)
                    priors |= priors_of(device.device_id, zip(("rpc_ms", "compute_ms", "energy_j"), est))
                pair_ms += penalty_ms
                energy += share * (wait * pair_ms + step)
                ms_by_rows[rows] = pair_ms
            provenance = _weakest(*(row.provenance for row in estimates))
            if executable and provenance != "measured":
                continue
            second_busy = busy_by_rows[second.device_id]
            option = OptionCost(first.device_id, energy * problem.penalty(estimates), MappingProxyType(ms_by_rows),
                                busy_by_rows[first.device_id][min(mix)], bytes_by_device[first.device_id], provenance,
                                tuple(sorted(priors)), energy, PARALLEL_VARIANT,
                                ((second.device_id, bytes_by_device[second.device_id], second_busy[min(mix)], "w"),),
                                tuple(zip(pair, shares)))
            # the FFN-only split leaves the projection unit and KV reads on the host, like WS11's options
            option = add_host(option, proj=True, kv=True, variant=PARALLEL_VARIANT, parts=option.parts)
            options.append(replace(option, shares=tuple(zip(pair, shares))))
    return options


def _prune(problem: PlacementProblem, options: Sequence[OptionCost]) -> list[OptionCost]:
    """Drop options another option dominates (cheaper or equal energy, no slower at any row count, a subset
    of the devices with no more bytes or busy on each, and every route anchor of the dropped option kept).
    The CPU option (index 0) is never dropped, so the solvers' "index 0 = CPU" convention holds."""
    def usage(option):
        table = {}
        for device, size, busy, kind in _usages(option):
            old = table.get((device, kind), (0, 0.0))
            table[(device, kind)] = (old[0] + size, old[1] + busy)
        return table

    def anchors(option):
        return {device for device, _, _, _ in _usages(option) if problem.device(device).route_anchor}

    kept = [options[0]]
    tables = [usage(option) for option in options]
    for index, option in enumerate(options[1:], start=1):
        dominated = False
        for other_index, other in enumerate(options):
            if other_index == index:
                continue
            if other.energy_j > option.energy_j + 1e-12:
                continue
            if any(other.ms_by_rows[rows] > option.ms_by_rows[rows] + 1e-9 for rows in option.ms_by_rows):
                continue
            mine, theirs = tables[index], tables[other_index]
            if any(key not in mine or value[0] > mine[key][0] or value[1] > mine[key][1] + 1e-9
                   for key, value in theirs.items()):
                continue
            if not anchors(option) <= anchors(other):
                continue
            strictly = (other.energy_j < option.energy_j - 1e-12 or theirs != mine
                        or any(other.ms_by_rows[rows] < option.ms_by_rows[rows] - 1e-9 for rows in option.ms_by_rows))
            if strictly or other_index < index:
                dominated = True
                break
        if not dominated:
            kept.append(option)
    return kept


# --------------------------------------------------------------------------------------------------
# solving
# --------------------------------------------------------------------------------------------------


def _adjusted(problem: PlacementProblem, inputs: OpGroupInputs) -> tuple[PlacementProblem, dict[str, list[str]]]:
    """The problem with the host's projection and KV time moved from ``other_step_ms`` into the layer items."""
    notes: dict[str, list[str]] = {}
    models = []
    for model in problem.models:
        moved = {}
        for layer in model.cpu_layers:
            host = _host_costs(problem, inputs, model, layer)
            for rows in problem.mix(model.model_id):
                moved[rows] = moved.get(rows, 0.0) + host.proj_ms[rows] + host.kv_ms[rows]
        other = {}
        for rows, value in model.other_step_ms.items():
            left = value - moved.get(rows, 0.0)
            if left < 0:
                notes.setdefault(model.model_id, []).append(
                    f"rows {rows}: host projection + KV time {moved.get(rows, 0.0):.1f} ms exceeds the inventory's "
                    f"placement-invariant {value:.1f} ms; clamped to 0 (latency budget slightly loose)")
            other[rows] = max(0.0, left)
        models.append(replace(model, other_step_ms=MappingProxyType(other)))
    return replace(problem, models=tuple(models)), notes


@dataclass(frozen=True)
class OpGroupPlan:
    models: Mapping[str, Mapping[str, object]]
    solver: str
    gap_bound_j_per_step: float
    kernels: Mapping[str, tuple[str, ...]]
    device_usage: Mapping[str, Mapping[str, int]]
    device_busy_ms: Mapping[str, Mapping[str, float]]
    priors_used: tuple[str, ...]
    notes: Mapping[str, tuple[str, ...]]

    def energy_j_per_token(self, model_id: str) -> float:
        return float(self.models[model_id]["energy_j_per_token"])

    def to_json(self) -> dict[str, object]:
        return {"schema": OPGROUP_SCHEMA, "solver": self.solver,
                "gap_bound_j_per_step": round(self.gap_bound_j_per_step, 6),
                "kernels": {device: list(value) for device, value in sorted(self.kernels.items())},
                "device_usage_bytes": {device: dict(sorted(scopes.items())) for device, scopes in sorted(self.device_usage.items())},
                "device_busy_ms_per_step": {device: {model: round(ms, 4) for model, ms in sorted(rows.items())}
                                            for device, rows in sorted(self.device_busy_ms.items())},
                "priors_used": list(self.priors_used),
                "notes": {model: list(rows) for model, rows in sorted(self.notes.items())},
                "models": {name: dict(row) for name, row in sorted(self.models.items())}}


def _group_owners(items: Sequence[_Item], choice: Sequence[int]) -> dict[str, dict[str, list[int]]]:
    """group -> owner label -> layers (``ffn``, ``attn_proj`` (fused@dev / separate@dev), ``attn_kv``)."""
    groups: dict[str, dict[str, list[int]]] = {"ffn": {}, "attn_proj": {}, "attn_kv": {}}
    for item, index in zip(items, choice):
        option = item.options[index]
        if option.variant == PARALLEL_VARIANT:
            groups["ffn"].setdefault("+".join(device for device, _ in option.shares), []).append(item.layer)
        else:
            groups["ffn"].setdefault(option.owner, []).append(item.layer)
        if option.variant.startswith("proj-fused"):
            groups["attn_proj"].setdefault("fused@" + option.owner, []).append(item.layer)
        elif option.variant == "proj-separate":
            device = option.parts[0][0] if option.parts else option.owner
            groups["attn_proj"].setdefault("separate@" + device, []).append(item.layer)
        else:
            groups["attn_proj"].setdefault(DESKTOP_CPU, []).append(item.layer)
        if option.variant == "proj-fused+kv":
            groups["attn_kv"].setdefault("fused@" + option.owner, []).append(item.layer)
        else:
            groups["attn_kv"].setdefault(DESKTOP_CPU, []).append(item.layer)
    return groups


def _head_decision(problem: PlacementProblem, inputs: OpGroupInputs, model: ModelLayers, items, choice,
                   limits, chain_by_rows: Mapping[int, float]) -> dict[str, object]:
    """GPU head vs each capable device, with the remaining capacity and latency slack of the layer plan."""
    profile = problem.profile
    power = profile.desktop_power
    mix = problem.mix(model.model_id)
    geometry = inputs.geometry[model.model_id]
    idle_w = sum(profile.idle_w(device.device_id) for device in problem.devices if device.available)
    host_bytes = geometry.head_bytes.get(inputs.host_format)
    _require(host_bytes is not None, f"{model.model_id} has no {inputs.host_format} head size")
    gpu_ms = host_bytes / inputs.gpu_head_bytes_per_s * 1000.0
    gpu_w = power.wait_w + idle_w + (inputs.gpu_active_w - inputs.gpu_loaded_idle_w)
    gpu_j = gpu_w * gpu_ms / 1000.0
    tokens = sum(rows * share for rows, share in mix.items())
    rows_out = [{"owner": "desktop-gpu", "ms_rows1": round(gpu_ms, 4), "j_per_token": round(gpu_j / tokens, 6),
                 "feasible": True, "provenance": "prior (288 GB/s peak, 34 / 27.6 W GPU active / loaded idle)"}]
    best = ("desktop-gpu", gpu_j)
    # usage of the layer plan, to check the head's bytes against what is left
    used: dict[tuple[str, str], list[int]] = {}
    for item, index in zip(items, choice):
        for device_id, size, _, kind in _usages(item.options[index]):
            used.setdefault(_usage_scope(problem.device(device_id), item.model_id, kind), []).append(size)
    budget = limits.latency_budget
    model_bytes = {row.model_id: dict(row.layer_bytes) for row in problem.models}
    for device in problem.devices:
        if not device.serves(model.model_id) or "output_head" not in device_kernels(device):
            continue
        fmt = device.formats[model.model_id]
        size = geometry.head_bytes.get(fmt)
        if size is None:
            continue
        link = inputs.link_bytes_per_s[device.transport]
        reference = model.bytes_for(fmt, sorted(model.cpu_layers)[0])
        energy, ms_rows, feasible = 0.0, {}, True
        for rows, share in mix.items():
            cost = profile.call_cost(device.device_id, model.model_id, fmt, device.transport, rows, reference, model_bytes)
            per_byte = cost.compute_ms.value / reference
            per_ms = cost.energy_j.value / cost.compute_ms.value if cost.compute_ms.value > 0 else 0.0
            compute = size * per_byte
            call_ms = max(0.0, cost.rpc_ms.value - cost.compute_ms.value) + compute + rows * geometry.n_vocab * 2 / link * 1000.0
            energy += share * ((power.wait_w + idle_w) * call_ms / 1000.0 + per_ms * compute)
            ms_rows[rows] = call_ms
            limit = budget.get((model.model_id, rows))
            if limit is not None and chain_by_rows.get(rows, 0.0) + call_ms - gpu_ms > limit + 1e-9:
                feasible = False
        trial = {key: list(value) for key, value in used.items()}
        trial.setdefault(_usage_scope(device, model.model_id, "w"), []).append(size)
        if not _usage_ok(trial, limits):
            feasible = False
        served = {item.model_id for item, index in zip(items, choice)
                  if any(device_id == device.device_id for device_id, _, _, _ in _usages(item.options[index]))}
        cap = limits.capacity.get((device.device_id, "#models"))
        if cap is not None and len(served | {model.model_id}) > cap:
            feasible = False
        rows_out.append({"owner": device.device_id, "ms_rows1": round(ms_rows[min(ms_rows)], 4),
                         "j_per_token": round(energy / tokens, 6), "bytes": size, "feasible": feasible,
                         "provenance": "scaled (device FFN rate by bytes, measured overhead, link prior)"})
        if feasible and energy < best[1] - 1e-12:
            best = (device.device_id, energy)
    return {"owner": best[0], "j_per_token": round(best[1] / tokens, 6),
            "saving_vs_gpu_j_per_token": round((gpu_j - best[1]) / tokens, 6), "options": rows_out}


def solve_op_groups(problem: PlacementProblem, inputs: OpGroupInputs, *, method: str = "auto",
                    max_states: int = 200_000) -> OpGroupPlan:
    """Minimum predicted energy per token over every model's CPU layers with each device's kernels."""
    _require(method in ("auto", "exact", "greedy"), "unknown placement method " + method)
    adjusted, notes = _adjusted(problem, inputs)
    fraction_options = []
    for model in adjusted.models:
        candidates = adjusted.column_fractions.get(model.model_id) or (1.0,)
        fraction_options.append(tuple(sorted(set(candidates), reverse=True)))
    best = None
    for fractions in itertools.product(*fraction_options):
        by_model = {model.model_id: fraction for model, fraction in zip(adjusted.models, fractions)}
        result = _solve_fixed(adjusted, inputs, by_model, method, max_states)
        if result is None:
            continue
        total = sum(row["objective_j_per_token"] for row in result[0].values())
        if best is None or total < best[0] - 1e-12:
            best = (total, result)
    _require(best is not None, "no feasible operator-group placement")
    models, solver, gap, usage, busy, priors = best[1]
    return OpGroupPlan(MappingProxyType(models), solver, gap,
                       MappingProxyType({device.device_id: tuple(sorted(device_kernels(device))) for device in problem.devices}),
                       MappingProxyType(usage), MappingProxyType(busy), tuple(sorted(priors)),
                       MappingProxyType({model: tuple(rows) for model, rows in notes.items()}))


def _binding_scopes(problem: PlacementProblem, items: Sequence[_Item], limits) -> frozenset:
    """Capacity scopes that some placement could overfill: a scope without session packing whose largest
    possible usage (every item on its biggest option there; weights and KV of a base scope together) stays
    within capacity can never bind, so the exact DP need not distinguish states by it."""
    worst: dict[tuple[str, str], int] = {}
    for item in items:
        biggest: dict[tuple[str, str], int] = {}
        for option in item.options:
            here: dict[tuple[str, str], int] = {}
            for device_id, size, _, kind in _usages(option):
                base = (device_id, _scope(problem.device(device_id), item.model_id))
                here[base] = here.get(base, 0) + size
            for base, size in here.items():
                biggest[base] = max(biggest.get(base, 0), size)
        for base, size in biggest.items():
            worst[base] = worst.get(base, 0) + size
    binding = set()
    for base, size in worst.items():
        if base in limits.session_slots or size > limits.capacity.get(base, 0):
            binding.add(base)
            binding.add((base[0], base[1] + KV_SUFFIX))
    return frozenset(binding)


def _relaxed_key(problem, key, item: _Item, split, capped: frozenset[str], binding: frozenset) -> tuple:
    """Like the WS11 DP state, but only the dimensions a constraint can bind: resident sizes of binding
    scopes, busy of devices with a busy cap; the FFN chain is dropped (the latency bound is verified on the
    result), and so are scopes that cannot fill (they are checked on the result too)."""
    sizes, busy = dict(key[0]), dict(key[1])
    for n, option in zip(split, item.options):
        if n == 0:
            continue
        for device_id, size, busy_ms, kind in _usages(option):
            device = problem.device(device_id)
            scope = _usage_scope(device, item.model_id, kind)
            if scope not in binding:
                # keep a zero-size marker so route anchoring still sees the device in use
                sizes[(scope[0], "#used:" + item.model_id)] = (0,)
                if device_id in capped:
                    busy[device_id] = round(busy.get(device_id, 0.0) + n * busy_ms, 9)
                continue
            sizes[scope] = tuple(sorted(sizes.get(scope, ()) + (size,) * n))
            if device_id in capped:
                busy[device_id] = round(busy.get(device_id, 0.0) + n * busy_ms, 9)
    return (tuple(sorted(sizes.items())), tuple(sorted(busy.items())))


def _solve_relaxed_exact(problem: PlacementProblem, items: Sequence[_Item], limits, max_states: int):
    """Exact minimum of the problem without its latency bound (busy dimensions only for capped devices).

    Same DP as WS11 (classes of interchangeable layers, Pareto fronts per model on shared scopes). The result
    is returned only when it also meets the latency bound -- then it is optimal for the full problem (the
    relaxation's optimum is feasible); None means "relaxation too large or its optimum breaks latency"."""
    from .layer_placement import _pareto, _shared_usage

    capped = frozenset(limits.busy_cap)
    binding = _binding_scopes(problem, items, limits)
    ok_cache: dict[tuple, bool] = {}
    order = []
    for item in items:
        if item.model_id not in order:
            order.append(item.model_id)
    fronts = []
    for model_id in order:
        members_all = [index for index, item in enumerate(items) if item.model_id == model_id]
        sub = [items[index] for index in members_all]
        # one item at a time (states x options per item): with many options per layer this beats the WS11
        # class compositions, whose count grows as C(n + k - 1, k - 1); the key is the full binding usage,
        # so merging equal keys keeps the DP exact
        states: dict[tuple, tuple[float, tuple]] = {((), ()): (0.0, ())}
        for item in sub:
            next_states: dict[tuple, tuple[float, tuple]] = {}
            for option_index, option in enumerate(item.options):
                split = tuple(1 if index == option_index else 0 for index in range(len(item.options)))
                for key, (energy, picks) in states.items():
                    new_key = _relaxed_key(problem, key, item, split, capped, binding)
                    if new_key != key:
                        ok = ok_cache.get(new_key)
                        if ok is None:
                            ok = _usage_ok({scope: sizes for scope, sizes in new_key[0]
                                            if not scope[1].startswith("#used:")}, limits) and not any(
                                value > limits.busy_cap[device] + 1e-9 for device, value in new_key[1])
                            ok_cache[new_key] = ok
                        if not ok:
                            continue
                    value = energy + option.energy_j
                    old = next_states.get(new_key)
                    if old is None or value < old[0] - 1e-12:
                        next_states[new_key] = (value, picks + (option_index,))
                    if len(next_states) > max_states:
                        return None
            states = next_states
            if not states:
                return None
        front: dict[tuple, tuple[float, tuple]] = {}
        for key, (energy, picks) in states.items():
            if problem.anchoring and not problem.anchored(scope[0] for scope, sizes in key[0] if sizes):
                continue
            real = tuple((scope, sizes) for scope, sizes in key[0] if not scope[1].startswith("#used:"))
            usage = _shared_usage((real, key[1], ()), limits)
            # devices this model uses (also through non-binding scopes) still count against max_models
            used = sorted({scope[0] for scope, sizes in key[0] if sizes})
            usage = tuple(sorted(set(usage) | {((device, "#models"), (1,)) for device in used
                                               if (device, "#models") in limits.capacity}))
            old = front.get(usage)
            if old is None or energy < old[0] - 1e-12:
                front[usage] = (energy, picks)
        if not front:
            return None
        fronts.append((members_all, sub, None, _pareto(front, limits)))
    combined: dict[tuple, tuple[float, tuple]] = {(): (0.0, ())}
    for _, _, _, front in fronts:
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
    for (members_all, _, _, _), picks in zip(fronts, chosen):
        for member, option_index in zip(members_all, picks):
            choice[member] = option_index
    return choice if _feasible(problem, items, choice, limits) else None


MB = 1 << 20


def _bound_dp(problem: PlacementProblem, items: Sequence[_Item], limits, max_states: int = 2_000_000):
    """Exact optimum of a RELAXATION: session packing -> total capacity, bytes rounded DOWN to MiB, latency
    dropped, busy caps kept (rounded down to 0.01 ms). Every feasible placement is feasible here, so the value
    is a lower bound on the objective; returns (choice, bound) or None when even this exceeds ``max_states``."""
    capped = frozenset(limits.busy_cap)
    binding = _binding_scopes(problem, items, limits)
    order = []
    for item in items:
        if item.model_id not in order:
            order.append(item.model_id)

    def capacity(scope):
        return limits.capacity.get(scope, 0) // MB

    fronts = []
    for model_id in order:
        members_all = [index for index, item in enumerate(items) if item.model_id == model_id]
        states: dict[tuple, tuple[float, tuple]] = {((), (), ()): (0.0, ())}
        for index in members_all:
            item = items[index]
            next_states: dict[tuple, tuple[float, tuple]] = {}
            for option_index, option in enumerate(item.options):
                add_mb: dict[tuple[str, str], int] = {}
                add_busy: dict[str, int] = {}
                used = set()
                for device_id, size, busy_ms, kind in _usages(option):
                    device = problem.device(device_id)
                    base = (device_id, _scope(device, model_id))
                    used.add(device_id)
                    if base in binding:
                        add_mb[base] = add_mb.get(base, 0) + size // MB
                    if device_id in capped:
                        add_busy[device_id] = add_busy.get(device_id, 0) + int(busy_ms * 100)
                for key, (energy, picks) in states.items():
                    mb, busy, devices = dict(key[0]), dict(key[1]), set(key[2])
                    ok = True
                    for scope, value in add_mb.items():
                        mb[scope] = mb.get(scope, 0) + value
                        if mb[scope] > capacity(scope):
                            ok = False
                    for device_id, value in add_busy.items():
                        busy[device_id] = busy.get(device_id, 0) + value
                        if busy[device_id] > limits.busy_cap[device_id] * 100 + 1:
                            ok = False
                    if not ok:
                        continue
                    devices |= used
                    new_key = (tuple(sorted(mb.items())), tuple(sorted(busy.items())), tuple(sorted(devices)))
                    value = energy + option.energy_j
                    old = next_states.get(new_key)
                    if old is None or value < old[0] - 1e-12:
                        next_states[new_key] = (value, picks + (option_index,))
                    if len(next_states) > max_states:
                        return None
            states = next_states
        front: dict[tuple, tuple[float, tuple]] = {}
        for key, (energy, picks) in states.items():
            if problem.anchoring and not problem.anchored(key[2]):
                continue
            # shared with other models: co-resident binding scopes and one model unit per max_models device
            shared = tuple(sorted([(scope, value) for scope, value in key[0] if scope[1] == "*"]
                                  + [((device, "#models"), 1) for device in key[2] if (device, "#models") in limits.capacity]))
            old = front.get(shared)
            if old is None or energy < old[0] - 1e-12:
                front[shared] = (energy, picks)
        if not front:
            return None
        fronts.append((members_all, front))
    combined: dict[tuple, tuple[float, tuple]] = {(): (0.0, ())}
    for _, front in fronts:
        next_combined: dict[tuple, tuple[float, tuple]] = {}
        for usage, (energy, chosen) in combined.items():
            for model_usage, (model_energy, picks) in front.items():
                merged = dict(usage)
                ok = True
                for scope, value in model_usage:
                    merged[scope] = merged.get(scope, 0) + value
                    limit = limits.capacity.get(scope, 0) if scope[1] == "#models" else capacity(scope)
                    if merged[scope] > limit:
                        ok = False
                if not ok:
                    continue
                key = tuple(sorted(merged.items()))
                value = energy + model_energy
                old = next_combined.get(key)
                if old is None or value < old[0] - 1e-12:
                    next_combined[key] = (value, chosen + (picks,))
                if len(next_combined) > max_states:
                    return None
        combined = next_combined
        if not combined:
            return None
    bound, chosen = min(combined.values(), key=lambda row: row[0])
    choice = [0] * len(items)
    for (members_all, _), picks in zip(fronts, chosen):
        for member, option_index in zip(members_all, picks):
            choice[member] = option_index
    return choice, bound


def _violation(problem: PlacementProblem, items: Sequence[_Item], choice: Sequence[int], limits) -> float:
    """How far a placement is from feasible: bytes over capacity (FFD leftovers of session scopes), busy and
    latency overruns (ms, weighted like 1 MiB each). 0 = feasible (the same checks as ``_feasible``)."""
    from .layer_placement import _anchors_ok
    if not _anchors_ok(problem, items, choice):
        return math.inf
    used: dict[tuple[str, str], list[int]] = {}
    busy: dict[tuple[str, str], float] = {}
    chain: dict[tuple[str, int], float] = {}
    for item, index in zip(items, choice):
        option = item.options[index]
        for rows, ms in option.ms_by_rows.items():
            chain[(item.model_id, rows)] = chain.get((item.model_id, rows), 0.0) + ms
        for device_id, size, busy_ms, kind in _usages(option):
            device = problem.device(device_id)
            used.setdefault(_usage_scope(device, item.model_id, kind), []).append(size)
            busy[(device_id, item.model_id)] = busy.get((device_id, item.model_id), 0.0) + busy_ms
    total = 0.0
    for scope, sizes in used.items():
        if scope[1].endswith(KV_SUFFIX):
            continue
        kv = used.get((scope[0], scope[1] + KV_SUFFIX), [])
        total += max(0, sum(sizes) + sum(kv) - limits.capacity.get(scope, 0)) / MB
        if scope in limits.session_slots:
            sessions, limit = limits.session_slots[scope]
            bins = [0] * sessions
            for size in sorted(sizes, reverse=True):
                for position in range(sessions):
                    if bins[position] + size <= limit:
                        bins[position] += size
                        break
                else:
                    total += size / MB
    models: dict[str, set] = {}
    for device_id, model_id in busy:
        models.setdefault(device_id, set()).add(model_id)
    for device_id, served in models.items():
        cap = limits.capacity.get((device_id, "#models"))
        if cap is not None and len(served) > cap:
            total += 1e6 * (len(served) - cap)
    for (device_id, _), value in busy.items():
        if device_id in limits.busy_cap:
            total += max(0.0, value - limits.busy_cap[device_id])
    for key, budget in limits.latency_budget.items():
        total += max(0.0, chain.get(key, 0.0) - budget)
    return total


def _improve(problem: PlacementProblem, items: Sequence[_Item], choice: list[int], limits) -> list[int]:
    """Single-item moves and pairwise identity swaps that lower the objective and stay feasible."""
    for _ in range(50):
        improved = False
        for index, item in enumerate(items):
            here = item.options[choice[index]].energy_j
            for option_index, option in enumerate(item.options):
                if option.energy_j < here - 1e-12:
                    old = choice[index]
                    choice[index] = option_index
                    if _feasible(problem, items, choice, limits):
                        here, improved = option.energy_j, True
                    else:
                        choice[index] = old
        for first, second in itertools.combinations(range(len(items)), 2):
            a, b = items[first], items[second]
            if a.model_id != b.model_id:
                continue
            for index_a, option_a in enumerate(a.options):
                for index_b, option_b in enumerate(b.options):
                    delta = (option_a.energy_j + option_b.energy_j - a.options[choice[first]].energy_j
                             - b.options[choice[second]].energy_j)
                    if delta >= -1e-12:
                        continue
                    old = (choice[first], choice[second])
                    choice[first], choice[second] = index_a, index_b
                    if _feasible(problem, items, choice, limits):
                        improved = True
                        break
                    choice[first], choice[second] = old
                else:
                    continue
                break
        if not improved:
            break
    return choice


def _repair(problem: PlacementProblem, items: Sequence[_Item], choice: Sequence[int], limits) -> list[int] | None:
    """From an infeasible placement (the relaxation's optimum), apply the single-item change with the best
    violation reduction per joule until feasible, then improve locally. None if it gets stuck."""
    choice = list(choice)
    violation = _violation(problem, items, choice, limits)
    for _ in range(4 * len(items) + 10):
        if violation <= 1e-9 and _feasible(problem, items, choice, limits):
            return _improve(problem, items, choice, limits)
        best = None
        for index, item in enumerate(items):
            current = item.options[choice[index]].energy_j
            for option_index, option in enumerate(item.options):
                if option_index == choice[index]:
                    continue
                old = choice[index]
                choice[index] = option_index
                after = _violation(problem, items, choice, limits)
                choice[index] = old
                reduction = violation - after
                if reduction <= 1e-9:
                    continue
                score = reduction / (max(0.0, option.energy_j - current) + 1e-6)
                if best is None or score > best[0]:
                    best = (score, index, option_index, after)
        if best is None:
            return None
        _, index, option_index, violation = best
        choice[index] = option_index
    return None


def _subset_candidate(problem: PlacementProblem, items: Sequence[_Item], keep, max_states: int) -> list[int] | None:
    """Optimum (or best effort) over the options ``keep`` accepts (the CPU option always), mapped back to the
    full items: a smaller kernel set's plan is a feasible start for a bigger one, so heuristics never lose
    to it."""
    sub_items, maps = [], []
    for item in items:
        indices = [index for index, option in enumerate(item.options) if index == 0 or keep(option)]
        sub_items.append(_Item(item.model_id, item.layer, tuple(item.options[index] for index in indices)))
        maps.append(indices)
    if all(len(indices) == len(item.options) for indices, item in zip(maps, items)):
        return None
    limits = _constraints(problem, sub_items)
    choice = None
    bound = _bound_dp(problem, sub_items, limits)
    if bound is not None and _feasible(problem, sub_items, bound[0], limits):
        choice = bound[0]
    if choice is None:
        choice = _solve_relaxed_exact(problem, sub_items, limits, max_states)
    if choice is None:
        choice = _solve_greedy(problem, sub_items, limits)
    return [mapping[index] for mapping, index in zip(maps, choice)]


def _solve_fixed(problem: PlacementProblem, inputs: OpGroupInputs, fractions: Mapping[str, float], method: str,
                 max_states: int):
    items: list[_Item] = []
    for model in problem.models:
        items.extend(_Item(model.model_id, layer, opgroup_options(problem, inputs, model, layer,
                                                                  fractions[model.model_id]))
                     for layer in sorted(model.cpu_layers))
    limits = _constraints(problem, items)
    cpu_choice = [0] * len(items)
    if not _feasible(problem, items, cpu_choice, limits):
        return None
    choice, solver, lower = None, "greedy", None
    if method in ("auto", "exact"):
        bound = _bound_dp(problem, items, limits)
        if bound is not None:
            relaxed_choice, lower = bound
            if _feasible(problem, items, relaxed_choice, limits):
                # the relaxation's optimum meets every true limit: optimal for the full problem
                choice, solver = relaxed_choice, "exact (relaxation optimum is feasible)"
        if choice is None:
            # the exact item DP on the true session multisets; a small budget in auto mode (it fails slowly)
            choice = _solve_relaxed_exact(problem, items, limits, max_states if method == "exact" else 30_000)
            if choice is not None:
                solver = "exact (relaxed latency, verified)"
        if choice is None and method == "exact":
            choice = _solve_exact(problem, items, limits, max_states)
            if choice is not None:
                solver = "exact"
            else:
                raise LayerPlacementError("exact operator-group state space exceeds max_states")
    if choice is None:
        candidates = [("greedy", _improve(problem, items, list(_solve_greedy(problem, items, limits)), limits))]
        if lower is not None:
            repaired = _repair(problem, items, bound[0], limits)
            if repaired is not None:
                candidates.append(("repair", repaired))
        for label, keep in (("ffn-only start", lambda option: not option.variant and not option.parts),
                            ("ffn + fused start", lambda option: not option.parts or option.variant == "proj-fused+kv"
                             and all(kind == "kv" for _, _, _, kind in option.parts))):
            start = _subset_candidate(problem, items, keep, 30_000)
            if start is not None and _feasible(problem, items, start, limits):
                candidates.append((label, _improve(problem, items, start, limits)))
        solver, choice = min(candidates, key=lambda row: _energy(items, row[1]))
    choice = _choose_layers(problem, items, choice)
    _require(_feasible(problem, items, choice, limits), "internal: chosen operator groups break a limit")
    choice = _prefer_current(problem, items, choice, limits)
    saving = _energy(items, cpu_choice) - _energy(items, choice)
    if solver.startswith("exact"):
        gap = 0.0
    elif lower is not None:
        solver = solver + " (certified by the relaxation bound)"
        gap = max(0.0, _energy(items, choice) - lower)
    else:
        gap = max(0.0, _saving_upper_bound(problem, items, limits) - saving)
    models, usage, busy, priors = {}, {}, {}, set()
    for model in problem.models:
        members = [index for index, item in enumerate(items) if item.model_id == model.model_id]
        sub_items = [items[index] for index in members]
        sub_choice = [choice[index] for index in members]
        tokens = sum(rows * share for rows, share in problem.mix(model.model_id).items())
        energy = sum(item.options[index].expected_j for item, index in zip(sub_items, sub_choice))
        objective = sum(item.options[index].energy_j for item, index in zip(sub_items, sub_choice))
        cpu_energy = sum(item.cpu.expected_j for item in sub_items)
        chain, cpu_chain = {}, {}
        for item, index in zip(sub_items, sub_choice):
            option = item.options[index]
            priors.update(option.priors)
            for rows, ms in option.ms_by_rows.items():
                chain[rows] = chain.get(rows, 0.0) + ms
            for rows, ms in item.cpu.ms_by_rows.items():
                cpu_chain[rows] = cpu_chain.get(rows, 0.0) + ms
            for device_id, size, busy_ms, kind in _usages(option):
                scope = _usage_scope(problem.device(device_id), model.model_id, kind)[1]
                usage.setdefault(device_id, {})
                usage[device_id][scope] = usage[device_id].get(scope, 0) + size
                busy.setdefault(device_id, {})
                busy[device_id][model.model_id] = busy[device_id].get(model.model_id, 0.0) + busy_ms
        groups = _group_owners(sub_items, sub_choice)
        head = _head_decision(problem, inputs, model, items, choice, limits, chain)
        head_saving = head["saving_vs_gpu_j_per_token"]
        models[model.model_id] = {
            "model_id": model.model_id,
            "column_fraction": fractions[model.model_id],
            "layers": [{"layer": item.layer, "ffn": item.options[index].owner,
                        "variant": item.options[index].variant or "ffn",
                        **({"parts": [list(part[:1]) + [part[1], round(part[2], 4), part[3]]
                                      for part in item.options[index].parts]} if item.options[index].parts else {}),
                        **({"shares": {device: columns for device, columns in item.options[index].shares}}
                           if item.options[index].shares else {})}
                       for item, index in zip(sub_items, sub_choice)],
            "groups": {group: {owner: layer_spec(layers) for owner, layers in sorted(owners.items())}
                       for group, owners in groups.items()},
            "output_head": head,
            "energy_j_per_token": round(energy / tokens - head_saving, 6),
            "objective_j_per_token": round(objective / tokens - head_saving, 6),
            "cpu_only_energy_j_per_token": round(cpu_energy / tokens, 6),
            "saving_vs_cpu_j_per_token": round((cpu_energy - energy) / tokens + head_saving, 6),
            "chain_ms_by_rows": {str(k): round(v, 4) for k, v in sorted(chain.items())},
            "cpu_only_chain_ms_by_rows": {str(k): round(v, 4) for k, v in sorted(cpu_chain.items())},
            "tokens_per_step": round(tokens, 6),
            "context": inputs.contexts[model.model_id].to_json(),
        }
    return models, solver, gap, usage, busy, priors


# --------------------------------------------------------------------------------------------------
# what pays: the decision report
# --------------------------------------------------------------------------------------------------


KERNEL_SETS = MappingProxyType({
    "ffn (today)": FFN_ONLY,
    "+ attn_proj fused": FFN_ONLY | {"attn_proj:fused"},
    "+ attn_proj separate": FFN_ONLY | {"attn_proj:separate"},
    "+ attn_proj fused + separate": FFN_ONLY | {"attn_proj:fused", "attn_proj:separate"},
    "+ attn_proj fused + attn_kv fused": FFN_ONLY | {"attn_proj:fused", "attn_kv:fused"},
    "+ output_head": FFN_ONLY | {"output_head"},
    "all kernels": ALL_KERNELS,
})


def _counts(plan: OpGroupPlan, model_id: str) -> dict[str, int]:
    row = plan.models[model_id]
    layers = row["layers"]
    parallel = sum(1 for layer in layers if layer["variant"] == PARALLEL_VARIANT)
    return {"ffn": sum(1 for layer in layers if layer["ffn"] != DESKTOP_CPU),
            # WS20: present only when a plan splits layers, so FFN-only / WS14 report JSON stays byte-identical
            **({"ffn_parallel": parallel} if parallel else {}),
            "attn_proj": sum(1 for layer in layers if layer["variant"].startswith("proj")),
            "attn_kv": sum(1 for layer in layers if layer["variant"] == "proj-fused+kv"),
            "output_head": int(row["output_head"]["owner"] != "desktop-gpu")}


def _unit_pays(problem: PlacementProblem, inputs: OpGroupInputs, kernels: frozenset[str]) -> dict[str, dict[str, bool]]:
    """Per model and group: does any single layer's option with that group beat the same layer's best option
    without it, ignoring capacity, sessions, latency and busy caps? (the group pays "per unit")."""
    adjusted, _ = _adjusted(with_kernels(problem, kernels), inputs)
    result = {}
    for model in adjusted.models:
        pays = {"attn_proj": False, "attn_kv": False, "ffn_parallel": False}
        for layer in model.cpu_layers:
            options = opgroup_options(adjusted, inputs, model, layer, 1.0)
            without_parallel = min(option.expected_j for option in options if option.variant != PARALLEL_VARIANT)
            with_parallel = [option.expected_j for option in options if option.variant == PARALLEL_VARIANT]
            if with_parallel and min(with_parallel) < without_parallel - 1e-9:
                pays["ffn_parallel"] = True
            without_proj = min(option.expected_j for option in options if not option.variant.startswith("proj"))
            with_proj = [option.expected_j for option in options if option.variant.startswith("proj")]
            if with_proj and min(with_proj) < without_proj - 1e-9:
                pays["attn_proj"] = True
            without_kv = min(option.expected_j for option in options if option.variant != "proj-fused+kv")
            with_kv = [option.expected_j for option in options if option.variant == "proj-fused+kv"]
            if with_kv and min(with_kv) < without_kv - 1e-9:
                pays["attn_kv"] = True
        result[model.model_id] = pays
    return result


VERDICT_SETS = ("ffn (today)", "+ attn_proj fused", "+ attn_proj separate", "+ attn_proj fused + attn_kv fused",
                "+ output_head")
# WS20: reported only when a caller asks for them (``kernel_sets`` naming one of these); the default report and
# its verdicts are byte-identical to WS14's
PARALLEL_SETS = MappingProxyType({
    "+ ffn parallel": FFN_ONLY | {PARALLEL_KERNEL},
    "+ ffn parallel + attn_proj fused": FFN_ONLY | {PARALLEL_KERNEL, "attn_proj:fused"},
})
ALL_SETS = MappingProxyType({**KERNEL_SETS, **PARALLEL_SETS})


def decision_table(problem: PlacementProblem, inputs: OpGroupInputs, *, tokens_per_run: Mapping[str, float],
                   method: str = "auto", fleet_kj: float | None = None,
                   kernel_sets: Sequence[str] = tuple(KERNEL_SETS), executable: bool = True) -> dict[str, object]:
    """Ideal plans for each kernel set (any layer, hypothetical kernels on every device) against the FFN-only
    ideal plan; the executable plan with the devices' qualified kernels and stored + qualified layers."""
    names = tuple(dict.fromkeys(("ffn (today)",) + tuple(kernel_sets) + VERDICT_SETS))
    plans = {name: solve_op_groups(with_kernels(problem, ALL_SETS[name]), inputs, method=method) for name in names}
    executable = solve_op_groups(restrict_to_executable(problem), inputs, method=method) if executable else None
    base = plans["ffn (today)"]
    rows = []
    for name, plan in plans.items():
        total_kj = 0.0
        cells = {}
        for model_id in sorted(plan.models):
            gain = base.energy_j_per_token(model_id) - plan.energy_j_per_token(model_id)
            kj = gain * tokens_per_run.get(model_id, 0.0) / 1000.0
            total_kj += kj
            cells[model_id] = {"gain_vs_ffn_only_j_per_token": round(gain, 4), "gain_kj_per_run": round(kj, 3),
                               "offloaded_units": _counts(plan, model_id),
                               "groups": plan.models[model_id]["groups"],
                               "output_head": plan.models[model_id]["output_head"]["owner"],
                               "chain_ms_rows1": plan.models[model_id]["chain_ms_by_rows"].get("1")}
        rows.append({"kernels": name, "models": cells, "gain_kj_per_run": round(total_kj, 3),
                     **({"gain_pct_of_fleet": round(100 * total_kj / fleet_kj, 2)} if fleet_kj else {}),
                     "solver": plan.solver, "gap_bound_j_per_step": round(plan.gap_bound_j_per_step, 6),
                     "device_usage_bytes": {d: dict(s) for d, s in plan.device_usage.items()},
                     "priors_used": len(plan.priors_used)})
    unit_names = ["+ attn_proj fused", "+ attn_proj separate", "+ attn_proj fused + attn_kv fused"]
    groups_sets = [("attn_proj fused", "+ attn_proj fused"), ("attn_proj separate", "+ attn_proj separate"),
                   ("attn_kv fused", "+ attn_proj fused + attn_kv fused"), ("output_head", "+ output_head")]
    if "+ ffn parallel" in names:
        unit_names.append("+ ffn parallel")
        groups_sets.append(("ffn_parallel", "+ ffn parallel"))
    per_unit = {name: _unit_pays(problem, inputs, ALL_SETS[name]) for name in unit_names}
    verdicts = {}
    for model_id in sorted(base.models):
        verdict = {}
        for group, kernel_set in groups_sets:
            plan = plans[kernel_set]
            gain = base.energy_j_per_token(model_id) - plan.energy_j_per_token(model_id)
            key = group.split()[0]
            units = _counts(plan, model_id).get(key, 0)
            if group == "attn_kv fused":
                # gain of KV on top of fused projections
                gain = plans["+ attn_proj fused"].energy_j_per_token(model_id) - plan.energy_j_per_token(model_id)
            if group == "attn_proj separate":
                units = sum(1 for layer in plan.models[model_id]["layers"] if layer["variant"] == "proj-separate")
            pays_unit = per_unit[kernel_set][model_id].get(key, False) if kernel_set in per_unit else False
            if units and gain > 1e-3:
                text = "offload"
            elif pays_unit and group != "output_head":
                text = "pays per unit, but blocked by capacity / latency / busy limits"
            else:
                text = "keep on desktop"
            missing = PARALLEL_KERNEL if group == "ffn_parallel" else key + (":" + group.split()[1] if " " in group else "")
            verdict[group] = {"verdict": text, "gain_j_per_token": round(gain, 4),
                              "gain_kj_per_run": round(gain * tokens_per_run.get(model_id, 0.0) / 1000.0, 3),
                              "units_offloaded": units, "missing_kernel": missing}
        verdicts[model_id] = verdict
    return {"plans": {name: plan.to_json() for name, plan in plans.items()},
            "executable": None if executable is None else executable.to_json(), "summary": rows,
            "verdicts": verdicts, "tokens_per_run": dict(tokens_per_run)}


def context_sweep(problem: PlacementProblem, inputs: OpGroupInputs, contexts: Sequence[int], *,
                  tokens_per_run: Mapping[str, float], method: str = "auto") -> list[dict[str, object]]:
    """attn_kv (on top of fused projections) at a constant context per request: where does it start to pay?"""
    rows = []
    for context in contexts:
        stats = {model: ContextStats.constant(context, inputs.contexts[model].prompt_tokens_per_decoded)
                 for model in inputs.contexts}
        swept = replace(inputs, contexts=MappingProxyType(stats))
        with_proj = solve_op_groups(with_kernels(problem, FFN_ONLY | {"attn_proj:fused"}), swept, method=method)
        with_kv = solve_op_groups(with_kernels(problem, FFN_ONLY | {"attn_proj:fused", "attn_kv:fused"}), swept,
                                  method=method)
        per_unit = _unit_pays(problem, swept, FFN_ONLY | {"attn_proj:fused", "attn_kv:fused"})
        cells = {}
        for model_id in sorted(with_kv.models):
            geometry = inputs.geometry[model_id]
            model = next(model for model in problem.models if model.model_id == model_id)
            kv_per_layer = [inputs.slots[model_id] * _kv_context(stats[model_id], geometry.layer(layer), peak=True)
                            * geometry.layer(layer).kv_bytes_per_token for layer in model.cpu_layers]
            gain = with_proj.energy_j_per_token(model_id) - with_kv.energy_j_per_token(model_id)
            units = _counts(with_kv, model_id)["attn_kv"]
            cells[model_id] = {
                "kv_layers_offloaded": units, "gain_j_per_token": round(gain, 4),
                "gain_kj_per_run": round(gain * tokens_per_run.get(model_id, 0.0) / 1000.0, 3),
                "pays_per_unit": per_unit[model_id]["attn_kv"],
                "kv_resident_bytes_per_layer_max": int(max(kv_per_layer)),
                "host_kv_ms_per_token_rows1": round(sum(
                    _host_costs(problem, swept, model, layer).kv_ms[1] for layer in model.cpu_layers), 3),
                "energy_ffn_proj_j_per_token": with_proj.energy_j_per_token(model_id),
                "energy_with_kv_j_per_token": with_kv.energy_j_per_token(model_id),
                "groups": with_kv.models[model_id]["groups"],
            }
        rows.append({"context": context, "models": cells})
    return rows


def scaled_profile(profile: LayerPlacementProfile, *, phone_energy: float = 1.0, desktop_wait: float = 1.0,
                   phone_bandwidth: float = 1.0) -> LayerPlacementProfile:
    """Profile with every helper's energy per call x ``phone_energy``, compute time / ``phone_bandwidth``,
    and the desktop's waiting power x ``desktop_wait`` (sensitivity analysis)."""
    copy = profile.copy()
    copy.desktop_power = replace(copy.desktop_power, wait_w=copy.desktop_power.wait_w * desktop_wait)
    for key, stats in list(copy.calls.items()):
        compute = stats.compute_ms / phone_bandwidth
        copy.calls[key] = replace(stats, compute_ms=compute, rpc_ms=stats.rpc_ms - stats.compute_ms + compute)
    for key, estimate in list(copy.energy_per_call.items()):
        copy.energy_per_call[key] = replace(estimate, value=estimate.value * phone_energy / phone_bandwidth)
    for device, priors in list(copy.device_priors.items()):
        copy.device_priors[device] = replace(
            priors, bytes_per_s=MappingProxyType({k: v * phone_bandwidth for k, v in priors.bytes_per_s.items()}),
            active_marginal_w=priors.active_marginal_w * phone_energy)
    return copy


def sensitivity(problem: PlacementProblem, inputs: OpGroupInputs, *, tokens_per_run: Mapping[str, float],
                factors: Sequence[float] = (0.75, 1.25), method: str = "auto") -> list[dict[str, object]]:
    """Verdict per group with one input moved: phone J/ms, desktop waiting power, phone bandwidth."""
    rows = []
    for parameter in ("phone_energy", "desktop_wait", "phone_bandwidth"):
        for factor in factors:
            moved = replace(problem, profile=scaled_profile(problem.profile, **{parameter: factor}))
            table = decision_table(moved, inputs, tokens_per_run=tokens_per_run, method=method,
                                   kernel_sets=VERDICT_SETS, executable=False)
            rows.append({"parameter": parameter, "factor": factor,
                         "verdicts": {model: {group: {"verdict": cell["verdict"], "gain_j_per_token": cell["gain_j_per_token"]}
                                              for group, cell in groups.items()}
                                      for model, groups in table["verdicts"].items()},
                         "ffn_only_vs_cpu_j_per_token": {
                             model: round(table["plans"]["ffn (today)"]["models"][model]["saving_vs_cpu_j_per_token"], 4)
                             for model in table["plans"]["ffn (today)"]["models"]}})
    return rows


def dumps(value: object) -> str:
    return json.dumps(value, indent=1, sort_keys=True)
