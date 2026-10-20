"""JSON form of a layer-placement inventory (models, helper devices, current ownership) and of
``S41SERVERFFNSHAPE`` server summaries (WS11). No file I/O here: callers pass parsed JSON / lines."""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Iterable, Mapping

from .layer_placement import (
    DESKTOP_CPU,
    HelperDevice,
    LayerPlacementError,
    ModelLayers,
    PlacementProblem,
    LayerPlacementProfile,
    layer_mask,
    layer_spec,
    mask_layers,
)


LAYER_PLACEMENT_INVENTORY_SCHEMA = "research-scheduler-layer-placement-inventory-v1"
SHAPE_PREFIX = "S41SERVERFFNSHAPE "
FUSED_SHAPE_PREFIX = "S41SERVERFFNFUSEDSHAPE "


def parse_layers(value: object) -> tuple[int, ...]:
    """``"0-5,8"`` / ``""`` / a list of ints / a hex mask string ``"0x..."``."""
    if isinstance(value, list):
        layers = tuple(sorted({int(item) for item in value}))
    elif isinstance(value, str) and value.startswith("0x"):
        layers = mask_layers(int(value, 16))
    elif isinstance(value, str):
        found: set[int] = set()
        for item in filter(None, value.split(",")):
            first, _, last = item.partition("-")
            begin, end = int(first), int(last) if last else int(first)
            if end < begin:
                raise LayerPlacementError("layer range " + item + " is reversed")
            found.update(range(begin, end + 1))
        layers = tuple(sorted(found))
    else:
        raise LayerPlacementError("layers must be a spec string or a list")
    layer_mask(layers)
    return layers


def _masks(value: Mapping[str, object] | None) -> Mapping[str, int]:
    return MappingProxyType({str(model): layer_mask(parse_layers(spec)) for model, spec in dict(value or {}).items()})


def model_from_json(row: Mapping[str, object]) -> ModelLayers:
    overrides = {str(fmt): MappingProxyType({int(layer): int(size) for layer, size in table.items()})
                 for fmt, table in dict(row.get("layer_bytes_overrides", {})).items()}
    return ModelLayers(
        model_id=str(row["model_id"]),
        cpu_layers=parse_layers(row["cpu_layers"]),
        layer_bytes=MappingProxyType({str(k): int(v) for k, v in dict(row["layer_bytes"]).items()}),
        layer_bytes_overrides=MappingProxyType(overrides),
        n_embd=int(row.get("n_embd", 0)),
        n_ff=int(row.get("n_ff", 0)),
        column_quantum=int(row.get("column_quantum", 0)),
        other_step_ms=MappingProxyType({int(k): float(v) for k, v in dict(row.get("other_step_ms", {})).items()}),
        host_format=str(row.get("host_format", "f16")),
    )


def model_to_json(model: ModelLayers) -> dict[str, object]:
    return {
        "model_id": model.model_id, "cpu_layers": layer_spec(model.cpu_layers),
        "layer_bytes": dict(sorted(model.layer_bytes.items())),
        **({"layer_bytes_overrides": {fmt: {str(k): v for k, v in sorted(table.items())}
                                      for fmt, table in sorted(model.layer_bytes_overrides.items())}}
           if model.layer_bytes_overrides else {}),
        "n_embd": model.n_embd, "n_ff": model.n_ff, "column_quantum": model.column_quantum,
        "other_step_ms": {str(k): v for k, v in sorted(model.other_step_ms.items())},
        "host_format": model.host_format,
    }


def device_from_json(row: Mapping[str, object]) -> HelperDevice:
    return HelperDevice(
        device_id=str(row["device_id"]),
        formats=MappingProxyType({str(k): str(v) for k, v in dict(row["formats"]).items()}),
        capacity_bytes=int(row["capacity_bytes"]),
        transport=str(row["transport"]),
        sessions=int(row.get("sessions", 1)),
        session_limit_bytes=int(row.get("session_limit_bytes", 0)),
        residency=str(row.get("residency", "co-resident")),
        available=bool(row.get("available", True)),
        unavailable_reason=row.get("unavailable_reason"),
        max_busy_ms_per_token=(None if row.get("max_busy_ms_per_token") is None
                               else float(row["max_busy_ms_per_token"])),
        allowed_layers=None if row.get("allowed_layers") is None else _masks(row["allowed_layers"]),
        busy_envelope_ms=None if row.get("busy_envelope_ms") is None else float(row["busy_envelope_ms"]),
        stored_layers=_masks(row.get("stored_layers")),
        qualified_layers=_masks(row.get("qualified_layers")),
        route_anchor=_route_anchor(row.get("route_anchor", True)),
        max_models=None if row.get("max_models") is None else _max_models(row["max_models"]),
        kernels=None if row.get("kernels") is None else _kernels(row["kernels"]),
        kernel_qualified_layers=(None if row.get("kernel_qualified_layers") is None
                                 else _kernel_masks(row["kernel_qualified_layers"])),
    )


def _kernel_masks(value: object) -> Mapping[str, Mapping[str, int]]:
    if not isinstance(value, Mapping):
        raise LayerPlacementError("device kernel_qualified_layers must map kernel -> model -> layers")
    return MappingProxyType({str(kernel): _masks(models) for kernel, models in dict(value).items()})


def _kernels(value: object) -> frozenset[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LayerPlacementError("device kernels must be a list of kernel names")
    return frozenset(value)


def _max_models(value: object) -> int:
    if type(value) is not int or value < 1:
        raise LayerPlacementError("device max_models must be an integer >= 1")
    return value


def _route_anchor(value: object) -> bool:
    if type(value) is not bool:
        raise LayerPlacementError("device route_anchor must be true or false")
    return value


def device_to_json(device: HelperDevice) -> dict[str, object]:
    def specs(value):
        return {model: layer_spec(mask_layers(mask)) for model, mask in sorted(value.items())}
    return {
        "device_id": device.device_id, "formats": dict(sorted(device.formats.items())),
        "capacity_bytes": device.capacity_bytes, "transport": device.transport, "sessions": device.sessions,
        "session_limit_bytes": device.session_limit_bytes, "residency": device.residency,
        "available": device.available, "unavailable_reason": device.unavailable_reason,
        "max_busy_ms_per_token": device.max_busy_ms_per_token,
        **({"busy_envelope_ms": device.busy_envelope_ms} if device.busy_envelope_ms is not None else {}),
        **({"allowed_layers": specs(device.allowed_layers)} if device.allowed_layers is not None else {}),
        "stored_layers": specs(device.stored_layers), "qualified_layers": specs(device.qualified_layers),
        # written only for co-helpers, so inventories without one stay byte-identical
        **({"route_anchor": False} if not device.route_anchor else {}),
        **({"max_models": device.max_models} if device.max_models is not None else {}),
        **({"kernels": sorted(device.kernels)} if device.kernels is not None else {}),
        **({"kernel_qualified_layers": {kernel: specs(models) for kernel, models
                                        in sorted(device.kernel_qualified_layers.items())}}
           if device.kernel_qualified_layers is not None else {}),
    }


def current_from_json(models: Iterable[ModelLayers], value: Mapping[str, object] | None) -> dict[str, dict[int, str]]:
    """``{model: {device: "0-17", ...}}`` -> model -> layer -> owner (unlisted CPU layers stay on the CPU)."""
    result = {}
    for model in models:
        owners = {layer: DESKTOP_CPU for layer in model.cpu_layers}
        for device, spec in dict((value or {}).get(model.model_id, {})).items():
            for layer in parse_layers(spec):
                if layer not in owners:
                    raise LayerPlacementError(f"{model.model_id} layer {layer} is not CPU-resident")
                owners[layer] = str(device)
        result[model.model_id] = owners
    return result


def problem_from_json(inventory: Mapping[str, object], profile: LayerPlacementProfile) -> PlacementProblem:
    if inventory.get("schema") != LAYER_PLACEMENT_INVENTORY_SCHEMA:
        raise LayerPlacementError("not a layer placement inventory")
    models = tuple(model_from_json(row) for row in inventory["models"])
    devices = tuple(device_from_json(row) for row in inventory["devices"])
    mix = {str(row["model_id"]): MappingProxyType({int(k): float(v) for k, v in dict(row.get("rows_mix", {"1": 1})).items()})
           for row in inventory["models"]}
    fractions = {str(k): tuple(float(item) for item in v) for k, v in dict(inventory.get("column_fractions", {})).items()}
    current = current_from_json(models, inventory.get("current"))
    return PlacementProblem(models=models, devices=devices, profile=profile, rows_mix=MappingProxyType(mix),
                            latency_ppm=inventory.get("latency_ppm"), column_fractions=MappingProxyType(fractions),
                            current=MappingProxyType({k: MappingProxyType(v) for k, v in current.items()}))


def inventory_to_json(problem: PlacementProblem) -> dict[str, object]:
    return {
        "schema": LAYER_PLACEMENT_INVENTORY_SCHEMA,
        "models": [{**model_to_json(model), "rows_mix": {str(k): v for k, v in problem.mix(model.model_id).items()}}
                   for model in problem.models],
        "devices": [device_to_json(device) for device in problem.devices],
        "current": {model: {owner: layer_spec(layer for layer, value in owners.items() if value == owner)
                            for owner in sorted(set(owners.values())) if owner != DESKTOP_CPU}
                    for model, owners in sorted(problem.current.items())},
        "latency_ppm": problem.latency_ppm,
        "column_fractions": {k: list(v) for k, v in sorted(problem.column_fractions.items())},
    }


def parse_shape_line(line: str) -> dict[str, object] | None:
    """One ``S41SERVERFFNSHAPE {...}`` line (any log prefix) -> its JSON object, else None."""
    position = line.find(SHAPE_PREFIX)
    if position < 0:
        return None
    try:
        row = json.loads(line[position + len(SHAPE_PREFIX):].strip())
    except json.JSONDecodeError as error:
        raise LayerPlacementError("malformed S41SERVERFFNSHAPE row: " + str(error)) from error
    for name in ("tokens", "columns", "calls", "rpc_mean_ms", "compute_mean_ms"):
        if name not in row:
            raise LayerPlacementError("S41SERVERFFNSHAPE row lacks " + name)
    return row


def shape_observations(lines: Iterable[str], *, model_id: str, n_ff: int, helpers: Mapping[str, Mapping[str, str]],
                       default_helper: str | None = None, evidence: str = "") -> list[dict[str, object]]:
    """Full-width server summary rows -> ``LayerPlacementProfile.observe_call_summary`` keyword rows.

    ``helpers`` maps the server's helper label to ``{"device_id", "format", "transport"}``; single-helper
    servers print no label, so ``default_helper`` names the label to assume. Rows of a partial column width
    are skipped (they time a column split, not a layer call).
    """
    rows = []
    for line in lines:
        row = parse_shape_line(line)
        if row is None or int(row["columns"]) != n_ff or int(row["calls"]) <= 0:
            continue
        label = row.get("helper", default_helper)
        if label not in helpers:
            raise LayerPlacementError("server helper label " + str(label) + " has no device mapping")
        helper = helpers[label]
        rows.append({"device_id": helper["device_id"], "model_id": model_id, "shard_format": helper["format"],
                     "transport": helper["transport"], "rows": int(row["tokens"]), "calls": int(row["calls"]),
                     "rpc_ms": float(row["rpc_mean_ms"]), "compute_ms": float(row["compute_mean_ms"]),
                     "evidence": evidence})
    return rows


def parse_fused_shape_line(line: str) -> dict[str, object] | None:
    """One ``S41SERVERFFNFUSEDSHAPE {...}`` line (WS15 fused call summary) -> its JSON object, else None."""
    position = line.find(FUSED_SHAPE_PREFIX)
    if position < 0:
        return None
    try:
        row = json.loads(line[position + len(FUSED_SHAPE_PREFIX):].strip())
    except json.JSONDecodeError as error:
        raise LayerPlacementError("malformed S41SERVERFFNFUSEDSHAPE row: " + str(error)) from error
    for name in ("kind", "tokens", "next_qkv", "calls", "rpc_mean_ms", "compute_mean_ms"):
        if name not in row:
            raise LayerPlacementError("S41SERVERFFNFUSEDSHAPE row lacks " + name)
    if row["kind"] != "attn_proj:fused" or type(row["next_qkv"]) is not bool:
        raise LayerPlacementError("S41SERVERFFNFUSEDSHAPE row has an unknown kind")
    return row


def fused_shape_observations(lines: Iterable[str], *, model_id: str, helpers: Mapping[str, Mapping[str, str]],
                             default_helper: str | None = None,
                             evidence: str = "") -> list[dict[str, object]]:
    """Fused call summaries -> ``LayerPlacementProfile.observe_fused_call_summary`` keyword rows.

    The stored format is the helper's FFN format, suffixed ``+next`` for calls that also returned the next
    layer's Q/K/V, so the planner can price both unit shapes from measurements."""
    rows = []
    for line in lines:
        row = parse_fused_shape_line(line)
        if row is None or int(row["calls"]) <= 0:
            continue
        label = row.get("helper", default_helper)
        if label not in helpers:
            raise LayerPlacementError("server helper label " + str(label) + " has no device mapping")
        helper = helpers[label]
        rows.append({"device_id": helper["device_id"], "model_id": model_id,
                     "shard_format": helper["format"] + ("+next" if row["next_qkv"] else ""),
                     "transport": helper["transport"], "rows": int(row["tokens"]), "calls": int(row["calls"]),
                     "rpc_ms": float(row["rpc_mean_ms"]), "compute_ms": float(row["compute_mean_ms"]),
                     "evidence": evidence})
    return rows
