"""Capacity-aware desktop parent placement for live GPU memory."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .capacity import DeviceMemoryCapacity
from .model_manifest import ModelManifest
from .runtime_capabilities import (
    desktop_control_placement_payload,
    RuntimeCompositeExecutorCapability,
    RuntimeCompositeOperatorPlacement,
)
from .types import canonical_sha256


class DesktopParentCapacityError(ValueError):
    pass


_DEFAULT_GPU_WEIGHT_ALLOCATION_PPM = 1_150_000


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DesktopParentCapacityError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise DesktopParentCapacityError(
            f"{name} must be non-empty ASCII text"
        )
    return value


def _layer_index(layer_id: str) -> int | None:
    if not layer_id.startswith("layer:"):
        return None
    try:
        return int(layer_id.removeprefix("layer:"))
    except ValueError as error:
        raise DesktopParentCapacityError(
            "desktop parent layer id is invalid"
        ) from error


@dataclass(frozen=True)
class DesktopParentCapacityCandidate:
    artifact_sha256: str
    executor_id: str
    gpu_first_layer: int
    gpu_layers: int
    gpu_weight_bytes: int
    gpu_weight_allocation_ppm: int
    gpu_weight_allocation_bytes: int
    gpu_kv_bytes: int
    gpu_workspace_bytes: int
    gpu_reserve_bytes: int
    live_free_vram_bytes: int
    required_with_reserve_bytes: int
    feasible: bool
    placement_sha256: str
    operator_placements: tuple[RuntimeCompositeOperatorPlacement, ...]
    adapter_parameters: Mapping[str, int | str]

    def __post_init__(self) -> None:
        _text("desktop parent artifact", self.artifact_sha256)
        _text("desktop parent executor", self.executor_id)
        _text("desktop parent placement hash", self.placement_sha256)
        for name in (
            "gpu_first_layer",
            "gpu_layers",
            "gpu_weight_bytes",
            "gpu_weight_allocation_ppm",
            "gpu_weight_allocation_bytes",
            "gpu_kv_bytes",
            "gpu_workspace_bytes",
            "gpu_reserve_bytes",
            "live_free_vram_bytes",
            "required_with_reserve_bytes",
        ):
            _integer("desktop parent " + name, getattr(self, name))
        if type(self.feasible) is not bool:
            raise DesktopParentCapacityError(
                "desktop parent feasibility is invalid"
            )
        placements = tuple(self.operator_placements)
        if not placements or len({row.operator_id for row in placements}) != len(
            placements
        ):
            raise DesktopParentCapacityError(
                "desktop parent operator placements are invalid"
            )
        object.__setattr__(
            self,
            "operator_placements",
            tuple(sorted(placements, key=lambda row: row.operator_id)),
        )
        object.__setattr__(
            self,
            "adapter_parameters",
            MappingProxyType(dict(sorted(self.adapter_parameters.items()))),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "adapter_parameters": dict(self.adapter_parameters),
            "artifact_sha256": self.artifact_sha256,
            "executor_id": self.executor_id,
            "feasible": self.feasible,
            "gpu_first_layer": self.gpu_first_layer,
            "gpu_kv_bytes": self.gpu_kv_bytes,
            "gpu_layers": self.gpu_layers,
            "gpu_reserve_bytes": self.gpu_reserve_bytes,
            "gpu_weight_allocation_bytes": self.gpu_weight_allocation_bytes,
            "gpu_weight_allocation_ppm": self.gpu_weight_allocation_ppm,
            "gpu_weight_bytes": self.gpu_weight_bytes,
            "gpu_workspace_bytes": self.gpu_workspace_bytes,
            "live_free_vram_bytes": self.live_free_vram_bytes,
            "operator_placements": [
                row.to_json() for row in self.operator_placements
            ],
            "placement_sha256": self.placement_sha256,
            "required_with_reserve_bytes": self.required_with_reserve_bytes,
        }


@dataclass(frozen=True)
class DesktopParentCapacitySelection:
    memory_snapshot_id: str
    memory_resource_id: str
    maximum_gpu_layers: int
    source_placement_sha256: str
    candidates: tuple[DesktopParentCapacityCandidate, ...]
    selected: DesktopParentCapacityCandidate

    def __post_init__(self) -> None:
        _text("desktop parent memory snapshot", self.memory_snapshot_id)
        _text("desktop parent memory resource", self.memory_resource_id)
        _integer("desktop parent maximum GPU layers", self.maximum_gpu_layers)
        _text("desktop parent source placement", self.source_placement_sha256)
        rows = tuple(self.candidates)
        if (
            not rows
            or self.selected not in rows
            or not self.selected.feasible
            or self.selected.gpu_layers != max(
                row.gpu_layers for row in rows if row.feasible
            )
        ):
            raise DesktopParentCapacityError(
                "desktop parent capacity selection is invalid"
            )
        object.__setattr__(self, "candidates", rows)

    @property
    def selection_sha256(self) -> str:
        return canonical_sha256(self.to_json(include_hash=False))

    def to_json(self, *, include_hash: bool = True) -> dict[str, object]:
        result = {
            "candidates": [row.to_json() for row in self.candidates],
            "maximum_gpu_layers": self.maximum_gpu_layers,
            "memory_resource_id": self.memory_resource_id,
            "memory_snapshot_id": self.memory_snapshot_id,
            "selected_placement_sha256": self.selected.placement_sha256,
            "source_placement_sha256": self.source_placement_sha256,
        }
        if include_hash:
            result["selection_sha256"] = self.selection_sha256
        return result


def _operator_placements(
    manifest: ModelManifest,
    source: RuntimeCompositeExecutorCapability,
    *,
    gpu_first_layer: int,
    cpu_device_id: str,
    gpu_device_id: str,
) -> tuple[RuntimeCompositeOperatorPlacement, ...]:
    source_by_id = {row.operator_id: row for row in source.operator_placements}
    if set(source_by_id) != {row.operator_id for row in manifest.operators}:
        raise DesktopParentCapacityError(
            "desktop parent source does not cover the model graph"
        )
    rows = []
    for operator in manifest.operators:
        original = source_by_id[operator.operator_id]
        if (
            original.helper_device_id is not None
            or original.split_axis != "none"
            or original.split_fraction_ppm != 0
            or original.assisted
        ):
            raise DesktopParentCapacityError(
                "desktop parent source is not an unassisted placement"
            )
        layer = _layer_index(operator.layer_id)
        primary = original.primary_device_id
        if layer is not None:
            primary = (
                gpu_device_id if layer >= gpu_first_layer else cpu_device_id
            )
        rows.append(RuntimeCompositeOperatorPlacement(
            operator_id=operator.operator_id,
            primary_device_id=primary,
            helper_device_id=None,
            split_axis="none",
            split_fraction_ppm=0,
        ))
    return tuple(sorted(rows, key=lambda row: row.operator_id))


def _placement_sha256(
    manifest: ModelManifest,
    placements: tuple[RuntimeCompositeOperatorPlacement, ...],
    cuda_graph_mode: str = "default",
) -> str:
    return canonical_sha256(desktop_control_placement_payload(
        manifest.artifact_sha256, placements, cuda_graph_mode
    ))


def _gpu_memory_components(
    manifest: ModelManifest,
    placements: tuple[RuntimeCompositeOperatorPlacement, ...],
    parameters: Mapping[str, int | str],
    gpu_device_id: str,
) -> tuple[int, int, int, int, int]:
    placement_by_id = {row.operator_id: row for row in placements}
    tensor_ids = {
        tensor_id
        for operator in manifest.operators
        if placement_by_id[operator.operator_id].primary_device_id
            == gpu_device_id
        for tensor_id in operator.tensor_ids
    }
    weight_bytes = sum(
        manifest.tensor_by_id[tensor_id].nbytes for tensor_id in tensor_ids
    )
    multiplier = parameters.get(
        "memory_model_weight_allocation_ppm:" + gpu_device_id,
        _DEFAULT_GPU_WEIGHT_ALLOCATION_PPM,
    )
    multiplier = _integer(
        "desktop parent GPU weight allocation factor", multiplier, 1_000_000
    )
    allocated_weights = (
        weight_bytes * multiplier + 1_000_000 - 1
    ) // 1_000_000

    context_size = _integer(
        "desktop parent context size", parameters.get("context_size"), 1
    )
    parallel = _integer(
        "desktop parent parallelism", parameters.get("parallel", 1), 1
    )
    padding = _integer(
        "desktop parent KV padding",
        parameters.get("kv_cache_swa_padding_tokens", 0),
    )
    kv_bytes = sum(
        manifest.preallocated_kv_cache_bytes(
            operator.operator_id,
            context_size=context_size,
            parallel=parallel,
            sliding_window_padding_tokens=padding,
        )
        for operator in manifest.operators
        if operator.kind == "kv_cache"
        and placement_by_id[operator.operator_id].primary_device_id
            == gpu_device_id
    )

    minimum_workspace = _integer(
        "desktop parent GPU workspace minimum",
        parameters.get(
            "memory_workspace_minimum_bytes:" + gpu_device_id,
            0,
        ),
    )
    batch_size = _integer(
        "desktop parent batch size", parameters.get("batch_size"), 1
    )
    ubatch_size = _integer(
        "desktop parent ubatch size", parameters.get("ubatch_size"), 1
    )
    estimated_workspace = (
        max(batch_size, ubatch_size) * manifest.embedding_length * 2
    )
    workspace_bytes = max(1, minimum_workspace, estimated_workspace)
    return (
        weight_bytes,
        multiplier,
        allocated_weights,
        kv_bytes,
        workspace_bytes,
    )


def select_desktop_parent_for_live_vram(
    manifest: ModelManifest,
    source: RuntimeCompositeExecutorCapability,
    capacity: DeviceMemoryCapacity,
    *,
    memory_snapshot_id: str,
    maximum_gpu_layers: int | None = None,
) -> DesktopParentCapacitySelection:
    """Select the largest source-compatible GPU suffix fitting live VRAM."""
    if not isinstance(manifest, ModelManifest):
        raise DesktopParentCapacityError("desktop parent manifest is invalid")
    if not isinstance(source, RuntimeCompositeExecutorCapability):
        raise DesktopParentCapacityError("desktop parent source is invalid")
    if not isinstance(capacity, DeviceMemoryCapacity):
        raise DesktopParentCapacityError("desktop parent capacity is invalid")
    if source.artifact_sha256 != manifest.artifact_sha256:
        raise DesktopParentCapacityError(
            "desktop parent source artifact differs from the manifest"
        )
    parameters = dict(source.adapter_parameters)
    cpu_device_id = _text(
        "desktop parent CPU device", parameters.get("cpu_device_id")
    )
    gpu_device_id = _text(
        "desktop parent GPU device", parameters.get("gpu_device_id")
    )
    configured_maximum = parameters.get(
        "capacity_parent_maximum_gpu_layers",
        parameters.get("gpu_layers"),
    )
    if maximum_gpu_layers is None:
        maximum_gpu_layers = configured_maximum
    maximum_gpu_layers = _integer(
        "desktop parent maximum GPU layers", maximum_gpu_layers
    )
    if maximum_gpu_layers > manifest.block_count:
        raise DesktopParentCapacityError(
            "desktop parent maximum GPU layers exceeds the model"
        )
    live_free = capacity.capacity_bytes - capacity.occupied_bytes
    candidates = []
    for gpu_layers in range(maximum_gpu_layers, -1, -1):
        gpu_first_layer = manifest.block_count - gpu_layers
        placements = _operator_placements(
            manifest,
            source,
            gpu_first_layer=gpu_first_layer,
            cpu_device_id=cpu_device_id,
            gpu_device_id=gpu_device_id,
        )
        (
            weight_bytes,
            multiplier,
            allocated_weights,
            kv_bytes,
            workspace_bytes,
        ) = _gpu_memory_components(
            manifest, placements, parameters, gpu_device_id
        )
        required = (
            allocated_weights
            + kv_bytes
            + workspace_bytes
            + capacity.reserve_bytes
        )
        candidate_parameters = {
            **parameters,
            "gpu_layers": gpu_layers,
            "memory_model_weight_allocation_ppm:" + gpu_device_id: (
                multiplier
            ),
        }
        candidates.append(DesktopParentCapacityCandidate(
            artifact_sha256=manifest.artifact_sha256,
            executor_id=source.executor_id,
            gpu_first_layer=gpu_first_layer,
            gpu_layers=gpu_layers,
            gpu_weight_bytes=weight_bytes,
            gpu_weight_allocation_ppm=multiplier,
            gpu_weight_allocation_bytes=allocated_weights,
            gpu_kv_bytes=kv_bytes,
            gpu_workspace_bytes=workspace_bytes,
            gpu_reserve_bytes=capacity.reserve_bytes,
            live_free_vram_bytes=live_free,
            required_with_reserve_bytes=required,
            feasible=required <= live_free,
            placement_sha256=_placement_sha256(
                manifest, placements, parameters.get("cuda_graph_mode", "default")
            ),
            operator_placements=placements,
            adapter_parameters=candidate_parameters,
        ))
    selected = next((row for row in candidates if row.feasible), None)
    if selected is None:
        raise DesktopParentCapacityError(
            "live VRAM cannot satisfy even the CPU desktop parent workspace"
        )
    source_placement = candidates[0].placement_sha256
    expected_source = parameters.get(
        "capacity_parent_source_placement_sha256"
    )
    if expected_source is not None and expected_source != source_placement:
        raise DesktopParentCapacityError(
            "desktop parent source placement identity differs"
        )
    return DesktopParentCapacitySelection(
        memory_snapshot_id=_text(
            "desktop parent memory snapshot", memory_snapshot_id
        ),
        memory_resource_id=capacity.resource_id,
        maximum_gpu_layers=maximum_gpu_layers,
        source_placement_sha256=source_placement,
        candidates=tuple(candidates),
        selected=selected,
    )
