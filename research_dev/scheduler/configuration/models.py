"""Typed scheduler configuration manifests: models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .common import (
    MODELS_MANIFEST_SCHEMA,
    _integer,
    _object,
    _optional_path,
    _optional_text,
    _path,
    _path_json,
    _require,
    _scalar_map,
    _sequence,
    _text,
    _text_map,
)


HOST_SHARE_POLICY_KEYS = ("ffn_host_share_drop_cache", "ffn_host_share_populate")


def validate_host_share_policy(
    parameters: Mapping[str, int | str],
) -> Mapping[str, int | str]:
    """Cache-policy flags belong to the dormant host share: a model without
    ``ffn_host_share_release == 1`` must not carry them. The launch contract
    rejects such a server at load time; refusing here fails the run at
    resolve/preflight instead of on the first physical transition."""
    present = [key for key in HOST_SHARE_POLICY_KEYS if key in parameters]
    _require(
        not present or parameters.get("ffn_host_share_release") == 1,
        "host share cache policy requires ffn_host_share_release == 1: "
        + ", ".join(present),
    )
    _require(
        all(parameters[key] in (0, 1) for key in present),
        "host share cache policy flags must be 0 or 1",
    )
    return parameters


@dataclass(frozen=True)
class CampaignModelConfiguration:
    model_key: str
    model_id: str
    trace_role: str
    kind: str
    host_artifact_path: Path
    checked_manifest_path: Path | None
    phone_artifact_path: str
    phone_ffn_shard_index_path: Path | None
    phone_ffn_shard_directory: str | None
    endpoint_ids: Mapping[str, str]
    backend_ids: Mapping[str, str]
    runtime_parameters: Mapping[str, int | str]
    phone_adapter_parameters: Mapping[str, int | str]
    calibration_key: str | None
    transition_load_metric: str | None
    transition_warm_metric: str | None
    gpu_reserve_bytes: int
    phone_residency_mode: str
    phone_resident_limit_bytes: int
    ffn_max_runtime_partitions: int
    phone_batch_plans: tuple[str, ...]
    qualified_phone_batch_plans: tuple[str, ...]
    cpu_runtime_parameters: Mapping[str, int | str] = field(default_factory=dict)
    cpu_evidence_ids: tuple[str, ...] = ()
    # offline FFN shards on secondary helper phones: device id -> (local index, phone directory)
    helper_phone_ffn_shards: Mapping[str, tuple[Path, str]] = field(default_factory=dict)

    @classmethod
    def from_json(
        cls, value: object, base: Path
    ) -> "CampaignModelConfiguration":
        row = _object(value, "campaign model")
        kind = _text(row.get("kind"), "campaign model kind")
        _require(kind in {"assisted", "overlay"}, "campaign model kind")
        calibration_key = _optional_text(
            row.get("calibration_key"), "model calibration key"
        )
        load_metric = _optional_text(
            row.get("transition_load_metric"), "transition load metric"
        )
        warm_metric = _optional_text(
            row.get("transition_warm_metric"), "transition warm metric"
        )
        manifest = _optional_path(
            row.get("checked_manifest_path"), base, "checked model manifest"
        )
        ffn_shard_index = _optional_path(
            row.get("phone_ffn_shard_index_path"),
            base,
            "phone FFN shard index",
        )
        ffn_shard_directory = _optional_text(
            row.get("phone_ffn_shard_directory"),
            "phone FFN shard directory",
        )
        _require(
            kind != "assisted"
            or all(value is not None for value in (
                calibration_key, load_metric, warm_metric, manifest,
            )),
            "assisted model evidence configuration is incomplete",
        )
        _require(
            (ffn_shard_index is None) == (ffn_shard_directory is None),
            "phone FFN shard configuration is incomplete",
        )
        residency_mode = _text(
            row.get("phone_residency_mode", "transition"),
            "phone residency mode",
        )
        _require(
            residency_mode in {"preloaded", "transition"},
            "phone residency mode",
        )
        batch_plans = tuple(
            _text(item, "phone batch plan")
            for item in _sequence(
                row.get("phone_batch_plans", []), "phone batch plans"
            )
        )
        qualified_batch_plans = tuple(
            _text(item, "qualified phone batch plan")
            for item in _sequence(
                row.get("qualified_phone_batch_plans", []),
                "qualified phone batch plans",
            )
        )
        _require(
            len(set(batch_plans)) == len(batch_plans)
            and set(qualified_batch_plans) <= set(batch_plans),
            "phone batch plan qualification",
        )
        return cls(
            model_key=_text(row.get("model_key"), "model key"),
            model_id=_text(row.get("model_id"), "model id"),
            trace_role=_text(row.get("trace_role"), "model trace role"),
            kind=kind,
            host_artifact_path=_path(
                row.get("host_artifact_path"), base, "host model artifact"
            ),
            checked_manifest_path=manifest,
            phone_artifact_path=_text(
                row.get("phone_artifact_path"), "phone model artifact"
            ),
            phone_ffn_shard_index_path=ffn_shard_index,
            phone_ffn_shard_directory=ffn_shard_directory,
            endpoint_ids=_text_map(row.get("endpoint_ids"), "model endpoints"),
            backend_ids=_text_map(row.get("backend_ids"), "model backends"),
            runtime_parameters=_scalar_map(
                row.get("runtime_parameters", {}), "model runtime parameters"
            ),
            phone_adapter_parameters=validate_host_share_policy(_scalar_map(
                row.get("phone_adapter_parameters", {}),
                "model phone adapter parameters",
            )),
            calibration_key=calibration_key,
            transition_load_metric=load_metric,
            transition_warm_metric=warm_metric,
            gpu_reserve_bytes=_integer(
                row.get("gpu_reserve_bytes", 0), "model GPU reserve"
            ),
            phone_residency_mode=residency_mode,
            phone_resident_limit_bytes=_integer(
                row.get("phone_resident_limit_bytes", 1),
                "phone resident limit",
                minimum=1,
            ),
            ffn_max_runtime_partitions=_integer(
                row.get("ffn_max_runtime_partitions", 1),
                "FFN maximum runtime partitions",
                minimum=1,
            ),
            phone_batch_plans=batch_plans,
            qualified_phone_batch_plans=qualified_batch_plans,
            cpu_runtime_parameters=_scalar_map(
                row.get("cpu_runtime_parameters", {}),
                "model CPU runtime parameters",
            ),
            cpu_evidence_ids=tuple(
                _text(item, "model CPU evidence")
                for item in _sequence(
                    row.get("cpu_evidence_ids", []), "model CPU evidence"
                )
            ),
            helper_phone_ffn_shards=MappingProxyType({
                _text(device_id, "helper phone shard device"): (
                    _path(_object(spec, "helper phone FFN shards").get("index_path"), base,
                          "helper phone FFN shard index"),
                    _text(_object(spec, "helper phone FFN shards").get("directory"),
                          "helper phone FFN shard directory"),
                )
                for device_id, spec in sorted(_object(
                    row.get("helper_phone_ffn_shards", {}), "helper phone FFN shards"
                ).items())
            }),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "calibration_key": self.calibration_key,
            "checked_manifest_path": _path_json(self.checked_manifest_path),
            "endpoint_ids": dict(self.endpoint_ids),
            "backend_ids": dict(self.backend_ids),
            "gpu_reserve_bytes": self.gpu_reserve_bytes,
            "host_artifact_path": str(self.host_artifact_path),
            "kind": self.kind,
            "model_id": self.model_id,
            "model_key": self.model_key,
            "phone_adapter_parameters": dict(self.phone_adapter_parameters),
            "phone_batch_plans": list(self.phone_batch_plans),
            "phone_artifact_path": self.phone_artifact_path,
            "phone_ffn_shard_directory": self.phone_ffn_shard_directory,
            "phone_ffn_shard_index_path": _path_json(
                self.phone_ffn_shard_index_path
            ),
            "phone_residency_mode": self.phone_residency_mode,
            "phone_resident_limit_bytes": self.phone_resident_limit_bytes,
            "qualified_phone_batch_plans": list(
                self.qualified_phone_batch_plans
            ),
            "runtime_parameters": dict(self.runtime_parameters),
            "trace_role": self.trace_role,
            "transition_load_metric": self.transition_load_metric,
            "transition_warm_metric": self.transition_warm_metric,
            "ffn_max_runtime_partitions": self.ffn_max_runtime_partitions,
            **(
                {"cpu_runtime_parameters": dict(self.cpu_runtime_parameters)}
                if self.cpu_runtime_parameters else {}
            ),
            **(
                {"cpu_evidence_ids": list(self.cpu_evidence_ids)}
                if self.cpu_evidence_ids else {}
            ),
            **(
                {"helper_phone_ffn_shards": {
                    device_id: {"directory": directory, "index_path": str(index_path)}
                    for device_id, (index_path, directory) in self.helper_phone_ffn_shards.items()
                }}
                if self.helper_phone_ffn_shards else {}
            ),
        }


@dataclass(frozen=True)
class ModelsManifest:
    manifest_cache_path: Path
    desktop_baseline_plans_path: Path
    models: tuple[CampaignModelConfiguration, ...]

    @classmethod
    def from_json(cls, value: object, base: Path) -> "ModelsManifest":
        row = _object(value, "models manifest")
        _require(
            row.get("schema") == MODELS_MANIFEST_SCHEMA,
            "models manifest schema",
        )
        models = tuple(
            CampaignModelConfiguration.from_json(item, base)
            for item in _sequence(row.get("models"), "models")
        )
        _require(
            models
            and len({item.model_key for item in models}) == len(models)
            and len({item.model_id for item in models}) == len(models)
            and len({item.trace_role for item in models}) == len(models),
            "model identities are not unique",
        )
        _require(
            sum(item.kind == "overlay" for item in models) == 1
            and sum(item.kind == "assisted" for item in models) >= 1,
            "models manifest needs assisted and overlay models",
        )
        return cls(
            manifest_cache_path=_path(
                row.get("manifest_cache_path"), base, "GGUF manifest cache"
            ),
            desktop_baseline_plans_path=_path(
                row.get("desktop_baseline_plans_path"),
                base,
                "desktop baseline plans",
            ),
            models=models,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "desktop_baseline_plans_path": str(
                self.desktop_baseline_plans_path
            ),
            "manifest_cache_path": str(self.manifest_cache_path),
            "models": [row.to_json() for row in self.models],
            "schema": MODELS_MANIFEST_SCHEMA,
        }

    @property
    def assisted_models(self) -> tuple[CampaignModelConfiguration, ...]:
        return tuple(row for row in self.models if row.kind == "assisted")

    @property
    def overlay_model(self) -> CampaignModelConfiguration:
        return next(row for row in self.models if row.kind == "overlay")

    @property
    def by_trace_role(self) -> Mapping[str, CampaignModelConfiguration]:
        return MappingProxyType({row.trace_role: row for row in self.models})
