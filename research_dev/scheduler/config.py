"""Typed, hash-bound configuration for portable physical campaigns."""

from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ._internal.types import canonical_json, canonical_sha256


from .configuration.common import (
    RIG_MANIFEST_SCHEMA as RIG_MANIFEST_SCHEMA,
    MODELS_MANIFEST_SCHEMA as MODELS_MANIFEST_SCHEMA,
    EVIDENCE_MANIFEST_SCHEMA as EVIDENCE_MANIFEST_SCHEMA,
    CAMPAIGN_MANIFEST_SCHEMA as CAMPAIGN_MANIFEST_SCHEMA,
    RESOLVED_CONFIGURATION_SCHEMA as RESOLVED_CONFIGURATION_SCHEMA,
    _ENVIRONMENT_NAME as _ENVIRONMENT_NAME,
    _SHA256 as _SHA256,
    SchedulerConfigurationError as SchedulerConfigurationError,
    _require as _require,
    _object as _object,
    _sequence as _sequence,
    _text as _text,
    _optional_text as _optional_text,
    _integer as _integer,
    _boolean as _boolean,
    _path as _path,
    _optional_path as _optional_path,
    _path_map as _path_map,
    _text_map as _text_map,
    _scalar_map as _scalar_map,
    _path_json as _path_json,
    _decode_override as _decode_override,
    _pointer_parts as _pointer_parts,
    _pointer_target as _pointer_target,
)
from .configuration.overrides import (
    ConfigurationOverride as ConfigurationOverride,
    ManifestIdentity as ManifestIdentity,
    _load_raw_manifest as _load_raw_manifest,
)
from .configuration.rig import (
    RigDeviceConfiguration as RigDeviceConfiguration,
    RigResourceConfiguration as RigResourceConfiguration,
    RigTopologyConfiguration as RigTopologyConfiguration,
    PhoneRigConfiguration as PhoneRigConfiguration,
    RigManifest as RigManifest,
)
from .configuration.models import (
    CampaignModelConfiguration as CampaignModelConfiguration,
    ModelsManifest as ModelsManifest,
)
from .configuration.evidence import (
    PhonePowerEvidenceConfiguration as PhonePowerEvidenceConfiguration,
    EvidenceManifest as EvidenceManifest,
)
from .configuration.campaign import (
    TraceConfiguration as TraceConfiguration,
    FixedPhoneResidencyConfiguration as FixedPhoneResidencyConfiguration,
    PhoneHtpMemoryCapConfiguration as PhoneHtpMemoryCapConfiguration,
    PhoneThermalStatusLimitConfiguration as PhoneThermalStatusLimitConfiguration,
    DeviceDecodeCapConfiguration as DeviceDecodeCapConfiguration,
    DeviceIdlePowerConfiguration as DeviceIdlePowerConfiguration,
    DevicePowerConfiguration as DevicePowerConfiguration,
    SpeculativeRowsModelConfiguration as SpeculativeRowsModelConfiguration,
    speculative_rows_configuration as speculative_rows_configuration,
    PhoneResidentModelReprovisioningConfiguration as PhoneResidentModelReprovisioningConfiguration,
    StartupDesktopParentConfiguration as StartupDesktopParentConfiguration,
    CampaignManifest as CampaignManifest,
)
from .configuration.resolved import ResolvedSchedulerConfiguration as ResolvedSchedulerConfiguration


def _load_typed_manifest(
    path: Path,
    schema: str,
    constructor,
    environ: Mapping[str, str],
) -> tuple[object, ManifestIdentity, tuple[ConfigurationOverride, ...]]:
    resolved, source_sha256, overrides = _load_raw_manifest(
        path, schema, environ
    )
    value = constructor(resolved, path.parent)
    resolved_sha256 = canonical_sha256(value.to_json())
    return (
        value,
        ManifestIdentity(path, schema, source_sha256, resolved_sha256),
        overrides,
    )


def load_scheduler_configuration(
    campaign_path: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> ResolvedSchedulerConfiguration:
    environment = os.environ if environ is None else environ
    campaign_path = campaign_path.resolve()
    campaign, campaign_identity, campaign_overrides = _load_typed_manifest(
        campaign_path,
        CAMPAIGN_MANIFEST_SCHEMA,
        CampaignManifest.from_json,
        environment,
    )
    assert isinstance(campaign, CampaignManifest)
    rig, rig_identity, rig_overrides = _load_typed_manifest(
        campaign.rig_manifest_path,
        RIG_MANIFEST_SCHEMA,
        RigManifest.from_json,
        environment,
    )
    models, models_identity, model_overrides = _load_typed_manifest(
        campaign.models_manifest_path,
        MODELS_MANIFEST_SCHEMA,
        ModelsManifest.from_json,
        environment,
    )
    evidence, evidence_identity, evidence_overrides = _load_typed_manifest(
        campaign.evidence_manifest_path,
        EVIDENCE_MANIFEST_SCHEMA,
        EvidenceManifest.from_json,
        environment,
    )
    assert isinstance(rig, RigManifest)
    assert isinstance(models, ModelsManifest)
    assert isinstance(evidence, EvidenceManifest)
    endpoint_ids = set(rig.endpoints)
    for model in models.models:
        required_bindings = (
            {"desktop", "phone"}
            if model.kind == "assisted"
            else {"desktop"}
        )
        _require(
            set(model.endpoint_ids.values()) <= endpoint_ids,
            "model endpoint reference is absent from rig manifest",
        )
        _require(
            set(model.helper_phone_ffn_shards)
            <= {row.device_id for row in rig.helper_phones},
            "model helper phone shards name an absent rig helper phone",
        )
        _require(
            required_bindings <= set(model.endpoint_ids)
            and required_bindings <= set(model.backend_ids)
            and set(model.backend_ids) >= set(model.endpoint_ids),
            "model physical bindings are incomplete",
        )
    identities = MappingProxyType({
        "campaign": campaign_identity,
        "evidence": evidence_identity,
        "models": models_identity,
        "rig": rig_identity,
    })
    overrides = tuple(sorted(
        (
            *campaign_overrides,
            *rig_overrides,
            *model_overrides,
            *evidence_overrides,
        ),
        key=lambda row: (
            row.manifest_schema, row.environment_name, row.pointer
        ),
    ))
    return ResolvedSchedulerConfiguration(
        campaign=campaign,
        rig=rig,
        models=models,
        evidence=evidence,
        manifest_identities=identities,
        overrides=overrides,
    )


def write_resolved_configuration(
    configuration: ResolvedSchedulerConfiguration,
    path: Path,
) -> None:
    _require(
        isinstance(configuration, ResolvedSchedulerConfiguration)
        and path.is_absolute()
        and not path.exists(),
        "resolved configuration output must be new and absolute",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(configuration.canonical_bytes())


__all__ = [
    'CAMPAIGN_MANIFEST_SCHEMA',
    'CampaignManifest',
    'CampaignModelConfiguration',
    'ConfigurationOverride',
    'DeviceDecodeCapConfiguration',
    'DeviceIdlePowerConfiguration',
    'DevicePowerConfiguration',
    'SpeculativeRowsModelConfiguration',
    'speculative_rows_configuration',
    'EVIDENCE_MANIFEST_SCHEMA',
    'EvidenceManifest',
    'FixedPhoneResidencyConfiguration',
    'MODELS_MANIFEST_SCHEMA',
    'ManifestIdentity',
    'ModelsManifest',
    'PhoneHtpMemoryCapConfiguration',
    'PhoneThermalStatusLimitConfiguration',
    'PhoneResidentModelReprovisioningConfiguration',
    'PhonePowerEvidenceConfiguration',
    'PhoneRigConfiguration',
    'RESOLVED_CONFIGURATION_SCHEMA',
    'RIG_MANIFEST_SCHEMA',
    'ResolvedSchedulerConfiguration',
    'RigDeviceConfiguration',
    'RigManifest',
    'RigResourceConfiguration',
    'RigTopologyConfiguration',
    'SchedulerConfigurationError',
    'StartupDesktopParentConfiguration',
    'TraceConfiguration',
    '_ENVIRONMENT_NAME',
    '_SHA256',
    '_boolean',
    '_decode_override',
    '_integer',
    '_load_raw_manifest',
    '_object',
    '_optional_path',
    '_optional_text',
    '_path',
    '_path_json',
    '_path_map',
    '_pointer_parts',
    '_pointer_target',
    '_require',
    '_scalar_map',
    '_sequence',
    '_text',
    '_text_map',
    'canonical_json',
    'canonical_sha256',
    'load_scheduler_configuration',
    'write_resolved_configuration',
]
