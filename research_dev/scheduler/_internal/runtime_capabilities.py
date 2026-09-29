"""Capability registration and raw heterogeneous runtime observations."""

from __future__ import annotations


from .placement import PLACEMENT_PROFILE_SCHEMA, PlacementHardwareProfile
from .policy import ResourceProfile
from .runtime_placement import RuntimePlacementSnapshot
from .runtime_system_cost import (
    ROUTE_MATURITY_STATES,
    RuntimeProtectedWorkObservation,
    RuntimeSystemCostProfile,
)


from .capability_contracts.common import (
    RUNTIME_CAPABILITY_SCHEMA as RUNTIME_CAPABILITY_SCHEMA,
    RUNTIME_SYSTEM_SNAPSHOT_SCHEMA as RUNTIME_SYSTEM_SNAPSHOT_SCHEMA,
    RESIDENCY_STATES as RESIDENCY_STATES,
    SPLIT_AXES as SPLIT_AXES,
    COMPOSITE_ROUTE_FAMILIES as COMPOSITE_ROUTE_FAMILIES,
    GENERATED_ROUTE_FAMILIES as GENERATED_ROUTE_FAMILIES,
    _MATURITY_RANK as _MATURITY_RANK,
    PHONE_POWER_ESTIMATION_VERSION as PHONE_POWER_ESTIMATION_VERSION,
    PHONE_POWER_EVIDENCE_ASSUMED_4P5W as PHONE_POWER_EVIDENCE_ASSUMED_4P5W,
    RuntimeCapabilityError as RuntimeCapabilityError,
    _text as _text,
    _integer as _integer,
    _boolean as _boolean,
    _object as _object,
    _list as _list,
    _placement_profile_json as _placement_profile_json,
    _texts as _texts,
    _fractions as _fractions,
)
from .capability_contracts.phone import (
    RuntimePhonePowerProfile as RuntimePhonePowerProfile,
    RuntimePhoneSessionCapability as RuntimePhoneSessionCapability,
)
from .capability_contracts.profiles import (
    RuntimeKernelShapeProfile as RuntimeKernelShapeProfile,
    RuntimeRouteShapeProfile as RuntimeRouteShapeProfile,
)
from .capability_contracts.executors import (
    _executor_capability_values as _executor_capability_values,
    _executor_adapter_and_shapes as _executor_adapter_and_shapes,
    _executor_phone_sessions as _executor_phone_sessions,
    _initialize_executor_capability as _initialize_executor_capability,
    _executor_overlay_structures as _executor_overlay_structures,
    RuntimeExecutorCapability as RuntimeExecutorCapability,
)
from .capability_contracts.desktop import (
    RuntimeCompositeOperatorPlacement as RuntimeCompositeOperatorPlacement,
    desktop_control_placement_payload as desktop_control_placement_payload,
    RuntimeDesktopControlProfile as RuntimeDesktopControlProfile,
)
from .capability_contracts.composites import (
    _composite_basic_values as _composite_basic_values,
    _validate_composite_route_contract as _validate_composite_route_contract,
    _composite_metadata_values as _composite_metadata_values,
    _initialize_composite_executor_capability as _initialize_composite_executor_capability,
    RuntimeCompositeExecutorCapability as RuntimeCompositeExecutorCapability,
)
from .capability_contracts.transitions import (
    RuntimeTransitionCapability as RuntimeTransitionCapability,
)
from .capability_contracts.catalog import (
    _validate_catalog_executor as _validate_catalog_executor,
    _catalog_base_values as _catalog_base_values,
    _catalog_composite_executors as _catalog_composite_executors,
    _catalog_desktop_controls as _catalog_desktop_controls,
    _catalog_transitions as _catalog_transitions,
    _catalog_cost_profiles as _catalog_cost_profiles,
    _catalog_phone_power_profiles as _catalog_phone_power_profiles,
    _initialize_runtime_capability_catalog as _initialize_runtime_capability_catalog,
    RuntimeCapabilityCatalog as RuntimeCapabilityCatalog,
)
from .capability_contracts.snapshots import (
    RuntimeExecutorState as RuntimeExecutorState,
    RuntimeLinkState as RuntimeLinkState,
    ModelResidencyObservation as ModelResidencyObservation,
    PhoneSessionResidencyObservation as PhoneSessionResidencyObservation,
    HeterogeneousRuntimeSnapshot as HeterogeneousRuntimeSnapshot,
)

__all__ = [
    'COMPOSITE_ROUTE_FAMILIES',
    'GENERATED_ROUTE_FAMILIES',
    'HeterogeneousRuntimeSnapshot',
    'ModelResidencyObservation',
    'PHONE_POWER_ESTIMATION_VERSION',
    'PHONE_POWER_EVIDENCE_ASSUMED_4P5W',
    'PLACEMENT_PROFILE_SCHEMA',
    'PhoneSessionResidencyObservation',
    'PlacementHardwareProfile',
    'RESIDENCY_STATES',
    'ROUTE_MATURITY_STATES',
    'RUNTIME_CAPABILITY_SCHEMA',
    'RUNTIME_SYSTEM_SNAPSHOT_SCHEMA',
    'ResourceProfile',
    'RuntimeCapabilityCatalog',
    'RuntimeCapabilityError',
    'RuntimeCompositeExecutorCapability',
    'RuntimeCompositeOperatorPlacement',
    'RuntimeDesktopControlProfile',
    'RuntimeExecutorCapability',
    'RuntimeExecutorState',
    'RuntimeKernelShapeProfile',
    'RuntimeLinkState',
    'RuntimePhonePowerProfile',
    'RuntimePhoneSessionCapability',
    'RuntimePlacementSnapshot',
    'RuntimeProtectedWorkObservation',
    'RuntimeRouteShapeProfile',
    'RuntimeSystemCostProfile',
    'RuntimeTransitionCapability',
    'SPLIT_AXES',
    '_MATURITY_RANK',
    '_boolean',
    '_catalog_base_values',
    '_catalog_composite_executors',
    '_catalog_cost_profiles',
    '_catalog_desktop_controls',
    '_catalog_phone_power_profiles',
    '_catalog_transitions',
    '_composite_basic_values',
    '_composite_metadata_values',
    '_executor_adapter_and_shapes',
    '_executor_capability_values',
    '_executor_overlay_structures',
    '_executor_phone_sessions',
    '_fractions',
    '_initialize_composite_executor_capability',
    '_initialize_executor_capability',
    '_initialize_runtime_capability_catalog',
    '_integer',
    '_list',
    '_object',
    '_placement_profile_json',
    '_text',
    '_texts',
    '_validate_catalog_executor',
    '_validate_composite_route_contract',
    'desktop_control_placement_payload',
]
