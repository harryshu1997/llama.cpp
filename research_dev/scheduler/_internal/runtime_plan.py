"""Hash-bound contracts for generated heterogeneous execution plans."""

from __future__ import annotations


from .runtime_capabilities import ROUTE_MATURITY_STATES, SPLIT_AXES
from .runtime_cost import RuntimeExecutorBinding, RuntimeMemoryDemand


from .plan_contracts.common import (
    RUNTIME_EXECUTION_PLAN_SCHEMA as RUNTIME_EXECUTION_PLAN_SCHEMA,
    AUTOMATED_CANDIDATE_SET_SCHEMA as AUTOMATED_CANDIDATE_SET_SCHEMA,
    RUNTIME_TRANSITION_RECEIPT_STATUSES as RUNTIME_TRANSITION_RECEIPT_STATUSES,
    RUNTIME_EXECUTION_RECEIPT_STATUSES as RUNTIME_EXECUTION_RECEIPT_STATUSES,
    RUNTIME_EXECUTION_MODES as RUNTIME_EXECUTION_MODES,
    RUNTIME_BATCH_PLANS as RUNTIME_BATCH_PLANS,
    HELPER_OPPORTUNITY_EVIDENCE_STATES as HELPER_OPPORTUNITY_EVIDENCE_STATES,
    RuntimePlanError as RuntimePlanError,
    _REJECTION_STAGE as _REJECTION_STAGE,
    primary_rejection_reason as primary_rejection_reason,
    _text as _text,
    _integer as _integer,
    _estimation_metadata as _estimation_metadata,
    _sha256 as _sha256,
)
from .plan_contracts.operators import (
    _OPERATOR_JSON_CACHE_LIMIT as _OPERATOR_JSON_CACHE_LIMIT,
    _OPERATOR_JSON_PLACEHOLDER as _OPERATOR_JSON_PLACEHOLDER,
    _operator_json_cache as _operator_json_cache,
    _operator_json_cache_lock as _operator_json_cache_lock,
    RuntimeExecutionContract as RuntimeExecutionContract,
    RuntimeOperatorAssignment as RuntimeOperatorAssignment,
    _canonical_operator_json as _canonical_operator_json,
)
from .plan_contracts.phone import (
    RuntimePhoneShard as RuntimePhoneShard,
    phone_session_map_sha256 as phone_session_map_sha256,
    phone_session_assignment_sha256 as phone_session_assignment_sha256,
    PhoneSessionReplacementAuthorization as PhoneSessionReplacementAuthorization,
)
from .plan_contracts.remote_resident import (
    REMOTE_RESIDENT_FFN_SCHEMA as REMOTE_RESIDENT_FFN_SCHEMA,
    REMOTE_RESIDENT_FFN_DTYPES as REMOTE_RESIDENT_FFN_DTYPES,
    RuntimeRemoteResidentSession as RuntimeRemoteResidentSession,
    RuntimeRemoteResidentFfn as RuntimeRemoteResidentFfn,
    remote_resident_tensor_ids as remote_resident_tensor_ids,
)
from .plan_contracts.transitions import (
    RuntimeResidencyEviction as RuntimeResidencyEviction,
    RuntimeTransitionPlan as RuntimeTransitionPlan,
    RuntimeTransitionReceipt as RuntimeTransitionReceipt,
)
from .plan_contracts.execution import (
    RuntimeExecutionReceipt as RuntimeExecutionReceipt,
    RuntimeExecutionPlan as RuntimeExecutionPlan,
    HelperOpportunity as HelperOpportunity,
    helper_preparation_changed_session_ids as helper_preparation_changed_session_ids,
    RuntimeHelperExecutionEnvelope as RuntimeHelperExecutionEnvelope,
)
from .plan_contracts.costs import (
    RuntimeTransferCost as RuntimeTransferCost,
    AutomatedRouteCost as AutomatedRouteCost,
)
from .plan_contracts.candidates import (
    AutomatedRouteCandidate as AutomatedRouteCandidate,
    _CANDIDATE_METADATA_REQUIRED as _CANDIDATE_METADATA_REQUIRED,
    _CANDIDATE_METADATA_LEGACY_EPOCH as _CANDIDATE_METADATA_LEGACY_EPOCH,
    _CANDIDATE_METADATA_PUBLISHED_EPOCH as _CANDIDATE_METADATA_PUBLISHED_EPOCH,
    _CANDIDATE_METADATA_CROSS_SHAPE as _CANDIDATE_METADATA_CROSS_SHAPE,
    _CANDIDATE_METADATA_INDEPENDENT as _CANDIDATE_METADATA_INDEPENDENT,
    _validate_placement_resolution as _validate_placement_resolution,
    _validate_route_template_shape_metadata as _validate_route_template_shape_metadata,
    _validate_search_metadata_counts as _validate_search_metadata_counts,
    _validate_published_epoch_identity as _validate_published_epoch_identity,
    _validate_published_epoch_collections as _validate_published_epoch_collections,
    _validate_published_epoch_metadata as _validate_published_epoch_metadata,
    _validate_optional_candidate_metadata as _validate_optional_candidate_metadata,
    _validate_candidate_search_metadata as _validate_candidate_search_metadata,
    _candidate_set_generation_sha256 as _candidate_set_generation_sha256,
    AutomatedCandidateSet as AutomatedCandidateSet,
)

__all__ = [
    'AUTOMATED_CANDIDATE_SET_SCHEMA',
    'AutomatedCandidateSet',
    'AutomatedRouteCandidate',
    'AutomatedRouteCost',
    'HELPER_OPPORTUNITY_EVIDENCE_STATES',
    'HelperOpportunity',
    'PhoneSessionReplacementAuthorization',
    'REMOTE_RESIDENT_FFN_DTYPES',
    'REMOTE_RESIDENT_FFN_SCHEMA',
    'ROUTE_MATURITY_STATES',
    'RUNTIME_BATCH_PLANS',
    'RUNTIME_EXECUTION_MODES',
    'RUNTIME_EXECUTION_PLAN_SCHEMA',
    'RUNTIME_EXECUTION_RECEIPT_STATUSES',
    'RUNTIME_TRANSITION_RECEIPT_STATUSES',
    'RuntimeExecutionContract',
    'RuntimeExecutionPlan',
    'RuntimeExecutionReceipt',
    'RuntimeExecutorBinding',
    'RuntimeHelperExecutionEnvelope',
    'RuntimeMemoryDemand',
    'RuntimeOperatorAssignment',
    'RuntimePhoneShard',
    'RuntimePlanError',
    'RuntimeRemoteResidentFfn',
    'RuntimeRemoteResidentSession',
    'RuntimeResidencyEviction',
    'RuntimeTransferCost',
    'RuntimeTransitionPlan',
    'RuntimeTransitionReceipt',
    'SPLIT_AXES',
    '_CANDIDATE_METADATA_CROSS_SHAPE',
    '_CANDIDATE_METADATA_INDEPENDENT',
    '_CANDIDATE_METADATA_LEGACY_EPOCH',
    '_CANDIDATE_METADATA_PUBLISHED_EPOCH',
    '_CANDIDATE_METADATA_REQUIRED',
    '_OPERATOR_JSON_CACHE_LIMIT',
    '_OPERATOR_JSON_PLACEHOLDER',
    '_REJECTION_STAGE',
    '_candidate_set_generation_sha256',
    '_canonical_operator_json',
    '_estimation_metadata',
    '_integer',
    '_operator_json_cache',
    '_operator_json_cache_lock',
    '_sha256',
    '_text',
    '_validate_candidate_search_metadata',
    '_validate_optional_candidate_metadata',
    '_validate_placement_resolution',
    '_validate_published_epoch_collections',
    '_validate_published_epoch_identity',
    '_validate_published_epoch_metadata',
    '_validate_route_template_shape_metadata',
    '_validate_search_metadata_counts',
    'helper_preparation_changed_session_ids',
    'phone_session_assignment_sha256',
    'phone_session_map_sha256',
    'primary_rejection_reason',
    'remote_resident_tensor_ids',
]
