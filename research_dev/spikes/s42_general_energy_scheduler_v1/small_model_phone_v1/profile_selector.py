#!/usr/bin/env python3
"""Bind the BGE experiment to scheduler-owned lifecycle profiles."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
S42_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
sys.path[:0] = [str(REPO_ROOT), str(S42_ROOT)]

from research_dev.scheduler import (  # noqa: E402
    LifecycleProfileSet,
    LifecycleReceipt,
    ProfileBundle,
    Request,
    RequestSemantics,
    UnifiedScheduleError,
    load_lifecycle_profile_set,
)


RESULT_ROOT = HERE / "results" / "4060ti_op15_20260808"
CUDA_EPOCH_OPEN = "cuda_epoch_open"
CUDA_EPOCH_REUSED = "cuda_epoch_reused"
CUDA_EPOCH_UNKNOWN = "cuda_epoch_unknown"
EPOCH_STATES = {
    CUDA_EPOCH_OPEN,
    CUDA_EPOCH_REUSED,
    CUDA_EPOCH_UNKNOWN,
}
PROFILE_PATHS = {
    CUDA_EPOCH_OPEN: RESULT_ROOT / "SCHEDULER_PROFILE_CUDA_EPOCH_OPEN.json",
    CUDA_EPOCH_REUSED: RESULT_ROOT / "SCHEDULER_PROFILE_CUDA_EPOCH_REUSED.json",
}
WORKLOAD_ID = "bge-small-en-v1.5-q8-batch32-resident"
MODEL_ID = "bge-small-en-v1.5-q8_0-batch32-34t"
BATCH_SIZE = 32
TOKENS_PER_EMBEDDING = 34


ProfileSelectionError = UnifiedScheduleError
CudaEpochReceipt = LifecycleReceipt


@dataclass(frozen=True)
class SelectedProfile:
    epoch_state: str
    path: Path
    raw: dict[str, Any]
    profile: ProfileBundle


def _ascii(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise ProfileSelectionError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ProfileSelectionError(f"{name} must be ASCII") from exc
    return value


@lru_cache(maxsize=1)
def _load_profiles() -> tuple[dict[str, dict[str, Any]], LifecycleProfileSet]:
    raw_profiles, profile_set = load_lifecycle_profile_set(
        "bge-small-cuda-lifecycle",
        "cuda0",
        PROFILE_PATHS,
        CUDA_EPOCH_OPEN,
        frozenset({CUDA_EPOCH_REUSED}),
    )
    request_contract = {
        "batch_size": BATCH_SIZE,
        "feature": "batch32_groups",
        "minimum_groups": 1,
        "tokens_per_embedding": TOKENS_PER_EMBEDDING,
    }
    for state, raw in raw_profiles.items():
        epoch = raw.get("cuda_epoch")
        if type(epoch) is not dict or epoch.get("state") != state:
            raise ProfileSelectionError(
                "profile CUDA epoch metadata mismatch"
            )
        if raw.get("request_contract") != request_contract:
            raise ProfileSelectionError("profile request contract mismatch")
    return dict(raw_profiles), profile_set


def select_epoch_state(receipt: CudaEpochReceipt | None) -> str:
    _, profile_set = _load_profiles()
    return profile_set.select_state(receipt)


def load_selected_profile(
    receipt: CudaEpochReceipt | None,
) -> SelectedProfile:
    raw_profiles, profile_set = _load_profiles()
    state = profile_set.select_state(receipt)
    return SelectedProfile(
        state,
        PROFILE_PATHS[state],
        raw_profiles[state],
        profile_set.profiles[state],
    )


def make_request(
    request_id: str,
    batch32_groups: int,
    arrival_us: int,
    deadline_us: int,
) -> Request:
    _ascii("request id", request_id)
    if type(batch32_groups) is not int or batch32_groups < 1:
        raise ProfileSelectionError(
            "batch32_groups must be an integer >= 1"
        )
    if type(arrival_us) is not int or arrival_us < 0:
        raise ProfileSelectionError(
            "arrival_us must be an integer >= 0"
        )
    if type(deadline_us) is not int or deadline_us <= arrival_us:
        raise ProfileSelectionError("deadline_us must follow arrival_us")
    return Request(
        request_id=request_id,
        workload_id=WORKLOAD_ID,
        arrival_us=arrival_us,
        deadline_us=deadline_us,
        input_tokens=TOKENS_PER_EMBEDDING * BATCH_SIZE * batch32_groups,
        output_tokens=1,
        quality_requirement="bounded_numeric",
        features={"batch32_groups": batch32_groups},
        semantics=RequestSemantics(
            kv_owner="none",
            full_logits_required=False,
            sampler_location="none",
        ),
    )
