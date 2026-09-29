"""Shared records and helpers; no independent controller state."""

from __future__ import annotations

from dataclasses import dataclass

from ..._internal.runtime_plan import RuntimePhoneShard


_MINIMUM_PERSISTENT_HASH_CACHE_BYTES = 1 << 30


@dataclass(frozen=True)
class _ShardResidencyWindow:
    """One shard's residency span in session-local generations."""

    shard: RuntimePhoneShard
    first_generation: int
    last_generation: int | None = None
