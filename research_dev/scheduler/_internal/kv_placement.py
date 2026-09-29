"""Per-layer F16 KV placement and phase-safe admission to the existing memory ledger.

This is a fixed-capacity context plan, not automatic KV growth or migration. The caller
supplies weight/workspace peaks separately from KV and must not count resident bytes twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .model_manifest import ModelManifest
from .runtime_cost import RuntimeMemoryDemand
from .runtime_resources import (
    RuntimeMemoryLedger, RuntimeResourceError, RuntimeRemoteResidentAccounting,
)
from .types import canonical_sha256


def _bytes(name, value):
    if type(value) is not int or value < 0:
        raise RuntimeResourceError(name + " must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class LayerKvPlacement:
    artifact_sha256: str
    context_size: int
    parallel: int
    cpu_layers: tuple[int, ...]
    bytes_by_pool: Mapping[str, int]
    layers: tuple[tuple[int, str, int], ...]
    device_cells: tuple[tuple[int, int], ...] = ()
    scratch_cells: int = 0

    def to_json(self):
        result = {
            "schema": "scheduler-layer-kv-placement-v1",
            "artifact_sha256": self.artifact_sha256,
            "context_size": self.context_size,
            "parallel": self.parallel,
            "type_k": "f16", "type_v": "f16", "kv_unified": True,
            "cpu_layers": list(self.cpu_layers),
            "bytes_by_pool": dict(self.bytes_by_pool),
            "layers": [{"layer": il, "pool": pool, "bytes": size} for il, pool, size in self.layers],
        }
        if self.device_cells:
            result.update(schema="scheduler-layer-kv-placement-v2", device_cells=dict(self.device_cells),
                          scratch_cells=self.scratch_cells, split_prefill_threshold=32)
        return result

    @property
    def plan_sha256(self):
        return canonical_sha256(self.to_json())


def plan_layer_kv(
    manifest: ModelManifest, *, context_size: int, parallel: int, ubatch_size: int,
    default_pool_by_layer: Mapping[int, str], host_pool: str,
    kv_budget_by_pool: Mapping[str, int], shared_kv_sources: Mapping[int, int] | None = None,
    kv_device_cells: Mapping[int, int] | None = None,
) -> LayerKvPlacement:
    """Spill complete global-attention caches to host; retain SWA on its default device.

    Budgets are available KV bytes AFTER weights, workspace and safety reserves. Shared-KV
    consumers are grouped explicitly with their allocating source. No future arrivals are used.
    """
    for name, value in (("context", context_size), ("parallel", parallel), ("ubatch", ubatch_size)):
        if type(value) is not int or value <= 0:
            raise RuntimeResourceError(name + " must be positive")
    if context_size > manifest.context_length:
        raise RuntimeResourceError("context exceeds the artifact's validated context limit")
    context_size = (context_size + 255) // 256 * 256
    if context_size > manifest.context_length:
        raise RuntimeResourceError("padded context exceeds the artifact's context limit")
    expected = set(range(manifest.block_count))
    if set(default_pool_by_layer) != expected:
        raise RuntimeResourceError("KV default placement does not cover every layer")
    budgets = {pool: _bytes("KV budget", value) for pool, value in kv_budget_by_pool.items()}
    if host_pool not in budgets or any(pool not in budgets for pool in default_pool_by_layer.values()):
        raise RuntimeResourceError("KV pool budget is missing")
    sharing = dict(shared_kv_sources or {})
    if any(il not in expected or source not in expected or source >= il or source in sharing
           for il, source in sharing.items()):
        raise RuntimeResourceError("shared KV sources must be preceding allocating layers")
    groups = {il: [il] for il in sorted(expected - sharing.keys())}
    for il, source in sharing.items():
        groups[source].append(il)
    prefixes = dict(kv_device_cells or {})
    if any(type(il) is not int or il not in expected or type(cells) is not int
           or cells < 0 or cells > context_size or cells % 256 for il, cells in prefixes.items()):
        raise RuntimeResourceError("KV device prefixes must be valid layers and aligned cell counts")
    if prefixes and (sharing or any(manifest._layer_is_sliding(il) for il in expected)):
        raise RuntimeResourceError("split KV requires non-shared, non-SWA attention")
    pools, sizes = {}, {}
    split_sizes = {}
    used = {pool: 0 for pool in budgets}
    for source, members in groups.items():
        pools[source] = default_pool_by_layer[source]
        sliding = manifest._layer_is_sliding(source)
        if any(manifest._layer_is_sliding(il) != sliding for il in members):
            raise RuntimeResourceError("shared KV layers disagree on attention type")
        cells = context_size
        if sliding:
            cells = min(context_size, ((manifest.sliding_window * parallel + ubatch_size + 255) // 256) * 256)
        key, value = manifest._layer_kv_lengths(source)
        sizes[source] = cells * manifest._layer_head_count_kv(source) * (key + value) * 2
        used[pools[source]] += sizes[source]
        if source in prefixes:
            prefix = prefixes[source]
            if pools[source] == host_pool and prefix != 0:
                raise RuntimeResourceError("a device KV prefix requires a device parent")
            if prefix == 0:
                used[pools[source]] -= sizes[source]
                pools[source] = host_pool
                used[host_pool] += sizes[source]
            elif prefix < cells:
                per_cell = sizes[source] // cells
                # Each slice has one scratch row per ubatch token; discarded writes cannot race.
                device_bytes = (prefix + ubatch_size) * per_cell
                host_bytes = (cells - prefix + ubatch_size) * per_cell
                used[pools[source]] += device_bytes - sizes[source]
                used[host_pool] += host_bytes
                split_sizes[source] = (device_bytes, host_bytes)
    # Largest global caches first minimizes the number of cross-device attention layers.
    for pool in sorted(used):
        if pool == host_pool:
            continue
        candidates = sorted((il for il in groups if pools[il] == pool
                             and il not in prefixes and not manifest._layer_is_sliding(il)), key=lambda il: (-sizes[il], il))
        for il in candidates:
            if used[pool] <= budgets[pool]:
                break
            pools[il] = host_pool
            used[pool] -= sizes[il]
            used[host_pool] += sizes[il]
        if used[pool] > budgets[pool]:
            raise RuntimeResourceError("KV device budget cannot retain pinned or sliding-window caches")
    if used[host_pool] > budgets[host_pool]:
        raise RuntimeResourceError("CPU KV exceeds the host budget")
    layer_rows, cpu = [], []
    for source, members in groups.items():
        if source in split_sizes:
            device_bytes, host_bytes = split_sizes[source]
            layer_rows.extend(((source, pools[source], device_bytes), (source, host_pool, host_bytes)))
            continue
        for il in members:
            layer_rows.append((il, pools[source], sizes[source] if il == source else 0))
            if pools[source] == host_pool:
                cpu.append(il)
    return LayerKvPlacement(manifest.artifact_sha256, context_size, parallel, tuple(sorted(cpu)),
                            MappingProxyType(used), tuple(sorted(layer_rows)), tuple(sorted(prefixes.items())),
                            ubatch_size if split_sizes else 0)


def reserve_layer_kv(
    ledger: RuntimeMemoryLedger, owner_id: str, plan: LayerKvPlacement, snapshot, *,
    host_pool: str, base_peak_bytes_by_pool: Mapping[str, int], prefill_policy: str,
    remote_accounting: RuntimeRemoteResidentAccounting | None = None,
):
    """Atomically reserve KV plus phase peaks. Never credit a future decode-only release.

    base_peak is the unrelocated allocation peak, including weights, workspace, loading and
    required restoration headroom, excluding KV and bytes already charged in the snapshot.
    Remote-prefill credit requires the caller's physically verified omission/owner accounting.
    CPU fallback for that mode tears down KV first; it is not an in-place restoration promise.
    """
    if prefill_policy not in ("local-prefill", "restore-before-prefill", "remote-prefill"):
        raise RuntimeResourceError("prefill policy must be explicit")
    peaks = {pool: _bytes("base phase peak", value) for pool, value in base_peak_bytes_by_pool.items()}
    if host_pool not in peaks:
        raise RuntimeResourceError("host phase peak is missing")
    if prefill_policy == "remote-prefill":
        proof = remote_accounting
        if (not isinstance(proof, RuntimeRemoteResidentAccounting) or not proof.verified
                or proof.proof is None or proof.proof.warmup != "validated"
                or proof.artifact_sha256 != plan.artifact_sha256 or not proof.recovery_feasible
                or proof.proof.layer_mask != proof.layer_mask
                or proof.proof.omitted_bytes != proof.omitted_bytes_planned
                or proof.desktop_pool_id != host_pool or proof.fallback_mode != "teardown"):
            raise RuntimeResourceError("remote prefill requires verified ownership and teardown recovery")
        # Only whole unmapped pages qualify; boundary bytes and reclaimable cache are not extra credit.
        credit = min(proof.reclaimed_bytes, proof.proof.unmapped_bytes)
        if credit > peaks[host_pool]:
            raise RuntimeResourceError("remote omission exceeds host phase peak")
        peaks[host_pool] -= credit
    elif remote_accounting is not None:
        raise RuntimeResourceError("decode-only release cannot fund prefill KV allocation")
    demands = tuple(RuntimeMemoryDemand(
        demand_id=f"{plan.plan_sha256}:{pool}", resource_id=pool, kind="context-kv-and-phase-peak",
        required_bytes=peaks.get(pool, 0) + plan.bytes_by_pool.get(pool, 0), resident_bytes=0,
        lifetime="request",
    ) for pool in sorted(set(peaks) | set(plan.bytes_by_pool))
      if peaks.get(pool, 0) + plan.bytes_by_pool.get(pool, 0) > 0)
    return ledger.reserve(owner_id, demands, snapshot)
