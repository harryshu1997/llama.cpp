#!/usr/bin/env python3
"""S14 CP1 measured-profile adapter.

Bridges the FROZEN CP0-c island catalog + the real mix-v1 trace into the S12-V2
two-level reducer (two_level_vq.Simulator), which PLAN section 8 names as the
state reducer S14 extends rather than rewrites. The reducer's own validate_profile
locks correctness_status to ASSUMED_SYNTHETIC_ONLY; this adapter instead emits a
CP1 MEASURED profile (honest schema + provenance binding) and translates it into
the plain catalog dict the Simulator consumes, so measured LOWER_BOUND data is
never mislabelled as synthetic. The Simulator mechanics are reused unchanged.

Honest scope (see RESULTS.md):
  - Only gemma_head_0_2 @ OP15/HTP0 has a measured phone latency, so ALL 177
    mix-v1 requests share that one island (their LLM head is the shared
    phone-candidate). bge_encoder_0_12 is EXCLUDED from the runtime profile: it
    is correctness-certified but has no measured phone latency, so it cannot be
    timed. This is a single-phone-island run, not two-island-diverse.
  - Every catalog row is dispatch-INELIGIBLE (no coherent latency+placement-cert
    run). CP1 therefore runs on LOWER_BOUND latency and its result is MECHANICS,
    not a certified performance claim.
  - server_full_us / server_tail_us / phone_us are all bound to in-tree S11
    artifacts (island_catalog.json + FUNCTIONAL_RESULT.json). deadline/priority
    are SYNTHETIC scheduling parameters (the real trace carries none); they are
    labelled as such and never presented as real SLOs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
S12_DIR = HERE.parent / "s12_trace_vq"

# Cross-spike reuse: S14 builds on the S12-V2 reducer (PLAN section 8).
sys.path.insert(0, str(S12_DIR))
sys.path.insert(0, str(HERE))

import catalog_common as cc  # noqa: E402

CP1_PROFILE_SCHEMA = "s14-cp1-measured-profile-v1"

CATALOG_PATH = HERE / "island_catalog.json"
FROZEN_CATALOG_HASH = "sha256:3cf13792185f41919af3a6ee47fdb41eb236b93e0967b31082ab062dd3d4b3d5"

# model_class in the mix-v1 trace -> the catalog model that owns the shared
# gemma LLM head island. Both generation and rag route their LLM head here.
MODEL_CLASS_TO_ISLAND = {
    "large_text_generation": "gemma_head_0_2",
    "rag": "gemma_head_0_2",
}

# server tail (server does [2,48) after the phone does [0,2)); bound to the S11
# fixed-route treatment host_us median in FUNCTIONAL_RESULT.json. server_full is
# the catalog's A6000 server_control p50. The 215 us gap is the 2-layer head's
# server compute -- deliberately tiny, which is the whole point CP1 quantifies.
SERVER_TAIL_US = 156736
FUNCTIONAL_RESULT_REL = "research_dev/spikes/s11_fixed_route_poc/FUNCTIONAL_RESULT.json"

# transfer/verify/prepare are REQUIRED by the reducer route shape but UNUSED in
# CP1's static-residency configs (weights pre-staged before the paid interval).
# The runtime asserts zero transfer activity, proving they were never read. Real
# S9 transfer/verify/prepare costs apply only to CP3 dynamic streaming.
STATIC_UNUSED_US = 1


class CP1Error(Exception):
    """Fail-closed CP1 adapter violation."""


def load_frozen_catalog(path: str | Path = CATALOG_PATH) -> dict[str, Any]:
    catalog = cc.load_json(path)
    recomputed = cc.sha256_of({k: v for k, v in catalog.items() if k != "catalog_hash"})
    if recomputed != catalog.get("catalog_hash"):
        raise CP1Error(f"catalog_hash mismatch: recomputed {recomputed}")
    if catalog.get("catalog_hash") != FROZEN_CATALOG_HASH:
        raise CP1Error(
            f"catalog is not the pinned CP0-c freeze (expected {FROZEN_CATALOG_HASH}, got {catalog.get('catalog_hash')})"
        )
    return catalog


def _row(catalog: dict[str, Any], island_id: str, device_backend_id: str) -> dict[str, Any]:
    for r in catalog["profile_rows"]:
        if r["island_id"] == island_id and r["device_backend_id"] == device_backend_id:
            return r
    raise CP1Error(f"no row for {island_id} @ {device_backend_id}")


def _island(catalog: dict[str, Any], island_id: str) -> dict[str, Any]:
    for d in catalog["islands"]:
        if d["island_id"] == island_id:
            return d
    raise CP1Error(f"no island descriptor for {island_id}")


def build_measured_profile(catalog: dict[str, Any]) -> dict[str, Any]:
    """CP1 measured profile: honest field names + provenance, one entry for the
    single measured-phone island (gemma_head_0_2 @ OP15/HTP0)."""
    island_id = "gemma_head_0_2"
    row = _row(catalog, island_id, "OP15/HTP0")
    desc = _island(catalog, island_id)
    if row["p50_us"] is None:
        raise CP1Error(f"{island_id}: no measured phone latency; cannot enter the runtime")
    server_full = row["server_control"]["p50_us"]
    profile = {
        "schema": CP1_PROFILE_SCHEMA,
        "scope": "CP1_STATIC_MIXED_RUNTIME_MECHANICS_NO_ENERGY",
        "energy_status": "NOT_RUN",
        "latency_class": "MEASURED_LOWER_BOUND",
        "source_catalog_hash": catalog["catalog_hash"],
        "server_tail_provenance": FUNCTIONAL_RESULT_REL,
        "excluded_islands": [
            {"island_id": "bge_encoder_0_12",
             "reason": "correctness-certified but no measured phone latency; cannot be timed"}
        ],
        "islands": [
            {
                "island_id": island_id,
                "model_id": desc["model_id"],
                "model_version": desc["model_version"],
                "weight_bytes": row["resident_bytes"],
                "weight_identity": desc["weight_set_id"],
                "server_full_us": server_full,
                "server_tail_us": SERVER_TAIL_US,
                "state_bytes": row["state_bytes"],
                "eligibility": "INELIGIBLE_lower_bound_no_placement_cert",
                "routes": [
                    {
                        "device_id": "OP15",
                        "backend": "HTP0",
                        "correctness_status": "MEASURED_LOWER_BOUND",
                        "measured_phone_us": row["p50_us"],
                        "boundary_in_bytes": row["boundary_in_bytes"],
                        "boundary_out_bytes": row["boundary_out_bytes"],
                    }
                ],
            }
        ],
    }
    if profile["islands"][0]["server_tail_us"] > server_full:
        raise CP1Error("server_tail exceeds server_full")
    return profile


def to_reducer_catalog(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Translate the CP1 measured profile into the plain dict shape
    two_level_vq.Simulator consumes (build_catalog output shape). The Simulator's
    internal key names retain the 'synthetic_' prefix from its synthetic origin;
    we populate them with MEASURED LOWER_BOUND values. Honesty lives in the CP1
    profile + correctness_status, not in the reducer's internal key names."""
    catalog: dict[str, dict[str, Any]] = {}
    for isl in profile["islands"]:
        routes = {}
        for r in isl["routes"]:
            routes[r["device_id"]] = {
                "device_id": r["device_id"],
                "backend": r["backend"],
                "correctness_status": r["correctness_status"],
                "synthetic_phone_us": r["measured_phone_us"],
                "transfer_us": STATIC_UNUSED_US,
                "verify_us": STATIC_UNUSED_US,
                "prepare_us": STATIC_UNUSED_US,
            }
        catalog[isl["island_id"]] = {
            "model_id": isl["model_id"],
            "island_id": isl["island_id"],
            "weight_bytes": isl["weight_bytes"],
            "weight_identity": isl["weight_identity"],
            "synthetic_server_full_us": isl["server_full_us"],
            "synthetic_server_tail_us": isl["server_tail_us"],
            "device_routes": [dict(v) for v in routes.values()],
            "routes": routes,
        }
    return catalog


def build_reducer_trace(
    mix_records: list[dict[str, Any]],
    reducer_catalog: dict[str, dict[str, Any]],
    deadline_factor: int,
    arrival_scale_num: int = 1,
    arrival_scale_den: int = 1,
) -> list[dict[str, Any]]:
    """Map mix-v1 request records to the reducer trace shape. arrival_us is the
    real t_us (optionally compressed by a labelled synthetic factor for the load
    stress sweep). deadline/priority are SYNTHETIC scheduling parameters."""
    if arrival_scale_num < 1 or arrival_scale_den < 1:
        raise CP1Error("arrival scale must be positive")
    trace: list[dict[str, Any]] = []
    prev = -1
    for rec in mix_records:
        model_class = rec["model_class"]
        if model_class not in MODEL_CLASS_TO_ISLAND:
            raise CP1Error(f"unmapped model_class {model_class!r}")
        island_id = MODEL_CLASS_TO_ISLAND[model_class]
        island = reducer_catalog[island_id]
        arrival = (int(rec["t_us"]) * arrival_scale_num) // arrival_scale_den
        if arrival < prev:
            raise CP1Error("arrival ordering broke after scaling")
        prev = arrival
        deadline = arrival + deadline_factor * island["synthetic_server_full_us"]
        trace.append(
            {
                "event_id": rec["event_id"],
                "arrival_us": arrival,
                "deadline_us": deadline,
                "priority_class": 0,
                "model_id": island["model_id"],
                "island_id": island_id,
                "provenance": "semi_synthetic",
            }
        )
    if not trace:
        raise CP1Error("empty trace")
    return trace


def build_reducer_config(
    reducer_catalog: dict[str, dict[str, Any]],
    trace: list[dict[str, Any]],
    policies: tuple[str, ...],
    queue_limit: int,
    horizon_slack_full: int = 20,
) -> dict[str, Any]:
    """Two-device config: OP15 holds the measured island (static + dynamic
    initial = pre-staged READY); OP12 is idle (no measured island). Horizon
    extends past the last arrival so every admitted request can terminalize."""
    island_id = "gemma_head_0_2"
    weight = reducer_catalog[island_id]["weight_bytes"]
    server_full = reducer_catalog[island_id]["synthetic_server_full_us"]
    last_arrival = trace[-1]["arrival_us"]
    horizon = last_arrival + horizon_slack_full * server_full
    # 10 GB usable phone RAM ceiling (op15 ~10G / op12 ~10G per prior device runs).
    cap = 10 * 1024 * 1024 * 1024
    if weight > cap:
        raise CP1Error("island weight exceeds phone capacity")
    return {
        "schema": "s12-two-level-config-v1",
        "scope": "SYNTHETIC_MIXED_RESIDENCY_MECHANICS_ONLY",
        "energy_status": "NOT_RUN",
        "horizon_us": horizon,
        "queue_limit": queue_limit,
        "server_slots": 1,
        "policies": list(policies),
        "profile_path": "unused_direct_drive",
        "trace_path": "unused_direct_drive",
        "devices": [
            {
                "device_id": "OP15",
                "boot_epoch": 1,
                "capacity_bytes": cap,
                "activation_slots": 1,
                "prefetch_queue_limit": 1,
                "static_islands": [island_id],
                "dynamic_initial_islands": [island_id],
            },
            {
                "device_id": "OP12",
                "boot_epoch": 1,
                "capacity_bytes": cap,
                "activation_slots": 1,
                "prefetch_queue_limit": 1,
                "static_islands": [],
                "dynamic_initial_islands": [],
            },
        ],
        "scheduler": {
            "min_prefetch_score_us": 1,
            "prefetch_min_observations": 1,
            "replicate_min_queue": 1,
            "reuse_horizon_requests": 1,
        },
    }
