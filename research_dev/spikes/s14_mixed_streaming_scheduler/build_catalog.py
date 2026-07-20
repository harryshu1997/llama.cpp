#!/usr/bin/env python3
"""Build the S14 CP0-c frozen candidate island catalog.

Deterministic: emits island_catalog.json (canonical serialization) from the
constants below plus the on-disk digests of the bound in-tree evidence files.
A rerun is byte-identical as long as the bound evidence files are unchanged
(they are frozen S11/S8 artifacts). The independent validate_catalog.py
re-checks every value this builder asserts.

Scope: MECHANICS, NO ENERGY. This freezes the finite candidate GEOMETRY (which
islands exist, their identity/boundary/state/route) plus the evidence that
already exists, with every gap labeled. It does NOT run fresh device work.

Honest evidence state (see CATALOG.md). NO row is scheduler-eligible: no single
run yields a coherent per-request latency AND a same-run HTP no-fallback
certificate.
  - gemma_head_0_2 @ OP15/HTP0: measured stage latency + memory + boundary +
    exact-output correctness (S11 batched sweep). fallback/kernel_provenance are
    UNKNOWN: the sweep disclaims a placement certificate, and exact tokens do not
    prove HTP execution. The E0 Long Readiness run DID certify HTP0 no-fallback
    for this route, but on a different binary/run with a B=8-group latency, so it
    is not bound to this row. verdict LOWER_BOUND; eligibility INELIGIBLE.
  - bge_encoder_0_12 @ OP15/HTP0 and OP12/HTP0: measured pooled-CLS cosine
    correctness (0.9973) + no-fallback (Gate-1 op-support gate, properly bound).
    Per-encode latency is UNMEASURED (Gate-1 captured only load-dominated
    whole-invocation walls), so eligibility INELIGIBLE_NO_LATENCY.
  - gemma_layer_2_3 @ OP12/HTP0: exact-output correctness point from the S11-B
    two-phone SERIAL run (PLAN section 2 excludes serial chains). No isolated
    latency and no OP12/v75 placement certificate; INELIGIBLE.
  - gemma_head_0_3 @ OP15/HTP0: DECLARED, unmeasured. Freezes a larger
    homogeneous-SWA head candidate so CP1/CP2 cannot invent a new cut later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import catalog_common as cc

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = Path(__file__).resolve().parent / "island_catalog.json"

# ---- model constants ----
GEMMA_MODEL_ID = "gemma-4-12b-it-f16"
GEMMA_MODEL_VERSION = "sha256:bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a"
GEMMA_N_LAYER = 48
GEMMA_EMBD = 3840
GEMMA_ELEM = 4  # f32 activations at the layer boundary
GEMMA_HOST_PATH = "/home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf"

BGE_MODEL_ID = "bge-small-en-v1.5-f16"
BGE_MODEL_VERSION = "sha256:4cd429b83d2805e4028d96f6174b153bc87657808ba9f3b3c4f84d374b481e03"
BGE_N_LAYER = 12
BGE_HIDDEN = 384
BGE_ELEM = 4  # f32 activations at the boundary
BGE_HOST_PATH = "models/bge-small-en-v1.5-f16.gguf"  # converted in Gate-1; not retained on host, digest pinned above

# ---- binary/build identities: per-role, NOT a single "exact artifact" ----
# Each gemma phone row pins the binary that produced ITS OWN measured field, not
# one uniform label. The batched sweep and the repaired-V2 checkpoint are
# DIFFERENT binaries (s11_batched_route_poc/MANIFEST.md).
#   sweep    : produced s11_batched_route.json B=1..16 phone stage latencies.
#   repaired : produced the exact-output two-phone chain AND the E0 Long
#              Readiness on-device HTP0 placement certificate.
GEMMA_SWEEP_BUILD = "sha256:786defa77b157aad6745aa2b37a0e1d11a2cb89a31ac938e5ea81d0694717c4d"     # host llama-layersplit (sweep)
GEMMA_SWEEP_DEVICE = "sha256:9472537438612242b5ce9ce25aa739e531b0f1013af7245bfd5a0c3703728979"    # Android llama-layersplit (sweep)
GEMMA_REPAIRED_BUILD = "sha256:914fd377acb5f604bbcad225ceb650b093adbcfa2faaa07e2347260a39140eb4"  # host llama-layersplit (repaired V2)
GEMMA_REPAIRED_DEVICE = "sha256:e518ed4534b38d75f476d6bd9afdc012742ec3e4503aacceddd48ec650b20fa1" # Android llama-layersplit (repaired V2)
BGE_BUILD_HASH = "sha256:64cc0e4e444cf14bdba2e206ad1d3691f02ac92987a1f9c00fb406d85e9cc25c"        # Android llama-embedding
BGE_V81_SKEL = "sha256:874e2cd4099b0a60f6f80d74483c72d991b69f7eaac7e6f49852ad6db606a7c7"          # OP15 v81 HTP skel
BGE_V75_SKEL = "sha256:d4104633f1ba3adce260eae97698b06e451e94a2b43d7b1bd4b7103d51ab01a8"          # OP12 v75 HTP skel

MIB = 1024 * 1024

# ---- bound in-tree evidence files (repo-relative) ----
SOURCES = [
    ("S11_BATCHED_PROFILE", "research_dev/spikes/s12_trace_vq/profiles/s11_batched_route.json",
     "gemma_head_0_2 OP15/HTP0 measured stage latency, activation bytes, HBM relief (B=1..16)"),
    ("S11_FUNCTIONAL_RESULT", "research_dev/spikes/s11_fixed_route_poc/FUNCTIONAL_RESULT.json",
     "A6000 server-only baseline wall + exact token match + selected-board HBM"),
    ("S11_TWO_PHONE_RESULTS", "research_dev/spikes/s11_batched_route_poc/RESULTS.md",
     "gemma_layer_2_3 OP12 exact-output correctness + KV numbers (serial two-phone)"),
    ("GATE1_BGE_DOC", "research_dev/spikes/s8_operator_island_affinity/GATE1_HTP_BERT_OPS.md",
     "bge_encoder_0_12 pooled-CLS cosine 0.9973 + no-fallback op support, both phones"),
    ("ATLAS_MATRIX", "research_dev/spikes/s8_operator_island_affinity/ATLAS_MATRIX.md",
     "predeclared row scoping contract: layer_range, attention_class, graph_hash, shape_envelope"),
    ("S11_E0_PLACEMENT_CERT", "research_dev/spikes/s11_fixed_route_poc/RESULTS.md",
     "E0 Long Readiness: OP15 [0,2) phone layer compute on HTP0, only GET_ROWS on CPU (no-fallback ROUTE feasibility; a different run/binary than the sweep latency, so not bound to gemma_head_0_2's row)"),
]


def _digest(rel_path: str) -> str:
    return cc.sha256_file(REPO_ROOT / rel_path)


def _tensorset(name: str, dtype: str, max_bytes: int) -> dict[str, Any]:
    return {"tensors": [{"name": name, "dtype": dtype, "max_bytes": max_bytes}]}


def _descriptor(
    island_id: str,
    service_class: str,
    op_classes: list[str],
    model_id: str,
    model_version: str,
    start: int,
    end: int,
    n_layer_total: int,
    attention_class: str,
    boundary_in: dict[str, Any],
    boundary_out: dict[str, Any],
    default_state_policy: str,
    supported_routes: list[str],
    replay_boundary: str,
) -> dict[str, Any]:
    layer_range = {"start": start, "end": end, "n_layer_total": n_layer_total}
    body: dict[str, Any] = {
        "schema_version": 1,
        "island_id": island_id,
        "service_class": service_class,
        "op_classes": op_classes,
        "model_id": model_id,
        "model_version": model_version,
        "graph_hash": cc.derived_graph_hash(model_version, layer_range, attention_class),
        "layer_range": layer_range,
        "attention_class": attention_class,
        "boundary_in": boundary_in,
        "boundary_out": boundary_out,
        "weight_set_id": cc.derived_weight_set_id(model_version, layer_range),
        "default_state_policy": default_state_policy,
        "supported_routes": supported_routes,
        "replay_boundary": replay_boundary,
    }
    body["descriptor_hash"] = cc.descriptor_hash(body)
    return body


def _row(descriptor: dict[str, Any], device_backend_id: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Base profile row consistent with its descriptor; overrides fill evidence."""
    row: dict[str, Any] = {
        "schema_version": 1,
        "island_id": descriptor["island_id"],
        "island_descriptor_ref": descriptor["descriptor_hash"],
        "model_version": descriptor["model_version"],
        "graph_hash": descriptor["graph_hash"],
        "layer_range": dict(descriptor["layer_range"]),
        "attention_class": descriptor["attention_class"],
        "device_backend_id": device_backend_id,
        "opt_arch": None,
        "shape_envelope": None,
        "state_class": descriptor["default_state_policy"],
        "supported": None,
        "fallback": "unknown",
        "kernel_provenance": "unknown",
        "correctness": "unknown",
        "correctness_metric": None,
        "resident_bytes": None,
        "scratch_bytes": None,
        "state_bytes": None,
        "boundary_in_bytes": None,
        "boundary_out_bytes": None,
        "transfer_measured_us": None,
        "warmup_load_us": None,
        "p50_us": None,
        "p95_us": None,
        "p99_us": None,
        "cov": None,
        "n_proc": 0,
        "corun_partner": None,
        "corun_slowdown": None,
        "server_control": None,
        "post_transfer_slo_feasible": None,
        "server_relief": None,
        "temperature_c": None,
        "build_hash": None,
        "device_binary_hash": None,
        "artifact_paths": [],
        "artifact_hashes": [],
        "verdict": "UNKNOWN",
    }
    row.update(overrides)
    return row


def build(digests: dict[str, str]) -> dict[str, Any]:
    # ---- Gemma head [0,2) : the one measured island with clean stage latency ----
    d_g02 = _descriptor(
        island_id="gemma_head_0_2",
        service_class="generation",
        op_classes=["mul_mat", "norm", "rope", "softmax", "scale", "add", "mul"],
        model_id=GEMMA_MODEL_ID,
        model_version=GEMMA_MODEL_VERSION,
        start=0, end=2, n_layer_total=GEMMA_N_LAYER,
        attention_class="causal_local_swa",
        boundary_in=_tensorset("hidden_in", "f32", 16 * 31 * GEMMA_EMBD * GEMMA_ELEM),
        boundary_out=_tensorset("hidden_out", "f32", 16 * 31 * GEMMA_EMBD * GEMMA_ELEM),
        default_state_policy="sticky",
        supported_routes=["OP15/HTP0"],
        replay_boundary="token_history",
    )
    r_g02 = _row(
        d_g02, "OP15/HTP0",
        {
            "opt_arch": 81,
            "shape_envelope": {
                "batch_min": 1, "batch_max": 16,
                "context_min": 96, "context_max": 96,
                "prompt_min": 28, "prompt_max": 28, "dtype": "f16",
            },
            # supported/fallback/kernel_provenance are UNKNOWN for this row: its
            # bound sources (batched sweep + server functional result) certify
            # exact tokens but NOT HTP no-fallback placement. Exact tokens do not
            # prove HTP execution (a CPU fallback yields identical tokens), and
            # the sweep RESULTS.md explicitly disclaims a placement certificate.
            # The E0 Long Readiness run DID certify HTP0 no-fallback for this
            # route, but on a different binary/run with a B=8-group latency (see
            # source S11_E0_PLACEMENT_CERT + CATALOG.md); it is not this row's run.
            "supported": None,
            "fallback": "unknown",
            "kernel_provenance": "unknown",
            "correctness": "pass",
            "correctness_metric": {"kind": "exact_token_ids", "value": 1.0, "reference": "SERVER_ONLY_greedy"},
            "resident_bytes": 2925710752,
            "state_bytes": 4 * MIB,
            "boundary_in_bytes": 31 * GEMMA_EMBD * GEMMA_ELEM,
            "boundary_out_bytes": 31 * GEMMA_EMBD * GEMMA_ELEM,
            "p50_us": 154962,
            "n_proc": 1,
            "server_control": {
                "device_backend_id": "A6000/CUDA",
                "p50_us": 156951, "gpu_ms": None, "hbm_peak_bytes": 24580 * MIB,
            },
            "post_transfer_slo_feasible": False,
            "server_relief": {"gpu_ms_freed": 0.0, "hbm_bytes_freed": 860 * MIB, "hbm_bw_freed": None},
            "build_hash": GEMMA_SWEEP_BUILD,       # sweep binary produced p50=154962
            "device_binary_hash": GEMMA_SWEEP_DEVICE,
            "artifact_paths": [SOURCES[0][1], SOURCES[1][1]],
            "artifact_hashes": [digests[SOURCES[0][1]], digests[SOURCES[1][1]]],
            "verdict": "LOWER_BOUND",
        },
    )

    # ---- Gemma layer [2,3) : OP12 exact-output correctness point (serial two-phone) ----
    d_g23 = _descriptor(
        island_id="gemma_layer_2_3",
        service_class="generation",
        op_classes=["mul_mat", "norm", "rope", "softmax", "scale", "add", "mul"],
        model_id=GEMMA_MODEL_ID,
        model_version=GEMMA_MODEL_VERSION,
        start=2, end=3, n_layer_total=GEMMA_N_LAYER,
        attention_class="causal_local_swa",
        boundary_in=_tensorset("hidden_in", "f32", 31 * GEMMA_EMBD * GEMMA_ELEM),
        boundary_out=_tensorset("hidden_out", "f32", 31 * GEMMA_EMBD * GEMMA_ELEM),
        default_state_policy="sticky",
        supported_routes=["OP12/HTP0"],
        replay_boundary="token_history",
    )
    r_g23 = _row(
        d_g23, "OP12/HTP0",
        {
            "opt_arch": 75,
            "shape_envelope": {
                "batch_min": 1, "batch_max": 1,
                "context_min": 96, "context_max": 96,
                "prompt_min": 28, "prompt_max": 28, "dtype": "f16",
            },
            # As with gemma_head_0_2: exact-output correctness is bound (repaired
            # two-phone chain), but no in-tree artifact certifies OP12/v75 HTP
            # no-fallback placement for [2,3) (ATLAS_MATRIX has no such row), so
            # fallback/kernel_provenance/supported stay UNKNOWN.
            "supported": None,
            "fallback": "unknown",
            "kernel_provenance": "unknown",
            "correctness": "pass",
            "correctness_metric": {"kind": "exact_token_ids", "value": 1.0, "reference": "SERVER_ONLY_greedy"},
            "state_bytes": 2 * MIB,
            "boundary_in_bytes": 31 * GEMMA_EMBD * GEMMA_ELEM,
            "boundary_out_bytes": 31 * GEMMA_EMBD * GEMMA_ELEM,
            "n_proc": 1,
            "build_hash": GEMMA_REPAIRED_BUILD,     # repaired two-phone chain produced the exact-output correctness
            "device_binary_hash": GEMMA_REPAIRED_DEVICE,
            "artifact_paths": [SOURCES[2][1]],
            "artifact_hashes": [digests[SOURCES[2][1]]],
            "verdict": "LOWER_BOUND",
        },
    )

    # ---- Gemma head [0,3) : declared larger homogeneous-SWA candidate, unmeasured ----
    d_g03 = _descriptor(
        island_id="gemma_head_0_3",
        service_class="generation",
        op_classes=["mul_mat", "norm", "rope", "softmax", "scale", "add", "mul"],
        model_id=GEMMA_MODEL_ID,
        model_version=GEMMA_MODEL_VERSION,
        start=0, end=3, n_layer_total=GEMMA_N_LAYER,
        attention_class="causal_local_swa",
        boundary_in=_tensorset("hidden_in", "f32", 16 * 31 * GEMMA_EMBD * GEMMA_ELEM),
        boundary_out=_tensorset("hidden_out", "f32", 16 * 31 * GEMMA_EMBD * GEMMA_ELEM),
        default_state_policy="sticky",
        supported_routes=["OP15/HTP0"],
        replay_boundary="token_history",
    )
    r_g03 = _row(
        d_g03, "OP15/HTP0",
        {
            "opt_arch": 81,
            "shape_envelope": {
                "batch_min": 1, "batch_max": 16,
                "context_min": 96, "context_max": 96,
                "prompt_min": 28, "prompt_max": 28, "dtype": "f16",
            },
            "build_hash": GEMMA_REPAIRED_BUILD,     # declared future measurement on the current repaired build
            "device_binary_hash": GEMMA_REPAIRED_DEVICE,
            "verdict": "UNKNOWN",
        },
    )

    # ---- BGE encoder [0,12) : correctness certified both phones, latency unmeasured ----
    # get_rows (f16 embed table) is a declared host CPU pre-stage OUTSIDE the island,
    # so the island itself is fallback=none. boundary_in is the F32 token embeddings.
    d_bge = _descriptor(
        island_id="bge_encoder_0_12",
        service_class="encoder",
        op_classes=["mul_mat", "norm", "softmax", "gelu", "scale", "add", "mul", "cpy"],
        model_id=BGE_MODEL_ID,
        model_version=BGE_MODEL_VERSION,
        start=0, end=12, n_layer_total=BGE_N_LAYER,
        attention_class="bidirectional_none",
        boundary_in=_tensorset("embedded_tokens", "f32", 512 * BGE_HIDDEN * BGE_ELEM),
        boundary_out=_tensorset("pooled_cls", "f32", BGE_HIDDEN * BGE_ELEM),
        default_state_policy="stateless",
        supported_routes=["OP15/HTP0", "OP12/HTP0"],
        replay_boundary="stage_input",
    )
    bge_env = {
        "batch_min": 1, "batch_max": 1,
        "context_min": 0, "context_max": 512,
        "prompt_min": 0, "prompt_max": 512, "dtype": "f16",
    }
    r_bge_op15 = _row(
        d_bge, "OP15/HTP0",
        {
            "opt_arch": 81,
            "shape_envelope": bge_env,
            "supported": True,
            "fallback": "none",
            "kernel_provenance": "explicit_attention",
            "correctness": "pass",
            "correctness_metric": {"kind": "pooled_cls_cosine", "value": 0.997331, "reference": "cpu_f32_ref"},
            "boundary_in_bytes": 64 * BGE_HIDDEN * BGE_ELEM,
            "boundary_out_bytes": BGE_HIDDEN * BGE_ELEM,
            "n_proc": 1,
            "build_hash": BGE_BUILD_HASH,
            "device_binary_hash": BGE_V81_SKEL,
            "artifact_paths": [SOURCES[3][1]],
            "artifact_hashes": [digests[SOURCES[3][1]]],
            "verdict": "LOWER_BOUND",
        },
    )
    r_bge_op12 = _row(
        d_bge, "OP12/HTP0",
        {
            "opt_arch": 75,
            "shape_envelope": bge_env,
            "supported": True,
            "fallback": "none",
            "kernel_provenance": "explicit_attention",
            "correctness": "pass",
            "correctness_metric": {"kind": "pooled_cls_cosine", "value": 0.997328, "reference": "cpu_f32_ref"},
            "boundary_in_bytes": 64 * BGE_HIDDEN * BGE_ELEM,
            "boundary_out_bytes": BGE_HIDDEN * BGE_ELEM,
            "n_proc": 1,
            "build_hash": BGE_BUILD_HASH,
            "device_binary_hash": BGE_V75_SKEL,
            "artifact_paths": [SOURCES[3][1]],
            "artifact_hashes": [digests[SOURCES[3][1]]],
            "verdict": "LOWER_BOUND",
        },
    )

    islands = [d_g02, d_g23, d_g03, d_bge]
    rows = [r_g02, r_g23, r_g03, r_bge_op15, r_bge_op12]

    source_bindings = [
        {"kind": kind, "path": path, "sha256": digests[path], "proves": proves}
        for (kind, path, proves) in SOURCES
    ]

    catalog: dict[str, Any] = {
        "schema_version": 1,
        "catalog_id": "s14-cp0c-island-catalog-v1",
        "scope": "CP0C_ISLAND_CATALOG_MECHANICS_NO_ENERGY",
        "frozen_status": "FROZEN_BEFORE_SCHEDULER_RESULTS",
        "energy_status": "NOT_RUN",
        "git_head": "5e92adf00",
        "eligibility_rule": (
            "A row is SCHEDULER_ELIGIBLE iff verdict in {PASS, LOWER_BOUND} and "
            "correctness==pass and fallback==none and supported==true and p50_us "
            "is non-null and boundary_in_bytes and boundary_out_bytes are non-null. "
            "If eligible with n_proc<7 it is ELIGIBLE_COARSE (provisional, single "
            "process). Missing latency => INELIGIBLE_NO_LATENCY. verdict==UNKNOWN "
            "=> INELIGIBLE_UNMEASURED. Serial two-phone chains are excluded from "
            "the initial catalog per PLAN section 2 and appear only as declared "
            "independent-placement candidates."
        ),
        "models": [
            {"model_id": GEMMA_MODEL_ID, "model_version": GEMMA_MODEL_VERSION,
             "n_layer_total": GEMMA_N_LAYER, "host_model_path": GEMMA_HOST_PATH,
             "notes": "SWA/global interleave from gguf sliding_window_pattern; layers 0-2 are causal_local_swa."},
            {"model_id": BGE_MODEL_ID, "model_version": BGE_MODEL_VERSION,
             "n_layer_total": BGE_N_LAYER, "host_model_path": BGE_HOST_PATH,
             "notes": "converted f16 in Gate-1; host file not retained, digest pinned; get_rows f16 embd on CPU pre-stage."},
        ],
        "islands": islands,
        "profile_rows": rows,
        "source_bindings": source_bindings,
    }
    catalog["catalog_hash"] = cc.sha256_of({k: v for k, v in catalog.items() if k != "catalog_hash"})
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the S14 CP0-c frozen island catalog")
    parser.add_argument("--output", default=str(CATALOG_PATH))
    parser.add_argument("--print-hash", action="store_true")
    args = parser.parse_args()

    digests = {path: _digest(path) for (_, path, _) in SOURCES}
    catalog = build(digests)
    data = cc.canonical_json(catalog)
    Path(args.output).write_bytes(data)
    if args.print_hash:
        print(catalog["catalog_hash"])
    else:
        print(f"wrote {args.output}")
        print(f"catalog_hash {catalog['catalog_hash']}")
        print(f"islands {len(catalog['islands'])} rows {len(catalog['profile_rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
