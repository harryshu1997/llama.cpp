#!/usr/bin/env python3
"""WS17: offline preparation of the quantized-regime experiments (all-GPU Q4 baseline, Qwen3-32B Q4_0 spill).

    tokenizer-check --gguf NEW.gguf --reference TRACE_TOKENIZER.gguf [--out CHECK.json]
    trace-variant   --source TRACE_DIR --out NEW_DIR --execution MODEL_ID=PATH[=KIND] ...
                    (or MODEL_ID=@SHA256HEX:BYTES:FILENAME[:KIND] when the artifact is not on this host)
    vram-fit        --gguf MODEL.gguf --context-size C --parallel P [--ubatch U] [--gpu-first-layer N|auto]
                    --free-vram-bytes F [--reserve-bytes R] [--workspace-bytes W] [--cuda-context-bytes X]
                    [--trace-dir DIR --tokenizer ID] [--out FIT.json]
    plans-seed      --base PLANS.json --out PLANS.json --row ARTIFACT_SHA256=ROLE:GPU_FIRST_LAYER:SEED.json ...
    template        --base TEMPLATE_DIR --spec SPEC.json --out NEW_TEMPLATE_DIR
    manifest        --gguf MODEL.gguf --model-id ID --out MANIFEST.json   (the checked GGUF manifest a model row binds)
    prior-receipt   --new-bench NEW.json --old-bench OLD.json --trace-dir DIR --tokenizer ID
                    [--assisted-new-ms-per-token X --assisted-old-ms-per-token Y] --out RECEIPT.json

``prior-receipt`` turns ``llama-bench -o json`` rows of the new artifact and of the calibrated one (same desktop,
same placement rule, ``-p 512 -n 128``) into the service-time ratio a calibration prior binds: the trace's mean
prompt and output lengths for that tokenizer weight the measured prefill and decode times. The assisted ratio
comes from measured per-token assisted decode times (E1c) when given, else it equals the desktop ratio.

``tokenizer-check`` compares every tokenizer field (vocabulary, merges, token types, special ids, chat template
is reported but not required) of two GGUF headers. The trace stores prompt TOKEN IDS from the Q4 tokenizer
files, so an execution artifact may replace the trace's only when its tokenizer is identical. Header-only
copies work (``truncate`` a downloaded header range to the full file size; tensor data is never read).

``trace-variant`` copies a trace directory and replaces only the manifest's ``model_inventory`` (execution model
id -> artifact bytes and sha256); request files, record counts and their digests are unchanged, which
``trace_inputs.validate_trace`` checks. The variant records its source manifest digest under ``ws17_variant``.

``vram-fit`` sizes one model's GPU footprint from its GGUF header: the GPU layers' weights (blk.i for
i >= gpu_first_layer, plus output.weight and output_norm), the KV cache of the GPU layers in f16 (sliding-window
layers hold ``parallel * window + ubatch`` cells, capped at the context), a compute workspace and the CUDA
context; ``auto`` returns the smallest gpu_first_layer that fits. With ``--trace-dir`` it also reports the
largest prompt + output of that tokenizer's trace rows (the context a slot actually needs).

``plans-seed`` appends a seed row for a new artifact to a desktop baseline plans file. The catalog only loads
QUALIFIED rows and ``desktop_parent_calibration.py`` only replaces existing rows, so a new artifact needs this
seed; its evidence id is the sha256 of the seed file (the vram-fit report), and the runbook replaces the row
with ``desktop_parent_calibration.py`` (live VRAM) before any measured arm.

``template`` derives a campaign template directory from a base one with a spec:
``{"schema": "ws17-regime-template-v1", "campaign_id_suffix": "...", "trace_dir": DIR,
"desktop_baseline_plans_path": PATH, "models": {"hot"|"cold": {FIELD: VALUE | null (remove), ...,
"runtime_parameters"|"phone_adapter_parameters": {KEY: VALUE | null}}}}``. Fields set to a value replace, ``null``
removes the field (or parameter); the derived models.json is re-loaded with the scheduler's own loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Mapping

_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

SPEC_SCHEMA = "ws17-regime-template-v1"
FIT_SCHEMA = "ws17-vram-fit-v1"
TOKENIZER_SCHEMA = "ws17-tokenizer-check-v1"
# must be equal: the token ids in the trace mean the same strings, and prompts get the same BOS treatment
REQUIRED_TOKENIZER_FIELDS = ("tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens",
                             "tokenizer.ggml.merges", "tokenizer.ggml.token_type", "tokenizer.ggml.add_bos_token")
# reported: special ids are set by the converter; BOS is required only when BOS is added to prompts
REPORTED_TOKENIZER_FIELDS = ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                             "tokenizer.ggml.padding_token_id", "tokenizer.ggml.add_eos_token")
TOKENIZER_FIELDS = REQUIRED_TOKENIZER_FIELDS + REPORTED_TOKENIZER_FIELDS
KV_BYTES_PER_ELEMENT = 2
GIB = 1024 ** 3


class RegimeError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegimeError(message)


def gguf_module():
    sys.path.insert(0, str(_REPO / "gguf-py"))
    import gguf  # noqa: PLC0415

    return gguf


def canonical_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=True).encode("ascii")).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


# ---------------------------------------------------------------- tokenizer-check

def tokenizer_fields(path: Path) -> dict[str, object]:
    reader = gguf_module().GGUFReader(str(path))
    out: dict[str, object] = {}
    for name in (*TOKENIZER_FIELDS, "tokenizer.chat_template"):
        field = reader.fields.get(name)
        if field is not None:
            value = field.contents()
            out[name] = value if not isinstance(value, list) else list(value)
    return out


def tokenizer_check(new: Path, reference: Path) -> dict[str, object]:
    a, b = tokenizer_fields(new), tokenizer_fields(reference)
    rows = {}
    for name in TOKENIZER_FIELDS:
        rows[name] = {"new": canonical_sha256(a.get(name)), "reference": canonical_sha256(b.get(name)),
                      "equal": a.get(name) == b.get(name), "present": name in a or name in b}
    template_equal = a.get("tokenizer.chat_template") == b.get("tokenizer.chat_template")
    adds_bos = bool(a.get("tokenizer.ggml.add_bos_token")) or bool(b.get("tokenizer.ggml.add_bos_token"))
    required = [name for name in REQUIRED_TOKENIZER_FIELDS if rows[name]["present"] and not rows[name]["equal"]]
    if adds_bos and not rows["tokenizer.ggml.bos_token_id"]["equal"]:
        required.append("tokenizer.ggml.bos_token_id")
    reported = [name for name in REPORTED_TOKENIZER_FIELDS if rows[name]["present"] and not rows[name]["equal"]]
    return {"schema": TOKENIZER_SCHEMA, "new": str(new), "reference": str(reference),
            "status": "PASS" if not required else "FAIL", "differs": required, "special_ids_differ": reported,
            "fields": rows, "chat_template_equal": template_equal, "adds_bos": adds_bos,
            "note": "trace prompts are token ids; the chat template only matters for re-rendered prompts; "
                    "requests run to their recorded output length, so EOS/pad ids do not change the replay"}


# ---------------------------------------------------------------- trace-variant

def parse_execution(spec: str) -> tuple[str, dict[str, object], Path | None]:
    model_id, sep, rest = spec.partition("=")
    require(bool(model_id) and sep == "=" and bool(rest), "execution spec is MODEL_ID=PATH[=KIND] or MODEL_ID=@SHA:BYTES:FILE[:KIND]")
    if rest.startswith("@"):
        parts = rest[1:].split(":")
        require(len(parts) in (3, 4) and len(parts[0]) == 64 and parts[1].isdigit(), "bad @SHA:BYTES:FILE[:KIND] spec")
        return model_id, {"artifact_bytes": int(parts[1]), "artifact_file": parts[2], "artifact_sha256": parts[0],
                          "kind": parts[3] if len(parts) == 4 else "text_decoder"}, None
    path_text, _, kind = rest.partition("=")
    path = Path(path_text)
    require(path.is_file(), f"execution artifact {path} is not a file")
    return model_id, {"artifact_bytes": path.stat().st_size, "artifact_file": path.name,
                      "artifact_sha256": file_sha256(path), "kind": kind or "text_decoder"}, path


def trace_variant(source: Path, out: Path, specs: list[str]) -> dict[str, object]:
    from research_dev.scheduler.campaigns.burstgpt.trace_inputs import (  # noqa: PLC0415
        TRACE_COLD_MODEL_ID, TRACE_HOT_MODEL_ID, validate_trace,
    )

    manifest_path = source / "TRACE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    inventory = {}
    for spec in specs:
        model_id, entry, _ = parse_execution(spec)
        require(model_id not in (TRACE_HOT_MODEL_ID, TRACE_COLD_MODEL_ID),
                f"{model_id} is a trace role id; use the execution model id")
        require(model_id not in inventory, f"duplicate execution model {model_id}")
        inventory[model_id] = entry
    old = manifest.get("model_inventory", {})
    require(len(inventory) == len(old), f"the variant must name {len(old)} execution models (has {len(inventory)})")
    overlay_ids = {json.loads(line)["execution_model_id"]
                   for line in (source / "REQUESTS_OVERLAY.jsonl").read_text().splitlines() if line.strip()}
    require(overlay_ids <= set(inventory), "the overlay execution model must stay in the inventory")
    require(not out.exists(), f"{out} exists")
    shutil.copytree(source, out)
    manifest["model_inventory"] = dict(sorted(inventory.items()))
    manifest["ws17_variant"] = {"source_manifest_sha256": "sha256:" + file_sha256(manifest_path),
                                "replaced": "model_inventory only"}
    (out / "TRACE_MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    validate_trace(out / "REQUESTS_SEMANTIC_SOURCE.jsonl", out / "REQUESTS_OVERLAY.jsonl", manifest)
    return {"out": str(out), "model_inventory": manifest["model_inventory"]}


# ---------------------------------------------------------------- vram-fit

def _header(path: Path) -> dict[str, object]:
    from research_dev.scheduler._internal.layer_placement_opgroups import read_gguf_header  # noqa: PLC0415

    return read_gguf_header(path, gguf_module())


def _tensor_bytes(row: dict[str, object]) -> int:
    gguf = gguf_module()
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[str(row["type"])]]
    elements = 1
    for dim in row["shape"]:
        elements *= int(dim)
    require(elements % block == 0, f"tensor of {elements} elements does not align to {row['type']} blocks")
    return elements // block * size


def trace_context_need(trace_dir: Path, tokenizer: str) -> int:
    need = 0
    for line in (trace_dir / "REQUESTS_SEMANTIC_SOURCE.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("prompt_tokenizer_model") == tokenizer:
            need = max(need, int(row["input_tokens"]) + int(row["output_tokens"]))
    require(need > 0, f"no trace rows tokenized with {tokenizer}")
    return need


def vram_fit(path: Path, *, context_size: int, parallel: int, ubatch: int, gpu_first_layer: int | None,
             free_vram_bytes: int, reserve_bytes: int, workspace_bytes: int, cuda_context_bytes: int,
             model_id: str = "model", trace_need: int | None = None) -> dict[str, object]:
    from research_dev.scheduler._internal.layer_placement_opgroups import model_geometry  # noqa: PLC0415

    header = _header(path)
    geometry = model_geometry(model_id, header, {"native": header})
    tensors = dict(header["tensors"])
    blocks = len(geometry.layers)
    layer_bytes = [0] * blocks
    other_gpu = 0
    for name, row in tensors.items():
        if name.startswith("blk."):
            layer_bytes[int(name.split(".")[1])] += _tensor_bytes(row)
        elif name in ("output.weight", "output_norm.weight"):
            other_gpu += _tensor_bytes(row)
    if "output.weight" not in tensors:      # tied head: llama.cpp keeps a GPU copy of the embedding for the head
        other_gpu += _tensor_bytes(tensors["token_embd.weight"])

    def kv_cells(layer) -> int:
        if layer.window is None:
            return context_size
        return min(context_size, parallel * layer.window + ubatch)

    def footprint(first: int) -> dict[str, int]:
        weights = sum(layer_bytes[first:]) + other_gpu
        kv = sum(geometry.layers[i].kv_bytes_per_token * kv_cells(geometry.layers[i]) for i in range(first, blocks))
        return {"weights": weights, "kv": kv, "workspace": workspace_bytes, "cuda_context": cuda_context_bytes,
                "reserve": reserve_bytes, "total": weights + kv + workspace_bytes + cuda_context_bytes + reserve_bytes}

    if gpu_first_layer is None:
        first = next((n for n in range(blocks + 1) if footprint(n)["total"] <= free_vram_bytes), None)
    else:
        first = gpu_first_layer
    chosen = footprint(first) if first is not None else None
    per_slot = context_size // parallel
    return {"schema": FIT_SCHEMA, "model": str(path), "model_id": model_id, "architecture": geometry.architecture,
            "block_count": blocks, "context_size": context_size, "parallel": parallel, "per_slot_context": per_slot,
            "ubatch": ubatch, "free_vram_bytes": free_vram_bytes, "gpu_first_layer": first,
            "gpu_layers": None if first is None else blocks - first,
            "fits": chosen is not None and chosen["total"] <= free_vram_bytes, "footprint_bytes": chosen,
            "footprint_gib": None if chosen is None else {k: round(v / GIB, 3) for k, v in chosen.items()},
            "cpu_layer_bytes": None if first is None else sum(layer_bytes[:first]),
            "trace_max_request_context": trace_need,
            "slot_context_ok": None if trace_need is None else per_slot >= trace_need,
            "assumptions": {"kv_type": "f16", "swa_cells": "min(context, parallel*window + ubatch)",
                            "token_embd": "CPU unless the head is tied",
                            "workspace": "compute buffers (stated, not measured here)"}}


# ---------------------------------------------------------------- plans-seed

def _inherited_memory_parameters(base_rows: list[dict], role: str, manifest) -> dict[str, object]:
    """Memory parameters a seeded row inherits from the base row of the same role (2026-10-06, gotcha 4 of the
    first all-GPU Q4 run): the catalog's MEMORY_CAPACITY projection needs the rig's
    ``memory_model_weight_allocation_ppm:*`` / ``memory_workspace_minimum_bytes:*`` / ``context_token_quantum``
    and, for a sliding-window model, ``kv_cache_swa_padding_tokens``. They describe the rig and the architecture,
    not the weight format, and ``desktop_parent_calibration`` keeps whatever the seed row carries."""
    source = next((row for row in base_rows if row.get("desktop_executor_id") == f"physical:{role}:desktop"), None)
    if source is None:
        return {}
    # a sliding-window model: a per-layer pattern, or a window shorter than the training context (Gemma-4 1024 of
    # 262144; Qwen3 reports sliding_window == context_length and has none)
    sliding_window = manifest is not None and (
        any(manifest.sliding_window_pattern) or 0 < manifest.sliding_window < manifest.context_length)
    return {key: value for key, value in source.get("adapter_parameters", {}).items()
            if key.startswith("memory_") or key == "context_token_quantum"
            or (key == "kv_cache_swa_padding_tokens" and sliding_window)}


def plans_seed(base: Path, out: Path, rows: list[str], *, manifests: Mapping[str, Path] | None = None,
               cuda_graph_mode: str = "default", only_new: bool = False,
               cpu_device_id: str = "desktop-cpu", gpu_device_id: str = "desktop-cuda") -> dict[str, object]:
    """Seed desktop baseline plan rows for new artifacts.

    2026-10-06 (first all-GPU Q4 run): the catalog accepts a seeded row only when (1) the plans file names no artifact
    outside the campaign (``only_new`` writes just the seeded rows), (2) ``desktop_placement_sha256`` is the
    operator-placement identity the catalog derives from the checked manifest and ``gpu_first_layer`` (given with
    ``manifests`` = artifact -> checked manifest JSON; without it the row keeps a placeholder the catalog rejects),
    and (3) the row's and the adapter parameters' ``cuda_graph_mode`` agree (``cuda_graph_mode``).
    """
    require(cuda_graph_mode in ("default", "disabled"), "cuda_graph_mode is default or disabled")
    value = json.loads(base.read_text())
    require(value.get("schema") == "s42-measured-desktop-baseline-plans-v1", "not a desktop baseline plans file")
    present = {row["artifact_sha256"] for row in value["plans"]}
    manifest_paths = {("sha256:" + key if not key.startswith("sha256:") else key): Path(path)
                      for key, path in (manifests or {}).items()}
    added = []
    for spec in rows:
        artifact, _, rest = spec.partition("=")
        role, first, seed = rest.split(":", 2)
        artifact = artifact if artifact.startswith("sha256:") else "sha256:" + artifact
        require(len(artifact) == 71 and artifact not in present, f"artifact {artifact} is invalid or already present")
        require(role in ("hot", "cold") and first.isdigit() and Path(seed).is_file(), f"bad seed row {spec}")
        seed_value = json.loads(Path(seed).read_text())
        require(seed_value.get("schema") == FIT_SCHEMA and seed_value.get("fits") is True
                and seed_value.get("gpu_first_layer") == int(first), "seed must be a passing vram-fit at that layer")
        evidence = "sha256:" + file_sha256(Path(seed))
        manifest_path = manifest_paths.get(artifact)
        manifest = None
        if manifest_path is not None:
            from types import SimpleNamespace  # noqa: PLC0415
            from research_dev.scheduler._internal.model_manifest import ModelManifest  # noqa: PLC0415
            from research_dev.scheduler._internal.desktop_parent import _placement_sha256  # noqa: PLC0415
            from research_dev.scheduler.adapters.catalog_materialization import (  # noqa: PLC0415
                desktop_operator_placements,
            )
            manifest = ModelManifest.from_json(json.loads(manifest_path.read_text()))
            require(manifest.artifact_sha256 == artifact, f"manifest {manifest_path} is not the seed row's artifact")
            require(int(first) <= manifest.block_count, "seed layer exceeds the model's block count")
            placements = desktop_operator_placements(
                manifest, int(first), SimpleNamespace(cpu_device_id=cpu_device_id, gpu_device_id=gpu_device_id))
            placement = _placement_sha256(manifest, placements, cuda_graph_mode)
            note = ("vram-fit seed bound to the catalog placement identity; replace with "
                    "desktop_parent_calibration.py before any measured arm")
        else:
            placement = canonical_sha256({"artifact_sha256": artifact, "gpu_first_layer": int(first),
                                          "seed": evidence, "role": role})
            note = ("vram-fit seed with a PLACEHOLDER placement identity (no manifest given): the catalog rejects it; "
                    "re-seed with --manifest or replace with desktop_parent_calibration.py")
        inherited = _inherited_memory_parameters(value["plans"], role, manifest)
        if inherited:
            note += "; inherited " + ",".join(sorted(inherited)) + f" from the {role} base row"
        row = {"adapter_parameters": {**inherited, "context_size": seed_value["context_size"],
                                      "parallel": seed_value["parallel"], "cuda_graph_mode": cuda_graph_mode},
               "artifact_sha256": artifact, "cuda_graph_mode": cuda_graph_mode,
               "desktop_control_profile_id": "desktop-control:" + artifact[7:23],
               "desktop_executor_id": f"physical:{role}:desktop", "desktop_placement_sha256": placement,
               "evidence_ids": [evidence], "gpu_first_layer": int(first), "identity_evidence_id": evidence,
               "maturity": "QUALIFIED", "ws17_seed": note}
        value["plans"].append(row)
        present.add(artifact)
        added.append(row)
    if only_new:
        value["plans"] = list(added)
    require(not out.exists(), f"{out} exists")
    out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return {"out": str(out), "added": added}


# ---------------------------------------------------------------- prior-receipt

PRIOR_RECEIPT_SCHEMA = "ws17-calibration-prior-receipt-v1"


def _bench_times(path: Path) -> dict[str, float]:
    """Per-token prefill and decode times (us) from ``llama-bench -o json`` rows (n_prompt>0 / n_gen>0)."""
    rows = json.loads(path.read_text())
    prefill = [row["avg_ns"] / 1e3 / row["n_prompt"] for row in rows if row.get("n_prompt") and not row.get("n_gen")]
    decode = [row["avg_ns"] / 1e3 / row["n_gen"] for row in rows if row.get("n_gen") and not row.get("n_prompt")]
    require(bool(prefill) and bool(decode), f"{path} needs a prefill-only and a decode-only llama-bench row")
    return {"prefill_us_per_token": sum(prefill) / len(prefill), "decode_us_per_token": sum(decode) / len(decode)}


def prior_receipt(args: argparse.Namespace) -> dict[str, object]:
    rows = [json.loads(line) for line in (args.trace_dir / "REQUESTS_SEMANTIC_SOURCE.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row.get("prompt_tokenizer_model") == args.tokenizer]
    require(bool(rows), f"no trace rows tokenized with {args.tokenizer}")
    prompt = sum(int(row["input_tokens"]) for row in rows) / len(rows)
    output = sum(int(row["output_tokens"]) for row in rows) / len(rows)
    new, old = _bench_times(args.new_bench), _bench_times(args.old_bench)
    cost = lambda t: prompt * t["prefill_us_per_token"] + output * t["decode_us_per_token"]  # noqa: E731
    desktop = cost(new) / cost(old)
    if (args.assisted_new_ms_per_token is None) != (args.assisted_old_ms_per_token is None):
        raise RegimeError("give both assisted per-token times or neither")
    assisted = desktop if args.assisted_new_ms_per_token is None else \
        args.assisted_new_ms_per_token / args.assisted_old_ms_per_token
    require(not args.out.exists(), f"{args.out} exists")
    value = {"schema": PRIOR_RECEIPT_SCHEMA, "tokenizer": args.tokenizer, "trace_rows": len(rows),
             "mean_prompt_tokens": prompt, "mean_output_tokens": output, "new": new, "old": old,
             "new_bench_sha256": "sha256:" + file_sha256(args.new_bench),
             "old_bench_sha256": "sha256:" + file_sha256(args.old_bench),
             "assisted_source": "desktop ratio" if args.assisted_new_ms_per_token is None else "measured E1c",
             "desktop_service_scale_ppm": int(round(desktop * 1_000_000)),
             "assisted_service_scale_ppm": int(round(assisted * 1_000_000))}
    args.out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    value["receipt_sha256"] = "sha256:" + file_sha256(args.out)
    return value


# ---------------------------------------------------------------- template

def _apply(target: dict[str, object], edits: dict[str, object]) -> None:
    for key, value in edits.items():
        if key in ("runtime_parameters", "phone_adapter_parameters") and isinstance(value, dict):
            current = dict(target.get(key, {}))
            for name, item in value.items():
                if item is None:
                    current.pop(name, None)
                else:
                    current[name] = item
            target[key] = current
        elif value is None:
            target.pop(key, None)
        else:
            target[key] = value


def derive_template(base: Path, spec_path: Path, out: Path) -> dict[str, object]:
    from research_dev.scheduler.configuration.models import ModelsManifest  # noqa: PLC0415

    spec = json.loads(spec_path.read_text())
    require(spec.get("schema") == SPEC_SCHEMA, "not a ws17 regime template spec")
    require(not out.exists(), f"{out} exists")
    shutil.copytree(base, out)
    campaign = json.loads((out / "campaign.json").read_text())
    models = json.loads((out / "models.json").read_text())
    campaign["campaign_id"] = campaign["campaign_id"] + "-" + spec["campaign_id_suffix"]
    for name in ("rig", "models", "evidence"):
        campaign[name + "_manifest_path"] = str(out.resolve() / (name + ".json"))
    if spec.get("trace_dir"):
        trace = Path(spec["trace_dir"])
        schedule = campaign["trace"].get("replay_schedule_path")
        if schedule:                    # the variant directory carries a copy of the same replay schedule
            schedule = str(trace / Path(schedule).name)
        campaign["trace"].update({
            "large_requests_path": str(trace / "REQUESTS_SEMANTIC_SOURCE.jsonl"),
            "overlay_requests_path": str(trace / "REQUESTS_OVERLAY.jsonl"),
            "replay_schedule_path": schedule,
            "trace_manifest_path": str(trace / "TRACE_MANIFEST.json"),
        })
    if spec.get("desktop_baseline_plans_path"):
        models["desktop_baseline_plans_path"] = spec["desktop_baseline_plans_path"]
    by_role = {row["model_key"]: row for row in models["models"]}
    for role, edits in spec.get("models", {}).items():
        require(role in by_role, f"template has no {role} model")
        _apply(by_role[role], edits)
    (out / "campaign.json").write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    (out / "models.json").write_text(json.dumps(models, indent=2, sort_keys=True) + "\n")
    loaded = ModelsManifest.from_json(models, out)
    note = out / "CHANGES.txt"
    note.write_text((note.read_text() if note.exists() else "") + "ws17 template from " + str(base) + " spec "
                    + str(spec_path) + " sha " + canonical_sha256(spec) + "\n")
    return {"out": str(out), "campaign_id": campaign["campaign_id"],
            "models": [{"key": row.model_key, "model_id": row.model_id, "artifact": str(row.host_artifact_path),
                        "matrix_precision": None if row.matrix_precision is None else row.matrix_precision.to_json(),
                        "calibration_prior": None if row.calibration_prior is None else row.calibration_prior.to_json()}
                       for row in loaded.models]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tokenizer-check")
    t.add_argument("--gguf", type=Path, required=True)
    t.add_argument("--reference", type=Path, required=True)
    t.add_argument("--out", type=Path)
    v = sub.add_parser("trace-variant")
    v.add_argument("--source", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--execution", action="append", required=True)
    f = sub.add_parser("vram-fit")
    f.add_argument("--gguf", type=Path, required=True)
    f.add_argument("--model-id", default="model")
    f.add_argument("--context-size", type=int, required=True)
    f.add_argument("--parallel", type=int, required=True)
    f.add_argument("--ubatch", type=int, default=512)
    f.add_argument("--gpu-first-layer", default="auto")
    f.add_argument("--free-vram-bytes", type=int, required=True)
    f.add_argument("--reserve-bytes", type=int, default=0)
    f.add_argument("--workspace-bytes", type=int, default=GIB)
    f.add_argument("--cuda-context-bytes", type=int, default=384 * 1024 ** 2)
    f.add_argument("--trace-dir", type=Path)
    f.add_argument("--tokenizer")
    f.add_argument("--out", type=Path)
    s = sub.add_parser("plans-seed")
    s.add_argument("--base", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--row", action="append", required=True)
    s.add_argument("--manifest", action="append", default=[], metavar="ARTIFACT=CHECKED_MANIFEST.json",
                   help="bind the row's placement identity to the catalog's (required for a catalog-valid seed)")
    s.add_argument("--cuda-graph-mode", choices=("default", "disabled"), default="default")
    s.add_argument("--only-new", action="store_true",
                   help="write only the seeded rows (a campaign's plans file may name no other artifact)")
    g = sub.add_parser("manifest")
    g.add_argument("--gguf", type=Path, required=True)
    g.add_argument("--model-id", required=True)
    g.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("prior-receipt")
    r.add_argument("--new-bench", type=Path, required=True)
    r.add_argument("--old-bench", type=Path, required=True)
    r.add_argument("--trace-dir", type=Path, required=True)
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--assisted-new-ms-per-token", type=float)
    r.add_argument("--assisted-old-ms-per-token", type=float)
    r.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("template")
    m.add_argument("--base", type=Path, required=True)
    m.add_argument("--spec", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "tokenizer-check":
        value = tokenizer_check(args.gguf, args.reference)
    elif args.command == "trace-variant":
        value = trace_variant(args.source, args.out, args.execution)
    elif args.command == "vram-fit":
        need = trace_context_need(args.trace_dir, args.tokenizer) if args.trace_dir else None
        value = vram_fit(args.gguf, context_size=args.context_size, parallel=args.parallel, ubatch=args.ubatch,
                         gpu_first_layer=None if args.gpu_first_layer == "auto" else int(args.gpu_first_layer),
                         free_vram_bytes=args.free_vram_bytes, reserve_bytes=args.reserve_bytes,
                         workspace_bytes=args.workspace_bytes, cuda_context_bytes=args.cuda_context_bytes,
                         model_id=args.model_id, trace_need=need)
    elif args.command == "plans-seed":
        manifests = {}
        for spec in args.manifest:
            artifact, sep, path = spec.partition("=")
            require(sep == "=" and bool(artifact) and bool(path), "--manifest takes ARTIFACT=CHECKED_MANIFEST.json")
            manifests[artifact] = Path(path)
        value = plans_seed(args.base, args.out, args.row, manifests=manifests,
                           cuda_graph_mode=args.cuda_graph_mode, only_new=args.only_new)
    elif args.command == "prior-receipt":
        value = prior_receipt(args)
    elif args.command == "manifest":
        from research_dev.scheduler._internal.model_manifest import GGUFModelManifestLoader  # noqa: PLC0415

        require(not args.out.exists(), f"{args.out} exists")
        manifest = GGUFModelManifestLoader.load(args.model_id, args.gguf)
        args.out.write_text(json.dumps(manifest.to_json(), indent=2, sort_keys=True) + "\n")
        value = {"out": str(args.out), "model_id": manifest.model_id, "artifact_sha256": manifest.artifact_sha256,
                 "block_count": manifest.block_count, "feed_forward_length": manifest.feed_forward_length}
    else:
        value = derive_template(args.base, args.spec, args.out)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if getattr(args, "out", None) is not None and args.command in ("tokenizer-check", "vram-fit"):
        require(not args.out.exists(), f"{args.out} exists")
        args.out.write_text(text)
    print(text, end="")
    if args.command == "tokenizer-check":
        return 0 if value["status"] == "PASS" else 2
    if args.command == "vram-fit":
        return 0 if value["fits"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
