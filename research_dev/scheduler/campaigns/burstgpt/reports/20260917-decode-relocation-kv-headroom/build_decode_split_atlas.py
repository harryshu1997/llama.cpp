"""Build the measured decode-split atlas (schema v2) from this stage's run records.

Each row is one measured (fraction, request shape) point: decode ms/token, decode-phase host power,
released bytes (dormant proof), prefill seconds, the exact execution environment it was measured in
(artifact, desktop placement, KV plan, context cells, batch shape, column quantum, phone sessions,
runtime bundle digest) and the validated request-shape range: the measured prompt length within the
interpolation tolerance (default 25 %) and a total context no larger than what the measurement
covered (prompt + generated tokens). The selector never uses a row outside that range or environment.

usage: build_decode_split_atlas.py PHYSICAL_DIR OUT_JSON [--prompt-tolerance 0.25]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RUNS = {
    # directory (relative to PHYSICAL_DIR): memory budget note (None = uncapped profile)
    "sweep-v2": None, "pair-v1/control": None, "pair-v1/combined": None, "combined-h0-v1": None,
    "select-smoke-v1": None,
    "capped-v1/control": "capped-18.06GiB", "capped-v1/combined": "capped-18.06GiB",
    "capped-v2/combined": "capped-17.06GiB",
}
N_FF = 17408


def digest(path: Path) -> str:
    return "sha256:" + hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def canonical_sha256(value) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("physical", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--prompt-tolerance", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.prompt_tolerance < 1:
        parser.error("prompt tolerance must be in [0, 1)")
    rows = []
    for rel, budget in RUNS.items():
        base = args.physical / rel
        result_path = base / "RESULT.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        plan = json.loads((base / "KV_PLAN.json").read_text())
        config = json.loads((base / "CONFIG.json").read_text())
        runtime = json.loads((base / "RUNTIME.json").read_text())
        prompt_tokens = len(json.loads((base / "REQUEST.json").read_text())["prompt_tokens"])
        plan_sha = canonical_sha256({k: v for k, v in plan.items() if k not in ("cpu_kv_bytes_per_token", "gpu_kv_bytes_per_token")})
        environment = {
            "artifact_sha256": plan["artifact_sha256"], "gpu_layers": config["gpu_layers"], "context_cells": plan["context_size"],
            "parallel": int(config.get("parallel", 1)), "batch": config["batch"], "ubatch": config["ubatch"], "kv_plan_sha256": plan_sha,
            "column_quantum": config["column_quantum"],
            "session_masks": sorted((name, int(mask)) for name, mask in config["phone"]["session_masks"].items()),
            "runtime_sha256": canonical_sha256(runtime),
        }
        proofs = {p["host_columns"]: p for p in result.get("dormant_proofs", []) if p.get("phase") == "decode"}
        for req in result.get("requests", []):
            if req.get("error") is not None or not req.get("decode_ms_per_token"):
                continue
            fraction_ppm = (N_FF - req["host_columns"]) * 1_000_000 // N_FF if req["split"] else 0
            decode = req.get("decode_host_energy") or {}
            host_w = None
            if decode.get("cpu_package_energy_j") is not None and req.get("decode_s"):
                host_w = (decode["cpu_package_energy_j"] + decode["gpu_board_energy_j"]) / req["decode_s"]
            rows.append({
                "run": rel, "request_id": req["request_id"], "result_sha256": digest(result_path), "environment": environment,
                "prompt_tokens": prompt_tokens, "output_tokens": req["output_tokens"],
                "validated_prompt_tokens_min": max(1, int(prompt_tokens * (1 - args.prompt_tolerance))),
                "validated_prompt_tokens_max": int(prompt_tokens * (1 + args.prompt_tolerance)),
                "validated_total_tokens_max": prompt_tokens + req["output_tokens"],
                "memory_budget": budget, "phone_attached": bool(result.get("phone_preload_s")),
                "split_fraction_ppm": fraction_ppm, "host_columns": req["host_columns"] if req["split"] else N_FF,
                "dormant_release": bool(result.get("dormant_host_share")) and req["split"],
                "decode_ms_per_token": req["decode_ms_per_token"], "prefill_s": req.get("prefill_s"),
                "decode_host_power_w": host_w,
                "released_bytes": proofs.get(req["host_columns"], {}).get("released_bytes", 0) if req["split"] else 0,
                "release_elapsed_us": proofs.get(req["host_columns"], {}).get("elapsed_us") if req["split"] else None,
                "kv_cpu_bytes_per_token": plan["cpu_kv_bytes_per_token"], "kv_gpu_bytes_per_token": plan["gpu_kv_bytes_per_token"],
            })
    atlas = {"schema": "scheduler-decode-split-atlas-v2", "model_id": "qwen3-14b-q4km-dequant-f16",
             "rig": "4060ti-i9-12900k+op15-3sessions-layers0-17", "prompt_tolerance": args.prompt_tolerance, "rows": rows}
    args.out.write_text(json.dumps(atlas, indent=1, sort_keys=True) + "\n")
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
