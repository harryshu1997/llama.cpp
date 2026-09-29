#!/usr/bin/env python3
"""Bind a wavefront profile template to one live GPU and LM-head worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


PROFILE_SCHEMA = "s42-fp16-burstgpt-gpu-wavefront-profile-v1"
WORKER_PATTERN = re.compile(
    r"\[lm-head-worker\] ready .*? vocab=(?P<vocab>[0-9]+) "
    r"rows=\[(?P<offset>[0-9]+),(?P<end>[0-9]+)\) "
    r"top_k=(?P<top_k>[0-9]+) .*? io=(?P<io>f16|f32) .*? "
    r"hash=(?P<hash>[0-9a-f]{16})"
)


class MaterializeError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--gpu-snapshot", type=Path, required=True)
    parser.add_argument("--worker-log", type=Path, required=True)
    parser.add_argument("--lm-head-weight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.template.is_file() or not args.gpu_snapshot.is_file() or not args.worker_log.is_file():
        parser.error("materialization inputs are missing")
    if not args.output.is_absolute() or args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be an unused absolute path")

    try:
        profile = json.loads(args.template.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializeError("cannot read profile template") from exc
    if type(profile) is not dict or profile.get("schema") != PROFILE_SCHEMA:
        raise MaterializeError("profile template schema mismatch")
    weight_sha256 = args.lm_head_weight_sha256.removeprefix("sha256:")
    if len(weight_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in weight_sha256
    ):
        raise MaterializeError("LM-head weight SHA-256 is invalid")

    fields = [
        value.strip()
        for value in args.gpu_snapshot.read_text(encoding="ascii").strip().split(",")
    ]
    if len(fields) != 3:
        raise MaterializeError("GPU snapshot must contain UUID,total MiB,free MiB")
    gpu_uuid = fields[0]
    try:
        total_bytes = int(fields[1]) * 1024 * 1024
        free_bytes = int(fields[2]) * 1024 * 1024
    except ValueError as exc:
        raise MaterializeError("GPU memory snapshot is invalid") from exc
    if not gpu_uuid.startswith("GPU-") or not 0 < free_bytes <= total_bytes:
        raise MaterializeError("GPU snapshot identity is invalid")

    matches = WORKER_PATTERN.findall(
        args.worker_log.read_text(encoding="utf-8", errors="replace")
    )
    if len(matches) != 1:
        raise MaterializeError("LM-head worker READY identity is not unique")
    vocab_text, offset_text, end_text, top_k_text, io, weight_hash = matches[0]
    vocab = int(vocab_text)
    offset = int(offset_text)
    end = int(end_text)
    top_k = int(top_k_text)
    worker = profile.get("worker")
    memory = profile.get("memory")
    if type(worker) is not dict or type(memory) is not dict:
        raise MaterializeError("profile runtime sections are invalid")
    if (
        worker.get("n_vocab") != vocab
        or worker.get("offset") != offset
        or worker.get("rows") != end - offset
        or worker.get("top_k") != top_k
        or worker.get("flags") != (1 if io == "f16" else 0)
    ):
        raise MaterializeError("worker READY geometry differs from the template")
    worker["weight_hash64"] = weight_hash
    model = profile.get("model")
    if type(model) is not dict:
        raise MaterializeError("profile model section is invalid")
    model["lm_head_weight_sha256"] = weight_sha256
    memory["gpu_uuid"] = gpu_uuid
    memory["total_bytes"] = total_bytes
    memory["free_bytes"] = free_bytes
    memory["snapshot_source"] = args.gpu_snapshot.name
    profile["runtime_materialization"] = {
        "gpu_snapshot": args.gpu_snapshot.name,
        "worker_log": args.worker_log.name,
    }
    args.output.write_text(
        json.dumps(profile, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
