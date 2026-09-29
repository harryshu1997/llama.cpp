#!/usr/bin/env python3
"""Bind a GPU-stage profile to live worker, GGUF, and VRAM receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
from typing import Any

from gpu_stage_wavefront_gate import (
    I32,
    IDENTITY_STRUCT,
    PROFILE_SCHEMA,
    STAGE_DETACH,
    STAGE_IDENTITY_MAGIC,
    STAGE_IDENTITY_VERSION,
    STAGE_V3_CAP_IDENTITY,
    STAGE_V3_CAP_TERMINAL,
    STAGE_V3_HELLO,
    STAGE_V3_IDENTITY,
    STAGE_V3_MAGIC,
    STAGE_V3_VERSION,
    V3_HELLO_STRUCT,
    recv_exact,
)


MANIFEST_SCHEMA = "s42-llama-gpu-stage-tensor-manifest-v1"
MEMORY_SCHEMA = "layersplit-memory-breakdown-v1"
MEMORY_PATTERN = re.compile(r"^MEMORYCERT (?P<value>\{.*\})$", re.MULTILINE)


class MaterializeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializeError(message)


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_record(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializeError(f"cannot read {name}") from exc
    require(type(value) is dict, f"invalid {name}")
    return value


def parse_memory_certificate(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise MaterializeError("cannot read stage worker log") from exc
    rows = MEMORY_PATTERN.findall(text)
    require(len(rows) == 1, "stage memory certificate is not unique")
    try:
        value = json.loads(rows[0])
    except json.JSONDecodeError as exc:
        raise MaterializeError("stage memory certificate is invalid") from exc
    require(
        type(value) is dict
        and value.get("schema") == MEMORY_SCHEMA
        and value.get("role") == "stagenet",
        "stage memory certificate identity",
    )
    fields = (
        "model_buffer_bytes",
        "kv_buffer_bytes",
        "compute_buffer_bytes",
        "host_model_buffer_bytes",
        "host_context_buffer_bytes",
        "host_compute_buffer_bytes",
    )
    require(
        all(type(value.get(field)) is int and value[field] >= 0 for field in fields)
        and value["model_buffer_bytes"] > 0
        and value["host_model_buffer_bytes"] > 0,
        "stage memory certificate values",
    )
    return value


def parse_gpu_snapshot(path: Path) -> tuple[str, int, int]:
    try:
        fields = [
            value.strip()
            for value in path.read_text(encoding="ascii").strip().split(",")
        ]
        require(len(fields) == 3, "GPU snapshot field count")
        gpu_uuid = fields[0]
        total_bytes = int(fields[1]) * 1024 * 1024
        free_bytes = int(fields[2]) * 1024 * 1024
    except (OSError, UnicodeError, ValueError) as exc:
        raise MaterializeError("GPU snapshot is invalid") from exc
    require(
        gpu_uuid.startswith("GPU-") and 0 < free_bytes <= total_bytes,
        "GPU snapshot identity",
    )
    return gpu_uuid, total_bytes, free_bytes


def validate_manifest(
    manifest: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    expected_record = manifest.get("record_sha256")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("record_sha256", None)
    require(
        manifest.get("schema") == MANIFEST_SCHEMA
        and manifest.get("status") == "EXACT_GGUF_TENSOR_RANGES"
        and expected_record == sha256_record(unsigned_manifest),
        "GPU-stage manifest identity",
    )
    model = profile.get("model")
    manifest_model = manifest.get("model")
    placement = manifest.get("stage_placement")
    require(
        type(model) is dict
        and type(manifest_model) is dict
        and type(placement) is dict,
        "GPU-stage manifest sections",
    )
    expected_weight = placement.get("stage_weight_sha256")
    unsigned_placement = dict(placement)
    unsigned_placement.pop("stage_weight_sha256", None)
    require(
        expected_weight == sha256_record(unsigned_placement)
        and manifest_model.get("sha256") == model.get("sha256")
        and manifest.get("block_count") == model.get("n_layer")
        and placement.get("model_sha256") == model.get("sha256")
        and placement.get("layer_start") == model.get("layer_start")
        and placement.get("layer_end") == model.get("layer_end")
        and type(placement.get("selected_tensor_count")) is int
        and placement["selected_tensor_count"] > 0
        and type(placement.get("selected_materialized_raw_bytes")) is int
        and placement["selected_materialized_raw_bytes"] > 0,
        "GPU-stage manifest geometry",
    )
    return placement


def probe_worker(host: str, port: int, timeout_s: float) -> dict[str, Any]:
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as worker:
            worker.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            worker.settimeout(timeout_s)
            worker.sendall(I32.pack(STAGE_V3_HELLO))
            hello_packet = recv_exact(worker, V3_HELLO_STRUCT.size)
            require(hello_packet is not None, "stage hello is absent")
            hello = V3_HELLO_STRUCT.unpack(hello_packet)
            require(
                hello[0] == STAGE_V3_MAGIC
                and hello[1] == STAGE_V3_VERSION
                and not (hello[10] & STAGE_V3_CAP_TERMINAL)
                and (hello[10] & STAGE_V3_CAP_IDENTITY),
                "stage hello identity",
            )
            worker.sendall(I32.pack(STAGE_V3_IDENTITY))
            identity_packet = recv_exact(worker, IDENTITY_STRUCT.size)
            digest = recv_exact(worker, 32)
            require(
                identity_packet is not None and digest is not None,
                "stage model identity is absent",
            )
            identity = IDENTITY_STRUCT.unpack(identity_packet)
            require(
                identity[:2]
                == (STAGE_IDENTITY_MAGIC, STAGE_IDENTITY_VERSION),
                "stage model identity header",
            )
            worker.sendall(I32.pack(STAGE_DETACH))
            ack = recv_exact(worker, I32.size)
            require(
                ack is not None and I32.unpack(ack)[0] == 0,
                "stage probe detach",
            )
    except (OSError, TimeoutError) as exc:
        raise MaterializeError("cannot probe resident GPU stage") from exc
    return {
        "capabilities": hello[10],
        "file_type": identity[2],
        "layer_end": hello[3],
        "layer_start": hello[2],
        "max_streams": hello[6],
        "model_sha256": digest.hex(),
        "n_batch": hello[8],
        "n_ctx_seq": hello[7],
        "n_embd": hello[5],
        "n_layer": hello[4],
        "n_ubatch": hello[9],
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_object(args.template, "profile template")
    require(profile.get("schema") == PROFILE_SCHEMA, "profile schema mismatch")
    manifest = load_object(args.stage_manifest, "GPU-stage manifest")
    placement = validate_manifest(manifest, profile)
    memory_cert = parse_memory_certificate(args.worker_log)
    gpu_uuid, total_bytes, free_bytes = parse_gpu_snapshot(args.gpu_snapshot)
    identity = probe_worker(args.worker_host, args.worker_port, args.timeout_s)

    model = profile.get("model")
    memory = profile.get("memory")
    candidate = profile.get("candidate")
    require(
        type(model) is dict
        and type(memory) is dict
        and type(candidate) is dict,
        "profile runtime sections",
    )
    require(
        identity["model_sha256"] == model.get("sha256")
        and identity["n_layer"] == model.get("n_layer")
        and identity["n_embd"] == model.get("n_embd")
        and identity["file_type"] == model.get("file_type")
        and identity["layer_start"] == model.get("layer_start")
        and identity["layer_end"] == model.get("layer_end")
        and identity["max_streams"] == 1
        and identity["n_batch"] == 512
        and identity["n_ubatch"] == 512,
        "live GPU stage differs from the template",
    )
    raw_bytes = placement["selected_materialized_raw_bytes"]
    resident_bytes = memory_cert["model_buffer_bytes"]
    require(
        raw_bytes <= resident_bytes
        and resident_bytes - raw_bytes < 64 * 1024 * 1024,
        "GPU-stage model allocation differs from selected raw tensors",
    )
    workspace_bytes = candidate.get("workspace_bytes")
    reserve_bytes = memory.get("reserve_bytes")
    require(
        type(workspace_bytes) is int
        and workspace_bytes > 0
        and type(reserve_bytes) is int
        and reserve_bytes > 0
        and free_bytes - reserve_bytes >= workspace_bytes,
        "GPU reserve is insufficient after stage residency",
    )

    model["resident_bytes"] = resident_bytes
    model["stage_weight_sha256"] = placement["stage_weight_sha256"]
    memory.update({
        "free_bytes": free_bytes,
        "gpu_uuid": gpu_uuid,
        "snapshot_source": args.gpu_snapshot.name,
        "stage_memory_certificate": memory_cert,
        "total_bytes": total_bytes,
    })
    profile["runtime_materialization"] = {
        "gpu_snapshot": args.gpu_snapshot.name,
        "stage_manifest": args.stage_manifest.name,
        "stage_manifest_record_sha256": manifest["record_sha256"],
        "worker_identity": identity,
        "worker_log": args.worker_log.name,
    }
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--gpu-snapshot", type=Path, required=True)
    parser.add_argument("--worker-log", type=Path, required=True)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--worker-host", default="127.0.0.1")
    parser.add_argument("--worker-port", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.worker_port <= 65535 or args.timeout_s <= 0:
        parser.error("invalid stage worker endpoint")
    if any(
        not path.is_file()
        for path in (
            args.template,
            args.gpu_snapshot,
            args.worker_log,
            args.stage_manifest,
        )
    ):
        parser.error("materialization inputs are missing")
    if (
        not args.output.is_absolute()
        or args.output.exists()
        or not args.output.parent.is_dir()
    ):
        parser.error("output must be an unused absolute path")
    try:
        profile = materialize(args)
        args.output.write_text(
            json.dumps(profile, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="ascii",
        )
    except (MaterializeError, OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"GPU-stage profile materialization failed: {exc}\n")
    print(json.dumps({
        "free_bytes": profile["memory"]["free_bytes"],
        "gpu_uuid": profile["memory"]["gpu_uuid"],
        "resident_bytes": profile["model"]["resident_bytes"],
        "stage_weight_sha256": profile["model"]["stage_weight_sha256"],
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
