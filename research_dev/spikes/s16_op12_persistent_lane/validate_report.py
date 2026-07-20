#!/usr/bin/env python3
"""Independently validate the OP12 persistent lane report."""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S16 = HERE.parent / "s16_mixed_persistent_energy"
if str(S16) not in sys.path:
    sys.path.insert(0, str(S16))

from experiment_contract import (  # noqa: E402
    N_GEN, digest_file, strict_object, validate_result_record,
)


class ValidationError(RuntimeError):
    pass


def prefixed(line: bytes, prefix: bytes, label: str) -> dict:
    if not line.startswith(prefix):
        raise ValidationError(f"missing {label} prefix")
    return strict_object(line[len(prefix):], label, False)


def placement(value: dict, pid: int) -> None:
    if value.get("schema") != "layersplit-scheduled-placement-v2" \
            or value.get("role") != "host_tail" or value.get("mode") != "pipedriver" \
            or value.get("layer_start") != 6 or value.get("layer_end") != 48 \
            or value.get("n_layer") != 48 or value.get("pid") != pid \
            or value.get("status") != "SCHEDULED_PLACEMENT_OK" \
            or value.get("run_rc") != 0 or value.get("missing_buffer_compute_nodes") != 0:
        raise ValidationError("host placement identity failed")
    by_op = value.get("compute_by_op_and_buffer")
    if type(by_op) is not dict or not by_op:
        raise ValidationError("host placement has no operation tally")
    if any(backend != "CUDA0" or type(count) is not int or count <= 0
           for buffers in by_op.values() for backend, count in buffers.items()):
        raise ValidationError("host tail compute escaped CUDA0")


def session(value: dict, index: int, end: str) -> None:
    if value.get("schema") != "ls-stagenet-session-v2" \
            or value.get("session_id") != index or value.get("session_end") != end \
            or value.get("expected_backend") != "HTP0" \
            or value.get("layer_start") != 0 or value.get("layer_end") != 6 \
            or value.get("n_layer") != 48 \
            or value.get("placement_status") != "SCHEDULED_PLACEMENT_OK" \
            or value.get("missing_buffer_compute_nodes") != 0 \
            or value.get("reset_applied") is not (end == "DETACH"):
        raise ValidationError("phone session identity failed")
    by_op = value.get("compute_by_op_and_buffer")
    if type(by_op) is not dict or not by_op:
        raise ValidationError("phone session has no operation tally")
    for op_name, buffers in by_op.items():
        for backend, count in buffers.items():
            if type(count) is not int or count <= 0 \
                    or (backend != "HTP0" and not (op_name == "GET_ROWS" and backend == "CPU")):
                raise ValidationError(f"phone compute escaped declared route: {op_name}@{backend}")


def validate() -> dict:
    report_path = HERE / "results/report.json"
    report = strict_object(report_path.read_bytes(), "report")
    if report.get("schema") != "s16-op12-persistent-lane-v1" \
            or report.get("status") != "OP12_PERSISTENT_B32_LANE_PASS" \
            or report.get("synthetic_slo_us") != 12_000_000 \
            or report.get("energy") != "UNKNOWN":
        raise ValidationError("report identity or scope failed")
    reference = report.get("reference_tokens")
    if type(reference) is not list or len(reference) != N_GEN:
        raise ValidationError("reference tokens are invalid")
    host_stdout = (HERE / "results/host.stdout.bin").read_bytes().splitlines(keepends=True)
    host_stderr = (HERE / "results/host.stderr.bin").read_bytes().splitlines(keepends=True)
    phone_stderr = (HERE / "results/phone.stderr.bin").read_bytes().splitlines(keepends=True)
    placements = [line for line in host_stderr if line.startswith(b"PLACEMENTCERT ")]
    markers = [line for line in host_stderr if line.startswith(b"PERSISTENT_DRIVER_EXCHANGE_END ")]
    sessions = [line for line in phone_stderr if line.startswith(b"SESSIONCERT ")]
    if not (len(host_stdout) == len(placements) == len(markers) == len(sessions) == 2):
        raise ValidationError("raw evidence count failed")
    results = []
    certs = []
    for index in (1, 2):
        end = "DETACH" if index == 1 else "STOP"
        result = strict_object(host_stdout[index - 1], "host result")
        validate_result_record(result, reference, "OP12", index, end)
        if result["route_wall_us"] > report["synthetic_slo_us"]:
            raise ValidationError("OP12 SLO failed")
        place = prefixed(placements[index - 1], b"PLACEMENTCERT ", "host placement")
        placement(place, result["host_pid"])
        marker = prefixed(markers[index - 1], b"PERSISTENT_DRIVER_EXCHANGE_END ", "marker")
        if marker != {"launch_id": index}:
            raise ValidationError("exchange marker failed")
        cert = prefixed(sessions[index - 1], b"SESSIONCERT ", "phone session")
        session(cert, index, end)
        results.append(result)
        certs.append(cert)
    if results != report.get("results") or certs != report.get("sessions") \
            or len({item["host_pid"] for item in results}) != 1 \
            or len({item["worker_pid"] for item in certs}) != 1 \
            or len({item["worker_boot_nonce"] for item in certs}) != 1:
        raise ValidationError("report binding or persistence failed")
    artifacts = report.get("artifacts")
    expected = {
        "android_binary": ROOT / "npu-harness/build/llamacpp/android-arm64-hexagon-release-eafdc75e/bin/llama-layersplit",
        "host_binary": ROOT / "build-cuda/bin/llama-layersplit",
        "full_model": Path("/home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf"),
    }
    for name, path in expected.items():
        if artifacts.get(name) != digest_file(path):
            raise ValidationError(f"artifact digest failed: {name}")
    if artifacts.get("shard") != \
            "sha256:d507b7bb453242dff12ba1ce0add53189755b8a9a2b960743ead1578d8f1a6b5" \
            or any(item.get("device_boot_id") != artifacts.get("device_boot_id")
                   for item in certs):
        raise ValidationError("shard or device boot binding failed")
    return {
        "status": report["status"],
        "route_wall_us": [item["route_wall_us"] for item in results],
        "report_sha256": digest_file(report_path),
    }


if __name__ == "__main__":
    try:
        print(json.dumps(validate(), sort_keys=True))
    except (ValidationError, OSError, ValueError) as exc:
        print(f"OP12_VALIDATE_ERROR {exc}")
        raise SystemExit(2)
