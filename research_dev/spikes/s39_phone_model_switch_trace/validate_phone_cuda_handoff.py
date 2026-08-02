#!/usr/bin/env python3
"""Validate a frozen S39 phone-to-CUDA handoff report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import phone_cuda_handoff_probe as probe


CERT_SCHEMA = "s39-phone-cuda-handoff-certificate-v1"
HERE = Path(__file__).resolve().parent


def read_canonical(path: Path, field: str) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=probe.strict_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise probe.HandoffError(f"{field}: invalid JSON") from exc
    probe.require(
        type(value) is dict and probe.canonical(value) == raw,
        f"{field}: not canonical",
    )
    return value, raw


def write_atomic(path: Path, value: dict[str, object]) -> None:
    raw = probe.canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def validate_run_context(
    value: object,
    contract: probe.Contract,
) -> None:
    root = probe.exact_keys(
        value,
        {
            "acquisition_unix_s",
            "base_git_commit",
            "contract_sha256",
            "cuda",
            "model_sha256",
            "op12",
            "op15",
            "schema",
            "sources",
        },
        "run_context",
    )
    probe.require(
        root["schema"] == "s39-phone-cuda-handoff-context-v1",
        "run_context: schema mismatch",
    )
    probe.require(
        probe.is_int(root["acquisition_unix_s"])
        and root["acquisition_unix_s"] > 0,
        "run_context: invalid acquisition time",
    )
    probe.require(
        type(root["base_git_commit"]) is str
        and len(root["base_git_commit"]) == 40
        and all(char in "0123456789abcdef" for char in root["base_git_commit"]),
        "run_context: invalid base commit",
    )
    probe.require(
        root["contract_sha256"] == contract.raw_sha256,
        "run_context: contract mismatch",
    )
    probe.require(
        root["model_sha256"] == contract.model_sha256,
        "run_context: model mismatch",
    )

    cuda = probe.exact_keys(
        root["cuda"],
        {
            "device",
            "head_port",
            "relay_port",
            "relay_sha256",
            "tail_port",
            "worker_sha256",
        },
        "run_context.cuda",
    )
    probe.require(cuda["device"] == "CUDA0", "run_context: CUDA device mismatch")
    for field in ("head_port", "relay_port", "tail_port"):
        probe.require(
            probe.is_int(cuda[field]) and 0 < cuda[field] <= 65535,
            f"run_context.cuda.{field}: invalid port",
        )
    for field in ("relay_sha256", "worker_sha256"):
        probe.checked_digest(cuda[field], f"run_context.cuda.{field}")

    expected_devices = {
        "op15": ([0, contract.n_layer - 18], True),
        "op12": ([contract.n_layer - 18, contract.n_layer], False),
    }
    for name, (layers, has_relay) in expected_devices.items():
        keys = {
            "adb_target",
            "boot_id",
            "layers",
            "shard_sha256",
            "wifi",
            "worker_sha256",
        }
        if has_relay:
            keys.add("relay_sha256")
        device = probe.exact_keys(
            root[name],
            keys,
            f"run_context.{name}",
        )
        probe.require(
            type(device["adb_target"]) is str and device["adb_target"] != "",
            f"run_context.{name}: invalid ADB target",
        )
        probe.require(
            type(device["boot_id"]) is str and device["boot_id"] != "",
            f"run_context.{name}: invalid boot ID",
        )
        probe.require(
            device["layers"] == layers,
            f"run_context.{name}: layer range mismatch",
        )
        probe.require(
            type(device["wifi"]) is str and device["wifi"] != "",
            f"run_context.{name}: invalid WiFi endpoint",
        )
        for field in ("shard_sha256", "worker_sha256"):
            probe.checked_digest(
                device[field],
                f"run_context.{name}.{field}",
            )
        if has_relay:
            probe.checked_digest(
                device["relay_sha256"],
                f"run_context.{name}.relay_sha256",
            )

    expected_sources = {
        "async_pipeline.py": (
            HERE.parent / "s22_slo_overlap_pipeline" / "async_pipeline.py"
        ),
        "phone_cuda_handoff_probe.py": HERE / "phone_cuda_handoff_probe.py",
        "qwen25_quality_probe.py": HERE / "qwen25_quality_probe.py",
        "run_w5_handoff_gate.sh": HERE / "run_w5_handoff_gate.sh",
        "stage_v3_client.py": (
            HERE.parent / "s22_slo_overlap_pipeline" / "stage_v3_client.py"
        ),
        "validate_phone_cuda_handoff.py": (
            HERE / "validate_phone_cuda_handoff.py"
        ),
    }
    sources = probe.exact_keys(
        root["sources"],
        set(expected_sources),
        "run_context.sources",
    )
    for name, path in expected_sources.items():
        digest = probe.checked_digest(
            sources[name],
            f"run_context.sources.{name}",
        )
        probe.require(path.is_file(), f"run_context: missing source {name}")
        probe.require(
            probe.sha256(path.read_bytes()) == digest,
            f"run_context: source digest mismatch for {name}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=probe.DEFAULT_CONTRACT,
    )
    parser.add_argument("--run-context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    contract = probe.load_contract(args.contract)
    report, report_raw = read_canonical(args.report, "report")
    run_context, run_context_raw = read_canonical(
        args.run_context,
        "run_context",
    )
    validate_run_context(run_context, contract)
    probe.validate_report(report, contract)
    certificate = {
        "contract_sha256": contract.raw_sha256,
        "report_sha256": probe.sha256(report_raw),
        "run_context_sha256": probe.sha256(run_context_raw),
        "scheduler_eligible": False,
        "schema": CERT_SCHEMA,
        "scope": "MECHANICS_ONLY",
        "status": report["status"],
    }
    write_atomic(args.output, certificate)
    print(probe.canonical(certificate).decode("ascii"), end="")
    return 0 if report["status"] == "HANDOFF_MECHANICS_PASS" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, probe.HandoffError, ValueError) as exc:
        print(probe.canonical({
            "error": str(exc),
            "status": "HANDOFF_VALIDATION_ERROR",
        }).decode("ascii"), end="")
        raise SystemExit(2)
