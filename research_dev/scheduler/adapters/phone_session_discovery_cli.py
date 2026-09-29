#!/usr/bin/env python3
"""Discover phone residency sessions and write a portable capability file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from .phone_session_discovery import (
    PhoneSessionDiscoveryConfiguration,
    parse_phone_session_probe,
    phone_session_discovery_json,
)


def _remote(adb: Path, port: int, serial: str, command: str) -> str:
    result = subprocess.run(
        [str(adb), "-P", str(port), "-s", serial, "shell", command],
        check=False,
        capture_output=True,
        text=True,
        encoding="ascii",
        errors="backslashreplace",
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "phone session discovery command failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    return result.stdout


def _remote_sha256(
    adb: Path, port: int, serial: str, path: str
) -> str:
    output = _remote(
        adb,
        port,
        serial,
        "sha256sum " + shlex.quote(path),
    )
    fields = output.split()
    if len(fields) < 2 or len(fields[0]) != 64:
        raise RuntimeError("phone worker SHA-256 is unavailable")
    return "sha256:" + fields[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", type=Path, required=True)
    parser.add_argument("--adb-port", type=int, default=5037)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--session-mib", type=int, required=True)
    parser.add_argument("--minimum-available-mib", type=int, default=1536)
    parser.add_argument("--maximum-sessions", type=int, default=16)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--endpoint-prefix", required=True)
    parser.add_argument("--shared-compute-resource", required=True)
    parser.add_argument(
        "--shared-transport-resource", action="append", required=True
    )
    parser.add_argument("--phone-memory-resource", required=True)
    parser.add_argument("--phone-wide-limit-bytes", type=int, required=True)
    parser.add_argument("--supported-layer-mask", type=int, required=True)
    parser.add_argument("--maximum-columns", type=int, required=True)
    parser.add_argument("--column-quantum", type=int, required=True)
    parser.add_argument("--data-type", action="append", required=True)
    parser.add_argument("--batch-plan", action="append", required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output == args.raw_output
        or args.output.exists()
        or args.raw_output.exists()
    ):
        raise RuntimeError("phone session discovery output already exists")
    root = str(Path(args.probe).parent)
    probe_command = " ".join((
        "LD_LIBRARY_PATH=" + shlex.quote(root),
        "ADSP_LIBRARY_PATH=" + shlex.quote(root),
        "GGML_HEXAGON_NDEV=" + str(args.maximum_sessions),
        "GGML_HEXAGON_VMEM=" + str(args.session_mib),
        shlex.quote(args.probe),
        "--session-mib", str(args.session_mib),
        "--minimum-available-mib", str(args.minimum_available_mib),
        "--maximum-sessions", str(args.maximum_sessions),
    ))
    output = _remote(
        args.adb,
        args.adb_port,
        args.serial,
        "su -c " + shlex.quote(probe_command),
    )
    raw = output.encode("ascii")
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw_temporary = args.raw_output.with_name(
        args.raw_output.name + "." + hashlib.sha256(raw).hexdigest()[:12]
    )
    raw_temporary.write_bytes(raw)
    raw_temporary.replace(args.raw_output)
    configuration = PhoneSessionDiscoveryConfiguration(
        device_id=args.device_id,
        endpoint_prefix=args.endpoint_prefix,
        worker_identity_sha256=_remote_sha256(
            args.adb, args.adb_port, args.serial, args.worker
        ),
        shared_compute_resource_id=args.shared_compute_resource,
        shared_transport_resource_ids=tuple(
            args.shared_transport_resource
        ),
        phone_wide_memory_resource_id=args.phone_memory_resource,
        phone_wide_limit_bytes=args.phone_wide_limit_bytes,
        supported_layer_mask=args.supported_layer_mask,
        maximum_columns=args.maximum_columns,
        column_quantum=args.column_quantum,
        supported_data_types=tuple(args.data_type),
        batch_plans=tuple(args.batch_plan),
    )
    sessions = parse_phone_session_probe(output, configuration)
    encoded = json.dumps(
        phone_session_discovery_json(sessions),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("ascii") + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(
        args.output.name + "." + hashlib.sha256(encoded).hexdigest()[:12]
    )
    temporary.write_bytes(encoded)
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "raw_output": str(args.raw_output),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "sessions": len(sessions),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
