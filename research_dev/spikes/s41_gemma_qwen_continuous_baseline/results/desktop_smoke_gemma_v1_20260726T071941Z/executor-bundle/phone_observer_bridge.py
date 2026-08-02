#!/usr/bin/env python3
"""Desktop-side SSH bridge for S40 phone identity and telemetry."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

from executor_bundle import BundleError, validate_runtime_environment


try:
    validate_runtime_environment(Path(__file__).resolve())
except (BundleError, OSError) as error:
    print(f"executor bundle failed: {error}", file=sys.stderr)
    raise SystemExit(2)

from a6000_ssh_control import load_config, validate_local_dependencies
from phone_gateway import (
    GatewayError,
    MAX_COMMAND_BYTES,
    canonical_bytes,
    require,
    strict_json_loads,
)


RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def remote_argv(
    config: dict[str, Any],
    action: str,
    *,
    model_id: str | None = None,
    run_id: str | None = None,
    interval_ms: int | None = None,
) -> list[str]:
    observer = str(
        Path(config["remote_script_path"]).with_name(
            "a6000_phone_observer.py"
        )
    )
    argv = [
        *config["ssh_argv"],
        config["remote_python_path"],
        "-B",
        "-s",
        observer,
        "--action",
        action,
        "--config",
        config["remote_config_path"],
    ]
    if model_id is not None:
        argv.extend(["--model", model_id])
    if run_id is not None:
        argv.extend(["--run-id", run_id])
    if interval_ms is not None:
        argv.extend(["--interval-ms", str(interval_ms)])
    return argv


def parse_canonical(raw: bytes, field: str) -> dict[str, Any]:
    require(
        0 < len(raw) <= MAX_COMMAND_BYTES and raw.endswith(b"\n"),
        f"{field} framing",
    )
    value = strict_json_loads(raw, field)
    require(
        type(value) is dict and canonical_bytes(value) == raw,
        f"{field} canonical JSON",
    )
    return value


def require_remote_package(
    config: dict[str, Any],
    value: dict[str, Any],
    field: str,
) -> None:
    require(
        value.get("schema") == "s40-phone-identity-v2"
        and value.get("remote_identity_package")
        == config["remote_identity_package"]
        and value.get("remote_identity_package_sha256")
        == config["remote_identity_package_sha256"],
        f"{field} remote dependency package",
    )


def identity(
    config: dict[str, Any],
    model_id: str,
    run_id: str,
    ssh_config_sha256: str,
) -> dict[str, Any]:
    local_package = validate_local_dependencies(config)
    argv = remote_argv(config, "identity", model_id=model_id)
    started_ns = time.monotonic_ns()
    process = subprocess.run(
        argv,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        env=config["ssh_env"],
    )
    validate_local_dependencies(config)
    completed_ns = time.monotonic_ns()
    require(
        process.returncode == 0 and not process.stderr,
        "phone identity SSH command failed",
    )
    remote = parse_canonical(process.stdout, "phone identity remote")
    require_remote_package(config, remote, "phone identity")
    return {
        "argv": argv,
        "completed_ns": completed_ns,
        "exit_code": process.returncode,
        "local_identity_package": local_package,
        "local_identity_package_sha256":
            config["local_identity_package_sha256"],
        "remote": remote,
        "remote_identity_package_sha256":
            config["remote_identity_package_sha256"],
        "run_id": run_id,
        "schema": "s40-phone-identity-bridge-v2",
        "ssh_config_sha256": ssh_config_sha256,
        "started_ns": started_ns,
        "success": True,
    }


def send_stop(
    config: dict[str, Any],
    run_id: str,
) -> None:
    validate_local_dependencies(config)
    process = subprocess.run(
        remote_argv(config, "stop", run_id=run_id),
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        env=config["ssh_env"],
    )
    validate_local_dependencies(config)
    require(
        process.returncode == 0 and not process.stderr,
        "phone telemetry stop command failed",
    )
    value = parse_canonical(process.stdout, "phone telemetry stop")
    require(
        value
        == {
            "run_id": run_id,
            "schema": "s40-phone-telemetry-stop-v1",
            "success": True,
        },
        "phone telemetry stop result",
    )


def telemetry(
    config: dict[str, Any],
    model_id: str,
    run_id: str,
    interval_ms: int,
    ssh_config_sha256: str,
) -> int:
    require(RUN_ID.fullmatch(run_id) is not None, "telemetry run ID")
    local_package = validate_local_dependencies(config)
    argv = remote_argv(
        config,
        "telemetry",
        model_id=model_id,
        run_id=run_id,
        interval_ms=interval_ms,
    )
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=config["ssh_env"],
    )
    require(
        process.stdout is not None and process.stderr is not None,
        "phone telemetry pipes",
    )
    stop_requested = False
    stop_sent = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    previous_int = signal.signal(signal.SIGINT, request_stop)
    rows = 0
    saw_header = False
    saw_footer = False
    try:
        while True:
            raw = process.stdout.readline(MAX_COMMAND_BYTES + 1)
            if not raw:
                break
            remote = parse_canonical(raw, "phone telemetry remote")
            schema = remote.get("schema")
            require(
                (
                    not saw_header
                    and schema == "s40-phone-telemetry-header-v1"
                )
                or (
                    saw_header
                    and not saw_footer
                    and schema
                    in (
                        "s40-phone-telemetry-sample-v2",
                        "s40-phone-telemetry-footer-v2",
                    )
                ),
                "phone telemetry row order",
            )
            if schema == "s40-phone-telemetry-header-v1":
                require_remote_package(
                    config,
                    remote["identity"],
                    "phone telemetry",
                )
            saw_header = saw_header or (
                schema == "s40-phone-telemetry-header-v1"
            )
            saw_footer = saw_footer or (
                schema == "s40-phone-telemetry-footer-v2"
            )
            wrapper = {
                "local_received_ns": time.monotonic_ns(),
                "local_identity_package": local_package,
                "local_identity_package_sha256":
                    config["local_identity_package_sha256"],
                "remote": remote,
                "remote_argv": argv,
                "remote_identity_package_sha256":
                    config["remote_identity_package_sha256"],
                "run_id": run_id,
                "schema": "s40-phone-telemetry-bridge-v2",
                "ssh_config_sha256": ssh_config_sha256,
            }
            sys.stdout.buffer.write(canonical_bytes(wrapper))
            sys.stdout.buffer.flush()
            rows += 1
            if stop_requested and not stop_sent:
                send_stop(config, run_id)
                stop_sent = True
            if saw_footer:
                break
        if stop_requested and not stop_sent:
            send_stop(config, run_id)
            stop_sent = True
        return_code = process.wait(timeout=30)
        stderr = process.stderr.read(MAX_COMMAND_BYTES + 1)
        validate_local_dependencies(config)
        require(
            return_code == 0
            and not stderr
            and saw_header
            and saw_footer
            and rows >= 3,
            "phone telemetry stream failed",
        )
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("identity", "telemetry"),
        required=True,
    )
    parser.add_argument("--interval-ms", type=int, default=1000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ssh-config", type=Path, required=True)
    args = parser.parse_args()
    require(args.ssh_config.is_absolute(), "observer SSH config path")
    require(
        RUN_ID.fullmatch(args.run_id) is not None,
        "observer run ID",
    )
    ssh_config_sha256 = hashlib.sha256(
        args.ssh_config.read_bytes()
    ).hexdigest()
    config = load_config(args.ssh_config)
    require(args.model in config["routes"], "observer model route")
    if args.action == "identity":
        sys.stdout.buffer.write(canonical_bytes(identity(
            config,
            args.model,
            args.run_id,
            ssh_config_sha256,
        )))
        sys.stdout.buffer.flush()
        return 0
    return telemetry(
        config,
        args.model,
        args.run_id,
        args.interval_ms,
        ssh_config_sha256,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        BundleError,
        GatewayError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(f"phone observer bridge failed: {error}", file=sys.stderr)
        raise SystemExit(2)
