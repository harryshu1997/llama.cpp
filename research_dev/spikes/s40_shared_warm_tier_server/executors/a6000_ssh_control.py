#!/usr/bin/env python3
"""Pinned desktop-to-A6000 lifecycle control for phone routes."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from phone_gateway import (
    MAX_COMMAND_BYTES,
    GatewayError,
    canonical_bytes,
    command_argv,
    exact_keys,
    integer,
    require,
    sha256_text,
    strict_json_loads,
    string,
)


BOOT_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
REMOTE_DEPENDENCY_ROLES = {
    "a6000_phone_observer",
    "a6000_phone_route_control",
    "adb",
    "phone_gateway",
    "python",
    "readiness_v23",
    "remote_config",
}
SSH_ENV = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def public_key_fingerprint(path: Path) -> str:
    return public_key_bytes_fingerprint(path.read_bytes())


def public_key_bytes_fingerprint(raw: bytes) -> str:
    require(
        raw.endswith(b"\n") and len(raw) <= 16 * 1024,
        "SSH public key framing",
    )
    fields = raw.strip().split()
    require(len(fields) in (2, 3), "SSH public key fields")
    try:
        blob = base64.b64decode(fields[1], validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise GatewayError(f"SSH public key encoding: {error}") from error
    require(blob, "SSH public key blob")
    encoded = base64.b64encode(hashlib.sha256(blob).digest()).decode("ascii")
    return "SHA256:" + encoded.rstrip("=")


def private_key_fingerprint(
    ssh_keygen: Path,
    identity_file: Path,
) -> str:
    completed = subprocess.run(
        [str(ssh_keygen), "-y", "-f", str(identity_file)],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        env=SSH_ENV,
    )
    require(
        completed.returncode == 0
        and not completed.stderr
        and completed.stdout
        and len(completed.stdout) <= 16 * 1024,
        "SSH private key proof",
    )
    raw = completed.stdout
    if not raw.endswith(b"\n"):
        raw += b"\n"
    return public_key_bytes_fingerprint(raw)


def locked_local_file(
    path: Path,
    field: str,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    require(
        path.is_absolute() and path.is_file() and not path.is_symlink(),
        f"{field} file",
    )
    require(path.stat().st_size == expected_bytes, f"{field} size changed")
    require(file_sha256(path) == expected_sha256, f"{field} changed")
    return {
        "bytes": expected_bytes,
        "path": str(path),
        "role": field,
        "sha256": expected_sha256,
    }


def parse_remote_identity_package(
    value: Any,
    expected_sha256: str,
    remote_config_path: str,
    remote_script_path: str,
    remote_python_path: str,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {"a6000_identity", "files", "host_boot_id", "schema"},
        "remote identity package",
    )
    require(
        value["schema"] == "s40-a6000-identity-package-v1",
        "remote identity package schema",
    )
    sha256_text(value["a6000_identity"], "remote A6000 identity")
    host_boot_id = string(value["host_boot_id"], "remote host boot ID")
    require(BOOT_ID.fullmatch(host_boot_id) is not None, "remote host boot ID")
    files = value["files"]
    require(
        type(files) is list and len(files) == len(REMOTE_DEPENDENCY_ROLES),
        "remote identity package files",
    )
    normalized = []
    seen = set()
    for index, row in enumerate(files):
        row = exact_keys(
            row,
            {"bytes", "path", "role", "sha256"},
            f"remote identity package file[{index}]",
        )
        role = string(row["role"], "remote dependency role")
        require(
            role in REMOTE_DEPENDENCY_ROLES and role not in seen,
            "remote dependency role",
        )
        seen.add(role)
        path = string(row["path"], "remote dependency path")
        require(Path(path).is_absolute(), "remote dependency absolute path")
        normalized.append({
            "bytes": integer(
                row["bytes"],
                "remote dependency bytes",
                1,
            ),
            "path": path,
            "role": role,
            "sha256": sha256_text(
                row["sha256"],
                "remote dependency SHA-256",
            ),
        })
    require(seen == REMOTE_DEPENDENCY_ROLES, "remote dependency role set")
    require(
        normalized == sorted(normalized, key=lambda row: row["role"]),
        "remote dependency order",
    )
    by_role = {row["role"]: row for row in normalized}
    require(
        by_role["remote_config"]["path"] == remote_config_path
        and by_role["a6000_phone_route_control"]["path"]
        == remote_script_path
        and by_role["a6000_phone_observer"]["path"]
        == str(Path(remote_script_path).with_name("a6000_phone_observer.py"))
        and by_role["python"]["path"] == remote_python_path,
        "remote dependency path binding",
    )
    require(
        hashlib.sha256(canonical_bytes(value)).hexdigest()
        == expected_sha256,
        "remote identity package digest",
    )
    return value


def validate_local_dependencies(config: dict[str, Any]) -> dict[str, Any]:
    files = config["local_dependency_files"]
    for role, row in files.items():
        locked_local_file(
            Path(row["path"]),
            role,
            row["bytes"],
            row["sha256"],
        )
    identity_file = Path(config["identity_file_path"])
    require(
        identity_file.is_absolute()
        and identity_file.is_file()
        and not identity_file.is_symlink()
        and identity_file.stat().st_uid == os.geteuid()
        and identity_file.stat().st_mode & 0o077 == 0,
        "SSH identity private file",
    )
    require(
        public_key_fingerprint(
            Path(files["identity_public_key"]["path"])
        )
        == config["identity_public_key_fingerprint"],
        "SSH public key fingerprint changed",
    )
    require(
        private_key_fingerprint(
            Path(files["ssh_keygen"]["path"]),
            identity_file,
        )
        == config["identity_public_key_fingerprint"],
        "SSH private key does not match the pinned public key",
    )
    package = {
        "files": [
            files[role]
            for role in sorted(files)
        ],
        "identity_file_path": config["identity_file_path"],
        "identity_public_key_fingerprint":
            config["identity_public_key_fingerprint"],
        "schema": "s40-local-ssh-identity-package-v1",
        "ssh_argv": list(config["ssh_argv"]),
        "ssh_env": config["ssh_env"],
    }
    require(
        hashlib.sha256(canonical_bytes(package)).hexdigest()
        == config["local_identity_package_sha256"],
        "local SSH identity package digest",
    )
    return package


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    require(0 < len(raw) <= MAX_COMMAND_BYTES, "SSH config size")
    value = strict_json_loads(raw, "SSH config JSON")
    require(canonical_bytes(value) == raw, "SSH config is not canonical JSON")
    value = exact_keys(
        value,
        {
            "identity_file_path",
            "identity_public_key_bytes",
            "identity_public_key_fingerprint",
            "identity_public_key_path",
            "identity_public_key_sha256",
            "known_hosts_path",
            "known_hosts_bytes",
            "known_hosts_sha256",
            "remote_config_path",
            "remote_identity_package",
            "remote_identity_package_sha256",
            "remote_python_path",
            "remote_script_path",
            "routes",
            "schema",
            "ssh_argv",
            "ssh_env",
            "ssh_executable_bytes",
            "ssh_executable_sha256",
            "ssh_keygen_bytes",
            "ssh_keygen_path",
            "ssh_keygen_sha256",
        },
        "ssh_config",
    )
    require(value["schema"] == "s40-a6000-ssh-control-v3", "SSH config schema")
    known_hosts = Path(string(value["known_hosts_path"], "known hosts path"))
    known_hosts_sha256 = sha256_text(
        value["known_hosts_sha256"],
        "known hosts SHA-256",
    )
    known_hosts_bytes = integer(
        value["known_hosts_bytes"],
        "known hosts bytes",
        1,
    )
    known_hosts_row = locked_local_file(
        known_hosts,
        "known_hosts",
        known_hosts_bytes,
        known_hosts_sha256,
    )
    ssh_argv = command_argv(value["ssh_argv"], "ssh argv")
    ssh_executable = Path(ssh_argv[0])
    ssh_row = locked_local_file(
        ssh_executable,
        "ssh_executable",
        integer(
            value["ssh_executable_bytes"],
            "SSH executable bytes",
            1,
        ),
        sha256_text(
            value["ssh_executable_sha256"],
            "SSH executable SHA-256",
        ),
    )
    require(
        ssh_executable.stat().st_mode & 0o111 != 0,
        "SSH executable mode",
    )
    identity_file = Path(
        string(value["identity_file_path"], "SSH identity private path")
    )
    public_key = Path(
        string(value["identity_public_key_path"], "SSH public key path")
    )
    public_key_row = locked_local_file(
        public_key,
        "identity_public_key",
        integer(
            value["identity_public_key_bytes"],
            "SSH public key bytes",
            1,
        ),
        sha256_text(
            value["identity_public_key_sha256"],
            "SSH public key SHA-256",
        ),
    )
    fingerprint = string(
        value["identity_public_key_fingerprint"],
        "SSH public key fingerprint",
    )
    require(
        fingerprint == public_key_fingerprint(public_key),
        "SSH public key fingerprint",
    )
    ssh_keygen = Path(
        string(value["ssh_keygen_path"], "SSH keygen path")
    )
    ssh_keygen_row = locked_local_file(
        ssh_keygen,
        "ssh_keygen",
        integer(value["ssh_keygen_bytes"], "SSH keygen bytes", 1),
        sha256_text(value["ssh_keygen_sha256"], "SSH keygen SHA-256"),
    )
    require(
        ssh_keygen.stat().st_mode & 0o111 != 0,
        "SSH keygen executable mode",
    )
    ssh_env = value["ssh_env"]
    require(ssh_env == SSH_ENV, "SSH environment")
    target = ssh_argv[-1]
    require(
        type(target) is str
        and target
        and not target.startswith("-")
        and ssh_argv
        == (
            str(ssh_executable),
            "-F",
            "/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={known_hosts}",
            "-o",
            "GlobalKnownHostsFile=/dev/null",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            "IdentityAgent=none",
            "-o",
            "LogLevel=ERROR",
            "-i",
            str(identity_file),
            target,
        ),
        "SSH argv is not the frozen fail-closed form",
    )
    remote_script = string(value["remote_script_path"], "remote script path")
    remote_config = string(value["remote_config_path"], "remote config path")
    remote_python = string(value["remote_python_path"], "remote Python path")
    require(
        remote_script.startswith("/")
        and remote_config.startswith("/")
        and remote_python.startswith("/"),
        "remote paths must be absolute",
    )
    remote_package_sha256 = sha256_text(
        value["remote_identity_package_sha256"],
        "remote identity package SHA-256",
    )
    remote_package = parse_remote_identity_package(
        value["remote_identity_package"],
        remote_package_sha256,
        remote_config,
        remote_script,
        remote_python,
    )
    routes = value["routes"]
    require(type(routes) is dict and 1 <= len(routes) <= 8, "SSH routes")
    for model_id, route in routes.items():
        string(model_id, "SSH route model")
        route = exact_keys(
            route,
            {
                "a6000_identity",
                "artifact_certificate_sha256",
                "model_sha256",
                "op12_boot_id",
                "op12_shard_sha256",
                "op15_boot_id",
                "op15_shard_sha256",
                "qualification_sha256",
                "readiness_lock_sha256",
                "readiness_phase_id",
                "worker_sha256",
            },
            "SSH route",
        )
        for key in (
            "model_sha256",
            "artifact_certificate_sha256",
            "op12_shard_sha256",
            "op15_shard_sha256",
            "qualification_sha256",
            "readiness_lock_sha256",
            "worker_sha256",
        ):
            sha256_text(route[key], f"SSH route {key}")
        sha256_text(route["a6000_identity"], "SSH route A6000 identity")
        for key in ("op12_boot_id", "op15_boot_id"):
            string(route[key], f"SSH route {key}")
        string(route["readiness_phase_id"], "SSH route readiness phase ID")
        require(
            route["a6000_identity"]
            == remote_package["a6000_identity"],
            "SSH route A6000 identity package mismatch",
        )
    local_files = {
        "identity_public_key": public_key_row,
        "known_hosts": known_hosts_row,
        "ssh_executable": ssh_row,
        "ssh_keygen": ssh_keygen_row,
    }
    local_package = {
        "files": [
            local_files[role]
            for role in sorted(local_files)
        ],
        "identity_file_path": str(identity_file),
        "identity_public_key_fingerprint": fingerprint,
        "schema": "s40-local-ssh-identity-package-v1",
        "ssh_argv": list(ssh_argv),
        "ssh_env": ssh_env,
    }
    value["ssh_argv"] = ssh_argv
    value["identity_file_path"] = str(identity_file)
    value["identity_public_key_fingerprint"] = fingerprint
    value["local_dependency_files"] = local_files
    value["local_identity_package"] = local_package
    value["local_identity_package_sha256"] = hashlib.sha256(
        canonical_bytes(local_package)
    ).hexdigest()
    value["remote_identity_package"] = remote_package
    value["remote_identity_package_sha256"] = remote_package_sha256
    value["remote_python_path"] = remote_python
    value["routes"] = routes
    value["ssh_env"] = ssh_env
    validate_local_dependencies(value)
    return value


def run_control(
    config: dict[str, Any],
    action: str,
    model_id: str,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> bytes:
    require(action in ("load", "rollback", "unload"), "SSH action")
    require(model_id in config["routes"], "SSH model")
    validate_local_dependencies(config)
    argv = list(config["ssh_argv"]) + [
        config["remote_python_path"],
        "-B",
        "-s",
        config["remote_script_path"],
        "--action",
        action,
        "--config",
        config["remote_config_path"],
        "--model",
        model_id,
    ]
    completed = runner(
        argv,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3600,
        env=config["ssh_env"],
    )
    validate_local_dependencies(config)
    require(completed.returncode == 0, "A6000 SSH lifecycle command failed")
    require(not completed.stderr, "A6000 SSH lifecycle wrote stderr")
    require(
        0 < len(completed.stdout) <= MAX_COMMAND_BYTES,
        "A6000 SSH lifecycle output size",
    )
    value = strict_json_loads(completed.stdout, "A6000 SSH lifecycle JSON")
    require(
        canonical_bytes(value) == completed.stdout,
        "A6000 SSH lifecycle output is not canonical JSON",
    )
    route = config["routes"][model_id]
    if action == "load":
        value = exact_keys(
            value,
            {
                "a6000_identity",
                "artifact_certificate_sha256",
                "model_id",
                "model_sha256",
                "op12_boot_id",
                "op12_shard_sha256",
                "op15_boot_id",
                "op15_shard_sha256",
                "qualification_sha256",
                "readiness_lock_sha256",
                "readiness_phase_id",
                "route_observation",
                "route_instance_id",
                "schema",
                "success",
                "worker_sha256",
            },
            "A6000 load result",
        )
        require(value["schema"] == "s40-phone-route-load-v3", "A6000 load schema")
        require(value["success"] is True, "A6000 load failed")
        for key, expected in route.items():
            require(value[key] == expected, f"A6000 load {key} mismatch")
        route_instance_id = string(
            value["route_instance_id"],
            "A6000 route instance",
        )
        observation = exact_keys(
            value["route_observation"],
            {
                "direct_peer",
                "model_id",
                "phones",
                "route_instance_id",
                "schema",
            },
            "A6000 route observation",
        )
        require(
            observation["schema"] == "s40-phone-route-observation-v1"
            and observation["model_id"] == model_id
            and observation["route_instance_id"] == route_instance_id
            and type(observation["phones"]) is dict
            and set(observation["phones"]) == {"op12", "op15"},
            "A6000 route observation mismatch",
        )
    elif action == "unload":
        value = exact_keys(
            value,
            {
                "model_id",
                "placements",
                "route_instance_id",
                "schema",
                "success",
            },
            "A6000 unload result",
        )
        require(
            value["schema"] == "s40-phone-route-unload-v2"
            and value["success"] is True
            and value["model_id"] == model_id,
            "A6000 unload result mismatch",
        )
        string(value["route_instance_id"], "A6000 unload instance")
        require(
            type(value["placements"]) is dict
            and set(value["placements"]) == {"op12", "op15"},
            "A6000 unload placements",
        )
    else:
        value = exact_keys(
            value,
            {"model_id", "route_instance_id", "schema", "success"},
            "A6000 rollback result",
        )
        require(
            value["schema"] == "s40-phone-route-rollback-v1"
            and value["success"] is True
            and value["model_id"] == model_id,
            "A6000 rollback result mismatch",
        )
        string(value["route_instance_id"], "A6000 rollback instance")
    return canonical_bytes(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("load", "rollback", "unload"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    require(args.config.is_absolute(), "SSH config path must be absolute")
    output = run_control(load_config(args.config), args.action, args.model)
    __import__("sys").stdout.buffer.write(output)
    __import__("sys").stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GatewayError, OSError, subprocess.SubprocessError) as error:
        print(f"A6000 SSH control failed: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)
