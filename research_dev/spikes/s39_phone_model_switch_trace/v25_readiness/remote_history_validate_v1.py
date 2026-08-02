#!/usr/bin/env python3
"""Validate the frozen V2.4 token history on its pinned RTX host."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time
from typing import Any


PLAN_SCHEMA = "s39-v25-remote-history-validation-plan-v1"
RECEIPT_SCHEMA = "s39-v25-remote-history-validation-receipt-v1"
REMOTE_SCHEMA = "s39-v25-remote-history-validation-output-v1"
SSH_TARGET = "zhihao@172.20.74.85"
REMOTE_PYTHON_PATH = "/usr/bin/python3.14"
MAX_JSON = 8 * 1024 * 1024
MAX_OUTPUT = 32 * 1024 * 1024

SSH_KEYS = {
    "boot_id_source",
    "connect_timeout_s",
    "host_key_alias",
    "identity_file_path",
    "identity_file_sha256",
    "identity_file_stat",
    "identity_public_key_fingerprint",
    "identity_public_key_path",
    "identity_public_key_sha256",
    "identity_public_key_stat",
    "known_hosts_path",
    "known_hosts_sha256",
    "known_hosts_stat",
    "nvidia_smi_path",
    "remote_python_path",
    "remote_python_sha256",
    "remote_python_stat",
    "shutdown_timeout_ms",
    "ssh_path",
    "ssh_port",
    "ssh_sha256",
    "ssh_stat",
    "ssh_keygen_path",
    "ssh_keygen_sha256",
    "ssh_keygen_stat",
    "ssh_target",
    "startup_timeout_ms",
    "gpu_uuid",
}
ARTIFACT_KEYS = {"bytes", "path", "sha256"}
STAT_KEYS = {
    "build_id",
    "ctime_ns",
    "device_id",
    "inode",
    "mode",
    "mtime_ns",
    "size",
}

VALIDATOR_WRAPPER = (
    "import runpy,sys;"
    "sys.path.insert(0,sys.argv.pop(1));"
    "runpy.run_path(sys.argv.pop(1),run_name='__main__')"
)

REMOTE_SOURCE = r'''
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

MAX_OUTPUT = 32 * 1024 * 1024

def fail(message):
    raise RuntimeError(message)

def snapshot(value):
    path = Path(value["path"])
    if not path.is_absolute():
        fail("E_REMOTE_PATH")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail("E_REMOTE_TYPE")
        digest = hashlib.sha256()
        consumed = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            consumed += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
        item.st_ctime_ns, item.st_mode,
    )
    if identity(before) != identity(after) or consumed != before.st_size:
        fail("E_REMOTE_CHANGED")
    observed = {
        "bytes": before.st_size,
        "path": str(path),
        "sha256": digest.hexdigest(),
        "stat": {
            "ctime_ns": before.st_ctime_ns,
            "device_id": before.st_dev,
            "inode": before.st_ino,
            "mode": before.st_mode,
            "mtime_ns": before.st_mtime_ns,
            "size": before.st_size,
        },
    }
    for key in ("bytes", "path", "sha256"):
        if type(observed[key]) is not type(value[key]) or observed[key] != value[key]:
            fail("E_REMOTE_IDENTITY")
    return observed

encoded_plan = __import__("base64").b64decode(sys.argv[1], validate=True)
plan = json.loads(encoded_plan.decode("ascii"))
if (
    json.dumps(
        plan,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    != encoded_plan
):
    fail("E_REMOTE_PLAN_CANONICAL")
if os.path.realpath(sys.executable) != plan["support"]["python"]["path"]:
    fail("E_REMOTE_PYTHON")
before_inputs = {
    key: snapshot(value)
    for key, value in sorted(plan["inputs"].items())
}
before_support = {
    key: snapshot(value)
    for key, value in sorted(plan["support"].items())
}
completed = subprocess.run(
    plan["command_argv"],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=plan["remote_cwd"],
    env={
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    },
    check=False,
    timeout=600,
)
if len(completed.stdout) > MAX_OUTPUT or len(completed.stderr) > MAX_OUTPUT:
    fail("E_REMOTE_OUTPUT")
after_inputs = {
    key: snapshot(value)
    for key, value in sorted(plan["inputs"].items())
}
after_support = {
    key: snapshot(value)
    for key, value in sorted(plan["support"].items())
}
if before_inputs != after_inputs or before_support != after_support:
    fail("E_REMOTE_TOCTOU")
result = {
    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip(),
    "observed_inputs": after_inputs,
    "observed_support": after_support,
    "returncode": completed.returncode,
    "schema": "s39-v25-remote-history-validation-output-v1",
    "stderr_base64": __import__("base64").b64encode(
        completed.stderr
    ).decode("ascii"),
    "stdout_base64": __import__("base64").b64encode(
        completed.stdout
    ).decode("ascii"),
}
print(
    __import__("base64").b64encode(
        (
            json.dumps(
                result,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    ).decode("ascii")
)
'''.strip()


class DriverError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DriverError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}",
    )


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def canonical_compact(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise DriverError("E_CANONICAL") from error


def canonical_bytes(value: Any) -> bytes:
    return canonical_compact(value) + b"\n"


def text(value: Any, field: str, maximum: int = 32768) -> str:
    require(
        type(value) is str
        and 0 < len(value) <= maximum
        and value.isascii()
        and "\x00" not in value
        and "\n" not in value,
        f"E_TEXT: {field}",
    )
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field, 64)
    require(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"E_DIGEST: {field}",
    )
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"E_INTEGER: {field}")
    return value


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict and set(value) == keys, f"E_KEYS: {field}")
    return value


def absolute_path(value: Any, field: str) -> str:
    value = text(value, field)
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, f"E_PATH: {field}")
    return value


def validate_stat(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, STAT_KEYS, field)
    exact(value["build_id"], None, f"{field}.build_id")
    for key in STAT_KEYS - {"build_id"}:
        integer(value[key], f"{field}.{key}")
    require(
        value["inode"] > 0
        and value["size"] > 0
        and stat.S_ISREG(value["mode"]),
        f"E_STAT: {field}",
    )
    return value


def validate_artifact(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, ARTIFACT_KEYS, field)
    integer(value["bytes"], f"{field}.bytes", 1)
    absolute_path(value["path"], f"{field}.path")
    digest(value["sha256"], f"{field}.sha256")
    return value


def validate_plan(value: Any) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "command_argv",
            "inputs",
            "remote_cwd",
            "schema",
            "ssh",
            "support",
        },
        "plan",
    )
    exact(value["schema"], PLAN_SCHEMA, "plan.schema")
    ssh = exact_keys(value["ssh"], SSH_KEYS, "plan.ssh")
    exact(ssh["boot_id_source"], "phase_fresh_snapshot", "ssh.boot_source")
    exact(absolute_path(ssh["ssh_path"], "ssh.path"), "/usr/bin/ssh", "ssh.path")
    exact(text(ssh["ssh_target"], "ssh.target", 255), SSH_TARGET, "ssh.target")
    exact(text(ssh["host_key_alias"], "ssh.alias", 255), "172.20.74.85", "ssh.alias")
    exact(integer(ssh["ssh_port"], "ssh.port", 1), 22, "ssh.port")
    require(integer(ssh["connect_timeout_s"], "ssh.timeout", 1) <= 60, "E_TIMEOUT")
    for key in (
        "identity_file_path",
        "identity_public_key_path",
        "known_hosts_path",
        "nvidia_smi_path",
        "remote_python_path",
        "ssh_keygen_path",
    ):
        absolute_path(ssh[key], f"ssh.{key}")
    exact(
        ssh["remote_python_path"],
        REMOTE_PYTHON_PATH,
        "ssh.remote_python_path",
    )
    for key in (
        "identity_file_sha256",
        "identity_public_key_sha256",
        "known_hosts_sha256",
        "remote_python_sha256",
        "ssh_keygen_sha256",
        "ssh_sha256",
    ):
        digest(ssh[key], f"ssh.{key}")
    for key in (
        "identity_file_stat",
        "identity_public_key_stat",
        "known_hosts_stat",
        "remote_python_stat",
        "ssh_keygen_stat",
        "ssh_stat",
    ):
        validate_stat(ssh[key], f"ssh.{key}")
    require(
        ssh["known_hosts_stat"]["mode"] & 0o222 == 0,
        "E_KNOWN_HOSTS_WRITABLE",
    )
    require(
        not ssh["known_hosts_path"].endswith("/.ssh/known_hosts"),
        "E_SHARED_KNOWN_HOSTS",
    )
    for key in ("connect_timeout_s", "shutdown_timeout_ms", "startup_timeout_ms"):
        integer(ssh[key], f"ssh.{key}", 1)
    require(ssh["connect_timeout_s"] <= 60, "E_TIMEOUT")
    text(ssh["gpu_uuid"], "ssh.gpu_uuid", 80)
    require(
        text(
            ssh["identity_public_key_fingerprint"],
            "ssh.fingerprint",
            128,
        ).startswith("SHA256:"),
        "E_FINGERPRINT",
    )
    inputs = exact_keys(
        value["inputs"],
        {"candidate", "corpus", "history", "tokenizer_plan"},
        "inputs",
    )
    support = exact_keys(
        value["support"],
        {"history_common", "python", "validator"},
        "support",
    )
    for group_name, group in (("inputs", inputs), ("support", support)):
        for key, artifact in group.items():
            validate_artifact(artifact, f"{group_name}.{key}")
    exact(
        support["python"]["path"],
        ssh["remote_python_path"],
        "support.python.path",
    )
    exact(
        support["python"]["sha256"],
        ssh["remote_python_sha256"],
        "support.python.sha256",
    )
    exact(
        support["python"]["bytes"],
        ssh["remote_python_stat"]["size"],
        "support.python.bytes",
    )
    remote_cwd = absolute_path(value["remote_cwd"], "remote_cwd")
    exact(
        remote_cwd,
        str(Path(inputs["candidate"]["path"]).parent),
        "remote_cwd",
    )
    command = value["command_argv"]
    require(
        type(command) is list
        and len(command) == 14
        and all(type(item) is str and item.isascii() for item in command),
        "E_COMMAND",
    )
    exact(
        command[0:6],
        [
            support["python"]["path"],
            "-I",
            "-c",
            VALIDATOR_WRAPPER,
            str(Path(support["validator"]["path"]).parent),
            support["validator"]["path"],
        ],
        "command.prefix",
    )
    for index, (flag, artifact) in enumerate(
        (
            ("--candidate", inputs["candidate"]),
            ("--corpus", inputs["corpus"]),
            ("--history", inputs["history"]),
            ("--tokenizer-plan", inputs["tokenizer_plan"]),
        )
    ):
        offset = 6 + index * 2
        exact(command[offset:offset + 2], [flag, artifact["path"]], f"command[{index}]")
    return value


def read_sealed(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_FILE_TYPE: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
        item.st_mode,
    )
    require(identity(before) == identity(after), f"E_CHANGED: {path}")
    return bytes(raw), after


def stat_record(metadata: os.stat_result) -> dict[str, Any]:
    return {
        "build_id": None,
        "ctime_ns": metadata.st_ctime_ns,
        "device_id": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "mtime_ns": metadata.st_mtime_ns,
        "size": metadata.st_size,
    }


def verify_local(
    path_value: str,
    expected_sha256: str,
    expected_stat: dict[str, Any],
    executable: bool,
) -> tuple[bytes, os.stat_result]:
    raw, metadata = read_sealed(Path(path_value))
    exact(hashlib.sha256(raw).hexdigest(), expected_sha256, f"sha256.{path_value}")
    exact(stat_record(metadata), expected_stat, f"stat.{path_value}")
    if executable:
        require(bool(metadata.st_mode & 0o111), f"E_EXECUTABLE: {path_value}")
    return raw, metadata


def parse_plan(raw_text: str, expected_sha256: str) -> dict[str, Any]:
    raw = text(raw_text, "plan_json", MAX_JSON).encode("ascii")
    exact(hashlib.sha256(raw).hexdigest(), digest(expected_sha256, "plan_sha256"), "plan.sha256")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                DriverError(f"E_JSON_NUMBER: {item}")
            ),
        )
    except json.JSONDecodeError as error:
        raise DriverError("E_PLAN_JSON") from error
    exact(canonical_compact(value), raw, "plan.canonical")
    return validate_plan(value)


def ssh_argv(plan: dict[str, Any]) -> list[str]:
    ssh = plan["ssh"]
    remote_plan = base64.b64encode(canonical_compact(plan)).decode("ascii")
    remote_command = shlex.join([
        plan["support"]["python"]["path"],
        "-I",
        "-c",
        REMOTE_SOURCE,
        remote_plan,
    ])
    return [
        ssh["ssh_path"],
        "-T",
        "-F",
        "/dev/null",
        "-p",
        str(ssh["ssh_port"]),
        "-i",
        ssh["identity_file_path"],
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={ssh['connect_timeout_s']}",
        "-o",
        f"HostKeyAlias={ssh['host_key_alias']}",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={ssh['known_hosts_path']}",
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
        ssh["ssh_target"],
        remote_command,
    ]


def public_key_fingerprint(raw: bytes) -> str:
    require(raw.endswith(b"\n") and len(raw) <= 16 * 1024, "E_SSH_PUBLIC_KEY")
    fields = raw.strip().split()
    require(len(fields) in (2, 3), "E_SSH_PUBLIC_KEY")
    try:
        blob = base64.b64decode(fields[1], validate=True)
    except (ValueError, binascii.Error) as error:
        raise DriverError("E_SSH_PUBLIC_KEY") from error
    require(bool(blob), "E_SSH_PUBLIC_KEY")
    encoded = base64.b64encode(hashlib.sha256(blob).digest()).decode("ascii")
    return "SHA256:" + encoded.rstrip("=")


def verify_ssh(plan: dict[str, Any]) -> None:
    ssh = plan["ssh"]
    verify_local(
        ssh["ssh_path"],
        ssh["ssh_sha256"],
        ssh["ssh_stat"],
        True,
    )
    verify_local(
        ssh["ssh_keygen_path"],
        ssh["ssh_keygen_sha256"],
        ssh["ssh_keygen_stat"],
        True,
    )
    verify_local(
        ssh["known_hosts_path"],
        ssh["known_hosts_sha256"],
        ssh["known_hosts_stat"],
        False,
    )
    _, private_stat = verify_local(
        ssh["identity_file_path"],
        ssh["identity_file_sha256"],
        ssh["identity_file_stat"],
        False,
    )
    public, _ = verify_local(
        ssh["identity_public_key_path"],
        ssh["identity_public_key_sha256"],
        ssh["identity_public_key_stat"],
        False,
    )
    exact(private_stat.st_uid, os.geteuid(), "ssh.identity.uid")
    exact(stat.S_IMODE(private_stat.st_mode), 0o600, "ssh.identity.mode")
    fingerprint = public_key_fingerprint(public)
    exact(
        fingerprint,
        ssh["identity_public_key_fingerprint"],
        "ssh.public_fingerprint",
    )
    completed = subprocess.run(
        [
            ssh["ssh_keygen_path"],
            "-y",
            "-f",
            ssh["identity_file_path"],
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    exact(completed.returncode, 0, "ssh_keygen.returncode")
    exact(completed.stderr, b"", "ssh_keygen.stderr")
    derived = completed.stdout
    if not derived.endswith(b"\n"):
        derived += b"\n"
    exact(
        public_key_fingerprint(derived),
        fingerprint,
        "ssh.private_fingerprint",
    )


def parse_remote(raw: bytes, expected_boot_id: str) -> dict[str, Any]:
    try:
        decoded = base64.b64decode(raw.strip(), validate=True)
        value = json.loads(
            decoded.decode("ascii"),
            object_pairs_hook=strict_object,
        )
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DriverError("E_REMOTE_OUTPUT") from error
    exact(canonical_bytes(value), decoded, "remote.canonical")
    exact_keys(
        value,
        {
            "boot_id",
            "observed_inputs",
            "observed_support",
            "returncode",
            "schema",
            "stderr_base64",
            "stdout_base64",
        },
        "remote",
    )
    exact(value["schema"], REMOTE_SCHEMA, "remote.schema")
    exact(value["boot_id"], expected_boot_id, "remote.boot_id")
    try:
        stdout = base64.b64decode(value["stdout_base64"], validate=True)
        stderr = base64.b64decode(value["stderr_base64"], validate=True)
    except ValueError as error:
        raise DriverError("E_REMOTE_STREAM") from error
    exact(value["returncode"], 0, "remote.returncode")
    exact(stdout, b"B8_HISTORY_VALIDATE_PASS\n", "remote.stdout")
    exact(stderr, b"", "remote.stderr")
    return {**value, "stdout": stdout, "stderr": stderr}


def run(
    plan: dict[str, Any],
    expected_boot_id: str,
) -> dict[str, Any]:
    verify_ssh(plan)
    started_ns = time.monotonic_ns()
    try:
        completed = subprocess.run(
            ssh_argv(plan),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            check=False,
            timeout=660,
        )
    except subprocess.TimeoutExpired as error:
        raise DriverError("E_SSH_TIMEOUT") from error
    completed_ns = time.monotonic_ns()
    require(
        len(completed.stdout) <= MAX_OUTPUT
        and len(completed.stderr) <= MAX_OUTPUT,
        "E_SSH_OUTPUT",
    )
    exact(completed.returncode, 0, "ssh.returncode")
    exact(completed.stderr, b"", "ssh.stderr")
    verify_ssh(plan)
    remote = parse_remote(completed.stdout, expected_boot_id)
    return {
        "boot_id": remote["boot_id"],
        "completed_ns": completed_ns,
        "executed_argv": plan["command_argv"],
        "host": "zhihao-Z690-C-ac",
        "observed_inputs": remote["observed_inputs"],
        "observed_support": remote["observed_support"],
        "phase": "A_ONLY",
        "plan_sha256": hashlib.sha256(canonical_compact(plan)).hexdigest(),
        "returncode": remote["returncode"],
        "schema": RECEIPT_SCHEMA,
        "ssh_target": plan["ssh"]["ssh_target"],
        "started_ns": started_ns,
        "stderr": remote["stderr"].decode("ascii"),
        "stdout": remote["stdout"].decode("ascii"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--boot-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = parse_plan(args.plan_json, args.plan_sha256)
        result = run(plan, args.boot_id)
        raw = canonical_bytes(result)
        descriptor = os.open(
            args.output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o644,
        )
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(descriptor, raw[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        return 0
    except (
        DriverError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"V25_REMOTE_HISTORY_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
