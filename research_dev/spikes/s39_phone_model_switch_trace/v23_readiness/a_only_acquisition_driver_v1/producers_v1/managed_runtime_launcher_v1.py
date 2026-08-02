#!/usr/bin/python3 -I
"""Launch one sealed CUDA or Android runtime bundle."""

from __future__ import annotations

import argparse
import base64
import binascii
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import time
from typing import Any


PLAN_SCHEMA = "s39-managed-runtime-launch-plan-v1"
PROCESS_SCHEMA = "s39-runtime-process-source-v1"
PROCESS_PREFIX = b"RUNTIMEPROCESS "
TRANSPORT_PROCESS_SCHEMA = "s39-managed-remote-transport-process-v1"
TRANSPORT_PROCESS_PREFIX = b"TRANSPORTPROCESS "
REMOTE_CLEANUP_SCHEMA = "s39-managed-remote-cleanup-v1"
REMOTE_CLEANUP_PREFIX = b"REMOTECLEANUP "
ADB_PORT = 5038
REMOTE_CUDA_TARGET = "zhihao@172.20.74.85"
REMOTE_CUDA_HOST_KEY_ALIAS = "172.20.74.85"
SSH_ENV = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
BOOT_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
ANDROID_TIME_RE = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r"\.(?P<fraction>\d{1,9}) (?P<zone>[+-]\d{4})$"
)
MAX_JSON = 8 * 1024 * 1024
MAX_ADB_OUTPUT = 16 * 1024 * 1024
REMOTE_TOKEN_ENV = "S39_MANAGED_REMOTE_TOKEN"
REMOTE_ROOT_PREFIX = "/tmp/s39-managed-runtime-v1-"

STAT_KEYS = {
    "build_id",
    "ctime_ns",
    "device_id",
    "inode",
    "mode",
    "mtime_ns",
    "size",
}
COMPONENT_KEYS = {
    "bytes",
    "component_id",
    "path",
    "sha256",
    "stat",
}
PLAN_KEYS = {
    "android",
    "bundle_id",
    "components",
    "endpoint",
    "launcher_component_id",
    "mode",
    "route",
    "schema",
    "ssh",
}
ANDROID_KEYS = {
    "adb_path",
    "adb_port",
    "adb_selector",
    "adb_sha256",
    "boot_id_source",
    "physical_serial",
    "shutdown_timeout_ms",
    "startup_timeout_ms",
}
WORKER_ROUTE_KEYS = {
    "devices",
    "driver_batch",
    "driver_context",
    "driver_max_prefill",
    "dynamic_cut",
    "kind",
    "kv_unified",
    "layer_end",
    "layer_start",
    "mode",
    "model_path",
    "model_sha256",
    "n_gpu_layers",
    "placement_cert",
    "port",
    "runtime_root",
}
RELAY_ROUTE_KEYS = {
    "emit_direct_frames",
    "head_host",
    "head_port",
    "kind",
    "listen_port",
    "runtime_root",
    "tail_host",
    "tail_port",
    "tail_source_port",
}
LOCAL_ROUTE_KEYS = {
    "argv",
    "cwd",
    "environment",
    "kind",
}
REMOTE_ROUTE_KEYS = {
    "argv",
    "cwd",
    "environment",
    "kind",
    "local_forward",
}
LOCAL_FORWARD_KEYS = {
    "local_host",
    "local_port",
    "remote_host",
    "remote_port",
}
SSH_KEYS = {
    "boot_id_source",
    "connect_timeout_s",
    "gpu_uuid",
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
}
GPU_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F-]{16,64}$")

REMOTE_PROCESS_SCRIPT = r'''
hex() { od -An -tx1 -v "$1" 2>/dev/null | tr -d ' \n'; }
pid="$1"
printf 'SERIAL %s\n' "$(getprop ro.serialno | tr -d '\r\n')"
printf 'BOOT %s\n' "$(cat /proc/sys/kernel/random/boot_id | tr -d '\r\n')"
printf 'EXE %s\n' "$(readlink "/proc/$pid/exe" | od -An -tx1 -v | tr -d ' \n')"
printf 'CMD %s\n' "$(hex "/proc/$pid/cmdline")"
printf 'STAT %s\n' "$(hex "/proc/$pid/stat")"
'''.strip()

REMOTE_CUDA_HELPER = r'''
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

MAX_PAYLOAD = 8 * 1024 * 1024
TOKEN_ENV = "S39_MANAGED_REMOTE_TOKEN"
TOKEN_PREFIX = "/tmp/s39-managed-runtime-v1-"

def refuse(message):
    print("S39_REMOTE_REFUSED: " + message, file=sys.stderr)
    raise SystemExit(2)

def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")

def canonical_arg(index, name):
    try:
        raw = base64.b64decode(sys.argv[index], validate=True)
        if not 0 < len(raw) <= MAX_PAYLOAD:
            refuse(name + " size")
        value = json.loads(raw.decode("ascii"))
    except Exception:
        refuse(name)
    if canonical(value) != raw or type(value) is not dict:
        refuse("canonical " + name)
    return value

def keys(value, expected):
    if set(value) != set(expected):
        refuse("keys")

def string(value, name):
    if type(value) is not str or not value or "\0" in value or "\n" in value:
        refuse(name)
    return value

def integer(value, name):
    if type(value) is not int or value < 1:
        refuse(name)
    return value

def boot():
    return Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()

def check_boot(value):
    expected = string(value, "boot_id")
    if boot() != expected:
        refuse("boot_id")
    return expected

def gpu(value, executable_fd=None):
    keys(value, {"boot_id", "gpu_uuid", "nvidia_smi_path"})
    boot_id = check_boot(value["boot_id"])
    gpu_uuid = string(value["gpu_uuid"], "gpu_uuid")
    executable = string(value["nvidia_smi_path"], "nvidia_smi_path")
    pass_fds = ()
    if executable_fd is not None:
        executable = "/proc/self/fd/" + str(executable_fd)
        pass_fds = (executable_fd,)
    result = subprocess.run(
        [
            executable,
            "--id",
            gpu_uuid,
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        pass_fds=pass_fds,
        timeout=15,
    )
    try:
        observed = result.stdout.decode("ascii").strip()
    except UnicodeDecodeError:
        refuse("gpu output")
    if result.returncode != 0 or result.stderr or observed != gpu_uuid:
        refuse("gpu identity")
    return boot_id, gpu_uuid

def start_ticks(pid):
    raw = Path("/proc") / str(pid) / "stat"
    value = raw.read_bytes()
    closing = value.rfind(b")")
    if closing <= 0 or value[closing + 1:closing + 2] != b" ":
        refuse("process stat")
    fields = value[closing + 2:].split()
    try:
        observed_pid = int(value[:value.find(b" ")])
        ticks = int(fields[19])
    except (IndexError, ValueError):
        refuse("process stat")
    if observed_pid != pid or ticks < 1:
        refuse("process stat")
    return ticks

def process_stat(pid):
    raw = (Path("/proc") / str(pid) / "stat").read_bytes()
    closing = raw.rfind(b")")
    if closing <= 0 or raw[closing + 1:closing + 2] != b" ":
        refuse("process stat")
    fields = raw[closing + 2:].split()
    try:
        observed_pid = int(raw[:raw.find(b" ")])
        parent_pid = int(fields[1])
        process_group = int(fields[2])
        ticks = int(fields[19])
    except (IndexError, ValueError):
        refuse("process stat")
    if (
        observed_pid != pid
        or parent_pid < 0
        or process_group < 1
        or ticks < 1
    ):
        refuse("process stat")
    return parent_pid, process_group, ticks

def file_stat(metadata):
    return {
        "build_id": None,
        "ctime_ns": metadata.st_ctime_ns,
        "device_id": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "mtime_ns": metadata.st_mtime_ns,
        "size": metadata.st_size,
    }

def stat_row(path):
    target = Path(string(path, "path"))
    try:
        metadata = target.lstat()
    except OSError:
        refuse("stat")
    if target.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        refuse("file type")
    return file_stat(metadata)

def validate_stat(value):
    keys(
        value,
        {
            "build_id",
            "ctime_ns",
            "device_id",
            "inode",
            "mode",
            "mtime_ns",
            "size",
        },
    )
    if (
        value["build_id"] is not None
        or any(
            type(value[key]) is not int or value[key] < 0
            for key in (
                "ctime_ns",
                "device_id",
                "inode",
                "mode",
                "mtime_ns",
                "size",
            )
        )
    ):
        refuse("component stat")
    return value

def checksum(value, name):
    value = string(value, name)
    if (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        refuse(name)
    return value

def component(value):
    keys(value, {"component_id", "path", "sha256", "stat"})
    component_id = string(value["component_id"], "component id")
    if any(
        not character.isalnum() and character not in "._-"
        for character in component_id
    ):
        refuse("component id")
    path = string(value["path"], "component path")
    if not Path(path).is_absolute():
        refuse("component path")
    expected_sha256 = checksum(value["sha256"], "component sha256")
    expected_stat = validate_stat(value["stat"])
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        refuse("component open")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            refuse("component type")
        if file_stat(before) != expected_stat:
            refuse("component stat")
        observed = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            observed.update(block)
        after = os.fstat(descriptor)
        if file_stat(after) != file_stat(before):
            refuse("component changed")
        if observed.hexdigest() != expected_sha256:
            refuse("component sha256")
        os.lseek(descriptor, 0, os.SEEK_SET)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor

def launch_token(value):
    value = string(value, "launch token")
    if (
        len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        refuse("launch token")
    return value

def runtime_root(token):
    return Path(TOKEN_PREFIX + launch_token(token))

def prepare_runtime_root(token):
    root = runtime_root(token)
    try:
        root.mkdir(mode=0o700)
    except OSError:
        refuse("runtime root")

def copy_components(values, token):
    if type(values) is not list or not values:
        refuse("components")
    root = runtime_root(token)
    try:
        metadata = root.lstat()
    except OSError:
        refuse("runtime root")
    if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        refuse("runtime root")
    try:
        if any(root.iterdir()):
            refuse("runtime root")
    except OSError:
        refuse("runtime root")
    descriptors = []
    paths = {}
    digests = {}
    previous_id = None
    try:
        for index, value in enumerate(values):
            descriptor = component(value)
            descriptors.append(descriptor)
            component_id = value["component_id"]
            if previous_id is not None and previous_id >= component_id:
                refuse("component order")
            previous_id = component_id
            if value["path"] in paths:
                refuse("component path reuse")
            destination = root / ("%04d" % index)
            output = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
            )
            copied = hashlib.sha256()
            try:
                while True:
                    block = os.read(descriptor, 1024 * 1024)
                    if not block:
                        break
                    copied.update(block)
                    offset = 0
                    while offset < len(block):
                        written = os.write(output, block[offset:])
                        if written < 1:
                            refuse("component copy")
                        offset += written
                os.fsync(output)
                os.fchmod(output, value["stat"]["mode"] & 0o777)
            finally:
                os.close(output)
            if copied.hexdigest() != value["sha256"]:
                refuse("component copy")
            paths[value["path"]] = str(destination)
            digests[component_id] = value["sha256"]
        os.chmod(root, 0o500)
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        remove_runtime_root(token)
        raise
    for descriptor in descriptors:
        os.close(descriptor)
    return paths, digests

def remove_runtime_root(token):
    root = runtime_root(token)
    try:
        metadata = root.lstat()
    except FileNotFoundError:
        return
    except OSError:
        refuse("runtime root")
    if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        refuse("runtime root")
    try:
        os.chmod(root, 0o700)
        for child in root.iterdir():
            child_metadata = child.lstat()
            if child.is_symlink() or not stat.S_ISREG(child_metadata.st_mode):
                refuse("runtime component")
            child.unlink()
        root.rmdir()
    except OSError:
        refuse("runtime root cleanup")

def check_interpreter(value):
    keys(value, {"path", "sha256", "stat"})
    path = string(value["path"], "interpreter path")
    checksum = string(value["sha256"], "interpreter sha256")
    if (
        len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        refuse("interpreter sha256")
    expected_stat = value["stat"]
    keys(
        expected_stat,
        {
            "build_id",
            "ctime_ns",
            "device_id",
            "inode",
            "mode",
            "mtime_ns",
            "size",
        },
    )
    if (
        expected_stat["build_id"] is not None
        or any(
            type(expected_stat[key]) is not int or expected_stat[key] < 0
            for key in (
                "ctime_ns",
                "device_id",
                "inode",
                "mode",
                "mtime_ns",
                "size",
            )
        )
    ):
        refuse("interpreter stat")
    if os.path.realpath(sys.executable) != path:
        refuse("interpreter executable")
    if stat_row(path) != expected_stat:
        refuse("interpreter stat")
    observed = hashlib.sha256()
    try:
        with open(path, "rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                observed.update(chunk)
    except OSError:
        refuse("interpreter hash")
    if observed.hexdigest() != checksum:
        refuse("interpreter hash")

def matching_processes(token, process_groups, extra_pids):
    expected = (TOKEN_ENV + "=" + launch_token(token)).encode("ascii")
    table = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            parent_pid, pgid, ticks = process_stat(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            continue
        try:
            environ = (entry / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            environ = []
        except OSError:
            environ = []
        table[pid] = {
            "parent_pid": parent_pid,
            "pgid": pgid,
            "pid": pid,
            "start_ticks": ticks,
            "token": expected in environ,
        }
    matched = {
        pid
        for pid, row in table.items()
        if (
            row["token"]
            or row["pgid"] in process_groups
            or pid in extra_pids
        )
    }
    changed = True
    while changed:
        before = len(matched)
        matched.update(
            pid
            for pid, row in table.items()
            if row["parent_pid"] in matched
        )
        changed = len(matched) != before
    return [
        {
            "pgid": table[pid]["pgid"],
            "pid": pid,
            "start_ticks": table[pid]["start_ticks"],
        }
        for pid in sorted(matched)
    ]

def compute_pids(executable_fd, gpu_uuid):
    executable = "/proc/self/fd/" + str(executable_fd)
    result = subprocess.run(
        [
            executable,
            "--id",
            gpu_uuid,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        pass_fds=(executable_fd,),
        timeout=15,
    )
    if result.returncode != 0 or result.stderr:
        refuse("nvml query")
    try:
        lines = result.stdout.decode("ascii").splitlines()
        pids = [
            int(line.strip())
            for line in lines
            if line.strip() and line.strip() != "[N/A]"
        ]
    except (UnicodeDecodeError, ValueError):
        refuse("nvml query")
    if any(pid < 1 for pid in pids):
        refuse("nvml query")
    return sorted(set(pids))

def cleanup(value):
    keys(
        value,
        {
            "boot_id",
            "gpu_uuid",
            "launch_token",
            "nvidia_smi",
            "pgid",
            "pid",
            "shutdown_timeout_ms",
            "start_ticks",
        },
    )
    boot_id = check_boot(value["boot_id"])
    token = launch_token(value["launch_token"])
    gpu_uuid = string(value["gpu_uuid"], "gpu_uuid")
    pid = value["pid"]
    ticks = value["start_ticks"]
    process_group = value["pgid"]
    timeout_ms = integer(value["shutdown_timeout_ms"], "shutdown timeout")
    if timeout_ms > 600000:
        refuse("shutdown timeout")
    if (
        type(pid) is not int
        or type(ticks) is not int
        or type(process_group) is not int
        or min(pid, ticks, process_group) < 0
        or bool(pid) != bool(ticks)
        or bool(pid) != bool(process_group)
    ):
        refuse("process identity")
    nvidia_descriptor = component(value["nvidia_smi"])
    try:
        checked_boot, checked_gpu = gpu(
            {
                "boot_id": boot_id,
                "gpu_uuid": gpu_uuid,
                "nvidia_smi_path": value["nvidia_smi"]["path"],
            },
            nvidia_descriptor,
        )
        if checked_boot != boot_id or checked_gpu != gpu_uuid:
            refuse("gpu identity")
        process_groups = {process_group} if process_group else set()
        nvml_matches = compute_pids(nvidia_descriptor, gpu_uuid)
        rows = matching_processes(
            token,
            process_groups,
            set(nvml_matches) | ({pid} if pid else set()),
        )
        process_groups.update(row["pgid"] for row in rows)
        if pid:
            root_rows = [row for row in rows if row["pid"] == pid]
            if root_rows and root_rows[0]["start_ticks"] != ticks:
                refuse("start_ticks")
        tracked = {
            (row["pid"], row["start_ticks"])
            for row in rows
        }
        if pid:
            tracked.add((pid, ticks))

        def signal_rows(number):
            groups = sorted({row["pgid"] for row in rows})
            for pgid in groups:
                try:
                    os.killpg(pgid, number)
                except ProcessLookupError:
                    pass
            for row in rows:
                try:
                    os.kill(row["pid"], number)
                except ProcessLookupError:
                    pass

        if rows:
            signal_rows(signal.SIGTERM)
        deadline = time.monotonic() + timeout_ms / 1000
        while rows and time.monotonic() < deadline:
            time.sleep(0.05)
            nvml_matches = compute_pids(nvidia_descriptor, gpu_uuid)
            rows = matching_processes(
                token,
                process_groups,
                set(nvml_matches) | ({pid} if pid else set()),
            )
            process_groups.update(row["pgid"] for row in rows)
            tracked.update(
                (row["pid"], row["start_ticks"])
                for row in rows
            )
        if rows:
            signal_rows(signal.SIGKILL)
        deadline = time.monotonic() + timeout_ms / 1000
        while rows and time.monotonic() < deadline:
            time.sleep(0.01)
            nvml_matches = compute_pids(nvidia_descriptor, gpu_uuid)
            rows = matching_processes(
                token,
                process_groups,
                set(nvml_matches) | ({pid} if pid else set()),
            )
            process_groups.update(row["pgid"] for row in rows)
            tracked.update(
                (row["pid"], row["start_ticks"])
                for row in rows
            )
        if rows:
            refuse("cleanup processes")
        deadline = time.monotonic() + timeout_ms / 1000
        nvml_matches = compute_pids(nvidia_descriptor, gpu_uuid)
        while nvml_matches and time.monotonic() < deadline:
            time.sleep(0.05)
            nvml_matches = compute_pids(nvidia_descriptor, gpu_uuid)
        if nvml_matches:
            refuse("cleanup nvml")
    finally:
        os.close(nvidia_descriptor)
    remove_runtime_root(token)
    return {
        "absent": [
            {"pid": pid, "start_ticks": ticks}
            for pid, ticks in sorted(tracked)
        ],
        "boot_id": boot_id,
        "clock": "RTX_CLOCK_MONOTONIC_RAW",
        "gpu_uuid": gpu_uuid,
        "launch_token": token,
        "matching_nvml_pids": nvml_matches,
        "matching_process_groups": sorted({
            row["pgid"]
            for row in rows
        }),
        "matching_processes": rows,
        "observed_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
        "pgid": process_group,
        "pid": pid,
        "schema": "s39-managed-remote-cleanup-v1",
        "start_ticks": ticks,
    }

def emit(value):
    sys.stdout.buffer.write(canonical(value) + b"\n")
    sys.stdout.buffer.flush()

if len(sys.argv) != 4:
    refuse("argv")
action = sys.argv[1]
value = canonical_arg(2, "payload")
check_interpreter(canonical_arg(3, "interpreter"))

if action == "identity":
    boot_id, gpu_uuid = gpu(value)
    emit({"boot_id": boot_id, "gpu_uuid": gpu_uuid})
elif action == "stat":
    keys(value, {"component_id", "path", "sha256", "stat"})
    descriptor = component(value)
    os.close(descriptor)
    emit({"sha256": value["sha256"], "stat": value["stat"]})
elif action == "prepare":
    keys(value, {"boot_id", "launch_token"})
    boot_id = check_boot(value["boot_id"])
    token = launch_token(value["launch_token"])
    prepare_runtime_root(token)
    emit({"boot_id": boot_id, "launch_token": token})
elif action == "launch":
    keys(
        value,
        {
            "argv",
            "boot_id",
            "components",
            "cwd",
            "environment",
            "gpu_uuid",
            "launch_token",
            "nvidia_smi_path",
        },
    )
    token = launch_token(value["launch_token"])
    os.environ[TOKEN_ENV] = token
    paths, component_sha256 = copy_components(value["components"], token)
    nvidia_path = paths.get(value["nvidia_smi_path"])
    if nvidia_path is None:
        refuse("nvidia component")
    boot_id, gpu_uuid = gpu({
        "boot_id": value["boot_id"],
        "gpu_uuid": value["gpu_uuid"],
        "nvidia_smi_path": nvidia_path,
    })
    argv = value["argv"]
    environment = value["environment"]
    if (
        type(argv) is not list
        or not argv
        or any(type(item) is not str or not item for item in argv)
        or not Path(argv[0]).is_absolute()
        or type(environment) is not dict
        or any(
            type(key) is not str
            or not key
            or "=" in key
            or type(item) is not str
            for key, item in environment.items()
        )
    ):
        refuse("launch payload")
    cwd = Path(string(value["cwd"], "cwd"))
    if not cwd.is_absolute():
        refuse("cwd")
    argv = [paths.get(item, item) for item in argv]
    environment = {
        key: paths.get(item, item)
        for key, item in environment.items()
    }
    if TOKEN_ENV in environment:
        refuse("launch environment")
    environment[TOKEN_ENV] = token
    os.chdir(cwd)
    pid = os.getpid()
    try:
        os.setpgid(0, 0)
    except PermissionError:
        if os.getpgrp() != pid:
            refuse("process group")
    unused_parent_pid, process_group, ticks = process_stat(pid)
    del unused_parent_pid
    if process_group != pid:
        refuse("process group")
    marker = {
        "boot_id": boot_id,
        "component_sha256": component_sha256,
        "gpu_uuid": gpu_uuid,
        "launch_token": token,
        "pgid": process_group,
        "pid": pid,
        "remote_observed_ns": time.clock_gettime_ns(
            time.CLOCK_MONOTONIC_RAW
        ),
        "start_ticks": ticks,
    }
    sys.stdout.buffer.write(b"S39CUDA " + canonical(marker) + b"\n")
    sys.stdout.buffer.flush()
    os.execve(argv[0], argv, environment)
elif action == "process":
    keys(
        value,
        {
            "argv",
            "boot_id",
            "executable_path",
            "gpu_uuid",
            "nvidia_smi_path",
            "pid",
            "start_ticks",
        },
    )
    boot_id, gpu_uuid = gpu({
        "boot_id": value["boot_id"],
        "gpu_uuid": value["gpu_uuid"],
        "nvidia_smi_path": value["nvidia_smi_path"],
    })
    pid = integer(value["pid"], "pid")
    ticks = start_ticks(pid)
    if ticks != integer(value["start_ticks"], "start_ticks"):
        refuse("start_ticks")
    process_root = Path("/proc") / str(pid)
    executable = os.readlink(process_root / "exe")
    cmdline = (process_root / "cmdline").read_bytes()
    if not cmdline.endswith(b"\0"):
        refuse("cmdline")
    try:
        argv = [item.decode("ascii") for item in cmdline[:-1].split(b"\0")]
    except UnicodeDecodeError:
        refuse("cmdline")
    if executable != value["executable_path"] or argv != value["argv"]:
        refuse("process identity")
    if start_ticks(pid) != ticks:
        refuse("process changed")
    emit({
        "argv": argv,
        "boot_id": boot_id,
        "executable_path": executable,
        "gpu_uuid": gpu_uuid,
        "pid": pid,
        "start_ticks": ticks,
    })
elif action == "signal":
    keys(value, {"boot_id", "pid", "signal", "start_ticks"})
    boot_id = check_boot(value["boot_id"])
    pid = integer(value["pid"], "pid")
    expected_ticks = integer(value["start_ticks"], "start_ticks")
    selected = string(value["signal"], "signal")
    if selected not in {"0", "TERM", "KILL"}:
        refuse("signal")
    try:
        ticks = start_ticks(pid)
    except FileNotFoundError:
        emit({"alive": False, "boot_id": boot_id})
        raise SystemExit(0)
    if ticks != expected_ticks:
        refuse("start_ticks")
    number = {
        "0": 0,
        "TERM": signal.SIGTERM,
        "KILL": signal.SIGKILL,
    }[selected]
    os.kill(pid, number)
    emit({"alive": True, "boot_id": boot_id})
elif action == "cleanup":
    emit(cleanup(value))
else:
    refuse("action")
'''.strip()


class LaunchError(RuntimeError):
    pass


class SubprocessRunner:
    def run(
        self,
        argv: list[str],
        *,
        timeout: float,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env=env,
        )

    def popen(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
    ) -> subprocess.Popen:
        return subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchError(message)


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


def canonical_bytes(value: Any) -> bytes:
    try:
        raw = (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise LaunchError("E_CANONICAL") from error
    require(len(raw) <= MAX_JSON, "E_JSON_SIZE")
    return raw


def canonical_compact(value: Any) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise LaunchError("E_CANONICAL") from error
    require(0 < len(raw) <= MAX_JSON, "E_JSON_SIZE")
    return raw


def exact_keys(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    require(set(value) == keys, f"E_KEYS: {field}")
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"E_INTEGER: {field}")
    return value


def text(value: Any, field: str, maximum: int = 4096) -> str:
    require(type(value) is str and 0 < len(value) <= maximum, f"E_TEXT: {field}")
    require(
        "\x00" not in value
        and "\n" not in value
        and all(0x20 <= ord(character) <= 0x7E for character in value),
        f"E_ASCII: {field}",
    )
    return value


def digest(value: Any, field: str) -> str:
    value = text(value, field, 64)
    require(DIGEST_RE.fullmatch(value) is not None, f"E_DIGEST: {field}")
    return value


def absolute_path(value: Any, field: str) -> str:
    value = text(value, field)
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, f"E_PATH: {field}")
    return value


def validate_argv(value: Any, field: str) -> list[str]:
    require(
        type(value) is list
        and bool(value)
        and len(value) <= 256,
        f"E_ARGV: {field}",
    )
    for index, item in enumerate(value):
        text(item, f"{field}[{index}]", 32 * 1024)
    return value


def validate_environment(value: Any, field: str) -> dict[str, str]:
    require(type(value) is dict and len(value) <= 128, f"E_ENV: {field}")
    for key, item in value.items():
        text(key, f"{field}.key", 128)
        text(item, f"{field}.{key}", 32 * 1024)
        require("=" not in key, f"E_ENV_KEY: {field}.{key}")
    return value


def validate_stat(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, STAT_KEYS, field)
    for key in ("ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"):
        integer(value[key], f"{field}.{key}")
    require(
        value["inode"] > 0
        and value["size"] > 0
        and stat.S_ISREG(value["mode"]),
        f"E_STAT: {field}",
    )
    exact(value["build_id"], None, f"{field}.build_id")
    return value


def validate_component(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, COMPONENT_KEYS, field)
    component_id = text(value["component_id"], f"{field}.component_id", 128)
    require(
        all(character.isalnum() or character in "._-" for character in component_id),
        f"E_COMPONENT_ID: {field}",
    )
    absolute_path(value["path"], f"{field}.path")
    size = integer(value["bytes"], f"{field}.bytes", 1)
    digest(value["sha256"], f"{field}.sha256")
    record = validate_stat(value["stat"], f"{field}.stat")
    exact(record["size"], size, f"{field}.stat.size")
    return value


def build_runtime_command(
    launcher_path: str,
    route: dict[str, Any],
) -> tuple[list[str], dict[str, str], str]:
    kind = text(route.get("kind"), "route.kind", 64)
    if kind == "stagenet_worker":
        exact_keys(route, WORKER_ROUTE_KEYS, "route")
        model_path = absolute_path(route["model_path"], "route.model_path")
        runtime_root = absolute_path(route["runtime_root"], "route.runtime_root")
        mode = text(route["mode"], "route.mode", 32)
        require(mode in ("stagenet", "tailv3", "monov3"), "E_WORKER_MODE")
        port = integer(route["port"], "route.port", 1)
        require(port <= 65535, "E_WORKER_PORT")
        layer_start = integer(route["layer_start"], "route.layer_start")
        layer_end = integer(route["layer_end"], "route.layer_end", 1)
        require(layer_start < layer_end, "E_LAYER_RANGE")
        driver_batch = integer(route["driver_batch"], "route.driver_batch", 1)
        driver_context = integer(
            route["driver_context"],
            "route.driver_context",
            1,
        )
        driver_max_prefill = integer(
            route["driver_max_prefill"],
            "route.driver_max_prefill",
            1,
        )
        require(
            driver_batch <= 4096
            and driver_context <= 1_048_576
            and driver_max_prefill <= driver_context,
            "E_WORKER_CAPACITY",
        )
        devices = text(route["devices"], "route.devices", 256)
        n_gpu_layers = integer(
            route["n_gpu_layers"],
            "route.n_gpu_layers",
        )
        model_sha256 = digest(route["model_sha256"], "route.model_sha256")
        for key in ("placement_cert", "kv_unified", "dynamic_cut"):
            require(type(route[key]) is bool, f"E_BOOL: route.{key}")
        argv = [
            launcher_path,
            "-m",
            model_path,
            "--mode",
            mode,
            "--port",
            str(port),
            "--driver-batch",
            str(driver_batch),
            "--driver-context",
            str(driver_context),
            "--driver-max-prefill",
            str(driver_max_prefill),
            "--devices",
            devices,
            "-ngl",
            str(n_gpu_layers),
        ]
        environment = {
            "ADSP_LIBRARY_PATH": runtime_root,
            "LAYERSPLIT_MODEL_SHA256": model_sha256,
            "LD_LIBRARY_PATH": runtime_root,
            "LLAMA_LAYER_END": str(layer_end),
            "LLAMA_LAYER_START": str(layer_start),
            "PATH": "/system/bin:/system/xbin",
        }
        if route["placement_cert"]:
            environment["LAYERSPLIT_PLACEMENT_CERT"] = "1"
        if route["kv_unified"]:
            environment["LAYERSPLIT_KV_UNIFIED"] = "1"
        if route["dynamic_cut"]:
            environment["LAYERSPLIT_DYNAMIC_CUT"] = "1"
        return argv, environment, runtime_root
    if kind == "direct_relay":
        exact_keys(route, RELAY_ROUTE_KEYS, "route")
        runtime_root = absolute_path(route["runtime_root"], "route.runtime_root")
        listen_port = integer(route["listen_port"], "route.listen_port", 1)
        head_port = integer(route["head_port"], "route.head_port", 1)
        tail_port = integer(route["tail_port"], "route.tail_port", 1)
        tail_source_port = integer(
            route["tail_source_port"],
            "route.tail_source_port",
            1,
        )
        require(
            max(listen_port, head_port, tail_port, tail_source_port) <= 65535,
            "E_RELAY_PORT",
        )
        head_host = text(route["head_host"], "route.head_host", 255)
        tail_host = text(route["tail_host"], "route.tail_host", 255)
        require(type(route["emit_direct_frames"]) is bool, "E_BOOL: route.emit")
        argv = [
            launcher_path,
            "--listen",
            str(listen_port),
            "--head",
            f"{head_host}:{head_port}",
            "--tail",
            f"{tail_host}:{tail_port}",
            "--tail-source-port",
            str(tail_source_port),
        ]
        if route["emit_direct_frames"]:
            argv.append("--emit-direct-frames")
        return argv, {
            "LD_LIBRARY_PATH": runtime_root,
            "PATH": "/system/bin:/system/xbin",
        }, runtime_root
    if kind in ("local_exec", "remote_exec"):
        exact_keys(
            route,
            REMOTE_ROUTE_KEYS if kind == "remote_exec" else LOCAL_ROUTE_KEYS,
            "route",
        )
        argv = validate_argv(route["argv"], "route.argv")
        exact(argv[0], launcher_path, "route.argv[0]")
        cwd = absolute_path(route["cwd"], "route.cwd")
        environment = validate_environment(route["environment"], "route.environment")
        if kind == "remote_exec":
            forward = exact_keys(
                route["local_forward"],
                LOCAL_FORWARD_KEYS,
                "route.local_forward",
            )
            for key in ("local_host", "remote_host"):
                exact(
                    forward[key],
                    "127.0.0.1",
                    f"route.local_forward.{key}",
                )
            for key in ("local_port", "remote_port"):
                port = integer(
                    forward[key],
                    f"route.local_forward.{key}",
                    1,
                )
                require(port <= 65535, "E_FORWARD_PORT")
        return argv, environment, cwd
    raise LaunchError("E_ROUTE_KIND")


def validate_plan(value: Any) -> dict[str, Any]:
    require(type(value) is dict, "E_TYPE: plan")
    if set(value) == PLAN_KEYS - {"ssh"}:
        require(
            value.get("mode") in ("android", "local_cuda"),
            "E_KEYS: plan",
        )
        value = {**value, "ssh": None}
    else:
        value = exact_keys(value, PLAN_KEYS, "plan")
    exact(value["schema"], PLAN_SCHEMA, "plan.schema")
    mode = text(value["mode"], "plan.mode", 32)
    require(mode in ("android", "local_cuda", "remote_cuda"), "E_MODE")
    endpoint = text(value["endpoint"], "plan.endpoint", 32)
    require(
        endpoint in ({"op12", "op15"} if mode == "android" else {"cuda"}),
        "E_ENDPOINT",
    )
    bundle_id = text(value["bundle_id"], "plan.bundle_id", 128)
    require(
        all(character.isalnum() or character in "._-" for character in bundle_id),
        "E_BUNDLE_ID",
    )
    components = value["components"]
    require(type(components) is list and bool(components), "E_COMPONENTS")
    component_map = {}
    paths = []
    for index, component in enumerate(components):
        component = validate_component(component, f"components[{index}]")
        component_id = component["component_id"]
        require(component_id not in component_map, "E_COMPONENT_REUSE")
        component_map[component_id] = component
        paths.append(component["path"])
    exact(
        list(component_map),
        sorted(component_map),
        "components.order",
    )
    require(len(paths) == len(set(paths)), "E_COMPONENT_PATH_REUSE")
    launcher_id = text(
        value["launcher_component_id"],
        "plan.launcher_component_id",
        128,
    )
    require(launcher_id in component_map, "E_LAUNCHER_COMPONENT")
    launcher_path = component_map[launcher_id]["path"]
    argv, environment, cwd = build_runtime_command(launcher_path, value["route"])
    value["_normalized"] = {
        "argv": argv,
        "component_map": component_map,
        "cwd": cwd,
        "environment": environment,
        "launcher_path": launcher_path,
    }
    if mode == "android":
        android = exact_keys(value["android"], ANDROID_KEYS, "plan.android")
        adb_path = absolute_path(android["adb_path"], "android.adb_path")
        digest(android["adb_sha256"], "android.adb_sha256")
        exact(integer(android["adb_port"], "android.adb_port", 1), ADB_PORT, "adb.port")
        selector = text(android["adb_selector"], "android.adb_selector", 255)
        require(":" in selector, "E_ADB_SELECTOR")
        serial = text(android["physical_serial"], "android.physical_serial", 255)
        require(selector != serial, "E_ADB_SELECTOR")
        exact(
            android["boot_id_source"],
            "phase_fresh_snapshot",
            "android.boot_id_source",
        )
        for key in ("startup_timeout_ms", "shutdown_timeout_ms"):
            timeout = integer(android[key], f"android.{key}", 1)
            require(timeout <= 600_000, f"E_TIMEOUT: android.{key}")
        value["_normalized"]["adb_path"] = adb_path
        exact(value["ssh"], None, "plan.ssh")
        require(value["route"]["kind"] == "stagenet_worker"
                or value["route"]["kind"] == "direct_relay", "E_ANDROID_ROUTE")
    elif mode == "remote_cuda":
        exact(value["android"], None, "plan.android")
        require(value["route"]["kind"] == "remote_exec", "E_REMOTE_ROUTE")
        ssh = exact_keys(value["ssh"], SSH_KEYS, "plan.ssh")
        exact(ssh["boot_id_source"], "phase_fresh_snapshot", "ssh.boot_id_source")
        ssh_path = absolute_path(ssh["ssh_path"], "ssh.ssh_path")
        exact(ssh_path, "/usr/bin/ssh", "ssh.ssh_path")
        digest(ssh["ssh_sha256"], "ssh.ssh_sha256")
        exact(
            text(ssh["ssh_target"], "ssh.ssh_target", 255),
            REMOTE_CUDA_TARGET,
            "ssh.ssh_target",
        )
        exact(
            text(ssh["host_key_alias"], "ssh.host_key_alias", 255),
            REMOTE_CUDA_HOST_KEY_ALIAS,
            "ssh.host_key_alias",
        )
        exact(integer(ssh["ssh_port"], "ssh.ssh_port", 1), 22, "ssh.ssh_port")
        connect_timeout = integer(
            ssh["connect_timeout_s"],
            "ssh.connect_timeout_s",
            1,
        )
        require(connect_timeout <= 60, "E_SSH_CONNECT_TIMEOUT")
        for key in (
            "known_hosts_path",
            "identity_file_path",
            "identity_public_key_path",
            "ssh_keygen_path",
        ):
            absolute_path(ssh[key], f"ssh.{key}")
        for key in (
            "known_hosts_sha256",
            "identity_file_sha256",
            "identity_public_key_sha256",
            "remote_python_sha256",
            "ssh_keygen_sha256",
        ):
            digest(ssh[key], f"ssh.{key}")
        for key in (
            "ssh_stat",
            "known_hosts_stat",
            "identity_file_stat",
            "identity_public_key_stat",
            "remote_python_stat",
            "ssh_keygen_stat",
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
        fingerprint = text(
            ssh["identity_public_key_fingerprint"],
            "ssh.identity_public_key_fingerprint",
            128,
        )
        require(fingerprint.startswith("SHA256:"), "E_SSH_FINGERPRINT")
        nvidia_smi = absolute_path(
            ssh["nvidia_smi_path"],
            "ssh.nvidia_smi_path",
        )
        remote_python = absolute_path(
            ssh["remote_python_path"],
            "ssh.remote_python_path",
        )
        gpu_uuid = text(ssh["gpu_uuid"], "ssh.gpu_uuid", 80)
        require(GPU_UUID_RE.fullmatch(gpu_uuid) is not None, "E_GPU_UUID")
        for key in ("startup_timeout_ms", "shutdown_timeout_ms"):
            timeout = integer(ssh[key], f"ssh.{key}", 1)
            require(timeout <= 600_000, f"E_TIMEOUT: ssh.{key}")
        environment = value["_normalized"]["environment"]
        require(REMOTE_TOKEN_ENV not in environment, "E_REMOTE_TOKEN_ENV")
        exact(
            environment.get("CUDA_VISIBLE_DEVICES"),
            gpu_uuid,
            "route.environment.CUDA_VISIBLE_DEVICES",
        )
        component_paths = {
            component["path"]
            for component in components
        }
        require(
            {nvidia_smi, remote_python}.issubset(component_paths),
            "E_REMOTE_HELPER_COMPONENT",
        )
        remote_python_component = next(
            component
            for component in components
            if component["path"] == remote_python
        )
        exact(
            remote_python_component["sha256"],
            ssh["remote_python_sha256"],
            "ssh.remote_python_sha256",
        )
        exact(
            remote_python_component["stat"],
            ssh["remote_python_stat"],
            "ssh.remote_python_stat",
        )
        require(
            ssh["remote_python_stat"]["mode"] & 0o111,
            "E_REMOTE_PYTHON_EXECUTABLE",
        )
        value["_normalized"]["ssh_path"] = ssh_path
    else:
        exact(value["android"], None, "plan.android")
        exact(value["ssh"], None, "plan.ssh")
        require(value["route"]["kind"] == "local_exec", "E_LOCAL_ROUTE")
    return value


def parse_plan_json(raw_text: str, expected_sha256: str) -> dict[str, Any]:
    text(raw_text, "plan_json", MAX_JSON)
    digest(expected_sha256, "plan_sha256")
    raw = raw_text.encode("ascii")
    exact(hashlib.sha256(raw).hexdigest(), expected_sha256, "plan.sha256")
    try:
        value = json.loads(
            raw_text,
            object_pairs_hook=strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                LaunchError(f"E_JSON_NUMBER: {item}")
            ),
        )
    except json.JSONDecodeError as error:
        raise LaunchError("E_PLAN_JSON") from error
    exact(canonical_compact(value), raw, "plan.canonical")
    return validate_plan(value)


def read_sealed_local(path: Path) -> tuple[bytes, os.stat_result]:
    require(path.is_absolute(), "E_LOCAL_PATH")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LaunchError(f"E_LOCAL_OPEN: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_LOCAL_TYPE: {path}")
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
    require(identity(before) == identity(after), f"E_LOCAL_CHANGED: {path}")
    require(len(raw) == before.st_size, f"E_LOCAL_CHANGED: {path}")
    return bytes(raw), after


def stat_sealed_local(path: Path) -> os.stat_result:
    require(path.is_absolute(), "E_LOCAL_PATH")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LaunchError(f"E_LOCAL_OPEN: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"E_LOCAL_TYPE: {path}")
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
    require(identity(before) == identity(after), f"E_LOCAL_CHANGED: {path}")
    return after


def local_dependency(component: dict[str, Any]) -> dict[str, Any]:
    metadata = stat_sealed_local(Path(component["path"]))
    actual = {
        "build_id": None,
        "ctime_ns": metadata.st_ctime_ns,
        "device_id": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "mtime_ns": metadata.st_mtime_ns,
        "path": component["path"],
        "size": metadata.st_size,
    }
    expected = {**component["stat"], "path": component["path"]}
    exact(actual, expected, f"{component['component_id']}.stat")
    return actual


def verify_adb(android: dict[str, Any]) -> None:
    raw, metadata = read_sealed_local(Path(android["adb_path"]))
    exact(
        hashlib.sha256(raw).hexdigest(),
        android["adb_sha256"],
        "adb.sha256",
    )
    require(metadata.st_mode & 0o111, "E_ADB_EXECUTABLE")


def public_key_fingerprint(raw: bytes) -> str:
    require(raw.endswith(b"\n") and len(raw) <= 16 * 1024, "E_SSH_PUBLIC_KEY")
    fields = raw.strip().split()
    require(len(fields) in (2, 3), "E_SSH_PUBLIC_KEY")
    try:
        blob = base64.b64decode(fields[1], validate=True)
    except (ValueError, binascii.Error) as error:
        raise LaunchError("E_SSH_PUBLIC_KEY") from error
    require(bool(blob), "E_SSH_PUBLIC_KEY")
    encoded = base64.b64encode(hashlib.sha256(blob).digest()).decode("ascii")
    return "SHA256:" + encoded.rstrip("=")


def verify_ssh(ssh: dict[str, Any]) -> None:
    rows = {}
    for name, path_key, digest_key, stat_key in (
        ("ssh", "ssh_path", "ssh_sha256", "ssh_stat"),
        (
            "known_hosts",
            "known_hosts_path",
            "known_hosts_sha256",
            "known_hosts_stat",
        ),
        (
            "identity",
            "identity_file_path",
            "identity_file_sha256",
            "identity_file_stat",
        ),
        (
            "identity_public",
            "identity_public_key_path",
            "identity_public_key_sha256",
            "identity_public_key_stat",
        ),
        (
            "ssh_keygen",
            "ssh_keygen_path",
            "ssh_keygen_sha256",
            "ssh_keygen_stat",
        ),
    ):
        raw, metadata = read_sealed_local(Path(ssh[path_key]))
        exact(
            hashlib.sha256(raw).hexdigest(),
            ssh[digest_key],
            f"ssh.{name}.sha256",
        )
        observed = {
            "build_id": None,
            "ctime_ns": metadata.st_ctime_ns,
            "device_id": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": metadata.st_mode,
            "mtime_ns": metadata.st_mtime_ns,
            "size": metadata.st_size,
        }
        exact(observed, ssh[stat_key], f"ssh.{name}.stat")
        rows[name] = (raw, metadata)
    require(rows["ssh"][1].st_mode & 0o111, "E_SSH_EXECUTABLE")
    require(rows["ssh_keygen"][1].st_mode & 0o111, "E_SSH_KEYGEN_EXECUTABLE")
    identity = rows["identity"][1]
    exact(identity.st_uid, os.geteuid(), "ssh.identity.uid")
    exact(stat.S_IMODE(identity.st_mode), 0o600, "ssh.identity.mode")
    fingerprint = public_key_fingerprint(rows["identity_public"][0])
    exact(
        fingerprint,
        ssh["identity_public_key_fingerprint"],
        "ssh.identity_public_key_fingerprint",
    )
    try:
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
            env=SSH_ENV,
        )
    except subprocess.SubprocessError as error:
        raise LaunchError("E_SSH_PRIVATE_KEY_PROOF") from error
    require(
        completed.returncode == 0
        and not completed.stderr
        and 0 < len(completed.stdout) <= 16 * 1024,
        "E_SSH_PRIVATE_KEY_PROOF",
    )
    public = completed.stdout
    if not public.endswith(b"\n"):
        public += b"\n"
    exact(
        public_key_fingerprint(public),
        fingerprint,
        "ssh.identity_private_key_fingerprint",
    )


def ssh_prefix(
    ssh: dict[str, Any],
    local_forward: dict[str, Any] | None = None,
) -> list[str]:
    exact(ssh["ssh_path"], "/usr/bin/ssh", "ssh.ssh_path")
    exact(ssh["ssh_target"], REMOTE_CUDA_TARGET, "ssh.ssh_target")
    exact(ssh["host_key_alias"], REMOTE_CUDA_HOST_KEY_ALIAS, "ssh.host_key_alias")
    exact(ssh["ssh_port"], 22, "ssh.ssh_port")
    result = [
        ssh["ssh_path"],
        "-F",
        "/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"HostKeyAlias={ssh['host_key_alias']}",
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
        "-o",
        f"ConnectTimeout={ssh['connect_timeout_s']}",
        "-p",
        str(ssh["ssh_port"]),
        "-i",
        ssh["identity_file_path"],
    ]
    if local_forward is not None:
        exact_keys(local_forward, LOCAL_FORWARD_KEYS, "local_forward")
        result.extend([
            "-o",
            "ExitOnForwardFailure=yes",
            "-L",
            (
                f"{local_forward['local_host']}:"
                f"{local_forward['local_port']}:"
                f"{local_forward['remote_host']}:"
                f"{local_forward['remote_port']}"
            ),
        ])
    result.append(ssh["ssh_target"])
    return result


def remote_helper_argv(
    ssh: dict[str, Any],
    action: str,
    payload: dict[str, Any],
    local_forward: dict[str, Any] | None = None,
) -> list[str]:
    require(
        action in (
            "cleanup",
            "identity",
            "launch",
            "prepare",
            "process",
            "signal",
            "stat",
        ),
        "E_REMOTE_ACTION",
    )
    raw = canonical_compact(payload)
    require(len(raw) <= 256 * 1024, "E_REMOTE_PAYLOAD_SIZE")
    encoded = base64.b64encode(raw).decode("ascii")
    interpreter = base64.b64encode(canonical_compact({
        "path": ssh["remote_python_path"],
        "sha256": ssh["remote_python_sha256"],
        "stat": ssh["remote_python_stat"],
    })).decode("ascii")
    command = shlex.join([
        ssh["remote_python_path"],
        "-I",
        "-c",
        REMOTE_CUDA_HELPER,
        action,
        encoded,
        interpreter,
    ])
    return ssh_prefix(ssh, local_forward) + [command]


def parse_canonical_json_line(raw: bytes, field: str) -> dict[str, Any]:
    require(0 < len(raw) <= MAX_JSON and raw.endswith(b"\n"), f"E_JSON: {field}")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                LaunchError(f"E_JSON_NUMBER: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaunchError(f"E_JSON: {field}") from error
    exact(canonical_bytes(value), raw, f"{field}.canonical")
    require(type(value) is dict, f"E_JSON_OBJECT: {field}")
    return value


def run_remote_helper(
    runner: Any,
    ssh: dict[str, Any],
    action: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    verify_ssh(ssh)
    try:
        completed = runner.run(
            remote_helper_argv(ssh, action, payload),
            timeout=timeout_s,
            env=SSH_ENV,
        )
    except subprocess.TimeoutExpired as error:
        raise LaunchError(f"E_REMOTE_TIMEOUT: {action}") from error
    require(
        type(completed.stdout) is bytes
        and type(completed.stderr) is bytes
        and len(completed.stdout) <= MAX_ADB_OUTPUT
        and len(completed.stderr) <= MAX_ADB_OUTPUT,
        f"E_REMOTE_OUTPUT: {action}",
    )
    exact(completed.returncode, 0, f"remote.{action}.returncode")
    exact(completed.stderr, b"", f"remote.{action}.stderr")
    return parse_canonical_json_line(completed.stdout, f"remote.{action}")


def adb_prefix(android: dict[str, Any]) -> list[str]:
    exact(android["adb_port"], ADB_PORT, "adb.port")
    require(":" in android["adb_selector"], "E_ADB_SELECTOR")
    return [
        android["adb_path"],
        "-P",
        str(ADB_PORT),
        "-s",
        android["adb_selector"],
    ]


def run_checked(
    runner: Any,
    argv: list[str],
    timeout_s: float,
    field: str,
) -> bytes:
    try:
        completed = runner.run(argv, timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        raise LaunchError(f"E_TIMEOUT: {field}") from error
    require(
        type(completed.stdout) is bytes
        and type(completed.stderr) is bytes
        and len(completed.stdout) <= MAX_ADB_OUTPUT
        and len(completed.stderr) <= MAX_ADB_OUTPUT,
        f"E_OUTPUT: {field}",
    )
    exact(completed.returncode, 0, f"{field}.returncode")
    exact(completed.stderr, b"", f"{field}.stderr")
    return completed.stdout


def validate_boot_id(value: Any, field: str = "boot_id") -> str:
    value = text(value, field, 64)
    require(BOOT_RE.fullmatch(value) is not None, f"E_BOOT_ID: {field}")
    return value


def verify_android_identity(
    runner: Any,
    android: dict[str, Any],
    expected_boot_id: str,
) -> None:
    expected_boot_id = validate_boot_id(expected_boot_id)
    prefix = adb_prefix(android)
    state = run_checked(runner, prefix + ["get-state"], 15, "adb.state")
    exact(state.strip(), b"device", "adb.state")
    serial = run_checked(
        runner,
        prefix + ["shell", "getprop", "ro.serialno"],
        15,
        "adb.serial",
    )
    exact(
        serial.decode("ascii").strip(),
        android["physical_serial"],
        "adb.physical_serial",
    )
    boot = run_checked(
        runner,
        prefix + [
            "shell",
            "cat",
            "/proc/sys/kernel/random/boot_id",
        ],
        15,
        "adb.boot",
    )
    exact(boot.decode("ascii").strip(), expected_boot_id, "adb.boot_id")


def parse_android_time(value: str, seconds: int, field: str) -> int:
    match = ANDROID_TIME_RE.fullmatch(value)
    require(match is not None, f"E_ANDROID_TIME: {field}")
    base = datetime.datetime.strptime(
        f"{match.group('base')} {match.group('zone')}",
        "%Y-%m-%d %H:%M:%S %z",
    )
    fraction = match.group("fraction").ljust(9, "0")
    result = int(base.timestamp()) * 1_000_000_000 + int(fraction)
    exact(result // 1_000_000_000, seconds, f"{field}.seconds")
    return result


def parse_android_stat_line(value: str, field: str) -> dict[str, Any]:
    expected = (
        "DEV",
        "INO",
        "SIZE",
        "MODE",
        "MTIME_S",
        "MTIME",
        "CTIME_S",
        "CTIME",
    )
    parts = value.split("|")
    require(len(parts) == len(expected), f"E_ANDROID_STAT_FIELDS: {field}")
    fields = {}
    for item, key in zip(parts, expected):
        prefix = f"{key}="
        require(item.startswith(prefix), f"E_ANDROID_STAT_FIELD: {field}.{key}")
        fields[key] = item[len(prefix):]
    try:
        device = int(fields["DEV"])
        inode = int(fields["INO"])
        size = int(fields["SIZE"])
        mode = int(fields["MODE"], 16)
        mtime_s = int(fields["MTIME_S"])
        ctime_s = int(fields["CTIME_S"])
    except ValueError as error:
        raise LaunchError(f"E_ANDROID_STAT_INTEGER: {field}") from error
    return {
        "build_id": None,
        "ctime_ns": parse_android_time(fields["CTIME"], ctime_s, f"{field}.ctime"),
        "device_id": device,
        "inode": inode,
        "mode": mode,
        "mtime_ns": parse_android_time(fields["MTIME"], mtime_s, f"{field}.mtime"),
        "size": size,
    }


def remote_component(
    runner: Any,
    android: dict[str, Any],
    component: dict[str, Any],
) -> dict[str, Any]:
    path = shlex.quote(component["path"])
    stat_format = (
        "DEV=%d|INO=%i|SIZE=%s|MODE=%f|"
        "MTIME_S=%Y|MTIME=%y|CTIME_S=%Z|CTIME=%z"
    )
    emit = f"stat -c {shlex.quote(stat_format)} -- {path}"
    script = "; ".join([
        "set -eu",
        f"test -f {path}",
        f"test ! -L {path}",
        emit,
    ])
    command = "sh -c " + shlex.quote(script)
    raw = run_checked(
        runner,
        adb_prefix(android) + ["shell", command],
        30,
        f"component.{component['component_id']}",
    )
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise LaunchError("E_ANDROID_STAT_ASCII") from error
    require(len(lines) == 1, "E_ANDROID_STAT_FRAMING")
    observed = parse_android_stat_line(lines[0], "component.stat")
    exact(observed, component["stat"], f"{component['component_id']}.stat")
    return {**observed, "path": component["path"]}


def remote_cuda_component(
    runner: Any,
    ssh: dict[str, Any],
    component: dict[str, Any],
) -> dict[str, Any]:
    observed = run_remote_helper(
        runner,
        ssh,
        "stat",
        component,
        30,
    )
    observed = exact_keys(
        observed,
        {"sha256", "stat"},
        f"remote_component.{component['component_id']}",
    )
    digest(
        observed["sha256"],
        f"remote_component.{component['component_id']}.sha256",
    )
    exact(
        observed["sha256"],
        component["sha256"],
        f"remote_component.{component['component_id']}.sha256",
    )
    observed_stat = validate_stat(
        observed["stat"],
        f"remote_component.{component['component_id']}.stat",
    )
    exact(
        observed_stat,
        component["stat"],
        f"remote_component.{component['component_id']}.stat",
    )
    return {
        **observed_stat,
        "path": component["path"],
        "sha256": observed["sha256"],
    }


def remote_cuda_identity(
    runner: Any,
    ssh: dict[str, Any],
    boot_id: str,
) -> None:
    boot_id = validate_boot_id(boot_id)
    row = run_remote_helper(
        runner,
        ssh,
        "identity",
        {
            "boot_id": boot_id,
            "gpu_uuid": ssh["gpu_uuid"],
            "nvidia_smi_path": ssh["nvidia_smi_path"],
        },
        30,
    )
    exact_keys(row, {"boot_id", "gpu_uuid"}, "remote.identity")
    exact(row["boot_id"], boot_id, "remote.identity.boot_id")
    exact(row["gpu_uuid"], ssh["gpu_uuid"], "remote.identity.gpu_uuid")


def remote_component_for_path(
    plan: dict[str, Any],
    path: str,
) -> dict[str, Any]:
    matches = [
        component
        for component in plan["components"]
        if component["path"] == path
    ]
    exact(len(matches), 1, f"remote.component_path[{path}]")
    return matches[0]


def remote_verified_paths(
    plan: dict[str, Any],
    launch_token: str,
) -> dict[str, str]:
    require(
        type(launch_token) is str
        and len(launch_token) == 32
        and all(character in "0123456789abcdef" for character in launch_token),
        "E_REMOTE_LAUNCH_TOKEN",
    )
    root = REMOTE_ROOT_PREFIX + launch_token
    return {
        component["path"]: f"{root}/{index:04d}"
        for index, component in enumerate(plan["components"])
    }


def parse_start_ticks(raw: bytes, expected_pid: int) -> int:
    closing = raw.rfind(b")")
    require(
        closing > 0 and raw[closing + 1:closing + 2] == b" ",
        "E_PROCESS_STAT",
    )
    try:
        pid = int(raw[:raw.find(b" ")])
        fields = raw[closing + 2:].split()
        ticks = int(fields[19])
    except (IndexError, ValueError) as error:
        raise LaunchError("E_PROCESS_STAT") from error
    require(pid == expected_pid and ticks > 0, "E_PROCESS_STAT")
    return ticks


def decode_hex(value: str, field: str) -> bytes:
    require(
        len(value) % 2 == 0
        and all(character in "0123456789abcdef" for character in value),
        f"E_HEX: {field}",
    )
    try:
        return bytes.fromhex(value)
    except ValueError as error:
        raise LaunchError(f"E_HEX: {field}") from error


def parse_process_snapshot(
    raw: bytes,
    android: dict[str, Any],
    pid: int,
    expected_executable: str,
    expected_argv: list[str],
    expected_boot_id: str,
    expected_start_ticks: int | None = None,
) -> tuple[int, str]:
    expected_boot_id = validate_boot_id(expected_boot_id)
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise LaunchError("E_PROCESS_ASCII") from error
    require(len(lines) == 5, "E_PROCESS_RECORDS")
    values = {}
    for line in lines:
        key, separator, value = line.partition(" ")
        require(separator and key not in values, "E_PROCESS_RECORD")
        values[key] = value
    exact(set(values), {"SERIAL", "BOOT", "EXE", "CMD", "STAT"}, "process.keys")
    exact(values["SERIAL"], android["physical_serial"], "process.serial")
    exact(values["BOOT"], expected_boot_id, "process.boot")
    try:
        executable = decode_hex(values["EXE"], "process.exe").decode("ascii")
        cmdline = decode_hex(values["CMD"], "process.cmdline")
    except UnicodeDecodeError as error:
        raise LaunchError("E_PROCESS_TEXT") from error
    exact(executable, expected_executable, "process.executable")
    require(cmdline.endswith(b"\x00"), "E_PROCESS_ARGV_END")
    try:
        argv = [item.decode("ascii") for item in cmdline[:-1].split(b"\x00")]
    except UnicodeDecodeError as error:
        raise LaunchError("E_PROCESS_ARGV") from error
    exact(argv, expected_argv, "process.argv")
    ticks = parse_start_ticks(decode_hex(values["STAT"], "process.stat"), pid)
    if expected_start_ticks is not None:
        exact(ticks, expected_start_ticks, "process.start_ticks")
    return ticks, executable


def remote_process_snapshot(
    runner: Any,
    android: dict[str, Any],
    pid: int,
    expected_executable: str,
    expected_argv: list[str],
    expected_boot_id: str,
) -> tuple[int, str]:
    command = " ".join([
        "sh -c",
        shlex.quote(REMOTE_PROCESS_SCRIPT),
        "s39-probe",
        str(pid),
    ])
    raw = run_checked(
        runner,
        adb_prefix(android) + ["shell", command],
        30,
        "process.snapshot",
    )
    return parse_process_snapshot(
        raw,
        android,
        pid,
        expected_executable,
        expected_argv,
        expected_boot_id,
    )


def remote_cuda_process_snapshot(
    runner: Any,
    ssh: dict[str, Any],
    boot_id: str,
    pid: int,
    start_ticks: int,
    expected_executable: str,
    expected_argv: list[str],
    nvidia_smi_path: str | None = None,
) -> None:
    boot_id = validate_boot_id(boot_id)
    integer(pid, "remote_cuda.pid", 1)
    integer(start_ticks, "remote_cuda.start_ticks", 1)
    row = run_remote_helper(
        runner,
        ssh,
        "process",
        {
            "argv": expected_argv,
            "boot_id": boot_id,
            "executable_path": expected_executable,
            "gpu_uuid": ssh["gpu_uuid"],
            "nvidia_smi_path": (
                ssh["nvidia_smi_path"]
                if nvidia_smi_path is None
                else nvidia_smi_path
            ),
            "pid": pid,
            "start_ticks": start_ticks,
        },
        30,
    )
    exact_keys(
        row,
        {
            "argv",
            "boot_id",
            "executable_path",
            "gpu_uuid",
            "pid",
            "start_ticks",
        },
        "remote.process",
    )
    exact(row["argv"], expected_argv, "remote.process.argv")
    exact(row["boot_id"], boot_id, "remote.process.boot_id")
    exact(
        row["executable_path"],
        expected_executable,
        "remote.process.executable",
    )
    exact(row["gpu_uuid"], ssh["gpu_uuid"], "remote.process.gpu_uuid")
    exact(row["pid"], pid, "remote.process.pid")
    exact(row["start_ticks"], start_ticks, "remote.process.start_ticks")


def process_record(
    plan: dict[str, Any],
    pid: int,
    start_ticks: int,
    dependencies: list[dict[str, Any]],
    observed_ns: int,
    boot_id: str,
    remote_observed_ns: int | None = None,
    launch_token: str | None = None,
    pgid: int | None = None,
) -> dict[str, Any]:
    normalized = plan["_normalized"]
    boot_id = validate_boot_id(boot_id)
    result = {
        "boot_id": boot_id,
        "bundle_id": plan["bundle_id"],
        "endpoint": plan["endpoint"],
        "launcher_path": normalized["launcher_path"],
        "loaded_repo_component_ids": sorted(normalized["component_map"]),
        "pid": pid,
        "schema": PROCESS_SCHEMA,
        "start_ticks": start_ticks,
        "system_dependencies": sorted(
            dependencies,
            key=lambda item: item["path"],
        ),
    }
    if remote_observed_ns is None:
        result["observed_ns"] = observed_ns
    else:
        result.update({
            "controller_clock": "CONTROLLER_MONOTONIC",
            "controller_observed_ns": observed_ns,
            "launch_token": text(
                launch_token,
                "runtime.launch_token",
                32,
            ),
            "pgid": integer(pgid, "runtime.pgid", 1),
            "remote_observed_ns": integer(
                remote_observed_ns,
                "runtime.remote_observed_ns",
                1,
            ),
            "remote_clock": "RTX_CLOCK_MONOTONIC_RAW",
        })
        exact(result["pgid"], pid, "runtime.pgid")
        require(
            len(result["launch_token"]) == 32
            and all(
                character in "0123456789abcdef"
                for character in result["launch_token"]
            ),
            "E_REMOTE_LAUNCH_TOKEN",
        )
    return result


def local_process_identity(
    pid: int,
    expected_executable: str,
    expected_argv: list[str],
    expected_parent_pid: int,
) -> int:
    pid = integer(pid, "local_process.pid", 1)
    expected_parent_pid = integer(
        expected_parent_pid,
        "local_process.parent_pid",
        1,
    )
    deadline = time.monotonic() + 5
    while True:
        try:
            executable = os.readlink(f"/proc/{pid}/exe")
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError as error:
            raise LaunchError(f"E_LOCAL_PROCESS: {error}") from error
        if cmdline:
            break
        if time.monotonic() >= deadline:
            raise LaunchError("E_LOCAL_PROCESS_ARGV_EMPTY")
        time.sleep(0.01)
    exact(executable, expected_executable, "local_process.executable")
    require(cmdline.endswith(b"\x00"), "E_LOCAL_PROCESS_ARGV_END")
    try:
        argv = [item.decode("ascii") for item in cmdline[:-1].split(b"\x00")]
    except UnicodeDecodeError as error:
        raise LaunchError("E_LOCAL_PROCESS_ARGV") from error
    exact(argv, expected_argv, "local_process.argv")

    def fields() -> tuple[int, int]:
        try:
            raw = Path(f"/proc/{pid}/stat").read_bytes()
        except OSError as error:
            raise LaunchError(f"E_LOCAL_PROCESS_STAT: {error}") from error
        closing = raw.rfind(b")")
        require(
            closing > 0 and raw[closing + 1:closing + 2] == b" ",
            "E_LOCAL_PROCESS_STAT",
        )
        try:
            observed_pid = int(raw[:raw.find(b" ")])
            values = raw[closing + 2:].split()
            parent_pid = int(values[1])
            start_ticks = int(values[19])
        except (IndexError, ValueError) as error:
            raise LaunchError("E_LOCAL_PROCESS_STAT") from error
        require(
            observed_pid == pid and parent_pid > 0 and start_ticks > 0,
            "E_LOCAL_PROCESS_STAT",
        )
        return parent_pid, start_ticks

    first = fields()
    second = fields()
    exact(second, first, "local_process.stable")
    exact(first[0], expected_parent_pid, "local_process.parent_pid")
    return first[1]


def transport_process_record(
    plan: dict[str, Any],
    pid: int,
    start_ticks: int,
    argv: list[str],
    managed_launcher_pid: int,
    managed_launcher_start_ticks: int,
    observed_ns: int,
    host_boot_id: str,
    remote_boot_id: str,
) -> dict[str, Any]:
    pid = integer(pid, "transport.pid", 1)
    start_ticks = integer(start_ticks, "transport.start_ticks", 1)
    managed_launcher_pid = integer(
        managed_launcher_pid,
        "transport.managed_launcher_pid",
        1,
    )
    managed_launcher_start_ticks = integer(
        managed_launcher_start_ticks,
        "transport.managed_launcher_start_ticks",
        1,
    )
    observed_ns = integer(observed_ns, "transport.observed_ns", 1)
    host_boot_id = validate_boot_id(host_boot_id, "transport.host_boot_id")
    remote_boot_id = validate_boot_id(
        remote_boot_id,
        "transport.remote_boot_id",
    )
    require(
        type(argv) is list
        and bool(argv)
        and all(type(item) is str and bool(item) for item in argv),
        "E_TRANSPORT_ARGV",
    )
    public_plan = {
        key: value
        for key, value in plan.items()
        if key != "_normalized"
    }
    return {
        "argv": argv,
        "bundle_id": plan["bundle_id"],
        "endpoint": plan["endpoint"],
        "host_boot_id": host_boot_id,
        "managed_launcher_pid": managed_launcher_pid,
        "managed_launcher_start_ticks": managed_launcher_start_ticks,
        "observed_ns": observed_ns,
        "pid": pid,
        "plan_sha256": hashlib.sha256(
            canonical_compact(public_plan)
        ).hexdigest(),
        "remote_boot_id": remote_boot_id,
        "schema": TRANSPORT_PROCESS_SCHEMA,
        "start_ticks": start_ticks,
    }


def local_start_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError as error:
        raise LaunchError(f"E_LOCAL_STAT: {error}") from error
    return parse_start_ticks(raw, pid)


def remote_wrapper(argv: list[str], environment: dict[str, str]) -> str:
    assignments = " ".join(
        f"{key}={shlex.quote(value)}"
        for key, value in sorted(environment.items())
    )
    command = shlex.join(argv)
    return (
        "printf 'S39PID %s\\n' \"$$\"; "
        "exec /system/bin/env -i "
        f"{assignments} {command}"
    )


def read_pid_marker(process: Any, deadline: float) -> tuple[int, bytes]:
    require(process.stdout is not None, "E_LAUNCH_STDOUT")
    try:
        descriptor = process.stdout.fileno()
    except (AttributeError, OSError, ValueError) as error:
        raise LaunchError("E_LAUNCH_STDOUT") from error
    buffer = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(descriptor, selectors.EVENT_READ)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LaunchError("E_LAUNCH_TIMEOUT")
            events = selector.select(min(remaining, 0.1))
            if events:
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    raise LaunchError("E_LAUNCH_EARLY_EXIT")
                buffer.extend(chunk)
                require(len(buffer) <= MAX_JSON, "E_PID_MARKER_SIZE")
                if b"\n" in buffer:
                    line, remainder = bytes(buffer).split(b"\n", 1)
                    match = re.fullmatch(rb"S39PID ([1-9][0-9]*)", line)
                    require(match is not None, "E_PID_MARKER")
                    return int(match.group(1)), remainder
            elif process.poll() is not None:
                raise LaunchError("E_LAUNCH_EARLY_EXIT")
    finally:
        selector.close()


def read_remote_cuda_marker(
    process: Any,
    deadline: float,
    expected_boot_id: str,
    expected_gpu_uuid: str,
    expected_launch_token: str | None = None,
    expected_component_sha256: dict[str, str] | None = None,
    stop_requested: Any = None,
) -> tuple[int, int, int, bytes]:
    expected_boot_id = validate_boot_id(expected_boot_id)
    require(GPU_UUID_RE.fullmatch(expected_gpu_uuid) is not None, "E_GPU_UUID")
    require(process.stdout is not None, "E_LAUNCH_STDOUT")
    try:
        descriptor = process.stdout.fileno()
    except (AttributeError, OSError, ValueError) as error:
        raise LaunchError("E_LAUNCH_STDOUT") from error
    buffer = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(descriptor, selectors.EVENT_READ)
    try:
        while True:
            if stop_requested is not None and stop_requested():
                raise LaunchError("E_LAUNCH_SIGNAL")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LaunchError("E_LAUNCH_TIMEOUT")
            events = selector.select(min(remaining, 0.1))
            if events:
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    raise LaunchError("E_LAUNCH_EARLY_EXIT")
                buffer.extend(chunk)
                require(len(buffer) <= MAX_JSON, "E_PID_MARKER_SIZE")
                if b"\n" in buffer:
                    line, remainder = bytes(buffer).split(b"\n", 1)
                    require(line.startswith(b"S39CUDA "), "E_PID_MARKER")
                    payload = line[len(b"S39CUDA "):]
                    try:
                        value = json.loads(
                            payload.decode("ascii"),
                            object_pairs_hook=strict_object,
                            parse_constant=lambda item: (_ for _ in ()).throw(
                                LaunchError(f"E_JSON_NUMBER: {item}")
                            ),
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise LaunchError("E_PID_MARKER") from error
                    exact(canonical_compact(value), payload, "marker.canonical")
                    expected_keys = {
                        "boot_id",
                        "gpu_uuid",
                        "pid",
                        "start_ticks",
                    }
                    if expected_launch_token is not None:
                        expected_keys |= {
                            "component_sha256",
                            "launch_token",
                            "pgid",
                            "remote_observed_ns",
                        }
                    exact_keys(value, expected_keys, "marker")
                    exact(value["boot_id"], expected_boot_id, "marker.boot_id")
                    exact(
                        value["gpu_uuid"],
                        expected_gpu_uuid,
                        "marker.gpu_uuid",
                    )
                    pid = integer(value["pid"], "marker.pid", 1)
                    ticks = integer(
                        value["start_ticks"],
                        "marker.start_ticks",
                        1,
                    )
                    if expected_launch_token is not None:
                        exact(
                            value["launch_token"],
                            expected_launch_token,
                            "marker.launch_token",
                        )
                        exact(
                            integer(value["pgid"], "marker.pgid", 1),
                            pid,
                            "marker.pgid",
                        )
                        exact(
                            value["component_sha256"],
                            expected_component_sha256,
                            "marker.component_sha256",
                        )
                        remote_observed_ns = integer(
                            value["remote_observed_ns"],
                            "marker.remote_observed_ns",
                            1,
                        )
                    else:
                        remote_observed_ns = 0
                    return pid, ticks, remote_observed_ns, remainder
            elif process.poll() is not None:
                raise LaunchError("E_LAUNCH_EARLY_EXIT")
    finally:
        selector.close()


def forward_process_output(
    process: Any,
    initial: bytes,
    stop_requested: Any,
    on_stop: Any,
    shutdown_timeout_s: float,
    sink: Any,
) -> int:
    require(process.stdout is not None, "E_LAUNCH_STDOUT")
    try:
        descriptor = process.stdout.fileno()
    except (AttributeError, OSError, ValueError) as error:
        raise LaunchError("E_LAUNCH_STDOUT") from error
    selector = selectors.DefaultSelector()
    selector.register(descriptor, selectors.EVENT_READ)
    scan_tail = b""
    reserved_prefixes = (
        PROCESS_PREFIX,
        TRANSPORT_PROCESS_PREFIX,
        REMOTE_CLEANUP_PREFIX,
    )

    def emit(chunk: bytes) -> None:
        nonlocal scan_tail
        if not chunk:
            return
        combined = scan_tail + chunk
        require(
            not any(prefix in combined for prefix in reserved_prefixes),
            "E_DUPLICATE_PROCESS_RECORD",
        )
        keep = max(len(prefix) for prefix in reserved_prefixes) - 1
        scan_tail = combined[-keep:] if keep else b""
        sink.write(chunk)
        sink.flush()

    try:
        emit(initial)
        while True:
            requested = stop_requested()
            require(type(requested) is int and requested >= 0, "E_SIGNAL_STATE")
            if requested:
                on_stop(requested)
                try:
                    process.wait(timeout=shutdown_timeout_s)
                except subprocess.TimeoutExpired as error:
                    raise LaunchError("E_CLEANUP_TIMEOUT") from error
                return 128 + requested
            events = selector.select(0.1)
            if events:
                chunk = os.read(descriptor, 64 * 1024)
                if chunk:
                    emit(chunk)
                    continue
            returncode = process.poll()
            if returncode is not None:
                require(type(returncode) is int, "E_PROCESS_RETURN")
                return returncode
    finally:
        selector.close()


def cleanup_remote(
    runner: Any,
    android: dict[str, Any],
    pid: int,
    start_ticks: int,
    expected_boot_id: str,
) -> None:
    expected_boot_id = validate_boot_id(expected_boot_id)
    prefix = adb_prefix(android)
    boot = run_checked(
        runner,
        prefix + ["shell", "cat /proc/sys/kernel/random/boot_id"],
        15,
        "cleanup.boot",
    )
    exact(boot.decode("ascii").strip(), expected_boot_id, "cleanup.boot_id")
    snapshot = runner.run(
        prefix + ["shell", f"cat /proc/{pid}/stat"],
        timeout=15,
    )
    if snapshot.returncode != 0:
        return
    require(
        type(snapshot.stdout) is bytes and type(snapshot.stderr) is bytes,
        "E_CLEANUP_STAT",
    )
    exact(snapshot.stderr, b"", "cleanup.stat.stderr")
    exact(parse_start_ticks(snapshot.stdout, pid), start_ticks, "cleanup.start_ticks")
    result = runner.run(
        prefix + ["shell", f"kill -TERM {pid}"],
        timeout=15,
    )
    exact(result.returncode, 0, "cleanup.term")
    deadline = time.monotonic() + android["shutdown_timeout_ms"] / 1000
    while time.monotonic() < deadline:
        alive = runner.run(
            prefix + ["shell", f"kill -0 {pid}"],
            timeout=15,
        )
        if alive.returncode != 0:
            return
        time.sleep(0.05)
    result = runner.run(
        prefix + ["shell", f"kill -KILL {pid}"],
        timeout=15,
    )
    exact(result.returncode, 0, "cleanup.kill")
    alive = runner.run(
        prefix + ["shell", f"kill -0 {pid}"],
        timeout=15,
    )
    require(alive.returncode != 0, "E_CLEANUP_FAILED")


def remote_cuda_signal(
    runner: Any,
    ssh: dict[str, Any],
    boot_id: str,
    pid: int,
    start_ticks: int,
    selected: str,
) -> bool:
    require(selected in ("0", "TERM", "KILL"), "E_REMOTE_SIGNAL")
    row = run_remote_helper(
        runner,
        ssh,
        "signal",
        {
            "boot_id": validate_boot_id(boot_id),
            "pid": integer(pid, "remote_cuda.pid", 1),
            "signal": selected,
            "start_ticks": integer(
                start_ticks,
                "remote_cuda.start_ticks",
                1,
            ),
        },
        30,
    )
    exact_keys(row, {"alive", "boot_id"}, "remote.signal")
    require(type(row["alive"]) is bool, "E_REMOTE_SIGNAL_STATE")
    exact(row["boot_id"], boot_id, "remote.signal.boot_id")
    return row["alive"]


def cleanup_remote_cuda(
    runner: Any,
    ssh: dict[str, Any],
    nvidia_smi_component: dict[str, Any],
    boot_id: str,
    launch_token: str,
    pid: int,
    start_ticks: int,
) -> dict[str, Any]:
    require(
        type(launch_token) is str
        and len(launch_token) == 32
        and all(character in "0123456789abcdef" for character in launch_token),
        "E_REMOTE_LAUNCH_TOKEN",
    )
    require(
        type(pid) is int
        and type(start_ticks) is int
        and pid >= 0
        and start_ticks >= 0
        and bool(pid) == bool(start_ticks),
        "E_REMOTE_PROCESS_IDENTITY",
    )
    row = run_remote_helper(
        runner,
        ssh,
        "cleanup",
        {
            "boot_id": validate_boot_id(boot_id),
            "gpu_uuid": ssh["gpu_uuid"],
            "launch_token": launch_token,
            "nvidia_smi": nvidia_smi_component,
            "pgid": pid,
            "pid": pid,
            "shutdown_timeout_ms": ssh["shutdown_timeout_ms"],
            "start_ticks": start_ticks,
        },
        30 + 2 * ssh["shutdown_timeout_ms"] / 1000,
    )
    exact_keys(
        row,
        {
            "absent",
            "boot_id",
            "clock",
            "gpu_uuid",
            "launch_token",
            "matching_nvml_pids",
            "matching_process_groups",
            "matching_processes",
            "observed_ns",
            "pgid",
            "pid",
            "schema",
            "start_ticks",
        },
        "remote.cleanup",
    )
    exact(row["schema"], REMOTE_CLEANUP_SCHEMA, "remote.cleanup.schema")
    exact(row["boot_id"], boot_id, "remote.cleanup.boot_id")
    exact(row["clock"], "RTX_CLOCK_MONOTONIC_RAW", "remote.cleanup.clock")
    exact(row["gpu_uuid"], ssh["gpu_uuid"], "remote.cleanup.gpu_uuid")
    exact(row["launch_token"], launch_token, "remote.cleanup.launch_token")
    exact(row["matching_processes"], [], "remote.cleanup.processes")
    exact(
        row["matching_process_groups"],
        [],
        "remote.cleanup.process_groups",
    )
    exact(row["matching_nvml_pids"], [], "remote.cleanup.nvml")
    integer(row["observed_ns"], "remote.cleanup.observed_ns", 1)
    exact(row["pid"], pid, "remote.cleanup.pid")
    exact(row["pgid"], pid, "remote.cleanup.pgid")
    exact(row["start_ticks"], start_ticks, "remote.cleanup.start_ticks")
    absent = row["absent"]
    require(type(absent) is list, "E_REMOTE_CLEANUP_ABSENT")
    for index, item in enumerate(absent):
        item = exact_keys(
            item,
            {"pid", "start_ticks"},
            f"remote.cleanup.absent[{index}]",
        )
        integer(item["pid"], f"remote.cleanup.absent[{index}].pid", 1)
        integer(
            item["start_ticks"],
            f"remote.cleanup.absent[{index}].start_ticks",
            1,
        )
    if pid:
        require(
            {"pid": pid, "start_ticks": start_ticks} in absent,
            "E_REMOTE_CLEANUP_PRODUCER",
        )
    return row


def launch_android(
    plan: dict[str, Any],
    runner: Any,
    boot_id: str,
) -> int:
    boot_id = validate_boot_id(boot_id)
    android = plan["android"]
    normalized = plan["_normalized"]
    verify_adb(android)
    verify_android_identity(runner, android, boot_id)
    dependencies = [
        remote_component(runner, android, component)
        for component in plan["components"]
    ]
    launch_argv = (
        adb_prefix(android)
        + [
            "shell",
            "sh -c "
            + shlex.quote(
                remote_wrapper(
                    normalized["argv"],
                    normalized["environment"],
                )
            ),
        ]
    )
    process = runner.popen(launch_argv)
    pid = 0
    ticks = 0
    signalled = 0
    previous_handlers = {}

    def on_signal(number: int, _frame: Any) -> None:
        nonlocal signalled
        signalled = number

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[number] = signal.signal(number, on_signal)
        deadline = time.monotonic() + android["startup_timeout_ms"] / 1000
        pid, initial = read_pid_marker(process, deadline)
        ticks, _ = remote_process_snapshot(
            runner,
            android,
            pid,
            normalized["launcher_path"],
            normalized["argv"],
            boot_id,
        )
        record = process_record(
            plan,
            pid,
            ticks,
            dependencies,
            time.monotonic_ns(),
            boot_id,
        )
        sys.stdout.buffer.write(PROCESS_PREFIX + canonical_bytes(record))
        sys.stdout.buffer.flush()
        returncode = forward_process_output(
            process,
            initial,
            lambda: signalled,
            lambda _number: cleanup_remote(
                runner,
                android,
                pid,
                ticks,
                boot_id,
            ),
            android["shutdown_timeout_ms"] / 1000,
            sys.stdout.buffer,
        )
        cleanup_remote(runner, android, pid, ticks, boot_id)
        return returncode
    except BaseException:
        if pid and ticks:
            cleanup_remote(runner, android, pid, ticks, boot_id)
        elif process.poll() is None:
            process.terminate()
        raise
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


def launch_remote_cuda(
    plan: dict[str, Any],
    runner: Any,
    boot_id: str,
) -> int:
    boot_id = validate_boot_id(boot_id)
    ssh = plan["ssh"]
    normalized = plan["_normalized"]
    verify_ssh(ssh)
    remote_cuda_identity(runner, ssh, boot_id)
    dependencies = [
        remote_cuda_component(runner, ssh, component)
        for component in plan["components"]
    ]
    launch_token = os.urandom(16).hex()
    component_sha256 = {
        component["component_id"]: component["sha256"]
        for component in plan["components"]
    }
    verified_paths = remote_verified_paths(plan, launch_token)
    effective_argv = [
        verified_paths.get(item, item)
        for item in normalized["argv"]
    ]
    effective_launcher = verified_paths[normalized["launcher_path"]]
    nvidia_smi_component = remote_component_for_path(
        plan,
        ssh["nvidia_smi_path"],
    )
    launch_payload = {
        "argv": normalized["argv"],
        "boot_id": boot_id,
        "components": plan["components"],
        "cwd": normalized["cwd"],
        "environment": normalized["environment"],
        "gpu_uuid": ssh["gpu_uuid"],
        "launch_token": launch_token,
        "nvidia_smi_path": ssh["nvidia_smi_path"],
    }
    process = None
    prepared = False
    pid = 0
    ticks = 0
    signalled = 0
    remote_cleaned = False
    previous_handlers = {}

    def on_signal(number: int, _frame: Any) -> None:
        nonlocal signalled
        signalled = number

    def cleanup_once() -> None:
        nonlocal remote_cleaned
        if remote_cleaned or not prepared:
            return
        cleanup_record = cleanup_remote_cuda(
            runner,
            ssh,
            nvidia_smi_component,
            boot_id,
            launch_token,
            pid,
            ticks,
        )
        sys.stdout.buffer.write(
            REMOTE_CLEANUP_PREFIX + canonical_bytes(cleanup_record)
        )
        sys.stdout.buffer.flush()
        remote_cleaned = True

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[number] = signal.signal(number, on_signal)
        prepared = True
        prepare = run_remote_helper(
            runner,
            ssh,
            "prepare",
            {
                "boot_id": boot_id,
                "launch_token": launch_token,
            },
            30,
        )
        exact_keys(
            prepare,
            {"boot_id", "launch_token"},
            "remote.prepare",
        )
        exact(prepare["boot_id"], boot_id, "remote.prepare.boot_id")
        exact(
            prepare["launch_token"],
            launch_token,
            "remote.prepare.launch_token",
        )
        require(not signalled, "E_LAUNCH_SIGNAL")
        verify_ssh(ssh)
        launch_argv = remote_helper_argv(
            ssh,
            "launch",
            launch_payload,
            plan["route"]["local_forward"],
        )
        process = runner.popen(
            launch_argv,
            env=SSH_ENV,
        )
        transport_pid = integer(
            getattr(process, "pid", None),
            "transport.pid",
            1,
        )
        managed_launcher_pid = os.getpid()
        transport_ticks = local_process_identity(
            transport_pid,
            ssh["ssh_path"],
            launch_argv,
            managed_launcher_pid,
        )
        managed_launcher_ticks = local_start_ticks(managed_launcher_pid)
        try:
            host_boot_id = Path(
                "/proc/sys/kernel/random/boot_id"
            ).read_text(encoding="ascii").strip()
        except OSError as error:
            raise LaunchError(f"E_LOCAL_BOOT: {error}") from error
        transport_record = transport_process_record(
            plan,
            transport_pid,
            transport_ticks,
            launch_argv,
            managed_launcher_pid,
            managed_launcher_ticks,
            time.monotonic_ns(),
            host_boot_id,
            boot_id,
        )
        sys.stdout.buffer.write(
            TRANSPORT_PROCESS_PREFIX + canonical_bytes(transport_record)
        )
        sys.stdout.buffer.flush()
        deadline = time.monotonic() + ssh["startup_timeout_ms"] / 1000
        pid, ticks, remote_observed_ns, initial = read_remote_cuda_marker(
            process,
            deadline,
            boot_id,
            ssh["gpu_uuid"],
            launch_token,
            component_sha256,
            lambda: signalled,
        )
        remote_cuda_process_snapshot(
            runner,
            ssh,
            boot_id,
            pid,
            ticks,
            effective_launcher,
            effective_argv,
            verified_paths[ssh["nvidia_smi_path"]],
        )
        record = process_record(
            plan,
            pid,
            ticks,
            dependencies,
            time.monotonic_ns(),
            boot_id,
            remote_observed_ns,
            launch_token,
            pid,
        )
        sys.stdout.buffer.write(PROCESS_PREFIX + canonical_bytes(record))
        sys.stdout.buffer.flush()
        returncode = forward_process_output(
            process,
            initial,
            lambda: signalled,
            lambda _number: cleanup_once(),
            ssh["shutdown_timeout_ms"] / 1000,
            sys.stdout.buffer,
        )
        cleanup_once()
        return returncode
    except BaseException:
        cleanup_once()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=ssh["shutdown_timeout_ms"] / 1000)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=ssh["shutdown_timeout_ms"] / 1000)
        raise
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


def launch_local(plan: dict[str, Any], boot_id: str) -> int:
    boot_id = validate_boot_id(boot_id)
    actual_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    exact(actual_boot_id, boot_id, "local.boot_id")
    normalized = plan["_normalized"]
    dependencies = [
        local_dependency(component)
        for component in plan["components"]
    ]
    pid = os.getpid()
    record = process_record(
        plan,
        pid,
        local_start_ticks(pid),
        dependencies,
        time.monotonic_ns(),
        boot_id,
    )
    sys.stdout.buffer.write(PROCESS_PREFIX + canonical_bytes(record))
    sys.stdout.buffer.flush()
    environment = {
        key: value
        for key, value in normalized["environment"].items()
    }
    os.chdir(normalized["cwd"])
    os.execve(normalized["argv"][0], normalized["argv"], environment)
    raise LaunchError("E_EXEC_RETURNED")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--boot-id", required=True)
    args = parser.parse_args(argv)
    try:
        plan = parse_plan_json(args.plan_json, args.plan_sha256)
        if plan["mode"] == "android":
            return launch_android(plan, SubprocessRunner(), args.boot_id)
        if plan["mode"] == "remote_cuda":
            return launch_remote_cuda(
                plan,
                SubprocessRunner(),
                args.boot_id,
            )
        return launch_local(plan, args.boot_id)
    except (
        LaunchError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"MANAGED_RUNTIME_LAUNCH_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
