#!/usr/bin/env python3
"""Materialize and validate the V2.6 runtime inventory."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import types
from typing import Any, Protocol


SCHEMA = "s39-cp0-r1-v26-runtime-inventory-v1"
SPEC_SCHEMA = "s39-cp0-r1-v26-runtime-inventory-spec-v1"
PHASES = {"A_ONLY", "B_ONLY", "PAIR"}
CAPTURE_KINDS = {
    "artifact_root",
    "cuda_monolithic",
    "fast_fresh_readiness",
    "joint_phone_cuda",
}
CAPTURE_BUNDLES = {
    "artifact_root": "cuda_monolithic",
    "cuda_monolithic": "cuda_monolithic",
    "fast_fresh_readiness": "cuda_route",
    "joint_phone_cuda": "cuda_route",
}
COMPONENT_ROLES = {
    "backend_library",
    "capture_entrypoint",
    "executable",
    "shared_library",
}
TRANSPORTS = {"android", "local"}
BODY_KEYS = {
    "bundles",
    "capture_entrypoints",
    "components",
    "managed_processes",
    "phase",
    "phase_id",
    "spec_sha256",
}
STAT_KEYS = {
    "build_id",
    "ctime_ns",
    "device_id",
    "inode",
    "mode",
    "mtime_ns",
    "size",
}
MAX_SPEC_BYTES = 16 * 1024 * 1024
MAX_COMPONENT_BYTES = 1 << 40
ADB_PATH = "/usr/lib/android-sdk/platform-tools/adb"
ADB_PORT = 5038


class InventoryError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InventoryError(message)


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}",
    )


def exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}")
    require(set(value) == expected, f"E_KEYS: {field}")
    return value


def text(value: Any, field: str, maximum: int = 4096) -> str:
    require(
        type(value) is str
        and 0 < len(value) <= maximum
        and value.isascii()
        and "\x00" not in value
        and "\n" not in value,
        f"E_TEXT: {field}",
    )
    return value


def absolute(value: Any, field: str) -> str:
    value = text(value, field)
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, f"E_PATH: {field}")
    return value


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(
        type(value) is int and minimum <= value <= (1 << 63) - 1,
        f"E_INTEGER: {field}",
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


def identifier(value: Any, field: str) -> str:
    value = text(value, field, 128)
    require(
        all(character.isalnum() or character in "._-" for character in value),
        f"E_IDENTIFIER: {field}",
    )
    return value


def filename(value: Any, field: str) -> str:
    value = text(value, field, 255)
    require(
        value not in {".", ".."}
        and "/" not in value
        and "\\" not in value,
        f"E_FILENAME: {field}",
    )
    return value


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise InventoryError(f"E_JSON_NUMBER: {value}")


def reject_float(value: str) -> None:
    raise InventoryError(f"E_JSON_FLOAT: {value}")


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as error:
        raise InventoryError("E_CANONICAL") from error


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_canonical(raw: bytes, field: str) -> dict[str, Any]:
    require(0 < len(raw) <= MAX_SPEC_BYTES, f"E_JSON_SIZE: {field}")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
            parse_float=reject_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InventoryError(f"E_JSON: {field}") from error
    require(type(value) is dict, f"E_JSON_TYPE: {field}")
    exact(canonical_bytes(value), raw, f"{field}.canonical")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stat_record(value: os.stat_result) -> dict[str, Any]:
    return {
        "build_id": None,
        "ctime_ns": value.st_ctime_ns,
        "device_id": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "mtime_ns": value.st_mtime_ns,
        "size": value.st_size,
    }


def _secure_directory(path: Path) -> None:
    require(path.is_absolute(), f"E_LOCAL_ROOT: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except OSError as error:
        raise InventoryError(f"E_LOCAL_ANCESTOR: {path}") from error
    finally:
        os.close(descriptor)


def _open_local_file(path: Path) -> int:
    require(path.is_absolute(), f"E_LOCAL_PATH: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.anchor, directory_flags)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        result = os.open(path.parts[-1], flags, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    return result


def secure_local_pin(path: Path, *, executable: bool) -> dict[str, Any]:
    require(path.is_absolute(), f"E_LOCAL_PATH: {path}")
    try:
        descriptor = _open_local_file(path)
    except OSError as error:
        raise InventoryError(f"E_LOCAL_READ: {path}") from error
    before = os.fstat(descriptor)
    require(stat.S_ISREG(before.st_mode), f"E_LOCAL_REGULAR: {path}")
    if executable:
        require(before.st_mode & 0o111 != 0, f"E_LOCAL_EXECUTABLE: {path}")
    try:
        hasher = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            hasher.update(block)
            total += len(block)
            require(total <= MAX_COMPONENT_BYTES, f"E_COMPONENT_SIZE: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    exact(_stat_identity(after), _stat_identity(before), f"local.fd_mutation.{path}")
    try:
        verify_fd = _open_local_file(path)
    except OSError as error:
        raise InventoryError(f"E_LOCAL_REOPEN: {path}") from error
    try:
        verify_stat = os.fstat(verify_fd)
    finally:
        os.close(verify_fd)
    exact(_stat_identity(verify_stat), _stat_identity(before), f"local.path_mutation.{path}")
    exact(total, before.st_size, f"local.bytes.{path}")
    return {
        "bytes": total,
        "path": str(path),
        "sha256": hasher.hexdigest(),
        "stat": _stat_record(before),
    }


class DirectoryObserver(Protocol):
    def observe(
        self,
        *,
        endpoint: str,
        root: str,
        files: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ...

    def pin_launcher(self, path: str) -> dict[str, Any]:
        ...


class LocalDirectoryObserver:
    def observe(
        self,
        *,
        endpoint: str,
        root: str,
        files: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del endpoint
        root_path = Path(root)
        _secure_directory(root_path)
        try:
            actual = sorted(item.name for item in os.scandir(root_path))
        except OSError as error:
            raise InventoryError(f"E_LOCAL_LIST: {root}") from error
        expected = sorted(item["filename"] for item in files)
        exact(actual, expected, f"closure.local.{root}")
        return [
            secure_local_pin(
                root_path / item["filename"],
                executable=item["role"] in {"capture_entrypoint", "executable"},
            )
            for item in files
        ]

    def pin_launcher(self, path: str) -> dict[str, Any]:
        return secure_local_pin(Path(path), executable=True)


class CommandRunner(Protocol):
    def run(self, argv: list[str], timeout: int) -> bytes:
        ...


class SubprocessRunner:
    def run(self, argv: list[str], timeout: int) -> bytes:
        result = subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if result.returncode != 0:
            message = result.stderr.decode("ascii", errors="replace").strip()
            raise InventoryError(
                f"E_COMMAND: rc={result.returncode}: {argv!r}: {message}"
            )
        return result.stdout


def _adb(serial: str, *arguments: str) -> list[str]:
    return [ADB_PATH, "-P", str(ADB_PORT), "-s", serial, *arguments]


def _remote_ancestors(root: str) -> list[str]:
    path = Path(root)
    require(path.is_absolute(), f"E_REMOTE_ROOT: {root}")
    current = Path(path.anchor)
    result = []
    for part in path.parts[1:]:
        current /= part
        result.append(str(current))
    return result


def _remote_stat_command(serial: str, path: str) -> list[str]:
    quoted = shlex.quote(path)
    format_string = (
        "DEV=%d|INO=%i|SIZE=%s|MODE=%f|MTIME=%Y|CTIME=%Z"
    )
    script = "; ".join(
        (
            "set -eu",
            f"test -f {quoted}",
            f"test ! -L {quoted}",
            f"stat -c {shlex.quote(format_string)} -- {quoted}",
        )
    )
    return _adb(serial, "shell", "sh -c " + shlex.quote(script))


def _parse_remote_stat(raw: bytes, field: str) -> dict[str, Any]:
    try:
        parts = raw.decode("ascii").strip().split("|")
        values = dict(part.split("=", 1) for part in parts)
        exact(
            set(values),
            {"CTIME", "DEV", "INO", "MODE", "MTIME", "SIZE"},
            f"{field}.keys",
        )
        result = {
            "build_id": None,
            "ctime_ns": int(values["CTIME"]) * 1_000_000_000,
            "device_id": int(values["DEV"]),
            "inode": int(values["INO"]),
            "mode": int(values["MODE"], 16),
            "mtime_ns": int(values["MTIME"]) * 1_000_000_000,
            "size": int(values["SIZE"]),
        }
    except (UnicodeDecodeError, ValueError) as error:
        raise InventoryError(f"E_REMOTE_STAT: {field}") from error
    require(
        result["inode"] > 0
        and result["size"] > 0
        and stat.S_ISREG(result["mode"]),
        f"E_REMOTE_REGULAR: {field}",
    )
    return result


class AdbDirectoryObserver:
    def __init__(self, serials: dict[str, str], runner: CommandRunner | None = None):
        self.serials = dict(serials)
        self.runner = runner or SubprocessRunner()

    def _serial(self, endpoint: str) -> str:
        require(endpoint in self.serials, f"E_ANDROID_ENDPOINT: {endpoint}")
        return text(self.serials[endpoint], f"serial.{endpoint}", 255)

    def _pin(self, serial: str, path: str, executable: bool) -> dict[str, Any]:
        before = _parse_remote_stat(
            self.runner.run(_remote_stat_command(serial, path), 30),
            f"{serial}:{path}.before",
        )
        command = _adb(serial, "shell", "sha256sum -- " + shlex.quote(path))
        first = self.runner.run(command, 600)
        second = self.runner.run(command, 600)
        exact(second, first, f"remote.digest_mutation.{serial}:{path}")
        after = _parse_remote_stat(
            self.runner.run(_remote_stat_command(serial, path), 30),
            f"{serial}:{path}.after",
        )
        exact(after, before, f"remote.stat_mutation.{serial}:{path}")
        try:
            fields = first.decode("ascii").strip().split()
        except UnicodeDecodeError as error:
            raise InventoryError(f"E_REMOTE_DIGEST: {path}") from error
        require(
            len(fields) == 2
            and len(fields[0]) == 64
            and all(character in "0123456789abcdef" for character in fields[0]),
            f"E_REMOTE_DIGEST: {path}",
        )
        if executable:
            require(before["mode"] & 0o111 != 0, f"E_REMOTE_EXECUTABLE: {path}")
        return {
            "bytes": before["size"],
            "path": path,
            "sha256": fields[0],
            "stat": before,
        }

    def observe(
        self,
        *,
        endpoint: str,
        root: str,
        files: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        serial = self._serial(endpoint)
        checks = []
        for ancestor in _remote_ancestors(root):
            quoted = shlex.quote(ancestor)
            checks.extend((f"test -d {quoted}", f"test ! -L {quoted}"))
        checks.append(f"LC_ALL=C ls -1A -- {shlex.quote(root)}")
        raw = self.runner.run(
            _adb(
                serial,
                "shell",
                "sh -c " + shlex.quote("; ".join(("set -eu", *checks))),
            ),
            30,
        )
        try:
            actual = raw.decode("ascii").splitlines()
        except UnicodeDecodeError as error:
            raise InventoryError(f"E_REMOTE_LIST: {endpoint}:{root}") from error
        require(
            actual == sorted(set(actual))
            and all(
                item and "/" not in item and item not in {".", ".."}
                for item in actual
            ),
            f"E_REMOTE_NAMES: {endpoint}:{root}",
        )
        exact(
            actual,
            sorted(item["filename"] for item in files),
            f"closure.android.{endpoint}.{root}",
        )
        return [
            self._pin(
                serial,
                f"{root}/{item['filename']}",
                item["role"] in {"capture_entrypoint", "executable"},
            )
            for item in files
        ]

    def pin_launcher(self, path: str) -> dict[str, Any]:
        raise InventoryError(f"E_ANDROID_LAUNCHER_LOCALITY: {path}")


def _load_managed_gate() -> types.ModuleType:
    path = Path(__file__).resolve().with_name("managed_plan_gate_v1.py")
    raw = path.read_bytes()
    module = types.ModuleType("s39_v26_managed_plan_gate_v1")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def _validate_component_spec(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(
        value,
        {"bytes", "component_id", "filename", "role", "sha256"},
        field,
    )
    integer(value["bytes"], f"{field}.bytes", 1)
    identifier(value["component_id"], f"{field}.component_id")
    filename(value["filename"], f"{field}.filename")
    require(value["role"] in COMPONENT_ROLES, f"E_ROLE: {field}")
    digest(value["sha256"], f"{field}.sha256")
    return value


def _validate_spec(value: Any) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "bundles",
            "capture_entrypoints",
            "managed_processes",
            "phase",
            "phase_id",
            "schema",
        },
        "spec",
    )
    exact(value["schema"], SPEC_SCHEMA, "spec.schema")
    require(value["phase"] in PHASES, "E_PHASE")
    identifier(value["phase_id"], "spec.phase_id")

    bundles = value["bundles"]
    require(type(bundles) is list and bool(bundles), "E_BUNDLES")
    bundle_map = {}
    component_ids = set()
    component_filenames = {}
    for index, bundle in enumerate(bundles):
        field = f"spec.bundles[{index}]"
        bundle = exact_keys(
            bundle,
            {
                "bundle_id",
                "endpoint",
                "files",
                "launcher_component_id",
                "managed_plan_required",
                "process_role",
                "root",
                "transport",
            },
            field,
        )
        bundle_id = identifier(bundle["bundle_id"], f"{field}.bundle_id")
        require(bundle_id not in bundle_map, f"E_BUNDLE_REUSE: {bundle_id}")
        identifier(bundle["endpoint"], f"{field}.endpoint")
        identifier(bundle["process_role"], f"{field}.process_role")
        absolute(bundle["root"], f"{field}.root")
        require(bundle["transport"] in TRANSPORTS, f"E_TRANSPORT: {field}")
        require(
            type(bundle["managed_plan_required"]) is bool,
            f"E_BOOL: {field}.managed_plan_required",
        )
        files = bundle["files"]
        require(type(files) is list and bool(files), f"E_FILES: {bundle_id}")
        filenames = set()
        ids = []
        for file_index, component in enumerate(files):
            component = _validate_component_spec(
                component,
                f"{field}.files[{file_index}]",
            )
            component_id = component["component_id"]
            require(
                component_id not in component_ids,
                f"E_COMPONENT_REUSE: {component_id}",
            )
            require(
                component["filename"] not in filenames,
                f"E_FILENAME_REUSE: {bundle_id}.{component['filename']}",
            )
            component_ids.add(component_id)
            filenames.add(component["filename"])
            ids.append(component_id)
            component_filenames[component_id] = (
                bundle_id,
                component["role"],
            )
        exact(ids, sorted(ids), f"{field}.files.order")
        launcher = identifier(
            bundle["launcher_component_id"],
            f"{field}.launcher_component_id",
        )
        require(launcher in ids, f"E_LAUNCHER: {bundle_id}")
        require(
            component_filenames[launcher][1] == "executable",
            f"E_LAUNCHER_ROLE: {bundle_id}",
        )
        bundle_map[bundle_id] = bundle
    exact(
        [item["bundle_id"] for item in bundles],
        sorted(bundle_map),
        "spec.bundles.order",
    )

    captures = value["capture_entrypoints"]
    require(type(captures) is list and len(captures) == 4, "E_CAPTURES")
    capture_map = {}
    for index, capture in enumerate(captures):
        field = f"spec.capture_entrypoints[{index}]"
        capture = exact_keys(
            capture,
            {"bundle_id", "component_id", "kind"},
            field,
        )
        kind = text(capture["kind"], f"{field}.kind", 64)
        require(kind in CAPTURE_KINDS and kind not in capture_map, "E_CAPTURE_KIND")
        bundle_id = identifier(capture["bundle_id"], f"{field}.bundle_id")
        component_id = identifier(capture["component_id"], f"{field}.component_id")
        exact(bundle_id, CAPTURE_BUNDLES[kind], f"{field}.bundle")
        require(
            component_filenames.get(component_id)
            == (bundle_id, "capture_entrypoint"),
            f"E_CAPTURE_COMPONENT: {kind}",
        )
        capture_map[kind] = capture
    exact(
        [item["kind"] for item in captures],
        sorted(CAPTURE_KINDS),
        "spec.capture_entrypoints.order",
    )

    processes = value["managed_processes"]
    require(type(processes) is list, "E_MANAGED_PROCESSES")
    process_map = {}
    for index, process in enumerate(processes):
        field = f"spec.managed_processes[{index}]"
        process = exact_keys(
            process,
            {
                "bound",
                "bundle_id",
                "expectation",
                "process_id",
                "prospective",
            },
            field,
        )
        process_id = identifier(process["process_id"], f"{field}.process_id")
        require(process_id not in process_map, f"E_PROCESS_REUSE: {process_id}")
        bundle_id = identifier(process["bundle_id"], f"{field}.bundle_id")
        require(
            bundle_id in bundle_map
            and bundle_map[bundle_id]["managed_plan_required"],
            f"E_PROCESS_BUNDLE: {process_id}",
        )
        exact(process_id, bundle_id, f"{field}.identity")
        for side in ("prospective", "bound"):
            side_value = exact_keys(
                process[side],
                {"argv", "boot_id"},
                f"{field}.{side}",
            )
            text(side_value["boot_id"], f"{field}.{side}.boot_id", 64)
            require(
                type(side_value["argv"]) is list and bool(side_value["argv"]),
                f"E_ARGV: {field}.{side}",
            )
        require(type(process["expectation"]) is dict, f"E_EXPECTATION: {field}")
        process_map[process_id] = process
    exact(
        [item["process_id"] for item in processes],
        sorted(process_map),
        "spec.managed_processes.order",
    )
    exact(
        set(process_map),
        {
            bundle_id
            for bundle_id, bundle in bundle_map.items()
            if bundle["managed_plan_required"]
        },
        "spec.managed_processes.closure",
    )
    return value


def _validate_stat(value: Any, field: str) -> dict[str, Any]:
    value = exact_keys(value, STAT_KEYS, field)
    exact(value["build_id"], None, f"{field}.build_id")
    for key in STAT_KEYS - {"build_id"}:
        integer(value[key], f"{field}.{key}")
    require(stat.S_ISREG(value["mode"]), f"E_STAT_MODE: {field}")
    return value


def _managed_components(
    components: dict[str, dict[str, Any]],
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "bytes": components[component_id]["bytes"],
            "component_id": component_id,
            "path": components[component_id]["path"],
            "sha256": components[component_id]["sha256"],
            "stat": components[component_id]["stat"],
        }
        for component_id in bundle["required_component_ids"]
    ]


def materialize_runtime_inventory(
    spec: dict[str, Any],
    *,
    local_observer: DirectoryObserver | None = None,
    android_observer: DirectoryObserver,
    managed_gate: Any | None = None,
) -> dict[str, Any]:
    """Observe exact closures and emit a canonical, digest-bound inventory."""

    spec = _validate_spec(json.loads(canonical_bytes(spec)))
    local_observer = local_observer or LocalDirectoryObserver()
    managed_gate = managed_gate or _load_managed_gate()
    observers = {
        "android": android_observer,
        "local": local_observer,
    }
    components = []
    bundles = []
    component_map = {}
    bundle_map = {}
    for bundle in spec["bundles"]:
        observed = observers[bundle["transport"]].observe(
            endpoint=bundle["endpoint"],
            root=bundle["root"],
            files=bundle["files"],
        )
        exact(len(observed), len(bundle["files"]), f"observe.{bundle['bundle_id']}")
        required_ids = []
        for expected, pin in zip(bundle["files"], observed):
            pin = exact_keys(
                pin,
                {"bytes", "path", "sha256", "stat"},
                f"pin.{expected['component_id']}",
            )
            exact(pin["bytes"], expected["bytes"], f"{expected['component_id']}.bytes")
            exact(
                pin["sha256"],
                expected["sha256"],
                f"{expected['component_id']}.sha256",
            )
            exact(
                pin["path"],
                f"{bundle['root']}/{expected['filename']}",
                f"{expected['component_id']}.path",
            )
            metadata = _validate_stat(pin["stat"], f"{expected['component_id']}.stat")
            exact(metadata["size"], pin["bytes"], f"{expected['component_id']}.size")
            row = {
                "bundle_id": bundle["bundle_id"],
                "bytes": pin["bytes"],
                "component_id": expected["component_id"],
                "endpoint": bundle["endpoint"],
                "path": pin["path"],
                "role": expected["role"],
                "sha256": pin["sha256"],
                "stat": metadata,
                "transport": bundle["transport"],
            }
            components.append(row)
            component_map[row["component_id"]] = row
            required_ids.append(row["component_id"])
        bundle_row = {
            "bundle_id": bundle["bundle_id"],
            "endpoint": bundle["endpoint"],
            "launcher_component_id": bundle["launcher_component_id"],
            "managed_plan_required": bundle["managed_plan_required"],
            "process_role": bundle["process_role"],
            "required_component_ids": required_ids,
            "root": bundle["root"],
            "transport": bundle["transport"],
        }
        bundles.append(bundle_row)
        bundle_map[bundle_row["bundle_id"]] = bundle_row

    managed_processes = []
    launcher_pins = {}
    for process in spec["managed_processes"]:
        expectation = json.loads(canonical_bytes(process["expectation"]))
        bundle = bundle_map[process["bundle_id"]]
        exact(
            expectation.get("components"),
            _managed_components(component_map, bundle),
            f"managed.{process['process_id']}.components",
        )
        managed_launcher = expectation.get("managed_launcher")
        require(type(managed_launcher) is dict, "E_MANAGED_LAUNCHER")
        launcher_path = absolute(
            managed_launcher.get("path"),
            f"managed.{process['process_id']}.launcher.path",
        )
        launcher_pin = launcher_pins.get(launcher_path)
        if launcher_pin is None:
            launcher_pin = local_observer.pin_launcher(launcher_path)
            launcher_pins[launcher_path] = launcher_pin
        exact(
            launcher_pin["sha256"],
            managed_launcher.get("sha256"),
            f"managed.{process['process_id']}.launcher.sha256",
        )
        result = managed_gate.validate_managed_plan_pair(
            process["prospective"]["argv"],
            process["bound"]["argv"],
            expectation=expectation,
            prospective_boot_id=process["prospective"]["boot_id"],
            bound_boot_id=process["bound"]["boot_id"],
        )
        managed_processes.append(
            {
                "bound": process["bound"],
                "bundle_id": process["bundle_id"],
                "expectation": expectation,
                "gate_result": result,
                "managed_launcher_pin": launcher_pin,
                "process_id": process["process_id"],
                "prospective": process["prospective"],
            }
        )

    body = {
        "bundles": bundles,
        "capture_entrypoints": spec["capture_entrypoints"],
        "components": components,
        "managed_processes": managed_processes,
        "phase": spec["phase"],
        "phase_id": spec["phase_id"],
        "spec_sha256": sha256(canonical_bytes(spec)),
    }
    record = {
        "inventory": body,
        "inventory_sha256": sha256(canonical_bytes(body)),
        "schema": SCHEMA,
    }
    validate_runtime_inventory(record, managed_gate=managed_gate)
    return record


def validate_runtime_inventory(
    record: Any,
    *,
    managed_gate: Any | None = None,
) -> dict[str, Any]:
    """Recompute all cross-record and managed-plan predicates."""

    managed_gate = managed_gate or _load_managed_gate()
    record = exact_keys(
        record,
        {"inventory", "inventory_sha256", "schema"},
        "record",
    )
    exact(record["schema"], SCHEMA, "record.schema")
    digest(record["inventory_sha256"], "record.inventory_sha256")
    body = exact_keys(record["inventory"], BODY_KEYS, "record.inventory")
    exact(
        record["inventory_sha256"],
        sha256(canonical_bytes(body)),
        "record.inventory_sha256",
    )
    require(body["phase"] in PHASES, "E_PHASE")
    identifier(body["phase_id"], "record.phase_id")
    digest(body["spec_sha256"], "record.spec_sha256")

    components = body["components"]
    require(type(components) is list and bool(components), "E_COMPONENTS")
    component_map = {}
    for index, component in enumerate(components):
        field = f"record.components[{index}]"
        component = exact_keys(
            component,
            {
                "bundle_id",
                "bytes",
                "component_id",
                "endpoint",
                "path",
                "role",
                "sha256",
                "stat",
                "transport",
            },
            field,
        )
        component_id = identifier(component["component_id"], f"{field}.component_id")
        require(component_id not in component_map, f"E_COMPONENT_REUSE: {component_id}")
        identifier(component["bundle_id"], f"{field}.bundle_id")
        identifier(component["endpoint"], f"{field}.endpoint")
        absolute(component["path"], f"{field}.path")
        require(component["role"] in COMPONENT_ROLES, f"E_ROLE: {field}")
        require(component["transport"] in TRANSPORTS, f"E_TRANSPORT: {field}")
        size = integer(component["bytes"], f"{field}.bytes", 1)
        digest(component["sha256"], f"{field}.sha256")
        metadata = _validate_stat(component["stat"], f"{field}.stat")
        exact(metadata["size"], size, f"{field}.stat.size")
        component_map[component_id] = component
    exact(
        [item["component_id"] for item in components],
        sorted(component_map),
        "record.components.order",
    )

    bundles = body["bundles"]
    require(type(bundles) is list and bool(bundles), "E_BUNDLES")
    bundle_map = {}
    for index, bundle in enumerate(bundles):
        field = f"record.bundles[{index}]"
        bundle = exact_keys(
            bundle,
            {
                "bundle_id",
                "endpoint",
                "launcher_component_id",
                "managed_plan_required",
                "process_role",
                "required_component_ids",
                "root",
                "transport",
            },
            field,
        )
        bundle_id = identifier(bundle["bundle_id"], f"{field}.bundle_id")
        require(bundle_id not in bundle_map, f"E_BUNDLE_REUSE: {bundle_id}")
        identifier(bundle["endpoint"], f"{field}.endpoint")
        identifier(bundle["process_role"], f"{field}.process_role")
        absolute(bundle["root"], f"{field}.root")
        require(bundle["transport"] in TRANSPORTS, f"E_TRANSPORT: {field}")
        require(
            type(bundle["managed_plan_required"]) is bool,
            f"E_BOOL: {field}.managed_plan_required",
        )
        required_ids = bundle["required_component_ids"]
        require(
            type(required_ids) is list
            and bool(required_ids)
            and required_ids == sorted(set(required_ids)),
            f"E_REQUIRED_COMPONENTS: {bundle_id}",
        )
        actual_ids = sorted(
            component_id
            for component_id, component in component_map.items()
            if component["bundle_id"] == bundle_id
        )
        exact(required_ids, actual_ids, f"{field}.closure")
        for component_id in required_ids:
            component = component_map[component_id]
            exact(component["endpoint"], bundle["endpoint"], f"{field}.endpoint")
            exact(component["transport"], bundle["transport"], f"{field}.transport")
            exact(
                str(Path(component["path"]).parent),
                bundle["root"],
                f"{field}.root",
            )
        launcher = bundle["launcher_component_id"]
        require(
            launcher in required_ids
            and component_map[launcher]["role"] == "executable",
            f"E_LAUNCHER: {bundle_id}",
        )
        bundle_map[bundle_id] = bundle
    exact(
        [item["bundle_id"] for item in bundles],
        sorted(bundle_map),
        "record.bundles.order",
    )
    exact(
        {
            component["bundle_id"]
            for component in components
        },
        set(bundle_map),
        "record.bundle_component_closure",
    )

    captures = body["capture_entrypoints"]
    require(type(captures) is list and len(captures) == 4, "E_CAPTURES")
    capture_map = {}
    for index, capture in enumerate(captures):
        field = f"record.capture_entrypoints[{index}]"
        capture = exact_keys(
            capture,
            {"bundle_id", "component_id", "kind"},
            field,
        )
        kind = text(capture["kind"], f"{field}.kind", 64)
        require(kind in CAPTURE_KINDS and kind not in capture_map, "E_CAPTURE_KIND")
        exact(capture["bundle_id"], CAPTURE_BUNDLES[kind], f"{field}.bundle")
        component_id = capture["component_id"]
        require(
            component_id in component_map
            and component_map[component_id]["bundle_id"] == capture["bundle_id"]
            and component_map[component_id]["role"] == "capture_entrypoint",
            f"E_CAPTURE_COMPONENT: {kind}",
        )
        capture_map[kind] = capture
    exact(
        [item["kind"] for item in captures],
        sorted(CAPTURE_KINDS),
        "record.capture_entrypoints.order",
    )

    processes = body["managed_processes"]
    require(type(processes) is list, "E_MANAGED_PROCESSES")
    process_map = {}
    launcher_pins = {}
    for index, process in enumerate(processes):
        field = f"record.managed_processes[{index}]"
        process = exact_keys(
            process,
            {
                "bound",
                "bundle_id",
                "expectation",
                "gate_result",
                "managed_launcher_pin",
                "process_id",
                "prospective",
            },
            field,
        )
        process_id = identifier(process["process_id"], f"{field}.process_id")
        require(process_id not in process_map, f"E_PROCESS_REUSE: {process_id}")
        exact(process_id, process["bundle_id"], f"{field}.identity")
        require(
            process["bundle_id"] in bundle_map
            and bundle_map[process["bundle_id"]]["managed_plan_required"],
            f"E_PROCESS_BUNDLE: {process_id}",
        )
        bundle = bundle_map[process["bundle_id"]]
        exact(
            process["expectation"].get("components"),
            _managed_components(component_map, bundle),
            f"{field}.components",
        )
        launcher_pin = exact_keys(
            process["managed_launcher_pin"],
            {"bytes", "path", "sha256", "stat"},
            f"{field}.managed_launcher_pin",
        )
        integer(launcher_pin["bytes"], f"{field}.launcher.bytes", 1)
        absolute(launcher_pin["path"], f"{field}.launcher.path")
        digest(launcher_pin["sha256"], f"{field}.launcher.sha256")
        metadata = _validate_stat(launcher_pin["stat"], f"{field}.launcher.stat")
        exact(metadata["size"], launcher_pin["bytes"], f"{field}.launcher.size")
        managed_launcher = process["expectation"].get("managed_launcher")
        require(type(managed_launcher) is dict, f"E_MANAGED_LAUNCHER: {field}")
        exact(
            {
                "path": launcher_pin["path"],
                "sha256": launcher_pin["sha256"],
            },
            managed_launcher,
            f"{field}.managed_launcher",
        )
        prior = launcher_pins.get(launcher_pin["path"])
        if prior is not None:
            exact(prior, launcher_pin, f"{field}.launcher_reuse")
        launcher_pins[launcher_pin["path"]] = launcher_pin
        for side in ("prospective", "bound"):
            side_value = exact_keys(
                process[side],
                {"argv", "boot_id"},
                f"{field}.{side}",
            )
            text(side_value["boot_id"], f"{field}.{side}.boot_id", 64)
            require(type(side_value["argv"]) is list, f"E_ARGV: {field}.{side}")
        result = managed_gate.validate_managed_plan_pair(
            process["prospective"]["argv"],
            process["bound"]["argv"],
            expectation=process["expectation"],
            prospective_boot_id=process["prospective"]["boot_id"],
            bound_boot_id=process["bound"]["boot_id"],
        )
        exact(result, process["gate_result"], f"{field}.gate_result")
        process_map[process_id] = process
    exact(
        [item["process_id"] for item in processes],
        sorted(process_map),
        "record.managed_processes.order",
    )
    exact(
        set(process_map),
        {
            bundle_id
            for bundle_id, bundle in bundle_map.items()
            if bundle["managed_plan_required"]
        },
        "record.managed_processes.closure",
    )
    return {
        "inventory_sha256": record["inventory_sha256"],
        "managed_process_count": len(process_map),
        "phase": body["phase"],
        "phase_id": body["phase_id"],
        "runtime_component_count": len(component_map),
        "schema": "s39-cp0-r1-v26-runtime-inventory-validation-v1",
        "status": "RUNTIME_INVENTORY_PASS",
    }
