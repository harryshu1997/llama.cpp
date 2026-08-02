#!/usr/bin/python3 -I
"""Read one exact Android process identity through a pinned ADB client."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any


SCHEMA = "s39-cp0-r1-android-process-probe-v1"
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
REMOTE_SCRIPT = r'''
hex() { od -An -tx1 -v "$1" 2>/dev/null | tr -d ' \n'; }
printf 'B %s\n' "$(hex /proc/sys/kernel/random/boot_id)"
for directory in /proc/[0-9]*; do
    [ -r "$directory/cmdline" ] || continue
    [ -r "$directory/stat" ] || continue
    executable="$(readlink "$directory/exe" 2>/dev/null)" || continue
    [ -n "$executable" ] || continue
    executable_hex="$(printf '%s' "$executable" | od -An -tx1 -v | tr -d ' \n')"
    printf 'P %s %s %s %s\n' \
        "${directory##*/}" \
        "$executable_hex" \
        "$(hex "$directory/cmdline")" \
        "$(hex "$directory/stat")"
done
'''.strip()
MAX_OUTPUT = 8 * 1024 * 1024


class ProbeError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


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
        raise ProbeError("E_CANONICAL") from error


def read_regular(path: Path) -> tuple[bytes, os.stat_result]:
    require(path.is_absolute(), "E_ADB_PATH")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProbeError(f"E_ADB_OPEN: {error}") from error
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "E_ADB_TYPE")
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )
    require(
        identity(before) == identity(after) and len(raw) == before.st_size,
        "E_ADB_CHANGED",
    )
    return bytes(raw), before


def parse_expected_argv(value: str) -> list[str]:
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ProbeError(f"E_JSON_NUMBER: {item}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ProbeError("E_EXPECTED_ARGV_JSON") from error
    require(
        type(parsed) is list
        and bool(parsed)
        and all(
            type(item) is str
            and bool(item)
            and "\x00" not in item
            and "\n" not in item
            for item in parsed
        ),
        "E_EXPECTED_ARGV",
    )
    return parsed


def decode_hex(value: str, field: str) -> bytes:
    require(
        len(value) % 2 == 0
        and all(character in "0123456789abcdef" for character in value),
        f"E_REMOTE_HEX: {field}",
    )
    try:
        return bytes.fromhex(value)
    except ValueError as error:
        raise ProbeError(f"E_REMOTE_HEX: {field}") from error


def parse_start_ticks(raw: bytes, pid: int) -> int:
    closing = raw.rfind(b")")
    require(closing > 0 and raw[closing + 1:closing + 2] == b" ", "E_REMOTE_STAT")
    try:
        stat_pid = int(raw[:raw.find(b" ")])
        fields = raw[closing + 2:].split()
        start_ticks = int(fields[19])
    except (IndexError, ValueError) as error:
        raise ProbeError("E_REMOTE_STAT") from error
    require(stat_pid == pid and start_ticks > 0, "E_REMOTE_STAT")
    return start_ticks


def parse_remote(
    raw: bytes,
    expected_boot_id: str,
    expected_executable: str,
    expected_argv: list[str],
    expected_port: int,
    adb_port: int,
    adb_selector: str,
) -> dict[str, Any]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ProbeError("E_REMOTE_ASCII") from error
    lines = text.splitlines()
    require(lines and lines[0].startswith("B "), "E_REMOTE_BOOT_RECORD")
    boot_raw = decode_hex(lines[0][2:], "boot_id")
    try:
        boot_id = boot_raw.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ProbeError("E_REMOTE_BOOT_ID") from error
    require(boot_id == expected_boot_id, "E_REMOTE_BOOT_ID")
    matches = []
    for index, line in enumerate(lines[1:]):
        fields = line.split(" ")
        require(len(fields) == 5 and fields[0] == "P", f"E_REMOTE_RECORD: {index}")
        try:
            pid = int(fields[1])
        except ValueError as error:
            raise ProbeError(f"E_REMOTE_PID: {index}") from error
        require(pid > 0, f"E_REMOTE_PID: {index}")
        try:
            executable = decode_hex(fields[2], f"exe[{index}]").decode("ascii")
            cmdline = decode_hex(fields[3], f"argv[{index}]")
        except UnicodeDecodeError as error:
            raise ProbeError(f"E_REMOTE_TEXT: {index}") from error
        require(cmdline.endswith(b"\x00"), f"E_REMOTE_ARGV_END: {index}")
        try:
            argv = [
                item.decode("ascii")
                for item in cmdline[:-1].split(b"\x00")
            ]
        except UnicodeDecodeError as error:
            raise ProbeError(f"E_REMOTE_ARGV: {index}") from error
        if executable == expected_executable and argv == expected_argv:
            matches.append({
                "adb_port": adb_port,
                "adb_selector": adb_selector,
                "argv": argv,
                "boot_id": boot_id,
                "executable_path": executable,
                "pid": pid,
                "port": expected_port,
                "schema": SCHEMA,
                "start_ticks": parse_start_ticks(
                    decode_hex(fields[4], f"stat[{index}]"),
                    pid,
                ),
            })
    require(len(matches) == 1, "E_REMOTE_PROCESS_COUNT")
    return matches[0]


def adb_argv(
    adb: Path,
    adb_port: int,
    adb_selector: str,
) -> list[str]:
    require(0 < adb_port <= 65535, "E_ADB_PORT")
    require(
        bool(adb_selector)
        and "\x00" not in adb_selector
        and "\n" not in adb_selector,
        "E_ADB_SELECTOR",
    )
    return [
        str(adb),
        "-P",
        str(adb_port),
        "-s",
        adb_selector,
        "shell",
        "sh",
        "-c",
        REMOTE_SCRIPT,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", type=Path, required=True)
    parser.add_argument("--adb-port", type=int, required=True)
    parser.add_argument("--adb-selector", required=True)
    parser.add_argument("--adb-sha256", required=True)
    parser.add_argument("--expected-boot-id", required=True)
    parser.add_argument("--expected-executable", required=True)
    parser.add_argument("--expected-argv-json", required=True)
    parser.add_argument("--expected-port", type=int, required=True)
    args = parser.parse_args()
    try:
        require(
            DIGEST_RE.fullmatch(args.adb_sha256) is not None,
            "E_ADB_SHA256",
        )
        adb_raw, metadata = read_regular(args.adb)
        require(
            hashlib.sha256(adb_raw).hexdigest() == args.adb_sha256,
            "E_ADB_SHA256",
        )
        require(metadata.st_mode & 0o111, "E_ADB_EXECUTABLE")
        require(
            0 < args.expected_port <= 65535,
            "E_EXPECTED_PORT",
        )
        expected_argv = parse_expected_argv(args.expected_argv_json)
        completed = subprocess.run(
            adb_argv(args.adb, args.adb_port, args.adb_selector),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        require(
            len(completed.stdout) <= MAX_OUTPUT
            and len(completed.stderr) <= MAX_OUTPUT,
            "E_ADB_OUTPUT_SIZE",
        )
        require(completed.returncode == 0, "E_ADB_EXIT")
        require(completed.stderr == b"", "E_ADB_STDERR")
        result = parse_remote(
            completed.stdout,
            args.expected_boot_id,
            args.expected_executable,
            expected_argv,
            args.expected_port,
            args.adb_port,
            args.adb_selector,
        )
        sys.stdout.buffer.write(canonical_bytes(result))
        return 0
    except (
        OSError,
        ProbeError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"REMOTE_ANDROID_PROCESS_PROBE_REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
