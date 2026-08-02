#!/usr/bin/env python3
"""Shared CP0-R1 V2.3 artifact-certificate helpers."""

from __future__ import annotations

import calendar
import datetime
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from phone_gateway import (
    MAX_COMMAND_BYTES,
    canonical_bytes,
    exact_keys,
    integer,
    require,
    sha256_text,
    strict_json_loads,
    string,
)


STAT_TIME = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})"
    r"\.(\d{1,9}) ([+-])(\d{2})(\d{2})$"
)


def absolute_path(value: Any, field: str) -> str:
    result = string(value, field)
    require(result.startswith("/") and not any(c.isspace() for c in result), field)
    return result


def stat_time_ns(value: str) -> int:
    match = STAT_TIME.fullmatch(value)
    require(match is not None, "remote stat timestamp")
    fields = [int(item) for item in match.groups()[:6]]
    fraction = int(match.group(7).ljust(9, "0"))
    offset = (int(match.group(9)) * 60 + int(match.group(10))) * 60
    if match.group(8) == "-":
        offset = -offset
    utc = calendar.timegm(datetime.datetime(*fields).timetuple()) - offset
    return utc * 1_000_000_000 + fraction


def local_stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "ctime_ns": value.st_ctime_ns,
        "device_id": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "mtime_ns": value.st_mtime_ns,
        "size": value.st_size,
    }


def parse_artifact_certificate(
    path: Path,
    expected_sha256: str,
    model_id: str,
    expected_phase: str,
    expected_slot: str,
    expected_route_lock_sha256: str,
) -> dict[str, Any]:
    sha256_text(expected_sha256, "expected artifact certificate SHA-256")
    sha256_text(expected_route_lock_sha256, "expected route lock SHA-256")
    string(expected_phase, "expected artifact phase")
    string(expected_slot, "expected artifact slot")
    raw = path.read_bytes()
    require(
        0 < len(raw) <= MAX_COMMAND_BYTES,
        "artifact certificate size",
    )
    require(
        hashlib.sha256(raw).hexdigest() == expected_sha256,
        "artifact certificate changed",
    )
    value = strict_json_loads(raw, "artifact certificate JSON")
    require(
        canonical_bytes(value) == raw,
        "artifact certificate is not canonical JSON",
    )
    value = exact_keys(
        value,
        {
            "artifacts",
            "completed_ns",
            "model_id",
            "phase",
            "route_lock_sha256",
            "schema",
            "slot",
            "started_ns",
        },
        "artifact certificate",
    )
    require(
        value["schema"] == "s39-cp0-r1-artifact-snapshot-v2.3",
        "artifact certificate schema",
    )
    require(value["model_id"] == model_id, "artifact certificate model")
    require(value["phase"] == expected_phase, "artifact certificate phase")
    require(value["slot"] == expected_slot, "artifact certificate slot")
    require(
        value["route_lock_sha256"] == expected_route_lock_sha256,
        "artifact certificate route lock",
    )
    started_ns = integer(
        value["started_ns"],
        "artifact certificate started",
        1,
    )
    completed_ns = integer(
        value["completed_ns"],
        "artifact certificate completed",
        1,
    )
    require(started_ns < completed_ns, "artifact certificate interval")
    artifacts = {}
    for row in value["artifacts"]:
        row = exact_keys(
            row,
            {"bytes", "endpoint", "path", "sha256", "stat"},
            "artifact certificate row",
        )
        endpoint = string(row["endpoint"], "artifact endpoint")
        artifact_path = absolute_path(row["path"], "artifact path")
        stat = exact_keys(
            row["stat"],
            {"ctime_ns", "device_id", "inode", "mode", "mtime_ns", "size"},
            "artifact stat",
        )
        for key in stat:
            integer(stat[key], f"artifact stat {key}")
        require(
            stat["inode"] > 0
            and stat["size"] > 0
            and stat["size"] == integer(row["bytes"], "artifact bytes", 1),
            "artifact stat values",
        )
        sha256_text(row["sha256"], "artifact SHA-256")
        key = (endpoint, artifact_path)
        require(key not in artifacts, "duplicate artifact certificate row")
        artifacts[key] = {
            **row,
            "path": artifact_path,
            "stat": stat,
        }
    require(artifacts, "empty artifact certificate")
    return {
        "artifacts": artifacts,
        "completed_ns": completed_ns,
        "phase": expected_phase,
        "path": str(path),
        "route_lock_sha256": expected_route_lock_sha256,
        "sha256": expected_sha256,
        "slot": expected_slot,
        "started_ns": started_ns,
    }


def parse_readiness_lock(
    path: Path,
    expected_sha256: str,
    phase: str,
    artifact_certificate: dict[str, Any],
    v2_2_phase_lock_sha256: str,
) -> dict[str, Any]:
    sha256_text(expected_sha256, "expected readiness lock SHA-256")
    sha256_text(v2_2_phase_lock_sha256, "expected V2.2 phase lock SHA-256")
    raw = path.read_bytes()
    require(0 < len(raw) <= MAX_COMMAND_BYTES, "readiness lock size")
    require(
        hashlib.sha256(raw).hexdigest() == expected_sha256,
        "readiness lock changed",
    )
    value = strict_json_loads(raw, "readiness lock JSON")
    require(canonical_bytes(value) == raw, "readiness lock is not canonical JSON")
    value = exact_keys(
        value,
        {
            "artifact_snapshot_sha256",
            "event_ns",
            "phase",
            "phase_id",
            "schema",
            "v2_2_phase_lock_sha256",
        },
        "readiness lock",
    )
    require(
        value["schema"] == "s39-cp0-r1-readiness-lock-v2.3",
        "readiness lock schema",
    )
    require(value["phase"] == phase, "readiness lock phase")
    require(
        value["artifact_snapshot_sha256"]
        == artifact_certificate["sha256"],
        "readiness lock artifact root",
    )
    require(
        value["v2_2_phase_lock_sha256"] == v2_2_phase_lock_sha256,
        "readiness lock phase root",
    )
    event_ns = integer(value["event_ns"], "readiness lock event", 1)
    require(
        artifact_certificate["completed_ns"] <= event_ns,
        "artifact certificate completed after readiness lock",
    )
    return {
        "event_ns": event_ns,
        "path": str(path),
        "phase_id": string(value["phase_id"], "readiness lock phase ID"),
        "sha256": expected_sha256,
    }
