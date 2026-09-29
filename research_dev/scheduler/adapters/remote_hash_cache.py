"""Boot-bound cache for hashes of artifacts deployed to remote executors."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Mapping


REMOTE_ARTIFACT_HASH_CACHE_SCHEMA = "remote-artifact-hash-cache-v1"


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


@dataclass(frozen=True)
class RemoteFileIdentity:
    boot_id: str
    stat_identity: str

    def __post_init__(self) -> None:
        stat_fields = self.stat_identity.split(":")
        if (
            type(self.boot_id) is not str
            or not self.boot_id
            or not self.boot_id.isascii()
            or type(self.stat_identity) is not str
            or not self.stat_identity
            or not self.stat_identity.isascii()
            or any(
                character not in "0123456789:"
                for character in self.stat_identity
            )
            or len(stat_fields) != 5
            or any(not field.isdigit() for field in stat_fields)
        ):
            raise ValueError("remote file identity is invalid")

    @property
    def size_bytes(self) -> int:
        return int(self.stat_identity.split(":")[2])


def _read_entries(path: Path) -> dict[tuple[str, str, str, str], str]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if type(value) is not dict or value.get("schema") != (
        REMOTE_ARTIFACT_HASH_CACHE_SCHEMA
    ):
        return {}
    rows = value.get("entries")
    if type(rows) is not list:
        return {}
    result = {}
    for row in rows:
        if type(row) is not dict:
            return {}
        fields = (
            row.get("serial"),
            row.get("boot_id"),
            row.get("path"),
            row.get("stat_identity"),
        )
        sha256 = row.get("sha256")
        if (
            any(
                type(field) is not str
                or not field
                or not field.isascii()
                for field in fields
            )
            or not _valid_sha256(sha256)
        ):
            return {}
        result[fields] = sha256
    return result


def cached_remote_hashes(
    path: Path,
    *,
    serial: str,
    identities: Mapping[str, RemoteFileIdentity],
) -> Mapping[str, str]:
    entries = _read_entries(path)
    result = {}
    for remote_path, identity in identities.items():
        key = (
            serial,
            identity.boot_id,
            remote_path,
            identity.stat_identity,
        )
        sha256 = entries.get(key)
        if sha256 is not None:
            result[remote_path] = sha256
    return MappingProxyType(dict(sorted(result.items())))


def update_remote_hash_cache(
    path: Path,
    *,
    serial: str,
    identities: Mapping[str, RemoteFileIdentity],
    hashes: Mapping[str, str],
) -> None:
    if set(hashes) - set(identities) or any(
        not _valid_sha256(value) for value in hashes.values()
    ):
        raise ValueError("remote hash cache update is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        entries = _read_entries(path)
        for remote_path, sha256 in hashes.items():
            identity = identities[remote_path]
            entries[
                (
                    serial,
                    identity.boot_id,
                    remote_path,
                    identity.stat_identity,
                )
            ] = sha256
        ordered = sorted(entries.items())[-2048:]
        value = {
            "entries": [
                {
                    "boot_id": key[1],
                    "path": key[2],
                    "serial": key[0],
                    "sha256": sha256,
                    "stat_identity": key[3],
                }
                for key, sha256 in ordered
            ],
            "schema": REMOTE_ARTIFACT_HASH_CACHE_SCHEMA,
        }
        encoded = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(encoded)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, path)
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass


__all__ = [
    "REMOTE_ARTIFACT_HASH_CACHE_SCHEMA",
    "RemoteFileIdentity",
    "cached_remote_hashes",
    "update_remote_hash_cache",
]
