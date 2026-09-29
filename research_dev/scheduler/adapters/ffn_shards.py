"""Resolve offline FFN shard GGUF files for phone sessions.

``research_dev/scheduler/native/ffn_shard_gguf.py`` writes, per model, a set
of shard GGUFs (one per intended session) and an index ``FFN_SHARDS.json``.
This module loads that index and answers, for one authorized phone shard
(artifact, layer mask, served columns), which stored shard file covers it.
The worker accepts any layer subset of the stored mask and any served suffix
up to the stored width, and computes the same weight hash as it would from the
complete model, so a resolved shard changes only the file the session opens.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

INDEX_SCHEMA = "s42-ffn-shard-index-v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._:/-]+$")


class FfnShardIndexError(ValueError):
    pass


@dataclass(frozen=True)
class FfnShardRecord:
    session_hint: str
    remote_path: str
    shard_sha256: str
    parent_sha256: str
    layer_mask: int
    columns: int
    n_ff: int
    shard_bytes: int
    weight_type: str

    def covers(self, layer_mask: int, columns: int) -> bool:
        return (
            layer_mask != 0
            and (layer_mask & ~self.layer_mask) == 0
            and 0 < columns <= self.columns
        )


@dataclass(frozen=True)
class FfnShardIndex:
    parent_sha256: str
    records: tuple[FfnShardRecord, ...]
    index_sha256: str

    @classmethod
    def from_index(cls, index: Mapping[str, object], remote_dir: str) -> "FfnShardIndex":
        if index.get("schema") != INDEX_SCHEMA:
            raise FfnShardIndexError("FFN shard index schema is not supported")
        parent = index.get("parent_sha256")
        if not isinstance(parent, str) or not _SHA256.match(parent):
            raise FfnShardIndexError("FFN shard index parent sha256 is invalid")
        if not _REMOTE_PATH.match(remote_dir):
            raise FfnShardIndexError("FFN shard remote directory is not a phone path")
        rows = index.get("shards")
        if not isinstance(rows, list) or not rows:
            raise FfnShardIndexError("FFN shard index has no shards")
        records = []
        seen_paths: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise FfnShardIndexError("FFN shard index row is invalid")
            try:
                path = str(row["path"])
                shard_sha256 = str(row["shard_sha256"])
                layer_mask = int(str(row["layer_mask"]), 16)
                columns = int(row["columns"])
                n_ff = int(row["n_ff"])
                shard_bytes = int(row["shard_bytes"])
                weight_type = str(row["weight_type"])
                session_hint = str(row["session_id"])
            except (KeyError, TypeError, ValueError) as error:
                raise FfnShardIndexError("FFN shard index row is incomplete") from error
            if (
                row.get("parent_sha256") != parent
                or not _SHA256.match(shard_sha256)
                or layer_mask <= 0
                or columns <= 0
                or n_ff < columns
                or shard_bytes <= 0
                or "/" in path
                or not path.endswith(".ffn.gguf")
            ):
                raise FfnShardIndexError(f"FFN shard index row {path!r} is invalid")
            remote_path = remote_dir.rstrip("/") + "/" + path
            if not _REMOTE_PATH.match(remote_path) or remote_path in seen_paths:
                raise FfnShardIndexError(f"FFN shard remote path {remote_path!r} is invalid")
            seen_paths.add(remote_path)
            records.append(FfnShardRecord(
                session_hint=session_hint,
                remote_path=remote_path,
                shard_sha256=shard_sha256,
                parent_sha256=parent,
                layer_mask=layer_mask,
                columns=columns,
                n_ff=n_ff,
                shard_bytes=shard_bytes,
                weight_type=weight_type,
            ))
        encoded = json.dumps(
            index,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return cls(
            parent_sha256=parent,
            records=tuple(records),
            index_sha256="sha256:" + hashlib.sha256(encoded).hexdigest(),
        )

    @classmethod
    def load(cls, index_path: Path, remote_dir: str) -> "FfnShardIndex":
        try:
            encoded = index_path.read_bytes()
            index = json.loads(encoded.decode("ascii"))
        except (OSError, ValueError) as error:
            raise FfnShardIndexError(f"cannot read FFN shard index {index_path}") from error
        parsed = cls.from_index(index, remote_dir)
        return cls(
            parent_sha256=parsed.parent_sha256,
            records=parsed.records,
            index_sha256="sha256:" + hashlib.sha256(encoded).hexdigest(),
        )

    def resolve(
        self,
        artifact_sha256: str,
        layer_mask: int,
        columns: int,
        *,
        session_id: str | None = None,
    ) -> FfnShardRecord | None:
        """Smallest stored shard of ``artifact_sha256`` covering the request."""
        if artifact_sha256 != self.parent_sha256:
            return None
        candidates = [
            record for record in self.records
            if record.covers(layer_mask, columns)
            and (
                session_id is None
                or record.session_hint == session_id
            )
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda r: (r.shard_bytes, r.remote_path))


def resolve_ffn_shard(
    indexes: Mapping[str, FfnShardIndex],
    artifact_sha256: str,
    layer_mask: int,
    columns: int,
    *,
    session_id: str | None = None,
) -> FfnShardRecord | None:
    index = indexes.get(artifact_sha256)
    if index is None:
        return None
    return index.resolve(
        artifact_sha256,
        layer_mask,
        columns,
        session_id=session_id,
    )


def remote_hash_entries(indexes: Mapping[str, FfnShardIndex]) -> dict[str, str]:
    """``{"ffn-shard:<sha256>": remote_path}`` for every indexed shard."""
    entries: dict[str, str] = {}
    for index in indexes.values():
        for record in index.records:
            entries["ffn-shard:" + record.shard_sha256] = record.remote_path
    return entries


def verify_remote_hashes(
    indexes: Mapping[str, FfnShardIndex],
    remote_hashes: Mapping[str, str],
    *,
    records: Sequence[FfnShardRecord] | None = None,
) -> None:
    """Every referenced shard on the phone must hash to its index entry."""
    wanted = records if records is not None else [
        record for index in indexes.values() for record in index.records
    ]
    for record in wanted:
        if remote_hashes.get("ffn-shard:" + record.shard_sha256) != record.shard_sha256:
            raise FfnShardIndexError(
                f"phone FFN shard {record.remote_path} differs from its index hash"
            )
