"""Persistent stat-bound cache for validated GGUF model manifests."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import tempfile

from .model_manifest import (
    GGUFModelManifestLoader,
    ModelManifest,
    ModelManifestError,
)


MODEL_MANIFEST_CACHE_SCHEMA = "model-manifest-cache-v2"


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        row = path.stat()
    except OSError as error:
        raise ModelManifestError(
            "cannot stat GGUF artifact: " + str(error)
        ) from error
    if not path.is_file():
        raise ModelManifestError("GGUF artifact does not exist")
    return (
        row.st_dev,
        row.st_ino,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )


def _read_entries(path: Path) -> dict[tuple[object, ...], ModelManifest]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if type(value) is not dict or value.get("schema") != (
        MODEL_MANIFEST_CACHE_SCHEMA
    ):
        return {}
    rows = value.get("entries")
    if type(rows) is not list:
        return {}
    result = {}
    try:
        for row in rows:
            if type(row) is not dict:
                return {}
            model_id = row.get("model_id")
            source_path = row.get("path")
            identity = row.get("file_identity")
            manifest = ModelManifest.from_json(row.get("manifest"))
            if (
                type(model_id) is not str
                or not model_id
                or not model_id.isascii()
                or type(source_path) is not str
                or not source_path.startswith("/")
                or not source_path.isascii()
                or type(identity) is not list
                or len(identity) != 5
                or any(type(value) is not int or value < 0 for value in identity)
                or manifest.model_id != model_id
                or manifest.artifact_bytes != identity[2]
            ):
                return {}
            result[(model_id, source_path, *identity)] = manifest
    except ModelManifestError:
        return {}
    return result


def _write_entries(
    path: Path,
    entries: dict[tuple[object, ...], ModelManifest],
) -> None:
    ordered = sorted(entries.items())[-128:]
    value = {
        "entries": [
            {
                "file_identity": list(key[2:]),
                "manifest": manifest.to_json(),
                "model_id": key[0],
                "path": key[1],
            }
            for key, manifest in ordered
        ],
        "schema": MODEL_MANIFEST_CACHE_SCHEMA,
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


def load_cached_gguf_manifest(
    model_id: str,
    source: str | Path,
    cache_path: Path | None,
) -> ModelManifest:
    path = Path(source).resolve()
    if cache_path is None:
        return GGUFModelManifestLoader.load(model_id, path)
    if not isinstance(cache_path, Path) or not cache_path.is_absolute():
        raise ModelManifestError("GGUF manifest cache path is invalid")
    identity = _file_identity(path)
    key = (model_id, str(path), *identity)
    entries = _read_entries(cache_path)
    cached = entries.get(key)
    if cached is not None:
        return cached
    manifest = GGUFModelManifestLoader.load(model_id, path)
    if _file_identity(path) != identity:
        raise ModelManifestError("GGUF artifact changed during inspection")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_name(cache_path.name + ".lock")
    try:
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            entries = _read_entries(cache_path)
            entries[key] = manifest
            _write_entries(cache_path, entries)
    except OSError:
        pass
    return manifest


__all__ = [
    "MODEL_MANIFEST_CACHE_SCHEMA",
    "load_cached_gguf_manifest",
]
