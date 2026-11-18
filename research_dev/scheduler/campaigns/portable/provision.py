"""Provisioning: what a helper phone serves, as a ``HelperBinding`` on disk.

A binding names the model (artifact sha256, geometry, activation), the shards on the phone (one per
session, each a column suffix of some layers) and the sha256 pins of the phone's worker bundle. It is
written once per (phone, model, share) and checked before every run.

    adopt     take shards that are already on the phone (their index was written by
              ``native/ffn_shard_gguf.py``): read-only, the phone's files are hashed and compared
    slice     write the shards of a share plan with ``native/ffn_shard_gguf.py`` (CPU-heavy; refused while
              a metered run holds the site lock)
    push      copy sliced shards (and, on request, a worker bundle) into the site's own directories on
              the phone and verify them

Nothing here writes outside the phone's ``work_directory`` / ``bundle_directory`` of the site file.
"""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Callable, Mapping, Sequence

from ...adapters.contracts import PhysicalAdapterError
from ..._internal.layer_placement import layer_spec
from ..._internal.layer_placement_onboarding import SPLIT_ARCHITECTURES
from ..._internal.shared_layer import StaticSharePlan
from ...configuration.site import SiteManifest, SitePhone
from ...drivers.adb import AdbGuard, sha256_file
from ...drivers.doctor import RunLock
from ...drivers.phone import HelperBinding, HelperShard, allowed_phone_roots


BINDING_SCHEMA = "research-scheduler-helper-binding-v1"
SHARD_INDEX_SCHEMA = "s42-ffn-shard-index-v1"
SHARD_INDEX_NAME = "FFN_SHARDS.json"


def binding_document(binding: HelperBinding, *, site_id: str, device_id: str, model_id: str, weight_type: str,
                     share_plan: Mapping[str, object] | None = None) -> dict[str, object]:
    return {"schema": BINDING_SCHEMA, "site_id": site_id, "device_id": device_id, "model_id": model_id,
            "weight_type": weight_type, "binding": binding.to_json(),
            **({"share_plan": dict(share_plan)} if share_plan is not None else {})}


def save_binding(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="ascii")


def load_binding(path: Path) -> tuple[HelperBinding, dict[str, object]]:
    """(binding, its document) from a binding file."""
    document = json.loads(Path(path).read_text(encoding="ascii"))
    if document.get("schema") != BINDING_SCHEMA:
        raise PhysicalAdapterError(f"{path} is not a helper binding")
    return HelperBinding.from_json(document["binding"]), document


def binding_path(site: SiteManifest, device_id: str, model_id: str, columns: int) -> Path:
    return site.paths.work_root / "bindings" / f"{device_id}-{model_id}-s{columns}.json"


def shards_from_index(index: Mapping[str, object], remote_directory: str) -> tuple[HelperShard, ...]:
    """The shards of an index written by ``ffn_shard_gguf.py``, at their place on the phone."""
    if index.get("schema") != SHARD_INDEX_SCHEMA or not index.get("shards"):
        raise PhysicalAdapterError("not an FFN shard index")
    shards = []
    for row in index["shards"]:
        mask = int(row["layer_mask"], 16)
        shards.append(HelperShard(
            session_id=str(row["session_id"]), remote_path=remote_directory.rstrip("/") + "/" + str(row["path"]),
            sha256=str(row["shard_sha256"]), layers=tuple(layer for layer in range(64) if mask >> layer & 1),
            columns=int(row["columns"]), bytes=int(row["shard_bytes"])))
    return tuple(shards)


def bundle_pins(adb: AdbGuard, phone: SitePhone) -> dict[str, str]:
    """sha256 of every file of the phone's worker bundle, as it is now (read-only)."""
    listing = adb.shell(phone.serial, "ls " + shlex.quote(phone.bundle_directory), root=phone.as_root,
                        timeout_s=30).split()
    paths = [phone.bundle_directory + "/" + name for name in listing]
    pins = adb.remote_sha256(phone.serial, paths, root=phone.as_root)
    if not pins:
        raise PhysicalAdapterError(f"the bundle directory {phone.bundle_directory} is empty or unreadable")
    return pins


def adopt(site: SiteManifest, device_id: str, model_id: str, index_path: Path, remote_directory: str, *,
          adb: AdbGuard | None = None) -> dict[str, object]:
    """A binding for shards that already sit on the phone; their sha256 must equal the index's."""
    phone = site.phone(device_id)
    if not any(remote_directory.rstrip("/").startswith(root) for root in allowed_phone_roots(phone)):
        raise PhysicalAdapterError(f"{remote_directory} is outside this site's directories on the phone")
    adb = adb or AdbGuard(site.paths.adb_path, phone.adb_port)
    index = json.loads(Path(index_path).read_text(encoding="ascii"))
    architecture = str(index.get("architecture"))
    if architecture not in SPLIT_ARCHITECTURES:
        raise PhysicalAdapterError("the shard index names an architecture without an FFN split: " + architecture)
    shards = shards_from_index(index, remote_directory)
    observed = adb.remote_sha256(phone.serial, [shard.remote_path for shard in shards], root=phone.as_root)
    wrong = [shard.remote_path for shard in shards if observed.get(shard.remote_path) != shard.sha256]
    if wrong:
        raise PhysicalAdapterError("shards on the phone differ from the index: " + ", ".join(wrong))
    binding = HelperBinding(str(index["parent_sha256"]), int(index["n_embd"]), int(index["n_ff"]),
                            SPLIT_ARCHITECTURES[architecture], shards, bundle_pins(adb, phone))
    return binding_document(binding, site_id=site.site_id, device_id=device_id, model_id=model_id,
                            weight_type=str(index["weight_type"]))


def shard_specs(plan: StaticSharePlan, session_prefix: str = "S") -> list[str]:
    """``NAME=LAYERS:COLUMNS`` arguments of ``ffn_shard_gguf.py`` for a share plan (one shard per session)."""
    if not plan.beneficial:
        raise PhysicalAdapterError("the share plan keeps nothing on the phone")
    return [f"{session_prefix}{index}={layer_spec(layers)}:{plan.columns}" for index, layers in enumerate(plan.sessions)]


def slice_shards(site: SiteManifest, model_id: str, parent_sha256: str, specs: Sequence[str], out_directory: Path, *,
                 run: Callable = subprocess.run, lock_held: Callable[[Path], bool] = RunLock.held) -> dict[str, object]:
    """Write the shards with the repository's slicing tool; returns the index it wrote."""
    if lock_held(site.paths.work_root):
        raise PhysicalAdapterError("a metered run holds the site lock; shard slicing would disturb it")
    tool = site.paths.repo_root / "research_dev/scheduler/native/ffn_shard_gguf.py"
    out_directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(tool), str(site.paths.models[model_id]), "--parent-sha256", parent_sha256,
               "--out-dir", str(out_directory), *(word for spec in specs for word in ("--shard", spec))]
    run(command, check=True, cwd=str(site.paths.repo_root))
    return json.loads((out_directory / SHARD_INDEX_NAME).read_text(encoding="ascii"))


def push_shards(site: SiteManifest, device_id: str, index: Mapping[str, object], local_directory: Path,
                remote_directory: str, *, adb: AdbGuard | None = None) -> dict[str, str]:
    """Copy the shards of an index to the phone (skipping those already there); remote path -> present | pushed."""
    phone = site.phone(device_id)
    adb = adb or AdbGuard(site.paths.adb_path, phone.adb_port)
    result = {}
    for shard, row in zip(shards_from_index(index, remote_directory), index["shards"]):
        result[shard.remote_path] = adb.push_verified(
            phone.serial, Path(local_directory) / str(row["path"]), shard.remote_path, shard.sha256,
            allowed_roots=allowed_phone_roots(phone), root=phone.as_root)
    return result


def push_bundle(site: SiteManifest, device_id: str, local_directory: Path, *,
                adb: AdbGuard | None = None) -> dict[str, str]:
    """Copy a worker bundle (binaries and their libraries) into the phone's bundle directory."""
    phone = site.phone(device_id)
    adb = adb or AdbGuard(site.paths.adb_path, phone.adb_port)
    files = sorted(path for path in Path(local_directory).iterdir() if path.is_file())
    if not files:
        raise PhysicalAdapterError(f"{local_directory} holds no bundle file")
    result = {}
    for path in files:
        remote = phone.bundle_directory + "/" + path.name
        result[remote] = adb.push_verified(phone.serial, path, remote, sha256_file(path),
                                           allowed_roots=allowed_phone_roots(phone), root=phone.as_root)
    adb.shell(phone.serial, "chmod 755 " + " ".join(shlex.quote(remote) for remote in result), root=phone.as_root,
              timeout_s=30)
    return result
