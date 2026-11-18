"""Discovery: read a site's facts from the machine and keep them current in ``site.json``.

``host_facts`` reads the host (CPU, memory, GPUs by index and UUID, readable RAPL zones);
``phone_facts`` reads one named phone over an existing adb server and its USB link from the host's sysfs.
``refresh`` replaces the facts in a site document and leaves every declared field as it is, and
``changes`` lists what moved since the last time (a phone on another port, a different kernel).

Discovery runs phone commands only for phones it is told about: a phone that merely appears in
``adb devices`` is listed with its host-side USB facts and nothing else.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Callable, Mapping, Sequence

from ..adapters.contracts import PhysicalAdapterError
from ..adapters.phone_helpers import find_usb_device_by_serial, observe_usb_port
from ..configuration.site import HOST_FACTS, PHONE_FACTS, SITE_MANIFEST_SCHEMA
from .adb import AdbGuard
from .phone import read_phone_facts


RAPL_ROOT = Path("/sys/class/powercap")
USB_ROOT = Path("/sys/bus/usb/devices")


def host_gpus(*, run: Callable = subprocess.run) -> list[dict[str, object]]:
    done = run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
               capture_output=True, text=True, timeout=10, check=True)
    gpus = []
    for line in done.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise PhysicalAdapterError("GPU listing is invalid: " + line)
        gpus.append({"index": int(fields[0]), "uuid": fields[1], "name": fields[2],
                     "memory_total_bytes": int(fields[3]) * 1024 * 1024})
    if not gpus:
        raise PhysicalAdapterError("no GPU is listed")
    return gpus


def rapl_zones(root: Path = RAPL_ROOT) -> list[str]:
    """Package zones whose energy counter this account can read."""
    zones = []
    for zone in sorted(root.glob("intel-rapl:*")):
        if zone.name.count(":") != 1:
            continue
        try:
            if zone.joinpath("name").read_text(encoding="ascii").strip().startswith("package"):
                int(zone.joinpath("energy_uj").read_text(encoding="ascii"))
                zones.append(zone.name)
        except (OSError, ValueError):
            continue
    return zones


def host_facts(*, run: Callable = subprocess.run, cpuinfo: Path = Path("/proc/cpuinfo"),
               meminfo: Path = Path("/proc/meminfo"), rapl_root: Path = RAPL_ROOT,
               cpu_count: Callable[[], int | None] = os.cpu_count) -> dict[str, object]:
    model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text(encoding="ascii", errors="replace")
                  .splitlines() if line.startswith("model name")), None)
    total = next((int(line.split()[1]) * 1024 for line in meminfo.read_text(encoding="ascii").splitlines()
                  if line.startswith("MemTotal:")), None)
    if model is None or total is None or not cpu_count():
        raise PhysicalAdapterError("host CPU or memory facts are unreadable")
    return {"cpu_model": model, "logical_cpus": int(cpu_count()), "memory_total_bytes": total,
            "gpus": host_gpus(run=run), "rapl_zones": rapl_zones(rapl_root)}


def usb_facts(serial: str, *, sysfs_root: Path = USB_ROOT) -> dict[str, object]:
    """Where the phone sits on the host's USB tree (host side only; nothing runs on the phone)."""
    device = find_usb_device_by_serial(serial, sysfs_root=sysfs_root)
    if device is None:
        raise PhysicalAdapterError("no unique USB device has serial " + serial)
    port = observe_usb_port(device, sysfs_root=sysfs_root)
    return {"usb_sysfs_device": port.sysfs_device, "usb_speed_mbps": port.negotiated_speed_mbps}


def phone_facts(adb: AdbGuard, serial: str, *, sysfs_root: Path = USB_ROOT) -> dict[str, object]:
    return {**read_phone_facts(adb, serial), **usb_facts(serial, sysfs_root=sysfs_root)}


def attached_phones(adb: AdbGuard, *, sysfs_root: Path = USB_ROOT) -> list[dict[str, object]]:
    """Every device of the listening adb server with its state and host-side USB facts."""
    rows = []
    for serial, state in adb.devices():
        row: dict[str, object] = {"serial": serial, "state": state}
        try:
            row.update(usb_facts(serial, sysfs_root=sysfs_root))
        except PhysicalAdapterError as error:
            row["usb"] = str(error)
        rows.append(row)
    return rows


def refresh(document: Mapping[str, object], host: Mapping[str, object],
            phones: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """A site document with its facts replaced (``phones``: serial -> facts); declared fields stay."""
    if document.get("schema") != SITE_MANIFEST_SCHEMA:
        raise PhysicalAdapterError("not a site manifest")
    out = dict(document)
    out["host"] = {**dict(document["host"]), **{name: host[name] for name in HOST_FACTS}}
    rows = []
    for row in document.get("phones", []):
        facts = phones.get(row["serial"])
        rows.append({**row, **({name: facts[name] for name in PHONE_FACTS} if facts else {})})
    out["phones"] = rows
    return out


def changes(before: Mapping[str, object], after: Mapping[str, object]) -> list[str]:
    """Human-readable differences between two site documents' facts."""
    out = []
    for name in HOST_FACTS:
        if before["host"].get(name) != after["host"].get(name):
            out.append(f"host {name}: {before['host'].get(name)!r} -> {after['host'].get(name)!r}")
    old = {row["serial"]: row for row in before.get("phones", [])}
    for row in after.get("phones", []):
        prior = old.get(row["serial"], {})
        for name in PHONE_FACTS:
            if prior.get(name) != row.get(name):
                out.append(f"phone {row['device_id']} {name}: {prior.get(name)!r} -> {row.get(name)!r}")
    return out


def new_site_document(site_id: str, host: Mapping[str, object], *, serving_gpu_index: int, threads: int,
                      paths: Mapping[str, object], cpu_affinity: str | None = None) -> dict[str, object]:
    """A first site document: the host's facts, the declared host fields and paths, no phone yet."""
    host_row = {"host_id": site_id, **{name: host[name] for name in HOST_FACTS},
                "serving_gpu_index": serving_gpu_index, "threads": threads}
    if cpu_affinity is not None:
        host_row["cpu_affinity"] = cpu_affinity
    return {"schema": SITE_MANIFEST_SCHEMA, "site_id": site_id, "host": host_row, "phones": [], "paths": dict(paths)}


def add_phone(document: Mapping[str, object], facts: Mapping[str, object],
              declared: Mapping[str, object]) -> dict[str, object]:
    """The site document with one more phone: its discovered facts plus the declared fields."""
    serial = declared["serial"]
    if any(row["serial"] == serial or row["device_id"] == declared["device_id"]
           for row in document.get("phones", [])):
        raise PhysicalAdapterError("the site already has this phone")
    out = dict(document)
    out["phones"] = [*document.get("phones", []), {**declared, **{name: facts[name] for name in PHONE_FACTS}}]
    return out


def phone_serials(document: Mapping[str, object]) -> Sequence[str]:
    return [row["serial"] for row in document.get("phones", [])]
