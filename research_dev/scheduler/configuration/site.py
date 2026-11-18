"""Typed scheduler configuration manifests: the site (one host and the phones attached to it).

A site file answers "what machine is this": facts that discovery reads from the machine (CPU, memory, GPUs
by index and UUID, RAPL zones; per phone its SoC, kernel, memory, root and USB link) and the few fields a
person declares once (which GPU serves, how many threads, where the binaries and models are; per phone
its transport, backend and directories). Nothing about a workload lives here, and no experiment template
needs to know a path, a serial or a USB port: those come from the site.

``discover`` (``drivers/discover.py``) writes and refreshes the facts and never overwrites a declared
field. Measured numbers (layer times, powers, the share) belong to the site profile
(``configuration/site_profile.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .common import (
    SchedulerConfigurationError,
    _boolean,
    _integer,
    _object,
    _optional_text,
    _path,
    _path_json,
    _path_map,
    _require,
    _sequence,
    _text,
    _text_map,
)


SITE_MANIFEST_SCHEMA = "research-scheduler-site-v1"
PHONE_TRANSPORTS = ("adb-tcp", "functionfs-bridge")
PHONE_KINDS = ("htp", "cpu-worker", "pixel-packed")     # the keys of layer_placement_onboarding.DEVICE_FORMATS
BATTERY_CURRENT_UNITS = ("mA", "uA")
# what discovery fills in (never typed by hand) and what a person declares; used by discover's merge
HOST_FACTS = ("cpu_model", "logical_cpus", "memory_total_bytes", "gpus", "rapl_zones")
PHONE_FACTS = ("model", "soc", "kernel_release", "memory_total_bytes", "root", "usb_sysfs_device", "usb_speed_mbps")


def _keys(row: Mapping[str, object], allowed: tuple[str, ...], name: str) -> None:
    unknown = sorted(set(row) - set(allowed))
    _require(not unknown, name + " has unknown keys: " + ",".join(unknown))


def _number(value: object, name: str, *, minimum: float = 0.0) -> float:
    _require(type(value) in (int, float) and value >= minimum, name + " is invalid")
    return float(value)


def _phone_path(value: object, name: str) -> str:
    text = _text(value, name)
    _require(text.startswith("/") and ":" not in text and " " not in text, name + " must be an absolute phone path")
    return text.rstrip("/") or "/"


@dataclass(frozen=True)
class SiteGpu:
    index: int
    uuid: str
    name: str
    memory_total_bytes: int

    @classmethod
    def from_json(cls, value: object) -> "SiteGpu":
        row = _object(value, "site GPU")
        _keys(row, ("index", "uuid", "name", "memory_total_bytes"), "site GPU")
        return cls(_integer(row.get("index"), "site GPU index"), _text(row.get("uuid"), "site GPU UUID"),
                   _text(row.get("name"), "site GPU name"),
                   _integer(row.get("memory_total_bytes"), "site GPU memory", minimum=1))

    def to_json(self) -> dict[str, object]:
        return {"index": self.index, "memory_total_bytes": self.memory_total_bytes, "name": self.name,
                "uuid": self.uuid}


@dataclass(frozen=True)
class SiteHost:
    host_id: str
    cpu_model: str
    logical_cpus: int
    memory_total_bytes: int
    gpus: tuple[SiteGpu, ...]
    rapl_zones: tuple[str, ...]
    serving_gpu_index: int
    threads: int
    cpu_affinity: str | None = None

    @classmethod
    def from_json(cls, value: object) -> "SiteHost":
        row = _object(value, "site host")
        _keys(row, ("host_id", *HOST_FACTS, "serving_gpu_index", "threads", "cpu_affinity"), "site host")
        gpus = tuple(SiteGpu.from_json(item) for item in _sequence(row.get("gpus"), "site GPUs"))
        indices = [gpu.index for gpu in gpus]
        _require(bool(gpus) and len(set(indices)) == len(indices), "site GPUs need distinct indices")
        result = cls(
            host_id=_text(row.get("host_id"), "site host id"),
            cpu_model=_text(row.get("cpu_model"), "site CPU model"),
            logical_cpus=_integer(row.get("logical_cpus"), "site logical CPUs", minimum=1),
            memory_total_bytes=_integer(row.get("memory_total_bytes"), "site host memory", minimum=1),
            gpus=gpus,
            rapl_zones=tuple(_text(item, "site RAPL zone") for item in _sequence(row.get("rapl_zones"), "site RAPL zones")),
            serving_gpu_index=_integer(row.get("serving_gpu_index"), "site serving GPU index"),
            threads=_integer(row.get("threads"), "site threads", minimum=1),
            cpu_affinity=_optional_text(row.get("cpu_affinity"), "site CPU affinity"),
        )
        _require(result.serving_gpu_index in indices, "site serving GPU index names no GPU of this host")
        _require(result.threads <= result.logical_cpus, "site threads exceed the logical CPUs")
        return result

    @property
    def serving_gpu(self) -> SiteGpu:
        return next(gpu for gpu in self.gpus if gpu.index == self.serving_gpu_index)

    def to_json(self) -> dict[str, object]:
        return {
            **({"cpu_affinity": self.cpu_affinity} if self.cpu_affinity is not None else {}),
            "cpu_model": self.cpu_model,
            "gpus": [gpu.to_json() for gpu in self.gpus],
            "host_id": self.host_id,
            "logical_cpus": self.logical_cpus,
            "memory_total_bytes": self.memory_total_bytes,
            "rapl_zones": list(self.rapl_zones),
            "serving_gpu_index": self.serving_gpu_index,
            "threads": self.threads,
        }


@dataclass(frozen=True)
class SitePhonePower:
    """How this phone's whole power is obtained.

    Measured: four sysfs nodes (relative to /sys/class/power_supply, or absolute) and the gauge
    calibration. Assumed: ``assumed_active_w`` / ``assumed_idle_w`` when the nodes cannot be read (for
    example without root); a report then marks the phone's energy ``assumed``.
    """

    usb_current_node: str | None = None
    usb_voltage_node: str | None = None
    battery_current_node: str | None = None
    battery_voltage_node: str | None = None
    battery_current_unit: str = "mA"
    battery_discharge_sign: int = 1
    gauge_ratio: float = 1.0
    gauge_ratio_provenance: str = "uncalibrated"
    needs_root: bool = False
    assumed_active_w: float | None = None
    assumed_idle_w: float | None = None

    _NODES = ("usb_current_node", "usb_voltage_node", "battery_current_node", "battery_voltage_node")

    @property
    def measured(self) -> bool:
        return self.usb_current_node is not None

    @classmethod
    def from_json(cls, value: object) -> "SitePhonePower":
        row = _object(value, "site phone power")
        _keys(row, (*cls._NODES, "battery_current_unit", "battery_discharge_sign", "gauge_ratio",
                    "gauge_ratio_provenance", "needs_root", "assumed_active_w", "assumed_idle_w"), "site phone power")
        nodes = [_optional_text(row.get(name), "site phone power node") for name in cls._NODES]
        _require(all(node is None for node in nodes) or all(node is not None for node in nodes),
                 "site phone power needs all four nodes or none")
        assumed = [row.get("assumed_active_w"), row.get("assumed_idle_w")]
        _require((assumed[0] is None) == (assumed[1] is None), "site phone power needs both assumed powers or none")
        _require(nodes[0] is not None or assumed[0] is not None, "site phone power is neither measured nor assumed")
        unit = _text(row.get("battery_current_unit", "mA"), "site phone battery current unit")
        _require(unit in BATTERY_CURRENT_UNITS, "site phone battery current unit must be mA or uA")
        sign = row.get("battery_discharge_sign", 1)
        _require(type(sign) is int and sign in (1, -1), "site phone battery discharge sign must be +1 or -1")
        return cls(
            *nodes, battery_current_unit=unit, battery_discharge_sign=sign,
            gauge_ratio=_number(row.get("gauge_ratio", 1.0), "site phone gauge ratio", minimum=1e-6),
            gauge_ratio_provenance=_text(row.get("gauge_ratio_provenance", "uncalibrated"),
                                         "site phone gauge ratio provenance"),
            needs_root=_boolean(row.get("needs_root", False), "site phone power root flag"),
            assumed_active_w=None if assumed[0] is None else _number(assumed[0], "site phone assumed active power"),
            assumed_idle_w=None if assumed[1] is None else _number(assumed[1], "site phone assumed idle power"),
        )

    def to_json(self) -> dict[str, object]:
        row: dict[str, object] = {}
        if self.measured:
            row.update({name: getattr(self, name) for name in self._NODES})
            row.update({"battery_current_unit": self.battery_current_unit,
                        "battery_discharge_sign": self.battery_discharge_sign, "gauge_ratio": self.gauge_ratio,
                        "gauge_ratio_provenance": self.gauge_ratio_provenance, "needs_root": self.needs_root})
        if self.assumed_active_w is not None:
            row.update({"assumed_active_w": self.assumed_active_w, "assumed_idle_w": self.assumed_idle_w})
        return dict(sorted(row.items()))


@dataclass(frozen=True)
class SitePhone:
    device_id: str
    serial: str
    adb_port: int
    # facts (discovery)
    model: str
    soc: str
    kernel_release: str
    memory_total_bytes: int
    root: bool
    usb_sysfs_device: str
    usb_speed_mbps: int
    # declared
    transport: str
    kind: str
    backend: str
    bundle_directory: str            # worker binaries and their libraries, on the phone
    work_directory: str              # our own directory on the phone: shards, sessions, sampler
    worker_port: int
    forward_port: int
    column_quantum: int
    max_tokens: int
    worker_environment: Mapping[str, str] = MappingProxyType({})
    sessions: int = 1
    session_limit_bytes: int = 0
    as_root: bool = False
    minimum_usb_speed_mbps: int = 480
    power: SitePhonePower | None = None
    functionfs: Mapping[str, object] | None = None     # the direct-USB section, kept verbatim

    _KEYS = ("device_id", "serial", "adb_port", *PHONE_FACTS, "transport", "kind", "backend", "bundle_directory",
             "work_directory", "worker_port", "forward_port", "column_quantum", "max_tokens", "worker_environment",
             "sessions", "session_limit_bytes", "as_root", "minimum_usb_speed_mbps", "power", "functionfs")

    @classmethod
    def from_json(cls, value: object) -> "SitePhone":
        row = _object(value, "site phone")
        _keys(row, cls._KEYS, "site phone")
        environment = _text_map(row.get("worker_environment", {}), "site phone worker environment")
        _require("LD_LIBRARY_PATH" not in environment, "site phone worker environment sets LD_LIBRARY_PATH")
        functionfs = row.get("functionfs")
        if functionfs is not None:
            try:
                functionfs = MappingProxyType(json.loads(json.dumps(_object(functionfs, "site phone functionfs"),
                                                                    sort_keys=True, allow_nan=False)))
            except (TypeError, ValueError) as error:
                raise SchedulerConfigurationError("site phone functionfs is not JSON: " + str(error)) from error
        result = cls(
            device_id=_text(row.get("device_id"), "site phone device id"),
            serial=_text(row.get("serial"), "site phone serial"),
            adb_port=_integer(row.get("adb_port"), "site phone ADB port", minimum=1, maximum=65535),
            model=_text(row.get("model"), "site phone model"),
            soc=_text(row.get("soc"), "site phone SoC"),
            kernel_release=_text(row.get("kernel_release"), "site phone kernel release"),
            memory_total_bytes=_integer(row.get("memory_total_bytes"), "site phone memory", minimum=1),
            root=_boolean(row.get("root"), "site phone root flag"),
            usb_sysfs_device=_text(row.get("usb_sysfs_device"), "site phone USB sysfs device"),
            usb_speed_mbps=_integer(row.get("usb_speed_mbps"), "site phone USB speed", minimum=1),
            transport=_text(row.get("transport"), "site phone transport"),
            kind=_text(row.get("kind"), "site phone kind"),
            backend=_text(row.get("backend"), "site phone backend"),
            bundle_directory=_phone_path(row.get("bundle_directory"), "site phone bundle directory"),
            work_directory=_phone_path(row.get("work_directory"), "site phone work directory"),
            worker_port=_integer(row.get("worker_port"), "site phone worker port", minimum=1, maximum=65535),
            forward_port=_integer(row.get("forward_port"), "site phone forward port", minimum=1, maximum=65535),
            column_quantum=_integer(row.get("column_quantum"), "site phone column quantum", minimum=32),
            max_tokens=_integer(row.get("max_tokens"), "site phone maximum tokens", minimum=1, maximum=512),
            worker_environment=environment,
            sessions=_integer(row.get("sessions", 1), "site phone sessions", minimum=1),
            session_limit_bytes=_integer(row.get("session_limit_bytes", 0), "site phone session limit"),
            as_root=_boolean(row.get("as_root", False), "site phone root launch flag"),
            minimum_usb_speed_mbps=_integer(row.get("minimum_usb_speed_mbps", 480), "site phone minimum USB speed",
                                            minimum=1),
            power=None if row.get("power") is None else SitePhonePower.from_json(row["power"]),
            functionfs=functionfs,
        )
        _require(result.transport in PHONE_TRANSPORTS, "site phone transport must be one of "
                 + ",".join(PHONE_TRANSPORTS))
        _require(result.kind in PHONE_KINDS, "site phone kind must be one of " + ",".join(PHONE_KINDS))
        _require(not result.as_root or result.root, "site phone launches as root but is not rooted")
        _require(result.transport != "functionfs-bridge" or (result.root and result.as_root
                                                             and result.functionfs is not None),
                 "the direct USB transport needs root and a functionfs section")
        _require(result.sessions == 1 or result.transport == "functionfs-bridge",
                 "several sessions on one phone need the direct USB transport")
        _require(result.power is None or not (result.power.measured and result.power.needs_root) or result.root,
                 "site phone power nodes need root on a phone without root")
        return result

    def to_json(self) -> dict[str, object]:
        row: dict[str, object] = {
            "adb_port": self.adb_port, "backend": self.backend, "bundle_directory": self.bundle_directory,
            "column_quantum": self.column_quantum, "device_id": self.device_id, "forward_port": self.forward_port,
            "kernel_release": self.kernel_release, "kind": self.kind, "max_tokens": self.max_tokens,
            "memory_total_bytes": self.memory_total_bytes, "minimum_usb_speed_mbps": self.minimum_usb_speed_mbps,
            "model": self.model, "root": self.root, "serial": self.serial, "soc": self.soc,
            "transport": self.transport, "usb_speed_mbps": self.usb_speed_mbps,
            "usb_sysfs_device": self.usb_sysfs_device, "work_directory": self.work_directory,
            "worker_port": self.worker_port,
        }
        if self.worker_environment:
            row["worker_environment"] = dict(self.worker_environment)
        if self.sessions != 1:
            row["sessions"] = self.sessions
        if self.session_limit_bytes:
            row["session_limit_bytes"] = self.session_limit_bytes
        if self.as_root:
            row["as_root"] = True
        if self.power is not None:
            row["power"] = self.power.to_json()
        if self.functionfs is not None:
            row["functionfs"] = dict(self.functionfs)
        return dict(sorted(row.items()))


@dataclass(frozen=True)
class SitePaths:
    repo_root: Path
    server_path: Path
    adb_path: Path
    work_root: Path                      # run directories, sliced shards, the site profile
    models: Mapping[str, Path]           # model id -> GGUF
    host_tools: Mapping[str, Path] = MappingProxyType({})   # e.g. usb_bridge, usb_close

    @classmethod
    def from_json(cls, value: object, *, base: Path) -> "SitePaths":
        row = _object(value, "site paths")
        _keys(row, ("repo_root", "server_path", "adb_path", "work_root", "models", "host_tools"), "site paths")
        tools = row.get("host_tools")
        return cls(
            repo_root=_path(row.get("repo_root"), base, "site repository root"),
            server_path=_path(row.get("server_path"), base, "site server path"),
            adb_path=_path(row.get("adb_path"), base, "site adb path"),
            work_root=_path(row.get("work_root"), base, "site work root"),
            models=_path_map(row.get("models"), base, "site models"),
            host_tools=MappingProxyType({}) if not tools else _path_map(tools, base, "site host tools"),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "adb_path": _path_json(self.adb_path),
            **({"host_tools": {name: _path_json(path) for name, path in self.host_tools.items()}}
               if self.host_tools else {}),
            "models": {name: _path_json(path) for name, path in self.models.items()},
            "repo_root": _path_json(self.repo_root),
            "server_path": _path_json(self.server_path),
            "work_root": _path_json(self.work_root),
        }


@dataclass(frozen=True)
class SiteManifest:
    site_id: str
    host: SiteHost
    phones: tuple[SitePhone, ...]
    paths: SitePaths

    @classmethod
    def from_json(cls, value: object, *, base: Path) -> "SiteManifest":
        row = _object(value, "site manifest")
        _require(row.get("schema") == SITE_MANIFEST_SCHEMA, "site manifest schema differs")
        _keys(row, ("schema", "site_id", "host", "phones", "paths"), "site manifest")
        phones = tuple(SitePhone.from_json(item) for item in _sequence(row.get("phones", []), "site phones"))
        for name, values in (("device ids", [phone.device_id for phone in phones]),
                             ("serials", [phone.serial for phone in phones]),
                             ("forward ports", [phone.forward_port for phone in phones])):
            _require(len(set(values)) == len(values), "site phone " + name + " are not unique")
        return cls(_text(row.get("site_id"), "site id"), SiteHost.from_json(row.get("host")), phones,
                   SitePaths.from_json(row.get("paths"), base=base))

    @classmethod
    def load(cls, path: Path) -> "SiteManifest":
        path = Path(path).resolve()
        try:
            value = json.loads(path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SchedulerConfigurationError("site manifest cannot be loaded: " + str(error)) from error
        return cls.from_json(value, base=path.parent)

    def to_json(self) -> dict[str, object]:
        return {"schema": SITE_MANIFEST_SCHEMA, "site_id": self.site_id, "host": self.host.to_json(),
                "phones": [phone.to_json() for phone in self.phones], "paths": self.paths.to_json()}

    def phone(self, device_id: str) -> SitePhone:
        for phone in self.phones:
            if phone.device_id == device_id:
                return phone
        raise SchedulerConfigurationError("site has no phone " + str(device_id))
