"""Phone driver for the direct USB bridge transport (``functionfs-bridge``): one or several NPU sessions.

A thin layer over ``adapters/phone_functionfs_bridge_session.py``. The site row supplies what is specific
to the phone (bundle directory, ports, the ``functionfs`` section with the session scripts and their
pins); a ``HelperBinding`` supplies the model and the shards. One shard = the single-session launch;
several shards = one session each through the bundle's resident workers and router.

Every adb call of the session goes through a guard that refuses when no adb server listens: a session
that lost its server fails instead of starting one under this account.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable, Mapping, Sequence

from ..adapters.contracts import PhysicalAdapterError
from ..adapters.phone_functionfs_bridge_session import FunctionFsBridgePhoneWorkerSession
from ..adapters.phone_tcp_session import AdbTcpWorkerConfiguration
from .._internal.layer_placement import layer_spec
from ..configuration.site import SitePaths, SitePhone
from .adb import AdbGuard, AdbUnavailable
from .meter import PhonePowerCalibration
from .phone import (
    RESIDENT_ROUTER_NAME,
    RESIDENT_WORKERS_NAME,
    WORKER_NAME,
    HelperBinding,
    HelperEndpoint,
    PhoneHealth,
    allowed_phone_roots,
    read_phone_health,
    register_phone_driver,
)


TRANSPORT = "functionfs-bridge"
MULTI_SESSION_PORT_OFFSET = 200


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("ascii")).hexdigest()


def session_manifest_rows(binding: HelperBinding, port_base: int) -> str:
    """The resident router's manifest: one row of twelve fields per session, joined by ``;``.

    ``session, session, parent sha256, shard path, layers, columns, port, endpoint id, bytes, geometry id,
    plan id, generation`` -- the row format of ``adapters/phone_session_ops/weights.py``; the three ids
    are digests of this binding's own description.
    """
    plan = _digest(json.dumps(binding.to_json(), sort_keys=True))
    rows = []
    for index, shard in enumerate(binding.shards):
        spec = layer_spec(shard.layers)
        rows.append(",".join((
            shard.session_id, shard.session_id, binding.artifact_sha256, shard.remote_path, spec,
            str(shard.columns), str(port_base + index), _digest("portable-endpoint:" + shard.session_id),
            str(shard.bytes), _digest(f"portable-geometry:{spec}:{shard.columns}"), plan, "1")))
    return ";".join(rows)


class FunctionFsPhoneDriver:
    def __init__(self, phone: SitePhone, paths: SitePaths, adb: AdbGuard, *,
                 session_factory: Callable = FunctionFsBridgePhoneWorkerSession,
                 run: Callable = subprocess.run) -> None:
        if phone.transport != TRANSPORT or phone.functionfs is None:
            raise PhysicalAdapterError("phone is not on the direct USB transport")
        self.phone, self.paths, self.adb = phone, paths, adb
        self._session_factory, self._run = session_factory, run
        self.session = None
        self.receipts: dict[str, object] = {}

    # -- read-only -------------------------------------------------------------------------------
    def describe(self) -> Mapping[str, object]:
        return self.phone.to_json()

    def health(self) -> PhoneHealth:
        return read_phone_health(self.adb, self.phone)

    def power_calibration(self) -> PhonePowerCalibration | None:
        power = self.phone.power
        if power is None or not power.measured:
            return None
        return PhonePowerCalibration(power.battery_current_unit, power.battery_discharge_sign, power.gauge_ratio,
                                     power.gauge_ratio_provenance)

    def pins(self, binding: HelperBinding) -> dict[str, str]:
        return {**dict(binding.pins), **{shard.remote_path: shard.sha256 for shard in binding.shards}}

    def verify(self, binding: HelperBinding) -> Mapping[str, str]:
        """sha256 of every pinned phone file; raises when one differs or is absent."""
        pins = self.pins(binding)
        observed = self.adb.remote_sha256(self.phone.serial, sorted(pins), root=self.phone.as_root)
        wrong = sorted(path for path, digest in pins.items() if observed.get(path) != digest)
        if wrong:
            raise PhysicalAdapterError("phone files differ from their pinned sha256: " + ", ".join(wrong))
        return observed

    def push(self, local: Path, remote: str, sha256: str) -> str:
        return self.adb.push_verified(self.phone.serial, local, remote, sha256,
                                      allowed_roots=allowed_phone_roots(self.phone), root=self.phone.as_root)

    # -- the session -----------------------------------------------------------------------------
    def _guarded_run(self, command: Sequence[str], **keywords):
        if command and str(command[0]) == str(self.paths.adb_path) and not self.adb.available():
            raise AdbUnavailable(f"the adb server on port {self.adb.port} is gone; none is started here")
        return self._run(command, **keywords)

    def worker_configuration(self, binding: HelperBinding) -> AdbTcpWorkerConfiguration:
        phone, bundle = self.phone, self.phone.bundle_directory
        worker_path = bundle + "/" + WORKER_NAME
        pins = self.pins(binding)
        if worker_path not in pins:
            raise PhysicalAdapterError("the helper binding does not pin the worker binary")
        if binding.stored_columns % phone.column_quantum:
            raise PhysicalAdapterError("stored columns are not a multiple of the phone's column quantum")
        fixed = dict(FunctionFsBridgePhoneWorkerSession.SCRIPT_FIXED_ENVIRONMENT, ADSP_LIBRARY_PATH=bundle)
        return AdbTcpWorkerConfiguration(
            device_id=phone.device_id, serial=phone.serial, adb_port=phone.adb_port, adb_path=self.paths.adb_path,
            worker_path=worker_path, library_directories=(bundle,), shard_path=binding.shards[0].remote_path,
            artifact_sha256=binding.artifact_sha256, layer_mask=binding.layer_mask, n_embd=binding.n_embd,
            columns=binding.stored_columns, column_quantum=phone.column_quantum, max_tokens=phone.max_tokens,
            swiglu=binding.activation == "swiglu", backend=phone.backend, phone_port=phone.worker_port,
            forward_port=phone.forward_port, max_requests=0,       # the bridge ends the session
            worker_environment={**fixed, **dict(phone.worker_environment)}, expected_sha256_by_path=pins,
            as_root=phone.as_root)

    def multi_session(self, binding: HelperBinding) -> dict[str, str] | None:
        if len(binding.shards) == 1:
            return None
        phone = self.phone
        if len(binding.shards) > phone.sessions:
            raise PhysicalAdapterError(f"{len(binding.shards)} shards need more sessions than the site's {phone.sessions}")
        base = int(phone.functionfs.get("multi_session_port_base", phone.worker_port + MULTI_SESSION_PORT_OFFSET))
        return {"resident_workers": phone.bundle_directory + "/" + RESIDENT_WORKERS_NAME,
                "resident_router": phone.bundle_directory + "/" + RESIDENT_ROUTER_NAME,
                "rows": session_manifest_rows(binding, base)}

    def _adb_state(self) -> tuple[int, str]:
        """(exit status, state) of the phone on the listening adb server; never starts a server."""
        try:
            done = self.adb.run(self.phone.serial, "get-state", timeout_s=10, check=False)
        except AdbUnavailable as error:
            return 1, str(error)
        return done.returncode, (done.stdout or "").strip()

    def _restored(self, timeout_s: float):
        from ..adapters import bridge
        return bridge.verify_android_usb_restored(
            serial=self.phone.serial, adb_port=self.phone.adb_port,
            minimum_speed_mbps=int(self.phone.functionfs.get("minimum_usb_speed_mbps", 5000)),
            timeout_s=timeout_s, adb_probe=self._adb_state)

    def build_session(self, binding: HelperBinding):
        functionfs = {key: value for key, value in self.phone.functionfs.items() if key != "multi_session_port_base"}
        return self._session_factory(self.worker_configuration(binding), functionfs, run=self._guarded_run,
                                     restored=self._restored, tolerate_launch_disconnect=True,
                                     multi_session=self.multi_session(binding))

    def launch_command(self, binding: HelperBinding, remote_root: str = "<session root>") -> str:
        """The phone-side launch command this binding would run (dry runs and tests)."""
        return self.build_session(binding)._remote_command(remote_root)

    def start(self, binding: HelperBinding, log_directory: Path) -> HelperEndpoint:
        if self.session is not None and self.session.active:
            raise PhysicalAdapterError("the phone already runs a session of this driver")
        if not self.adb.available():
            raise AdbUnavailable(f"no adb server listens on port {self.adb.port}")
        self.session = self.build_session(binding)
        self.receipts = {"preflight": self.session.preflight().to_json()}
        self.receipts["launch"] = self.session.start(Path(log_directory) / "helper-worker.log").to_json()
        self.receipts["launch_dropped"] = self.session.launch_dropped
        port = self.session.transport_contract().control_port
        return HelperEndpoint(
            server_environment={"S41_SERVER_FFN_HOST": "127.0.0.1", "S41_SERVER_FFN_TRANSPORT": "tcp",
                                "S41_SERVER_FFN_PORT": str(port),
                                "S41_SERVER_FFN_LAYER_MASK": str(binding.layer_mask)},
            layers=binding.layers, stored_columns=binding.stored_columns, transport=TRANSPORT)

    def alive(self) -> bool:
        return self.session is not None and self.session.active

    def stop(self, *, served_calls: int) -> Mapping[str, object]:
        if self.session is None:
            raise PhysicalAdapterError("no phone session to stop")
        return self.session.stop(served_calls=served_calls).to_json()

    def release(self) -> Sequence[str]:
        """Failure path: end what is left of the session; returns what could not be cleaned."""
        if self.session is None or not self.session.active:
            return []
        return list(self.session.release(force_close=True))


register_phone_driver(TRANSPORT, FunctionFsPhoneDriver)
