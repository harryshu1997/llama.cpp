"""Phone driver for the generic transport (``adb-tcp``): any phone that answers adb.

One worker process on the phone, reached through ``adb forward``; no patched kernel, no USB mode change,
root optional. This is the path a phone the system has never run on takes on its first day: it needs the
worker bundle built for its backend and one shard in the site's work directory, nothing else. The link
costs more per call than the direct USB bridge; onboarding measures it and the share plan follows.

A thin layer over ``adapters/phone_tcp_session.py``; every adb call goes through the guard.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable, Mapping, Sequence

from ..adapters.contracts import PhysicalAdapterError
from ..adapters.phone_tcp_session import AdbTcpPhoneWorkerSession, AdbTcpWorkerConfiguration
from ..configuration.site import SitePaths, SitePhone
from .adb import AdbGuard, AdbUnavailable
from .meter import PhonePowerCalibration
from .phone import (
    WORKER_NAME,
    HelperBinding,
    HelperEndpoint,
    PhoneHealth,
    allowed_phone_roots,
    read_phone_health,
    register_phone_driver,
)


TRANSPORT = "adb-tcp"


class AdbTcpPhoneDriver:
    def __init__(self, phone: SitePhone, paths: SitePaths, adb: AdbGuard, *,
                 session_factory: Callable = AdbTcpPhoneWorkerSession, run: Callable = subprocess.run,
                 popen: Callable = subprocess.Popen) -> None:
        if phone.transport != TRANSPORT:
            raise PhysicalAdapterError("phone is not on the adb-tcp transport")
        self.phone, self.paths, self.adb = phone, paths, adb
        self._session_factory, self._run, self._popen = session_factory, run, popen
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
    def _guard(self, function: Callable) -> Callable:
        def guarded(command: Sequence[str], **keywords):
            if command and str(command[0]) == str(self.paths.adb_path) and not self.adb.available():
                raise AdbUnavailable(f"the adb server on port {self.adb.port} is gone; none is started here")
            return function(command, **keywords)
        return guarded

    def worker_configuration(self, binding: HelperBinding) -> AdbTcpWorkerConfiguration:
        phone, bundle = self.phone, self.phone.bundle_directory
        if len(binding.shards) != 1:
            raise PhysicalAdapterError("the adb-tcp transport runs one session: the binding must hold one shard")
        worker_path = bundle + "/" + WORKER_NAME
        pins = self.pins(binding)
        if worker_path not in pins:
            raise PhysicalAdapterError("the helper binding does not pin the worker binary")
        if binding.stored_columns % phone.column_quantum:
            raise PhysicalAdapterError("stored columns are not a multiple of the phone's column quantum")
        environment = dict(phone.worker_environment)
        if phone.kind == "htp":
            environment.setdefault("ADSP_LIBRARY_PATH", bundle)      # where the NPU finds its own library
        return AdbTcpWorkerConfiguration(
            device_id=phone.device_id, serial=phone.serial, adb_port=phone.adb_port, adb_path=self.paths.adb_path,
            worker_path=worker_path, library_directories=(bundle,), shard_path=binding.shards[0].remote_path,
            artifact_sha256=binding.artifact_sha256, layer_mask=binding.layer_mask, n_embd=binding.n_embd,
            columns=binding.stored_columns, column_quantum=phone.column_quantum, max_tokens=phone.max_tokens,
            swiglu=binding.activation == "swiglu", backend=phone.backend, phone_port=phone.worker_port,
            forward_port=phone.forward_port, max_requests=0,        # resident: ended by a signal while idle
            worker_environment=environment, expected_sha256_by_path=pins, as_root=phone.as_root)

    def build_session(self, binding: HelperBinding):
        return self._session_factory(self.worker_configuration(binding), run=self._guard(self._run),
                                     popen=self._guard(self._popen))

    def launch_command(self, binding: HelperBinding) -> str:
        """The phone-side worker command this binding would run (dry runs and tests)."""
        return self.worker_configuration(binding).worker_shell_command()

    def start(self, binding: HelperBinding, log_directory: Path) -> HelperEndpoint:
        if self.session is not None and self.session.active:
            raise PhysicalAdapterError("the phone already runs a session of this driver")
        if not self.adb.available():
            raise AdbUnavailable(f"no adb server listens on port {self.adb.port}")
        self.session = self.build_session(binding)
        self.receipts = {"preflight": self.session.preflight().to_json()}
        self.receipts["launch"] = self.session.start(Path(log_directory) / "helper-worker.log").to_json()
        self.receipts["launch_dropped"] = None
        environment = dict(self.session.transport_contract().server_environment())
        environment["S41_SERVER_FFN_LAYER_MASK"] = str(binding.layer_mask)
        return HelperEndpoint(server_environment=environment, layers=binding.layers,
                              stored_columns=binding.stored_columns, transport=TRANSPORT)

    def alive(self) -> bool:
        return self.session is not None and self.session.active

    def stop(self, *, served_calls: int) -> Mapping[str, object]:
        """End the idle worker with SIGTERM (never while a client is connected, never SIGKILL)."""
        if self.session is None:
            raise PhysicalAdapterError("no phone session to stop")
        return self.session.stop(served_calls=served_calls, allow_idle_signal=True, timeout_s=60.0).to_json()

    def release(self) -> Sequence[str]:
        """Failure path: end the worker this session started and remove its forward."""
        if self.session is None or not self.session.active:
            return []
        try:
            self.session.stop(allow_idle_signal=True, release_exited=True)
            return []
        except Exception as error:  # noqa: BLE001 - report what could not be cleaned
            return ["adb-tcp worker not released: " + str(error)[:300]]


register_phone_driver(TRANSPORT, AdbTcpPhoneDriver)
