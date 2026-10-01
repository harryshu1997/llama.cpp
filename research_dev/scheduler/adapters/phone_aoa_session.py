"""Pixel FFN worker reached over a direct USB (Android Open Accessory) bridge instead of ``adb forward``.

Opt-in transport ``aoa-bridge`` of a static co-helper phone (rig ``helper_phones[].transport``; WS10, design in
``~/.cache/claude-work/ws10-pixel-usb/DESIGN.md``). The worker binary, its environment, its launch under the phone
lock and its lifecycle are exactly those of :class:`AdbTcpPhoneWorkerSession`; only the host endpoint differs:

* the Pixel is switched into accessory+adb mode (18d1:2d01, adb keeps working) before anything is launched,
  selected strictly by its sysfs port path and serial (never by VID:PID: the OP15 gadget is 18d1:2d00);
* a phone relay (``native/aoa_bridge/s43_aoa_relay.c``, rooted, under its own lock) reads ``/dev/usb_accessory``
  and connects each stream to the worker's phone-local port;
* a host bridge process (``adapters/aoa_bridge.py serve``) listens on the helper's fixed ``forward_port`` -- the
  port llama-server already dials -- and carries every TCP connection over the accessory bulk endpoints.

llama-server is unchanged (it still sees ``S41_SERVER_FFN_TRANSPORT=tcp`` to 127.0.0.1:forward_port). The
co-helper transport parameters gain ``ffn_link_transport: "aoa-bridge"`` so plans and receipts show the link.

Liveness adds the bridge process, its status file and the relay; a bridge or relay exit is a lost helper (the
elastic join restarts all three). Every signal is SIGTERM, never SIGKILL; nothing persistent is changed on the
phone or the host (keep-awake measures are process-scoped and end with the relay / bridge).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shlex
import socket
import subprocess
import time
from types import MappingProxyType
from typing import Callable, Mapping

from .._internal.types import canonical_sha256
from . import aoa_bridge
from .contracts import PhysicalAdapterError
from .phone_tcp_session import AdbTcpPhoneWorkerSession, AdbTcpWorkerConfiguration, AdbTcpWorkerReceipt, _phone_path


AOA_LINK_TRANSPORT = "aoa-bridge"
RELAY_READY_MARKER = "[s43-aoa-relay] ready "
DEFAULT_RELAY_LOCK = "/data/local/tmp/.s43-pixel-aoa-relay.lock"
BRIDGE_SCRIPT = Path(aoa_bridge.__file__).resolve()
# option -> (low, high); every option is optional, absent = the program's default (keep-awake off)
RELAY_OPTION_RANGES = MappingProxyType({
    "uclamp_min": (0, 1024), "fifo_priority": (1, 99), "qos_latency_us": (0, 1_000_000),
    "qos_window_ms": (0, 3_600_000),
})
RELAY_TEXT_OPTIONS = ("cpus",)  # hexadecimal CPU mask
BRIDGE_OPTION_RANGES = MappingProxyType({
    "keepalive_ms": (0, 1000), "keepalive_window_ms": (0, 600_000), "ping_interval_ms": (100, 600_000),
    "ping_timeout_ms": (500, 600_000), "handshake_timeout_ms": (100, 600_000), "status_interval_ms": (100, 600_000),
})
CONFIGURATION_KEYS = frozenset({
    "usb_sysfs_device", "relay_path", "relay_sha256", "bridge_script_sha256", "relay_lock_path",
    "forbidden_serials", "relay_options", "bridge_options", "python_path", "mode_switch_timeout_s",
    "ready_timeout_s", "trace",
})


def _sha256(name: str, value: object) -> str:
    if type(value) is not str or len(value) != 71 or not value.startswith("sha256:") \
            or any(character not in "0123456789abcdef" for character in value[7:]):
        raise PhysicalAdapterError(name + " must be sha256:<64 hex>")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _options(name: str, value: object, ranges: Mapping[str, tuple], text_keys: tuple[str, ...] = ()) -> dict:
    if type(value) is not dict:
        raise PhysicalAdapterError(name + " must be an object")
    result = {}
    for key, item in value.items():
        if key in text_keys:
            if type(item) is not str or not item or any(c not in "0123456789abcdefABCDEF" for c in item) \
                    or not 0 < int(item, 16) <= 0xFFFFFFFF:
                raise PhysicalAdapterError(f"{name}.{key} must be a nonzero hexadecimal CPU mask")
            result[key] = item.lower()
            continue
        if key not in ranges:
            raise PhysicalAdapterError(f"{name} has an unknown option {key!r}")
        low, high = ranges[key]
        if type(item) not in (int, float) or type(item) is bool or not low <= item <= high:
            raise PhysicalAdapterError(f"{name}.{key} must be within [{low}, {high}]")
        if key in RELAY_OPTION_RANGES and type(item) is not int:
            raise PhysicalAdapterError(f"{name}.{key} must be an integer")
        result[key] = item
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class AoaBridgeConfiguration:
    """Everything the bridge and relay of one helper run with; hashed into the transport identity."""

    usb_sysfs_device: str
    relay_path: str
    relay_sha256: str
    bridge_script_sha256: str
    relay_lock_path: str | None = DEFAULT_RELAY_LOCK
    forbidden_serials: tuple[str, ...] = aoa_bridge.DEFAULT_FORBIDDEN_SERIALS
    relay_options: Mapping[str, int | str] = field(default_factory=dict)
    bridge_options: Mapping[str, int | float] = field(default_factory=dict)
    python_path: str = "/usr/bin/python3"
    mode_switch_timeout_s: float = 40.0
    ready_timeout_s: float = 60.0
    trace: bool = False

    def __post_init__(self) -> None:
        if type(self.usb_sysfs_device) is not str or not aoa_bridge._valid_sysfs_device(self.usb_sysfs_device):
            raise PhysicalAdapterError("AOA bridge USB sysfs device must look like 2-9.2")
        _phone_path("AOA relay path", self.relay_path)
        if self.relay_lock_path is not None:
            _phone_path("AOA relay lock", self.relay_lock_path)
        _sha256("AOA relay", self.relay_sha256)
        _sha256("AOA bridge script", self.bridge_script_sha256)
        serials = tuple(self.forbidden_serials)
        if any(type(value) is not str or not value or not value.isascii() or " " in value for value in serials):
            raise PhysicalAdapterError("AOA bridge forbidden serials are invalid")
        if type(self.python_path) is not str or not self.python_path.startswith("/"):
            raise PhysicalAdapterError("AOA bridge python path must be absolute")
        for name in ("mode_switch_timeout_s", "ready_timeout_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or type(value) is bool or not 1 <= value <= 3600:
                raise PhysicalAdapterError("AOA bridge " + name + " is invalid")
        if type(self.trace) is not bool:
            raise PhysicalAdapterError("AOA bridge trace must be boolean")
        relay = _options("AOA relay options", dict(self.relay_options), RELAY_OPTION_RANGES, RELAY_TEXT_OPTIONS)
        bridge = _options("AOA bridge options", dict(self.bridge_options), BRIDGE_OPTION_RANGES)
        object.__setattr__(self, "forbidden_serials", serials)
        object.__setattr__(self, "relay_options", MappingProxyType(relay))
        object.__setattr__(self, "bridge_options", MappingProxyType(bridge))

    @classmethod
    def from_json(cls, value: object) -> "AoaBridgeConfiguration":
        if type(value) is not dict or set(value) - CONFIGURATION_KEYS:
            raise PhysicalAdapterError("AOA bridge configuration is invalid: "
                                       + ",".join(sorted(set(value) - CONFIGURATION_KEYS)
                                                  if type(value) is dict else ["not an object"]))
        raw = dict(value)
        if "forbidden_serials" in raw:
            if type(raw["forbidden_serials"]) is not list:
                raise PhysicalAdapterError("AOA bridge forbidden serials must be a list")
            raw["forbidden_serials"] = tuple(raw["forbidden_serials"])
        try:
            return cls(**raw)
        except TypeError as error:
            raise PhysicalAdapterError("AOA bridge configuration is incomplete: " + str(error)) from error

    def to_json(self) -> dict[str, object]:
        return {
            "bridge_options": dict(self.bridge_options), "bridge_script_sha256": self.bridge_script_sha256,
            "forbidden_serials": list(self.forbidden_serials), "mode_switch_timeout_s": self.mode_switch_timeout_s,
            "python_path": self.python_path, "ready_timeout_s": self.ready_timeout_s,
            "relay_lock_path": self.relay_lock_path, "relay_options": dict(self.relay_options),
            "relay_path": self.relay_path, "relay_sha256": self.relay_sha256, "trace": self.trace,
            "usb_sysfs_device": self.usb_sysfs_device,
        }

    @property
    def options_sha256(self) -> str:
        """Keep-awake and liveness settings: part of what a qualification measured."""
        return canonical_sha256({"bridge_options": dict(self.bridge_options),
                                 "relay_options": dict(self.relay_options)})

    def relay_argv(self, worker_port: int) -> tuple[str, ...]:
        arguments = [self.relay_path, "--worker-port", str(worker_port)]
        for key, value in self.relay_options.items():
            arguments.extend(("--" + key.replace("_", "-"), str(value)))
        return tuple(arguments)

    def bridge_argv(self, script: Path, serial: str, listen_port: int, ready: Path, status: Path,
                    trace: Path | None) -> tuple[str, ...]:
        arguments = [self.python_path, str(script), "serve", "--sysfs-device", self.usb_sysfs_device,
                     "--serial", serial, "--listen-port", str(listen_port), "--ready-file", str(ready),
                     "--status-file", str(status), "--shutdown-relay-on-exit"]
        for serial_text in self.forbidden_serials:
            arguments.extend(("--forbid-serial", serial_text))
        for key, value in self.bridge_options.items():
            arguments.extend(("--" + key.replace("_", "-"), str(value)))
        if trace is not None:
            arguments.extend(("--trace-file", str(trace)))
        return tuple(arguments)


class AoaBridgePhoneWorkerSession(AdbTcpPhoneWorkerSession):
    """:class:`AdbTcpPhoneWorkerSession` whose host endpoint is the AOA bridge + phone relay.

    Injectable for tests: ``observe_usb(sysfs_device) -> aoa_bridge.UsbDeviceState``, ``switch_mode(...)``,
    ``bridge_script`` (default: this package's ``aoa_bridge.py``), ``extra_bridge_arguments`` /
    ``extra_relay_arguments`` (loopback links), ``wall_clock``."""

    def __init__(
        self,
        configuration: AdbTcpWorkerConfiguration,
        bridge: AoaBridgeConfiguration,
        *,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        connect: Callable[[int], socket.socket] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        observe_usb: Callable[[str], "aoa_bridge.UsbDeviceState"] | None = None,
        switch_mode: Callable[..., Mapping[str, object]] | None = None,
        bridge_script: Path | None = None,
        extra_bridge_arguments: tuple[str, ...] = (),
        extra_relay_arguments: tuple[str, ...] = (),
    ) -> None:
        super().__init__(configuration, run=run, popen=popen, connect=connect, sleep=sleep, clock=clock)
        if not isinstance(bridge, AoaBridgeConfiguration):
            raise PhysicalAdapterError("AOA bridge configuration is invalid")
        if configuration.forward_port <= 0:
            raise PhysicalAdapterError("the AOA bridge needs a fixed host port (forward_port)")
        if configuration.serial in bridge.forbidden_serials:
            raise PhysicalAdapterError("the AOA bridge helper serial is also a forbidden serial")
        self.bridge = bridge
        self._wall = wall_clock
        self._observe_usb = observe_usb or (lambda device: aoa_bridge.read_usb_device(device))
        self._switch_mode = switch_mode or aoa_bridge.switch_to_accessory
        self._bridge_script = Path(bridge_script) if bridge_script is not None else BRIDGE_SCRIPT
        self._extra_bridge = tuple(extra_bridge_arguments)
        self._extra_relay = tuple(extra_relay_arguments)
        self._relay_process: subprocess.Popen | None = None
        self._bridge_process: subprocess.Popen | None = None
        self._paths: dict[str, Path] = {}
        self._mode: Mapping[str, object] = {}
        self._relay_pids: tuple[int, ...] = ()
        self._last_relay_launch: tuple[str, tuple[int, ...]] | None = None
        self._stop_record: dict[str, object] = {}

    # --- phone side ---------------------------------------------------------------------------

    def relay_shell_command(self) -> str:
        command = "exec " + shlex.join((*self.bridge.relay_argv(self.configuration.phone_port), *self._extra_relay))
        if self.bridge.relay_lock_path is not None:
            command = ("exec 9>" + shlex.quote(self.bridge.relay_lock_path)
                       + "\nflock -n 9 9>&9 || exit 73\n" + command)
        return "su -c " + shlex.quote(command) if self.configuration.as_root else command

    def _signal_pid(self, pid: int) -> None:
        self._shell(self._signal_command(pid))

    # --- preflight / mode ---------------------------------------------------------------------

    def _usb_state(self) -> "aoa_bridge.UsbDeviceState":
        try:
            state = self._observe_usb(self.bridge.usb_sysfs_device)
            aoa_bridge.check_pixel(state, serial=self.configuration.serial,
                                   forbidden_serials=self.bridge.forbidden_serials, require_accessory=None)
        except aoa_bridge.BridgeError as error:
            raise PhysicalAdapterError("AOA bridge USB selection refused: " + str(error)) from error
        return state

    def preflight(self) -> AdbTcpWorkerReceipt:
        """The adb-tcp preflight plus: no relay runs, relay and host bridge match their pins, the host port is
        free and the pinned port holds the pinned Pixel (normal or accessory+adb, never accessory-only)."""
        receipt = super().preflight()
        try:
            return self._aoa_preflight(receipt)
        except BaseException:
            self._boot_id = None  # a refused preflight never licenses a start
            raise

    def _aoa_preflight(self, receipt: AdbTcpWorkerReceipt) -> AdbTcpWorkerReceipt:
        bridge = self.bridge
        if self._executable_pids(bridge.relay_path):
            raise PhysicalAdapterError(f"{self.configuration.serial} already runs {bridge.relay_path}")
        listing = self._shell(shlex.join(["sha256sum", bridge.relay_path]), timeout_s=120)
        observed = "sha256:" + (listing.split() or ["?"])[0]
        if observed != bridge.relay_sha256:
            raise PhysicalAdapterError("the phone relay differs from its pinned sha256")
        script = file_sha256(self._bridge_script)
        if script != bridge.bridge_script_sha256:
            raise PhysicalAdapterError("the host bridge script differs from its pinned sha256")
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", self.configuration.forward_port))
            except OSError as error:
                raise PhysicalAdapterError(f"host port {self.configuration.forward_port} is occupied") from error
        state = self._usb_state()
        details = dict(receipt.details)
        details["aoa_bridge"] = {"relay_sha256": observed, "bridge_script_sha256": script, "relay_absent": True,
                                 "host_port_free": True, "usb": state.to_json(),
                                 "options_sha256": bridge.options_sha256}
        return AdbTcpWorkerReceipt("preflight", receipt.device_id, receipt.serial, receipt.boot_id,
                                   MappingProxyType(details))

    def _ensure_accessory_mode(self) -> Mapping[str, object]:
        """Accessory+adb before anything runs over adb: the switch re-enumerates the USB device."""
        state = self._usb_state()
        if state.product == aoa_bridge.ACCESSORY_ADB_PRODUCT:
            return {"switched": False, "usb": state.to_json()}
        try:
            result = dict(self._switch_mode(self.bridge.usb_sysfs_device, self.configuration.serial,
                                            forbidden_serials=self.bridge.forbidden_serials,
                                            timeout_s=self.bridge.mode_switch_timeout_s))
        except aoa_bridge.BridgeError as error:
            raise PhysicalAdapterError("AOA accessory switch failed: " + str(error)) from error
        deadline = self._clock() + self.bridge.mode_switch_timeout_s
        while True:  # adb returns once the accessory+adb composite has enumerated
            try:
                state_text = self._run(self._adb("get-state"), check=False, capture_output=True, text=True,
                                       stdin=subprocess.DEVNULL, timeout=10).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                state_text = ""
            if state_text == "device":
                break
            if self._clock() > deadline:
                raise PhysicalAdapterError("adb did not return after the AOA accessory switch")
            self._sleep(0.25)
        return {"switched": True, **result}

    def start(self, log_path: Path, *, release_failed_start: bool = False) -> AdbTcpWorkerReceipt:
        if self.active:
            raise PhysicalAdapterError("adb-tcp worker session is already active")
        if self._boot_id is None:
            self.preflight()
        self._mode = self._ensure_accessory_mode()
        return super().start(log_path, release_failed_start=release_failed_start)

    # --- host endpoint ------------------------------------------------------------------------

    def _wait_for(self, predicate: Callable[[], bool], process: subprocess.Popen, log: Path, what: str) -> None:
        deadline = self._clock() + self.bridge.ready_timeout_s
        while not predicate():
            if process.poll() is not None or self._clock() > deadline:
                text = log.read_text(encoding="ascii", errors="backslashreplace") if log.exists() else ""
                raise PhysicalAdapterError(f"{what} did not become ready: " + text[-2000:])
            self._sleep(0.1)

    def _open_host_endpoint(self, log_path: Path) -> int:
        configuration, bridge = self.configuration, self.bridge
        stem = log_path.with_suffix("")
        self._paths = {name: stem.with_name(stem.name + suffix) for name, suffix in (
            ("relay_log", "-relay.log"), ("bridge_log", "-bridge.log"), ("ready", "-bridge-ready.json"),
            ("status", "-bridge-status.json"), ("trace", "-bridge-trace.jsonl"))}
        self._stop_record = {}
        relay_log = self._paths["relay_log"]
        command = self._adb("shell", "-T", self.relay_shell_command())
        with relay_log.open("x", encoding="ascii", errors="backslashreplace") as log:
            self._relay_process = self._popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            self._wait_for(lambda: RELAY_READY_MARKER in relay_log.read_text(encoding="ascii", errors="replace"),
                           self._relay_process, relay_log, "AOA relay")
            self._relay_pids = tuple(self._executable_pids(bridge.relay_path))
            self._last_relay_launch = (self._boot_id or "", self._relay_pids)
            argv = (*bridge.bridge_argv(self._bridge_script, configuration.serial, configuration.forward_port,
                                        self._paths["ready"], self._paths["status"],
                                        self._paths["trace"] if bridge.trace else None), *self._extra_bridge)
            with self._paths["bridge_log"].open("x", encoding="ascii", errors="backslashreplace") as log:
                self._bridge_process = self._popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                                   stderr=subprocess.STDOUT)
            self._wait_for(self._paths["ready"].exists, self._bridge_process, self._paths["bridge_log"], "AOA bridge")
            ready = json.loads(self._paths["ready"].read_text())
            if ready.get("listen_port") != configuration.forward_port:
                raise PhysicalAdapterError("the AOA bridge listens on another port")
        except BaseException:
            self._stop_bridge_and_relay([], clean=False)
            raise
        return configuration.forward_port

    def _bridge_status(self) -> Mapping[str, object] | None:
        path = self._paths.get("status")
        try:
            return json.loads(path.read_text()) if path is not None else None
        except (OSError, ValueError):
            return None

    def _host_endpoint_alive(self, host_port: int) -> bool:
        bridge, relay = self._bridge_process, self._relay_process
        if bridge is None or relay is None or bridge.poll() is not None or relay.poll() is not None:
            return False
        if not self._executable_pids(self.bridge.relay_path):
            return False
        status = self._bridge_status()
        interval_s = float(self.bridge.bridge_options.get("status_interval_ms", 2000)) / 1000
        return (status is not None and status.get("state") == "up"
                and self._wall() - float(status.get("updated_epoch_s", 0)) <= max(10.0, 5 * interval_s))

    def client_exited(self) -> bool:
        """The worker's adb client, the relay's adb client or the bridge exited: the helper is gone for good."""
        if super().client_exited():
            return True
        return any(process is not None and process.poll() is not None
                   for process in (self._relay_process, self._bridge_process))

    def _wait_local(self, process: subprocess.Popen | None, timeout_s: float) -> int | None:
        if process is None:
            return None
        deadline = self._clock() + timeout_s
        while process.poll() is None and self._clock() < deadline:
            self._sleep(0.1)
        return process.poll()

    def _terminate_local(self, process: subprocess.Popen | None, errors: list[str], name: str) -> int | None:
        if process is None:
            return None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                errors.append(name + " did not exit")
        return process.poll()

    def _stop_bridge_and_relay(self, errors: list[str], *, clean: bool) -> bool:
        """SIGTERM the bridge (it sends SHUTDOWN, so the relay exits 0 and its adb client ends), then make sure
        the relay is gone: still-running relay pids on the phone are SIGTERMed (never SIGKILL) before the
        local adb client is ended. Returns True when the bridge exited and no relay remains."""
        bridge_process, relay_process = self._bridge_process, self._relay_process
        if bridge_process is not None and bridge_process.poll() is None:
            bridge_process.terminate()
        bridge_code = self._wait_local(bridge_process, 15.0)
        if bridge_process is not None and bridge_code is None:
            bridge_code = self._terminate_local(bridge_process, errors, "AOA bridge")
        relay_code = self._wait_local(relay_process, 10.0 if clean else 3.0)
        signalled: list[int] = []
        remaining: list[int] = []
        try:
            remaining = self._executable_pids(self.bridge.relay_path)
            for pid in remaining:
                self._signal_pid(pid)
                signalled.append(pid)
            deadline = self._clock() + 10.0
            while remaining and self._clock() < deadline:
                self._sleep(0.25)
                remaining = self._executable_pids(self.bridge.relay_path)
        except (OSError, subprocess.SubprocessError, PhysicalAdapterError) as error:
            errors.append("relay: " + type(error).__name__ + ": " + str(error))
        if relay_code is None:
            relay_code = self._wait_local(relay_process, 5.0)
            if relay_process is not None and relay_code is None:
                relay_code = self._terminate_local(relay_process, errors, "AOA relay adb client")
        status = self._bridge_status()
        self._stop_record = {
            "bridge_exit_code": bridge_code, "relay_client_exit_code": relay_code,
            "relay_signalled_pids": signalled, "relay_pids_after": remaining,
            "bridge_final_status": {key: status.get(key) for key in (
                "state", "exit_code", "exit_reason", "counters", "hello", "pong_rtt_us_p50")} if status else None,
        }
        self._bridge_process = None
        self._relay_process = None
        return bridge_code is not None and not remaining

    def _release_host_endpoint(self, host_port: int, errors: list[str]) -> bool:
        return self._stop_bridge_and_relay(errors, clean=False)

    def _close_host_endpoint(self, host_port: int) -> None:
        errors: list[str] = []
        gone = self._stop_bridge_and_relay(errors, clean=True)
        if not gone or errors or self._stop_record.get("bridge_exit_code") != 0:
            raise PhysicalAdapterError("AOA bridge stop failed: " + repr({"errors": errors, **self._stop_record}))

    def _endpoint_details(self) -> Mapping[str, object]:
        details: dict[str, object] = {
            "mode": dict(self._mode), "options_sha256": self.bridge.options_sha256,
            "relay_pids": list(self._relay_pids),
            "bridge_pid": self._bridge_process.pid if self._bridge_process is not None else None,
            "paths": {name: str(path) for name, path in sorted(self._paths.items())},
        }
        if self._stop_record:
            details["stop"] = dict(self._stop_record)
        return {"host_endpoint": AOA_LINK_TRANSPORT, "aoa_bridge": details}

    def release_orphaned_worker(self, *, timeout_s: float = 10.0) -> AdbTcpWorkerReceipt | None:
        """Also SIGTERM a relay this session launched that outlived it (same boot, recorded pids only)."""
        receipt = super().release_orphaned_worker(timeout_s=timeout_s)
        if self.active or self._last_relay_launch is None:
            return receipt
        boot_id, launched = self._last_relay_launch
        pids = self._executable_pids(self.bridge.relay_path) if launched and self._boot() == boot_id else []
        if not pids or not set(pids) <= set(launched):
            return receipt
        for pid in pids:
            self._signal_pid(pid)
        deadline = self._clock() + timeout_s
        remaining = self._executable_pids(self.bridge.relay_path)
        while remaining and self._clock() < deadline:
            self._sleep(0.25)
            remaining = self._executable_pids(self.bridge.relay_path)
        if remaining:
            raise PhysicalAdapterError("orphaned AOA relay did not exit after SIGTERM (never killed)")
        base = dict(receipt.details) if receipt is not None else {"signalled_pids": [], "worker_pids_after": []}
        return AdbTcpWorkerReceipt("release_orphan", self.configuration.device_id, self.configuration.serial, boot_id,
                                   MappingProxyType({**base, "relay_signalled_pids": pids, "relay_pids_after": []}))

    def transport_parameters(self) -> Mapping[str, int | str]:
        return MappingProxyType({**super().transport_parameters(), "ffn_link_transport": AOA_LINK_TRANSPORT})

