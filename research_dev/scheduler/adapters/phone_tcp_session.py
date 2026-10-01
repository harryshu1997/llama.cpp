"""Direct protocol-v6 FFN worker on a helper phone, reached through ``adb forward``.

This is the lifecycle the Pixel 10 Pro server qualification used
(``reports/20260922-fast-path-M3/qualify_pixel_server.py``): the worker listens on a phone-local
TCP port, the host forwards a port to it with ``adb -s <serial> forward``, and llama-server
connects to the forward as a plain TCP FFN client. It never touches a USB gadget and needs
no root.

The worker has no TCP shutdown message. With a finite request budget (``max_requests > 0``)
:meth:`AdbTcpPhoneWorkerSession.stop` consumes the remaining budget with zero-input calls
outside any measurement, so the worker exits normally; this is the M3 practice and never
signals the worker. A resident worker (``max_requests == 0``) can only be stopped by a signal
while no client is connected, which the caller must request explicitly.

Elastic phones add a liveness check (:meth:`alive`), the release of a lost worker
(:meth:`release_lost`), of a failed start (``start(release_failed_start=True)``) and of a worker
that outlived its session (:meth:`release_orphaned_worker`), and an authorized fault injection;
every signal is SIGTERM, never SIGKILL.

Every subprocess and socket boundary is injectable; the unit tests drive it against a local
CPU worker through a fake ``adb``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shlex
import socket
import struct
import subprocess
import time
from types import MappingProxyType
from typing import Callable, Mapping

from .contracts import PhysicalAdapterError
from .phone_helpers import PhoneHelperBinding, layer_spec
from .phone_transport import PhoneTransportContract, phone_transport_contract


READY_MARKER = "[ffn-worker] ready backend="
CONNECTED_MARKER = "[ffn-worker] client connected"
DISCONNECTED_MARKER = "[ffn-worker] client disconnected"
PROTOCOL_MAGIC = 0x46534631
PROTOCOL_VERSION = 6
HELLO_REQUEST = struct.Struct("<IHHQIIHH32s4x")
HELLO_RESPONSE = struct.Struct("<IHHHHIIIIII4xQQIHH32s")
EXECUTE_REQUEST = struct.Struct("<IHHIiIIIII")
EXECUTE_RESPONSE = struct.Struct("<IHHHHIiIIIIIQ")
_TCP_LISTEN, _TCP_ESTABLISHED = "0A", "01"


def _fnv32(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def _idle_log(text: str) -> bool:
    """No client is connected: none ever was, or the last one disconnected."""
    return text.rfind(DISCONNECTED_MARKER) >= text.rfind(CONNECTED_MARKER)


def _phone_path(name: str, value: object) -> str:
    if type(value) is not str or not value.startswith("/") or not value.isascii() \
            or any(character.isspace() or character == ":" for character in value):
        raise PhysicalAdapterError(name + " must be an absolute phone path")
    return value


@dataclass(frozen=True)
class AdbTcpWorkerConfiguration:
    device_id: str
    serial: str
    adb_port: int
    adb_path: Path
    worker_path: str
    library_directories: tuple[str, ...]
    shard_path: str
    artifact_sha256: str
    layer_mask: int
    n_embd: int
    columns: int
    column_quantum: int
    max_tokens: int
    swiglu: bool
    backend: str
    phone_port: int
    forward_port: int = 0
    max_requests: int = 0
    worker_environment: Mapping[str, str] = field(default_factory=dict)
    # phone path -> sha256:... of the worker, its runtime libraries and the shard it opens
    expected_sha256_by_path: Mapping[str, str] = field(default_factory=dict)
    launch_timeout_s: float = 240.0
    as_root: bool = False
    phone_lock_path: str | None = None

    def __post_init__(self) -> None:
        if type(self.as_root) is not bool:
            raise PhysicalAdapterError("adb-tcp worker root mode is invalid")
        if self.phone_lock_path is not None:
            _phone_path("adb-tcp worker lock", self.phone_lock_path)
        for name in ("device_id", "serial", "backend"):
            value = getattr(self, name)
            if type(value) is not str or not value or not value.isascii() or " " in value:
                raise PhysicalAdapterError("adb-tcp worker " + name + " is invalid")
        if not isinstance(self.adb_path, Path):
            raise PhysicalAdapterError("adb-tcp worker adb path is invalid")
        _phone_path("adb-tcp worker path", self.worker_path)
        _phone_path("adb-tcp shard path", self.shard_path)
        directories = tuple(_phone_path("adb-tcp library directory", value) for value in self.library_directories)
        if not directories:
            raise PhysicalAdapterError("adb-tcp worker needs its runtime library directories")
        if type(self.artifact_sha256) is not str or len(self.artifact_sha256) != 71 \
                or not self.artifact_sha256.startswith("sha256:"):
            raise PhysicalAdapterError("adb-tcp worker artifact is invalid")
        integers = {
            "ADB port": (self.adb_port, 1, 65535), "phone port": (self.phone_port, 1, 65535),
            "forward port": (self.forward_port, 0, 65535), "n_embd": (self.n_embd, 1, 1 << 20),
            "columns": (self.columns, 32, 1 << 20), "column quantum": (self.column_quantum, 32, 1 << 20),
            "maximum tokens": (self.max_tokens, 1, 512), "request budget": (self.max_requests, 0, 1 << 40),
        }
        for name, (value, low, high) in integers.items():
            if type(value) is not int or not low <= value <= high:
                raise PhysicalAdapterError("adb-tcp worker " + name + " is invalid")
        if type(self.layer_mask) is not int or not 0 < self.layer_mask < 1 << 64 \
                or self.columns % self.column_quantum or type(self.swiglu) is not bool:
            raise PhysicalAdapterError("adb-tcp worker geometry is invalid")
        environment = dict(self.worker_environment)
        if "LD_LIBRARY_PATH" in environment or any(
            not key.isidentifier() or type(value) is not str or not value.isascii()
            or any(character.isspace() for character in value)
            for key, value in environment.items()
        ):
            raise PhysicalAdapterError("adb-tcp worker environment is invalid")
        hashes = {_phone_path("adb-tcp hashed file", path): value
                  for path, value in dict(self.expected_sha256_by_path).items()}
        if any(type(value) is not str or len(value) != 71 or not value.startswith("sha256:")
               for value in hashes.values()) or self.worker_path not in hashes or self.shard_path not in hashes:
            raise PhysicalAdapterError("adb-tcp worker and shard must be pinned by sha256")
        object.__setattr__(self, "library_directories", directories)
        object.__setattr__(self, "worker_environment", MappingProxyType(dict(sorted(environment.items()))))
        object.__setattr__(self, "expected_sha256_by_path", MappingProxyType(dict(sorted(hashes.items()))))

    def worker_command(self) -> tuple[str, ...]:
        """The qualified Pixel launch: ``env LD_LIBRARY_PATH=... K=V worker ...``."""
        return (
            "env", "LD_LIBRARY_PATH=" + ":".join(self.library_directories),
            *(f"{key}={value}" for key, value in self.worker_environment.items()),
            self.worker_path, "-m", self.shard_path, "--artifact-sha256", self.artifact_sha256,
            "--layers", layer_spec(self.layer_mask), "--columns", str(self.columns),
            "--column-quantum", str(self.column_quantum), "--backend", self.backend,
            "--port", str(self.phone_port), "--bind", "127.0.0.1", "--f16-io",
            "--max-tokens", str(self.max_tokens), "--max-requests", str(self.max_requests),
        )

    def worker_shell_command(self) -> str:
        command = "exec " + shlex.join(self.worker_command())
        if self.phone_lock_path is not None:
            command = ("exec 9>" + shlex.quote(self.phone_lock_path)
                       + "\nflock -n 9 9>&9 || exit 73\n" + command)
        return "su -c " + shlex.quote(command) if self.as_root else command


@dataclass(frozen=True)
class AdbTcpWorkerReceipt:
    """Launch or stop evidence of one helper worker."""

    phase: str
    device_id: str
    serial: str
    boot_id: str
    details: Mapping[str, object]

    def to_json(self) -> dict[str, object]:
        return {"boot_id": self.boot_id, "device_id": self.device_id, "phase": self.phase,
                "serial": self.serial, **dict(self.details)}


class AdbTcpPhoneWorkerSession:
    """One direct TCP FFN worker on one phone, owned by this process."""

    def __init__(
        self,
        configuration: AdbTcpWorkerConfiguration,
        *,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        connect: Callable[[int], socket.socket] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(configuration, AdbTcpWorkerConfiguration):
            raise PhysicalAdapterError("adb-tcp worker configuration is invalid")
        self.configuration = configuration
        self._run, self._popen, self._sleep, self._clock = run, popen, sleep, clock
        self._connect = connect or (lambda port: socket.create_connection(("127.0.0.1", port), timeout=60))
        self._process: subprocess.Popen | None = None
        self._log_path: Path | None = None
        self._boot_id: str | None = None
        self._host_port: int | None = None
        self._ready_line: str | None = None
        # (boot id, worker pids) of the last launch of this session, kept after it ends
        self._last_launch: tuple[str, tuple[int, ...]] | None = None
        self.fault_injections: list[dict[str, object]] = []

    # --- adb plumbing -------------------------------------------------------------------

    def _adb(self, *arguments: str) -> list[str]:
        return [str(self.configuration.adb_path), "-P", str(self.configuration.adb_port),
                "-s", self.configuration.serial, *arguments]

    def _shell(self, command: str, *, timeout_s: float = 60) -> str:
        return self._run(self._adb("shell", command), check=True, capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, timeout=timeout_s).stdout

    def _worker_pids(self) -> list[int]:
        return self._executable_pids(self.configuration.worker_path)

    def _executable_pids(self, path: str) -> list[int]:
        """Phone processes running exactly ``path`` (a rooted basename match is confirmed by its exe link)."""
        pids = []
        for line in self._shell("ps -A -o PID,ARGS").splitlines():
            fields = line.split(None, 1)
            if len(fields) != 2 or not fields[0].isdigit():
                continue
            executable = fields[1].split()[:1]
            if executable == [path]:
                pids.append(int(fields[0]))
            elif self.configuration.as_root and executable == [Path(path).name]:
                target = self._shell("su -c " + shlex.quote(f"readlink /proc/{fields[0]}/exe || true")).strip()
                if target == path:
                    pids.append(int(fields[0]))
        return pids

    def _port_states(self) -> set[str]:
        suffix = ":%04X" % self.configuration.phone_port
        return {
            columns[3] for columns in (line.split() for line in
                                       self._shell("cat /proc/net/tcp /proc/net/tcp6").splitlines())
            if len(columns) > 3 and columns[1].endswith(suffix)
        }

    def _boot(self) -> str:
        return self._shell("cat /proc/sys/kernel/random/boot_id").strip()

    @property
    def active(self) -> bool:
        return self._process is not None

    # --- lifecycle ----------------------------------------------------------------------

    def preflight(self) -> AdbTcpWorkerReceipt:
        """Refuse an occupied phone or port and any worker, library or shard hash mismatch."""
        configuration = self.configuration
        if self._worker_pids():
            raise PhysicalAdapterError(f"{configuration.serial} already runs {configuration.worker_path}")
        if self._port_states() & {_TCP_LISTEN, _TCP_ESTABLISHED}:
            raise PhysicalAdapterError(f"phone port {configuration.phone_port} is occupied on {configuration.serial}")
        listing = self._shell(shlex.join(["sha256sum", *configuration.expected_sha256_by_path]), timeout_s=900)
        observed = {}
        for line in listing.splitlines():
            digest, _, path = line.strip().partition("  ")
            observed[path] = "sha256:" + digest
        mismatched = sorted(path for path, digest in configuration.expected_sha256_by_path.items()
                            if observed.get(path) != digest)
        if mismatched:
            raise PhysicalAdapterError("phone files differ from their pinned sha256: " + ",".join(mismatched))
        self._boot_id = self._boot()
        return AdbTcpWorkerReceipt("preflight", configuration.device_id, configuration.serial, self._boot_id,
                                   MappingProxyType({"observed_sha256_by_path": observed, "port_free": True,
                                                     "worker_absent": True}))

    def start(self, log_path: Path, *, release_failed_start: bool = False) -> AdbTcpWorkerReceipt:
        """Launch the worker and forward a host port to it.

        A failed start keeps the session active, so the trace's stop policy signals the worker.
        ``release_failed_start`` (elastic phones only) instead SIGTERMs the worker this start launched
        on the preflighted boot and leaves the session inactive for a later join."""
        if type(release_failed_start) is not bool:
            raise PhysicalAdapterError("adb-tcp worker failed-start release is invalid")
        if self.active:
            raise PhysicalAdapterError("adb-tcp worker session is already active")
        if self._boot_id is None:
            self.preflight()
        configuration = self.configuration
        command = self._adb("shell", "-T", configuration.worker_shell_command())
        self._ready_line = None
        with log_path.open("x", encoding="ascii", errors="backslashreplace") as log:
            self._process = self._popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        self._log_path = log_path
        try:
            deadline = self._clock() + configuration.launch_timeout_s
            while self._ready_line is None:
                text = log_path.read_text(encoding="ascii", errors="backslashreplace")
                self._ready_line = next((line for line in text.splitlines() if line.startswith(READY_MARKER)), None)
                if self._ready_line is not None:
                    break
                if self._process.poll() is not None or self._clock() > deadline:
                    raise PhysicalAdapterError("adb-tcp worker did not become ready: " + text[-2000:])
                self._sleep(0.25)
            self._host_port = self._open_host_endpoint(log_path)
        except BaseException:
            if release_failed_start:
                self._abandon_failed_start()
            raise
        pids = self._worker_pids()
        # the pids this session launched on this boot: a join may release them if they outlive it
        self._last_launch = (self._boot_id, tuple(pids))
        return AdbTcpWorkerReceipt("launch", configuration.device_id, configuration.serial, self._boot_id or "",
                                   MappingProxyType({"command": command, "host_port": self._host_port,
                                                     "phone_port": configuration.phone_port,
                                                     "ready_line": self._ready_line,
                                                     "worker_pids": pids, **self._endpoint_details()}))

    # --- host endpoint: the adb forward llama-server dials (a subclass may replace it) ------

    def _open_host_endpoint(self, log_path: Path) -> int:
        """Create the host endpoint of a READY worker; returns the host port."""
        configuration = self.configuration
        forward = self._run(
            self._adb("forward", "--no-rebind", f"tcp:{configuration.forward_port}", f"tcp:{configuration.phone_port}"),
            check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30,
        ).stdout.strip()
        return configuration.forward_port or int(forward)

    def _host_endpoint_alive(self, host_port: int) -> bool:
        """The host endpoint still reaches this worker (subprocess errors propagate)."""
        configuration = self.configuration
        listing = self._run(self._adb("forward", "--list"), check=True, capture_output=True, text=True,
                            stdin=subprocess.DEVNULL, timeout=30).stdout
        expected = f"{configuration.serial} tcp:{host_port} tcp:{configuration.phone_port}"
        return expected in (line.strip() for line in listing.splitlines())

    def _release_host_endpoint(self, host_port: int, errors: list[str]) -> bool:
        """Best-effort removal for a lost worker; failures are appended to ``errors``."""
        try:
            self._run(self._adb("forward", "--remove", f"tcp:{host_port}"), check=True,
                      capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30)
            return True
        except (OSError, subprocess.SubprocessError) as error:
            errors.append("forward: " + type(error).__name__ + ": " + str(error))
            return False

    def _close_host_endpoint(self, host_port: int) -> None:
        """Removal at a normal stop, after the worker exited; raises on failure."""
        self._run(self._adb("forward", "--remove", f"tcp:{host_port}"), check=True,
                  capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30)

    def _endpoint_details(self) -> Mapping[str, object]:
        """Extra receipt fields of the host endpoint (none for the adb forward)."""
        return {}

    def _abandon_failed_start(self) -> None:
        """Elastic phones: a failed start leaves the session inactive. The worker it launched may have
        reached READY (or run under ``su`` past its adb client): on the preflighted boot its pids are
        SIGTERMed (never SIGKILL) and recorded, so a join can still release one that outlives this;
        then the local adb client ends. Each phone step is best effort (the phone may be gone)."""
        process, boot_id = self._process, self._boot_id
        pids: list[int] = []
        try:
            if boot_id is not None and self._boot() == boot_id:
                pids = self._worker_pids()
                for pid in pids:
                    self._shell(self._signal_command(pid))
        except (OSError, subprocess.SubprocessError, PhysicalAdapterError):
            pass
        if boot_id is not None:
            self._last_launch = (boot_id, tuple(pids))
        self._process, self._host_port, self._boot_id = None, None, None
        if process is not None:
            deadline = self._clock() + 10.0
            while process.poll() is None and self._clock() < deadline:
                self._sleep(0.1)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    def release_orphaned_worker(self, *, timeout_s: float = 10.0) -> AdbTcpWorkerReceipt | None:
        """Elastic phones, join: SIGTERM a worker this session launched that outlived the session (a
        failed start, or a lost worker whose release could not reach the phone) and wait for it to
        exit. Only while the phone still runs the boot of that launch and every running worker pid is
        one the launch recorded; otherwise None and nothing is signalled (never a foreign worker)."""
        if self.active or self._last_launch is None:
            return None
        boot_id, launched = self._last_launch
        if not launched or self._boot() != boot_id:
            return None
        pids = self._worker_pids()
        if not pids or not set(pids) <= set(launched):
            return None
        for pid in pids:
            self._shell(self._signal_command(pid))
        deadline = self._clock() + timeout_s
        remaining = self._worker_pids()
        while remaining and self._clock() < deadline:
            self._sleep(0.25)
            remaining = self._worker_pids()
        configuration = self.configuration
        receipt = AdbTcpWorkerReceipt("release_orphan", configuration.device_id, configuration.serial, boot_id,
                                      MappingProxyType({"signalled_pids": pids, "worker_pids_after": remaining}))
        if remaining:
            raise PhysicalAdapterError("orphaned adb-tcp worker did not exit after SIGTERM (never killed): "
                                       + repr(receipt.to_json()))
        return receipt

    def _signal_command(self, pid: int) -> str:
        command = f"kill -TERM {pid}"
        return "su -c " + shlex.quote(command) if self.configuration.as_root else command

    def alive(self, *, connect: bool = False) -> bool:
        """The started worker still serves: its adb client runs, its process is on the phone and
        the forward is listed; with ``connect`` the forward must also accept a TCP connection
        (off by default: the worker serves one client at a time). Any adb failure is not alive."""
        process, host_port = self._process, self._host_port
        if process is None or host_port is None or process.poll() is not None:
            return False
        try:
            if not self._worker_pids():
                return False
            if not self._host_endpoint_alive(host_port):
                return False
        except (OSError, subprocess.SubprocessError, PhysicalAdapterError):
            return False
        if connect:
            try:
                with self._connect(host_port):
                    pass
            except OSError:
                return False
        return True

    def client_exited(self) -> bool:
        """The local adb client of the started worker has exited: this session's worker is gone for
        good (the worker died, or its adb shell ended with it). Unlike an adb failure, no later
        check can undo it."""
        process = self._process
        return process is not None and process.poll() is not None

    def release_lost(self) -> AdbTcpWorkerReceipt:
        """Forget a worker that stopped serving: SIGTERM whatever of it still runs on the phone
        (never SIGKILL), end the local adb client and remove the forward. Each step is best effort
        (the phone may be gone); the receipt lists what failed and the session becomes inactive."""
        if self._process is None:
            raise PhysicalAdapterError("adb-tcp worker session is not active")
        configuration = self.configuration
        process, host_port = self._process, self._host_port
        errors, signalled = [], []
        try:
            for pid in self._worker_pids():
                self._shell(self._signal_command(pid))
                signalled.append(pid)
        except (OSError, subprocess.SubprocessError, PhysicalAdapterError) as error:
            errors.append("signal: " + type(error).__name__ + ": " + str(error))
        deadline = self._clock() + 10.0
        while process.poll() is None and self._clock() < deadline:
            self._sleep(0.1)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                errors.append("local adb client did not exit")
        forward_removed = False
        if host_port is not None:
            forward_removed = self._release_host_endpoint(host_port, errors)
        try:
            boot = self._boot()
        except (OSError, subprocess.SubprocessError) as error:
            boot = ""
            errors.append("boot: " + type(error).__name__ + ": " + str(error))
        previous_boot = self._boot_id
        self._process, self._host_port, self._boot_id = None, None, None
        return AdbTcpWorkerReceipt("release_lost", configuration.device_id, configuration.serial, boot,
                                   MappingProxyType({"boot_unchanged": bool(boot) and boot == previous_boot,
                                                     "errors": errors, "exit_code": process.poll(),
                                                     "forward_removed": forward_removed,
                                                     "signalled_pids": signalled, **self._endpoint_details()}))

    def terminate_for_fault_injection(
        self, *, authorized: bool, elastic_phones: Mapping[str, object] | None,
    ) -> AdbTcpWorkerReceipt:
        """Hardware fault injection: SIGTERM the worker (never SIGKILL), only for an elastic-phones
        run and only when the caller passes the user's explicit authorization. Logged."""
        if authorized is not True or not elastic_phones:
            raise PhysicalAdapterError("fault injection needs elastic phones and explicit authorization")
        if not self.active:
            raise PhysicalAdapterError("adb-tcp worker session is not active")
        configuration = self.configuration
        pids = self._worker_pids()
        if not pids:
            raise PhysicalAdapterError("adb-tcp worker has no process to signal")
        started_ns = time.monotonic_ns()
        for pid in pids:
            self._shell(self._signal_command(pid))
        receipt = AdbTcpWorkerReceipt("fault_injection", configuration.device_id, configuration.serial,
                                      self._boot_id or "", MappingProxyType({
                                          "signal": "TERM", "signalled_at_monotonic_ns": started_ns,
                                          "worker_pids": pids}))
        self.fault_injections.append(receipt.to_json())
        return receipt

    def transport_parameters(self) -> Mapping[str, int | str]:
        """``phone_transport_contract`` input of the live forward."""
        if self._host_port is None:
            raise PhysicalAdapterError("adb-tcp worker session is not active")
        return MappingProxyType({
            "adb_port": self.configuration.adb_port,
            "adb_serial": self.configuration.serial,
            "ffn_transport": "adb-tcp",
            "ffn_worker_host": "127.0.0.1",
            "ffn_worker_port": self._host_port,
            "phone_worker_port": self.configuration.phone_port,
        })

    def transport_contract(self) -> PhoneTransportContract:
        return phone_transport_contract(self.transport_parameters())

    def binding(self, label: str) -> PhoneHelperBinding:
        return PhoneHelperBinding(
            device_id=self.configuration.device_id, serial=self.configuration.serial,
            layer_mask=self.configuration.layer_mask, label=label,
            transport_parameters=self.transport_parameters(),
        )

    def _drain(self, served_calls: int) -> int:
        """Consume the rest of a finite budget with zero-input calls; returns the calls sent."""
        configuration = self.configuration
        remaining = configuration.max_requests - served_calls
        if served_calls < 0 or remaining < 0:
            raise PhysicalAdapterError("served calls exceed the worker's request budget")
        if remaining == 0:
            return 0
        payload = bytes(2 * configuration.n_embd)
        flags = 1 | (2 if configuration.swiglu else 0)
        layer = (configuration.layer_mask & -configuration.layer_mask).bit_length() - 1
        with self._connect(self._host_port) as stream:
            stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            stream.sendall(HELLO_REQUEST.pack(
                PROTOCOL_MAGIC, PROTOCOL_VERSION, 1, configuration.layer_mask, configuration.n_embd,
                configuration.columns, flags, configuration.max_tokens,
                bytes.fromhex(configuration.artifact_sha256[7:])))
            hello = HELLO_RESPONSE.unpack(self._receive(stream, HELLO_RESPONSE.size))
            if hello[:4] != (PROTOCOL_MAGIC, PROTOCOL_VERSION, 2, 0):
                raise PhysicalAdapterError(f"adb-tcp drain HELLO was refused: {hello[:4]!r}")
            for request_id in range(1, remaining + 1):
                try:
                    stream.sendall(EXECUTE_REQUEST.pack(
                        PROTOCOL_MAGIC, PROTOCOL_VERSION, 3, request_id, layer, configuration.n_embd,
                        len(payload), _fnv32(payload), configuration.columns, 1) + payload)
                    response = EXECUTE_RESPONSE.unpack(self._receive(stream, EXECUTE_RESPONSE.size))
                    output = self._receive(stream, len(payload))
                except (OSError, PhysicalAdapterError):
                    # the server's count was low: the worker reached its budget and exited on its own
                    if self._exited_normally(10.0):
                        return request_id - 1
                    raise
                if response[:6] != (PROTOCOL_MAGIC, PROTOCOL_VERSION, 4, 0, 0, request_id) \
                        or response[9] != _fnv32(output):
                    raise PhysicalAdapterError(f"adb-tcp drain call {request_id} failed: {response!r}")
        return remaining

    def _exited_normally(self, timeout_s: float) -> bool:
        deadline = self._clock() + timeout_s
        while self._process.poll() is None and self._clock() < deadline:
            self._sleep(0.1)
        return self._process.poll() == 0

    def _wait_disconnected(self, timeout_s: float) -> None:
        """The worker serves one client at a time: wait until the server's client has left."""
        deadline = self._clock() + timeout_s
        while True:
            text = self._log_path.read_text(encoding="ascii", errors="backslashreplace")
            if _idle_log(text) or self._process.poll() is not None:
                return
            if self._clock() > deadline:
                raise PhysicalAdapterError("the adb-tcp worker still serves a client; it is left running")
            self._sleep(0.25)

    @staticmethod
    def _receive(stream: socket.socket, count: int) -> bytes:
        data = bytearray()
        while len(data) < count:
            part = stream.recv(count - len(data))
            if not part:
                raise PhysicalAdapterError("adb-tcp worker closed the connection early")
            data.extend(part)
        return bytes(data)

    def _worker_gone(self) -> bool:
        """The phone runs no worker of this path. A worker killed during a call never logs that its
        client left, so only the phone's process list tells it is gone; an adb failure proves nothing."""
        try:
            return not self._worker_pids()
        except (OSError, subprocess.SubprocessError, PhysicalAdapterError):
            return False

    def stop(
        self,
        *,
        served_calls: int | None = None,
        allow_idle_signal: bool = False,
        timeout_s: float = 120.0,
        release_exited: bool = False,
    ) -> AdbTcpWorkerReceipt:
        """Let the worker exit normally, remove the forward and check the phone did not reboot.

        ``served_calls`` is the helper's call count from the server's per-helper summary; it is
        required for a finite budget. A resident worker is signalled only with
        ``allow_idle_signal`` and only while no client is connected. An in-flight worker is never
        signalled or killed: the stop fails and leaves it running.

        ``release_exited`` (elastic phones): a worker already gone from the phone (lost mid-call, so
        its log still shows the server's client) is neither drained nor signalled; its local adb
        client is ended, the forward removed and the receipt (``already_exited``) returned whatever
        its exit code. A reboot of the phone still fails the stop (after the forward is removed).
        """
        if type(release_exited) is not bool:
            raise PhysicalAdapterError("adb-tcp worker exited-release is invalid")
        if not self.active or self._log_path is None:
            raise PhysicalAdapterError("adb-tcp worker session is not active")
        configuration = self.configuration
        drained, signalled = 0, False
        exited = release_exited and self._worker_gone()
        if exited:
            deadline = self._clock() + 10.0
            while self._process.poll() is None and self._clock() < deadline:
                self._sleep(0.1)
            if self._process.poll() is None:
                # nothing is left on the phone to relay: the local client is ours to end
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        elif configuration.max_requests > 0:
            if served_calls is None:
                raise PhysicalAdapterError("a finite worker budget needs the served call count to drain")
            self._wait_disconnected(timeout_s)
            if self._process.poll() is None:
                drained = self._drain(served_calls)
        else:
            text = self._log_path.read_text(encoding="ascii", errors="backslashreplace")
            idle = _idle_log(text) and _TCP_ESTABLISHED not in self._port_states()
            if not allow_idle_signal or not idle:
                raise PhysicalAdapterError(
                    "resident adb-tcp worker stays running: stopping it needs allow_idle_signal and no client"
                )
            for pid in self._worker_pids():
                self._shell(self._signal_command(pid))
                signalled = True
        deadline = self._clock() + timeout_s
        while self._process.poll() is None:
            if self._clock() > deadline:
                raise PhysicalAdapterError("adb-tcp worker did not exit; it is left running, never killed")
            self._sleep(0.25)
        self._close_host_endpoint(self._host_port)
        boot = self._boot()
        exit_code = self._process.returncode
        receipt = AdbTcpWorkerReceipt("stop", configuration.device_id, configuration.serial, boot,
                                      MappingProxyType({"boot_unchanged": boot == self._boot_id,
                                                        "drained_calls": drained, "exit_code": exit_code,
                                                        "forward_removed": True, "signalled": signalled,
                                                        "worker_pids_after": self._worker_pids(),
                                                        **({"already_exited": exited}
                                                           if release_exited else {}),
                                                        **self._endpoint_details()}))
        self._process = None
        self._host_port = None
        if boot != self._boot_id or (exit_code != 0 and not signalled and not exited):
            raise PhysicalAdapterError("adb-tcp worker stop failed: " + repr(receipt.to_json()))
        return receipt
