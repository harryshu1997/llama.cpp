"""Execute scheduler-selected whole-model llama-server routes on Android."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import shlex
import secrets
import subprocess
import threading
import time
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit

from .._internal.model_manifest import ModelManifest
from .._internal.runtime_plan import RuntimeHelperExecutionEnvelope
from .._internal.capability_contracts.executors import whole_phone_launch_parameters
from .probes import (
    AndroidProcessIdentity, parse_android_process_identity,
    parse_android_process_allocation, parse_android_process_memory_peak,
)
from .contracts import PhysicalAdapterError
from .llama_server import (
    LlamaServerExecutionMarker,
    LlamaServerExecutionProof,
    ManagedLlamaServer,
    llama_server_launch_contract,
)
from .ticket import PhysicalExecutionCommand, PhysicalTransitionCommand


ANDROID_LLAMA_SERVER_ADAPTER = "android-llama-server-v1"
TOKEN_ID_REQUEST_PROTOCOL = "token-ids-v1"
NCM_CONTROL_SCRIPT = Path(__file__).with_name("native") / "android_ncm_adb_control.sh"


def ncm_control_script_sha256() -> str:
    return "sha256:" + hashlib.sha256(NCM_CONTROL_SCRIPT.read_bytes()).hexdigest()


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise PhysicalAdapterError(name + " is invalid")
    return value


def _integer(name: str, value: object, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise PhysicalAdapterError(name + " is invalid")
    return value


@dataclass(frozen=True)
class AndroidLlamaServerProcessConfiguration:
    adb_path: Path
    serial: str
    adb_port: int
    remote_server_path: str
    remote_library_directory: str
    remote_model_paths_by_artifact: Mapping[str, str]
    remote_state_directory: str
    executable_device_name: str
    output_directory: Path
    control_transport: str = "adb-usb"
    ncm_adb_endpoint: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.adb_path, Path)
            or not self.adb_path.is_file()
            or not isinstance(self.output_directory, Path)
            or not self.output_directory.is_dir()
        ):
            raise PhysicalAdapterError(
                "Android llama-server process path is invalid"
            )
        _text("Android llama-server serial", self.serial)
        _integer("Android llama-server ADB port", self.adb_port)
        if self.control_transport not in {"adb-usb", "adb-ncm"}:
            raise PhysicalAdapterError("Android control transport is invalid")
        if self.control_transport == "adb-ncm":
            try:
                host, port = self.ncm_adb_endpoint.split(":")
                address = ipaddress.IPv4Address(host)
                valid = address.is_private and not address.is_loopback and 0 < int(port) <= 65535
            except (AttributeError, ValueError):
                valid = False
            if not valid:
                raise PhysicalAdapterError("Android NCM ADB endpoint is invalid")
        elif self.ncm_adb_endpoint is not None:
            raise PhysicalAdapterError("Android USB control cannot name an NCM endpoint")
        for name in (
            "remote_server_path",
            "remote_library_directory",
            "remote_state_directory",
            "executable_device_name",
        ):
            _text("Android llama-server " + name, getattr(self, name))
        models = dict(self.remote_model_paths_by_artifact)
        if not models or any(
            type(artifact) is not str
            or not artifact.startswith("sha256:")
            or len(artifact) != 71
            or type(path) is not str
            or not path
            or not path.isascii()
            for artifact, path in models.items()
        ):
            raise PhysicalAdapterError(
                "Android llama-server model map is invalid"
            )
        object.__setattr__(self, "remote_model_paths_by_artifact", models)


class ManagedAndroidLlamaServer:
    """One Android server, ADB forward, and scheduler execution proof."""

    def __init__(
        self,
        managed: ManagedLlamaServer,
        launcher: "AndroidLlamaServerProcessLauncher",
        forward_port: int,
        pid_file: str,
        remote_pid: int,
        *,
        process_identity: AndroidProcessIdentity,
        artifact_sha256: str,
        launch_parameters: Mapping[str, int | str],
        endpoint: str,
    ) -> None:
        if not isinstance(process_identity, AndroidProcessIdentity) or process_identity.process_id != remote_pid:
            raise PhysicalAdapterError("Android process identity differs from managed endpoint")
        self._managed = managed
        self._launcher = launcher
        self._forward_port = forward_port
        self._pid_file = pid_file
        self._remote_pid = remote_pid
        self._lock = threading.Lock()
        self._stopped = False
        self.process_identity = process_identity
        self.artifact_sha256 = artifact_sha256
        self.launch_parameters = whole_phone_launch_parameters(launch_parameters)
        self.endpoint = endpoint

    @property
    def process(self):
        return self._managed.process

    def observe_allocation(self) -> dict[str, object]:
        with self._lock:
            if self._stopped or self.process is None or self.process.poll() is not None:
                raise PhysicalAdapterError("Android allocation endpoint is not live")
        observed = self._launcher.probe_process_allocation(self.process_identity, self._pid_file)
        with self._lock:
            if self._stopped or self.process is None or self.process.poll() is not None:
                raise PhysicalAdapterError("Android allocation endpoint stopped during observation")
        return observed

    def begin_execution(
        self,
        command: PhysicalExecutionCommand,
        manifest: ModelManifest,
    ) -> LlamaServerExecutionMarker:
        return self._managed.begin_execution(command, manifest)

    def finish_execution(
        self,
        marker: LlamaServerExecutionMarker,
        command: PhysicalExecutionCommand,
        manifest: ModelManifest,
        *,
        output_tokens: int,
        adaptive_observation=None,
        static_control_ack: Mapping[str, object] | None = None,
        helper_envelopes: Sequence[RuntimeHelperExecutionEnvelope] = (),
    ) -> LlamaServerExecutionProof:
        return self._managed.finish_execution(
            marker,
            command,
            manifest,
            output_tokens=output_tokens,
            adaptive_observation=adaptive_observation,
            static_control_ack=static_control_ack,
            helper_envelopes=helper_envelopes,
        )

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        errors = []
        try:
            self._launcher.stop_remote(self._remote_pid, self._pid_file, expected=self.process_identity)
        except BaseException:
            with self._lock:
                self._stopped = False
            raise
        try:
            self._managed.stop()
        except BaseException as error:
            errors.append(error)
        try:
            self._launcher.remove_forward(self._forward_port)
        except BaseException as error:
            errors.append(error)
        if errors:
            raise PhysicalAdapterError(
                "Android llama-server cleanup failed: "
                + "; ".join(str(error) for error in errors)
            )


class AndroidLlamaServerProcessLauncher:
    """Launch only the Android endpoint encoded in a scheduler ticket."""

    def __init__(
        self, configuration: AndroidLlamaServerProcessConfiguration
    ) -> None:
        if not isinstance(
            configuration, AndroidLlamaServerProcessConfiguration
        ):
            raise PhysicalAdapterError(
                "Android llama-server configuration is invalid"
            )
        self._configuration = configuration
        self._control_boot_id: str | None = None
        self._control_root: str | None = None
        self._control_connected_here = False
        self.control_events: list[dict[str, object]] = []

    def _adb_command(self, serial: str, *arguments: str) -> list[str]:
        return [str(self._configuration.adb_path), "-P", str(self._configuration.adb_port),
                "-s", serial, *arguments]

    @property
    def control_serial(self) -> str:
        return self._configuration.ncm_adb_endpoint or self._configuration.serial

    @property
    def ncm_control_prepared(self) -> bool:
        return self._control_boot_id is not None and self._control_root is not None

    def prepare_ncm_control(self, gadget_path: str) -> None:
        if self._configuration.control_transport != "adb-ncm":
            return
        if not gadget_path.startswith("/config/usb_gadget/"):
            raise PhysicalAdapterError("Android NCM gadget path is invalid")
        bootstrap = lambda *args: subprocess.run(
            self._adb_command(self._configuration.serial, *args), check=True,
            capture_output=True, text=True, timeout=10,
        )
        identity = bootstrap("shell", "getprop ro.serialno; cat /proc/sys/kernel/random/boot_id").stdout.splitlines()
        if len(identity) != 2 or identity[0] != self._configuration.serial or len(identity[1]) != 36:
            raise PhysicalAdapterError("Android bootstrap device identity differs")
        self._control_boot_id = identity[1]
        token = secrets.token_hex(12)
        root = self._configuration.remote_state_directory.rstrip("/") + "/control-" + token
        staging = "/data/local/tmp/s42-ncm-control-" + token + ".sh"
        remote = root + "/control.sh"
        bootstrap("shell", "su -c " + shlex.quote("umask 077; mkdir -p " + shlex.quote(root)))
        bootstrap("push", str(NCM_CONTROL_SCRIPT), staging)
        bootstrap("shell", "su -c " + shlex.quote(
            "mv " + shlex.quote(staging) + " " + shlex.quote(remote)
            + " && chmod 600 " + shlex.quote(remote)))
        digest = bootstrap("shell", "su -c " + shlex.quote("sha256sum " + shlex.quote(remote))).stdout.split()
        if not digest or "sha256:" + digest[0] != ncm_control_script_sha256():
            raise PhysicalAdapterError("Android NCM bootstrap script hash differs")
        port = self._configuration.ncm_adb_endpoint.rsplit(":", 1)[1]
        body = ("nohup " + shlex.join(["sh", remote, root, gadget_path, port,
                                       self._configuration.serial, "7200"])
                + " > " + shlex.quote(root + "/control.log") + " 2>&1 < /dev/null &")
        bootstrap("shell", "su -c " + shlex.quote(body))
        self._control_root = root
        for _ in range(20):
            result = bootstrap("shell", "su -c " + shlex.quote(
                "cat " + shlex.quote(root + "/control.ready") + " 2>/dev/null || true"))
            if result.stdout.strip() == "armed":
                self.control_events.append({"kind": "NCM_CONTROL_ARMED", "timestamp_ns": time.monotonic_ns(),
                                            "boot_id": self._control_boot_id, "script_sha256": ncm_control_script_sha256(),
                                            "endpoint": self.control_serial, "remote_root": root})
                return
            time.sleep(0.1)
        raise PhysicalAdapterError("Android NCM control bootstrap did not arm")

    def connect_ncm_control(self) -> None:
        if self._configuration.control_transport != "adb-ncm":
            return
        if self._control_boot_id is None:
            raise PhysicalAdapterError("Android NCM control was not bootstrapped")
        result = subprocess.run(
            [str(self._configuration.adb_path), "-P", str(self._configuration.adb_port),
             "connect", self.control_serial], capture_output=True, text=True, timeout=3, check=True,
        )
        if result.stdout.strip() not in {"connected to " + self.control_serial,
                                        "already connected to " + self.control_serial}:
            raise PhysicalAdapterError("Android NCM control connection failed: " + result.stdout.strip())
        self._control_connected_here |= result.stdout.strip() == "connected to " + self.control_serial
        identity = self._adb("shell", "getprop ro.serialno; cat /proc/sys/kernel/random/boot_id", timeout_s=3).stdout.splitlines()
        if identity != [self._configuration.serial, self._control_boot_id]:
            raise PhysicalAdapterError("Android NCM device or boot identity differs")
        self.control_events.append({"kind": "NCM_CONTROL_CONNECTED", "timestamp_ns": time.monotonic_ns(),
                                    "boot_id": self._control_boot_id, "endpoint": self.control_serial})

    def close_control(self) -> None:
        if self._control_root is not None:
            # USB has been restored by the existing session cleanup at this point.
            subprocess.run(self._adb_command(self._configuration.serial, "shell", "su -c " + shlex.quote(
                "touch " + shlex.quote(self._control_root + "/control.cancel"))),
                capture_output=True, text=True, timeout=3, check=True)
            self._control_root = None
        if self._control_connected_here:
            prefix = [str(self._configuration.adb_path), "-P", str(self._configuration.adb_port)]
            devices = subprocess.run([*prefix, "devices"], capture_output=True, text=True, timeout=3, check=True)
            if any(row.split()[0] == self.control_serial for row in devices.stdout.splitlines()[1:] if row.split()):
                subprocess.run([*prefix, "disconnect", self.control_serial],
                               capture_output=True, text=True, timeout=3, check=True)
            self._control_connected_here = False

    def _adb(self, *arguments: str, check: bool = True, timeout_s: float = 30):
        return subprocess.run(
            self._adb_command(self.control_serial, *arguments),
            check=check,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
            timeout=timeout_s,
        )

    def _su(self, command: str, *, check: bool = True, timeout_s: float = 30):
        return self._adb(
            "shell", "su -c " + shlex.quote(command), check=check, timeout_s=timeout_s,
        )

    @staticmethod
    def _process_identity_command(process_id: int, pid_file: str) -> str:
        _integer("Android allocation process id", process_id)
        _text("Android allocation pid file", pid_file)
        return ("cat /proc/sys/kernel/random/boot_id /proc/" + str(process_id) + "/stat\n"
                + "readlink /proc/" + str(process_id) + "/exe\ncat " + shlex.quote(pid_file))

    def probe_process_allocation(self, expected: AndroidProcessIdentity, pid_file: str) -> dict[str, object]:
        identity_command = self._process_identity_command(expected.process_id, pid_file)
        command = ("set -e\n" + identity_command
                   + "\nprintf '\\nS42_PROCESS_MEMORY_V1\\n'\n"
                   + "dumpsys -t 1 meminfo -s " + str(expected.process_id)
                   + "\nprintf '\\nS42_PROCESS_MEMORY_V1\\n'\n" + identity_command)
        started = time.monotonic_ns()
        result = self._su(command, timeout_s=3)
        return parse_android_process_allocation(
            result.stdout, expected, captured_at_ns=started, finished_at_ns=time.monotonic_ns(),
        )

    def probe_process_memory_peak(self, expected: AndroidProcessIdentity, pid_file: str) -> dict[str, object]:
        identity = self._process_identity_command(expected.process_id, pid_file)
        marker = "\nprintf '\\nS42_PROCESS_PEAK_V1\\n'\n"
        gpu_root = "/sys/class/kgsl/kgsl/proc/" + str(expected.process_id) + "/"
        paths = " ".join(gpu_root + name for name in (
            "kernel", "kernel_max", "user", "user_max", "imported_mem",
        ))
        command = ("set -e\n" + identity + marker
                   + "cat /proc/" + str(expected.process_id) + "/status" + marker
                   + "cat " + paths + marker + identity)
        started = time.monotonic_ns()
        result = self._su(command, timeout_s=3)
        return parse_android_process_memory_peak(
            result.stdout, expected, captured_at_ns=started, finished_at_ns=time.monotonic_ns(),
        )

    @staticmethod
    def _endpoint(endpoint: str) -> int:
        parsed = urlsplit(endpoint)
        if not (
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost"}
            and parsed.port is not None
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
        ):
            raise PhysicalAdapterError(
                "Android llama-server endpoint is not an ADB-forwarded port"
            )
        return parsed.port

    @staticmethod
    def _healthy(port: int) -> bool:
        connection = http.client.HTTPConnection(
            "127.0.0.1", port, timeout=1
        )
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            payload = json.loads(response.read())
            return (
                response.status == 200
                and type(payload) is dict
                and payload.get("status") == "ok"
            )
        finally:
            connection.close()

    def remove_forward(self, port: int) -> None:
        self._adb("forward", "--remove", "tcp:" + str(port), check=False)

    def stop_remote(self, pid: int, pid_file: str, *, expected: AndroidProcessIdentity | None = None) -> None:
        identity_command = self._process_identity_command(pid, pid_file)
        probe = "if [ -d /proc/" + str(pid) + " ]; then " + identity_command + "; else echo S42_EXITED; fi"
        observed = self._su(probe, timeout_s=3).stdout.strip()
        if observed == "S42_EXITED":
            return
        identity = parse_android_process_identity(observed, pid)
        if ((expected is not None and identity != expected)
            or identity.executable != self._configuration.remote_server_path
            or (self._control_boot_id is not None and identity.boot_id != self._control_boot_id)):
            raise PhysicalAdapterError("Android stop process identity differs")
        self._su("kill -TERM " + str(pid), timeout_s=3)
        for _ in range(50):
            observed = self._su(probe, timeout_s=3).stdout.strip()
            if observed == "S42_EXITED":
                return
            if parse_android_process_identity(observed, pid) != identity:
                raise PhysicalAdapterError("Android process identity changed while stopping")
            time.sleep(0.1)
        raise PhysicalAdapterError("Android server did not terminate")

    def _remote_sha256(self, path: str) -> str:
        result = self._su("sha256sum " + shlex.quote(path))
        fields = result.stdout.strip().split()
        if len(fields) < 1 or len(fields[0]) != 64:
            raise PhysicalAdapterError(
                "Android llama-server remote hash is invalid"
            )
        return "sha256:" + fields[0]

    def launch(
        self,
        command: PhysicalTransitionCommand,
        manifest: ModelManifest,
        *,
        label: str,
        control_check: Callable[[], None],
    ) -> ManagedAndroidLlamaServer:
        parameters = command.adapter_parameters
        mode = parameters.get("android_control_transport", "adb-usb")
        if mode != self._configuration.control_transport or (
            mode == "adb-ncm" and (
                parameters.get("android_control_endpoint") != self.control_serial
                or parameters.get("android_control_script_sha256") != ncm_control_script_sha256()
                or parameters.get("request_transport_generation") != "adb-ncm-token-http-v1"
            )
        ):
            raise PhysicalAdapterError("Android control transport differs from the ticket")
        self.connect_ncm_control()
        if (
            parameters.get("execution_adapter")
                != ANDROID_LLAMA_SERVER_ADAPTER
            or parameters.get("request_io_protocol")
                != TOKEN_ID_REQUEST_PROTOCOL
        ):
            raise PhysicalAdapterError(
                "Android whole-model execution contract is absent"
            )
        contract = llama_server_launch_contract(command, manifest)
        configured_model = (
            self._configuration.remote_model_paths_by_artifact.get(
                manifest.artifact_sha256
            )
        )
        if configured_model is None:
            raise PhysicalAdapterError(
                "Android llama-server model artifact is not configured"
            )
        remote_model = _text(
            "Android llama-server remote model",
            parameters.get("remote_model_path"),
        )
        remote_server = _text(
            "Android llama-server remote executable",
            parameters.get("remote_server_path"),
        )
        remote_library = _text(
            "Android llama-server remote library directory",
            parameters.get("remote_library_directory"),
        )
        executable_device = _text(
            "Android llama-server executable device",
            parameters.get("executable_device"),
        )
        remote_port = _integer(
            "Android llama-server remote port",
            parameters.get("remote_port"),
        )
        forward_port = self._endpoint(command.participant.endpoint)
        if (
            remote_model != configured_model
            or remote_server != self._configuration.remote_server_path
            or remote_library
                != self._configuration.remote_library_directory
            or executable_device
                != self._configuration.executable_device_name
            or parameters.get("forward_port") != forward_port
            or contract.gpu_device_id != command.participant.device_id
            or contract.gpu_layers != manifest.block_count
        ):
            raise PhysicalAdapterError(
                "Android llama-server configuration differs from the ticket"
            )
        if (
            self._remote_sha256(remote_server)
                != parameters.get("remote_server_sha256")
            or self._remote_sha256(remote_model)
                != manifest.artifact_sha256
        ):
            raise PhysicalAdapterError(
                "Android llama-server artifact differs from the ticket"
            )
        pid_file = (
            self._configuration.remote_state_directory.rstrip("/")
            + "/" + label + ".pid"
        )
        arguments = [
            remote_server,
            "--model", remote_model,
            "--alias", contract.model_alias,
            "--ctx-size", str(contract.context_size),
            "--parallel", str(contract.parallel),
            "--batch-size", str(contract.batch_size),
            "--ubatch-size", str(contract.ubatch_size),
            "--cont-batching",
            "--cache-type-k", "f16",
            "--cache-type-v", "f16",
            "--host", "127.0.0.1",
            "--port", str(remote_port),
            "--n-gpu-layers", str(contract.gpu_layers),
            "--device", executable_device,
            "--flash-attn", "off",
            "--slots",
            "--metrics",
            "--no-webui",
            "--log-colors", "off",
        ]
        body = " ".join(shlex.quote(value) for value in arguments)
        shell = (
            "mkdir -p "
            + shlex.quote(self._configuration.remote_state_directory)
            + " && cd " + shlex.quote(remote_library)
            + " && export LD_LIBRARY_PATH=."
            + " && echo $$ > " + shlex.quote(pid_file)
            + " && exec " + body
        )
        self._adb(
            "forward",
            "--no-rebind",
            "tcp:" + str(forward_port),
            "tcp:" + str(remote_port),
        )
        managed = ManagedLlamaServer(
            tuple([
                str(self._configuration.adb_path),
                "-P", str(self._configuration.adb_port),
                "-s", self.control_serial,
                "shell", "su -c " + shlex.quote(shell),
            ]),
            os.environ.copy(),
            self._configuration.output_directory,
            label,
            contract,
        )
        managed.start()
        remote_pid = None
        deadline = time.monotonic() + 300
        try:
            while time.monotonic() < deadline:
                control_check()
                if managed.process is None or managed.process.poll() is not None:
                    raise PhysicalAdapterError(
                        "Android llama-server exited during model load"
                    )
                if remote_pid is None:
                    result = self._su(
                        "cat " + shlex.quote(pid_file), check=False
                    )
                    value = result.stdout.strip()
                    if value.isdigit() and int(value) > 0:
                        remote_pid = int(value)
                try:
                    ready = self._healthy(forward_port)
                except (OSError, ValueError, json.JSONDecodeError):
                    ready = False
                if ready and remote_pid is not None:
                    identity = parse_android_process_identity(
                        self._su(self._process_identity_command(remote_pid, pid_file)).stdout, remote_pid,
                    )
                    if identity.executable != remote_server:
                        raise PhysicalAdapterError("Android process executable differs from verified launch")
                    return ManagedAndroidLlamaServer(
                        managed,
                        self,
                        forward_port,
                        pid_file,
                        remote_pid,
                        process_identity=identity,
                        artifact_sha256=manifest.artifact_sha256,
                        launch_parameters=parameters,
                        endpoint=command.participant.endpoint,
                    )
                time.sleep(0.1)
            raise PhysicalAdapterError(
                "Android llama-server model load timed out"
            )
        except BaseException:
            if remote_pid is not None:
                self.stop_remote(remote_pid, pid_file)
            managed.stop()
            self.remove_forward(forward_port)
            raise
