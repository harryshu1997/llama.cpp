"""The host driver of the portable path: one llama-server on one GPU of this site.

``HostLaunchPolicy`` is the part of a server launch that does not depend on the machine (layers on the
GPU, context, slots, cache handling); the driver adds what does: the model's path, the serving GPU by
index, the threads, the library path. With a ``HelperBinding`` the server shares its CPU-resident FFN
layers with a phone: the driver sets the model-side variables, the phone driver's ``HelperEndpoint`` adds
the link.

The launch policy defaults record what the first shared host cost in lost runs: no prompt cache and a
slot-save path with an erase after each request (an idle slot save reads the whole attention cache back
from the GPU), a long load wait (a cold page cache on a busy host), and host weights kept resident when
memory allows (releasing and restoring them costs every request its first token).
"""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Callable, Mapping, Protocol, Sequence

from ..adapters.contracts import PhysicalAdapterError
from ..configuration.site import SiteManifest
from .gates import ForeignLoadWatch, gpu_free
from .meter import HostMeter
from .phone import HelperBinding


HOST_SHARES = ("resident", "dormant")
FFN_PREFIX = "S41_SERVER_FFN_"
SERVER_COMMAND = "SERVER_COMMAND.json"
SERVER_LOG = "server.log"


@dataclass(frozen=True)
class HostLaunchPolicy:
    gpu_layers: int
    ctx_size: int
    parallel: int
    batch_size: int = 512
    ubatch_size: int = 512
    flash_attn: str = "off"
    kv_unified: bool = True
    warmup: bool = False
    prompt_cache_mib: int | None = 0          # --cache-ram (None = the server's default)
    slot_save: bool = True                    # --slot-save-path: enables erasing a slot after its request
    host_share: str = "resident"              # host copy of the shared columns: kept resident or released
    max_tokens: int = 4                       # rows of several requests carried by one phone call
    timeout_ms: int = 120000
    load_timeout_s: float = 5400.0
    extra_arguments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.host_share not in HOST_SHARES:
            raise PhysicalAdapterError("host share must be resident or dormant")
        if min(self.gpu_layers, self.ctx_size, self.parallel, self.batch_size, self.ubatch_size) < 0 \
                or self.parallel < 1 or self.ctx_size < 1:
            raise PhysicalAdapterError("host launch policy is invalid")

    def arguments(self, model_path: Path, alias: str, port: int, threads: int, slot_directory: Path | None) -> list[str]:
        words = ["-m", str(model_path), "--alias", alias, "--host", "127.0.0.1", "--port", str(port),
                 "-c", str(self.ctx_size), "-b", str(self.batch_size), "-ub", str(self.ubatch_size),
                 "-ngl", str(self.gpu_layers), "-t", str(threads), "-tb", str(threads),
                 "--parallel", str(self.parallel), "--flash-attn", self.flash_attn]
        if not self.warmup:
            words.append("--no-warmup")
        if self.kv_unified:
            words.append("--kv-unified")
        if self.prompt_cache_mib is not None:
            words += ["--cache-ram", str(self.prompt_cache_mib)]
        if self.slot_save and slot_directory is not None:
            words += ["--slot-save-path", str(slot_directory)]
        return words + list(self.extra_arguments)

    def to_json(self) -> dict[str, object]:
        return {"batch_size": self.batch_size, "ctx_size": self.ctx_size, "extra_arguments": list(self.extra_arguments),
                "flash_attn": self.flash_attn, "gpu_layers": self.gpu_layers, "host_share": self.host_share,
                "kv_unified": self.kv_unified, "load_timeout_s": self.load_timeout_s, "max_tokens": self.max_tokens,
                "parallel": self.parallel, "prompt_cache_mib": self.prompt_cache_mib, "slot_save": self.slot_save,
                "timeout_ms": self.timeout_ms, "ubatch_size": self.ubatch_size, "warmup": self.warmup}

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "HostLaunchPolicy":
        row = dict(value)
        row["extra_arguments"] = tuple(row.get("extra_arguments", ()))
        return cls(**row)


def shared_layer_environment(binding: HelperBinding, policy: HostLaunchPolicy) -> dict[str, str]:
    """The model-side ``S41_SERVER_FFN_*`` variables of a server that shares layers with a helper."""
    environment = {
        FFN_PREFIX + "ACTIVATION": binding.activation,
        FFN_PREFIX + "ARTIFACT_SHA256": binding.artifact_sha256,
        FFN_PREFIX + "COLUMNS": str(binding.stored_columns),
        FFN_PREFIX + "F16_IO": "1",
        FFN_PREFIX + "MAX_TOKENS": str(policy.max_tokens),
        FFN_PREFIX + "N_EMBD": str(binding.n_embd),
        FFN_PREFIX + "RUNTIME_CONTROL": "1",
        FFN_PREFIX + "TIMEOUT_MS": str(policy.timeout_ms),
    }
    if policy.host_share == "dormant":
        environment.update({FFN_PREFIX + "DORMANT_HOST_SHARE": "1", FFN_PREFIX + "DORMANT_DROP_CACHE": "0",
                            FFN_PREFIX + "DORMANT_POPULATE": "1"})
    else:
        environment[FFN_PREFIX + "DORMANT_HOST_SHARE"] = "0"
    return environment


class ServerLoadFailure(PhysicalAdapterError):
    """The server did not come up: ``kind`` is ``gpu-out-of-memory``, ``exited`` or ``timeout``."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"server load failed ({kind}): {detail}")
        self.kind = kind


@dataclass
class ServerHandle:
    process: object
    port: int
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    log_path: Path
    parallel: int

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def pid(self) -> int:
        return self.process.pid


class HostDriver(Protocol):
    def describe(self) -> Mapping[str, object]: ...
    def gpu_state(self) -> Mapping[str, object]: ...
    def meter(self, *, interval_s: float = 0.1) -> HostMeter: ...
    def foreign_load(self, own_pids: Callable[[], Sequence[int]], *, limit_cores: float, window_s: float,
                     on_trip: Callable[[], None]) -> ForeignLoadWatch: ...
    def command(self, model_id: str, alias: str, port: int, policy: HostLaunchPolicy, *,
                binding: HelperBinding | None, helper_environment: Mapping[str, str],
                output: Path) -> tuple[list[str], dict[str, str]]: ...
    def launch(self, model_id: str, alias: str, port: int, policy: HostLaunchPolicy, *,
               binding: HelperBinding | None, helper_environment: Mapping[str, str], output: Path) -> ServerHandle: ...
    def wait_ready(self, handle: ServerHandle, timeout_s: float) -> None: ...
    def tokenize(self, handle: ServerHandle, text: str) -> list[int]: ...
    def kill(self, handle: ServerHandle) -> None: ...
    def slots_busy(self, handle: ServerHandle) -> list[object]: ...
    def erase_slot(self, handle: ServerHandle, slot: int, timeout_s: float = 20) -> Mapping[str, object]: ...
    def stop(self, handle: ServerHandle) -> Mapping[str, object]: ...


def _request(port: int, method: str, path: str, body: object | None = None, timeout_s: float = 30):
    """(status, decoded JSON or None) of one local HTTP request; (None, None) when the server is unreachable."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
    try:
        connection.request(method, path, body=None if body is None else json.dumps(body),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read()
        try:
            return response.status, json.loads(data)
        except ValueError:
            return response.status, None
    except (OSError, http.client.HTTPException):
        return None, None
    finally:
        connection.close()


class LinuxNvidiaHostDriver:
    """A Linux host with NVIDIA GPUs: the serving board is chosen by index (``CUDA_VISIBLE_DEVICES`` in
    PCI bus order, the order ``nvidia-smi`` lists), so a host with several boards and other users works."""

    def __init__(self, site: SiteManifest, *, popen: Callable = subprocess.Popen, run: Callable = subprocess.run,
                 request: Callable = _request, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, environ: Mapping[str, str] | None = None) -> None:
        self.site = site
        self._popen, self._run, self._request, self._clock, self._sleep = popen, run, request, clock, sleep
        self._environ = environ

    # -- facts and gates -------------------------------------------------------------------------
    def describe(self) -> Mapping[str, object]:
        return self.site.host.to_json()

    def gpu_state(self) -> Mapping[str, object]:
        return gpu_free(self.site.host.serving_gpu_index, run=self._run)

    def meter(self, *, interval_s: float = 0.1) -> HostMeter:
        return HostMeter(self.site.host.serving_gpu_index, interval_s=interval_s)

    def foreign_load(self, own_pids, *, limit_cores, window_s, on_trip) -> ForeignLoadWatch:
        return ForeignLoadWatch(own_pids, limit_cores=limit_cores, window_s=window_s, on_trip=on_trip)

    # -- the server ------------------------------------------------------------------------------
    def command(self, model_id, alias, port, policy, *, binding, helper_environment, output):
        """(argv, the environment this launch adds) -- what ``launch`` will run, without running it."""
        paths, host = self.site.paths, self.site.host
        if model_id not in paths.models:
            raise PhysicalAdapterError("the site has no model " + str(model_id))
        if (binding is None) != (not helper_environment):
            raise PhysicalAdapterError("a shared launch needs both the helper binding and its link environment")
        argv = [str(paths.server_path),
                *policy.arguments(paths.models[model_id], alias, port, host.threads,
                                  Path(output) / "slot-save" if policy.slot_save else None)]
        environment = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(host.serving_gpu_index),
                       "LD_LIBRARY_PATH": str(paths.server_path.parent)}
        if binding is not None:
            environment.update(shared_layer_environment(binding, policy))
            foreign = [name for name in helper_environment if not name.startswith(FFN_PREFIX)]
            if foreign:
                raise PhysicalAdapterError("helper link environment sets " + ",".join(sorted(foreign)))
            environment.update(helper_environment)
        if host.cpu_affinity is not None:
            argv = ["taskset", "--cpu-list", host.cpu_affinity, *argv]
        return argv, environment

    def launch(self, model_id, alias, port, policy, *, binding, helper_environment, output) -> ServerHandle:
        output = Path(output)
        argv, added = self.command(model_id, alias, port, policy, binding=binding,
                                   helper_environment=helper_environment, output=output)
        with socket.socket() as probe:          # refuse a busy port before anything is loaded
            probe.bind(("127.0.0.1", port))
        if policy.slot_save:
            (output / "slot-save").mkdir(exist_ok=True)
        # a variable inherited from the caller's shell must never configure the split
        environment = {name: value for name, value in (os.environ if self._environ is None else self._environ).items()
                       if not name.startswith(FFN_PREFIX)}
        environment.update(added)
        (output / SERVER_COMMAND).write_text(json.dumps({"argv": argv, "environment": added}, indent=2,
                                                        sort_keys=True) + "\n", encoding="ascii")
        with (output / SERVER_LOG).open("x") as log:
            process = self._popen(argv, env=environment, stdin=subprocess.DEVNULL, stdout=log,
                                  stderr=subprocess.STDOUT)
        return ServerHandle(process, port, tuple(argv), added, output / SERVER_LOG, policy.parallel)

    def wait_ready(self, handle: ServerHandle, timeout_s: float) -> None:
        deadline = self._clock() + timeout_s
        while True:
            if handle.process.poll() is not None:
                tail = handle.log_path.read_text(errors="replace")[-4000:] if handle.log_path.exists() else ""
                kind = "gpu-out-of-memory" if "out of memory" in tail else "exited"
                raise ServerLoadFailure(kind, f"exit status {handle.process.returncode}")
            status, _ = self._request(handle.port, "GET", "/health")
            if status == 200:
                return
            if self._clock() > deadline:
                raise ServerLoadFailure("timeout", f"not ready after {timeout_s:.0f} s")
            self._sleep(0.5)

    def tokenize(self, handle: ServerHandle, text: str) -> list[int]:
        """Token ids of a text in the loaded model's vocabulary."""
        status, reply = self._request(handle.port, "POST", "/tokenize", {"content": text, "add_special": True}, 60)
        if status != 200 or not isinstance(reply, dict) or not reply.get("tokens"):
            raise PhysicalAdapterError(f"tokenization failed (status {status})")
        return [int(token) for token in reply["tokens"]]

    def slots_busy(self, handle: ServerHandle) -> list[object]:
        """Ids of the slots still processing; an unreachable server counts as busy."""
        status, slots = self._request(handle.port, "GET", "/slots")
        if status != 200 or not isinstance(slots, list):
            return ["unreachable"]
        return [slot.get("id") for slot in slots if slot.get("is_processing")]

    def erase_slot(self, handle: ServerHandle, slot: int, timeout_s: float = 20) -> Mapping[str, object]:
        """Drop one slot's cells from the shared attention cache; a refusal is recorded, not raised."""
        status, reply = self._request(handle.port, "POST", f"/slots/{slot}?action=erase", {}, timeout_s)
        if status == 200 and isinstance(reply, dict):
            return reply
        return {"error": f"status {status}"}

    def stop(self, handle: ServerHandle) -> Mapping[str, object]:
        """End the server and read its own summary: exit status, phone calls, the per-share timing rows."""
        if handle.process.poll() is None:
            handle.process.send_signal(signal.SIGINT)
            try:
                handle.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                handle.process.kill()
                handle.process.wait(timeout=30)
        log = handle.log_path.read_text(errors="replace").splitlines() if handle.log_path.exists() else []
        return {"exit_status": handle.process.returncode,
                "phone_calls": sum(1 for line in log if "S41SERVERFFNCALL " in line),
                "shape_lines": [line for line in log if "S41SERVERFFNSHAPE " in line or "S41SERVERFFN {" in line]}

    def kill(self, handle: ServerHandle) -> None:
        if handle.process.poll() is None:
            handle.process.kill()
            handle.process.wait(timeout=30)
