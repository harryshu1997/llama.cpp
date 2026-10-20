#!/usr/bin/env python3
"""Stock ``ggml-rpc-server`` instances on the helper phones for the WS18 RPC whole-layer offload baseline.

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.rpc_phone_servers probe 127.0.0.1:50052 [...]
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.rpc_phone_servers preflight --phones PHONES.json
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.rpc_phone_servers meminfo --phones PHONES.json

The baseline is llama.cpp's own RPC backend: each phone runs the upstream ``tools/rpc`` server (built for Android
arm64 without changes) on its loopback, and the desktop's ``llama-server --rpc`` reaches it over USB TCP:
``adb forward`` (both phones; the OP15 must be in its normal Android USB configuration, i.e. NOT inside a FunctionFS
session) or, for the Pixel, the WS10 AOA bridge (a generic TCP carrier: host bridge ``serve --listen-port`` + phone
relay ``--worker-port``). This module only starts, probes and stops those servers; it never reboots, flashes or
switches a phone's kernel, and it sends SIGTERM only to the pid it started.

``PHONES.json`` (schema ``ws18-rpc-phones-v1``) lists the phones in ``--rpc`` order (RPC0 first):

    {"schema": "ws18-rpc-phones-v1", "phones": [
      {"name": "op15", "serial": "3C15AU002CL00000", "adb_port": 5037, "phone_dir": "/data/local/tmp/s43-rpc-v1",
       "phone_port": 50052, "host_port": 50052, "transport": "adb-forward", "threads": null, "devices": null,
       "cache": true, "as_root": true, "min_mem_available_bytes": 9000000000,
       "expected_sha256": {"ggml-rpc-server": "...", "libggml.so": "...", ...}}, ...]}

The RPC wire format used by ``probe`` (``ggml/src/ggml-rpc/ggml-rpc.cpp``): request = cmd (1 byte) | size (u64) |
data, response = size (u64) | data. A connection must start with HELLO (cmd 14, 24 capability bytes; zero = plain
TCP); the reply is major, minor, patch, pad, 24 capability bytes. DEVICE_COUNT is cmd 15, GET_DEVICE_MEMORY cmd 11.
The server serves one client at a time, so a probe is short and closes its connection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import shlex
import socket
import struct
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

PHONES_SCHEMA = "ws18-rpc-phones-v1"
RECEIPT_SCHEMA = "ws18-rpc-phone-servers-receipt-v1"
SERVER_BINARY = "ggml-rpc-server"
CMD_HELLO, CMD_DEVICE_COUNT, CMD_GET_DEVICE_MEMORY = 14, 15, 11
CONN_CAPS = 24
TRANSPORTS = ("adb-forward", "aoa-bridge")

Runner = Callable[[Sequence[str], float], subprocess.CompletedProcess]


class RpcPhoneError(RuntimeError):
    pass


def run_command(argv: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout_s, check=False,
                          stdin=subprocess.DEVNULL)


# ---------------------------------------------------------------- RPC probe

def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise RpcPhoneError("rpc server closed the connection")
        data += chunk
    return data


def _command(sock: socket.socket, cmd: int, payload: bytes, response_size: int) -> bytes:
    sock.sendall(bytes([cmd]) + struct.pack("<Q", len(payload)) + payload)
    size = struct.unpack("<Q", _recv_exact(sock, 8))[0]
    if size != response_size:
        raise RpcPhoneError("rpc response size %d != %d (cmd %d)" % (size, response_size, cmd))
    return _recv_exact(sock, size)


def probe(endpoint: str, timeout_s: float = 5.0) -> dict[str, Any]:
    """HELLO + device count + memory of every device of one ``host:port`` rpc server."""
    host, _, port = endpoint.rpartition(":")
    with socket.create_connection((host, int(port)), timeout=timeout_s) as sock:
        hello = _command(sock, CMD_HELLO, bytes(CONN_CAPS), 4 + CONN_CAPS)
        count = struct.unpack("<I", _command(sock, CMD_DEVICE_COUNT, b"", 4))[0]
        devices = []
        for index in range(count):
            free, total = struct.unpack("<QQ", _command(sock, CMD_GET_DEVICE_MEMORY, struct.pack("<I", index), 16))
            devices.append({"device": index, "free_bytes": free, "total_bytes": total})
    return {"endpoint": endpoint, "protocol": "%d.%d.%d" % (hello[0], hello[1], hello[2]),
            "device_count": count, "devices": devices}


def wait_for_probe(endpoint: str, timeout_s: float, *, interval_s: float = 0.5,
                   probe_fn: Callable[[str], Mapping[str, Any]] = probe) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            return dict(probe_fn(endpoint))
        except (OSError, RpcPhoneError, struct.error) as error:
            last = error
            time.sleep(interval_s)
    raise RpcPhoneError("rpc server %s did not answer within %.0f s: %s" % (endpoint, timeout_s, last))


# ---------------------------------------------------------------- phone configuration

@dataclass(frozen=True)
class PhoneRpcServer:
    name: str
    serial: str
    adb_port: int
    phone_dir: str
    phone_port: int
    host_port: int
    expected_sha256: Mapping[str, str]
    transport: str = "adb-forward"
    threads: int | None = None
    devices: str | None = None
    cache: bool = True
    as_root: bool = True
    min_mem_available_bytes: int = 0
    aoa: Mapping[str, Any] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        return "127.0.0.1:%d" % self.host_port


def load_phones(path: Path) -> list[PhoneRpcServer]:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != PHONES_SCHEMA:
        raise RpcPhoneError("phones file schema must be " + PHONES_SCHEMA)
    phones = []
    for row in raw.get("phones", ()):
        phone = PhoneRpcServer(
            name=str(row["name"]), serial=str(row["serial"]), adb_port=int(row.get("adb_port", 5037)),
            phone_dir=str(row["phone_dir"]), phone_port=int(row["phone_port"]), host_port=int(row["host_port"]),
            expected_sha256=dict(row["expected_sha256"]), transport=str(row.get("transport", "adb-forward")),
            threads=None if row.get("threads") is None else int(row["threads"]),
            devices=row.get("devices"), cache=bool(row.get("cache", True)), as_root=bool(row.get("as_root", True)),
            min_mem_available_bytes=int(row.get("min_mem_available_bytes", 0)), aoa=dict(row.get("aoa") or {}))
        validate_phone(phone)
        phones.append(phone)
    if not phones:
        raise RpcPhoneError("phones file lists no phone")
    if len({p.host_port for p in phones}) != len(phones) or len({p.serial for p in phones}) != len(phones):
        raise RpcPhoneError("phones need distinct serials and host ports")
    return phones


def validate_phone(phone: PhoneRpcServer) -> None:
    if phone.transport not in TRANSPORTS:
        raise RpcPhoneError("%s: transport must be one of %s" % (phone.name, ", ".join(TRANSPORTS)))
    if not phone.phone_dir.startswith("/data/local/tmp/") or any(c in phone.phone_dir for c in " '\"$;&|`"):
        raise RpcPhoneError("%s: phone_dir must be a plain path under /data/local/tmp/" % phone.name)
    if SERVER_BINARY not in phone.expected_sha256:
        raise RpcPhoneError("%s: expected_sha256 must pin %s" % (phone.name, SERVER_BINARY))
    for name, digest in phone.expected_sha256.items():
        if "/" in name or not name or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RpcPhoneError("%s: expected_sha256 needs bare file names and hex sha256 values" % phone.name)
    if phone.devices is not None and not all(part.replace("-", "").isalnum() for part in phone.devices.split(",")):
        raise RpcPhoneError("%s: devices must be a comma list of device names" % phone.name)
    if phone.transport == "aoa-bridge":
        for key in ("bridge_script", "sysfs_device", "relay_path"):
            if not phone.aoa.get(key):
                raise RpcPhoneError("%s: aoa-bridge transport needs aoa.%s" % (phone.name, key))


# ---------------------------------------------------------------- adb commands

def adb(phone: PhoneRpcServer, *args: str) -> list[str]:
    return ["adb", "-P", str(phone.adb_port), "-s", phone.serial, *args]


def shell(phone: PhoneRpcServer, command: str) -> list[str]:
    return adb(phone, "shell", "su -c " + shlex.quote(command) if phone.as_root else command)


def server_command(phone: PhoneRpcServer) -> str:
    """The phone shell line that starts the stock server in the background and prints its pid."""
    directory = phone.phone_dir
    argv = ["./" + SERVER_BINARY, "-H", "127.0.0.1", "-p", str(phone.phone_port)]
    if phone.threads is not None:
        argv += ["-t", str(phone.threads)]
    if phone.cache:
        argv += ["-c"]
    if phone.devices:
        argv += ["-d", phone.devices]
    environment = "LD_LIBRARY_PATH=%s LLAMA_CACHE=%s/cache" % (directory, directory)
    # 2026-10-06 (first phone run): ``nohup ... & echo $!`` never returns over ``adb shell`` on the OnePlus 15 and
    # the Pixel (adbd keeps waiting for the stdout pipe the background job holds), with or without su. A subshell
    # double fork under ``setsid`` returns at once; the pid then comes from ``pidof`` after a short settle (the
    # preflight has refused if a server was already running, so there is exactly one).
    return ("cd %s && (%s setsid %s > %s/rpc-server.log 2>&1 < /dev/null &) ; sleep 0.5; pidof %s"
            % (directory, environment, " ".join(argv), directory, SERVER_BINARY))


def _checked(runner: Runner, argv: Sequence[str], timeout_s: float, what: str) -> str:
    completed = runner(argv, timeout_s)
    if completed.returncode != 0:
        raise RpcPhoneError("%s failed (rc %s): %s" % (what, completed.returncode,
                                                      (completed.stderr or completed.stdout or "").strip()[-400:]))
    return completed.stdout or ""


def mem_available_bytes(meminfo: str) -> int:
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RpcPhoneError("MemAvailable missing from /proc/meminfo")


def preflight(phone: PhoneRpcServer, runner: Runner = run_command) -> dict[str, Any]:
    """Refuses unless the phone is an adb ``device``, the pinned files are in place, no rpc server runs and the
    phone has the memory its planned layers need."""
    state = _checked(runner, adb(phone, "get-state"), 20, phone.name + " get-state").strip()
    if state != "device":
        raise RpcPhoneError("%s: adb state is %r (the OP15 must be in its normal Android USB configuration, "
                            "not inside a FunctionFS session)" % (phone.name, state))
    names = sorted(phone.expected_sha256)
    listing = _checked(runner, shell(phone, "cd %s && sha256sum %s" % (phone.phone_dir, " ".join(names))), 60,
                       phone.name + " sha256sum")
    found = {}
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) == 2:
            found[parts[1].lstrip("*").split("/")[-1]] = parts[0]
    mismatched = [name for name in names if found.get(name) != phone.expected_sha256[name]]
    if mismatched:
        raise RpcPhoneError("%s: pinned files differ or are missing: %s" % (phone.name, ", ".join(mismatched)))
    running = runner(shell(phone, "pidof " + SERVER_BINARY), 20)
    if (running.stdout or "").strip():
        raise RpcPhoneError("%s: an rpc server already runs (pid %s)" % (phone.name, running.stdout.strip()))
    available = mem_available_bytes(_checked(runner, adb(phone, "shell", "cat /proc/meminfo"), 20,
                                             phone.name + " meminfo"))
    if available < phone.min_mem_available_bytes:
        raise RpcPhoneError("%s: MemAvailable %d < required %d" % (phone.name, available,
                                                                   phone.min_mem_available_bytes))
    return {"name": phone.name, "serial": phone.serial, "adb_state": state, "files_sha256": found,
            "mem_available_bytes": available, "transport": phone.transport}


@dataclass
class StartedServer:
    phone: PhoneRpcServer
    pid: int
    started_ns: int
    probe: dict[str, Any] = field(default_factory=dict)
    bridge: Any = None
    relay: Any = None

    def receipt(self) -> dict[str, Any]:
        return {"name": self.phone.name, "pid": self.pid, "started_ns": self.started_ns, "probe": self.probe,
                "endpoint": self.phone.endpoint, "transport": self.phone.transport,
                "command": server_command(self.phone)}


def start(phone: PhoneRpcServer, *, runner: Runner = run_command,
          popen: Callable[..., Any] = subprocess.Popen, out_dir: Path | None = None,
          probe_fn: Callable[[str], Mapping[str, Any]] = probe, ready_timeout_s: float = 60.0) -> StartedServer:
    output = _checked(runner, shell(phone, server_command(phone)), 30, phone.name + " start")
    pids = [token for token in output.split() if token.isdigit()]
    if not pids:
        raise RpcPhoneError("%s: the start command printed no pid: %r" % (phone.name, output))
    started = StartedServer(phone, int(pids[-1]), time.monotonic_ns())
    try:
        if phone.transport == "adb-forward":
            _checked(runner, adb(phone, "forward", "tcp:%d" % phone.host_port, "tcp:%d" % phone.phone_port), 20,
                     phone.name + " adb forward")
        else:
            started.relay, started.bridge = _start_aoa(phone, popen, out_dir)
        started.probe = wait_for_probe(phone.endpoint, ready_timeout_s, probe_fn=probe_fn)
    except Exception:
        stop(started, runner=runner)
        raise
    return started


def _start_aoa(phone: PhoneRpcServer, popen: Callable[..., Any], out_dir: Path | None):
    """WS10 AOA bridge as a plain TCP carrier: phone relay -> rpc server port; host bridge owns host_port. The
    Pixel must already be in accessory+adb mode (``aoa_bridge.py switch``, a separate runbook step)."""
    aoa = phone.aoa
    log_dir = Path(out_dir) if out_dir is not None else Path(".")
    relay_argv = adb(phone, "shell", "-T", "su -c " + shlex.quote(
        "exec %s --worker-port %d" % (aoa["relay_path"], phone.phone_port)))
    relay = popen(relay_argv, stdout=open(log_dir / ("aoa-relay-%s.log" % phone.name), "wb"),
                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    ready = log_dir / ("aoa-bridge-%s.ready" % phone.name)
    bridge_argv = [aoa.get("python", sys.executable), aoa["bridge_script"], "serve",
                   "--sysfs-device", aoa["sysfs_device"], "--serial", phone.serial,
                   "--listen-port", str(phone.host_port), "--ready-file", str(ready)]
    bridge = popen(bridge_argv, stdout=open(log_dir / ("aoa-bridge-%s.log" % phone.name), "wb"),
                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    return relay, bridge


def stop(started: StartedServer, *, runner: Runner = run_command, out_dir: Path | None = None,
         exit_timeout_s: float = 20.0) -> dict[str, Any]:
    """SIGTERM the started pid (never SIGKILL), remove the transport, keep the server log; report what is left."""
    phone = started.phone
    receipt: dict[str, Any] = {"name": phone.name, "pid": started.pid}
    runner(shell(phone, "kill -TERM %d" % started.pid), 20)
    deadline = time.monotonic() + exit_timeout_s
    exited = False
    while time.monotonic() < deadline:
        if runner(shell(phone, "kill -0 %d" % started.pid), 20).returncode != 0:
            exited = True
            break
        time.sleep(0.5)
    receipt["exited"] = exited
    if phone.transport == "adb-forward":
        runner(adb(phone, "forward", "--remove", "tcp:%d" % phone.host_port), 20)
    else:
        for process in (started.bridge, started.relay):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        runner(shell(phone, "kill -TERM $(pidof s43-aoa-relay) 2>/dev/null; true"), 20)
    log = runner(shell(phone, "cat %s/rpc-server.log" % phone.phone_dir), 30)
    if out_dir is not None:
        (Path(out_dir) / ("rpc-server-%s.log" % phone.name)).write_text(log.stdout or "")
    leftover = (runner(shell(phone, "pidof " + SERVER_BINARY), 20).stdout or "").strip()
    receipt["leftover_pids"] = leftover
    receipt["clean"] = exited and not leftover
    receipt["stopped_ns"] = time.monotonic_ns()
    return receipt


# ---------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_probe = sub.add_parser("probe", help="HELLO + devices of rpc servers (host:port ...)")
    p_probe.add_argument("endpoints", nargs="+")
    for name in ("preflight", "meminfo"):
        p = sub.add_parser(name)
        p.add_argument("--phones", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "probe":
        print(json.dumps([probe(endpoint) for endpoint in args.endpoints], indent=2, sort_keys=True))
        return 0
    phones = load_phones(args.phones)
    if args.command == "preflight":
        print(json.dumps([preflight(phone) for phone in phones], indent=2, sort_keys=True))
        return 0
    rows = []
    for phone in phones:
        text = _checked(run_command, adb(phone, "shell", "cat /proc/meminfo"), 20, phone.name + " meminfo")
        rows.append({"name": phone.name, "mem_available_bytes": mem_available_bytes(text)})
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
