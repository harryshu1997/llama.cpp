#!/usr/bin/env python3
"""WS10 Phase B: cadence-matched A/B of the Pixel helper link, adb forward vs the AOA bridge.

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.pixel_transport_ab plan    CONFIG.json
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.pixel_transport_ab run     CONFIG.json OUT [--only a,b]
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.pixel_transport_ab analyze OUT [--reference adb]

Run from the repository root under the rig lock (it launches the Pixel worker, relay and bridge; OP15 is never
touched). Every arm uses the scheduler's own session class for its transport (AdbTcpPhoneWorkerSession or
AoaBridgePhoneWorkerSession) with an exact finite budget, so the stop is the normal budget exit.

Arm kinds
  ffn   real FFN worker (DVFS-fixed, same shard, same env) at the serving cadence: ``calls_per_token`` layer calls
        separated by ``gap_call_ms`` (host attention between the Pixel's consecutive layers), then a
        ``gap_token_ms`` [low, high] uniform pause (rest of the token); rows 1/2/4 per segment. Records rpc,
        worker compute (from the EXEC response), non-compute overhead and the call's position in its token;
        dumps the first measured token of every segment for the byte-identity receipt.
  echo  transport only: the relay's phone-local TCP echo server, reached through adb forward (``adb-tcp``), through
        the bridge and the relay's TCP hop (``aoa-bridge``) or relay-local echo streams (``aoa-local``): the three
        together isolate the phone TCP hop and the link.
USB mode: every arm runs with the Pixel in ``usb_mode`` (config default "accessory" = accessory+adb 18d1:2d01, so
adb and AOA arms share one USB configuration; "normal" restores 18d1:4ee7 first = the production adb setting).
  lifecycle  (aoa-bridge) the elastic-helper path on hardware: resident worker, two sequential server connections
        (re-arm), liveness, a drop (``drop``: "bridge-kill" or "usb-reset" = a real accessory exit), loss detection,
        release, and a re-join on the same session object (switch, launch, serve), then the idle stop; writes
        ``aoa-bridge-scheduler-launched-session.json`` (the scheduler-launched-session receipt).
Keep-awake factors (per arm): relay_options / bridge_options (AOA only: QoS window, pinning, uclamp, keep-alive
NOPs) and ``phone_qos_latency_us`` (a separate rooted holder of /dev/cpu_dma_latency for the whole arm, usable
with adb too). Host USB LPM is recorded (read-only) before and after every arm; changing it needs root (RUNBOOK).

``analyze`` writes AB_TABLE.md/.json and, when every ffn arm's dumps agree, the receipts
``aoa-bridge-byte-identity.json`` and ``aoa-bridge-round-trip.json`` (link calibration of the best qualified AOA
arm in the same schema the adb-forward calibration used).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import socket
import statistics
import subprocess
import sys
import time
from typing import Callable, Mapping

from research_dev.scheduler.adapters import aoa_bridge
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler.adapters.phone_aoa_session import AoaBridgeConfiguration, AoaBridgePhoneWorkerSession
from research_dev.scheduler.adapters.phone_tcp_session import (
    EXECUTE_REQUEST, EXECUTE_RESPONSE, HELLO_REQUEST, HELLO_RESPONSE, PROTOCOL_MAGIC, PROTOCOL_VERSION,
    AdbTcpPhoneWorkerSession, AdbTcpWorkerConfiguration, _fnv32,
)

SCHEMA = "ws10-pixel-transport-ab-v1"
TRANSPORTS = ("adb-tcp", "aoa-bridge", "aoa-local")
LOGICAL_ROW_BYTES = 10240


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def quantile(values: list[float], q: float) -> float | None:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))] if ordered else None


def receive(stream: socket.socket, count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        part = stream.recv(count - len(data))
        if not part:
            raise PhysicalAdapterError("the endpoint closed the connection early")
        data.extend(part)
    return bytes(data)


# --- configuration --------------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema") != SCHEMA:
        raise SystemExit("config schema must be " + SCHEMA)
    names = [arm["name"] for arm in config["arms"]]
    if len(set(names)) != len(names):
        raise SystemExit("arm names must be unique")
    for arm in config["arms"]:
        if arm.get("kind", "ffn") not in ("ffn", "echo", "lifecycle") or arm["transport"] not in TRANSPORTS:
            raise SystemExit("arm kind/transport is invalid: " + arm["name"])
        if arm["transport"] == "aoa-local" and arm.get("kind", "ffn") != "echo":
            raise SystemExit("aoa-local is an echo transport")
        if arm.get("kind") == "lifecycle" and (arm["transport"] != "aoa-bridge"
                                               or arm.get("drop", "bridge-kill") not in ("bridge-kill", "usb-reset")):
            raise SystemExit("a lifecycle arm is aoa-bridge with drop bridge-kill or usb-reset")
    return config


def arm_order(config: dict) -> list[dict]:
    """ABBA over the configured arm list, ``repeats`` times; every run gets a unique directory name.
    Lifecycle arms run once, after the measured arms."""
    arms = [arm for arm in config["arms"] if arm.get("kind") != "lifecycle"]
    lifecycles = [arm for arm in config["arms"] if arm.get("kind") == "lifecycle"]
    sequence = []
    for repeat in range(config.get("repeats", 2)):
        block = arms if repeat % 2 == 0 else list(reversed(arms))
        base = len(sequence)
        sequence.extend([{**arm, "run": f"{base + index + 1:02d}-{arm['name']}"} for index, arm in enumerate(block)])
    base = len(sequence)
    sequence.extend([{**arm, "run": f"{base + index + 1:02d}-{arm['name']}"} for index, arm in enumerate(lifecycles)])
    return sequence


def calls_per_arm(config: dict) -> int:
    cadence = config["cadence"]
    return sum((segment["warmup_tokens"] + segment["tokens"]) * cadence["calls_per_token"]
               for segment in config["segments"])


def estimate_seconds(config: dict) -> float:
    cadence = config["cadence"]
    low, high = cadence["gap_token_ms"]
    per_token = (cadence["calls_per_token"] - 1) * cadence["gap_call_ms"] + (low + high) / 2
    ffn = {1: 14.0, 2: 21.0, 4: 33.0}  # measured adb per-layer RPC (ms) of the DVFS-fixed worker
    total = 0.0
    for arm in arm_order(config):
        for segment in config["segments"]:
            tokens = segment["warmup_tokens"] + segment["tokens"]
            rpc = ffn.get(segment["rows"], 35.0) if arm.get("kind", "ffn") == "ffn" else 3.0
            total += tokens * (per_token + cadence["calls_per_token"] * rpc) / 1000
        total += config.get("cooldown_s", 20) + 40  # launch, switch/handshake, stop
    return total


def worker_configuration(config: dict, arm: dict) -> AdbTcpWorkerConfiguration:
    budget = 0 if arm.get("kind") == "lifecycle" else calls_per_arm(config)
    phone = config["phone"]
    hashes = dict(phone["common_sha256"])
    hashes[phone["worker_path"]] = phone["worker_sha256"]
    return AdbTcpWorkerConfiguration(
        device_id=phone.get("device_id", "pixel10pro-phone"), serial=phone["serial"], adb_port=phone.get("adb_port", 5037),
        adb_path=Path(phone.get("adb_path", "/usr/bin/adb")), worker_path=phone["worker_path"],
        library_directories=tuple(phone["library_directories"]), shard_path=phone["shard_path"],
        artifact_sha256=phone["artifact_sha256"], layer_mask=sum(1 << layer for layer in phone["layers"]),
        n_embd=phone["n_embd"], columns=phone["columns"], column_quantum=phone["column_quantum"],
        max_tokens=phone["max_tokens"], swiglu=True, backend=phone.get("backend", "CPU"),
        phone_port=phone["phone_port"], forward_port=config["forward_port"], max_requests=budget,
        worker_environment=phone["worker_environment"], expected_sha256_by_path=hashes,
        as_root=phone.get("as_root", True), phone_lock_path=phone.get("phone_lock_path"),
        launch_timeout_s=phone.get("launch_timeout_s", 240.0))


def bridge_configuration(config: dict, arm: dict) -> AoaBridgeConfiguration:
    base = dict(config["aoa_bridge"])
    base["relay_options"] = {**base.get("relay_options", {}), **arm.get("relay_options", {})}
    base["bridge_options"] = {**base.get("bridge_options", {}), **arm.get("bridge_options", {})}
    base["trace"] = bool(arm.get("trace", base.get("trace", False)))
    return AoaBridgeConfiguration.from_json(base)


# --- phone / host state (read-only) ---------------------------------------------------------------

PHONE_STATE_SCRIPT = (
    "for s in /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state*; do "
    "echo \"idle $s $(cat $s/name) $(cat $s/latency) $(cat $s/usage) $(cat $s/time) $(cat $s/disable)\"; done; "
    "for p in /sys/devices/system/cpu/cpufreq/policy*; do echo \"freq $p $(cat $p/scaling_cur_freq)\"; done; "
    "grep -i -E 'dwc3|usb' /proc/interrupts | sed 's/^/irq /'; cat /proc/sys/kernel/random/boot_id")


class Harness:
    """Injectable process/phone access (tests replace adb, popen, sessions and the USB observer)."""

    def __init__(self, config: dict, *, run: Callable = subprocess.run, popen: Callable = subprocess.Popen,
                 session_options: Mapping[str, object] | None = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], int] = time.monotonic_ns, sysfs_root: Path = aoa_bridge.USB_SYSFS_ROOT,
                 relay_extra: tuple[str, ...] = (), bridge_extra: tuple[str, ...] = ()) -> None:
        self.config, self._run, self._popen = config, run, popen
        self.session_options = dict(session_options or {})
        self.sleep, self.clock, self.sysfs_root = sleep, clock, sysfs_root
        self.relay_extra, self.bridge_extra = relay_extra, bridge_extra
        phone = config["phone"]
        self.adb = [phone.get("adb_path", "/usr/bin/adb"), "-P", str(phone.get("adb_port", 5037)), "-s", phone["serial"]]

    def shell(self, command: str, *, root: bool = True, timeout: float = 60) -> str:
        text = "su -c " + shlex.quote(command) if root and self.config["phone"].get("as_root", True) else command
        return self._run(self.adb + ["shell", text], check=True, capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, timeout=timeout).stdout

    def phone_state(self) -> str:
        try:
            return self.shell(PHONE_STATE_SCRIPT)
        except (OSError, subprocess.SubprocessError) as error:
            return "unavailable: " + repr(error)

    def host_usb(self) -> dict:
        device = self.config["aoa_bridge"]["usb_sysfs_device"]
        try:
            state = aoa_bridge.read_usb_device(device, sysfs_root=self.sysfs_root).to_json()
        except aoa_bridge.BridgeError as error:
            state = {"error": str(error)}
        usbfs = Path("/sys/module/usbcore/parameters/usbfs_memory_mb")
        return {"device": state, "lpm": aoa_bridge.lpm_state(device, sysfs_root=self.sysfs_root),
                "usbfs_memory_mb": usbfs.read_text().strip() if usbfs.exists() else None}

    def session(self, arm: dict):
        worker = worker_configuration(self.config, arm)
        if arm["transport"] == "adb-tcp":
            return AdbTcpPhoneWorkerSession(worker, **{k: v for k, v in self.session_options.items()
                                                       if k in ("run", "popen", "connect", "sleep", "clock")})
        accepted = ("run", "popen", "connect", "sleep", "clock", "wall_clock", "observe_usb", "switch_mode",
                    "bridge_script")
        return AoaBridgePhoneWorkerSession(worker, bridge_configuration(self.config, arm),
                                           extra_relay_arguments=self.relay_extra,
                                           extra_bridge_arguments=self.bridge_extra,
                                           **{k: v for k, v in self.session_options.items() if k in accepted})

    def ensure_usb_mode(self, mode: str) -> dict:
        """Put the pinned Pixel in the arm's USB mode (selection by port path + serial only) and wait for adb."""
        bridge = AoaBridgeConfiguration.from_json(self.config["aoa_bridge"])
        serial = self.config["phone"]["serial"]
        observe = self.session_options.get("observe_usb") or (
            lambda device: aoa_bridge.read_usb_device(device, sysfs_root=self.sysfs_root))
        state = observe(bridge.usb_sysfs_device)
        aoa_bridge.check_pixel(state, serial=serial, forbidden_serials=bridge.forbidden_serials, require_accessory=None)
        if mode == "any" or (mode == "accessory") == state.accessory:
            return {"mode": mode, "changed": False, "usb": state.to_json()}
        if mode == "accessory":
            change = self.session_options.get("switch_mode") or aoa_bridge.switch_to_accessory
        elif mode == "normal":
            change = self.session_options.get("restore_mode") or aoa_bridge.restore_normal
        else:
            raise SystemExit("usb_mode must be accessory, normal or any")
        result = dict(change(bridge.usb_sysfs_device, serial, forbidden_serials=bridge.forbidden_serials,
                             timeout_s=bridge.mode_switch_timeout_s))
        deadline = time.monotonic() + bridge.mode_switch_timeout_s
        while self._run(self.adb + ["get-state"], check=False, capture_output=True, text=True,
                        stdin=subprocess.DEVNULL, timeout=10).stdout.strip() != "device":
            if time.monotonic() > deadline:
                raise PhysicalAdapterError("adb did not return after the USB mode change")
            time.sleep(0.25)
        return {"mode": mode, "changed": True, **result}

    # --- optional whole-arm phone QoS holder (works with either transport) --------------------------
    HOLDER_SLEEP = "sleep 86417"

    def start_qos_holder(self, latency_us: int, log: Path) -> subprocess.Popen:
        script = f"exec 3>/dev/cpu_dma_latency; printf '%x' {int(latency_us)} >&3; exec {self.HOLDER_SLEEP}"
        with log.open("x") as handle:
            return self._popen(self.adb + ["shell", "-T", "su -c " + shlex.quote(script)],
                               stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)

    def stop_qos_holder(self, process: subprocess.Popen) -> dict:
        signalled = []
        try:
            for line in self.shell("ps -A -o PID,ARGS").splitlines():
                fields = line.split(None, 1)
                if len(fields) == 2 and fields[0].isdigit() and fields[1].strip() == self.HOLDER_SLEEP:
                    self.shell(f"kill -TERM {fields[0]}")
                    signalled.append(int(fields[0]))
        except (OSError, subprocess.SubprocessError) as error:
            return {"error": repr(error)}
        try:
            code = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            code = process.wait(timeout=15)
        return {"signalled": signalled, "exit_code": code}


# --- drivers -------------------------------------------------------------------------------------

def load_inputs(config: dict) -> dict[tuple[int, int], bytes]:
    directory = Path(config["input_dir"])
    phone = config["phone"]
    inputs = {}
    for layer in phone["layers"]:
        for row in range(max(segment["rows"] for segment in config["segments"])):
            data = (directory / f"input-layer{layer}-row{row}.f16").read_bytes()
            if len(data) != 2 * phone["n_embd"]:
                raise SystemExit(f"input layer {layer} row {row} has the wrong size")
            inputs[(layer, row)] = data
    return inputs


def pause_schedule(config: dict, rng: random.Random, position: int) -> float:
    cadence = config["cadence"]
    if position + 1 < cadence["calls_per_token"]:
        return cadence["gap_call_ms"] / 1000
    return rng.uniform(*cadence["gap_token_ms"]) / 1000


def drive_ffn(port: int, config: dict, inputs: dict, harness: Harness) -> tuple[list[dict], dict[str, bytes]]:
    phone, cadence = config["phone"], config["cadence"]
    layers = phone["layers"]
    rng = random.Random(cadence.get("seed", 1))
    rows, dumps = [], {}
    with socket.create_connection(("127.0.0.1", port), timeout=120) as stream:
        stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        stream.sendall(HELLO_REQUEST.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, 1, sum(1 << layer for layer in layers),
                                          phone["n_embd"], phone["columns"], 3, phone["max_tokens"],
                                          bytes.fromhex(phone["artifact_sha256"][7:])))
        hello = HELLO_RESPONSE.unpack(receive(stream, HELLO_RESPONSE.size))
        if hello[:4] != (PROTOCOL_MAGIC, PROTOCOL_VERSION, 2, 0):
            raise PhysicalAdapterError(f"HELLO refused: {hello[:4]}")
        request_id = 0
        for segment in config["segments"]:
            count = segment["rows"]
            dump = bytearray()
            for token in range(segment["warmup_tokens"] + segment["tokens"]):
                for position in range(cadence["calls_per_token"]):
                    layer = layers[position % len(layers)]
                    payload = b"".join(inputs[(layer, row)] for row in range(count))
                    request_id += 1
                    started = harness.clock()
                    stream.sendall(EXECUTE_REQUEST.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, 3, request_id, layer,
                                                        phone["n_embd"] * count, len(payload), _fnv32(payload),
                                                        phone["columns"], count) + payload)
                    response = EXECUTE_RESPONSE.unpack(receive(stream, EXECUTE_RESPONSE.size))
                    output = receive(stream, len(payload))
                    elapsed_us = (harness.clock() - started) / 1000
                    if response[:6] != (PROTOCOL_MAGIC, PROTOCOL_VERSION, 4, 0, 0, request_id) \
                            or response[9] != _fnv32(output):
                        raise PhysicalAdapterError(f"EXEC {request_id} failed: {response}")
                    rows.append({"segment": segment["name"], "rows": count, "token": token - segment["warmup_tokens"],
                                 "position": position, "layer": layer, "rpc_us": elapsed_us,
                                 "compute_us": response[-1], "overhead_us": elapsed_us - response[-1],
                                 "output_fnv": response[9], "started_ns": started})
                    if token == segment["warmup_tokens"]:
                        dump.extend(output)
                    harness.sleep(pause_schedule(config, rng, position))
            dumps[segment["name"]] = bytes(dump)
    return rows, dumps


def drive_echo(port: int, config: dict, harness: Harness) -> list[dict]:
    cadence = config["cadence"]
    rng = random.Random(cadence.get("seed", 1))
    payload_rng = random.Random(7)
    rows = []
    with socket.create_connection(("127.0.0.1", port), timeout=60) as stream:
        stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for segment in config["segments"]:
            size = LOGICAL_ROW_BYTES * segment["rows"] + 40
            for token in range(segment["warmup_tokens"] + segment["tokens"]):
                for position in range(cadence["calls_per_token"]):
                    payload = payload_rng.randbytes(size)
                    started = harness.clock()
                    stream.sendall(payload)
                    echoed = receive(stream, size)
                    elapsed_us = (harness.clock() - started) / 1000
                    if echoed != payload:
                        raise PhysicalAdapterError("echo payload differs")
                    rows.append({"segment": segment["name"], "rows": segment["rows"],
                                 "token": token - segment["warmup_tokens"], "position": position, "rpc_us": elapsed_us,
                                 "compute_us": 0, "overhead_us": elapsed_us, "started_ns": started})
                    harness.sleep(pause_schedule(config, rng, position))
    return rows


class EchoEndpoint:
    """Relay with its phone-local echo server; host side = adb forward, the bridge's worker port (relay TCP hop)
    or the bridge's echo port (relay-local echo, no phone TCP hop)."""

    def __init__(self, harness: Harness, arm: dict, directory: Path) -> None:
        self.harness, self.arm, self.directory = harness, arm, directory
        self.config = harness.config
        self.bridge = bridge_configuration(self.config, arm)
        self.relay = self.bridge_process = None
        self.forward = None

    def start(self) -> int:
        harness, config, bridge = self.harness, self.config, self.bridge
        echo_port = config["echo_phone_port"]
        state = aoa_bridge.read_usb_device(bridge.usb_sysfs_device, sysfs_root=harness.sysfs_root) \
            if harness.session_options.get("observe_usb") is None else harness.session_options["observe_usb"](
                bridge.usb_sysfs_device)
        aoa_bridge.check_pixel(state, serial=config["phone"]["serial"], forbidden_serials=bridge.forbidden_serials,
                               require_accessory=True)
        relay_argv = (*bridge.relay_argv(echo_port), "--echo-server", str(echo_port), *harness.relay_extra)
        command = "exec " + shlex.join(relay_argv)
        if bridge.relay_lock_path:
            command = f"exec 9>{shlex.quote(bridge.relay_lock_path)}\nflock -n 9 9>&9 || exit 73\n" + command
        if config["phone"].get("as_root", True):
            command = "su -c " + shlex.quote(command)
        log = self.directory / "relay.log"
        with log.open("x") as handle:
            self.relay = harness._popen(harness.adb + ["shell", "-T", command], stdin=subprocess.DEVNULL,
                                        stdout=handle, stderr=subprocess.STDOUT)
        self._wait(lambda: "[s43-aoa-relay] ready " in log.read_text(errors="replace"), self.relay, log)
        if self.arm["transport"] == "adb-tcp":
            self.forward = harness._run(harness.adb + ["forward", "--no-rebind", f"tcp:{config['forward_port']}",
                                                       f"tcp:{echo_port}"], check=True, capture_output=True,
                                        text=True, stdin=subprocess.DEVNULL, timeout=30)
            return config["forward_port"]
        ready = self.directory / "bridge-ready.json"
        extra = ("--echo-port", str(config["echo_host_port"])) if self.arm["transport"] == "aoa-local" else ()
        script = harness.session_options.get("bridge_script") or Path(aoa_bridge.__file__)
        argv = (*bridge.bridge_argv(Path(script), config["phone"]["serial"], config["forward_port"], ready,
                                    self.directory / "bridge-status.json",
                                    self.directory / "bridge-trace.jsonl" if bridge.trace else None),
                *extra, *harness.bridge_extra)
        blog = self.directory / "bridge.log"
        with blog.open("x") as handle:
            self.bridge_process = harness._popen(argv, stdin=subprocess.DEVNULL, stdout=handle,
                                                 stderr=subprocess.STDOUT)
        self._wait(ready.exists, self.bridge_process, blog)
        return config["echo_host_port"] if self.arm["transport"] == "aoa-local" else config["forward_port"]

    def _wait(self, predicate, process, log: Path) -> None:
        deadline = time.monotonic() + 60
        while not predicate():
            if process.poll() is not None or time.monotonic() > deadline:
                raise PhysicalAdapterError("echo endpoint did not start: " + log.read_text(errors="replace")[-1500:])
            time.sleep(0.1)

    def stop(self) -> dict:
        result = {}
        if self.bridge_process is not None:
            self.bridge_process.terminate()
            result["bridge_exit_code"] = self.bridge_process.wait(timeout=30)
        if self.forward is not None:
            self.harness._run(self.harness.adb + ["forward", "--remove", f"tcp:{self.config['forward_port']}"],
                              check=False, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30)
        if self.relay is not None:
            try:
                result["relay_exit_code"] = self.relay.wait(timeout=5)
            except subprocess.TimeoutExpired:
                path = self.bridge.relay_path
                for line in self.harness.shell("ps -A -o PID,ARGS").splitlines():
                    fields = line.split(None, 1)
                    if len(fields) == 2 and fields[0].isdigit() and fields[1].split()[:1] in ([path],
                                                                                            [Path(path).name]):
                        self.harness.shell(f"kill -TERM {fields[0]}")
                result["relay_exit_code"] = self.relay.wait(timeout=30)
        return result


def lifecycle(harness: Harness, arm: dict, directory: Path, inputs: dict) -> dict:
    """Resident worker over the AOA bridge: serve, re-arm, drop, detect, release, re-join, serve, idle stop."""
    config = harness.config
    short = {**config, "segments": [{"name": "m1", "rows": 1, "warmup_tokens": 0, "tokens": 2}]}
    session = harness.session(arm)
    steps: dict[str, object] = {"preflight": session.preflight().to_json(),
                                "launch": session.start(directory / "worker.log").to_json()}
    port = session.transport_contract().control_port
    first, _ = drive_ffn(port, short, inputs, harness)
    second, _ = drive_ffn(port, short, inputs, harness)  # the server re-arms by reconnecting
    steps["served_before_drop"] = len(first) + len(second)
    steps["alive"] = session.alive()
    steps["alive_connect"] = session.alive(connect=True)
    if arm.get("drop", "bridge-kill") == "bridge-kill":
        session._bridge_process.kill()
        session._bridge_process.wait(timeout=30)
    else:
        bridge = session.bridge
        restore = harness.session_options.get("restore_mode") or aoa_bridge.restore_normal
        steps["usb_reset"] = dict(restore(bridge.usb_sysfs_device, config["phone"]["serial"],
                                          forbidden_serials=bridge.forbidden_serials,
                                          timeout_s=bridge.mode_switch_timeout_s))
        # the reset re-enumerated the Pixel: release and re-join go over adb, so wait for it first
        deadline = time.monotonic() + 60
        while harness._run(harness.adb + ["get-state"], check=False, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=10).stdout.strip() != "device":
            if time.monotonic() > deadline:
                raise PhysicalAdapterError("adb did not return after the USB reset")
            time.sleep(0.5)
    deadline = time.monotonic() + 60
    while session.alive() and time.monotonic() < deadline:
        time.sleep(0.5)
    steps["alive_after_drop"] = session.alive()
    steps["client_exited_after_drop"] = session.client_exited()
    steps["release_lost"] = session.release_lost().to_json()
    orphan = session.release_orphaned_worker()
    steps["release_orphan"] = orphan.to_json() if orphan is not None else None
    steps["rejoin_preflight"] = session.preflight().to_json()
    steps["rejoin_launch"] = session.start(directory / "worker-rejoin.log").to_json()
    third, dump = drive_ffn(session.transport_contract().control_port, short, inputs, harness)
    steps["served_after_rejoin"] = len(third)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        text = session._log_path.read_text(errors="replace")
        if text.rfind("client disconnected") >= text.rfind("client connected"):
            break
        time.sleep(0.2)
    steps["stop"] = session.stop(allow_idle_signal=True).to_json()
    stop = steps["stop"]
    passed = (steps["alive"] and steps["alive_connect"] and not steps["alive_after_drop"]
              and steps["client_exited_after_drop"] and steps["served_after_rejoin"] == len(third) > 0
              and stop["boot_unchanged"] and stop["worker_pids_after"] == []
              and stop["aoa_bridge"]["stop"]["relay_pids_after"] == []
              and stop["aoa_bridge"]["stop"]["bridge_exit_code"] == 0)
    receipt = {"status": "PASS" if passed else "FAIL", "drop": arm.get("drop", "bridge-kill"), "steps": steps,
               "lifecycle": "AoaBridgePhoneWorkerSession: resident worker, re-armed connections, drop -> loss "
                            "detected -> released -> re-joined on the same session -> idle-only stop"}
    write_json(directory / "aoa-bridge-scheduler-launched-session.json", receipt)
    if not passed:
        raise PhysicalAdapterError("lifecycle arm failed: see aoa-bridge-scheduler-launched-session.json")
    return {"calls": len(first) + len(second) + len(third), "dumps": {"m1": dump["m1"]}}


# --- run ------------------------------------------------------------------------------------------

def run_arm(harness: Harness, arm: dict, output: Path, inputs: dict | None) -> dict:
    directory = output / arm["run"]
    directory.mkdir()
    write_json(directory / "ARM.json", arm)
    write_json(directory / "USB_MODE.json", harness.ensure_usb_mode(arm.get("usb_mode",
                                                                        harness.config.get("usb_mode", "accessory"))))
    write_json(directory / "HOST_USB_BEFORE.json", harness.host_usb())
    (directory / "PHONE_STATE_BEFORE.txt").write_text(harness.phone_state())
    holder = None
    if arm.get("phone_qos_latency_us") is not None:
        holder = harness.start_qos_holder(arm["phone_qos_latency_us"], directory / "qos-holder.log")
    result: dict = {"arm": arm["name"], "run": arm["run"], "kind": arm.get("kind", "ffn"),
                    "transport": arm["transport"], "status": "FAILED"}
    try:
        if arm.get("kind", "ffn") == "echo":
            endpoint = EchoEndpoint(harness, arm, directory)
            try:
                port = endpoint.start()
                rows = drive_echo(port, harness.config, harness)
            finally:
                result["endpoint_stop"] = endpoint.stop()
            dumps = {}
        elif arm.get("kind") == "lifecycle":
            outcome = lifecycle(harness, arm, directory, inputs)
            rows, dumps = [], {}
            result["lifecycle_calls"] = outcome["calls"]
        else:
            session = harness.session(arm)
            write_json(directory / "PREFLIGHT.json", session.preflight().to_json())
            write_json(directory / "LAUNCH.json", session.start(directory / "worker.log").to_json())
            rows, dumps = [], {}
            try:
                rows, dumps = drive_ffn(session.transport_contract().control_port, harness.config, inputs, harness)
                for name, data in dumps.items():
                    (directory / f"{name}.f16").write_bytes(data)
            finally:
                if session.active:
                    stop = session.stop(served_calls=len(rows))
                    write_json(directory / "STOP.json", stop.to_json())
                    result["stop"] = stop.to_json()
        write_json(directory / "CALLS.json", rows)
        result.update(status="PASS", calls=len(rows),
                      dump_sha256={name: sha256_bytes(data) for name, data in dumps.items()})
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if holder is not None:
            result["qos_holder"] = harness.stop_qos_holder(holder)
        (directory / "PHONE_STATE_AFTER.txt").write_text(harness.phone_state())
        write_json(directory / "HOST_USB_AFTER.json", harness.host_usb())
        write_json(directory / "RESULT.json", result)
    return result


def run(config: dict, output: Path, harness: Harness, only: set[str] | None = None) -> list[dict]:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "CONFIG.json", config)
    inputs = load_inputs(config) if any(arm.get("kind", "ffn") in ("ffn", "lifecycle")
                                        for arm in config["arms"]) else None
    results = []
    for arm in arm_order(config):
        if only and arm["name"] not in only:
            continue
        results.append(run_arm(harness, arm, output, inputs))
        write_json(output / "SUITE_RESULT.json", results)
        harness.sleep(config.get("cooldown_s", 20))
    return results


# --- analysis -------------------------------------------------------------------------------------

def _stats(values: list[float]) -> dict:
    return {"n": len(values), "p50": quantile(values, 0.5), "p90": quantile(values, 0.9),
            "mean": statistics.fmean(values) if values else None}


def trace_breakdown(path: Path) -> dict | None:
    """AOA arms with a bridge trace: per request, host round trip vs phone residence (relay rx -> tx, phone
    clock); their difference is link + relay USB wake-up, with no clock synchronisation."""
    if not path.exists():
        return None
    frames = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    link, residence = [], []
    pending_out = None
    last_in = None
    for frame in frames:
        if frame["kind"] != aoa_bridge.DATA:
            continue
        if frame["dir"] == "out":
            if last_in is not None and pending_out is not None:
                host = (last_in["host_ns"] - pending_out) / 1000
                phone = (last_in["peer_stamp_ns"] - last_in["aux"]) / 1000
                link.append(host - phone)
                residence.append(phone)
                last_in = None
                pending_out = None
            if pending_out is None:
                pending_out = frame["host_ns"]
        else:
            last_in = frame
    if last_in is not None and pending_out is not None:
        link.append((last_in["host_ns"] - pending_out) / 1000 - (last_in["peer_stamp_ns"] - last_in["aux"]) / 1000)
        residence.append((last_in["peer_stamp_ns"] - last_in["aux"]) / 1000)
    return {"link_and_relay_wakeup_us": _stats(link), "phone_residence_us": _stats(residence)}


def analyze(output: Path, reference: str = "adb") -> dict:
    config = json.loads((output / "CONFIG.json").read_text())
    table, dumps = [], {}
    for directory in sorted(path for path in output.iterdir() if path.is_dir()):
        result_path = directory / "RESULT.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        if result.get("status") != "PASS":
            table.append({"run": directory.name, "arm": result.get("arm"), "status": result.get("status")})
            continue
        if result["kind"] == "lifecycle":
            receipt = directory / "aoa-bridge-scheduler-launched-session.json"
            if receipt.exists():
                (output / receipt.name).write_text(receipt.read_text())
            table.append({"run": directory.name, "arm": result["arm"], "kind": "lifecycle", "status": "PASS"})
            continue
        calls = [row for row in json.loads((directory / "CALLS.json").read_text()) if row["token"] >= 0]
        for segment in config["segments"]:
            rows = [row for row in calls if row["segment"] == segment["name"]]
            first = [row["overhead_us"] for row in rows if row["position"] == 0]
            others = [row["overhead_us"] for row in rows if row["position"] > 0]
            table.append({
                "run": directory.name, "arm": result["arm"], "kind": result["kind"], "transport": result["transport"],
                "segment": segment["name"], "rows": segment["rows"], "status": "PASS",
                "rpc_us": _stats([row["rpc_us"] for row in rows]),
                "compute_us": _stats([row["compute_us"] for row in rows]),
                "overhead_us": _stats([row["overhead_us"] for row in rows]),
                "overhead_first_call_us": _stats(first), "overhead_other_calls_us": _stats(others),
                "trace": trace_breakdown(next(iter(sorted(directory.glob("*bridge-trace.jsonl"))),
                                              directory / "bridge-trace.jsonl"))
                if segment is config["segments"][0] else None,
            })
        if result["kind"] == "ffn":
            dumps[directory.name] = (result["arm"], result["transport"], result["dump_sha256"])
    ffn_dumps = {value[2].get(segment["name"]) for value in dumps.values() for segment in config["segments"]}
    by_segment = {segment["name"]: {run: value[2].get(segment["name"]) for run, value in dumps.items()}
                  for segment in config["segments"]}
    identical = bool(dumps) and all(len(set(values.values())) == 1 for values in by_segment.values())
    identity = {"status": "PASS" if identical and any(v[1] == "aoa-bridge" for v in dumps.values())
                and any(v[1] == "adb-tcp" for v in dumps.values()) else "FAIL",
                "segments": by_segment, "arms": {run: value[:2] for run, value in dumps.items()},
                "rows": sorted({segment["rows"] for segment in config["segments"]}),
                "binding": "same worker sha256 (preflight), same shard, same environment, same inputs; "
                           "adb-tcp vs aoa-bridge sessions, finite budget, clean stops",
                "distinct_output_hashes": len(ffn_dumps)}
    write_json(output / "aoa-bridge-byte-identity.json", identity)
    summary = {"schema": SCHEMA, "table": table, "byte_identity": identity["status"]}
    aoa = [row for row in table if row.get("transport") == "aoa-bridge" and row.get("kind") == "ffn"]
    arm_name = config.get("calibration_arm") or (aoa[0]["arm"] if aoa else None)
    chosen = [row for row in aoa if row["arm"] == arm_name]
    if chosen and identity["status"] == "PASS":
        by_rows = {}
        for rows in sorted({row["rows"] for row in chosen}):
            values = [row["overhead_us"]["p50"] for row in chosen if row["rows"] == rows]
            by_rows[rows] = statistics.median(values)
        receipt = {"status": "PASS", "calibration_arm": arm_name, "overhead_us_by_rows": by_rows,
                   "runs": sorted({row["run"] for row in chosen}), "cadence": config["cadence"],
                   "note": "median over runs of the per-run p50 non-compute overhead at the serving cadence"}
        if 1 in by_rows and 4 in by_rows and by_rows[4] > by_rows[1]:
            bandwidth = round(2 * (4 - 1) * LOGICAL_ROW_BYTES * 1e6 / (by_rows[4] - by_rows[1]))
            receipt.update(effective_transfer_bytes_per_s=bandwidth,
                           one_direction_fixed_us=round(max(0, by_rows[1] / 2 - LOGICAL_ROW_BYTES * 1e6 / bandwidth)))
        write_json(output / "aoa-bridge-round-trip.json", receipt)
        summary["round_trip"] = receipt
    write_json(output / "AB_TABLE.json", summary)
    lines = ["| run | arm | transport | rows | rpc p50 | compute p50 | overhead p50 / p90 | first-call ovh p50 | "
             "other-call ovh p50 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda value: "-" if value is None else f"{value / 1000:.2f}"  # noqa: E731
    for row in table:
        if row.get("kind") == "lifecycle":
            lines.append(f"| {row['run']} | {row['arm']} | lifecycle PASS |  |  |  |  |  |  |")
            continue
        if row.get("status") != "PASS":
            lines.append(f"| {row['run']} | {row.get('arm')} | FAILED |  |  |  |  |  |  |")
            continue
        lines.append(f"| {row['run']} | {row['arm']} | {row['transport']} | {row['rows']} | {fmt(row['rpc_us']['p50'])} | "
                     f"{fmt(row['compute_us']['p50'])} | {fmt(row['overhead_us']['p50'])} / "
                     f"{fmt(row['overhead_us']['p90'])} | {fmt(row['overhead_first_call_us']['p50'])} | "
                     f"{fmt(row['overhead_other_calls_us']['p50'])} |")
    lines.append("")
    lines.append(f"Byte identity (every ffn arm, every segment): **{identity['status']}** (ms above)")
    (output / "AB_TABLE.md").write_text("\n".join(lines) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "run", "analyze"))
    parser.add_argument("path", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--only", default=None)
    parser.add_argument("--reference", default="adb")
    args = parser.parse_args(argv)
    if args.command == "analyze":
        print(json.dumps(analyze(args.path, args.reference)["byte_identity"]))
        return 0
    config = load_config(args.path)
    if args.command == "plan":
        order = arm_order(config)
        print(json.dumps({"runs": [arm["run"] for arm in order], "calls_per_ffn_arm": calls_per_arm(config),
                          "estimated_minutes": round(estimate_seconds(config) / 60, 1)}, indent=2))
        return 0
    if args.output is None:
        parser.error("run needs OUT")
    only = set(args.only.split(",")) if args.only else None
    results = run(config, args.output, Harness(config), only)
    print(json.dumps([{key: row.get(key) for key in ("run", "status", "calls")} for row in results]))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
