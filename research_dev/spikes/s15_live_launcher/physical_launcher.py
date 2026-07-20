#!/usr/bin/env python3
"""Prepared one-shot OP15 B32 bridge for the S15 physical executor."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COHORT_DIR = HERE.parent / "s15_burst_cohort"
sys.path.insert(0, str(COHORT_DIR))

import validate_cohort  # noqa: E402


ARTIFACTS = ROOT / "research_dev/spikes/s14_mixed_streaming_scheduler/persistence_v2/artifacts"
HOST = ARTIFACTS / "host"
ANDROID = ARTIFACTS / "android"
COHORT_PATH = COHORT_DIR / "cohort.json"
INPUT_PATH = COHORT_DIR / "input_manifest.json"
RAW_ROOT = HERE / "results/raw"
FULL_MODEL = Path("/home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf")
SERIAL = "3C15AU002CL00000"
GPU_UUID = "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"
REMOTE = "/data/local/tmp/ls-s15-live"
SHARD = "/data/local/tmp/ls-npu/12b-f16-head-0-8.gguf"
SHARD_SHA256 = "a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8"
COHORT_FILE_SHA256 = "85e0614d31a6f35ad7d09c3b35b2ab4fa9ef2916bc4a1dcfc32d6f4c70a51ec4"
INPUT_FILE_SHA256 = "ea1f2d47b8c6dec6096c8b95b6073181ecba6838d97741ca92ec2ceca6937858"
PORT = 5941
BATCH = 32
N_GEN = 8
CONTEXT = 512
MAX_PREFILL = 64
PREPARE_TIMEOUT_S = 240
THERMAL_START_MAX_MILLIC = 60_000
THERMAL_END_MAX_MILLIC = 85_000
REQUEST_SCHEMA = "s15-physical-request-v1"
SESSION_SCHEMA = "s15-physical-session-v1"
BOUNDARY_SCHEMA = "s15-boundary-certificate-v1"
EXPECTED_ROUTE = "op15-gemma-head-0-8"
EXPECTED_PROFILE = "sha256:947f1fd95b3f1a7c881b71d793bf3177fc9612a46d959fcef987da0026531cae"
EXPECTED_DEVICE = "op15:3C15AU002CL00000"
EXPECTED_WORKER = "sha256:2494f191515ca576cacd8c411de79fa8985f719cba2a11cd6371aa6e1024be9f"
EXPECTED_COHORT = "sha256:" + COHORT_FILE_SHA256
EXPECTED_INPUT = "sha256:" + INPUT_FILE_SHA256
ALLOWED_CPU_OPS = {"GET_ROWS"}


class LauncherError(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n").encode("ascii")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_object(payload: bytes, label: str) -> dict:
    if type(payload) is not bytes or not payload or len(payload) > 4 * 1024 * 1024:
        raise LauncherError(f"{label} has an invalid byte length")

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise LauncherError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value):
        raise LauncherError(f"invalid JSON constant {value!r} in {label}")

    try:
        value = json.loads(
            payload, object_pairs_hook=no_duplicates, parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LauncherError(f"invalid {label} JSON: {exc}") from exc
    if type(value) is not dict:
        raise LauncherError(f"{label} must be a JSON object")
    return value


def load_workload() -> tuple[dict, dict, tuple[str, ...], str]:
    if sha256(COHORT_PATH) != COHORT_FILE_SHA256 or sha256(INPUT_PATH) != INPUT_FILE_SHA256:
        raise LauncherError("frozen cohort/input artifact digest mismatch")
    try:
        validated = validate_cohort.validate()
    except Exception as exc:
        raise LauncherError(f"independent cohort validation failed: {exc}") from exc
    cohort = strict_object(COHORT_PATH.read_bytes(), "cohort")
    inputs = strict_object(INPUT_PATH.read_bytes(), "input manifest")
    if cohort != validated:
        raise LauncherError("cohort changed after independent validation")
    requests = cohort.get("requests")
    payloads = inputs.get("request_payloads")
    prompt = inputs.get("prompt_text")
    if type(requests) is not list or len(requests) != BATCH \
            or type(payloads) is not list or len(payloads) != BATCH \
            or type(prompt) is not str or not prompt:
        raise LauncherError("frozen workload shape is invalid")
    request_ids = tuple(item.get("event_id") for item in requests)
    if any(type(value) is not str or not value for value in request_ids) \
            or len(set(request_ids)) != BATCH:
        raise LauncherError("frozen workload request identities are invalid")
    prompt_digest = "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if payloads != [
        {"event_id": request_id, "payload_sha256": prompt_digest}
        for request_id in request_ids
    ]:
        raise LauncherError("frozen workload payload binding is invalid")
    return cohort, inputs, request_ids, prompt


def validate_request(value: dict, expected_ids: tuple[str, ...] | None = None) -> None:
    required = {
        "schema", "command", "protocol_version", "launch_id", "route_id",
        "profile_id", "device_id", "route_epoch", "residency_epoch",
        "lease_epoch", "device_boot_epoch", "registry_generation",
        "compatibility_key", "request_ids", "cohort_sha256",
        "input_manifest_sha256", "timeout_us", "expected_boundary_schema",
        "worker_binary_sha256", "worker_generation", "device_boot_id",
        "layer_range",
    }
    if set(value) != required:
        raise LauncherError("request has missing or unknown fields")
    expected = {
        "schema": REQUEST_SCHEMA,
        "command": "EXECUTE",
        "protocol_version": 1,
        "route_id": EXPECTED_ROUTE,
        "profile_id": EXPECTED_PROFILE,
        "device_id": EXPECTED_DEVICE,
        "cohort_sha256": EXPECTED_COHORT,
        "input_manifest_sha256": EXPECTED_INPUT,
        "compatibility_key": "gemma-4-12b-it-f16|decode|gemma-head-0-8",
        "expected_boundary_schema": BOUNDARY_SCHEMA,
        "worker_binary_sha256": EXPECTED_WORKER,
        "worker_generation": 1,
        "layer_range": [0, 8],
    }
    for key, expected_value in expected.items():
        if type(value.get(key)) is not type(expected_value) or value.get(key) != expected_value:
            raise LauncherError(f"request identity mismatch: {key}")
    for key in (
        "launch_id", "route_epoch", "residency_epoch", "lease_epoch",
        "device_boot_epoch", "registry_generation", "timeout_us",
    ):
        if type(value[key]) is not int or value[key] < 1:
            raise LauncherError(f"request {key} must be a positive integer")
    if value["route_epoch"] != 12 or value["residency_epoch"] != 1 \
            or value["lease_epoch"] != 1 or value["device_boot_epoch"] != 1 \
            or value["registry_generation"] != 1:
        raise LauncherError("request carries a stale or unsupported epoch")
    if type(value["device_boot_id"]) is not str or not value["device_boot_id"]:
        raise LauncherError("request device_boot_id is invalid")
    if expected_ids is not None and value["request_ids"] != list(expected_ids):
        raise LauncherError("request IDs do not match the frozen cohort")


def run(args: list[str], *, timeout: float, env: dict | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            args, stdin=subprocess.DEVNULL, capture_output=True,
            timeout=timeout, check=False, env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LauncherError(f"command failed: {args[0]}: {exc}") from exc


def adb(*args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return run(["adb", "-s", SERIAL, *args], timeout=timeout)


def adb_checked(*args: str, timeout: float = 120) -> bytes:
    process = adb(*args, timeout=timeout)
    if process.returncode != 0:
        raise LauncherError(f"ADB command failed: {args[0] if args else 'adb'}")
    return process.stdout


def verify_artifacts() -> dict[str, str]:
    expected = {}
    for line in (ARTIFACTS / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in expected or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise LauncherError("invalid frozen artifact manifest")
        expected[relative] = digest
    for relative, digest in expected.items():
        path = ARTIFACTS / relative
        if not path.is_file() or sha256(path) != digest:
            raise LauncherError(f"frozen artifact mismatch: {relative}")
    if sha256(FULL_MODEL) != "bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a":
        raise LauncherError("full model digest mismatch")
    return expected


def deploy(expected: dict[str, str]) -> tuple[str, dict[str, str]]:
    if adb_checked("get-state").decode("ascii", errors="strict").strip() != "device":
        raise LauncherError("OP15 is not in device state")
    adb_checked("shell", "rm -rf /data/local/tmp/ls-s15-live && mkdir -p /data/local/tmp/ls-s15-live")
    names = (
        "llama-layersplit", "libggml-base.so", "libggml-cpu.so",
        "libggml-hexagon.so", "libggml-opencl.so", "libggml.so",
        "libllama-common.so", "libllama.so", "libc++_shared.so", "libggml-htp-v81.so",
    )
    deployed = {}
    for name in names:
        process = adb("push", str(ANDROID / name), f"{REMOTE}/{name}", timeout=240)
        if process.returncode != 0:
            raise LauncherError(f"failed to deploy {name}")
        remote = adb_checked("shell", f"sha256sum {REMOTE}/{name}").decode("ascii").split()[0]
        if remote != expected[f"android/{name}"]:
            raise LauncherError(f"deployed digest mismatch: {name}")
        deployed[name] = "sha256:" + remote
    adb_checked("shell", f"chmod 755 {REMOTE}/llama-layersplit")
    shard = adb_checked("shell", f"sha256sum {SHARD}").decode("ascii").split()[0]
    if shard != SHARD_SHA256:
        raise LauncherError("deployed OP15 shard digest mismatch")
    deployed[SHARD] = "sha256:" + shard
    boot_id = adb_checked("shell", "cat /proc/sys/kernel/random/boot_id").decode("ascii").strip()
    return boot_id, deployed


def thermal_snapshot() -> dict:
    command = (
        "for z in /sys/class/thermal/thermal_zone*; do "
        "ty=$(cat $z/type 2>/dev/null); t=$(cat $z/temp 2>/dev/null); "
        "case $ty in nsphmx-*) echo $ty=$t;; esac; done"
    )
    process = adb("shell", command)
    sensors = {}
    for line in process.stdout.decode("ascii", errors="ignore").splitlines():
        if "=" not in line:
            continue
        name, raw = line.split("=", 1)
        try:
            value = int(raw)
        except ValueError:
            continue
        if 10_000 <= value <= 120_000:
            sensors[name] = value
    return {
        "sensors_millic": dict(sorted(sensors.items())),
        "max_millic": max(sensors.values()) if sensors else None,
        "valid": process.returncode == 0 and bool(sensors),
    }


def discover_thermal_paths() -> tuple[dict[str, str], dict]:
    command = (
        "for z in /sys/class/thermal/thermal_zone*; do "
        "ty=$(cat $z/type 2>/dev/null); t=$(cat $z/temp 2>/dev/null); "
        "case $ty in nsphmx-*) echo $ty'|'$z'|'$t;; esac; done"
    )
    process = adb("shell", command)
    paths = {}
    sensors = {}
    for line in process.stdout.decode("ascii", errors="ignore").splitlines():
        fields = line.split("|")
        if len(fields) != 3:
            continue
        name, path, raw = fields
        if not name.startswith("nsphmx-") \
                or re.fullmatch(r"/sys/class/thermal/thermal_zone[0-9]+", path) is None:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if 10_000 <= value <= 120_000:
            paths[name] = path
            sensors[name] = value
    snapshot = {
        "sensors_millic": dict(sorted(sensors.items())),
        "max_millic": max(sensors.values()) if sensors else None,
        "valid": process.returncode == 0 and bool(sensors),
    }
    if len(paths) != len(sensors) or not thermal_ok(snapshot, THERMAL_START_MAX_MILLIC):
        raise LauncherError("OP15 thermal path discovery failed")
    return dict(sorted(paths.items())), snapshot


def thermal_ok(value: dict, maximum: int) -> bool:
    sensors = value.get("sensors_millic")
    observed = value.get("max_millic")
    return value.get("valid") is True and type(sensors) is dict and bool(sensors) \
        and type(observed) is int and observed == max(sensors.values()) and observed <= maximum


def prefixed_objects(payload: bytes, prefix: bytes) -> list[dict]:
    return [
        strict_object(line[len(prefix):], prefix.decode("ascii").strip())
        for line in payload.splitlines() if line.startswith(prefix)
    ]


class ReadyProcess:
    def __init__(self, command: list[str], env: dict | None, stdout_path: Path,
                 stderr_path: Path, ready_prefix: bytes | None,
                 ready_streams: tuple[str, ...] = ("stderr",)) -> None:
        if not ready_streams or any(value not in ("stdout", "stderr") for value in ready_streams):
            raise LauncherError("ready_streams must name stdout or stderr")
        self.stdout_handle = stdout_path.open("wb")
        self.stderr_handle = stderr_path.open("wb")
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env,
        )
        self.ready = threading.Event()
        self.ready_prefix = ready_prefix
        self.threads = [
            threading.Thread(
                target=self._copy,
                args=(self.process.stdout, self.stdout_handle, "stdout" in ready_streams),
                daemon=True,
            ),
            threading.Thread(
                target=self._copy,
                args=(self.process.stderr, self.stderr_handle, "stderr" in ready_streams),
                daemon=True,
            ),
        ]
        for thread in self.threads:
            thread.start()

    def _copy(self, source, destination, watch: bool) -> None:
        for line in iter(source.readline, b""):
            destination.write(line)
            destination.flush()
            if watch and self.ready_prefix is not None and line.startswith(self.ready_prefix):
                self.ready.set()

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ready.wait(0.1):
                return
            if self.process.poll() is not None:
                raise LauncherError("prepared process exited before readiness")
        raise LauncherError("prepared process readiness timeout")

    def go(self) -> None:
        if not self.ready.is_set() or self.process.stdin is None:
            raise LauncherError("GO before host readiness")
        self.process.stdin.write(b"GO\n")
        self.process.stdin.flush()

    def wait(self, timeout: float) -> int:
        try:
            returncode = self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self.terminate()
            raise LauncherError("prepared process completion timeout") from exc
        for thread in self.threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in self.threads):
            raise LauncherError("prepared process output reader did not terminate")
        self.close()
        return returncode

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for thread in self.threads:
            thread.join(timeout=2)
        self.close()

    def close(self) -> None:
        for handle in (self.stdout_handle, self.stderr_handle):
            if not handle.closed:
                handle.close()


class ThermalMonitor:
    def __init__(self, output_path: Path, error_path: Path,
                 thermal_paths: dict[str, str]) -> None:
        if type(thermal_paths) is not dict or not thermal_paths:
            raise LauncherError("thermal monitor requires discovered paths")
        self._expected_names = frozenset(thermal_paths)
        samples = []
        for name, path in sorted(thermal_paths.items()):
            if not name.startswith("nsphmx-") \
                    or re.fullmatch(r"/sys/class/thermal/thermal_zone[0-9]+", path) is None:
                raise LauncherError("thermal monitor path binding is invalid")
            samples.append(f"t=$(cat {path}/temp); printf '{name}=%s ' \"$t\"")
        command = "while true; do printf 'THERMAL '; " + "; ".join(samples) \
            + "; echo; sleep 0.2; done"
        self.output_handle = output_path.open("wb")
        self.error_handle = error_path.open("wb")
        self.process = subprocess.Popen(
            ["adb", "-s", SERIAL, "shell", command],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=self.error_handle,
        )
        self._lock = threading.Lock()
        self._latest = None
        self._latest_host_ns = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in iter(self.process.stdout.readline, b""):
            self.output_handle.write(line)
            self.output_handle.flush()
            snapshot = self._parse(line)
            if snapshot is not None:
                with self._lock:
                    self._latest = snapshot
                    self._latest_host_ns = time.monotonic_ns()
                self._ready.set()

    def _parse(self, line: bytes) -> dict | None:
        if not line.startswith(b"THERMAL "):
            return None
        sensors = {}
        for item in line[len(b"THERMAL "):].split():
            if b"=" not in item:
                continue
            name, raw = item.split(b"=", 1)
            try:
                value = int(raw)
                decoded = name.decode("ascii")
            except (ValueError, UnicodeDecodeError):
                continue
            if decoded.startswith("nsphmx-") and 10_000 <= value <= 120_000:
                sensors[decoded] = value
        if set(sensors) != self._expected_names:
            return None
        return {
            "sensors_millic": dict(sorted(sensors.items())),
            "max_millic": max(sensors.values()),
            "valid": True,
        }

    def wait_ready(self, timeout_s: float = 10) -> None:
        if not self._ready.wait(timeout_s):
            raise LauncherError("continuous thermal monitor did not become ready")

    def snapshot(self) -> dict:
        with self._lock:
            if self._latest is None or self._latest_host_ns is None:
                raise LauncherError("continuous thermal monitor has no sample")
            value = dict(self._latest)
            value["sample_age_us"] = (time.monotonic_ns() - self._latest_host_ns) // 1000
            return value

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self._thread.join(timeout=3)
        self.output_handle.close()
        self.error_handle.close()


def run_reference(prompt: str, raw_dir: Path) -> list[int]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    env["LD_LIBRARY_PATH"] = str(HOST)
    env.pop("LLAMA_LAYER_START", None)
    env.pop("LLAMA_LAYER_END", None)
    command = [
        str(HOST / "llama-layersplit"), "-m", str(FULL_MODEL), "-ngl", "99",
        "--mode", "monodriver", "-p", prompt, "-n", str(N_GEN),
        "--driver-requests", str(BATCH), "--driver-warmup", "0",
        "--driver-batch", str(BATCH), "--driver-context", str(CONTEXT),
        "--driver-max-prefill", str(MAX_PREFILL),
    ]
    process = run(command, timeout=PREPARE_TIMEOUT_S, env=env)
    (raw_dir / "reference.stdout.bin").write_bytes(process.stdout)
    (raw_dir / "reference.stderr.bin").write_bytes(process.stderr)
    rows = prefixed_objects(process.stderr, b"ROUTEJSON ")
    if process.returncode != 0 or len(rows) != BATCH \
            or sorted(row.get("stream_index") for row in rows) != list(range(BATCH)):
        raise LauncherError("same-batch A6000 reference did not complete 32 streams")
    tokens = rows[0].get("token_ids")
    if type(tokens) is not list or len(tokens) != N_GEN \
            or any(type(token) is not int for token in tokens) \
            or any(row.get("token_ids") != tokens for row in rows):
        raise LauncherError("same-input A6000 reference streams disagree")
    return tokens


def prepare_route(prompt: str, raw_dir: Path) -> tuple[ReadyProcess, ReadyProcess]:
    adb("forward", "--remove", f"tcp:{PORT}")
    if adb("forward", f"tcp:{PORT}", f"tcp:{PORT}").returncode != 0:
        raise LauncherError("failed to install OP15 port forward")
    phone_command = (
        f"cd {REMOTE} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
        "GGML_HEXAGON_MBUF=4192 LLAMA_LAYER_END=8 LAYERSPLIT_PLACEMENT_CERT=1 "
        f"./llama-layersplit -m {SHARD} --devices HTP0 -ngl 99 --mode stagenet "
        f"--port {PORT} -n {N_GEN} --driver-batch {BATCH} --driver-context {CONTEXT} "
        f"--driver-max-prefill {MAX_PREFILL}"
    )
    phone = ReadyProcess(
        ["adb", "-s", SERIAL, "shell", phone_command], None,
        raw_dir / "phone.stdout.bin", raw_dir / "phone.stderr.bin", b"[stagenet] listening",
        ("stdout", "stderr"),
    )
    phone.wait_ready(PREPARE_TIMEOUT_S)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    env["LD_LIBRARY_PATH"] = str(HOST)
    env["LLAMA_LAYER_START"] = "8"
    env.pop("LLAMA_LAYER_END", None)
    host_command = [
        str(HOST / "llama-layersplit"), "-m", str(FULL_MODEL), "-ngl", "99",
        "--mode", "pipedriver", "--host", "127.0.0.1", "--port", str(PORT),
        "-p", prompt, "-n", str(N_GEN), "--driver-requests", str(BATCH),
        "--driver-warmup", "0", "--driver-batch", str(BATCH),
        "--driver-context", str(CONTEXT), "--driver-max-prefill", str(MAX_PREFILL),
        "--wait-for-go",
    ]
    host = ReadyProcess(
        host_command, env, raw_dir / "host.stdout.bin", raw_dir / "host.stderr.bin",
        b"DRIVER_READY ",
    )
    try:
        host.wait_ready(PREPARE_TIMEOUT_S)
    except Exception:
        host.terminate()
        phone.terminate()
        raise
    return host, phone


def validate_completion(host_stderr: bytes, phone_output: bytes,
                        reference: list[int]) -> tuple[dict, list[dict]]:
    rows = prefixed_objects(host_stderr, b"ROUTEJSON ")
    done = prefixed_objects(host_stderr, b"DRIVER_DONE ")
    certs = prefixed_objects(phone_output, b"PLACEMENTCERT ")
    if len(rows) != BATCH or len(done) != 1 or len(certs) != 1:
        raise LauncherError("route lacks one complete ROUTEJSON/PLACEMENTCERT evidence set")
    if done[0].get("status") != "ok" or done[0].get("requests") != BATCH \
            or sorted(row.get("stream_index") for row in rows) != list(range(BATCH)):
        raise LauncherError("route completion set is incomplete")
    for row in rows:
        if row.get("status") != "ok" or row.get("batch_size") != BATCH \
                or row.get("generated_tokens") != N_GEN or row.get("token_ids") != reference:
            raise LauncherError("route token/correctness gate failed")
    cert = certs[0]
    if cert.get("status") != "SCHEDULED_PLACEMENT_OK" \
            or cert.get("layer_start") != 0 or cert.get("layer_end") != 8 \
            or cert.get("missing_buffer_compute_nodes") != 0:
        raise LauncherError("route placement certificate failed")
    mapping = cert.get("compute_by_op_and_buffer")
    if type(mapping) is not dict or not mapping:
        raise LauncherError("route placement map is missing")
    htp_nodes = 0
    for op, buffers in mapping.items():
        if type(op) is not str or type(buffers) is not dict or not buffers:
            raise LauncherError("route placement map is invalid")
        for backend, count in buffers.items():
            if type(count) is not int or count <= 0:
                raise LauncherError("route placement node count is invalid")
            if backend == "HTP0":
                htp_nodes += count
            elif backend != "CPU" or op not in ALLOWED_CPU_OPS:
                raise LauncherError("route used an undeclared backend fallback")
    if htp_nodes == 0:
        raise LauncherError("route has no HTP0 work")
    return {
        "status": cert["status"],
        "layer_start": cert["layer_start"],
        "layer_end": cert["layer_end"],
        "missing_buffer_compute_nodes": cert["missing_buffer_compute_nodes"],
        "compute_by_op_and_buffer": mapping,
    }, rows


def error_record(request: dict) -> dict:
    return {
        "schema": SESSION_SCHEMA,
        "protocol_version": request["protocol_version"],
        "launch_id": request["launch_id"],
        "route_id": request["route_id"],
        "profile_id": request["profile_id"],
        "device_id": request["device_id"],
        "route_epoch": request["route_epoch"],
        "residency_epoch": request["residency_epoch"],
        "lease_epoch": request["lease_epoch"],
        "device_boot_epoch": request["device_boot_epoch"],
        "registry_generation": request["registry_generation"],
        "compatibility_key": request["compatibility_key"],
        "request_ids": request["request_ids"],
        "cohort_sha256": request["cohort_sha256"],
        "input_manifest_sha256": request["input_manifest_sha256"],
        "worker_binary_sha256": request["worker_binary_sha256"],
        "worker_generation": request["worker_generation"],
        "device_boot_id": request["device_boot_id"],
        "layer_range": request["layer_range"],
        "session_id": 1,
        "outcome": "error",
        "boundary_schema": request["expected_boundary_schema"],
        "placement": None,
        "boundaries": [],
    }


def completed_record(request: dict, placement: dict) -> dict:
    result = error_record(request)
    result["outcome"] = "completed"
    result["placement"] = placement
    result["boundaries"] = [
        {
            "request_id": request_id,
            "identity_ok": True,
            "epoch_ok": True,
            "correctness_ok": True,
            "d2h_complete": True,
        }
        for request_id in request["request_ids"]
    ]
    return result


def cleanup(host: ReadyProcess | None, phone: ReadyProcess | None,
            thermal_monitor: ThermalMonitor | None) -> None:
    for process in (host, phone):
        if process is not None:
            process.terminate()
    adb("forward", "--remove", f"tcp:{PORT}")
    adb("shell", "pkill -9 -f llama-layersplit")
    if thermal_monitor is not None:
        thermal_monitor.stop()


def main() -> int:
    raw_dir = RAW_ROOT / "launch-1"
    if raw_dir.exists():
        print("launcher raw directory already exists", file=sys.stderr)
        return 2
    raw_dir.mkdir(parents=True)
    host = None
    phone = None
    thermal_monitor = None
    request = None
    try:
        cohort, inputs, request_ids, prompt = load_workload()
        (raw_dir / "cohort.json").write_bytes(COHORT_PATH.read_bytes())
        (raw_dir / "input_manifest.json").write_bytes(INPUT_PATH.read_bytes())
        expected = verify_artifacts()
        boot_id, deployed = deploy(expected)
        thermal_paths, start_thermal = discover_thermal_paths()
        thermal_monitor = ThermalMonitor(
            raw_dir / "thermal_stream.log", raw_dir / "thermal_stream.stderr.bin",
            thermal_paths,
        )
        thermal_monitor.wait_ready()
        reference = run_reference(prompt, raw_dir)
        (raw_dir / "reference_tokens.json").write_bytes(canonical({"token_ids": reference}))
        host, phone = prepare_route(prompt, raw_dir)
        preflight = {
            "schema": "s15-live-preflight-v1",
            "scope": "SETUP_OUTSIDE_PAID_WINDOW_PROMPT_VISIBLE_BEFORE_ARRIVAL",
            "cohort_sha256": EXPECTED_COHORT,
            "input_manifest_sha256": EXPECTED_INPUT,
            "inner_cohort_hash": cohort["cohort_hash"],
            "inner_input_manifest_hash": inputs["input_manifest_hash"],
            "artifact_manifest_sha256": "sha256:" + sha256(ARTIFACTS / "SHA256SUMS.txt"),
            "reference_stderr_sha256": "sha256:" + sha256(raw_dir / "reference.stderr.bin"),
            "device_boot_id": boot_id,
            "worker_binary_sha256": EXPECTED_WORKER,
            "deployed_sha256": deployed,
            "thermal_start": start_thermal,
            "thermal_paths": thermal_paths,
            "host_pid": host.process.pid,
            "phone_adb_pid": phone.process.pid,
        }
        (raw_dir / "preflight.json").write_bytes(canonical(preflight))
        print("LAUNCHER_READY " + json.dumps({
            "cohort_sha256": EXPECTED_COHORT,
            "input_manifest_sha256": EXPECTED_INPUT,
            "device_boot_id": boot_id,
        }, sort_keys=True), file=sys.stderr, flush=True)

        payload = sys.stdin.buffer.read(4 * 1024 * 1024 + 1)
        request = strict_object(payload, "request")
        validate_request(request, request_ids)
        if request["device_boot_id"] != boot_id:
            raise LauncherError("live device boot identity does not match the request")
        (raw_dir / "request.json").write_bytes(payload)

        paid_start_ns = time.monotonic_ns()
        host.go()
        timeout_s = request["timeout_us"] / 1_000_000
        host_rc = host.wait(timeout_s)
        paid_end_ns = time.monotonic_ns()
        end_thermal = thermal_monitor.snapshot()
        if end_thermal.get("sample_age_us", 1_000_001) > 1_000_000 \
                or not thermal_ok(end_thermal, THERMAL_END_MAX_MILLIC):
            raise LauncherError("OP15 continuous end thermal gate failed")
        phone_rc = phone.wait(30)
        if host_rc != 0 or phone_rc != 0:
            raise LauncherError("prepared host or phone process failed")
        placement, rows = validate_completion(
            (raw_dir / "host.stderr.bin").read_bytes(),
            (raw_dir / "phone.stdout.bin").read_bytes()
            + (raw_dir / "phone.stderr.bin").read_bytes(),
            reference,
        )
        paid = {
            "schema": "s15-live-paid-window-v1",
            "start_monotonic_ns": paid_start_ns,
            "end_monotonic_ns": paid_end_ns,
            "elapsed_us": (paid_end_ns - paid_start_ns) // 1000,
            "route_wall_us_max": max(row["request_wall_us"] for row in rows),
            "thermal_end": end_thermal,
        }
        (raw_dir / "paid_window.json").write_bytes(canonical(paid))
        record = completed_record(request, placement)
        sys.stdout.buffer.write(canonical(record))
        sys.stdout.buffer.flush()
    except Exception as exc:
        (raw_dir / "launcher_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="ascii", errors="backslashreplace",
        )
        if request is None:
            return 2
        record = error_record(request)
        sys.stdout.buffer.write(canonical(record))
        sys.stdout.buffer.flush()
    finally:
        cleanup(host, phone, thermal_monitor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
