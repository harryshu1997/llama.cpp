#!/usr/bin/env python3
"""S14 CP-E live mixed-priority, one-GPU, two-phone experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import bge_server_prof as BGE
import cp_b_phone_bge as CPB
import cp_d_priority as CPD
import pipe3_device as P3
import stage_a_gpu_board as A
import stageb_headcert as SB


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SELECTED_GPU_UUID = CPD.SELECTED_GPU_UUID
SECOND_GPU_UUID = CPD.SECOND_GPU_UUID
K1 = 8
K2 = 12
ANDROID_LAYERSPLIT = REPO / "build-snapdragon/bin/llama-layersplit"
REMOTE_DIR = "/data/local/tmp/ls-s14-cpe"
REMOTE_BIN = "llama-layersplit"
ANDROID_RUNTIME_LIBS = (
    "libggml-base.so", "libggml-cpu.so", "libggml-hexagon.so", "libggml-opencl.so",
    "libggml.so", "libllama-common.so", "libllama.so",
)
PHONE_START_MAX_MILLIC = 60_000
PHONE_END_MAX_MILLIC = 85_000
DRIVER_WARMUP_GROUPS = 1


class LiveGateError(RuntimeError):
    pass


class ReadyProcess:
    def __init__(
        self,
        command: list[str],
        env: dict[str, str],
        ready_prefix: str,
        done_prefix: str | None,
    ) -> None:
        self.command = command
        self.env = env
        self.ready_prefix = ready_prefix
        self.done_prefix = done_prefix
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []
        self.ready = threading.Event()
        self.done_time_s: float | None = None
        self.process: subprocess.Popen[str] | None = None
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
            bufsize=1,
        )
        assert self.process.stdout is not None and self.process.stderr is not None
        self.threads = [
            threading.Thread(target=self._read, args=(self.process.stdout, self.stdout_lines, False), daemon=True),
            threading.Thread(target=self._read, args=(self.process.stderr, self.stderr_lines, True), daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def _read(self, stream: Any, output: list[str], is_stderr: bool) -> None:
        for raw in stream:
            line = raw.rstrip("\n")
            output.append(line)
            if is_stderr and line.startswith(self.ready_prefix):
                self.ready.set()
            if is_stderr and self.done_prefix and line.startswith(self.done_prefix):
                self.done_time_s = time.time()

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.ready.wait(timeout=0.1):
                return
            if self.process is not None and self.process.poll() is not None:
                raise LiveGateError(f"process exited before ready: {self.stderr_tail()}")
        raise LiveGateError(f"process ready timeout: {self.stderr_tail()}")

    def go(self) -> None:
        if self.process is None or self.process.stdin is None or not self.ready.is_set():
            raise LiveGateError("GO before process readiness")
        self.process.stdin.write("GO\n")
        self.process.stdin.flush()

    def wait(self, timeout_s: float) -> int:
        if self.process is None:
            raise LiveGateError("wait before process start")
        try:
            returncode = self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            self.terminate()
            raise LiveGateError("process completion timeout") from exc
        for thread in self.threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in self.threads):
            raise LiveGateError("process output reader did not stop")
        return returncode

    def terminate(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for thread in self.threads:
            thread.join(timeout=5)

    def stdout_text(self) -> str:
        return "\n".join(self.stdout_lines)

    def stderr_text(self) -> str:
        return "\n".join(self.stderr_lines)

    def stderr_tail(self) -> str:
        return "\n".join(self.stderr_lines[-30:])


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise LiveGateError("empty latency sample set")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = int(rank)
    if lower + 1 == len(ordered):
        return ordered[lower]
    return ordered[lower] + (rank - lower) * (ordered[lower + 1] - ordered[lower])


def persist_process_logs(log_dir: Path, high: ReadyProcess | None, low: ReadyProcess | None) -> None:
    for name, process in (("high", high), ("low", low)):
        if process is None:
            continue
        (log_dir / f"{name}.stdout").write_text(process.stdout_text(), encoding="utf-8")
        (log_dir / f"{name}.stderr").write_text(process.stderr_text(), encoding="utf-8")


def remote_sha256(serial: str, path: str) -> str:
    try:
        digest = SB.sha256_remote(serial, path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError) as exc:
        raise LiveGateError(f"cannot hash {path} on {serial}") from exc
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise LiveGateError(f"invalid remote digest for {path} on {serial}")
    return "sha256:" + digest


def capture_artifacts() -> dict[str, Any]:
    local_android_digest = CPD.sha256_file(ANDROID_LAYERSPLIT)
    android_runtime = {
        name: {
            "path": str(REPO / "build-snapdragon/bin" / name),
            "sha256": CPD.sha256_file(REPO / "build-snapdragon/bin" / name),
        }
        for name in ANDROID_RUNTIME_LIBS
    }
    phone_binary = {}
    phone_runtime = {}
    phone_skeleton = {}
    for name, serial in (("op15", P3.OP15_SERIAL), ("op12", P3.OP12_SERIAL)):
        remote_path = f"{REMOTE_DIR}/{REMOTE_BIN}"
        remote_digest = remote_sha256(serial, remote_path)
        if remote_digest != local_android_digest:
            raise LiveGateError(f"{name} binary differs from current Android build")
        phone_binary[name] = {
            "serial": serial,
            "remote_path": remote_path,
            "sha256": remote_digest,
        }
        phone_runtime[name] = {}
        for library, local in android_runtime.items():
            remote_library = f"{REMOTE_DIR}/{library}"
            remote_library_digest = remote_sha256(serial, remote_library)
            if remote_library_digest != local["sha256"]:
                raise LiveGateError(f"{name} runtime {library} differs from current Android build")
            phone_runtime[name][library] = {
                "serial": serial,
                "remote_path": remote_library,
                "sha256": remote_library_digest,
            }
        skeleton_name = "libggml-htp-v81.so" if name == "op15" else "libggml-htp-v75.so"
        skeleton_path = f"{REMOTE_DIR}/{skeleton_name}"
        phone_skeleton[name] = {
            "serial": serial,
            "remote_path": skeleton_path,
            "sha256": remote_sha256(serial, skeleton_path),
        }
    return {
        "host": {
            "cp_e_live_priority.py": {
                "path": str(Path(__file__)),
                "sha256": CPD.sha256_file(Path(__file__)),
            },
            "bge_server_prof.py": {
                "path": str(Path(BGE.__file__)),
                "sha256": CPD.sha256_file(Path(BGE.__file__)),
            },
            "cp_b_phone_bge.py": {
                "path": str(Path(CPB.__file__)),
                "sha256": CPD.sha256_file(Path(CPB.__file__)),
            },
            "stage_a_gpu_board.py": {
                "path": str(Path(A.__file__)),
                "sha256": CPD.sha256_file(Path(A.__file__)),
            },
            "pipe3_device.py": {
                "path": str(Path(P3.__file__)),
                "sha256": CPD.sha256_file(Path(P3.__file__)),
            },
            "stageb_headcert.py": {
                "path": str(Path(SB.__file__)),
                "sha256": CPD.sha256_file(Path(SB.__file__)),
            },
            "llama_embedding": {
                "path": str(Path(BGE.CUDA_BIN)),
                "sha256": CPD.sha256_file(Path(BGE.CUDA_BIN)),
            },
            "llama_layersplit": {
                "path": str(Path(SB.HOST_BIN)),
                "sha256": CPD.sha256_file(Path(SB.HOST_BIN)),
            },
            "android_llama_layersplit": {
                "path": str(ANDROID_LAYERSPLIT),
                "sha256": local_android_digest,
            },
            "bge_server_result": {
                "path": str(CPD.BGE_PROFILE),
                "sha256": CPD.sha256_file(CPD.BGE_PROFILE),
            },
            "gemma_batch_profile": {
                "path": str(HERE / "stage_d_batch_scaling.json"),
                "sha256": CPD.sha256_file(HERE / "stage_d_batch_scaling.json"),
            },
        },
        "phone_binary": phone_binary,
        "android_runtime": android_runtime,
        "phone_runtime": phone_runtime,
        "phone_skeleton": phone_skeleton,
        "phone_shards": {
            "op15": {
                "serial": P3.OP15_SERIAL,
                "remote_path": P3.HEAD_SHARD.format(k1=K1),
                "sha256": remote_sha256(P3.OP15_SERIAL, P3.HEAD_SHARD.format(k1=K1)),
            },
            "op12": {
                "serial": P3.OP12_SERIAL,
                "remote_path": P3.MID_SHARD.format(k1=K1, k2=K2),
                "sha256": remote_sha256(P3.OP12_SERIAL, P3.MID_SHARD.format(k1=K1, k2=K2)),
            },
        },
    }


def wait_for_phone_thermal(serial: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = CPB.thermal_snapshot(serial)
        if last.get("valid") is True and type(last.get("max_millic")) is int \
                and last["max_millic"] <= PHONE_START_MAX_MILLIC:
            return last
        time.sleep(5)
    raise LiveGateError(f"phone {serial} did not reach the start thermal envelope: {last}")


def validate_end_thermal(snapshot: dict[str, Any], serial: str) -> None:
    if snapshot.get("valid") is not True or type(snapshot.get("max_millic")) is not int \
            or snapshot["max_millic"] > PHONE_END_MAX_MILLIC:
        raise LiveGateError(f"phone {serial} left the end thermal envelope: {snapshot}")


def bge_command(batch: int, words: int, exact_tokens: int, reps: int) -> ReadyProcess:
    corpus = HERE / "logs_live_priority" / f"bge_w{words}_b{batch}.txt"
    corpus.parent.mkdir(exist_ok=True)
    BGE.write_corpus(corpus, BGE.make_prompt(words), batch)
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": SELECTED_GPU_UUID,
        "LD_LIBRARY_PATH": BGE.CUDA_LIB,
        "BGEPROF_REPS": str(reps),
        "BGEPROF_WARMUP": "5",
        "BGEPROF_WAIT_FOR_GO": "1",
    })
    command = [
        BGE.CUDA_BIN, "-m", BGE.GGUF, "-ngl", "99", "--pooling", "cls",
        "--embd-normalize", "2", "-fa", "off", "-f", str(corpus),
        "--parallel", str(max(batch, 2)), "-c", str(BGE.ctx_for(batch, exact_tokens)),
        "-b", str(BGE.ctx_for(batch, exact_tokens)), "--embd-output-format", "array",
    ]
    return ReadyProcess(command, env, "BGEPROF_READY ", None)


def low_host_command(
    label: str,
    batch: int,
    requests: int,
    n_gen: int,
    prompt: str,
    port_a: int,
    port_b: int,
) -> ReadyProcess:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU_UUID
    env["LD_LIBRARY_PATH"] = str(Path(SB.HOST_BIN).parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["LAYERSPLIT_PLACEMENT_CERT"] = "1"
    env.pop("LLAMA_LAYER_END", None)
    command = [SB.HOST_BIN, "-m", SB.FULL_MODEL, "-ngl", "99"]
    if label == "P0":
        env.pop("LLAMA_LAYER_START", None)
        command += ["--mode", "monodriver"]
    elif label == "P2":
        env["LLAMA_LAYER_START"] = str(K2)
        command += [
            "--mode", "pipedriver", "--host", "127.0.0.1",
            "--port", str(port_a), "--port2", str(port_b),
        ]
    else:
        raise LiveGateError(f"unsupported label {label}")
    command += [
        "-p", prompt, "-n", str(n_gen), "--driver-requests", str(requests),
        "--driver-warmup", str(DRIVER_WARMUP_GROUPS), "--driver-batch", str(batch),
        "--driver-context", "512", "--driver-max-prefill", "64", "--wait-for-go",
    ]
    return ReadyProcess(command, env, "DRIVER_READY ", "DRIVER_DONE ")


def parse_bge(process: ReadyProcess, batch: int, exact_tokens: int, reps: int) -> dict[str, Any]:
    records = [line for line in process.stdout_lines if line.startswith("BGEPROF ")]
    if len(records) != 1:
        raise LiveGateError(f"expected one BGEPROF record, found {len(records)}")
    record = json.loads(records[0][len("BGEPROF "):])
    if record.get("finite") is not True or record.get("batch") != batch \
            or record.get("total_tokens") != batch * exact_tokens or record.get("reps") != reps:
        raise LiveGateError("BGE output shape or finite gate failed")
    placement = CPD.parse_placement(process.stderr_text())
    samples = [float(value) for value in record.get("us", [])]
    if len(samples) != reps or not all(math.isfinite(value) and value > 0 for value in samples):
        raise LiveGateError("BGE latency samples invalid")
    return {
        "batch": batch,
        "seq_len_exact": exact_tokens,
        "reps": reps,
        "encodes": batch * reps,
        "paid_start_s": record["paid_start_s"],
        "paid_end_s": record["paid_end_s"],
        "lat_us_p50": percentile(samples, 0.50),
        "lat_us_p95": percentile(samples, 0.95),
        "lat_us_p99": percentile(samples, 0.99),
        "latency_samples_us": samples,
        "placement": placement,
    }


def parse_low(
    label: str,
    process: ReadyProcess,
    batch: int,
    requests: int,
    reference: list[int],
    cert_a: dict | None,
    cert_b: dict | None,
) -> dict[str, Any]:
    if process.done_time_s is None:
        raise LiveGateError("low-priority driver emitted no DRIVER_DONE")
    rows = SB.parse_routejson(process.stderr_text())
    if len(rows) != requests:
        raise LiveGateError(f"low-priority result count {len(rows)} != {requests}")
    if any(row.get("batch_size") != batch or row.get("token_ids") != reference
           or row.get("generated_tokens") != len(reference) for row in rows):
        raise LiveGateError("low-priority batch or token correctness failed")
    host_cert = SB.parse_placement_cert(process.stderr_text())
    host_start = 0 if label == "P0" else K2
    validate_host_cert(host_cert, host_start, SB.N_LAYER)
    if label == "P2":
        reasons_a = P3.validate_stage_cert(cert_a, 0, K1)
        reasons_b = P3.validate_stage_cert(cert_b, K1, K2)
        if reasons_a or reasons_b:
            raise LiveGateError(f"phone placement failed: op15={reasons_a}, op12={reasons_b}")
    wall = [float(row["request_wall_us"]) for row in rows]
    generated = sum(int(row["generated_tokens"]) for row in rows)
    return {
        "batch": batch,
        "requests": requests,
        "generated_tokens": generated,
        "lat_us_p50": percentile(wall, 0.50),
        "lat_us_p95": percentile(wall, 0.95),
        "lat_us_p99": percentile(wall, 0.99),
        "done_time_s": process.done_time_s,
        "all_tokens_match": True,
        "phone_live": label == "P2",
        "op15_cert": cert_a,
        "op12_cert": cert_b,
        "host_cert": host_cert,
    }


def validate_host_cert(cert: dict | None, expected_start: int, expected_end: int) -> None:
    if cert is None:
        raise LiveGateError("host emitted no placement certificate")
    if cert.get("status") != "SCHEDULED_PLACEMENT_OK" or cert.get("missing_buffer_compute_nodes") != 0:
        raise LiveGateError("host placement status or missing-buffer gate failed")
    if cert.get("layer_start") != expected_start or cert.get("layer_end") != expected_end:
        raise LiveGateError("host placement layer range mismatch")
    buffers = cert.get("compute_by_buffer_type", {})
    if sum(value for name, value in buffers.items() if "CUDA" in name) <= 0:
        raise LiveGateError("host placement has no CUDA compute")
    for op, by_buffer in cert.get("compute_by_op_and_buffer", {}).items():
        for buffer, count in by_buffer.items():
            if "CUDA" not in buffer:
                raise LiveGateError(f"host fallback {op}@{buffer}:{count}")


def run_cohort(
    label: str,
    repeat: int,
    bge_batch: int,
    gemma_batch: int,
    requests: int,
    n_gen: int,
    bge_words: int,
    bge_exact: int,
    bge_reps: int,
    prompt: str,
    reference: list[int],
    ready_timeout_s: int,
    execution_timeout_s: int,
) -> dict[str, Any]:
    log_dir = HERE / "logs_live_priority" / f"{label.lower()}_r{repeat}"
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in ("high.stdout", "high.stderr", "low.stdout", "low.stderr"):
        (log_dir / name).unlink(missing_ok=True)
    port_a = 15770 + repeat * 4 + (0 if label == "P0" else 2)
    port_b = port_a + 1
    stage_a = stage_b = handle_a = handle_b = None
    low = high = None
    phone_thermal_start = phone_thermal_end = None
    sampler = A.Sampler(CPD.GPU_INDEX)
    try:
        if label == "P2":
            phone_thermal_start = {
                "op15": wait_for_phone_thermal(P3.OP15_SERIAL, timeout_s),
                "op12": wait_for_phone_thermal(P3.OP12_SERIAL, timeout_s),
            }
            stage_a, handle_a = P3.start_stage(
                P3.OP15_SERIAL, P3.HEAD_SHARD.format(k1=K1), 0, K1, port_a, gemma_batch,
                n_gen, 512, 64, SB.mbuf_for_k(K1), log_dir / "op15.log",
                REMOTE_DIR, REMOTE_BIN, ready_timeout_s,
            )
            stage_b, handle_b = P3.start_stage(
                P3.OP12_SERIAL, P3.MID_SHARD.format(k1=K1, k2=K2), K1, K2, port_b, gemma_batch,
                n_gen, 512, 64, max(3072, (K2 - K1) * 428 + 768), log_dir / "op12.log",
                REMOTE_DIR, REMOTE_BIN, ready_timeout_s,
            )
        low = low_host_command(label, gemma_batch, requests, n_gen, prompt, port_a, port_b)
        low.start()
        low.wait_ready(ready_timeout_s)
        high = bge_command(bge_batch, bge_words, bge_exact, bge_reps)
        high.start()
        high.wait_ready(ready_timeout_s)
        if not CPD.second_gpu_idle():
            raise LiveGateError("second GPU active before GO")
        sampler.start()
        time.sleep(1.0)
        go_time_s = time.time()
        high.go()
        low.go()
        high_rc = high.wait(execution_timeout_s)
        low_rc = low.wait(execution_timeout_s)
        if high_rc != 0 or low_rc != 0:
            raise LiveGateError(f"process failure high={high_rc} low={low_rc}")
        time.sleep(0.4)
        bge = parse_bge(high, bge_batch, bge_exact, bge_reps)
        cohort_end_s = max(float(bge["paid_end_s"]), float(low.done_time_s or 0.0))
        energy = CPD.persisted_energy_window(sampler, go_time_s, cohort_end_s)
    finally:
        sampler.stop()
        if high is not None:
            high.terminate()
        if low is not None:
            low.terminate()
        persist_process_logs(log_dir, high, low)
        if stage_b is not None:
            P3.stop_stage(P3.OP12_SERIAL, stage_b, handle_b, port_b, REMOTE_BIN)
        if stage_a is not None:
            P3.stop_stage(P3.OP15_SERIAL, stage_a, handle_a, port_a, REMOTE_BIN)

    if label == "P2":
        phone_thermal_end = {
            "op15": CPB.thermal_snapshot(P3.OP15_SERIAL),
            "op12": CPB.thermal_snapshot(P3.OP12_SERIAL),
        }
        validate_end_thermal(phone_thermal_end["op15"], P3.OP15_SERIAL)
        validate_end_thermal(phone_thermal_end["op12"], P3.OP12_SERIAL)

    cert_a = cert_b = None
    if label == "P2":
        op15_text = (log_dir / "op15.log").read_text(encoding="utf-8", errors="replace")
        op12_text = (log_dir / "op12.log").read_text(encoding="utf-8", errors="replace")
        cert_a = SB.parse_placement_cert(op15_text)
        cert_b = SB.parse_placement_cert(op12_text)
    low_result = parse_low(label, low, gemma_batch, requests, reference, cert_a, cert_b)
    overlap_start_s = max(go_time_s, float(bge["paid_start_s"]))
    overlap_end_s = min(float(bge["paid_end_s"]), float(low_result["done_time_s"]))
    overlap_s = max(0.0, overlap_end_s - overlap_start_s)
    shorter_service_s = min(
        float(bge["paid_end_s"]) - float(bge["paid_start_s"]),
        float(low_result["done_time_s"]) - go_time_s,
    )
    overlap_fraction = overlap_s / shorter_service_s if shorter_service_s > 0 else 0.0
    return {
        "label": label,
        "repeat": repeat,
        "go_time_s": go_time_s,
        "second_gpu_idle_before_go": True,
        "bge": bge,
        "gemma": low_result,
        "concurrent_overlap_s": overlap_s,
        "concurrent_overlap_fraction_of_shorter": overlap_fraction,
        "phone_thermal_start": phone_thermal_start,
        "phone_thermal_end": phone_thermal_end,
        "selected_gpu_cohort": energy,
    }


def summarize(
    rows: list[dict[str, Any]],
    high_slowdown_gate: float,
    low_slowdown_gate: float,
    min_overlap_s: float,
    min_overlap_fraction: float,
) -> dict[str, Any]:
    by_label = {label: [row for row in rows if row.get("label") == label] for label in ("P0", "P2")}
    if any(len(group) != 3 for group in by_label.values()):
        raise LiveGateError("expected three P0 and three P2 cohorts")
    if any(row["gemma"].get("phone_live") is not True for row in by_label["P2"]):
        raise LiveGateError("P2 lacks live phone execution")
    overlap_gate = all(
        row.get("concurrent_overlap_s", 0.0) >= min_overlap_s
        and row.get("concurrent_overlap_fraction_of_shorter", 0.0) >= min_overlap_fraction
        for row in rows
    )
    high_work = {row["bge"]["encodes"] for row in rows}
    low_work = {row["gemma"]["generated_tokens"] for row in rows}
    if len(high_work) != 1 or len(low_work) != 1:
        raise LiveGateError("cohorts do not contain matched work")
    p0_energy = statistics.median(row["selected_gpu_cohort"]["energy_j"] for row in by_label["P0"])
    p2_energy = statistics.median(row["selected_gpu_cohort"]["energy_j"] for row in by_label["P2"])
    p0_high_p95 = statistics.median(row["bge"]["lat_us_p95"] for row in by_label["P0"])
    p2_high_p95 = statistics.median(row["bge"]["lat_us_p95"] for row in by_label["P2"])
    p0_low_p95 = statistics.median(row["gemma"]["lat_us_p95"] for row in by_label["P0"])
    p2_low_p95 = statistics.median(row["gemma"]["lat_us_p95"] for row in by_label["P2"])
    saving = 1.0 - p2_energy / p0_energy
    slowdown = p2_high_p95 / p0_high_p95
    low_slowdown = p2_low_p95 / p0_low_p95
    return {
        "p0_selected_gpu_energy_j_median": p0_energy,
        "p2_selected_gpu_energy_j_median": p2_energy,
        "selected_gpu_energy_saving_frac": saving,
        "p0_bge_p95_us_median": p0_high_p95,
        "p2_bge_p95_us_median": p2_high_p95,
        "bge_p95_ratio_p2_over_p0": slowdown,
        "p0_gemma_p95_us_median": p0_low_p95,
        "p2_gemma_p95_us_median": p2_low_p95,
        "gemma_p95_ratio_p2_over_p0": low_slowdown,
        "high_priority_gate": slowdown <= high_slowdown_gate,
        "low_priority_gate": low_slowdown <= low_slowdown_gate,
        "overlap_gate": overlap_gate,
        "minimum_overlap_s": min(row["concurrent_overlap_s"] for row in rows),
        "minimum_overlap_fraction_of_shorter": min(
            row["concurrent_overlap_fraction_of_shorter"] for row in rows
        ),
        "server_board_relief_gate": saving > 0.0,
        "matched_bge_encodes": high_work.pop(),
        "matched_gemma_tokens": low_work.pop(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bge-batch", type=int, default=16)
    parser.add_argument("--gemma-batch", type=int, default=64)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--n-gen", type=int, default=64)
    parser.add_argument("--bge-seq-target", type=int, default=32)
    parser.add_argument("--bge-words", type=int, default=25)
    parser.add_argument("--bge-reps", type=int, default=0, help="0 selects the measured 5-second minimum")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--high-slowdown-gate", type=float, default=1.05)
    parser.add_argument("--low-slowdown-gate", type=float, default=2.0)
    parser.add_argument("--min-overlap-s", type=float, default=0.1)
    parser.add_argument("--min-overlap-fraction", type=float, default=0.90)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--ready-timeout", type=int, default=120)
    parser.add_argument("--prompt", default="Explain batching.")
    parser.add_argument("--output", default=str(HERE / "cp_e_live_priority_result.json"))
    args = parser.parse_args()
    positive = (args.bge_batch, args.gemma_batch, args.requests, args.n_gen, args.bge_seq_target,
                args.bge_words, args.timeout, args.ready_timeout)
    if any(value <= 0 for value in positive) or args.bge_reps < 0 \
            or not all(math.isfinite(value) and value > 0 for value in (
                args.high_slowdown_gate, args.low_slowdown_gate, args.min_overlap_s,
                args.min_overlap_fraction,
            )):
        raise LiveGateError("invalid positive numeric argument")
    if args.repeats != 3 or args.requests % args.gemma_batch != 0:
        raise LiveGateError("requires three repeats and requests divisible by the Gemma batch")

    bge_points, knee, exact_tokens, atlas_digest = CPD.load_bge_profile(
        CPD.BGE_PROFILE, args.bge_seq_target,
    )
    if knee != args.bge_batch:
        raise LiveGateError(f"requested BGE batch {args.bge_batch} differs from measured knee {knee}")
    bge_point = next(point for point in bge_points if point.batch_size == args.bge_batch)
    minimum_bge_reps = math.ceil(5_000_000 / bge_point.duration_us)
    bge_reps = minimum_bge_reps if args.bge_reps == 0 else args.bge_reps
    if bge_reps < minimum_bge_reps:
        raise LiveGateError(f"BGE repetitions {bge_reps} are below the 5-second minimum {minimum_bge_reps}")
    gemma_points, gemma_digest = CPD.load_gemma_profile(
        HERE / "stage_d_batch_scaling.json", args.gemma_batch,
    )
    decisions = CPD.policy_decisions(bge_points, gemma_points)
    artifacts = capture_artifacts()
    reference = SB.run_mono_reference(
        SELECTED_GPU_UUID, args.prompt, args.n_gen, args.timeout,
        batch=args.gemma_batch, ctx=512, max_prefill=64,
    )
    rows = []
    failed_label = None
    failed_repeat = None
    try:
        for repeat in range(args.repeats):
            order = ("P0", "P2") if repeat % 2 == 0 else ("P2", "P0")
            for label in order:
                failed_label = label
                failed_repeat = repeat
                rows.append(run_cohort(
                    label, repeat, args.bge_batch, args.gemma_batch, args.requests, args.n_gen,
                    args.bge_words, exact_tokens, bge_reps, args.prompt, reference,
                    args.ready_timeout, args.timeout,
                ))
    except Exception as exc:  # noqa: BLE001
        failure = {
            "schema": "s14-cp-e-live-priority-v1",
            "status": "LIVE_MIXED_PRIORITY_FAIL_MEASUREMENT",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_label": failed_label,
            "failed_repeat": failed_repeat,
            "completed_rows": rows,
            "artifacts": artifacts,
            "route": {"op15": [0, K1], "op12": [K1, K2], "server": [K2, SB.N_LAYER]},
            "batches": {"high_priority_bge": args.bge_batch, "low_priority_gemma": args.gemma_batch},
            "requests_per_cohort": args.requests,
            "reference_tokens": reference,
            "limits": [
                "No energy, latency, or scheduler PASS is authorized from this failed acquisition.",
                "Phone/USB/host-wall/total energy remain unknown.",
            ],
        }
        Path(args.output).write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps({
            "status": failure["status"],
            "failed_label": failed_label,
            "failed_repeat": failed_repeat,
            "error": str(exc),
        }, indent=2))
        return 2
    summary = summarize(
        rows,
        args.high_slowdown_gate,
        args.low_slowdown_gate,
        args.min_overlap_s,
        args.min_overlap_fraction,
    )
    if not CPD.second_gpu_idle():
        raise LiveGateError("second GPU active after cohort")
    passed = summary["high_priority_gate"] and summary["low_priority_gate"] \
        and summary["overlap_gate"] and summary["server_board_relief_gate"]
    result = {
        "schema": "s14-cp-e-live-priority-v1",
        "status": "LIVE_MIXED_PRIORITY_PASS" if passed else "LIVE_MIXED_PRIORITY_FAIL_GATE",
        "scope": "one selected A6000 GPU_BOARD plus live OP15/OP12 mechanics; phone/USB/host-wall/total energy unknown",
        "selected_gpu_uuid": SELECTED_GPU_UUID,
        "second_gpu_uuid": SECOND_GPU_UUID,
        "second_gpu_idle_at_endpoints": True,
        "experiment_cuda_visible_devices": SELECTED_GPU_UUID,
        "priority_provenance": "synthetic",
        "slo_provenance": "synthetic relative p95 gates",
        "gates": {
            "high_priority_p95_ratio_max": args.high_slowdown_gate,
            "low_priority_p95_ratio_max": args.low_slowdown_gate,
            "concurrent_overlap_s_min": args.min_overlap_s,
            "concurrent_overlap_fraction_of_shorter_min": args.min_overlap_fraction,
            "phone_start_millic_max": PHONE_START_MAX_MILLIC,
            "phone_end_millic_max": PHONE_END_MAX_MILLIC,
        },
        "policy_decisions": decisions,
        "profiles": {"bge_server_atlas": atlas_digest, "gemma_phone_batch": gemma_digest},
        "artifacts": artifacts,
        "route": {"op15": [0, K1], "op12": [K1, K2], "server": [K2, SB.N_LAYER]},
        "batches": {"high_priority_bge": args.bge_batch, "low_priority_gemma": args.gemma_batch},
        "workload": {
            "bge": {
                "seq_len_target": args.bge_seq_target,
                "seq_len_exact": exact_tokens,
                "words": args.bge_words,
                "reps": bge_reps,
                "isolated_window_us_min": 5_000_000,
            },
            "gemma": {
                "prompt": args.prompt,
                "n_gen": args.n_gen,
                "warmup_groups": DRIVER_WARMUP_GROUPS,
            },
        },
        "requests_per_cohort": args.requests,
        "reference_tokens": reference,
        "summary": summary,
        "rows": rows,
        "limits": [
            "Priority and SLO labels are synthetic.",
            "CUDA process scheduling is measured behavior, not kernel-level preemption.",
            "Only selected-GPU board energy is measured; total-system energy is unknown.",
        ],
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], **summary}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
