#!/usr/bin/env python3
"""Measure resident desktop and whole-phone routes on one exact model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


HERE = Path(__file__).resolve().parent
S42_ROOT = HERE.parent
KERNEL_ENERGY = S42_ROOT / "kernel_energy_v1"
if str(KERNEL_ENERGY) not in sys.path:
    sys.path.insert(0, str(KERNEL_ENERGY))

import energy_common  # noqa: E402
import measure_desktop  # noqa: E402


CONFIRMATION = "RUN_S42_WHOLE_TASK_PHONE_V1"
SERIAL = "3C15AU002CL00000"
MODEL_ID = "llama-3.2-1b-instruct-q4_0"
MODEL_SHA256 = "4b90b1d7ae7324676194755a6dfce11cb6e457982c4c01a1db2857be1ed064ad"
MODEL_BYTES = 770_928_288
TRACE_SCHEMA = "s42-six-model-burstgpt-mixed-v1"
RESULT_SCHEMA = "s42-whole-task-resident-route-result-v1"
LOGGER = "/data/local/tmp/s41-opoffload-dmabuf-v1/phone_power_logger.sh"
POLICY = "/data/local/tmp/s41-opoffload-dmabuf-v1/phone_power_allow.rules"
ROUTES = ("desktop-cuda", "desktop-cpu", "phone-adreno")


class CampaignError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run(
    command: list[str], *, check: bool = True, timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=timeout,
        check=check,
    )


def adb(
    *args: str, check: bool = True, timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    return run(
        ["adb", "-P", "5037", "-s", SERIAL, *args],
        check=check,
        timeout=timeout,
    )


def adb_su(
    command: str, *, check: bool = True, timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    return adb(
        "shell", f"su -c {shlex.quote(command)}", check=check, timeout=timeout
    )


def phone_file_exists(path: str) -> bool:
    return adb_su(f"test -f {shlex.quote(path)}", check=False).returncode == 0


def wait_phone_file(path: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if phone_file_exists(path):
            return
        time.sleep(0.1)
    raise CampaignError(f"phone marker timeout: {path}")


def phone_anchor() -> dict[str, int]:
    before = time.monotonic_ns()
    value = adb("shell", "cat", "/proc/uptime").stdout.split()[0]
    after = time.monotonic_ns()
    return {
        "host_monotonic_ns": (before + after) // 2,
        "phone_uptime_ns": int(round(float(value) * 1e9)),
        "round_trip_ns": after - before,
    }


def map_host_to_phone(
    host_ns: int, before: dict[str, int], after: dict[str, int]
) -> int:
    host_delta = after["host_monotonic_ns"] - before["host_monotonic_ns"]
    phone_delta = after["phone_uptime_ns"] - before["phone_uptime_ns"]
    require(host_delta > 0 and phone_delta > 0, "phone clock direction")
    estimate_before = (
        before["phone_uptime_ns"]
        + host_ns
        - before["host_monotonic_ns"]
    )
    estimate_after = (
        after["phone_uptime_ns"]
        + host_ns
        - after["host_monotonic_ns"]
    )
    require(
        abs(estimate_before - estimate_after) <= 50_000_000,
        "phone clock anchor disagreement",
    )
    return (estimate_before + estimate_after) // 2


def http_json(url: str, payload: object | None = None, timeout: float = 30.0) -> Any:
    data = None if payload is None else canonical(payload)
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_server(port: int, process: "ManagedProcess", timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CampaignError(f"server exited before ready: {process.name}")
        try:
            value = http_json(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if value.get("status") == "ok":
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    raise CampaignError(f"server readiness timeout: {process.name}")


class ManagedProcess:
    def __init__(
        self,
        name: str,
        command: list[str],
        log_path: Path,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.log_stream = log_path.open("xb")
        self.process = subprocess.Popen(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=self.log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def stop(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)
        self.log_stream.close()


def desktop_server_command(
    binary: Path,
    model: Path,
    alias: str,
    port: int,
    backend: str,
) -> list[str]:
    command = [
        str(binary),
        "--model", str(model),
        "--alias", alias,
        "--fit", "off",
        "--ctx-size", "4096",
        "--parallel", "1",
        "--batch-size", "2048",
        "--ubatch-size", "512",
        "--cont-batching",
        "--cache-type-k", "f16",
        "--cache-type-v", "f16",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-webui",
        "--log-colors", "off",
    ]
    if backend == "cuda":
        command.extend([
            "--flash-attn", "on",
            "--split-mode", "none",
            "--n-gpu-layers", "all",
            "--main-gpu", "0",
            "--device", "CUDA0",
        ])
    else:
        command.extend([
            "--n-gpu-layers", "0",
            "--threads", "8",
            "--threads-batch", "8",
        ])
    return command


def phone_server_command(bin_dir: str, model: str, port: int) -> list[str]:
    body = " ".join(shlex.quote(value) for value in [
        f"{bin_dir}/llama-server",
        "--model", model,
        "--alias", MODEL_ID,
        "--ctx-size", "4096",
        "--parallel", "1",
        "--batch-size", "1024",
        "--ubatch-size", "256",
        "--cont-batching",
        "--cache-type-k", "f16",
        "--cache-type-v", "f16",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--n-gpu-layers", "99",
        "--device", "GPUOpenCL",
        "--flash-attn", "off",
        "--no-webui",
        "--log-colors", "off",
    ])
    shell = (
        f"cd {shlex.quote(bin_dir)} && "
        f"export LD_LIBRARY_PATH=. && exec {body}"
    )
    return [
        "adb", "-P", "5037", "-s", SERIAL,
        "shell", f"su -c {shlex.quote(shell)}",
    ]


def completion(port: int, row: dict[str, Any]) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    response = http_json(
        f"http://127.0.0.1:{port}/completion",
        {
            "cache_prompt": False,
            "ignore_eos": True,
            "n_predict": row["output_tokens"],
            "prompt": row["prompt_tokens"],
            "stream": False,
            "temperature": 0,
        },
        timeout=120.0,
    )
    completed_ns = time.monotonic_ns()
    require(
        response.get("tokens_predicted") == row["output_tokens"],
        "output token count",
    )
    timings = response.get("timings")
    require(type(timings) is dict, "response timings")
    return {
        "content_sha256": hashlib.sha256(
            str(response.get("content", "")).encode("utf-8")
        ).hexdigest(),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
        "tokens_evaluated": response.get("tokens_evaluated"),
        "tokens_predicted": response.get("tokens_predicted"),
        "wall_us": (completed_ns - started_ns) // 1000,
    }


def read_trace(path: Path, request_indices: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = [
        row for row in rows if row.get("mixed_request_index") in request_indices
    ]
    require(len(selected) == len(request_indices), "request selection")
    require(
        all(
            row.get("schema") == TRACE_SCHEMA
            and row.get("execution_model_id") == MODEL_ID
            and row.get("prompt_transport") == "tokens"
            and len(row.get("prompt_tokens", [])) == row.get("input_tokens")
            for row in selected
        ),
        "trace request identity",
    )
    return sorted(selected, key=lambda row: request_indices.index(
        row["mixed_request_index"]
    ))


def phone_temperature_millic() -> int:
    raw = adb_su("cat /sys/class/power_supply/battery/temp").stdout
    return int(raw.strip()) * 100


def wait_gpu_idle(timeout_s: float = 60.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    consecutive = 0
    latest: dict[str, object] | None = None
    while time.monotonic() < deadline:
        raw = run([
            "nvidia-smi",
            "--query-gpu=uuid,pstate,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]).stdout.strip()
        fields = [field.strip() for field in raw.split(",")]
        require(len(fields) == 4, "GPU idle sample")
        latest = {
            "power_w": float(fields[3]),
            "pstate": fields[1],
            "utilization_pct": int(fields[2]),
            "uuid": fields[0],
        }
        idle = (
            fields[0] == measure_desktop.GPU_UUID
            and fields[1] == "P8"
            and int(fields[2]) == 0
            and float(fields[3]) <= 12.0
        )
        consecutive = consecutive + 1 if idle else 0
        if consecutive >= 3:
            return latest
        time.sleep(0.5)
    raise CampaignError(f"GPU did not reach resident idle state: {latest}")


def start_phone_logger(case_id: str) -> dict[str, str]:
    phone_root = f"/data/local/tmp/s42-whole-task-phone-v1/{case_id}"
    paths = {
        "root": phone_root,
        "samples": f"{phone_root}/samples.tsv",
        "active": f"{phone_root}/active",
        "armed": f"{phone_root}/power.armed",
        "done": f"{phone_root}/power.done",
        "log": f"{phone_root}/logger.log",
    }
    require(
        adb_su(f"test ! -e {shlex.quote(phone_root)}", check=False).returncode
        == 0,
        "phone case output exists",
    )
    adb_su(f"mkdir -p {shlex.quote(phone_root)}")
    command = (
        f"nohup sh {shlex.quote(LOGGER)} {shlex.quote(paths['samples'])} "
        f"{shlex.quote(paths['active'])} {shlex.quote(paths['armed'])} "
        f"{shlex.quote(paths['done'])} 30 > {shlex.quote(paths['log'])} "
        "2>&1 </dev/null &"
    )
    adb_su(command)
    wait_phone_file(paths["armed"], 15)
    adb_su(f"touch {shlex.quote(paths['active'])}")
    return paths


def stop_phone_logger(paths: dict[str, str], output: Path) -> list[dict[str, Any]]:
    adb_su(f"rm -f {shlex.quote(paths['active'])}")
    wait_phone_file(paths["done"], 15)
    samples = adb_su(f"cat {shlex.quote(paths['samples'])}").stdout
    output.write_text(samples, encoding="ascii")
    return energy_common.read_phone_samples(output)


def measure_case(
    case_id: str,
    route: str,
    row: dict[str, Any] | None,
    idle_duration_s: float,
    ports: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    gpu_idle_before = wait_gpu_idle()
    temperature_before = phone_temperature_millic()
    require(temperature_before <= 45_000, "phone thermal precondition")
    prefix = hashlib.sha256(str(output_dir).encode("ascii")).hexdigest()[:12]
    phone_paths = start_phone_logger(f"{prefix}-{case_id}")
    sampler = measure_desktop.Sampler(0.1)
    sampler_started = False
    logger_stopped = False
    try:
        time.sleep(0.7)
        sampler.start()
        sampler_started = True
        time.sleep(0.7)
        anchor_before = phone_anchor()
        paid_start_ns = time.monotonic_ns()
        if route == "idle":
            time.sleep(idle_duration_s)
            response = None
        else:
            require(row is not None, "route request")
            response = completion(ports[route], row)
        paid_end_ns = time.monotonic_ns()
        anchor_after = phone_anchor()
        time.sleep(0.7)
        sampler.stop()
        sampler_started = False
        phone_samples = stop_phone_logger(
            phone_paths, output_dir / f"{case_id}.phone.tsv"
        )
        logger_stopped = True
    finally:
        if sampler_started:
            sampler.stop()
        if not logger_stopped:
            adb_su(f"rm -f {shlex.quote(phone_paths['active'])}", check=False)
    phone_start_ns = map_host_to_phone(
        paid_start_ns, anchor_before, anchor_after
    )
    phone_end_ns = map_host_to_phone(paid_end_ns, anchor_before, anchor_after)
    phone_energy = energy_common.phone_energy_summary(
        phone_samples, phone_start_ns, phone_end_ns
    )
    cpu_j = energy_common.integrate_rapl(
        sampler.rows, paid_start_ns, paid_end_ns, "monotonic_ns"
    )
    gpu_j = energy_common.integrate_power(
        [
            (int(sample["monotonic_ns"]), sample["gpu_power_mw"] / 1000.0)
            for sample in sampler.rows
        ],
        paid_start_ns,
        paid_end_ns,
    )
    duration_s = (paid_end_ns - paid_start_ns) / 1e9
    phone_j = phone_energy["whole_phone_energy_j"]
    gpu_utilization = [
        sample["gpu_utilization_pct"]
        for sample in sampler.rows
        if paid_start_ns <= sample["monotonic_ns"] <= paid_end_ns
    ]
    if route in {"idle", "desktop-cpu", "phone-adreno"}:
        require(max(gpu_utilization, default=0) <= 5, "GPU was not idle")
    return {
        "accounted_fleet_energy_j": cpu_j + gpu_j + phone_j,
        "anchors": {"after": anchor_after, "before": anchor_before},
        "case_id": case_id,
        "cpu_package_average_power_w": cpu_j / duration_s,
        "cpu_package_energy_j": cpu_j,
        "duration_s": duration_s,
        "gpu_board_average_power_w": gpu_j / duration_s,
        "gpu_board_energy_j": gpu_j,
        "gpu_idle_before": gpu_idle_before,
        "gpu_utilization_max_pct": max(gpu_utilization, default=0),
        "input_tokens": 0 if row is None else row["input_tokens"],
        "mixed_request_index": (
            None if row is None else row["mixed_request_index"]
        ),
        "output_tokens": 0 if row is None else row["output_tokens"],
        "paid_end_monotonic_ns": paid_end_ns,
        "paid_start_monotonic_ns": paid_start_ns,
        "phone": phone_energy,
        "phone_temperature_after_millic": phone_temperature_millic(),
        "phone_temperature_before_millic": temperature_before,
        "response": response,
        "route": route,
        "sample_count_desktop": len(sampler.rows),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    require(args.execute and args.confirm == CONFIRMATION, "confirmation")
    require(args.output.is_absolute() and not args.output.exists(), "output")
    for path in (
        args.trace,
        args.desktop_model,
        args.desktop_cuda_server,
        args.desktop_cpu_server,
        args.cuda_lib_dir,
        args.cpu_lib_dir,
    ):
        require(path.exists(), f"missing path: {path}")
    require(
        args.desktop_model.stat().st_size == MODEL_BYTES
        and digest_file(args.desktop_model) == MODEL_SHA256,
        "desktop model identity",
    )
    require(
        adb("get-state").stdout.strip() == "device", "OP15 unavailable"
    )
    require(
        adb_su(
            f"test -x {shlex.quote(args.phone_bin_dir + '/llama-server')} "
            f"-a -f {shlex.quote(args.phone_model)}"
        ).returncode
        == 0,
        "phone runtime",
    )
    phone_model = adb_su(
        f"sha256sum {shlex.quote(args.phone_model)}"
    ).stdout.split()[0]
    require(phone_model == MODEL_SHA256, "phone model identity")
    foreign = run(["pgrep", "-x", "llama-server"], check=False)
    require(foreign.returncode == 1, "foreign desktop llama-server")
    phone_foreign = adb_su("pidof llama-server", check=False)
    require(phone_foreign.returncode != 0, "foreign phone llama-server")
    compute = run([
        "nvidia-smi",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]).stdout.strip()
    require(not compute, "foreign CUDA process")
    adb_su(f"/product/bin/magiskpolicy --live --apply {POLICY}")
    return {
        "desktop_cpu_server_sha256": digest_file(args.desktop_cpu_server),
        "desktop_cuda_server_sha256": digest_file(args.desktop_cuda_server),
        "desktop_model_sha256": MODEL_SHA256,
        "gpu": measure_desktop.gpu_sample(),
        "hostname": run(["hostname"]).stdout.strip(),
        "phone_bin_dir": args.phone_bin_dir,
        "phone_kernel": adb("shell", "uname", "-r").stdout.strip(),
        "phone_model_sha256": phone_model,
        "phone_serial": SERIAL,
        "phone_server_sha256": adb_su(
            f"sha256sum {shlex.quote(args.phone_bin_dir + '/llama-server')}"
        ).stdout.split()[0],
        "trace_sha256": digest_file(args.trace),
    }


def start_servers(
    args: argparse.Namespace, output: Path
) -> tuple[list[ManagedProcess], dict[str, int], int]:
    ports = {
        "desktop-cuda": args.cuda_port,
        "desktop-cpu": args.cpu_port,
        "phone-adreno": args.phone_forward_port,
    }
    cuda_environment = {
        **os.environ,
        "LD_LIBRARY_PATH": str(args.cuda_lib_dir),
    }
    cpu_environment = {
        **os.environ,
        "LD_LIBRARY_PATH": str(args.cpu_lib_dir),
    }
    processes = [
        ManagedProcess(
            "desktop-cuda",
            desktop_server_command(
                args.desktop_cuda_server,
                args.desktop_model,
                f"{MODEL_ID}-cuda",
                args.cuda_port,
                "cuda",
            ),
            output / "desktop-cuda-server.log",
            cuda_environment,
        ),
        ManagedProcess(
            "desktop-cpu",
            desktop_server_command(
                args.desktop_cpu_server,
                args.desktop_model,
                f"{MODEL_ID}-cpu",
                args.cpu_port,
                "cpu",
            ),
            output / "desktop-cpu-server.log",
            cpu_environment,
        ),
    ]
    adb("forward", f"tcp:{args.phone_forward_port}", f"tcp:{args.phone_port}")
    phone = ManagedProcess(
        "phone-adreno",
        phone_server_command(args.phone_bin_dir, args.phone_model, args.phone_port),
        output / "phone-adreno-server.log",
    )
    processes.append(phone)
    for process, port in zip(processes, ports.values()):
        wait_server(port, process, 90)
    phone_pids = adb_su("pidof llama-server").stdout.split()
    require(len(phone_pids) == 1 and phone_pids[0].isdigit(), "phone server pid")
    return processes, ports, int(phone_pids[0])


def stop_servers(
    processes: list[ManagedProcess], phone_pid: int | None, phone_forward_port: int
) -> None:
    for process in reversed(processes):
        process.stop()
    if phone_pid is not None:
        adb_su(f"kill -INT {phone_pid}", check=False)
        time.sleep(0.3)
        adb_su(f"kill -TERM {phone_pid}", check=False)
    adb("forward", "--remove", f"tcp:{phone_forward_port}", check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--desktop-model", type=Path, required=True)
    parser.add_argument("--desktop-cuda-server", type=Path, required=True)
    parser.add_argument("--desktop-cpu-server", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--cpu-lib-dir", type=Path, required=True)
    parser.add_argument(
        "--phone-bin-dir", default="/data/local/tmp/llama-ubatch-op15/bin"
    )
    parser.add_argument(
        "--phone-model",
        default=(
            "/data/local/tmp/unifer/llamacpp/"
            "Llama-3.2-1B-Instruct-Q4_0.gguf"
        ),
    )
    parser.add_argument("--request-indices", default="2,50,99")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--idle-duration-s", type=float, default=5.0)
    parser.add_argument("--cuda-port", type=int, default=19380)
    parser.add_argument("--cpu-port", type=int, default=19381)
    parser.add_argument("--phone-port", type=int, default=18382)
    parser.add_argument("--phone-forward-port", type=int, default=29382)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()

    require(args.repetitions >= 1, "repetitions")
    request_indices = tuple(int(value) for value in args.request_indices.split(","))
    require(len(request_indices) >= 1 and len(set(request_indices)) == len(request_indices), "request indices")
    requests = read_trace(args.trace, request_indices)
    identity = preflight(args)
    args.output.mkdir(parents=True)
    (args.output / "preflight.json").write_bytes(canonical(identity))

    processes: list[ManagedProcess] = []
    phone_pid: int | None = None
    cases: list[dict[str, Any]] = []
    failure: str | None = None
    try:
        processes, ports, phone_pid = start_servers(args, args.output)
        warm = dict(min(requests, key=lambda row: row["input_tokens"]))
        warm["output_tokens"] = 2
        for route in ROUTES:
            completion(ports[route], warm)
        time.sleep(2.0)
        for repetition in range(1, args.repetitions + 1):
            idle_id = f"idle-r{repetition}"
            cases.append(measure_case(
                idle_id,
                "idle",
                None,
                args.idle_duration_s,
                ports,
                args.output,
            ))
            route_order = ROUTES[repetition - 1:] + ROUTES[:repetition - 1]
            for row in requests:
                for route in route_order:
                    case_id = (
                        f"{route}-q{row['mixed_request_index']}-r{repetition}"
                    )
                    cases.append(measure_case(
                        case_id,
                        route,
                        row,
                        args.idle_duration_s,
                        ports,
                        args.output,
                    ))
                    time.sleep(1.0)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        stop_servers(processes, phone_pid, args.phone_forward_port)

    if failure is not None:
        (args.output / "FAILURE.json").write_bytes(canonical({
            "error": failure,
            "schema": "s42-whole-task-resident-route-failure-v1",
            "status": "FAIL",
        }))
        raise CampaignError(failure)
    result = {
        "cases": cases,
        "identity": identity,
        "model_id": MODEL_ID,
        "request_indices": list(request_indices),
        "repetitions": args.repetitions,
        "routes": list(ROUTES),
        "schema": RESULT_SCHEMA,
        "status": "PASS",
    }
    (args.output / "RESULT.json").write_bytes(canonical(result))
    print(json.dumps({
        "cases": len(cases),
        "output": str(args.output / "RESULT.json"),
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignError as error:
        print(f"S42_WHOLE_TASK_PHONE_ERROR: {error}")
        raise SystemExit(2)
