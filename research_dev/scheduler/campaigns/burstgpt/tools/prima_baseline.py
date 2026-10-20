"""prima.cpp baseline (2026-10-16): desktop rank 0 (CUDA) + phones (CPU) serve ONE model over prima.cpp's ZeroMQ ring.

The ring is carried by adb port tunnels (no phone network settings): every node addresses its peers at 127.0.0.1 and
prima.cpp maps rank r to ports DATA+r / SIGNAL+r, so ``adb forward`` (host -> phone r) and ``adb reverse`` (phone r ->
host, for the master and for the next phone) make the loopback endpoints meet; this is the RPC baseline's transport.

What is reused from default_llamacpp_baseline: the replay plan, the request body, the SSE client, the host energy
sampler (RAPL + NVML) and RESULT.json; phone energy comes from meter_phones.sh around this run (as the RPC baseline).
The 1B overlay model gets its own stock-mode server (world 1) on the desktop GPU, as in every other arm.

Usage (on the desktop, PYTHONPATH = deployed source):
  prima_baseline.py run --plan PLAN.json --out OUT --server prima/llama-server --phones PHONES.json
      --ring-model qwen3-32b-q8_0 [--overlay-model llama-3.2-1b-instruct-q4_0] [--ctx 16384] [--parallel 4]
      [--batch 2048] [--ubatch 512] [--layer-window 40,12,12] [--prefetch] [--gpu-mem GIB] [--threads N]
      [--library-dir DIR ...] [--port 18620] [--overlay-port 18621] [--startup-timeout-s 2400]
      [--max-duration-s 14400] [--switch-gate none|fifo-drain] [--gpu-index I] [--allow-partial-energy]

PHONES.json (schema s43-prima-phones-v1): {"world": 3, "phones": [
  {"name": "op15", "rank": 1, "transport": "adb", "serial": "...", "adb_port": 5037,
   "phone_dir": "/data/local/tmp/s43-prima-v1", "model_path": "/data/local/tmp/s43-q8/Qwen3-32B-Q8_0.gguf",
   "threads": null, "min_mem_available_bytes": 0},
  {"name": "sim", "rank": 2, "transport": "local", "binary": "/path/llama-cli", "model_path": "...", "workdir": "..."}]}
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from research_dev.scheduler.campaigns.burstgpt.common import canonical, load_object
from research_dev.scheduler.campaigns.burstgpt.tools import default_llamacpp_baseline as baseline

ARM = "prima"
PHONES_SCHEMA = "s43-prima-phones-v1"
RECEIPT_SCHEMA = "s43-prima-receipt-v1"
DATA_PORT = 9000
SIGNAL_PORT = 10000
HOST = "127.0.0.1"


class PrimaError(RuntimeError):
    pass


def _run(argv: Sequence[str], timeout_s: float = 60.0, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
    if check and proc.returncode != 0:
        raise PrimaError("%s -> %d: %s" % (" ".join(argv[:6]), proc.returncode, (proc.stderr or proc.stdout)[:300]))
    return proc


# ---------------------------------------------------------------- phones

class Phone:
    def __init__(self, row: Mapping[str, Any], world: int) -> None:
        self.name = str(row["name"])
        self.rank = int(row["rank"])
        self.transport = str(row.get("transport", "adb"))
        self.model_path = str(row["model_path"])
        self.threads = row.get("threads")
        self.world = world
        if not 0 < self.rank < world:
            raise PrimaError("phone %s: rank %d outside 1..%d" % (self.name, self.rank, world - 1))
        if self.transport == "adb":
            self.serial = str(row["serial"])
            self.adb_port = int(row.get("adb_port", 5037))
            self.phone_dir = str(row.get("phone_dir", "/data/local/tmp/s43-prima-v1"))
            self.min_mem_available_bytes = int(row.get("min_mem_available_bytes", 0))
        elif self.transport == "local":
            self.binary = str(row["binary"])
            self.workdir = str(row.get("workdir") or Path(self.binary).parent)
        else:
            raise PrimaError("phone %s: transport must be adb or local" % self.name)
        self.pids: list[int] = []
        self.process: subprocess.Popen | None = None
        self._log_path: Path | None = None
        self.tunnels: list[tuple[str, int]] = []  # ("forward"|"reverse", port)

    # adb helpers
    def adb(self, *args: str) -> list[str]:
        return ["adb", "-P", str(self.adb_port), "-s", self.serial, *args]

    def shell(self, command: str, timeout_s: float = 60.0) -> str:
        return _run(self.adb("shell", command), timeout_s).stdout.replace("\r", "")

    @property
    def next_rank(self) -> int:
        return (self.rank + 1) % self.world

    def worker_argv(self) -> list[str]:
        argv = ["./llama-cli" if self.transport == "adb" else self.binary, "-m", self.model_path,
                "--world", str(self.world), "--rank", str(self.rank), "--master", HOST, "--next", HOST]
        if self.threads:
            argv += ["-t", str(int(self.threads))]
        return argv

    def preflight(self, host_model_bytes: int | None) -> dict[str, Any]:
        if self.transport == "local":
            size = Path(self.model_path).stat().st_size
            if host_model_bytes is not None and size != host_model_bytes:
                raise PrimaError("%s model size %d != host %d" % (self.name, size, host_model_bytes))
            return {"name": self.name, "transport": "local", "model_bytes": size}
        state = _run(self.adb("get-state"), 30, check=False).stdout.strip()
        if state != "device":
            raise PrimaError("%s (%s) adb state %r" % (self.name, self.serial, state))
        out = self.shell("stat -c %%s %s 2>/dev/null; echo ---; grep MemAvailable /proc/meminfo; echo ---; "
                         "ls %s/llama-cli %s/fio 2>&1; echo ---; pidof llama-cli; echo ---; df %s | tail -1"
                         % (shlex.quote(self.model_path), self.phone_dir, self.phone_dir, self.phone_dir))
        parts = [p.strip() for p in out.split("---")]
        size = int(parts[0]) if parts[0].isdigit() else None
        if size is None:
            raise PrimaError("%s: model missing on phone: %s" % (self.name, self.model_path))
        if host_model_bytes is not None and size != host_model_bytes:
            raise PrimaError("%s model size %d != host %d" % (self.name, size, host_model_bytes))
        mem_kb = int(re.search(r"(\d+)", parts[1]).group(1)) if re.search(r"(\d+)", parts[1]) else 0
        if mem_kb * 1024 < self.min_mem_available_bytes:
            raise PrimaError("%s MemAvailable %d MB below %d MB" % (self.name, mem_kb // 1024,
                                                                   self.min_mem_available_bytes // 2 ** 20))
        if "No such file" in parts[2]:
            raise PrimaError("%s: worker binaries missing: %s" % (self.name, parts[2]))
        stale = [int(p) for p in parts[3].split() if p.isdigit()]
        for pid in stale:  # only our own binary is ever killed
            exe = self.shell("readlink /proc/%d/exe 2>/dev/null" % pid).strip()
            if exe.startswith(self.phone_dir):
                self.shell("kill -KILL %d" % pid)
        return {"name": self.name, "serial": self.serial, "transport": "adb", "model_bytes": size,
                "mem_available_bytes": mem_kb * 1024, "stale_workers_killed": stale, "df": parts[4]}

    def open_tunnels(self) -> list[dict[str, Any]]:
        if self.transport != "adb":
            return []
        rows = []
        # a crashed earlier run may have left its tunnels behind: remove ours first (errors ignored)
        for base in (DATA_PORT, SIGNAL_PORT):
            _run(self.adb("forward", "--remove", "tcp:%d" % (base + self.rank)), 30, check=False)
            _run(self.adb("reverse", "--remove", "tcp:%d" % base), 30, check=False)
            if self.next_rank != 0:
                _run(self.adb("reverse", "--remove", "tcp:%d" % (base + self.next_rank)), 30, check=False)
        for base in (DATA_PORT, SIGNAL_PORT):
            # host -> this phone: the master (and anyone) reaches rank r at 127.0.0.1:base+r
            port = base + self.rank
            _run(self.adb("forward", "tcp:%d" % port, "tcp:%d" % port))
            self.tunnels.append(("forward", port))
            rows.append({"phone": self.name, "kind": "forward", "port": port})
            # this phone -> master at 127.0.0.1:base
            _run(self.adb("reverse", "tcp:%d" % base, "tcp:%d" % base))
            self.tunnels.append(("reverse", base))
            rows.append({"phone": self.name, "kind": "reverse", "port": base})
            # this phone -> next phone through the host's forward listener
            if self.next_rank != 0:
                nport = base + self.next_rank
                _run(self.adb("reverse", "tcp:%d" % nport, "tcp:%d" % nport))
                self.tunnels.append(("reverse", nport))
                rows.append({"phone": self.name, "kind": "reverse", "port": nport})
        return rows

    def close_tunnels(self) -> None:
        for kind, port in reversed(self.tunnels):
            _run(self.adb(kind, "--remove", "tcp:%d" % port), 30, check=False)
        self.tunnels = []

    def start(self, out: Path, prefetch: bool) -> dict[str, Any]:
        argv = self.worker_argv() + (["--prefetch"] if prefetch else [])
        started_ns = time.monotonic_ns()
        if self.transport == "local":
            self._log_path = out / ("worker-%s.log" % self.name)
            log = open(self._log_path, "wb")
            self.process = subprocess.Popen(argv, cwd=self.workdir, stdout=log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
            return {"name": self.name, "argv": argv, "pid": self.process.pid, "started_monotonic_ns": started_ns}
        quoted = " ".join(shlex.quote(a) for a in argv)
        # nohup+& over adb never returns (adbd waits on the pipe): the RPC baseline's setsid-in-subshell form
        command = ("cd %s && PATH=%s:$PATH && ( env setsid %s > worker.log 2>&1 < /dev/null & ) ; sleep 1; pidof llama-cli"
                   % (self.phone_dir, self.phone_dir, quoted))
        out_text = self.shell(command, 90)
        self.pids = [int(p) for p in out_text.split() if p.isdigit()]
        if not self.pids:
            raise PrimaError("%s: worker did not start: %r" % (self.name, out_text[-300:]))
        return {"name": self.name, "argv": argv, "pids": self.pids, "started_monotonic_ns": started_ns}

    def log_tail(self, lines: int = 40) -> str:
        if self.transport == "local":
            path = Path(self.workdir) / "worker.log"
            try:
                return "\n".join(Path(self._log_path).read_text(errors="replace").splitlines()[-lines:])
            except (OSError, AttributeError):
                return ""
        return _run(self.adb("shell", "tail -n %d %s/worker.log" % (lines, self.phone_dir)), 30, check=False).stdout

    def alive(self) -> bool:
        if self.transport == "local":
            return self.process is not None and self.process.poll() is None
        if not self.pids:
            return False
        out = _run(self.adb("shell", "pidof llama-cli"), 30, check=False).stdout
        return any(str(pid) in out.split() for pid in self.pids)

    def stop(self, out: Path) -> dict[str, Any]:
        row: dict[str, Any] = {"name": self.name}
        if self.transport == "local":
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            row["exit_code"] = None if self.process is None else self.process.returncode
            return row
        if self.pids and self.alive():
            pids = " ".join(map(str, self.pids))
            _run(self.adb("shell", "kill -TERM %s" % pids), 30, check=False)
            for _ in range(30):
                if not self.alive():
                    break
                time.sleep(1)
            if self.alive():
                _run(self.adb("shell", "kill -KILL %s" % pids), 30, check=False)
                row["killed"] = True
        else:
            row["already_exited"] = True  # prima workers leave when the master closes the ring
        _run(self.adb("pull", "%s/worker.log" % self.phone_dir, str(out / ("worker-%s.log" % self.name))), 120,
             check=False)
        row["clean"] = not self.alive()
        return row


def load_phones(path: Path) -> tuple[int, list[Phone]]:
    data = load_object(path)
    if data.get("schema") != PHONES_SCHEMA:
        raise PrimaError("phones file schema %r != %s" % (data.get("schema"), PHONES_SCHEMA))
    world = int(data["world"])
    phones = [Phone(row, world) for row in data["phones"]]
    ranks = sorted(p.rank for p in phones)
    if ranks != list(range(1, world)):
        raise PrimaError("phone ranks %s must be exactly 1..%d" % (ranks, world - 1))
    return world, sorted(phones, key=lambda p: p.rank)


# ---------------------------------------------------------------- servers

def ring_server_command(args: argparse.Namespace, model_path: str, world: int) -> list[str]:
    command = [args.server, "-m", model_path, "-a", args.ring_model, "--host", HOST, "--port", str(args.port),
               "-c", str(args.ctx), "-np", str(args.parallel), "-cb", "-b", str(args.batch), "-ub", str(args.ubatch)]
    if world > 1:
        command += ["--world", str(world), "--rank", "0", "--master", HOST, "--next", HOST]
        if args.layer_window:
            command += ["-lw", args.layer_window]
            if args.ngl is not None:  # fixed windows skip the solver, so the master's GPU layers must be given too
                command += ["-ngl", str(args.ngl)]
        if args.prefetch:
            command.append("--prefetch")
    elif args.ngl is not None:
        command += ["-ngl", str(args.ngl)]
    if args.gpu_mem is not None:
        command += ["--gpu-mem", str(args.gpu_mem)]
    if args.threads:
        command += ["-t", str(args.threads)]
    if args.keep_out_in_cuda:
        command.append("--keep-out-in-cuda")
    command += list(args.server_arg or [])
    return command


def overlay_server_command(args: argparse.Namespace, model_path: str) -> list[str]:
    return [args.server, "-m", model_path, "-a", args.overlay_model, "--host", HOST, "--port", str(args.overlay_port),
            "-c", str(args.overlay_ctx), "-np", "2", "-cb", "-ngl", "99"]


def wait_healthy(proc: baseline.RouterProcess, port: int, timeout_s: float, what: str) -> float:
    started = time.monotonic()
    while True:
        if not proc.alive():
            raise PrimaError("%s exited during startup (code %s); see its log" % (what, proc.process.returncode))
        try:
            if baseline.http_get_json(HOST, port, "/health").get("status") == "ok":
                return time.monotonic() - started
        except Exception:  # noqa: BLE001 - not ready yet
            pass
        if time.monotonic() - started > timeout_s:
            raise PrimaError("%s not healthy after %.0f s" % (what, timeout_s))
        time.sleep(1.0)


# common.cpp (prima.cpp): "  - Assignment Set : M1", "  - N Layer Window : 40", "  - N GPU Layers   : 12",
# "Using window size: %d, GPU layers: %d", "Remove device %s (rank %d) with only %d layer assigned.", "Device %d is weak"
LAYER_WINDOW_RE = re.compile(r"(N Layer Window|N GPU Layers|Assignment Set|Using window size|Remove device|is weak|"
                             r"Reassign layers|Rank \d+ +\|)")


def parse_layer_windows(lines: Sequence[baseline.LogLine]) -> list[str]:
    found = []
    for line in lines:
        if LAYER_WINDOW_RE.search(line.text):
            found.append(line.text.strip()[:200])
    return found[-24:]


# ---------------------------------------------------------------- run

class _Interrupted(Exception):
    pass


def run(args: argparse.Namespace) -> int:
    plan = load_object(Path(args.plan).resolve())
    baseline.validate_plan(plan)
    out = Path(args.out)
    args.server = str(Path(args.server).resolve())
    if args.phones:
        args.phones = str(Path(args.phones).resolve())
    args.library_dir = [str(Path(d).resolve()) for d in (args.library_dir or [])]
    receipt_workdir = args.workdir
    if out.exists() and any(out.iterdir()):
        raise SystemExit("output directory is not empty: " + str(out))
    server = Path(args.server)
    if not server.is_file():
        raise SystemExit("prima llama-server not found: " + str(server))
    models = {row["model_id"]: row for row in plan["models"]}
    if args.ring_model not in models:
        raise SystemExit("ring model %s not in plan models %s" % (args.ring_model, sorted(models)))
    request_models = {row["model_id"] for row in plan["requests"]}
    overlay_needed = args.overlay_model in request_models
    extra = request_models - {args.ring_model} - ({args.overlay_model} if overlay_needed else set())
    if extra:
        raise SystemExit("requests use models with no server: %s" % sorted(extra))
    world, phones = load_phones(Path(args.phones)) if args.phones else (1, [])
    ring_model_path = models[args.ring_model]["path"]
    host_model_bytes = Path(ring_model_path).stat().st_size
    for port in (args.port, args.overlay_port):
        if baseline.port_in_use(HOST, port):
            raise SystemExit("port %d is in use" % port)
    environment, overrides, removed = baseline.router_environment(os.environ, list(args.library_dir or []),
                                                                  server.parent, args.cuda_visible_devices)
    out.mkdir(parents=True, exist_ok=True)
    out = out.resolve()
    # prima's profiler benchmarks the disk under the cwd (fio test files): use a scratch dir on the model's
    # filesystem so Halda sees the disk that actually holds the weights, never the source tree
    workdir = Path(args.workdir).resolve() if args.workdir else out
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    streams = out / "streams"
    streams.mkdir()
    (out / "REPLAY_PLAN.json").write_bytes(canonical(plan))
    (out / "REPLAY_SCHEDULE.json").write_bytes(canonical(plan["replay_schedule"]))

    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "world": world, "phones_preflight": [], "tunnels": [],
                               "workers_started": [], "workers_stopped": [], "startup": {}, "layer_windows": []}
    ring_cmd = ring_server_command(args, ring_model_path, world)
    overlay_cmd = overlay_server_command(args, models[args.overlay_model]["path"]) if overlay_needed else None
    receipt["ring_server_command"] = ring_cmd
    receipt["overlay_server_command"] = overlay_cmd
    ring = baseline.RouterProcess(ring_cmd, environment, out / "server.log")
    overlay = baseline.RouterProcess(overlay_cmd, environment, out / "overlay-server.log") if overlay_cmd else None
    rapl_absent = baseline.rapl_probe()
    gpu_sampler = ("adapters.host_runtime.nvidia_gpu_snapshot" if args.gpu_index is None
                   else "nvidia-smi --id=%d (local multi-GPU host)" % args.gpu_index)
    sampler = baseline.HostEnergySampler(baseline.host_metric_callbacks(args.gpu_index))

    def sender(host, port, body, headers, path, timeout_s, clock):
        model = json.loads(body)["model"]
        target = args.port if model == args.ring_model else args.overlay_port
        return baseline.stream_completion(host, target, body, headers, path, timeout_s, clock)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(signum, _frame):
        raise _Interrupted("signal %d" % signum)

    for sig in previous:
        signal.signal(sig, interrupted)
    epoch_ns = paid_end_ns = time.monotonic_ns()
    replay_started = False
    outcome = baseline.ReplayOutcome()
    energy: dict[str, Any] = {"coverage": {}, "cpu_package_energy_j": None, "gpu_board_energy_j": None}
    diagnostics: dict[str, Any] = {}
    exit_codes: dict[str, Any] = {}
    failure: str | None = None
    try:
        for phone in phones:
            receipt["phones_preflight"].append(phone.preflight(host_model_bytes))
        for phone in phones:
            receipt["tunnels"] += phone.open_tunnels()
        for phone in phones:  # workers first: they block until the master's startup broadcast arrives
            receipt["workers_started"].append(phone.start(out, args.prefetch))
        sampler.start()
        t0 = time.monotonic()
        if overlay is not None:  # first, so the ring's VRAM profiling sees what the overlay model already holds
            overlay.start()
            receipt["startup"]["overlay_health_s"] = round(wait_healthy(overlay, args.overlay_port, 600,
                                                                        "overlay server"), 1)
        ring.start()
        receipt["startup"]["ring_health_s"] = round(wait_healthy(ring, args.port, args.startup_timeout_s,
                                                                 "ring server"), 1)
        receipt["layer_windows"] = parse_layer_windows(ring.snapshot())
        receipt["workers_removed_by_scheduler"] = []
        for phone in phones:
            if not phone.alive():
                tail = phone.log_tail()
                if "No layer is assigned" in tail or "Remove device" in tail:
                    # prima's device selection dropped this phone (weak device): a legitimate baseline outcome
                    receipt["workers_removed_by_scheduler"].append({"name": phone.name, "log_tail": tail[-600:]})
                    print("note: prima removed worker %s (device selection)" % phone.name, flush=True)
                else:
                    raise PrimaError("worker %s is not running after the ring came up: %s" % (phone.name, tail[-400:]))
        receipt["startup"]["total_s"] = round(time.monotonic() - t0, 1)
        receipt["models_listing"] = baseline.http_get_json(HOST, args.port, "/v1/models")
        if not baseline.wait_for_samples(sampler.latest_rows, time.monotonic_ns(), rapl_absent is None):
            raise PrimaError("host energy sampler is not producing samples")
        epoch_ns = time.monotonic_ns()
        replay_started = True
        max_ns = None if args.max_duration_s is None else epoch_ns + int(args.max_duration_s * 1e9)
        print("replay start: %d requests, gate %s, world %d, ring port %d, overlay %s" % (
            len(plan["requests"]), args.switch_gate, world, args.port, args.overlay_port if overlay else "-"),
            flush=True)
        outcome = baseline.replay(plan, HOST, args.port, args.switch_gate, epoch_ns, streams, deadline_ns=max_ns,
                                  tick=sampler.latest_rows, sender=sender)
        paid_end_ns = time.monotonic_ns()
        if not baseline.wait_for_samples(sampler.latest_rows, paid_end_ns, rapl_absent is None):
            print("warning: no host sample after the paid window end", file=sys.stderr, flush=True)
    except _Interrupted as error:
        outcome.aborted = str(error)
        paid_end_ns = time.monotonic_ns()
    except PrimaError as error:
        failure = str(error)
        print("prima baseline failed: " + failure, file=sys.stderr, flush=True)
    finally:
        exit_codes["ring_server"] = ring.stop()
        if overlay is not None:
            exit_codes["overlay_server"] = overlay.stop()
        for phone in reversed(phones):
            try:
                receipt["workers_stopped"].append(phone.stop(out))
            except Exception as error:  # noqa: BLE001
                receipt["workers_stopped"].append({"name": phone.name, "error": str(error)})
        for phone in phones:
            try:
                phone.close_tunnels()
            except Exception as error:  # noqa: BLE001
                receipt.setdefault("tunnel_errors", []).append({"name": phone.name, "error": str(error)})
        try:
            sampler.stop()
        except Exception as error:  # noqa: BLE001
            diagnostics["sampler_stop_error"] = "%s: %s" % (type(error).__name__, error)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        samples = list(sampler.rows())
        with (out / "resource-samples.jsonl").open("wb") as sink:
            for row in samples:
                sink.write(canonical(row))
    if replay_started and paid_end_ns > epoch_ns:
        energy = baseline.host_energy(samples, epoch_ns, paid_end_ns)
    diagnostics.update(sampler.diagnostics())
    (out / "host-power-diagnostics.json").write_bytes(canonical(diagnostics))
    receipt["layer_windows"] = receipt["layer_windows"] or parse_layer_windows(ring.snapshot())
    receipt["exit_codes"] = exit_codes
    receipt["failure"] = failure
    (out / "PRIMA_RECEIPT.json").write_bytes(canonical(receipt))
    server_block = {**baseline.server_identity(server), "environment_overrides": overrides,
                    "environment_removed": removed, "exit_code": exit_codes.get("ring_server"), "host": HOST,
                    "port": args.port, "router_command": ring_cmd, "overlay_command": overlay_cmd,
                    "prima": {"world": world, "layer_windows": receipt["layer_windows"],
                              "startup": receipt["startup"], "phones": [p.name for p in phones]}}
    harness = {"argv": sys.argv, "hostname": platform.node(), "python": platform.python_version(), "arm": ARM}
    if failure:
        outcome.aborted = outcome.aborted or ("STARTUP_FAILED: " + failure)
    result = baseline.build_result(plan, outcome, [], energy, epoch_ns, paid_end_ns,
                                   endpoint="http://%s:%d" % (HOST, args.port), rapl_absent_reason=rapl_absent,
                                   gpu_sampler=gpu_sampler, server=server_block, gate_mode=args.switch_gate,
                                   models_max=2 if overlay else 1, log_verbosity=None,
                                   host_diagnostics={k: v for k, v in diagnostics.items() if k != "events"},
                                   harness=harness)
    result["prima_receipt"] = receipt
    (out / "MODEL_PLACEMENTS.json").write_bytes(canonical({"instances": [], "by_model": result["model_placements"],
                                                           "schema": baseline.PLACEMENT_SCHEMA}))
    (out / "RESULT.json").write_bytes(canonical(result))
    print(json.dumps({"counts": result["counts"], "duration_us": result["duration_us"],
                      "energy_evidence": result["energy_evidence"], "status": result["status"],
                      "status_reasons": result["status_reasons"], "layer_windows": receipt["layer_windows"],
                      "startup": receipt["startup"]}, sort_keys=True), flush=True)
    if failure:
        return 2
    return 0 if result["status"] == "PASS" or (args.allow_partial_energy and not [
        reason for reason in result["status_reasons"] if not reason.startswith("ENERGY_")]) else 1


# ---------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--plan", required=True, help="replay plan from default_llamacpp_baseline plan")
    p.add_argument("--out", required=True)
    p.add_argument("--server", required=True, help="prima.cpp llama-server (rank 0 build, CUDA + HiGHS)")
    p.add_argument("--phones", help="PHONES.json (%s); omit for a single-node run" % PHONES_SCHEMA)
    p.add_argument("--ring-model", required=True, help="plan model_id served by the ring")
    p.add_argument("--overlay-model", default="llama-3.2-1b-instruct-q4_0")
    p.add_argument("--ctx", type=int, default=16384, help="server total context (old server: per slot = ctx/parallel)")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--ubatch", type=int, default=512)
    p.add_argument("--overlay-ctx", type=int, default=8192)
    p.add_argument("--layer-window", help="fixed -lw (skips profiling), e.g. 40,12,12")
    p.add_argument("--prefetch", action="store_true")
    p.add_argument("--gpu-mem", type=int, help="prima --gpu-mem (GiB) for rank 0")
    p.add_argument("--ngl", type=int, help="-ngl for rank 0 (single node, or with --layer-window)")
    p.add_argument("--keep-out-in-cuda", action="store_true")
    p.add_argument("--threads", type=int)
    p.add_argument("--server-arg", action="append", help="extra argument for the ring server (repeatable)")
    p.add_argument("--library-dir", action="append")
    p.add_argument("--workdir", help="cwd for the servers (prima profiles the disk here; default: --out)")
    p.add_argument("--cuda-visible-devices")
    p.add_argument("--gpu-index", type=int)
    p.add_argument("--port", type=int, default=18620)
    p.add_argument("--overlay-port", type=int, default=18621)
    p.add_argument("--startup-timeout-s", type=float, default=2400.0)
    p.add_argument("--max-duration-s", type=float, default=14_400.0)
    p.add_argument("--switch-gate", choices=baseline.GATE_MODES, default="none",
                   help="none: the two servers are independent (as in every other arm)")
    p.add_argument("--allow-partial-energy", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run(args)
    raise SystemExit(2)


if __name__ == "__main__":
    sys.exit(main())
