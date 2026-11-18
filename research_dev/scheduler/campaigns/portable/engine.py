"""The run engine of the portable path: one llama-server, an optional helper phone, a schedule.

Alone, the server runs plainly and the phone is not touched. Shared, the helper holds a column suffix of
each CPU-resident FFN layer; a request's share (``columns``) is applied at its first decoded token and the
server batches the slots that carry the same share into one phone call.

What the engine guarantees on a shared, busy host (each item was a lost run once):

* a metered phase starts only on a quiet host, is cancelled when other users' load trips the watch and is
  repeated when its mean excess was too high (``phase_retry``); the server and the phone session stay up;
* a finished request's slot is erased, so later tokens do not pay for its cells;
* loaded-idle windows before and after, and meters kept running until a fixed window closes, so two arms
  can be compared over the same time;
* on any failure the server is ended and the phone released before the error leaves.

Everything specific to the machine comes from the drivers; the engine itself names no device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable, Mapping, Sequence

from ... import AdaptiveDecodeControl, AdaptiveDecodePolicy, canonical_sha256
from ...adapters.contracts import PhysicalAdapterError
from ...adapters.http_backend import LlamaCppCompletionPayload, LlamaCppHttpClient
from ...configuration.site import SiteManifest
from ...drivers.adb import AdbGuard, sha256_file
from ...drivers.doctor import RunLock
from ...drivers.host import HostLaunchPolicy, LinuxNvidiaHostDriver, ServerLoadFailure
from ...drivers.meter import phone_meter as site_phone_meter
from ...drivers.phone import HelperBinding, phone_driver
from . import rundir
from .schedule import UNMETERED_PREFIXES, validate

SAMPLER_SCRIPT = "research_dev/scheduler/campaigns/burstgpt/tools/rig/phone_power_sampler.sh"


class PhaseCancelled(Exception):
    """Raised inside a request of a phase that other users' load disturbed."""


class HostBusy(RuntimeError):
    """Other users' load kept a phase from running clean."""


@dataclass(frozen=True)
class RunPlan:
    """What one run serves: the model, the server policy and, shared, the helper and its shards."""

    model_id: str
    alias: str
    port: int
    policy: HostLaunchPolicy
    n_embd: int
    n_ff: int
    artifact_sha256: str
    device_id: str | None = None
    binding: HelperBinding | None = None
    idle_s: float = 20.0
    tail_idle_s: float = 20.0
    slot_erase: bool = True

    def __post_init__(self) -> None:
        if (self.device_id is None) != (self.binding is None):
            raise PhysicalAdapterError("a shared run needs both the helper phone and its binding")
        if self.binding is not None and (self.binding.n_embd, self.binding.n_ff, self.binding.artifact_sha256) != (
                self.n_embd, self.n_ff, self.artifact_sha256):
            raise PhysicalAdapterError("the helper binding belongs to another model")
        if self.slot_erase and not self.policy.slot_save:
            raise PhysicalAdapterError("slot erase needs the server's slot-save path")

    @property
    def shared(self) -> bool:
        return self.binding is not None

    def alone(self) -> "RunPlan":
        """The same run with the host alone (the phone is not touched)."""
        return RunPlan(self.model_id, self.alias, self.port, self.policy, self.n_embd, self.n_ff,
                       self.artifact_sha256, None, None, self.idle_s, self.tail_idle_s, self.slot_erase)

    def to_json(self) -> dict[str, object]:
        return {"alias": self.alias, "artifact_sha256": self.artifact_sha256,
                "binding": None if self.binding is None else self.binding.to_json(), "device_id": self.device_id,
                "idle_s": self.idle_s, "model_id": self.model_id, "n_embd": self.n_embd, "n_ff": self.n_ff,
                "policy": self.policy.to_json(), "port": self.port, "slot_erase": self.slot_erase,
                "tail_idle_s": self.tail_idle_s}

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "RunPlan":
        binding = value.get("binding")
        return cls(str(value["model_id"]), str(value["alias"]), int(value["port"]),
                   HostLaunchPolicy.from_json(value["policy"]), int(value["n_embd"]), int(value["n_ff"]),
                   str(value["artifact_sha256"]), value.get("device_id"),
                   None if binding is None else HelperBinding.from_json(binding),
                   float(value.get("idle_s", 20.0)), float(value.get("tail_idle_s", 20.0)),
                   bool(value.get("slot_erase", True)))


@dataclass
class RunOptions:
    min_window_s: float = 0.0        # keep the meters running this long after the start of the last phase
    phase_retry: bool = True
    foreign_limit_cores: float = 6.0
    quiet_cores: float = 4.0
    quiet_s: float = 60.0
    excess_cores: float = 0.10
    rewarm_tokens: int = 300
    max_wait_s: float = 8 * 3600.0
    max_attempts: int = 40
    seed: int = 17
    request_timeout_s: float = 3600.0
    log: Callable[..., None] = field(default=lambda *words: print(*words, flush=True))


def run_configuration(site: SiteManifest, plan: RunPlan, argv: Sequence[str], environment: Mapping[str, str],
                      helper_kind: str | None) -> dict[str, object]:
    """``CONFIG.json`` of a run directory (the reports read the geometry and the helper's layers from it)."""
    helper = None
    if plan.binding is not None:
        helper = {"kind": helper_kind, "device_id": plan.device_id, "layers": list(plan.binding.layers),
                  "columns": plan.binding.stored_columns, "binding": plan.binding.to_json()}
    return {"site_id": site.site_id, "alias": plan.alias, "model_id": plan.model_id,
            "model": str(site.paths.models[plan.model_id]), "server": str(site.paths.server_path),
            "server_port": plan.port, "gpu_index": site.host.serving_gpu_index,
            "server_arguments": list(argv[1:]), "server_environment": dict(environment),
            "phone": {"artifact_sha256": plan.artifact_sha256, "columns": plan.n_ff, "n_embd": plan.n_embd},
            "helper": helper, "slot_erase": plan.slot_erase, "idle_s": plan.idle_s, "tail_idle_s": plan.tail_idle_s,
            "policy": plan.policy.to_json()}


def dry_run(site: SiteManifest, plan: RunPlan, output: Path, *, host=None, phone=None) -> dict[str, object]:
    """What a run would launch, without launching anything: the server command and environment and, shared,
    the phone-side launch command."""
    host = host or LinuxNvidiaHostDriver(site)
    helper_environment, launch = {}, None
    if plan.shared:
        phone = phone or phone_driver(site, plan.device_id)
        port = int(site.phone(plan.device_id).functionfs["bridge_port"]) if site.phone(plan.device_id).functionfs \
            else site.phone(plan.device_id).forward_port
        helper_environment = {"S41_SERVER_FFN_HOST": "127.0.0.1", "S41_SERVER_FFN_TRANSPORT": "tcp",
                              "S41_SERVER_FFN_PORT": str(port),
                              "S41_SERVER_FFN_LAYER_MASK": str(plan.binding.layer_mask)}
        launch = phone.launch_command(plan.binding) if hasattr(phone, "launch_command") else None
    argv, environment = host.command(plan.model_id, plan.alias, plan.port, plan.policy, binding=plan.binding,
                                     helper_environment=helper_environment, output=output)
    return {"argv": argv, "environment": environment, "phone_launch": launch}


class Engine:
    def __init__(self, site: SiteManifest, plan: RunPlan, schedule: Mapping[str, object], output: Path,
                 options: RunOptions | None = None, *, host=None, phone=None, host_meter=None, helper_meter=None,
                 client_factory: Callable = LlamaCppHttpClient, lock: Callable = RunLock,
                 clock_ns: Callable[[], int] = time.monotonic_ns, sleep: Callable[[float], None] = time.sleep) -> None:
        self.site, self.plan, self.schedule, self.output = site, plan, json.loads(json.dumps(schedule)), Path(output)
        self.options = options or RunOptions()
        self.host = host or LinuxNvidiaHostDriver(site)
        self.phone = phone if phone is not None or not plan.shared else phone_driver(site, plan.device_id)
        self.meter = host_meter or self.host.meter()
        if helper_meter is None and plan.shared:
            site_phone = site.phone(plan.device_id)
            helper_meter = site_phone_meter(AdbGuard(site.paths.adb_path, site_phone.adb_port), site_phone,
                                            site.paths.repo_root / SAMPLER_SCRIPT)
        self.helper_meter = helper_meter
        self._client_factory, self._lock, self._clock_ns, self._sleep = client_factory, lock, clock_ns, sleep
        self.abort = threading.Event()
        self.phase_cancel = threading.Event()
        self._mutex = threading.Lock()
        self._generation = 0
        self.failures: list[dict[str, object]] = []
        self.handle = None
        self.watch = None
        self.layers = plan.binding.layers if plan.binding is not None else ()

    # -- small helpers ---------------------------------------------------------------------------
    def _save(self, name: str, value: object) -> None:
        rundir.save(self.output / name, value)

    def _energy(self, start_ns: int, end_ns: int) -> dict[str, object]:
        return self.meter.energy(start_ns, end_ns)

    def _window(self, seconds: float) -> tuple[int, int]:
        start = self._clock_ns()
        self._sleep(seconds)
        return start, self._clock_ns()

    def _wait_quiet(self, what: str) -> float:
        """Block until foreign work stayed below ``quiet_cores`` for ``quiet_s``; returns the seconds waited."""
        options, start, told = self.options, self._clock_ns(), False
        while True:
            waited = (self._clock_ns() - start) / 1e9
            if self.watch.quiet(options.quiet_cores, options.quiet_s):
                if told:
                    options.log("HOST_QUIET", what, "after", round(waited), "s")
                return waited
            recent = self.watch.recent(15.0)
            if not told and waited > 20 and recent and max(recent) >= options.quiet_cores:
                options.log("HOST_WAIT", what, "foreign cores", recent[-3:])
                told = True
            if waited > options.max_wait_s:
                raise HostBusy("HOST_BUSY never quiet before " + what)
            if self.handle.process.poll() is not None:
                raise RuntimeError(f"server exited with {self.handle.process.poll()} while waiting for a quiet host")
            self._sleep(1.0)

    def _settle(self) -> None:
        """After a cancelled phase: wait until the server dropped its requests, then clear every slot."""
        deadline = self._clock_ns() + 900 * 1_000_000_000
        while True:
            busy = self.host.slots_busy(self.handle)
            if not busy:
                break
            if self._clock_ns() > deadline:
                raise RuntimeError("server kept decoding a cancelled phase: " + json.dumps(busy))
            self._sleep(1.0)
        for slot in range(self.handle.parallel):
            reply = self.host.erase_slot(self.handle, slot, 60)
            if "error" in reply:
                self.options.log("SLOT_ERASE_ERROR", slot, reply["error"])

    # -- one request -----------------------------------------------------------------------------
    def _control(self, client, request_id: str, slot: int, columns: int) -> dict[str, object]:
        with self._mutex:
            self._generation += 1
            number = self._generation
        policy = AdaptiveDecodePolicy(
            route_id=request_id, executor_id="portable-run",
            operator_plan_sha256=canonical_sha256({"schedule": self.schedule["name"], "columns": columns}),
            desktop_parent_route_id="portable-run-parent",
            desktop_placement_sha256=canonical_sha256(list(self.handle.argv)),
            layer_indices=self.layers, layer_mask=sum(1 << layer for layer in self.layers), columns=columns,
            split_fraction_ppm=columns * 1_000_000 // self.plan.n_ff,
            resource_ids=("host-cpu", "host-cuda", str(self.plan.device_id)))
        sent = self._clock_ns()
        ack, received = client.apply_ffn_control(self.handle.endpoint, AdaptiveDecodeControl(request_id, slot, number, policy),
                                                 timeout_s=180)
        return {"sent_ns": sent, "received_ns": received, "generation": number,
                "applied_token_index": ack.get("applied_token_index"), "slot_id": ack.get("slot_id")}

    def _run_request(self, phase_name: str, index: int, request: Mapping[str, object], origin_ns: int, sink: list,
                     attempt: int) -> None:
        request_id = f"{self.schedule['name']}-{phase_name}-{index:02d}" + (f"-r{attempt}" if attempt else "")
        columns = int(request.get("columns", 0))
        tokens = tuple(int(token) for token in request["prompt_tokens"])
        state: dict[str, object] = {"first": None, "slot": None, "control": None, "progress": []}
        client = self._client_factory()

        def check() -> None:
            if self.phase_cancel.is_set():
                raise PhaseCancelled()

        def on_first(instant_ns: int) -> None:
            state["first"] = instant_ns

        def on_progress(slot: int, predicted: int, instant_ns: int, terminal: bool) -> None:
            state["progress"].append((predicted, instant_ns))
            if state["slot"] is None:
                state["slot"] = slot
                if columns:
                    state["control"] = self._control(client, request_id, slot, columns)

        target = origin_ns + int(float(request["arrival_s"]) * 1e9)
        try:
            while (now := self._clock_ns()) < target and not self.abort.is_set() and not self.phase_cancel.is_set():
                self._sleep(min(0.2, (target - now) / 1e9))
            if self.abort.is_set() or self.phase_cancel.is_set():
                return
            payload = LlamaCppCompletionPayload(
                request_id=request_id, expected_model_alias=self.plan.alias, input_tokens=len(tokens),
                output_tokens=int(request["output_tokens"]), prompt_tokens=tokens, seed=self.options.seed,
                stream_path=self.output / f"{request_id}.raw", on_first_token=on_first,
                on_decode_progress=on_progress, quality_mode="accounting-only",
                timeout_s=self.options.request_timeout_s)
            submitted = self._clock_ns()
            result = client.complete(self.handle.endpoint, payload, check)
            finished = self._clock_ns()
            if len(result["tokens"]) != request["output_tokens"] or state["first"] is None:
                raise RuntimeError("incomplete generated tokens")
            first = state["first"]
            row = {"request_id": request_id, "id": request.get("id", index), "phase": phase_name, "index": index,
                   "columns": columns, "arrival_s": float(request["arrival_s"]), "input_tokens": len(tokens),
                   "output_tokens": int(request["output_tokens"]), "target_ns": target, "submitted_ns": submitted,
                   "first_token_ns": first, "finished_ns": finished, "slot": state["slot"],
                   "control": state["control"], "prompt_ms": result.get("prompt_ms"),
                   "predicted_ms": result.get("predicted_ms"),
                   "progress": [[predicted, (instant - first) // 1000] for predicted, instant in state["progress"]],
                   "tokens_sha256": hashlib.sha256(json.dumps(result["tokens"]).encode()).hexdigest(),
                   "tokens": result["tokens"], "slot_probe_errors": result.get("endpoint_slot_probe_errors", [])}
            if self.plan.slot_erase and state["slot"] is not None:
                # a refusal or a timeout (the slot already serves the next request) is recorded, not a failure
                row["slot_erase"] = dict(self.host.erase_slot(self.handle, state["slot"]))
                row["slot_erase"]["ms"] = round((self._clock_ns() - finished) / 1e6, 1)
            with self._mutex:
                sink.append(row)
            self.options.log("REQUEST_DONE", request_id, columns, "ttft_s", round((first - target) / 1e9, 2),
                             "decode_s", round((finished - first) / 1e9, 2), "ms_token",
                             round((finished - first) / 1e6 / (request["output_tokens"] - 1), 1))
        except BaseException as error:  # noqa: BLE001 - one failed request ends the run
            if self.phase_cancel.is_set():
                return                  # the phase is being discarded (foreign load), not a failure
            with self._mutex:
                self.failures.append({"request_id": request_id, "error": repr(error)})
            self.abort.set()

    # -- one phase -------------------------------------------------------------------------------
    def _run_phase(self, name: str, requests: Sequence[Mapping[str, object]], attempt: int, window_s: float = 0.0):
        """(rows, origin, end, cancelled). The meters keep running until origin + window_s."""
        sink: list = []
        self.phase_cancel.clear()
        self.watch.rearm()
        origin = self._clock_ns() + 2_000_000_000
        threads = [threading.Thread(target=self._run_request, args=(name, index, request, origin, sink, attempt),
                                    daemon=True) for index, request in enumerate(requests)]
        for thread in threads:
            thread.start()
        process = self.handle.process
        while any(thread.is_alive() for thread in threads):
            if self.abort.is_set() and process.poll() is None:
                self.host.kill(self.handle)             # unblocks the other request threads
            if process.poll() is not None and not self.abort.is_set():
                self.abort.set()
                self.failures.append({"request_id": None, "error": f"server exited with {process.poll()}"})
            self._sleep(0.2)
        retry = self.options.phase_retry
        if not retry and self.watch.tripped is not None:
            raise self._host_busy()
        if self.failures:
            raise RuntimeError("request failed: " + json.dumps(self.failures)[:1500])
        close = origin + int(window_s * 1e9)
        while self._clock_ns() < close and not self.phase_cancel.is_set() and (retry or self.watch.tripped is None):
            self._sleep(0.2)
        if not retry and self.watch.tripped is not None:
            raise self._host_busy()
        end = max((row["finished_ns"] for row in sink), default=self._clock_ns())
        return sorted(sink, key=lambda row: row["index"]), origin, end, self.phase_cancel.is_set()

    def _host_busy(self) -> HostBusy:
        self._save(rundir.HOST_BUSY, {"tripped": self.watch.tripped, "samples": self.watch.samples[-40:]})
        return HostBusy("HOST_BUSY foreign load above the limit: " + json.dumps(self.watch.tripped["cores"]))

    def _phases(self) -> tuple[list[dict[str, object]], int]:
        options, schedule = self.options, self.schedule
        out: list[dict[str, object]] = []
        rewarm_prompt = schedule["phases"][0]["requests"][0]["prompt_tokens"]
        rewarms, last_origin = 0, 0
        for number, phase in enumerate(schedule["phases"]):
            window_s = options.min_window_s if number == len(schedule["phases"]) - 1 else 0.0
            attempt = 0
            while True:
                if options.phase_retry:
                    waited = self._wait_quiet(phase["name"])
                    if waited > 60 and options.rewarm_tokens > 1 and not phase["name"].startswith(UNMETERED_PREFIXES):
                        # after a long wait the GPU board has cooled: one unmetered host-only request first
                        rewarms += 1
                        _, _, _, cancelled = self._run_phase(f"rewarm{rewarms}", [{
                            "id": "rewarm", "arrival_s": 0.0, "prompt_tokens": rewarm_prompt,
                            "output_tokens": options.rewarm_tokens, "columns": 0}], 0)
                        options.log("REWARM", rewarms, "cancelled" if cancelled else "done")
                        if cancelled:
                            self._settle()
                            continue
                        if self._wait_quiet(phase["name"]) > 20:
                            continue
                rows, origin, end, cancelled = self._run_phase(phase["name"], phase["requests"], attempt, window_s)
                if not options.phase_retry:
                    break
                judged_end = max(end, origin + int(window_s * 1e9))
                while not cancelled and self.watch.samples and self.watch.samples[-1][0] < judged_end:
                    self._sleep(0.2)            # the sample that covers the last seconds of the phase
                seen = self.watch.disturbance(origin, judged_end + 5_000_000_000, options.quiet_cores)
                if not cancelled and seen["mean_excess_cores"] <= options.excess_cores:
                    break
                self._save(rundir.disturbed_phase_file(number, attempt), {
                    "name": phase["name"], "origin_ns": origin, "cancelled": cancelled, "foreign": seen,
                    "tripped": self.watch.tripped, "done_requests": len(rows)})
                options.log("PHASE_DISTURBED", phase["name"], "attempt", attempt,
                            "cancelled" if cancelled else "finished", seen)
                if cancelled:
                    self._settle()
                attempt += 1
                if attempt >= options.max_attempts:
                    raise HostBusy(f"HOST_BUSY phase {phase['name']} disturbed {attempt} times")
            last_origin = origin
            record = {"name": phase["name"], "origin_ns": origin, "end_ns": end, "service_s": (end - origin) / 1e9,
                      "service_host_energy": self._energy(origin, end),
                      "foreign_load": self.watch.summary(origin, end), "attempts": attempt + 1, "requests": rows}
            self._save(rundir.phase_file(number), record)
            out.append(record)
            options.log("PHASE_DONE", phase["name"], round(record["service_s"], 1))
            self._sleep(float(schedule.get("gap_s", 5)))
        return out, last_origin

    def _resolve_prompts(self) -> None:
        """Requests that carry text get their token ids from the loaded model (cached per text)."""
        cache: dict[str, list[int]] = {}
        for phase in self.schedule["phases"]:
            for request in phase["requests"]:
                text = request.pop("prompt_text", None)
                if text is None:
                    continue
                if text not in cache:
                    cache[text] = self.host.tokenize(self.handle, text)
                request["prompt_tokens"] = list(cache[text])

    # -- the run ---------------------------------------------------------------------------------
    def run(self) -> dict[str, object]:
        plan, options, output = self.plan, self.options, self.output
        quantum = self.site.phone(plan.device_id).column_quantum if plan.shared else 1
        validate(self.schedule, stored_columns=plan.binding.stored_columns if plan.shared else 0,
                 column_quantum=quantum)
        output.mkdir(parents=True)
        self._save(rundir.SCHEDULE, self.schedule)
        helper_started = False
        with self._lock(self.site.paths.work_root):
            try:
                helper_environment: Mapping[str, str] = {}
                if plan.shared:
                    if self.helper_meter is not None:
                        self.helper_meter.start(output, output.name)
                    helper_started = True
                    endpoint = self.phone.start(plan.binding, output)
                    receipts = getattr(self.phone, "receipts", {})
                    self._save(rundir.HELPER_PREFLIGHT, receipts.get("preflight"))
                    self._save(rundir.HELPER_LAUNCH, receipts.get("launch"))
                    options.log("HELPER_READY", receipts.get("launch_dropped"))
                    helper_environment = endpoint.server_environment
                self.handle = self.host.launch(plan.model_id, plan.alias, plan.port, plan.policy,
                                               binding=plan.binding, helper_environment=helper_environment,
                                               output=output)
                self._save(rundir.CONFIG, run_configuration(
                    self.site, plan, self.handle.argv, self.handle.environment,
                    self.site.phone(plan.device_id).transport if plan.shared else None))
                self._save(rundir.IDENTITY, self._identity())
                self.host.wait_ready(self.handle, plan.policy.load_timeout_s)
                options.log("SERVER_READY")
                self._resolve_prompts()
                self._save(rundir.SCHEDULE, self.schedule)
                self.watch = self.host.foreign_load(
                    lambda: (self.handle.pid, os.getpid()), limit_cores=options.foreign_limit_cores,
                    window_s=15.0 if options.phase_retry else 30.0,
                    on_trip=self.phase_cancel.set if options.phase_retry else self.abort.set)
                self.watch.start()
                self.meter.start()
                self.meter.wait_past(self._clock_ns(), 30.0)
                if options.phase_retry:
                    self._wait_quiet("idle")
                start, end = self._window(plan.idle_s)
                self._save(rundir.IDLE, {"started_ns": start, "finished_ns": end,
                                         "host_energy": self._energy(start, end)})
                phases, last_origin = self._phases()
                window = None
                if options.min_window_s > 0:
                    close = last_origin + int(options.min_window_s * 1e9)
                    window = {"origin_ns": last_origin, "end_ns": close, "window_s": options.min_window_s,
                              "host_energy": self._energy(last_origin, close)}
                if options.phase_retry:
                    self._wait_quiet("tail idle")
                self.watch.stop()
                self._save(rundir.FOREIGN_LOAD, {"limit_cores": options.foreign_limit_cores,
                                                 "samples": self.watch.samples})
                start, end = self._window(plan.tail_idle_s)
                self._save(rundir.TAIL_IDLE, {"started_ns": start, "finished_ns": end,
                                              "host_energy": self._energy(start, end)})
                summary = self.host.stop(self.handle)
                self._save(rundir.SERVER_FFN_SHAPES, summary["shape_lines"])
                result = {
                    "status": "PASS" if summary["exit_status"] == 0 else "CHECK",
                    "schedule": self.schedule["name"], "mode": "shared" if plan.shared else "alone",
                    "phone_calls": summary["phone_calls"], "server_exit": summary["exit_status"],
                    "phases": [{key: value for key, value in phase.items() if key != "requests"} | {"requests": [
                        {key: value for key, value in row.items() if key not in ("tokens", "progress")}
                        for row in phase["requests"]]} for phase in phases],
                    "fixed_window": window, "finished_epoch_s": time.time()}
                self._save(rundir.RESULT, result)
                if plan.shared:
                    try:    # the measurement is saved; a helper stop error is recorded, not raised
                        self._save(rundir.HELPER_STOP, self.phone.stop(served_calls=summary["phone_calls"]))
                    except Exception as error:  # noqa: BLE001
                        self._save(rundir.HELPER_STOP_ERROR, {"error": repr(error),
                                                              "cleanup": list(self.phone.release())})
                        options.log("HELPER_STOP_ERROR", repr(error)[:200])
                return result
            except BaseException as error:
                self.abort.set()
                cleanup = []
                if self.handle is not None and self.handle.process.poll() is None:
                    self.host.kill(self.handle)
                    cleanup.append("server killed")
                if helper_started and self.phone is not None:
                    try:        # never leave the phone with a worker running or away from adb
                        cleanup += list(self.phone.release()) or ["phone released"]
                    except Exception as release_error:  # noqa: BLE001
                        cleanup.append("release failed: " + repr(release_error))
                kind = error.kind if isinstance(error, ServerLoadFailure) else type(error).__name__
                self._save(rundir.FAILURE, {"error": repr(error), "kind": kind, "helper_cleanup": cleanup,
                                            "server_status": None if self.handle is None
                                            else self.handle.process.poll(), "finished_epoch_s": time.time()})
                raise
            finally:
                if self.watch is not None:
                    self.watch.stop()
                try:
                    self.meter.stop()
                finally:
                    self.meter.save(output)
                if helper_started and self.helper_meter is not None:
                    try:
                        self.helper_meter.collect(output)
                    except Exception as error:  # noqa: BLE001 - the run result does not depend on the phone trace
                        self._save("PHONE_SAMPLER_FAILURE.json", {"error": repr(error)})

    def _identity(self) -> dict[str, object]:
        server = self.site.paths.server_path
        model = self.site.paths.models[self.plan.model_id]
        return {"server_sha256": sha256_file(server) if server.exists() else None,
                "server_libraries": {str(path): sha256_file(path) for path in sorted(server.parent.glob("*.so"))},
                "model_stat": list(model.stat()) if model.exists() else None, "plan": self.plan.to_json()}


def run(site: SiteManifest, plan: RunPlan, schedule: Mapping[str, object], output: Path,
        options: RunOptions | None = None, **dependencies) -> dict[str, object]:
    return Engine(site, plan, schedule, output, options, **dependencies).run()
