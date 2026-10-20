#!/usr/bin/env python3
"""Bounded physical gates for remote-resident FFN weights (first milestone).

Two desktop parents of the same model are launched one after the other with identical
requests and decoding settings:

* ``full``    - the calibrated desktop control (every weight local);
* ``reduced`` - the same placement with the dense FFN weights of ``--remote-layer-mask``
  omitted from the desktop process and executed by the owning FFN worker session.

Gates (each recorded in REMOTE_RESIDENT_GATE.json with the raw proofs):

* A correctness: every request completes on both parents with valid output; generated tokens
  are compared token by token (identical prompts, seed, temperature 0).
* B memory: the reduced server's own omission record, the kernel VMA table of the model file
  (/proc/<pid>/smaps), process RSS, process VRAM and the loader buffer lines, versus the full
  parent. The accounting record credits reclaimed bytes only after the proof matches.
* C KV capacity: both parents are launched at each ``--capacity-context-size``; the record
  says which contexts each arm can serve. PASS only when the reduced parent serves a context the
  full parent cannot; otherwise RECORDED. This is a capacity comparison, not a matched savings
  comparison.
* D recovery: while the reduced parent decodes, the owner is stopped; the request must fail
  visibly, the reduced parent must refuse further FFN work, and after its teardown the full
  parent is relaunched and serves a request (the accounting record states the feasibility).

Owner modes: ``--owner tcp:HOST:PORT[:PIDFILE]`` uses an already running ``llama-ffn-split-worker``
(a desktop-hosted stand-in when the phone is absent: it proves omission and execution, not
memory leaving the host). ``--owner phone:ENDPOINT --preflight-only`` checks the existing
canonical phone configuration. Phone preparation/execution wiring is not implemented yet;
a successful phone preflight is not a successful relocation gate.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    DeviceMemoryCapacity,
    RuntimePlacementSnapshot,
)
from research_dev.scheduler.adapters import (
    DirectPhoneFfnSession,
    HostEnergySampler,
    LlamaCppHttpClient,
    LlamaServerLaunchContract,
    LlamaServerProcessConfiguration,
    LlamaServerProcessLauncher,
    PhysicalAdapterError,
    default_host_metric_callbacks,
    nvidia_gpu_snapshot,
    probe_nvidia_process_memory_bytes,
    server_energy_summary,
    load_transport_qualification_identity,
)
from research_dev.scheduler.adapters.ffn_shards import FfnShardIndex  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt import runner  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.desktop_parent_calibration import (
    _endpoint_is_free,
    _payload,
    _runtime_libraries_sha256,
    _sha256,
    _wait_for_sampler,
    _write_new,
    require,
)
from research_dev.scheduler._internal.runtime_plan import remote_resident_tensor_ids
from research_dev.scheduler._internal.runtime_resources import (  # noqa: E402
    RuntimeRemoteResidentOmissionProof,
    remote_resident_accounting,
)

SCHEMA = "s42-remote-resident-gate-v1"
PAGE = os.sysconf("SC_PAGE_SIZE")


class GateError(RuntimeError):
    pass


# ---- kernel-level memory records -----------------------------------------------------------


def _smaps_for_file(pid: int, path: Path) -> dict[str, int]:
    """Mapped and resident bytes of one file inside a process, from /proc/<pid>/smaps."""
    real = str(path.resolve())
    mapped = 0
    rss = 0
    vmas = 0
    in_file = False
    with open(f"/proc/{pid}/smaps", "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 5 and "-" in parts[0] and parts[1][0] in "r-":
                in_file = len(parts) >= 6 and " ".join(parts[5:]) == real
                if in_file:
                    start, end = (int(value, 16) for value in parts[0].split("-"))
                    mapped += end - start
                    vmas += 1
                continue
            if in_file and line.startswith("Rss:"):
                rss += int(parts[1]) * 1024
    return {"mapped_bytes": mapped, "rss_bytes": rss, "vma_count": vmas}


def _host_memory() -> dict[str, int]:
    values = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            name, _, rest = line.partition(":")
            if name in ("MemTotal", "MemAvailable", "MemFree"):
                values[name] = int(rest.split()[0]) * 1024
    return values


def _vm_rss(pid: int) -> int:
    with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def _omitted_ranges_absent(pid: int, path: Path, ranges: list[tuple[str, int, int]]) -> dict[str, object]:
    """Check that no VMA of the model file covers a page lying completely inside an omitted tensor."""
    real = str(path.resolve())
    vmas = []
    with open(f"/proc/{pid}/maps", "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 6 and " ".join(parts[5:]) == real:
                start, end = (int(value, 16) for value in parts[0].split("-"))
                offset = int(parts[2], 16)
                vmas.append((offset, offset + (end - start)))
    rows = []
    overlap_total = 0
    for name, offs, nbytes in ranges:
        first = (offs + PAGE - 1) & ~(PAGE - 1)
        last = (offs + nbytes) & ~(PAGE - 1)
        overlap = 0
        if last > first:
            for lo, hi in vmas:
                overlap += max(0, min(hi, last) - max(lo, first))
        overlap_total += overlap
        rows.append({"name": name, "offs": offs, "nbytes": nbytes, "page_first": first,
                     "page_last": max(first, last), "vma_overlap_bytes": overlap})
    return {"tensors": rows, "vma_overlap_total_bytes": overlap_total, "file_vma_count": len(vmas)}


def _tensor_ranges(model_path: Path, layer_mask: int) -> list[tuple[str, int, int]]:
    sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
    import gguf  # noqa: E402

    reader = gguf.GGUFReader(str(model_path))
    wanted = set(remote_resident_tensor_ids(layer_mask))
    rows = []
    for tensor in reader.tensors:
        if tensor.name in wanted:
            rows.append((tensor.name, int(tensor.data_offset), int(tensor.n_bytes)))
    require(len(rows) == len(wanted), "omitted tensors are absent from the GGUF")
    return sorted(rows, key=lambda row: row[1])


# ---- owner ---------------------------------------------------------------------------------


class Owner:
    """The FFN worker session that owns the omitted weights."""

    def __init__(self, spec: str) -> None:
        kind, _, rest = spec.partition(":")
        self.kind = kind
        self.pid_file: Path | None = None
        if kind == "tcp":
            host, _, tail = rest.partition(":")
            port_text, _, pid_file = tail.partition(":")
            self.host = host
            self.port = int(port_text)
            self.pid_file = Path(pid_file) if pid_file else None
            self.endpoint = f"tcp://{host}:{port_text}"
        elif kind == "phone":
            self.endpoint = rest
        else:
            raise GateError("owner must be tcp:HOST:PORT[:PIDFILE] or phone:ENDPOINT")

    def server_environment(self) -> dict[str, str]:
        if self.kind == "tcp":
            return {
                "S41_SERVER_FFN_HOST": self.host,
                "S41_SERVER_FFN_PORT": str(self.port),
                "S41_SERVER_FFN_TRANSPORT": "tcp",
            }
        raise GateError("phone owner transport must come from the catalog phone composite")

    def stop(self) -> dict[str, object]:
        if self.kind == "tcp" and self.pid_file is not None:
            pid = int(self.pid_file.read_text().strip())
            os.kill(pid, signal.SIGKILL)
            return {"kind": "tcp", "pid": pid, "signal": "SIGKILL", "at_ns": time.monotonic_ns()}
        raise GateError("owner stop requires a tcp pid file in this milestone")


def _phone_owner_preflight(args, models, manifests, selection, gpu):
    """Audit the exact deployed stack without loading weights or changing USB mode."""
    record = {
        "schema": "s42-remote-resident-phone-preflight-v1",
        "status": "BLOCKED",
        "started_epoch_ns": time.time_ns(),
        "command": list(sys.argv),
        "selection": selection.to_json(),
        "gpu": gpu,
        "gates": {},
        "physical_inference_started": False,
    }
    try:
        require(args.usb_qualification_identity is not None, "phone transport identity is required")
        identity = load_transport_qualification_identity(args.usb_qualification_identity)
        dependencies = dict(args.transport_host_dependency)
        actual = {"host_binary_sha256": _sha256(args.server)}
        actual.update({
            "host_dependency_sha256:" + name: _sha256(path)
            for name, path in dependencies.items()
        })
        record["transport_identity"] = identity.to_json()
        record["host_software_mismatches"] = {
            name: {"expected": identity.software_identity.get(name), "observed": value}
            for name, value in actual.items()
            if identity.software_identity.get(name) != value
        }
        configuration = runner._direct_phone_configuration(
            args, models, manifests, dependencies, identity
        )
        phone = DirectPhoneFfnSession(configuration)
        record["receipt"] = phone.preflight().to_json()
        record["status"] = "PREFLIGHT_PASS"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
    finally:
        record["finished_epoch_ns"] = time.time_ns()
        _write_new(args.output / "PHONE_OWNER_PREFLIGHT.json", record)
    return record


# ---- gate driver ---------------------------------------------------------------------------


def _launch_contract_for(selected, gpu_device_id: str, context_size: int | None,
                         ffn_environment: dict[str, str], phone_device_id: str | None) -> LlamaServerLaunchContract:
    parameters = selected.adapter_parameters
    return LlamaServerLaunchContract(
        model_alias=str(parameters["model_alias"]),
        context_size=int(parameters["context_size"]) if context_size is None else context_size,
        parallel=int(parameters["parallel"]),
        batch_size=int(parameters["batch_size"]),
        ubatch_size=int(parameters["ubatch_size"]),
        gpu_layers=selected.gpu_layers,
        cpu_device_id=str(parameters["cpu_device_id"]),
        gpu_device_id=gpu_device_id,
        phone_device_id=phone_device_id,
        ffn_environment=ffn_environment,
        cuda_graph_mode=parameters.get("cuda_graph_mode", "default"),
        desktop_launch_mode=parameters.get("desktop_launch_mode", "canonical"),
        threads=int(parameters.get("threads", 0)),
        threads_batch=int(parameters.get("threads_batch", 0)),
        cpu_affinity=(None if "cpu_affinity" not in parameters else str(parameters["cpu_affinity"])),
    )


def _reduced_environment(args, manifest, owner: Owner, remote_mask: int, ubatch_size: int,
                         layer_mask: int, session_id: str) -> dict[str, str]:
    activation = "geglu" if manifest.architecture == "gemma4" else "swiglu"
    environment = {
        "S41_SERVER_FFN_ACTIVATION": activation,
        "S41_SERVER_FFN_ARTIFACT_SHA256": manifest.artifact_sha256,
        "S41_SERVER_FFN_COLUMNS": str(manifest.feed_forward_length),
        "S41_SERVER_FFN_F16_IO": "0" if owner.kind == "tcp" else "1",
        "S41_SERVER_FFN_LAYER_MASK": str(layer_mask),
        "S41_SERVER_FFN_MAX_TOKENS": str(ubatch_size),
        "S41_SERVER_FFN_N_EMBD": str(manifest.embedding_length),
        "S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK": str(remote_mask),
        "S41_SERVER_FFN_RUNTIME_CONTROL": "1",
        "S41_SERVER_FFN_SHARDS": f"{session_id}@{owner.endpoint}:{remote_mask}",
        "S41_SERVER_FFN_TIMEOUT_MS": str(args.owner_timeout_ms),
    }
    environment.update(owner.server_environment())
    return environment


def _memory_record(process, model_path: Path, ranges) -> dict[str, object]:
    pid = process.pid
    proof = process.remote_resident_proof()
    record = {
        "pid": pid,
        "vm_rss_bytes": _vm_rss(pid),
        "host_memory": _host_memory(),
        "model_file": _smaps_for_file(pid, model_path),
        "process_vram_bytes": probe_nvidia_process_memory_bytes(pid),
        "gpu": nvidia_gpu_snapshot(),
        "omission_proof": None if proof is None else proof.to_json(),
        "loader_lines": [
            line for line in process.stderr_lines
            if "REMOTE_RESIDENT_FFN" in line or "model buffer size" in line
            or "unmap_remote_resident_weights" in line
        ],
    }
    if ranges:
        record["omitted_ranges"] = _omitted_ranges_absent(pid, model_path, ranges)
    return record


def _wait_past(sampler, end_ns: int, timeout_s: float = 15.0) -> None:
    """Block until both GPU and RAPL samples exist after ``end_ns`` (interpolation needs them)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rows = [
            row for row in sampler.rows()
            if type(row.get("gpu")) is dict and type(row.get("rapl_package")) is dict
        ]
        if rows and any(
            int(row["gpu"]["sample_t_ns"]) > end_ns and int(row["rapl_package"]["sample_t_ns"]) > end_ns
            for row in rows[-4:]
        ):
            return
        time.sleep(0.05)
    raise GateError("host sampler produced no GPU/RAPL sample after the measured interval")


def _energy(sampler, start_ns: int, end_ns: int) -> dict[str, object]:
    _wait_past(sampler, end_ns)
    rows = sampler.rows_between(start_ns, end_ns)
    return dict(server_energy_summary(rows, start_ns, end_ns))


def _run_requests(client, endpoint, rows, alias, output: Path, label: str, sampler) -> list[dict[str, object]]:
    results = []
    for index, row in enumerate(rows):
        request_id = f"{label}-{index:02d}-{row['request_index']}"
        payload = _payload(row, output, request_id=request_id, model_alias=alias)
        started_ns = time.monotonic_ns()
        error = None
        result = None
        try:
            result = client.complete(endpoint, payload, lambda: None)
        except Exception as failure:  # noqa: BLE001 - recorded, the gate decides
            error = f"{type(failure).__name__}: {failure}"
        finished_ns = time.monotonic_ns()
        results.append({
            "request_id": request_id,
            "request_index": row["request_index"],
            "input_tokens": int(row["input_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            "duration_us": (finished_ns - started_ns) // 1000,
            "energy": _energy(sampler, started_ns, finished_ns),
            "error": error,
            "tokens": None if result is None else list(result.get("tokens", [])),
            "result_keys": None if result is None else sorted(result.keys()),
        })
    return results


def _token_agreement(full_rows, reduced_rows) -> dict[str, object]:
    pairs = []
    for full, reduced in zip(full_rows, reduced_rows):
        a = full["tokens"] or []
        b = reduced["tokens"] or []
        common = min(len(a), len(b))
        first_divergence = next((i for i in range(common) if a[i] != b[i]), None)
        agree = sum(1 for i in range(common) if a[i] == b[i])
        pairs.append({
            "request_id": full["request_id"],
            "full_tokens": len(a),
            "reduced_tokens": len(b),
            "agreeing_tokens": agree,
            "first_divergence_index": first_divergence,
            "identical": a == b and bool(a),
        })
    return {"pairs": pairs, "identical_requests": sum(1 for row in pairs if row["identical"]),
            "requests": len(pairs)}


def run(args: argparse.Namespace) -> dict[str, object]:
    models = runner._load_trace_models(args)
    scheduler, manifests, _ = runner._build_scheduler(args, models, load_adaptive_observations=False)
    expected = {runner.GEMMA_ROLE: models.expected_gemma, runner.QWEN_ROLE: models.expected_qwen}[
        args.desktop_parent_role]
    manifest = manifests[expected.model_id]
    catalog = models.catalog
    control = catalog.desktop_control_by_artifact[manifest.artifact_sha256]
    source = catalog.composite_executor_by_id[control.executor_id]
    gpu_device_id = str(source.adapter_parameters["gpu_device_id"])
    gpu_resource_id = catalog.placement_profile.devices[gpu_device_id].memory_pool_id
    model_path = Path(models.model_paths[manifest.model_id])
    owner = Owner(args.owner)
    remote_mask = int(args.remote_layer_mask, 0)
    require(remote_mask > 0 and remote_mask >> manifest.block_count == 0, "remote layer mask is invalid")
    ranges = _tensor_ranges(model_path, remote_mask)
    omitted_bytes = sum(nbytes for _name, _offs, nbytes in ranges)

    shard_index = None
    if args.shard_index:
        require(bool(args.shard_remote_dir), "--shard-remote-dir is required with --shard-index")
        shard_index = FfnShardIndex.load(args.shard_index, args.shard_remote_dir)
    shard_record = None
    if shard_index is not None:
        shard_record = shard_index.resolve(
            manifest.artifact_sha256, remote_mask, manifest.feed_forward_length, session_id=args.session_id,
        )
        require(shard_record is not None, "shard index lacks a complete shard for the remote layers")
        require(shard_record.parent_sha256 == manifest.artifact_sha256, "shard parent differs from the model")

    gpu = nvidia_gpu_snapshot()
    reserve_bytes = min(512 * 1024**2, int(gpu["memory_free_bytes"]))
    memory = RuntimePlacementSnapshot(
        snapshot_id="remote-resident-gate-live-nvml", captured_at_us=0, valid_until_us=60_000_000,
        capacities={gpu_resource_id: DeviceMemoryCapacity(
            gpu_resource_id, int(gpu["memory_total_bytes"]),
            int(gpu["memory_total_bytes"]) - int(gpu["memory_free_bytes"]), reserve_bytes)},
    )
    selection = scheduler.select_live_vram_desktop_parent(
        manifest.model_id, memory, cuda_graph_mode=args.cuda_graph_mode,
        preserve_placement=True, maximum_gpu_layers=getattr(args, "maximum_gpu_layers", None),
        launch_overrides={},
    )
    selected = selection.selected
    if owner.kind == "phone":
        preflight = _phone_owner_preflight(args, models, manifests, selection, gpu)
        if args.preflight_only or preflight["status"] != "PREFLIGHT_PASS":
            return preflight
    elif args.preflight_only:
        raise GateError("--preflight-only requires a phone owner")
    endpoint = source.endpoint
    require(_endpoint_is_free(endpoint), "gate endpoint is already active")
    ubatch_size = int(selected.adapter_parameters["ubatch_size"])
    layer_mask = remote_mask if args.resident_layer_mask is None else int(args.resident_layer_mask, 0)
    require(remote_mask & ~layer_mask == 0, "remote layers exceed the resident layer mask")

    full_contract = _launch_contract_for(selected, gpu_device_id, None, {}, None)
    reduced_contract = _launch_contract_for(
        selected, gpu_device_id, None,
        _reduced_environment(args, manifest, owner, remote_mask, ubatch_size, layer_mask, args.session_id),
        None,
    )
    launcher = LlamaServerProcessLauncher(LlamaServerProcessConfiguration(
        server_path=args.server,
        model_paths_by_artifact={manifest.artifact_sha256: model_path},
        library_paths_by_device={gpu_device_id: (args.cuda_lib_dir,)},
        executable_device_names={gpu_device_id: "CUDA0"},
        output_directory=args.output,
        common_library_paths=(args.server.parent, args.cuda_lib_dir),
    ))
    client = LlamaCppHttpClient()
    _, selected_requests, _, _ = runner._select_replay(args, models, manifests)
    candidate_rows = [
        item["row"] for item in selected_requests
        if item["source"] == "large" and runner.trace_role(item["row"]) == args.desktop_parent_role
        and int(item["row"]["input_tokens"]) + int(item["row"]["output_tokens"]) <= full_contract.context_size
        and int(item["row"]["output_tokens"]) <= args.maximum_output_tokens
    ]
    require(bool(candidate_rows), "no short gate requests in the trace")
    rows = sorted(candidate_rows, key=lambda row: (int(row["output_tokens"]), int(row["input_tokens"])))[:args.requests]

    sampler = HostEnergySampler(default_host_metric_callbacks(), interval_s=0.1)
    record: dict[str, object] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "trace_role": args.desktop_parent_role,
        "artifact_sha256": manifest.artifact_sha256,
        "artifact_bytes": manifest.artifact_bytes,
        "model_path": str(model_path),
        "runtime_binary_sha256": _sha256(args.server),
        "runtime_libraries_sha256": _runtime_libraries_sha256(args.server),
        "owner": {"spec": args.owner, "kind": owner.kind, "endpoint": owner.endpoint},
        "remote_layer_mask": remote_mask,
        "resident_layer_mask": layer_mask,
        "omitted_bytes_planned": omitted_bytes,
        "omitted_tensor_ids": [name for name, _offs, _nbytes in ranges],
        "shard": None if shard_record is None else {
            "session_hint": shard_record.session_hint, "remote_path": shard_record.remote_path,
            "shard_sha256": shard_record.shard_sha256, "shard_bytes": shard_record.shard_bytes,
            "columns": shard_record.columns, "layer_mask": shard_record.layer_mask,
        },
        "selection": selection.to_json(),
        "launch_contracts": {
            "full": {"context_size": full_contract.context_size, "gpu_layers": full_contract.gpu_layers,
                     "ubatch_size": full_contract.ubatch_size, "ffn_environment": {}},
            "reduced": {"context_size": reduced_contract.context_size, "gpu_layers": reduced_contract.gpu_layers,
                        "ubatch_size": reduced_contract.ubatch_size,
                        "ffn_environment": dict(reduced_contract.ffn_environment)},
        },
        "requests": [{"request_index": row["request_index"], "input_tokens": int(row["input_tokens"]),
                      "output_tokens": int(row["output_tokens"])} for row in rows],
        "gates": {},
    }
    process = None
    sampler.start()
    try:
        _wait_for_sampler(sampler, 2)
        arms = {}
        for arm, contract in (("full", full_contract), ("reduced", reduced_contract)):
            baseline_gpu = nvidia_gpu_snapshot()
            load_start_ns = time.monotonic_ns()
            process = launcher.launch_contract(endpoint, contract, manifest, label=f"gate-{arm}",
                                               control_check=lambda: None)
            load_end_ns = time.monotonic_ns()
            results = _run_requests(client, endpoint, rows, contract.model_alias, args.output, arm, sampler)
            memory_record = _memory_record(process, model_path, ranges if arm == "reduced" else [])
            arms[arm] = {
                "load_duration_us": (load_end_ns - load_start_ns) // 1000,
                "load_energy": _energy(sampler, load_start_ns, load_end_ns),
                "baseline_gpu": baseline_gpu,
                "requests": results,
                "memory": memory_record,
                "stderr_path": str(args.output / f"gate-{arm}.stderr"),
            }
            process.stop()
            process = None
            time.sleep(1.0)
        record["arms"] = arms
        # ---- gate A ----
        full_ok = all(row["error"] is None for row in arms["full"]["requests"])
        reduced_ok = all(row["error"] is None for row in arms["reduced"]["requests"])
        agreement = _token_agreement(arms["full"]["requests"], arms["reduced"]["requests"])
        record["gates"]["A_correctness"] = {
            "full_requests_ok": full_ok, "reduced_requests_ok": reduced_ok,
            "token_agreement": agreement,
            "status": "PASS" if full_ok and reduced_ok else "FAIL",
        }
        # ---- gate B ----
        proof_json = arms["reduced"]["memory"]["omission_proof"]
        proof = None if proof_json is None else RuntimeRemoteResidentOmissionProof.from_json(proof_json)
        overlap = arms["reduced"]["memory"]["omitted_ranges"]["vma_overlap_total_bytes"]
        host_before = arms["full"]["memory"]["model_file"]
        host_after = arms["reduced"]["memory"]["model_file"]
        gate_b = {
            "proof": proof_json,
            "proof_matches_plan": proof is not None and proof.layer_mask == remote_mask
                and proof.omitted_bytes == omitted_bytes and proof.warmup == "validated",
            "omitted_pages_vma_overlap_bytes": overlap,
            "model_file_mapped_bytes": {"full": host_before["mapped_bytes"], "reduced": host_after["mapped_bytes"]},
            "model_file_rss_bytes": {"full": host_before["rss_bytes"], "reduced": host_after["rss_bytes"]},
            "vm_rss_bytes": {"full": arms["full"]["memory"]["vm_rss_bytes"],
                             "reduced": arms["reduced"]["memory"]["vm_rss_bytes"]},
            "process_vram_bytes": {"full": arms["full"]["memory"]["process_vram_bytes"],
                                   "reduced": arms["reduced"]["memory"]["process_vram_bytes"]},
        }
        gate_b["status"] = "PASS" if gate_b["proof_matches_plan"] and overlap == 0 else "FAIL"
        record["gates"]["B_memory"] = gate_b
        # ---- gate C ----
        if args.capacity_context_sizes:
            record["gates"]["C_kv_capacity"] = _capacity_gate(
                args, launcher, endpoint, manifest, selected, gpu_device_id, reduced_contract.ffn_environment)
        # ---- gate D (last: it stops the owner) ----
        if args.recovery:
            process = launcher.launch_contract(endpoint, reduced_contract, manifest, label="gate-recovery",
                                               control_check=lambda: None)
            recovery = _recovery_gate(args, client, endpoint, rows, reduced_contract.model_alias,
                                      process, owner, sampler)
            teardown_started_ns = time.monotonic_ns()
            try:
                process.stop()
            finally:
                process = None
            recovery["reduced_teardown_us"] = (time.monotonic_ns() - teardown_started_ns) // 1000
            time.sleep(1.0)
            recovery["fallback"] = _fallback_gate(args, launcher, endpoint, manifest, full_contract, client, rows, sampler)
            if recovery["fallback"]["status"] != "PASS":
                recovery["status"] = "FAIL"
            record["gates"]["D_recovery"] = recovery
        # ---- accounting ----
        record["accounting"] = _accounting(record, manifest, proof, omitted_bytes, remote_mask, args, gpu_resource_id)
        statuses = [gate["status"] for gate in record["gates"].values() if "status" in gate]
        record["status"] = (
            "PASS" if statuses and all(value in ("PASS", "RECORDED") for value in statuses) else "PARTIAL"
        )
    except Exception as error:  # noqa: BLE001 - recorded, then re-raised
        record["status"] = "FAILED"
        record["failure"] = {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        raise
    finally:
        if process is not None:
            try:
                process.stop()
            except Exception:  # noqa: BLE001
                pass
        sampler.stop()
        _write_new(args.output / "REMOTE_RESIDENT_GATE.json", record)
    return record


def _recovery_gate(args, client, endpoint, rows, alias, process, owner: Owner, sampler) -> dict[str, object]:
    """Stop the owner mid-decode: the request must fail visibly and the server must refuse."""
    import threading

    row = max(rows, key=lambda item: int(item["output_tokens"]))
    payload = _payload(row, args.output, request_id="recovery-victim", model_alias=alias)
    outcome: dict[str, object] = {}

    def victim():
        started = time.monotonic_ns()
        try:
            result = client.complete(endpoint, payload, lambda: None)
            outcome["result"] = {"tokens": len(result.get("tokens", []))}
        except Exception as failure:  # noqa: BLE001
            outcome["error"] = f"{type(failure).__name__}: {failure}"
        outcome["duration_us"] = (time.monotonic_ns() - started) // 1000

    thread = threading.Thread(target=victim, daemon=True)
    thread.start()
    time.sleep(args.recovery_kill_delay_s)
    stop_record = owner.stop()
    thread.join(timeout=120)
    time.sleep(0.5)
    alive = process.process is not None and process.process.poll() is None
    refused = None
    if alive:
        probe_payload = _payload(rows[0], args.output, request_id="recovery-after-owner-loss", model_alias=alias)
        try:
            client.complete(endpoint, probe_payload, lambda: None)
            refused = False
        except Exception as failure:  # noqa: BLE001
            refused = f"{type(failure).__name__}: {failure}"
    return {
        "owner_stop": stop_record,
        "victim": outcome,
        "victim_failed_visibly": "error" in outcome,
        "server_alive_after_owner_loss": alive,
        "server_refuses_after_owner_loss": refused,
        "status": "PASS" if "error" in outcome and (not alive or refused) else "FAIL",
    }


def _fallback_gate(args, launcher, endpoint, manifest, full_contract, client, rows, sampler) -> dict[str, object]:
    """Relaunch the full-weight parent after the reduced parent is torn down and serve one request."""
    started_ns = time.monotonic_ns()
    process = launcher.launch_contract(endpoint, full_contract, manifest, label="gate-fallback",
                                       control_check=lambda: None)
    try:
        ready_ns = time.monotonic_ns()
        results = _run_requests(client, endpoint, rows[:1], full_contract.model_alias, args.output, "fallback", sampler)
        return {
            "relaunch_duration_us": (ready_ns - started_ns) // 1000,
            "request": results[0],
            "status": "PASS" if results[0]["error"] is None else "FAIL",
        }
    finally:
        process.stop()


def _capacity_gate(args, launcher, endpoint, manifest, selected, gpu_device_id, reduced_environment) -> dict[str, object]:
    results = []
    for context_size in args.capacity_context_sizes:
        row = {"context_size": context_size}
        for arm, environment in (("full", {}), ("reduced", dict(reduced_environment))):
            contract = _launch_contract_for(selected, gpu_device_id, context_size, environment, None)
            started = time.monotonic_ns()
            try:
                process = launcher.launch_contract(endpoint, contract, manifest, label=f"capacity-{arm}-{context_size}",
                                                   control_check=lambda: None)
                try:
                    row[arm] = {"launched": True, "process_vram_bytes": probe_nvidia_process_memory_bytes(process.pid),
                                "vm_rss_bytes": _vm_rss(process.pid), "duration_us": (time.monotonic_ns() - started) // 1000}
                finally:
                    process.stop()
            except PhysicalAdapterError as error:
                row[arm] = {"launched": False, "error": str(error), "duration_us": (time.monotonic_ns() - started) // 1000}
            time.sleep(1.0)
        results.append(row)
    differentiating = [
        row["context_size"] for row in results
        if row["reduced"].get("launched") and not row["full"].get("launched")
    ]
    return {
        "rows": results,
        "differentiating_context_sizes": differentiating,
        "status": "PASS" if differentiating else ("RECORDED" if results else "SKIPPED"),
        "note": ("capacity comparison, not a matched savings comparison; PASS only when the reduced parent "
                 "serves a context the full parent cannot; otherwise the pool that bounds the context is "
                 "reported and the freed memory is in another pool"),
    }


def _accounting(record, manifest, proof, omitted_bytes, remote_mask, args, gpu_resource_id):
    full_memory = record["arms"]["full"]["memory"]
    reduced_memory = record["arms"]["reduced"]["memory"]
    # live figures observed while the reduced parent was running
    live = {
        "host-ram": int(reduced_memory["host_memory"].get("MemAvailable", 0)),
        gpu_resource_id: int(reduced_memory["gpu"]["memory_free_bytes"]),
    }
    reduced_allocation = {"host-ram": reduced_memory["vm_rss_bytes"], gpu_resource_id: reduced_memory["process_vram_bytes"] or 0}
    required = {"host-ram": full_memory["vm_rss_bytes"], gpu_resource_id: full_memory["process_vram_bytes"] or 0}
    account = remote_resident_accounting(
        artifact_sha256=manifest.artifact_sha256,
        desktop_pool_id="host-ram",
        layer_mask=remote_mask,
        desktop_weights_full_bytes=manifest.artifact_bytes,
        omitted_bytes_planned=omitted_bytes,
        proof=proof,
        phone_weights_bytes_by_session={args.session_id: omitted_bytes},
        phone_workspace_bytes=0,
        kv_bytes_by_pool={},
        transition_peak_bytes_by_pool={},
        live_available_bytes_by_pool=live,
        reduced_allocation_bytes_by_pool=reduced_allocation,
        recovery_required_bytes_by_pool=required,
        fallback_mode=args.fallback_mode,
    )
    result = account.to_json()
    result["note"] = ("recovery figures use the measured full-parent RSS/VRAM as the fallback demand; "
                      "a desktop-hosted tcp owner does not move memory off the host")
    return result


def parse_args() -> argparse.Namespace:
    parser = runner._build_parser()
    parser.add_argument("--desktop-parent-role", required=True, choices=(runner.GEMMA_ROLE, runner.QWEN_ROLE))
    parser.add_argument("--desktop-baseline-plans", type=Path, required=True)
    parser.add_argument("--owner", required=True, help="tcp:HOST:PORT[:PIDFILE] or phone:ENDPOINT")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--remote-layer-mask", required=True, help="layers whose FFN weights live on the owner")
    parser.add_argument("--resident-layer-mask", default=None, help="owner's full resident mask (default: remote mask)")
    parser.add_argument("--session-id", default="HTP0")
    parser.add_argument("--shard-index", type=Path, default=None, help="FFN_SHARDS.json of the owner's shards")
    parser.add_argument("--shard-remote-dir", default=None, help="device directory holding the shard files")
    parser.add_argument("--owner-timeout-ms", type=int, default=20000)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--maximum-output-tokens", type=int, default=64)
    parser.add_argument("--capacity-context-sizes", type=lambda text: [int(v) for v in text.split(",") if v], default=[])
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--recovery-kill-delay-s", type=float, default=2.0)
    parser.add_argument("--fallback-mode", choices=("teardown", "alongside"), default="teardown")
    parser.add_argument("--cuda-graph-mode", choices=("default", "disabled"), default="default")
    parser.add_argument("--maximum-gpu-layers", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), "output directory must be new")
    args.output.mkdir(parents=True)
    try:
        result = run(args)
    except Exception as error:
        _write_new(args.output / "FAILURE.json", {
            "schema": SCHEMA, "status": "FAIL",
            "command": list(sys.argv), "observed_epoch_ns": time.time_ns(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })
        raise
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "gates": {name: gate.get("status") for name, gate in result["gates"].items()}},
                     sort_keys=True))
    return 0 if result["status"] in {"PASS", "PREFLIGHT_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
