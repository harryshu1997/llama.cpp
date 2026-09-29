"""Decode-only FFN relocation with per-layer host KV: a bounded physical gate (not route qualification).

Prefill stays on the desktop with every weight resident. At the first generated token the runtime
control applies the phone split (host prefix columns on the desktop, suffix on the phone sessions) and
the server releases the page tables and page cache of the phone-owned share (dormant host share). Host KV
buffers are zeroed page-granularly, so released weight pages can back KV growth during decode. Arms:
``control`` (desktop only), ``combined`` (phone split + dormant release), ``sweep`` (ratio scan).
Never kill an in-flight phone worker or touch unrelated host processes.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[6]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_dev.scheduler import ModelManifest  # noqa: E402
from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeControl, AdaptiveDecodePolicy  # noqa: E402
from research_dev.scheduler._internal.capacity import DeviceMemoryCapacity  # noqa: E402
from research_dev.scheduler._internal.decode_split_selection import (  # noqa: E402
    DecodeReleaseAccountant, DecodeSplitAtlas, DecodeSplitEnvironment, ShareBinding, select_decode_split,
)
from research_dev.scheduler._internal.runtime_cost import RuntimeMemoryDemand  # noqa: E402
from research_dev.scheduler._internal.runtime_placement import RuntimePlacementSnapshot  # noqa: E402
from research_dev.scheduler._internal.runtime_resources import RuntimeMemoryLedger  # noqa: E402
from research_dev.scheduler._internal.kv_placement import plan_layer_kv  # noqa: E402
from research_dev.scheduler._internal.runtime_plan import RuntimeExecutionContract, RuntimePhoneShard, RuntimeTransitionPlan  # noqa: E402
from research_dev.scheduler._internal.runtime_resources import RuntimeHostShareReleaseProof, RuntimeResourceError, host_share_release_lower_bound_bytes  # noqa: E402
from research_dev.scheduler._internal.types import canonical_sha256  # noqa: E402
from research_dev.scheduler.adapters import (  # noqa: E402
    HostEnergySampler, LlamaCppHttpClient, LlamaServerLaunchContract,
    LlamaServerProcessConfiguration, LlamaServerProcessLauncher, default_host_metric_callbacks,
)
from research_dev.scheduler.adapters.http_backend import LlamaCppCompletionPayload  # noqa: E402
from research_dev.scheduler.adapters.llama_server_contracts import LlamaServerPhoneSessionProof, parse_llama_server_ffn_call  # noqa: E402
from research_dev.scheduler.adapters.phone_session import DirectPhoneFfnSession, DirectPhoneFfnSessionConfiguration  # noqa: E402
from research_dev.scheduler.adapters.phone_transport import PhoneTransportContract  # noqa: E402
from research_dev.scheduler.adapters.ticket import PhysicalParticipantCommand, PhysicalTransitionCommand  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.remote_resident_gate import _energy, _memory_record  # noqa: E402

PAGE = 4096
PHONE_IDLE_W = 0.875


def save(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=1, sort_keys=True)
        stream.write("\n")


def sha(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def launch_contract(config, plan, manifest, with_phone, environment):
    return LlamaServerLaunchContract(
        manifest.model_id, plan.context_size, int(config.get("parallel", 1)), config["batch"], config["ubatch"],
        config["gpu_layers"], "cpu", "gpu", "op15-phone" if with_phone else None, environment,
        threads=config.get("threads", 0), threads_batch=config.get("threads_batch", 0),
        cpu_affinity=config.get("cpu_affinity"), kv_cpu_layers=plan.cpu_layers, cuda_graph_mode="default",
        scheduler_trace_path=config.get("scheduler_trace_path"),
        logits_trace_path=config.get("logits_trace_path"),
        ffn_row_diagnostic_steps=config.get("ffn_row_diagnostic_steps", 0) if with_phone else 0,
        ffn_host_share_drop_cache=config.get("ffn_host_share_drop_cache", 1) if with_phone else 1,
        ffn_host_share_populate=config.get("ffn_host_share_populate", 1) if with_phone else 1)


def record_server_identity(managed, port, output, runtime):
    command = Path(f"/proc/{managed.pid}/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    expected = list(managed.command)
    if expected[:2] == ["taskset", "--cpu-list"]:
        expected = expected[3:]
    listeners = subprocess.check_output(["ss", "-ltnp", f"sport = :{port}"], text=True)
    if command != expected or f"pid={managed.pid}," not in listeners:
        raise RuntimeError("answering server process differs from the launch contract")
    contract = {field.name: getattr(managed.launch_contract, field.name) for field in fields(managed.launch_contract)}
    contract["ffn_environment"] = dict(contract["ffn_environment"])
    identity = {"runtime": runtime, "launch_contract": contract}
    save(output / "SERVER_IDENTITY.json", {"pid": managed.pid, "command": command, "listeners": listeners,
        **identity, "runtime_launch_sha256": canonical_sha256(identity)})
    return canonical_sha256(identity)


def inward(first, last):
    first = (first + PAGE - 1) // PAGE * PAGE
    last = last // PAGE * PAGE
    return (first, last) if last > first else None


def share_ranges(gguf_dir, model_path, layer_indices, host_columns, n_embd, n_ff):
    """Page-inward file ranges of the phone share, mirroring llama_model::ffn_host_share_release."""
    sys.path.insert(0, str(gguf_dir))
    import gguf  # noqa: PLC0415

    reader = gguf.GGUFReader(str(model_path))
    ranges = []
    for tensor in reader.tensors:
        name = tensor.name
        if not name.startswith("blk.") or int(name.split(".")[1]) not in layer_indices:
            continue
        element = {"F16": 2, "F32": 4, "BF16": 2}.get(tensor.tensor_type.name)
        if element is None:
            continue
        offs = int(tensor.data_offset)
        if name.endswith((".ffn_gate.weight", ".ffn_up.weight")):
            assert int(tensor.shape[0]) == n_embd and int(tensor.shape[1]) == n_ff, (name, tensor.shape)
            piece = inward(offs + host_columns * n_embd * element, offs + n_ff * n_embd * element)
            if piece:
                ranges.append(piece)
        elif name.endswith(".ffn_down.weight"):
            assert int(tensor.shape[0]) == n_ff and int(tensor.shape[1]) == n_embd, (name, tensor.shape)
            row = n_ff * element
            for r in range(n_embd):
                piece = inward(offs + r * row + host_columns * element, offs + (r + 1) * row)
                if piece:
                    ranges.append(piece)
    return ranges


def model_mappings(pid, real_model):
    rows = []
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].strip() == real_model:
            start, end = (int(v, 16) for v in parts[0].split("-"))
            rows.append((start, end, int(parts[2], 16)))
    return rows


def residency(pid, real_model, ranges):
    """Resident and total pages of the file ranges, from /proc/<pid>/pagemap bit 63."""
    mappings = model_mappings(pid, real_model)
    resident = total = 0
    with open(f"/proc/{pid}/pagemap", "rb", buffering=0) as pagemap:
        for first, last in ranges:
            for vstart, vend, offset in mappings:
                lo, hi = max(first, offset), min(last, offset + (vend - vstart))
                if lo >= hi:
                    continue
                vaddr = vstart + (lo - offset)
                pages = (hi - lo) // PAGE
                pagemap.seek((vaddr // PAGE) * 8)
                data = pagemap.read(pages * 8)
                resident += sum(1 for i in range(7, len(data), 8) if data[i] & 0x80)
                total += len(data) // 8
    return {"resident_pages": resident, "total_pages": total, "vma_count": len(mappings),
            "resident_fraction": (resident / total) if total else None}


def model_file_rss(pid, real_model):
    total = 0
    in_model = False
    for line in Path(f"/proc/{pid}/smaps").read_text().splitlines():
        head = line.split(" ", 1)[0]
        if "-" in head and len(line.split()) >= 5:
            parts = line.split(maxsplit=5)
            in_model = len(parts) == 6 and parts[5].strip() == real_model
            continue
        if in_model and line.startswith("Rss:"):
            total += int(line.split()[1]) * 1024
    return total


def process_status(pid):
    values = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmHWM", "VmSwap"):
            values[key] = int(rest.split()[0]) * 1024
    return values


def meminfo():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable", "MemFree", "Cached", "Mapped", "SwapFree"):
            values[key] = int(rest.split()[0]) * 1024
    return values


def cgroup_memory():
    """memory.current/max/events and the file/anon split of this process's cgroup (v2)."""
    try:
        entry = Path("/proc/self/cgroup").read_text().strip().splitlines()[0]
        relative = entry.split(":", 2)[2]
        root = Path("/sys/fs/cgroup") / relative.lstrip("/")
        values = {"path": relative}
        for name in ("memory.current", "memory.max", "memory.swap.max", "memory.peak"):
            path = root / name
            if path.exists():
                text = path.read_text().strip()
                values[name] = text if text == "max" else int(text)
        events = root / "memory.events"
        if events.exists():
            values["memory.events"] = {k: int(v) for k, v in (line.split() for line in events.read_text().splitlines() if line.strip())}
        stat = root / "memory.stat"
        if stat.exists():
            wanted = {"anon", "file", "file_mapped", "active_file", "inactive_file", "pgmajfault", "pgscan", "pgsteal"}
            values["memory.stat"] = {k: int(v) for k, v in (line.split() for line in stat.read_text().splitlines()) if k in wanted}
        return values
    except (OSError, ValueError, IndexError) as error:
        return {"error": repr(error)}


def kv_bytes_per_token(manifest, layers):
    return 2 * manifest.head_count_kv * manifest.key_length * 2 * len(layers)


def selection_environment(config, plan, runtime):
    """The exact execution environment a measured atlas row must match (mirrors build_decode_split_atlas.py)."""
    plan_json = plan.to_json()
    defaults = {"threads": 0, "threads_batch": 0, "cpu_affinity": None,
                "ffn_host_share_drop_cache": 1, "ffn_host_share_populate": 1,
                "ffn_max_tokens": config["ubatch"], "scheduler_trace_path": None, "logits_trace_path": None,
                "usb_batch_plan": "split-row", "ffn_row_diagnostic_steps": 0, "cohort_submission_order": []}
    overrides = {name: config[name] for name, value in defaults.items() if name in config and config[name] != value}
    runtime_identity = {**runtime, "launch_overrides": overrides} if overrides else runtime
    return DecodeSplitEnvironment(
        artifact_sha256=plan.artifact_sha256, gpu_layers=config["gpu_layers"], context_cells=plan.context_size,
        parallel=int(config.get("parallel", 1)), batch=config["batch"], ubatch=config["ubatch"], kv_plan_sha256=canonical_sha256(plan_json),
        column_quantum=int(config["column_quantum"]),
        session_masks=tuple(sorted((name, int(mask)) for name, mask in config["phone"]["session_masks"].items())),
        runtime_sha256=canonical_sha256(runtime_identity))


def cgroup_memory_max():
    """MemoryMax of this process's cgroup (bytes) or None when unlimited."""
    value = cgroup_memory().get("memory.max")
    return value if isinstance(value, int) else None


def wait_for_release_proof(stderr_path, seen, timeout_s):
    """Block until a new phase=decode dormant proof line appears in the server log; returns (proof, generation)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        proofs = [p for p in (RuntimeHostShareReleaseProof.parse_line(line) for line in Path(stderr_path).read_text(errors="replace").splitlines())
                  if p is not None and p.phase == "decode"]
        if len(proofs) > seen:
            return proofs[seen], seen + 1
        time.sleep(0.5)
    raise RuntimeError("no decode release proof appeared in the server log")


def spawn_consumer(nbytes, log_path):
    """Occupy nbytes of touched anonymous memory in this cgroup until terminated (a stand-in tenant)."""
    import subprocess  # noqa: PLC0415
    code = ("import mmap, signal, sys, time\n"
            f"m = mmap.mmap(-1, {nbytes})\n"
            f"for off in range(0, {nbytes}, 4096): m[off] = 1\n"
            "print('CONSUMER_READY', flush=True)\n"
            "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
            "while True: time.sleep(1)\n")
    log = open(log_path, "w")
    process = subprocess.Popen([sys.executable, "-c", code], stdout=log, stderr=subprocess.STDOUT)
    return process


def phone_owner(config, manifest, output, plan, fractions_ppm, dormant):
    """Three resident phone sessions own the complete FFN weights of their layers; the desktop keeps
    its own copy resident for prefill and releases the phone share only while decoding."""
    phone = config["phone"]
    artifact = manifest.artifact_sha256
    n_ff = manifest.feed_forward_length
    max_tokens = config.get("ffn_max_tokens", config["ubatch"])
    parallel = config.get("parallel", 1)
    if type(parallel) is not int or parallel not in (1, 2, 4, 8):
        raise ValueError("decode gate parallel must be 1, 2, 4 or 8")
    if type(max_tokens) is not int or not parallel <= max_tokens <= min(config["ubatch"], 512):
        raise ValueError("phone maximum tokens must cover every decode slot and fit the native limit")
    masks = {name: int(mask) for name, mask in phone["session_masks"].items()}
    mask = 0
    shards = []
    for name, value in masks.items():
        if value <= 0 or value & mask or value >> manifest.block_count:
            raise ValueError("invalid or overlapping phone layer masks")
        mask |= value
        weight_bytes = sum(manifest.tensor_by_id[f"blk.{il}.ffn_{kind}.weight"].nbytes
            for il in range(manifest.block_count) if value >> il & 1 for kind in ("gate", "up", "down"))
        shards.append(RuntimePhoneShard(name, f"session://op15-phone/{name}", value, n_ff, weight_bytes,
            canonical_sha256({"artifact": artifact, "mask": value, "columns": n_ff}), plan.plan_sha256, artifact, 1))
    if any(il >= max(0, manifest.block_count + 1 - config["gpu_layers"])
           for il in range(manifest.block_count) if mask >> il & 1):
        raise ValueError("phone FFNs must belong to the unchanged CPU parent")
    usb = PhoneTransportContract("functionfs-usb", "devmem", 4, 4, manifest.embedding_length * 2 * max_tokens,
        True, False, "functionfs-dmabuf-async-ring-v2", "decode-relocation-kv-gate",
        int(Path("/sys/module/usbcore/parameters/usbfs_memory_mb").read_text()) * 1024**2,
        65536, 0x18d1, 0x2d00, "127.0.0.1", 0,
        batch_plan=config.get("usb_batch_plan", "split-row"))
    if parallel > 1 and usb.batch_plan != "coalesced-batch":
        raise ValueError("multi-slot decode gate requires one coalesced phone call")
    cfg = DirectPhoneFfnSessionConfiguration(
        adb_path=Path("/usr/bin/adb"), usb_close_path=Path(phone["usb_close"]), serial=phone["serial"], adb_port=5037,
        session_script=phone["session_script"], restore_script=phone["restore_script"],
        session_root="/data/local/tmp/" + output.parent.name + "-" + output.name,
        worker_paths_by_artifact={artifact: phone["worker"]}, model_paths_by_artifact={artifact: phone["model"]},
        backend_by_device={"op15-phone": "HTP0"}, minimum_usb_speed_mbps=5000,
        required_kernel_release=phone["kernel_release"], launch_timeout_s=180, session_timeout_s=14400,
        diagnostic_port=18383, diagnostic_host="192.168.42.1", busybox_path="/data/adb/magisk/busybox",
        network_manager_path=Path("/usr/bin/nmcli"), android_gadget_path="/config/usb_gadget/g1",
        functionfs_gadget_path="/config/usb_gadget/g2", functionfs_root_path="/dev/usb-ffs/s41",
        phone_usb_controller="a600000.dwc3", resident_workers_path=phone["resident_workers"],
        resident_router_path=phone["resident_router"], multi_session_port_base=26760,
        multi_session_device_count=len(shards), remote_hash_cache_path=Path(config["phone_hash_cache"]))
    owner = DirectPhoneFfnSession(cfg)
    allowed = tuple(sorted({0, 1000000, *fractions_ppm}))
    execution = RuntimeExecutionContract(execution_mode="adaptive-split", initial_split_fraction_ppm=0,
        allowed_adaptive_fractions_ppm=allowed, batch_plan=usb.batch_plan, maximum_batch_size=max_tokens,
        queue_depth=4, phone_device_id="op15-phone", phone_endpoint="session://op15-phone",
        operator_kind="ffn", phone_shards=tuple(shards))
    parameters = {
        "model_alias": manifest.model_id, "cpu_device_id": "cpu", "gpu_device_id": "gpu", "phone_device_id": "op15-phone",
        "context_size": plan.context_size, "gpu_layers": config["gpu_layers"], "parallel": parallel,
        "batch_size": config["batch"], "ubatch_size": config["ubatch"], "ffn_transport": "functionfs-usb",
        "ffn_activation": "swiglu" if manifest.architecture != "gemma4" else "geglu",
        "ffn_column_quantum": config["column_quantum"], "ffn_n_embd": manifest.embedding_length,
        "ffn_max_tokens": max_tokens, "ffn_timeout_ms": 120000,
        "ffn_assistance_phase": "decode", "ffn_runtime_control_protocol": "decode-boundary-v1",
        "ffn_host_share_release": 1 if dormant else 0,
        "ffn_host_share_drop_cache": config.get("ffn_host_share_drop_cache", 1),
        "ffn_host_share_populate": config.get("ffn_host_share_populate", 1),
        "usb_allocator": "devmem", "usb_queue_depth": 4, "usb_concurrent_streams": 4,
        "usb_max_payload_bytes": usb.max_payload_bytes, "usb_full_duplex": 1, "usb_split_h2d": 0,
        "usb_slot_safety_bytes": usb.slot_safety_bytes, "usb_vendor_id": usb.vendor_id, "usb_product_id": usb.product_id,
        "usb_transport_generation": usb.generation, "usb_transport_profile_id": usb.profile_id,
        "usbfs_available_bytes": usb.usbfs_available_bytes, "usb_batch_plan": usb.batch_plan,
    }
    # the static plan is full width (the sessions hold every column); the decode-boundary runtime control
    # applies the actual host/phone split at the first generated token of each request
    operators = [{"operator_id": f"layer:{il}:ffn", "operator_kind": "ffn", "device_ids": ["cpu", "op15-phone"],
                  "split_axis": "none", "split_fraction_ppm": 0}
                 for il in range(manifest.block_count) if mask >> il & 1]
    command = PhysicalTransitionCommand("kv-decode-preload", "kv-decode-preload", artifact, "decode-relocation-kv-gate",
        plan.plan_sha256,
        PhysicalParticipantCommand("physical:op15-phone", "op15-phone", "session://op15-phone", "hexagon-htp", ("op15-htp",)),
        RuntimeTransitionPlan(transition_id="kv-decode-preload", device_id="op15-phone", source_state="cold", target_state="hot",
            latency_us=0, energy_uj=0, resource_ids=("op15-htp",), maturity="SHADOW", phone_shards=tuple(shards)),
        execution, parameters, phone_layout_generation=1, selection_mode="calibration", operator_plan_protocol="llama-server-http-v1",
        operator_plan={"route_id": "decode-relocation-kv-gate", "plan_sha256": plan.plan_sha256, "operators": operators,
            "assisted_operator_kind": "ffn", "execution_contract": execution.to_json()})
    env = dict(usb.server_environment())
    env.update({"S41_SERVER_FFN_ACTIVATION": parameters["ffn_activation"], "S41_SERVER_FFN_ARTIFACT_SHA256": artifact,
        "S41_SERVER_FFN_COLUMNS": str(n_ff), "S41_SERVER_FFN_F16_IO": "1", "S41_SERVER_FFN_LAYER_MASK": str(mask),
        "S41_SERVER_FFN_MAX_TOKENS": str(max_tokens), "S41_SERVER_FFN_N_EMBD": str(manifest.embedding_length),
        "S41_SERVER_FFN_RUNTIME_CONTROL": "1", "S41_SERVER_FFN_TIMEOUT_MS": "120000",
        "S41_SERVER_FFN_SHARDS": ";".join(f"{name}@session://op15-phone/{name}:{value}" for name, value in masks.items())})
    if dormant:
        env["S41_SERVER_FFN_DORMANT_HOST_SHARE"] = "1"
    return owner, command, usb, env, mask, shards


def split_phone_proofs(lines, shards, manifest, expectations):
    """Per-request phone proofs for the decode split: every owned layer must receive exactly the phone
    suffix columns for every assisted decode step. ``expectations`` maps request_id -> (phone_columns,
    expected rows per layer). Structural violations raise; row-count mismatches are recorded."""
    n_embd = manifest.embedding_length
    rows = {rid: {} for rid in expectations}
    totals = {rid: {shard.session_id: [0, 0, 0] for shard in shards} for rid in expectations}
    seen = set()
    for line in lines:
        call = parse_llama_server_ffn_call(line.rstrip())
        if call is None:
            continue
        matched = [rid for rid in expectations if call.rows_for(rid) > 0]
        if not matched:
            continue
        if (call.request_id in seen or len(matched) != len(call.contexts)
                or sum(call.rows_for(rid) for rid in matched) != call.tokens):
            raise ValueError("phone FFN call has duplicate or unowned contexts")
        phone_columns = expectations[matched[0]][0]
        owners = [shard for shard in shards if shard.layer_mask >> call.layer & 1]
        if (len(owners) != 1 or call.columns != phone_columns or call.payload_bytes != call.tokens * n_embd * 2
                or any(expectations[rid][0] != phone_columns for rid in matched)):
            raise ValueError(f"phone FFN call differs from the split plan: layer={call.layer} columns={call.columns} "
                             f"expected={phone_columns} payload={call.payload_bytes} tokens={call.tokens}")
        seen.add(call.request_id)
        for rid in matched:
            request_rows = call.rows_for(rid)
            rows[rid][call.layer] = rows[rid].get(call.layer, 0) + request_rows
            row = totals[rid][owners[0].session_id]
            row[0] += 1
            row[1] += request_rows
            row[2] += request_rows * n_embd * 2
    proofs, summary = {}, {}
    for rid, (phone_columns, expected_rows) in expectations.items():
        owned = [il for shard in shards for il in range(manifest.block_count) if shard.layer_mask >> il & 1]
        observed = rows[rid]
        summary[rid] = {"phone_columns": phone_columns, "expected_rows_per_layer": expected_rows,
                        "layers_observed": len(observed), "layers_owned": len(owned),
                        "rows_min": min(observed.values()) if observed else 0, "rows_max": max(observed.values()) if observed else 0,
                        "exact": all(observed.get(il) == expected_rows for il in owned),
                        "calls_by_session": {sid: t[0] for sid, t in totals[rid].items()}}
        proofs[rid] = tuple(LlamaServerPhoneSessionProof(
            shard.session_id, shard.endpoint, shard.artifact_sha256, shard.resident_geometry_sha256,
            shard.operator_plan_sha256, shard.session_generation, shard.layer_mask, *totals[rid][shard.session_id])
            for shard in shards if totals[rid][shard.session_id][0] > 0)
    return proofs, summary


class DecodeWatchdog:
    """Observe progress without cancelling requests or stopping a phone worker."""

    def __init__(self, request_ids):
        self.request_ids = tuple(request_ids)
        self.progress = {}
        self.lock = threading.Lock()

    def observe(self, request_id, predicted, observed_ns, terminal):
        with self.lock:
            self.progress[request_id] = (predicted, observed_ns, terminal)

    def stalled(self, now_ns):
        with self.lock:
            if not self.progress:
                return {}
            first = min(row[1] for row in self.progress.values())
            return {rid: row for rid in self.request_ids
                    if not (row := self.progress.get(rid, (0, first, False)))[2]
                    and now_ns - row[1] > 60_000_000_000}


def run_decode_cohort(client, endpoint, manifest, output, prompts, output_tokens,
                      arm, host_columns, policy, owner, sampler, record, submission_order=(), allocation_probe=None):
    """Issue one concurrent group and apply a common policy at its decode boundary."""
    members = tuple(f"kvd-{arm}-s{index}-h{host_columns}" for index in range(len(prompts)))
    watch = DecodeWatchdog(members)
    barrier = threading.Barrier(len(members))
    lock = threading.Lock()
    states = {rid: {"first": [], "progress": [], "slot": None, "result": {}, "error": None}
              for rid in members}
    controls = []
    apply_started = threading.Event()
    failed_watch = False
    if owner:
        for rid in members:
            owner.bind_ticket_generation(rid)

    def make_payload(index):
        rid = members[index]
        state = states[rid]

        def progress(slot, predicted, observed_ns, terminal):
            watch.observe(rid, predicted, observed_ns, terminal)
            with lock:
                state["slot"] = slot
                state["progress"].append((predicted, observed_ns, terminal))
                ready = all(row["slot"] is not None for row in states.values())
                apply = policy is not None and ready and not apply_started.is_set()
                if apply:
                    apply_started.set()
                    rows = tuple((key, states[key]["slot"]) for key in members)
            if apply:
                control = AdaptiveDecodeControl(rows[0][0], rows[0][1], 1, policy)
                issued = time.monotonic_ns()
                if len(rows) == 1:
                    ack, received = client.apply_ffn_control(endpoint, control, timeout_s=180)
                    ack["cohort_members"] = [{"request_id": rows[0][0], "slot_id": rows[0][1],
                        "applied_token_index": ack["applied_token_index"], "plan_generation": 1}]
                else:
                    ack, received = client.apply_ffn_cohort_control(endpoint, control, rows, timeout_s=180)
                controls.append({"issued_ns": issued, "received_ns": received, "control": control.to_json(), "ack": ack})
                save(output / "COHORT_CONTROL.json", controls[0])

        payload = LlamaCppCompletionPayload(
            request_id=rid, expected_model_alias=manifest.model_id, input_tokens=len(prompts[index]),
            output_tokens=output_tokens, prompt_tokens=prompts[index], seed=17,
            stream_path=output / f"{rid}.raw", on_first_token=state["first"].append,
            on_decode_progress=progress, timeout_s=14400, cohort_submission_order=submission_order)
        if payload.cohort_submission_order and len(payload.cohort_submission_order) != len(members):
            raise ValueError("cohort submission order does not cover its members")
        return payload

    payloads = [make_payload(index) for index in range(len(members))]

    def execute(index):
        rid, payload = members[index], payloads[index]
        state = states[rid]
        if not payload.cohort_submission_order:
            barrier.wait()
        state["started_ns"] = time.monotonic_ns()
        try:
            state["result"] = client.complete(endpoint, payload, lambda: None)
        except Exception as error:  # noqa: BLE001 - retained before any cleanup
            state["error"] = repr(error)
        finally:
            state["finished_ns"] = time.monotonic_ns()

    threads = [threading.Thread(target=execute, args=(index,), daemon=True) for index in range(len(members))]
    order = payloads[0].cohort_submission_order
    allocation_observations = []
    if order and not callable(allocation_probe):
        raise ValueError("ordered cohort requires native slot allocation receipts")
    allocation_base = len(allocation_probe()) if order else 0
    for ordinal, index in enumerate(order or range(len(members))):
        thread = threads[index]
        thread.start()
        if order and ordinal + 1 < len(members):
            while thread.is_alive():
                allocated = allocation_probe()[allocation_base:]
                if len(allocated) >= ordinal + 1:
                    allocation_observations.append({"observed_ns": time.monotonic_ns(),
                        "submitted_indices": list(order[:ordinal + 1]), "allocations": allocated})
                    break
                time.sleep(0.001)
            else:
                thread.join()
                raise RuntimeError("native server did not confirm slot allocation before completion")
    while True:
        alive = any(thread.is_alive() for thread in threads)
        errors = {rid: state["error"] for rid, state in states.items() if state["error"]}
        stalled = watch.stalled(time.monotonic_ns())
        uncertain_phone = bool(errors and apply_started.is_set())
        if (stalled or uncertain_phone) and not failed_watch:
            save(output / "WATCHDOG_FAILURE.json", {"observed_ns": time.monotonic_ns(),
                "timeout_s": 60, "stalled": stalled, "errors": errors,
                "action": "Keep rig lock and session; await normal stream completion, never kill a worker."})
            print("WATCHDOG_FAILED: retaining rig lock until all requests drain normally", flush=True)
            failed_watch = True
        if not alive and not uncertain_phone:
            break
        time.sleep(1)
    entries = []
    for index, rid in enumerate(members):
        state = states[rid]
        result = state["result"]
        tokens = list(result.get("tokens", []))
        first = state["first"][0] if state["first"] else None
        started, finished = state["started_ns"], state["finished_ns"]
        entry = {"request_id": rid, "index": index, "slot_id": state["slot"], "host_columns": host_columns,
                 "phone_columns": policy.columns if policy else 0, "split": policy is not None,
                 "started_ns": started, "first_token_ns": first, "finished_ns": finished,
                 "request_s": (finished - started) / 1e9, "prefill_s": (first - started) / 1e9 if first else None,
                 "decode_s": (finished - first) / 1e9 if first else None,
                 "prompt_ms": result.get("prompt_ms"), "predicted_ms": result.get("predicted_ms"),
                 "output_tokens": len(tokens), "decode_ms_per_token": result.get("predicted_ms", 0) / len(tokens) if tokens else None,
                 "controls": controls, "error": state["error"], "tokens_sha256": canonical_sha256(tokens)}
        save(output / f"EXECUTION-{rid}.json", {**entry, "tokens": tokens, "prompt_tokens": prompts[index],
            "progress": state["progress"], "execution": {k: v for k, v in result.items() if k != "tokens"}})
        entries.append(entry)
    record["requests"].extend(entries)
    if any(entry["error"] or entry["output_tokens"] != output_tokens for entry in entries) or failed_watch:
        raise RuntimeError("cohort did not pass the completion/hang check")
    if policy is not None and len(controls) != 1:
        raise RuntimeError("cohort did not acknowledge exactly one common policy")
    # Intersect decode intervals so prefill and the draining tail do not enter decode power.
    started = min(entry["started_ns"] for entry in entries)
    finished = max(entry["finished_ns"] for entry in entries)
    decode_start = max(entry["first_token_ns"] for entry in entries)
    if controls:
        decode_start = max(decode_start, controls[0]["received_ns"])
    decode_end = min(entry["finished_ns"] for entry in entries)
    fixture = {"prompts": prompts, "output_tokens": output_tokens, "submission_order": list(order)}
    if order:
        fixture["submission_boundary"] = "native-prefill-start-log"
    record["cohort"] = {"members": members, "parallel": len(members), "started_ns": started, "finished_ns": finished,
        "submission_order": list(order), "request_fixture_sha256": canonical_sha256(fixture),
        "slot_allocation_observations": allocation_observations,
        "decode_start_ns": decode_start, "decode_end_ns": decode_end,
        "decode_s": (decode_end - decode_start) / 1e9, "request_s": (finished - started) / 1e9,
        "request_host_energy": _energy(sampler, started, finished),
        "decode_host_energy": _energy(sampler, decode_start, decode_end),
        "phone_active_union_s": (finished - controls[0]["received_ns"]) / 1e9 if controls else 0,
        "watchdog_passed": True}
    if not controls:
        return {}
    return {row["request_id"]: (policy.columns, output_tokens - row["applied_token_index"])
            for row in controls[0]["ack"]["cohort_members"]}


def run(config, output, arm, options):
    output.mkdir(parents=False, exist_ok=False)
    manifest = ModelManifest.from_json(json.loads(Path(config["manifest"]).read_text()))
    n_ff, n_embd = manifest.feed_forward_length, manifest.embedding_length
    quantum = int(config["column_quantum"])
    parallel = config.get("parallel", 1)
    if type(parallel) is not int or parallel not in (1, 2, 4, 8):
        raise ValueError("gate parallel must be one of 1, 2, 4, 8")
    if parallel > 1 and (not options.prompt_tokens_list or arm not in ("control", "combined") or options.select):
        raise ValueError("cohort gate requires explicit per-slot prompts and a fixed control or combined arm")
    if n_ff % quantum:
        raise ValueError("column quantum must divide the FFN width")
    selection = None
    if arm == "sweep":
        columns_list = [int(v) for v in options.sweep_host_columns.split(",")]
    elif arm == "combined" and options.select:
        # scheduler-side choice from the measured atlas: objective[:required_release_bytes]; the prompt regime is
        # derived from the tokenized prompt length below, so the choice is made after tokenization
        columns_list = None
    elif arm == "combined":
        columns_list = [int(options.host_columns if options.host_columns is not None else config["host_columns"])]
    else:
        columns_list = [n_ff]
    for hc in columns_list or ():
        if not 0 <= hc <= n_ff or hc % quantum:
            raise ValueError(f"host columns {hc} must be a multiple of the quantum {quantum} within [0, {n_ff}]")
    dormant = arm != "control" and not options.no_dormant
    output_tokens = int(options.output_tokens or (config["sweep_output_tokens"] if arm == "sweep" else config["output_tokens"]))
    # this native revision counts the output layer in n_gpu_layers
    first_gpu_layer = max(0, manifest.block_count + 1 - config["gpu_layers"])
    default = {il: "cpu" if il < first_gpu_layer else "gpu" for il in range(manifest.block_count)}
    plan = plan_layer_kv(manifest, context_size=config["context"], parallel=int(config.get("parallel", 1)), ubatch_size=config["ubatch"],
        default_pool_by_layer=default, host_pool="cpu", kv_budget_by_pool=config["kv_budgets"])
    server_path, model = Path(config["server"]), Path(config["model"])
    real_model = str(model.resolve())
    save(output / "CONFIG.json", {**config, "arm": arm, "host_columns_list": columns_list, "dormant": dormant,
                                  "output_tokens": output_tokens, "prompt_chars": options.prompt_chars, "select": options.select})
    save(output / "KV_PLAN.json", {**plan.to_json(), "cpu_kv_bytes_per_token": kv_bytes_per_token(manifest, plan.cpu_layers),
                                   "gpu_kv_bytes_per_token": kv_bytes_per_token(manifest, [il for il in range(manifest.block_count) if il not in plan.cpu_layers])})
    model_sha = sha(model)
    runtime = {"server": sha(server_path), "model": model_sha,
               "libraries": {path.name: sha(path) for path in sorted(server_path.parent.glob("*.so*")) if not path.is_symlink()}}
    save(output / "RUNTIME.json", runtime)
    if model_sha != manifest.artifact_sha256:
        raise ValueError("model artifact mismatch")
    import subprocess  # noqa: PLC0415
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
    if active.strip():
        raise RuntimeError("another GPU workload is active: " + active)
    if config.get("drop_model_cache", False):
        fd = os.open(model, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    sampler = HostEnergySampler(default_host_metric_callbacks(), interval_s=0.1)
    managed = owner = None
    phone_started = False
    stop = threading.Event()
    record = {"arm": arm, "status": "RUNNING", "kv_plan_sha256": plan.plan_sha256, "prefill_policy": "local-prefill",
              "dormant_host_share": dormant, "host_columns_list": columns_list, "output_tokens": output_tokens,
              "split_selected_by": "atlas-selector" if options.select else "operator", "selection": None,
              "scheduler_qualified": False, "requests": []}
    try:
        sampler.start()
        time.sleep(0.3)
        paid_start = time.monotonic_ns()
        environment, shards, mask = {}, (), 0
        with_phone = arm != "control"
        if with_phone:
            if columns_list is None:
                atlas = DecodeSplitAtlas.load(Path(config["decode_split_atlas"]))
                fractions = sorted({row.split_fraction_ppm for row in atlas.rows
                                    if row.environment.artifact_sha256 == manifest.artifact_sha256 and row.split_fraction_ppm})
            else:
                fractions = [(n_ff - hc) * 1000000 // n_ff for hc in columns_list if hc < n_ff]
            owner, command, usb, environment, mask, shards = phone_owner(config, manifest, output, plan, fractions, dormant)
            save(output / "PHONE_COMMAND.json", command.to_json())
            save(output / "PHONE_PREFLIGHT.json", owner.preflight().to_json())
            started = time.monotonic_ns()
            phone_started = True
            ready = owner.start(command, manifest, usb)
            record["phone_preload_s"] = (time.monotonic_ns() - started) / 1e9
            save(output / "PHONE_READY.json", ready.to_json())
            print("PHONE_READY", record["phone_preload_s"], flush=True)
        layer_indices = [il for il in range(manifest.block_count) if mask >> il & 1]
        launcher = LlamaServerProcessLauncher(LlamaServerProcessConfiguration(server_path=server_path,
            model_paths_by_artifact={manifest.artifact_sha256: model}, library_paths_by_device={"gpu": (Path(config["cuda_lib_dir"]),)},
            executable_device_names={"gpu": "CUDA0"}, output_directory=output,
            common_library_paths=(server_path.parent, Path(config["cuda_lib_dir"]))))
        contract = launch_contract(config, plan, manifest, with_phone, environment)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        endpoint = f"http://127.0.0.1:{port}"
        started = time.monotonic_ns()
        managed = launcher.launch_contract(endpoint, contract, manifest, label=arm, control_check=lambda: None)
        record["desktop_load_s"] = (time.monotonic_ns() - started) / 1e9
        runtime_launch_sha256 = record_server_identity(managed, port, output, runtime)
        record["memory_ready"] = _memory_record(managed, model, [])
        record["memory_ready"]["status"] = process_status(managed.pid)
        record["memory_ready"]["cgroup"] = cgroup_memory()
        record["memory_ready"]["meminfo"] = meminfo()
        record["memory_ready"]["kv_lazy_lines"] = [line for line in managed.stderr_lines if "zeroed lazily" in line]
        save(output / "READY.json", record)
        print("DESKTOP_READY", arm, record["desktop_load_s"], flush=True)
        pid = managed.pid

        def observe():
            count = 0
            with (output / "MEMORY.jsonl").open("x") as stream:
                while not stop.is_set():
                    try:
                        row = {"time_ns": time.monotonic_ns(), "status": process_status(pid), "cgroup": cgroup_memory(),
                               "meminfo": meminfo()}
                        if count % 8 == 0:
                            row["model_file_rss"] = model_file_rss(pid, real_model)
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        stream.flush()
                    except (OSError, ValueError):
                        pass
                    count += 1
                    stop.wait(0.5)
        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        def tokenize(text, add_special):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            connection.request("POST", "/tokenize", json.dumps({"content": text, "add_special": add_special}),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError("tokenization failed")
            value = tuple(json.loads(response.read())["tokens"])
            connection.close()
            return value

        document = Path(config["prompt_file"]).read_text()
        if options.prompt_chars:
            document = document[:options.prompt_chars]
        prompt = document + config.get("prompt_suffix", "")
        tokens = tokenize(prompt, True)
        prompt_variants = [("", tokens)]
        if options.prompt_tokens_list:
            # exact prompt lengths: the document's first N-k tokens plus the tokenized task suffix (k tokens)
            document_tokens = tokenize(Path(config["prompt_file"]).read_text(), True)
            suffix_tokens = tokenize(config.get("prompt_suffix", ""), False)
            prompt_variants = []
            for length in (int(v) for v in options.prompt_tokens_list.split(",")):
                if length <= len(suffix_tokens) + 1 or length - len(suffix_tokens) > len(document_tokens):
                    raise ValueError(f"prompt length {length} is not reachable from the document")
                prompt_variants.append((f"p{length}", tuple(document_tokens[:length - len(suffix_tokens)]) + suffix_tokens))
            tokens = prompt_variants[0][1]
        slot_cells = plan.context_size // int(config.get("parallel", 1))
        for _tag, variant in prompt_variants:
            if len(variant) + output_tokens > slot_cells:
                raise ValueError(f"prompt of {len(variant)} tokens exceeds the slot context {slot_cells}; do not silently truncate")
        if columns_list is None:
            objective, _, required = options.select.partition(":")
            selection = select_decode_split(DecodeSplitAtlas.load(Path(config["decode_split_atlas"])),
                                            environment=selection_environment(config, plan, runtime), prompt_tokens=len(tokens),
                                            output_tokens=output_tokens, feed_forward_length=n_ff, objective=objective,
                                            required_release_bytes=int(required or 0), phone_available=with_phone)
            columns_list = [selection.host_columns]
            record["selection"] = selection.to_json()
            record["host_columns_list"] = columns_list
            save(output / "SELECTION.json", selection.to_json())
            print("SELECTED", objective, "fraction_ppm", selection.split_fraction_ppm, "host_columns", selection.host_columns,
                  "expected_release", selection.expected_release_bytes, flush=True)
        save(output / "REQUEST.json", {"prompt_sha256": "sha256:" + hashlib.sha256(prompt.encode()).hexdigest(),
                                       "prompt_tokens": tokens, "output_tokens": output_tokens,
                                       "prompt_variants": {tag: len(variant) for tag, variant in prompt_variants},
                                       "slot_prompts": [variant for _, variant in prompt_variants]})
        print("REQUEST", len(tokens), output_tokens, "variants", [len(v) for _, v in prompt_variants], flush=True)
        client = LlamaCppHttpClient()
        expectations = {}
        generation = 0
        def decode_policy(host_columns):
            phone_columns = n_ff - host_columns
            desktop = {"artifact": manifest.artifact_sha256, "gpu_layers": config["gpu_layers"], "context": plan.context_size,
                       "batch": config["batch"], "ubatch": config["ubatch"], "kv_plan": plan.plan_sha256,
                       "runtime_launch_sha256": runtime_launch_sha256}
            plan_json = {"desktop": desktop, "layers": layer_indices, "columns": phone_columns, "host_columns": host_columns,
                         "sessions": {shard.session_id: shard.layer_mask for shard in shards}, "activation": "swiglu",
                         "io": "f16", "transport": "functionfs", "column_quantum": quantum}
            return AdaptiveDecodePolicy(
                route_id=f"decode-relocation-kv-{phone_columns * 100 // n_ff}", executor_id="native-calibration",
                operator_plan_sha256=canonical_sha256(plan_json), desktop_parent_route_id="calibration-cuda-parent",
                desktop_placement_sha256=canonical_sha256(desktop), layer_indices=tuple(layer_indices),
                layer_mask=mask, columns=phone_columns, split_fraction_ppm=phone_columns * 1000000 // n_ff,
                resource_ids=("desktop-cpu", "desktop-cuda", "op15-htp", "op15-functionfs"))

        cohort = parallel > 1 or config.get("usb_batch_plan", "split-row") == "coalesced-batch"
        if cohort:
            if len(prompt_variants) != parallel or len(columns_list) != 1:
                raise ValueError("cohort gate requires exactly one prompt per slot and one split")
            host_columns = columns_list[0]
            policy = decode_policy(host_columns) if with_phone and host_columns < n_ff else None
            expectations = run_decode_cohort(client, endpoint, manifest, output,
                [variant for _, variant in prompt_variants], output_tokens, arm, host_columns, policy, owner, sampler, record,
                tuple(config.get("cohort_submission_order", ())), managed.slot_allocation_events)
        for index, host_columns in enumerate(columns_list):
            if cohort:
                break
            split = with_phone and host_columns < n_ff
            phone_columns = n_ff - host_columns
            request_id = f"kvd-{arm}-{index:02d}-h{host_columns}"
            ranges = share_ranges(config["gguf_py_dir"], model, layer_indices, host_columns, n_embd, n_ff) if split else []
            first, controls, progress, samples, mid_stats = [], [], [], [], []
            done = threading.Event()

            def first_token(observed_ns, split=split, request_id=request_id, host_columns=host_columns, phone_columns=phone_columns):
                nonlocal generation
                first.append(observed_ns)
                if not split:
                    return
                generation += 1
                policy = decode_policy(host_columns)
                control = AdaptiveDecodeControl(request_id, 0, generation, policy)
                issued = time.monotonic_ns()
                ack, received = LlamaCppHttpClient.apply_ffn_control(endpoint, control, timeout_s=180.0)
                controls.append({"issued_ns": issued, "received_ns": received, "control": control.to_json(), "ack": ack})

            def on_progress(slot, predicted, t_ns, terminal):
                progress.append((predicted, t_ns, terminal))

            def side_effects(request_id=request_id, split=split, ranges=ranges):
                if not done.wait(8.0):
                    try:
                        samples.append({"tag": "prefill+8s", "time_ns": time.monotonic_ns(),
                                        "share_residency": residency(pid, real_model, ranges) if ranges else None,
                                        "status": process_status(pid)})
                    except OSError as error:
                        samples.append({"tag": "prefill+8s", "error": repr(error)})
                while not first and not done.is_set():
                    done.wait(0.1)
                for delay in (20.0, 120.0):
                    if done.wait(delay):
                        return
                    try:
                        samples.append({"tag": f"decode+{delay:.0f}s", "time_ns": time.monotonic_ns(),
                                        "share_residency": residency(pid, real_model, ranges) if ranges else None,
                                        "status": process_status(pid), "cgroup": cgroup_memory()})
                    except OSError as error:
                        samples.append({"tag": f"decode+{delay:.0f}s", "error": repr(error)})
                    if split and not mid_stats:
                        try:
                            stats, _received = LlamaCppHttpClient.read_ffn_stats(endpoint, request_id, 0, timeout_s=180.0)
                            mid_stats.append(stats)
                        except Exception as error:  # noqa: BLE001 - recorded
                            mid_stats.append({"error": repr(error)})

            if owner:
                owner.bind_ticket_generation(request_id)
            payload = LlamaCppCompletionPayload(
                request_id=request_id, expected_model_alias=manifest.model_id, input_tokens=len(tokens),
                output_tokens=output_tokens, prompt_tokens=tokens, seed=17, stream_path=output / f"{request_id}.raw",
                on_first_token=first_token, on_decode_progress=on_progress, timeout_s=14400)
            side = threading.Thread(target=side_effects, daemon=True)
            started = time.monotonic_ns()
            side.start()
            error = None
            result = {}
            try:
                result = client.complete(endpoint, payload, lambda: None)
            except Exception as failure:  # noqa: BLE001 - recorded; the gate decides
                error = f"{type(failure).__name__}: {failure}"
            finished = time.monotonic_ns()
            done.set()
            side.join(timeout=60)
            out_tokens = list(result.get("tokens", []))
            entry = {"request_id": request_id, "index": index, "host_columns": host_columns, "phone_columns": phone_columns if split else 0,
                     "split": split, "started_ns": started, "first_token_ns": first[0] if first else None, "finished_ns": finished,
                     "request_s": (finished - started) / 1e9, "prefill_s": (first[0] - started) / 1e9 if first else None,
                     "decode_s": (finished - first[0]) / 1e9 if first else None,
                     "prompt_ms": result.get("prompt_ms"), "predicted_ms": result.get("predicted_ms"), "output_tokens": len(out_tokens),
                     "decode_ms_per_token": (float(result["predicted_ms"]) / len(out_tokens)) if out_tokens and result.get("predicted_ms") else None,
                     "controls": controls, "residency_samples": samples, "phone_stats_mid_decode": mid_stats[0] if mid_stats else None,
                     "error": error, "tokens_sha256": "sha256:" + hashlib.sha256(json.dumps(out_tokens).encode()).hexdigest(),
                     "result_keys": sorted(result.keys())}
            if error is None:
                entry["request_host_energy"] = _energy(sampler, started, finished)
                if first:
                    entry["prefill_host_energy"] = _energy(sampler, started, first[0])
                    entry["decode_host_energy"] = _energy(sampler, first[0], finished)
            save(output / f"EXECUTION-{request_id}.json", {**entry, "tokens": out_tokens, "progress": progress,
                 "execution": {k: v for k, v in result.items() if k != "tokens"}})
            record["requests"].append(entry)
            if split:
                # the control is acknowledged at applied_token_index; every later decode step is assisted
                applied = 1
                for control in controls:
                    ack = control.get("ack")
                    if isinstance(ack, dict) and isinstance(ack.get("applied_token_index"), int):
                        applied = ack["applied_token_index"]
                expectations[request_id] = (phone_columns, output_tokens - applied)
            print("COMPLETED", request_id, f"prefill={entry['prefill_s']} decode_ms/tok={entry['decode_ms_per_token']} error={error}", flush=True)
            if error is not None:
                raise RuntimeError(f"request {request_id} failed: {error}")
        finished_all = time.monotonic_ns()
        record["memory_finished"] = _memory_record(managed, model, [])
        record["memory_finished"]["status"] = process_status(pid)
        record["memory_finished"]["cgroup"] = cgroup_memory()
        record["paid_host_energy"] = _energy(sampler, paid_start, finished_all)
        record["paid_s"] = (finished_all - paid_start) / 1e9
        decode_s = sum(r["decode_s"] or 0.0 for r in record["requests"] if r["split"])
        if cohort:
            decode_s = record["cohort"]["phone_active_union_s"]
        record["phone_energy_note"] = ("Assumed, not measured. 'paid': active power over the whole paid span (conservative); "
                                       "'assisting': active power only while the phone assists decode, idle power otherwise. "
                                       "Control arm: idle power over the paid span.")
        record["assumed_phone_j"] = {str(p): {"paid": record["paid_s"] * (p if with_phone else PHONE_IDLE_W),
                                              "assisting": (decode_s * p + (record["paid_s"] - decode_s) * PHONE_IDLE_W) if with_phone
                                              else record["paid_s"] * PHONE_IDLE_W} for p in (3, 4.5, 6)}
        stop.set()
        observer.join(timeout=5)
        managed.stop()
        lines = list(managed.stderr_lines)
        proofs = [proof for proof in (RuntimeHostShareReleaseProof.parse_line(line) for line in lines) if proof is not None]
        record["dormant_proofs"] = [{"phase": p.phase, "layer_mask": p.layer_mask, "host_columns": p.host_columns,
                                     "released_bytes": p.released_bytes, "ranges": p.ranges, "elapsed_us": p.elapsed_us} for p in proofs]
        record["dormant_lower_bounds"] = {str(hc): host_share_release_lower_bound_bytes(manifest, mask, hc)
                                          for hc in columns_list if with_phone and hc < n_ff}
        save(output / "SERVER_FFN_LINES.json", [line for line in lines if "S41SERVERFFN" in line])
        if owner:
            phone_proofs, summary = split_phone_proofs(lines, shards, manifest, expectations)
            record["phone_proof_summary"] = summary
            save(output / "NATIVE_EXECUTION_PROOFS.json", {rid: [p.to_json() for p in proofs] for rid, proofs in phone_proofs.items()})
            for rid, request_proofs in phone_proofs.items():
                if request_proofs:
                    owner.record_execution_proof(rid, manifest.artifact_sha256, request_proofs)
        managed = None
        if owner:
            close = owner.finish(require_execution=True)
            save(output / "PHONE_CLOSE.json", close.to_json())
            phone_started = False
        record["status"] = "COMPLETED"
        save(output / "RESULT.json", record)
        print("COMPLETED", arm, record["paid_s"], flush=True)
    except BaseException as error:
        if managed and "memory_ready" in record:
            record["memory_finished"] = {"cgroup": cgroup_memory()}
        save(output / "FAILURE.json", {**record, "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        stop.set()
        if managed:
            managed.stop()
        if owner and phone_started:
            save(output / "PHONE_ABORT.json", owner.abort().to_json())
        sampler.stop()
        save(output / "POWER_SAMPLES.json", sampler.rows())


def run_admission(config, output, options):
    """Two-request admission gate on one server (must run inside a MemoryMax scope):
    book share -> prompt 1 -> decode proof credits the release -> tenant occupies the released room (ledger +
    touched anonymous memory) -> prompt 2 is refused while the room is taken -> tenant leaves -> share re-reserved
    -> prompt 2 runs (the server populates the share, decodes, releases again as generation 2)."""
    import subprocess  # noqa: PLC0415
    output.mkdir(parents=False, exist_ok=False)
    manifest = ModelManifest.from_json(json.loads(Path(config["manifest"]).read_text()))
    n_ff, n_embd = manifest.feed_forward_length, manifest.embedding_length
    quantum = int(config["column_quantum"])
    output_tokens = int(options.output_tokens or config["sweep_output_tokens"])
    memory_max = cgroup_memory_max()
    if memory_max is None:
        raise RuntimeError("the admission arm must run inside a cgroup scope with MemoryMax set")
    first_gpu_layer = max(0, manifest.block_count + 1 - config["gpu_layers"])
    default = {il: "cpu" if il < first_gpu_layer else "gpu" for il in range(manifest.block_count)}
    plan = plan_layer_kv(manifest, context_size=config["context"], parallel=int(config.get("parallel", 1)), ubatch_size=config["ubatch"],
        default_pool_by_layer=default, host_pool="cpu", kv_budget_by_pool=config["kv_budgets"])
    server_path, model = Path(config["server"]), Path(config["model"])
    real_model = str(model.resolve())
    save(output / "CONFIG.json", {**config, "arm": "admission", "output_tokens": output_tokens, "prompt_chars": options.prompt_chars,
                                  "select": options.select, "memory_max_bytes": memory_max})
    save(output / "KV_PLAN.json", {**plan.to_json(), "cpu_kv_bytes_per_token": kv_bytes_per_token(manifest, plan.cpu_layers),
                                   "gpu_kv_bytes_per_token": kv_bytes_per_token(manifest, [il for il in range(manifest.block_count) if il not in plan.cpu_layers])})
    model_sha = sha(model)
    runtime = {"server": sha(server_path), "model": model_sha,
               "libraries": {path.name: sha(path) for path in sorted(server_path.parent.glob("*.so*")) if not path.is_symlink()}}
    save(output / "RUNTIME.json", runtime)
    if model_sha != manifest.artifact_sha256:
        raise ValueError("model artifact mismatch")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
    if active.strip():
        raise RuntimeError("another GPU workload is active: " + active)
    if config.get("drop_model_cache", False):
        fd = os.open(model, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    atlas = DecodeSplitAtlas.load(Path(config["decode_split_atlas"]))
    fractions = sorted({row.split_fraction_ppm for row in atlas.rows
                        if row.environment.artifact_sha256 == manifest.artifact_sha256 and row.split_fraction_ppm})
    sampler = HostEnergySampler(default_host_metric_callbacks(), interval_s=0.1)
    managed = owner = consumer = None
    phone_started = False
    stop = threading.Event()
    events = []
    ledger = RuntimeMemoryLedger()
    pool = "desktop-host"
    accountant = DecodeReleaseAccountant(ledger, host_pool=pool)
    record = {"arm": "admission", "status": "RUNNING", "kv_plan_sha256": plan.plan_sha256, "prefill_policy": "local-prefill",
              "dormant_host_share": True, "memory_max_bytes": memory_max, "output_tokens": output_tokens,
              "split_selected_by": "atlas-selector", "scheduler_qualified": False, "events": events, "requests": []}

    def event(kind, **fields):
        row = {"kind": kind, "time_ns": time.monotonic_ns(), "cgroup": cgroup_memory(), **fields}
        if managed is not None:
            try:
                row["server_status"] = process_status(managed.pid)
            except OSError:
                pass
        row["ledger_reserved_bytes"] = ledger.snapshot()["by_resource_bytes"].get(pool, 0)
        events.append(row)
        print("EVENT", kind, {k: v for k, v in fields.items() if k not in ("ledger",)}, flush=True)
        return row

    try:
        sampler.start()
        time.sleep(0.3)
        paid_start = time.monotonic_ns()
        owner, command, usb, environment, mask, shards = phone_owner(config, manifest, output, plan, fractions, True)
        save(output / "PHONE_COMMAND.json", command.to_json())
        save(output / "PHONE_PREFLIGHT.json", owner.preflight().to_json())
        started = time.monotonic_ns()
        phone_started = True
        ready = owner.start(command, manifest, usb)
        record["phone_preload_s"] = (time.monotonic_ns() - started) / 1e9
        save(output / "PHONE_READY.json", ready.to_json())
        print("PHONE_READY", record["phone_preload_s"], flush=True)
        layer_indices = [il for il in range(manifest.block_count) if mask >> il & 1]
        launcher = LlamaServerProcessLauncher(LlamaServerProcessConfiguration(server_path=server_path,
            model_paths_by_artifact={manifest.artifact_sha256: model}, library_paths_by_device={"gpu": (Path(config["cuda_lib_dir"]),)},
            executable_device_names={"gpu": "CUDA0"}, output_directory=output,
            common_library_paths=(server_path.parent, Path(config["cuda_lib_dir"]))))
        contract = launch_contract(config, plan, manifest, True, environment)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        endpoint = f"http://127.0.0.1:{port}"
        started = time.monotonic_ns()
        managed = launcher.launch_contract(endpoint, contract, manifest, label="admission", control_check=lambda: None)
        record["desktop_load_s"] = (time.monotonic_ns() - started) / 1e9
        runtime_launch_sha256 = record_server_identity(managed, port, output, runtime)
        pid = managed.pid
        stderr_path = output / "admission.stderr"
        ready_status = process_status(pid)
        record["memory_ready"] = {"status": ready_status, "cgroup": cgroup_memory(), "meminfo": meminfo(), "model_file_rss": model_file_rss(pid, real_model)}
        print("DESKTOP_READY admission", record["desktop_load_s"], flush=True)

        def observe():
            count = 0
            with (output / "MEMORY.jsonl").open("x") as stream:
                while not stop.is_set():
                    try:
                        row = {"time_ns": time.monotonic_ns(), "status": process_status(pid), "cgroup": cgroup_memory(), "meminfo": meminfo()}
                        if count % 8 == 0:
                            row["model_file_rss"] = model_file_rss(pid, real_model)
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        stream.flush()
                    except (OSError, ValueError):
                        pass
                    count += 1
                    stop.wait(0.5)
        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        document = Path(config["prompt_file"]).read_text()
        if options.prompt_chars:
            document = document[:options.prompt_chars]
        prompt = document + config.get("prompt_suffix", "")
        connection.request("POST", "/tokenize", json.dumps({"content": prompt, "add_special": True}), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError("tokenization failed")
        tokens = tuple(json.loads(response.read())["tokens"])
        connection.close()
        if len(tokens) + output_tokens > plan.context_size:
            raise ValueError("prompt exceeds context; do not silently truncate")
        save(output / "REQUEST.json", {"prompt_sha256": "sha256:" + hashlib.sha256(prompt.encode()).hexdigest(),
                                       "prompt_tokens": tokens, "output_tokens": output_tokens})
        objective, _, required = (options.select or "energy").partition(":")
        selection = select_decode_split(atlas, environment=selection_environment(config, plan, runtime), prompt_tokens=len(tokens),
                                        output_tokens=output_tokens, feed_forward_length=n_ff, objective=objective,
                                        required_release_bytes=int(required or 0), phone_available=True)
        if not selection.dormant_release:
            raise RuntimeError("the admission gate needs a split with a decode-phase release")
        record["selection"] = selection.to_json()
        save(output / "SELECTION.json", selection.to_json())
        host_columns, phone_columns, expected = selection.host_columns, selection.phone_columns, selection.expected_release_bytes
        print("SELECTED", objective, "fraction_ppm", selection.split_fraction_ppm, "host_columns", host_columns, "expected_release", expected, flush=True)
        ranges = share_ranges(config["gguf_py_dir"], model, layer_indices, host_columns, n_embd, n_ff)

        # ---- ledger: the live host budget is the scope's MemoryMax; the server's resident weights (minus the
        # share), a workspace allowance and the request KV are reserved as ordinary demands, the share via the accountant
        safety = int(options.safety_reserve_bytes)
        snapshot = RuntimePlacementSnapshot("admission-scope", 0, 10**15, {pool: DeviceMemoryCapacity(pool, memory_max, 0, safety)})
        base = max(1, ready_status["VmRSS"] - expected)
        kv_bytes = (len(tokens) + output_tokens) * kv_bytes_per_token(manifest, plan.cpu_layers)
        ledger.reserve("server:base", (RuntimeMemoryDemand("server:base", pool, "resident-weights-and-anon", base, 0, "request"),), snapshot)
        ledger.reserve("server:workspace", (RuntimeMemoryDemand("server:workspace", pool, "prefill-workspace", int(options.workspace_bytes), 0, "request"),), snapshot)
        ledger.reserve("server:kv", (RuntimeMemoryDemand("server:kv", pool, "context-kv", kv_bytes, 0, "request"),), snapshot)
        binding = ShareBinding(endpoint, manifest.artifact_sha256, mask, host_columns, expected)
        accountant.reserve_share("server", binding, snapshot)
        event("share_booked", base_bytes=base, workspace_bytes=int(options.workspace_bytes), kv_bytes=kv_bytes, expected_release_bytes=expected,
              memory_max_bytes=memory_max, safety_reserve_bytes=safety, accountant=accountant.to_json())
        client = LlamaCppHttpClient()
        expectations = {}
        generation = [0]
        proofs_seen = [0]

        def execute(request_id, expect_generation):
            first, controls, progress, samples = [], [], [], []
            done = threading.Event()

            def first_token(observed_ns):
                first.append(observed_ns)
                generation[0] += 1
                desktop = {"artifact": manifest.artifact_sha256, "gpu_layers": config["gpu_layers"], "context": plan.context_size,
                           "batch": config["batch"], "ubatch": config["ubatch"], "kv_plan": plan.plan_sha256,
                           "runtime_launch_sha256": runtime_launch_sha256}
                plan_json = {"desktop": desktop, "layers": layer_indices, "columns": phone_columns, "host_columns": host_columns,
                             "sessions": {shard.session_id: shard.layer_mask for shard in shards}, "activation": "swiglu",
                             "io": "f16", "transport": "functionfs", "column_quantum": quantum}
                policy = AdaptiveDecodePolicy(
                    route_id=f"admission-gate-{phone_columns * 100 // n_ff}", executor_id="native-calibration",
                    operator_plan_sha256=canonical_sha256(plan_json), desktop_parent_route_id="calibration-cuda-parent",
                    desktop_placement_sha256=canonical_sha256(desktop), layer_indices=tuple(layer_indices),
                    layer_mask=mask, columns=phone_columns, split_fraction_ppm=phone_columns * 1000000 // n_ff,
                    resource_ids=("desktop-cpu", "desktop-cuda", "op15-htp", "op15-functionfs"))
                control = AdaptiveDecodeControl(request_id, 0, generation[0], policy)
                issued = time.monotonic_ns()
                ack, received = LlamaCppHttpClient.apply_ffn_control(endpoint, control, timeout_s=180.0)
                controls.append({"issued_ns": issued, "received_ns": received, "control": control.to_json(), "ack": ack})

            def on_progress(slot, predicted, t_ns, terminal):
                progress.append((predicted, t_ns, terminal))

            def credit_release():
                # the scheduler-side credit: wait for this server's decode proof, bind and consume it once
                while not first and not done.is_set():
                    done.wait(0.1)
                if done.is_set():
                    return
                try:
                    proof, seen = wait_for_release_proof(stderr_path, proofs_seen[0], 120.0)
                    proofs_seen[0] = seen
                    credited = accountant.enter_decode("server", proof, snapshot, endpoint=endpoint, release_generation=seen)
                    samples.append(event("release_credited", request_id=request_id, release_generation=seen, credited_bytes=credited,
                                         proof={"released_bytes": proof.released_bytes, "elapsed_us": proof.elapsed_us, "ranges": proof.ranges},
                                         headroom_bytes=accountant.decode_phase_headroom_bytes(snapshot),
                                         share_residency=residency(pid, real_model, ranges), accountant=accountant.to_json()))
                    if seen != expect_generation:
                        raise RuntimeError(f"unexpected release generation {seen}")
                except Exception as error:  # noqa: BLE001 - recorded, checked after the request
                    samples.append(event("release_credit_failed", request_id=request_id, error=repr(error)))

            owner.bind_ticket_generation(request_id)
            payload = LlamaCppCompletionPayload(request_id=request_id, expected_model_alias=manifest.model_id, input_tokens=len(tokens),
                output_tokens=output_tokens, prompt_tokens=tokens, seed=17, stream_path=output / f"{request_id}.raw",
                on_first_token=first_token, on_decode_progress=on_progress, timeout_s=3600)
            side = threading.Thread(target=credit_release, daemon=True)
            started_ns = time.monotonic_ns()
            side.start()
            result = client.complete(endpoint, payload, lambda: None)
            finished_ns = time.monotonic_ns()
            done.set()
            side.join(timeout=130)
            out = list(result.get("tokens", []))
            entry = {"request_id": request_id, "host_columns": host_columns, "phone_columns": phone_columns, "split": True,
                     "started_ns": started_ns, "first_token_ns": first[0] if first else None, "finished_ns": finished_ns,
                     "request_s": (finished_ns - started_ns) / 1e9, "prefill_s": (first[0] - started_ns) / 1e9 if first else None,
                     "decode_s": (finished_ns - first[0]) / 1e9 if first else None, "prompt_ms": result.get("prompt_ms"),
                     "predicted_ms": result.get("predicted_ms"), "output_tokens": len(out),
                     "decode_ms_per_token": (float(result["predicted_ms"]) / len(out)) if out and result.get("predicted_ms") else None,
                     "controls": controls, "error": None, "tokens_sha256": "sha256:" + hashlib.sha256(json.dumps(out).encode()).hexdigest(),
                     "request_host_energy": _energy(sampler, started_ns, finished_ns)}
            if first:
                entry["prefill_host_energy"] = _energy(sampler, started_ns, first[0])
                entry["decode_host_energy"] = _energy(sampler, first[0], finished_ns)
            save(output / f"EXECUTION-{request_id}.json", {**entry, "tokens": out, "progress": progress, "execution": {k: v for k, v in result.items() if k != "tokens"}})
            record["requests"].append(entry)
            applied = 1
            for control in controls:
                ack = control.get("ack")
                if isinstance(ack, dict) and isinstance(ack.get("applied_token_index"), int):
                    applied = ack["applied_token_index"]
            expectations[request_id] = (phone_columns, output_tokens - applied)
            print("COMPLETED", request_id, f"prefill={entry['prefill_s']} decode_ms/tok={entry['decode_ms_per_token']}", flush=True)
            return entry

        # ---- request 1: prompt admitted (share resident), release credited at the first token
        execute("adm-01", 1)
        if accountant.state("server") != "decode-released":
            raise RuntimeError("request 1 did not credit its release: " + accountant.state("server"))
        event("request_1_done", share_residency=residency(pid, real_model, ranges))
        # ---- a tenant takes the released room: ledger reservation + touched anonymous memory in this cgroup
        consume = int(options.consume_bytes) if options.consume_bytes else max(1, expected - 1024**3)
        accountant.reserve_decode_growth("tenant", consume, snapshot)
        consumer = spawn_consumer(consume, output / "consumer.log")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and "CONSUMER_READY" not in (output / "consumer.log").read_text():
            if consumer.poll() is not None:
                raise RuntimeError("consumer exited early")
            time.sleep(0.2)
        consumer_status = process_status(consumer.pid)
        event("tenant_admitted", consume_bytes=consume, consumer_pid=consumer.pid, consumer_status=consumer_status,
              headroom_bytes=accountant.decode_phase_headroom_bytes(snapshot))
        # ---- request 2 arrives: the share must be resident again before its prompt -> refused while the room is taken
        admissible = accountant.preview_prompt_admission("server", snapshot)
        blocked_error = None
        try:
            accountant.restore_before_prompt("server", snapshot)
        except RuntimeResourceError as error:
            blocked_error = repr(error)
        charged = cgroup_memory().get("memory.current")
        would_exceed = (charged if isinstance(charged, int) else 0) + expected - memory_max
        event("request_2_blocked", admissible=admissible, restore_error=blocked_error, state=accountant.state("server"),
              charged_bytes=charged, restore_would_exceed_cap_by_bytes=would_exceed, accountant=accountant.to_json())
        if admissible or blocked_error is None:
            raise RuntimeError("the second prompt was admitted although the released room is taken")
        replay_error = None
        try:  # the consumed proof must not credit again
            proof_1 = [p for p in (RuntimeHostShareReleaseProof.parse_line(line) for line in stderr_path.read_text(errors="replace").splitlines())
                       if p is not None and p.phase == "decode"][0]
            accountant.enter_decode("server", proof_1, snapshot, endpoint=endpoint, release_generation=1)
        except Exception as error:  # noqa: BLE001 - expected
            replay_error = repr(error)
        event("replay_refused", error=replay_error)
        if replay_error is None:
            raise RuntimeError("a consumed release proof credited twice")
        time.sleep(float(options.hold_seconds))
        event("held", hold_seconds=float(options.hold_seconds))
        # ---- the tenant leaves: capacity returns, the share is re-reserved, the prompt is admitted
        consumer.terminate()
        consumer.wait(timeout=60)
        consumer = None
        accountant.release_growth("tenant")
        time.sleep(1.0)
        admissible = accountant.preview_prompt_admission("server", snapshot)
        accountant.restore_before_prompt("server", snapshot)
        event("request_2_admitted", admissible=admissible, state=accountant.state("server"), accountant=accountant.to_json())
        if not admissible or accountant.state("server") != "prefill-resident":
            raise RuntimeError("the second prompt was not admitted after the tenant left")
        # ---- request 2: the server populates the share before prefill, decodes, releases again (generation 2)
        execute("adm-02", 2)
        restores = [p for p in (RuntimeHostShareReleaseProof.parse_line(line) for line in stderr_path.read_text(errors="replace").splitlines())
                    if p is not None and p.phase == "local"]
        event("request_2_done", server_restore_proofs=[{"restored_bytes": p.released_bytes, "elapsed_us": p.elapsed_us} for p in restores],
              share_residency=residency(pid, real_model, ranges))
        if len(restores) != 1:
            raise RuntimeError(f"expected exactly one server-side restore before prompt 2, saw {len(restores)}")
        finished_all = time.monotonic_ns()
        record["paid_s"] = (finished_all - paid_start) / 1e9
        record["paid_host_energy"] = _energy(sampler, paid_start, finished_all)
        stop.set()
        observer.join(timeout=5)
        managed.stop()
        lines = list(managed.stderr_lines)
        proofs = [proof for proof in (RuntimeHostShareReleaseProof.parse_line(line) for line in lines) if proof is not None]
        record["dormant_proofs"] = [{"phase": p.phase, "layer_mask": p.layer_mask, "host_columns": p.host_columns,
                                     "released_bytes": p.released_bytes, "ranges": p.ranges, "elapsed_us": p.elapsed_us} for p in proofs]
        save(output / "SERVER_FFN_LINES.json", [line for line in lines if "S41SERVERFFN" in line][:20000])
        phone_proofs, summary = split_phone_proofs(lines, shards, manifest, expectations)
        record["phone_proof_summary"] = summary
        save(output / "NATIVE_EXECUTION_PROOFS.json", {rid: [p.to_json() for p in prs] for rid, prs in phone_proofs.items()})
        for rid, request_proofs in phone_proofs.items():
            if request_proofs:
                owner.record_execution_proof(rid, manifest.artifact_sha256, request_proofs)
        managed = None
        close = owner.finish(require_execution=True)
        save(output / "PHONE_CLOSE.json", close.to_json())
        phone_started = False
        record["accountant"] = accountant.to_json()
        record["status"] = "COMPLETED"
        save(output / "RESULT.json", record)
        print("COMPLETED admission", record["paid_s"], flush=True)
    except BaseException as error:
        save(output / "FAILURE.json", {**record, "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        stop.set()
        if consumer is not None and consumer.poll() is None:
            consumer.terminate()
        if managed:
            managed.stop()
        if owner and phone_started:
            save(output / "PHONE_ABORT.json", owner.abort().to_json())
        sampler.stop()
        save(output / "POWER_SAMPLES.json", sampler.rows())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "combined", "sweep", "pair", "admission"), required=True)
    parser.add_argument("--consume-bytes", type=int, default=None, help="admission arm: tenant size (default: expected release - 1 GiB)")
    parser.add_argument("--hold-seconds", type=float, default=15.0, help="admission arm: how long the refused prompt is held before the tenant leaves")
    parser.add_argument("--safety-reserve-bytes", type=int, default=256 * 1024**2, help="admission arm: ledger reserve below MemoryMax")
    parser.add_argument("--workspace-bytes", type=int, default=1024**3, help="admission arm: prefill workspace allowance reserved for the server")
    parser.add_argument("--host-columns", type=int, default=None, help="combined arm: desktop prefix columns (multiple of the quantum)")
    parser.add_argument("--select", default=None, help="combined arm: choose the split from the measured atlas: latency|energy|memory[:required_release_bytes]")
    parser.add_argument("--sweep-host-columns", default="13056,8704,4352,0", help="sweep arm: comma-separated host column counts")
    parser.add_argument("--no-dormant", action="store_true", help="split without releasing the host share")
    parser.add_argument("--output-tokens", type=int, default=None)
    parser.add_argument("--prompt-chars", type=int, default=0, help="truncate the document to this many characters (0 = whole)")
    parser.add_argument("--prompt-tokens-list", default=None, help="sweep arm: comma-separated exact prompt lengths in tokens (document prefix + task suffix)")
    parser.add_argument("--wait-for-rig-seconds", type=int, default=0)
    options = parser.parse_args()
    if not 0 <= options.wait_for_rig_seconds <= 3600:
        parser.error("rig wait must be between 0 and 3600 seconds")
    config = json.loads(options.config.read_text())
    with Path(config["execution_lock"]).open("a") as lock:
        deadline = time.monotonic() + options.wait_for_rig_seconds
        announced = False
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise
                if not announced:
                    print("WAITING_FOR_RIG", options.wait_for_rig_seconds, flush=True)
                    announced = True
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        if options.arm == "pair":
            options.output.mkdir(parents=False, exist_ok=False)
            run(config, options.output / "combined", "combined", options)
            run(config, options.output / "control", "control", options)
        elif options.arm == "admission":
            run_admission(config, options.output, options)
        else:
            run(config, options.output, options.arm, options)


if __name__ == "__main__":
    main()
