"""Decode-only weight relocation gate: dormant host share of the FFN columns the phone executes.

Three arms on the calibrated CUDA desktop parent (23 GPU layers) with one HTP session (the HTP0
shard, layers 0-7 at full width) over FunctionFS:

* plain    - desktop only.
* split    - assisted decode split at --phone-percent of the FFN columns (control at the first
             token), weights fully resident on the desktop.
* dormant  - the same split with S41_SERVER_FFN_DORMANT_HOST_SHARE=1: the server releases the
             pages of the phone-executed column suffix while every slot decodes and populates
             them before the next prompt. Requests run back to back so the restore cost lands in
             the following prefill. A final owner-loss request kills the phone worker mid-decode;
             the request must fail visibly and the next local request must complete.

Kernel accounting is the proof: /proc/<pid>/pagemap residency of the exact share ranges during
decode and prefill, /proc/<pid>/smaps RSS of the model file, MemAvailable, and the server's
S41SERVERFFN dormant_host_share lines. One session of three, not a scheduler qualification.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

SERIAL = "3C15AU002CL00000"
VENDOR, PRODUCT = 0x18d1, 0x2d00
PAGE = os.sysconf("SC_PAGE_SIZE")


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=1)
        stream.write("\n")


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def layer_spec(mask):
    indices = [i for i in range(64) if mask >> i & 1]
    spans, first, previous = [], indices[0], indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        spans.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = index
    spans.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(spans), indices


def inward(first, last):
    first = (first + PAGE - 1) // PAGE * PAGE
    last = last // PAGE * PAGE
    return (first, last) if last > first else None


def share_ranges(gguf_dir, model_path, layer_indices, host_columns, n_embd, n_ff):
    """Page-inward file ranges of the phone share, mirroring llama_model::ffn_host_share_release."""
    sys.path.insert(0, str(gguf_dir))
    import gguf  # noqa: PLC0415

    reader = gguf.GGUFReader(str(model_path))
    ranges, tensors = [], []
    for tensor in reader.tensors:
        name = tensor.name
        if not name.startswith("blk."):
            continue
        layer = int(name.split(".")[1])
        if layer not in layer_indices:
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
            tensors.append({"name": name, "offs": offs, "nbytes": n_ff * n_embd * element})
        elif name.endswith(".ffn_down.weight"):
            assert int(tensor.shape[0]) == n_ff and int(tensor.shape[1]) == n_embd, (name, tensor.shape)
            row = n_ff * element
            for r in range(n_embd):
                piece = inward(offs + r * row + host_columns * element, offs + (r + 1) * row)
                if piece:
                    ranges.append(piece)
            tensors.append({"name": name, "offs": offs, "nbytes": n_embd * row})
    return ranges, tensors


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
                # little-endian 64-bit entries: the present bit (63) is the top bit of byte 7
                resident += sum(1 for i in range(7, len(data), 8) if data[i] & 0x80)
                total += len(data) // 8
    return {"resident_pages": resident, "total_pages": total, "vma_count": len(mappings),
            "resident_fraction": (resident / total) if total else None}


def model_file_rss(pid, real_model):
    total = 0
    in_model = False
    for line in Path(f"/proc/{pid}/smaps").read_text().splitlines():
        if "-" in line.split(" ", 1)[0] and len(line.split()) >= 5:
            parts = line.split(maxsplit=5)
            in_model = len(parts) == 6 and parts[5].strip() == real_model
            continue
        if in_model and line.startswith("Rss:"):
            total += int(line.split()[1]) * 1024
    return total


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
        for name in ("memory.current", "memory.max", "memory.swap.max"):
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


def proof_lines(path):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if "S41SERVERFFN dormant_host_share" in line:
            fields = {}
            for word in line.split("S41SERVERFFN dormant_host_share", 1)[1].split():
                key, _, value = word.partition("=")
                fields[key] = int(value) if value.isdigit() else value
            fields["line"] = line.strip()
            rows.append(fields)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--gguf-py", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--shard-remote-dir", required=True)
    parser.add_argument("--session-id", default="HTP0")
    parser.add_argument("--worker", required=True)
    parser.add_argument("--usb-close", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phone-percent", type=int, default=75)
    parser.add_argument("--requests-per-arm", type=int, default=2)
    parser.add_argument("--dormant-requests", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=96)
    parser.add_argument("--gpu-layers", type=int, default=23)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--column-quantum", type=int, default=1280)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--owner-timeout-ms", type=int, default=120000)
    parser.add_argument("--recovery-kill-delay-s", type=float, default=8.0)
    parser.add_argument("--skip-recovery", action="store_true")
    parser.add_argument("--kernel-release", default="6.12.23-android16-5-o-g227664cbe007-4k")
    parser.add_argument("--sessions", type=int, choices=(1, 3), default=1,
                        help="1: HTP0 only through the direct worker; 3: HTP0/1/2 (layers 0-23) through the canonical session controller")
    parser.add_argument("--phone-model", default="/data/local/tmp/s41-opoffload-dmabuf-v1/gemma-4-12B-Q40-dequant-f16.gguf")
    parser.add_argument("--resident-workers", default="/data/local/tmp/s42-per-session-correctness-20260903-v2-bin/llama-ffn-split-resident-workers")
    parser.add_argument("--resident-router", default="/data/local/tmp/s42-ready-subset-router-20260905-v2/llama-ffn-split-resident-router")
    parser.add_argument("--hash-cache", type=Path, default=None)
    parser.add_argument("--prompt-repeat", type=int, default=1, help="repeat the document tokens to build a longer prompt")
    parser.add_argument("--arms", default="plain,split,dormant", help="comma-separated subset of plain,split,dormant")
    parser.add_argument("--dry-run-command", action="store_true", help="build the canonical session command, print it, exit")
    parser.add_argument("--drop-model-cache", action="store_true",
                        help="evict the model file from the page cache before every desktop launch so a memory cgroup on this process is charged for the pages it faults in")
    args = parser.parse_args()
    arms = tuple(args.arms.split(","))
    assert all(arm in ("plain", "split", "dormant") for arm in arms) and arms
    if not args.dry_run_command:
        args.output.mkdir()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.repo))
    from research_dev.scheduler import ModelManifest
    from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeControl, AdaptiveDecodePolicy
    from research_dev.scheduler._internal.types import canonical_sha256
    from research_dev.scheduler.adapters import (
        HostEnergySampler, LlamaCppHttpClient, LlamaServerLaunchContract, LlamaServerProcessConfiguration,
        LlamaServerProcessLauncher, PhysicalParticipantCommand, PhysicalTransitionCommand,
        default_host_metric_callbacks, server_energy_summary,
    )
    from research_dev.scheduler._internal.runtime_plan import RuntimeExecutionContract, RuntimePhoneShard
    from research_dev.scheduler._internal.plan_contracts.transitions import RuntimeTransitionPlan
    from research_dev.scheduler.adapters.android_llama_server import (
        AndroidLlamaServerProcessConfiguration, AndroidLlamaServerProcessLauncher,
    )
    from research_dev.scheduler.adapters.bridge import probe_functionfs_usb_device, verify_android_usb_restored
    from research_dev.scheduler.adapters.ffn_shards import FfnShardIndex
    from research_dev.scheduler.adapters.http_backend import LlamaCppCompletionPayload
    from research_dev.scheduler.adapters.llama_server import PhoneFfnExecutionContract
    from research_dev.scheduler.adapters.phone_session import DirectPhoneFfnSession, DirectPhoneFfnSessionConfiguration
    from research_dev.scheduler.adapters.phone_transport import PhoneTransportContract
    from research_dev.scheduler.adapters.probes import parse_android_process_identity, probe_phone_runtime

    manifest = ModelManifest.from_json(json.loads(args.manifest.read_text()))
    artifact = manifest.artifact_sha256
    n_ff, n_embd = manifest.feed_forward_length, manifest.embedding_length
    phone_columns = n_ff * args.phone_percent // 100
    host_columns = n_ff - phone_columns
    assert 0 < phone_columns <= n_ff and phone_columns % args.column_quantum == 0
    index = FfnShardIndex.load(args.index, args.shard_remote_dir)
    assert index.parent_sha256 == artifact
    session_ids = ("HTP0", "HTP1", "HTP2") if args.sessions == 3 else (args.session_id,)
    session_masks = {"HTP0": 0xff, "HTP1": 0xff00, "HTP2": 0xff0000}
    shard_by_session = {}
    for session_id in session_ids:
        record = index.resolve(artifact, session_masks[session_id], n_ff, session_id=session_id)
        assert record is not None and record.columns == n_ff, session_id
        shard_by_session[session_id] = record
    shard = shard_by_session[session_ids[0]]
    mask = sum(record.layer_mask for record in shard_by_session.values())
    layers, layer_indices = layer_spec(mask)
    requests = json.loads(args.requests.read_text())
    document_tokens = tuple(int(v) for v in requests["rows"][0]["prompt_tokens"])
    prompt_tokens = document_tokens * max(1, args.prompt_repeat)
    limit = args.context - args.output_tokens - 16
    if len(prompt_tokens) > limit:
        prompt_tokens = prompt_tokens[:limit]
    alias = manifest.model_id
    activation = "geglu" if manifest.architecture == "gemma4" else "swiglu"
    payload_bytes = n_embd * 2 * args.max_tokens
    real_model = str(Path(os.path.realpath(args.model)))
    if args.dry_run_command:
        ranges, share_tensors = [], []
    else:
        ranges, share_tensors = share_ranges(args.gguf_py, args.model, set(layer_indices), host_columns, n_embd, n_ff)
    expected_release = sum(last - first for first, last in ranges)
    spec = {
        "scope": "decode-only relocation gate; dormant host share; not a scheduler qualification",
        "cgroup_at_start": cgroup_memory(),
        "argv": sys.argv, "artifact_sha256": artifact, "index_sha256": index.index_sha256,
        "shard": {"session_id": shard.session_hint, "remote_path": shard.remote_path, "layer_mask": mask,
                  "layers": layers, "columns": shard.columns, "shard_bytes": shard.shard_bytes,
                  "shard_sha256": shard.shard_sha256},
        "sessions": {session_id: {"remote_path": record.remote_path, "layer_mask": record.layer_mask,
                                  "shard_bytes": record.shard_bytes, "shard_sha256": record.shard_sha256}
                     for session_id, record in shard_by_session.items()},
        "arms_requested": arms, "prompt_repeat": args.prompt_repeat,
        "split": {"phone_percent": args.phone_percent, "phone_columns": phone_columns, "host_columns": host_columns,
                  "expected_release_bytes": expected_release, "share_range_count": len(ranges),
                  "share_tensor_bytes": sum(t["nbytes"] for t in share_tensors)},
        "prompt": {"input_tokens": len(prompt_tokens), "document_tokens": len(document_tokens),
                   "prompt_sha256": requests.get("prompt_sha256"), "output_tokens": args.output_tokens},
        "desktop": {"gpu_layers": args.gpu_layers, "context": args.context, "batch": args.batch, "ubatch": args.ubatch,
                    "server_sha256": digest(args.server),
                    "libraries": {p.name: digest(p) for p in sorted(args.server.parent.glob("*.so*")) if p.is_file() and not p.is_symlink()}},
        "phone": {"worker": args.worker, "max_tokens": args.max_tokens, "column_quantum": args.column_quantum,
                  "transport": "functionfs-dmabuf-async-ring-v2", "queue_depth": 4, "payload_bytes": payload_bytes},
        "arms": {"plain": args.requests_per_arm, "split": args.requests_per_arm, "dormant": args.dormant_requests,
                 "recovery": not args.skip_recovery},
    }
    if not args.dry_run_command:
        save(args.output / "SPEC.json", spec)
    remote_root = "/data/local/tmp/" + args.output.parent.name + "-" + args.output.name
    config = AndroidLlamaServerProcessConfiguration(
        adb_path=Path("/usr/bin/adb"), serial=SERIAL, adb_port=5037,
        remote_server_path=args.worker, remote_library_directory=str(Path(args.worker).parent),
        remote_model_paths_by_artifact={artifact: shard.remote_path},
        remote_state_directory=remote_root, executable_device_name="HTP0", output_directory=args.output,
    )
    android = AndroidLlamaServerProcessLauncher(config)
    usb_transport = PhoneTransportContract(
        transport="functionfs-usb", allocator="devmem", queue_depth=4, concurrent_streams=4,
        max_payload_bytes=payload_bytes, full_duplex=True, split_h2d=False,
        generation="functionfs-dmabuf-async-ring-v2", profile_id="native-gemma-decode-only-relocation-gate",
        usbfs_available_bytes=int(Path("/sys/module/usbcore/parameters/usbfs_memory_mb").read_text()) * 1024**2,
        slot_safety_bytes=65536, vendor_id=VENDOR, product_id=PRODUCT,
        control_host="127.0.0.1", control_port=0, batch_plan="split-row",
    )
    direct = DirectPhoneFfnSession(DirectPhoneFfnSessionConfiguration(
        adb_path=config.adb_path, usb_close_path=args.usb_close, serial=SERIAL, adb_port=5037,
        session_script="/data/local/tmp/s42-hal-runtime-probe-20260906-v4/direct_phone_ffn_session.sh",
        restore_script="/data/local/tmp/s42-unified-direct-v1/restore_android_usb.sh", session_root=remote_root,
        worker_paths_by_artifact={artifact: args.worker}, model_paths_by_artifact={artifact: shard.remote_path},
        backend_by_device={"op15-phone": "HTP0"}, minimum_usb_speed_mbps=5000,
        required_kernel_release=args.kernel_release, launch_timeout_s=60, session_timeout_s=7200,
        diagnostic_port=18383, diagnostic_host="192.168.42.1", busybox_path="/data/adb/magisk/busybox",
        network_manager_path=Path("/usr/bin/nmcli"), android_gadget_path="/config/usb_gadget/g1",
        functionfs_gadget_path="/config/usb_gadget/g2", functionfs_root_path="/dev/usb-ffs/s41",
        phone_usb_controller="a600000.dwc3",
    ))
    cfg = direct.configuration
    canonical = None
    if args.sessions == 3:
        canonical = DirectPhoneFfnSession(DirectPhoneFfnSessionConfiguration(
            adb_path=config.adb_path, usb_close_path=args.usb_close, serial=SERIAL, adb_port=5037,
            session_script=cfg.session_script, restore_script=cfg.restore_script, session_root=remote_root + "-canonical",
            worker_paths_by_artifact={artifact: args.worker}, model_paths_by_artifact={artifact: args.phone_model},
            ffn_shards_by_artifact={artifact: index}, backend_by_device={"op15-phone": "HTP0"},
            minimum_usb_speed_mbps=5000, required_kernel_release=args.kernel_release, launch_timeout_s=120,
            session_timeout_s=7200, diagnostic_port=18383, diagnostic_host="192.168.42.1",
            busybox_path="/data/adb/magisk/busybox", network_manager_path=Path("/usr/bin/nmcli"),
            android_gadget_path="/config/usb_gadget/g1", functionfs_gadget_path="/config/usb_gadget/g2",
            functionfs_root_path="/dev/usb-ffs/s41", phone_usb_controller="a600000.dwc3",
            remote_hash_cache_path=args.hash_cache, resident_workers_path=args.resident_workers,
            resident_router_path=args.resident_router, multi_session_port_base=26760, multi_session_device_count=3,
        ))

    def canonical_command():
        """Hand-built transition command: three resident sessions at full width for layers 0-23."""
        rows = [{"operator_id": f"layer:{il}:ffn", "operator_kind": "ffn", "device_ids": ["desktop-cpu", "op15-phone"],
                 "split_axis": "none", "split_fraction_ppm": 0} for il in layer_indices]
        route_id = f"decode-only-relocation:{args.phone_percent}:sessions-3"
        plan_sha256 = canonical_sha256({"route_id": route_id, "operators": rows, "sessions": sorted(session_ids)})
        shards = tuple(RuntimePhoneShard(
            session_id=session_id, endpoint=f"session://op15-phone/{session_id}", layer_mask=record.layer_mask,
            maximum_columns=n_ff, resident_bytes=record.shard_bytes,
            resident_geometry_sha256=canonical_sha256({"artifact": artifact, "layer_mask": record.layer_mask,
                                                       "columns": n_ff, "dtype": "f16", "shard_sha256": record.shard_sha256}),
            operator_plan_sha256=plan_sha256, artifact_sha256=artifact, session_generation=1,
        ) for session_id, record in shard_by_session.items())
        contract = RuntimeExecutionContract(
            execution_mode="adaptive-split", initial_split_fraction_ppm=0,
            allowed_adaptive_fractions_ppm=(0, args.phone_percent * 10000), batch_plan="split-row",
            maximum_batch_size=args.max_tokens, queue_depth=usb_transport.queue_depth, phone_device_id="op15-phone",
            phone_endpoint="session://op15-phone", operator_kind="ffn", phone_shards=shards,
        )
        parameters = {
            "batch_size": args.batch, "context_size": args.context, "cpu_device_id": "desktop-cpu",
            "ffn_activation": activation, "ffn_assistance_phase": "decode", "ffn_column_quantum": args.column_quantum,
            "ffn_max_tokens": args.max_tokens, "ffn_n_embd": n_embd, "ffn_runtime_control_protocol": "decode-boundary-v1",
            "ffn_timeout_ms": args.owner_timeout_ms, "ffn_transport": "functionfs-usb", "gpu_device_id": "desktop-cuda",
            "gpu_layers": args.gpu_layers, "model_alias": alias, "parallel": 1, "phone_device_id": "op15-phone",
            "ubatch_size": args.ubatch, "usb_allocator": usb_transport.allocator, "usb_batch_plan": usb_transport.batch_plan,
            "usb_concurrent_streams": usb_transport.concurrent_streams, "usb_full_duplex": int(usb_transport.full_duplex),
            "usb_max_payload_bytes": usb_transport.max_payload_bytes, "usb_product_id": usb_transport.product_id,
            "usb_queue_depth": usb_transport.queue_depth, "usb_slot_safety_bytes": usb_transport.slot_safety_bytes,
            "usb_split_h2d": int(usb_transport.split_h2d), "usb_transport_generation": usb_transport.generation,
            "usb_transport_profile_id": usb_transport.profile_id, "usb_vendor_id": usb_transport.vendor_id,
            "usbfs_available_bytes": usb_transport.usbfs_available_bytes,
        }
        plan = {"route_id": route_id, "plan_sha256": plan_sha256, "operators": rows, "assisted_operator_kind": "ffn",
                "execution_contract": contract.to_json()}
        transition = RuntimeTransitionPlan(
            transition_id="decode-only-relocation:preload", device_id="op15-phone", source_state="cold",
            target_state="hot", latency_us=0, energy_uj=0, resource_ids=("op15-htp",), maturity="QUALIFIED",
            phone_shards=shards,
        )
        return PhysicalTransitionCommand(
            ticket_id="decode-only-relocation:preload:0", request_id="decode-only-relocation-preload",
            artifact_sha256=artifact, route_id=route_id, operator_plan_sha256=plan_sha256,
            participant=PhysicalParticipantCommand(executor_id="physical:op15-phone", device_id="op15-phone",
                                                   endpoint="session://op15-phone", backend="hexagon-htp",
                                                   resource_ids=("op15-htp",)),
            transition=transition, execution_contract=contract, adapter_parameters=parameters,
            phone_layout_generation=1, operator_plan_protocol="llama-server-http-v1", operator_plan=plan,
        )

    if args.dry_run_command:
        command = canonical_command() if args.sessions == 3 else None
        print(json.dumps({"sessions": args.sessions, "layer_mask": mask, "layers": layers,
                          "expected_release_bytes": expected_release, "share_range_count": len(ranges),
                          "prompt_tokens": len(prompt_tokens),
                          "command": None if command is None else command.to_json()}, indent=1, sort_keys=True))
        return

    sampler = HostEnergySampler(default_host_metric_callbacks())
    client = LlamaCppHttpClient()
    stop = threading.Event()
    commands, runs, intervals, latest, residency_samples, events = [], [], [], [], [], []
    managed = None
    worker_process = None
    identity = None
    session_root = remote_root
    diagnostic_endpoint = "http://192.168.42.1:18383"
    direct_started = False
    generation = 0
    current_label = {"arm": None, "request": None}

    def shell(command, required=True):
        result = android._su(command, timeout_s=30, check=False)
        commands.append({"command": command, "time_ns": time.monotonic_ns(), "returncode": result.returncode,
                         "stdout": result.stdout[-4000:], "stderr": result.stderr[-2000:]})
        if required and result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        return result.stdout

    def check():
        if managed is not None and managed.process is not None and managed.process.poll() is not None:
            raise RuntimeError("desktop server exited")
        if latest and latest[-1].value is not None:
            value = latest[-1].value
            if not value.thermal_qualified or value.temperature_millic >= 90000 or value.battery_ppm < 50000:
                raise RuntimeError("phone thermal/battery check failed")

    def observe_phone():
        with (args.output / "PHONE_HEALTH.jsonl").open("x") as stream:
            while not stop.is_set():
                try:
                    observation = probe_phone_runtime(diagnostic_endpoint, diagnostic=True)
                    latest[:] = [observation]
                    entry = observation.to_json()
                    entry["values"] = None if observation.value is None else asdict(observation.value)
                    stream.write(json.dumps(entry) + "\n")
                    stream.flush()
                except Exception as error:  # noqa: BLE001 - recorded
                    stream.write(json.dumps({"error": repr(error), "time_ns": time.monotonic_ns()}) + "\n")
                stop.wait(2)

    def observe_server_memory():
        with (args.output / "SERVER_MEMORY.jsonl").open("x") as stream:
            while not stop.is_set():
                server = managed
                if server is not None and server.process is not None and server.process.poll() is None:
                    pid = server.process.pid
                    try:
                        status = Path(f"/proc/{pid}/status").read_text()
                        fields = {line.split(":", 1)[0]: int(line.split()[1]) * 1024 for line in status.splitlines()
                                  if line.startswith(("VmRSS:", "VmHWM:", "RssFile:", "RssAnon:"))}
                        stream.write(json.dumps({"observed_ns": time.monotonic_ns(), "pid": pid, "bytes": fields,
                                                 "model_file_rss": model_file_rss(pid, real_model), "meminfo": meminfo(),
                                                 "cgroup": cgroup_memory(),
                                                 "arm": current_label["arm"], "request": current_label["request"]}) + "\n")
                        stream.flush()
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                stop.wait(0.5)

    def sample_residency(tag):
        server = managed
        if server is None or server.process is None or server.process.poll() is not None:
            return None
        pid = server.process.pid
        started = time.monotonic_ns()
        row = {"tag": tag, "observed_ns": started, "arm": current_label["arm"], "request": current_label["request"],
               **residency(pid, real_model, ranges), "model_file_rss": model_file_rss(pid, real_model),
               "meminfo": meminfo(), "scan_us": (time.monotonic_ns() - started) // 1000}
        residency_samples.append(row)
        return row

    launcher = LlamaServerProcessLauncher(LlamaServerProcessConfiguration(
        server_path=args.server, model_paths_by_artifact={artifact: args.model},
        library_paths_by_device={"desktop-cuda": (args.cuda_lib_dir,)}, executable_device_names={"desktop-cuda": "CUDA0"},
        output_directory=args.output, common_library_paths=(args.server.parent, args.cuda_lib_dir),
    ))

    def ffn_environment(dormant):
        environment = {
            "S41_SERVER_FFN_ACTIVATION": activation, "S41_SERVER_FFN_ARTIFACT_SHA256": artifact,
            "S41_SERVER_FFN_COLUMNS": str(n_ff), "S41_SERVER_FFN_F16_IO": "1",
            "S41_SERVER_FFN_LAYER_MASK": str(mask), "S41_SERVER_FFN_MAX_TOKENS": str(args.max_tokens),
            "S41_SERVER_FFN_N_EMBD": str(n_embd), "S41_SERVER_FFN_RUNTIME_CONTROL": "1",
            "S41_SERVER_FFN_TIMEOUT_MS": str(args.owner_timeout_ms),
        }
        if dormant:
            environment["S41_SERVER_FFN_DORMANT_HOST_SHARE"] = "1"
        if args.sessions == 3:
            environment["S41_SERVER_FFN_SHARDS"] = ";".join(
                f"{session_id}@session://op15-phone/{session_id}:{record.layer_mask}"
                for session_id, record in shard_by_session.items())
        environment.update(usb_transport.server_environment())
        return environment

    def drop_model_cache(label):
        """POSIX_FADV_DONTNEED over the whole model file; only clean, unmapped pages are dropped."""
        before = meminfo()
        started = time.monotonic_ns()
        fd = os.open(args.model, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        after = meminfo()
        events.append({"kind": "model_cache_drop", "label": label, "elapsed_us": (time.monotonic_ns() - started) // 1000,
                       "cached_before": before.get("Cached"), "cached_after": after.get("Cached"),
                       "cgroup": cgroup_memory()})
        print("MODEL_CACHE_DROP", label, "Cached", before.get("Cached"), "->", after.get("Cached"), flush=True)

    def launch_desktop(label, with_phone, dormant=False):
        nonlocal managed
        if args.drop_model_cache:
            drop_model_cache(label)
        with socket.socket() as probe_socket:
            probe_socket.bind(("127.0.0.1", 0))
            port = probe_socket.getsockname()[1]
        contract = LlamaServerLaunchContract(
            model_alias=alias, context_size=args.context, parallel=1, batch_size=args.batch, ubatch_size=args.ubatch,
            gpu_layers=args.gpu_layers, cpu_device_id="desktop-cpu", gpu_device_id="desktop-cuda",
            phone_device_id="op15-phone" if with_phone else None,
            ffn_environment=ffn_environment(dormant) if with_phone else {},
            cuda_graph_mode="default", desktop_launch_mode="runtime-defaults",
        )
        endpoint = f"http://127.0.0.1:{port}"
        started = time.monotonic_ns()
        managed = launcher.launch_contract(endpoint, contract, manifest, label=label, control_check=check)
        ready = time.monotonic_ns()
        intervals.append({"kind": "desktop_load_" + label, "start_ns": started, "end_ns": ready, "with_phone": with_phone,
                          "dormant": dormant, "ffn_environment": dict(contract.ffn_environment),
                          "stderr": str(args.output / (label + ".stderr"))})
        return endpoint

    def run_request(endpoint, arm, split, index, *, kill_worker_after_s=None, sample_after_first_token_s=(12.0,)):
        request_id = f"{arm}-{index:02d}"
        current_label.update(arm=arm, request=request_id)
        first, controls, samples, kill, mid_stats = [], [], [], {}, []

        def first_token(observed_ns):
            nonlocal generation
            first.append(observed_ns)
            if not split:
                return
            generation += 1
            desktop = {"artifact": artifact, "gpu_layers": args.gpu_layers, "context": args.context,
                       "batch": args.batch, "ubatch": args.ubatch, "server_sha256": digest(args.server)}
            plan = {"desktop": desktop, "layers": layer_indices, "columns": phone_columns, "shard_sha256": shard.shard_sha256,
                    "session_id": shard.session_hint, "activation": activation, "io": "f16",
                    "transport": "functionfs", "column_quantum": args.column_quantum}
            policy = AdaptiveDecodePolicy(
                route_id=f"decode-only-relocation-{args.phone_percent}", executor_id="native-calibration",
                operator_plan_sha256=canonical_sha256(plan), desktop_parent_route_id="calibration-cuda-23",
                desktop_placement_sha256=canonical_sha256(desktop), layer_indices=tuple(layer_indices),
                layer_mask=mask, columns=phone_columns, split_fraction_ppm=args.phone_percent * 10000,
                resource_ids=("desktop-cpu", "desktop-cuda", "op15-htp", "op15-functionfs"),
            )
            control = AdaptiveDecodeControl(request_id, 0, generation, policy)
            issued = time.monotonic_ns()
            # the server answers controls from its main loop; under a memory cap a decode step can take seconds
            ack, received = LlamaCppHttpClient.apply_ffn_control(endpoint, control, timeout_s=180.0)
            controls.append({"issued_ns": issued, "received_ns": received, "control": control.to_json(), "ack": ack})

        def side_effects():
            # residency samples during decode (after the first token) and, optionally, owner loss
            while not first and not done.is_set():
                done.wait(0.1)
            if done.is_set():
                return
            for delay in sample_after_first_token_s:
                if done.wait(delay):
                    return
                row = sample_residency(f"{request_id}:decode+{delay:.0f}s")
                if row is not None:
                    samples.append(row)
                if split and not mid_stats:
                    # the FFN stats endpoint answers only while the request's slot is active
                    try:
                        stats, _received = LlamaCppHttpClient.read_ffn_stats(endpoint, request_id, 0, timeout_s=180.0)
                        mid_stats.append(stats)
                    except Exception as error:  # noqa: BLE001 - recorded
                        mid_stats.append({"error": repr(error)})
            if kill_worker_after_s is not None:
                if done.wait(kill_worker_after_s):
                    return
                root = canonical._remote_root if args.sessions == 3 else session_root
                pid_name = "resident-workers.pid" if args.sessions == 3 else "worker.pid"
                process_name = "llama-ffn-split-resident-workers" if args.sessions == 3 else "llama-ffn-split-worker"
                pid = ""
                for _attempt in range(3):
                    pid = shell("cat " + root + "/" + pid_name + " 2>/dev/null", False).strip()
                    if pid.isdecimal():
                        break
                    time.sleep(0.5)
                if not pid.isdecimal():
                    # fall back to the live process list: the owner is the only worker on the phone
                    listed = shell("pidof " + process_name + " 2>/dev/null || ps -A -o PID,NAME | awk '$2 == \"" + process_name + "\" {print $1}'", False).split()
                    pid = listed[0] if listed and listed[0].isdecimal() else ""
                    kill["pid_source"] = "process list"
                else:
                    kill["pid_source"] = "pid file"
                kill["issued_ns"] = time.monotonic_ns()
                kill["worker_pid"] = pid
                kill["pid_file"] = root + "/" + pid_name
                kill["result"] = shell(f"kill -9 {pid}", False) if pid.isdecimal() else "no worker pid"
                kill["process_list_after"] = shell("ps -A -o PID,NAME | grep -i ffn", False)[-800:]

        done = threading.Event()
        prefill_sampler = threading.Thread(target=lambda: (time.sleep(6.0), None if done.is_set() else samples.append(
            sample_residency(f"{request_id}:prefill+6s"))), daemon=True)
        side = threading.Thread(target=side_effects, daemon=True)
        payload = LlamaCppCompletionPayload(
            request_id=request_id, expected_model_alias=alias, input_tokens=len(prompt_tokens),
            output_tokens=args.output_tokens, prompt_tokens=prompt_tokens, seed=42, quality_mode="semantic",
            timeout_s=1800, stream_path=args.output / f"{request_id}.raw", on_first_token=first_token,
        )
        check()
        meminfo_before = meminfo()
        started = time.monotonic_ns()
        prefill_sampler.start()
        side.start()
        error = None
        response = {}
        try:
            response = client.complete(endpoint, payload, lambda: None)
        except Exception as failure:  # noqa: BLE001 - recorded, the gate decides
            error = f"{type(failure).__name__}: {failure}"
        finished = time.monotonic_ns()
        done.set()
        side.join(timeout=30)
        prefill_sampler.join(timeout=30)
        tokens = list(response.get("tokens", []))
        predicted_ms = float(response.get("predicted_ms") or 0.0)
        prompt_ms = float(response.get("prompt_ms") or 0.0)
        entry = {"arm": arm, "request_id": request_id, "index": index, "split": split,
                 "started_ns": started, "finished_ns": finished, "first_token_ns": first[0] if first else None,
                 "wall_s": (finished - started) / 1e9, "prompt_ms": prompt_ms, "predicted_ms": predicted_ms,
                 "output_tokens": len(tokens), "decode_ms_per_token": predicted_ms / len(tokens) if tokens else None,
                 "tokens": tokens, "controls": controls, "error": error, "residency_samples": [s for s in samples if s],
                 "phone_stats_mid_decode": mid_stats[0] if mid_stats else None, "kill": kill, "meminfo_before": meminfo_before, "meminfo_after": meminfo(),
                 "result_keys": sorted(response.keys())}
        save(args.output / f"EXECUTION-{request_id}.json", {**entry, "execution": {k: v for k, v in response.items() if k != "tokens"}})
        runs.append(entry)
        print("COMPLETED", request_id, f"wall={entry['wall_s']:.2f}s decode_ms/tok={entry['decode_ms_per_token']} "
              f"prompt_ms={prompt_ms:.0f} error={error}", flush=True)
        current_label.update(request=None)
        return entry

    def start_phone_session(tag):
        """Launch one direct FunctionFS worker session; the worker completes when its host disconnects."""
        nonlocal worker_process, identity, direct_started, session_root
        session_root = remote_root + "-" + tag
        phone_live = shell("ps -A -o PID,NAME; getprop sys.usb.config; uname -r")
        assert not any(line.split()[-1] in ("llama-server", "llama-ffn-split-worker", "llama-ffn-split-resident-workers",
                                            "llama-ffn-split-resident-router") for line in phone_live.splitlines() if line.split()), "phone worker already running"
        assert shell("uname -r").strip() == cfg.required_kernel_release, "phone kernel is not the qualified candidate"
        assert not shell("cat " + cfg.functionfs_gadget_path + "/UDC").strip(), "FunctionFS gadget already bound"
        shell("mkdir " + shlex.quote(session_root))
        worker_environment = {
            "LD_LIBRARY_PATH": config.remote_library_directory, "ADSP_LIBRARY_PATH": config.remote_library_directory,
            "GGML_HEXAGON_NDEV": "1", "GGML_HEXAGON_NHVX": "4", "GGML_HEXAGON_MBUF": "4192", "GGML_HEXAGON_VMEM": "3328",
            "S41_DISABLE_GRAPH_CACHE": "1", "S42_RESIDENCY_SESSION_ID": shard.session_hint,
            "S42_RESIDENCY_SESSION_GENERATION": "1",
            "SCHEDULER_ANDROID_GADGET": cfg.android_gadget_path, "SCHEDULER_FUNCTIONFS_GADGET": cfg.functionfs_gadget_path,
            "SCHEDULER_FUNCTIONFS_ROOT": cfg.functionfs_root_path, "SCHEDULER_PHONE_UDC": cfg.phone_usb_controller,
            "S42_USB_NCM": "1", "S42_DIAGNOSTIC_PORT": str(cfg.diagnostic_port), "S42_BUSYBOX": cfg.busybox_path,
            "S41_FFN_F16_IO": "1", "S41_FFN_STAGED_DMABUF": "0", "S41_FFN_MAX_TOKENS": str(args.max_tokens),
            "S41_FFN_COLUMN_QUANTUM": str(args.column_quantum), "S41_FFN_QUEUE_DEPTH": str(usb_transport.queue_depth),
        }
        arguments = ["sh", cfg.session_script, args.worker, shard.remote_path, layers, str(n_ff), "HTP0",
                     session_root, cfg.restore_script, str(cfg.session_timeout_s), "0", artifact]
        body = ("nohup env " + " ".join(shlex.quote(k + "=" + v) for k, v in worker_environment.items())
                + " " + shlex.join(arguments) + " > " + session_root + "/session-launch.log 2>&1 < /dev/null &")
        save(args.output / f"WORKER_LAUNCH-{tag}.json", {"arguments": arguments, "environment": worker_environment,
                                                          "session_root": session_root})
        started = time.monotonic_ns()
        worker_process = subprocess.Popen(android._adb_command(SERIAL, "shell", "su -c " + shlex.quote(body)),
                                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 180
        while True:
            try:
                usb = probe_functionfs_usb_device(vendor_id=f"{VENDOR:04x}", product_id=f"{PRODUCT:04x}")
            except Exception:  # noqa: BLE001 - not enumerated yet
                usb = None
            if usb is not None:
                assert usb.negotiated_speed_mbps >= 5000, usb
                direct._connect_diagnostic_ncm(usb)
                latest[:] = [probe_phone_runtime(diagnostic_endpoint, diagnostic=True)]
                direct_started = True
                residency_log = direct._read_diagnostic_file("residency.log")
                assert "descriptors_ready backend=Hexagon" in residency_log, residency_log[-2000:]
                save(args.output / f"USB-{tag}.json", vars(usb))
                break
            terminal = shell("cat " + session_root + "/terminal.status 2>/dev/null", False).strip()
            if terminal:
                raise RuntimeError("phone session terminated before USB enumeration: " + terminal + "\n"
                                   + shell("tail -40 " + session_root + "/session-launch.log " + session_root + "/worker.log", False))
            if time.monotonic() > deadline:
                raise TimeoutError("phone session not enumerated after 180 s\n" + shell("tail -40 " + session_root + "/worker.log", False))
            time.sleep(0.5)
        pid = shell("cat " + session_root + "/worker.pid", False).strip()
        identity = None
        if pid.isdecimal():
            identity = parse_android_process_identity(shell(android._process_identity_command(int(pid), session_root + "/worker.pid")), int(pid))
        ready = time.monotonic_ns()
        intervals.append({"kind": "phone_preload_" + tag, "start_ns": started, "end_ns": ready})
        save(args.output / f"READY-{tag}.json", {"started_ns": started, "ready_ns": ready, "session_root": session_root,
                                                "process_identity": None if identity is None else identity.to_json()})
        print("PHONE_READY", tag, (ready - started) / 1e9, flush=True)

    def start_phone_sessions_canonical(tag):
        """Three resident sessions through DirectPhoneFfnSession.start (resident workers + router)."""
        nonlocal direct_started, session_root
        session_root = canonical.configuration.session_root
        started = time.monotonic_ns()
        preflight = canonical.preflight()
        save(args.output / f"PREFLIGHT-{tag}.json", preflight.to_json())
        receipt = canonical.start(canonical_command(), manifest, usb_transport, None)
        ready = time.monotonic_ns()
        direct_started = True
        try:
            canonical._connect_diagnostic_ncm(receipt.usb)
        except Exception as error:  # noqa: BLE001 - diagnostics are best effort here
            events.append({"kind": "diagnostic_ncm_unavailable", "tag": tag, "error": repr(error)})
        try:
            latest[:] = [probe_phone_runtime(diagnostic_endpoint, diagnostic=True)]
        except Exception as error:  # noqa: BLE001
            events.append({"kind": "phone_probe_unavailable", "tag": tag, "error": repr(error)})
        intervals.append({"kind": "phone_preload_" + tag, "start_ns": started, "end_ns": ready, "sessions": 3})
        save(args.output / f"READY-{tag}.json", {"started_ns": started, "ready_ns": ready, "session_root": session_root,
                                                "remote_root": canonical._remote_root, "receipt": receipt.to_json()})
        print("PHONE_READY", tag, "sessions=3", (ready - started) / 1e9, flush=True)

    def finish_phone_sessions_canonical(tag, *, aborted):
        nonlocal direct_started
        try:
            if aborted:
                record = canonical.abort()
                payload = record.to_json() if hasattr(record, "to_json") else str(record)
            else:
                record = canonical.finish(require_execution=False)
                payload = record.to_json()
        except Exception as error:  # noqa: BLE001 - recorded; USB restoration is re-verified below
            payload = {"error": repr(error)}
        restoration = verify_android_usb_restored(serial=SERIAL, adb_port=5037, minimum_speed_mbps=5000, timeout_s=120)
        save(args.output / f"PHONE_LOG-{tag}.json", {"close": payload, "restoration": str(restoration)})
        direct_started = False
        events.append({"kind": "phone_sessions_finished", "tag": tag, "aborted": aborted, "time_ns": time.monotonic_ns()})
        print("PHONE_FINISHED", tag, "aborted" if aborted else "finished", flush=True)

    def phone_calls_of(stats):
        if not isinstance(stats, dict):
            return None
        runtime = stats.get("runtime_stats") if isinstance(stats.get("runtime_stats"), dict) else stats
        value = runtime.get("calls")
        return value if isinstance(value, int) else None

    def note_phone_calls(entry):
        calls = phone_calls_of(entry.get("phone_stats_mid_decode"))
        entry["phone_calls_mid_decode"] = calls
        if not calls:
            events.append({"kind": "phone_calls_not_observed", "request": entry["request_id"],
                           "stats": entry.get("phone_stats_mid_decode")})
            print("WARNING no phone calls observed mid-decode for", entry["request_id"], flush=True)

    def start_sessions(tag):
        if args.sessions == 3:
            start_phone_sessions_canonical(tag)
        else:
            start_phone_session(tag)

    def finish_sessions(tag, *, aborted=False):
        if args.sessions == 3:
            finish_phone_sessions_canonical(tag, aborted=aborted)
        else:
            finish_phone_session(tag)

    def finish_phone_session(tag):
        """After the host server detached (or the worker was killed): wait for completion and USB restoration."""
        nonlocal worker_process, identity, direct_started
        deadline = time.monotonic() + 120
        status = None
        while time.monotonic() < deadline:
            status = shell("cat " + session_root + "/terminal.status 2>/dev/null", False).strip()
            if status:
                break
            time.sleep(1.0)
        restoration = verify_android_usb_restored(serial=SERIAL, adb_port=5037, minimum_speed_mbps=5000, timeout_s=120)
        save(args.output / f"PHONE_LOG-{tag}.json", {"terminal_status": status, "worker_log": shell("cat " + session_root + "/worker.log", False)[-200000:],
                                                    "session_launch_log": shell("cat " + session_root + "/session-launch.log", False)[-20000:],
                                                    "restoration": restoration if isinstance(restoration, (dict, list, str, int, float, type(None))) else str(restoration)})
        if worker_process is not None:
            try:
                worker_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker_process.kill()
        worker_process = None
        identity = None
        direct_started = False
        events.append({"kind": "phone_session_finished", "tag": tag, "terminal_status": status, "time_ns": time.monotonic_ns()})
        print("PHONE_FINISHED", tag, status, flush=True)

    try:
        sampler.start()
        memory_observer = threading.Thread(target=observe_server_memory, daemon=True)
        memory_observer.start()
        live = subprocess.check_output(["ps", "-eo", "comm="], text=True).splitlines()
        assert not any(name.strip() in ("llama-server", "llama-cli") for name in live), "desktop server already running"
        measured = shell("sha256sum " + shlex.quote(shard.remote_path) + " " + shlex.quote(args.worker))
        assert measured.split()[0] == shard.shard_sha256.removeprefix("sha256:"), "phone shard digest differs"
        save(args.output / "PHONE_HASHES.json", measured)
        while len(sampler.rows()) < 2:
            time.sleep(0.1)
        phone_observer = threading.Thread(target=observe_phone, daemon=True)

        # ---- arm: plain (no phone) ----
        if "plain" in arms:
            endpoint = launch_desktop("plain", with_phone=False)
            for index in range(args.requests_per_arm):
                run_request(endpoint, "plain", False, index)
            managed.stop()
            managed = None
            time.sleep(1.0)

        # ---- arm: split, weights resident; one phone session (set) for this server ----
        if "split" in arms:
            start_sessions("split")
            if not phone_observer.is_alive():
                phone_observer.start()
            check()
            endpoint = launch_desktop("split", with_phone=True, dormant=False)
            for index in range(args.requests_per_arm):
                entry = run_request(endpoint, "split", True, index)
                note_phone_calls(entry)
            managed.stop()
            managed = None
            finish_sessions("split")
            time.sleep(1.0)

        if "dormant" not in arms:
            raise RuntimeError("the dormant arm is required for the result record")
        # ---- arm: dormant, back to back; a fresh phone session (set) for this server ----
        start_sessions("dormant")
        if not phone_observer.is_alive():
            phone_observer.start()
        check()
        endpoint = launch_desktop("dormant", with_phone=True, dormant=True)
        idle_before = sample_residency("dormant:idle-after-load")
        events.append({"kind": "idle_after_load", "sample": idle_before})
        for index in range(args.dormant_requests):
            entry = run_request(endpoint, "dormant", True, index)
            note_phone_calls(entry)
            time.sleep(1.5)
            sample_residency(f"dormant-{index:02d}:idle-after-request")
        # ---- recovery: owner loss mid-decode, then a local request on the same server ----
        recovery = None
        if not args.skip_recovery:
            victim = run_request(endpoint, "recovery-victim", True, 0, kill_worker_after_s=args.recovery_kill_delay_s,
                                 sample_after_first_token_s=(4.0,))
            time.sleep(2.0)
            after = sample_residency("recovery:after-owner-loss")
            follow = run_request(endpoint, "recovery-local", False, 1, sample_after_first_token_s=(8.0,))
            recovery = {"victim": {k: victim[k] for k in ("request_id", "error", "wall_s", "output_tokens", "kill")},
                        "after_owner_loss": after,
                        "local_follow_up": {k: follow[k] for k in ("request_id", "error", "wall_s", "output_tokens",
                                                                    "prompt_ms", "decode_ms_per_token")}}
        managed.stop()
        managed = None
        finish_sessions("dormant", aborted=not args.skip_recovery)
        time.sleep(0.5)
        for entry in runs:
            energy = dict(server_energy_summary(sampler.rows_between(entry["started_ns"], entry["finished_ns"]),
                                                entry["started_ns"], entry["finished_ns"]))
            energy["boundary"] = "request_service_including_local_prefill_and_decode"
            entry["energy"] = energy
        proofs = {}
        for interval in intervals:
            if interval.get("with_phone"):
                proofs[interval["kind"]] = proof_lines(interval["stderr"])
        save(args.output / "RESULT.json", {"status": "COMPLETED", "runs": runs, "intervals": intervals,
                                           "residency_samples": residency_samples, "proof_lines": proofs,
                                           "recovery": recovery, "events": events,
                                           "expected_release_bytes": expected_release, "share_range_count": len(ranges),
                                           "sessions": args.sessions, "arms": arms, "scheduler_qualified": False})
        print("RESULT written", flush=True)
    except BaseException as error:
        save(args.output / "FAILURE.json", {"error": repr(error), "traceback": traceback.format_exc(), "runs": runs,
                                            "residency_samples": residency_samples})
        raise
    finally:
        stop.set()
        cleanup_errors = []
        if managed is not None:
            try:
                managed.stop()
            except Exception as error:  # noqa: BLE001
                cleanup_errors.append("server stop: " + repr(error))
        try:
            if direct_started and canonical is not None:
                try:
                    canonical.abort()
                except Exception as error:  # noqa: BLE001
                    cleanup_errors.append("canonical abort: " + repr(error))
                direct_started = False
            if direct_started:
                for _attempt in range(30):
                    try:
                        probe_functionfs_usb_device(vendor_id=f"{VENDOR:04x}", product_id=f"{PRODUCT:04x}")
                    except Exception:  # noqa: BLE001
                        state = android._adb("get-state", check=False, timeout_s=2).stdout.strip()
                        if state == "device":
                            break
                        time.sleep(0.2)
                    else:
                        direct._close_direct_usb(PhoneFfnExecutionContract("op15-phone", n_embd, tuple(layer_indices), mask,
                                                                           n_ff, args.max_tokens, activation), usb_transport, artifact)
                        break
                verify_android_usb_restored(serial=SERIAL, adb_port=5037, minimum_speed_mbps=5000, timeout_s=90)
            if direct_started:
                save(args.output / "PHONE_LOG-cleanup.json", shell("cat " + session_root + "/worker.log", False)[-200000:])
            if identity is not None:
                try:
                    android.stop_remote(identity.process_id, session_root + "/worker.pid", expected=identity)
                except Exception as error:  # noqa: BLE001 - the owner-loss step already killed it
                    cleanup_errors.append("stop_remote (expected after owner loss): " + repr(error))
            if worker_process is not None:
                worker_process.wait(timeout=15)
        except Exception as error:  # noqa: BLE001
            cleanup_errors.append(repr(error))
        sampler.stop()
        save(args.output / "POWER_SAMPLES.json", sampler.rows())
        save(args.output / "COMMANDS.json", commands)
        save(args.output / "CLEANUP.json", {"errors": cleanup_errors, "status": "PASS" if not cleanup_errors else "FAIL"})
        if cleanup_errors:
            print("CLEANUP_ERRORS", cleanup_errors, flush=True)


if __name__ == "__main__":
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("gate interrupted; stopping owned endpoints")
    signal.signal(signal.SIGTERM, interrupted)
    main()
