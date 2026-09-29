"""Bounded per-layer KV physical gate. This is not route/energy qualification.

Run one arm at a time under the same externally enforced memory budget. Both arms use
the same native binary and KV plan; remote-prefill omits whole owned FFNs before context
allocation. Never kill an in-flight phone worker or change unrelated host processes.
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_dev.scheduler import ModelManifest
from research_dev.scheduler._internal.kv_placement import plan_layer_kv
from research_dev.scheduler._internal.runtime_plan import RuntimeExecutionContract, RuntimePhoneShard, RuntimeTransitionPlan
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler.adapters import (
    HostEnergySampler, LlamaCppHttpClient, LlamaServerLaunchContract,
    LlamaServerProcessConfiguration, LlamaServerProcessLauncher, default_host_metric_callbacks,
)
from research_dev.scheduler.adapters.http_backend import LlamaCppCompletionPayload
from research_dev.scheduler.adapters.phone_session import DirectPhoneFfnSession, DirectPhoneFfnSessionConfiguration
from research_dev.scheduler.adapters.phone_transport import PhoneTransportContract
from research_dev.scheduler.adapters.ticket import PhysicalParticipantCommand, PhysicalTransitionCommand
from research_dev.scheduler.adapters.llama_server_contracts import LlamaServerPhoneSessionProof, parse_llama_server_ffn_call
from research_dev.scheduler.campaigns.burstgpt.remote_resident_gate import _energy, _memory_record, _tensor_ranges


def save(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def sha(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def native_phone_proofs(lines, shards, manifest, request_id, prompt_tokens, output_tokens):
    """Verify the isolated full-FFN request, excluding unscoped launch warm-up calls."""
    by_layer, totals, seen = {}, {}, set()
    for line in lines:
        call = parse_llama_server_ffn_call(line.rstrip())
        if call is None or call.rows_for(request_id) == 0:
            continue
        owners = [shard for shard in shards if shard.layer_mask >> call.layer & 1]
        if (len(owners) != 1 or call.request_id in seen or len(call.contexts) != 1
                or call.columns != manifest.feed_forward_length
                or call.payload_bytes != call.tokens * manifest.embedding_length * 2):
            raise ValueError("native full-FFN execution differs from its owner")
        seen.add(call.request_id)
        shard = owners[0]
        by_layer[call.layer] = by_layer.get(call.layer, 0) + call.tokens
        row = totals.setdefault(shard.session_id, [0, 0, 0])
        row[0] += 1
        row[1] += call.tokens
        row[2] += call.payload_bytes
    expected = {il: prompt_tokens + output_tokens - 1 for shard in shards
                for il in range(manifest.block_count) if shard.layer_mask >> il & 1}
    if by_layer != expected or any(shard.artifact_sha256 != manifest.artifact_sha256 for shard in shards):
        raise ValueError("native full-FFN execution is incomplete")
    return tuple(LlamaServerPhoneSessionProof(
        shard.session_id, shard.endpoint, shard.artifact_sha256, shard.resident_geometry_sha256,
        shard.operator_plan_sha256, shard.session_generation, shard.layer_mask, *totals[shard.session_id])
        for shard in shards)


def phone_owner(config, manifest, output, plan):
    phone = config["phone"]
    artifact = manifest.artifact_sha256
    n_ff = manifest.feed_forward_length
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
        raise ValueError("remote FFNs must belong to the unchanged CPU parent")
    usb = PhoneTransportContract("functionfs-usb", "devmem", 4, 4, manifest.embedding_length * 2 * config["ubatch"],
        True, False, "functionfs-dmabuf-async-ring-v2", "layer-kv-native-gate",
        int(Path("/sys/module/usbcore/parameters/usbfs_memory_mb").read_text()) * 1024**2,
        65536, 0x18d1, 0x2d00, "127.0.0.1", 0)
    cfg = DirectPhoneFfnSessionConfiguration(
        adb_path=Path("/usr/bin/adb"), usb_close_path=Path(phone["usb_close"]), serial=phone["serial"], adb_port=5037,
        session_script=phone["session_script"], restore_script=phone["restore_script"],
        session_root="/data/local/tmp/" + output.parent.name + "-" + output.name,
        worker_paths_by_artifact={artifact: phone["worker"]}, model_paths_by_artifact={artifact: phone["model"]},
        backend_by_device={"op15-phone": "HTP0"}, minimum_usb_speed_mbps=5000,
        required_kernel_release=phone["kernel_release"], launch_timeout_s=180, session_timeout_s=7200,
        diagnostic_port=18383, diagnostic_host="192.168.42.1", busybox_path="/data/adb/magisk/busybox",
        network_manager_path=Path("/usr/bin/nmcli"), android_gadget_path="/config/usb_gadget/g1",
        functionfs_gadget_path="/config/usb_gadget/g2", functionfs_root_path="/dev/usb-ffs/s41",
        phone_usb_controller="a600000.dwc3", resident_workers_path=phone["resident_workers"],
        resident_router_path=phone["resident_router"], multi_session_port_base=26760,
        multi_session_device_count=len(shards), remote_hash_cache_path=output.parent / "PHONE_HASH_CACHE.json")
    owner = DirectPhoneFfnSession(cfg)
    execution = RuntimeExecutionContract(execution_mode="adaptive-split", initial_split_fraction_ppm=0,
        allowed_adaptive_fractions_ppm=(0, 1000000), batch_plan="split-row", maximum_batch_size=config["ubatch"],
        queue_depth=4, phone_device_id="op15-phone", phone_endpoint="session://op15-phone",
        operator_kind="ffn", phone_shards=tuple(shards))
    parameters = {
        "model_alias": manifest.model_id, "cpu_device_id": "cpu", "gpu_device_id": "gpu", "phone_device_id": "op15-phone",
        "context_size": plan.context_size, "gpu_layers": config["gpu_layers"], "parallel": 1,
        "batch_size": config["batch"], "ubatch_size": config["ubatch"], "ffn_transport": "functionfs-usb",
        "ffn_activation": "swiglu" if manifest.architecture != "gemma4" else "geglu",
        "ffn_column_quantum": config["column_quantum"], "ffn_n_embd": manifest.embedding_length,
        "ffn_max_tokens": config["ubatch"], "ffn_timeout_ms": 120000,
        "ffn_assistance_phase": "decode", "ffn_runtime_control_protocol": "decode-boundary-v1",
        "usb_allocator": "devmem", "usb_queue_depth": 4, "usb_concurrent_streams": 4,
        "usb_max_payload_bytes": usb.max_payload_bytes, "usb_full_duplex": 1, "usb_split_h2d": 0,
        "usb_slot_safety_bytes": usb.slot_safety_bytes, "usb_vendor_id": usb.vendor_id, "usb_product_id": usb.product_id,
        "usb_transport_generation": usb.generation, "usb_transport_profile_id": usb.profile_id,
        "usbfs_available_bytes": usb.usbfs_available_bytes, "usb_batch_plan": usb.batch_plan,
    }
    operators = [{"operator_id": f"layer:{il}:ffn", "operator_kind": "ffn", "device_ids": ["cpu", "op15-phone"],
                  "split_axis": "none", "split_fraction_ppm": 0}
                 for il in range(manifest.block_count) if mask >> il & 1]
    command = PhysicalTransitionCommand("kv-preload", "kv-preload", artifact, "kv-capacity-gate", plan.plan_sha256,
        PhysicalParticipantCommand("physical:op15-phone", "op15-phone", "session://op15-phone", "hexagon-htp", ("op15-htp",)),
        RuntimeTransitionPlan(transition_id="kv-preload", device_id="op15-phone", source_state="cold", target_state="hot",
            latency_us=0, energy_uj=0, resource_ids=("op15-htp",), maturity="SHADOW", phone_shards=tuple(shards)),
        execution, parameters, phone_layout_generation=1, selection_mode="calibration", operator_plan_protocol="llama-server-http-v1",
        operator_plan={"route_id": "kv-capacity-gate", "plan_sha256": plan.plan_sha256, "operators": operators,
            "assisted_operator_kind": "ffn", "execution_contract": execution.to_json()})
    env = dict(usb.server_environment())
    env.update({"S41_SERVER_FFN_ACTIVATION": parameters["ffn_activation"], "S41_SERVER_FFN_ARTIFACT_SHA256": artifact,
        "S41_SERVER_FFN_COLUMNS": str(n_ff), "S41_SERVER_FFN_F16_IO": "1", "S41_SERVER_FFN_LAYER_MASK": str(mask),
        "S41_SERVER_FFN_MAX_TOKENS": str(config["ubatch"]), "S41_SERVER_FFN_N_EMBD": str(manifest.embedding_length),
        "S41_SERVER_FFN_RUNTIME_CONTROL": "1", "S41_SERVER_FFN_TIMEOUT_MS": "120000",
        "S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK": str(mask),
        "S41_SERVER_FFN_SHARDS": ";".join(f"{name}@session://op15-phone/{name}:{value}" for name, value in masks.items())})
    return owner, command, usb, env, mask


def run(config, output, arm):
    output.mkdir(parents=False, exist_ok=False)
    manifest = ModelManifest.from_json(json.loads(Path(config["manifest"]).read_text()))
    # This native revision counts the output layer in n_gpu_layers.
    first_gpu_layer = max(0, manifest.block_count + 1 - config["gpu_layers"])
    default = {il: "cpu" if il < first_gpu_layer else "gpu" for il in range(manifest.block_count)}
    plan = plan_layer_kv(manifest, context_size=config["context"], parallel=1, ubatch_size=config["ubatch"],
        default_pool_by_layer=default, host_pool="cpu", kv_budget_by_pool=config["kv_budgets"])
    server_path, model = Path(config["server"]), Path(config["model"])
    save(output / "CONFIG.json", config)
    save(output / "KV_PLAN.json", plan.to_json())
    model_sha = sha(model)
    save(output / "RUNTIME.json", {"server": sha(server_path), "model": model_sha,
         "libraries": {path.name: sha(path) for path in sorted(server_path.parent.glob("*.so*")) if not path.is_symlink()}})
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
    sampler = HostEnergySampler(default_host_metric_callbacks(), interval_s=0.1)
    managed = owner = None
    phone_started = False
    stop = threading.Event()
    record = {"arm": arm, "status": "RUNNING", "kv_plan_sha256": plan.plan_sha256,
              "prefill_policy": "remote-prefill" if arm == "phone" else "local-prefill", "scheduler_qualified": False}
    try:
        sampler.start()
        time.sleep(0.3)
        paid_start = time.monotonic_ns()
        environment = {}
        ranges = []
        if arm == "phone":
            owner, command, usb, environment, mask = phone_owner(config, manifest, output, plan)
            save(output / "PHONE_COMMAND.json", command.to_json())
            save(output / "PHONE_PREFLIGHT.json", owner.preflight().to_json())
            started = time.monotonic_ns()
            phone_started = True
            ready = owner.start(command, manifest, usb)
            record["phone_preload_s"] = (time.monotonic_ns() - started) / 1e9
            save(output / "PHONE_READY.json", ready.to_json())
            ranges = _tensor_ranges(model, mask)
            print("PHONE_READY", record["phone_preload_s"], flush=True)
        launcher = LlamaServerProcessLauncher(LlamaServerProcessConfiguration(server_path=server_path,
            model_paths_by_artifact={manifest.artifact_sha256: model}, library_paths_by_device={"gpu": (Path(config["cuda_lib_dir"]),)},
            executable_device_names={"gpu": "CUDA0"}, output_directory=output,
            common_library_paths=(server_path.parent, Path(config["cuda_lib_dir"]))))
        contract = LlamaServerLaunchContract(manifest.model_id, plan.context_size, 1, config["batch"], config["ubatch"],
            config["gpu_layers"], "cpu", "gpu", "op15-phone" if arm == "phone" else None, environment,
            kv_cpu_layers=plan.cpu_layers, cuda_graph_mode="default")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        endpoint = f"http://127.0.0.1:{port}"
        started = time.monotonic_ns()
        managed = launcher.launch_contract(endpoint, contract, manifest, label=arm, control_check=lambda: None)
        record["desktop_load_s"] = (time.monotonic_ns() - started) / 1e9
        record["memory_ready"] = _memory_record(managed, model, ranges)
        save(output / "READY.json", record)
        print("DESKTOP_READY", arm, record["desktop_load_s"], flush=True)

        def observe():
            with (output / "MEMORY.jsonl").open("x") as stream:
                while not stop.is_set():
                    try:
                        cgroup = Path("/proc/self/cgroup").read_text().strip().split(":", 2)[2]
                        cg = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
                        row = {"time_ns": time.monotonic_ns(), "cgroup": cgroup,
                            "status": Path(f"/proc/{managed.pid}/status").read_text(),
                            "memory": {name: (cg / name).read_text().strip() for name in ("memory.current", "memory.max", "memory.events")}}
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                    except (OSError, ValueError):
                        pass
                    stop.wait(0.5)
        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        prompt = Path(config["prompt_file"]).read_text() + config.get("prompt_suffix", "")
        connection.request("POST", "/tokenize", json.dumps({"content": prompt, "add_special": True}),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError("tokenization failed")
        tokens = tuple(json.loads(response.read())["tokens"])
        connection.close()
        if len(tokens) + config["output_tokens"] > plan.context_size:
            raise ValueError("prompt exceeds context; do not silently truncate")
        save(output / "REQUEST.json", {"prompt_sha256": "sha256:" + hashlib.sha256(prompt.encode()).hexdigest(), "prompt_tokens": tokens,
                                       "output_tokens": config["output_tokens"]})
        print("REQUEST", len(tokens), config["output_tokens"], flush=True)
        request_id = f"kv-{arm}"
        if owner:
            owner.bind_ticket_generation(request_id)
        first = []
        started = time.monotonic_ns()
        result = LlamaCppHttpClient().complete(endpoint, LlamaCppCompletionPayload(
            request_id, manifest.model_id, len(tokens), config["output_tokens"], tokens, 17, output / "output.raw", first.append), lambda: None)
        finished = time.monotonic_ns()
        record.update(result=result, request_s=(finished - started) / 1e9,
                      prefill_to_first_s=None if not first else (first[0] - started) / 1e9,
                      memory_finished=_memory_record(managed, model, ranges))
        record["request_host_energy"] = _energy(sampler, started, finished)
        record["paid_host_energy"] = _energy(sampler, paid_start, finished)
        record["paid_s"] = (finished - paid_start) / 1e9
        record["phone_energy_note"] = "Assumed, conservative upper scenario: phone active power over the entire paid span, including host loading. Desktop arm uses phone idle power."
        record["assumed_phone_j"] = {str(power): record["paid_s"] * (power if arm == "phone" else 0.875) for power in (3, 4.5, 6)}
        stop.set()
        observer.join(timeout=2)
        managed.stop()
        if owner:
            proofs = native_phone_proofs(managed.stderr_lines, command.execution_contract.phone_shards,
                manifest, request_id, len(tokens), config["output_tokens"])
            save(output / "NATIVE_EXECUTION_PROOFS.json", [proof.to_json() for proof in proofs])
            owner.record_execution_proof(request_id, manifest.artifact_sha256, proofs)
        managed = None
        if owner:
            close = owner.finish(require_execution=True)
            save(output / "PHONE_CLOSE.json", close.to_json())
            phone_started = False
        record["status"] = "COMPLETED"
        save(output / "RESULT.json", record)
        print("COMPLETED", arm, record["request_s"], flush=True)
    except BaseException as error:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("desktop", "phone", "pair"), required=True)
    parser.add_argument("--wait-for-rig-seconds", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.wait_for_rig_seconds <= 600:
        parser.error("rig wait must be between 0 and 600 seconds")
    config = json.loads(args.config.read_text())
    with Path(config["execution_lock"]).open("a") as lock:
        deadline = time.monotonic() + args.wait_for_rig_seconds
        announced = False
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise
                if not announced:
                    print("WAITING_FOR_RIG", args.wait_for_rig_seconds, flush=True)
                    announced = True
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        if args.arm == "pair":
            args.output.mkdir(parents=False, exist_ok=False)
            run(config, args.output / "phone", "phone")
            run(config, args.output / "desktop", "desktop")
        else:
            run(config, args.output, args.arm)


if __name__ == "__main__":
    main()
