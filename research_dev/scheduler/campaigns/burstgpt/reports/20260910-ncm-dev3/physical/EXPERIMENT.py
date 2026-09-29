"""Configuration and evidence capture for the unchanged three-request gate."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "preflight", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    options = parser.parse_args()
    root, repo = options.root, options.repo
    frozen = Path("/mnt/storage/s42-cuda-graph-v1-20260909/reference-inputs")
    previous = Path("/mnt/storage/s42-telemetry-qwen-20260910-v1")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from research_dev.scheduler import RuntimeCapabilityCatalog
    from research_dev.scheduler.adapters import verify_android_usb_restored
    from research_dev.scheduler.config import load_scheduler_configuration
    from research_dev.scheduler.campaigns.burstgpt.launch import (
        _run_streamed, _source_manifest, command_manifest, file_sha256,
        preflight_command, runner_command, write_new,
    )

    inputs = root / "inputs"
    catalog_path = inputs / "CATALOG.json"

    def configuration():
        return load_scheduler_configuration(inputs / "campaign.json", environ={})

    def observation():
        return {
            "epoch_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
            "cpu_stat": Path("/proc/stat").read_text().splitlines()[0],
            "processes": subprocess.check_output(
                ["ps", "-eo", "pid,user,etimes,pcpu,comm,args"], text=True),
            "gpu": subprocess.check_output([
                "nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,power.draw",
                "--format=csv,noheader"], text=True),
        }

    if options.stage == "prepare":
        root.mkdir()
        inputs.mkdir()
        rig = read(previous / "inputs/rig.json")
        rig["repo_root"] = str(repo)
        rig["binaries"]["close_helper"] = str(repo / "research_dev/scheduler/adapters/close_resident_bridge.py")
        rig["phone"].update(
            session_root="/data/local/tmp/" + root.name,
            whole_state_directory="/data/local/tmp/" + root.name + "-whole",
            remote_hash_cache_path=str(inputs / "PHONE_HASH_CACHE.json"),
            whole_control_transport="adb-ncm", whole_ncm_adb_endpoint="192.168.42.1:5555",
        )
        write_new(inputs / "rig.json", rig)
        write_new(inputs / "PHONE_HASH_CACHE.json", read(frozen / "PHONE_HASH_CACHE.json"))
        models = read(frozen / "matched-models.json")
        overlay = next(row for row in models["models"] if row["kind"] == "overlay")
        overlay["endpoint_ids"]["whole_phone"] = "llama_whole_phone"
        overlay["backend_ids"]["whole_phone"] = "android-llama-server-opencl"
        overlay["phone_adapter_parameters"].update(
            persistent_residency=1, whole_model_peak_memory_bytes=3_000_000_000,
        )
        write_new(inputs / "models.json", models)
        evidence = read(frozen / "evidence.json")
        evidence["prematerialized_catalog_path"] = str(catalog_path)
        write_new(inputs / "evidence.json", evidence)
        campaign = read(previous / "inputs/campaign.json")
        campaign.update(campaign_id=root.name, rig_manifest_path=str(inputs / "rig.json"),
                        models_manifest_path=str(inputs / "models.json"),
                        evidence_manifest_path=str(inputs / "evidence.json"))
        assert campaign["fixed_phone_residency"] is None
        assert campaign["selection_mode"] == "energy-aware" and campaign["include_startup_preparation"]
        write_new(inputs / "campaign.json", campaign)
        cfg = configuration()
        from research_dev.scheduler._internal.model_manifest_cache import load_cached_gguf_manifest
        from research_dev.scheduler.campaigns.burstgpt.catalog import (
            _physical_topology, _register_overlay_whole_phone,
        )
        catalog = RuntimeCapabilityCatalog.from_json(read(frozen / "matched-CATALOG.json"))
        overlay_cfg = cfg.models.overlay_model
        manifest = load_cached_gguf_manifest(overlay_cfg.model_id, overlay_cfg.host_artifact_path,
                                             cfg.models.manifest_cache_path)
        topology = _physical_topology(cfg.rig, cfg.rig.topology,
                                     catalog.executor_by_device[cfg.rig.topology.phone_device_id].phone_sessions)
        catalog = _register_overlay_whole_phone(
            catalog, manifest, overlay_cfg, cfg.rig, topology, cfg.rig.endpoints,
            (cfg.rig.phone.whole_server_sha256,),
        )
        write_new(catalog_path, catalog.to_json())
        write_new(root / "RESOLVED_CONFIGURATION.json", cfg.to_json())
        write_new(root / "EXPERIMENT_SPEC.json", {
            "workload": "burstgpt_dev3_long_v1.json", "requests": [36, 37, 50],
            "arrivals_s": [1, 61, 91], "output_tokens": [292, 292, 71],
            "selection_mode": "energy-aware", "fixed_residency": None,
            "phone_powers_mw": [3000, 4500, 6000], "phone_idle_power_mw": 875,
            "runtime_preparation_and_cleanup_included": True,
            "baseline_rerun": False, "longer_trace": False,
            "cache_policy": "No cache flush, prefetch or advance phone preload",
            "comparison_kind": "historical-reference, not matched A/B",
            "differences": ["scheduler source", "NCM whole-phone capability and control",
                            "persistent whole-phone service with conservative 3 GB declaration",
                            "fresh artifact namespace"],
            "memory_caveat": "Configured peak is reserved, not a measured OpenCL peak qualification",
            "concurrency_caveat": "Existing shared HTP resource exclusion remains unchanged",
            "initial_evidence": {name: file_sha256(frozen / name) for name in (
                "INITIAL_AUTOMATED_OBSERVATIONS.json", "INITIAL_ADAPTIVE_OBSERVATIONS.json")},
        })
        write_new(root / "HOST_BEFORE_PREFLIGHT.json", observation())
        print("PREPARED", root, flush=True)
        return

    cfg = configuration()
    if options.stage == "preflight":
        output = root / "preflight"
        output.mkdir()
        receipt = verify_android_usb_restored(
            serial=cfg.rig.phone.serial, adb_port=cfg.rig.phone.adb_port,
            minimum_speed_mbps=cfg.rig.phone.minimum_usb_speed_mbps, timeout_s=60)
        write_new(output / "PHONE_USB_BEFORE.json", receipt.to_json())
        command = preflight_command(cfg, catalog_path=catalog_path,
                                    normal_usb_receipt_path=output / "PHONE_USB_BEFORE.json",
                                    output_path=output / "PHYSICAL_PREFLIGHT.json")
        write_new(output / "COMMAND.json", list(command))
        _run_streamed(command, cwd=repo, log_path=output / "RUN.log")
        return

    assert read(root / "preflight/PHYSICAL_PREFLIGHT.json")["status"] == "PASS"
    assert not (root / "run").exists()
    source = inputs / "SOURCE_MANIFEST_EXECUTION.json"
    write_new(source, _source_manifest(cfg))
    command = runner_command(cfg, catalog_path=catalog_path, source_manifest_path=source,
                             output_path=root / "run", execute=True)
    write_new(root / "RUN_COMMAND_EXECUTION.json", list(command))
    write_new(root / "COMMAND_MANIFEST_EXECUTION.json", command_manifest(cfg, command, catalog_path))
    os.environ.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    nsys = "/mnt/storage/s21_deps/nsys-2026.3.1/opt/nvidia/nsight-systems-cli/2026.3.1/bin/nsys"
    profiled = (nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                "--cuda-graph-trace=graph", "--cuda-trace-all-apis=true",
                "--output", str(root / "CUDA"), *command)
    write_new(root / "PROFILE_COMMAND.json", list(profiled))
    write_new(root / "HOST_BEFORE_RUN.json", observation())
    stop = threading.Event()

    def monitor():
        with (root / "HOST_ACTIVITY.jsonl").open("x") as stream:
            while not stop.is_set():
                stream.write(json.dumps(observation(), sort_keys=True) + "\n")
                stream.flush()
                stop.wait(1)

    sampler = threading.Thread(target=monitor, daemon=True)
    sampler.start()
    try:
        _run_streamed(profiled, cwd=repo, log_path=root / "RUN.log")
    finally:
        stop.set()
        sampler.join(timeout=10)
        write_new(root / "HOST_AFTER_RUN.json", observation())
    subprocess.run([nsys, "export", "--type=sqlite", "--output", str(root / "CUDA.sqlite"),
                    str(root / "CUDA.nsys-rep")], check=True)


if __name__ == "__main__":
    main()
