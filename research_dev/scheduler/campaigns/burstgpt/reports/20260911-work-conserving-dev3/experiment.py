"""Configuration and persistence for the unchanged three-request experiment."""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "preflight", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--inputs-name", default="inputs")
    parser.add_argument("--preflight-name", default="preflight")
    args = parser.parse_args()
    root, repo = args.root, args.repo
    sys.path.insert(0, str(repo))
    from research_dev.scheduler import ModelManifest, RuntimeCapabilityCatalog, RuntimeRouteShapeProfile
    from research_dev.scheduler.adapters import (
        RuntimeModelEndpointCapability, materialize_cpu_phone_endpoints, verify_android_usb_restored,
    )
    from research_dev.scheduler.adapters.phone_session_discovery import phone_session_discovery_json
    from research_dev.scheduler.adapters.transport_profiles import TransportQualificationIdentity
    from research_dev.scheduler.config import load_scheduler_configuration
    from research_dev.scheduler.campaigns.burstgpt.catalog import _physical_topology
    from research_dev.scheduler.campaigns.burstgpt.launch import (
        _run_streamed, _source_manifest, file_sha256, preflight_command, runner_command, write_new,
    )

    previous = Path("/mnt/storage/s42-ncm-dev3-20260910-v1/inputs")
    inputs = root / args.inputs_name
    catalog_path = inputs / "CATALOG.json"
    binaries = root / "cuda-build/bin"
    if args.stage == "prepare":
        inputs.mkdir()
        calibration_path = root / "cpu-parent-v2/RESULT.json"
        calibration = read(calibration_path)
        assert calibration["status"] == "PASS"
        assert calibration["server_sha256"] == file_sha256(binaries / "llama-server")
        assert calibration["gpu_process_bytes"] is None
        manifest = ModelManifest.from_json(calibration["model_manifest"])
        calibration_sha = file_sha256(calibration_path)
        rig = read(previous / "rig.json")
        rig["repo_root"] = str(repo)
        for key in ("server", "resident_server"):
            rig["binaries"][key] = str(binaries / "llama-server")
        rig["binaries"]["close_helper"] = str(repo / "research_dev/scheduler/adapters/close_resident_bridge.py")
        rig["library_directories"]["resident"] = str(binaries)
        rig["transport_host_dependencies"] = {
            key: str(binaries / Path(value).name)
            for key, value in rig["transport_host_dependencies"].items()
        }
        rig["phone"].update(
            session_root="/data/local/tmp/" + root.name,
            whole_state_directory="/data/local/tmp/" + root.name + "-whole",
            remote_hash_cache_path=str(inputs / "PHONE_HASH_CACHE.json"),
        )
        write_new(inputs / "rig.json", rig)
        write_new(inputs / "PHONE_HASH_CACHE.json", read(previous / "PHONE_HASH_CACHE.json"))
        models = read(previous / "models.json")
        models["manifest_cache_path"] = str(inputs / "GGUF_MANIFEST_CACHE.json")
        overlay = next(row for row in models["models"] if row["kind"] == "overlay")
        assert overlay["model_id"] == manifest.model_id
        overlay["phone_ffn_shard_index_path"] = "/mnt/storage/s42-llama-htp-split-20260911-v1-tMRgjn/shards/FFN_SHARDS.json"
        overlay["phone_ffn_shard_directory"] = "/data/local/tmp/s42-llama-htp-split-20260911-v1-tMRgjn"
        write_new(inputs / "models.json", models)
        evidence = read(previous / "evidence.json")
        source = RuntimeCapabilityCatalog.from_json(read(previous / "CATALOG.json"))
        sessions = source.executor_by_device[rig["topology"]["phone_device_id"]].phone_sessions
        sessions = tuple(replace(row, supported_data_types=tuple(sorted({
            *row.supported_data_types, "Q4_0",
        }))) for row in sessions)
        write_new(inputs / "PHONE_SESSION_DISCOVERY.json", phone_session_discovery_json(sessions))
        old_identity = TransportQualificationIdentity.from_json(read(evidence["transport_qualification_identity_path"]))
        transport_source = Path("/mnt/storage/s42-llama-packed-prefix-20260911-v1-JcmCWJ/source/examples/layersplit/ffn-split-client.cpp")
        assert file_sha256(transport_source) == old_identity.software_identity["transport_client_source_sha256"]
        identity = replace(old_identity, identity_id=root.name + ":transport-binding",
                           hardware_identity=dict(old_identity.hardware_identity), software_identity={
            **old_identity.software_identity,
            "host_binary_sha256": file_sha256(binaries / "llama-server"),
            **{"host_dependency_sha256:" + key: file_sha256(Path(value))
               for key, value in rig["transport_host_dependencies"].items()},
        })
        write_new(inputs / "TRANSPORT_IDENTITY.json", identity.to_json())
        evidence.update(prematerialized_catalog_path=str(catalog_path),
                        phone_session_discovery_path=str(inputs / "PHONE_SESSION_DISCOVERY.json"),
                        transport_qualification_identity_path=str(inputs / "TRANSPORT_IDENTITY.json"))
        write_new(inputs / "evidence.json", evidence)
        campaign = read(previous / "campaign.json")
        campaign.update(campaign_id=root.name, rig_manifest_path=str(inputs / "rig.json"),
                        models_manifest_path=str(inputs / "models.json"),
                        evidence_manifest_path=str(inputs / "evidence.json"))
        assert campaign["selection_mode"] == "energy-aware" and campaign["fixed_phone_residency"] is None
        assert campaign["include_startup_preparation"]
        write_new(inputs / "campaign.json", campaign)
        cfg = load_scheduler_configuration(inputs / "campaign.json", environ={})
        topology = _physical_topology(cfg.rig, cfg.rig.topology, sessions)
        source = replace(source, executors=tuple(
            replace(row, phone_sessions=sessions) if row.device_id == topology.phone_device_id else row
            for row in source.executors
        ), placement_profile=replace(source.placement_profile, links=tuple(
            replace(link, qualification_identity_sha256=identity.identity_sha256)
            if link.qualification_identity_sha256 == old_identity.identity_sha256 else link
            for link in source.placement_profile.links
        )))
        parent = source.composite_executor_by_id[source.desktop_control_by_artifact[manifest.artifact_sha256].executor_id]
        cpu_id = "physical:cpu-parent:" + manifest.artifact_sha256[7:23]
        cpu_parameters = dict(calibration["runtime_parameters"])
        peak = max(row["high_water_bytes"] for row in calibration["memory"])
        cpu_parameters["memory_model_weight_allocation_ppm:" + topology.cpu_device_id] = math.ceil(
            peak * 1_000_000 / manifest.tensor_bytes)
        model = RuntimeModelEndpointCapability(
            manifest=manifest, desktop_executor_id=parent.executor_id, desktop_endpoint=parent.endpoint,
            desktop_backend=parent.backend, desktop_gpu_first_layer=manifest.block_count - parent.adapter_parameters["gpu_layers"],
            adapter_parameters=parent.adapter_parameters, desktop_evidence_ids=parent.evidence_ids,
            phone_executor_prefix=cpu_id + ":unused-gpu-phone", phone_endpoint=parent.endpoint, phone_backend=parent.backend,
            phone_evidence_ids=(old_identity.software_identity["phone_worker_sha256"],),
            phone_preloaded=False, phone_resident_limit_bytes=max(s.resident_memory_limit_bytes for s in sessions),
            ffn_column_quantum=2048, ffn_max_runtime_partitions=manifest.block_count,
            phone_runtime_control_protocol="decode-boundary-v1",
            phone_adapter_parameters=dict(ffn_activation="swiglu", ffn_timeout_ms=120000,
                ffn_transport_slot_payload_multiplier=4, ffn_weight_buffer_layout="selected-width",
                split_output_width="full", usb_product_id=11520, usb_vendor_id=6353, usb_split_h2d=0),
            cpu_executor_id=cpu_id, cpu_endpoint="http://127.0.0.1:18684", cpu_backend="llama-server-cpu",
            cpu_phone_executor_prefix=cpu_id + ":npu", cpu_phone_endpoint="http://127.0.0.1:18684",
            cpu_phone_backend="llama-server-cpu-op15", cpu_adapter_parameters=cpu_parameters,
            cpu_evidence_ids=(calibration_sha,),
        )
        runs = calibration["runs"]
        service = max(row["duration_us"] for row in runs)
        energy = math.ceil(max(row["energy"]["server_compute_device_energy_j"] for row in runs) * 1_000_000)
        shape = calibration["request"]
        profile = RuntimeRouteShapeProfile(
            selector_id=cpu_id + ":measured-parent", artifact_sha256=manifest.artifact_sha256,
            executor_id=cpu_id, route_family="layer_placement", device_ids=(topology.cpu_device_id,),
            assisted_operator_kind=None, split_axis="none", split_fraction_ppm=0, residency_variant="hot",
            minimum_input_tokens=shape["input_tokens"], maximum_input_tokens=shape["input_tokens"],
            minimum_output_tokens=shape["output_tokens"], maximum_output_tokens=shape["output_tokens"],
            service_fixed_us=service, service_input_token_us=0, service_output_token_us=0,
            service_upper_add_us=math.ceil(service / 4), energy_fixed_uj=energy,
            energy_input_token_uj=0, energy_output_token_uj=0, energy_lower_error_ppm=250000,
            energy_upper_error_ppm=250000, sample_count=len(runs), maturity="QUALIFIED", evidence_ids=(calibration_sha,),
        )
        catalog = materialize_cpu_phone_endpoints(source, model, topology,
            transition_latency_us=math.ceil(calibration["load"]["duration_us"] * 1.25),
            transition_energy_uj=math.ceil(calibration["load"]["energy"]["server_compute_device_energy_j"] * 1.25e6),
            cpu_transition_energy_maturity="QUALIFIED", route_shape_profiles=(profile,))
        write_new(catalog_path, catalog.to_json())
        write_new(inputs / "SPEC.json", {
            "requests": [36, 37, 50], "arrivals_s": [1, 61, 91], "output_tokens": [292, 292, 71],
            "selection_mode": "energy-aware", "forced_routes": False, "advance_demand": False,
            "runtime_preparation_and_cleanup_included": True, "live_request_migration": "unsupported; finish on current parent",
            "comparison_kind": "diagnostic only; changed CPU launch and binary, no matched baseline",
            "cpu_qualification": calibration_sha, "cpu_cost_bounds": "25% heuristic padding; concurrent interference not measured by this calibration",
            "transport_binding": "New executable/dependency hashes, identical transport client source and existing measured receipts",
            "native_worker_proof": "20260910-llama-packed-prefix/physical/packed-v2; worker hash unchanged",
            "initial_evidence": {key: file_sha256(Path(value)) for key, value in evidence.items()
                                 if key in ("observation_store_path", "adaptive_observation_store_path")},
        })
        print("PREPARED", flush=True)
        return

    cfg = load_scheduler_configuration(inputs / "campaign.json", environ={})
    if args.stage == "preflight":
        output = root / args.preflight_name
        output.mkdir()
        receipt = verify_android_usb_restored(serial=cfg.rig.phone.serial, adb_port=cfg.rig.phone.adb_port,
                                             minimum_speed_mbps=cfg.rig.phone.minimum_usb_speed_mbps, timeout_s=30)
        write_new(output / "PHONE_USB_BEFORE.json", receipt.to_json())
        command = preflight_command(cfg, catalog_path=catalog_path, normal_usb_receipt_path=output / "PHONE_USB_BEFORE.json",
                                    output_path=output / "PHYSICAL_PREFLIGHT.json")
        write_new(output / "COMMAND.json", list(command))
        _run_streamed(command, cwd=repo, log_path=output / "RUN.log")
        return
    assert read(root / args.preflight_name / "PHYSICAL_PREFLIGHT.json")["status"] == "PASS"
    source_manifest = inputs / "SOURCE_MANIFEST.json"
    write_new(source_manifest, _source_manifest(cfg))
    command = runner_command(cfg, catalog_path=catalog_path, source_manifest_path=source_manifest,
                             output_path=root / "run", execute=True)
    write_new(root / "COMMAND.json", list(command))
    _run_streamed(command, cwd=repo, log_path=root / "RUN.log")


if __name__ == "__main__":
    main()
