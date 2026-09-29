"""Materialize pinned Stage A evidence and fresh matched development trace inputs."""

import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.environ["S42_UNIFIED_REPO_ROOT"])
from research_dev.scheduler import RuntimePhonePowerProfile  # noqa: E402
from research_dev.scheduler._internal.types import canonical_sha256  # noqa: E402
from research_dev.scheduler.adapters.phone_helpers import observe_usb_port  # noqa: E402
from research_dev.scheduler.adapters.phone_transport import ADB_TCP_TRANSPORT_GENERATION  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    root = Path(sys.argv[1])
    attempt = sys.argv[2]
    source = Path(os.environ["S42_UNIFIED_REPO_ROOT"])
    config = read(root / "GATE_CONFIG.json")
    helper = config["helper_phone"]
    device = helper["device_id"]
    manifest = read(config["manifest"])
    artifact = manifest["artifact_sha256"]
    evidence_dir = root / ("qualification-" + attempt)
    evidence_dir.mkdir(exist_ok=True)
    receipt_paths = {}

    def receipt(kind, value):
        value["status"] = "PASS"
        path = evidence_dir / (kind + ".json")
        write(path, value)
        receipt_paths[kind] = str(path)

    usb = observe_usb_port("2-9.2")
    assert usb.serial == helper["serial"] and usb.negotiated_speed_mbps >= 5000
    receipt("usb-link-speed", usb.to_json())
    numerical = root / "qualification/numerical"
    suite = read(numerical / "SUITE_RESULT.json")
    arm = suite["arms"]["04-sdot-pair-dynamic64"]
    hashes = read(numerical / "EXPECTED_PHONE_HASHES.json")
    assert suite["status"] == arm["numerical_status"] == "PASS"
    assert arm["arm"]["cpu_pair_dot"] == 1 and arm["arm"]["cpu_row_chunk"] == 64
    assert arm["max_relative_l2"] < 0.001
    assert set(value[7:] for value in helper["expected_sha256_by_path"].values()) <= set(hashes.values())
    calls = read(numerical / "SUITE_CALLS.json")["04-sdot-pair-dynamic64"]
    assert {1, 2, 4} <= {row["tokens"] for row in calls}
    receipt("numerical-rows-1-2-4", {"suite_sha256": sha(numerical / "SUITE_RESULT.json"),
        "calls_sha256": sha(numerical / "SUITE_CALLS.json"), "worker_hashes_sha256": sha(numerical / "EXPECTED_PHONE_HASHES.json"),
        "max_relative_l2": arm["max_relative_l2"], "rows": sorted({row["tokens"] for row in calls}),
        "packed_parent_receipt_sha256": sha(root.parent / "s42-pixel10pro-packed-server-20260924-v1/PACKED_PARENT.json"),
        "note": "Independent desktop F16 reference, prior same worker/libraries/weights; arithmetic is not bit-exact F16."})
    token_checks = []
    for run, names in (("mechanism-r1", ("both-matched", "both-full")),
                       ("mechanism-b4-r1", ("both-full",))):
        baseline = sorted((read(p) for p in (root / run / "desktop-before").glob("EXECUTION-*.json")), key=lambda x: x["index"])
        for name in names:
            executions = sorted((read(p) for p in (root / run / name).glob("EXECUTION-*.json")), key=lambda x: x["index"])
            assert len(baseline) == len(executions) and all(a["tokens"] == b["tokens"] and a["prompt_tokens"] == b["prompt_tokens"]
                                                          for a, b in zip(baseline, executions))
            proof_path = root / run / name / "TWO_PHONE_RESULT.json"
            proof = read(proof_path)
            assert proof["helper_served_calls"] == 366 and proof["helper_stop"]["exit_code"] == 0
            token_checks.append({"run": run, "arm": name, "outputs": len(executions), "tokens_each": 64,
                                 "proof_sha256": sha(proof_path), "result_sha256": sha(root / run / name / "RESULT.json")})
    receipt("server-token-identity", {"checks": token_checks})
    timing_path = root / "tcp-calibration-r2/RESULT.json"
    timing = read(timing_path)
    assert timing["status"] == "PASS" and len(timing["calls"]) == 216 and timing["stop"]["exit_code"] == 0
    receipt("adb-forward-round-trip", {"result_sha256": sha(timing_path), "calls": 216, "rows": [1, 2, 4]})
    idle = read(root / "idle-stop-r2/RESULT.json")
    assert idle["status"] == "PASS" and idle["stop"]["signalled"] and idle["stop"]["boot_unchanged"]
    assert idle["stop"]["forward_removed"] and not idle["stop"]["worker_pids_after"]
    assert len(idle["launch"]["worker_pids"]) == 1 and len(idle["connections"]) == 2
    receipt("scheduler-launched-session", {"launch_sha256": sha(root / "mechanism-r1/both-full/PHONE_READY.json"),
        "stop_sha256": sha(root / "mechanism-r1/both-full/PHONE_CLOSE.json"),
        "idle_lifecycle_sha256": sha(root / "idle-stop-r2/RESULT.json"),
        "lifecycle": "AdbTcpPhoneWorkerSession; finite budget and rooted resident idle-only stop physically qualified"})
    median_by_rows = {batch: statistics.median(row["overhead_us"] for row in timing["calls"]
                                             if row["repeat"] >= 3 and row["tokens"] == batch) for batch in (1, 2, 4)}
    bandwidth = round(2 * (40960 - 10240) * 1e6 / (median_by_rows[4] - median_by_rows[1]))
    fixed_us = round(max(0, median_by_rows[1] / 2 - 10240 * 1e6 / bandwidth))
    worker_log = (root / "mechanism-r1/both-full/helper-worker.log").read_text()
    medians = [(int(n), int(us)) for n, us in re.findall(r"requests=(\d+) compute_p50_us=(\d+)", worker_log)]
    compute_us = [us for n, us in medians if n <= 366][-1]
    logical_bytes = 3 * 5120 * 17408 * 2
    rate = logical_bytes * 1_000_000 // compute_us
    cost_receipt = {"status": "PASS", "kernel_compute_us": compute_us, "logical_f16_weight_bytes_per_layer": logical_bytes,
                    "effective_logical_bytes_per_s": rate, "transfer_overhead_us_by_rows": median_by_rows,
                    "one_direction_fixed_us": fixed_us, "effective_transfer_bytes_per_s": bandwidth,
                    "note": "Measured latency; logical F16 equivalent throughput, not physical memory bandwidth. "
                            "Active/idle phone power 4.5/0.875W is separately assumed. B1 kernel prior is conservative at B4.",
                    "kernel_source_sha256": sha(root / "mechanism-r1/both-full/helper-worker.log"),
                    "link_source_sha256": sha(timing_path)}
    write(evidence_dir / "COST_CALIBRATION.json", cost_receipt)
    cost_hash = sha(evidence_dir / "COST_CALIBRATION.json")
    domain = device + "-system"
    fragment = {
        "devices": [{"device_id": device, "kind": "phone", "memory_pool_id": "pixel-ram",
                     "allocation_limit_bytes": 12 * 1024**3, "ready": True}],
        "memory_pools": [{"pool_id": "pixel-ram", "capacity_bytes": 12 * 1024**3, "reserved_bytes": 0}],
        "domains": [{"domain_id": domain, "idle_power_mw": 875, "status": "estimated", "evidence_ids": ["ASSUMED_4P5W"]}],
        "idle_charge_domains": [domain],
        "kernels": [{"device_id": device, "domain_id": domain, "active_power_mw": 4500,
                     "effective_bytes_per_s": rate, "effective_ops_per_s": rate,
                     "kernel_id": "prior:" + device + ":ffn", "profile_id": "prior:" + device + ":ffn",
                     "launch_us": 0, "status": "measured", "evidence_ids": [cost_hash, "ASSUMED_4P5W"]}],
        "links": [{"link_id": "pixel-adb-" + direction, "source_device": a, "target_device": b,
                   "bandwidth_bytes_per_s": bandwidth, "fixed_latency_us": fixed_us, "fixed_dynamic_uj": 0,
                   "dynamic_pj_per_byte": 0, "domain_active_power_mw": {}, "status": "measured", "ready": True,
                   "evidence_ids": [cost_hash], "concurrent_streams": 4, "maximum_payload_bytes": 40960}
                  for direction, a, b in (("out", "desktop-cpu", device), ("in", device, "desktop-cpu"))],
    }
    server = Path(config["server"])
    host_paths = {"host_binary_sha256": str(server), "host_impl_sha256": str(server.parent / "libllama-server-impl.so"),
                  "transport_client_source_sha256": "/mnt/storage/s42-trace-v2-20260921-prep/source/examples/layersplit/ffn-split-client.cpp"}
    software = {name: sha(path) for name, path in host_paths.items()}
    software.update({"phone_worker_sha256": helper["expected_sha256_by_path"][helper["worker_path"]],
                     "phone_shard_sha256": helper["expected_sha256_by_path"][helper["shard_path"]],
                     "worker_environment_sha256": canonical_sha256(helper["worker_environment"]),
                     **{"phone_library_sha256:" + path: value for path, value in helper["expected_sha256_by_path"].items()
                        if path.endswith(".so")}})
    kernel = subprocess.check_output(["/usr/bin/adb", "-P", "5037", "-s", helper["serial"], "shell", "uname -r"],
                                     stdin=subprocess.DEVNULL, text=True).strip()
    worker = {key: helper[key] for key in ("device_id", "serial", "adb_port", "adb_path", "worker_path", "library_directories",
        "shard_path", "layer_mask", "columns", "column_quantum", "max_tokens", "backend", "phone_port",
        "worker_environment", "expected_sha256_by_path", "as_root", "phone_lock_path")}
    worker.update(artifact_sha256=artifact, n_embd=5120, swiglu=True, forward_port=26991, max_requests=0)
    bundle = {"schema": "s42-static-helper-evidence-v1", "status": "PASS", "worker": worker,
        "profile_fragment": fragment, "power": RuntimePhonePowerProfile.assumed_4p5w(
            device_id=device, domain_id=domain, allow_assumed_for_scheduling=True, minimum_battery_ppm=50000).to_json(),
        "receipt_paths": receipt_paths, "host_software_paths": host_paths,
        "transport_identity": {"schema": "research-scheduler-phone-helper-transport-identity-v1", "device_id": device,
            "transport": "adb-tcp", "transport_generation": ADB_TCP_TRANSPORT_GENERATION, "minimum_usb_speed_mbps": 5000,
            "hardware_identity": {"adb_usb_identity": usb.vendor_product, "host_usb_controller": usb.host_controller,
                "phone_kernel_release": kernel, "phone_usb_serial": helper["serial"], "phone_usb_sysfs_device": "2-9.2"},
            "software_identity": software, "receipts": {kind: sha(path) for kind, path in receipt_paths.items()}}}
    bundle_path = evidence_dir / "PIXEL_EVIDENCE.json"
    write(bundle_path, bundle)
    shard_index = evidence_dir / "PIXEL_FFN_SHARDS.json"
    shard_hash = software["phone_shard_sha256"]
    record = {"path": "QWEN_PACKED.ffn.gguf", "parent_sha256": artifact, "shard_sha256": shard_hash,
              "layer_mask": "0000000000fc0000", "columns": 17408, "n_ff": 17408, "shard_bytes": 948389216,
              "weight_type": "MIXED_Q4_K_Q6_K", "session_id": "PIXEL10PRO0"}
    write(shard_index, {"schema": "s42-ffn-shard-index-v1", "parent_sha256": artifact, "shards": [record]})
    previous = Path("/home/zhihao/s42-trace-longtaildev2-allon-20260924-inputs")
    identity = Path("/mnt/storage/s42-trace-v2-20260921-prep/TRANSPORT_QUALIFICATION_IDENTITY_PIXEL_STAGEA_COALESCED_BOTH.json")
    for arm in ("desktop", "op15", "two-phone"):
        inputs = root / ("inputs-" + arm + "-" + attempt)
        inputs.mkdir()
        campaign, rig, models, evidence = [read(previous / (name + ".json")) for name in ("campaign", "rig", "models", "evidence")]
        rig["repo_root"] = str(source)
        campaign["campaign_id"] = "s42-pixel-stagea-dev2-" + arm + "-20260924-" + attempt
        for name in ("rig", "models", "evidence"):
            campaign[name + "_manifest_path"] = str(inputs / (name + ".json"))
        campaign["selection_mode"] = "desktop-baseline" if arm == "desktop" else "energy-aware"
        evidence["transport_qualification_identity_path"] = str(inputs / "TRANSPORT_QUALIFICATION_IDENTITY.json")
        write(inputs / "TRANSPORT_QUALIFICATION_IDENTITY.json", read(identity))
        if arm == "two-phone":
            rig["helper_phones"] = [{**{k: v for k, v in worker.items() if k in (
                "device_id", "serial", "adb_port", "backend", "worker_path", "library_directories", "column_quantum",
                "max_tokens", "forward_port", "max_requests", "worker_environment", "as_root", "phone_lock_path")},
                "worker_port": worker["phone_port"], "kernel_release": kernel, "transport": "adb-tcp",
                "minimum_usb_speed_mbps": 5000, "usb_sysfs_device": "2-9.2"}]
            rig["devices"].append({"device_id": device, "kind": "phone", "memory_capacity_bytes": 12 * 1024**3})
            rig["resources"].extend({"resource_id": name, "identity": name, "capacity": 1} for name in ("pixel-cpu", "pixel-adb"))
            rig["topology"]["helper_phones"] = [{"device_id": device, "memory_resource_id": "pixel-ram",
                "compute_resource_ids": ["pixel-cpu"], "transport_resource_ids": ["desktop-usb-root", "pixel-adb"]}]
            qwen = next(row for row in models["models"] if row["model_key"] == "hot")
            qwen["helper_phone_ffn_shards"] = {device: {"index_path": str(shard_index),
                                                       "directory": str(Path(worker["shard_path"]).parent)}}
            evidence["helper_phone_evidence_paths"] = {device: str(bundle_path)}
        for name, value in (("campaign", campaign), ("rig", rig), ("models", models), ("evidence", evidence)):
            write(inputs / (name + ".json"), value)
    print("MATERIALIZED", root)


if __name__ == "__main__":
    main()
