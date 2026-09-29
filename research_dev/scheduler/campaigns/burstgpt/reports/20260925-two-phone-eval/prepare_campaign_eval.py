"""Materialize the Pixel evidence for the DVFS-fixed worker and fresh matched trace inputs (two-phone eval).

    python3 prepare_campaign_eval.py ROOT ATTEMPT TEMPLATE TAG

Copy of ../20260924-pixel-integration-2/prepare_campaign_int2.py (same arguments, same USB, receipt,
bundle, profile fragment and rig/models/evidence additions) with these changes:

1. numerical-rows-1-2-4. The int-2 check required every configured sha256 to be in the numerical suite's
   EXPECTED_PHONE_HASHES. That still holds for the libraries and the shard. A worker hash outside that set
   is accepted only through byte_identity_bridge(): the configured worker with its exact environment must
   produce byte-identical outputs to the worker the suite qualified (GATE_CONFIG
   `helper_phone_qualified_reference`, whose hash must be in the suite) over the scheduler's own
   AdbTcpPhoneWorkerSession at rows 1, 2 and 4 (cpugpu tcp1: hash-bound by preflight, clean finite-budget
   stops, dumps re-hashed from the files), plus phone-local rows 3 (cpugpu d3) and the flag matrix (d1).
2. server-token-identity. The fresh Pixel-only llama-server run of the configured worker
   (server-identity-r1: desktop / Pixel 50 % / Pixel 100 % / desktop, 64 tokens each, all identical) is
   required; the int-2 OP15+Pixel mechanism checks are kept unchanged.
3. scheduler-launched-session. The rooted idle-TERM qualification must have launched the configured
   worker and environment (idle-stop-fixed). ATTEMPT "pre" builds a bundle from the previous worker's
   idle receipt only to run that qualification, and materializes no arm inputs.
4. COST_CALIBRATION.json. Kernel from the fixed worker's TCP qualification calls (cpugpu tcp1, the
   configured arm): bytes rate from the B1 mean per-layer compute, ops rate from the slower of B2/B4.
   The link keeps the int-2 adb-forward-round-trip derivation (conservative: its non-compute overhead
   is 11.8/16.0/20.5 ms against 7.1/10.8/13.7 ms measured with the fixed worker).
5. Campaign ids are s43-two-phone-eval-<tag>-<arm>-<attempt>.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

LOGICAL_BYTES = 3 * 5120 * 17408 * 2
LOGICAL_OPS = 2 * 3 * 5120 * 17408


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


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def command_environment(command):
    """S4x_PIXEL_* assignments in a worker launch shell string."""
    return dict(token.split("=", 1) for token in command.split()
                if re.fullmatch(r"S4\d_PIXEL_[A-Z0-9_]+=\S*", token))


def worker_launched(command, worker):
    """The launch string runs exactly this worker, libraries, shard and environment."""
    tokens = command.split()
    need("LD_LIBRARY_PATH=" + ":".join(worker["library_directories"]) in tokens, "launch library directory differs")
    need(worker["worker_path"] in tokens, "launch worker path differs")
    need(tokens[tokens.index("-m") + 1] == worker["shard_path"], "launch shard differs")
    need(command_environment(command) == worker["worker_environment"], "launch environment differs")


def clean_stop(stop):
    return (stop.get("exit_code") == 0 and stop.get("signalled") is False and stop.get("boot_unchanged") is True
            and stop.get("forward_removed") is True and stop.get("worker_pids_after") == [])


def tcp_identity(tcp_dir, configured, reference, qualified_hashes):
    """Rows 1/2/4 over the scheduler's adb-tcp session: configured worker == qualified worker, byte for byte."""
    suite = read(tcp_dir / "SUITE_RESULT.json")
    worker_hash = configured["expected_sha256_by_path"][configured["worker_path"]]
    reference_hash = reference["expected_sha256_by_path"][reference["worker_path"]]
    need(reference_hash[7:] in qualified_hashes, "reference worker is not the numerically qualified one")
    candidates = [arm for arm in suite if arm["worker_sha256"] == worker_hash
                  and arm["environment"] == configured["worker_environment"]]
    references = [arm for arm in suite if arm["worker_sha256"] == reference_hash
                  and arm["environment"] == reference["worker_environment"]]
    need(candidates and references, "no configured or no reference arm in the TCP qualification")
    shared = {value for path, value in configured["expected_sha256_by_path"].items() if path != configured["worker_path"]}
    dumps = None
    rows = set()
    for arm in candidates + references:
        name = arm["arm"]
        need(arm["status"] == "PASS" and arm["calls"] > 0 and clean_stop(arm["stop"]), "TCP arm failed or stopped uncleanly: " + name)
        files = {segment: sha(tcp_dir / name / (segment + ".f16"))[7:] for segment in arm["dump_sha256"]}
        need(files == arm["dump_sha256"], "TCP dump files differ from the recorded hashes: " + name)
        dumps = files if dumps is None else dumps
        need(files == dumps, "TCP outputs are not byte-identical to the qualified worker: " + name)
        calls = read(tcp_dir / name / "CALLS.json")
        for segment in files:
            need({row["rows"] for row in calls if row["segment"] == segment} == {int(segment[1:])},
                 "TCP segment rows differ: " + name + " " + segment)
        observed = read(tcp_dir / name / "PREFLIGHT.json")["observed_sha256_by_path"]
        launch = read(tcp_dir / name / "LAUNCH.json")["command"][-1]
        if arm in candidates:
            need(observed == configured["expected_sha256_by_path"], "TCP preflight differs from the configured pins: " + name)
            worker_launched(launch, configured)
            rows |= {row["rows"] for row in calls}
        else:
            reference_worker = [path for path, value in observed.items() if value == reference_hash]
            need(len(reference_worker) == 1 and set(observed.values()) - {reference_hash} == shared
                 and reference_worker[0] in launch.split()
                 and command_environment(launch) == reference["worker_environment"],
                 "TCP reference arm is not the qualified worker on the same libraries and shard: " + name)
    need({1, 2, 4} <= rows, "TCP qualification lacks rows 1, 2 or 4")
    return {"suite_sha256": sha(tcp_dir / "SUITE_RESULT.json"), "rows": sorted(rows),
            "candidate_arms": sorted(arm["arm"] for arm in candidates),
            "reference_arms": sorted(arm["arm"] for arm in references), "dump_sha256": dumps,
            "reference_worker_path": reference_worker[0],
            "binding": "worker sha256 by preflight on the configured paths; AdbTcpPhoneWorkerSession; finite budget"}


def local_arm_outputs(arms_dir, arm, segments):
    directory = arms_dir / arm["name"]
    for name in ("WORKER_EXIT.txt", "CLIENT_EXIT.txt"):
        need((directory / name).read_text().strip() == "0", "phone-local arm exited nonzero: " + arm["name"])
    outputs = {}
    for segment in segments:
        path = directory / ("replay." + segment + ".f16")
        need(path.stat().st_size > 0, "phone-local output missing: " + str(path))
        outputs[segment] = sha(path)
    return outputs


def phone_local_identity(suite_path, arms_dir, configured, reference, reference_worker, required_rows):
    """Phone-local replay: the configured worker+environment matches the qualified worker at required_rows."""
    suite = read(suite_path)
    segments = [row["name"] for row in suite["segments"]]
    need(required_rows <= {row["rows"] for row in suite["segments"]}, "phone-local suite lacks the required rows")
    references = [arm for arm in suite["arms"] if arm.get("prod") and arm["env"] == reference["worker_environment"]
                  and reference_worker in arm["worker_command"].split()]
    candidates = [arm for arm in suite["arms"] if arm["env"] == configured["worker_environment"]
                  and configured["worker_path"] in arm["worker_command"].split()]
    need(references and candidates, "phone-local suite lacks the configured or the reference arm")
    expected = local_arm_outputs(arms_dir, references[0], segments)
    for arm in candidates + references[1:]:
        worker_launched("exec " + arm["worker_command"], configured if arm in candidates else {
            "library_directories": configured["library_directories"], "worker_path": reference_worker,
            "shard_path": configured["shard_path"], "worker_environment": reference["worker_environment"]})
        need(local_arm_outputs(arms_dir, arm, segments) == expected,
             "phone-local outputs are not byte-identical to the qualified worker: " + arm["name"])
    return {"suite_sha256": sha(suite_path), "segments": segments, "candidate_arms": [arm["name"] for arm in candidates],
            "reference_arms": [arm["name"] for arm in references], "output_sha256": expected,
            "binding": "worker path in the same staged phone dir (phone-local replay records no per-arm sha256)"}


def flag_matrix(suite_path, arms_dir, reference, reference_worker):
    """Every CPU arm of the phone-local cadence suite (each flag alone and combined) matches the qualified worker."""
    suite = read(suite_path)
    segments = [row["name"] for row in suite["segments"]]
    cpu = [arm for arm in suite["arms"] if arm.get("backend", "CPU") == "CPU"]
    references = [arm for arm in cpu if arm.get("prod") and arm["env"] == reference["worker_environment"]
                  and reference_worker in arm["worker_command"].split()]
    need(references, "flag matrix lacks the reference arm")
    expected = local_arm_outputs(arms_dir, references[0], segments)
    for arm in cpu:
        need(local_arm_outputs(arms_dir, arm, segments) == expected,
             "flag-matrix outputs differ from the qualified worker: " + arm["name"])
    return {"suite_sha256": sha(suite_path), "segments": segments, "identical_cpu_arms": [arm["name"] for arm in cpu]}


def byte_identity_bridge(evidence, configured, reference, qualified_hashes):
    tcp = tcp_identity(evidence / "tcp1", configured, reference, qualified_hashes)
    rows3 = phone_local_identity(evidence / "suites/d3/SUITE.json", evidence / "d3/d3-m3", configured, reference,
                                 tcp["reference_worker_path"], {3})
    flags = flag_matrix(evidence / "suites/d1/SUITE.json", evidence / "d1/d1-cadence", reference, tcp["reference_worker_path"])
    return {"qualified_worker_sha256": reference["expected_sha256_by_path"][reference["worker_path"]],
            "configured_worker_sha256": configured["expected_sha256_by_path"][configured["worker_path"]],
            "tcp_rows_1_2_4": tcp, "phone_local_rows_3": rows3, "phone_local_flag_matrix": flags}


def numerical_receipt(numerical, configured, reference, evidence):
    suite = read(numerical / "SUITE_RESULT.json")
    arm = suite["arms"]["04-sdot-pair-dynamic64"]
    hashes = set(read(numerical / "EXPECTED_PHONE_HASHES.json").values())
    need(suite["status"] == arm["numerical_status"] == "PASS", "numerical suite failed")
    need(arm["arm"]["cpu_pair_dot"] == 1 and arm["arm"]["cpu_row_chunk"] == 64, "numerical arm differs")
    need(arm["max_relative_l2"] < 0.001, "numerical error too large")
    calls = read(numerical / "SUITE_CALLS.json")["04-sdot-pair-dynamic64"]
    need({1, 2, 4} <= {row["tokens"] for row in calls}, "numerical suite lacks rows 1, 2 or 4")
    pins = configured["expected_sha256_by_path"]
    need({value[7:] for path, value in pins.items() if path != configured["worker_path"]} <= hashes,
         "configured libraries or shard are not the numerically qualified ones")
    value = {"suite_sha256": sha(numerical / "SUITE_RESULT.json"), "calls_sha256": sha(numerical / "SUITE_CALLS.json"),
             "worker_hashes_sha256": sha(numerical / "EXPECTED_PHONE_HASHES.json"),
             "max_relative_l2": arm["max_relative_l2"], "rows": sorted({row["tokens"] for row in calls}),
             "note": "Independent desktop F16 reference, prior same worker/libraries/weights; arithmetic is not bit-exact F16."}
    if pins[configured["worker_path"]][7:] not in hashes:
        value["byte_identity_bridge"] = byte_identity_bridge(evidence, configured, reference, hashes)
        value["note"] += (" The configured worker is not in the suite; it inherits the qualification through outputs"
                          " byte-identical to the qualified worker (byte_identity_bridge).")
    return value


def server_identity(run_dir, configured, server):
    """Fresh Pixel-only llama-server run of the configured worker: desktop and Pixel outputs identical."""
    need(not (run_dir / "FAILURE.json").exists(), "server identity run failed")
    result = read(run_dir / "RESULT.json")
    need(result["status"] == "PASS" and result["token_identity"] == [True, True, True, True]
         and result["output_tokens_each"] == 64, "server identity outputs differ")
    need(result["server_exit"] == 0 and result["worker_exit"] == 0, "server identity run exited nonzero")
    need(result["phone_calls"] > 0 and set(result["call_columns"]) == {"8704", "17408"}
         and all(int(count) > 0 for count in result["call_columns"].values()), "server identity run made no Pixel calls")
    config = read(run_dir / "CONFIG.json")
    need(config["phone_worker"] == configured["worker_path"] and config["phone_model"] == configured["shard_path"]
         and [config["phone_library_dir"]] == configured["library_directories"]
         and config["phone_environment"] == configured["worker_environment"] and config.get("phone_root") is True,
         "server identity run used another worker configuration")
    identity = read(run_dir / "IDENTITY.json")
    observed = {line.split(None, 1)[1].strip(): "sha256:" + line.split()[0]
                for line in identity["phone_hashes"].splitlines() if line.strip()}
    need(observed == configured["expected_sha256_by_path"], "server identity phone hashes differ from the pins")
    need("sha256:" + identity["server_sha256"] == sha(server), "server identity run used another server")
    impl = str(Path(server).parent / "libllama-server-impl.so")
    need("sha256:" + identity["server_libraries"][impl] == sha(impl), "server identity run used another server library")
    worker_launched(read(run_dir / "WORKER_COMMAND.json")[-1], configured)
    return {"run": run_dir.name, "result_sha256": sha(run_dir / "RESULT.json"), "identity_sha256": sha(run_dir / "IDENTITY.json"),
            "outputs": 4, "tokens_each": 64, "arms": ["desktop", "pixel-8704", "pixel-17408", "desktop"],
            "phone_calls": result["phone_calls"], "call_columns": result["call_columns"],
            "note": "Pixel-only llama-server (single adb-tcp helper, layers 18-23), rooted configured worker"}


def idle_lifecycle(idle_dir, configured):
    idle = read(idle_dir / "RESULT.json")
    stop = idle["stop"]
    need(idle["status"] == "PASS" and stop["signalled"] and stop["boot_unchanged"] and stop["forward_removed"]
         and not stop["worker_pids_after"] and stop["exit_code"] == 0, "idle stop failed")
    need(len(idle["launch"]["worker_pids"]) == 1 and len(idle["connections"]) == 2, "idle stop sequence differs")
    need(idle["preflight"]["observed_sha256_by_path"] == configured["expected_sha256_by_path"], "idle stop pins differ")
    worker_launched(idle["launch"]["command"][-1], configured)
    return sha(idle_dir / "RESULT.json")


def fixed_worker_kernel(tcp_dir, configured):
    """Per-layer compute of the configured worker over TCP (measured steps only), pooled over its arms."""
    worker_hash = configured["expected_sha256_by_path"][configured["worker_path"]]
    arms = [arm["arm"] for arm in read(tcp_dir / "SUITE_RESULT.json")
            if arm["worker_sha256"] == worker_hash and arm["environment"] == configured["worker_environment"]]
    need(arms, "no TCP calls of the configured worker")
    calls = [row for arm in arms for row in read(tcp_dir / arm / "CALLS.json") if row["step"] >= 0]
    by_rows = {}
    for rows in (1, 2, 4):
        group = [row for row in calls if row["rows"] == rows]
        need(len(group) >= 24, "too few calibration calls at rows %d" % rows)
        by_rows[rows] = {"calls": len(group), "compute_mean_us": round(statistics.fmean(r["compute_us"] for r in group)),
                         "compute_p50_us": round(statistics.median(r["compute_us"] for r in group)),
                         "overhead_mean_us": round(statistics.fmean(r["overhead_us"] for r in group)),
                         "rpc_mean_us": round(statistics.fmean(r["rpc_us"] for r in group))}
    bytes_rate = LOGICAL_BYTES * 1_000_000 // by_rows[1]["compute_mean_us"]
    ops_rate = min(rows * LOGICAL_OPS * 1_000_000 // by_rows[rows]["compute_mean_us"] for rows in (2, 4))
    return {"arms": arms, "by_rows": by_rows, "effective_logical_bytes_per_s": bytes_rate,
            "effective_logical_ops_per_s": ops_rate, "kernel_compute_us": by_rows[1]["compute_mean_us"]}


def main():
    root = Path(sys.argv[1])
    attempt = sys.argv[2]
    template = Path(sys.argv[3])
    tag = sys.argv[4]
    source = Path(os.environ["S42_UNIFIED_REPO_ROOT"])
    sys.path.insert(0, str(source))
    from research_dev.scheduler import RuntimePhonePowerProfile
    from research_dev.scheduler._internal.types import canonical_sha256
    from research_dev.scheduler.adapters.phone_helpers import observe_usb_port
    from research_dev.scheduler.adapters.phone_transport import ADB_TCP_TRANSPORT_GENERATION

    config = read(root / "GATE_CONFIG.json")
    helper = config["helper_phone"]
    reference = config["helper_phone_qualified_reference"]
    device = helper["device_id"]
    manifest = read(config["manifest"])
    artifact = manifest["artifact_sha256"]
    evidence_dir = root / ("qualification-" + attempt)
    evidence_dir.mkdir(exist_ok=False)
    cpugpu = root / "cpugpu-evidence"
    receipt_paths = {}

    def receipt(kind, value):
        value["status"] = "PASS"
        path = evidence_dir / (kind + ".json")
        write(path, value)
        receipt_paths[kind] = str(path)

    usb = observe_usb_port("2-9.2")
    assert usb.serial == helper["serial"] and usb.negotiated_speed_mbps >= 5000
    receipt("usb-link-speed", usb.to_json())
    numerical = numerical_receipt(root / "qualification/numerical", helper, reference, cpugpu)
    numerical["packed_parent_receipt_sha256"] = sha(root.parent / "s42-pixel10pro-packed-server-20260924-v1/PACKED_PARENT.json")
    receipt("numerical-rows-1-2-4", numerical)
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
                                 "proof_sha256": sha(proof_path), "result_sha256": sha(root / run / name / "RESULT.json"),
                                 "worker": "qualified reference worker (byte-identical outputs, numerical-rows-1-2-4)"})
    # S43_SERVER_IDENTITY_DIR: the server-token-identity run to bind (default r1); a rebuilt server library needs
    # a fresh run (tools/qualify_pixel_server.py via run_chain_eval --server-identity) in a new directory.
    fresh = server_identity(root / os.environ.get("S43_SERVER_IDENTITY_DIR", "server-identity-r1"), helper, config["server"])
    receipt("server-token-identity", {"configured_worker_check": fresh, "checks": token_checks})
    timing_path = root / "tcp-calibration-r2/RESULT.json"
    timing = read(timing_path)
    assert timing["status"] == "PASS" and len(timing["calls"]) == 216 and timing["stop"]["exit_code"] == 0
    kernel = fixed_worker_kernel(cpugpu / "tcp1", helper)
    receipt("adb-forward-round-trip", {"result_sha256": sha(timing_path), "calls": 216, "rows": [1, 2, 4],
        "fixed_worker_overhead_mean_us_by_rows": {rows: row["overhead_mean_us"] for rows, row in kernel["by_rows"].items()},
        "note": "Link calibration unchanged from the previous worker (conservative); the fixed worker's own non-compute overhead is listed for reference."})
    idle_dir = root / ("idle-stop-r2" if attempt == "pre" else "idle-stop-fixed")
    idle_hash = idle_lifecycle(idle_dir, reference if attempt == "pre" else helper)
    receipt("scheduler-launched-session", {"launch_sha256": sha(root / "mechanism-r1/both-full/PHONE_READY.json"),
        "stop_sha256": sha(root / "mechanism-r1/both-full/PHONE_CLOSE.json"),
        "idle_lifecycle_sha256": idle_hash, "idle_lifecycle_run": idle_dir.name,
        "lifecycle": "AdbTcpPhoneWorkerSession; finite budget and rooted resident idle-only stop physically qualified"
                     + (" (previous worker; pre-bundle for the fixed worker's idle qualification only)" if attempt == "pre" else
                        " with the configured worker and environment")})
    median_by_rows = {batch: statistics.median(row["overhead_us"] for row in timing["calls"]
                                             if row["repeat"] >= 3 and row["tokens"] == batch) for batch in (1, 2, 4)}
    bandwidth = round(2 * (40960 - 10240) * 1e6 / (median_by_rows[4] - median_by_rows[1]))
    fixed_us = round(max(0, median_by_rows[1] / 2 - 10240 * 1e6 / bandwidth))
    rate, ops_rate = kernel["effective_logical_bytes_per_s"], kernel["effective_logical_ops_per_s"]
    cost_receipt = {"status": "PASS", "kernel_compute_us": kernel["kernel_compute_us"],
                    "logical_f16_weight_bytes_per_layer": LOGICAL_BYTES, "logical_ops_per_layer_row": LOGICAL_OPS,
                    "effective_logical_bytes_per_s": rate, "effective_logical_ops_per_s": ops_rate,
                    "kernel_by_rows": kernel["by_rows"], "kernel_arms": kernel["arms"],
                    "transfer_overhead_us_by_rows": median_by_rows,
                    "one_direction_fixed_us": fixed_us, "effective_transfer_bytes_per_s": bandwidth,
                    "note": "Measured latency of the fixed worker at server-like cadence over adb TCP (mean per layer, "
                            "first layer of each token included); logical F16 equivalent throughput, not physical memory "
                            "bandwidth. B1 sets the bytes rate, the slower of B2/B4 the ops rate. Link from the previous "
                            "worker's calibration (conservative). Active/idle phone power 4.5/0.875W is separately assumed.",
                    "kernel_source_sha256": sha(cpugpu / "tcp1/SUITE_RESULT.json"),
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
                     "effective_bytes_per_s": rate, "effective_ops_per_s": ops_rate,
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
    kernel_release = subprocess.check_output(["/usr/bin/adb", "-P", "5037", "-s", helper["serial"], "shell", "uname -r"],
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
                "phone_kernel_release": kernel_release, "phone_usb_serial": helper["serial"], "phone_usb_sysfs_device": "2-9.2"},
            "software_identity": software, "receipts": {kind: sha(path) for kind, path in receipt_paths.items()}}}
    bundle_path = evidence_dir / "PIXEL_EVIDENCE.json"
    write(bundle_path, bundle)
    shard_index = evidence_dir / "PIXEL_FFN_SHARDS.json"
    shard_hash = software["phone_shard_sha256"]
    record = {"path": "QWEN_PACKED.ffn.gguf", "parent_sha256": artifact, "shard_sha256": shard_hash,
              "layer_mask": "0000000000fc0000", "columns": 17408, "n_ff": 17408, "shard_bytes": 948389216,
              "weight_type": "MIXED_Q4_K_Q6_K", "session_id": "PIXEL10PRO0"}
    write(shard_index, {"schema": "s42-ffn-shard-index-v1", "parent_sha256": artifact, "shards": [record]})
    if attempt == "pre":
        print("PRE-BUNDLE ONLY", bundle_path)
        return
    previous = template
    # S43_TRANSPORT_IDENTITY overrides the transport identity bound into every arm (e.g. the S2a re-pinned identity
    # after a server rebuild); default = the identity every 2026-09-25 arm used.
    identity = Path(os.environ.get(
        "S43_TRANSPORT_IDENTITY",
        "/mnt/storage/s42-trace-v2-20260921-prep/TRANSPORT_QUALIFICATION_IDENTITY_PIXEL_STAGEA_COALESCED_BOTH.json"))
    for arm in ("desktop", "op15", "two-phone"):
        inputs = root / ("inputs-" + arm + "-" + attempt)
        inputs.mkdir()
        campaign, rig, models, evidence = [read(previous / (name + ".json")) for name in ("campaign", "rig", "models", "evidence")]
        rig["repo_root"] = str(source)
        campaign["campaign_id"] = "s43-two-phone-eval-" + tag + "-" + arm + "-" + attempt
        for name in ("rig", "models", "evidence"):
            campaign[name + "_manifest_path"] = str(inputs / (name + ".json"))
        campaign["selection_mode"] = "desktop-baseline" if arm == "desktop" else "energy-aware"
        evidence["transport_qualification_identity_path"] = str(inputs / "TRANSPORT_QUALIFICATION_IDENTITY.json")
        write(inputs / "TRANSPORT_QUALIFICATION_IDENTITY.json", read(identity))
        if arm == "two-phone":
            rig["helper_phones"] = [{**{k: v for k, v in worker.items() if k in (
                "device_id", "serial", "adb_port", "backend", "worker_path", "library_directories", "column_quantum",
                "max_tokens", "forward_port", "max_requests", "worker_environment", "as_root", "phone_lock_path")},
                "worker_port": worker["phone_port"], "kernel_release": kernel_release, "transport": "adb-tcp",
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
