"""Frozen long-context configuration and exclusive invocation of canonical gates."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PRIOR = Path("/mnt/storage/s42-remote-resident-three-session-20260914-v1-2C2ipE")
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, "/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/validator-v2")
from src.run_direct_usb_device import execution_lock, preflight, shell
from research_dev.scheduler import RuntimeCapabilityCatalog


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def base_command():
    command = json.loads((PRIOR / "run-v1-command.json").read_text())["resolved"]
    command[1] = str(ROOT / "source/research_dev/scheduler/campaigns/burstgpt/remote_resident_gate.py")
    for option in ("--phone-remote-hash-cache", "--gguf-manifest-cache"):
        source = Path(command[command.index(option) + 1])
        target = ROOT / source.name
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(source.read_bytes())
        command[command.index(option) + 1] = str(target)
    for option in ("--observation-store-input", "--observation-source-catalog",
                   "--adaptive-observation-store-input", "--adaptive-observation-source-catalog"):
        if option in command:
            index = command.index(option)
            del command[index:index + 2]
    return command


def calibrated_catalog(command):
    path = ROOT / "calibration-v1/DESKTOP_PARENT_CALIBRATION.json"
    measured = json.loads(path.read_text())
    assert measured["status"] == "PASS"
    assert measured["launch_contract"]["context_size"] == 8192
    assert measured["launch_contract"]["gpu_layers"] == 23
    assert measured["launch_contract"]["cuda_graph_mode"] == "default"
    plans = ROOT / "calibration-v1/MEASURED_DESKTOP_BASELINE_PLANS_V1.json"
    plan = next(row for row in json.loads(plans.read_text())["plans"]
                if row["artifact_sha256"] == measured["artifact_sha256"])
    catalog = json.loads(Path(command[command.index("--capability-catalog") + 1]).read_text())
    control = next(row for row in catalog["desktop_control_profiles"]
                   if row["artifact_sha256"] == measured["artifact_sha256"])
    assert control["placement_sha256"] == measured["placement_sha256"]
    source_id = control["executor_id"]
    source = next(row for row in catalog["composite_executors"] if row["executor_id"] == source_id)
    context_resource = source["adapter_parameters"]["context_resource_id"]
    quantum = source["adapter_parameters"]["context_token_quantum"]
    related = set()
    for row in catalog["composite_executors"]:
        if row["executor_id"] == source_id or row["baseline_executor_id"] == source_id:
            related.add(row["executor_id"])
            parameters = {key: value for key, value in row["adapter_parameters"].items()
                          if not key.startswith("capacity_parent_")}
            parameters.update(plan["adapter_parameters"])
            row["adapter_parameters"] = parameters
            # The native desktop launch has new cold/hot physical evidence. Assisted
            # long-context execution and energy still require the bounded gate.
            row["maturity"] = "QUALIFIED" if row["executor_id"] == source_id else "CALIBRATION_PENDING"
            if row["executor_id"] == source_id:
                row["evidence_ids"] = [sha(path)]
    control["evidence_ids"] = [sha(path)]
    for row in catalog["resources"]:
        if row["resource_id"] == context_resource:
            row["capacity"] = 8192 // quantum
    for row in catalog["route_shape_profiles"]:
        if row["artifact_sha256"] == measured["artifact_sha256"]:
            row["maturity"] = "SHADOW"
    for row in catalog["transitions"]:
        if row.get("executor_id") in related:
            row["energy_maturity"] = "CALIBRATION_PENDING"
    catalog["catalog_id"] += ":context8192-calibration"
    validated = RuntimeCapabilityCatalog.from_json(catalog)
    write(ROOT / "CONTEXT_CATALOG.json", validated.to_json())
    write(ROOT / "CATALOG_DIFF.json", {
        "desktop_calibration_sha256": sha(path), "context_before": 2560, "context_after": 8192,
        "context_resource_capacity": 8192 // quantum, "related_executors": sorted(related),
        "operator_placement_unchanged": True, "assisted_qualification_inherited": False,
        "long_shape_energy_qualified": False,
    })


def main():
    mode = sys.argv[1]
    assert mode in {"calibrate", "configure", "preflight", "run"}
    command = base_command()
    if mode == "configure":
        calibrated_catalog(command)
        return 0
    command[command.index("--output") + 1] = str(ROOT / (
        "calibration-v1" if mode == "calibrate" else "gate-" + mode + "-v1"))
    if mode == "calibrate":
        command[1] = str(ROOT / "source/research_dev/scheduler/campaigns/burstgpt/desktop_parent_calibration.py")
        for option in ("--owner", "--remote-layer-mask", "--resident-layer-mask", "--session-id",
                       "--shard-index", "--shard-remote-dir", "--owner-timeout-ms", "--requests",
                       "--maximum-output-tokens", "--output-comparison", "--diagnostic-top-logprobs",
                       "--fallback-mode", "--capacity-context-sizes", "--recovery-kill-delay-s"):
            if option in command:
                index = command.index(option)
                del command[index:index + 2]
        command[command.index("--replay-schedule") + 1] = str(ROOT / "CALIBRATION_REPLAY.json")
        command += ["--context-size", "8192", "--preserve-desktop-placement"]
    else:
        replacements = {"--capability-catalog": str(ROOT / "CONTEXT_CATALOG.json"),
            "--desktop-baseline-plans": str(ROOT / "calibration-v1/MEASURED_DESKTOP_BASELINE_PLANS_V1.json"),
            "--phone-session-root": "/data/local/tmp/" + ROOT.name,
            "--requests": "1", "--maximum-output-tokens": "64"}
        for option, value in replacements.items():
            if option in command:
                command[command.index(option) + 1] = value
            else:
                command += [option, value]
        command += ["--prompt-file", str(ROOT / "PROMPT.txt"), "--minimum-input-tokens", "4096"]
        if mode == "preflight":
            command += ["--preflight-only"]
    candidate = json.loads(Path(
        "/mnt/storage/s42-remote-resident-phone-20260914-v3b-r9Rdsh/TRANSPORT_BOOT.json"
    ).read_text())["candidate"]
    with execution_lock():
        check = ROOT / (mode + "-idle-check")
        check.mkdir()
        boot_id = preflight(check)
        identities = shell("sha256sum /sys/kernel/notes /sys/kernel/btf/vmlinux").splitlines()
        assert [row.split()[0] for row in identities] == [candidate["identity"]["notes"], candidate["identity"]["btf"]]
        boot_path = ROOT / "RUN_BOOT.json"
        if boot_path.exists():
            assert json.loads(boot_path.read_text())["boot_id"] == boot_id
        else:
            assert mode == "calibrate"
            write(boot_path, {"boot_id": boot_id, "identity": candidate["identity"],
                "qualification_boot_id": candidate["boot_id"], "kernel_hashes_match": True})
        write(ROOT / (mode + "-command.json"), command)
        started = time.time_ns()
        result = subprocess.run(command)
        after = ROOT / (mode + "-postflight")
        after.mkdir()
        postflight = "PASS" if preflight(after) == boot_id else "FAIL"
        write(ROOT / (mode + "-launch.json"), {
            "started_epoch_ns": started, "finished_epoch_ns": time.time_ns(),
            "returncode": result.returncode, "postflight": postflight})
        return result.returncode or (0 if postflight == "PASS" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
