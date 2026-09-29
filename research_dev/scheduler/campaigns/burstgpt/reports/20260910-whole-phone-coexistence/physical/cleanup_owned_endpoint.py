"""Recover only the exact endpoint recorded by the failed diagnostic cleanup."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.repo))
    from research_dev.scheduler.adapters.android_llama_server import (
        AndroidLlamaServerProcessConfiguration, AndroidLlamaServerProcessLauncher,
    )
    from research_dev.scheduler.adapters.probes import AndroidProcessIdentity, parse_android_process_identity

    samples = json.loads((args.attempt / "WHOLE_TELEMETRY.json").read_text())
    allocation, = samples[-1]["resident_allocations"]
    expected = AndroidProcessIdentity(**allocation["process_identity"])
    execution = json.loads((args.attempt / "WHOLE_EXECUTION.json").read_text())
    parameters = execution["command"]["adapter_parameters"]
    remote_root = "/data/local/tmp/" + args.attempt.parent.name + "/" + args.attempt.name + "-whole"
    pid_file = remote_root + "/large-model-2-physical-op15-phone.pid"
    launcher = AndroidLlamaServerProcessLauncher(AndroidLlamaServerProcessConfiguration(
        Path("/usr/bin/adb"), "3C15AU002CL00000", 5037, parameters["remote_server_path"],
        parameters["remote_library_directory"],
        {allocation["artifact_sha256"]: parameters["remote_model_path"]}, remote_root,
        parameters["executable_device"], args.attempt,
    ))
    before = launcher._su(launcher._process_identity_command(expected.process_id, pid_file)).stdout
    if parse_android_process_identity(before, expected.process_id) != expected:
        raise RuntimeError("Recorded process identity no longer matches; not signalling")
    if launcher._remote_sha256(parameters["remote_server_path"]) != parameters["remote_server_sha256"]:
        raise RuntimeError("Server hash differs; not signalling")
    launcher.stop_remote(expected.process_id, pid_file, expected=expected)
    with (args.attempt / "POST_RUN_AUDIT.json").open("x") as stream:
        json.dump({
            "overall_status": "FAIL", "execution_status": "PASS", "cleanup_status": "RECOVERED",
            "reason": "Diagnostic disconnected an already-owned NCM connection before server cleanup",
            "recorded_identity": allocation["process_identity"], "verified_before_stop": before,
            "recovery": "Identity-checked TERM of this attempt's endpoint; no other process signalled",
            "result_note": "The original RESULT.json records execution only; it preceded failed cleanup",
        }, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("OWNED_ENDPOINT_CLEANUP_VERIFIED", expected.process_id)


if __name__ == "__main__":
    main()
