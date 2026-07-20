#!/usr/bin/env python3
"""Reopen and validate the S15 typed persistent physical evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
BATCH = 32
N_GEN = 8
EXPECTED_ARTIFACTS = {
    "child_command", "child_result", "child_stdout", "child_stderr",
    "session_cert", "host_placement", "tokens", "placement",
}


class ValidationError(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def parse(payload: bytes, label: str, require_canonical: bool = True) -> dict:
    if not payload or not payload.endswith(b"\n") or payload.count(b"\n") != 1:
        raise ValidationError(f"{label} is not one JSON line")

    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValidationError(f"duplicate key in {label}: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid {label}: {exc}") from exc
    if type(value) is not dict:
        raise ValidationError(f"{label} is not an object")
    if require_canonical and canonical(value) != payload:
        raise ValidationError(f"{label} is not canonical")
    return value


def prefixed(payload: bytes, prefix: bytes, label: str) -> list[tuple[bytes, dict]]:
    result = []
    for line in payload.splitlines(keepends=True):
        if line.startswith(prefix):
            result.append((line, parse(line[len(prefix):], label, False)))
    return result


def integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{label} is not an integer >= {minimum}")
    return value


def validate_phone(cert: dict, index: int, route: dict,
                   worker_identity: tuple[int, str] | None) -> tuple[int, str]:
    ending = "DETACH" if index == 1 else "STOP"
    if cert.get("schema") != "ls-stagenet-session-v2" or cert.get("proto_version") != 2 \
            or cert.get("session_id") != index or cert.get("session_end") != ending \
            or cert.get("device_boot_id") != route["device_boot_id"] \
            or [cert.get("layer_start"), cert.get("layer_end")] != route["layer_range"] \
            or cert.get("n_layer") != 48 or cert.get("expected_backend") != "HTP0" \
            or cert.get("placement_status") != "SCHEDULED_PLACEMENT_OK" \
            or cert.get("missing_buffer_compute_nodes") != 0:
        raise ValidationError("phone session identity or placement failed")
    identity = (integer(cert.get("worker_pid"), "worker_pid", 1), cert.get("worker_boot_nonce"))
    if type(identity[1]) is not str or not identity[1] \
            or worker_identity is not None and identity != worker_identity:
        raise ValidationError("phone worker identity changed")
    if cert.get("reset_applied") is not (index == 1) \
            or integer(cert.get("steps_session"), "steps_session", 1) != 384 \
            or integer(cert.get("steps_total"), "steps_total", 1) != 384 * index:
        raise ValidationError("phone reset or step sequence failed")
    mapping = cert.get("compute_by_op_and_buffer")
    if type(mapping) is not dict or not mapping:
        raise ValidationError("phone placement map is empty")
    htp_nodes = 0
    for op, buffers in mapping.items():
        if type(op) is not str or type(buffers) is not dict or not buffers:
            raise ValidationError("phone placement entry is malformed")
        for backend, count in buffers.items():
            integer(count, "phone placement count", 1)
            if backend == "HTP0":
                htp_nodes += count
            elif backend != "CPU" or op != "GET_ROWS":
                raise ValidationError("phone placement contains undeclared fallback")
    if htp_nodes == 0:
        raise ValidationError("phone placement has no HTP0 compute")
    return identity


def validate_host(value: dict, index: int, route: dict, host_pid: int | None) -> int:
    expected = {
        "schema": "layersplit-scheduled-placement-v2",
        "role": "host_tail",
        "mode": "pipedriver",
        "layer_start": route["host_tail_range"][0],
        "layer_end": route["host_tail_range"][1],
        "n_layer": route["host_tail_range"][1],
        "run_rc": 0,
        "missing_buffer_compute_nodes": 0,
        "status": "SCHEDULED_PLACEMENT_OK",
    }
    if any(type(value.get(key)) is not type(wanted) or value.get(key) != wanted
           for key, wanted in expected.items()):
        raise ValidationError(f"host placement identity failed for exchange {index}")
    pid = integer(value.get("pid"), "host pid", 1)
    if host_pid is not None and pid != host_pid:
        raise ValidationError("host PID changed")
    compute_nodes = integer(value.get("compute_nodes"), "host compute_nodes", 1)
    by_buffer = value.get("compute_by_buffer_type")
    if type(by_buffer) is not dict or set(by_buffer) != {"CUDA0"} \
            or by_buffer["CUDA0"] != compute_nodes:
        raise ValidationError("host tail was not exclusively on CUDA0")
    by_op = value.get("compute_by_op_and_buffer")
    if type(by_op) is not dict or not by_op:
        raise ValidationError("host operation placement is empty")
    if any(type(buffers) is not dict or set(buffers) != {"CUDA0"}
           for buffers in by_op.values()):
        raise ValidationError("host operation used an unexpected backend")
    return pid


def validate(report_path: Path = RESULTS / "report.json",
             results_root: Path = RESULTS) -> dict:
    report_payload = report_path.read_bytes()
    report = parse(report_payload, "report")
    if report.get("schema") != "s15-persistent-typed-physical-gate-v1" \
            or report.get("verdict") != "TYPED_PERSISTENT_OP15_B32_PHYSICAL_PASS_ENERGY_UNKNOWN" \
            or report.get("problems") != [] \
            or report.get("formal_energy_claim") != "NONE" \
            or report.get("phone_energy") != "UNKNOWN" \
            or report.get("total_system_energy") != "UNKNOWN":
        raise ValidationError("report verdict or energy scope is invalid")
    route = report.get("route")
    if type(route) is not dict or route.get("batch_size") != BATCH \
            or route.get("max_n_gen") != N_GEN or route.get("layer_range") != [0, 8] \
            or route.get("host_tail_range") != [8, 48]:
        raise ValidationError("route shape is invalid")

    file_bindings = {
        "cohort_sha256": ROOT / "research_dev/spikes/s15_burst_cohort/cohort.json",
        "input_manifest_sha256": ROOT / "research_dev/spikes/s15_burst_cohort/input_manifest.json",
        "worker_binary_sha256": ROOT / "npu-harness/build/llamacpp/android-arm64-hexagon-release-eafdc75e/bin/llama-layersplit",
        "host_binary_sha256": ROOT / "build-cuda/bin/llama-layersplit",
        "physical_mux_sha256": HERE / "physical_mux.py",
        "bridge_sha256": ROOT / "research_dev/spikes/s15_persistent_live_launcher/persistent_bridge.py",
        "live_adapter_sha256": ROOT / "research_dev/spikes/s15_persistent_live_launcher/live_adapter.py",
    }
    for field, path in file_bindings.items():
        if not path.is_file() or report.get(field) != digest_file(path):
            raise ValidationError(f"report file binding failed: {field}")
    if route.get("profile_id") != digest_file(
            ROOT / "research_dev/spikes/s15_batch32_gate/results/gate_report.json") \
            or route.get("evidence_sha256") != digest_file(
                ROOT / "research_dev/spikes/s15_persistent_host_tail/results/report.json") \
            or route.get("worker_binary_sha256") != report["worker_binary_sha256"]:
        raise ValidationError("route evidence binding failed")

    executions = report.get("executions")
    children = report.get("child_results")
    phone_sessions = report.get("phone_sessions")
    host_placements = report.get("host_placements")
    if any(type(values) is not list or len(values) != 2 for values in (
            executions, children, phone_sessions, host_placements)):
        raise ValidationError("report does not contain two complete exchanges")
    reference = report.get("reference_tokens")
    if type(reference) is not list or len(reference) != N_GEN \
            or any(type(token) is not int or token < 0 for token in reference):
        raise ValidationError("reference tokens are invalid")

    bridge_replies = []
    host_pid = None
    worker_identity = None
    for index in (1, 2):
        ending = "DETACH" if index == 1 else "STOP"
        execution = executions[index - 1]
        child = children[index - 1]
        if execution != {
                "launch_id": index, "session_end": ending,
                "finish_us": execution.get("finish_us"), "certificate_count": BATCH,
        } or integer(execution.get("finish_us"), "finish_us", 1) <= (
                executions[index - 2]["finish_us"] if index == 2 else 0):
            raise ValidationError("typed execution record is invalid")
        expected_child = {
            "schema": "layersplit-persistent-result-v1",
            "launch_id": index,
            "outcome": "completed",
            "host_pid": child.get("host_pid"),
            "request_count": BATCH,
            "batch_size": BATCH,
            "n_gen": N_GEN,
            "session_end": ending,
            "elapsed_us": child.get("elapsed_us"),
            "route_wall_us": child.get("route_wall_us"),
            "token_ids": [reference] * BATCH,
        }
        if child != expected_child \
                or integer(child.get("host_pid"), "child host pid", 1) != (
                    child["host_pid"] if host_pid is None else host_pid) \
                or integer(child.get("elapsed_us"), "child elapsed_us", 1) > 5_000_000 \
                or integer(child.get("route_wall_us"), "route_wall_us", 1) > child["elapsed_us"]:
            raise ValidationError("child result failed")
        host_pid = child["host_pid"]

        launch_dir = results_root / "bridge-child-artifacts" / f"launch-{index:06d}"
        reply_dir = results_root / "outer-artifacts" / f"exchange-{index:06d}-launch-{index}"
        reply = parse((reply_dir / "reply.json").read_bytes(), "bridge reply")
        bridge_replies.append(canonical(reply))
        bindings = reply.get("artifact_hashes")
        if type(bindings) is not dict or set(bindings) != EXPECTED_ARTIFACTS:
            raise ValidationError("bridge artifact binding set is incomplete")
        payloads = {}
        for name in EXPECTED_ARTIFACTS:
            binding = bindings[name]
            relative = f"launch-{index:06d}/{name}.bin"
            path = launch_dir / f"{name}.bin"
            payload = path.read_bytes()
            if type(binding) is not dict or binding != {
                    "path": relative, "sha256": digest_bytes(payload)}:
                raise ValidationError(f"bridge artifact binding failed: {name}")
            payloads[name] = payload
        if payloads["child_result"] != canonical(child) \
                or payloads["child_stdout"] != payloads["child_result"] \
                or parse(payloads["tokens"], "tokens") != {"token_ids": [reference] * BATCH}:
            raise ValidationError("child result/token artifact failed")
        cert_line = payloads["session_cert"]
        cert_items = prefixed(cert_line, b"SESSIONCERT ", "SESSIONCERT")
        if len(cert_items) != 1 or cert_items[0][1] != phone_sessions[index - 1]:
            raise ValidationError("phone certificate artifact differs from report")
        worker_identity = validate_phone(cert_items[0][1], index, route, worker_identity)
        placement_line = payloads["host_placement"]
        placement_items = prefixed(placement_line, b"PLACEMENTCERT ", "PLACEMENTCERT")
        if len(placement_items) != 1 or placement_items[0][1] != host_placements[index - 1]:
            raise ValidationError("host placement artifact differs from report")
        validate_host(placement_items[0][1], index, route, host_pid)
        stderr_lines = payloads["child_stderr"].splitlines(keepends=True)
        if stderr_lines.count(cert_line) != 1 or stderr_lines.count(placement_line) != 1:
            raise ValidationError("child stderr does not bind both placement records")

    outer_stdout = (results_root / "outer-artifacts/process.stdout.bin").read_bytes()
    if outer_stdout != b"".join(bridge_replies):
        raise ValidationError("outer persistent stdout differs from its replies")
    process = parse((results_root / "outer-artifacts/process.json").read_bytes(), "outer process")
    if process.get("exchanges") != 2 or process.get("poison_reason") is not None \
            or process.get("returncode") != 0:
        raise ValidationError("outer persistent process did not stop cleanly")

    mux_host = (results_root / "mux-artifacts/host.stdout.bin").read_bytes()
    if mux_host != b"".join(canonical(value) for value in children):
        raise ValidationError("mux host stdout differs from report")
    mux_host_certs = [value for _, value in prefixed(
        (results_root / "mux-artifacts/host.stderr.bin").read_bytes(),
        b"PLACEMENTCERT ", "mux host placement",
    )]
    mux_phone_certs = [value for _, value in prefixed(
        (results_root / "mux-artifacts/phone.stderr.bin").read_bytes(),
        b"SESSIONCERT ", "mux phone session",
    )]
    if mux_host_certs != host_placements or mux_phone_certs != phone_sessions:
        raise ValidationError("mux raw streams differ from report")
    for thermal_name in ("thermal_start", "thermal_end"):
        thermal = report.get(thermal_name)
        if type(thermal) is not dict or thermal.get("valid") is not True \
                or integer(thermal.get("max_millic"), thermal_name, 1) > 85_000:
            raise ValidationError("thermal evidence failed")
    manifest_path = results_root / "run_manifest.json"
    manifest = parse(manifest_path.read_bytes(), "run manifest")
    if manifest.get("schema") != "s15-persistent-typed-run-manifest-v1" \
            or manifest.get("report_sha256") != digest_bytes(report_payload) \
            or manifest.get("phone_serial") != "3C15AU002CL00000" \
            or manifest.get("phone_boot_id") != route["device_boot_id"] \
            or manifest.get("phone_shard_sha256") != \
            "sha256:a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8" \
            or manifest.get("full_model_sha256") != \
            "sha256:bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a" \
            or manifest.get("selected_gpu_uuid") != \
            "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f" \
            or manifest.get("energy_scope") != "UNKNOWN":
        raise ValidationError("run manifest identity failed")
    local = manifest.get("local_artifacts")
    if type(local) is not dict or not local:
        raise ValidationError("run manifest local artifact set is empty")
    for relative, expected in local.items():
        path = ROOT / relative
        if type(relative) is not str or type(expected) is not str \
                or not path.is_file() or digest_file(path) != expected:
            raise ValidationError(f"run manifest local artifact drift: {relative}")
    remote = manifest.get("remote_runtime")
    if type(remote) is not dict or len(remote) != 10 \
            or remote.get("llama-layersplit") != report["worker_binary_sha256"]:
        raise ValidationError("run manifest remote runtime failed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=RESULTS / "report.json")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    report = validate(args.report, args.results)
    print(
        "VALID_TYPED_PERSISTENT_GATE "
        f"host_pid={report['child_results'][0]['host_pid']} "
        f"worker_pid={report['phone_sessions'][0]['worker_pid']} sessions=2 energy=UNKNOWN"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValidationError) as exc:
        print(f"VALIDATION_ERROR {exc}")
        raise SystemExit(2)
