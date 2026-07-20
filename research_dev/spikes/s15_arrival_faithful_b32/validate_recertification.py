#!/usr/bin/env python3
"""Independent validator for the post-load OP15 B32 recertification."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
REPORT = RESULTS / "report.json"
COHORT_DIR = HERE.parent / "s15_burst_cohort"
LIVE = HERE.parent / "s15_live_launcher"
sys.path.insert(0, str(COHORT_DIR))

import validate_cohort  # noqa: E402


EXPECTED_COHORT = "sha256:85e0614d31a6f35ad7d09c3b35b2ab4fa9ef2916bc4a1dcfc32d6f4c70a51ec4"
EXPECTED_INPUT = "sha256:ea1f2d47b8c6dec6096c8b95b6073181ecba6838d97741ca92ec2ceca6937858"
EXPECTED_WORKER = "sha256:2494f191515ca576cacd8c411de79fa8985f719cba2a11cd6371aa6e1024be9f"
HOST_DIR = ROOT / "build-cuda/bin"
SOURCE = ROOT / "examples/layersplit/layersplit.cpp"
LOCAL_PATHS = {
    "llama-layersplit": HOST_DIR / "llama-layersplit",
    "layersplit.cpp": SOURCE,
    "libllama-common.so.0": (HOST_DIR / "libllama-common.so.0").resolve(),
    "libllama.so.0": (HOST_DIR / "libllama.so.0").resolve(),
    "libggml.so.0": (HOST_DIR / "libggml.so.0").resolve(),
    "libggml-base.so.0": (HOST_DIR / "libggml-base.so.0").resolve(),
    "libggml-cpu.so.0": (HOST_DIR / "libggml-cpu.so.0").resolve(),
    "libggml-cuda.so.0": (HOST_DIR / "libggml-cuda.so.0").resolve(),
}


class ValidationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def strict_bytes(payload: bytes, label: str) -> object:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail(f"duplicate key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value):
        fail(f"invalid constant {value!r} in {label}")

    try:
        return json.loads(payload, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON in {label}: {exc}") from exc


def strict_object(path: Path) -> dict:
    value = strict_bytes(path.read_bytes(), str(path))
    if type(value) is not dict:
        fail(f"{path} is not an object")
    return value


def prefixed(payload: bytes, prefix: bytes, label: str) -> list[dict]:
    values = []
    for line in payload.splitlines():
        if not line.startswith(prefix):
            continue
        value = strict_bytes(line[len(prefix):], label)
        if type(value) is not dict:
            fail(f"non-object {label} record")
        values.append(value)
    return values


def thermal_ok(value: object, limit: int) -> bool:
    if type(value) is not dict or value.get("valid") is not True:
        return False
    sensors = value.get("sensors_millic")
    maximum = value.get("max_millic")
    age = value.get("sample_age_us")
    return type(sensors) is dict and bool(sensors) \
        and all(type(name) is str and name.startswith("nsphmx-")
                and type(item) is int for name, item in sensors.items()) \
        and type(maximum) is int and maximum == max(sensors.values()) and maximum <= limit \
        and type(age) is int and 0 <= age <= 1_000_000


def thermal_samples(payload: bytes) -> set[tuple[tuple[str, int], ...]]:
    samples = set()
    for line in payload.splitlines():
        if not line.startswith(b"THERMAL "):
            fail("continuous thermal stream contains an invalid record")
        sensors = {}
        for field in line[len(b"THERMAL "):].split():
            if field.count(b"=") != 1:
                fail("continuous thermal stream contains an invalid field")
            raw_name, raw_value = field.split(b"=", 1)
            try:
                name = raw_name.decode("ascii")
                value = int(raw_value)
            except (UnicodeDecodeError, ValueError) as exc:
                raise ValidationError("continuous thermal stream is not numeric ASCII") from exc
            if not name.startswith("nsphmx-") or name in sensors \
                    or value < 10_000 or value > 120_000:
                fail("continuous thermal stream sensor failed")
            sensors[name] = value
        if not sensors:
            fail("continuous thermal stream has an empty sample")
        samples.add(tuple(sorted(sensors.items())))
    if not samples:
        fail("continuous thermal stream is empty")
    return samples


def validate_placement(cert: dict) -> dict:
    if cert.get("status") != "SCHEDULED_PLACEMENT_OK" \
            or cert.get("layer_start") != 0 or cert.get("layer_end") != 8 \
            or cert.get("missing_buffer_compute_nodes") != 0:
        fail("placement certificate header failed")
    mapping = cert.get("compute_by_op_and_buffer")
    if type(mapping) is not dict or not mapping:
        fail("placement map failed")
    htp = 0
    for op, buffers in mapping.items():
        if type(op) is not str or type(buffers) is not dict or not buffers:
            fail("placement entry failed")
        for backend, count in buffers.items():
            if type(count) is not int or count <= 0:
                fail("placement count failed")
            if backend == "HTP0":
                htp += count
            elif backend != "CPU" or op != "GET_ROWS":
                fail("undeclared placement fallback")
    if htp == 0:
        fail("placement contains no HTP compute")
    return {
        "status": cert["status"],
        "layer_start": 0,
        "layer_end": 8,
        "missing_buffer_compute_nodes": 0,
        "compute_by_op_and_buffer": mapping,
    }


def validate() -> dict:
    report = strict_object(REPORT)
    cohort = validate_cohort.validate()
    prompt = strict_object(COHORT_DIR / "input_manifest.json").get("prompt_text")
    if type(prompt) is not str or not prompt:
        fail("frozen prompt failed")
    if digest(COHORT_DIR / "cohort.json") != EXPECTED_COHORT \
            or digest(COHORT_DIR / "input_manifest.json") != EXPECTED_INPUT:
        fail("frozen workload digest failed")
    if report.get("schema") != "s15-post-load-b32-recertification-v1" \
            or report.get("verdict") != "POST_LOAD_B32_ROUTE_CERTIFIED" \
            or report.get("scope") != "REAL_OP15_A6000_POST_LOAD_PROMPT_LATENCY_CORRECTNESS_PLACEMENT_ENERGY_UNKNOWN" \
            or report.get("energy_scope") != "UNKNOWN" \
            or report.get("prompt_visible_during_preflight") is not False \
            or report.get("problems") != [] \
            or report.get("processes_required") != 7 \
            or report.get("cohort_sha256") != EXPECTED_COHORT \
            or report.get("input_manifest_sha256") != EXPECTED_INPUT:
        fail("report scope failed")

    actual_hashes = {
        str(path.relative_to(RESULTS)): digest(path)
        for path in sorted(RESULTS.rglob("*")) if path.is_file() and path != REPORT
    }
    if report.get("artifact_hashes_before_report") != actual_hashes:
        fail("raw artifact hash set failed")
    expected_local = {name: digest(path) for name, path in LOCAL_PATHS.items()}
    if report.get("host_artifacts") != expected_local:
        fail("current host artifact binding failed")
    if report.get("frozen_phone_manifest_sha256") != digest(
            ROOT / "research_dev/spikes/s14_mixed_streaming_scheduler/persistence_v2/artifacts/SHA256SUMS.txt"):
        fail("frozen phone manifest binding failed")
    if report.get("deployed", {}).get("llama-layersplit") != EXPECTED_WORKER:
        fail("deployed phone worker binding failed")

    reference_rows = prefixed(
        (RESULTS / "reference.stderr.bin").read_bytes(), b"ROUTEJSON ", "reference",
    )
    reference = report.get("reference_tokens")
    if type(reference) is not list or len(reference) != 8 \
            or len(reference_rows) != 32 \
            or sorted(row.get("stream_index") for row in reference_rows) != list(range(32)) \
            or any(row.get("token_ids") != reference for row in reference_rows):
        fail("same-batch CUDA reference failed")
    observed_thermal = thermal_samples((RESULTS / "thermal_stream.log").read_bytes())
    thermal_paths = report.get("thermal_paths")
    if type(thermal_paths) is not dict or set(thermal_paths) != {
            "nsphmx-0", "nsphmx-1", "nsphmx-2", "nsphmx-3"}:
        fail("thermal sensor identity failed")

    records = report.get("records")
    if type(records) is not list or len(records) != 7:
        fail("physical process set failed")
    elapsed = []
    for index, record in enumerate(records):
        raw = RESULTS / f"process-{index}"
        if record != strict_object(raw / "process.json"):
            fail("reported process differs from its raw record")
        host_command = strict_bytes((raw / "host.command.json").read_bytes(), "host command")
        phone_command = strict_bytes((raw / "phone.command.json").read_bytes(), "phone command")
        if host_command != record.get("host_command") or phone_command != record.get("phone_command") \
                or type(host_command) is not list or "-p" in host_command \
                or prompt in host_command or record.get("prompt_in_argv") is not False:
            fail("host input visibility or command binding failed")
        pre_prompt = (raw / "host.pre_prompt.stderr.bin").read_bytes()
        host_stderr = (raw / "host.stderr.bin").read_bytes()
        if not host_stderr.startswith(pre_prompt) \
                or pre_prompt.count(b"DRIVER_INPUT_READY ") != 1 \
                or b"DRIVER_INPUT_ACCEPTED " in pre_prompt \
                or prompt.encode("utf-8") in pre_prompt:
            fail("pre-prompt byte evidence failed")
        ready = prefixed(host_stderr, b"DRIVER_INPUT_READY ", "input ready")
        accepted = prefixed(host_stderr, b"DRIVER_INPUT_ACCEPTED ", "input accepted")
        if len(ready) != 1 or ready[0] != {
                "schema": "layersplit-driver-input-v1", "max_prompt_bytes": 16384} \
                or len(accepted) != 1 or accepted[0] != {
                    "schema": "layersplit-driver-input-v1",
                    "prompt_bytes": len(prompt.encode("utf-8")),
                }:
            fail("post-load input marker failed")
        timestamps = [
            record.get("ready_observed_ns"), record.get("paid_start_ns"),
            record.get("prompt_submitted_ns"), record.get("paid_end_ns"),
        ]
        if any(type(value) is not int for value in timestamps) \
                or timestamps != sorted(timestamps):
            fail("prompt submission timeline failed")
        host_exit_ns = record.get("host_exit_observed_ns")
        phone_exit_ns = record.get("phone_exit_observed_ns")
        if type(host_exit_ns) is not int or type(phone_exit_ns) is not int \
                or timestamps[-1] != max(host_exit_ns, phone_exit_ns):
            fail("event-driven completion timestamp failed")
        measured = (timestamps[-1] - timestamps[1]) // 1000
        if measured != record.get("completion_elapsed_us") or measured <= 0 or measured > 4_000_000:
            fail("post-admission completion misses the remaining deadline budget")
        if record.get("eligible") is not True or record.get("stream_count") != 32 \
                or record.get("token_ids") != reference \
                or not thermal_ok(record.get("thermal_start"), 60_000) \
                or not thermal_ok(record.get("thermal_end"), 85_000):
            fail("process eligibility or thermal gate failed")
        if tuple(sorted(record["thermal_start"]["sensors_millic"].items())) not in observed_thermal \
                or tuple(sorted(record["thermal_end"]["sensors_millic"].items())) not in observed_thermal:
            fail("reported thermal vector is absent from the continuous stream")

        rows = prefixed(host_stderr, b"ROUTEJSON ", "route")
        certs = prefixed(
            (raw / "phone.stdout.bin").read_bytes()
            + (raw / "phone.stderr.bin").read_bytes(),
            b"PLACEMENTCERT ", "placement",
        )
        if len(rows) != 32 or len(certs) != 1 \
                or sorted(row.get("stream_index") for row in rows) != list(range(32)) \
                or any(row.get("status") != "ok" or row.get("batch_size") != 32
                       or row.get("generated_tokens") != 8 or row.get("token_ids") != reference
                       for row in rows):
            fail("physical route correctness failed")
        if record.get("placement") != validate_placement(certs[0]) \
                or record.get("route_wall_us_max") != max(row["request_wall_us"] for row in rows):
            fail("physical placement or route timing projection failed")
        elapsed.append(measured)

    cov = statistics.pstdev(elapsed) / statistics.mean(elapsed)
    if report.get("completion_elapsed_us_by_process") != elapsed \
            or report.get("completion_p50_us") != statistics.median(elapsed) \
            or report.get("completion_conservative_us") != max(elapsed) \
            or report.get("completion_cov") != cov or cov > 0.05 \
            or report.get("remaining_deadline_budget_us") != 4_000_000:
        fail("profile summary failed")
    identity = report.get("profile_identity")
    if type(identity) is not dict or identity.get("host_artifacts") != expected_local \
            or identity.get("phone_worker_sha256") != EXPECTED_WORKER \
            or identity.get("device_id") != "op15:3C15AU002CL00000" \
            or identity.get("cohort_sha256") != EXPECTED_COHORT \
            or identity.get("input_manifest_sha256") != EXPECTED_INPUT \
            or identity.get("layer_range") != [0, 8] or identity.get("batch") != 32 \
            or identity.get("n_gen") != 8 \
            or identity.get("completion_elapsed_us_by_process") != elapsed:
        fail("profile identity failed")
    if report.get("profile_id") != "sha256:" + hashlib.sha256(canonical(identity)).hexdigest():
        fail("profile content address failed")
    if cohort["admission_schedule"]["earliest_deadline_us"] \
            - cohort["admission_schedule"]["planned_launch_us"] != 4_000_000:
        fail("cohort remaining deadline derivation failed")
    return report


def main() -> int:
    report = validate()
    print(
        "VALID_POST_LOAD_B32 "
        f"p50_us={report['completion_p50_us']} "
        f"conservative_us={report['completion_conservative_us']} "
        f"profile_id={report['profile_id']} energy=UNKNOWN"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
