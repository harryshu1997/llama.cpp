#!/usr/bin/env python3
"""Bind seven matched OP12 head measurements into one runtime route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
from pathlib import Path

from power_frontier_policy import CertifiedBatchPoint
from priority_batch_runtime import RouteConfig


HERE = Path(__file__).resolve().parent
ENERGY = HERE / "energy"
WRAPPER = ENERGY / "op12_headcert.py"
STAGEB = ENERGY / "stageb_headcert.py"


class OP12ProfileError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict:
        value = {}
        for key, item in pairs:
            if key in value:
                raise OP12ProfileError(f"duplicate key {key!r} in {path}")
            value[key] = item
        return value

    try:
        data = json.loads(path.read_bytes(), object_pairs_hook=no_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise OP12ProfileError(str(exc)) from exc
    if type(data) is not dict:
        raise OP12ProfileError("profile must be an object")
    return data


def _hex(name: str, value: object) -> str:
    if type(value) is not str or len(value) != 64 \
            or any(char not in "0123456789abcdef" for char in value):
        raise OP12ProfileError(f"invalid {name}")
    return value


def _thermal(data: dict) -> None:
    thermal = data.get("thermal")
    if type(thermal) is not dict:
        raise OP12ProfileError("missing thermal evidence")
    for point, limit_key in (("start", "start_max_millic"), ("end", "end_max_millic")):
        sample = thermal.get(point)
        limit = thermal.get(limit_key)
        if type(sample) is not dict or type(limit) is not int or sample.get("valid") is not True:
            raise OP12ProfileError("invalid thermal sample")
        sensors = sample.get("sensors_millic")
        maximum = sample.get("max_millic")
        if type(sensors) is not dict or not sensors or type(maximum) is not int \
                or any(type(value) is not int for value in sensors.values()) \
                or maximum != max(sensors.values()) or maximum > limit:
            raise OP12ProfileError("thermal envelope failed")


def _placement(row: dict, requests: int, layer_end: int, reference: list[int]) -> None:
    if row.get("layer_range") != [0, layer_end] or row.get("batch") != 1 \
            or row.get("n_requests_measured") != requests \
            or row.get("placement_status") != "SCHEDULED_PLACEMENT_OK" \
            or row.get("missing_buffer_compute_nodes") != 0 \
            or row.get("host_returncode") != 0 \
            or row.get("token_match_vs_mono") is not True \
            or row.get("all_tokens_match_vs_mono") is not True:
        raise OP12ProfileError("profile row failed range, completion, placement, or correctness")
    token_sets = row.get("token_ids_by_request")
    if row.get("token_ids") != reference or type(token_sets) is not list \
            or len(token_sets) != requests or any(tokens != reference for tokens in token_sets):
        raise OP12ProfileError("stored token evidence does not match the reference")
    cert = row.get("placement_cert")
    if type(cert) is not dict or cert.get("status") != "SCHEDULED_PLACEMENT_OK" \
            or cert.get("layer_start") != 0 or cert.get("layer_end") != layer_end:
        raise OP12ProfileError("invalid placement certificate")
    htp_nodes = 0
    mapping = cert.get("compute_by_op_and_buffer")
    if type(mapping) is not dict:
        raise OP12ProfileError("missing placement map")
    for op, buffers in mapping.items():
        if type(buffers) is not dict:
            raise OP12ProfileError("invalid placement map")
        for buffer, count in buffers.items():
            if type(count) is not int or count <= 0:
                raise OP12ProfileError("invalid placement count")
            if buffer == "HTP0":
                htp_nodes += count
            elif op != "GET_ROWS":
                raise OP12ProfileError(f"undeclared CPU work {op}@{buffer}")
    if htp_nodes <= 0:
        raise OP12ProfileError("no HTP compute")


def _artifact_files(data: dict, run_id: str) -> tuple[set[str], dict[str, Path]]:
    files = data.get("artifact_files")
    if type(files) is not list or len(files) != 2:
        raise OP12ProfileError("exactly two raw logs are required")
    expected_suffixes = {".log", ".stderr"}
    seen_suffixes = set()
    paths = set()
    by_suffix = {}
    for record in files:
        if type(record) is not dict or set(record) != {"path", "bytes", "sha256"}:
            raise OP12ProfileError("invalid raw-log binding")
        relative = record["path"]
        if type(relative) is not str or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise OP12ProfileError("invalid raw-log path")
        if Path(relative).parent != Path("op12_logs") / run_id:
            raise OP12ProfileError("raw log is not owned by its run")
        path = ENERGY / relative
        if type(record["bytes"]) is not int or record["bytes"] <= 0 \
                or path.stat().st_size != record["bytes"] \
                or _sha(path) != _hex("raw-log digest", record["sha256"]):
            raise OP12ProfileError("raw-log binding mismatch")
        seen_suffixes.add(path.suffix)
        paths.add(str(path.resolve()))
        by_suffix[path.suffix] = path
    if seen_suffixes != expected_suffixes:
        raise OP12ProfileError("phone and host logs are both required")
    if len(paths) != 2:
        raise OP12ProfileError("raw logs must be distinct")
    return paths, by_suffix


def _line_objects(text: str, prefix: str) -> list[dict]:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict:
        value = {}
        for key, item in pairs:
            if key in value:
                raise OP12ProfileError(f"duplicate key {key!r} in raw {prefix.strip()} record")
            value[key] = item
        return value

    out = []
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            value = json.loads(line[len(prefix):], object_pairs_hook=no_duplicates)
        except json.JSONDecodeError as exc:
            raise OP12ProfileError(f"invalid raw {prefix.strip()} record") from exc
        if type(value) is not dict:
            raise OP12ProfileError(f"raw {prefix.strip()} record is not an object")
        out.append(value)
    return out


def _raw_evidence(files: dict[str, Path], row: dict, reference: list[int], requests: int) -> None:
    host_rows = _line_objects(files[".stderr"].read_text(encoding="utf-8", errors="strict"), "ROUTEJSON ")
    indexes = [item.get("request_index") for item in host_rows]
    if len(host_rows) != requests or any(type(index) is not int for index in indexes) \
            or sorted(indexes) != list(range(requests)):
        raise OP12ProfileError("raw host log has an incomplete request cohort")
    for item in host_rows:
        if item.get("batch_size") != 1 or item.get("token_ids") != reference:
            raise OP12ProfileError("raw host log failed batch or token correctness")
        for field in ("stage_a_us", "host_us", "request_wall_us"):
            if type(item.get(field)) is not int or item[field] <= 0:
                raise OP12ProfileError(f"raw host log has invalid {field}")
    expected = {
        "stage_a_us_p50": sorted(item["stage_a_us"] for item in host_rows)[requests // 2],
        "host_us_p50": sorted(item["host_us"] for item in host_rows)[requests // 2],
        "request_wall_us_p50": sorted(item["request_wall_us"] for item in host_rows)[requests // 2],
    }
    if any(row.get(field) != value for field, value in expected.items()):
        raise OP12ProfileError("stored timing does not match the raw host log")
    certs = _line_objects(files[".log"].read_text(encoding="utf-8", errors="strict"), "PLACEMENTCERT ")
    if len(certs) != 1 or certs[0] != row.get("placement_cert"):
        raise OP12ProfileError("stored placement certificate does not match the raw phone log")


def _check_disjoint(intervals: list[tuple[int, int]]) -> None:
    ordered = sorted(intervals)
    for previous, current in zip(ordered, ordered[1:]):
        if previous[1] > current[0]:
            raise OP12ProfileError("profile process intervals overlap")


def load_op12_head_route(paths: list[Path], route_epoch: int) -> RouteConfig:
    if type(route_epoch) is not int or route_epoch <= 0 or len(paths) != 7:
        raise OP12ProfileError("positive route epoch and exactly seven profiles are required")
    current_wrapper = _sha(WRAPPER)
    current_stageb = _sha(STAGEB)
    records = []
    artifact_digests = set()
    run_ids = set()
    pids = set()
    common = None
    layer_end = None
    boot_id = None
    intervals = []
    raw_paths = set()
    for path in paths:
        data = _load(path)
        if data.get("schema") != "s14-op12-head-placement-v1" \
                or data.get("status") != "OP12_HEAD_POINT_PASS" \
                or data.get("formal_claim") != "OP12_BATCH1_HEAD_PLACEMENT_CORRECTNESS_AND_LATENCY_POINT" \
                or data.get("device") != {"serial": "5ae7a43d", "soc": "SM8650", "hexagon": "v75", "backend": "HTP0"} \
                or data.get("gpu_uuid") != "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f" \
                or data.get("model_version") != "sha256:bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a":
            raise OP12ProfileError("profile is not a passing OP12 head point")
        if data.get("decode_fa_policy") != "AUTO_WITH_V75_FUSED_FA_DISABLED_BY_BACKEND_GATE":
            raise OP12ProfileError("profile has the wrong v75 attention policy")
        measured_range = data.get("layer_range")
        if type(measured_range) is not list or len(measured_range) != 2 \
                or measured_range[0] != 0 or measured_range[1] not in {6, 8}:
            raise OP12ProfileError("unsupported OP12 head range")
        if layer_end is None:
            layer_end = measured_range[1]
        elif layer_end != measured_range[1]:
            raise OP12ProfileError("profiles mix island ranges")
        if data.get("wrapper_sha256") != current_wrapper or data.get("stageb_source_sha256") != current_stageb:
            raise OP12ProfileError("measurement source digest is stale")
        config = data.get("measurement_config")
        if type(config) is not dict or set(config) != {
                "batch", "requests", "warmups", "n_gen", "driver_context",
                "driver_max_prefill", "prompt"} or config.get("batch") != 1 \
                or type(config.get("requests")) is not int or config["requests"] < 8 \
                or type(config.get("warmups")) is not int or config["warmups"] < 0 \
                or type(config.get("n_gen")) is not int or config["n_gen"] <= 0 \
                or type(config.get("driver_context")) is not int or config["driver_context"] <= 0 \
                or type(config.get("driver_max_prefill")) is not int or config["driver_max_prefill"] <= 0 \
                or type(config.get("prompt")) is not str or not config["prompt"]:
            raise OP12ProfileError("invalid measurement configuration")
        common_value = (
            json.dumps(config, sort_keys=True, separators=(",", ":")),
            data.get("model_version"), data.get("gpu_uuid"),
            _hex("phone binary digest", data.get("phone_binary_sha256")),
            _hex("host binary digest", data.get("host_binary_sha256")),
            json.dumps(data.get("reference_token_ids"), separators=(",", ":")),
            data.get("decode_fa_policy"),
        )
        if common is None:
            common = common_value
        elif common_value != common:
            raise OP12ProfileError("profiles do not bind the same workload and binaries")
        row = data.get("point")
        if type(row) is not dict:
            raise OP12ProfileError("missing profile row")
        shard_digest = _hex("shard digest", row.get("shard_sha256"))
        if common is not None and len(records) > 0 and shard_digest != records[0][2]:
            raise OP12ProfileError("profiles use different shard payloads")
        reference = data.get("reference_token_ids")
        if type(reference) is not list or not reference or any(type(token) is not int for token in reference):
            raise OP12ProfileError("invalid reference token stream")
        _placement(row, config["requests"], layer_end, reference)
        _thermal(data)
        run_id = data.get("run_id")
        if type(run_id) is not str or not run_id or run_id in run_ids:
            raise OP12ProfileError("profiles do not represent unique runs")
        paths_for_run, files_for_run = _artifact_files(data, run_id)
        if raw_paths.intersection(paths_for_run):
            raise OP12ProfileError("raw-log paths are reused across processes")
        raw_paths.update(paths_for_run)
        _raw_evidence(files_for_run, row, reference, config["requests"])
        identity = data.get("process_identity")
        if type(identity) is not dict or type(identity.get("pid")) is not int or identity["pid"] <= 0 \
                or type(identity.get("started_utc_us")) is not int \
                or type(identity.get("ended_utc_us")) is not int \
                or identity["started_utc_us"] >= identity["ended_utc_us"] \
                or type(identity.get("host_boot_id")) is not str or not identity["host_boot_id"]:
            raise OP12ProfileError("invalid process identity")
        if boot_id is None:
            boot_id = identity["host_boot_id"]
        elif boot_id != identity["host_boot_id"]:
            raise OP12ProfileError("profiles cross host boots")
        if identity["pid"] in pids:
            raise OP12ProfileError("profiles do not represent unique processes")
        run_ids.add(run_id)
        pids.add(identity["pid"])
        intervals.append((identity["started_utc_us"], identity["ended_utc_us"]))
        artifact_digest = "sha256:" + _sha(path)
        if artifact_digest in artifact_digests:
            raise OP12ProfileError("duplicate profile artifact")
        artifact_digests.add(artifact_digest)
        duration = row.get("request_wall_us_p50")
        if type(duration) is not int or duration <= 0:
            raise OP12ProfileError("invalid route duration")
        records.append((duration, artifact_digest, shard_digest))
    _check_disjoint(intervals)
    durations = [duration for duration, _, _ in records]
    mean = statistics.fmean(durations)
    if statistics.pstdev(durations) / mean > 0.05:
        raise OP12ProfileError("profile process CoV exceeds 0.05")
    binding = "\n".join(sorted(artifact_digests))
    digest = "sha256:" + hashlib.sha256(binding.encode("ascii")).hexdigest()
    point = CertifiedBatchPoint(
        1, sorted(durations)[len(durations) // 2],
        f"{digest}#same-batch-token-correctness-7proc",
        f"{digest}#scheduled-placement-7proc",
    )
    return RouteConfig(
        route_id=f"op12-gemma-head-0-{layer_end}",
        service_class="generation",
        model_id="gemma-4-12b-it-f16",
        island_id=f"gemma-head-0-{layer_end}",
        profile_id=digest,
        route_epoch=route_epoch,
        roofline_class="memory_bound",
        points=(point,),
        high_priority_max=0,
    )


def profile_summary(paths: list[Path], route_epoch: int) -> dict:
    route = load_op12_head_route(paths, route_epoch)
    route.validate()
    durations = []
    stage_durations = []
    bindings = []
    for path in paths:
        data = _load(path)
        durations.append(data["point"]["request_wall_us_p50"])
        stage_durations.append(data["point"]["stage_a_us_p50"])
        bindings.append({"path": str(path), "sha256": _sha(path)})
    mean = statistics.fmean(durations)
    point = route.points[0]
    return {
        "schema": "s14-op12-route-profile-v1",
        "status": "OP12_ROUTE_PROFILE_PASS",
        "formal_claim": "OP12_BATCH1_HEAD_7PROC_PLACEMENT_CORRECTNESS_AND_LATENCY",
        "route": {
            "route_id": route.route_id,
            "service_class": route.service_class,
            "model_id": route.model_id,
            "island_id": route.island_id,
            "profile_id": route.profile_id,
            "route_epoch": route.route_epoch,
            "roofline_class": route.roofline_class,
            "batch_size": point.batch_size,
            "duration_us": point.duration_us,
            "correctness_certificate_id": point.correctness_certificate_id,
            "placement_certificate_id": point.placement_certificate_id,
        },
        "n_processes": len(paths),
        "request_wall_us_p50_by_process": durations,
        "phone_stage_us_p50_by_process": stage_durations,
        "request_wall_process_cov": statistics.pstdev(durations) / mean,
        "cov_limit": 0.05,
        "source_artifacts": sorted(bindings, key=lambda item: item["path"]),
        "adapter_sha256": _sha(Path(__file__)),
        "energy_scope": "UNKNOWN",
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-epoch", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("profiles", nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists")
    summary = profile_summary([Path(path) for path in args.profiles], args.route_epoch)
    _write_json(args.output, summary)
    print(json.dumps({"status": summary["status"], "profile_id": summary["route"]["profile_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
