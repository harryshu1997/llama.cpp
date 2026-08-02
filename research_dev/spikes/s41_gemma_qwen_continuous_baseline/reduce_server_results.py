#!/usr/bin/env python3
"""Normalize S39-style server runs for the S41 paper graphs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


SCHEMA = "s41-server-baseline-normalized-v1"
COMPLETE_ENERGY_SCOPE = "SELECTED_GPU_BOARD_COMPLETE_RUN"
PREFIX_ENERGY_SCOPE = "SELECTED_GPU_BOARD_BRACKETED_PREFIX"


class ReductionError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                   sort_keys=True) + "\n"
    ).encode("ascii")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReductionError(f"{path}: invalid JSON") from error
    if type(value) is not dict:
        raise ReductionError(f"{path}: expected JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError as error:
        raise ReductionError(f"{path}: cannot read") from error
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(lines):
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReductionError(f"{path}:{index + 1}: invalid JSON") from error
        if type(value) is not dict:
            raise ReductionError(f"{path}:{index + 1}: expected object")
        rows.append(value)
    if not rows:
        raise ReductionError(f"{path}: empty JSONL")
    return rows


def require_int(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReductionError(f"{field}: expected integer >= {minimum}")
    return value


def require_string(value: Any, field: str) -> str:
    if type(value) is not str or not value:
        raise ReductionError(f"{field}: expected non-empty string")
    return value


def verify_manifest(root: Path) -> None:
    path = root / "SHA256SUMS.txt"
    if not path.is_file():
        raise ReductionError(f"{root}: missing SHA256SUMS.txt")
    seen: set[str] = set()
    for index, line in enumerate(path.read_text(encoding="ascii").splitlines()):
        fields = line.split("  ", 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ReductionError(f"{path}:{index + 1}: malformed")
        digest, name = fields
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ReductionError(f"{path}:{index + 1}: unsafe path")
        if name in seen or name == "SHA256SUMS.txt":
            raise ReductionError(f"{path}:{index + 1}: duplicate path")
        target = root / relative
        if not target.is_file() or digest_file(target) != digest:
            raise ReductionError(f"{target}: digest mismatch")
        seen.add(name)
    expected = {
        str(item.relative_to(root))
        for item in root.rglob("*")
        if item.is_file() and item != path
    }
    if seen != expected:
        raise ReductionError(f"{root}: manifest file set mismatch")


def percentile(values: list[int], numerator: int, denominator: int) -> int:
    if not values:
        raise ReductionError("percentile: no values")
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[max(1, rank) - 1]


def _event_bounds(
    events: list[dict[str, Any]],
    allow_failed_prefix: bool,
) -> tuple[int, int]:
    starts = [row for row in events if row.get("kind") == "replay_start"]
    ends = [row for row in events if row.get("kind") == "replay_end"]
    if len(starts) != 1 or len(ends) > 1:
        raise ReductionError("events: invalid replay bounds")
    start_ns = require_int(starts[0].get("t_ns"), "replay_start.t_ns", 1)
    if ends:
        end_ns = require_int(ends[0].get("t_ns"), "replay_end.t_ns", 1)
    elif allow_failed_prefix:
        completions = [
            require_int(row.get("completion_ns"), "request_complete.completion_ns",
                        1)
            for row in events if row.get("kind") == "request_complete"
        ]
        if not completions:
            raise ReductionError("events: failed prefix has no completion")
        end_ns = max(completions)
    else:
        raise ReductionError("events: missing replay_end")
    if end_ns <= start_ns:
        raise ReductionError("events: invalid paid window")
    return start_ns, end_ns


def _report(root: Path) -> dict[str, Any] | None:
    candidates = [path for path in (root / "replay.json", root / "dual.json")
                  if path.is_file()]
    if len(candidates) > 1:
        raise ReductionError(f"{root}: multiple terminal reports")
    if not candidates:
        return None
    report = read_json(candidates[0])
    schema = require_string(report.get("schema"), f"{candidates[0]}.schema")
    if not (schema.startswith("s39-") or schema.startswith("s41-")):
        raise ReductionError(f"{candidates[0]}: unsupported schema {schema}")
    return report


def _request_rows(
    events: list[dict[str, Any]],
    model_ids: tuple[str, str],
) -> tuple[list[dict[str, Any]], int]:
    arrivals = [row for row in events if row.get("kind") == "request_arrival"]
    completions = [
        row for row in events if row.get("kind") == "request_complete"
    ]
    if not arrivals:
        raise ReductionError("events: no request arrivals")
    arrival_by_index: dict[int, dict[str, Any]] = {}
    for row in arrivals:
        index = require_int(row.get("request_index"),
                            "request_arrival.request_index")
        if index in arrival_by_index:
            raise ReductionError(f"events: duplicate arrival {index}")
        model_id = require_string(
            row.get("model_id"), f"request_arrival[{index}].model_id")
        if model_id not in model_ids:
            raise ReductionError(f"events: unexpected model {model_id}")
        arrival_by_index[index] = row

    rows = []
    seen: set[int] = set()
    for row in completions:
        index = require_int(row.get("request_index"),
                            "request_complete.request_index")
        if index in seen or index not in arrival_by_index:
            raise ReductionError(f"events: invalid completion {index}")
        seen.add(index)
        arrival = arrival_by_index[index]
        model_id = require_string(
            row.get("model_id"), f"request_complete[{index}].model_id")
        if model_id != arrival.get("model_id") or model_id not in model_ids:
            raise ReductionError(f"events: completion model mismatch {index}")
        scheduled_ns = row.get("scheduled_arrival_ns")
        if scheduled_ns is None:
            scheduled_ns = arrival.get("scheduled_t_ns")
        scheduled_ns = require_int(
            scheduled_ns, f"request_complete[{index}].scheduled_arrival_ns", 1)
        first_ns = require_int(
            row.get("first_token_ns"),
            f"request_complete[{index}].first_token_ns", 1)
        completion_ns = require_int(
            row.get("completion_ns"),
            f"request_complete[{index}].completion_ns", 1)
        if not scheduled_ns <= first_ns <= completion_ns:
            raise ReductionError(f"events: invalid request timing {index}")
        tokens = row.get("tokens")
        if not isinstance(tokens, list) or not tokens or any(
                type(token) is not int for token in tokens):
            raise ReductionError(f"events: invalid output tokens {index}")
        slo_us = row.get("slo_us", arrival.get("slo_us"))
        slo_ns = require_int(
            slo_us, f"request_complete[{index}].slo_us", 1) * 1000
        rows.append({
            "completion_latency_ns": completion_ns - scheduled_ns,
            "completion_ns": completion_ns,
            "model_id": model_id,
            "output_token_count": len(tokens),
            "request_index": index,
            "slo_met": completion_ns - scheduled_ns <= slo_ns,
            "ttft_ns": first_ns - scheduled_ns,
        })
    return rows, len(arrivals)


def _integrate_energy(
    samples: list[dict[str, Any]],
    start_ns: int,
    end_ns: int,
) -> tuple[int, list[dict[str, int]]]:
    ordered = sorted(samples, key=lambda row: row.get("t_ns", -1))
    if not ordered:
        raise ReductionError("resource samples: empty")
    for index, row in enumerate(ordered):
        require_int(row.get("t_ns"), f"resource_samples[{index}].t_ns", 1)
        require_int(
            row.get("gpu_power_instant_mw"),
            f"resource_samples[{index}].gpu_power_instant_mw")
    if ordered[0]["t_ns"] > start_ns or ordered[-1]["t_ns"] < end_ns:
        raise ReductionError("resource samples: energy window not bracketed")
    energy_nj = 0
    timeline = [{"t_offset_ns": 0,
                 "cumulative_selected_gpu_board_energy_nj": 0}]
    for left, right in zip(ordered, ordered[1:]):
        lo = max(start_ns, left["t_ns"])
        hi = min(end_ns, right["t_ns"])
        if hi <= lo:
            continue
        energy_nj += left["gpu_power_instant_mw"] * (hi - lo) // 1000
        timeline.append({
            "cumulative_selected_gpu_board_energy_nj": energy_nj,
            "t_offset_ns": hi - start_ns,
        })
        if hi == end_ns:
            break
    if timeline[-1]["t_offset_ns"] != end_ns - start_ns:
        raise ReductionError("resource samples: incomplete energy integration")
    return energy_nj, timeline


def _cpu_field(samples: list[dict[str, Any]]) -> str | None:
    candidates = (
        "server_cpu_utilization_milli_pct",
        "cpu_utilization_milli_pct",
        "system_cpu_utilization_milli_pct",
        "controller_process_cpu_utilization_milli_pct",
    )
    present = [
        name for name in candidates if any(name in row for row in samples)
    ]
    if len(present) > 1:
        raise ReductionError("resource samples: ambiguous CPU utilization field")
    if not present:
        return None
    name = present[0]
    if any(name not in row for row in samples):
        raise ReductionError("resource samples: partial CPU utilization series")
    return name


def _rss_field(samples: list[dict[str, Any]]) -> str:
    candidates = (
        "server_rss_bytes",
        "combined_process_rss_bytes",
        "process_rss_bytes",
    )
    present = [name for name in candidates if all(name in row for row in samples)]
    if not present:
        raise ReductionError("resource samples: missing server RSS")
    return present[0]


def _resource_summary(
    samples: list[dict[str, Any]],
    paid_start_ns: int,
    paid_end_ns: int,
) -> dict[str, Any]:
    in_window = [
        row for row in samples
        if paid_start_ns <= require_int(row.get("t_ns"), "sample.t_ns", 1)
        <= paid_end_ns
    ]
    if not in_window:
        raise ReductionError("resource samples: no paid-window samples")
    rss_name = _rss_field(in_window)
    cpu_name = _cpu_field(in_window)
    rss_values = [
        require_int(row.get(rss_name), f"resource.{rss_name}") for row in in_window
    ]
    cpu_values = None
    if cpu_name is not None:
        cpu_values = [
            require_int(row.get(cpu_name), f"resource.{cpu_name}")
            for row in in_window
        ]
    points = []
    for index, row in enumerate(in_window):
        point = {
            "server_rss_bytes": rss_values[index],
            "t_offset_ns": row["t_ns"] - paid_start_ns,
        }
        if cpu_values is not None:
            point["cpu_utilization_milli_pct"] = cpu_values[index]
        points.append(point)
    available_values = [
        require_int(row["system_mem_available_bytes"],
                    "resource.system_mem_available_bytes")
        for row in in_window if "system_mem_available_bytes" in row
    ]
    if available_values and len(available_values) != len(in_window):
        raise ReductionError("resource samples: partial memory-available series")
    return {
        "cpu_utilization_field": cpu_name,
        "cpu_utilization_milli_pct_p50": (
            int(statistics.median(cpu_values)) if cpu_values else None
        ),
        "minimum_system_mem_available_bytes": (
            min(available_values) if available_values else None
        ),
        "peak_server_rss_bytes": (
            max(rss_values) if max(rss_values) > 0 else None
        ),
        "points": points,
        "server_rss_field": rss_name,
    }


def _switches(
    events: list[dict[str, Any]],
    paid_start_ns: int,
    paid_end_ns: int,
) -> tuple[list[dict[str, Any]], int | None, bool]:
    starts = {
        require_int(row.get("intent_index"), "switch_started.intent_index"):
        row for row in events if row.get("kind") == "switch_started"
    }
    published = {
        require_int(row.get("intent_index"), "model_published.intent_index"):
        row for row in events if row.get("kind") == "model_published"
    }
    if len(starts) != len([
            row for row in events if row.get("kind") == "switch_started"]):
        raise ReductionError("events: duplicate switch intent index")
    if len(published) != len([
            row for row in events if row.get("kind") == "model_published"]):
        raise ReductionError("events: duplicate publication intent index")
    if not set(published) <= set(starts):
        raise ReductionError("events: publication without switch intent")
    result = []
    gaps = []
    for intent_index, row in sorted(starts.items()):
        intent_ns = require_int(
            row.get("scheduled_t_ns", row.get("t_ns")),
            f"switch[{intent_index}].intent_ns", 1)
        if intent_ns > paid_end_ns:
            continue
        marker = {
            "intent_index": intent_index,
            "intent_t_offset_ns": max(0, intent_ns - paid_start_ns),
            "publication_t_offset_ns": None,
            "to_model_id": require_string(
                row.get("to_model_id"), f"switch[{intent_index}].to_model_id"),
        }
        publication = published.get(intent_index)
        if publication is not None:
            published_ns = require_int(
                publication.get("published_ns", publication.get("t_ns")),
                f"switch[{intent_index}].published_ns", 1)
            if published_ns > paid_end_ns:
                result.append(marker)
                continue
            gap_ns = published_ns - intent_ns
            if gap_ns < 0:
                raise ReductionError(f"switch[{intent_index}]: negative gap")
            reported_gap = publication.get("publication_gap_ns")
            if reported_gap is not None and require_int(
                    reported_gap, f"switch[{intent_index}].publication_gap_ns"
                    ) != gap_ns:
                raise ReductionError(f"switch[{intent_index}]: gap mismatch")
            marker["publication_t_offset_ns"] = published_ns - paid_start_ns
            gaps.append(gap_ns)
        result.append(marker)
    censored = any(
        marker["publication_t_offset_ns"] is None for marker in result)
    maximum = max(gaps) if gaps and not censored else None
    return result, maximum, censored


def _load_spans(
    events: list[dict[str, Any]],
    paid_start_ns: int,
    paid_end_ns: int,
    model_ids: tuple[str, str],
) -> list[dict[str, Any]]:
    starts = [
        row for row in events if row.get("kind") == "model_load_start"
    ]
    ready = [row for row in events if row.get("kind") == "model_ready"]
    used: set[int] = set()
    spans = []
    for start in starts:
        start_ns = require_int(start.get("t_ns"), "model_load_start.t_ns", 1)
        model_id = require_string(
            start.get("model_id"), "model_load_start.model_id")
        if model_id not in model_ids:
            raise ReductionError(f"load span: unexpected model {model_id}")
        label = start.get("label")
        matches = [
            (index, row) for index, row in enumerate(ready)
            if index not in used
            and row.get("model_id") == model_id
            and (label is None or row.get("label") == label)
            and require_int(row.get("t_ns"), "model_ready.t_ns", 1) >= start_ns
        ]
        if not matches:
            if start_ns >= paid_start_ns:
                spans.append({
                    "end_t_offset_ns": None,
                    "model_id": model_id,
                    "start_t_offset_ns": start_ns - paid_start_ns,
                })
            continue
        index, end = min(matches, key=lambda item: item[1]["t_ns"])
        used.add(index)
        end_ns = end["t_ns"]
        if end_ns < paid_start_ns or start_ns > paid_end_ns:
            continue
        spans.append({
            "end_t_offset_ns": min(end_ns, paid_end_ns) - paid_start_ns,
            "model_id": model_id,
            "start_t_offset_ns": max(start_ns, paid_start_ns) - paid_start_ns,
        })
    return spans


def _throughput_timeline(
    requests: list[dict[str, Any]],
    paid_start_ns: int,
    paid_end_ns: int,
    model_ids: tuple[str, str],
    bin_width_ns: int,
) -> list[dict[str, Any]]:
    duration_ns = paid_end_ns - paid_start_ns
    bin_count = (duration_ns + bin_width_ns - 1) // bin_width_ns
    counts = [{model_id: 0 for model_id in model_ids}
              for _ in range(bin_count)]
    for row in requests:
        offset_ns = min(
            duration_ns - 1, max(0, row["completion_ns"] - paid_start_ns))
        index = offset_ns // bin_width_ns
        counts[index][row["model_id"]] += row["output_token_count"]
    result = []
    for index, values in enumerate(counts):
        lo = index * bin_width_ns
        hi = min(duration_ns, (index + 1) * bin_width_ns)
        width = hi - lo
        result.append({
            "model_milli_tokens_per_second": {
                model_id: values[model_id] * 1_000_000_000_000 // width
                for model_id in model_ids
            },
            "t_offset_ns": hi,
        })
    return result


def reduce_run(
    root: Path,
    label: str,
    mode: str,
    repeat_index: int,
    cache_regime: str,
    model_ids: tuple[str, str],
    bin_width_ns: int = 1_000_000_000,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    if verify_hashes:
        verify_manifest(root)
    report = _report(root)
    failure = None
    failure_path = root / "failure.json"
    if failure_path.is_file():
        failure = read_json(failure_path)
    events = read_jsonl(root / "events.jsonl")
    samples = read_jsonl(root / "resource_samples.jsonl")
    paid_start_ns, paid_end_ns = _event_bounds(
        events, allow_failed_prefix=report is None)
    if report is not None and (
            report.get("paid_start_ns") != paid_start_ns
            or report.get("paid_end_ns") != paid_end_ns):
        raise ReductionError(f"{root}: report/event paid window mismatch")
    requests, offered_count = _request_rows(events, model_ids)
    completed_count = len(requests)
    if completed_count > offered_count:
        raise ReductionError(f"{root}: completions exceed arrivals")
    stranded_count = offered_count - completed_count
    paid_ns = paid_end_ns - paid_start_ns
    energy = report.get("energy") if report is not None else None
    if report is not None and type(energy) is not dict:
        raise ReductionError(f"{root}: missing energy report")
    energy_start_ns = paid_start_ns
    if energy is not None:
        energy_start_ns = require_int(
            energy.get("window_start_ns"), "energy.window_start_ns", 1)
        energy_end_ns = require_int(
            energy.get("window_end_ns"), "energy.window_end_ns", 1)
    else:
        last_sample_ns = max(
            require_int(row.get("t_ns"), "resource.t_ns", 1)
            for row in samples
        )
        bracketed = [
            row["completion_ns"] for row in requests
            if row["completion_ns"] <= last_sample_ns
        ]
        if not bracketed:
            raise ReductionError(f"{root}: no completion bracketed by energy")
        energy_end_ns = max(bracketed)
    if energy_start_ns != paid_start_ns \
            or not energy_start_ns < energy_end_ns <= paid_end_ns:
        raise ReductionError(f"{root}: energy window escapes paid window")
    energy_nj, energy_timeline = _integrate_energy(
        samples, energy_start_ns, energy_end_ns)
    if energy is not None and energy_nj != require_int(
            energy.get("energy_nj"), "energy.energy_nj"):
        raise ReductionError(f"{root}: raw/report energy mismatch")
    energy_complete = (
        stranded_count == 0
        and energy_start_ns == paid_start_ns
        and energy_end_ns == paid_end_ns
    )
    switches, maximum_gap_ns, gap_censored = _switches(
        events, paid_start_ns, paid_end_ns)
    if switches and maximum_gap_ns is None and not gap_censored:
        raise ReductionError(f"{root}: missing publication gap")
    counts = Counter(row["model_id"] for row in requests)
    tokens = Counter()
    for row in requests:
        tokens[row["model_id"]] += row["output_token_count"]
    ttft = [row["ttft_ns"] for row in requests]
    completion = [row["completion_latency_ns"] for row in requests]
    resources = _resource_summary(samples, paid_start_ns, paid_end_ns)
    if report is None and stranded_count == 0:
        if mode not in {
                "C2_GEMMA_GPU_QWEN_CPU",
                "C2_QWEN_GPU_GEMMA_CPU",
        } or switches \
                or failure is None \
                or failure.get("schema") != "s41-dual-server-failure-v1" \
                or failure.get("status") != "S41_GPU_CPU_DUAL_READY_FAILED" \
                or failure.get("error") != "dual replay grew swap" \
                or failure.get("repeat_index") != repeat_index:
            raise ReductionError(
                f"{root}: successful run lacks terminal report")
        status = "RESOURCE_FAIL_SWAP_GROWTH"
    elif report is not None:
        status = require_string(report.get("status"), "report.status")
    else:
        status = "FAIL_STRANDED_REQUESTS"
    verdict = (
        "PASS" if stranded_count == 0 and "PASS" in status
        else status
    )
    return {
        "cache_regime": require_string(cache_regime, "cache_regime"),
        "completed_output_tokens": sum(tokens.values()),
        "completed_request_count": completed_count,
        "completion_latency_p95_ns": (
            percentile(completion, 95, 100) if completion else None
        ),
        "energy_equal_work_eligible": energy_complete,
        "energy_is_lower_bound": not energy_complete,
        "gpu_energy_scope": (
            COMPLETE_ENERGY_SCOPE if energy_complete else PREFIX_ENERGY_SCOPE
        ),
        "label": require_string(label, "label"),
        "maximum_publication_gap_ns": maximum_gap_ns,
        "mode": require_string(mode, "mode"),
        "model_completed_request_counts": {
            model_id: counts[model_id] for model_id in model_ids
        },
        "model_throughput_milli_tps": {
            model_id: tokens[model_id] * 1_000_000_000_000 // paid_ns
            for model_id in model_ids
        },
        "offered_request_count": offered_count,
        "paid_ns": paid_ns,
        "peak_server_rss_bytes": resources["peak_server_rss_bytes"],
        "publication_gap_censored": gap_censored,
        "repeat_index": require_int(repeat_index, "repeat_index"),
        "selected_gpu_board_energy_nj": energy_nj,
        "server_cpu_utilization_milli_pct_p50": (
            resources["cpu_utilization_milli_pct_p50"]
        ),
        "slo_goodput_milli_rps": (
            sum(row["slo_met"] for row in requests)
            * 1_000_000_000_000 // paid_ns
        ),
        "slo_met_count": sum(row["slo_met"] for row in requests),
        "stranded_request_count": stranded_count,
        "timeline": {
            "bin_width_ns": bin_width_ns,
            "energy": energy_timeline,
            "loads": _load_spans(
                events, paid_start_ns, paid_end_ns, model_ids),
            "resources": resources["points"],
            "switches": switches,
            "throughput": _throughput_timeline(
                requests, paid_start_ns, paid_end_ns, model_ids, bin_width_ns),
        },
        "ttft_p95_ns": percentile(ttft, 95, 100) if ttft else None,
        "verdict": verdict,
    }


def parse_model(value: str) -> dict[str, str]:
    fields = value.split("=", 1)
    if len(fields) != 2 or not all(fields):
        raise ReductionError("--model must be MODEL_ID=DISPLAY_LABEL")
    return {"id": fields[0], "label": fields[1]}


def parse_run(value: str) -> tuple[str, str, int, str, Path]:
    fields = value.split("|", 4)
    if len(fields) != 5:
        raise ReductionError(
            "--run must be LABEL|MODE|REPEAT|CACHE_REGIME|RUN_DIRECTORY")
    try:
        repeat_index = int(fields[2])
    except ValueError as error:
        raise ReductionError("--run REPEAT must be an integer") from error
    return fields[0], fields[1], repeat_index, fields[3], Path(fields[4])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-ms", type=int, default=1000)
    parser.add_argument(
        "--skip-manifest-verification", action="store_true",
        help="Development-only; never use for paper evidence.",
    )
    args = parser.parse_args()
    try:
        models = [parse_model(value) for value in args.model]
        if len(models) != 2 or len({row["id"] for row in models}) != 2:
            raise ReductionError("exactly two distinct --model values required")
        model_ids = (models[0]["id"], models[1]["id"])
        if args.bin_ms < 100:
            raise ReductionError("--bin-ms must be >= 100")
        runs = []
        for raw in args.run:
            label, mode, repeat_index, cache_regime, root = parse_run(raw)
            runs.append(reduce_run(
                root.resolve(), label, mode, repeat_index, cache_regime,
                model_ids, args.bin_ms * 1_000_000,
                not args.skip_manifest_verification,
            ))
        result = {
            "energy_claim": (
                "Selected GPU board only; not server-wall or "
                "total-system energy."
            ),
            "models": models,
            "runs": runs,
            "schema": SCHEMA,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(result))
        print(args.output)
    except (OSError, ReductionError) as error:
        print(f"ERROR: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
