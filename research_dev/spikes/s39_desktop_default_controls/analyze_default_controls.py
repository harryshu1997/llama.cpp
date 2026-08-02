#!/usr/bin/env python3
"""Independently reduce and graph the stock-default controls."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "s39_desktop_swap_baseline"
sys.path.insert(0, str(BASE))

import analyze_campaign as cp0d_analysis  # noqa: E402
import run_desktop_baseline as baseline  # noqa: E402
import validate_contract  # noqa: E402


MODELS = ("qwen3-8b-q8_0", "qwen3-14b-q4_k_m")
COLORS = {
    "qwen3-8b-q8_0": "#2563eb",
    "qwen3-14b-q4_k_m": "#e56b00",
    "energy": "#16803c",
    "grid": "#d6d9de",
    "marker": "#7b2cbf",
    "text": "#20242a",
}


class AnalysisError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return baseline.canonical(value)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if type(value) is not dict:
        raise AnalysisError(f"{path}: expected object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return baseline.read_jsonl(path)


def verify_manifest(path: Path) -> None:
    cp0d_analysis.verify_manifest(path)


def verify_campaign_manifest(path: Path) -> None:
    manifest = path / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise AnalysisError(f"{path}: missing campaign manifest")
    seen: set[str] = set()
    for index, line in enumerate(
            manifest.read_text(encoding="ascii").splitlines()):
        fields = line.split("  ", 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise AnalysisError(f"{manifest}:{index + 1}: malformed")
        digest, name = fields
        if name in seen or name == "SHA256SUMS.txt":
            raise AnalysisError(
                f"{manifest}:{index + 1}: duplicate or recursive"
            )
        target = path / name
        if not target.is_file() or baseline.digest_file(target) != digest:
            raise AnalysisError(f"{target}: digest mismatch")
        seen.add(name)
    expected = {
        str(item.relative_to(path))
        for item in path.rglob("*")
        if item.is_file() and item != manifest
    }
    if seen != expected:
        raise AnalysisError(f"{path}: campaign manifest file set mismatch")


def validate_stock_commands(path: Path) -> dict[str, Any]:
    contract = validate_contract.load_contract()
    events = read_jsonl(path / "events.jsonl")
    starts = [row for row in events if row.get("kind") == "model_load_start"]
    ready = [row for row in events if row.get("kind") == "model_ready"]
    if not starts or len(starts) != len(ready):
        raise AnalysisError(f"{path}: model start/readiness mismatch")
    forbidden = set(contract["tuning_arguments_forbidden"])
    required = set(contract["required_server_arguments"])
    placements: list[dict[str, Any]] = []
    for start, loaded in zip(starts, ready):
        command = start.get("command")
        if type(command) is not list \
                or any(type(item) is not str for item in command):
            raise AnalysisError(f"{path}: malformed server command")
        options = {item for item in command if item.startswith("--")}
        if forbidden & options or not required <= options:
            raise AnalysisError(f"{path}: command is not stock-default")
        if loaded.get("serving_profile") != "stock_default" \
                or loaded.get("model_id") != start.get("model_id"):
            raise AnalysisError(f"{path}: serving-profile binding failed")
        matches = loaded.get("full_cuda_offload_matches")
        if type(matches) is not list or not matches:
            raise AnalysisError(f"{path}: missing CUDA placement observation")
        slots = loaded.get("props", {}).get("total_slots")
        if type(slots) is not int or slots < 1:
            raise AnalysisError(f"{path}: invalid realized slot count")
        placements.append({
            "model_id": loaded["model_id"],
            "offloaded_layers": int(matches[-1][0]),
            "slots": slots,
            "total_layers": int(matches[-1][1]),
        })
    preflight = read_json(path / "preflight.json")
    if preflight.get("serving_profile") != "stock_default" \
            or preflight.get("server_sha256") \
            != contract["server"]["sha256"]:
        raise AnalysisError(f"{path}: preflight profile mismatch")
    return {
        "load_count": len(starts),
        "placements": placements,
    }


def request_rows(
        path: Path, events: list[dict[str, Any]],
        paid_start_ns: int) -> list[dict[str, Any]]:
    frozen = read_jsonl(BASE / "DESKTOP_REQUESTS.jsonl")
    arrivals = {
        row["request_index"]: row
        for row in events if row.get("kind") == "request_arrival"
    }
    dispatches = {
        row["request_index"]: row
        for row in events if row.get("kind") == "request_dispatched"
    }
    completions = {
        row["request_index"]: row
        for row in events if row.get("kind") == "request_complete"
    }
    if len(arrivals) != 74:
        raise AnalysisError(f"{path}: arrival conservation failed")
    rows: list[dict[str, Any]] = []
    for source in frozen:
        index = source["request_index"]
        arrival = arrivals.get(index)
        scheduled_ns = paid_start_ns + source["arrival_us"] * 1000
        if arrival is None \
                or arrival.get("event_id") != source["event_id"] \
                or arrival.get("model_id") != source["model_id"] \
                or arrival.get("scheduled_t_ns") != scheduled_ns:
            raise AnalysisError(f"{path}: arrival {index} binding failed")
        if index not in completions:
            if index in dispatches:
                raise AnalysisError(f"{path}: dispatched request lacks result")
            continue
        dispatch = dispatches.get(index)
        completion = completions[index]
        if dispatch is None \
                or dispatch.get("model_id") != source["model_id"] \
                or completion.get("event_id") != source["event_id"] \
                or completion.get("model_id") != source["model_id"] \
                or completion.get("dispatch_model_id") != source["model_id"] \
                or completion.get("predicted_n") != 8 \
                or len(completion.get("tokens", [])) != 8:
            raise AnalysisError(f"{path}: completion {index} binding failed")
        dispatch_ns = dispatch["t_ns"]
        first_ns = completion["first_token_ns"]
        completion_ns = completion["completion_ns"]
        if not scheduled_ns <= dispatch_ns <= first_ns <= completion_ns:
            raise AnalysisError(f"{path}: request {index} timing failed")
        token_count, final = cp0d_analysis.parse_stream(
            path / f"stream-{index:03d}.raw"
        )
        if token_count != 8 or not final:
            raise AnalysisError(f"{path}: stream {index} failed")
        latency_ns = completion_ns - scheduled_ns
        rows.append({
            "completion_latency_ns": latency_ns,
            "completion_ns": completion_ns,
            "event_id": source["event_id"],
            "first_token_ns": first_ns,
            "model_id": source["model_id"],
            "queue_ns": dispatch_ns - scheduled_ns,
            "request_index": index,
            "service_ttft_ns": first_ns - dispatch_ns,
            "slo_met": latency_ns <= source["slo_us"] * 1000,
            "ttft_ns": first_ns - scheduled_ns,
        })
    if len(dispatches) != len(rows) or len(completions) != len(rows):
        raise AnalysisError(f"{path}: duplicate request event")
    return rows


def validate_dual(path: Path, repeat_index: int) -> dict[str, Any]:
    verify_manifest(path)
    command_result = validate_stock_commands(path)
    report = read_json(path / "dual.json")
    if report.get("schema") != "s39-stock-default-dual-v1" \
            or report.get("status") != "STOCK_DEFAULT_DUAL_REPLAY_PASS" \
            or report.get("repeat_index") != repeat_index:
        raise AnalysisError(f"{path}: dual report identity failed")
    events = read_jsonl(path / "events.jsonl")
    starts = [row for row in events if row.get("kind") == "replay_start"]
    ends = [row for row in events if row.get("kind") == "replay_end"]
    if len(starts) != 1 or len(ends) != 1:
        raise AnalysisError(f"{path}: dual replay bounds failed")
    paid_start_ns = starts[0]["t_ns"]
    paid_end_ns = ends[0]["t_ns"]
    if report.get("paid_start_ns") != paid_start_ns \
            or report.get("paid_end_ns") != paid_end_ns:
        raise AnalysisError(f"{path}: dual report bounds mismatch")
    requests = request_rows(path, events, paid_start_ns)
    if len(requests) != 74:
        raise AnalysisError(f"{path}: dual completion conservation failed")
    samples = read_jsonl(path / "resource_samples.jsonl")
    energy = cp0d_analysis.integrate_power(
        samples, paid_start_ns, paid_end_ns
    )
    if energy != report.get("energy"):
        raise AnalysisError(f"{path}: dual energy mismatch")
    swap_totals = {row["system_swap_total_bytes"] for row in samples}
    swap_used = [
        row["system_swap_total_bytes"] - row["system_swap_free_bytes"]
        for row in samples
    ]
    if len(swap_totals) != 1 or max(swap_used) > min(swap_used):
        raise AnalysisError(f"{path}: dual swap grew")
    process_after = report.get("process_after", {})
    if set(process_after) != set(MODELS) or any(
            row.get("process_swap_bytes") != 0
            for row in process_after.values()):
        raise AnalysisError(f"{path}: dual process swap failed")
    ready = report.get("ready")
    if type(ready) is not dict or set(ready) != set(MODELS):
        raise AnalysisError(f"{path}: dual readiness failed")
    if not all(row.get("serving_profile") == "stock_default"
               for row in ready.values()):
        raise AnalysisError(f"{path}: dual readiness profile mismatch")
    paid_ns = paid_end_ns - paid_start_ns
    counts = Counter(row["model_id"] for row in requests)
    slo_count = sum(row["slo_met"] for row in requests)
    frozen_switches = read_jsonl(BASE / "DESKTOP_SWITCHES.jsonl")
    markers = [
        {
            "published_ns": paid_start_ns + row["t_us"] * 1000,
            "scheduled_ns": paid_start_ns + row["t_us"] * 1000,
        }
        for row in frozen_switches
    ]
    return {
        "command_evidence": command_result,
        "completion_latency_p50_ns": cp0d_analysis.percentile(
            [row["completion_latency_ns"] for row in requests], 1, 2
        ),
        "completion_latency_p95_ns": cp0d_analysis.percentile(
            [row["completion_latency_ns"] for row in requests], 95, 100
        ),
        "energy_bracketed_request_count": 74,
        "energy_is_lower_bound": False,
        "energy_nj": energy["energy_nj"],
        "energy_per_request_mj": energy["energy_nj"] // 74 // 1_000_000,
        "energy_per_token_mj": energy["energy_nj"] // (74 * 8) // 1_000_000,
        "energy_scope": "SELECTED_GPU_BOARD_COMPLETE_RUN",
        "energy_window_ns": paid_ns,
        "combined_process_rss_bytes": sum(
            row["process_rss_bytes"] for row in process_after.values()
        ),
        "minimum_system_mem_available_bytes": min(
            row["system_mem_available_bytes"] for row in samples
        ),
        "model_request_counts": dict(counts),
        "model_throughput_milli_rps": {
            model_id: counts[model_id] * 1_000_000_000_000 // paid_ns
            for model_id in MODELS
        },
        "paid_ns": paid_ns,
        "peak_gpu_memory_used_bytes": max(
            row["gpu_memory_used_bytes"] for row in samples
        ),
        "queue_p50_ns": cp0d_analysis.percentile(
            [row["queue_ns"] for row in requests], 1, 2
        ),
        "queue_p95_ns": cp0d_analysis.percentile(
            [row["queue_ns"] for row in requests], 95, 100
        ),
        "regime": "WARM_CACHE",
        "repeat_index": repeat_index,
        "request_count": 74,
        "requests": requests,
        "resource_samples": samples,
        "slo_goodput_milli_rps": (
            slo_count * 1_000_000_000_000 // paid_ns
        ),
        "slo_met_count": slo_count,
        "stranded_request_count": 0,
        "switches": markers,
        "throughput_milli_rps": 74 * 1_000_000_000_000 // paid_ns,
        "ttft_p50_ns": cp0d_analysis.percentile(
            [row["ttft_ns"] for row in requests], 1, 2
        ),
        "ttft_p95_ns": cp0d_analysis.percentile(
            [row["ttft_ns"] for row in requests], 95, 100
        ),
        "unbracketed_completion_count": 0,
        "verdict": "PASS",
    }


def validate_swap(
        path: Path, regime: str, repeat_index: int) -> dict[str, Any]:
    command_result = validate_stock_commands(path)
    frozen_requests = read_jsonl(BASE / "DESKTOP_REQUESTS.jsonl")
    frozen_switches = read_jsonl(BASE / "DESKTOP_SWITCHES.jsonl")
    if (path / "replay.json").exists():
        result = cp0d_analysis.validate_replay(
            path, regime, repeat_index, frozen_requests, frozen_switches
        )
    else:
        result = cp0d_analysis.validate_failed_cold_replay(
            path, repeat_index, frozen_requests, frozen_switches
        )
    result["command_evidence"] = command_result
    return result


def median_int(values: list[int]) -> int:
    return int(statistics.median(values))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "completion_latency_p50_ns": median_int([
            row["completion_latency_p50_ns"] for row in rows
        ]),
        "completion_latency_p95_ns": median_int([
            row["completion_latency_p95_ns"] for row in rows
        ]),
        "energy_nj": median_int([row["energy_nj"] for row in rows]),
        "energy_per_request_mj": median_int([
            row["energy_per_request_mj"] for row in rows
        ]),
        "minimum_system_mem_available_bytes": median_int([
            row["minimum_system_mem_available_bytes"] for row in rows
        ]),
        "model_throughput_milli_rps": {
            model_id: median_int([
                row["model_throughput_milli_rps"][model_id] for row in rows
            ])
            for model_id in MODELS
        },
        "paid_ns": median_int([row["paid_ns"] for row in rows]),
        "peak_gpu_memory_used_bytes": median_int([
            row["peak_gpu_memory_used_bytes"] for row in rows
        ]),
        "process_rss_bytes": median_int([
            row.get(
                "combined_process_rss_bytes",
                row.get("peak_process_rss_bytes", 0),
            )
            for row in rows
        ]),
        "repeat_count": len(rows),
        "request_count": median_int([row["request_count"] for row in rows]),
        "slo_goodput_milli_rps": median_int([
            row["slo_goodput_milli_rps"] for row in rows
        ]),
        "slo_met_count": median_int([row["slo_met_count"] for row in rows]),
        "stranded_request_count": median_int([
            row.get("stranded_request_count", 0) for row in rows
        ]),
        "throughput_milli_rps": median_int([
            row["throughput_milli_rps"] for row in rows
        ]),
        "ttft_p50_ns": median_int([row["ttft_p50_ns"] for row in rows]),
        "ttft_p95_ns": median_int([row["ttft_p95_ns"] for row in rows]),
        "verdicts": [row["verdict"] for row in rows],
    }


def bin_run(run: dict[str, Any]) -> dict[str, Any]:
    value = cp0d_analysis.bin_run(run)
    value["label"] = run["label"]
    return value


def graph_png(
        panels: list[dict[str, Any]], path: Path, energy: bool) -> None:
    width, height = 1500, 1220
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    title_font = ImageFont.load_default(size=27)
    title = (
        "RTX 4060 Ti stock-default cumulative GPU energy"
        if energy else
        "RTX 4060 Ti stock-default model throughput"
    )
    subtitle = (
        "Selected-GPU board energy; cold failure is a bracketed lower bound"
        if energy else
        "Five-second trailing request throughput by model"
    )
    draw.text((75, 18), title, fill=COLORS["text"], font=title_font)
    draw.text((75, 57), subtitle, fill="#555555", font=font)
    left, right, top, panel_h, gap = 100, 75, 150, 245, 105
    max_duration = max(panel["duration_s"] for panel in panels)
    if energy:
        maximum = max(
            1.0, max(panel["cumulative_energy_j"][-1] for panel in panels)
        )
    else:
        maximum = max(
            1.0,
            max(
                max(values)
                for panel in panels
                for values in panel["throughput_series_rps"].values()
            ),
        )
    for panel_index, panel in enumerate(panels):
        y0 = top + panel_index * (panel_h + gap)
        y1 = y0 + panel_h
        x0, x1 = left, width - right
        draw.text(
            (x0, y0 - 62), panel["label"],
            fill=COLORS["text"], font=font,
        )
        if energy:
            qualifier = ">=" if panel["energy_is_lower_bound"] else ""
            detail = (
                f'final measured {qualifier}'
                f'{panel["cumulative_energy_j"][-1]:.1f} J'
            )
        else:
            rates = panel["model_throughput_rps"]
            detail = (
                f'8B {rates["qwen3-8b-q8_0"]:.3f} req/s | '
                f'14B {rates["qwen3-14b-q4_k_m"]:.3f} req/s | '
                f'total {panel["throughput_rps"]:.3f} req/s | '
                f'{panel["slo_met_count"]}/74 SLO'
            )
            if panel["stranded_request_count"]:
                detail += f' | {panel["stranded_request_count"]} stranded'
        draw.text((x0, y0 - 35), detail, fill="#555555", font=font)
        for tick in range(6):
            y = int(y1 - tick * panel_h / 5)
            draw.line((x0, y, x1, y), fill=COLORS["grid"], width=1)
            value = tick * maximum / 5
            text = f"{value:.0f}" if energy else f"{value:.1f}"
            draw.text((x0 - 66, y - 8), text, fill=COLORS["text"], font=font)
        for marker in panel["switches_s"]:
            x = int(x0 + (x1 - x0) * marker / max_duration)
            for y in range(y0, y1, 10):
                draw.line(
                    (x, y, x, min(y + 5, y1)),
                    fill=COLORS["marker"], width=2,
                )
        if energy:
            points = [(x0, y1)]
            points.extend(
                (
                    int(x0 + (x1 - x0) * (index + 1) / max_duration),
                    int(y1 - panel_h * value / maximum),
                )
                for index, value in enumerate(panel["cumulative_energy_j"])
            )
            draw.line(points, fill=COLORS["energy"], width=3)
        else:
            for model_id, values in panel["throughput_series_rps"].items():
                points = [
                    (
                        int(x0 + (x1 - x0) * (index + 0.5) / max_duration),
                        int(y1 - panel_h * value / maximum),
                    )
                    for index, value in enumerate(values)
                ]
                if len(points) >= 2:
                    draw.line(points, fill=COLORS[model_id], width=3)
        step = max(1, max_duration // 8)
        for second in range(0, max_duration + 1, step):
            x = int(x0 + (x1 - x0) * second / max_duration)
            draw.text((x - 12, y1 + 8), str(second),
                      fill=COLORS["text"], font=font)
    draw.text(
        (width // 2 - 90, height - 56), "time since replay start (s)",
        fill=COLORS["text"], font=font,
    )
    if energy:
        legend = [
            ("cumulative selected-GPU energy", COLORS["energy"]),
            ("model publication / target change", COLORS["marker"]),
        ]
    else:
        legend = [
            ("Qwen3-8B", COLORS["qwen3-8b-q8_0"]),
            ("Qwen3-14B", COLORS["qwen3-14b-q4_k_m"]),
            ("model publication / target change", COLORS["marker"]),
        ]
    legend_y = height - 25
    x = 220
    for label, color in legend:
        draw.line((x, legend_y, x + 24, legend_y), fill=color, width=4)
        draw.text((x + 30, legend_y - 8), label,
                  fill=COLORS["text"], font=font)
        x += 420
    image.save(path)


def graph_svg(
        panels: list[dict[str, Any]], path: Path, energy: bool) -> None:
    png_path = path.with_suffix(".png")
    graph_png(panels, png_path, energy)
    png = Image.open(png_path).convert("RGB")
    width, height = png.size
    import base64
    import io
    buffer = io.BytesIO()
    png.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<image width="{width}" height="{height}" '
        f'href="data:image/png;base64,{encoded}"/>\n</svg>\n',
        encoding="ascii",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_contract.validate()
    if args.output.exists():
        raise AnalysisError(f"output exists: {args.output}")
    args.output.mkdir(parents=True)
    campaign = args.campaign.resolve()
    verify_campaign_manifest(campaign)
    progress = read_json(campaign / "campaign_progress.json")
    phases = progress.get("phases")
    expected_names = (
        [f"swap-warm-{index}" for index in range(3)]
        + [f"swap-cold-{index}" for index in range(3)]
        + [f"dual-warm-{index}" for index in range(3)]
    )
    if type(phases) is not list \
            or [row.get("name") for row in phases] != expected_names \
            or [row.get("returncode") for row in phases] \
            != [0, 0, 0, 2, 2, 2, 0, 0, 0] \
            or any(row.get("lingering_servers") != [] for row in phases):
        raise AnalysisError("campaign phase ledger mismatch")

    warm = [
        validate_swap(campaign / f"swap-warm-{index}", "WARM_CACHE", index)
        for index in range(3)
    ]
    cold = [
        validate_swap(campaign / f"swap-cold-{index}", "COLD_NVME", index)
        for index in range(3)
    ]
    dual = [
        validate_dual(campaign / f"dual-warm-{index}", index)
        for index in range(3)
    ]
    representatives = [
        cp0d_analysis.median_run(warm),
        cp0d_analysis.median_run(cold),
        cp0d_analysis.median_run(dual),
    ]
    labels = (
        "Stock default: one-model warm swap",
        "Stock default: one-model cold-NVMe swap",
        "Stock default: two resident servers",
    )
    panels = []
    for row, label in zip(representatives, labels):
        row["label"] = label
        panels.append(bin_run(row))
    (args.output / "timeline_data.json").write_bytes(canonical({
        "panels": panels,
        "schema": "s39-stock-default-timeline-v1",
    }))
    graph_svg(panels, args.output / "throughput_timeline.svg", False)
    graph_svg(panels, args.output / "energy_timeline.svg", True)
    summary = {
        "aggregates": {
            "STOCK_DEFAULT_DUAL_WARM": aggregate(dual),
            "STOCK_DEFAULT_SWAP_COLD": aggregate(cold),
            "STOCK_DEFAULT_SWAP_WARM": aggregate(warm),
        },
        "campaign": str(campaign),
        "contract_sha256": baseline.digest_file(
            HERE / "DEFAULT_CONTROL_CONTRACT.json"
        ),
        "placements": {
            "dual": dual[0]["command_evidence"]["placements"],
            "swap": warm[0]["command_evidence"]["placements"],
        },
        "representative_repeats": {
            label: row["repeat_index"]
            for label, row in zip(labels, representatives)
        },
        "schema": "s39-stock-default-analysis-v1",
        "status": "STOCK_DEFAULT_CONTROLS_COMPLETE",
    }
    (args.output / "summary.json").write_bytes(canonical(summary))
    files = sorted(
        path for path in args.output.iterdir()
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    (args.output / "SHA256SUMS.txt").write_text(
        "".join(
            f"{baseline.digest_file(path)}  {path.name}\n"
            for path in files
        ),
        encoding="ascii",
    )
    print(json.dumps({
        "status": summary["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
            AnalysisError, cp0d_analysis.AnalysisError,
            baseline.RunError, OSError, ValueError) as exc:
        print(f"STOCK_DEFAULT_ANALYSIS_ERROR: {exc}")
        raise SystemExit(2)
