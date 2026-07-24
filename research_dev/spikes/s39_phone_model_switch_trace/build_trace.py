#!/usr/bin/env python3

import argparse
import collections
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any


VERSION = "s39-phone-model-switch-trace-v1"
WINDOW_US = 1_200_000_000
COLD_BURST_US = 120_000_000
EXPECTED_BIN_INDEX = 21_424
EXPECTED_CANDIDATE_COUNT = 156

HERE = Path(__file__).resolve().parent
S8_DIR = HERE.parent / "s8_operator_island_affinity"
S8_NORMALIZER = S8_DIR / "normalize_trace.py"
S8_CONFIG = S8_DIR / "configs" / "burstgpt.config.json"

MODEL_SPECS = {
    "ChatGPT": {
        "model_id": "gemma-4-12b-it-q4_0",
        "file_name": "gemma-4-12B-it-Q4_0.gguf",
        "bytes": 6_975_878_176,
        "sha256": "494518c2262a26e2a607af0e40bca11c4de5a0b108e7c21308906dbbfb1c6f8c",
        "role": "hot",
        "planned_initial_placement": "server_cuda",
    },
    "GPT-4": {
        "model_id": "qwen3-14b-q4_k_m",
        "file_name": "Qwen3-14B-Q4_K_M.gguf",
        "bytes": 9_001_752_960,
        "sha256": "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
        "role": "cold",
        "planned_initial_placement": "phone_resident",
    },
}


class BuildError(RuntimeError):
    pass


def load_s8() -> Any:
    spec = importlib.util.spec_from_file_location("s8_normalize_trace", S8_NORMALIZER)
    if spec is None or spec.loader is None:
        raise BuildError(f"cannot import {S8_NORMALIZER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


S8 = load_s8()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def write_bytes(path: Path, raw: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def output_record(path: Path, records: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "records": records,
        "sha256": file_sha256(path),
    }


@dataclasses.dataclass
class Window:
    bin_index: int
    source_rows: int = 0
    successful_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_counts: collections.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )
    first_success_model: str | None = None
    first_cold_position: int | None = None
    previous_success_model: str | None = None
    model_transitions: int = 0
    cold_episodes: int = 0
    cold_by_session: dict[str, deque[tuple[int, int]]] = dataclasses.field(
        default_factory=dict
    )
    best_cold_burst: tuple[tuple[int, int], ...] = ()
    best_cold_session: str | None = None

    @property
    def offered_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, event: Any) -> None:
        self.source_rows += 1
        if event.output_tokens == 0:
            return

        model = event.source_fields["model"]
        if model not in MODEL_SPECS:
            raise BuildError(
                f"source_row_id={event.source_row_id}: unsupported model {model!r}"
            )
        if self.first_success_model is None:
            self.first_success_model = model
        if model == "GPT-4" and self.first_cold_position is None:
            self.first_cold_position = self.successful_requests
        if (
            self.previous_success_model is not None
            and model != self.previous_success_model
        ):
            self.model_transitions += 1
        if model == "GPT-4" and self.previous_success_model != "GPT-4":
            self.cold_episodes += 1
        self.previous_success_model = model

        self.successful_requests += 1
        self.input_tokens += event.input_tokens
        self.output_tokens += event.output_tokens
        self.model_counts[model] += 1

        if model != "GPT-4" or event.session_id is None:
            return
        queue = self.cold_by_session.setdefault(event.session_id, deque())
        queue.append((event.source_t_us, event.source_row_id))
        while event.source_t_us - queue[0][0] >= COLD_BURST_US:
            queue.popleft()
        candidate = tuple(queue)
        if len(candidate) > len(self.best_cold_burst):
            self.best_cold_burst = candidate
            self.best_cold_session = event.session_id

    def is_candidate(self) -> bool:
        hot = self.model_counts["ChatGPT"]
        cold = self.model_counts["GPT-4"]
        return (
            self.successful_requests <= 100
            and hot >= 10
            and cold >= 3
            and hot > cold
            and self.first_success_model == "ChatGPT"
            and self.first_cold_position is not None
            and self.first_cold_position >= 3
            and len(self.best_cold_burst) >= 3
        )


def select_window(source_path: Path, config: dict[str, Any]) -> tuple[Window, int, int]:
    current: Window | None = None
    selected: Window | None = None
    candidate_count = 0
    nonempty_bins = 0
    previous_t_us: int | None = None
    parsed_records = 0

    def finish(window: Window | None) -> None:
        nonlocal selected, candidate_count, nonempty_bins
        if window is None:
            return
        nonempty_bins += 1
        if not window.is_candidate():
            return
        candidate_count += 1
        if selected is None or (window.offered_tokens, window.bin_index) < (
            selected.offered_tokens,
            selected.bin_index,
        ):
            selected = window

    for expected_row_id, event in enumerate(S8.iter_events(source_path, config)):
        parsed_records += 1
        if event.source_row_id != expected_row_id:
            raise BuildError(
                f"expected source_row_id={expected_row_id}, got {event.source_row_id}"
            )
        if previous_t_us is not None and event.source_t_us < previous_t_us:
            raise BuildError(f"timestamp decreased at source_row_id={event.source_row_id}")
        previous_t_us = event.source_t_us
        bin_index = event.source_t_us // WINDOW_US
        if current is None or current.bin_index != bin_index:
            finish(current)
            current = Window(bin_index)
        current.add(event)
    finish(current)

    if parsed_records != config["origin"]["record_count"]:
        raise BuildError(
            f"parsed {parsed_records} records, expected "
            f"{config['origin']['record_count']}"
        )
    if selected is None:
        raise BuildError("no window satisfies the frozen selection rule")
    if selected.bin_index != EXPECTED_BIN_INDEX:
        raise BuildError(
            f"selected bin {selected.bin_index}, expected {EXPECTED_BIN_INDEX}"
        )
    if candidate_count != EXPECTED_CANDIDATE_COUNT:
        raise BuildError(
            f"candidate count {candidate_count}, expected {EXPECTED_CANDIDATE_COUNT}"
        )
    return selected, candidate_count, nonempty_bins


def collect_window(
    source_path: Path,
    config: dict[str, Any],
    selected: Window,
) -> list[Any]:
    events = []
    for event in S8.iter_events(source_path, config):
        bin_index = event.source_t_us // WINDOW_US
        if bin_index < selected.bin_index:
            continue
        if bin_index > selected.bin_index:
            break
        events.append(event)
    if not events:
        raise BuildError("selected window is empty")
    return events


def make_request(event: Any, config: dict[str, Any], bin_index: int) -> dict[str, Any]:
    t_us = event.source_t_us - bin_index * WINDOW_US
    if not 0 <= t_us < WINDOW_US:
        raise BuildError(f"source_row_id={event.source_row_id}: outside selected window")
    record = {
        "schema_version": 1,
        "event_id": f"{config['source']}:{event.source_row_id}",
        "source": config["source"],
        "provenance": config["provenance"],
        "t_us": t_us,
        "service": event.service,
        "model_class": event.model_class,
        "session_id": event.session_id,
        "input_tokens": event.input_tokens,
        "output_tokens": event.output_tokens,
        "images": config["gate_a"]["images"],
        "audio_ms": config["gate_a"]["audio_ms"],
        "retrieved_chunks": event.retrieved_chunks,
        "cache_keys": event.cache_keys,
        "observed_latency_us": config["gate_a"]["observed_latency_us"],
        "priority_class": config["gate_a"]["priority_class"],
        "deadline_us": config["gate_a"]["deadline_us"],
        "priority_provenance": config["gate_a"]["priority_provenance"],
        "deadline_provenance": config["gate_a"]["deadline_provenance"],
        "source_fields": event.source_fields,
    }
    S8.validate_record(record)
    return record


def verify_model(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if path.name != spec["file_name"]:
        raise BuildError(f"expected model file {spec['file_name']}, got {path.name}")
    size = path.stat().st_size
    if size != spec["bytes"]:
        raise BuildError(f"{path}: bytes {size}, expected {spec['bytes']}")
    digest = file_sha256(path)
    if digest != spec["sha256"]:
        raise BuildError(f"{path}: SHA-256 {digest}, expected {spec['sha256']}")
    return {
        "file_name": path.name,
        "bytes": size,
        "sha256": digest,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source_file.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = S8.validate_config(S8.load_json_strict(S8_CONFIG))
    with tempfile.TemporaryDirectory(prefix="s39_source_snapshot_") as temporary:
        snapshot_path = Path(temporary) / config["origin"]["filename"]
        S8.copy_source_snapshot(source_path, snapshot_path)
        inspection = S8.inspect_source(snapshot_path, config)
        selected, candidate_count, nonempty_bins = select_window(
            snapshot_path,
            config,
        )
        events = collect_window(snapshot_path, config, selected)
    requests = [make_request(event, config, selected.bin_index) for event in events]
    requests.sort(
        key=lambda item: (
            item["t_us"],
            int(item["event_id"].rsplit(":", 1)[1]),
        )
    )

    request_bytes = b"".join(S8.canonical_json(item) + b"\n" for item in requests)
    request_path = output_dir / "requests.jsonl"
    write_bytes(request_path, request_bytes)
    request_sha256 = file_sha256(request_path)

    model_paths = {
        "ChatGPT": args.gemma_model.resolve(),
        "GPT-4": args.qwen_model.resolve(),
    }
    mappings = {}
    for source_model, spec in MODEL_SPECS.items():
        mappings[source_model] = {
            "artifact": verify_model(model_paths[source_model], spec),
            "model_id": spec["model_id"],
            "planned_initial_placement": spec["planned_initial_placement"],
            "role": spec["role"],
        }
    assignment = {
        "schema_version": 1,
        "provenance": "semi_synthetic",
        "mapping_rule": "exact source_fields.model lookup",
        "source_trace_sha256": request_sha256,
        "mappings": mappings,
        "execution_contract": {
            "duration_us": WINDOW_US,
            "failed_source_requests": "skip",
            "payload": "absent_in_burstgpt_source",
            "priority": "unset",
            "deadline": "unset",
        },
    }
    assignment_path = output_dir / "model_assignment.json"
    write_bytes(assignment_path, canonical_bytes(assignment))

    failed = sum(item["source_fields"]["burstgpt_failed"] for item in requests)
    first_row = min(event.source_row_id for event in events)
    last_row = max(event.source_row_id for event in events)
    burst_start = selected.best_cold_burst[0][0] - selected.bin_index * WINDOW_US
    burst_end = selected.best_cold_burst[-1][0] - selected.bin_index * WINDOW_US
    manifest = {
        "schema_version": 1,
        "builder_version": VERSION,
        "builder_sha256": file_sha256(Path(__file__).resolve()),
        "s8_normalizer_sha256": file_sha256(S8_NORMALIZER),
        "s8_config_sha256": file_sha256(S8_CONFIG),
        "trace_provenance": "real",
        "model_assignment_provenance": "semi_synthetic",
        "source": {
            "name": config["source"],
            "url": config["origin"]["url"],
            "revision": config["origin"]["revision"],
            "license": config["origin"]["license"],
            "file_name": config["origin"]["filename"],
            "bytes": inspection.byte_count,
            "records": config["origin"]["record_count"],
            "sha256": inspection.sha256.removeprefix("sha256:"),
        },
        "selection": {
            "rule": (
                "minimum-successful-offered-tokens-then-earliest-aligned-bin-"
                "among-hot-majority-same-session-cold-burst-windows-v1"
            ),
            "window_us": WINDOW_US,
            "eligible_request": "output_tokens > 0",
            "constraints": {
                "maximum_successful_requests": 100,
                "minimum_hot_requests": 10,
                "minimum_cold_requests": 3,
                "hot_strictly_exceeds_cold": True,
                "first_success_model": "ChatGPT",
                "minimum_hot_requests_before_first_cold": 3,
                "minimum_same_session_cold_burst_requests": 3,
                "cold_burst_window_us": COLD_BURST_US,
            },
            "candidate_count": candidate_count,
            "nonempty_bin_count": nonempty_bins,
            "bin_index": selected.bin_index,
            "source_start_us": selected.bin_index * WINDOW_US,
            "source_end_us": (selected.bin_index + 1) * WINDOW_US,
            "source_row_first": first_row,
            "source_row_last": last_row,
            "time_scale_num": 1,
            "time_scale_den": 1,
        },
        "statistics": {
            "source_rows": len(requests),
            "successful_requests": selected.successful_requests,
            "failed_requests": failed,
            "successful_model_counts": dict(sorted(selected.model_counts.items())),
            "successful_input_tokens": selected.input_tokens,
            "successful_output_tokens": selected.output_tokens,
            "successful_offered_tokens": selected.offered_tokens,
            "successful_model_transitions": selected.model_transitions,
            "cold_episodes": selected.cold_episodes,
            "same_session_cold_burst": {
                "requests": len(selected.best_cold_burst),
                "start_us": burst_start,
                "end_us": burst_end,
                "session_id": selected.best_cold_session,
            },
            "first_arrival_us": requests[0]["t_us"],
            "last_arrival_us": requests[-1]["t_us"],
        },
        "limitations": {
            "prompt_text": "not_present",
            "source_elapsed_time_is_slo": False,
            "phone_qwen_execution_certified": False,
            "phone_to_server_kv_handoff_certified": False,
        },
        "outputs": {
            "requests": output_record(request_path, len(requests)),
            "model_assignment": output_record(assignment_path, 1),
        },
    }
    manifest_path = output_dir / "manifest.json"
    write_bytes(manifest_path, canonical_bytes(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deterministic S39 BurstGPT model-switch trace"
    )
    parser.add_argument("--source-file", required=True, type=Path)
    parser.add_argument("--gemma-model", required=True, type=Path)
    parser.add_argument("--qwen-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = build(parse_args())
    except (BuildError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"S39_BUILD_ERROR: {error}") from None
    print(json.dumps(result["statistics"], sort_keys=True))
