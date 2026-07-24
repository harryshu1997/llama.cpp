#!/usr/bin/env python3

import argparse
import collections
import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Any

import build_trace as base


VERSION = "s39-phone-model-switch-frequent-trace-v1"
EXPECTED_BIN_INDEX = 16_522
MAX_SUCCESSFUL_REQUESTS = 100
MAX_OFFERED_TOKENS = 50_000
TRIGGER_GAP_US = 60_000_000
MIN_TARGET_DWELL_US = 30_000_000
MIN_COLD_REQUEST_TOKENS = 256


@dataclasses.dataclass(frozen=True)
class Point:
    source_t_us: int
    source_row_id: int
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def offered_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclasses.dataclass
class Window:
    bin_index: int
    source_rows: int = 0
    points: list[Point] = dataclasses.field(default_factory=list)

    def add(self, event: Any) -> None:
        self.source_rows += 1
        if event.output_tokens == 0:
            return
        model = event.source_fields["model"]
        if model not in base.MODEL_SPECS:
            raise base.BuildError(
                f"source_row_id={event.source_row_id}: unsupported model {model!r}"
            )
        self.points.append(
            Point(
                source_t_us=event.source_t_us,
                source_row_id=event.source_row_id,
                model=model,
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
            )
        )

    @property
    def model_counts(self) -> collections.Counter[str]:
        return collections.Counter(point.model for point in self.points)

    @property
    def input_tokens(self) -> int:
        return sum(point.input_tokens for point in self.points)

    @property
    def output_tokens(self) -> int:
        return sum(point.output_tokens for point in self.points)

    @property
    def offered_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def model_transitions(self) -> int:
        return sum(
            left.model != right.model
            for left, right in zip(self.points, self.points[1:])
        )

    def switch_cycles(self) -> list[dict[str, Any]]:
        cycles = []
        index = 0
        start_us = self.bin_index * base.WINDOW_US
        while index < len(self.points):
            end = index + 1
            while (
                end < len(self.points)
                and self.points[end].model == self.points[index].model
            ):
                end += 1
            run = self.points[index:end]
            if run[0].model == "GPT-4":
                trigger_index = next(
                    (
                        offset
                        for offset in range(1, len(run))
                        if (
                            run[offset].source_t_us
                            - run[offset - 1].source_t_us
                            <= TRIGGER_GAP_US
                        )
                    ),
                    None,
                )
                if trigger_index is not None:
                    return_us = (
                        self.points[end].source_t_us
                        if end < len(self.points)
                        else start_us + base.WINDOW_US
                    )
                    trigger_us = run[trigger_index].source_t_us
                    maximum_request_tokens = max(
                        point.offered_tokens for point in run
                    )
                    dwell_us = return_us - trigger_us
                    cycles.append(
                        {
                            "start_us": run[0].source_t_us - start_us,
                            "trigger_us": trigger_us - start_us,
                            "last_cold_arrival_us": (
                                run[-1].source_t_us - start_us
                            ),
                            "return_us": return_us - start_us,
                            "has_return_request": end < len(self.points),
                            "target_dwell_us": dwell_us,
                            "request_count": len(run),
                            "input_tokens": sum(
                                point.input_tokens for point in run
                            ),
                            "output_tokens": sum(
                                point.output_tokens for point in run
                            ),
                            "maximum_request_tokens": maximum_request_tokens,
                            "viable_30s": dwell_us >= MIN_TARGET_DWELL_US,
                            "substantial_request": (
                                maximum_request_tokens
                                >= MIN_COLD_REQUEST_TOKENS
                            ),
                            "useful_handoff_cycle": (
                                dwell_us >= MIN_TARGET_DWELL_US
                                and maximum_request_tokens
                                >= MIN_COLD_REQUEST_TOKENS
                            ),
                        }
                    )
            index = end
        return cycles

    def selection_key(self) -> tuple[int, int, int, int, int, int] | None:
        counts = self.model_counts
        if (
            not self.points
            or len(self.points) > MAX_SUCCESSFUL_REQUESTS
            or counts["ChatGPT"] <= counts["GPT-4"]
            or self.offered_tokens > MAX_OFFERED_TOKENS
        ):
            return None
        cycles = self.switch_cycles()
        useful = sum(cycle["useful_handoff_cycle"] for cycle in cycles)
        if useful == 0:
            return None
        viable = sum(cycle["viable_30s"] for cycle in cycles)
        return (
            -useful,
            -viable,
            -len(cycles),
            -self.model_transitions,
            self.offered_tokens,
            self.bin_index,
        )


def select_window(
    source_path: Path,
    config: dict[str, Any],
) -> tuple[Window, int, int]:
    current: Window | None = None
    selected: Window | None = None
    selected_key: tuple[int, int, int, int, int, int] | None = None
    candidate_count = 0
    nonempty_bins = 0
    parsed_records = 0
    previous_t_us: int | None = None

    def finish(window: Window | None) -> None:
        nonlocal selected, selected_key, candidate_count, nonempty_bins
        if window is None:
            return
        nonempty_bins += 1
        key = window.selection_key()
        if key is None:
            return
        candidate_count += 1
        if selected_key is None or key < selected_key:
            selected = window
            selected_key = key

    for expected_row_id, event in enumerate(base.S8.iter_events(source_path, config)):
        parsed_records += 1
        if event.source_row_id != expected_row_id:
            raise base.BuildError(
                f"expected source_row_id={expected_row_id}, "
                f"got {event.source_row_id}"
            )
        if previous_t_us is not None and event.source_t_us < previous_t_us:
            raise base.BuildError(
                f"timestamp decreased at source_row_id={event.source_row_id}"
            )
        previous_t_us = event.source_t_us
        bin_index = event.source_t_us // base.WINDOW_US
        if current is None or current.bin_index != bin_index:
            finish(current)
            current = Window(bin_index)
        current.add(event)
    finish(current)

    if parsed_records != config["origin"]["record_count"]:
        raise base.BuildError(
            f"parsed {parsed_records} records, expected "
            f"{config['origin']['record_count']}"
        )
    if selected is None:
        raise base.BuildError("no window satisfies the frequent-switch rule")
    if selected.bin_index != EXPECTED_BIN_INDEX:
        raise base.BuildError(
            f"selected bin {selected.bin_index}, expected {EXPECTED_BIN_INDEX}"
        )
    return selected, candidate_count, nonempty_bins


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source_file.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = base.S8.validate_config(base.S8.load_json_strict(base.S8_CONFIG))

    with tempfile.TemporaryDirectory(prefix="s39_frequent_snapshot_") as temporary:
        snapshot_path = Path(temporary) / config["origin"]["filename"]
        base.S8.copy_source_snapshot(source_path, snapshot_path)
        inspection = base.S8.inspect_source(snapshot_path, config)
        selected, candidate_count, nonempty_bins = select_window(
            snapshot_path,
            config,
        )
        events = base.collect_window(snapshot_path, config, selected)

    requests = [
        base.make_request(event, config, selected.bin_index) for event in events
    ]
    requests.sort(
        key=lambda item: (
            item["t_us"],
            int(item["event_id"].rsplit(":", 1)[1]),
        )
    )
    request_path = output_dir / "requests.jsonl"
    base.write_bytes(
        request_path,
        b"".join(base.S8.canonical_json(item) + b"\n" for item in requests),
    )
    request_sha256 = base.file_sha256(request_path)

    model_paths = {
        "ChatGPT": args.gemma_model.resolve(),
        "GPT-4": args.qwen_model.resolve(),
    }
    mappings = {}
    for source_model, spec in base.MODEL_SPECS.items():
        mappings[source_model] = {
            "artifact": base.verify_model(model_paths[source_model], spec),
            "model_id": spec["model_id"],
            "planned_initial_placement": spec["planned_initial_placement"],
            "role": spec["role"],
        }
    assignment = {
        "schema_version": 1,
        "provenance": "semi_synthetic",
        "profile": "frequent_switch",
        "mapping_rule": "exact source_fields.model lookup",
        "source_trace_sha256": request_sha256,
        "mappings": mappings,
        "execution_contract": {
            "duration_us": base.WINDOW_US,
            "failed_source_requests": "skip",
            "payload": "absent_in_burstgpt_source",
            "priority": "unset",
            "deadline": "unset",
        },
    }
    assignment_path = output_dir / "model_assignment.json"
    base.write_bytes(assignment_path, base.canonical_bytes(assignment))

    cycles = selected.switch_cycles()
    useful_cycles = sum(cycle["useful_handoff_cycle"] for cycle in cycles)
    viable_cycles = sum(cycle["viable_30s"] for cycle in cycles)
    target_switches = len(cycles) + sum(
        cycle["has_return_request"] for cycle in cycles
    )
    failed = sum(item["source_fields"]["burstgpt_failed"] for item in requests)
    first_row = min(event.source_row_id for event in events)
    last_row = max(event.source_row_id for event in events)

    manifest = {
        "schema_version": 1,
        "builder_version": VERSION,
        "builder_sha256": base.file_sha256(Path(__file__).resolve()),
        "shared_builder_sha256": base.file_sha256(Path(base.__file__).resolve()),
        "s8_normalizer_sha256": base.file_sha256(base.S8_NORMALIZER),
        "s8_config_sha256": base.file_sha256(base.S8_CONFIG),
        "trace_provenance": "real",
        "model_assignment_provenance": "semi_synthetic",
        "profile": "frequent_switch",
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
                "maximize-useful-handoff-cycles-then-viable-cycles-"
                "then-cycles-transitions-minimum-load-earliest-bin-v1"
            ),
            "window_us": base.WINDOW_US,
            "eligible_request": "output_tokens > 0",
            "constraints": {
                "maximum_successful_requests": MAX_SUCCESSFUL_REQUESTS,
                "hot_strictly_exceeds_cold": True,
                "maximum_offered_tokens": MAX_OFFERED_TOKENS,
                "cold_trigger_gap_us": TRIGGER_GAP_US,
                "minimum_target_dwell_us": MIN_TARGET_DWELL_US,
                "minimum_cold_request_tokens": MIN_COLD_REQUEST_TOKENS,
            },
            "candidate_count": candidate_count,
            "nonempty_bin_count": nonempty_bins,
            "bin_index": selected.bin_index,
            "source_start_us": selected.bin_index * base.WINDOW_US,
            "source_end_us": (selected.bin_index + 1) * base.WINDOW_US,
            "source_row_first": first_row,
            "source_row_last": last_row,
            "time_scale_num": 1,
            "time_scale_den": 1,
        },
        "statistics": {
            "source_rows": len(requests),
            "successful_requests": len(selected.points),
            "failed_requests": failed,
            "successful_model_counts": dict(sorted(selected.model_counts.items())),
            "successful_input_tokens": selected.input_tokens,
            "successful_output_tokens": selected.output_tokens,
            "successful_offered_tokens": selected.offered_tokens,
            "successful_model_transitions": selected.model_transitions,
            "qualifying_switch_cycles": len(cycles),
            "viable_30s_cycles": viable_cycles,
            "useful_handoff_cycles": useful_cycles,
            "target_switches_during_arrivals": target_switches,
            "switch_cycles": cycles,
            "first_arrival_us": requests[0]["t_us"],
            "last_arrival_us": requests[-1]["t_us"],
        },
        "limitations": {
            "prompt_text": "not_present",
            "source_elapsed_time_is_slo": False,
            "minimum_target_dwell_is_provisional": True,
            "phone_qwen_execution_certified": False,
            "phone_to_server_catchup_handoff_certified": False,
        },
        "outputs": {
            "requests": base.output_record(request_path, len(requests)),
            "model_assignment": base.output_record(assignment_path, 1),
        },
    }
    manifest_path = output_dir / "manifest.json"
    base.write_bytes(manifest_path, base.canonical_bytes(manifest))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deterministic S39 frequent-switch trace"
    )
    parser.add_argument("--source-file", required=True, type=Path)
    parser.add_argument("--gemma-model", required=True, type=Path)
    parser.add_argument("--qwen-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = build(parse_args())
    except (base.BuildError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"S39_FREQUENT_BUILD_ERROR: {error}") from None
    print(json.dumps(result["statistics"], sort_keys=True))
