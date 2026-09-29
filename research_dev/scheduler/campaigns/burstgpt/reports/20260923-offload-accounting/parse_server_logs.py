#!/usr/bin/env python3
"""Parse the per-server llama-server stderr logs of a trace run into one JSON file.

Each `large-model-<n>-physical-<role>.stderr` is one server process (one desktop model load).
Log timestamps are `M.SS.mmm.uuu` relative to process start. The process is aligned to the trace
clock (seconds after RESULT.paid_start_ns) through file mtimes: the stderr mtime is the last write,
which happens at the last log line, and RESULT.json is written at paid_end. Accuracy is about one
second, which is enough for load durations of 15 to 110 s.

Extracted per process: role, model file, load duration (first line -> "model loaded"),
print_timing blocks (prompt eval / eval ms and tokens per task), dormant host share release /
restore lines (elapsed_us, bytes, layer_mask, host_columns), "release skipped: mixed slot policies"
counters, S41SERVERFFNSHAPE summaries and FFNCONTROL lines.

    python3 parse_server_logs.py <run dir> --out server_logs.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re

TS = re.compile(r"^(\d+)\.(\d\d)\.(\d\d\d)\.(\d\d\d) ")
DORMANT = re.compile(r"dormant_host_share phase=(\w+) layer_mask=(\d+) host_columns=(\d+) "
                     r"(released|restored)_bytes=(\d+)(?: ranges=(\d+))? elapsed_us=(\d+)")
MIXED = re.compile(r"release skipped: mixed slot policies \((\d+) so far\)")
SHAPE = re.compile(r"S41SERVERFFNSHAPE (\{.*\})")
CONTROL = re.compile(r"FFNCONTROL request=(\S+) slot=(\d+) generation=(\d+) token=(\d+) mask=(\d+) columns=(\d+)")
TIMING = re.compile(r"print_timing: id\s+(\d+) \| task\s+(\d+) \|\s+(prompt eval time|eval time|total time) =\s+([\d.]+) ms /\s+(\d+) tokens")
LAUNCH = re.compile(r"launch_slot_: id\s+(\d+) \| task\s+(\d+) \| processing task")
RELEASE = re.compile(r"slot\s+release: id\s+(\d+) \| task\s+(\d+) \| stop processing: n_tokens = (\d+)")
STOP = re.compile(r"srv\s+stop:")


def stamp(line: str):
    match = TS.match(line)
    if not match:
        return None
    minutes, seconds, millis, micros = (int(group) for group in match.groups())
    return minutes * 60 + seconds + millis / 1e3 + micros / 1e6


def parse_process(path: pathlib.Path, trace_end_s: float, result_mtime: float) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    stamps = [stamp(line) for line in lines]
    last = next((value for value in reversed(stamps) if value is not None), 0.0)
    first = next((value for value in stamps if value is not None), 0.0)
    start_s = trace_end_s - (result_mtime - os.path.getmtime(path)) - last
    name = path.name
    role = "control" if "desktop-control" in name else ("hot" if "-hot-" in name else "cold")
    record = {"file": name, "role": role, "start_s": round(start_s, 3), "end_s": round(start_s + last, 3),
              "index": int(re.search(r"large-model-(\d+)", name).group(1)),
              "model_path": None, "loaded_after_s": None, "timings": {}, "dormant": [], "mixed_skips": [],
              "shapes": [], "controls": [], "launches": [], "stops": 0}
    timing_tmp: dict = {}
    for line, at in zip(lines, stamps):
        if record["model_path"] is None and "loading model '" in line:
            record["model_path"] = line.split("loading model '")[1].rstrip("'")
        if record["loaded_after_s"] is None and "model loaded" in line and at is not None:
            record["loaded_after_s"] = round(at - first, 3)
        match = DORMANT.search(line)
        if match and at is not None:
            record["dormant"].append({
                "t_s": round(start_s + at, 3), "phase": match.group(1), "layer_mask": int(match.group(2)),
                "host_columns": int(match.group(3)), "kind": "release" if match.group(4) == "released" else "restore",
                "bytes": int(match.group(5)), "ranges": int(match.group(6)) if match.group(6) else None,
                "elapsed_us": int(match.group(7))})
            continue
        match = MIXED.search(line)
        if match and at is not None:
            record["mixed_skips"].append([round(start_s + at, 3), int(match.group(1))])
            continue
        match = SHAPE.search(line)
        if match:
            record["shapes"].append(json.loads(match.group(1)))
            continue
        match = CONTROL.search(line)
        if match and at is not None:
            record["controls"].append({"t_s": round(start_s + at, 3), "request_id": match.group(1),
                                       "slot": int(match.group(2)), "generation": int(match.group(3)),
                                       "token": int(match.group(4)), "mask": int(match.group(5)),
                                       "columns": int(match.group(6))})
            continue
        match = TIMING.search(line)
        if match and at is not None:
            slot, task, kind, ms, tokens = match.groups()
            entry = timing_tmp.setdefault(task, {"slot": int(slot), "t_s": round(start_s + at, 3)})
            key = {"prompt eval time": "prompt", "eval time": "eval", "total time": "total"}[kind]
            entry[key + "_ms"] = float(ms)
            entry[key + "_tokens"] = int(tokens)
            continue
        match = LAUNCH.search(line)
        if match and at is not None:
            record["launches"].append({"t_s": round(start_s + at, 3), "slot": int(match.group(1)),
                                       "task": int(match.group(2))})
            continue
        match = RELEASE.search(line)
        if match and at is not None:
            entry = timing_tmp.setdefault(match.group(2), {"slot": int(match.group(1))})
            entry["released_t_s"] = round(start_s + at, 3)
            entry["n_tokens"] = int(match.group(3))
            continue
        if STOP.search(line):
            record["stops"] += 1
    record["timings"] = timing_tmp
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run / "RESULT.json").read_text())
    trace_end_s = (result["paid_end_ns"] - result["paid_start_ns"]) / 1e9
    result_mtime = os.path.getmtime(args.run / "RESULT.json")
    processes = [parse_process(path, trace_end_s, result_mtime)
                 for path in sorted(args.run.glob("large-model-*.stderr"))]
    processes.sort(key=lambda row: row["index"])
    payload = {"run": str(args.run), "trace_end_s": trace_end_s, "processes": processes}
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    for row in processes:
        print(f"{row['index']:>3} {row['role']:<7} start {row['start_s']:8.1f} load {row['loaded_after_s']!s:>8} "
              f"dormant {len(row['dormant']):>3} mixed {max([c for _, c in row['mixed_skips']], default=0):>4} "
              f"timings {len(row['timings']):>3} controls {len(row['controls']):>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
