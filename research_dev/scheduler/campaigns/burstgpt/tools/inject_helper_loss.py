#!/usr/bin/env python3
"""SIGTERM one co-helper FFN worker of a running elastic-phones campaign (hardware fault injection).

    python3 inject_helper_loss.py --device pixel10pro-phone --when request=12 --signal TERM \
        --run-dir /path/to/campaign-output/run --authorized

Use only with the user's explicit authorization for this run (``--authorized``).

Expected layout (``launch.py``): the campaign output directory holds ``RESOLVED_CONFIGURATION.json``
and the runner's output directory ``run/``; ``--run-dir`` is that ``run/`` directory (it holds
``CO_HELPER_LIFECYCLE.json`` and ``streams/``), so the resolved configuration is read one level up
(``--resolved-configuration`` names another path). The tool

1. refuses a run whose resolved campaign does not declare ``elastic_phones`` (a static run cannot
   recover from a lost helper);
2. waits for the rig's ``CO_HELPER_LIFECYCLE.json`` and takes the device's latest ``launch``
   receipt (serial, adb port, worker path and pids; a join appends a newer one);
3. waits for the condition: ``request=NNN`` = the stream ``streams/request-NNN.raw`` holds at
   least ``--min-stream-bytes`` (the request is decoding), ``t=SECONDS`` = SECONDS after the first
   co-helper launch receipt of the run (its ``monotonic_ns``; the trace start when a helper
   started with it; this host's CLOCK_MONOTONIC, so the tool runs on the rig's host);
4. checks that every pid still runs exactly the launched worker executable
   (``readlink /proc/PID/exe`` equals the receipt's worker path) and sends ``kill -TERM`` through
   ``adb shell su -c`` (root) on the receipt's adb server port. SIGKILL is never sent;
5. writes ``FAULT_INJECTED.json`` (exclusive create) with the condition, pids and timestamps, also
   when nothing was signalled (``status`` says why).

Standard library only, so it runs from any directory on the desktop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Callable

SCHEMA = "s42-helper-fault-injection-v1"
LIFECYCLE_FILE = "CO_HELPER_LIFECYCLE.json"
RESULT_FILE = "FAULT_INJECTED.json"
RESOLVED_CONFIGURATION_FILE = "RESOLVED_CONFIGURATION.json"


class InjectionError(RuntimeError):
    pass


def parse_condition(value: str) -> tuple[str, int | float]:
    kind, separator, raw = value.partition("=")
    try:
        if separator and kind == "request":
            number = int(raw)
            if number >= 0:
                return kind, number
        if separator and kind in ("t", "active"):
            seconds = float(raw)
            if seconds >= 0:
                return kind, seconds
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("--when must be request=NNN, t=SECONDS or active=SECONDS")


def launched_layer_range(launch: dict[str, object]) -> tuple[int, int] | None:
    """The inclusive FFN layer range the receipt's worker serves (``--layers A-B``)."""
    words = _launch_shell_words(launch)
    if not words or "--layers" not in words:
        return None
    index = words.index("--layers")
    first, separator, last = (words[index + 1] if index + 1 < len(words) else "").partition("-")
    if separator and first.isdigit() and last.isdigit() and int(first) <= int(last):
        return int(first), int(last)
    return None


def helper_call_count(run_dir: Path, layers: tuple[int, int]) -> tuple[str | None, int]:
    """(newest hot-model server stderr, number of ``S41SERVERFFNCALL`` lines whose ``layer=`` lies in
    ``layers``) - the per-call log grows while the helper serves, so a growing count means "serving now"."""
    candidates = sorted(run_dir.glob("large-model-*-hot-*.stderr"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return None, 0
    path = candidates[-1]
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.startswith(b"S41SERVERFFNCALL "):
                position = line.find(b" layer=")
                if position >= 0:
                    digits = line[position + 7:].split(b" ", 1)[0]
                    if digits.isdigit() and layers[0] <= int(digits) <= layers[1]:
                        count += 1
    return str(path), count


def campaign_elastic_phones(path: Path) -> dict[str, object]:
    """The ``elastic_phones`` object of a run's resolved campaign; refuses any other run."""
    try:
        resolved = json.loads(path.read_text(encoding="ascii"))
    except (OSError, ValueError) as error:
        raise InjectionError(f"the run's resolved configuration is unreadable: {path}") from error
    campaign = resolved.get("campaign") if isinstance(resolved, dict) else None
    elastic = campaign.get("elastic_phones") if isinstance(campaign, dict) else None
    if not isinstance(elastic, dict) or not elastic:
        raise InjectionError("the run's campaign does not declare elastic_phones: no helper loss is injected "
                             "into a static-phone run")
    return elastic


def _lifecycle_rows(run_dir: Path) -> list[dict[str, object]]:
    try:
        rows = json.loads((run_dir / LIFECYCLE_FILE).read_text(encoding="ascii"))
    except (OSError, ValueError):
        return []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def latest_launch(run_dir: Path, device_id: str) -> dict[str, object] | None:
    launches = [row for row in _lifecycle_rows(run_dir)
                if row.get("phase") == "launch" and row.get("device_id") == device_id]
    return launches[-1] if launches else None


def trace_origin_monotonic_ns(run_dir: Path) -> int | None:
    """``monotonic_ns`` of the run's first co-helper launch receipt (None until one exists)."""
    return next((row["monotonic_ns"] for row in _lifecycle_rows(run_dir)
                 if row.get("phase") == "launch" and type(row.get("monotonic_ns")) is int), None)


def _launch_shell_words(launch: dict[str, object]) -> list[str] | None:
    """The worker launch of a receipt's adb command, as shell words (``su -c`` and lock unwrapped)."""
    command = launch.get("command")
    if type(command) is not list or not command or type(command[-1]) is not str:
        return None
    try:
        words = shlex.split(command[-1])
        if words[:2] == ["su", "-c"] and len(words) == 3:
            words = shlex.split(words[2].splitlines()[-1])
    except (ValueError, IndexError):
        return None
    return words


def launched_worker_path(launch: dict[str, object]) -> str | None:
    """The absolute worker executable the receipt launched (the word before ``-m``)."""
    words = _launch_shell_words(launch)
    if not words or "-m" not in words:
        return None
    index = words.index("-m")
    path = words[index - 1] if index > 0 else ""
    return path if path.startswith("/") and "=" not in path else None


def launched_adb_port(launch: dict[str, object]) -> int | None:
    command = launch.get("command")
    if type(command) is not list or "-P" not in command:
        return None
    value = command[command.index("-P") + 1] if command.index("-P") + 1 < len(command) else None
    return int(value) if type(value) is str and value.isdigit() and 0 < int(value) <= 65535 else None


def condition_met(run_dir: Path, condition: tuple[str, int | float], *, min_stream_bytes: int,
                  monotonic: Callable[[], float], state: dict[str, object] | None = None) -> dict[str, object] | None:
    kind, value = condition
    if kind == "request":
        stream = run_dir / "streams" / f"request-{value:03d}.raw"
        try:
            size = stream.stat().st_size
        except OSError:
            return None
        return None if size < min_stream_bytes else {
            "kind": kind, "request_index": value, "stream_path": str(stream), "stream_bytes": size}
    origin_ns = trace_origin_monotonic_ns(run_dir)
    if origin_ns is None:
        return None
    elapsed = monotonic() - origin_ns / 1e9
    if elapsed < value:
        return None
    if kind == "active":
        # Fire only while the helper is serving: its per-call lines must have grown since the last poll.
        layers = launched_layer_range(latest_launch(run_dir, state["device_id"]) or {}) if state is not None else None
        if layers is None:
            return None
        path, count = helper_call_count(run_dir, layers)
        previous = state.get("helper_calls")
        state["helper_calls"] = count
        if previous is None or count <= previous:
            return None
        return {"kind": kind, "seconds": value, "trace_origin_monotonic_ns": origin_ns, "elapsed_s": elapsed,
                "layers": list(layers), "server_stderr": path, "helper_calls_before": previous,
                "helper_calls_after": count}
    return {"kind": kind, "seconds": value, "trace_origin_monotonic_ns": origin_ns, "elapsed_s": elapsed}


def adb_shell(run, adb: str, adb_port: int, serial: str, command: str) -> subprocess.CompletedProcess:
    return run([adb, "-P", str(adb_port), "-s", serial, "shell", command], check=False,
               capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30)


def inject(
    args: argparse.Namespace,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
) -> dict[str, object]:
    if args.authorized is not True:
        raise InjectionError("fault injection needs the user's explicit authorization (--authorized)")
    if args.signal != "TERM":
        raise InjectionError("only SIGTERM is ever sent")
    resolved = (args.resolved_configuration if args.resolved_configuration is not None
                else args.run_dir.parent / RESOLVED_CONFIGURATION_FILE)
    elastic = campaign_elastic_phones(resolved)
    output = args.run_dir / RESULT_FILE
    if output.exists():
        raise InjectionError(f"{output} already exists")
    record: dict[str, object] = {
        "authorized": True, "device_id": args.device, "elastic_phones": elastic,
        "resolved_configuration": str(resolved), "run_dir": str(args.run_dir), "schema": SCHEMA,
        "signal": "TERM", "timestamps": {"tool_started_wall_s": wall_clock()},
        "when": {"kind": args.when[0], "value": args.when[1]},
    }
    deadline = monotonic() + args.timeout_s
    met = None
    condition_state: dict[str, object] = {"device_id": args.device}
    while met is None:
        if latest_launch(args.run_dir, args.device) is not None:
            met = condition_met(args.run_dir, args.when, min_stream_bytes=args.min_stream_bytes,
                                monotonic=monotonic, state=condition_state)
        if met is None:
            if monotonic() > deadline:
                record.update(status="NOT_INJECTED", reason="condition not met before the timeout")
                return _write(output, record)
            sleep(args.poll_s)
    record["condition"] = met
    record["timestamps"]["condition_met_wall_s"] = wall_clock()
    launch = latest_launch(args.run_dir, args.device) or {}
    serial = launch.get("serial")
    pids = launch.get("worker_pids")
    worker_path = launched_worker_path(launch)
    adb_port = launched_adb_port(launch)
    adb_port_source = "launch_receipt"
    if adb_port is None and args.adb_port is not None:
        adb_port, adb_port_source = args.adb_port, "--adb-port"
    record.update(serial=serial, launch_boot_id=launch.get("boot_id"), launch_worker_pids=pids,
                  launch_worker_path=worker_path, adb_port=adb_port, adb_port_source=adb_port_source)
    if type(serial) is not str or not serial or type(pids) is not list or not pids:
        record.update(status="NOT_INJECTED", reason="the launch receipt names no serial or worker pid")
        return _write(output, record)
    if worker_path is None or adb_port is None:
        record.update(status="NOT_INJECTED",
                      reason="the launch receipt names no worker path or adb port (and no --adb-port)")
        return _write(output, record)
    signalled, checks = [], {}
    for pid in pids:
        if type(pid) is not int or pid <= 0:
            continue
        exe = adb_shell(run, args.adb, adb_port, serial,
                        "su -c " + shlex.quote(f"readlink /proc/{pid}/exe || true")).stdout.strip()
        checks[str(pid)] = exe
        if exe != worker_path:
            continue
        record["timestamps"].setdefault("signal_sent_wall_s", wall_clock())
        result = adb_shell(run, args.adb, adb_port, serial,
                           "su -c " + shlex.quote(f"kill -TERM {pid}"))
        if result.returncode == 0:
            signalled.append(pid)
    record.update(worker_exe_by_pid=checks, signalled_pids=signalled,
                  status="INJECTED" if signalled else "NOT_INJECTED")
    if not signalled:
        record["reason"] = "no launch pid still runs the worker executable"
    record["timestamps"]["finished_wall_s"] = wall_clock()
    return _write(output, record)


def _write(path: Path, record: dict[str, object]) -> dict[str, object]:
    with path.open("x", encoding="ascii") as sink:
        sink.write(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", required=True, help="co-helper device id, e.g. pixel10pro-phone")
    parser.add_argument("--when", required=True, type=parse_condition, help="request=NNN (fires once that request has COMPLETED - streams are written at request end), t=SECONDS (after the first launch receipt) or active=SECONDS (after SECONDS, the next moment the helper is serving calls)")
    parser.add_argument("--signal", required=True, choices=("TERM",), help="only TERM; never KILL")
    parser.add_argument("--run-dir", required=True, type=Path, help="the campaign's run directory")
    parser.add_argument("--authorized", action="store_true",
                        help="the user explicitly authorized this fault injection")
    parser.add_argument("--resolved-configuration", type=Path,
                        help="the run's RESOLVED_CONFIGURATION.json (default: next to --run-dir, one level up)")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--adb-port", type=int,
                        help="adb server port, only when the launch receipt names none")
    parser.add_argument("--min-stream-bytes", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--poll-s", type=float, default=0.2)
    args = parser.parse_args(argv)
    if not args.run_dir.is_dir():
        parser.error("--run-dir must be an existing directory")
    if args.min_stream_bytes < 1 or args.timeout_s <= 0 or args.poll_s <= 0:
        parser.error("stream bytes, timeout and poll interval must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        record = inject(args)
    except InjectionError as error:
        print("inject_helper_loss: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps({key: record.get(key) for key in ("status", "signalled_pids", "condition")}, sort_keys=True))
    return 0 if record["status"] == "INJECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
