#!/usr/bin/env python3
"""Replay device-power policies over the timeline of a recorded run (no hardware).

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.device_power_replay \\
        --result RESULT.json --variant always-on=none --variant online=campaign.json ... [--json]

A variant is ``LABEL=none`` (no controller) or ``LABEL=<file>`` holding either a campaign manifest
(its ``device_power`` object) or the ``device_power`` object itself. The GPU timeline comes from
``RESULT.request_results`` of a run that did not need the controller: arrival
(``replay_arrival_us``), end of submission (``scheduling_overhead``), the last ACQUIRED dispatch,
execution start/finish (``completion.execution_receipt``) and the first token. The window between
ACQUIRED and the execution start is a model transition when it lasts at least ``--load-min-s``;
the time between submission and ACQUIRED is a queued ticket. Every variant drives the real
``DevicePowerController`` through these inputs exactly as the runner and the rig would (oracle:
the next arrival before it happens; online: observed arrivals only) with a fake clock and
instantaneous commands, converging on every input and at the controller's own wake-ups. A policy
without ``arrival_information`` is replayed as explicit ``oracle`` (same decisions, plus
telemetry); the CPU EPP knob is not modelled.

The report gives, per variant, the seconds of every (activity, state) pair (activity = idle, load,
prefill before the first token, decode), modelled GPU energy from ``--power-w`` (defaults: the
dp1/dp2/tp2 measurements, see ``DEFAULT_POWER_W``), prefill seconds under the decode cap,
restore waits modelled as ``--step-ms`` per command of every synchronous restore, and the
prediction quality of ``device_power_energy.prediction_quality``. The timeline is held fixed:
effects of the controller on the work itself (capped prefill runs 1.3-1.9x longer, restore waits
delay a start) are reported, not fed back.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from research_dev.scheduler.adapters.device_power import DevicePowerController
from research_dev.scheduler.adapters.ticket import PhysicalExecutionCommand, PhysicalParticipantCommand
from research_dev.scheduler.campaigns.burstgpt.tools.device_power_energy import (
    prediction_quality,
    state_timeline_us,
)
from research_dev.scheduler.configuration.campaign import DevicePowerConfiguration

SCHEMA = "s42-device-power-replay-v1"
EPOCH_NS = 10 ** 15
GPU_DEVICE = "desktop-cuda"
ACTIVITIES = ("idle", "load", "prefill", "decode")
# GPU board W per (activity, controller state); measured on the RTX 4060 Ti desktop:
# decode uncapped 32.6 W and IDLE_MIN 14.4 W and LOAD_MIN 15.1 W (dp1), DECODE_CAP 1,200 MHz 27.3 W (dp2),
# loading uncapped ~38 W (tp2/cj2), loaded idle 20 W (cj2; tp2 27.6 W). Prefill power was not
# measured separately and uses the decode values.
DEFAULT_POWER_W = {
    "idle": {"RESTORED": 20.0, "IDLE_MIN": 14.4, "LOAD_MIN": 14.4, "DECODE_CAP": 20.0, "UNCONTROLLED": 20.0},
    "load": {"RESTORED": 38.0, "IDLE_MIN": 15.1, "LOAD_MIN": 15.1, "DECODE_CAP": 38.0, "UNCONTROLLED": 38.0},
    "prefill": {"RESTORED": 32.6, "IDLE_MIN": 32.6, "LOAD_MIN": 32.6, "DECODE_CAP": 27.3, "UNCONTROLLED": 32.6},
    "decode": {"RESTORED": 32.6, "IDLE_MIN": 32.6, "LOAD_MIN": 32.6, "DECODE_CAP": 27.3, "UNCONTROLLED": 32.6},
}


class _InstantRun:
    """``subprocess.run`` stand-in for the replay: every permitted command succeeds at once."""

    def __init__(self) -> None:
        self.sm_mhz = 2595
        self.calls = 0

    def __call__(self, argv, *, capture_output, text, timeout, check, input=None):
        self.calls += 1
        if argv[:3] == ["sudo", "-n", "-l"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "-lgc" in argv:
            self.sm_mhz = int(argv[argv.index("-lgc") + 1].split(",")[1])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "-rgc" in argv:
            self.sm_mhz = 2595
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if any(part.startswith("--query-gpu=") for part in argv):
            return SimpleNamespace(returncode=0, stdout=f"{self.sm_mhz}, 8751, P2\n", stderr="")
        raise ValueError("replay: unexpected command " + " ".join(argv))


def _last_acquired_us(row: Mapping[str, Any], fallback: int) -> int:
    acquired = [receipt["observed_at_us"] for receipt in row.get("dispatch_receipts") or ()
                if receipt.get("status") == "ACQUIRED" and type(receipt.get("observed_at_us")) is int]
    return max(acquired) if acquired else fallback


def extract_timeline(result: Mapping[str, Any], *, load_min_s: float = 1.0) -> dict[str, Any]:
    """Per-request timeline on the RESULT clock (us) from a recorded RESULT."""
    paid_start_ns, paid_end_ns = result["paid_start_ns"], result["paid_end_ns"]
    rows = []
    for row in result.get("request_results") or ():
        receipt = (row.get("completion") or {}).get("execution_receipt") or {}
        started, finished = receipt.get("started_us"), receipt.get("finished_us")
        if type(started) is not int or type(finished) is not int or type(row.get("replay_arrival_us")) is not int:
            continue
        participants = (row.get("execution_command") or {}).get("participants") or ()
        arrival = row["replay_arrival_us"]
        overhead = row.get("scheduling_overhead") or {}
        submitted = arrival + (int(overhead.get("snapshot_capture_ns", 0)) + int(overhead.get("scheduler_submit_ns", 0))) // 1000
        acquired = min(_last_acquired_us(row, started), started)
        first_token = row.get("first_token_ns")
        first_token_us = started if type(first_token) is not int else max(started, (first_token - paid_start_ns) // 1000)
        rows.append({
            "request_id": row["request_id"], "model_id": row.get("model_id", "model"), "arrival_us": arrival,
            "submitted_us": submitted, "acquired_us": acquired, "started_us": started,
            "first_token_us": min(first_token_us, finished), "finished_us": finished,
            "gpu": any(part.get("device_id") == GPU_DEVICE for part in participants),
            "load": started - acquired >= load_min_s * 1e6,
        })
    rows.sort(key=lambda item: (item["arrival_us"], item["request_id"]))
    return {"end_us": (paid_end_ns - paid_start_ns) // 1000, "requests": rows}


def _command(row: Mapping[str, Any]) -> PhysicalExecutionCommand:
    participant = PhysicalParticipantCommand("physical:replay", GPU_DEVICE if row["gpu"] else "desktop-cpu",
                                             "replay", "replay", ())
    return PhysicalExecutionCommand(
        ticket_id=row["request_id"] + ":replay", request_id=row["request_id"], model_id=row["model_id"],
        artifact_sha256="sha256:" + "0" * 64, route_id="replay", executor_id="physical:replay", endpoint="replay",
        operator_plan_protocol="replay", operator_plan_sha256="sha256:" + "0" * 64, planned_start_us=row["started_us"],
        planned_finish_us=row["finished_us"], planned_finish_upper_us=row["finished_us"], operator_plan={},
        participants=(participant,), leases=(), memory_reservations=(), transitions=(), execution_contract=None,
        adapter_parameters={})


def _actions(timeline: Mapping[str, Any], policy: DevicePowerConfiguration,
             counters: dict[str, int]) -> list[tuple[int, int, Callable[[DevicePowerController], None]]]:
    """(time_us, order, action) in the order the runner and rig would issue them; at equal time
    releases (order 0) come before arrivals/notes (1), acquisitions and starts (2) and first
    tokens (3)."""
    requests = timeline["requests"]
    actions: list[tuple[int, int, Callable[[DevicePowerController], None]]] = []

    def queued(delta: int) -> Callable[[DevicePowerController], None]:
        def apply(controller: DevicePowerController) -> None:
            counters["queued"] += delta
            controller.note_queued_start_us(0 if counters["queued"] else None)
        return apply

    def loading(delta: int) -> Callable[[DevicePowerController], None]:
        def apply(controller: DevicePowerController) -> None:
            counters["load"] += delta
            if delta > 0 and counters["load"] == 1:
                controller.note_transition_active(True)
                controller.on_load_begin()
            elif delta < 0 and counters["load"] == 0:
                controller.on_load_end()
                controller.note_transition_active(False)
        return apply

    if policy.online:
        for row in requests:
            actions.append((row["arrival_us"], 1, lambda c, r=row: c.note_arrival_observed(
                r["request_id"], r["model_id"], r["arrival_us"])))
    else:
        # the runner notes arrival i+1 once request i is submitted, and None after the last
        notes = [(0, requests[0]["arrival_us"] if requests else None)]
        for index, row in enumerate(requests):
            following = requests[index + 1]["arrival_us"] if index + 1 < len(requests) else None
            notes.append((row["submitted_us"], following))
        for at_us, value in notes:
            actions.append((at_us, 1, lambda c, v=value: c.note_next_arrival_us(v)))
    for row in requests:
        if row["acquired_us"] > row["submitted_us"]:
            actions.append((row["submitted_us"], 1, queued(+1)))
            actions.append((row["acquired_us"], 0, queued(-1)))
        if row["load"]:
            actions.append((row["acquired_us"], 2, loading(+1)))
            actions.append((row["started_us"], 0, loading(-1)))
        command = _command(row)
        actions.append((row["started_us"], 2, lambda c, cmd=command: c.on_execution_start(cmd)))
        if policy.decode_cap is not None and policy.decode_cap.protect_prefill:
            actions.append((row["first_token_us"], 3, lambda c, r=row: c.note_first_token(r["request_id"])))
        actions.append((row["finished_us"], 0, lambda c, cmd=command: c.on_execution_finish(cmd)))
    return sorted(actions, key=lambda item: (item[0], item[1]))


def activity_timeline(timeline: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    """``[(start_us, end_us, activity)]``: prefill over decode over load over idle."""
    end_us = timeline["end_us"]
    points = {0, end_us}
    for row in timeline["requests"]:
        points.update(value for value in (row["acquired_us"], row["started_us"], row["first_token_us"],
                                          row["finished_us"]) if 0 <= value <= end_us)
    ordered = sorted(points)
    pieces = []
    for start, stop in zip(ordered, ordered[1:]):
        middle = (start + stop) / 2
        activity = "idle"
        for row in timeline["requests"]:
            if not row["gpu"]:
                continue
            if row["started_us"] <= middle < row["first_token_us"]:
                activity = "prefill"
                break
            if row["first_token_us"] <= middle < row["finished_us"]:
                activity = "decode"
            elif activity == "idle" and row["load"] and row["acquired_us"] <= middle < row["started_us"]:
                activity = "load"
        if pieces and pieces[-1][2] == activity and pieces[-1][1] == start:
            pieces[-1] = (pieces[-1][0], stop, activity)
        else:
            pieces.append((start, stop, activity))
    return pieces


def replay_policy(timeline: Mapping[str, Any], policy: DevicePowerConfiguration | None, *,
                  tick_s: float = 0.5) -> dict[str, Any]:
    """Drive a controller over ``timeline``; events, telemetry and state timeline (None: no controller)."""
    end_us = timeline["end_us"]
    if policy is None:
        return {"events": [], "telemetry": None,
                "states": [{"state": "UNCONTROLLED", "reason": None, "start_us": 0, "end_us": end_us}]}
    row = policy.to_json()
    if row.get("idle") is not None:
        row["idle"] = {**row["idle"], "cpu_epp": None}
    if row.get("arrival_information") is None:
        row["arrival_information"] = "oracle"
    policy = DevicePowerConfiguration.from_json(row)
    clock = [EPOCH_NS]
    with tempfile.TemporaryDirectory(prefix="device-power-replay-") as directory:
        controller = DevicePowerController(policy, epoch_ns_provider=lambda: EPOCH_NS, output_directory=Path(directory),
                                           run=_InstantRun(), monotonic_ns=lambda: clock[0], tick_interval_s=tick_s)
        if not controller.capability_probe():
            raise RuntimeError("replay controller probe failed")
        counters = {"queued": 0, "load": 0}
        batches: dict[int, list[Callable[[DevicePowerController], None]]] = {}
        for at_us, _order, action in _actions(timeline, policy, counters):
            batches.setdefault(max(0, min(end_us, at_us)), []).append(action)
        next_wake_us = 0
        for at_us in sorted(batches) + [None]:
            stop_us = end_us if at_us is None else at_us
            while next_wake_us < stop_us:
                clock[0] = EPOCH_NS + next_wake_us * 1000
                controller._converge()
                next_wake_us += max(1, int(controller._wake_timeout_s * 1e6))
            clock[0] = EPOCH_NS + stop_us * 1000
            if at_us is None:
                controller.end_trace()
                break
            # the rig's synchronous hooks converge inside the actions; the thread converges once
            # after the inputs of one instant (it is woken by them, not between them)
            for action in batches[at_us]:
                action(controller)
            controller._converge()
            next_wake_us = at_us + max(1, int(controller._wake_timeout_s * 1e6))
        events = [dict(event) for event in controller.events]
        telemetry = controller.telemetry
        controller.close()
    return {"events": events, "telemetry": telemetry, "states": state_timeline_us(events, end_us)}


def _intersect(pieces: Sequence[tuple[int, int, str]], states: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    seconds: dict[str, dict[str, float]] = {activity: {} for activity in ACTIVITIES}
    for start, stop, activity in pieces:
        for row in states:
            overlap = min(stop, row["end_us"]) - max(start, row["start_us"])
            if overlap > 0:
                state = "RESTORED" if row["state"] == "OFF" else row["state"]
                seconds[activity][state] = seconds[activity].get(state, 0.0) + overlap / 1e6
    return seconds


def summarize(timeline: Mapping[str, Any], replayed: Mapping[str, Any], *, power_w: Mapping[str, Mapping[str, float]],
              step_ms: float, break_even_s: float | None = None) -> dict[str, Any]:
    pieces = activity_timeline(timeline)
    seconds = _intersect(pieces, replayed["states"])
    energy_kj = sum(value * power_w[activity][state] for activity, states in seconds.items()
                    for state, value in states.items()) / 1000
    events = replayed["events"]
    synchronous = [event for event in events if event["reason"] in ("late_restore", "prefill_restore")]
    report = {
        "activity_state_seconds": {activity: dict(sorted(states.items())) for activity, states in seconds.items()},
        "gpu_kj_modelled": energy_kj,
        "capped_prefill_s": seconds["prefill"].get("DECODE_CAP", 0.0),
        "lowered_prefill_s": sum(seconds["prefill"].get(state, 0.0) for state in ("DECODE_CAP", "IDLE_MIN", "LOAD_MIN")),
        "state_changes": sum(1 for event in events if event["from"] != event["to"]) - (1 if events else 0),
        "reasons": {},
        "synchronous_restores": len(synchronous),
        "restore_wait_ms_modelled": sum(len(event["result"]) for event in synchronous) * step_ms,
        "prediction_quality": None,
    }
    for event in events[1:]:
        report["reasons"][event["reason"]] = report["reasons"].get(event["reason"], 0) + 1
    report["reasons"] = dict(sorted(report["reasons"].items()))
    if replayed["telemetry"] is not None:
        quality = prediction_quality(events, replayed["telemetry"], timeline["end_us"],
                                     None if break_even_s is None else int(break_even_s * 1e6))
        report["prediction_quality"] = {key: value for key, value in quality.items()
                                        if key not in ("intervals", "idle_min_episode_rows", "restore_waits_by_request")}
    return report


def load_variant(value: str) -> DevicePowerConfiguration | None:
    if value == "none":
        return None
    row = json.loads(Path(value).read_text())
    if type(row) is dict and "schema" in row and "device" not in row:
        row = row.get("device_power")
        if row is None:
            return None
    return DevicePowerConfiguration.from_json(row)


def replay(result: Mapping[str, Any], variants: Sequence[tuple[str, DevicePowerConfiguration | None]], *,
           power_w: Mapping[str, Mapping[str, float]] = DEFAULT_POWER_W, step_ms: float = 130.0,
           load_min_s: float = 1.0, break_even_s: float | None = 2.0) -> dict[str, Any]:
    timeline = extract_timeline(result, load_min_s=load_min_s)
    rows = []
    for label, policy in variants:
        replayed = replay_policy(timeline, policy)
        rows.append({"label": label, "policy": None if policy is None else policy.to_json(),
                     **summarize(timeline, replayed, power_w=power_w, step_ms=step_ms, break_even_s=break_even_s)})
    baseline = next((row["gpu_kj_modelled"] for row in rows if row["policy"] is None), None)
    for row in rows:
        row["gpu_kj_saved_vs_uncontrolled"] = None if baseline is None else baseline - row["gpu_kj_modelled"]
    activity = {name: 0.0 for name in ACTIVITIES}
    for start, stop, name in activity_timeline(timeline):
        activity[name] += (stop - start) / 1e6
    return {"schema": SCHEMA, "window_s": timeline["end_us"] / 1e6, "requests": len(timeline["requests"]),
            "activity_seconds": activity, "power_w": copy.deepcopy(dict(power_w)), "step_ms": step_ms,
            "break_even_s": break_even_s, "variants": rows}


def render(report: Mapping[str, Any]) -> str:
    activity = report["activity_seconds"]
    lines = [f"window {report['window_s']:.1f} s, {report['requests']} requests; activity: "
             + ", ".join(f"{name} {activity[name]:.1f} s" for name in ACTIVITIES)
             + ("" if report.get("break_even_s") is None else f"; quality horizon {report['break_even_s']:.1f} s")]
    lines.append(f"{'variant':<14} {'gpu_kj':>7} {'saved':>7} {'idle_min_s':>10} {'missed_s':>9} {'false':>5} "
                 f"{'sync_rst':>8} {'wait_ms':>8} {'cap_pref_s':>10} {'changes':>7}")
    for row in report["variants"]:
        quality = row["prediction_quality"] or {}
        saved = row["gpu_kj_saved_vs_uncontrolled"]
        lines.append(
            f"{row['label']:<14} {row['gpu_kj_modelled']:>7.2f} {'-' if saved is None else format(saved, '.2f'):>7} "
            f"{quality.get('captured_us', 0) / 1e6:>10.1f} "
            f"{'-' if not quality else format(quality['missed_opportunity_us'] / 1e6, '.1f'):>9} "
            f"{'-' if not quality else quality['false_idle_drops']:>5} {row['synchronous_restores']:>8} "
            f"{row['restore_wait_ms_modelled']:>8.0f} {row['capped_prefill_s']:>10.1f} {row['state_changes']:>7}")
    for row in report["variants"]:
        lines.append(f"  {row['label']}: " + "; ".join(
            f"{name} " + ", ".join(f"{state} {value:.1f}" for state, value in states.items())
            for name, states in row["activity_state_seconds"].items() if states))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", type=Path, required=True, help="RESULT.json of a recorded run")
    parser.add_argument("--variant", action="append", required=True, metavar="LABEL=none|FILE")
    parser.add_argument("--power-w", type=Path, help="JSON {activity: {state: W}} replacing DEFAULT_POWER_W")
    parser.add_argument("--step-ms", type=float, default=130.0, help="wall time of one command + readback")
    parser.add_argument("--load-min-s", type=float, default=1.0, help="ACQUIRED->start windows this long are loads")
    parser.add_argument("--break-even-s", type=float, default=2.0,
                        help="one horizon for every variant's prediction quality (0 = each policy's min_gap_s)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    variants = []
    for item in args.variant:
        label, separator, value = item.partition("=")
        if not separator or not label or not value:
            parser.error("--variant takes LABEL=none|FILE")
        variants.append((label, load_variant(value)))
    power = DEFAULT_POWER_W if args.power_w is None else json.loads(args.power_w.read_text())
    report = replay(json.loads(args.result.read_text()), variants, power_w=power, step_ms=args.step_ms,
                    load_min_s=args.load_min_s, break_even_s=args.break_even_s or None)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
