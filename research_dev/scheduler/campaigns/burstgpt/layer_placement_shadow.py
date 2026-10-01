"""Shadow mode of ``dispatch_policy.measured_placement`` (WS11): replay a finished run through the
layer-placement controller and write what it would have decided, plus the profile updated with the run.

Inputs are only what the run recorded: each model server's stderr (``large-model-<n>-*.stderr``: the
model it loaded, its helper ownership lines and its ``S41SERVERFFNSHAPE`` summaries), the helper and
device membership events and thermal deferrals in RESULT, and the request results (demand). Server
launch times are approximate (a server's exit is its log's last write; its launch is the previous
server's exit), every other event keeps its recorded time. Nothing here feeds back into the run.

Outputs (run directory): ``LAYER_PLACEMENT.json`` (configuration, the plan report at trace start, every
event, plan and decision) and ``LAYER_PLACEMENT_PROFILE.json`` (input profile + this run's measurements:
point the next run's ``profile_path`` at it to keep learning across runs).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from research_dev.scheduler._internal.layer_placement import (
    LayerPlacementError,
    LayerPlacementProfile,
    placement_report,
    with_envelope_caps,
)
from research_dev.scheduler._internal.layer_placement_control import (
    LayerPlacementController,
    MeasuredPlacementConfig,
    PlacementEvent,
)
from research_dev.scheduler._internal.layer_placement_io import (
    current_from_json,
    device_from_json,
    model_from_json,
    parse_shape_line,
    problem_from_json,
    shape_observations,
)


SHADOW_SCHEMA = "research-scheduler-layer-placement-shadow-v1"
_SERVER_LOG = re.compile(r"^large-model-(\d+)-.*\.stderr$")
_LOADED = re.compile(r"loaded meta data with .* from (\S+)")
_HELPER = re.compile(r"S41SERVERFFNHELPER label=(\S+) layer_mask=(\d+)")
_MEMBERSHIP = {
    "HELPER_LOST": "DEVICE_LEFT", "JOINED": "DEVICE_JOINED", "DEVICE_QUARANTINED": "DEVICE_QUARANTINED",
    "DEVICE_READMITTED": "DEVICE_READMITTED", "DEVICE_ABSENT_AT_START": "DEVICE_LEFT",
    "THERMAL_DEFERRAL": "THERMAL_EXCLUDED", "THERMAL_DEFERRAL_CLEARED": "THERMAL_CLEARED",
}


def load_inputs(config: MeasuredPlacementConfig) -> tuple[LayerPlacementProfile, dict]:
    """Profile and inventory of a configuration (builtin rig profile or the given files)."""
    if config.profile_path is not None:
        profile = LayerPlacementProfile.from_json(json.loads(Path(config.profile_path).read_text()))
    else:
        from research_dev.scheduler.campaigns.burstgpt.layer_placement_rig import measured_profile_v1
        profile = measured_profile_v1()
    if config.inventory_path is not None:
        inventory = json.loads(Path(config.inventory_path).read_text())
    else:
        from research_dev.scheduler.campaigns.burstgpt.layer_placement_rig import rig_inventory_v1
        inventory = rig_inventory_v1(profile=profile, pixel_transport=config.pixel_transport)
    problem_from_json(inventory, profile)      # validates the pair before a run starts
    return profile, inventory


def server_logs(run_dir: Path) -> list[Path]:
    rows = [(int(match.group(1)), path) for path in run_dir.iterdir()
            if (match := _SERVER_LOG.match(path.name)) is not None]
    return [path for _, path in sorted(rows)]


def _helper_map(inventory: Mapping[str, object], model_id: str) -> dict[str, dict[str, str]]:
    labels = {}
    for row in inventory["devices"]:
        formats = dict(row["formats"])
        if model_id in formats:
            labels[str(row["device_id"]).split("-", 1)[0]] = {
                "device_id": str(row["device_id"]), "format": formats[model_id], "transport": str(row["transport"])}
    return labels


def run_events(run_dir: Path, result: Mapping[str, object], path_to_model: Mapping[str, str],
               inventory: Mapping[str, object], *, trace_start_epoch_s: float | None = None,
               primary_label: str = "op15") -> list[PlacementEvent]:
    """The run as placement events, in time order."""
    models = {str(row["model_id"]): row for row in inventory["models"]}
    events: list[PlacementEvent] = [PlacementEvent("START", 0.0)]
    previous_exit = 0.0
    for index, path in enumerate(server_logs(run_dir)):
        lines = path.read_text(errors="replace").splitlines()
        model_path = next((match.group(1) for line in lines if (match := _LOADED.search(line))), None)
        model_id = path_to_model.get(model_path or "")
        if model_id not in models:
            continue
        if trace_start_epoch_s is not None:
            exit_s = max(previous_exit, os.stat(path).st_mtime - trace_start_epoch_s)
        else:
            exit_s = previous_exit + 600.0
        launch_s = previous_exit
        labels = _helper_map(inventory, model_id)
        helpers = {match.group(1) for line in lines if (match := _HELPER.search(line))}
        default = primary_label if not helpers else None
        shapes = [line for line in lines if parse_shape_line(line) is not None]
        rows = shape_observations(shapes, model_id=model_id, n_ff=int(models[model_id]["n_ff"]), helpers=labels,
                                  default_helper=default, evidence=f"{run_dir.name}/{path.name}") if shapes else []
        events.append(PlacementEvent("SERVER_LAUNCH", launch_s, model_id=model_id))
        if rows:
            events.append(PlacementEvent("SHAPES_OBSERVED", exit_s, model_id=model_id,
                                         details=MappingProxyType({"rows": rows})))
        events.append(PlacementEvent("SERVER_EXIT", exit_s, model_id=model_id))
        previous_exit = exit_s
    for key in ("helper_membership_events", "device_membership_events", "thermal_deferral_events"):
        for row in result.get(key, ()) or ():
            kind = _MEMBERSHIP.get(str(row.get("kind")))
            device = row.get("device_id")
            at_us = row.get("at_us", row.get("onset_at_us", row.get("observed_at_us")))
            if kind is None or device is None or at_us is None:
                continue
            events.append(PlacementEvent(kind, max(0.0, float(at_us) / 1e6), device_id=str(device)))
    completed: dict[str, float] = {}
    for row in sorted(result.get("request_results", ()) or (),
                      key=lambda item: int(item.get("trace_arrival_us", 0)) + int(item.get("actual_latency_us", 0))):
        model_id = row.get("model_id")
        if model_id not in models or not row.get("output_tokens"):
            continue
        at_s = (int(row.get("trace_arrival_us", 0)) + int(row.get("actual_latency_us", 0))) / 1e6
        completed[model_id] = completed.get(model_id, 0.0) + int(row["output_tokens"])
        events.append(PlacementEvent("DEMAND", at_s, model_id=model_id,
                                     details=MappingProxyType({"tokens_per_s": completed[model_id] / max(at_s, 1.0)})))
    order = {"START": 0, "SERVER_EXIT": 1, "DEVICE_LEFT": 2, "DEVICE_QUARANTINED": 2, "THERMAL_EXCLUDED": 2,
             "DEVICE_JOINED": 3, "DEVICE_READMITTED": 3, "THERMAL_CLEARED": 3, "DEMAND": 4, "SHAPES_OBSERVED": 5,
             "SERVER_LAUNCH": 6}
    return sorted(events, key=lambda event: (event.at_s, order.get(event.kind, 9)))


def run_shadow(config: MeasuredPlacementConfig, run_dir: Path, result: Mapping[str, object],
               path_to_model: Mapping[str, str], *, trace_start_epoch_s: float | None = None,
               restart_j: Mapping[str, float] | None = None, write: bool = True) -> dict[str, object]:
    """Replay one finished run; returns (and with ``write`` stores) the shadow record."""
    profile, inventory = load_inputs(config)
    if restart_j is None:
        from research_dev.scheduler.campaigns.burstgpt.layer_placement_rig import restart_energy_j
        restart_j = restart_energy_j()
    policy = config.policy(restart_j)
    problem = replace(problem_from_json(inventory, profile), uncertainty_ppm=policy.uncertainty_ppm,
                      confident_calls=policy.confident_calls)
    if policy.busy_growth_ppm is not None:
        problem = with_envelope_caps(problem, policy.busy_growth_ppm)
    report = placement_report(problem)
    models = [model_from_json(row) for row in inventory["models"]]
    devices = [device_from_json(row) for row in inventory["devices"]]
    mix = {str(row["model_id"]): {int(k): float(v) for k, v in dict(row.get("rows_mix", {"1": 1})).items()}
           for row in inventory["models"]}
    controller = LayerPlacementController.create(
        models, devices, profile, initial=current_from_json(models, inventory.get("current")),
        policy=policy, rows_mix=mix, latency_ppm=inventory.get("latency_ppm"))
    events = run_events(run_dir, result, path_to_model, inventory, trace_start_epoch_s=trace_start_epoch_s)
    errors = []
    for event in events:
        try:
            controller.handle(event)
        except LayerPlacementError as error:
            errors.append({"at_s": event.at_s, "kind": event.kind, "error": str(error)})
    counts: dict[str, int] = {}
    for event in events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    record = {"schema": SHADOW_SCHEMA, "configuration": config.to_json(), "event_counts": dict(sorted(counts.items())),
              "event_errors": errors, "initial_report": report, "controller": controller.to_json()}
    if write:
        (run_dir / "LAYER_PLACEMENT.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        (run_dir / "LAYER_PLACEMENT_PROFILE.json").write_text(
            json.dumps(controller.profile.to_json(), indent=1, sort_keys=True) + "\n")
    return record
