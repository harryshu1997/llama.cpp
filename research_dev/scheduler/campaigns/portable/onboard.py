"""Onboarding: measure a host and a phone once and write the site profile.

    onboard-host   a few host-only legs of one model: time per token alone, loaded idle power, decode power,
                   whether the host keeps its copy of the shared weights resident
    onboard-phone  a share sweep with one phone (host-only legs between the shared ones): the shared-layer
                   fit, the capacity-aware static share plan, the phone's record; it also gives the host
                   its layer time, the fixed cost of sharing and the power of the time a share removes
                   (the server reports its own layer time only for layers it shares)

Both can ``ingest`` finished run directories, so a profile can be rebuilt (or built from the runs of an
earlier harness) without machine time. Every number names the run it came from. A sweep that shows no
beneficial share is a valid result: the plan says ``not-beneficial`` and nothing is provisioned.
"""

from __future__ import annotations

from pathlib import Path
import statistics
from typing import Callable, Mapping, Sequence

from ..._internal.gguf_metadata import read_gguf_metadata
from ..._internal.layer_placement import (
    DesktopPower,
    LayerPlacementError,
    LayerPlacementProfile,
    ffn_layer_bytes,
)
from ..._internal.layer_placement_onboarding import SPLIT_ARCHITECTURES
from ..._internal.shared_layer import (
    PhoneCapacity,
    SharedLayerFit,
    StaticSharePlan,
    fold_fit_into_profile,
    plan_static_share,
    pool_deltas,
)
from ...adapters.contracts import PhysicalAdapterError
from ...configuration.site import SiteManifest
from ...configuration.site_profile import SiteProfile
from ...drivers.adb import sha256_file
from ...drivers.phone import PhoneHealth
from . import report
from .rundir import IDLE, RunDirectory

PROFILE_NAME = "PROFILE.json"
MEMORY_SHARE_OF_AVAILABLE = 0.75     # of a phone's available memory, the part shards may take
RESIDENT_HEADROOM = 1.1              # the host keeps the shared weights resident when the model fits with room


def profile_path(site: SiteManifest) -> Path:
    return site.paths.work_root / PROFILE_NAME


def model_geometry(path: Path, gpu_layers: int) -> dict[str, object]:
    """Architecture, FFN geometry, weight type and CPU-resident layers of a GGUF at this ``-ngl``."""
    reader = read_gguf_metadata(path)
    architecture = str(reader.fields["general.architecture"].parts[-1])
    if architecture not in SPLIT_ARCHITECTURES:
        raise PhysicalAdapterError(f"{architecture} has no FFN split (supported: {', '.join(sorted(SPLIT_ARCHITECTURES))})")
    block_count = int(reader.fields[architecture + ".block_count"].parts[-1])
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    gate = tensors.get("blk.0.ffn_gate.weight")
    if gate is None or len(gate.shape) != 2:
        raise PhysicalAdapterError("the model has no dense FFN in its first layer")
    n_embd, n_ff = int(gate.shape[0]), int(gate.shape[1])
    types = {tensors[f"blk.{layer}.ffn_{name}.weight"].tensor_type.name
             for layer in range(block_count) for name in ("gate", "up", "down")
             if f"blk.{layer}.ffn_{name}.weight" in tensors}
    if len(types) != 1:
        raise PhysicalAdapterError("FFN weights of mixed types: " + ",".join(sorted(types)))
    weight_type = types.pop()
    # llama.cpp offloads the last (ngl - 1) blocks plus the output layer when ngl <= block_count
    first_gpu = max(block_count + 1 - gpu_layers, 0) if gpu_layers > 0 else block_count
    cpu_layers = list(range(min(first_gpu, block_count, 64)))
    return {"architecture": architecture, "activation": SPLIT_ARCHITECTURES[architecture], "block_count": block_count,
            "n_embd": n_embd, "n_ff": n_ff, "weight_type": weight_type, "gpu_layers": gpu_layers,
            "cpu_layers": cpu_layers, "layer_bytes": ffn_layer_bytes(n_embd, n_ff, weight_type)}


def memory_available_bytes(meminfo: Path = Path("/proc/meminfo")) -> int:
    for line in meminfo.read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise PhysicalAdapterError("MemAvailable is unreadable")


def host_share(model_bytes: int, available_before_load: int) -> str:
    """Keep the host's copy of the shared weights resident when the whole model fits with room.

    Releasing them during decode and restoring them at the next prompt costs every request its first
    token (0.3-1 s measured), so it is done only when memory is short.
    """
    return "resident" if available_before_load >= model_bytes * RESIDENT_HEADROOM else "dormant"


def host_record(run: RunDirectory, geometry: Mapping[str, object], *, artifact_sha256: str,
                memory_available: int, model_bytes: int) -> dict[str, object]:
    """A host's record for one model from a host-only run (its legs are the host alone)."""
    legs = [leg for leg in run.legs() if leg["columns"] == 0]
    if not legs:
        raise LayerPlacementError("a host record needs host-only legs")
    idle = run.idle(IDLE)
    return {**{key: geometry[key] for key in ("architecture", "n_embd", "n_ff", "block_count", "weight_type",
                                              "gpu_layers", "cpu_layers", "layer_bytes")},
            "artifact_sha256": artifact_sha256,
            "token_ms_alone": statistics.fmean(leg["ms_token"] for leg in legs),
            "p_idle_w": idle["cpu_w"] + idle["gpu_w"],
            "p_decode_w": statistics.fmean(leg["cpu_w"] + leg["gpu_w"] for leg in legs),
            "host_share": host_share(model_bytes, memory_available),
            "memory_available_bytes": memory_available, "threads": run.threads, "legs": len(legs),
            "evidence": [run.path.name]}


def phone_capacity(site: SiteManifest, device_id: str, available_bytes: int) -> PhoneCapacity:
    phone = site.phone(device_id)
    return PhoneCapacity(int(available_bytes * MEMORY_SHARE_OF_AVAILABLE), phone.sessions, phone.session_limit_bytes)


def phone_record(site: SiteManifest, device_id: str, capacity: PhoneCapacity, quality: str,
                 health: PhoneHealth | None, evidence: Sequence[str],
                 caps_floor: Mapping[str, int] | None = None) -> dict[str, object]:
    """A phone's record. ``cpu_caps_khz`` is the bar for later runs: the floor of its clock caps over a few
    seconds at onboarding when given (some phones move an idle cluster's cap while awake), else one reading."""
    phone = site.phone(device_id)
    caps = dict(caps_floor) if caps_floor else ({} if health is None else dict(health.cpu_caps_khz))
    return {"transport": phone.transport, "kind": phone.kind, "usable_bytes": capacity.usable_bytes,
            "sessions": capacity.sessions, "session_limit_bytes": capacity.session_limit_bytes,
            "power_quality": quality, "awake": None if health is None else health.awake,
            "cpu_caps_khz": caps, "usb_speed_mbps": phone.usb_speed_mbps, "evidence": list(evidence)}


def ingest_host(site: SiteManifest, profile: SiteProfile, model_id: str, run: RunDirectory, *, gpu_layers: int,
                artifact_sha256: str | None = None, memory_available: int | None = None) -> dict[str, object]:
    """Fold a finished host-only run into the profile."""
    model = site.paths.models[model_id]
    geometry = model_geometry(model, gpu_layers)
    record = host_record(run, geometry, artifact_sha256=artifact_sha256 or sha256_file(model),
                         memory_available=memory_available_bytes() if memory_available is None else memory_available,
                         model_bytes=model.stat().st_size)
    profile.set_host_model(model_id, record)
    return record


def ingest_phone(site: SiteManifest, profile: SiteProfile, device_id: str, model_id: str,
                 runs: Sequence[RunDirectory], *, capacity: PhoneCapacity, health: PhoneHealth | None = None,
                 stored_columns: int | None = None, update_host: bool | None = None,
                 caps_floor: Mapping[str, int] | None = None) -> dict[str, object]:
    """Fold finished sweep runs of one phone and model into the profile: fit, share plan, planner rows.

    ``runs``: the last one that swept at least three shares gives the fit; all of them give the pooled
    savings. ``stored_columns`` bounds the plan when the production shards already exist. The host's own
    numbers from the sweep (layer time, fixed cost of sharing, the power of the removed time) are written
    to the host record when it has none yet, so a second phone does not replace what the first measured
    (``update_host`` True forces it, False forbids it); the pair always keeps its own fit.
    """
    host = profile.host_model(model_id)
    phone = site.phone(device_id)
    sweep = report.sweep_report(runs)
    if "sweep" not in sweep:
        raise LayerPlacementError("no run swept at least three shares")
    fit = SharedLayerFit.from_json(sweep["sweep"]["fit"])
    if fit.n_ff != host["n_ff"]:
        raise LayerPlacementError("the sweep ran another model than the profile's host record")
    plan = plan_static_share(fit, cpu_layers=host["cpu_layers"], n_embd=int(host["n_embd"]),
                             column_quantum=phone.column_quantum, weight_type=str(host["weight_type"]),
                             capacity=capacity, stored_columns=stored_columns)
    # savings are comparable only between runs that kept as many layers on the phone as the plan does
    # (a probe sweep keeps a few); the power of the time a share removes does not depend on that
    same_layers = pool_deltas([row for row in sweep["legs"] if row["layers"] == len(plan.layers)])
    chosen = same_layers.get(str(round(plan.share, 4))) if plan.beneficial else None
    best = chosen or max(sweep["pooled"].values(), key=lambda row: row["time_saving_mean"], default=None)
    p_f = best["marginal_host_w"] if best else None
    evidence = [run.path.name for run in runs]
    if update_host or (update_host is None and "t_host_ms" not in host):
        profile.update_host_model(
            model_id, t_host_ms=fit.t_host_ms, c_host_ms=fit.c_host_ms, p_f_w=p_f,
            other_step_ms=float(host["token_ms_alone"]) - len(host["cpu_layers"]) * fit.t_host_ms,
            shared_layer_evidence=evidence)
    quality = "/".join(sweep["phone_energy_quality"])
    profile.set_phone(device_id, phone_record(site, device_id, capacity, quality, health, evidence, caps_floor))
    measured = None
    if chosen is not None:
        measured = {"time_saving_pct": chosen["time_saving_mean"], "energy_saving_pct": chosen["energy_saving_mean"],
                    "legs": chosen["n_legs"], "phone_metered": chosen["phone_metered"]}
    pair = {"device_id": device_id, "model_id": model_id, "transport": phone.transport,
            "usb_sysfs_device": phone.usb_sysfs_device, "usb_speed_mbps": phone.usb_speed_mbps,
            "fit": fit.to_json(), "share_plan": plan.to_json(), "measured_at_plan_share": measured,
            "marginal_host_w": p_f, "evidence": evidence}
    profile.set_pair(pair)
    # the same measurements in the planner's own profile
    planner = (LayerPlacementProfile.from_json(profile.layer_placement_profile)
               if profile.layer_placement_profile is not None else
               LayerPlacementProfile(DesktopPower(float(host["p_decode_w"]), float(host["p_decode_w"]),
                                                  float(host["p_decode_w"]), "prior")))
    phone_power = site.phone(device_id).power
    fold_fit_into_profile(planner, fit, model_id=model_id, device_id=device_id,
                          shard_format=str(host["weight_type"]).lower(), transport=phone.transport,
                          layer_bytes=int(host["layer_bytes"]), evidence=evidence[-1], host_ffn_w=p_f,
                          phone_active_w=None if phone_power is None else phone_power.assumed_active_w,
                          phone_idle_w=0.0 if phone_power is None or phone_power.assumed_idle_w is None
                          else phone_power.assumed_idle_w)
    profile.layer_placement_profile = planner.to_json()
    profile.revision += 1
    return pair


def format_pair(pair: Mapping[str, object], host: Mapping[str, object]) -> str:
    fit, plan = pair["fit"], StaticSharePlan.from_json(pair["share_plan"])
    lines = [f"{pair['device_id']} with {pair['model_id']} over {pair['transport']} "
             f"(USB {pair['usb_speed_mbps']} Mb/s):",
             f"  host layer {fit['t_host_ms']:.2f} ms (+{fit['c_host_ms']:.2f} when shared); phone call "
             f"{fit['t_link_ms']:.2f} + share x {fit['t_phone_ms']:.2f} ms; both end together at "
             f"{fit['f_star'] * 100:.1f} % ({fit['t_star_ms']:.2f} ms)"]
    if not plan.beneficial:
        lines.append("  plan: not beneficial -- the link costs more than the share saves; nothing to provision")
        return "\n".join(lines)
    lines.append(f"  plan: {plan.columns} columns ({plan.share * 100:.1f} %) of {len(plan.layers)} of "
                 f"{len(host['cpu_layers'])} CPU layers in {len(plan.sessions)} session(s), "
                 f"{plan.resident_bytes / 1e9:.2f} GB on the phone, bound by {plan.binding_constraint}")
    lines.append(f"  predicted {plan.predicted_token_ms_saved:.1f} ms saved per token of "
                 f"{host['token_ms_alone']:.1f} ms alone "
                 f"({plan.predicted_token_ms_saved / float(host['token_ms_alone']) * 100:.1f} %)")
    measured = pair.get("measured_at_plan_share")
    if measured:
        lines.append(f"  measured at that share: time -{measured['time_saving_pct']:.1f} %, energy "
                     f"-{measured['energy_saving_pct']:.1f} % over {measured['legs']} legs"
                     + ("" if measured["phone_metered"] else " (phone energy not measured)"))
    return "\n".join(lines)


def sweep_columns(stored_columns: int, column_quantum: int, steps: int = 8) -> list[int]:
    """The shares an onboarding sweep visits: up to ``steps`` multiples of the quantum up to the stored width."""
    count = stored_columns // column_quantum
    if count < 3:
        raise PhysicalAdapterError("a sweep needs at least three shares below the stored columns")
    stride = max(1, count // steps)
    columns = [column_quantum * index for index in range(stride, count + 1, stride)]
    return columns if columns[-1] == stored_columns else columns + [stored_columns]
