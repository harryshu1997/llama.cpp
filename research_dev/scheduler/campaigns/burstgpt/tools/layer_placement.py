#!/usr/bin/env python3
"""Measured layer placement (WS11): build the profile, plan, replay recorded runs, onboard models and devices.

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.layer_placement today --out-dir DIR
    python3 -m ... profile --out PROFILE.json                 # measured profile of the 4060 Ti rig
    python3 -m ... inventory --out INVENTORY.json [--pixel-transport adb-tcp|aoa-bridge] [--gemma-pixel-format F]
    python3 -m ... plan --profile PROFILE.json --inventory INVENTORY.json [--risk] [--growth-ppm N]
    python3 -m ... replay [--out REPLAY.json]                 # recorded block-1/2 servers through the controller
    python3 -m ... shapes --model M --n-ff N --label op15=op15-phone:f16:functionfs-usb SERVER.stderr...
    python3 -m ... onboard MODEL.gguf --ngl N --device htp:op15-phone:f16:9625939968:3:3208646656 [--origin Q.gguf]
    python3 -m ... probe-plan --device D --format F --transport T --layer-bytes B
    python3 -m ... probe-ingest --profile P.json --plan PLAN.json --model M --calls CALLS.json --out P2.json

Read-only on everything it is pointed at; writes only ``--out`` / ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research_dev.scheduler._internal.layer_placement import (  # noqa: E402
    DESKTOP_CPU,
    LayerPlacementError,
    LayerPlacementProfile,
    layer_mask,
    placement_report,
    provisioning_needs,
    solve_placement,
    with_envelope_caps,
)
from research_dev.scheduler._internal.layer_placement_control import (  # noqa: E402
    LayerPlacementController,
    PlacementEvent,
    RebalancePolicy,
)
from research_dev.scheduler._internal.layer_placement_io import (  # noqa: E402
    current_from_json,
    device_from_json,
    model_from_json,
    parse_layers,
    problem_from_json,
    shape_observations,
)
from research_dev.scheduler._internal.layer_placement_onboarding import (  # noqa: E402
    DeviceCapability,
    ProbePlan,
    onboard_model,
    probe_plan,
    profile_from_probe,
)
from research_dev.scheduler.campaigns.burstgpt import layer_placement_rig as rig  # noqa: E402


RISK = MappingProxyType({"scaled": 100_000, "prior": 250_000})
CONFIDENT_CALLS = 2000
FLEET_FULL_SYSTEM_KJ = 114.4   # block 1 full-system mean (talks.md 2026-09-30 10:45)


def assisted_tokens_per_run(runs=("p0m1", "p0m2", "p0m3")) -> dict[str, float]:
    """Decode tokens the phones assisted per block-1 run (OP15 calls / its layers x rows), mean over ``runs``."""
    totals: dict[str, float] = {}
    layers = {rig.QWEN: 18, rig.GEMMA: 24}
    for server in rig.recorded_servers():
        if server["run"] not in runs:
            continue
        for row in rig.server_observations(server):
            if row["device_id"] == rig.OP15:
                totals[row["model_id"]] = totals.get(row["model_id"], 0.0) + row["calls"] * row["rows"] / layers[row["model_id"]]
    return {model: value / len(runs) for model, value in totals.items()}


def _variant(name, note, profile, *, transport="adb-tcp", risk=False, growth=None, gemma_format="q4_0-packed",
             op15_capacity=None, op15_gemma_layers=None, available=None):
    inventory = rig.rig_inventory_v1(profile=profile, pixel_transport=transport, gemma_pixel_format=gemma_format,
                                     available=available, **({"op15_capacity": op15_capacity} if op15_capacity else {}))
    problem = problem_from_json(inventory, profile)
    if risk:
        problem = replace(problem, uncertainty_ppm=RISK, confident_calls=CONFIDENT_CALLS)
    if growth is not None:
        problem = with_envelope_caps(problem, growth)
    if op15_gemma_layers is not None:
        problem = replace(problem, devices=tuple(
            replace(device, allowed_layers=MappingProxyType({rig.QWEN: layer_mask(range(25)),
                                                             rig.GEMMA: layer_mask(parse_layers(op15_gemma_layers))}))
            if device.device_id == rig.OP15 else device for device in problem.devices))
    plan = solve_placement(problem)
    return {"name": name, "note": note, "transport": transport, "risk": risk, "growth_ppm": growth,
            "pixel_gemma_format": gemma_format, "plan": plan.to_json(),
            "provisioning_needs": [row.to_json() for row in provisioning_needs(problem, plan)]}


def today(out_dir: Path) -> dict[str, object]:
    profile = rig.measured_profile_v1()
    tokens = assisted_tokens_per_run()
    variants = [
        _variant("measured-adb", "today's deployed Pixel transport (adb forward), expected value", profile),
        _variant("measured-aoa-risk", "Pixel over the qualified AOA bridge, risk premiums (scaled +10 %, prior +25 %, "
                 f"< {CONFIDENT_CALLS} calls = scaled)", profile, transport="aoa-bridge", risk=True),
        _variant("measured-aoa-expected", "AOA, plain expected value", profile, transport="aoa-bridge"),
        _variant("measured-aoa-envelope", "AOA, safe exploration: busy/step <= 1.25 x measured envelope", profile,
                 transport="aoa-bridge", growth=250_000),
        _variant("op15-capacity-lowest-live", "OP15 capacity = lowest live HTP limit seen (9.27 GB, 2026-09-24)",
                 profile, op15_capacity=rig.OP15_LOWEST_LIVE_LIMIT),
        _variant("op15-gemma-held-24-q4_0", "OP15 Gemma kept at 0-23 (no new OP15 shards): Pixel Q4_0 packed (prior)",
                 profile, op15_gemma_layers="0-23"),
        _variant("op15-gemma-held-24-f16", "same, Pixel Gemma as f16 shards (derived rate)", profile,
                 op15_gemma_layers="0-23", gemma_format="f16"),
        _variant("op15-absent", "robustness: OP15 unavailable", profile, available={rig.OP15: False}),
        _variant("pixel-absent", "robustness: Pixel unavailable", profile, available={rig.PIXEL: False}),
    ]
    rows = []
    for variant in variants:
        plan = variant["plan"]
        total_kj = 0.0
        cells = {}
        for model, row in plan["models"].items():
            owners = {device: value["layers"] for device, value in row["owners"].items() if value["count"]}
            saving = row["saving_vs_current_j_per_token"]
            kj = saving * tokens.get(model, 0.0) / 1000.0
            total_kj += kj
            cells[model] = {"owners": owners, "j_per_token": row["energy_j_per_token"],
                            "saving_j_per_token": saving, "saving_kj_per_run": round(kj, 3),
                            "chain_ms_rows1": row["ffn_chain_ms_by_rows"]["1"],
                            "chain_ms_rows1_now": None}
        rows.append({"name": variant["name"], "models": cells, "saving_kj_per_run": round(total_kj, 3),
                     "saving_pct_of_fleet": round(100 * total_kj / FLEET_FULL_SYSTEM_KJ, 2),
                     "priors_used": plan["priors_used"], "solver": plan["solver"]})
    current_chain = {}
    base = problem_from_json(rig.rig_inventory_v1(profile=profile, pixel_transport="adb-tcp"), profile)
    for model in base.models:
        single = solve_placement(replace(base, devices=tuple(
            replace(device, allowed_layers=MappingProxyType({
                name: layer_mask(layer for layer, owner in base.current[name].items() if owner == device.device_id)
                for name in device.formats})) for device in base.devices)))
        current_chain[model.model_id] = single.models[model.model_id].chain_ms_by_rows[1]
    for row in rows:
        for model, cell in row["models"].items():
            cell["chain_ms_rows1_now"] = round(current_chain[model], 3)
    result = {"schema": "ws11-today-placement-v1", "assisted_tokens_per_run": tokens,
              "fleet_full_system_kj": FLEET_FULL_SYSTEM_KJ, "profile": profile.to_json(), "variants": variants,
              "summary": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "TODAY_PLACEMENT.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (out_dir / "TODAY_PLACEMENT.md").write_text(_today_markdown(result) + "\n")
    return result


def _today_markdown(result) -> str:
    short = {rig.QWEN: "Qwen", rig.GEMMA: "Gemma"}
    lines = ["| variant | Qwen owners | Gemma owners | saving J/token Qwen / Gemma | kJ/run | % fleet | FFN chain ms "
             "(rows 1) Qwen / Gemma, now -> plan | priors |", "|---|---|---|---|---|---|---|---|"]
    for row in result["summary"]:
        cells = row["models"]

        def owners(model):
            return ", ".join(f"{device.split('-')[0]} {layers}" for device, layers in sorted(cells[model]["owners"].items())
                             if device != DESKTOP_CPU) + (f", CPU {cells[model]['owners'][DESKTOP_CPU]}"
                                                         if DESKTOP_CPU in cells[model]["owners"] else "")
        lines.append(
            f"| {row['name']} | {owners(rig.QWEN)} | {owners(rig.GEMMA)} | "
            f"{cells[rig.QWEN]['saving_j_per_token']:.2f} / {cells[rig.GEMMA]['saving_j_per_token']:.2f} | "
            f"{row['saving_kj_per_run']:.1f} | {row['saving_pct_of_fleet']:.1f} | "
            f"{cells[rig.QWEN]['chain_ms_rows1_now']:.0f} -> {cells[rig.QWEN]['chain_ms_rows1']:.0f} / "
            f"{cells[rig.GEMMA]['chain_ms_rows1_now']:.0f} -> {cells[rig.GEMMA]['chain_ms_rows1']:.0f} | "
            f"{len(row['priors_used'])} |")
    lines.append("")
    lines.append("Assisted tokens per run (block 1 mean): " + ", ".join(
        f"{short.get(k, k)} {v:.0f}" for k, v in sorted(result["assisted_tokens_per_run"].items())))
    return "\n".join(lines)


def replay(*, policy: RebalancePolicy | None = None, drop: dict[str, float] | None = None) -> dict[str, object]:
    """Recorded block-1/2 servers, in run order, through the controller (shadow): what would it have done?

    Starts from the measured profile with the recorded shapes REMOVED (priors + the desktop only), then
    feeds every server as SERVER_LAUNCH -> SHAPES_OBSERVED -> SERVER_EXIT, one hour apart per run, so the
    profile is learned online exactly as a live run would learn it. ``drop`` = device -> time of a
    DEVICE_LEFT (robustness replay).
    """
    full = rig.measured_profile_v1()
    learning = LayerPlacementProfile.from_json({**full.to_json(), "calls": [], "call_energy": []})
    inventory = rig.rig_inventory_v1(profile=full, pixel_transport="adb-tcp")
    models = [model_from_json(row) for row in inventory["models"]]
    devices = [device_from_json(row) for row in inventory["devices"]]
    mix = {row["model_id"]: {int(k): float(v) for k, v in row["rows_mix"].items()} for row in inventory["models"]}
    policy = policy or RebalancePolicy(restart_j=MappingProxyType(rig.restart_energy_j()), default_tokens_per_s=1.0,
                                       uncertainty_ppm=RISK, confident_calls=CONFIDENT_CALLS)
    controller = LayerPlacementController.create(models, devices, learning, initial=current_from_json(
        models, inventory["current"]), policy=policy, rows_mix=mix, latency_ppm=inventory["latency_ppm"])
    for model in models:
        for device in devices:
            # energy per call is meter-derived in the full profile; the learner gets it from the meters too
            for rows in (1, 2):
                key = (device.device_id, model.model_id, rows)
                if key in full.energy_per_call:
                    controller.profile.energy_per_call[key] = full.energy_per_call[key]
    controller.handle(PlacementEvent("START", 0.0))
    clock = 0.0
    pending = sorted((drop or {}).items(), key=lambda row: row[1])
    for server in rig.recorded_servers():
        if server["run"].startswith("s43-pixel-aoa"):
            continue
        clock += 600.0
        while pending and pending[0][1] <= clock:
            device, at_s = pending.pop(0)
            controller.handle(PlacementEvent("DEVICE_LEFT", at_s, device_id=device))
        model = str(server["model"])
        controller.handle(PlacementEvent("SERVER_LAUNCH", clock, model_id=model))
        controller.handle(PlacementEvent("SHAPES_OBSERVED", clock + 300.0, model_id=model,
                                         details={"rows": rig.server_observations(server)}))
        controller.handle(PlacementEvent("SERVER_EXIT", clock + 301.0, model_id=model))
    return controller.to_json()


def _shapes(args) -> list[dict[str, object]]:
    labels = {}
    for text in args.label:
        label, _, rest = text.partition("=")
        device, shard_format, transport = rest.split(":")
        labels[label] = {"device_id": device, "format": shard_format, "transport": transport}
    rows = []
    for path in args.stderr:
        rows.extend(shape_observations(Path(path).read_text(errors="replace").splitlines(), model_id=args.model,
                                       n_ff=args.n_ff, helpers=labels, default_helper=args.default_label,
                                       evidence=str(path)))
    return rows


def gguf_metadata(path: Path) -> dict[str, object]:
    """Architecture, geometry and per-layer FFN tensor types/shapes of a GGUF (gguf-py, header only)."""
    sys.path.insert(0, str(_REPO / "gguf-py"))
    import gguf  # noqa: E402

    reader = gguf.GGUFReader(str(path))
    architecture = reader.fields["general.architecture"].contents()
    block_count = reader.fields[f"{architecture}.block_count"].contents()
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    layers = []
    for index in range(block_count):
        row = {"index": index, "moe": f"blk.{index}.ffn_gate_inp.weight" in tensors}
        for name in ("gate", "up", "down"):
            tensor = tensors.get(f"blk.{index}.ffn_{name}.weight")
            if tensor is None:
                break
            row[name] = {"type": gguf.GGMLQuantizationType(tensor.tensor_type).name,
                         "shape": [int(tensor.shape[0]), int(tensor.shape[1])]}
        else:
            layers.append(row)
    gate = tensors.get("blk.0.ffn_gate.weight")
    return {"model_id": path.stem, "architecture": architecture, "block_count": block_count,
            "n_embd": int(gate.shape[0]) if gate is not None else 0,
            "n_ff": int(gate.shape[1]) if gate is not None else 0, "layers": layers}


def _capability(text: str) -> DeviceCapability:
    parts = text.split(":")
    kind, device, formats, capacity = parts[:4]
    sessions = int(parts[4]) if len(parts) > 4 else 1
    limit = int(parts[5]) if len(parts) > 5 else 0
    return DeviceCapability(device, kind, tuple(formats.split(",")), int(capacity), "unknown", sessions, limit,
                            "per-model" if sessions > 1 else "co-resident")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("today")
    command.add_argument("--out-dir", type=Path, required=True)
    command = sub.add_parser("profile")
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("inventory")
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--pixel-transport", default="adb-tcp", choices=("adb-tcp", "aoa-bridge"))
    command.add_argument("--gemma-pixel-format", default="q4_0-packed", choices=("q4_0-packed", "f16"))
    command = sub.add_parser("plan")
    command.add_argument("--profile", type=Path, required=True)
    command.add_argument("--inventory", type=Path, required=True)
    command.add_argument("--risk", action="store_true")
    command.add_argument("--growth-ppm", type=int)
    command.add_argument("--method", default="auto", choices=("auto", "exact", "greedy"))
    command = sub.add_parser("replay")
    command.add_argument("--out", type=Path)
    command.add_argument("--drop", action="append", default=[], metavar="DEVICE@SECONDS")
    command = sub.add_parser("shapes")
    command.add_argument("--model", required=True)
    command.add_argument("--n-ff", type=int, required=True)
    command.add_argument("--label", action="append", required=True, metavar="LABEL=DEVICE:FORMAT:TRANSPORT")
    command.add_argument("--default-label")
    command.add_argument("stderr", nargs="+")
    command = sub.add_parser("onboard")
    command.add_argument("model", type=Path)
    command.add_argument("--ngl", type=int, required=True)
    command.add_argument("--device", action="append", default=[],
                         metavar="KIND:DEVICE:FORMATS:CAPACITY[:SESSIONS:SESSION_LIMIT]")
    command.add_argument("--origin", type=Path)
    command.add_argument("--origin-exact", action="store_true",
                         help="the exact-dequantization check of --origin against MODEL passed")
    command = sub.add_parser("probe-plan")
    command.add_argument("--device", required=True)
    command.add_argument("--format", required=True)
    command.add_argument("--transport", required=True)
    command.add_argument("--layer-bytes", type=int, required=True)
    command = sub.add_parser("probe-ingest")
    command.add_argument("--profile", type=Path, required=True)
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--model", required=True)
    command.add_argument("--calls", type=Path, required=True)
    command.add_argument("--meter-j", type=float)
    command.add_argument("--idle-w", type=float, default=0.0)
    command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "today":
            result = today(args.out_dir)
            print((args.out_dir / "TODAY_PLACEMENT.md").read_text())
            return 0 if result["summary"] else 1
        if args.command == "profile":
            args.out.write_text(json.dumps(rig.measured_profile_v1().to_json(), indent=1, sort_keys=True) + "\n")
            return 0
        if args.command == "inventory":
            value = rig.rig_inventory_v1(pixel_transport=args.pixel_transport, gemma_pixel_format=args.gemma_pixel_format)
            args.out.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
            return 0
        if args.command == "plan":
            profile = LayerPlacementProfile.from_json(json.loads(args.profile.read_text()))
            problem = problem_from_json(json.loads(args.inventory.read_text()), profile)
            if args.risk:
                problem = replace(problem, uncertainty_ppm=RISK, confident_calls=CONFIDENT_CALLS)
            if args.growth_ppm is not None:
                problem = with_envelope_caps(problem, args.growth_ppm)
            print(json.dumps(placement_report(problem, method=args.method), indent=1, sort_keys=True))
            return 0
        if args.command == "replay":
            drop = {}
            for text in args.drop:
                device, _, at_s = text.partition("@")
                drop[device] = float(at_s)
            result = replay(drop=drop)
            text = json.dumps(result, indent=1, sort_keys=True)
            if args.out:
                args.out.write_text(text + "\n")
            print(json.dumps({key: result[key] for key in ("decision_counts", "target", "busy_envelope_ms", "errors")},
                             indent=1, sort_keys=True))
            return 0
        if args.command == "shapes":
            print(json.dumps(_shapes(args), indent=1, sort_keys=True))
            return 0
        if args.command == "onboard":
            origin = gguf_metadata(args.origin) if args.origin else None
            verdict = onboard_model(gguf_metadata(args.model), n_gpu_layers=args.ngl,
                                    devices=[_capability(text) for text in args.device], origin_metadata=origin,
                                    origin_dequantizes_exactly=True if args.origin_exact else None)
            print(json.dumps(verdict.to_json(), indent=1, sort_keys=True))
            return 0 if verdict.placeable else 2
        if args.command == "probe-plan":
            print(json.dumps(probe_plan(args.device, args.format, args.transport, args.layer_bytes).to_json(),
                             indent=1, sort_keys=True))
            return 0
        if args.command == "probe-ingest":
            profile = LayerPlacementProfile.from_json(json.loads(args.profile.read_text()))
            raw = json.loads(args.plan.read_text())
            plan = ProbePlan(raw["device_id"], raw["format"], raw["transport"], raw["layers"], raw["layer_bytes"],
                             tuple(raw["rows"]), raw["calls_per_rows"], raw["warmup_calls"], raw["cadence_ms"],
                             raw["meter"])
            result = profile_from_probe(profile, plan, args.model, json.loads(args.calls.read_text()),
                                        meter_j_above_idle=args.meter_j, idle_w=args.idle_w, evidence=str(args.calls))
            args.out.write_text(json.dumps(profile.to_json(), indent=1, sort_keys=True) + "\n")
            print(json.dumps(result.to_json(), indent=1, sort_keys=True))
            return 0
    except (LayerPlacementError, OSError, KeyError, ValueError) as error:
        print("layer_placement: " + str(error), file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
