#!/usr/bin/env python3
"""Measured layer placement (WS11): build the profile, plan, replay recorded runs, onboard models and devices.

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.layer_placement today --out-dir DIR
    python3 -m ... profile --out PROFILE.json                 # measured profile of the 4060 Ti rig
    python3 -m ... inventory --out INVENTORY.json [--pixel-transport adb-tcp|aoa-bridge] [--gemma-pixel-format F]
    python3 -m ... plan --profile PROFILE.json --inventory INVENTORY.json [--risk] [--growth-ppm N]
    python3 -m ... replay [--out REPLAY.json]                 # recorded block-1/2 servers through the controller
    python3 -m ... shapes --model M --n-ff N --label op15=op15-phone:f16:functionfs-usb SERVER.stderr...
    python3 -m ... onboard MODEL.gguf --ngl N --device htp:op15-phone:f16:9625939968:3:3208646656[:approx] [--origin Q.gguf]
    python3 -m ... probe-plan --device D --format F --transport T --layer-bytes B [--layers N] [--rows 1,2,4]
    python3 -m ... probe-ingest --profile P.json --plan PLAN.json --model M --calls CALLS.json --out P2.json
    python3 -m ... profile --out P.json [--base P0.json] [--add-shapes ROWS.json ...]   (ROWS from ``shapes``)
    python3 -m ... inventory --out I.json [--pixel-transport aoa-bridge] [--stored DEV:MODEL=LAYERS ...]
                   [--qualified DEV:MODEL=LAYERS ...] [--busy-envelope DEV=MS ...]
    python3 -m ... geometry --model ID=F16.gguf --format ID:FORMAT=ORIGIN.gguf ... --out GEOMETRY.json   (WS14)
    python3 -m ... opgroups --out-dir DIR [--contexts 2048,8192,32768,131072] [--no-sensitivity]       (WS14)

Read-only on everything it is pointed at; writes only ``--out`` / ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

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
    fused_shape_observations,
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
             op15_capacity=None, op15_gemma_layers=None, available=None, pixel_runtime="co-helper"):
    inventory = rig.rig_inventory_v1(profile=profile, pixel_transport=transport, gemma_pixel_format=gemma_format,
                                     available=available, pixel_runtime=pixel_runtime,
                                     **({"op15_capacity": op15_capacity} if op15_capacity else {}))
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
            "pixel_gemma_format": gemma_format, "pixel_runtime": pixel_runtime, "plan": plan.to_json(),
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
        _variant("op15-gemma-held-24-q4_0", "OP15 Gemma kept at 0-23 (no new OP15 shards): Pixel Q4_0 packed (prior); "
                 "assumes multi-artifact helper support (not built)",
                 profile, op15_gemma_layers="0-23", pixel_runtime="hypothetical-multi-model"),
        _variant("op15-gemma-held-24-f16", "same, Pixel Gemma as f16 shards (derived rate)", profile,
                 op15_gemma_layers="0-23", gemma_format="f16", pixel_runtime="hypothetical-multi-model"),
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


def add_shape_rows(profile: LayerPlacementProfile, rows) -> int:
    """Fold ``shapes`` observation rows (server ``S41SERVERFFNSHAPE`` summaries) into ``profile``."""
    for row in rows:
        profile.observe_call_summary(str(row["device_id"]), str(row["model_id"]), str(row["shard_format"]),
                                     str(row["transport"]), int(row["rows"]), int(row["calls"]), float(row["rpc_ms"]),
                                     float(row["compute_ms"]), str(row.get("evidence", "")))
    return len(rows)


def add_desktop_layer(profile: LayerPlacementProfile, text: str) -> None:
    """``MODEL=MS[@ROWS][:EVIDENCE]``: the desktop CPU time of one full FFN layer (e.g. twice the server's host
    branch at half width, as the 14B row was measured)."""
    model, _, rest = text.partition("=")
    value, _, evidence = rest.partition(":")
    ms, _, rows = value.partition("@")
    if not model or not ms:
        raise ValueError("expected MODEL=MS[@ROWS][:EVIDENCE], got " + text)
    profile.observe_desktop_layer(model, int(rows or 1), float(ms), evidence)


def add_fused_shape_rows(profile: LayerPlacementProfile, rows) -> int:
    """Fold WS15 fused call rows (``fused_shape_observations``) into ``profile``'s fused call store."""
    for row in rows:
        profile.observe_fused_call_summary(str(row["device_id"]), str(row["model_id"]), str(row["shard_format"]),
                                           str(row["transport"]), int(row["rows"]), int(row["calls"]),
                                           float(row["rpc_ms"]), float(row["compute_ms"]),
                                           str(row.get("evidence", "")))
    return len(rows)


def _override(text: str) -> tuple[str, str, str]:
    owner, _, layers = text.partition("=")
    device, _, model = owner.partition(":")
    if not (device and model and layers):
        raise ValueError("expected DEVICE:MODEL=LAYERS, got " + text)
    return device, model, layers


def override_inventory(value: dict, *, stored=(), qualified=(), busy_envelope=(), kernels=(),
                       qualified_kernel=()) -> dict:
    """Inventory with measured/provisioned facts replaced: stored and qualified layers per device and model,
    a device's measured busy envelope (ms per token), and (WS15) a device's kernels and the layers each
    non-FFN kernel's evidence qualified. Unknown devices are refused."""
    value = json.loads(json.dumps(value))
    devices = {row["device_id"]: row for row in value["devices"]}
    for text in kernels:
        device, _, names = text.partition("=")
        if device not in devices or "ffn" not in names.split(","):
            raise ValueError("expected a known DEVICE=ffn[,KERNEL...], got " + text)
        devices[device]["kernels"] = sorted(set(names.split(",")))
    for text in qualified_kernel:
        owner, _, layers = text.partition("=")
        device, _, rest = owner.partition(":")
        model, _, kernel = rest.partition(":")      # model ids have no colon; kernel names may (attn_proj:fused)
        if device not in devices or not (model and kernel and layers) or kernel == "ffn":
            raise ValueError("expected a known DEVICE:MODEL:KERNEL=LAYERS (non-FFN kernel), got " + text)
        parse_layers(layers)
        devices[device].setdefault("kernel_qualified_layers", {}).setdefault(kernel, {})[model] = layers
    for key, entries in (("stored_layers", stored), ("qualified_layers", qualified)):
        for text in entries:
            device, model, layers = _override(text)
            if device not in devices:
                raise ValueError("unknown inventory device " + device)
            parse_layers(layers)
            devices[device].setdefault(key, {})[model] = layers
    for text in busy_envelope:
        device, _, ms = text.partition("=")
        if device not in devices or not ms:
            raise ValueError("expected a known DEVICE=MS, got " + text)
        devices[device]["busy_envelope_ms"] = float(ms)
    return value


def _shapes(args) -> list[dict[str, object]]:
    labels = {}
    for text in args.label:
        label, _, rest = text.partition("=")
        device, shard_format, transport = rest.split(":")
        labels[label] = {"device_id": device, "format": shard_format, "transport": transport}
    rows = []
    for path in args.stderr:
        text = Path(path).read_text(errors="replace")
        try:   # qualify_pixel_server_transport's SERVER_FFN_SHAPES.json: a JSON list of server log lines
            decoded = json.loads(text)
        except ValueError:
            decoded = None
        lines = decoded if isinstance(decoded, list) and all(isinstance(line, str) for line in decoded) \
            else text.splitlines()
        if getattr(args, "fused", False):
            # WS15: S41SERVERFFNFUSEDSHAPE rows (SERVER_FFN_FUSED_SHAPES.json) -> profile --add-fused-shapes rows
            rows.extend(fused_shape_observations(lines, model_id=args.model, helpers=labels,
                                                 default_helper=args.default_label, evidence=str(path)))
            continue
        rows.extend(shape_observations(lines, model_id=args.model,
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


def _gguf_module():
    sys.path.insert(0, str(_REPO / "gguf-py"))
    import gguf  # noqa: E402
    return gguf


def geometry_file(models: list[str], formats: list[str]) -> dict[str, object]:
    """WS14 model geometry from GGUF headers: ``ID=F16.gguf`` (shapes; also the ``f16`` format) and
    ``ID:FORMAT=ORIGIN.gguf`` (the tensor types a packed format stores)."""
    from research_dev.scheduler._internal.layer_placement_opgroups import model_geometry, read_gguf_header

    gguf = _gguf_module()
    by_model = {}
    for text in models:
        model_id, _, path = text.partition("=")
        by_model[model_id] = {"path": Path(path), "formats": {}}
    for text in formats:
        owner, _, path = text.partition("=")
        model_id, _, shard_format = owner.partition(":")
        if model_id not in by_model or not shard_format or not path:
            raise ValueError("expected ID:FORMAT=ORIGIN.gguf for a --model ID, got " + text)
        by_model[model_id]["formats"][shard_format] = Path(path)
    result = {}
    for model_id, row in by_model.items():
        header = read_gguf_header(row["path"], gguf)
        headers = {"f16": header}
        sources = [f"f16: {row['path'].name} ({row['path'].stat().st_size} B)"]
        for shard_format, path in sorted(row["formats"].items()):
            headers[shard_format] = read_gguf_header(path, gguf)
            sources.append(f"{shard_format}: tensor types of {path.name} ({path.stat().st_size} B)")
        result[model_id] = model_geometry(model_id, header, headers, sources=sources).to_json()
    return {"schema": "research-scheduler-model-geometry-set-v1", "models": result}


def concurrency_inputs(path: Path | None, *, penalty_ms: float | None = None,
                       degradation: float | None = None) -> dict[str, object]:
    """WS20 planner inputs for the parallel column split.

    ``path``: a CONCURRENCY.json written by ``tools/parallel_split_microbench.py analyze`` (MEASURED penalty and
    degradation -> an executable plan may contain parallel splits); without it the values are labelled priors
    (ideal plans only). Explicit ``penalty_ms`` / ``degradation`` override the file (and stay priors)."""
    provenance = "prior"
    penalty, factor = 1.0, 1.0
    if path is not None:
        data = json.loads(Path(path).read_text())
        if data.get("schema") != "ws20-parallel-split-concurrency-v1" or data.get("status") != "PASS":
            raise SystemExit("--concurrency-from needs a PASS ws20-parallel-split-concurrency-v1 file")
        penalty = float(data["penalty_ms"])
        factor = float(data["degradation"])
        provenance = "measured"
    if penalty_ms is not None:
        penalty, provenance = float(penalty_ms), "prior"
    if degradation is not None:
        factor, provenance = float(degradation), "prior"
    return {"parallel_split": True, "concurrency_penalty_ms": penalty, "concurrency_degradation": factor,
            "concurrency_provenance": provenance}


def opgroups_report(*, contexts=(2048, 8192, 32768, 131072), sensitivity: bool = True,
                    method: str = "auto", parallel: Mapping[str, object] | None = None) -> dict[str, object]:
    """WS14 gate: which operator groups pay on today's rig (adb and AOA Pixel), at the trace's contexts and
    swept ones, and how robust each verdict is to +-25 % on the phone energy, desktop waiting power and
    phone bandwidth. Expected value (no risk premium) plus the AOA risk variant."""
    from research_dev.scheduler._internal import layer_placement_opgroups as og

    profile = rig.measured_profile_v1()
    inputs = rig.opgroup_inputs_v1(**(parallel or {}))
    kernel_sets = tuple(og.KERNEL_SETS) + (tuple(og.PARALLEL_SETS) if parallel else ())
    tokens = assisted_tokens_per_run()
    result: dict[str, object] = {"schema": "ws14-opgroups-report-v1", "tokens_per_run": tokens,
                                 **({"parallel_split": dict(parallel)} if parallel else {}),
                                 "fleet_full_system_kj": FLEET_FULL_SYSTEM_KJ,
                                 "inputs": {"contexts": {m: c.to_json() for m, c in inputs.contexts.items()},
                                            "slots": dict(inputs.slots),
                                            "link_bytes_per_s": dict(inputs.link_bytes_per_s),
                                            "provenance": dict(inputs.provenance)},
                                 "variants": {}}
    for name, transport, risk in (("adb", "adb-tcp", False), ("aoa", "aoa-bridge", False),
                                  ("aoa-risk", "aoa-bridge", True)):
        problem = problem_from_json(rig.rig_inventory_v1(profile=profile, pixel_transport=transport), profile)
        if risk:
            problem = replace(problem, uncertainty_ppm=RISK, confident_calls=CONFIDENT_CALLS)
        table = og.decision_table(problem, inputs, tokens_per_run=tokens, method=method, fleet_kj=FLEET_FULL_SYSTEM_KJ,
                                  kernel_sets=kernel_sets)
        variant = {"transport": transport, "risk": risk, **table}
        if name == "aoa":
            variant["context_sweep"] = og.context_sweep(problem, inputs, contexts, tokens_per_run=tokens, method=method)
            if sensitivity:
                variant["sensitivity"] = og.sensitivity(problem, inputs, tokens_per_run=tokens, method=method)
        if name == "adb" and sensitivity:
            variant["sensitivity"] = og.sensitivity(problem, inputs, tokens_per_run=tokens, method=method)
        result["variants"][name] = variant
    return result


def _opgroups_markdown(report: dict) -> str:
    short = {rig.QWEN: "Qwen", rig.GEMMA: "Gemma"}
    lines = ["## Verdict per operator group (ideal kernels on every device, any layer; gain vs the FFN-only ideal plan)",
             "", "| model | group | adb: verdict | adb: J/token | adb: kJ/run | AOA: verdict | AOA: J/token | AOA: kJ/run "
             "| AOA+risk: verdict | missing kernel |", "|---|---|---|---|---|---|---|---|---|---|"]
    adb, aoa, aoa_risk = (report["variants"][name] for name in ("adb", "aoa", "aoa-risk"))
    for model in sorted(adb["verdicts"]):
        for group, cell in adb["verdicts"][model].items():
            other = aoa["verdicts"][model][group]
            risky = aoa_risk["verdicts"][model][group]
            lines.append(f"| {short.get(model, model)} | {group} | {cell['verdict']} | {cell['gain_j_per_token']:.2f} | "
                         f"{cell['gain_kj_per_run']:.1f} | {other['verdict']} | {other['gain_j_per_token']:.2f} | "
                         f"{other['gain_kj_per_run']:.1f} | {risky['verdict']} | `{cell['missing_kernel']}` |")
    for name in ("adb", "aoa"):
        variant = report["variants"][name]
        lines += ["", f"## Kernel sets, {name} Pixel transport (gain vs FFN-only ideal; kJ/run x block-1 assisted tokens)", "",
                  "| kernels | kJ/run | % fleet | Qwen groups | Gemma groups | solver |", "|---|---|---|---|---|---|"]
        for row in variant["summary"]:
            def groups(model):
                cell = row["models"][model]
                parts = []
                for group in ("ffn", "attn_proj", "attn_kv"):
                    owners = {k: v for k, v in cell["groups"][group].items() if k != DESKTOP_CPU and v}
                    if owners:
                        parts.append(group + " " + ", ".join(f"{k.replace('-phone', '')} {v}" for k, v in sorted(owners.items())))
                if cell.get("offloaded_units", {}).get("ffn_parallel"):
                    parts.append(f"parallel x{cell['offloaded_units']['ffn_parallel']}")
                if cell["output_head"] != "desktop-gpu":
                    parts.append("head " + cell["output_head"])
                return "; ".join(parts) or "-"
            lines.append(f"| {row['kernels']} | {row['gain_kj_per_run']:.1f} | {row.get('gain_pct_of_fleet', 0):.1f} | "
                         f"{groups(rig.QWEN)} | {groups(rig.GEMMA)} | {row['solver']} |")
        executable = variant["executable"]
        lines += ["", f"Executable today ({name}): " + "; ".join(
            f"{short.get(m, m)} ffn " + ", ".join(f"{k.replace('-phone', '')} {v}" for k, v in sorted(row['groups']['ffn'].items())
                                                 if k != DESKTOP_CPU and v) for m, row in sorted(executable["models"].items()))]
    sweep = report["variants"]["aoa"].get("context_sweep", [])
    if sweep:
        lines += ["", "## attn_kv (fused, on top of fused projections) vs context, AOA", "",
                  "| context | model | KV layers offloaded | gain J/token | pays per unit | KV resident per layer (max) | host KV ms/token |",
                  "|---|---|---|---|---|---|---|"]
        for row in sweep:
            for model, cell in sorted(row["models"].items()):
                lines.append(f"| {row['context']} | {short.get(model, model)} | {cell['kv_layers_offloaded']} | "
                             f"{cell['gain_j_per_token']:.2f} | {cell['pays_per_unit']} | "
                             f"{cell['kv_resident_bytes_per_layer_max'] / 1e6:.0f} MB | {cell['host_kv_ms_per_token_rows1']:.1f} |")
    for name in ("adb", "aoa"):
        rows = report["variants"][name].get("sensitivity", [])
        if not rows:
            continue
        lines += ["", f"## Sensitivity ({name}): verdict (gain J/token) with one input moved", "",
                  "| input | factor | Qwen attn_proj fused | Qwen attn_proj separate | Qwen attn_kv | Qwen head | Gemma attn_proj fused | Gemma attn_kv |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in rows:
            q, g = row["verdicts"][rig.QWEN], row["verdicts"][rig.GEMMA]
            cell = lambda c: f"{c['verdict'].split(',')[0]} ({c['gain_j_per_token']:.2f})"
            lines.append(f"| {row['parameter']} | {row['factor']} | {cell(q['attn_proj fused'])} | "
                         f"{cell(q['attn_proj separate'])} | {cell(q['attn_kv fused'])} | {cell(q['output_head'])} | "
                         f"{cell(g['attn_proj fused'])} | {cell(g['attn_kv fused'])} |")
    return "\n".join(lines)


def _capability(text: str) -> DeviceCapability:
    parts = text.split(":")
    # WS17: an optional 7th field "approx" marks a device whose quantized execution is qualified by the
    # approximate-numerics evidence class (tools/approximate_numerics_gate.py); absent = exact numerics only
    approximate = len(parts) > 6 and parts[6] == "approx"
    if len(parts) > 6 and not approximate:
        raise SystemExit(f"device spec field 7 must be 'approx': {text}")
    kind, device, formats, capacity = parts[:4]
    sessions = int(parts[4]) if len(parts) > 4 else 1
    limit = int(parts[5]) if len(parts) > 5 else 0
    return DeviceCapability(device, kind, tuple(formats.split(",")), int(capacity), "unknown", sessions, limit,
                            "per-model" if sessions > 1 else "co-resident", allow_approximate=approximate)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("today")
    command.add_argument("--out-dir", type=Path, required=True)
    command = sub.add_parser("profile")
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--base", type=Path, help="fold into this profile instead of the recorded v1 profile")
    command.add_argument("--add-shapes", type=Path, action="append", default=[],
                         help="observation rows printed by the shapes command (repeatable)")
    command.add_argument("--add-fused-shapes", type=Path, action="append", default=[],
                         help="WS15 fused call rows (fused_shape_observations JSON list; repeatable)")
    command.add_argument("--desktop-layer", action="append", default=[], metavar="MODEL=MS[@ROWS][:EVIDENCE]",
                         help="measured desktop CPU time of one full FFN layer of a model (onboarding a new model)")
    command = sub.add_parser("inventory")
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--pixel-transport", default="adb-tcp", choices=("adb-tcp", "aoa-bridge"))
    command.add_argument("--gemma-pixel-format", default="q4_0-packed", choices=("q4_0-packed", "f16"))
    command.add_argument("--pixel-runtime", default="co-helper", choices=("co-helper", "helper-anchored"),
                         help="WS23: helper-anchored = the runtime built with campaign "
                              "elastic_phones.helper_independence (the Pixel may anchor its own route)")
    command.add_argument("--stored", action="append", default=[], metavar="DEVICE:MODEL=LAYERS",
                         help="replace a device's stored layers of a model (after a shard push)")
    command.add_argument("--qualified", action="append", default=[], metavar="DEVICE:MODEL=LAYERS",
                         help="replace a device's qualified layers of a model (after its evidence)")
    command.add_argument("--kernels", action="append", default=[], metavar="DEVICE=ffn,attn_proj:fused",
                         help="WS15: replace a device's kernels (what has an executable path)")
    command.add_argument("--qualified-kernel", action="append", default=[], metavar="DEVICE:MODEL:KERNEL=LAYERS",
                         help="WS15: layers a non-FFN kernel's evidence qualified (after its fused evidence)")
    command.add_argument("--busy-envelope", action="append", default=[], metavar="DEVICE=MS",
                         help="measured sustained busy ms per token (e.g. from probe-ingest)")
    command = sub.add_parser("plan")
    command.add_argument("--profile", type=Path, required=True)
    command.add_argument("--inventory", type=Path, required=True)
    command.add_argument("--risk", action="store_true")
    command.add_argument("--growth-ppm", type=int)
    command.add_argument("--method", default="auto", choices=("auto", "exact", "greedy"))
    command.add_argument("--outages", action="store_true",
                         help="WS23: price each plan with one helper's layers back on the desktop CPU")
    command = sub.add_parser("replay")
    command.add_argument("--out", type=Path)
    command.add_argument("--drop", action="append", default=[], metavar="DEVICE@SECONDS")
    command = sub.add_parser("shapes")
    command.add_argument("--model", required=True)
    command.add_argument("--n-ff", type=int, required=True)
    command.add_argument("--label", action="append", required=True, metavar="LABEL=DEVICE:FORMAT:TRANSPORT")
    command.add_argument("--default-label")
    command.add_argument("--fused", action="store_true",
                         help="WS15: read fused call rows (S41SERVERFFNFUSEDSHAPE) for profile --add-fused-shapes")
    command.add_argument("stderr", nargs="+")
    command = sub.add_parser("onboard")
    command.add_argument("model", type=Path)
    command.add_argument("--ngl", type=int, required=True)
    command.add_argument("--device", action="append", default=[],
                         metavar="KIND:DEVICE:FORMATS:CAPACITY[:SESSIONS:SESSION_LIMIT]")
    command.add_argument("--origin", type=Path)
    command.add_argument("--origin-exact", action="store_true",
                         help="the exact-dequantization check of --origin against MODEL passed")
    command = sub.add_parser("geometry")
    command.add_argument("--model", action="append", required=True, metavar="ID=F16.gguf")
    command.add_argument("--format", action="append", default=[], metavar="ID:FORMAT=ORIGIN.gguf")
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("opgroups")
    command.add_argument("--out-dir", type=Path, required=True)
    command.add_argument("--contexts", default="2048,8192,32768,131072")
    command.add_argument("--no-sensitivity", action="store_true")
    command.add_argument("--method", default="auto", choices=("auto", "exact", "greedy"))
    command.add_argument("--parallel-split", action="store_true",
                         help="WS20: also report the parallel column split across helpers (opt-in)")
    command.add_argument("--concurrency-from", type=Path,
                         help="WS20: CONCURRENCY.json of tools/parallel_split_microbench.py (measured penalty)")
    command.add_argument("--concurrency-penalty-ms", type=float, help="WS20: override the pair penalty (a prior)")
    command.add_argument("--concurrency-degradation", type=float,
                         help="WS20: override the concurrent compute slowdown factor >= 1 (a prior)")
    command = sub.add_parser("probe-plan")
    command.add_argument("--device", required=True)
    command.add_argument("--format", required=True)
    command.add_argument("--transport", required=True)
    command.add_argument("--layer-bytes", type=int, required=True)
    command.add_argument("--layers", type=int, default=2, help="resident layers called per token")
    command.add_argument("--rows", default="1,2,4")
    command.add_argument("--calls-per-rows", type=int, default=96)
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
            profile = (LayerPlacementProfile.from_json(json.loads(args.base.read_text())) if args.base
                       else rig.measured_profile_v1())
            folded = add_shape_rows(profile, [row for path in args.add_shapes for row in json.loads(path.read_text())])
            fused = add_fused_shape_rows(profile, [row for path in args.add_fused_shapes
                                                   for row in json.loads(path.read_text())])
            for text in args.desktop_layer:
                add_desktop_layer(profile, text)
            args.out.write_text(json.dumps(profile.to_json(), indent=1, sort_keys=True) + "\n")
            if args.add_shapes or args.add_fused_shapes:
                print(json.dumps({"folded_rows": folded, **({"folded_fused_rows": fused} if fused else {})},
                                 sort_keys=True))
            return 0
        if args.command == "inventory":
            value = rig.rig_inventory_v1(pixel_transport=args.pixel_transport, gemma_pixel_format=args.gemma_pixel_format,
                                         pixel_runtime=args.pixel_runtime)
            value = override_inventory(value, stored=args.stored, qualified=args.qualified,
                                       busy_envelope=args.busy_envelope, kernels=args.kernels,
                                       qualified_kernel=args.qualified_kernel)
            args.out.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
            return 0
        if args.command == "plan":
            profile = LayerPlacementProfile.from_json(json.loads(args.profile.read_text()))
            problem = problem_from_json(json.loads(args.inventory.read_text()), profile)
            if args.risk:
                problem = replace(problem, uncertainty_ppm=RISK, confident_calls=CONFIDENT_CALLS)
            if args.growth_ppm is not None:
                problem = with_envelope_caps(problem, args.growth_ppm)
            print(json.dumps(placement_report(problem, method=args.method, outages=args.outages), indent=1, sort_keys=True))
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
        if args.command == "geometry":
            args.out.write_text(json.dumps(geometry_file(args.model, args.format), indent=1, sort_keys=True) + "\n")
            return 0
        if args.command == "opgroups":
            parallel = None
            if args.parallel_split or args.concurrency_from is not None:
                parallel = concurrency_inputs(args.concurrency_from, penalty_ms=args.concurrency_penalty_ms,
                                              degradation=args.concurrency_degradation)
            report = opgroups_report(contexts=tuple(int(value) for value in args.contexts.split(",")),
                                     sensitivity=not args.no_sensitivity, method=args.method, parallel=parallel)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / "OPGROUPS_REPORT.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
            text = _opgroups_markdown(report)
            (args.out_dir / "OPGROUPS_REPORT.md").write_text(text + "\n")
            print(text)
            return 0
        if args.command == "probe-plan":
            print(json.dumps(probe_plan(args.device, args.format, args.transport, args.layer_bytes, layers=args.layers,
                                        rows=tuple(int(value) for value in args.rows.split(",")),
                                        calls_per_rows=args.calls_per_rows).to_json(), indent=1, sort_keys=True))
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
            lowest = result.by_rows[min(result.by_rows)]
            # sustained busy per token at the lowest row count: the device's measured envelope (inventory
            # --busy-envelope) once the probe ran at the planned layer count
            print(json.dumps({**result.to_json(), "busy_ms_per_token": round(plan.layers * lowest["rpc_ms"], 3)},
                             indent=1, sort_keys=True))
            return 0
    except (LayerPlacementError, OSError, KeyError, ValueError) as error:
        print("layer_placement: " + str(error), file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
