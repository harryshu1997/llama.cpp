"""Measured layer-placement profile and inventory of the 4060 Ti + OP15 + Pixel rig (WS11).

Builds the :class:`LayerPlacementProfile` from recorded measurements only, each with its provenance:

* helper call costs: every ``S41SERVERFFNSHAPE`` summary of the two-phone block-1/2 runs and of the
  2026-10-01 Pixel transport identity runs (``data/layer_placement/RECORDED_SERVER_FFN_SHAPES_20261001.json``,
  read-only copies of the rig's server logs);
* helper energy per call: phone meters of block 1 (``p0m1-3``), energy above each phone's idle baseline
  divided over the run's calls, as energy per compute-millisecond (``CALL_ENERGY_J_PER_COMPUTE_MS``);
* desktop: Qwen CPU FFN per layer = twice the server's host branch at half width (same identity runs);
  host power per state fitted (least squares) to the four decode operating points WS2 measured (Qwen and
  Gemma, desktop-only and assisted), which also yields the Gemma CPU layer time;
* priors (flagged ``prior``) only where nothing is measured: Pixel f16 and Q4_0-packed compute rates.

``rig_inventory_v1`` describes today's models (CPU layers from the server's ``-ngl`` log lines), helpers
(capacity, sessions, shard formats, stored and qualified layers) and the static placement in use.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from research_dev.scheduler._internal.layer_placement import (
    DesktopPower,
    DeviceCostPriors,
    LayerPlacementProfile,
    PowerOperatingPoint,
    fit_desktop_power,
    ffn_layer_bytes,
)
from research_dev.scheduler._internal.layer_placement_io import (
    LAYER_PLACEMENT_INVENTORY_SCHEMA,
    shape_observations,
)


DATA = Path(__file__).resolve().parent / "data" / "layer_placement"
RECORDED_SHAPES = DATA / "RECORDED_SERVER_FFN_SHAPES_20261001.json"
QWEN = "qwen3-14b-q4km-dequant-f16"
GEMMA = "gemma-4-12b-q40-dequant-f16"
OP15 = "op15-phone"
PIXEL = "pixel10pro-phone"

QWEN_GEOMETRY = (5120, 17408, 40)          # n_embd, n_ff, block_count
GEMMA_GEOMETRY = (3840, 15360, 48)
QWEN_F16_LAYER = ffn_layer_bytes(5120, 17408, "F16")            # 534,773,760
GEMMA_F16_LAYER = ffn_layer_bytes(3840, 15360, "F16")           # 353,894,400
GEMMA_Q4_0_LAYER = ffn_layer_bytes(3840, 15360, "Q4_0")         # 99,532,800
QWEN_Q4K_LAYER = ffn_layer_bytes(5120, 17408, "Q4_K")           # 150,405,120
QWEN_Q4K_Q6K_DOWN_LAYER = ffn_layer_bytes(5120, 17408, "Q4_K", "Q4_K", "Q6_K")   # 173,383,680
# Qwen3-14B Q4_K_M (500a8806) keeps ffn_down in Q6_K for these layers (llama-quantize use_more_bits:
# i < n/8, i >= 7n/8, (i - n/8) % 3 == 2 with n = 40); verified on the file by native/pixel_packed_shard.py.
QWEN_Q6K_DOWN_LAYERS = tuple(i for i in range(40) if i < 5 or i >= 35 or (i - 5) % 3 == 2)

OP15_SESSION_LIMIT = 3_208_646_656        # models.json phone_resident_limit_bytes (6 Qwen f16 layers + 4 KiB)
OP15_SESSIONS = 3
OP15_LOWEST_LIVE_LIMIT = 9_273_319_424    # lowest live HTP limit seen (reports/20260924-phone-reprovision run 5)
PIXEL_ALLOCATION_LIMIT = 12 * 1024 ** 3   # helper evidence fragment allocation_limit_bytes

# Phone meters, block 1 two-phone arms (p0m1-3): (arm energy - idle W of the desktop arm x duration) over the
# run's calls, per compute-millisecond of those calls (fleet_block1 + RECORDED_SERVER_FFN_SHAPES).
CALL_ENERGY_J_PER_COMPUTE_MS = MappingProxyType({OP15: 0.0182, PIXEL: 0.0200})
CALL_ENERGY_EVIDENCE = "meters p0m1-3: OP15 0.129-0.145 J/call, Pixel 0.170-0.179 J/call above idle"
PHONE_IDLE_W = MappingProxyType({OP15: 1.11, PIXEL: 0.44})   # desktop arms p0m1-2 OP15, p0m1-3 Pixel

# Desktop: Qwen host FFN branch at half width 9.324 (adb) / 9.372 ms (AOA) per layer, server identity runs
# 2026-10-01 (columns 8704) -> 18.70 ms full layer.
QWEN_CPU_LAYER_MS = 9.348 * 2
DESKTOP_ROWS_FACTOR = MappingProxyType({1: 1.0, 2: 1.025, 3: 1.029, 4: 1.034})   # Qwen step 611/626/629/632 ms
# WS2 decode operating points (joint_planner_model.PROVENANCE): (step s, host W, CPU FFN layers, waits)
OPERATING_POINTS = (
    ("qwen desktop-only b1 (tp2 audit)", 0.611, 120.0, QWEN, 25, ()),
    ("qwen assisted b1 OP15 18 + Pixel 6 (s1c-s2a)", 0.489, 45.6, QWEN, 1, ((OP15, QWEN, 18), (PIXEL, QWEN, 6))),
    ("gemma desktop-only b1 (tp2 audit)", 0.464, 120.0, GEMMA, 27, ()),
    ("gemma assisted b1 OP15 24 (s1a-s2a)", 0.398, 50.1, GEMMA, 3, ((OP15, GEMMA, 24),)),
)

PIXEL_PRIORS = DeviceCostPriors(
    bytes_per_s=MappingProxyType({
        "q4k-packed": QWEN_Q4K_LAYER / 8.47e-3,     # measured packed compute (adb), as a rate
        "f16": QWEN_F16_LAYER / 18.291e-3,          # PIXEL_BANDWIDTH_RESULTS: six-thread f16 FFN 18.291 ms
        "q4_0-packed": QWEN_Q4K_LAYER / 8.47e-3 / 2,  # generic ggml Q4_0 x Q8_0 + residual: two weight passes
    }),
    overhead_ms=MappingProxyType({
        "adb-tcp": MappingProxyType({1: 4.92, 2: 5.35, 4: 6.5}),
        "aoa-bridge": MappingProxyType({1: 0.87, 2: 1.28, 4: 2.0}),
    }),
    rows_factor=MappingProxyType({1: 1.0, 2: 1.14, 3: 1.46, 4: 1.98}),
    active_marginal_w=CALL_ENERGY_J_PER_COMPUTE_MS[PIXEL] * 1000,
    idle_w=PHONE_IDLE_W[PIXEL],
    provenance=MappingProxyType({
        "q4k-packed": "measured rate (S41SERVERFFNSHAPE adb rows 1)",
        "f16": "derived: PIXEL_BANDWIDTH_RESULTS six-thread CPU f16 FFN, not in-server",
        "q4_0-packed": "prior: no Q4_0 packed kernel measured; two passes over the weights assumed",
        "adb-tcp": "measured (S41SERVERFFNSHAPE rpc - compute)",
        "aoa-bridge": "measured rows 1 (2026-10-01 identity run); rows 2 = half-width row; rows 4 prior",
        "rows_factor": "measured adb rows 2 + s1c rows 3/4",
    }),
)
OP15_PRIORS = DeviceCostPriors(
    bytes_per_s=MappingProxyType({"f16": QWEN_F16_LAYER / 9.258e-3}),
    overhead_ms=MappingProxyType({"functionfs-usb": MappingProxyType({1: 0.58, 2: 0.85, 3: 3.0, 4: 3.6})}),
    rows_factor=MappingProxyType({1: 1.0, 2: 1.013, 3: 1.07, 4: 1.23}),
    active_marginal_w=CALL_ENERGY_J_PER_COMPUTE_MS[OP15] * 1000,
    idle_w=PHONE_IDLE_W[OP15],
    provenance=MappingProxyType({"f16": "measured rate (S41SERVERFFNSHAPE rows 1)",
                                 "functionfs-usb": "measured rows 1/2, s1c rows 3/4",
                                 "rows_factor": "measured rows 2, s1c rows 3/4"}),
)
HELPER_LABELS = MappingProxyType({
    "op15": {"device_id": OP15, "format": "f16", "transport": "functionfs-usb"},
    "pixel10pro": {"device_id": PIXEL, "format": "q4k-packed", "transport": "adb-tcp"},
})
GEOMETRY = MappingProxyType({QWEN: QWEN_GEOMETRY, GEMMA: GEMMA_GEOMETRY})


def recorded_servers(path: Path = RECORDED_SHAPES) -> list[Mapping[str, object]]:
    return list(json.loads(path.read_text())["servers"])


def server_observations(server: Mapping[str, object]) -> list[dict[str, object]]:
    """One recorded server's summary rows as profile observations."""
    model = str(server["model"])
    labels = {key: dict(value) for key, value in HELPER_LABELS.items()}
    for helper in server.get("helpers", ()):
        if helper["transport"] == "aoa-bridge":
            labels[helper["label"]]["transport"] = "aoa-bridge"
    default = "op15" if len(server.get("helpers", ())) != 1 else server["helpers"][0]["label"]
    return shape_observations(server["shape_lines"], model_id=model, n_ff=GEOMETRY[model][1], helpers=labels,
                              default_helper=default, evidence=f"{server['run']}/{server['stderr']}")


def _wait_ms(profile: LayerPlacementProfile, device: str, model: str) -> float:
    shard_format = "q4k-packed" if device == PIXEL else "f16"
    transport = "adb-tcp" if device == PIXEL else "functionfs-usb"
    size = QWEN_Q4K_LAYER if device == PIXEL else (QWEN_F16_LAYER if model == QWEN else GEMMA_F16_LAYER)
    return profile.call_cost(device, model, shard_format, transport, 1, size).rpc_ms.value


def fit_desktop(profile: LayerPlacementProfile, qwen_layer_ms: float = QWEN_CPU_LAYER_MS) -> tuple[DesktopPower, float]:
    """Host power per state and the Gemma CPU layer time that best explain the four operating points.

    The Gemma layer time enters the fit bilinearly, so it is found by a 10-microsecond grid search; for each
    candidate the three powers are the linear least-squares solution. Returns (power, gemma_layer_ms).
    """
    best = None
    for step in range(800, 2001):
        gemma_ms = step / 100.0
        points = []
        for label, step_s, watts, model, cpu_layers, waits in OPERATING_POINTS:
            ffn_s = cpu_layers * (qwen_layer_ms if model == QWEN else gemma_ms) / 1000.0
            wait_s = sum(count * _wait_ms(profile, device, wait_model) for device, wait_model, count in waits) / 1000.0
            points.append(PowerOperatingPoint(ffn_s, wait_s, step_s - ffn_s - wait_s, watts * step_s, label))
        try:
            power = fit_desktop_power(points)
        except ValueError:
            continue
        residual = sum((p.ffn_s * power.ffn_w + p.wait_s * power.wait_w + p.other_s * power.other_w - p.energy_j) ** 2
                       for p in points)
        if best is None or residual < best[0]:
            best = (residual, power, gemma_ms)
    _, power, gemma_ms = best
    evidence = ("WS2 operating points (joint_planner_model PROVENANCE): " + "; ".join(row[0] for row in OPERATING_POINTS),
                f"fit residual {best[0]:.2e} J^2 over 4 points, 4 unknowns: exactly determined, no redundancy")
    return DesktopPower(power.ffn_w, power.wait_w, power.other_w, "derived", evidence), gemma_ms


def measured_profile_v1(*, shapes_path: Path = RECORDED_SHAPES) -> LayerPlacementProfile:
    profile = LayerPlacementProfile(
        desktop_power=DesktopPower(120.0, 30.0, 60.0, "prior"),
        desktop_bytes_per_s=QWEN_F16_LAYER / (QWEN_CPU_LAYER_MS / 1000.0),
        desktop_rows_factor=DESKTOP_ROWS_FACTOR,
        device_priors={OP15: OP15_PRIORS, PIXEL: PIXEL_PRIORS},
    )
    for server in recorded_servers(shapes_path):
        for row in server_observations(server):
            profile.observe_call_summary(**row)
    profile.observe_desktop_layer(QWEN, 1, QWEN_CPU_LAYER_MS,
                                  "2 x host branch at half width, server identity adb-r5/aoa-r4 2026-10-01")
    power, gemma_ms = fit_desktop(profile)
    profile.observe_desktop_power(power)
    profile.observe_desktop_layer(GEMMA, 1, gemma_ms, "fitted with the desktop power states (4 operating points)",
                                  provenance="derived")
    # call energy: meter-derived J per compute-ms x each measured bucket's compute time
    for key, stats in sorted(profile.calls.items(), key=lambda item: (item[0].device_id, item[0].model_id,
                                                                      item[0].rows, item[0].transport)):
        rate = CALL_ENERGY_J_PER_COMPUTE_MS.get(key.device_id)
        if rate is not None and (key.device_id, key.model_id, key.rows) not in profile.energy_per_call:
            profile.observe_call_energy(key.device_id, key.model_id, key.rows, rate * stats.compute_ms, 1,
                                        CALL_ENERGY_EVIDENCE)
    for (device, model, rows), estimate in list(profile.energy_per_call.items()):
        profile.energy_per_call[(device, model, rows)] = type(estimate)(estimate.value, "derived", estimate.evidence)
    return profile


def rows_mix_v1(path: Path = RECORDED_SHAPES) -> dict[str, dict[int, float]]:
    """Share of decode steps by rows, from the OP15's calls (it serves every assisted step of both models)."""
    steps: dict[str, dict[int, float]] = {}
    layers = {QWEN: 18, GEMMA: 24}
    for server in recorded_servers(path):
        for row in server_observations(server):
            if row["device_id"] == OP15:
                table = steps.setdefault(row["model_id"], {})
                table[row["rows"]] = table.get(row["rows"], 0.0) + row["calls"] / layers[row["model_id"]]
    return {model: {rows: value / sum(table.values()) for rows, value in sorted(table.items())}
            for model, table in steps.items()}


def busy_envelopes_v1(path: Path = RECORDED_SHAPES, sustained_calls: int = 2000) -> dict[str, float]:
    """Busy time per decode step each helper sustained in recorded runs: owned layers x mean round trip at
    the lowest row count, over servers with at least ``sustained_calls`` calls in that bucket."""
    default_layers = {QWEN: 18, GEMMA: 24}
    envelope: dict[str, float] = {}
    for server in recorded_servers(path):
        masks = {row["label"]: bin(int(row["layer_mask"])).count("1") for row in server.get("helpers", ())}
        for row in server_observations(server):
            if row["rows"] != 1 or row["calls"] < sustained_calls:
                continue
            label = next(key for key, value in HELPER_LABELS.items() if value["device_id"] == row["device_id"])
            layers = masks.get(label, default_layers[row["model_id"]])
            envelope[row["device_id"]] = max(envelope.get(row["device_id"], 0.0), layers * row["rpc_ms"])
    return envelope


def rig_inventory_v1(*, pixel_transport: str = "aoa-bridge", gemma_pixel_format: str = "q4_0-packed",
                     op15_capacity: int = OP15_SESSIONS * OP15_SESSION_LIMIT, op15_busy_cap_ms: float | None = None,
                     pixel_busy_cap_ms: float | None = None, available: Mapping[str, bool] | None = None,
                     gemma_desktop_ms: float | None = None, profile: LayerPlacementProfile | None = None) -> dict:
    """Today's rig as a placement inventory (JSON form)."""
    profile = profile or measured_profile_v1()
    qwen_ms = QWEN_CPU_LAYER_MS
    gemma_ms = gemma_desktop_ms or profile.desktop_ffn_ms[(GEMMA, 1)].value
    available = dict(available or {})
    envelope = busy_envelopes_v1()
    qwen_overrides = {str(layer): QWEN_Q4K_Q6K_DOWN_LAYER for layer in QWEN_Q6K_DOWN_LAYERS if layer <= 24}
    mix = rows_mix_v1()
    return {
        "schema": LAYER_PLACEMENT_INVENTORY_SCHEMA,
        "models": [
            {"model_id": QWEN, "cpu_layers": "0-24", "n_embd": 5120, "n_ff": 17408, "column_quantum": 4352,
             "layer_bytes": {"f16": QWEN_F16_LAYER, "q4k-packed": QWEN_Q4K_LAYER},
             "layer_bytes_overrides": {"q4k-packed": qwen_overrides},
             "rows_mix": {str(k): round(v, 6) for k, v in mix[QWEN].items()},
             "other_step_ms": {"1": round(611.0 - 25 * qwen_ms, 3)}},
            {"model_id": GEMMA, "cpu_layers": "0-26", "n_embd": 3840, "n_ff": 15360, "column_quantum": 3840,
             "layer_bytes": {"f16": GEMMA_F16_LAYER, "q4_0-packed": GEMMA_Q4_0_LAYER},
             "rows_mix": {str(k): round(v, 6) for k, v in mix[GEMMA].items()},
             "other_step_ms": {"1": round(464.0 - 27 * gemma_ms, 3)}},
        ],
        "devices": [
            {"device_id": OP15, "formats": {QWEN: "f16", GEMMA: "f16"}, "capacity_bytes": op15_capacity,
             "transport": "functionfs-usb", "sessions": OP15_SESSIONS, "session_limit_bytes": OP15_SESSION_LIMIT,
             "residency": "per-model", "available": available.get(OP15, True),
             "unavailable_reason": None if available.get(OP15, True) else "ABSENT",
             "max_busy_ms_per_token": op15_busy_cap_ms, "busy_envelope_ms": round(envelope[OP15], 3),
             # on the phone: qwen 0-17 (s42-ffn-shards-20260904-v1), gemma 0-23 (gemma24) and 0-25 (gemma26)
             "stored_layers": {QWEN: "0-17", GEMMA: "0-25"},
             "qualified_layers": {QWEN: "0-17", GEMMA: "0-23"}},
            {"device_id": PIXEL, "formats": {QWEN: "q4k-packed", GEMMA: gemma_pixel_format},
             "capacity_bytes": PIXEL_ALLOCATION_LIMIT, "transport": pixel_transport, "residency": "co-resident",
             "available": available.get(PIXEL, True),
             "unavailable_reason": None if available.get(PIXEL, True) else "ABSENT",
             "max_busy_ms_per_token": pixel_busy_cap_ms, "busy_envelope_ms": round(envelope[PIXEL], 3),
             "stored_layers": {QWEN: "18-23"}, "qualified_layers": {QWEN: "18-23"}},
        ],
        "current": {QWEN: {OP15: "0-17", PIXEL: "18-23"}, GEMMA: {OP15: "0-23"}},
        "latency_ppm": 1_250_000,
        "column_fractions": {QWEN: [1.0, 0.75, 0.5], GEMMA: [1.0, 0.75, 0.5]},
    }


def restart_energy_j() -> dict[str, float]:
    """Desktop reload energy per model (joint_planner_model load least squares: a + b x mean duration)."""
    return {QWEN: 1021.0 + 13.3 * 74.2, GEMMA: 1027.0 + 9.4 * 51.2}
