"""Synthetic run directories for the portable-path tests (no hardware, synthetic ids)."""
from __future__ import annotations

import json
from pathlib import Path

N_FF = 6400
SECOND = 1_000_000_000


def power_samples(start_ns: int, end_ns: int, power_at, period_ns: int = SECOND // 10) -> list[dict]:
    """Host sampler rows; ``power_at(t_ns)`` -> (package W, board W), piecewise constant between samples."""
    rows, energy_uj, instant = [], 1_000_000, start_ns
    while instant <= end_ns:
        cpu_w, gpu_w = power_at(instant)
        rows.append({"t_ns": instant,
                     "gpu": {"sample_t_ns": instant, "power_mw": int(round(gpu_w * 1000))},
                     "rapl_package": {"sample_t_ns": instant, "energy_uj": int(round(energy_uj)),
                                      "max_energy_range_uj": 1 << 40, "name": "package-0"}})
        energy_uj += cpu_w * period_ns / 1e3      # W * ns = nJ -> uJ
        instant += period_ns
    return rows


def request(request_id, columns, submitted_ns, first_ns, finished_ns, output_tokens, *, sha="same",
            target_ns=None, input_tokens=16) -> dict:
    steps = output_tokens - 1
    return {"id": request_id, "request_id": f"{request_id}-c{columns}", "columns": columns,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "target_ns": submitted_ns if target_ns is None else target_ns, "submitted_ns": submitted_ns,
            "first_token_ns": first_ns, "finished_ns": finished_ns, "tokens_sha256": sha,
            "progress": [[index + 1, (finished_ns - first_ns) * (index + 1) // steps // 1000]
                         for index in range(steps)]}


def phase(name, requests) -> dict:
    origin = min(row["target_ns"] for row in requests)
    end = max(row["finished_ns"] for row in requests)
    return {"name": name, "origin_ns": origin, "end_ns": end, "service_s": (end - origin) / 1e9,
            "attempts": 1, "requests": list(requests)}


def write_run(path: Path, *, phases, samples, mode="shared", layers=(0, 1), idle=None, tail_idle=None,
              phone_lines=None, phone_clock=None, phone_power=None, shapes=None, extra_result=None) -> Path:
    path.mkdir(parents=True)

    def save(name, value):
        (path / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")

    helper = None if mode == "alone" else {"kind": "synthetic-helper", "layers": list(layers), "columns": N_FF // 2}
    save("CONFIG.json", {"alias": "synthetic-model", "phone": {"columns": N_FF, "n_embd": 64}, "helper": helper,
                         "server_arguments": ["-t", "8", "--parallel", "4"]})
    save("RESULT.json", {"status": "PASS", "mode": mode, "phone_calls": 0 if mode == "alone" else 100,
                         **(extra_result or {})})
    save("POWER_SAMPLES.json", samples)
    for number, row in enumerate(phases):
        save(f"PHASE-{number}.json", row)
    for name, window in (("IDLE.json", idle), ("TAIL_IDLE.json", tail_idle)):
        if window is not None:
            start, end, cpu_w, gpu_w = window
            save(name, {"started_ns": start, "finished_ns": end,
                        "host_energy": {"cpu_package_average_power_w": cpu_w, "gpu_board_average_power_w": gpu_w}})
    if phone_clock is not None:
        save("PHONE_CLOCK.json", {"phone_uptime_s": phone_clock[0], "host_monotonic_ns": phone_clock[1]})
    if phone_lines is not None:
        (path / "phone-power.txt").write_text("\n".join(phone_lines) + "\n", encoding="ascii")
    if phone_power is not None:
        save("PHONE_POWER.json", phone_power)
    if shapes is not None:
        save("SERVER_FFN_SHAPES.json", ["S41SERVERFFNSHAPE " + json.dumps(row) for row in shapes])
    return path


def phone_lines(start_uptime_s: float, seconds: int, *, usb_w: float, battery_ma: float = 0.0,
                battery_uv: float = 4_000_000.0) -> list[str]:
    """One sampler line per second at a constant USB input power (5 V) and gauge current."""
    usb_ua = usb_w / 5.0 * 1e6
    return ["# start uptime %.2f period_s 1" % start_uptime_s] + [
        f"{start_uptime_s + index:.2f} usb_ua={usb_ua:.0f} usb_uv=5000000 bat_ma={battery_ma:g} "
        f"bat_uv={battery_uv:.0f} battery=30000 note=text" for index in range(seconds)]
