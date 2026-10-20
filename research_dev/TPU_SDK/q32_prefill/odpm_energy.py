"""ODPM rail energy over the probe's timed window vs an idle baseline (Pixel 10 Pro, iio:device0).

usage: odpm_energy.py RUN_DIR TAG M [IDLE_S]
Samples: odpm.TAG.txt (root sampler, ~0.1 s cadence: "t=<ms>" + "CHn(T=<ms>)[RAIL], <uJ>").
Window: "timed_boottime_ms begin=.. end=.." from TAG.log (CLOCK_BOOTTIME, same base as ODPM T).
Idle baseline: the first IDLE_S seconds of samples (sampler starts 3 s before the probe).
Energy per invoke = (window energy - idle power * window) / timed calls. Units: uJ counters -> J.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

RAILS_OF_INTEREST = ["S7M_VDD_TPU", "S9M_VDD_TPU_M", "S4S_VDD2H_MEM", "S5S_VDDQ_MEM", "S8S_VDD_GMC",
                     "S9S_VDD_INFRA", "S2M_VDD_CPU2", "S3M_VDD_CPU1", "S4M_VDD_CPU", "VSYS_PWR_VBATT"]


def main():
    run, tag, m = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    idle_s = float(sys.argv[4]) if len(sys.argv) > 4 else 2.5
    text = (run / f"odpm.{tag}.txt").read_text()
    series = {}
    for t, rail, val in re.findall(r"CH\d+\(T=(\d+)\)\[([^\]]+)\], (\d+)", text):
        series.setdefault(rail, []).append((int(t), int(val)))
    log = (run / f"{tag}.log").read_text()
    b, e = map(float, re.search(r"timed_boottime_ms begin=([\d.]+) end=([\d.]+)", log).groups())
    calls = int(re.search(r"timed_window_ns=\d+ calls=(\d+)", log).group(1))
    out = {"tag": tag, "M": m, "window_s": (e - b) / 1e3, "timed_calls": calls, "rails": {}}
    for rail, pts in series.items():
        pts = sorted(set(pts))
        t = np.array([p[0] for p in pts], dtype=np.float64)
        v = np.array([p[1] for p in pts], dtype=np.float64)
        if t[0] > b or t[-1] < e:
            continue
        e_win = (np.interp(e, t, v) - np.interp(b, t, v)) / 1e6           # J
        t0 = t[0]
        idle_end = t0 + idle_s * 1e3
        p_idle = (np.interp(idle_end, t, v) - v[0]) / 1e6 / (idle_s)       # W
        p_win = e_win / ((e - b) / 1e3)
        out["rails"][rail] = {"window_J": e_win, "window_W": p_win, "idle_W": p_idle,
                              "net_J_per_invoke": (e_win - p_idle * (e - b) / 1e3) / calls}
    keys = [r for r in RAILS_OF_INTEREST if r in out["rails"]]
    for r in keys:
        x = out["rails"][r]
        print(f"{tag:16s} {r:16s} win {x['window_W']:7.3f} W idle {x['idle_W']:7.3f} W net/invoke {x['net_J_per_invoke']:7.3f} J")
    tpu = sum(out["rails"][r]["net_J_per_invoke"] for r in ("S7M_VDD_TPU", "S9M_VDD_TPU_M") if r in out["rails"])
    mem = sum(out["rails"][r]["net_J_per_invoke"] for r in ("S4S_VDD2H_MEM", "S5S_VDDQ_MEM", "S8S_VDD_GMC") if r in out["rails"])
    batt = out["rails"].get("VSYS_PWR_VBATT", {}).get("net_J_per_invoke")
    out["summary"] = {"tpu_rails_J_per_invoke": tpu, "dram_gmc_rails_J_per_invoke": mem, "vbatt_net_J_per_invoke": batt,
                      "tpu_rails_mJ_per_token": tpu / m * 1e3}
    print(json.dumps(out["summary"]))
    (run / f"odpm.{tag}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
