#!/usr/bin/env python3
"""Host sampler for the CPU legs: host_sampler.py OUT STOPFILE [period_s]
Line (CLOCK_MONOTONIC ns): t_ns pkg_uj core_uj temp_mc freq_khz_mean throttle_count   (-1 = not readable)."""
import glob
import os
import sys
import time

out, stop = sys.argv[1], sys.argv[2]
period = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
PKG = "/sys/class/powercap/intel-rapl:0/energy_uj"
CORE = "/sys/class/powercap/intel-rapl:0:0/energy_uj"


def read(path):
    try:
        with open(path) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return -1


def temp_path():
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            name = open(h + "/name").read().strip()
        except OSError:
            continue
        if name in ("coretemp", "k10temp"):
            for label in sorted(glob.glob(h + "/temp*_label")):
                text = open(label).read().strip()
                if text.startswith("Package") or text in ("Tctl", "Tdie"):
                    return label.replace("_label", "_input")
            return h + "/temp1_input"
    return None


TEMP = temp_path()
FREQS = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"))
THROTTLE = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/thermal_throttle/package_throttle_count"))[:1]
max_uj = read("/sys/class/powercap/intel-rapl:0/max_energy_range_uj")
with open(out, "w") as f:
    f.write(f"# period_s {period} temp {TEMP} max_energy_range_uj {max_uj} cpus {len(FREQS)}\n")
    n = 0
    freq = -1
    while not os.path.exists(stop):
        if n % 10 == 0:
            values = [read(p) for p in FREQS]
            values = [v for v in values if v > 0]
            freq = sum(values) // len(values) if values else -1
        f.write(f"{time.monotonic_ns()} {read(PKG)} {read(CORE)} {read(TEMP) if TEMP else -1} {freq} "
                f"{read(THROTTLE[0]) if THROTTLE else -1}\n")
        f.flush()
        n += 1
        time.sleep(period)
