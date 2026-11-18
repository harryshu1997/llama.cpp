#!/usr/bin/env python3
"""GPU board sampler (NVML through nvidia-smi): gpu_sampler.py OUT STOPFILE [period_ms] [gpu_index]
Line: t_ns (CLOCK_MONOTONIC, stamped when the line arrives) then the nvidia-smi fields of FIELDS, comma separated."""
import os
import subprocess
import sys
import time

out, stop = sys.argv[1], sys.argv[2]
period_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 100
gpu_index = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "" else None
FIELDS = ["power.draw.instant", "power.draw.average", "temperature.gpu", "utilization.gpu", "utilization.memory",
          "memory.used", "clocks.current.graphics", "clocks.current.memory", "pstate", "clocks_event_reasons.active",
          "fan.speed"]
proc = subprocess.Popen(["nvidia-smi"] + (["-i", gpu_index] if gpu_index is not None else [])
                        + ["--query-gpu=" + ",".join(FIELDS), "--format=csv,noheader,nounits", "-lms", str(period_ms)],
                        stdout=subprocess.PIPE, text=True, bufsize=1)
with open(out, "w") as f:
    f.write("# t_ns," + ",".join(FIELDS) + "\n")
    for line in proc.stdout:
        f.write(f"{time.monotonic_ns()},{line.strip().replace(' ', '')}\n")
        f.flush()
        if os.path.exists(stop):
            break
proc.terminate()
