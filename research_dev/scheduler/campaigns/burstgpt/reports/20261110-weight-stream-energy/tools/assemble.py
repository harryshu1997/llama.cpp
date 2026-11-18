#!/usr/bin/env python3
"""Assemble RESULTS.md from the generated fragments (run make_results.sh first)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
B = Path("/mnt/data_h16t/zs89458/xband-runs/stream-energy-20261110")
T = R / "tables"
summary = json.loads((T / "SUMMARY.json").read_text())
net = json.loads((T / "SUMMARY_NET.json").read_text())
cal = {d: json.loads((B / d / "CALIBRATION.json").read_text()) for d in ("op15", "op12", "pixel")}
bytes_ = json.loads((T / "MODEL_BYTES.json").read_text())
q4k = json.loads((B / "pixel-q4k/RESULTS.json").read_text()) if (B / "pixel-q4k/RESULTS.json").exists() else []
dev = summary["devices"]
names = list(dev)
cpu_measured = dev["Desktop CPU"]["j_per_gb_origin"] is not None or dev["Server CPU"]["j_per_gb_origin"] is not None

LABEL = {"qwen25-05b": "Qwen2.5-0.5B", "qwen3-06b": "Qwen3-0.6B *", "llama32-1b": "Llama-3.2-1B",
         "qwen3-17b": "Qwen3-1.7B *", "gemma4-e2b": "Gemma-4-E2B", "qwen3-4b": "Qwen3-4B *", "qwen3-8b": "Qwen3-8B",
         "gemma4-12b": "Gemma-4-12B", "qwen3-14b": "Qwen3-14B", "qwen3-32b": "Qwen3-32B"}
SHORT = {"OnePlus 15 NPU": "OP15 NPU", "OnePlus 12 NPU": "OP12 NPU", "OnePlus 11 NPU": "OP11 NPU",
         "Pixel 10 Pro CPU": "Pixel CPU", "Desktop GPU": "Desktop GPU", "Desktop CPU": "Desktop CPU",
         "Server CPU": "Server CPU", "Server GPU": "Server GPU"}
gpu = json.loads((B / "desktop-gpu/RESULTS.json").read_text()) if (B / "desktop-gpu/RESULTS.json").exists() else []
server_gpu = json.loads((B / "server-gpu/RESULTS.json").read_text()) if (B / "server-gpu/RESULTS.json").exists() else []

server_repeat = ""
if (T / "SERVER_CPU_REPEAT.json").exists():   # second server CPU leg, 11-12: reported as a repeatability check
    rep = json.loads((T / "SERVER_CPU_REPEAT.json").read_text())["devices"]["Server CPU repeat"]
    first = {r["tag"]: r for r in json.loads((B / "server-cpu/RESULTS.json").read_text())}
    again = {r["tag"]: r for r in json.loads((B / "server-cpu-20261112/RESULTS.json").read_text())}
    big = [t for t in again if t in first and again[t]["mb"] >= 1000]
    worst = max(abs(again[t]["j_per_gb"] / first[t]["j_per_gb"] - 1) for t in big)
    quiet = [r["idle_w"] for r in again.values() if r["idle_w"] < 90]
    busy = [r["idle_w"] for r in again.values() if r["idle_w"] >= 90]
    hot = [r for r in again.values() if not r["cool"]]
    server_repeat = (
        f"- Server CPU repeat (11-12, after the A6000 GPU leg, `server-cpu-20261112`): fit "
        f"{rep['j_per_gb_origin']:.2f} J/GB over {rep['used']} runs against {dev['Server CPU']['j_per_gb_origin']:.2f} "
        f"on 11-11, DRAM-bound {rep['dram_bound_gb_per_s']:.1f} GB/s against "
        f"{dev['Server CPU']['dram_bound_gb_per_s']:.1f}; the {len(big)} runs of 1 GB and more agree within "
        f"{worst * 100:.1f} % in J/GB.\n  Idle package {min(quiet):.0f}-{max(quiet):.0f} W before {len(quiet)} runs"
        + (f" and {min(busy):.0f}-{max(busy):.0f} W before the last {len(busy)} (other activity on this shared host)"
           if busy else "")
        + (f"; {len(hot)} runs started at {min(r['pre_temp_c'] for r in hot):.1f}-{max(r['pre_temp_c'] for r in hot):.1f} C, "
           f"above the 76 C start gate, after the 5-minute wait limit" if hot else "")
        + ". The tables use the 11-11 leg.\n")

thread_sweep = ""
SWEEP = B / "server-cpu-threads-20261113"
if SWEEP.exists():   # Threadripper thread sweep on one DRAM-bound set (Qwen3-14B, 8 layers), one run per setting
    rows = []
    for t in (64, 32, 16, 8, 4):
        f = SWEEP / f"t{t}" / "RESULTS.json"
        if f.exists():
            r = json.loads(f.read_text())[0]
            rows.append(f"| {t} | {r['gb_per_s']:.1f} | {r['watts']:.0f} | {r['idle_w']:.0f} | {r['j_per_gb']:.2f} | {r['net_j_per_gb']:.2f} |")
    thread_sweep = ("## Server CPU thread sweep (11-13)\n\nQwen3-14B, 8 layers (4.28 GB working set), one run per setting, same protocol. "
                    "The tables above use 64 threads, the fastest setting; 32 threads stream as fast for less power, and "
                    "16 threads give the lowest energy per GB.\n\n"
                    "| Threads | GB/s | Package W | Idle W | J/GB | J/GB above idle |\n|---:|---:|---:|---:|---:|---:|\n"
                    + "\n".join(rows) + "\n\n")


def num(value, digits=3):
    if value is None:
        return "-"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.{digits}f}" if value < 1 else f"{value:.2f}"


def g(name, src=dev):
    v = src[name].get("j_per_gb_origin")
    return "pending" if v is None and name in ("Desktop CPU", "Server CPU") else ("no power data" if v is None else f"{v:.3f}")


energy_names = [n for n in names if n != "OnePlus 11 NPU"]
final = ["| Model | GB/token | " + " | ".join(f"{SHORT[n]} J/GB | {SHORT[n]} J/token" for n in energy_names) + " |",
         "|---|---:|" + "---:|---:|" * len(energy_names)]
for model in bytes_:
    row = [LABEL[model], f"{bytes_[model]['streamed_bytes'] / 1e9:.2f}"]
    for n in energy_names:
        pm = dev[n]["per_model"].get(model, {})
        mark = " †" if pm.get("from_device_fit") and pm.get("j_per_gb") is not None else ""
        row += [num(pm.get("j_per_gb")) + mark, num(pm.get("j_per_token"), 2) + mark]
    final.append("| " + " | ".join(row) + " |")
final.append("| **Device fit, all runs** | | " + " | ".join(f"**{num(dev[n]['j_per_gb_origin'])}** | " for n in energy_names) + " |")
final_table = "\n".join(final)

bw = ["| Model | " + " | ".join(f"{SHORT[n]} GB/s | s/token" for n in names) + " |", "|---|" + "---:|---:|" * len(names)]
for model in bytes_:
    cells = []
    for n in names:
        pm = dev[n]["per_model"].get(model)
        if not pm:
            cells += ["-", "-"]
            continue
        mark = " †" if pm.get("from_device_fit") else ""
        cells += [f"{pm['gb_per_s']:.1f}{mark}", f"{bytes_[model]['streamed_bytes'] / 1e9 / pm['gb_per_s']:.2f}{mark}"]
    bw.append(f"| {LABEL[model]} | " + " | ".join(cells) + " |")
bandwidth_table = "\n".join(bw)

rows = []
for model, m in bytes_.items():
    b = m["bytes"]
    rows.append(f"| {LABEL[model]} | {m['architecture']} | {m['blocks']} | {m['streamed_bytes']/1e9:.2f} | {b.get('ffn',0)/1e9:.2f} | "
                f"{b.get('attention',0)/1e9:.2f} | {b.get('output_head',0)/1e9:.2f} | {m['ffn_share']*100:.0f} % |")
model_table = "\n".join(["| Model | Arch | Blocks | GB/token | FFN | Attention | Head | FFN share |",
                         "|---|---|---:|---:|---:|---:|---:|---:|"] + rows)

fit_rows = "\n".join(
    f"| {n} | {dev[n]['used']}/{dev[n]['runs']} | {g(n)} | "
    + (f"{dev[n]['with_intercept']['slope_j_per_gb']:.3f} + {dev[n]['with_intercept']['intercept_j']*1e3:.1f} mJ (R2 {dev[n]['with_intercept']['r2']:.4f})" if dev[n].get("with_intercept") else "-")
    + f" | {g(n, net['devices'])} | {', '.join(dev[n]['flagged']) or 'none'} |" for n in names)

q4k_rows = "\n".join(
    f"| {r['layers']} | {r['mb']:.0f} | {r['layers'] * 534.77376:.0f} | {r['compute_ms']:.2f} | {r['compute_ms'] / r['layers']:.2f} | {r['gb_per_s']:.1f} | "
    f"{r['watts']:.2f} | {r['j_per_call']:.4f} | {r['j_per_gb']:.3f} | {r['j_per_call'] / (r['layers'] * 0.53477376):.3f} | "
    f"{r['hot']['pre_c']:.0f}-{r['hot']['end_c']:.0f} | {'YES ' + format(r['drift'] * 100, '+.0f') + ' %' if r['throttled'] else 'no'} |" for r in q4k)
q4k_mean = sum(r["j_per_call"] / (r["layers"] * 0.53477376) for r in q4k) / len(q4k) if q4k else None

def span(values, spec=".0f"):
    values = [v for v in values if v is not None]
    return "-" if not values else (format(min(values), spec) if format(min(values), spec) == format(max(values), spec)
                                   else f"{format(min(values), spec)}-{format(max(values), spec)}")


def gpu_text(runs, name, what_row, host_idle, gate):
    """Rows and notes of one GPU leg (board power from NVML; host CPU package next to it)."""
    big = [r for r in runs if r["mb"] >= 1000]
    small = [r for r in runs if r["mb"] < 1000]
    with_cpu = [r for r in runs if r["cpu_package_w"]]
    sys_fit = (sum(r["mb"] / 1e3 * (r["watts"] + r["cpu_package_w"]) * r["compute_ms"] / 1e3 for r in with_cpu)
               / sum((r["mb"] / 1e3) ** 2 for r in with_cpu)) if with_cpu else None
    agree = max(abs(r["watts_nvml_1s_average"] / r["watts"] - 1) for r in runs if r["watts_nvml_1s_average"])
    flagged = [r["tag"] for r in runs if r["throttled"]]
    capped = [r for r in runs if r.get("power_capped")]
    note = (f" {name}: the NVML 1-s average over the same window agrees within {agree * 100:.1f} %; the host thread that feeds the GPU"
            f" holds the CPU package at {span([r['cpu_package_w'] for r in runs])} W ({host_idle} idle)"
            + (f", and with that package power added the fit is {sys_fit:.3f} J/GB." if sys_fit else "."))
    cap = ("" if not capped else
           f" {len(capped)} of {len(runs)} runs sat at the board's power limit ({span([r['watts'] for r in capped])} W; NVML software power cap,"
           f" graphics clock {span([r['gr_mhz'] for r in capped])} MHz): this is the card's normal operating point under this load, so those"
           f" runs stay in the fits.")
    flag = (f"- {name}: {span([r['gb_per_s'] for r in big])} GB/s at {span([r['watts'] for r in big])} W on working sets of 1 GB and more, "
            f"{span([r['gb_per_s'] for r in small])} GB/s at {span([r['watts'] for r in small])} W on smaller ones (per-kernel launch cost). "
            f"Board idle {span([r['idle_w'] for r in runs])} W with no CUDA context, {span([r['resident_idle_w'] for r in runs])} W with the weights resident. "
            f"GPU temperature {span([r['temp_start_c'] for r in runs])} C at the start of the measured window, at most {max(r['temp_max_c'] for r in runs):.0f} C; "
            f"NVML reported no thermal slowdown{'' if not flagged else ' except in ' + ', '.join(flagged)}.{cap} {gate}\n")
    return what_row, note, flag


gpu_row = gpu_flag = gpu_note = ""
if gpu:
    row, note, flag = gpu_text(
        gpu, "Desktop GPU",
        "| Desktop (4060 Ti rig) | GPU RTX 4060 Ti, 16 GB | `ffn-stream-bench --device CUDA0` on the deployed `libggml-cuda.so` "
        "| weights resident in GPU memory, one host thread feeds the GPU |\n",
        "2 W", "The card stops its fans when idle and settles at 42-43 C, so the start gate is 44 C.")
    gpu_row += row; gpu_note += note; gpu_flag += flag
if server_gpu:
    row, note, flag = gpu_text(
        server_gpu, "Server GPU",
        "| A6000 server FCHLLX01 | GPU RTX A6000, 48 GB (one of two, index 1) | `ffn-stream-bench --device CUDA0` on `build-cuda-s43` "
        "| weights resident in GPU memory, one host thread feeds the GPU |\n",
        "66-78 W", "Start gate 45 C (the card idles at 38-40 C and needs about four minutes to return there).")
    gpu_row += row; gpu_note += note; gpu_flag += flag
if gpu_note:
    gpu_note = (" GPU board power = NVML instantaneous board power (`nvidia-smi`, 10 Hz), time-weighted over the measured"
                " window. A GPU row counts the board only." + gpu_note)

cpu_section = "" if cpu_measured else f"""## CPU legs: energy pending

`/sys/class/powercap/intel-rapl:0/energy_uj` is mode 0400 root on both hosts since their last boot and there is no
passwordless sudo, so the CPU runs carry compute time, GB/s and temperature only. To finish:

    sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj      # once on the desktop and once on FCHLLX01
    bash tools/rerun_cpu_energy.sh desktop                          # ~45 min each, host otherwise idle
    bash tools/rerun_cpu_energy.sh server
    bash tools/make_results.sh && python3 tools/assemble.py

Earlier measurements on the same desktop bracket what to expect (not from this session, not in the table):

| Source | Regime | Package W | GB/s | J/GB |
|---|---|---:|---:|---:|
| 10-13 single-request decode, Qwen3-14B f16, CPU layers, `-t 8` | continuous decode (closest to this benchmark) | 90 | about 29 | about 3.1 |
| same, `-t 4` | continuous decode | 60 | about 29 | about 2.1 |
| 10-07 microbenchmark, one Qwen3-14B FFN layer, 8 threads | serving cadence (calls 5 ms apart, 300 ms per token) | 26 above idle | 29.1 | 0.91 above idle |

At about 3 J/GB a full Qwen3-14B token would cost the desktop CPU about 85 J, against 4.3-4.5 J on the NPUs and
9.8 J on the Pixel CPU.

"""

text = f"""# Weight-streaming energy per device (2026-11-10 / 11 UTC)

One decode call = the FFN (gate, up, activation, down) of 2, 4 or 8 consecutive layers of an f16 model at batch 1,
repeated back to back. Each device runs alone; only the energy that device draws while it computes is counted
(no desktop waiting, no USB transfer: the client runs on the device itself).

## Result: J/GB and J/token

Gross device energy (device watts x compute time). J/token = model J/GB x the GB a full model streams per decoded
token (every block matrix plus the output head, two bytes per weight), whether or not the device could hold it.

{final_table}

`*` = extra model downloaded for this run (newer, similar size). `†` = every run of that model sat in the CPU
cache, so the device's DRAM-bound fit is used. `-` = {'not measured' if cpu_measured else 'pending: the CPU package energy counter (RAPL) is root-only on both hosts (see "CPU legs")'}.
OnePlus 11 has no root, so it has bandwidth only (below).

Device fits over all unflagged runs (J per call against GB per call):

| Device | Runs used | J/GB (through origin) | J/GB + fixed J/call (with intercept) | J/GB above idle | Flagged runs |
|---|---:|---:|---|---:|---|
{fit_rows}

"Above idle" subtracts the device's idle power measured before each run.

Bandwidth (weight bytes / compute time over the model's unflagged, DRAM-bound runs) and the time a full model
would take per token at that rate:

{bandwidth_table}

## Pixel 10 Pro production path (packed Q4_K), for reference

The paper's Pixel does not stream f16: its qualified worker streams the original Q4_K/Q6_K bytes with int8
activations. Same protocol, Qwen3-14B layers 16-23 of the production shard `QWEN_PACKED_1_23.ffn.gguf`:

| Layers | Physical MB | f16-equivalent MB | Compute ms | ms per layer | Physical GB/s | Device W | J/call | J per physical GB | J per f16-equivalent GB | Big-core C | Throttled |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
{q4k_rows}

Per layer this path is about 3.3x faster and 2.5x cheaper than the Pixel's f16 path ({num(q4k_mean)} against
{num(dev['Pixel 10 Pro CPU']['per_model']['qwen3-14b']['j_per_gb'])} J per f16-equivalent GB), at 12.5 W and with the big cores at 93-96 C.

## What ran

| Device | Engine | Software | Setting |
|---|---|---|---|
| OnePlus 15 (SM8850) | NPU, Hexagon v81 | production FFN worker (09-04 build), backend HTP0, 4 HVX threads | device-local client over loopback, no USB traffic during a run |
| OnePlus 12 (SM8650) | NPU, Hexagon v75 | same worker source, 10-09 build + v75 skeleton built 11-10 | same |
| OnePlus 11 (SM8550) | NPU, Hexagon v73 | same worker source, 09-21 build | same; no root, so no power data |
| Pixel 10 Pro (Tensor G5) | CPU, 6 threads on cores 2-7 | deployed Pixel CPU worker (09-25 build), ggml f16 kernel, production DVFS settings (uclamp 1024, poll 100) | same |
{gpu_row}| Desktop (4060 Ti rig) | CPU i9-12900K | `ffn-stream-bench` on the deployed `libggml-cpu.so` | 8 threads (the paper's server setting) |
| A6000 server FCHLLX01 | CPU Threadripper PRO 5995WX | `ffn-stream-bench` on `build-cpu` | 64 threads (fastest of 16-128) |

Not measured: the Ryzen 7 9700X desktop (not available). CPU package energy = RAPL package counter, sampled at
10 Hz, over the same measured window (readable after `sudo chmod a+r` on 11-11; resets at reboot).{gpu_note}

Protocol per run: sampler starts, 5 s idle, load the N-layer shard, idle window with the weights resident
(12 s phone, 10 s host), warm-up (2-3 s), measured window (20 s) of back-to-back calls. Phone runs wait for the
SoC zone and the skin sensor to fall back under a start gate (OnePlus 15: NPU 43 C, back shell 36.8 C, 37.2 C
for the last four models; OnePlus 12: 40 / 33 C; Pixel: big cores 36-40 C, skin 31 C then 34 C); host runs wait
for the package temperature to return to its idle level; GPU runs (added 11-12) wait for 44 C (4060 Ti) or 45 C (A6000).

Models (first 8 layers as one f16 shard each; BF16 parents converted to F16):

{model_table}

Qwen3-8B, Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B (BF16) and Qwen2.5-0.5B (f16) were downloaded anonymously from Hugging
Face and checked against the published SHA-256. Qwen3-32B was BF16 on disk. Qwen2.5 (architecture `qwen2`) is
served under the shard architecture `qwen3` (same bias-free SwiGLU FFN; the worker only accepts three names).
Gemma-4-E2B layers 0-14 have n_ff 6,144 (measured); layers 15-34 have 12,288.

## Phone energy

Phone power = USB input (V x I) + battery discharge (battery current x voltage / ratio). The ratio is a session
calibration of the battery current against the coulomb counter over the run windows, with the counter converted
to design-capacity mAh (design x SOH / gauge full-charge capacity). The OPLUS gauge refreshes `current_now` every
~5.2 s, so a OnePlus battery reading is used only from 10.8 s after the load starts; the Pixel gauge refreshes
in under 0.5 s.

| Phone | Counter drop in run windows | current x V / counter x V | Capacity scale | Ratio used |
|---|---:|---:|---:|---:|
| OnePlus 15 | {cal['op15']['run_windows']['counter_j']:.0f} J | {cal['op15']['run_windows']['ratio']:.3f} | {cal['op15']['capacity_scale']:.3f} | {cal['op15']['ratio_for_power']:.3f} |
| OnePlus 12 | {cal['op12']['run_windows']['counter_j']:.0f} J | {cal['op12']['run_windows']['ratio']:.3f} | {cal['op12']['capacity_scale']:.3f} | {cal['op12']['ratio_for_power']:.3f} |
| Pixel 10 Pro | {cal['pixel']['run_windows']['counter_j']:.0f} J | {cal['pixel']['run_windows']['ratio']:.3f} | 1.000 | {cal['pixel']['ratio_for_power']:.3f} |

For the OnePlus phones an ideal series two-cell pack would give 0.5; using 0.5 instead lowers OnePlus 15 device
watts by about {(1 - (2.46 + 9.0 * cal['op15']['ratio_for_power'] / 0.5) / (2.46 + 9.0)) * 100:.0f} % and OnePlus 12 by about {(1 - (2.23 + 4.0 * cal['op12']['ratio_for_power'] / 0.5) / (2.23 + 4.0)) * 100:.0f} %. Treat OnePlus watts as +-10 %; the Pixel's
current and counter agree to 0.2 %.

## Per-run results

{(T / 'RUNS.md').read_text().strip()}

## Flags

- OnePlus 15, thermal: Android reported thermal status 1 in every run (it does so from about 34 C battery; the
  phone sat at 35-38 C). No NPU, DDR or CPU cooling device engaged in any run of the final campaign. The vendor
  policy holds the prime CPU cluster at 2,438 MHz (of 4,608) from the first second of load.
- OnePlus 15, pilot runs (33 s of streaming, 45-100 s cool-down, in `op15-pilot-33s/`): two runs throttled, call
  time +15-19 % after the back shell passed about 38.2 C (status 2, CPU clusters capped to 61 % / 35 %). The final
  protocol (22 s, shell-temperature gate) avoids this; those pilot runs are not in the tables.
- OnePlus 15, two operating points: with the prime cluster at 2.4 GHz the NPU path streams 60-72 GB/s at 10-13 W;
  when the governor leaves the CPUs near 0.9 GHz (long calls: Qwen3-8B and larger) it streams about 55 GB/s at
  about 9 W, about 10 % less energy per GB. `qwen3-8b-L4` switched between the two while measured and is
  excluded from the fits.
- OnePlus 12: no thermal status change, no cooling device, CPUs stay at 0.5-0.8 GHz in every run.
- OnePlus 11: 25 of 30 runs; five runs with one NPU buffer of 2.4 GB or more failed to allocate (`rpcmem_alloc`).
  `gemma4-12b-L4` sped up 6 % mid-run (unsteady, excluded).
- Pixel 10 Pro, f16: no cooling device and no slowdown in any run; big cores reach 80-89 C within 22 s, thermal
  status 0 throughout. Production path: big cores 93-96 C, the 8-layer run slowed 6 % (flagged).
- Both rooted OnePlus phones show `battery_notify_code` 512 (charging-overtime latch): they do not recharge until
  replugged or rebooted. OnePlus 15 went from 79 % to 73 %, OnePlus 12 from 70 % to 68 %.
- CPU legs: runs whose GB/s exceeds 1.15x the device's large-working-set rate are served partly from the CPU cache
  (desktop: up to about 150 MB; server: up to about 1 GB against a 256 MB L3). A full model never fits, so those
  runs are excluded from the fits and the J/token extrapolation.
{gpu_flag}- Desktop CPU: package 89-92 W on DRAM-bound sets (up to 129 W on small cached sets) against 2 W idle; no run
  slowed; package temperature at most 55 C.
- Server CPU: package 275-286 W on DRAM-bound sets against 66-78 W idle (94-96 W before the two Qwen3-14B runs:
  other activity on this shared host); flagged: {', '.join(dev['Server CPU']['flagged']) or 'none'} (calls slowed
  more than 5 % within the run). Tctl 66-88 C; the host idles at 57-75 C.
{server_repeat}
{cpu_section}{thread_sweep}## Files

- Tools: `tools/` (shard maker, device-local client, host bench, samplers, runners, analysis, this assembler).
- Raw runs: `/mnt/data_h16t/zs89458/xband-runs/stream-energy-20261110/` (`op15`, `op12`, `op11`, `pixel`,
  `pixel-q4k`, `desktop-gpu`, `server-gpu`, `desktop-cpu*`, `server-cpu*`, pilots). Shards: `/mnt/data_h16t/zs89458/xband-data/models/stream-energy-shards/`.
- Phone staging: `/data/local/tmp/stream-energy-20261110/` on each phone (scripts, client, run logs; shards
  removed from the OnePlus 12, OnePlus 11 and Pixel, still on the OnePlus 15: 19 GB).
"""
(R / "RESULTS.md").write_text(text)
print("wrote", R / "RESULTS.md", len(text.splitlines()), "lines")
