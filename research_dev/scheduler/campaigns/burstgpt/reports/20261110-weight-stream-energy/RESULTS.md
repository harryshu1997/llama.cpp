# Weight-streaming energy per device (2026-11-10 / 11 UTC)

One decode call = the FFN (gate, up, activation, down) of 2, 4 or 8 consecutive layers of an f16 model at batch 1,
repeated back to back. Each device runs alone; only the energy that device draws while it computes is counted
(no desktop waiting, no USB transfer: the client runs on the device itself).

## Result: J/GB and J/token

Gross device energy (device watts x compute time). J/token = model J/GB x the GB a full model streams per decoded
token (every block matrix plus the output head, two bytes per weight), whether or not the device could hold it.

| Model | GB/token | OP15 NPU J/GB | OP15 NPU J/token | OP12 NPU J/GB | OP12 NPU J/token | Pixel CPU J/GB | Pixel CPU J/token | Desktop GPU J/GB | Desktop GPU J/token | Desktop CPU J/GB | Desktop CPU J/token | Server GPU J/GB | Server GPU J/token | Server CPU J/GB | Server CPU J/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 0.99 | 0.177 | 0.18 | 0.133 | 0.13 | 0.381 | 0.38 | 0.449 | 0.44 | 2.74 | 2.71 | 0.500 | 0.49 | 3.44 † | 3.40 † |
| Qwen3-0.6B * | 1.19 | 0.165 | 0.20 | 0.117 | 0.14 | 0.382 | 0.45 | 0.444 | 0.53 | 2.97 † | 3.54 † | 0.523 | 0.62 | 3.44 † | 4.10 † |
| Llama-3.2-1B | 2.47 | 0.175 | 0.43 | 0.137 | 0.34 | 0.353 | 0.87 | 0.383 | 0.95 | 2.91 | 7.20 | 0.441 | 1.09 | 3.44 † | 8.50 † |
| Qwen3-1.7B * | 3.44 | 0.173 | 0.60 | 0.136 | 0.47 | 0.345 | 1.19 | 0.392 | 1.35 | 2.92 | 10.1 | 0.446 | 1.54 | 3.44 † | 11.8 † |
| Gemma-4-E2B | 4.60 | 0.173 | 0.80 | 0.128 | 0.59 | 0.354 | 1.63 | 0.414 | 1.91 | 2.90 | 13.3 | 0.457 | 2.10 | 3.44 † | 15.8 † |
| Qwen3-4B * | 8.04 | 0.178 | 1.43 | 0.141 | 1.13 | 0.350 | 2.82 | 0.391 | 3.15 | 2.95 | 23.8 | 0.434 | 3.49 | 3.09 | 24.8 |
| Qwen3-8B | 15.14 | 0.154 | 2.34 | 0.150 | 2.28 | 0.343 | 5.19 | 0.367 | 5.55 | 2.98 | 45.1 | 0.426 | 6.45 | 3.34 | 50.5 |
| Gemma-4-12B | 23.81 | 0.157 | 3.73 | 0.151 | 3.58 | 0.340 | 8.11 | 0.373 | 8.88 | 2.98 | 71.0 | 0.426 | 10.1 | 3.01 | 71.6 |
| Qwen3-14B | 27.98 | 0.161 | 4.50 | 0.154 | 4.30 | 0.349 | 9.77 | 0.367 | 10.3 | 2.96 | 82.7 | 0.424 | 11.9 | 3.48 | 97.4 |
| Qwen3-32B | 63.97 | 0.161 | 10.3 | 0.155 | 9.90 | 0.354 | 22.6 | 0.369 | 23.6 | 2.97 | 190 | 0.424 | 27.1 | 3.46 | 221 |
| **Device fit, all runs** | | **0.161** |  | **0.152** |  | **0.350** |  | **0.370** |  | **2.97** |  | **0.425** |  | **3.44** |  |

`*` = extra model downloaded for this run (newer, similar size). `†` = every run of that model sat in the CPU
cache, so the device's DRAM-bound fit is used. `-` = not measured.
OnePlus 11 has no root, so it has bandwidth only (below).

Device fits over all unflagged runs (J per call against GB per call):

| Device | Runs used | J/GB (through origin) | J/GB + fixed J/call (with intercept) | J/GB above idle | Flagged runs |
|---|---:|---:|---|---:|---|
| OnePlus 15 NPU | 29/30 | 0.161 | 0.160 + 2.3 mJ (R2 0.9992) | 0.147 | qwen3-8b-L4 |
| OnePlus 12 NPU | 29/29 | 0.152 | 0.154 + -5.4 mJ (R2 0.9993) | 0.132 | none |
| OnePlus 11 NPU | 0/25 | no power data | - | no power data | gemma4-12b-L4 |
| Pixel 10 Pro CPU | 30/30 | 0.350 | 0.351 + -1.6 mJ (R2 0.9997) | 0.328 | none |
| Desktop GPU | 30/30 | 0.370 | 0.367 + 9.0 mJ (R2 0.9998) | 0.338 | none |
| Desktop CPU | 22/30 | 2.968 | 2.981 + -39.8 mJ (R2 1.0000) | 2.903 | none |
| Server GPU | 30/30 | 0.425 | 0.422 + 8.6 mJ (R2 1.0000) | 0.390 | none |
| Server CPU | 9/30 | 3.439 | 3.568 + -468.8 mJ (R2 0.9990) | 2.463 | gemma4-12b-L8 |

"Above idle" subtracts the device's idle power measured before each run.

Bandwidth (weight bytes / compute time over the model's unflagged, DRAM-bound runs) and the time a full model
would take per token at that rate:

| Model | OP15 NPU GB/s | s/token | OP12 NPU GB/s | s/token | OP11 NPU GB/s | s/token | Pixel CPU GB/s | s/token | Desktop GPU GB/s | s/token | Desktop CPU GB/s | s/token | Server GPU GB/s | s/token | Server CPU GB/s | s/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 62.2 | 0.02 | 32.0 | 0.03 | 43.8 | 0.02 | 26.8 | 0.04 | 260.9 | 0.00 | 34.3 | 0.03 | 589.9 | 0.00 | 81.8 † | 0.01 † |
| Qwen3-0.6B * | 59.9 | 0.02 | 29.3 | 0.04 | 42.2 | 0.03 | 26.5 | 0.05 | 256.1 | 0.00 | 30.4 † | 0.04 † | 561.6 | 0.00 | 81.8 † | 0.01 † |
| Llama-3.2-1B | 71.0 | 0.03 | 41.3 | 0.06 | 48.6 | 0.05 | 28.6 | 0.09 | 270.1 | 0.01 | 31.5 | 0.08 | 676.0 | 0.00 | 81.8 † | 0.03 † |
| Qwen3-1.7B * | 70.1 | 0.05 | 39.8 | 0.09 | 47.7 | 0.07 | 28.7 | 0.12 | 268.9 | 0.01 | 32.0 | 0.11 | 666.0 | 0.01 | 81.8 † | 0.04 † |
| Gemma-4-E2B | 68.4 | 0.07 | 38.3 | 0.12 | 46.3 | 0.10 | 28.0 | 0.16 | 267.5 | 0.02 | 32.5 | 0.14 | 651.1 | 0.01 | 81.8 † | 0.06 † |
| Qwen3-4B * | 71.8 | 0.11 | 42.9 | 0.19 | 48.5 | 0.17 | 28.6 | 0.28 | 271.2 | 0.03 | 31.3 | 0.26 | 686.5 | 0.01 | 91.6 | 0.09 |
| Qwen3-8B | 55.4 | 0.27 | 43.9 | 0.34 | 48.6 | 0.31 | 29.3 | 0.52 | 272.6 | 0.06 | 30.8 | 0.49 | 699.4 | 0.02 | 84.7 | 0.18 |
| Gemma-4-12B | 55.8 | 0.43 | 44.2 | 0.54 | 49.1 | 0.48 | 29.2 | 0.81 | 272.7 | 0.09 | 30.2 | 0.79 | 700.8 | 0.03 | 91.4 | 0.26 |
| Qwen3-14B | 55.6 | 0.50 | 44.6 | 0.63 | 49.5 | 0.57 | 29.2 | 0.96 | 273.2 | 0.10 | 30.5 | 0.92 | 704.1 | 0.04 | 83.0 | 0.34 |
| Qwen3-32B | 56.0 | 1.14 | 45.4 | 1.41 | 48.0 | 1.33 | 29.3 | 2.19 | 273.5 | 0.23 | 30.4 | 2.10 | 705.9 | 0.09 | 82.5 | 0.78 |

## Pixel 10 Pro production path (packed Q4_K), for reference

The paper's Pixel does not stream f16: its qualified worker streams the original Q4_K/Q6_K bytes with int8
activations. Same protocol, Qwen3-14B layers 16-23 of the production shard `QWEN_PACKED_1_23.ffn.gguf`:

| Layers | Physical MB | f16-equivalent MB | Compute ms | ms per layer | Physical GB/s | Device W | J/call | J per physical GB | J per f16-equivalent GB | Big-core C | Throttled |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2 | 324 | 1070 | 11.86 | 5.93 | 27.3 | 12.65 | 0.1500 | 0.463 | 0.140 | 30-95 | no |
| 4 | 648 | 2139 | 24.13 | 6.03 | 26.8 | 12.44 | 0.3002 | 0.464 | 0.140 | 34-95 | no |
| 8 | 1272 | 4278 | 47.49 | 5.94 | 26.8 | 12.39 | 0.5883 | 0.462 | 0.138 | 34-93 | YES +6 % |

Per layer this path is about 3.3x faster and 2.5x cheaper than the Pixel's f16 path (0.139 against
0.349 J per f16-equivalent GB), at 12.5 W and with the big cores at 93-96 C.

## What ran

| Device | Engine | Software | Setting |
|---|---|---|---|
| OnePlus 15 (SM8850) | NPU, Hexagon v81 | production FFN worker (09-04 build), backend HTP0, 4 HVX threads | device-local client over loopback, no USB traffic during a run |
| OnePlus 12 (SM8650) | NPU, Hexagon v75 | same worker source, 10-09 build + v75 skeleton built 11-10 | same |
| OnePlus 11 (SM8550) | NPU, Hexagon v73 | same worker source, 09-21 build | same; no root, so no power data |
| Pixel 10 Pro (Tensor G5) | CPU, 6 threads on cores 2-7 | deployed Pixel CPU worker (09-25 build), ggml f16 kernel, production DVFS settings (uclamp 1024, poll 100) | same |
| Desktop (4060 Ti rig) | GPU RTX 4060 Ti, 16 GB | `ffn-stream-bench --device CUDA0` on the deployed `libggml-cuda.so` | weights resident in GPU memory, one host thread feeds the GPU |
| A6000 server FCHLLX01 | GPU RTX A6000, 48 GB (one of two, index 1) | `ffn-stream-bench --device CUDA0` on `build-cuda-s43` | weights resident in GPU memory, one host thread feeds the GPU |
| Desktop (4060 Ti rig) | CPU i9-12900K | `ffn-stream-bench` on the deployed `libggml-cpu.so` | 8 threads (the paper's server setting) |
| A6000 server FCHLLX01 | CPU Threadripper PRO 5995WX | `ffn-stream-bench` on `build-cpu` | 64 threads (fastest of 16-128) |

Not measured: the Ryzen 7 9700X desktop (not available). CPU package energy = RAPL package counter, sampled at
10 Hz, over the same measured window (readable after `sudo chmod a+r` on 11-11; resets at reboot). GPU board power = NVML instantaneous board power (`nvidia-smi`, 10 Hz), time-weighted over the measured window. A GPU row counts the board only. Desktop GPU: the NVML 1-s average over the same window agrees within 0.1 %; the host thread that feeds the GPU holds the CPU package at 27-28 W (2 W idle), and with that package power added the fit is 0.471 J/GB. Server GPU: the NVML 1-s average over the same window agrees within 0.3 %; the host thread that feeds the GPU holds the CPU package at 112-116 W (66-78 W idle), and with that package power added the fit is 0.587 J/GB.

Protocol per run: sampler starts, 5 s idle, load the N-layer shard, idle window with the weights resident
(12 s phone, 10 s host), warm-up (2-3 s), measured window (20 s) of back-to-back calls. Phone runs wait for the
SoC zone and the skin sensor to fall back under a start gate (OnePlus 15: NPU 43 C, back shell 36.8 C, 37.2 C
for the last four models; OnePlus 12: 40 / 33 C; Pixel: big cores 36-40 C, skin 31 C then 34 C); host runs wait
for the package temperature to return to its idle level; GPU runs (added 11-12) wait for 44 C (4060 Ti) or 45 C (A6000).

Models (first 8 layers as one f16 shard each; BF16 parents converted to F16):

| Model | Arch | Blocks | GB/token | FFN | Attention | Head | FFN share |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | qwen2 | 24 | 0.99 | 0.63 | 0.09 | 0.27 | 64 % |
| Qwen3-0.6B * | qwen3 | 28 | 1.19 | 0.53 | 0.35 | 0.31 | 44 % |
| Llama-3.2-1B | llama | 16 | 2.47 | 1.61 | 0.34 | 0.53 | 65 % |
| Qwen3-1.7B * | qwen3 | 28 | 3.44 | 2.11 | 0.70 | 0.62 | 61 % |
| Gemma-4-E2B | gemma4 | 35 | 4.60 | 3.11 | 0.59 | 0.81 | 68 % |
| Qwen3-4B * | qwen3 | 36 | 8.04 | 5.38 | 1.89 | 0.78 | 67 % |
| Qwen3-8B | qwen3 | 36 | 15.14 | 10.87 | 3.02 | 1.24 | 72 % |
| Gemma-4-12B | gemma4 | 48 | 23.81 | 16.99 | 4.81 | 2.01 | 71 % |
| Qwen3-14B | qwen3 | 40 | 27.98 | 21.39 | 5.03 | 1.56 | 76 % |
| Qwen3-32B | qwen3 | 64 | 63.97 | 50.33 | 12.08 | 1.56 | 79 % |

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
| OnePlus 15 | 5139 J | 0.420 | 1.012 | 0.415 |
| OnePlus 12 | 1398 J | 0.640 | 1.438 | 0.445 |
| Pixel 10 Pro | 3386 J | 1.002 | 1.000 | 1.002 |

For the OnePlus phones an ideal series two-cell pack would give 0.5; using 0.5 instead lowers OnePlus 15 device
watts by about 13 % and OnePlus 12 by about 7 %. Treat OnePlus watts as +-10 %; the Pixel's
current and counter agree to 0.2 %.

## Per-run results

### OnePlus 15 NPU (Hexagon v81)

| Model | Layers | MB | Compute ms | GB/s | Device W | Idle W | Resident idle W | J/call | SoC C start-end | Skin C start-end | Big-core MHz | Android status | Throttled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| Qwen2.5-0.5B | 2 | 52 | 0.833 | 62.8 | 11.07 | 0.73 | 1.82 | 0.0092 | 40-72 | 36.5-37.0 | 2375 | 1 | no |
| Qwen2.5-0.5B | 4 | 105 | 1.689 | 61.9 | 11.06 | 0.76 | 1.74 | 0.0187 | 41-76 | 36.7-37.2 | 2493 | 1 | no |
| Qwen2.5-0.5B | 8 | 209 | 3.362 | 62.2 | 11.03 | 0.73 | 1.76 | 0.0371 | 41-73 | 36.7-37.2 | 2445 | 1 | no |
| Qwen3-0.6B | 2 | 38 | 0.629 | 60.0 | 9.91 | 0.76 | 1.64 | 0.0062 | 41-71 | 36.8-37.2 | 2424 | 1 | no |
| Qwen3-0.6B | 4 | 75 | 1.260 | 59.9 | 9.87 | 0.74 | 1.92 | 0.0124 | 41-71 | 36.8-37.2 | 2420 | 1 | no |
| Qwen3-0.6B | 8 | 151 | 2.522 | 59.9 | 9.90 | 0.87 | 1.64 | 0.0250 | 41-70 | 36.8-37.2 | 2445 | 1 | no |
| Llama-3.2-1B | 2 | 201 | 2.816 | 71.5 | 12.49 | 0.82 | 1.64 | 0.0352 | 41-82 | 36.7-37.2 | 2423 | 1 | no |
| Llama-3.2-1B | 4 | 403 | 5.702 | 70.6 | 12.41 | 0.73 | 1.60 | 0.0707 | 41-79 | 36.8-37.2 | 2417 | 1 | no |
| Llama-3.2-1B | 8 | 805 | 11.322 | 71.1 | 12.46 | 0.71 | 1.70 | 0.1411 | 41-81 | 36.7-37.3 | 2396 | 1 | no |
| Qwen3-1.7B | 2 | 151 | 2.125 | 71.1 | 12.05 | 0.73 | 1.64 | 0.0256 | 41-80 | 36.7-37.2 | 2445 | 1 | no |
| Qwen3-1.7B | 4 | 302 | 4.317 | 70.0 | 12.07 | 0.79 | 1.86 | 0.0521 | 41-78 | 36.8-37.2 | 2442 | 1 | no |
| Qwen3-1.7B | 8 | 604 | 8.634 | 70.0 | 12.12 | 0.76 | 1.71 | 0.1047 | 41-78 | 36.8-37.4 | 2396 | 1 | no |
| Gemma-4-E2B | 2 | 113 | 1.636 | 69.2 | 11.88 | 1.15 | 1.98 | 0.0194 | 41-79 | 36.8-37.3 | 2417 | 1 | no |
| Gemma-4-E2B | 4 | 226 | 3.334 | 67.9 | 11.82 | 0.84 | 1.76 | 0.0394 | 41-78 | 36.8-37.4 | 2417 | 1 | no |
| Gemma-4-E2B | 8 | 453 | 6.619 | 68.4 | 11.84 | 0.81 | 1.69 | 0.0784 | 41-78 | 36.8-37.2 | 2438 | 1 | no |
| Qwen3-4B | 2 | 299 | 4.163 | 71.8 | 12.70 | 0.74 | 1.78 | 0.0529 | 41-82 | 36.8-37.6 | 2423 | 1 | no |
| Qwen3-4B | 4 | 598 | 8.337 | 71.7 | 12.79 | 0.82 | 1.72 | 0.1067 | 41-82 | 36.8-37.3 | 2396 | 1 | no |
| Qwen3-4B | 8 | 1195 | 16.631 | 71.9 | 12.80 | 0.80 | 1.75 | 0.2129 | 41-81 | 36.8-37.6 | 2399 | 1 | no |
| Qwen3-8B | 2 | 604 | 10.860 | 55.6 | 8.64 | 0.95 | 1.81 | 0.0938 | 42-69 | 37.1-37.7 | 883 | 1 | no |
| Qwen3-8B | 4 | 1208 | 17.698 | 68.3 | 12.91 | 0.81 | 1.88 | 0.2285 | 42-83 | 37.2-37.8 | 2169 | 1 | no (unsteady: sped up mid-run) |
| Qwen3-8B | 8 | 2416 | 43.621 | 55.4 | 8.55 | 0.76 | 1.76 | 0.3731 | 42-70 | 37.4-37.9 | 883 | 1 | no |
| Gemma-4-12B | 2 | 708 | 12.619 | 56.1 | 8.80 | 0.77 | 1.82 | 0.1110 | 42-71 | 37.1-37.7 | 883 | 1 | no |
| Gemma-4-12B | 4 | 1416 | 25.300 | 56.0 | 8.71 | 0.87 | 1.72 | 0.2204 | 41-70 | 37.2-37.7 | 883 | 1 | no |
| Gemma-4-12B | 8 | 2831 | 50.868 | 55.7 | 8.73 | 0.76 | 1.57 | 0.4442 | 41-70 | 37.4-37.9 | 883 | 1 | no |
| Qwen3-14B | 2 | 1070 | 18.908 | 56.6 | 8.88 | 0.76 | 1.72 | 0.1680 | 42-71 | 37.3-37.8 | 883 | 1 | no |
| Qwen3-14B | 4 | 2139 | 38.290 | 55.9 | 8.88 | 0.76 | 1.69 | 0.3399 | 42-70 | 37.3-37.8 | 883 | 1 | no |
| Qwen3-14B | 8 | 4278 | 77.474 | 55.2 | 8.93 | 0.77 | 1.97 | 0.6915 | 41-71 | 37.5-37.9 | 883 | 1 | no |
| Qwen3-32B | 2 | 1573 | 27.693 | 56.8 | 9.04 | 0.85 | 1.80 | 0.2503 | 41-72 | 37.2-37.9 | 883 | 1 | no |
| Qwen3-32B | 4 | 3146 | 55.968 | 56.2 | 9.02 | 0.98 | 1.75 | 0.5049 | 42-71 | 37.3-38.0 | 883 | 1 | no |
| Qwen3-32B | 8 | 6291 | 112.963 | 55.7 | 9.01 | 0.73 | 1.77 | 1.0182 | 41-68 | 37.9-38.4 | 883 | 1 | no |

### OnePlus 12 NPU (Hexagon v75)

| Model | Layers | MB | Compute ms | GB/s | Device W | Idle W | Resident idle W | J/call | SoC C start-end | Skin C start-end | Big-core MHz | Android status | Throttled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| Qwen2.5-0.5B | 2 | 52 | 1.619 | 32.3 | 4.20 | 0.90 | 1.36 | 0.0068 | 33-48 | 29.0-30.4 | 774 | 0 | no |
| Qwen2.5-0.5B | 4 | 105 | 3.250 | 32.2 | 3.96 | 1.91 | 1.46 | 0.0129 | 33-48 | 29.4-30.7 | 789 | 0 | no |
| Qwen2.5-0.5B | 8 | 209 | 6.561 | 31.9 | 4.31 | 0.88 | 1.69 | 0.0282 | 33-48 | 29.7-30.9 | 801 | 0 | no |
| Qwen3-0.6B | 2 | 38 | 1.286 | 29.4 | 3.46 | 1.94 | 1.68 | 0.0044 | 33-46 | 29.8-30.7 | 816 | 0 | no |
| Qwen3-0.6B | 4 | 75 | 2.568 | 29.4 | 3.45 | 0.96 | 1.64 | 0.0088 | 34-45 | 29.9-30.7 | 824 | 0 | no |
| Qwen3-0.6B | 8 | 151 | 5.175 | 29.2 | 3.41 | 0.69 | 1.49 | 0.0177 | 34-44 | 29.9-30.8 | 780 | 0 | no |
| Llama-3.2-1B | 2 | 201 | 4.879 | 41.3 | 5.93 | 2.63 | 1.71 | 0.0289 | 34-58 | 29.9-31.8 | 793 | 0 | no |
| Llama-3.2-1B | 4 | 403 | 9.786 | 41.1 | 5.78 | 0.77 | 1.95 | 0.0566 | 34-57 | 30.8-32.4 | 719 | 0 | no |
| Llama-3.2-1B | 8 | 805 | 19.490 | 41.3 | 5.62 | 1.57 | 1.35 | 0.1095 | 35-57 | 31.0-32.9 | 725 | 0 | no |
| Qwen3-1.7B | 2 | 151 | 3.791 | 39.8 | 5.17 | 0.73 | 1.89 | 0.0196 | 35-56 | 31.1-32.9 | 740 | 0 | no |
| Qwen3-1.7B | 4 | 302 | 7.544 | 40.0 | 5.23 | 0.92 | 1.45 | 0.0395 | 35-56 | 31.1-32.9 | 704 | 0 | no |
| Qwen3-1.7B | 8 | 604 | 15.220 | 39.7 | 5.45 | 0.54 | 1.73 | 0.0829 | 35-54 | 31.2-33.0 | 748 | 0 | no |
| Gemma-4-E2B | 2 | 113 | 2.972 | 38.1 | 4.91 | 1.03 | 1.60 | 0.0146 | 34-55 | 31.2-32.9 | 748 | 0 | no |
| Gemma-4-E2B | 4 | 226 | 5.878 | 38.5 | 4.87 | 1.66 | 1.57 | 0.0286 | 35-54 | 31.3-33.1 | 716 | 0 | no |
| Gemma-4-E2B | 8 | 453 | 11.870 | 38.2 | 4.88 | 1.05 | 1.58 | 0.0580 | 35-53 | 31.5-33.0 | 693 | 0 | no |
| Qwen3-4B | 2 | 299 | 6.987 | 42.8 | 6.07 | 1.62 | 1.97 | 0.0424 | 35-61 | 31.4-33.6 | 801 | 0 | no |
| Qwen3-4B | 4 | 598 | 13.988 | 42.7 | 5.99 | 0.91 | 1.61 | 0.0838 | 36-60 | 31.7-33.7 | 714 | 0 | no |
| Qwen3-4B | 8 | 1195 | 27.820 | 43.0 | 6.05 | 0.79 | 1.33 | 0.1682 | 36-60 | 31.9-34.1 | 721 | 0 | no |
| Qwen3-8B | 2 | 604 | 13.783 | 43.8 | 6.51 | 0.91 | 1.61 | 0.0897 | 36-63 | 32.0-34.2 | 701 | 0 | no |
| Qwen3-8B | 4 | 1208 | 27.432 | 44.0 | 6.49 | 1.17 | 1.44 | 0.1782 | 36-64 | 32.3-34.2 | 691 | 0 | no |
| Qwen3-8B | 8 | 2416 | 55.064 | 43.9 | 6.63 | 0.72 | 1.43 | 0.3651 | 36-65 | 32.9-34.8 | 698 | 0 | no |
| Gemma-4-12B | 2 | 708 | 16.032 | 44.1 | 6.84 | 0.68 | 1.40 | 0.1097 | 37-64 | 32.9-34.8 | 698 | 0 | no |
| Gemma-4-12B | 4 | 1416 | 31.991 | 44.2 | 6.65 | 2.06 | 1.54 | 0.2129 | 36-61 | 33.0-35.0 | 727 | 0 | no |
| Gemma-4-12B | 8 | 2831 | 64.093 | 44.2 | 6.64 | 0.83 | 1.42 | 0.4254 | 36-65 | 33.3-35.1 | 675 | 0 | no |
| Qwen3-14B | 2 | 1070 | 23.957 | 44.6 | 6.79 | 0.78 | 1.77 | 0.1626 | 37-64 | 33.1-35.3 | 714 | 0 | no |
| Qwen3-14B | 4 | 2139 | 47.545 | 45.0 | 6.82 | 0.77 | 1.68 | 0.3244 | 37-66 | 33.2-35.2 | 685 | 0 | no |
| Qwen3-14B | 8 | 4278 | 96.508 | 44.3 | 6.84 | 0.89 | 1.99 | 0.6601 | 37-67 | 34.1-35.9 | 683 | 0 | no |
| Qwen3-32B | 2 | 1573 | 34.625 | 45.4 | 7.06 | 0.58 | 1.43 | 0.2443 | 37-68 | 33.2-35.6 | 695 | 0 | no |
| Qwen3-32B | 4 | 3146 | 69.246 | 45.4 | 7.02 | 0.77 | 2.03 | 0.4864 | 37-67 | 33.7-35.6 | 698 | 0 | no |

### OnePlus 11 NPU (Hexagon v73, no root: no power)

| Model | Layers | MB | Compute ms | GB/s | Device W | Idle W | Resident idle W | J/call | SoC C start-end | Skin C start-end | Big-core MHz | Android status | Throttled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| Qwen2.5-0.5B | 2 | 52 | 1.196 | 43.7 | - | - | - | - | 33-70 | 30.1-29.3 | 3187 | 0 | no |
| Qwen2.5-0.5B | 4 | 105 | 2.388 | 43.8 | - | - | - | - | 34-71 | 30.6-29.6 | 3187 | 0 | no |
| Qwen2.5-0.5B | 8 | 209 | 4.777 | 43.8 | - | - | - | - | 36-72 | 32.3-30.4 | 3187 | 0 | no |
| Qwen3-0.6B | 2 | 38 | 0.894 | 42.2 | - | - | - | - | 36-68 | 32.7-31.2 | 3187 | 0 | no |
| Qwen3-0.6B | 4 | 75 | 1.787 | 42.2 | - | - | - | - | 36-70 | 33.0-31.7 | 3187 | 0 | no |
| Qwen3-0.6B | 8 | 151 | 3.575 | 42.2 | - | - | - | - | 37-69 | 32.9-32.2 | 3187 | 0 | no |
| Llama-3.2-1B | 2 | 201 | 4.112 | 49.0 | - | - | - | - | 37-78 | 34.1-31.6 | 3187 | 0 | no |
| Llama-3.2-1B | 4 | 403 | 8.223 | 49.0 | - | - | - | - | 37-78 | 34.7-32.2 | 3187 | 0 | no |
| Llama-3.2-1B | 8 | 805 | 16.669 | 48.3 | - | - | - | - | 38-79 | 34.2-33.3 | 3187 | 0 | no |
| Qwen3-1.7B | 2 | 151 | 3.110 | 48.6 | - | - | - | - | 38-79 | 35.2-32.7 | 3187 | 0 | no |
| Qwen3-1.7B | 4 | 302 | 6.336 | 47.7 | - | - | - | - | 38-78 | 35.3-33.1 | 3187 | 0 | no |
| Qwen3-1.7B | 8 | 604 | 12.732 | 47.4 | - | - | - | - | 38-77 | 35.5-33.2 | 3187 | 0 | no |
| Gemma-4-E2B | 2 | 113 | 2.392 | 47.3 | - | - | - | - | 39-78 | 35.4-33.2 | 3187 | 0 | no |
| Gemma-4-E2B | 4 | 226 | 4.786 | 47.3 | - | - | - | - | 37-78 | 35.4-33.4 | 3187 | 0 | no |
| Gemma-4-E2B | 8 | 453 | 9.948 | 45.5 | - | - | - | - | 38-78 | 35.5-33.5 | 3187 | 0 | no |
| Qwen3-4B | 2 | 299 | 6.103 | 49.0 | - | - | - | - | 38-79 | 35.4-33.3 | 3187 | 0 | no |
| Qwen3-4B | 4 | 598 | 12.423 | 48.1 | - | - | - | - | 38-78 | 35.5-33.4 | 3187 | 0 | no |
| Qwen3-4B | 8 | 1195 | 24.570 | 48.7 | - | - | - | - | 39-80 | 35.7-33.2 | 3187 | 0 | no |
| Qwen3-8B | 2 | 604 | 12.519 | 48.2 | - | - | - | - | 39-80 | 34.9-33.8 | 3187 | 0 | no |
| Qwen3-8B | 4 | 1208 | 24.761 | 48.8 | - | - | - | - | 39-79 | 35.7-33.7 | 3187 | 0 | no |
| Gemma-4-12B | 2 | 708 | 14.410 | 49.1 | - | - | - | - | 38-80 | 35.2-32.7 | 3095 | 0 | no |
| Gemma-4-12B | 4 | 1416 | 29.103 | 48.6 | - | - | - | - | 38-80 | 35.4-33.2 | 2729 | 0 | no (unsteady: sped up mid-run) |
| Qwen3-14B | 2 | 1070 | 21.724 | 49.2 | - | - | - | - | 38-80 | 34.4-33.3 | 3127 | 0 | no |
| Qwen3-14B | 4 | 2139 | 43.109 | 49.6 | - | - | - | - | 38-81 | 35.9-32.7 | 2537 | 0 | no |
| Qwen3-32B | 2 | 1573 | 32.735 | 48.0 | - | - | - | - | 38-66 | 34.9-33.1 | 864 | 0 | no |

### Pixel 10 Pro CPU (6 threads, f16)

| Model | Layers | MB | Compute ms | GB/s | Device W | Idle W | Resident idle W | J/call | SoC C start-end | Skin C start-end | Big-core MHz | Android status | Throttled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| Qwen2.5-0.5B | 2 | 52 | 1.928 | 27.1 | 10.10 | 0.52 | 0.52 | 0.0195 | 31-80 | 27.9-30.2 | 3168 | 0 | no |
| Qwen2.5-0.5B | 4 | 105 | 3.869 | 27.0 | 10.18 | 0.57 | 0.53 | 0.0394 | 33-83 | 29.5-31.6 | 3168 | 0 | no |
| Qwen2.5-0.5B | 8 | 209 | 7.855 | 26.6 | 10.19 | 0.55 | 0.54 | 0.0800 | 34-84 | 30.6-32.7 | 3168 | 0 | no |
| Qwen3-0.6B | 2 | 38 | 1.390 | 27.2 | 10.08 | 1.33 | 0.55 | 0.0140 | 35-83 | 30.9-33.1 | 3129 | 0 | no |
| Qwen3-0.6B | 4 | 75 | 2.849 | 26.5 | 9.99 | 0.54 | 0.54 | 0.0285 | 34-84 | 30.8-33.0 | 3168 | 0 | no |
| Qwen3-0.6B | 8 | 151 | 5.742 | 26.3 | 10.08 | 0.50 | 0.53 | 0.0579 | 35-84 | 31.7-33.8 | 3092 | 0 | no |
| Llama-3.2-1B | 2 | 201 | 7.045 | 28.6 | 10.40 | 0.70 | 0.75 | 0.0733 | 36-88 | 32.4-34.6 | 3168 | 0 | no |
| Llama-3.2-1B | 4 | 403 | 14.066 | 28.6 | 10.42 | 0.70 | 0.69 | 0.1465 | 37-89 | 33.3-35.4 | 3168 | 0 | no |
| Llama-3.2-1B | 8 | 805 | 28.148 | 28.6 | 10.00 | 0.72 | 0.78 | 0.2814 | 37-85 | 33.4-35.4 | 3168 | 0 | no |
| Qwen3-1.7B | 2 | 151 | 5.280 | 28.6 | 9.99 | 0.80 | 0.72 | 0.0528 | 37-85 | 33.5-35.5 | 3168 | 0 | no |
| Qwen3-1.7B | 4 | 302 | 10.608 | 28.5 | 9.97 | 0.79 | 0.80 | 0.1058 | 37-85 | 33.7-35.8 | 3168 | 0 | no |
| Qwen3-1.7B | 8 | 604 | 20.962 | 28.8 | 9.90 | 0.77 | 0.75 | 0.2076 | 37-85 | 33.7-35.8 | 3168 | 0 | no |
| Gemma-4-E2B | 2 | 113 | 3.993 | 28.4 | 9.92 | 0.80 | 0.74 | 0.0396 | 38-85 | 33.8-35.9 | 3168 | 0 | no |
| Gemma-4-E2B | 4 | 226 | 8.107 | 27.9 | 9.90 | 0.78 | 0.77 | 0.0803 | 37-78 | 33.8-35.9 | 3168 | 0 | no |
| Gemma-4-E2B | 8 | 453 | 16.210 | 27.9 | 9.88 | 0.68 | 0.67 | 0.1602 | 37-85 | 33.7-35.8 | 3168 | 0 | no |
| Qwen3-4B | 2 | 299 | 10.381 | 28.8 | 9.96 | 0.71 | 0.69 | 0.1034 | 37-85 | 33.8-35.9 | 3168 | 0 | no |
| Qwen3-4B | 4 | 598 | 20.749 | 28.8 | 9.89 | 0.67 | 0.65 | 0.2052 | 38-85 | 33.8-35.9 | 3168 | 0 | no |
| Qwen3-4B | 8 | 1195 | 41.952 | 28.5 | 10.05 | 0.67 | 0.62 | 0.4214 | 37-85 | 33.9-36.0 | 3168 | 0 | no |
| Qwen3-8B | 2 | 604 | 20.558 | 29.4 | 9.94 | 0.69 | 0.71 | 0.2043 | 38-86 | 33.8-35.9 | 3168 | 0 | no |
| Qwen3-8B | 4 | 1208 | 41.210 | 29.3 | 10.00 | 0.68 | 0.62 | 0.4122 | 37-85 | 33.8-35.9 | 3168 | 0 | no |
| Qwen3-8B | 8 | 2416 | 82.368 | 29.3 | 10.07 | 0.67 | 0.63 | 0.8295 | 37-86 | 33.9-36.0 | 3129 | 0 | no |
| Gemma-4-12B | 2 | 708 | 24.358 | 29.1 | 10.00 | 0.67 | 0.64 | 0.2436 | 37-85 | 33.8-35.9 | 3168 | 0 | no |
| Gemma-4-12B | 4 | 1416 | 48.720 | 29.1 | 10.03 | 0.64 | 0.63 | 0.4885 | 37-86 | 33.8-35.9 | 3168 | 0 | no |
| Gemma-4-12B | 8 | 2831 | 96.312 | 29.4 | 9.97 | 0.67 | 0.72 | 0.9599 | 37-86 | 33.9-36.0 | 3168 | 0 | no |
| Qwen3-14B | 2 | 1070 | 36.701 | 29.1 | 10.09 | 0.62 | 0.63 | 0.3705 | 37-86 | 33.9-36.0 | 3168 | 0 | no |
| Qwen3-14B | 4 | 2139 | 73.168 | 29.2 | 10.20 | 0.55 | 0.56 | 0.7464 | 37-80 | 34.0-36.2 | 3168 | 0 | no |
| Qwen3-14B | 8 | 4278 | 146.215 | 29.3 | 10.23 | 0.62 | 0.65 | 1.4951 | 38-87 | 34.2-36.3 | 3168 | 0 | no |
| Qwen3-32B | 2 | 1573 | 53.689 | 29.3 | 10.34 | 0.55 | 0.55 | 0.5552 | 37-88 | 33.8-36.1 | 3168 | 0 | no |
| Qwen3-32B | 4 | 3146 | 107.514 | 29.3 | 10.35 | 0.63 | 0.59 | 1.1131 | 37-88 | 34.1-36.2 | 3168 | 0 | no |
| Qwen3-32B | 8 | 6291 | 215.063 | 29.3 | 10.35 | 0.71 | 0.62 | 2.2265 | 38-89 | 34.8-36.7 | 3168 | 0 | no |

### Desktop GPU (RTX 4060 Ti 16 GB, CUDA)

| Model | Layers | MB | Compute ms | GB/s | GPU board W | Idle W | Resident idle W | J/call | GPU C start-end | Clock MHz | Host CPU package W | At power limit | Throttled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| Qwen2.5-0.5B | 2 | 52 | 0.205 | 255 | 113.8 | 8.5 | 26.9 | 0.0234 | 52-56 | 2820 | 27.8 | no | no |
| Qwen2.5-0.5B | 4 | 105 | 0.402 | 260 | 116.2 | 8.5 | 27.1 | 0.0467 | 55-58 | 2820 | 27.4 | no | no |
| Qwen2.5-0.5B | 8 | 209 | 0.796 | 263 | 118.1 | 8.8 | 27.2 | 0.0940 | 56-59 | 2820 | 27.7 | no | no |
| Qwen3-0.6B | 2 | 38 | 0.152 | 248 | 113.1 | 8.6 | 27.2 | 0.0172 | 56-59 | 2820 | 27.6 | no | no |
| Qwen3-0.6B | 4 | 75 | 0.296 | 255 | 115.5 | 8.8 | 27.1 | 0.0342 | 56-59 | 2820 | 27.7 | no | no |
| Qwen3-0.6B | 8 | 151 | 0.583 | 259 | 114.2 | 8.9 | 27.2 | 0.0666 | 55-59 | 2820 | 27.9 | no | no |
| Llama-3.2-1B | 2 | 201 | 0.751 | 268 | 102.8 | 8.7 | 27.1 | 0.0772 | 53-57 | 2820 | 27.4 | no | no |
| Llama-3.2-1B | 4 | 403 | 1.492 | 270 | 103.1 | 8.6 | 27.2 | 0.1538 | 54-57 | 2820 | 27.5 | no | no |
| Llama-3.2-1B | 8 | 805 | 2.974 | 271 | 103.7 | 8.6 | 27.2 | 0.3083 | 54-57 | 2820 | 27.5 | no | no |
| Qwen3-1.7B | 2 | 151 | 0.567 | 266 | 104.8 | 8.6 | 27.2 | 0.0594 | 54-57 | 2820 | 27.8 | no | no |
| Qwen3-1.7B | 4 | 302 | 1.124 | 269 | 105.2 | 8.6 | 27.2 | 0.1183 | 54-57 | 2820 | 27.8 | no | no |
| Qwen3-1.7B | 8 | 604 | 2.239 | 270 | 105.7 | 8.3 | 27.1 | 0.2367 | 54-57 | 2820 | 27.8 | no | no |
| Gemma-4-E2B | 2 | 113 | 0.429 | 264 | 109.8 | 8.8 | 27.1 | 0.0471 | 55-58 | 2820 | 27.8 | no | no |
| Gemma-4-E2B | 4 | 226 | 0.848 | 267 | 110.9 | 8.7 | 27.1 | 0.0941 | 55-58 | 2820 | 27.4 | no | no |
| Gemma-4-E2B | 8 | 453 | 1.687 | 269 | 111.2 | 8.7 | 27.1 | 0.1876 | 55-58 | 2820 | 27.4 | no | no |
| Qwen3-4B | 2 | 299 | 1.108 | 270 | 105.4 | 8.7 | 27.2 | 0.1168 | 54-57 | 2820 | 27.9 | no | no |
| Qwen3-4B | 4 | 598 | 2.205 | 271 | 105.9 | 8.8 | 27.2 | 0.2336 | 54-57 | 2820 | 27.9 | no | no |
| Qwen3-4B | 8 | 1195 | 4.400 | 272 | 106.3 | 8.7 | 27.2 | 0.4675 | 54-57 | 2820 | 27.5 | no | no |
| Qwen3-8B | 2 | 604 | 2.222 | 272 | 100.5 | 8.7 | 27.0 | 0.2233 | 53-58 | 2820 | 27.5 | no | no |
| Qwen3-8B | 4 | 1208 | 4.433 | 273 | 100.5 | 9.0 | 27.0 | 0.4455 | 53-58 | 2820 | 27.9 | no | no |
| Qwen3-8B | 8 | 2416 | 8.855 | 273 | 99.8 | 8.8 | 27.0 | 0.8838 | 53-58 | 2820 | 27.9 | no | no |
| Gemma-4-12B | 2 | 708 | 2.602 | 272 | 103.4 | 8.6 | 27.0 | 0.2691 | 53-57 | 2820 | 27.9 | no | no |
| Gemma-4-12B | 4 | 1416 | 5.192 | 273 | 103.6 | 8.8 | 27.0 | 0.5378 | 53-57 | 2820 | 28.0 | no | no |
| Gemma-4-12B | 8 | 2831 | 10.373 | 273 | 101.2 | 8.8 | 27.1 | 1.0497 | 53-58 | 2820 | 27.5 | no | no |
| Qwen3-14B | 2 | 1070 | 3.922 | 273 | 101.5 | 8.5 | 27.1 | 0.3981 | 53-58 | 2820 | 27.5 | no | no |
| Qwen3-14B | 4 | 2139 | 7.832 | 273 | 100.5 | 8.5 | 27.0 | 0.7870 | 53-58 | 2820 | 27.5 | no | no |
| Qwen3-14B | 8 | 4278 | 15.652 | 273 | 100.2 | 8.6 | 27.0 | 1.5687 | 53-58 | 2820 | 27.9 | no | no |
| Qwen3-32B | 2 | 1573 | 5.758 | 273 | 101.3 | 8.4 | 27.0 | 0.5833 | 53-58 | 2820 | 28.0 | no | no |
| Qwen3-32B | 4 | 3146 | 11.505 | 273 | 100.7 | 8.7 | 27.0 | 1.1589 | 53-58 | 2820 | 27.5 | no | no |
| Qwen3-32B | 8 | 6291 | 22.998 | 274 | 101.0 | 8.5 | 27.1 | 2.3240 | 53-58 | 2820 | 27.5 | no | no |

### Server GPU (RTX A6000 48 GB, CUDA)

| Model | Layers | MB | Compute ms | GB/s | GPU board W | Idle W | Resident idle W | J/call | GPU C start-end | Clock MHz | Host CPU package W | At power limit | Throttled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| Qwen2.5-0.5B | 2 | 52 | 0.095 | 549 | 298.8 | 25.7 | 77.2 | 0.0284 | 58-71 | 1880 | 113.9 | yes | no |
| Qwen2.5-0.5B | 4 | 105 | 0.179 | 585 | 298.9 | 23.6 | 77.3 | 0.0535 | 58-71 | 1854 | 114.7 | yes | no |
| Qwen2.5-0.5B | 8 | 209 | 0.346 | 604 | 298.3 | 21.4 | 77.5 | 0.1034 | 59-72 | 1841 | 113.1 | yes | no |
| Qwen3-0.6B | 2 | 38 | 0.074 | 510 | 291.2 | 26.3 | 77.5 | 0.0216 | 58-71 | 1890 | 113.3 | no | no |
| Qwen3-0.6B | 4 | 75 | 0.136 | 554 | 299.3 | 21.4 | 77.5 | 0.0408 | 58-72 | 1869 | 113.1 | yes | no |
| Qwen3-0.6B | 8 | 151 | 0.260 | 580 | 299.1 | 23.9 | 77.5 | 0.0779 | 59-72 | 1873 | 113.7 | yes | no |
| Llama-3.2-1B | 2 | 201 | 0.305 | 660 | 297.1 | 24.8 | 77.6 | 0.0906 | 58-72 | 1888 | 114.0 | yes | no |
| Llama-3.2-1B | 4 | 403 | 0.598 | 674 | 298.6 | 21.2 | 77.4 | 0.1784 | 58-71 | 1881 | 115.2 | yes | no |
| Llama-3.2-1B | 8 | 805 | 1.182 | 681 | 299.6 | 25.6 | 77.6 | 0.3542 | 59-72 | 1873 | 113.9 | yes | no |
| Qwen3-1.7B | 2 | 151 | 0.234 | 646 | 298.0 | 25.5 | 77.6 | 0.0697 | 58-72 | 1879 | 113.2 | yes | no |
| Qwen3-1.7B | 4 | 302 | 0.455 | 663 | 299.9 | 21.4 | 77.6 | 0.1366 | 58-72 | 1871 | 113.1 | yes | no |
| Qwen3-1.7B | 8 | 604 | 0.898 | 673 | 298.5 | 24.9 | 77.5 | 0.2680 | 58-71 | 1865 | 112.4 | yes | no |
| Gemma-4-E2B | 2 | 113 | 0.181 | 626 | 299.5 | 23.8 | 77.4 | 0.0542 | 58-71 | 1868 | 113.1 | yes | no |
| Gemma-4-E2B | 4 | 226 | 0.350 | 647 | 299.3 | 21.4 | 77.5 | 0.1047 | 58-71 | 1856 | 115.1 | yes | no |
| Gemma-4-E2B | 8 | 453 | 0.687 | 659 | 299.5 | 25.6 | 77.4 | 0.2057 | 59-71 | 1844 | 112.9 | yes | no |
| Qwen3-4B | 2 | 299 | 0.443 | 675 | 299.3 | 25.5 | 77.5 | 0.1326 | 58-71 | 1865 | 112.7 | yes | no |
| Qwen3-4B | 4 | 598 | 0.873 | 685 | 299.4 | 22.1 | 77.4 | 0.2613 | 59-71 | 1862 | 112.0 | yes | no |
| Qwen3-4B | 8 | 1195 | 1.732 | 690 | 298.8 | 21.2 | 77.4 | 0.5175 | 59-72 | 1859 | 113.1 | yes | no |
| Qwen3-8B | 2 | 604 | 0.871 | 693 | 298.3 | 22.2 | 77.5 | 0.2598 | 58-71 | 1885 | 112.4 | yes | no |
| Qwen3-8B | 4 | 1208 | 1.729 | 698 | 298.4 | 21.7 | 77.5 | 0.5161 | 59-71 | 1882 | 113.2 | yes | no |
| Qwen3-8B | 8 | 2416 | 3.444 | 701 | 298.5 | 25.5 | 77.7 | 1.0280 | 59-72 | 1875 | 113.0 | yes | no |
| Gemma-4-12B | 2 | 708 | 1.018 | 695 | 298.4 | 22.0 | 77.5 | 0.3038 | 58-71 | 1875 | 113.2 | yes | no |
| Gemma-4-12B | 4 | 1416 | 2.022 | 700 | 299.2 | 21.2 | 77.5 | 0.6051 | 58-71 | 1872 | 113.4 | yes | no |
| Gemma-4-12B | 8 | 2831 | 4.029 | 703 | 298.6 | 22.9 | 77.5 | 1.2029 | 59-72 | 1882 | 113.0 | yes | no |
| Qwen3-14B | 2 | 1070 | 1.527 | 700 | 299.2 | 25.0 | 77.6 | 0.4569 | 59-72 | 1873 | 111.6 | yes | no |
| Qwen3-14B | 4 | 2139 | 3.041 | 704 | 299.2 | 26.6 | 77.6 | 0.9096 | 59-72 | 1875 | 112.0 | yes | no |
| Qwen3-14B | 8 | 4278 | 6.065 | 705 | 299.2 | 23.1 | 77.8 | 1.8145 | 58-71 | 1875 | 115.5 | yes | no |
| Qwen3-32B | 2 | 1573 | 2.236 | 703 | 299.1 | 23.3 | 77.7 | 0.6687 | 59-72 | 1873 | 112.5 | yes | no |
| Qwen3-32B | 4 | 3146 | 4.458 | 706 | 299.4 | 25.2 | 77.8 | 1.3349 | 59-72 | 1870 | 112.9 | yes | no |
| Qwen3-32B | 8 | 6291 | 8.904 | 707 | 299.4 | 25.1 | 78.1 | 2.6655 | 61-73 | 1864 | 114.0 | yes | no |

### Desktop CPU (i9-12900K, 8 threads)

| Model | Layers | MB | Compute ms | GB/s | Package W | Idle W | Resident idle W | J/call | CPU C start-end | Slowed in run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Qwen2.5-0.5B | 2 | 52 | 0.792 | 66.0 | 107.6 | 2.0 | 2.0 | 0.0853 | 46-49 | no |
| Qwen2.5-0.5B | 4 | 105 | 2.557 | 40.9 | 96.9 | 2.1 | 1.9 | 0.2476 | 45-47 | no |
| Qwen2.5-0.5B | 8 | 209 | 6.096 | 34.3 | 94.0 | 2.3 | 1.9 | 0.5732 | 45-45 | no |
| Qwen3-0.6B | 2 | 38 | 0.341 | 110.6 | 128.5 | 1.9 | 1.9 | 0.0439 | 53-55 | no |
| Qwen3-0.6B | 4 | 75 | 1.574 | 48.0 | 101.4 | 2.0 | 1.9 | 0.1596 | 48-49 | no |
| Qwen3-0.6B | 8 | 151 | 4.139 | 36.5 | 95.8 | 1.9 | 1.9 | 0.3964 | 46-47 | no |
| Llama-3.2-1B | 2 | 201 | 5.755 | 35.0 | 92.6 | 2.0 | 1.9 | 0.5328 | 46-46 | no |
| Llama-3.2-1B | 4 | 403 | 12.498 | 32.2 | 91.7 | 2.0 | 1.9 | 1.1462 | 45-46 | no |
| Llama-3.2-1B | 8 | 805 | 25.851 | 31.2 | 91.3 | 2.0 | 1.9 | 2.3599 | 44-46 | no |
| Qwen3-1.7B | 2 | 151 | 4.085 | 37.0 | 94.5 | 2.0 | 1.9 | 0.3859 | 45-47 | no |
| Qwen3-1.7B | 4 | 302 | 9.120 | 33.1 | 93.4 | 1.9 | 2.0 | 0.8518 | 45-47 | no |
| Qwen3-1.7B | 8 | 604 | 19.162 | 31.5 | 93.0 | 2.0 | 1.9 | 1.7822 | 45-48 | no |
| Gemma-4-E2B | 2 | 113 | 2.854 | 39.7 | 95.2 | 2.1 | 2.0 | 0.2716 | 45-47 | no |
| Gemma-4-E2B | 4 | 226 | 6.667 | 34.0 | 93.3 | 2.1 | 2.0 | 0.6223 | 45-48 | no |
| Gemma-4-E2B | 8 | 453 | 14.234 | 31.8 | 93.5 | 2.3 | 2.0 | 1.3305 | 46-48 | no |
| Qwen3-4B | 2 | 299 | 9.029 | 33.1 | 92.1 | 2.0 | 2.0 | 0.8317 | 46-48 | no |
| Qwen3-4B | 4 | 598 | 18.941 | 31.6 | 92.2 | 2.1 | 2.0 | 1.7455 | 45-46 | no |
| Qwen3-4B | 8 | 1195 | 38.770 | 30.8 | 91.6 | 1.9 | 1.9 | 3.5523 | 45-46 | no |
| Qwen3-8B | 2 | 604 | 19.122 | 31.6 | 91.5 | 2.1 | 1.9 | 1.7495 | 44-46 | no |
| Qwen3-8B | 4 | 1208 | 39.099 | 30.9 | 91.4 | 2.0 | 1.9 | 3.5755 | 45-48 | no |
| Qwen3-8B | 8 | 2416 | 79.125 | 30.5 | 91.2 | 1.9 | 1.9 | 7.2199 | 44-46 | no |
| Gemma-4-12B | 2 | 708 | 22.809 | 31.0 | 90.7 | 2.0 | 1.9 | 2.0699 | 45-45 | no |
| Gemma-4-12B | 4 | 1416 | 46.616 | 30.4 | 90.4 | 1.9 | 1.9 | 4.2159 | 44-45 | no |
| Gemma-4-12B | 8 | 2831 | 94.426 | 30.0 | 89.5 | 2.0 | 1.9 | 8.4486 | 44-45 | no |
| Qwen3-14B | 2 | 1070 | 34.508 | 31.0 | 90.4 | 2.0 | 1.9 | 3.1192 | 45-46 | no |
| Qwen3-14B | 4 | 2139 | 69.946 | 30.6 | 90.0 | 1.9 | 1.9 | 6.2930 | 45-45 | no |
| Qwen3-14B | 8 | 4278 | 140.719 | 30.4 | 90.1 | 1.9 | 1.9 | 12.6730 | 44-46 | no |
| Qwen3-32B | 2 | 1573 | 51.153 | 30.7 | 90.5 | 2.1 | 1.9 | 4.6288 | 45-48 | no |
| Qwen3-32B | 4 | 3146 | 103.258 | 30.5 | 90.3 | 1.9 | 1.9 | 9.3263 | 45-46 | no |
| Qwen3-32B | 8 | 6291 | 207.445 | 30.3 | 90.2 | 2.0 | 2.0 | 18.7177 | 45-46 | no |

### Server CPU (Threadripper PRO 5995WX, 64 threads)

| Model | Layers | MB | Compute ms | GB/s | Package W | Idle W | Resident idle W | J/call | CPU C start-end | Slowed in run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Qwen2.5-0.5B | 2 | 52 | 0.169 | 309.2 | 177.4 | 66.4 | 70.3 | 0.0300 | 62-66 | no |
| Qwen2.5-0.5B | 4 | 105 | 0.295 | 354.2 | 176.5 | 71.1 | 69.1 | 0.0521 | 63-68 | no |
| Qwen2.5-0.5B | 8 | 209 | 0.393 | 532.0 | 200.1 | 72.2 | 78.5 | 0.0787 | 66-70 | no |
| Qwen3-0.6B | 2 | 38 | 0.119 | 316.7 | 183.1 | 69.5 | 70.4 | 0.0218 | 67-71 | no |
| Qwen3-0.6B | 4 | 75 | 0.250 | 302.1 | 179.2 | 72.6 | 73.0 | 0.0448 | 68-73 | no |
| Qwen3-0.6B | 8 | 151 | 0.329 | 459.1 | 180.6 | 77.4 | 69.4 | 0.0594 | 68-72 | no |
| Llama-3.2-1B | 2 | 201 | 0.226 | 890.4 | 214.1 | 72.9 | 76.9 | 0.0484 | 70-75 | no |
| Llama-3.2-1B | 4 | 403 | 2.354 | 171.1 | 286.3 | 72.7 | 76.1 | 0.6739 | 78-84 | no |
| Llama-3.2-1B | 8 | 805 | 7.874 | 102.3 | 282.7 | 72.5 | 71.2 | 2.2262 | 77-84 | no |
| Qwen3-1.7B | 2 | 151 | 0.183 | 823.9 | 193.4 | 76.8 | 72.9 | 0.0354 | 72-76 | no |
| Qwen3-1.7B | 4 | 302 | 0.942 | 320.5 | 286.3 | 76.6 | 74.3 | 0.2698 | 80-85 | no |
| Qwen3-1.7B | 8 | 604 | 5.217 | 115.8 | 285.8 | 75.0 | 74.0 | 1.4910 | 78-85 | no |
| Gemma-4-E2B | 2 | 113 | 0.246 | 460.5 | 178.5 | 71.5 | 78.0 | 0.0439 | 72-76 | no |
| Gemma-4-E2B | 4 | 226 | 0.336 | 673.5 | 219.7 | 72.8 | 75.1 | 0.0739 | 75-80 | no |
| Gemma-4-E2B | 8 | 453 | 3.215 | 140.9 | 286.3 | 78.8 | 75.5 | 0.9204 | 80-86 | no |
| Qwen3-4B | 2 | 299 | 0.853 | 350.3 | 286.3 | 74.3 | 76.5 | 0.2442 | 78-84 | no |
| Qwen3-4B | 4 | 598 | 4.987 | 119.8 | 285.9 | 75.2 | 75.1 | 1.4258 | 78-85 | no |
| Qwen3-4B | 8 | 1195 | 13.048 | 91.6 | 282.8 | 76.1 | 73.9 | 3.6899 | 77-84 | no |
| Qwen3-8B | 2 | 604 | 4.920 | 122.8 | 285.4 | 73.9 | 77.1 | 1.4043 | 77-84 | no |
| Qwen3-8B | 4 | 1208 | 12.837 | 94.1 | 284.5 | 73.7 | 73.2 | 3.6520 | 78-85 | no |
| Qwen3-8B | 8 | 2416 | 28.540 | 84.7 | 282.7 | 75.6 | 74.3 | 8.0672 | 78-85 | no |
| Gemma-4-12B | 2 | 708 | 6.311 | 112.2 | 285.1 | 72.8 | 72.3 | 1.7991 | 77-88 | no |
| Gemma-4-12B | 4 | 1416 | 15.495 | 91.4 | 274.6 | 82.9 | 72.8 | 4.2553 | 74-82 | no |
| Gemma-4-12B | 8 | 2831 | 34.155 | 82.9 | 279.3 | 71.0 | 72.2 | 9.5395 | 76-84 | YES +5 % |
| Qwen3-14B | 2 | 1070 | 12.336 | 86.7 | 285.3 | 72.9 | 71.3 | 3.5198 | 77-84 | no |
| Qwen3-14B | 4 | 2139 | 25.350 | 84.4 | 286.1 | 94.0 | 94.1 | 7.2521 | 79-85 | no |
| Qwen3-14B | 8 | 4278 | 52.526 | 81.4 | 286.3 | 95.7 | 99.3 | 15.0365 | 80-86 | no |
| Qwen3-32B | 2 | 1573 | 18.124 | 86.8 | 285.7 | 82.4 | 72.3 | 5.1779 | 78-85 | no |
| Qwen3-32B | 4 | 3146 | 38.379 | 82.0 | 284.9 | 74.9 | 80.6 | 10.9338 | 78-85 | no |
| Qwen3-32B | 8 | 6291 | 76.946 | 81.8 | 283.2 | 74.4 | 73.1 | 21.7879 | 78-85 | no |

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
- Desktop GPU: 272-274 GB/s at 100-106 W on working sets of 1 GB and more, 248-272 GB/s at 100-118 W on smaller ones (per-kernel launch cost). Board idle 8-9 W with no CUDA context, 27 W with the weights resident. GPU temperature 52-56 C at the start of the measured window, at most 59 C; NVML reported no thermal slowdown. The card stops its fans when idle and settles at 42-43 C, so the start gate is 44 C.
- Server GPU: 690-707 GB/s at 298-299 W on working sets of 1 GB and more, 510-695 GB/s at 291-300 W on smaller ones (per-kernel launch cost). Board idle 21-27 W with no CUDA context, 77-78 W with the weights resident. GPU temperature 58-61 C at the start of the measured window, at most 73 C; NVML reported no thermal slowdown. 29 of 30 runs sat at the board's power limit (297-300 W; NVML software power cap, graphics clock 1841-1888 MHz): this is the card's normal operating point under this load, so those runs stay in the fits. Start gate 45 C (the card idles at 38-40 C and needs about four minutes to return there).
- Desktop CPU: package 89-92 W on DRAM-bound sets (up to 129 W on small cached sets) against 2 W idle; no run
  slowed; package temperature at most 55 C.
- Server CPU: package 275-286 W on DRAM-bound sets against 66-78 W idle (94-96 W before the two Qwen3-14B runs:
  other activity on this shared host); flagged: gemma4-12b-L8 (calls slowed
  more than 5 % within the run). Tctl 66-88 C; the host idles at 57-75 C.
- Server CPU repeat (11-12, after the A6000 GPU leg, `server-cpu-20261112`): fit 3.43 J/GB over 9 runs against 3.44 on 11-11, DRAM-bound 80.0 GB/s against 81.8; the 11 runs of 1 GB and more agree within 2.6 % in J/GB.
  Idle package 66-74 W before 25 runs and 108-117 W before the last 5 (other activity on this shared host); 2 runs started at 78.6-79.4 C, above the 76 C start gate, after the 5-minute wait limit. The tables use the 11-11 leg.

## Server CPU thread sweep (11-13)

Qwen3-14B, 8 layers (4.28 GB working set), one run per setting, same protocol. The tables above use 64 threads, the fastest setting; 32 threads stream as fast for less power, and 16 threads give the lowest energy per GB.

| Threads | GB/s | Package W | Idle W | J/GB | J/GB above idle |
|---:|---:|---:|---:|---:|---:|
| 64 | 80.8 | 267 | 69 | 3.31 | 2.45 |
| 32 | 81.2 | 207 | 72 | 2.54 | 1.66 |
| 16 | 72.0 | 167 | 75 | 2.32 | 1.27 |
| 8 | 44.9 | 144 | 71 | 3.21 | 1.63 |
| 4 | 42.6 | 150 | 73 | 3.53 | 1.81 |

## Files

- Tools: `tools/` (shard maker, device-local client, host bench, samplers, runners, analysis, this assembler).
- Raw runs: `/mnt/data_h16t/zs89458/xband-runs/stream-energy-20261110/` (`op15`, `op12`, `op11`, `pixel`,
  `pixel-q4k`, `desktop-gpu`, `server-gpu`, `desktop-cpu*`, `server-cpu*`, pilots). Shards: `/mnt/data_h16t/zs89458/xband-data/models/stream-energy-shards/`.
- Phone staging: `/data/local/tmp/stream-energy-20261110/` on each phone (scripts, client, run logs; shards
  removed from the OnePlus 12, OnePlus 11 and Pixel, still on the OnePlus 15: 19 GB).
