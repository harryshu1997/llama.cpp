# analyze.py results/20260928-cpu-arm

model /home/myid/zs89458/Documents/models/Qwen3-14B-Q4KM-dequant-f16.gguf: 40 layers, CPU-resident layers 0-23 + output head = 17.41 GB/step; FFN 534.8 MB/layer

| arm | c | step ms | ms/tok | tok/s | CPU-res ms | rtt p50 | rtt p99 | comp p50 | host p50 | GPU W | GPU J/tok | CPU J/tok | ident | step vs cpu | GPU J vs cpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | 1 | 224.4 | 224.4 | 4.46 | 208.3 | - | - | - | - | 102.2 | 21.71 | - | - | +0.0% | +0.0% |
| cpu | 4 | 246.0 | 61.5 | 16.26 | 229.1 | - | - | - | - | 109.2 | 6.34 | - | - | +0.0% | +0.0% |
| cpu-helpers | 1 | 244.9 | 244.9 | 4.08 | 228.7 | - | - | - | - | 109.6 | 25.22 | - | 5/5 | +9.1% | +16.2% |
| cpu-helpers | 4 | 264.4 | 66.1 | 15.13 | 247.5 | - | - | - | - | 111.5 | 6.94 | - | 6/8 | +7.5% | +9.5% |
| gpu | 1 | 42.7 | 42.7 | 23.41 | - | - | - | - | - | 293.7 | 12.04 | - | 4/5 | -81.0% | -44.6% |
| gpu | 4 | 44.7 | 11.2 | 89.45 | - | - | - | - | - | 291.7 | 3.22 | - | 7/8 | -81.8% | -49.2% |

GPU part of a step: 16.1 (gpu-arm); RAPL: permission denied (energy_uj is root-only on FCHLLX01); CPU package energy recorded as null

time model (CPU-resident ms per step): BW_cpu fitted on the cpu arm = 83.6 GB/s (reference 69); phones: op15 60.1 GB/s rtt 1.50 ms (config), pixel 83.6 GB/s rtt 1.50 ms (config)

| arm | c | measured | overlap (fit) | err | aggregate (fit) | err | overlap @69 GB/s |
|---|---|---|---|---|---|---|---|
| cpu | 1 | 208.3 | 208.3 | +0% | 208.3 | +0% | 252.3 |
| cpu | 4 | 229.1 | 229.1 | -0% | 229.1 | -0% | 252.3 |
| cpu-helpers | 1 | 228.7 | 208.3 | -9% | 208.3 | -9% | 252.3 |
| cpu-helpers | 4 | 247.5 | 229.1 | -7% | 229.1 | -7% | 252.3 |
