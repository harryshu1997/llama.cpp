# analyze.py results/20260928-cpu-pair-contended

model /home/myid/zs89458/Documents/models/Qwen3-14B-Q4KM-dequant-f16.gguf: 40 layers, CPU-resident layers 0-23 + output head = 17.41 GB/step; FFN 534.8 MB/layer

| arm | c | step ms | ms/tok | tok/s | CPU-res ms | rtt p50 | rtt p99 | comp p50 | host p50 | GPU W | GPU J/tok | CPU J/tok | ident | step vs cpu | GPU J vs cpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | 1 | 246.9 | 246.9 | 4.05 | 229.3 | - | - | - | - | 112.3 | 26.15 | - | - | +0.0% | +0.0% |
| cpu | 4 | 267.4 | 66.8 | 14.96 | 249.8 | - | - | - | - | 111.6 | 7.07 | - | - | +0.0% | +0.0% |
| cpu-helpers | 1 | 253.5 | 253.5 | 3.95 | 235.8 | - | - | - | - | 107.3 | 25.32 | - | - | +2.7% | -3.1% |
| cpu-helpers | 4 | 264.7 | 66.2 | 15.11 | 247.1 | - | - | - | - | 111.7 | 7.02 | - | - | -1.0% | -0.7% |

GPU part of a step: 17.6 (bytes/600GBs); RAPL: permission denied (energy_uj is root-only on FCHLLX01); CPU package energy recorded as null

time model (CPU-resident ms per step): BW_cpu fitted on the cpu arm = 75.9 GB/s (reference 69); phones: op15 60.1 GB/s rtt 1.50 ms (config), pixel 83.6 GB/s rtt 1.50 ms (config)

| arm | c | measured | overlap (fit) | err | aggregate (fit) | err | overlap @69 GB/s |
|---|---|---|---|---|---|---|---|
| cpu | 1 | 229.3 | 229.3 | -0% | 229.3 | -0% | 252.3 |
| cpu | 4 | 249.8 | 249.8 | +0% | 249.8 | +0% | 252.3 |
| cpu-helpers | 1 | 235.8 | 229.3 | -3% | 229.3 | -3% | 252.3 |
| cpu-helpers | 4 | 247.1 | 249.8 | +1% | 249.8 | +1% | 252.3 |
