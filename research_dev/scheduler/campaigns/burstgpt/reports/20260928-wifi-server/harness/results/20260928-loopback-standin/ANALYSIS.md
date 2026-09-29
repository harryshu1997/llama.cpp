# analyze.py results/20260928-loopback-standin

model /home/myid/zs89458/Documents/models/Qwen3-14B-f16.gguf: 40 layers, CPU-resident layers 0-23 + output head = 17.41 GB/step; FFN 534.8 MB/layer

| arm | c | step ms | ms/tok | tok/s | CPU-res ms | rtt p50 | rtt p99 | comp p50 | host p50 | GPU W | GPU J/tok | CPU J/tok | ident | step vs cpu | GPU J vs cpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | 1 | 226.3 | 226.3 | 4.42 | 208.7 | - | - | - | - | 100.0 | 20.70 | - | - | +0.0% | +0.0% |
| cpu | 4 | 245.1 | 61.3 | 16.32 | 227.5 | - | - | - | - | 97.7 | 5.80 | - | - | +0.0% | +0.0% |
| cpu-helpers | 1 | 224.5 | 224.5 | 4.45 | 206.9 | - | - | - | - | 102.1 | 21.03 | - | 4/4 | -0.8% | +1.6% |
| cpu-helpers | 4 | 245.1 | 61.3 | 16.32 | 227.5 | - | - | - | - | 103.6 | 5.84 | - | 4/4 | +0.0% | +0.8% |
| phone | 1 | 333.1 | 333.1 | 3.00 | 315.5 | 10.93 | - | 10.67 | 0.01 | 98.9 | 30.81 | - | 4/4 | +47.2% | +48.8% |
| phone | 4 | 371.5 | 92.9 | 10.77 | 353.9 | 10.93 | - | 10.67 | 0.01 | 99.1 | 8.52 | - | 4/4 | +51.6% | +47.0% |
| split-50 | 1 | 279.2 | 279.2 | 3.58 | 261.6 | 9.02 | - | 8.53 | 3.87 | 104.9 | 27.79 | - | 4/4 | +23.4% | +34.2% |
| split-50 | 4 | 324.4 | 81.1 | 12.33 | 306.8 | 9.02 | - | 8.53 | 3.87 | 104.5 | 8.20 | - | 4/4 | +32.4% | +41.5% |

GPU part of a step: 17.6 (bytes/600GBs); RAPL: permission denied (energy_uj is root-only on FCHLLX01); CPU package energy recorded as null

| arm | helper | calls | rpc p50 | rpc p99 | compute p50 | net+overhead p50 | host p50 | wait p50 |
|---|---|---|---|---|---|---|---|---|
| phone | op15-htp0 | 696 | 9.94 | - | 9.83 | 0.11 | 0.01 | 10.00 |
| phone | op15-htp1 | 696 | 12.92 | - | 12.81 | 0.11 | 0.02 | 12.97 |
| phone | op15-htp2 | 696 | 10.93 | - | 10.67 | 0.26 | 0.01 | 11.06 |
| phone | pixel | 696 | 8.79 | - | 8.67 | 0.12 | 0.01 | 8.86 |
| split-50 | op15-htp0 | 696 | 8.34 | - | 8.03 | 0.31 | 3.83 | 4.59 |
| split-50 | op15-htp1 | 696 | 9.23 | - | 8.53 | 0.71 | 3.87 | 5.54 |
| split-50 | op15-htp2 | 696 | 8.25 | - | 7.96 | 0.29 | 3.82 | 4.50 |
| split-50 | pixel | 696 | 9.02 | - | 8.85 | 0.17 | 4.01 | 5.01 |

time model (CPU-resident ms per step): BW_cpu fitted on the cpu arm = 83.4 GB/s (reference 69); phones: op15 50.1 GB/s rtt 0.11 ms (phone-arm), pixel 61.7 GB/s rtt 0.12 ms (phone-arm)

| arm | c | measured | overlap (fit) | err | aggregate (fit) | err | overlap @69 GB/s |
|---|---|---|---|---|---|---|---|
| cpu | 1 | 208.7 | 208.7 | +0% | 208.7 | +0% | 252.3 |
| cpu | 4 | 227.5 | 227.5 | +0% | 227.5 | +0% | 252.3 |
| cpu-helpers | 1 | 206.9 | 208.7 | +1% | 208.7 | +1% | 252.3 |
| cpu-helpers | 4 | 227.5 | 227.5 | -0% | 227.5 | -0% | 252.3 |
| phone | 1 | 315.5 | 301.7 | -4% | 172.4 | -45% | 313.1 |
| phone | 4 | 353.9 | 306.6 | -13% | 177.3 | -50% | 313.1 |
| split-50 | 1 | 261.6 | 179.6 | -31% | 134.5 | -49% | 191.1 |
| split-50 | 4 | 306.8 | 184.6 | -40% | 146.4 | -52% | 191.1 |
