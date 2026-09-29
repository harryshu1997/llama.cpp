# analyze.py results/20260928-wifi-s1

model /home/myid/zs89458/Documents/models/Qwen3-14B-Q4KM-dequant-f16.gguf: 40 layers, CPU-resident layers 0-23 + output head = 17.41 GB/step; FFN 534.8 MB/layer

| arm | c | step ms | ms/tok | tok/s | CPU-res ms | rtt p50 | rtt p99 | comp p50 | host p50 | GPU W | GPU J/tok | CPU J/tok | ident | step vs cpu | GPU J vs cpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu | 1 | 246.3 | 246.3 | 4.06 | 228.6 | - | - | - | - | 102.0 | 23.75 | - | - | +0.0% | +0.0% |
| cpu | 4 | 267.6 | 66.9 | 14.95 | 250.0 | - | - | - | - | 109.1 | 6.94 | - | - | +0.0% | +0.0% |
| phone | 1 | 423.5 | 423.5 | 2.36 | 405.9 | 14.17 | - | 7.82 | 0.01 | 104.5 | 41.86 | - | 4/5 | +72.0% | +76.2% |
| phone | 4 | 707.4 | 176.8 | 5.65 | 689.7 | 14.17 | - | 7.82 | 0.01 | 101.5 | 16.85 | - | 7/8 | +164.3% | +142.8% |
| split-25 | 1 | 254.1 | 254.1 | 3.94 | 236.5 | 7.26 | - | 2.18 | 5.60 | 110.3 | 26.58 | - | 5/5 | +3.2% | +11.9% |
| split-25 | 4 | 361.0 | 90.2 | 11.08 | 343.3 | 7.26 | - | 2.18 | 5.60 | 107.7 | 9.18 | - | 5/8 | +34.9% | +32.3% |

GPU part of a step: 17.6 (bytes/600GBs); RAPL: permission denied (energy_uj is root-only on FCHLLX01); CPU package energy recorded as null

| arm | helper | calls | rpc p50 | rpc p99 | compute p50 | net+overhead p50 | host p50 | wait p50 |
|---|---|---|---|---|---|---|---|---|
| phone | op15-htp0 | 5328 | 14.41 | - | 7.75 | 6.66 | 0.01 | 14.50 |
| phone | op15-htp1 | 5328 | 14.17 | - | 7.82 | 6.35 | 0.01 | 14.24 |
| phone | op15-htp2 | 5328 | 14.03 | - | 7.82 | 6.21 | 0.01 | 14.11 |
| phone | pixel | 5328 | 11.36 | - | 6.75 | 4.61 | 0.01 | 11.42 |
| split-25 | op15-htp0 | 5328 | 7.67 | - | 2.09 | 5.58 | 5.60 | 2.18 |
| split-25 | op15-htp1 | 5328 | 7.07 | - | 2.18 | 4.89 | 5.57 | 1.60 |
| split-25 | op15-htp2 | 5328 | 7.26 | - | 2.19 | 5.07 | 5.58 | 1.75 |
| split-25 | pixel | 5328 | 6.91 | - | 1.86 | 5.05 | 5.60 | 1.46 |

time model (CPU-resident ms per step): BW_cpu fitted on the cpu arm = 76.2 GB/s (reference 69); phones: op15 68.4 GB/s rtt 6.35 ms (phone-arm), pixel 79.2 GB/s rtt 4.61 ms (phone-arm)

| arm | c | measured | overlap (fit) | err | aggregate (fit) | err | overlap @69 GB/s |
|---|---|---|---|---|---|---|---|
| cpu | 1 | 228.6 | 228.6 | +0% | 228.6 | +0% | 252.3 |
| cpu | 4 | 250.0 | 250.0 | +0% | 250.0 | +0% | 252.3 |
| phone | 1 | 405.9 | 383.3 | -6% | 289.0 | -29% | 389.6 |
| phone | 4 | 689.7 | 388.9 | -44% | 294.6 | -57% | 389.6 |
| split-25 | 1 | 236.5 | 247.4 | +5% | 328.5 | +39% | 253.6 |
| split-25 | 4 | 343.3 | 253.0 | -26% | 345.9 | +1% | 253.6 |
