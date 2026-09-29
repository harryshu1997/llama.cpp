# Three-model runtime scheduling R1

This is one matched physical pair on the RTX 4060 Ti 16 GiB desktop and OP15.
Both cases use trace SHA-256
`2eb73767d0a4e766db7e4c5f7047ee1436821e94a045d2630d84f9edc38ef814`,
all 84 requests, the same 13,132 output tokens, exact model hashes, zero
inference-process swap, and the same paid accounting boundary.

| Metric | All-server baseline | Runtime scheduler | Change |
| --- | ---: | ---: | ---: |
| Makespan | 190.364 s | 186.348 s | -2.11% |
| Fleet compute energy | 31.519 kJ | 29.597 kJ | -6.10% |
| Server compute energy | 31.380 kJ | 29.245 kJ | -6.80% |
| CPU package energy | 7.870 kJ | 5.602 kJ | -28.82% |
| GPU board energy | 23.510 kJ | 23.643 kJ | +0.57% |
| Whole-phone energy | 0.139 kJ | 0.352 kJ | +153.48% |
| Throughput | 68.984 tok/s | 70.470 tok/s | +2.16% |
| SLOs met | 61/84 | 65/84 | +4 |
| Mean GPU utilization | 85.36% | 88.69% | +3.33 points |

The scheduler generated ten runtime decisions and selected `phone-adreno` for
all ten Llama requests. Their mean completion time fell from 36.595 s to
4.013 s, maximum completion fell from 74.925 s to 10.305 s, and SLOs improved
from 6/10 to 10/10. Qwen and Gemma kept the same GPU sequence in both runs.

The 1.922 kJ fleet saving comes from 2.268 kJ less CPU-package energy, minus
0.213 kJ more whole-phone energy and 0.133 kJ more GPU-board energy. This is
why the result is positive but does not approach the earlier 25% two-model
FP16 placement result: only 10 of 84 requests changed device, and large-model
GPU work still dominates total energy.

Two longest phone requests exceeded the measured full-task latency upper bound
by 98.518 ms and 109.150 ms. Both still met the 30 s SLO. The other apparent
late lease in each request is the expected 1 ms USB dispatch phase and is now
classified separately by the runner. The profile needs a small concurrency
calibration margin before this route is called conservatively qualified.

Initial model load and warmup are outside the paid interval. The runtime case
preloaded the phone model in 14.252 s; therefore this is a resident-task policy
screen, not a cold-start energy claim. A single matched pair demonstrates the
mechanism but is not a repeated A-B-B-A qualification.

Raw evidence remains on the physical host under:

`/home/zhihao/s41-dynamic-ffn-v1/campaign/s42-three-model-runtime-v1`

The comparison inputs are hash-bound in
[`COMPARISON_R1.json`](COMPARISON_R1.json).
