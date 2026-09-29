# Stage 6 physical A/B

This runner preserves the hash-qualified I3 `llama-server`, bridge, worker,
models, FunctionFS transport, and dynamic FFN policy. Before changing device
state, `plan_unified_burstgpt.py` loads the exact cohort certificate through
`research_dev.scheduler`, selects the route, and emits a hash-bound execution
plan. The backend adapter derives its model, runtime, operator-split, worker,
DMA-BUF, and USB settings from that plan. It also adds two sidecars:

- a host sampler records the exact two-server topology, RTX thermal state,
  CPU package throttle counter, 5 Gb/s USB state, and bridge progress; and
- a phone-local sampler records Android thermal status plus battery, shell,
  CPU, NPU, GPU, and DDR temperatures while ADB is unavailable.

The sidecars do not alter tensor traffic or the bridge/worker binaries. Both
arms run the complete 74-request source-length BurstGPT trace with 33,843
input tokens and 11,605 output tokens. The control keeps the connected phone
idle, and the treatment selects FFN columns dynamically from:

```text
1:9664,3:8192,8:4096,128:8192,512:11136
```

Run one arm on the physical 4060 Ti host with:

```sh
run_stage6_arm.sh cpu 1 /absolute/result/cpu-r1 5037
run_stage6_arm.sh op15 1 /absolute/result/op15-r1 5037
```

`RUNTIME_GATE_RECEIPT.json` is fail-closed. It requires exact paid topology,
no desktop or Android thermal throttle, 5 Gb/s USB, no reset recovery, and a
fresh bridge-process plus bound-USB topology heartbeat. RPC progress must
reach all 52,320 treatment calls with no reset; idle gaps between RPC bursts
are reported as a diagnostic and are not treated as worker failure. Full result
validation and the paired comparison reuse the strict I3 campaign validator.
`analyze_stage6_pair.py` binds both runtime receipts, the synchronized energy
records, completed work, overlap, reset, and pinned MMLU64 authority into one
record; `render_stage6_pair.py` emits its compact comparison table.

The receipt also requires the result and preflight to carry the selected
unified-plan hash and route. The phone thermal sidecar is reduced after ADB is
restored, so this remains monitored validation rather than an atomic live
phone-thermal gate immediately before dispatch.

## Unified scheduler-owned result

The fresh repeat-5 pair on 2026-08-08 is recorded in
`UNIFIED_SCHEDULER_BURSTGPT_4060TI_OP15_R5_V1.json` and its companion report.
The control plan selected `i3-cold-cpu-control-v1`; the treatment plan selected
`i3-cold-cpu-op15-ffn-v1` for `VERIFIED_COHORT_ENERGY_SAVING`. Accounted fleet
energy changed from 135.663 kJ to 113.956 kJ (-16.00%), and makespan changed
from 735.298 seconds to 631.548 seconds (-14.11%). Both arms completed all 74
requests and 11,605 output tokens, met 55 SLOs, and passed their scheduler-plan
and runtime-gate bindings.

## GPU overflow shadow result

`GPU_OVERFLOW_BURSTGPT_4060TI_OP15_V1.json` records the matched 2026-08-09
GPU-switch pair. The scheduler reserved all CUDA lanes for the resident hot
model and switch, dispatched three evidence-supported cold requests to the
CPU plus OP15 helper only when their conservative finish bounds fit before the
switch cutoff, retired the helper at 71.126 seconds, and sent all 14 remaining
cold requests to full CUDA after the new model became resident.

| metric | GPU-only switch | overflow shadow | change |
| --- | ---: | ---: | ---: |
| makespan | 178.985 s | 177.933 s | -0.59% |
| throughput | 64.838 tok/s | 65.221 tok/s | +0.59% |
| SLO requests met | 55 | 57 | +2 |
| CPU package energy | 5.476 kJ | 8.022 kJ | +46.50% |
| GPU board energy | 23.011 kJ | 22.942 kJ | -0.30% |
| connected phone energy | 0.132 kJ | 0.256 kJ | +93.53% |
| accounted fleet energy | 28.619 kJ | 31.220 kJ | +9.09% |

The overflow route therefore remains shadow-only for latency and is not an
energy-saving route. The helper recovered two SLOs and 1.052 seconds of
makespan, but waking the desktop CPU package added 2.546 kJ while GPU energy
was nearly unchanged. Promotion requires a route whose measured full-fleet
energy beats waiting for the GPU, not merely one that finishes before the
switch.
