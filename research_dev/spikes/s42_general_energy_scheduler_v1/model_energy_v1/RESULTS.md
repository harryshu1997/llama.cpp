# S42 Gemma4 and Qwen3 energy model V1 results

Date: 2026-08-06 EDT.

Verdict: `OPERATOR_ENERGY_MECHANICS_PASS; GEMMA_Q4_LOCAL_LAYER_AGGREGATE_PASS; QWEN_MIXED_QUANT_LOCAL_LAYER_AGGREGATE_PASS; PER_OPERATOR_ATTRIBUTION_ESTIMATED; FULL_ROUTE_CALIBRATION_PENDING`.

## Gemma4 local-layer pilot

A bounded physical RTX 4060 Ti pilot now calibrates a deterministic Q4_0
Gemma-shaped local-layer operator model at KV=136 and KV=8192 and validates it
at held-out KV=512. The model predicts 54.013 mJ/layer versus 57.446 mJ/layer
measured, an absolute error of 5.98%. Three fresh executions were acquired at
every context using 50 ms GPU-board power integration. This is a complete
layer proxy and does not load the GGUF model.

The detailed per-operator table and evidence are in
`GEMMA4_OPERATOR_ENERGY_PILOT_V1.md`. The aggregate layer passes the 10% pilot
gate. Individual operator rows remain traffic-attributed estimates, not
direct measurements, so they are not yet eligible for scheduler enforcement.

## Qwen3-14B mixed-quantization local-layer pilot

The corresponding Qwen3-14B proxy uses the checkpoint's exact per-tensor
block mix: Q, K, O, gate, and up are Q4_K; V and down are Q6_K. On the
physical RTX 4060 Ti, KV=136 and KV=8192 fit the model and KV=512 remains held
out. It predicts 106.662 mJ/layer versus 105.794 mJ/layer measured, an
absolute error of 0.82%. Three fresh executions were acquired at every
context using 50 ms GPU-board power integration.

The detailed table and evidence are in
`QWEN3_OPERATOR_ENERGY_PILOT_V1.md`. The three FFN projections receive 81.55%
of the traffic-attributed estimate. This is not a measured per-projection
breakdown, and Qwen phone energy remains pending because the current HTP route
does not support its Q4_K gate and up tensors.

An independent RTX A6000 GPU 1 diagnostic passes the same held-out gate at
7.73% absolute error. Its device-specific coefficients are not reused for the
RTX 4060 Ti profile.

## Implemented

The bounded calibration substrate is complete:

- one immutable nine-case grid covering resident idle, prefill-heavy,
  decode-heavy, concurrency 1/4/8, and two held-out cases;
- exact Qwen3-14B and Gemma4-12B model and architecture identities;
- one-model-at-a-time llama-server acquisition with model loading excluded;
- synchronized RAPL and NVML integration for every paid case;
- a server-only wrapper that never contacts or mutates a phone;
- synchronized OP15 USB-input plus battery-discharge attachment for every
  case interval;
- an explicitly estimated `5 W * paid duration` phone component for the first
  prototype when no phone-energy receipt is supplied;
- exact cohort, token-row, decode-step, and estimated prefill-ubatch work;
- nonnegative component-energy fitting with held-out error gates;
- no-extrapolation route profiles; and
- fail-closed checks for model, runtime, device, swap, phone, bridge, and
  foreign-process state.

The scheduler energy representation is now operator-composed. For every
operator it records invocation count, primitive compute operations, physical
memory bytes, effective kernel throughput and bandwidth, launch time, and its
energy domain. It publishes operator and device-domain breakdowns, sums energy
across concurrent devices, rejects double-counted active time, and rejects
route comparisons with different energy boundaries.

The older cohort model is retained as a whole-route validation and residual
check. It is no longer intended to be the primary cross-model estimator.

The initial routes are `qwen-cuda`, `gemma-cpu`, and `gemma-op15`. Model-load
and model-switch energy remain a separate future profile.

The accounting boundaries are:

```text
measured server = CPU package RAPL + GPU board NVML integral
estimated OP15 route = measured server + 5 W * phone-reserved duration
```

The second line is intentionally labeled estimated in the generated profile.
It cannot satisfy a measured fleet-energy gate. It also charges no phone term
to a server-only route, making the initial offload comparison conservative.

As a sensitivity check on the already completed I3 BurstGPT campaign, applying
this rule to the published three-pair averages gives about 115.739 kJ for the
offload arm (`112.586 kJ + 5 W * 630.594 s`) versus 136.144 kJ for the
server-only arm, or about 14.99% lower estimated accounted energy. This is a
recalculation of existing evidence, not a new physical run.

## Whole-route physical status

The complete nine-case model route has not yet been acquired. During its first
attempt, two foreign Gemma llama-server processes appeared on ports 18663 and
18664 before the first paid acquisition. They held about 12.5 GiB of VRAM.
That whole-route Qwen pilot failed during CUDA model allocation before warmup,
preflight publication, or any paid case. It is an unpaid setup failure and is
not a calibration observation.

The later bounded Qwen layer acquisition waited for the shared BurstGPT
campaign to terminate, rechecked CUDA ownership, and completed without a
foreign process. A separate A6000 GPU 0 attempt was rejected after unrelated
SGLang processes appeared; only the clean A6000 GPU 1 diagnostic is retained.

The strengthened wrapper now exits with status 75 before touching the phone
when any foreign llama-server or CUDA process exists. A live check reproduced
that fail-closed path. No foreign process was stopped.

Blocked pilot evidence:

```text
/home/zhihao/s42-model-energy-v1/results/qwen-cuda-pilot-r1/FAILURE.json
error: RunError: hot server exited during load
cause: cudaMalloc failed while two foreign processes held 6,254 MiB each
paid cases: 0
```

## Verification

- Runner tests: 5 / 5 pass.
- Fitter tests: 5 / 5 pass.
- Phone attachment tests: 1 / 1 pass.
- Operator-energy tests: 9 / 9 pass.
- Complete S42 tests: 57 / 57 pass.
- Qwen CPU and CUDA path-matched correctness: exact.
- Qwen RTX 4060 held-out layer error: 0.82%.
- Qwen RTX A6000 GPU 1 held-out layer error: 7.73%.
- Qwen raw-log, power-sample, and source-snapshot hashes: pass.
- Python syntax: pass.
- Shell syntax: pass.
- ASCII scan: pass.
- `git diff --check`: pass.
- Earlier remote source deployment syntax and hashes: pass.
- Latest server-only wrapper and 5 W fitter update: local only; not deployed
  while the shared desktop is owned by another experiment.
- Foreign-process preflight: exits 75 before phone or model mutation.
- No commit or push performed.

## Next action

Acquire the remaining bounded kernel profile in this order:

1. RTX 4060 Ti CUDA Q4 GEMV/matmul and KV-scan buckets;
2. i9-12900K CPU Q4 GEMV/matmul, KV scan, and elementwise buckets; and
3. OP15 HTP FFN, direct-DMA transfer, and host merge.

Materialize Qwen and Gemma operator work from runtime counters, then use the
nine-case whole-route grid as held-out validation. Only coefficients and shape
buckets with at most 10% route-level error may become measured scheduler
profiles.
