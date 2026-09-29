# S42 operator energy model V1

Date: 2026-08-06 EDT.

## Scope

The scheduler estimates route energy by adding operator energy across the CPU
package, GPU board, phone, and transport energy domains. It does not infer
energy from model parameter count alone and it does not use nominal peak
TFLOPS as effective throughput.

Each kernel profile is bound to a backend, data type, operator family, and
shape bucket. Its effective throughput, bandwidth, launch time, idle power,
and active power must come from the target device or be labeled estimated.

## Equation

For operator `i` using kernel profile `k`:

```text
compute_us = ceil(compute_ops * 1e6 / effective_ops_per_s)
memory_us  = ceil(memory_bytes * 1e6 / effective_bytes_per_s)
active_us  = invocations * launch_us + max(compute_us, memory_us)

dynamic_uj = ceil((active_power_mw - idle_power_mw) * active_us / 1000)
```

For every energy domain reserved by the route:

```text
idle_uj = ceil(idle_power_mw * route_service_us / 1000)

route_energy_uj = fixed_uj + sum(idle_uj) + sum(dynamic_uj)
```

`compute_ops` counts primitive arithmetic operations. A dense `M x K` by
`K x N` matmul therefore contributes `2 * M * N * K`. `memory_bytes` includes
the bytes actually read or written by that kernel, including weights,
activations, KV data, and outputs. Quantized weights use their physical byte
count, not an F16 equivalent.

Compute and memory time use a roofline maximum because a kernel can overlap
both. Launch time remains additive. Energies from concurrent CPU, GPU, and
phone branches add even though their latencies overlap.

For the conservative first OP15 profile, include the phone domain only in the
offload route and set both idle and active power to 5,000 mW. This charges 5 W
for the complete route-service interval. A later measured profile may separate
connected-idle and active phone power.

`placement_planner.py` applies this same kernel equation to every candidate.
It then adds fixed and byte-dependent energy for every directed transfer hop.
For a fork-join operator, branch latency uses a maximum while branch energy is
additive. Phone HTP and GPU share one memory pool and phone-system energy
domain; independently overlapping kernels in that domain require one measured
fused-domain profile rather than two additive estimates.

## Safety rules

- A kernel cannot report active power below its domain idle power.
- Work without an invocation is rejected.
- Summed active time within one energy domain cannot exceed route service
  time. Fused or overlapping kernels in one domain must be represented as one
  measured kernel profile instead of being double counted.
- Missing request work features fail closed.
- Scheduler enforcement requires measured energy on both routes, identical
  energy-boundary IDs, and the existing uncertainty margin.
- Operator profiles are initially estimated. They become measured only after
  physical coefficient acquisition and full-route held-out error at most 10%.

## Initial calibration set

Keep the first physical set narrow:

1. CUDA Q4 GEMV/matmul and attention KV scan on the RTX 4060 Ti.
2. CPU Q4 GEMV/matmul, attention KV scan, and elementwise/norm on the
   i9-12900K.
3. OP15 HTP FFN slice and the direct-DMA activation path.
4. Host merge and USB transfer for the existing FFN offload route.

Qwen3-14B and Gemma4-12B then reuse those kernel rows. Whole-model runs are
held-out validation rather than a separate fitted model for every placement.
