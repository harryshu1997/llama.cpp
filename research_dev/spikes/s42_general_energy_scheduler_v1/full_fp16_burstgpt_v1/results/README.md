# Two-model F16 BurstGPT result

Status: PASS, energy saving.

This is a physical A-B-B-A comparison on the desktop RTX 4060 Ti 16 GiB
host and the OP15 phone. Both arms execute the same 74-request BurstGPT
trace with 33,843 input tokens and 11,605 output tokens. The output lengths
are not shortened.

The model files are F16 dequantized proxies:

- Qwen3-14B: `Qwen3-14B-Q4KM-dequant-f16.gguf`, 29,543,423,360 bytes,
  SHA-256 `d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718`
- Gemma4-12B: `gemma-4-12B-Q40-dequant-f16.gguf`, 23,832,065,056 bytes,
  SHA-256 `ed76f2183d2d1d65091986033023e6c78d27f6276c1b0c5826cc92acf73538cf`
- Trace SHA-256:
  `b20a9ba66ee3558d835a0e19ed3cfa4c31a4a9e8b4f9c085b29a14f80250a0ff`

## Policy under test

The GPU holds one partial model placement at a time. Qwen runs first with 18
layers on CUDA, then the scheduler performs one model switch and runs Gemma
with 25 of 49 layers on CUDA.

The control sends the non-GPU work to the CPU. The treatment uses the same
GPU placement and lets the unified scheduler route qualified FFN work to the
phone:

| Phase | Phone execution | Resident sessions | Qualified shapes |
|---|---|---|---|
| Qwen | Full replacement for FFN layers 0-11 | HTP1 and HTP2 | M=1..4, 17,408 columns, SwiGLU |
| Gemma | Parallel suffix split for FFN layers 0-22 | HTP0 | M=1..16, 6,144 columns, GeGLU |

All three weight-slice sessions are loaded before the paid trace and remain
resident. USB carries activation rows and control messages during inference;
it does not reload the model slices for every operator call. The phone was
kept resident in both arms so that its idle platform cost is included fairly.

## Mean result

| Metric | CPU-only overflow control | CPU plus OP15 treatment | Change |
|---|---:|---:|---:|
| Makespan | 2,779.145 s | 2,578.050 s | -7.236% |
| CPU package energy | 201.509 kJ | 128.806 kJ | -36.079% |
| GPU board energy | 94.643 kJ | 88.868 kJ | -6.103% |
| Server compute energy | 296.152 kJ | 217.673 kJ | -26.499% |
| Whole-phone energy | 4.721 kJ | 6.365 kJ | +34.823% |
| Fleet energy | 300.873 kJ | 224.038 kJ | -25.537% |
| Actual GPU switch interval | 58.580 s | 48.950 s | -16.438% |

The mean fleet saving is 76.835 kJ. The phone consumed 1.644 kJ more, but
that enabled a 72.703 kJ CPU-package reduction and a 5.776 kJ GPU-board
reduction. The shorter critical path reduces the time for which the GPU and
host remain powered while CPU overflow work is pending.

The two independently paired fleet savings were 26.195% and 24.877%. This
means the headline result is not caused by only one favorable repeat.

## Run accounting

| Run | Makespan | CPU | GPU | Phone | Fleet | Phone operator calls |
|---|---:|---:|---:|---:|---:|---:|
| A1 control | 2,778.250 s | 201.493 kJ | 95.480 kJ | 4.559 kJ | 301.531 kJ | 0 |
| B1 OP15 | 2,559.802 s | 127.589 kJ | 88.693 kJ | 6.263 kJ | 222.545 kJ | 42,370 |
| B2 OP15 | 2,596.298 s | 130.022 kJ | 89.043 kJ | 6.466 kJ | 225.530 kJ | 44,625 |
| A2 control | 2,780.040 s | 201.524 kJ | 93.807 kJ | 4.883 kJ | 300.214 kJ | 0 |

Every run completed all 74 requests and all 11,605 output tokens. Every
treatment phone call was conserved across the server, bridge, and resident
router receipts. Reset recovery count and sampled inference-process swap were
both zero. The phone retained at least 2 GiB available memory after loading
all three resident HTP sessions.

All four arms had zero original-trace SLO hits because the large F16 proxy
models are much slower than the quantized models for which those deadlines
were defined. This experiment therefore supports an energy and makespan
claim, not an absolute SLO claim.

## Evidence

The aggregate record is
[`FULL_FP16_BURSTGPT_ABBA_V2.json`](FULL_FP16_BURSTGPT_ABBA_V2.json).

- Record SHA-256:
  `612992910a109c61f967e0f2a42033284c47675b8124aa403d7cbc4bb1f86a6c`
- File SHA-256:
  `b62721f12d02b83e0f0f102809a7b741a6edcf9dcdda3fe166562a1a7594b92d`
- Unified plan file SHA-256 in every arm:
  `a55806a14c5808e442c31831b7afc6a68f7f195813343176c28b0f5f0641cc3b`

The immutable raw evidence remains on the measurement host under:

- `/home/zhihao/s42-full-fp16-burstgpt-v1/control-r1b`
- `/home/zhihao/s42-full-fp16-burstgpt-v1/op15-r1`
- `/home/zhihao/s42-full-fp16-burstgpt-v1/op15-r2`
- `/home/zhihao/s42-full-fp16-burstgpt-v1/control-r2`

An earlier `control-r1` integration attempt failed before Gemma inference and
is not used. `control-r1b` is the first complete, strictly validated A arm.

## Successor policy screen

A shape-aware Gemma suffix split was subsequently tested in a physical
Gemma-only A-B-B-A screen. It improved mean makespan by 1.97% but increased
mean fleet energy by 0.79%, with one of two paired repeats regressing both
metrics. It therefore was not promoted to this complete trace. The fixed
6,144-column Gemma split described above remains the qualified policy and the
25.537% fleet-energy result remains the current full-trace result. See
[`../shape_balance_v1`](../shape_balance_v1/README.md).

A later protected-first shared-phone screen kept Qwen and Gemma connected to
the three resident HTP sessions at once. Its native arbiter passed all timing,
completion, request-accounting, and reset gates, but the matched four-run
screen reduced mean fleet energy by only 3.206%. Gemma decode slowed 4.451%,
whole-phone energy rose 4.408%, and one duration pair regressed. The apparent
total-time gain came from unchanged CPU-only prefill. That candidate is also
not promoted. See
[`../gpu_wavefront_v1/PHONE_ARBITER_RESULTS_V1.md`](../gpu_wavefront_v1/PHONE_ARBITER_RESULTS_V1.md).

## Runtime-selected placement retest

The unified scheduler now captures a live GPU, host-RAM, and phone-RAM
snapshot before the run. It evaluates whole-trace placement candidates, applies
capacity and measured-evidence gates, and emits a hash-bound execution receipt.
The physical runner obtains its layer counts and phone routing values from that
receipt instead of hardcoded command-line placement values.

On the retest snapshot, the scheduler selected the qualified CPU plus OP15
placement above. The full-GPU candidate failed the 4060 Ti VRAM gate. The
Qwen-15/Gemma-1 wavefront candidate failed the protected host-RAM gate. The
selected placement required 15,168,700,416 GPU bytes, 19,734,474,752 host
bytes, and 9,673,170,944 phone bytes. All three fit the captured capacities
after the configured reserves.

One exact-work physical retest completed all 74 requests and all 11,605 output
tokens:

| Metric | GPU plus CPU mean | Runtime-selected GPU, CPU, OP15 | Change |
|---|---:|---:|---:|
| Makespan | 2,779.145 s | 2,666.618 s | -4.049% |
| CPU package energy | 201.509 kJ | 135.855 kJ | -32.581% |
| GPU board energy | 94.643 kJ | 92.529 kJ | -2.234% |
| Server compute energy | 296.152 kJ | 228.384 kJ | -22.883% |
| Whole-phone energy | 4.721 kJ | 6.523 kJ | +38.181% |
| Fleet energy | 300.873 kJ | 234.907 kJ | -21.925% |
| Actual GPU switch interval | 58.580 s | 49.755 s | -15.064% |

The retest therefore saves 65.966 kJ and 112.527 s relative to the matched
GPU plus CPU baseline mean. The phone consumes 1.802 kJ more, but it removes
65.654 kJ from CPU package energy and 2.114 kJ from GPU board energy.

This single retest is 4.851% higher in fleet energy and 3.435% slower than the
prior two-run OP15 mean. Qwen reproduced the earlier call geometry and phase
time. Gemma transferred approximately the same activation volume but split it
into 28,336 calls, versus 22,126 and 24,357 in the qualified repeats. Its mean
prefill time rose to 45.093 s, producing a longer critical tail even though
the median phone RPC became faster. This is batching variance, not a capacity,
reset, swap, or USB failure.

Mean GPU utilization was 42.598%, with p50 14% and p95 100%. The selected
qualified policy is still a sequential partial-model switch, so it does not
keep CUDA continuously busy while CPU and phone dependencies complete. A
concurrent wavefront was considered at runtime but was not admitted by the
protected host-memory snapshot.

The validated compact record is
[`RUNTIME_PLACEMENT_RETEST_V1.json`](RUNTIME_PLACEMENT_RETEST_V1.json).

- Record SHA-256:
  `91fb5e49ee19d24a581b395ef3cdc831b631fe0b6633179fef410f320685ba9c`
- File SHA-256:
  `43e6ec79447d8356cbac8c62d9c1708379fd6e065d32d78b88ea339a42d878d3`
- Runtime decision SHA-256:
  `sha256:d15457008cb65ca707cd098ec277897b326fa3025c03960f54872320e838764d`

The immutable raw evidence remains on the measurement host under:

- `/home/zhihao/s42-runtime-placement-full-fp16-v1/op15-r51`
- `/home/zhihao/s42-runtime-placement-full-fp16-v1/op15-r51.phone-capture`

This one run confirms the runtime binding path. It does not replace the prior
two-pair ABBA campaign as the qualification evidence for the placement.

## Runtime utilization graph

![Runtime GPU, CPU, and OP15 activity](../../../../figures/RUNTIME_DEVICE_UTILIZATION_V1.png)

The three aligned panels use the exact paid-trace clock. The GPU line is
measured NVML utilization in five-second display bins. CPU bars are the
system-wide `sar` utilization intervals covering the run; the dashed CPU line
is the ten-second RAPL package-power trace. The OP15 line is a normalized
whole-phone-power activity proxy. This run did not capture a Qualcomm HTP
hardware-utilization counter, so the last panel must not be interpreted as
literal DSP occupancy.

The vertical transition markers show Qwen completion at 1,879.087 seconds and
Gemma CUDA readiness at 1,928.843 seconds. The arrows between the CPU and OP15
panels identify Qwen FFN routing to HTP1/HTP2 and Gemma suffix routing to HTP0.

- Scalable graph: [`RUNTIME_DEVICE_UTILIZATION_V1.svg`](../../../../figures/RUNTIME_DEVICE_UTILIZATION_V1.svg)
- PNG graph: [`RUNTIME_DEVICE_UTILIZATION_V1.png`](../../../../figures/RUNTIME_DEVICE_UTILIZATION_V1.png)
- CPU evidence: [`RUNTIME_PLACEMENT_CPU_SAR_V1.json`](RUNTIME_PLACEMENT_CPU_SAR_V1.json)
- Reproducer: [`../plot_runtime_device_utilization.py`](../plot_runtime_device_utilization.py)
