# Matmul offload solver v1

This directory contains an offline screening solver for one desktop and one
OP15. It accepts one matmul, an ordered matmul chain, a fused gated FFN, or an
MoE expert group. It enumerates legal cuts and reports the backend assignment,
cut size, activation traffic, resident phone weights, median estimate, p90
estimate, and rejection reasons.

It does not modify llama.cpp placement. A recommendation is a benchmark
shortlist, not a runtime claim. The exact shape still needs an end-to-end
measurement before use.

## Quick start

```sh
solver=research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/offload_solver_v1/offload_solver.py
examples=research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/offload_solver_v1/examples

python3 "$solver" "$examples/single_matmul_q8_cpu.json"
python3 "$solver" "$examples/ordered_chain_q8_cpu.json"
python3 "$solver" "$examples/gemma4_12b_ffn_q8_cpu.json" --format json
python3 "$solver" "$examples/gemma4_26b_a4b_moe_q8_cpu.json"
```

Use a different profile with `--profile`. `--top 0` emits every candidate.
`--allow-unqualified` lets provisional kernels participate in selection; it
does not remove warnings or the exact-shape validation requirement.

## Input contract

Every workload uses schema `s41-offload-workload-v1` and selects a desktop
backend, phone backends, weight format, activation format, and transport.
`transport: "auto"` evaluates both AOA and NCM.
Set `require_profiled_shape: true` when an extrapolated batch shape must be a
hard blocker instead of a provisional shortlist candidate.

For a matmul, use the convention:

```text
W[N,K] * X[K,M] = Y[N,M]
```

The four workload kinds are:

| kind | required shape | legal search |
| --- | --- | --- |
| `matmul` | `m`, `n`, `k` | N/output rows, K/reduction columns, M/batch, or full offload |
| `chain` | ordered `operations`, each with `m`, `n`, `k` | host/phone operation islands |
| `ffn` | `m`, `hidden`, `intermediate`, `gate_count` | aligned intermediate columns across gate, up, and down |
| `moe` | `m`, `hidden`, expert widths and counts | whole active experts |

The chain validator requires `N[i] == K[i+1]` and the same M. Consecutive
phone operations form one island: their intermediate activation remains on
the phone, and USB is paid only on entry and exit. A bounded beam search also
allows multiple phone islands separated by host operations.

For MoE, `expert_residency: "all"` reserves the full expert bank so arbitrary
routed IDs are ready before inference. `expert_residency: "active"` reserves
only one request's active set and is valid only if another mechanism prepares
the routed weights in advance.

## Cost model

For work with `F` GFLOP and `B` MB of memory traffic on backend `d`:

```text
T_device(d) =
    dispatch(d) +
    max(1000 * F / (P_d * eta_compute),
        B / (BW_d * eta_memory))
```

`P_d` is effective GFLOP/s and `BW_d` is effective GB/s. The phone shape
model lowers `eta_compute` and `eta_memory` for narrow or small-batch work.
M=1 is the measured regime. M>1 uses a conservative saturation heuristic and
is labeled as an extrapolation in candidate details.

For a parallel cut:

```text
T_remote = T_gpu_staging + T_wire + T_phone
T_split  = max(T_host_remaining, T_remote) + T_merge
```

For an ordered chain, all segments are dependent and therefore add:

```text
T_chain = sum(T_host_segments) + sum(T_phone_islands)
```

The current transport fits use total request plus response payload `S_KiB`:

```text
AOA p50 = 0.160 + 0.00860 * S_KiB ms
NCM p50 = 1.750 + 0.00256 * S_KiB ms
```

Their median crossover is about 263 KiB. AOA is normally selected for decode
activations; NCM is selected for sufficiently large batch payloads. The AOA
profile also includes the observed idle-gap penalty.

The default acceptance rule is:

```text
candidate median <= 0.95 * all-desktop median
candidate p90    <= all-desktop p90
```

Both must pass. Weight capacity, backend qualification, and GPU staging must
also pass.

## Split traffic

For activation element width `a` bytes:

| split | host/phone work | request bytes | response bytes | merge |
| --- | --- | ---: | ---: | --- |
| output rows Np | `(N-Np)` and `Np` rows | `M*K*a` | `M*Np*a` | concatenate |
| reduction Kp | `(K-Kp)` and `Kp` columns | `M*Kp*a` | `M*N*a` | sum partial Y |
| batch Mp | `(M-Mp)` and `Mp` tokens | `Mp*K*a` | `Mp*N*a` | concatenate |
| FFN width Ip | remaining and phone intermediate columns | `M*H*a` | `M*H*a` | sum residual |
| MoE Ep | remaining and phone whole experts | `M*H*a` | `M*H*a` | weighted sum |

Batch splitting duplicates the full weight matrix on both devices. It can
improve batch throughput, but its single-request latency still has to beat
the same transfer and merge policy.

## Current profile

`current_profile.json` records the screening values gathered in the S41
campaign:

| path | effective memory rate | effective compute | recurring dispatch |
| --- | ---: | ---: | ---: |
| isolated desktop CPU | 50.0 GB/s | screening ceiling | 0 |
| full desktop CPU engine | 29.7 GB/s | screening ceiling | 0 |
| RTX 4060 Ti Q8 FFN | 269 GB/s | 22 TFLOP/s screening value | 0.010 ms |
| OP15 HTP fused Q8 FFN | 54 GB/s | 90.6 GFLOP/s | 0.091 ms |
| OP15 OpenCL large M=1 matmul | 70 GB/s | provisional | 0.532 ms |
| OP15 persistent GPU | 70 GB/s provisional | provisional | 0.00682 ms |

The OpenCL FFN/MoE family override is deliberately much lower than the large
single-matmul rate. The real 1,792-column complete FFN took 2.810 ms on
OpenCL versus 0.743 ms on HTP. Many queue-bound kernels cannot inherit a
large GEMV bandwidth number.

The HTP path is also qualified for the measured Q4_0 M=1 FFN family. The
latest exact Gemma4 12B profile uses four HVX threads, FP16 activation wire
packing, a persistent desktop I/O thread, and 9664 of 15360 intermediate
columns. Its 1.217-1.223 ms HTP median and direct DMA-BUF transport are an
exact-shape calibration, not a general replacement for the screening
roofline. It raises phone compute slightly for the HTP casts but cuts each
wire transfer from 15488 to 7808 bytes. The direct transport is not yet added
to the solver's linear transport choices because the two full-model points
are exact F32 and F16 FFN calibrations, not an independent transport fit.

The persistent GPU entry is diagnostic only. The 0.00682 ms doorbell result
was a one-workgroup SQR control, not a quantized persistent matmul. It is
blocked from recommendations unless `--allow-unqualified` is used.

The RTX 4060 Ti profile intentionally has no D2H/H2D staging fit. When the
input starts in VRAM, every phone candidate is blocked with
`missing measured GPU D2H/H2D staging profile`. Enter pinned PCIe direction
fits only after measuring them; assuming zero staging would create a false
phone win.

## Example screening results

With the checked-in profile:

| workload | result |
| --- | --- |
| Q8 M=1 `N=15360, K=3840` against isolated CPU | model selects HTP output-row split: 5,152 phone rows and 10,208 host rows |
| Q8 dense FFN `H=3840, I=15360` against isolated CPU | model selects HTP: 6,720 phone columns and 8,640 host columns |
| Q8 top-8 expert group `H=2816, Iexpert=704` | model selects HTP: 2 phone experts and 6 host experts |
| three dependent matmuls in the example chain | keep all on desktop |
| any current RTX 4060 Ti input | keep all on desktop because PCIe staging is unmeasured |

The MoE example reserves all 128 synthetic expert weight sets on the phone:
808,845,312 bytes resident. Only 12,638,208 bytes of weights are read for the
two selected experts in one request. Its model estimate is 0.764 ms versus a
1.014 ms isolated-CPU roofline, but this remains provisional until that exact
grouped-expert kernel and concurrent host branch are measured.

These examples are working shapes from the current spike, not authoritative
model metadata. Read the actual GGUF dimensions and expert counts into the
input before making a placement decision.

## Calibration and limitations

Edit or copy `current_profile.json` when hardware or kernels change.
`family_overrides` prevents a fused multi-kernel graph from inheriting an
unrelated GEMV rate. Keep unqualified paths blocked until correctness,
recurring dispatch, exact-shape median, and exact-shape p90 all exist.

Important missing effects in v1:

- no exact-shape lookup table, so aligned performance cliffs such as the
  measured 1,664-column anomaly are not predicted;
- no CPU/USB process-scheduling contention multiplier;
- no thermal or DVFS state model;
- no phone RAM contention between GPU, HTP, and CPU;
- no weight preparation time in the request path;
- one phone only, and one selected phone backend per ordered chain;
- no numerical-error budget beyond choosing structurally valid split axes.

The output explicitly reports preloaded weights and excludes one-time
preparation from request latency. If weights cannot remain resident, the
candidate must be rejected or the preparation cost must be added externally.

## Tests

```sh
python3 -m unittest -v \
  research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/offload_solver_v1/test_offload_solver.py
```
