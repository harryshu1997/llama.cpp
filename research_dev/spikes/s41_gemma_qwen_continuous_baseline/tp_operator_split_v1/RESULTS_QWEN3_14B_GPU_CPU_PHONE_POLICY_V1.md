# Qwen3 14B GPU plus CPU plus OP15 policy screen

Date: 2026-08-06 EDT

Status: physical RTX 4060 Ti and i9-12900K Q6_K head components plus a
modeled OP15 HTP branch. No three-way end-to-end execution has run.

## Verdict

Do not move whole Qwen layers away from an available RTX 4060 Ti. The exact
8.4 GB Q4_K_M model fits the 16 GB card, and ordinary projection and FFN
boundaries are too frequent for GPU-to-host-to-phone traffic.

There is one strong three-way candidate: split the Q6_K vocabulary rows across
CUDA, CPU, and OP15, run the three row ranges concurrently, and merge their
results once per requested logit row. Each device keeps its weight rows
resident. The 5,120-element final hidden vector is copied from CUDA to host
once; the host copy is also the phone request.

For `Mh=1`, the first screening point is:

| backend | vocabulary rows | fraction | measured or modeled branch p50 |
| --- | ---: | ---: | ---: |
| RTX 4060 Ti | 121,728 | 80.1% | 1.886 ms measured compute |
| i9 CPU, 8 threads | 15,360 | 10.1% | 1.539 ms measured compute |
| OP15 HTP | 14,848 | 9.8% | 1.883 ms modeled RPC, compute, and top-k |

With a compact local top-k contract, the screen gives about 1.91 ms p50
versus 2.352 ms for the full CUDA matmul, an 18.8% operator reduction. The
conservative modeled p90 is 2.20 ms versus a 2.355 ms CUDA compute p90, a
6.7% margin. These percentages are not full-model speedups.

If arbitrary full-logit sampling must be preserved, GPU plus CPU is the first
implementation. A phone branch would need to return its approximately 29 KiB
F16 logit slice instead of local top-k. That adds little bulk-transfer time,
but the full-logit HTP response path has not been implemented or validated.

## Physical CPU and CUDA measurements

The existing backend benchmark used the target RTX 4060 Ti machine, Q6_K
weights, `K=5120`, F16 input/output, one matrix, and the exact Qwen vocabulary
width. Weights were prepared before timing.

| backend and rows | compute p50 | compute p90 | complete p50 | complete p90 |
| --- | ---: | ---: | ---: | ---: |
| CUDA, 151,936 rows | 2.352 ms | 2.355 ms | 2.735 ms | 2.806 ms |
| CUDA, 136,576 rows | 2.115 ms | 2.117 ms | 2.449 ms | 2.474 ms |
| CUDA, 121,728 rows | 1.886 ms | 1.889 ms | 2.186 ms | 2.253 ms |
| CPU, 15,360 rows | 1.539 ms | 1.609 ms | 1.557 ms | 1.627 ms |

`complete` includes benchmark input publication and retrieval of every F16
logit. A compact-top-k runtime would replace the large CUDA output retrieval
with device-side reduction in both the control and treatment. The compute
column is therefore the comparable physical anchor for that contract.

The GPU plus CPU full-logit component bound is:

```text
max(CUDA 136576 rows, CPU 15360 rows) + coordination
    = max(2.449, 1.557) ms + coordination
```

It removes 10.1% of the CUDA rows and has about 0.29 ms raw head-latency
margin before the 2.735 ms full-CUDA total. The concurrent implementation
must still measure the final-hidden D2H copy, thread wakeup, interference,
and result concatenation.

## Dynamic head split

The phone estimates use the physical 111.2 GOP/s Q6_K HTP result, 465 MB/s
direct DMA, the 0.235 ms RPC floor, the measured Q6_K worker overhead, and a
conservative p90 uplift. CPU and CUDA entries are physical component timings.

| requested logit rows Mh | CPU rows | phone rows | CUDA rows | modeled p50 gain | p90 decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 15,360 | 14,848 | 121,728 | 18.8% | shortlist; 6.7% margin |
| 2 | 15,360 | 7,936 | 128,640 | 14.3% | shadow; only 1.4% margin |
| 3-4 | 12,288 to 14,336 | 0 | remainder | about 8-9% CPU-only | phone p90 screen fails |
| 5-8 | 8,192 | 0 | 143,744 | about 5% CPU-only at Mh=8 | keep phone disabled |
| above 8 | profile again or all CUDA | 0 | remainder | unknown | fail closed |

The phone share falls rapidly as `Mh` grows because CUDA turns the head into a
more efficient GEMM while HTP work scales with `Mh`. Continuous batching does
not imply that every physical prompt row requests logits; select this policy
using `Mh`, not the graph's total `M`.

For dynamic fallback, retain the complete output head on CUDA and also keep
the maximum CPU and phone shards prepared. CUDA then selects the row prefix
for the current `Mh`. This duplicates about 61.5 MiB in desktop RAM and 90.6
MiB in the HTP mapping but avoids runtime repacking or model reads.

## Why the head is different

The head has all properties needed for heterogeneous overlap:

1. its 608.6 MiB Q6_K weight matrix is large enough to expose independent
   GPU, CPU, and HTP memory/compute resources;
2. all branches consume the same small final hidden vector;
3. output vocabulary rows are independent;
4. each branch can reduce locally and return only candidate IDs and scores;
5. there is one synchronization point per requested logit row, not one after
   every transformer operation.

The M=1 compact boundary is 10 KiB to the phone and 0.25 KiB back for top-32.
CPU needs the same host hidden-vector copy and no USB transfer.

## Placement of the remaining graph

| graph region | GPU plus CPU plus phone decision | reason |
| --- | --- | --- |
| embedding through final norm | CUDA | the exact model fits and CUDA owns the activations |
| Q/K/V and output projections | CUDA | Q4_K HTP is unavailable; complete CUDA operations are too short for a new boundary |
| FFN gate/up | CUDA | Q4_K blocks HTP; a CPU shard is unqualified |
| FFN down | CUDA | HTP Q6_K works, but it cannot start until CUDA produces SwiGLU output and its fixed RPC is near the complete CUDA down time |
| long-context attention | CUDA; HTP GQA group is shadow-only at `C>=8192` | prior proxy gained 9.6% at 8K, but physical 4060 timing and production RoPE/cache semantics are missing |
| vocabulary head | CUDA plus CPU; add HTP at `Mh=1` after validation | large independent rows and one compact merge |
| sampling | CPU | merge candidates or the three full-logit row ranges before sampling |

A fused CPU FFN island could theoretically take a small mixed Q4_K/Q6_K
intermediate slice, but it would add two CUDA/host fences in every layer. It
needs a physical one-layer test before it belongs in the policy. A phone FFN
is not currently available for the exact Qwen weights because HTP cannot run
the Q4_K gate and up matrices.

## Required implementation and test

The first bounded experiment should implement GPU plus CPU head sharding,
because it uses only qualified desktop kernels and can preserve full logits:

1. keep rows `[0,136576)` on the CUDA branch and rows `[136576,151936)` on
   the CPU branch;
2. publish the final F16 hidden vector to pinned host memory asynchronously;
3. run both Q6_K matmuls concurrently on fixed CPU and bridge cores;
4. concatenate full logits or perform matched local top-k in both control and
   treatment;
5. compare correctness, p50, p90, and full-token latency.

After that passes, add the HTP suffix at the M=1 row split above and use direct
DMA. The OP15 Q6_K `[7936,5120] x [7936,1]` shape already passed correctness,
but the proposed 14,848-row K=5,120 head and its recurring p90 remain
unmeasured. The phone is currently absent from ADB, so no reset or physical
three-way run was attempted.
