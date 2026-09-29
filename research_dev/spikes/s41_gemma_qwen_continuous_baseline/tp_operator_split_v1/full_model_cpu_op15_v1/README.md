# Gemma 4 12B CPU and OP15 full-model comparison

This experiment compares two real full-model decode routes on the physical
i9-12900K desktop and OnePlus 15 (OP15):

1. `CPU_ONLY`: all 48 transformer layers and the vocabulary head run on the
   desktop CPU.
2. `CPU_OP15_PREFIX_K`: OP15 owns transformer layers `[0,K)` and the desktop
   CPU owns `[K,48)` plus the vocabulary head.

The all-phone route is deferred until another phone is available.

## Frozen model and shape

- Model: Gemma 4 12B IT Q8_0, SHA-256 prefix `7b56cbd0`.
- Model structure: 48 layers, hidden width 3840, FFN width 15360.
- Host: i9-12900K, 30 GiB RAM, CPU-only llama.cpp build.
- Phone: OP15, Snapdragon 8 Elite Gen 5, resident prefix weights.
- Prompt: `Explain why arithmetic intensity matters for offloading a matrix multiplication.`
- Chat template: enabled.
- Decode: greedy, 32 requested output tokens, one warmup, five paid requests.
- Host threads: 8 decode and 8 batch threads, selected by a physical thread
  sweep.
- Context: 256 tokens; maximum prefill: 128 tokens.

The CPU build has no CUDA backend. The RTX 4060 Ti identifies the target
desktop but is not an inference backend in either route.

## Boundary cost

Gemma 4 12B has a 3840-element hidden state. A phone prefix receives token IDs
and returns one FP32 cut activation:

```text
decode response = 3840 * 4 = 15,360 bytes per token
prefill response = prompt_tokens * 15,360 bytes
```

Weights and KV state stay resident on their owning device. The experiment
uses one persistent stage connection, so model loading and connection setup
are outside paid request latency.

## Selection rule

The prefix depth and phone backend are calibrated before the paid comparison.
The eligible treatment minimizes median full-request time while satisfying:

- the phone placement certificate reports the requested backend;
- the host placement certificate reports CPU only;
- all requests finish and return the requested token count;
- no process or system swap growth occurs during a run;
- generated token agreement with the CPU control is reported, not assumed.

Q8_0 with GPUOpenCL is the matched-quantization candidate. An HTP candidate
may use an existing F16 prefix shard and is reported separately because that
changes numerical precision at the cut.

## Physical result

The 2026-08-02 physical run selected GPUOpenCL K28 by measured latency. It
reached a 1.268x steady-state speedup over CPU-only execution. Android zram
showed nonzero page activity, so the result report distinguishes the measured
performance winner from a literal zero phone swap-I/O gate.

See [results/physical_20260802/RESULTS.md](results/physical_20260802/RESULTS.md).

## Intra-layer FFN overlap prototype

The `overlapdriver` route replaces the sequential layer-prefix boundary with
one decode-time FFN tensor-parallel boundary. For a selected dense layer, the
desktop computes attention and FFN normalization, then splits the intermediate
width:

```text
normalized hidden x
        |
        +-- desktop: gate/up/down columns [0, 13568) ---------+
        |                                                     +-- sum -> post-FFN norm -> residual
        +-- OP15: gate/up/down columns [13568, 15360) --------+
```

The two down projections both produce a 3840-element raw FFN residual. They
must be summed before Gemma's post-FFN RMSNorm and residual addition. Moving a
whole layer would serialize the devices and does not exercise this overlap.

The selected suffix is 1,792 intermediate columns from layer 24. Its three
Q8_0 weight slices occupy 20.92 MiB. Weights and the backend graph stay
resident in `llama-ffn-split-worker`; each decode call transfers one normalized
hidden state and returns one partial residual. F16 I/O is 7,680 payload bytes
in each direction, versus 15,360 bytes in each direction for F32. Prefill is
left on the desktop and only single-token decode graphs are split.

### Desktop loopback gate

The loopback worker uses the real layer-24 tensors from the frozen Gemma 4 12B
GGUF. This is a correctness and concurrency gate, not a phone performance
result. The worker competes with inference on the same i9 CPU.

| gate | result |
| --- | ---: |
| 32-token F16 sequence vs frozen CPU control | 32/32 token positions match |
| distinct overlapped decode calls | 31 |
| F16 RPC p50 | 4.074 ms |
| loopback worker compute p50 | 3.669 ms |
| desktop FFN-prefix branch p50 | 6.579 ms |
| merge wait p50 | 0.000747 ms |

An immediate short same-load A/B measured F32 RPC at 2.514 ms and F16 RPC at
2.395 ms. Both had a sub-microsecond median merge wait and identical token
IDs. End-to-end decode stayed flat, as expected: only one of 48 layers was
split and the loopback worker consumed the same CPU resources as the host.

The Android arm64 worker builds successfully with both Hexagon and Adreno
backends. A speed claim remains gated on the physical OP15 AOA run. That run
must use `S41_DISABLE_GRAPH_CACHE=1` for HTP changing-input correctness and
must compare alternating all-CPU and overlap requests with the same prompt,
thread count, and thermal state.

### Multi-layer resident gate

The resident protocol now accepts a layer mask and keeps one graph and one
set of suffix weights per selected layer behind a single connection. An
all-layer desktop gate selected layers 0 through 47 and ran two decode steps:

| gate | result |
| --- | ---: |
| selected dense layers | 48 |
| completed split calls | 96/96 |
| generated token agreement | 3/3 |
| resident worker weights | 1004.06 MiB |
| F16 RPC p50 | 3.230 ms |
| worker compute p50 | 3.064 ms |
| desktop branch p50 | 6.473 ms |
| merge wait p50 | 0.000206 ms |

This proves that the boundary is operator-level at every selected layer. It
does not imply that all 48 suffixes should reside on one phone; the planner
must still enforce the phone memory budget.

## Vocabulary-row LM-head overlap

The LM-head route assigns the final 46,080 vocabulary rows to a resident
worker. The desktop concurrently evaluates rows `[0,216064)`. The worker
returns only its top 32 approximate candidates, and the desktop gathers those
rows from its original tied output matrix and recomputes their scores before
sampling. Non-candidate suffix rows are set to negative infinity.

For two eight-token requests, all 14 decode calls completed and both sequences
matched the unsplit CPU control at all 16 positions. The worker held 179.30
MiB of Q8_0 weights. Median times were 24.418 ms RPC, 23.850 ms worker matmul,
0.117 ms worker top-k reduction, 35.949 ms desktop prefix matmul, 0.00129 ms
merge wait, and 0.600 ms exact candidate rescore. The largest observed
worker-versus-desktop candidate score difference was 0.117 before Gemma's
final logit softcap.

The same run with four FFN layers enabled completed 56 FFN calls and 14
LM-head calls, again with exact token agreement. Same-CPU loopback decode was
flat to slightly slower than the CPU control, so this is a concurrency and
correctness result rather than a speed result.

## Whole-expert MoE overlap

A separate worker implements one complete routed-expert branch for Gemma 4
26B A4B. It owns the router, softmax and top-8 selection, all 128 experts,
weighted reduction, and expert post-norm. The desktop computes the shared
dense MLP from the same `attn_out` concurrently and sums the two post-norm
branches at the original graph boundary.

The real BF16 checkpoint has 30 layers, hidden width 2816, shared FFN width
2112, expert width 704, 128 experts, and top-8 routing. One resident layer
uses 1453.41 MiB of worker weight storage. The 47.02 GiB full checkpoint was
tested on the local 125 GiB Threadripper workstation because the i9 target's
30 GiB RAM cannot hold it.

| gate | F32 I/O | F16 I/O |
| --- | ---: | ---: |
| completed split calls | 7/7 | 7/7 |
| generated token agreement | 8/8 | 8/8 |
| RPC p50 | 2.341 ms | 2.355 ms |
| routed branch compute p50 | 2.071 ms | 2.199 ms |
| shared MLP branch p50 | 0.641 ms | 0.676 ms |
| merge wait p50 | 1.904 ms | 1.823 ms |

This shape is correct but is not hidden by the shared branch on that CPU: the
routed branch is longer than the available 0.6 to 0.7 ms overlap window. A
phone placement is eligible only if its routed compute plus AOA transport is
competitive with that window. Runtime HTP/OpenCL support and physical timing
remain deferred until OP15 is available; all three workers currently pass the
Android arm64 HTP/OpenCL cross-build.
