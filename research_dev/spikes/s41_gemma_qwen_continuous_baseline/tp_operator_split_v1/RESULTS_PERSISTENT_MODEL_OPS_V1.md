# OP15 persistent Qwen3-14B operator controls

Date: 2026-08-02 EDT.

Verdict:
`PERSISTENT_OPENCL_REMOVES_RELAUNCH_FOR_SMALL_FUSED_OPS; HTP_WINS_ATTENTION; TRANSPORT_AND_KERNEL_QUALITY_DOMINATE; NOT_A_FULL_MODEL_OR_ENERGY_RESULT`.

## Question and scope

This experiment replaces the earlier SQR dummy with three Qwen3-14B-shaped
operator controls on the real OP15 attached directly to the RTX 4060 Ti host
over 5 Gbit/s USB 3.0:

| operator | shape | wire input | wire output | resident state |
| --- | --- | ---: | ---: | --- |
| RMSNorm plus scale | hidden=5,120 | 10,240 B | 10,240 B | 5,120 f16 scales |
| SwiGLU | intermediate=17,408 | 69,632 B | 34,816 B | none |
| one GQA attention core | 5 Q heads, 1 KV head, d=128, KV=8,192 | 1,280 B | 1,280 B | 4 MiB f16 K/V |

The shapes are from Qwen3-14B. Inputs, normalization weights, and KV state are
deterministic synthetic values, not model artifact tensors. The attention
control begins after Q/K/V projection and ends before output projection. It
does not include Q/K normalization, RoPE, a mask, or a cache update.

Four paths use the same AOA protocol and changing inputs:

- `HTPGraph`: GGML operators through the already resident FastRPC/dspqueue
  executor. HTP keeps a command processor and worker pool resident, but does
  not keep one arithmetic kernel spinning.
- `OpenCLGraph`: the normal optimized GGML OpenCL operator path.
- `OpenCLDispatch`: the custom OpenCL operator loop relaunched for every
  request.
- `OpenCLPersistent`: the exact custom loop remains resident and receives
  requests through fine-grained SVM atomics.

The persistent OpenCL program has five 128-work-item groups and three
precompiled opcodes. RMS normalization and scaling are fused. SwiGLU is fused.
Attention keeps K/V and score scratch resident, maps one group to each query
head, uses half4 QK dot products, and enables relaxed math. This is a bounded
operator executor, not arbitrary runtime GPU code.

Each row below is the median of three alternating processes. Each process has
50 warmups followed by 300 measured requests cycling through four inputs.
Thermal gating was intentionally omitted at the user's direction, and DVFS
was not controlled.

## Primary result

`Response ready` ends when the complete response is in host memory. It
includes host packing, USB OUT, phone work, and USB IN. It excludes the
subsequent Python full-vector oracle comparison. The raw `e2e_ms` field keeps
that expensive validation for auditability.

| operator | path | response ready median | response ready p90 | backend interval |
| --- | --- | ---: | ---: | ---: |
| RMSNorm | HTP graph | 0.414 ms | 0.565 ms | 0.102 ms |
| RMSNorm | OpenCL graph | 0.958 ms | 1.638 ms | 0.629 ms |
| RMSNorm | matched dispatch | 1.930 ms | 2.303 ms | 1.365 ms |
| RMSNorm | **persistent OpenCL** | **0.300 ms** | **0.367 ms** | **0.0785 ms** |
| SwiGLU | **HTP graph** | **0.943 ms** | 1.232 ms | 0.106 ms |
| SwiGLU | OpenCL graph | 1.415 ms | 1.529 ms | 0.555 ms |
| SwiGLU | matched dispatch | 2.536 ms | 3.053 ms | 1.041 ms |
| SwiGLU | persistent OpenCL | 1.061 ms | **1.230 ms** | **0.0213 ms** |
| attention | **HTP graph** | **0.459 ms** | **0.540 ms** | **0.310 ms** |
| attention | OpenCL graph | 5.336 ms | 5.958 ms | 4.928 ms |
| attention | matched dispatch | 9.697 ms | 10.927 ms | 9.286 ms |
| attention | persistent OpenCL | 7.198 ms | 7.278 ms | 6.825 ms |

Against the exact relaunched custom kernel, persistence reduces response-ready
latency by 84.45% for RMSNorm, 58.16% for SwiGLU, and 25.76% for attention.
The corresponding backend-interval reductions are 94.25%, 97.95%, and
26.50%.

This does not mean persistent OpenCL is always the best backend. It is 27.54%
faster than HTP end to end for RMSNorm, but 12.53% slower for SwiGLU. For
attention it is 34.90% slower than the optimized ordinary OpenCL graph and
14.68x slower than HTP end to end.

## Where the time goes

The fastest paths illustrate three different limits. Cells are medians of
run medians and are not expected to add exactly because each stage has its own
distribution.

| operator/path | USB OUT | validate/decode/set | backend | get/hash | response residual | response ready |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RMSNorm persistent OpenCL | 0.041 ms | 0.028 ms | 0.078 ms | 0.027 ms | 0.122 ms | 0.300 ms |
| SwiGLU HTP | 0.454 ms | 0.163 ms | 0.106 ms | 0.089 ms | 0.143 ms | 0.943 ms |
| SwiGLU persistent OpenCL | 0.578 ms | 0.172 ms | 0.021 ms | 0.088 ms | 0.189 ms | 1.061 ms |
| attention HTP | 0.051 ms | 0.004 ms | 0.310 ms | 0.005 ms | 0.066 ms | 0.459 ms |

RMSNorm is launch-bound, so the resident doorbell helps. SwiGLU arithmetic is
only 21 us in the persistent kernel, but its 102 KiB bidirectional activation
boundary dominates. Offloading SwiGLU by itself is therefore not a useful
system route. It only becomes plausible when fused with resident gate/up and
down projections so the wire sees a small hidden vector instead.

Attention is the opposite case. The boundary and transport are compact, but
the custom persistent kernel is poor. It materializes scores, makes separate
QK, exponential, and probability-times-V passes, and rereads K for five query
heads. Persistence removes about 2.46 ms from the matched direct path, but its
6.83 ms arithmetic remains slower than GGML OpenCL flash attention at 4.93 ms
and HTP flash attention at 0.31 ms. The scheduler should choose HTP here; it
should not choose persistence merely because the mechanism exists.

The matched dispatched OpenCL event medians further separate queueing and
work:

| operator | queued to submit | submitted to start | event duration |
| --- | ---: | ---: | ---: |
| RMSNorm | 111 us | 440 us | 395 us |
| SwiGLU | 96 us | 439 us | 141 us |
| attention | 49 us | 273 us | 8,785 us |

Event duration includes observing the host stop command and kernel exit; it
is not a pure arithmetic timer. The persistent program pays 6.0-13.6 ms once
at launch, then remains resident for the serving interval. Its active polling
cost is not measured.

## Correctness and stability

All 10,800 measured responses passed the independent CPU oracle. Every
backend reproduced its four output CRCs across three fresh processes. The
matched direct-dispatch and persistent paths were byte-exact for every input.
Maximum relative L2 error was:

- RMSNorm: 0.00000966 on all paths;
- SwiGLU: 0 on all paths after f16 publication;
- attention: 0.004783 on HTP, 0.0000294 on GGML OpenCL, and 0.0000775 on the
  two direct OpenCL paths.

Two setup failures are excluded from the 36 acquisitions:

- one cold ordinary-OpenCL process exceeded the original 45-second readiness
  window while loading kernels;
- later, the first unmeasured OpenCLGraph SwiGLU request hung twice, including
  once with a 60-second USB timeout. No matching KGSL fault, hang, or reset was
  logged. No result JSON was published. One OP15 reboot restored the path and
  the missing acquisitions then completed.

The reboot was recovery from a reproducible OpenCL context failure, not a
thermal policy. Pre- and post-reboot latency differs, especially on the large
SwiGLU transfer, so the table uses the predeclared median of three processes
and the raw run medians remain in `ANALYSIS.json`. This device-state
sensitivity is itself a system concern for a persistent-phone design.

## System consequence

The useful rule is now narrower:

1. Use a resident GPU command loop only for a precompiled fused island whose
   optimized persistent kernel is already competitive and whose payload is
   compact.
2. Use HTP's resident command executor for supported FFN/expert and attention
   kernels; its queue is already short and its optimized attention kernel wins
   decisively here.
3. Do not offload RMSNorm or SwiGLU alone. Their server work is too small or
   their activation boundary is too large.
4. A future persistent GPU attention path must reuse a tiled FlashAttention or
   xgemm-quality kernel, share K across GQA heads, and use online softmax rather
   than the current materialized-score implementation. Dispatch work alone
   cannot close the measured gap.

The most useful next integration remains an operator with resident large
state, a small input, and a compact reduction, such as a sharded vocabulary
head returning local top-k. For current Qwen attention at KV=8,192, integrate
the measured HTP path first rather than expanding this experimental GPU
executor.

No CUDA overlap, complete layer, full model, BurstGPT, phone power, total
energy, or server-energy result is established by this acquisition.

## Evidence

Evidence root:
`results/persistent_model_ops_v1/run_20260802T_model_ops_v1/`.

`ANALYSIS.json` is reproduced by
`persistent_model_ops_v1/analyze_campaign.py`. Raw per-request JSON, worker
logs, direct-dispatch event traces, four-input references, binaries, source
hashes, and runtime hashes are retained under the evidence root.

The evidence manifest verifies with SHA256
`b7c1a3742f02029bfe732317ade51fc7476c6d43d58129656e7847be48df273f`.
The analysis digest is
`7331038a12609eb25cc4a851184a18652544bffa75567b6fbec1ad4efed00506`.
`ORCHESTRATION_NOTES.txt` records the setup-only timeout and resume changes;
the final source snapshot must not be read as an archive of every earlier
setup-only runner revision.
