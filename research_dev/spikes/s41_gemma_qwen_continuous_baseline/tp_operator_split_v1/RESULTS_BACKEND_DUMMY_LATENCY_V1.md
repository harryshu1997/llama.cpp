# OP15 NPU and GPU dummy-kernel latency breakdown

Date: 2026-08-02 EDT.

Verdict:
`HTP_LOWER_DUMMY_GRAPH_LATENCY; FIXED_PATH_AND_BACKEND_QUEUES_DOMINATE; NOT_AN_OPERATOR_OR_MODEL_RESULT`.

## What was measured

The same Android worker and AOA protocol ran on the real rooted OP15 attached
directly to the RTX 4060 Ti desktop at an enumerated 5,000 Mbit/s. Each
request sent and returned 2,816 FP16 elements, the hidden-width boundary of
the synthetic Gemma-4-26B-A4B expert probe. The wire sizes were 5,656 bytes
out and 5,728 bytes back, including headers and phone timing records.

The controls were deliberately model-free:

- `noop`: no compute node, followed by an explicit backend fence;
- `sqr_1`: one real elementwise SQR kernel;
- `sqr_8`: eight dependent real SQR kernels in one graph.

These are dummy operators, not sleeps or mocked timings. The primary numbers
use profiling-disabled libraries. Separate profiling-enabled processes
attribute native HTP and OpenCL time. Every configuration used 50 warmups and
600 measured continuous requests.

`HTPFast` and `OpenCLFast` use the same worker executable and the same
profiling-disabled GGML library set; only the selected backend changes.
`HTP` and `OpenCL` are attribution controls with their native profilers
enabled and are not used for the primary E2E comparison.

## Primary unprofiled result

| backend | graph | E2E median | E2E p90 | submit + sync median |
| --- | --- | ---: | ---: | ---: |
| HTP v81 | no-op | 0.256 ms | 0.284 ms | 0.001 ms |
| Adreno 840 OpenCL | no-op | 0.274 ms | 0.329 ms | 0.004 ms |
| HTP v81 | 1 x SQR | 0.400 ms | 0.418 ms | 0.089 ms |
| Adreno 840 OpenCL | 1 x SQR | 0.840 ms | 1.034 ms | 0.560 ms |
| HTP v81 | 8 x SQR | 0.431 ms | 0.533 ms | 0.169 ms |
| Adreno 840 OpenCL | 8 x SQR | 1.294 ms | 1.501 ms | 1.017 ms |

For one dummy kernel, HTP is 2.10x faster end to end and its backend interval
is 6.31x shorter. For eight kernels, HTP is 3.00x faster end to end and its
backend interval is 6.02x shorter. HTP adds 0.080 ms going from one to eight
operations; OpenCL adds 0.457 ms.

The no-op result is the important floor: even without arithmetic, this
5.6-KiB round trip costs about 0.26-0.27 ms. A candidate operator must remove
more server work than the complete phone deadline, not merely beat the phone
kernel time.

## Matched stage medians

| stage | HTP 1 op | OpenCL 1 op | HTP 8 ops | OpenCL 8 ops |
| --- | ---: | ---: | ---: | ---: |
| host pack | 0.004 ms | 0.004 ms | 0.004 ms | 0.004 ms |
| USB out | 0.063 | 0.062 | 0.064 | 0.064 |
| phone CRC + FP16 decode | 0.027 | 0.027 | 0.027 | 0.027 |
| backend tensor set | 0.003 | 0.008 | 0.002 | 0.008 |
| graph submit | 0.088 | 0.045 | 0.168 | 0.343 |
| graph synchronize | 0.000 | 0.515 | 0.000 | 0.674 |
| backend tensor get | 0.003 | 0.007 | 0.003 | 0.008 |
| phone FP16 encode + CRC | 0.029 | 0.029 | 0.029 | 0.029 |
| response write, USB in, host wake | 0.114 | 0.122 | 0.115 | 0.118 |
| host validate | 0.010 | 0.010 | 0.010 | 0.010 |
| measured E2E | **0.400** | **0.840** | **0.431** | **1.294** |

Each stage cell is its own distribution median, so the cells are not expected
to add exactly to the E2E median. Raw per-request samples are preserved.

## Native backend attribution

HTP profiling reports the following medians after discarding the same 50
warmups:

| graph | HTP batch envelope | native operations | native fraction |
| --- | ---: | ---: | ---: |
| 1 x SQR | 91 us | 3 us | 3.3% |
| 8 x SQR | 129 us | 16 us | 12.4% |

The NPU arithmetic is only a few microseconds. FastRPC, descriptor queueing,
and the HTP batch envelope are the target. The eight operations still share
one flush, which is why their marginal cost is small.

The OpenCL event timeline agrees with the unprofiled worker interval:

- One SQR spends 96.6 us from queued to submitted, 404.8 us waiting to start,
  and only 29.4 us executing. First queue to device completion is 532 us,
  close to the unprofiled 560 us submit-plus-sync interval.
- For eight SQRs, host enqueue spans 358 us, summed kernel execution is only
  118.6 us, and first queue to final device completion is 1,022 us. The
  unprofiled submit-plus-sync interval is 1,017 us.

Therefore the OpenCL result is not a slow SQR kernel. It is queue construction,
driver/GMU scheduling, and a fence per graph. Spin polling cannot remove this
because the dominant interval is before the GPU starts.

## Design consequence

Use HTP for a compact, multi-operation island such as a fused FFN/expert path
when its supported kernels are fast enough. Optimize one FastRPC graph and
one compact return, not isolated arithmetic instructions.

Use the phone GPU only when the work can be expressed as one large or fused
kernel with enough arithmetic or memory traffic to amortize roughly a
0.5-ms warm queue floor. Eight tiny OpenCL kernels are the wrong geometry.
Long-context attention or a large dense projection can still qualify, but it
must be measured with its real kernel.

The next bounded experiment is to retain this exact instrumentation while
replacing SQR with one real fused candidate at a time: FFN/expert on HTP,
local-reduction vocabulary head on HTP, and one fused long-context attention
shard on OpenCL. Only then should each phone deadline be overlapped against
the physical 4060 Ti CUDA control.

## Limits and verification

This is one OP15, one acquisition per configuration, continuous warm traffic,
uncontrolled phone DVFS, and no thermal gate by user request. Payload CRCs
passed for all 7,800 requests. All 12 workers reached
`complete requests=650`; 1,300 HTP batches and 5,850 OpenCL operations were
captured in the profiling controls. The Android build, Python syntax, shell
syntax, JSON structure, and raw record counts pass.

No model weights or matmuls were used. No operator, complete-layer,
full-model, changing-input correctness, energy, multi-phone, CUDA overlap, or
BurstGPT claim follows. The phone was restored to normal `22d9:2772` USB mode,
and no dummy worker remains.

The AOA transition sometimes exposed either normal-mode PID `22d9:2769` or
`22d9:2772` and occasionally needed a retry after reset. The runner now tries
both PIDs, truncates each worker log before testing readiness, removes only
the two case-local OpenCL profile outputs before a profiled run, and retries
reset until adb returns. This avoids accepting a stale ready record.

## Reproduction entry points

- Phone worker: `backend_dummy_worker.cpp`
- Wire layout: `backend_dummy_protocol.h`
- Host acquisition and raw JSON: `backend_dummy_bench.py`
- One-case AOA lifecycle: `run_backend_dummy_case.sh`

Use `HTPFast` and `OpenCLFast` for primary timing, then `HTP` and `OpenCL`
for attribution. The Android binary and every runtime/source digest used in
this acquisition are preserved under the evidence root.

Evidence root:
`results/backend_dummy_latency_v1/run_20260802T035814Z/`.
The machine-readable summary is `ANALYSIS.json`.
