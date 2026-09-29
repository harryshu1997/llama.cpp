# OP15 OpenCL queue controls

Date: 2026-08-02 EDT.

Verdict:
`EARLY_FLUSH_NO_MATERIAL_GAIN; PERSISTENT_SVM_DOORBELL_REMOVES_RECURRING_DUMMY_DISPATCH; NOT_A_REAL_OPERATOR_OR_ENERGY_RESULT`.

## Question

The previous dummy trace showed one Adreno SQR spending about 0.5 ms before
29 us of arithmetic. This experiment tested two bounded responses:

- force `clFlush()` when GGML finishes enqueuing a graph;
- keep one OpenCL workgroup resident and submit requests through fine-grained
  SVM atomics.

The physical path was the OP15 Adreno 840 attached directly to the RTX 4060 Ti
desktop over 5 Gbit/s USB 3.0. Every request sent and returned 2,816 FP16
elements over AOA and computed one F32 SQR on the phone.

## Primary result

Each row is the median of three alternating runs. Each run used 50 warmups and
600 measured requests.

| path | E2E median | E2E p90 | backend submit + completion | result |
| --- | ---: | ---: | ---: | --- |
| rebuilt OpenCL control | 1.172 ms | 1.431 ms | 0.844 ms | baseline |
| early `clFlush()` | 1.167 ms | 1.458 ms | 0.842 ms | 0.4% E2E reduction |
| matched doorbell kernel, dispatched/request | 1.241 ms | 1.480 ms | 0.907 ms | geometry control |
| persistent SVM doorbell | **0.266 ms** | **0.280 ms** | **0.00682 ms** | **77.3% below OpenCL control** |

Early flush did not remove queueing. It moved time from the subsequent fence
into `graph_compute`: control submit/sync medians were 0.050/0.794 ms, while
early flush changed them to 0.193/0.648 ms. Their sums remained 0.844 and
0.842 ms. The 0.4% E2E difference is not a meaningful win relative to run
variation.

Persistent dispatch reduced E2E by 77.3% (4.40x) versus the rebuilt OpenCL
control and reduced the recorded backend interval by 99.2% (123.7x). Against
the exact same doorbell workgroup relaunched for every request, persistent
dispatch reduced E2E by 78.5% and the backend interval by 99.25%.

The host observed the persistent doorbell, one SQR, and completion publication
in 6.6 us median. Input and output SVM copies were about 2.3 and 2.7 us. The
remaining E2E floor is AOA, FP16 conversion, CRC, and host wakeup.

## Native queue evidence

The profiling control reproduces the original queue attribution:

| stock GGML kernel | queued -> submitted | submitted -> start | device execution |
| --- | ---: | ---: | ---: |
| control | 94.4 us | 400.3 us | 28.9 us |
| early flush | 90.6 us | 406.2 us | 29.2 us |

The flush does not shorten either native queue interval.

The matched doorbell kernel, when relaunched for every request, reports 100.4
us queued -> submitted and 400.7 us submitted -> start. Its 57.9 us event
duration includes the SQR plus observing the host stop doorbell and exiting;
it is not a pure arithmetic time. This matched control confirms that the
persistent result comes from amortizing the recurring OpenCL/GMU launch, not
only from changing the SQR work geometry.

The persistent kernel itself paid a larger one-time launch: 9.7-13.6 ms in
the OpenCL queued interval plus 0.53-0.81 ms to start. It must therefore be
prepared before traffic and amortized over a serving session.

## Why HTP does not show the same long queue

HTP is not queue-free. The earlier native trace measured a 91 us median HTP
batch envelope around 3 us of SQR arithmetic, leaving about 88 us of FastRPC,
DSP queue, and batch overhead.

Its path is shorter for structural reasons visible in the backend:

- one FastRPC CDSP session and exported `dspqueue` are created once;
- shared `rpcmem` buffers remain mapped into that session;
- up to 256 HTP operations are packed into one queue packet;
- FastRPC latency QoS is enabled for the session;
- `graph_compute_async` writes the packet and calls `sess->flush()`, so the
  completion wait is already charged to graph submission.

The Adreno path instead constructs OpenCL commands and crosses the graphics
driver, KGSL, and GMU scheduling path for every dispatch. Qualcomm's internal
firmware split is not observable, but the OpenCL event timestamps establish
that about 400 us occurs after driver submission and before device execution.
HTP is a dedicated compute session rather than a graphics command queue, so
it avoids that long per-kernel scheduling path.

## Correctness and limits

The primary controls contain 7,200 exact semantic matches. The two profiling
runs add 1,200 matches. Output bytes were checked against an independently
computed F32 repeated-SQR oracle, not only against their transmitted CRC.

The persistent kernel remained resident for about 3.1-3.2 seconds per final
run. It completed all requests, returned to normal USB mode, left no worker,
and produced no matched KGSL/GPU fault, hang, reset, or watchdog record.

This remains a dummy result:

- one workgroup and one elementwise SQR are not a model operator;
- a real matmul or attention shard needs multi-workgroup scheduling,
  reductions, resident weights, and a bounded command format;
- continuously polling the GPU may consume substantial phone power;
- phone energy was not measured, so no energy-saving claim follows;
- DVFS was uncontrolled and the thermal gate was intentionally omitted;
- the mechanism requires fine-grained SVM buffers and SVM atomics, both of
  which the OP15 reports but other phones may not support;
- this does not establish concurrency with HTP or desktop CUDA.

The next bounded test should replace SQR with one real fused, memory-bound
candidate while retaining the matched per-request/persistent controls. Phone
power must be measured in the same acquisition before integrating the path
into the system scheduler.

## Evidence

Evidence root:
`results/backend_opencl_queue_v1/run_20260802T_opencl_queue_v1/`.

`ANALYSIS.json` is reproduced by `analyze_opencl_queue_v1.py`. Raw JSON keeps
all samples; native CSV/trace files and matched-dispatch logs preserve the
OpenCL timestamps. The exact Android executables and source/runtime hashes are
also stored under the evidence root.
