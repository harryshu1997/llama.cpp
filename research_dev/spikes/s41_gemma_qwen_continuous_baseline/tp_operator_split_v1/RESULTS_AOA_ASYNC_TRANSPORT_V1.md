# OP15 direct-AOA asynchronous transport

Date: 2026-08-02 EDT.

Verdict:
`REAL_AOA_ASYNC_BUFFERING_VALIDATED; SINGLE_DEPENDENT_DECODE_NO_WIN; CONTINUOUS_BATCH_TRANSPORT_UP_TO_53.46_PERCENT_FASTER; HOST_TO_PHONE_REMAINS_THE_BULK_BOTTLENECK; MODEL_OPERATOR_AND_ENERGY_NOT_RUN`.

## Question and implementation

This experiment asks whether reusable buffers, a dedicated phone reader and
writer, and asynchronous preposted USB transfers remove the direct-AOA
direction imbalance and reduce the phone operator boundary.

The bounded implementation has two paths:

- the Android daemon can run the original serial read/write loop or a reusable
  ring with separate reader and writer pthreads;
- the native libusb host can run blocking OUT then IN, or prepost IN before
  OUT and keep 1, 2, or 4 sequence-checked exchanges in flight.

Every request and response carries an exact sequence ID, byte count, and two
sentinels. The host refuses a short, reordered, stale, or malformed response.
The buffered path does not perform model computation; it isolates transport
and queue behavior at the exact wire sizes used by the existing model-operator
protocol.

The physical path was OP15 serial `3C15AU002CL00000` cabled directly at USB
5 Gbit/s to the RTX 4060 Ti host. Each model-shaped configuration used 50
warmups and 300 paid requests in three fresh processes. The 1 MiB direction
controls used 20 warmups and 100 paid requests. Configuration and workload
orders changed across repetitions. Every process used a fresh AOA USB
enumeration outside the timed interval.

## Correctness and completeness

All 90 planned runs and 90 phone workers completed. The raw captures contain
21,000 paid round trips. Online host validation passed every exact response,
and the independent validator rechecked the full experiment matrix, timing
array lengths, phone lifecycle logs, environment, and terminal state.

The phone finished in `ptp,adb` at SuperSpeed with no
`aoa_buffered_daemon` process. The source rebuilt byte-for-byte to the two
measured binaries:

- desktop host: `81720e541c9946b96a738d21a2854dcbc584e26542ef67ec27f8876ff5315a45`;
- Android daemon: `de8f9359f46e7792bd3729cd3179032b9b2505dbd485c68ec1ae843eabd170f4`.

This prototype intentionally did not gate temperature or lock phone clocks.
Alternating order and paired analysis reduce drift but do not eliminate it.
One depth-4 SwiGLU process was much faster than the other two and is retained
in the raw evidence; the median paired claim is determined by the other two.
No p99 or thermal-stability claim is made.

## Single dependent request

For a decode dependency, only one request is available. The comparison below
uses the median paired change against serial/synchronous inside each
repetition. A negative latency change is better.

| boundary | on-wire request/response | serial median | async depth 1 median | paired latency change | latency wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention state | 1,308 / 1,384 B | 0.225 ms | 0.177 ms | +13.03% | 1/3 |
| hidden, M=1 | 10,268 / 10,344 B | 0.256 ms | 0.293 ms | +14.04% | 0/3 |
| standalone SwiGLU | 69,660 / 34,920 B | 1.029 ms | 1.155 ms | +11.00% | 0/3 |
| hidden, M=8 | 81,948 / 82,024 B | 1.456 ms | 1.551 ms | +7.67% | 0/3 |
| 1 MiB host to phone | 1,048,604 / 104 B | 12.894 ms | 12.536 ms | -2.78% | 3/3 |
| 1 MiB phone to host | 64 / 1,048,680 B | 4.170 ms | 4.409 ms | +5.26% | 0/3 |

The attention unpaired medians appear favorable because the asynchronous run
was unusually fast in repetition 1. Paired analysis reverses that conclusion:
two of three repetitions are slower. The system must therefore keep the
serial AOA path for one dependency-bound decode request. Preposting alone is
not a valid single-token speedup.

## Multiple independent requests

Queuing helps sustained service because the phone reader and writer and the
two USB directions work on different request IDs. The table selects a bounded
depth for each shape. Throughput change is the median paired change versus
serial/synchronous. Queued latency is submission-to-response for an individual
request and includes its time behind other requests.

| boundary | selected depth | serial -> queued rate | paired throughput change | throughput wins | queued median | paired latency change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention state | 2 | 11.70 -> 15.60 MB/s | +33.33% | 2/3 | 0.334 ms | +45.89% |
| hidden, M=1 | 4 | 74.35 -> 115.62 MB/s | +53.46% | 3/3 | 0.691 ms | +174.79% |
| standalone SwiGLU | 4 | 96.79 -> 103.07 MB/s | +6.50% | 3/3 | 3.922 ms | +270.00% |
| hidden, M=8 | 2 | 108.76 -> 128.47 MB/s | +18.13% | 3/3 | 2.412 ms | +66.41% |
| 1 MiB host to phone | 2 | 79.72 -> 87.28 MB/s | +9.48% | 3/3 | 23.709 ms | +83.88% |
| 1 MiB phone to host | 2 | 242.96 -> 286.54 MB/s | +17.29% | 3/3 | 7.085 ms | +69.38% |

Depth 4 is useful for the 10 KiB boundary but is over-buffered for larger
payloads. At the 80 KiB and 1 MiB boundaries it provides no material
throughput beyond depth 2 and approximately doubles queued latency again.
SwiGLU depth 2 has a larger median throughput change but wins only two of
three pairs; depth 4 is shown because all three pairs improve.

These gains require independent requests, such as continuous batching across
sequences. They do not reduce the causal latency of one sequence.

## Direction and latency breakdown

The 1 MiB controls locate the remaining asymmetry:

| serial direction | response-ready | host OUT completion | post-OUT tail | phone read syscall | phone write syscall | aggregate rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| host to phone | 12.894 ms | 12.710 ms | 0.207 ms | 12.993 ms | 0.013 ms | 79.72 MB/s |
| phone to host | 4.170 ms | 0.126 ms | 4.025 ms | 0.614 ms | 3.558 ms | 242.96 MB/s |

The phone-to-host direction is about 3.05 times faster in this synchronous
control. At depth 2 it reaches 286.54 MB/s, but host-to-phone reaches only
87.28 MB/s. The large request spends almost all its time in host OUT and the
phone blocking read. More host-side IN requests cannot fix that path. The
behavior is consistent with the small receive queue and 16 KiB buffers in
common Android `f_accessory` implementations, but those exact kernel constants
were not verified on this OP15 build.

The model-shape serial breakdown is also directional:

| boundary | response-ready | host OUT completion | post-OUT tail |
| --- | ---: | ---: | ---: |
| attention state | 0.225 ms | 0.040 ms | 0.184 ms |
| hidden, M=1 | 0.256 ms | 0.062 ms | 0.193 ms |
| standalone SwiGLU | 1.029 ms | 0.792 ms | 0.230 ms |
| hidden, M=8 | 1.456 ms | 1.002 ms | 0.445 ms |

Compact attention is return/wakeup dominated. SwiGLU and batched hidden
states are mostly host-to-phone transfer dominated. This is why buffering can
raise batch throughput while failing to improve a single response.

## Lifecycle defect found and fixed

Keeping the phone in accessory mode while repeatedly closing and reopening
both endpoint owners produced a stale session after five valid cases. The
next transfer completed with zero bytes, and a diagnostic retry received a
stale response. A clean USB re-enumeration immediately made the exact same
10,268/10,344-byte case pass.

The runner now waits for a stable normal USB identity before every fresh AOA
process and restores normal mode afterward. This is outside all timed
intervals. A production service should instead keep one long-lived endpoint
owner and use the same reset rule only for reconnect recovery.

## System decision

- Keep serial direct AOA for one causal decode request.
- For continuous batching, use payload-aware credits: depth 4 near the 10 KiB
  boundary and depth 2 for the larger tested boundaries.
- Do not claim that asynchronous buffering fixes AOA upload bandwidth. Large
  host-to-phone staging should still prefer NCM, whose earlier exploratory
  stream reached about 326 MB/s, when its 1.75 ms RPC floor can be amortized.
- The next narrow gate is a real three-stage phone operator pipeline:
  reader ring -> one HTP compute owner -> writer ring, with independent
  request IDs from the server. It must beat the serial real-operator path and
  pass changing-input correctness before energy is measured.

This experiment does not run HTP, OpenCL, CUDA compute, a complete layer, a
full model, BurstGPT, power, or energy. The maximum +53.46% result is transport
throughput at one model-shaped boundary, not inference throughput.

## Evidence

Evidence root:
`results/aoa_async_transport_v1/run_20260802T213651Z/`.

Key files are `ANALYSIS.json`, `VALIDATION.json`, `ENVIRONMENT.txt`, all 90
raw JSON captures, all 90 phone logs, and the SHA-256 manifests. The bounded
implementation is in `aoa_async_transport_v1/`.

Both C/C++ programs compile with `-Wall -Wextra -Werror`; all three shell
drivers pass `sh -n`; both Python reducers compile; the independent campaign
validator reports `PASS`; the source rebuilds match the deployed binaries;
and both evidence manifests verify. No commit or push was performed.
