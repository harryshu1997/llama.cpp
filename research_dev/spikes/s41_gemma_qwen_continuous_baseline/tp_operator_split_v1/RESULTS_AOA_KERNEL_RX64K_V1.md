# OP15 AOA 64 KiB kernel receive buffer

Date: 2026-08-02 EDT.

Verdict:
`REAL_KERNEL_RX64K_VALIDATED; ONE_MIB_UPLOAD_LATENCY_51.28_PERCENT_LOWER; SWIGLU_TRANSPORT_35.78_PERCENT_LOWER; HIDDEN_M8_TRANSPORT_36.37_PERCENT_LOWER; PHONE_TO_HOST_UNCHANGED; RX_RING_MODEL_AND_ENERGY_NOT_RUN`.

## Question and change

This experiment tests whether the Android Accessory receive request size is
the cause of the slow host-to-phone direction. The official OP15 kernel source
uses 16 KiB buffers for both directions. Its read path allocates two receive
requests but `acc_read()` submits only `rx_req[0]`, waits for it, copies the
result, and repeats.

The bounded patch separates the constants and changes only the receive side:

- control: 16 KiB receive requests and 16 KiB transmit requests;
- treatment: 64 KiB receive requests and 16 KiB transmit requests.

No request ring, AOA protocol, host code, phone daemon, or transmit size was
changed. The patch is in
`aoa_kernel_rx_v1/android_f_accessory_rx64k.patch`.

Both kernels were built from OnePlus common commit
`227664cbe007bbad49aa74259179ac99608a2113` with byte-identical GKI configs
and the same build environment. The unmodified control and the treatment were
temporarily booted with `fastboot boot`; no partition was flashed. The full
OnePlus wrapper could not build because the published source omits the
unrelated `oplus_ex_gpio` package, so the official isolated GKI targets were
used:

```
tools/bazel build //common:kernel_aarch64
tools/bazel build //common:kernel_aarch64_gki_artifacts
```

The physical path was OP15 serial `3C15AU002CL00000` directly connected at
USB 5 Gbit/s to the RTX 4060 Ti desktop. Every result below is real transport
on that path. The phone daemon validates the request size and sequence, and
the host validates every response sequence, size, and sentinel.

## Measurement protocol

Each variant used three fresh AOA processes per workload. The 1 MiB direction
controls used 20 warmups and 100 paid requests per process. Model-shaped
boundaries used 50 warmups and 300 paid requests per process. The independent
validator checked all 36 JSON files, all 36 worker logs, all timing array
lengths, stored percentiles, exact workload sizes, successful lifecycle
records, and both kernel traces. In total, 8,400 paid requests passed.

Kernel boots were grouped rather than randomized for every repetition. The
large effects below win all three aligned repetitions, but the small-boundary
results should not be interpreted as thermal or tail-latency claims. Phone
temperature and clocks were not gated for this prototype.

## Latency results

The first change column compares the median of three fresh-process medians.
The aligned column first compares treatment and control within repetition and
then takes the median of those three changes. Negative latency is better.

| boundary | request / response | control | RX 64 KiB | change of medians | aligned median change | wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention state | 1,308 / 1,384 B | 0.1439 ms | 0.1091 ms | -24.16% | -0.72% | 2/3 |
| hidden, M=1 | 10,268 / 10,344 B | 0.1808 ms | 0.1892 ms | +4.66% | +4.01% | 1/3 |
| standalone SwiGLU | 69,660 / 34,920 B | 0.8311 ms | 0.5337 ms | -35.78% | -32.95% | 3/3 |
| hidden, M=8 | 81,948 / 82,024 B | 1.0468 ms | 0.6660 ms | -36.37% | -37.95% | 3/3 |
| 1 MiB host to phone | 1,048,604 / 104 B | 9.7466 ms | 4.7483 ms | -51.28% | -50.86% | 3/3 |
| 1 MiB phone to host | 64 / 1,048,680 B | 3.4436 ms | 3.4061 ms | -1.09% | -0.94% | 2/3 |

The attention change of medians is misleading because the tiny runs drifted
between kernel boots; its aligned change is only -0.72%. The 10 KiB case is
also unchanged or slightly worse. A larger receive request begins to help
when the host upload crosses multiple old 16 KiB chunks.

For the 1 MiB upload, median aggregate payload rate rose from 111.78 MB/s to
215.29 MB/s, a 92.61% change of medians and a 91.23% aligned median gain. All
three processes improved. Phone-to-host rate did not change materially, as
expected because the transmit path was deliberately kept at 16 KiB.

The source-matched custom control is the only latency baseline for this patch.
It is faster than the earlier OTA-kernel asynchronous campaign, which measured
12.894 ms for the same serial 1 MiB boundary. Cross-campaign comparisons would
mix kernel builds, session history, and measurement order.

## Kernel trace mechanism

The OTA-kernel baseline trace contains one exact 1,048,604-byte request as:

```
64 x 16,384-byte receive queues + one 1,024-byte tail queue
```

All 65 queue and completion records use the same request pointer. Successful
completions total exactly 1,048,604 bytes. Between one completion and the next
queue submission, the userspace-driven serial path loses 86 us median, 158 us
p90, and 1.395 ms maximum. Those 64 gaps total 10.137 ms in the traced request.

The live treatment trace contains:

```
16 x 65,536-byte receive queues + one 1,024-byte tail queue
```

All 17 queue records use the same request pointer. The treatment trace did not
record the matching live DWC3 giveback events, so completion is not inferred
from those trace records. Instead it is independently established by the
native host's exact response validation and the phone worker's successful
request record. Trace timings are instrumentation diagnostics and are not used
in the latency table.

An earlier attempted treatment trace used an unavailable wireless ADB path, so
tracing was never enabled. Its transport result is retained but explicitly
excluded from the structural trace claim.

## Build and device safety

The patch passes kernel `checkpatch.pl` with zero errors and zero warnings.
The two configs are byte-identical with SHA-256
`9f03ed30a44329ebc6337dca7157f3eaa67c3143519883b026c51abd0d7dda43`.

| build | Image SHA-256 | boot.img SHA-256 |
| --- | --- | --- |
| control | `8ebcb8d1b5f1c8f9727fa0daa577bf61029b776d5a304cf573b89515cc905917` | `441fa2e82ce2d9ab26c61d73012a7355659d2b82f52b6666d6877de5343ddb05` |
| RX 64 KiB | `c2c527a06cab483e554df69e0d46d6542ba0db017904a404c03c544431c54a34` | `c3e8b1335bab9cf14417d3a8d355c5728831a4c0eeda7d1c7feeefdfb84a8912` |

Both temporary kernels reached full Android with root, the vendor stack, and
`/dev/usb_accessory`. After the experiment the phone was rebooted to its stock
partition. Terminal state was stock kernel
`6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k`, `ptp,adb`, USB SuperSpeed,
tracing disabled, and zero AOA worker processes.

## System implication and limit

The 16 KiB kernel receive size was a real bottleneck. A minimal 64 KiB receive
request removes about half of the 1 MiB host-to-phone latency and about one
third of the transport time for the 70-82 KiB model boundaries. It does not
help payloads that already fit in one old request, and it does not improve the
reverse direction.

This is still a serial receive loop. The kernel allocates two RX requests, but
`acc_read()` continues to submit only one at a time. The next narrow transport
optimization would be a disconnect-safe prequeued RX ring, likely four 64 KiB
requests, with ordered partial-read handling. That is a larger state-machine
change and should be reviewed separately before implementation.

This experiment runs no HTP, OpenCL, CUDA, model operator, complete layer,
full model, BurstGPT trace, power, or energy measurement. The result establishes
a faster staging primitive. The next system gate is the same control/treatment
comparison around one real phone operator to determine how much of this
transport gain survives compute and server overlap.

## Evidence

Evidence root:
`results/aoa_kernel_rx_v1/run_20260803T010732Z/`.

The root contains `ANALYSIS.json`, `BUILD_BINDINGS.json`, the 36 raw captures,
36 worker logs, stock and treatment DWC3 traces, and the measured kernel config.
The validator is `aoa_kernel_rx_v1/analyze_results.py`. No commit or push was
performed.
