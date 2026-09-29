# Rooted Pixel 10 Pro AOA FFN qualification

Measured 2026-09-24 17:17 UTC. Remote evidence: `/mnt/storage/s42-pixel10pro-aoa-20260924-v1`.

Functional qualification PASS. Continuous-call latency improvement PASS. Retaining that
improvement across idle gaps FAIL. Temporary wake-lock remedy FAIL. Cleanup PASS.
Full server latency, generated tokens, host/phone energy and scheduler integration are
NOT MEASURED in these arms.

## Matched results

All times below are milliseconds per complete request/response, not per token within
a multi-row request. Values are pooled medians of two runs per transport. The order is
ADB, AOA, AOA, ADB for each geometry. All rows use the same rooted phone, private worker,
CPU configuration, weights and inputs. Both transports retain accessory+adb mode during
the comparisons, with the Pixel link at 5000 Mb/s.

| Test | ADB median ms | AOA median ms | Latency reduction | Measured calls per transport |
| --- | ---: | ---: | ---: | ---: |
| FFN B1, half width, continuous | 7.858 | 3.918 | 50.14% | 600 |
| FFN B1, full width, continuous | 10.565 | 6.152 | 41.77% | 600 |
| FFN B2, full width, continuous | 13.882 | 8.859 | 36.19% | 360 |
| FFN B4, full width, continuous | 22.091 | 15.872 | 28.15% | 360 |
| Echo B1 payload, continuous | 1.970 | 0.213 | 89.17% | 600 |
| Echo B2 payload, continuous | 2.624 | 0.418 | 84.06% | 600 |
| Echo B4 payload, continuous | 2.884 | 0.690 | 76.07% | 600 |
| FFN B1 full, 5 ms gap | 29.824 | 28.179 | 5.51% | 360 |
| Echo B1 payload, 5 ms gap | 3.090 | 4.529 | -46.60% | 600 |
| FFN B1 full, 5 ms gap + wake lock | 29.597 | 28.724 | 2.95% | 360 |
| Echo B1 payload, 5 ms gap + wake lock | 3.023 | 2.689 | 11.03% | 600 |

Negative reduction means a regression. The inserted 5 ms pause is after each completed
RPC and is excluded from its measured latency. It is a synthetic cadence check, not a
replay of the real server. B2/4 duplicate each layer's B1 input row; this qualifies the
transport and batching geometry but does not add a new diverse-input numerical study.

Continuous B1 full-width individual run medians were 12.404/6.157/6.152/9.647 ms in
ADB/AOA/AOA/ADB order. Half-width medians were 7.758/3.782/4.294/8.420 ms. The ADB
controls drift, so the pooled percentage is an observed result, not a universal speedup.

## Latency breakdown and the remaining gap

These are independent medians. The columns need not add exactly. Outside-compute time
is RPC minus the worker's compute timer and includes USB, scheduling and protocol/copy
work on both endpoints; it is not a measurement of wire time alone.

| Full-width FFN case | ADB compute ms | AOA compute ms | ADB outside compute ms | AOA outside compute ms |
| --- | ---: | ---: | ---: | ---: |
| B1 continuous | 9.282 | 5.804 | 1.234 | 0.349 |
| B2 continuous | 12.437 | 8.270 | 1.408 | 0.570 |
| B4 continuous | 19.682 | 14.981 | 1.666 | 0.840 |
| B1, 5 ms gap | 24.212 | 23.645 | 4.905 | 2.464 |
| B1, 5 ms gap + wake lock | 24.183 | 23.801 | 4.912 | 4.629 |

AOA clearly reduces continuous-call transport overhead, but part of the total FFN
improvement is a change in compute time. With a requested 5 ms interval, compute rises
to about 24 ms on both paths and the full FFN gain shrinks to 5.51%. The transport-only
idle test regresses 46.60% in pooled median, with substantial variation between arms.
Thus the continuous result cannot be assumed for a server with gaps between offloads.

A read-only inspection found Android `mWakefulness=Dozing`, CPU governor `sched_pixel`,
and the host Pixel USB runtime state active with `power/control=on` and zero suspended
time. The hub's power policy was left unchanged. To test actual system sleep as a
cause, eight additional finite arms held a named kernel wake lock with a 120-second
expiry and an EXIT release. Every lock was observed active; successful-suspend counters
remained 0 before/after these arms. The lock did not restore the continuous-call latency.
Dozing alone is not evidence of a suspend event. The remaining CPU idle/frequency,
USB link power-state and scheduling contributions are not isolated by these measurements.
No CPU clock, governor, SELinux policy, kernel image or persistent power setting changed.

## Correctness and software

- PASS: 9,936 FFN calls / 14,256 output rows. Every output row is byte-identical
  to the archived packed CPU candidate. This includes B1/2/4, half/full width, both
  transports, and both wake-lock settings. Protocol IDs, sizes, FNV hashes and repeated
  response bytes were checked. All workers exited normally with status 0.
- PASS: 6,820 finite echo exchanges, with sequence/header/trailer and complete
  response-byte validation. The aggregate includes early smoke and slower-audit arms.
- PASS: Android C and C++ builds with `-Wall -Wextra -Werror`; Python syntax and pyflakes.
- PASS: staged worker, libraries and 948,389,216-byte packed weight artifact hashes match.

The phone is Pixel 10 Pro `5A040DLCH004ES`, physical port `2-9.2`, stock
`6.6.102-android15-8-g6eb5b2a8c46b-ab14739656-4k`, SELinux Enforcing, root UID 0.
Boot ID remained `573ce3f8-b84b-4a29-a9fd-a546747c90a7` throughout.

This test uses stock `/dev/usb_accessory` with direct libusb bulk I/O. OP15's current fast
path uses FunctionFS, so this is not a Pixel FunctionFS qualification. All USB opens and
mode changes require the Pixel serial at its physical port; generic accessory VID/PID
selection is deliberately avoided. No OP15-targeted operation was issued. Its USB state
changed independently during the shared session and the final state is recorded in cleanup.

The private worker derives from `software/pixel10pro-packed-cpu-v1`, adding a buffered
accessory stream reader and an opt-in endpoint choice. It preserves the existing v6
HELLO/EXEC protocol and coalesced response. Buffering retains bytes when a USB transfer
contains both a small header and its payload. The same rebuilt binary handles ADB/TCP
and AOA. The finite echo worker derives from the prior AOA buffered transport probe;
its serial loop is shared between TCP and AOA.

FFN geometry: Qwen layers 18-23, input/output K=5120, full intermediate width 17408,
half width 8704, quantum 4352, FP16 wire payloads. B1 request/response sizes are
10280/10288 bytes, B2 20520/20528, B4 41000/41008. One request is outstanding at a time.
CPU uses six pinned threads (cores 2-7), a persistent pool, corrected packed weights,
paired SDOT and a 64-row dynamic queue. No compute kernel changed for this experiment.

Packed artifact SHA256:
`940f5f1f2ce0c68d726713e0b1ec86808334c7ca769feac07cd3fa8581c4eae9`.
CPU library SHA256:
`b911532d756cad93e74391e86ed4d0e8e6f66773ec0dab79f8aa893021b0589d`.

`measure_pixel_aoa.py` caches the checksums of repeated test inputs/outputs after the
first warmup sample and compares every later output byte-for-byte. The initial harness
recomputed Python FNV and touched files between calls; it altered cadence materially.
Those earlier arms remain in the raw evidence and correctness count but are excluded
from the headline timing comparisons. The preserved source is
`software/pixel10pro-aoa-v1/measure_pixel_aoa_slow_audit.py`.

## Failures and cleanup

Three startup-only harness failures are retained: `adb-smoke` could not push into a
root-owned directory; `adb-smoke2` hit Android shell close-on-exec on the lock descriptor;
`adb-smoke3` correctly refused to rebind the prior attempt's own forward. Fixes were to
create the run directory as shell, pass `9>&9` to flock, and remove the stale owned
forward. None reached a phone computation. No in-flight worker was killed.

Cleanup PASS: all finite workers exited; no probe process, forward or wake lock remains;
Pixel lock available and restored to shell ownership; USB restored to `18d1:4ee7`,
`sys.usb.config=adb`, 5000 Mb/s, with ADB and root working. The targeted libusb reset
returned -5 during re-enumeration, but identity/mode/ADB checks verified successful
restoration. The harness now handles that return only by verifying the expected final
mode. No phone reboot occurred. Staged files remain in the isolated experiment directory.

The shared desktop inference server was not rebuilt or restarted. The scheduler was not
changed. No campaign energy, full-model output tokens, server overlap or two-phone
integration was measured. A new server trial is needed before making an energy or
end-to-end serving claim; the idle sensitivity must be included in that trial.

## Evidence and reproduction

- [Machine-readable audit](physical/pixel10pro-aoa-v1/SUMMARY.json)
- [Cleanup](physical/pixel10pro-aoa-v1/CLEANUP.json)
- [Per-arm configuration, raw calls, logs and outputs](physical/pixel10pro-aoa-v1/results/)
- [Private transport diff](software/pixel10pro-aoa-v1/TRANSPORT.patch)
- [Build provenance](software/pixel10pro-aoa-v1/BUILD_PROVENANCE.json)
- [Builder](build_pixel_aoa.py), [host harness](measure_pixel_aoa.py), [offline audit](analyze_pixel_aoa.py)

Example desktop invocation, from the isolated remote directory after staging and a
Pixel-only mode switch:

```sh
python3 measure_pixel_aoa.py inspect
python3 measure_pixel_aoa.py switch
python3 measure_pixel_aoa.py measure --mode aoa --kind ffn --tag FRESH_UNIQUE_TAG \
  --warmup 20 --repeats 50 --columns 17408 --tokens 1
```

Run directories must be fresh. The phone-specific flock is acquired for every arm.
Do not use the old generic `aoa_bench.py reset/bench` commands on a multi-phone rig.

## Every completed arm

Older-audit and smoke arms are included here for provenance, not pooled with v2/v3.
RPC medians exclude warmup; completed counts include it.

| Arm | Kind | Completed | RPC median ms | Compute median ms | Outside compute median ms |
| --- | --- | ---: | ---: | ---: | ---: |
| adb-ffn-b1-full-r1 | ffn | 420 | 16.036 | 14.547 | 1.443 |
| adb-ffn-b1-full-r2 | ffn | 420 | 26.898 | 22.041 | 4.366 |
| adb-ffn-b1-full-r3 | ffn | 420 | 27.602 | 22.504 | 4.413 |
| adb-ffn-smoke | ffn | 78 | 23.509 | 19.652 | 4.039 |
| adb-smoke4 | echo | 110 | 2.192 | - | - |
| aoa-ffn-b1-full-r1 | ffn | 420 | 13.475 | 12.982 | 0.483 |
| aoa-ffn-b1-full-r2 | ffn | 420 | 13.979 | 13.503 | 0.477 |
| aoa-ffn-smoke | ffn | 78 | 25.384 | 22.982 | 2.154 |
| aoa-smoke | echo | 110 | 0.245 | - | - |
| v2-1-adb-b1-full | ffn | 420 | 12.404 | 10.745 | 1.509 |
| v2-1-adb-b1-half | ffn | 420 | 7.758 | 6.321 | 1.271 |
| v2-2-aoa-b1-full | ffn | 420 | 6.157 | 5.825 | 0.332 |
| v2-2-aoa-b1-half | ffn | 420 | 3.782 | 3.406 | 0.324 |
| v2-3-aoa-b1-full | ffn | 420 | 6.152 | 5.791 | 0.364 |
| v2-3-aoa-b1-half | ffn | 420 | 4.294 | 3.910 | 0.392 |
| v2-4-adb-b1-full | ffn | 420 | 9.647 | 8.447 | 1.100 |
| v2-4-adb-b1-half | ffn | 420 | 8.420 | 7.025 | 1.336 |
| v2-batch-1-adb-b2 | ffn | 270 | 14.777 | 13.253 | 1.511 |
| v2-batch-1-adb-b4 | ffn | 270 | 22.937 | 20.429 | 1.730 |
| v2-batch-2-aoa-b2 | ffn | 270 | 8.819 | 8.212 | 0.569 |
| v2-batch-2-aoa-b4 | ffn | 270 | 15.864 | 14.970 | 0.858 |
| v2-batch-3-aoa-b2 | ffn | 270 | 8.960 | 8.364 | 0.572 |
| v2-batch-3-aoa-b4 | ffn | 270 | 15.883 | 15.008 | 0.816 |
| v2-batch-4-adb-b2 | ffn | 270 | 12.714 | 11.168 | 1.328 |
| v2-batch-4-adb-b4 | ffn | 270 | 21.461 | 19.262 | 1.603 |
| v2-echo-1-adb-b1 | echo | 330 | 0.999 | - | - |
| v2-echo-1-adb-b2 | echo | 330 | 2.747 | - | - |
| v2-echo-1-adb-b4 | echo | 330 | 2.579 | - | - |
| v2-echo-2-aoa-b1 | echo | 330 | 0.218 | - | - |
| v2-echo-2-aoa-b2 | echo | 330 | 0.396 | - | - |
| v2-echo-2-aoa-b4 | echo | 330 | 0.814 | - | - |
| v2-echo-3-aoa-b1 | echo | 330 | 0.211 | - | - |
| v2-echo-3-aoa-b2 | echo | 330 | 0.462 | - | - |
| v2-echo-3-aoa-b4 | echo | 330 | 0.658 | - | - |
| v2-echo-4-adb-b1 | echo | 330 | 2.416 | - | - |
| v2-echo-4-adb-b2 | echo | 330 | 2.211 | - | - |
| v2-echo-4-adb-b4 | echo | 330 | 2.986 | - | - |
| v2-gap5-1-adb-b1 | ffn | 270 | 29.729 | 24.179 | 4.798 |
| v2-gap5-2-aoa-b1 | ffn | 270 | 28.222 | 23.676 | 2.655 |
| v2-gap5-3-aoa-b1 | ffn | 270 | 27.953 | 23.599 | 2.384 |
| v2-gap5-4-adb-b1 | ffn | 270 | 29.938 | 24.304 | 5.061 |
| v2-gap5-echo-1-adb-b1 | echo | 330 | 3.254 | - | - |
| v2-gap5-echo-2-aoa-b1 | echo | 330 | 2.830 | - | - |
| v2-gap5-echo-3-aoa-b1 | echo | 330 | 4.636 | - | - |
| v2-gap5-echo-4-adb-b1 | echo | 330 | 2.898 | - | - |
| v3-wake-echo-1-adb-b1 | echo | 330 | 3.063 | - | - |
| v3-wake-echo-2-aoa-b1 | echo | 330 | 2.386 | - | - |
| v3-wake-echo-3-aoa-b1 | echo | 330 | 3.686 | - | - |
| v3-wake-echo-4-adb-b1 | echo | 330 | 3.004 | - | - |
| v3-wake-ffn-1-adb-b1 | ffn | 270 | 29.345 | 24.095 | 4.746 |
| v3-wake-ffn-2-aoa-b1 | ffn | 270 | 28.708 | 23.764 | 4.720 |
| v3-wake-ffn-3-aoa-b1 | ffn | 270 | 28.737 | 23.858 | 4.561 |
| v3-wake-ffn-4-adb-b1 | ffn | 270 | 29.767 | 24.307 | 5.100 |
