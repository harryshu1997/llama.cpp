# OP15 adoption energy screen A-B-B-A V1

Status: `ENERGY_SCREEN_FAIL_FULL_TRACE_BLOCKED`.

This matched physical screen tested early same-process Gemma weight adoption
on the RTX 4060 Ti plus OP15 before authorizing the 74-request F16 BurstGPT
campaign. Every arm completed the same six BurstGPT Qwen requests, the same
Gemma request 50, and the same verified 2,013,265,920-byte transition. Model
source preparation, transition, request execution, CPU-package energy,
GPU-board energy, and synchronized whole-phone energy were inside the paid
boundary.

The control staged Gemma during the second Qwen cohort and served Gemma after
the Qwen tail. The dynamic arm staged during the first cohort and began Gemma
as soon as the placement became READY, concurrently with the second Qwen
cohort.

| Metric | Delayed control | Early dynamic | Change |
| --- | ---: | ---: | ---: |
| Duration | 262.205 s | 325.561 s | +24.163% |
| CPU package energy | 9.465 kJ | 10.366 kJ | +9.521% |
| GPU board energy | 7.770 kJ | 9.468 kJ | +21.861% |
| Server compute energy | 17.234 kJ | 19.834 kJ | +15.084% |
| Whole-phone energy | 0.519 kJ | 0.640 kJ | +23.365% |
| Fleet energy | 17.753 kJ | 20.474 kJ | +15.326% |
| Gemma READY | 193.733 s | 130.396 s | -32.693% |

The paired fleet-energy savings were -17.277% and -13.402%. All validity
gates passed: exact per-request token hashes, equal work, identical placement
and transition bytes, GPU reserve, synchronized phone coverage, zero process
and cgroup swap, and zero OOM events.

Early READY did not help because a complete Gemma request was too coarse for a
Qwen GPU bubble. The Qwen tail and Gemma shared desktop CPU and CUDA resources.
Mean Qwen request-47 prompt time rose from 48.406 to 86.042 seconds, and mean
Gemma prompt and decode time rose from 37.997 and 30.348 seconds to 63.886 and
131.172 seconds. The scheduler must use a bounded micro-filler with measured
CPU and CUDA contention, not concurrent whole-request execution.

The full 74-request dynamic campaign was not run. The plan requires a positive
conservative screen before that campaign, so the measured static policy remains
the fail-closed fallback at 25.537% fleet-energy savings versus CPU overflow.
