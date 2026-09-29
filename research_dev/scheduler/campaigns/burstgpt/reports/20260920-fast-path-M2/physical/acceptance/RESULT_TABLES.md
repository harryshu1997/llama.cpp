## Utilization and power: FAIL

| N | Phone compute ms/token/slot | Phone RPC ms/token/slot | Host-only decode W, measured | Phone-arm host decode W, measured | Phone W, assumed | Full-cohort steps/layer |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 164.651 | 194.934 | 124.304 | 66.846 | 4.5 active; 0.875 idle | 573 |
| 2 | 82.999 | 103.113 | 124.962 | 67.197 | 4.5 active; 0.875 idle | 572 |
| 4 | 51.488 | 67.398 | 133.308 | 66.018 | 4.5 active; 0.875 idle | 571 |
| 8 | 64.171 | 77.996 | 145.646 | 57.087 | 4.5 active; 0.875 idle | 567 |

N=4 to N=8 compute rises 24.633%; RPC rises 15.724%. The plan requires phone time per token per slot to fall with N, so M2 fails. No cause is established by these single pairs.

The denominator is full-cohort physical calls * N / 18 owned layers. Partial start/drain calls remain in the records and are excluded from these fixed-N points.

## Per-slot correctness: PASS under the amended rule

| N | Request | Slot, both arms | Prompt tokens | Acceptance | Matching positions | First mismatch |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 256 | EXACT | 576/576 | - |
| 2 | 0 | 1 | 256 | EXACT | 576/576 | - |
| 2 | 1 | 0 | 257 | EXACT | 576/576 | - |
| 4 | 0 | 3 | 256 | EXACT | 576/576 | - |
| 4 | 1 | 2 | 257 | EXACT | 576/576 | - |
| 4 | 2 | 1 | 258 | NEAR_TIE | 141/576 | 80 |
| 4 | 3 | 0 | 259 | EXACT | 576/576 | - |
| 8 | 0 | 7 | 256 | EXACT | 576/576 | - |
| 8 | 1 | 6 | 257 | EXACT | 576/576 | - |
| 8 | 2 | 5 | 258 | EXACT | 576/576 | - |
| 8 | 3 | 4 | 259 | EXACT | 576/576 | - |
| 8 | 4 | 3 | 260 | EXACT | 576/576 | - |
| 8 | 5 | 2 | 261 | EXACT | 576/576 | - |
| 8 | 6 | 1 | 262 | EXACT | 576/576 | - |
| 8 | 7 | 0 | 263 | EXACT | 576/576 | - |

| N | Request / slot | Step | Host / phone token | Host top-1 logit | Host top-2 logit | Margin | NMSE | Later differing positions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 / 1 | 80 | 19 / 15 | 25.099 | 25.099 | 3.891e-04 | 8.226e-07 | 434 |

Both displayed N=4 logits round to the same value; the margin above uses the unrounded raw values. Every mismatch retains its step, both host logits, margin and NMSE in CHECK_N4_pair.json. Only the first mismatch decides acceptance; 434 later differences are labeled after_context_divergence.

## Single-pair energy and latency

| N | Arm | Request host kJ, measured | Decode ms/token by request index | Phone W, assumed | Request phone kJ, assumed |
| --- | --- | --- | --- | --- | --- |
| 1 | control | 44.457 | 616.324 | 0.875 idle | 0.316 |
| 1 | combined | 20.388 | 518.871 | 4.5 active; 0.875 idle | 1.346 |
| 2 | control | 45.628 | 629.805 / 625.968 | 0.875 idle | 0.324 |
| 2 | combined | 21.623 | 545.869 / 541.543 | 4.5 active; 0.875 idle | 1.408 |
| 4 | control | 50.662 | 657.491 / 653.627 / 649.476 / 645.610 | 0.875 idle | 0.339 |
| 4 | combined | 26.488 | 678.298 / 674.360 / 670.098 / 665.989 | 4.5 active; 0.875 idle | 1.738 |
| 8 | control | 60.732 | 725.834 / 721.996 / 717.085 / 713.368 / 709.529 / 705.589 / 701.570 / 697.547 | 0.875 idle | 0.376 |
| 8 | combined | 41.735 | 1223.169 / 1220.323 / 1217.133 / 1213.956 / 1210.076 / 1206.038 / 1201.857 / 1197.622 | 4.5 active; 0.875 idle | 3.126 |

Host energy is RAPL package plus NVML board, counted once per concurrent request group. Phone energy is separate and assumed over the union of active decode intervals plus idle time. Decode power uses the common decode interval; per-request ms/token is the server predicted_ms divided by output tokens. Both long arms write raw logits; these timings include that instrumentation. The phone arm has lower host power at every N but is slower end to end at N=4 and N=8.

## Memory and validation

| Arm | memory.peak GiB | memory.events.max ready | finish | oom_kill finish |
| --- | --- | --- | --- | --- |
| determinism/run1 | 27.793 | 0 | 0 | 0 |
| determinism/run2 | 27.799 | 0 | 0 | 0 |
| regression | 28.994 | 0 | 0 | 0 |
| n1-control | 27.963 | 0 | 0 | 0 |
| n1-combined | 28.246 | 0 | 0 | 0 |
| n2-control | 28.444 | 0 | 0 | 0 |
| n2-combined | 28.845 | 0 | 0 | 0 |
| n4-control | 29.083 | 0 | 0 | 0 |
| n4-combined | 29.073 | 0 | 0 | 0 |
| n8-control | 29.098 | 0 | 0 | 0 |
| n8-combined | 29.108 | 0 | 0 | 0 |

All 11 arms use fresh MemoryMax=infinity, MemorySwapMax=0 scopes. All 100 rig unittests pass and pyflakes is clean before each arm. Cleanup passed for all 11 scopes and owned server PIDs; all seven phone closes report terminal status 0 and RESTORED. The rig lock is free, the GPU has no compute process, and OP15 is visible on ADB 5037. No other phone was used.
