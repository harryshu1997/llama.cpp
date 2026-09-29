# Multi-session phone results

Date: 2026-08-09 EDT.

Verdict: `PHYSICAL_SCREEN_PASS; FULL_TRACE_ADMISSION_PENDING`.

Three resident HTP sessions successfully held about 9225 MiB of useful Gemma
and Qwen FFN weights. HTP1 and HTP2 jointly replaced the complete Qwen SWIGLU
FFN for layers 0-11 during decode M=1. The RTX 4060 Ti kept the same 18-layer
placement in both arms, prefills stayed local, and the CPU and phone handled
the portion that did not fit on the GPU.

## Matched repeated energy screen

The ABBA screen replayed BurstGPT rows 52, 53, and 31 once for warmup and once
inside the paid interval. Requested input and output lengths were unchanged.
All four runs returned exactly the same token IDs. Energy is the synchronized
sum of Intel package RAPL, RTX 4060 Ti NVML board energy, OP15 USB input, and
simultaneous OP15 battery discharge.

| mean over two repeats | GPU plus CPU control | GPU plus CPU plus OP15 | change |
| --- | ---: | ---: | ---: |
| paid duration | 32.291 s | 28.400 s | -12.05% |
| CPU package energy | 2968.209 J | 1469.643 J | -50.49% |
| GPU board energy | 995.339 J | 897.411 J | -9.84% |
| desktop CPU plus GPU | 3963.549 J | 2367.054 J | -40.28% |
| whole-phone energy | 73.805 J | 82.495 J | +11.77% |
| accounted fleet energy | 4037.354 J | 2449.548 J | -39.33% |

The phone used 8.689 J more, while the desktop saved 1596.495 J. Net fleet
saving was 1587.806 J. The per-pair fleet savings were 39.01% and 39.65%, so
the result reproduced in both orders rather than depending on one run.

Each treatment executed 1296 phone calls: 648 warm calls plus 648 paid calls.
The two control runs executed zero calls while keeping the same weights warm.
Both treatment runs recorded zero reset recoveries. Composite Qwen arming used
HTP1 for layers 0-5 and HTP2 for layers 6-11 with union mask `0x0fff`.

HTP compute p50 was about 9.69 to 9.89 ms for one complete layer FFN. Complete
phone RPC p50 was 10.72 ms and 10.88 ms in the two treatment repeats. The
full-replacement route is faster because it removes the corresponding desktop
CPU FFN work; its phone time is intentionally on the critical path and is not
a partial-split join wait.

## What failed before this route

Offloading only the 4608-column Qwen suffix for all 23 CPU layers slowed the
requests by roughly 26% to 33%. The view-safe CPU remainder lost its favorable
repack behavior, so hiding the phone branch was not enough. That candidate is
rejected. The accepted route instead moves a complete FFN for 12 layers and
leaves the remaining local graph intact.

A 25-block full-width HTP graph also exceeded `HTP_OP_MAX_BUFS`. One full-width
block per six-layer worker avoids that limit. A 12800-column trial was rejected
after inspection showed that Qwen's actual FFN width is 17408.

## Scope

This is strong evidence for the exact three-request Qwen decode M=1 screen,
not yet a full BurstGPT or arbitrary-shape certificate. The next promotion
step is a full two-large-model BurstGPT ABBA pair driven by the unified
scheduler, with the same resident control boundary and per-request fallback.

The strict hash-bound record is
`results/QWEN_FULL_FFN_ENERGY_SCREEN_R1_R2.json`. The checked-in scheduler
residency plan is `results/OP15_THREE_SESSION_RESIDENCY_PLAN_V1.json`. The
canonical scheduler selects it for `ENERGY_POSITIVE_RESIDENT_PHONE_ROUTE` in
`results/UNIFIED_QWEN_FULL_FFN_DECISION_V1.json`, with a conservative 39.01%
fleet-energy margin and one composite HTP1 plus HTP2 arm.
