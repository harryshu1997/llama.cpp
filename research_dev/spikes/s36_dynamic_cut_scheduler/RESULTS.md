# S36 Results

## Verdict

`MIXED_CONTINUOUS_BATCH_AND_DYNAMIC_CUT_MECHANICS_PASS_CUDA_RELIEF_FAIL`

The real three-device runtime completed the frozen 60-request trace in all
three paired repetitions. It proved request-level cut pinning, concurrent OP12
and OP15 lanes, B32-capable unified KV, mixed prefill and decode in one phone
call, exact token agreement, SLO preservation, and complete lease cleanup.

Selected-CUDA stage time did not improve reproducibly. One pair improved by
1.78 percent, while two regressed. Median treatment stage time was 1.86 percent
higher than median control. No energy-saving claim follows from S36.

## Physical Configuration

- selected server device: A6000 `CUDA0`; the second A6000 was excluded;
- OP12 and OP15: HTP0, resident Gemma 4 12B layers `[0,8)`;
- CUDA prefix: `[0,4)`; CUDA terminal tail: resident `[4,48)`;
- request routes: CUDA cut 4, or either phone at cut 4 or 8;
- logical model: Gemma 4 12B F16; phone and host GGUF shards have distinct
  whole-file hashes;
- trace: the frozen S23 densest two-second 60-request arrival window, with the
  previously declared synthetic priority/SLO sidecar and four-token physical
  prompt proxy;
- row knee: 32; gather window: 20 ms; unified KV enabled only for these
  LayerSplit workers.

Unified KV reduced each phone's B32 KV allocation from the failing 469 MiB
independent-stream layout to a 14 MiB, 256-cell shared cache. OP12 used explicit
attention because its v75 fused path is disabled; OP15 used fused HTP attention.

## Route Profile

Every B8 and B32 point used two physical repetitions and returned terminal
tokens `[532,532,532,532]`.

| Route | B8 median | B32 median | Conservative bound |
| --- | ---: | ---: | ---: |
| CUDA cut 4 | 480.403 ms | 518.013 ms | 621.896 ms |
| OP12 cut 4 | 886.444 ms | 1142.951 ms | 1236.553 ms |
| OP12 cut 8 | 1300.479 ms | 2009.097 ms | 2135.231 ms |
| OP15 cut 4 | 763.859 ms | 981.378 ms | 1201.056 ms |
| OP15 cut 8 | 1074.665 ms | 1576.126 ms | 1675.050 ms |

The bound is the maximum observed B8/B32 request latency; policy adds a fixed
20 percent margin.

## Three Paired Runs

| Pair | Control CUDA stage | Treatment CUDA stage | Relief | Mechanics |
| --- | ---: | ---: | ---: | --- |
| 1 | 2399.231 ms | 2356.497 ms | +1.78% | PASS |
| 2 | 2313.433 ms | 2413.824 ms | -4.34% | PASS |
| 3 | 2305.950 ms | 2341.199 ms | -1.53% | PASS |
| Median | 2313.433 ms | 2356.497 ms | -1.86% | PASS |

Every pair completed 60/60 requests with zero SLO misses and identical output
tokens. Priority-0 p95 remained within the frozen 5 percent gate. Both phones
and cuts 4 and 8 were exercised in every treatment.

## Mixed-Phase Evidence

Each treatment produced one physical phone mixed-phase batch. In pair 1, OP12
executed one HTP range call at cut 8 with 35 rows:

- 3 decode rows at position 5;
- 32 prefill rows from eight four-token requests;
- one priority band and one active range `[0,8)`;
- release reason `BATCH_KNEE`;
- measured call time 257.425 ms.

This is mixed prefill and decode inside one real `llama_decode` graph execution,
not two overlapping processes or an offline simulator.

## Placement And Cleanup

Treatment session certificates report `SCHEDULED_PLACEMENT_OK`, zero missing
buffers, and HTP0 compute on both phones. `GET_ROWS` is the declared CPU lookup.
All four workers ended every paid session with zero active sequences; runtime
route pins, sequence leases, and physical credits were empty.

## Interpretation

The requested basic mechanisms now work on real hardware. The negative relief
result is consistent with the shallow phone islands and extra tail fragmentation:
the treatment removes prefix CUDA work but creates more, smaller terminal-tail
calls. The next performance experiment must increase useful phone depth or
coalesce tail work; S36 does not justify an energy claim.

## Scope

Measured: physical request routing, B32 unified-KV operation, mixed-phase phone
batching, cut 4/8 execution, per-request tokens and SLOs, and selected-CUDA
blocking stage-call time.

Not measured: GPU board energy, phone energy, USB/WiFi energy, total-system
energy, production `llama-server` behavior, or observed-length BurstGPT service
capacity.
