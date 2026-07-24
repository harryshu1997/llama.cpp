# S39 W0 route and trace implementation

Verdict:

`TRACE_REPLAY_MECHANICS_PASS; QWEN_BATCH_PROVISIONAL; TWO_MODEL_ROUTE_GATE_BLOCKED`

No model switch, request catch-up, SLO benefit, or energy benefit is claimed.

## Implemented

- Dense Qwen3 supports experimental partial weight loading, layer-bounded graph
  execution, injected hidden-state input, and layer-filtered KV allocation.
- LayerSplit head and tail modes emit executed placement and B1 service records.
- The active BurstGPT trace is reduced deterministically to five promotion
  windows and nine target changes.
- A strict route-evidence reducer hashes and parses the raw phone/CUDA records
  and derives `CURRENT_ROUTE_READINESS.json`.
- A fail-closed controller implements the ordered promotion state machine and
  refuses an intent unless both the source and target routes are `PASS`.

## Real route screen

Prompt: `The capital of France is`

Generation: eight greedy tokens.

| Metric | Qwen3 14B Q4_K_M | Gemma 4 12B Q4_0 |
|---|---:|---:|
| OP15 layers | `[0,30)` | `[0,30)` |
| OP12 layers | `[30,40)` | `[30,48)` |
| Phone TTFT | 1.574 s | 1.317 s |
| Phone service wall | 3.668 s | 2.724 s |
| CUDA service wall | 0.163 s | 0.149 s |
| Phone/CUDA token match | 8/8 exact | 0/8 sequence match |
| Placement | clean | clean |
| Derived status | `PROVISIONAL_BATCH` | `FAIL_CORRECTNESS` |

Qwen phone and CUDA token IDs:

```text
12095, 13, 3555, 374, 279, 6722, 315, 279
```

Gemma phone token IDs:

```text
496, 45518, 100, 108, 100, 45518, 107, 101
```

Gemma CUDA token IDs:

```text
9079, 236761, 107, 100, 45518, 100, 101, 6372
```

The Gemma graph split itself is not the immediate cause: a same-artifact CUDA
head/tail split at cut 30 reproduces the monolithic top-5 exactly. The remaining
failure is in the heterogeneous phone route or backend numerical path.

## Trace replay

The frozen intent times are:

```text
PROMOTE  60 s
DEMOTE  129 s
PROMOTE 359 s
DEMOTE  421 s
PROMOTE 498 s
DEMOTE  546 s
PROMOTE 911 s
DEMOTE  989 s
PROMOTE 1154 s
```

`replay_intents.jsonl` and `replay_manifest.json` rebuild byte-identically.
The controller refuses this entire trace because Gemma is incorrect and Qwen
is only provisional. A test-only two-PASS fixture completes all 54 ordered
state actions without accepting an overlapping or out-of-order transition.

## Verification

- replay reducer: 18 tests pass;
- route readiness reducer: 6 tests pass;
- warm-tier controller: 11 tests pass;
- host CUDA build: pass;
- Android Hexagon/OpenCL build: pass;
- Qwen same-host full/split correctness at cuts 24, 30, and 32: pass;
- real OP15 plus OP12 Qwen B1 route: provisional pass;
- real OP15 plus OP12 Gemma B1 route: correctness fail;
- active controller CLI: fail-closed exit 2.

Raw route records and their checksums are under
`results/w0_route_screen/`.

## Real Qwen batch screen

The same resident OP15 and OP12 worker processes ran B1, B8, and B32 in order.
The first two sessions detached and reset request-local KV; the last stopped
the workers. All 328 generated-token checks match the same-artifact CUDA
sequence.

| Cohort | P50 | P95 | Max | Request throughput | Gain vs B1 |
|---:|---:|---:|---:|---:|---:|
| B1 | 3.825 s | 3.825 s | 3.825 s | 0.261 req/s | 1.00x |
| B8 | 24.859 s | 24.861 s | 24.861 s | 0.322 req/s | 1.23x |
| B32 | 87.548 s | 87.554 s | 87.555 s | 0.365 req/s | 1.40x |

For B32, each stage executed one 160-row prefill call and seven 32-row decode
calls. OP15 reported 95,250 OpenCL compute nodes plus 127 declared CPU
`GET_ROWS` nodes. OP12 reported 32,130 OpenCL compute nodes and no CPU compute
nodes. Both placement certificates passed with zero missing buffers.

The 5,120-element F32 boundary produces 7,864,320 activation bytes per WiFi
leg at B32, or 15,728,640 bytes across the OP15-to-host and host-to-OP12 legs.
This payload is small compared with the 87.5-second service time. The current
screen is compute-bound at the route level, but it is serial across stages and
does not prove pipelined steady-state service.

Weights were copied over USB before the run and loaded from each phone's UFS.
Runtime activations used WiFi TCP through the host coordinator. Because endpoint
and interface evidence was not captured by the runner, the path fact is marked
posthoc and remains provisional. Raw reports, worker logs, `RUN_CONTEXT.json`,
and the derived certificate are under `results/w0_qwen_batch_wifi/`.

The measured buffer records are component allocations, not RSS or a zero-swap
certificate:

| Device | OpenCL model | OpenCL KV | OpenCL compute | CPU-mapped model |
|---|---:|---:|---:|---:|
| OP15 | 5,618.47 MiB | 1,920.00 MiB | 127.01 MiB | 417.30 MiB |
| OP12 | 2,543.29 MiB | 640.00 MiB | 306.75 MiB | 417.30 MiB |

## Next gate

Do not start CP2 handoff from this result. W0 needs:

1. a repeated-process prompt corpus for Qwen with exact token/logit tolerance;
2. separated TTFT/decode, continuous arrivals, and 7-process variance;
3. pre-captured WiFi endpoints, routes, and interface counters;
4. zero-swap and prepared-image memory records on both phones;
5. either a repaired Gemma route or a second decoder using the proven Qwen
   partial-stage substrate;
6. exact tokenizer, template, context, KV, and sampler identity records.

Only after both models derive `PASS` may the physical promotion controller run.
