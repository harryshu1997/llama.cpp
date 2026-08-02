# S22 results

## Verdict

`MIXED_SLO_MECHANICS_PASS_NUMERICALLY_UNCERTIFIED`

Current-source Gemma-4-12B executed across OP15, OP12, and one RTX 4060 Ti.
The phones hold duplicate `[0,8)` F16 prefixes and independent KV. The desktop
holds a Q8 CUDA `[0,8)` control head and one shared Q8 CUDA `[8,48)` tail.

## Three-device batching base

| Lane | Range | Backend | Rows | Batches | Median request latency |
| --- | --- | --- | ---: | ---: | ---: |
| OP15 head | `[0,8)` | HTP0 v81 | 16 | 4 x B4 | 1580.18 ms |
| OP12 head | `[0,8)` | HTP0 v75 | 16 | 4 x B4 | 2487.35 ms |
| 4060 Ti tail | `[8,48)` | CUDA0 | 32 | 8 x B4 | N/A |

All eight requests met the configured 30 s mechanics SLO and returned
`[532,236772,236772,564]`. Placement certificates contain 2320 OP15 and 2728
OP12 substantive HTP0 nodes, with only declared `GET_ROWS` on CPU. The tail has
25542 CUDA0 compute nodes and no CPU compute or missing output buffer.

The tail used B4 rather than B8 because the phones finish at different times.
The runtime does not impose a global phone barrier.

## Mixed-SLO physical result

The profile loader reopens and hashes every evidence artifact before admission.
Six requests arrived together. The selector maximizes offloaded depth, then
uses the slowest feasible equal-depth phone to preserve faster capacity.

| SLO class | Selected route | Predicted finish | Actual maximum | SLO |
| --- | --- | ---: | ---: | ---: |
| tight | CUDA `[0,8)` -> CUDA tail | 800 ms | 1219 ms | 1400 ms PASS |
| medium | OP15 `[0,8)` -> CUDA tail | 1800 ms | 1299 ms | 2200 ms PASS |
| loose | OP12 `[0,8)` -> CUDA tail | 2800 ms | 2228 ms | 3500 ms PASS |

A 5 ms immediate-gather run produced tail mean batch 1.04 and maximum B2. The
repaired policy divides unused SLO slack across every remaining head and tail
gather point. Tail mean batch rose to 2.40 and maximum batch to B4 while every
SLO still passed. CUDA and OP12 stayed B2. OP15 was B2 for two rounds, then B1
because downstream skew exceeded its 50 ms budget.

The result is mechanics-only. `R0_CUDA` requires the explicit
`--allow-numeric-uncertified` flag. Its last token is 107 while both phone
routes end in 564. The runtime reports
`MECHANICS_PASS_NUMERICALLY_UNCERTIFIED`; SLO success is not a quality claim.

## Multi-token prefill screen

A four-token prompt with two live requests per phone executed as B8 on OP15,
B8 on OP12, B8 twice on the shared tail, then B2 decode. This proves the
protocol, KV lineage, and batch runtime accept several contiguous positions per
sequence.

It does not have an output-quality certificate. OP15 returned `[496,3103]`,
while OP12 returned `[236772,496]`. The same-route control also differs: OP15
sequential prefill returns `[496,3103]`, but chunk-4 returns `[236789,496]`.
CP7 below shows why this is a token-agreement screen failure, not proof that the
pipeline computes the wrong graph. Chunked prefill remains outside the eligible
atlas until the phone boundary and real-prompt quality gates pass.

## CP7 batch-shape numerical screen

`activation_compare.py` runs sequential A, chunked, and sequential B in one
resident worker and clears the same sequence slot between trials. Both
sequential runs are byte-identical at every layer-8 row on every tested host
backend. Sequential versus chunked activations differ as follows:

| Head backend and weights | Maximum relative L2 | Minimum cosine | Existing 0.5% gate |
| --- | ---: | ---: | --- |
| CPU, F16 | 0.219% | 0.9999976 | PASS |
| A6000 CUDA, F16 | 0.307% | 0.9999954 | PASS |
| 4060 Ti CUDA, Q8 | 5.317% | 0.9985867 | FAIL |

The full, unsplit Gemma-4-12B F16 model was then served through the same V3
batch API on the A6000. Sequential A and B both returned
`[532,236772,236772,236789]`; one four-token batch returned
`[532,236772,236772,236761]`. Thus exact greedy-token equality across batch
shapes is not a valid standalone oracle even for ordinary monolithic
inference. CPU and CUDA F16 boundary errors are within the existing project
relative-L2 gate. The Q8 result is not, and neither phone HTP backend has yet
run this boundary probe.

The verdict is unchanged: the mixed F16-phone/Q8-tail route remains
numerically uncertified. Admission needs a measured phone boundary row and a
real-prompt output-quality gate, not a bitwise token requirement on one
synthetic near-tie prompt.

`profiles-4060-b4-cp7.json` supersedes the frozen CP5 profile for future runs
and labels R0, R1, and R2 `NUMERICALLY_UNCERTIFIED`. The old `EXACT_POINT`
labels meant only that one frozen token sequence agreed; they are preserved in
the historical result but are not eligible evidence after CP7.

## Mechanics established

- StageNet V3 preserves `(request_id, route_epoch, seq_id, position)` lineage.
- New sequences enter and retire independently with per-sequence KV removal.
- Each device forms bounded batches from its own ready queue.
- Three prefix lanes progress independently and join one shared CUDA tail.
- A finite p95 policy pins one route epoch before execution.
- Gather time is bounded by device caps and per-request SLO slack.
- Lineage mutation, queue overflow, stale removal, and drain violations fail
  closed.

Review repaired a `submit()` versus `stop()` race that could strand a future.
The worker also rejects out-of-vocabulary token IDs. The current Python suite
is 42/42.

## Capacity and limits

OP12 cannot reserve 32 sequence slots at 512 tokens per sequence for this
eight-layer shard: its 896 MiB HTP KV allocation fails. Successful workers use
four or eight slots. Admission must bind resident-KV capacity, not only nominal
batch fields.

- The mixed-SLO result uses one BOS token and four greedy decode steps.
- Multi-token prefill executes; its phone boundary and workload-level quality
  gates have not run.
- Phone heads are F16 and the desktop model is Q8.
- Only the fixed layer-8 cut is executable; arbitrary cuts and an
  OP12-to-OP15 chain are not implemented.
- There is no energy measurement or llama-server integration.

## Evidence

- `results/three-device-4060-b4.json`
- `results/mixed-slo-final.json`
- `results/final-op15.log`, `results/final-op12.log`
- `results/final-4060-head.log`, `results/final-4060-tail.log`
- `results/three-device-prefill-b8.json`
- `results/op15-prefill-sequential.json`
- `results/op15-prefill-chunk4.json`
- `results/cpu-f16-activation-sequential-vs-chunk4.json`
- `results/a6000-f16-activation-sequential-vs-chunk4.json`
- `results/4060-activation-sequential-vs-chunk4.json`
- `results/a6000-f16-full-token-sequential-vs-chunk4.json`
- `SHA256SUMS.txt`
