# CUDA replay-partition diagnostic result

Verdict:
`BATCH_SHAPE_NUMERICAL_SENSITIVITY; STOP_CURRENT_QWEN_Q8_ROUTE`.

This is a separate CUDA-only diagnostic. W9 was not rerun, repaired, or
replaced. Its contract, root manifest, and final ledger record remain
byte-identical.

## Frozen identity

- Contract:
  `REPLAY_PARTITION_DIAGNOSTIC.json`
- Contract SHA-256:
  `0b654d1b5185c10dad8cda9657771cc563cc5df0a3511a92fb20c4187969c0bc`
- Exact input:
  `W9_HISTORIES.json`
- Input SHA-256:
  `b617bba50fe4b40d3f3760aaa777f46e862ce8bc4df0906912564666f607fc92`
- Acquisition:
  `results/run_20260725T_REPLAY_PARTITION/`
- Acquisition manifest SHA-256:
  `853c5d922bf4a15075027609c13237f6d4c60b385af70c5c2adaac04d62b6844`
- Reduced analysis SHA-256:
  `6cff9ee6b98000dff94ec2524abfaf7437d708bc2468df7c77f5c37dcfac0900`

The input builder reopened and verified the complete immutable W9 root
manifest, all 114 hash-chained ledger records, and the exact W8-R1 report.
Reconstructed F0 and F1 match W9's recorded digests:

| Frontier | Width per request | SHA-256 |
|---|---:|---|
| F0 | 11 | `fd000074...6f67` |
| F1 | 12 | `03d09124...5243` |
| W9 CUDA continuation | 11 | `01d21f14...b1a` |

## Acquisition

Seven route launches ran on the selected RTX A6000 CUDA0. Every fresh
repetition created new head, tail, and relay processes. The same-process case
alone executed both paths in one route lifetime.

| Path | Repetitions | Replay calls | Repeatable |
|---|---:|---|---|
| Incremental F0 plus delta | 2 fresh + 1 same-process | `[8,3] + [1]` | Yes |
| Full F1 | 2 fresh + 1 same-process | `6 x [2]` | Yes |
| Full F1 optional control | 2 fresh | `[8,4]` | Yes |

All path captures report state counts `0 -> 8 -> 0`. The second path in the
same-process run starts at zero state and exactly matches its fresh-process
counterpart. Fourteen placement certificates pass: `[0,30)` and `[30,48)` for
all seven launches, CUDA0 compute only except the declared head `GET_ROWS` on
CUDA host, and zero missing-buffer compute nodes.

The raw reports contain every input and output row, request/epoch/sequence
lineage, position, token, call shape, both same-process continuation vectors,
and all state counts. They were written before the separate validator evaluated
equality. Top-2 logit margins were not captured because the unchanged StageNet
V3 terminal protocol returns selected token IDs, not logits.

## Exactness result

| Comparison | Result |
|---|---|
| Incremental fresh repetition 1 vs 2 | Exact |
| Full-F1 chunk-2 repetition 1 vs 2 | Exact |
| Full-F1 `[8,4]` repetition 1 vs 2 | Exact |
| Same-process incremental vs fresh incremental | Exact |
| Same-process full F1 vs fresh full F1 | Exact |
| All incremental paths vs W9 ledger | Exact |
| Full-F1 chunk-2 vs full-F1 `[8,4]` | Exact |
| Incremental vs either full-F1 geometry | Different |

The frozen reducer defines first mismatch in canonical continuation-vector
order: sequence first, then token offset. It records sequence 2, absolute
position 19, incremental token 943 versus full-F1 token 14135. Inspecting the
same persisted vectors by earliest token position shows an earlier temporal
divergence at sequence 5, position 12: token 16 versus token 15. Three of eight
sequences diverge, covering 19 of 88 continuation tokens.

The result rules out the two more serious hypotheses for this capture:

- identical geometry is repeatable, so this is not observed backend or KV
  nondeterminism;
- fresh and same-process paths agree with zero residual state, so this is not
  an observed cleanup/reset leak;
- the incremental path reproduces W9 exactly, so the W9 ledger and incremental
  replay/delta implementation are internally consistent.

Both one-shot full-F1 partitions agree, while incremental F0 replay followed by
one-token F1-F0 ingestion differs. Under the frozen interpretation this is
batch-shape numerical sensitivity, more specifically sensitivity to the
incremental append boundary.

## Decision

Future exactness gates must use a path-matched oracle. Cross-geometry agreement
is diagnostic and cannot veto an otherwise exact path unless a separate
quality contract requires it.

No more real-phone runs are authorized for this Qwen2.5 14B Q8_0 route. Its
independent task-quality failure remains decisive, so the route cannot become
scheduler-eligible.

The next bounded experiment is one reduced forward/reverse switch on the
actual RTX 4060 Ti using a scheduler-eligible model pair and this corrected
path-matched primitive. The paper's primary claim remains capacity and
continuous service during model replacement. Energy remains secondary until
that eligible bidirectional cycle passes.

That cycle was not run here because S39 CP0 does not yet contain two eligible
decoder routes. Both Qwen replacement candidates failed the frozen quality
gate. Executing a nominal Gemma/Qwen cycle would violate the requested
eligibility condition and could not support the system claim.
