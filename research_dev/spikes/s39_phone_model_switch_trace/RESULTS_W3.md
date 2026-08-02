# S39 W3 matched row-order gate

Verdict:

`ROW_ORDER_EFFECT_PASS`

This is a real-device latency result for one fixed Qwen route. It is not an
energy result, a model-switch result, or an inter-stage-overlap result.

## Controlled comparison

Both treatments used:

- Qwen3 14B Q4_K_M with the same model identity;
- OP15 `GPUOpenCL` layers `[0,30)`;
- direct WiFi activations to OP12 `GPUOpenCL` layers `[30,40)`;
- 32 independent requests, a five-token prompt, and eight generated tokens;
- one 160-row prefill call followed by seven B32 decode calls;
- 384 rows and 7,864,320 direct activation bytes per run;
- no host activation payload;
- the same resident worker PID and boot nonce within each pair.

The only treatment variable was sequence-row order. `sorted` used sequence IDs
0 through 31. `shuffled` used the fixed permutation
`(17 * index + 11) mod 32`. The experiment ran sorted then shuffled on one
resident pair and shuffled then sorted on a freshly loaded pair.

## Result

| Pair | First treatment | Sorted compute | Shuffled compute | Shuffled/sorted |
|---|---|---:|---:|---:|
| AB | sorted | 24.911 s | 110.115 s | 4.420x |
| BA | shuffled | 23.580 s | 109.498 s | 4.643x |
| Aggregate | balanced | 48.491 s | 219.613 s | 4.528x |

All four runs passed:

- 1,024 generated-token comparisons in total;
- exact physical batch shapes;
- exact request, route-epoch, sequence, and position lineage;
- clean persistent-session reset and sequence removal;
- `SCHEDULED_PLACEMENT_OK` on both phones;
- no undeclared CPU fallback;
- identical direct and host activation byte counts.

The two treatments also had balanced GPU thermal starts. The mean start
temperature difference was 0.55 C on OP12 and 0.20 C on OP15, below the
predeclared 3 C limit.

## Mechanism

The worker certificates show a structural difference rather than ordinary
timing noise:

| Treatment | OP15 compute nodes | OP12 compute nodes | Derived graph executions |
|---|---:|---:|---:|
| sorted | 5,528 | 1,880 | 8 |
| shuffled | 93,976 | 31,960 | 136 |

Each sorted physical call remained one graph execution. The non-canonical
sequence order caused each physical batch to fragment into many internal graph
executions. This accounts for the earlier W1/W2 discrepancy: W1's concurrent
submitters produced unordered sequence groups, while W2 emitted canonical
phase and sequence order.

The scheduler must therefore canonicalize rows after admission:

1. decode before prefill;
2. priority and dispatch deadline;
3. sequence ID;
4. token position within a sequence.

Admission order remains useful for fairness and tie-breaking, but it must not
become physical tensor order.

## Integration

`MixedPhaseBatcher` now separates admission selection from physical layout.
Priority and dispatch deadlines determine which rows enter a batch. The
selected rows are then ordered by phase, sequence ID, and token position before
the backend call. Each selected row retains its original future, so result
ownership and caller-visible submission order do not change.

The integrated batcher completed a new real OP15-to-OP12 mixed session:

```text
80P, 16D+80P, 32D, 32D, 32D, 32D, 32D, 32D, 16D
```

All 256 generated-token checks and both placement certificates pass. Maximum
completion was 23.493 s. The nine physical calls took 24.955 s in total, which
is consistent with the earlier canonical W2/W3 points and not the fragmented
110 s treatment.

The host regression suite also submits a deliberately shuffled B32 prefill and
verifies that the backend receives `seq0/pos0..4` through `seq31/pos0..4`,
while futures still resolve in the caller's original order.

The integration evidence is under `results/w3_canonical_integration/`.

## Limits and next gate

The current relay permits one in-flight batch, so OP15 and OP12 still execute
serially for every physical call. W3 does not claim continuous inter-stage
overlap or energy saving.

The next gate adds bounded downstream credits and two batch buffers. It should
overlap OP15 batch `k+1` with OP12 batch `k` while retaining canonical row
order, exact lineage, finite memory, and the existing persistent reset
contract.

Evidence and the independent certificate are under
`results/w3_order_gate/`.
