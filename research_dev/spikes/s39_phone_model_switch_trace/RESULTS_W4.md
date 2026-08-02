# S39 W4 second-model quality screen

Verdict:
`QWEN25_Q4_Q8_ROUTE_QUALITY_FAIL; SCHEDULER_INELIGIBLE; REPLACEMENT_PATH_STOPPED; BIDIRECTIONAL_REPLAY_BLOCKED`

## Question

Can Qwen2.5 14B Instruct at Q4_0 or Q8_0 provide the second complete
phone-warm model needed by the active warm-tier design?

The route must agree with a same-artifact CUDA control over a corpus, not only
one prompt, before it can enter the model-switch scheduler.

## Artifacts and route

Full model:

- path: `/home/myid/zs89458/Documents/models/Qwen2.5-14B-Instruct-Q4_0.gguf`;
- bytes: 8,517,726,048;
- SHA-256:
  `924a4c39ef9fc6c139875ab6771c2e8172a3b40ffec5c720eca69ad7a0edfae7`;
- architecture: Qwen2, 48 decoder layers, embedding width 5,120.

Two complete phone assignments were tested:

| Backend | OP15 | OP12 |
|---|---|---|
| HTP | `[0,30)` | `[30,48)` |
| GPUOpenCL | `[0,32)` | `[32,48)` |

Weights were copied to phone UFS before the paid run. Runtime cut activations
traveled directly from OP15 to OP12 over WiFi TCP.

## Frozen quality gate

The corpus contains 128 deterministically selected WikiText prompts tokenized
by this exact Qwen model:

- eight input tokens per prompt;
- eight greedy output decisions per prompt;
- four B32 cohorts;
- four 64-row prefill calls and seven B32 decode calls per cohort;
- same cut, row order, batch sequence, and model artifact on CUDA.

Corpus SHA-256:
`7bd8b321ccbfc05c3a30f5b5e89f5df482774ce1fa082a0f850311b39ee62e1b`.

The thresholds were frozen before the full acquisitions:

| Metric | Threshold |
|---|---:|
| First-token agreement | at least 95% |
| All token-decision agreement | at least 95% |
| Exact eight-token sequences | at least 80% |

## Results

| Route | First-token match | Decision match | Exact sequences | Median phone cohort | Verdict |
|---|---:|---:|---:|---:|---|
| HTP cut 30, fused FA | 125/128 (97.66%) | 918/1024 (89.65%) | 96/128 (75.00%) | 34.804 s | fail |
| HTP cut 30, explicit FA | 124/128 (96.88%) | 922/1024 (90.04%) | 99/128 (77.34%) | 18.578 s | fail |
| GPUOpenCL cut 32 | 122/128 (95.31%) | 880/1024 (85.94%) | 90/128 (70.31%) | 29.016 s | fail |

The first-token threshold alone passes. Both sequence-sensitive thresholds
fail on every route. The gate is conjunctive, so all three verdicts are
`QUALITY_FAIL`.

The earlier GPUOpenCL B32 screen matched one repeated prompt exactly. This
corpus result shows why that point remained provisional.

## Execution evidence

The final GPUOpenCL acquisition completed:

- 44 physical batches;
- 1,920 total rows;
- 39,321,600 direct activation bytes;
- zero host activation payload bytes;
- 31,020 OP15 and 15,708 OP12 realized compute nodes;
- zero missing-buffer compute nodes;
- only declared OP15 `GET_ROWS` work on CPU.

OP12 logged an optional `sub_group_shuffle_xor` OpenCL compile failure. The
placement certificate still attributes every realized OP12 compute node to
OpenCL, so this is recorded as a backend limitation rather than a CPU
fallback.

`validate_qwen25_quality.py` independently reloads the canonical report,
recomputes all metrics and token digests, validates both CUDA and phone stage
certificates, and checks both direct relays. A post-run record re-hashes both
staged shards and the worker binaries without a device reboot; its boot IDs
match the paid session certificates. This narrows artifact attribution but is
not equivalent to a pre-run hash record. The validator returns exit code 3
with a canonical `QUALITY_FAIL` certificate. The combined probe and validator
suite passes 22/22 tests, including six adversarial evidence mutations.

## Decision

Qwen2.5 14B Q4_0 is not scheduler-eligible. The active warm-tier system still
lacks two correct phone routes, so the reduced A-to-B-to-A physical replay and
the 74-request trace remain blocked.

Do not repeat this Q4_0 acquisition, average it with the one-prompt result, or
lower the quality thresholds. It led to the bounded Q8_0 screen below.

## Q8_0 final bounded candidate

The Qwen2.5 14B Q8_0 artifact is pinned to revision
`05244aa5d871c661c80082a15d3bce44714d068d`:

- full bytes: 15,701,598,336;
- full SHA-256:
  `23ca481b8226b2492ba8f3eb7af41e0f99d8605c16fb6dec7bc5cf6716b673cf`;
- OP15 `[0,30)` shard: 9,608,904,640 bytes,
  `b9611440eb4901764cef6afb55e08418acb114f1dfdc5b97e2c4acafefcc5375`;
- OP12 `[30,48)` shard: 6,098,628,640 bytes,
  `b66f9f6ace28da341f21f4f0d03ffa05c31cea647d43da4023e373f6021551ac`.

The terminal shard omits `token_embd.weight` because this model has a distinct
`output.weight` and a hidden-injected tail never reads the token embedding.
This removes 827,228,160 bytes without changing tied-output or non-Qwen shard
selection. A Qwen2.5 0.5B control produced the same split and monolithic CPU
argmax and logit after this change.

Both phone shards load and execute B32. OP12 retained only about 0.67 GiB of
available memory after the B32 point and used about 78 MiB of process swap.
The route is therefore memory-feasible but does not pass a strict zero-swap
gate.

The unchanged 128-prompt quality screen completed twice with identical
physical and CUDA token hashes:

| Metric | Result | Threshold | Verdict |
|---|---:|---:|---|
| First-token agreement | 123/128 (96.09%) | at least 95% | pass |
| Token-decision agreement | 872/1024 (85.16%) | at least 95% | fail |
| Exact eight-token sequences | 90/128 (70.31%) | at least 80% | fail |

The persisted certificate binds 44 phone batches, 1,920 rows, 39,321,600
direct activation bytes, zero host activation bytes, 29,084 OP15 compute
nodes, 17,644 OP12 compute nodes, and zero missing-buffer nodes. Median phone
cohort time was 30.650 seconds. The matched CUDA median was 0.526 seconds.

`validate_qwen25_quality.py` now consumes an immutable route specification, so
the same validator checks Q4_0 and Q8_0 without hardcoded cross-contamination.
It independently recomputes the Q8_0 metrics, identities, cut, row counts,
placement, direct-relay bytes, device boot IDs, shard hashes, and runtime
hashes. It returns exit code 3 with `QUALITY_FAIL`.

The complete S39 Python suite passes 141/141 tests. This includes Q4 and Q8
real-evidence reducers, producer phase diagnostics, shard-selection controls,
and adversarial evidence mutations.

## Final decision

Qwen2.5 Q8_0 is not scheduler-eligible. The planned Qwen2.5 replacement path
stops after this failure, as frozen before acquisition. The active warm-tier
model-switch design remains mechanically plausible but does not yet have two
quality-certified phone routes, so no bidirectional replay or energy result is
authorized.

The next reviewed question is the correctness contract: either preserve exact
same-artifact greedy-token agreement and diagnose backend numerical
convergence, or predeclare a task-level quality metric appropriate to the
serving workload. Trying another model or weakening the observed thresholds is
not the next checkpoint.
