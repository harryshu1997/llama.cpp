# W9 profile-driven zero-extra cutover result

Verdict:
`PROFILED_ZERO_EXTRA_CUTOVER_FAIL; STOP_AFTER_P1_T_EXACTNESS`.

Scope remains `MECHANICS_ONLY` and `scheduler_eligible=false`. W9 does not
authorize controller, quality, trace, model-switch, latency-benefit, or energy
claims.

## Prospective contract and tests

The contract was frozen before acquisition:

- `W9_PROFILED_CUTOVER_CONTRACT.json`
- SHA-256:
  `85d3fd2ebce18741586c3a8131d7d614dd701ae23251b7642792ad981076a575`
- exact pair order: `P1.T, P1.C, P2.T, P2.C, P3.T, P3.C, P4.T, P4.C`
- fixed B8 live cohort, two preexisting tokens, and 13 post-start tokens;
- selected CUDA UUID:
  `GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f`;
- frozen current-profile decision: `k_extra=0`.

Before acquisition:

- focused W9 tests: 32/32 pass;
- full S39 host tests: 229/229 pass;
- changed shell syntax: pass;
- `git diff --check`: pass;
- both USB phones, every deployed binary and shard, the exact Qwen2.5 14B
  Q8_0 model, selected A6000, W8 source manifest, and frozen W9 source map:
  pass.

The independent loader reopens every file in the W8 manifest and verifies that
the W8 observations are no greater than their frozen conservative predictor
bounds. The prospective commit cost is labeled separately and is not presented
as a measured W8 field.

## Consumed real attempt

The immutable attempt is:

`results/w9_profiled_cutover/run_20260725T_W9/`

Its manifest SHA-256 is:

`895470a6edf4f7be79ca5015f357246469abbc475725d17512987cacbd81b1a6`.

`P1.T` crossed its durable paid-start marker and is therefore consumed. It
failed the mandatory same-frontier CUDA continuation comparison:

```text
treatment: CUDA continuation mismatch
```

Per the frozen no-replacement rule, the launcher stopped. It did not rerun
P1, execute P1.C, or create P2-P4. The failure record contains exactly
`paid_pair_ordinals=["P1"]` and `completed_pair_ordinals=[]`. No four-pair
median or performance pass exists.

## Evidence before refusal

The failure does not erase mechanics that can be reconstructed directly from
the immutable hash-chained ledger:

| Observation | P1.T |
|---|---:|
| B8 post-start token publications | 104 |
| Tokens per request | 13 |
| Position range per request | 10-22 |
| `PHONE_F0` publications | 8 |
| `PHONE_INFLIGHT` publications | 8 |
| `CUDA_CONTINUATION` publications | 88 |
| `F0` positions | `[10] x 8` |
| `F1` positions | `[11] x 8` |
| CUDA launch delay | 100.061 ms |
| CUDA ready | 2.914 s |
| First post-trigger publication | 1.596 s |
| Ledger completion | 3.747 s |
| Replay-start to `F1_ACK` window | 120.772 ms |

The 114-record ledger validates as one ordered SHA-256 chain. It contains one
durable `CUDA_COMMITTED` record, a second fsynced marker bound to that record's
digest and durable timestamp, one phone release, and one `COMPLETE`. CUDA
publications occur only after the durable owner transition.

All four treatment workers emitted one valid terminal session certificate:

- OP15: OpenCL `[0,30)`, 88 steps, zero missing-buffer nodes;
- OP12: OpenCL `[30,48)`, 88 steps, zero missing-buffer nodes;
- CUDA head: CUDA0 `[0,30)`, 352 steps, zero missing-buffer nodes;
- CUDA tail: CUDA0 `[30,48)`, 352 steps, zero missing-buffer nodes.

The selected GPU was idle for five prelaunch samples over more than one
second. Its ready-time process list contains exactly the two CUDA worker PIDs,
and its post-treatment samples contain no compute process. Treatment-start
phone temperatures were 39,900 millicelsius on OP15 and 42,800 millicelsius
on OP12. Host and phone route processes were absent after cleanup. The root
manifest revalidates with `sha256sum -c`.

These are failure diagnostics, not a W9 mechanics pass. In particular, the
paired fresh-CUDA trace control did not run.

## Blocking exactness result

CUDA replayed `F0`, ingested the one-token `F1-F0` delta, committed ownership,
and produced all remaining publications. The subsequent fresh same-frontier
CUDA replay did not produce the identical autonomous continuation vector.
That comparison is mandatory, so the treatment exited nonzero before writing
a passing report or pair certificate.

The failure path persisted only the refusal string, not the two compared token
vectors. The exact first mismatch request and position therefore cannot be
derived from this attempt. The likely boundary to investigate in a separate
prospectively frozen gate is incremental `F0` replay plus one-token delta
ingestion versus one-shot `F1` replay. W9 itself cannot be rerun or repaired
after its consumed paid failure.

## Claim boundary

W8-R1 remains the latest passing live-session mechanics result. W9 establishes
that the profile-derived zero-extra execution reached the intended real
in-flight and durable-ownership path once, but it rejects exact state
equivalence at that path. The Qwen2.5 Q8 task-quality failure remains in
force. No scheduler eligibility, repeated performance result, full model
switch, or energy result follows from W9.
