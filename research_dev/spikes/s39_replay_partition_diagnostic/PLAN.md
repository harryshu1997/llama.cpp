# CUDA replay-partition diagnostic

Status:
`REPLAY_PARTITION_DIAGNOSTIC_COMPLETE; BATCH_SHAPE_NUMERICAL_SENSITIVITY; STOP_CURRENT_QWEN_Q8_ROUTE`.

## Scope

This is a separate CUDA-only diagnostic. It is not a W9 rerun, replacement, or
repair. It consumes the immutable W9 P1.T ledger and the exact W8-R1 histories
that W9 bound. It cannot authorize controller integration, performance,
quality, scheduler eligibility, or energy claims.

The Qwen2.5 14B Q8_0 phone route already failed its independent task-quality
gate. After this diagnostic, no more phone acquisition is allowed for that
route.

## Frozen inputs

Reconstruct each B8 history from immutable evidence:

1. W8-R1 supplies the eight-token prompt and two preexisting tokens.
2. W9 `PHONE_F0` supplies position 10.
3. W9 `PHONE_INFLIGHT` supplies position 11.
4. W9 `CUDA_CONTINUATION` supplies positions 12 through 22.

Before acquisition, the reconstruction must match:

- F0 history SHA-256
  `fd0000745ad7b5ddefcb7a4645769a920536934bafbfb14df52e1847e40e6f67`;
- F1 history SHA-256
  `03d09124220ce94fa865708f8d53c860e889c507484b0881adea896b90cb5243`;
- W9 continuation SHA-256
  `01d21f1435e314a3047184cbf375cfec44e6ded1f3ec2938e634a02718453b1a`.

The builder must also verify the complete W9 root manifest and 114-record
ledger chain. `W9_HISTORIES.json` and
`REPLAY_PARTITION_DIAGNOSTIC.json` are frozen before any new CUDA execution.

## Matrix

Use the exact W9 Q8 model, CUDA UUID, `[0,30)` head, `[30,48)` tail, B8, and 11
autonomous continuation tokens.

1. Two fresh-process incremental repetitions:
   replay F0 as `[8,3]`, ingest F1-F0 as `[1]`, then continue.
2. Two fresh-process full-F1 repetitions:
   replay F1 as `6 x [2]`, then continue.
3. One fresh route with two paths in the same processes:
   incremental, remove all eight sequences, verify zero state, then full F1.
4. Two optional fresh-process full-F1 repetitions:
   replay F1 as `[8,4]`, then continue.

Every route launch creates new head, tail, and relay processes. The same-process
case alone reuses one route across its two paths.

## Evidence order

The probe persists a canonical raw report before any equality evaluation. It
contains:

- every input row and returned row;
- request, epoch, sequence, position, and token fields;
- phase and exact call shape;
- both same-process continuation vectors;
- state counts before execution, before removal, and after removal;
- the raw token vector from every fresh repetition.

A separate validator reopens these bytes, reconstructs the expected rows,
checks realized CUDA0 placement, and only then compares vectors. Top-2 logit
margins are not captured because StageNet V3 terminal responses expose selected
token IDs but not logits. The runtime protocol is not expanded for this
diagnostic.

## Interpretation

- Identical geometry is not repeatable:
  `BACKEND_OR_KV_STATE_BUG_STOP_INTEGRATION`.
- Fresh geometries repeat but differ:
  `BATCH_SHAPE_NUMERICAL_SENSITIVITY`. Future exact gates need a path-matched
  oracle; cross-geometry equality becomes diagnostic.
- Fresh paths match but the same-process sequence differs:
  `CLEANUP_RESET_BUG`.
- Incremental fresh replay differs from the W9 ledger:
  `REPLAY_DELTA_IMPLEMENTATION_OR_PROVENANCE_BUG`.

Multiple findings may coexist. The reducer records every applicable finding
and chooses the most safety-critical primary diagnosis.

## Stop boundary

After independent validation:

1. freeze raw and reduced manifests;
2. record the result without changing W9;
3. stop Qwen Q8 phone runs;
4. carry only a corrected, path-matched primitive to a scheduler-eligible model
   pair;
5. run one reduced forward/reverse cycle on the actual RTX 4060 Ti.

The paper claim remains capacity and continuous service during model
replacement. Energy is secondary until the eligible bidirectional 4060 Ti
cycle passes.
