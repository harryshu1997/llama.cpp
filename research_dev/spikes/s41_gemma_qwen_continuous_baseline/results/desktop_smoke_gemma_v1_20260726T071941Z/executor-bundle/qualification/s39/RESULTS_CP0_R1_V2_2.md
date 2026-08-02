# CP0-R1 V2.2 evidence correction

Verdict:

`V2_2_EVIDENCE_READY_ACQUISITION_NOT_RUN`

V2.2 is an additive successor to the frozen V2.1 bundle. V1, V2, and V2.1
remain unchanged. No hardware acquisition, model execution, Qwen3 14B
qualification, Qwen3 8B acquisition, switch cycle, trace replay, controller
integration, or energy acquisition ran.

## Frozen quality input

`CP0_R1_MMLU64_SOURCES_V2_2.json` binds all 57 test parquets from
`cais/mmlu` revision
`bc5d09e5f0d160a95bcd36354bb5e16e50afe270`. The source files total
3,643,066 bytes and remain outside the worktree.

`build_cp0_r1_mmlu64_v22.py` verifies every source byte count and SHA-256,
reads the exact parquet row order, and applies the previously declared
round-robin selection. The frozen corpus contains row zero from all 57
ASCII-sorted subjects followed by row one from the first seven subjects.

The canonical 64-row JSONL has SHA-256
`3ffafee1615ae2de690a2726b880823e167a3d9c210c5faed86d8f0e93ecff4f`.
The evaluator removes only phase wrapper fields and requires every acquired
corpus row to equal this artifact. Rebinding output and artifact hashes around
a different question no longer passes.

## Closed bypasses

- CUDA must answer at least 25 of 64 canonical MMLU items correctly. This is a
  prospective above-chance sanity floor in addition to phone-vs-CUDA
  noninferiority.
- Incumbent A must use `GPUOpenCL`, cut 30, OP15 stored layers `[0,32)`,
  OP12 stored layers `[24,40)`, and the exact two candidate-bound shard
  digests.
- Phone, tested CUDA route, and independent monolithic CUDA oracle must each
  produce exactly eight continuation tokens for each of the eight requests.
- Each phone mechanics request completion must be no later than its linked
  publication. Every publication remains strictly before the linked CUDA
  readiness event. Bridge timestamps must equal their phase event times.
- A, B, and PAIR phase IDs must be pairwise distinct in addition to having
  disjoint ordered intervals.
- `cp0_r1_evidence_v22.py --authorize-cycle` accepts only A, B, and PAIR raw
  bundle roots. It reopens and re-evaluates all three. There is no legacy
  status-result input.

V2.2 retains V2.1's exact phase locks, full readiness commands, memory
accounting, oracle geometry, bridge identities, transfer byte calculation, and
full-shard local-UFS checks.

## Verification

- focused V2.2 suite: 13/13 passed;
- complete S39 suite: 323/323 passed;
- all 57 source sizes and SHA-256 values match the pinned revision's upstream
  LFS metadata;
- synthetic positive chain: A, B, PAIR, and root-derived cycle authorization
  passed;
- V1, V2, and V2.1 manifest digests remained unchanged.

Negative tests cover a fully rebound noncanonical corpus, matched low-quality
CUDA and phone outputs, all six incumbent-route fields, equal seven-token
paths, publication before linked completion, a detached bridge event time,
reused A/B and B/PAIR IDs, tampered predecessor roots, and a legacy status-only
authorization attempt.

The contract-only CLI emits
`V2_2_EVIDENCE_READY_ACQUISITION_NOT_RUN`. The next authorized acquisition
remains Qwen3 14B `A_ONLY`, but it was not run here.
