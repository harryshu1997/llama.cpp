# Scheduler replay fixture

`s42_saved_runs_v1.json.gz.b64` is a gzip-compressed, base64-encoded canonical
JSON fixture. It contains the catalogs, observation stores, model manifests,
request records, and saved arrival snapshots needed by
`test_replay_determinism.py`.

The fixture covers all 24 requests from the sparse-locality24 v8 run and all 16
requests from the session-COW v3 gate. The v8 replay is split at physical
lifecycle discontinuities that arrival snapshots cannot reconstruct. Each
segment still uses the saved catalog, observations, request inputs, and exact
arrival snapshots; one segment seeds its ready layout from two saved physical
snapshots before replaying its late requests.

Generate candidate hashes and decoded payloads in a temporary directory with:

```sh
PYTHONPATH=.:research_dev/scheduler/tests python3 \
  research_dev/scheduler/tests/test_replay_determinism.py \
  --regenerate-goldens /tmp/s42-replay-goldens
```

Run that command before and after a deliberate scheduler behavior change and
include the decoded JSON diff with the golden update. A hash-only update is not
an acceptable golden change.

## Golden history

2026-09-11: deliberate phase-scoped lease change. The old payloads were
reproduced from the before-images, with both old hashes matching exactly.
All 40 selected route IDs and the arrival-phase helper events are unchanged.
In both preparation branches, nine operator-plan hash references change
because execution slots no longer include capacity-wide preparation claims.
No preparation outcome, session generation, or rollback branch changes.

In v8's `late-gemma-tail` segment, the same generation-1 phone geometry
`67bcbf8a...14b2b` is proposed for arrived request `88135` instead of waiting
for `88139`. The lease calendar now exposes the existing helper candidate
at the earlier arrival. Proposal benefit is `19818086400` instead of
`77856768000` uJ, reflecting only the earlier arrived work; evaluation
payloads at indices 46-50 change accordingly. Event count stays 66.
There is no new residency heuristic or future-demand input.

Full before/after payloads and the field-level decoded diff are in
[the phase-lease report](../../../campaigns/burstgpt/reports/20260911-phase-scoped-dev3/).
New hashes: v3 `5d52e867...7caf4`, v8 `24192446...2b917`.

2026-09-07: exact hot-reuse projection preserves measured residency instead of
replacing it with a modelled allocation. The v3 replay consequently avoids two
redundant selection passes: its phone event entries 26 and 27, both identical
`EVALUATED / NO_FEASIBLE_PHONE_RESIDENCY` events for the same overlay arrival,
are removed. No route, helper event, generation, geometry, or preparation
success/failure branch changes. The full deleted payloads and old/new hashes
are in [20260907-hot-reuse-decoded-diff.json](20260907-hot-reuse-decoded-diff.json).
v3 changes from `f78d2b2c...1829` to `ea5b30c9...ca27d`; v8 is byte-identical.
Both old hashes were reproduced from the preserved pre-change deployment;
the test checks repeat-run byte identity as well as each golden.

| Date | session_cow_gate_v3 | sparse_locality24_v8 | Decoded diff |
|---|---|---|---|
| 2026-09-03 00:12 (Stage 0b) | `e3f7cc9d...61b0` | `05337c76...8898` | initial hardened oracle |
| 2026-09-03 (per-session state machine) | `0a4e50fd...9521` | `87bdec1e...3d6c0` | additive only: three `SESSION_VERIFIED_FROM_OBSERVATION` phone-layout events (HTP0/HTP1/HTP2, layout generation 1, state VERIFIED) in the first arrival segment of each case, plus one `SESSION_VERIFIED` event for the replaced session at generation 2 in the copy-on-write branch. No existing event, payload key, ordering, or hash input changed. Verified by regenerating both cases from the pre-change tree (Stage 2 snapshot, reproduces the previous hashes) and diffing the decoded JSON. |
| 2026-09-04 (current residency reserve) | `f78d2b2c...1829` | `965f218b...868d` | The previously documented v29 reserve change lets v3 use 18 Qwen layers instead of 17: resident bytes `9,091,153,920 -> 9,625,927,680`, masks `31/2016/129024 -> 63/4032/258048`, with corresponding geometry and operator-plan hashes. The generation-2 one-session replacement remains one-session, but moves from HTP1 with a 6-layer Gemma shard to HTP0 with a 9-layer Gemma shard; total target bytes change `8,005,877,760 -> 9,602,334,720`. Selected route IDs and completed/failed lifecycle event-kind sequences are unchanged. The v3 replay driver repeats the last saved Gemma observation after minimum residency to supply the third stable selection snapshot; v8 is byte-identical. |
