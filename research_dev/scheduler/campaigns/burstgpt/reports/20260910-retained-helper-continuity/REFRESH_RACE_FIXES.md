# Atomic refresh and queued-control invalidation

Local software corrections, 2026-09-10. Not deployed or physically measured.
Earlier reports and physical artifacts retain their original results and hashes.

## Changes

1. Successive refreshes previously modified candidates and evidence before a
   pending-policy conflict raised. Refresh now computes on the existing
   controller's session clone and publishes only on success. Exactly compatible
   unsent refreshes coalesce. Incompatible refreshes leave the entire prior
   checkpoint unchanged and can be retried after the pending control completes.
   A maintenance drain is not silently replaced. Already-issued controls and
   their original acknowledgement identities are not rewritten.
2. Batch changes previously cleared evidence and reservations but left a queued
   positive winner and deferred challenger. They now invalidate those unsent
   selections too. The baseline ACK opens a baseline window; a later positive
   probe requires new complete-pair admission under the current context.
   Exhausted requests do not resume the old winner. Request-wide exploration
   counters and spent energy are not reset.
3. Helper loss previously left positive intent for the next baseline ACK to
   issue. It now cancels unsent refresh, challenger, retry and verification
   selections. Both acknowledgement processing and final control issuance
   recheck availability. A positive control already issued may still be ACKed:
   its actual identity is recorded, then a zero control is issued immediately.
   The transition's calls and energy remain in non-qualifying history.

No artifact, parent, geometry, generation, lease, qualification or physical-proof
validation was weakened. Session replacement and fraction/energy policy were not
redesigned. The previous compatible-evidence and measured-rejection fixes remain.

## Validation

Thirteen added regressions cover successive unsent and issued refreshes,
idempotency, incompatible and injected-failure atomicity, maintenance-mask
preservation, batch 2-to-1 revalidation, exhausted budgets, deferred challengers,
helper loss before and after control issuance, delayed recovery accounting, and
availability checks at acknowledgement and final issuance.

The initial regressions reproduced all three reported failures before the
production changes. Final command:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_dev/scheduler/tests:. python3 -m unittest test_sustained_assistance test_adaptive_decode test_adaptive_runtime test_late_helper_energy_policy test_session_cow_transaction test_replay_determinism -q
```

**203 tests passed in 86.269 s.** The two replay tests include byte-identical
repeated runs and compare to these unchanged canonical goldens:

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

The focused suite also covers the previous Gemma unassisted-tail regression,
strict helper identity, leases, session generations and physical rollback fakes.
No physical experiment or complete scheduler harness was run in this follow-up.

## Exact files changed

Paths relative to `research_dev/`:

- `scheduler/_internal/adaptive_decode_ops/helpers.py`
- `scheduler/_internal/adaptive_decode_ops/windows.py`
- `scheduler/_internal/adaptive_decode_ops/sequencing.py`
- `scheduler/tests/test_sustained_assistance.py`
- `scheduler/campaigns/burstgpt/reports/20260910-retained-helper-continuity/README.md`
- `scheduler/campaigns/burstgpt/reports/20260910-retained-helper-continuity/REFRESH_RACE_FIXES.md`
- `talks.md`

Source SHA-256 values:

| File | SHA-256 |
| --- | --- |
| helpers.py | b346034700c900460f8044e326ade9c76e3385e8df2cc351e8f5ce0036f46f09 |
| windows.py | eb752547ddfdfd1ee52238015840644e3ed0901a7cfa34840e13885d3ffd63fb |
| sequencing.py | b38f98d363043b5f1970390b3b6b45aab6793da2d38e5cb6f9a15b2a851da516 |
| test_sustained_assistance.py | 2a039f99a458d53c3128403d5cd2d7774e00267bd975c7ed7fb52339da88b40f |

Frozen `baselines/cuda_graph_v1/COMPARISON.json` remains
`83054fdc4179b4c64939a1ed07ce0891e833a3f642218464c47e62eaf27ba022`.
No new savings, physical-coverage, or all-issues-resolved claim follows from this
software validation.
