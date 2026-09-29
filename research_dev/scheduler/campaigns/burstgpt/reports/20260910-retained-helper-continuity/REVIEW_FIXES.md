# Review corrections: deferred controls and refresh at zero assistance

Implemented locally on 2026-09-10. These corrections have not been deployed or
physically measured. The original source manifest and failed preflight under
`physical/` are unchanged; they describe the earlier implementation.

## P1: stale deferred execution plan

Deferred policies now participate in the same exact-compatible remapping as
incumbents and challengers. A missing or incompatible mapping discards the
deferred policy and its pending reason. The current valid winner is retained.
The retry path independently checks that the policy belongs to the current
candidate set before issuing control.

Restoring a winner puts the controller into EXPLOITING. Retrying a deferred
challenger now restores PROBING first, so complete-pair admission still applies.
An unaffordable retry continues the valid incumbent; a measured rejection of
the refreshed challenger also returns to that incumbent.

## P2: rejection and attempt identities lost at zero assistance

Compatible evidence retention is now independent of the executing fraction.
A request on desktop at 0% keeps the exact-compatible candidate's measurements,
rejection reason and canonical attempt identity through an envelope refresh.
It does not retry a rejected 100% policy or regain exhausted attempts merely
because bookkeeping hashes changed.

Real execution changes without an exact compatibility mapping still require
bounded revalidation. Original receipts, energy accounting, request-wide probe
limits, physical identity checks and leases are unchanged.

## Regression results

Eight added tests cover:

1. Deferred 25% challenger uses the new plan after refreshed-winner acknowledgement,
   reserves a complete pair, and returns to the winner after measured rejection.
2. Changed challenger geometry is discarded without substituting another fraction.
3. Missing compatibility proof discards the deferred authorization.
4. The retry boundary fences an injected stale candidate.
5. Insufficient remaining opportunity cannot be bypassed after winner restoration.
6. Refresh at 0% preserves LEARNING_NO_PAIRED_IMPROVEMENT and does not reprobe.
7. Refresh at 0% preserves exhausted incomplete-attempt limits without inventing rejection.
8. A genuine contract change still starts bounded revalidation.

Seven failed before the production changes; the genuine-change control already
passed. Final command:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_dev/scheduler/tests:. python3 -m unittest test_sustained_assistance test_adaptive_decode test_adaptive_runtime test_late_helper_energy_policy test_session_cow_transaction test_replay_determinism -q
```

**190 tests passed in 85.709 s.** Both replay cases remain byte-identical across
repeated runs, with unchanged goldens:

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

Production changes are limited to `_internal/adaptive_decode_ops/helpers.py`
and `_internal/adaptive_decode_ops/sequencing.py`. Tests are in
`tests/test_sustained_assistance.py`; this note, the report index and
`research_dev/talks.md` record the follow-up.

Source hashes after the fix:

| File | SHA-256 |
| --- | --- |
| helpers.py | 0a1e79f6d8d252e3494fe76eb54bd72d2ba8155a74ec11eeb69dbc9c330eed4c |
| sequencing.py | 0a24ff46e4b2e1997482fce6d7f382f81de91359ec5405a2823327e920976950 |
| test_sustained_assistance.py | d03458bdd90c05dfb799e9bbb1aed326d96cc3305c98c72aab13e259ee4d9c93 |

No baseline or physical experiment was run, and no completed artifact was
overwritten. No savings or all-issues-resolved claim follows from these tests.
