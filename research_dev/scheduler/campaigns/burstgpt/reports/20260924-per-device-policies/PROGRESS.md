# Per-device adaptive phone policies: progress log (newest last)

Task: per-device adaptive phone policies (which device set assists: OP15 only / OP15+Pixel / Pixel only,
x split fraction), decided per device set and batch composition from measured evidence; then dev_v2
matched arms, then the full longtail_v1 evaluation against the all-desktop baseline.
Scratch trees: `$SCRATCH/perdev/{base,root}` (base = main at 00:22 UTC 2026-09-25; reports/ and
non-scheduler research_dev entries symlinked read-only).

## 2026-09-25 00:22 UTC - start
- Read AGENTS.md, NEXT_AGENT_PROMPT_SCHEDULER.md (incl. the ~21:40 UTC update), pixel-integration-2
  README + PROGRESS, two-phone-readiness GAPS_README, coherent-policy-coalesced README, server-probe-fix
  README, EVIDENCE_FIXES_README.
- Rig: lock free, nothing running (00:14 UTC).

## 2026-09-25 00:40 UTC - change of plan (parent, from the user)
- "we can stop the trace, we can fix the pixel problem first": NO rig or phone action (no preflight,
  arms or adb); the rig lock and the Pixel are reserved for a Pixel CPU+GPU worker workstream.
  Scope is now code only: design decision, implementation in the isolated copy, tests, full suite,
  PER_DEVICE_POLICIES.diff, merge into main with backups, full suite on main, README with the arms
  to run later.
- Desktop contact so far: one read-only ssh at 00:14 UTC (`date`, lock probe with `flock -n`
  returning LOCK_FREE, `ps`, `df`, `ls /mnt/storage`). Nothing synced, written or started on the
  desktop; no adb command was run.

## 2026-09-25 01:10 UTC - design decision + implementation in `perdev/root`
- Server C++ read (no change needed): the runtime control already accepts any layer mask inside the
  resident union (`server-context.cpp` ~2600/2750), `s41_server_ffn_runtime::apply_policy`
  (`server.cpp` ~870) gives every helper its owned subset and sets a helper without an active layer to
  (0, 0) (deferred connection until first use), and the dormant host share releases by
  (layer mask, host columns), restoring the previous release first (`llama-model.cpp` ~1792). So
  switching a whole phone off per token boundary is already supported; only different column counts
  per phone would need C++ (+ rebuild + identity re-materialization).
- Decision: device-set sub-policies inside the one two-phone route ("(a) without a server change").
  (b) (alternative routes) would need separate launches/catalog identities and a model reload per switch.
- Implemented: device-subset policies in `_envelope_subpolicies`; device-set-aware candidate sampling
  (primary alone first, then adding co-helpers; co-helper-only sets are fallbacks); coherent server
  probe explores device sets per batch composition (phase 1 vs host, phase 2 challenger vs incumbent
  with the unchanged bounds), drops per (batch, device set), failure drops (set + supersets, every
  composition) with fallback to the remaining sets, server-driven fallback for new owners; proofs
  require calls only from phones of executed device sets (co-helper plans only); primary phone not
  charged for co-helper-only windows; drain of a primary session leaves co-helper-only policies;
  decision records / window events / snapshots carry device sets (multi-device only).
- New tests `tests/test_per_device_policies.py` (18). Updated three Stage A assertions that pinned
  "every policy drives both phones" (test_two_phone_gaps x2, test_two_phone_activation x1).
- Full suite on root: running.

## 2026-09-25 01:02 UTC - tests, suites, merge
- `tests/test_per_device_policies.py` (18): base tree 13 FAIL/ERROR + 5 guards pass
  (`NEW_TESTS_BASE.log`), root 18/18 (`NEW_TESTS_ROOT.log`). Single-phone digest guard
  `sha256:e4eaec0f...` identical on base and root; existing `SinglePhoneUnchangedTests` pass.
- Added `prepare_trace_inputs_v2.py --drop-dispatch-policy` (+ test) so the legacy all-desktop arm is
  derived with tooling from the same template as the other arms.
- Root full suite (final code): 141 modules / 2,021 tests; only the known 10 ms timing assertion
  `test_cached_synthetic_refinement_is_below_ten_milliseconds` failed (10.09 ms). Interleaved timing
  of the measured quantity base vs root: medians 9.6-10.1 vs 9.95-10.85 ms (noise; a cProfile of 40
  calls shows the touched planning functions at 0.12 ms on both trees and root faster overall).
- Main scheduler tree verified byte-identical to base before the merge; `git apply --check` OK;
  backups of the 16 modified files + SHA256SUMS in `$SCRATCH/perdev/premerge-backup/` (the new test
  file was absent). APPLIED to main 01:01 UTC; all 17 files byte-identical to root. Main full suite
  running.
- 01:10 UTC main full suite after the merge: 141 modules / 2,021 tests, exit 0
  (`SUITE_MAIN_AFTER_MERGE.log`). README written (design, rules, tests, arms to run later).
  The desktop deploy was NOT synced (no rig action).
