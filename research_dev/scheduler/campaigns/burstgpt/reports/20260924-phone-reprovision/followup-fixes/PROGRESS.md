# PROGRESS: change #4 follow-up fixes, second attempt (server verdict reset, boundary storm)

Workspace: this directory. `base/` and `root/` = fresh rsync copies (no `__pycache__`) of the CURRENT main
`research_dev/scheduler`, taken 2026-09-24 ~15:11 EDT. Edit only `root/`. Times local (EDT).

## Checkpoint 1 (15:05-15:15): reading, workspace, artifacts

- Read AGENTS.md, 20260924-phone-reprovision/{README,RIG_RESULTS}.md, followup-fixes/{PROGRESS.md,
  REPROVISION_FIXES_PARTIAL.diff}, 20260924-coherent-policy-coalesced/README.md (1-4.4),
  20260924-server-probe-fix/README.md; coherence.py, adaptive_decode_state.py, history.py, candidates.py,
  phone_shards.py (layout/shard identities), adaptive_decode.py (checkpoint/restore).
- Run artifacts: local read-only copies already in `../runs/{coherentRP,coherentEF,coherentEF2}` (coherentRP
  RESULT.json size equals the desktop file, 41,477,534 B). Streamed `run-dev2allon-1` RESULT.json,
  ADAPTIVE_DECODE_OBSERVATIONS.json, adaptive-timing-events.json to `../runs/allon/` (ssh cat; nothing run
  on the desktop beyond ls/cat). allon = coherence + reprovisioning + dispatch_policy(model_affinity,
  work_conserving_admission), PASS 9/9, 812 s.

### Judgement of the partial diff

- Unrelated hunk: `reports/20260922-fast-path-M3/measure_pixel_aoa.py` (someone else's main edit between the
  previous agent's base and root copies). Must not be in the deliverable.
- **Bug**: `server_policy_key` became a 3-tuple, but `AdaptiveDecodeController.restore` rejects server-policy
  keys with `len(key) != 4` -> every checkpoint restore with a coherence group would raise. Keep 4 elements.
- Problem 1 plumbing (identity kwarg on start/helper_ready/helper_rebound, `artifact_layout_identity_sha256`)
  is sound in shape; to be re-done on the fresh copy and checked against every helper call site.
- Problem 2 gate: signature-based rate limit is a reasonable shape, but it was only wired at the boundary
  hook, untested, and its "note" step runs after a successful evaluation only. Re-check against the real
  root cause (dedup compares learning-demand `source_route_id` with compiler status) before reusing.

## Checkpoint 2 (15:15-15:30): root causes confirmed on the run data, both fixes implemented in root/

Run data (local copies; scripts in `scratch/`):
- Problem 1 (`verdicts.py`, `policies.py` on coherentRP): 005 (gen 9) and 007 (gen 15) have the same desktop placement
  (`5bfd230a`), the same whole-layout geometry (`0249915b`, 24 Gemma layers) and the same P100 policy identity
  (columns 15360, mask 0xFFFFFF, coalesced executor, same resources); only the policy *hash* differs. 007's group:
  owner 007, `verdicts {}` -> INSUFFICIENT_OPPORTUNITY x7. Qwen 002/004 (gen 6) and 006 (gen 12) likewise share
  geometry `1ff4e78b` (17 layers) -> 006 re-decided verdict[1]. allon run: same storm and same resets.
- Per-request history (`history._group_matches_session`) matches on whole-layout geometry, not generation: no
  generation reset there (007's `historical_groups 0` = context bucket 8 vs 005's bucket 12 plus 001's
  ineligible windows). Left unchanged (see README).
- Problem 2: storm reproduced through the real boundary hook in the portfolio test harness
  (`scratch/storm_repro.py`): base, knob on: 60 boundaries -> 60 REPROVISION_RETAINED. Trigger confirmed: the
  base dedup compares the recorded learning status `source_route_id` with the compiler's UNUSABLE status (no
  source_route_id) -> never equal. Knob off in the same harness: 60 LEARNING_RETAINED on base (latent base
  defect; left unchanged, knob-off must stay identical).

Implemented (root/):
- `phone_shards.artifact_layout_identity_sha256`; `_AdaptiveSession.helper_layout_identity_sha256`,
  `_AdaptiveServerPolicy.layout_identity_sha256`; `server_policy_key` = (model, placement, None, identity) when
  the identity is present (4-tuple kept: checkpoint restore checks len 4), else the old (gen, geometry) key;
  kwargs on start / helper_ready / helper_rebound (identity fixed at attachment like the generation); unified
  layer derives it only with `server_policy_coherence` (else None -> records unchanged); decision records:
  `server_policy.layout_identity_sha256`, snapshot `helper_layout_identity_sha256`, ASSISTANCE_DECISION field
  only when not None.
- `boundary_reevaluation_interval_us` knob (default 10 s, 0 = every boundary); `_boundary_reevaluation_due`
  (state signature: non-terminal tickets + dispatch/transition state, ready/target generations, session
  states/generations/resident model/refs, in-use sessions, work buckets, route evidence, uncovered set) placed
  before the base dedup (only removes evaluations vs base); `_coalesce_unchanged_decision` drops a RETAINED/HOLD
  record equal to the last EVALUATED record; counters (`boundary_evaluations_skipped`,
  `unchanged_decisions_coalesced`, + `_total`) go into the next record.
- Root storm repro: knob on 60 boundaries -> 1 event; knob off -> 60 LEARNING_RETAINED (unchanged).

## Checkpoint 3 (15:30-15:40): tests, guards, diff

- New tests: `tests/test_adaptive_layout_identity.py` (10; base: 8 error on the missing identity API, the
  2 unchanged-behaviour tests pass), `tests/test_phone_reprovision_boundary.py` (6; base: 5 fail on the
  storm, knob-off guard passes); `test_phone_resident_model_reprovision.py` config test updated (30 OK).
- Option-off digest guard (`scratch/option_off_digest.py`, base vs root): knob-off events (65) and
  coherence-off directives + ASSISTANCE_DECISION payloads (6) byte-identical digests.
- Harness timing (`scratch/storm_timing.py`, 600 boundaries): base 1.84 ms/boundary, 501 records; root
  0.11 ms/boundary, 10 records.
- Offline gate replay on recorded streams (`scratch/gate_replay.py`): coherentRP 1,202 RETAINED -> ~73
  evaluations / ~33 records; allon 1,224 -> ~68 / ~32. RETAINED records are 19.9 MB of RESULT.json.
- Identity on the real RP layouts (`scratch/identity_check.py`): Gemma 24 gens 3/9/15 -> one identity;
  Qwen 17 gens 6/12 -> one identity; intermediate 16/8-layer stages recur too (gens 4/8/10, 5/7/11).
- `make_diff.sh` -> REPROVISION_FIXES.diff (17 files, +686/-13); `git apply --check` from the repo root OK.
- pyflakes (package minus reports): clean on root and base. Full suite running (`tests_root.log`).

## Checkpoint 4 (15:40-15:50): first full run, one fix, second full run

- Full run 1 (`tests_root_run1.log`): 2 failing files. `test_resident_router_subset.py` (environmental, needs
  `examples/`); `test_session_cow_transaction.py` 2 errors from my change: tests use SimpleNamespace tickets
  without `model`, a bare `object.__new__(UnifiedScheduler)` without `_adaptive_decode_config`, and an exact
  `assert_called_once_with` on `helper_ready` / `helper_rebound`.
- Fix: `_adaptive_layout_identity_sha256(ticket, generation)` checks coherence first (getattr on the config)
  and only then reads `ticket.model`; `helper_ready` / `helper_rebound` get `phone_layout_identity_sha256`
  only when it is set (calls unchanged with coherence off). COW, runtime, coherence, decode, server-probe,
  evidence-fix, reprovision and new test modules OK; digest guard unchanged; base check of the new files
  unchanged (8 errors / 10 failures, 2 + 1 guards pass).
- Diff rebuilt, `git apply --check` OK. Full run 2 started (`tests_root.log`).

## Checkpoint 5 (15:50-16:00): final

- Full run 2 (`tests_root.log` / `tests_root.json`): 110 files, 1,654 tests, only `test_resident_router_subset.py`
  fails (environmental). Flaky/admission, kv-touch, split-kv (module) passed; two-phone native skipped.
- pyflakes clean; diff all ASCII; main unchanged for every touched file; `git apply --check` OK (stdin closed).
- Deliverables copied to reports/20260924-phone-reprovision/followup-fixes/ (README.md, REPROVISION_FIXES.diff,
  PROGRESS.md, tools/); the superseded REPROVISION_FIXES_PARTIAL.diff was removed there (copy kept in this
  directory as `REPROVISION_FIXES_PARTIAL.diff`).
