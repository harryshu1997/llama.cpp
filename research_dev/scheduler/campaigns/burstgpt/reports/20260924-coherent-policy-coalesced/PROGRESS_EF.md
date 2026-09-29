# PROGRESS_EF: dev_v2 A/B after the evidence fixes (plainEF vs coherentEF)

Checkpoint file for a restartable run. Newest stage last. Times are desktop local (EDT) unless marked UTC.

## Plan

- Arms (inputs on the desktop): `dev2plainEF` = `/home/zhihao/s42-trace-longtaildev2-plainEF-20260924-inputs`,
  `dev2coherentEF` = `/home/zhihao/s42-trace-longtaildev2-coherentEF-20260924-inputs`. Both derived by the current main
  tree's `prepare_trace_inputs_v2.py` from `/home/zhihao/s42-trace-longtail-treatment-20260923-inputs` + dev_v2 trace +
  `--transport-receipts-dir /home/zhihao/s43-transport-receipts-61440-20260924/receipts-all` + candidate boot f13c7c03,
  identity = `/mnt/storage/s42-trace-v2-20260921-prep/TRANSPORT_QUALIFICATION_IDENTITY_COALESCED_BOTH.json`
  (file sha256 e95f0b94..., id s43-coalesced-both-20260924). coherentEF adds `--adaptive-decode-overrides-json
  '{"server_policy_coherence": true}' --qualify-phone-batch-plan hot=coalesced-batch --qualify-phone-batch-plan
  cold=coalesced-batch`.
- Tooling (local copies in this dir, desktop copies in `/home/zhihao/s42-coherent-20260924-rig/`):
  `derive_inputs_ef.sh` (outside the lock, writes only the two input dirs), `run_arms_ef.py` (under
  `flock -w 7200 <rig lock>`: sync staging -> deploy source, then per arm physical preflight, admission check
  (plain: `check_admission.py split-row`; coherent: `check_admission_both.py`), run; battery + notify code before/after
  every stage, STOP on 512; CANCEL file in an input dir stops the chain before its next stage), `analyze_ef.py`.
- Staging dir for the sync: `/home/zhihao/s42-coherentEF-20260924-staging/research_dev/scheduler/` (rsync -rc of the
  local tree, same excludes as the deploy sync). Chain status: `/home/zhihao/s42-trace-longtaildev2-EF-20260924-STATUS.jsonl`.

## Stage 0 (2026-09-24 10:50 EDT): state checked, nothing on the rig touched

- Local tests: test_adaptive_evidence_fixes 12 OK, test_adaptive_coherence 30 OK, test_prepare_trace_inputs_v2 4 OK.
- Deploy source differs from the local tree in exactly 15 files (rsync -n -rc): the adaptive_decode files
  (adaptive_decode.py, _contracts, _state, ops/{budgeting,helpers,promotion,reporting,sequencing,windows}),
  _unified/adaptive_decode_control.py, campaigns/burstgpt/{build_realistic_trace,prepare_trace_inputs_v2}.py,
  tests/{test_adaptive_evidence_fixes (new),test_build_realistic_trace,test_prepare_trace_inputs_v2}.py. Deploy has
  no CANDIDATE_REQUALIFIED / TOKEN_STREAM_CATCH_UP / --allow-mixed-phone-batch-plans.
- Rig idle (no launch/runner/llama-server processes), lock file present, not held.
- `run_arms_next.py` named in the task does not exist anywhere on the desktop; `run_arms_ef.py` is new.

## Stage 1 (11:05 EDT): staged + derived, rig untouched

- Staging `/home/zhihao/s42-coherentEF-20260924-staging/research_dev/scheduler/` = local tree (rsync -rc, excludes
  __pycache__/reports/*.pyc); staging vs deploy source: exactly the 15 manifest files
  (`/home/zhihao/s42-coherent-20260924-rig/SYNC_MANIFEST_EF.sha256`, sha256 per path of the local tree). The 3 test
  files pass in staging on the desktop python (12 / 30 / 4 OK).
- Tooling on the desktop rig dir: `derive_inputs_ef.sh`, `run_arms_ef.py`, `analyze_ef.py`, `check_admission_both.py`
  (copied from `/home/zhihao/s43-coalesced-both-20260924-rig/`). analyze_ef.py dry-run on the prior dev2 arms OK
  (`results/ANALYSIS_EF_priorarms.json`).
- Derived (derive_inputs_ef.sh from staging): `...-plainEF-20260924-inputs`, `...-coherentEF-20260924-inputs`; identity
  file sha256 e95f0b94... (15 receipts), boot f13c7c03 in both rig.json. Diffs: plainEF vs coherentEF = only
  `adaptive_decode_overrides.server_policy_coherence` + both models' `qualified_phone_batch_plans` split-row ->
  coalesced-batch (+ campaign id); coherentEF vs the offline-admitted coalescedboth inputs = campaign id only; plainEF vs
  the earlier dev2plain = receipts dir only (receipts-all vs task1 09-22) + identity file.
- Next: `nohup python3 run_arms_ef.py dev2plainEF dev2coherentEF` (flock inside) -> sync, plainEF preflight/admission/run,
  coherentEF preflight/admission/run. Status: `/home/zhihao/s42-trace-longtaildev2-EF-20260924-STATUS.jsonl`,
  stdout `/home/zhihao/s42-coherent-20260924-rig/RUN_ARMS_EF-1.log`. Restart: if a stage failed, fix, then
  `ARM_ATTEMPT=2 python3 run_arms_ef.py <remaining arms>` (the sync is idempotent; set REUSE_PREFLIGHT=1 only
  within the same lock hold).

## Stage 2 (10:52 EDT = 14:52 UTC): chain launched, lock held, deploy synced

- `run_arms_ef.py` launcher PID 2622143 (flock child 2622145, locked python 2622146), lock ACQUIRED 14:52:51 UTC.
- Battery before sync: level 80, status 4 (not charging), USB powered, max charging current 500 mA, temp 25.4 C,
  notify 0.
- Sync DONE 14:52:53 UTC: exactly the 15 manifest files changed; deploy digests = manifest afterwards
  (rsync output `/home/zhihao/s42-coherent-20260924-rig/SYNC_EF-1.txt`).
- plainEF physical preflight STARTED 14:52:54 UTC -> `.../plainEF-20260924-inputs/preflight-1`.

## Stage 3 (14:59 UTC): plainEF preflight PASS, admission PASS, run started

- plainEF physical preflight 14:52:54 -> 14:59:05 UTC exit 0 (catalog `sha256:e61a682f...`); battery before/after:
  level 80, status 4, notify 0.
- Admission (`check_admission.py ... split-row`, cwd/PYTHONPATH = deploy source) PASS: one QUALIFIED split-row
  `operator_split` helper per desktop parent for both models (hot:cpu, hot:desktop, cold:cpu, cold:desktop), the
  coalesced variants SHADOW (`ADMISSION-1.log`, `TRANSPORT_ADMISSION-preflight-1.json` in the inputs dir).
- plainEF run STARTED 14:59:06 UTC -> `.../plainEF-20260924-inputs/run-dev2plainEF-1/run/`.

## Stage 4 (15:17 UTC): plainEF run PASS; coherentEF preflight started

- plainEF run 14:59:06 -> 15:16:52 UTC, exit 0, RESULT.json PASS 9/9. Battery after: level 79, status 2 (charging),
  notify 0.
- First numbers (local copy of the artifacts, snapshots excluded, in the session scratchpad; analysis is run off the
  desktop so it does not load the measured host during the coherentEF run): 1,035.1 s, host 76.31 kJ
  (46.37 CPU + 29.93 GPU) = -21.1 % vs dev2base (96.75 kJ), -6.7 % vs the pre-fix dev2plain (81.81 kJ).
  Qwen phone-policy share 278/595 (46.7 %, was 18.8 %), Gemma 692/797 (86.8 %). Qwen server: 121 mixed passes, all
  1,692 Qwen and 11,760 Gemma phone calls 1-row. Outputs 6/9 identical vs dev2base (Qwen 003 @103, Qwen 004 @210,
  Gemma 005 @146). Evidence-fix markers: HELPER_PHONE_SESSION_LOAD 16 decisions (Gemma 000), TOKEN_STREAM_CATCH_UP 1
  window, CANDIDATE_REQUALIFIED 0. Qwen 002 and 006 now end on the phone.
- coherentEF physical preflight STARTED 15:16:52 UTC (battery level 79, status 2, notify 0).

## Stage 5 (15:22 UTC): coherentEF preflight PASS, admission PASS, run started

- coherentEF physical preflight 15:16:52 -> 15:22:25 UTC exit 0; battery level 80, status 2, notify 0.
- Admission (`check_admission_both.py`) PASS, identity s43-coalesced-both-20260924 (15 receipts, catalog link identity
  `sha256:866f628e...`): one QUALIFIED `operator_split:coalesced-batch` helper per desktop parent for BOTH models
  (hot:cpu, hot:desktop, cold:cpu, cold:desktop; split-row variants SHADOW). Qwen 4 rows = 40,960 B -> capacity
  payload-40960; Gemma models.json parallel 8 = 61,440 B -> payload-61440; Gemma helpers carry the catalog
  `parallel=2` -> 2-row call 15,360 B (capacity payload-38400).
- coherentEF run STARTED 15:22:26 UTC -> `.../coherentEF-20260924-inputs/run-dev2coherentEF-1/run/`. No analysis on
  the desktop until it ends (host energy is measured).

## Stage 6 (15:38 UTC): coherentEF run PASS, chain done, lock released

- coherentEF run 15:22:26 -> 15:38:20 UTC exit 0, RESULT.json PASS 9/9; ARMS_DONE 15:38:20; no chain processes left.
  Battery at end: level 79, status 2, USB powered, 500 mA, temp 32.5 C, notify 0 (never 512 during the chain).
- coherentEF: 924.6 s, host 71.9 kJ (43.5 CPU + 28.4 GPU): -25.7 % vs dev2base, -5.8 % vs plainEF.
- Next: full analysis off the desktop (compare_trace_energy, analyze_longdecode_pair x3, analyze_ef), results appended
  to README section 4.4; desktop copies of the result JSONs in `/home/zhihao/s42-coherent-20260924-rig/results/`.

## Stage 7 (11:55 EDT): analysis done, README section 4.4 written. DONE.

- Analysis ran off the desktop on copies (snapshots excluded). Outputs in `data-ef/` (ANALYSIS_EF.json, ENERGY_EF.json,
  PAIR_*.json, PHASE_ENERGY_EF.jsonl, TABLES_EF.md, ADMISSION_*.json, STATUS.jsonl); desktop copies in
  `/home/zhihao/s42-coherent-20260924-rig/results/`. New scripts: `analyze_ef.py`, `tables_ef.py`, `phase_energy.py`.
- Headline: coherentEF 71.86 kJ / 924.5 s (-25.7 % / -19.9 % vs dev2base; -5.8 % / -10.7 % vs plainEF 76.31 kJ /
  1,035.1 s); coherentEF 9/9 outputs identical to dev2base, plainEF 6/9. Qwen pair: 660 two-row calls, 1 mixed pass
  (plainEF 121), pair phase 7.62 kJ / 72.7 s vs 13.36 kJ / 113.0 s. New defect: a single-window batch-1 pair
  (15.4 % saving < the 19 % n=1 threshold) became a final `verdict[1]=host` (SERVER_PAIR_NOT_IMPROVED), costing about
  +3.4 kJ of Qwen batch-1 decode (002 tail, 004 tail, all of 006 on the host).
- Nothing left running on the desktop (chain exited, lock released, monitors stopped, no remote tails). No commits.
