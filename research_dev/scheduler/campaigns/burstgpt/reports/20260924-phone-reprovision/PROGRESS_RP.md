# PROGRESS_RP: change #4 on the rig (dev_v2, coherentRP vs a coherentEF repeat)

Checkpoint file for a restartable run. Newest stage last. Times are desktop local (EDT) unless marked UTC.

## Plan

- Arms (inputs on the desktop, derived by `derive_inputs_rp.sh` = `derive_inputs_ef.sh dev2coherentEF` flags):
  - `dev2coherentRP` = `/home/zhihao/s42-trace-longtaildev2-coherentRP-20260924-inputs`: coherentEF +
    campaign `"phone_resident_model_reprovisioning": {}` (defaults: prior 200 MB/s, 2 learned samples);
  - `dev2coherentEF2` = `/home/zhihao/s42-trace-longtaildev2-coherentEF2-20260924-inputs`: an exact repeat of
    coherentEF (only the campaign id differs) = run-to-run noise.
  Both from `/home/zhihao/s42-trace-longtail-treatment-20260923-inputs` + trace `/mnt/storage/burstgpt-source/longtail_dev_v2`
  + receipts `s43-transport-receipts-61440-20260924/receipts-all`, identity COALESCED_BOTH (`s43-coalesced-both-20260924`),
  boot f13c7c03, `server_policy_coherence` on, coalesced-batch qualified for both models.
- Tooling (this dir; desktop copies in `/home/zhihao/s42-reprovision-20260924-rig/`): `derive_inputs_rp.sh` (outside the
  lock), `run_arms_rp.py` (flock -w 7200 on the rig lock: sync staging -> deploy, per arm preflight, `check_admission_both.py`,
  run; battery + notify before/after every stage, STOP on 512; CANCEL file in an input dir stops the chain),
  `analyze_reprovision.py` (re-provisioning timeline). Energy/window analysis reuses `../20260924-coherent-policy-coalesced/`
  `analyze_ef.py`, `tables_ef.py`, `phase_energy.py` and the rig's `analyze_longdecode_pair.py`.
- Staging: `/home/zhihao/s42-reprovision-20260924-staging/research_dev/scheduler/` (rsync -rc of the main tree, excludes
  __pycache__/reports/*.pyc). Status: `/home/zhihao/s42-trace-longtaildev2-RP-20260924-STATUS.jsonl`.

## Stage 0 (11:50-12:10 EDT): main tree checked, flag added, nothing on the rig touched

- Main tree has change #4 (`_unified/phone_residency_ops/reprovision.py`, launch/runner/arguments plumbing, campaign field)
  and the inert two-phone code. `prepare_trace_inputs_v2.py` had no way to set the field -> added
  `--phone-resident-model-reprovisioning-json JSON` (JSON object -> campaign `phone_resident_model_reprovisioning`, refused
  with `fixed_phone_residency`, recorded in CHANGES.txt) + test
  `test_phone_resident_model_reprovisioning_is_declared_and_exclusive_with_fixed_residency`.
- Local tests: test_prepare_trace_inputs_v2 5 OK, test_phone_resident_model_reprovision 30 OK,
  test_phone_reprovision_portfolio 7 OK, test_adaptive_evidence_fixes 12 OK, test_adaptive_coherence 30 OK,
  test_campaign_inputs 9 OK, test_campaign_model_configuration 3 OK, test_runner_rejections 2 OK,
  test_remote_resident_launch 12 OK; pyflakes clean on the two edited files.
- Deploy source unchanged since the EF sync (no file newer than `SYNC_EF-1.txt`). Main tree vs deploy: exactly 28 files
  differ (rsync -n -rc), manifest `SYNC_MANIFEST_RP.sha256`: #4 (reprovision.py new; phone_residency.py,
  runtime_requests.py, helper_preparation_ops/start.py, phone_residency_ops/{common,economics,portfolio}.py, config.py,
  scheduler.py, configuration/campaign.py, campaigns/burstgpt/{arguments,launch,runner}.py, README.md, 2 new tests),
  two-phone (adapters/{phone_helpers,phone_tcp_session}.py new, llama_server_contracts.py, phone_transport.py,
  configuration/{models,rig}.py, campaigns/burstgpt/{preflight,two_phone_gate}.py, 2 new tests; all gated on
  `phone_helpers` / `helper_phones` / `adb-tcp`, none present in these inputs), prepare_trace_inputs_v2.py + its test.
- Rig idle (no launch/runner/llama-server processes; lock not held). C++ server binary not rebuilt (not needed).

## Stage 1 (11:55-12:00 EDT): staged + derived, rig untouched

- Staging `/home/zhihao/s42-reprovision-20260924-staging/research_dev/scheduler/` = local main tree (rsync -rc dry run: 0
  differences); manifest files all OK; staging vs deploy source = exactly the 28 manifest files. Tests in staging on the
  desktop python 3.14: prepare 5, reprovision 30, reprovision portfolio 7, evidence fixes 12, coherence 30, campaign
  inputs 9 OK (the two reprovision files need `PYTHONPATH=<deploy>/source/gguf-py`: their sibling-test import chain reaches
  test_gguf_cost -> `gguf`, an environment gap of the staging copy, not a code issue).
- Derived (`derive_inputs_rp.sh dev2coherentRP dev2coherentEF2`, from staging): identity e95f0b94... (15 receipts,
  s43-coalesced-both-20260924), boot f13c7c03 in both rig.json. Diffs vs `...-coherentEF-20260924-inputs` with the arm name
  masked: coherentRP = only campaign `"phone_resident_model_reprovisioning": {}` (+ CHANGES.txt line); coherentEF2 = none
  (campaign.json, rig.json, models.json, evidence.json, identity, materialize command all identical).
- Rig idle 12:00 EDT, OP15 present on adb 5037, notify code 0, GPU 1.26 GB used / 7 W.
- Next: `nohup python3 run_arms_rp.py dev2coherentRP dev2coherentEF2` (flock inside). Stdout
  `/home/zhihao/s42-reprovision-20260924-rig/RUN_ARMS_RP-1.log`.

## Stage 2 (16:00-16:08 UTC): lock held, deploy synced, coherentRP preflight + admission PASS, run started

- `run_arms_rp.py` launcher PID 2653392 (flock 2653393, locked python 2653394), lock ACQUIRED 16:00:39 UTC.
  Battery before sync: level 80, status 4 (not charging), USB, 500 mA, temp 26.2 C, notify 0.
- Sync DONE 16:01:33 UTC: exactly the 28 manifest files changed, deploy digests = manifest, and a second rsync dry run
  staging -> deploy found no difference (`SYNC_RP-1.txt` in the rig dir).
- coherentRP physical preflight 16:01:33 -> 16:07:42 UTC exit 0 (PASS); battery level 80, status 4, notify 0.
- Admission (`check_admission_both.py`) PASS, same as coherentEF: identity s43-coalesced-both-20260924 (15 receipts, catalog
  link identity `sha256:866f628e...`), one QUALIFIED coalesced-batch helper per desktop parent for both models
  (hot:cpu, hot:desktop, cold:cpu, cold:desktop; split-row SHADOW); Qwen payload-40960, Gemma payload-61440, Gemma
  catalog parallel=2 -> payload-38400.
- coherentRP run STARTED 16:07:42 UTC -> `.../coherentRP-20260924-inputs/run-dev2coherentRP-1/run/`; the runner command
  carries `--phone-resident-model-reprovisioning-json {"load_bytes_per_second":200000000,"minimum_learned_samples":2,
  "mode":"resident-model"}`.

## Stage 3 (16:23-16:29 UTC): coherentRP run PASS; coherentEF2 preflight + admission PASS, run started

- coherentRP run 16:07:42 -> 16:23:07 UTC exit 0, RESULT.json PASS 9/9 (18 attempts, 0 rejected). Battery after: level 79,
  status 2 (charging), temp 34.3 C, notify 0.
- First numbers (local copy, snapshots excluded, copied during the EF2 preflight, which measures nothing): 898.1 s, host
  53.69 kJ (26.03 CPU + 27.65 GPU) = -44.5 % vs dev2base, -25.3 % vs coherentEF (71.86 kJ), -29.6 % vs plainEF.
  15 layout generations, 15 session loads (127-281 MB/s, learned rate 161 -> 217 MB/s), every desktop switch followed:
  Qwen 17 layers (HTP0 6 + HTP1 6 + HTP2 5: the live limit kept the third Qwen session at 5 layers), Gemma 24 layers.
  Phone ready before the desktop finished loading in all 4 switches (Qwen 002: dispatch 89.4 -> 17 layers 133.1, desktop
  execution 162.4; Gemma 005: 356.2 -> 391.3 / 403.1; Qwen 006: 675.5 -> 721.2 / 759.8; Gemma 007: 838.1 -> 870.2 / 879.6).
  0 WAITING_FOR_HELPER_RELEASE, 0 preparation/transition failures. Qwen phone share 87.7 % (EF 39.3 %), Gemma 81.9 % (EF
  87.3 %; Gemma 007 INSUFFICIENT_OPPORTUNITY on the new layout generation, 0 phone tokens).
- coherentEF2 preflight 16:23:07 -> 16:28:28 UTC exit 0; admission PASS, identical to coherentRP/coherentEF (identity
  s43-coalesced-both-20260924, catalog link `sha256:866f628e...`, one QUALIFIED coalesced-batch helper per parent, both
  models). Battery level 79, status 2, notify 0.
- coherentEF2 run STARTED 16:28:29 UTC -> `.../coherentEF2-20260924-inputs/run-dev2coherentEF2-1/run/`. Nothing is read
  from the desktop until it ends.

## Stage 4 (16:46 UTC): coherentEF2 run PASS, chain done, lock released

- coherentEF2 run 16:28:29 -> 16:46:22 UTC exit 0, RESULT.json PASS 9/9; ARMS_DONE 16:46:22; no chain, launch, runner or
  llama-server process left, lock not held. Battery at end: level 78, status 2, USB 500 mA, temp 33.7 C, notify 0 (never
  512 during the chain).
- coherentEF2: 1,042.8 s, host 69.64 kJ (39.91 CPU + 29.73 GPU) = -3.1 % kJ / +12.8 % s vs coherentEF; 9/9 identical to
  dev2base and to coherentEF.

## Stage 5 (12:50-13:20 EDT): analysis done, RIG_RESULTS.md written. DONE.

- Off the desktop on copies (snapshots excluded). Outputs in `data-rp/` (ANALYSIS_RP.json, TABLES_RP.md,
  REPROVISION_RP.{json,md}, PHASE_ENERGY_RP.jsonl, PHASE_BATCH_RP.jsonl, PAIR_*.json, ADMISSION_*.json, STATUS.jsonl,
  SYNC_RP-1.txt, SYNC_MANIFEST_RP.sha256); desktop copies in `/home/zhihao/s42-reprovision-20260924-rig/results/`.
  New scripts: `analyze_reprovision.py`, `phase_energy_batch.py`.
- Headline: coherentRP 53.69 kJ / 898.1 s (-25.3 % vs coherentEF, -22.9 % vs coherentEF2, -44.5 % vs dev2base); Qwen 17 /
  Gemma 24 phone layers after every switch, all 4 swaps verified inside the desktop load window; exactness 6/9 (003 @103,
  004 @57 new, 005 @146). Noise: coherentEF vs coherentEF2 3.1 % host energy, 12.8 % duration (cold-page-cache Gemma
  load in EF2), Qwen verdicts inverted between the two runs.
- Findings for #4 (not fixed): server verdicts reset at every layout generation (Gemma 007 lost the phone,
  INSUFFICIENT_OPPORTUNITY); per-token REPROVISION_RETAINED re-evaluation (1,202 events) while the other model is queued;
  pessimistic first-swap estimate; 17 not 18 Qwen layers (live limit, no regrowth).
- No commits.
