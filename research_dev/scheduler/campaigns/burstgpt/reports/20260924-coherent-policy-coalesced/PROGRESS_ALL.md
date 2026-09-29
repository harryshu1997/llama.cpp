# PROGRESS_ALL: dev_v2 "all on" A/B under the new dispatcher (dev2baseDP vs dev2allon) + Gemma 26-layer shards

Checkpoint file for a restartable run. Newest stage last. Times UTC unless marked.

## Plan

- Arms (inputs on the desktop, derived by `derive_inputs_allon.sh` from the CURRENT main tree's
  `prepare_trace_inputs_v2.py`, trace `/mnt/storage/burstgpt-source/longtail_dev_v2`):
  - `dev2baseDP` = `/home/zhihao/s42-trace-longtaildev2-baseDP-20260924-inputs`: the 09-23 long-tail baseline inputs
    (desktop-baseline selection, deploy transport identity copied unchanged, exactly like dev2base) + campaign
    `"dispatch_policy": {"work_conserving_admission": true, "model_affinity": true}` = the new desktop-only reference;
  - `dev2allon` = `/home/zhihao/s42-trace-longtaildev2-allon-20260924-inputs`: the coherentRP configuration
    (09-23 treatment inputs, energy-aware, `server_policy_coherence`, coalesced-batch qualified for both models,
    `phone_resident_model_reprovisioning {}`, identity `s43-coalesced-both-20260924`, receipts-all, boot f13c7c03)
    + the same `dispatch_policy`.
- Tooling (this dir; desktop copies in `/home/zhihao/s42-allon-20260924-rig/`): `derive_inputs_allon.sh` (outside
  the lock, writes only the two input dirs), `run_arms_allon.py` (`flock -w 14400` on the rig lock: manifest-checked
  sync staging -> deploy source, per arm physical preflight, admission check (`check_admission.py ... split-row` for
  baseDP, `check_admission_both.py` for allon), run; then under the same lock hold the Gemma 26-layer shard
  generation on `/mnt/storage`, local sha256 verify, `df /data`, push to a NEW phone dir, sha256 verify on the phone;
  `dumpsys battery` + notify code before/after every stage, STOP on 512; CANCEL file in an input dir stops the chain),
  `analyze_allon.py` (extends `analyze_ef.py` with dispatch-policy stats, loads, pairs, probe reasons; reuses
  `../20260924-phone-reprovision/analyze_reprovision.py` for the re-provisioning timeline).
- Staging on `/mnt/storage/s42-allon-20260924-staging/research_dev/scheduler/` (NVMe nearly full); rsync -rc of the
  local main tree, excludes `__pycache__`, `reports`, `*.pyc`. Status: `/home/zhihao/s42-trace-longtaildev2-ALLON-20260924-STATUS.jsonl`.
- Rules: never commit/push; OP15 only via `adb -P 5037 -s 3C15AU002CL00000`; the RP chain
  (`/home/zhihao/s42-trace-longtaildev2-RP-20260924-STATUS.jsonl`) holds the lock -> queue behind it; kill only own PIDs.

## Stage 0 (16:29-16:45): state checked, flag added to the MAIN tree, nothing on the rig touched

- Rig lock held by the RP chain (flock PID 2653393 / python 2653394): `dev2coherentRP` run PASS 16:07:42-16:23:07
  (898.1 s, host 53.69 kJ = 26.03 CPU + 27.65 GPU, first numbers read from its RESULT.json); `dev2coherentEF2`
  preflight PASS 16:23-16:28, admission PASS, run STARTED 16:28:29 -> expected to end about 16:45.
- Main tree state: dispatcher merge present (`_internal/runtime_dispatch_policy.py`, `_unified/automated_requests_ops/affinity.py`,
  campaign field `dispatch_policy`, launch/arguments/runner plumbing, `tests/test_dispatch_policy.py`), re-provisioning
  (#4) present, server-probe fix present (`coherence.py`: `SERVER_PAIR_INCONCLUSIVE`, `SERVER_REFERENCE_BASELINE`,
  `CO_TENANT_POLICY_FOLLOW`; `tests/test_adaptive_server_probe.py`). `prepare_trace_inputs_v2.py` had no way to
  declare `dispatch_policy` -> added `--dispatch-policy-json JSON` (non-empty JSON object -> campaign `dispatch_policy`,
  recorded in CHANGES.txt; mirrors `--phone-resident-model-reprovisioning-json`) + test
  `test_dispatch_policy_is_declared_and_validated_by_the_campaign_manifest` (the campaign validator `_dispatch_policy`
  accepts the declared object; `[]` and `{}` refused). Local: test_prepare_trace_inputs_v2 6 OK; test_dispatch_policy +
  test_adaptive_server_probe + test_adaptive_coherence 61 OK; pyflakes clean on the two edited files; ASCII only.
- Deploy source = RP sync (28 files) on top of the EF sync; no file newer than `SYNC_RP-1.txt`. Main tree vs deploy
  therefore differs by: dispatcher merge (18 files), server-probe fix (3 files + new test), the prepare flag + test,
  and whatever else the coordinator merged since; the manifest is computed at staging time (stage 1).
- Desktop: `/` 90 GB free (82 % used), `/mnt/storage` 3.1 TB free. Gemma parent model present
  (`/home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf`, 23.8 GB); `gguf-py` and `native/ffn_shard_gguf.py` in the deploy source.
- Coordinator FYI (16:40): since the RP sync the MAIN tree also gained two merged, inert-by-default changes: the
  server-probe fix (`coherence.py` `_server_pair_next_measurement`, per-batch reasons, `CO_TENANT_POLICY_FOLLOW`,
  `tests/test_adaptive_server_probe.py`) and the two-phone gaps 1-3 (`plan_contracts/co_helpers.py`,
  `adapters/co_helper_lifecycle.py`, catalog materialization / route generation / adaptive policy `device_layer_masks`,
  preflight BLOCKED rows, `tests/test_two_phone_gaps.py`, `two_phone_harness.py`). Neither activates without a
  co-helper declaration, so they do not affect these arms; the sync copies about 40 more files than the RP chain's
  28 -> expected in the itemized rsync list, hashes verified against the manifest as planned.
- Next: wait for the EF2 run to end (no analysis/tests on the desktop while it measures host energy), then stage +
  derive + local test run in staging (stage 1), then launch the chain (stage 2).

## Stage 1 (16:46-16:52): RP chain done, staged + manifest + derived, rig untouched

- RP chain ARMS_DONE 16:46:22 (`dev2coherentEF2` run exit 0; battery level 78, notify 0), lock released, no chain
  processes left. Reference artifacts (dev2base, plainEF, coherentEF, coherentRP, coherentEF2; snapshots excluded)
  copied to the local scratchpad for the off-desktop analysis.
- Staging `/mnt/storage/s42-allon-20260924-staging/research_dev/scheduler/` seeded by `cp -a` from the RP staging,
  then `rsync -rc` from the local main tree (47 files changed; a second dry run local -> staging: 0 differences).
- Manifest `/home/zhihao/s42-allon-20260924-rig/SYNC_MANIFEST_ALLON.sha256`: exactly 47 files differ between staging
  and the deploy source (sha256 of the staged file per path; `SYNC_FILES_ALLON.txt`, `SYNC_DRYRUN_ALLON.txt`):
  dispatcher #2/#5 (18: runtime_dispatch_policy.py, runtime_queue.py, runtime_controller.py + ops/{admission,dispatch},
  automated_requests_ops/{affinity,observations,replan,selection}, automated_selection_ops/resources, runtime_requests,
  configuration/campaign, campaigns/burstgpt/{arguments,launch,runner}, README, __init__, tests/test_dispatch_policy),
  server-probe fix (adaptive_decode_ops/coherence.py, adaptive_decode_state.py, tests/test_adaptive_coherence.py,
  tests/test_adaptive_server_probe.py), two-phone gaps 1-3 (plan_contracts/co_helpers.py, adapters/co_helper_lifecycle.py,
  adapters/{__init__,catalog_materialization,contracts,llama_server_contracts,residency,ticket}, llama_server_ops/proofs,
  capability_contracts/catalog, route_generation/{costing_parameters,envelopes,patterns}, adaptive_decode_{contracts,planning},
  automated_selection_ops/{attachment,dormant}, _unified/common, campaigns/burstgpt/{catalog,preflight},
  tests/{test_two_phone_gaps,test_two_phone_helpers,two_phone_harness}), and the prepare flag + test
  (campaigns/burstgpt/prepare_trace_inputs_v2.py, tests/test_prepare_trace_inputs_v2.py). C++ server NOT rebuilt.
- Tests in staging on the desktop python 3.14 (`PYTHONPATH=staging:<deploy>/source/gguf-py`): prepare 6, dispatch_policy 22,
  adaptive_server_probe 9, adaptive_coherence 30, adaptive_evidence_fixes 12, phone_resident_model_reprovision 30,
  phone_reprovision_portfolio 7, two_phone_gaps 29, campaign_inputs 9, campaign_model_configuration 3: all OK.
- Derived (`derive_inputs_allon.sh dev2baseDP dev2allon`): `s42-trace-longtaildev2-baseDP-20260924-inputs`
  (desktop-baseline, deploy identity `s42-trace-v2-20260921-prep` 6 receipts, file sha256 82617a5c..., boot 26e8d418...)
  and `s42-trace-longtaildev2-allon-20260924-inputs` (energy-aware, identity `s43-coalesced-both-20260924` 15 receipts,
  file sha256 e95f0b94..., boot f13c7c03). Diffs with arm paths masked: baseDP vs dev2base = `campaign_id` +
  `dispatch_policy` only (rig/models/evidence/identity identical); allon vs coherentRP = `campaign_id` + `dispatch_policy`
  only. `dispatch_policy` = `{"model_affinity": true, "work_conserving_admission": true}` in both.
- Next: `nohup python3 -u run_arms_allon.py dev2baseDP dev2allon --shards` (flock -w 14400 inside). Status
  `/home/zhihao/s42-trace-longtaildev2-ALLON-20260924-STATUS.jsonl`, stdout `/home/zhihao/s42-allon-20260924-rig/RUN_ARMS_ALLON-1.log`.
  Restart: fix, then `ARM_ATTEMPT=2 python3 run_arms_allon.py <remaining arms> [--shards]` (sync idempotent;
  REUSE_PREFLIGHT=1 only within the same lock hold).

## Stage 2 (16:51 UTC): chain launched, lock held, deploy synced, baseDP preflight started

- `run_arms_allon.py dev2baseDP dev2allon --shards` launcher PID 2702345 (python 2702347, flock 2702348, locked python
  2702349); lock ACQUIRED 16:51:02 UTC (rig was idle: no launch/runner/llama-server processes, lock free).
- Battery before sync: level 78, status 2 (charging), USB powered, 500 mA, temp 32.1 C, notify 0.
- Sync DONE 16:51:07 UTC: exactly the 47 manifest files changed (`changed_files` == manifest, `unchanged_manifest_files`
  empty), deploy digests == manifest afterwards, second staging -> deploy dry run empty (`SYNC_ALLON-1.txt` in the rig dir).
- baseDP physical preflight STARTED 16:51:07 UTC -> `.../baseDP-20260924-inputs/preflight-1`; battery level 78, notify 0.
- Read-only phone check (adb 5037, OP15): `/data` 479 GB, 17.9 GB available (97 % used; old shard sets
  `s42-ffn-shards-20260904-v1` 12 GB and `-v2` 7.9 GB remain, nothing deleted); `/data/local/tmp/s43-ffn-shards-20260924-v3`
  does not exist yet -> the 9.2 GB Gemma-26 set fits with the 0.5 GB margin, leaving about 8.7 GB.

## PAUSE (16:58 UTC): coordinator relayed the user's pause

- Rule applied: arms already started finish normally (never kill an in-flight run/worker); arms not started are not
  started; the Gemma shard stage had not begun -> skipped, PENDING. `dev2baseDP` (preflight in flight) runs to completion
  (preflight + run). `dev2allon` had not started -> `CANCEL` file written into its inputs dir at 16:58 UTC; the chain
  checks it before that arm's preflight, logs `CANCELLED`, exits before the shard stage and releases the lock.
- Consequence: only the desktop-only reference under the new dispatcher (A = dev2baseDP vs legacy dev2base = the
  dispatcher gain) is measured in this session; B (dev2allon) and the (B) - (A) phone-contribution decomposition and
  the Gemma-26 shards are pending (restart commands at the end of this file).
