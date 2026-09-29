# Per-device adaptive phone policies (2026-09-25)

With two phones, the scheduler now decides for each device and batch composition, from measured
evidence, which device set assists a request:

- OP15 alone,
- OP15 + Pixel,
- Pixel alone (only as a fallback after the OP15 failed).

Nothing is hard-coded, including "Pixel only at batch 1".

Status: code, tests and merge are done. No hardware was run. The user changed the plan at about
00:40 UTC ("we can stop the trace, we can fix the pixel problem first"): the rig lock and the Pixel
are reserved for the Pixel CPU+GPU worker workstream. Section 6 lists the arms to run later and how.
Nothing is committed or pushed. The progress log is `PROGRESS.md`.

| Deliverable | Path (this directory) |
| --- | --- |
| Scheduler patch: 17 files, 1 new; `git apply --check` passes with stdin closed | `PER_DEVICE_POLICIES.diff` |
| Regenerate the diff (base/root trees under `$SCRATCH/perdev`) | `make_diff.sh BASE ROOT OUT` |
| Per-device accounting of a run (calls per device session, windows per device set and batch, device-set decision timeline) | `analyze_per_device.py RUN_DIR` |
| New tests on the base tree (13 fail/error, 5 guards pass) and on the root tree (18 pass) | `NEW_TESTS_BASE.log`, `NEW_TESTS_ROOT.log` |
| Full suites: root copy, and main after the merge | `SUITE_ROOT_FINAL.log`, `SUITE_MAIN_AFTER_MERGE.log` |

## 1. Design decision: runtime per-device control vs alternative routes

### What the server already does (read, not changed)

- **The control takes any layer mask inside the resident union.** The per-slot and cohort `ffn_split`
  controls accept any `layer_mask` with `layer_mask & ~ffn_split_max_layer_mask == 0`.
  `ffn_split_max_layer_mask` is the union of the OP15 and Pixel layers
  (`tools/server/server-context.cpp`, around lines 2600-2620 and 2745-2760).
- **The server switches a phone off when it owns no active layer.** It calls
  `s41_server_ffn_runtime::apply_policy` (`tools/server/server.cpp:868-905`), which gives each helper
  its owned subset, `policy mask & helper mask`. A helper with an empty subset gets
  `set_runtime_policy(0, 0)`: it is switched off and its layers run on the host. A helper that was
  never targeted stays deferred and connects on its first use.
- **The host keeps the weights of a switched-off phone's layers.** The dormant host share releases by
  (layer mask, host columns). A new mask first restores the previous release and then releases only
  the new layers (`src/llama-model.cpp:1792-1845`).
- **The memory ledger accepts a subset release.** `ShareBinding.covers` credits a release of fewer
  layers, and only the proven bytes.

So the server can already switch a whole phone off at every token boundary, with no rebuild.

### The two options

| | (a) runtime per-helper control | (b) alternative routes (OP15-only vs OP15+Pixel) |
| --- | --- | --- |
| Switch a phone on/off per batch or request | **Already supported** by the union mask. No C++ change, no rebuild, no identity re-materialization. | Needs two server launches or two catalog residency identities. Every switch is a server relaunch or model reload (a Qwen load measured 47-114 s). |
| Follow batch-composition changes within a request | Yes: every decode boundary, with per-batch verdicts | No. A mid-request change stays on the chosen route, so the Pixel would keep running at B4. |
| Different split fractions per phone (e.g. OP15 100 % + Pixel 50 %) | Would need C++: per-helper columns in the control, the client and the dormant host share, then a rebuild and `MATERIALIZE_TRANSPORT*.sh` | Same limitation per route |
| Catalog | Unchanged (one two-phone route) | Stage A catalogs are one-phone or two-phone per model (`resident_model_identity_sha256` ignores co-helpers). Both at once needs new identity work. |

**Chosen: (a) without a server change.** Each device set becomes a sub-policy of the one two-phone
envelope, and the existing runtime control switches between sub-policies at token boundaries. This is
the smallest option: scheduler Python only, zero C++, no identity work. It is also the most robust:
- the device set can change per batch composition inside a request, which (b) cannot do at all;
- coherence already keeps every slot of the server on one policy, so no slot runs a mixed device set.

**Trade-offs:**
- All active phones run the same fraction.
- The Pixel's resources stay leased by the route even when it is switched off.
- The Pixel cannot assist Qwen alone while the OP15 serves Gemma: helper availability is per envelope
  (Stage B territory).

## 2. Policy space and decision rules

**Policy space.** For a two-phone envelope (Qwen: OP15 layers 0-17, Pixel layers 18-23, grid 4,352 =
25 %), the policies are every non-empty device set on the envelope's column grid:
{OP15+Pixel, OP15, Pixel} x {25, 50, 75, 100 %} = 12 policies (Stage A had 4). Details:
- A subset policy's `layer_mask` is its devices' owned layers, and its `device_layer_masks` names them.
- `route_id` is `<envelope>:decode-window:<fraction>:devices:<a+b>`.
- Every subset policy is in the envelope's probe contract, so the ticket admits its controls.
- One phone: no subsets, and every hash is unchanged.

**Evidence** is kept per policy identity. The identity includes the layer mask, so it is kept per
device set. Windows carry `active_batch`, and every comparison uses windows of the same batch
composition only (unchanged).

**Decision rules**, all in the coherent server probe (`_internal/adaptive_decode_ops/coherence.py`).
Every rule is unchanged for one phone.

| Situation | Rule |
| --- | --- |
| Cold start at a batch composition | The proposal is the cheapest set: **the primary phone alone** (OP15). The request-local sample is `[OP15@100, OP15+Pixel@100, OP15@75, OP15@50]`, capped by `maximum_probe_candidates`. With coherence only the first entry matters. A new composition inherits the running fraction but restarts from the cheapest eligible set, not from the running two-phone policy. |
| Phase 1: set vs host | The unchanged server pair: the same bounds, latency bound, F1a/F2 inconclusive and reference rules, and probe budget with attempt cap. A decisive rejection **drops that set for this composition** (`SERVER_PAIR_NOT_IMPROVED`, `SERVER_<elimination>`, `SERVER_PROBE_BUDGET_EXHAUSTED`) and probes the next eligible set with a fresh budget. The host becomes the verdict only when no set remains. |
| Phase 2: adding co-helpers | Once a set holds the verdict at a composition, each eligible set that adds co-helpers challenges it; for two phones, OP15+Pixel challenges OP15. **It replaces the incumbent only when** its energy upper bound is at most the incumbent's lower bound x (1 - minimum saving). It must also be at most the host's lower bound x (1 - minimum saving) when the host was measured, and its latency upper bound must be within the host latency bound. The unchanged 1.25x bound and minimum saving apply. |
| Phase 2 outcomes | - Means favour the challenger but the bounds overlap: keep measuring both, alternating by band width, within the shared budget. <br>- Means against the challenger and only one incumbent window: take one more incumbent window. <br>- Otherwise the challenger is dropped for this composition: `SERVER_DEVICE_SET_NOT_IMPROVED`, or `SERVER_DEVICE_SET_LATENCY_BOUND_EXCEEDED` when its latency mean breaks the bound. <br>- Exhausted attempts, up to `maximum_probe_attempts_per_context`: `SERVER_DEVICE_SET_PROBE_BUDGET_EXHAUSTED`. <br>- Only challenger windows are charged to the budget. |
| A set that measures worse | It is dropped for **that** composition only. The B4 drop does not touch the B1 verdict (tested: OP15+Pixel keeps B1 and is dropped at B2). |
| Re-check | The phone verdict is monitored as before (`_update_elimination`). If it is eliminated, its set is dropped for the composition and the next eligible set is re-probed against the host. For example, OP15+Pixel dominated by OP15 falls back to OP15. Dropped sets are mirrored into the session's eliminations, so no per-request rule compares against them. Drops live in the server group, keyed by the model's shard layout identity. A new layer set starts fresh; the existing re-qualification rules are not loosened. |
| Device failure | A failed phone control or window of set S drops S and every superset **for every composition**, because which phone failed is not known. The server goes to the host, then falls back to the remaining sets: OP15 first, then the Pixel alone, and the Pixel alone only once the OP15 alone has failed. The host verdict (`SERVER_PHONE_POLICY_FAILED`) comes only when nothing is left. A later owner session gets the fallback proposal from the server, because its own probe order starts with the phone that failed. The request-local controller (coherence off) eliminates S and its supersets. |
| Decision records | Present only with several phones: <br>- `ASSISTANCE_DECISION` carries `device_set`, `device_layer_masks` and `active_batch`. <br>- `server_policy` carries `device_sets`, `policy_device_set`, `proposal_device_set`, `verdict_device_sets` per batch, `device_set_drops` (batch, set, reason) and `device_set_attempts`. <br>- `LEARNING_WINDOW_RECORDED` carries `device_set`, `active_batch` and `token_count`. <br>- The controller snapshot carries `device_sets`. |
| Proofs | Per device session, as before. For a plan with co-helpers, a phone must have served calls only if an executed policy drove one of its layers. An OP15-only request needs no Pixel calls, and the reverse holds too. Every phone of an executed set must still have served calls. One phone: every shard is still required. |
| Assumed phone energy | A window whose set excludes the primary phone does not charge the OP15 active power. The Pixel is charged as before. |
| Session drain | A primary-session drain (re-provisioning) leaves a Pixel-only policy running, reported as `already_applied`. Before this change it raised "removes all active layers". A two-phone policy keeps only the retained OP15 layers, as in Stage A. |

## 3. Change set (`PER_DEVICE_POLICIES.diff`, 17 files, +912 / -50 lines, 484 of the added lines are tests)

| File | Change |
| --- | --- |
| `_internal/adaptive_decode_contracts.py` | `policy_device_set`, `device_set_order` (policy hash and JSON unchanged) |
| `_internal/adaptive_decode_planning.py` | `_device_subset_policies` from `_envelope_subpolicies` |
| `_internal/adaptive_decode_state.py` | `_AdaptiveServerPolicy.device_sets / device_drops / device_attempts` (tuples, checkpoint-safe) |
| `_internal/adaptive_decode_ops/coherence.py` | device-set exploration, challenger, drops, failure fallback, snapshot fields |
| `_internal/adaptive_decode_ops/candidates.py` | representatives per (device set, fraction); cold-start order; refinement stays in the leader's set |
| `_internal/adaptive_decode_ops/windows.py`, `completion.py` | pass the failed policy; same-set preference when re-seeding a probe |
| `_internal/adaptive_decode_ops/helpers.py` | drain leaves co-helper-only policies running |
| `_internal/adaptive_decode_ops/reporting.py`, `_unified/adaptive_decode_control.py` | device sets in snapshots and decision/window records |
| `adapters/llama_server_ops/proofs.py` | required sessions per executed device set (co-helper plans only) |
| `adapters/http_backend.py` | primary not charged for co-helper-only windows |
| `campaigns/burstgpt/prepare_trace_inputs_v2.py` | `--drop-dispatch-policy` (derive the legacy all-desktop arm with tooling) |
| `tests/test_per_device_policies.py` (new, 18 tests) | section 4 |
| `tests/test_two_phone_gaps.py`, `test_two_phone_activation.py`, `test_prepare_trace_inputs_v2.py` | Three Stage A assertions ("every policy drives both phones") now check the full set on the grid plus the per-device subsets. One test for `--drop-dispatch-policy`. |

## 4. Tests

`tests/test_per_device_policies.py`. "base" is main at 00:22 UTC, running the same file.

| Required | Tests | base | root |
| --- | --- | --- | --- |
| Device-set candidates generated | `test_two_phone_envelope_yields_every_device_set_on_the_grid`: 3 sets x 4 widths, masks per device, in the probe contract | FAIL | pass |
| Per-device-set evidence and verdicts per batch | `test_co_helper_that_improves_replaces_the_primary_at_that_batch`: OP15 verdict, then the OP15+Pixel challenger wins B1; evidence per set | FAIL | pass |
| Slower device dropped for a composition | `test_slower_co_helper_is_dropped_for_that_batch_composition_only`: B2 restarts from OP15; OP15+Pixel is 3x slower and dropped at B2 (`LATENCY_BOUND_EXCEEDED`); B1 keeps OP15+Pixel, and returns to it when the partner leaves. `test_co_helper_that_does_not_improve_is_dropped_and_the_primary_kept` | FAIL | pass |
| Failure fallback | `test_failures_fall_back_to_the_remaining_device_then_the_host`: OP15+Pixel fails, then OP15 (measured pair re-qualifies at once); OP15 fails, then Pixel alone; Pixel fails, then host verdict. `test_primary_session_drain_leaves_a_co_helper_only_policy_running` (includes the server-driven fallback for a new owner). `test_request_local_controller_drops_the_failed_set_and_its_supersets` | FAIL / ERROR | pass |
| Cold-start order | `test_cold_start_probes_the_primary_alone_first_then_adds_the_co_helper`, `test_first_server_proposal_is_the_primary_alone` | FAIL | pass |
| Proofs per device | `test_primary_only_request_needs_no_co_helper_calls`, `test_co_helper_only_request_needs_no_primary_calls` (ERROR on base); guards `test_every_phone_of_an_executed_set_must_serve_calls`, `test_single_phone_proofs_still_require_every_shard` | ERROR / guard | pass |
| Decision records carry the device set | `test_decision_records_carry_the_device_set_and_the_server_choice`; guard `test_single_phone_decision_records_are_unchanged` | FAIL / guard | pass |
| Energy per device | `test_co_helper_only_window_leaves_the_primary_idle` | FAIL | pass |
| Single-phone digest guard | `test_single_phone_server_trace_matches_the_base_tree`: a coherent one-phone trace pinned to digest `sha256:e4eaec0f...` computed on base, identical on root. The trace covers probe, B1 verdict, a co-tenant at B2 with the comparison window, the partner leaving, a failed window, and the `_sample_candidates` output. Also `test_single_phone_envelope_has_no_device_subsets`. The existing `SinglePhoneUnchangedTests` still pass: catalog, candidates, policies, plans and launch digests from the base tree. | guard (passes on both, by design) | pass |

A guard pins base behaviour, so by construction it cannot fail on base.

Full suite (`/usr/bin/python3 research_dev/scheduler/tests/run_all.py`): see section 7.

## 5. Single-phone byte identity

Everything new is gated by "several device sets", which requires policies with `device_layer_masks`,
and those exist only when a co-helper is declared. One phone keeps:
- the same policies and hashes;
- the same `_representative_candidates`, sampling and refinement (device set `()` for all);
- the same coherence branches (`group.device_sets == ()`), snapshots and decision-record keys;
- the same proofs, energy charges and drain behaviour.

This is checked by the new digest guard and the existing `SinglePhoneUnchangedTests`.

## 6. Arms to run later (not run here)

Rules as in the task: one outer `flock -w 7200 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock`
across sync, prepare, preflight and runs; `adb -P 5037` only; battery logging before and after each
preflight and run; stop on OP15 notify 512. New directories go under `/mnt/storage/s43-perdev-*`.

### 6.0 Before any arm

- **Pixel worker.** The Pixel CPU+GPU workstream may replace the Pixel worker. Re-derive the Pixel
  evidence bundle for whatever worker it qualifies: `prepare_campaign_int2.py` reads `GATE_CONFIG.json`,
  the numerical suite, mechanism runs, TCP calibration and idle-stop receipts. The current bundle
  describes the packed-CPU worker: 38 ms/layer at B1, 82 ms at B4.
- **Sync.** Sync the merged main scheduler into `/mnt/storage/s42-trace-v2-20260921-prep/source/research_dev/scheduler/`
  (the rsync + tree-digest check in `../20260924-pixel-integration-2/run_chain_int2.py`). There is no
  server rebuild, so the identity `TRANSPORT_QUALIFICATION_IDENTITY_PIXEL_STAGEA_COALESCED_BOTH.json`
  (library a87e7772...) stays valid.
- **New root.** Make a new root `/mnt/storage/s43-perdev-20260925/` holding sha256-verified copies of
  the int-2 inputs to the Pixel receipts (as int-2 did from the Pixel agent's directories):
  `GATE_CONFIG.json`, `qualification/numerical`, `mechanism-r1`, `mechanism-b4-r1`,
  `tcp-calibration-r2`, `idle-stop-r2`, plus `prepare_campaign_int2.py`. Never write into
  `/mnt/storage/s42-two-phone-pixel-20260924-v1` or `/mnt/storage/s42-pixel10pro-*`.
- **No zero-Pixel stop.** Do not pass `:require-helper-calls` for the two-phone arm. With per-device
  policies, zero Pixel calls is a valid outcome when the Pixel never measures better. Check instead
  that `analyze_per_device.py` shows OP15+Pixel challenger windows, or a drop reason, at each
  composition.

### 6.1 dev_v2 matched arms (gate for 6.2)

```sh
# on the desktop, whole chain under ONE outer lock
flock -w 7200 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock \
  python3 /mnt/storage/s43-perdev-20260925/run_chain_int2.py \
    --status /mnt/storage/s43-perdev-20260925/chains/CHAIN-dev2.jsonl \
    --stage <staged copy of the merged main repo root> \
    --prepare /mnt/storage/s43-perdev-20260925 dev2a /mnt/storage/s43-pixel-int2-20260924/template-dev2 dev2 \
    --prepare-script /mnt/storage/s43-perdev-20260925/prepare_campaign_int2.py \
    --arm /mnt/storage/s43-perdev-20260925/inputs-desktop-dev2a \
    --arm /mnt/storage/s43-perdev-20260925/inputs-op15-dev2a \
    --arm /mnt/storage/s43-perdev-20260925/inputs-two-phone-dev2a
```

- **What each arm is.**
  - The desktop arm is `desktop-baseline` with the dispatch policy.
  - The op15 arm is OP15 all-on (coherence, coalesced both, re-provisioning, dispatch policy).
  - The two-phone arm is OP15+Pixel with per-device policies. It is the int-2 two-phone configuration;
    the per-device behaviour comes from the merged code, with no flag.
- **Repeat.** Repeat the op15 and two-phone arms counterbalanced (two-phone first), as int-2 did.
- **Analysis.**
  - `compare_trace_energy.py` for host kJ and duration.
  - `analyze_longdecode_pair.py` against the desktop arm for exactness.
  - `analyze_per_device.py <run>` for calls per device session, windows per (device set, batch) and the
    device-set timeline.
  - `../20260924-coherent-policy-coalesced/analyze_allon.py` for loads and pairs.
  - `../20260924-pixel-integration-2/interval_energy_int2.py` for the Qwen window.
- **Gate.** Go to 6.2 only if per-device is at least OP15 all-on within noise (3.1 % energy / 13 %
  duration between repeats). The expected behaviour, from the int-2 per-layer RPCs:
  - B4: OP15+Pixel is dropped by latency;
  - B1: OP15+Pixel is either kept or dropped, whichever measures better;
  - so per-device should match OP15 all-on at B>=2 and be at least as good at B1.

### 6.2 Full `longtail_v1` (31 requests, about 80 min per arm), in this order

```sh
# 1. longtail template from the int-2 dev_v2 template (tooling only; check the trace file names with ls first)
python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py \
  --source /mnt/storage/s43-pixel-int2-20260924/template-dev2 \
  --output /mnt/storage/s43-perdev-20260925/template-longtail --campaign-id s43-perdev-longtail-template \
  --old-deploy /mnt/storage/s42-trace-v2-20260921-prep --new-deploy /mnt/storage/s42-trace-v2-20260921-prep \
  --replay-schedule /mnt/storage/burstgpt-source/longtail_v1/burstgpt_longtail_v1.json \
  --trace-manifest /mnt/storage/burstgpt-source/longtail_v1/TRACE_MANIFEST.json \
  --large-requests /mnt/storage/burstgpt-source/longtail_v1/REQUESTS_SEMANTIC_SOURCE.jsonl \
  --overlay-requests /mnt/storage/burstgpt-source/longtail_v1/REQUESTS_OVERLAY.jsonl
# 2. evidence + the desktop / op15 / two-phone arms (inside the locked chain, --prepare ... lt1 <template-longtail> lt)
# 3. the legacy all-desktop arm from the desktop arm, dispatcher removed
python3 research_dev/scheduler/campaigns/burstgpt/prepare_trace_inputs_v2.py \
  --source /mnt/storage/s43-perdev-20260925/inputs-desktop-lt1 \
  --output /mnt/storage/s43-perdev-20260925/inputs-desktop-legacy-lt1 --campaign-id s43-perdev-lt-desktop-legacy-lt1 \
  --old-deploy /mnt/storage/s42-trace-v2-20260921-prep --new-deploy /mnt/storage/s42-trace-v2-20260921-prep \
  --drop-dispatch-policy
cp /mnt/storage/s43-perdev-20260925/inputs-desktop-lt1/TRANSPORT_QUALIFICATION_IDENTITY.json \
   /mnt/storage/s43-perdev-20260925/inputs-desktop-legacy-lt1/   # the deploy identity, as prepare_campaign_int2 writes it
# 4. run, under one outer lock, in this order:
#    --arm inputs-desktop-legacy-lt1   (all-desktop baseline; 09-23 on older code: 523.7 kJ / 4,901 s)
#    --arm inputs-desktop-lt1          (desktop + dispatch_policy)
#    --arm inputs-op15-lt1             (OP15 all-on)
#    --arm inputs-two-phone-lt1        (OP15+Pixel per-device)
```

**Report** against the all-desktop baseline and against desktop+dispatcher:
- host kJ, duration and exactness (near-tie flips at fixed positions are expected);
- the phone-energy caveat (assumed 4.5 W / 0.875 W model);
- loads and switches;
- phone shares per model and device, and the device-set timeline, from `analyze_per_device.py`.

## 7. Full suite and merge

| Tree | Result |
| --- | --- |
| root copy, first run (before the last edits: drain, fallback proposal, window-event fields) | 141 modules / 2,020 tests; only the known `test_cached_synthetic_refinement_is_below_ten_milliseconds` (10 ms timing) failed |
| root copy, final code (`SUITE_ROOT_FINAL.log`) | 141 modules / 2,021 tests; only the known 10 ms timing assertion failed (10.09 ms). Isolated reruns: 1 of 3 OK. Interleaved timing of the measured quantity: base medians 9.6-10.1 ms, root 9.95-10.85 ms. A cProfile puts the touched planning code at 0.12 ms per call on both trees, so this is machine noise, not the change. |
| main after the merge (`SUITE_MAIN_AFTER_MERGE.log`) | 141 modules / 2,021 tests, exit 0 (all pass, including the timing test) |

Merge (01:01 UTC):
- **Before.** The main scheduler tree was byte-identical to the base snapshot, and `git apply --check`
  passed with stdin closed.
- **Backups.** The 16 modified files plus `SHA256SUMS` are in `$SCRATCH/perdev/premerge-backup/`
  (`$SCRATCH` = `/tmp/claude-1761612022/-home-myid-zs89458-Documents-llama-cpp-release/6fd7e39c-bd31-4e92-b800-5784f94be902/scratchpad`).
  The new test file was absent from main.
- **After.** `git apply` on main; all 17 files are byte-identical to the root copy.
- **Rig.** The desktop deploy was not synced: nothing on the rig has this code yet.

## 8. Caveats and open items

- **Nothing is measured on hardware.** Whether the Pixel is kept at B1 is decided at run time by the
  bounds. With the 2 % uncertainty floor, a gain below about 5 % over OP15 alone never resolves: the
  challenger then spends at most `maximum_probe_attempts_per_context` x `maximum_probe_tokens`
  (4 x 80) tokens at that composition before it is dropped.
- **One fraction for all active phones.** Per-phone fractions need the C++ change of section 1.
- **The Pixel alone is a failure fallback, not a cold-start probe.** It becomes eligible only after the
  OP15 alone failed. It also cannot assist while OP15 serves another model, because helper
  availability is per envelope.
- **Assumed OP15 active power in mixed windows.** It still uses the server's aggregate RPC time over
  both phones (a pre-existing overcharge, biased against the Pixel). Per-helper timing needs per-helper
  server stats (C++).
- **Stage A's always-both-phones behaviour cannot be reproduced from main any more.** The pre-merge
  backups in `$SCRATCH/perdev/premerge-backup/` hold it.
- **Request-local path.** With coherence off, the fallback to the Pixel alone after an OP15 failure is
  not driven, but failed sets are eliminated. The arms use coherence.
