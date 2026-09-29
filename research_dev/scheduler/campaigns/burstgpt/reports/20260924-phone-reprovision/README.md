# Scheduler change #4: re-provision the phone for the desktop's model

Source: `research_dev/scheduler/campaigns/burstgpt/reports/20260923-offload-accounting/README.md`,
section 8, change #4. Deliverable: `REPROVISION.diff` (16 files, repo-root relative). It applies to
the current main tree:

```sh
cd /home/myid/zs89458/Documents/llama.cpp-release
git apply --check /tmp/claude-1761612022/-home-myid-zs89458-Documents-llama-cpp-release/6fd7e39c-bd31-4e92-b800-5784f94be902/scratchpad/reprovision/REPROVISION.diff < /dev/null   # passes
git apply        /tmp/claude-1761612022/-home-myid-zs89458-Documents-llama-cpp-release/6fd7e39c-bd31-4e92-b800-5784f94be902/scratchpad/reprovision/REPROVISION.diff < /dev/null
```

Nothing was committed, run on hardware, or pushed to a phone.

Update: the change is now in the main tree and was run once on the rig (dev_v2, coherentEF configuration plus the
field). Result: 53.69 kJ against 71.86 / 69.64 kJ for two coherentEF runs, with 17 Qwen / 24 Gemma phone layers
after every switch. Details and findings are in `RIG_RESULTS.md`.

## Problem (run 5, measured)

The phone portfolio sums the arrived decode work of every model. With Qwen and Gemma both queued,
the learning selector settled on 12 Qwen + 8 Gemma layers at t = 251 s and kept them for the rest
of the run (72 `LEARNING_RETAINED`). The swap happened when the first Gemma request *arrived*
(220 s before Gemma ran), and afterwards the layers of whichever model was not on the desktop sat
idle about half the time (48-50 %).

## What the change does (opt-in)

Campaign manifest field (no field = the old behaviour, byte for byte):

```json
"phone_resident_model_reprovisioning": {}
```

Optional fields: `load_bytes_per_second` (session-load prior, default 200000000; run 5 measured
205 MB/s) and `minimum_learned_samples` (default 2). It cannot be combined with
`fixed_phone_residency`. `launch.py` passes it as `--phone-resident-model-reprovisioning-json` and
`runner.py` calls `UnifiedScheduler.configure_phone_resident_model_reprovisioning(...)`.

When the field is set, `_update_phone_residency_portfolio` restricts the layout demand before it
builds candidates. The logic lives in the new module `_unified/phone_residency_ops/reprovision.py`:

- **Which models the phone follows** (checked in this order): phone-capable models the desktop is
  *loading* (an ACQUIRED ticket with a pending non-trivial load on a desktop device), then models
  it is *executing*, then models the snapshot shows `hot` on a desktop device (then `warm`), and
  finally the models followed last time.
  - One followed model with remaining work is **FOLLOW**: every session that RAM and the stored
    shards allow goes to that model.
  - Several followed models are **PROPORTIONAL**: the ready sessions are split by remaining decode
    work (largest remainder).
  - No desktop knowledge is also PROPORTIONAL, over all arrived work. A single model then gets the
    whole phone.
  - A followed model with no remaining work is **HOLD**: the layout is kept, so a queued model is
    loaded when the desktop switches to it, not when it arrives.
- **Timing**: when `wait_runtime_request` commits a ticket whose plan loads a model on the desktop,
  the portfolio is re-evaluated at the dispatch time. The first phone session swap starts at once
  and overlaps the desktop load. Selection is confirmed without the 3-snapshot debounce while the
  commitment is live, through the existing `force` path of `confirm_phone_layout_candidate`.
- **Staging**: each proposal changes one session. The stage is the best-ranked *generated* layout
  with exactly one changed session, so it is memory-feasible in the intermediate state. After the
  stage is READY, the existing `_reevaluate_ready_layout_portfolio` proposes the next one. The
  physical path is unchanged (helper-preparation watcher -> `begin_phone_layout_transition` ->
  copy-on-write load -> verify). Shard files, hashes and certification are untouched.
- **In-use sessions (fail-closed)**: these sessions are never selected for a change:
  - sessions an ACQUIRED helper uses, by the same test as `phone_layout_transition_blockers`;
  - sessions with `active_helper_references`;
  - sessions that are not EMPTY/READY.

  If the full target is blocked this way, the decision records `blocked_session_ids`, and the
  request-completion hook (placed after `release_request`) re-evaluates. On the physical path,
  `begin_request_helper_preparation` returns `DEFERRED WAITING_FOR_HELPER_RELEASE` for our
  proposals instead of draining the blocker (`_defer_preparation_for_blockers`, which drains, is
  still used for every other proposal). A stage that would need an infeasible helper
  revalidation is deferred.
- **Degraded sessions** (a current session not ready) go to the unchanged base selector (forced
  degradation path). A configured HTP memory-cap rebalance still takes precedence.
- **Swap estimate**: the load rate is learned from complete `SESSION_LOADING` ->
  `SESSION_VERIFIED` windows. Each layout generation is one wall window, and a retried load
  restarts its window. Run 5 gives 4 windows and 205 MB/s. The rate feeds
  `transition_latency_us_by_session` and the request-impact preview. The decision records
  `stage_swap_latency_us`, `target_swap_latency_us` and `fits_load_window`.
- **Observability**: every `EVALUATED` event of this path carries `desktop_reprovision` with these
  fields:
  - mode, followed models, commitment source, leader request, load window;
  - in-use and blocked sessions;
  - target and selected layers per model, and target session counts;
  - learned rate and sample count.

## Diff summary

| file | +/- | what |
| --- | ---: | --- |
| `_unified/phone_residency_ops/reprovision.py` (new) | +577 | commitment detection, demand restriction, selection/staging, learned rate, triggers, release wait |
| `_unified/phone_residency_ops/portfolio.py` | +8 -2 | demand hook, confirmation via `force`, event field |
| `_unified/phone_residency_ops/economics.py` | +8 | reprovision selector before the default selector |
| `_unified/phone_residency_ops/common.py` | +6 | `reprovision`/`confirmed` fields (defaults keep old behaviour) |
| `_unified/phone_residency.py` | +30 -1 | mixin methods |
| `_unified/runtime_requests.py` | +4 | desktop-load dispatch hook, post-release hook (both only when configured) |
| `_unified/helper_preparation_ops/start.py` | +3 -1 | wait-for-release instead of drain, only for our proposals |
| `configuration/campaign.py`, `config.py`, `scheduler.py` | +44 | knob, validation, manifest round trip, export, state |
| `campaigns/burstgpt/{arguments,launch,runner}.py` | +11 -1 | plumbing (one hunk each, self-contained) |
| `README.md` | +15 | knob documentation |
| `tests/test_phone_resident_model_reprovision.py` (new) | +598 | 30 unit tests |
| `tests/test_phone_reprovision_portfolio.py` (new) | +286 | 7 tests through the real portfolio + placement controller |

The previous agent's edit to `_internal/phone_shards.py` (`progressive_ffn_residency_layouts`) was
reverted. It changed the stage benefits of the memory-cap arm even with the knob off.

## Tests

- New tests, with every requested property checked:
  - single-model demand gives the full-RAM layout;
  - two-model demand gives a proportional split;
  - the swap starts inside the desktop load window and converges in 3 stages (synthetic: 24 s of
    a 47 s window);
  - in-use sessions are never evicted, and the swap resumes at release;
  - RAM and session caps hold for every stage;
  - selection is deterministic, including with reversed candidate order;
  - with the knob off the behaviour is the base behaviour: identical demand objects and
    `LEARNING_RETAINED`;
  - the run-5 regression: an arrival does not swap an idle phone;
  - the learned rate matches run 5 (205 MB/s) and handles retries;
  - trigger gating (including `wait_runtime_request` firing the dispatch hook only when
    configured) and the wait-for-release path.
- **Red on base**:
  - `test_phone_reprovision_portfolio.py` imports only base-tree names. 6 of its 7 tests fail on
    base; the knob-off test passes, as it should. On base, the "arrival" case gives
    `PHONE_RESIDENCY_LEARNING_EXPLORATION`, which is the run-5 swap-on-arrival.
  - `test_phone_resident_model_reprovision.py` fails to import on base.
- **Full suite** (`run_tests.py root`, one process per file): 102 files passed; see `root_tests.log`.
  After the last test edit, the 21 phone, runtime and campaign files were rerun and all pass
  (`test_cached_synthetic_refinement_is_below_ten_milliseconds` failed once at load 7 and then
  passed twice). The only failure is `test_resident_router_subset.py`, which needs `examples/`
  next to the tree and fails in any copy. pyflakes finds nothing in the whole package, as on base.

## Estimated effect (calibrated replay of `hypotheticals.py`, observed admission)

`estimate/estimate_reprovision.py` imports the report's `hypotheticals.py` read-only and changes only
the layer counts and the reload rate. The simulator adds `max(0, reload - load)` to each switch. That
is conservative, because the implementation never makes the desktop wait for the phone.

| variant (vs replay 4,734 s / 410.1 kJ) | today's phone policy | + coherent coalesced (#1) |
| --- | ---: | ---: |
| report D: 18 Qwen / 26 Gemma, 270 MB/s | 256.6 kJ (-37.4 %), -4.7 % time | 219.1 kJ (-46.6 %) |
| as implemented, existing shards: 18 / 24, 205 MB/s | 269.9 kJ (-34.2 %), -4.3 % | 232.9 kJ (-43.2 %) |
| live phone limit below 9.63 GB: 17 / 24, 205 MB/s | 275.6 kJ (-32.8 %), -4.1 % | 239.3 kJ (-41.6 %) |
| after the Gemma 24-25 shards below: 18 / 26, 205 MB/s | 257.1 kJ (-37.3 %), -4.3 % | 219.6 kJ (-46.5 %) |

Full reloads at 205 MB/s take:

- Qwen: 17 layers in 44.3 s and 18 layers in 47.0 s, against a mean desktop load of 51.3 s.
- Gemma: 24 layers in 41.4 s and 26 layers in 44.9 s, against a mean desktop load of 41.2 s.

Each stage (one session, about 15 s) is usable as soon as it is verified.

## Shards to generate (written, not run)

Qwen needs nothing new. The phone has `s42-ffn-shards-20260904-v1/qwen` with HTP0 0-5, HTP1 6-11
and HTP2 12-17 (verified by listing). 6 Qwen layers (3,208,642,560 B) fill the 3,208,646,656 B
session limit, and a 4th session does not fit in phone RAM, so 18 is the ceiling.

Gemma: `v2/gemma24` holds 0-23 (8 per session). Gemma keeps layers 0-25 on the CPU (48 layers,
22 offloaded to the GPU). 9 layers (3,185,049,600 B) fit one session, so 26 = 9 + 9 + 8
(9,201,254,400 B), which is under the lowest live limit seen in run 5 (9,273,319,424 B). The FFN
shard format (`s42.ffn_shard.*`) is written by `native/ffn_shard_gguf.py`;
`research_dev/shard_gguf.py` writes layer-split stage GGUFs instead.

```sh
# desktop (zhihao@172.20.74.85)
cd /mnt/storage/s42-trace-v2-20260921-prep/source && export PYTHONPATH=$PWD
OUT=/home/zhihao/s42-ffn-shards-20260924-v3-gemma26
python3 research_dev/scheduler/native/ffn_shard_gguf.py /home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf \
    --parent-sha256 sha256:ed76f2183d2d1d65091986033023e6c78d27f6276c1b0c5826cc92acf73538cf \
    --out-dir $OUT/gemma26 --verify-parent \
    --shard HTP0=0-8:15360 --shard HTP1=9-17:15360 --shard HTP2=18-25:15360

# OP15 (via the desktop's adb server; ~9.2 GB free needed on /data)
PHONE=/data/local/tmp/s42-ffn-shards-20260924-v3/gemma26
ADB="adb -P 5037 -s 3C15AU002CL00000"
$ADB shell df -h /data
$ADB shell mkdir -p $PHONE
for f in HTP0.ffn.gguf HTP1.ffn.gguf HTP2.ffn.gguf FFN_SHARDS.json; do $ADB push $OUT/gemma26/$f $PHONE/$f; done
$ADB shell "cd $PHONE && sha256sum HTP0.ffn.gguf HTP1.ffn.gguf HTP2.ffn.gguf"   # must equal shard_sha256 in FFN_SHARDS.json

# runner / campaign: replace the Gemma shard index
--gemma-ffn-shards $OUT/gemma26/FFN_SHARDS.json=$PHONE
```

## Caveats

- **Not validated on hardware.** The physical chain is assumed to behave as in run 5's HTP2 swap:
  1. the dispatched leader's (or a queued same-model request's) helper watcher materializes the
     preparation envelope;
  2. it loads the session during the desktop load;
  3. `_reevaluate_ready_layout_portfolio` proposes the next stage.

  The first gate should check `desktop_reprovision` events against `SESSION_LOADING`/`VERIFIED`
  times and the desktop load receipts.
- **Snapshot-less evaluation.** The dispatch-hook evaluation has no snapshot, so its (first-stage)
  limit is the static session and pool limit, not the live limit. A one-session swap stays under
  the live limit in both directions (Gemma to Qwen adds at most 0.38 GB). Later stages are
  evaluated with the completion snapshot. An infeasible stage is deferred by the existing
  `LIVE_MEMORY_CAPACITY_NOT_READY` path.
- **Dispatch-time demand rows.** The dispatch-time evaluation needs a phone demand row without a
  snapshot: route evidence, or the online-learning demand cached by earlier submits. Without one
  it records `PHONE_RESIDENCY_DEMAND_UNAVAILABLE`, and the swap starts at the next evaluation that
  has a snapshot (submit, replan or helper completion).
- **17 vs 18 Qwen layers.** 18 Qwen layers need a live limit of at least 9.626 GB. Run 5 saw
  9.27-9.67 GB, so often 17 layers result.
- **No regrowth.** A session packed short by a temporarily tight limit keeps that shard. The
  generator only regrows under a configured HTP cap.
- **Minimum residency not considered.** Stage choice ignores the 30 s minimum residency. A quick
  switch-back may pick a just-loaded session, and `propose_phone_layout` then defers it
  (`PHONE_SESSION_MINIMUM_RESIDENCY`) until a later evaluation.
- **Startup split can churn.** PROPORTIONAL without desktop knowledge (startup only) follows the
  queue mix. It uses the normal 3-snapshot confirmation plus 30 s residency, but can still swap as
  the mix changes before the first load.
- **Short requests.** The swap does not check whether the followed model's remaining decode time
  exceeds the swap time. `_reject_stale_phone_layout_proposal` (existing) drops proposals whose
  demand fell below the adaptive minimum.
- **Estimates are replay estimates.** Per-layer gains are extrapolated linearly to 17/18/24/26
  layers, and the energy saving is host-side only. Phone energy is not in the model.

## Workspace

- `base/`: a copy of main `research_dev/scheduler`, identical to the current main tree except
  for new report directories.
- `root/`: the implementation.
- `make_diff.sh`: regenerates `REPROVISION.diff`.
- `run_tests.py`: runs one process per test file (`python3 run_tests.py root`).
- `root_tests.log`: the final full run.
- `basecheck/`: base plus the new tests (red on base).
- `estimate/`: the change-#4 estimate driver and its output.
- `patch_root.py`: the previous agent's wiring script. It is superseded, no longer matches `root/`,
  and should not be re-run.
