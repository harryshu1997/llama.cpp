# Earlier useful residency, bounded winner reuse, stale-proposal revalidation

Status: implemented, full suite and both replay goldens unchanged, one
unchanged three-request gate (v10) PASS 3/3 with warm page cache disclosed.
Gemma reached its 24-layer layout at 46.5 s (v9: never before completion),
ran 212 tokens on it, no preparation was deferred, Qwen received its own
layout at Gemma's completion, and fleet energy was 19.756 kJ at 4.5 W in
236.0 s (v9: 22.689 kJ, 267.2 s). Successive versions, not a matched A/B.
Order followed: plan items 1 + 2 together, then stale-proposal revalidation,
with the tail measurement added to the analyzer. Safe Qwen load overlap is
assessed below and not implemented: on this host it is infeasible while Gemma
is resident. No commit, push, baseline rerun or longer trace.

## 1. Expansion is not eviction (`_internal/model_placement_ops/economics.py`)

The learning-mode guard returned `PHONE_RESIDENCY_LEARNING_RETAINED` whenever
every demanded artifact already appeared somewhere in the current layout, before
any economics ran. That blocked adding sessions for the same artifact (8 -> 16
-> 24 layers) exactly as it blocked evicting one. The guard now separates the
two: when demand is covered, only layouts that keep every resident shard
byte-identical (same artifact, resident geometry and operator plan, same
session) and fill previously empty sessions remain candidates; they pass through
the unchanged single-session rule, revalidation deferrals (`impact_reason`) and
the positive-gain check, and are selected with the new reason
`PHONE_RESIDENCY_LEARNING_EXPANSION`. Replacing or reshaping a resident shard
with demand covered stays retained. Memory, transition-cost and verification
checks downstream (`phone_layout_transition_blockers`, memory reservation,
`_defer_phone_layout_revalidation`) are untouched; existing assistance keeps
running because the resident sessions are not changed.

Both call sites benefit: `_phone_candidate_choice` (new proposals) and
`_defer_phone_layout_revalidation` (the v9 deferral at 37.1 s came from the
latter and lasted until Gemma completed).

Offline replay on v9's own PROPOSED layouts (rebuilt from
`phone_residency_events`, one Gemma shard per session, layer masks 0xFF,
0xFF00, 0xFF0000, byte-identical resident shards across generations):

| Current -> candidate | Before-image | Patched |
| --- | --- | --- |
| gen 1 (HTP0) -> gen 2 (HTP0+HTP1) | current, LEARNING_RETAINED | candidate, LEARNING_EXPANSION |
| gen 2 -> gen 3 (HTP0+HTP1+HTP2) | current, LEARNING_RETAINED | candidate, LEARNING_EXPANSION |

## 2. Bounded reuse of the useful fraction (`_internal/adaptive_decode_ops/helpers.py`)

On a layout expansion the controller rebinds the running fraction to its
expanded counterpart (existing `helper_rebound`, `expanded_matching`), then in
LEARNING state appended the whole coarse sweep after it (`learning_reprobe`):
v8 and v9 re-ran 100/75/50/25 on 16 layers after having run it on 8. Now the
probe list after a contract change is only the expanded form of the running
fraction plus, if different, the expanded form of the best-measured fraction so
far (`_expanded_prior_candidates`). Nothing else is probed, `refinement_added`
is set, and `probe_tokens` and `verification_attempts` are left as they are.
The prior is not inherited qualification: the expanded policy must be measured
against the same baseline in the current context before it becomes incumbent
(`_select_probe_winner` -> `_qualifies` / bounded comparable-measurement block
/ `_finish_verification`), and a negative pair rejects it. Exact-compatible
rebinds (same mask, aliased evidence) keep the existing continuity path.

The v9 finding stands: its adaptive observation store input had zero groups
(`INITIAL_ADAPTIVE_OBSERVATIONS.json`, `groups: []`), so no cached winner
existed there; this change only affects reuse within a request across layout
generations. Any warm-evidence experiment must state its starting store.

## 3. Stale proposals are revalidated before loading (`helper_preparation_ops/authorization.py`, `start.py`)

`begin_request_helper_preparation` now calls
`_reject_stale_phone_layout_proposal` right after resolving the PROPOSED
layout. It takes the artifacts of the shards the proposal would load
(`changed_session_ids`), sums arrived and running decode tokens for those
artifacts from `_phone_queue_demand` (queued requests count their full output,
active ones their remaining tokens), and if none reaches
`_adaptive_envelope_minimum_remaining_tokens` it rejects the proposal
(`reject_phone_layout_proposal`, reason `PHONE_RESIDENCY_PROPOSAL_STALE`),
records a REJECTED helper event with the live token counts, and requests a
portfolio recompute so current demand (Qwen in v9) is planned instead.
Demand for a different model does not keep a Gemma expansion alive; demand
for the proposal's own artifact from any request does. Proposals that are not
the current target, or with no changed sessions, are left alone.

## 4. Tails measured separately (analyze.py)

Per request: wait for resources (arrival -> ACQUIRED dispatch), launch/load/
prefill (dispatch -> first token), decode (first token -> completion); per run:
startup before the first dispatch and cleanup after the last completion; per
server log: file read (`load_tensors` -> warm-up), warm-up/init (-> listening).
Terminal-proof processing has no timestamp in RESULT and is reported as such.
Nothing was shortened.

| Run | Startup | Gemma wait / load / decode | Llama wait / load / decode | Qwen wait / load / decode | Cleanup |
| --- | ---: | --- | --- | --- | ---: |
| v8 | 6.0 s | 5.0 / 13.5 / 123.5 s | 0.2 / 2.7 / 24.6 s | 52.2 / 49.1 / 43.1 s | 8.4 s |
| v9 | 6.5 s | 5.5 / 11.6 / 118.4 s | 75.6 / 2.7 / 3.7 s | 52.0 / 70.6 / 44.2 s | 9.4 s |

Qwen's hot-desktop server in v9: 60.5 s reading the 28 GB file, 1.7 s warm-up
and initialization, listening at 62.5 s. Gemma's cold-desktop server (warm
cache): 2.5 s read, 1.0 s warm-up.

## Qwen load overlap: assessed, not implemented

The measured cost is the file read, and the host has 30 GB of RAM with Gemma's
24 GB weights mapped while it runs. A prefetch of Qwen's 28 GB during Gemma
would evict Gemma's pages and slow the protected request; the memory check
the plan requires would refuse it on this rig every time. The phone shards do
not remove the desktop copy. What remains actionable is (a) starting the read
the moment the GPU frees rather than after the replan round-trip, which the
timeline shows is already within a second, and (b) hardware or file-cache
policy outside the scheduler. No capacity-bounded prefetch was added because
it would never be admitted here and could not be exercised physically.

## Tests

TESTS.json. Full suite 1,358 tests: the two pre-existing errors, plus one
existing COW authorization test whose hand-built scheduler stub lacked the
new downstream helper; it now stubs it like the other helpers and the module
passes. Both replay goldens unchanged. New: expansion-versus-eviction cases
on partial layouts, bounded-prior cases (single prior; prior plus measured
leader; no 25% re-probe; budgets untouched), five stale-proposal cases, and
the rewritten rebind test whose old assertion encoded the sweep resume.
Offline replay of v9's real layouts above.

## Physical gate v10

Same deploy root, the nine files in CHANGES.json (remote hashes match), a
fresh preflight (PASS), weights pre-warmed as for v8 and v9 (Gemma's file
was still cached; Qwen's was partially cached this time, see below).

| Request | Executor | Wait / launch+load / decode | Assisted tokens | Fraction-weighted | Layers/columns |
| --- | --- | --- | ---: | ---: | --- |
| Gemma 36 | cold desktop | 4.6 / 11.0 / 111.3 s | 265/292 (90.8%) | 256.25 (87.8%) | L24/C15360/100% 212, L16/100% 19, L16/50% 11, L8 75%/100% 11 each, L8/50% 1 |
| Llama 37 | GPU control after Gemma | 67.0 / 3.0 / 1.3 s | 0 | 0 | none |
| Qwen 50 | hot desktop | 41.3 / 51.6 / 43.3 s | 17/71 (23.9%) | 17 (23.9%) | L6/C17408/100%, then MEASURED_REJECTION at token 29 |

Duration 235.997 s; fleet 19.613 / 19.756 / 19.899 kJ at 3 / 4.5 / 6 W;
startup 5.6 s, cleanup 8.7 s. Five execution attempts (three requests, the
startup preload, Llama's replan), zero fallback recoveries, no
PREPARATION_DEFERRED and no REJECTED events, four phone loads (HTP0, HTP1,
HTP2 Gemma; HTP2 replaced by Qwen at 130.6 s), 299 MiB GPU used and
`ptp,adb` after cleanup.

What changed in the timeline:

| | v9 | v10 |
| --- | --- | --- |
| Gemma 16-layer layout READY | 36.7 s | 35.5 s |
| Gemma 24-layer layout | proposed 36.5 s, deferred LEARNING_RETAINED until 136.3 s, loaded for nobody | proposed 35.5 s, loading 36.1 s, READY 46.5 s |
| 16-layer probes after expansion | 25/50/75/100 (44 tokens) | 50 (running fraction, 11 tokens) then 100 (measured leader, 19 tokens) |
| 24-layer probes after expansion | none | 100 only (running fraction), verified then exploited for 212 tokens |
| Qwen layout | none; 17 x PHONE_HELPER_UNAVAILABLE | HTP2 -> Qwen proposed 129.4 s at Gemma's completion; 17 assisted tokens |

Phase energy (RAPL package + NVML board, the same intervals as the prior
report): Gemma decode 10.57 kJ over 116.0 s at 91.1 W (v9: 13.10 kJ, 124.0 s,
105.6 W); Qwen load wait 1.86 kJ (v9 1.68); Qwen decode 5.57 kJ (v9 6.04);
startup, Llama and cleanup within 0.2 kJ of v9. The 2.93 kJ difference is
2.53 kJ Gemma decode and 0.47 kJ Qwen decode.

Two cautions on that number. First, Qwen's 28 GB read took 38.3 s (v9:
60.5 s), most likely partial page-cache residue from v9's Qwen load; that
shortens duration by about 22 s but the load-wait energy did not fall. Second,
the unattributed host CPU step recurs: package power was 14-16 W from 50 to
70 s while Gemma decoded at 24 layers and 360-400 ms/token, then about 70 W
from 75 s onward at the same latency; Gemma's per-token whole-fleet energy
went from 18-21 J to 37-39 J with no change in policy, layout, phone calls or
scheduler events. The same step sits at 72-74 s in v7 and v9 (masked by the
co-run in v8). It is not caused by anything the runner or rig schedules
(no timers or hashing at that offset were found). Attribution needs per-
process CPU sampling in the host sampler; until then per-token energies after
about 73 s are not comparable across runs, and the honest reading of the
24-layer layout is faster decode (371 versus 404 ms/token in the same host
state) with an energy benefit that the totals show but the per-token windows
cannot isolate.

| Run | Change set | Llama route | Duration | Fleet kJ at 4.5 W | Gemma assisted / weighted |
| --- | --- | --- | ---: | ---: | --- |
| v4 | epoch/helper admission | GPU after Gemma | 266.9 s | 22.562 | 93.5% / 86.9% |
| v6 | + READY CPU dispatch | CPU concurrent | 250.0 s | 25.394 | 43.8% / 35.4% |
| v8 | + measurement context (warm cache) | CPU concurrent | 243.8 s | 23.686 | 72.3% / 65.7% |
| v9 | + incremental admission | GPU after Gemma | 267.2 s | 22.689 | 91.1% / 84.5% |
| v10 | + expansion, bounded prior, stale proposals | GPU after Gemma | 236.0 s | 19.756 | 90.8% / 87.8% |

Open after this step: the host CPU step (attribution; resolved in
../20260911-host-power-step/: the lease-renewal thread spinning after the
preparation-phase leases expired), Qwen's assistance is
still rejected on measurement after a handful of tokens in every run, the
Qwen file read remains the largest idle cost and cannot be hidden on a 30 GB
host, and objective-level rejection reasons are still absent from the
decision log.

## Evidence

- CHANGES.json: before/after hashes of the eight changed or added files;
  source-before/ holds the before-images.
- analyze.py: per-window, per-context, phase and server-startup summaries.
- physical/dev3-ready-parent-v10, physical/preflight-expansion-v10 and
  inputs/: exact copies of the remote artifact tree, the passed preflight and
  the v10 campaign/rig manifests (only campaign_id, rig path, phone session
  root and whole-state directory differ from v9). ANALYSIS.json and
  CONTEXT_WINDOWS.json come from analyze.py.
