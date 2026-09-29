# Measurement context, incumbent recovery and lossless helper events

Status: implementation complete for plan sections 2-4 and 6; full suite and
both replay goldens unchanged; three unchanged three-request gates run.
v7 (sections 2-4, cold page cache) PASS 3/3 but produced no concurrency;
v8 (sections 2-4, warm cache) PASS 3/3 and physically shows detect -> 0% ->
recover around Llama's CPU co-run; v9 (sections 2-4 + 6, warm cache) PASS 3/3
with the CPU start refused, Gemma 91% assisted and the lowest fleet energy of
the series. Sections 7-9 were not started. No commit, push, baseline rerun or
longer trace.
This continues ../20260911-epoch-helper-admission/ (v6) without repeating
calibration or baselines.

## What v6 actually did at token 152

v6's own window records (ADAPTIVE_DECODE_OBSERVATIONS.json) show the sequence.
Gemma exploited 100% over the 16 resident layers at 417 ms/token and 32 J/token
(whole-fleet, assumed 4.5 W phone). Llama started on the READY CPU parent at
61.27 s. Gemma's next windows, still 100% and still HTTP batch 1, measured
443-522 ms/token and 64-78 J/token because the fleet energy now included
Llama's CPU draw and the CPU was contended. The controller compared those
windows against baseline evidence taken before the co-run (windows 1-4, about
455 ms/token and 56 J/token), found the incumbent no longer beneficial, moved
to 50% (window 39), then eliminated it and issued 0% at token 152 (79.75 s).
Llama finished at 82.68 s. The remaining 138 tokens ran at 0%: 460 ms/token
and about 63 J/token, versus 32 J/token assisted before the co-run.

Two mechanisms, both in the controller, explain the tail:

1. Comparability. `_context_identity` and `_valid_records` partition evidence
   by artifact, parent, context bucket, batch and membership only. Another
   request executing on the desktop changes every window's latency and
   whole-fleet energy but did not change the context, so pre-co-run baseline
   evidence was compared with co-run assisted evidence.
2. Recovery. `eliminated_policy_reasons` is cleared only on a batch change,
   and `_next_after_window` re-enters PROBING only in stage `initial_baseline`.
   Once the incumbent was rejected inside the co-run there was no boundary at
   which the controller could reconsider it after Llama finished.

The exact per-window reason was not recoverable from v6's export because 4,085
LEASES_RENEWED events filled the 4,096-entry helper-event buffer, and every
retained event carried `event_index` 4096 (the index was the buffer length).

## Corrections (sections 2-4)

Measurement context is a separate identity from execution compatibility.

- `runtime_requests.runtime_external_desktop_activity(ticket, start, end)`
  names the other acquired tickets holding server (non-phone) execution leases
  across a decode window, from the runtime controller's acquisition history.
  Queued requests, lease renewals and phone-only work do not change it. None
  means the history cannot cover the window.
- `adaptive_decode_control.record_adaptive_decode_window` stamps that identity
  on each raw window observation (`external_activity_sha256`) and records a
  `CONTEXT_CHANGED` helper event with reason
  `EXTERNAL_DESKTOP_ACTIVITY_CHANGED`, the peer ticket ids and both identities
  when it changes. ASSISTANCE_DECISION events now carry the identity too.
- `windows.record_window` treats a changed identity like a compatible
  membership change: the crossing window keeps its latency and energy in
  accounting but is not measurement-eligible; the context record start moves;
  eliminations, probe budgets and verification state reset; a qualifying
  incumbent enters the existing bounded context monitor (paired verification
  in the new context, including acknowledgement and warmup); with no incumbent
  the previous winner or cached winner becomes the single probe candidate.
  The first observed identity only names the context already measured. An
  unknown identity (None) never resets anything.
- Request-wide limits are preserved: `probe_tokens`, `verification_attempts`
  and `maximum_probe_attempts_per_context` are not refilled by a context
  change. In the synthetic regression a second change mid-verification
  consumes the second and last verification attempt.
- Receipts gain optional `external_activity_sha256` and
  `external_activity_changed`; both are omitted from JSON when absent, so
  historical record hashes and replay goldens are unchanged.
- Helper events: `event_index` is now monotonic from the previous event (not
  the buffer length); consecutive LEASES_RENEWED events of one request collapse
  into the latest lease state with `renewal_count`, `first_observed_at_us` and
  `first_event_index`; when the buffer still overflows, renewals are evicted
  before any lifecycle or reason event.

Outcomes remain distinct in tests: a measured negative pair keeps 0% for that
context (`CURRENT_PAIR_NOT_IMPROVED` / `MEASURED_REJECTION`); an incomplete
pair defers with the existing bounded retry; helper loss stops assistance
first (`PHONE_HELPER_UNAVAILABLE`) and a later generation must be re-verified;
a pending maintenance drain is applied before any monitoring; insufficient
remaining tokens stay at baseline with eliminations cleared but no probe.

## Section 6: energy-aware concurrency admission (implemented, not deployed)

v6's Llama decision compared the READY CPU route (route energy 979-1,193 J,
protected-work extension upper 384 J) against the GPU control route whose
route energy lower bound was 5,574 J. Of that, about 5,330 J was idle-domain
energy charged for a predicted 424 s queue wait; Gemma's own execution energy
during that wait is spent in both scenarios. Lost phone assistance was not a
term at all, although it cost roughly 4 kJ in v6.

- `costing._candidate_marginal_system_cost` now records
  `protected_overlap_us`, `protected_overlap_idle_uj` (idle-charge-domain
  power over the part of the queue wait that overlaps the protected critical
  path) and `route_energy_incremental_lower/upper_uj` (route energy minus that
  common idle) in `marginal_system_cost`.
- `objectives.protected_work_budget_allows` compares incremental energies and
  adds `assistance_loss_upper_uj` to the start-now side. None (unknown loss)
  is never admitted; `strict` stays the default.
- `AdaptiveDecodeController.assistance_summaries()` reports, per active
  request, whether phone assistance is in use and the operational
  baseline-upper minus assisted-lower energy per token; missing bounds are
  None. `AutomatedSelectionMixin._protected_assistance_loss_upper_uj` sums
  that loss over the tokens each assisted peer would decode during the
  candidate's upper service time (computed once per selection context).
- Isolated CPU calibration is still the only CPU evidence; the catalog has no
  route profile keyed by `large_phase_id`, so concurrent CPU execution is not
  separately qualified. That remains open and is why the CPU route's own
  concurrent slowdown (v6: 21.4 s versus 10.8 s isolated service) is covered
  only by the service upper bound (25.3 s), not by a concurrent profile.

Replaying v6's Llama numbers through the new rule: incremental GPU lower is
about 241 J, CPU incremental upper 1,193 J, extension 384 J, so the budget
(241 x 0.99 - 1,193) is negative and the CPU start is refused before lost
assistance is even counted. Available CPU capacity alone no longer admits an
energy-aware start.

## Tests

TESTS.json records the runs. New cases: first-identity no-op; change with a
qualifying incumbent -> monitor -> measured rejection stays 0%; change back ->
eliminations cleared -> re-probe -> incumbent restored; unknown identity
never resets; change midway through verification; insufficient remaining
tokens; helper loss and later generation; drain precedes monitoring; runtime
integration on real acquisition history with HTTP batch fixed at 1; event
buffer saturation with coalescing, eviction order and monotonic indices;
incremental budget with common-idle cancellation and lost assistance;
lost-assistance bound from active sessions; costing keys for immediate and
waiting routes.

Full suite: 1,344 tests, 2 errors, both reproduced byte-identically with this
report's before-images substituted (`test_arrival_coordinator...v12...` and
`test_physical_residency...raw_endpoint_fields`), so they predate this work
and are reported, not counted. `test_cached_synthetic_refinement_is_below_ten_
milliseconds` is timing-based and flaked once at 10.5-12.5 ms before the
per-context summary cache; it passed 3/3 and 2/2 afterwards. The known
cold-cohort wait is excluded as before. Both replay goldens unchanged (v3
5d52e867..., v8 24192446...).

## Physical gate v7: PASS 3/3, but no concurrency occurred

Only the section 2-4 change set was deployed (CHANGES.json, remote hashes
match). Preflight passed 103 checks with 2 advisory warnings. Unchanged
burstgpt_dev3_long_v1: Gemma 36 (1 s), Llama 37 (61 s), Qwen 50 (91 s).

| Request | Executor | Arrival->execution | Execution | Assisted tokens | Fraction-weighted | Layers/columns used |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Gemma 36 | cold desktop | 53.0 s | 136.98 s | 63/292 (21.6%) | 43.75 (15.0%) | 8 layers only: C3840/25% 8, C7680/50% 21, C11520/75% 11, C15360/100% 23 |
| Llama 37 | GPU control (after Gemma) | 130.0 s | 1.45 s | 0 | 0 | none |
| Qwen 50 | hot desktop | 106.2 s | 48.94 s | 12/71 (16.9%) | 12 (16.9%) | 6 layers C17408/100% |

Duration 355.195 s; fleet energy 28.275 / 28.412 / 28.549 kJ at 3 / 4.5 / 6 W.
Zero fallback recoveries. Four phone loads (gen 1-4), each with individual
LOADING/VERIFIED/READY; no reload was caused by a fraction change.

Why it differs from v6: Gemma's server log shows `load_tensors` from 1.7 s to
44.4 s (mmap, 24 GB read from disk) versus `listening` at 3.67 s in v6. The
desktop has 30 GB RAM; v4/v6 inherited Gemma's weights in the page cache from
the immediately preceding attempt, while v7 ran right after the preflight had
evicted them. Gemma therefore first decoded at 58.75 s instead of 9.7 s. At
Llama's arrival (61.03 s) phone layout generation 2 was loading (54.8-90.4 s),
the protected-work power was not measurable, the READY CPU parent was rejected
with MARGINAL_SYSTEM_COST_UNKNOWN (plus MODEL_EPOCH_AUDIT_ONLY), and Llama
waited for the GPU (QUALIFIED_FALLBACK_NO_SAFE_ALTERNATIVE). No external
context change happened; both Gemma and Qwen saw only the empty-peer identity
(b95f9ad0...). The measurement-context path was therefore not exercised.

Gemma's unassisted tail (tokens 79-292) is fully explained by the export: the
8-layer exploration spent 51 of the 80 request-wide probe tokens before the
16-layer layout was READY (90.9 s), a learning candidate was rejected
(LEARNING_NO_PAIRED_IMPROVEMENT at token 42), two probes were incomplete
(tokens 59, 67) and from token 79 every decision reads INSUFFICIENT_OPPORTUNITY
with 29 probe tokens left; the 16-layer expansions at 91.4 s and 102.9 s were
never affordable. That is a probe-budget limit under a late start, not the v6
mechanism, and this work did not change budgets.

Coalescing worked: 88,844 helper events were emitted (indices 0-88843) and the
retained export holds 16 LEASES_RENEWED rows plus every lifecycle, decision
and rejection event (87 ASSISTANCE_DECISION for the three requests).

## Physical gate v8: unchanged requests, warm page cache

Only the section 2-4 change set was deployed (same hashes as v7). The single
environmental difference from v7: Gemma's and Llama's weight files were read
once into the page cache immediately before launch (`cat file > /dev/null`,
33 s), reproducing the warm-cache condition v4 and v6 inherited from their
preceding attempts. Gemma's server listened at 6.96 s (v6: 3.67 s, v7: 45.9 s).

| Request | Executor | Arrival->execution | Execution | Assisted tokens | Fraction-weighted | Layers/columns used |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Gemma 36 | cold desktop | 13.5 s | 128.58 s | 211/292 (72.3%) | 191.75 (65.7%) | L24/C15360/100% 126, L16/C15360/100% 30, L16 25/50/75% 11 each, L8 75%/100% 11 each |
| Llama 37 | READY CPU parent (concurrent) | 0.25 s | 27.25 s | 0 | 0 | none |
| Qwen 50 | hot desktop | 97.2 s | 47.25 s | 11/71 (15.5%) | 11 (15.5%) | L6/C17408/100% |

Duration 243.794 s; fleet energy 23.539 / 23.686 / 23.834 kJ at 3 / 4.5 / 6 W.
Four execution attempts (three requests plus the startup preload), zero
fallback recoveries, ticket-bound proofs for all four, USB restored, 299 MiB
GPU used after cleanup. Phone loads: HTP0 and HTP1 once (Gemma gen 1), HTP2
twice (Gemma gen 3 at 62-79 s, then the Qwen replacement at 188-210 s); no
reload followed a fraction change.

Llama ran on the CPU beside Gemma from 61.25 s to 88.50 s, admitted under the
old rule (BUDGETED_PROTECTED_WORK_ENERGY_SAVING). Gemma's measurement-context
identity changed twice while its HTTP batch stayed 1, exactly at the peer's
start and finish:

| Gemma window | Tokens | Paid time | Context | Controller action |
| --- | --- | ---: | --- | --- |
| 26 (crossing) | 98-102 | 61.21 s | -> {llama attempt 0} | not measurement-eligible; CONTEXT_CHANGED event with the peer ticket id; incumbent 100%/16L retained as prior; monitor requested |
| 27-38 | 102-149 | 62.5-89.3 s | co-run | CONTEXT_MONITOR_UNAFFORDABLE (14 probe tokens left, verification attempts spent): baseline at 0% for the whole co-run; 614 ms/token, 87.6 J/token whole-fleet |
| 39 (crossing) | 149-153 | 91.16 s | -> {} (Llama done) | not measurement-eligible; CONTEXT_CHANGED with an empty peer set; eliminations cleared |
| 40-42 | 153-164 | 93-96 s | quiet | fresh baseline windows: 456 ms/token, 57.8 J/token |
| 43-74 | 164-290 | 97.8-142.3 s | quiet | previous winner re-probed and exploited: 100% over 24 layers (layout gen 3), 365 ms/token, 38.7 J/token |

Comparable eligible-window means (CONTEXT_WINDOWS.json): quiet assisted
100%/16L 413.6 ms and 31.7 J per token; co-run baseline 614.1 ms and 87.6 J;
quiet baseline 456.1 ms and 57.8 J; quiet assisted 100%/24L 365.0 ms and 38.7 J.
The co-run energy per token includes Llama's own CPU draw and is not an
isolated cost of the co-run.

Pass conditions from the plan, as observed:

- Llama's start and finish were detected at HTTP batch 1: yes, twice, with
  the peer ticket id recorded.
- The compatible winner was preserved as prior evidence and re-verified only
  under measurements from the new context: yes; because the request-wide
  verification budget was already spent, the monitor was correctly refused
  (CONTEXT_MONITOR_UNAFFORDABLE) rather than granting new budget, and Gemma
  stayed at 0% for the co-run.
- Reconsideration after Llama finished even though Gemma was at 0%: yes; the
  previous winner was re-probed at token 164 and exploited for the rest.
- No unexplained permanent tail: every zero-assistance decision names its
  reason and context; the 47-token co-run tail is explained by the budget.
- Energy and coverage remain distinct from a savings claim. Against v6 this
  is -1.708 kJ (-6.7%) and -6.2 s with coverage 65.7% versus 35.4%
  fraction-weighted; against sequential v4 it is still +1.124 kJ (+5.0%)
  although 23.1 s shorter. Successive scheduler versions, not a matched A/B.

Qwen assisted 11 tokens then INCONCLUSIVE, as in v7 (v6: 27 tokens); not
investigated here.

## Physical gate v9: unchanged requests with the section 6 admission rule

CHANGES_ADMISSION.json records the eight additional files deployed (remote
hashes match). A fresh preflight passed 103 checks. Weights were pre-warmed as
for v8 (the cache was still warm, 0 s).

| Request | Executor | Arrival->execution | Execution | Assisted tokens | Fraction-weighted | Layers/columns used |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Gemma 36 | cold desktop | 12.4 s | 123.10 s | 266/292 (91.1%) | 246.75 (84.5%) | L16/C15360/100% 211, L16 25/50/75% 11 each, L8 75%/100% 11 each |
| Llama 37 | GPU control after Gemma | 75.5 s | 3.76 s | 0 | 0 | none |
| Qwen 50 | hot desktop | 116.7 s | 48.81 s | 0/71 | 0 | none (PHONE_HELPER_UNAVAILABLE x17) |

Duration 267.222 s; fleet energy 22.573 / 22.689 / 22.804 kJ at 3 / 4.5 / 6 W.
Five execution attempts (three requests, the startup preload and Llama's
replan at Gemma's completion), zero fallback recoveries, ticket-bound proofs
for the four executed tickets, USB restored, 299 MiB GPU used after cleanup.
Three phone loads (HTP0, HTP1, HTP2), all Gemma's artifact; no fraction change
caused a reload.

At Llama's arrival (61.04 s) the READY CPU parent was admitted by route
generation (rejection_reasons empty, QUALIFIED) yet the decision was
QUALIFIED_FALLBACK_NO_SAFE_ALTERNATIVE for the GPU control route, i.e. every
alternative was refused at objective ranking. The recorded marginal cost of
the CPU route carries the new keys: protected_overlap_idle 0.47 J (37.7 ms
queue), route energy incremental upper 1,192.2 J, extension upper 316.1 J. The
GPU route's queue was 428.5 s inside Gemma's predicted critical path, so its
5,595.9 J lower bound becomes about 249 J incremental and the budget
(249 x 0.99 - 1,192) is negative before lost assistance is counted; Gemma was
assisted at 100%/16L at that moment, so the loss term was also positive. The
objective-level rejection string itself is not exported by the decision log
(only route-generation reasons are), a gap noted below.

Gemma kept its incumbent through the whole request: no external context
change was observed (only the empty-peer identity), no zero-assistance
decision after token 30, and 204 eligible 100%/16L tokens. Its per-token
whole-fleet energy at 100%/16L was 32 J before 73.5 s and 45-47 J after, with
latency improving from 426 to 396 ms/token at the same moment; the host CPU
package stepped from about 42 W to 80 W at 73.5 s while no scheduler event,
server launch or phone load occurred (decision log, helper events and server
logs checked). v7 shows the same step at 74 s. This is unattributed (DVFS is
the obvious candidate) and it inflates every later per-token figure in v7 and
v9 relative to v8; the fleet totals are unaffected as measurements.

Successive-version comparison (same three requests, preparation-inclusive,
assumed 4.5 W phone, all diagnostic attribution):

| Run | Change set | Llama route | Duration | Fleet energy | Gemma assisted / fraction-weighted |
| --- | --- | --- | ---: | ---: | --- |
| v4 (prior report) | epoch/helper admission | GPU after Gemma | 266.854 s | 22.562 kJ | 93.5% / 86.9% |
| v6 (prior report) | + READY CPU dispatch | CPU concurrent | 250.013 s | 25.394 kJ | 43.8% / 35.4% |
| v7 | + sections 2-4, cold cache | GPU after Gemma | 355.195 s | 28.412 kJ | 21.6% / 15.0% |
| v8 | + sections 2-4, warm cache | CPU concurrent | 243.794 s | 23.686 kJ | 72.3% / 65.7% |
| v9 | + section 6, warm cache | GPU after Gemma | 267.222 s | 22.689 kJ | 91.1% / 84.5% |

v9 lands within 0.6% of v4's energy with 91% Gemma coverage, and v8 shows the
recovery mechanism under a real co-run at 1.7 kJ below v6. None of this is a
matched A/B against a frozen control; the page-cache state alone moved v7 by
several kJ. A formal savings claim still needs the separately authorized
matched control.

## Open observations (not changed here)

1. Objective-level rejection reasons (PROTECTED_WORK_DELAY and the others in
   `_objective_row_rejection`) are not in the decision log's candidate rows;
   only the executed attempt's ticket keeps `decision.rejected`. Adding them to
   the log record would change replay goldens and needs its own step.
2. The unattributed host CPU power step at about 74 s in v7 and v9.
3. Qwen received no phone layout in v9: the third layout (deferred at 37 s by
   PHONE_RESIDENCY_LEARNING_RETAINED) started loading Gemma's 24-layer artifact
   at 136.3 s, the instant Gemma completed, and Qwen decoded with all three
   sessions holding Gemma. v8 instead replaced HTP2 with Qwen at 188 s. Qwen
   assisted 27 (v6), 12 (v7), 11 (v8) and 0 (v9) tokens.
4. Request-wide probe budget: in v7 the 8-layer exploration consumed 51 of 80
   probe tokens before the 16-layer layout was ready; in v8 the co-run monitor
   was unaffordable with 14 tokens left. Budgets were deliberately not changed.
5. Concurrent CPU execution has no separately qualified route profile; the
   admission rule relies on the isolated CPU profile's service upper bound.
6. A transient `adb: device not found` at 26.6 s (v9) deferred phone
   telemetry for 0.1 s and recovered; no effect on the run.

## Evidence

- physical/dev3-ready-parent-v7, -v8, -v9: exact copies of the remote
  artifact trees (run/RESULT.json, decision log, observations, snapshots,
  server logs) plus ANALYSIS.json and CONTEXT_WINDOWS.json from analyze.py.
  physical/preflight-context-recovery-v7, physical/preflight-admission-v9 and
  physical/resolve-context-recovery-v7 hold the passed preflights and the
  resolved configuration (catalog sha256 2128811c..., unchanged from v6).
  Remote roots: /mnt/storage/s42-phase-concurrency-20260911-v1-nEEi6I/
  dev3-ready-parent-v7|v8|v9 and deploy/.
- inputs/: the v7, v8 and v9 campaign and rig manifests (only campaign_id,
  rig path, phone session root and whole-state directory differ from v6).
- CHANGES.json: sections 2-4 source/test before, after and deployed hashes.
  CHANGES_ADMISSION.json: section 6 hashes (not deployed).
- TESTS.json, analyze.py, physical/*/ANALYSIS.json.
