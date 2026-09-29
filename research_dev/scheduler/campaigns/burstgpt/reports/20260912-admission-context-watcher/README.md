# Admission safety, explicit context for request 49, cheaper queued watching

Status: complete. Admission safety, the 2,560-token Gemma desktop
qualification and the event-driven watcher are deployed (v15); the
controller then learned to run a compatible prior under monitoring when a
short request cannot afford its paired verification and to use the
nearest prompt-length bucket's winner as a starting point (v16). The
24-request trace completed 24/24 in every arm. Matched fail-closed
comparisons: v15 vs v15c 15.96% whole-fleet saving (p95 latency ratio
1.05); v16 vs v16c 25.12% (p95 1.01, adaptive 5% faster), meeting the
25% target on this pair.

Context: the first 24-request attempt (report 20260911-host-power-step,
section 9) aborted at arrival 49 because a 2,375-token Gemma request could
never be preallocated on the 2,048-token desktop context, the mandatory
desktop control could not be generated, and the runner treated the
submission error as fatal. Host diagnostics from the same run showed the
runner's per-request helper watcher threads consuming 1.3-2.0 CPU-s per
5 s each while their requests were merely queued.

## 1. Admission safety (scheduler, coordinator, runner, preflight)

- `_unified/automated_requests_ops/admission.py` judges a request's shape
  statically against the registered catalog: for every preallocated
  coordinator of the model, capacity tokens = context resource capacity x
  context token quantum; the request fits when prompt + output tokens are
  at most that capacity (equivalently ceil(tokens/quantum) slots fit).
  Two exact, permanent reasons: `REQUEST_EXCEEDS_CONTEXT_CAPACITY` (no
  coordinator of the model can hold it) and
  `REQUEST_EXCEEDS_DESKTOP_CONTROL_CONTEXT` (some coordinator could, but
  the mandatory desktop control cannot). Anything else is not judged here:
  resource contention still queues through the calendar preview.
- `submit_automated_request` raises `RequestShapeUnsupportedError`
  (status REJECTED, reason, details) before any planning. No journal
  record is written for a rejected request, matching the existing
  contract for refused submissions.
- `CanonicalArrivalCoordinator.submit` returns None for that error and
  keeps the rejection (`rejections()`, `RuntimeSubmissionRejection`); the
  request is never registered for drain, every other request runs. Any
  other scheduling error still propagates as before.
- The runner records each rejection with its exact reason, index and
  token counts (`RESULT.rejected_requests`, `counts.rejected`,
  `counts.submitted`), excludes rejected requests from journal-coverage
  expectations, and reports `status: PARTIAL` instead of PASS, so a run
  with rejections can never be mistaken for a completed workload
  (`compare_ab` also refuses it as incomplete).
- Before the paid interval the runner screens every trace request
  (`RESULT.request_shape_preflight`), and the campaign preflight passes
  every trace row to `run_physical_preflight(trace_requests=...)`, whose
  new `request-shapes` check BLOCKS the preflight when any request can
  never be preallocated; the report lists them under
  `request_shapes.unsupported` and `trace.unsupported_requests`.

## 2. Explicit context for request 49

Request 49 (`burstgpt-v2:88132`) needs 1,884 + 491 = 2,375 tokens. The
Gemma desktop plan preallocated a 2,048-token context in 512-token quanta
(capacity 4). The smallest quantum-aligned context that preserves the
request's full history is 2,560 tokens (5 slots of 512). It is qualified
the same way the 2,048-token plan was, with the capacity-aware desktop
parent calibration (`desktop_parent_calibration.py --desktop-parent-role
cold --context-size 2560 ...`): one server load plus one cold and one hot
completion of the largest fitting Gemma request of the trace, which is
request 49 itself, so the qualification run executes exactly the shape
that aborted the campaign.

The first qualification attempt at 2,560 tokens selected 29 GPU layers
(the GPU had 16.4 GB free, against 13.4 GB when the 2,048-token plan was
qualified), a new desktop placement (18d62d12...) with a 15.5 GB VRAM
peak. That would have changed the desktop control's identity and
orphaned every measured Gemma route and assistance observation, and it
leaves no VRAM headroom for the other resident model. The qualification
was therefore repeated with `--maximum-gpu-layers 23`, pinning the
already-qualified 23-layer placement (a87d0996...) so only the context
changes; the plan's capacity parameters are re-measured, not inherited.
Context shifting was not needed and is not enabled anywhere (no
`--context-shift`, `cache_prompt` false, `ignore_eos` true).

## 3. Event-driven request-helper watcher

`adapters/runtime.py`: the per-request watcher used to capture a full
runtime snapshot (two canonical JSON files, an nvidia-smi fork, monitor
deep copies) and re-project the helper envelope every 50 ms, holding the
scheduler's runtime lock three times per iteration, even when the request
was queued and nothing had changed. The placement controller now keeps a
helper-state generation that advances on every phone layout event,
request helper event, request acquisition and pending-candidate update
(`runtime_helper_state_generation()`); the watcher re-plans only when it
moves or a bounded 1 s fallback elapses, and otherwise only re-reads the
ticket every 50 ms (cheap, no lock). Background preparation, ownership
arbitration, progressive preload stages, the 5 s ready refresh and
next-boundary attachment are unchanged; existing watcher tests pass
unmodified, and a regression counts snapshots while queued (one at
start, none in 0.6 s, one within 50 ms of a state change, fallback paced).

### Qualification runs (physical/desktop-parent-calibration-cold-2560-v2, -v3)

| | 2,048-token plan (Sep 9) | v2: 2,560, layers free | v3: 2,560, `--maximum-gpu-layers 23` |
| --- | ---: | ---: | ---: |
| GPU layers / placement | 23 / a87d0996 | 29 / 18d62d12 | 23 / a87d0996 |
| live free VRAM at qualification | 13.39 GB | 16.41 GB | 16.41 GB |
| peak process VRAM | 12.69 GB | 15.50 GB | 12.70 GB |
| required with reserve | 13.08 GB | 16.29 GB | 13.09 GB |
| model load | 61.0 s | 34.4 s | 4.8 s (page cache warm) |
| cold / hot completion of request 49 (1,884 + 491 tokens) | n/a (did not fit) | 193.3 / 192.1 s | 235.2 / 235.2 s |
| status | PASS | PASS (not adopted) | PASS (adopted) |

v3 is the adopted plan: evidence f220dfd5..., the 512-token quantum
unchanged, so `context:cold:desktop` and `context:cold:cpu` get capacity
5 (2,560 tokens) and every Gemma coordinator carries context 2,560; Qwen's
plan is untouched (2,048 / 1,024 / capacity 2). The catalog re-materialized
to d8f9397e.... Preflight v15: 104 checks PASS, the new `request-shapes`
check reports all 24 trace requests fit (0 unsupported), and the Gemma
desktop-parent capacity check selects the same 23-layer placement
(required 13.09 GB against 16.41 GB free).

## 4. Validation

- New tests: request-shape admission (4), runner rejections (2), preflight
  block (1), queued-watcher pacing (1); existing watcher tests pass
  unmodified. Full suite 1,389 tests: the two pre-existing errors plus the
  known 10 ms timing flake, which passes in isolation; replay goldens
  unchanged (TESTS.json).
- All-request preflight on the rig: PASS, 24/24 shapes supported.

## 5. Rerun of the 24-request trace (v15, energy-aware) and the matched desktop control

v15 adaptive (physical/sparse24-v15/): PASS, 24 of 24 requests completed
in 1,510.0 s, no rejections (`counts.rejected` 0, `request_shape_preflight`
24 checked / 0 unsupported), fleet 120.892 kJ at 4.5 W assumed phone power
(CPU package 73.196, GPU board 45.712, phone 1.984 kJ). Request 49 ran on
the Gemma desktop at the 2,560-token context: 137.7 s queued, 9.0 s to
first token, 217.3 s of decode for its 491 tokens, 96.5% of them
phone-assisted at fraction 100%.

| | v14 attempt 1 (875 s, aborted) | v15 (1,510 s, complete) |
| --- | ---: | ---: |
| LEASES_RENEWED / authorization expiries | 236 / 0 | 347 / 0 |
| runtime-lease-renewal thread CPU | 0.73 s | 1.52 s |
| request-helper watcher threads / CPU | 19 / 111.9 s | 27 / 31.2 s |
| watcher CPU per queued request-second | 0.045 (2484 queued s) | 0.006 (5,052 queued s) |
| busiest watcher thread | 0.21 cores | 0.027 cores |
| runtime snapshot files written | 18,631 | 3,498 |
| package power mean / max | 48.8 / 92.1 W | 48.1 / 99.1 W |

Scheduling decisions were left to the unchanged policy: Gemma's three
long requests ran 92.5%, 93.2% and 96.5% phone-assisted at 100% fraction;
its three short ones (11-41 tokens) stayed on the desktop; Qwen's measured
rejection held for 12 of 15 requests, while requests 35, 42 and 43 ended
at 100% fraction with 61%, 77% and 95% of tokens assisted; the three Llama
overlays ran on the desktop control after waiting 250-280 s behind GPU
work. Queue waits are large under this trace (up to 431 s for Qwen
arrivals behind Gemma's residency) and are recorded per request in
ANALYSIS_SPARSE24.json; they are not optimized here.

### Matched desktop-baseline control (v15c) and comparison

The control arm (physical/sparse24-v15c/) reran the same trace in
`desktop-baseline` mode from the adaptive run's materialized catalog
(byte copy, catalog d8f9397e), the same deploy tree and binaries, the same
observation stores and startup preparation; only the selection mode, the
campaign id and the phone session roots differ. Preflight PASS (24/24
shapes), run PASS 24/24 in 1,421.3 s, no phone work (0 s phone active,
0 renewals, watcher threads 8.8 CPU-s total, 82 snapshot files).

`compare_ab.py` initially refused the pair: every launch writes its own
SOURCE_MANIFEST.json embedding the resolved campaign, so the two manifest
digests differ although both pin the same 387 source files at the same
commit. The comparator now accepts two manifests that hash to what their
results recorded and pin identical files, head and branch
(`source_identity: source-files-equal`); binaries, CUDA graph modes and
every other identity field must still be equal, and a missing or
mismatching manifest still blocks (test added). Comparison:
physical/sparse24-v15-AB_COMPARISON.json (`comparison_valid: true`).

| | control (desktop-baseline) | adaptive (energy-aware) | change |
| --- | ---: | ---: | ---: |
| duration | 1,421.3 s | 1,510.0 s | +6.2% |
| fleet energy at 4.5 W assumed phone power | 143.847 kJ | 120.892 kJ | -22.955 kJ, saving 15.96% |
| CPU package / GPU board / phone | 97.847 / 44.756 / 1.244 kJ | 73.196 / 45.712 / 1.984 kJ | -24.65 / +0.96 / +0.74 kJ |
| package power mean | 68.2 W | 48.1 W | |
| per-request latency ratio (adaptive / control) | | mean 0.99, p50 1.00, p95 1.05, max 1.09 | limit 1.25 |
| throughput | 1.600 tok/s | 1.506 tok/s | |
| residency reloads / transitions | 6 / 6 | 6 / 6 | |
| tokens by fraction (adaptive) | | 0%: 501, 25-75%: 33, 100%: 1,348 | |

The saving is the trace-level whole-fleet figure the comparator computes
(target 25%, `target_met: false`); phone energy is the assumed 4.5 W
active / 875 mW idle policy shared by both arms, and the estimator
errors it reports (energy MAPE 17.6% control, 20.9% adaptive) concern the
per-request diagnostic windows, not the boundary totals. The saving comes
from the CPU package: Gemma's and part of Qwen's FFN work moved to the
phone while the GPU total stayed within 1 kJ. Request 49 completed in
both arms (control: 97.4 s queued, 226.3 s decode; adaptive: 137.7 s
queued, 217.3 s decode, 96.5% assisted). This is one matched pair on one
day, not a replicated estimate.

## 6. Controller optimization for short requests (v16)

The v15 energy map (below, section 5) left 83% of the energy in decode
and showed which requests the controller could not help: 12 of 15 Qwen
requests and 3 short Gemma ones ended at 0%. Per-window decisions explain
each: (i) 171 Qwen tokens in requests of 26-71 tokens sat at 0% with
VERIFICATION_INCOMPLETE / COMPLETE_PAIR_BUDGET although the same winner
had just been verified on the same layout, because the operational
verification demands a complete paired block the request cannot afford
and then falls back to the desktop; (ii) 81 tokens saw
INSUFFICIENT_OPPORTUNITY while a verified winner existed for the same
layout in a neighbouring prompt-length bucket, because the operational
seed matched the exact power-of-two bucket only; (iii) the first requests
on a new layout (124 tokens) cannot afford the 80-token four-candidate
sweep and, with no prior anywhere, correctly stay on the desktop.

Two online changes, no assumption about arrivals:

- Prior under monitoring (`adaptive_decode_ops/sequencing.py`). When the
  seeded operational winner's paired verification is unaffordable but the
  winner itself can still be measured (one warm-up window plus one
  measured window), the request starts at that winner as a starting point
  (reason `VERIFICATION_MONITORING`). Every measured window must beat the
  historical baseline by the configured saving margin with the usual
  uncertainty and stay within the latency limit; the first window that
  does not ends the monitor, eliminates the policy for this context
  (`PRIOR_MONITOR_NOT_IMPROVED`) and returns the request to the desktop.
  Nothing is promoted to incumbent, no verification attempt is spent, and
  the retry limit and too-short cases are unchanged.
- Nearest-bucket starting point (`adaptive_decode_ops/candidates.py`). The
  operational seed searches the exact bucket first and then the nearest
  buckets with evidence, the same widening the historical seed already
  uses; a measured rejection in the request's own context still wins.

Six controller tests cover these paths; two existing tests changed
contract consciously (a different prompt bucket is a starting point, an
unaffordable pair monitors instead of deferring).

v16 adaptive (physical/sparse24-v16/): PASS 24/24 in 1,419.1 s, no
rejections, fleet 109.592 kJ at 4.5 W (CPU package 64.226, GPU board
43.355, phone 2.010 kJ). Eight requests that v15 left at 0% now ran their
prior under monitoring (`VERIFICATION_MONITORING`) and none was rejected
by the monitor: Qwen assisted tokens rose from 390 to 628 of 830 (47% to
76%), Gemma from 991 to 1,014 of 1,114 (89% to 91%). Requests still
unassisted: the first Qwen requests on each new layout (34 at 2.8%, 35
and 41 at 0%: no prior anywhere and the sweep unaffordable) and four
requests of 3-14 tokens whose helper was not attached before they ended
(48, 52, 53, 55). Renewals 402, expiries 0; watcher CPU 27.0 s over
30 threads; the run was 91 s shorter than v15 because assisted Qwen
windows decode faster than the CPU-bound baseline and queues drained
sooner (total queue wait 4,206 s vs 5,052 s).

| decode phase (pkg + GPU) | v15c control | v15 adaptive | v16 adaptive |
| --- | ---: | ---: | ---: |
| Qwen decode energy / J per token | 61.9 kJ / 74.6 | 49.1 kJ / 59.2 | 40.3 kJ / 48.5 |
| Gemma decode energy / J per token | 64.5 kJ / 57.9 | 50.3 kJ / 45.2 | 50.0 kJ / 44.9 |
| loads and prefill | 14.4 kJ | 17.4 kJ | 13.6 kJ |
| idle gaps | 1.9 kJ | 2.1 kJ | 2.5 kJ |

### Matched v16c control and comparison

The control arm reran the same trace in `desktop-baseline` mode from the
same prematerialized catalog (d8f9397e) on the same deploy tree and
binaries: PASS 24/24 in 1,499.3 s, 146.353 kJ at 4.5 W (CPU package
98.127, GPU 46.914, phone 1.312 kJ), no phone work. `compare_ab`
(physical/sparse24-v16-AB_COMPARISON.json, `comparison_valid: true`,
`source_identity: source-files-equal`):

| | v16c control | v16 adaptive | change |
| --- | ---: | ---: | ---: |
| duration | 1,499.3 s | 1,419.1 s | -5.3% |
| fleet energy at 4.5 W | 146.353 kJ | 109.592 kJ | -36.76 kJ, saving 25.12% |
| CPU package / GPU board / phone | 98.13 / 46.91 / 1.31 kJ | 64.23 / 43.36 / 2.01 kJ | -33.90 / -3.56 / +0.70 kJ |
| latency ratio adaptive / control | | mean 0.96, p50 0.96, p95 1.01, max 1.09 | limit 1.25 |
| throughput | 1.517 tok/s | 1.602 tok/s | +5.6% |
| tokens by fraction (adaptive) | | 0%: 235, 25-75%: 33, 100%: 1,609 | |
| residency reloads / transitions | 6 / 6 | 6 / 6 | |

The comparator's 25% energy-saving target is met (`target_met: true`) on
this pair, with the adaptive arm also faster per request. Two matched
pairs now exist on this catalog: v15/v15c (15.96%) and v16/v16c
(25.12%); the two controls differ by 1.7% in energy and 5.5% in duration
between themselves, which bounds the run-to-run noise of a single pair.
Phone energy remains the assumed 4.5 W active / 875 mW idle policy in
both arms.

## Evidence

- CHANGES.json, TESTS.json, source-before/ (exact copies from the v14 deploy).
- inputs/: models-v15.json, sparse24-v15(-control) campaign/rig/evidence,
  the adopted plan (MEASURED_DESKTOP_BASELINE_PLANS_V1-cold2560-v3.json).
- physical/desktop-parent-calibration-cold-2560-v2, -v3 (+COMMAND, logs),
  resolve-sparse24-v15, preflight-sparse24-v15, preflight-sparse24-v15c,
  sparse24-v15 (adaptive) and sparse24-v15c (control) attempt trees with
  ANALYSIS_SPARSE24.json each, sparse24-v15-AB_COMPARISON.json;
  sparse24-v16 / sparse24-v16c likewise with sparse24-v16-AB_COMPARISON.json,
  preflight-sparse24-v16(c), run-v16-both.sh/.log;
  analyze_sparse24.py (renewals, watcher CPU, power bins, per-request
  table), analyze_decisions.py (per-request assistance decisions),
  analyze_phases.py (energy by load/prefill/decode/idle phase).
