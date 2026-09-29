# Compatible context continuity: implementation and dev3 validation

Status: the narrow fixes and 213 focused tests pass. The one physical adaptive
dev3 run completed 3/3 with valid output and terminal proofs. **The performance
gate did not pass:** nominal fleet saving is 13.90% versus the frozen matched
desktop, and adaptive uses 21.34% more energy than fixed GGG. The 24-request
trace was not launched. No baseline was rerun. All artifacts are preserved.
The separate [gate decision](GATE_DECISION.json) records the performance FAIL;
the raw runner's execution PASS is not authorization for the larger trace.

## Narrow controller changes

- A physical session drain has its own pending slot. An optimization refresh
  cannot overwrite its mask or acknowledgement transaction. A cost-context
  change cancels obsolete optimization intent, not maintenance. Hard helper
  unavailability still cancels unsent assistance and preserves issued ACKs.
- A supported batch change can retain the currently accepted fraction for a
  bounded monitoring pair. The existing complete-pair reservation covers its
  measurements and controls; only that fraction is revalidated, not the entire
  sweep. Original measurements are exported as `PRIOR_ONLY`, never copied into
  new-context qualified records. Current paired evidence decides continuation.
  Rejection, insufficient opportunity and exhausted retry/token/energy budgets
  still stop assistance. Repeated changes cannot renew monitoring indefinitely.
- The unified scheduler authorizes continuity only against a READY helper,
  its exact desktop parent, and its supported batch limit. Existing shard,
  generation, operator, lease and physical safety checks remain authoritative.
- Window role is captured when the window opens and carried through the
  transition and receipt. A later refresh/state change cannot relabel already
  executing work or avoid its exploration accounting.

The model placement controller, residency selection, queue-demand calculation,
transition costs, hysteresis, model artifacts and native binaries are unchanged.
This is not a new helper controller or campaign scheduling policy.

## Software validation

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_dev/scheduler/tests:. python3 -m unittest test_sustained_assistance test_adaptive_decode test_adaptive_runtime test_late_helper_energy_policy test_session_cow_transaction test_replay_determinism -q
```

213 tests passed in 85.321 s. Ten added regressions cover supported batch
continuity, fresh paired evidence and rejection, exhausted and repeated-change
budgets, pending/applied maintenance masks, role accounting, READY/parent/batch
admission, refresh during monitoring and helper loss during monitoring.
Both replay cases are byte-identical across repeated runs; goldens unchanged:

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

Older refresh assertions now inspect the distinct optimization slot; their
atomicity, exact-ACK, retry and loss requirements are unchanged. No complete
scheduler harness was run.

## Physical experiment specification

Fresh artifact directory: `/mnt/storage/s42-context-continuity-20260910-v1/`.
Deployment: `/mnt/storage/s42-context-continuity-20260910-v1-deploy/`.
The previous retained-helper deployment and failed preflight remain unchanged.

The canonical campaign receives the frozen source-v7 catalog, initial evidence,
desktop parents, CUDA graph mode, model/worker/shard binaries, requests and paid
boundaries. Only fixed assignment is disabled; normal scheduler selection is
used. The workload is unchanged `burstgpt_dev3_long_v1.json`: Gemma 36 at 1 s
with 292 tokens, Llama 37 at 61 s with 292 tokens, Qwen 50 at 91 s with 71 tokens.
Preparation and cleanup remain paid. No fraction, route or session is forced.

The six frozen references are reused without rerunning or overwriting them.
Comparisons are historical-reference comparisons, not fresh matched A/B tests:
the shared controller changes could affect the fixed phone arms too. The existing
comparison checks must reject unrelated input, binary, artifact, desktop-parent,
initial-evidence or accounting mismatches.

The 24-request trace remains conditional on a clean small-gate result and its
comparison. A worse result is preserved and diagnosed, not hidden by truncating
requests, subtracting overlapping preparation, or weakening the baseline.

## Changed paths

Under `research_dev/scheduler/`:

- `_internal/adaptive_decode.py`, `_internal/adaptive_decode_state.py`,
  `_internal/adaptive_decode_contracts.py`
- `_internal/adaptive_decode_ops/helpers.py`, `windows.py`, `sequencing.py`,
  `reporting.py`, `completion.py`
- `_unified/adaptive_decode_control.py`
- `tests/test_sustained_assistance.py`, `tests/test_adaptive_decode.py`
- This report, its configuration/persistence-only `experiment.py`,
  `COMPARISON.json`, `GATE_DECISION.json`, and the new physical artifact copy

The pre-change scoped sources are retained locally at
`/tmp/s42-context-continuity.1kM7lq/` for a scoped rollback without touching
unrelated worktree changes. No commit or push.

## Physical result and stop decision

All comparisons below use the six final `references-source-v7` results only.
CPU-package and GPU-board energy is measured; phone active power is assumed.
The same 0.875 W idle-phone treatment is used for every arm. Preparation,
exploration, model swaps, idle time and cleanup are included in the paid span.
These are historical-reference comparisons, not fresh matched A/B results.

| Arm | Duration (s) | Fleet energy at 4.5 W (kJ) | Adaptive saving versus arm |
| --- | ---: | ---: | ---: |
| Clean upstream desktop | 371.20 | 31.866 | 14.35% |
| Matched modified desktop | 355.14 | 31.699 | 13.90% |
| Fixed GGG | 312.20 | 22.492 | -21.34% |
| Fixed GGQ | 315.72 | 25.337 | -7.72% |
| Fixed GQQ | 331.50 | 31.963 | 14.61% |
| Fixed QQQ | 384.84 | 33.325 | 18.10% |
| New adaptive | 391.70 | 27.292 | - |

| Phone active assumption | Adaptive fleet (kJ) | Saving vs matched desktop | Saving vs fixed GGG |
| --- | ---: | ---: | ---: |
| 3 W | 27.141 | 14.38% | -21.42% |
| 4.5 W | 27.292 | 13.90% | -21.34% |
| 6 W | 27.444 | 13.43% | -21.26% |

Nominal domain totals are CPU 16.548575 kJ, GPU 10.034731 kJ and phone
0.708860 kJ. No steady-state saving or break-even count is inferred by
subtracting overlapping preparation windows.

The adaptive candidate is not promoted over the existing references and physical
expansion stops here. The correctness fixes remain uncommitted: this experiment
is not a same-revision before/after measurement of those fixes, and its compatible
batch-monitor branch was never exercised. Reverting accounting or ACK safety to
improve a reported ratio would reintroduce known defects without identifying the
performance cause. The scoped pre-change backup remains available. Fixed GGG
remains the best tested reference; no fixed route was forced into this adaptive
run and no existing reference was overwritten.

The earlier 28.43% dev3 matched result in the sustained-assistance report used a
different baseline specification, desktop parents and source. It remains valid
for its own comparison, but is not an interchangeable before measurement for
this CUDA-reference experiment. The immediately preceding retained-helper
attempt never ran inference because its preflight lacked GPU headroom.

## Request coverage and remaining losses

| Request | Service (s) | Arrival-to-execution (s) | Assisted/all output tokens | Weighted/all tokens | Phone calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 36 | 142.233 | 73.751 | 254/292 (86.99%) | 81.34% | 5,720 |
| Llama 37 | 1.720 | 163.673 | 0/292 | 0% | 0 |
| Qwen 50 | 61.918 | 231.735 | 11/71 (15.49%) | 15.49% | 66 |

Service includes prefill and decode. Arrival-to-execution includes preparation
and queueing; it is not pure queue wait. Arrival-to-completion is
215.984/165.393/293.653 s, respectively. Scheduler decision time is 164.552 ms
median, 783.569 ms maximum over six decision/replan records.

Gemma uses 100% for 221 tokens and 75%/50%/25% for 11 tokens each, giving
237.5 fraction-weighted token equivalents. Its eligible denominator is 291
decode tokens after the first token: weighted eligible coverage is 81.62%.
Qwen has 11 equivalents over 70 eligible tokens, or 15.71%. Llama has no
supported helper in this run and is not labeled a measured phone rejection.

These fractions apply to CPU-resident FFN columns, not the entire model. Gemma
initially executes layers 0-23 with widths 15,360/11,520/7,680/3,840. Its final
47 assisted tokens use only layers 0-15 at width 15,360. Qwen's 11 assisted
tokens use layers 12-17 at width 17,408. Native calls and terminal per-layer
counts, not controller intent alone, determine coverage.

Two limits remain, neither hidden as a successful performance result:

- Gemma's initial layout is fully READY before decode, so there is no earlier
  qualified 16-layer winner in this run. Replacement changes the actual mask
  from 24 layers to 16. The exact refresh correctly retains identities for
  HTP0/HTP1, but maintenance-mask observations are non-qualifying and the
  24-layer winner cannot qualify a 16-layer route. At token 263, 29 output
  tokens and 36 request-wide probe tokens remain; new measurement admission
  fails. The zero control applies at token 266, leaving 26 output tokens at
  0%. The first 12 output tokens are initial baseline/control delay, for
  38 unassisted tokens total. This is not an unrelated-generation rejection.
- Qwen has one valid baseline window (tokens 9-13) and one valid 100% window
  (20-24). Their nominal means are 85.297/72.999 J per token and
  698.553/649.178 ms per token: positive means, but insufficient confidence.
  The candidate energy upper bound is 80.299 J/token; the baseline lower
  bound with the effective 1% margin requires at most 76.000 J/token.
  No incumbent is admitted. Further probing is unaffordable at token 24;
  the zero control applies at 27, leaving 44 tail tokens on desktop. There
  is no `LEARNING_NO_PAIRED_IMPROVEMENT` rejection in this sequence. The
  existing admission estimate also leaves only 1.998 s before its deadline
  at token 20, whereas that observed window takes 2.597 s. The completed
  observation is preserved; the cause is not reported as a negative mean.

The exported `CONTEXT_CHANGED` events are startup/terminal membership events
with active batch 1 on both sides, not a live batch 2-to-1 incumbent change.
There are zero `context_monitor_prior` events. The new compatible-batch monitor
is therefore covered by the focused regressions, not demonstrated by this
three-request physical workload. Both requests have direct exported reasons
for returning to 0%; no repeated attachment exception or budget reset occurs.
The maximum attempts for any recorded candidate/context is one.

## Independent-session evidence

| Operation | Physical load authorization to READY (s) | Scheduler READY (s from paid start) | Session generations |
| --- | ---: | ---: | --- |
| Initial Gemma HTP0 | 9.826 | 24.437 | 1 / empty / empty |
| Add Gemma HTP1 | 17.168 | 42.152 | 1 / 1 / empty |
| Add Gemma HTP2 | 10.455 | 53.367 | 1 / 1 / 1 |
| Selected HTP2 to Qwen | 17.786 | 201.121 | 1 / 1 / 2 |

All four session loads have independent LOADING, VERIFIED and READY events;
all four layout proposals prepare and reach READY. There are no failed
transitions. Only HTP2 reloads. Final resident bytes are 8,870,952,960, with
unchanged Gemma artifacts and generation 1 on HTP0/HTP1. All weight sources
are the deployed FFN shard paths, not complete-GGUF fallback.

Initial proposal is at 5.056 s and preparation at 5.843 s (0.787 s later).
Initial phone preparation overlaps Gemma's desktop transition (7.099-74.374 s).
The phone is fully published at 53.367 s; attachment is at 74.468 s, followed
by desktop execution at 74.751 s. This run does not exercise serving before
the third initial session is ready, since desktop preparation is slower.

The replacement drain is requested at 180.909051 s. The next safe boundary
and reduced-mask control intent occur at 181.642945 s (token 216).
Physical application and scheduler quiescence are acknowledged at
182.850355 s (token 219, native mask 65535, columns 15360).
Loading starts at 182.900591 s and READY is published at 201.120984 s.
Request-to-ACK is 1.941304 s; ACK-to-load is 0.050236 s. The maintenance
mask/ACK is not replaced by an optimization refresh.

| Retained session | Native calls before load | During scheduler load-to-READY | After READY | Total |
| --- | ---: | ---: | ---: | ---: |
| HTP0, generation 1 | 1,659 | 341 | 32 | 2,032 |
| HTP1, generation 1 | 1,656 | 340 | 36 | 2,032 |

These counts classify native `S41SERVERFFNUSB.d2h_completed_ns` against the
host's paid clock and scheduler loading interval. Separately, the phone's
sparsely logged cumulative call milestones rise 1,664 to 2,000 (HTP0) and
1,664 to 1,984 (HTP1) during its physical load-authorization-to-READY interval.
Those milestone values are cumulative counters, not values to sum. Native
terminal proofs give the exact final totals. Gemma makes another 1,656 calls
on HTP2 generation 1 before eviction; Qwen makes 66 on HTP2 generation 2
after readiness. Historical generations are preserved in the proofs.

This proves continued service, not independent HTP compute or a new bounded
inter-call-gap claim. It does not inject another reverse or rollback; those
paths are covered by the focused session-COW tests. Zero request recoveries,
fallback counters and USB resets are recorded. Normal terminal cleanup
restores USB to ADB at 5,000 Mbps. No campaign process remains; GDM and other
users' processes were not stopped, and VRAM returned to 3,179 MiB used.

## Why the end-to-end result is slower

| Desktop transition | Adaptive (s) | Frozen matched desktop (s) | Frozen GGG (s) |
| --- | ---: | ---: | ---: |
| Gemma | 67.275 | 37.014 | 41.638 |
| Llama | 6.924 | 2.799 | 8.708 |
| Qwen | 92.894 | 79.088 | 46.337 |

The measured desktop-transition durations total 167.094 s, 70.411 s more
than fixed GGG's 96.682 s. The total span is 79.505 s longer. These
transition intervals include runtime loading/verification and dispatch work;
they are not a causal attribution of all slowdown to phone interference.
Host page-cache/storage conditions were not controlled. Phone replacement
overlaps Gemma execution and cannot simply be added again to the paid total.
Gemma service is 142.233 s versus matched desktop 160.203 s and GGG 135.389 s;
Qwen service is 61.918 s versus 57.909 s and 61.303 s, respectively.

Before another physical attempt, the evidence points to bounded retained-mask
revalidation before the remaining opportunity is consumed, a complete-pair
budget based on observed control/window duration, and isolating the desktop
loading variability. None justifies copying 24-layer evidence to 16 layers,
overriding Qwen's uncertainty checks, increasing exploration without a bound,
or claiming that these fixes guarantee 25% fleet savings.

## Compatibility, graph proof and immutable artifacts

`COMPARISON.json` contains decoded compatibility differences for every arm.
Against modified-runtime references, all artifacts, binaries, desktop-parent
placements/qualifications, catalog, requests/arrivals, initial evidence,
controller configuration and accounting boundaries match. Differences are the
source manifest and the intended adaptive/fixed or desktop selection mode.
Clean upstream additionally has its own upstream binaries, catalog and parent
qualification identities. The unchanged strict matched validator rejects the
comparison with `A/B source or binary identity differs`; it is not bypassed.

Actual CUDA graph evidence: 78 captures, 53 instantiations, 78 executable
updates, 2,386 graph launches/activities, and 25 executable-reuse recaptures
under the existing metric. Graph mode is `default` for all three artifacts.
The new shared Python controller may affect fixed arms as well, so this run
is not an isolated proof of dynamic-placement superiority.

Remote: `/mnt/storage/s42-context-continuity-20260910-v1/`.
Local immutable copy: [physical](physical/). All 2,073 regular files were copied,
including results, commands, snapshots, journals, samples and CUDA evidence.
The raw RESULT hash was checked against the remote after copying.

| Artifact | SHA-256 |
| --- | --- |
| `physical/run/RESULT.json` | `c782a44bfe4901908f881a1c3bc55bf20667a71f4cb12a9f61be221084384641` |
| `COMPARISON.json` | `ac8692df7b30856aa0a6a41e2bec24f3f564abf4bafa7987bb70f211a99b51b0` |
| `physical/inputs/SOURCE_MANIFEST.json` | `3f4a55171380bdd573c8be45d33737fd58bb14c31dfc360dbe056b3f21b07907` |
| `physical/preflight/PHYSICAL_PREFLIGHT.json` | `74dc2cf2af394d36688b592bbdfe8b50b4351fc91186dd2316835f4eb0470f94` |
| `physical/run/SCHEDULER_DECISION_LOG.json` | `278ec092fea7e0d70caf0e3bafd6858c71c298f2db989c268bd3127ff6bcbf8b` |
| `physical/run/ADAPTIVE_DECODE_OBSERVATIONS.json` | `0255d6ae65f342f853fcfe31cb81c7750d80f089107004aa78f4452b385acdf2` |
| `physical/RUN_COMMAND.json` | `bee90531aeec41aa58b9adc338a54b3971bc487fb24c739bf5206ce49e6ec2d4` |
| Frozen `cuda_graph_v1/COMPARISON.json` (unchanged) | `83054fdc4179b4c64939a1ed07ce0891e833a3f642218464c47e62eaf27ba022` |

The journal's terminal chain hash is
`d4ed9160b53da02fe3263cd4abe6ace58f81e4f1068fcba881ad900740c84b3c`.
Full model, parent, shard, worker and host-library identities remain in RESULT,
DIRECT_PHONE_PREFLIGHT, the source manifest and the comparison decoded diff.

Reproduce the historical comparison into a new file (the reporter refuses to
overwrite an existing output):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 research_dev/scheduler/campaigns/burstgpt/reports/20260910-retained-helper-continuity/analyze.py research_dev/scheduler/campaigns/burstgpt/reports/20260910-context-continuity/physical/run/RESULT.json /tmp/context-continuity-comparison-new.json
```
