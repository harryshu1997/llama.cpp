# Queue-aware READY dispatch and helper interference scope

Status: implementation complete; 184 focused tests pass, both replay goldens
unchanged. One pre-existing cold-cohort test wait remains documented below and
is not counted as passing. Preflight passed 101 checks with two advisory warnings.
Physical v4 passed 3/3 with Gemma phone work but no CPU/Gemma concurrency.
Final v6 passed 3/3 and demonstrated 21.410539 s of CPU/Gemma execution overlap.
It used more fleet energy than v4; useful energy-saving concurrency is not proven.
v3 and v5 failures are preserved. No longer trace, baseline, commit or push.
This continues the exact-parent startup result in
../20260911-concurrency-calibration/ without repeating calibration or baselines.

## Causes and corrections

1. Live selection compares request completion bounds, including queue time.
   Epoch publication instead compared service-only latency. In the saved v2
   Llama decision, READY CPU service was 25.319638 s versus GPU 16.532652 s,
   but the GPU was occupied. The 1.25 latency gate rejected CPU publication,
   despite live selection preferring it. Dispatch to an already-resident
   desktop parent now uses the same arrival-to-completion latency basis.
   Cold transitions and phone residency economics are unchanged.
2. A helper snapshot counted its own running desktop request as separate
   protected work. The rig now excludes only that exact request's execution
   tickets from peer protection. Other active tickets remain protected; an
   unscoped peer or unavailable power measurement remains unknown. Measured
   CPU/GPU power and historical overlap checks are not bypassed.
3. Generated qualified FFN USB links were treated as extra unmeasured hardware,
   despite mapping to the phone sessions' declared shared transport. Costing
   now resolves these aliases to that physical group, without modifying lease
   resources or adding zero-cost evidence. Unqualified, ambiguous or uncovered
   physical transport remains unknown.

## Source scope

Before-images are preserved in source-before/. Production changes:

- _unified/placement_epochs.py
- _unified/placement_epochs_ops/selection.py
- _internal/route_generation/costing.py
- adapters/heterogeneous_rig.py
- adapters/probes.py (physical v3 telemetry-ordering failure)

Focused regressions:

- tests/test_work_conserving_start.py
- tests/test_phase_scoped_energy.py
- tests/test_phone_allocation_snapshot.py
- tests/test_phone_power_probe.py

The READY CPU regression failed before the change and passes afterward. It
also verifies GPU remains selected when its queue is idle. Helper tests cover
own work, real peers, missing peer scope, switching, qualified transport aliases
and genuinely missing interference evidence. The snapshot integration test
preserves the resident endpoint's generation.

## Existing cold-cohort test blocker

The broader focused group encounters a pre-existing wait in
test_cold_phone_cohort_shares_transition_identity. Its second request waits
while the first is ACTIVE. Bounded inspection with both the before-image
publication function and the current function gives identical queue state:
second QUEUED, causal_ready=false, active_conflict=true, both load:helper-c
transitions PENDING. Injecting the first verified receipt alone does not release
the peer. This is not counted as a passing test, and no assertion, queue barrier
or production cohort behavior was weakened to proceed. It is outside the
independent-model three-request path and requires a separate phase/cohort audit.

An initial 35-second diagnostic timeout also interrupted the computationally
long sparse replay, not a hash mismatch. Both replay tests then passed normally
in 90.609 seconds without code or golden changes. The final replay check after
all corrections passed in 84.740 seconds. TESTS.json records counts, the two
unchanged hashes and the test interruption history. The full suite was not run.

## Physical plan

Only unchanged burstgpt_dev3_long_v1.json: Gemma 36 (1 s, 292 output tokens),
Llama 37 (61 s, 292), Qwen 50 (91 s, 71). Normal energy-aware selection,
explicit energy-budgeted protected-work policy, exact qualified CPU preload,
unchanged initial evidence, CUDA graphs enabled, binaries, models and shards.
Preparation and cleanup stay inside the paid boundary. No forced CPU or phone
route, baseline rerun or long trace.

Fresh campaign and rig inputs for v3 through v6 are in inputs/. Each attempt
has a separate artifact and phone directory under the same remote deployment.
Initial calibration inputs were reused, not updated with preceding attempts.
No preceding result, failed attempt or user change was overwritten.

## Preserved v3 failure and narrow telemetry correction

The new v3 attempt failed at paid time 39.085606 s while recording Gemma's
desktop load: `phone charging interval crosses a reboot`. Llama subsequently
completed on the now-free GPU; Qwen was cancelled. This does not test CPU/Gemma
concurrency or helper reuse. No successful RESULT was fabricated.

Power diagnostics show an ADB sample at host 9869491160274 ns with phone uptime
78286220000000 ns. Recovered HTTP returned uptime 78286150000000 ns, 70 ms
older, but the probe stamped it with its later fetch midpoint 9869774090154 ns.
The sampler consequently recorded PHONE_REBOOT. Following uptime samples
continued near 78286 s; the device had not rebooted to a new near-zero uptime.
Read-only post-run checks reported uptime 78447.63 s, the expected phone kernel,
normal ptp,adb and 299 MiB GPU use. The host and healthy services were untouched.

probe_phone_power already validated HTTP Date/capture freshness but discarded
the resulting capture timestamp. It now retains that validated timestamp, so
the sampler's existing ordering guard does not misclassify an older cached
sample as a newer reboot. The regression reproduces the exact recorded 70 ms
decrease and fails before the correction. All 32 power tests pass afterward,
including explicit charging and measured-energy rejection for actual reboot
boundaries. No reboot check, measurement coverage or health check was weakened.

The v3 source manifest and all failure artifacts are under physical/. v4 uses
the same workload/evidence/configuration with fresh output and phone directories.
The successful preflight preceded a diagnostic-only scope export adjustment and
the power timestamp fix; their focused checks and exact deployed hashes are
recorded separately. Unchanged qualification was not repeated.

## v4 physical result and remaining freshness races

v4 completed all three requests with valid semantic output and terminal proofs.
Gemma made 4,192 phone FFN calls: HTP0 2,184 and HTP1 2,008. Native calls and
acknowledged layer/column policies prove 273/292 assisted decode tokens (93.49%)
and 253.75 fraction-weighted token equivalents (86.90% of all decode tokens).
This is assistance over the resident 8- then 16-layer CPU FFN subset, not that
percentage of the entire model's FFN work. Its final selected fraction was 100%.
Llama and Qwen made no phone calls. Full paid duration was 266.853539 s and
nominal fleet energy 22,561.699142 J, including preparation and cleanup.
There is no matched savings claim or useful CPU/Gemma concurrency claim.

Llama's READY CPU candidate was rejected with MARGINAL_SYSTEM_COST_UNKNOWN,
not epoch nonconvergence. Its arrival snapshot reported
PROTECTED_POWER_SAMPLES_STALE. A focused reproduction shows that a background
sample newer than capture time causes this rejection even when earlier samples
fully cover a fresh interval. Protected-work costing now filters samples to
the capture boundary, without extending the 2.5-second freshness limit or
using a future sample. The regression checks the exact physical CPU/GPU sum
from only the covered interval and leaves genuine stale/missing/overlap checks.

v5 then aborted at Qwen's arrival: its captured snapshot was valid from
91.708782 s through 94.208782 s but had expired before admission. Gemma had
started late, at 66.791498 s, and was making real phone calls when the arrival
failed. No successful result is fabricated for this interrupted run. Its phone
power diagnostics contain no reboot boundary, confirming the earlier fix.

The rig's public snapshot() now retries a capture that expires during assembly,
up to three real observations. It never alters an old sample's validity or
memory values. Expired-capture timestamps are exported with a successful retry;
persistent expiry fails closed. The two focused regressions fail before and
pass after this correction, retaining the exact residency and generations.
The old snapshot body is now _snapshot_once; this is a narrow capture-lifecycle
repair, not a new scheduler or coordinator API. No runner policy was added.

## Final v6: concurrent execution proven, energy benefit not proven

All three requests completed with exact requested output lengths, accepted
semantic output, ticket-bound native execution proofs and a valid decision
journal. There were zero recorded fallback recoveries. The exact CPU parent
was verified READY at paid time 3.627282 s, generation 2, after 1.564608 s of
startup preparation. Llama reused it with a fresh request ticket and no load
transition. Startup and runtime preparation and cleanup remain inside payment.

| Request | Arrival-to-execution wait | Execution interval | Execution duration | Phone calls |
| --- | ---: | --- | ---: | ---: |
| Gemma 36, 292 tokens | 8.681222 s | 9.681222-145.121669 s | 135.440447 s | 1,840 |
| Llama 37, 292 tokens, READY CPU | 0.265470 s | 61.265470-82.676009 s | 21.410539 s | 0 |
| Qwen 50, 71 tokens | 104.438791 s | 195.438791-242.141842 s | 46.703051 s | 162 |

The entire Llama execution overlaps Gemma. Its arrival snapshot has fresh
measured protected-work power: CPU 49.362 W plus GPU 33.455 W, 82.817 W total,
from three samples ending at 60.777439 s, age 0.223929 s at capture. Normal
energy-aware selection admitted the READY CPU route. No route, session or
fraction was forced. Qwen still waits for GPU availability and preparation;
the result does not demonstrate concurrent cold loading.

Gemma execution is 11.784898 s (9.53%) longer than v4. Within v6, six eligible
100% windows on the same 16-layer mask average 417.409 ms/token before Llama;
six such windows during Llama average 491.240 ms/token. Their context lengths
and system concurrency differ, so these are direct timing observations, not
an isolated causal estimate of CPU-start slowdown or a matched baseline.

### Assistance and independent sessions

| Model | Tokens with any phone execution / all decode tokens | Fraction-weighted / all decode tokens | Per-session calls |
| --- | ---: | ---: | --- |
| Gemma | 128/292 = 43.84% | 103.25/292 = 35.36% | HTP0: 1,024; HTP1: 816 |
| Llama | 0/292 = 0% | 0% | None; CPU execution |
| Qwen | 27/71 = 38.03% | 27/71 = 38.03% | HTP2: 162 |

These percentages use all output tokens, not only eligible tokens. Gemma used
CPU-resident layers 0-7 then 0-15, Qwen layers 12-17. They are not percentages
of the whole model's FFN work. Fractions remain acknowledged work masks over
resident FFN shard files, with exact parent, artifact, operator plan and
generation proofs; no weights were changed by fraction controls.

| READY layout view | Published READY observation | Per-session generations | Changed session |
| --- | ---: | --- | --- |
| Gemma HTP0 | 22.802099 s | HTP0=1 | HTP0 initial load |
| Gemma HTP0/1 | 33.575133 s | HTP0=1, HTP1=1 | HTP1 initial load |
| Gemma HTP0/1/2 | 94.731665 s | HTP0=1, HTP1=1, HTP2=1 | HTP2 initial load |
| Gemma HTP0/1, Qwen HTP2 | 139.903836 s | HTP0=1, HTP1=1, HTP2=2 | HTP2 replacement |

The table uses the scheduler READY observation timestamps; the separate event
publication timestamps and phone-clock phases are preserved in
CONTINUITY_AND_TIMING.json. Native load-authorization-to-READY times are
9.580847 s, 10.151385 s, 13.865188 s, then 13.821396 s for those four loads.
The last consists mostly of a 12.422923 s weight read, 0.092209 s HTP init,
and 1.066019 s upload. Do not conflate these with queue or host preparation time.

HTP0 and HTP1 loaded exactly once and stayed at generation 1. Only HTP2
reloaded. All four loads have individual LOADING, VERIFIED and READY records;
none failed. Gemma's phone work stopped before the replacement began, so this
run does NOT prove retained-session calls during replacement or an interruption
bound. It also does not inject a reverse/rollback fault; those paths were
preserved and covered by the focused transaction tests, not re-proven here.

The unchanged wait analyzer reports four proposals, four READY, zero failures,
zero never-prepared layouts, 21.8 s with no layout, and first Qwen residency
48.9 s after arrival. The third initial Gemma load still waited 46.2 s before
preparation. Llama's analyzer classification of no phone residency is expected
for its CPU route. PHONE_WAIT_TIMELINE.json keeps the v4/v6 analyzer tables.

### Energy and comparison limits

| Assumed phone active power | Earlier sequential v4 fleet | Concurrent v6 fleet |
| --- | ---: | ---: |
| 3 W | 22.459 kJ | 25.291 kJ |
| 4.5 W | 22.562 kJ | 25.394 kJ |
| 6 W | 22.664 kJ | 25.497 kJ |

The same 0.875 W phone-idle treatment applies to both. CPU and GPU energy
are measured; phone energy is assumed. At 4.5 W, v6 contains 17.318913 kJ CPU,
7.607766 kJ GPU and 0.467366 kJ phone. Paid duration decreased from
266.853539 s to 250.013227 s, but energy increased 2.832346 kJ (12.55%).
Thus earlier Llama completion does not establish fleet-energy savings.

This is a diagnostic comparison of successive scheduler versions, NOT a fresh
matched A/B, isolated concurrency experiment, or comparison to default
llama.cpp. CUDA graphs remain enabled in the same modified binary; artifacts,
desktop-parent placements, request inputs and initial evidence were reused.
The source manifest differs after the telemetry fixes. The catalog also differs
from frozen desktop controls; those baselines were neither rerun nor relabelled.

### Findings not fixed in this turn

1. Gemma changed to 0% at token 152, paid time 79.749177 s, while Llama was
   running, and stayed there after Llama finished at 82.676009 s. Completed
   windows alone prove a 138-token unassisted tail through token 290. Native
   terminal counts and the acknowledged-policy audit cover the full request.
   Windows spanning the CPU start and finish still report batch 1 and remain
   measurement-eligible. Whole-fleet energy per token changes sharply with
   this extra CPU work. This suggests concurrency-context comparability and
   incumbent recovery need investigation; it does not prove a beneficial
   rejected fraction or justify overriding measured rejection.
2. The bounded 4,096-event helper export contains 4,085 LEASES_RENEWED events.
   Earlier Gemma attach/zero-reason events were evicted. Only eight late Qwen
   MEASURED_REJECTION decisions remain. Qwen records
   LEARNING_NO_PAIRED_IMPROVEMENT. We cannot reconstruct an exact reason for
   every Gemma zero interval from this export or claim complete lifecycle-log
   coverage. Detailed adaptive windows and terminal native proofs are retained.
3. The cold-cohort test wait documented above remains unresolved. No full-suite
   pass or independent-session concurrency acceptance is claimed from this run.

The next bounded investigation is comparability across external CPU co-runs,
incumbent recovery after that work ends, and lossless lifecycle-reason logging.
Do not launch a larger trace or claim an energy optimization from this result.

## Evidence and immutable identities

- Remote final artifact:
  /mnt/storage/s42-phase-concurrency-20260911-v1-nEEi6I/dev3-ready-parent-v6.
- Local exact copies: physical/dev3-ready-parent-v3 through v6, plus the
  successful preflight. Failed v3/v5 have no fabricated successful RESULT.
- Final RESULT SHA-256:
  095dae6abffa4651164d7db93a23f8f3e08935cce8c8bffb58d06116fde1f3a2.
- Deployed source HEAD: 99449bafade0b2c15de4410feda832035c2f2d83;
  dirty source manifest identity:
  c21163d332e34a76df181ca6f5d4dde03a658e85504908c0a543d34d17732f72.
  The local HEAD is 5f89a2d9d33be547a1bdef5fd0f504a279c50800;
  scoped source hashes, not a claim of identical repository HEADs, prove the
  deployed changes. No repository was committed to reconcile these identities.
- Runtime binary SHA-256:
  c1c1613611e34ac6c96c5552bf5f0f3ee07e01be4e22ff96b921f1c212e2b6cb.
- Capability catalog SHA-256:
  2128811ce9cf29bce558fb70842e2bdfe3102938888829f7a236ea8f5e846555.
- Exact Llama CPU parent SHA-256:
  c1ccad2708f47f9e9326f57f184d09b386b78c2a78fdc1a8c73c86e31584d895.
- Replay v3 unchanged:
  5d52e8673fc60f974ca167964b3ad579e59abea28feb3d66357cf39c1ca7caf4.
- Replay v8 unchanged:
  241924463ac27ead0c2d7bcb5da214e85097049c91d5800848e0b29effd2b917.

CHANGES.json contains the nine exact source/test before, after and deployed
hashes. ARTIFACTS.json records every attempt's artifact tree digest, key result,
journal, observations, source/command manifest and power-sample hashes.
PHYSICAL_AUDIT.json contains strict native-call/ack/proof-derived coverage and
validated journals. CONTINUITY_AND_TIMING.json records raw measured boundaries,
session generations and physical phase durations. TESTS.json lists 184 passing
focused tests and the excluded existing wait. No baseline, source evidence,
physical failure or unrelated dirty-worktree change was removed.

Post-run read-only checks found 299 MiB GPU used, no remaining host campaign or
inference process, the unchanged phone boot ID and normal ptp,adb after cleanup.
GDM and other users' processes were not stopped or changed.
