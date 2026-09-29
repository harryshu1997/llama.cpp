# Host power step: attribution instrumentation, Qwen paired windows, load-phase labels

Status: resolved (sections 1-7); the attached-helper horizon hole is fixed (section 8); the 24-request trace aborted on a 2,375-token request that exceeds the 2,048-token desktop context (section 9), renewal behaviour clean over 875 s. The 55 W host step was the runner's lease-renewal
thread spinning once a request's completed preparation-phase leases had
passed their predicted end (about 2,600 renewals per second, plus the
package-frequency lift it caused). Attributed with per-process, then
per-thread CPU sampling (v11, v12), fixed in the renewal coordinator and
validated in v13: Gemma's renewals 117,928 -> 79, package power through
Gemma's 24-layer decode 65-71 W -> 19-22 W, fleet 18.644 kJ at 4.5 W. Qwen's
rejection was diagnosed from its paired windows and left intact. Load
labels corrected; full-model overlap still assessed unsafe here.

## 1. Sampling and accounting artifacts, from the existing data

Checked on v7, v9 and v10 `resource-samples.jsonl` before adding anything:
sample intervals are steady (median 228-230 ms, minimum 217 ms, isolated
maxima 428-573 ms), the RAPL package counter never wraps (range 262,143 J,
zero negative deltas), and the RAPL sample time and wall time deltas agree to
the millisecond. In v10 the package rises from 5-25 W to about 70 W between
two consecutive 220 ms samples at 72.4 s while GPU board power stays at
33.8 W; in v9 from 15-30 W to 70-90 W at 73.4 s; in v7 from 55-65 W to
95-105 W at 73.6 s. The step is real host package power, not an accounting
or sampling artifact. Host `MemAvailable` stays near 28 GB throughout every
run because mapped model pages are reclaimable page cache; it cannot show a
working set, which is why per-process RSS was added below.

## 2. Sampler extension (`adapters/host_runtime.py`)

`HostMetricCallbacks` gains an optional fourth callback, `host_activity`,
wired by `default_host_metric_callbacks` to `linux_host_activity`: for every
sample it records the aggregate `/proc/stat` CPU jiffies (raw, all ten
fields), the clock tick rate, the mean/min/max `scaling_cur_freq` over all
CPUs, and per-process cumulative CPU ticks (utime + stime) and RSS from
`/proc/<pid>/stat`, with its own scan timestamp and duration. The sampler loop
keeps the previous ticks and emits only processes that used CPU since the
last sample or hold at least 256 MiB RSS, so a row carries a few processes,
not hundreds. All counters stay cumulative and raw; deltas are taken in
analysis next to the RAPL counters. A probe failure records a
`host_activity` probe error and still emits the row; rows without the
callback keep the old shape, so existing tests and consumers are unchanged.
On the desktop a scan costs about 10 ms (24 CPUs, some 300 processes).

Idle baseline, desktop with nothing of ours running except the sampler
(`physical/idle-host-activity-v1.jsonl`, 120 s, 503 rows): 3.3-3.9 W package,
0.9-1.0 % busy, about 1,250 MHz mean; the top consumers are the sampler
itself (0.05 cores), zerotier-one (0.03), the `irq/145-iwlwifi` thread (0.025)
and sshd (0.02). No periodic consumer appears at idle; `systemd` timers due
during runs are `sysstat-collect` every 10 minutes (seconds of work) only.

## 3. Qwen's rejection is a shard-size result, not a residency gap

Paired windows from the adaptive stores (`ADAPTIVE_DECODE_OBSERVATIONS`):

| Run | Baseline (0%) eligible windows | Assisted 100% / 6 layers / 17,408 columns, eligible window | Phone compute / RPC / USB per token |
| --- | --- | --- | --- |
| v6 | 626-638 ms, 83.4-83.6 J | 2,392 ms, 308 J | 83 / 96 / 12.5 ms |
| v8 | 606-614 ms, 76-84 J | 565 ms, 71.2 J (one 4-token window) | 54 / 59 / 4.4 ms |
| v10 | 647 ms, 82.1 J | 1,170 ms, 148.8 J | 109 / 134 / 25.5 ms |

Qwen3-14B has 40 layers; 6 phone FFN layers save on the order of 60 ms of
CPU per token, while the phone path costs 120-270 ms per token, dominated
by per-call RPC and transfer rather than phone compute. The controller's
`LEARNING_NO_PAIRED_IMPROVEMENT` / `MEASURED_REJECTION` are therefore
correct on the evidence in two of three runs, and the one favourable pair is
a single window. Selection is left unchanged. What would change the answer
is a larger Qwen shard per call (more layers per session, or Qwen across all
three sessions once Gemma has finished), which is a residency planning
question, not a measurement one.

## 4. Load diagnosis tightened

`analyze.py` no longer labels the `load_tensors` to warm-up interval as
"file read"; it is `tensor_load_s` with an explicit note that it includes
any host-to-device transfer and buffer allocation, and disk read is not
isolated. Resident working set now comes from per-process RSS in the new
samples (large-RSS trajectories per llama-server process) rather than from
mapped file size or `MemAvailable`. Full-model overlap of Qwen's load with
Gemma remains unsafe under current accounting; a bounded prefetch decision
needs the RSS evidence this run collects, and on a 30 GB host with Gemma's
working set resident it is expected to be refused.

## 5. Gate v11: process-level attribution

PASS 3/3, 267.5 s, 20.464 kJ at 4.5 W (Gemma 91.4% / 89.6% assisted,
24-layer layout again READY at 46 s; Llama GPU after Gemma; Qwen 12 assisted
tokens then rejection). The new samples attribute the step at process level:

| Bin | RAPL package | CPU busy | Mean freq | Gemma llama-server (pid 83902) | Runner python3 (pid 83719) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50-70 s | 14-17 W | 12.9-13.9 % | 1.7-2.0 GHz | 13.2-14.0 CPU-s per 5 s (2.7 cores) | 0.17-0.38 CPU-s per 5 s |
| 75-120 s | 65-71 W | 16.0-16.4 % | 3.1-3.4 GHz | 12.4-13.3 CPU-s per 5 s | 4.68-4.92 CPU-s per 5 s (about one core) |

At 1 s resolution the runner process goes from 0.02-0.04 CPU-s per second to
0.70-0.95 at 73 s, in the same second the package rises from 13.9 W to
71.6 W and the mean frequency from 1.1 GHz to 3.0-3.5 GHz. Gemma's server keeps
the same CPU time and its eligible 24-layer windows keep the same latency
(384 ms before, 365 ms after) while their whole-fleet energy doubles (19.9 ->
39.0 J per token). So the extra energy is one runner core plus the frequency
lift it causes for every other running core; the same pattern recurs at
210-260 s during Qwen's decode (runner at 4.4-4.9 CPU-s per 5 s). The
runner's process tree (pids 83713 launch.py -> 83719 runner -> 83734 sampler,
llama-servers 83739/83828/83902/85310/85408, adb children) rules out an
external consumer. Thread-level attribution needs the v12 samples below.

Phase energies, split at the step: Gemma decode before 73 s 4.80 kJ over
59.9 s (80.2 W), after 73 s 5.74 kJ over 55.2 s (104.0 W); Qwen load wait
1.97 kJ (77.4 s; this time the file was read cold: 60 s), Qwen decode 5.97 kJ.

## 6. Gate v12: thread-level attribution and the root cause

v12 (PASS 3/3, 307.9 s, 21.835 kJ) added per-thread CPU ticks for the runner
process with each thread's creation time relative to the runner's start,
because Linux truncates thread names to 15 bytes. The thread that consumes
the core is `runtime-lease-r...` (`runtime-lease-renewal-<request>`,
`RuntimeLeaseRenewalCoordinator`), created at +47.6 s when Gemma's helper
leases were attached, idle until 72.9 s, then 4.3-4.8 CPU-s per 5 s until
Gemma completed at 153 s; it is idle during v12's Qwen phase (Qwen held no
helper leases there) and busy during v11's (Qwen held leases). The helper
event export agrees: Gemma's LEASES_RENEWED count is 117,928 in v11 and
178,319 in v12, about 2,600 renewals per second from 73 s, each a scheduler
transaction and a journal event.

Root cause (`_internal/runtime_execution.py`, `RuntimeLeaseRenewalCoordinator._run`):
the wait before the next renewal is `min(ticket.final_reserved_until_us) -
guard - now`. That minimum ran over every lease of the ticket, including the
completed preparation-phase leases, which keep their predicted end in
`final_reserved_until_us` after `finish_prepare_phase` and are never extended
again. Gemma's preparation leases (context, coordinator, cuda0, desktop-cpu)
carried predicted end 73.353 s in v11; from 73.1 s (end minus the 250 ms
guard) the wait was zero, every iteration extended the live leases and the
helper leases (whose own horizon stayed a healthy 1.75-1.83 s ahead), and the
loop never slept. The onsets match each run's own preparation-phase end:
72.4 s (v10), 73.4 s (v9), 73.6 s (v7), 72.9 s (v12). v8's Llama co-run
masked it. The cost is one core plus the package-frequency lift it induces
(1.1-2.2 GHz to 3.0-3.5 GHz) for every other running core; v6's 4,085
renewals within its 4,096-event buffer were the same fault seen through a
smaller window.

Fix: the coordinator takes its base horizon over `ticket.live_leases` only
(falling back to all leases when none are live), and after any renewal that
still leaves the live horizon within the guard it waits one guard interval
and counts a stalled renewal (`diagnostics()`), so no future horizon defect
can turn into a hot loop. With the same stub scenario (a dead preparation
lease five seconds in the past, live lease a minute away) the before-image
coordinator renews 203,668 times in 0.5 s; the fixed one renews zero times.
Two regression tests cover the dead lease and a genuinely stalled horizon
paced at the guard.

## 7. Gate v13: unchanged three requests with the fix

PASS 3/3; five files deployed (CHANGES.json, remote hashes match); fresh
preflight PASS; weights pre-warmed as before, although both large models
were nonetheless read cold this time (Gemma tensor load 12.4 s, Qwen 106.1 s
against 2.1-2.5 s and 38-60 s earlier), so the 321.6 s duration is an
environmental outlier and is not compared.

| | v12 (before fix) | v13 (after fix) |
| --- | ---: | ---: |
| Gemma LEASES_RENEWED (renewals) | 178,319 | 79 |
| Runner CPU during Gemma decode after 73 s | 4.3-4.8 CPU-s per 5 s | 0.4-0.9 CPU-s per 5 s (helper watcher, 1 Hz snapshots) |
| Package power, Gemma 24-layer decode after 73 s | 65-71 W, 3.0-3.4 GHz | 19-22 W, 1.7-2.3 GHz |
| Gemma 24-layer eligible windows: before 70 s / after 78 s | 369 ms 19.2 J / 366 ms 38.8 J | 362 ms 20.6 J / 398 ms 22.9 J |
| Gemma decode phase energy from 73 s | 8.30 kJ over 80.3 s (103 W) | 3.99 kJ over 72.6 s (55 W) |
| Fleet CPU package / GPU board / phone | 12.785 / 8.395 / 0.655 kJ | 9.462 / 8.518 / 0.664 kJ |
| Fleet at 3 / 4.5 / 6 W | 21.675 / 21.835 / 21.994 kJ | 18.486 / 18.644 / 18.803 kJ |
| Gemma assisted / fraction-weighted | 91.4% / 89.6% (v11) | 94.9% / 87.4% |

Gemma's per-token energy no longer doubles at 73 s; the small latency
increase after the step time (362 -> 398 ms) is the absence of the frequency
lift the spinning thread had been buying, which the user's earlier caution
about per-token comparisons anticipated. Every other phase is within run-
to-run variance. Successive versions under uncontrolled cache state, not a
matched A/B; the fix removes an instrumentation-independent defect that
inflated every physical energy figure since the renewal coordinator was
introduced, including v4 and v6 from the earlier reports (v6's 4,085
renewals inside its 4,096-event buffer were the same fault).

What this changes in earlier conclusions: v4/v6/v8/v9/v10/v11/v12 all paid
the spin from their own preparation-phase end until Gemma completed (and
during Qwen's decode whenever Qwen held helper leases), so their fleet
totals are 2-4 kJ high relative to the fixed scheduler; comparisons among
them remain valid because the defect applied to all, but any absolute
figure should be re-measured before it is quoted.


## 8. The remaining renewal hole: a stalled attached-helper horizon

Review of the v13 fix reproduced a second hot loop the v13 coordinator
could still enter: 40,307 renewals in 0.161 s with healthy base leases
and a stale attached-helper horizon. v13 computed the wake-up from the
live base leases and the scheduler's helper horizon
(`_runtime_lease_renewal_horizon`, the attachment's
`lease_reserved_until_us`), but the post-renewal check refreshed only the
live base horizon. When the helper leases are not renewed by
`extend_runtime_request` (it renews them only while the request's
fraction is above zero), the helper horizon passes, the wake-up wait is
zero, every renewal "succeeds" on the base leases, and the loop never
sleeps. The longer trace, with its 0% monitoring periods, would have
brought the removed CPU waste straight back.

Fix (`_internal/runtime_execution.py`, deployed for v14 only):

- One `_effective_horizon_us()` for both the wake-up and the post-renewal
  check: the earliest end among live base leases and the scheduler's
  attached-helper horizon.
- After each renewal the complete horizon is refreshed. If it did not
  advance, `stalled_renewals` increments and the loop pauses for at most
  one guard (never past the remaining validity) before retrying; a
  horizon that advanced by less than a guard is retried only when that
  validity is about to end. Sleeping never touches a lease.
- Expiry safety: retries continue only while the authorization is still
  valid. Once `now >= horizon` after a renewal, `expired_horizons`
  increments and the new scheduler callback
  `_expire_request_helper_authorization(ticket, now_us)` runs: at
  fraction 0 it releases the idle helper window leases (the same path the
  baseline acknowledgement uses) and records
  `HELPER_AUTHORIZATION_EXPIRED / IDLE_HELPER_LEASES_RELEASED`; above 0
  it calls the adaptive controller's `helper_unavailable` (the existing
  lost-readiness path) and records `ASSISTANCE_DETACHING_AT_NEXT_BOUNDARY`.
  With nothing attached it returns False and the coordinator surfaces
  `RuntimeExecutionCoordinatorError` through `check()`; the same happens
  after 60 consecutive expiries. `diagnostics()` now reports
  `expired_horizons`, `renewals`, `stalled_renewals`.

Regressions (`tests/test_runtime_controller.py`, 25 -> 30 tests): stalled
helper horizon with healthy base leases (one renewal, one stall, one
expiry that releases the idle leases, then idle for the rest of the
test); stalled-then-recovering helper horizon (two renewals at least
90 ms apart, one stall, no expiry); expiry with no handler and with a
handler that cannot release (failure surfaced, handler called once);
late attachment (`wake()` leads to exactly one renewal at the late
horizon); the scheduler handler's three outcomes. The v13
stalled-live-horizon test now uses a valid-but-stuck horizon and asserts
that the expiry surfaces once the horizon passes. Focused modules 93
tests OK, both replay goldens unchanged, full suite 1,370 tests with only
the two pre-existing errors (TESTS.json).

## 9. Gate v14: the 24-request trace (sparse_locality24) - aborted at arrival 49

Inputs: the v13 campaign with only the trace path
(`burstgpt_sparse_locality24_v1.json`, sha256 78b7582e...), a new campaign id
and fresh phone session roots (inputs/sparse24-v14-*.json); same catalog
(2128811c...), same observation stores, Qwen's selection and placement policy
untouched. Fresh resolve and preflight PASS; Gemma and Llama pre-warmed
(already cached). Launched 23:23:11 desktop time.

What ran (physical/sparse24-v14-attempt1-FAILED/): 16 of the 24 arrivals
were submitted and 10 large requests completed in 875 s of trace time -
five Qwen requests on the resident GPU model (1-81 s), Gemma 88120 with a
cold GPU transition at 278 s and 88128 on the then-resident Gemma, then
Qwen 88125 (Qwen reloaded, acquired 619.5 s), 88126, 88127; 88129 had just
acquired the GPU at 868.96 s while 88130, Gemma 88131 and the three Llama
overlays (681-721 s) were queued. The overlays never started: Llama's
desktop-control route waits for the GPU like everyone else.

Renewal evidence on the long trace (the point of the run):

| | v13 (3 requests, 321.6 s) | v14 attempt 1 (16 arrivals, 875 s) |
| --- | ---: | ---: |
| LEASES_RENEWED total | 89 | 236 (6 requests, 2 to 76 each) |
| Renewals per second of attachment | 0.28-0.7 | 0.41-0.70 |
| HELPER_AUTHORIZATION_EXPIRED | 0 | 0 |
| runtime-lease-renewal thread CPU, whole run | 0.25 s (8 threads) | 0.73 s (16 threads) |
| 5 s bins with a renewal thread in the runner's top 3 | 0 / 67 | 1 / 178 (0.01 s) |

No stall, no expiry and no spin over fourteen minutes of arrivals, helper
attachments and detachments; the fix from section 8 changed nothing else
visible in the helper event stream (kinds and counts are the normal
lifecycle set).

Host power: package mean 48.8 W over the 177 valid 5 s bins (one bin at 85 s
is a counter wrap and is excluded), peaks 90-92 W during GPU decode of the
hot model, 8-14 W idle minutes. A second host consumer is now visible
because the renewal spin is gone: the runner's per-request helper watcher
threads (`request-helper-<request>`, `adapters/runtime.py`, a 50 ms poll)
each burn 1.3-2.0 CPU-s per 5 s while their request is queued; with three
Qwen requests queued behind the first one, 3.1-4.4 CPU-s per 5 s (0.6-0.9
cores) from 40 s to 110 s, and 0.3-0.5 CPU-s per 5 s for the single queued
88129 between 845 and 865 s. Not changed in this step; it is the next host
CPU item and it scales with queue depth, so it matters more on long traces.

Adaptive sessions completed before the abort: Gemma 88120 95.9% and 88128
94.3% assisted at 100% fraction (24 layers); Qwen 88118/88119/88122/88123/
88125 finished at 0% (0-2.8% assisted, the measured rejection), while Qwen
88121, 88126 and 88127 ended at 100% fraction with 72%, 79% and 94% of
tokens assisted - the unchanged policy's own measured decisions, recorded
here, not evaluated.

The abort. Arrival 49 (`burstgpt-v2:88132`, 876 s) is a Gemma request of
1,884 prompt + 491 output tokens = 2,375 tokens. Every Gemma coordinator in
the catalog preallocates request memory in 512-token quanta on a context
resource of capacity 4 (a 2,048-token server context): the request needs
ceil(2375/512) = 5 slots, `_validate_internal_capacity` raises "resource
lease exceeds resource capacity", every Gemma-target route (desktop, CPU,
phone-assisted) is rejected RESOURCE_CALENDAR_INFEASIBLE, and the mandatory
desktop control cannot be generated (`_desktop_control`,
DesktopControlUnavailableError -> UnifiedScheduleError in
`submit_automated_request`). The runner treats a submission error as fatal:
FAILURE.json, 88129/88130/88131 and the overlays CANCELLED, no RESULT.json.
It is the only request in the trace that exceeds a context (table below);
88131 with 320 tokens had queued normally 25 s earlier. Deterministic, and
independent of the renewal change (no expiry event exists, the last helper
event is 88129's normal INSUFFICIENT_OPPORTUNITY at 875.2 s).

| index | request | target | tokens | slots (quantum) | capacity |
| --- | --- | --- | ---: | ---: | ---: |
| 49 | 88132 | Gemma (cold desktop) | 2,375 | 5 (512) | 4 |
| 44 | 88128 | Gemma | 1,492 | 3 (512) | 4 |
| 36 | 88120 | Gemma | 1,207 | 3 (512) | 4 |
| 45 | 88129 | Qwen (hot desktop) | 933 | 1 (1024) | 2 |
| others | | | <= 822 | 1 | 2 or 4 |

Remedies, none applied here because each changes an input or a policy:
(a) a 23-request variant without index 49 (changes the trace);
(b) reject an infeasible request at submission with a terminal REJECTED
decision and continue the campaign (a scheduler/runner behaviour change
with decision-log and golden consequences; the replan path already treats
this error as retryable in desktop-baseline mode);
(c) a catalog with a 4,096-token desktop context (requires recalibration,
the energy and latency profiles are bound to the catalog). Recommended:
(b), because a single oversize request will abort any real trace.


## Evidence

- CHANGES.json, TESTS.json, source-before/.
- physical/idle-host-activity-v1.jsonl: the idle baseline.
- physical/dev3-ready-parent-v11, -v12, -v13 with their preflights and
  inputs/: exact copies of the remote artifact trees; ANALYSIS.json from
  analyze.py includes the per-bin process and thread attribution.
- physical/sparse24-v14-attempt1-FAILED/ (+ preflight-sparse24-v14, launch
  logs, inputs/sparse24-v14-*.json): the aborted 24-request attempt with
  FAILURE.json and the FAILURE_* captures; ANALYSIS_ATTEMPT1.json from the
  renewal/host analysis; analyze_sparse24.py is the many-request analyzer.
