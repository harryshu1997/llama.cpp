# S42 Physical Optimization Loop Contract

## Fixed question

For the same real BurstGPT trace and completed output work, does one declared
scheduler change reduce accounted fleet energy without increasing total trace
latency or reducing service quality?

Only one scheduler, route, batching, placement, or device change is evaluated
per iteration. Kernel tuning needed to realize that declared change is part of
the same iteration and must be listed.

## Frozen workload

- Trace: `REQUESTS_SEMANTIC_LONG.jsonl`
- SHA-256: `ccde6e3e53dee4547e4eb80f9f090032afb04b1d3fec3fd07f0f961bed60cf8f`
- Requests: 74
- Input tokens: 18,211
- Requested output tokens: 2,175
- Last scheduled arrival: 57.7 seconds
- SLO: 30 seconds per request
- Hot role: 57 requests on the RTX 4060 Ti route
- Cold role: 17 requests on the CPU or CPU-plus-phone route

Changing the trace, token caps, model artifacts, output budgets, arrival
times, or SLO creates a new campaign and cannot be compared as another loop
iteration.

## Successor source-length campaign

I3 is a separately identified successor campaign rather than a relabeled I0
iteration. It uses `REQUESTS_SEMANTIC_SOURCE.jsonl`, SHA-256
`b20a9ba66ee3558d835a0e19ed3cfa4c31a4a9e8b4f9c085b29a14f80250a0ff`,
with 74 requests, 33,843 input tokens, and 11,605 output tokens. Its own strict
aggregate validator applies the same repetition, synchronized-energy,
completed-work, SLO, quality, placement, cleanup, and overlap principles. I0
records remain immutable and are not filled with I3 values.

## Primary physical table

Every iteration must report these real-device values, averaged across valid
repetitions:

| metric | control | treatment | change |
| --- | ---: | ---: | ---: |
| total trace makespan, seconds | required | required | required |
| accounted fleet energy, joules | required | required | required |
| accounted energy per output token, joules | required | required | required |
| completed requests / output tokens | required | required | required |
| SLO requests met | required | required | required |

Predictions and simulated fleet scaling must appear in a separate diagnostic
table. They never fill a missing physical cell.

## Phone-work table

Every treatment reports all denominators below:

- requests that invoked phones / all trace requests;
- requests that invoked phones / eligible requests;
- paid phone calls, excluding warmup;
- phone MACs / eligible island MACs;
- phone MACs / all-model MACs, or `UNCLAIMED` if the full graph was not
  counted;
- paid bytes host-to-phone and phone-to-host, excluding warmup;
- resident phone weight bytes; and
- phone compute, RPC, host branch, join wait, and end-to-end island latency.

Every split route must also report arithmetic-mean host-branch, phone-RPC,
join-wait, and island times from sum/count counters. Percentiles remain useful
diagnostics but cannot fill these mean fields. Define:

```text
branch_imbalance = abs(T_phone - T_host) / max(T_phone, T_host)
exposed_join_wait = T_join_wait / T_island
phone_work_hidden = 1 - T_join_wait / T_phone
```

The executor launches ready host and phone branches asynchronously. A split is
balanced when mean exposed join wait is at most 5% of mean island time. The
phone should normally finish just before the host's conservative completion
bound. If either branch finishes early, its resource is released immediately
for other ready work instead of being reserved until the join.

The current reporter derives means from `overlap_sample_count` plus
`host_branch_sum_ns`, `rpc_sum_ns`, `phone_compute_sum_ns`, `wait_sum_ns`, and
`overlap_sum_ns` in the executor's FFN summary. It does not accept
caller-supplied means.

The loop does not require offloading a minimum fraction. It searches for the
energy-minimizing fraction under latency and quality constraints.

## Energy boundary

The primary V1 metric is accounted device energy over one common paid trace
interval:

```text
E_accounted = sum(device_average_power_w * paid_trace_duration_s)
```

Equivalently, a device may sum active-power times active-duration and
idle-power times idle-or-wait-duration over that same interval. Waiting is
never free. The control arm keeps the same phones connected and charges their
idle energy for the full control interval.

The required component set is the server CPU package, server GPU board, and
every connected campaign phone as a whole device. Each receipt lists exact
device identities, measurement sources, average power, duration, samples, and
joules. Control and treatment must contain the same component set. Omitted
platform terms such as DRAM, motherboard, storage, or USB-controller loss are
listed explicitly, so this result is called `accounted device energy`, not
whole-system energy.

A gross AC meter covering the desktop and USB-powered phones is the preferred
cross-check and stronger publication boundary. Do not add a phone rail value
to gross desktop AC when that AC measurement already supplies the phone. Do
not combine a gross receipt and component receipts into one total.

Each receipt binds the result SHA-256, boundary ID, paid start/end, component
set, sample counts, joules, and validity state. Component duration must match
the paid interval and component joules must equal average watts times duration
within the reporter's fixed 0.5% arithmetic-reconciliation tolerance. Meter
accuracy remains a separate uncertainty term when fitting route bounds.
Missing or invalid receipts make the iteration incomplete.

The receipt schema is `s42-accounted-device-energy-receipt-v1`. Each component
uses `kind`, `device_id`, `source`, `sample_count`, `average_power_w`,
`duration_s`, and `energy_j`; the top level includes the reconciled `energy_j`
sum and an explicit `excluded_components` list.

## Repetition and ordering

- Use at least three alternating pairs, with order balanced across the
  campaign when practical.
- Recapture device, model, trace, runtime, placement, clock-policy, and energy
  identities before each paid run.
- Report arithmetic averages for the headline table plus the full run range.
- Do not replace a paid failure or tune after seeing a paid result. Start a
  successor iteration with a new ID and declared change.

## Pass target

An iteration passes only when all conditions hold:

- 74 / 74 requests and 2,175 / 2,175 output tokens complete in every run;
- treatment average accounted fleet energy is at most 90% of control;
- treatment average makespan is at most 100% of control;
- treatment mean exposed join wait is at most 5% of split-island time;
- treatment SLO count is not lower than control;
- the declared exact, bounded-numeric, or approximate quality gate passes;
- no model swap, placement violation, worker failure, USB reset, thermal
  invalidation, or leaked process occurs; and
- phone-work and transfer accounting reconcile with executor counters.

Results saving 5-10% energy are useful scheduler-admission points but do not
meet this loop's paper-strength 10% target. Results with lower power but higher
joules fail.

## Iteration handoff format

Each report ends with exactly four lines:

```text
VERDICT: PASS | FAIL | INCOMPLETE
BEST_REAL_RESULT: latency_s, accounted_energy_j, energy_change_pct, phone_work_pct
BLOCKER: none or one concrete missing physical requirement
NEXT_ONE_CHANGE: one bounded change for the next real-device iteration
```
