# S42 per-resource leases and queue prediction V1

## Contract

A route may now describe disjoint resource phases instead of holding every
resource for its complete service time. Each phase declares a stable lease ID,
resource, slot count, start offset, nominal duration, and optional duration
upper-bound addition.

```json
{
  "resource_slots": {
    "cpu-cold": 1,
    "op15-htp": 1,
    "op15-usb": 1
  },
  "resource_leases": [
    {
      "lease_id": "host-branch",
      "resource_id": "cpu-cold",
      "slots": 1,
      "start_offset_us": 0,
      "duration_us": 200
    },
    {
      "lease_id": "upload",
      "resource_id": "op15-usb",
      "slots": 1,
      "start_offset_us": 0,
      "duration_us": 100
    },
    {
      "lease_id": "phone-compute",
      "resource_id": "op15-htp",
      "slots": 1,
      "start_offset_us": 100,
      "duration_us": 300,
      "duration_ucb_add_us": 20
    },
    {
      "lease_id": "download",
      "resource_id": "op15-usb",
      "slots": 1,
      "start_offset_us": 400,
      "duration_us": 100
    }
  ]
}
```

Offsets and durations may also use the existing affine request-feature
expressions. A legacy route without `resource_leases` receives one implicit
full-route lease per declared resource.

## Queue solver

`ResourceTimeline` keeps an interval calendar for every capacity lane. Preview
is transactional and does not mutate the calendar. It finds a common route
start for which all relative lease intervals fit, assigns lane combinations,
and repeats across resources until the common start is stable. Commit verifies
that the calendar did not change and atomically installs every interval.

The calendar reserves `duration + duration_ucb_add_us`; it reports the nominal
completion separately. This prevents nominal-duration overlap from silently
consuming latency uncertainty. Multiple disjoint phases may reuse one resource,
and a later request may fill a gap after an earlier phase is released.

Every decision reports:

- route queue time;
- the resources that pushed the predicted start;
- per-resource blocking delay, which is diagnostic and not additive across
  resources;
- lease token, lane, start, nominal completion, and reserved completion; and
- replay aggregates for queue delay and predicted/reserved lane time.

An executor may call `release(token, actual_end_us)` when a phase completes
before its reserved upper bound. The next preview immediately sees the freed
interval. Completion after the reserved bound fails closed; Stage 5 owns late
stall, failure, cancellation, and atomic fallback handling.

`resource_snapshot(at_us)` reports readiness, capacity, slots free at that
instant, active and queued owners, the active interval tail, the complete
reservation tail, and the earliest one-microsecond gap. `next_available_us()`
provides the same gap query for a requested duration and slot count. These are
forecast diagnostics; a route preview still solves all of its real phase
durations transactionally before commit.

## Validation

The test suite covers legacy whole-route behavior, disjoint CPU/USB/phone
phases, iterative cross-resource queueing, upper-bound reservations, early
completion, independent capacity lanes, serialization, missing resources,
over-capacity overlap, leases that exceed service time, and live resource
forecasts before and after release or readiness changes.
