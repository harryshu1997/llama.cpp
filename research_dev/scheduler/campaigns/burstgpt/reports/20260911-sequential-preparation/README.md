# Sequential preparation and READY-endpoint concurrency

Status: software changes validated; physical calibration and the unchanged
three-request gate remain blocked by NVIDIA. No deployment or physical run.

## Corrections

The physical adapter executes `plan.transitions` sequentially. Previously, both
preview builders reserved every load at offset zero, renewal extended all loads
together, and receipt publication waited until the entire sequence finished.

- Both previews now use cumulative offsets in the adapter's transition order.
- Renewal extends the active unfinished preparation and shifts dependent loads
  and execution together. Full execution duration and lease identities survive.
- Each verified physical receipt is published before the next adapter load.
  A successful receipt prefix keeps the request PENDING and non-executable.
  Previously verified receipts cannot be replaced, omitted or rebound.
- Each preparation reservation ends at its own verified completion, including
  early completion. Only its tokens leave the dispatch queue; memory remains
  reserved. Remaining preparation and execution retain their capacity claims.
- Queue conflicts use the proposed phase windows, not every prepare resource
  stretched through the request's final horizon. Projection recognizes individual
  completed transitions and follows the retimed phase starts.
- The existing scheduler transaction restores the ticket, queue, calendar and
  memory ledger on failure. No new transaction/controller subsystem was added.

Protected-work extension power previously included GPU power only. It now uses
RAPL CPU package plus NVML GPU board energy over a common, fresh interval after
the protected executions started. The current acquisition ledger rejects samples
containing unrelated server work, including historical overlap. This prevents
the alternative's own work from entering the protected-power estimate again.
The existing marginal-cost calculation charges extension energy once; the
alternative's execution/tail energy is not added to that power observation.

The stronger CPU/GPU concurrency case exposed one additional blocker in rough
candidate ranking: it used the final reservation on any lane as the resource's
busy horizon. This could prune the READY CPU parent despite three spare CPU
slots. Rough ranking now uses the existing next-free-slot lower bound. Exact
phase reservation and cold-load capacity checks still run afterward; no route
is forced and no search-budget or qualification limit was increased.

Missing, stale, wrong-package, mixed-phase, overlapping or horizon-less evidence
remains unknown (`measured=false`), with the exact reason in the observation ID.
Valid observations export CPU/GPU power, sample start/end, age and validity in
snapshot cost features. No configured power is relabelled as measured residency
or measured protected work. Assumed phone power and receipt attribution are
unchanged. `strict` remains the default; `energy-budgeted` remains explicit and
calibration remains strict.

## What the concurrency check establishes

The new synthetic integration case uses the real snapshot builder and normal
model registration. An independently qualified CPU parent is observed READY
while an acquired large request holds CPU/GPU execution capacity. Normal
energy-aware selection with explicit energy-budgeted protected-work admission
selects the CPU route, acquires fresh execution leases and requires no load.
The large request remains ACQUIRED. The CPU request takes one CPU slot and no
GPU lease. The source catalog still requires all four CPU slots for a cold load.

This proves software admission/ownership behavior only. It is not physical
Llama/Gemma overlap, a measured slowdown, or new qualification. A READY endpoint
must be the exact registered/calibrated parent, not another CPU endpoint with
a similar model or a substituted placement hash.

| Check | Result |
|---|---|
| Final affected software set | 457 passed across four batches |
| New focused regressions | 7 |
| Replay oracle | Both unchanged; repeated outputs byte-identical, 40 requests |
| Compile and changed-file whitespace | Pass |
| Physical execution/load qualification | Not run |
| Three-request physical overlap, waits, phone help, slowdown | Not measured |

Coverage includes two sequential loads sharing a capacity-wide resource, early
completion, load overruns, per-receipt adapter publication, post-release
transaction rollback, immutable receipt prefixes, CPU/GPU extension accounting,
unknown/contaminated power, READY CPU concurrency, and existing attribution,
helper, session replacement, rollback and terminal-proof tests. It is an affected
set, not a full-harness run. The existing Python 3.13 fork warning remains.
The initial 454-test run passed before the stronger concurrency case exposed
the ranking bug. Following that fix, the final affected checks passed in
batches of 296, 29, 130 and 2 tests, including search and both replay tests.

Goldens were not modified:

- v3: `5d52e8673fc60f974ca167964b3ad579e59abea28feb3d66357cf39c1ca7caf4`
- v8: `241924463ac27ead0c2d7bcb5da214e85097049c91d5800848e0b29effd2b917`

## Physical blocker and next bounded work

Read-only confirmation at 2026-09-11 15:05:33 UTC:

- `nvidia-smi`: `Failed to initialize NVML: Driver/library version mismatch`.
- Loaded kernel: 595.84; NVML library: 595.91.07.
- No GDM stop, driver replacement, reboot, process kill or USB reset attempted.

After the user/admin restores a consistent stack:

1. Use the canonical calibration campaign and inspect the observation store for
   actual QUALIFIED CPU execution and GPU/CPU loading evidence separately, with
   the required attributable samples in each shape bucket. No copied profiles.
2. Keep the scheduler-managed calibrated CPU endpoint READY for the concurrency
   check. Record its initial residency, artifact, placement, endpoint, generation,
   allocation and binary identity before the paid workload. Do not substitute
   the pre-existing fallback endpoint or pretend cold loading is already proven.
3. Run only unchanged `burstgpt_dev3_long_v1.json`, energy-aware, with the explicit
   protected-work policy recorded: Gemma 36 at 1 s / 292 output tokens, Llama 37
   at 61 s / 292, Qwen 50 at 91 s / 71. No route/fraction/session overrides.
4. Report actual overlap, wait reasons, phone assistance, and protected slowdown.
   Conservative evidence or load-capacity rejection remains a valid outcome.
   Keep concurrent cold loading as a separate unvalidated capability.

No new throughput or energy claim is made. The current catalog differs from
the frozen desktop control, so the eventual run is not a matched savings A/B.
No long trace, baseline rerun, commit, push or PR.

## Exact files changed

Production (all under `research_dev/scheduler`):

- `_internal/runtime_resources.py`
- `_internal/route_generation/costing.py` (rough availability lower bound only)
- `_internal/runtime_controller_ops/leases.py`
- `_internal/runtime_controller_ops/queue.py`
- `_internal/runtime_controller.py`
- `_internal/runtime_queue.py`
- `_internal/request_contracts/ticket.py`
- `_internal/runtime_residency_projection.py`
- `_unified/automated_requests_ops/observations.py`
- `_unified/runtime_requests.py`
- `adapters/runtime.py`
- `adapters/heterogeneous_rig.py`

Tests: `tests/test_phase_scoped_energy.py`, `tests/test_work_conserving_start.py`.
Documentation: this report and `research_dev/talks.md`.

Before/after hashes against the turn-start dirty tree are in
[CHANGES.json](CHANGES.json); preserved before-images are in `source-before/`.
Validation details: [TESTS.json](TESTS.json). Physical read-only evidence:
[BLOCKER.json](BLOCKER.json). Existing user changes and physical artifacts were
not removed or overwritten.
