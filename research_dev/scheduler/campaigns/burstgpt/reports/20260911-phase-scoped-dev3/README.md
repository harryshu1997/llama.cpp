# Phase-scoped leases and receipt attribution

Status: 447 affected tests pass; physical calibration and the three-request
retest are blocked before launch by the rig's NVIDIA driver/library mismatch.
No deployment, physical inference, baseline run, long trace, commit, or push.

## Implemented scope

1. Receipt attribution comes from the runtime acquisition ledger, including
   completed overlapping peers. A non-phone route with no tracked phone work
   and no other acquired owner of charged server resources is `isolated`.
   Other receipts and campaign-wide windows remain `diagnostic`, with an
   exported attribution reason, ticket ID and overlapping ticket IDs.
   Server sample coverage remains mandatory. Assumed idle phone energy is
   retained and labelled assumed; measured phone mode still requires its samples.

2. Both preview builders use the same phase-demand constructor. Transition
   capacity is reserved at offset zero for its predicted preparation duration.
   Execution follows with one compute slot and the existing context-token slots.
   CPU parents retain the same capacity-wide load rule as other parents.
   Validated transition receipts retime the phase boundary atomically and release
   preparation tokens. Early and overrun loads are covered; execution renewal
   and completion coverage exclude released preparation tokens.
   The queue now uses sets of lane tokens and lane unions, not a last-row-wins
   resource dictionary. Memory projections retain the selected residency through
   execution instead of expiring at the shorter preparation boundary.

   Cold-plan compatibility keys are preserved. Acquisition does not borrow a
   cohort's execution leases for a request-owned pending preparation phase;
   READY execution still uses ordinary shared cohort admission. Native shared
   transition and execution protocols are unchanged.

3. Normal model configuration can register an overlay CPU parent using
   `endpoint_ids.cpu`, `backend_ids.cpu`, `cpu_runtime_parameters` and
   independent `cpu_evidence_ids`. The existing materializer supplies its CPU
   and CPU/NPU families. No single-point route-shape profile is synthesized.
   Load-energy priors remain SHADOW until receipt learning qualifies them.
   The old calibration script and its physical artifacts are preserved as
   historical evidence, not used as the new qualification workflow.

4. Protected-work horizons now come from active acquired tickets and current
   lease predictions. A missing or expired forecast for active work is unknown,
   not an idle system. The existing unknown-cost rejection remains conservative.

   Deliberate policy choice: add opt-in `protected_work_policy=energy-budgeted`
   for the requested work-conserving experiment; the default is still `strict`.
   Under energy-aware/energy-first selection it admits interference only when:

   `alternative route upper energy + protected extension upper energy
   < baseline route lower energy * (1 - configured margin)`.

   The existing cost already contains the extension, so it is not added twice.
   This gate applies to every alternative, requires measured system-cost
   evidence, and does not bypass qualification, memory, leases or latency checks.
   Calibration remains strict even when the optional policy is configured.

5. Idle-epoch reselection is deliberately deferred, as requested.

## Validation

The focused final run covers receipt attribution, phase leases, queue/controller
behavior, route learning and calibration, model configuration, physical adapter
contracts, cohorts, telemetry recovery, residency/rollback, and both replays.
Result: 447 passed in 103.676 s, including both repeated byte-identical replay
goldens. Compile and whitespace checks pass. The existing Python 3.13
multi-threaded fork deprecation warning remains. The exact command and counts
are in [TESTS.json](TESTS.json). It is not a complete-harness run.

The combined run first exposed a too-broad cohort-key guard. The existing
`test_cold_decode_plan_can_form_a_cohort` assertion was not changed. The guard
was moved to acquisition eligibility, keeping cold/hot compatibility keys equal
and retaining READY cohort operation; the 39-test cohort/phase follow-up passed.

New focused cases also cover completed-peer attribution, absent ledger identity,
phone activity, overrun retiming, atomic rollback after a retiming failure,
unknown protected horizons, strict versus budgeted selection, strict calibration,
512-slot calendar packing, and normal CPU-parent registration without a copied
GPU profile.

## Deliberate replay re-freeze

Both original hashes were independently reproduced from source before-images.
The final fixtures retain all 40 selected route IDs and arrival-phase helper
events. The preparation success/failure branches and session generations remain
unchanged.

| Case | Old SHA-256 | New SHA-256 |
|---|---|---|
| session_cow_gate_v3 | ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d | 5d52e8673fc60f974ca167964b3ad579e59abea28feb3d66357cf39c1ca7caf4 |
| sparse_locality24_v8 | 965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d | 241924463ac27ead0c2d7bcb5da214e85097049c91d5800848e0b29effd2b917 |

Each preparation branch changes nine operator-plan hash references because
execution resource slots no longer contain capacity-wide preparation claims.
In v8's late-gemma-tail segment, the same generation-1 geometry is exposed at
request 88135 rather than 88139. Its queued benefit is 19818086400 instead of
77856768000 uJ, reflecting the earlier arrived work. Phone event count stays 66;
events 46-50 carry the earlier proposal/evaluation payloads.
No arrival inputs, saved observations or future-demand hints were changed.

Full evidence: [decoded diff](REPLAY_DECODED_DIFF.json),
[old outputs](replay-before/), [new outputs](replay-candidates-v2/).
The first failed replay-generation attempt is retained in replay-candidates/.

## Physical blocker and next sequence

At 2026-09-11 14:27:18 UTC, `nvidia-smi` on `zhihao@172.20.74.85` failed:
`Failed to initialize NVML: Driver/library version mismatch`.

| Component | Observed version |
|---|---|
| Loaded kernel driver | 595.84 |
| NVML library | 595.91.07 |

The installed stack has no matching 595.84 library in the checked library/cache
locations. GDM and all existing GPU processes were left alone. No sudo, driver
replacement, reboot, worker restart or USB reset was attempted. One slow
read-only file search started by this task was stopped; no other process was
signalled. See [BLOCKER.json](BLOCKER.json).

A user/admin must restore a consistent NVIDIA stack before NVML memory admission
and board-power measurement can be trusted. After it clears:

1. Deploy this scheduler revision into a fresh directory, preserving native
   binaries, artifacts and prior results.
2. Register the independently proven CPU launch through models.json and run
   normal strict calibration. Inspect the observation store for four attributable
   samples per required shape bucket and QUALIFIED route/load energy evidence.
   Do not copy the former single-point profile.
3. Only then run unchanged burstgpt_dev3_long_v1 in energy-aware mode with the
   explicit energy-budgeted protected-work policy:
   Gemma 36 at 1 s / 292 tokens, Llama 37 at 61 s / 292, Qwen 50 at 91 s / 71.
4. Judge selection reasons and concurrency, not a matched savings claim. The
   catalog differs from the frozen desktop control.

New physical samples, new learned qualification, early CPU execution and energy
savings have NOT been demonstrated by this change. Capacity-wide cold loading
can still correctly wait for occupied resources; the optional budget gate does
not bypass those leases or other admission checks.

## Files changed

Paths below are relative to research_dev/scheduler. The complete before/after
hashes, including the new test file, are in [CHANGES.json](CHANGES.json).
Existing dirty-worktree changes are preserved. No production code outside this
tree was changed. This report and research_dev/talks.md are the progress record.

- `_internal/policy.py`
- `_internal/request_contracts/ticket.py`
- `_internal/route_generation/costing.py`
- `_internal/route_generation/feasibility.py`
- `_internal/runtime_controller.py`
- `_internal/runtime_controller_ops/leases.py`
- `_internal/runtime_controller_ops/queue.py`
- `_internal/runtime_decode_cohort.py`
- `_internal/runtime_queue.py`
- `_internal/runtime_residency_projection.py`
- `_internal/runtime_resources.py`
- `_unified/automated_requests_ops/observations.py`
- `_unified/automated_selection_ops/objectives.py`
- `_unified/automated_selection_ops/resources.py`
- `_unified/runtime_requests.py`
- `adapters/catalog_materialization.py`
- `adapters/energy.py`
- `adapters/heterogeneous_rig.py`
- `adapters/http_backend.py`
- `campaigns/burstgpt/arguments.py`
- `campaigns/burstgpt/catalog.py`
- `campaigns/burstgpt/launch.py`
- `campaigns/burstgpt/runner.py`
- `configuration/campaign.py`
- `configuration/models.py`
- `scheduler.py`
- `tests/data/replay/README.md`
- `tests/test_automated_runtime.py`
- `tests/test_decode_cohort.py`
- `tests/test_replay_determinism.py`
- `tests/test_work_conserving_start.py`
- `tests/test_phase_scoped_energy.py`
