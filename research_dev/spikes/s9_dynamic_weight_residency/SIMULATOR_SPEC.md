# Simulator Spec (S9-V0)

Status: frozen V0 spec for the deterministic offline research simulator
`sim/residency_sim.py`. It is an ISOLATED research tool. It is NOT integrated into
llama-server, and it builds no phone daemon, downloader service, kernel, or
production scheduler. Input is `sim_config` (validated), output is a `sim_run_manifest`
(validated) plus a byte-identical replay hash. Synthetic fixtures only; no
real-trace result until S8 Gate A is independently confirmed passing. Section 7
defines the required versioned V1 correction after the dual-phone USB measurement;
V0 results are retained as mechanics regression evidence only.

Version scope: sections 1-7 describe historical V0. The later V0-R/R1 sections
describe changes to the reused simulator path, but passing those suites does not
certify a production scheduler or its dispatch validator. The v3 DRAINING-lease
case is fixed; the remaining static contract blockers are recorded in `RESULTS.md`
and `V0R1_REPAIR.md`.

## 1. Determinism

- Time is INTEGER MICROSECONDS. No floats anywhere in the model or output
  (durations are integer-divided `ceil`/`floor`; slowdowns are per-mille integers).
- The event queue is a min-heap keyed by a total order `(t_us, priority_rank, seq)`
  where `seq` is a monotone global counter assigned at enqueue. Ties never depend on
  insertion-hash or wall clock, so two runs schedule identically.
- No `Date.now()`, no wall clock, no `random` except a seeded integer LCG whose seed
  is `sim_config.seed`; the LCG is used only where the config explicitly asks for a
  deterministic tie-break, never for durations.
- Output (`goodput_results`) is serialized as canonical JSON (sorted keys, no
  spaces, ASCII, integers only) and `deterministic_replay_sha256` is the SHA-256 of
  those bytes. Re-running with the same config + seed reproduces it exactly
  (required test: byte-identical deterministic replay).

## 2. Modeled entities

- SERVER: `gpu_lane_slots` in-order GPU lanes, `hbm_credit_bytes` capacity, per-class
  GPU-us and HBM-bytes cost maps. Requests routed to the server queue for GPU time +
  HBM credits; freed GPU-us / HBM-bytes are the relief signal.
- USB SHARED CONTROLLER (V0 LIMITATION): one aggregate
  `usb_shared_controller_bytes_per_s` ceiling is assumed for all phone links, and
  the implementation statically divides it by configured phone count. The measured
  OP12/OP15 topology has separate root buses, so V0 is not a valid fleet model.
- PER-PHONE LINK (V0 LIMITATION): one symmetric `goodput_bytes_per_s` swept axis +
  `rtt_us`; direction and evidence provenance are absent.
- PHONE: UFS write rate, verify rate, materialize rate, per-backend prepare rates,
  warmup us, bounded HTP lane + bounded GPU lane (`*_lane_slots`), LPDDR budget,
  thermal eligibility. Runs the RESIDENCY state machine per weight set and the
  bounded lanes for execution.
- CANONICAL vs BACKEND-DERIVED RAM: tracked separately in the LPDDR ledger
  (WEIGHT_RESIDENCY_CONTRACT section 6). Canonical F16 is HTP<->GPU shareable; a GPU
  xmem-prepacked image adds `derived_bytes_gpu` that is NOT shareable.
- ACTIVATION BUFFERS + STATE LEASES: per-request activation bytes + (sticky) state
  bytes reserved in the ledger; state leases pin residency.
- INTERFERENCE: HTP<->GPU and transfer<->compute slowdown applied ONLY when
  `interference.measured==true`; otherwise the per-mille factor is pinned at 1000
  (1.0x) and no interference is modeled (honest: no interference is claimed without
  a measured matrix).
- FAILURES: injected deterministically from `failure_schedule` (hash_mismatch,
  partial_transfer, thermal_trip, link_drop, stale_epoch, unknown_profile,
  unsupported_backend) at fixed `at_us`; no random failures.

## 3. Residency pipeline in the sim

Each weight set on each phone advances through the RESIDENCY machine
(WEIGHT_RESIDENCY_CONTRACT section 3). Stage durations:

```
RECEIVING     = ceil(bytes / effective_goodput) + rtt          [BULK, preemptible]
VERIFYING     = ceil(bytes / verify_bytes_per_s)
MATERIALIZING = ceil(canonical_bytes / materialize_bytes_per_s)
PREPARING_*   = ceil(derived_bytes_* / prepare_*_bytes_per_s)   [0 for gpu_plain share]
WARMING       = warmup_us
```

Only when the machine reaches READY_* is a ReadyCertificate issued and the set
becomes dispatchable. A hash_mismatch at VERIFYING -> QUARANTINED (no publish); a
partial_transfer -> resume from the last verified chunk (not a restart).
Activation frames preempt in-progress RECEIVING on the same link (priority, section
6 of TRANSPORT_CONTRACT): a preempted transfer resumes after the activation drains,
extending its finish time -- a directly observable behavior.

V0 accounts for request input transfer but marks execution complete without a D2H
`output_bytes` stage. This is a known model gap, not permission to treat phone
results as free. V1 must return output before completion and resource release.

## 4. Baselines + sweep

- Runs all `baselines` in `sim_config` across every goodput in
  `link_goodput_sweep_bytes_per_s` (canonical sweep: 40, 100, 250, 400, 550 MiB/s =
  41943040, 104857600, 262144000, 419430400, 576716800 bytes/s).
- Per (goodput, baseline) it reports: completed, dispatched_to_phone,
  fell_back_to_server, weight_misses, server_gpu_us used/freed, server_hbm_bytes
  freed, transfer_bytes, prepare_us_total, evictions, deadline hits/misses,
  completion p50/p95, causal_score_total, and waited_for_weights.
- Policy behaviors: `server_only` never dispatches to a phone; `per_request_fetch`
  blocks each request on fetch+prepare (waits, diagnostic); `relief_predictive`
  never waits (section 2 of PREFETCH_POLICY) and prefetches by causal score;
  `clairvoyant` places with perfect foreknowledge (upper bound).

## 5. Required behaviors (mapped to tests)

The simulator + its harness must demonstrate, on fixtures:
1. Byte-identical deterministic replay (same config+seed -> same replay hash).
2. Hash-mismatch recovery: a hash_mismatch fault -> QUARANTINED, request falls back,
   never dispatched from bad bytes.
3. Partial-transfer recovery: resume from the last verified chunk, not a restart.
4. Duplicate / stale-epoch / reordered message rejection (frame-level, fail-closed).
5. Lease-safe eviction + exact physical-byte accounting (ledger partitions equal the
   total at every step; no eviction of a set with a live state lease).
6. No dispatch from partial or merely on-disk weights (VERIFIED_ON_DISK alone yields
   no ReadyCertificate -> FALLBACK_SERVER).
7. No live-state eviction.
8. Activation traffic preempts bulk prefetch (a decode activation delays an
   in-progress weight transfer, observable in the finish time).
9. Unknown profile / unsupported backend fail closed (hard gate -> FALLBACK_SERVER).
10. `relief_predictive.waited_for_weights == 0`.

## 6. Measured and symbolic parameters (honest)

Measured in `CURRENT_SLOW_LINK.md`: negotiated endpoint speed, root-bus topology,
and end-to-end ADB host-to-phone-file / phone-file-to-host staging rates. Those
staging rates cannot replace V0 `goodput_bytes_per_s` directly because V0 then
charges UFS again.

Still symbolic in S9-V0 fixtures:
- native memory-to-memory H2D/D2H transport goodput and RTT;
- `verify/materialize/prepare_*_bytes_per_s`, `warmup_us`, `phone_compute_us`,
  `per_class_gpu_us`, `hbm` costs: placeholder integers chosen for a legible fixture;
  they are NOT measured and are flagged as such in RESULTS.md.
- Interference per-mille: pinned at 1000 (no interference) unless a measured matrix
  is supplied; S9-V0 supplies none, so no interference is claimed.
- Derived-image bytes: the xmem formula (~1x extra per prepacked tensor,
  SUBSTRATE_AUDIT F3) informs the fixture ratio; exact per-tensor values are
  symbolic.

The simulator's role in S9-V0 is to prove CONTRACT MECHANICS (state machine, ledger,
eligibility, preemption, determinism, recovery), not to produce a capacity number.
A capacity claim requires the V1 model, the remaining measured rates, and confirmed
real traces.

## 7. Required S9-V1 link model

V1 is a versioned schema/simulator extension. Schema v1 and its replay hash remain
frozen. V1 must add:

```text
per device:
  link_profile_id, evidence_id, path_kind
  h2d_bytes_per_s, d2h_bytes_per_s, rtt_us
  contention_domain_id

per contention domain:
  h2d_cap_bytes_per_s, d2h_cap_bytes_per_s
  duplex_mode
```

Required behavior:

1. Share a directional domain cap only among active streams in that domain. An idle
   configured phone consumes no share; unrelated domains do not interfere.
2. Represent OP12 Bus 006 and OP15 Bus 008 as separate measured domains. Preserve
   the measured concurrent makespan result as evidence, not as a raw USB rate.
3. Distinguish `staged_file` from `native_buffer`. The staged H2D profile ends on
   the phone file path and must not blindly add a full second UFS-write duration;
   durable fsync/publish is a separate measured stage. The decomposed native profile
   charges transport, storage/materialization, verification, and preparation
   separately.
4. Schedule request input H2D and result D2H. Completion, state release, and server
   notification occur only after D2H succeeds.
5. Bind a measured profile to raw byte count, elapsed samples, direction, device
   path, timestamp, tool/mode, and payload hash. Unknown or mismatched evidence
   fails closed.
6. Retain the V0 40-550 MiB/s sweep as regression sensitivity. A measured staged
   scenario is separate. Optional 800/1000 MiB/s points are an explicitly
   unmeasured USB 10 Gbps future-device sensitivity.

V1 tests must cover single-active full rate, same-domain contention,
different-domain independence, directionality, staged-path no-double-count,
result-transfer completion, evidence rejection, failure recovery, and
byte-identical replay.

## V0-R repaired mechanics (implemented in sim/residency_sim.py)

The repaired simulator (bundle version 2; frozen V0 at
golden/v0_historical/residency_sim_v0.py) implements:

- Terminal taxonomy: every request ends in exactly one of completed_phone,
  completed_server, fallback, rejected, timed_out; the run asserts
  sum(buckets) == request count.
- Epoch binding: pipeline stages carry (pipeline_ver, generation, boot); dispatches
  carry (boot, generation, route, state). A stale stage is dropped; a stale completion
  is rejected fail-closed and its bytes rolled back. DRAINING rejects new work.
- Per-domain links: transfers contend within a contention_domain; separate domains
  (separate USB buses) are additive. Activation (H2D) and result (D2H) preempt a bulk
  transfer while retaining its ownership; a resume re-sends only the remaining verified
  range and counts retry bytes.
- Reserve-on-acquire with rollback: UFS at RECEIVING, LPDDR-canonical at MATERIALIZING,
  derived at PREPARING, scratch at publish, activation+state at dispatch. Any failure
  rolls the pipeline back (oversub_rollbacks). Sticky state persists until reset.
- D2H is an explicit completion blocker (the directional rate split is the S9-V1 item).
- Interference is applied ONLY over the actual resource-overlap window, and only when
  interference.measured is true.

Objective (documented, dimensionally valid). relief_predictive ranks candidates by a
LEXICOGRAPHIC key of INTEGER fields, higher is better, with NO cross-unit summation:

  (slo_feasible, server_relief_gpu_us, -cold_cost_us, reuse_permille,
   -eviction_loss_bytes, -interference_risk, -device_order)

server_relief_gpu_us and cold_cost_us are microseconds; reuse_permille is per-mille;
eviction_loss_bytes is bytes -- each compared only within its own key position, never
added across positions. The run also accumulates objective_relief_us_total and
objective_cost_us_total separately (both microseconds) for reporting.

## V0-R1 fail-closed closure (sim/residency_sim.py)

An independent adversarial pass found the V0-R simulator mechanics were still fail-open;
the R1 simulator (bundle version 3; frozen V0-R at golden/v0r_historical/) closes them:

- Readiness is backend/generation-qualified: is_ready(dev, ws, back) requires the
  residency's backend to match; a wrong-backend dispatch is refused.
- stale_epoch sets the device DRAINING and non-dispatchable; a request arriving after it
  falls back instead of re-prefetching onto a stale device.
- A stale pipeline stage rolls back EXACTLY ONCE (stale_cleaned), releases its contention
  domain, and wakes the bounded prefetch queue (domain_releases_on_stale).
- The full epoch stack (boot/generation/route/state) is rechecked after D2H before the
  output is accepted (post_d2h_epoch_rejects).
- Mutable state is stored per request/session (sticky_sessions map); reset is DEFERRED
  while a session is pinned (reset_deferred) and executed at unpin.
- HTP/GPU lane admission is bounded by queues.lane_depth; overflow -> explicit rejected
  terminal (lane_rejects).
- Event processing STOPS at the horizon (the run loop breaks when t > horizon_us; no
  unbounded polling).
- A partial failure carries an explicit verified offset (failure_schedule.verified_offset);
  failures at different offsets produce different retry bytes and completion times.
- link_drop is resumable in the bulk phase (link_drop_resumed) and terminal fail-closed
  in the activation/compute/d2h phases (link_drop_terminal), driven by
  failure_schedule.phase.
- Server relief is credited ONLY after a valid successful phone completion (in d2h_done).
- Measured interference is REJECTED as unsupported (MeasuredInterferenceUnsupported); the
  v3 sim_config schema also pins interference.measured to false.

Every request reaches exactly one terminal outcome and the run asserts
completed_phone + completed_server + fallback + rejected + timed_out == arrivals.

### 2026-07-14 review repairs

The current R1 simulator additionally:

- compares the exact device/weight owner before releasing a contention-domain bulk
  transfer, and scopes a phone link fault to that phone;
- executes deferred eviction after rejection reaches the final unpin;
- allows at most one in-flight mutation for a sticky/rebuildable session;
- cleans lane, activation, provisional state, pin, and server reservations on the
  simulation horizon;
- charges `server_gpu_us_used` only when server work is admitted; and
- checks lane, ledger, state, pin, session-owner, server-credit, and UFS-residency
  conservation after every event.

These are simulator mechanics only. They do not close the remaining v3 static
contract gaps or establish a capacity result.
