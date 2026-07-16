# Prefetch / Residency Policy (S9-V0)

Status: frozen historical S9-V0 simulator policy. Its additive causal score is
not the current scheduler objective. Current policy is defined by
`TWO_LEVEL_SCHEDULER.md`; R1 simulator mechanics and the replacement
lexicographic objective are documented in `SIMULATOR_SPEC.md` and
`V0R1_REPAIR.md`. The never-wait-for-weights invariant remains active.

This is a POLICY specification for the offline simulator (SIMULATOR_SPEC.md). No
production scheduler is built in S9-V0. All results use synthetic fixtures until
S8 Gate A is independently confirmed passing; no real-trace residency result is
claimed.

## 1. Two loops

- SLOW loop (residency / placement): runs at a residency horizon. Decides which
  weight sets to prefetch, materialize, prepare, and lease onto which phones, and
  when to evict. Uses hysteresis (`min_hold_us`) so it does not thrash residency
  per request. Updated by the predictor (section 4).
- FAST loop (dispatch): reacts at every arrival and completion. For each request it
  evaluates the DispatchDecision eligibility rule (WEIGHT_RESIDENCY_CONTRACT section
  4). It does NOT trigger a download; it only routes to a READY phone or to the
  server. Runtime activation traffic must never trigger an implicit model download
  (MIXED_WORKLOAD_DESIGN section 5).

## 2. The inviolable rule: never wait for weights

The production candidate (`relief_predictive`) NEVER blocks a request on weight
arrival. On a weight MISS (no ReadyCertificate for the required set on any eligible
phone), the request IMMEDIATELY falls back to the server, and the miss only updates
the slow-loop predictor. Prefetch/materialize/prepare happen in the background on
the slow loop; they never sit on a request's critical path.

Consequences (all are required simulator behaviors + tests):
- `SimRunManifest.baselines[].waited_for_weights == 0` for `relief_predictive`.
- A miss is a FALLBACK_SERVER dispatch with `reason_code: not_resident`, not a stall.
- `per_request_fetch` (below) is the DIAGNOSTIC opposite -- it DOES wait, to quantify
  the cost the production candidate avoids -- and is labeled a losing baseline.

## 3. Baselines (simulator implements all eight)

| Policy | Slow-loop behavior | Label |
|---|---|---|
| `server_only` | never uses phones | reference floor |
| `per_request_fetch` | on each request, fetch+prepare weights THEN run on phone (waits) | DIAGNOSTIC losing |
| `static_placement` | a fixed model-to-phone assignment, never changes | baseline |
| `lru` | evict least-recently-used weight set on pressure | baseline |
| `lfu_ttl` | evict least-frequently-used; expire past a TTL | baseline |
| `fastest_ready` | dispatch to the phone that is READY soonest | baseline |
| `relief_predictive` | causal-score residency + never-wait fallback | PRODUCTION candidate |
| `clairvoyant` | perfect-knowledge placement | UPPER BOUND (labeled) |

- `clairvoyant` is an upper bound, clearly labeled `is_upper_bound=true`; it is NOT
  achievable and is never presented as the proposed system's result.
- `per_request_fetch` is the losing diagnostic, `is_diagnostic_losing=true`; it
  isolates the cold-start cost.
- Only `relief_predictive` obeys section 2 (never waits). The other seven are
  comparison points; `static_placement`/`lru`/`lfu_ttl`/`fastest_ready` may or may
  not wait depending on residency, which is exactly what the sweep measures.

## 4. Historical V0 causal residency score (not current policy)

V0 scored each candidate (weight set -> phone) placement with the additive
utility below. It is retained only to reproduce the frozen V0 result; terms with
different units are not a valid current objective.

```
score = expected_server_relief
      + avoided_cold_start
      + expected_reuse
      - transfer_cost
      - preparation_cost
      - eviction_loss
      - ram_thermal_interference_risk
```

- `expected_server_relief`: predicted freed server GPU-us + HBM bytes for requests
  the placement would serve, valued under current server pressure (a vector, not one
  latency number; MIXED_WORKLOAD_DESIGN section 8.1). Relief is COMPARATIVE (vs the
  server running it), never absolute phone speed.
- `avoided_cold_start`: cold-start (transfer + materialize + prepare + warm) cost
  that future requests avoid because the set is already READY.
- `expected_reuse`: predicted hit count over the reuse horizon (EWMA of observed
  arrivals per weight set, `predictor_ewma_permille`).
- `transfer_cost`: path-aware bytes / measured directional goodput. For a
  `staged_file` profile, use the measured host-to-phone-file wall time and do not
  blindly add a second full UFS-write charge; measure durable fsync/publish
  separately. For a `native_buffer` profile, use H2D transport plus separately
  measured storage/materialization stages. Never substitute negotiated USB
  signaling rate for application goodput.
- `preparation_cost`: materialize + backend-derived prepare + warmup us.
- `eviction_loss`: value of what must be evicted to make room (RAM pressure).
- `ram_thermal_interference_risk`: penalty for tight RAM, thermal ineligibility, or
  measured HTP/GPU pairwise interference; honored only when interference is measured
  (else the term is 0, matching `interference.measured==false`).

The slow loop admits placements in descending score while the physical-byte ledger
(WEIGHT_RESIDENCY_CONTRACT section 6) stays valid, respecting `min_hold_us`
hysteresis. The predictor is updated by every miss/hit; a miss raises the predicted
reuse of that set, so the slow loop may prefetch it -- but the missing request
itself already fell back (section 2).

The current staged H2D medians make the prediction horizon concrete: a 900 MiB set
needs about 4.2 s on OP12 or 3.4 s on OP15 before verification, backend preparation,
and warmup. A 10 GiB set needs roughly 47 s or 39 s. Therefore the slow loop should
operate seconds to tens of seconds ahead, and model turnover should be driven by
reuse windows and server-pressure forecasts, never by the current request.

## 5. Eligibility interactions (hard gates before scoring)

Scoring only ranks ELIGIBLE placements. A placement is ineligible (never scored) if
the model is not `partial_load_supported` for the requested sub-range
(arbitrary-model REJECT, SUBSTRATE_AUDIT G3), the SoC profile is unknown, the
backend is unsupported for the island, the phone is thermally ineligible or stale,
or the LPDDR ledger cannot fit weights+derived+scratch+activations+state. These are
`hard_gate_failures`; they force FALLBACK_SERVER and never a silent phone route.

## 6. Objective modes (reported separately)

- Capacity mode: maximize SLO-valid admitted work while valuing freed server GPU-us
  / HBM (the causal score's relief term). This is the S9-V0 objective.
- Energy mode: DISABLED. No energy is inferred from latency or capacity; an energy
  claim requires synchronized physical instrumentation (MIXED_WORKLOAD_DESIGN
  section 9) that this spike does not have. The simulator emits resource ledgers, not
  joules.
