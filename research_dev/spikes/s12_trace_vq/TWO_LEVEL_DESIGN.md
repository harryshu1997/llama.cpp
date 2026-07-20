# S12-V2 Two-Level Mixed-Residency Scheduler

Status: `SYNTHETIC_TWO_LEVEL_MECHANICS_PASS`

## 1. Scope

S12-V2 is a new deterministic replay. S12-V0 and S12-V1 remain frozen as
regression oracles. V2 answers one mechanics question:

Can a causal slow loop move complete operator-island weight sets across OP12
and OP15 while a fast loop dispatches only symbolically READY replicas with
matching synthetic identities, preserves bounded resources, and immediately
falls back to the server on a cold miss?

The V2 fixture is entirely synthetic. OP12 has no measured complete-route row
and the second model has no measured phone profile. Every output is labeled:

~~~text
SYNTHETIC_MIXED_RESIDENCY_MECHANICS_ONLY
LATENCY_CAPACITY_ENERGY_NOT_CLAIMED
ENERGY_NOT_RUN
~~~

## 2. Frozen V2a Boundary

V2a models one READY operator island per request followed by a server tail. It
does not model multi-island DAG traversal, cross-model batching, the measured
WiFi/USB phase decomposition, or physical energy. Synthetic route durations
exist only to order events and exercise resource races.

The fixture has no measured correctness or readiness-certificate artifact.
READY binds symbolic weight identity, model, backend, device boot epoch,
residency generation, and status sequence. This is a state-machine test, not
an S9 certificate-validation claim.

The host keeps full-model fallback residency throughout the replay. Phone
placement receives no A6000 HBM relief credit. Exclusive host eviction and
tail-only residency remain blocked on measured load/unload transitions.

## 3. Two Loops

The slow loop receives an immutable snapshot containing only:

- the current timestamp;
- currently queued requests and their execution signatures;
- cumulative arrivals already observed per island;
- current replica states, generations, pins, and capacity; and
- already-issued residency intents.

It cannot inspect the future event heap or future arrivals. It may emit:

- `KEEP`: retain a coherent READY/LEASED replica;
- `PREFETCH`: create the first replica of an island;
- `REPLICATE`: create another device-local replica; or
- `DEFERRED_EVICT_PREFETCH`: drain a pinned victim, then replace it.

The engine validates and applies intents. An island advances through:

~~~text
RECEIVING -> VERIFYING -> PREPARING -> READY
READY -> LEASED -> READY
READY/LEASED -> DRAINING -> EVICTING -> absent
~~~

Every transition binds device boot epoch, residency generation, and status
sequence. Cross-device replicas have independent symbolic allocations,
generations, and LPDDR charges. Only immutable island identity is shared.

The fast loop may emit:

- `DISPATCH_PHONE`, only against a READY replica with a current generation and
  an available device activation slot;
- `BATCH_TAIL`, when a phone result reaches the shared server lane; or
- `SERVER_FALLBACK`, immediately when the server is free and no READY phone
  route has already claimed the request.

Provisioning never blocks a free server. A DRAINING replica rejects new phone
dispatch. Pins remain live until the corresponding server tail completes.

`BATCH_TAIL` carries `batch_size=1` in V2a. No tail merging or cross-request
batching is implemented in this slice.

When placements compete, the slow loop enumerates every feasible assignment
of zero, one, or two intents. It maximizes `(total_score_us, action_count)` and
uses a stable identity tie break. The score uses current queued demand and
synthetic durations; it is not a validated arrival predictor or energy
objective.

## 4. Deterministic Fixture

The fixture contains two symbolic models and seven requests:

- island A is initially READY on OP15 and compatible with both phones;
- island B is compatible only with OP12;
- four A requests arrive at time zero, causing causal replication of A to
  OP12 while OP15 and the server begin work;
- three B requests arrive later; the first cold miss uses the server;
- the second observed B request creates an OP12 prefetch intent;
- A on OP12 is still pinned, so eviction enters DRAINING;
- completion releases the pin before eviction at the same timestamp;
- B then progresses to READY and is dispatched on OP12.

The final placement is A on OP15 and B on OP12. The replay must show a positive
OP12/OP15 compute overlap, independent replica charges, safe deferred eviction,
and exact terminal conservation.

## 5. Resource And Event Rules

- One finite host virtual queue.
- One server lane shared by full fallback work and phone-result tails; tails
  have priority once ready.
- One compute/context slot per phone.
- One bounded activation/result credit per phone.
- One bounded bulk residency pipeline per phone.
- Exact LPDDR capacity and transfer/eviction byte ledgers.
- Stable same-time order: execution completion, residency transition, arrival,
  slow loop, fast loop, insertion sequence.
- Horizon cleanup terminalizes every uncompleted request and releases every
  pin and credit exactly once.

Terminal conservation is:

~~~text
completed_server + completed_phone + tardy_server + tardy_phone
  + rejected_queue_full + timed_out == arrivals
~~~

Energy status is always `NOT_RUN`; joule fields and saving labels are absent.

## 6. V2a Gates

- [x] Strict ASCII JSON, duplicate-key rejection, exact keys, and bool/int
      separation.
- [x] V0/V1 replay hashes unchanged.
- [x] Causal prefix invariance under future-trace mutation.
- [x] Cold residency never delays an otherwise free server.
- [x] Dispatch before READY rejected.
- [x] Pinned eviction enters DRAINING and completes only after the last pin.
- [x] Cross-device replicas are charged independently.
- [x] OP12 and OP15 execution overlap; same-device execution serializes.
- [x] Every queue, lane, activation slot, LPDDR charge, and pin stays bounded.
- [x] Transfer, eviction, residency, dispatch, and terminal ledgers reconcile.
- [x] Horizon and queue overflow produce explicit terminals.
- [x] Canonical replay is identical across hash seeds and bundle locations.
- [x] No latency, capacity, or energy conclusion is expressible.

Passing V2a authorizes only a richer synthetic workload and measured profile
adapter. It does not authorize a live daemon or `llama-server` integration.
