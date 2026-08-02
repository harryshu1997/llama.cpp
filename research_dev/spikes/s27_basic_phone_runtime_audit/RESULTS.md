# S27 Basic Phone Runtime Audit

Verdict: `FIXED_ROUTE_CONTINUOUS_BATCHING_PASS_ARBITRARY_LAYER_EXIT_NOT_IMPLEMENTED`

This is a post-S24 capability audit. It does not replace S24's
`BENEFIT_GATE_FAIL` verdict and makes no energy claim.

## Capability status

| Capability | Status | Evidence |
| --- | --- | --- |
| Persistent phone weights | PASS | One OP12 and one OP15 process served both physical sessions without reload. |
| Per-request KV lifecycle | PASS | Request and route epochs, sequence removal, slot reuse, and zero final KV/leases passed. |
| Phone continuous batching | PASS, bounded | Variable-row HTP batches formed online; each phone currently has four sequence slots and a B4 knee. |
| Server-tail continuous batching | PASS, bounded | One shared CUDA tail accepted all routes, reached B8 on the observed-length run, and drained cleanly. |
| Request-level placement | PASS for fixed routes | C3 selected R0 or R2 per request from SLO profiles and live route/resource credits. |
| Static contiguous layer cuts | PASS at tested Gemma 4 cuts | Workers load a startup-fixed `[layer_start,layer_end)` slice; layers 8 and 16 are physically certified boundaries. |
| Exit at any layer per request | NOT IMPLEMENTED | A resident worker has one fixed layer range. The runtime only selects pre-provisioned exits at layers 8 and 16. |
| Semantic early exit | NOT IMPLEMENTED | Every request still executes all 48 layers before token generation. |
| Dynamic model switching | NOT IMPLEMENTED | This run keeps one Gemma model and its slices resident. |

## Physical topology

- OP12 HTP0: Gemma 4 F16 layers `[0,8)`, four sequence slots.
- OP15 HTP0: Gemma 4 F16 layers `[8,16)`, four sequence slots.
- RTX 4060 Ti CUDA0: `[0,8)`, `[8,16)`, and shared `[16,48)` workers with
  four, four, and eight sequence slots.
- Routes: R0 is all CUDA; R1 uses CUDA -> OP15 -> CUDA; R2 uses OP12 ->
  OP15 -> CUDA.
- Desktop-to-phone activation transport was direct WiFi. The A6000 only kept
  the USB-connected phone processes alive and did not execute model layers.

## Batching defect and repair

The first 60-request run exposed a real batching defect in
`DeviceBatcher._run`. Once the oldest row's gather deadline had expired, the
worker dispatched that row alone without draining rows already present in the
queue. An expired deadline now prevents additional waiting but still drains
the ready backlog up to the device knee. A deterministic regression test
requires the post-fix batch sequence `[1,2]` for an overdue two-row backlog.

The before/after dense runs used the same trace, but the dynamic C3 route mix
changed with timing, so the makespan delta is informative rather than a
controlled speedup claim.

| Dense metric | Before repair | After repair |
| --- | ---: | ---: |
| OP12 mean/max batch | 1.05 / 4 | 2.40 / 4 |
| OP15 mean/max batch | 1.05 / 4 | 2.40 / 4 |
| CUDA-tail mean/max batch | 1.07 / 4 | 3.58 / 6 |
| Makespan | 11.374 s | 8.315 s |
| Route mix | R0 40, R2 20 | R0 48, R2 12 |

## Real-device trace results

Both runs passed `validate_cp6_workloads.sh` and produced
`WORKLOAD_GATES_PASS`.

| Trace | Physical work | Placement | Completion and SLO |
| --- | --- | --- | --- |
| Dense mechanics | 60 BurstGPT-derived arrivals at 0/1/2 s; one-token prompt and four-step physical proxy | Dynamic C3: R0 48, R2 12 | 60/60 complete, 0 rejected, 0 misses, 8.315 s |
| Observed context | 28 BurstGPT-derived arrivals with observed input/output lengths; 12,586 prompt tokens and 958 output tokens | Pinned C2: R0 6, R1 9, R2 13 | 28/28 complete, 0 rejected, 27 misses, 894.182 s |

The observed-context batching result is:

| Worker | Batches | Mean batch | Max batch |
| --- | ---: | ---: | ---: |
| OP12 `[0,8)` | 1,875 | 3.573 | 4 |
| OP15 `[8,16)` | 2,995 | 3.613 | 4 |
| CUDA prefix `[0,8)` | 1,916 | 3.557 | 4 |
| CUDA middle `[8,16)` | 785 | 3.434 | 4 |
| CUDA tail `[16,48)` | 2,216 | 6.099 | 8 |

The observed-length run proves lifecycle and batching under unequal request
lengths. Its 27 SLO misses also show that the current fixed phone route is not
a viable policy for these synthetic deadlines.

## Validation scope

For both sessions, the validator proved exact row counts, request/route
lineage, position continuity, scheduled backend placement, zero missing
buffers, zero residual software leases, and zero residual worker KV. OP12's
only CPU compute was the declared metadata `GET_ROWS`; OP15 stayed on HTP0;
the desktop stages stayed on CUDA0 except the declared CUDA-host metadata
lookup.

The traces retain real arrivals and observed token counts from BurstGPT, but
token values, priorities, and SLOs are synthetic. Phone F16 to desktop Q8
boundaries remain numerically uncertified, so this is not a model-quality
certificate.

Raw evidence is under:

`research_dev/spikes/s24_overlap_handoff_poc/results/cp6_batchfix_real_workloads/`

The local `SHA256SUMS.txt` verifies all synchronized runtime reports, worker
logs, session certificates, and validation records.

## Next implementation gate

1. Represent several resident overlapping cuts as a finite measured route
   catalog, then select among them per request. Do not mutate a worker's graph
   while it has live KV.
2. Separate prefill row capacity from decode sequence capacity so long prompts
   can use a larger measured prefill knee while decode remains continuous.
3. Add priority-aware ordering inside each device queue and rerun an
   observed-length C3 treatment against an all-CUDA control.
4. Add a matched monolithic output oracle before making a quality claim.
