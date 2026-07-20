# S16 Mixed Persistent Energy Gate

## Question

Can a READY OP15 resident island execute low-priority Gemma decode work while
one selected A6000 preserves high-priority BGE latency and uses less GPU-board
energy than an otherwise identical server-only policy?

This is a real-device selected-GPU diagnostic. Phone, USB, host-wall, and total
system energy remain unknown.

## Frozen work

- One A6000 only. The second A6000 must have no compute process.
- High priority: BGE-small-en-v1.5, sequence target 32, measured knee B16.
- Low priority: the frozen 32-request BurstGPT cohort, Gemma-4-12B, 8 tokens.
- Twenty consecutive low-priority B32 cohorts in every paid row.
- P0: persistent full Gemma `[0,48)` on the selected A6000.
- P2: persistent OP15 `[0,8)` HTP0 plus persistent A6000 `[8,48)`.
- Three matched P0/P2 pairs in rotated order: P0,P2 / P2,P0 / P0,P2.
- Model preparation and readiness occur outside the paid window in both plans.
- The paid window is the exact BGE benchmark window. All low work must complete
  inside it.

Priorities, the 5 s low-priority SLO, and repeated payloads are synthetic. The
arrival cohort itself is observed BurstGPT data.

## Frozen gates

1. Every row completes identical BGE encodes and 20 x 32 x 8 Gemma tokens.
2. Every Gemma result matches the frozen CUDA token reference.
3. Every exchange has a current placement certificate. P0 permits only CUDA0
   compute plus `GET_ROWS` on CUDA_Host. P2 requires OP15 HTP0 plus the declared
   phone `GET_ROWS` seam and a CUDA0 tail.
4. One persistent host PID is retained within a row. P2 also retains one phone
   PID and boot nonce. DETACH resets KV; only the last exchange uses STOP.
5. Median high-priority BGE p95(P2) / p95(P0) is at most 1.05.
6. Every low-priority B32 cohort finishes within 5,000,000 us.
7. Every paid power window is bracketed, has a maximum sample gap of 250,000 us,
   at least 100 actual power-value changes, and one invariant power limit.
8. The median paired selected-GPU saving is positive after subtracting the
   Ampere NVML 5 W uncertainty floor from both members of each pair.

Gate 8 authorizes only a `GPU_BOARD` diagnostic. It cannot authorize a server
wall, phone, USB, or total-system energy claim.

## Execution order

1. Build CPU and CUDA `llama-layersplit`.
2. Run the persistent monodriver reset regression.
3. Run one short P0/P2 real-device screen without an energy verdict.
4. Run the frozen six-row acquisition once.
5. Reopen raw samples and process evidence with the independent validator.

