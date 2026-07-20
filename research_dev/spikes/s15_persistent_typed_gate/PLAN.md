# S15 typed persistent physical gate

Status: COMPLETE.

Verdict: `TYPED_PERSISTENT_OP15_B32_PHYSICAL_PASS_ENERGY_UNKNOWN`.

## Question

Can one typed runtime session keep both an OP15 `[0,8)` HTP head and an A6000
`[8,48)` tail resident across request cohorts, while preserving exact results,
placement, epochs, and bounded completion ownership?

## Frozen route

- Workload: the frozen 32-request BurstGPT cohort and fixed synthetic prompt
  from `s15_burst_cohort`.
- Batch: exact B32; 8 greedy decode tokens per request.
- Phone: OP15 HTP0, Gemma-4 layers `[0,8)`.
- Server: one selected A6000, layers `[8,48)`.
- Session sequence: launch 1 `DETACH`, launch 2 `STOP`.
- Deadline: 5,000,000 us per exchange.
- Required evidence: exact tokens, contiguous SESSIONCERT, stable phone and
  host process identity, HTP0 placement, CUDA0 tail placement, and one admitted
  boundary certificate per request.

## Gate

PASS requires all of the following:

1. Both typed `ExecutionRequest` objects complete through the persistent
   transport, bridge, physical mux, live adapter, and `PhysicalExecutor`.
2. All 32 token rows in each exchange equal the current same-batch CUDA
   reference, with exactly 8 tokens per row.
3. The OP15 PID, boot nonce, boot id, and layer range remain unchanged.
4. Session ids are `[1,2]`; DETACH applies reset and STOP terminates cleanly.
5. Phone compute is on HTP0, except the declared GET_ROWS CPU seam.
6. The A6000 tail records positive CUDA0 compute, zero missing buffers, and no
   CPU or other-backend compute.
7. Every persisted artifact reopens with its declared digest, and raw mux
   streams reproduce the report.
8. No phone, USB, server-wall, or total-system energy claim is emitted.

## Next gate

Run repeated mixed BGE+Gemma windows under matched server-only and READY-phone
policies. Report selected-GPU board energy and per-class SLO together. Add OP12
later as an independent request lane, never as an OP15-to-OP12 serial chain.
