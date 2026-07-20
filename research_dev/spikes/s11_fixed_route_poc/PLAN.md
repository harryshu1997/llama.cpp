# S11 Fixed-Route Proof of Concept

## Question

Can a resident phone-owned contiguous layer island reduce work on one selected
A6000, release server VRAM, and preserve exact greedy output?

This is the smallest physical test of the Q-PIM direction. It does not test the
optimizer yet. The routes are fixed before each run:

1. `SERVER_ONLY`: all layers execute on one selected CUDA GPU.
2. `A0_OP15`: OP15 owns `[0, k2)` and CUDA owns `[k2, n)`.
3. `A0_OP15_OP12`: OP15 owns `[0, k2)`, OP12 owns `[k2, k3)`, and CUDA owns
   `[k3, n)`.

## Mechanism

`llama-layersplit` gains:

- `monodriver`, a resident full-model control using the same greedy request loop
  as `pipedriver`;
- an optional second phone stage;
- resettable persistent `stagenet` contexts;
- batched prompt prefill followed by single-token incremental decode;
- warmup before a `DRIVER_READY` barrier;
- exact per-request `ROUTEJSON` token and timing records.

`run_fixed_route.py` v4:

- selects one physical GPU by UUID;
- uses ABBA route order across pairs;
- starts measurement only after all route weights and warmups are resident;
- requires exact token-id equality for every paired request;
- records selected-GPU memory at the ready barrier;
- records integer monotonic timestamps and NVML board power;
- integrates gross GPU-board energy by left-edge zero-order hold;
- applies the frozen 100-update, 250 ms gap, and +/-5 W gates;
- records p-state transitions as an outcome instead of rejecting the mechanism
  that the experiment is intended to observe;
- requires at least 32 realized generated tokens per request, eight complete
  pairs, and a predeclared p95 group-latency SLO in measurement mode;
- restricts measurement to the frozen OP15 `[0,2)` route, selected GPU UUID,
  B=8, 32 requested tokens, chat formatting, and two or more warmup groups;
- hashes every raw power stream and rejects reuse or mixed GPU UUIDs; and
- reports a conservative 10 percent gate plus exact J/request and J/token
  numerators and denominators.

## Claim Boundary

The strongest possible label from this runner is:

`GPU_BOARD_DIAGNOSTIC_10PCT_OBSERVED_TOTAL_ENERGY_UNKNOWN`

It is not a formal energy-saving result. NVML excludes the host CPU, DRAM, PSU,
USB, phones, and chargers. The plan also lacks an independent enumerable
pre-registration anchor. Phone-local and total-wall energy remain blocked.

The paid boundary is steady-state only: the host timestamp immediately before
the `GO` release through `DRIVER_DONE`, after model loading, weight provisioning,
and warmup. It does not measure cold start, weight transfer, teardown, or
lifecycle energy.

## Gates

1. Both routes complete without fallback or process failure.
2. Every measured request has the same prompt-token count, requested-token
   count, EOG state, and exact generated token IDs.
3. VRAM relief is the paired difference in selected-GPU `memory.used` at the
   ready barrier. It is a resource result, not an energy result.
4. Measurement mode has at least eight complete pairs.
5. Every timeline passes the sample-quality gates.
6. A board-relief observation requires:

   `(control_energy - control_uncertainty) >
    (treatment_energy + treatment_uncertainty)`

Any failed gate produces no relief label.

## Result and Next Checkpoint

The repaired Android toolchain build and real OP15 readiness gate pass as of
2026-07-17. A 512-request pair produced exact greedy tokens for all requests,
888 MiB selected-A6000 VRAM relief, valid scheduled-buffer placement evidence,
and valid continuous phone thermal evidence. The control and treatment p95
group latencies were 1,331,642 us and 3,235,782 us. HMX temperature peaked at
49.2 C with Android thermal status 0.

`ACQUISITION_FREEZE.json` fixed 512 requests per timeline and a 3,500,000 us p95
SLO before energy acquisition. The eight-pair B=8 selected-A6000 board
diagnostic then completed with all exactness, evidence, and SLO gates valid.
For identical work, server-only consumed 197.4 kJ and the phone route consumed
302.3 kJ on the selected A6000 board. Lower average board power (289 W to 192 W)
did not offset the 2.31x aggregate runtime. The fixed serial `[0,2)` route is
rejected as an energy-saving primitive.

Placement evidence is deliberately scoped to scheduled output-buffer location.
The public graph callback runs before backend execution and cannot prove that a
kernel completed. Process exit status, exact output, backend support, and the
scheduled-buffer certificate are separate gates. B=16, the two-phone route,
and the S12 WiFi-input/USB-result topology require a separate reviewed plan.
Per the stop rule, do not run B=16 or another layer boundary to rescue this
result. The next mechanism must overlap phone work with useful server work, for
example replicated phone-resident islands serving concurrent batches or mixed
jobs while the A6000 executes already-ready tails. It needs its own matched
server-only control and cannot inherit an energy claim from this experiment.
