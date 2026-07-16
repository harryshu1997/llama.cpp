# S10-V0 invalidation audit

Status: INVALID / INCONCLUSIVE. PF1 remains unauthorized.

The historical screen under `../s10_power_frontier/` produced internally
reproducible files, but its abstractions did not implement the experiment in the
S10 plan.

## Blocking findings

1. The oracle selected placement plus all-or-none batching under one fixed EDF
   schedule. It did not enumerate arbitrary batch partitions or DAG orders.
2. A legal `[16] + [32]` partition yields zero misses for deadlines
   `[550,1200,1200]`, while the V0 oracle returns one miss.
3. C0 was not FIFO, C1 did not optimize DAG order/power/lazy claims, C3 was
   static placement, and C5 evaluated arrived waves using future requests.
4. The workload gave all requests in a wave the same release and slack. It had
   no tight batch spoiler or dependency unlocker capable of demonstrating the
   proposed mechanism.
5. Multiple model and weight names reused one measured Gemma4 FFN profile. They
   were not independently measured model routes.
6. Energy was a function of total busy time. It omitted P0/P8 transition and
   gap placement, which are load-bearing to the power-frontier claim.
7. Auto-entered P8 was incorrectly treated as unusable. Avoiding a P8-to-P0
   wake is itself a possible scheduler lever.
8. The standalone checker accepted resealed certificates with wrong schedule
   identities, forged busy/objective fields, phantom exclusive HBM relief, and
   extra phone completions.
9. The verdict path hardcoded physical/mechanism inputs and did not enforce the
   complete C5 causal threshold.
10. The atlas did not meet the planned independent-process protocol and did not
    separate all transfer legs. Complete-wall energy remained blocked.

## Evidence retained

The following remain useful component observations, not a system verdict:

- measured A6000 dense-FFN latency versus M;
- correct OP12/OP15 resident M=16 FFN execution;
- measured phone end-to-end latency for that island;
- repaired S9 provisioning/transport evidence; and
- the absence of a valid synchronized total-wall power boundary.

The correct historical label is `S10-V0 INVALID / INCONCLUSIVE`, not a
falsification of Q-PIM.

## Repair verification

Independent review of S10-V0-R found and closed additional fail-open behavior:

- Python booleans were accepted as integer timing and energy fields;
- terminal nodes did not have to close their request DAGs;
- additional devices could claim `kind=server` but be charged as phones;
- malformed unused batch profiles and request priorities escaped the checker;
- causal planning after time zero could return an action in the past; and
- exact-search metadata admitted inconsistent zero counts.

The repaired record contract now also requires `evidence.scope` to be
`MECHANICS_ONLY`; these synthetic and retained component inputs cannot be
mistaken for a measured system-energy result.

The review also found that the repaired solver's earliest-start construction was
inexact for transition-aware energy and activation-constrained placement. Those
defects are now closed in the bounded domain: TEMPORAL mode enumerates intentional
delay, while EARLIEST mode is restricted to the case where start time cannot change
the objective. A structurally independent exhaustive reference verifies the
in-domain optima.

A later adversarial check found that plausible search counters did not prove
optimality: the checker accepted a resealed, feasible, suboptimal schedule. Search
counters are no longer signed certificate fields. Default exact-certificate mode
recomputes the optimum independently; explicit feasibility-only mode retains
schedule/accounting checks without claiming optimality. The temporal foundation now
passes, but typed evidence still blocks C0-C5.
