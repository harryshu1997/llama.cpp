# S19 distributed continuous-batching runtime

Status: CP0 server audit complete; CP1 implementation not started.

## Goal

Replace fixed B32 cohorts with arrival-aware, SLO-bounded continuous batching
across persistent OP12/OP15 prefixes, one CUDA cut-normalization bridge, and one
CUDA tail. Borrow the proven slot/batch lifecycle from `llama-server`, while
retaining only the route, priority, deadline, and measured-batch decisions
needed by the focused Q-PIM Funnel system.

This gate is physical-runtime work. A simulator result cannot pass it.

## Invariants

1. One admitted request owns one stable route and one KV sequence in every
   stage for its lifetime.
2. Completing or canceling request A cannot clear or mutate request B.
3. Every stage consumes the same request/sequence/position/route epoch
   manifest for a token step.
4. New prompt work may enter unused capacity while older requests decode.
5. Batch release uses measured candidates and the earliest latest-start time;
   it never waits unconditionally for B32.
6. Downstream credits are reserved before a phone batch launches.
7. Unknown, stale, duplicated, or out-of-order rows fail closed.
8. The selected A6000 is the only server executor; the second GPU remains idle.

## CP0 - reuse audit

- [x] Trace llama-server slot, queue, batch, compatibility, and per-sequence KV
      lifecycle.
- [x] Map reusable mechanics to the current StageNet protocol.
- [x] Record why public intermediate-activation APIs and direct
      `server_context` copying are rejected.
- [x] Freeze `SERVER_REUSE_AUDIT.md` before editing the runtime.

## CP1 - versioned StageNet sequence protocol

Add an opt-in protocol version without changing the existing v1/v2 opcodes.

- [ ] `BATCH_PREFILL_ROWS`: bounded rows carrying global request ID,
      phone-local sequence ID, position, token, route epoch, and optional input
      activation. Multiple contiguous positions may belong to one sequence.
- [ ] `BATCH_DECODE_ROWS`: bind the existing variable-row decode payload to
      request and route epochs, and echo the accepted manifest identity.
- [ ] `SEQ_REMOVE`: remove exactly one sequence with
      `llama_memory_seq_rm(..., seq_id, -1, -1)` and acknowledge its generation.
- [ ] `SESSION_DRAIN`: reject new work, retire live sequences, then detach or
      stop. Do not use a global memory clear during a continuous session.
- [ ] Maintain a bounded worker shadow table with FREE/PREFILL/DECODE states and
      the next expected position. Validate slot reuse and stale epochs before
      calling `llama_decode`.
- [ ] Add malformed, duplicate-row, stale-epoch, position-gap, double-remove,
      capacity, and disconnect tests. Existing protocol behavior must remain a
      regression test.

## CP2 - LayerSplit slot and batch lifecycle

Implement a small executor-local equivalent of the useful server mechanics:

~~~text
distributed_slot
  FREE -> PREFILL -> DECODE -> FINISHED/CANCELED -> FREE

distributed_batch_manifest
  rows[(request_id, seq_id, token, pos, route_epoch, output)]
~~~

- [ ] Keep a bounded pending queue ordered by priority and latest start.
- [ ] At each update, retire finished/canceled slots, admit feasible pending
      prompts, and add one current token for each compatible decode slot.
- [ ] Render one manifest for the phone and the matching activation batch for
      the CUDA tail.
- [ ] Use per-sequence removal in the local tail; never clear the complete tail
      while another slot is live.
- [ ] Preserve exact terminal ownership and conservation accounting.
- [ ] Expose batch membership, queue delay, TTFT, TBT, completion, selected
      measured candidate, and release reason in JSONL.

`llama-server`'s task/LoRA/sampler classes are not copied. Greedy sampling is
kept for the first correctness gate; sampler generalization follows only after
the distributed lifecycle passes.

## CP3 - host correctness and lifecycle gate

- [ ] Use staggered arrivals and unequal output lengths.
- [ ] Prove at least one mid-decode retirement and one admission into the freed
      slot while another sequence remains live.
- [ ] Compare every generated token with a same-request monolithic CUDA oracle.
- [ ] Prove canceling one request leaves all other token streams unchanged.
- [ ] Prove the active batch membership changes between token steps.
- [ ] Run fail-closed protocol negatives and ASan/UBSan host tests.

## CP4 - one-phone physical gate

- [ ] Deploy the versioned worker to OP15 without replacing the frozen S18
      binary.
- [ ] Run OP15 `[0,8)` plus persistent CUDA `[8,48)` over at least seven
      staggered sessions.
- [ ] Require token correctness, HTP/CUDA placement certificates, persistent
      process identities, sequence-isolated KV, zero stale rows, and bounded
      SLO misses.
- [ ] Compare fixed-cohort versus continuous-batch valid throughput and queue
      delay. Do not make an energy claim here.

## CP5 - heterogeneous-cut normalization and shared tail

Rows with different layer cuts cannot share one native llama batch. Normalize
both routes to layer 8 before the one shared tail:

~~~text
OP12 [0,6) -> CUDA bridge [6,8) --+
                                      +-> CUDA tail [8,48)
OP15 [0,8) --------------------------+
~~~

- [ ] Load one persistent CUDA `[6,8)` bridge and one persistent CUDA `[8,48)`
      tail. The weight ranges are disjoint and the tail has one weight image.
- [ ] Add a CUDA `[0,8)` fallback head that feeds the same tail. Account for the
      initial `[6,8)` head/bridge overlap explicitly; no `[8,48)` tensor may be
      duplicated.
- [ ] Give each phone an independent local slot namespace; map both to unique
      tail sequence IDs in the host manifest.
- [ ] Keep bridge KV only for OP12-owned requests and tail KV for every admitted
      phone request. A completion removes only the matching sequences.
- [ ] Admit and retire requests continuously on both phones. Combine returned
      phone and fallback-head layer-8 rows into the next CUDA-tail batch subject
      to its measured knee, memory, and earliest SLO.
- [ ] Prove no phone waits for the other when a deadline requires release.
- [ ] Compare peak HBM with S18's 45,674 MiB treatment and 26,555 MiB control.
- [ ] Stop before energy if one-tail execution loses SLO-valid goodput or does
      not remove duplicated tail residency.

This CUDA bridge is the only cut-normalization mechanism in the paper-critical
path. The failed S17 OP15 middle route, phone-to-phone transfer, arbitrary
merge-cut selection, and per-row variable-start graphs are out of scope.

## CP6 - mixed trace and selected-GPU screen

- [ ] Drive observed BurstGPT arrivals plus the frozen synthetic priority/SLO
      sidecar while high-priority BGE runs at its measured server knee.
- [ ] Run four equal-work controls:
      `C0` optimized server-only continuous batching;
      `C1` S18 fixed B32 independent phone tails;
      `C2` both phones at common cut 6 with one shared tail; and
      `Q` heterogeneous cuts plus normalization and continuous shared tail.
- [ ] Sweep a small predeclared set of generation/embedding load ratios rather
      than one BGE-dominated point. Keep arrivals fixed across all four routes.
- [ ] Require exact equal work, request conservation, SLO validity, and lower
      peak HBM before acquiring selected-GPU board energy.
- [ ] Attribute effects separately: `Q-C1` tests one-copy tail plus continuous
      coalescing; `Q-C2` tests heterogeneous-cut normalization; `Q-C0` tests the
      complete system.
- [ ] Report phone, USB, host-wall, and total-system energy as UNKNOWN.

## Stop rules

- Any cross-request KV mutation, stale acceptance, lost/duplicated terminal, or
  token-oracle failure stops the spike.
- A fixed cohort shrinking after EOG is not evidence of continuous admission.
- Batch 1/2 is allowed only for an imminent latest-start or insufficient
  compatible backlog; it is never forced merely to keep a device busy.
- CP5 cannot claim shared-tail capacity unless exactly one CUDA tail weight
  image is resident and independently measured.
- The focused energy mechanism fails on this hardware if no predeclared mixed
  load point reaches either 10 percent lower selected-GPU J/equal work or 10
  percent higher SLO-valid goodput at the same selected-GPU power boundary.
- No energy run begins before physical CP4 and CP5 mechanics pass.
