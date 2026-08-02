# S22 SLO overlap pipeline

## Scope

Build the smallest executable proof for the request-level design:

1. Keep model weights and per-request KV resident on each device.
2. Admit and retire sequences independently.
3. Form device-local batches from ready rows without a global barrier.
4. Pin a request to one profiled route after prefill.
5. Join routes at common layer boundaries on the desktop.

The proof route set is finite:

- R0: 4060 Ti `[0,8)` -> 4060 Ti `[8,48)` (next control route)
- R1: OP15 `[0,8)` -> 4060 Ti `[8,48)`
- R2: OP12 `[0,8)` -> 4060 Ti `[8,48)`

R1 and R2 deliberately duplicate the first eight layers. This gives the
request-level scheduler a route choice without changing a request's route after
prefill. A later chained phone route requires an eligible `[8,k)` OP15 island;
it is not implied by the current prefix workers.

## Checkpoints

- [x] CP0: audit existing static batch, layer-window, and transport code.
- [x] CP1: add an opt-in StageNet V3 protocol with request identity, route
  epoch, arbitrary live-sequence batches, per-sequence removal, status, and
  drain.
- [x] CP2: run the V3 lifecycle gate on both phones with Gemma-4-12B.
- [x] CP3: add asynchronous bounded join queues and a continuously batched
  desktop tail.
- [x] CP4: add the finite R0/R1/R2 route scheduler using offline profiles.
- [x] CP5: run the end-to-end three-device mixed-SLO mechanics proof.
- [x] CP6: screen multi-token chunked prefill against sequential prefill.
- [x] CP7: localize chunked-prefill differences at the layer-8 boundary and
  compare against an unsplit full-model batch-shape control.
- [ ] CP8: run the boundary screen on both phone HTP workers and a real-prompt
  output-quality set before admitting chunked prefill.

## Fail-closed invariants

- A sequence slot is bound to `(request_id, route_epoch)` until removal.
- Positions for a live sequence are contiguous and cannot repeat.
- A new sequence starts at position zero.
- A draining worker admits no new batch.
- Sequence removal clears only that sequence's KV.
- A response must echo the complete request lineage before it is accepted.
- Legacy StageNet commands cannot run while V3 sequences are live.

## CP3 physical state

OP15, OP12, and the RTX 4060 Ti desktop are reachable. The current-source B4
run passes with OP15 and OP12 executing `[0,8)` on HTP0 and the desktop
executing one shared `[8,48)` Q8 tail on CUDA0. Each phone reforms four ready
rows into B4 on every step. The tail receives the faster and slower phone
cohorts without a global barrier and forms B4 at their independent cadences.

This is continuous ready-row rebatching in the LayerSplit proof runtime, not an
integration with llama-server. CP4/CP5 add a CUDA-head control and mixed-SLO
selection. CP6 proves that chunked prefill executes as B8 on both phones. CP7
shows that exact greedy-token agreement across batch shapes is too strict: an
unsplit full-F16 model also changes one token on the frozen synthetic prompt.
Chunked phone prefill remains ineligible until CP8 measures its boundary error
and output quality. A chained phone route, arbitrary runtime cuts, and energy
savings are still open. See `RESULTS.md`.
