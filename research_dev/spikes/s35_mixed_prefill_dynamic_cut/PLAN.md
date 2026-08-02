# S35: Mixed Prefill/Decode and Dynamic Layer Cut

## Scope

S35 closes two specific runtime gaps in the LayerSplit proof system:

1. Prove that a phone executes prompt rows for a newly admitted request and a
   decode-shaped row for an already-live request in one physical
   `llama_decode` batch.
2. Let a request select an active layer interval inside a worker's already
   resident layer interval, without reloading weights.

This is a proof runtime, not a production `llama-server` integration. Existing
StageNet V3 messages remain wire-compatible.

## CP1: Mixed batch gate

The frozen treatment is:

- seed request A at positions 0 through 3;
- execute one physical B=5 batch containing A at position 4 and request B at
  positions 0 through 3;
- compare every returned activation with a serial same-worker oracle;
- require finite values and relative L2 no greater than 0.005 per row;
- require active-sequence counts 1, 2, and 0 after seed, mixed batch, and
  removals;
- repeat on OP12 and OP15 with placement certification enabled.

The quality threshold is frozen before the device runs. A mechanics pass and a
numeric failure must be reported separately.

## CP2: Dynamic cut contract

- A worker loads a resident interval `[resident_start, resident_end)` once.
- Each request pins one active interval inside that resident interval.
- All rows in one physical batch use the same active interval.
- A head keeps `active_start == 0`; a terminal tail keeps
  `active_end == n_layer`.
- The active interval is part of graph reuse identity.
- A live sequence cannot change interval. Changing a cut requires removing the
  sequence and admitting a new route epoch.
- The scheduler batches only requests with the same active interval.

## Gates

- [x] CP1 unit tests pass.
- [x] CP1 OP12 physical gate passes.
- [x] CP1 OP15 physical gate passes.
- [x] CP2 host and Android builds pass.
- [x] CP2 rejects an interval outside resident weights.
- [x] CP2 rejects a cut change for a live sequence.
- [x] CP2 executes at least two distinct cuts without reloading the worker.

## Status

`MIXED_BATCH_AND_DYNAMIC_EXIT_MECHANICS_PASS`

See `RESULTS.md` for the measured scope and limitations.

