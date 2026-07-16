# S9 golden artifacts

Frozen bundles support the repair rounds as historical regression evidence.

## `v0_historical/` -- the "before" side

The pre-repair (bundle version 1) simulator and its output, frozen byte-for-byte,
plus the provenance hashes and the relabeled V0 status
(`SUITES PASS; MECHANICS NOT YET CERTIFIED; CAPACITY UNPROVEN`). See
`v0_historical/MANIFEST.txt`.

The original V0 input config was not preserved. Tests therefore verify the frozen
module hash and stored-manifest self-consistency only; they do not claim an executable
V0 replay.

## `v0r/` -- the "after" side

`baseline_sweep.v0r.manifest.json` is the golden output of the REPAIRED simulator for
`sim/scenarios/baseline_sweep.config.json`. `sim/test_golden_replay.py` recomputes the
run in a fresh process and asserts byte-identity against this stored file (not against
a second in-process run), and a mutation of the stored file must make that test FAIL
(the golden-replay-mutation adversarial case).

Regenerate the V0-R golden only via `python3 sim/make_golden.py` and only after a
deliberate, reviewed mechanics change; never hand-edit it.
