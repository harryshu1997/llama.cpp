# Two-phone gaps 1-3 (Stage A: Pixel static co-helper) - progress log

Newest first. No adb, no phone, no rig execution, no commits. Desktop reads read-only (campaign
inputs copied to `desktop-inputs/`, desktop plan gpu_first_layer read with ssh).

## 2026-09-24 C4 - rebased onto current main, final checks (DONE)

- Coordinator FYIs: the dispatcher (#2/#5) merge and the server-probe fix (coherence.py,
  adaptive_decode_state.py, test_adaptive_server_probe.py, ...) are in main. None of the 23 patched
  files had changed. base2 = fresh copy of main; root2 = base2 + patch (`patch -p1`, clean).
- Goldens re-computed on base2: identical to the first base.
- `test_two_phone_gaps` (29) + `test_two_phone_helpers` + adaptive runtime/coherence/server-probe/
  dispatch-policy: 157 tests OK (1 skip) on root2. Gap-2 dormant tests moved into their own class, so on
  base they fail on behaviour (dormant contract refused, residency subset False), not on imports.
- Full suite: base2 1723 tests, errors=2; root2 1752 tests, errors=2 (same two pre-existing errors:
  resident_router_subset native compile needs `examples/`, split_kv_attention relative import).
- `make_diff.sh` -> `TWO_PHONE_GAPS.diff` (23 files, 2827 lines); `git apply --check` passes against the
  live tree with stdin closed; applying it to copies of main's files reproduces root2 byte for byte.
- First-generation `base/`, `root/` deleted (superseded by `base2/`, `root2/`); experiment scripts in `scripts/`.
- `README.md` written (implemented vs designed, design, tests, gap-4 hook and wiring points, remaining
  gaps, updated smoke plan).

## 2026-09-24 C3 - implemented + tested on the first base (before rebase)

- Code (root vs base): new `_internal/plan_contracts/co_helpers.py`, `adapters/co_helper_lifecycle.py`
  (gap 4, designed hook), `tests/test_two_phone_gaps.py`, `tests/two_phone_harness.py`; edits in
  catalog materialization, capability-catalog validation, route patterns/envelopes/parameters, dormant
  whitelist + storage superset + attachment superset, residency match, ticket validation, launch
  contract, adaptive policy contract + planning, proofs, campaign catalog, preflight, 1 assertion in
  `test_two_phone_helpers.py` (2 new BLOCKED preflight rows).
- `test_two_phone_gaps.py`: 29 tests, all PASS on root. On base with only the test files copied in:
  25 ERROR/FAIL (every gap test), 4 PASS (single-phone regression guards: catalog/candidates/policies/
  plans/launch digests computed on base, policy hash, proofs, catalog without co-helper state).
- Full suite (first base, before my test file existed): base 1691 tests errors=2, root 1692 errors=3;
  the 3rd root error was a race (another agent updated `tests/test_prepare_trace_inputs_v2.py` in main
  between my two rsyncs); reset to the base copy. Remaining errors on both: `test_resident_router_subset`
  setUpClass (copy lacks `examples/`), `test_split_kv_attention` (package-relative import under discover).
- pyflakes clean on all 23 changed files.
- Coordinator FYI: dispatcher #2/#5 merged into main after my copy -> rebase before delivery.

## 2026-09-24 C2 - route generation experiment

Synthetic rig (8 layers, 128 columns; OP15 quantum 16, Pixel 32 owning CPU layers 4-5):
- operator_split candidate: device_ids (cpu, gpu, op15, pixel), FFN 0-3 on (cpu, op15), 4-5 on
  (cpu, pixel), union mask 63, `phone_helpers` {op15:15, pixel:48}, quantum 32 -> fractions
  0/25/50/75/100 %.
- adaptive policies 32/64/96/128 columns with per-device masks; adaptive-decode ticket launches
  HELPERS=2 (HELPER0 functionfs, HELPER1 tcp 127.0.0.1:26991); energy-aware selects the desktop parent
  whose dormant contract carries the union + `phone_helpers` -> same two-helper launch environment.

## 2026-09-24 C1 - read-only map + design decision

Request path read: catalog materialization (`adapters/catalog_materialization.py`), route patterns
(`_internal/route_generation/patterns.py`), envelopes/partition parameters (`envelopes.py`,
`costing_parameters.py`), dormant parent (`_unified/automated_selection_ops/{dormant,attachment}.py`,
`_unified/common.py`, `adapters/contracts.py`), residency match (`adapters/residency.py`), ticket
(`adapters/ticket.py`), launch (`adapters/llama_server_contracts.py`), adaptive planning
(`_internal/adaptive_decode_planning.py`, `adaptive_decode_contracts.py`), proofs
(`adapters/llama_server_ops/proofs.py`), campaign catalog (`campaigns/burstgpt/catalog.py`).

Design (Stage A, static co-helper):
- The Pixel is a real participant of the model's phone-assisted composites (own executor capability,
  own resources, own operator assignments in every plan), NOT a phone session of OP15: its shard
  never enters `phone_shards`, the OP15 residency layout, helper preparation or COW replacement.
- Declaration `phone_co_helpers_v1` (composite adapter parameter) carries primary label/serial + per
  co-helper device, serial, label, proof session id, static layer mask, column quantum, max tokens,
  shard sha/bytes, adb-tcp transport parameters.
- Route generation: co-helper FFN operators get (base, pixel, fraction) assignments; primary-phone
  envelopes exclude co-helper layers; envelope quantum = LCM(primary, co-helper quanta); decode-resident
  plans carry the union `ffn_resident_layer_mask` and `phone_helpers` (first = the ticket's phone).
- Dormant/residency/ticket whitelist and superset checks understand `phone_helpers`; adaptive policies
  gain optional per-device masks (hash unchanged when absent); proofs synthesize a co-helper shard and
  check helper request-id ranges.
- Gap 4: lifecycle hook only; gaps 5-6: fail-closed.

## 2026-09-24 C0 - setup

- base/ and root/ = `rsync -a --exclude __pycache__` of main `research_dev/scheduler`; the 27 GB
  `campaigns/burstgpt/reports` tree is a symlink to main's (read-only data, never edited) in both copies;
  `research_dev/spikes` and `gguf-py` symlinked read-only (tests need them).
- Read AGENTS.md, reports/20260924-two-phone-readiness/{README,PROGRESS}.md, M3 README (Pixel sections).
