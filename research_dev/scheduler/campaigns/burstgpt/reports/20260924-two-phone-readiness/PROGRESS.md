# Two-phone (OP15 FunctionFS + Pixel 10 Pro adb-tcp) preparation - progress log

Newest first. No adb, no phone, no rig execution, no commits. Desktop reads are read-only ssh.

## 2026-09-24 M4 - rebase onto the live tree, design/gap list, smoke plan (DONE)

- The coordinator merged phone re-provisioning into main (config.py, campaigns/burstgpt/{arguments,launch,runner}.py,
  _unified/phone_residency*, helper_preparation_ops/start.py, runtime_requests.py, configuration/campaign.py,
  scheduler.py, 2 tests). Of the files this work edits only `config.py` and `launch.py` changed; both 3-way merged
  cleanly (`git merge-file`), and `base/`+`root/` now mirror the live tree plus the patch.
- `make_diffs.sh` regenerates `TWO_PHONE.diff` / `TWO_PHONE_SERVER.diff` against the live tree; both pass
  `git apply --check` (stdin closed), and applying them to a copy of the live files reproduces `root/` exactly.
- Rebased root: full scheduler suite 1717 tests, 1 error = pre-existing `test_split_kv_attention` relative import
  under discover (passes as a module); re-provisioning tests pass with the patch (68 tests incl. the two-phone ones).
- README.md: implemented vs designed, native design, scheduler contracts, per-helper identity + Pixel receipt
  inventory, USB bus sharing (read-only lsusb/sysfs), Stage A/B dispatch design incl. how desktop-follow
  re-provisioning generalizes per device, ranked gap list, 5-step smoke plan, risks.
- Read-only map of single-phone assumptions on the request path gathered for the gap list (catalog, plan contracts,
  helper preparation, residency, rig adapters, proofs, identity).

## 2026-09-24 M3 - scheduler side (IMPLEMENTED where testable; dispatch integration DESIGNED only)

Replaced the stopped agent's invasive rig change and dead wiring (kept in `agent-prev/`). Implemented:

| Piece | File | Status |
| --- | --- | --- |
| additive `helper_phones` (rig) + `topology.helper_phones`; legacy `phone` untouched, legacy JSON round-trips byte for byte | configuration/rig.py | implemented + tested |
| per-model `helper_phone_ffn_shards` (device -> index, phone dir) + loader cross-check | configuration/models.py, config.py | implemented + tested |
| `adb-tcp` transport contract (direct worker behind adb forward; `tcp` keeps its bridge meaning) | adapters/phone_transport.py | implemented + tested |
| bindings, disjoint ownership, HELPER<k> env, per-device call accounting by owner + id range, per-helper transport identity + missing receipts, sysfs USB topology | adapters/phone_helpers.py (new) | implemented + tested |
| adb-tcp worker lifecycle: preflight (occupancy, sha256 pins, boot id), start (ready + forward), finite-budget drain stop (no signal), explicit idle-only signal for resident workers | adapters/phone_tcp_session.py (new) | implemented; tested against a real local CPU worker through a fake adb |
| `phone_helpers` ticket parameter -> multi-helper server env | adapters/llama_server_contracts.py (+6 lines) | implemented + tested with the real launch-contract builder |
| preflight: helper USB rows + BLOCKED `two-phone-dispatch`; launch: real run with helper phones refused | campaigns/burstgpt/{preflight,launch}.py (small hunks) | implemented + tested |
| M3 mechanism gate: wraps the M0/M2 decode-relocation gate, adds the helper worker, union mask, PIXEL0 proof shard, drain, TWO_PHONE_RESULT.json | campaigns/burstgpt/two_phone_gate.py (new) | implemented; mechanics tested with fakes; NOT run |

`tests/test_two_phone_helpers.py`: 28 tests PASS on root (27 at this milestone, +1 drain test later); on base the module fails at import.

Full scheduler suite (`unittest discover -s research_dev/scheduler/tests`, `S42_LLAMA_BUILD_BIN=build-root/bin`):
root 1676 tests, base 1632 tests. Both show the same 14 errors, all missing `research_dev/spikes` data in the
scratch copy plus the pre-existing package-relative import of `test_split_kv_attention` under discover; base also
had a timing flake (`test_cached_synthetic_refinement_is_below_ten_milliseconds`). With `research_dev/spikes`
symlinked read-only, the 7 affected modules (90 tests) PASS on root. No regression introduced.

## 2026-09-24 M2 - native server: one llama-server, N FFN helper clients (IMPLEMENTED, built, locally tested)

Rewrote the runtime from the original `server.cpp` with minimal hunks (the previous agent's version is kept as
`server.cpp.agentversion`). Fixes over it: rollback restored the NEW sub-policy (applied_* were overwritten
before a later helper failed) -> now saved values are restored; nested `helpers` stats removed; at most one
functionfs-usb helper; legacy HOST/TRANSPORT/PORT next to HELPERS rejected; duplicate labels rejected; static
table/shape policies allowed with several helpers and validated against every helper; geometry check at init.

Local build: `build-base/` (main tree) and `build-root/` (`src/` = main + patch), CPU, `S41_SERVER_FFN_SPLIT=ON`,
`-DLLAMA_ALL_WARNINGS=ON`: no warnings in server.cpp / ffn-split-client.cpp.

New native test `tests/test_two_phone_server_native.py` (tiny llama, local CPU `llama-ffn-split-worker`s over TCP):

| Test | root build | base build |
| --- | --- | --- |
| two helpers (A: layers 0-1, B: 2-3) vs one legacy helper (layers 0-3), static policy 16:128: tokens identical, per-layer calls identical, B ids start at 2^24+1, one summary per helper | PASS | FAIL (HELPERS ignored, no calls) |
| runtime control, deferred helpers, quanta 64/128: 64 cols refused with `helper pixel:` and no 64-col call ever made (rollback), 128 applied, sub-mask 0b0110 applied | PASS | FAIL |
| startup rejects overlap, uncovered, legacy+HELPERS, two functionfs helpers, duplicate label | PASS | FAIL (server starts) |

Regression: `test_remote_resident_native.py` 17/17 PASS against `build-root` (legacy single-client path).

## 2026-09-24 M1 - review of the stopped agent's partial work (base vs root)

`base/` equals the current main tree for all 601 copied files (checked with cmp at start).
`diff -ru -x __pycache__ base root` = 2301 lines over 15 files + 3 new files.

| Piece | Verdict | Action |
| --- | --- | --- |
| `examples/layersplit/ffn-split-client.{h,cpp}` `first_request_id` | sound, 3 lines | keep |
| `tools/server/server.cpp` multi-helper `s41_server_ffn_runtime` (HELPERS=N, HELPER<k>_* env, owner-by-layer eval routing, union mask + owned sub-masks, rollback, per-helper summaries) | design sound; never compiled | keep, fix: (1) nested `helpers` array added to the control-ack runtime stats would be rejected by `http_backend._runtime_stats` (strict int whitelist) -> remove; (2) no guard against two functionfs-usb helpers although the host USB client opens the first 18d1:2d00 device (`libusb_open_device_with_vid_pid`) -> add; (3) static table/shape policies forbidden for >1 helper although each client reads the same env and serves its own mask -> allow, validate every helper; (4) build + native test |
| `configuration/rig.py` (`phones` list, made legacy `whole_*`/boot fields Optional, forced adb-tcp phones to carry FunctionFS dummy fields) | invasive: changes the typed primary-phone contract used everywhere as `rig.phone` | replace by an additive `helper_phones` list with its own small dataclass; legacy `phone` untouched |
| `adapters/phone_transport.py` adb-tcp contract | sound | keep, small fixes |
| `adapters/phone_helpers.py` (bindings, disjoint ownership, env composition, call attribution, per-helper identity, sysfs USB topology) | sound core | keep, trim, adapt to rig change |
| `adapters/phone_tcp_session.py` (adb-forward worker lifecycle, injectable) | mostly sound | fix colon-separated library path handling; stop semantics documented as a policy decision (SIGTERM of an idle worker vs finite budget) |
| `adapters/llama_server_contracts.py` hook (`phone_helpers` parameter -> multi-helper env) | sound, small | keep |
| `campaigns/burstgpt/runner.py` | **syntax error** (import inserted inside a parenthesised import); builds helper sessions although no dispatch path uses them | drop runner/heterogeneous_rig/transitions wiring (dead code); keep preflight fail-closed check |
| `heterogeneous_rig*.py` `_prepare_adb_tcp_phone` | unreachable (no route produces an adb-tcp transition) | drop |
| `tests/test_two_phone_helpers.py` | 27 tests, 2 failing | rewrite for the revised rig format |

Other findings while reviewing (read-only):
- Pixel transport = protocol-v6 TCP worker behind `adb -P 5037 -s 5A040DLCH004ES forward tcp:<host> tcp:<phone>`,
  qualified launch: `LD_LIBRARY_PATH=<gemv-tune-confirm>:<ffn-coalesced> S42_PIXEL_F16_{WG=128,ROWS=8,SUBGROUP=128}
  llama-ffn-split-worker -m .../HTP0.ffn.gguf --layers 18-23 --columns 17408 --column-quantum 4352 --backend Vulkan0
  --port P --bind 127.0.0.1 --f16-io --max-tokens 4 --max-requests N` (physical/pixel10pro-gemv-server-1/run1).
- OP15 Qwen HELLO quantum 2176, Pixel 4352 -> union policy widths must be multiples of 4352 (25/50/75/100 %).
- USB (read-only sysfs on the desktop): OP15 2-2 (22d9:2772, 5000M, root port 2), Pixel 2-9.2 (18d1:4ee7, 5000M,
  behind ASM107x hub 174c:3074 on root port 9); both on the one Alder Lake-S xHCI 0000:00:14.0.
- `S41SERVERFFNCALL` lines carry no helper label; per-device accounting is by layer owner + distinct request-id
  namespaces (helper k starts at 1 + k*2^24).
