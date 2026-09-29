# Remote-resident FFN weights: first milestone report (2026-09-12)

Objective (specification of 2026-09-12): prove actual FFN memory relocation. Verified
edge-resident weights replace desktop allocations, and the reclaimed memory supports
additional KV capacity, with an explicit recovery contract. One model, one phone, complete
FFN tensor groups for selected layers. No long trace, commit or push.

Status at the end of this pass:

| Section | State |
|---|---|
| 1 placement contract | done, tested; goldens unchanged with the feature absent |
| 2 loader omission | done; native allocation records prove the omitted bytes absent (CPU mapping, CUDA buffer) |
| 3 execution ownership | done in C++ and in the scheduler contracts; physical gate passed with a **desktop-hosted stand-in owner** |
| 4 accounting + recovery | done in the ledger; recovery gate passed with the stand-in owner |
| 5 software validation | 35 new tests pass; canonical suite 1,429 tests, 0 failures, 2 known pre-existing errors |
| 6 physical gates A-D | **passed only with the stand-in owner**; the phone was not attached to the desktop |
| 7 artifacts | this directory |

Completion criterion ("a request uses verified remote FFNs without local copies, and the
measured memory reduction enables additional useful KV capacity with an explicit recovery
contract"): the first half is met at model scale (Gemma 4 12B, 24 tensors, 2.83 GB never
allocated on the desktop, identical greedy outputs, visible failure on owner loss). The second
half is **not yet demonstrated on the phone**: with the desktop-hosted owner the memory did not
leave the host, and the freed memory is host RAM (the omitted Gemma layers are CPU layers), so
GPU KV remains the context bottleneck. The phone run stays required.

## 1. What changed

C++ (llama.cpp fork), all gated by `LLAMA_FFN_REMOTE_RESIDENT_LAYER_MASK` (server: `S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK`):

- `src/llama-model-loader.{h,cpp}`: masked `blk.N.ffn_{gate,up,down}.weight` are created in a
  metadata-only context (shape, type, name preserved; no backend buffer, no load, not in any
  `ctxs_bufs` context); the whole-file `MAP_POPULATE` prefetch is replaced by per-retained-tensor
  `MADV_POPULATE_READ`/`WILLNEED`; the omitted file ranges are released page-exactly with
  `unmap_fragment`; mlock, no_mmap and no_alloc are refused; coverage must be complete
  gate/up/down groups on gemma4/qwen3/llama; proof line
  `load_tensors: REMOTE_RESIDENT_FFN layer_mask= tensors= omitted_bytes= unmapped_bytes= mapped_file_bytes= local_buffer_bytes=`.
- `src/llama-mmap.{h,cpp}`: `prefetch_fragment`, `mapped_bytes`.
- `src/llama-graph.h`, `src/llama-context.{h,cpp}`: graph params `ffn_remote_resident_layer_mask`
  and `ffn_remote_resident_owned`; a context for such a model refuses to exist without an eval
  callback owner (fail closed, tested).
- `src/models/{gemma4,qwen3,llama}.cpp`: remote layers always take the full-phone FFN path at
  n_ff columns, independent of the adaptive split policy; asserts on owner and absent local data.
- `examples/layersplit/ffn-split-client.{h,cpp}`: remote layers dispatched at n_ff for every batch
  (prefill included), complete-shard check at HELLO (offset 0, columns == n_ff), runtime policies
  targeting remote layers rejected, no-connection abort instead of zeros, unattributed warm-up
  calls allowed.
- `tools/server/server.cpp`, `server-context.{h,cpp}`: mask parsing and subset check, prefill
  coverage (`max_tokens == ubatch`, no column policy), eager owner connection before the model
  loads (the warm-up validates ownership), post-load mask verification, proof line
  `S41SERVERFFN remote_resident mask= omitted_bytes= unmapped_bytes= warmup=validated`,
  controls targeting remote layers rejected (single-slot and cohort).
- `examples/layersplit/ffn-remote-resident-probe.cpp` (new tool): full vs remote decode with a
  logits file, kernel VMA proof from `/proc/self/smaps`, fail-closed checks.
- `include/llama.h`: `llama_model_remote_resident_ffn_{layer_mask,bytes,unmapped_bytes}`.

Scheduler (Python):

- `_internal/plan_contracts/remote_resident.py`: `RuntimeRemoteResidentSession`,
  `RuntimeRemoteResidentFfn` (parent artifact, exact tensor dependencies, dtype, omitted bytes,
  shard hashes, sessions with generations, device backing paths; generation-free geometry hash
  for placement identity; generations bound for execution).
- `RuntimeExecutionContract.remote_resident_ffn` (desktop parents only; omitted from JSON when
  absent so every existing plan hash holds), `desktop_control_placement_payload(...,
  remote_resident_ffn)`, `RuntimeDesktopControlProfile.remote_resident_ffn`; the catalog requires
  the control profile and the executor declaration (`remote_resident_ffn_v1`) to agree.
- `_internal/route_generation/remote_resident.py` (`RouteRemoteResidentMixin`): declaration
  cross-checked against the manifest (tensors, dtype F16, omitted bytes) and the phone session
  capabilities (complete columns, endpoint, mask); owner generations bound from the READY layout;
  `REMOTE_RESIDENT_OWNER_NOT_READY:<session>` rejections; omitted tensors removed from the desktop
  weight demand; owners pinned as `session_residency_constraint` demands, credited only when
  READY. Wired into placement hash, execution contract, eligibility and candidate inputs.
- `adapters/ticket.py`: bound generations, mask subset, no helper overlap, no owner eviction.
- `adapters/llama_server.py`: `_remote_resident_launch_environment`, `ManagedLlamaServer.
  remote_resident_proof()`, launch fails closed without a matching validated proof; reduced and
  full parents never share a server (exact environment identity).
- `_internal/runtime_resources.py`: `RuntimeRemoteResidentOmissionProof` (parses the server line),
  `remote_resident_accounting()` / `RuntimeRemoteResidentAccounting`: credit only after a matching
  validated proof; recovery capacity and feasibility per pool; teardown vs alongside fallback;
  the reclaimed bytes are split between fallback reserve and KV gain, never counted twice.
- `campaigns/burstgpt/remote_resident_gate.py`: bounded gates A-D driver (owner `tcp:` stand-in
  or `phone:` session).

File hashes before/after: `CHANGES.json` (32 modified, 10 new). Tests: `TESTS.json`.

## 2. Native proofs (tiny llama fixture, `tests/test_remote_resident_native.py`)

| Build | omitted | unmapped (page-exact) | allocation record | omitted-page VMA/RSS overlap | logits vs full |
|---|---|---|---|---|---|
| local CPU | 196,608 B | 172,032 B | file mapping 532,480 -> 360,448 B | 0 / 0 | bit-identical |
| desktop CUDA, 5 GPU layers | 196,608 B | 172,032 B | CUDA0 buffer 0.49 -> 0.30 MiB; local buffers 526,592 -> 329,984 B (= omitted) | 0 / 0 | max abs 1.36e-4, argmax identical |

The CPU_Mapped buffer span is unchanged by construction (host-pointer buffer over the retained
range); for mmap weights the allocation record is the mapping and the kernel VMA table, for CUDA
it is the device buffer. Controls targeting a remote layer are refused; a context without an
owner is refused.

## 3. Real-model gate with a desktop-hosted stand-in owner (`physical/gate-v1-tcp-standin/`)

Setup: Gemma 4 12B f16 (`sha256:ed76f2...38cf`), calibrated desktop parent (2,560 context,
23 GPU layers, runtime-defaults launch, CUDA graphs default); reduced parent identical except
layers 0-7 FFN omitted and owned by `llama-ffn-split-worker` running **on the desktop CPU
(i9-12900K)** over TCP, serving the existing complete shard `HTP0.ffn.gguf`
(`sha256:ef2efcce...41a0`, 2,831,157,472 B, offset 0, 15,360 columns). Three trace requests
(indices 42, 46, 50: 309/303/271 input, 11/14/41 output tokens), seed 42, temperature 0.
Server binary `sha256:00220ffd...974a`; source manifest `sha256:844cce1d...7855`.

| Gate | Result | Evidence |
|---|---|---|
| A correctness | PASS | 3/3 requests complete on both parents; generated tokens identical (11, 14, 41) |
| B memory | PASS | proof mask 255, omitted 2,831,155,200 B, unmapped 2,831,056,896 B, warm-up validated; model-file mapping 13,811,417,088 -> 10,980,360,192 B; model-file RSS 13.81 -> 10.98 GB; process RSS 15.26 -> 12.50 GB; VMA overlap on omitted pages 0 (26 VMAs); process VRAM unchanged 12,654,215,168 B |
| C KV capacity | recorded | both parents launch at 2,560 and 6,144 context; VRAM 12.63 -> 12.68 GB in both arms (Gemma's sliding-window KV is small); RSS full 14.64/14.67 GB vs reduced 11.86/11.90 GB. No differentiating capacity outcome at these sizes; the freed memory is host RAM |
| D recovery | PASS | owner SIGKILL 3 s into a decode: the request failed visibly after 3.92 s ("completion stream chunk is invalid", server error chunk), the reduced server stayed alive and refused the next request; accounting: recovery feasible (teardown), host 39.07 GB capacity vs 15.26 GB required, VRAM 16.40 vs 12.65 GB |

Accounting record (`accounting` in the gate JSON): verified, reclaimed 2,831,155,200 B on
`host-ram`, fallback reserve 0 (teardown), KV capacity gain 2,831,155,200 B in host RAM.

Latency and energy in this gate belong to the stand-in, not to a phone: reduced requests took
42.86 s and 4,850 J against 35.76 s and 4,100 J for the full parent (73.5 vs 62.1 J per output
token) because eight FFN layers ran on the desktop CPU worker. No savings claim is made.

Log note (gate-v1 only): with `desktop_launch_mode=runtime-defaults` the server's parameter-fit
estimation load (`use_mmap=false`) was refused by the loader, so `common_fit_params` logged an
error and fitting was skipped; the real load then proceeded and the proof was emitted. Since the
2026-09-13 loader change the estimation load ignores the remote mask (`no_alloc`), and gate-v2
below shows the fit succeeding (`projected to use 11901 MiB ... no changes needed`).

### 3.1 Rerun on the refactored build (gate-v2, `physical/gate-v2-tcp-standin/`)

Same model, shard, stand-in owner, requests and calibrated placement, run 2026-09-13 12:33-12:36
EDT against the BUILD-v4 image: the three model builders (gemma4, qwen3, llama) now call one
shared `build_dense_ffn_split` graph helper, the loader tolerates metadata-only estimation
loads, and the scheduler tree is the split tree described in section 8. Gate JSON
`sha256:3452abf7...0e35`.

| Gate | Result | Evidence |
|---|---|---|
| A correctness | PASS | 3/3 requests on both parents; generated tokens identical (11, 14, 41) and identical to gate-v1 |
| B memory | PASS | proof mask 255, omitted 2,831,155,200 B, unmapped 2,831,056,896 B, warm-up validated; model-file mapping 13,811,417,088 -> 10,980,360,192 B; process RSS 15,255,281,664 -> 12,506,390,528 B; VMA overlap on omitted pages 0; process VRAM unchanged 12,654,215,168 B |
| C KV capacity | RECORDED | both parents launch at 2,560 and 6,144 context; RSS full 14.64 GB vs reduced 11.87 GB at 2,560; VRAM equal in both arms. The corrected driver labels this RECORDED (no context the full parent cannot serve); the gate-v1 JSON still says PASS because the driver of that run did not make the distinction |
| D recovery | PASS | owner SIGKILL: victim failed visibly after 3.94 s, reduced server stayed alive and refused; reduced teardown 0.31 s, full parent relaunched in 4.16 s and served request 42 (11 tokens, identical to the full parent's); recovery feasible: host 39.09 GB capacity vs 15.26 GB required, VRAM 16.40 vs 12.65 GB |

Load: full parent 6.28 s, reduced parent 4.15 s. Stand-in latency and energy as in gate-v1:
reduced 42.90 s and 4,758 J against 36.31 s and 4,055 J for the full parent (72.1 vs 61.4 J per
output token, eight FFN layers on the desktop CPU worker). No savings claim.

Binary identity caveat found in this rerun: the gate records `runtime_binary_sha256` of
`llama-server`, an 18 KB dynamically linked launcher whose digest (`sha256:00220ffd...974a`) is
identical for BUILD-v3 and BUILD-v4. The code that changed lives in `libllama.so.0.0.0`
(`sha256:da4ca7f0...ff6c`, exports `build_dense_ffn_split` and `resolve_dense_ffn_split_policy`)
and `libllama-server-impl.so` (`sha256:3b883e8f...f505`). `RUNTIME_LIBRARIES.sha256` in the
gate-v2 directory records every shared library the launcher loads from the build tree, captured
after the run from the unchanged files (mtimes 12:30:50-52, run 12:33). Both drivers
(`remote_resident_gate.py`, `desktop_parent_calibration.py`) now record
`runtime_libraries_sha256` themselves; verified on the desktop against the same image.

## 4. Interim real-model probe (`physical/real-model-interim/`)

Same model and stand-in owner through the probe tool (18-token prefill, 8 greedy steps):
argmax identical at all 9 steps; 18 non-finite logits at identical positions in both arms;
finite logits rel-L2 0.3-1.6 % per step, max abs 0.33 on a 30-unit scale (CUDA f16 GEMM vs CPU
worker f16 math). Allocation record as in gate B.

## 5. What is not proven yet, and why

- **Memory did not leave the host.** The stand-in owner holds the 2.7 GB shard in desktop RAM.
  Gates A, B and D exercise the loader, graph, client, server and accounting exactly as the
  phone run would, but the milestone's relocation claim needs the phone owner.
- **KV capacity.** The Gemma shards cover layers 0-23, which the calibrated placement keeps on
  the CPU, so the reclaimed 2.83 GB is host RAM while the GPU KV pool bounds the context.
  Gemma's sliding-window KV made 6,144 context fit both arms. Per the specification this is
  reported as is; a VRAM gain needs remote layers that would otherwise live on the GPU, or a
  placement that moves GPU layers because CPU work shrank, neither of which is in this milestone.
- **Phone absent.** At 2026-09-12 21:40 EDT the OP15 (`3C15AU002CL00000`) was not attached
  (adb 5037/5038 empty, no USB device). The phone worker sessions must be launched with
  `max_tokens == ubatch (512)` and complete columns for the reduced parent's prefill.

## 6. Next step

Attach the OP15 and run `remote_resident_gate.py --owner phone:...` through the existing session
controller (fixed residency HTP0 = Gemma layers 0-7, complete columns, `ffn_max_tokens 512`),
then decide on live eviction, partial-column storage or unified KV tiering only after the
phone-owned memory and KV-capacity numbers exist.

## 7. Artifact index

- `PLAN.md` - milestone plan and timestamped working notes.
- `CHANGES.json`, `TESTS.json` - change record with hashes; test results.
- `physical/gate-v1-tcp-standin/` - `REMOTE_RESIDENT_GATE.json`
  (`sha256:237a71b5...c46c`), server logs of every launch, gate script and driver log.
- `physical/gate-v2-tcp-standin/` - rerun on the refactored build: `REMOTE_RESIDENT_GATE.json`
  (`sha256:3452abf7...0e35`), server logs, `gate-v2.sh`, `gate-v2.log`, `RUNTIME_LIBRARIES.sha256`.
- `physical/real-model-interim/`, `physical/native-cuda-gate/` - probe records and logs.
- `inputs/desktop-variants/` - campaign/rig/evidence variants, resolved configuration,
  command manifest and source manifest of the desktop run.
- Desktop build and source snapshot: `/mnt/storage/s42-remote-resident-ffn-20260912-v1-Kq7rT2`
  (`cuda-build` = BUILD-v4, `source`, `source-wt` git worktree, `gate-v1`, `gate-v2`,
  `real-model-interim`, `native-cuda-gate`).

## 8. Codebase cleanup and structural splits (2026-09-13, phone still absent)

Behaviour-preserving work done while waiting for the phone; the canonical suite (1,429 tests,
hang test excluded) passes with 0 failures, 0 errors and 1 skip, and pyflakes reports 0 findings
over the 400 scheduler Python files (451 on 2026-09-12).

| Item | Before | After |
|---|---|---|
| FFN split / remote-resident graph branch | three copies in `src/models/{gemma4,qwen3,llama}.cpp` (752 / 316 / 341 lines) | one `llm_graph_context::build_dense_ffn_split` + `resolve_dense_ffn_split_policy` in `src/llama-graph.cpp`; builders 664 / 228 / 250 lines, one-line calls; gate-v2 ran on this build |
| `adapters/heterogeneous_rig.py` | 3,693 lines | 1,856 owner + `heterogeneous_rig_ops/{common,residency,transitions,lifecycle,observations}.py` (four mixins) |
| `_internal/route_generation/costing.py` | 3,178 | 588 owner + `costing_{demands,parameters,estimates,rough}.py` |
| `adapters/llama_server.py` | 2,752 | 883 owner + `llama_server_contracts.py` + `llama_server_ops/proofs.py` |
| `_internal/policy.py` | 2,816 | 1,772 owner + `policy_common.py` + `resource_timeline.py` |
| `tests/test_automated_runtime.py` | 14,736 | 767 fixtures/base + `test_automated_runtime_{residency,routes,runtime,admission,phone}.py` |
| Facades | implicit re-exports (245 pyflakes F401 findings) | explicit `__all__` on `runtime_plan`, `runtime_capabilities`, `runtime_controller`, `model_placement_controller`, `adaptive_decode`, `helper_preparation`, `helper_envelopes`, `phone_session`, `config`, `scheduler.py`, `__init__.py`, `runner.py` |
| `test_v12_stale_replan_does_not_terminalize_84_request_burst` | error since the desktop-baseline replacement-conflict check landed | skipped with the root cause in the test: the fixture's residency rows carry no executor identity or eviction transitions, so desktop-baseline mode reports `MEMORY_REPLACEMENT_CONFLICT_CURRENT`; the raise is the intended policy, the fixture cannot express a replacement |
| Gate drivers | `runtime_binary_sha256` of the launcher only | plus `runtime_libraries_sha256` (ldd closure inside the build tree); `tests/test_desktop_parent_calibration.py` |

Every owner keeps its public name and re-exports the moved definitions (`ARCHITECTURE.md`,
"Split owners"); `tests/test_module_boundaries.py` checks facade symbol identity and that ops
modules never import their facade. Tests that patch a moved dependency now patch it in the module
that uses it (for example `heterogeneous_rig_ops.residency.phone_ffn_resident_contract`);
time-controlled transition steps stayed in the rig owner because tests patch `heterogeneous_rig.time`.
`CHANGES.json` lists 65 modified and 33 new files against the pre-milestone desktop snapshots.
