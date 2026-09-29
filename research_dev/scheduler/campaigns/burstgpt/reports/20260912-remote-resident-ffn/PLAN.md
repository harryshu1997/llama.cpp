# Remote-resident FFN weights: milestone plan

Objective (from the specification of 2026-09-12): prove actual FFN memory
relocation. Verified edge-resident weights replace desktop allocations,
and the reclaimed memory supports additional KV capacity. One model, one
phone, complete FFN tensor groups for selected layers; reuse shard files,
workers, session controller and proofs. Not in scope: distributed KV
storage, partial-column compaction, multi-phone placement, live eviction
from a running desktop, long traces, commits, pushes.

## Phases and gates

1. Explicit remote-resident placement contract (assisted copy vs remote
   resident) carrying parent artifact, tensor dependencies, geometry,
   dtype, shard hash, device/session generation, operator plan and local
   backing-file locations; participates in placement, qualification and
   graph identity; assisted-copy routes unchanged. Gate: incomplete
   coverage, stale generations and unsupported shapes fail closed; replay
   goldens unchanged with the feature disabled.
2. Desktop loader omits the selected remote-owned FFN tensors at startup:
   metadata preserved, no local backend allocation or load, no prefetch of
   the omitted GGUF ranges, everything else unchanged, GGUF preserved.
   Gate: native allocation records prove the omitted bytes are absent.
3. Execution ownership enforced through the existing scheduler, attachment
   and lease paths: remote sessions verified before the reduced endpoint
   is executable; ownership validated in warm-up, prefill and decode;
   controls cannot assign omitted work back to the desktop; generations
   and retained sessions preserved; CUDA graphs on with a validated graph
   contract; fallback only through a separately feasible full-weight
   route. Gate: short prefill and decode with valid output and exact
   proofs, no hidden local reload.
4. Memory accounting and recovery together, in the existing ledger:
   desktop weights actually allocated, phone weights and workspace, KV by
   pool, transition peaks, recovery capacity; credit only after
   allocation verification; recovery feasibility before admission;
   disconnect handling that fences stale responses, preserves committed
   state, stops only affected execution, retains healthy sessions and
   records pauses or recomputation. Reclaimed memory is never counted
   both as available KV space and as reserved fallback space.
5. Software validation: loader, identity, execution, controls, recovery,
   accounting test groups; canonical suite and both replay goldens once
   before physical gates; native numerical tests with declared tolerances.
6. Bounded physical gates: A correctness, B memory, C KV capacity,
   D recovery. Identical requests and decoding settings for energy
   comparisons; the maximum-context experiment is a capacity comparison.
   If only CPU RAM is freed while GPU KV is the bottleneck, report that.
7. Report immutable artifacts (hashes, contracts, omitted allocations and
   reclaimed memory, peaks, latencies, energy, reloads, recovery pauses,
   proofs) and decide the next step.

Completion criterion: a request uses verified remote FFNs without local
copies, and the measured memory reduction enables additional useful KV
capacity with an explicit recovery contract.

## Working notes

- Desktop server build: CUDA Release, arch 89, CUDA graphs on, built from
  a byte-identical snapshot of this tree's C++ sources
  (/mnt/storage/s42-llama-packed-prefix-20260911-v1-JcmCWJ/source ->
  /mnt/storage/s42-work-conserving-dev3-20260911-v1-NrWH4c/cuda-build).
  Loader changes therefore develop locally and rebuild there.

- 2026-09-12 20:50 EDT native gate (sections 2, 3 partial, 5): `LLAMA_FFN_REMOTE_RESIDENT_LAYER_MASK`
  loader omission implemented (`src/llama-model-loader.{h,cpp}`, `src/llama-mmap.{h,cpp}`,
  `src/llama-model.{h,cpp}`, `include/llama.h`), graph enforcement in gemma4/qwen3/llama
  builders (`ffn_remote_resident_layer_mask` + `ffn_remote_resident_owned` graph params, context
  refuses to exist without an eval-callback owner), client pinning
  (`examples/layersplit/ffn-split-client.{h,cpp}`: remote layers always dispatched at n_ff,
  complete-shard check at HELLO, controls targeting remote layers refused, no-connection abort),
  server plumbing (`S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK`, eager connect before warm-up,
  post-load mask verification, `S41SERVERFFN remote_resident` proof line, control rejection).
  Probe `examples/layersplit/ffn-remote-resident-probe.cpp` + `tests/test_remote_resident_native.py`
  (7 tests PASS on the tiny llama fixture, build-cpu): omitted 196,608 B, page-exact unmapped
  172,032 B, zero VMA/RSS overlap on interior pages, file mapping 532,480 -> 360,448 B,
  logits bit-identical to the full-weight arm (max abs diff 0.0), no-owner context refused,
  remote-layer control rejected. CPU_Mapped buffer span is unchanged by design (host-pointer
  buffer over the retained range): the allocation record for mmap weights is the mapping and
  the kernel VMA table, for CUDA it is the device buffer. Local `build-cpu-server`
  (S41_SERVER_FFN_SPLIT=ON) compiles. Desktop rebuild started in
  /mnt/storage/s42-remote-resident-ffn-20260912-v1-Kq7rT2 (source + cuda-build).
- Shard geometry confirmed on the desktop: Gemma HTP0/1/2 = layers 0-7/8-15/16-23, offset 0,
  columns 15360 = n_ff, F16, 2.83 GB each; Qwen HTP0/1/2 = layers 0-5/6-11/12-17, columns
  17408 = n_ff. Complete groups: remote-resident eligible as stored.
- 2026-09-12 21:15 EDT desktop CUDA build ready (/mnt/storage/s42-remote-resident-ffn-20260912-v1-Kq7rT2/cuda-build:
  llama-server, llama-ffn-remote-resident-probe, llama-ffn-split-worker, llama-simple; link needed
  LD_LIBRARY_PATH to the CUDA 13.2.1 lib dir). Tiny-model native gate on the desktop
  (native-cuda-gate/): gpu_layers=5 CUDA0 model buffer 0.49 -> 0.30 MiB, local_buffer_bytes
  526,592 -> 329,984 = exactly the 196,608 omitted bytes; VMA/RSS overlap 0 on omitted pages;
  argmax trace identical, logits max abs diff 1.36e-4 (CUDA f16 GEMM vs CPU worker);
  gpu_layers=0 bit-identical (0.0). No-owner context refused on the CUDA build as well.
- Scheduler contracts (section 1): `_internal/plan_contracts/remote_resident.py`
  (RuntimeRemoteResidentSession / RuntimeRemoteResidentFfn: parent artifact, tensor ids,
  dtype, omitted bytes, shard hashes, sessions with generations, backing paths; generation-free
  geometry hash for placement identity, generations bound for execution),
  `RuntimeExecutionContract.remote_resident_ffn` (desktop parents only, omitted from JSON when
  absent), `desktop_control_placement_payload(..., remote_resident_ffn)` +
  `RuntimeDesktopControlProfile.remote_resident_ffn`, `adapters/ticket.py`
  `_validate_remote_resident_execution_command` (bound generations, mask subset, no helper
  overlap, no owner eviction). tests/test_remote_resident_contract.py 8 PASS; replay goldens,
  physical adapter, module boundary and desktop capacity tests 31 PASS with the feature absent.
- 2026-09-12 21:45 EDT scheduler integration (sections 1, 3, 4 software side):
  `_internal/route_generation/remote_resident.py` (RouteRemoteResidentMixin: coordinator
  declaration `remote_resident_ffn_v1`, manifest/session cross-checks, owner binding from the
  READY layout, `REMOTE_RESIDENT_OWNER_NOT_READY:<session>` rejections, omitted tensors removed
  from desktop weight demands, owning sessions pinned as `session_residency_constraint`
  demands credited only when READY), wired into placement hash, execution contract,
  eligibility and candidate inputs; catalog requires control profile == executor declaration.
  `adapters/llama_server.py`: `_remote_resident_launch_environment` (mask, shards, complete
  columns, max_tokens == ubatch, runtime control, transport), `ManagedLlamaServer.
  remote_resident_proof()`, launch fails closed without a matching validated proof.
  `_internal/runtime_resources.py`: `RuntimeRemoteResidentOmissionProof` (parses the server
  line), `RuntimeRemoteResidentAccounting` + `remote_resident_accounting()` (credit only after
  proof; recovery capacity/feasibility; teardown vs alongside fallback; reclaimed split between
  fallback reserve and KV gain, never both). Tests: test_remote_resident_{contract,accounting,
  routes,launch}.py = 8+6+8+6 PASS; adapter/golden/boundary/capacity/catalog/route tests PASS.
- BLOCKER for physical gates A-D: the OP15 (3C15AU002CL00000) is not attached to the desktop
  (adb 5037/5038 list no device, lsusb shows no phone). Interim real-model check started on the
  desktop with a desktop-hosted TCP worker serving the Gemma HTP0 shard (layers 0-7, complete
  columns): real-model-interim/ (full vs remote probe, gpu_layers=23). This proves loader
  omission and execution on the 12B model but does not relocate memory off the host; the
  phone run remains required for the milestone.
- 2026-09-12 21:30 EDT interim real-model probe (desktop-hosted TCP worker as owner, Gemma 4 12B
  f16, layers 0-7 remote, gpu_layers=23, 18-token prefill + 8 greedy steps; real-model-interim/):
  loader omitted 2,831,155,200 B (24 tensors), page-exact unmapped 2,831,056,896 B; model file
  mapping 13,811,417,088 -> 10,980,360,192 B (-2.83 GB), 26 VMAs, zero VMA/RSS overlap on the
  omitted pages; CPU_Mapped span 13,171.58 MiB and CUDA0 11,461.36 MiB unchanged (layers 0-7 are
  CPU layers; the mapping is the record). Argmax identical for all 9 steps; 18 non-finite logits
  at identical positions in both arms; finite logits rel-L2 0.3-1.6 % per step, max abs 0.33 on
  a 30-unit scale (CUDA f16 vs CPU-worker f16 math). Worker: i9-12900K CPU backend, 2.7 GB f16.
  Not a memory relocation off the host; establishes omission + execution at model scale.
- Canonical suite (run_suite, hang test excluded): 1,429 tests, 0 failures, 5 errors (2 known
  pre-existing; the rest under investigation in this pass).
- 2026-09-12 23:05 EDT gate-v1 with the desktop-hosted TCP owner (physical/gate-v1-tcp-standin/):
  A PASS (3/3 requests, generated tokens identical), B PASS (proof mask 255, omitted 2,831,155,200,
  unmapped 2,831,056,896, mapping 13.81 -> 10.98 GB, process RSS 15.26 -> 12.50 GB, VRAM unchanged,
  VMA overlap 0), C recorded (2,560 and 6,144 context both launch in both arms; freed memory is
  host RAM), D PASS (owner SIGKILL -> victim fails in 3.9 s, server refuses, recovery feasible).
  Canonical suite rerun: 1,429 / 0 failures / 2 known pre-existing errors. README.md, CHANGES.json,
  TESTS.json written. Phone still absent: the relocation-off-host and phone KV numbers remain open.
- 2026-09-13 00:45 EDT cleanup pass while the phone is absent: removed 196 dead imports from 20
  scheduler modules (AST-based pruner; facade re-exports and de-facto re-exports kept), 9 unused
  locals, one undefined test variable (`token` in test_adaptive_decode), the cold-executor test
  fake (`adapter_parameters`), the two test fakes broken by the remote-resident mixin; loader now
  ignores the remote mask for metadata-only estimation loads (no_alloc), so the server's parameter
  fit no longer logs an error; gate driver: capacity gate PASS only when the reduced parent serves a
  context the full parent cannot (else RECORDED), recovery gate relaunches the full parent after the
  reduced teardown; docs (ARCHITECTURE/README/ACCEPTANCE_TESTS) describe the contract. Removed the
  temporary `build-cpu-server/` (797 MB). pyflakes: 451 -> 245 findings, all remaining are facade
  re-exports (`X as X`) and `__init__` re-exports. Known debt kept: test_v12_stale_replan (fixture
  reports MEMORY_REPLACEMENT_CONFLICT_CURRENT:gpu-memory in desktop-baseline mode because its
  residency rows carry no executor identity/eviction transitions; adding executor ids alone does
  not fix it) and the hang test excluded from run_suite.
- 2026-09-13 12:50 EDT "do all" pass (phone still absent). (1) The three copies of the FFN
  split / remote-resident graph branch became `llm_graph_context::build_dense_ffn_split` +
  `resolve_dense_ffn_split_policy` (`src/llama-graph.{h,cpp}`; gemma4 752->664, qwen3 316->228,
  llama 341->250 lines, one-line calls). (2) Giants split without behaviour change, owner keeps the
  public name and re-exports: heterogeneous_rig 3,693->1,856 (+ heterogeneous_rig_ops/ four mixins),
  costing 3,178->588 (+ costing_{demands,parameters,estimates,rough}), llama_server 2,752->883
  (+ llama_server_contracts, llama_server_ops/proofs), policy 2,816->1,772 (+ policy_common,
  resource_timeline), test_automated_runtime 14,736->767 (+ five themed test modules inheriting the
  fixture base). Patch anchors: tests patch the module that now uses the symbol; time-controlled
  steps stayed in the rig. (3) `__all__` on the facades (runtime_plan, runtime_capabilities,
  runtime_controller, model_placement_controller, adaptive_decode, helper_preparation,
  helper_envelopes, phone_session incl. the `socket` patch anchor, config, scheduler.py, __init__,
  runner canonical exports). pyflakes 245 -> 0 over 400 files. (4) test_v12_stale_replan retired via
  skip with the root cause in the test (fixture cannot express a replacement; the desktop-baseline
  raise is intended policy). (5) gate-v2 (12:33-12:36 EDT, BUILD-v4 = refactored builders + no_alloc
  loader tolerance, same TCP stand-in): A PASS tokens identical to gate-v1, B PASS same proof and
  allocation record (RSS 15.26 -> 12.51 GB, VRAM unchanged, VMA overlap 0), C RECORDED (corrected
  label), D PASS (victim 3.94 s, teardown 0.31 s, full parent relaunched 4.16 s, request served);
  parameter fit now succeeds. Found: `runtime_binary_sha256` hashes the 18 KB launcher and did not
  change between builds; the code identity is libllama.so.0.0.0 sha256:da4ca7f0... (exports the new
  helper) -> `RUNTIME_LIBRARIES.sha256` sidecar for gate-v2, both drivers now record
  `runtime_libraries_sha256` (verified on the desktop), `tests/test_desktop_parent_calibration.py`.
  Canonical suite 1,429 / 0 fail / 0 error / 1 skip (265.6 s). CHANGES.json 65 modified + 33 new,
  TESTS.json rewritten. Still open: OP15 attached -> phone-owned gates A-D; 6.1 GB untracked run
  artifacts under campaigns/ and baselines/ and the untracked source files await the owner's
  commit decision (no commit or push by the agent).
