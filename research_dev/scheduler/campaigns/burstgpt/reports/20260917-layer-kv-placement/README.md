# Per-layer host KV: first physical milestone

## Outcome

`pair-r3` passed on RTX 4060 Ti + OP15. Both arms processed the same 9,722-token
document and generated the same 32 tokens with a 32,768-token KV allocation.
Binary/library, model, request and KV-plan identities match. This proves the
combined per-layer CPU-KV and phone-FFN path, not a larger maximum usable context
than the desktop or a qualified automatic scheduler route.

| Metric | Desktop control | Phone-owned FFNs |
| --- | ---: | ---: |
| CPU / GPU KV allocation | 4 / 1 GiB | 4 / 1 GiB |
| End-of-request process RSS | 21.800 GB | 12.380 GB |
| Model-file mapped bytes | 18.072 GB | 8.446 GB |
| Desktop load to READY | 58.373 s | 58.634 s |
| Phone preload | None | 50.359 s |
| Prefill | 281.234 s | 424.382 s |
| Decode | 765.039 ms/token | 765.984 ms/token |
| Request duration | 305.718 s | 448.896 s |
| Paid preparation + request | 364.310 s | 591.789 s |
| Paid CPU package energy | 21.277 kJ | 17.787 kJ |
| Paid GPU board energy | 13.807 kJ | 17.368 kJ |
| Assumed phone energy at 4.5 W active | 0.319 kJ idle | 2.663 kJ |
| Combined paid energy | 35.403 kJ | 37.818 kJ |

RSS decreased by 9.420 GB (43.21%). Exact page-unmapping proof accounts for
9,625,706,496 bytes of the planned 9,625,927,680-byte FFN relocation; boundary
pages are not credited. Both cgroups still approached the 22 GiB charged-memory
cap because of file cache. This is a reduced process working set, not a claim
that whole-system charged memory fell by 9.42 GB. Neither arm had an OOM event.

No performance or energy win is established by this pair. Prefill increased
50.90%, decode differed by only +0.12%, and the request took 46.83% longer. With
phone preparation included, energy increased 6.82% at assumed 4.5 W. Estimated
savings at 3 / 4.5 / 6 W are **-4.32% / -6.82% / -9.33%**. These are one-pair
diagnostics, not repeated-run statistical or energy qualification. The paid
boundary excludes initial model-hash verification and final process/USB cleanup.

Phone proof: HTP0 / HTP1 / HTP2 retain disjoint layers 0-5 / 6-11 / 12-17,
generation 1, 3,208,642,560 resident bytes each. Request-scoped proofs count
642 logical FFN calls and 58,518 layer-token rows per session, with every expected
row verified. Terminal physical subrequests, including warm-up, total 4,734
(1,578 per session). Zero USB reset recoveries; Android restored at 5 Gbit/s.
GDM remained active. No gate processes remain.

Final remote artifacts: `/mnt/storage/s42-layer-kv-20260917-MrfN48/pair-r3`.
Local copies are under `physical/pair-r3/`; derived figures are in `SUMMARY.json`.

| Artifact | SHA-256 |
| --- | --- |
| Desktop RESULT | `1da2366d90cc8a179f7fdd8376ba3ac20cfdb9716c16d5aec69921472dfc0baf` |
| Phone RESULT | `6d26796c59ec870298c8d81122a343c028688e00c3e7795d41c8c5dba6b50a8d` |
| Shared runtime manifest | `79a890b85463aaaae3a3c9514ae33f6b44bcecd87df1f0b0d5af6444af8c2f36` |
| Shared request | `23fef13066e8adbe18397cc28c6a490e60cd293698457a5515b8dd2b8032d6ba` |
| Native request proofs | `3502b771127e2e222eaa5d8868be96b90097479fc186880d9406bc4dc7ce2092` |
| Phone close receipt | `d7d727235c553031560b89e98e0d8a4d0cbb88e9c4aa7cc410d4465d0ef1f7c6` |

## Scope

User approved per-layer KV placement first, before within-layer CPU/GPU attention
merging. This change supports a fixed-capacity F16 KV context whose global-attention
layers can keep their KV and execute attention on the CPU independently of the
layer's GPU projection weights. It does not implement growing KV allocation,
automatic live KV migration, or within-layer partial-softmax merging.

The first physical gate uses Qwen3-14B's existing dequantized-F16 artifact, RTX 4060
Ti and OP15. It is a native memory/attention gate using canonical launch and phone
session adapters, not a qualified automated scheduler route or a full trace.

## Implementation

- `--kv-cpu-layers` and copied context parameters select zero-based layers.
- KV allocators honor the selection, including the SWA/non-SWA wrapper. The
  attention graph pins the selected attention subgraph to CPU, not its projection
  matmuls. Both flash and ordinary attention are covered.
- Unsupported recurrent/hybrid, MLA, tensor-split and externally shared contexts
  reject explicit overrides. Shared Gemma KV consumers must agree with their
  source layer. Physical validation in this milestone is Qwen, not every model.
- `scheduler/_internal/kv_placement.py` derives F16 KV bytes from model metadata
  and an explicit per-pool KV budget. It spills whole global-attention caches,
  preserves sliding-window placement, and hashes the plan. Context limits remain
  those in the artifact; this does not grant unsupported 128k Qwen contexts.
- Launch identity includes the KV layer selection, so an old endpoint cannot be
  reused with a different selection.
- The memory-admission helper reuses `RuntimeMemoryLedger`. It refuses credit for
  a future decode-only release, requires verified omission and teardown recovery
  for remote-prefill credit, and excludes boundary pages from that credit. This
  helper is tested but is not yet integrated into automatic route selection or
  live context growth. Phone weights/workspace and host phase peaks are explicit
  caller-supplied demands, not an additional free memory pool.
- Restoration no longer treats advisory `WILLNEED` as completed population. It
  synchronously populates pages, propagating real errors; this still does not pin
  those pages against later memory pressure.
- A physical setup failure exposed missing spaces before `]` in the existing
  phone-launch shell tests. Fixed both predicates and exercised the real shell
  for STARTING, RUNNING, exited and terminal states, including quoted paths.
- The corrected FFN-enabled binary emitted a valid omission proof after an ANSI
  reset from the template diagnostic. The proof parser now strips only leading
  SGR color escapes; unrelated prefixes and all identity/memory checks remain
  strict. `pair-r1` preserves that admission failure; no request was executed.

Native changes are in `include/llama.h`, `common/{arg,common}.{cpp,h}`, and
`src/llama-{cparams,context,graph,kv-cache,kv-cache-iswa,model,mmap}.{cpp,h}` as
applicable. The native probe and its tiny model generator exercise the new API.
Existing unrelated changes in this dirty research tree were retained.
The context-parameter C struct changed: rebuild native callers with this header;
do not mix an older caller ABI with the new `libllama`.

## Validation

- 88 focused Python/native tests passed together in 1.872 s, including real-shell
  failure detection, strict colored-proof parsing and remote-resident admission.
- CPU/GPU mixed placement, flash on and off: identical greedy token decisions
  with FP16-appropriate logit tolerance on a tiny model. Passed on both the local
  A6000 and physical RTX 4060 Ti.
- CPU-only native equivalence and existing remote-resident/release tests passed.
- Python static checks and scoped whitespace checks passed. No full trace, full
  scheduler suite, commit, push or PR.

## Physical protocol

Remote artifact root: `/mnt/storage/s42-layer-kv-20260917-MrfN48`.
Both arms use the same deployed native binaries/libraries, model artifact, prompt,
32 output tokens, 16 GPU layers, CUDA graphs, batch 512, ubatch 128, and KV plan.
The prompt is the complete real `docs/build.md` plus a retrieval question; 9,722
tokens, with no truncation. Allocated context is 32,768 tokens, not the observed
prompt length. The KV plan allocates 4 GiB host RAM plus 1 GiB GPU RAM. Layers
0-31 use CPU KV/attention; 32-39 use GPU KV/attention. GPU projection/FFN parent
placement remains layers 25-39 plus the output head. This native revision counts
the output layer in `--n-gpu-layers 16`; it is not 16 repeating blocks. Corrected
the gate's initial default-pool formula accordingly. The final constrained KV
plan remains identical (CPU 0-31, GPU 32-39), so neither arm's actual KV placement
or native arguments changes.

Both arms run sequentially under a 22 GiB `MemoryMax` scope with swap disabled.
The initial individual checks used separate scopes with the same cap. The model
cache is dropped by file descriptor after artifact verification, with no other
GPU workload active; system-wide caches are not dropped. GDM and other unrelated
processes are untouched. A shared nonblocking rig execution lock is held.

The treatment chooses **remote-prefill**: complete FFNs of layers 0-17 are owned
by three phone sessions and omitted from the desktop mapping before context
allocation. This is intentionally different from the earlier 75% decode-only
release. Static KV allocation cannot spend a future decode release before the
prefill that creates it. Phone load, host load, prefill and decode are recorded
separately. The phone reads selected tensors from its existing full GGUF; this
gate does not claim new shard-file loading improvements.

The desktop control has the same new CPU-KV plan; it is not a claim about stock
llama.cpp defaults or the earlier frozen BurstGPT baseline. CPU/GPU energy is
sampled; phone power is assumed and reported at 3/4.5/6 W, not measured. The paid
phone estimate conservatively charges active power over the entire paid span,
including host loading. No repeated-run energy qualification is implied.

Build provenance: current `libllama`, common/server libraries and executables
were built locally with portable CPU settings. The rig's compatible GGML CUDA
backend/dependencies were reused from its prior native build; relevant GGML
header/backend source identities were checked, and the mixed-KV numerical test
passed on this exact bundle. Every loaded bundle library hash is in each arm's
`RUNTIME.json`; both arms use the identical bundle.

Required build options are `GGML_NATIVE=OFF`, `GGML_CUDA=ON`,
`LLAMA_BUILD_EXAMPLES=ON`, and `S41_SERVER_FFN_SPLIT=ON`. The latter was absent
from the first bundle. `phone-r2` loaded all three sessions but was correctly
rejected at desktop readiness because the binary never initialized its FFN
client. No request was sent. It restored Android USB normally. The corrected
bundle is kept separately in `bin-ffn-v2`; `desktop-r1` remains an initial KV-only
check, not the denominator for the corrected phone comparison. New matched arms
must both use `bin-ffn-v2`.

The first phone attempt, `phone-r1`, failed before inference: the experimental
128-column quantum produced 136 weight buffers and exceeded the deployed HTP
backend's operation buffer limit. Its worker log and failure remain preserved.
No FFN transfer was in flight when its failed launcher was stopped. The retry,
`phone-r2`, uses one full-width partition (17,408 columns), appropriate for this
fixed whole-FFN experiment. This is not an adaptive split/fraction qualification.
All three sessions reached READY in 46.426 s. The desktop control's unused phone
quantum value has no effect on its native command or execution.

## Initial physical evidence (not the final comparison)

| Check | Result |
| --- | --- |
| `desktop-r1` | 9,722 prompt tokens + 32 generated tokens completed, no truncation |
| Output check | Correctly identifies `-DGGML_CUDA=ON` and `-DGGML_SYCL=ON` from the document |
| Native KV buffers | CPU 4,096 MiB; CUDA 1,024 MiB |
| Prefill / decode | 281.571 s / 759.971 ms per output token |
| Desktop load / request | 80.878 s / 305.893 s |
| End-of-request process RSS | 21,451,563,008 bytes |
| `phone-r2` preload | Three sessions READY in 46.426 s; no request executed |

`desktop-r1/RESULT.json` SHA-256:
`2e6a32d8f8da7922fbccc1f999a1764349cb67ab91506f29376d5094d6d0ae13`.
The desktop's output is a direct factual check, not broad quality qualification.

The next attempt was refused before creating `phone-r3`: another experiment
acquired `/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock`.
No lock was bypassed and no unrelated process was stopped. `pair-r2` subsequently
completed real phone-assisted prompt execution with the corrected binary and
proof parser, but failed the terminal gate: this standalone harness had omitted
registration of its request-scoped native call proof with the phone owner. It
restored USB normally; its failure is preserved, not promoted to PASS. The fix
validates every expected full-width FFN row per layer, excludes unscoped warm-up
calls, binds the native request to the loaded session generations and registers
the resulting proofs before the unchanged strict terminal check. It adds a
negative/positive proof regression and validates the saved physical call log.

`pair-r3` reran the pair using that harness repair and unchanged native binaries,
placement, prompt and budgets. Both arms passed; final measurements are above.
No measured maximum-context extension or energy saving is claimed.

The bounded pair entry point supports `--arm pair --wait-for-rig-seconds 600`,
holds the cooperative lock through both arms, and stops if the phone arm fails.
The wait has a deadline and performs no device preparation before acquisition.

## Reproduction and next milestone

The final command, from the frozen remote Python source directory, was:

```sh
systemd-run --user --scope --unit=s42-layer-kv-pair-r3 \
  -p MemoryMax=22G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 \
  python3 -u -m research_dev.scheduler.campaigns.burstgpt.layer_kv_gate \
    --config /mnt/storage/s42-layer-kv-20260917-MrfN48/rig_config.json \
    --output /mnt/storage/s42-layer-kv-20260917-MrfN48/pair-r3 \
    --arm pair --wait-for-rig-seconds 600
```

Use a fresh unit and output directory for a new run. Do not overwrite artifacts
or bypass the shared rig lock. `CHANGES.json` distinguishes this work from the
pre-existing dirty tree; its before-images were preserved.

Remaining work is to wire the phase-safe memory reservation into automatic route
admission, then fill a longer context under the same fixed budgets and establish
the practical capacity boundary. The artifact's context limit remains 40,960.
Dynamic KV growth/migration, per-layer GPU-streamed prefill and within-layer
CPU/GPU partial-softmax merging are not implemented. The measured remote-prefill
penalty makes prefill staging the next performance issue, separate from the
successful CPU-KV/FFN memory mechanism. Phone-loss recovery with in-flight DMA-BUF
work was not physically injected or qualified.
