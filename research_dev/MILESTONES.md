# Historical Design A - Build Plan, De-Risk Spikes and Milestones

*Companion to [DESIGN.md](DESIGN.md) · drafted 2026-07-06*

> **DO NOT EXECUTE THIS FILE AS THE CURRENT PLAN.** It preserves the original
> Design A M0-M5 sequencing and risk assumptions. The primary target is now the
> executable phone warm tier in
> [ACTIVE_WARM_TIER_DESIGN.md](ACTIVE_WARM_TIER_DESIGN.md), with executable
> W0-W5 gates and the first bounded experiment in
> [S39](spikes/s39_phone_model_switch_trace/PLAN.md). Design A remains route
> `A0` and useful historical evidence.

> Measurement correction (2026-07-10): historical "zero interference" entries
> mean concurrent wall overlap only. Equivalent solo-versus-co-run lane times
> were not collected; S3-H0 now measures them. Static HTP B=16/B=32 works, while
> dynamic continuous admission and eviction remain unproven.

> Two named spikes (**S1** batched-decode-on-phone, **S2** single-copy weight sharing) run inside M0/M1 and are allowed to **veto the program early** — that is their purpose. Sequence the work so a fatal "no" surfaces in week 1–2, before the orchestrator is built.

Exit criteria are concrete go/no-go gates. Two named de-risk spikes (S1, S2) run inside M0/M1 and can veto the whole program early — that is their job.

**M0 — Ground truth + the two vetoes (week 1–2).** Establish the real model, and run both spikes before writing any orchestrator code.
- Sub-tasks: `gguf_dump` the staged Gemma-4 12B; patch the gemma4 n_layer switch so it loads as a known type; stand up the naive 3-node RPC ring (server→OP15→OP12→server) with device order [op15, op12, A6000] and the first-3-layers split; add `ggml_cast`→F16 at boundaries.
- **S1 (batched-decode-on-phone veto):** `llama-batched-bench -np 1,2,4,8,16` on op12 AND op15 for BOTH `HTP0` and `GPUOpenCL`, each run wrapped in `timeout 120` with an adb watchdog + DSP-SSR detector (a fastRPC hang wedges the process and returns no error code). Record `max-npl` = largest npl that (a) completes and (b) matches CPU-reference logits (a silently-wrong block-diagonal mask can "pass" on speed alone). Reproduce and root-cause the op15 `n_parallel=2` hang with `GGML_HEXAGON` verbose/opfilter.
- **EXIT M0:** 12B scalars documented (block_count, head_count_kv, key_length/_swa, sliding_window_pattern, shared_kv_layers, PLE flag) and the model loads as a known type. S1 verdict recorded per device/backend. **Decision:** if NPU `max-npl`>1 on op12 → NPU batched decode is upside; if it hangs (expected) → **commit phone decode to GPU-only, NPU prefill-only** and proceed — Design A is not blocked, only its NPU-decode energy upside is deferred.

**M1 — Correct static pipeline + single-copy weight sharing (week 2–4).**
- Sub-tasks: pre-provision per-phone mmap'd fp16 shard files with **global** `blk.{g}.*` names + owned-layer SWA/RoPE metadata; verify no `shared_kv_layers` group straddles a cut (assign a KV-self-contained block); confirm `lm_head`+sampler resolve to the A6000 and `token_embd` to host CPU; instrument on-wire dtype/nbytes per hop.
- **S2 (mmap+fastRPC+OpenCL single-copy spike):** ~100-line standalone on-device probe on op12 AND op15 — `clGetDeviceInfo(CL_DEVICE_EXTENSIONS)` for `cl_qcom_dmabuf_host_ptr`/`ion_host_ptr`; read `CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM` + `CL_DEVICE_PAGE_SIZE_QCOM`; `rpcmem_alloc2(SYSTEM heap, size+padding)` → `rpcmem_to_fd` → `clCreateBuffer(CL_MEM_EXT_HOST_PTR_QCOM, dmabuf fd)`; run `kernel_mul_mat_f16_f32` over the imported buffer and bit-compare vs the `clEnqueueWriteBuffer` path; run an HMX matmul on the SAME buffer concurrently and re-check for corruption; benchmark Adreno read bandwidth from the shared buffer vs a native `cl_mem`.
- **EXIT M1:** single-token 3-stage pipeline produces logits matching a full-model CPU reference within tolerance; measured per-hop payload is F16 (cast confirmed) or F32 (documented); S2 gives a per-device verdict: single-copy import **works** (proceed) OR **fails/uncached-slow** (adopt the two-copy fallback and budget the RAM). xmem-image path explicitly flagged as non-shareable.

**M2 — Interconnect + orchestrator overlap (week 4–6).**
- Sub-tasks: measure goodput (iperf3 -R) and p99 latency (983,040-byte length-prefixed TCP ping-pong with TCP_NODELAY) on server↔OP15, server↔OP12, the host-forwarded middle hop, and the WiFi topology; pick server-relay-over-USB vs shared-WiFi by *measured added latency per decode step*. Replace the ggml-rpc lockstep loop with a streaming, double-buffered stage server; separate prefill/decode sockets; build orchestrator-level k-in-flight-microbatch overlap over thread-parallel blocking RPC.
- **EXIT M2:** steady-state pipeline throughput gates on `max(stage_compute, p99 hop_comms)` not their sum (demonstrated by ≥1.5× speedup from overlap vs serial); p99 per-hop ≤ measured per-stage phone compute, including the host-forwarded segment under concurrent server compute; no head-of-line blocking between prefill and decode frames.

**M3 — Batched continuous decode end-to-end (week 6–9).** Gated on S1.
- Sub-tasks: run the full 3-stage ring under `llama-server --cont-batching` (dynamic admit/evict, ragged lengths — exercises paged-KV fragmentation that static batched-bench does not); enforce `kv_unified=true` (ne[3]==1, block-diagonal F16 mask); confirm attention stays on the phone accelerator (grep op-support dump for CPU fallbacks — a per-layer DSP↔CPU ping-pong collapses energy); validate KV RAM against `/proc/meminfo MemAvailable` at ctx {512, 2k, 4k}.
- **EXIT M3:** 64–128-seq continuous-batched decode runs stably through all 3 stages at 8 fps admission with no hang, no CPU fallback on the attention path, and correct logits vs reference. If S1 said NPU-hang, this runs on GPU-only phone decode.

**M4 — Reliability + energy accounting (week 9–12).**
- Sub-tasks: build the out-of-band control plane (heartbeat, per-hop deadlines on GET_TENSOR via `SO_RCVTIMEO`/`TCP_USER_TIMEOUT`, de-assert the client aborts, agent-reforks the phone `rpc-server`); test with mid-step SIGSTOP (live hang → must fail over, not block forever) and SIGKILL (crash → must not GGML_ABORT the orchestrator). Stand up unified J/tok on a **whole-device** plane on both sides: A6000 = nvidia-smi board power **plus** host CPU/NIC/PSU-loss lift; phones = coulomb method (charging off, `charge_counter` slope × V above idle) with the USB rail explicitly attributed; multi-minute steady-state windows; error bars for the ~10% DVFS variance.
- **EXIT M4:** a wedged phone is detected and recovered without killing in-flight sequences; a defensible fleet J/tok with error bars, measured on matched boundaries, with the USB-rail and server-non-GPU terms accounted (not the apples-to-oranges "matches exactly" identity).

**M5+ — Layer rebalance toward an actual energy win (week 12+).**
- Sub-tasks: move progressively more **SWA/window-capped** layers onto the phones, bounded by measured free RAM and sustained thermal power; per step re-verify KV-self-containment (R5) and re-measure fleet J/tok vs the 0.15 J/tok server baseline; if batched GPU decode must be energy-competitive, wire the xmem image path (accepting the derived second weight copy).
- **EXIT M5+:** the layer split at which fleet J/tok first crosses below the pure-server baseline is identified and reproducibly measured — or a documented finding that Design A does not cross it within the phones' RAM/thermal envelope (the honest negative result that would vindicate the Design-B recommendation).

---

## First-Week Actions

- [ ] gguf_dump the actual staged Gemma-4 12B and record block_count, embedding_length, head_count_kv, key_length, value_length, key_length_swa/value_length_swa, sliding_window, sliding_window_pattern, shared_kv_layers, and embedding_length_per_layer_input (PLE). Replace every 'confirm from gguf' placeholder with the real scalar; if it loads as LLM_TYPE_UNKNOWN, patch the gemma4 n_layer switch so it loads as a known type.
- [ ] Run S1 NOW (highest priority, it can veto the program): llama-batched-bench -np 1,2,4,8,16 on op12 AND op15 for both HTP0 and GPUOpenCL, each under `timeout 120` with an adb/DSP-SSR watchdog and a CPU-reference logits check. Reproduce the recorded op15 n_parallel=2 hang and root-cause it with GGML_HEXAGON verbose/opfilter. Record max-npl per device/backend.
- [ ] Write the ~100-line S2 on-device probe and run it on both phones: check CL_DEVICE_EXTENSIONS for cl_qcom_dmabuf_host_ptr vs ion_host_ptr, read the EXT_MEM_PADDING/PAGE_SIZE QCOM caps, then rpcmem_alloc2->rpcmem_to_fd->clCreateBuffer(CL_MEM_EXT_HOST_PTR_QCOM) and bit-compare a kernel_mul_mat_f16_f32 read against the write-buffer path, with a concurrent HMX read for corruption. This settles single-copy sharing before any ggml integration.
- [ ] Confirm the network reality: tether both phones, `ip -br addr` (expect the 192.168.42.0/24 subnet+gateway collision), `ip route`, and `ip route get <OP12-ip>` + ping from OP15 (expect FAIL). Decide server-mediated relay vs single-LAN/WiFi now, since it shapes the transport code.
- [ ] Stand up the naive 3-node RPC ring with device order [op15, op12, A6000] and the phones owning the FIRST 3 layers; instrument the on-wire tensor dtype/nbytes per hop and confirm lm_head+sampler resolve to the A6000 (not op12) and token_embd to the host CPU. Add the ggml_cast->F16 boundary and verify the payload halves.
- [ ] Baseline the interconnect with iperf3 -R and a 983,040-byte length-prefixed TCP_NODELAY ping-pong (median + p99) on server<->OP15, server<->OP12, and the host-forwarded middle hop, so the latency budget rests on measured RNDIS/NCM goodput, not the USB3 ceiling.
- [ ] Write the M0/M1 risk-register entries R1, R3, R5, R11 into the repo README with the current verdicts, and record the decision rule: if S1 shows an NPU hang on op12, commit phone decode to GPU-only / NPU-prefill-only as the baseline and proceed.

---

## Build-Plan Rationale & Repo Layout

*From the planning subsystem — detailed spike definitions and the proposed `research_dev/` code layout. Where its milestone numbering differs slightly from the authoritative plan above, the plan above wins.*

**Arch (confirmed from gguf on staged model).** The staged small model is `general.architecture = gemma4`, `general.name = "Gemma 4 E2B"`. Confirmed fields for **E2B** (from `gemma-4-E2B-fp16.gguf` via `gguf.GGUFReader`): `block_count=35`, `embedding_length=1536`, `feed_forward_length=12288`, `head_count=8`, `head_count_kv=1` (aggressive GQA — 1 KV head), `key_length=512`, `value_length=512`, `attention.sliding_window=512`, `sliding_window_pattern` present (interleaved local-SWA / global layers), and **`embedding_length_per_layer_input=256`** (per-layer input embeddings, PLE — a Gemma-3n/matformer trait).

**The 12B is NOT staged here** (only E2B variants exist under `/home/myid/zs89458/Documents/models`). For 12B, `block_count (~48)`, `embedding_length (~3840)`, `feed_forward_length`, `head_count_kv`, `key_length`, and whether it retains PLE are **confirm from gguf** (run `gguf_dump --no-tensors` on the real 24 GB fp16 file before M0). Two arch traits drive the design and must be verified for 12B:

- **Per-layer input embeddings (PLE).** If 12B keeps PLE, *every* transformer layer needs a per-token embedding indexed by token id, not just the previous hidden state. A phone stage owning layers L..L+k must therefore either (a) hold the PLE table *slice* for its layers (pre-provisioned, indexable) and **receive the token ids** each step, or (b) receive precomputed per-layer embedding vectors from the server. Token ids are tiny (≤128×int32 = 0.5 KB/step), so option (a) is cheap on the wire but adds a table slice to each shard. **This is a real plumbing requirement, not an afterthought — surface it in M1.**
- **Interleaved SWA/global attention.** KV size differs per layer (SWA layers cap context at the window; global layers grow with sequence). KV never crosses the wire (each stage owns its layers' KV), but the *per-layer KV footprint on the phone* — which bounds how many layers a phone can host at batch 64–128 — depends on which layers are global vs SWA. Assign **SWA-heavy layer ranges to phones** to keep phone KV small.

**Transport & pipeline substrate (confirmed by code read).** llama.cpp's RPC backend (`ggml/src/ggml-rpc/`, `tools/rpc/rpc-server.cpp`) is **TCP/IP sockets** (`socket_t::create_server(host,port)`, `MAX_CHUNK_SIZE=1 GiB`). It registers a remote node as a **ggml-backend device**; `rpc-server -d <dev>` selects which local backend the node exposes (e.g. a Hexagon `HTP0` or an OpenCL `GPUOpenCL` device — **confirm exact device strings** with `rpc-server` startup log / `ggml_backend_dev_get_description`). The orchestrator wires nodes in via `common/arg.cpp:add_rpc_devices()` (`--rpc host:port,host:port`). Layer→device assignment + KV-per-device is llama.cpp's normal `LLAMA_SPLIT_MODE_LAYER` behavior, so **Design A is a configuration of the existing sched, not a new engine** — this is the single biggest schedule saver, and also the biggest thing to verify (see risks).

**Two code-confirmed risks that shape the milestones:**

1. **Automatic pipeline overlap may silently disable.** `src/llama-context.cpp:366-393`: llama.cpp only enables microbatch pipeline-parallel overlap when *every* non-CPU backend reports `props.caps.async && props.caps.events`. If the RPC backend, ggml-hexagon, or ggml-opencl does not advertise both, `pipeline_parallel=false` and stages run **serially with bubbles** (server waits for op15 waits for op12). We must measure these caps in S1/M2 and, if false, either add async/event support to the RPC backend or accept serial execution (still correct, just bubble-heavy) for early milestones.
2. **Inter-phone routing is via the host, which is actually convenient.** In ggml-backend sched, a tensor produced on device A and consumed on device B is copied through host memory unless a direct A→B path exists. Between two RPC devices there is no direct path, so **op15→op12 hidden states route server→op15→server→op12→server**. This doubles hops but **sidesteps the USB "phones can't peer" trap for free** — every link is host↔phone. Activation sizes are tiny (batch-128 hidden-3840 fp16 ≈ 0.94 MB/hop; 512-tok prefill ≈ 3.9 MB), hundreds of× under USB/WiFi, so the extra hop is latency, not a bandwidth problem.

---

## 1. De-risking spikes (do these FIRST; they gate everything)

### S1 — Batched-decode-on-phone hang/throughput spike (BLOCKING)
**Why:** the roofline M-sweep is one synthetic matmul; it does **not** prove N sequences with separate KV + block-diagonal mask run on the phone backend. The project already recorded **"NPU n_parallel=2 HANGS" on op15**. If real continuous batching cannot run on the phone NPU, the whole batched-decode premise of Design A is at risk and we fall back to GPU-only decode on-phone.

**Method:** on **op12** first (v75, more stable per memory notes), then op15, using the already-staged `llama-batched-bench` and a staged small model (`gemma-4-E2B-it-fp16.gguf` / `-Q4_0`):
```
# NPU path
llama-batched-bench -m gemma-4-E2B-it-fp16.gguf --device HTP0  \
    -c 4096 -b 512 -ub 512 -npp 128 -ntg 64 -npl 1,2,4,8,16
# GPU path
llama-batched-bench -m gemma-4-E2B-it-fp16.gguf --device GPUOpenCL \
    -c 4096 -b 512 -ub 512 -npp 128 -ntg 64 -npl 1,2,4,8,16
```
**Record per npl:** does it hang (and at which npl)? aggregate decode tok/s, per-step latency, and (with the M0 power harness stub) J/tok. Also dump `ggml_backend_dev_get_props().caps.async/.events` for HTP0 and GPUOpenCL (feeds risk #1).
**Exit criteria:** we have a definitive npl-vs-{hang,tok/s,J/tok} table for both engines on both phones, and a documented **max stable npl** per engine. If NPU hangs at npl≥2 on both phones, the decode plan is **GPU-only on-phone** and NPU is prefill-only — record that decision. **No milestone past M2 starts until S1 is green or its fallback is chosen.**

### S2 — mmap + fastRPC + OpenCL shared-weight spike (BLOCKING for M4)
**Why:** the hard constraint is one physical weight copy shared by Hexagon and Adreno via a single mmap'd dmabuf; duplicate buffers would blow the phone RAM budget and forbid concurrent NPU-prefill / GPU-decode on the same weights.
**Method:** a standalone C++ harness on op12 that (1) `mmap`s one shard tensor file, (2) wraps the mapping as a fastRPC/ION dmabuf handed to the Hexagon backend, (3) imports the *same* fd as an OpenCL buffer/image (Adreno), (4) runs a MUL_MAT on each engine and checks numerical agreement, (5) verifies via `/proc/<pid>/smaps` + dmabuf accounting that **RSS reflects one copy** (not two).
**Exit criteria:** both engines compute correct MUL_MAT from the same fd; measured resident weight bytes ≈ 1× shard size (± page rounding), not 2×. If a single fd cannot be shared, fall back to two mmaps of the *same file* (still one page-cache copy, OS-dedup'd) and document the RAM delta.

---

## 2. Milestones (concrete exit criteria)

| M | Scope | Key artifacts | Exit criteria |
|---|---|---|---|
| **M0** | Single-device 12B fp16 baseline + power harness | `configs/workloads/gemma4_12b_a6000.yml`; power harness driver | 12B fp16 runs on A6000; reproduce **floor ≈0.15 J/tok net @ batch 256, OOM @ 320**; harness logs tok/s + J/tok + Wall/GPU power; `gguf_dump` of real 12B recorded (layers/hidden/PLE/SWA-pattern) |
| **S1/S2** | (above) run in parallel with M0 | spike reports | S1 npl table green-or-fallback; S2 one-copy proof |
| **M1** | Server↔1-phone **2-stage** pipeline, 1 layer on phone, greedy | `proto/pipeline.proto` (or reuse RPC wire); `configs/topology/2stage_op12.yml` | With **op12 hosting layer N via `rpc-server -d <cpu|HTP0>`**, server offloads 1 layer over `--rpc`; **greedy output token-for-token identical** to M0 single-device on a fixed prompt (correctness gate); PLE token-id plumbing verified if 12B uses PLE |
| **M2** | **3-stage** prefill pipeline: server + OP15(2 layers) + OP12(1 layer) | `configs/topology/3stage_baseline.yml` | 512-tok prefill flows server→op15→op12→server; output matches M0; per-hop activation size logged (~3.9 MB); **async/events caps recorded** → note whether auto pipeline-overlap is on or serial |
| **M3** | Continuous batching + 64/128 batched decode through the 3-stage pipeline | scheduler config; batched decode driver | Continuous-batch decode at batch 64 then 128 runs **without hangs** (bounded by S1 max-npl); dynamic admit/evict keeps batch topped; aggregate tok/s + J/tok logged; **correctness** vs single-device on same seeds |
| **M4** | Intra-phone NPU-prefill / GPU-decode + **shared weights** | shard-provisioning tool; per-phone dual-engine launcher | On each phone, prefill routes to Hexagon, decode GEMV routes to Adreno, **on the S2 shared dmabuf**; concurrent co-schedule shows **≈zero interference** (reproduce the measured finding); RAM shows one weight copy |
| **M5** | Energy vs A6000 + layer rebalancing | `bench/energy_report/`; rebalance sweep | End-to-end **J/tok (server+2 phones, wall power)** vs M0 A6000 floor, at matched throughput/quality; rebalance layer counts (2+1 → more) up to phone RAM/thermal limits; produce the **layers-on-phone vs J/tok vs tok/s** Pareto curve; honest verdict on whether Design A saves net energy |

**Rebalancing bound (M5).** Per-layer 12B fp16 weight ≈ 0.4–0.5 GB (confirm from gguf). Phone RAM and **5–12 W sustained thermal** are the two ceilings; phone KV at batch 128 adds to RAM (size per layer depends on SWA-vs-global — assign SWA layers to phones). Expect the Pareto to show phones winning only on **prefill** offload (NPU 7.12 TFLOPS compute-bound) and batched decode being marginal (`~0.05–0.08 J/tok phone estimate is UNVERIFIED` until M3/M5).

---

## 3. `research_dev/` repo layout

The repo already has a working harness at `npu-harness/` (a `controller.py` CLI with `setup/build/push/run/session/simulate`, and `configs/{devices,workloads,frameworks,_base}/`). **Extend that, don't fork it.** The top-level `research_dev/` currently holds only `talks.md`; grow it into the Design-A subsystem, reusing `npu-harness/research_dev/lib` for device/build/push plumbing.

```
research_dev/
  README.md                      # this build plan, exit criteria, run recipes
  design/
    designA_pipeline.md          # architecture: layer split, KV placement, PLE/SWA notes
    risks.md                     # async/events caps, npl-hang, host-routing, PLE
  spikes/
    s1_batched_decode/           # llama-batched-bench sweep driver + result parser
      run_op12.sh  run_op15.sh  parse_npl.py  RESULTS.md
    s2_shared_weights/           # mmap+fastRPC+OpenCL one-dmabuf harness
      shared_mmap_spike.cpp  CMakeLists.txt  verify_rss.py  RESULTS.md
  topology/                      # pipeline stage/layer maps (drive --rpc + layer split)
    2stage_op12.yml  3stage_baseline.yml  3stage_rebalanced_*.yml
  provision/                     # PRE-inference shard creation & push (NO runtime weight RPC)
    shard_layers.py              # slice 12B gguf -> per-stage shard + PLE-slice + assignment
    push_shards.sh               # adb/scp shards to phones' local storage
  orchestrator/
    launch_pipeline.py           # start rpc-server on each phone (-d HTP0/GPUOpenCL), wire --rpc
    scheduler.md                 # continuous-batch admit/evict config notes
  proto/
    pipeline.proto               # activation/token-id envelope IF we bypass raw RPC wire
  power/
    harness.py                   # A6000 (nvml) + phone (coulomb: charge_counter slope) power
    coulomb.md                   # op15/op12 method: charge off + charge_counter slope
  bench/
    m0_baseline/  m3_batched/  m5_energy/    # per-milestone result dirs + plots
  configs/  ->  symlink or reuse npu-harness/configs
  lib/       ->  reuse npu-harness/research_dev/lib  (device, build, push, runner)
```

**Transport config lives in `topology/*.yml`**: each file lists stages (server, op15, op12), the layer range per stage, the `rpc-server` device string per phone, and the host↔phone IP/port (USB-tether IP or shared-LAN IP — recall phones can't peer, so **all links terminate at the host**).

---

## Consolidated Task Checklist

Per-subsystem dev tasks pulled from each design section.

### Topology, Layer Partitioning & Dataflow (Design A, 3-stage pipeline)
- [ ] Run gguf_dump.py on the real Gemma-4 12B fp16 shard and record n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff, n_vocab, n_kv_shared_layers, n_embd_per_layer, is_swa_impl pattern, and whether any layer is MoE; freeze these into the partition config.
- [ ] Implement the assert_kv_local check: for every phone-hosted layer verify has_kv(il)==true (or its KV-source layer is co-located); fail the build and emit an interior-block fallback split if not.
- [ ] Write the offline sharder that consumes the partition config and produces per-node mmap-ready local shard files (server: tok_embd+layers 0..44+output_norm+lm_head; op15: layers 45-46; op12: layer 47), with a manifest recording each layer's is_swa and has_kv.
- [ ] Build the stage runtime that ships/receives a single [n_embd, n_tok] fp16 hidden tensor + positions per hop (server->op15->op12->server), with the server-local loop-back: output_norm -> lm_head -> softcap -> sample -> re-embed at layer 0.
- [ ] De-risk batched phone decode FIRST: run llama-batched-bench on op15 and op12 with one hosted layer at B=16/64/128, on both Adreno (OpenCL) and Hexagon paths, to reproduce/resolve the n_parallel=2 NPU hang and confirm per-sequence KV+masks work before any energy milestone.
- [ ] Validate end-to-end prefill+decode correctness of the 3-stage pipeline against a single-node reference (logits/argmax match) at B=1, then B=64-128.
- [ ] Add the per-layer-embedding path conditionally (only if gguf has n_embd_per_layer>0): ship each stage its layers' per-layer embedding inputs alongside the hidden tensor.
- [ ] Attach the vision encoder + mmproj on the server in front of layer 0, injecting image embeddings as raw (unscaled) inpL, and prefill 8-fps frames through the same pipeline.
- [ ] Instrument the rebalancing knob: script one-layer-at-a-time re-sharding + llama-batched-bench re-measurement, logging per-stage step time, phone thermal/clock, and phone KV RAM, to find the thermal-bounded max layers per phone.

### Interconnect, Transport & Wire Protocol (server ↔ OP15 ↔ OP12)
- [ ] Bring up USB tethering on OP15 and OP12, assign static /30 subnets, set net.ipv4.ip_forward=1 + routes; verify OP15↔OP12 reachability with ping; measure goodput and RTT with iperf3 and a small-message socket ping-pong, and record actuals vs the 600 MB/s / 0.1–1 ms assumptions.
- [ ] Implement pipe_hdr framing + pipe_msg_type enum on top of the existing socket_t/send_msg/recv_msg from ggml-rpc/transport.cpp; one persistent pre-established TCP connection per directed hop, TCP_NODELAY + TCP_QUICKACK, raised usb0 MTU.
- [ ] Implement zero-copy activation serialize (header + seq-table + straight send of the contiguous fp16 hidden-state buffer) and deserialize directly into the stage-input tensor; assert little-endian, packed nb, hidden==gguf value.
- [ ] Implement the control plane: SESSION_INIT (layer range/dtype/hidden/max_batch/kv_pool), SEQ_ALLOC/SEQ_FREE lock-stepped with the server's continuous-batch admit/evict, CREDIT flow control, HEARTBEAT (throttle signal), ERROR/resync on batch_id gaps.
- [ ] Wire the server scheduler to split each decode batch into k in-flight microbatches; instrument pipeline fill/drain and steady-state throughput; tune k and credit-window depth against measured per-stage decode times.
- [ ] Backpressure/robustness test: sustained 8 fps prefill + injected DVFS/thermal throttle on a phone; confirm credits stall upstream, bounded server queue sheds cleanly, and heartbeat-driven rebalance works.
- [ ] Add structural guards + a test proving KV and token-ids never serialize (only the hidden-state tensor is a bulk payload); confirm hidden_dim, layer count, vocab, and native dtype from gguf_dump on the actual 12B shard.

### Weight provisioning: pre-downloaded shards (no runtime weight RPC)
- [ ] Run gguf_dump on the real gemma-4-12b-f16.gguf and fill the §2 table: block_count, hidden size, SWA n_pattern/dense_first, tied vs separate lm_head, per-layer and embd byte sizes.
- [ ] Write extract_shards.py over gguf-py (fork tools/gguf-split's tensor-copy loop): partition by blk layer index per plan.json, keep global tensor names (Path B), copy hparams + vocab strings + pipeline.layer_start/count/is_first/is_last KV, emit per-shard files.
- [ ] Emit manifest.json with source_gguf_sha256, per-file sha256, layer ranges, dtype, swa_pattern, and monotonic version; add a verifier that recomputes hashes on-device before mmap.
- [ ] Patch the llama.cpp model loader to gate per-layer create_tensor on [pipeline.layer_start, +count) while keeping full hparams so is_swa(il)/rope_base(il) stay correct; add a build-time assert that the range reproduces the full model's is_swa/RoPE sequence.
- [ ] Add a pipeline stage run-mode to the graph builder: input hidden-state tensor when !is_first, run only owned layers, output hidden state when !is_last (final norm+lm_head+sample only when is_last).
- [ ] Define on-device cache layout /data/local/tmp/llmshard/<model_id>/<version>/ with a current-> pointer, staging+fsync+verify fetch, and refuse-on-mismatch cold start; wire the same mmap into both Hexagon and OpenCL backends.
- [ ] Implement re-shard/rebalance: regenerate shards under a new version, diff-fetch changed files, drain pipeline, atomic current-> flip, reload+re-mmap+rebuild stage graph, GC old versions.
- [ ] Validate end-to-end: provision server(0-44)/op15(45-46)/op12(47) shards, confirm each loads and produces correct hidden states, and run llama-batched-bench on the phone shards to de-risk batched decode before scaling layers.

### On-Phone Single-Copy Weight Sharing: mmap + fastRPC (NPU) + OpenCL (GPU)
- [ ] Confirm Gemma-4 12B fp16 per-tensor weight shapes/offsets via gguf_dump and compute per-layer ION buffer size for OP15 (2 layers) and OP12 (1 layer).
- [ ] On OP15 and OP12, probe clGetDeviceInfo(CL_DEVICE_EXTENSIONS) for cl_qcom_dmabuf_host_ptr / cl_qcom_ion_host_ptr and read CL_DEVICE_PAGE_SIZE_QCOM and CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM.
- [ ] Write a standalone smoke test: rpcmem_alloc2 an ION buffer, write a known f16 pattern, register with fastRPC, import the dmabuf fd into OpenCL via CL_MEM_EXT_HOST_PTR_QCOM, and verify both HTP and Adreno read back the identical bytes (compare to CPU reference).
- [ ] Factor ggml_hexagon_shared_buffer so base/fd/size are exposed via an accessor; create the shared ggml-phone-shared module holding the ION allocator + fastRPC registration + OpenCL importer returning {base, dmabuf_fd, size, base cl_mem}.
- [ ] Add the OpenCL import branch: new alloc path using clCreateBuffer(CL_MEM_READ_ONLY|CL_MEM_EXT_HOST_PTR_QCOM), device-extension capability gating, per-tensor clCreateSubBuffer views, and skip clEnqueueWriteBuffer for f16 weights already resident in the shared buffer.
- [ ] Set the phone-shared buffer-type alignment to max(page size, EXT_MEM_PADDING, MEM_BASE_ADDR_ALIGN) and validate galloc places every f16 weight tensor on a legal sub-buffer offset with no overlap.
- [ ] Implement one-time cache clean after weight load (rpcmem flush + msync) and confirm no per-step coherence maintenance is issued during a concurrent HMX-prefill + Adreno-GEMV run.
- [ ] Build the fused ggml-phone backend that owns both a Hexagon session and an OpenCL context, exposes the phone-shared buffer type, and dispatches ops (prefill/large-M MUL_MAT to HTP, decode GEMV to Adreno) reading weights from the shared buffer via each engine's handle.
- [ ] Implement the quantized/no-import fallback: keep shared buffer for f16-compatible tensors and allocate engine-native disjoint copies for quantized matmul weights; document the extra memory cost.
- [ ] End-to-end validate on a single layer that HTP prefill and Adreno decode produce correct outputs from one shared ION copy, measuring DRAM footprint to prove no weight duplication.

### Continuous Batching, Prefill/Decode Disaggregation & 3-Stage Pipeline Scheduling
- [ ] Run gguf_dump on the staged Gemma-4 12B fp16 to fix layer count, per-layer tensor bytes, GQA/MLP widths, and lm_head/vocab; replace every 'confirm from gguf' constant in the budget.
- [ ] De-risk phone batched decode FIRST: on op12 and op15 run llama-batched-bench with -np 2,4,8,16,32 on Hexagon and on OpenCL with a small staged model; record hang/no-hang and per-step scaling; root-cause the op15 npl=2 hang.
- [ ] Validate masked block-diagonal attention on-device with test-backend-ops (FLASH_ATTN_EXT with dst ne[3]==1, and SOFT_MAX with a real per-row mask) at N=2..32 on both phones.
- [ ] Build the orchestrator process on A6000: prefill_queue, ready_set, batch table (slot->seq->position), and the sampler + lm_head + next-token embed loop; run it in serialized single-sequence-per-stage mode first (proven ops only) as a correctness harness end-to-end across server->OP15->OP12->server.
- [ ] Implement admit/evict as a lock-step control broadcast so the three distributed KV pools stay index-aligned; verify eviction of a finished slot frees KV on the owning stage only.
- [ ] Implement the software microbatch pipeliner (K~3-4 in-flight microbatches, one worker thread per stage edge over blocking RPC); measure steady-state stage occupancy and confirm all 3 stages busy.
- [ ] Implement occupancy-OR-timeout batch formation (LOW=64/HIGH=128 + max-wait timer) and Sarathi-style chunked-prefill fusion (fused [decode ++ prefill-chunk] ubatch with per-row mask), gated on the de-risk tasks passing.
- [ ] Enable intra-phone concurrent prefill-on-NPU + decode-on-GPU over one mmap'd weight copy; confirm the measured zero-interference co-scheduling holds under the real scheduler.
- [ ] Wire USB tethering + host IP-forwarding for the OP15<->OP12 hop; measure per-hop throughput/RTT vs WiFi and pick transport; feed measured stage latencies into per-stage microbatch sizing to absorb v81/v75 DVFS skew.
- [ ] Add backpressure: bounded admission queue, dynamic HIGH occupancy, and a layer-rebalance hook; add per-stage rolling-latency monitor to detect thermal throttling and resize microbatches.

### Distributed KV Cache & Intra-Phone NPU-Prefill / GPU-Decode Execution
- [ ] Run gguf_dump on the real Gemma-4 12B fp16 shard; record n_layer, d_model, n_head/n_kv_head, head_dim, vocab, per-layer attention type + sliding_window; recompute the §2.2 KV budget table with confirmed numbers.
- [ ] Execute §5.1 Step A (llama-batched-bench --device HEXAGON0, npl sweep 1..64, -fa on) on op12 and op15 to reproduce or clear the 'n_parallel=2 HANGS' finding on the NPU.
- [ ] Execute §5.1 Step B on --device GPUOpenCL (npl 16..128, -fa on), then re-run with GGML_OPENCL_ADRENO_XMEM_GEMM=1, capturing tok/s and correctness.
- [ ] Execute §5.1 Step C correctness diff (llama-batched, temp=0): identical prompts must yield identical tokens across sequences and vs CPU/CUDA reference; then distinct prompts to catch cross-sequence KV bleed.
- [ ] Add a FLASH_ATTN_EXT + SOFT_MAX N-sequence unified-KV test case (GQA 16:8, F16 block-diagonal mask, dst->ne[3]==1) to tests/test-backend-ops.cpp and diff phone backends against CPU.
- [ ] Set kv_unified=true in the phone context config and verify the built graph produces FA ops with ne[3]==1 (add an assert/log in the graph path).
- [ ] Verify the KV cache buffer type on each phone is the shared LPDDR/dma-buf buffer (not a Hexagon-private buffer) so decode-on-GPU reads NPU-written KV with zero copy; add a startup check.
- [ ] Implement the server-side scheduler control channel: global seq_id assignment, per-step batch descriptor (active seq_ids + positions) shipped with each activation hop, and admit/evict/preempt broadcast; phones apply seq_id->KV-slot mapping and NAK on RAM cap.
- [ ] Instrument concurrent NPU-prefill + GPU-decode on one phone with two backend_sched streams and confirm the zero-interference measurement holds under real layer graphs (LPDDR bandwidth counters).
- [ ] Derive and enforce per-device admission caps (max_active_seq x max_ctx) from measured free RAM after mmap weights + engine scratch; wire the tightest-stage cap into the server scheduler as the pipeline admission bound.
- [ ] Resolve the server KV OOM wall: prototype KV quantization (q8_0) and/or context cap and re-measure the max batch the A6000 sustains for the ~45-layer share at 12B fp16.

### Orchestration, Energy Accounting, Observability, Thermal/DVFS, and Failure Recovery
- [ ] M0 GATE: run llama-batched-bench -npl 1..128 on the fp16 gemma path on op15 and op12; confirm continuous-batched decode with separate KV + block-diagonal mask runs (reproduce/triage the npl=2 NPU hang) before building anything else.
- [ ] Build the phone-agent daemon: control TCP socket, 500 ms heartbeat carrying seq/last-graph-ts/thermal/clocks/charge_counter/voltage/engine, and control verbs (set_governor, disable_charging, drain, reprovision-hash-check); launch via adb, colocate with the existing RPC server binary.
- [ ] Implement the host control-plane daemon: per-hop deadline watchdog (deadline = p99 x 3) that abandons the RPC socket on stall, marks stage DOWN, and triggers agent RPC-process refork to clear wedged HTP0 state.
- [ ] Implement DEGRADE-to-server-only: keep phone-assigned layers loadable on demand on the A6000 so failover is a routing change, not a reload; prove it by killing a phone RPC process mid-decode and observing continuity.
- [ ] Build the energy aggregator: nvidia-smi 200 ms stream (minus ~25 W idle) + phone coulomb slope (disable charging, charge_counter/voltage @1 Hz, ignore current_now) + authoritative token count from the sampler; emit per-window rows with fleet J/tok, per-stage shares, clock trace, and delta vs 0.15 baseline.
- [ ] Implement closed-loop admission/backpressure at 8 fps: measure per-stage EWMA service time, hold at most K_pipe in-flight prefills via a credit window, shed to server-only on queue/thermal limits; keep decode batch occupancy in [64,128] as an energy control variable.
- [ ] Instrument distributed-KV eviction: broadcast evictions to all stages, reconcile per-stage kv_pages every heartbeat, alarm on divergence.
- [ ] Build observability: Prometheus text + JSONL trace, per-step bubble Gantt (server/OP15-NPU/OP15-GPU/OP12-NPU/OP12-GPU/head), hop_latency for the OP15->OP12-via-host bounce, batch occupancy, prefill/decode tokens/s; cross-validate live per-stage decode t/s against standalone llama-batched-bench JSON for the same npl.
- [ ] Implement per-phase DVFS policy: performance/race-to-idle governor for the prefill(NPU) lane, no-downclock for the decode(GPU) lane; nudge DDR via sustained demand (cannot pin), tag every energy window with clock_mhz, and use within-run relative comparison for all energy claims.
- [ ] Implement thermal-throttle routing: on warn threshold stop admitting new prefill to that phone and drain; on hard throttle DEGRADE the stage and re-add on cool-down; never fight the vendor thermal governor.

### Phased Milestones, De-Risking Spikes & Repo Layout
- [ ] Run gguf_dump --no-tensors on the real 12B fp16 file; record block_count, embedding_length, feed_forward_length, head_count_kv, key/value_length, sliding_window_pattern, and presence of per-layer-input embeddings; write to research_dev/design/designA_pipeline.md.
- [ ] S1: write research_dev/spikes/s1_batched_decode/{run_op12.sh,run_op15.sh} driving llama-batched-bench with --device HTP0 and GPUOpenCL at -npl 1,2,4,8,16; also dump backend async/events caps; parse into RESULTS.md with hang/tok-s/J-tok table; decide NPU-decode vs GPU-only-decode fallback.
- [ ] S2: write research_dev/spikes/s2_shared_weights/shared_mmap_spike.cpp proving one mmap'd fastRPC dmabuf fd is read by Hexagon and OpenCL with matching MUL_MAT output and ~1x RSS; verify via /proc/self/smaps.
- [ ] M0: build the power harness (research_dev/power/harness.py: nvml for A6000, coulomb charge_counter-slope for phones); reproduce 12B fp16 A6000 baseline (0.15 J/tok @ batch256, OOM@320).
- [ ] Write research_dev/provision/shard_layers.py to slice the 12B gguf into per-stage shard files (+ PLE-table slice + layer-assignment manifest) and push_shards.sh to pre-provision phones (NO runtime weight RPC).
- [ ] M1: write research_dev/orchestrator/launch_pipeline.py to start rpc-server -d on op12 and wire the server with --rpc; assign 1 layer to op12; verify greedy output is token-identical to M0; verify PLE token-id plumbing if applicable.
- [ ] M2: extend topology to 3stage_baseline.yml (op15=2 layers, op12=1); run 512-tok prefill through server->op15->op12->server; confirm correctness and log per-hop activation bytes and the async/events pipeline-overlap decision.
- [ ] M3: enable continuous batching + batched decode at batch 64 then 128 through the 3-stage pipeline bounded by S1 max-npl; validate no hang, correctness vs single-device, and log aggregate tok/s + J/tok.
- [ ] M4: implement per-phone dual-engine launcher routing prefill->Hexagon and decode->Adreno on the S2 shared dmabuf; reproduce zero-interference co-schedule and one-copy RAM.
- [ ] M5: run the layers-on-phone rebalance sweep (2+1 -> more, bounded by phone RAM + 5-12W thermal), produce the layers vs J/tok vs tok/s Pareto vs the A6000 M0 floor, and write an honest net-energy verdict in research_dev/bench/m5_energy/.
