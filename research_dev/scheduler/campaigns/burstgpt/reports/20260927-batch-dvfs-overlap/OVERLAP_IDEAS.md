# Overlapping computation and memory transfer on the phone-assisted desktop (ideas, 2026-09-27)

Grounded in the tp2 audit (Gemma token 406 ms = OP15 round trip 193 | CPU attention between phone FFNs 116 |
rest 100; Qwen 516 = 198 | 117 | 197 incl. the Pixel). Mechanism today: one ggml graph per decode step; every
phone-owned FFN layer is an input tensor (`ffn_phone_partial`) filled by the server's eval callback with a
**synchronous** USB/TCP RPC (`build_dense_ffn_split`, `src/llama-graph.cpp`). So the phone idles while the
desktop computes (52 % of the token), the desktop idles during the round trip (48 %), the GPU idles 90 %, and
the PCIe x8 link idles ~100 %. Bytes are conserved: the CPU must stream whatever does not fit in VRAM (11 GB
weights) or phone RAM (8.5-9.6 GB), ~3.6 GB per Gemma token at ~17-20 GB/s. Every idea below either fills an
idle link with transfer, or fills an idle device with another stream's compute.

| # | idea | what overlaps with what | est. token time (Gemma 406) | est. energy | effort | novelty |
|---|------|-------------------------|-----------------------------|-------------|--------|---------|
| M | PCIe weight streaming into the remote-FFN latency hole | H2D of CPU-resident weights ∥ the phone round trip | ~290-330 ms | CPU pkg −50..−70 % | high (ggml/CUDA) | high |
| A | two-stream ping-pong across the USB boundary | stream A's phone FFN ∥ stream B's desktop attention/GPU | **CORRECTED 09-27: ≈ plain batching** (2 tokens per 2·max(p,d)=426 ms vs batch-2 (p+d)(1+ε)=426 ms) and 2× the phone weight traffic | none over batching at B≤4 | high (server) | only for incompatible rows |
| E | speculative rows fill the free batch | draft tokens ∥ (nothing; uses row-invariant step cost) | same step, ~2 tokens/step | J/token ÷ ~2 for lone requests | medium | low (known) |
| D | queue-aware pre-staging of the next model | disk→RAM→VRAM/phone of model B ∥ decode tail of model A | switches 60 s → ~15 s | −10 kJ/run, −200 s waits | medium | medium |
| C | PCIe as an additive channel during CPU compute | H2D + GPU GEMV of a column share ∥ CPU GEMV of the rest | CPU-resident slice −28 % | CPU pkg −25 % | high | medium |
| F | streamed rows inside a multi-row call | row r+1 h2d ∥ row r compute; row r d2h ∥ row r+1 | −25..−30 ms/token at rows≥2 | small | low (worker) | low |
| B | fold attention into the phone round trip | replaces 24 RPCs by 1 (transfer disappears) | ~300 ms | CPU pkg −40 % | high + RAM-bound | medium |

## M. PCIe weight streaming scheduled into the phone round-trip holes  (flagship)
While the desktop waits ~8 ms per phone layer (24 times per token, 193 ms total), the GPU and its PCIe 4.0 x8
link (~12 GB/s with pinned memory) do nothing. 193 ms × 12 GB/s = 2.3 GB per token can cross to VRAM inside
those holes — exactly the size of the CPU's attention-weight stream (24 layers × 94 MB). Design:
- load the CPU-resident tensors into a **pinned** host buffer (`--no-mmap` gives `CUDA_Host` buffers already);
- a 2-3 GB VRAM ring (there are ~3 GB free at peak) receives the next layers' weights via `cudaMemcpyAsync` on a
  copy stream, issued by the eval callback *at the moment it starts a phone RPC* (it knows the graph order);
- when the graph reaches those layers, the GPU computes them (0.3 ms per attention layer, 1.2 ms per FFN layer)
  instead of the CPU (5 ms / 10.5 ms). Steady-state variant C keeps the copy stream busy during CPU compute
  too: ~400 ms × 12 GB/s ≈ 4.8 GB per token > the whole 3.6 GB CPU-resident set, so with a measured split
  (CPU ~1 GB, PCIe ~2.6 GB) the CPU nearly idles.
- Estimate: 406 → ~290-330 ms per token, CPU package −50..−70 %, GPU +≈1 W. RAM: the pinned copy replaces the
  page-cache copy (same bytes) but competes with the other model's cache → the 64 GB RAM upgrade helps here too.
- Risk: PCIe rate under DVFS, ring eviction policy when the phone is thermally excluded (then the CPU stream is
  8.5 GB larger and PCIe can hide only ~2.3-4.8 GB of it — still a win). Known relatives: FlexGen/DeepSpeed weight
  streaming (full offload, which loses here because 12 GB/s < CPU 17-20 GB/s); the twist is *hiding transfer
  under a remote-accelerator latency hole* and keeping the CPU as a co-streamer.

## A. Two-stream ping-pong across the USB boundary  (decode-time pipeline parallelism with the phone as a stage)
**Correction (2026-09-27, after checking the arithmetic):** with per-token stage times p (phone) and d (desktop),
two staggered streams deliver 2 tokens per 2·max(p, d) = 426 ms (Gemma: p 193, d 213), while one batch of two
rows delivers 2 tokens per (p + d)(1 + ε) ≈ 426 ms because both stages are bandwidth-bound and rows are nearly
free (phone +3-13 %, desktop step flat to B=4). 2·max(p,d) ≥ p+d always, so staggering never beats batching for
rows that CAN be batched, and it reads the phone weights twice per token pair. The claim "×1.8 on top of
batching" below is withdrawn. Staggering is worth it only for rows that cannot share a call (different
ffn_split_policy during a join transient, different layout generation, different model) or beyond the
bandwidth-bound regime (B ≳ 16-32), which this trace never reaches. Original text kept for the record:
Run two micro-batches on two `llama_context`s over one `llama_model` (weights shared; llama.cpp supports many
contexts per model), each in its own decode thread with one shared CPU threadpool. Context A blocks in the eval
callback on its layer-k phone RPC while context B computes attention k and its GPU layers; then they swap.
The phone stage (193 ms) and the desktop stage (213 ms) are balanced for Gemma, so 2 tokens come out every
~215-230 ms instead of 406 → ×1.8 tokens/s at the same ~75 W → J/token ÷ 1.8, *on top of* batching (each
micro-batch can itself carry 2 rows). The scheduler chooses the pairing and the offset from its per-model
per-layer evidence. This is 1F1B pipelining applied at token granularity over a USB accelerator; the novelty
is the placement of the pipeline boundary inside a layer (FFN remote, attention local) and scheduler-chosen
stage balance. Cost: server slot→context assignment, per-step token synchronisation, CUDA streams per context.

## E. Speculative rows to fill the free batch
A decode step costs the same at 1 or 4 rows (615-632 ms at B=4 vs 611 solo; phone calls +3-13 % at 2-3 rows).
For a lone request, fill the empty rows with draft tokens from the already-resident llama-3.2-1B (≈5 ms/token
on the GPU), verify 4 rows in one big-model step through the phone contract (`max_tokens=4`), keep the accepted
prefix. Output stays byte-identical to the big model. Typical acceptance 2-3 of 3 → J/token ÷ ~2 for single
requests, which is most of this trace. Known technique; the energy framing on a row-invariant heterogeneous path
is what makes it pay here.

## D. Queue-aware pre-staging of the next model during the current decode tail
Switches cost ~410 s (22 % of the run) because a 24-30 GB f16 file is faulted in through mmap at 400-500 MB/s and
30 GB of RAM cannot cache both models (the same load takes 3.6 s when cached). The dispatcher knows the next
model ≥ 100 s ahead (queue + calendar). Overlap: (1) a readahead thread with O_DIRECT/`readv` streams the next
model's CPU-resident tensors into a bounded RAM window during the current decode (NVMe ~2 GB/s vs 0.45 GB/s
faults); (2) spare VRAM (~3 GB) pre-receives the next model's first GPU layers; (3) the phone keeps one HTP
session as a staging slot for the next model's first shard (needs RAM: 3 × 3.2 GB used of ~10 GB). Expected
load 60 → ~15 s, −200 s of waiting, ~−10 kJ per run. Scheduler-driven speculative residency; the "prefix
re-provisioning" already in the tree is the phone half of this.

## C. PCIe as an additive memory channel during CPU compute
Column-split every CPU-resident weight matrix: the CPU streams its share from RAM (~17-20 GB/s measured) while
the GPU streams the complement over PCIe (~12 GB/s) and computes it. 20 + 12 = 32 GB/s effective for the
CPU-resident bytes → that slice −28 %. Usually done as full offload (loses); the additive split with a share
chosen from measured rates is the twist. Subsumed by M when M's ring is large enough.

## F. Streamed rows inside a multi-row phone call
With rows ≥ 2 the worker can start row 0 while row 1's 10 KB is still arriving and return row 0 while computing
row 1, hiding most of the 1.5 ms per call (37 ms per token, 9 %). Worker + transport only.

## B. Fold attention into the phone round trip
The phone already owns FFN of layers 0-23; giving it their attention weights (+94 MB/layer) and KV (~1 GB for
2,560 ctx × 2 slots) turns 24 RPCs into one per token and removes the CPU's 116 ms of attention: ≈300 ms/token.
This is the layer-split design that the 2026-07 review set aside (pipeline chain, hub power); RAM is the blocker
today (9.6 of ~10 GB used) unless the phone drops to ~19 layers. Kept for completeness.

## Checked and rejected
- Re-partitioning attention to the GPU (`--override-tensor`) does not help: VRAM bytes are conserved, so every
  attention layer moved in pushes a full layer out to the CPU; the CPU streams the same ~3.6 GB either way.
- Prefetching a layer's weights into the DSP's VTCM during the wait: VTCM is MBs, layers are hundreds of MB.
- Dual-engine (NPU+GPU) on the phone: same DRAM bus, measured 1.01×/0.91× in the campaign.
