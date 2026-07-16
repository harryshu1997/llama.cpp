# S6 Phone Operator-Energy Scheduler Screen -- PLAN

Energy is the PRIMARY metric; latency is secondary. This is a bounded screen, NOT a
production scheduler. No Gemma graph / KV / backend-scheduler / serving-path edits. The
pre-existing uncommitted edits in `layersplit.cpp`, `microop.cpp`, `oplayerprof.cpp`,
`ggml-hexagon.cpp` are preserved untouched.

## Precondition (gates everything): a valid phone-energy boundary

The primary metric (whole-phone J/work) requires a VALID power boundary via the
`research_dev/energy/` harness, whose validity gate is `usb_pinned_frac <= 5%`. Two ways:
- **USB-rail path:** Full battery + high-wattage PD/SUPERVOOC charger (cap > ~5-7 W draw)
  + WiFi-ADB -> `usb V x I ~= system power`, pin -> 0.
- **Battery-coulomb path:** unplugged + WiFi-ADB + battery run down off Full.

Only **op15 (CPH2749)** has root (power nodes are root-gated); **op12 (CPH2583) has no su**
-> op12 energy is impossible until rooted (and the task defers op12 until op15 passes).

## Step 1 -- Certify Efficient Kernels (ENERGY-INDEPENDENT; runnable now over USB)

One real Gemma SWA layer, resident weights, via `llama-phone-oplayerprof` (real single-layer
harness with built-in batched-vs-serial correctness). Confirm on OP15:
- HTP dense M={1,4,5,8,16,32,128,512}: `GGML_HEXAGON_PROFILE=2` shows HMX (`hmx-tiled`) for
  every eligible M>=5.
- HTP decode B={5,8,16,32}, C={512,1024}: fused `FLASH_ATTN_EXT` (single op), no fallback.
- GPU prefill T={16,64,256,512}: `cl_profiling.csv` confirms `kernel_gemm_xmem_f16_f32_os8`.
  xmem cache-off and cache-on in SEPARATE processes; cache-on first/repeated/final
  correctness; no CPU fallback; record prepack RAM.

## Step 2 -- Three operator islands (definitions)

- **D**: complete decode layer on HTP (attn/KV -> output proj -> FFN).
- **P**: complete prefill layer on GPU/xmem (independent request group).
- **F**: complete FFN island on HTP (RMS -> gate/up -> GEGLU -> down -> residual).
Never remotely schedule RMS/RoPE/ADD/GEGLU individually (island granularity only).

## Steps 3-5 -- ENERGY-DEPENDENT (BLOCKED until the boundary exists)

- **Step 3 static scheduling** (A: D solo, B: P solo, C: serialized, D: D||P concurrent,
  E: negative control D-on-GPU||P-on-HTP), genuine solo legs (dualengine labels
  insufficient), 5x 60 s windows/mode, whole-phone J per completed layer-token / FFN row.
  Cross-stream FFN batching F1-F4 (merge decode+prefill rows sharing model/layer/weights/
  dtype into one FFN island; scatter after -- NOT matrix splitting).
- **Step 4 minimal dynamic policy** (B_target={8,16,32}, Wmax={0,2,5,10} ms; HTP M>=5; GPU
  xmem-eligible T>=16; HTP steals prefill only when decode queue empty; never split one GEMM;
  idle-GPU valid if it lowers total J). Replay one bursty trace at 50/80/95% load.
- **Step 5 fleet energy** (CUDA_OPT vs CUDA+PHONE), persistent TCP_NODELAY, resident weights,
  double-buffered async, NO adb push/pull. Gross fleet J/completed token.

## Pass gates

HMX/xmem for all eligible dense calls; correctness no fallback; co-run wall <= 1.10x
max(true solo legs); each co-running leg slowdown <= 10%; phone-local J/work <= 0.90x
serialized baseline; gross fleet J/token <= 0.90x optimized CUDA-only; p95 latency <= 1.10x
CUDA-only; thermal throughput loss <= 10%. A faster result without energy savings is
PERF_ONLY.

## Stop rules

Stop before networking if neither merged FFN nor HTP/GPU concurrency saves >=10% phone
energy. Test OP12 only after OP15 passes. Do not resume row/column splitting or build a
production scheduler until the gross fleet gate passes.

## Disposition of THIS run

Step 1 executed on OP15 over USB (energy-independent). Steps 3-5 primary metric (phone
J/work) is BLOCKED: op15 USB rail pinned at the 500 mA / 2.5 W cap (`usb_pinned_frac ~ 1.0`),
no WiFi-ADB, op12 unrooted. STOP after Step 1 + the energy blocker; deliver the physical
enabler needed to resume.
