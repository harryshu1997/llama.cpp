# S6 Phone Operator-Energy Scheduler Screen -- RESULTS

Status: **Step 1 (Certify Efficient Kernels) COMPLETE and PASS** on OP15. **Steps 3-5 (the
energy screen -- the primary deliverable) BLOCKED** on a physical power boundary. Verdict +
required enabler at the bottom. Energy-independent, no source edits; pre-existing uncommitted
edits preserved untouched.

Repository cleanup note: the old `research_dev/energy/` harness is intentionally
not published. A later audit found its runtime flags and plugged-device validity
contract inconsistent. The physical energy verdict remains BLOCKED.

## 0. Headline -- the primary metric (phone J/work) is physically unmeasurable now

Energy is the PRIMARY metric of this spike. It requires a valid whole-phone power boundary
(`research_dev/energy/`, validity gate `usb_pinned_frac <= 5%`). Measured live on OP15 (root):

```text
usb/current_now = 497000 uA   input_current_limit = 500000 uA   -> 99.4% PINNED (idle already)
usb/voltage_now = 5.03 V   battery status = Charging   capacity = 99%
```

The dev-machine USB port is a 500 mA / 2.5 W SDP source; inference draws ~5-7 W, so the rail
pins at the cap and `usb V x I` clips -> `usb_pinned_frac ~ 1.0` -> **every energy record would
be `valid=false <usb_capped>`** (exactly the state the harness's own 2026-07-11 validation
flagged). The battery-coulomb fallback also fails: battery is 99% and Charging (no discharge).
And **op12 (CPH2583) has no root** -> its power nodes are unreadable regardless. WiFi-ADB is
not set up (both phones on `usb:5-3`/`usb:5-4`).

**=> No valid phone J/op is obtainable in the current setup. The energy pass-gates cannot be
evaluated.** See section 7 for the exact physical enabler.

## 1. Build / device identity

```text
Git revision: 933c722f6 (+ preserved uncommitted fused-FA / FA-toggle / per-tensor-share edits)
Binary: llama-phone-oplayerprof, rebuilt from current source 2026-07-11 (build-snapdragon,
        docker snapdragon-toolchain-hostgcc:v0.3, NDK r28b, arm64-android-31,
        GGML_HEXAGON=ON GGML_OPENCL=ON GGML_OPENCL_PROFILING=ON), deployed /data/local/tmp/s6cert.
Device: OP15 = OnePlus 15 CPH2749 / SM8850 / Hexagon v81 (HTP0: 8thr/8hvx/1hmx/8MB vtcm) +
        Adreno 840 (GPUOpenCL). Root via Magisk. ~10 GB usable RAM.
Shard: 12b-f16-mid-2-3.gguf (one real Gemma-4 12B SWA layer, blk.2, F16, resident weights).
```

## 2. Step 1a -- HTP dense: HMX certification (GGML_HEXAGON_PROFILE=2)

Per-op unit selection read from the profiler (`hmx-tiled` vs `hvx-tiled`), real SWA layer:

| M (batch) | 1 | 4 | 5 | 8 | 16 | 32 | 128 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|
| MUL_MAT unit | hvx | hvx | **hmx** | **hmx** | **hmx** | **hmx** | **hmx** | **hmx** | **hmx** |

**PASS: every eligible M>=5 uses HMX** (`hmx-tiled`, VTCM ~8 MB); M=1,4 use HVX (below the
HMX gate, expected). M=5-32 via decode GEMMs; M=128,512,1024 via the prefill GEMMs. Every op
placed on HTP0 -- **no CPU fallback**.

## 3. Step 1b -- HTP decode: fused FLASH_ATTN_EXT

Decode profile shows a single fused op
`FLASH_ATTN_EXT | Qcur_pos x cache_k x cache_v x mask -> __fattn__` reading the HTP-owned KV
(`SET_ROWS` into cache_k/cache_v), on HTP0, at B={8,16,32}, C={512,1024}. **PASS: fused FA, no
CPU fallback.** (This is the native HMX FA, not the isolated-op artifact that made S5's
test-backend-ops HTP FA unreliable.)

## 4. Step 1c -- GPU prefill: xmem certification (cache-off / cache-on, separate processes)

`cl_profiling.csv` kernel-name confirmation (real SWA layer, GPUOpenCL,
`GGML_OPENCL_ADRENO_XMEM_GEMM=1`):

| run | xmem GEMM kernel | prepack dispatches (timed) | rel_L2 (B16 vs serial) | argmax mm | free_mb | wall p50 (C=64) |
|---|---|---|--:|--:|--:|--:|
| xmem cache-OFF | `kernel_gemm_xmem_f16_f32_os8` (126x) | every call | 0.0186 | 0 | 8432 (C16) | 83.2 ms |
| xmem cache-ON | `kernel_gemm_xmem_f16_f32_os8` (189x) | **0** (cache hit) | 0.0186 | 0 | 7993 | 22.7 ms |
| STOCK (no xmem) | (stock GEMV/GEMM) | n/a | **2.16e-05** | 0 | 8418 | 87.5 ms |

- **xmem GEMM confirmed** (`kernel_gemm_xmem_f16_f32_os8` + helpers `adreno_xmem_prepack_weight_f16`,
  `_pack_src_f32`, `_store_dst_f32`). **No CPU-compute fallback** (only the logits/output buffer
  sits on CPU, which is normal). SWA `flash_attn_f32_f16` also runs on GPU.
- **Prepack cache is SAFE with resident weights**: cache-ON does **0 prepack redispatches**
  during timed rounds (prepacked once, reused), and rel_L2 is identical to cache-OFF and stable
  across 14/10 rounds -> first/repeated/final correctness hold. This is the exact distinction
  from the S3 stale-prepack hazard (which recycled slice-buffer handles); resident weights have
  stable `(cl_mem, offset)` keys.
- **Accuracy caveat (flag, not a hard fail here):** the xmem os8 image-f16 GEMM deviates
  **rel_L2 = 0.0186 (1.86%)** from stock (2.16e-05). argmax is stable (token decision unchanged
  at 1 layer), but 1.86%/layer is NOT bit-accurate and must be validated end-to-end (48 layers)
  before production use.
- **Prepack RAM ~= 0.4 GB per SWA layer** (cache-ON retains ~408 MB more than cache-OFF at
  matched C -- a second image-form copy of the layer's GEMM weights). Material at scale.
- Bonus (latency, not the metric): xmem cache-ON is ~3.8x faster than stock for this
  decode+prefill (22.7 vs 87.5 ms) -- consistent with S5's finding that stock Adreno GEMM is
  non-viable at batch, and that xmem is the only credible GPU-prefill path.

## 5. Genuine solo legs (Step 3 A/B, timing only -- energy BLOCKED)

Current-source, fused-graph, real single SWA layer (not the old dualengine labels):

| leg | backend | B / C | wall p50 (ms/layer) | rel_L2 | free_mb |
|---|---|---|--:|--:|--:|
| **D solo** (decode layer) | HTP0 | 16 / 512 | **25.2** | 5.1e-4 | 6555 |
| **D solo** | HTP0 | 32 / 512 | **33.5** | 5.1e-4 | 2578 |
| **P solo** (prefill+decode, xmem cache-on) | GPUOpenCL | 16 / 256 | **31.9** | 0.0186 | 6620 |

D-solo HTP B32 = 33.5 ms cross-validates S3 H0 (op15 34.3 ms) -> the fused-graph numbers are
trustworthy (unlike S5's isolated-op HTP FA). Serialized C = D+P; concurrent D||P and the
negative control were NOT run (they need the dualengine/island harness and their metric is
energy). Co-run zero-interference is already established (S3 H0: HTP-decode leg load-insensitive
26.2 vs 24.9 ms under 4x GPU load; dualengine 1.76x/1.92x overlap).

## 6. Merged-FFN (F1-F4) compute basis -- from HMX flatness (energy BLOCKED)

HTP FFN gate/up GEMM (HMX) is near-FLAT in M (S5, op15 HTP0): M5=3696, M16=3749, M32=3756,
M128=4055, M512=5163 us -- only **+9.7% from M5 to M128** for 25x the rows. Therefore cross-
stream FFN batching is compute-cheaper:

| mode | FFN gate/up compute (us, est.) | vs separate |
|---|--:|---|
| F1: decode M32 + prefill M96 SEPARATE | 3756 + ~4020 = ~7776 | baseline |
| F2: merged M128 | 4055 | **~1.9x cheaper** |
| F3: decode M32 + prefill M480 SEPARATE | 3756 + ~5100 = ~8856 | baseline |
| F4: merged M512 | 5163 | **~1.7x cheaper** |

So merged-FFN has a strong COMPUTE (hence likely energy, at ~const HMX power) basis: ~40-48%
less FFN GEMM work. **But the actual joules are BLOCKED** -- this is a projection, not a measured
J/FFN-row. It DOES motivate the merged-FFN island as the first thing to test once energy is
measurable.

## 7. Pass gates

| gate | result |
|---|---|
| all eligible phone dense calls use HMX/xmem | **PASS** (HMX M>=5; xmem GEMM confirmed) |
| correctness, no CPU fallback | **PASS** for HTP (rel_L2<=5e-4) and stock GPU (2e-5); xmem **argmax-stable but 1.86% rel_L2** vs stock (flagged, needs 48-layer end-to-end check) |
| co-run wall <= 1.10x max(true solo legs) | **NOT EVALUATED** (concurrent run deferred; metric is energy) |
| each co-running leg slowdown <= 10% | prior evidence PASS (S3 zero-interference); not re-run |
| phone-local J/work <= 0.90x serialized baseline | **BLOCKED** (no valid phone energy) |
| gross fleet J/token <= 0.90x optimized CUDA-only | **BLOCKED** |
| p95 latency <= 1.10x CUDA-only | **BLOCKED** (fleet run gated on energy) |
| thermal throughput loss <= 10% | not measured (energy-run scope) |

## 8. Verdict -- STOP; deliver the physical enabler

**Step 1 kernels CERTIFIED (PASS)** on OP15: HMX for all M>=5, fused FLASH_ATTN_EXT, xmem
`kernel_gemm_xmem_f16_f32_os8` with a safe resident-weight prepack cache, no CPU fallback,
prepack RAM ~0.4 GB/layer. One accuracy flag: xmem os8 is ~1.86% rel_L2 vs stock (argmax-stable
at 1 layer). The merged-FFN energy thesis has a strong compute basis (HMX near-flat in M ->
~1.7-1.9x less FFN work when cross-stream rows are merged).

**The energy screen (Steps 3-5) is BLOCKED** and cannot start until a valid phone-power
boundary exists. This is NOT a code or method gap -- it is physical. To unblock, provide EITHER:

1. **USB-rail path (easiest):** connect OP15 to a **high-wattage PD/SUPERVOOC charger** (cap
   comfortably above ~7 W) with a **full battery**, and switch to **WiFi-ADB** -> `usb V x I`
   tracks system power and `usb_pinned_frac -> 0`. (Current port is 500 mA / 2.5 W -> pinned.)
2. **Battery-coulomb path:** **unplug** OP15, connect via **WiFi-ADB**, and run the battery
   down off Full so the gauge integrates (`bat_delta_w`).

For OP12 energy at all: **root OP12** first (it returns `no su`; power nodes are root-gated).
Per the task, OP12 is tested only after OP15 passes.

Per the stop rule ("stop before networking if neither merged FFN nor HTP/GPU concurrency saves
>=10% phone energy") -- energy is unmeasurable, so I STOP before the static-energy, dynamic-
policy, and fleet-networking steps. No production scheduler and no row/column splitting are
resumed. Once the boundary is in place, resume at Step 3 (D/P solo + D||P energy, then merged
FFN F1-F4) on OP15.

## Raw artifacts (scratchpad/s6_energy_scheduler/cert/)

`cert_hmx.jsonl` (HMX M-sweep + correctness), `cert_fa.jsonl` (FA at C=512/1024),
`cert_xmem_off.jsonl` / `cert_xmem_on.jsonl` / `cert_stock.jsonl` (xmem vs stock),
`cert_dsolo.jsonl` (genuine HTP D-solo legs), `cl_profiling.csv` (kernel-name confirmation),
`hmx_profile_units.txt` (per-op HMX/HVX lines), `usb_pinned_proof.txt` (energy-blocker proof).
