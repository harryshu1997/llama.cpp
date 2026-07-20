# S14 energy Stage D: measured phone||server overlap -- clean, but throughput-bounded

Status: MEASURED 2026-07-17 on real hardware (two A6000 boards + OP15/HTP0). A6000
GPU_BOARD energy + throughput ONLY; phone energy UNKNOWN. No commit.

    result   stage_d_result.json  (schema s14-stage-d-overlap-v1)
    driver   stage_d_overlap.py

## Question

Stages A/B/C assumed the phone runs CONCURRENTLY with a busy server at no cost.
Stage D measures that with two live processes:
  - process S (GPU0): a saturated full-model decode backlog = "server busy on other task";
  - process P (phone): OP15 runs `[0,8)` heads continuously, tail on GPU1 so GPU0's OWN
    workload is byte-identical control vs treatment (isolating pure concurrency cost).

## Result

| GPU0 (measured board) | energy/token | throughput | power | util |
|---|---:|---:|---:|---:|
| control -- phone IDLE | 799.1 mJ/tok | 369 tok/s | 294 W | 98% |
| treatment -- phone `[0,8)` decoding concurrently | 802.8 mJ/tok | 367 tok/s | 295 W | 98% |

- **Overlap is CLEAN.** With the phone pipeline live (loading, USB relay, host CPU,
  GPU1 tail all active), GPU0's own work changes by **-0.3% throughput / +0.46%
  energy per token** -- both inside NVML board-average noise (+/-5 W). The phone runs
  alongside a saturated server essentially for free (no measurable interference).
  This matches the same-phone cross-engine result but now across the USB boundary.
- **But one phone is throughput-bounded.** The phone sustains `[0,8)` heads at
  **8.2 tok/s** single-stream (measured concurrently, token-correct) vs the A6000's
  **369 tok/s**. So one phone carries only **2.2%** of the A6000's decode rate, and
  the REALISED GPU-board saving at a saturated server is **f_sat * s(8) = 0.34%**.

## What this means for "do we have savings?"

- The overlap MECHANISM the design needs is real and measured (clean, zero
  interference). The per-token saving is real (Stage A: 15.1% at `[0,8)`).
- But the Stage C accounting ceiling (13.3% on the mix, at offload fraction 0.879)
  is UNREACHABLE with one single-stream phone against a busy A6000: the phone would
  have to carry 87.9% of the decode tokens and it can carry ~2.2%. **Realised saving
  with one phone at a saturated server = ~0.34%.**
- To realise the depth-`[0,8)` saving at a saturated A6000 you need a FLEET:
  `369 / 8.2 ~= 45` phones for full offload (`~40` for the 87.9% generation share).
  This restates the S5 additive-capacity finding energetically: 2 phones ~= 1%.
- Levers that raise the per-phone share: a working BATCHED HTP head (multi-seq HTP
  decode currently hangs, S1) would multiply 8.2 tok/s; or a slower/edge server
  (lower tok/s denominator); or the light-load regime (below) -- all unmeasured here.

## Regimes and honesty

- This is the SATURATED-server regime (A6000 at 369 tok/s, the "busy" case the user
  named). The realised saving is throughput-bounded and small with one phone.
- The real mix-v1 is LIGHT (~21 generation tok/s, server ~3% utilised in CP1). There
  one phone (8.2 tok/s) carries a much larger share of the generation rate -- but the
  A6000 is then IDLE-dominated, a different energy regime that Stage A (saturated
  batch-16) did not measure, so the light-load net saving is NOT quantified here and
  must not be claimed.
- A6000 GPU_BOARD only; phone energy UNKNOWN. Idle-wait avoided by construction
  (GPU0 stays saturated), so this is the overlap mechanism at its most favourable.

## Addendum -- batching the phone head (does it close the throughput gap?)

Since the A6000 is ~45x faster single-stream, the obvious lever is to BATCH the
phone head (decode is memory-bandwidth-bound: batch B loads the 3.4 GiB of `[0,8)`
weights ONCE and emits B tokens). Measured `[0,8)` on OP15/HTP0, FA on
(`stage_d_batch_scaling.json`):

| batch | forward time | throughput | speedup |
|---:|---:|---:|---:|
| 1 | 118.5 ms | 8.4 tok/s | 1.0x |
| 2 | 130.2 ms | 15.4 tok/s | 1.8x |
| 4 | 137.0 ms | 29.2 tok/s | 3.5x |
| 8 | 204.1 ms | 39.2 tok/s | 4.7x |
| 16 | 240.7 ms | 66.5 tok/s | 7.9x |
| 32 | 285.1 ms | 112.2 tok/s | 13.4x |
| 48 | 323.7 ms | 148.3 tok/s | 17.7x |
| 64 | 382.9 ms | **167.1 tok/s** | **19.9x** |

(b48/b64 from the extended sweep `batch_sweep_ext_result.json`; b32 re-measured
there at 112.5 tok/s == the 112.2 above, so prompt length does not affect decode
throughput.) **B64 is the binary's ceiling**: `layersplit.cpp` rejects
`--driver-batch > 64`, and `n_ubatch` is capped at `min(n_batch, 512)` so the
prefill shape needs `prompt_tokens * batch <= 512` (B64 needs a <=8-token prompt).
B96/B128 would require a host+phone rebuild. Returns are diminishing (B32->B48->B64
= +32%, +13%) and forward time grows ~linearly with batch here, so the phone is now
compute-bound and near saturation -- 167 tok/s is close to the practical top for
`[0,8)` on this SoC.

- **Batched multi-seq HTP decode WORKS and is CORRECT with FA on** (the S1 hang was
  the `GGML_DECODE_NO_FA` path) and keeps scaling to batch 32 = **112 tok/s (13.4x)** --
  forward time only grew 204->285 ms from B8->B32, so the phone is NOT yet
  compute-saturated even at 32.
- **The "token mismatch" was a WRONG-REFERENCE artifact, not a bug.** Batched greedy
  decode is NOT bit-identical to single-stream: float non-associativity in batched
  attention/GEMM flips argmax at near-ties, and this is INHERENT -- the A6000 full
  model ALONE gives batch-size-dependent greedy paths (b2 drops 1 token, b4 drops 2).
  Proof the phone split is correct: (a) batch-1 is BIT-EXACT vs mono; (b) the phone
  route at batch 2 == the GPU full model at batch 2 EXACTLY; (c) output is coherent
  text at every batch. Greedy bit-matching is simply the wrong correctness test for
  batched inference; certify via activation error / coherence (batch-1 already does).
- **Overlap math (certified, B=32, 112 tok/s):** one phone carries 112/369 = **30%**
  of the A6000 (was 2.2% single-stream), realised board saving **~4.6%/phone**, and
  the two phones on hand batched carry ~61% -> **~9.1% realised**, approaching the
  13.3% mix ceiling. Batching cuts the fleet from ~40 phones to **~3** for full
  generation offload -- one phone still can't saturate the A6000 alone, but two
  batched phones now get most of the way.

## Reproduce

    cd research_dev/spikes/s14_mixed_streaming_scheduler/energy
    /usr/bin/python3 stage_d_overlap.py --batch 16 --steps 800 --phone-requests 60
    # batched phone head scaling:
    for B in 1 4 8; do /usr/bin/python3 stageb_headcert.py --k-list 8 --batch $B \
      --requests $B --warmups $B --n-gen 8; done
