# S14 energy Stage B: deep-head phone island placement + feasibility ceiling

Status: MEASURED 2026-07-17 on real hardware (OP15 = OnePlus15, SM8850, Hexagon
v81, HTP0; A6000 host tail). OP15/HTP0 island feasibility + placement + phone
stage latency ONLY; phone ENERGY is UNKNOWN (no phone power instrument). No commit.

    result   stageb_result.json      (schema s14-stage-b-deep-head-placement-v1)
    driver   stageb_headcert.py
    binary   phone ls-s14/llama-layersplit sha256 ba6ab706... (v81 skel libggml-htp-v81.so)

## Question

Stage A showed the A6000 saves ~0.9x(k/48) of decode energy when the phone runs
the head `[0,k)` instead of the server (11.3% at k=6, 22.8% at k=12). That saving
is real ONLY if a phone can actually HOLD + RUN `[0,k)`. Stage B measures, for
gemma-4-12B-it-f16 on OP15/HTP0: does the head load resident, run on the NPU with
NO fallback, produce the correct tokens, and at what stage latency -- as a
function of depth k?

Method: the phone runs `--mode stagenet` (head `[0,k)`, `LLAMA_LAYER_END=k`,
`LAYERSPLIT_PLACEMENT_CERT=1`) on HTP0; the A6000 host `--mode pipedriver` runs
the tail `[k,48)` and drives incremental single-stream decode over adb/USB,
emitting `stage_a_us` (phone head) per request; a host mono baseline gives the
reference tokens. Reuses existing binaries UNCHANGED; does NOT touch the frozen
S11-E0 harness.

## Result (batch=1 single-stream, FA on, 12 requests + 3 warmups)

| head `[0,k)` | placement | HTP0 compute / CPU | missing_buf | HTP0 weights | +CPU_Mapped embd | resident | phone head p50 | server tail p50 | tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `[0,2)` | **SCHEDULED_PLACEMENT_OK** | 7080 / 120 | 0 | 855 MiB | 1920 MiB | 2775 MiB | 442.6 ms | 291.9 ms | match |
| `[0,6)` | **SCHEDULED_PLACEMENT_OK** | 20880 / 120 | 0 | 2599 MiB | 1920 MiB | 4519 MiB | 782.7 ms | 269.8 ms | match |
| `[0,8)` | **SCHEDULED_PLACEMENT_OK** | 27840 / 120 | 0 | 3454 MiB | 1920 MiB | 5374 MiB | 934.1 ms | 258.6 ms | match |
| `[0,10)` | **DSP_QUEUE_ABORT** | -- | -- | 4309 MiB (loaded) | 1920 MiB | -- | -- | -- | -- |
| `[0,12)` | **DSP_QUEUE_ABORT** | -- | -- | 5198 MiB (loaded) | 1920 MiB | -- | -- | -- | -- |

- **`[0,2)`, `[0,6)`, `[0,8)` are fully certified**: all compute on HTP0 except
  ~120 nodes (the f16 `token_embd` GET_ROWS, a declared CPU_Mapped exception),
  `missing_buffer_compute_nodes=0` (no silent fallback), and the head+tail
  pipeline reproduces the mono baseline token-for-token (`token_match=True`).
- **Max feasible single-phone head is `[0,8)`.** `[0,10)` and `[0,12)` LOAD fine
  (the weight buffers allocate: 4309 / 5198 MiB) but ABORT on the FIRST forward
  with `ggml-hex: dspqueue_read failed: 0x2e` in `flush_pending`. The boundary is
  a **~4 GiB (2^32 B) cap on a single HTP weight buffer**: `[0,8)` = 3454 MiB
  passes, `[0,10)` = 4309 MiB fails. Confirmed independently in `--mode head`.
- Phone head latency grows with depth (443 -> 783 -> 934 ms, single-stream over
  USB, an UPPER bound incl. relay); server tail shrinks (292 -> 270 -> 259 ms) as
  fewer layers remain. gemma-4-12B has global attention at layers 5, 11, ...
  (SWA period 6); the certified cuts cross layer 5 correctly.

## What this means for the energy story (ties to Stage A)

- The **realisable** single-phone GPU-board saving is bounded by the deepest
  feasible head `[0,8)` = **15.1% MEASURED** (Stage A `stage_a_k8_result.json`:
  730.0 vs 859.9 mJ/tok, HBM -3454 MiB == the HTP0 weight buffer), NOT the 22.8%
  ceiling. `[0,12)` (22.8%) is blocked by the DSP single-buffer cap on ONE phone.
  Folded onto the real mix-v1 workload (Stage C), routing all generation
  (87.9% of decode) to the phone head realises **13.3%** mixed GPU-board saving.
- To reach 22.8% the head must be **split across both phones** (e.g. op15 `[0,8)`
  + op12 `[8,12)`, a 4-layer ~1.8 GiB mid slice well under the cap) or the
  Hexagon single-buffer limit lifted (multi-buffer weights). op12 is available;
  this is the Stage D two-phone path.

## Honest scope / caveats

- OP15/HTP0 feasibility + placement + phone stage latency ONLY. Phone/USB/total
  energy is UNKNOWN (no phone power instrument on this host).
- `token_embd` (f16, 1920 MiB) is CPU_Mapped (declared GET_ROWS exception), not a
  fallback failure. `SCHEDULED_PLACEMENT_OK` + `missing_buffer=0` is the
  no-fallback certificate; the 120 CPU compute nodes are this embd lookup.
- Single-stream (batch=1), matching the frozen S11-E0 `GREEDY_SINGLE_STREAM`
  method. Batched multi-seq HTP decode is a separate known hang (S1) at k>=6.
- `GGML_DECODE_NO_FA=1` forces the non-FA decode path, which HANGS the
  global-attention layers (>=5) on v81 -- so flash attention is left ON (v81 is
  correct with FA per prior validation). This was the fix for the initial k>=6
  stagenet hang.

## Reproduce

    cd research_dev/spikes/s14_mixed_streaming_scheduler/energy
    /usr/bin/python3 stageb_headcert.py --k-list 2,6,8,10,12 \
      --requests 12 --warmups 3 --batch 1 --n-gen 8
