# S14 energy Stage A: A6000 GPU-board energy decomposition

Status: MEASURED 2026-07-17 on real hardware (1x RTX A6000, NVML). Selected-A6000
`GPU_BOARD` energy only; phone/USB/total-wall UNKNOWN. Diagnostic, not a
total-system claim. No commit.

    result   stage_a_result.json  (sha256 0a7d4c5b...)
    driver   stage_a_gpu_board.py (sha256 218d8873...)

## Question

How much A6000 GPU-board energy does offloading the head `[0,k)` of Gemma-4-12B
decode to a phone save, at equal decoded work? Measured by running the A6000 on
the FULL model `[0,48)` vs only the TAIL `[k,48)` (partial load,
`LLAMA_LAYER_START=k`, `layersplit --mode tailbench`), integrating NVML power
over the timed decode window (ZOH), matched work, rotated repeats.

## Result (batch 16, 24,000 tok/run, 3 rotated repeats each)

| A6000 runs | mJ/token (repeats) | avg power | util | HBM | GPU energy saved |
|---|---|---:|---:|---:|---:|
| full `[0,48)` | 859.94 / 860.06 | 297 W | 99% | 22,713 MiB | baseline |
| tail `[6,48)` (offload `[0,6)`) | 763.06 / 763.04 | 297 W | 99% | 20,114 MiB | **11.3%** |
| tail `[12,48)` (offload `[0,12)`) | 663.86 / 663.46 / 663.04 | 297 W | 100% | 17,515 MiB | **22.8%** |

- Repeats agree within ~0.1 percent; power is flat (297 W, near the 300 W cap)
  and utilisation stays 99-100 percent across all conditions.
- The saving is therefore purely **fewer GPU-seconds** (36 vs 48 layers), not
  throttling. Saved fraction ~= 0.9 x layer fraction offloaded (the 0.9 is the
  fixed tail overhead the A6000 still runs: lm_head, final norm, embedding).
- HBM relief scales with the offloaded weights: 5.2 GB freed at `[0,12)`.

## Mechanism and honest scope

- This is the energy the A6000 does NOT spend because it runs `[k,48)` instead of
  `[0,48)`. It is real ONLY if (a) a phone actually produces the `[0,k)` output
  (feasibility = Stage B, not yet run) and (b) the A6000 stays BUSY during the
  phone's head compute (overlap). The measurement assumes saturation
  (`gap_ms=0`), which models perfect overlap.
- **Idle-wait energy is NOT modelled.** If the A6000 idles waiting for a phone
  with no other work, it still burns ~297 W -- the S11-E0 serial-route failure
  (+53 percent board energy). The 22.8 percent figure is the CEILING achievable
  under overlap, not a guaranteed realised saving.
- Selected-A6000 `GPU_BOARD` only. Phone energy, USB/VBUS, and total-wall are
  UNKNOWN (no instrument on this host). Per NEXT_PLAN section 11 this is a
  mechanism diagnostic; it cannot satisfy the total-wall gate and does not reuse
  the frozen S11-E0 cohort.

## What this unblocks / next

- Stage B: confirm a phone can hold + run `[0,12)` (~7.4 GB, fits OP15 ~10 GB) at
  a real stage latency with a no-fallback placement certificate. That turns the
  ceiling into a realisable saving and makes the deeper island dispatch-eligible.
- Stage C: fold the measured tail energies into the CP1 mixed runtime so the
  mixed-workload A6000 GPU-board energy is reported vs offload fraction at a
  relaxed SLO.

## Reproduce

    cd research_dev/spikes/s14_mixed_streaming_scheduler/energy
    /usr/bin/python3 stage_a_gpu_board.py --k-list 0,6,12 --repeats 3 --steps 1500 --batch 16
