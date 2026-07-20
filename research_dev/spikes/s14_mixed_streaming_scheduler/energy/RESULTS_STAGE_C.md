# S14 energy Stage C: mixed-workload A6000 GPU-board energy vs offload fraction

Status: DERIVED 2026-07-17 from MEASURED Stage A (A6000 NVML) + MEASURED Stage B
(OP15/HTP0) + the real mix-v1 trace. A6000 `GPU_BOARD` decode energy ONLY; phone
energy UNKNOWN. Overlap ceiling (idle-wait not modelled). No commit.

    result   stagec_result.json  (schema s14-stage-c-mixed-gpu-energy-v1)
    driver   stagec_mixed_energy.py
    inputs   stage_a_k8_result.json + stageb_result.json + fixtures/mix_v1.trace.jsonl (all sha256-pinned in the result)

## What Stage C does

Folds Stage A's MEASURED per-token tail energies onto the real mix-v1 workload at
the deepest phone-FEASIBLE head depth Stage B certified (`k* = [0,8)`), and reports
the mixed A6000 GPU-board decode energy as a function of how much of the decode is
offloaded to the phone head. Because Stage A showed A6000 power is flat (297 W),
energy scales with GPU-seconds = layers run, so:

    E_gpu(f) = total_tokens * [ f*e_tail(k*) + (1-f)*e_full ]
    saving(f) = f * (1 - e_tail(k*)/e_full) = f * s(k*)

with MEASURED `e_full = 859.9`, `e_tail[8,48) = 730.0` mJ/tok, `s(8) = 15.1%`.

## Result

mix-v1 = 21,595 decode tokens; generation (api + conversation) = 87.9%; rag_qa =
12.1%. Server-only baseline = 18,570 J.

| offload token fraction f | A6000 GPU-board energy | saving vs server-only |
|---|---:|---:|
| 0.000 (server-only) | 18,570 J | 0.0% |
| 0.250 | 17,869 J | 3.8% |
| 0.500 | 17,167 J | 7.6% |
| **0.879 (all generation -> phone `[0,8)`)** | **16,103 J** | **13.3%** |
| 1.000 (if rag_qa also offloads) | 15,765 J | 15.1% |

- **Realisable mixed-workload A6000 GPU-board saving = 13.3%** when all generation
  decode is routed to the phone head `[0,8)` (the deepest single-phone-feasible
  island). The remaining 12.1% (rag_qa) stays on the server; if its generation
  phase also used the head, the ceiling is `s(8) = 15.1%`.
- This is BELOW the 22.8% single-token ceiling because `[0,12)` DSP-aborts on one
  phone (Stage B). Reaching 22.8% needs the two-phone split (op15 `[0,8)` + op12
  `[8,12)`) -- Stage D.

## SLO / relaxed-latency context

The phone route is much slower per request than server-only: Stage B measured the
phone head `[0,8)` at ~934 ms and the server tail `[8,48)` at ~259 ms, so the
serial phone route is ~1193 ms/request vs ~292 ms server-only. The 13.3% saving is
therefore realisable only at a RELAXED SLO (as the user authorised: "relax on
latency to save total energy"), and only as an OVERLAP ceiling -- it assumes the
A6000 stays busy with independent work while the phone runs the head. If the A6000
idles waiting for the phone, it still burns ~297 W and the saving evaporates (the
S11-E0 +53% serial-route failure). A true concurrent overlap measurement is Stage D.

## Honest scope

- A6000 `GPU_BOARD` decode energy ONLY. Phone/USB/total-wall UNKNOWN.
- Overlap ceiling; idle-wait NOT modelled.
- Offload fraction is a workload/scheduler knob; the achievable f at a given SLO is
  bounded by the Stage B phone-route latency. Stage C reports the energy-vs-f curve
  and the measured route latency; wiring the deep `[0,8)` island into the reducer's
  routing at a swept deadline (to derive f(SLO) endogenously) is the CP2/CP3 step.

## Reproduce

    cd research_dev/spikes/s14_mixed_streaming_scheduler/energy
    /usr/bin/python3 stagec_mixed_energy.py
