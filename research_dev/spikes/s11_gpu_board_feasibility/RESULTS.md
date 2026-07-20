# S11 GPU-Board Feasibility Bound

Status: `TIMING_BOUND_COMPLETE; PHYSICAL_ENERGY_NOT_RUN`

## Purpose

This is an analytical screen for the S11-E0 measurement. It does not replace
NVML acquisition and makes no physical energy claim. It asks what A6000 tail
and phone-wait power ratios would be required for a 10 percent board-energy
reduction, using only an exact prior timing trace.

The two-state model is:

~~~text
E_treatment / E_control
  = r_tail * treatment_tail_us / control_wall_us
  + r_gap  * treatment_gap_us  / control_wall_us
~~~

`r_tail` and `r_gap` are sensitivities relative to average control-board power,
not measured values.

## Existing B=8 Trace

Source:
`scratchpad/s11_runner_v2_final_r2_b8_20260716/summary.json`

SHA-256:
`3a9bc141f740b63b06b6b6741e90bb9af8b75eb8014993fbe6c69a2ac8306ed2`

~~~text
control wall:        204.764 ms
treatment tail:      202.954 ms
treatment phone:     246.687 ms
timing residual:       0.015 ms
treatment gap:       246.702 ms  (phone plus residual)
treatment wall:      449.656 ms
tail/control time:   0.991161
gap/control time:    1.204811
~~~

At equal treatment-tail and control power, the maximum allowed gap-power ratio
is negative (`-0.075664`). Therefore, zero A6000 power during the entire phone
interval would still not reach the 10 percent gate if tail power were unchanged.

Required maximum tail-power ratios are:

| Gap power / control power | Maximum tail power / control power |
|---:|---:|
| 0.00 | 0.9080 |
| 0.05 | 0.8472 |
| 0.10 | 0.7865 |
| 0.15 | 0.7257 |
| 0.20 | 0.6649 |

Thus, with a gap drawing 10 percent of control-board power, the tail must draw
no more than about 78.6 percent of control power. Offloading two of 48 layers
does not establish such a reduction. The physical run must measure it.

This makes S11-E0 a strong falsification test. A failure would direct the next
mechanism toward a larger phone island or scheduler-created A6000 batch/power
interval, rather than further tuning the same two-layer placement.

## Verification

~~~sh
PYTHONDONTWRITEBYTECODE=1 python3 \
  research_dev/spikes/s11_gpu_board_feasibility/test_analyze.py

PYTHONDONTWRITEBYTECODE=1 python3 \
  research_dev/spikes/s11_gpu_board_feasibility/analyze.py \
  --summary scratchpad/s11_runner_v2_final_r2_b8_20260716/summary.json
~~~

The test suite passes 6/6. It covers the recorded bound, exact threshold
closure, schema/route/aggregate identity, explicit batch-index matching,
invalid timing, inexact work, and invalid sensitivity inputs.
