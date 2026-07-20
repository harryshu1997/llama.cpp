# S11-E0 Selected-A6000 Board-Energy Diagnostic

Status: `ACQUISITION_NOT_RUN`

## 1. Question

For identical closed Gemma-4 12B work at a frozen p95 group-latency SLO, does
the resident OP15 `[0,2)` route reduce gross energy on one selected A6000 by at
least 10 percent after conservative NVML uncertainty?

This is a narrow falsification test of the current static route. It is not a
test of the S12 dynamic scheduler, the future WiFi-input/USB-result runtime, or
total-system energy.

## 2. Frozen Routes

- Control: `SERVER_ONLY`, full Gemma-4 12B F16 on the selected A6000.
- Treatment: `A0_OP15`, OP15 HTP owns layers `[0,2)` and the same A6000 owns
  layers `[2,48)`.
- Server residency is static for each complete timeline: `FULL_MODEL` for the
  control and `TAIL_ONLY` for the treatment.
- OP15 weights are resident before the paid window. No weight streaming is
  charged or claimed by this test.
- Both routes use the same executable sources, model identity, prompt, greedy
  settings, batch size, context limits, request count, and generated work.

The selected board UUID is
`GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf`. Every one of the 16 raw power
artifacts must bind this UUID; mixing boards is a hard error.

## 3. Measurement Boundary

Measured scope: `GPU_BOARD` for the selected A6000 only.

Paid window: the host timestamp immediately before releasing `GO` through
`DRIVER_DONE`. The driver has already emitted `DRIVER_READY`; model load, phone
weight provisioning, warmup, setup, and teardown are outside the window. Energy
is gross left-edge zero-order-hold integration; there is no idle subtraction.

Always report these exclusions:

- `PHONE_ENERGY_UNKNOWN`;
- `TOTAL_SYSTEM_ENERGY_UNKNOWN`;
- host CPU, DRAM, PSU loss, USB, WiFi, phone, battery, and charger energy are
  not measured; and
- the result is steady-state board energy, not lifecycle energy.

NVML `power.draw` on this Ampere board is a one-second average with a stated
+/-5 W accuracy. For every timeline, use the v3 uncertainty bound:

~~~text
uncertainty_nJ = 5000 mW * paid_window_us
               + 2 * power_limit_mW * 1000000 us
~~~

P-state transitions are recorded as an outcome. They are not a validity
failure because creating a lower-power interval is part of the tested
mechanism.

## 4. Frozen Workload Procedure

The first acquisition is B=8 only:

- batch size: 8;
- requested generation: 32 tokens;
- every realized request must generate at least 32 tokens;
- chat template and one fixed prompt that does not terminate early;
- context and prefill limits fixed before acquisition;
- at least two resident warmup groups before `DRIVER_READY`;
- integer request count, divisible by 8, chosen from a non-energy readiness run
  with a 90-120 second control-window sizing target; and
- no more than 4096 requests per timeline.

The 90-120 second range is a sizing target, not a separate validity gate. The
implemented duration-independent gate is at least 100 in-window power-value
changes. The readiness run may determine only request count and a p95 SLO. Record its
raw timings, then freeze both values in the plan before the first measured
slot. Do not adjust either after observing energy. Do not use B=16 to rescue a
failed B=8 result.

## 5. Acquisition

Checkpoint E0-A - offline and build integrity:

- [x] v4 runner rejects mixed board UUIDs, reused power artifacts, incomplete
      work, invalid sample streams, fewer than eight pairs, and missing SLO;
- [x] v4 unit tests pass (offline suite 104/104);
- [x] release and ASan/UBSan host builds pass (CUDA / CPU / ASan+UBSan rc=0,
      binary SHA-256s recorded in RESULTS.md, 2026-07-16);
- [x] Android release build passes and the deployed OP15 executable hash is
      recorded in `ACQUISITION_FREEZE.json`; and
- [ ] no stale phone process or adb forward exists before the measured run.

Checkpoint E0-B - non-energy readiness:

- [x] run B=8 with 32 generated tokens without `--measure`;
- [x] require exact token IDs for every paired request and no fallback;
- [x] require continuous OP15 thermal status 0;
- [x] record p95 group wall time and choose 512 fixed requests per timeline;
- [x] freeze the positive 3,500,000 us p95 SLO before acquisition; and
- [x] preserve the readiness artifacts separately from measured artifacts.

Checkpoint E0-C - matched acquisition:

- [x] run eight complete pairs in the runner's fixed ABBA rotation;
- [x] retain every attempted slot, including failures and partial directories;
- [x] use a new empty output directory and never select a successful subset;
- [x] require at least 100 in-window power-value changes per timeline;
- [x] require a maximum in-window sample gap of 250 ms;
- [x] record selected-GPU UUID, power limit, raw power-stream hash, p-states,
      utilization, memory, exact work, and boundary timestamps;
- [x] reject any competing selected-GPU process observed during a paid window;
- [x] record phone thermal state throughout each treatment timeline, not only
      at its boundaries; and
- [x] do not restart, replace, or discard a cohort after seeing its energy.

These harness changes are now implemented and unit-tested (2026-07-16). The
runner reopens each raw power stream once as immutable bytes, verifies its
SHA-256 and sample count, and recomputes the energy, quality, power-limit set,
and uncertainty; aggregation consumes only the recomputed validity. A continuous
selected-GPU process monitor (bounded probe, persisted errors) recomputes
contamination from bytes. An on-device OP15 thermal logger records the whole
treatment window locally and is reopened, revalidated, and bound to every
treatment timeline (control `NOT_APPLICABLE`). The power limit is bound per slot
and required identical across all sixteen slots from the recomputed evidence. A
machine-readable scheduled-buffer placement certificate is emitted per run from
an observe-only `cb_eval` callback in `examples/layersplit/layersplit.cpp` and is
strictly parsed, hash-bound, and reintegrated; missing, duplicate, zero-compute,
CPU-fallback, or wrong-backend evidence invalidates the run. Every new artifact
fails closed and was adversarially reviewed. The callback observes the scheduler's
assigned output buffers before execution; completion is separately gated by
process status and exact output. The Android build and non-measured readiness
pair now pass. E0-C remains unrun.

Checkpoint E0-D - aggregate:

- [x] prove identical closed work and exact token IDs in all 16 slots;
- [x] sum all eight control and all eight treatment timelines;
- [x] require both routes to meet the frozen p95 SLO in every pair;
- [x] report gross and conservative energy, J/request, J/generated token,
      useful throughput, p95, VRAM relief, p-states, and thermal range; and
- [x] write the result without upgrading `formal_claim` from `NONE`.

Result: `GPU_BOARD_DIAGNOSTIC_RELIEF_FAIL`. Control energy is 197.4 kJ;
treatment energy is 302.3 kJ. The conservative control-minus-treatment lower
bound is -125.8 kJ. All eight pairs are exact, valid, and within the 3.5 s SLO.

## 6. Decision Rule

Let:

~~~text
C = sum(control_energy_nJ - control_uncertainty_nJ)
T = sum(treatment_energy_nJ + treatment_uncertainty_nJ)
~~~

The board diagnostic reaches the 10 percent observation gate only if:

~~~text
T * 10 <= C * 9
~~~

and every route meets the frozen SLO. Possible labels are:

- `GPU_BOARD_DIAGNOSTIC_10PCT_OBSERVED_TOTAL_ENERGY_UNKNOWN`;
- `GPU_BOARD_DIAGNOSTIC_BELOW_10PCT_TOTAL_ENERGY_UNKNOWN`;
- `GPU_BOARD_DIAGNOSTIC_RELIEF_FAIL`;
- `GPU_BOARD_DIAGNOSTIC_SLO_FAIL`; or
- `MEASUREMENT_INVALID`.

Even the first label is exploratory. The external enumerable commitment,
witnessed launcher, server-wall instrument, phone energy, and total-system
boundary remain blocked.

## 7. Stop Rule

Stop after the B=8 result and request review.

- If exactness, instrumentation, contamination, or thermal validity fails,
  repair only that measurement defect and do not make an energy statement.
- If the treatment misses the SLO or conservative 10 percent gate, reject the
  current `[0,2)` route as an energy-saving mechanism. The next design must
  change the mechanism, such as a larger certified island, HBM admission
  relief, or scheduler-created A6000 batch/power intervals.
- If it passes, retain it as a selected-GPU board diagnostic only. Do not claim
  server or total energy and do not start B=16 without a reviewed plan.

The current route is expected to be difficult: it removes only 2 of 48 layers
from the A6000 while adding a slower phone stage. A clean failure is useful and
prevents building the dynamic scheduler around a non-saving primitive.
