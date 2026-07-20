# Post-S11-E0 Mechanism Decision

Status: `WAITING_FOR_S11_E0_PHYSICAL_RESULT`

## 1. Persisted Necessary Condition

The repaired exact B=8 trace records:

~~~text
full A6000 control wall:       204.764 ms
A6000 tail in A0 treatment:    202.954 ms
control-minus-tail wall:         1.810 ms  (0.884 percent)
phone plus residual gap:       246.702 ms
~~~

At unchanged CUDA-path power and zero A6000 power during the complete gap, the
10 percent energy gate still requires at least:

~~~text
0.10 * 204.764 ms = 20.476 ms
~~~

of control-minus-tail wall reduction. The current route removes 11.3x less.
This timing difference is not a measured GPU active-residency interval.
Equivalently,
the current treatment tail must average no more than 90.8 percent of control
power if gap power is zero, or 78.6 percent if the gap draws 10 percent of
control power.

These are timing sensitivity bounds, not physical power observations. The
32-token S11-E0 readiness result must replace the four-token timing before a
new route is selected.

## 2. Interpret S11-E0 First

- `MEASUREMENT_INVALID`: repair measurement only. Do not change the route.
- Valid 10 percent diagnostic: retain it as `GPU_BOARD` evidence only. Phone
  and total-system energy remain unknown.
- Valid failure or SLO failure: reject static OP15 `[0,2)` as an energy-saving
  mechanism and follow the gates below.

Do not run B=16 merely to rescue a failed B=8 cohort. Existing B=1 through B=16
latency data show that batching improves phone utilization but does not remove
enough A6000-tail work.

## 3. Gate N1 - Tail-Only Necessary-Condition Sweep

Before running a larger phone island, measure the selected A6000 tail alone for
candidate contiguous cuts `k` under identical B=8 and one adjacent useful
batch. Use representative, hash-bound cut activations and state. Compare gross
tail-board energy against the full-model control.

Define conservative selected-board bounds:

~~~text
tail_upper(k) = measured_tail_energy(k) + tail_uncertainty(k)
control_lower = measured_control_energy - control_uncertainty
~~~

Reject a cut immediately when:

~~~text
tail_upper(k) > 0.9 * control_lower
~~~

Selected-A6000 gap energy, transfer-related GPU activity, and transition energy
are nonnegative, so no phone implementation can rescue a cut that fails this
bound. Equality remains a mathematical survivor because the frozen 10 percent
gate accepts equality; any positive added board energy would then make it fail.

For surviving cuts, add measured phone-island time, thermal state, activation
traffic, and the A6000 gap-power/transition term. Select the smallest cut that
passes the complete bound and SLO. Do not extrapolate layer count linearly:
Gemma SWA/global layers and kernels are heterogeneous.

## 4. Gate N2 - Frontier Bundle Instead of Serial Offload

A larger serial phone island will normally lengthen the phone gap. The
plausible scheduler mechanism is therefore:

1. execute certified resident phone islands ahead while the A6000 runs other
   independent READY DAG work;
2. buffer exact returned activations under bounded credits and leases;
3. merge compatible CUDA tails into a denser native batch;
4. execute tails in a contiguous A6000 active burst; and
5. use only measured SLO-safe power caps or idle intervals outside that burst.

The server never waits solely for a late phone. Compare against an optimized
server-only policy with identical DAG reordering, batching, and power caps.
The claimed cause must be a measured dense-tail burst or lower-power interval,
not phone utilization or skipped GPU-us.

## 5. Gate N3 - HBM Admission

If no compute-energy cut survives, test the strongest existing phone benefit:
exclusive HBM ownership. Current exact checkpoints release 888 MiB for OP15
B=8 and 1288 MiB for the two-phone B=1 route.

Use a mixed-model horizon to test whether this crosses a real threshold that:

- admits another resident model;
- permits a larger native CUDA batch; or
- avoids a measured load, eviction, or rejection.

Include ownership transitions and reload cost. Do not infer a capacity win from
released MiB alone, and do not turn selected-board energy into a second-GPU or
total-server claim.

## 6. Stop Conditions

Stop this Gemma route if neither a conservative tail-energy cut nor a measured
HBM admission threshold exists. Do not revive:

- per-GEMM output-row splitting rejected by S3;
- Adreno attention at realistic context rejected by S4; or
- raw additive phone throughput rejected by S5.

A different service or model needs a new correctness, latency, boundary, and
thermal atlas row.
