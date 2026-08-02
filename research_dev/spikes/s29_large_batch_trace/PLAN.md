# S29 Throughput-Aware Full Trace

Status: `S29_LARGE_BATCH_FULL_TRACE_PASS`.

## Objective

Repeat the frozen S28 60-request physical trace without the B4 experiment cap.
Use the largest measured, SLO-feasible batch on each route while preserving the
urgent/background isolation and one shared CUDA tail.

## Fixed topology

- R0: RTX 4060 Ti `[0,6)` -> RTX 4060 Ti `[6,8)` -> RTX 4060 Ti `[8,48)`.
- R2: OP12 HTP `[0,6)` -> OP15 HTP `[6,8)` -> RTX 4060 Ti `[8,48)`.
- Phone activations travel directly over WiFi to the RTX 4060 Ti host.
- USB is control and provisioning only during this checkpoint.

## Batch policy

1. Freshly measure B1, B4, B24, and B32 on R0 and R2 with the exact workers,
   model shards, context, and route used by the campaign.
   Measure R2 first so an HTP session is not left idle through the CUDA sweep.
2. The runtime may select only points present in the resulting digest-bound
   profile.
3. R2 targets B32. Smaller phone batches are allowed only at latest-safe start;
   urgent P0 always uses R0.
4. One CUDA-tail queue serves both routes. A physical batch may mix P1 and P2,
   but may never mix P0 with background work.
5. Worker slot capacity is 32 on each phone and every CUDA stage. The executed
   context envelope is 1 input token plus 4 output steps, with a 16-token
   per-sequence context allocation.
6. A B32 background cohort occupies the complete CUDA tail. There is no
   fictitious urgent tail-slot reserve. P0 requests arriving while that cohort
   owns the tail queue behind it, and the matched run must measure whether this
   still satisfies the P0 latency gate.
7. Because the shared B32 tail is non-preemptive, background offload is guarded
   until one second has elapsed since the most recent observed P0 arrival. This
   is an online quiet-period rule; it does not inspect future trace arrivals.

## Physical sequence

- [x] Start high-capacity resident workers on OP12, OP15, and the RTX 4060 Ti.
- [x] Measure route points and build the immutable profile.
- [x] Run all-CUDA control, ending with DETACH.
- [x] Run priority treatment, ending with STOP.
- [x] Reopen logs and reports, verify placement, conservation, SLOs, batch
      eligibility, shared-tail ownership, and artifact hashes.
- [x] Plot the new request and device timeline.

## Claims and stop rules

This checkpoint can claim request conservation, physical placement, batch
formation, SLO outcomes, and measured CUDA compute time. It cannot claim phone,
network, or total-system energy. The F16-phone/Q8-server route remains
numerically uncertified unless the same-batch output comparison passes.

Stop on an unmeasured batch, missing or duplicate request, SLO miss, placement
fallback, nonempty KV state after drain, stale worker generation, or any worker
capacity below the declared profile.

The first B48 and B32 `[0,8)` launch attempts are retained as negative evidence. OP12 failed
before readiness because its minimum per-sequence KV allocation required an
HTP mapping of about 705 MB at B48 and 470 MB at B32. The route therefore uses
the already-proven OP12 `[0,6)` slice. B32 remains inside the requested
high-batch regime.

The 64- and 48-slot CUDA-tail loads are retained as negative evidence. Their
3.25 GiB and 2.44 GiB KV allocations exceeded the 16 GiB RTX 4060 Ti budget
once all three resident layer slices were present. The campaign therefore uses
a 32-slot tail and does not claim simultaneous P0 capacity beside a B32 cohort.

The first R2 calibration attempt reached every R0 point and then lost the OP15
HTP queue on the first `[6,16)` step after a long idle interval. A fresh-session
R2-first retry failed at the same point with a larger Hexagon buffer, proving
that island ineligible. Neither attempt produced a profile. The executable
route narrows OP15 to `[6,8)` and moves the shared CUDA tail to `[8,48)`; the
loader and placement certificate must both report this exact range.

The first complete B32 trace is retained as negative evidence. It reduced
summed CUDA compute by 20.36% and had zero synthetic SLO misses, but a B32 R2
cohort launched at 1.43 s just before the 2.0 s P0 burst. Non-preemptive tail
ownership raised P0 p95 from 1.50 s to 6.27 s, failing the 1.10x priority gate.
The quiet-period rule above is the bounded repair; no slot reserve is invented.
