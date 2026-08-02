# S29 Throughput-Aware Full Trace Results

Verdict: `S29_LARGE_BATCH_FULL_TRACE_PASS` for the predefined mechanics and
selected-CUDA-compute gate. This is not a total-system energy or numeric-quality
pass.

## Executed system

- RTX 4060 Ti 16 GiB, OP12 HTP v75, and OP15 HTP v81.
- R0: CUDA `[0,6)` -> CUDA `[6,8)` -> CUDA `[8,48)`.
- R2: OP12 `[0,6)` -> OP15 `[6,8)` -> CUDA `[8,48)`.
- Every worker exposed 32 sequence slots. Phone activations used direct WiFi;
  USB was control/provisioning only in the paid runtime.
- The 60-request dense trace contains 10 P0, 20 P1, and 30 P2 requests over a
  two-second arrival span. Priority and SLO labels are synthetic.

## Fresh route calibration

Each cell is the range across two physical repetitions for one input token and
four decode steps. No calibrated point is inferred from another batch.

| Route | B1 | B4 | B24 | B32 |
|---|---:|---:|---:|---:|
| R0 wall | 0.274-0.360 s | 0.290-0.316 s | 0.415-0.434 s | 0.388-0.394 s |
| R2 wall | 0.724-1.264 s | 1.545-1.784 s | 4.800-4.973 s | 5.805-6.529 s |

The scheduler consumes a canonical profile derived from the persisted
calibration with a 1.10x latency envelope. The profile digest is
`sha256:50b1a0dcf36baa20392468ff5bfef77e5799727b48a07b19e6129e0375f542bc`.

## Matched full trace

| Metric | All-CUDA control | Priority phone treatment | Result |
|---|---:|---:|---:|
| Completed | 60 | 60 | PASS |
| SLO misses | 0 | 0 | PASS |
| Route distribution | R0=60 | R0=28, R2=32 | B32 offload |
| Phone physical batches | none | OP12 B32 x4; OP15 B32 x4 | PASS |
| P0 p95 latency | 1.386 s | 0.636 s | -54.13%, PASS |
| Summed CUDA compute | 6.668 s | 5.609 s | -15.88%, PASS |
| Makespan | 4.505 s | 12.138 s | 2.69x, cost |

The one-second urgent-quiet guard is load-bearing. An earlier real run launched
R2 at 1.43 s, before the last P0 burst at 2.0 s. The non-preemptive B32 tail
then raised P0 p95 from 1.50 s to 6.27 s. The final policy uses only observed
arrival history: it waits for one P0-quiet second before admitting background
R2. It does not reserve nonexistent tail slots or inspect future arrivals.

The CUDA control did not form B32 during the dynamic trace; its per-stage
maximum was B18. This is a valid online outcome: compatible rows were not ready
together. The phone route deliberately waited for a B32 background cohort.
Thus the mechanism is device-local, SLO-constrained batch formation, not one
fixed fleet-wide batch size.

## Physical validation

- Five workers retained one PID, boot nonce, and device boot ID across exactly
  three sessions: calibration DETACH, control DETACH, treatment STOP.
- Every active session reported `SCHEDULED_PLACEMENT_OK`; missing-buffer count
  was zero. The control phone sessions correctly reported placement unobserved
  because they performed no compute.
- All request, route-epoch, sequence-slot, batch-event, and terminal ownership
  checks passed. Every worker drained before DETACH or STOP.
- Both phone stages and the shared CUDA tail emitted four B32 rows, one per
  decode step. P0 was never mixed with background work.
- Phone/USB/network/total-system energy is unknown. Only summed CUDA kernel
  time is reported as server work relief.
- F16 phone execution versus Q8 CUDA control matched greedy tokens for only
  6/60 requests. The route remains numerically uncertified and cannot support
  a quality or equal-output energy claim.

## Capacity findings

- OP12 `[0,8)` could not provision B32/B48 because of HTP KV mapping size; the
  valid B32 prefix is `[0,6)`.
- CUDA `[8,48)` at 32 slots consumed 12,844 MiB. Together with CUDA `[0,6)`
  (1,900 MiB) and `[6,8)` (762 MiB), the GPU used 15,822 MiB. Tail capacities
  48 and 64 OOMed.
- OP15 `[6,16)` lost the HTP queue on its first step in two fresh attempts,
  including a larger Hexagon buffer. Narrowing it to `[6,8)` passed B1 through
  B32. The failed island is not eligible.

## Artifacts

- Accepted reports: `results/physical_20260721T203914Z_final2/`.
- Phone session: `results/phone_20260721T203914Z_final2/`.
- Desktop session: `results/desktop_20260721T203914Z_final2/`.
- Independent verdict: `results/physical_20260721T203914Z_final2/validation.json`.
- Timeline: `execution_timeline.{png,svg}`; request table:
  `request_placements.csv`.

Offline verification: 10 S29 tests plus 28 inherited S26/S28 tests pass;
Python syntax, shell syntax, and `git diff --check` pass. Nothing was committed
or pushed.
