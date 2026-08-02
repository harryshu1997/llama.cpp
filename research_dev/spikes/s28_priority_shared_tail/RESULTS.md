# S28 Results

Verdict: `PHYSICAL_PRIORITY_SHARED_TAIL_PASS`

## Real-device setup

- Server: one RTX 4060 Ti 16 GiB.
- Accelerators: OP12 HTP `[0,8)` and OP15 HTP `[8,16)`.
- Shared server tail: RTX 4060 Ti `[16,48)`.
- Workload: frozen 60-request dense BurstGPT mechanics trace.
- Executed shape: one input token and four output steps per request.
- Controls used the same resident worker PIDs and model weights.

## Matched result

| Metric | All CUDA | Priority phone offload |
| --- | ---: | ---: |
| Completed | 60 | 60 |
| SLO misses | 0 | 0 |
| Route counts | R0=60 | R0=10, R2=50 |
| Makespan | 5.007 s | 41.977 s |
| P0 p95 latency | 860.246 ms | 850.127 ms |
| Summed CUDA-island compute | 5.767 s | 4.697 s |

Measured CUDA-island compute relief is 18.55%. This is not an energy number.
P0 p95 changed by -1.18%, within the frozen 10% guard. The background policy
deliberately waited toward latest-safe start to form measured B4 groups, so
the treatment makespan is 8.38x the control even though all synthetic SLOs
passed. P2 max latency was 39.874 s against its 40 s synthetic SLO.

## Batching and placement

| Worker | Treatment rows | Mean batch | Max batch | Placement |
| --- | ---: | ---: | ---: | --- |
| OP12 `[0,8)` | 200 | 3.846 | 4 | HTP0, GET_ROWS CPU only |
| OP15 `[8,16)` | 200 | 3.846 | 4 | HTP0 only |
| CUDA tail `[16,48)` | 240 | 3.038 | 4 | CUDA0 only |

The tail executed both R0 and R2 through one queue and changed route class
seven times. No batch mixed P0 with P1/P2. Every worker reported contiguous
session IDs `[1,2]`, one resident PID/boot nonce, DETACH reset after C0, STOP
after C1, zero missing placement buffers, and `SCHEDULED_PLACEMENT_OK` whenever
it performed compute.

## Honest limits

- F16 phone shards feed a Q8 server tail. Only 10/60 output-token sequences
  matched the all-CUDA control. Numeric correctness is not certified.
- Priorities and SLOs are synthetic; arrivals and observed demand metadata are
  real-derived.
- The scheduler uses immutable request-level routes and measured B1/B4 points.
  It does not switch routes during a request or exit at arbitrary layers.
- P1 and P2 co-batching is unit-tested but did not occur in this physical run;
  the admission controller dispatched homogeneous priority groups.
- GPU-board, phone, network, and total-system energy were not measured.

## Execution timeline

`execution_timeline.png` and `execution_timeline.svg` plot every request from
arrival to completion, every physical device batch, and the measured benefit
and cost relative to the all-CUDA control. `request_placements.csv` provides
one row per request with its route, placement, timing, SLO, and outcome.

This run uses fixed layer handoff, not semantic early exit: every request still
executes all 48 layers. The StageNet queues batch requests on each worker, but
the experiment intentionally restricts dispatch to the measured B1/B4 points;
it is not production `llama-server` continuous batching.

## Artifacts

- `results/physical_20260721T182301Z/{control,treatment,validation}.json`
- `results/a6000_phones_20260721T182048Z/`
- `results/desktop_cuda_20260721T182150Z/`
- `execution_timeline.{png,svg}`
- `request_placements.csv`
- `plot_execution_timeline.py`

`validate_physical_run.py` reopens the reports and session logs, verifies both
manifests, recomputes the gates, and emits the digest-bound validation record.
