# Pixel 10 Pro kernel tuning, 2026-09-23

**PASS: a small confirmed improvement.** With 128 threads, a 128-lane subgroup,
and 8 output rows per workgroup, full-FFN worker time decreased from **33.098
ms to 32.399 ms (2.111%)**. Full-width round-trip time decreased from 37.182
to 36.587 ms (1.601%). The selected build also passed a real-server token check.

## Tuning process

1. Built an isolated Vulkan specialization override from the previously
   qualified source. Reused all 132 shader objects. The existing arithmetic
   remains FP16 weights with FP32 input and accumulation; wire tensors are FP16.
2. Tested 13 distinct alternative workgroup/subgroup/output-row combinations,
   plus the rebuilt default, block size 8704, and graph reordering using the
   existing backend optimizer. Timing runs had no profiler enabled.
3. Used deterministic inputs for layers 18-23 at both 8704 and 17408 columns.
   Every output was saved and checked against CPU. Ten repeats per sweep arm,
   first two excluded from timing, gave 48 warm calls per width per arm.
4. Repeated the best two settings with 20 repeats in reversed order:
   original, row8, row4, original, row4, row8, original. Each arm then had
   108 warm calls per width. Compared each candidate to the mean of its
   surrounding original controls, then averaged its two replicates.
5. Loaded the winner in a real Qwen server and verified four 64-token requests:
   desktop, half-width phone, full-width phone, desktop. All outputs matched.

All runs used the shared rig lock and ADB port 5037. Workers had finite request
budgets, exited normally, and left no owned forwarding rules. The phone boot
remained unchanged. Battery temperature during confirmation was 30.9-31.8 C;
GPU clocks and SoC temperature were not controlled.

## Confirmed results

These times cover one layer's FFN worker execution, including transfer to/from
its GPU and synchronization. They are not individual shader timings.

| Setting | Half FFN mean ms | Change vs controls | Full FFN mean ms | Change vs controls |
| --- | ---: | ---: | ---: | ---: |
| Original matched controls, 128/128/2 | 18.535 | baseline | 33.098 | baseline |
| 128 threads / 128 lanes / 4 rows | 18.711 | +0.950% | 32.601 | -1.499% |
| **128 threads / 128 lanes / 8 rows** | **18.397** | **-0.743%** | **32.399** | **-2.111%** |

The two full-width row8 replicate means were 32.368 and 32.430 ms, against
33.096 and 33.099 ms matched controls. Half-width performance did not improve
in both replicates, so its small average gain is not treated as robust.
No confidence interval is claimed.

All 1,680 confirmation outputs were byte-identical to the original. Across
all four sweeps, **4,920/4,920 CPU comparisons PASS**, maximum relative L2
0.000325483 (limit 0.01). The repeated inputs represent 12 distinct layer/width
cases, not 4,920 independent random inputs. Alternative reduction orders were
within tolerance but were not all bit-identical; the selected setting was.

The full FFN has 534,773,760 matrix FLOPs. Its confirmed whole-worker rate is
**16.506 GFLOP/s**. This does not establish the GPU's peak arithmetic rate or
its memory-bandwidth ceiling. The calculation is still a one-token GEMV.

## Rejected candidates

Numbers below are exploratory arm means compared to that arm's surrounding
controls. The confirmation table above is the selection evidence.

| Candidate | Full worker mean ms | Change vs controls | Performance result |
| --- | ---: | ---: | --- |
| Rebuilt original settings | 33.161 | +0.028% | Control agrees |
| 128 threads / 1 row | 35.135 | +5.982% | FAIL |
| 256 threads / 2 rows | 38.347 | +15.672% | FAIL |
| 512 threads / 2 or 4 rows | 55.790-56.604 | +67.198% to +69.637% | FAIL |
| 32/64-lane subgroup variants | 36.563-39.861 | +9.726% to +19.622% | FAIL |
| Existing graph optimizer, quantum 4352 | 33.321 | +0.804% | FAIL |
| Larger block, quantum 8704 | 33.882 | +2.502% | FAIL |
| Larger block plus graph optimizer | 33.886 | +2.514% | FAIL |

Numerical checks passed for all these candidates. The graph-order worker is
retained as reproducible negative evidence and was not selected.

## Real-server check

**PASS:** raw SSE token audit, all 4 x 64 tokens identical, and 744 phone calls
with complete per-layer coverage: 372 each for half/full width across layers
18-23. The selected specialization appears in the worker startup log.
Server and worker exited 0; 24 finite-budget completion calls were outside
measurement. Owned forwarding rules were removed and the phone did not reboot.

| Arm | Phone worker mean ms/call | Phone RPC mean ms/call | Decode seconds | Host request energy J |
| --- | ---: | ---: | ---: | ---: |
| Desktop controls, mean of two | - | - | 39.360 | 4935.632 |
| Tuned Pixel, half-width | 19.848 | 24.707 | 42.637 | 4364.318 |
| Tuned Pixel, full-width | 28.642 | 33.239 | 45.398 | 4322.180 |

Relative to the mean of the two desktop controls, half/full decode was
8.326%/15.340% slower, while sampled CPU-package plus GPU-board request energy
was 11.575%/12.429% lower. This is one short request per split, not a full trace.
Phone energy was not measured. These comparisons do not isolate the kernel
change: a matched original-phone real-server arm was not run in this test.
The confirmed kernel improvement comes from the paired synthetic measurements.

## Selected build and evidence

Use the original coalesced worker with the isolated tuned Vulkan library and:

```sh
S42_PIXEL_F16_WG=128
S42_PIXEL_F16_ROWS=8
S42_PIXEL_F16_SUBGROUP=128
```

Keep quantum 4352. [Selected configuration](PIXEL_GEMV_SELECTED.json) records
library search paths, hashes, qualification scope and the server config.
The original runtime and production default remain unchanged. The tuning
specializes only the F16-weight/F32-input one-column GEMV path; other models
and multi-token performance are unverified.

- [Kernel patch](software/pixel10pro-gemv-tune-subgroup/GEMV_TUNE.patch),
  [builder](build_pixel_gemv_tune.py),
  [build provenance](software/pixel10pro-gemv-tune-subgroup/BUILD_PROVENANCE.json).
- [Initial sweep](PIXEL_GEMV_SWEEP.json),
  [graph sweep](PIXEL_GRAPH_SWEEP.json),
  [subgroup sweep](PIXEL_SUBGROUP_SWEEP.json).
- [Confirmation audit](PIXEL_GEMV_CONFIRM.json),
  [aggregate calculation](PIXEL_GEMV_CONFIRM_SUMMARY.json),
  [raw confirmation](physical/pixel10pro-gemv-confirm-1/run1/RESULT.json).
- [Server audit](PIXEL_GEMV_SERVER.json),
  [server configuration](PIXEL_GEMV_SERVER_CONFIG.json),
  [raw server result](physical/pixel10pro-gemv-server-1/run1/RESULT.json).

To reproduce a physical test, use the checked-in harness/config with a fresh
output directory and the shared rig lock. Existing archived output directories
must not be reused. No additional tuning jobs remain queued.
