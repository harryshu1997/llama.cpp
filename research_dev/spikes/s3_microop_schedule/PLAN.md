# S3 Heterogeneous Micro-Operator and Bandwidth Spike

Status: archived after H1. Do not continue H2/H3 or integrate the row-split
mechanism. OP15 failed, OP12 xmem-off failed, and OP12 xmem-on is invalid due to
stale prepack-cache reuse. The active experiment is
`../s4_streamed_batch_decode/PLAN.md`.

## Questions

1. Does concurrent HTP and OpenCL execution increase the useful memory service
   rate for real Gemma-4 12B operators on OP12 or OP15?
2. At equal useful work, which granularity is best: one operator split by output
   rows, independent FFN branches, or independent request streams?
3. Can an llm.npu-style ready-task policy reduce backend bubbles without adding
   a generic scheduler or violating sequence/KV ownership?
4. Does any local win survive complete-operator, full-layer, energy, and thermal
   accounting?

This is a bounded real-device experiment. It is not a generic operator scheduler
project.

## Batched Decode Contract

The primary workload is static lockstep batch decode, not an arbitrary GEMM row
sweep. Define one decode round as:

```text
B                   = number of live independent sequences
tokens_per_sequence = 1
n_tokens            = B
M_kernel             = B for each dense projection in the decode graph
```

Every real decode control must issue one `llama_decode` call containing B rows.
Each row has a distinct `seq_id`, private KV history, distinct deterministic
input, and a position that advances once per round. Compare the B-way result
against serial replay from the same initial KV state on the same backend. Record
the context length because attention cost and backend interference change with
KV length.

The standalone H1 projection uses `[K,B]` to reproduce the dense-operator shape
seen during batch decode. This is `workload=batch_shaped_projection`; it is not
`workload=batched_decode` because it contains no attention, mask, KV mutation, or
llama graph scheduling. An H1 pass authorizes only a proposal for a one-layer
real batch-decode integration. That integration requires a separate review.

## Research Basis

Read [RELATED_WORK.md](RELATED_WORK.md) before implementation.

The key correction is that HeteroInfer's decode result comes from splitting one
matrix multiplication weight into disjoint output-row ranges. It does not come
from two independent request streams. llm.npu does not measure DRAM bandwidth;
its transferable idea is offline profiling plus ready-task ordering that keeps
the critical NPU queue occupied.

The publications do not provide a reproducible DDR measurement recipe for this
tree. Every bandwidth result in S3 must identify whether it is a direct memory
controller counter or a useful-byte/time model.

## Restrictions

- Do not change `ggml_backend_sched` or advertise unsupported async/event caps.
- Do not change `src/models/gemma4.cpp` for this spike.
- Do not change `examples/layersplit/layersplit.cpp` in the first two gates.
- Preserve the current uncommitted Fused-FA and per-tensor-sharing work.
- Do not split `FLASH_ATTN_EXT`, RoPE, masks, softmax, or KV writes.
- Do not move one live sequence between HTP and OpenCL.
- Do not implement K-dimension matrix splitting; it requires partial-sum
  reduction across backends.
- Abort on unsupported operations. Do not count CPU fallback as accelerator
  execution.
- Do not call an effective byte-rate estimate physical DDR bandwidth.

## Proposed Files

```text
examples/layersplit/microop.cpp
examples/layersplit/CMakeLists.txt
research_dev/spikes/s3_microop_schedule/RELATED_WORK.md
research_dev/spikes/s3_microop_schedule/PLAN.md
research_dev/spikes/s3_microop_schedule/sweep.sh
research_dev/spikes/s3_microop_schedule/RESULTS.md
```

Add a separate `llama-phone-microop` target. Leave all existing
`llama-layersplit` modes unchanged.

## Real Model Data

Start with a one-layer shard retaining absolute tensor names:

```sh
python3 research_dev/shard_gguf.py gemma-4-12b-f16.gguf \
  gemma-4-12b-layer2-f16.gguf --start 2 --end 3
```

Validate the actual GGUF metadata before allocating anything. Expected dense
FFN weights are:

```text
blk.2.ffn_gate.weight  [3840, 15360], F16
blk.2.ffn_up.weight    [3840, 15360], F16
blk.2.ffn_down.weight  [15360, 3840], F16
```

Each weight is approximately 112.5 MiB. Read only required tensor byte ranges
using GGUF offsets and `pread`; do not read the whole model.

Representative projection classes are:

```text
FFN expansion:   [3840, 15360]
FFN contraction: [15360, 3840]
attention Q:     discover and record exact GGUF shape
attention K/V:   discover and record exact GGUF shape
```

Run the first output-row experiment on `ffn_gate`. Add `ffn_down` only after the
first shape is correct. Add attention projections only after the FFN projection
classes are characterized.

For a ggml weight `[K, N]` and input `[K, M]`, split output rows along `ne1=N`:

```text
W_h = W[:, 0:n_h]
W_g = W[:, n_h:N]
Y_h = mul_mat(W_h, X)
Y_g = mul_mat(W_g, X)
Y   = concat_ne0(Y_h, Y_g)       # [N, M]
```

This produces disjoint output channels and needs concatenation, not numerical
reduction.

## Harness Architecture

- Create one HTP backend object and one OpenCL backend object by exact device
  name, not backend type.
- Use one persistent worker thread per backend and a reusable start barrier.
- Never invoke the same backend object from two threads concurrently.
- Build direct backend graphs. Do not use the stock multi-backend scheduler.
- Check every node with backend support APIs before allocation and execution.
- Upload weights once and exclude GGUF I/O, allocation, upload, and prepack
  warmup from steady-state kernel timing.
- Generate deterministic, non-denormal F32 inputs from a fixed seed.
- Use separate processes for OpenCL stock and xmem/prepack policies because
  backend environment options latch during registry initialization.
- Put a timeout around every device run and retain DSP SSR/backend logs.

The initial experiment uses private input tensors on each backend. Measure the
input fanout before the concurrent launch and the output merge afterward. Do not
require mutable cross-backend sharing to establish the concurrency upper bound.

Test these weight-storage policies explicitly:

```text
assigned-slice-private:
  each backend allocates and uploads only its disjoint row range

shared-parent-view:
  one physical full weight is imported by both backends; each graph uses a
  nonzero-offset row view

full-parent-per-backend control:
  each backend owns a full weight copy, even though it computes only a slice
```

`assigned-slice-private` establishes the useful compute/traffic upper bound.
Production eligibility requires `shared-parent-view` or another design whose
total persistent weight storage is one logical tensor. Verify that a backend
does not silently materialize or prepack the entire parent for a small view.

## Measurement Contract

### Timing sources

Record host monotonic timestamps around the start barrier and completion of each
worker. When enabled, also retain:

```text
HTP:    GGML_HEXAGON_PROFILE per-op time and configured PMU events
OpenCL: GGML_OPENCL_PROFILING event queued/start/end timestamps
```

These are engine-local timing/counter sources. They are not automatically
whole-SoC DRAM counters. HTP and OpenCL timestamps also use different time
domains, so align concurrency with the host barrier and monotonic clock.

Reuse existing measurement hooks before adding new ones:

```text
tests/test-backend-ops.cpp
  GEMV_MEM, GEMM_COMPUTE, GEMM_ROOFLINE, and large CPY perf cases

npu-harness/scripts/profile_npu_op12.sh
  existing sustained HTP profiler launcher

npu-harness/scripts/profile_concurrent_op12.sh
  existing two-process concurrency/Snapdragon Profiler launcher
```

The existing GEMM cases report FLOPS, and CPY reports logical source plus
destination bytes. Neither is a physical LPDDR counter. The concurrent script
also uses a different model/workload, so retain it as an instrumentation smoke,
not the Gemma-4 result.

Record:

```text
HTP solo time
OpenCL solo time
HTP co-run time
OpenCL co-run time
worker start skew
worker completion skew
concurrent wall time
input fanout time
output merge/copy time
synchronization time
complete operation time
```

### Bandwidth provenance

Use one of these labels in every result:

```text
bw_source=ddr_counter
bw_source=effective_min_weight_read
bw_source=none
```

For `ddr_counter`, record the exact tool, counter names, read/write definition,
units, sampling interval, permissions/root requirement, wrap handling, and the
start/end commands. If those details are absent, reject the value.

OP15 currently exposes the `dcvs/bw_hwmon_meas` tracepoint with `mbps` and
sample duration fields. Treat it as an aggregate sampled bandwidth estimate and
retain the raw trace. OP12 shell access to the comparable tracepoint is denied;
use a documented Snapdragon Profiler capture if available, otherwise report no
direct DDR result.

Capture idle windows immediately before and after every direct-counter run.
Report gross aggregate bandwidth as primary and idle-subtracted bandwidth only
as a diagnostic. Use the host barrier timestamps to crop the counter trace to
the common workload interval.

For the portable fallback, compute:

```text
effective_min_weight_read_GBps = unique_assigned_weight_bytes / concurrent_wall
```

This is a lower-bound useful weight-read rate. It is not physical traffic: it
does not include activation traffic, writes, cache hits/misses, backend tiling,
speculative reads, or prepack traffic.

Additional derived metrics:

```text
traffic_amplification = logical_weight_bytes_read / unique_model_weight_bytes
overlap_fraction      = (T_h_solo + T_g_solo - T_concurrent) / min(T_h_solo, T_g_solo)
htp_slowdown          = T_h_concurrent / T_h_solo
gpu_slowdown          = T_g_concurrent / T_g_solo
completion_imbalance  = abs(T_h_concurrent - T_g_concurrent) / max(T_h_concurrent, T_g_concurrent)
operator_speedup      = best_intact_complete_time / split_complete_time
```

An ideal non-interfering overlap has `overlap_fraction` near 1. A value near 0
is serial-equivalent, and a negative value indicates destructive contention or
overhead.

Do not publish utilization as a percentage of peak unless an empirical ceiling
was measured on that exact device, build, governor, and thermal state with a
fully documented byte-counting method. HeteroInfer's 61.9 GB/s value is not the
ceiling for either phone in this project.

### Cache and prepack controls

- Use weights substantially larger than CPU/GPU cache and VTCM.
- Rotate across gate/up/down or multiple layer weights to detect an artificial
  hot-weight result.
- Record the first iteration separately from steady-state iterations.
- Record xmem/prepack allocation growth and whether prepack covers only the
  assigned slice or the full parent.
- Run a separate-copy control to expose cache aliasing or allocation effects.

### Energy and thermal data

Record temperature, clocks, governor, charge source, RSS, free RAM, and major
faults before and after each sweep block. Randomize configuration order within a
block.

Android `/sys/class/power_supply` may be retained as diagnostic telemetry with
its sample interval. USB-powered or full-battery telemetry is not a workload
energy result. Phone J/token gates require the physical measurement setup
defined in `NEXT_PLAN.md`.

## S3-H0: Existing Baselines and Measurement Readiness

No new scheduling code is allowed in H0.

- [ ] Record device/SoC, Android build, backend names, driver/skel versions,
      governor, temperature, clocks, and memory state.
- [ ] Check whether an accessible memory-controller counter exists. Record the
      failed command and permission error if it does not.
- [ ] Verify `GGML_HEXAGON_PROFILE=1` per-op timing. Use mode 2 only with
      documented PMU event meanings.
- [ ] Rebuild OpenCL with `GGML_OPENCL_PROFILING=ON` and retain
      `cl_profiling.csv` and `cl_trace.json`.
- [ ] Re-run intact `ffn_gate` projections on HTP and OpenCL at real shapes.
- [ ] Run intact static B-way decode separately on HTP and OpenCL with one token
      per sequence, distinct sequence IDs, private KV histories, and positions
      advanced across multiple rounds.
- [ ] Compare each B-way decode with serial per-sequence replay on the same
      backend. Include row permutation and one-row perturbation controls.
- [ ] Retain current `dualengine` output as a historical overlap control, then
      reproduce its leg shapes in the standalone harness with equal warmups and
      genuine solo and co-run phases.
- [ ] Record HTP-decode plus OpenCL-prefill and the reverse on different
      route-owned work.
- [ ] Confirm no CPU fallback and no hidden second persistent weight copy.

The current `dualengine` labels its co-running leg times as `alone` and gives the
prefill leg no equivalent untimed warmup. Existing overlap numbers are therefore
controls, not yet proof of zero interference. H0 must measure each leg both solo
and co-running.

H0 output is a measurement-provenance table plus raw timing logs. Failure to
obtain a DDR counter does not block H1; it changes the reported metric to
`effective_min_weight_read`.

## S3-H1: Batch-Shaped Output-Row Split of One Real Matrix Multiply

For each batch occupancy `B`, set `M_kernel=B` and:

1. Run the full projection on HTP alone.
2. Run the full projection on OpenCL alone.
3. Run each proposed HTP row slice alone.
4. Run the complementary OpenCL row slice alone.
5. Launch complementary slices from the barrier.
6. Merge outputs and compare with both intact references.
7. Repeat with each weight-storage policy.

Coarse HTP shares:

```text
0, 25, 50, 75, 100 percent of output rows
```

Refine around the best coarse point in 12.5-percent steps. Round row boundaries
to the intersection of HTP and OpenCL kernel constraints. Discover constraints
from current backend code and support checks; do not assume the paper's 256-row
prototype alignment is required here.

The split selector is offline and keyed by:

```text
device, backend build, operator shape, B, weight type, xmem policy, thermal band
```

It may select 0 or 100 percent. A split is not mandatory.

For each candidate use measured co-run times, not a FLOPS ratio:

```text
T_candidate = T_fanout
            + max(T_h_co_run(part), T_g_co_run(complement))
            + T_sync
            + T_merge

T_selected = min(T_candidate, T_htp_intact, T_gpu_intact)
```

Persist the raw profile table and selected boundary. Do not fit a generic model
until the static table has shown stable interpolation across adjacent B values.
This gate is an operator screen. It does not establish a batched-decode speedup
until the selected split runs inside at least one real layer in a B-way
`llama_decode` graph after separate approval.

## S3-H2: Equal-Work Granularity Comparison

Only begin H2 after H1 correctness passes. Compare these modes at equal total
batch occupancy B and useful work:

```text
single:
  one backend runs the complete work

row:
  HTP and OpenCL compute complementary output rows of one operator

branch:
  HTP computes FFN gate while OpenCL computes FFN up, then join

streams:
  HTP processes B_h independent sequences and OpenCL processes B_g independent
  sequences, B_h + B_g = B; each backend reads the full weight
```

For the FFN branch comparison, reproduce the exact current Gemma graph:

```text
gate   = mul_mat(ffn_gate, x)
up     = mul_mat(ffn_up, x)
joined = geglu_split(gate, up)
out    = mul_mat(ffn_down, joined)
```

Verify the argument order and activation implementation in the current model
code rather than assuming them from this document.

For a row-split FFN, use the same output-row boundary for gate and up so each
backend can apply `geglu_split` to its local slices. Concatenate the joined
slices before the down projection. Measure both choices for the down projection:

```text
full down on the selected join backend
output-row-split down after input fanout
```

The streams control must report `traffic_amplification`: at small M it can read
the full weight once per backend for the same total number of rows. Higher
physical traffic is not automatically useful work or lower energy.

Optional Q/K/V branch tests run only if FFN passes or if the measured FFN output
handoff is the blocker. Keep Q/K norm, RoPE, KV update, and fused attention on
one backend.

## S3-H3: Ready-Task Scheduling Replay

Do not build a generic runtime scheduler. First replay a bounded task graph with
one persistent worker and one ready queue per backend.

Use independent sequences to create safe ready work:

```text
both backends prefill different prompt chunks
both backends decode independent route-affine microbatches
HTP decode plus OpenCL prefill
OpenCL decode plus HTP prefill
```

Compare:

```text
FIFO
naive launch-as-ready
critical-lane priority
best static schedule found from the measured trace
```

The initial priority model is diagnostic, not production policy:

```text
score = critical_backend_ms_unlocked
      - activation_handoff_ms
      - predicted_contention_ms
      - energy_penalty_ms_equivalent
```

All component estimates come from H0-H2 real-device profiles. Log the predicted
and observed score components. Same-sequence attention/KV dependencies remain
ordered, and one sequence may have at most one KV mutation in flight.

Report backend busy time, ready-but-idle time, queue age, makespan, aggregate
rows/s, tokens/s, TTFT, TPOT, and p95. Bubble reduction is a scheduling metric,
not a DRAM-bandwidth metric.

## Experiment Matrix

Primary static batch-decode occupancy sweep:

```text
B = 1, 2, 4, 5, 8, 16, 32, 64
tokens_per_sequence = 1
M_kernel = B
```

Run B=128 only if measured KV, scratch, and free-RAM budgets permit it. The B=5
point is mandatory because it crosses the current HMX selection gate. Use the
same B values for intact real batch decode and the H1 batch-shaped projection.

Run the full B sweep at one documented representative KV length. Repeat the
promising B=16 and B=32 configurations at short, medium, and long KV lengths
that fit both phones. Report exact `n_past` values; do not call a position-zero
projection a complete serving result.

Separate compute-heavy prefill diagnostic:

```text
M = 128, 256, 512
```

These points are not batch-decode evidence and must use
`workload=prefill_shaped_projection`.

Run on OP12 and OP15 with:

```text
OpenCL stock
OpenCL xmem with prepack cache
```

If B=128 is run, it overlaps the prefill-shaped M=128 diagnostic deliberately.
Use the same input and weight hashes to check repeatability while keeping the
workload labels distinct.

Use at least five warmups. Collect at least 30 measured iterations and at least
five seconds of steady-state work per point. Repeat promising points in three
randomized blocks and retain each block separately.

## Correctness

Before performance sweeps:

- [ ] Run one real B-way `llama_decode` per round with distinct sequence IDs,
      private KV histories, one token per sequence, and advancing positions.
- [ ] Compare B-way decode with serial replay from equivalent KV state on the
      same backend.
- [ ] Permute batch rows and perturb one sequence while verifying that other
      sequence outputs do not change beyond the established tolerance.
- [ ] Run every complete projection on HTP and OpenCL with identical input and
      weight bytes.
- [ ] Compare every isolated row slice with the corresponding rows of the intact
      result.
- [ ] Compare concatenated row-split output with intact HTP and OpenCL outputs.
- [ ] Compare branch/full-FFN output with intact full-FFN references.
- [ ] Repeat each split five times to detect nondeterminism.
- [ ] Test a one-row perturbation and row permutation at batched B.
- [ ] Verify nonzero-offset shared-parent views with sentinel rows on both sides
      of each boundary.
- [ ] Verify no unsupported node, CPU fallback, timeout, process death, or DSP
      SSR.

Pass criteria:

```text
all values finite
projection rel-L2 <= 5e-3
full FFN rel-L2 <= 1e-2
repeat-to-repeat rel-L2 <= 1e-7
sentinel rows unchanged
```

Always report max absolute error. Hidden-vector argmax is not the sole veto.

## JSONL Schema

Emit one JSON object per configuration and iteration block. Include:

```text
schema version, git revision, dirty state
device, SoC, Android/build/driver/skel identifiers
backend names and profiler build options
model and tensor hashes, layer, shape, weight type
workload, B, tokens per sequence, n_past min/max, M_kernel
mode, split rows/ratio, join backend, xmem policy, storage policy
warmups, iterations, block/order seed
unique assigned bytes, logical bytes, copied bytes, prepack bytes
bw_source and direct-counter provenance when present
all raw timing statistics and derived metrics
rel-L2, max absolute error, finite flag, sentinel result
RSS before/after/peak, free RAM, major faults
temperature, clocks, governor, charge/power source
backend errors, CPU fallback, timeout, DSP SSR
```

Retain p50, p95, mean, standard deviation, min, max, and the raw per-iteration
samples for complete operation time.

## Gates

### Gate H0: Trustworthy Measurement

Proceed when:

```text
timing sources are identified and repeatable within 5 percent
bandwidth values carry valid provenance labels
single-backend references pass correctness
no CPU fallback or hidden second weight copy is present
```

Direct DDR counters are optional. Mislabeling a useful-byte estimate as DDR
bandwidth fails H0.

### Gate H1: Output-Row Feasibility

For at least two adjacent useful B values on a phone:

```text
complete row-split operator p50 >= 10 percent faster than best intact backend
effective minimum weight-read rate >= 15 percent above best intact backend
p95 no worse than best intact backend
completion imbalance <= 20 percent after ratio refinement
correctness and stability pass
```

If only kernel overlap wins but fanout/merge erases the complete-operator win,
record the upper bound and evaluate the mutable-sharing gate below. Do not
integrate the split.

If one phone passes and the other fails, keep a per-device profile. Do not force
one policy across devices.

This is only the batch-shaped operator gate. Before claiming value for batch
decode, a separately approved one-layer integration must pass real B-way
`llama_decode` correctness and improve complete decode-round latency or tokens/s
against the best intact backend at the same B and KV length.

### Gate H2: Granularity Selection

Choose row, branch, streams, or single separately for each device, phase, shape,
and occupancy (`B` for decode, `M` for prefill). A heterogeneous mode proceeds
to a full-layer test only when:

```text
complete projection or FFN p50 >= 15 percent faster than best single mode
p95 is no worse
aggregate useful work improves at equal total B
traffic amplification and persistent memory are reported
correctness passes
```

If independent streams win only by duplicating full-weight traffic, they still
must pass the later J/token gate. If all micro-operator modes lose, retain the
sequence-affine whole-graph lanes.

### Gate H3: Scheduling Value

The ready-task policy proceeds beyond trace replay only when, versus the best
static/FIFO control:

```text
critical-backend ready-but-idle time drops by at least 25 percent
aggregate service throughput improves by at least 10 percent
TTFT/TPOT p95 do not regress beyond the selected SLO
queue memory stays bounded
all sequence/KV invariants pass
```

### Mutable Shared-Activation Gate

Investigate mutable shared activations only if:

```text
H1 or H2 raw concurrency passes
the complete mode loses with copies
an ideal-no-copy calculation predicts at least a 15 percent complete-op win
copy/synchronization explains at least 80 percent of the lost benefit
```

The follow-up needs a dedicated buffer bridge with one writer, generation
counters, explicit ownership/fences, and cache visibility. Prove HTP-write to
OpenCL-read/write to HTP-read for at least 10,000 alternating patterns, sizes,
and offsets on both phones. Any stale read vetoes integration.

### Production Gate

Do not add micro-operator routing to the Gemma model graph until a real full
layer and serving comparison show:

```text
at least 15 percent sustained full-layer latency or service-throughput gain
at least 10 percent whole-phone J/token improvement
no server/fleet SLO regression
30-minute thermal stability
no additional persistent weight copy
bounded activation memory and at least 15 percent free RAM
```

Fleet acceptance still uses the gross server-plus-phones energy gate in
`NEXT_PLAN.md`.

## Agent Checkpoints

The implementing agent stops after:

1. H0 measurement inventory, exact tensor/view design, JSON schema, and device
   commands, before editing code;
2. host build plus CPU/synthetic merge and sentinel smoke;
3. OP12 and OP15 real batch-decode baselines, H0/H1 JSONL, and Gate H1
   operator-screen verdict;
4. one-layer real B-way integration proposal, only if H1 passes and before any
   model-graph edit;
5. H2 interface proposal, before branch/full-FFN additions;
6. H2 device data and granularity verdict;
7. H3 trace-replay proposal, only if requested after H2;
8. mutable-activation proposal only when its gate is met.

At each checkpoint provide the diff, exact commands, complete output, raw JSONL,
profiler logs, and deviations from this plan. Do not commit or push.
