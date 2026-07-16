# S4 Multi-Stream Batch Decode Operator Pipeline

Status: design and first-veto contract. No S4 runtime code has been approved or
written.

This spike replaces the unstarted S3-H2/H3 micro-operator continuation. S3's
output-row experiment remains an archived screen; S4 asks a different question.

## Research Question

Can a phone-local, NanoFlow-style operator pipeline improve real multi-sequence
decode by running dense projections on Hexagon HTP while Adreno executes
stateful attention for other independent request groups, without losing HMX
batching efficiency or increasing joules per completed token?

The candidate contribution is narrow:

```text
profile-driven HTP-Adreno inter-operator pipelining
for continuous batched decode across independent KV-owning request groups,
integrated into a server-phone layer pipeline and evaluated by
gross fleet joules per completed output token under service SLOs
```

Do not claim that batching, concurrent streams, operator pipelining, mobile
heterogeneous execution, or J/token is individually novel. Read
[RELATED_WORK.md](RELATED_WORK.md) before describing the result.

## Why S3 Does Not Answer This

S3 split the output rows of one projection. That tests a tensor-partition
mechanism already represented by HeteroInfer and does not exercise attention,
KV ownership, request-group dependencies, dynamic batch occupancy, or service
energy. S4 does not refine or integrate that row split.

## Terms

```text
B       total active decode sequences in one equal-work comparison
S       number of disjoint in-flight request groups, or microstreams
b_s     number of sequences in group s; sum(b_s) = B
C       stored KV rows before GPU-B appends the current decode token
C_eff   actual mask-visible rows after append; expected min(C+1, SWA window)
        for simple SWA, but always log the graph's actual visible rows
D       number of bounded cross-operator buffer slots; D=S initially
round   one new token for every active sequence
```

`S` is pipeline occupancy, not multiple simultaneous commands on one backend.
The first runtime uses one serial HTP worker and one serial OpenCL worker.

## Sequence Lifecycle and Policy Scope

An S4 sequence is assigned to the S4 route at admission. Its GPU KV is allocated
before prompt processing and populated by GPU-B during prefill as well as decode.
It never starts with HTP-owned KV and then switches to GPU-owned KV. V1/V2 may
restore a captured GPU KV snapshot to isolate decode mechanics, but V3 must run
the complete prefill-to-decode lifecycle and charge its cost.

The first service experiment compares mutually exclusive phone-local policies:

```text
intact-HTP:
  HTP owns KV and executes the complete stage

dual-phase:
  HTP decodes one set of sequence-affine requests while OpenCL prefills another

wholegraph-routes:
  HTP and OpenCL own disjoint requests and private KV for their full lifetimes

S4-operator-pipeline:
  HTP executes dense work and OpenCL owns attention/KV for S4 requests during
  both prefill and decode
```

Do not run S4 and an independent dual-route policy on the same backend workers
in the first implementation. Compare them under the same request trace. A later
policy may select a mode for newly admitted requests using measured queue,
batch, context, energy, and thermal curves, with hysteresis and drain between
incompatible ownership modes. Backend utilization alone is not a selector.

## Initial Operator Boundary

For request group `s` at layer `l`:

```text
HTP-A(s,l):
  input RMS norm
  Q, K, and V projections
  Q, K, and V normalization
  RoPE using the group's positions
  retain the layer input for the residual
  export Q, K, and V

GPU-B(s,l):
  consume Q, K, V, the causal/SWA mask, and the group descriptor
  validate sequence, position, KV-slot, slot-epoch, and generation metadata
  update GPU-owned K and V for the exact sequence slots
  execute fused attention
  export kqv_out before the output projection

HTP-C(s,l):
  consume kqv_out
  output projection
  attention post-norm and residual
  FFN norm, gate, up, activation, down, post-norm, and residual
```

Dependencies:

```text
HTP-A(s,l) -> GPU-B(s,l) -> HTP-C(s,l) -> HTP-A(s,l+1)
```

The phone-local sequence coordinator is the only writer of the group descriptor.
It owns sequence/slot epochs and supplies positions to both HTP-A and GPU-B. V0
must identify whether the current graph builds the mask and KV-cell indices on
the host, HTP, or GPU. Their generation, upload, validation, bytes, and setup time
are part of GPU-B; they are not free control traffic.

This is a two-machine reentrant flow shop because HTP executes both A and C.
The initial static HTP priority is:

```text
1. oldest ready HTP-C task, to unblock its next layer;
2. HTP-A task needed to prevent the GPU queue from draining;
3. remaining HTP-A tasks in request-group order.
```

Compare this policy with FIFO and a measured-duration critical-path policy.
Do not build a generic scheduler or online search engine in S4.

The current Gemma graph constructs Q/K/V before `build_attn`, but KV stores,
attention, and `wo` are combined inside `build_attn`. The implementing agent
must identify the exact current overload and graph boundary before code. Do not
edit `src/models/gemma4.cpp`, `src/llama-graph.cpp`, KV code, or
`ggml_backend_sched` during the first two gates.

## Ownership and Safety Invariants

- GPU-B owns all KV state for participating sequences and layers.
- HTP never mutates or mirrors that KV.
- No second persistent HTP KV allocation exists for an S4 layer or sequence.
- GPU-B populates the same KV during S4 prefill; decode does not migrate KV from
  another backend.
- The first spike uses only layers with `has_kv(il) == true`. A layer that reuses
  another layer's KV cannot be assigned independently until the whole KV-sharing
  group has an ownership contract.
- Each sequence has at most one KV-mutating GPU-B task in flight.
- A group contains one token per sequence and distinct `seq_id` values.
- A slot carries `sequence_epoch`, `slot_epoch`, layer, group, and generation.
- A GPU-B descriptor carries the exact sequence IDs, positions, KV-cell indices,
  mask shape/type, epochs, and generation used for that mutation.
- A consumer waits for the producer generation and an explicit visibility fence.
- HTP-A retains the layer input until the matching HTP-C completes.
- No CPU fallback is accepted as HTP/GPU execution.
- Keep `FLASH_ATTN_EXT` fused. Do not split attention heads or attention tensors.
- Do not reuse the S3 xmem row-split path or its stale prepack-cache results.
- Weight and activation traffic, derived prepack storage, and KV bytes are
  accounted separately.

## Equal-Work Baselines

Every comparison processes the same sequence IDs, tokens, positions, logical KV
contents, layers, rounds, dtype, and total B. Backend-specific KV layouts may
differ, but their logical rows and masks must be equivalent. For route controls,
use deterministic sequence assignment and rotate assignments across paired
blocks so one backend does not receive systematically easier requests.

For V0-V2, throughput means:

```text
layer_tokens_per_s = B / complete one-layer round makespan
```

These are not completed model-output tokens. Reserve completed output tokens/s
and J/completed-token for the V3/V4 service experiments.

```text
HTP-full-B:
  intact fused layer or stage on HTP at total B; primary local baseline

HTP-chunked:
  intact HTP work run as the same S groups; isolates lost batching and repeated
  weight reads

GPU-full-B:
  intact GPU layer or stage; capability and affinity control

optype-serial:
  HTP-A -> GPU-B -> HTP-C for each group without cross-group overlap

optype-pipeline:
  proposed reentrant schedule across independent groups

wholegraph-routes:
  HTP processes B_h sequences and GPU processes B_g sequences,
  B_h + B_g = B; sequence-affine coarse control

dual-phase-service:
  HTP decode for admitted sequences while GPU prefills different requests;
  evaluated only over the complete lifecycle of those requests
```

The union or intact total-B run is the primary comparator. Comparing only with
serial microstreams would hide the cost of rereading every projection weight S
times and dropping below the HMX batch threshold.

## S4-V0: Capability Profile and Offline Schedule Veto

No new runtime code is allowed in V0. First dump the real 12B metadata and label
every candidate phone layer by `is_swa`, `has_kv`, Q/K/V head dimensions, SWA
window, and KV-sharing group. `blk.2` is the existing seed shard, not proof that
one layer represents every attention class.

Profile at least one `has_kv` layer from each attention class that the candidate
phone placement will execute. If the model has both SWA and full-attention
layers, report them separately. A V0 pass applies only to the measured class;
do not extrapolate an SWA result to a full-attention layer.

Profile each selected intact layer separately on HTP and OpenCL:

```text
B       = 4, 5, 8, 16, 32
C       = 32, 256, 512, 1024
HTP     = fused attention enabled
OpenCL  = fused attention enabled
```

Add OP15 B=64 only if KV, scratch, and free-RAM budgets pass. OP12 B=64 remains
blocked by the known all-layer KV over-allocation unless the bounded harness
avoids it.

Extract or measure:

```text
H_A     HTP-A time
H_C     HTP-C time
G_B     GPU-B fused-attention and KV time
A_H     HTP KV-write and fused-attention time
X_AG    HTP-to-GPU Q/K/V bytes and measured copy/fence time
X_GC    GPU-to-HTP kqv_out bytes and measured copy/fence time
X_META  mask, position, sequence, KV-cell, and epoch bytes plus setup/upload time
```

Retain engine-local traces plus host monotonic boundaries. Measure solo and
co-running task times; do not infer interference from solo profiles.

Run profiling-on and profiling-off intact-layer controls for both backends. If
profiling changes complete wall time by more than 5 percent, use profiling only
for device-event decomposition and the profiling-off run for the gate. If the
two cannot be reconciled without an assumption, mark the affected timing
`BLOCKED`; do not scale a perturbed trace into a passing result.

Use the existing HTP per-op profiler and OpenCL profiling build where they expose
the required task boundary. If a required aggregate, handoff, or interference
term cannot be measured without a source edit, mark that part of V0 `BLOCKED`,
show the missing observability, and propose the smallest measurement-only change
for review. Do not fill a gate with estimated or summed solo time.

Every bandwidth field must use one provenance label:

```text
ddr_counter:
  a documented memory-controller counter with tool, domain, units, sampling,
  wrap handling, and time alignment

effective_min_weight_read:
  logical compulsory weight bytes divided by measured task time; a useful-byte
  lower-bound model, not physical DDR bandwidth

none:
  no bandwidth number claimed
```

OP15's aggregate `dcvs/bw_hwmon_meas` tracepoint may be used only after its units
and scope are recorded. If comparable access remains unavailable on OP12, report
`effective_min_weight_read` or `none`; do not invent a cross-phone DDR comparison.
Repeated dense-weight reads from microstreaming must still be reported as logical
traffic amplification even when no physical counter is available.

V0 GPU-attention correctness uses the same inputs, positions, masks, sequence
IDs, and logical initial KV for OpenCL FA-on and an OpenCL FA-off or CPU
reference. Compare the complete layer output and the appended logical KV rows:

```text
all values finite
hidden-state rel-L2 <= 1e-2
appended KV rows match after declared dtype conversion within rel-L2 <= 1e-2
all pre-existing and non-target logical KV rows remain unchanged
repeat rel-L2 <= 1e-7
```

Report HTP FA-on as an additional reference. Failure of either the output or KV
correctness gate vetoes that phone.

Replay the exact dependency DAG offline for `S=1,2,4` and the balanced groupings
in the V2 matrix. `b_s=4` is a measured below-HMX negative control. Include fill,
drain, HTP reentry, measured co-run slowdown, handoff, and
repeated-weight-read costs.

Necessary steady-state condition before transfer costs:

```text
max(H_A + H_C, G_B) < H_A + A_H + H_C
```

V0 passes per phone only when:

```text
GPU fused attention compiles and executes with no fallback
attention and KV correctness pass
ideal compute-only layer-tokens/s ratio vs HTP-full-B >= 1.20
bounded ratio including measured handoff and metadata remains >= 1.15
the same S>=2 production schedule passes at B=16 and B=32
for the same layer class and C
every HTP group expected to carry production work has b_s >= 5
every passing dense HTP task is observed on HMX, not inferred from b_s
no duplicate persistent KV allocation exists
at least 15 percent free RAM remains
```

OP12 rule: if the recorded OpenCL flash-attention compile failure reproduces,
mark OP12 `UNSUPPORTED` and stop S4 there. Do not substitute explicit slow
attention. OP15 proceeds independently.

## S4-V1: Attention, Handoff, and Coherency Spike

Only begin after a reviewed V0 pass.

Build a standalone `llama-phone-op-pipeline` target under
`examples/layersplit`. Do not extend `llama-phone-microop`.

First implement a bare GPU-B graph with synthetic and captured Q/K/V. It owns a
bounded KV cache and must match intact HTP and CPU references across positions,
sequence permutations, and KV lengths.

Test both one-token decode and multi-token causal prefill. For prefill, use
prompt chunks of 32, 128, and 512 tokens where memory permits, then decode at
least one token from the resulting GPU KV. Compare pre-`wo` `kqv_out`, newly
written logical KV rows, and the following decode result with an equivalent
direct HTP-B graph and CPU reference. Use intact full-layer HTP only for the
post-`wo` and lifecycle controls.

V1 numerical gates are:

```text
all values finite
kqv_out rel-L2 <= 1e-2
new KV rows match after declared dtype conversion within rel-L2 <= 1e-2
old and non-target logical KV rows remain unchanged
repeat rel-L2 <= 1e-7
sequence permutation and one-sequence perturbation preserve isolation
```

Measure two handoff modes separately:

```text
explicit-copy:
  blocking, host-visible correctness path and measured upper overhead

shared-activation:
  rpcmem/dma-buf allocation imported by OpenCL, one producer and one consumer,
  explicit ownership, generation, and cache-visibility fences
```

The shared path must prove both directions over at least 10,000 alternating
patterns, sizes, offsets, and slot generations:

```text
HTP write -> OpenCL read
OpenCL write -> HTP read
```

V1 passes only when:

```text
zero stale or torn reads
zero wrong-slot or wrong-generation reads
the V1 numerical and prefill-to-decode correctness gates pass
p99 total A->B plus metadata setup/upload plus B->C handoff
    <= 10 percent of intact layer time
the p99 gate holds for the real B=16 and B=32 QKV/mask/kqv_out shapes
    at the same class, C, S, and D that passed V0
no process death, timeout, DSP SSR, or CPU fallback
bounded activation buffers and >= 15 percent free RAM
the V0 predicted complete gain remains >= 1.15x
```

If shared activation fails, retain explicit-copy data as a negative result and
stop before model-graph integration.

## S4-V2: One-Layer Reentrant Pipeline

Construct direct backend subgraphs for one real layer and persistent GPU KV.
Use one HTP worker, one GPU worker, and `D=S` bounded slots in the initial
sweep. Do not use the stock multi-backend scheduler.

Initial matrix:

| Total B | S | Group sizes | Purpose |
|---:|---:|---|---|
| 8 | 1, 2 | 8; 4+4 | low-load and below-HMX control |
| 16 | 1, 2, 4 | 16; 8+8; 4x4 | first useful and fragmentation control |
| 32 | 1, 2, 4 | 32; 16+16; 4x8 | primary screen |
| 64 | 1, 2, 4 | 64; 32+32; 4x16 | optional, OP15 first |

Run the first complete sweep at `C=512`. Advance only the two best schedules to
`C=128,512,2048`. Run C=4096 only after memory accounting leaves at least 15
percent free RAM.

Per configuration:

- use at least five warmups;
- collect at least 100 rounds and 60 seconds of steady work;
- repeat promising points in at least six paired randomized blocks versus
  HTP-full-B;
- retain every round and task timestamp;
- record the actual HMX/HVX path for every HTP task;
- record logical minimum weight bytes per group and direct-counter DDR bytes only
  when a documented counter provides them;
- record fill, drain, ready-idle, queue age, and co-run slowdown.

For the one-layer harness, `decode_round_makespan` starts when all B layer inputs
are ready and ends when the last of the B layer outputs completes. Report its
p50/p95/p99. Also report each group's and each sequence's ready-to-output latency;
do not substitute a favorable group latency for the complete-round metric.

Correctness requires:

```text
all values finite
one-layer hidden-state rel-L2 <= 1e-2
repeat rel-L2 <= 1e-7
batched output matches serial replay from equivalent logical KV state
row permutation preserves per-sequence results
perturbing one sequence does not change another sequence
no stale KV or duplicate KV mutation
```

Run at least 1,000 KV-mutating rounds at the selected point before advancing.

Before graph integration, run a representative one-layer lifecycle check. Build
GPU-owned KV through the S4 prefill path for prompt lengths 128 and 512, then run
128 decode rounds at B=16 and B=32. Record prefill time, decode time, and:

```text
lifecycle_layer_tokens = sum(prompt tokens) + B * decode rounds
lifecycle_layer_tokens_per_s = lifecycle_layer_tokens / complete lifecycle time
```

The lifecycle output/KV correctness gates are the V1 gates plus the V2 hidden
state and isolation gates. The complete lifecycle must remain at least 1.05x
the matched intact-HTP layer lifecycle before proposing graph integration; the
final V3 service gate remains 1.10x.

Paired block-level throughput ratios, not per-round or per-task samples, are the
independent statistical units for the V2 confidence interval. A layer class may
enter V3 only after its own V2 pass. Unvalidated classes must remain on an
explicit intact path and cannot be counted as S4 operator-pipeline coverage.

V2 passes per phone at B=16 and B=32 for the same layer class and C, using the
same `S>=2` schedule, when:

```text
lower 95 percent CI for layer-tokens/s ratio vs HTP-full-B >= 1.10
p95 decode-round latency <= 1.05 * HTP-full-B
optype-pipeline layer-tokens/s >= 1.10 * optype-serial
correctness and sequence isolation pass
no additional persistent dense-weight copy
no duplicate persistent KV allocation
every passing dense HTP task logs the HMX path
at least 15 percent free RAM remains
```

## S4-V2E: Local Phone-Energy Veto

When a valid whole-phone physical boundary is available, run this gate directly
after V2 and before graph integration. Use matched one-layer runs, include fill
and drain, and define:

```text
whole_phone_J_per_layer_token =
    whole-phone joules over the run / (B * measured decode rounds)
```

Precondition each mode for ten minutes. Use at least six randomized paired
windows of at least ten measured minutes each and the same B, C, layer, rounds,
starting thermal state, and correctness gate for S4 and HTP-full-B. Compute the
confidence interval from window-level paired log ratios, not per-round samples.
Record start/end battery state and require matched charging/discharging behavior
using the V4 battery-state rule.

```text
upper 95 percent CI for S4 whole-phone J/layer-token
    <= 0.90 * matched HTP-full-B whole-phone J/layer-token
```

If this gate fails, S4 may continue only as a capacity result; stop the claimed
energy-optimization path. If valid instrumentation is unavailable, V3 timing may
continue but all phone and fleet energy verdicts remain `BLOCKED`.

## S4-V3: Multi-Layer and Dynamic Service

Only after V2 passes, propose the minimal llama graph boundary refactor and stop
for review. No automatic edit of `gemma4.cpp`, `llama-graph.cpp`, KV allocation,
or the scheduler is authorized.

Use one deterministic request corpus and predeclared arrival traces:

```text
prompt lengths: repeated fixed distribution of 128, 512, and 2048
output: exactly 128 generated tokens per request with EOS disabled
sampler: fixed configuration and per-request seeds
identity: identical prompt tokens, request IDs, and arrival timestamps
B targets: 16 and 32; add 64 only after the capacity gate
arrival grid: bracket 8 requests/s and each policy's saturation point
```

A teacher-forced recorded-token run may be added as a separately labeled
compute-control trace. It does not replace the generated-token service result.

Compare A6000-only, current distributed intact-HTP stages, the selected S4
operator pipeline, the best sequence-affine whole-graph route, and the complete
HTP-decode plus GPU-prefill lifecycle. Keep layer placement constant across
distributed controls. The primary V3 service control is the best non-S4
distributed policy that meets the SLO on the identical trace and layer
placement. A6000-only is the V4 fleet-energy comparator, not a substitute for
that service control.

For the S4 mode, include prompt processing through HTP-A/GPU-B/HTP-C so the GPU
KV is created under the same ownership rule used by decode. Report prefill
throughput and TTFT separately; a decode-only win cannot pass if full-lifecycle
service or energy regresses.

Run two different experiments.

### V3-A: SLO-Constrained Capacity

For each policy, sweep the predeclared arrival-rate grid. Define maximum
sustainable arrival rate as the largest rate that passes a 30-minute window
with:

```text
completion rate >= 99.9 percent
queues and activation credits remain bounded
TTFT and TPOT p95 meet the declared SLO
no stale slot, process failure, or >5 percent sustained throughput decay
```

Use at least six paired randomized capacity sweeps and treat each sweep's
maximum sustainable rate as one statistical unit. V3-A passes only when the
lower 95 percent confidence bound for the S4/control maximum-sustainable-rate
ratio is at least 1.10. A closed-backlog drain-throughput run is a useful
diagnostic, but it does not replace the SLO-constrained arrival sweep.

### V3-B: Matched-Service Trace

Set one common rate to 80 percent of the lowest maximum sustainable rate among
the compared policies and replay the identical arrival trace. Equal completed
throughput is expected here, so there is no throughput-gain gate. Require both
policies to pass completion, bounded-queue, thermal, TTFT, and TPOT SLOs, and
require S4 TTFT/TPOT p95 to be no worse than 1.05x the primary control. This is
the service condition used by V4 energy windows.

## S4-V4: Energy Contract

Energy is part of the hypothesis, not optional supporting telemetry.

Primary V4 service metrics:

```text
whole_phone_J_per_completed_output_token
gross_fleet_J_per_completed_token
```

Gross fleet energy includes the host, A6000, PSU losses, host CPU and transport,
and both phone USB rails over one non-overlapping physical boundary. A calibrated
upstream AC meter is preferred. If separate instruments are used, document and
sum non-overlapping boundaries only. NVML is diagnostic attribution, not fleet
energy.

Use two explicit topology comparisons:

```text
scheduler comparison:
  all three devices remain physically present in S4 and non-S4 distributed modes

architecture comparison:
  distributed fleet includes both phones and transport; A6000-only has the
  phones disconnected from the measured boundary
```

Also report an A6000 deployment control with powered but idle phones when
possible. It answers a different operational question and does not replace the
phones-disconnected architecture baseline.

Equal B is useful for a mechanism diagnostic, but it is not the final A6000
architecture baseline. Predeclare and calibrate the A6000-only continuous-batch
policy, including its batch target, before energy windows. Use the most
energy-efficient A6000-only policy that meets the identical V3-B request trace,
SLO, and memory constraints; it may use a larger batch than the phones. Report
the equal-B server diagnostic separately so the production-efficient server is
not artificially handicapped.

```text
gross_fleet_J_per_completed_token =
    integrated physical energy over [T0,T1] / completed output tokens
```

The denominator excludes failed or cancelled outputs, but their energy remains
in the numerator. Idle-adjusted energy is diagnostic only.

Measurement protocol:

- warm model, kernels, weights, and KV outside the window;
- collect five minutes of idle before and after each block;
- issue a batch-ID-tagged ARMED/GO marker and record host-monotonic T0;
- record T1 only after the final result is returned and all queues drain;
- map phone monotonic clocks to host time before and after each block;
- precondition each configuration for ten minutes;
- run at least six paired, randomized 30-minute windows;
- match starting phone temperature within 2 C and state of charge within five
  percentage points;
- record start/end state of charge, voltage, current, charge counter, and charging
  mode for every phone; reject a pair whose battery-state change differs by more
  than one percentage point unless a calibrated, predeclared battery-energy
  correction is applied;
- use the identical request trace and offered load for each pair; both candidate
  and control windows must pass the V3 completion, TTFT, and TPOT gates;
- retain throttled windows because throttling is part of sustainable behavior;
- compute paired confidence intervals from window-level results.

A lower-energy window that overloads, drops work, or misses a service gate is a
failure, not an energy win.

USB full-battery telemetry and the observed near-constant 1.7-1.8 W USB reading
cannot pass an energy gate. An unplugged battery plus WiFi run is a different
topology and must be labeled separately.

The local one-layer mechanism gate is V2E J/layer-token. Do not compare that
quantity with service J/completed-token. The final architecture gate is:

```text
upper 95 percent CI for distributed gross fleet J/completed-token
    <= 0.90 * matched A6000-only gross fleet J/completed-token
```

If physical instrumentation is unavailable, V0-V3 timing work may continue but
the energy verdict remains `BLOCKED`. Do not claim energy savings.

## Required Metrics and JSONL

Emit one JSON object per task, round, and summary block with a shared schema
version and run ID. Include:

```text
git revision, dirty state, build flags, model/tensor hashes
device/SoC/driver/skel identifiers, backend names
B, S, D, group sizes, C, C_eff, layer class, round, sequence and slot epochs
mode, policy, HMX/HVX path, FA implementation, fallback status
task ready/start/end, backend, layer, group, slot, generation
solo/co-run time, handoff bytes/time, fence time, fill/drain, queue age
mask/position/KV-cell metadata bytes, build time, upload time
bandwidth source/tool/domain/units, logical minimum weight bytes
direct-counter DDR bytes and source when a documented counter provides them
effective minimum weight-read rate, KV bytes, activation bytes, prepack bytes
layer-tokens/s or completed output tokens/s as appropriate
TTFT, TPOT, p50/p95/p99, completed/failed/cancelled tokens
rel-L2, max absolute error, repeatability, permutation and isolation results
RSS, free RAM, faults, clocks, governors, temperatures
power source, physical boundary, raw power artifact, T0/T1, gross joules
```

Retain raw task traces, raw power samples, exact commands, and checksums. Do not
summarize a result whose raw artifacts exist only in an ephemeral session path.

## Stop Rules

- Stop a phone at V0 if GPU fused attention is unsupported or falls back.
- Stop before implementation if the same `S>=2` schedule is below 1.20x ideal
  replay or 1.15x measured-bound replay at B=16 or B=32 for the same layer class
  and C.
- Stop before graph integration if shared mutable handoff is incorrect.
- Stop if microstreaming drops required HTP work below HMX and erases the bound.
- Stop if one-layer measured throughput or p95 gates fail.
- Stop the energy path if the V2E whole-phone J/layer-token gate fails.
- Treat a throughput-only win with worse J/token as capacity mode, not an energy
  result.
- Stop adding scheduling complexity if the final gross fleet energy gate fails.

## Agent Checkpoints

The implementing agent stops after:

1. exact graph/API/capability inventory and V0 command plan, before code;
2. OP12/OP15 per-op profiles plus offline replay and V0 verdict;
3. GPU-B and mutable-buffer interface proposal, before new source files;
4. V1 correctness, coherency, and handoff measurements;
5. one-layer static scheduler interface proposal, before V2 implementation;
6. V2 device data and per-phone verdict;
7. V2E physical phone-energy protocol/result when instrumentation exists;
8. llama graph/KV integration proposal, only if V2 and the available energy
   veto pass;
9. V3 service result;
10. physical fleet-energy protocol review, then V4 result.

At every checkpoint report the dirty worktree, exact diff, build output, exact
commands, raw artifact paths and checksums, gate calculations, failures, and
deviations. Do not commit, push, create a PR, or write a commit message.
