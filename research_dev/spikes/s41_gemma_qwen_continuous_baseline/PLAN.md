# S41 Gemma-Qwen continuous-batching baseline

Status: `SERVER_ONLY_BASELINE_ACQUIRED_AND_GRAPHED; OPERATOR_ISLAND_PROXY_PASS; PHYSICAL_4060_AND_FULL_MODEL_NOT_RUN; REPLICATION_INCOMPLETE`

## Goal

First establish an all-server baseline for two large models on one RTX 4060 Ti
plus its desktop CPU/RAM. This acquisition contains no phone inference. It
measures the cost of one-GPU model replacement and the best bounded server-only
alternative before the phone warm tier is evaluated as a treatment.

S41 is a versioned successor experiment. It may reuse S39/S40 mechanisms, but
it must not modify, relabel, or combine their frozen Qwen/Qwen evidence with
new Gemma/Qwen results.

## Prototype update

A real T2 no-promotion prototype has now replayed the exact 74-request trace
with Gemma on RTX 4060 Ti and Qwen on OP15 plus OP12. It completed 74/74, but
the phone Qwen route met 0/17 SLOs. This is transport feasibility evidence,
not an S41 treatment pass. It also exposed a shared StageNet socket race
between batches and sequence removal; the passing rerun used a prototype-only
serialized wrapper. The next implementation gate is a reviewed core
serialization and credit-admission fix, followed by one bounded Qwen
phone-to-CUDA promotion with path-matched replay and `k_extra=0`. See
`prototype_t2_phone_trace_v1/RESULTS.md`.

## Operator-island prototype update

Three bounded CUDA plus OP15 HTP routes now pass on the 240/5001 MHz A6000
proxy:

- complete FFN residual slices improve Qwen3-14B by 7.54% and Gemma-4-12B by
  9.64%;
- sharded vocabulary heads with phone-side top-1 reduction improve them by
  16.10% and 17.12%; and
- one Qwen GQA group improves median and p90 by about 9.6% at an 8,192-entry
  KV length, but is rejected at shorter contexts because latency or tails
  regress.

Each selected point has three fresh 300-iteration workers, exact final argmax,
and a path-matched numerical gate. These are isolated operator results. They
do not enter the server baseline or authorize a full-model, BurstGPT, physical
4060 Ti, energy, or multi-phone claim.

The next bounded implementation is a disabled-by-default sharded greedy
vocabulary head in the native executor, followed by complete FFN residual
slices. Attention waits for Q/K normalization, RoPE, masking, rolling cache
positions, and a full-model 8,192-entry oracle. Only after direct AOA is
attached or validly bridged to the physical 4060 Ti may the integrated route
enter the fixed trace. See `tp_operator_split_v1/RESULTS.md`.

## Fixed model pair

- `gemma-4-12b-it-q8_0`
  - bytes: `12669645856`
  - SHA-256:
    `7b56cbd0e0d96d5c8d7df9c21b39d67264e58cbabf7a96d4681eff8b3d492848`
  - the exact Q8_0 parent bytes are required on CUDA; later phone shards must
    preserve Q8_0 and bind their own hashes and layer ranges to this parent
- `qwen3-14b-q4_k_m`
  - bytes: `9001752960`
  - SHA-256:
    `500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0`
  - the existing versioned phone shards remain provisional until requalified

The two full GGUFs total about 20.18 GiB, so they cannot both fit fully on a
16 GiB GPU. S41 still requires a physical two-order non-co-residency test
under the exact serving envelope.

## Common serving envelope

Desktop modes use:

```text
RTX 4060 Ti UUID GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08
full CUDA offload
context 4096
parallel slots 8
continuous batching enabled
n_batch 2048
n_ubatch 512
F16 K/V cache
split mode none
greedy sampling
8 output tokens per request
```

CUDA flash attention is enabled unless a prospectively screened alternative
wins without a quality, memory, or placement regression.

## Workload

Preserve the S39 trace's 74 request identities, arrival times, input-length
distribution, nine historical target changes, eight-token output budget, and
30-second synthetic SLO. Remap the 57 former Qwen3-8B requests to Gemma and
retain the 17 Qwen3-14B requests.

The trace uses synthetic token arrays for geometry. Every token must be in the
target model vocabulary and the new request bytes must be separately hashed.
Task quality is evaluated by the independent MMLU64 gate, not by the synthetic
performance prompts.

## Current server-only gates

Acquisition checkpoint: D0 and D1 are complete. S1 has two valid warm
repetitions and three failed cold repetitions. S2 has one run per placement,
but the Gemma-GPU placement failed zero swap and neither acquisition bound CPU
affinity or NUMA placement. S3 produced the acquired goodput, throughput,
latency, timeline, selected-GPU energy, and sampled-process RSS figures.
Replication, CPU affinity, VRAM timelines, and physical batch-size
distributions remain incomplete.

### D0 - Desktop qualification

Run both models independently on the target 4060 Ti at B1 and B8. Require exact
binary, model, GPU, command, and boot identity; at least 512 MiB free VRAM;
eight live slots; continuous batching; zero swap growth; and clean teardown.
Then prove non-co-residency in both model load orders.

### D1 - Versioned trace binding

Generate a new canonical contract and trace root for the Gemma/Qwen pair.
Preserve all 74 request identities, arrival times, prompt lengths, output
budgets, SLOs, and nine target changes. Rebind the 57 former Qwen3-8B requests
to Gemma and retain the 17 Qwen3-14B requests. Validate the exact model,
request, switch, executable, GPU, and host-boot identities before every paid
run.

### S1 - GPU-only model switching

Run three rotated repetitions with warm host page cache and three with cold
NVMe. Only one full-CUDA model is resident at a time. On a target change,
close admission, drain active work, unload, prepare the target cache regime,
load, verify health and VRAM headroom, publish the model, and reopen its FIFO.

### S2 - GPU plus CPU/RAM dual-ready control

Run the same trace with both model routes ready for the full run:

- `GPU_CPU_DUAL_READY`: both routes are ready for the full trace, one full-CUDA
  and one CPU/RAM, with no model replacement.

Use eight slots and continuous batching for both executors. Freeze CPU thread
count, affinity, and NUMA placement before acquisition. Run three repetitions
per placement, six total, with order rotation. A CPU-only two-model run and
dual partial offload are shortened feasibility controls only; they do not
replace the two primary server baselines.

### S3 - Reduction and graphs

Reduce every run from raw request, switch, resource, and energy records.
Generate the fixed server-only figures before any phone result is added:

1. SLO goodput and stacked Gemma/Qwen throughput by mode;
2. P95 TTFT, completion latency, and maximum publication gap;
3. representative warm/cold GPU-switch timelines with load spans;
4. selected-GPU joules per completed output token for equal completed work;
5. peak VRAM, total server RSS, CPU utilization, and physical batch-size
   distributions.

Show every repetition and the median. Mark stranded work on the graph. Do not
normalize a failed cold run as an equal-work energy result.

The acquired v1 reducer lacks peak-VRAM, total dual-process RSS, CPU
utilization, and physical batch-size series. Its RSS plot follows the sampled
process and is not a total for C2. Those omissions must either be filled in a
versioned v2 campaign or removed prospectively from the final paper matrix.

## Deferred phone treatment

No phone result enters the current baseline. After the all-server campaign and
graphs are frozen, the phone treatment may resume with these bounded gates:

1. Gemma Q8 all-phone Adreno OpenCL capacity and kernel qualification;
2. Qwen3-14B Q4_K_M GPUOpenCL route refresh;
3. MMLU64 noninferiority against same-artifact CUDA;
4. both UFS reprepare directions and one reduced model cycle; and
5. phone warm-tier and no-promotion treatment runs against the frozen
   server-only baselines.

The already captured cut-32 and cut-33 Gemma OpenCL load diagnostics are
capacity diagnostics only. Both used process swap and therefore failed. They
are not baseline runs, kernel results, or reasons to delay the all-server
campaign.

## Server baseline matrix

All modes use the same controller, trace bytes, serving envelope, metrics, and
target GPU.

1. `C1_GPU_SWITCH_WARM`: one full-CUDA model at a time, warm host cache.
2. `C1_GPU_SWITCH_COLD`: one full-CUDA model at a time, cold NVMe.
3. `C2_GPU_CPU_DUAL_READY`: both routes ready without promotion, one on CUDA and one
   on CPU/RAM.

Run three rotated repetitions for C1 warm and C1 cold, plus three repetitions
for each C2 placement. A CPU-only run and two-model partial-offload layout are
shortened feasibility rows only. A second-GPU oracle, phones, and broad policy
sweeps are out of scope for this baseline.

## Metrics and figures

Primary metrics:

- completed and stranded requests;
- SLO goodput;
- per-model and total output tokens/s;
- new-request TTFT P50/P95/P99;
- promotion response gap P95;
- completion latency P50/P95/P99;
- model drain, unload, load, and publication time;
- GPU and CPU physical batch-size distributions.

Generate for server-only modes:

1. SLO goodput and stacked per-model throughput by mode, showing every
   repetition and the median;
2. P95 TTFT, promotion response gap, and completion latency;
3. representative warm and cold C1 throughput timelines with
   load/publication markers;
4. selected-GPU joules per completed output token and server RSS/CPU; and
5. GPU and CPU physical batch-size distributions.

Cold runs with stranded work are labeled as failures and are not normalized as
equal-work energy bars. Selected-GPU board energy is not server-wall or
total-system energy. Phone energy remains unknown without an external meter.

The fixed trace supports a trace-specific SLO-goodput and continuity baseline.
Do not claim server capacity from this single offered load. A later
three-point arrival-rate sweep is authorized only after a complete treatment
also passes.

## Stop conditions

Stop the affected route on wrong identity, placement fallback, swap growth,
less than 512 MiB headroom, task-quality failure, request/token loss, stale
ownership, cleanup failure, or inability to reproduce the selected kernel.
Do not replace Gemma, change precision, run phones in the baseline, add direct
KV import, or broaden the scheduler in S41.
