# Related Work: Heterogeneous Mobile LLM Scheduling

This note records what HeteroInfer and llm.npu actually demonstrate, what they
measure, and which ideas are valid for the Gemma-4 12B phone-offload pipeline.
It is evidence for `PLAN.md`, not evidence that the same result will hold on
OP12 or OP15.

## Primary Sources

- HeteroLLM was the early name used by the work that became HeteroInfer. Use
  the current [SOSP 2025 paper](https://arxiv.org/pdf/2501.14794) as the
  canonical source. The [v1 paper](https://arxiv.org/pdf/2501.14794v1) retains
  some older prototype details.
- llm.npu is [Fast On-device LLM Inference with NPUs](https://xumengwei.github.io/files/ASPLOS25-NPU.pdf),
  ASPLOS 2025. Its primary implementation materials are the
  [artifact](https://zenodo.org/records/14392760) and
  [MLLM repository](https://github.com/UbiquitousLearning/mllm).

## HeteroInfer

### What it runs

HeteroInfer runs OpenCL on an Adreno 750 and QNN on the Hexagon NPU of a
Snapdragon 8 Gen 3. Its published experiments use batch size one. The CPU is a
control and synchronization plane, not a dense-compute backend.

Its common model path is W4A16, while this project is testing F16 weights and
batched decode through a direct ggml Hexagon backend. OP12 shares the Adreno 750
class, but neither the published ratio nor its operator crossover transfers
without measurement. OP15 must be treated as a separate platform.

The decode optimization does not use two independent request streams. It splits
one matrix multiplication weight along output rows, assigns disjoint row ranges
to GPU and NPU, runs both partitions concurrently, and concatenates their
outputs. No partial-sum reduction is needed.

For a ggml weight with shape `[K, N]`, this means partitioning `ne1=N`:

```text
Y_h = mul_mat(W[:, 0:n_h], X)
Y_g = mul_mat(W[:, n_h:N], X)
Y   = concat_output_channels(Y_h, Y_g)
```

It selects the split offline from real-device profiles. The objective includes
both engine times and synchronization/copy cost, and it can select GPU-only or
NPU-only when splitting loses. The paper's decode choices are generally
GPU-heavy; its 75/25 example is not a portable constant.

For prefill, HeteroInfer additionally uses activation/sequence partitioning to
put standard static shapes on the NPU and dynamic remainders on the GPU. This is
a different decision from decode's output-row split.

### What it reports about bandwidth

On Snapdragon 8 Gen 3 the paper reports:

```text
theoretical LPDDR bandwidth:                 68.0 GB/s
maximum achieved by sustained bulk traffic: 61.9 GB/s
single CPU, GPU, or NPU decode workload:     40-45 GB/s
GPU-only decode:                             43.3 GB/s
concurrent GPU+NPU decode:                   59.5 GB/s
```

The 59.5 GB/s result is reported as 96 percent of the 61.9 GB/s achievable
ceiling. The ceiling is described as coming from large continuous CPU
`memcpy`/NEON operations and GPU vector/image-buffer loads.

The publication does not disclose the DDR counters, tool/API, sampling period,
read/write accounting, cache accounting, or a bytes-over-time formula. No
author implementation artifact was found. Its exact GB/s methodology therefore
cannot be reproduced from the paper and must not be copied as if it were a
complete measurement protocol.

### Why co-execution helps there

- GPU and NPU read disjoint weight ranges concurrently.
- A measured, non-50/50 ratio balances their completion times.
- A persistent host/GPU/NPU mapped pool avoids repeated allocation and copies.
- Fast synchronization avoids a conventional fence or `clFinish` cost reported
  near 400 us.
- The synchronization thread sleeps for a predicted kernel duration and then
  polls a completion flag for a few microseconds.

The mutable mapped-buffer mechanism is not proven in this tree. Existing
per-tensor phone sharing covers read-only weights. It does not establish
OpenCL-write to HTP-read cache coherency for activations.

## llm.npu

### What it runs

The implemented llm.npu system is CPU plus Hexagon NPU and targets prefill. Its
dense path uses NPU-friendly W8A8 per-tensor matrix multiplication. A small
floating-point shadow path corrects selected activation outliers on CPU/GPU,
while LayerNorm, attention, and other floating-point operators also remain on
CPU/GPU.

The artifact uses `rpcmem_alloc`, registers the fd as QNN ION memory, and lets
CPU code alias NPU activation storage. It does not prove OpenCL/HTP mutable
sharing.

Its GPU+NPU section is simulated with TFLite. It does not provide a real
GPU+NPU concurrent implementation, and prefill does not improve in that
simulation because the NPU remains the critical path. Decode remains on CPU.

### What "fully utilize" means

llm.npu does not report physical DRAM bandwidth or DDR performance counters.
It measures operator latency, per-token prefill latency, memory, energy, and the
fraction of the NPU critical path left idle. "Fully utilize the NPU" means:

- keep dense work in efficient per-tensor NPU GEMMs;
- use static shapes and a profiled prefill chunk size;
- reshape tensors into NPU-favorable layouts;
- keep the NPU ready queue occupied.

The reported scheduler example reduces NPU bubbles from 37 percent to 0.7
percent. That is queue/critical-path utilization, not LPDDR utilization.

### Transferable scheduling idea

llm.npu profiles subgraph times and dependencies offline. At runtime it chooses
an input-ready task based on how much NPU work that task unlocks, rather than
choosing only the shortest task or maximizing the number of simultaneous tasks.
Each processor executes at most one subgraph at a time.

For this project the safe first scheduling space is independent requests:

- prefill chunks belonging to different sequences;
- decode microbatches for route-affine sequences;
- one backend prefilling while the other decodes different sequences.

Same-sequence chunks and decode steps retain their attention/KV dependencies.
The scheduler must not create a second KV mutation in flight for one sequence.

### Energy limitation

llm.npu samples Android `/sys/class/power_supply` every 100 ms, repeats three
times, and reports prefill energy on one rooted Redmi K60 Pro. This is useful as
a paper result but is not a sufficient method for this project's gross fleet
J/token gate. It excludes the A6000/server and has no continuous batched-decode
or sustained thermal result.

## Answer for This Project

Yes, concurrent HTP and OpenCL can increase useful memory service rate, but only
measurement can show whether it improves latency and energy on these phones.
There are three materially different mechanisms:

| Mechanism | Weight traffic | Merge | Main risk |
|---|---|---|---|
| Output-row split of one operator | Disjoint row ranges; one logical weight read | Concatenate channels | synchronization, view/prepack behavior |
| FFN gate/up branch split | Two disjoint full weights | activation plus elementwise join | fixed ratio, large remote activation |
| Independent request streams | Each backend reads full layer weights | none across lanes | duplicated weight traffic and contention |

Independent streams may raise physical traffic while wasting energy. A higher
GB/s number alone is not a success. Output-row splitting is the closest transfer
of HeteroInfer because it divides useful work without a K-dimension reduction or
duplicating the whole weight.

Do not split along `K` first. A K split produces partial sums that require a
cross-backend reduction. llm.npu also reports large overheads for small/grouped
subtensor matrix operations, so arbitrary micro-tiling is not justified.

## Measurement Semantics

When direct memory-controller counters are unavailable, report:

```text
effective_min_weight_read_GBps = unique_assigned_weight_bytes / concurrent_wall
traffic_amplification          = logical_weight_bytes_read / unique_model_weight_bytes
overlap_fraction               = (T_h + T_g - T_concurrent) / min(T_h, T_g)
backend_slowdown_i             = T_i_concurrent / T_i_solo
completion_imbalance           = abs(T_h - T_g) / max(T_h, T_g)
```

`effective_min_weight_read_GBps` is a useful-byte model, not physical DDR
bandwidth. It omits cache effects, internal tiling, speculative loads, writes,
and backend-specific prepacking. Never label it DDR bandwidth.

If a vendor DDR counter is accessible, record the exact counter, units,
read/write definition, sampling boundary, permissions, and wrap handling as a
separate `ddr_counter_GBps` field. HTP PMU counters and OpenCL event timestamps
are engine-local measurements and are not automatically whole-SoC DRAM
counters.

## Decisions Carried Into S3

1. Measure current single-backend and independent-stream controls first.
2. Make output-channel row splitting the first new micro-operator mechanism.
3. Compare row, branch, and stream partitioning at equal total batch/work.
4. Use real-device, per-operator, per-batch static profiles; do not import the
   paper's split ratios.
5. Add an llm.npu-style critical-lane scheduling replay only after the primitive
   profiles exist.
6. Keep fused attention and KV mutation on one backend.
7. Require whole-layer latency and whole-phone energy wins before integration.
