# BurstGPT plus one small-model overlay V1

## Workload

`REQUESTS_BURSTGPT_LLAMA1B_84.jsonl` contains the original 74-request
BurstGPT trace plus ten Llama 3.2 1B Q4_0 requests on the same arrival time
axis. It has three execution models:

| Model | Requests | Artifact bytes |
| --- | ---: | ---: |
| Qwen3-14B Q4_K_M | 57 | 9,001,752,960 |
| Gemma4-12B Q4_0 | 17 | 6,975,878,176 |
| Llama-3.2-1B Q4_0 | 10 | 770,928,288 |

The original 74 events, prompts, arrival times, output lengths, and SLOs are
unchanged. The Llama stream reuses source request shapes at indices
`0, 8, 16, 24, 32, 41, 49, 57, 65, 73` and arrives 200 ms after each donor.
This gives 38,948 input tokens and 13,132 output tokens across 84 requests.

The small model is useful because the same byte-identical artifact already
has measured resident desktop CPU, RTX 4060 Ti CUDA, and OP15 Adreno route
profiles. The route evidence includes CUDA activity-tail energy, so an idle
GPU that must open a new activity epoch is not treated as free.

## Placement acceptance cases

`PLACEMENT_EXPECTATIONS.json` evaluates every small-model request under four
atomic runtime states:

| Runtime state | Expected small-model routes |
| --- | --- |
| Current saturated placement, no qualified transition | 10 desktop CPU |
| CUDA epoch opens, all routes resident | 10 OP15 Adreno |
| CUDA tail already charged, all routes resident | 8 CUDA, 2 OP15 Adreno |
| CUDA busy, phone resident | 10 OP15 Adreno |

The two OP15 choices in the reused-CUDA case are request indices 5 and 6.
They have input/output shapes `722/26` and `625/12`; the measured conservative
energy bounds favor the phone for those shapes while the other eight favor
CUDA.

The saturated snapshot is also an explicit capacity test. After the qualified
large-model GPU allocation and reserve, only 653,262,848 bytes remain, which
is 117,665,440 bytes short of the Llama artifact. The measured three-session
phone state has 248,168,448 stageable bytes beyond its 2 GiB reserve, which is
522,759,840 bytes short. Without a qualified load or eviction transition, the
scheduler must keep both accelerator routes unavailable and use CPU.

## Reproduce

From the repository root:

```sh
python3 research_dev/spikes/s42_general_energy_scheduler_v1/small_model_overlay_v1/build_small_model_overlay.py
python3 research_dev/spikes/s42_general_energy_scheduler_v1/small_model_overlay_v1/verify_small_model_overlay.py
python3 research_dev/spikes/s42_general_energy_scheduler_v1/small_model_overlay_v1/evaluate_small_model_placement.py
python3 research_dev/spikes/s42_general_energy_scheduler_v1/tests/test_small_model_overlay.py
```

On the physical RTX 4060 Ti plus OP15 host, capture the all-server baseline
and runtime-scheduled case with:

```sh
bash research_dev/spikes/s42_general_energy_scheduler_v1/small_model_overlay_v1/run_three_model_arm.sh \
  /absolute/output/server-baseline server-baseline 5037
bash research_dev/spikes/s42_general_energy_scheduler_v1/small_model_overlay_v1/run_three_model_arm.sh \
  /absolute/output/runtime-scheduler runtime-scheduler 5037
```

`compare_three_model_runs.py` rejects a comparison unless both runs conserve
all 84 requests and 13,132 output tokens, use the same trace and model hashes,
cover the same paid server and phone interval, and report zero inference
process swap.

## Runtime path

`run_three_model_trace.py` no longer consumes a control result or a predefined
request placement. For each Llama arrival it:

1. probes the live CPU and OP15 executors;
2. captures host RAM, CUDA VRAM, and phone RAM capacity;
3. binds the exact GGUF bytes and SHA-256 to discovered executors;
4. asks `UnifiedScheduler.estimate_runtime_costs` for shape-specific prefill,
   decode, latency, and fleet-energy estimates;
5. schedules the request through the unified resource timeline; and
6. executes only the returned route, then releases its leases.

The CPU model stays resident as the fallback. A missing OP15 executor, stale
snapshot, wrong model identity, inadequate capacity, or unready resource
removes that alternative without removing the CPU baseline. The Llama CUDA
route is rejected while the large Qwen/Gemma GPU residency owns the device
and no separate resident Llama CUDA executor exists.

## Physical result

The first exact-work matched pair is recorded in
[`results/4060ti_op15_20260812`](results/4060ti_op15_20260812/README.md).
Runtime scheduling selected OP15 for all ten Llama requests. Relative to the
GPU-switch plus CPU-small-model baseline, fleet compute energy fell 6.10%,
makespan fell 2.11%, and SLOs improved from 61/84 to 65/84. This is one matched
pair, not an A-B-B-A qualification.

## Boundary

This trace and oracle define what the runtime estimator must demonstrate. The
physical comparison uses CPU-package RAPL, GPU-board NVML, and simultaneous
whole-phone USB plus battery energy over the exact paid trace interval. Initial
model loading and warmup occur before that interval, so this is a resident-task
policy result. It does not qualify a cold-load policy or dynamic large-model
layer migration. Qwen and Gemma retain the same measured GPU switch order in
both cases; only the ten small-model requests receive per-arrival placement.
