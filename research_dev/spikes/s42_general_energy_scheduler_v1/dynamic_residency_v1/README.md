# Dynamic residency shadow replay

Status: `CAPACITY_PASS; SERVICE_ABBA_DIRECTION_PASS; EXACT_TENSOR_MANIFEST_PASS; QWEN18_ATOMIC_STAGING_BLOCKED; FENCE_TRANSFER_QUALIFIED; SAME_PROCESS_WEIGHT_ADOPTION_QUALIFIED; ADOPTION_ENERGY_SCREEN_FAIL; FULL_TRACE_BLOCKED`.

This experiment exercises the dynamic-residency gates in
`research_dev.scheduler` against the measured two-F16-model BurstGPT result.
The policy replay remains shadow-only, while the disabled physical adapter now
qualifies protected transfer into a Gemma-owned CUDA allocation and subsequent
Gemma execution.

## Preserved fallback

The fallback remains the measured static policy:

- Qwen3-14B F16 proxy with 18 CUDA layers, followed by one switch;
- Gemma4-12B F16 proxy with 25 CUDA layers;
- HTP0 holding the Gemma FFN slice;
- HTP1 and HTP2 holding the two Qwen FFN slices; and
- CPU plus the resident phone slices serving overflow work.

That A-B-B-A result completed all 74 requests, 33,843 input tokens, and
11,605 output tokens. Mean fleet energy was 224.038 kJ, 25.537% below the
CPU-overflow control, and mean makespan was 2,578.050 seconds, 7.236% lower.
Those are static-policy savings. No incremental dynamic-policy saving is
claimed.

## Shadow result

The trace contains 57 effective Qwen requests and 17 effective Gemma requests.
The static runner holds all 17 Gemma requests for the post-Qwen phase, so they
are candidate work for earlier service only if a measured GPU filler is safe.
The source arrival order contains 16 model transitions; arrival order alone is
not a GPU-bubble receipt.

The phone already holds 9,673,170,944 bytes of verified weights. Across the
two measured treatment repeats, its conservative minimum available memory was
2,395,652,096 bytes. Preserving the mandatory 2 GiB reserve leaves only
248,168,448 bytes for atomic staging. The smallest current full slice is
3,208,646,656 bytes, leaving a 2,960,478,208-byte shortfall. A fourth slice of
the current size therefore cannot be staged while the old generation remains
READY. The strict policy also rejects the proposed HTP3 rotation because its
load, prepare, and energy rows have not been measured.

The first exact target-4060 capacity probe is checked in under
`results/RTX4060TI_QWEN15_GEMMA1_CAPACITY_V1/`. It ran inside a
`MemorySwapMax=0` scope, hash-checked both F16 proxy models, and cleaned all
CUDA processes. Qwen with 15 GPU layers used 12,744,392,704 bytes above the
idle GPU baseline. Adding a one-GPU-layer Gemma server required another
2,444,230,656 bytes. The dual-resident state left 717,225,984 bytes free, or
180,355,072 bytes beyond its 512 MiB reserve, with zero process swap.

The capacity result is supplemented by a physical A-B-B-A service screen. The
control kept Qwen-18 on CUDA and Gemma on CPU. The treatment used Qwen-15 plus
Gemma-1 on CUDA. Every arm served the same concurrent BurstGPT-derived Qwen
request 52 (16 input and 9 output tokens) and Gemma request 50 (271 input and
41 output tokens). The measured CPU-package plus GPU-board interval excluded
model load and warmup.

Both treatment arms used less server energy and wall time than their paired
controls. Mean CPU-package energy fell 24.055%, GPU-board energy rose 45.047%,
their sum fell 8.656%, and wall service time fell 7.248%. Qwen produced the
same 9 tokens, but its mean first-token time regressed 8.904%. Gemma had 37/41
positional token agreement and a 36-token common prefix. The checked result is
`results/DUAL_RESIDENCY_SERVICE_ABBA_V1.json` and is deliberately labeled
`REPEATED_SERVICE_DIRECTION_PASS_NO_ADMISSION`.

Exact GGUF tensor manifests now bind all four measured placements. Qwen-18
contains Qwen-15 plus 33 tensors from blocks 23, 24, and 25. Removing those
tensors releases 1,981,934,592 raw bytes. Gemma-1 adds `output_norm.weight`
and the tied `token_embd.weight`, totaling 2,013,281,280 raw bytes. The raw
sizes match every logged llama.cpp model-buffer allocation within the
two-decimal MiB logging precision. The checked manifests and their physical
capacity binding are:

- `results/QWEN_GPU_TENSOR_MANIFEST_V1.json`;
- `results/GEMMA_GPU_TENSOR_MANIFEST_V1.json`; and
- `results/GPU_TENSOR_MANIFEST_BUNDLE_V1.json`.

The final Qwen-15 plus Gemma-1 placement fits, but a strict stage-before-evict
transition does not. During Qwen-18 service, only 205,520,896 bytes are
available above the 512 MiB reserve. The measured Gemma runtime allocation is
2,444,230,656 bytes, leaving a 2,238,709,760-byte staging shortfall. The safe
plan uses two residency epochs. First, a READY CPU plus OP15 Qwen route covers
a fallback-backed Qwen-18 to Qwen-15 drain/reload. Qwen-15 then leaves
2,624,585,728 bytes stageable above reserve, so the measured Gemma allocation
can be added atomically with 180,355,072 bytes remaining. This separate
fallback-backed transition mode does not weaken or relabel strict atomic
mode.

The unified scheduler implements the fallback contract and receipt state
machine. It acquires the READY fallback before drain, leases CPU, HTP, and DMA
through the restore bound, and requires exact source-epoch restoration after
failure.

A disabled-by-default physical fence probe now qualifies the transfer part of
the executor on the RTX 4060 Ti plus OP15. It uses two Qwen-only HTP sessions:
HTP0 holds Qwen layers 0 through 5 and HTP1 holds layers 6 through 11. The
existing three-session Gemma plus Qwen plus Qwen placement missed the 2 GiB
phone reserve on this boot, so the probe did not weaken the reserve and did
not load the unused Gemma phone slice.

The bridge opens one protected copy window around each complete ordered group
of 12 Qwen FFN calls. The source tensor is fully resident in pinned desktop
memory before the paid interval, the CUDA context is warmed identically in
both arms, each copy completes before protected Qwen work resumes, and the
destination is checked byte for byte after the paid interval. The 256 MiB
observe-prefetch-prefetch-observe result completed the same three Qwen
requests and exact output tokens in all four runs. Mean accounted fleet energy
changed from 2,866.794 J to 2,868.857 J (+0.072%) and duration changed from
31.680 s to 31.686 s (+0.020%). The paired fleet-energy changes were +0.682%
and -0.532%, so the measured transfer cost is below the observed pair
variation; it is not an energy saving by itself.

A successor pilot copied the complete 2,013,265,920-byte tied
`token_embd.weight` tensor in 480 4 MiB chunks across all 54 paid group
windows. Its group-copy p50 and p90 were 22.836 and 22.880 ms, it had zero
copied-window overrun, and its source and destination hashes matched. Minimum
free GPU memory was 989,396,992 bytes, leaving 452,526,080 bytes beyond the
512 MiB reserve. This pilot stages the exact tensor into a helper-process CUDA
allocation. That allocation is not visible to the Gemma executor, so it is
qualified transfer evidence rather than a READY residency receipt.

The successor same-process pilot removes that ownership gap. Qwen first becomes
READY with 15 CUDA layers and two OP15 HTP sessions. Gemma then allocates its
one-layer CUDA placement in the same process that will execute it. Across 54
protected Qwen FFN-group windows, the loader copies and reads back all
2,013,265,920 bytes before publishing the model. It then serves BurstGPT Gemma
request 50 from that placement and reproduces the established exact 41-token
SHA-256.

The qualified run enforced `MemorySwapMax=0`, observed zero swap for Qwen and
Gemma across 3,958 50 ms samples, peaked at 31,435,599,872 cgroup bytes, and
recorded no OOM event. Minimum free GPU memory was 995,688,448 bytes, leaving
458,817,536 bytes beyond the 512 MiB reserve. Copy p90 was 51.539 ms; p90
protected-window overrun was zero and the one-run maximum was 3.983 ms. The
checked result and raw receipts are under
`results/OP15_SAME_PROCESS_ADOPTION_V1/` and are labeled
`SAME_PROCESS_WEIGHT_ADOPTION_QUALIFIED_ENERGY_ABBA_PENDING`.

The next matched A-B-B-A screen moved source preparation, the full verified
2,013,265,920-byte transition, service, CPU-package energy, GPU-board energy,
and synchronized whole-phone energy inside one paid boundary. Every arm served
the same six BurstGPT Qwen requests and Gemma request 50 with exact per-request
output hashes. The delayed control staged Gemma during the second Qwen cohort
and served it after the Qwen tail. The early dynamic arm staged during the
first cohort and ran the complete Gemma request concurrently with the second
Qwen cohort.

| Metric | Delayed control | Early dynamic | Change |
| --- | ---: | ---: | ---: |
| Duration | 262.205 s | 325.561 s | +24.163% |
| CPU-package energy | 9.465 kJ | 10.366 kJ | +9.521% |
| GPU-board energy | 7.770 kJ | 9.468 kJ | +21.861% |
| Whole-phone energy | 0.519 kJ | 0.640 kJ | +23.365% |
| Fleet energy | 17.753 kJ | 20.474 kJ | +15.326% |
| Gemma READY | 193.733 s | 130.396 s | -32.693% |

Both paired fleet-energy savings were negative, at -17.277% and -13.402%.
All validity gates passed, including equal work, exact outputs, GPU reserve,
zero process and cgroup swap, and zero OOM events. Early READY did not become
an energy saving because a complete Gemma request is too coarse for the Qwen
wait intervals. CPU prompt work and separate CUDA contexts contended: mean
Qwen request-47 prompt time increased from 48.406 to 86.042 seconds, while
Gemma decode time increased from 30.348 to 131.172 seconds. The checked result
is under `results/OP15_ADOPTION_ENERGY_SCREEN_ABBA_V1/` and is labeled
`ENERGY_SCREEN_FAIL_FULL_TRACE_BLOCKED`.

Coarse paid-interval NVML samples also disprove continuous GPU occupancy. The
four matched runs had nonzero utilization in 27.1% to 33.9% of samples, means
of 5.9% to 8.6%, maxima of 43% to 67%, and no sample at or above 90%. The full
tensor pilot had nonzero utilization in 40.7% of samples and a 68% maximum.
The useful 256 MiB copy engine work occupied only about 0.51% of each paid
interval; the full tensor copy occupied about 3.83%. These one-second samples
are descriptive and do not measure individual bubbles. The scheduler targets
minimum fleet energy for fixed work, not 100% GPU utilization.

The checked qualification, all run-level receipts, power samples, logs, and
outputs are under `results/OP15_FENCED_GPU_PREFETCH_V1/`. The aggregate result
is deliberately labeled
`FENCE_TRANSFER_QUALIFIED_NO_WEIGHT_ADOPTION`, and its incremental dynamic
energy-saving field is `null`.

This establishes that the tested whole-request overlap is not an admissible
dynamic policy. It does not establish a saving. The pre-admission plan requires
a positive conservative repeated screen before the full 74-request campaign,
so that campaign was not run. The measured 25.537% static-policy saving remains
the fail-closed fallback. Dynamic energy admission remains blocked pending:

- fallback-backed drain/reload transition receipts;
- restore upper latency;
- a contention-adjusted, preemptible micro-filler that fits an explicit GPU
  lower-bound fence; and
- a positive repeated whole-fleet screen before full-trace equal-work and
  output-quality qualification.

GPU utilization samples alone cannot authorize a filler. The server must emit
an explicit lower bound for when protected work can next use CUDA, and the
filler upper bound plus restore and guard must finish before that fence.

## Reproduce

From the repository root:

```sh
output=$(mktemp /tmp/s42-dynamic-shadow.XXXXXX.json)
rm "$output"
python3 research_dev/spikes/s42_general_energy_scheduler_v1/dynamic_residency_v1/shadow_replay.py \
    --output "$output"
abba=$(mktemp /tmp/s42-dynamic-abba.XXXXXX.json)
rm "$abba"
python3 research_dev/spikes/s42_general_energy_scheduler_v1/dynamic_residency_v1/analyze_dual_residency_abba.py \
    --output "$abba"
tensor_bundle=$(mktemp /tmp/s42-gpu-tensor-bundle.XXXXXX.json)
rm "$tensor_bundle"
python3 research_dev/spikes/s42_general_energy_scheduler_v1/dynamic_residency_v1/analyze_gpu_tensor_manifests.py \
    --output "$tensor_bundle"
python3 research_dev/spikes/s42_general_energy_scheduler_v1/tests/test_dynamic_residency_shadow.py
python3 research_dev/spikes/s42_general_energy_scheduler_v1/tests/test_dynamic_residency_abba.py
python3 research_dev/spikes/s42_general_energy_scheduler_v1/tests/test_gpu_tensor_manifest_bundle.py
python3 research_dev/spikes/s42_general_energy_scheduler_v1/tests/test_prefetch_fence_analysis.py
python3 research_dev/spikes/s42_general_energy_scheduler_v1/tests/test_prefetch_qualification.py
python3 research_dev/spikes/s42_general_energy_scheduler_v1/tests/test_adoption_qualification.py
energy_screen=$(mktemp /tmp/s42-adoption-energy-screen.XXXXXX.json)
rm "$energy_screen"
python3 research_dev/spikes/s42_general_energy_scheduler_v1/dynamic_residency_v1/analyze_adoption_energy_screen_abba.py \
    --control-r1 research_dev/spikes/s42_general_energy_scheduler_v1/dynamic_residency_v1/results/OP15_ADOPTION_ENERGY_SCREEN_ABBA_V1/raw/control-r9 \
    --dynamic-r1 research_dev/spikes/s42_general_energy_scheduler_v1/dynamic_residency_v1/results/OP15_ADOPTION_ENERGY_SCREEN_ABBA_V1/raw/dynamic-r9 \
    --dynamic-r2 research_dev/spikes/s42_general_energy_scheduler_v1/dynamic_residency_v1/results/OP15_ADOPTION_ENERGY_SCREEN_ABBA_V1/raw/dynamic-r10 \
    --control-r2 research_dev/spikes/s42_general_energy_scheduler_v1/dynamic_residency_v1/results/OP15_ADOPTION_ENERGY_SCREEN_ABBA_V1/raw/control-r10 \
    --output "$energy_screen"
python3 research_dev/spikes/s42_general_energy_scheduler_v1/tests/test_adoption_energy_screen.py
```

The generated JSON reports proposed, admitted, useful, and evicted bytes, the
measured dual-residency capacity candidate, the repeated service-only
direction, memory headroom, the exact missing measurements, and `null` rather
than zero for unmeasured incremental dynamic energy and GPU-bubble coverage.
It also reports exact tensor bytes and the measured atomic-staging shortfall.
`analyze_prefetch_qualification.py` independently re-hashes the physical
prefetch bundle and reports the bounded transfer result, GPU reserve, copy
window coverage, and coarse GPU utilization without converting them into an
unmeasured dynamic-energy claim. `analyze_adoption_qualification.py` does the
same for the Gemma-owned destination, no-swap cgroup, exact post-publication
execution, and synchronized Qwen/OP15 receipts.
`analyze_adoption_energy_screen_abba.py` independently validates the matched
A-B-B-A boundary, exact work, transition, memory, energy, and admission gates.
