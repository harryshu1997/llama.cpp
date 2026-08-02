# S41 server-only reduction and graphs

These tools normalize and plot the Gemma 4 12B Q8_0 plus Qwen3 14B server
baseline. They do not acquire a run and do not read phone evidence.

## Raw input

`reduce_server_results.py` consumes one or more S39-style run directories.
Each directory must contain:

- `events.jsonl` with one paid `replay_start`/`replay_end`, request arrivals
  and completions, and any load/switch/publication events;
- `resource_samples.jsonl` with selected-GPU instantaneous power and
  server-process RSS;
- exactly one of `replay.json` or `dual.json` for a successful run, binding
  the paid and energy windows; a failed cold prefix may omit this terminal
  report and paid work then ends at its last completion; and
- a complete `SHA256SUMS.txt`.

The reducer derives request conservation, SLO goodput, output-token
throughput, tail latency, publication gaps, load spans, selected-GPU energy,
and server RSS from the raw records. A supported CPU-utilization field is
included when it is present in every paid-window resource sample.

The normalized output schema is
`s41-server-baseline-normalized-v1`. The renderer accepts only this schema.
This boundary keeps plotting independent of whether a run used the optimized
GPU-switch runner or the desktop-default runner.

Rendering requires CairoSVG for deterministic SVG-to-PNG conversion.

## Reduction CLI

Run from the repository root. Repeat `--run` for every physical repetition:

```sh
python3 research_dev/spikes/s41_gemma_qwen_continuous_baseline/reduce_server_results.py \
  --model 'gemma-4-12b-it-q8_0=Gemma 4 12B Q8_0' \
  --model 'qwen3-14b-q4_k_m=Qwen3 14B Q4_K_M' \
  --run 'C1 warm r0|C1_GPU_SWITCH_WARM|0|WARM_HOST_CACHE|/path/to/run' \
  --run 'C1 cold r0|C1_GPU_SWITCH_COLD|0|COLD_NVME|/path/to/run' \
  --output /path/to/s41-normalized-summary.json
```

The `--run` form is:

```text
LABEL|MODE|REPEAT_INDEX|CACHE_REGIME|RUN_DIRECTORY
```

`--skip-manifest-verification` exists only for local development. It must not
be used for paper evidence.

## Graph CLI

```sh
python3 research_dev/spikes/s41_gemma_qwen_continuous_baseline/render_server_graphs.py \
  --summary /path/to/s41-normalized-summary.json \
  --output-dir /path/to/graphs
```

The renderer emits matching SVG and PNG files:

- `01_slo_goodput_throughput`: every repetition and the per-mode median;
- `02_tail_latency_publication`: P95 TTFT, P95 completion, and maximum
  publication gap;
- `03_timeline_*`: one representative run per mode/cache cell, chosen nearest
  median SLO goodput, with load spans and intent/publication markers;
- `04_selected_gpu_board_energy`: joules per completed output token for
  complete equal-work runs only; and
- `05_server_resources`: peak sampled-process RSS and CPU utilization when
  sampled. The current dual-route raw sampler follows the GPU executor, so its
  RSS is not the sum of both server processes.

Use repeated `--timeline-mode MODE` arguments to restrict timeline figures.

## Fail-closed rules

- A completed run without TTFT or completion latency is rejected.
- A completed dual-ready run stopped only by the exact zero-swap gate may be
  reduced from its raw paid window plus `failure.json`; it is labeled
  `RESOURCE_FAIL_SWAP_GROWTH`, never `PASS`.
- A completed switch sequence without a publication-gap result is rejected.
- An incomplete switch is represented as censored, never as an observed
  maximum.
- Request and per-model completion counts must conserve exactly.
- Energy must reproduce from raw selected-GPU power samples and its persisted
  integration window.
- A stranded or prefix-energy run is visibly marked and excluded from
  equal-work energy normalization.
- The only energy labels are `SELECTED_GPU_BOARD_COMPLETE_RUN` and
  `SELECTED_GPU_BOARD_BRACKETED_PREFIX`. Neither is server-wall or
  total-system energy.

## Focused tests

```sh
python3 -m unittest discover \
  -s research_dev/spikes/s41_gemma_qwen_continuous_baseline/tests \
  -p 'test_*.py' -v
```
