# S39 desktop stock-default controls

## Verdict

`STOCK_DEFAULT_CONTROLS_COMPLETE`

The exact stock-default `llama-server` control was run three times per
configuration on the RTX 4060 Ti using the frozen 74-request trace.

- Warm one-model swapping completed all requests and met all SLOs.
- Cold-NVMe one-model swapping failed all three times, completing only the 17
  Qwen3-14B requests and stranding all 57 Qwen3-8B requests.
- Two simultaneous default servers completed all requests and met all SLOs,
  but auto-fit kept Qwen3-8B on CUDA and moved Qwen3-14B entirely to CPU/RAM.

The two-server result is therefore a GPU-plus-CPU fallback baseline, not proof
that both models coexist on the GPU.

## Default binding

The measured binary is `llama-server` build `b9874-99449bafa`, SHA-256
`c95e04dd...72d4e1`. Its complete 51,986-byte `--help` output was hashed before
acquisition.

Only model identity, endpoint, and observability arguments were supplied.
GPU layers, fit, context, slots, batch limits, flash attention, KV layout, KV
types, and continuous batching were omitted from the command and left at the
binary defaults. Every recorded command was independently checked against the
frozen forbidden-option list.

The realized default was four slots for both models. In the one-model swap
control, Qwen3-8B used 37/37 CUDA layers and Qwen3-14B used 41/41. In the
two-server control:

| Model | CUDA layers | Realized slots | Role |
|---|---:|---:|---|
| Qwen3-8B Q8_0 | 37/37 | 4 | GPU-resident |
| Qwen3-14B Q4_K_M | 0/41 | 4 | CPU/RAM fallback |

## Median results

All values are medians of three fresh-process repetitions.

| Control | Completed | SLO | Stranded | Throughput | p95 completion | GPU energy | Peak VRAM | Process RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stock default, warm swap | 74/74 | 74/74 | 0 | 1.221 req/s | 8.989 s | 3147.3 J | 14.60 GiB | 8.69 GiB |
| Stock default, cold swap | 17/74 | 1/74 | 57 | 0.197 req/s | 82.318 s | >=2963.9 J | 14.60 GiB | 8.69 GiB |
| Stock default, two servers | 74/74 | 74/74 | 0 | 1.214 req/s | 5.432 s | 3211.4 J | 14.02 GiB | 11.62 GiB |

The cold energy value is a lower bound over the bracketed completed prefix and
must not be compared as a complete-run energy result.

The two-server control reduces p95 completion latency relative to warm swapping
because neither model becomes unavailable. It has 0.6 percent lower throughput
and 2.0 percent higher selected-GPU energy than warm swapping. Its CPU energy is
not measured, so no total-energy comparison is authorized.

## Interpretation

This strengthens the baseline rather than guaranteeing a system win:

1. Default cold swapping is a reproducible service failure under the trace.
2. Default auto-fit can avoid swapping by using host CPU/RAM, so a phone warm
   tier must be compared against this alternative, not only cold NVMe.
3. A useful phone result should preserve the two-server latency benefit while
   reducing server CPU/RAM pressure or measured server energy.
4. Phone energy and total-system energy remain unknown.

## Figures

- Throughput over time:
  `results/analysis_default_controls_20260725T211840Z_v4/throughput_timeline.png`
- Cumulative selected-GPU energy:
  `results/analysis_default_controls_20260725T211840Z_v4/energy_timeline.png`
- Vector versions:
  `results/analysis_default_controls_20260725T211840Z_v4/throughput_timeline.svg`
  and
  `results/analysis_default_controls_20260725T211840Z_v4/energy_timeline.svg`
- Reduced data:
  `results/analysis_default_controls_20260725T211840Z_v4/timeline_data.json`
- Raw campaign:
  `results/campaigns/default_controls_20260725T211840Z/`

Every raw output directory, the campaign root, and the analysis directory has
a SHA-256 manifest. No phone command, commit, or push was performed.
