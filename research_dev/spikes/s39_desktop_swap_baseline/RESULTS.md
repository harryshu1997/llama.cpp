# S39 CP0-D desktop two-model swap baseline

Status: `DESKTOP_BASELINE_WARM_PASS_COLD_FAIL`

Date: 2026-07-25 EDT

## Verdict

The desktop-only control is complete on the target RTX 4060 Ti. Both models
serve the frozen B8 envelope independently, and physical attempts in both load
orders prove that the two serving processes cannot coexist on the selected
16 GiB GPU.

The frozen warm-page-cache replay passes in all three repetitions. The frozen
cold-NVMe replay fails in all three repetitions. Cold loads take about 8
seconds each; switch intents become overdue while a model is loading, so the
policy immediately drains into the next switch before admitting the queued
Qwen3 8B requests. The final Qwen3 14B target completes its 17 requests, while
all 57 Qwen3 8B requests remain stranded.

This is a valid negative desktop baseline. It does not pass the CP0-D exit
gate, which prospectively required every one of the six replays to complete
all 74 requests. It does quantify the service gap that a future executable
warm tier must remove. It does not prove a phone benefit.

## Frozen inputs

- Source: the existing S39 frequent-switch BurstGPT trace.
- Successful requests: 74, comprising 57 Qwen3 8B and 17 Qwen3 14B requests.
- Model target changes: 9.
- Time scale: source arrivals divided by 20, for a roughly 60-second stress
  replay.
- Payload: deterministic token arrays, input capped at 128 tokens, exactly 8
  output tokens, greedy sampling, and a prospective synthetic 30-second SLO.
- Policy: enqueue, close admission, drain, stop, prepare cache, load, publish,
  and reopen. The policy and all server arguments are identical across runs.
- Input manifest SHA-256:
  `543a788a2cad9327c4b84886e0db8d6586cac05c30dcad7673824c5d927b6cc3`.
- Contract SHA-256:
  `b2e87de77c86d2ff880882dea8210c9cff314b2b63ff673c93836d9c86a77d2f`.

## Model acquisition

Only the authorized desktop model was downloaded. No phone command ran and no
phone shard was created.

| Model | Bytes | SHA-256 result |
|---|---:|---|
| Qwen3 8B Q8_0 | 8,709,518,112 | `408b9555...e883d6`, PASS |
| Qwen3 14B Q4_K_M | 9,001,752,960 | `500a8806...b81f0`, existing artifact PASS |

The Qwen3 8B HTTP response binds repository commit
`6cfbfc7d8ab95bf485c79fcc40be60930d5b4c8c`, expected byte length, and linked
digest. The downloaded bytes reproduce that digest.

## Independent qualification

Both models started in fresh processes with 8 continuous-batching slots,
full CUDA offload, F16 KV, 4096 context, flash attention, and fixed
`n_batch=2048` / `n_ubatch=512`. Each returned exactly 8 tokens for every
request in one simultaneous B8 cohort.

| Model | CUDA layers | Load to ready | B8 wall | Used VRAM | Free VRAM | Result |
|---|---:|---:|---:|---:|---:|---|
| Qwen3 8B Q8_0 | 37/37 | 1.764 s | 0.613 s | 8.593 GiB | 6.981 GiB | PASS |
| Qwen3 14B Q4_K_M | 41/41 | 1.812 s | 0.926 s | 9.171 GiB | 6.403 GiB | PASS |

Physical non-co-residency passes in both orders. With either first model
healthy, the second full-CUDA process exits during load. The first process
remains healthy and retains more than the frozen 512 MiB headroom.

## Repeated replay

| Rep | Regime | Completed | SLO met | Throughput | Mean load | Max publish gap | GPU board energy |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | warm | 74/74 | 74 | 1.228 req/s | 1.762 s | 2.406 s | 2.985 kJ |
| 1 | cold | 17/74 | 2 | 0.208 req/s | 8.027 s | 27.049 s | >=2.784 kJ |
| 2 | cold | 17/74 | 2 | 0.209 req/s | 7.946 s | 26.583 s | >=2.712 kJ |
| 3 | warm | 74/74 | 74 | 1.228 req/s | 1.724 s | 2.390 s | 3.083 kJ |
| 4 | warm | 74/74 | 74 | 1.228 req/s | 1.729 s | 2.399 s | 3.119 kJ |
| 5 | cold | 17/74 | 2 | 0.208 req/s | 7.980 s | 26.292 s | >=2.708 kJ |

The representative run is the median paid-duration repetition within each
regime: warm repetition 0 and cold repetition 5.

| Representative metric | Warm cache | Cold NVMe |
|---|---:|---:|
| Qwen3 8B throughput | 0.945 req/s | 0 req/s |
| Qwen3 14B throughput | 0.282 req/s | 0.208 req/s |
| SLO goodput | 1.228 req/s | 0.024 req/s |
| Queue p50 / p95 | 0.650 / 4.666 s | 58.186 / 77.204 s |
| TTFT p50 / p95 | 0.935 / 4.766 s | 58.832 / 77.298 s |
| Completion p50 / p95 | 1.267 / 5.228 s | 59.215 / 78.086 s |
| Mean unload | 0.214 s | 0.214 s |
| Peak selected-GPU VRAM | 9.177 GiB | 9.181 GiB |
| Peak server RSS | 8.693 GiB | 8.693 GiB |
| Minimum host available RAM | 27.556 GiB | 27.967 GiB |

No process swap or system swap growth occurred.

## Energy boundary

The energy evidence is `SELECTED_GPU_BOARD` from 100 ms NVML samples. The
desktop exposes no readable wall-power or RAPL instrument, so
`SERVER_WALL_ENERGY` and `TOTAL_SYSTEM_ENERGY` remain unknown.

Warm runs have complete sample brackets. A failed cold run terminates its
sampler 43-99 ms before its last completed request. The reducer does not
extrapolate. It integrates through the last completion bracketed by samples
(16 of 17 completed requests) and labels the displayed cold values as lower
bounds. Request, queueing, TTFT, completion, and SLO metrics still include all
17 completed requests.

## CPU/RAM fallback

No CPU/RAM serving fallback was added. The existing server could run a
CPU-offload control, but it would change the host page-cache and RAM conditions
that define this baseline and would not be a small like-for-like one-GPU swap
control.

## Artifacts

- Raw campaign:
  `results/campaigns/campaign_20260725T191334Z/`
- Independent analysis:
  `results/analysis_campaign_20260725T191334Z_v11/`
- Timeline data:
  `results/analysis_campaign_20260725T191334Z_v11/timeline_data.json`
- Throughput timeline:
  `results/analysis_campaign_20260725T191334Z_v11/throughput_timeline.png`
- Throughput timeline, vector:
  `results/analysis_campaign_20260725T191334Z_v11/throughput_timeline.svg`
- Selected-GPU energy timeline:
  `results/analysis_campaign_20260725T191334Z_v11/energy_timeline.png`
- Selected-GPU energy timeline, vector:
  `results/analysis_campaign_20260725T191334Z_v11/energy_timeline.svg`
- Download and HTTP evidence:
  `results/acquisition/`

Every acquisition subdirectory and the analysis directory has a SHA-256
manifest. The independent reducer reopens raw events, streams, resource
samples, cache records, placement records, and manifests before producing
metrics or the graphs. Throughput uses a five-second trailing window for each
model. Energy is cumulative selected-GPU board energy; the failed cold replay
is explicitly shown as a lower bound over its bracketed prefix.

## Next decision

Do not retune the frozen control after seeing this result. The next reviewed
gate should compare the same trace and switch policy against an eligible
executable warm tier. Its minimum benefit target is concrete: admit useful
requests during the 8-second cold-load intervals, prevent the 57-request
stranding failure, and preserve the already passing warm-cache behavior.
Phone qualification remains paused until this desktop result is reviewed.
