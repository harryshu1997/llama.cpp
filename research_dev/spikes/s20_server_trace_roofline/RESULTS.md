# S20 server-only real-trace resource timeline

Verdict: `SERVER_TRACE_TIMELINE_PASS`.

## Headline

A real BurstGPT arrival/token-shape cohort replayed through Gemma-4-12B F16 on
one A6000 changes resource regime over time. It is not permanently
compute-bound or memory-bound:

| executed interval | wall span | mean DRAM pressure | mean tensor active | mean SM issue |
|---|---:|---:|---:|---:|
| prefill only / compute-dominant | 8.1 s | 42.0% | 44.1% | 14.0% |
| overlapping prefill + decode / mixed | 3.0 s | 44.2% | 43.0% | 14.1% |
| decode only / memory-dominant | 11.7 s | 76.0% | 6.8% | 4.9% |
| idle or transition | 0.2 s | 0.5% | 0.0% | 0.0% |

The decisive transition occurs around 11.1 s. During decode-only execution,
tensor activity falls to 6.8% while DRAM pressure rises to 76.0% on average and
peaks at 89.3%. This is direct hardware evidence for the scheduler's temporal
resource-complementarity premise.

Open the graph:

- `results/server_trace.html` - self-contained explanation and SVG;
- `results/server_trace.svg` - scalable paper/slide figure;
- `results/server_trace.png` - 1440 x 850 raster copy.

## Workload and execution

- Source: frozen S15 BurstGPT cohort, 32 observed request identities.
- Arrivals: 21 requests at relative 0 s and 11 requests at relative 1 s.
- Observed work: 21,203 input tokens and 1,898 output tokens.
- Synthetic only: token values, because BurstGPT does not publish prompt text.
- Model: `gemma-4-12B-it-f16.gguf`, SHA-256
  `bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a`.
- Runtime: current-source `llama-server`, 32 slots, continuous batching,
  unified F16 KV, 4096 logical batch, 512 physical microbatch.
- Device: physical A6000 index 0,
  `GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f`.
- Replay wall: 22.77 s.
- Request latency: p50 12.887 s, p95 21.769 s, max 21.846 s.
- TTFT: p50 8.498 s, p95 11.116 s, max 11.117 s.

All 32 responses report the exact requested prompt and generation token counts.
Every request has a persisted ordered start, first-token, and completion event.
GPU 1 was sampled 328 times: utilization stayed 0%, memory utilization stayed
0%, and its driver allocation remained 57-61 MiB.

## Measurement meaning

The background phase label is execution-derived:

- request start to first returned token is the prefill/TTFT interval;
- first token to exact completion is the decode interval;
- concurrent intervals produce the mixed label.

The lines are independent Nsight Systems GA10x hardware metrics sampled at
1 kHz and averaged into 100 ms bins. DRAM pressure is read plus write throughput
capped at 100%; tensor active and SM issue retain their Nsight definitions.

This is a model-level temporal classification, not a formal per-kernel roofline
certificate. TTFT includes any runtime delay before the first token, although
all 32 slots were provisioned and the cohort never intentionally exceeded slot
capacity. Short zero-activity gaps during prefill are preserved rather than
smoothed away.

## Integrity

The first acquisition is rejected: inference completed, but the sampler assumed
the documented object form of `/slots.next_token`; this branch emits a one-item
array. The parser was repaired, the response shape was made fail-closed, and the
entire physical run was repeated. Only run 2 enters this result.

Load-bearing run-2 hashes:

| artifact | SHA-256 |
|---|---|
| Nsight QDSTRM | `23a9771bc165fb73eef61304012f196db9b56633b25e4ec9aff64852bde8d86c` |
| exported Nsight SQLite | `a59b96e12193689559899ab72ad0662fca8d591fa8a2e85c0b47e7965ebcd91c` |
| request events | `647d337ffecfd01c7c6737d54f0abb22fd8d3f30a34d03014af1e0ff23eaf896` |
| runtime samples | `b4d5f715944e0f07f80d5312d3b946fb983222305264644ea9335e365107042d` |
| run manifest | `04b6db7615f70befbffc60106c864c4bd4a9532c67d877cccf5f4c1e3d34e126` |
| GPU-1 control | `38b2053b31c84f60f92ba98e857f0f427b960f749ce588d1ea49f61453359447` |
| reduced timeline | `b85efdb347a7f3405a82e4a1e11ba3924c2e4d95adb03719eee16e61186dade1` |

Raw evidence is under `scratchpad/s20_server_trace_roofline/`; it is not a
source artifact and should not be committed.

