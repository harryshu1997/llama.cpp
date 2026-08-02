# S39 CP0-D desktop two-model swap baseline

Status: `FROZEN_DESIGN; INPUTS_NOT_BUILT; ACQUISITION_NOT_RUN`

## Question

What throughput, queueing, SLO, memory, and selected-GPU energy cost does a
one-GPU desktop pay when two useful models cannot coexist and service demand
forces repeated model replacement?

This is the control for the executable phone warm-tier design. It does not use
phones and cannot establish a phone benefit by itself.

## Authorized scope

This gate may:

- download only the pinned Qwen3 8B Q8_0 artifact to the RTX 4060 Ti desktop;
- verify its exact byte count and SHA-256;
- use the already present Qwen3 14B Q4_K_M desktop artifact;
- run both models independently on the one selected RTX 4060 Ti;
- attempt one simultaneous full-CUDA load to prove non-co-residency;
- replay the frozen requests under warm-cache and cold-NVMe conditions; and
- preserve, validate, summarize, and graph desktop evidence.

This gate must not:

- run `adb` or any phone command;
- create, download, or modify a phone shard;
- modify V1, V2, V2.1, or V2.2 evidence;
- implement or run a phone-assisted switch;
- claim whole-server or total-system energy;
- commit or push.

## Frozen hardware and models

Selected device:

- host: `zhihao-Z690-C-ac`;
- SSH target: `zhihao@172.20.74.85`;
- GPU: `NVIDIA GeForce RTX 4060 Ti`;
- GPU UUID: `GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08`;
- reported VRAM: 17,175,674,880 bytes.

Models:

- Qwen3 8B Q8_0, `/home/zhihao/models/Qwen3-8B-Q8_0.gguf`,
  8,709,518,112 bytes, SHA-256
  `408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6`,
  repository `Qwen/Qwen3-8B-GGUF`, revision
  `6cfbfc7d8ab95bf485c79fcc40be60930d5b4c8c`;
- Qwen3 14B Q4_K_M, `/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf`,
  9,001,752,960 bytes, SHA-256
  `500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0`.

## Frozen replay

Source identity and arrival order come from
`../s39_phone_model_switch_trace/bundle_frequent/requests.jsonl`. The replay
keeps all 74 successful rows and skips the one recorded source failure.

The source trace has no prompt bytes, deadlines, or priorities. CP0-D therefore
uses a separately hashed semi-synthetic mechanics sidecar:

- `ChatGPT` rows map to Qwen3 8B Q8_0;
- `GPT-4` rows map to Qwen3 14B Q4_K_M;
- arrivals are divided by 20 with exact integer floor, retaining total order;
- input length is `min(source_input_tokens, 128)`;
- output length is exactly 8 tokens;
- prompt token IDs are deterministically derived from the event ID and a
  frozen palette of token IDs already executed by the Qwen3 route;
- every successful request has a prospective 30-second SLO;
- sampling is greedy, EOS is ignored, and prompt caching is disabled.

The nine source-derived target changes remain in their exact order and are
scaled by the same factor. The initial hot model is Qwen3 8B.

## Frozen serving envelope

Each server process uses:

- one visible GPU and `CUDA0` only;
- all model layers on CUDA with automatic fitting disabled;
- eight slots with continuous batching and unified F16 KV;
- context size 4096, logical batch 2048, physical microbatch 512;
- flash attention enabled;
- one request per slot and at most eight active requests.

Each model must independently:

- start in a fresh process;
- publish healthy readiness;
- complete one simultaneous B8 cohort;
- return exactly eight generated tokens for every request;
- leave at least 512 MiB of VRAM free after readiness;
- show no process or system swap growth; and
- persist server command, binary/model digests, logs, request rows, and memory
  samples.

Non-co-residency is a physical statement. With one model ready under the
serving envelope, a second full-CUDA server is started on the same GPU. The
pair passes the non-co-residency proof only if the second cannot become ready
without OOM, CPU fallback, reducing the serving envelope, or violating the
512 MiB headroom gate, while the first server remains healthy.

## Frozen switch policy

The policy is identical in every replay:

1. enqueue each arrival into its model FIFO;
2. admit only requests for the published model, up to eight active slots;
3. at a target-change intent, close admission;
4. drain all active requests without cancellation;
5. stop the old server and wait for process exit;
6. prepare the target file cache according to the run regime;
7. start the target server with the same serving envelope;
8. publish the model only after `/health` is ready and memory checks pass;
9. reopen admission and serve the target FIFO.

Intents are never skipped or reordered. Arrivals continue to queue during
drain and load. If the final target leaves requests for another model with no
later intent, the run fails closed.

## Cache regimes and repetition

- `WARM_CACHE`: before the paid trace, both complete model files must have at
  least 95 percent resident pages. No cache refill is permitted during the
  replay.
- `COLD_NVME`: immediately before every target load, the old process must be
  gone, `POSIX_FADV_DONTNEED` is applied to the complete target file, and at
  most 5 percent of its pages may remain resident. Cache preparation time is
  retained in the switch gap.

Run three fresh-process repetitions of each regime in the predeclared order:
`WARM_CACHE, COLD_NVME, COLD_NVME, WARM_CACHE, WARM_CACHE, COLD_NVME`.

## Evidence and metrics

Preserve raw bytes for:

- contract, payloads, source bindings, exact commands, and environment;
- all server stdout/stderr and HTTP completion streams;
- request arrival, dispatch, first token, completion, and error events;
- drain, cache preparation, unload, load, readiness, and publication events;
- 100 ms NVML power, utilization, pstate, and VRAM samples;
- process RSS/swap and system RAM/swap samples;
- cache-residency probes;
- model, binary, source, and result digests.

Derive:

- request queue delay, TTFT, service TTFT, and completion latency;
- per-model completions and tokens per second;
- publication gap for every target change;
- SLO attainment and SLO goodput;
- load, unload, and drain duration;
- peak VRAM, process RSS, free RAM, and swap growth;
- selected-GPU board energy and energy per completed request/token.

`SERVER_WALL_ENERGY` and `TOTAL_SYSTEM_ENERGY` remain unknown because the
desktop has no wall-power instrument and exposes no readable RAPL counter.

## Exit gates

CP0-D passes only if:

- the frozen inputs reproduce byte-identically;
- both model digests and desktop identity match;
- both independent B8 qualifications pass;
- physical non-co-residency is proved;
- all six replays complete all 74 successful requests exactly once;
- every request and switch event is internally ordered and hash-bound;
- cache residency meets the declared regime before every paid load;
- no swap growth, CPU fallback, missing sample, or unbounded queue occurs;
- the independent reducer reproduces all reported metrics from raw evidence;
- the timeline graph is generated only from validated raw evidence.

The result may report either success or failure. No phone acquisition becomes
authorized automatically.
