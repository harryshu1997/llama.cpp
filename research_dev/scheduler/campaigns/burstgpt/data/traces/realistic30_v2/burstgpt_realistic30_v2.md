# burstgpt_realistic30_v2: a BurstGPT conversation-log window replayed at its own pace

Source: BurstGPT conversation logs (`burstgpt_3.csv`, sha256 `2299986a07388aa3...`),
one contiguous window of 1800 s starting 46800 s after the
first conversation row (scanned 5279 windows for 20 to 32 eligible
requests). Arrival scale 1.0; caps prompt 2048 / output 512 tokens
(clipped 0 prompts, 5 outputs; source values kept in the rows).

| | Value |
| --- | ---: |
| Requests | 20: 10 ChatGPT -> hot (Qwen), 8 GPT-4 -> cold (Gemma), 2 shortest ChatGPT -> small model (Llama 1B overlay) |
| Replay span | 1599 s |
| Inter-arrival p50 / p90 / max | 56.0 / 141.0 / 477.0 s |
| Prompt tokens p50 / p90 / max | 390 / 793 / 1511 |
| Output tokens p50 / p90 / max | 292 / 512 / 512 |
| Total prompt / output tokens | 7603 / 5939 (overlay 930 / 86) |

Prompts are the unified trace's templated body cut to the source token count with each model's chat
template, tokenized by the codec against the pinned tokenizer models. The overlay rows are the window's
shortest ChatGPT requests executed on the small model, with the Llama 3 chat template.
The replay preserves BurstGPT's inter-arrival times (times the arrival scale); no route, fraction or
transition is prescribed. Built by `build_realistic_trace.py`; nothing here is a measurement.
