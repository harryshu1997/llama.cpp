# burstgpt_longdecode_v1: a BurstGPT conversation-log window replayed at its own pace

Source: BurstGPT conversation logs (`burstgpt_3.csv`, sha256 `2299986a07388aa3...`),
one contiguous window of 1800 s starting 0 s after the
first conversation row (scanned 5279 windows for 12 to 20 eligible
requests). Arrival scale 1.0; caps prompt 2048 / output 1100 tokens
(clipped 0 prompts, 0 outputs; source values kept in the rows).

| | Value |
| --- | ---: |
| Requests | 19: 13 ChatGPT -> hot (Qwen), 4 GPT-4 -> cold (Gemma), 2 shortest ChatGPT -> small model (Llama 1B overlay) |
| Replay span | 1625 s |
| Inter-arrival p50 / p90 / max | 56.5 / 236.0 / 391.0 s |
| Prompt tokens p50 / p90 / max | 147 / 1631 / 1779 |
| Output tokens p50 / p90 / max | 152 / 446 / 472 |
| Total prompt / output tokens | 10578 / 3267 (overlay 159 / 13) |

Prompts are the unified trace's templated body cut to the source token count with each model's chat
template, tokenized by the codec against the pinned tokenizer models. The overlay rows are the window's
shortest ChatGPT requests executed on the small model, with the Llama 3 chat template.
The replay preserves BurstGPT's inter-arrival times (times the arrival scale); no route, fraction or
transition is prescribed. Built by `build_realistic_trace.py`; nothing here is a measurement.
