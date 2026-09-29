# burstgpt_longtail_v1: a BurstGPT conversation-log window replayed at its own pace

Source: BurstGPT conversation logs (`burstgpt_3.csv`, sha256 `2299986a07388aa3...`),
one contiguous window of 1800 s starting 217800 s after the
first conversation row (scanned 5279 windows for 20 to 32 eligible
requests). Arrival scale 1.0; caps prompt 2048 / output 1100 tokens
(clipped 1 prompts, 1 outputs; source values kept in the rows).
Long-tail rule: first window whose request and output-token long-tail shares are each within tolerance of the log-wide shares (tolerance 0.05). Requests with source output > 512 tokens: log 18.6% of requests / 49.3% of output tokens; this window 16.1% / 49.4% (5 of 31).

| | Value |
| --- | ---: |
| Requests | 31: 16 ChatGPT -> hot (Qwen), 12 GPT-4 -> cold (Gemma), 3 shortest ChatGPT -> small model (Llama 1B overlay) |
| Replay span | 1579 s |
| Inter-arrival p50 / p90 / max | 34.5 / 92.0 / 315.0 s |
| Prompt tokens p50 / p90 / max | 860 / 1480 / 2048 |
| Output tokens p50 / p90 / max | 168 / 638 / 1100 |
| Total prompt / output tokens | 23604 / 8207 (overlay 454 / 129) |

Prompts are the unified trace's templated body cut to the source token count with each model's chat
template, tokenized by the codec against the pinned tokenizer models. The overlay rows are the window's
shortest ChatGPT requests executed on the small model, with the Llama 3 chat template.
The replay preserves BurstGPT's inter-arrival times (times the arrival scale); no route, fraction or
transition is prescribed. Built by `build_realistic_trace.py`; nothing here is a measurement.
