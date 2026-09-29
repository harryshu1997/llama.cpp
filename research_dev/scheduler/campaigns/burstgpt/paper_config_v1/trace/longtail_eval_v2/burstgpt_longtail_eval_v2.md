# burstgpt_longtail_eval_v2: a BurstGPT conversation-log window replayed at its own pace

Source: BurstGPT conversation logs (`burstgpt_3.csv`, sha256 `2299986a07388aa3...`),
one contiguous window of 1800 s starting 156600 s after the
first conversation row (scanned 5279 windows for 14 to 18 eligible
requests). Arrival scale 1.0; caps prompt 2048 / output 1100 tokens
(clipped 0 prompts, 0 outputs; source values kept in the rows).
Long-tail rule: first window whose request and output-token long-tail shares are each within tolerance of the log-wide shares (tolerance 0.05). Requests with source output > 512 tokens: log 18.6% of requests / 49.3% of output tokens; this window 21.4% / 49.5% (3 of 14).
Development bound: windows with more than 4000 output tokens (after the output cap) were skipped; this window has 3604.

| | Value |
| --- | ---: |
| Requests | 14: 7 ChatGPT -> hot (Qwen), 6 GPT-4 -> cold (Gemma), 1 shortest ChatGPT -> small model (Llama 1B overlay) |
| Replay span | 1675 s |
| Inter-arrival p50 / p90 / max | 113.0 / 277.0 / 283.0 s |
| Prompt tokens p50 / p90 / max | 210 / 670 / 1961 |
| Output tokens p50 / p90 / max | 236 / 594 / 614 |
| Total prompt / output tokens | 5385 / 3604 (overlay 54 / 46) |

Prompts are the unified trace's templated body cut to the source token count with each model's chat
template, tokenized by the codec against the pinned tokenizer models. The overlay rows are the window's
shortest ChatGPT requests executed on the small model, with the Llama 3 chat template.
The replay preserves BurstGPT's inter-arrival times (times the arrival scale); no route, fraction or
transition is prescribed. Built by `build_realistic_trace.py`; nothing here is a measurement.
