# mooncake_longprompt_fresh_v1: a single user's long-prompt requests (fresh variant)

Source: kvcache-ai/Mooncake FAST25-release/traces/conversation_trace.jsonl (`mooncake_conversation_trace.jsonl`, sha256 `b8cbb061a85206d7...`): 2967 rows with
2000 <= input <= 8000 and 4 <= output <= 512 tokens; 12 sampled
(seed 20261005), paced one at a time with 60 s think time after each estimated completion.

| | Value |
| --- | ---: |
| Requests | 12: 9 hot (Qwen) / 3 cold (Gemma); 1 overlay row |
| Prompt tokens p50 / p90 / max | 5734 / 6407 / 7497 |
| New (uncached) prompt tokens p50 / p90 | 5734 / 6407 |
| Output tokens p50 / p90 / max | 259 / 423 / 475 |
| Prefix-reusable share of prompt tokens | 0.0% |
| Replay span / last estimated completion | 2758 s / 2789 s |
| Prompts whose token count differs from the source | 0 (0 questions padded to 64 tokens) |

Every request carries its own synthesized document: no prefix reuse, the pure prefill-bound case.
Pacing: arrival(k+1) = arrival(k) + new_prefill_tokens x prefill + output_tokens x decode + think with prefill 0.002 s/token and decode
0.65 s/token (desktop-only worst-case ASSUMPTIONS). Bodies are the realistic builder's
templated text cut to the source token count under each model's chat template. Built by `build_longprompt_trace.py`;
nothing here is a measurement.
