# wildchat_conversation_v1: one user's conversations replayed at think-time pace

Source: allenai/WildChat-1M conversations (`wildchat_english_multiturn_sample.json`, sha256 `8fe1c8d40f40f737...`), English, >= 3
turns, not toxic/redacted; seeded order (seed 20261005); one conversation at a time, 90 s between
conversations; context cap 6000 tokens; output cap 512 tokens. Window 3600 s.

| | Value |
| --- | ---: |
| Conversations / turns | 2 / 8 (1 hot (Qwen) / 1 cold (Gemma) conversations; 7 / 1 turns); 1 overlay row(s) |
| Turns per conversation p50 / max | 4 / 7 |
| Prompt (context) tokens p50 / p90 / max | 1021 / 3291 / 4401 |
| New prompt tokens per turn p50 / p90 | 530 / 1110 |
| Output tokens p50 / p90 / max | 512 / 512 / 512 |
| Think time p50 / p90 (s) | 214 / 300 ({'session start': 2, 'timestamps': 6}) |
| Prefix-reusable share of prompt tokens | 63.5% |
| Replay span / last estimated completion | 3456 s / 3551 s |
| Conversations cut by context cap / duration | 1 / 1 (duration reached) |

Pacing: arrival(k+1) = arrival(k) + new_prefill_tokens x prefill + output_tokens x decode + think with prefill 0.02 s/token and decode
0.65 s/token (desktop-only worst-case ASSUMPTIONS, so turns never overlap on any arm);
think time = assistant-stamp gap minus reply chars / 60 chars/s, clipped to
[15, 300] s, default 90 s without stamps.
Prompts are the real transcripts under each model's chat template, tokenized per segment so every turn's prompt is
the token-exact prefix of the next; the dataset's replies stand in the history (identical on every arm). The server
reuses the prefix only with the opt-in ``cache_idle_slots`` launch parameter. Built by `build_conversation_trace.py`;
nothing here is a measurement.
