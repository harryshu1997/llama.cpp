# WS24 - traces beyond BurstGPT: a personal multi-turn conversation trace and a long-prompt (RAG) trace

Date: 2026-10-05. Direction (user, 00:10 UTC): build traces beyond BurstGPT; test them once the Pixel saves more
energy; for every new trace report BOTH latency and energy benefits. Nothing in this document is a measurement;
numbers marked ASSUMPTION are builder constants recorded in each trace's manifest.

## 1. Why two traces

| trace | what it stands for | what the phone design does to it | role in the paper |
|---|---|---|---|
| `wildchat_conversation_*` | one person chatting with a local assistant: short questions, long replies, a growing history, think-time gaps | decode-heavy -> the phones' lever (decode FFN + fused projections) applies to most of the token time; the prefix cache makes each turn's prefill small | showcase |
| `mooncake_longprompt_{fresh,cached}` | document / retrieval chat: thousands of prompt tokens, a few hundred output tokens | prefill-bound -> prefill stays on the desktop, the phones touch only the short decode; the `cached` variant shows how much a warm prefix cache gives back | boundary case |

The BurstGPT longtail trace stays the arrival-realism trace (14 requests / 28 min, mean context 330 / 800 tokens).

## 2. Sources

* WildChat-1M (`allenai/WildChat-1M`, not gated). Rows: `conversation_hash`, `model`, `turn`, `language`, `toxic`,
  `redacted`, `conversation[]` with `role`, `content`, `timestamp` (assistant messages only; user stamps are null).
  Fetched with `tools/fetch_wildchat_sample.py` (datasets-server rows API, 100 rows per call); the sample used here is
  rows 0..999 filtered to English, >= 4 turns, not toxic, not redacted: 117 conversations
  (`samples/wildchat_english_multiturn_sample.json`, provenance beside it).
* Mooncake conversation trace (`kvcache-ai/Mooncake`, `FAST25-release/traces/conversation_trace.jsonl`, 12,031 rows,
  59 min): `timestamp` ms, `input_length`, `output_length`, `hash_ids`. Input p50 6.9k / p90 27k tokens, output p50 350.
  Hash-prefix chaining finds only 8 multi-turn chains, so it is used as a LONG-PROMPT source, not a conversation one.

## 3. Conversation trace (`build_conversation_trace.py`)

* One row = one user turn. Prompt = chat template over the full history: user_1, the dataset's assistant_1, ...,
  user_k, assistant opener. The dataset's replies stand in the history, so the prompt is identical on every arm.
* Token-exact prefixes: each turn's prompt is tokenized as the previous prompt's tokens + one new SEGMENT
  (previous reply + closers + new user message + opener), with the codec's BOS stripped from later segments
  (Gemma BOS 2, Llama 128000, Qwen none). Verified on the built samples: every turn's `prompt_tokens` starts with the
  previous turn's, and `history_tokens` equals that length. The server reuses it only with the opt-in launch flag
  (section 5).
* `output_tokens` = the dataset reply's token count under the role's tokenizer, clipped to `--output-cap` (512) and
  raised to `--min-output` (4). `source_output_tokens` keeps the unclipped value.
* Roles: the first conversation hot (Qwen), the second cold (Gemma), then the running share tracks `--hot-share` 0.7.
  At least one conversation per role is required (the loader needs both roles in the manifest).
* Context cap `--context-cap` 6000 tokens (prompt + output): the deployed servers run `--kv-unified` with 24,576
  cells for Qwen (4 slots) and 32,768 for Gemma (8 slots); keeping 4 idle slots' caches needs 4 x 6,000 <= 24,576.
  A conversation is cut when its next turn would not fit (recorded as `conversation_truncated_by: context_cap`).
* Pacing (open loop, sequential by construction; ASSUMPTIONS):
  `arrival(k+1) = arrival(k) + new_tokens(k) x prefill_s_per_token + output_tokens(k) x decode_s_per_token + think(k+1)`,
  `--session-gap-s` 90 between conversations, one conversation at a time.
  Defaults: prefill 0.02 s/token, decode 0.65 s/token (desktop-only Qwen worst case, so turns never overlap on any arm;
  faster arms idle longer, which is what a single user sees). The desktop-only 64k-context gate measured prefill at
  1.78 ms/token (57,871 tokens in 102.9 s); `samples/conversation_v1` keeps the 0.02 default, the `_dense` conversation
  sample and both long-prompt samples were built with `--prefill-s-per-token 0.002` (see PROGRESS.md).
* Think time (ASSUMPTION): WildChat stamps assistant messages only. think = (stamp_{k+1} - stamp_k) - reply_chars / 60,
  clipped to [15, 300] s; 90 s when a stamp is missing. The 60 chars/s stands for the API's generation speed in 2023.
* Overlay: >= 1 row (the shortest first user message of an unused conversation) on the Llama 1B overlay model, placed
  5 s after the first conversation ends. Overlay rows carry the EXECUTION artifact's identity from the model inventory
  (the paper's pinned Q4_0 Llama), not the local tokenizer file's; the tokenizer is quantization-independent.
* Row analysis fields: `conversation_id`, `turn_index`, `turns_total`, `history_tokens`, `new_tokens`, `think_s`,
  `think_source`, `service_estimate_s`, `conversation_truncated_by`, `source_*`.

## 4. Long-prompt trace (`build_longprompt_trace.py`)

* N (12) Mooncake rows with 2000 <= input <= 8000 and 4 <= output <= 512 tokens (2,967 eligible), seeded sample,
  roles by `--hot-share` 0.75 (>= 1 per role), paced like section 3 with `--think-s` 60 after each estimated completion.
* `fresh`: each prompt is the realistic builder's templated body cut to the source token count (unique per request):
  no reusable prefix.
* `cached`: every request of a role starts with the SAME 4,000-token synthesized document (token-exact, document
  first, then a unique question of `input_length - 4000` tokens, at least 64). `history_tokens` = 4000 from the second
  request of a role on (the first pays the full prefill); `prompt_tokens` differ from the source only when the question
  had to be padded to 64 tokens (2 of 12 in the sample).
* Overlay: one synthesized 48-token / 32-token row 5 s after the first request's estimated completion.

## 5. Template change: `cache_idle_slots` (opt-in, byte-identical default)

`LlamaServerLaunchContract.cache_idle_slots: bool = False`; when true the launcher omits `--no-cache-idle-slots`
(everything else unchanged, `--kv-unified` kept). It is read from the model's `runtime_parameters` (models.json) or
the desktop plan's `adapter_parameters` as a JSON boolean (anything else is refused), flows through
`llama_server_launch_contract` / `llama_server_capability_contract` / `desktop_parent_calibration`, and makes contracts
unequal, so a running server launched without it is never reused for a plan that needs it. A conversation or
`cached` arm sets `"cache_idle_slots": true` in its template's `runtime_parameters` for both models; the default
template stays byte-identical. Pool arithmetic to respect: retained idle caches share the unified KV pool
(4 x context cap <= 24,576 for Qwen).

## 6. Rebuilding on the desktop (pinned artifacts)

The samples in `samples/` were built on the workstation with `build-cuda/bin/llama-token-codec` and the local
tokenizer models (Qwen3-14B-Q4_K_M sha 500a8806..., gemma-4-12B-it-Q4_0 sha 494518c2... = the paper's pinned
tokenizers; Llama-3.2-1B-Instruct-Q2_K for the overlay tokenizer, its rows carry the pinned Q4_0 identity) with
`--model-inventory-from paper_config_v1/trace/longtail_eval_v2/TRACE_MANIFEST.json`. On the desktop use the paper's
pinned files instead:

```
python3 -m research_dev.scheduler.campaigns.burstgpt.build_conversation_trace \
  --sample <samples>/wildchat_english_multiturn_sample.json --output-dir <out> --duration-s 3600 \
  --prefill-s-per-token 0.002 --codec $CODEC --library-dir $LIB --qwen-tokenizer-model $QTOK \
  --gemma-tokenizer-model $GTOK --llama-tokenizer-model $LTOK \
  --execution-artifact qwen3-14b-q4km-dequant-f16=... --execution-artifact gemma-4-12b-q40-dequant-f16=... \
  --execution-artifact llama-3.2-1b-instruct-q4_0=...
```
(same for `build_longprompt_trace.py --variant fresh|cached --prefill-s-per-token 0.002`). The output is deterministic
for a given sample, tokenizers and options, so a desktop rebuild must reproduce the `base_trace.sha256` of the sample
manifests when the same arguments are used. Trace inputs are then prepared like the BurstGPT ones
(`prepare_trace_inputs_v2.py` with the new trace directory and name; the campaign template needs `cache_idle_slots`).

## 7. Running the arms and what to report

Arms per trace (same measured placement as the fused A/B, 3 interleaved repeats, cool-rig gate):
desktop-only (no phones) / phones FFN-only / phones fused. For the long-prompt trace add `cached` vs `fresh` on
the desktop-only arm (the prefix-cache effect alone).

Report per trace and arm:
1. per-turn (or per-request) TTFT: first-token time from the request's arrival (prefill + queue);
2. decode ms/token (median over turns, and per-conversation growth with context);
3. fleet kJ (host CPU package + GPU board + both phones' metered energy) and kJ per generated token, vs desktop-only;
4. for the conversation trace: the prefix-cache hit (server `prompt_n` vs `input_tokens`; `history_tokens` is the
   expected reusable part) so the saving is attributed correctly;
5. identity: desktop-majority token identity across arms as for the BurstGPT runs (greedy decoding).

The RAG expectation to state in advance: `fresh` saves little or loses (prefill dominates; the desktop GPU does it);
`cached` recovers part of it; the conversation trace should show the largest decode-side saving of the three.
