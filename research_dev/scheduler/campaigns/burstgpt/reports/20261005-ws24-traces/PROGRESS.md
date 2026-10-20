# WS24 progress - 2026-10-05 (fork, offline; no desktop work, nothing committed)

## Built

| file | what |
|---|---|
| `campaigns/burstgpt/ws24_trace_common.py` (new) | chat templates as segments (Qwen / Gemma / Llama overlay, BOS handling), segment tokenizer, `Pacing` (service estimate), `think_time_s`, row constructors, `write_trace` (the loader's four files, manifest byte-compatible with `build_realistic_trace.py`), `role_sequence` |
| `campaigns/burstgpt/build_conversation_trace.py` (new) | WildChat multi-turn builder: pure `build_conversation_trace(sample, codecs, options)` + CLI (`--sample`, caps, pacing, think model, `--execution-artifact` / `--model-inventory-from`) + `<trace>.md` |
| `campaigns/burstgpt/build_longprompt_trace.py` (new) | Mooncake long-prompt builder, `--variant fresh|cached`, shared token-exact document in `cached` |
| `campaigns/burstgpt/tools/fetch_wildchat_sample.py` (new) | datasets-server rows API pagination + filter (English, >= min turns, not toxic/redacted, optional model) + provenance JSON; `--from-fetched` offline path |
| `adapters/llama_server_contracts.py` (changed) | `LlamaServerLaunchContract.cache_idle_slots: bool = False` (validated), `_boolean()` parameter reader, plumbed in `llama_server_launch_contract` and `llama_server_capability_contract` |
| `adapters/llama_server.py` (changed) | `--no-cache-idle-slots` emitted only when `cache_idle_slots` is false (default launch byte-identical) |
| `campaigns/burstgpt/desktop_parent_calibration.py` (changed) | passes `cache_idle_slots` from the selected parent's parameters (default false) |
| `tests/test_ws24_traces.py` (new) | 10 tests, fake codec: prefix exactness + sequential pacing + determinism, think-time derivation, context cap truncation, duration cut / two-conversation rule, loader acceptance (`validate_trace` -> `merge_rows` -> `apply_named_replay_schedule`), long-prompt fresh/cached (token-exact shared document per role), fetch tool offline filter, `cache_idle_slots` contract + launch command |
| `reports/20261005-ws24-traces/SPEC.md` | design, pacing model, caps, template change, desktop rebuild, arms and metrics |
| `reports/20261005-ws24-traces/samples/` | the four built traces (loader files + `.md`), the slimmed WildChat sample (1.6 MB, conversation text only) + provenance |

## Tests (PYTHONPATH = repo root + tests dir, /usr/bin/python3, TMPDIR under /mnt/data_h16t)

```
tests.test_ws24_traces                    Ran 10 tests in 0.133s   OK
tests.test_cuda_graph_mode                Ran 6 tests in 0.156s    OK
tests.test_desktop_parent_calibration     Ran 2 tests in 0.012s    OK
tests.test_elastic_drop_recovery          Ran 47 tests in 8.560s   OK
tests.test_elastic_g1d_recovery           Ran 22 tests in 7.112s   OK
tests.test_elastic_join                   Ran 62 tests in 1.822s   OK
tests.test_host_share_release             Ran 12 tests in 60.159s  OK
tests.test_kv_placement                   Ran 12 tests in 0.147s   OK
tests.test_llama_server_adapter           Ran 41 tests in 31.383s  OK
tests.test_remote_resident_launch         Ran 12 tests in 0.016s   OK
tests.test_remote_resident_reuse          Ran 10 tests in 16.301s  OK
tests.test_scheduler_inventory            Ran 9 tests in 31.449s   OK
tests.test_speculative_rows               Ran 22 tests in 0.043s   OK
tests.test_two_phone_gaps                 Ran 31 tests in 1.668s   OK
tests.test_two_phone_helpers              Ran 31 tests in 3.977s   OK
tests.test_ws23_helper_independence       Ran 17 tests in 9.220s   OK
(also tests.test_kv_lazy_backing, tests.test_split_kv_attention with test_cuda_graph_mode: Ran 15 tests OK)
```
(every module that references the launch contract; `py_compile` clean on the new files; pyflakes is not installed here).

## Sample traces (built on the workstation: `build-cuda/bin/llama-token-codec`, local Qwen3-14B-Q4_K_M sha 500a8806... and gemma-4-12B-it-Q4_0 sha 494518c2... = the paper's pinned tokenizers; overlay tokenizer Llama-3.2-1B-Instruct-Q2_K, rows carry the pinned Q4_0 identity 4b90b1d7...; `--model-inventory-from` the paper's longtail manifest; every sample passes the real loader and the token-exact prefix check)

| sample | options | rows | what came out |
|---|---|---|---|
| `conversation_v1` | 3600 s, caps 6000/512, pacing 0.02 / 0.65 s/token (defaults), seed 20261005 | 8 turns + 1 overlay | 2 conversations: gpt-4 chat 76a948bb hot 7 turns (prompt 23 -> 287 -> 696 -> 1346 -> 2070 -> 3291 -> 4401 tokens, outputs 236 then 512 capped x6, think 72-300 s, cut by the context cap at turn 8) and a7ec85e1 cold 1 turn (cut by the duration). Prefix-reusable share 63.5 %. Replay span 3456 s, last completion 3551 s. Roles 7 hot / 1 cold turns. |
| `conversation_v1_dense` | 3600 s, output cap 256, prefill 0.002 s/token | 13 turns + 1 overlay | same two conversations, 7 + 6 turns (cold conversation prompt 31 -> 775 tokens, outputs 26-153), think p50 135 s, prefix share 65 %, span 3503 s. Roles 7 hot / 6 cold. |
| `longprompt_fresh_v1` | 12 requests, 2000..8000 in / <= 512 out, think 60 s, prefill 0.002 | 12 + 1 | prompt p50 5,735 (3,509-7,497) tokens, output p50 259 (18-475); 9 hot / 3 cold; span 2758 s; no reusable prefix. 2,967 eligible Mooncake rows. |
| `longprompt_cached_v1` | as above, document 4000 tokens | 12 + 1 | identical requests with a shared 4,000-token document per role; new (uncached) tokens p50 1,741; prefix-reusable share 61 %; 2 questions padded to 64 tokens (their prompts are 4,064 instead of the source 3,509 / 3,941); span 2687 s. |

With the default `--duration-s 1800` the conversation builder refuses (fewer than two conversations fit): at the
desktop-only worst-case pace a 512-token reply alone is 333 s and WildChat think times are 70-300 s, so a 7-turn
GPT-4 conversation takes ~50 min. The 3600 s samples are what fits; see the open questions.

## Open questions for the parent

1. **Density vs faithfulness of the pacing.** Open-loop sequential pacing at the desktop-only worst case gives 8-13 turns
   per hour. Choices: (a) accept and run 1 h arms; (b) pace at a measured per-arm decode rate (not comparable across
   arms); (c) closed-loop pacing in the runner (next turn = actual completion + think) - faithful and dense, but a
   runner change and arrival times then differ per arm; (d) shorten think times (x0.5) or the output cap (the `_dense`
   sample). The builder exposes every constant; nothing was decided.
2. **Prefill constant.** The directive's 0.02 s/token is ~10x the desktop-only measurement (1.78 ms/token at 64k
   context). `conversation_v1` keeps 0.02 (default); the other samples use 0.002. Pick one before the arms.
3. **Role balance in short windows.** The first conversation is hot by rule; with two conversations per hour the cold
   role gets one conversation and, in `conversation_v1`, a single turn. A longer window, `--hot-share 0.5`, or starting
   with the cold conversation would balance it.
4. **Think-time ceiling.** WildChat assistant-stamp gaps (minus the reply's generation at 60 chars/s) hit the 300 s cap in
   about a third of the turns; the cap and the 60 chars/s are assumptions.
5. **WildChat stored copy.** 117 conversations (text + stamps only, 1.6 MB) are stored under `samples/` so a rebuild is
   deterministic (verified: the rows rebuild byte-identical from the slimmed copy). Licence: AI2 ImpACT (low-risk
   artifacts); keep attribution in the paper if any text is quoted.
6. **Template for the arms.** A conversation / `cached` arm needs `"cache_idle_slots": true` in both models'
   `runtime_parameters`; that changes the launch contract (and the desktop plan identity), so the server identity run
   should be repeated for that template. The Qwen pool arithmetic (4 x 6000 <= 24,576) holds; Gemma's 8 slots x 6000
   would exceed 32,768 if eight Gemma conversations were retained, but the trace keeps one conversation at a time.
7. **Not done:** no desktop rebuild, no `prepare_trace_inputs_v2` run, no arms (the rig is on the fused A/B). The
   long-prompt `cached` variant is synthesized text; a RAG-realistic document corpus can replace `document_text()` later
   without touching the pacing or the loader contract.

## Parent decisions - 2026-10-05 09:00 UTC (rig session)

| open question | decision |
|---|---|
| 1 pacing density | (a) accept 1 h arms at the desktop-only worst-case pace; conversation window 3600 s |
| 2 prefill constant | 0.002 s/token everywhere (the measured desktop prefill; 0.02 was 10x off) |
| 3 role balance | unchanged (first conversation hot by rule); the single-user story is mostly one model |
| 4 think-time ceiling | unchanged (300 s cap, 60 chars/s) |
| 6 template | `cache_idle_slots` only for the conversation and cached long-prompt templates; spelled **`1`**, not `true`: the runtime catalog's adapter parameters are int or str (`RuntimeModelEndpointCapability._validated_adapter_parameters`), so a JSON boolean passes the configuration check but fails preflight at materialization. `llama_server_contracts._boolean` now accepts 0/1 as well as true/false. The server identity gate pins server/impl sha + worker pins only, so no identity re-run. |

**Hard constraint found on the rig:** the paper's Qwen hot desktop server is a VRAM-fitted capacity parent with
`--ctx-size 4096 --parallel 4 --kv-unified` (desktop baseline plan `MEASURED_DESKTOP_BASELINE_PLANS_V1.json`
overrides models.json's 24576). Every hot-role request must satisfy prompt + output <= 4096, so the v1 samples
above (conversation context cap 6000, Mooncake prompts p50 5.7k) cannot run. Rebuilt as **v2** on the desktop
(`/mnt/storage/ws24-traces/{conversation_v2,longprompt_fresh_v2,longprompt_cached_v2}`):

| trace | options | note |
|---|---|---|
| `wildchat_conversation_v2` | 3600 s, `--context-cap 3500 --output-cap 512`, prefill 0.002 | 4012 <= 4096 |
| `mooncake_longprompt_fresh_v2` | 12 requests, `--prompt-cap 3500 --output-cap 512` | 708 eligible rows (2000..3500 in), input p50 2680, output p50 287 |
| `mooncake_longprompt_cached_v2` | as fresh + `--document-tokens 2000` | prefix-reusable about 2000 of each prompt |

Arms: queues `p9conv`, `p9lpf`, `p9lpc` = `nph=desktop` vs `pix=derived-mpaoafused` (measured placement OP15 Qwen 0 +
Pixel 1-23, fused attention-projection kernel) on the `-ts3` thermal policy, chained after the fused A/B pair `p7f2`
(`$R/rig/block_t_chain.sh` -> `block_t_traces.sh build|templates|arms`, `queue_retry.sh` re-queues a failed arm once).
Analysis: `~/.cache/claude-work/analyze_traces.sh` (fleet energy vs the trace's own desktop arm, TTFT/TPOT/e2e, prefix-cache hit).
A larger hot-server context (8k) needs a new capacity-parent calibration + plan + identity run: follow-up.
