# Output-quality noninferiority protocol `ws4-quality-gsm8k-ni-v1`

Written 2026-09-30, before any quality run. Frozen parameters live in `protocol.py`; the answer rule in
`gsm8k.py`; the tests in `stats.py`; `build_trace.py` and `score.py` implement the design. Any change to
these after the first confirmatory run needs a new protocol id.

## 1. Why a new criterion

"Exact output match" is not a usable pass rule: two all-desktop runs of `longtail_eval_v2` agree on only
12 of 14 outputs (EVAL_V2_DATA, A4), because batch composition changes the kernels' reduction order and
greedy decoding amplifies the rounding difference at near-tie tokens. The phone arms (11-13 of 14) cannot
be judged against a rule the baseline fails against itself. The question that matters is whether offloading
decode FFN work to the phones lowers task quality. This protocol answers it with an objective task score,
a paired design and a margin fixed before the data.

## 2. Claim, estimand, hypotheses

Estimand: d = p_T - p_B, the difference in GSM8K accuracy (exact match of the final number) between a
treatment arm and the all-desktop legacy baseline, over the pooled Qwen3-14B and Gemma-4-12B items of the
suite (the two models the phones serve), with each arm's run-to-run numerical behaviour included.

- H1 (primary): `paper_config_v1` (frozen full system) vs legacy. H0: d <= -0.03, H1: d > -0.03.
- H2 (secondary confirmatory, the strict test): max-offload arm vs legacy, same margin. Tested only when
  H1 is shown (fixed-sequence procedure, each at one-sided alpha 0.025, family-wise error <= 0.025).
- Noninferiority is shown when Tango's score statistic at d = -0.03 exceeds z(0.975) = 1.96, equivalently
  when the lower end of the two-sided 95 % Tango score interval lies above -3 accuracy points.

The Llama-3.2-1B rows are a control (never phone-assisted in v1): they carry no hypothesis.

## 3. Dataset

GSM8K (Cobbe et al. 2021), MIT license, openai/grade-school-math at commit
`3101c7d5072418e28b9008a6636bde82a006892c`:

| split | rows | sha256 | use |
| --- | ---: | --- | --- |
| test.jsonl | 1,319 | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` | confirmatory items |
| train.jsonl | 7,473 | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` | pilot only |

Why GSM8K: every answer is one integer, so scoring needs no judge and no fuzzy matching; chain-of-thought
solutions are 100-400 tokens, so every item exercises many decode steps (the only phase the phones serve);
reasoning chains are sensitive to a single changed token, which makes it a demanding test of numerical
deviations; and it is a standard benchmark readers know.

A second generation task was considered and deferred. Code generation (HumanEval, MIT, 164 problems) needs
sandboxed execution of model output and has too few items for a noninferiority margin; summarization
(ROUGE, the MLPerf precedent) scores closeness rather than correctness. Either would roughly double the rig
time that GSM8K alone already needs (section 9). The phone path is task-agnostic (the same FFN kernels for
any prompt), and the secondary similarity metrics below measure open-ended output drift on every item.

## 4. Items, models, shards

- One permutation of the test split, `random.Random(20260929).shuffle`, is cut into shards. Shard k holds,
  in this order of the permutation, 2 Llama, 32 Gemma and 32 Qwen items: 64 confirmatory pairs per shard.
  The two large models get disjoint items, so pooled pairs are independent.
- Planned: 8 shards (512 pairs). Blinded interim after shard 4 of every arm; final count 8 to 12 shards
  (768 pairs) by the rule of section 8. Twelve shards use 792 of the 1,319 items.
- Each shard is one campaign trace: Llama control rows at 1 s and 2 s, then the Gemma block, then the Qwen
  block (one model switch per shard). Within a block, arrivals come every 0.8 x the estimated batched
  service time per request (budget x measured s2a period / batch rows: Gemma 0.40 s / 2 rows, Qwen 0.50 s /
  4 rows); the Qwen block starts at 0.9 x the Gemma block's estimated service end. Pacing is chosen only to
  keep runs short; it does not change what is scored.
- Every arm replays the same shard files (sha256 in SUITE.json); the scorer refuses a run whose prompt
  hash, request index or budget differs from the key. Reference answers live only in QUALITY_KEY.jsonl.

## 5. Prompts and decoding

User message, identical for all models:

    Solve the following math word problem. Reason step by step, then write the final answer on its own
    line in the form "Final answer: <number>".

    Problem: {question}

(one line in the file; wrapped here). Each model gets its official chat-template rendering with the
generation prompt; the token codec of the rig adds BOS where the tokenizer wants it:

| model | rendering |
| --- | --- |
| Qwen3-14B | `<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` |
| Gemma-4-12B | `<|turn>user\n{content}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>` |
| Llama-3.2-1B | default system header (Cutting Knowledge Date: December 2023, Today Date: 26 Jul 2024), user, assistant header |

Qwen3 thinking mode: disabled with the template's own `enable_thinking=false` rendering (an empty think
block in the prompt). Thinking traces on GSM8K run 1-3k tokens; any affordable cap would cut most of them
before the answer and make the score depend on the cap. The `/no_think` soft switch is not used: the model
would still have to generate the think tags. Gemma 4's default rendering is likewise non-thinking (empty
thought channel), matching the frozen `longtail_eval_v2` prompts. The builder checks that the codec's
tokens start and end with each template's special-token ids (verified on the real tokenizers 2026-09-30).

Decoding is the runner's, unchanged: greedy (temperature 0.0, argmax), seed = combined request index
(identical across arms, irrelevant at temperature 0), `cache_prompt` false, streamed, and a fixed budget of
`n_predict` tokens with every end-of-generation token banned (`ignore_eos`). Each output is therefore
exactly the budget long, and text after the model's first conclusion is an arbitrary continuation.

## 6. Output budget (pilot, train split)

The budget per model is fixed before any test item runs. Pilot: 24 train items per model (seed 20260930),
all at 512 tokens, legacy arm only, never part of any confirmatory analysis. For each model the budget is
the smallest of 256 / 320 / 384 / 512 whose token prefix holds the first answer (the rule below applied to
the truncated text) of all but floor(0.05 x 24) = 1 pilot outputs; missing outputs count as not held. If
even 512 fails the rule, the budget is 512 and the pilot report flags it. `score pilot` prints the
builder arguments; they are recorded in PROGRESS before the confirmatory suite is built. Prompts are not
revised on pilot results (that would need a new protocol id).

## 7. Scoring

Rule `gsm8k-first-final-answer-v1` (`gsm8k.extract_answer`): the earliest conclusion in the output, never
the last number. A conclusion is "final answer" followed by `:`, `is` or `=` (markdown emphasis allowed)
with a number within 80 characters, or a `\boxed{...}` with a number within 40 characters, whichever
starts first; only if neither exists, the first "answer is" followed by a number within 40 characters.
Numbers may carry a sign, `$` and thousands commas; comparison is numeric (18.00 = 18). No conclusion =
unscorable = incorrect.

Missing outputs (request rejected, absent from RESULT, stream absent or shorter than the budget): primary
analysis uses worst-case imputation (missing treatment = wrong, missing baseline = right); the
complete-case analysis is a sensitivity readout. A comparison with more than 2 % missing pooled items in
either arm is INVALID. For each (arm, shard) the first run that completes with RESULT status PASS is
scored; a failed run is re-run in full; runs are never dropped because of their outputs; every attempt is
logged.

## 8. Test, margin, sample size

Test: Tango (1998) asymptotic score test / interval for the paired difference of proportions (recommended
by Fagerland, Lydersen and Laake 2014, Stat Med 33:2850). The implementation reproduces their Table V
example (Bentur et al.: 95 % CI -0.517 to -0.026) and an independent grid inversion.

**Status 2026-09-30: M is PROVISIONAL (not frozen).** It will be set from independent calibration evidence plus an
explicit application tolerance, and frozen here and in `protocol.py` (with the file hash recorded) before any
confirmatory outcome is examined. Calibration (off the rig, GSM8K train split excluding pilot items, workstream WS9):
original f16/bf16 vs matched Q4 (quantization penalty), Q4 vs Q4-dequantized f16 (execution-format difference), and a
same-format repeat (noise floor). The rig's "f16" artifacts are Q4 weights dequantized to f16, so the confirmatory
desktop-vs-phone comparison on identical weights measures only the OFFLOADING penalty; the quantization penalty is
context, not an allowance: an offloading loss is on top of it. The pilot (24 items per model, one answer = 4.17
points) sets only the output budget and is not evidence for M.

Provisional rationale for M = 3 accuracy points (absolute), to be revisited with the calibration results:

1. The margin must sit above the baseline's own run-to-run noise, otherwise an A/A comparison could fail.
   Greedy-decoding nondeterminism studies of 7-8B instruct models report run-to-run accuracy standard
   deviations of about 0.3-0.9 points on MATH500 across FP16/BF16 runtime configurations (Yuan et al.
   2025, "Give Me FP32 or Give Me Death?", Table 3); the difference of two runs is sqrt(2) x that. Our own
   A/A token identity is 12/14.
2. Three points is small relative to what changes a deployment decision on this task (a model-size step,
   e.g. the Llama-1B control versus the 12-14B models, is tens of points).
3. Feasibility: at the planning discordance, M = 2 needs 1,088 pairs (17 shards per arm, about twice the
   rig time of M = 3); M = 3 needs 512.
4. The lower confidence bound is always reported, so a reader can apply a stricter margin; whether it also
   exceeds -2 points is reported descriptively, without a claim.

Power: exact enumeration of the trinomial of (gains, losses) with the same Tango test (`stats.py`), true
d = 0, one-sided alpha 0.025. The discordance psi (share of items right in one arm and wrong in the other)
drives everything; highly concordant pairs contribute nothing. Planning value psi = 0.05: FP16/BF16
standard deviations above imply psi of roughly 0.01-0.07 on n = 500 (psi ~ 2 n sd^2); our treatment arms
diverge token-wise on 1-3 of 14 outputs (A/A 2 of 14), and only divergences before the answer can flip it.

| pairs | psi 0.02 | 0.03 | 0.05 | 0.08 | 0.10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 (interim) | 0.76 | 0.65 | 0.48 | 0.35 | 0.30 |
| 512 (planned) | 0.98 | 0.94 | 0.82 | 0.64 | 0.55 |
| 768 (cap) | 1.00 | 0.99 | 0.95 | 0.82 | 0.73 |

Required pairs for power 0.8 (grid of 64): psi 0.03 -> 384, 0.04 -> 448, 0.05 -> 512, 0.06 -> 576,
0.07 -> 704, 0.08 -> 768, 0.10 -> 960. Rejection rate at d = -M (the test's size): 0.022 at 512 pairs.
With a true loss of 1 point (d = -0.01) power at 512 pairs is 0.63 / 0.47 / 0.34 for psi 0.03 / 0.05 / 0.08.

Blinded sample-size re-estimation (`score interim`): after shards 0-3 of every arm, psi-hat = discordant
pairs / pairs (complete cases) for each treatment arm; only these totals are shown, never accuracies, gains,
losses or signs. With psi = max over treatment arms, required pairs = the exact rule above; final shards =
min(12, max(8, ceil(required / 64))). The interim cannot stop early or shrink the plan, so it does not
spend alpha; blinded re-estimation of a nuisance parameter keeps the type-I error at its nominal level for
practical purposes.

## 9. Arms

| arm | configuration | role |
| --- | --- | --- |
| legacy | the all-desktop legacy arm of the s2a reference comparison (desktop-baseline selection, legacy dispatcher), trace swapped | baseline |
| paper | `paper_config_v1` template unchanged (two phones) | H1 |
| forced | `paper_config_v1` + max-offload overrides (RUNBOOK) | H2, strict test |
| legacy repeat | optional second legacy pass | A/A noise floor, no test |

The strict arm is valid only if the token-weighted phone-assisted share of its Qwen + Gemma decode steps is
at least 0.90 (per request: busiest layer's phone-call count / (budget - 1), capped at 1; approximate
because batched calls count once per request). Below that, H2 is reported as "strict test not achieved".
No existing configuration key pins the phone fraction; the overrides only remove the energy and latency
reasons for declining the phones.

Shards are run arm-interleaved (shard k of each arm before shard k+1, rotating the arm order), so rig drift
over days hits all arms alike.

## 10. Secondary and descriptive readouts (no claims)

- Per model: counts, Tango interval, z at -M.
- Sensitivity: Newcombe hybrid score interval (method 10), stratified paired bootstrap (10,000 resamples
  within model, seed 20260929), exact McNemar p for d = 0, complete-case Tango.
- Per protocol: pairs whose treatment request had assisted share >= 0.90.
- Similarity to the baseline output (both present): identical outputs, identical through the baseline's
  answer (first divergence at or after the answer's last token), same extracted answer, first divergence
  position of diverged outputs, `difflib` ratio of the answer regions.
- Phone coverage per treatment; Llama control accuracy and identity; lists of lost and gained item ids.
- Optional A/A: legacy repeat versus legacy with the same readouts.

## 11. Limitations

- One run per arm per item: each pair mixes the system effect with one draw of run-to-run noise, which the
  paired test accounts for through psi; it does not separate the two.
- Items sharing a decode batch are not fully independent (one changed batch composition can move several);
  shards are too few for a cluster-robust interval, so intervals may be slightly narrow.
- One task family (grade-school math). Coverage is an approximation from per-request proofs.
- Energy is out of scope; quality runs are not energy evidence (dense arrivals, different trace).
