# Sustained assistance: incumbent continuity and small matched experiment

Status: small matched experiment complete at 2026-09-09 17:24 EDT. Adaptive and
fresh Desktop-CUDA-B0 each completed 3/3 requests, with semantic and terminal
checks passing. Canonical `compare_ab.summarize` reports `comparison_valid=true`
and `target_met=true`. The post-replacement coverage limit below remains open.
No 24- or 84-request trace was run. No physical campaign remains active.

Implementation was frozen before either arm: 257 focused tests PASS in 87.740 s,
preflight PASS, and both replay goldens unchanged. The resumed control uses that
same frozen source, qualification, artifacts, binaries, desktop parents, prompts,
seeds, token counts, arrivals, graph modes, and energy boundary. Only its output
directory differs from the interrupted control command.

## Measured small result

| Phone active power assumption | Desktop fleet kJ | Adaptive fleet kJ | Fleet saving |
| --- | ---: | ---: | ---: |
| 3 W | 28.040510 | 19.924717 | 28.94% |
| 4.5 W | 28.040510 | 20.067684 | 28.43% |
| 6 W | 28.040510 | 20.210652 | 27.92% |

Paid duration: desktop 346.132091 s, adaptive 301.452163 s, a 44.679928 s reduction.
Runtime preparation, exploration, queueing and idle time are included, without
subtracting preparation afterward. CPU/GPU energy is physically measured by
RAPL/NVML. Desktop CPU/GPU energy is 18.380867/9.356777 kJ; adaptive is
10.607024/8.851384 kJ. Phone energy is assumed, with identical 0.875 W idle
treatment. Nominal phone energy is 0.302866/0.609276 kJ, desktop/adaptive.

| Request | Applied positive FFN fractions | Phone-assisted output tokens | Fraction-weighted eligible coverage | Phone calls |
| --- | --- | ---: | ---: | ---: |
| Gemma 36 | 100%, 75%, 50%, 25%; incumbent 100% | 243/292 (83.22%) | 76.89% | 5,352 |
| Llama 37 | None; desktop route | 0/292 | Not eligible | 0 |
| Qwen 50 | 100% | 59/71 (83.10%) | 84.29% | 354 |

Eligible decode denominators are 291/70 for Gemma/Qwen, excluding the first
output token. Any-assistance coverage within those denominators is 83.51%/84.29%.
Weighted numerators are 223.75/59 token equivalents. Native control generations,
column widths and per-layer call counts match the current observation groups and
terminal proofs. No inferred or unmeasured tail is counted as measured cost.

Fractions apply only to the current CPU-resident FFN layer mask. Gemma executes
layers 0-15, expands to 0-23, then retains 0-15 during replacement. Qwen executes
layers 12-17. This is not 100% model offload, nor equal FFN work across models.
Desktop parents remain the qualified Gemma GPU22 and Qwen GPU16 placements.

| Request | Desktop service s | Adaptive service s | Desktop delay before execution s | Adaptive delay before execution s |
| --- | ---: | ---: | ---: | ---: |
| Gemma 36 | 155.216 | 140.449 | 76.341 | 19.291 |
| Llama 37 | 1.736 | 1.799 | 174.949 | 102.008 |
| Qwen 50 | 55.375 | 54.362 | 197.010 | 150.679 |

Service includes prefill and decode but excludes preceding queue/preparation.
The before-execution column includes both queue and preparation, not queue alone.
Gemma service is 9.51% shorter; Qwen 1.83% shorter; Llama 3.59% longer. All are
within the configured relative 25% allowance. Scheduler decision latency is
213.116 ms median, 980.852 ms maximum over five adaptive decision records.

This is one completed matched pair with diagnostic attribution, not a replicated
or steady-state result. Startup variability is material: acquisition-to-execution
took 75.353/18.294 s for Gemma and 50.260/76.638 s for Qwen, desktop/adaptive.
Those intervals include preparation and dispatch work; their differences cannot
all be attributed to phone assistance. OS page-cache state was not controlled.
The 28.43% result therefore is an end-to-end observation for this small workload,
not an isolated decode-only effect of this patch or evidence for the long trace.
No independent steady-state saving or break-even count is inferred by subtracting
overlapping preparation/request windows. Preflight, artifact transfer and common
bootstrap warmup precede both paid boundaries.

## Session continuity and zero-assistance evidence

Phone preparation begins at 1.997804 s, 0.984923 s after its first proposal.
Desktop execution begins at 20.290919 s, before the first published phone READY
at 20.492149 s. The initial phone preparation and desktop preparation overlap.
Gemma attaches at 38.333100 s and receives its first positive acknowledgement
at 41.779728 s, while the third session is still preparing. Qwen attaches before
execution and receives its first positive acknowledgement at 260.694993 s.

| Physical session operation | Physical load-to-READY s | Scheduler READY time s | Session generations after publication |
| --- | ---: | ---: | --- |
| First Gemma HTP0 | 9.377 | 20.492 | 1 / empty / empty |
| Add Gemma HTP1 | 16.073 | 38.287 | 1 / 1 / empty |
| Add Gemma HTP2 | 10.718 | 49.813 | 1 / 1 / 1 |
| Replace selected HTP2 with Qwen | 14.227 | 140.397 | 1 / 1 / 2 |

Four physical session loads total; only HTP2 reloads. All four preparations reach
READY, with no failed transition. During HTP2 load, retained HTP0/HTP1 each make
264 calls, with 1,648 before and 32 after that physical interval. Their artifact,
shard and generation remain unchanged. Totals by session are 1,944/1,944/1,818;
HTP2's total comprises 1,464 Gemma calls at generation 1 and 354 Qwen calls at
generation 2. This proves retained execution during loading, not independent HTP
compute or a newly measured maximum-gap bound. Reverse/rollback were covered by
the focused regressions, not injected again in this three-request workload.

The existing terminal audit records zero fallback, USB reset or execution
recovery. Both arms report three desktop model loads/transitions, separate from
the adaptive phone-session loads above. One startup telemetry deferral recovers
after 1.375101 s; two early helper rematerialization attempts fail closed under
unavailable telemetry and recover. They are preserved, not relabeled successful.

Gemma selects its valid 100% incumbent in 35 logged decisions after rejecting
50%/25% challengers; Qwen retains 100% in 13 decisions through completion.
Maximum observed attempts per candidate/context is one, below the configured
bound of two. These intent counts are corroborated by native calls; intent alone
does not authorize physical execution.

Every recorded adaptive zero-policy interval has an exported decision reason:

- Gemma: 3 decode tokens unavailable helper, 4 initial baseline, 3 awaiting
  acknowledgement, and 38 insufficient opportunity after the replacement.
- Qwen: 8 initial baseline and 3 awaiting acknowledgement; no unassisted tail.
- Llama: unsupported helper route, derived from its desktop execution contract;
  it is not mislabeled as an adaptive measured rejection.

Remaining limit: replacement completion refreshes Gemma's helper component and
candidate identities. At token 251, 41 output tokens and only 14 request-wide
probe tokens remain; complete-pair revalidation is not admitted, and the final
38 tokens run at 0%. Retained sessions were physically usable, but the refreshed
context has no accepted incumbent. This is incomplete/unaffordable revalidation,
not evidence that Gemma assistance is energy-negative. The next focused work is
retained-mask incumbent/evidence continuity across exact compatible rebinds, with
bounded revalidation where identity really changes. Do not raise the cap or reuse
incompatible qualification merely to fill this tail.

## Physical artifact provenance

All paths below extend `/home/zhihao/s42-sustained-assistance-20260909-v1`.
The original user-interrupted control at `-desktop/run` remains untouched and is
excluded; the completed control is `-desktop-resume1/run`.

- Adaptive `-adaptive/run/RESULT.json`: `4728ddd2262c867929cf5ad36edd2e98506271d80cf385904d8d97a3a07f2eff`
- Desktop `-desktop-resume1/run/RESULT.json`: `87d030b9a2a078143f2ab135baac2678e0dad0f80e03997e6bca68d2251543a7`
- `-resume1-inputs/COMPARISON.json`: `505aab2697d077b53387ac19eab769603802931eb771f9bcb5d912b7e26a2f5f`
- `-resume1-inputs/MATCHED_SUMMARY.json`: `4fa5dc44eee61b17729a49e1d0ec65a80488b4860481b479de17aa36376297ca`
- `-inputs/ADAPTIVE_NATIVE_AUDIT.json`: `2d1ba0a910d2c2085e8ee7636740a4924745974dfa41d5e9a91b729a99afb941`
- `-inputs/SOURCE_MANIFEST.json`: `2809b31b9d6ef13c2df59b93049354532df7bc3229aefa40e3d9b3c10cae27ca`
- `-inputs/Desktop-CUDA-B0.json`: `143e8d25a754bdc5cef645ef2dbdf4681adbb663ad5d7f70c538fe602916e937`
- `-inputs/FINAL_SAVED_REPLAY.json`: `4eed100367fa44dedd74426aaa98ba1abbc00fa5f4be0ab8730ac42f8c7dff38`
- Interrupted `-desktop/run/FAILURE.json`: `e2f2593c4ebebaf7a89edaa359e021d0aef8316dc97ec45d69a32d9d07e2af6d`

`COMPARISON.json` carries all checked phone-worker, router, session-script,
parent-model and deployed FFN-shard hashes. Each RESULT carries host binary,
artifact, desktop parent, current execution ticket and terminal-proof identities.
The source manifest pins all 216 deployed Python/shell files as well as HEAD
`99449bafade0b2c15de4410feda832035c2f2d83`; HEAD alone is not used as the dirty-tree
identity. No scheduler source changed between the two completed arms.

Resume bookkeeping lives in `-resume1-inputs/RESUME_CHECK.json`, `COMMAND.json`,
`COMMAND_MANIFEST.json` and `DESKTOP.log`. The artifact-only first resume launcher
duplicated a SHA prefix and stopped before inference; its original script is
preserved alongside the corrected launcher. The earlier preflight/freeze float
serialization error likewise happened before inference and was corrected only in
artifact metadata. No failed artifact was overwritten or promoted to PASS.

## Policy change

The adaptive session now keeps an accepted incumbent, a challenger, and the
physically acknowledged policy separately. A rejected or incomplete challenger
does not erase a compatible, still-beneficial incumbent. Complete-pair admission
is shared by fresh exploration, cached verification, and retries. It budgets
warmup, both measurements, control acknowledgements, expected incremental energy,
and a remaining exploitation window. Retry counts are bounded per candidate and
context; the existing request-wide probe-token count and a monotonic estimated
exploration-overhead high-water mark survive membership changes.

Continuation compares future desktop and assisted energy, the configured margin,
and a new switching cost only when the accepted policy is not already physically
applied. It does not require future tokens to repay sunk exploration overhead.
Every original energy receipt remains in whole-request and paid-interval totals.
Estimated exploration overhead is a budget guard, not a new physical energy meter.
An unmeasured candidate can use an explicitly labeled uncertainty-adjusted baseline
prior for bounded admission; that prior cannot qualify a policy. Missing baseline
energy defers admission until an estimate or observation is available.

LEARNING now uses the same configured latency allowance as other operational
selection. In this campaign that is 1,250,000 ppm, allowing up to 25% relative
latency regression subject to the existing energy margin and deadline treatment.
The former LEARNING-only strict-faster condition is removed deliberately. This
does not override an actual rejection under the configured allowance. Physical
control/observation failure is incomplete cost evidence, not measured negative
energy; the existing recovery and exact execution checks still apply.

Every adaptive decision exports an ASSISTANCE_DECISION event, including its
incumbent, challenger, acknowledged policy, context identity, remaining opportunity,
probe budget, evidence counts, and reason. These are intentions, not physical
acknowledgements; existing FRACTION_APPLIED events and terminal/native proofs
remain the execution authority. Unsupported non-adaptive routes need separate
classification from their execution contracts.

## Saved-window checks

| Saved case | New decision |
| --- | --- |
| Gemma 44, token 45, 220 tokens remaining | Collect the missing current baseline; do not invent qualification |
| Gemma 49, token 67 | Reject 25% and retain the measured 100% winner |
| Qwen 43 after batch 2-to-1, token 97 | 100% and 75% pass future-energy and configured-latency checks; 50% and 25% do not qualify |

These are decision reconstructions from saved physical windows, not newly measured
counterfactual energy. Qwen's 100% route was previously rejected for being about
1.5% slower despite the declared 25% allowance; the 75% route was blocked by sunk
exploration payback. Its 65 already-spent probe tokens remain counted.

Both canonical replay goldens are unchanged:

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

Focused coverage includes early completion, bounded retries, unavailable sessions,
incumbent/challenger separation, configured latency rejection, unknown/insufficient
energy admission, membership changes, fresh leases, session masks, rollback,
terminal tails, cancellation cleanup, and historical generation proofs. No complete
scheduler harness was run. A latency-only fixture now uses consistent synthetic
energy units; failure classification and fake-controller fixtures were updated for
the explicit new outcome/event contracts without relaxing physical assertions.

## Frozen small workload

The existing `burstgpt_dev3_long_v1.json` is unchanged:

| Request | Arrival | Output tokens |
| --- | ---: | ---: |
| Gemma 36 | 1 s | 292 |
| Llama 37 | 61 s | 292 |
| Qwen 50 | 91 s | 71 |

Trace SHA-256: `c63ae8c85b579ead40b8bf07e0abc4c20d48cbeb1a198fe0fb4397dedbe8f173`.
Normal energy-aware scheduling chooses the adaptive routes, sessions and fractions.
There are no forced assignments, fractions, arrival changes or future-demand hints.
Cold runtime preparation remains inside each complete paid boundary. This is not
an OS-page-cache-cold experiment or a verified warm-start measurement.

Fresh physical prefix: `/home/zhihao/s42-sustained-assistance-20260909-v1`.
Inputs, commands, source manifest, preflight, frozen Desktop-CUDA-B0 identity and
read-only audits are under `-inputs`; completed runs use `-adaptive/run` and
`-desktop-resume1/run`, with the interrupted `-desktop/run` preserved.
Desktop-CUDA-B0 is a newly run, same-build CUDA control with Qwen/Gemma graphs
explicitly disabled, not default llama.cpp and not the old graph-disabled result.
Both completed arms passed the exact comparison checks described above.
CPU/GPU energy is physically measured; phone sensitivity uses 3/4.5/6 W with the
same 0.875 W idle treatment in both arms.

Incremental production changes:

- `_internal/adaptive_decode.py`
- `_internal/adaptive_decode_contracts.py`
- `_unified/adaptive_decode_control.py`

Focused tests: `test_sustained_assistance.py`, `test_adaptive_decode.py`, and
`test_adaptive_runtime.py`. The physical adapter, helper preparation, session
replacement, envelope authorization, terminal validators, cleanup, native binaries,
shard formats and memory limits are unchanged. No commit, push, PR, or unrelated
worktree edits.

The only repository edits during resume were this report and `research_dev/talks.md`.
Artifact-local launch/check scripts select no scheduler routes or fractions. No
focused or broad test suite was repeated on resume because the tested production
file hashes were unchanged.
