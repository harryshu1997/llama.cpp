# Probe continuity: bounded PASS, reduced coverage FAIL

The Qwen 42/43 concurrency gate passes on the corrected scheduler. This proves
the control mechanism, not matched fleet-energy savings. Request 41 is an
original 24-token cold-setup request; all setup time and energy remain recorded.
The reduced 24-request adaptive run subsequently completed, but coverage
regressed. The user stopped its fresh matched desktop control after 10/24
completed streams to prioritize the remaining fixes. There is no valid new
matched energy comparison. Follow-up: `../20260909-probe-recovery/README.md`.

## What changed

- Admitted learning probes remain authorized after control acknowledgement and
  warmup, subject to the existing resource, safety, time, and request-token caps.
- Live server membership is refreshed at decode-window boundaries. Crossing
  windows retain accounting but cannot qualify a policy. A membership change
  does not replenish the request-wide exploration cap.
- Helper attachment and probing use the same complete-pair affordability check.
- Deferred fresh controls reserve a complete pair at their actual retry boundary.
  An unaffordable retry clears its pending policy rather than bypassing admission.
- Control-delay samples are scoped to the current membership epoch. Historical
  samples crossing a membership change are not used as comparable control costs.
- Exact request FFN counters are captured before the live membership query can
  outlive a terminal slot. Terminal rejection errors now include failed fields.
- The declared 10000-ppm energy margin is preserved by catalog materialization;
  the previous overlay silently made it 50000 ppm. No qualification is fabricated.

No native binary, graph mode, placement, artifact, shard format, phone generation,
lease validation, memory limit, replacement, or rollback implementation changed.
No route, session, or fraction was forced in the runner.

## Physical result

The same original Qwen requests were used in v2 and v3. This before/after table is
a mechanism comparison across scheduler revisions, not an energy A/B result.

| Request 43 metric | v2, failed measurement | v3, corrected |
| --- | ---: | ---: |
| Live 2-to-1 context observed at token | 59 | 64 |
| Valid phone probes after that change | 0 | 3 |
| Phone calls | 180 | 5,436 |
| Tokens with physical phone execution / eligible tokens | 10 / 340 | 302 / 340 |
| Fraction-weighted eligible-token coverage | 2.94% | 82.43% |
| Terminal proof and semantic output | PASS | PASS |

The v3 native audit matches every positive-policy token and weighted fraction to
request-and-control-generation-keyed native calls. Coverage describes FFN column
assistance within the 18 resident CPU-layer mask, not the entire model workload.

| v3 timeline, seconds from paid start | Request 42 | Request 43 |
| --- | ---: | ---: |
| Execution starts | 84.468 | 84.455 |
| Execution completes | 163.694 | 309.204 |
| Output tokens | 60 | 341 |
| Phone calls | 90 | 5,436 |
| Fraction-weighted coverage | 8.47% | 82.43% |

At 164.907 s, request 43 records the context change at token 64. Its crossing
window is non-comparable. A fresh singleton baseline precedes valid 100%, 75%,
and 50% probe windows at tokens 79-83, 90-94, and 101-105. There are 58 subsequent
conservative-measured positive helper selections. The next probe is declined at
token 105 as `INSUFFICIENT_OPPORTUNITY`; the controller retains the measured
winner instead. The cumulative probe count is 65 at the last membership event,
below the unchanged 80-token cap. The final selected fraction is 100%.

All 3/3 requests, including setup, complete with accepted semantic outputs and
exact terminal proofs. Native audit: 5,526 total calls; zero fallback, USB reset,
or execution recovery. All three phone sessions retain generation 1. Each loads
once and has no subsequent reload. Load-authorized-to-READY durations are 11.402 s,
47.644 s, and 19.794 s for HTP0/1/2. These variable physical preparation costs have
not been optimized by this fix. The whole paid gate lasts 314.394 s.

## Tests and saved-state diagnosis

152 focused helper/adaptive/adapter/replay tests pass in 85.902 s, including three
new regressions that failed before the final budget correction. No full harness
was rerun during these incremental fixes. Both 40-request replay oracles retain
their existing hashes:

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

Reconstructing v2 request 43 at token 59, including only history completed by that
boundary, reproduces rejection with the old 9.434196-second control-delay sample.
With the correct context scope, a complete pair needs 7.140276 s within a
32.287854-second allowance, with 282 tokens remaining. This is a saved-state
budget diagnosis; it does not synthesize phone measurements or qualification.

Changed scheduler files:

- `_internal/adaptive_decode.py`
- `_internal/adaptive_decode_contracts.py`
- `_unified/adaptive_decode_control.py`
- `_unified/helper_preparation.py`
- `adapters/http_backend.py`
- `adapters/llama_server.py`
- `campaigns/burstgpt/catalog.py`
- `tests/test_adaptive_decode.py`
- `tests/test_llama_server_adapter.py`

Documentation: this report and `research_dev/talks.md`. Existing unrelated dirty
changes and every earlier physical artifact remain intact. No commit or push.

## Reduced 24-request result: execution PASS, coverage FAIL

The new adaptive run completes 24/24 with accepted output and exact terminal
proofs. Native call reconciliation passes. All four layouts prepare and reach
READY, with zero transition failures, unprepared proposals, fallback, USB reset,
or execution recovery. Only 4/24 requests perform phone work, however.

| Model | Requests with phone work | Physical calls | Token-position coverage | Fraction-weighted coverage |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 1/15 | 828 | 8.80% | 6.03% |
| Gemma | 3/6 | 2,968 | 34.19% | 29.91% |
| Llama | 0/3 | 0 | Not eligible | Not eligible |

Llama has no supported helper route here. Coverage is over the directly exported
eligible interval, with current group/ticket and native-call identity checked.
Qwen retains 12 resident CPU FFN layers after replacement; Gemma has eight.
Fractions describe columns within those masks, not the entire model workload.

The before/after analyzer comparison is diagnostic, not same-revision energy A/B:

| Analyzer metric | Previous 20260908 reduced run | New 20260909 reduced run |
| --- | ---: | ---: |
| Layouts proposed / READY | 4 / 4 | 4 / 4 |
| Failed / never prepared | 0 / 0 | 0 / 0 |
| Phone without any layout | 20.8 s | 22.4 s |
| Executing work without model residency | 2.5 s | 2.6 s |
| Requests with physical phone work | 14 | 4 |
| Attached but not probed | 0 | 8 |

The analyzer's short-request classification uses its default token threshold;
it is not a reconstruction of each controller's dynamic complete-pair budget.
The eight attached/no-probe cases need decision-level explanations, not a claim
that phone energy was measured negative. Qwen/Gemma's first covering residency
appears 22.4/15.6 seconds after first arrival. One HTP1 replacement reaches READY
in 13.154 seconds from LOAD_AUTHORIZED. Retained HTP0/2 keep generation 1, but
there were no Qwen calls during that load, so this run is not an interference
or retained-call-gap proof. There was no reverse transition in normal demand.

Saved-state diagnostic `-inputs/COVERAGE_DIAGNOSTIC.json` establishes:

- Qwen 43 does complete valid post-concurrency probes. At token 97 its 100%
  candidate is slightly slower than the current baseline (636919 versus
  627610 us/token), failing the existing LEARNING latency-improvement rule.
  The 75% candidate improves measured latency and energy but does not repay
  the conservative accumulated exploration cost plus margin. No measured
  rejection or exploration cap is relaxed to obtain coverage.
- Gemma 44 starts positive probes using historical baseline evidence, then
  fails qualification because it has no valid current baseline window. The
  initial-start and final-qualification requirements disagree. It has 220
  tokens left when the four-fraction sweep ends, but remains at 0%.
- Gemma 49's token-67 `PROBE_INCOMPLETE` is reproducible without an expired
  reservation: its quarter-fraction probe still has token limit 77 and a
  40.274-second allowance. `_update_elimination` marks it `ENERGY_DOMINATED`
  against the already measured 100% candidate, then `_probe_admitted` becomes
  false. The warmup branch labels that as incomplete and returns to 0% instead
  of selecting the still-valid winner. The existing qualification checks pass
  for that 100% candidate at the same boundary, with 424 tokens remaining.
- Separately, an initial unaffordable fresh probe enters `EXPLOITING`; that
  state's next-window branch does not retry after new baseline evidence. The
  saved zero-call requests show this state sequence. Reconstruct their exact
  candidate budgets before attributing every such request to this path.

This is a read-only diagnosis using completed physical windows and history
available by each boundary. It synthesizes no measurements or qualification.
No production changes followed the deployment freeze. The next correction must
distinguish candidate elimination, incomplete measurement, and whole-request
rejection, while preserving the same safety checks and request-wide probe cap.

Reduced artifacts: `/home/zhihao/s42-normal-reduced24-20260909-v1` with suffixes
`-gate/run`, `-desktop/run`, and `-inputs`. Adaptive RESULT SHA256:
`5d97d6ae4ad92c324aa8e76b7aada21d83c18926aca3c62cbab937d6b1b6673e`.
Source manifest: `d20bc67e21514011e9bc6085d49d08bfd99576733925e00b555e0d57238d86f4`.
Preflight: `e3cb358acd7410b9824005a04e335aedf3199063d0fc9e9dc58ff73802eb34ed`.
The desktop control was stopped by SIGINT on the user's instruction. Its
KeyboardInterrupt failure artifacts and `-inputs/STOPPED_CONTROL.json` are
preserved. No matched savings result is available from this interrupted pair.

## Bounded gate artifacts

Desktop prefix: `/home/zhihao/s42-probe-continuity-20260909-v3`.
`-gate/run/RESULT.json`, `ADAPTIVE_DECODE_OBSERVATIONS.json`, native stderr,
snapshots, streams, and terminal session proofs are preserved.
`-inputs/CONCURRENCY_AUDIT.json`, `NATIVE_AUDIT.json`, and
`V2_BUDGET_DIAGNOSTIC.json` contain the decoded evidence and audit details.

SHA256:

- v3 RESULT: `3d7189771658cedbb3d3a7ca02dccdb67bd531e009bad5bf95d335382069d39e`
- v3 source manifest: `b7209dc3d47f7c0d59018606bc6971a0277c7010f4e36d0064963f7a54c79395`
- v3 preflight: `871290ce8b0db7441644d525fd837f1f01fb9cd73735696a34a2eac8d872599c`
- preserved v2 RESULT: `1357642b5a6a44a69d45ec7f62c388479e014577d35c1e3ba29aa25e5ed3be49`

The v1 terminal failure and v2 measurement failure remain labeled FAIL. The v3
PASS does not rewrite them. The reduced comparison uses prefix
`/home/zhihao/s42-normal-reduced24-20260909-v1`, adaptive first, then a fresh
identical-source desktop control only after clean execution. Phone power remains
assumed; any energy report must show 3/4.5/6 W with the same idle treatment.
