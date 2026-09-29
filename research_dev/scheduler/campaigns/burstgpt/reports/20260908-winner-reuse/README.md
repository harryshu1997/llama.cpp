# Bounded winner reuse and mixed-session validation - 2026-09-08

PASS: 172 focused tests, both unchanged replay goldens, live preflight, and one
five-request physical lifecycle gate. No long trace, baseline rerun, native
rebuild, qualification change, GDM interference, commit or push.

## Changes

Paths below are relative to `research_dev/scheduler/`:

- `_internal/adaptive_decode.py`: reuse a positive diagnostic candidate only
  with explicit assumed-phone operational permission and matching artifact,
  desktop parent, component capability, layout geometry, batch and context
  bucket. Try that fraction and verify a fresh desktop/phone pair, rather than
  repeating the full sweep. The new pair must pass the existing energy,
  latency and opportunity checks without historical energy masking a
  regression. A negative pair returns to desktop. Exact helper authorization,
  generations, leases, draining and rollback are unchanged. Diagnostic energy
  never becomes qualified evidence.
- `campaigns/burstgpt/compare_ab.py`: select only observation groups referenced
  by current completed request proofs, validate their hashes/tickets/artifacts/
  parents, reconcile calls and bytes with generation-keyed native terminals,
  and compare phone preflight worker/router/shard identities across arms.
- `tests/test_adaptive_decode.py`: five new regressions for positive reuse,
  fresh negative evidence, compatibility/permission rejection, late READY,
  stale generation rejection, and insufficient opportunity.
- `tests/test_matched_comparison.py`: four new reporting regressions covering
  historical pollution, current proof mismatch, nested terminal counters and
  phone identity validation.

Documentation changes: this report and `research_dev/talks.md`. No changes to
the physical lifecycle, gate policy, native binaries or shard formats. All four
source/test files match the frozen physical execution manifest byte for byte.

## Physical result

The existing calibration-mode bounded gate uses Qwen request 43 and Gemma
request 3 as payloads for five lifecycle checks, not a trace. Qwen uses its
qualified GPU16 parent; Gemma uses its qualified GPU22 parent. Both retain
graph-disabled mode and the deployed F16 FFN shard indexes. Requests have 341
Qwen output tokens and 837 Gemma output tokens.

| Request phase | Phone calls | Positive eligible-token coverage | Fraction-weighted eligible coverage |
| --- | ---: | ---: | ---: |
| Cold Qwen | 4,734 | Not recomputed | Not recomputed |
| Online Qwen / forward replacement | 3,774 | 90.00% | 75.00% |
| Gemma on the READY shard | 6,416 | 95.93% | 89.68% |
| Qwen during injected reverse failure | 3,648 | 89.41% | 89.41% |
| Qwen during reverse retry | 4,926 | 89.41% | 74.41% |

All five complete. Request and native terminal counters agree at **23,498**.
The eligible denominator starts at token 1: 340 Qwen or 836 Gemma tokens.
Fractions refer to the selected CPU-resident FFN slices, not the whole model.

The scheduler selected HTP2 dynamically. Its physical sequence was
Qwen/gen1 -> Gemma/gen2 -> Qwen/gen3 -> restored Gemma/gen4 -> Qwen/gen5.
HTP0 and HTP1 remained Qwen/gen1. Each retained session loaded once; HTP2
loaded five times, including the injected target, restoration and clean retry.
There were zero attachment/fraction-change reloads, unexpected fallbacks,
USB resets, stale-generation failures or global phone restarts.

| Transition | Native load-to-READY | Retained calls HTP0 / HTP1 | Worst matched-class gap / median |
| --- | ---: | ---: | ---: |
| Forward to Gemma/gen2 | 9.532 s | 108 / 108 | 1.256x |
| Reverse target plus physical rollback | 21.336 s | 240 / 242 | 1.196x |
| Clean reverse retry to Qwen/gen5 | 11.502 s | 132 / 132 | 1.214x |

Every one of the 66 equivalent-call classes passes the unchanged 2x bound,
using 30 preceding reference intervals at matching fraction and retained mask.
The failed reverse target itself took 11.497 s; restoration took 9.534 s.
The combined 21.336 s includes the interval between those operations.

Forward drain timing, seconds relative to the online desktop measurement epoch:

| Event | Time |
| --- | ---: |
| Drain requested | 63.454555 |
| Next safe boundary | 63.813391 |
| Reduced-mask control issued | 64.308726 |
| Applied acknowledgement and quiescence | 64.800379 |
| Host replacement loading started | 65.257968 |
| Physical READY acknowledgement | 74.840154 |
| Scheduler READY publication | 74.892638 |

Drain-to-quiescence is 1.345824 s. The shortened 15-token window retains
accounting but is explicitly measurement-ineligible. Loading does not wait for
the configured 32-token measurement window to finish.

## Reuse and preparation

Both reverse Qwen requests reuse a single 100% candidate. First positive
assistance is at token 3; a fresh 32-token phone window and 32-token desktop
window finish verification at token 69, with the applied exploitation window
starting at token 71. There is no repeated 100/75/50/25 sweep for unchanged
geometry. The clean retry later expands the Qwen shard set and legitimately
re-probes that changed geometry; it does not inherit qualification for it.
All these energy windows remain diagnostic.

Fresh first helper leases are `lease-33` through `lease-39` for reverse-fault
Qwen and `lease-53` through `lease-59` for reverse-retry Qwen. Their request
tickets and proof identities differ; retained physical generations do not.
First positive assistance is 13.394 s and 11.846 s after execution start,
respectively, including prefill. Desktop execution proceeds during this time.

Cold native load times for HTP0/1/2 are 13.045 / 17.280 / 18.218 s. The first
native session is usable after 13.045 s; all three are native READY 63.702 s
after the first native load authorization, including the serving-overlap check.
The scheduler preload ledger starts at 74.894269 s, publishes its first READY
at 97.899766 s and all READY at 140.781516 s: 23.005 s / 65.887 s from that
ledger start. Native phone clocks and host measurement clocks are reported
separately, not subtracted from one another.

Resident shard bytes are 9,625,927,680. The preparation ledger attributes
6.039 kJ fleet energy to its preparation windows at assumed phone power 4.5 W
(CPU 4.030 kJ, GPU 1.745 kJ, phone 0.263 kJ). These windows include concurrent
desktop work; this is not isolated incremental phone-loading energy. The
whole cold-validation interval uses 19.482 kJ and the online lifecycle interval
100.501 kJ. No steady-state saving or break-even count is inferred from this
unmatched lifecycle gate.

Sensitivity of those same recorded intervals, retaining 0.875 W phone idle
power and physically measured CPU/GPU energy:

| Fleet energy | Phone active 3 W | Phone active 4.5 W | Phone active 6 W |
| --- | ---: | ---: | ---: |
| Cold validation, including preload and Qwen execution | 19.343 kJ | 19.482 kJ | 19.620 kJ |
| Online lifecycle, including replacements and requests | 100.248 kJ | 100.501 kJ | 100.754 kJ |

These are power-model sensitivities, not measured phone joules or matched-arm
savings. The implied phone active times are 92.563 s and 168.907 s.

## Reporting correction and savings boundary

Re-analyzing the previous exact matched pair leaves its **23.8139%** fleet
saving unchanged at 4.5 W assumed phone active power. It is 24.1214% at 3 W
and 23.5064% at 6 W, with the same 0.875 W idle treatment. This is the previous
two-request energy-aware experiment, not a new saving caused by winner reuse.

| Previous pair reporting field | Before | Corrected |
| --- | ---: | ---: |
| Recorded token positions in fraction histogram | 8,309, including history | 826, current requests only |
| Phone calls | 0 from wrong flat field | 7,500, matched to native terminals |
| Fleet saving at assumed 4.5 W | 23.8139% | 23.8139% |

The histogram excludes unmeasured tails and is not an eligible-coverage
denominator. Current fractions are 48 positions at 0%, 714 at 100%, 24 at
75%, 20 at 50%, and 20 at 25%. Phone worker/router/shard checks now live in
the comparator rather than relying on the separate manual audit.

## Tests and artifacts

Focused unittest modules: `test_matched_comparison`, `test_adaptive_decode`,
`test_late_helper_energy_policy`, `test_adaptive_runtime`,
`test_session_cow_transaction`, `test_offline_phone_residency`, and
`test_replay_determinism`: 172 PASS in 123.013 s. A final reporting-only
adjustment was rechecked with all 10 matched-comparison tests. No full harness
was rerun. Replay goldens are unchanged:

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

Desktop artifact prefix: `/home/zhihao/s42-winner-reuse-20260908-v1`.
`-deploy` is frozen source; `-gate/run` contains the complete physical result;
`-inputs` contains commands, manifests, preflight, tests, source-before copies
and audits. File SHA256s:

| Artifact | SHA256 |
| --- | --- |
| `-gate/run/RESULT.json` | `b5d1f17ce5d4feab3a73814ce4f6f33a96ec25bbe5f3869672be9ddc5bb8aa64` |
| `-gate/run/TERMINAL_PROOF.json` | `ef7400cf362be843b42feacac75e6f40bbd899f0d82f2dcdde2336ac83cdc2a8` |
| `-inputs/EXECUTION_SOURCE_MANIFEST.json` | `28c1266692bbf7795e521f5c96da422492596ffc51d599798e16043e9148fae0` |
| `-inputs/PREFLIGHT.json` | `0e4556d560c6ff165f317889971c12fbde442b8702fb7c4940788d561f2c4729` |
| `-inputs/BOUNDED_GATE_AUDIT_V2.json` | `f16903bd55993c269819ae5ac12a6f57b920ffe4dc6d54d6b7351215ee06f792` |
| `-inputs/RUN_ARTIFACT_HASHES.json` | `4e296f007aca06b5a86830a5dceae72a27ea056e60dcece3e6082777805ba9dd` |
| `-inputs/PREVIOUS_PAIR_COMPARISON_CORRECTED.json` | `cac4c81ab47d568ed07eb91198b4c165f971872fff109927ba0760fe818fb185` |

The run inventory covers 10,282 files / 455,687,255 bytes. The initial derived
`BOUNDED_GATE_AUDIT.json` is preserved but superseded: its load counter looked
for scheduler `LOADING` in native events, whose name is `LOAD_AUTHORIZED`.
V2 corrects that reporting-only field. Raw physical results are unchanged.

## Remaining work

- Incremental energy saving from this optimization still needs a fresh matched
  A/B; the older 23.81% result cannot answer that question.
- Desktop model loading/GPU queueing remains a substantial latency cost; this
  change does not alter placement or scheduling contention policy.
- The cached path deliberately rejects different geometry, parent, batch or
  context bucket. Changed layouts still need measurement.
- This validation uses batch 1. Overlapping shared-cohort RPC attribution needs
  its own comparison audit before a batched trace. Ambiguous native/request
  counter reconciliation fails closed rather than reporting a fabricated count.

No physical blocker remains for this bounded gate. Normal cleanup completed;
ADB reports the phone available and desktop VRAM returned to 3,178 MiB used,
12,770 MiB free, with GDM left running.
