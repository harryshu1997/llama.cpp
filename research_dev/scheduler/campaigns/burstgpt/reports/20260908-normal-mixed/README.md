# Normal-scheduling mixed replay

The cold three-request adaptive gate and a fresh matched desktop arm both passed
3/3, with semantic output and exact native terminal proofs. No fraction, route,
session assignment or future arrival was forced. All 215 scheduler source files
are byte-identical to the tested paired-verification deployment. Only replay
configuration, measurement artifacts and documentation were added this turn.

| Cold paid interval | Desktop | Adaptive |
| --- | ---: | ---: |
| Duration | 364.60 s | 316.76 s |
| CPU energy, measured | 18.757 kJ | 10.048 kJ |
| GPU energy, measured | 9.969 kJ | 8.947 kJ |
| Phone energy, assumed 4.5 W active | 0.319 kJ | 0.604 kJ |
| Fleet energy | 29.046 kJ | 19.599 kJ |

Fleet saving is 32.99%, 32.52%, and 32.06% at assumed phone active power of
3, 4.5, and 6 W. Both arms use 0.875 W phone idle power. The complete paid
boundary includes online preparation and normal shutdown; no preparation was
subtracted. Duration improved 13.12%. This is one bounded matched comparison,
not an execution-only steady-state result or an isolated old/new-fix ablation.

| Request | Desktop / adaptive inference | Final FFN column split | Tokens with phone work | Fraction-weighted coverage | Calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 36, 292 outputs | 155.26 / 133.11 s | 100% | 94.50% | 88.66% | 6,016 |
| Llama 37, 292 outputs | 1.96 / 1.81 s | 0% | 0% | Not eligible | 0 |
| Qwen 50, 71 outputs | 55.97 / 55.35 s | 0% after exploration | 54.29% | 42.86% | 228 |

Coverage uses 291 eligible Gemma decode tokens and 70 Qwen tokens. Native
request/control-scoped per-layer call counts and column widths reproduce the
acknowledged-window coverage exactly: Gemma 275 positive / 258 weighted tokens;
Qwen 38 / 30. The split is within the active CPU FFN layer mask, not the fraction
of the entire model. Gemma's mask shrank from 24 layers to the retained 16;
Qwen used six CPU-resident layers. Llama used its desktop route.

## Attachment and independent replacement

Times are seconds after the adaptive paid boundary unless labelled phone-native.

- Initial proposal 1.010; preparation 1.965, a 0.955 s delay.
- Gemma session publications: HTP0 20.438, HTP1 33.205, HTP2 45.836.
- Gemma desktop execution starts at 43.307 while HTP2 still loads. First decode
  is at 60.471; 100% physical control is applied at 65.235 (token 11).
- Normal scheduling proposes HTP2 replacement at 143.683. Drain is requested
  at 143.686, the retained-mask acknowledgement is recorded at 144.763,
  preparation starts at 144.944, and Qwen READY publishes at 157.724.
- HTP0/HTP1 remain generation 1. HTP2 alone changes from Gemma generation 1
  to Qwen generation 2. Retained calls during loading: 247 / 248; before and
  after counts are nonzero. No reset, fallback, stale proof or request restart.
- Physical native load-to-READY intervals: 9.881, 11.426, 10.972 s for initial
  HTP0/1/2; 12.509 s for replacement. Initial startup/transport and scheduler
  publication overhead are additional and remain inside the paid interval.
- Both retained sessions pass the existing `s42-retained-session-call-gap-v2`
  metric, 15 equivalent call classes each, maximum ratios 1.578x and 1.440x.
  The 2x threshold and 30-reference-interval requirement were unchanged.
- Four successful session preparations, no failures or abandoned proposals;
  physical load counts HTP0=1, HTP1=1, HTP2=2. The fourth load is the requested
  replacement, not a fraction-change reload. No reverse was forced.

Arrival-to-execution delays, including queueing and desktop preparation, were
42.307 / 120.983 / 165.572 s for Gemma / Llama / Qwen. They are not pure queue
time or phone waits. Qwen was phone-ready well before its desktop execution at
256.572 s. Its first shard nevertheless appeared 66.724 s after arrival; that
does not meet the earlier one-load-time-from-arrival target.

## Unassisted intervals and remaining issues

`MIXED_AUDIT_WITH_GAPS.json` under the remote inputs records every zero-fraction
decode interval, with record/event hashes. These are derived annotations over
exported windows and helper events, not a new direct controller reason field.
Initial baseline/control intervals are verification; brief 75% warmup yields
are shared-helper admission deferrals. The bid events say PREDICTED with no
winning competitor, not measured-negative energy. This warmup admission behavior
still deserves improvement. No inference interval waited for shard preparation.

Qwen's paired samples completed, but current conservative bounds narrowly missed
the effective 5% margin: candidate upper 62.814719 J/token exceeds the allowable
62.351190. Its means were favorable (57.104290 versus 72.925369), so this is not
evidence of negative mean saving. At selection, 21 tokens remained, below the
24-token further-opportunity guard. The tail stayed at zero. See
[CURRENT_PAIR_DIAGNOSIS.json](CURRENT_PAIR_DIAGNOSIS.json) for the read-only
current-window calculation and its limits. This is not the earlier cached-pair
1.98-second reservation failure. More evidence could help, but was not collected
by changing the workload or forcing assistance.

Configuration issue found, not changed: the campaign requests 1% minimum saving,
but catalog merging with the overlay raises the effective catalog margin to 5%.
Both arms use the same effective catalog. Lowering it after observing this outcome
would not be a valid unchanged-policy experiment.

## Evidence and reproducibility

Prefix: `/home/zhihao/s42-normal-mixed3-20260908-v1`.

- `-gate/run`: adaptive results, windows, helper events, commands, native logs,
  snapshots, session phases and generation-keyed terminal proofs.
- `-desktop/run`: fresh matched desktop evidence.
- `-inputs`: immutable configuration, preflight, source/command manifests,
  exact comparison, audits, unassisted annotations and artifact inventory.
- `-deploy`: frozen canonical scheduler used in both arms.

Strict comparison: [COMPARISON.json](COMPARISON.json); phone-power sensitivity:
[MATCHED_SUMMARY.json](MATCHED_SUMMARY.json). Same revision, dirty source manifest,
graph-disabled binaries, model/FFN artifacts, GPU22 Gemma and GPU16 Qwen parents,
prompts, output lengths, arrivals, catalog and accounting boundaries. GNOME and
unrelated processes were untouched. No native rebuild, commit, push or PR.

SHA-256:

- Adaptive RESULT: `2bdbce3c04fdf706c6baa29c5b7e81b9a9bb5f200c6d31a5f910a6e495dae568`
- Desktop RESULT: `bccd14332da3213f327f2d4b39c9964c002ac635653d4046ce926daeefba06e5`
- Source manifest file: `36682f3ea0c19535100746446b0a38aca1f147cc21618e2bf3956fd48577ad0e`
- Comparison file: `f4245d0fa5f9f9dd3fd9698a777b41d750211ed784554ed4fb530b7309807f20`
- Full mixed audit: `b8d7bf0c29345b356e69bb33215b9643adee49e2409198306f294dc1ecd8e1f4`
- Run artifact inventory: `512fbe2fb9d7ac55e470239a652242991a23bbfa0d55296a284fb7820b376bc7`

Three development-trace tests passed this turn. No production source changed;
the prior 204 focused tests and unchanged v3/v8 replay goldens remain applicable.
Historical results and dirty-worktree changes are preserved.

After the matched gate passed, the unchanged normal scheduler proceeded to the
24-request sparse-locality replay, adaptive first. Its separate evidence uses
`/home/zhihao/s42-normal-reduced24-20260908-v1`; do not extrapolate the 32.52%
bounded result to that trace.
