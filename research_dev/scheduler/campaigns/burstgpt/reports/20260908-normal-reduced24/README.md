# Normal-scheduling reduced 24-request replay

Adaptive and fresh matched desktop both completed 24/24 with semantic output
and exact terminal proofs. The canonical comparison is valid, but the target
is not met: nominal fleet saving is 3.89%, and adaptive is 5.07% slower.

| Complete cold paid interval | Desktop | Adaptive |
| --- | ---: | ---: |
| Duration | 1,616.43 s | 1,698.31 s |
| CPU energy, measured | 98.578 kJ | 90.399 kJ |
| GPU energy, measured | 50.111 kJ | 51.861 kJ |
| Phone energy, assumed 4.5 W active | 1.414 kJ | 2.010 kJ |
| Fleet energy | 150.103 kJ | 144.269 kJ |

Fleet savings at assumed phone active powers of 3 / 4.5 / 6 W are
4.0306% / 3.8862% / 3.7419%, with the same 0.875 W idle treatment. The adaptive
fleet energies are 144.052703 / 144.269407 / 144.486112 kJ. CPU energy falls
8.178875 kJ, offset by 1.750191 kJ more GPU energy and 0.595341 kJ more nominal
phone energy. This is one matched cold pair, not an isolated fix ablation,
pre-resident steady-state result or statistically established saving.

This follows the passing three-request normal mixed gate, not a forced-fraction
or prescribed-session experiment. The exact saved arrivals span 1 through
1161 seconds; original prompts and output lengths are unchanged. The cold paid
boundary includes online preparation and normal shutdown. The earlier five/six
minute sizing applies to the short mixed gate, not this 19.3-minute arrival span.

[COMPARISON.json](COMPARISON.json) records the strict checks and failed target;
[MATCHED_SUMMARY.json](MATCHED_SUMMARY.json) gives every request's matched
latency and phone-power sensitivity. The preceding short mixed gate's 32.52%
saving does not generalize to this reduced trace.

## Queueing and latency

Arrival-to-execution delay, including desktop preparation, has median / maximum
282.16 / 501.93 s adaptive versus 234.40 / 432.89 s desktop. Arrival-to-ACQUIRED
dispatch delay has median / maximum 276.47 / 501.68 s versus 229.23 / 432.77 s.
These are not phone-wait measurements. Both arms perform six desktop model
preparations; their receipt-duration sums are 333.316379 s adaptive and
254.802543 s desktop, a 78.513836 s difference. This is close to the 81.873322 s
paid-duration difference, but overlap and different live conditions prevent
treating those sums as a causal wall-time decomposition. No preparation is
subtracted from the comparison.

Per-request inference latency ratio has median 0.992, p95 1.316 and maximum
1.469, exceeding the configured 1.25 target for requests 35, 42, 54 and 55.
Request 55 makes no phone calls, so its slowdown is not direct phone-RPC time.
There is still desktop loading, concurrency and runtime variability to explain.
The 49 adaptive scheduling decisions total 19.227 s, median 0.320 s, p95
0.989 s and maximum 1.669 s; the desktop arm's 48 decisions total 20.865 s.

## Phone use and fraction coverage

| Model | Requests with calls / total | FFN column fractions physically used | Tokens with phone work | Fraction-weighted coverage | Calls |
| --- | ---: | --- | ---: | ---: | ---: |
| Gemma | 4 / 6 | 25%, 50%, 75%, 100% | 87.47% | 82.53% | 7,592 |
| Qwen | 10 / 15 | 75%, 100% | 19.39% | 18.53% | 2,142 |
| Llama | 0 / 3 | None | Not eligible | Not eligible | 0 |

Denominators are 1,085 eligible Gemma and 784 eligible Qwen decode tokens.
Native request/control-scoped layer call counts and column widths reproduce
window-plus-tail coverage exactly. The split is within the current CPU FFN mask,
not a percentage of the whole model. Gemma uses layers 8-15 on one session;
Qwen initially uses layers 0-17, then retained layers 0-5 and 12-17. All are
CPU-resident under their exact qualified GPU22 Gemma / GPU16 Qwen parents.
Llama legitimately remains desktop-only.

Gemma 44 now attaches before execution, first applies 100% about one second
after its first token, and makes 1,304 calls. Its repeated control changes still
reduce weighted coverage to 53.79%; the model aggregate must not hide this.
Other long Gemma requests 36 and 49 reach 90.21% and 94.18% weighted coverage.
Qwen 43 receives only 168 calls and 3.75% weighted coverage despite residency.

## Independent residency and overlap

| Publication | Changed session | Generation | Proposal / preparation / READY, s | Native load-to-READY |
| --- | --- | --- | --- | ---: |
| Initial Qwen shard | HTP0 | 1 | 1.007 / 1.630 / 21.786 | 11.369 s |
| Second Qwen shard | HTP1 | 1 | 21.786 / 22.084 / 37.250 | 14.865 s |
| Third Qwen shard | HTP2 | 1 | 37.250 / 37.343 / 55.174 | 17.459 s |
| Qwen to Gemma | HTP1 | 2 | 243.121 / 243.121 / 255.028 | 11.330 s |

Every demanded proposal starts preparation within one second. Desktop loading
runs at 2.464-66.489 s while phone preparation runs at 1.630-55.174 s. The first
session publishes independently; no all-three readiness barrier is added.
Physical phases separately retain weight-read, HTP-init, weight-upload,
verification and READY timestamps. Weight reads dominate the native load times.
The table uses scheduler `observed_at_us`; separate `published_at_us` and
phone-monotonic timestamps remain in the full audit, without mixing clocks.

Only dynamically selected HTP1 changes during replacement. HTP0 and HTP2 remain
Qwen generation 1; their identical original proof identities remain usable.
Physical session load counts are 1 / 2 / 1, including the requested replacement.
There are no fraction-change reloads, failed transitions, fallbacks, resets,
stale executions, request restarts or abandoned proposals. Final resident shard
bytes are 9,248,440,320, plus separately ledgered workspace. Llama is not resident
on the phone. Initial three-Qwen shard bytes are 9,625,927,680.

This trace does not exercise retained calls *during* the replacement: all first
Qwen-burst executions have finished when Gemma arrives. Each retained session
has 318 calls before and 594 after; zero during reflects no active request,
not a measured interruption. The preceding short mixed gate supplies the actual
overlapping proof: 247 / 248 retained calls during replacement, unchanged 2x
equivalent-class gap checks passing at 1.578x / 1.440x. No reverse or fault was
forced into this normal trace.

The analyzer reports zero failed and zero never-prepared layouts, 20.8 s with
no phone layout, 2.5 s executing without model residency (Llama), and first
Qwen/Gemma residency 20.8 / 14.0 s after first arrival. Queued work without
residency remains 423.8 s. These are distinct from useful assisted coverage.

The same analyzer on historical `s42-mixed-matched24-20260907-v6-adaptive`
reported 115.6 s to first Gemma residency, 42.9 s executing without model
residency and 12 requests with calls, versus 14.0 s / 2.5 s / 14 now. That is
a diagnostic before/after, not a same-revision energy comparison. Its default
32-token short-request grouping hides the 24/26-token eligibility gap; the
native audit above uses actual ELIGIBLE events instead.

## Remaining limitations, not changed during measurement

- A fresh 75% Qwen probe can lose admission after warmup, before its first valid
  measurement. Its bounded-learning bid requires an outstanding control, but
  that control has already been acknowledged. A negative prediction can then
  stop exploration even without a competing helper. Qwen 43 records 77 denied
  windows with no competing winner; the first uses predicted evidence, the
  following 76 use conservative measured bounds from 100%, not a measured 75%
  rejection. Its last 308 tokens stay at zero.
- The request-scoped HTTP path samples active batch at first token. Qwen 43's
  88 records retain batch 2 even after request 42 completes. Thus its
  workload-change reset does not fire, despite a much faster later desktop
  cadence. This needs boundary-time batch refresh and correctly scoped evidence,
  not relaxed energy or latency checks.
- Qwen 41 and 46 attach with 24 / 26 outputs but never probe. After first token,
  their expired-deadline 15% exploration budgets allow only 3.45 / 3.75
  token-equivalents, below a four-token window. Attachment eligibility and
  complete-measurement affordability disagree. They are not energy-negative
  measurements. Five other zero-call large-model requests export explicit
  insufficient-opportunity rejections; three Llama requests have no phone route.
- Catalog overlay merging retains an effective 5% saving margin, despite the
  campaign asking for 1%. Both arms preserve this same effective policy.
- Every zero-policy decode interval is indexed in the native audit with record
  or event evidence. Some reason categories are derived from those records;
  exact controller stop reasons are not directly exported for every interval.
  The report does not treat all zero assistance as measured rejection.

The earlier complete-pair fix is for cached diagnostic winner verification.
These fresh mixed-context learning observations are a separate admission and
context-refresh gap; this is not a recurrence of the 1.98 s cached allowance.
Gemma 36/49 complete valid baseline and all four fraction samples and exploit
100%. No qualification, fraction, assignment or workload was forced to improve
the outcome.

The decoded records, exact bid payloads, token budgets and code references are
in [COVERAGE_DIAGNOSIS.json](COVERAGE_DIAGNOSIS.json). No follow-on optimization
was introduced between the adaptive and baseline arms.

Telemetry remains fail-closed: one HTTP timeout recovered within 1.077 s.
An additional request's recovery event appears at 15.506 s, but that is its
planning revisit, not a continuous physical outage. At 729.041 s a stale HTTP
sample and unavailable ADB fallback defer planning; the next saved snapshot is
valid at 729.460 s, within 0.419 s. The latter lacks a request-level recovery
event. READY identities are retained and no USB cleanup or worker restart is
used to recover telemetry.

## Reproducibility

Prefix: `/home/zhihao/s42-normal-reduced24-20260908-v1`.

- `-gate/run`: adaptive RESULT, native logs, windows, helper events, snapshots,
  per-session phases and generation-keyed terminal proofs.
- `-desktop/run`: fresh matched desktop evidence.
- `-inputs`: configurations, commands, initial observations, catalog, preflight,
  source manifest, comparison and decoded audits.
- `-deploy`: frozen canonical scheduler used by both arms.

All 215 scheduler source files still match the tested paired-verification
deployment. Same graph-disabled binaries, model and FFN shard indexes, qualified
desktop parents, catalog, prompts, output lengths, arrivals and energy boundary.
New mixed observations were not silently seeded into the initial observation
store. CPU/GPU energy is measured physically; phone power is assumed at
3 / 4.5 / 6 W active, with identical 0.875 W idle treatment in both arms.

Three focused development-trace tests passed this turn. The prior 204 focused
tests and both replay goldens apply to the identical production sources; no
broad harness loop was run. Goldens remain:

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

No production code, native binary, user dirty change or old physical artifact was
edited. No GDM stop, process kill, USB reset, commit, push or PR. Changed repo
files in this measurement turn are the two new normal-run report directories
and `research_dev/talks.md`; campaign inputs and read-only audit scripts are
preserved with their fresh physical artifacts.

SHA-256:

- Adaptive RESULT: `5a7c1a74a78a6093713f16f728e70b39385f5d1b506bffd3205949fadcf97d5e`
- Desktop RESULT: `e641a39846cb8fae78c6270114c700477a72d76286fb506f948e6ace06be044d`
- Source manifest file: `f2b2b30e3e3facaa65afb02e8c7aea2268015557182f13469f9964886d1939be`
- Catalog file: `667499be3aa638ff93f818f55618341ce161f4b6459af45a42f35449d661f75d`
- Replay input file: `78b7582ebfe063cdbb14added28b060ee955362d907cdfc1b5b7c6aa14192ce9`
- Preflight file: `6c798b0f85775dea0634d0f1a48fcabfd050f62949eac3d6b04aa2a933010a07`
- Coverage diagnosis: `cbe393e60689655cd82a079a2ca7814c15524bc8af85e0024e0685193c126764`
- Comparison file: `eebfa9bd7623e100d4610427a5202461746b45c9fc0c66926a80aa1af17d3967`
- Matched summary: `fd8fec92bccb198f7a1a96c1e8c617e4a3ad63d1bc4cef298f147f5546bfddaf`
- Full native audit: `4d758362df680c95b8bad7739a9dc6d062094ff3fbd4768e435afb293fe47456`
- Run artifact inventory: `7c03412217a9df30f7cb21e030364708cdad4a56633a9085939b2f9bf0176cca`

The inventory covers 35,648 files / 2,686,398,945 bytes from both complete runs,
including failed or deferred diagnostic records without deleting them. It and
the final health/source checks are in `-inputs`.
After both arms, ADB is available and GPU memory is back to 3,178 MiB used /
12,770 MiB free with GNOME untouched. No longer trace was launched.
