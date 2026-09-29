# Late helper operational-energy binding

The bounded mixed gate passes 2/2. Qwen's fraction-weighted eligible coverage
increases from 9.56% to 86.91%; Gemma remains at 92.96%. A fresh same-revision
desktop arm also passes 2/2. Matched fleet energy decreases 23.81% at the
assumed 4.5 W phone power, and the paid duration decreases 8.08%. This is one
bounded two-request comparison, not a longer-trace or statistical result.

## Cause and bounded fix

An adaptive request may start with neither a helper nor a usable dormant
opportunity. Its config then correctly disallows assumed phone power because
there is no exact phone device to look up. Later READY attachment replaced the
helper candidates and identities, but did not bind the operational energy
permission from the actual helper's phone power profile. Qwen could therefore
probe, then ignore its diagnostic energy windows and return to baseline.

Both decode-boundary attachment and READY materialization now bind that
permission after helper identity validation. Caller-supplied configs retain
their explicit permission. Missing and denied profiles stay conservative.
No fraction ordering, energy threshold, latency limit, placement rule, lease,
generation, transport, native binary or qualification rule was changed.
Assumed phone energy can inform operational selection only when the profile
explicitly permits it; it remains ineligible for qualified energy evidence.
The request event `HELPER_ENERGY_POLICY_BOUND` records a changed permission.

The previous Qwen observation group is
`sha256:fd840c0adf379d74988ef05d78ab406c4c91d02fc57e5f815a494277a182f0a4`.
Its valid 100% windows were 43.929 and 46.994 J/token, compared with baseline
windows around 75 J/token, but bids remained PREDICTED. These are diagnostic
windows, not a matched end-to-end savings measurement.

## Software evidence

The reproducer fails before the change with two substantive assertions:
late attachment leaves the permission false, and measured positive windows
cannot sustain the phone policy. It passes after the change. Additional tests
cover publication before the next boundary, absent/denied profiles, explicit
caller overrides, rejection of an unavailable helper, diagnostic-only evidence,
and idempotent READY polling. Two existing COW mock fixtures now include the
real helper/ticket contract fields consumed by the new lookup.

138 focused tests pass: late-helper energy policy, adaptive decode, adaptive
runtime, session COW and both replay goldens. No complete harness was run.

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

Both hashes are unchanged; no golden was regenerated.

Production files changed:

- `_internal/adaptive_decode.py`
- `_unified/adaptive_decode_control.py`
- `_unified/helper_envelopes.py`

Tests: new `tests/test_late_helper_energy_policy.py` and fixture additions in
`tests/test_session_cow_transaction.py`. Documentation: this report and the
completion entry in `research_dev/talks.md`.

## Physical scope

Fresh prefix: `/home/zhihao/s42-late-helper-energy-20260908-v1`.
Inputs, commands, preflight, source-before copies and test logs are in
`-inputs`; frozen scheduler source is in `-deploy`. The bounded mixed run uses
only Qwen 43 and Gemma 49, arriving at 1 and 91 seconds. The existing qualified
GPU16/GPU22 parents, graph-disabled binaries and real F16 FFN shard indexes are
unchanged. Prior V1/V2 and matched-24 artifacts remain untouched.

A new desktop-only arm uses these same two requests, catalog, source manifest,
binaries, placements and accounting boundary. Only the canonical selection
mode changes to desktop-baseline. It does not reuse the old 24-request baseline.

## Mixed physical result

Result: `/home/zhihao/s42-late-helper-energy-20260908-v1-gate/run/RESULT.json`.
SHA256: `40b682a6d51dd62953998b119d5fe7b5c26757ec987b932c7e09756a755ca7c1`.
Preflight file SHA256:
`84960796d51e46fe7cc84d70a76607174ec24974279d174e511e8efb62ea83cf`.
Source manifest file SHA256:
`8e39a22225e66a69ac5030ad76f57878740718e16e21a9e95a7b5fcfd8ca1857`.

| Request | Output tokens | Phone calls | Weighted eligible coverage | Positive-policy token coverage |
| --- | --- | --- | --- | --- |
| Qwen 43 | 341 | 3,732 | 86.91% | 91.47% |
| Gemma 49 | 491 | 3,768 | 92.96% | 96.12% |

### Separate FFN split and assisted-token ratios

Report these as two primary quantities, rather than calling their product
"phone coverage":

1. FFN split: the applied phone fraction of the selected resident FFN slice
   for an assisted token. Report the final selected fraction and the mean
   over assisted tokens; exclude 0% tokens from that conditional mean.
2. Assisted-token coverage: tokens receiving physical phone FFN work divided
   by all generated output tokens. Count each token once, regardless of the
   number of layers, sessions, or RPCs serving it.

| Request | FFN split: final / mean when assisted | Assisted output tokens / all output tokens |
| --- | --- | --- |
| Qwen 43 | 100% / 95.02% | 311 / 341 = 91.20% |
| Gemma 49 | 100% / 96.71% | 471 / 491 = 95.93% |

For applied fraction `s_i`, assisted-token set `A`, and total output count `N`:

```text
mean_ffn_split_when_assisted = sum(s_i for i in A) / len(A)
assisted_token_coverage = len(A) / N
```

The exact audit supplies 295.5 and 455.5 fraction-weighted token-equivalents.
Thus the conditional split means are `295.5 / 311` and `455.5 / 471`.
Both requests probe 25/50/75/100% and finish at 100%; the probes make the
conditional mean smaller than the final fraction.

Native request proofs corroborate the assisted-token counts in this M=1 run:
Qwen has 311 calls at each of 12 selected layers (3,732 total); Gemma has
471 calls at each of eight selected layers (3,768 total). The Qwen layers
are 0-5 and 12-17 on HTP0/HTP2; Gemma layers are 8-15 on HTP1. A general
batched run must count distinct request/token identities, not divide total
RPCs by a fixed layer count.

The 100% split is scoped to those selected resident FFN slices, not all
model layers or all model computation. Record the layer mask and stored
column range alongside this ratio; different helper sizes are not comparable
from the split percentage alone.

The historical table above remains an eligible-token audit: it excludes
the first output token and therefore uses denominators 340 and 490. The new
assisted-token ratio deliberately uses all 341/491 outputs, including the
initial token and time spent waiting for assistance. Neither ratio is an
energy-saving percentage. No frozen result or measurement has been changed.

Both complete with accepted semantic-sanity output and exact execution-ticket,
terminal-ticket, artifact, layout, operator-plan and grouped-observation proof
matches. Total request calls, generation-keyed session proofs and native
terminal counts agree at 7,500. There are no request recoveries or helper
rejection reasons. Both final fractions are 100%, selected by existing policy.

The Qwen energy permission binds once at 108.533716 s, after physical READY
verification, with `qualification_state=DIAGNOSTIC`. Its first
CONSERVATIVE_MEASURED bid occurs at 119.772462 s with a conservative estimated
gain of 21.083428 J/token. Qwen has 67 such accepted bids and eight prediction
bids. Gemma has 106 measurement-backed bids and eight prediction bids. The
audit parses exact current-request window hashes and verifies that assumed
phone energy never becomes qualified energy evidence.

Qwen execution spans 80.220858-284.537651 s; Gemma spans
334.941628-605.312641 s. The campaign duration is 628.690589 s. Gemma's queue
delay is GPU serialization and desktop model preparation, not waiting for its
already READY phone shard. Initial phone preparation overlaps desktop loading.
Qwen's first token is at 93.264707 s, before replacement starts at 94.165989 s;
desktop decode continues during loading. Its first positive fraction is applied
at 110.216516 s, and sustained 100% begins at 138.796714 s. Gemma's first token
is at 369.269064 s and first positive fraction at 374.508371 s. These are host
times relative to the measured campaign boundary.

| Physical load | Generation | Native load-to-READY | Host READY publication |
| --- | --- | --- | --- |
| Qwen HTP0 | 1 | 11.088970 s | 20.913968 s |
| Qwen HTP1 | 1 | 19.486934 s | 42.078577 s |
| Qwen HTP2 | 1 | 12.302118 s | 57.827979 s |
| Selected HTP1 becomes Gemma | 2 | 13.491633 s | 108.570008 s |

There are four loads, all from actual FFN shards: three initial loads and one
replacement. Fraction changes add no reloads. HTP0/HTP2 keep Qwen generation 1
and each makes 1,866 calls; selected HTP1 generation 2 makes 3,768 Gemma calls.
The maximum resident layout is 9,625,927,680 bytes plus 2,676,736 workspace
bytes, below the declared 10 GB capacity. No Llama service is resident.

Limit: replacement finishes before the first Qwen phone call. The retained
sessions make zero calls before/during this load, so the unchanged equivalent-
class interruption test is INSUFFICIENT, not PASS. This run does not newly
prove the serving-through-load gap bound, reverse replacement or rollback.
It validates the energy-binding fix and sustained assistance after readiness.

## Matched energy and latency

The canonical fail-closed comparison accepts the source manifest, catalog,
requests, prompts, token counts, model artifacts, runtime mode, host binaries,
live-VRAM parent qualification and measurement boundary. An additional audit
requires both physical preflight receipts to have identical shard indexes,
remote shard hashes, worker/router hashes, session/restore scripts, kernel and
USB-close binary. No route is forced in the runner.

Both large-model parents are the existing physically qualified placements:

- Qwen GPU16: `sha256:42a30600f56e90e50aca7b72df6e311eeaff1f1e071919a7b07dc5cff2b08477`.
- Gemma GPU22: `sha256:5bfd230a8572b808d2d1c45fedb8e36f21a70fe160bb88300c6af5d8edd1cf0e`.

Both arms use source HEAD `99449bafade0b2c15de4410feda832035c2f2d83` plus the
same exact dirty-tree source manifest. No commit was made. Both graph modes
are disabled; the same parent is used for desktop control and assistance.

| Paid-interval measurement | Desktop | Adaptive |
| --- | --- | --- |
| CPU package energy | 42.821 kJ | 29.518 kJ |
| GPU board energy | 20.405 kJ | 18.082 kJ |
| Assumed phone energy, nominal profile | 0.598 kJ | 1.024 kJ |
| Fleet energy | 63.824 kJ | 48.625 kJ |
| Duration | 683.953 s | 628.691 s |

CPU package energy is measured with RAPL; GPU board power is sampled with
NVML. Phone energy is assumed, not physically measured. The same 0.875 W
idle-power treatment is used in both arms. The boundary is canonical
`begin_trace` through `end_trace`: online model/shard preparation, execution,
and normal executor/phone shutdown are included. Deployment, preflight and
initial rig startup are outside that boundary. No preparation cost is
subtracted to improve the result.

| Assumed active phone power | Adaptive fleet energy | Saving versus matched desktop |
| --- | --- | --- |
| 3 W | 48.429 kJ | 24.12% |
| 4.5 W | 48.625 kJ | 23.81% |
| 6 W | 48.822 kJ | 23.51% |

Qwen execution latency decreases from 224.443 to 204.317 s (ratio 0.9103).
Gemma decreases from 275.701 to 270.371 s (ratio 0.9807). Both arms produce
341/491 tokens, pass semantic-sanity validation and contain no request
recoveries, fallback or USB-reset evidence. The 25% energy target is not met.
The small pair does not establish pre-resident execution-only energy or a
reuse break-even count; those are not inferred from these measurements.

## Recovery and remaining limitations

5,404 saved planning observations contain 5,402 VALID and two UNAVAILABLE
samples. During the initial FunctionFS changeover, HTTP timed out and ADB was
unavailable. The existing background HTTP path recovered after a sampled
1.157355 s without USB reset or worker restart. No outage remains at the end.
Maximum valid sample age is 3.418746 s against the unchanged 5 s limit. All
physical transition admissions use valid, unexpired observations.

The generic comparison's fraction histogram includes historical observation
groups, and its flat transport counter is not the nested native terminal
counter. Neither is used for the coverage/call numbers above. EXACT_AUDIT and
ENERGY_BINDING_AUDIT select each request's exact current grouped-observation
hash and cross-check generation-keyed native terminal proofs. The generic
execution identity also omits phone-side binary fields; MATCHED_SUMMARY
supplements it with explicit physical-preflight identity equality. These
reporting/checker gaps are recorded, not changed during this scoped fix.

No long trace, additional qualification, broad test sweep, native rebuild,
GDM interference, commit or push was performed.

## Final artifacts

All commands and audit scripts are preserved under the fresh `-inputs`
directory. All earlier failed and successful physical artifacts are intact.

- Adaptive result: `-gate/run/RESULT.json`, SHA256
  `40b682a6d51dd62953998b119d5fe7b5c26757ec987b932c7e09756a755ca7c1`.
- Desktop result: `-desktop/run/RESULT.json`, SHA256
  `2abbd1be0639940c9ce3ab784e8d4f473ba094d43bba0e0ddc469c49cda6c641`.
- `COMPARISON.json` file SHA256:
  `6e08768c9cd52bc312eaf1f2d500bec34cd037d16aa2dae6e6cf2d1e5e115547`.
- `MATCHED_SUMMARY.json` SHA256:
  `b4b9849d3f71ceadf3de0ab23af96f7f9b5b3b2c1fa90119bee2d16bdbe5ef8c`.
- `EXACT_AUDIT.json` SHA256:
  `c02a63af967dd7e86bcd6e9a6d5dbcd7a2e6a558f600d639cd8648dd6e3834a6`.
- `ENERGY_BINDING_AUDIT.json` SHA256:
  `1348131555fec46c4c79219195ccf33ec41ed1f4f194d79add987085a5d091f1`.
- `TELEMETRY_AUDIT.json` SHA256:
  `df9693b78c365cc3caca1260015964f20bfcffc040bc99fb0618ca90c6427fc4`.
- `RUNTIME_AUDIT.json` SHA256:
  `45e400b8b74f80ab97954eae944c19e7bd926a4c88709b95df0972df67e9a6bc`.
- `RUN_ARTIFACT_HASHES.json` SHA256:
  `ef1c9e40174b0f544ff0acf13a074c8b760a925668924f17e2811e5e219c7b7d`.

The final inventory covers 10,854 files and 744,340,366 bytes across both
finalized runs. The deployment differs from the preceding gate in only the
three production files and two test files listed above, plus a pre-existing
user test (`test_development_trace.py`) which was preserved unchanged.
