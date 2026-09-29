# Newest winner-reuse fix: matched physical measurement

2026-09-08. Two Qwen requests, 43 then 67, run through the normal energy-aware
scheduler and a fresh desktop-baseline arm. Both arms complete 2/2. The strict
comparator accepts their identities. Fleet saving is 30.95% at assumed 4.5 W
phone active power, but the second request exposes an unfinished verification
budget issue. This is not a new mixed-model result or proof of improved
incremental savings over the previous adaptive version.

## Matched result

| Metric | Desktop | Newest adaptive |
| --- | ---: | ---: |
| CPU energy | 27.386 kJ | 15.202 kJ |
| GPU energy | 11.974 kJ | 11.430 kJ |
| Assumed phone energy, 4.5 W active | 0.388 kJ | 0.814 kJ |
| Fleet energy | 39.747 kJ | 27.446 kJ |
| Paid duration | 442.938 s | 462.473 s |

Saving: 12.301 kJ / 30.9477%. Paid duration increases 4.4105%, not a latency
win. After the last execution receipt, normal cleanup/accounting takes
18.729 s adaptive versus 0.760 s desktop. These costs remain included.

| Assumed phone active power | Adaptive fleet energy | Fleet saving |
| --- | ---: | ---: |
| 3 W | 27.277 kJ | 31.3742% |
| 4.5 W | 27.446 kJ | 30.9477% |
| 6 W | 27.616 kJ | 30.5212% |

CPU/RAPL and GPU/NVML energy are physically measured. Phone energy is assumed,
with identical 0.875 W idle treatment in both arms. The paid boundary includes
online preparation, the fixed arrival gap and normal shutdown. This is one
bounded pair, not a statistical estimate or a pre-resident steady-state run.
The older 23.81% result used a different Qwen/Gemma workload and is not a
control for the incremental winner-reuse change.

## Reuse and coverage

| Request | Desktop execution | Adaptive execution | Phone calls | Weighted eligible coverage |
| --- | ---: | ---: | ---: | ---: |
| Qwen 43, 341 output tokens | 222.881 s | 190.890 s | 5,886 | 91.62% |
| Qwen 67, 133 output tokens | 91.425 s | 93.158 s | 180 | 7.58% |

Coverage is checked against native per-layer calls and physical column widths,
not historical observation counts. Both requests use the 18-layer CPU-resident
FFN set, layers 0-17. Every layer has equal request-scoped counts. Request 43
has 327 positive positions / 340 eligible tokens and 311.5 fraction-weighted
tokens; request 67 has 10 / 132 and 10.0 respectively. Any-phone coverage is
96.18% / 7.58%; aggregate weighted coverage is 68.11%, below 70%.

One shard-set load: exactly one native LOAD_AUTHORIZED for each HTP session.
All sessions stay at generation 1. Each session serves 1,962 calls for request
43 and 60 for 67, totaling 6,066. Native terminals, server calls and exact
request proofs agree. There are zero execution recoveries/fallbacks, resets,
stale-generation errors, rejected helper candidates, request restarts or
between-request weight reloads. Fresh first helper lease sets are 11-17 for
43 and 31-37 for 67; the complete sets are disjoint. No replacement is
requested in this reuse-only measurement.

Host timestamps below are seconds from the paid epoch:

- Request 43 arrives at 1.000; layout PROPOSED at 1.009777 and preparation
  starts at 1.632517, a 0.622740 s delay.
- Sessions publish READY progressively at 21.444441, 38.308422 and 53.742869.
  Layout audit generations advance 1/2/3; physical session generations remain
  1/1/1. The two kinds of generation are not interchangeable.
- Request 43 attaches the first READY pair at 47.573369, starts desktop
  execution at 47.962820 while the third session is loading, gets its first
  token at 60.067665 and first positive control at token 11 / 66.346405 s.
  It finishes at 238.852457, before request 67 arrives.
- Request 67 arrives at 350.000, receives a fresh attachment at 350.328275,
  starts execution at 350.586691, gets its first token at 361.359991, and
  applies the cached 100% candidate at token 3 / 362.621916 s.
- Its control returns to 0% at token 13 / 367.866011 s. Execution continues
  on desktop and finishes at 443.744545 s.

Native per-session load-to-READY times are 11.001775 / 16.395122 / 12.431074 s.
First native authorization to all native READY takes 50.540947 s, including
gaps between sessions. The scheduler preparation interval is 52.110352 s
from initial start through last READY publication. Host and phone clocks are
not subtracted from each other.

The three phone preparation receipts attribute 2.015270 kJ fleet energy to
their windows: CPU 0.676066, GPU 1.120665, phone 0.218538 kJ. They overlap
desktop work and must not be added again to paid energy or treated as isolated
incremental preload cost. No break-even request count is claimed.

Execution-receipt energy is diagnostic: request 43 is 26.047003 / 13.057663 kJ
desktop/adaptive; request 67 is 10.441887 / 10.299710 kJ. The second request
has resident weights in both arms and saves only 1.36% in that recorded warm
execution interval. The first request still overlaps final phone preparation.

## Why request 67 loses assistance

The optimization does find the compatible diagnostic winner and starts it at
token 3, without repeating four fractions. It does not finish the fresh pair.
At token 11 (366.825772 s):

- 122 output tokens remain, above the configured minimum of 24.
- Only 10 probe tokens have been used, below the maximum of 80.
- The source request's deadline is 380 s. The positive deadline remainder is
  13.174228 s; its 15% exploration allowance is 1.976134 s.
- The historical measured desktop latency is 0.612141 s/token. A four-token
  window needs 2.448564 s, so `_can_probe()` returns false.
- `_next_after_window()` takes its generic budget-exhaustion branch before
  `cached_candidate` can schedule the paired baseline. With no valid current
  baseline, `_qualifies()` rejects exploitation; the state becomes EXPLOITING
  at 0%. Later baseline windows are recorded but do not resume verification.

This is not an observed negative phone-energy result. The valid phone window
at tokens 7-11 records 34.97 J/token and 0.525 s/token. The later desktop
window at 17-21 records 74.98 J/token and 0.626 s/token. These are short,
diagnostic samples, not qualification or an isolated causal saving estimate.

`diagnose_probe_budget.py` reconstructs the boundary using the frozen
controller methods, actual receipts and only preceding history. It reproduces
the 0% EXPLOITING decision. Two independent outputs are byte-identical. No
production fix, golden regeneration, SLO override or qualification weakening
was made during measurement.

Next correction to review: budget a complete bounded fresh pair before
starting cached verification, and distinguish incomplete verification from
measured rejection so eligible desktop execution can retry safely. Preserve
current-pair, energy, latency, memory, generation and lease checks.

## Exact identity and artifacts

Same head `99449bafade0b2c15de4410feda832035c2f2d83` plus the same frozen dirty
source manifest in both arms. All 215 source files remain byte-identical to
the deployment and current tree. The comparator also matches source, binaries,
phone worker/router/shard hashes, catalog, prompts, seeds, token counts,
qualified desktop parent, graph mode and energy boundaries.

Effective execution parameters in both arms: GPU16, context 4096, batch 2048,
ubatch 512, parallel 4; one active request at a time; CUDA graphs disabled.
This is the existing qualified route, not a newly weakened desktop parent.

- Parent placement: `sha256:42a30600f56e90e50aca7b72df6e311eeaff1f1e071919a7b07dc5cff2b08477`.
- Parent qualification: `sha256:af83748d0868362ef0321f205a12c092f2bc28420adb2b318032da4ced54b901`.
- Qwen artifact: `sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718`.
- READY geometry: `sha256:7bf55ae9bb49799de955f276b0ef0abc18ff69f2ca937afccc42e3f633604141`.
- Catalog: `sha256:667499be3aa638ff93f818f55618341ce161f4b6459af45a42f35449d661f75d`.

Native weight-source proof records `ffn_shard` and opens
`/data/local/tmp/s42-ffn-shards-20260904-v1/qwen/HTP0.ffn.gguf`,
`HTP1.ffn.gguf`, and `HTP2.ffn.gguf`. Bytes loaded are
3,208,644,448 / 3,208,644,448 / 3,208,644,480; tensor residency is
3,208,642,560 bytes per session. Stored and executed masks, columns, index,
shard and parent hashes and load/verification/READY timestamps are preserved
in `RESULT.json`'s `launch.weight_sources` within the direct phone receipts.

Desktop prefix: `/home/zhihao/s42-winner-reuse-matched-20260908-v1`.
`-deploy` is the frozen tree; `-inputs` holds configuration, commands, manifest,
preflight and audits. `-gate/run` and `-desktop/run` preserve both complete
physical results, snapshots, streams, journals and native terminal proofs.

| Artifact | File SHA256 |
| --- | --- |
| `-gate/run/RESULT.json` | `7ead8acbe5ae58fd726f86f29180b1b75fffa96986bd77ce6b711490d2dec6d2` |
| `-desktop/run/RESULT.json` | `efe0c0eacf046cb75248558f8d29b18122ec80ec50ad7b83b3bab1c94ec2a4a4` |
| `-inputs/SOURCE_MANIFEST.json` | `174b088e5b91bdf0a3d0ac51f7a835f666b632508e6e07f06f7ef11907167ae1` |
| `-inputs/PREFLIGHT.json` | `fd94007ce220a4205e23aff5f947d9f52f815ace3753c9022af945b607d0a913` |
| `-inputs/COMPARISON.json` | `c48ef2c0c17c6e686b960487686fb60f1b7ecd860bdfc04bf5b1d955efb7455f` |
| `-inputs/MATCHED_SUMMARY.json` | `507416f1b745ece285fea2a460a6432f750fd7b1c09703873c79797c5d7332fa` |
| `-inputs/REUSE_AUDIT.json` | `efd81b992539077978c3ed9cea336f741c6f1868af2300efedd124ee4535a143` |
| `-inputs/PROBE_BUDGET_DIAGNOSIS.json` | `fee7f1508d546de420fbf4457864e618fcbbd12d4a31c6d76d56ea43215bf470` |
| `-inputs/RUN_ARTIFACT_HASHES.json` | `6cb823b8860bee381611803beff7a9b501c80d8867a7d3a3f66e5bf5df5b1d31` |

The run inventory covers 2,274 files / 167,397,450 bytes. Local JSON files
beside this report are exact copies of the derived comparison and audits.

No software suite was rerun for this measurement-only turn. The unchanged
source previously passed 172 focused tests and both replay goldens:
v3 `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`,
v8 `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`.
New checks: live preflight, both physical arms, canonical comparator, native
reuse audit and repeated in-memory boundary diagnosis.

Only this report directory and `research_dev/talks.md` change in the working
tree. Measurement configuration and audit scripts stay with fresh external
artifacts. No production code, test, native binary, shard, calibration or
qualification changed. No long trace, commit or push. Normal cleanup restores
ADB availability and VRAM to 3,178 MiB used / 12,770 MiB free; GDM and other
existing processes are untouched.
