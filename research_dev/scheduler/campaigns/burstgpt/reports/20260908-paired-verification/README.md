# Complete-pair verification: fixed and physically retested

2026-09-08. The bounded Qwen 43/67 adaptive run and fresh matched desktop arm
both complete 2/2. The canonical comparator accepts the identities and reports
42.88% fleet-energy saving at assumed 4.5 W phone active power. Request 67 now
finishes verification and exploits the reused phone layout instead of staying
at 0%. This is Qwen-only, not a mixed-model or long-trace result.

## Measured result

| Metric | Fresh desktop | Fixed adaptive |
| --- | ---: | ---: |
| CPU energy | 27.345 kJ | 10.855 kJ |
| GPU energy | 11.866 kJ | 10.877 kJ |
| Assumed phone energy, 4.5 W active | 0.387 kJ | 0.886 kJ |
| Fleet energy | 39.598 kJ | 22.619 kJ |
| Paid duration | 442.553 s | 438.322 s |

Saving is 16.979443 kJ / 42.8795%; paid duration is 0.9561% shorter. CPU/RAPL
and GPU/NVML are physically measured. Phone power is assumed, with the same
0.875 W idle treatment in both arms. The paid boundary includes online
preparation, the fixed arrival gap, and normal executor/phone shutdown.

| Assumed active phone power | Adaptive fleet energy | Fleet saving |
| --- | ---: | ---: |
| 3 W | 22.411 kJ | 43.4049% |
| 4.5 W | 22.619 kJ | 42.8795% |
| 6 W | 22.827 kJ | 42.3540% |

| Request | Desktop execution | Adaptive execution | Phone calls | Weighted eligible coverage |
| --- | ---: | ---: | ---: | ---: |
| Qwen 43, 341 output tokens | 219.351 s | 188.838 s | 5,886 | 91.62% |
| Qwen 67, 133 output tokens | 91.002 s | 82.261 s | 2,160 | 90.91% |

Execution is 13.91% / 9.61% faster per request. Native per-layer counts and
column widths reconcile to 311.5 weighted tokens / 340 eligible tokens for
43, and 120 / 132 for 67. Any-phone coverage is 96.18% / 90.91%. Aggregate
weighted coverage is 91.42%. These fractions describe the eligible CPU-FFN
assistance set, not all model operators. The assisted layers are 0-17,
CPU-resident in the exact qualified parent.

Relative to the previous adaptive measurement, request 67 improves from
7.58% coverage / 180 calls / 93.158 s to 90.91% / 2,160 / 82.261 s.
Those are successive runs, not a randomized old/new ablation. The savings
above use the fresh same-source desktop control, not the old baseline.

## Cause and correction

The old `_can_probe()` allowance shrank between the candidate and baseline
halves. At the saved token-11 boundary, the next window required 2.448564 s
but the remaining exploration allowance was 1.976134 s. There were still
122 tokens left. The generic budget exit selected baseline before collecting
the valid current pair, and EXPLOITING at 0% never retried. This was incomplete
measurement, not measured negative phone economics.

The existing adaptive controller now reserves a complete bounded pair before
starting diagnostic winner verification. It includes remaining warmup windows,
both measurements, observed control/acknowledgement overhead, an uncertainty
margin, a token cap, and remaining exploitation opportunity. A reservation
does not shrink independently at each half. For a source SLO already
unattainable on desktop, budgeting uses the same relative latency ceiling as
the existing `_qualifies()` path; the source deadline is not rewritten.

Incomplete evidence or an exhausted reservation produces INCOMPLETE, desktop
continuation, and at most one feasible retry. A valid non-improving pair or
execution/control failure remains REJECTED. Retries use current paired
evidence, not historical energy to mask a regression. A session drain cancels
the pair without restoring the removed session's mask. Artifact, parent,
geometry, generation, leases, memory, output and qualification checks remain.
Diagnostic verification events are exported through existing helper events
in RESULT.json; assumed-phone evidence is never promoted to qualified energy.

The shutdown delay had a separate cause: `DirectPhoneFfnSession.finish()`
waited up to 15 s for normal USB before issuing the close that restores it.
It now makes one read-only observation first, then uses the existing exact
close and terminal/restoration checks. The final restoration allowance stays
90 s. No USB reset, healthy-worker interruption or proof bypass was added.
The adaptive post-last-receipt interval falls from 18.728762 s previously to
4.960444 s; the fresh desktop interval is 0.799006 s. This interval still
includes coordination and all shutdown work, not just the USB close operation.

## Physical timeline and reuse

Times below are seconds from the adaptive paid epoch. Receipt observation and
READY publication are distinct; phone-native durations use only the phone clock.

- Layout PROPOSED: 1.006956; initial preparation: 1.630310, a 0.623354 s delay.
- Request 43 attaches the READY pair at 57.014378 and begins desktop execution
  at 57.327141 while the third session is still LOADING. First token: 69.290120;
  first positive control is token 11 at 75.391537. Completion: 246.165519.
- Request 67 arrives at 350.000, attaches with fresh leases at 350.789231,
  begins execution at 351.101105, and produces its first token at 361.835183.
- Verification is RESERVED at token 1 / 362.470900. Required pair budget:
  14.029320 s and 22 tokens; allowance: 15.069656 s.
- 100% applies at token 3 / 363.093495. The paired baseline applies at token
  13 / 368.393590. Current phone tokens 7-11 measure 35.585 J/token and
  0.530323 s/token; current baseline tokens 17-21 measure 74.030 J/token and
  0.627326 s/token. Warmup/transition samples are excluded from qualification.
- VERIFICATION_VERIFIED: token 21 / 373.413808. 100% exploitation applies at
  token 23 / 374.662671 and continues to completion at 433.361621. No retry
  was needed physically. The pair plus return acknowledgement takes 12.191771 s.
- Paid end: 438.322065, after normal cleanup; this boundary was not moved.

| Session | LOADING starts | READY observed | READY published | Native load-to-READY |
| --- | ---: | ---: | ---: | ---: |
| HTP0 | 1.630310 s | 20.637103 s | 20.851754 s | 11.391240 s |
| HTP1 | 21.995450 s | 42.911339 s | 42.988461 s | 20.605761 s |
| HTP2 | 43.504022 s | 59.664767 s | 59.931625 s | 13.883924 s |

First/all publication: 20.851754 / 59.931625 s. Initial preparation through
last publication: 58.301315 s. Native first LOAD_AUTHORIZED through last READY:
56.616050 s, including between-session gaps. Cold loading remains a real cost.

Exactly one shard-set load: one LOAD_AUTHORIZED per HTP0/1/2, no reload between
requests, and unchanged physical generations 1/1/1. Global layout audit
generations progress 1/2/3 during publication; they are not physical epochs.
Each session serves 1,962 calls for request 43 and 720 for request 67, totaling
2,682 per session / 8,046 overall. Request/native/generation-keyed terminal
proofs agree, with 82,391,040 payload bytes. Queue depth reaches 4; one active
request is used. Fresh first lease sets are 11-17 and 31-37; full sets are
disjoint. Zero fallback, reset, stale execution, request restart, helper
rematerialization error or rejected helper candidate. Semantic checks pass.

The native weight-source proof opens the existing Qwen `HTP0.ffn.gguf`,
`HTP1.ffn.gguf`, `HTP2.ffn.gguf` paths with `weight_source=ffn_shard`. File
bytes are 3,208,644,448 / 3,208,644,448 / 3,208,644,480; resident tensor bytes
are 3,208,642,560 per session. Stored masks are layers 0-5 / 6-11 / 12-17,
maximum width 17,408. Index, shard, parent, worker, geometry and operator-plan
identities are checked. No shard generation or native rebuild was performed.

## Software and replay evidence

204 focused tests PASS in 122.908 s: `test_matched_comparison`,
`test_adaptive_decode`, `test_late_helper_energy_policy`, `test_adaptive_runtime`,
`test_session_cow_transaction`, `test_offline_phone_residency`,
`test_bridge_lifecycle`, and `test_replay_determinism`. The full scheduler
harness was not repeatedly run. `git diff --check` passes.

Both canonical replay goldens are unchanged:

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`.
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`.

The additional saved physical-window replay changes only the diagnosed
decision sequence: token 11 stays PROBING instead of EXPLOITING at 0%; the
current baseline is collected and token 21 selects the existing 100% candidate.
Its reservation is 14.639015 s within 15.150489 s. Two replay outputs are
byte-identical. No old observations, hashes or golden payloads were replaced.

Exact code/test files changed, relative to `research_dev/scheduler/`:

- `_internal/adaptive_decode.py`
- `_unified/adaptive_decode_control.py`
- `adapters/bridge.py`
- `adapters/phone_session.py`
- `tests/test_adaptive_decode.py`
- `tests/test_adaptive_runtime.py`
- `tests/test_bridge_lifecycle.py`

Documentation changes are `research_dev/talks.md` and this report directory:
`README.md`, `COMPARISON.json`, `MATCHED_SUMMARY.json`, `REUSE_AUDIT.json`,
`SAVED_PAIR_REPLAY.json`. All unrelated dirty/untracked user files and previous
physical artifacts remain untouched. No runner policy, route forcing, commit,
push, long trace, GDM stop or unrelated-process termination.

## Identity, artifacts and limitations

Physical prefix: `/home/zhihao/s42-paired-verification-20260908-v1`.
`-deploy` is the frozen source; `-inputs` preserves configs, preflight,
commands, audits and source manifest. `-gate/run` and `-desktop/run` preserve
both complete runs, streams, snapshots, journals and terminal proofs.

Head: `99449bafade0b2c15de4410feda832035c2f2d83`, plus identical dirty source
in both arms. All 215 deployed source files remain identical to the tested
local tree. Internal source-manifest identity:
`sha256:cdf585e5c3db43eab3871ab98d33f6bd74a758df363688b7ab2712892603eb23`.
The table below lists file hashes, not internal record hashes.

| Artifact | File SHA256 |
| --- | --- |
| `-gate/run/RESULT.json` | `3b8e420623b675785d11cc738fe2dedbadd4d4be4e381c7695e1b3b75532a82f` |
| `-desktop/run/RESULT.json` | `2f30088052871ed8b69fd7cb5514fd7cd01db6fcc2cad15c7b674086a3a3a62c` |
| `-inputs/SOURCE_MANIFEST.json` | `e40bd822cc3c0bddf18c010199108ef94a56c4a6aef6e4c32aaa58a40a481ac9` |
| `-inputs/PREFLIGHT.json` | `1d808a59df22002ba5271e9128f5122265d18b17861ca49e0163be89402f9e06` |
| `-inputs/COMPARISON.json` | `2be1bf616de0c1560ad3558a342da3f6ec7fa0f179fc36e81d4a66d467036071` |
| `-inputs/MATCHED_SUMMARY.json` | `d54a78b74f1c996e6c85687d6ac67c1792472a0e2d8f4ebc85e47bc29674d523` |
| `-inputs/REUSE_AUDIT.json` | `9e429a5136d5409d23bbff223c277295a21c21bb89dc483ffa121cbea73a4536` |
| `-inputs/SAVED_PAIR_REPLAY.json` | `eab1df5f5bb9033773f84f57e30d9623e366258f164dbffa8b5fdf50e3ff7dc9` |
| `-inputs/RUN_ARTIFACT_HASHES.json` | `acfda8f0e595e8b67937d989ae96e540b7d63b04c39ae122b0bf51f275c41da0` |

The inventory covers 1,978 files / 150,760,384 bytes. Configuration, prompts,
seeds, 341/133 token requirements, models, binaries, phone shard hashes,
catalog, placement, qualification and accounting boundaries match. Effective
runtime remains GPU16, context 4096, batch 2048, ubatch 512, parallel 4,
one active request, CUDA graphs disabled. Exact desktop parent:
`sha256:42a30600f56e90e50aca7b72df6e311eeaff1f1e071919a7b07dc5cff2b08477`;
qualification:
`sha256:af83748d0868362ef0321f205a12c092f2bc28420adb2b318032da4ced54b901`.

This is one bounded pair, not a statistical estimate. It does not establish
mixed-model savings or isolate the incremental energy contribution of each fix.
Cold adaptive execution starts 14.125 s later than desktop control, even
though loading overlaps and execution is faster. The fixed arrival gap masks
some first-request speedup. Phone shutdown still costs more than desktop-only.

The warm second execution has both parents and shards already resident. Its
recorded execution energy is 10.445469 kJ desktop / 5.800860 kJ adaptive,
44.47% lower at nominal phone power, but it is not a separately controlled
pre-resident steady-state campaign. Preparation overlaps desktop work, so its
receipt energy cannot be added again or treated as isolated phone preload.
No measured preparation break-even count is claimed. No further trace ran.
Final checks leave ADB available and GPU memory at 3,178 MiB used / 12,770 MiB
free, 0% utilization; GDM is untouched.
