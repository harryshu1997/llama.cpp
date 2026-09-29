# Probe recovery after reduced-run coverage failure

Status: bounded physical v2 PASS, 3/3 requests with exact terminal proofs,
semantic output acceptance, and nonzero phone work. All 198 distinct focused
tests pass; both replay goldens are unchanged. The first failed retest and the
user-stopped 24-request desktop control remain preserved. No larger run or
baseline was resumed, and this is not a matched savings comparison.

## Failure and correction

| Failure | Correction |
| --- | --- |
| Initial pair budget fails, then fresh baseline evidence cannot restart probing | Retry only the still-pending initial-baseline stage when the same affordability check passes |
| Historical baseline allows probe-first startup, but a current baseline is missing at qualification | Complete a fresh paired measurement before exploitation; historical evidence is not substituted for it |
| Inferior quarter-fraction probe is eliminated during warmup, falsely labeled incomplete, and a valid winner is discarded | Record candidate rejection separately, advance the existing probe order, or select the still-qualified winner |

All three added regressions failed before the patch. The final focused run:

```
python3 -m unittest \
  research_dev.scheduler.tests.test_adaptive_decode \
  research_dev.scheduler.tests.test_adaptive_runtime \
  research_dev.scheduler.tests.test_late_helper_energy_policy \
  research_dev.scheduler.tests.test_llama_server_adapter \
  research_dev.scheduler.tests.test_replay_determinism -q
```

155 tests PASS in 86.617 s. No full scheduler harness was run. The intermediate
patch also exposed two existing-regression failures; narrowing the new warmup
branch restored the established measured-rejection reason and verification
cleanup. Existing assertions were not weakened.

Replay hashes, unchanged:

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

Reconstruction of saved Gemma 49 at token 67 now issues
`PROBE_CANDIDATE_REJECTED` and the existing qualified 100% policy instead of
`PROBE_INCOMPLETE` and 0%. The policy's artifact, geometry, parent, measurements,
and budgets are unchanged; this is a decision replay, not a counterfactual
physical energy measurement. Genuine negative latency/payback results, the
80-token cap, memory, leases, generations, and terminal checks remain strict.

## Scope and artifacts

Incremental production/test files:

- `_internal/adaptive_decode.py`
- `_unified/adaptive_decode_control.py`
- `tests/test_adaptive_decode.py`

Documentation: this report, the preceding probe-continuity report, and
`research_dev/talks.md`. No native binary, graph mode, parent placement, shard
format, session transaction, reset path, qualification, commit, or push changed.

Fresh physical prefix: `/home/zhihao/s42-probe-recovery-20260909-v1`.
The bounded fixture contains original Qwen 43 (341 output tokens), Gemma 44
(265), and Gemma 49 (491), arriving at 1/21/41 seconds. It uses normal energy-aware
scheduling and measured adaptive history from the completed failed-coverage
run. Routes, sessions, and fractions are not forced. This changed arrival
fixture is a mechanism retest, not a matched trace or savings comparison.

Preceding adaptive failure evidence remains under
`/home/zhihao/s42-normal-reduced24-20260909-v1-gate/run`. The user-stopped desktop
control remains under the corresponding `-desktop/run`; its exact runner PID
2785134 was sent SIGINT, cleanup completed, and no unrelated process was
stopped. Ten original request streams completed before the interruption.

V1 pre-terminal-repair file hashes:

- Corrected adaptive controller: `fa8e16f559de6cc3eec377fd268f876c3b34ce4bb5dbe160ef14a1d0e86cb55b`
- Unified event export: `584b3e0555d80cb63f213526495b5fe7b6ce5424f74695f114c34744cff478b4`
- Focused adaptive tests: `b9f6655b0f80fc20b0bf86920c868e8afd94381c250fd8ce77dd48127098d683`
- `-inputs/COVERAGE_DIAGNOSTIC_AFTER.json`: `1b631195099c0c1624a70e24b394b183a393a9a6b22596ea9a911316519949d2`

## Bounded v1 failure and terminal repair

Preserved failure: `/home/zhihao/s42-probe-recovery-20260909-v1-gate/run/FAILURE.json`.
SHA-256: `69525a60dc9e6df306d68c4bb50b561db80bf9219d214976fb05cfcce7c2a8f1`.
Source manifest: `e4cfac26c929606517324325769648dc4de198516c4c0a97abfdfae9925378eb`.
Preflight: `3d350cf33e8cf45d890f210dcf154ff917596fd789c3a18d7853d2c0aff131ec`.

Qwen 43 produced 341 tokens and 1206 native FFN calls. Gemma 44 produced all
265 tokens and 2000 calls, but its terminal validation failed; Gemma 49 never
ran. These are stream/native counts, not a successful 3/3 gate or savings claim.

The final Gemma stats query arrived after its server slot was released. The
adapter recorded `stale_slot_stats_discarded` as window 66 under a positive
policy. The terminal validator correctly rejected that invalid observation.
Ticket, artifact, and GPU22 parent identities matched. A secondary cleanup
error was `runtime terminal memory status is invalid`; its artifact is retained.
That cleanup defect was then reproduced and repaired: a staged capacity release
had already freed the reservations, but cancellation retained the intermediate
RELEASED status. Cancellation now uses its existing terminal CANCELLED status,
like the failure path, without releasing memory again or relaxing validation.
The regression verifies the original terminal-proof error is still raised,
capacity was released first, no execution receipt is invented, and cleanup
retries are idempotent. Another 41 adapter/runtime-controller tests PASS in
1.277 s, giving 198 distinct focused tests across both sets. The production
file is `_internal/runtime_controller.py`; the test is `tests/test_physical_adapter.py`.

The correction extends the existing released-tail transaction only. After an
exact terminal slot/token confirmation, a stable acknowledged phone policy may
seal at its last valid measurement. The unavailable sample is not appended,
used for qualification, or assigned synthetic counters. The tail is explicitly
unmeasured, recorded by the adapter timing hook as `SLOT_STATS_UNMEASURED_TAIL`,
and still subject to the
unchanged native call-count, generation, contract, and terminal checks. Whole
execution energy boundaries are unchanged. Pending controls/drains, changed
policies, missing acknowledgement history, and non-exploiting states fail closed.

Two additional focused tests cover this path and exact terminal identity.
The native proof regression also rejects missing calls, extra calls, and wrong
generations in a multi-token unmeasured tail. Final focused result: 157 tests
PASS in 88.030 s; both replay hashes above are unchanged. Additional files:
`adapters/http_backend.py` and `tests/test_llama_server_adapter.py`.

Fresh retest prefix: `/home/zhihao/s42-probe-recovery-20260909-v2`, using the same
three-request fixture and history as v1. No remaining reduced24 arrivals,
baseline, long trace, native rebuild, or route forcing.

Preflight SHA-256: `d9e96acf893f68546308d8ac16abe25b375bce96cb78ae1724b843ea67f2d84d`.
The small cancellation correction was deployed after preflight; it changes no
admission or physical contract. Both source snapshots and the explicit two-file
delta are preserved in `SOURCE_MANIFEST.json`, `SOURCE_MANIFEST_FINAL.json`,
and `PREFLIGHT_CODE_DELTA.json`; execution uses the final source manifest.

## Bounded v2 physical result

Artifact: `/home/zhihao/s42-probe-recovery-20260909-v2-gate/run/RESULT.json`.
Read-only native audit: `/home/zhihao/s42-probe-recovery-20260909-v2-inputs/NATIVE_AUDIT.json`.
The paid interval was 745.589 s. All 1097 requested output tokens completed.
No fallback, reset, execution recovery, invalid terminal window, or stale
execution generation was accepted. All three execution receipts, terminal
tickets, adaptive groups, native proofs, and artifact/parent identities match.

| Request | Output tokens | Physical calls | Eligible tokens assisted | Fraction-weighted coverage | Execution interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 43 | 341 | 3912 | 326/340 | 95.88% | 207.678 s |
| Gemma 44 | 265 | 2000 | 250/264 | 94.70% | 148.874 s |
| Gemma 49 | 491 | 3808 | 476/490 | 97.14% | 266.003 s |

Each request used 100% of the FFN columns in its current resident layer mask,
not 100% of the model. Qwen used 12 CPU-resident FFN layers on HTP0/HTP2;
Gemma used eight CPU-resident layers on HTP1. Token-position coverage equals
fraction-weighted coverage here because every positive interval used 100%.
Native calls independently match both numerator calculations. Prefill, loading,
and queueing are outside the eligible decode-token denominator, not removed
from the paid energy interval.

Each request starts at 0%, applies its bounded nonzero policy at token 4,
measures the current baseline, records `VERIFICATION_VERIFIED` with
`CURRENT_PAIR_IMPROVES` at token 23, and resumes positive exploitation at
token 26. These verification outcomes remain DIAGNOSTIC, not copied or
synthesized qualification. All three finish with the 100% policy. The native
audit found no zero-call eligible request and no execution recovery.

Qwen and Gemma 44 exercised the repaired released-phone-tail path: their last
valid measured boundaries were tokens 338 and 262, leaving three explicitly
unmeasured tail tokens each. Gemma 49 used the existing released-control tail
path for its last two tokens. Every native call, including these tails, passed
the unchanged exact-count and generation checks. The unavailable samples did
not enter qualification. The grouped observations persist the tail counts and
reasons; the generic BurstGPT runner does not currently export the adapter's
raw timing-hook list. This persistence limitation is not hidden by claiming
that `SLOT_STATS_UNMEASURED_TAIL` itself appears in RESULT.json.

## Residency and timing

| Publication | Session/model | Session generation | Load authorization to physical READY | Scheduler READY time |
| --- | --- | ---: | ---: | ---: |
| Layout 1 | HTP0 Qwen | 1 | 11.317 s | 21.687 s |
| Layout 2 | HTP1 Qwen | 1 | 19.967 s | 42.264 s |
| Layout 3 | HTP2 Qwen | 1 | 16.527 s | 59.449 s |
| Layout 4 | HTP1 Gemma | 2 | 11.767 s | 86.552 s |

The generator selected HTP1; it was not forced. HTP0 and HTP2 retain generation
1 and their exact Qwen identities. They produce 1956 calls each. HTP1 generation
2 produces 5808 Gemma calls. There are four physical loads total: the initial
three Qwen sessions plus one Gemma replacement. Gemma 49 causes no shard reload
or session-generation change. The maximum initial resident-weight total is
9,625,927,680 bytes; the mixed total is 9,248,440,320 bytes, with recorded layout
workspace 6,118,912 bytes. No Llama phone service is admitted in this fixture.

The first request begins desktop execution at 74.882 s, before mixed-layout
READY at 86.552 s. Its first decode token is after READY. This supports overlap
with desktop prefill, not serving continuity during replacement: the physical
call audit has zero retained-session calls before or during this load, because
replacement finished before Qwen's first phone call. No retained-session gap
ratio, reverse replacement, or rollback is claimed from this particular run.

Arrival-to-execution delays were 73.882 s, 305.670 s, and 435.196 s respectively;
they include loading and waiting behind the desktop execution queue. They are
not labeled phone-preparation waits. There were six scheduling attempts for
three actual executions; auxiliary attempt lists include the exact final
execution ticket in each case. Scheduler decision time: median 212.552 ms,
maximum 679.818 ms, total 1.841 s across six decisions.

One telemetry outage was explicitly deferred and recovered after 39.279 s;
it began with an HTTP timeout and an unavailable ADB device during FunctionFS
startup. One early Qwen helper-refresh rejection (`ready helper rematerialization
input is invalid`) at 74.723 s was retried successfully against layout 4 at
86.598 s, before decode. It did not recur or leave the request at 0%. These
recoverable observations remain in RESULT.json; they are not relabeled absent.

## Energy, scope, and remaining work

CPU RAPL measured 37.5469 kJ and GPU NVML measured 22.2718 kJ over the paid
interval. Phone power is assumed, using the existing 0.875 W idle treatment.

| Assumed active phone power | Assumed phone energy | Fleet energy including preparation |
| --- | ---: | ---: |
| 3 W | 0.9833 kJ | 60.8020 kJ |
| 4.5 W | 1.2170 kJ | 61.0357 kJ |
| 6 W | 1.4506 kJ | 61.2693 kJ |

There is no new matched desktop baseline, so no fleet-saving percentage or
break-even count is asserted. The cold preparation/queue delay, startup
telemetry outage, raw timing-event export, and actual serving-during-load
measurement remain separate work. The bounded three-request coverage result
does not prove that all 24 requests will have the same coverage or savings.
No additional physical run is active.

Final hashes:

- RESULT.json: `36e78df08a6572bc14860709dc9970affab9d99ab2499705a9a597f392765604`
- ADAPTIVE_DECODE_OBSERVATIONS.json: `31679e1578cb03e9f6d05d8ff9d21c3d4e9bfdee2b1cc2440c89c07e3d5b56a0`
- NATIVE_AUDIT.json: `2ff1455194866bc49573cf12e15197b46ab4a4f1f4dba0d4ffe5ce2690d8ff71`
- SOURCE_MANIFEST_FINAL.json: `e16bf94c7db6443613718c41d1649fd644561e031c89d1ac534f18ea2e182023`
- PREFLIGHT_CODE_DELTA.json: `267dddbde3bb4bdab3f66f1177d1d8b852fc4a252ac9d61528b9f0ac52001964`

Final incremental production/test files for this stop/fix/retest task:

- `_internal/adaptive_decode.py`
- `_internal/runtime_controller.py`
- `_unified/adaptive_decode_control.py`
- `adapters/http_backend.py`
- `tests/test_adaptive_decode.py`
- `tests/test_llama_server_adapter.py`
- `tests/test_physical_adapter.py`

Documentation changes are this report, `reports/20260909-probe-continuity/README.md`,
and `research_dev/talks.md`. Fresh configuration, command, diagnostic, and audit
files are under the v1/v2 `-inputs` directories. No native binary, model/shard
format, unrelated worktree change, commit, push, or PR was modified by this fix.
