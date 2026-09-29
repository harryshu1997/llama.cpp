# Mixed-session host-stall investigation

Status: V3 passed the bounded forward/Gemma/reverse/fault/retry gate. Reduced
matched V4 completed both arms (24/24), with 3.10% lower nominal fleet energy
but 6.49% longer duration. It failed residency acceptance with 95 layout failures.
V5 had four successful preparations and zero preparation failures, but its last
request hit a historical-acknowledgement recovery defect. That correction passes
111 focused tests and all 1,038 canonical tests. Both V6 arms completed 24/24.
Adaptive has four READY preparations, zero failures and 8,414 calls. The strict
matched comparison is valid: 5.39% lower nominal fleet energy, but 8.57% longer
duration. Coverage and latency acceptance remain incomplete; the 25% energy
target is not met.
Historical telemetry-recovery V1-V4 and this
investigation's V1 artifacts are preserved as recorded.

## Defect fixed in this turn

Progressive READY publication can queue a second control while the first
control is awaiting its server acknowledgement. If the first control executes
for only the transition interval, it never opens a regular measurement window.
The controller discarded its applied acknowledgement, making its real calls
unverifiable at terminal validation despite exact aggregate call counts.

The existing adaptive transaction now retains that exact acknowledgement for
an immediately superseded control. A shortened receipt carries the original
generation and policy but stays excluded from measurement and qualification.
Deferring the second control preserves the old acknowledgement. Cleanup clears
the pending acknowledgement. No validator, wire format, session lifecycle,
shard format, generation fence, fraction policy or native binary was changed.

Changed production file: `_internal/adaptive_decode.py`.
Regression: `tests/test_adaptive_decode.py`. The regression failed before the
fix, then passed, including rejection of a different control generation.
`research_dev/talks.md` records the investigation and preserved failure.

Focused adaptive/server-proof/replay tests: 82 PASS in 69.909 seconds. The
additional deferred-control subcase passed separately. These are overlapping
test runs, not 83 distinct tests. Both replay goldens remain unchanged:

- v3: `f78d2b2c37a3880a523eba4f5315ada0207678c841d633229782bfa3a05c1829`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

## Physical investigation V1

Artifact: `/home/zhihao/s42-mixed-host-stall-20260907-v1-gate/run`.
The native Qwen request produced 341 tokens and 4,728 calls, but its terminal
proof failed. All three sessions remained generation 1. Twelve calls under
control generation 4 lacked their acknowledgement in the one-token window
114-115. Control generation is distinct from physical session generation.
The mixed replacement phase was not reached. See [V1_AUDIT.json](V1_AUDIT.json).

| Session | Calls | Native load to READY |
| --- | ---: | ---: |
| HTP0 | 1,806 | 12.348 s |
| HTP1 | 1,566 | 19.016 s |
| HTP2 | 1,356 | 17.245 s |

The first-to-all native READY interval was 62.509 s. No native same-token,
same-width host submission gap above the diagnostic 40 ms search filter was
found in V1; this was not the mixed-phase interruption acceptance check.
The 2x equivalent-call interruption threshold remains unchanged.

## Profiling scope

`/home/zhihao/s42-mixed-host-stall-20260907-v1-host-proc` contains 18,532
read-only /proc samples of gate descendants at a nominal 20 ms cadence.
Sampling reads CPU runtime, run-queue delay, task state/wait channel, syscall,
page faults and system pressure. It changes no affinity, priority, sysctl,
worker or USB state. The observer used 100.976 CPU seconds over 375.060 wall
seconds, about 0.269 CPU cores. These diagnostic runs are not a matched energy
comparison. The 229.510 ms maximum sample duration overlapped a cold model
disk-page wait, not a demonstrated serving-time stall.

The original telemetry V4 243.966 ms desktop pre-submission stall remains
unresolved until the profiled replacement phase is exercised. It must not be
attributed to the phone uploader merely from temporal overlap.

V2 uses fresh `-v2-inputs`, `-v2-deploy`, `-v2-gate`, and `-v2-host-proc`
directories. Preflight passed with the same qualified graph-disabled server,
desktop parents, model artifacts, and real FFN shard indexes. Source, command,
profiling and result files are preserved in those directories.

## V2 outcome and retry fix

See [V2_AUDIT.json](V2_AUDIT.json). Cold Qwen, online Qwen, Gemma and reverse-
fault Qwen completed with 4,866, 2,292, 6,416 and 1,632 phone calls respectively.
Only the dynamically selected HTP2 changed. HTP0/HTP1 each made 198 calls
during forward loading and passed the unchanged equivalent-call 2x check.
Gemma attached without another load and reached 89.683% weighted coverage.
Qwen selected zero after its probes and reached only 39.412% coverage.
Reverse loading began after that Qwen request finished, so rollback serving
interference is unproven, not PASS. The terminal has 15,206 calls, status 0,
and zero reset recoveries.

The injected reverse load changed HTP2 to Qwen generation 3 and physically
restored Gemma generation 4. Retained sessions stayed generation 1. Clean
retry failed with `phone transition session generation differs`: the restored
READY view still carried the old forward transition's generation-1 source
authorization. The controller now retires that obsolete source metadata from
the restored view, preserving costs, geometry and historical proofs. The
strict compiler check is unchanged. An extension to the existing offline COW
regression reproduced the physical failure, then passed retry generation 5
with no retained-session reload. Focused placement/COW/offline/replay:
139 PASS in 109.023 seconds, both replay hashes unchanged.

Additional changed files: `_internal/model_placement_controller.py` and
`tests/test_offline_phone_residency.py`. V3 uses new input/deploy/result paths
and omits the optional /proc observer; ordinary native timing, call, telemetry,
energy and proof logging stay enabled. No new binary qualification is needed
because no native binary, desktop placement or runtime mode changed.

## V3 bounded physical PASS

Artifact: `/home/zhihao/s42-mixed-host-stall-20260907-v3-gate/run/RESULT.json`.
SHA256: `27ba3e17f7f7313d4be77799681518e09d823b16364c0430e5238f7f21d45b2b`.
See [V3_AUDIT.json](V3_AUDIT.json) and
[V3_TELEMETRY_AUDIT.json](V3_TELEMETRY_AUDIT.json).

| Request | Phone calls | Weighted coverage |
| --- | ---: | ---: |
| Cold Qwen | 4,812 | Progressive preload validation |
| Online Qwen | 3,726 | 75.000% |
| Gemma | 6,416 | 89.683% |
| Reverse-fault Qwen | 3,648 | Retained-service validation |
| Reverse-retry Qwen | 4,230 | Retained/new-session validation |

All five requests completed with exact terminal proofs. Only the dynamically
selected HTP2 changed: Qwen 1 -> Gemma 2 -> Qwen 3 -> restored Gemma 4 ->
Qwen 5. HTP0/HTP1 remained Qwen generation 1, with one physical load each.
HTP2 had five loads, including preload, forward, fault target, restoration
and clean retry. Runtime fractions and cross-request attachment caused no
reload. There was one phone terminal, status 0, 22,832 calls and zero resets.

| Transition | Native load to READY | Retained calls HTP0 / HTP1 | Worst matched gap ratio |
| --- | ---: | ---: | ---: |
| Forward Gemma 2 | 19.605 s | 214 / 210 | 1.708x |
| Reverse Qwen 3, then restored Gemma 4 | 15.043 s + 36.697 s | 552 / 558 | 1.706x |
| Clean reverse Qwen 5 | 17.956 s | 199 / 198 | 1.359x |

Every retained-session equivalent-call class passed the unchanged 2x limit.
The cold session loads took 11.087 / 18.561 / 11.852 s; first-to-all native
READY took 55.717 s. Forward drain-to-quiescence took 1.418 s, including
closing a seven-token old-policy window without qualifying that short sample.
The distinct host loading-to-publication interval was 19.744 s.

There were 9,540 valid planning snapshots, 247 unavailable, nine initially
missing, and none stale. Sampled recoveries ranged from 0.176 to 6.096 s
(plus initial observation startup); they are not continuous wire measurements.
All six replacement admissions used valid samples, maximum age 3.062 s.
The fault restoration remains tied to its authorized compensation transaction.
All captures/recaptures reported by the native graph-mode timing parser were
zero. The original 243.966 ms pre-USB host stall did not recur in these passes;
its cause is not claimed fixed by the two lifecycle corrections.

The cold validation and online lifecycle intervals contain both preparation
and inference. They are recorded separately, not described as isolated load
energy or as matched savings. A new same-source/runtime/parent 24-request
comparison is the next experiment; no old baseline will be used for a claim.

## Reduced matched control V1 and admission correction

`/home/zhihao/s42-mixed-matched24-20260907-v1-baseline/run` is preserved as
FAIL, with five completed request journals before arrival 41 stopped admission
at 431 s. No adaptive arm ran. The scheduler projected a queued Gemma hot-to-hot
reuse as a new modelled allocation, replacing 12,662,603,776 measured GPU
reclaimable bytes with 12,419,124,020 bytes. Qwen's later load then lost valid
eviction credit. Exact hot reuse now preserves the physical allocation; the
saved admission replay queues Qwen without changing its desktop parent or
memory thresholds. See the 01:14 EDT entry in `research_dev/talks.md`.

Two additional helper regression causes are corrected: a warm template now
recognizes its existing exact queue authorization, and alternate-parent probe
derivation excludes non-desktop/unadmitted rows rather than discarding valid
desktop contracts when encountering one. No helper assertions were weakened.
The telemetry test import now works under discovery.

Canonical suite: 1,034 PASS in 183.116 s. v8's replay hash is unchanged. v3
changes to `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`:
only two duplicate EVALUATED events disappear; selected routes, helper events,
geometry/generations and success/failure lifecycle branches are unchanged.
The full decoded diff is checked in under `tests/data/replay/`.

## Reduced matched V2: control PASS, adaptive FAIL

The preserved prefix is `/home/zhihao/s42-mixed-matched24-20260907-v2`.
Both arms used source manifest
`ca5d7749885c2103158c62c1aeb4de199564122e5cf565742408f8a40c0a4b6b`,
Qwen GPU16 and Gemma GPU22, unchanged qualified graph-disabled binaries,
and the real FFN shard indexes. No GPU process or GDM was stopped.

| Arm | Successful terminal proofs | Paid span | Fleet energy |
| --- | ---: | ---: | ---: |
| Desktop | 24/24 | 1,623.450 s | 149.327 kJ |
| Adaptive | 14/24 | Incomplete | No valid total or saving |

Desktop CPU/GPU energy was physically measured: 98.341924190 / 49.564182555
kJ. Its assumed idle phone energy was 1.420518820 kJ. Desktop RESULT SHA256:
`c447d2915dd48ce24a577db76ec6747d3541824a7f4f0877d9e631374fdefab9`.
Adaptive has two failed and seven cancelled requests; Gemma 44 produced its
output but failed terminal proof. A dispatch queue's COMPLETED state releases
capacity and is not sufficient evidence of successful request completion.

Qwen 39 failed because an FFN statistics request, queued at server 144.191452
s, had a five-second HTTP deadline while another slot's 178-token prefill ran
for 10.288999 s. Qwen 45 later hit the same problem applying a control. This
was a desktop control-channel delay, not a phone health telemetry outage.
The adapter now uses the existing bounded execution-service budget, capped by
the request timeout. Independent lease renewal and native execution continue.
The native FFN RPC timeout, binary and transport are unchanged.

Gemma 44 applied control generation 61 at token 264 of 265 and made eight real
calls under that control. The grouped observation omitted its final ack because
no subsequent measurement window opened. It now retains that exact ack for
tail call counts and generation validation. Wrong request, slot, token, time,
policy hash and generation remain rejected; old observations serialize
unchanged when no final ack exists. This is evidence preservation, not a
relaxation of the terminal proof.

New production changes for these defects: `adapters/http_backend.py`,
`adapters/llama_server.py`, `_internal/adaptive_decode.py`, and
`_internal/adaptive_decode_contracts.py`. Regressions are in
`tests/test_adaptive_decode.py` and `tests/test_llama_server_adapter.py`.
Both regressions reproduced the old failures before passing with the fixes.

Final canonical suite: 1,035 PASS in 184.062 s. Both replay goldens are unchanged
by these fixes. The preceding run's only failure was the pre-existing 10 ms
timing test at 10.435 ms; both its isolated rerun and full verification passed
without source or threshold changes. Logs are preserved under
`/tmp/s42-control-prefill-recovery-20260907-MCUMF7/`.

Adaptive FAILURE SHA256:
`972b99d1fa851f7e74ccfff5b7af0ecd9286db86674f36aba0e37b47ec564535`.
FAILURE_REQUEST_HELPER_EVENTS SHA256:
`85dd60acc77f68d46408a17336ee0fc3183734553097f3c7cf11b4828a072dc3`.
Historical adaptive observations are present in the saved store; they must not
be mistaken for this run's requests when auditing placement or coverage.

## Reduced matched V3: last-request release race

Prefix: `/home/zhihao/s42-mixed-matched24-20260907-v3`.
Desktop PASS: 24/24 in 1,682.063140 s, 149.757023688 kJ fleet energy.
CPU/GPU: 98.239388357 / 50.045830083 kJ measured; phone idle:
1.471805248 kJ assumed. RESULT SHA256:
`d9e98b13cac0023a76c42c4fbf60b12732ac8e8577308941330fad15dc89265f`.

Adaptive completed Qwen 39, Qwen 45 and Gemma 44, including terminal proofs;
the preceding two defects did not recur. It failed after 23 successes on Qwen
54 (31 tokens). Qwen 55 (12 tokens) completed. Qwen 54 had acknowledged 0%
at control generation 2, token 7. Its native slot was released at 4:41.066484;
a stats task at 4:45.284557 correctly rejected that released slot. Recovery
then rejected an ordinary pending window because it was not a sealed tail.

The correction waits for matching terminal progress and records the unavailable
baseline interval as diagnostic, not qualified. It reuses the exact historical
zero-policy acknowledgement and preserves earlier phone calls. It does not
allow the old all-baseline exception to erase those calls. Wrong slot, token
count, zero acknowledgement or unexpected generation still fail. No native
binary, qualification, session lifecycle or memory policy changed.

Files: `adapters/http_backend.py`, `adapters/llama_server.py`,
`_internal/adaptive_decode.py`, `_unified/adaptive_decode_control.py`,
`tests/test_adaptive_decode.py`, `tests/test_llama_server_adapter.py`.
Focused: 108 PASS. Full canonical verification: 1,036 PASS in 184.825 s.
Both replay goldens unchanged. Three focused terminal/proof tests also pass
after the final current-policy equality guard.

Adaptive FAILURE SHA256:
`e72c2fe0b5c21c1dc58cace48a1899709b2f94701c133583bd27d05584567b84`.
Failure helper-events SHA256:
`1f655f8c8fa3f2fc4f4c1524e0167842cb4111fe8e7a0ac1a75dcc130102f9df`.
Failure adaptive-store SHA256:
`6ed055114825c7c86f9f31893bd72d61047835d26827c85b7a27ff0cb953ff6d`.
There is no valid complete adaptive energy total or matched saving for V3.

V4 uses fresh directories. Its source-manifest SHA256 is
`8dc08e269e72058cf1b0b02c8f3599d6a2de9fd543eed2aa4117369d25360e5a`;
preflight SHA256 is
`6d18d2421a914abfd5e8680566f732a60f70d4ce95c133cf2666d82223b8af8c`.
The adaptive arm runs first, followed by a fresh matching baseline only if it
passes. Catalog, replay, binaries, FFN shards and desktop parents are unchanged.

## Reduced matched V4: requests complete, residency acceptance fails

Adaptive RESULT SHA256:
`5de568fbeee3cb711d7ea0fc7da41271b262dff92c2e72e219bb07bc736cb885`.
All 24 requests completed with terminal proof. Ten used the phone (11,016
calls), but 95 layout transitions failed. The first failure was the third
initial Qwen shard at 55.095104 s: physical loading completed after its owner,
request 35, was re-admitted under a new ticket. Completion looked up the removed
attempt-0 ticket and raised `runtime ticket is unknown`. Later plans encountered
the resulting unpublished physical occupancy and correctly rejected incomplete
eviction authorization. The cascade, not telemetry admission, delayed Gemma.

| Metric | V4 adaptive |
| --- | ---: |
| Successful requests | 24/24 |
| Phone-assisted requests | 10/24 |
| Layout proposals / READY / failed | 99 / 4 / 95 |
| No READY phone layout | 20.7 s |
| Executing without model residency | 320.1 s / 1,694.0 s (18.9%) |
| First Qwen / Gemma residency after arrival | 20.7 / 651.8 s |

The request-level PASS must not be presented as scheduler acceptance. Gemma
has 4,056 calls on two of six requests; Qwen has 6,960 on eight of fifteen.
Among requests with direct eligibility events, fraction-weighted coverage is
92.74% Gemma and 41.87% Qwen. The Gemma number excludes requests 36 and 44,
which finished before their model became resident; it is not trace-wide
coverage. Llama's three requests remain desktop-only.

All five physical load admissions had VALID phone observations; maximum age
was 1.881157 s against the unchanged 5 s limit. One planner deferral/recovery
pair spans 19.309495 s, which includes time until the next planning check and
is not a continuously measured network outage. The HTTP fallback supplied
observations during FunctionFS service; no USB reset was used.

The correction captures immutable model/executor identity in the existing
preparation record, without retaining request leases. READY publication uses
that identity and exact physical evidence even if the request replans or is
cancelled. Portfolio refresh uses current nonterminal requests. Changed:
`_unified/common.py`, `_unified/helper_preparation.py`, and
`tests/test_replay_determinism.py`. The regression reproduced both old-ticket
failures and now passes; changed physical generations still reject.
Focused lifecycle/telemetry/replay: 78 PASS in 125.448 s. Both goldens unchanged.
Final canonical verification: 1,037 PASS in 191.516 s, goldens unchanged.
V4 baseline completed 24/24 in 1,610.563762 s. Its RESULT SHA256 is
`9e9c63e465ed3fa9bae163fdafac99864e011c5983becf28b0c9badeb91523b3`.
The strict same-source, artifact, runtime, parent and boundary comparison passed
identity validation. See [V4_MATCHED_COMPARISON.json](V4_MATCHED_COMPARISON.json)
and [V4_MATCHED_AUDIT.json](V4_MATCHED_AUDIT.json).

| Phone active assumption | Desktop fleet | Adaptive fleet | Saving |
| --- | ---: | ---: | ---: |
| 3 W | 147.178 kJ | 142.384 kJ | 3.26% |
| 4.5 W | 147.178 kJ | 142.622 kJ | 3.10% |
| 6 W | 147.178 kJ | 142.859 kJ | 2.93% |

CPU/GPU energy was physically measured: desktop 96.690 / 49.079 kJ, adaptive
89.900 / 50.647 kJ. Phone energy is assumed, using 0.875 W idle in both arms.
These are end-to-end paid trace totals including online preparation, not
steady-state-only savings. Preparation overlaps inference, so isolated load
energy and break-even reuse are not identified by this experiment.
The run span was 6.49% longer; per-request latency ratios have median 1.013,
95th percentile 1.428 and maximum 1.498, exceeding the 1.25 maximum target.

Known report-only limitations: the comparison's auxiliary fraction histogram
includes imported historical observations, and its flat transport summary does
not read nested terminal receipt counters. Do not use those fields as coverage
or zero-call evidence. The separate audit follows each request's exact current
observation hash; per-session physical proofs give 2,916 / 5,382 / 2,718 calls.
These reporting issues are recorded, not silently used to claim acceptance.
V5 preflight follows normal V4 owned teardown in a fresh artifact directory.

## V5 publication succeeds; historical zero-ack recovery correction

V5 completed 23 request proofs before Qwen 54 failed terminal recovery. Its
four preparations all published READY, at 22.315 / 46.471 / 60.805 / 317.163 s.
The first three progressively loaded Qwen; the fourth replaced only HTP1 with
Gemma generation 2, retaining Qwen HTP0/HTP2 generation 1. The first Gemma
residency still arrived 76.163 s after model arrival, missing the one-load-time
target. This planning/preparation delay is separate from the fixed publication
defect and remains visible rather than being labelled PASS.

Artifact: `/home/zhihao/s42-mixed-matched24-20260907-v5-adaptive/run`.
FAILURE SHA256:
`cdb598b562939f4929db7dfd565a5d2279a20ad201c1fc8c0992e3c05c67d2bb`.
The failure is `adaptive boundary acknowledgement differs`: recovery copied
the zero-mask acknowledgement applied at token 7 into a later unchanged
baseline window. An applied acknowledgement belongs only to its actual start
boundary. The correction preserves it in the original window and validates
continued baseline execution against the historical record. Missing or stale
acknowledgements and unexpected phone calls under the zero generation still
fail; no boundary invariant or qualification check is weakened.

Changed: `_internal/adaptive_decode.py`, `adapters/llama_server.py`,
`tests/test_adaptive_decode.py`, `tests/test_llama_server_adapter.py`.
Focused adaptive/server/cohort/replay: 111 PASS in 78.038 s; goldens unchanged.
Final canonical suite: 1,038 PASS in 186.825 s. Full test-log SHA256:
`6f8e23c9646fd29cdafa362f0e72651878391b4577c49380f2719cc5e61e992e`.
The log is preserved in the V6 inputs directory alongside the source manifest
and focused-test log. No source changed after freezing the V6 deployment.
V5's third load finished before its owner replanned, so the forced software
re-admission regression supplies that particular ordering proof. V6 uses fresh
inputs/deploy/results. No V5 baseline is run and no matched V5 energy claim is
valid. All failed artifacts and their native controls are preserved.

## V6 adaptive: terminal and publication PASS, coverage still incomplete

Artifact: `/home/zhihao/s42-mixed-matched24-20260907-v6-adaptive/run`.
RESULT SHA256:
`990718518f65172f548e26bae20edb4a8436c9529b10634be7eb6049481cc7d1`.
Adaptive observation store SHA256:
`4ae21b0ebddcc01d873c7b827528332476a97a182b6d928323e5887feb358bf6`.
Source manifest SHA256:
`35db998b7025b4c5c6252a2286374378f0b327ad9b068bc1454cc24b128d8048`.

All 24 requests completed with accepted output quality and exact terminal proofs.
Qwen 54 completed with 60 calls, retaining its generation-2 zero acknowledgement
at token 7. V6 did not exercise its particular released-slot exception ordering;
the focused regression supplies that ordering proof. There were no request
recovery actions, four physical loads, and one phone terminal with status 0 and
zero reset recoveries. All four load admissions used VALID observations, maximum
age 1.978076 s against the unchanged 5 s limit. The last three admissions used
the HTTP snapshot path while FunctionFS was active.

| Session load | Native load to READY | Host preparation to verification |
| --- | ---: | ---: |
| Initial Qwen HTP0, generation 1 | 11.330 s | 19.047 s |
| Initial Qwen HTP1, generation 1 | 18.539 s | 18.863 s |
| Initial Qwen HTP2, generation 1 | 13.743 s | 14.263 s |
| Gemma on selected HTP1, generation 2 | 21.558 s | 22.642 s |

Logical READY observation times are 20.696541, 39.709990, 54.209280 and
356.609508 s; publication times are 20.793833, 39.955722, 54.421535 and
356.821403 s. Only HTP1 changes artifact/generation. HTP0 and HTP2 retain Qwen
generation 1. Historical Qwen HTP1 generation-1 calls remain in the terminal
proof alongside Gemma HTP1 generation-2 calls. There is no active Qwen request
during this trace's replacement, so retained-call interruption is not measured
here; the preceding bounded gate supplies that separate proof.

The native weight-source receipts identify FFN shard files for every load, with
index/parent/shard hashes, stored/executed masks and widths. Serialized Qwen
HTP0/HTP1 files load 3,208,644,448 bytes each; HTP2 loads 3,208,644,480 bytes.
Gemma HTP1 loads 2,831,157,504 bytes. Resident weights differ from serialized
file size: 3,208,642,560 bytes per Qwen shard and 2,831,155,200 for Gemma.
Initial three-shard residency is 9,625,927,680 bytes plus 243,712 workspace;
mixed residency is 9,248,440,320 bytes plus 4,951,552 workspace. Existing memory
and identity checks are unchanged. Runtime fraction changes cause no reload.

| Reduced trace metric | Historical v8 | V6 adaptive |
| --- | ---: | ---: |
| Completed requests | 24/24 | 24/24 |
| Phone-assisted requests | 2/24 | 12/24 |
| Layouts failed / never prepared | 7 / 4 | 0 / 0 |
| Time without a READY phone layout | 124.1 s | 19.7 s |
| Executing without model residency | 414.7 s (27.7%) | 42.9 s (2.54%) |
| First Gemma residency after arrival | 223.8 s | 115.6 s |

This is a diagnostic historical comparison, not a same-revision energy claim.
The old analyzer used a 32-token short-request threshold; V6 uses the configured
24-token threshold. First Gemma residency still misses the roughly 45 s target:
the target is observed at 241.041177 and 271.017414 s, but its required third
confirmation does not arrive until 333.978688 s. Loading is only the final
22.6 s of that delay. No hysteresis or selection policy was changed mid-pair.

The analyzer displays missing preparation timing for generation 4 because its
`observed_at_us` is 11.043 ms earlier than the proposal's timestamp. Event indices
and `published_at_us` preserve the causal order: proposal 333.982104 s,
PREPARING 334.944708 s. This report uses the direct events above rather than
mistaking that analyzer limitation for a missing load.

| Model | Requests with calls | Calls | Eligible fraction-weighted coverage |
| --- | ---: | ---: | ---: |
| Qwen | 8/15 | 1,782 | 16.14% |
| Gemma | 4/6 | 6,632 | 75.00% |
| Llama | 0/3 | 0 | No resident phone route |

Token-position coverage is 16.84% Qwen and 79.87% Gemma. These denominators
include direct eligibility events only: 784 Qwen and 1,038 Gemma tokens.
Fractions describe the helper's supported FFN mask, not the fraction of all
model layers or total model computation executed on the phone.
Four attached Qwen requests (35, 40, 41, 46) never probe. Three of those lack an
exported terminal rejection reason; request 40 eventually reports insufficient
remaining opportunity. Qwen 43 reaches only 3.75% weighted coverage: early shared-
resource contention is followed by negative-gain/latency window bids and zero
assistance. Gemma 44 attaches and makes 664 calls, but its 24.62% weighted
coverage remains low. Residency correctness is not full coverage acceptance.

The read-only coverage audit uncovered an incomplete auxiliary attempt list for
Qwen 38: it lists attempt 0, while the execution receipt, terminal ticket,
physical proof and exact adaptive observation all agree on attempt 1. The
audit now checks those authoritative identities and flags the auxiliary-list
omission. No production proof or recorded result is rewritten. Imported
historical groups remain excluded by exact observation hash.

Adaptive paid duration is 1,709.418224 s. Measured CPU/GPU energy is
89.576359395 / 50.651073405 kJ; nominal assumed phone energy is 2.011673628 kJ.
The fresh matched comparison is reported below. Isolated preparation energy,
steady-state savings and break-even count are not identified by overlapping
trace intervals.

## Final V6 matched comparison

Desktop artifact: `/home/zhihao/s42-mixed-matched24-20260907-v6-baseline/run`.
RESULT SHA256:
`22ca631ef88e12792df98e13844b7ff0287b2b00bc617048bbae4ea9cfd2f7e5`.
Both arms completed 24/24 requests and 2,274 output tokens. The canonical
comparator accepts the same source manifest, requests, artifacts, binaries,
capability catalog, qualified desktop parents, graph mode and energy boundary.
Qwen uses the qualified 16-GPU-layer parent and Gemma the qualified 22-layer
parent. Neither desktop qualification nor runtime mode was changed to obtain
this comparison. GNOME and unrelated processes were left alone.

| Phone active assumption | Desktop fleet | Adaptive fleet | Saving |
| --- | ---: | ---: | ---: |
| 3 W | 150.338 kJ | 142.026 kJ | 5.53% |
| 4.5 W | 150.338 kJ | 142.239 kJ | 5.39% |
| 6 W | 150.338 kJ | 142.453 kJ | 5.25% |

| Metric | Desktop | Adaptive |
| --- | ---: | ---: |
| Measured CPU-package energy | 98.859 kJ | 89.576 kJ |
| Measured GPU-board energy | 50.101 kJ | 50.651 kJ |
| Assumed phone energy, nominal | 1.378 kJ | 2.012 kJ |
| Paid duration | 1,574.494 s | 1,709.418 s |
| Arrival-to-execution delay, median | 219.037 s | 264.241 s |
| Arrival-to-execution delay, 95th percentile | 396.779 s | 480.310 s |
| Arrival-to-execution delay, maximum | 417.365 s | 495.063 s |

Phone idle power is 0.875 W in both arms. CPU/GPU energy is physically measured;
phone energy is assumed. This is one matched diagnostic pair over the canonical
paid trace interval, including online preparation and scheduler completion
overhead. Offline file generation/transfer and preflight are outside that
interval. It is not an isolated steady-state experiment or a statistical
confidence claim. Isolated fleet preparation energy and a break-even request
count cannot be recovered from these overlapping measurements.

Adaptive saves 8.098827433 kJ at nominal phone power but takes 134.924479 s
longer (8.569%). Per-request execution latency ratio is median 1.023,
95th percentile 1.593 and maximum 1.621, failing the 1.25 maximum target.
Arrival-to-execution delay includes queueing and desktop loading; it must not
be labelled pure phone-preparation wait. Scheduler decision latency is median
0.327450 s, 95th percentile 0.895452 s, maximum 5.990116 s. Each arm performs
six desktop model loads; the adaptive phone separately performs its four loads.

Telemetry audit: 17,973 saved planning snapshots, all VALID, including repeated
cached observations. Maximum valid age is 2.908706 s; 17,971 observations use
HTTP and two use ADB. No outage or recovery interval is observed at this saved-
snapshot resolution, and no expired sample authorizes a load. The preceding
bounded gate separately demonstrates recovery while sessions serve; its
sampled FunctionFS recovery intervals range from 0.176 to 6.096 s.

### Evidence files

- [Strict comparison](V6_MATCHED_COMPARISON.json), SHA256
  `0a556a6d85dbbbf026132ad40024bef66d9fc3a6770b96d62afa92ba92845dff`.
- [Exact current-run coverage/energy audit](V6_MATCHED_EXACT_AUDIT.json), SHA256
  `9b5a313507873f7b12fe1c3b6ed85a5b7071f746cc3334ddc9c4e2f35a34ecc6`.
- [Load, generation, shard and terminal evidence](V6_RUNTIME_EVIDENCE_AUDIT.json), SHA256
  `93bafb590608b987e74eb4df8f593273fcbc3e28c4edbed6ec0fd7003e89ab82`.
- [Phone wait timeline](V6_ADAPTIVE_TIMELINE.json), SHA256
  `cfe35cb429699b112c408dfca43ea55ee6f17ee5936cda3de75b118090821579`.
- [Telemetry snapshot audit](V6_TELEMETRY_AUDIT.json), SHA256
  `29db6bf7ea679ddc90da3e2f76c338817f86bde90392d9bc10952510c24a09df`.

These reports and the read-only analysis scripts are also preserved in
`/home/zhihao/s42-mixed-matched24-20260907-v6-inputs`. Its source manifest,
configuration, preflight, exact commands, focused/full test logs and raw run
directories are retained. Prior failed artifacts remain unchanged. The canonical
comparison's auxiliary historical fraction histogram and flat transport counters
still have the previously documented report-only limitations; the direct audits
above, not those fields, supply current-run coverage and phone calls.
`RUN_ARTIFACT_HASHES.json` in the V6 inputs directory records every file in both
final run directories: 36,110 files, 2,649,930,542 bytes. Its SHA256 is
`a4c0d1b59f87d45943a88aebdc589e778dd40021da6be352ee969af8f5448df2`.

### Source changes in this continuation

Production corrections, all in the existing scheduler path:

- `_internal/adaptive_decode.py`, `_internal/adaptive_decode_contracts.py`:
  retain superseded and final acknowledgements; account released baseline tails
  against their original historical acknowledgement without qualifying them.
- `adapters/http_backend.py`, `adapters/llama_server.py`,
  `_unified/adaptive_decode_control.py`: bounded control-service deadlines,
  exact terminal-release recovery, and proof-preserving tail accounting.
- `_internal/model_placement_controller.py`: retire obsolete forward
  authorization from the physically restored READY view before a reverse retry.
- `_internal/runtime_residency_projection.py`: preserve measured allocation
  for exact hot reuse so subsequent eviction credit is not lost.
- `_unified/automated_selection.py`, `_internal/adaptive_decode_planning.py`:
  reuse exact warm queue authorization and admitted desktop-parent contracts.
- `_unified/common.py`, `_unified/helper_preparation.py`: retain immutable
  model/executor preparation identity, not a request ticket's lifetime or leases.
- `campaigns/burstgpt/compare_ab.py`: accept the explicit saved reduced replay
  while retaining strict matched-input and qualification checks.

Test changes: `tests/test_adaptive_decode.py`, `tests/test_llama_server_adapter.py`,
`tests/test_offline_phone_residency.py`, `tests/test_automated_runtime.py`,
`tests/test_telemetry_recovery.py` (discovery import),
`tests/test_matched_comparison.py`, and `tests/test_replay_determinism.py`.
Replay documentation and decoded evidence:
`tests/data/replay/README.md` and `tests/data/replay/20260907-hot-reuse-decoded-diff.json`.
Documentation/results: this report directory and `research_dev/talks.md`.
No native binary, wire format, shard generator, graph-mode implementation,
qualified parent, memory threshold, or unrelated user change was modified.

Final verification: 111 focused tests and all 1,038 canonical tests PASS.
Final replay goldens are:

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`.
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`.

Only the earlier measured-hot-reuse correction changes v3 in this continuation:
two obsolete duplicate rejection events disappear, as recorded in the decoded
diff. Routes and lifecycle proofs are unchanged. All later corrections preserve
both goldens. The remaining confirmation delay, unprobed requests, low Qwen
coverage, latency regression, report limitations and historical host-stall
uncertainty are not labelled fixed. No further trace, commit or push is launched.
