# Actual-parent helper attachment

Gemma READY-shard reuse is physically proven by v4. The complete bounded
gate remains unproven. V6 stopped on the unchanged retained-session
interruption bound before Gemma submission or reverse/rollback. No more
physical retries are running. No trace or matched energy comparison was run.

## Cause and repair

Two separate ownership mistakes prevented attachment:

- Helper derivation assumed the catalog's desktop control parent. A request
  actually executing on CPU could not obtain a compatible helper from a
  previously prepared GPU+CPU envelope.
- A fresh execution envelope for an already READY replacement attempted to
  recover the old load authorization from request-owned preparation caches.

Helper generation now uses the acquired request's exact desktop parent.
The existing compiler rebuilds operator mapping, endpoint binding,
resource requirements, transfers, and costs. Physical session identity is
checked independently. A new request gets its own envelope and leases;
execution of READY weights does not require an old loading transaction.
PROPOSED partial loads still require their exact replacement authorization.

An unqualified parent/helper combination remains LEARNING/DIAGNOSTIC. It
does not inherit GPU+CPU energy qualification. Missing parent-compatible
helpers are rejected once per unchanged ticket, parent, catalog, geometry,
session-generation map, and learning state; a relevant change permits retry.

## Preserved physical results

| Run | Result | Cause or evidence |
| --- | --- | --- |
| v3 | FAIL | READY execution still requested old replacement authority; 61 failed attempts plus one REJECTED event. |
| v4 | FAIL overall; Gemma reuse proven | Gemma completes with 6,416 calls and no load; reverse checker incorrectly reads generation from a geometry object. |
| v5 | FAIL | Qwen coverage is 34.7058%; no helper rejection or identity failure. |
| v6 | FAIL | Retained-session next-token gaps exceed 2x matched medians near replacement load start; reverse/rollback not reached. |

Artifacts for each run are under
`/home/zhihao/s42-ffn-mixed-session-20260905-vN-gate/run`, with sibling
`vN-inputs` and `vN-deploy` directories. Earlier failures were not overwritten
or relabelled. The v4 decoded audit is [V4_HELPER_AUDIT.json](V4_HELPER_AUDIT.json).
V6 measurements and native terminal proof are in
[V6_FAILURE_AUDIT.json](V6_FAILURE_AUDIT.json). Exact artifact and source
hashes are listed in [SHA256SUMS](SHA256SUMS).

In v4, cold Qwen, online Qwen, and Gemma complete with 4,986, 3,720, and
6,416 calls respectively. Dynamic S is HTP2. Gemma reuses generation 2;
HTP0/HTP1 remain Qwen generation 1. Loads before and after Gemma are both
1/1/2. Fresh Gemma helper leases are lease-20 through lease-26. Its helper
is recorded as LEARNING/DIAGNOSTIC. The native terminal totals 15,122 calls,
zero reset recoveries, and status 0, including separate historical HTP2
generation-1 Qwen and generation-2 Gemma proofs.

Gemma's logical ATTACHED event is at 287.081395 s in the online epoch;
desktop execution starts at 287.477843 s. Its first positive applied control
is at 312.839968 s after the initial learning window; phone calls begin
at token 35. All 837 requested tokens complete. The envelope preserves
desktop parent `b3b0ee5d24c48e6ec19ebaf7eb360f5bb62691bd9a367117368dabbaa33cedcf`.
Helper operator plan:
`5616da520118fdc4a4d1c883442f3688e1b3b08bba37d7f1f6596d59eb365118`.
Physical Gemma execution proof:
`ecfa9304da8730a5adf77b227a0ef95f53f45b09b2fa86743f9bdd7dc67fb789`.

V4 forward drain-to-quiescence is 1.180344 s. Cold per-session load-to-READY
intervals are 10.774505, 10.730639, and 10.884299 s. First/all readiness
occur 10.774505/46.436289 s after the first load authorization. Desktop
execution precedes phone loading, and HTP0 serves while HTP1 loads.

## Remaining physical blocker

V6 completes cold/online Qwen with 4,962/3,720 calls and no helper
rejection or rematerialization error. Dynamic S is HTP2; it reaches Gemma
generation 2 while retained Qwen HTP0/HTP1 stay at generation 1. Loads are
1/1/2. Each retained session records 54 calls before loading, 114 during,
and 1,668 after. The full native terminal contains 8,682 calls, zero reset
recoveries, and status 0. It preserves HTP2's historical generation-1 proof
and records its new generation-2 identity with zero calls because Gemma
execution was not reached.

The versioned `s42-retained-session-call-gap-v2` check still fails:

| Retained session | Matched median | Worst next-token gap | Ratio | Required |
| --- | ---: | ---: | ---: | ---: |
| HTP0 | 0.5545265 s | 1.138335 s | 2.052805x | <= 2x |
| HTP1 | 0.5563080 s | 1.130085 s | 2.031402x | <= 2x |

There is no missing reference data: each class has 30 equivalent reference
intervals at the same fraction, retained mask, and column width. For the
next-token classes these references are after READY, using v2's declared
reference selection, not 30 preceding calls. All six next-token classes
per retained session fail; all five within-token classes pass. This is not
the old pooled-median assertion, whose historical FAIL remains unchanged.

HTP0's worst interval starts 0.100884 s after phone LOAD_AUTHORIZED and
ends 1.239219 s after it, while WEIGHT_READ is starting. The separate
phone-manager load-to-READY interval is 10.733656 s. The evidence locates
the stall near load start but does not prove which of desktop dispatch,
shared phone memory, HTP, or USB causes it. Further work should correlate
host dispatch and phone load-start timing; changing the bound or repeatedly
rerunning would not resolve that cause.

The shortened-window drain remains functional. Times below share the
online desktop measurement epoch:

| Event | Time |
| --- | ---: |
| Drain requested | 58.090313 s |
| Next safe boundary, token 41 | 58.704958 s |
| Reduced-mask control issued | 59.258092 s |
| Applied ACK and quiescence | 59.978756 s |
| Replacement loading started | 60.463756 s |
| Physical READY acknowledgement | 71.255956 s |
| READY publication | 71.342474 s |

Drain-to-quiescence is 1.888443 s. The six-token old-policy window retains
its 126 calls and energy accounting but is measurement-ineligible.
Reverse replacement and injected physical rollback remain NOT_REACHED in
these runs. Their focused software tests pass; this is not a physical
acceptance claim. V6's safety failure occurs before the deferred coverage
check and has not been bypassed.

After exit, the phone returns to 5,000 Mbps normal USB. GPU use returns to
3,178 MiB with 12,770 MiB free; GDM was not stopped. All 208 deployed files
still match the execution source manifest. Previous results are preserved.

## Parent selection and qualification

The old v2 CPU submission journal was not persisted. Its snapshot shows
414,187,520 available VRAM bytes after reserve while Qwen was resident;
that is evidence of memory pressure, not a reconstructed exact decision.
New SUBMISSION files persist both the ticket and full candidate journal.

V3/v4 select GPU+CPU automatically. CPU is rejected as
HIGHER_CALIBRATION_PRIORITY; no GPU route is forced by the runner.
CPU-only parent derivation, identity rejection, and non-inheritance of GPU
qualification are software-tested, not physically demonstrated by those
GPU+CPU runs.

In v5 the first 100% valid probe consumes 47.266051 estimated fleet J/token
versus the initial baseline's 59.077266, but latency is 0.528518 versus
0.487888 s/token. The 75% probe is also slightly slower at 0.489595 s/token.
No probe passes both LEARNING improvement checks against that baseline.
The scheduler returns to zero assistance at token 204; it is not a physical
fallback or attachment failure. No fraction or qualification was forced.

The gate's weighted coverage is relative to its configured resident column
superset, not all CPU/model FFNs. Gemma's shard covers eight layers and
15,360 of 20,480 columns. Its normalized 100% policy is not whole-model
100% offload. V4 contract-relative coverage is 75% Qwen and 89.6830% Gemma.
These are not energy-saving claims.

## Validation and changed files

170 focused tests pass in 113.480 s, including both replay cases with
unchanged goldens. The full scheduler harness was not rerun. Exact command
and goldens are in [TESTS.json](TESTS.json).

Changed source and test files for this repair:

- `_internal/adaptive_decode_planning.py`
- `_internal/route_generation/compiler.py`
- `_unified/automated_candidates.py`
- `_unified/automated_selection.py`
- `_unified/automated_requests.py`
- `_unified/helper_envelopes.py`
- `campaigns/burstgpt/offline_residency_gate.py`
- `tests/test_adaptive_runtime.py`
- `tests/test_offline_phone_residency.py`
- `tests/test_burstgpt_replay.py`

`research_dev/talks.md` and this report record the work. The physical
transaction/controller, adapters, wire formats, native binaries, and FFN
shard files were not rebuilt or redesigned. No commit, push, PR, GDM stop,
unrelated process termination, memory-check relaxation, or trace run.

The campaign-only corrections use the authoritative selected-session
generation map and defer the unchanged 70% coverage requirement until
after lifecycle/terminal evidence is saved. A coverage miss still fails
the entire gate. Safety and identity checks still fail immediately.
