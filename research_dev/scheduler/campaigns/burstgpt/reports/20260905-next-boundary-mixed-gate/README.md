# Next-boundary drain: bounded mixed-session gate v2

The drain-window fix passes physically. The complete gate remains FAIL:
Gemma cannot attach to its selected desktop parent, so reverse replacement
and physical rollback were not reached. There was one physical rerun, no
trace, no A/B comparison, and no energy-saving claim.

## Drain timing

All following timestamps use the online desktop measurement epoch. Relative
times start at REBIND_DRAIN_POLICY_BOUND, not at the physical weight load.

| Event | Epoch-relative seconds | Seconds after drain request |
| --- | ---: | ---: |
| Drain requested | 65.048335 | 0.000000 |
| Next safe boundary observed, token 41 | 65.327678 | 0.279343 |
| Reduced-mask control issued | 65.841096 | 0.792761 |
| Applied ACK and quiescence confirmed | 66.362939 | 1.314604 |
| Replacement physical transaction started | 66.851201 | 1.802866 |
| Physical READY receipt returned | 82.247667 | 17.199332 |
| Verified READY published | 82.354271 | 17.305936 |

Drain-to-quiescence decreases from 14.221488 s to 1.314604 s, removing
12.906884 s of delay. The old-policy window ends after six tokens rather
than waiting for its normal 32-token minimum. Its 126 phone calls, energy,
and timing remain accounted, with `measurement_eligible=false`.

The retained mask is HTP0/HTP1, layers 0-11, at 100% of the configured FFN
columns. The same drain-policy hash appears in the control and quiescence
events; native `FFNCONTROL` confirms generation 2 at token 43 with mask 4095
and 17,408 columns. HTP2's final generation-1 call completes 727,204 us before the phone
manager authorizes its replacement load. No removed-session call crosses
that load authorization.

Physical manager LOAD_AUTHORIZED-to-READY is separately **15.356244 s**,
versus 9.665491 s in v1. The desktop receipt spans 15.396466 s. These are not
the drain-window delay; this run does not establish a load-time improvement.

## Retained service and versioned interruption metric

Metric `s42-retained-session-call-gap-v2` was deployed before execution.
It compares exact within-token layer pairs and same-layer next-token
intervals at identical fraction, layer mask, column width, and batch.
All 11 classes per retained session have 30 equivalent reference intervals.
For this forward transition, within-token classes use one preceding and
29 post-READY references; same-layer classes use 30 post-READY references.
No reference collection delays the drain. Exact reference and transition
rows are in `FORWARD_GAPS_V2.json`.

| Retained session | Generation | Calls before / during / after load | Worst gap / matched median |
| --- | ---: | ---: | ---: |
| HTP0 | 1 | 54 / 168 / 1,614 | 1.36x, PASS under 2x |
| HTP1 | 1 | 54 / 168 / 1,614 | 1.51x, PASS under 2x |

The original v1 pooled-median gate stays FAIL. Its artifact SHA-256 remains
`feb6c563db493ca1f21a2f65f23b74d1c9d8d47abe15713d1a790b9241e2ea38`.
The v2 run also retains its own pooled-v1 calculations as FAIL diagnostics.
Neither original result was overwritten or relabelled.

Cold Qwen completes with 4,962 calls; online Qwen completes with 3,720.
Online per-session calls are 1,836 / 1,836 / 48. Only the dynamically
selected HTP2 changes, Qwen/gen1 -> Gemma/gen2. Final generations and load
counts are both 1 / 1 / 2. Retained physical records are byte-equivalent to
their source records, not reloaded or republished.

The recorded authorization equals the authorization recomputed from the
source and final physical maps. Its assignment hash is
`sha256:842aa5ab43727646360a00eb7cbf16482ed198643f0ec7fd00b618366e393747`.
The terminal contains 8,682 calls, zero reset recoveries, and status 0.
Generation-keyed historical Qwen proofs remain present, including replaced
HTP2/gen1. Completed Qwen requests have no fallback recoveries.

Cold per-session load-to-READY times are 11.591032 / 11.404534 / 18.708996 s.
Relative to the first load authorization, first readiness is 11.591032 s
and all-session readiness is 53.305628 s. Desktop restart/reuse and the
0/25/50/75/100% fraction sweep pass without a phone endpoint restart or a
fraction-induced reload. These times include the existing staged test's
inter-stage serving checks, not just raw file transfer.

## Why the complete gate fails

Gemma's READY preparation envelope uses GPU+CPU parent:
`sha256:b3b0ee5d24c48e6ec19ebaf7eb360f5bb62691bd9a367117368dabbaa33cedcf`.
Its request instead selects `physical:cold:cpu`, with parent:
`sha256:511e007be85beca63f12cdacd441f8057a54d73d43b0bb326136587bc3141e7a`.
There is no exact reusable helper for that actual parent. The resolver
correctly refuses the mismatched envelope; the gate records 62
`HELPER_REMATERIALIZATION_FAILED` events and zero Gemma phone calls.
The ATTACHED wait expires after 300 s, and normal gate cleanup interrupts
the unfinished Gemma request at 371/837 output tokens.

This is a distinct parent-compatible helper availability problem, not a
generation, eviction-source, or drain-window failure. No parent identity
check was relaxed and no runner route was forced. Reverse replacement,
fault injection, rollback, and reverse retry remain unproven by this run.

One additional existing measurement defect was found but not changed:
the gate's final attachment-latency assertion subtracts execution start
from a logical ATTACHED event that can occur before execution. Here those
timestamps are 29.116378 s and 27.060584 s, respectively. It must distinguish
logical authorization from physical applied-control time; it was not reached
in this run and did not cause the recorded failure.

## Validation and changes

143 focused helper/adaptive/adapter/gate tests pass, plus the replay test
covering both saved cases twice: 144 tests total. Both replay goldens remain
unchanged. Exact commands and hashes are in `TESTS.json`. No full-harness run.

Files changed for this measurement stage:

- `campaigns/burstgpt/offline_residency_gate.py`: versioned matched-class
  measurement, direct drain timeline, and failure evidence persistence.
- `adapters/http_backend.py`: direct safe-boundary and control-issue timestamps.
- `campaigns/burstgpt/INTERRUPTION_METRIC.md`: versioned metric definition.
- `tests/test_burstgpt_replay.py`: six measurement regressions.
- `tests/test_llama_server_adapter.py`: timing-event assertions.
- `research_dev/talks.md` and this report directory.

The previously tested drain fix in `_internal/adaptive_decode.py` is carried
forward unchanged. No native binary, shard, transport protocol, helper
authorization policy, memory limit, or session transaction was redesigned.

## Artifacts and cleanup

Run: `/home/zhihao/s42-ffn-mixed-session-20260905-v2-gate/run`.
Inputs: `/home/zhihao/s42-ffn-mixed-session-20260905-v2-inputs`.
Deploy: `/home/zhihao/s42-ffn-mixed-session-20260905-v2-deploy`.
Commands, preflight, source manifest, and pulled phone terminal logs are in
the fresh inputs directory. `PARTIAL_GATE_AUDIT.json` records their hashes.

- FAILURE.json: `6d2d330412a5154f8845ceba16bb085c79f1e05764863a597b83006f686684a3`.
- DRAIN_TIMELINE.json: `5c113ef703b92387d93a4d0cb320104d7af40574d65cf147da128c8aaa0a23d7`.
- SOURCE_MANIFEST.json: `dfe53559f7ee02ccfdb8002f4688d68e34d5baf2ab7d5ca66e53fcf1fbab8e66`.

Normal USB is restored at 5,000 Mbps. Gate-owned desktop processes exited;
GDM is untouched and VRAM is back to 12,770 MiB free / 3,178 MiB used.
No second rerun, trace, commit, push, or prior-artifact overwrite.
