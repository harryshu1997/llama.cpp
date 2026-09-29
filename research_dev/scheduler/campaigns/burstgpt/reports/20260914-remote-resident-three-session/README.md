# Three-session FFN weight relocation

Status: PASS, one bounded physical gate. Three full-parent and three
remote-parent requests completed. Native terminal status 0; clean postflight.

This bounded follow-on extends the existing one-session phone-owner gate to the
existing fixed-residency configuration. It does not change native loaders,
workers, shard formats, adaptive policy or session lifecycle. The scheduler
still packs and admits the declared assignment and the canonical preloader
publishes each verified session. No long trace, flash, reboot or process
interference is authorized.

Target: the three existing full-width Gemma FFN shards for layers 0-23. Their
tensor payload is 8,493,465,600 bytes, subject to the existing phone memory,
workspace, desktop-parent and per-session checks. Only measured server omission
and physical execution count as relocation. Host RAM and GPU VRAM are reported
separately. Larger context and energy savings are not assumed from capacity.

The comparison retains the previous three short requests, model, native
binaries, desktop placement and decoding settings. Explicit semantic-sanity
mode records token differences without treating them alone as failure. Output
completion, semantic sanity, ownership, generations, memory and terminal proofs
remain strict. Original exact-output failures and all previous artifacts stay
unchanged. No in-flight owner cancellation is attempted.

## Result: three times the demonstrated weight relocation

| Measurement | Full desktop parent | Three-session remote parent |
| --- | ---: | ---: |
| Server model-file mappings, bytes | 13,811,417,088 | 5,318,246,400 |
| Server model-file RSS, bytes | 13,811,417,088 | 5,318,148,096 |
| Server process RSS, bytes | 15,269,101,568 | 6,983,045,120 |
| Server process GPU memory, bytes | 12,654,215,168 | 12,654,215,168 |
| Request-attributed phone FFN calls | 0 | 1,656 |

The native omission proof covers 72 gate/up/down tensors, 8,493,465,600 tensor
bytes. It unmaps 8,493,170,688 complete-page bytes with zero remaining VMA
overlap and validates warm-up through the phone. The process RSS reduction is
8,286,056,448 bytes; non-weight allocations explain why RSS is not identical to
the weight-page difference. GPU VRAM receives no memory credit.

Each existing indexed shard contains 2,831,155,200 tensor bytes. File sizes with
metadata are HTP0 2,831,157,472; HTP1/HTP2 2,831,157,504 each. The authorized
phone weights plus 805,306,368-byte workspace allowance total 9,298,771,968
bytes under the unchanged 10,000,000,000-byte pool. This allowance is a
reservation, not a measured phone allocation high-water mark. No whole-model
OpenCL service is loaded by this gate.

The exact parent remains 23 GPU layers, context 2560, batch 2048, ubatch 512,
parallel 1, default CUDA graphs, same F16 artifact and native libraries. Layer
placements come from the selected parent, and the gate rejects non-CPU FFNs.
The phone computes full FFN width for layers 0-23 during prefill and decode;
attention, KV and all other operations remain on the desktop. This is not a
100% offload of the whole model or a proof of increased usable context.

## Session readiness, calls and reuse

| Session | FFN layers | Native authorization to READY | Scheduler stage start to verified | Request calls |
| --- | --- | ---: | ---: | ---: |
| HTP0 | 0-7 | 10.881 s | 20.339 s | 552 |
| HTP1 | 8-15 | 9.431 s | 9.675 s | 552 |
| HTP2 | 16-23 | 9.884 s | 10.351 s | 552 |

Scheduler post-verification completes at 20.339, 30.661 and 41.015 seconds from
the first scheduler stage start. The outer measured preload call lasts
41.024811 seconds. Native read/init/upload/verification/READY timestamps and
host receipt/verification timestamps are preserved separately in
`physical/SUMMARY_V2.json`; no phone/host monotonic clocks are subtracted.
`SUMMARY.json` remains preserved, but its publication-labelled fields were
receipt-ready timestamps. V2 explicitly distinguishes the subsequent scheduler
verification, approximately 0.09-0.19 seconds later. No gate data was changed.

Each session loads once, keeps generation 1 across all requests, and has its own
artifact, geometry, operator plan and generation-keyed proof. Requests 42/46/50
produce 288/360/1008 calls (96/120/336 on each session). Native total 1,752 adds
96 warm-up calls. Each request receives a distinct ticket and fresh lease
tokens. The audit checks every owner against physical READY and execution
proofs. No subsequent shard reload, fallback, stale generation, reset recovery
or internal endpoint restart was observed. Normal final cleanup restores
Android USB; same-boot idle postflight passes.

This test proves progressive publication and later reuse of all three owners.
It does not test concurrent inference during their loading or owner loss:
the reduced parent requires its complete declared owner set before launch.
Semantic-sanity mode passes; two outputs are token-identical and request 50
differs first at token 37. This difference remains recorded, per the user's
explicit experimental acceptance. Historical exact-mode FAILs are unchanged.

## Energy and speed: diagnostic, not an end-to-end savings claim

The existing warm-request analyzer excludes request 42 because its reduced-arm
interval includes desktop loading. Requests 46 and 50 have identical input and
output counts (55 output tokens total) and no load transitions in either warm
request interval. Both use the same modified graph-enabled binary and parent
placement. This is one sample of two short requests, not a frozen matched
performance protocol or route qualification.

| Assumed active phone power | Desktop fleet energy | Remote-parent fleet energy | Diagnostic reduction |
| --- | ---: | ---: | ---: |
| 3 W | 3.351 kJ | 1.665 kJ | 50.33% |
| 4.5 W | 3.351 kJ | 1.714 kJ | 48.87% |
| 6 W | 3.351 kJ | 1.763 kJ | 47.40% |

CPU package and GPU board energy are physically sampled. This gate persists
their interval aggregates and receipts, not the sampler's raw power time
series. A matched performance gate must retain those raw samples for audit.
The desktop arm
includes phone idle at 0.875 W. The remote arm conservatively charges assumed
phone active power throughout each request. These rows exclude preload and
cleanup and must not be quoted as trace or end-to-end savings.

Warm request wall time is 29.016600 s desktop versus 32.689569 s remote
(12.66% slower). Native server timings explain the direction: decode falls
from 24.02438 to 21.03992 s (12.42% less decode time), but prefill rises from
4.81489 to 10.83775 s (2.25x). Additional wrapper/proof overhead is included
in the wall times. More relocated layers reduced decode energy/time here,
but the current prefill path prevents an overall speedup on these short
outputs. Optimizing prefill is follow-up work, not part of this change.

The entire phone-preparation call separately measures 589.034525 J of server
energy. Charging the phone active for its full 41.024811 s gives 712.109,
773.646 and 835.183 J at 3/4.5/6 W. The individual transition receipts are also
preserved; they exclude inter-stage gaps and must not be added to this outer
preparation measurement. No overlap is subtracted. The full desktop control
load took 62.115482 s, including cold file reads; the subsequent reduced arm
benefits from file cache warmed by the control. That asymmetry and the existing
control/adapter boundaries prevent an honest end-to-end comparison or measured
break-even claim from this gate.

## Implementation and validation

Changed existing files only:

- `campaigns/burstgpt/remote_resident_gate.py`: reuse the existing
  `--fixed-phone-residency-json` assignment for multiple owners; validate exact
  indexed coverage/full width, the packed session assignment, and the selected
  CPU parent; retain legacy one-session defaults. Account for each actual READY
  session rather than attributing the entire remote set to the anchor session.
- `tests/test_remote_resident_gate.py`: six focused regressions for multi-owner
  input, legacy behavior, missing/incompatible shards, packed identity and
  authoritative per-session memory accounting, including distinct generations.

No loader, worker, shard format, session state machine, adaptive policy or
qualification code changed. The gate remains an explicit fixed-residency,
calibration-mode experiment, not a policy-driven trace. The prior working
deployment was copied into a fresh directory, and only the two files above
were overlaid. Physical safety checks and the existing execution lock remained
enabled. A shell launch used unavailable `python` once before preflight; it was
corrected to `python3` without any physical action. Preflight and the physical
gate then each ran once and passed. No inference retry was needed.

63 focused tests PASS in 21.159 s. Both replay tests PASS in 77.693 s, with
unchanged goldens. No broad suite, long trace, native rebuild, shard generation,
reboot, flash, GDM change, unrelated process intervention, commit or push.

Replay goldens:

- COW v3: `sha256:5d52e8673fc60f974ca167964b3ad579e59abea28feb3d66357cf39c1ca7caf4`.
- Sparse v8: `sha256:241924463ac27ead0c2d7bcb5da214e85097049c91d5800848e0b29effd2b917`.

## Immutable artifacts and reproduction

Remote root: `/mnt/storage/s42-remote-resident-three-session-20260914-v1-2C2ipE/`.
The local `physical/` copy includes the resolved commands, pre/postflight,
gate results, snapshots, energy aggregates/receipts, native logs and summary.
`EXPERIMENT.json` is the frozen input assignment; `run_gate.py` invokes the
canonical gate under the existing hardware lock. It has no scheduling policy.

```sh
python summarize_gate.py physical/gate-run-v1 --output SUMMARY_FRESH.json
python ../20260914-relocation-performance/analyze_hot_requests.py physical/gate-run-v1 --output HOT_FRESH.json
```

- Gate result SHA-256: `3cc6c3bab98beeb13cec70c9ad09d212ca2fe8b4a9a8819584678011ca3afe6e`.
- Summary V2 SHA-256: `34164a42af434eb76e16f4e97857e0386d265350ea94dcbbf29aed9d38917b2a`.
- Warm diagnostic SHA-256: `2f4ac8e42f6a6ac8bf4641816764558acaec891e42c45d4044c6b1656c975d0a`.
- Deployment archive SHA-256: `81ea20b45d0d0231394b52159fa2c93b8888d660036ee2fc5861e68250984a9f`.
- Before-image archive SHA-256: `8fbadaebb02049385c76a563a741516e2f38ce45509ce855df96d3a386620666`.
- Assignment SHA-256: `8fe570045e06285bc7a22b59809a7a1f51c6778ba2995cdc1b39f6f60ac4e9da`.
- Native server SHA-256: `00220ffd27aa167752de240d82cdf1062104ec087ce62c388e2e193b8b1d974a`.
- Parent model SHA-256: `ed76f2183d2d1d65091986033023e6c78d27f6276c1b0c5826cc92acf73538cf`.
- Shard index SHA-256: `5d0fc13e8be1875269a6c8ef2eeaaded8c968ecf262703de529bdbc71ba13681`.
- HTP0 shard SHA-256: `ef2efcce04b08ee78b7a232d7d5f33f62ccebbee9add93669c5100e97ae041a0`.
- HTP1 shard SHA-256: `e0760134bb39ad9a57f3adc389008c38b75bd6cfcac7cb2c08e2ee532fd0dc11`.
- HTP2 shard SHA-256: `52881334dbc7b55eba81aabb96ec95c47a82a472c8daa3d650bab2f70da7f7b0`.

Next: a per-pool admitted real long-prompt/decode capacity test. Server startup
alone is not context-capacity proof, and the unused GPU pool must independently
fit its growing KV/workspace. End-to-end energy and prefill performance need a
separately frozen matched measurement; no further physical run was launched.
