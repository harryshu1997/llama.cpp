# S12-V0 Results

## S12-V2 Two-Level Mixed Residency

Verdict:

```text
SYNTHETIC_TWO_LEVEL_MECHANICS_PASS
TWO_MODELS_TWO_PHONES_CAUSAL_PLACEMENT
PHYSICAL_LATENCY_CAPACITY_ENERGY_NOT_CLAIMED
ENERGY_NOT_RUN
```

V2 is a separate replay in `two_level_vq.py`; V0 and V1 remain frozen. The
fixture has two symbolic models, two complete operator islands, OP12, OP15,
one server lane, and seven requests. Synthetic durations exist only to order
events and expose state races.

Verification:

```text
91 unit tests across V0/V1/V2
4 CLI-negative tests
5 PYTHONHASHSEED runs per replay, 15 total
0 failures
V0 replay: sha256:b952f880eec584909f620939fa03a8dc7f20b734a6574fac1b4587f522bca734
V1 replay: sha256:87fea5665e17832dde9bac3554eeab89b89a3a4ef43f0be55d7bcbfb37fc55fa
V2 replay: sha256:2a0ce54e4b91de35a694c5f2278ef30b8c9db4f0844033b01a766899b6458cd9
```

The deterministic fixture produced:

| Policy | Server terminals | Phone terminals | Reject / timeout | Transfer MiB | Useful MiB | Evict MiB | Two-phone overlap us |
|---|---:|---:|---:|---:|---:|---:|---:|
| server only | 5 | 0 | 1 / 1 | 0 | 0 | 0 | 0 |
| static two phone | 3 | 4 | 0 / 0 | 0 | 0 | 0 | 50,000 |
| dynamic two level | 2 | 5 | 0 / 0 | 128 | 128 | 64 | 60,000 |

These numbers are not performance comparisons. The static policy starts with
both synthetic islands resident, while the dynamic policy starts with only A
on OP15. The server-only row reaches the bounded queue and horizon. The table
exists to show different state paths and exact terminal accounting, not a
speedup or admission claim.

The dynamic action sequence is:

```text
0 us:      KEEP A on OP15
0 us:      REPLICATE A to OP12, synthetic score 100000 us
60000 us:  A becomes READY on OP12 and is used
161000 us: DRAIN A on OP12; queue B replacement while A is pinned
190000 us: A tail releases the pin; evict A and begin B prefetch
250000 us: B becomes READY on OP12 and is used
```

The final placement is A on OP15 and B on OP12. Each phone has an independent
64 MiB charge and generation. The reducer refuses stale boot, generation,
status-sequence, content-identity, last-copy, and pre-READY actions. A cold
miss never blocks a free server. Horizon cleanup releases all pins, activation
slots, and prefetch credits without starting new work at the horizon.

The slow loop sees only an immutable current-state snapshot. With two phones,
it exhaustively selects the feasible instantaneous assignment maximizing
`(total_score_us, action_count)`. This is not a temporal optimum. The score is
based on current queued demand and synthetic route durations; it is not a
validated reuse, thermal, or energy predictor.

Residual scope limits are explicit: priority does not change ordering,
deadlines classify outcomes only, `BATCH_TAIL` is singleton, the manifest does
not cryptographically prove which source code executed, and no link fault or
physical profile is represented. READY is a symbolic state with bound model,
weight, backend, boot, generation, and sequence identity, not a measured S9
certificate.

The next gate is a profile adapter populated by persisted measurements for at
least two real operator islands across both phones, followed by replay of the
normalized mixed-workload trace. Live daemon or `llama-server` integration is
still blocked.

## S12-V1 Dual-Path Mechanics

Verdict:

```text
DUAL_PATH_CAUSAL_MECHANICS_PASS
STATIC_HOST_RESIDENCY_ONLY
DYNAMIC_MIXED_POLICY_BLOCKED
SINGLE_PHONE_CONTEXT_NO_CROSS_GROUP_OVERLAP
PATH_RATES_AND_INTERFERENCE_UNMEASURED
RUNTIME_TWO_SOCKET_PATH_NOT_IMPLEMENTED
REAL_PROFILE_COVERAGE_BLOCKED
ENERGY_NOT_RUN
```

The V0 replay and its hash remain unchanged. The versioned V1 replay adds
explicit WiFi H2P, phone compute, USB P2H, and A6000-tail phases with bounded
buffers and exact terminal conservation. It now fails closed on two execution
constraints found during review: host model residency is static for a replay,
and OP15 has one KV-owning context.

Verification after the extension:

```text
57 unit tests
3 CLI-negative tests
5 PYTHONHASHSEED runs per replay, 10 total
0 failures
V0 replay:   sha256:b952f880eec584909f620939fa03a8dc7f20b734a6574fac1b4587f522bca734
V1 replay:   sha256:87fea5665e17832dde9bac3554eeab89b89a3a4ef43f0be55d7bcbfb37fc55fa
```

The synthetic V1 fixture demonstrates only resource mechanics:

| Policy | Host residency | Server done | Phone done | S/P batches | Peak A6000 MiB | WiFi/USB overlap us |
|---|---|---:|---:|---:|---:|---:|
| offline server optimized | FULL_MODEL | 12 | 0 | 5/0 | 25,484 | 0 |
| causal server batch | FULL_MODEL | 12 | 0 | 5/0 | 25,484 | 0 |
| fixed phone | TAIL_ONLY | 0 | 12 | 0/4 | 24,564 | 0 |

`FULL_MODEL` and `TAIL_ONLY` are charged for the entire replay, including idle
periods. Their conservative max-profile delta is 920 MiB. The fixed-phone
result records that delta as residency relief; server-only results record zero
actual relief. V1 rejects the V0 `memory_admission_triggered` policy because no
measured transition can switch the A6000 between these process images.

The scheduler completes a phone request only after four causally ordered
quanta:

```text
prefill:  WiFi input -> phone proxy -> USB result -> A6000 tail
decode 0: WiFi input -> phone proxy -> USB result -> A6000 tail
decode 1: WiFi input -> phone proxy -> USB result -> A6000 tail
decode 2: WiFi input -> phone proxy -> USB result -> A6000 tail
```

Each decode WiFi input is released only by the preceding tail. WiFi/USB overlap
is zero in V1. The current stage has one `llama_context`; a second group would
reset or collide with the first group's KV state. The in-flight group limit is
therefore exactly one. Independent paths are represented, but exploiting them
requires multiple leased KV contexts or explicit spill/restore support.

The table is not a performance result. The fixture assumes 100 MiB/s WiFi H2P,
200 MiB/s USB P2H, and fixed path latencies. More importantly, it reuses S11
`phone_stage_us` as a conservative proxy even though that measurement includes
the old transport transaction. It therefore cannot compare the new topology to
the old route or to server-only latency. The aggregate old-path phone and tail
times are split across quanta by activation-row count solely for this mechanics
test.

The current A0 route's directional data is intentionally asymmetric. At B=8,
the modeled application payload is 1,236 bytes of token/sequence input over
WiFi and 3,809,312 bytes of cut activation over USB. This fixes the V0 ambiguity
where one `activation_bytes` field had no direction.

The next physical gate is a decomposed dual-path profile plus an interference
matrix. The next runtime gate is a paired WiFi-input/USB-result connection with
shared transaction and epoch identity, followed by a measured multi-context or
state-switch mechanism. Until these pass, V1 remains an isolated research
scheduler and the dynamic mixed-route policy remains blocked.

Verdict:

```text
VQ_MECHANICS_PASS
REAL_PROFILE_COVERAGE_BLOCKED
SHAPE_SHADOW_SYNTHETIC_ONLY
ENERGY_NOT_RUN
```

## What Passed

The frozen V0 bounded virtual queue implements all four policies, strict profile
coverage, finite admission, exact terminal accounting, and deterministic
replay. The test result is:

```text
30 unit tests
2 CLI-negative tests
5 independent PYTHONHASHSEED processes
0 failures
replay sha256:b952f880eec584909f620939fa03a8dc7f20b734a6574fac1b4587f522bca734
```

Mutation tests reject duplicate JSON keys, bool-as-integer identity, profile
digest drift, activation-byte drift, duplicate requests, unordered traces,
finite-queue overflow, a phone choice while server admission is feasible, and
a background-HBM increase that oversubscribes an active route. A trace-swap
regression proves parsing and hashing perform one read of the same bytes. A
second regression proves offline peak HBM includes a non-overcommitting rise
inside an active route, not merely the background value at dispatch.

## Strict Coverage

The checked-in varied fixture has 12 distinct request shapes and payload IDs.
Under `strict_real`, all 12 are unprofiled and all four V0 policies conserve 12
terminals without assigning a latency:

```text
claim scope: SYNTHETIC_FIXTURE_PROFILE_COVERAGE_ONLY
profiled:    0
unprofiled: 12
energy:      NOT_RUN
replay:      sha256:dc06b44f60c24ca365c607e236391af0913d71bbba799fafb0485c07e4db075f
```

The pinned real-source audit is stricter still. BurstGPT has only 22 rows with
the same token counts globally and none in the frozen windows; RAGPulse has
zero. Neither source identifies the S11 prompt, Gemma model digest, or context.
Those rows cannot pass the strict identity gate even when token counts match.

Therefore no real trace row can currently receive an S11 server or phone
latency. This is the principal S12 result, not a missing implementation.

This was also executed against the final normalized component artifacts, not
only inferred from source-wide counts:

```text
BurstGPT median: 165 total, 0 profiled, 165 unprofiled
RAGPulse median:  12 total, 0 profiled,  12 unprofiled
```

Both checks used `strict_real` with current S11 artifact verification enabled.
Every row retained `REAL_ARRIVALS_PROFILE_COVERAGE_ONLY`; energy stayed
`NOT_RUN`.

## Synthetic Mechanics

The shape-shadow fixture preserves the 12 bursty timestamps but replaces each
payload with the exact S11 workload. It is labeled
`SYNTHETIC_ONLY_NO_REAL_TRACE_PERFORMANCE_CLAIM`.

The frozen HBM fixture starts with 3500 MiB occupied by other server work and
releases it at 400000 us. A6000 capacity is 26000 MiB.

| Policy | Server done | Phone done | Batches S/P | Queue p95 us | Completion p95 us | Makespan us | Peak A6000 MiB | Activation bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| offline server optimized | 12 | 0 | 3/0 | 400000 | 604341 | 1066957 | 24306 | 0 |
| causal server batch | 12 | 0 | 4/0 | 400000 | 604341 | 1109619 | 24306 | 0 |
| fixed phone | 0 | 12 | 0/4 | 678790 | 1176192 | 1545172 | 25894 | 5713920 |
| memory admission triggered | 10 | 2 | 3/2 | 678790 | 883131 | 1202369 | 25894 | 952320 |

All rows complete 12 requests with zero queue rejection and zero timeout. The
offline result is explicitly clairvoyant. It is not a deployable policy.

The memory policy makes two phone selections. For both, the result record has
`server_admission_feasible=false`; the selected `B=1` phone route fits the
22500 MiB currently available while even the `B=1` full-server route does not.
The charged activation volume is exactly:

```text
2 * 1 * 31 * 3840 * 4 = 952320 bytes
```

This fixture does not show a latency win. Fixed phone is slower, and the causal
memory fallback increases p95 latency relative to waiting for the later HBM
release. It proves only that the scheduler can preserve admission under a
frozen memory constraint without opportunistically choosing the phone while
the server fits.

## Evidence Bindings

The profile binds all five raw S11 plan and summary files plus the S11 plan,
results, manifest, and functional-result digests. The current S12 file hashes
are:

```text
6a72380c98bdeee91fee0b13ce9747722ac70b8641cbcac8c6a7cc824d8ec880  s12lib.py
cf6aa4ced9d88d9dd556bcf0147032ee5174f902e617339cc1658b05e1391806  profile_coverage.py
64475b4de5f4fc4b76ec12e9aafc11faf8e1ce3667a18b1c201a42126ad825b4  policies.py
7369e3e86f1120306123deac14575221ddc6be067a205a604e97d868c91fb97a  vq_sim.py
217f01f4251d7e02dd35df5617e44ceaa867131f81b7b184553da5028d1b5bce  profiles/s11_batched_route.json
035554709f1e5cfb51a7d6a55c141cc538aa036160f083fb46063aa9e9901b23  fixtures/varied_arrivals.jsonl
```

These are code/artifact identities, not proof that the underlying latency
measurements generalize.

## Decision

The virtual-queue foundation is usable for mechanics, but a real-trace system
comparison remains blocked. The next legitimate measurement step is a varied
payload atlas for exact Gemma shapes and at least one second service class,
followed by S8 normalized trace replay.

Do not integrate this simulator into `llama-server` and do not interpret the
shape-shadow table as real workload, capacity, or energy evidence.
