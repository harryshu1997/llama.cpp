# S40 shared warm-tier server status

Status: `OFFLINE_MECHANICS_AND_DESKTOP_SMOKE_PASS_A_ONLY_ACQUISITION_BLOCKED`

## Implemented

- One experimental llama-server controller drives GPU, CPU, and phone
  executors through the same persistent Unix-socket contract.
- The frozen 74-request, two-model trace, policy, token budget, and metrics are
  shared across C1, C2, T1, and T2.
- The physical orchestrator captures exact runtime binaries, private dynamic
  dependencies, commands, source preflight, executor evidence, controller
  events, resource samples, selected-GPU observations, and cleanup results.
- The physical manifest recomputes controller and executor correctness from raw
  records and rejects stale epochs, duplicate publication, missing cleanup,
  wrong placement, source drift, foreign GPU processes, and non-native
  transport.
- A per-run random internal capability is restricted to the router and desktop
  gateways. It is removed after all teardown actions are attempted; phone
  gateways, model children, cache helpers, observers, samplers, and trace
  drivers do not inherit it.
- Physical preflight and independent manifest validation require the exact
  frozen serving envelope. Coordinated plan, manifest, launch-command, alias,
  duplicate, and missing-flag mutations are rejected.
- The v3 primary campaign fixes 13 prospectively ordered runs by default:
  three C1 warm, three C1 cold, three C2 warm, three T1 warm, and one T2.
  T2 may instead be frozen at three repetitions before acquisition.
- The campaign software lock binds the exact acquisition, validation, and
  reduction sources. All repetitions require the same device stable IDs and
  boot IDs.

## Verified

- The complete S40 Python suite passes 253/253 tests.
- Release and ASan/UBSan builds each pass the two focused CTests:
  `test-server-warm-tier` and `test-warm-tier-executors`.
- The unchanged S39 main suite passes 323/323 tests and its V2.3 readiness
  mechanics suite passes 40/40 tests.
- `git diff --check` passes.

The capability prevents accidental helper use and unauthenticated warm-tier
HTTP calls. It is not a hostile same-UID security boundary because same-UID
processes may still inspect each other on permissive hosts. Separate process
identities or a peer-credential control socket would be required for that
claim.

## Transport result

The obsolete fresh-process Python bridge failed and is forbidden for physical
performance runs. The final persistent native Unix-socket diagnostic on the
exact RTX 4060 Ti reported:

| Fanout | Native P95 | Direct P95 | Increment |
|---|---:|---:|---:|
| 1 | 574,042 ns | 977,131 ns | 0 ns |
| 8 | 1,215,926 ns | 6,128,644 ns | 0 ns |

The transport-v2 record binds the final binary
`795f52ce6b0fd75e4b37da577439068182b8d112de8a89d4abbecb07e464d6be`,
its peer process identity, exact argv, and executor bundle. The frozen evidence
is under `results/final_transport_v2_20260726/`.

## Desktop smoke

Both models passed real B1 and B8 execution through the same native router on
the exact RTX 4060 Ti:

| Model | B1, 8 decode rounds | B8, 8 decode rounds | Result |
|---|---:|---:|---|
| Qwen3-8B Q8_0 | 0.336 s | 0.983 s | PASS |
| Qwen3-14B Q4_K_M | 0.349 s | 1.452 s | PASS |

Each request produced exactly eight continuation tokens. The first request was
token-identical between B1 and B8 for each model. Both child processes used the
bound GPU UUID, then unloaded with no active model or GPU compute process left.
The successful records, prior failed attempts, router logs, and shutdown record
are frozen under `results/desktop_smoke_v2_20260726/`.

## Blockers

1. Finish and freeze the canonical production V2.4 A_ONLY acquisition plan,
   post-reboot identity/network binding, orchestration source closure, and
   no-model preflight.
2. Immediately before acquisition, capture a new phase lock and <=5-second
   identity snapshot for the RTX 4060 Ti, OP15, and OP12.
3. Run and independently validate one complete Qwen3-14B A_ONLY raw bundle.
4. Only after A_ONLY passes, provision the versioned Qwen3-8B phone shards and
   proceed to B_ONLY.

C3 is fail-closed because no role-tagged raw dual-residency profile exists.
C4 is skipped because no second suitable GPU is in scope. Desktop serving
mechanics are established, but no phone-route qualification, service-capacity,
SLO-goodput, or energy result is claimed yet. Phone energy, server-wall energy,
and total-system energy remain unknown.
