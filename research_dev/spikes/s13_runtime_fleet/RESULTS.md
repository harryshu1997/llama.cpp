# S13 Live Phone Fleet Results

## Verdict

```text
REAL_TWO_PHONE_FFN_RUNTIME_MECHANICS_PASS
PERSISTENT_SESSION_WORK_STEALING_PASS
MIXED_MODEL_SERVER_INTEGRATION_NOT_IMPLEMENTED
LATENCY_CAPACITY_ENERGY_NOT_CLAIMED
```

## What Ran

The compiled `llama-phone-pim-fleet` target ran one real Gemma4 dense FFN
island (`blk.2`, M=16) on OP12 HTP0 and OP15 HTP0. Both phones used the exact
464,114,176-byte shard with SHA-256:

```text
5cfba18d2a47acc190f317d650895bcc53e914e9a0bc61631860be9591ed360d
```

Before the fleet run, the existing `llama-phone-pim-host` independently passed
against the production llama.cpp callback oracle on each phone. The retained
records are `../s10_power_frontier/artifacts/cp0_ffn_op12.json` and
`../s10_power_frontier/artifacts/cp0_ffn_op15.json`:

| Device | Verdict | Repetitions | Max relative L2 | Phone compute p50 |
|---|---|---:|---:|---:|
| OP12 | `PRESTAGED_FFN_PASS` | 7 | 0.0002947 | 16.396 ms |
| OP15 | `PRESTAGED_FFN_PASS` | 7 | 0.0002924 | 12.165 ms |

The fleet runner then used a local CPU instance of the same FFN graph as the
per-job oracle. This is an execution/correctness harness, not a production
server result.

## Cold Residency

Both phones began non-READY at generation 7. PREPARE verified and loaded the
resident island before dispatch.

| Metric | Fleet | OP12 | OP15 |
|---|---:|---:|---:|
| Remote setup / PREPARE | 18620.807 ms | 3366.403 ms | 18570.962 ms |
| Jobs assigned | 8 | 4 | 4 |
| Last RPC complete | 222.064 ms | - | - |
| Validated dispatch makespan | 222.221 ms | - | - |
| Client operation p50 | - | 44.167 ms | 55.458 ms |
| Worker-reported compute p50 | - | 16.046 ms | 14.830 ms |
| Max relative L2 | 0.0003044 | 0.0002769 | 0.0003044 |

The cold setup dominates. It is reported separately and receives no latency or
energy credit. Keeping the verified island resident is required for useful
steady-state scheduling. OP15's 18.6-second load in this acquisition is a
transient observation, not a stable cold-load estimate.

## Warm Residency

A second invocation deliberately supplied stale generation hint 999. STATUS
reported authoritative generation 7, both workers were already READY, and the
idempotent PREPARE check took about 2 ms per device.

| Metric | Fleet | OP12 | OP15 |
|---|---:|---:|---:|
| Remote fleet setup | 53.790 ms | - | - |
| PREPARE E2E | - | 1.851 ms | 3.708 ms |
| Jobs assigned | 16 | 9 | 7 |
| Last RPC complete | 407.878 ms | - | - |
| Validated dispatch makespan | 408.023 ms | - | - |
| Client operation p50 | - | 45.410 ms | 53.646 ms |
| Worker-reported compute p50 | - | 16.062 ms | 15.098 ms |
| Max relative L2 | 0.0003044 | 0.0002943 | 0.0003044 |

Assignment is completion driven. Runs with different transient transport
latency produced different splits; the final cold and warm records are 4/4 and
9/7. This demonstrates that the
queue is live and does not use a frozen simulated rate. It does not establish a
stable capacity ratio; the sample count is too small and the ADB activation
path is not the intended WiFi/USB split.

Each relative-L2 comparison runs inline before its device claims another job.
The validated dispatch makespan includes that harness-only work.
`last_rpc_complete_ms` excludes the final comparison but does not remove the
earlier comparisons from queue pacing. Neither value is a production latency.

## Verification

```text
release build: phone-pim protocol/client/CLI/store/stream 6/6 PASS
ASan/UBSan:    phone-pim protocol/client/CLI/store/stream 6/6 PASS
-Wall -Wextra -Wpedantic -Werror: fleet and client targets PASS
real devices: OP12 + OP15 REAL_FLEET_FFN_PASS
```

The client tests cover stale generation discovery, mutation-before-STATUS,
every correlated response-header field, inconsistent STATUS residency,
PREPARE identity publication, exact PREPARE/EXECUTE/RELEASE sequencing,
overflowing or impossible worker timing, exact generation advance, lossless
uint64 JSON identity, and typed remote errors.
Worker generation advances on RELEASE and a later stale hint is ignored in
favor of the current STATUS payload.

The final cold/warm records use schema v2. Protocol uint64 identities are
decimal strings, so JSON consumers cannot round adjacent session epochs.

Raw cold/warm JSON, exact commands, binary/content hashes, ADB mappings, USB
topology, and worker logs are retained in `artifacts/` and `MANIFEST.md`.

## Honest Boundary

- This is real llama.cpp/ggml execution on both phones, not S12 simulation.
- It is one FFN operator island from one model, not mixed-model inference.
- Local model hashing plus CPU-oracle preparation and job execution are outside
  remote setup and dispatch, and are reported as separate fields.
- Per-result CPU comparison is inside validated dispatch and is disclosed
  separately from worker-reported compute.
- No A6000 work is removed, batched, or power-shaped by this harness.
- No request-level server fallback exists yet.
- Activations currently use ADB TCP forwarding rather than separate WiFi H2P
  and USB P2H paths.
- Physical serials are caller assertions cross-checked by the manifest; protocol
  v3 does not bind a serial or worker binary digest into HELLO.
- No latency benefit, server capacity relief, HBM relief, or energy saving is
  established by these rows.
