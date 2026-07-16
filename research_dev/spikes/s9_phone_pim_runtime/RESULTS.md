# S9 Phone PIM-Style Runtime Results

Date: 2026-07-14.

Current verdict: `DYNAMIC_PROVISIONING_MECHANICS_PASS; CAPACITY_UNPROVEN`.

## Current Dynamic V3 Result

Exact-final case: empty private store, 464,114,176-byte Gemma4 shard, 111 x 4
MiB chunks, `blk.2`, M=16, seven executions, HTP0. Both phones ran worker
SHA-256 `0a50ca72...e749` and explicitly prepared from `published_store`.

| Device | HTP | Stage | Useful goodput | Full verify | Prepare | rel-L2 max | Compute p50 | Warm e2e p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OP15 | v81 | 94.622 s | 4.678 MiB/s | 17.814 s | 531.144 ms | 2.924e-4 | 12.300 ms | 51.136 ms |
| OP12 | v75 | 29.690 s | 14.908 MiB/s | 2.866 s | 644.251 ms | 2.947e-4 | 16.442 ms | 44.736 ms |

Both runs uploaded all 111 chunks, published an exact-size mode-0400 object with
SHA-256 `5cfba18d...360d`, executed on the requested HTP backend, and showed no
CPU fallback. Release and shutdown completed. These are mechanics/correctness
results, not stable latency or capacity rows: warm-path CoV was 26.1 percent on
OP15 and 5.59 percent on OP12.

Verification passed: 22 protocol checks, 45 storage checks, 45 real-process
integration checks, release CTest 3/3, ASan/UBSan CTest 3/3, and Android
protocol/storage suites on both phones. Exact JSON records are
`artifacts/op15_dynamic_v3.json` and `artifacts/op12_dynamic_v3.json`.

The dynamic path is currently software-limited at 4.7-14.9 MiB/s, far below the
separately measured staged `adb push` rates. The next gate is bounded pipelined
bulk transfer, not scheduler integration. Full invariants, recovery tests,
limitations, and next steps are in `DYNAMIC_RESULTS.md`.

## Historical Pre-Staged V2 Result

Date: 2026-07-14. Verdict: `PRESTAGED_FFN_MECHANICS_PASS`.

## Implemented

- `examples/phone-pim/`: protocol/socket library, persistent dense-FFN island,
  phone worker, host driver, production-graph oracle, and protocol tests.
- Capability: `prestaged_gemma4_dense_ffn_v2` for one dense Gemma4 layer.
- Operations: RMS norm, gate/up projections, GEGLU, down projection,
  post-FFW RMS norm, and residual add.
- Commands: HELLO, PREPARE, EXECUTE, STATUS, RELEASE, PING, CLOSE, SHUTDOWN,
  and ERROR.

The worker opens the shard once, hashes that descriptor, parses it through
`/proc/self/fd`, reads all five weights from the same inode, verifies it did not
change, uploads once, and reuses the graph. The host alternates two independent
production-graph inputs.

## Device Results

Common case: `blk.2`, Gemma4 12B F16 shard, M=16, seven executions, HTP0.

| Device | HTP | rel-L2 max | Prepare | Verify | Warm e2e p50 | Warm e2e p95 | HTP compute p50 | Protocol/copies p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OP15 | v81 | 2.924e-4 | 24030.5 ms | 22048.7 ms | 49.044 ms | 53.918 ms | 12.451 ms | 36.492 ms |
| OP12 | v75 | 2.947e-4 | 3280.1 ms | 2783.0 ms | 45.037 ms | 48.648 ms | 16.402 ms | 28.649 ms |

Both routes stayed on HTP0 and were below the 5e-3 correctness gate. The resident
backend allocation was 341.99 MiB on both devices. CPU loopback also passed with
relative L2 exactly zero.

These are one-process prototype measurements. OP15's 22-second full-file SHA scan
is an observed cold-path outlier that needs separate UFS/page-cache profiling.
The 29-36 ms non-compute component includes F32 serialization, copies, forwarded
TCP/ADB behavior, and protocol work; it is the immediate warm-path optimization
target. No latency, throughput, or capacity gate is claimed.

## Failure Tests

- Corrupt same-size shard: PREPARE rejected by full-file SHA-256.
- RELEASE: state cleared and generation advanced.
- Reconnect with old generation: rejected as stale.
- Reconnect with new generation: PREPARE and EXECUTE passed.
- Backend failure path: state clears before generation advance.
- Initial generation `UINT64_MAX`: rejected.
- Non-loopback worker bind: rejected.
- Worker restart: a fresh random session epoch prevents replay of an old frame.
- Host oracle failure: forces RELEASE before returning failure.

## Verification

- Host build: worker, host, and test targets pass.
- Host CTest: `phone-pim-protocol` passes.
- Android cross-build: worker and protocol test pass.
- OP12 protocol test: 20/20.
- OP15 protocol test: 20/20.
- Independent post-fix code review: no blocker for the declared trusted-localhost,
  pre-staged prototype scope.

## Honest Boundary

This proves that command-driven resident execution is possible on both phones. It
does not yet prove the general S9 scheduler. Request serialization still allocates,
the file is pre-staged rather than streamed, the route epoch is a startup value,
and there is no live READY/lease/credit service. Energy remains deferred.
