# S9 Sequential Dynamic Provisioning Results

Date: 2026-07-14.

Verdict: `DYNAMIC_PROVISIONING_MECHANICS_PASS; CAPACITY_UNPROVEN`.

## Scope

This slice adds one sequential, content-addressed provisioning stream to the
phone-PIM worker. It moves one complete Gemma4 GGUF shard, publishes it before
PREPARE, and then executes the existing dense-FFN island. It does not implement
tensor-granular live graph replacement, multiple concurrent downloads,
multi-model eviction, scheduler leases, or llama-server integration.

The worker and host use protocol v3. PREPARE explicitly selects either
`prestaged` or `published_store`; a missing published object cannot fall back to
the worker's `--model` path.

## Storage Invariants

- STAGE_BEGIN binds object bytes, SHA-256, chunk map, manifest hash, ticket,
  route epoch, and target generation.
- The worker reserves the complete logical object with `posix_fallocate` before
  acknowledging the transfer.
- Chunks are ordered. Each accepted chunk is payload-hashed, written with
  `pwrite`, and `fdatasync`ed before the returned prefix digest advances.
- Duplicate chunks are acknowledged only after rereading and hashing the durable
  bytes. Reconnect and process restart reconstruct the contiguous verified prefix.
- Commit performs a full-file SHA-256 and file sync, then a no-replace rename.
  The directory entry is synced before the inode is sealed mode 0400 and synced
  again. This ordering prevents a crash from restoring a read-only `.part`.
  The published object is keyed by its SHA-256.
- PREPARE opens the verified published descriptor and checks its identity while
  reading metadata and tensors. A process-local verified-descriptor cache avoids
  a second full-file hash after an unchanged lookup.
- A storage or publication error blocks READY. If directory durability becomes
  indeterminate, the running store instance fails closed instead of dispatching.

The trusted-lab boundary remains explicit. The protocol has integrity checks but
no peer authentication, and this slice does not claim crash consistency for
filesystems that do not honor the required sync and rename semantics.

## Exact Final Device Runs

Common case: empty private store, 464,114,176-byte shard, SHA-256
`5cfba18d...360d`, 4 MiB chunks, `blk.2`, M=16, seven executions, HTP0. Both
phones used worker binary SHA-256 `0a50ca72...e749`.

| Device | HTP | Stage | Useful goodput | Remote full verify | Prepare | rel-L2 max | HTP compute p50 | Warm e2e p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OP15 | v81 | 94.622 s | 4.678 MiB/s | 17.814 s | 531.144 ms | 2.924e-4 | 12.300 ms | 51.136 ms |
| OP12 | v75 | 29.690 s | 14.908 MiB/s | 2.866 s | 644.251 ms | 2.947e-4 | 16.442 ms | 44.736 ms |

Each run uploaded 111 of 111 chunks, published an exact-size mode-0400 file with
the expected SHA-256, selected `model_source=published_store`, initialized the
requested Qualcomm HTP route, and showed no CPU fallback. The production Gemma4
graph was the correctness oracle.

OP15's seven-sample warm-path CoV was 26.1 percent because of latency outliers;
OP12's was 5.59 percent. Therefore these runs are correctness and mechanics
evidence, not stable latency rows. The raw records are
`artifacts/op15_dynamic_v3.json` and `artifacts/op12_dynamic_v3.json`.

## Recovery And Adversarial Tests

- Host protocol suite: 22 checks.
- Host storage suite: 45 checks.
- Real worker/socket integration suite: 45 checks.
- Android protocol suite: 22 checks on each phone.
- Android storage suite: 45 checks on each phone.
- A CPU end-to-end run was terminated after three durable chunks, restarted,
  resumed at byte 12,582,912, published, prepared, and completed with rel-L2 0.
- A different active object is rejected unless the host explicitly requests
  replacement; explicit replacement aborts the old staging transfer first.
- Tests cover stale generation, corrupt payload, exact prefix ACKs, duplicate
  chunks, mid-frame disconnect, worker SIGKILL/restart, hidden partial objects,
  corrupt finals, symlink/hard-link/FIFO rejection, published-file mutation, and
  rejection of a cache hit that would replace a different active transfer.

The tests do not inject every kernel/filesystem syscall failure. In particular,
there is no fault-injection matrix for short `pwrite`, `fdatasync`, file `fsync`,
rename, directory `fsync`, or process death at every instruction boundary.

## Performance Finding

The mechanics pass, but this transport is not the path to a capacity result. The
sequential framed stream reached only 4.7-14.9 MiB/s in the exact-final clean
runs, versus the separately measured staged `adb push` rates of about 262 MiB/s
on OP15 and 216 MiB/s on OP12. The current path sends one request/ACK per 4 MiB
chunk and performs durable sync work before every ACK; hashing, copied buffers,
ADB-forwarded TCP, and UFS work are not pipelined.

The phones enumerate at 5 Gbit/s on separate USB domains, so this result does not
show a physical USB limit. It shows a software-path limit. Do not use the 13-15
MiB/s range or the OP15 cold-path outlier as the S9 link profile, and do not infer
useful lookahead, capacity, latency, or energy benefit from them.

## Next Gate

Keep the protocol and storage invariants, but replace the stop-and-wait data path
with a bounded native bulk path and measure each stage independently:

1. H2D transport into a preallocated staging buffer/file without ADB shell or
   per-chunk round trips in the timed data path;
2. UFS write plus durability cost, with staged-file and direct-stream modes kept
   distinct so bytes are not double-counted;
3. SHA-256 cost, backend materialization, warmup, and D2H result cost;
4. pipeline depth and credit limits that preserve the durable-prefix invariant;
5. simultaneous OP12/OP15 streams on their separate contention domains; and
6. only then, a forecast threshold for when queued reuse amortizes provisioning.

No live two-level scheduler work is authorized by this result alone.
