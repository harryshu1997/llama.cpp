# S9-V1A: Dynamic Provisioning Transport - Profile and Improve

Goal: explain where the sequential dynamic-provisioning wall time goes, then, only if the
profile shows headroom, add a bounded pipelined host data path that improves median
provisioning goodput >= 1.20x on both phones WITHOUT weakening any correctness, durability,
resume, or publication invariant. Do NOT start the mixed-workload scheduler.

## Certified baseline (reproduced at CP0, 2026-07-14)

Protocol v3, sequential (stop-and-wait) provisioning. One request/ACK per 4 MiB chunk; each
chunk is hashed + `pwrite` + `fdatasync`ed before its durable-prefix ACK.

- Host CTest: release 3/3, ASan/UBSan 3/3; integration (stream) 45/0.
- Android on BOTH phones: protocol 22/0, store 45/0.
- Android worker SHA-256 `0a50ca72...e749` (matches the baseline exactly).
- Exact-final 464,114,176-byte shard, 111 x 4 MiB chunks, published_store:
  - OP15 (v81): 4.678 MiB/s, stage wall 94.6 s.
  - OP12 (v75): 14.908 MiB/s, stage wall 29.7 s.
- Separately measured staged `adb push` control: ~262 (OP15) / ~216 (OP12) MiB/s.
- Bottleneck is software, not the 5 Gbit/s USB link.

Raw records: `research_dev/spikes/s9_phone_pim_runtime/artifacts/op1{5,2}_dynamic_v3.json`.

### What the existing worker metrics already show (from the raw records)

The worker already reports per-stage remote costs. Summed (non-overlapping at window=1):

| device | wall | remote chunk-hash | remote write | remote fdatasync | remote full-verify | remote-accounted | UNACCOUNTED |
|---|---:|---:|---:|---:|---:|---:|---:|
| OP15 | 94.6 s | 21.4 s | 0.9 s | 1.5 s | 17.8 s | 41.6 s (44%) | 53.0 s (56%) |
| OP12 | 29.7 s | 4.7 s | 0.3 s | 0.6 s | 2.9 s | 8.5 s (29%) | 21.2 s (71%) |

The unaccounted 21-53 s is host-send + wire + worker-recv + ACK-return round-trip idle - the
stop-and-wait cost that the worker metrics cannot see. CP1 must measure it explicitly.

## Invariants to preserve (non-negotiable)

- ACK only a durable, manifest-verified contiguous prefix.
- Resume reconstructs and verifies the prefix; sends only bytes after it.
- Rename is directory-fsynced before chmod(0400), then file and directory synced again.
- A cache hit cannot replace a different active transfer.
- PREPARE(published_store) never falls back to --model.
- SHA-256, full-object verification, and publication ordering are never weakened.
- Freeze protocol v3 evidence. A wire-format change is protocol v4, not a silent v3 change.

## Checkpoint 1 - explain the missing time (measurement-only)

Add a measurement-only path (prefer a separate bench mode/binary over touching the production
STAGE handler). Measure separately, per stage, with p50/p95/CoV and >=5 rotated reps:

- host: file read, chunk hashing, blocked send time, ACK wait time;
- worker: socket receive time, outer frame SHA-256, manifest chunk SHA-256, pwrite, fdatasync;
- commit: full-file SHA-256, file fsync, rename, directory fsync;
- D2H memory-source transfer; H2D memory-sink transfer (no UFS, no publication);
- `adb push` staged-file control.

Gate: for window=1, the non-overlapped accounting explains >= 90% of wall time. Do not sum
overlapping stages.

Decision input for CP2: the memory-sink (pure H2D, no UFS/publish) result. If memory-sink wall
is much lower than full-stage wall, there is pipelining headroom; if memory-sink is already
near full-stage wall, the bottleneck is durability/hashing and windowing will not help.

## Checkpoint 2 - bounded windowed streaming (only if headroom)

- `--stage-window N`, N in {1,2,4,8}, default 1; window=1 behaviorally identical.
- Decouple request send from response receive; keep request IDs, command seq, ACK validation exact.
- Cap total outstanding payload at 64 MiB; retain each expected cumulative prefix digest until
  its ACK; on error close + recover from the worker's verified prefix (never guess in-flight).
- No worker concurrency initially: first test whether queued frames alone remove round-trip idle.
- Wire change (if any) -> protocol v4.

Required adversarial tests, measurement matrix (64 MiB windows 1/2/4/8 x>=5, 256 MiB best-two
x>=3, per phone, then simultaneous, one full shard, and a blk.2 M=16 repeat=7 HTP oracle run),
and the >=1.20x performance gate are enumerated in the task. If the window gate fails, STOP and
use the profile to pick the next candidate (dedup hashing / batched ACKs / adb-sync adapter /
different USB transport) - do not implement those without reporting the failed screen first.

## Status

- [x] CP0 reproduced (host + both devices; SHAs + raw records verified).
- [x] CP1 profiler + accounting: 99.85% (OP12) / 99.80% (OP15) of window=1 wall accounted.
      SHA-256 = 61%/69% of wall (4x redundant); round-trip idle 26%/33%; transport not the bottleneck.
- [x] CP2 windowing implemented + gated: median goodput 2.44x (OP12) / 6.71x (OP15) at window 8,
      both >= 1.20x, 0 correctness failures / 0 wasted bytes over 40 runs; T1/T2/T3 adversarial PASS.

Verdict: S9-V1A_PIPELINED_TRANSPORT_PASS. See RESULTS.md. Bigger levers (SHA de-dup, batched ACKs)
profiled and reported, NOT implemented (need review / protocol v4). No commit/push.

## S9-V1A-R repair (2026-07-15) - see RESULTS_R.md (ASCII)

The V1A profiling was repaired: profiling is now opt-in (default OFF), stage-scoped/per-opcode,
splits host outer-frame SHA from socket send, measures worker prefix-hash, labels hash domains
(envelope+data vs data-only), and NEVER sums host-CPU and phone-CPU timers as a wall fraction. The
V1A "61%/69% SHA", "four hashes", and "26%/33% RTT" claims are WITHDRAWN and replaced with honest,
separated measurements. analyze_sweep.py is fail-closed (18 mutation self-tests); adversarial_tests.py
uses structural JSON, fails closed, persists records, and computes retry/waste. Byte accounting
(attempted/written/durable/wasted/retried), a FAIL_PROVISION record on disconnect, an
envelope-inclusive checked 64 MiB bound, and `--stage-window {1,2,4,8}` restriction were added; the
bench validates sequence numbers + host/device agreement; an ASan/UBSan test exercises the real host
window>1 path.

- [x] CP0-R integrity + pre-edit tests 3/3.
- [x] CP1-R honest accounting: host timeline 99.86% (OP12) / 99.96% (OP15); host_send = frame_sha +
      socket_send (socket write ~75 ms); phone-side timers separate; profiling overhead ~1.00x.
- [x] CP2-R fail-closed gate: OVERALL PASS. OP12 2.61x median / 1.89x conservative; OP15 2.28x / 1.27x.
- [x] CP3-R accounting/bounds; adversarial T1/T2/T3 PASS both phones (retry/waste from persisted records).
- [x] CP4-R matrix (64/256 MiB, full-shard gate, simultaneous), 0 errors, provenance + thermal captured.

Honest labels: full-shard gate PASS (partly DVFS; OP15 thermally throttled at 95 C); contract
INCOMPLETE; capacity UNPROVEN; energy DEFERRED. No commit/push.
