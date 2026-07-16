# S9-V1A-R: Pipelined-Transport Evidence Repair

Date: 2026-07-15 (ASCII only). Devices: OP12 (5ae7a43d, HTP v75, USB 6-2), OP15
(3C15AU002CL00000, HTP v81, USB 8-3). This document repairs and re-grounds the S9-V1A
evidence. The original V1A files are preserved unchanged: raw artifacts in artifacts/,
V1A harness copies in v1a_historical/, and the original narrative in RESULTS.md. New
repair artifacts are in artifacts_r/.

CP0 integrity (before any edit): HEAD 933c722f6, unchanged; the phone-pim tree is untracked;
release + ASan/UBSan phone-pim CTest 3/3 each; certified worker 0a50ca72...e749 present and
unchanged on both phones. Full record: artifacts_r/cp0_integrity.txt, artifacts_r/cp0_pretests.txt.
No commit or push.

## 1. Claims withdrawn from V1A (and why)

The V1A profiling had three defects; the following V1A claims are WITHDRAWN:

- "SHA-256 is 61% (OP12) / 69% (OP15) of the wall." INVALID: it summed HOST-CPU timers
  (host hash + host prefix hash) with PHONE-CPU timers (worker frame-SHA + store chunk-hash +
  full-verify) as a single fraction of wall time. Those run on two different CPUs and overlap
  in wall time, so their sum is not a wall fraction. Repaired reporting keeps host-side and
  phone-side timers strictly separate (Section 3).
- "Each 4 MiB chunk is hashed four times over identical bytes." INACCURATE on count and domain.
  The correct structural count is SIX SHA passes per chunk over the chunk data (three host,
  three phone) plus one whole-file phone re-verify at commit, and the domains differ: the
  outer-frame SHA covers the 56-byte STAGE_CHUNK envelope PLUS the data, while the
  manifest/chunk/prefix SHA covers the data only. See Section 3.
- "Round-trip idle is 26% (OP12) / 33% (OP15)." WITHDRAWN: it came from a process-lifetime
  worker counter that also accumulated the idle recv waits during HELLO/PREPARE/EXECUTE/RELEASE/
  SHUTDOWN and the host-oracle gaps. The repaired counter is stage-scoped (STAGE_* opcodes only)
  and is reported as a measured value, not as "round-trip idle" (recv_header also includes the
  real socket header transfer, not only idle).

## 2. Profiling repairs (measurement-only, opt-in, default OFF; no v3 wire change)

- Receive profiling is now a real CLI opt-in: worker `--profile-recv` (default OFF). The worker
  no longer profiles unconditionally.
- Counters are stage-scoped and per-frame: the socket layer records only the MOST RECENT frame's
  split; the worker attributes it by opcode and sums ONLY STAGE_BEGIN/CHUNK/COMMIT/ABORT, so
  HELLO/PREPARE/EXECUTE/RELEASE/SHUTDOWN and oracle gaps are excluded.
- Host outer-frame SHA time is separated from actual socket-send time: host `--profile-transport`
  (default OFF) splits `host_send` into `host_frame_sha` (outer-frame SHA over envelope+data) and
  `host_socket_send` (the socket write).
- Worker prefix-hash time is measured separately (store `prefix_hash_us`, printed via stderr,
  never serialized over the wire).
- Hash domains are labelled at the source and in every report: outer-frame SHA = 56-byte envelope
  + data; manifest/chunk/prefix SHA = data only.
- Host and phone CPU timers are reported separately and never summed as a wall fraction.
- Profiling overhead is measured directly (Section 3): counterbalanced profiling-ON vs OFF reps.

## 3. Repaired CP1 accounting (window=1, 6 profiling-ON + 6 profiling-OFF counterbalanced reps/phone)

Goodput (window=1) and profiling overhead (median wall ON/OFF):

| device | ON p50 | ON p95 | ON CoV | OFF p50 | overhead (ON/OFF wall) |
|---|---:|---:|---:|---:|---:|
| OP12 | 15.49 | 15.53 | 0.2% | 15.41 | 0.995x (none) |
| OP15 | 17.54 | 17.93 | 4.6% | 17.67 | 1.008x (none) |

Profiling adds no measurable overhead.

HOST-side serial timeline (single CPU; these sum validly), medians in ms:

| device | wall | read | hash(data) | prefix_hash(data) | send | ack_wait | host accounted |
|---|---:|---:|---:|---:|---:|---:|---:|
| OP12 | 28568 | 225 | 1767 | 1480 | 1553 | 23502 | 99.86% |
| OP15 | 25240 | 229 | 1799 | 1507 | 1523 | 20172 | 99.96% |

`host_send` splits into the outer-frame SHA (envelope+data) and the actual socket write:
OP12 frame_sha=1478 + socket_send=75; OP15 frame_sha=1452 + socket_send=71. The socket write is
~75 ms; the old "send" cost was almost entirely the outer-frame SHA.

PHONE-side timers, which occur INSIDE host_ack_wait on a DIFFERENT CPU and are therefore reported
separately and NEVER added to the host timeline (medians, ms):

| device | recv_header | recv_payload | frame_sha(env+data) | chunk_hash(data) | prefix_hash(data) | write | data_sync | full_verify |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OP12 | 5068 | 4337 | 7399 | 4737 | 3226 | 295 | 586 | 2795 |
| OP15 | 6586 | 1579 | 6562 | 3944 | 3114 | 170 | 291 | 2917 |

Thermal band during the profiling reps (NOT energy): OP12 zone_max 55-57 C, cpu_freq 1709-1824 MHz;
OP15 zone_max 95 C, cpu_freq 883-1325 MHz. OP15 was thermally throttled (95 C, downclocked), which
is the honest cause of its higher variance below.

Note: the stage-scoped recv_header is 5068/28568 = 17.7% (OP12) and 6586/25240 = 26.1% (OP15) of wall,
and it includes the real socket header transfer, not only idle. This supersedes the withdrawn
"26%/33% round-trip idle" figures, which were process-lifetime and contaminated by oracle gaps.

Structural hash inventory per 4 MiB chunk (from the code, independent of timing):
- HOST: manifest verify SHA (data), rolling durable-prefix SHA (data), outer-frame SHA in
  send_frame (envelope+data) = 3 passes.
- PHONE: outer-frame SHA in receive_frame (envelope+data), store chunk verify SHA (data),
  rolling durable-prefix SHA (data) = 3 passes; plus ONE whole-file re-verify SHA at commit.
Host and phone hashing overlap in wall time; no single combined wall fraction is claimed.

## 4. CP4 measurement matrix

Counterbalanced (rotating Latin-square) window order; full provenance per row (device serial,
worker SHA, host SHA, USB path, source-object SHA, exact byte counters, harness version);
thermal-zone and CPU-frequency snapshots captured around each run (NOT energy).

### Full-shard window gate (464,114,176-byte shard, blk.2 M=16 HTP oracle)

5 counterbalanced reps per window; the fail-closed analyzer validated device serial, worker SHA,
model SHA, bytes, chunk count, verdict DYNAMIC_FFN_PASS, published_store, rel-L2 finite < 5e-3,
accepted_chunks == 111, duplicate_chunks == 0, wasted_bytes == 0 on EVERY row. Median goodput MiB/s:

| device | w1 | w2 | w4 | w8 | best | median ratio | conservative min(best)/max(w1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| OP12 | 14.12 | 36.10 | 36.91 | 35.32 | w4 | 2.61x | 1.89x |
| OP15 | 16.21 | 34.57 | 23.83 | 36.99 | w8 | 2.28x | 1.27x |

`analyze_sweep.py --gate` returns 0: OVERALL GATE PASS. Both phones clear >=1.20x on BOTH the median
and the conservative min(best)/max(w1) ratio. OP15's w4 median (23.83) is depressed by thermal
throttling (95 C during the run, see Section 3); w2/w8 are consistent and the gate holds regardless.
0 errors, 0 duplicate chunks, 0 wasted bytes across all 40 gate runs; every run transferred exactly
464,114,176 bytes.

### 64 MiB and 256 MiB provisioning goodput (provision-only, synthetic object)

64 MiB, windows 1/2/4/8 x5 (median MiB/s, ratio vs w1):

| device | w1 | w2 | w4 | w8 | best ratio |
|---|---:|---:|---:|---:|---:|
| OP12 | 15.33 | 32.41 | 33.83 | 33.40 | 2.21x (w4) |
| OP15 | 13.09 | 21.60 | 21.69 | 21.77 | 1.66x (w8) |

256 MiB, best two windows (w4, w8) x3 (median MiB/s): OP12 w4 36.92 / w8 36.86; OP15 w4 25.21 /
w8 35.95 (OP15 w4 had a 4.86 thermal-throttle outlier; w8 is stable). Larger objects amortize the
per-transfer fixed cost, so goodput rises with object size on OP12; OP15 stays throttle-limited.

### Simultaneous (both phones, separate USB buses, window 4, x3)

OP12 37.45 MiB/s, OP15 22.86 MiB/s, both with min~=max (tight). Each phone's simultaneous goodput
matches its solo goodput (OP12 37.45 vs 36.91 solo; OP15 22.86 vs its throttled solo), confirming
the two phones run on independent USB domains with no measurable cross-interference.

## 5. Byte accounting and bounds (CP3)

Every host record now carries exact data-byte counters: attempted, socket_written,
acked_durable, wasted (= written - durable), retried. On a disconnect the host emits a
FAIL_PROVISION record with these counters so a resumed run can be reconciled. The 64 MiB
outstanding-payload bound is enforced on the COMPLETE request payload (56-byte envelope + data)
with overflow-safe arithmetic. `--stage-window` is restricted to the frozen tested set {1,2,4,8}
and rejects other values. The standalone bench validates monotonic data sequence numbers, exact
payload size, per-frame ACK sequence match (FIFO), a final summary frame, and host/device frame
+ byte agreement.

### Adversarial windowing tests (structural JSON, fail-closed; both phones exit 0)

adversarial_tests.py parses records structurally, persists every host stdout/stderr/exit-status,
and returns nonzero if any test fails. Both phones: T1/T2/T3 PASS.

- T1 windowed resume-after-partial (window 4): stop after 40 durable chunks, reconnect, resume ->
  DYNAMIC_PROVISION_PASS, resume_offset == 167,772,160 (== 40 x 4 MiB) on both phones.
- T2 windowed kill-mid-flight (window 8): the first host FAILED (nonzero exit) and emitted a
  FAIL_PROVISION record; restart + resume completed with the correct object SHA and 0 < resume_offset
  < object_size. Retry/waste computed from the persisted first-run and resumed-run records:
  - OP12: first written 184,549,376 / durable 150,994,944 -> waste 33,554,432 (8 chunks, the in-flight
    window-8 payload discarded by the kill); resume_offset 150,994,944; retry 33,554,432.
  - OP15: first written 50,331,648 / host-acked durable 16,777,216 -> waste upper bound 33,554,432;
    resume_offset 20,971,520. Note resume_offset (5 chunks) EXCEEDS the host's acked-durable (4 chunks):
    the worker had durably written a 5th chunk whose ACK the host never received before the kill. The
    host does not guess this; it resumes from the worker's reconstructed verified prefix. So the host's
    waste figure is a correct UPPER BOUND; true wire waste was one chunk less.
- T3 durable-result identity (window 8, full oracle): DYNAMIC_FFN_PASS, rel-L2 2.9e-4 (OP12) /
  2.9e-4 (OP15), duplicate_chunks 0, published SHA 5cfba18d...360d on both.

The window-agnostic store/worker invariants (corrupt/reorder/stale-epoch/duplicate/cache-hit-vs-active/
mode-0400 publication/symlink) are enforced by the worker identically regardless of host windowing and
are covered by the store(45)+stream(45) CTest suites, green under release and ASan/UBSan.

## 6. Verification of the repaired tooling

- ASan/UBSan host window>1 test (asan_window_test.sh): windows 1/2/4/8 + windowed resume, zero
  sanitizer diagnostics -- PASS.
- Analyzer fail-closed self-tests (analyze_sweep.py --selftest): 18/18 mutation cases fail closed.
- Release + ASan/UBSan phone-pim CTest: 3/3 each, before and after the edits.

## 7. Honest final labels

- Independent full-shard window gate: PASS on both phones (OP12 2.61x median / 1.89x conservative;
  OP15 2.28x median / 1.27x conservative; fail-closed analyzer exit 0). The wall-clock win is real
  but partly a DVFS effect: a full window keeps the phone CPU busy so the dominant SHA work runs at a
  higher clock (OP15 was throttled to 95 C / <=1.3 GHz and shows the effect and the variance most).
- Contract completeness: INCOMPLETE. This is a transport-and-measurement slice with repaired,
  honest profiling and a validated windowed data path. It is NOT a full dispatch/scheduler contract,
  NOT multi-model, and NOT a llama-server integration.
- Capacity: UNPROVEN (single object, single dense-FFN island, no live scheduler or eviction).
- Energy: DEFERRED. Only thermal-zone and CPU-frequency snapshots were captured; these are device-
  state context, explicitly NOT energy measurements.

Not done by design (reported, not implemented, per the task): SHA-256 de-duplication, protocol v4
(batched ACKs), mixed-workload scheduling, llama-server integration, kernel/model-graph changes.
No commit or push.
