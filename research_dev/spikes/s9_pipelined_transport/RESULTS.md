# S9-V1A: Dynamic Provisioning Transport — Profile and Improve

Date: 2026-07-14. Devices: OP15 (`3C15AU002CL00000`, HTP v81, USB 8-3), OP12
(`5ae7a43d`, HTP v75, USB 6-2), both enumerating at 5 Gbit/s on separate USB domains.

Scope: profile the certified sequential (window=1) dynamic-provisioning transport, then add a
bounded windowed host data path and gate it at ≥1.20× median goodput without weakening any
correctness, durability, resume, or publication invariant. No mixed-workload scheduler work.
No change to `json-graph.cpp`, `ggml_backend_sched`, model graphs, HTP kernels, or llama-server.

Verdict: **`S9-V1A_PIPELINED_TRANSPORT_PASS`** (both phones ≥1.20×; see CP2). All measurement is
opt-in and does not alter the v3 wire format; the certified baseline worker SHA `0a50ca72…e749`
is untouched on-device (a separately-named measurement worker `aaa928b8…` carries the stderr
timing only).

---

## Checkpoint 0 — reproduce the certified baseline

- Host release CTest 3/3, ASan/UBSan CTest 3/3, integration stream 45/0.
- Android on BOTH phones: protocol 22/0, store 45/0.
- Android certified worker SHA-256 `0a50ca72…e749` present and unchanged on both phones.
- Exact-final 464,114,176-byte shard (SHA `5cfba18d…360d`), 111 × 4 MiB chunks,
  `model_source=published_store`, `blk.2` M=16 repeat=7 HTP0, rel-L2 < 5e-3.
- Raw record: `artifacts/cp0_reproduce.json`.

The certified window=1 goodput reproduced at 13.8–15.9 MiB/s (OP12) — consistent with the prior
14.9 MiB/s record. OP15's 94.6 s / 4.678 MiB/s baseline is the cold-path outlier its own report
flagged (26% CoV); a warm run here reproduces 18.8 MiB/s, and the sweep re-measures w1 cleanly.

---

## Checkpoint 1 — explain the missing time (measurement-only)

Two measurement paths were added, both opt-in and wire-neutral:

1. **Production host timers** (no wire change): split `Client::call` so the stage wall
   decomposes into `host_read`, `host_hash` (manifest re-verify), `host_prefix_hash` (rolling
   durable-prefix SHA), `host_send` (blocked send), `host_ack_wait` (blocked ACK wait). Emitted
   as new `stage_host_*` JSON fields.
2. **Worker `recv_profile`** (stderr only, opt-in flag): splits `receive_frame` into socket
   `recv_header`, `recv_payload`, and the mandatory outer-frame `frame_sha`. Combined with the
   existing store `StageMetrics` (chunk_hash / write / data_sync / full_verify).
3. **Separate transport bench** `llama-phone-pim-bench` (roles sink/source/host): its own 16-byte
   frame header — never touches v3 — runs over the same adb-forwarded USB socket to isolate pure
   H2D/D2H transport, the frame-SHA cost, and the window effect with **no UFS/publish**.
4. **`adb push` staged-file control** for a transport upper reference.

### Window=1 accounting (instrumented full-shard run)

| device | wall (ms) | host_read | host_hash | host_prefix_hash | host_send | host_ack_wait | top-level accounted |
|---|---:|---:|---:|---:|---:|---:|---:|
| OP12 | 30090 | 143 | 1469 | 1873 | 1545 | 25012 | **99.85%** |
| OP15 | 23531 | 161 | 1529 | 1827 | 1586 | 18380 | **99.80%** |

Both clear the ≥90% gate with essentially no unexplained wall. Inside `host_ack_wait`, the worker
`recv_profile` + `StageMetrics` attribute it to:

| device | recv_payload (transfer) | frame_sha | store chunk_hash | write | data_sync | full_verify | recv_header (round-trip idle) |
|---|---:|---:|---:|---:|---:|---:|---:|
| OP12 | 5228 | 7460 | 4744 | 318 | 587 | 2879 | 7836 |
| OP15 | 1629 | 6290 | 3681 | 151 | 292 | 2902 | 7759 |

### What the profile shows

- **SHA-256 dominates: 61% (OP12) / 69% (OP15) of the wall**, and it is largely *redundant* —
  each 4 MiB chunk is hashed **four times** (host verify + host rolling-prefix + worker frame-SHA
  + worker store chunk-hash) and the whole 464 MiB is re-hashed once more at commit (full_verify).
- **Round-trip idle** (`recv_header`, the worker blocked waiting for the host's next frame) is
  26% (OP12) / 33% (OP15) of the wall — this is what a bounded window removes.
- **Pure transfer** (`recv_payload`) is only 17% (OP12) / 7% (OP15).
- The **memory-sink bench** moves 64 MiB at window=1 with no UFS/hash at **120 MiB/s (OP12) /
  105 MiB/s (OP15)** — ~8× the full-stage goodput — so the wire is not the bottleneck.
  Pure-transport window sweep is flat-to-slightly-up (OP12 0.92×, OP15 1.07–1.08× at w4/w8);
  the OP15 streaming ceiling (single final ACK) is 246 MiB/s vs 105 at w1 (2.35×), i.e. per-frame
  reverse-ACK traffic caps forward throughput over adb-USB.
- `adb push` control: OP12 **249 MiB/s**, OP15 **86 MiB/s** (OP15 is UFS-write bound; the bench
  streams to RAM and hits 246 MiB/s, so OP15's *disk write*, not the wire, caps adb-push).

Raw: `artifacts/op1{2,5}_prod_instrumented.json` (+ `_worker.log`), `artifacts/op1{2,5}_bench.jsonl`,
`artifacts/adb_push_control.json`, `artifacts/cp1_accounting.json`.

### CP2 headroom decision

Memory-sink wall (≈3.9 s for the full shard) ≪ full-stage wall (30 s) → pipelining headroom
exists (task rule). The windowed lower bound (worker serial chain) implies a ceiling of **1.42×
(OP12) / 1.57× (OP15)**, both above the 1.20× gate. Proceed to CP2 and measure the real gate
(the bench cannot capture host↔worker overlap or the DVFS effect, so the gate must be measured).

---

## Checkpoint 2 — bounded windowed streaming

`--stage-window N` (1–64, default 1). At most `N` STAGE_CHUNK requests outstanding, never more
than **64 MiB** of unacknowledged payload. Chunks are still sent strictly in ascending index
order; the worker stays single-threaded and order-strict; each outstanding request retains its
own expected cumulative durable-prefix digest until its ACK is validated in FIFO order. `window=1`
is byte-identical to the prior stop-and-wait path (`call` = `send_request` + `recv_response`).
No wire-format change → protocol stays v3.

### Correctness (device)

window=1 and window=4 both: `DYNAMIC_FFN_PASS`, `model_source=published_store`, 111/111 chunks,
published object SHA `5cfba18d…360d`, rel-L2 2.95e-4 — bit-identical durable result.

### Performance gate (full-shard sweep, 5 rotated reps/window, fresh store each run)

Median useful provisioning goodput (MiB/s) and ratio vs window=1:

| device | w1 | w2 | w4 | w8 | best ratio | gate ≥1.20× |
|---|---:|---:|---:|---:|---:|:--:|
| OP12 | 15.59 (CoV 2.5%) | 37.49 (2.40×) | 37.94 (2.43×) | 38.02 (2.44×) | **2.44×** | **PASS** |
| OP15 | 4.89 (CoV 62.5%) | 21.54 (4.41×) | 31.92 (6.53×) | 32.81 (6.71×) | **6.71×** | **PASS** |

**0 correctness failures and 0 errors across all 40 runs** (every run: `DYNAMIC_FFN_PASS`,
`published_store`, 111/111 chunks, SHA `5cfba18d…360d`, rel-L2 < 5e-3). Raw:
`artifacts/op1{2,5}_sweep.jsonl`, `artifacts/cp2_sweep_summary.json`.

Window=4 is the recommended setting: it captures the full benefit on OP12 (w8 adds nothing) and
is stable; window=8 is marginally better on OP15 and is the more stable high-throughput setting
there. Bytes transferred = 464,114,176 exactly per run; **0 retried/wasted bytes** (no duplicate
chunks; the store reported `duplicate_chunks=0` on every clean run).

**Variability / thermal honesty**: OP15's window=1 is highly variable (CoV 62.5%, median 4.89 but
range 4.58–18.37) — the cold-path outlier its own baseline flagged — which inflates OP15's ratio.
Even against OP15's *fastest* observed w1 (18.37), window=8 (~33) is still ~1.8×, so the gate holds
regardless. OP12 is tight (CoV ≤2.5% at every window). Runs were rotated (w1,w2,w4,w8 per rep) with
a 3 s cooldown between runs to bound thermal drift; back-to-back full-shard provisioning still heats
the phones, so absolute numbers carry a DVFS/thermal band (see caveat).

### Adversarial windowing tests (device)

Window-agnostic store/worker invariants (corrupt payload, out-of-order/duplicate chunk, stale
epoch/generation, cache-hit-vs-active-transfer, hidden-partial invisibility, mode-0400/one-link
publication, symlink/FIFO rejection) are enforced by the worker/store **identically regardless of
host windowing** (the worker is single-threaded and order-strict) and are covered by the store(45)
+ stream(45) suites, green under release and ASan/UBSan. The new host-pipeline behavior was tested
end-to-end on both phones:

| test | OP12 | OP15 |
|---|:--:|:--:|
| T1 windowed resume-after-partial (stop 40 durable, reconnect, resume w4 → complete; resume offset == 40×4 MiB, SHA ok) | PASS | PASS |
| T2 windowed kill-mid-flight (SIGKILL worker with ≤8 requests outstanding; host errors out; restart, resume w8 → recovers from worker's verified prefix, SHA ok) | PASS (resumed @163.6 MB) | PASS (resumed @138.4 MB) |
| T3 durable-result identity (window=8 publishes the same object SHA `5cfba18d…` as window=1, rel-L2 < 5e-3) | PASS | PASS |

T2 confirms the key invariant under pipelining: on failure the host never guesses which in-flight
chunks became durable — it reconnects and resumes from exactly the worker's reconstructed,
hash-verified contiguous prefix. Raw: `artifacts/op1{2,5}_adversarial.txt`.

### Honest caveats

- Part of the windowed speedup is a **DVFS effect**: window=1 idles the CPU between round-trips so
  the governor down-clocks, making the (dominant) SHA work slower; a full window keeps the CPU busy
  and clocked up (e.g. OP12 worker `frame_sha` dropped 8.3 s→2.7 s for the *same* 464 MiB simply
  because the CPU stayed at a higher clock). The wall-clock improvement is real and gated on wall
  time, but it is not purely "transport pipelining"; it is "keep the pipe and the CPU busy".
- The measurement worker (`aaa928b8…`) adds only opt-in stderr timing; the v3 wire is byte-identical
  and the **certified worker `0a50ca72…` is untouched on both phones**. `--stage-window` default is 1,
  so nothing changes unless explicitly requested.

## Next levers (profiled, NOT implemented — reported for review per the task)

The window gate PASSED, so windowing stands. Beyond it, the profile ranks the remaining levers:

1. **Redundant SHA-256 de-duplication (biggest, 61–69% of wall).** Each 4 MiB chunk is hashed 4×
   (host verify + host rolling-prefix; worker frame-SHA + worker store chunk-hash) plus a full
   464 MiB re-verify at commit. The two worker hashes are of identical bytes and could be a single
   pass (feed `receive_frame`'s digest into `put_chunk`); likewise the two host hashes. This does
   **not** weaken verification (same bytes, same digest, still one durable-prefix + one full verify),
   but it changes internal APIs and warrants its own review — not done here.
2. **Batched durability ACKs.** The OP15 memory-sink streaming ceiling (246 MiB/s vs 105 at w1)
   shows per-frame reverse-ACK traffic caps forward throughput over adb-USB. ACKing every K durable
   chunks (still only durable, manifest-verified prefixes) would cut reverse chatter. This is a wire
   change → **protocol v4**, out of scope for this v3-frozen slice.
3. **adb-sync staged-file import adapter / a different USB transport** — only relevant if 1–2 are
   exhausted; `adb push` already reaches 249 MiB/s (OP12) but is UFS-write bound on OP15 (86).

## Files

Measurement-only, all opt-in (default off), no v3 wire change:
- `examples/phone-pim/phone_pim_bench.cpp` (+ CMake target `llama-phone-pim-bench`) — transport probe.
- `phone_pim_host.cpp` — `Client::{send_request,recv_response}` split; `--stage-window`; bounded FIFO
  pipeline; `stage_host_*` timing fields.
- `phone_pim_socket.{h,cpp}` — opt-in `set_receive_profiling` / `ReceiveProfile`.
- `phone_pim_worker.cpp` — enables recv profiling; prints `recv_profile` at shutdown.
Deliverable artifacts + harnesses under `research_dev/spikes/s9_pipelined_transport/`.
