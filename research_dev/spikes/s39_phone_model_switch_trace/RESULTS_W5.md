# S39 W5 batched phone-to-CUDA handoff

Status:
`HANDOFF_MECHANICS_PASS; TASK_QUALITY_AND_CONCURRENT_DELTA_OPEN; SCHEDULER_INELIGIBLE`.

## Question

Can CUDA reconstruct native request state from exact phone-committed token
histories in a larger batch, then take publication ownership without a missing
or duplicated token?

This checkpoint tests only that mechanism. It does not replace the failed W4
Qwen2.5 Q8_0 quality gate.

## Frozen contract

`W5_HANDOFF_CONTRACT.json` was written before the physical acquisition:

- model: Qwen2.5 14B Q8_0,
  `23ca481b8226b2492ba8f3eb7af41e0f99d8605c16fb6dec7bc5cf6716b673cf`;
- batch: 8;
- prompt: 8 frozen WikiText token IDs per request;
- phone-authoritative prefix: 4 generated tokens;
- CUDA continuation: 8 generated tokens;
- control history chunk: 2 tokens per sequence;
- catch-up history chunk: 8 tokens per sequence;
- maximum physical rows: 64;
- scope: `MECHANICS_ONLY`;
- scheduler eligibility on pass: false.

Contract SHA-256:
`0da7977b4576bafa0cb91d639270fd0f6edb3f7e575b009fb7eb3966ede51f52`.

## Physical route

The phone route was OP15 GPUOpenCL `[0,30)` followed directly over WiFi by
OP12 GPUOpenCL `[30,48)`. The CUDA control and catch-up route used two local
CUDA0 workers at the same cut. All routes used the same full-model identity
and GGUF file type 7.

The phone route retained eight live sequence states while CUDA ran the control
and catch-up. CUDA control state was removed before catch-up. After catch-up
was prepared, all phone states were removed before the owner epoch changed to
CUDA.

## Result

| Metric | Control | Batched catch-up |
|---|---:|---:|
| History batches | 6 | 2 |
| Continuation batches | 7 | 7 |
| Rows | 152 | 152 |
| Compute and relay wall | 555,645 us | 408,641 us |
| Continuation equality | reference | 8/8 sequences exact |

The 26.5 percent wall-time difference is a single-run observation. It is not a
latency claim.

Every request has:

```text
published history = 4 phone tokens + 8 CUDA tokens
CUDA replay input  = 8 prompt tokens + 4 phone tokens
```

There are no missing or duplicated published tokens. The final state counters
are zero on both routes.

Placement and transport records:

- OP15: `SCHEDULED_PLACEMENT_OK`, 0 missing buffers, 88 rows;
- OP12: `SCHEDULED_PLACEMENT_OK`, 0 missing buffers, 88 rows;
- CUDA head and tail: `SCHEDULED_PLACEMENT_OK`, 0 missing buffers, 304 rows;
- phone relay: 7 batches, 88 rows, 1,802,240 activation bytes, 0 host
  activation bytes;
- CUDA relay: 22 batches, 304 rows, 6,225,920 activation bytes.

OP12 printed the known compile diagnostic for an optional split
flash-attention OpenCL variant. The process continued, the realized graph
completed on OpenCL, and the placement certificate passed.

## Evidence

Directory:
`results/w5_phone_cuda_handoff/run_20260724T193106Z/`.

The probe and validator both exited zero. `SHA256SUMS.txt` verifies every
saved file. The canonical report SHA-256 is
`a8c3e0de4bc599781927ab618451024afc8828a36bfc56c5783caecb1f24cc66`.
The certificate also binds the canonical run context
`d5691ea60ce1dba22e2340d74927356d5670144ef0ad3e116f3a01dc1f769c15`.
That context binds every Python source used by the probe, the launcher,
workers, relays, model shards, phone boot IDs, and the prospective contract.
`W5_ACTIVE_RESULT.json` selects this provenance-hardened repeat.

Offline verification:

```text
S39 full suite after provenance hardening: 157 tests PASS
W5 replay and mutation subset after provenance hardening: 16 tests PASS
```

The mutations cover token loss, token duplication, continuation divergence,
false owner epochs, false sequence-state counts, false row and batch counts,
route model mismatch, and an attempted scheduler-eligibility promotion.

## Limits and next gate

W5 proves a fixed-frontier state reconstruction seam. It does not prove that
phones keep decoding while CUDA loads or catches up. The next bounded gate must
freeze a small generated delta, let phones advance after the first snapshot,
replay both the snapshot and delta into CUDA, and commit exactly one frontier.

The route remains scheduler-ineligible because W4 task quality failed. A
prospective task-quality contract and ground-truth workload are required
before this model route can enter the two-model trace.
