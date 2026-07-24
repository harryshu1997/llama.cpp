# S39 W2 mixed-phase direct phone chain

Verdict:

`DIRECT_MIXED_BATCH_MECHANICS_PASS`

No model-switch, throughput, latency-comparison, or energy benefit is claimed.

## Executed mechanism

The host kept request admission and final token ownership. OP15 ran Qwen3 14B
Q4_K_M layers `[0,30)`, sent each cut activation directly over WiFi to OP12,
and OP12 ran `[30,40)`. Both shards and binaries were already resident on
phone UFS after USB provisioning.

The first 16 requests were prefetched, then remained live in decode. While
they were live, 16 new five-token prompts arrived. The route-local batcher
formed this physical call:

```text
one llama_decode, 96 rows
  rows  0..15: decode,  priority 0, position 5
  rows 16..95: prefill, priority 2, positions 0..4
```

Decode rows were placed first even though the prefill group was submitted
first. The full physical batch sequence was:

```text
80P, 16D+80P, 32D, 32D, 32D, 32D, 32D, 32D, 16D
```

This is continuous admission mechanics: new prompts joined a route with live
decode KV and both phases executed in one `llama_decode`. It is not yet a
long-running arrival trace or a multi-model scheduler.

## Real-device result

| Item | Result |
|---|---:|
| Requests | 16 existing decode + 16 new prefill |
| Generated-token checks | 256 / 256 exact |
| Physical batches | 9 |
| Largest batch | 96 rows |
| Seed cohort completion | 23.219 s |
| New cohort completion | 20.169 s after admission |
| Direct activation payload | 7,864,320 bytes |
| Host activation payload | 0 bytes |

Every request generated:

```text
12095, 13, 3555, 374, 279, 6722, 315, 279
```

The direct relay certified nine batches, 384 rows, cut 30, matching Q4_K_M
identity, and zero host activation payload. Both workers ended with `STOP`,
reported 384 session steps, had zero missing compute buffers, and emitted
`SCHEDULED_PLACEMENT_OK`.

OP15 reported 6,900 OpenCL compute nodes and ten declared CPU `GET_ROWS`
nodes. OP12 reported 2,350 OpenCL compute nodes and no CPU compute. OP12 again
failed compilation of one specialized flash-attention variant and used the
generic OpenCL kernel. This is a kernel fallback, not CPU fallback.

## Important performance boundary

The 23.219-second maximum is much lower than W1's 90.796-second B32 point, but
the runs do not isolate one cause. W1 used concurrent request threads whose
sequence order varied inside each batch. W2 uses canonical phase and sequence
ordering, splits initial prefill across two calls, and ran at a different
device state. Therefore W2 records the latency but does not claim a speedup.

The next measurement must compare sorted and deterministically shuffled row
orders with the same workers, batches, model, route, and thermal window. If
the difference persists, the relevant optimization is not nominal batch size;
it is scheduler ordering that prevents internal ubatch fragmentation.

## Verification

- 66 pre-acquisition S39 tests passed.
- The real report enforces the exact nine-batch plan and one mixed batch.
- The independent reducer binds report, relay, lifecycle, and placement logs.
- Ten evidence and mutation tests fail closed.
- All 32 sequences were removed and both workers terminated on `STOP`.

The reducer output is
`results/w2_direct_mixed/direct_mixed_certificate.json`.

## Remaining boundary

The relay still executes one batch through OP15 and then OP12 synchronously.
It cannot overlap OP15 batch `k+1` with OP12 batch `k`; that requires a
bounded multi-inflight protocol and downstream credits. Interface counters
and a host-issued cryptographic reservation descriptor also remain absent.
Phone and total-system energy remain unknown.
