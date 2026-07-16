# S9 Phone PIM-Style Runtime Slice

Status: pre-staged and sequential dynamic mechanics executed on OP12 and OP15 on
2026-07-14. Capacity and energy remain unproven.

## Objective

Build the smallest real command-driven path that exercises the S9 design:

1. accept either a pre-staged shard or a sequential host-provisioned shard;
2. durably verify and atomically publish dynamic content before READY;
3. keep one complete dense Gemma4 FFN island resident on a phone backend;
4. send only a bounded command and boundary activation for warm execution;
5. return the boundary result; and
6. validate it against an independent production Gemma4 graph.

This is PIM-style placement, not literal PIM or shared server memory. The phone
cannot dereference host addresses.

## Gates

- Protocol: explicit little-endian frame, bounded payload, header CRC, payload
  SHA-256, absolute deadline, monotonic request/command IDs, and complete epoch
  checks.
- Residency: PREPARE binds file byte count and SHA-256; verify, GGUF parse, and
  tensor reads use one open inode; READY follows load, backend support checks,
  upload, and warmup.
- Execution: the worker owns one persistent graph and backend context; all graph
  nodes must be supported by the requested backend.
- Correctness: host expectation comes from `llama_decode` and production Gemma4
  graph callbacks, not from the worker's standalone FFN builder. Relative L2 must
  be at most 5e-3 and all values finite.
- Lifecycle: a fresh boot nonce prevents restart replay; RELEASE and backend or
  oracle failure clear residency and advance its generation; stale generations
  fail closed.
- Trust boundary: worker binds only `127.0.0.1` and is reached through ADB port
  forwarding. The protocol is not authenticated.
- Dynamic storage: reserve the full object; bind ticket, manifest, chunks, and
  generation; ACK only a durable verified prefix; reconstruct it after restart;
  and use no-replace publication plus directory sync.

## Stop Boundary

Do not infer mixed-workload capacity, server relief, energy, or scheduler safety
from this slice. It now has one sequential dynamic transfer and durable publish,
but no multi-model cache, authoritative leases, live credit ledger, batching
policy, or llama-server integration.

The next runtime slice is a bounded pipelined/native bulk path with decomposed
H2D, UFS, verify, materialize, warmup, and D2H measurements. The current
ADB-forwarded stop-and-wait path is mechanics evidence, not a usable link profile.
Scheduler integration starts only after the transfer path and an eligible
operator-island atlas row pass their own gates. See `DYNAMIC_RESULTS.md`.
