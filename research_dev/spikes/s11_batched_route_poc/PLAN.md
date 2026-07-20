# S11-B Static Batched Route Proof of Concept

## Question

Can an exact resident phone layer island process multiple independent sequence
rows in one HTP invocation, while preserving the server tail, per-sequence KV,
and exact greedy output?

This is a bounded mechanics and capacity test. It is not the dynamic scheduler
and it does not authorize an energy claim.

## Routes

1. `SERVER_ONLY`: the complete Gemma-4 12B F16 model runs on one A6000.
2. `A0_OP15`: OP15 HTP owns layers `[0,2)` and the same A6000 owns `[2,48)`.
3. `A0_OP15_OP12`: OP15 owns `[0,2)`, OP12 owns `[2,3)`, and the A6000 owns
   `[3,48)`. This route is an exactness checkpoint, not the batch sweep.

All weights and contexts are resident before the paid request window.

## Implementation

- `stagenet` accepts bounded batched prefill and decode commands.
- Each row carries an explicit sequence ID and owns a distinct KV sequence.
- Batched prefill uses `B x prompt_tokens` real rows. It does not share or copy
  a common prefix between sequences.
- Batched decode submits one active row per live sequence.
- The host tail uses the same batch size and sequence IDs.
- The server-only control uses the same static batch and greedy argmax loop.
- Every output record carries request, batch, and stream identity.
- Group timing is recorded once and copied to the group's request records. The
  analyzer rejects inconsistent copies and never divides group latency by `B`.
- A versioned stage hello binds each stage's exact layer range, model layer
  count, and embedding width. The driver rejects gaps, overlaps, wrong order,
  and a host that does not own the terminal tail. The client requests the hello
  and reads the complete frame under one five-second deadline, so stale or
  partial workers fail without deadlock.
- Head stages accept token-only frames; middle stages require the injected
  activation width. Premature EOF and malformed role frames fail closed.

The bounded experiment uses:

```text
prompt tokens: 28
generated tokens: 4
B: 1, 2, 4, 8, 16
per-sequence context: 96
maximum prompt: 64
maximum physical prefill microbatch: 512 rows
```

The driver rejects a shape whose real `B x prompt_tokens` exceeds the physical
microbatch. This avoids ambiguous dense cut-activation ordering across multiple
microbatches.

## KV Capacity Repair

Gemma-4 LayerSplit now filters KV allocation to the same
`LLAMA_LAYER_START/END` range used by partial weight loading and graph
construction. Full-model behavior is unchanged when the environment variables
are absent. A cut that separates a shared-KV layer from its source fails during
context construction.

## Gates

1. Host CUDA, host CPU, and Android builds pass.
2. The existing one-phone and two-phone `B=1` routes remain exact.
3. OP15 `[0,2)` and OP12 `[2,3)` allocate KV only for their owned layers.
4. Every `B` returns exactly the server-only token IDs for every stream.
5. Useful treatment throughput increases by at least 4x from `B=1` to `B=8`.
6. The repeated `B=8` treatment has group-wall CoV at most 5 percent.
7. A server-relief claim requires treatment throughput at least equal to the
   equally batched server-only control. Otherwise the result is phone
   utilization and memory capacity only.
8. Missing phone or server wall energy remains `NOT_RUN`, never zero.
9. The v2 runner binds model, binary, runner, `layersplit.cpp`, and
   `llama-model.cpp` hashes. Historical v1 artifacts are not parsed as v2.
10. Per-node HTP placement remains a separate certificate; selecting `HTP0`
    and observing graph splits is not by itself a no-fallback proof.
