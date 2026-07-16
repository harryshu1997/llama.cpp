# S7-V0 Ragged HMX Attention Tile Skip

## Question

Can batched phone decode keep a large admitted batch while avoiding attention work for padded KV
regions of shorter streams?

This is a kernel screen, not a scheduler or model integration. It uses self-defined tensors in
`test-backend-ops` and changes only the HMX flash-attention block loop behind a default-off flag.

## Candidate

The existing HMX kernel iterates every KV block up to the rectangular batch context. Before K/V
DMA, build a per-sequence worklist and omit a block only when every mask element in that block is
exact fp16 negative infinity.

The omitted work includes:

- K and V DMA
- K and V tile interleave
- QK HMX multiplication
- block softmax
- P x V HMX update

Partially masked blocks, finite mask biases, prefill, and entirely masked sequences retain the
existing path. The selector is `GGML_HEXAGON_FA_SKIP_MASKED=1`; default is off.

## Self-defined tensors

Use the Gemma-4 SWA decode shape without loading a model:

```text
Q    F32 [256, 1, 16, B]
K/V  F16 [256, 512, 8, B]
mask F16 [512, 1, 1, B]
out  F32 [256, 16, 1, B]
```

Test `B={8,16,32}` with deterministic masks:

- `ALL_VALID`: dense overhead control
- `RAGGED_PREFIX`: four repeating valid lengths, including non-block-aligned boundaries
- `RAGGED_HOLES`: one fully masked interior quarter with valid KV after it

## Gates

1. OP15 v81 reports backend support and enters HMX for this F16, 256-wide shape.
2. Flag off and flag on both pass the CPU-reference test at B=8 and B=16.
3. Dense flag-on latency regresses by no more than 5 percent.
4. Prefix and hole cases improve by at least 1.10x at B=16 and B=32.
5. The primary prefix result has process-level CoV below 5 percent.

OP12 v75 is excluded because its fused flash-attention path is already correctness-gated off.

## Stop boundary

Do not integrate into a model graph, continuous scheduler, or energy claim until this isolated
operator gate passes. A pass authorizes a real-layer and trace-driven validation, not production
enablement.
