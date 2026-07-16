# S7-V0 Ragged HMX Attention Tile Skip - Results

## Verdict

**PASS on OP15 for the isolated operator screen.** CPU-reference correctness passes, the dense
control is within 1 percent, and useful ragged masks improve HMX attention by 1.27-1.49x at both
B=16 and B=32.

This is not yet a real-layer, request-trace, end-to-end latency, or energy result.

## Implementation under test

- `tests/test-backend-ops.cpp`: deterministic `ALL_VALID`, `RAGGED_PREFIX`, and `RAGGED_HOLES`
  mask profiles with self-defined Gemma-4 SWA tensors.
- `ggml/src/ggml-hexagon/htp/hmx-flash-attn-ops.c`: per-sequence active KV-block worklist before
  the first K/V DMA; both pipeline and sequential loops consume the worklist.
- `ggml/src/ggml-hexagon/htp/htp-ops.h`: opt-in operation flag.
- `ggml/src/ggml-hexagon/ggml-hexagon.cpp`: default-off environment selector
  `GGML_HEXAGON_FA_SKIP_MASKED`.

The classifier accepts only exact fp16 negative infinity (`0xfc00`). It runs only for single-token
decode, F16 broadcast masks, and leaves the original path in place when no block can be removed.
When the feature is disabled, the active-list VLA is one element rather than context-sized.

## Build and device

Built both required targets from current source:

```bash
cmake --build build-snapdragon --target htp-v81 test-backend-ops --parallel
```

Device: OP15, Hexagon v81, 8 HVX threads, HMX enabled, 8 MiB VTCM. The tensor shape is:

```text
Q=[256,1,16,B], K/V=[256,512,8,B], mask=[512,1,1,B]
```

This shape unconditionally attempts the HMX path. The candidate speedup can only come from the new
HMX loop because the HVX fallback does not inspect the new flag.

## Correctness

`test-backend-ops test` compared HTP0 against CPU for all three mask profiles at B=8 and B=16.

| flag | cases | result |
|---|---:|---|
| off | 6 | 6/6 pass |
| on | 6 | 6/6 pass |

The prefix profile includes partially masked boundary blocks. The hole profile proves that a valid
block after a skipped block is handled correctly by the DMA ping-pong and deferred output update.

## Latency

Times are the harness-reported `us/run`. Baseline and candidate receive the identical tensor and
mask. Prefix medians and CoVs use five independent processes. Dense and hole rows are initial
screens whose individual processes contain hundreds of timed operator runs.

| B | mask | flag off | flag on | speedup | process CoV off/on |
|---:|---|---:|---:|---:|---:|
| 16 | all valid | 4826.93 | 4816.95 | 1.002x | not repeated |
| 16 | ragged prefix | 4813.83 | 3233.16 | **1.489x** | 0.20% / 0.39% |
| 16 | ragged holes | 4821.28 | 3767.84 | **1.280x** | not repeated |
| 32 | all valid | 9531.68 | 9603.58 | 0.993x | not repeated |
| 32 | ragged prefix | 9576.93 | 6420.72 | **1.492x** | 0.19% / 0.45% |
| 32 | ragged holes | 9564.24 | 7508.49 | **1.274x** | not repeated |

For C=512 the current HMX source rule selects four 128-token KV blocks. From the deterministic mask
construction, the prefix case executes an average 2.5 blocks per stream and the hole case executes
3. These are source-derived counts, not device counter measurements. The observed speedups are
close to, but below, the corresponding work-reduction bounds of 1.60x and 1.33x.

The harness still reports rectangular FLOP counts, so its displayed GFLOPS is not a meaningful
candidate metric after blocks are omitted.

## Device instability

One monolithic nine-case baseline sweep lost the OP15 USB connection and produced no result. The
phone reappeared without intervention. Final correctness and timing used short isolated processes;
all reported runs completed normally.

The measured C=512 shape selects the pipeline loop. The sequential worklist loop compiled and was
code-reviewed, but was not selected by this device test.

## Claim boundary and next gate

The result validates a kernel mechanism for ragged continuous decode without reducing admitted
batch size. It does not establish novelty by itself; variable-length attention exists elsewhere.
The system-level question is whether real server traces create enough padded KV blocks, and whether
the phone saves server work and total joules while meeting service latency.

Next, replay a real arrival/output-length trace to form dynamic batches, feed its per-stream KV
lengths into one real Gemma layer, and compare:

1. rectangular HMX attention,
2. ragged HMX attention with the same admitted batch,
3. smaller compacted batches including scheduler/compaction cost.

Only after that passes should the mask scan be replaced by scheduler-provided lengths or a compact
block bitmap and integrated into the continuous phone pipeline.

Raw outputs and the command/hash manifest are under `scratchpad/s7_ragged_fa/`.
