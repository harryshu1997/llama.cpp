# Operator routes v2 summary

Date: 2026-07-29

Verdict:
`DYNAMIC_INPUT_COHERENCE_PASS; FFN_PASS; TOP8_HEAD_PASS; ATTN_16K_STATE_PASS; SHORT_CONTEXT_QWEN_LAYER_PASS; SHORT_CONTEXT_GEMMA_MEDIAN_ONLY; LARGE_CONTEXT_LAYER_FAIL; FULL_MODEL_NOT_RUN`.

Hardware:

- Host GPU: NVIDIA RTX A6000,
  `GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f`.
- Final clock check: 240 MHz graphics, 5001 MHz memory.
- Phone: OP15 `3C15AU002CL00000`, HTP v81, direct AOA USB.
- OP12 did not participate.

Primary results:

| route | control | treatment | change |
| --- | ---: | ---: | ---: |
| Qwen FFN, CUDA-resident result | 1.0805 ms | 1.0032 ms | -7.16% |
| Gemma FFN, CUDA-resident result | 0.7690 ms | 0.7357 ms | -4.33% |
| Qwen top-8 vocabulary head | 3.2062 ms | 2.7733 ms | -13.50% |
| Gemma top-8 vocabulary head | 4.3744 ms | 3.6516 ms | -16.52% |
| Qwen attention, KV=16K, state return | 1.2497 ms | 1.1905 ms | -4.74% |
| Qwen complete layer, KV=136 | 1.5588 ms | 1.5059 ms | -3.39% |
| Gemma complete layer, KV=136 | 1.1707 ms | 1.1440 ms | -2.29% median |
| Qwen complete layer, KV=8K | 2.1727 ms | 2.3661 ms | +8.90% |

The Gemma KV=136 treatment p90 is 6.28% worse than control. The Qwen KV=8K
and Gemma KV=8K complete-layer paths are ineligible. See `../../../RESULTS.md`
for the full interpretation and raw log names.

Verification:

- Seven final host/Android spike programs compile with
  `-Wall -Wextra -Werror`.
- Host and worker reject a two-group phone-attention request with exit 2.
- All source and binary hashes in `SHA256SUMS.txt` verify.
- The Hexagon core source follow-up was not rebuilt because its old CMake
  cache depends on unavailable `/workspace` and `/opt/hexagon` paths. The
  physical tests use the independently validated compute-buffer worker fix.
