# llama-server continuous-batching CPU + OP15 result

Date: 2026-08-05

Status: PASS for the matched `--no-repack` comparison. This result does not yet
show that OP15 is faster than llama.cpp's default repacked CPU path.

## Test

- Hot model: Qwen3 14B Q4_K_M on RTX 4060 Ti, 4 server slots.
- Cold model: Gemma 4 12B Q4_0 on either desktop CPU alone or desktop CPU +
  OP15 HTP, 8 server slots.
- Cold server: one llama-server process with continuous batching, unified KV,
  `--batch-size 4096`, `--ubatch-size 512`, and 8 CPU threads.
- Trace: all 74 source BurstGPT requests at their original arrival times. The
  trace contains 57 hot requests and 17 cold requests, 33,843 prompt tokens,
  and 11,605 output tokens.
- Request trace SHA-256:
  `b20a9ba66ee3558d835a0e19ed3cfa4c31a4a9e8b4f9c085b29a14f80250a0ff`.
- Both cold arms used the same model, server settings, CPU affinity, and
  `--no-repack`. Only the OP15 FFN client was enabled in the treatment arm.
- OP15 split policy: 9,664 columns for M=1, 8,192 columns for 2 <= M <= 128,
  and 11,136 columns for M > 128.

The runner sends every request through the normal streaming `/completion`
endpoint. It keeps concurrent HTTP requests in flight so llama-server, rather
than the test harness, owns request queueing and physical batch construction.

## Paired result

| Metric | CPU | CPU + OP15 | Change |
| --- | ---: | ---: | ---: |
| Cold makespan | 879.777 s | 708.813 s | -19.43%, 1.241x faster |
| Cold output throughput | 7.864 tok/s | 9.761 tok/s | +24.12% |
| Cold completion p50 | 635.506 s | 462.416 s | -27.24% |
| Cold TTFT p50 | 233.505 s | 132.649 s | -43.19% |
| Cold prefill p50 | 68.763 s | 40.299 s | -41.39% |
| Cold decode p50 | 233.917 s | 211.457 s | -9.60% |
| Hot makespan | 100.997 s | 101.218 s | +0.22% |
| Hot output throughput | 46.397 tok/s | 46.296 tok/s | -0.22% |
| Total output throughput | 13.191 tok/s | 16.372 tok/s | +24.12% |

All 17 cold requests completed faster with OP15. Their latency reductions
ranged from 20.56% to 44.95%, with a 26.59% median. All 74 requests and all
11,605 requested output tokens completed in each arm. Neither model process
used swap.

The hot-model result was effectively unchanged, but the source trace stops
issuing hot requests at about 58 seconds and the hot work completes at about
101 seconds. This run tests real overlap during that interval; it is not a
perpetually saturated-GPU experiment.

## Continuous batching and phone path

The treatment completed 52,320 operator RPCs with exact server, bridge, and
phone counts and zero USB reset recoveries. A prior serialized execution of
the same trace required 334,800 FFN calls, so server batching reduced the
number of graph/RPC invocations by 6.40x.

| Physical graph shape | Calls | Phone columns | RPC p50 | HTP compute p50 |
| --- | ---: | ---: | ---: | ---: |
| M=1 | 864 | 9,664 | 2.173 ms | 1.752 ms |
| 2 <= M <= 128 | 50,208 | 8,192 | - | - |
| M > 128 | 1,248 | 11,136 | - | - |
| All shapes | 52,320 | dynamic | 4.471 ms | 3.049 ms |

The legacy client summary names every M > 1 call `prefill`; that label is not
valid under continuous batching because most of those physical graphs contain
batched decode tokens. The physical M values are the authoritative shape data.

For every selected dense FFN, llama-server builds a desktop prefix of the FFN
columns and an OP15 suffix. The evaluation callback sends the normalized FFN
input to OP15 while the desktop computes its prefix, then adds both partial
outputs. This is operator-level parallelism inside each layer, not layer-wise
model partitioning.

## Correctness and limits

- The opt-in server target rebuilt successfully after the integration audit. A
  separate clean build with `S41_SERVER_FFN_SPLIT=OFF` also passed and contained
  no `S41SERVERFFN` runtime strings.
- The OP15 worker ended with status 0: 52,320 requests and zero recoveries.
- The phone kernel log contained no panic, BUG, call trace, DMA-BUF fault, SMMU
  fault, or IOMMU fault.
- All 17 cold outputs detokenized to coherent text with no replacement
  characters. Two token sequences were bit-for-bit equal to the CPU run.
  Different token choices are expected because the split path adds an F16
  activation/partial-sum boundary.
- The desktop still maps the complete cold model and all server contexts, so
  this version improves speed but does not reduce desktop RAM use.
- Existing shape-specific CPU repacking cannot directly repack arbitrary FFN
  submatrix views. The comparison therefore disables repacking in both arms.
  A packed host-prefix implementation is required before comparing against the
  fastest normal CPU configuration.
- Transport failure currently terminates the split path; there is no transparent
  CPU fallback for an in-flight request.
- Streaming completion, request queueing, 8 slots, continuous batching, unified
  KV, mixed prompt/decode physical batches, and sampling were exercised.
  Cancellation, context shifting, grammars, and speculative decoding were not.
- This is one paired full-trace run. Alternating repetitions are still needed
  before calling the gain statistically consistent.

## Artifacts

- Runner: `burstgpt_gpu_cpu_op15_v1/run_server_trace.py`
- CPU result on the RTX 4060 Ti host:
  `/home/zhihao/s41-dynamic-ffn-v1/server-traces/cpu-r1/RESULT.json`
- OP15 result on the RTX 4060 Ti host:
  `/home/zhihao/s41-dynamic-ffn-v1/server-traces/op15-r1/RESULT.json`
- Detokenized treatment outputs:
  `/home/zhihao/s41-dynamic-ffn-v1/server-traces/op15-r1/DETOKENIZED_COLD.json`
