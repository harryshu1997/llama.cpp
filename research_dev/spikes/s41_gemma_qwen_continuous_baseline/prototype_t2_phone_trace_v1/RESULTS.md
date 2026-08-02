# S41 T2 real-phone BurstGPT prototype

Status: `BURSTGPT_T2_PHONE_TRACE_PROTOTYPE_PASS`

## Scope

This is a bounded prototype, not an S41 qualification or paper-final
treatment. It replays the exact versioned 74-request BurstGPT-derived trace:
57 Gemma 4 12B IT Q8_0 requests, 17 Qwen3 14B Q4_K_M requests, eight output
tokens per request, and a 30 second synthetic SLO.

Gemma remains continuously resident on RTX 4060 Ti CUDA0 with eight slots and
continuous batching. Qwen remains continuously resident on the OP15
`[0,30)` plus OP12 `[30,40)` OpenCL route. The nine switch records are
hash-bound but are not executed. This is therefore
`T2_PHONE_NO_PROMOTION_PROTOTYPE`, not a phone-to-CUDA handoff, GPU model
switch, or bidirectional Gemma/Qwen cycle.

## Result

The real-device run completed all 74 arrivals and all 592 requested output
tokens.

| Route | Completed | SLO met | P95 TTFT | P95 completion |
| --- | ---: | ---: | ---: | ---: |
| Gemma on RTX 4060 Ti | 57/57 | 57/57 | 1.029 s | 1.842 s |
| Qwen on OP15 plus OP12 | 17/17 | 0/17 | 57.436 s | 89.913 s |

The paid trace duration was 124.376 seconds and aggregate throughput was
4.760 output tokens/s. Overall SLO goodput was 0.466 requests/s. The phone
wire ledger contains 57 physical batches, 1,982 rows, a maximum batch of 64,
and a mean batch size of 34.772 rows.

For context, the existing warm-cache all-server C1 median completes the same
trace at 9.818 output tokens/s, 1.227 SLO-good requests/s, 5.097 second P95
TTFT, and 5.360 second P95 completion. The prototype is therefore a transport
and concurrent-service success but not a capacity or latency win. It is not
resource-equivalent to C1: it pins Gemma on the GPU for the entire trace and
never promotes Qwen.

Gemma used full CUDA placement, reported zero process swap, and used
14,676,918,272 GPU bytes at readiness. During the active phone route, OP12
reported about 3.13 GiB RSS and 43 MiB process swap; OP15 reported about
3.08 GiB RSS and 44 MiB process swap. This prototype did not freeze a
no-phone-swap gate and cannot be used as phone capacity qualification.

## Transport finding

The first paid prototype attempt failed after completing the Gemma work.
Concurrent StageNet batches and sequence-removal commands used one socket
without common serialization. Replies interleaved, producing one status
version mismatch, worker status 3 errors, leaked sequence credits, and later
admission refusals.

The rerun used a prototype-only serialized StageNet client wrapper. No model,
trace, arrival, batching, sampling, or threshold input changed. The rerun
completed without protocol or sequence-credit errors and all CUDA and phone
processes cleaned to zero. This is evidence for a required core fix, not the
final fix itself. The shared phone gateway needs one serialized protocol
owner and credit-aware waiting or scheduler admission, with a concurrent
batch-plus-remove regression test.

## Claim boundary and next step

This result proves that the versioned trace can concurrently drive a real
full-CUDA large model and a real two-phone sharded large model. It does not
prove promotion, replay, exact ownership transfer, energy savings, or
bidirectional model switching. It also does not make Gemma phone-eligible.

The next bounded prototype should retain Gemma on CUDA, let phones serve the
first Qwen work while CUDA loads Qwen, use path-matched token-history replay
with `k_extra=0`, transfer ownership once, and finish Qwen on CUDA. Compare it
to the existing warm-cache GPU-switch control. Do not run the full trace
matrix until that one-way handoff passes. A full Gemma/Qwen reverse cycle
remains blocked by Gemma phone capacity and correctness.

## Evidence

Passing run on the acquisition desktop:

`/home/zhihao/s41-t2-phone-trace-prototype-v1/run_20260727T042800Z`

- `RESULT.json`: SHA-256
  `2197315a4b88c9a39b7f0692e9bd3bdbc67a967c1e6fd76e2ccc017751eced25`
- `trace-events.jsonl`: SHA-256
  `8892c0c43d8d4c5329b4306539ba216e21ba8166716741757806d440f28cd20d`
- `phone-wire.jsonl`: SHA-256
  `40083d26dbfe692dfa75df56f12789ad1c1fed046852be959823daea440dc408`
- runner source: SHA-256
  `c6c2d8d69ec89e6c798d53d6cbceb34630415ecbcb6ded69debfcf328d56cfe2`

The immutable requests, switches, and input-manifest SHA-256 values are
`94c36fe3ac43281dc0c83a29a1519c7e72ac445ed9ded98d474b2231041c1735`,
`66d07fdeeb594ee3160fcd07e25a3780414e13ec12ad1af97da31b1febfe6ba1`,
and `8a3d2b59f1de9f4d9a4b64b27cf044977a1a9ddeed87bc848d992da01e575df0`.

The failed transport-race attempt is preserved at
`/home/zhihao/s41-t2-phone-trace-prototype-v1/run_20260727T041700Z`;
its `FAILURE.json` SHA-256 is
`490dc5512b9bc071bfe57f2a7f7594e3898a041f68b037bb911ea8579f5d88b3`.

