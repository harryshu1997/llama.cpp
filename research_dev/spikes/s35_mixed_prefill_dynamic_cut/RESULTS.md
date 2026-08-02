# S35 Results

## Verdict

`MIXED_BATCH_AND_DYNAMIC_EXIT_MECHANICS_PASS`

The phone runtime now has a measured mixed prefill/decode batch and the
LayerSplit graph can select a request-pinned cut inside an already resident
weight interval. A phone-to-CUDA pipeline executed two different cuts without
reloading either worker.

This is a mechanics and same-route token screen. It is not an energy result,
an SLO policy result, or a production `llama-server` integration.

## CP1: one phone batch contains prefill and decode

The frozen B=5 treatment was one decode row for live request A at position 4
plus four prompt rows for newly admitted request B at positions 0 through 3.
The serial oracle seeded A with four prompt rows and then issued its position-4
decode as B=1. The first exploratory run used an invalid five-row prompt oracle
and is excluded; it directly motivated this correction.

| Device | Physical batch | Mixed call | Maximum relative L2 | Minimum cosine | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| OP15 | 1 decode + 4 prefill | 178.620 ms | 4.03e-7 | 0.9999999999999 | pass |
| OP12 | 1 decode + 4 prefill | 76.500 ms | 7.00e-7 | 0.9999999999998 | pass |

Both workers retained one PID across the serial and treatment sessions. Both
reported `SCHEDULED_PLACEMENT_OK` and zero missing-buffer compute nodes. OP12
used HTP0 for all compute except token lookup on CPU. OP15 used HTP0 plus its
known CPU-side token lookup and projection support nodes. Therefore the claim
is phone-local mixed batching, not NPU-only execution.

Artifacts:

- `results/phones_20260722T063300Z/op12.json`
- `results/phones_20260722T063300Z/op15.json`
- matching worker logs in the same directory

## CP2: request-pinned active layer interval

StageNet gained an opt-in range-batch opcode. The worker still loads one fixed
resident interval. Every range batch selects a subinterval, and the sequence
table pins that interval until removal. Rows with different cuts cannot share
one physical graph batch.

Each phone loaded `[0,2)` once and executed `[0,1)`, `[0,2)`, and `[0,1)`:

| Device | Shallow repeat relative L2 | Shallow/deep relative L2 | Live cut change | Result |
| --- | ---: | ---: | --- | --- |
| OP15 | 0 | 1.06448 | rejected | pass |
| OP12 | 0 | 1.06352 | rejected | pass |

The cut-dependent activations were different, the repeated shallow activation
was bit-identical, and the worker rejected a cut change on an already-live
sequence while keeping the connection usable.

Artifacts:

- `results/cuts_20260722T062805Z/op12.json`
- `results/cuts_20260722T062805Z/op15.json`
- matching worker logs in the same directory

## End-to-end dynamic exit

One resident OP12 worker held F16 layers `[0,8)`. One resident A6000 worker held
F16 layers `[4,48)`. Two requests used different complete routes:

1. OP12 `[0,4)` -> A6000 `[4,48)`
2. OP12 `[0,8)` -> A6000 `[8,48)`

Each request kept its selected route for its four-row prefill and two decode
steps. Both routes returned the same generated token sequence:

`[236761, 236744, 236761]`

| Route | Prefill head + tail | Decode 1 head + tail | Decode 2 head + tail |
| --- | ---: | ---: | ---: |
| cut 4 | 246.179 ms | 116.959 ms | 112.373 ms |
| cut 8 | 200.392 ms | 202.980 ms | 204.486 ms |

This timing is a single mechanics run and is not a performance comparison.
The phone used a stage shard while CUDA used the full GGUF, so the files do not
share one whole-file digest. The equal tokens are a same-model route screen,
not a monolithic-reference certificate.

Artifacts:

- `results/pipeline_20260722T063500Z/report.json`
- `results/pipeline_20260722T063500Z/op12.log`
- `results/pipeline_20260722T063500Z/tail.log`

## Verification

- S35 Python tests: 9/9 pass.
- StageNet V3 client tests: 14/14 pass.
- Host CUDA build: pass.
- Host CPU build: pass.
- Android Hexagon/OpenCL build: pass.
- Two ASan/UBSan builds: pass.
- CPU ASan/UBSan dynamic-cut execution: pass with no sanitizer diagnostic.
- `git diff --check`: pass.

The final compatibility review made the range setter fail closed on
non-Gemma-4 models and kept ordinary non-dynamic V3 batches on their existing
path. The final Android binary and `libllama.so` have SHA-256 values
`c81e692f...b3496a2` and `81bb22a2...0c76e`; the final mixed, cut, and pipeline
runs above all used those exact files. The final CUDA tail binary and
`libllama.so` have SHA-256 values `57e12992...a23e9` and
`074cad1d...6b761`.

## Implementation boundary

- Existing V3 messages are unchanged.
- Dynamic cuts are advertised only when `LAYERSPLIT_DYNAMIC_CUT` is set.
- The loaded weight interval remains controlled by `LLAMA_LAYER_START/END`.
- The active interval is part of graph-reuse identity.
- Head intervals must start at zero; terminal intervals must end at the final
  model layer; every active interval must be inside resident weights.
- A physical batch has one cut. A scheduler must group requests by cut.

## Not yet claimed

- no energy or throughput benefit from dynamic cuts;
- no mixed-phase B32 run;
- no SLO policy selecting the cut from an arrival trace;
- no simultaneous mixed-phase batch with multiple cut groups;
- no integration into the upstream server slot scheduler;
- no total-system energy accounting.
