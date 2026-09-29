# S42 llama.cpp ubatch graph adapter V1

## Scope

`llama-ubatch-graph` submits one real mixed continuous batch through
`llama_decode`. It seeds seven independent decode sequences, appends one
prefill sequence, and records the backend-assigned ggml graph for every
physical ubatch through `cb_eval`.

The capture includes:

- actual physical token and graph-output shapes;
- every scheduled node, source root, tensor layout, byte count, and buffer
  type;
- logical decode/prefill spans and llama-context KV ownership;
- model dimensions and runtime backend identity.

`ubatch_graph_adapter.py` validates the capture and converts it into resident
allocation facts and structurally legal placement candidates. It recognizes
dense FFN, expert FFN, Q/K/V projections, attention output projections, and
the vocabulary projection. Normalization, RoPE, masking, softmax, elementwise
work, and unrecognized nodes stay local.

This stage does not choose a device. All candidates leave `placement_bound`,
`energy_bound`, and `runtime_route` false. Stage 3 must match them to measured
rows and compile an executable epoch-bound route.

## Build and run

```sh
cmake --build build-cpu --target llama-ubatch-graph -j8

build-cpu/bin/llama-ubatch-graph \
  --model MODEL.gguf \
  --device CUDA0 \
  --ubatch 32 \
  --prefill 65 \
  --decode-requests 7 \
  --decode-context 32 \
  -ngl 99 \
  --output ubatch-graph.json

python3 \
  research_dev/spikes/s42_general_energy_scheduler_v1/graph_adapter_v1/ubatch_graph_adapter.py \
  --input ubatch-graph.json \
  --output placement-graph.json
```

Both tools refuse invalid input. The Python adapter also refuses to overwrite
an existing output.

## RTX 4060 Ti validation

The physical validation used `n_ubatch=32`, 65 prefill tokens, seven decode
requests, and 32 seeded KV tokens per decode request. The deployed runtime
formed three physical ubatches:

| Physical ubatch | Tokens | Requested outputs | Graph outputs |
| ---: | ---: | ---: | ---: |
| 0 | 32 | 7 | 7 |
| 1 | 32 | 0 | 1 |
| 2 | 8 | 1 | 1 |

The one-row output in physical ubatch 1 is llama.cpp's graph-shape floor. It
does not add a requested token output.

| Model | Graph nodes per ubatch | Placement operators | Split options | Resident allocations |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 12B Q4_0 | 1,970 | 699 | 699 | 668 |
| Qwen3 14B Q4_K_M | 1,406 | 603 | 603 | 443 |

Gemma families were 144 dense FFNs, 144 Q, 144 K, 120 V, 144 attention
outputs, and three vocabulary projections. Qwen families were 120 of each
layer family plus three vocabulary projections. Counts include all three
physical ubatches.

Physical artifacts remain on the RTX 4060 Ti host under
`/home/zhihao/s42-kernel-energy-v1`:

| Artifact | SHA-256 |
| --- | --- |
| `llama-ubatch-graph-v3` | `cb80c12d915acdadb11e63bc5875a868f1eb6ec973f7e5477b88a2100a0dc2e4` |
| `gemma-ubatch-graph-mixed-v3.json` | `1203809693999d3f1117c09760ea90f66e5a50323407cd7ae0b08a38baf8fd00` |
| `gemma-placement-graph-mixed-v5.json` | `64d8f80c5e9bf58b18b0ff6662b4024c685fc2f1c692243000fbb18801c8b6a2` |
| `qwen-ubatch-graph-mixed-v1.json` | `d515fb7e645618bab8c17660e557a34bb5966590e75393efdc50c7d482cc177b` |
| `qwen-placement-graph-mixed-v2.json` | `15d926f6837f188d519192dfd143367ebb62ec166c270be02b8c22fe87a50a61` |

The raw captures are intentionally not checked into Git.
