# Six-model mixed BurstGPT trace V1

## Result

`REQUESTS_MIXED_114.jsonl` is a deterministic 114-request successor to the
74-request source-length BurstGPT trace. It keeps every source event, arrival,
prompt token sequence, output length, and SLO, then interleaves four additional
ten-request streams over the same arrival window.

The trace SHA-256 is:

```text
0622e5fe39d0f6f8bed8e0602680e5ed35941eeb571e686f235fa740ef61fae3
```

| Execution model | Requests | Artifact bytes | Workload |
| --- | ---: | ---: | --- |
| Qwen3-14B Q4_K_M | 57 | 9,001,752,960 | Original large text route |
| Gemma4-12B Q4_0 | 17 | 6,975,878,176 | Original large text route |
| Qwen3-0.6B Q8_0 | 10 | 804,753,632 | Tiny text overlay |
| Llama-3.2-1B Q4_0 | 10 | 770,928,288 | Small text overlay |
| Qwen3-8B Q8_0 | 10 | 8,709,518,112 | Medium text overlay |
| Gemma4-E2B Q8_0 plus projector | 10 | 5,524,862,816 | VLM overlay |

The trace contains 49,368 text input tokens and 16,506 requested output
tokens. Its first arrival is 1.950 seconds, its last arrival is 58.100 seconds,
and its arrival span is 56.150 seconds.

## Construction

The source trace is unchanged at:

```text
research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1/REQUESTS_SEMANTIC_SOURCE.jsonl
```

Its SHA-256 remains
`b20a9ba66ee3558d835a0e19ed3cfa4c31a4a9e8b4f9c085b29a14f80250a0ff`.
Each successor source row carries the hash and schema of its exact parent.
The validator checks every inherited source field against the parent row.

The three text overlays use source request indices
`0, 8, 16, 24, 32, 41, 49, 57, 65, 73` as matched input and output geometry.
Their arrivals are offset from the donor by 100, 200, and 300 milliseconds.
The VLM requests are offset by 400 milliseconds. This makes all six models
contend during the same workload rather than running separate phases.

The exact Qwen3-8B artifact on the RTX 4060 Ti was checked against all ten
materialized Qwen3 overlay prompts. Its token sequences matched the local
Qwen3-0.6B tokenizer sequences exactly. The check used the bound Qwen3-8B
artifact SHA-256 from `TRACE_MANIFEST.json`.

## VLM stream

The VLM stream uses three existing, hash-pinned repository images:

- a Moon-landing newspaper page;
- a lotus flower; and
- an Android Studio screenshot.

The images repeat across ten visual questions. This deliberately exposes both
cold image encoding and possible image-embedding cache reuse. Each case has a
normalized answer-group scorer. Image byte and pixel counts are present, but
the model-specific image token count remains `runtime_measured`; the trace does
not invent a patch-token value before the exact multimodal runtime processes
the image.

Text requests use `prompt_transport=tokens`. VLM requests instead carry the
raw visual question with `prompt_transport=multimodal_message`. The separate
formatted prompt text is diagnostic only; a runner must not send it through a
chat template a second time.

## Reproduce and verify

From the repository root:

```sh
python3 research_dev/spikes/s42_general_energy_scheduler_v1/mixed_model_trace_v1/build_mixed_trace.py
python3 research_dev/spikes/s42_general_energy_scheduler_v1/mixed_model_trace_v1/verify_mixed_trace.py
python3 research_dev/spikes/s42_general_energy_scheduler_v1/tests/test_mixed_model_trace.py
```

The builder verifies the local codec model and image hashes before deriving
the trace. If the trace or manifest already exists, it compares bytes and
refuses a different derivation.

## Boundary

This result materializes and validates the workload only. It does not claim
that all six physical routes are qualified, resident, energy beneficial, or
connected to the current two-model physical runner. Route profiling, runtime
model residency, VLM image-token capture, and the matched physical baselines
remain separate gates.

The live-plan V3 control and adaptive arms send all 114 requests through one
`research_dev.scheduler.UnifiedScheduler` instance. Executor bindings are
derived from the control result's actual backend, route, and server slot count;
model residency bytes come from `TRACE_MANIFEST.json`.

The GPU model order is the output of the canonical residency planner, not a
route table in the runner. The planner currently has one qualified measured
transition chain, Qwen14 -> Qwen8 -> Gemma12. Missing transition directions
fail closed. The physical runner recomputes this decision from the hash-bound
residency problem before executing it, then admits each promoted model's
requests using its actual ready timestamp.

The control profile exposes only Llama's measured CPU route. The adaptive
profile exposes its measured CPU and resident-phone routes. The other five
models have one evidence-backed route each, so they receive real queue and
resource-lease decisions but not a device choice. This is full-trace scheduler
coverage, not full-trace placement freedom.
