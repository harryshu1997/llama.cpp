# Mixed-Workload Trace Contract

Status: authoritative trace-source and normalization contract as of 2026-07-17.

## 1. Honest trace policy

No public trace selected for this project contains all of these fields at once:

- real arrival timestamps;
- input and output demand;
- multiple AI service types and modalities;
- true request priority; and
- true deadline or SLO.

We therefore keep three labels separate:

- `real`: one source trace replayed without resampling its arrivals or sizes;
- `real_decomposed`: a real multi-stage request mapped to documented service
  stages; and
- `semi_synthetic`: real marginal streams superposed with a deterministic
  transform.

The single record-format authority is `schema_version`; there is no
`trace_version`. The mix provenance value is `semi_synthetic` (never
`semi_synthetic_mix`). See `spikes/s8_operator_island_affinity/SCHEMA_CONTRACT.md`.

Observed completion latency is never reused as a deadline. Doing so would leak
the original serving system into our policy.

## 2. Recommended sources

### T0: BurstGPT v2.0

Primary generation trace and S7 ragged/dynamic-batch input.

- Source: https://github.com/HPMLL/BurstGPT
- Release: https://github.com/HPMLL/BurstGPT/releases/tag/v2.0
- License: CC BY 4.0.
- Fields: timestamp, model, request tokens, response tokens, API/conversation
  class, session ID, elapsed time, and failures depending on file.

Use `BurstGPT_3.csv` for overload/failure studies and its no-failure variant for
pure compute replay. Preserve timestamps and row order. Select deterministic
low, median, high, and burst 15-minute windows. If time scaling is required for
the phone testbed, multiply every inter-arrival gap by one published factor; do
not independently resample requests.

### T1: Azure LLM Inference Trace 2024

Simple production fallback with distinct Code and Conversation services.

- Description: https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md
- License: CC BY Attribution.
- Fields: invocation timestamp, context tokens, generated tokens.

The Code and Conversation CSV clocks must be checked before merging. If their
epochs are not directly comparable, replay them as separate real scenarios or
superpose them only under the `semi_synthetic` label.

### T2: Azure LMM Inference Trace 2025

Primary real multimodal scenario.

- Description: https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md
- License: CC BY Attribution.
- Fields: invocation timestamp, number of images, context tokens, generated
  tokens.

Each request decomposes into a measured image-encoder island followed by LLM
prefill and decode. The trace does not expose prompt content or true deadline.

### T3: RAGPulse

Primary real multi-stage RAG scenario.

- Source: https://github.com/flashserve/RAGPulse
- Public trace: 7,106 requests from a one-week university Q&A service.
- Fields: timestamp, input/output token length, session ID, and privacy-safe
  hash IDs for system prompt, passages, history, web search, and user input.

The trace supports an embedding/retrieval/context/generation DAG and cache
locality experiments. It does not provide actual component execution times;
those must come from our measured profile atlas. Confirm dataset redistribution
terms before copying source data into this repository.

#### S38 executable distributed-RAG composition

S38 combines two independently pinned sources without presenting the result as
a real production trace:

- RAGPulse supplies observed arrival times, token demand, sessions, and cache
  identifiers;
- MultiHop-RAG supplies 2,556 executable questions and answers, 609 documents,
  and ground-truth evidence; and
- a deterministic URL-hash partition assigns each document to OP12 or OP15.

The result is labeled `semi_synthetic`. Each query fans out to both phones for
local embedding, retrieval, and reranking, followed by a deterministic global
top-k merge and a collaborative Gemma-4 reasoning route. Priority and deadline
remain null until matching physical stage profiles exist. The canonical bundle
is outside git at `rag_assets/composed/s38-v1`; its builder, exact hashes, and
validation result are under `spikes/s38_distributed_rag_trace/`.

### T4: Mooncake traces

Optional cache-aware conversation and tool-agent scenario.

- Source: https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces
- Repository license: Apache 2.0; confirm trace-specific redistribution terms.
- Fields include timestamps, input/output lengths, and remapped prefix-block
  hash IDs.

Use for prefix/KV locality, not as evidence of real priorities or deadlines.

### T5: TraceLab

Optional coding-agent scenario.

- Source: https://github.com/uw-syfi/TraceLab
- Release: https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.1
- Data license: CC BY 4.0; code license: Apache 2.0.
- Contains real Claude/Codex rounds, model/provider, prefix/new/output tokens,
  sessions, tool timing events, errors, and cache behavior.

### T6: local AEA-derived assistant trace

The Unifer workspace already contains a frozen mixed VLM, chat, ASR, detector,
and translation timeline plus replay tools under:

```text
/home/myid/zs89458/Documents/Unifer/research_dev/services/lazyvlm/eval/
```

This is valuable for a controlled live-assistant experiment. Its README and
source manifest must determine which events are recorded and which are
synthetically generated. Do not call the whole trace a production server trace.

## 3. Canonical request schema

Normalized traces use JSONL. All times are integer microseconds from the start
of the normalized window.

Note: the normative record shape is `spikes/s8_operator_island_affinity/
schemas/request.schema.json`; the block below is illustrative. The version key is
`schema_version` (single authority; there is no `trace_version`).

```json
{
  "schema_version": 1,
  "event_id": "source:row",
  "source": "burstgpt-v2",
  "provenance": "real",
  "t_us": 1200000,
  "service": "conversation_generation",
  "model_class": "large_text_generation",
  "session_id": "optional",
  "input_tokens": 512,
  "output_tokens": 96,
  "images": 0,
  "audio_ms": 0,
  "retrieved_chunks": 0,
  "cache_keys": [],
  "observed_latency_us": null,
  "priority_class": null,
  "deadline_us": null,
  "source_fields": {}
}
```

Required fields are version, event ID, source, provenance, timestamp, service,
and model class. Demand fields default to zero only when the source semantics
make zero meaningful. Unknown is represented by `null`, not silently by zero.

Every normalized file has a sidecar manifest:

```text
source URL and release/tag
source file name and SHA-256
license and citation
normalizer version/hash
filters and exclusion counts
window bounds and source row IDs
time scaling, if any
composition seed, if any
output row count and SHA-256
```

## 4. Service DAG mapping

The trace describes requests; the workload catalog maps them into islands.

```text
text generation:
  tokenize -> prefill -> repeated decode -> sample

multimodal generation:
  image decode/preprocess -> vision encode -> projection -> LLM prefill -> decode

RAG:
  query embed -> retrieval -> rerank -> context assembly -> LLM prefill -> decode

coding agent:
  prefill/decode -> tool wait -> optional tool-result append -> prefill/decode
```

CPU-only stages remain in the trace even when they are not phone candidates,
because they affect dependencies and server idle windows.

## 5. Deterministic mixed trace

No cross-service correlation is invented and presented as real. To build a
mixed server workload:

1. Select source windows by a committed deterministic rule.
2. Normalize each window to `t_us=0` while preserving internal order and gaps.
3. Apply one documented intensity multiplier per stream.
4. Offset streams using a fixed published seed.
5. Merge by `(t_us, source_rank, source_row_id)`.
6. Store all transforms in the sidecar manifest.

Recommended `mix-v1` lanes are:

- BurstGPT or Azure Code/Conversation generation;
- RAGPulse RAG requests;
- Azure LMM multimodal requests; and
- an optional TraceLab agent lane.

The headline report must also include each real source replay separately so a
result does not depend only on synthetic superposition.

### S14 initial execution freeze

The first S14 scheduler workload uses exactly two lanes:

1. BurstGPT v2.0 generation; and
2. RAGPulse RAG.

The source-component normalization gate passes, but the deterministic `mix-v1`
composition and structural mixed replay are not implemented yet. They are S14-
CP0, before scheduler or device results. The transform must follow the frozen S8
integer ordering and hash rules and label the output `semi_synthetic`. Each real
source replay remains a mandatory control.

RAGPulse identifies stages and demand but does not provide executable text or
measured component times. Synthetic payload text must be labeled synthetic, and
latency comes only from a matching measured profile. The preferred second phone
service is the BGE embedding/reranking funnel. It enters `mix-v1` execution only
after backend support, CPU-reference correctness, no-fallback, boundary, memory,
and latency gates pass. If it does not pass, S14 stops with
`TRACE_OR_SERVICE_BLOCKED`; a symbolic second model is not a real mixed-service
result.

## 6. Synthetic deadline and priority sensitivity

Public sources do not expose true priority or deadlines. When needed, define an
explicit experiment policy:

```text
predicted_solo_ms = server-only profile for the request class and shape
deadline = arrival + alpha[class] * predicted_solo_ms
priority = declared class mapping
```

Report every alpha, class fraction, and mapping. Sweep them. Never label these
fields as trace-provided.

## 7. Own trace generator fallback

If a source cannot be redistributed or does not cover a necessary service, the
project may generate a trace from empirical distributions. The generator must:

- fit arrival intensity by time window rather than assume stationary Poisson;
- preserve empirical input/output and session correlations where available;
- include deterministic burst and outage controls;
- use a fixed seed and versioned config;
- write the canonical JSONL plus manifest; and
- label every generated row `synthetic`.

The generated trace is a sensitivity tool, not a substitute for real-trace
headlines.

## 8. First trace gate

MW0 passes only when:

- at least one BurstGPT or Azure LLM real window normalizes deterministically;
- Azure LMM or RAGPulse provides a second real service DAG;
- repeated normalization produces byte-identical output;
- malformed, missing, negative, or overflowing demand fields fail closed;
- filtering and context-limit exclusions are reported rather than clipped;
- source licenses and checksums are recorded; and
- a server-only replay consumes the normalized file without reordering it.

Current status: the separate BurstGPT and RAGPulse component rows satisfy the
normalization/structural portion, but the mixed composition, executable second
service, and exact measured-profile coverage remain open. Therefore MW0/S14-CP0
is not complete.
