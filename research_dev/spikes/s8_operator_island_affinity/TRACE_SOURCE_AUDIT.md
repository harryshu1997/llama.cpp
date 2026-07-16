# Trace Source Audit (S8-V0a-R2)

R2 notes: single version authority is `schema_version` (no `trace_version`); the
mix provenance value is `semi_synthetic`. Exact per-source PARSING (encoding,
BOM/newline, CSV dialect, column roles, integer/timestamp grammar, monotonicity,
failure/exclusion behavior) is FROZEN in NORMALIZATION_SPEC.md section 2, with the
`source_revision` pinned in the normalization config and the header verified
byte-for-byte at parse time. Normalized outputs are bound to the full hash set
(source + revision + config + code + output + sidecar) per NORMALIZATION_SPEC
section 11.

Status: inspection only. No dataset was downloaded. This file records, for each
candidate source, the primary URL, release/tag, license, schema/fields,
approximate size, and the checksum procedure, and states which fields are
directly OBSERVED versus which priority/deadline fields would be explicitly
SYNTHETIC. It is the provenance contract for the MW0 normalizer; it does not
implement the normalizer.

Authority: this file and `WORKLOAD_TRACES.md` are INFORMATIVE background. The
single authoritative contract is, highest first, `schemas/` + `configs/`, then
`NORMALIZATION_SPEC.md`, then `SCHEMA_CONTRACT.md` (see SCHEMA_CONTRACT section 0).
Neither this file nor `WORKLOAD_TRACES.md` overrides those; the earlier
"`WORKLOAD_TRACES.md` wins" precedence is withdrawn. The EXACT pinned sizes,
SHA-256, commits, and column facts are now recorded in `SOURCE_PINS.md` and the
per-source configs (no longer UNKNOWN).

## 0. Honest-trace policy (restated)

No selected public trace contains all of {real arrival time, input/output
demand, multiple service classes/modalities, true priority, true deadline}.
Three provenance labels are kept separate and never mixed silently:

- `real`: one source replayed without resampling arrivals or sizes;
- `real_decomposed`: a real multi-stage request mapped to its documented DAG;
- `semi_synthetic`: real marginal streams superposed by a deterministic,
  seeded transform.

Observed completion latency is NEVER reused as a deadline (that would leak the
origin serving system into our policy).

## 1. Generation class -- BurstGPT v2.0 (primary), Azure LLM 2024 (fallback)

### T0 BurstGPT v2.0
- Primary URL: https://github.com/HPMLL/BurstGPT
- Release: https://github.com/HPMLL/BurstGPT/releases/tag/v2.0
- License: CC BY 4.0 (attribution).
- Files of interest: `BurstGPT_3.csv` (the largest / overload window) and the
  documented no-failure variant (pure compute replay).
- Schema / fields (CSV): timestamp, model, request tokens, response tokens,
  API/conversation class, session ID, elapsed time (fields present vary by file).
- CORRECTED size (review finding 9): `BurstGPT_3` is roughly **220 MB with about
  5.34 million rows** (per the BurstGPT "Main Characteristics"), NOT tens of MB.
  It is a large single file; deterministic 15-minute window selection
  (NORMALIZATION_SPEC section 3 `quantile_window`) is REQUIRED so V0b never needs
  the whole file resident. Exact bytes + SHA-256 recorded in the sidecar at fetch.
- CORRECTED failure semantics (review finding 9): BurstGPT has **no separate
  failure column**; a failed request is represented by **zero response tokens**.
  The normalizer keeps such rows as `output_tokens:0` and flags
  `source_fields.burstgpt_failed:true`; a "no-failure" (pure compute) window is
  produced only by an EXPLICIT `exclude_zero_output` filter whose count is
  reported (NORMALIZATION_SPEC section 4-5). Zero response tokens are never
  silently dropped.
- Observed fields -> canonical schema: `t_us` (from timestamp), `input_tokens`
  (request tokens), `output_tokens` (response tokens), `service`/`model_class`
  (API vs conversation class), `session_id`.
- Explicitly SYNTHETIC (labeled): `priority_class`, `deadline_us`. The
  BurstGPT "elapsed time" column is NEVER converted into a deadline.
- Reference: BurstGPT Main Characteristics,
  https://github.com/HPMLL/BurstGPT#main-characteristics.
- Windowing: select deterministic low, median, high, and burst 15-minute
  windows; preserve timestamps and row order. If time-scaling is needed for the
  phone testbed, multiply every inter-arrival gap by one published factor; do
  not independently resample requests. Record the factor in the manifest.

### T1 Azure LLM Inference Trace 2024 (fallback)
- Description URL:
  https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md
- License: CC BY (attribution).
- Fields: invocation timestamp, context tokens, generated tokens; two services
  (Code, Conversation) in separate CSVs.
- Approximate size: small (order of MB). Confirm at fetch.
- Observed -> canonical: `t_us`, `input_tokens` (context), `output_tokens`
  (generated), `service` (Code vs Conversation).
- Caveat: the Code and Conversation CSV clocks must be checked before merging.
  If epochs are not directly comparable, replay them as separate `real`
  scenarios or combine only under `semi_synthetic`.
- Explicitly SYNTHETIC: `priority_class`, `deadline_us`.

## 2. Second real DAG -- RAGPulse (RAG) [primary second source]

### T3 RAGPulse
- Primary URL: https://github.com/flashserve/RAGPulse
- Public trace: ~7,106 requests from a one-week university Q&A service.
- License / redistribution: confirm dataset redistribution terms in the repo
  before copying any source rows into this tree. Keep source data outside git.
- Fields: timestamp, input/output token length, session ID, and privacy-safe
  hash IDs for system prompt, passages, history, web search, and user input.
- Approximate size: small (thousands of rows). Confirm at fetch.
- Service DAG (`real_decomposed`): query embed -> retrieval -> rerank -> context
  assembly -> LLM prefill -> decode.
- Observed -> canonical: `t_us`, `input_tokens`, `output_tokens`, `session_id`,
  `cache_keys` (from the hash IDs for prefix/passage/history locality).
- IMPORTANT: RAGPulse does NOT provide per-component execution times. Those come
  only from our measured profile atlas; the trace supplies arrivals, demand, and
  cache-locality structure, never component latencies.
- Explicitly SYNTHETIC: `priority_class`, `deadline_us`.

## 3. Encoder / vision class -- Azure LMM 2025 (primary), local AEA audio/ASR (alt)

### T2 Azure LMM Inference Trace 2025 [encoder/vision source]
- Description URL:
  https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md
- License: CC BY (attribution).
- Fields: invocation timestamp, number of images, context tokens, generated
  tokens.
- Approximate size: small-to-moderate (order of MB). Confirm at fetch.
- Service DAG (`real_decomposed`): image decode/preprocess -> vision encode ->
  projection -> LLM prefill -> decode. The vision-encode stage is the concrete
  encoder/background island for the atlas.
- Observed -> canonical: `t_us`, `images`, `input_tokens` (context),
  `output_tokens` (generated).
- Not exposed: prompt content, true deadline, image pixels. The image-encoder
  island's compute is sized from `images` + a measured per-image encoder profile
  in the atlas, not from the trace.
- Explicitly SYNTHETIC: `priority_class`, `deadline_us`.

### T6 local AEA-derived assistant trace (alternate encoder/audio source)
- Location:
  `/home/myid/zs89458/Documents/Unifer/research_dev/services/lazyvlm/eval/`
- Content: a frozen mixed VLM, chat, ASR, detector, and translation timeline
  plus replay tools. Provides an audio/ASR and detector encoder class.
- Provenance caveat: its README and source manifest MUST determine which events
  are recorded versus synthetically generated. It is NOT a production server
  trace and must be labeled by its actual recorded/synthetic provenance per
  event. Use only if Azure LMM cannot be normalized reproducibly, and label
  honestly.

## 4. Optional cache-locality sources (reference only for V0)

- T4 Mooncake: https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces
  (repo Apache 2.0; confirm trace-specific redistribution). Prefix/KV locality
  only; no real priorities/deadlines.
- T5 TraceLab: https://github.com/uw-syfi/TraceLab (release v0.0.1; data CC BY
  4.0, code Apache 2.0). Coding-agent rounds with tool timing. Reference for an
  agent lane; not required for V0.

## 5. Checksum and manifest procedure (applies to every source)

No source data is committed to this repo. For each fetched file the MW0
normalizer writes a sidecar manifest containing:

1. source URL and release/tag;
2. source file name and its SHA-256 (computed with `sha256sum <file>` or
   `hashlib.sha256` over the raw bytes at fetch time);
3. license and citation;
4. normalizer version/hash;
5. filters and exclusion counts (context-limit exclusions REPORTED, never
   silently clipped);
6. selected window bounds and the source row IDs included;
7. time-scaling factor, if any;
8. composition seed, if any (mix only);
9. output row count and the SHA-256 of the normalized JSONL.

Determinism gate: running the normalizer twice on the same input must produce
byte-identical JSONL and byte-identical manifest hashes. Only a small
license-compatible fixture is kept in git for tests; multi-GB data is never
added.

Hash binding (review requirement). A normalized JSONL file is trusted ONLY
alongside a sidecar (`schemas/trace_manifest.schema.json`) that carries BOTH:
`source_sha256` (SHA-256 of the raw source bytes) and `output_sha256` (SHA-256 of
the canonical JSONL bytes, per SCHEMA_CONTRACT section 2). The exact serialization
and hashing algorithm is frozen in NORMALIZATION_SPEC section 11. A file whose
bytes do not hash to its sidecar `output_sha256`, or a sidecar whose source hash
does not match the fetched file, is rejected fail-closed. This binds every
downstream atlas/oracle input to an auditable (source, normalizer_version,
output) triple.

## 6. Observed vs synthetic field summary

| canonical field | BurstGPT | Azure LLM24 | RAGPulse | Azure LMM25 |
|---|---|---|---|---|
| t_us (arrival) | observed | observed | observed | observed |
| input_tokens | observed | observed | observed | observed |
| output_tokens | observed | observed | observed | observed |
| service / model_class | observed | observed | observed (DAG) | observed (DAG) |
| session_id | observed | UNKNOWN | observed | UNKNOWN |
| images | 0 | 0 | 0 | observed |
| cache_keys | UNKNOWN | UNKNOWN | observed (hashes) | UNKNOWN |
| retrieved_chunks | 0 | 0 | observed/derivable | 0 |
| observed_latency_us | present but UNUSED as deadline | absent | absent | absent |
| priority_class | SYNTHETIC (labeled) | SYNTHETIC | SYNTHETIC | SYNTHETIC |
| deadline_us | SYNTHETIC (labeled) | SYNTHETIC | SYNTHETIC | SYNTHETIC |

Synthetic deadline policy (from `WORKLOAD_TRACES.md` section 6, restated as a
contract): `deadline_us = arrival + alpha[class] * predicted_solo_us`, where
`predicted_solo_us` comes from an independently measured server-only service
curve. Every `alpha`, class fraction, and mapping is reported and swept; these
fields are always labeled synthetic and never presented as trace-provided.

## 7. Selection for MW0 (recommendation, pending human review)

- Real generation window: BurstGPT v2.0 (`BurstGPT_3` no-failure variant),
  deterministic 15-minute median + burst windows.
- Second real DAG: RAGPulse (RAG) for the retrieval/rerank/generation DAG and
  cache-locality experiments.
- Encoder/vision: Azure LMM 2025 image-encoder island.
- Deterministic `semi_synthetic` (`mix-v1`): superpose the three real
  streams with published per-stream intensity multipliers and a fixed seed;
  always also report each real source replay separately.

This selection satisfies "two real scenarios plus one labeled mixed
composition" (S8 Gate A) with two distinct service DAGs (generation + RAG) and a
distinct encoder class (vision). Redistribution terms for RAGPulse and any
trace-specific Azure terms are confirmed before any source bytes are copied.
