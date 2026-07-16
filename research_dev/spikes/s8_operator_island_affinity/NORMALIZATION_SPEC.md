# Deterministic Normalization Spec (S8-V0b-P0)

Status: frozen Gate-A normalization semantics. Exact enough that two independent
implementations produce byte-identical JSONL and identical `output_sha256`. P0
pins the two real sources (SOURCE_PINS.md), moves the exact per-source parsing +
mappings into versioned configs (`configs/burstgpt.config.json`,
`configs/ragpulse.config.json`, validated by `schemas/source_config.schema.json`),
replaces decimal quantiles with rational-integer arithmetic, adds a per-source
timestamp policy (BurstGPT `require_nondecreasing`; RAGPulse `sort_stable`, since
inspection found 1 non-monotonic pair), and fully freezes the mixed-trace formula.

Authority (single contract; no cycles): the machine-checked artifacts win, in
order -- (1) `schemas/*.schema.json` + `configs/*.config.json` for structure and
values; (2) this file for the algorithm; (3) design/audit prose
(MIXED_WORKLOAD_DESIGN, WORKLOAD_TRACES, TRACE_SOURCE_AUDIT) is INFORMATIVE and
never overrides (1)-(2). Single version authority: `schema_version` (no
`trace_version`). Single mix provenance value: `semi_synthetic`.

## 1. Pipeline (per real source)

```text
raw source file (outside git)
  -> [1] hash source bytes -> source_sha256; record pinned source_revision
  -> [2] verify header line == pinned committed header (else fail closed)
  -> [3] parse rows (source-specific reader, no reordering)
  -> [4] convert timestamps to integer us; check monotonicity (no tolerance)
  -> [5] compute per-bin offered-token load; select the scenario window (bins)
  -> [6] per-row field derivation -> canonical request objects
  -> [7] exclusion filtering (report counts; never silent clip)
  -> [8] time rebase to bin start + exact integer scaling
  -> [9] deterministic total-order sort
  -> [10] canonical JSONL serialization -> output bytes
  -> [11] hash output; write sidecar manifest with the full binding set
```

Every step is a pure function of (source bytes, committed normalization config).
No wall clock, no locale, no hash-map iteration order, no floating time, no PRNG.

## 2. Source-specific parsing contract

Exact header column names are PINNED to the committed source revision and
verified byte-for-byte at parse time; a header mismatch fails closed. (We do not
fetch datasets in R2, so the pinned header string is finalized at first fetch and
committed into the normalization config; the parser never guesses.)

Common rules (both sources):
- Encoding: UTF-8. A UTF-8 BOM at file start is REJECTED (fail closed).
- Newlines: LF or CRLF accepted and normalized to LF for field parsing; a lone CR
  (0x0D not followed by 0x0A) is REJECTED.
- CSV dialect: RFC 4180 -- comma delimiter, `"` quoting, `""` escapes a quote, no
  comment character. A blank line is REJECTED (not skipped). Field whitespace is
  significant and NOT trimmed.
- Integer grammar: token/count fields match `^[0-9]+$` (no sign, decimal point, or
  exponent); anything else fails closed.
- Timestamp grammar: epoch-seconds fields match `^[0-9]+(\.[0-9]+)?$`. Conversion
  to microseconds is exact decimal: `us = round_half_up(Decimal(field) * 1000000)`
  computed on the exact decimal string (no binary float). ISO-8601 timestamps, if
  a source uses them, are parsed as UTC with a committed format string to epoch us
  by the same exact rule. Overflow: `us > 9007199254740991` fails closed.
- Timestamp order policy (PER SOURCE, `timestamp.policy` in the config):
  - `require_nondecreasing` (BurstGPT): source timestamps MUST be non-decreasing
    in source row order; a strictly-decreasing timestamp fails closed with the
    offending `source_row_id`. NO tolerance. Inspection: BurstGPT_3.csv has 0
    inversions, so this policy holds.
  - `sort_stable` (RAGPulse): the source is treated as an arrival SET; the reader
    applies a deterministic STABLE sort by `(t_us, source_row_id)` and records the
    count of decreasing adjacent input pairs in the sidecar
    (`window.input_nonmonotonic_pairs`, and `window.timestamp_policy`). Inspection:
    RAGPulse `data/0_trace.jsonl` has exactly 1 non-monotonic pair, which is why a
    single global `require_nondecreasing` rule would (wrongly) reject it. The
    stable sort is deterministic, so the output is still byte-reproducible.
  The semantic validator enforces `input_nonmonotonic_pairs > 0 => policy ==
  sort_stable`. Equal timestamps are always ordered by `source_row_id`
  (section 9).
- Row identity: `source_row_id` = 0-based index after the header, before any
  filtering; `event_id = "<source>:<source_row_id>"`.

The exact frozen facts are in SOURCE_PINS.md and the per-source configs; the two
subsections below are a human summary and defer to the config on any detail.

### 2.1 BurstGPT v2.0 (see `configs/burstgpt.config.json`)
- Pinned: release `v2.0` asset `BurstGPT_3.csv`, 231682327 bytes, sha256
  `2299986a...a43b8f`, 5344021 data rows, CC-BY-4.0 (SOURCE_PINS.md section 1).
- Exact header (8 columns) pinned in the config; `Timestamp` (float seconds) ->
  `t`, `Request tokens` -> `input_tokens`, `Response tokens` -> `output_tokens`,
  `Log Type` -> `service` (`Conversation log`->`conversation_generation`,
  `API log`->`api_generation`), `Model` -> `model_class` (`GPT-4`/`ChatGPT`->
  `large_text_generation`), `Session ID` -> `session_id` (BLANK on all API-log
  rows -> `null`). `retrieved_chunks:0`, `cache_keys:[]`. `Elapsed time`,
  `Total tokens`, `Model`, `Log Type` are kept verbatim in `source_fields`.
- Failure semantics: `Response tokens == 0` (387963 rows, 7.26%) is KEPT as
  `output_tokens:0` with `source_fields.burstgpt_failed:true`; a no-failure replay
  is a SEPARATE sensitivity config using the explicit `exclude_zero_output` filter
  whose count is reported. Never the default.

### 2.2 RAGPulse (see `configs/ragpulse.config.json`)
- Pinned: commit `99a62769a91d5ebd17a2d4ddbbc88c1d16edc0e8`, file
  `data/0_trace.jsonl`, 1923473 bytes, sha256 `cd371571...801e65`, 7106 records,
  MIT (SOURCE_PINS.md section 2).
- `timestamp` (integer seconds string) -> `t`; `input_length` -> `input_tokens`;
  `output_length` -> `output_tokens`; `session_id` (never blank) -> `session_id`;
  `service` const `rag_qa`, `model_class` const `rag`.
- `retrieved_chunks = len(hash_ids.passages_ids)`. `cache_keys` is the namespaced
  concatenation, in the committed key order `[sys_prompt, passages_ids, history,
  web_search, user_input]`, of `"{key}:{id}"` for each int id in each list
  (in-list order preserved). `passages_ids`/`web_search` may be empty.
- RAGPulse supplies arrival/length/hash-locality ONLY (no query/document text);
  payload fixtures for the embedding funnel are labeled synthetic
  (EMBEDDING_MODEL_FUNNEL.md section 3.2).

## 3. Aligned load-metric windows (replaces row-index quantiles)

The scenario windows are aligned time bins selected by an offered-token load
metric, not by row-index quantiles.

- Bin width: `W = 900000000` us (15 minutes).
- Bin alignment: aligned to the SOURCE time origin (source time 0 after
  conversion). A request at source time `s_us` has bin index `k = floor(s_us / W)`;
  bin `k` covers `[k*W, (k+1)*W)`. Alignment is data-independent.
- Load metric: `offered_tokens`. `L_k = sum over requests in bin k of
  (input_tokens + output_tokens)`. Both terms are observed integers.
- Non-empty bins: `B = { k : count_k >= 1 }`, `N = |B|`.
- Total order on bins: `S = B sorted ascending by (L_k, k)`. `k` is unique so the
  order is total.
- Quantile convention: NEAREST-RANK with RATIONAL-INTEGER arithmetic (no floats).
  A quantile is a committed integer pair `(q_num, q_den)` with `0 < q_num <=
  q_den`. The 1-based rank is `r = ceil(q_num * N / q_den)` computed as integer
  `r = (q_num * N + q_den - 1) // q_den`, clamped to `[1, N]`; the quantile bin is
  `S[r-1]`. Committed selectors:
  - low = `(1, 10)`  -> `r = (N + 9) // 10`
  - median = `(1, 2)` -> `r = (N + 1) // 2`
  - high = `(9, 10)` -> `r = (9*N + 9) // 10`
  No `0.10 / 0.50 / 0.90` float appears anywhere.
- Burst: `burst_k = argmax_{k in B} L_k`, ties broken by smallest `k`.
- Overlap: each scenario is one whole 15-minute bin; bins are disjoint in time by
  construction. Two SELECTORS may resolve to the same bin (small `N`); this is
  ALLOWED and recorded (each scenario records its `bin_index` + `metric_value`),
  not an error. A "four distinct bins" requirement, if needed, is a separate
  committed constraint, not the default.
- Manifest: each scenario's `window` records `rule`, `load_metric:"offered_tokens"`,
  `scenario`, `bin_index k`, `t_start_us = k*W`, `t_end_us = (k+1)*W`,
  `metric_value = L_k`, `n_nonempty_bins = N`, `quantile_rank = r` (low/median/high),
  and the emitted `source_row_first`/`source_row_last`.

## 4. Field derivation

Per-source column map (section 2). Missing optional demand fields become `null`
only where zero is not semantically meaningful; otherwise `0` (e.g. `images:0` for
text). `observed_latency_us` may be carried but is NEVER copied into a deadline.
Unmapped source columns go verbatim (as scalar strings) into `source_fields`.

## 5. Exclusion filtering (reported, never clipped)

Over-context requests are NOT clipped; they are kept (default) or excluded by an
explicit filter id whose count is recorded in `exclusion_counts`. Malformed rows
(bad integer/timestamp grammar, negative demand, monotonicity violation) FAIL
CLOSED with the offending `source_row_id`; the normalizer never skips-and-
continues. Empty output (a window with zero emitted rows) FAILS CLOSED
(`output_row_count` minimum is 1).

## 6. Time rebase and exact integer scaling

The window origin is the BIN START `k*W` (deterministic, data-independent). For a
request at source time `s_us`:

```text
rebased = s_us - k*W                       # >= 0 within the bin
t_us    = (rebased * num) / den            # integer floor division
```

with `num = time_scale_num >= 1`, `den = time_scale_den >= 1` (both positive
integers; zero/negative are schema-rejected). Floor division is monotone
non-decreasing, so ordering and equal-timestamp ties are preserved. Default
`num = den = 1` (identity). The product `rebased * num` and result are checked
against `9007199254740991`; overflow fails closed. Scaling never resamples
requests.

## 7. Deterministic mixed trace (`semi_synthetic`) -- fully frozen

A mix superposes already-normalized real component traces. Provenance value is
`semi_synthetic`. No PRNG.

### 7.1 Per-event time map (the one formula)

For a component event at time `t_component` (already an integer us in its own
normalized trace) belonging to a stream with committed `scale_num`, `scale_den`,
`offset_us`:

```text
t_mix = floor(t_component * scale_num / scale_den) + offset_us
```

- Checked integer arithmetic: `scale_num >= 1`, `scale_den >= 1`, `offset_us >=
  0` (schema-enforced). Compute `p = t_component * scale_num`; if `p >
  9007199254740991` fail closed. `q = p // scale_den` (integer floor). If `q +
  offset_us > 9007199254740991` fail closed. No binary floats.
- Stretch vs compress: `scale_num > scale_den` STRETCHES time (inter-arrival gaps
  grow, arrivals spread out); `scale_num < scale_den` COMPRESSES time (gaps
  shrink, load intensifies); `scale_num == scale_den` is identity. This is stated
  so a config author knows which direction a ratio moves load.
- Floor division is monotone non-decreasing, so within a stream ordering and
  equal-time ties are preserved.

### 7.2 Streams, ranks, ordering

- Stream `rank` values MUST be UNIQUE (semantic validator rejects duplicates).
- `streams[]` in the sidecar MUST be serialized in ascending `rank` order
  (semantic validator rejects out-of-order).
- The merged output is sorted by the section-9 total-order key
  `(t_mix, rank, source_row_id)`; `rank` is the cross-stream tie-break so two
  streams whose events land on the same `t_mix` interleave deterministically.

### 7.3 Provenance rewriting and event IDs

- Deterministic provenance rewriting: every emitted mixed event has
  `provenance = "semi_synthetic"` (it is no longer `real`/`real_decomposed`).
- Rank-namespaced event IDs: a component event `<source>:<row>` from the stream
  with rank `R` becomes `mix:<R>:<source>:<row>`. Because ranks are unique and
  each component `event_id` is unique within its source, mixed IDs are unique by
  construction.
- Duplicate output event IDs are REJECTED (fail closed) as a final guard, even
  though the namespacing makes a collision impossible under a correct config.

### 7.4 Component binding

The sidecar `streams[]` binds, per stream: `source`, `source_revision`, `rank`,
`scale_num`, `scale_den`, `offset_us`, `input_output_sha256` (component
normalized-trace hash), and `input_manifest_sha256` (component sidecar hash). No
two streams may bind the same `(input_output_sha256, input_manifest_sha256)`
component (semantic validator rejects duplicates).

The headline report ALWAYS also includes each real source replay separately.

## 8. Gate-A v1 deadlines and priorities (deferred synthetic SLO)

Gate-A v1 normalization config sets, for EVERY record: `deadline_us = null`
(`deadline_provenance:"none"`) and `priority_class = null`
(`priority_provenance:"none"`). No deadline or priority is produced in Gate-A v1.

Synthetic SLO/priority is DEFERRED to a later normalization-config version that
references an atlas/profile content hash. That later version is the ONLY place a
synthetic deadline may be produced, and only with `*_provenance:"synthetic"`
(which the request schema enforces). Gate A therefore has NO dependency on MW1
server profiles; the earlier circular `predicted_solo_us` policy is removed.

## 9. Deterministic ordering

Output is sorted by the total-order key:

```text
(t_us ASC, rank ASC, source_row_id ASC)
```

`rank` is 0 for a single-source normalization. `source_row_id` is unique within a
source, so the order is total; equal `t_us` never depends on parser or hash-map
order.

## 10. Canonical bytes and checked arithmetic

Governed by SCHEMA_CONTRACT.md section 2. Summary for the normalizer:
- UTF-8 output with `ensure_ascii=True` (pure-ASCII bytes; non-ASCII escaped
  `\uXXXX`); one object per line; `\n` (0x0A) terminators; exactly one trailing
  `\n`; no `\r`, blank lines, or trailing whitespace.
- Trace records carry NO floats (all numeric fields integer); `source_fields`
  values are string/integer/boolean/null only. The serializer REJECTS a float in
  a trace record or a non-scalar `source_fields` value, and rejects NaN/Infinity
  (`allow_nan=False`).
- Empty normalized output is forbidden (fail closed).
- All time arithmetic is checked integer/rational (section 6); overflow fails
  closed.

## 11. Hash binding

Bind every normalized output to the full set (schemas `trace_manifest` +
`artifact_manifest`):
- `source_sha256` (raw source bytes) and `source_revision` (pinned);
- `normalization_config_hash` (the committed config: window rule, filters,
  scaling, column map);
- `normalizer_version` (normalizer CODE hash);
- `output_sha256` (canonical JSONL bytes);
- the sidecar-manifest hash, recorded by the run's `artifact_manifest`
  (`outputs[].sidecar_manifest_sha256`).

A mixed output additionally binds EVERY component via `streams[]`
(`input_output_sha256` + `input_manifest_sha256` per component). A normalized
file is trusted only alongside a sidecar whose `output_sha256` matches the file's
actual bytes and whose `source_sha256` matches the fetched source.

Determinism gate (Gate A): running the normalizer twice on the same source +
config reproduces `output_sha256` exactly; V0b asserts this.

## 12. Negative tests V0b must implement

Missing/duplicate `event_id`; non-monotone or overflowing timestamp;
negative/missing token or modality demand; malformed CSV (bad dialect, BOM, lone
CR, blank row); header mismatch vs pinned; unsupported `schema_version`;
inconsistent source checksum; empty output; zero/negative scaling; a mix without
`streams`; a real manifest carrying `streams`; and an unlabeled synthetic
deadline/priority. Each asserts fail-closed behavior, not skip-and-continue. The
schema-level subset is already covered by `run_schema_tests.py`
(`fixtures/`) and the semantic subset by `validate_manifests.py` (`fixtures/
semantic/`); the parser-level subset is a V0b task.

## 13. Server-only replay (STRICTLY structural; NOT inference)

"Server-only replay" is a READER / ORDER / DAG structural pass over a normalized
trace. It is a Gate-A consumer that proves the normalized trace is usable in
arrival order. It is DEFINED here but NOT implemented in P0. It does exactly and
only:

1. Schema validation: every line validates against `request.schema.json`.
2. Hash validation: the file's bytes hash to the sidecar `output_sha256`, and the
   sidecar `source_sha256` matches the pinned source (and, for a mix, every
   component `input_output_sha256` / `input_manifest_sha256`).
3. Arrival-order preservation: events are consumed in file order, which is the
   frozen total order `(t_us, rank, source_row_id)`; the reader asserts the order
   is non-decreasing in `t_us` and never reorders.
4. Service-to-DAG mapping: each event's `service` maps to a static service DAG
   (`service_dag.schema.json`) -- a structural lookup, no execution.
5. Demand accounting: it sums/records observed demand (input_tokens,
   output_tokens, images, audio_ms, retrieved_chunks) per service/DAG-stage, as
   counts only.

It MUST NOT: run any model or kernel; use synthetic text, a model profile, a
deadline, a priority, or any performance/latency number; produce a capacity or
energy result; or consult the atlas. Its only outputs are structural: validity,
order-preserved, per-service demand counts, and coverage. Anything beyond that is
MW1+ and out of Gate A.
