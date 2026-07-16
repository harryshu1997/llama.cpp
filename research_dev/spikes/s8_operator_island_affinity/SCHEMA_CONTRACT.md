# Schema Contract (S8-V0b-P0)

Status: Gate-A trace/config serialization is frozen. The decision/action/lease
schemas that carry `status: DRAFT_BLOCKED_BEFORE_V0C` remain draft and do not
define scheduler policy. The JSON Schema files under `schemas/` (draft 2020-12)
are normative only for the bytes and validation of their own version. This
document is the human-readable index plus rules a JSON Schema cannot express.

Serialization authority for S8 records, highest first:
1. `schemas/*.schema.json` + `configs/*.config.json` -- structure and values
   (machine-checked by `run_schema_tests.py`).
2. `NORMALIZATION_SPEC.md` -- the algorithm (windows, mix, ordering, bytes).
3. `SCHEMA_CONTRACT.md` (this file) + `validate_manifests.py` -- cross-field
   rules a schema cannot express.
4. `MIXED_WORKLOAD_DESIGN.md` and `TWO_LEVEL_SCHEDULER.md` define architecture
   and policy; schemas cannot override them. Audit/result prose remains evidence.

For normalization bytes, items 1-3 above win. For scheduler architecture,
objective, and loop ownership, the main design files win. The earlier global
"WORKLOAD_TRACES.md wins" text is removed.

Frozen vocabulary: `schema_version` is the SINGLE record-format authority (no
`trace_version`); normalizer code identity is `normalizer_version` (a content
hash) in the trace sidecar. The mix provenance value is `semi_synthetic`. A mix
is composed by EXPLICIT committed integer stream offsets (`streams[].offset_us`);
there is no PRNG, no `composition_seed`, and no `composition_algorithm`.

## 0. Normative files

| record | file | family |
|---|---|---|
| shared $defs | `schemas/_defs.schema.json` | - |
| request (trace event) | `schemas/request.schema.json` | trace |
| service DAG | `schemas/service_dag.schema.json` | catalog (immutable) |
| DAG island cover | `schemas/dag_cover.schema.json` | catalog (immutable) |
| island descriptor | `schemas/island_descriptor.schema.json` | catalog (immutable) |
| model/shard manifest | `schemas/model_manifest.schema.json` | catalog (immutable) |
| model-residency lease | `schemas/model_residency_lease.schema.json` | lease (slow loop) |
| request-state lease | `schemas/request_state_lease.schema.json` | lease (fast loop) |
| phone capability | `schemas/phone_capability.schema.json` | inventory |
| telemetry | `schemas/telemetry.schema.json` | telemetry (fast, lossy) |
| compound route action | `schemas/route_action.schema.json` | decision |
| profile atlas row | `schemas/profile_row.schema.json` | atlas |
| route decision | `schemas/route_decision.schema.json` | decision |
| trace sidecar manifest | `schemas/trace_manifest.schema.json` | trace |
| artifact manifest | `schemas/artifact_manifest.schema.json` | run |

## 1. Four separated record families (mandated by review)

The V0a single "island" record conflated immutable identity with runtime state.
It is now split so that an immutable thing is never mutated and a lease is never
mistaken for a capability:

1. IMMUTABLE catalog: `island_descriptor`, `service_dag`, `dag_cover`,
   `model_manifest`. Content-addressed (`descriptor_hash`, `graph_hash`,
   `model_version`). No epochs, no runtime state. A change is a NEW hash, never
   an in-place edit.
2. MODEL-RESIDENCY lease (slow loop): `model_residency_lease`. Leases WEIGHTS on
   a `(device, backend)` for a horizon; carries `residency_epoch`,
   `ready_state`, and `share_registry_generation`. Shared by many requests.
3. REQUEST-STATE lease (fast loop): `request_state_lease`. Owns ONE request's
   KV/state on one owner; carries `route_epoch` + `lease_epoch` +
   `seq_slot_epoch` and `depends_on_residency_lease_id` +
   `depends_on_residency_epoch`. A completion whose epochs mismatch is rejected.
4. TELEMETRY (fast, lossy): `telemetry`. Continuous occupancy/thermal; explicitly
   NOT a carrier of readiness/lease/manifest edges (those are discrete and
   reliable; see SUBSTRATE_AUDIT.md section G3 and H). Staleness uses
   `last_receiver_ts_us` (receiver clock).

A model-residency lease going STALE (its `residency_epoch` bumped) invalidates
every request-state lease that `depends_on_residency_epoch` on the old value.

## 2. Canonical JSONL serialization (byte-identical rule)

A JSON Schema validates a document but does not fix its bytes. These rules do, so
two independent implementations produce identical files:

1. Encoding: UTF-8, no BOM.
2. One JSON object per line; lines terminated by a single `\n` (0x0A). No `\r`.
   The file ends with exactly one trailing `\n`.
3. No blank lines, no comments, no leading/trailing whitespace on a line.
4. Each object is serialized with:
   - keys sorted ascending by Unicode code point (equivalent to Python
     `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`);
   - separators exactly `,` and `:` with NO spaces;
   - `ensure_ascii=True` (all non-ASCII escaped as `\uXXXX`), so the byte stream
     is ASCII even for Unicode content.
5. TRACE records (`request`) carry NO floats: every numeric field is an integer
   (microseconds, bytes, counts), and `source_fields` values are string / integer
   / boolean / null only. This is enforced by `request.schema.json` and makes
   trace bytes reproducible across languages (no float formatting divergence).
   Floats appear ONLY in atlas/decision records (`profile_row.cov`,
   `correctness_metric.value`, `objective_score.*`), never in a trace record.
6. NaN and Infinity are FORBIDDEN everywhere. Serialize with `allow_nan=False`;
   a non-finite float is a serializer error (fail closed). JSON has no NaN/Inf
   literal, so a conforming file can never contain one.
7. Empty output is FORBIDDEN: a normalized trace with zero rows fails closed
   (`trace_manifest.output_row_count` minimum is 1). There is exactly one byte
   form for any non-empty output (the rules above); there is no valid empty form.
8. All derived arithmetic (time rebase/scaling) is checked integer/rational
   (NORMALIZATION_SPEC.md section 6): positive integer `num`/`den`, floor
   division, overflow checked against 9007199254740991, fail closed on overflow.
   The serializer REJECTS unsupported values (a float in a trace record, a
   non-scalar `source_fields` value, NaN/Inf) rather than coercing them.
9. `null` means "not measured / unknown". An unknown key is present with value
   `null`; never omitted, never `0`. `0` appears only where zero is semantically
   real (e.g. `images:0`). Booleans are `true`/`false`, never `1`/`0`.
10. Line order within a normalized trace file is the NORMALIZATION_SPEC.md
    total-order sort key, not input order.

The trace sidecar records `output_sha256` over the exact bytes produced by these
rules; re-running the normalizer must reproduce it.

Validation: the normative schemas under `schemas/` validate standalone (no
preloaded refs) under jsonschema 4.10.3 and ajv-cli 5.0.0 `--spec=draft2020`.
`run_schema_tests.py` pins+verifies both validators, prechecks fixtures, runs
BOTH validators over `fixtures/` (21 valid + 30 adversarial), and exits nonzero on
any unexpected result. `validate_manifests.py --selftest` runs the semantic
cross-field checks over `fixtures/semantic/` (3 valid + 12 adversarial).

## 2b. Hash preimages (exactly which bytes each hash covers)

Every hash is SHA-256, lowercase hex, written `sha256:<64hex>`. Each states its
preimage precisely so two implementations agree:

- `source_sha256` (config `origin.sha256`, sidecar `source_sha256`): the RAW
  source file bytes exactly as fetched (no normalization).
- `output_sha256` (sidecar): the canonical JSONL OUTPUT bytes (section 2), the
  full file including its single trailing `\n`.
- `normalization_config_hash` (sidecar) and `config_hash` (artifact_manifest):
  the config file serialized in the section-2 canonical object form (UTF-8,
  sorted keys, `separators=(",",":")`, `ensure_ascii=True`) -- i.e. the hash of
  `canonical_json(config)`, NOT of the pretty-printed file on disk, so formatting
  never changes the hash.
- `normalizer_version` (sidecar) and `code_version` (artifact_manifest): the
  NORMALIZER CODE BUNDLE hash = SHA-256 over the newline-joined lines of a
  manifest listing, in ascending path order, `"<relpath>  <sha256-of-file-bytes>"`
  for each committed normalizer source file (the exact file set is itself
  committed in the run config). This binds the code without depending on archive
  format.
- `input_output_sha256` / `input_manifest_sha256` (mix `streams[]`): the
  component's `output_sha256` and the SHA-256 of that component's canonical
  sidecar bytes (`canonical_json(sidecar)`), respectively.
- `outputs[].sidecar_manifest_sha256` (artifact_manifest, required when
  `kind=normalize`): SHA-256 of `canonical_json(sidecar)` for the output.
- `deterministic_replay_sha256` (artifact_manifest): SHA-256 over the
  concatenation, in ascending output-path order, of every `outputs[].sha256`
  string joined by `\n`.

A `canonical_json(x)` preimage always means the section-2 canonical serialization
of `x` (single line, sorted keys, no spaces, ASCII), never the on-disk
pretty-printed bytes. Any hash whose recomputation does not match its stored value
fails closed.

## 3. Eligibility rule (the one rule the oracle enforces)

A route/action is eligible for a request only if ALL hold:

1. The island's atlas row (`profile_row`) has `verdict == PASS`: every
   eligibility measurement non-null, `correctness == pass`, `fallback == none`,
   `n_proc >= 7`, and the request's shape is INSIDE the row's `shape_envelope`
   AND the row's `layer_range` + `attention_class` + `graph_hash` match the
   island actually being run. A row measured on `(2,3)` `causal_local_swa` does
   NOT satisfy a full-depth `(0,48)` island (review finding 6).
2. A `model_residency_lease` is `READY` on the target `(device, backend)` with a
   current `residency_epoch`.
3. A `request_state_lease` can be granted without violating RAM/state credits and
   without migrating existing hot state.
4. For a `route_action` of kind `corun_pair`, `corun_pair_certified == true`.
5. For a `route_action` of kind `merged_batch`, all `request_ids` share one
   `model_version`.
6. No hard gate in the `route_decision.hard_gate_failures` fires.

A `null` measurement, a `blocked`/`fail`/`unknown` correctness, a `cpu`/
`host_copy` fallback, `n_proc < 7`, or a shape/layer/attention/graph mismatch each
make the route INELIGIBLE. There is no path from UNKNOWN to a silently-estimated
PASS.

## 4. Gate-B fields (added by review)

A Gate-B pass additionally requires, in the `profile_row`:
- `server_control` present (the SAME island/shape measured on A6000) so relief is
  comparative, not absolute;
- `post_transfer_slo_feasible == true` (p95 + measured transfer + interference
  still meets the SLO); and
- `server_relief` present with measured `gpu_ms_freed` / `hbm_bytes_freed` /
  `hbm_bw_freed` (the actual freed server resource, not phone latency).

See ATLAS_MATRIX.md section on the amended Gate B.

## 5. What a JSON Schema here does and does not enforce

Enforced by the schema files: field presence/types, enums, null-ability, ranges,
the "deadline/priority must be labeled synthetic" conditional (request schema
`allOf`), `additionalProperties:false`.

NOT enforceable by JSON Schema, enforced by the tool instead (documented here so
they are not lost): DAG-cover partition validity (union == all nodes AND pairwise
disjoint, `dag_cover.partition_valid`); atomic multi-lane reservation
(`route_action.atomic_reservation_ok`); shape-envelope containment of a specific
request; cross-file hash consistency (an atlas row's `graph_hash` equals the
island descriptor's); JSONL byte-canonicalization (section 2); determinism
(`output_sha256` reproduction).
