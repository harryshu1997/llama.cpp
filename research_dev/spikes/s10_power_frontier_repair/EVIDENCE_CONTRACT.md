# S10-V0-R-E1 typed evidence integrity contract (frozen)

Status: FROZEN for the evidence gate. Schema version 3.

This contract binds every solver-visible evidence quantity to a hashed artifact.
Only the synthetic mechanics path is currently authorized. E1 rejects every
physical `MEASURED` instance until a typed matched wall-timeline comparison
exists. It does not authorize C0-C5, PF1, measurements, or runtime work.

All arithmetic stays integer: time in us, power in mW, energy in nJ (= mW * us),
bytes in bytes, ratios and thresholds in ppm. Floats are rejected everywhere,
including NaN and Infinity.

## 1. Why this gate exists

The v2 mechanics instance is a self-consistent number soup. `duration_us: 100` is
accepted because it is an integer, not because anything ever ran that fast. The
temporal foundation proves the solver finds the true optimum OF THE NUMBERS IT IS
GIVEN. It proves nothing about whether those numbers describe a real machine.

`evidence.scope = MECHANICS_ONLY` is a safety label, not evidence. This contract
makes the difference between "a label" and "a hashed artifact" mechanically
checkable, and makes the absence of evidence fail closed instead of defaulting to
a convenient number.

## 2. Evidence-derived vs workload-declared

The validator computes the set of required bindings FROM THE INSTANCE, never from
the binding list. Dropping a binding entry therefore cannot skip a check; it is a
missing-binding error.

Evidence-derived (MUST bind to an eligible record):

| instance field | record type | bound field |
|---|---|---|
| `server_power.p8_mw` | PowerProfile | `idle_mw` |
| `server_power.p0_mw` | PowerProfile | `active_mw` |
| `server_power.wake_us` | PowerProfile | `wake_us` |
| `server_power.idle_entry_us` | PowerProfile | `idle_entry_us` |
| `server_power.transition_nj` | PowerProfile | `transition_nj` |
| `devices.<D>.active_mw` | PowerProfile | `active_mw` |
| `nodes.<N>.routes.<D>.duration_us` | RouteProfile | `latency.p95_us` |
| `nodes.<N>.routes.<D>.extra_energy_nj` | BoundaryProfile | `energy_nj` |
| `nodes.<N>.output_bytes` | BoundaryProfile | `output_bytes` |
| `batch_profiles.<K>.<S>` | RouteProfile | `latency.p95_us` |
| `activation_mem_bound_bytes` | RouteProfile | `memory_bytes` |

Workload-declared (topology and policy, NOT physical claims, never bound):
`horizon_us`, `requests[*].{arrival_us,deadline_us,priority,terminal_node}`,
`nodes[*].{id,request_id,predecessors,release_us,batch_key,tokens,kv_tokens}`, and the
`model_id` / `weight_set_id` identity strings. Identity strings are not evidence,
but they are cross-checked: a node may only bind a record carrying the same
`model_id`, `weight_set_id`, `graph_id`, and `island_id`.

`duration_us := latency.p95_us` is FROZEN. There is deliberately no statistic
selector: a selectable statistic is a fail-open surface, because a binder could
pick whichever percentile makes the schedule it wants.

KNOWN WEAKNESS, recorded rather than hidden: `activation_mem_bound_bytes` is a
device activation-memory CAPACITY, but there is no DeviceCapacityProfile record
type yet, so it currently binds to the `memory_bytes` of a designated
capacity-probe RouteProfile. That binding is mechanically enforced like any other
(missing binding, wrong value, and non-PASS records all fail closed), but the
record type is a proxy for the quantity it represents. A real capacity record is
deferred to the checkpoint that first needs a measured memory bound. This matters
because the activation bound is the one evidence-derived field where a LARGER
value makes a schedule more feasible, so a sloppy bound is favorable-biased.

## 3. Record types

Every record carries `record_id`, `kind`, `record_version`, `status`,
`reason_code`, `artifacts` (non-empty list of artifact ids), and `record_sha256`
(SHA-256 over the canonical bytes of the record with `record_sha256` removed).

Immutable identity is separate from eligibility. A record's digest covers its
identity and values. `status` is exactly one of:

- `PASS` - the record met its own gate and may be bound;
- `FAIL` - the measurement ran and did not meet its gate; retained, never bound;
- `UNKNOWN` - the quantity was not measured, or is currently unmeasurable;
  retained with a reason, never bound, and never silently read as 0; or
- `INELIGIBLE` - the measurement exists but does not meet the frozen gate
  (one process, no artifact, stale build, out-of-envelope); retained, never bound.

Only `PASS` records are bindable. Reason codes are stable strings (section 7).

### ArtifactDescriptor

`artifact_id`, `kind` (LOG|JSON|CSV|SQLITE|BINARY|NOTE), `path`, `sha256`,
`producing_tool`, `command`, `source_revision`, `build_revision`, `device_id`,
`backend_build`, `timestamp_utc`, positive `validity_us`, `provenance`
(MEASURED|SYNTHETIC|DERIVED). The bundle carries one signed
`evaluation_timestamp_utc`. Every artifact interval must contain that epoch;
future and expired artifacts fail with `E_STALE`.

The live validator runs the draft-2020 schema, resolves every artifact path
under one explicit trusted root, rejects absolute paths, traversal, and
symlinks, and hashes the actual regular-file bytes. A descriptor whose file is
missing or whose digest does not match is not evidence. SYNTHETIC artifacts use
the same byte check; there is no synthetic exemption.

The evaluation epoch is an attributable as-of assertion inside the signed
bundle. It prevents silently combining records that were already expired at the
declared evaluation point. It is self-declared and therefore is not proof of
wall-clock recency; a trusted external timestamp or append-only ledger would be
needed for that stronger claim.

`provenance` is load-bearing: a SYNTHETIC artifact can only ever support a
MECHANICS_ONLY bundle. It can never support a MEASURED bundle, so no synthetic
number can leak into a physical claim.

### CorrectnessCertificate

`island_id`, `graph_id`, `model_id`, `weight_set_id`, `shape_envelope`,
`device_id`, `backend_build`, `kernel_id`, `route_kind`, `reference_route`
(`reference_kind=CPU_FP32`, `device_id=REFERENCE_CPU`, backend, kernel, and an
exact reference artifact), `metric` (REL_L2|MAX_ABS|EXACT),
`threshold_ppm`, `observed_ppm`, `no_fallback_proof`
(`method`, `artifact_id`, `fallback_ops_observed`), `verdict`.

Gate: `status=PASS` requires `observed_ppm <= threshold_ppm`, `verdict=PASS`, and
`no_fallback_proof.fallback_ops_observed == 0`. A component correctness point is
NOT an island correctness point: `island_id` names the complete operator island
that was compared end to end against the reference route.

`metric=EXACT` is stricter: both `observed_ppm` and `threshold_ppm` MUST be zero.
Calling a nonzero tolerance "exact" fails closed.

### RouteProfile

`island_id`, `model_id`, `weight_set_id`, `graph_id`, `layer_class`,
`attention_class`, `device_id`, `backend_build`, `kernel_id`, `boundary_id`,
`shape_envelope`, `batch_size`, `latency_scope=END_TO_END`, `latency`
(`p50_us`, `p95_us`, `max_us`), `process_count`,
`sample_count`, `memory_bytes`, `h2d_bytes`, `d2h_bytes`, `correctness_id`,
`thermal_id`.

Gate: `status=PASS` requires `process_count >= MIN_PROCESSES` (3),
`sample_count >= MIN_SAMPLES` (30), a `correctness_id` resolving to a PASS
CorrectnessCertificate with identical island/graph/model/weight/device/backend/
kernel identity, a `thermal_id` resolving to a PASS ThermalInterferenceProfile
for the same device, non-empty `artifacts`, and monotone latency
(`p50_us <= p95_us <= max_us`). The correctness envelope must cover the route's
entire token/KV envelope, and the instance point `(tokens, kv_tokens)` must be
inside it. `latency.p95_us` is end-to-end for the route; boundary transfer and
preparation time may not be omitted and then presented as route latency.

### BoundaryProfile

`direction` (HOST_TO_PHONE|PHONE_TO_HOST|HOST_TO_GPU|GPU_TO_HOST|INTRA_HOST),
`transport_domain` (USB3|PCIE|SHARED_MEMORY|NONE), `h2d_bytes`, `d2h_bytes`,
`output_bytes`, `transfer_us`, `verification_us`, `materialization_us`,
`prepare_us`, `warmup_us`, `boundary_wall_us`, `contention_state`
(ISOLATED|CONTENDED),
`energy_nj`, `energy_scope` (GPU_BOARD|SERVER_WALL|TOTAL_WALL|NONE),
`process_count`, `sample_count`.

Gate: `status=PASS` requires the same process/sample minimums and non-empty
artifacts. `boundary_wall_us` must cover every reported boundary stage, and a
route's end-to-end p95 cannot be shorter than that wall time. A boundary that
carries a nonzero `energy_nj` must declare an
`energy_scope` other than `NONE`, and that scope obeys section 4.

`h2d_bytes` and `d2h_bytes` are physical link directions regardless of the
route command direction. `output_bytes` is the solver-visible live activation
produced by the island. These quantities are intentionally separate: an output
can remain resident, be returned later, or differ from the route input size.

### ThermalInterferenceProfile

`device_id`, `backend_build`, `thermal_state`
(COLD|STEADY|THROTTLED|UNKNOWN), `envelope`
(`temp_min_c`, `temp_max_c`), `duration_us`, `co_runners`, `slowdown_ppm`,
`validity_us`.

Gate: `status=PASS` requires `thermal_state != UNKNOWN` and a positive
`validity_us`. A route may bind only a profile measured on the same device and
backend binary. A route measured in one thermal state may not be bound to an
instance asserting another.

### PowerProfile

`scope` (GPU_BOARD|SERVER_WALL|TOTAL_WALL), `device_id`, `instrument_kind`, `instrument`,
`synchronization`, `sample_rate_hz`, `sample_count`, `idle_mw`, `active_mw`,
`wake_us`, `idle_entry_us`, `transition_nj`, `uncertainty_mw`, `accounting_model`,
`included_rails`,
`excluded_rails`, `validity_us`.

Gate: `status=PASS` requires `active_mw >= idle_mw`, `sample_count >=
MIN_POWER_SAMPLES` (100), a positive `sample_rate_hz`, a declared `instrument`
and `synchronization`, non-empty `artifacts`, and disjoint rail lists.
Rail ids are globally qualified as `<meter-domain>/<rail>` so equal local names
on two devices are not confused and one physical rail cannot be added twice.
`instrument_kind` fixes the meter capability; changing a free-form instrument
name cannot promote NVML from GPU_BOARD to wall power.

### InstanceEvidenceBinding

Carried inside the v3 instance as `evidence`:

    "evidence": {
      "schema_version": 3,
      "scope": "MECHANICS_ONLY" | "MEASURED",
      "bundle_sha256": "<64 hex>",
      "bindings": [
        {"target": "server_power.p0_mw",
         "record_id": "pwr.server",
         "record_sha256": "<64 hex>",
         "field": "active_mw",
         "value": 300000},
        ...
      ]
    }

Each binding names the exact record digest and the exact derived value. The
validator requires `value == record[field]` AND `value == instance[target]`.
A record digest that does not match the record in the bundle is a hard error, so
a record cannot be edited after binding, and a binding cannot be moved onto a
different record.

## 4. Energy boundary semantics (FROZEN)

`PowerProfile.scope` is exactly one of three, and they are never equated:

| scope | boundary | supports at most |
|---|---|---|
| `GPU_BOARD` | GPU board rails only (e.g. NVML board power) | none in E1 |
| `SERVER_WALL` | host/server wall, included rails explicit | none in E1 |
| `TOTAL_WALL` | one synchronized complete wall timeline | none in E1; reserved for a later matched comparison |

Rules the validator enforces:

1. NVML A6000 board power is `GPU_BOARD`. It is not server wall power. Presenting
   a GPU_BOARD instrument under a SERVER_WALL or TOTAL_WALL label is `E_SCOPE`.
2. `SERVER_WALL` and `TOTAL_WALL` may not bind additive per-device solver terms.
   The solver adds device powers, while wall meters produce aggregate timelines;
   using either as a route/device term is not the measured schedule. E1 therefore
   rejects every `MEASURED` instance with `E_SCOPE`. Physical relief requires a
   later typed control/treatment comparison record.
3. Unknown phone energy is `UNKNOWN` with a reason code. It is never `0` and never
   omitted. Encoding unknown phone energy as zero is `E_UNKNOWN_AS_ZERO`.
4. If phones are powered through a metered server wall boundary, that
   SERVER_WALL profile lists `SERVER/USB_VBUS` in `included_rails`, and no phone device
   may then bind a separate profile that adds the same rail again. Doing so is
   `E_DOUBLE_COUNT`.
5. A total-system claim additionally requires identical completed work, identical
   SLO outcomes, and matched measurement windows between control and treatment.
   Mismatch is `E_WORK_MISMATCH`, `E_SLO_MISMATCH`, or `E_WINDOW_MISMATCH`.

### Frozen later-result labels and equations

These are frozen NOW and deliberately NOT measured in this checkpoint:

    delta_E_server = E_server_control - E_server_qpim
    phone_plus_external_break_even_budget = delta_E_server

A positive `delta_E_server` is NOT a total-system energy saving. It states only
that the excluded phone, USB, charger, and relay energy may consume at most
`phone_plus_external_break_even_budget` before the system stops breaking even.
A GPU-board delta is not `delta_E_server`: CPU, DRAM, storage, fans, and PSU losses
are outside that boundary, so no server-wide or phone-energy budget follows from
it. The excluded terms are unmeasured, so the sign of the total is unknown.

Result labels reserved for that later record:

- `GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED`
- `SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED`
- `RELIEF_FAIL`
- `MEASUREMENT_INVALID`

E1 emits none of these physical labels. Only a synchronized, matched `TOTAL_WALL`
control/treatment comparison can authorize `SYSTEM_ENERGY_SAVING`; that record
type is not implemented in E1.

## 5. Bundle and instance versioning

- v2 mechanics files (`schemas/instance.schema.json`,
  `schemas/certificate.schema.json`, all three `fixtures/*.json`) are FROZEN and
  byte-unchanged. v2 is never reinterpreted.
- v3 adds `schemas/evidence_bundle.v3.schema.json`,
  `schemas/instance.v3.schema.json`, `schemas/certificate.v3.schema.json`.
- A v3 instance is a v2 mechanics core plus a v3 `evidence` binding block.
- The binder projects a v3 instance to the v2 mechanics core IN MEMORY ONLY, to
  reuse the proven solver. The projection is never the signed identity: the
  certificate binds `instance_sha256` of the FULL v3 instance and repeats
  `bundle_sha256`, so the bundle digest and every derived field stay
  checker-visible and a swapped projection cannot validate.
- The oracle and checker both consume the same validated binding. They still
  share no optimality-search code: the checker proves optimality only through the
  independently written `checker/reference.py`.

## 5b. One physical thing, one record (record coherence)

Freezing `duration_us := latency.p95_us` closes statistic selection. Record
selection is the SAME fail-open surface one level up, and it is closed here. An
adversarial audit of this gate built a server that no record described by taking
the wake ramp from one profile and the transition energy from another, and built
a schedule that cannot run by binding a COLD route for one node and a STEADY
route for another. Both validated. Both are now refused:

1. Every power field of one device MUST bind the SAME PowerProfile
   (`server_power.*` and `devices.SERVER.active_mw` are one measurement).
   Violation: `E_INCOHERENT`.
2. Every RouteProfile bound for one device MUST share ONE `thermal_id`. A device
   is in one thermal state during one schedule. Violation: `E_THERMAL`.
3. Every RouteProfile bound for one device MUST share ONE `backend_build`. No
   single binary produces a mixed-build schedule. Violation: `E_STALE`.
4. `activation_mem_bound_bytes` may only bind a RouteProfile whose `layer_class`
   is `CAPACITY_PROBE`. Violation: `E_IDENTITY`.
5. A BoundaryProfile binds by identity like any other record: matching
   `model_id`/`weight_set_id`/`graph_id`/`island_id`, and for `extra_energy_nj`
   also matching `device_id` and a `direction` consistent with that device's kind
   (phone -> HOST_TO_PHONE/PHONE_TO_HOST; server -> INTRA_HOST/HOST_TO_GPU/
   GPU_TO_HOST). Violation: `E_IDENTITY`.
6. Each RouteProfile pins one exact `boundary_id`. Its duration and energy
   bindings must use that route and boundary, and every route boundary must agree
   with its H2D/D2H bytes and the node's output bytes. Cross-record boundary cherry-picking is
   `E_INCOHERENT`.
7. A batch RouteProfile must cover every legal SERVER subset represented by its
   batch key and aggregate token count. Those subsets must have one unambiguous
   aggregate output size, its pinned boundary must carry that size and a
   SERVER-compatible direction, and `energy_nj` must be zero because the frozen
   solver has no batch-boundary energy term. Otherwise the profile is refused.

## 5c. The type gate

Integer discipline is enforced ONCE, at load, by `canon.check_integers`, before
any comparison runs. This is not defensive redundancy; it is the only thing that
can catch this class:

- JSON Schema CANNOT help. Draft6+ defines `integer` as any number with a zero
  fractional part, so `900000.0` validates as an integer. `tests/test_evidence.py`
  proves this against `/usr/bin/jsonschema` rather than asserting it in prose.
- Python then makes `900000.0 == 900000` true, so a float satisfies every binding
  equality AND every `>`/`<` eligibility comparison.
- `bool` is a subclass of `int`, so `True == 1` and an unwary counter reads `True`
  as one.

Consequently every eligibility guard is written to fail closed on a wrong type
(`if not is_int(a) or not is_int(b) or <bad>`). Written the other way round
(`if is_int(a) and is_int(b) and <bad>`) a wrong type SKIPS the check and binds.
That exact inversion was a live hole in this gate's first implementation, at six
sites, and it is the reason for this section.

Related: a missing record field resolves to `canon.MISSING`, never `None`, because
a binding carrying `value: null` would otherwise satisfy `record[field] == value`
by `None == None`.

## 6. Fail-closed rules

- Missing, stale, ambiguous, mismatched, or ineligible evidence is an error. There
  is no default value, no best-effort, and no warning-and-continue.
- Any required value that is UNKNOWN yields no measured-eligible instance.
- A `MEASURED` bundle requires every bound record to have `provenance=MEASURED`
  artifacts. A single SYNTHETIC artifact makes the bundle MECHANICS_ONLY-only.
- A `MECHANICS_ONLY` bundle is valid and proves mechanics. It authorizes no
  physical claim and no energy verdict whatsoever.
- Unknown record kind or version, duplicate JSON keys, unknown fields, empty
  identifiers, non-integers, NaN, Infinity, overflow past the declared bounds, and
  truncated input are all hard errors.
- The checked-in draft-2020 schemas run in the live binder and checker path. If
  the schema engine is missing or fails, validation refuses with
  `E_SCHEMA_ENGINE`; hand-written semantic checks are never a fallback.

## 7. Stable diagnostic codes

    E_SCHEMA            schema or type violation, duplicate key, unknown field
    E_SCHEMA_ENGINE     live draft-2020 schema engine unavailable or failed
    E_VERSION           unsupported schema or record version
    E_DUPLICATE_ID      two records share a record_id or artifact_id
    E_MISSING_RECORD    a binding names a record that is not in the bundle
    E_MISSING_ARTIFACT  a record names an artifact that is not in the bundle
    E_ARTIFACT_PATH     artifact path escapes or violates the trusted root
    E_ARTIFACT_MISSING  artifact file is absent or unreadable
    E_ARTIFACT_TYPE     artifact path is not a regular file
    E_ARTIFACT_HASH     artifact file bytes do not match the declared digest
    E_EMPTY_ARTIFACTS   a record binds nothing to a hashed artifact
    E_HASH              record_sha256 or bundle_sha256 mismatch
    E_STATUS            a bound record is not PASS
    E_IDENTITY          model/weight/graph/island/device/backend/kernel mismatch
    E_ENVELOPE          shape or token count outside the record envelope
    E_FALLBACK          CPU fallback observed in a route claimed as accelerated
    E_CORRECTNESS       missing, failed, or unknown correctness evidence
    E_SAMPLES           process or sample count below the frozen minimum
    E_THERMAL           missing, unknown, or mismatched thermal state
    E_STALE             validity window expired, or stale source/build revision
    E_SCOPE             power scope misuse (GPU_BOARD presented as wall power)
    E_UNKNOWN_AS_ZERO   an UNKNOWN quantity encoded as 0
    E_DOUBLE_COUNT      the same rail counted twice
    E_WORK_MISMATCH     control and treatment completed different work
    E_SLO_MISMATCH      control and treatment had different SLO outcomes
    E_WINDOW_MISMATCH   unmatched measurement windows
    E_BINDING_MISSING   an evidence-derived instance field has no binding
    E_BINDING_EXTRA     a binding names a field that is not evidence-derived
    E_BINDING_VALUE     bound value differs from the record or the instance
    E_PROVENANCE        synthetic artifact supporting a measured claim
    E_INCOHERENT        independently selected records describe no one route/device

## 8. Frozen minimums

    MIN_PROCESSES       = 3
    MIN_SAMPLES         = 30
    MIN_POWER_SAMPLES   = 100
    MAX_RECORDS         = 256 per kind
    MAX_BINDINGS        = 1024
    MAX_STRING          = 256
    MAX_INT             = 2**53 - 1

These are gates. They are never lowered to populate the atlas. If existing
evidence cannot meet them, the row is INELIGIBLE and the atlas stays blocked.
