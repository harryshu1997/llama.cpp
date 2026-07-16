# Weight Residency Contract (S9-V0)

Status: frozen historical S9-V0/v1 contract for a HOST-MANAGED, PIM-STYLE phone
accelerator. Its schemas and `validate_manifests.py` remain normative for V0/v1
record bytes and regression only. Current R1 record mechanics are versioned under
`schemas/v3/` and checked by `bundle_validate.py`. v3 rejects DRAINING leases, but
it is not scheduler authority: DeviceInventory/expiry gates, complete executable
identity, transport epochs, and state-to-ledger reservations remain unbound.
Current scheduler architecture is in `MIXED_WORKLOAD_DESIGN.md` and
`TWO_LEVEL_SCHEDULER.md`. This document does not authorize a production runtime.

PIM-style, NOT literal cache-coherent PIM. The phone cannot read arbitrary server
memory. The host transfers weights and boundary tensors; the phone materializes,
prepares, and executes them under host-issued commands and leases. Every
performance conclusion is parameterized by MEASURED link goodput
(TRANSPORT_CONTRACT.md, SIMULATOR_SPEC.md); nothing here assumes the faster cable.

Historical V0/v1 record authority, highest first:
1. `schemas/*.schema.json` -- structure, enums, ranges, fail-closed conditionals.
2. `validate_manifests.py` -- cross-field arithmetic/identity a schema cannot
   express (byte ledgers, digest recomputation, chunk tiling, state-machine edges).
3. This file + TRANSPORT_CONTRACT.md + PREFETCH_POLICY.md -- prose rules.
4. SUBSTRATE_AUDIT.md and DESIGN.md -- supporting evidence only.

This precedence is scoped to V0/v1 serialization and validation. It does not
override the active system architecture or the v3 bundle.

Content identity is SHA-256, written `sha256:<64 lowercase hex>`.

## 0. Record families (normative files)

| Object | File | Family |
|---|---|---|
| shared $defs (informative) | `schemas/_defs.schema.json` | - |
| ModelManifest | `schemas/model_manifest.schema.json` | catalog (immutable) |
| WeightSegment + transfer chunks | `schemas/weight_segment.schema.json` | catalog (immutable) |
| Atomic WeightSet | `schemas/weight_set.schema.json` | catalog (immutable) |
| Backend PreparedImage | `schemas/prepared_image.schema.json` | derived (immutable) |
| IslandExecutable | `schemas/island_executable.schema.json` | catalog (immutable) |
| TransferTicket | `schemas/transfer_ticket.schema.json` | control |
| ReadyCertificate | `schemas/ready_certificate.schema.json` | attestation |
| ResidencyLease (slow loop) | `schemas/residency_lease.schema.json` | lease |
| StateLease (fast loop) | `schemas/state_lease.schema.json` | lease |
| DeviceInventory | `schemas/device_inventory.schema.json` | telemetry+ledger |
| DispatchDecision | `schemas/dispatch_decision.schema.json` | decision |
| TransportFrame | `schemas/transport_frame.schema.json` | wire |
| SimConfig / SimRunManifest | `schemas/sim_config.schema.json`, `schemas/sim_run_manifest.schema.json` | run |

Catalog objects are content-addressed and IMMUTABLE: a change is a new hash, never
an in-place edit. Leases carry runtime state and epochs. Never mutate a catalog
object; never treat a lease as a capability.

## 1. Object model (what each thing is)

- WeightSegment: one contiguous unit of CANONICAL (backend-neutral) weight bytes,
  identified by `sha256` over its raw bytes, with an ordered `chunks[]` list whose
  per-chunk digests make a partial transfer resumable and verified. Generalizes the
  in-tree `llama_tensor_weight` locator (SUBSTRATE_AUDIT A1/B3) by adding the digest
  and chunk map the tree lacks.
- WeightSet (Atomic): an ALL-OR-NONE grouping of segments. A route may use a set
  ONLY when EVERY segment is VERIFIED_ON_DISK and materialized. `set_digest` binds
  the members; `atomicity` is a const so the rule is machine-visible. Replaces the
  loader's count-only split completeness check (SUBSTRATE_AUDIT B2) with a
  content-addressed member list.
- ModelManifest: the server-published, content-addressed description of a model and
  its weight sets. `partial_load_supported` is the ARBITRARY-MODEL gate: only a
  model whose backend honors a layer window (gemma4 today, SUBSTRATE_AUDIT G3) may
  declare multiple sub-range weight sets; every other model declares one full-range
  set or is resident-whole.
- PreparedImage: a backend-specific artifact DERIVED from a canonical weight set for
  one (backend, SoC, layout, build, boot, generation). Its `derived_image_digest`
  binds all nine identity fields (section 2) so a stale-alias can never resolve to
  the wrong image -- the exact failure the current name-keyed share registry and
  pointer-keyed xmem cache exhibit (SUBSTRATE_AUDIT E1, F2).
- IslandExecutable: a runnable unit binding a graph + weight set(s) + prepared
  image(s) + backend route + correctness certificate + server fallback.
- TransferTicket: authorizes one resumable, verified bulk transfer of a segment (or
  chunk sub-range) to one device. `priority_class` is const `bulk_weight_background`
  so weights never outrank live traffic.
- ReadyCertificate: fail-closed-by-construction attestation that a weight set is
  actually READY on one (device, backend) at a boot epoch + residency generation
  (`warmup_passed` const true, `correctness.verdict` const pass).
- ResidencyLease (slow loop): leases a weight set's residency on a (device, backend)
  for a horizon with a min-hold hysteresis; shared by many requests.
- StateLease (fast loop): owns one request's mutable KV/state on one owner, depends
  on a residency lease, carries route/lease/seq epochs.
- DeviceInventory: per-device physical-byte ledger + advisory telemetry.
- DispatchDecision: the eligibility record encoding the section 4 rule.

## 2. Content-identity preimages (exactly which bytes each hash covers)

Every hash is SHA-256, lowercase hex, `sha256:<64hex>`. A hash whose recomputation
does not match its stored value fails closed.

- `WeightSegment.sha256`: the raw canonical segment bytes exactly as fetched.
- `WeightSegment.chunks[].sha256`: the raw bytes of `[offset, offset+bytes)`.
- `WeightSet.set_digest`: SHA-256 over the segments' `sha256` strings sorted
  ascending and joined by LF (checked in `validate_manifests.py`).
- `ModelManifest.model_version` / `source_sha256`: the source model (gguf) content.
- `ModelManifest.graph_hash`: the compiled graph identity.
- `PreparedImage.derived_image_digest`: SHA-256 over the canonical-JSON of exactly
  the NINE binding fields, in this key order:
  `{arch, backend_build, boot_epoch, graph_hash, layout_version, model_version,
  residency_generation, soc, tensor_digest}` (canonical JSON = sorted keys, no
  spaces, ASCII; recomputation checked in `validate_manifests.py`). This is the
  derived-image identity the prompt mandates; NONE of it exists in the current tree.
- `ReadyCertificate` binds `weight_set_digest` + `prepared_image_digest` +
  `boot_epoch` + `residency_generation`; a mismatch on any is rejected at dispatch.
- `SimRunManifest.deterministic_replay_sha256`: SHA-256 over canonical-JSON of
  `goodput_results`.

Rationale (SUBSTRATE_AUDIT D2): the only in-tree content hash is FNV-1a 64-bit, and
the RPC server serves a cache hit WITHOUT re-comparing bytes -- a silent
wrong-weights risk. S9 replaces it with SHA-256 over the full binding; a fast
non-crypto prefilter is permitted but is never authoritative.

## 3. The two linked state machines

Machine RESIDENCY (per WeightSet per (device, backend)); enum
`weightset_state` in `_defs.schema.json`:

```
ABSENT
  -> RECEIVING          bulk chunks arriving under a TransferTicket
  -> VERIFYING          each received chunk hashed vs the segment/chunk digests
  -> VERIFIED_ON_DISK   all segments hash-match AND fsync(file)+fsync(dir) done
  -> MATERIALIZING      canonical bytes read into an owned, locked LPDDR buffer
                        (NOT an mmap alias of a mutable file; SUBSTRATE_AUDIT A5)
  -> LPDDR_READY        canonical image resident + re-verified in RAM
  -> PREPARING_HTP | PREPARING_GPU   backend-derived image produced + identity-bound
  -> WARMING            warmup + correctness sentinel runs
  -> READY_HTP | READY_GPU           ReadyCertificate issued
any state -> ERROR | QUARANTINED
```

Machine LEASE (per ResidencyLease); enum `lease_state`:

```
READY
  -> LEASED       one+ requests hold state leases that depend on this residency
  -> DRAINING     no new dispatch; in-flight work finishes; backend fence pending
  -> EVICTING     backend fence complete; buffers + share/prepack entries reclaimed
                  -> ABSENT (or back to LPDDR_READY if only the derived image is dropped)
any state -> ERROR | QUARANTINED
```

Linkage rules (enforced by the RUNTIME/simulator, which advances the machines;
`validate_manifests.py` checks per-record arithmetic/digests, NOT cross-record state
edges -- see section 8):
1. LEASE may enter LEASED only from READY, and only when the RESIDENCY machine is at
   READY_HTP or READY_GPU with a matching `residency_generation`.
2. EVICTING may not begin until DRAINING has fenced the backend (no in-flight
   kernel reads the buffers) AND no StateLease depends on the residency generation
   being evicted (lease-safe eviction; section 5).
3. Bumping a residency's `residency_generation` (a reload) STALES every StateLease
   whose `depends_on_residency_generation` is the old value; those completions are
   rejected.
4. VERIFIED_ON_DISK is durable: it survives crash/power-loss because it is gated on
   `fsync(file)+fsync(parent dir)` AFTER the hash gate (SUBSTRATE_AUDIT C6). A
   MATERIALIZING/LPDDR_READY step may resume from VERIFIED_ON_DISK; it never trusts
   an unverified on-disk prefix.
5. A VERIFYING failure (hash mismatch), an unknown SoC profile, or an unsupported
   backend routes to QUARANTINED, never silently retried into the same name.

## 4. Dispatch eligibility rule (the one rule)

A route is DISPATCHABLE for a request only when ALL hold. Clauses 1 (cert present),
2, 3, 4, the `ResidencyLease`-present half of 5, and 6 are SCHEMA-enforced in
`dispatch_decision.schema.json` via the `verdict==DISPATCH` conditional (the same-
object gates). The two CROSS-object clauses -- the `prepared_image_digest` match
against the island's `required_prepared_images` (clause 1), and the StateLease-
present requirement for sticky/rebuildable islands (clause 5) -- are enforced by the
RUNTIME/dispatcher (the dispatch_decision record carries no island `state_policy`
key to condition on), not by the schema:

1. Every required weight set is READY: a `ReadyCertificate` exists for each
   (`ready_certificate_ids` non-empty), each with `correctness.verdict==pass` and
   `warmup_passed==true` (both schema-const on the ReadyCertificate). The dispatcher
   additionally checks each cert's `prepared_image_digest` matches the island's
   `required_prepared_images` (cross-object; runtime-enforced).
2. All epochs match: `epoch_match.boot`, `.residency`, `.route`, `.state` all true.
   A stale boot epoch (reboot) or a bumped residency generation makes the route
   ineligible. (schema)
3. Correctness passed: `correctness_verdict == pass`. (schema)
4. Credits cover EVERY resource class: `credits.weights_ok`, `.derived_ok`,
   `.scratch_ok`, `.activations_ok`, `.state_ok` all true (weights, derived images,
   scratch, activations, and mutable state each fit the device ledger, section 6).
   (schema)
5. A `ResidencyLease` is present (`residency_lease_id` non-null, schema-enforced);
   for sticky/rebuildable islands a StateLease must also be present (runtime-enforced,
   since the record has no `state_policy` to key it on in-schema).
6. No hard gate fired: `hard_gate_failures` is empty. (schema)

If any fails, `verdict == FALLBACK_SERVER` with a non-`dispatch_ok` `reason_code`.
There is NO representable path from a null/partial/unknown input to DISPATCH:
`verdict==DISPATCH` is impossible unless the schema's conditional block is
satisfied in full. Specifically:
- No dispatch from partial or merely on-disk weights: VERIFIED_ON_DISK alone yields
  no ReadyCertificate, so `ready_certificate_ids` is empty -> FALLBACK_SERVER
  (`reason_code: not_resident` / `partial_on_disk`).
- No dispatch to an unknown profile or unsupported backend: those are
  `hard_gate_failures` (`unknown_profile` / `unsupported_backend`) -> FALLBACK_SERVER.
- The production candidate NEVER waits for weights: a miss is an immediate
  FALLBACK_SERVER that only updates the slow-loop predictor (PREFETCH_POLICY.md).

## 5. Lease-safe eviction + no live-state eviction

- A WeightSet may be evicted only through DRAINING -> EVICTING. DRAINING refuses new
  dispatch; EVICTING begins only after the backend fence completes and NO StateLease
  depends on the residency generation being reclaimed.
- Live mutable state is never evicted: a StateLease with `reserved_state_bytes>0`
  (sticky/rebuildable) pins its residency; the residency cannot be evicted while the
  StateLease is active. Hot KV is never migrated over the wire -- an uncertain phone
  failure resets and replays from token history (MIXED_WORKLOAD_DESIGN section 10).
- Eviction reclaims BOTH the canonical buffer and the backend-derived image plus its
  share/prepack registry entry (the teardown the current code lacks:
  SUBSTRATE_AUDIT E2, F2). Reclaim is generation-scoped so a freed-then-reallocated
  buffer cannot be resolved to a dead fd/pointer.

## 6. Physical-byte accounting (exact ledger)

`DeviceInventory.physical_byte_accounting` partitions the LPDDR budget EXACTLY
(checked in `validate_manifests.py`):

```
weights_resident + derived_images + scratch + activations + mutable_state + free
  == lpddr.total_bytes
```

No field may exceed the total. Canonical vs derived is a first-class split
(SUBSTRATE_AUDIT F3): an F16 canonical weight is HTP<->GPU shareable (one copy), but
a GPU xmem-prepacked image is a DISTINCT derived artifact (~1x extra per prepacked
tensor) that cannot be linearly shared with HTP. A phone holding both an HTP-linear
and a GPU-prepacked form pays roughly double weight RAM for those tensors, and the
ledger must show it: `derived_images` counts the prepacked bytes separately from
`weights_resident`. A dispatch's `credits.*_ok` are true only if admitting the route
keeps every partition non-negative and the sum equal to the total.

## 7. Correctness of derived images (not byte-equality)

A GPU-prepared (xmem os8) image is NOT bit-identical to the canonical weight -- it
accumulates in f16 and measured ~1.86% rel_L2 vs the stock kernel (SUBSTRATE_AUDIT
F4). Therefore residency verification of a DERIVED image is a TOLERANCE check bound
to `backend_build` + `layout_version` (recorded in `island_executable.correctness`
and the ReadyCertificate `metric_digest`), NOT a SHA-256 equality against canonical
bytes. Only CANONICAL bytes are verified by SHA-256 equality. A derived image whose
measured tolerance exceeds the island's certified bound fails WARMING and never
gets a ReadyCertificate.

## 8. What a JSON Schema enforces vs what the tool enforces

Schema-enforced: field presence/types, enums, ranges, null-ability, the
fail-closed conditionals (ReadyCertificate `warmup_passed`/`verdict` consts;
DispatchDecision `DISPATCH` block; StateLease `stateless`->0 and
`sticky`/`rebuildable`->>0; TransportFrame channel/priority binding),
`additionalProperties:false`.

Tool-enforced (`validate_manifests.py`, documented here so they are not lost):
WeightSegment chunk tiling (offsets ascending + contiguous + indices 0..n-1 + sum ==
segment bytes); WeightSet `set_digest` recomputation + `total_bytes` == sum of
segment bytes; ModelManifest `partial_load_supported==false` => exactly one
full-range weight set; PreparedImage `derived_image_digest` recomputation over the
nine binding fields + `tensor_digest` == `source_weight_set_digest`; ReadyCertificate
`physical_bytes.total` == sum of parts; DeviceInventory ledger partition equality +
no field > total; ResidencyLease `horizon.end_us` >= `start_us` + `min_hold_us`;
TransferTicket `chunk_range.first <= last`; safe-integer bounds (<= 2^53-1)
everywhere. Cross-record STATE-EDGE linkage (section 3) is enforced by the
runtime/simulator, not here.
