#!/usr/bin/env python3
"""S9-V0-R shared identity library (bundle version 2).

Single source of truth for canonical JSON and the domain-separated content digests
of every content-addressed record. Both the fixture generator (make_fixtures_v2.py)
and the strict bundle validator (bundle_validate.py) import this module, so a digest
is defined in exactly one place and the two can never drift.

DOMAIN SEPARATION: a record digest is SHA-256 over  "s9:<domain>:v2\\n" + canonical(fields).
The domain tag makes an island digest, a lease digest, and an allocation digest live
in disjoint pre-image spaces even if their bound fields coincide, so a digest lifted
from one record kind can never be replayed as another. Every quantity is deterministic;
no clock, no PRNG. ASCII only.
"""
import hashlib
import json

SAFE_MAX = 2 ** 53 - 1
BUNDLE_VERSION = 2
BUNDLE_VERSION_R1 = 3

# record kinds recognized by bundle version 2 and the schema file each maps to.
# (kind, schema_version) -> schema filename under schemas/v2/. Unknown pairs fail closed.
# v3 (R1) reuses the same kinds/filenames (record shapes are identical; the R1 teeth are
# in the validator's coherent-chain checks). Digests are content hashes, version-agnostic.
V2_KINDS = {
    "canonical_allocation": "canonical_allocation.schema.json",
    "prepared_image": "prepared_image.schema.json",
    "correctness_certificate": "correctness_certificate.schema.json",
    "island_executable": "island_executable.schema.json",
    "ready_certificate": "ready_certificate.schema.json",
    "residency_lease": "residency_lease.schema.json",
    "state_lease": "state_lease.schema.json",
    "dispatch_decision": "dispatch_decision.schema.json",
    "transfer_ticket": "transfer_ticket.schema.json",
    "transport_frame": "transport_frame.schema.json",
    "weight_set": "weight_set.schema.json",
    "weight_segment": "weight_segment.schema.json",
    "model_manifest": "model_manifest.schema.json",
    "device_inventory": "device_inventory.schema.json",
}

V3_KINDS = dict(V2_KINDS)   # same kinds/filenames, resolved under schemas/v3/


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha_over(text):
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def domain_digest(domain, fields):
    return sha_over("s9:" + domain + ":v2\n" + canonical(fields))


def set_digest(seg_hashes):
    """Content digest of an atomic weight set: unchanged from v1 (LF-join of sorted
    segment sha256). NOT domain-separated because it is a pure content hash of bytes."""
    return sha_over("\n".join(sorted(seg_hashes)))


# ---- domain-separated identity builders (pull EXACTLY the bound fields) ----

def alloc_digest(r):
    return domain_digest("canonical_allocation", {
        "boot_epoch": r["boot_epoch"], "canonical_bytes": r["canonical_bytes"],
        "device_id": r["device_id"], "dtype": r["dtype"],
        "layout_version": r["layout_version"], "residency_generation": r["residency_generation"],
        "soc": r["soc"], "weight_set_digest": r["weight_set_digest"],
        "weight_set_id": r["weight_set_id"],
    })


def prepared_image_digest(r):
    """derived_image_digest: binds backend, image_class, preparation algorithm, backend
    build, layout, source allocation/digest, derived byte count, sharing mode,
    boot/generation, derived_payload_sha256 (plus arch/soc/model/graph/tensor)."""
    return domain_digest("prepared_image", {
        "arch": r["arch"], "backend": r["backend"], "backend_build": r["backend_build"],
        "boot_epoch": r["boot_epoch"], "derived_bytes": r["derived_bytes"],
        "derived_payload_sha256": r["derived_payload_sha256"], "graph_hash": r["graph_hash"],
        "image_class": r["image_class"], "layout_version": r["layout_version"],
        "model_version": r["model_version"], "preparation_algorithm": r["preparation_algorithm"],
        "residency_generation": r["residency_generation"], "sharing_mode": r["sharing_mode"],
        "soc": r["soc"], "source_allocation_id": r["source_allocation_id"],
        "source_weight_set_digest": r["source_weight_set_digest"], "tensor_digest": r["tensor_digest"],
    })


def correctness_digest(r):
    """Binds correctness to island, kernel route, backend build, profile-row digest,
    and the EXACT shape envelope."""
    return domain_digest("correctness_certificate", {
        "backend": r["backend"], "backend_build": r["backend_build"],
        "island_id": r["island_id"], "metric_digest": r["metric_digest"],
        "profile_row_digest": r["profile_row_digest"],
        "required_kernel_path": r["required_kernel_path"],
        "shape_envelope": r["shape_envelope"], "verdict": r["verdict"],
    })


def island_digest(r):
    return domain_digest("island_executable", {
        "backend": r["backend"], "correctness_digest": r["correctness_digest"],
        "graph_hash": r["graph_hash"], "layer_range": r["layer_range"],
        "model_version": r["model_version"], "required_kernel_path": r["required_kernel_path"],
        "required_prepared_image_digests": sorted(r["required_prepared_image_digests"]),
        "required_weight_set_digests": sorted(r["required_weight_set_digests"]),
        "service_class": r["service_class"], "state_policy": r["state_policy"],
    })


def ready_cert_digest(r):
    return domain_digest("ready_certificate", {
        "backend": r["backend"], "boot_epoch": r["boot_epoch"],
        "correctness_digest": r["correctness_digest"], "device_id": r["device_id"],
        "prepared_image_digest": r["prepared_image_digest"], "prepared_image_id": r["prepared_image_id"],
        "residency_generation": r["residency_generation"], "soc": r["soc"], "state": r["state"],
        "weight_set_digest": r["weight_set_digest"], "weight_set_id": r["weight_set_id"],
    })


def residency_lease_digest(r):
    return domain_digest("residency_lease", {
        "backend": r["backend"], "boot_epoch": r["boot_epoch"], "device_id": r["device_id"],
        "ready_certificate_digest": r["ready_certificate_digest"],
        "ready_certificate_id": r["ready_certificate_id"],
        "residency_generation": r["residency_generation"],
        "share_registry_generation": r["share_registry_generation"],
        "source_allocation_id": r["source_allocation_id"],
        "weight_set_digest": r["weight_set_digest"], "weight_set_id": r["weight_set_id"],
    })


def state_lease_digest(r):
    return domain_digest("state_lease", {
        "backend": r["backend"], "boot_epoch": r["boot_epoch"],
        "depends_on_residency_generation": r["depends_on_residency_generation"],
        "depends_on_residency_lease_id": r["depends_on_residency_lease_id"],
        "device_id": r["device_id"], "island_id": r["island_id"],
        "lease_epoch": r["lease_epoch"], "request_id": r["request_id"],
        "route_epoch": r["route_epoch"], "seq_slot_epoch": r["seq_slot_epoch"],
        "state_policy": r["state_policy"],
    })


def transfer_ticket_digest(r):
    return domain_digest("transfer_ticket", {
        "byte_range": r["byte_range"], "chunk_range": r["chunk_range"],
        "device_id": r["device_id"], "expected_sha256": r["expected_sha256"],
        "model_version": r["model_version"],
        "resumable_from_verified_offset": r["resumable_from_verified_offset"],
        "resume_partial_sha256": r["resume_partial_sha256"], "segment_id": r["segment_id"],
        "weight_set_id": r["weight_set_id"],
    })


DIGEST_BUILDERS = {
    "canonical_allocation": ("allocation_digest", alloc_digest),
    "prepared_image": ("derived_image_digest", prepared_image_digest),
    "correctness_certificate": ("correctness_digest", correctness_digest),
    "island_executable": ("island_digest", island_digest),
    "ready_certificate": ("ready_certificate_digest", ready_cert_digest),
    "residency_lease": ("residency_lease_digest", residency_lease_digest),
    "state_lease": ("state_lease_digest", state_lease_digest),
    "transfer_ticket": ("transfer_ticket_digest", transfer_ticket_digest),
}


# ======================================================================================
# S9-V0-R2 (bundle version 4) v4 digest family.  APPEND-ONLY: nothing above is touched,
# so every v2/v3 digest output is byte-for-byte unchanged (v3 reuses the v2 builders).
#
# The v4 family lives in a DISJOINT pre-image space ("s9:<domain>:v4\n" vs the v2 tag)
# and additionally BINDS the identity fields the review found unbound in v3:
#   - PreparedImage.source_weight_set_id           (repair 5)
#   - TransferTicket.issued_boot_epoch + issued_residency_generation   (repair 6)
#   - StateLease.reserved_state_bytes + reserved_activation_bytes + in_flight_mutation_seq
#                                                   (repair 9)
# The other five builders are re-domained verbatim so a whole v4 bundle forms one
# coherent chain of v4 digests. Records keep the SAME digest FIELD names; only the value
# (and its pre-image) change under bundle_version 4.
# ======================================================================================
BUNDLE_VERSION_R2 = 4
V4_KINDS = dict(V2_KINDS)   # same kinds/filenames, resolved under schemas/v4/


def domain_digest_v4(domain, fields):
    return sha_over("s9:" + domain + ":v4\n" + canonical(fields))


def alloc_digest_v4(r):
    return domain_digest_v4("canonical_allocation", {
        "boot_epoch": r["boot_epoch"], "canonical_bytes": r["canonical_bytes"],
        "device_id": r["device_id"], "dtype": r["dtype"],
        "layout_version": r["layout_version"], "residency_generation": r["residency_generation"],
        "soc": r["soc"], "weight_set_digest": r["weight_set_digest"],
        "weight_set_id": r["weight_set_id"],
    })


def prepared_image_digest_v4(r):
    """v4: adds source_weight_set_id to the v2 preimage (repair 5) so a prepared image can
    no longer name a foreign source set with an unchanged digest."""
    return domain_digest_v4("prepared_image", {
        "arch": r["arch"], "backend": r["backend"], "backend_build": r["backend_build"],
        "boot_epoch": r["boot_epoch"], "derived_bytes": r["derived_bytes"],
        "derived_payload_sha256": r["derived_payload_sha256"], "graph_hash": r["graph_hash"],
        "image_class": r["image_class"], "layout_version": r["layout_version"],
        "model_version": r["model_version"], "preparation_algorithm": r["preparation_algorithm"],
        "residency_generation": r["residency_generation"], "sharing_mode": r["sharing_mode"],
        "soc": r["soc"], "source_allocation_id": r["source_allocation_id"],
        "source_weight_set_digest": r["source_weight_set_digest"],
        "source_weight_set_id": r["source_weight_set_id"], "tensor_digest": r["tensor_digest"],
    })


def correctness_digest_v4(r):
    return domain_digest_v4("correctness_certificate", {
        "backend": r["backend"], "backend_build": r["backend_build"],
        "island_id": r["island_id"], "metric_digest": r["metric_digest"],
        "profile_row_digest": r["profile_row_digest"],
        "required_kernel_path": r["required_kernel_path"],
        "shape_envelope": r["shape_envelope"], "verdict": r["verdict"],
    })


def island_digest_v4(r):
    return domain_digest_v4("island_executable", {
        "backend": r["backend"], "correctness_digest": r["correctness_digest"],
        "graph_hash": r["graph_hash"], "layer_range": r["layer_range"],
        "model_version": r["model_version"], "required_kernel_path": r["required_kernel_path"],
        "required_prepared_image_digests": sorted(r["required_prepared_image_digests"]),
        "required_weight_set_digests": sorted(r["required_weight_set_digests"]),
        "service_class": r["service_class"], "state_policy": r["state_policy"],
    })


def ready_cert_digest_v4(r):
    return domain_digest_v4("ready_certificate", {
        "backend": r["backend"], "boot_epoch": r["boot_epoch"],
        "correctness_digest": r["correctness_digest"], "device_id": r["device_id"],
        "prepared_image_digest": r["prepared_image_digest"], "prepared_image_id": r["prepared_image_id"],
        "residency_generation": r["residency_generation"], "soc": r["soc"], "state": r["state"],
        "weight_set_digest": r["weight_set_digest"], "weight_set_id": r["weight_set_id"],
    })


def residency_lease_digest_v4(r):
    return domain_digest_v4("residency_lease", {
        "backend": r["backend"], "boot_epoch": r["boot_epoch"], "device_id": r["device_id"],
        "ready_certificate_digest": r["ready_certificate_digest"],
        "ready_certificate_id": r["ready_certificate_id"],
        "residency_generation": r["residency_generation"],
        "share_registry_generation": r["share_registry_generation"],
        "source_allocation_id": r["source_allocation_id"],
        "weight_set_digest": r["weight_set_digest"], "weight_set_id": r["weight_set_id"],
    })


def state_lease_digest_v4(r):
    """v4: adds the three physical-reservation fields to the v2 preimage (repair 9) so a
    reservation size or the in-flight mutation sequence cannot be altered silently."""
    return domain_digest_v4("state_lease", {
        "backend": r["backend"], "boot_epoch": r["boot_epoch"],
        "depends_on_residency_generation": r["depends_on_residency_generation"],
        "depends_on_residency_lease_id": r["depends_on_residency_lease_id"],
        "device_id": r["device_id"], "in_flight_mutation_seq": r["in_flight_mutation_seq"],
        "island_id": r["island_id"], "lease_epoch": r["lease_epoch"], "request_id": r["request_id"],
        "reserved_activation_bytes": r["reserved_activation_bytes"],
        "reserved_state_bytes": r["reserved_state_bytes"], "route_epoch": r["route_epoch"],
        "seq_slot_epoch": r["seq_slot_epoch"], "state_policy": r["state_policy"],
    })


def transfer_ticket_digest_v4(r):
    """v4: adds issued_boot_epoch + issued_residency_generation to the v2 preimage (repair 6)
    so a ticket cannot be replayed under a different boot/residency generation."""
    return domain_digest_v4("transfer_ticket", {
        "byte_range": r["byte_range"], "chunk_range": r["chunk_range"],
        "device_id": r["device_id"], "expected_sha256": r["expected_sha256"],
        "issued_boot_epoch": r["issued_boot_epoch"],
        "issued_residency_generation": r["issued_residency_generation"],
        "model_version": r["model_version"],
        "resumable_from_verified_offset": r["resumable_from_verified_offset"],
        "resume_partial_sha256": r["resume_partial_sha256"], "segment_id": r["segment_id"],
        "weight_set_id": r["weight_set_id"],
    })


DIGEST_BUILDERS_V4 = {
    "canonical_allocation": ("allocation_digest", alloc_digest_v4),
    "prepared_image": ("derived_image_digest", prepared_image_digest_v4),
    "correctness_certificate": ("correctness_digest", correctness_digest_v4),
    "island_executable": ("island_digest", island_digest_v4),
    "ready_certificate": ("ready_certificate_digest", ready_cert_digest_v4),
    "residency_lease": ("residency_lease_digest", residency_lease_digest_v4),
    "state_lease": ("state_lease_digest", state_lease_digest_v4),
    "transfer_ticket": ("transfer_ticket_digest", transfer_ticket_digest_v4),
}


# ======================================================================================
# S9-V0-R3 (bundle version 5). v4 remains frozen as adversarial RED evidence.
#
# v5 binds every schema-allowed content field other than the record envelope and the
# digest field itself. This avoids another hand-maintained partial preimage: v5 schemas
# reject unknown fields, so a new identity-bearing field cannot be added without also
# changing the schema/version and therefore the digest domain.
# ======================================================================================
BUNDLE_VERSION_R3 = 5
V5_KINDS = dict(V2_KINDS)


def domain_digest_v5(domain, record, digest_field):
    fields = {k: v for k, v in record.items()
              if k not in ("schema_version", "kind", digest_field)}
    return sha_over("s9:" + domain + ":v5\n" + canonical(fields))


def _v5(domain, digest_field):
    return lambda record: domain_digest_v5(domain, record, digest_field)


DIGEST_BUILDERS_V5 = {
    "canonical_allocation": ("allocation_digest", _v5("canonical_allocation", "allocation_digest")),
    "prepared_image": ("derived_image_digest", _v5("prepared_image", "derived_image_digest")),
    "correctness_certificate": ("correctness_digest", _v5("correctness_certificate", "correctness_digest")),
    "island_executable": ("island_digest", _v5("island_executable", "island_digest")),
    "ready_certificate": ("ready_certificate_digest", _v5("ready_certificate", "ready_certificate_digest")),
    "residency_lease": ("residency_lease_digest", _v5("residency_lease", "residency_lease_digest")),
    "state_lease": ("state_lease_digest", _v5("state_lease", "state_lease_digest")),
    "transfer_ticket": ("transfer_ticket_digest", _v5("transfer_ticket", "transfer_ticket_digest")),
}
