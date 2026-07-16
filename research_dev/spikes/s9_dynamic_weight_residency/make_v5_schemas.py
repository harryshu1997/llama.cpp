#!/usr/bin/env python3
"""Derive the S9 bundle-version-5 schemas from the frozen v4 schemas.

v5 preserves the v4 record shapes and changes the schema/version domain. It makes
min_soc mandatory and rejects exact duplicate set-valued entries at the schema
boundary. Identity-key uniqueness is also enforced by the bundle validator.
"""
import copy
import json
import os


HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.join(HERE, "schemas", "v4")
V5 = os.path.join(HERE, "schemas", "v5")


def bump(obj):
    out = copy.deepcopy(obj)
    if isinstance(out.get("$id"), str):
        out["$id"] = out["$id"].replace("/schemas/v4/", "/schemas/v5/")
    if isinstance(out.get("title"), str):
        out["title"] = out["title"].replace("(v4)", "(v5)")
    props = out.get("properties", {})
    if isinstance(props.get("schema_version"), dict):
        props["schema_version"]["const"] = 5
    if isinstance(props.get("bundle_version"), dict):
        props["bundle_version"]["const"] = 5
    return out


def patch_manifest(obj):
    item = obj["properties"]["required_backends"]["items"]
    if "min_soc" not in item["required"]:
        item["required"].append("min_soc")
    obj["properties"]["required_backends"]["uniqueItems"] = True
    obj["properties"]["compatible_soc"]["uniqueItems"] = True
    return obj


def patch_island(obj):
    for field in ("required_weight_set_ids", "required_weight_set_digests",
                  "required_prepared_image_ids", "required_prepared_image_digests"):
        obj["properties"][field]["uniqueItems"] = True
    return obj


def patch_dispatch(obj):
    obj["properties"]["required_tuples"]["uniqueItems"] = True
    obj["properties"]["satisfied_tuples"]["uniqueItems"] = True
    return obj


def patch_weight_set(obj):
    obj["properties"]["segments"]["uniqueItems"] = True
    return obj


def patch_comment(obj, text):
    obj["$comment"] = text
    return obj


COMMENTS = {
    "bundle.schema.json":
        "Self-contained v5 envelope. The strict bundle validator selects every record's v5 schema, "
        "recomputes its digest, checks cross-record links, and rejects unknown versions, duplicate "
        "keys, malformed records, missing records, and mismatched identities. bundle_version is const "
        "5, so older record sets cannot be reinterpreted under v5 rules.",
    "correctness_certificate.schema.json":
        "Self-contained correctness attestation for one island, kernel route, backend build, and exact "
        "shape envelope. verdict is const pass. The v5 correctness_digest binds every schema-allowed "
        "content field, including the measured tolerance and evidence digests.",
    "dispatch_decision.schema.json":
        "Self-contained dispatch decision. DISPATCH requires non-empty equal tuple sets, passing "
        "correctness, every epoch and credit gate, and no hard-gate failure. The bundle validator "
        "derives cross-record device, lease, identity, snapshot, and resource truth.",
    "island_executable.schema.json":
        "Self-contained runnable island binding model, graph, layer range, atomic weight sets, prepared "
        "images, backend route, correctness certificate, I/O bounds, state policy, and fallback. The "
        "v5 island_digest binds every schema-allowed content field.",
    "prepared_image.schema.json":
        "Self-contained backend artifact derived from one canonical allocation. The v5 digest binds "
        "every schema-allowed content field, including IDs, source identity, backend build, format, "
        "payload digest, boot, generation, and physical byte charge.",
    "ready_certificate.schema.json":
        "Fail-closed READY attestation for one weight set and prepared image on one device/backend. "
        "Warmup and correctness are schema-const pass conditions. The v5 digest binds every "
        "schema-allowed attestation, identity, timestamp, and physical-accounting field.",
    "residency_lease.schema.json":
        "Slow-loop residency lease for one READY weight set on one device/backend. The v5 digest binds "
        "every schema-allowed ID, model identity, epoch, state, horizon, hysteresis, certificate, and "
        "reserved-byte field.",
    "state_lease.schema.json":
        "Fast-loop ownership of one request's mutable state. It depends on one residency generation "
        "and carries route, lease, slot, and mutation epochs. The v5 digest binds every schema-allowed "
        "identity, epoch, policy, and reservation field.",
    "transport_frame.schema.json":
        "Bounded v5 control or bulk frame. Bulk frames carry an exact ticket/chunk binding; EXECUTE "
        "and RESULT payloads require integrity hashes. Device, epoch stack, sequence, and idempotency "
        "fields are required so cross-record validation and the runtime can reject stale or duplicate work.",
}


PATCHERS = {
    "dispatch_decision.schema.json": patch_dispatch,
    "island_executable.schema.json": patch_island,
    "model_manifest.schema.json": patch_manifest,
    "weight_set.schema.json": patch_weight_set,
}


def main():
    os.makedirs(V5, exist_ok=True)
    expected = set()
    for fn in sorted(os.listdir(V4)):
        if not fn.endswith(".schema.json"):
            continue
        obj = bump(json.load(open(os.path.join(V4, fn))))
        if fn in PATCHERS:
            obj = PATCHERS[fn](obj)
        if fn in COMMENTS:
            obj = patch_comment(obj, COMMENTS[fn])
        with open(os.path.join(V5, fn), "w") as f:
            f.write(json.dumps(obj, indent=2) + "\n")
        expected.add(fn)
    stale = {fn for fn in os.listdir(V5) if fn.endswith(".schema.json")} - expected
    if stale:
        raise RuntimeError(f"stale v5 schemas: {sorted(stale)}")
    print(f"generated {len(expected)} v5 schemas under schemas/v5")


if __name__ == "__main__":
    main()
