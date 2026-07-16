#!/usr/bin/env python3
"""S9 strict bundle validator (bundle versions 2 through 5).

ONE entrypoint that ALWAYS runs JSON Schema + semantic + cross-record and fails closed.
bundle_version 2 keeps the frozen V0-R behavior byte-for-byte; bundle_version 3 (the R1
contract) additionally derives ONE coherent chain and rejects the fail-open cases the
frozen V0-R validator accepted:

  R1 (v3 only):
  - the dispatch island MUST exist (E_ISLAND_ABSENT);
  - each satisfied tuple must form the exact chain
    WeightSet -> CanonicalAllocation -> PreparedImage -> ReadyCertificate ->
    ResidencyLease, matched on device, backend, boot epoch, residency generation, ids,
    and digests (E_CHAIN_BROKEN);
  - DispatchDecision device/backend/route-epoch/tuple-coverage must match the island
    (E_DISPATCH_MISMATCH);
  - a sticky/rebuildable island requires a StateLease matching request/island/device/
    backend/route_epoch/state_policy; a stateless island requires a null StateLease
    (E_STATE_LEASE);
  - alias and live-lease sets are DERIVED from the bundle records and must equal the
    allocation's declared sets and refcounts exactly, and reclaimable must follow from
    those records (E_ALIAS_SET).

bundle_version 4 (the R2 STATIC-COHERENCE contract) keeps ALL of the v3 checks (via the
shared cross_record + cross_record_r1) and additionally binds a DISPATCH to authoritative
device state, executable identity, transport epochs, and a record-derived physical ledger.
It is STATIC_SNAPSHOT_COHERENT only -- it is NOT live dispatch certification (atomic
snapshot acquisition, compare-and-reserve pins, and completion races remain V0c work):

  R2 (v4 only):
  - a DISPATCH binds exactly one DeviceInventory snapshot: device present, backend
    supported, boot_epoch + status_seq pinned, accepting=true, draining=false,
    stale=false, thermal.eligible=true (E_DEVICE_ABSENT / E_DEVICE_STALE /
    E_DEVICE_INELIGIBLE);
  - every dispatched ResidencyLease is unexpired at DispatchDecision.decision_ts_us
    (E_LEASE_EXPIRED);
  - the caller-supplied epoch_match / credits booleans must EQUAL the truth DERIVED from
    records (E_GATE_UNDERIVED);
  - the executable identity is fully bound: manifest/model/model_version/graph, backend,
    backend_build (Correctness vs PreparedImage), island vs prepared image vs manifest
    (E_IDENTITY_MISMATCH), and PreparedImage.source_weight_set_id == the served set
    (E_PI_SOURCE, also bound into the v4 prepared-image digest);
  - BULK frames bind their ticket/device/issued epoch stack and EXECUTE frames bind their
    dispatch/request/device/complete epoch stack (E_FRAME_EPOCH);
  - the DeviceInventory physical ledger is DERIVED from live records by single-copy
    canonical accounting; missing or excess charges are rejected (E_LEDGER_DERIVED).

Every rejection carries a stable error code. Schema validation uses PRIVATE per-call
temp files (no shared /tmp names), so concurrent validators cannot race. Digests are
recomputed from s9lib, which is shared with the fixture generators. v5 binds every
schema-allowed content-addressed field and closes the static identity, causality, and
transfer holes that remained in v4. This proves coherence of one immutable snapshot,
not live scheduler safety or atomic compare-and-reserve behavior.
"""
import json
import os
import subprocess
import sys
import tempfile

import s9lib
from s9lib import canonical, set_digest, SAFE_MAX, V2_KINDS, V3_KINDS, V4_KINDS, V5_KINDS

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMAS_V2 = os.path.join(HERE, "schemas", "v2")
SCHEMAS_V3 = os.path.join(HERE, "schemas", "v3")
SCHEMAS_V4 = os.path.join(HERE, "schemas", "v4")
SCHEMAS_V5 = os.path.join(HERE, "schemas", "v5")
JSONSCHEMA = "/usr/bin/jsonschema"

CODES = [
    "E_JSON_PARSE", "E_DUPLICATE_KEY", "E_ENVELOPE", "E_UNKNOWN_KIND", "E_UNKNOWN_VERSION",
    "E_SCHEMA", "E_SAFE_INT", "E_DIGEST_MISMATCH", "E_CHUNK_TILING", "E_TOTAL_MISMATCH",
    "E_LEDGER", "E_HORIZON", "E_RANGE", "E_ID_COLLISION", "E_ALLOC_REFCOUNT",
    "E_MISSING_RECORD", "E_TUPLE_MISMATCH", "E_CORRECTNESS_BINDING", "E_FRAME_BINDING",
    "E_RESUME_PREFIX", "E_EPOCH_MISMATCH",
    # R1 (bundle_version 3) additions:
    "E_ISLAND_ABSENT", "E_CHAIN_BROKEN", "E_DISPATCH_MISMATCH", "E_STATE_LEASE", "E_ALIAS_SET",
    # R2 (bundle_version 4) additions:
    "E_DEVICE_ABSENT", "E_DEVICE_STALE", "E_DEVICE_INELIGIBLE", "E_LEASE_EXPIRED",
    "E_GATE_UNDERIVED", "E_IDENTITY_MISMATCH", "E_PI_SOURCE", "E_FRAME_EPOCH", "E_LEDGER_DERIVED",
    # R3 (bundle_version 5) additions:
    "E_SNAPSHOT_CAUSAL", "E_FRAME_DUPLICATE",
]


class DupKey(Exception):
    pass


def _no_dup_hook(pairs):
    seen = set()
    for k, _ in pairs:
        if k in seen:
            raise DupKey(k)
        seen.add(k)
    return dict(pairs)


def err(code, msg):
    return (code, msg)


def _js_validate(schema_path, obj):
    """Validate obj against a schema using a PRIVATE temp file (race-free)."""
    fd, path = tempfile.mkstemp(prefix="s9_bv_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(obj))
        r = subprocess.run([JSONSCHEMA, "-i", path, schema_path], capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr)
    finally:
        os.unlink(path)


def _safe_ints(obj, path="$"):
    out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, int):
        if obj > SAFE_MAX or obj < -SAFE_MAX:
            out.append(err("E_SAFE_INT", f"{path}: integer {obj} exceeds 2^53-1"))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += _safe_ints(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _safe_ints(v, f"{path}[{i}]")
    return out


def schema_check(rec, idx, kinds, sdir, expected_ver):
    kind = rec.get("kind")
    ver = rec.get("schema_version")
    if kind not in kinds:
        return [err("E_UNKNOWN_KIND", f"record[{idx}] kind={kind!r} not a bundle kind")]
    if ver != expected_ver:
        return [err("E_UNKNOWN_VERSION", f"record[{idx}] {kind} schema_version={ver!r} (bundle expects {expected_ver})")]
    rc, out = _js_validate(os.path.join(sdir, kinds[kind]), rec)
    if "Traceback (most recent call last)" in out:
        return [err("E_SCHEMA", f"record[{idx}] {kind}: validator crash {out.strip()[:120]}")]
    if rc != 0:
        return [err("E_SCHEMA", f"record[{idx}] {kind}: {out.strip()[:200]}")]
    return []


def digest_check(rec, idx, builders):
    kind = rec.get("kind")
    out = []
    if kind in builders:
        field, builder = builders[kind]
        try:
            want = builder(rec)
        except Exception as ex:
            return [err("E_DIGEST_MISMATCH", f"record[{idx}] {kind}: cannot recompute {field}: {ex}")]
        if rec.get(field) != want:
            out.append(err("E_DIGEST_MISMATCH", f"record[{idx}] {kind}.{field} mismatch"))
    if kind == "weight_set":
        seg_hashes = [s["sha256"] for s in rec.get("segments", [])]
        if rec.get("set_digest") != set_digest(seg_hashes):
            out.append(err("E_DIGEST_MISMATCH", f"record[{idx}] weight_set.set_digest mismatch"))
        if rec.get("total_bytes") != sum(s.get("bytes", 0) for s in rec.get("segments", [])):
            out.append(err("E_TOTAL_MISMATCH", f"record[{idx}] weight_set.total_bytes != sum(segment bytes)"))
    return out


def semantic_check(rec, idx):
    kind = rec.get("kind")
    out = []
    if kind == "weight_segment":
        off = 0
        for i, c in enumerate(rec.get("chunks", [])):
            if c.get("index") != i:
                out.append(err("E_CHUNK_TILING", f"record[{idx}] chunk[{i}].index != {i}"))
            if c.get("offset") != off:
                out.append(err("E_CHUNK_TILING", f"record[{idx}] chunk[{i}].offset != {off}"))
            off += c.get("bytes", 0)
        if off != rec.get("bytes"):
            out.append(err("E_CHUNK_TILING", f"record[{idx}] chunk bytes sum {off} != segment bytes {rec.get('bytes')}"))
    elif kind == "ready_certificate":
        pb = rec.get("physical_bytes", {})
        parts = sum(pb.get(k, 0) for k in ("canonical", "derived", "scratch", "activations_reserved", "state_reserved"))
        if pb.get("total") != parts:
            out.append(err("E_TOTAL_MISMATCH", f"record[{idx}] ready_certificate.physical_bytes.total != sum"))
    elif kind == "device_inventory":
        a = rec.get("physical_byte_accounting", {})
        total = rec.get("lpddr", {}).get("total_bytes")
        fields = ["weights_resident", "derived_images", "scratch", "activations", "mutable_state", "free"]
        s = sum(a.get(f, 0) for f in fields)
        if s != total:
            out.append(err("E_LEDGER", f"record[{idx}] ledger sums to {s} != lpddr.total_bytes {total}"))
        for f in fields:
            if a.get(f, 0) > total:
                out.append(err("E_LEDGER", f"record[{idx}] ledger.{f} exceeds total"))
    elif kind == "residency_lease":
        hz = rec.get("horizon", {})
        if hz.get("end_us", 0) < hz.get("start_us", 0) + rec.get("min_hold_us", 0):
            out.append(err("E_HORIZON", f"record[{idx}] residency_lease horizon.end_us < start+min_hold"))
    elif kind == "canonical_allocation":
        if rec.get("alias_refcount") != len(rec.get("alias_prepared_image_ids", [])):
            out.append(err("E_ALLOC_REFCOUNT", f"record[{idx}] alias_refcount != len(alias_prepared_image_ids)"))
        want = (rec.get("alias_refcount", 0) == 0 and rec.get("lease_refcount", 0) == 0)
        if rec.get("reclaimable") != want:
            out.append(err("E_ALLOC_REFCOUNT", f"record[{idx}] reclaimable must be {want} for these refcounts"))
    elif kind == "transfer_ticket":
        off = rec.get("resumable_from_verified_offset", 0)
        if rec.get("byte_range", {}).get("offset") != off:
            out.append(err("E_RESUME_PREFIX", f"record[{idx}] byte_range.offset != resumable_from_verified_offset"))
        if off >= 1 and not rec.get("resume_partial_sha256"):
            out.append(err("E_RESUME_PREFIX", f"record[{idx}] nonzero resume without verified prefix digest"))
        cr = rec.get("chunk_range", {})
        if cr.get("first", 0) > cr.get("last", 0):
            out.append(err("E_RESUME_PREFIX", f"record[{idx}] chunk_range.first > last"))
    return out


def _index(records):
    by = {k: {} for k in V2_KINDS}
    lists = {"dispatch_decision": [], "transport_frame": []}
    id_field = {
        "weight_set": "weight_set_id", "weight_segment": "segment_id", "model_manifest": "model_id",
        "canonical_allocation": "allocation_id", "prepared_image": "prepared_image_id",
        "correctness_certificate": "correctness_id", "island_executable": "island_id",
        "ready_certificate": "cert_id", "residency_lease": "residency_lease_id",
        "state_lease": "state_lease_id", "transfer_ticket": "ticket_id", "device_inventory": "device_id",
    }
    collisions = []
    for rec in records:
        k = rec.get("kind")
        if k in lists:
            lists[k].append(rec)
        elif k in id_field:
            key = rec.get(id_field[k])
            if key in by[k]:
                collisions.append(err("E_ID_COLLISION", f"{k} id {key!r} appears more than once"))
            by[k][key] = rec
    return by, lists, collisions


def _check_ranges(mm):
    out = []
    n = mm.get("n_layer_total")
    sets = mm.get("weight_sets", [])
    seen = set()
    for ws in sets:
        wid = ws.get("weight_set_id")
        if wid in seen:
            out.append(err("E_ID_COLLISION", f"manifest lists weight_set {wid!r} twice"))
        seen.add(wid)
        lr = ws.get("layer_range", {})
        st, en, nlt = lr.get("start"), lr.get("end"), lr.get("n_layer_total")
        if not (isinstance(st, int) and isinstance(en, int) and st < en <= n):
            out.append(err("E_RANGE", f"weight_set {wid}: require start<end<=n_layer_total ({st},{en},{n})"))
        if nlt != n:
            out.append(err("E_RANGE", f"weight_set {wid}: layer_range.n_layer_total {nlt} != manifest {n}"))
    ranges = sorted((ws["layer_range"]["start"], ws["layer_range"]["end"]) for ws in sets
                    if isinstance(ws.get("layer_range", {}).get("start"), int))
    policy = mm.get("coverage_policy")
    if policy == "contiguous_partition":
        cursor = 0
        for st, en in ranges:
            if st != cursor:
                out.append(err("E_RANGE", f"contiguous_partition: gap/overlap at layer {cursor} (next start {st})"))
                break
            cursor = en
        else:
            if cursor != n:
                out.append(err("E_RANGE", f"contiguous_partition: coverage ends at {cursor} != {n}"))
    elif policy == "sharded_disjoint":
        prev_end = 0
        for st, en in ranges:
            if st < prev_end:
                out.append(err("E_RANGE", f"sharded_disjoint: overlap at layer {st} (prev end {prev_end})"))
            prev_end = max(prev_end, en)
    return out


def _tset(tuples):
    return sorted((t["weight_set_id"], t["prepared_image_id"], t["ready_certificate_id"], t["residency_lease_id"])
                  for t in tuples)


def cross_record(records):
    """Common (v2 + v3) cross-record checks -- byte-identical to the frozen V0-R behavior."""
    out = []
    by, lists, collisions = _index(records)
    out += collisions
    ws_by, seg_by, alloc_by, pi_by = by["weight_set"], by["weight_segment"], by["canonical_allocation"], by["prepared_image"]
    cc_by, isl_by, rc_by, rl_by = by["correctness_certificate"], by["island_executable"], by["ready_certificate"], by["residency_lease"]
    sl_by, tt_by = by["state_lease"], by["transfer_ticket"]

    for mm in by["model_manifest"].values():
        out += _check_ranges(mm)
        for ws in mm.get("weight_sets", []):
            wsr = ws_by.get(ws.get("weight_set_id"))
            if wsr is not None:
                if wsr.get("total_bytes") != ws.get("total_bytes"):
                    out.append(err("E_TOTAL_MISMATCH", f"manifest ws {ws['weight_set_id']} total_bytes != record"))
                if wsr.get("set_digest") != ws.get("set_digest"):
                    out.append(err("E_DIGEST_MISMATCH", f"manifest ws {ws['weight_set_id']} set_digest != record"))

    for pi in pi_by.values():
        al = alloc_by.get(pi.get("source_allocation_id"))
        if al is None:
            out.append(err("E_MISSING_RECORD", f"prepared_image {pi['prepared_image_id']} source_allocation absent"))
        elif al.get("weight_set_digest") != pi.get("source_weight_set_digest"):
            out.append(err("E_DIGEST_MISMATCH", f"prepared_image {pi['prepared_image_id']} source digest != allocation"))

    for al in alloc_by.values():
        for pid in al.get("alias_prepared_image_ids", []):
            pi = pi_by.get(pid)
            if pi is None:
                out.append(err("E_MISSING_RECORD", f"allocation {al['allocation_id']} alias {pid} absent"))
            elif pi.get("source_allocation_id") != al.get("allocation_id"):
                out.append(err("E_ALLOC_REFCOUNT", f"allocation {al['allocation_id']} alias {pid} does not point back"))
        live = sum(1 for rl in rl_by.values() if rl.get("source_allocation_id") == al.get("allocation_id"))
        if live > al.get("lease_refcount", 0):
            out.append(err("E_ALLOC_REFCOUNT", f"allocation {al['allocation_id']} has {live} leases > lease_refcount"))

    for rc in rc_by.values():
        pi = pi_by.get(rc.get("prepared_image_id"))
        if pi is None:
            out.append(err("E_MISSING_RECORD", f"ready_certificate {rc['cert_id']} prepared_image absent"))
        elif pi.get("derived_image_digest") != rc.get("prepared_image_digest"):
            out.append(err("E_DIGEST_MISMATCH", f"ready_certificate {rc['cert_id']} prepared_image_digest mismatch"))
        cc = cc_by.get(rc.get("correctness_id"))
        if cc is None:
            out.append(err("E_MISSING_RECORD", f"ready_certificate {rc['cert_id']} correctness absent"))
        elif cc.get("correctness_digest") != rc.get("correctness_digest"):
            out.append(err("E_DIGEST_MISMATCH", f"ready_certificate {rc['cert_id']} correctness_digest mismatch"))

    for rl in rl_by.values():
        rc = rc_by.get(rl.get("ready_certificate_id"))
        if rc is None:
            out.append(err("E_MISSING_RECORD", f"residency_lease {rl['residency_lease_id']} ready_certificate absent"))
        elif rc.get("ready_certificate_digest") != rl.get("ready_certificate_digest"):
            out.append(err("E_DIGEST_MISMATCH", f"residency_lease {rl['residency_lease_id']} ready_certificate_digest mismatch"))
        if rl.get("source_allocation_id") not in alloc_by:
            out.append(err("E_MISSING_RECORD", f"residency_lease {rl['residency_lease_id']} source_allocation absent"))

    for isl in isl_by.values():
        cc = cc_by.get(isl.get("correctness_id"))
        if cc is None:
            out.append(err("E_MISSING_RECORD", f"island {isl['island_id']} correctness absent"))
        else:
            if cc.get("correctness_digest") != isl.get("correctness_digest"):
                out.append(err("E_CORRECTNESS_BINDING", f"island {isl['island_id']} correctness_digest mismatch"))
            if cc.get("island_id") != isl.get("island_id") or cc.get("backend") != isl.get("backend") \
                    or cc.get("required_kernel_path") != isl.get("required_kernel_path"):
                out.append(err("E_CORRECTNESS_BINDING", f"island {isl['island_id']} correctness not bound to island/kernel/backend"))

    for sl in sl_by.values():
        rl = rl_by.get(sl.get("depends_on_residency_lease_id"))
        if rl is None:
            out.append(err("E_MISSING_RECORD", f"state_lease {sl['state_lease_id']} residency dependency absent"))
        elif rl.get("residency_generation") != sl.get("depends_on_residency_generation"):
            out.append(err("E_EPOCH_MISMATCH", f"state_lease {sl['state_lease_id']} depends_on generation mismatch"))
        if sl.get("island_id") not in isl_by:
            out.append(err("E_MISSING_RECORD", f"state_lease {sl['state_lease_id']} island absent"))

    for dd in lists["dispatch_decision"]:
        if dd.get("verdict") != "DISPATCH":
            continue
        if _tset(dd.get("required_tuples", [])) != _tset(dd.get("satisfied_tuples", [])):
            out.append(err("E_TUPLE_MISMATCH", f"dispatch {dd['request_id']}: required != satisfied tuples"))
        isl = isl_by.get(dd.get("island_id"))
        if isl is not None:
            need = sorted(isl.get("required_weight_set_ids", []))
            got = sorted(t["weight_set_id"] for t in dd.get("required_tuples", []))
            if need != got:
                out.append(err("E_TUPLE_MISMATCH", f"dispatch {dd['request_id']}: tuples do not cover island weight sets"))
        for t in dd.get("satisfied_tuples", []):
            if t["weight_set_id"] not in ws_by or t["prepared_image_id"] not in pi_by \
                    or t["ready_certificate_id"] not in rc_by or t["residency_lease_id"] not in rl_by:
                out.append(err("E_MISSING_RECORD", f"dispatch {dd['request_id']}: satisfied tuple references absent record"))

    for tt in tt_by.values():
        seg = seg_by.get(tt.get("segment_id"))
        if seg is not None:
            if seg.get("sha256") != tt.get("expected_sha256"):
                out.append(err("E_DIGEST_MISMATCH", f"transfer_ticket {tt['ticket_id']} expected_sha256 != segment"))
            if tt.get("chunk_range", {}).get("last", 0) >= len(seg.get("chunks", [])):
                out.append(err("E_FRAME_BINDING", f"transfer_ticket {tt['ticket_id']} chunk_range.last out of range"))

    for tf in lists["transport_frame"]:
        if tf.get("channel") != "bulk":
            continue
        bb = tf.get("bulk_binding") or {}
        tt = tt_by.get(bb.get("ticket_id"))
        if tt is None:
            out.append(err("E_MISSING_RECORD", f"bulk frame ticket {bb.get('ticket_id')} absent"))
            continue
        if bb.get("segment_id") != tt.get("segment_id"):
            out.append(err("E_FRAME_BINDING", f"bulk frame segment != ticket segment"))
        if tf.get("payload_sha256") != bb.get("chunk_sha256"):
            out.append(err("E_FRAME_BINDING", f"bulk frame payload_sha256 != bulk_binding.chunk_sha256"))
        seg = seg_by.get(bb.get("segment_id"))
        if seg is not None:
            ci = bb.get("chunk_index")
            chunks = seg.get("chunks", [])
            if not (isinstance(ci, int) and 0 <= ci < len(chunks)):
                out.append(err("E_FRAME_BINDING", f"bulk frame chunk_index {ci} out of range"))
            else:
                c = chunks[ci]
                if bb.get("chunk_offset") != c.get("offset") or bb.get("chunk_length") != c.get("bytes") \
                        or bb.get("chunk_sha256") != c.get("sha256"):
                    out.append(err("E_FRAME_BINDING", f"bulk frame binding != segment chunk[{ci}]"))
    return out


def cross_record_r1(records):
    """R1 (bundle_version 3): coherent-chain + dispatch/state-lease + record-derived
    alias/lease-set checks that close the fail-open holes the frozen V0-R accepted."""
    out = []
    by, lists, _ = _index(records)
    ws_by, alloc_by, pi_by = by["weight_set"], by["canonical_allocation"], by["prepared_image"]
    isl_by, rc_by, rl_by, sl_by = by["island_executable"], by["ready_certificate"], by["residency_lease"], by["state_lease"]

    # every residency lease's weight set MUST match the weight set of the allocation it names,
    # standalone (not only inside a DISPATCH) -- else a ws2 lease charged to a ws1 allocation lets
    # ws2's canonical allocation derive reclaimable while ws2 residency is live (use-after-free).
    for rl in rl_by.values():
        al = alloc_by.get(rl.get("source_allocation_id"))
        if al is not None and (al.get("weight_set_id") != rl.get("weight_set_id")
                               or al.get("weight_set_digest") != rl.get("weight_set_digest")):
            out.append(err("E_CHAIN_BROKEN", f"residency_lease {rl['residency_lease_id']} weight set != its source allocation"))

    # a StateLease's boot epoch MUST match the residency lease it depends on (the rest of the chain
    # is boot-guarded; a lease minted in a prior boot generation must not attach to a live chain).
    for sl in sl_by.values():
        rl = rl_by.get(sl.get("depends_on_residency_lease_id"))
        if rl is not None and rl.get("boot_epoch") != sl.get("boot_epoch"):
            out.append(err("E_STATE_LEASE", f"state_lease {sl['state_lease_id']} boot_epoch != residency lease boot_epoch"))

    # (5,6) alias + live-lease sets DERIVED from records: exact set equality + exact refcounts + reclaimable.
    # A lease counts toward an allocation ONLY when their weight sets match (a mis-charged lease is
    # already rejected above; here it must not be silently attributed to the wrong allocation).
    for al in alloc_by.values():
        aid = al["allocation_id"]
        derived_aliases = sorted(pi["prepared_image_id"] for pi in pi_by.values() if pi.get("source_allocation_id") == aid)
        declared = sorted(al.get("alias_prepared_image_ids", []))
        if derived_aliases != declared:
            out.append(err("E_ALIAS_SET", f"allocation {aid}: derived alias set {derived_aliases} != declared {declared}"))
        derived_leases = sorted(rl["residency_lease_id"] for rl in rl_by.values()
                                if rl.get("source_allocation_id") == aid and rl.get("weight_set_digest") == al.get("weight_set_digest"))
        if al.get("lease_refcount") != len(derived_leases):
            out.append(err("E_ALIAS_SET", f"allocation {aid}: lease_refcount {al.get('lease_refcount')} != derived leases {len(derived_leases)}"))
        want_reclaim = (len(derived_aliases) == 0 and len(derived_leases) == 0)
        if al.get("reclaimable") != want_reclaim:
            out.append(err("E_ALIAS_SET", f"allocation {aid}: reclaimable must be {want_reclaim} given referencing records"))

    for dd in lists["dispatch_decision"]:
        if dd.get("verdict") != "DISPATCH":
            continue
        rid = dd.get("request_id")
        dev, back, route = dd.get("device_id"), dd.get("backend"), dd.get("route_epoch")

        # (4) island must exist
        isl = isl_by.get(dd.get("island_id"))
        if isl is None:
            out.append(err("E_ISLAND_ABSENT", f"dispatch {rid}: island {dd.get('island_id')!r} absent"))
            continue

        # (1) dispatch/island coherence
        if dd.get("backend") != isl.get("backend"):
            out.append(err("E_DISPATCH_MISMATCH", f"dispatch {rid}: backend {back} != island backend {isl.get('backend')}"))

        # the DISPATCHED tuples must match the island's AUTHORITATIVE declared sets exactly, on
        # DIGESTS as well as ids (the island digest binds required_weight_set_digests +
        # required_prepared_image_ids/digests) -- else a signed island runs against phantom /
        # different-content weight sets or prepared images.
        tup = dd.get("required_tuples", [])
        got_ws_digs = sorted(ws_by[t["weight_set_id"]].get("set_digest") for t in tup if t["weight_set_id"] in ws_by)
        if got_ws_digs != sorted(isl.get("required_weight_set_digests", [])):
            out.append(err("E_DISPATCH_MISMATCH", f"dispatch {rid}: served weight-set digests != island required_weight_set_digests"))
        got_pi_ids = sorted(t["prepared_image_id"] for t in tup)
        if got_pi_ids != sorted(isl.get("required_prepared_image_ids", [])):
            out.append(err("E_DISPATCH_MISMATCH", f"dispatch {rid}: served prepared-image ids != island required_prepared_image_ids"))
        got_pi_digs = sorted(pi_by[t["prepared_image_id"]].get("derived_image_digest") for t in tup if t["prepared_image_id"] in pi_by)
        if got_pi_digs != sorted(isl.get("required_prepared_image_digests", [])):
            out.append(err("E_DISPATCH_MISMATCH", f"dispatch {rid}: served prepared-image digests != island required_prepared_image_digests"))

        # (1,3) coherent chain per satisfied tuple
        for t in dd.get("satisfied_tuples", []):
            ws = ws_by.get(t["weight_set_id"]); pi = pi_by.get(t["prepared_image_id"])
            rc = rc_by.get(t["ready_certificate_id"]); rl = rl_by.get(t["residency_lease_id"])
            if not (ws and pi and rc and rl):
                out.append(err("E_CHAIN_BROKEN", f"dispatch {rid}: tuple references absent record"))
                continue
            al = alloc_by.get(pi.get("source_allocation_id"))
            probs = []
            if rl.get("state") not in ("READY", "LEASED"):   # DRAINING/EVICTING/ERROR/QUARANTINED are not dispatchable
                probs.append(f"residency_lease state {rl.get('state')!r} is not dispatchable")
            if al is None:
                probs.append("prepared_image source_allocation absent")
            else:
                if al.get("weight_set_id") != ws["weight_set_id"] or al.get("weight_set_digest") != ws.get("set_digest"):
                    probs.append("allocation not for this weight set")
                if rl.get("source_allocation_id") != al.get("allocation_id"):
                    probs.append("residency_lease allocation != prepared_image allocation")
            if pi.get("tensor_digest") != ws.get("set_digest") or pi.get("source_weight_set_digest") != ws.get("set_digest"):
                probs.append("prepared_image not derived from this weight set")
            if rc.get("weight_set_id") != ws["weight_set_id"] or rc.get("prepared_image_id") != pi["prepared_image_id"] \
                    or rc.get("prepared_image_digest") != pi.get("derived_image_digest"):
                probs.append("ready_certificate not for this (weight set, prepared image)")
            if rc.get("correctness_id") != isl.get("correctness_id") \
                    or rc.get("correctness_digest") != isl.get("correctness_digest"):
                probs.append("ready_certificate correctness not for this island")
            if rl.get("weight_set_id") != ws["weight_set_id"] or rl.get("ready_certificate_id") != rc["cert_id"] \
                    or rl.get("ready_certificate_digest") != rc.get("ready_certificate_digest"):
                probs.append("residency_lease not for this ready certificate")
            # device / backend / epoch / generation coherence across the chain
            devs = {al.get("device_id") if al else dev, rc.get("device_id"), rl.get("device_id"), dev}
            if len(devs) != 1:
                probs.append(f"device mismatch across chain {sorted(str(x) for x in devs)}")
            backs = {pi.get("backend"), rc.get("backend"), rl.get("backend"), isl.get("backend"), back}
            if len(backs) != 1:
                probs.append(f"backend mismatch across chain {sorted(str(x) for x in backs)}")
            boots = {x.get("boot_epoch") for x in (al, pi, rc, rl) if x is not None}
            if len(boots) != 1:
                probs.append("boot_epoch mismatch across chain")
            gens = {x.get("residency_generation") for x in (al, pi, rc, rl) if x is not None}
            if len(gens) != 1:
                probs.append("residency_generation mismatch across chain")
            for p in probs:
                out.append(err("E_CHAIN_BROKEN", f"dispatch {rid} tuple {t['weight_set_id']}: {p}"))

        # (2) state lease binding for the island's state policy
        sp = isl.get("state_policy")
        slid = dd.get("state_lease_id")
        if sp in ("sticky", "rebuildable"):
            if not slid:
                out.append(err("E_STATE_LEASE", f"dispatch {rid}: {sp} island requires a StateLease"))
            else:
                sl = sl_by.get(slid)
                if sl is None:
                    out.append(err("E_STATE_LEASE", f"dispatch {rid}: state_lease {slid} absent"))
                elif not (sl.get("request_id") == rid and sl.get("island_id") == dd.get("island_id")
                          and sl.get("device_id") == dev and sl.get("backend") == back
                          and sl.get("route_epoch") == route and sl.get("state_policy") == sp):
                    out.append(err("E_STATE_LEASE", f"dispatch {rid}: foreign/mismatched StateLease {slid}"))
                elif sl.get("depends_on_residency_lease_id") not in {
                        t.get("residency_lease_id") for t in dd.get("satisfied_tuples", [])}:
                    out.append(err("E_STATE_LEASE", f"dispatch {rid}: StateLease dependency is not a dispatched residency lease"))
        else:  # stateless
            if slid is not None:
                out.append(err("E_STATE_LEASE", f"dispatch {rid}: stateless island must have null state_lease_id"))
    return out


def _derive_ledger(records):
    """Single-copy canonical physical ledger DERIVED from the live bundle records, per device.
    Returns {device_id: {"ok": bool, "errors": [...]}} where a partition is charged EXACTLY once
    (one canonical allocation = one weights charge; a shared_canonical prepared image adds 0 derived
    bytes; a private image adds its derived_bytes). EVERY device a physically-resident record names
    is derived: a device carrying live bytes but NO DeviceInventory to account them is itself a
    missing-charge failure (records must not escape the ledger by omitting the inventory)."""
    by, _, _ = _index(records)
    alloc_by, pi_by, rl_by, sl_by, di_by = (by["canonical_allocation"], by["prepared_image"],
                                            by["residency_lease"], by["state_lease"], by["device_inventory"])

    def partitions(dev):
        weights = sum(a.get("canonical_bytes", 0) for a in alloc_by.values() if a.get("device_id") == dev)
        derived = sum(pi.get("derived_bytes", 0) for pi in pi_by.values()
                      if (alloc_by.get(pi.get("source_allocation_id")) or {}).get("device_id") == dev)
        scratch = sum(rl.get("reserved_bytes", {}).get("scratch", 0) for rl in rl_by.values() if rl.get("device_id") == dev)
        activations = sum(sl.get("reserved_activation_bytes", 0) for sl in sl_by.values() if sl.get("device_id") == dev)
        state = sum(sl.get("reserved_state_bytes", 0) for sl in sl_by.values() if sl.get("device_id") == dev)
        return {"weights_resident": weights, "derived_images": derived, "scratch": scratch,
                "activations": activations, "mutable_state": state}

    ref_devices = {a.get("device_id") for a in alloc_by.values()}
    ref_devices |= {rl.get("device_id") for rl in rl_by.values()}
    ref_devices |= {sl.get("device_id") for sl in sl_by.values()}
    ref_devices |= {(alloc_by.get(pi.get("source_allocation_id")) or {}).get("device_id") for pi in pi_by.values()}
    ref_devices |= set(di_by.keys())
    ref_devices.discard(None)

    result = {}
    for dev in sorted(ref_devices):
        errs = []
        want = partitions(dev)
        di = di_by.get(dev)
        if di is None:
            if sum(want.values()) > 0:
                errs.append(err("E_LEDGER_DERIVED", f"device {dev}: {sum(want.values())} live bytes across records "
                               f"but NO DeviceInventory to account them (missing charge)"))
            result[dev] = {"ok": not errs, "errors": errs}
            continue
        total = di.get("lpddr", {}).get("total_bytes")
        acct = di.get("physical_byte_accounting", {})
        for f, v in want.items():
            if acct.get(f) != v:
                errs.append(err("E_LEDGER_DERIVED", f"device {dev}: ledger.{f}={acct.get(f)} != derived-from-records {v}"))
        used = sum(want.values())
        if isinstance(total, int):
            free = total - used
            if free < 0:
                errs.append(err("E_LEDGER_DERIVED", f"device {dev}: derived reservations {used} exceed lpddr.total {total}"))
            elif acct.get("free") != free:
                errs.append(err("E_LEDGER_DERIVED", f"device {dev}: ledger.free={acct.get('free')} != derived {free}"))
        result[dev] = {"ok": not errs, "errors": errs}
    return result


def cross_record_v4(records):
    """R2 (bundle_version 4): STATIC-COHERENCE repairs layered on top of v3. Binds a DISPATCH
    to authoritative DeviceInventory state, lease expiry, record-derived epoch/credit gates,
    full executable identity, transport-frame epoch stacks, and a record-derived physical
    ledger. STATIC_SNAPSHOT_COHERENT only -- NOT live dispatch certification."""
    out = []
    by, lists, _ = _index(records)
    ws_by, alloc_by, pi_by = by["weight_set"], by["canonical_allocation"], by["prepared_image"]
    isl_by, rc_by, rl_by, sl_by = by["island_executable"], by["ready_certificate"], by["residency_lease"], by["state_lease"]
    cc_by, mm_by, di_by, tt_by = by["correctness_certificate"], by["model_manifest"], by["device_inventory"], by["transfer_ticket"]

    # (10,11) DERIVE the physical ledger from live records; report the per-device deltas.
    ledger = _derive_ledger(records)
    for info in ledger.values():
        out += info["errors"]

    # (10) attestations must reconcile with the physical device: a ReadyCertificate cannot attest
    # a footprint larger than the device LPDDR, and a ResidencyLease's weight reservation must equal
    # the single-copy canonical allocation it names.
    for rc in rc_by.values():
        di = di_by.get(rc.get("device_id"))
        pb = rc.get("physical_bytes", {})
        total = di.get("lpddr", {}).get("total_bytes") if di is not None else None
        if isinstance(total, int) and pb.get("total", 0) > total:
            out.append(err("E_LEDGER_DERIVED", f"ready_certificate {rc.get('cert_id')}: physical_bytes.total "
                           f"{pb.get('total')} exceeds device {rc.get('device_id')} lpddr.total {total}"))
    for rl in rl_by.values():
        al = alloc_by.get(rl.get("source_allocation_id"))
        rb = rl.get("reserved_bytes", {})
        if al is not None and rb.get("weights") != al.get("canonical_bytes"):
            out.append(err("E_LEDGER_DERIVED", f"residency_lease {rl.get('residency_lease_id')}: reserved weights "
                           f"{rb.get('weights')} != source allocation canonical_bytes {al.get('canonical_bytes')}"))
        # a lease's derived reservation must equal the derived bytes of the images on its allocation
        derived = sum(pi.get("derived_bytes", 0) for pi in pi_by.values()
                      if pi.get("source_allocation_id") == rl.get("source_allocation_id"))
        if rb.get("derived") != derived:
            out.append(err("E_LEDGER_DERIVED", f"residency_lease {rl.get('residency_lease_id')}: reserved derived "
                           f"{rb.get('derived')} != images on its allocation {derived}"))

    # a canonical allocation must hold the FULL weight set and be single-copy per (content, device, gen)
    seen_alloc = {}
    for al in alloc_by.values():
        ws = ws_by.get(al.get("weight_set_id"))
        if ws is not None and al.get("canonical_bytes") != ws.get("total_bytes"):
            out.append(err("E_LEDGER_DERIVED", f"allocation {al.get('allocation_id')}: canonical_bytes "
                           f"{al.get('canonical_bytes')} != weight set total_bytes {ws.get('total_bytes')}"))
        key = (al.get("weight_set_digest"), al.get("device_id"), al.get("residency_generation"))
        if key in seen_alloc:
            out.append(err("E_LEDGER_DERIVED", f"allocation {al.get('allocation_id')}: duplicate single-copy "
                           f"allocation of {key[0]} on {key[1]} at generation {key[2]} (also {seen_alloc[key]})"))
        seen_alloc[key] = al.get("allocation_id")

    # the ReadyCertificate's embedded correctness attestation must match its correctness certificate
    for rc in rc_by.values():
        cc = cc_by.get(rc.get("correctness_id"))
        emb = rc.get("correctness", {})
        if cc is not None and (emb.get("metric_digest") != cc.get("metric_digest")
                               or emb.get("verdict") != cc.get("verdict")):
            out.append(err("E_CORRECTNESS_BINDING", f"ready_certificate {rc.get('cert_id')}: embedded correctness "
                           f"(verdict/metric) != its correctness certificate"))

    # a TransferTicket's issued boot generation must match the device it targets
    for tt in tt_by.values():
        di = di_by.get(tt.get("device_id"))
        if di is not None and tt.get("issued_boot_epoch") != di.get("boot_epoch"):
            out.append(err("E_FRAME_EPOCH", f"transfer_ticket {tt.get('ticket_id')}: issued_boot_epoch "
                           f"{tt.get('issued_boot_epoch')} != device boot_epoch {di.get('boot_epoch')}"))

    # a request has EXACTLY ONE dispatch decision; two DISPATCH records, or a DISPATCH and a
    # FALLBACK_SERVER, for the same request_id are contradictory outcomes for one request.
    req_seen = set()
    for dd in lists["dispatch_decision"]:
        rid = dd.get("request_id")
        if rid in req_seen:
            out.append(err("E_DISPATCH_MISMATCH", f"dispatch {rid}: more than one dispatch_decision for this request"))
        req_seen.add(rid)

    # dispatch-by-request (for EXECUTE frame binding) + residency generation per dispatch.
    dd_by_req = {}
    for dd in lists["dispatch_decision"]:
        if dd.get("verdict") == "DISPATCH":
            dd_by_req[dd.get("request_id")] = dd

    for dd in lists["dispatch_decision"]:
        if dd.get("verdict") != "DISPATCH":
            continue
        rid = dd.get("request_id")
        dev, back, route = dd.get("device_id"), dd.get("backend"), dd.get("route_epoch")
        isl = isl_by.get(dd.get("island_id"))
        tuples = dd.get("satisfied_tuples", [])
        tuple_leases = [rl_by.get(t.get("residency_lease_id")) for t in tuples]
        tuple_leases = [rl for rl in tuple_leases if rl is not None]

        # ---- (1) bind exactly one DeviceInventory snapshot ----
        ref = dd.get("device_status_ref") or {}
        di = di_by.get(dev)
        snapshot_ok = True
        if di is None:
            out.append(err("E_DEVICE_ABSENT", f"dispatch {rid}: no DeviceInventory for device {dev!r}"))
            snapshot_ok = False
        else:
            if ref.get("device_id") != dev or di.get("boot_epoch") != ref.get("boot_epoch") \
                    or di.get("status_seq") != ref.get("status_seq"):
                out.append(err("E_DEVICE_STALE", f"dispatch {rid}: pinned snapshot (dev {ref.get('device_id')!r}, "
                               f"boot {ref.get('boot_epoch')}, seq {ref.get('status_seq')}) != DeviceInventory "
                               f"(dev {dev!r}, boot {di.get('boot_epoch')}, seq {di.get('status_seq')})"))
                snapshot_ok = False
            # the live device boot MUST equal the boot the dispatched residency was minted at
            if any(rl.get("boot_epoch") != di.get("boot_epoch") for rl in tuple_leases):
                out.append(err("E_DEVICE_STALE", f"dispatch {rid}: dispatched residency boot != device boot_epoch {di.get('boot_epoch')}"))
                snapshot_ok = False
            if not di.get("accepting", False) or di.get("draining", True) or di.get("stale", True):
                out.append(err("E_DEVICE_INELIGIBLE", f"dispatch {rid}: device not accepting "
                               f"(accepting={di.get('accepting')} draining={di.get('draining')} stale={di.get('stale')})"))
            if not di.get("thermal", {}).get("eligible", False):
                out.append(err("E_DEVICE_INELIGIBLE", f"dispatch {rid}: device thermally ineligible"))
            if back not in di.get("backends", []):
                out.append(err("E_DEVICE_INELIGIBLE", f"dispatch {rid}: backend {back!r} not supported by device {di.get('backends')}"))

        # ---- (2) reject residency leases expired at the decision timestamp ----
        ts = dd.get("decision_ts_us")
        for rl in tuple_leases:
            hz = rl.get("horizon", {})
            if isinstance(ts, int) and not (hz.get("start_us", 0) <= ts <= hz.get("end_us", 0)):
                out.append(err("E_LEASE_EXPIRED", f"dispatch {rid}: residency lease {rl['residency_lease_id']} "
                               f"not live at decision_ts_us {ts} (horizon {hz.get('start_us')}..{hz.get('end_us')})"))

        # ---- (4,5) full executable identity ----
        if isl is not None:
            mm = next((m for m in mm_by.values() if m.get("model_id") == isl.get("model_id")), None)
            # fail CLOSED when the island's model is not anchored to a present manifest
            if mm is None:
                out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid}: island model_id {isl.get('model_id')!r} has no model manifest in the bundle"))
            elif isl.get("model_version") != mm.get("model_version") or isl.get("graph_hash") != mm.get("graph_hash"):
                out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid}: island model_version/graph_hash != manifest"))
            for t in tuples:
                ws = ws_by.get(t.get("weight_set_id")); pi = pi_by.get(t.get("prepared_image_id"))
                rc = rc_by.get(t.get("ready_certificate_id"))
                if not (ws and pi):
                    continue
                if pi.get("model_version") != isl.get("model_version") or pi.get("graph_hash") != isl.get("graph_hash") \
                        or pi.get("backend") != isl.get("backend"):
                    out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid} tuple {t.get('weight_set_id')}: prepared image model/graph/backend != island"))
                if ws.get("model_version") != isl.get("model_version"):
                    out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid} tuple {t.get('weight_set_id')}: weight set model_version != island"))
                cc = cc_by.get(rc.get("correctness_id")) if rc is not None else None
                if cc is not None and cc.get("backend_build") != pi.get("backend_build"):
                    out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid} tuple {t.get('weight_set_id')}: correctness backend_build != prepared image backend_build"))
                if pi.get("source_weight_set_id") != ws.get("weight_set_id"):
                    out.append(err("E_PI_SOURCE", f"dispatch {rid} tuple {t.get('weight_set_id')}: prepared image "
                                   f"source_weight_set_id {pi.get('source_weight_set_id')!r} != served set {ws.get('weight_set_id')!r}"))
                # the remaining digest-bound identity fields (arch / soc / layout_version) must ALSO
                # agree across the chain + the dispatch device -- a coherent digest alone does not
                # make a gemma4 image legal on an op12/v75 device or under a foreign arch/layout.
                al = alloc_by.get(pi.get("source_allocation_id"))
                if pi.get("arch") != ws.get("arch") or (mm is not None and pi.get("arch") != mm.get("arch")):
                    out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid} tuple {t.get('weight_set_id')}: arch disagreement (prepared image / weight set / manifest)"))
                socs = {pi.get("soc")}
                if rc is not None:
                    socs.add(rc.get("soc"))
                if al is not None:
                    socs.add(al.get("soc"))
                if di is not None:
                    socs.add(di.get("soc"))
                if len(socs) != 1:
                    out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid} tuple {t.get('weight_set_id')}: soc disagreement across chain/device {sorted(str(s) for s in socs)}"))
                lvs = {pi.get("layout_version"), ws.get("layout_version")}
                if al is not None:
                    lvs.add(al.get("layout_version"))
                if mm is not None:
                    lvs.add(mm.get("layout_version"))
                if len(lvs) != 1:
                    out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid} tuple {t.get('weight_set_id')}: layout_version disagreement {sorted(str(v) for v in lvs)}"))
                # the derived-image FORMAT must be legal for the serving backend (an htp HMX kernel
                # cannot consume a gpu_xmem image); enum names are backend-prefixed by convention.
                if not str(pi.get("image_class", "")).startswith(pi.get("backend", "")) \
                        or not str(pi.get("preparation_algorithm", "")).startswith(pi.get("backend", "")):
                    out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid} tuple {t.get('weight_set_id')}: image_class/preparation_algorithm not for backend {pi.get('backend')!r}"))
                # the served backend + build and the SoC must be sanctioned by the manifest.
                if mm is not None:
                    if not any(rb.get("backend") == pi.get("backend") and rb.get("backend_build") == pi.get("backend_build")
                               for rb in mm.get("required_backends", [])):
                        out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid} tuple {t.get('weight_set_id')}: served (backend {pi.get('backend')!r}, build) not in manifest required_backends"))
                    if di is not None and di.get("soc") not in mm.get("compatible_soc", []):
                        out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid} tuple {t.get('weight_set_id')}: device soc {di.get('soc')!r} not in manifest compatible_soc {mm.get('compatible_soc')}"))
                    dtypes = {ws.get("dtype"), mm.get("dtype")}
                    if al is not None:
                        dtypes.add(al.get("dtype"))
                    if len(dtypes) != 1:
                        out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid} tuple {t.get('weight_set_id')}: dtype disagreement {sorted(str(d) for d in dtypes)}"))

        # ---- (3) caller epoch/credit booleans are NOT authority: derive and require consistency ----
        chain_boot_ok = di is not None and all(rl.get("boot_epoch") == di.get("boot_epoch") for rl in tuple_leases)
        epoch_true = snapshot_ok and chain_boot_ok
        credit_true = ledger.get(dev, {}).get("ok", False)
        if dd.get("epoch_match", {}).get("boot") and not epoch_true:
            out.append(err("E_GATE_UNDERIVED", f"dispatch {rid}: epoch_match.boot claimed true but not derivable from records"))
        if any(dd.get("credits", {}).values()) and not credit_true:
            out.append(err("E_GATE_UNDERIVED", f"dispatch {rid}: credits claimed true but device ledger not derivable from records"))

    # ---- (7,8) transport frame epoch stacks bound to their ticket / dispatch ----
    for tf in lists["transport_frame"]:
        if tf.get("channel") == "bulk":
            bb = tf.get("bulk_binding") or {}
            tt = tt_by.get(bb.get("ticket_id"))
            if tt is None:
                continue   # absence already reported by the v2 cross_record frame check
            if tf.get("device_id") != tt.get("device_id") \
                    or tf.get("boot_epoch") != tt.get("issued_boot_epoch") \
                    or tf.get("residency_epoch") != tt.get("issued_residency_generation"):
                out.append(err("E_FRAME_EPOCH", f"bulk frame (ticket {bb.get('ticket_id')}): device/boot/residency "
                               f"!= ticket issued (dev {tt.get('device_id')!r} boot {tt.get('issued_boot_epoch')} "
                               f"gen {tt.get('issued_residency_generation')})"))
        elif tf.get("msg_type") in ("EXECUTE", "RESULT"):
            # EXECUTE and RESULT are symmetric live-payload frames: both carry the full epoch
            # stack + request_id and must bind to the dispatch they belong to.
            mt = tf.get("msg_type")
            dd = dd_by_req.get(tf.get("request_id"))
            if dd is None:
                out.append(err("E_FRAME_EPOCH", f"{mt} frame request {tf.get('request_id')!r}: no DISPATCH for this request"))
                continue
            probs = []
            if tf.get("device_id") != dd.get("device_id"):
                probs.append("device != dispatch device")
            if tf.get("route_epoch") != dd.get("route_epoch"):
                probs.append("route_epoch != dispatch route_epoch")
            di = di_by.get(dd.get("device_id"))
            if di is not None and tf.get("boot_epoch") != di.get("boot_epoch"):
                probs.append("boot_epoch != device boot_epoch")
            gens = {rl_by[t["residency_lease_id"]].get("residency_generation")
                    for t in dd.get("satisfied_tuples", []) if t.get("residency_lease_id") in rl_by}
            if len(gens) == 1 and tf.get("residency_epoch") != next(iter(gens)):
                probs.append("residency_epoch != dispatched residency generation")
            sl = sl_by.get(dd.get("state_lease_id"))
            if sl is not None and tf.get("state_epoch") != sl.get("seq_slot_epoch"):
                probs.append("state_epoch != state lease seq_slot_epoch")
            for p in probs:
                out.append(err("E_FRAME_EPOCH", f"{mt} frame request {tf.get('request_id')}: {p}"))
    return out


def _covers_layer_range(ranges, target):
    """Return true when the union of ranges covers target without a gap."""
    cursor = target[0]
    for start, end in sorted(ranges):
        if end <= cursor:
            continue
        if start > cursor:
            return False
        cursor = max(cursor, end)
        if cursor >= target[1]:
            return True
    return cursor >= target[1]


def cross_record_v5(records):
    """R3 (bundle version 5): close the static holes found after the v4 audit.

    These checks operate only after every record has passed its v5 JSON Schema. They
    establish coherence of a captured bundle, not freshness or atomicity of live state.
    Payload hashing, frame CRC verification, and compare-and-reserve remain runtime
    responsibilities.
    """
    out = []
    by, lists, _ = _index(records)
    mm_by = by["model_manifest"]
    ws_by = by["weight_set"]
    seg_by = by["weight_segment"]
    pi_by = by["prepared_image"]
    alloc_by = by["canonical_allocation"]
    cc_by = by["correctness_certificate"]
    rc_by = by["ready_certificate"]
    rl_by = by["residency_lease"]
    sl_by = by["state_lease"]
    isl_by = by["island_executable"]
    di_by = by["device_inventory"]
    tt_by = by["transfer_ticket"]

    soc_rank = {"op12": 0, "op15": 1}

    for dd in lists["dispatch_decision"]:
        if dd["verdict"] != "DISPATCH":
            continue
        rid = dd["request_id"]
        decision_ts = dd["decision_ts_us"]
        isl = isl_by.get(dd["island_id"])
        di = di_by.get(dd["device_id"])

        if di is not None and di["receiver_ts_us"] > decision_ts:
            out.append(err("E_SNAPSHOT_CAUSAL", f"dispatch {rid}: device snapshot timestamp is after decision"))
        if isl is None:
            continue
        mm = mm_by.get(isl["model_id"])
        if mm is None:
            continue

        island_range = isl["layer_range"]
        if not (island_range["n_layer_total"] == mm["n_layer_total"] and
                0 <= island_range["start"] < island_range["end"] <= mm["n_layer_total"]):
            out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid}: island layer range is outside manifest"))
        cc = cc_by.get(isl["correctness_id"])
        if cc is not None:
            io = isl["io_schema"]
            shape = cc["shape_envelope"]
            if any((
                    io["input_dtype"] != shape["input_dtype"],
                    io["output_dtype"] != shape["output_dtype"],
                    io["max_input_bytes"] > shape["max_input_bytes"],
                    io["max_output_bytes"] > shape["max_output_bytes"],
            )):
                out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid}: island I/O exceeds correctness envelope"))

        backend_keys = [(entry["backend"], entry["backend_build"]) for entry in mm["required_backends"]]
        if len(backend_keys) != len(set(backend_keys)):
            out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid}: manifest repeats a required backend identity"))

        island_lists = (
            isl["required_weight_set_ids"],
            isl["required_weight_set_digests"],
            isl["required_prepared_image_ids"],
            isl["required_prepared_image_digests"],
        )
        if len({len(values) for values in island_lists}) != 1 or any(
                len(values) != len(set(values)) for values in island_lists):
            out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid}: island requirements are duplicated or not one-to-one"))
        for tuple_field in ("required_tuples", "satisfied_tuples"):
            tuples = [(
                item["weight_set_id"], item["prepared_image_id"],
                item["ready_certificate_id"], item["residency_lease_id"])
                for item in dd[tuple_field]]
            if len(tuples) != len(set(tuples)):
                out.append(err("E_DISPATCH_MISMATCH", f"dispatch {rid}: {tuple_field} contains duplicates"))

        served_ranges = []
        for tup in dd["satisfied_tuples"]:
            ws = ws_by.get(tup["weight_set_id"])
            pi = pi_by.get(tup["prepared_image_id"])
            rc = rc_by.get(tup["ready_certificate_id"])
            rl = rl_by.get(tup["residency_lease_id"])
            if not all((ws, pi, rc, rl)):
                continue

            if rc["weight_set_id"] != ws["weight_set_id"] or rc["weight_set_digest"] != ws["set_digest"]:
                out.append(err("E_CHAIN_BROKEN", f"dispatch {rid}: ready certificate is not for served weight set {ws['weight_set_id']}"))
            if rl["weight_set_id"] != ws["weight_set_id"] or rl["weight_set_digest"] != ws["set_digest"]:
                out.append(err("E_CHAIN_BROKEN", f"dispatch {rid}: residency lease is not for served weight set {ws['weight_set_id']}"))
            if rl["model_id"] != isl["model_id"] or rl["model_version"] != isl["model_version"]:
                out.append(err("E_CHAIN_BROKEN", f"dispatch {rid}: residency lease model identity differs from island"))

            allocation = alloc_by.get(pi["source_allocation_id"])
            state_lease = sl_by.get(dd["state_lease_id"])
            if allocation is not None:
                physical = rc["physical_bytes"]
                expected_physical = {
                    "canonical": allocation["canonical_bytes"],
                    "derived": pi["derived_bytes"],
                    "scratch": rl["reserved_bytes"]["scratch"],
                    "activations_reserved": state_lease["reserved_activation_bytes"] if state_lease else 0,
                    "state_reserved": state_lease["reserved_state_bytes"] if state_lease else 0,
                }
                if any(physical[name] != value for name, value in expected_physical.items()):
                    out.append(err("E_LEDGER_DERIVED", f"dispatch {rid}: ready certificate physical partition is not record-derived"))
            if di is not None:
                if rc["free_ram_after_bytes"] != di["physical_byte_accounting"]["free"]:
                    out.append(err("E_LEDGER_DERIVED", f"dispatch {rid}: ready certificate free RAM differs from device ledger"))
                if rc["issued_receiver_ts_us"] != di["receiver_ts_us"]:
                    out.append(err("E_SNAPSHOT_CAUSAL", f"dispatch {rid}: ready certificate and device inventory are not the same snapshot"))

            manifest_ws = next((entry for entry in mm["weight_sets"]
                                if entry["weight_set_id"] == ws["weight_set_id"]), None)
            if manifest_ws is None or any((
                    manifest_ws.get("set_digest") != ws["set_digest"],
                    manifest_ws.get("total_bytes") != ws["total_bytes"],
                    manifest_ws.get("layer_range") != ws["layer_range"],
                    ws["model_id"] != mm["model_id"],
                    ws["model_version"] != mm["model_version"],
                    ws["arch"] != mm["arch"],
                    ws["dtype"] != mm["dtype"],
                    ws["layout_version"] != mm["layout_version"],
            )):
                out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid}: served weight set {ws['weight_set_id']} is not the exact manifest entry"))
            else:
                lr = ws["layer_range"]
                served_ranges.append((lr["start"], lr["end"]))

            for segment_ref in ws["segments"]:
                segment = seg_by.get(segment_ref["segment_id"])
                segment_ok = segment is not None
                if segment_ok:
                    slr = segment["layer_range"]
                    wlr = ws["layer_range"]
                    segment_ok = all((
                        segment["sha256"] == segment_ref["sha256"],
                        segment["bytes"] == segment_ref["bytes"],
                        segment["weight_set_id"] == ws["weight_set_id"],
                        segment["model_id"] == ws["model_id"],
                        segment["model_version"] == ws["model_version"],
                        slr["n_layer_total"] == wlr["n_layer_total"],
                        wlr["start"] <= slr["start"] < slr["end"] <= wlr["end"],
                    ))
                if not segment_ok:
                    out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid}: segment {segment_ref['segment_id']} does not exactly resolve inside weight set {ws['weight_set_id']}"))
            segment_ids = [segment_ref["segment_id"] for segment_ref in ws["segments"]]
            if len(segment_ids) != len(set(segment_ids)):
                out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid}: weight set {ws['weight_set_id']} repeats a segment"))

            required_backend = next((entry for entry in mm["required_backends"]
                                     if entry["backend"] == pi["backend"]
                                     and entry["backend_build"] == pi["backend_build"]), None)
            if required_backend is not None and di is not None:
                minimum = required_backend.get("min_soc", "op12")
                if soc_rank[di["soc"]] < soc_rank[minimum]:
                    out.append(err("E_DEVICE_INELIGIBLE", f"dispatch {rid}: device {di['soc']} is below required {minimum}"))

            if rc["issued_receiver_ts_us"] > decision_ts:
                out.append(err("E_SNAPSHOT_CAUSAL", f"dispatch {rid}: ready certificate was issued after decision"))
            if rc["issued_receiver_ts_us"] > rl["horizon"]["start_us"]:
                out.append(err("E_SNAPSHOT_CAUSAL", f"dispatch {rid}: residency lease starts before ready certificate issuance"))

        target = (island_range["start"], island_range["end"])
        if not _covers_layer_range(served_ranges, target):
            out.append(err("E_IDENTITY_MISMATCH", f"dispatch {rid}: served weight-set ranges do not cover island layer range"))

    # A ticket is an exact authorization for a contiguous set of whole chunks on one
    # device. The declared payload digest still has to be checked against bytes at runtime.
    for tt in tt_by.values():
        seg = seg_by.get(tt["segment_id"])
        ws = ws_by.get(tt["weight_set_id"])
        di = di_by.get(tt["device_id"])
        problems = []
        if seg is None:
            out.append(err("E_FRAME_BINDING", f"transfer ticket {tt['ticket_id']}: segment is absent"))
            continue
        if ws is None:
            problems.append("ticket weight set is absent")
        if di is None:
            problems.append("target device has no inventory")
        if seg["weight_set_id"] != tt["weight_set_id"]:
            problems.append("segment does not belong to ticket weight set")
        member = next((entry for entry in ws["segments"] if entry["segment_id"] == seg["segment_id"]), None) \
            if ws is not None else None
        if member is None or member["sha256"] != seg["sha256"] or member["bytes"] != seg["bytes"]:
            problems.append("segment is not an exact member of ticket weight set")
        if ws is not None and (tt["model_id"] != ws["model_id"] or tt["model_version"] != ws["model_version"]):
            problems.append("ticket model identity differs from weight set")
        if tt["expected_sha256"] != seg["sha256"]:
            problems.append("ticket digest differs from segment")
        first = tt["chunk_range"]["first"]
        last = tt["chunk_range"]["last"]
        chunks = seg["chunks"]
        if first <= last < len(chunks):
            expected_offset = chunks[first]["offset"]
            expected_end = chunks[last]["offset"] + chunks[last]["bytes"]
            if tt["byte_range"] != {"offset": expected_offset, "length": expected_end - expected_offset}:
                problems.append("ticket byte range is not its exact whole-chunk span")
        else:
            problems.append("ticket chunk range is outside segment")
        for problem in problems:
            out.append(err("E_FRAME_BINDING", f"transfer ticket {tt['ticket_id']}: {problem}"))

    seq_seen = set()
    idem_seen = set()
    for tf in lists["transport_frame"]:
        seq_key = (tf["device_id"], tf["boot_epoch"], tf["channel"], tf["seq"])
        idem_key = (tf["device_id"], tf["idempotency_key"])
        if seq_key in seq_seen:
            out.append(err("E_FRAME_DUPLICATE", f"duplicate frame sequence identity {seq_key}"))
        if idem_key in idem_seen:
            out.append(err("E_FRAME_DUPLICATE", f"duplicate frame idempotency identity {idem_key}"))
        seq_seen.add(seq_key)
        idem_seen.add(idem_key)

        if tf["channel"] != "bulk":
            continue
        binding = tf["bulk_binding"]
        tt = tt_by.get(binding["ticket_id"])
        if tt is None:
            continue
        chunk_index = binding["chunk_index"]
        first = tt["chunk_range"]["first"]
        last = tt["chunk_range"]["last"]
        if tf["request_id"] != tt["ticket_id"]:
            out.append(err("E_FRAME_BINDING", f"bulk frame request_id does not equal ticket_id {tt['ticket_id']}"))
        if not first <= chunk_index <= last:
            out.append(err("E_FRAME_BINDING", f"bulk frame chunk {chunk_index} is outside ticket chunk range"))
        if tf["payload_bytes"] != binding["chunk_length"]:
            out.append(err("E_FRAME_BINDING", f"bulk frame payload length differs from bound chunk length"))
    return out


def validate_bundle(path):
    """Returns a list of (code, message). Empty list == valid."""
    try:
        bundle = json.loads(open(path).read(), object_pairs_hook=_no_dup_hook)
    except DupKey as ex:
        return [err("E_DUPLICATE_KEY", f"duplicate JSON key {ex}")]
    except Exception as ex:
        return [err("E_JSON_PARSE", str(ex))]

    if not isinstance(bundle, dict):
        return [err("E_ENVELOPE", "bundle must be a JSON object")]
    bver = bundle.get("bundle_version")
    if bver == 2:
        kinds, sdir, sver, builders = V2_KINDS, SCHEMAS_V2, 2, s9lib.DIGEST_BUILDERS
    elif bver == 3:
        kinds, sdir, sver, builders = V3_KINDS, SCHEMAS_V3, 3, s9lib.DIGEST_BUILDERS
    elif bver == 4:
        kinds, sdir, sver, builders = V4_KINDS, SCHEMAS_V4, 4, s9lib.DIGEST_BUILDERS_V4
    elif bver == 5:
        kinds, sdir, sver, builders = V5_KINDS, SCHEMAS_V5, 5, s9lib.DIGEST_BUILDERS_V5
    else:
        return [err("E_ENVELOPE", f"bundle_version {bver!r} not supported (expected 2, 3, 4, or 5)")]
    envelope = os.path.join(sdir, "bundle.schema.json")

    rc, out = _js_validate(envelope, bundle)
    if rc != 0:
        return [err("E_ENVELOPE", out.strip()[:200])]

    records = bundle["records"]
    errors = _safe_ints(records, "records")
    schema_errors = []
    for i, rec in enumerate(records):
        schema_errors += schema_check(rec, i, kinds, sdir, sver)
    errors += schema_errors
    # Cross-record code assumes the required shape. Never dereference a record after a
    # schema failure; malformed tuples previously reached _tset() and raised KeyError.
    if schema_errors:
        return errors
    for i, rec in enumerate(records):
        errors += digest_check(rec, i, builders)
        errors += semantic_check(rec, i)
    errors += cross_record(records)
    if bver in (3, 4, 5):
        errors += cross_record_r1(records)
    if bver in (4, 5):
        errors += cross_record_v4(records)
    if bver == 5:
        errors += cross_record_v5(records)
    return errors


def selftest(index_rel):
    base = os.path.join(HERE, os.path.dirname(index_rel))
    idx = json.load(open(os.path.join(HERE, index_rel)))
    fails = 0
    listed = {rec["file"] for rec in idx}
    if len(listed) != len(idx):
        print("  FAIL fixture index contains duplicate paths")
        fails += 1
    actual = set()
    for sub in ("valid", "invalid"):
        d = os.path.join(base, sub)
        if os.path.isdir(d):
            actual.update(os.path.join(sub, name) for name in os.listdir(d) if name.endswith(".json"))
    if listed != actual:
        print(f"  FAIL fixture index mismatch missing={sorted(actual - listed)} phantom={sorted(listed - actual)}")
        fails += 1
    for rec in idx:
        path = os.path.join(base, rec["file"])
        if not (os.path.exists(path) and os.path.isfile(path)):
            print(f"  FAIL missing fixture {rec['file']}")
            fails += 1
            continue
        errs = validate_bundle(path)
        codes_set = sorted({c for c, _ in errs})
        ok = (len(errs) == 0) == (rec["expect"] == "valid")
        if "expected_codes" in rec:
            ok = ok and codes_set == sorted(rec["expected_codes"])
        if not ok:
            fails += 1
        codes = ",".join(codes_set) if errs else "-"
        print(f"  {'PASS' if ok else 'FAIL'} [{rec['expect']:7}] {rec['file']:36} codes={codes}")
    print(f"\n{index_rel}: {len(idx)} fixtures  failures: {fails}")
    return fails


def main(argv):
    if "--selftest" in argv:
        n = selftest("fixtures/v2/bundles/index.json")
        n += selftest("fixtures/v3/bundles/index.json")
        n += selftest("fixtures/v4/bundles/index.json")
        n += selftest("fixtures/v5/bundles/index.json")
        return 1 if n else 0
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print("usage: bundle_validate.py <bundle.json> | --selftest")
        return 2
    errs = validate_bundle(args[0])
    if errs:
        for c, m in errs:
            print(f"INVALID [{c}] {m}")
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
