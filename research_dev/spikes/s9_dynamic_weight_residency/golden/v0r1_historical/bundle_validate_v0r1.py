#!/usr/bin/env python3
"""S9-V0-R1 strict bundle validator (bundle versions 2 and 3).

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

Every rejection carries a stable error code. Schema validation uses PRIVATE per-call
temp files (no shared /tmp names), so concurrent validators cannot race. Digests are
recomputed from s9lib (shared with the fixture generator).
"""
import json
import os
import subprocess
import sys
import tempfile

import s9lib_v0r1 as s9lib
from s9lib_v0r1 import canonical, set_digest, SAFE_MAX, V2_KINDS, V3_KINDS

HERE = os.path.dirname(os.path.abspath(__file__))
# frozen under golden/v0r1_historical/: schemas live two levels up (never edited here)
SCHEMAS_V2 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "schemas", "v2")
SCHEMAS_V3 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "schemas", "v3")
JSONSCHEMA = "/usr/bin/jsonschema"

CODES = [
    "E_JSON_PARSE", "E_DUPLICATE_KEY", "E_ENVELOPE", "E_UNKNOWN_KIND", "E_UNKNOWN_VERSION",
    "E_SCHEMA", "E_SAFE_INT", "E_DIGEST_MISMATCH", "E_CHUNK_TILING", "E_TOTAL_MISMATCH",
    "E_LEDGER", "E_HORIZON", "E_RANGE", "E_ID_COLLISION", "E_ALLOC_REFCOUNT",
    "E_MISSING_RECORD", "E_TUPLE_MISMATCH", "E_CORRECTNESS_BINDING", "E_FRAME_BINDING",
    "E_RESUME_PREFIX", "E_EPOCH_MISMATCH",
    # R1 (bundle_version 3) additions:
    "E_ISLAND_ABSENT", "E_CHAIN_BROKEN", "E_DISPATCH_MISMATCH", "E_STATE_LEASE", "E_ALIAS_SET",
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


def digest_check(rec, idx):
    kind = rec.get("kind")
    out = []
    if kind in s9lib.DIGEST_BUILDERS:
        field, builder = s9lib.DIGEST_BUILDERS[kind]
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


def validate_bundle(path):
    """Returns a list of (code, message). Empty list == valid."""
    try:
        bundle = json.loads(open(path).read(), object_pairs_hook=_no_dup_hook)
    except DupKey as ex:
        return [err("E_DUPLICATE_KEY", f"duplicate JSON key {ex}")]
    except Exception as ex:
        return [err("E_JSON_PARSE", str(ex))]

    bver = bundle.get("bundle_version")
    if bver == 2:
        kinds, sdir, sver, envelope = V2_KINDS, SCHEMAS_V2, 2, os.path.join(SCHEMAS_V2, "bundle.schema.json")
    elif bver == 3:
        kinds, sdir, sver, envelope = V3_KINDS, SCHEMAS_V3, 3, os.path.join(SCHEMAS_V3, "bundle.schema.json")
    else:
        return [err("E_ENVELOPE", f"bundle_version {bver!r} not supported (expected 2 or 3)")]

    rc, out = _js_validate(envelope, bundle)
    if rc != 0:
        return [err("E_ENVELOPE", out.strip()[:200])]

    records = bundle["records"]
    errors = _safe_ints(records, "records")
    for i, rec in enumerate(records):
        se = schema_check(rec, i, kinds, sdir, sver)
        errors += se
        if se:
            continue
        errors += digest_check(rec, i)
        errors += semantic_check(rec, i)
    errors += cross_record(records)
    if bver == 3:
        errors += cross_record_r1(records)
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
