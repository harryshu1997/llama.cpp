#!/usr/bin/env python3
"""S9-V0-R strict bundle validator (bundle version 2).

ONE entrypoint that ALWAYS runs three layers over a record bundle and fails closed:
  1. JSON Schema   -- envelope + each record routed by (kind, schema_version) to its
                      v2 schema via /usr/bin/jsonschema (the pinned validator).
  2. semantic      -- per-record arithmetic / digest recomputation JSON Schema cannot do.
  3. cross-record  -- referenced records exist, referenced digests match, dispatch
                      tuples are EXACTLY required==satisfied, partial ranges obey the
                      declared gap/overlap policy, ledgers partition, ids are unique.

Every rejection carries a STABLE error code (see CODES). Duplicate JSON keys, unknown
record kinds, unknown schema versions, missing referenced records, and mismatched
digests are all rejected. Digests are recomputed from s9lib (shared with the fixture
generator) so the validator and the fixtures cannot drift.

CLI:
  bundle_validate.py <bundle.json>   -> exit 0 valid / 1 invalid (prints codes)
  bundle_validate.py --selftest      -> run fixtures/v2/bundles/index.json
"""
import json
import os
import subprocess
import sys

import s9lib_v0r as s9lib
from s9lib_v0r import canonical, set_digest, SAFE_MAX, V2_KINDS

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMAS_V2 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "schemas", "v2")
JSONSCHEMA = "/usr/bin/jsonschema"

CODES = [
    "E_JSON_PARSE", "E_DUPLICATE_KEY", "E_ENVELOPE", "E_UNKNOWN_KIND", "E_UNKNOWN_VERSION",
    "E_SCHEMA", "E_SAFE_INT", "E_DIGEST_MISMATCH", "E_CHUNK_TILING", "E_TOTAL_MISMATCH",
    "E_LEDGER", "E_HORIZON", "E_RANGE", "E_ID_COLLISION", "E_ALLOC_REFCOUNT",
    "E_MISSING_RECORD", "E_TUPLE_MISMATCH", "E_CORRECTNESS_BINDING", "E_FRAME_BINDING",
    "E_RESUME_PREFIX", "E_EPOCH_MISMATCH",
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


def load_bundle_text(path):
    with open(path, "r") as f:
        return f.read()


def err(code, msg):
    return (code, msg)


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


def schema_check(rec, idx):
    kind = rec.get("kind")
    ver = rec.get("schema_version")
    if kind not in V2_KINDS:
        return [err("E_UNKNOWN_KIND", f"record[{idx}] kind={kind!r} not a bundle-v2 kind")]
    if ver != 2:
        return [err("E_UNKNOWN_VERSION", f"record[{idx}] kind={kind} schema_version={ver!r} (bundle v2 requires 2)")]
    schema = os.path.join(SCHEMAS_V2, V2_KINDS[kind])
    tmp = os.path.join("/tmp", f"s9_bv_rec_{idx}.json")
    with open(tmp, "w") as f:
        f.write(json.dumps(rec))
    r = subprocess.run([JSONSCHEMA, "-i", tmp, schema], capture_output=True, text=True)
    if "Traceback (most recent call last)" in r.stderr:
        return [err("E_SCHEMA", f"record[{idx}] {kind}: validator crash {r.stderr.strip()[:120]}")]
    if r.returncode != 0:
        return [err("E_SCHEMA", f"record[{idx}] {kind}: {(r.stdout + r.stderr).strip()[:200]}")]
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
        floor = hz.get("start_us", 0) + rec.get("min_hold_us", 0)
        if hz.get("end_us", 0) < floor:
            out.append(err("E_HORIZON", f"record[{idx}] residency_lease horizon.end_us < start+min_hold"))
    elif kind == "canonical_allocation":
        if rec.get("alias_refcount") != len(rec.get("alias_prepared_image_ids", [])):
            out.append(err("E_ALLOC_REFCOUNT", f"record[{idx}] alias_refcount != len(alias_prepared_image_ids)"))
        want_reclaim = (rec.get("alias_refcount", 0) == 0 and rec.get("lease_refcount", 0) == 0)
        if rec.get("reclaimable") != want_reclaim:
            out.append(err("E_ALLOC_REFCOUNT", f"record[{idx}] reclaimable must be {want_reclaim} for these refcounts"))
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
    by = {}
    for k in V2_KINDS:
        by[k] = {}
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
            out.append(err("E_RANGE", f"weight_set {wid}: layer_range.n_layer_total {nlt} != manifest n_layer_total {n}"))
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
                out.append(err("E_RANGE", f"contiguous_partition: coverage ends at {cursor} != n_layer_total {n}"))
    elif policy == "sharded_disjoint":
        prev_end = 0
        for st, en in ranges:
            if st < prev_end:
                out.append(err("E_RANGE", f"sharded_disjoint: overlap at layer {st} (prev end {prev_end})"))
            prev_end = max(prev_end, en)
    return out


def cross_record(records):
    out = []
    by, lists, collisions = _index(records)
    out += collisions
    ws_by = by["weight_set"]
    seg_by = by["weight_segment"]
    alloc_by = by["canonical_allocation"]
    pi_by = by["prepared_image"]
    cc_by = by["correctness_certificate"]
    isl_by = by["island_executable"]
    rc_by = by["ready_certificate"]
    rl_by = by["residency_lease"]
    sl_by = by["state_lease"]
    tt_by = by["transfer_ticket"]

    # model manifest: ranges + per-set totals/digests against present WeightSet records
    for mm in by["model_manifest"].values():
        out += _check_ranges(mm)
        for ws in mm.get("weight_sets", []):
            wsr = ws_by.get(ws.get("weight_set_id"))
            if wsr is not None:
                if wsr.get("total_bytes") != ws.get("total_bytes"):
                    out.append(err("E_TOTAL_MISMATCH", f"manifest weight_set {ws['weight_set_id']} total_bytes != WeightSet record"))
                if wsr.get("set_digest") != ws.get("set_digest"):
                    out.append(err("E_DIGEST_MISMATCH", f"manifest weight_set {ws['weight_set_id']} set_digest != WeightSet record"))

    # prepared image: source allocation present with matching digest
    for pi in pi_by.values():
        al = alloc_by.get(pi.get("source_allocation_id"))
        if al is None:
            out.append(err("E_MISSING_RECORD", f"prepared_image {pi['prepared_image_id']} source_allocation_id absent"))
        elif al.get("weight_set_digest") != pi.get("source_weight_set_digest"):
            out.append(err("E_DIGEST_MISMATCH", f"prepared_image {pi['prepared_image_id']} source_weight_set_digest != allocation"))

    # canonical allocation: aliases resolve, lease count sane
    for al in alloc_by.values():
        for pid in al.get("alias_prepared_image_ids", []):
            pi = pi_by.get(pid)
            if pi is None:
                out.append(err("E_MISSING_RECORD", f"allocation {al['allocation_id']} alias {pid} absent"))
            elif pi.get("source_allocation_id") != al.get("allocation_id"):
                out.append(err("E_ALLOC_REFCOUNT", f"allocation {al['allocation_id']} alias {pid} does not point back"))
        live_leases = sum(1 for rl in rl_by.values() if rl.get("source_allocation_id") == al.get("allocation_id"))
        if live_leases > al.get("lease_refcount", 0):
            out.append(err("E_ALLOC_REFCOUNT", f"allocation {al['allocation_id']} has {live_leases} leases > lease_refcount"))

    # ready certificate: referenced prepared image + correctness digests match
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

    # residency lease: referenced ready certificate + allocation
    for rl in rl_by.values():
        rc = rc_by.get(rl.get("ready_certificate_id"))
        if rc is None:
            out.append(err("E_MISSING_RECORD", f"residency_lease {rl['residency_lease_id']} ready_certificate absent"))
        elif rc.get("ready_certificate_digest") != rl.get("ready_certificate_digest"):
            out.append(err("E_DIGEST_MISMATCH", f"residency_lease {rl['residency_lease_id']} ready_certificate_digest mismatch"))
        if rl.get("source_allocation_id") not in alloc_by:
            out.append(err("E_MISSING_RECORD", f"residency_lease {rl['residency_lease_id']} source_allocation absent"))

    # island + correctness reciprocal binding
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

    # state lease: residency dependency present + generation matches
    for sl in sl_by.values():
        rl = rl_by.get(sl.get("depends_on_residency_lease_id"))
        if rl is None:
            out.append(err("E_MISSING_RECORD", f"state_lease {sl['state_lease_id']} residency dependency absent"))
        elif rl.get("residency_generation") != sl.get("depends_on_residency_generation"):
            out.append(err("E_EPOCH_MISMATCH", f"state_lease {sl['state_lease_id']} depends_on_residency_generation mismatch"))
        if sl.get("island_id") not in isl_by:
            out.append(err("E_MISSING_RECORD", f"state_lease {sl['state_lease_id']} island absent"))

    # dispatch decisions: exact required==satisfied tuple set, satisfied records present
    def tset(tuples):
        return sorted((t["weight_set_id"], t["prepared_image_id"], t["ready_certificate_id"], t["residency_lease_id"]) for t in tuples)

    for dd in lists["dispatch_decision"]:
        if dd.get("verdict") != "DISPATCH":
            continue
        req = tset(dd.get("required_tuples", []))
        sat = tset(dd.get("satisfied_tuples", []))
        if req != sat:
            out.append(err("E_TUPLE_MISMATCH", f"dispatch {dd['request_id']}: required tuples != satisfied tuples"))
        isl = isl_by.get(dd.get("island_id"))
        if isl is not None:
            need = sorted(isl.get("required_weight_set_ids", []))
            got = sorted(t["weight_set_id"] for t in dd.get("required_tuples", []))
            if need != got:
                out.append(err("E_TUPLE_MISMATCH", f"dispatch {dd['request_id']}: required tuples do not cover island required weight sets"))
        for t in dd.get("satisfied_tuples", []):
            if t["weight_set_id"] not in ws_by or t["prepared_image_id"] not in pi_by \
                    or t["ready_certificate_id"] not in rc_by or t["residency_lease_id"] not in rl_by:
                out.append(err("E_MISSING_RECORD", f"dispatch {dd['request_id']}: a satisfied tuple references an absent record"))

    # transfer tickets: expected segment digest + chunk_range within segment
    for tt in tt_by.values():
        seg = seg_by.get(tt.get("segment_id"))
        if seg is not None:
            if seg.get("sha256") != tt.get("expected_sha256"):
                out.append(err("E_DIGEST_MISMATCH", f"transfer_ticket {tt['ticket_id']} expected_sha256 != segment sha256"))
            ncnk = len(seg.get("chunks", []))
            if tt.get("chunk_range", {}).get("last", 0) >= ncnk:
                out.append(err("E_FRAME_BINDING", f"transfer_ticket {tt['ticket_id']} chunk_range.last out of segment chunk list"))

    # bulk transport frames: binding matches ticket + the segment's chunk
    for tf in lists["transport_frame"]:
        if tf.get("channel") != "bulk":
            continue
        bb = tf.get("bulk_binding") or {}
        tt = tt_by.get(bb.get("ticket_id"))
        if tt is None:
            out.append(err("E_MISSING_RECORD", f"bulk frame ticket {bb.get('ticket_id')} absent"))
            continue
        if bb.get("segment_id") != tt.get("segment_id"):
            out.append(err("E_FRAME_BINDING", f"bulk frame segment {bb.get('segment_id')} != ticket segment"))
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
                    out.append(err("E_FRAME_BINDING", f"bulk frame binding does not match segment chunk[{ci}]"))
    return out


def validate_bundle(path):
    """Returns a list of (code, message). Empty list == valid."""
    text = load_bundle_text(path)
    try:
        bundle = json.loads(text, object_pairs_hook=_no_dup_hook)
    except DupKey as ex:
        return [err("E_DUPLICATE_KEY", f"duplicate JSON key {ex}")]
    except Exception as ex:
        return [err("E_JSON_PARSE", str(ex))]

    errors = []
    # envelope
    tmp = os.path.join("/tmp", "s9_bv_envelope.json")
    with open(tmp, "w") as f:
        f.write(json.dumps(bundle))
    r = subprocess.run([JSONSCHEMA, "-i", tmp, os.path.join(SCHEMAS_V2, "bundle.schema.json")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        errors.append(err("E_ENVELOPE", (r.stdout + r.stderr).strip()[:200]))
        return errors
    if bundle.get("bundle_version") != s9lib.BUNDLE_VERSION:
        errors.append(err("E_ENVELOPE", f"bundle_version {bundle.get('bundle_version')} != {s9lib.BUNDLE_VERSION}"))
        return errors

    records = bundle["records"]
    errors += _safe_ints(records, "records")
    for i, rec in enumerate(records):
        se = schema_check(rec, i)
        errors += se
        if se:
            continue                       # skip digest/semantic on a record that failed schema
        errors += digest_check(rec, i)
        errors += semantic_check(rec, i)
    errors += cross_record(records)
    return errors


def selftest():
    base = os.path.join(HERE, "fixtures", "v2", "bundles")
    idx = json.load(open(os.path.join(base, "index.json")))
    fails = 0
    for rec in idx:
        path = os.path.join(base, rec["file"])
        if not (os.path.exists(path) and os.path.isfile(path)):
            print(f"  FAIL missing fixture {rec['file']}")
            fails += 1
            continue
        errs = validate_bundle(path)
        got_valid = (len(errs) == 0)
        want_valid = (rec["expect"] == "valid")
        ok = got_valid == want_valid
        if not ok:
            fails += 1
        codes = ",".join(sorted({c for c, _ in errs})) if errs else "-"
        print(f"  {'PASS' if ok else 'FAIL'} [{rec['expect']:7}] {rec['file']:36} codes={codes}")
    print(f"\nbundle fixtures: {len(idx)}  failures: {fails}")
    return 1 if fails else 0


def main(argv):
    if "--selftest" in argv:
        return selftest()
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
