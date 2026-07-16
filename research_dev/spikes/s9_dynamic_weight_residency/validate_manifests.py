#!/usr/bin/env python3
"""S9 semantic validator (V0).

Checks the cross-field / arithmetic / digest constraints that JSON Schema cannot
express. Schema validity is a PRE-REQUISITE (run run_schema_tests.py first); this
adds the semantic layer.

Checks:
  weight_segment  : chunks tile [0,bytes) exactly -- indices 0..n-1, offsets
                    ascending + contiguous from 0, sum(chunk bytes) == segment bytes
  weight_set      : set_digest == sha256(LF-join(sorted segment sha256));
                    total_bytes == sum(segment bytes)
  model_manifest  : partial_load_supported == false => exactly one weight set whose
                    layer_range covers [0, n_layer_total)
  prepared_image  : derived_image_digest == sha256(canonical_json(9 binding fields));
                    tensor_digest == source_weight_set_digest
  ready_certificate: physical_bytes.total == sum of its parts
  device_inventory: ledger partitions sum to lpddr.total_bytes; no field > total
  residency_lease : horizon.end_us >= start_us + min_hold_us
  transfer_ticket : chunk_range.first <= last
  all             : every integer <= 2^53-1

CLI:
  validate_manifests.py <file> [--kind KIND]   -> exit 0 valid / 1 invalid
  validate_manifests.py --selftest             -> run fixtures/semantic/*
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SAFE_MAX = 2 ** 53 - 1


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha_over(text):
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def set_digest(seg_hashes):
    return sha_over("\n".join(sorted(seg_hashes)))


def derived_digest(m):
    fields = {
        "arch": m["arch"], "backend_build": m["backend_build"], "boot_epoch": m["boot_epoch"],
        "graph_hash": m["graph_hash"], "layout_version": m["layout_version"],
        "model_version": m["model_version"], "residency_generation": m["residency_generation"],
        "soc": m["soc"], "tensor_digest": m["tensor_digest"],
    }
    return sha_over(canonical(fields))


def _safe_ints(obj, path="$"):
    errs = []
    if isinstance(obj, bool):
        return errs
    if isinstance(obj, int):
        if obj > SAFE_MAX or obj < -SAFE_MAX:
            errs.append(f"{path}: integer {obj} exceeds safe bound 2^53-1")
    elif isinstance(obj, dict):
        for k, val in obj.items():
            errs += _safe_ints(val, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            errs += _safe_ints(val, f"{path}[{i}]")
    return errs


def check_weight_segment(m):
    e = []
    total = m.get("bytes", 0)
    chunks = m.get("chunks", [])
    off = 0
    for i, c in enumerate(chunks):
        if c.get("index") != i:
            e.append(f"chunk[{i}].index {c.get('index')} != {i}")
        if c.get("offset") != off:
            e.append(f"chunk[{i}].offset {c.get('offset')} != expected {off}")
        off += c.get("bytes", 0)
    if off != total:
        e.append(f"chunk bytes sum {off} != segment bytes {total}")
    return e


def check_weight_set(m):
    e = []
    seg_hashes = [s.get("sha256") for s in m.get("segments", [])]
    seg_bytes = sum(s.get("bytes", 0) for s in m.get("segments", []))
    if m.get("set_digest") != set_digest(seg_hashes):
        e.append("set_digest does not match sha256(LF-join(sorted segment sha256))")
    if m.get("total_bytes") != seg_bytes:
        e.append(f"total_bytes {m.get('total_bytes')} != sum(segment bytes) {seg_bytes}")
    return e


def check_model_manifest(m):
    e = []
    if m.get("partial_load_supported") is False:
        ws = m.get("weight_sets", [])
        n = m.get("n_layer_total")
        if len(ws) != 1:
            e.append(f"partial_load_supported=false requires exactly 1 weight set, got {len(ws)}")
        elif ws:
            lr = ws[0].get("layer_range", {})
            if not (lr.get("start") == 0 and lr.get("end") == n and lr.get("n_layer_total") == n):
                e.append(f"partial_load_supported=false requires the single weight set to cover [0,{n}); got {lr}")
    return e


def check_prepared_image(m):
    e = []
    if m.get("derived_image_digest") != derived_digest(m):
        e.append("derived_image_digest does not match sha256(canonical(9 binding fields))")
    if m.get("tensor_digest") != m.get("source_weight_set_digest"):
        e.append("tensor_digest != source_weight_set_digest")
    return e


def check_ready_certificate(m):
    e = []
    pb = m.get("physical_bytes", {})
    parts = pb.get("canonical", 0) + pb.get("derived", 0) + pb.get("scratch", 0) + \
        pb.get("activations_reserved", 0) + pb.get("state_reserved", 0)
    if pb.get("total") != parts:
        e.append(f"physical_bytes.total {pb.get('total')} != sum of parts {parts}")
    return e


def check_device_inventory(m):
    e = []
    a = m.get("physical_byte_accounting", {})
    total = m.get("lpddr", {}).get("total_bytes")
    fields = ["weights_resident", "derived_images", "scratch", "activations", "mutable_state", "free"]
    s = sum(a.get(f, 0) for f in fields)
    if s != total:
        e.append(f"physical_byte_accounting sums to {s} != lpddr.total_bytes {total}")
    for f in fields:
        if a.get(f, 0) > total:
            e.append(f"physical_byte_accounting.{f} {a.get(f)} exceeds lpddr.total_bytes {total}")
    return e


def check_residency_lease(m):
    e = []
    hz = m.get("horizon", {})
    floor = hz.get("start_us", 0) + m.get("min_hold_us", 0)
    if hz.get("end_us", 0) < floor:
        e.append(f"horizon.end_us {hz.get('end_us')} < start_us+min_hold_us {floor}")
    return e


def check_transfer_ticket(m):
    e = []
    cr = m.get("chunk_range", {})
    if cr.get("first", 0) > cr.get("last", 0):
        e.append(f"chunk_range.first {cr.get('first')} > last {cr.get('last')}")
    return e


CHECKS = {
    "weight_segment": check_weight_segment,
    "weight_set": check_weight_set,
    "model_manifest": check_model_manifest,
    "prepared_image": check_prepared_image,
    "ready_certificate": check_ready_certificate,
    "device_inventory": check_device_inventory,
    "residency_lease": check_residency_lease,
    "transfer_ticket": check_transfer_ticket,
}


def detect_kind(m):
    if "chunks" in m and "segment_id" in m:
        return "weight_segment"
    if "segments" in m and "set_digest" in m:
        return "weight_set"
    if "weight_sets" in m and "partial_load_supported" in m:
        return "model_manifest"
    if "derived_image_digest" in m:
        return "prepared_image"
    if "physical_bytes" in m and "warmup_passed" in m:
        return "ready_certificate"
    if "physical_byte_accounting" in m:
        return "device_inventory"
    if "horizon" in m and "min_hold_us" in m:
        return "residency_lease"
    if "chunk_range" in m and "ticket_id" in m:
        return "transfer_ticket"
    return None


def validate_file(path, kind=None):
    m = json.load(open(path))
    kind = kind or detect_kind(m)
    errs = _safe_ints(m)
    if kind in CHECKS:
        errs += CHECKS[kind](m)
    elif kind is None:
        errs.append("could not detect record kind (pass --kind)")
    return errs


def selftest():
    base = os.path.join(HERE, "fixtures", "semantic")
    idx = json.load(open(os.path.join(base, "index.json")))
    fails = 0
    for rec in idx:
        path = os.path.join(base, rec["file"])
        if not (os.path.exists(path) and os.path.isfile(path)):
            print(f"  FAIL missing fixture {rec['file']}")
            fails += 1
            continue
        try:
            errs = validate_file(path, rec.get("kind"))
        except Exception as ex:
            print(f"  FAIL parse/validate error {rec['file']}: {ex}")
            fails += 1
            continue
        got_valid = (len(errs) == 0)
        want_valid = (rec["expect"] == "valid")
        ok = got_valid == want_valid
        if not ok:
            fails += 1
        print(f"  {'PASS' if ok else 'FAIL'} [{rec['expect']:7}] {rec['file']:46} errs={errs if errs else '-'}")
    print(f"\nsemantic fixtures: {len(idx)}  failures: {fails}")
    return 1 if fails else 0


def main(argv):
    if "--selftest" in argv:
        return selftest()
    kind = None
    if "--kind" in argv:
        kind = argv[argv.index("--kind") + 1]
    args = [a for a in argv if not a.startswith("--") and a != kind]
    if not args:
        print("usage: validate_manifests.py <file> [--kind KIND] | --selftest")
        return 2
    errs = validate_file(args[0], kind)
    if errs:
        for x in errs:
            print("INVALID:", x)
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
