#!/usr/bin/env python3
"""S8 semantic manifest validator (V0b-P0).

Checks the cross-field / arithmetic constraints that JSON Schema cannot express,
for trace sidecar manifests and normalize artifact manifests. Schema validity is a
PRE-REQUISITE (run run_schema_tests.py first); this adds semantic checks.

Checks (trace_manifest):
  - window bin arithmetic: t_start_us == bin_index*W and t_end_us == (bin_index+1)*W (W=900000000)
  - source_row_first <= source_row_last
  - 1 <= quantile_rank <= n_nonempty_bins (when both present)
  - input_nonmonotonic_pairs > 0 requires timestamp_policy == sort_stable
  - output_row_count >= 1 (row-count agreement)
  - mix: stream ranks unique AND streams serialized in ascending rank order
  - mix: each stream binds BOTH component hashes, and no two streams bind the
    same (input_output_sha256, input_manifest_sha256) component
  - safe-integer bounds: every integer <= 2^53-1

Checks (artifact_manifest, kind=normalize):
  - outputs non-empty and each output carries sidecar_manifest_sha256
  - safe-integer bounds

CLI:
  validate_manifests.py <file> [--kind trace|artifact]   -> exit 0 valid / 1 invalid
  validate_manifests.py --selftest                        -> run fixtures/semantic/*
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
W = 900000000
SAFE_MAX = 2 ** 53 - 1  # 9007199254740991


def _safe_ints(obj, path="$"):
    errs = []
    if isinstance(obj, bool):
        return errs
    if isinstance(obj, int):
        if obj > SAFE_MAX or obj < -SAFE_MAX:
            errs.append(f"{path}: integer {obj} exceeds safe bound 2^53-1")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            errs += _safe_ints(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            errs += _safe_ints(v, f"{path}[{i}]")
    return errs


def check_trace_manifest(m):
    e = []
    e += _safe_ints(m)
    prov = m.get("provenance")
    if prov in ("real", "real_decomposed"):
        w = m.get("window", {})
        bi = w.get("bin_index")
        if bi is not None:
            if w.get("t_start_us") != bi * W:
                e.append(f"window.t_start_us {w.get('t_start_us')} != bin_index*W {bi*W}")
            if w.get("t_end_us") != (bi + 1) * W:
                e.append(f"window.t_end_us {w.get('t_end_us')} != (bin_index+1)*W {(bi+1)*W}")
        if w.get("source_row_first", 0) > w.get("source_row_last", 0):
            e.append("window.source_row_first > source_row_last")
        nb, qr = w.get("n_nonempty_bins"), w.get("quantile_rank")
        if nb is not None and qr is not None and not (1 <= qr <= nb):
            e.append(f"quantile_rank {qr} not in [1,{nb}]")
        if w.get("input_nonmonotonic_pairs", 0) > 0 and w.get("timestamp_policy") != "sort_stable":
            e.append("input_nonmonotonic_pairs>0 requires timestamp_policy sort_stable")
        if m.get("output_row_count", 0) < 1:
            e.append("output_row_count < 1")
    elif prov == "semi_synthetic":
        streams = m.get("streams", [])
        ranks = [s.get("rank") for s in streams]
        if len(set(ranks)) != len(ranks):
            e.append("stream ranks are not unique")
        if ranks != sorted(ranks):
            e.append("streams are not serialized in ascending rank order")
        comps = [(s.get("input_output_sha256"), s.get("input_manifest_sha256")) for s in streams]
        for i, c in enumerate(comps):
            if not (c[0] and c[1]):
                e.append(f"stream[{i}] missing a component hash")
        if len(set(comps)) != len(comps):
            e.append("two streams bind the same component (duplicate hash pair)")
    return e


def check_artifact_manifest(m):
    e = []
    e += _safe_ints(m)
    if m.get("kind") == "normalize":
        outs = m.get("outputs", [])
        if len(outs) < 1:
            e.append("normalize artifact has no outputs")
        for i, o in enumerate(outs):
            if not o.get("sidecar_manifest_sha256"):
                e.append(f"normalize output[{i}] missing sidecar_manifest_sha256")
    return e


def detect_kind(m):
    if "kind" in m and "run_id" in m:
        return "artifact"
    return "trace"


def validate_file(path, kind=None):
    m = json.load(open(path))
    kind = kind or detect_kind(m)
    errs = check_artifact_manifest(m) if kind == "artifact" else check_trace_manifest(m)
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
        print(f"  {'PASS' if ok else 'FAIL'} [{rec['expect']:7}] {rec['file']:44} errs={errs if errs else '-'}")
    print(f"\nsemantic fixtures: {len(idx)}  failures: {fails}")
    return 1 if fails else 0


def main(argv):
    if "--selftest" in argv:
        return selftest()
    args = [a for a in argv if not a.startswith("--")]
    kind = None
    if "--kind" in argv:
        kind = argv[argv.index("--kind") + 1]
        kind = {"trace": "trace", "artifact": "artifact"}.get(kind, kind)
    if not args:
        print("usage: validate_manifests.py <file> [--kind trace|artifact] | --selftest")
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
