#!/usr/bin/env python3
"""S9 schema test runner (V0).

Runs BOTH validators against every fixture and asserts each expected-valid fixture
passes and each expected-invalid fixture fails on BOTH. Returns nonzero on ANY
unexpected result, precheck failure, version mismatch, or schema that fails to
compile/load. Adapted from the S8 runner.

PINNED validators (verified at startup):
  - /usr/bin/jsonschema 4.10.3 (exact, hard-checked)
  - npx --yes ajv-cli@5.0.0 validate --spec=draft2020 (pinned spec + functional smoke)

Usage:
  python3 run_schema_tests.py               # full suite
  python3 run_schema_tests.py --index PATH  # alternate index (proves a missing fixture fails)
"""
import json, os, subprocess, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMAS_DIR = os.path.join(HERE, "schemas")
FIXROOT = os.path.join(HERE, "fixtures")
INDEX = os.path.join(FIXROOT, "index.json")

JSONSCHEMA = ["/usr/bin/jsonschema"]
JSONSCHEMA_PIN = "4.10.3"
AJV = ["npx", "--yes", "ajv-cli@5.0.0"]
AJV_PIN = "ajv-cli@5.0.0"


def _run(cmd):
    return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)


def jsonschema_valid(schema, inst):
    r = _run(JSONSCHEMA + ["-i", inst, schema])
    if r.returncode == 0:
        return True, False
    if "Traceback (most recent call last)" in r.stderr:
        return False, True
    return False, False


def ajv_valid(schema, inst):
    r = _run(AJV + ["validate", "--spec=draft2020", "--strict-types=false", "-s", schema, "-d", inst])
    if r.returncode == 0:
        return True, False
    if "invalid" in (r.stdout + r.stderr):
        return False, False
    return False, True


def ajv_compiles(schema):
    r = _run(AJV + ["compile", "--spec=draft2020", "--strict-types=false", "-s", schema])
    return r.returncode == 0, (r.stdout + r.stderr)


def jsonschema_loads(schema):
    import tempfile
    trivial = os.path.join(tempfile.gettempdir(), "s9_trivial_instance.json")
    with open(trivial, "w") as f:
        f.write("{}\n")
    r = _run(JSONSCHEMA + ["-i", trivial, schema])
    return "Traceback (most recent call last)" not in r.stderr, r.stderr


def verify_validators():
    errs = []
    r = _run(JSONSCHEMA + ["--version"])
    ver = (r.stdout + r.stderr).strip()
    if ver != JSONSCHEMA_PIN:
        errs.append(f"jsonschema version {ver!r} != pinned {JSONSCHEMA_PIN!r}")
    else:
        print(f"  jsonschema {ver} (pinned OK)")
    import tempfile
    td = tempfile.gettempdir()
    open(os.path.join(td, "s9_true.schema.json"), "w").write("true")
    open(os.path.join(td, "s9_false.schema.json"), "w").write("false")
    open(os.path.join(td, "s9_inst.json"), "w").write("{}\n")
    rv = _run(AJV + ["validate", "--spec=draft2020", "-s", os.path.join(td, "s9_true.schema.json"), "-d", os.path.join(td, "s9_inst.json")])
    ri = _run(AJV + ["validate", "--spec=draft2020", "-s", os.path.join(td, "s9_false.schema.json"), "-d", os.path.join(td, "s9_inst.json")])
    if rv.returncode != 0 or ri.returncode == 0:
        errs.append(f"ajv-cli functional smoke failed (true->{rv.returncode}, false->{ri.returncode})")
    else:
        print(f"  {AJV_PIN} (pinned spec; functional smoke OK)")
    return errs


def precheck_index(index, index_path, fixroot):
    errs = []
    seen = set()
    for i, rec in enumerate(index):
        for key in ("file", "schema", "expect"):
            if key not in rec:
                errs.append(f"index[{i}] missing key {key!r}")
        if "file" not in rec or "schema" not in rec:
            continue
        inst = os.path.join(fixroot, rec["file"])
        schema = os.path.join(HERE, rec["schema"])
        seen.add(os.path.normpath(inst))
        if not os.path.exists(inst):
            errs.append(f"fixture missing: {rec['file']}")
            continue
        if not os.path.isfile(inst):
            errs.append(f"fixture not a regular file: {rec['file']}")
        else:
            try:
                json.load(open(inst))
            except Exception as ex:
                errs.append(f"fixture not valid JSON: {rec['file']}: {ex}")
        if not (os.path.exists(schema) and os.path.isfile(schema)):
            errs.append(f"schema missing/not-a-file: {rec['schema']}")
        if rec.get("expect") not in ("valid", "invalid"):
            errs.append(f"index[{i}] bad expect: {rec.get('expect')!r}")
    for sub in ("valid", "invalid"):
        d = os.path.join(fixroot, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".json"):
                continue
            p = os.path.normpath(os.path.join(d, fn))
            if p not in seen:
                errs.append(f"fixture present but NOT referenced by {os.path.basename(index_path)}: {sub}/{fn}")
    return errs


def run_suite(name, index_path, fixroot):
    try:
        index = json.load(open(index_path))
    except Exception as ex:
        print(f"FAIL: cannot load index {index_path}: {ex}")
        return None
    print(f"== [{name}] fixture prechecks ==")
    perrs = precheck_index(index, index_path, fixroot)
    for e in perrs:
        print("  ERR", e)
    if perrs:
        print(f"FAIL: {len(perrs)} precheck error(s) in {name}")
        return None
    print(f"  OK: {len(index)} index entries, all present, regular files, valid JSON, index complete")

    print(f"== [{name}] fixture validation (jsonschema + ajv) ==")
    fails = n_valid = n_invalid = 0
    for rec in index:
        inst = os.path.join(fixroot, rec["file"])
        schema = os.path.join(HERE, rec["schema"])
        expect_valid = rec["expect"] == "valid"
        jv, jerr = jsonschema_valid(schema, inst)
        av, aerr = ajv_valid(schema, inst)
        if expect_valid:
            n_valid += 1
        else:
            n_invalid += 1
        ok = (not jerr) and (not aerr) and (jv == expect_valid) and (av == expect_valid)
        if not ok:
            fails += 1
        detail = f"js={'valid' if jv else 'invalid'}{'(ERR)' if jerr else ''} ajv={'valid' if av else 'invalid'}{'(ERR)' if aerr else ''}"
        print(f"  {'PASS' if ok else 'FAIL'} [{rec['expect']:7}] {rec['file']:52} {detail}")
    print(f"  [{name}] fixtures: {len(index)}  valid: {n_valid}  invalid: {n_invalid}  failures: {fails}")
    return {"n": len(index), "valid": n_valid, "invalid": n_invalid, "fails": fails}


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    v1_index = INDEX
    if "--index" in argv:
        v1_index = argv[argv.index("--index") + 1]

    print("== validator pin/verify ==")
    verrs = verify_validators()
    for e in verrs:
        print("  ERR", e)
    if verrs:
        print("FAIL: validator pin/verify failed")
        return 1

    all_schemas = sorted(glob.glob(os.path.join(SCHEMAS_DIR, "*.schema.json")) +
                         glob.glob(os.path.join(SCHEMAS_DIR, "v2", "*.schema.json")) +
                         glob.glob(os.path.join(SCHEMAS_DIR, "v3", "*.schema.json")) +
                         glob.glob(os.path.join(SCHEMAS_DIR, "v4", "*.schema.json")) +
                         glob.glob(os.path.join(SCHEMAS_DIR, "v5", "*.schema.json")))
    print("== schema compile/load check (ajv compile + jsonschema load), v1 + v2 + v3 + v4 + v5 bundles ==")
    compile_fail = 0
    for s in all_schemas:
        a_ok, a_out = ajv_compiles(s)
        j_ok, j_out = jsonschema_loads(s)
        rel = os.path.relpath(s, HERE)
        print(f"  {'OK ' if (a_ok and j_ok) else 'ERR'} {rel}  ajv={'ok' if a_ok else 'ERR'} jsonschema={'ok' if j_ok else 'ERR'}")
        if not (a_ok and j_ok):
            compile_fail += 1
            sys.stderr.write(a_out + j_out + "\n")
    if compile_fail:
        print(f"FAIL: {compile_fail} schema(s) did not compile/load")
        return 1

    suites = [("v1-frozen", v1_index, FIXROOT),
              ("v2-repaired", os.path.join(FIXROOT, "v2", "index.json"), os.path.join(FIXROOT, "v2")),
              ("v3-r1", os.path.join(FIXROOT, "v3", "index.json"), os.path.join(FIXROOT, "v3")),
              ("v4-r2", os.path.join(FIXROOT, "v4", "index.json"), os.path.join(FIXROOT, "v4")),
              ("v5-r3", os.path.join(FIXROOT, "v5", "index.json"), os.path.join(FIXROOT, "v5"))]
    total_fail = 0
    total_n = 0
    for name, idx, root in suites:
        res = run_suite(name, idx, root)
        if res is None:
            return 1
        total_fail += res["fails"]
        total_n += res["n"]

    print(f"\nTOTAL schema fixtures: {total_n}  failures: {total_fail}")
    if total_fail:
        print("RESULT: FAIL")
        return 1
    print("RESULT: ALL PASS (both validators agree with every fixture expectation, v1 + v2 + v3 + v4 + v5)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
