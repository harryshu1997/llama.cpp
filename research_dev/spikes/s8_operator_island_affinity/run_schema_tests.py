#!/usr/bin/env python3
"""S8 schema test runner (V0b-P0).

Repeatable. Runs BOTH validators against every fixture and asserts each
expected-valid fixture passes and each expected-invalid fixture fails on BOTH.
Returns nonzero on ANY unexpected result, prechecks failure, version mismatch, or
schema that fails to compile/load.

PINNED validators (verified at startup):
  - /usr/bin/jsonschema 4.10.3 (exact, hard-checked)
  - npx --yes ajv-cli@5.0.0 validate --spec=draft2020 (pinned spec + functional smoke)

Usage:
  python3 run_schema_tests.py                 # full suite
  python3 run_schema_tests.py --index PATH     # use an alternate fixture index
                                               # (used to prove a missing fixture fails)
ajv-cli is fetched by npx on first use; no dataset network access.
"""
import json, os, subprocess, sys

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
    """Return (is_valid, errored). errored=True means a tool/schema error, not an
    instance verdict. Schemas are pre-compiled (ajv) before this runs, so a nonzero
    exit here is an instance failure unless the CLI raised a Python traceback."""
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
    blob = (r.stdout + r.stderr)
    # instance failure prints "<data> invalid"; a genuine tool/schema error prints "error:"
    if "invalid" in blob:
        return False, False
    return False, True


def ajv_compiles(schema):
    r = _run(AJV + ["compile", "--spec=draft2020", "--strict-types=false", "-s", schema])
    return r.returncode == 0, (r.stdout + r.stderr)


def jsonschema_loads(schema):
    """True if the schema LOADS under the jsonschema CLI from its path (a validation
    failure of the trivial instance is fine; only a Python traceback = load error)."""
    import tempfile
    trivial = os.path.join(tempfile.gettempdir(), "s8_trivial_instance.json")
    with open(trivial, "w") as f:
        f.write("{}\n")
    r = _run(JSONSCHEMA + ["-i", trivial, schema])
    return "Traceback (most recent call last)" not in r.stderr, r.stderr


def verify_validators():
    """Pin + verify both validators. Hard-fail on any mismatch."""
    errs = []
    r = _run(JSONSCHEMA + ["--version"])
    ver = (r.stdout + r.stderr).strip()
    if ver != JSONSCHEMA_PIN:
        errs.append(f"jsonschema version {ver!r} != pinned {JSONSCHEMA_PIN!r}")
    else:
        print(f"  jsonschema {ver} (pinned OK)")
    # ajv: pinned by the @5.0.0 spec; functional smoke against boolean schemas
    import tempfile
    td = tempfile.gettempdir()
    strue = os.path.join(td, "s8_true.schema.json"); open(strue, "w").write("true")
    sfalse = os.path.join(td, "s8_false.schema.json"); open(sfalse, "w").write("false")
    inst = os.path.join(td, "s8_inst.json"); open(inst, "w").write("{}\n")
    rv = _run(AJV + ["validate", "--spec=draft2020", "-s", strue, "-d", inst])
    ri = _run(AJV + ["validate", "--spec=draft2020", "-s", sfalse, "-d", inst])
    if rv.returncode != 0 or ri.returncode == 0:
        errs.append(f"ajv-cli functional smoke failed (true->{rv.returncode}, false->{ri.returncode})")
    else:
        print(f"  {AJV_PIN} (pinned spec; functional smoke OK)")
    return errs


def precheck_index(index, index_path):
    """Existence, regular-file, JSON-parse, and index-completeness checks. Returns
    a list of error strings; any nonempty list fails the runner."""
    errs = []
    seen = set()
    for i, rec in enumerate(index):
        for key in ("file", "schema", "expect"):
            if key not in rec:
                errs.append(f"index[{i}] missing key {key!r}")
        if "file" not in rec or "schema" not in rec:
            continue
        inst = os.path.join(FIXROOT, rec["file"])
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
    # completeness: every *.json under fixtures/{valid,invalid} must be in the index
    for sub in ("valid", "invalid"):
        d = os.path.join(FIXROOT, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".json"):
                continue
            p = os.path.normpath(os.path.join(d, fn))
            if p not in seen:
                errs.append(f"fixture present but NOT referenced by {os.path.basename(index_path)}: {sub}/{fn}")
    return errs


def main(argv=None):
    import glob
    argv = argv if argv is not None else sys.argv[1:]
    index_path = INDEX
    if "--index" in argv:
        index_path = argv[argv.index("--index") + 1]

    print("== validator pin/verify ==")
    verrs = verify_validators()
    for e in verrs:
        print("  ERR", e)
    if verrs:
        print("FAIL: validator pin/verify failed")
        return 1

    try:
        index = json.load(open(index_path))
    except Exception as ex:
        print(f"FAIL: cannot load index {index_path}: {ex}")
        return 1

    print("== fixture prechecks (existence / regular-file / JSON parse / index completeness) ==")
    perrs = precheck_index(index, index_path)
    for e in perrs:
        print("  ERR", e)
    if perrs:
        print(f"FAIL: {len(perrs)} precheck error(s)")
        return 1
    print(f"  OK: {len(index)} index entries, all present, regular files, valid JSON, index complete")

    # Compile/load EVERY schema file from its own filesystem path under BOTH
    # validators (not only those referenced by fixtures; no preloaded refs).
    all_schemas = sorted(glob.glob(os.path.join(SCHEMAS_DIR, "*.schema.json")))
    print("== schema compile/load check (ajv compile + jsonschema load) ==")
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

    print("== fixture validation (jsonschema + ajv) ==")
    fails = 0
    n_valid = n_invalid = 0
    for rec in index:
        inst = os.path.join(FIXROOT, rec["file"])
        schema = os.path.join(HERE, rec["schema"])
        expect_valid = rec["expect"] == "valid"
        jv, jerr = jsonschema_valid(schema, inst)
        av, aerr = ajv_valid(schema, inst)
        if expect_valid:
            n_valid += 1
        else:
            n_invalid += 1
        ok = (not jerr) and (not aerr) and (jv == expect_valid) and (av == expect_valid)
        status = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1
        detail = f"js={'valid' if jv else 'invalid'}{'(ERR)' if jerr else ''} ajv={'valid' if av else 'invalid'}{'(ERR)' if aerr else ''}"
        print(f"  {status} [{rec['expect']:7}] {rec['file']:52} {detail}")

    print(f"\nfixtures: {len(index)}  expected-valid: {n_valid}  expected-invalid: {n_invalid}  failures: {fails}")
    if fails:
        print("RESULT: FAIL")
        return 1
    print("RESULT: ALL PASS (both validators agree with every fixture expectation)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
