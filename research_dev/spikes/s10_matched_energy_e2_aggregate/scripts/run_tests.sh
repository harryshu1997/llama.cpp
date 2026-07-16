#!/usr/bin/env bash
# The E2A suite. Verifies that every listed E1/E2 baseline file is byte-identical
# BEFORE and AFTER, so a change to reviewed dependencies is a visible failure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/../../.." && pwd)"

E1_MANIFEST="$ROOT/../s10_matched_energy_e2/E1_BASELINE_MANIFEST.txt"
E2_MANIFEST="$ROOT/E2_BASELINE_MANIFEST.txt"

check_baseline() {
    local name="$1" manifest="$2" base="$3" when="$4"
    ( cd "$base" && sha256sum -c "$manifest" >/dev/null 2>&1 ) || {
        echo "$name BASELINE CHANGED ($when): a pinned file differs" >&2
        ( cd "$base" && sha256sum -c "$manifest" 2>&1 | grep -v ': OK' | head -5 ) >&2
        exit 1; }
    echo "$name pinned baseline files unchanged ($when): "\
"$(wc -l < "$manifest") files"
}

# Bytecode is build output. Remove it before any manifest is taken so a .pyc can
# never appear in a frozen tree or in an ASCII check.
cleanup() {
    find "$ROOT" -name '__pycache__' -type d -exec rm -rf {} + \
        2>/dev/null || true
    find "$ROOT" -name '*.pyc' -delete 2>/dev/null || true
}
trap cleanup EXIT
cleanup

check_baseline "E1" "$E1_MANIFEST" "$REPO" "before"
check_baseline "E2" "$E2_MANIFEST" "$ROOT/../s10_matched_energy_e2" "before"

cd "$ROOT"
python3 tests/make_e2a_fixtures.py
python3 -m unittest discover -s tests -p 'test_*.py' 2>&1 | tail -4

# The headline, printed rather than buried: the production path on real inputs.
python3 - <<'PY'
import sys
sys.path.insert(0, "src")
import aggregate, resolver
try:
    aggregate.evaluate("fixtures/bundle.json")
    raise SystemExit("E2A_FAIL: a bundle reached a label with no enumerable anchor")
except (aggregate.AggregateError, resolver.ResolveError) as exc:
    print(f"valid 8-pair bundle: structural prechecks reached anchor refusal -> "
          f"{exc.code}")
import anchors
capability = anchors.describe_host_capability()
print(f"host anchor capability: best={capability['best_property']} "
      f"required={capability['required_property']} "
      f"sufficient={capability['sufficient']}")
PY

bash scripts/e2a_negative.sh | tail -1

# Determinism: the fixture bundle must be reproducible byte-for-byte.
python3 - <<'PY'
import hashlib
import pathlib
import subprocess
import sys

digests = []
for _ in range(3):
    subprocess.run([sys.executable, "tests/make_e2a_fixtures.py"],
                   check=True, capture_output=True)
    hasher = hashlib.sha256()
    for path in sorted(pathlib.Path("fixtures").rglob("*")):
        if path.is_file():
            relative = path.relative_to("fixtures").as_posix().encode("ascii")
            data = path.read_bytes()
            hasher.update(len(relative).to_bytes(8, "big"))
            hasher.update(relative)
            hasher.update(len(data).to_bytes(8, "big"))
            hasher.update(data)
    digests.append(hasher.hexdigest())
if len(set(digests)) != 1:
    raise SystemExit(f"E2A_FAIL: fixtures are not deterministic: {digests}")
print(f"E2A fixture determinism across 3 regenerations: {digests[0]}")
PY

cleanup

check_baseline "E1" "$E1_MANIFEST" "$REPO" "after"
check_baseline "E2" "$E2_MANIFEST" "$ROOT/../s10_matched_energy_e2" "after"

echo "S10_E2A_AGGREGATE_TESTS_PASS"
