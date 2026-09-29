#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
    echo "usage: $0 <new-absolute-output-root> <adb-port>" \
        "<phase-campaign-root> <phase-service-unit|->" \
        "<physical-shape-calibration> <initial-marginal-profile>" \
        "<direct-energy-profile> <precompiled-ffn-policy>" >&2
    exit 2
fi

output=$1
adb_port=$2
phase_root=$3
phase_unit=$4
physical_calibration=$5
initial_marginal=$6
direct_energy_profile=$7
compiled_policy=$8
if [[ $output != /* || -e $output ]]; then
    echo "output must be a new absolute path: $output" >&2
    exit 2
fi
if [[ ! $adb_port =~ ^[0-9]+$ ]]; then
    echo "invalid adb port: $adb_port" >&2
    exit 2
fi
for path in "$phase_root" "$physical_calibration" "$initial_marginal" \
        "$direct_energy_profile" "$compiled_policy"; do
    if [[ $path != /* || ! -e $path ]]; then
        echo "pipeline input must be an existing absolute path: $path" >&2
        exit 2
    fi
done

here=$(cd -- "$(dirname -- "$0")" && pwd)
factorial=$here/run_fp16_small_overlay_2x2_abba.sh
split=$here/run_ffn_split_qualification_abba.sh
for path in "$factorial" "$split"; do
    [[ -f $path ]] || {
        echo "missing pipeline dependency: $path" >&2
        exit 1
    }
done

mkdir -p "$output"
if [[ $phase_unit != - ]]; then
    while true; do
        state=$(systemctl --user show "$phase_unit" \
            --property=ActiveState --value 2>/dev/null || true)
        if [[ $state != active && $state != activating \
                && $state != deactivating ]]; then
            break
        fi
        sleep 30
    done
    result=$(systemctl --user show "$phase_unit" \
        --property=Result --value 2>/dev/null || true)
    if [[ -n $result && $result != success ]]; then
        echo "phase calibration unit did not succeed: $result" >&2
        exit 1
    fi
fi

phase_profile=$phase_root/PHASE_PROFILE_V2.json
phase_audit=$phase_root/PHASE_PROFILE_AUDIT_V2.json
python3 - "$phase_profile" "$phase_audit" <<'PY'
import hashlib
import json
import sys

profile_path, audit_path = sys.argv[1:]
profile_raw = open(profile_path, "rb").read()
profile = json.loads(profile_raw)
audit = json.load(open(audit_path, encoding="ascii"))
if not (
    profile.get("schema") == "s42-general-scheduler-profile-v1"
    and audit.get("schema") == "s42-fp16-llama1b-contention-calibration-v2"
    and audit.get("status") == "PASS"
    and audit.get("all_variants_measured") is True
    and audit.get("profile_sha256") == hashlib.sha256(profile_raw).hexdigest()
    and all(
        int(value) >= 2
        for value in audit.get("natural_validation_count_by_class", {}).values()
    )
    and len(audit.get("natural_validation_count_by_class", {})) == 8
):
    raise SystemExit("phase profile failed qualification")
PY

seed=$output/seed-2x2
heldout=$output/heldout-2x2
ffn=$output/ffn-split

bash "$factorial" "$seed" "$adb_port" \
    "$phase_profile" "$initial_marginal"
seed_marginal=$seed/NEXT_MARGINAL_SYSTEM_PROFILE.json
python3 - "$seed_marginal" <<'PY'
import json
import sys

path = sys.argv[1]
value = json.load(open(path, encoding="ascii"))
required = {"cpu-overflow", "op15-assistance"}
qualification = value.get("qualification", {})
if not (
    value.get("schema")
        == "s42-fp16-overlay-marginal-system-profile-v1"
    and qualification.get("status") == "PASS"
    and set(qualification.get("required_arms", [])) == required
    and qualification.get("arms")
        == {policy: True for policy in required}
):
    raise SystemExit("seed marginal profile failed qualification")
PY
bash "$factorial" "$heldout" "$adb_port" \
    "$phase_profile" "$seed_marginal"
heldout_marginal=$heldout/NEXT_MARGINAL_SYSTEM_PROFILE.json

python3 - "$seed" "$heldout" "$heldout_marginal" \
        "$output/HELDOUT_2X2_RECEIPT.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

seed, heldout, marginal_path, output = map(Path, sys.argv[1:])
records = {}
for label, root in (("seed", seed), ("heldout", heldout)):
    path = root / "FACTORIAL_2X2_ABBA.json"
    value = json.loads(path.read_text(encoding="ascii"))
    if value.get("status") != "PASS":
        raise SystemExit(f"{label} 2x2 result failed")
    records[label] = {
        "factorial_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "result": value,
    }
heldout_result = records["heldout"]["result"]
required = {"cpu-overflow", "op15-assistance"}
eligibility = heldout_result.get("scheduler_energy_claim_eligible", {})
if not (
    heldout_result.get("energy_qualification_status") == "PASS"
    and eligibility == {policy: True for policy in required}
):
    raise SystemExit("held-out scheduler energy confidence gate failed")
marginal = json.loads(marginal_path.read_text(encoding="ascii"))
qualification = marginal.get("qualification", {})
if not (
    marginal.get("schema")
        == "s42-fp16-overlay-marginal-system-profile-v1"
    and qualification.get("status") == "PASS"
    and set(qualification.get("required_arms", [])) == required
    and qualification.get("arms")
        == {policy: True for policy in required}
):
    raise SystemExit("held-out marginal profile failed qualification")
receipt = {
    "gates": {
        "heldout_energy_ci_positive": True,
        "heldout_marginal_profile_qualified": True,
        "seed_marginal_profile_qualified": True,
    },
    "heldout_marginal_sha256": hashlib.sha256(
        marginal_path.read_bytes()
    ).hexdigest(),
    "records": records,
    "schema": "s42-full-scheduler-heldout-2x2-receipt-v1",
    "status": "PASS",
}
output.write_bytes((json.dumps(
    receipt,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
) + "\n").encode("ascii"))
PY

direct_status=$(python3 - "$direct_energy_profile" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="ascii"))
if value.get("schema") != "s42-llama1b-ffn-direct-energy-v1":
    raise SystemExit("invalid direct energy artifact")
print(value.get("status", ""))
PY
)
if [[ $direct_status == PASS ]]; then
    bash "$split" "$ffn" "$adb_port" "$phase_profile" \
        "$physical_calibration" "$heldout_marginal" \
        "$direct_energy_profile" "$compiled_policy"
    optional_ffn_record=$ffn/FFN_SPLIT_ROUTE_CALIBRATION.json
elif [[ $direct_status == FAIL ]]; then
    mkdir -p "$ffn"
    optional_ffn_record=$ffn/FFN_SPLIT_REJECTION.json
    python3 - "$direct_energy_profile" "$optional_ffn_record" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
direct = json.loads(source.read_text(encoding="ascii"))
if not (
    direct.get("schema") == "s42-llama1b-ffn-direct-energy-v1"
    and direct.get("status") == "FAIL"
    and direct.get("qualification", {}).get("status") == "FAIL"
):
    raise SystemExit("direct energy rejection identity")
value = {
    "direct_energy_record_sha256": direct["record_sha256"],
    "direct_energy_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "reason": "DIRECT_INCREMENTAL_ENERGY_GATE_FAILED",
    "route_admission": "SHADOW_ONLY_UNTIL_GATES_PASS",
    "schema": "s42-ffn-split-route-rejection-v1",
    "status": "REJECTED",
}
output.write_bytes((json.dumps(
    value,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
) + "\n").encode("ascii"))
PY
else
    echo "invalid direct energy qualification status" >&2
    exit 1
fi

python3 - "$output" "$phase_profile" "$phase_audit" \
        "$physical_calibration" "$initial_marginal" \
        "$direct_energy_profile" "$compiled_policy" \
        "$optional_ffn_record" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
inputs = [Path(value) for value in sys.argv[2:-1]]
optional_ffn_path = Path(sys.argv[-1])
direct_path = inputs[-2]
paths = {
    "heldout_2x2": root / "heldout-2x2/FACTORIAL_2X2_ABBA.json",
    "heldout_marginal": (
        root / "heldout-2x2/NEXT_MARGINAL_SYSTEM_PROFILE.json"
    ),
    "optional_ffn": optional_ffn_path,
    "seed_2x2": root / "seed-2x2/FACTORIAL_2X2_ABBA.json",
}
for path in paths.values():
    if not path.is_file():
        raise SystemExit(f"missing pipeline result: {path}")
direct = json.loads(direct_path.read_text(encoding="ascii"))
ffn = json.loads(optional_ffn_path.read_text(encoding="ascii"))
if direct.get("status") == "PASS":
    if ffn.get("status") != "PASS":
        raise SystemExit("FFN split route is not qualified")
    optional_status = "QUALIFIED_FOR_RUNTIME_SELECTION"
elif direct.get("status") == "FAIL":
    if not (
        ffn.get("schema") == "s42-ffn-split-route-rejection-v1"
        and ffn.get("status") == "REJECTED"
        and ffn.get("direct_energy_record_sha256")
            == direct.get("record_sha256")
    ):
        raise SystemExit("FFN split rejection is not bound")
    optional_status = "REJECTED_BY_DIRECT_ENERGY_GATE"
else:
    raise SystemExit("invalid direct energy status")
receipt = {
    "input_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs
    },
    "result_sha256": {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(paths.items())
    },
    "optional_routes": {"cpu-phone-ffn-split": optional_status},
    "schema": "s42-full-scheduler-qualification-pipeline-v1",
    "status": "PASS",
}
(root / "PIPELINE_RESULT.json").write_bytes((json.dumps(
    receipt,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
) + "\n").encode("ascii"))
PY
