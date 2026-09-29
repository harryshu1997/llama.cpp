#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 <new-output-root> <prematerialized-catalog> <overlay-catalog> <observation-store> <adaptive-observation-store>" >&2
    exit 2
fi

output=$1
input_catalog=$2
input_overlay=$3
input_observations=$4
input_adaptive_observations=$5

if [[ $output != /* || -e $output ]]; then
    echo "output must be a new absolute path" >&2
    exit 2
fi
for path in "$input_catalog" "$input_overlay" "$input_observations" \
        "$input_adaptive_observations"; do
    if [[ ! -f $path ]]; then
        echo "matched campaign input is absent: $path" >&2
        exit 2
    fi
done
if pgrep -f '[r]esearch_dev/scheduler/campaigns/burstgpt/runner.py' \
        >/dev/null; then
    echo "another unified BurstGPT campaign is active" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
repo_root=${S42_UNIFIED_REPO_ROOT:-$(cd -- "$here/../../../.." && pwd)}
launcher=$here/run_physical_campaign.sh
comparator=$here/compare_ab.py
preflight=$here/preflight.py
data_dir=$here/data
burst_dir=$repo_root/research_dev/spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1
overlay_dir=$repo_root/research_dev/spikes/s42_general_energy_scheduler_v1/full_fp16_burstgpt_v1/small_model_overlay_v1

session_discovery=${S42_PHONE_SESSION_DISCOVERY:-}
if [[ -z $session_discovery || ! -f $session_discovery ]]; then
    echo "S42_PHONE_SESSION_DISCOVERY must name a physical discovery file" >&2
    exit 2
fi

server_root=${S42_SERVER_ROOT:-/home/zhihao/s42-direct-usb-v2/build-server-cuda/bin}
server=${S42_SERVER:-$server_root/llama-server}
resident_root=${S42_RESIDENT_ROOT:-/home/zhihao/s42-unified-scheduler-v1/build-s41-server-ffn/bin}
resident_server=${S42_RESIDENT_SERVER:-$resident_root/llama-server}
cuda_lib_dir=${S42_CUDA_LIB_DIR:-/mnt/storage/s21_deps/cuda-13.2.1/lib}
resident_lib_dir=${S42_RESIDENT_LIB_DIR:-$resident_root}
bridge=${S42_LEGACY_BRIDGE:-/home/zhihao/s41-dynamic-ffn-v1/ffn_dmabuf_bridge-unified-v9-repair-r3}
close_helper=${S42_LEGACY_CLOSE_HELPER:-$repo_root/research_dev/scheduler/adapters/close_resident_bridge.py}
qwen_model=${S42_QWEN_MODEL:-/home/zhihao/models/Qwen3-14B-Q4KM-dequant-f16.gguf}
gemma_model=${S42_GEMMA_MODEL:-/home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf}
llama_model=${S42_LLAMA_MODEL:-/home/zhihao/models/Llama-3.2-1B-Instruct-Q4_0.gguf}
phone_serial=${S42_PHONE_SERIAL:-3C15AU002CL00000}
adb_port=${S42_ADB_PORT:-5037}
phone_battery_ppm=${S42_PHONE_BATTERY_PPM:-250000}
phone_diagnostic_endpoint=${S42_PHONE_DIAGNOSTIC_ENDPOINT:-http://192.168.42.1:18383}

for path in "$launcher" "$comparator" "$preflight" "$server" \
        "$resident_server" "$bridge" "$close_helper" "$qwen_model" \
        "$gemma_model" "$llama_model"; do
    if [[ ! -e $path ]]; then
        echo "matched campaign dependency is absent: $path" >&2
        exit 2
    fi
done

umask 077
mkdir -p -- "$output/inputs"
catalog=$output/inputs/CAPABILITY_CATALOG.json
overlay=$output/inputs/OVERLAY_CATALOG.json
observations=$output/inputs/AUTOMATED_OBSERVATIONS.json
adaptive_observations=$output/inputs/ADAPTIVE_DECODE_OBSERVATIONS.json
discovery=$output/inputs/PHONE_SESSION_DISCOVERY.json
cp -- "$input_catalog" "$catalog"
cp -- "$input_overlay" "$overlay"
cp -- "$input_observations" "$observations"
cp -- "$input_adaptive_observations" "$adaptive_observations"
cp -- "$session_discovery" "$discovery"

source_manifest=$output/inputs/SOURCE_MANIFEST.json
python3 - "$repo_root" "$source_manifest" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
rows = []
for path in sorted((root / "research_dev" / "scheduler").rglob("*")):
    if path.is_file() and path.suffix in {".py", ".sh", ".zsh"}:
        rows.append({
            "path": str(path.relative_to(root)),
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        })
def git(*args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
body = {
    "branch": git("branch", "--show-current"),
    "files": rows,
    "head": git("rev-parse", "HEAD"),
    "schema": "research-scheduler-source-manifest-v1",
}
encoded = json.dumps(
    body, ensure_ascii=True, separators=(",", ":"), sort_keys=True,
).encode("ascii")
body["manifest_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
output.write_text(
    json.dumps(body, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    + "\n",
    encoding="ascii",
)
PY

printf '%q ' "$0" "$@" > "$output/RUN_MATCHED_AB_COMMAND.txt"
printf '\n' >> "$output/RUN_MATCHED_AB_COMMAND.txt"
sha256sum "$catalog" "$overlay" "$observations" \
    "$adaptive_observations" "$discovery" "$source_manifest" \
    > "$output/INPUT_SHA256SUMS.txt"

normal_usb=$output/PHONE_USB_BEFORE.json
PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
python3 - "$phone_serial" "$adb_port" "$normal_usb" <<'PY'
import json
from pathlib import Path
import sys
from research_dev.scheduler.adapters import verify_android_usb_restored

receipt = verify_android_usb_restored(
    serial=sys.argv[1],
    adb_port=int(sys.argv[2]),
    minimum_speed_mbps=5000,
    timeout_s=60,
)
Path(sys.argv[3]).write_text(
    json.dumps(receipt.to_json(), ensure_ascii=True, sort_keys=True) + "\n",
    encoding="ascii",
)
PY

preflight_output=$output/PHYSICAL_PREFLIGHT.json
python3 "$preflight" \
    --large-requests "$burst_dir/REQUESTS_SEMANTIC_SOURCE.jsonl" \
    --overlay-requests "$overlay_dir/REQUESTS_LLAMA1B_10.jsonl" \
    --trace-manifest "$overlay_dir/TRACE_MANIFEST.json" \
    --capability-catalog "$catalog" \
    --observation-store-input "$observations" \
    --adaptive-observation-store-input "$adaptive_observations" \
    --qwen-manifest "$data_dir/QWEN_MANIFEST.json" \
    --gemma-manifest "$data_dir/GEMMA_MANIFEST.json" \
    --qwen-model "$qwen_model" \
    --gemma-model "$gemma_model" \
    --llama-model "$llama_model" \
    --server "$server" \
    --resident-server "$resident_server" \
    --cuda-lib-dir "$cuda_lib_dir" \
    --resident-lib-dir "$resident_lib_dir" \
    --bridge "$bridge" \
    --phone-session-discovery "$discovery" \
    --close-helper "$close_helper" \
    --phone-diagnostic-endpoint "$phone_diagnostic_endpoint" \
    --phone-battery-ppm "$phone_battery_ppm" \
    --phone-usb-serial "$phone_serial" \
    --phone-normal-usb-receipt "$normal_usb" \
    --adb-port "$adb_port" \
    --minimum-usb-speed-mbps 5000 \
    --output "$preflight_output" \
    > "$output/preflight.log"

export S42_PHONE_SESSION_DISCOVERY=$discovery
export S42_REQUIRED_PHONE_SESSIONS=3
export S42_MAXIMUM_LATENCY_PPM=1250000
export S42_PREMATERIALIZED_CATALOG=$catalog
export S42_PREMATERIALIZED_SOURCE_MANIFEST=$source_manifest
# A single A/B trace is diagnostic for route learning. Trace-level fleet
# energy remains authoritative for the comparison below.
export S42_ENERGY_ATTRIBUTION_KIND=diagnostic

run_arm() {
    local arm_output=$1
    local mode=$2
    local indices=${3:-}
    "$launcher" "$arm_output" "$mode" "$overlay" "$indices" \
        "$observations" "$adaptive_observations"
}

run_arm "$output/gate-a-desktop" desktop-baseline 0,1,2,43,67
run_arm "$output/gate-b-energy-aware" energy-aware 0,1,2,43,67

python3 - \
    "$output/gate-a-desktop/run/RESULT.json" \
    "$output/gate-b-energy-aware/run/RESULT.json" \
    "$output/GATE_COMPARISON.json" <<'PY'
import json
from pathlib import Path
import sys

def fail(message):
    raise SystemExit("physical gate blocked: " + message)

def load(path):
    return json.loads(Path(path).read_text(encoding="ascii"))

def uses_phone(row):
    plan = row["terminal_ticket"]["execution_plan"]
    contract = plan.get("execution_contract") or {}
    proof = row.get("physical_execution_proof") or {}
    return bool(contract.get("phone_shards") or proof.get("phone_call_count", 0))

def uses_gpu(row):
    text = json.dumps(
        {
            "binding": row["terminal_ticket"].get("binding"),
            "plan": row["terminal_ticket"].get("execution_plan"),
        },
        sort_keys=True,
    ).lower()
    return "desktop-cuda" in text or "cuda0" in text

def recursive_counter(value, key):
    if isinstance(value, dict):
        return int(value.get(key, 0) or 0) + sum(
            recursive_counter(row, key) for row in value.values()
        )
    if isinstance(value, list):
        return sum(recursive_counter(row, key) for row in value)
    return 0

a_path, b_path, output = map(Path, sys.argv[1:])
a, b = load(a_path), load(b_path)
for name, value, mode in (("A", a, "desktop-baseline"), ("B", b, "energy-aware")):
    counts = value.get("counts") or {}
    if value.get("status") != "PASS" or counts.get("requests") != 5 \
            or counts.get("terminals") != 5:
        fail(name + " is incomplete")
    if value.get("selection_mode") != mode:
        fail(name + " selection mode differs")
    if any(not (row.get("output_quality") or {}).get("accepted", False)
           for row in value["request_results"]):
        fail(name + " output quality failed")
if a.get("catalog_sha256") != b.get("catalog_sha256") \
        or a.get("trace_identity") != b.get("trace_identity") \
        or a.get("model_artifacts") != b.get("model_artifacts") \
        or a.get("execution_identity") != b.get("execution_identity"):
    fail("A/B physical identity differs")
if any(uses_phone(row) for row in a["request_results"]):
    fail("desktop gate executed phone work")
roles = a.get("model_roles") or {}
large_models = {roles.get("qwen"), roles.get("gemma")}
if any(
    row.get("model_id") in large_models and not uses_gpu(row)
    for row in a["request_results"]
):
    fail("desktop large-model gate did not use CUDA")
session_calls = {}
for row in b["request_results"]:
    proof = row.get("physical_execution_proof") or {}
    for session in proof.get("phone_calls_by_session") or []:
        key = str(session.get("session_id"))
        session_calls[key] = session_calls.get(key, 0) + int(session.get("calls", 0))
if not all(session_calls.get(name, 0) > 0 for name in ("HTP0", "HTP1", "HTP2")):
    fail("energy-aware gate did not physically use all three sessions")
if any(row.get("recoveries") for row in b["request_results"]):
    fail("energy-aware gate used fallback")
if recursive_counter(b.get("direct_phone_receipts") or [], "reset_recoveries"):
    fail("energy-aware gate had a USB reset")
encoded = {
    "baseline_duration_us": a["duration_us"],
    "baseline_fleet_energy_uj": sum(a["trace_energy"]["fleet_energy_uj_by_domain"].values()),
    "catalog_sha256": a["catalog_sha256"],
    "energy_aware_duration_us": b["duration_us"],
    "energy_aware_fleet_energy_uj": sum(b["trace_energy"]["fleet_energy_uj_by_domain"].values()),
    "session_calls": dict(sorted(session_calls.items())),
    "status": "PASS",
}
Path(output).write_text(
    json.dumps(encoded, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    + "\n",
    encoding="ascii",
)
PY

run_arm "$output/a-desktop" desktop-baseline
python3 - "$output/a-desktop/run/RESULT.json" <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
counts = value.get("counts") or {}
if value.get("status") != "PASS" \
        or value.get("selection_mode") != "desktop-baseline" \
        or counts.get("requests") != 84 \
        or counts.get("terminals") != 84 \
        or {name: counts.get(name) for name in ("qwen", "gemma", "llama")} \
            != {"qwen": 57, "gemma": 17, "llama": 10}:
    raise SystemExit("desktop arm did not complete 84 requests")
roles = value.get("model_roles") or {}
large_models = {roles.get("qwen"), roles.get("gemma")}
for row in value["request_results"]:
    ticket = row["terminal_ticket"]
    plan = ticket["execution_plan"]
    contract = plan.get("execution_contract") or {}
    if contract.get("phone_shards") \
            or (row.get("physical_execution_proof") or {}).get("phone_call_count", 0):
        raise SystemExit("desktop arm physically executed phone work")
    if row.get("model_id") in large_models:
        text = json.dumps({
            "binding": ticket.get("binding"),
            "plan": plan,
        }, sort_keys=True).lower()
        if "desktop-cuda" not in text and "cuda0" not in text:
            raise SystemExit(
                "desktop large-model arm did not physically use CUDA"
            )
PY

run_arm "$output/b-energy-aware" energy-aware
python3 "$comparator" \
    "$output/a-desktop/run/RESULT.json" \
    "$output/b-energy-aware/run/RESULT.json" \
    --output "$output/COMPARISON.json" \
    > "$output/comparison.stdout.json"

sha256sum \
    "$output/PHYSICAL_PREFLIGHT.json" \
    "$output/GATE_COMPARISON.json" \
    "$output/a-desktop/run/RESULT.json" \
    "$output/b-energy-aware/run/RESULT.json" \
    "$output/a-desktop/run/SCHEDULER_DECISION_LOG.json" \
    "$output/b-energy-aware/run/SCHEDULER_DECISION_LOG.json" \
    "$output/COMPARISON.json" \
    > "$output/RESULT_SHA256SUMS.txt"
