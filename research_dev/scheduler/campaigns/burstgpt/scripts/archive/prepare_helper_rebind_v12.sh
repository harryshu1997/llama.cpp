#!/usr/bin/env bash
set -euo pipefail

root=${HELPER_ROOT:-/home/zhihao/s42-helper-rebind-ggg-ggq-gate-20260901-v20}
previous=${HELPER_PREVIOUS:-/home/zhihao/s42-helper-rebind-ggg-ggq-gate-20260901-v19}
repo=${HELPER_REPO:-/home/zhihao/s42-helper-rebind-deploy-20260901-v12/llama.cpp-release}
manager=${HELPER_MANAGER:-/home/zhihao/s42-helper-rebind-deploy-20260901-v12/android-bin/llama-ffn-split-resident-workers}
phone_bin_root=${HELPER_PHONE_BIN_ROOT:-/data/local/tmp/s42-helper-rebind-ggg-ggq-v14-bin}
phone_manager=$phone_bin_root/llama-ffn-split-resident-workers
phone_session=$phone_bin_root/direct_phone_ffn_session.sh
phone_session_root=${HELPER_PHONE_SESSION_ROOT:-${phone_bin_root%-bin}}
command_old_repo=${HELPER_COMMAND_OLD_REPO:-/home/zhihao/s42-helper-rebind-deploy-20260901-v10/llama.cpp-release}
command_old_manager=${HELPER_COMMAND_OLD_MANAGER:-/data/local/tmp/s42-helper-rebind-ggg-ggq-v13-bin/llama-ffn-split-resident-workers}
command_old_phone_bin_root=${HELPER_COMMAND_OLD_PHONE_BIN_ROOT:-/data/local/tmp/s42-helper-rebind-ggg-ggq-v13-bin}
command_old_phone_session_root=${HELPER_COMMAND_OLD_PHONE_SESSION_ROOT:-${command_old_phone_bin_root%-bin}}
server_root=${HELPER_SERVER_ROOT:-/home/zhihao/s42-phone-coverage-scoped-deploy-20260831-v1/build-fresh/bin}
command_old_server_root=${HELPER_COMMAND_OLD_SERVER_ROOT:-/home/zhihao/s42-phone-coverage-scoped-deploy-20260831-v1/build-fresh/bin}
derivation_reason=${HELPER_DERIVATION_REASON:-scheduler-only-v20-helper-lease-renewal-through-zero-ack}
replay_spec=${HELPER_REPLAY_SPEC:-49:1000000,43:155000000}
replay_name=${HELPER_REPLAY_NAME:-helper_rebind_long_gemma_then_qwen_v1}

test ! -e "$root"
test -d "$previous/inputs"
test -f "$previous/RUN_COMMAND.txt"
test -d "$repo/research_dev/scheduler"
test -x "$manager"
mkdir -p "$root"
cp -a "$previous/inputs" "$root/inputs"

python3 - \
    "$root/inputs/REPLAY_SCHEDULE.json" \
    "$replay_spec" \
    "$replay_name" <<'PY'
import json
from pathlib import Path
import sys

arrivals = []
for item in sys.argv[2].split(","):
    index, separator, arrival = item.partition(":")
    if not separator or not index.isdigit() or not arrival.isdigit():
        raise SystemExit("helper replay specification is invalid")
    arrivals.append({
        "combined_request_index": int(index),
        "replay_arrival_us": int(arrival),
    })
if (
    not arrivals
    or any(
        current["replay_arrival_us"] < previous["replay_arrival_us"]
        for previous, current in zip(arrivals, arrivals[1:])
    )
):
    raise SystemExit("helper replay arrivals are invalid")
value = {
    "arrivals": arrivals,
    "schema": "research-scheduler-burstgpt-replay-v1",
    "trace_name": sys.argv[3],
}
Path(sys.argv[1]).write_text(
    json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    + "\n",
    encoding="ascii",
)
PY

python3 - \
    "$repo" \
    "$root/inputs/TRANSPORT_QUALIFICATION_IDENTITY.json" \
    "$root/inputs/UNIFIED_RUNTIME_CATALOG.json" \
    "$manager" \
    "$repo/research_dev/scheduler/adapters/native/direct_phone_ffn_session.sh" \
    "$server_root/llama-server" \
    "$server_root/libllama-server-impl.so" \
    "$repo/examples/layersplit/ffn-split-client.cpp" \
    "$root/TRANSPORT_IDENTITY_DERIVATION.json" \
    "$derivation_reason" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

source_root = Path(sys.argv[1]).resolve()
identity_path = Path(sys.argv[2])
catalog_path = Path(sys.argv[3])
manager_path = Path(sys.argv[4])
session_path = Path(sys.argv[5])
host_binary_path = Path(sys.argv[6])
server_impl_path = Path(sys.argv[7])
transport_client_source_path = Path(sys.argv[8])
derivation_path = Path(sys.argv[9])
derivation_reason = sys.argv[10]
sys.path.insert(0, str(source_root))

from research_dev.scheduler.adapters.transport_profiles import (
    TransportQualificationIdentity,
)

identity_value = json.loads(identity_path.read_text(encoding="ascii"))
parent = TransportQualificationIdentity.from_json(identity_value)
manager_sha256 = "sha256:" + hashlib.sha256(
    manager_path.read_bytes()
).hexdigest()
session_sha256 = "sha256:" + hashlib.sha256(
    session_path.read_bytes()
).hexdigest()
host_binary_sha256 = "sha256:" + hashlib.sha256(
    host_binary_path.read_bytes()
).hexdigest()
server_impl_sha256 = "sha256:" + hashlib.sha256(
    server_impl_path.read_bytes()
).hexdigest()
transport_client_source_sha256 = "sha256:" + hashlib.sha256(
    transport_client_source_path.read_bytes()
).hexdigest()
identity_value["software_identity"][
    "phone_resident_workers_sha256"
] = manager_sha256
identity_value["software_identity"][
    "phone_session_sha256"
] = session_sha256
identity_value["software_identity"][
    "host_binary_sha256"
] = host_binary_sha256
identity_value["software_identity"][
    "host_dependency_sha256:llama-server-impl"
] = server_impl_sha256
identity_value["software_identity"][
    "transport_client_source_sha256"
] = transport_client_source_sha256
identity = TransportQualificationIdentity.from_json(identity_value)
identity_path.write_text(
    json.dumps(
        identity.to_json(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n",
    encoding="ascii",
)

catalog = json.loads(catalog_path.read_text(encoding="ascii"))
replacement_count = 0

def replace_identity(value):
    global replacement_count
    if isinstance(value, dict):
        return {key: replace_identity(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_identity(item) for item in value]
    if value == parent.identity_sha256:
        replacement_count += 1
        return identity.identity_sha256
    return value

catalog = replace_identity(catalog)
if replacement_count <= 0:
    raise SystemExit("transport identity was not present in active catalog")
catalog_path.write_text(
    json.dumps(
        catalog,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n",
    encoding="ascii",
)

changed_fields = []
if (
    parent.software_identity["phone_resident_workers_sha256"]
    != manager_sha256
):
    changed_fields.append(
        "software_identity.phone_resident_workers_sha256"
    )
if parent.software_identity["phone_session_sha256"] != session_sha256:
    changed_fields.append("software_identity.phone_session_sha256")
if parent.software_identity["host_binary_sha256"] != host_binary_sha256:
    changed_fields.append("software_identity.host_binary_sha256")
if (
    parent.software_identity[
        "host_dependency_sha256:llama-server-impl"
    ] != server_impl_sha256
):
    changed_fields.append(
        "software_identity.host_dependency_sha256:llama-server-impl"
    )
if (
    parent.software_identity["transport_client_source_sha256"]
    != transport_client_source_sha256
):
    changed_fields.append(
        "software_identity.transport_client_source_sha256"
    )

derivation = {
    "changed_fields": changed_fields,
    "data_path_receipts_reused": list(identity.receipt_sha256s),
    "new_identity_sha256": identity.identity_sha256,
    "new_manager_sha256": manager_sha256,
    "new_phone_session_sha256": session_sha256,
    "new_host_binary_sha256": host_binary_sha256,
    "new_server_impl_sha256": server_impl_sha256,
    "new_transport_client_source_sha256": (
        transport_client_source_sha256
    ),
    "parent_identity_sha256": parent.identity_sha256,
    "reason": derivation_reason,
    "schema": "research-scheduler-transport-identity-derivation-v1",
    "updated_catalog_references": replacement_count,
}
derivation_path.write_text(
    json.dumps(
        derivation,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n",
    encoding="ascii",
)
PY

python3 - \
    "$repo" \
    "$manager" \
    "$server_root/llama-server" \
    "$server_root/libllama-server-impl.so" \
    "$root/SOURCE_MANIFEST.json" \
    "$previous/SOURCE_MANIFEST.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

source_root = Path(sys.argv[1]).resolve()
manager = Path(sys.argv[2]).resolve()
server = Path(sys.argv[3]).resolve()
server_impl = Path(sys.argv[4]).resolve()
output = Path(sys.argv[5])
parent = json.loads(Path(sys.argv[6]).read_text(encoding="ascii"))
rows = []
for path in sorted((source_root / "research_dev/scheduler").rglob("*")):
    if path.is_file() and path.suffix in {".py", ".sh", ".zsh"}:
        rows.append({
            "path": str(path.relative_to(source_root)),
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        })
native_sources = (
    source_root / "examples/layersplit/ffn-split-resident-workers.cpp",
    source_root / "examples/layersplit/ffn-split-client.cpp",
    source_root / "examples/layersplit/ffn-split-client.h",
    source_root / "tools/server/server.cpp",
)
rows.extend(
    {
        "path": str(path.relative_to(source_root)),
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for path in native_sources
)
rows.extend((
    {
        "path": str(manager),
        "sha256": "sha256:" + hashlib.sha256(manager.read_bytes()).hexdigest(),
    },
    {
        "path": str(server),
        "sha256": "sha256:" + hashlib.sha256(server.read_bytes()).hexdigest(),
    },
    {
        "path": str(server_impl),
        "sha256": "sha256:" + hashlib.sha256(server_impl.read_bytes()).hexdigest(),
    },
))

body = {
    "branch": parent["branch"],
    "files": rows,
    "head": parent["head"],
    "parent_manifest_sha256": parent["manifest_sha256"],
    "schema": "research-scheduler-source-manifest-v1",
}
encoded = json.dumps(
    body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
).encode("ascii")
body["manifest_sha256"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
output.write_text(
    json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ) + "\n",
    encoding="ascii",
)
PY

mapfile -d '' command < <(
    python3 - \
        "$previous/RUN_COMMAND.txt" \
        "$previous" \
        "$root" \
        "$command_old_repo" \
        "$repo" \
        "$command_old_manager" \
        "$phone_manager" \
        "$command_old_phone_bin_root" \
        "$phone_bin_root" \
        "$command_old_phone_session_root" \
        "$phone_session_root" \
        "$command_old_server_root" \
        "$server_root" \
        "$phone_session" \
        "$root/inputs/REPLAY_SCHEDULE.json" <<'PY'
import shlex
import sys

(
    path, old_root, new_root, old_repo, new_repo, old_manager, new_manager,
    old_phone_bin_root, new_phone_bin_root, old_phone_session_root,
    new_phone_session_root, old_server_root, new_server_root, new_session,
    replay_schedule,
) = (
    sys.argv[1:]
)
arguments = shlex.split(open(path, encoding="ascii").read())
replacements = (
    (old_root, new_root),
    (old_repo, new_repo),
    (old_manager, new_manager),
    (old_phone_bin_root, new_phone_bin_root),
    (old_phone_session_root, new_phone_session_root),
    (old_server_root, new_server_root),
    (
        "/data/local/tmp/s42-route-evidence-physical-v1/direct_phone_ffn_session.sh",
        new_session,
    ),
)
for option in ("--arrival-scale", "--request-indices", "--replay-schedule"):
    while option in arguments:
        index = arguments.index(option)
        del arguments[index:index + 2]
insert_at = arguments.index("--execute")
arguments[insert_at:insert_at] = ["--replay-schedule", replay_schedule]
for argument in arguments:
    for old, new in replacements:
        argument = argument.replace(old, new)
    sys.stdout.buffer.write(argument.encode("ascii") + b"\0")
PY
)

printf '%q ' "${command[@]}" > "$root/RUN_COMMAND.txt"
printf '\n' >> "$root/RUN_COMMAND.txt"
sha256sum "$root"/inputs/*.json \
    "$root/SOURCE_MANIFEST.json" \
    "$root/TRANSPORT_IDENTITY_DERIVATION.json" \
    > "$root/LAUNCH_INPUT_SHA256SUMS.txt"
sha256sum \
    "$repo/research_dev/scheduler/scheduler.py" \
    "$repo/research_dev/scheduler/adapters/phone_session.py" \
    "$repo/research_dev/scheduler/adapters/phone_transport.py" \
    "$repo/research_dev/scheduler/adapters/runtime.py" \
    "$repo/research_dev/scheduler/adapters/native/direct_phone_ffn_session.sh" \
    "$repo/research_dev/scheduler/_internal/model_placement_controller.py" \
    "$repo/research_dev/scheduler/_internal/runtime_execution.py" \
    "$repo/research_dev/scheduler/_internal/runtime_plan.py" \
    "$repo/examples/layersplit/ffn-split-resident-workers.cpp" \
    "$repo/examples/layersplit/ffn-split-client.cpp" \
    "$repo/examples/layersplit/ffn-split-client.h" \
    "$repo/tools/server/server.cpp" \
    "$server_root/llama-server" \
    "$server_root/libllama-server-impl.so" \
    "$manager" \
    > "$root/KEY_SOURCE_SHA256SUMS.txt"
parent_result="$previous/run/RESULT.json"
if [ ! -f "$parent_result" ]; then
    parent_result="$previous/run/FAILURE.json"
fi
if [ ! -f "$parent_result" ]; then
    parent_result="$previous/CAMPAIGN_STDERR.log"
fi
sha256sum "$parent_result" "$previous/RUN_COMMAND.txt" \
    > "$root/PARENT_EVIDENCE_SHA256.txt"

adb shell mkdir -p "$(dirname "$phone_manager")"
adb push "$manager" "$phone_manager" > "$root/ADB_PUSH_MANAGER.log"
adb push "$repo/research_dev/scheduler/adapters/native/direct_phone_ffn_session.sh" \
    "$phone_session" > "$root/ADB_PUSH_SESSION.log"
adb shell chmod 0755 "$phone_manager" "$phone_session"
adb shell sha256sum "$phone_manager" "$phone_session" \
    > "$root/PHONE_BINARIES_SHA256.txt"

if [ "${RUN_GATE:-0}" = 1 ]; then
    "${command[@]}" \
        > "$root/CAMPAIGN_STDOUT.log" \
        2> "$root/CAMPAIGN_STDERR.log"
fi
