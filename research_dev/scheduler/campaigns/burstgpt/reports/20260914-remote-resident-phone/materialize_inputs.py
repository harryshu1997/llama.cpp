"""Freeze fresh transport receipts and the bounded relocation command."""

import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = Path("/mnt/storage/s42-remote-resident-phone-20260913-v2-WTUd4V")
sys.path.insert(0, str(ROOT / "source"))
from research_dev.scheduler.adapters.transport_profiles import (
    build_transport_qualification_identity, materialize_measured_usb_links,
)
from research_dev.scheduler._internal.types import canonical_json
from research_dev.scheduler import ResourceProfile


def save(path, value):
    with path.open("x") as stream:
        stream.write(canonical_json(value) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


command = json.loads((PREVIOUS / "GATE_COMMAND.json").read_text())
save(ROOT / "PREVIOUS_COMMAND.json", command)
value = lambda flag: command[command.index(flag) + 1]
adb = [value("--adb"), "-P", value("--adb-port"), "-s", value("--phone-usb-serial")]
phone_hash = lambda path: "sha256:" + subprocess.check_output(
    adb + ["shell", "sha256sum " + path], text=True).split()[0]
candidate = json.loads((ROOT / "TRANSPORT_BOOT.json").read_text())["candidate"]
qualification = json.loads((ROOT / "TRANSPORT_RESULT.json").read_text())
assert qualification["status"] == "PASS" and qualification["cases"] == 9
assert qualification["boot_id"] == candidate["boot_id"]
old = json.loads((PREVIOUS / "TRANSPORT_IDENTITY.json").read_text())
hardware = {**old["hardware_identity"], "phone_boot_image_sha256": "sha256:" + candidate["image_sha256"]}
dependencies = dict(word.split("=", 1) for i, word in enumerate(command)
                    if i and command[i - 1] == "--transport-host-dependency")
workers, router = ROOT / "resident-workers.android", ROOT / "resident-router.android"
for path, local in ((value("--phone-resident-workers"), workers),
                    (value("--phone-resident-router"), router)):
    assert not local.exists()
    subprocess.run(adb + ["pull", path, str(local)], check=True)
    assert digest(local) == phone_hash(path)
identity = build_transport_qualification_identity(
    identity_id=ROOT.name + ":candidate-boot-fresh-transport",
    transport_generation=old["transport_generation"], hardware_identity=hardware,
    phone_session_sha256=phone_hash(value("--phone-session")),
    phone_worker_sha256=phone_hash(value("--phone-worker")),
    qualification_phone_session_sha256=phone_hash(
        "/data/local/tmp/s42-rrphone-20260914-v3b/functionfs_transport_session.sh"),
    qualification_phone_worker_sha256=phone_hash(
        "/data/local/tmp/s41-ffs-dmabuf-v1/ffs_dmabuf_phone.android"),
    host_binary_path=Path(value("--server")), qualification_binary_path=ROOT / "ffs_dmabuf_host",
    transport_client_source_path=ROOT / "source/examples/layersplit/ffn-split-usb-client.cpp",
    qualified_allocators=("devmem",), receipt_paths=tuple(sorted((ROOT / "transport").glob("*.json"))),
    minimum_usb_speed_mbps=5000,
    host_dependency_paths={name: Path(path) for name, path in dependencies.items()},
    phone_resident_workers_path=workers, phone_resident_router_path=router,
)
save(ROOT / "TRANSPORT_IDENTITY.json", identity.to_json())
catalog = json.loads((PREVIOUS / "GATE_CATALOG.json").read_text())
links = materialize_measured_usb_links((ROOT / "transport",), identity,
    host_device_id="desktop-cpu", phone_device_id="op15-phone")
catalog["placement_profile"]["links"] = [row for row in catalog["placement_profile"]["links"]
    if not row.get("transport_generation", "").startswith("functionfs")] + [dataclasses.asdict(row) for row in links]
resources = {row["resource_id"]: row for row in catalog["resources"]}
for link in links:
    key = "link:" + link.link_id
    resources.setdefault(key, dataclasses.asdict(ResourceProfile(
        resource_id=key, kind="transport", capacity=1, ready=True, identity=link.link_id)))
catalog["resources"] = [resources[key] for key in sorted(resources)]
save(ROOT / "GATE_CATALOG.json", catalog)
updates = {
    "--capability-catalog": str(ROOT / "GATE_CATALOG.json"),
    "--usb-qualification-identity": str(ROOT / "TRANSPORT_IDENTITY.json"),
    "--phone-boot-image-sha256": hardware["phone_boot_image_sha256"],
    "--phone-session-root": "/data/local/tmp/" + ROOT.name,
    "--phone-remote-hash-cache": str(ROOT / "PHONE_HASH_CACHE.json"),
    "--output": str(ROOT / "gate-v1"),
    "--selection-mode": "calibration",
}
for flag, replacement in updates.items():
    command[command.index(flag) + 1] = replacement
command[1] = str(ROOT / "source/research_dev/scheduler/campaigns/burstgpt/remote_resident_gate.py")
command.remove("--recovery")
command.extend(("--maximum-phone-sessions", "1"))
save(ROOT / "GATE_COMMAND.json", command)
save(ROOT / "INPUT_DIFF.json", {
    "source_catalog": str(PREVIOUS / "GATE_CATALOG.json"), "updated_command_fields": updates,
    "transport_identity_sha256": identity.identity_sha256,
    "model_weights": "existing FFN shard files; no regeneration or native rebuild",
    "energy_qualification": "remote parent remains CALIBRATION_PENDING; transport receipts qualify USB only",
    "owner_loss": "in-flight owner loss is not attempted; idle USB cancellation does not qualify HTP cancellation",
    "kernel_boot_id": candidate["boot_id"],
})
print(identity.identity_sha256)
