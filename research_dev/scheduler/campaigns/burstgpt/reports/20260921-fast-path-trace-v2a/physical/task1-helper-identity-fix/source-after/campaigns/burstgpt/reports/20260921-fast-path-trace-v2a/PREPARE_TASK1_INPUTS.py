"""Fresh Task 1 inputs bound to the measured coalesced transport."""

import json
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys

DEPLOY = Path("/mnt/storage/s42-trace-v2-20260921-prep")
TRANSPORT = Path("/mnt/storage/s42-task1-transport-20260922-v2")
SOURCE = Path("/home/zhihao/s42-trace-v2a-m4a3-20260921-inputs")
QUALIFIER = Path("/mnt/storage/s42-ffn-microbatch-20260916-v4-arwuw3")
ADB = ["/usr/bin/adb", "-P", "5037", "-s", "3C15AU002CL00000"]


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def phone_hash(path):
    command = "sha256sum " + shlex.quote(path)
    return "sha256:" + subprocess.check_output(
        [*ADB, "shell", "su", "-c", shlex.quote(command)],
        text=True, stdin=subprocess.DEVNULL).split()[0]


def main(destination):
    if load(TRANSPORT / "RESULT.json")["status"] != "PASS":
        raise RuntimeError("transport measurement did not pass")
    before = load(TRANSPORT / "BEFORE.json")
    boot = load(Path("/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/CANDIDATE_BOOT_RESULT.json"))
    observed = [row.split()[0] for row in before["kernel_hashes"].splitlines()]
    if observed != [boot["identity"]["notes"], boot["identity"]["btf"]]:
        raise RuntimeError("running kernel differs from the known boot image")
    image = Path("/mnt/storage/s42-dmabuf-cancel-20260914-v1-gBJdFx/candidate-boot.img")
    if hashlib.sha256(image.read_bytes()).hexdigest() != boot["image_sha256"]:
        raise RuntimeError("verified boot image file changed")
    if subprocess.check_output([*ADB, "shell", "cat", "/proc/sys/kernel/random/boot_id"],
                               text=True, stdin=subprocess.DEVNULL).strip() != before["boot_id"]:
        raise RuntimeError("phone rebooted after qualification")
    destination.mkdir(exist_ok=False)
    for name in ("campaign.json", "rig.json", "models.json", "evidence.json"):
        data = json.loads((SOURCE / name).read_text().replace(str(SOURCE), str(destination)))
        if name == "campaign.json":
            data["campaign_id"] = destination.name.removesuffix("-inputs")
            data["adaptive_decode_overrides"] = {"server_policy_coherence": True}
        elif name == "rig.json":
            data["phone"]["boot_image_sha256"] = "sha256:" + boot["image_sha256"]
            data["binaries"]["phone_boot_image"] = str(image)
        elif name == "models.json":
            for model in data["models"]:
                if model["model_key"] == "hot":
                    model["phone_adapter_parameters"].pop("usb_batch_plan", None)
                    model["phone_batch_plans"] = ["coalesced-batch"]
                    model["qualified_phone_batch_plans"] = ["coalesced-batch"]
        elif name == "evidence.json":
            data["transport_qualification_directories"] = [str(TRANSPORT / "receipts")]
            data["transport_qualification_identity_path"] = str(destination / "TRANSPORT_QUALIFICATION_IDENTITY.json")
        save(destination / name, data)
    command = load(DEPLOY / "software/MATERIALIZE_COMMAND.json")
    replacements = {
        "--identity-id": destination.name,
        "--phone-boot-image-sha256": "sha256:" + boot["image_sha256"],
        "--qualification-binary": str(QUALIFIER / "ffs_dmabuf_host"),
        "--qualification-phone-session-sha256": phone_hash(
            "/data/local/tmp/s42-rrphone-20260914-v3b/functionfs_transport_session.sh"),
        "--qualification-phone-worker-sha256": phone_hash(
            "/data/local/tmp/s41-ffs-dmabuf-v1/ffs_dmabuf_phone.android"),
        "--output": str(destination / "TRANSPORT_QUALIFICATION_IDENTITY.json"),
    }
    for flag, value in replacements.items():
        command[command.index(flag) + 1] = value
    while "--receipt" in command:
        index = command.index("--receipt")
        del command[index:index + 2]
    for receipt in sorted((TRANSPORT / "receipts").glob("*.json")):
        command.extend(("--receipt", str(receipt)))
    previous = load(DEPLOY / "TRANSPORT_QUALIFICATION_IDENTITY.json")
    for name, path in (
        ("phone_session", "/data/local/tmp/s42-hal-runtime-probe-20260906-v4/direct_phone_ffn_session.sh"),
        ("phone_worker", "/data/local/tmp/s42-ffn-shards-20260904-v1-bin/llama-ffn-split-worker"),
        ("phone_resident_workers", "/data/local/tmp/s42-per-session-correctness-20260903-v2-bin/llama-ffn-split-resident-workers"),
        ("phone_resident_router", "/data/local/tmp/s42-ready-subset-router-20260905-v2/llama-ffn-split-resident-router"),
    ):
        if phone_hash(path) != previous["software_identity"][name + "_sha256"]:
            raise RuntimeError("phone inference binary changed: " + name)
    save(destination / "MATERIALIZE_COMMAND.json", command)
    subprocess.run(command, cwd=DEPLOY / "source", check=True, stdin=subprocess.DEVNULL)
    identity = load(destination / "TRANSPORT_QUALIFICATION_IDENTITY.json")
    rig = load(destination / "rig.json")
    for field, identity_field in (("boot_image_sha256", "phone_boot_image_sha256"),
                                  ("kernel_release", "phone_kernel_release"),
                                  ("serial", "phone_usb_serial")):
        if rig["phone"][field] != identity["hardware_identity"][identity_field]:
            raise RuntimeError("rig and transport identity differ: " + field)
    save(destination / "INPUT_DELTA.json", {
        "source": str(SOURCE), "server_policy_coherence": True,
        "qwen_usb_batch_plan": "coalesced-batch", "maximum_qualified_payload_bytes": 40960,
        "transport_result": str(TRANSPORT / "RESULT.json"),
        "kernel_notes_and_btf_match_boot_image": boot["image_sha256"],
        "native_rebuild": False, "kernel_changed": False,
    })


if __name__ == "__main__":
    main(Path(sys.argv[1]))
