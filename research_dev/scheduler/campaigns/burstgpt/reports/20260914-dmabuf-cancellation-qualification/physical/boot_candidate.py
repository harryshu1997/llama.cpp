"""Use the existing RAM-boot recorder for one approved isolated kernel candidate."""
import importlib.util
import json
from pathlib import Path
import time

from src import run_direct_usb_device as device

ROOT = Path(__file__).resolve().parent
IMAGE_SHA = "f13c7c033ce74f32ea4aa6349cb531704407708f5fb35756a6c361ffb48322a3"
RECORDER_SHA = "82fa4ebf4b0934aa2339419d5db70bb07859f6a707215534d18675d9a98893f4"
FASTBOOT_SHA = "76dde33fee8b1fd00bcaf2e7f94ddef6407f0beb5bc3a98a3d4127307af23f3a"
EXPECTED = {
    "notes": "40bbacf73c0d35195693c566ae695803f59fbf7ec8ce817a1921fb44b3807b51",
    "btf": "77a8ce5acc215506e8dff1bf84e4561ef316680bcba803e9dfc16acecc449735",
    "config": "9f03ed30a44329ebc6337dca7157f3eaa67c3143519883b026c51abd0d7dda43",
}


def main():
    if device.digest(ROOT / "candidate-boot.img") != IMAGE_SHA:
        raise RuntimeError("candidate image differs")
    if device.digest(ROOT / "restore-original.py") != RECORDER_SHA:
        raise RuntimeError("RAM-boot recorder differs")
    rig = json.loads(Path("/mnt/storage/s42-cuda-graph-v1-20260909/reference-inputs/matched-rig.json").read_text())
    if device.digest(rig["binaries"]["fastboot"]) != FASTBOOT_SHA:
        raise RuntimeError("fastboot differs")
    rig["phone"]["boot_image_sha256"] = "sha256:" + IMAGE_SHA
    rig["binaries"]["phone_boot_image"] = str(ROOT / "candidate-boot.img")
    device.save(ROOT, "CANDIDATE_RIG.json", rig)
    spec = importlib.util.spec_from_file_location("recorded_boot", ROOT / "restore-original.py")
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)
    recorder.ROOT = ROOT / "boot"
    recorder.RIG = ROOT / "CANDIDATE_RIG.json"
    recorder.IMAGE_SHA256 = IMAGE_SHA
    with device.execution_lock():
        preflight = ROOT / "boot-preflight"
        preflight.mkdir()
        prior_boot = device.preflight(preflight)
        prior_identity = device.shell("sha256sum /sys/kernel/notes /sys/kernel/btf/vmlinux")
        device.save(ROOT, "BOOT_AUTHORIZATION.json", {
            "user_authorized": "temporary RAM-boot new OP15 kernel and bounded cancellation prerequisite; yes",
            "scope": "RAM boot only; no flash, erase, unlock, slot change, desktop reboot or unrelated process stop",
            "image_sha256": IMAGE_SHA, "recorder_sha256": RECORDER_SHA,
            "prior_boot_id": prior_boot, "prior_identity": prior_identity,
            "started_epoch_ns": time.time_ns(), "new_kernel_qualified": False,
            "note": "The archived recorder's restoration/dev3 labels do not qualify this new candidate.",
        })
        recorder.main()
        commands = {
            "notes": "sha256sum /sys/kernel/notes",
            "btf": "sha256sum /sys/kernel/btf/vmlinux",
            "config": "set -o pipefail; zcat /proc/config.gz | sha256sum",
        }
        identity = {key: device.shell(command).split()[0] for key, command in commands.items()}
        current_boot = device.shell("cat /proc/sys/kernel/random/boot_id")
        if identity != EXPECTED or current_boot == prior_boot:
            raise RuntimeError("candidate boot identity was not proven")
        device.save(ROOT, "CANDIDATE_BOOT_RESULT.json", {
            "status": "CANDIDATE_BOOTED_IDENTITY_VERIFIED", "identity": identity,
            "image_sha256": IMAGE_SHA, "boot_id": current_boot,
            "finished_epoch_ns": time.time_ns(), "partition_flash_count": 0,
            "cancellation_tested": False, "model_inference_tested": False,
        })
        print("CANDIDATE_BOOTED_IDENTITY_VERIFIED " + current_boot, flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        device.save(ROOT, "BOOT_FAILED.json", {"error": repr(error), "finished_epoch_ns": time.time_ns()})
        raise
