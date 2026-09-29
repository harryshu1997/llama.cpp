"""Run the existing M3 gate arms sequentially under one shared-rig lock."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import subprocess
import time
from types import SimpleNamespace

sys.path.insert(0, os.environ["S42_UNIFIED_REPO_ROOT"])
from research_dev.scheduler.campaigns.burstgpt import two_phone_gate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--arms", default="desktop-before,op15-full,both-matched,both-full,desktop-after")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["parallel"] = args.parallel
    args.output.mkdir()
    statuses = []
    with Path(config["execution_lock"]).open("a") as lock:
        deadline = time.monotonic() + 900
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        for label, arm, helper, host_columns in (
            ("desktop-before", "control", False, 17408),
            ("op15-full", "combined", False, 0),
            ("both-matched", "combined", True, 4352),
            ("both-full", "combined", True, 0),
            ("desktop-after", "control", False, 17408),
        ):
            if label not in args.arms.split(","):
                continue
            options = SimpleNamespace(
                host_columns=host_columns, select=None, no_dormant=False,
                sweep_host_columns="13056,8704,4352,0", output_tokens=64,
                prompt_chars=0, prompt_tokens_list=",".join(
                    str(256 + 64 * index) for index in range(args.parallel)),
            )
            gate = two_phone_gate.install(two_phone_gate.load_gate(), with_helper=helper)
            row = {"arm": label, "started_epoch_s": time.time(), "status": "RUNNING"}
            statuses.append(row)
            try:
                print("ARM_START", label, flush=True)
                adb = ["/usr/bin/adb", "-P", "5037", "-s", config["phone"]["serial"]]
                battery = subprocess.check_output(adb + ["shell", "dumpsys battery"],
                                                  stdin=subprocess.DEVNULL, text=True, timeout=30)
                notify = subprocess.check_output(adb + ["shell", "su -c 'cat /sys/class/oplus_chg/battery/battery_notify_code'"],
                                                 stdin=subprocess.DEVNULL, text=True, timeout=30).strip()
                (args.output / (label + "-battery-before.json")).write_text(json.dumps(
                    {"battery": battery, "battery_notify_code": notify}, indent=2) + "\n")
                if notify == "512":
                    raise RuntimeError("OP15 charger latched off: battery_notify_code=512")
                gate.run(config, args.output / label, arm, options)
                row["status"] = "COMPLETED"
            except BaseException as error:
                row.update(status="FAIL", error=repr(error))
                raise
            finally:
                row["finished_epoch_s"] = time.time()
                (args.output / "STATUS.json").write_text(json.dumps(statuses, indent=2) + "\n")
            print("ARM_DONE", label, flush=True)


if __name__ == "__main__":
    main()
