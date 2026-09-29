"""Sync the staged main scheduler into the shared deploy, optionally materialize inputs, then preflight
and run arms in order. Run the whole chain under ONE outer rig lock, e.g.

    flock -w 7200 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock \
        python3 run_chain_int2.py --status S.jsonl --stage STAGE --prepare ROOT ATTEMPT TEMPLATE TAG \
        --arm INPUTS[:require-helper-calls] ...

Nothing here takes the lock again (a nested flock on the same file would deadlock). Every arm is logged
with OP15 `dumpsys battery` + `battery_notify_code` and Pixel `dumpsys battery` before and after its
preflight and its run; OP15 notify code 512 stops the chain. The chain stops at the first failed step.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

DEPLOY = Path("/mnt/storage/s42-trace-v2-20260921-prep/source")
OP15, PIXEL = "3C15AU002CL00000", "5A040DLCH004ES"
ADB = ["/usr/bin/adb", "-P", "5037"]


def tree_digest(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if (not path.is_file() or "__pycache__" in path.parts
                or relative.startswith("campaigns/burstgpt/reports/")):
            continue
        rows.append(relative + " " + hashlib.sha256(path.read_bytes()).hexdigest())
    return "sha256:" + hashlib.sha256("\n".join(rows).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True, help="staged repo root holding research_dev/scheduler")
    parser.add_argument("--prepare", nargs=4, metavar=("ROOT", "ATTEMPT", "TEMPLATE", "TAG"))
    parser.add_argument("--prepare-script", type=Path)
    parser.add_argument("--arm", action="append", default=[])
    args = parser.parse_args()
    if args.status.exists():
        raise SystemExit("status file exists")
    env = {**os.environ, "GIT_DIR": str(DEPLOY / ".git"), "GIT_WORK_TREE": str(DEPLOY),
           "GIT_OPTIONAL_LOCKS": "0", "S42_UNIFIED_REPO_ROOT": str(DEPLOY), "LANG": "C.UTF-8",
           "LD_LIBRARY_PATH": "/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64"}

    def record(**row):
        row["at"] = datetime.now(timezone.utc).isoformat()
        with args.status.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    def shell(serial, command):
        return subprocess.run(ADB + ["-s", serial, "shell", command], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=60).stdout

    def batteries(directory: Path, name: str) -> None:
        result = {"op15_dumpsys": shell(OP15, "dumpsys battery"),
                  "op15_notify_code": shell(OP15, "su -c 'cat /sys/class/oplus_chg/battery/battery_notify_code'").strip(),
                  "pixel_dumpsys": shell(PIXEL, "dumpsys battery"),
                  "at": datetime.now(timezone.utc).isoformat()}
        (directory / (name + "-BATTERY.json")).write_text(json.dumps(result, indent=1) + "\n")
        level = lambda text: next((line.split(":")[1].strip() for line in text.splitlines()
                                   if line.strip().startswith("level:")), None)
        record(stage="battery", name=name, op15_level=level(result["op15_dumpsys"]),
               op15_notify=result["op15_notify_code"], pixel_level=level(result["pixel_dumpsys"]))
        if result["op15_notify_code"] == "512":
            raise RuntimeError("OP15 battery_notify_code 512 (charger latched off): stop and replug")

    try:
        target = DEPLOY / "research_dev/scheduler"
        before = tree_digest(target)
        subprocess.run(["rsync", "-a", "--checksum", "--exclude=__pycache__", "--exclude=campaigns/burstgpt/reports",
                        str(args.stage / "research_dev/scheduler") + "/", str(target) + "/"],
                       check=True, stdin=subprocess.DEVNULL)
        after, staged = tree_digest(target), tree_digest(args.stage / "research_dev/scheduler")
        record(stage="sync", before=before, after=after, staged=staged, status="PASS" if after == staged else "FAIL")
        if after != staged:
            return 1
        if args.prepare:
            root, attempt, template, tag = args.prepare
            with (Path(root) / ("PREPARE-" + attempt + ".log")).open("x") as log:
                code = subprocess.run([sys.executable, str(args.prepare_script), root, attempt, template, tag],
                                      cwd=DEPLOY, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                      stderr=subprocess.STDOUT).returncode
            record(stage="prepare", attempt=attempt, exit_code=code)
            if code:
                return code
        for spec in args.arm:
            inputs, _, option = spec.partition(":")
            inputs = Path(inputs)
            for stage in ("preflight", "run"):
                if (inputs / "CANCEL").exists():
                    raise RuntimeError("CANCEL requested")
                output = inputs / (stage + "-int2")
                batteries(inputs, stage + "-before")
                record(stage=stage, inputs=str(inputs), status="STARTED")
                command = [sys.executable, "research_dev/scheduler/campaigns/burstgpt/launch.py",
                           str(inputs / "campaign.json"), str(output)]
                if stage == "preflight":
                    command.append("--preflight-only")
                with (inputs / (stage + "-int2.log")).open("x") as log:
                    code = subprocess.run(command, cwd=DEPLOY, env=env, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT).returncode
                batteries(inputs, stage + "-after")
                row = {"stage": stage, "inputs": str(inputs), "exit_code": code}
                if stage == "preflight" and (output / "PHYSICAL_PREFLIGHT.json").is_file():
                    row["preflight_status"] = json.loads((output / "PHYSICAL_PREFLIGHT.json").read_text()).get("status")
                if stage == "run" and (output / "run/RESULT.json").is_file():
                    result = json.loads((output / "run/RESULT.json").read_text())
                    row["result_status"] = result.get("status")
                    helper = sum(int(entry.get("calls", 0)) for proof in result.get("physical_execution_proofs", {}).values()
                                 for entry in proof.get("phone_calls_by_session", [])
                                 if str(entry.get("session_id", "")).startswith("PIXEL"))
                    row["helper_phone_calls"] = helper
                record(**row)
                if code or row.get("preflight_status", "PASS") != "PASS":
                    return code or 1
                if stage == "run" and option == "require-helper-calls" and not row.get("helper_phone_calls"):
                    record(stage="stop", reason="two-phone arm made zero helper-phone calls")
                    return 1
        return 0
    except Exception as error:
        record(stage="exception", status="FAIL", error=repr(error))
        return 1
    finally:
        record(stage="done")


if __name__ == "__main__":
    sys.exit(main())
