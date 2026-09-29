"""Preflight and run one fresh Pixel development arm under the shared rig lock."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    inputs = Path(sys.argv[1])
    attempt = sys.argv[2]
    preflight = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    campaign = json.loads((inputs / "campaign.json").read_text())
    rig = json.loads(Path(campaign["rig_manifest_path"]).read_text())
    source = Path(rig["repo_root"])
    env = {**os.environ, "GIT_DIR": "/mnt/storage/s42-trace-v2-20260921-prep/source/.git",
           "GIT_WORK_TREE": str(source), "GIT_OPTIONAL_LOCKS": "0", "S42_UNIFIED_REPO_ROOT": str(source),
           "LANG": "C.UTF-8", "LD_LIBRARY_PATH": "/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64"}
    lock = ["flock", "-w", "900", "/home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock"]
    status = inputs / ("STATUS-" + attempt + ".jsonl")
    if status.exists():
        raise RuntimeError("attempt already exists")

    def record(**row):
        row["at"] = datetime.now(timezone.utc).isoformat()
        with status.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def batteries(stage):
        result = {}
        for serial in ("3C15AU002CL00000", "5A040DLCH004ES"):
            adb = ["/usr/bin/adb", "-P", "5037", "-s", serial, "shell"]
            result[serial] = subprocess.check_output(adb + ["dumpsys battery"], stdin=subprocess.DEVNULL,
                                                     text=True, timeout=30)
            if serial == "3C15AU002CL00000":
                result["battery_notify_code"] = subprocess.check_output(adb + ["su -c 'cat /sys/class/oplus_chg/battery/battery_notify_code'"],
                    stdin=subprocess.DEVNULL, text=True, timeout=30).strip()
        (inputs / (stage + "-BATTERY-" + attempt + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        if result["battery_notify_code"] == "512":
            raise RuntimeError("OP15 charger latched off; user replug required")

    try:
        for stage in ("preflight", "run"):
            if (inputs / "CANCEL").exists():
                raise RuntimeError("CANCEL requested before stage")
            if stage == "preflight" and preflight is not None:
                result = json.loads((preflight / "PHYSICAL_PREFLIGHT.json").read_text())
                assert result["status"] == "PASS"
                record(stage=stage, status="REUSED", output=str(preflight))
                continue
            output = inputs / (stage + "-" + attempt)
            if output.exists():
                raise RuntimeError("output already exists: " + str(output))
            batteries(stage + "-before")
            record(stage=stage, status="STARTED", output=str(output))
            command = lock + ["python3", "research_dev/scheduler/campaigns/burstgpt/launch.py",
                              str(inputs / "campaign.json"), str(output)]
            if stage == "preflight":
                command.append("--preflight-only")
            with (inputs / (stage + "-" + attempt + ".log")).open("x") as log:
                code = subprocess.run(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            (inputs / (stage + "-EXIT-" + attempt + ".txt")).write_text(str(code) + "\n")
            record(stage=stage, exit_code=code)
            batteries(stage + "-after")
            if code:
                return code
        result = json.loads((inputs / ("run-" + attempt) / "run/RESULT.json").read_text())
        record(stage="complete", status=result.get("status"))
        return 0
    except Exception as error:
        record(stage="exception", status="FAIL", error=repr(error))
        return 1
    finally:
        (inputs / ("DONE-" + attempt + ".json")).write_text(json.dumps({"finished_at": datetime.now(timezone.utc).isoformat()}) + "\n")


if __name__ == "__main__":
    sys.exit(main())
