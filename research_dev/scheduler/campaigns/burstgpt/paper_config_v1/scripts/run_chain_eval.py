"""Sync the staged main scheduler into the shared deploy, then run ordered steps. Run the whole chain under
ONE outer rig lock, e.g.

    flock -w 7200 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock \
        python3 run_chain_eval.py --status S.jsonl --stage STAGE [--prepare-script P] \
        [--server-identity CONFIG OUT] [--prepare ROOT ATTEMPT TEMPLATE TAG] [--idle-stop BUNDLE OUT] \
        [--arm INPUTS] ...

Copy of ../20260924-pixel-integration-2/run_chain_int2.py (same sync + tree-digest check, same battery log
and OP15 notify-512 stop, same launch.py preflight/run per arm, stop at the first failure) with:
- steps run in command-line order; besides arms: the Pixel-only server identity run
  (tools/qualify_pixel_server.py) and the rooted idle-stop qualification (tools/qualify_idle_stop.py);
- during every arm run and the server identity run, a phone-side ~1 Hz sysfs power sampler on both phones
  (tools/power_sampler.sh; stops on a stop file, never killed), pulled next to the run with host/phone
  time anchors;
- after every run: Pixel worker processes and adb forwards are listed (read-only).
Nothing here takes the lock again (a nested flock on the same file would deadlock).
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DEPLOY = Path("/mnt/storage/s42-trace-v2-20260921-prep/source")
OP15, PIXEL = "3C15AU002CL00000", "5A040DLCH004ES"
ADB = ["/usr/bin/adb", "-P", "5037"]
HERE = Path(__file__).resolve().parent
PHONE_DIR = "/data/local/tmp/s43-two-phone-eval-20260925"
SAMPLER_MAX_S = 5 * 3600


def tree_digest(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if (not path.is_file() or "__pycache__" in path.parts
                or relative.startswith("campaigns/burstgpt/reports/")):
            continue
        rows.append(relative + " " + hashlib.sha256(path.read_bytes()).hexdigest())
    return "sha256:" + hashlib.sha256("\n".join(rows).encode()).hexdigest()


class Step(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        steps = getattr(namespace, "steps", None) or []
        steps.append((self.dest, values if isinstance(values, list) else [values]))
        namespace.steps = steps


def shell(serial, command, timeout=60):
    return subprocess.run(ADB + ["-s", serial, "shell", command], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=timeout).stdout


def uptime(serial):
    return float(shell(serial, "cat /proc/uptime").split()[0])


class PowerSamplers:
    """Phone-side samplers on both phones for one run; stopped by a stop file, never signalled."""

    def __init__(self, directory: Path, name: str):
        self.directory, self.name, self.started = directory, name, {}

    def start(self):
        for label, serial in (("op15", OP15), ("pixel", PIXEL)):
            out, stop = f"{PHONE_DIR}/{self.name}-{label}.txt", f"{PHONE_DIR}/{self.name}-{label}.stop"
            if shell(serial, f"ls {out} {stop} 2>/dev/null").strip():
                raise RuntimeError("power sampler files exist: " + out)
            host_before = time.time()
            phone_uptime = uptime(serial)
            host_after = time.time()
            shell(serial, f"su -c 'nohup sh {PHONE_DIR}/power_sampler.sh {out} {stop} {SAMPLER_MAX_S} "
                          f"> /dev/null 2>&1 < /dev/null &'")
            self.started[label] = {"serial": serial, "out": out, "stop": stop, "anchor_host_epoch_s": (host_before + host_after) / 2,
                                   "anchor_host_monotonic_s": time.monotonic() - (time.time() - (host_before + host_after) / 2),
                                   "anchor_phone_uptime_s": phone_uptime, "anchor_uncertainty_s": host_after - host_before}

    def stop(self):
        result = {}
        for label, row in self.started.items():
            serial = row["serial"]
            shell(serial, "su -c 'touch " + row["stop"] + "'")
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and "# stop" not in shell(serial, "tail -n 1 " + row["out"]):
                time.sleep(1)
            host_before = time.time()
            phone_uptime = uptime(serial)
            host_after = time.time()
            local = self.directory / f"{self.name}-POWER-{label}.txt"
            subprocess.run(ADB + ["-s", serial, "pull", row["out"], str(local)], stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=120, check=True)
            text = local.read_text()
            result[label] = {**row, "local": str(local), "samples": sum(1 for line in text.splitlines() if not line.startswith("#")),
                             "stopped": "# stop" in text, "end_host_epoch_s": (host_before + host_after) / 2,
                             "end_phone_uptime_s": phone_uptime}
        (self.directory / f"{self.name}-POWER.json").write_text(json.dumps(result, indent=1) + "\n")
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True, help="staged repo root holding research_dev/scheduler")
    parser.add_argument("--prepare-script", type=Path)
    parser.add_argument("--server-identity", nargs=2, action=Step, metavar=("CONFIG", "OUT"))
    parser.add_argument("--prepare", nargs=4, action=Step, metavar=("ROOT", "ATTEMPT", "TEMPLATE", "TAG"))
    parser.add_argument("--idle-stop", nargs=2, action=Step, metavar=("BUNDLE", "OUT"))
    parser.add_argument("--arm", action=Step)
    args = parser.parse_args()
    steps = getattr(args, "steps", None) or []
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

    def leftovers(directory: Path, name: str) -> None:
        processes = shell(PIXEL, "ps -A -o PID,ARGS")
        workers = [line for line in processes.splitlines() if "llama-ffn-split-worker" in line]
        forwards = subprocess.run(ADB + ["forward", "--list"], stdin=subprocess.DEVNULL, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        (directory / (name + "-LEFTOVERS.json")).write_text(json.dumps(
            {"pixel_workers": workers, "adb_forwards": forwards}, indent=1) + "\n")
        record(stage="leftovers", name=name, pixel_workers=len(workers), adb_forwards=len(forwards.splitlines()))

    def run_logged(command, log_path, cwd=DEPLOY):
        with log_path.open("x") as log:
            return subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                  stderr=subprocess.STDOUT).returncode

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
        for label, serial in (("op15", OP15), ("pixel", PIXEL)):
            subprocess.run(ADB + ["-s", serial, "push", str(HERE / "tools/power_sampler.sh"), PHONE_DIR + "/"],
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=60, check=True)
        for kind, values in steps:
            if kind == "server_identity":
                config, output = Path(values[0]), Path(values[1])
                parent = output.parent
                batteries(parent, output.name + "-before")
                samplers = PowerSamplers(parent, output.name)
                samplers.start()
                record(stage="server-identity", output=str(output), status="STARTED")
                try:
                    code = run_logged([sys.executable, str(HERE / "tools/qualify_pixel_server.py"), "--config", str(config),
                                       "--output", str(output)], parent / (output.name + ".log"))
                finally:
                    samplers.stop()
                batteries(parent, output.name + "-after")
                leftovers(parent, output.name)
                status = json.loads((output / "RESULT.json").read_text()).get("status") if (output / "RESULT.json").is_file() else None
                record(stage="server-identity", output=str(output), exit_code=code, result_status=status,
                       failure=(output / "FAILURE.json").is_file())
                if code or status != "PASS":
                    return code or 1
            elif kind == "prepare":
                root, attempt, template, tag = values
                code = run_logged([sys.executable, str(args.prepare_script), root, attempt, template, tag],
                                  Path(root) / ("PREPARE-" + attempt + ".log"))
                record(stage="prepare", attempt=attempt, exit_code=code)
                if code:
                    return code
            elif kind == "idle_stop":
                bundle, output = Path(values[0]), Path(values[1])
                batteries(output.parent, output.name + "-before")
                code = run_logged([sys.executable, str(HERE / "tools/qualify_idle_stop.py"), str(bundle), str(output)],
                                  output.parent / (output.name + ".log"))
                batteries(output.parent, output.name + "-after")
                leftovers(output.parent, output.name)
                status = json.loads((output / "RESULT.json").read_text()).get("status") if (output / "RESULT.json").is_file() else None
                record(stage="idle-stop", output=str(output), exit_code=code, result_status=status)
                if code or status != "PASS":
                    return code or 1
            elif kind == "arm":
                inputs = Path(values[0])
                for stage in ("preflight", "run"):
                    if (inputs / "CANCEL").exists():
                        raise RuntimeError("CANCEL requested")
                    output = inputs / (stage + "-eval")
                    batteries(inputs, stage + "-before")
                    record(stage=stage, inputs=str(inputs), status="STARTED")
                    command = [sys.executable, "research_dev/scheduler/campaigns/burstgpt/launch.py",
                               str(inputs / "campaign.json"), str(output)]
                    if stage == "preflight":
                        command.append("--preflight-only")
                        code = run_logged(command, inputs / (stage + "-eval.log"))
                    else:
                        samplers = PowerSamplers(inputs, inputs.name + "-run")
                        samplers.start()
                        try:
                            code = run_logged(command, inputs / (stage + "-eval.log"))
                        finally:
                            samplers.stop()
                    batteries(inputs, stage + "-after")
                    leftovers(inputs, stage)
                    row = {"stage": stage, "inputs": str(inputs), "exit_code": code}
                    if stage == "preflight" and (output / "PHYSICAL_PREFLIGHT.json").is_file():
                        row["preflight_status"] = json.loads((output / "PHYSICAL_PREFLIGHT.json").read_text()).get("status")
                    if stage == "run" and (output / "run/RESULT.json").is_file():
                        result = json.loads((output / "run/RESULT.json").read_text())
                        row["result_status"] = result.get("status")
                        row["helper_phone_calls"] = sum(
                            int(entry.get("calls", 0)) for proof in result.get("physical_execution_proofs", {}).values()
                            for entry in proof.get("phone_calls_by_session", [])
                            if str(entry.get("session_id", "")).startswith("PIXEL"))
                    if stage == "run":
                        row["failure"] = (output / "run/FAILURE.json").is_file()
                    record(**row)
                    if code or row.get("preflight_status", "PASS") != "PASS":
                        return code or 1
        return 0
    except Exception as error:
        record(stage="exception", status="FAIL", error=repr(error))
        return 1
    finally:
        record(stage="done")


if __name__ == "__main__":
    sys.exit(main())
