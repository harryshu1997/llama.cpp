"""Prepare or audit finite, phone-local CPU/GPU bandwidth sweeps."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import statistics


def read(path):
    return json.loads(path.read_text())


def save(path, obj):
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2)
        stream.write("\n")


def prepare(args):
    root = args.root
    root.mkdir()
    config = read(args.config)
    phone = config["phone_dir"]
    if not phone.startswith("/data/local/tmp/s42-pixel10pro-bandwidth-"):
        raise ValueError("private directory required")
    hashes = {}
    for name in ("pixel-bandwidth", "read_v1_u1.spv", "read_v4_u1.spv", "read_v4_u4.spv", "read_v4_u8.spv"):
        payload = (args.software / name).read_bytes()
        (root / name).write_bytes(payload)
        hashes[phone + "/" + name] = hashlib.sha256(payload).hexdigest()
    save(root / "CONFIG.json", config)
    save(root / "EXPECTED_HASHES.json", hashes)
    (root / "EXPECTED_HASHES.sha256").write_text("".join(f"{v}  {k}\n" for k, v in hashes.items()))
    script = """#!/system/bin/sh
cd PHONE || exit 2
exec 9>/data/local/tmp/.s42-pixel-ffn-kernels.lock
flock -n 9 9>&9 || exit 3
mkdir raw || exit 4
ps -A -o PID,ARGS > raw/PROCESSES_BEFORE.txt
if grep -Eq '[l]lama-ffn|/[p]ixel-bandwidth ' raw/PROCESSES_BEFORE.txt; then echo 'FAIL existing worker'; exit 5; fi
cat /proc/sys/kernel/random/boot_id > raw/BOOT_BEFORE.txt
cat /proc/meminfo > raw/MEMORY_BEFORE.txt
sha256sum HASH_FILES > raw/HASHES.txt || exit 6
cmp raw/HASHES.txt EXPECTED_HASHES.sha256 || exit 6
snapshot() {
    date -u
    dumpsys battery
    for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq /sys/class/devfreq/*/cur_freq; do
        echo "$f"
        cat "$f"
    done
}
run_arm() {
    name="$1"
    shift
    mkdir "raw/$name" || return 7
    snapshot > "raw/$name/BEFORE.txt" 2>&1
    "$@" > "raw/$name/RESULTS.jsonl" 2> "raw/$name/STDERR.txt"
    code=$?
    echo "$code" > "raw/$name/EXIT.txt"
    snapshot > "raw/$name/AFTER.txt" 2>&1
    if [ "$code" != 0 ]; then echo "FAIL $name exit=$code"; return "$code"; fi
    echo "PASS $name"
}
""".replace("PHONE", shlex.quote(phone)).replace("HASH_FILES", shlex.join(hashes))
    for arm in config["arms"]:
        command = [phone + "/pixel-bandwidth", arm["mode"], phone + f"/read_v{arm.get('width', 4)}_u{arm.get('unroll', 1)}.spv",
                   str(arm.get("width", 4)), str(arm.get("wg", 128)), str(arm.get("lanes", 65536)),
                   str(arm.get("threads", 4)), arm.get("mask", "0"), str(arm.get("prefetch", 0)),
                   str(config.get("seconds", 2)), str(config.get("MiB", 512))]
        if "cpu_chunk_KiB" in arm:
            command.append(str(arm["cpu_chunk_KiB"]))
        script += shlex.join(["run_arm", arm["name"], *command]) + " || exit $?\n"
    script += """cat /proc/sys/kernel/random/boot_id > raw/BOOT_AFTER.txt
ps -A -o PID,ARGS > raw/PROCESSES_AFTER.txt
cat /proc/meminfo > raw/MEMORY_AFTER.txt
if grep -Eq '[l]lama-ffn|/[p]ixel-bandwidth ' raw/PROCESSES_AFTER.txt; then echo 'FAIL remaining worker'; exit 12; fi
cmp raw/BOOT_BEFORE.txt raw/BOOT_AFTER.txt || exit 13
echo PASS > raw/DONE.txt
echo 'PASS all arms'
"""
    (root / "RUN_PHONE.sh").write_text(script)
    (root / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(f"PREPARED {len(config['arms'])} bounded arms")


def audit(args):
    root, raw = args.root, args.root / "raw"
    if (root / "EXIT.txt").read_text().strip() != "0" or (raw / "DONE.txt").read_text().strip() != "PASS":
        raise ValueError("run failed/incomplete")
    if (raw / "BOOT_BEFORE.txt").read_bytes() != (raw / "BOOT_AFTER.txt").read_bytes():
        raise ValueError("boot changed")
    if (raw / "HASHES.txt").read_bytes() != (root / "EXPECTED_HASHES.sha256").read_bytes():
        raise ValueError("binary identity mismatch")
    config = read(root / "CONFIG.json")
    results = []
    for arm in config["arms"]:
        path = raw / arm["name"]
        if (path / "EXIT.txt").read_text().strip() != "0":
            raise ValueError("arm failed")
        entries = [json.loads(line) for line in (path / "RESULTS.jsonl").read_text().splitlines()]
        if entries[-1]["type"] != "audit" or entries[-1]["status"] != "PASS":
            raise ValueError("checksums failed")
        for row in entries:
            if row["type"] != "result":
                continue
            if row["passes"] != len(row["wall_ms"]) or row["passes"] < 10:
                raise ValueError("insufficient passes")
            expected_bytes = config.get("MiB", 512) * 1024**2 * (2 if "copy" in row["engine"] else 1)
            if row["bytes_per_pass"] != expected_bytes:
                raise ValueError("byte count differs")
            rate = row["passes"] * expected_bytes / row["elapsed_s"] / 1e9
            if abs(rate / row["GBps"] - 1) > 1e-6:
                raise ValueError("bandwidth arithmetic differs")
            row["median_wall_ms"] = statistics.median(row["wall_ms"])
            row["p10_GBps"] = expected_bytes / sorted(row["wall_ms"])[int(0.9 * (len(row["wall_ms"]) - 1))] / 1e6
        if arm["mode"] == "both":
            joint = next(row for row in entries if row["type"] == "joint")
            if joint["overlap_fraction"] < 0.95:
                raise ValueError("insufficient concurrent overlap")
        results.append({"arm": arm, "entries": entries})
    save(root / "RESULT.json", {"status": "PASS", "config": config, "arms": results,
         "scope": "logical streaming bytes / elapsed time; no physical DRAM counters or energy"})
    for arm in results:
        print(arm["arm"]["name"], [(r.get("engine", r["type"]), round(r["GBps"], 3))
              for r in arm["entries"] if "GBps" in r])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("root", type=Path)
    prepare_parser.add_argument("--software", type=Path, required=True)
    prepare_parser.add_argument("--config", type=Path, required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("root", type=Path)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else audit)(args)


if __name__ == "__main__":
    main()
