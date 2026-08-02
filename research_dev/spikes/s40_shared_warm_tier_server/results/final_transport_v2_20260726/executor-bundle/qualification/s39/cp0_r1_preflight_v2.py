#!/usr/bin/env python3

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import cp0_r1_evidence_v2 as evidence


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2.json"
DEFAULT_CANDIDATE = HERE / "CP0_R1_CANDIDATE.json"


def run_probe(argv: list[str], timeout_s: int) -> dict[str, Any]:
    started = datetime.datetime.now(datetime.timezone.utc)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_s,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="backslashreplace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="backslashreplace")
        timed_out = True
    ended = datetime.datetime.now(datetime.timezone.utc)
    return {
        "argv": argv,
        "ended_utc": ended.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "returncode": returncode,
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "stderr": stderr,
        "stdout": stdout,
        "timed_out": timed_out,
    }


def parse_adb_devices(stdout: str) -> dict[str, dict[str, str]]:
    result = {}
    for line in stdout.splitlines():
        if not line or line.startswith("List of devices attached"):
            continue
        fields = line.split()
        if len(fields) < 2 or fields[1] != "device":
            continue
        tags = {}
        for field in fields[2:]:
            if ":" in field:
                key, value = field.split(":", 1)
                tags[key] = value
        result[fields[0]] = tags
    return result


def parse_gpu(stdout: str, contract: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return False, {}
    host = lines[0]
    matches = []
    for line in lines[1:]:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not re.fullmatch(r"[0-9]+", fields[2]):
            continue
        matches.append(
            {
                "memory_total_bytes": int(fields[2]) * 1024 * 1024,
                "name": fields[0],
                "uuid": fields[1],
            }
        )
    expected = contract["devices"]["cuda"]
    exact = [
        item
        for item in matches
        if item["name"] == expected["name"]
        and item["uuid"] == expected["uuid"]
        and item["memory_total_bytes"] == expected["memory_total_bytes"]
    ]
    return host == expected["host"] and len(exact) == 1, {
        "gpus": matches,
        "hostname": host,
    }


def parse_phone_identity(stdout: str) -> dict[str, str]:
    lines = stdout.splitlines()
    if len(lines) != 4:
        return {}
    return {
        "boot_id": lines[3].strip(),
        "device": lines[2].strip(),
        "model": lines[0].strip(),
        "product": lines[1].strip(),
    }


def write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    fd = os.open(path, flags, 0o644)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire(
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate_raw: bytes,
    ssh_target: str,
) -> dict[str, Any]:
    probes = {}
    problems = []
    gpu_probe = run_probe(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            ssh_target,
            (
                "hostname; "
                "nvidia-smi --query-gpu=name,uuid,memory.total "
                "--format=csv,noheader,nounits"
            ),
        ],
        12,
    )
    gpu_match, gpu_parsed = parse_gpu(gpu_probe["stdout"], contract)
    gpu_probe["derived"] = gpu_parsed
    gpu_probe["identity_match"] = gpu_probe["returncode"] == 0 and gpu_match
    probes["cuda"] = gpu_probe
    if not gpu_probe["identity_match"]:
        problems.append("E_CUDA_IDENTITY")

    devices_by_port = {}
    for port in (5037, 5038):
        probe = run_probe(["adb", "-P", str(port), "devices", "-l"], 8)
        parsed = parse_adb_devices(probe["stdout"])
        probe["derived_devices"] = parsed
        probes[f"adb_{port}"] = probe
        devices_by_port[port] = parsed if probe["returncode"] == 0 else {}
        if probe["returncode"] != 0:
            problems.append(f"E_ADB_{port}")

    for phone in ("op15", "op12"):
        expected = contract["devices"][phone]
        serial = expected["serial"]
        locations = [
            port for port, devices in devices_by_port.items() if serial in devices
        ]
        if len(locations) != 1:
            problems.append(f"E_{phone.upper()}_PRESENCE")
            probes[phone] = {
                "identity_match": False,
                "locations": locations,
                "serial": serial,
            }
            continue
        port = locations[0]
        tags = devices_by_port[port][serial]
        tag_match = all(
            tags.get(field) == expected[field]
            for field in ("device", "model", "product")
        )
        identity_probe = run_probe(
            [
                "adb",
                "-P",
                str(port),
                "-s",
                serial,
                "shell",
                (
                    "getprop ro.product.model; "
                    "getprop ro.product.name; "
                    "getprop ro.product.device; "
                    "cat /proc/sys/kernel/random/boot_id"
                ),
            ],
            8,
        )
        parsed = parse_phone_identity(identity_probe["stdout"])
        shell_match = (
            identity_probe["returncode"] == 0
            and parsed.get("model") == expected["model"]
            and parsed.get("product") == expected["product"]
            and parsed.get("device") == expected["device"]
            and evidence.UUID_RE.fullmatch(parsed.get("boot_id", "")) is not None
        )
        identity_probe["adb_port"] = port
        identity_probe["derived"] = parsed
        identity_probe["identity_match"] = tag_match and shell_match
        identity_probe["serial"] = serial
        identity_probe["transport_tags"] = tags
        probes[phone] = identity_probe
        if not identity_probe["identity_match"]:
            problems.append(f"E_{phone.upper()}_IDENTITY")

    return {
        "candidate_sha256": evidence.sha256_bytes(candidate_raw),
        "collector_sha256": sha256_file(Path(__file__)),
        "contract_sha256": evidence.sha256_bytes(contract_raw),
        "forbidden_work_executed": False,
        "probes": probes,
        "problems": sorted(set(problems)),
        "schema": "s39-cp0-r1-no-model-preflight-v2",
        "status": "NO_MODEL_PREFLIGHT_PASS" if not problems else "NO_MODEL_PREFLIGHT_FAIL",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CP0-R1 v2 identity preflight")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--ssh-target", default="zhihao@172.20.74.85")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract, contract_raw = evidence.load_canonical_path(args.contract)
        candidate, candidate_raw = evidence.load_canonical_path(args.candidate)
        evidence.validate_contract(contract)
        evidence.validate_candidate(candidate, candidate_raw, contract)
        args.output.mkdir(parents=True, exist_ok=False)
        result = acquire(contract, contract_raw, candidate_raw, args.ssh_target)
        result_path = args.output / "PREFLIGHT.json"
        write_exclusive(result_path, evidence.canonical_bytes(result))
        manifest = f"{sha256_file(result_path)}  PREFLIGHT.json\n".encode("ascii")
        write_exclusive(args.output / "SHA256SUMS.txt", manifest)
        print(evidence.canonical_bytes(result).decode("ascii"), end="")
        return 0 if result["status"] == "NO_MODEL_PREFLIGHT_PASS" else 2
    except (evidence.EvidenceError, OSError, KeyError) as exc:
        print(f"CP0_R1_PREFLIGHT_REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
