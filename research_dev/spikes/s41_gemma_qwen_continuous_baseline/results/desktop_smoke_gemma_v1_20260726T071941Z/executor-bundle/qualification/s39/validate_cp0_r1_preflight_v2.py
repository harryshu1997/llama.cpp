#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cp0_r1_evidence_v2 as evidence


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2.json"
DEFAULT_CANDIDATE = HERE / "CP0_R1_CANDIDATE.json"
COLLECTOR = HERE / "cp0_r1_preflight_v2.py"


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


def parse_gpu(stdout: str) -> tuple[str, list[dict[str, Any]]]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    evidence.require(bool(lines), "E_PREFLIGHT_GPU: missing hostname")
    gpus = []
    for line in lines[1:]:
        fields = [field.strip() for field in line.split(",")]
        evidence.require(len(fields) == 3, "E_PREFLIGHT_GPU: malformed row")
        evidence.require(
            re.fullmatch(r"[0-9]+", fields[2]) is not None,
            "E_PREFLIGHT_GPU: malformed memory",
        )
        gpus.append(
            {
                "memory_total_bytes": int(fields[2]) * 1024 * 1024,
                "name": fields[0],
                "uuid": fields[1],
            }
        )
    return lines[0], gpus


def parse_phone(stdout: str) -> dict[str, str]:
    lines = stdout.splitlines()
    evidence.require(len(lines) == 4, "E_PREFLIGHT_PHONE: malformed identity")
    result = {
        "boot_id": lines[3].strip(),
        "device": lines[2].strip(),
        "model": lines[0].strip(),
        "product": lines[1].strip(),
    }
    evidence.require(
        evidence.UUID_RE.fullmatch(result["boot_id"]) is not None,
        "E_PREFLIGHT_PHONE: invalid boot ID",
    )
    return result


def validate_probe_base(
    probe: Any,
    extra: set[str],
    field: str,
) -> dict[str, Any]:
    probe = evidence.exact_keys(
        probe,
        {
            "argv",
            "ended_utc",
            "returncode",
            "started_utc",
            "stderr",
            "stdout",
            "timed_out",
        }
        | extra,
        field,
    )
    evidence.require(
        type(probe["argv"]) is list
        and bool(probe["argv"])
        and all(type(item) is str and bool(item) for item in probe["argv"]),
        f"E_PREFLIGHT_ARGV: {field}",
    )
    evidence.integer(probe["returncode"], f"{field}.returncode")
    evidence.string(probe["stdout"], f"{field}.stdout", allow_empty=True)
    evidence.string(probe["stderr"], f"{field}.stderr", allow_empty=True)
    evidence.string(probe["started_utc"], f"{field}.started_utc")
    evidence.string(probe["ended_utc"], f"{field}.ended_utc")
    evidence.require(
        probe["started_utc"] <= probe["ended_utc"],
        f"E_PREFLIGHT_TIME: {field}",
    )
    evidence.require(type(probe["timed_out"]) is bool, f"E_TYPE: {field}.timed_out")
    return probe


def validate(
    result: dict[str, Any],
    result_raw: bytes,
    manifest_raw: bytes,
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate_raw: bytes,
) -> dict[str, Any]:
    evidence.exact_keys(
        result,
        {
            "candidate_sha256",
            "collector_sha256",
            "contract_sha256",
            "forbidden_work_executed",
            "probes",
            "problems",
            "schema",
            "status",
        },
        "preflight",
    )
    evidence.exact(
        result["schema"],
        "s39-cp0-r1-no-model-preflight-v2",
        "preflight.schema",
    )
    evidence.exact(
        result["candidate_sha256"],
        evidence.sha256_bytes(candidate_raw),
        "preflight.candidate",
    )
    evidence.exact(
        result["contract_sha256"],
        evidence.sha256_bytes(contract_raw),
        "preflight.contract",
    )
    collector_raw = evidence.secure_read(COLLECTOR.parent, COLLECTOR.name)
    evidence.exact(
        result["collector_sha256"],
        evidence.sha256_bytes(collector_raw),
        "preflight.collector",
    )
    evidence.exact(
        manifest_raw,
        (
            f"{evidence.sha256_bytes(result_raw)}  PREFLIGHT.json\n"
        ).encode("ascii"),
        "preflight.manifest",
    )
    evidence.exact(result["forbidden_work_executed"], False, "preflight.forbidden")
    evidence.exact(result["problems"], [], "preflight.problems")
    evidence.exact(result["status"], "NO_MODEL_PREFLIGHT_PASS", "preflight.status")

    probes = evidence.exact_keys(
        result["probes"],
        {"adb_5037", "adb_5038", "cuda", "op12", "op15"},
        "preflight.probes",
    )
    adb_devices = {}
    for port in (5037, 5038):
        field = f"preflight.probes.adb_{port}"
        probe = validate_probe_base(probes[f"adb_{port}"], {"derived_devices"}, field)
        evidence.exact(probe["argv"], ["adb", "-P", str(port), "devices", "-l"], f"{field}.argv")
        evidence.exact(probe["returncode"], 0, f"{field}.returncode")
        evidence.exact(probe["timed_out"], False, f"{field}.timed_out")
        parsed = parse_adb_devices(probe["stdout"])
        evidence.exact(probe["derived_devices"], parsed, f"{field}.derived_devices")
        adb_devices[port] = parsed

    cuda = validate_probe_base(
        probes["cuda"],
        {"derived", "identity_match"},
        "preflight.probes.cuda",
    )
    evidence.exact(cuda["returncode"], 0, "preflight.cuda.returncode")
    evidence.exact(cuda["timed_out"], False, "preflight.cuda.timed_out")
    hostname, gpus = parse_gpu(cuda["stdout"])
    derived_cuda = {"gpus": gpus, "hostname": hostname}
    evidence.exact(cuda["derived"], derived_cuda, "preflight.cuda.derived")
    expected_cuda = contract["devices"]["cuda"]
    matching = [
        gpu
        for gpu in gpus
        if gpu["name"] == expected_cuda["name"]
        and gpu["uuid"] == expected_cuda["uuid"]
        and gpu["memory_total_bytes"] == expected_cuda["memory_total_bytes"]
    ]
    evidence.require(
        hostname == expected_cuda["host"] and len(matching) == 1,
        "E_PREFLIGHT_CUDA_IDENTITY",
    )
    evidence.exact(cuda["identity_match"], True, "preflight.cuda.identity_match")

    derived_phones = {}
    for phone in ("op15", "op12"):
        field = f"preflight.probes.{phone}"
        probe = validate_probe_base(
            probes[phone],
            {
                "adb_port",
                "derived",
                "identity_match",
                "serial",
                "transport_tags",
            },
            field,
        )
        expected = contract["devices"][phone]
        evidence.exact(probe["serial"], expected["serial"], f"{field}.serial")
        locations = [
            port
            for port, devices in adb_devices.items()
            if expected["serial"] in devices
        ]
        evidence.exact(len(locations), 1, f"{field}.locations")
        evidence.exact(probe["adb_port"], locations[0], f"{field}.adb_port")
        tags = adb_devices[locations[0]][expected["serial"]]
        evidence.exact(probe["transport_tags"], tags, f"{field}.transport_tags")
        for key in ("device", "model", "product"):
            evidence.exact(tags.get(key), expected[key], f"{field}.tags.{key}")
        evidence.exact(probe["returncode"], 0, f"{field}.returncode")
        evidence.exact(probe["timed_out"], False, f"{field}.timed_out")
        parsed = parse_phone(probe["stdout"])
        evidence.exact(probe["derived"], parsed, f"{field}.derived")
        for key in ("device", "model", "product"):
            evidence.exact(parsed[key], expected[key], f"{field}.{key}")
        evidence.exact(probe["identity_match"], True, f"{field}.identity_match")
        derived_phones[phone] = {
            "adb_port": locations[0],
            "boot_id": parsed["boot_id"],
            "serial": expected["serial"],
        }

    return {
        "artifact_sha256": evidence.sha256_bytes(result_raw),
        "candidate_sha256": evidence.sha256_bytes(candidate_raw),
        "contract_sha256": evidence.sha256_bytes(contract_raw),
        "devices": {
            "cuda": {
                "host": hostname,
                "memory_total_bytes": matching[0]["memory_total_bytes"],
                "name": matching[0]["name"],
                "uuid": matching[0]["uuid"],
            },
            **derived_phones,
        },
        "schema": "s39-cp0-r1-no-model-preflight-validation-v2",
        "status": "NO_MODEL_PREFLIGHT_VALIDATED",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently validate a persisted CP0-R1 v2 preflight"
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract, contract_raw = evidence.load_canonical_path(args.contract)
        candidate, candidate_raw = evidence.load_canonical_path(args.candidate)
        evidence.validate_contract(contract)
        evidence.validate_candidate(candidate, candidate_raw, contract)
        result_raw = evidence.secure_read(args.run_dir, "PREFLIGHT.json")
        result = evidence.parse_json(result_raw, "PREFLIGHT.json")
        evidence.require(
            evidence.canonical_bytes(result) == result_raw,
            "E_CANONICAL: PREFLIGHT.json",
        )
        manifest_raw = evidence.secure_read(args.run_dir, "SHA256SUMS.txt")
        validated = validate(
            result,
            result_raw,
            manifest_raw,
            contract,
            contract_raw,
            candidate_raw,
        )
        print(evidence.canonical_bytes(validated).decode("ascii"), end="")
        return 0
    except (evidence.EvidenceError, OSError, KeyError) as exc:
        print(f"CP0_R1_PREFLIGHT_VALIDATION_REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
