#!/usr/bin/env python3

import argparse
import copy
import shlex
from pathlib import Path
from typing import Any

import build_cp0_r1_v21 as builder
import cp0_r1_evidence_v2 as v2


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2_1.json"
DEFAULT_CANDIDATE = HERE / "CP0_R1_CANDIDATE.json"
PARENT_CONTRACT = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2.json"

PHASE_SLOT = {"A_ONLY": "A", "B_ONLY": "B"}
PRE_ACQUISITION_ROLES = {
    "phase.lock",
    "phase.preflight",
    "quality.corpus",
}


def validate_inputs(
    contract_path: Path,
    candidate_path: Path,
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, Any],
]:
    contract, contract_raw = v2.load_canonical_path(contract_path)
    v2.exact(contract, builder.build_contract(), "contract")
    candidate, candidate_raw = v2.load_canonical_path(candidate_path)
    parent, parent_raw = v2.load_canonical_path(PARENT_CONTRACT)
    v2.exact(
        v2.sha256_bytes(parent_raw),
        contract["parent"]["contract_sha256"],
        "contract.parent.contract_sha256",
    )
    v2.validate_contract(parent)
    v2.validate_candidate(candidate, candidate_raw, parent)
    return contract, contract_raw, candidate, candidate_raw, parent


def phase_roles(
    contract: dict[str, Any],
    phase: str,
) -> list[str]:
    v2.require(
        phase in contract["phase_protocol"]["phase_roles"],
        f"E_PHASE: {phase}",
    )
    return contract["phase_protocol"]["phase_roles"][phase]


def _parse_phase_jsonl(
    raw: bytes,
    role: str,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    v2.require(raw.endswith(b"\n"), f"E_CANONICAL: {role}: missing final newline")
    rows = []
    previous_event = None
    for index, line in enumerate(raw.splitlines(keepends=True)):
        v2.require(line != b"\n", f"E_EMPTY: {role}[{index}]")
        row = v2.parse_json(line, f"{role}[{index}]")
        v2.require(type(row) is dict, f"E_TYPE: {role}[{index}]")
        v2.require(v2.canonical_line(row) == line, f"E_CANONICAL: {role}[{index}]")
        field = f"{role}[{index}]"
        v2.exact(row.get("role"), role, f"{field}.role")
        v2.exact(
            row.get("acquisition_id"),
            manifest["phase_id"],
            f"{field}.acquisition_id",
        )
        v2.exact(row.get("phase_id"), manifest["phase_id"], f"{field}.phase_id")
        v2.exact(row.get("phase"), manifest["phase"], f"{field}.phase")
        event_ns = v2.integer(row.get("event_ns"), f"{field}.event_ns", 1)
        v2.require(
            manifest["phase_opened_ns"]
            <= event_ns
            <= manifest["phase_closed_ns"],
            f"E_PHASE_INTERVAL: {field}",
        )
        if previous_event is not None:
            v2.require(previous_event <= event_ns, f"E_EVENT_ORDER: {role}")
        previous_event = event_ns
        if (
            role in PRE_ACQUISITION_ROLES
            or role.endswith(".route_lock")
        ):
            v2.require(
                event_ns < manifest["acquisition_started_ns"],
                f"E_PROSPECTIVE: {field}",
            )
        else:
            v2.require(
                event_ns >= manifest["acquisition_started_ns"],
                f"E_ACQUISITION_ORDER: {field}",
            )
        rows.append(row)
    v2.require(bool(rows), f"E_EMPTY: {role}")
    return rows


def load_bundle(
    root: Path,
    manifest_name: str,
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate_raw: bytes,
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, list[dict[str, Any]]],
    dict[str, str],
]:
    manifest_raw = v2.secure_read(root, manifest_name)
    manifest = v2.parse_json(manifest_raw, manifest_name)
    v2.require(type(manifest) is dict, "E_TYPE: bundle")
    v2.require(v2.canonical_bytes(manifest) == manifest_raw, "E_CANONICAL: bundle")
    v2.exact_keys(
        manifest,
        {
            "acquisition_started_ns",
            "artifacts",
            "candidate_sha256",
            "clock_id",
            "contract_sha256",
            "phase",
            "phase_closed_ns",
            "phase_id",
            "phase_opened_ns",
            "schema",
        },
        "bundle",
    )
    v2.exact(manifest["schema"], "s39-cp0-r1-evidence-bundle-v2.1", "bundle.schema")
    phase = v2.string(manifest["phase"], "bundle.phase")
    phase_id = v2.string(manifest["phase_id"], "bundle.phase_id")
    v2.exact(
        manifest["clock_id"],
        contract["phase_protocol"]["clock_id"],
        "bundle.clock_id",
    )
    opened = v2.integer(manifest["phase_opened_ns"], "bundle.phase_opened_ns", 1)
    started = v2.integer(
        manifest["acquisition_started_ns"],
        "bundle.acquisition_started_ns",
        1,
    )
    closed = v2.integer(manifest["phase_closed_ns"], "bundle.phase_closed_ns", 1)
    v2.require(opened < started <= closed, "E_PHASE_INTERVAL: bundle")
    v2.exact(
        manifest["contract_sha256"],
        v2.sha256_bytes(contract_raw),
        "bundle.contract_sha256",
    )
    v2.exact(
        manifest["candidate_sha256"],
        v2.sha256_bytes(candidate_raw),
        "bundle.candidate_sha256",
    )
    expected_roles = phase_roles(contract, phase)
    artifacts = manifest["artifacts"]
    v2.require(
        type(artifacts) is list and len(artifacts) == len(expected_roles),
        "E_ROLE_SET: artifact count",
    )
    by_role = {}
    paths = set()
    digests = set()
    for index, artifact in enumerate(artifacts):
        field = f"bundle.artifacts[{index}]"
        v2.exact_keys(
            artifact,
            {"bytes", "format", "path", "role", "sha256"},
            field,
        )
        role = v2.string(artifact["role"], f"{field}.role")
        v2.require(role not in by_role, f"E_ROLE_REUSE: {role}")
        path = v2.string(artifact["path"], f"{field}.path")
        v2._normalize_relative(path)
        v2.require(path != manifest_name, f"E_PATH: manifest cannot be evidence")
        v2.require(path not in paths, f"E_PATH_REUSE: {path}")
        declared_digest = v2.digest(artifact["sha256"], f"{field}.sha256")
        v2.require(declared_digest not in digests, f"E_DIGEST_REUSE: {role}")
        v2.exact(
            artifact["format"],
            "CANONICAL_ASCII_JSONL",
            f"{field}.format",
        )
        size = v2.integer(artifact["bytes"], f"{field}.bytes", 1)
        v2.require(size <= 67_108_864, f"E_SIZE: {role}")
        by_role[role] = artifact
        paths.add(path)
        digests.add(declared_digest)
    v2.exact(sorted(by_role), sorted(expected_roles), "bundle.role_set")

    rows_by_role = {}
    artifact_digests = {}
    for role in expected_roles:
        artifact = by_role[role]
        raw = v2.secure_read(root, artifact["path"])
        v2.exact(len(raw), artifact["bytes"], f"{role}.bytes")
        v2.exact(v2.sha256_bytes(raw), artifact["sha256"], f"{role}.sha256")
        rows_by_role[role] = _parse_phase_jsonl(raw, role, manifest)
        artifact_digests[role] = artifact["sha256"]
    v2.exact(phase_id, rows_by_role["phase.lock"][0]["acquisition_id"], "phase_id")
    return manifest, manifest_raw, rows_by_role, artifact_digests


def _normalize_rows(
    rows: list[dict[str, Any]],
    extras_by_kind: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    extras_by_kind = extras_by_kind or {}
    result = []
    for row in rows:
        normalized = copy.deepcopy(row)
        for key in ("event_ns", "phase", "phase_id"):
            normalized.pop(key)
        for key in extras_by_kind.get(normalized["kind"], set()):
            normalized.pop(key)
        result.append(normalized)
    return result


def _model_by_slot(candidate: dict[str, Any], slot: str) -> dict[str, Any]:
    matches = [model for model in candidate["models"] if model["slot"] == slot]
    v2.require(len(matches) == 1, f"E_MODEL_SLOT: {slot}")
    return matches[0]


def _phase_result_sha256(result: dict[str, Any]) -> str:
    return v2.sha256_bytes(v2.canonical_bytes(result))


def validate_phase_lock(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    artifact_digests: dict[str, str],
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate_raw: bytes,
    prior_result_sha256s: list[str],
) -> dict[str, Any]:
    v2.require(len(rows) == 1, "E_ROWS: phase.lock")
    row = v2.exact_keys(
        rows[0],
        {
            "acquisition_id",
            "candidate_sha256",
            "clock_id",
            "contract_sha256",
            "event_ns",
            "kind",
            "model_slot",
            "phase",
            "phase_id",
            "prior_phase_result_sha256s",
            "quality_corpus_sha256",
            "role",
            "route_lock_sha256",
        },
        "phase.lock",
    )
    v2.exact(row["kind"], "phase_lock", "phase.lock.kind")
    v2.exact(row["clock_id"], manifest["clock_id"], "phase.lock.clock_id")
    v2.exact(
        row["candidate_sha256"],
        v2.sha256_bytes(candidate_raw),
        "phase.lock.candidate_sha256",
    )
    v2.exact(
        row["contract_sha256"],
        v2.sha256_bytes(contract_raw),
        "phase.lock.contract_sha256",
    )
    v2.exact(
        row["prior_phase_result_sha256s"],
        prior_result_sha256s,
        "phase.lock.prior_phase_result_sha256s",
    )
    phase = manifest["phase"]
    if phase in PHASE_SLOT:
        slot = PHASE_SLOT[phase]
        model_id = contract["candidate_lock"]["models"][0 if slot == "A" else 1][
            "model_id"
        ]
        v2.exact(row["model_slot"], slot, "phase.lock.model_slot")
        v2.exact(
            row["quality_corpus_sha256"],
            artifact_digests["quality.corpus"],
            "phase.lock.quality_corpus_sha256",
        )
        v2.exact(
            row["route_lock_sha256"],
            artifact_digests[f"model.{model_id}.route_lock"],
            "phase.lock.route_lock_sha256",
        )
    else:
        v2.exact(row["model_slot"], "PAIR", "phase.lock.model_slot")
        v2.exact(row["quality_corpus_sha256"], "NONE", "phase.lock.corpus")
        v2.exact(row["route_lock_sha256"], "NONE", "phase.lock.route")
    return row


def validate_route_lock(
    rows: list[dict[str, Any]],
    role: str,
    manifest: dict[str, Any],
    model: dict[str, Any],
    contract: dict[str, Any],
    parent: dict[str, Any],
) -> dict[str, Any]:
    extras = {
        "activation_dtype",
        "activation_element_bytes",
        "cuda_model_path",
        "hidden_size",
        "op12_shard_bytes",
        "op12_shard_path",
        "op15_shard_bytes",
        "op15_shard_path",
    }
    v2.require(len(rows) == 1, f"E_ROWS: {role}")
    row = rows[0]
    geometry = contract["model_geometry"][model["model_id"]]
    v2.exact(row["event_ns"], row["frozen_ns"], f"{role}.event_ns")
    v2.exact(row["activation_dtype"], geometry["activation_dtype"], f"{role}.dtype")
    v2.exact(
        row["activation_element_bytes"],
        geometry["activation_element_bytes"],
        f"{role}.element_bytes",
    )
    v2.exact(row["hidden_size"], geometry["hidden_size"], f"{role}.hidden_size")
    v2.exact(
        row["cuda_model_path"],
        geometry["cuda_model_path"],
        f"{role}.cuda_model_path",
    )
    for phone in ("op15", "op12"):
        v2.integer(row[f"{phone}_shard_bytes"], f"{role}.{phone}.shard_bytes", 1)
        path = v2.string(row[f"{phone}_shard_path"], f"{role}.{phone}.shard_path")
        v2.require(path.startswith("/data/local/tmp/"), f"E_PATH: {role}.{phone}")
    known = geometry.get("known_shards")
    if known is not None:
        for phone in ("op15", "op12"):
            v2.exact(
                row[f"{phone}_shard_bytes"],
                known[phone]["bytes"],
                f"{role}.{phone}.known_bytes",
            )
            v2.exact(
                row[f"{phone}_shard_path"],
                known[phone]["path"],
                f"{role}.{phone}.known_path",
            )
            v2.exact(
                row[f"{phone}_shard_sha256"],
                known[phone]["sha256"],
                f"{role}.{phone}.known_sha256",
            )
    normalized = _normalize_rows(rows, {"route_lock": extras})
    base = v2.validate_route_lock(
        normalized,
        role,
        manifest["phase_id"],
        manifest["acquisition_started_ns"],
        manifest["clock_id"],
        model,
        parent,
    )
    return {**base, **{key: row[key] for key in extras}}


def _probe_commands(
    contract: dict[str, Any],
    model: dict[str, Any],
    lock: dict[str, Any],
) -> dict[str, list[str]]:
    device = contract["devices"]["cuda"]
    model_path = shlex.quote(lock["cuda_model_path"])
    cuda_script = (
        "set -eu; hostname; "
        f"nvidia-smi --id={shlex.quote(device['uuid'])} "
        "--query-gpu=name,uuid,memory.total --format=csv,noheader,nounits; "
        f"printf 'MODEL_BYTES='; stat -c %s {model_path}; "
        f"printf 'MODEL_SHA256='; sha256sum {model_path} | cut -d' ' -f1"
    )
    commands = {
        "adb_5037": ["adb", "-P", "5037", "devices", "-l"],
        "adb_5038": ["adb", "-P", "5038", "devices", "-l"],
        f"cuda_{model['slot']}": [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            contract["preflight"]["ssh_target"],
            cuda_script,
        ],
    }
    for phone in ("op15", "op12"):
        expected = contract["devices"][phone]
        shard_path = shlex.quote(lock[f"{phone}_shard_path"])
        script = (
            "set -eu; getprop ro.product.model; getprop ro.product.name; "
            "getprop ro.product.device; cat /proc/sys/kernel/random/boot_id; "
            f"printf 'SHARD_BYTES='; stat -c %s {shard_path}; "
            f"printf 'SHARD_SHA256='; sha256sum {shard_path} | cut -d' ' -f1"
        )
        commands[f"{phone}_{model['slot']}"] = [
            "adb",
            "-P",
            str(contract["preflight"]["phone_adb_port"]),
            "-s",
            expected["serial"],
            "shell",
            script,
        ]
    return commands


def expected_preflight_commands(
    contract: dict[str, Any],
    model_locks: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, list[str]]:
    commands = {
        "adb_5037": ["adb", "-P", "5037", "devices", "-l"],
        "adb_5038": ["adb", "-P", "5038", "devices", "-l"],
    }
    for model, lock in model_locks:
        commands.update(_probe_commands(contract, model, lock))
    return commands


def _parse_adb(stdout: str) -> dict[str, dict[str, str]]:
    result = {}
    for line in stdout.splitlines():
        if not line or line.startswith("List of devices attached"):
            continue
        fields = line.split()
        if len(fields) < 2 or fields[1] != "device":
            continue
        tags = {}
        for item in fields[2:]:
            if ":" in item:
                key, value = item.split(":", 1)
                tags[key] = value
        result[fields[0]] = tags
    return result


def _parse_cuda(stdout: str, field: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    v2.require(len(lines) == 4, f"E_PREFLIGHT_OUTPUT: {field}")
    fields = [item.strip() for item in lines[1].split(",")]
    v2.require(len(fields) == 3 and fields[2].isdigit(), f"E_PREFLIGHT_GPU: {field}")
    v2.require(lines[2].startswith("MODEL_BYTES="), f"E_PREFLIGHT_MODEL: {field}")
    v2.require(lines[3].startswith("MODEL_SHA256="), f"E_PREFLIGHT_MODEL: {field}")
    return {
        "host": lines[0],
        "memory_total_bytes": int(fields[2]) * 1024 * 1024,
        "model_bytes": int(lines[2].split("=", 1)[1]),
        "model_sha256": lines[3].split("=", 1)[1],
        "name": fields[0],
        "uuid": fields[1],
    }


def _parse_phone(stdout: str, field: str) -> dict[str, Any]:
    lines = stdout.splitlines()
    v2.require(len(lines) == 6, f"E_PREFLIGHT_OUTPUT: {field}")
    v2.require(lines[4].startswith("SHARD_BYTES="), f"E_PREFLIGHT_SHARD: {field}")
    v2.require(lines[5].startswith("SHARD_SHA256="), f"E_PREFLIGHT_SHARD: {field}")
    return {
        "boot_id": lines[3],
        "device": lines[2],
        "model": lines[0],
        "product": lines[1],
        "shard_bytes": int(lines[4].split("=", 1)[1]),
        "shard_sha256": lines[5].split("=", 1)[1],
    }


def validate_preflight(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    contract: dict[str, Any],
    model_locks: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    commands = expected_preflight_commands(contract, model_locks)
    v2.require(len(rows) == len(commands) + 1, "E_ROWS: phase.preflight")
    probes = {}
    for index, row in enumerate(rows[:-1]):
        field = f"phase.preflight[{index}]"
        row = v2.exact_keys(
            row,
            {
                "acquisition_id",
                "argv",
                "event_ns",
                "kind",
                "label",
                "phase",
                "phase_id",
                "returncode",
                "role",
                "started_ns",
                "stderr",
                "stdout",
                "timed_out",
            },
            field,
        )
        v2.exact(row["kind"], "probe", f"{field}.kind")
        label = v2.string(row["label"], f"{field}.label")
        v2.require(label not in probes, f"E_PREFLIGHT_REUSE: {label}")
        v2.require(label in commands, f"E_PREFLIGHT_LABEL: {label}")
        v2.exact(row["argv"], commands[label], f"E_PREFLIGHT_ARGV: {label}")
        v2.exact(row["returncode"], 0, f"E_PREFLIGHT_EXIT: {label}")
        v2.exact(row["timed_out"], False, f"E_PREFLIGHT_TIMEOUT: {label}")
        v2.exact(row["stderr"], "", f"E_PREFLIGHT_STDERR: {label}")
        v2.string(row["stdout"], f"{field}.stdout", allow_empty=True)
        started_ns = v2.integer(row["started_ns"], f"{field}.started_ns", 1)
        v2.require(
            manifest["phase_opened_ns"] <= started_ns <= row["event_ns"],
            f"E_PHASE_INTERVAL: {field}.started_ns",
        )
        probes[label] = row
    v2.exact(sorted(probes), sorted(commands), "phase.preflight.labels")
    meta = v2.exact_keys(
        rows[-1],
        {
            "acquisition_id",
            "completed_ns",
            "event_ns",
            "forbidden_work_executed",
            "kind",
            "phase",
            "phase_id",
            "probe_labels",
            "role",
        },
        "phase.preflight.meta",
    )
    v2.exact(meta["kind"], "meta", "phase.preflight.meta.kind")
    v2.exact(meta["forbidden_work_executed"], False, "preflight.forbidden")
    v2.exact(meta["probe_labels"], sorted(commands), "preflight.probe_labels")
    v2.exact(meta["completed_ns"], meta["event_ns"], "preflight.completed_ns")
    v2.exact(
        meta["completed_ns"],
        max(row["event_ns"] for row in rows),
        "preflight.completed_max",
    )
    v2.require(
        meta["completed_ns"] < manifest["acquisition_started_ns"],
        "E_PREFLIGHT_ORDER",
    )
    v2.require(
        manifest["acquisition_started_ns"] - meta["completed_ns"]
        <= contract["gates"]["phase_preflight_maximum_age_ns"],
        "E_PREFLIGHT_STALE",
    )

    adb = {
        port: _parse_adb(probes[f"adb_{port}"]["stdout"])
        for port in contract["preflight"]["adb_ports"]
    }
    for phone in ("op15", "op12"):
        expected = contract["devices"][phone]
        locations = [
            port for port, devices in adb.items() if expected["serial"] in devices
        ]
        v2.exact(
            locations,
            [contract["preflight"]["phone_adb_port"]],
            f"E_PREFLIGHT_LOCATION: {phone}",
        )
        tags = adb[locations[0]][expected["serial"]]
        for key in ("device", "model", "product"):
            v2.exact(tags.get(key), expected[key], f"E_PREFLIGHT_TAG: {phone}.{key}")

    for model, lock in model_locks:
        slot = model["slot"]
        cuda = _parse_cuda(probes[f"cuda_{slot}"]["stdout"], f"cuda_{slot}")
        expected_cuda = contract["devices"]["cuda"]
        for key in ("host", "memory_total_bytes", "name", "uuid"):
            v2.exact(cuda[key], expected_cuda[key], f"E_PREFLIGHT_CUDA: {slot}.{key}")
        v2.exact(cuda["model_bytes"], model["artifact"]["bytes"], f"cuda_{slot}.bytes")
        v2.exact(
            cuda["model_sha256"],
            model["artifact"]["sha256"],
            f"cuda_{slot}.sha256",
        )
        for phone in ("op15", "op12"):
            parsed = _parse_phone(
                probes[f"{phone}_{slot}"]["stdout"],
                f"{phone}_{slot}",
            )
            expected_phone = contract["devices"][phone]
            for key in ("device", "model", "product"):
                v2.exact(
                    parsed[key],
                    expected_phone[key],
                    f"E_PREFLIGHT_PHONE: {phone}.{key}",
                )
            v2.require(
                v2.UUID_RE.fullmatch(parsed["boot_id"]) is not None,
                f"E_PREFLIGHT_BOOT: {phone}",
            )
            v2.exact(
                parsed["shard_bytes"],
                lock[f"{phone}_shard_bytes"],
                f"E_PREFLIGHT_SHARD: {phone}.bytes",
            )
            v2.exact(
                parsed["shard_sha256"],
                lock[f"{phone}_shard_sha256"],
                f"E_PREFLIGHT_SHARD: {phone}.sha256",
            )
    return {"completed_ns": meta["completed_ns"], "probe_count": len(probes)}


def _validate_execution_geometry(
    rows_by_role: dict[str, list[dict[str, Any]]],
    model: dict[str, Any],
    phase_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prefix = f"model.{model['model_id']}"
    roles = [
        (f"{prefix}.mechanics.phone", "PHONE_COLLECTIVE"),
        (f"{prefix}.oracle.cuda_route", "CUDA0"),
        (f"{prefix}.oracle.cuda_monolithic", "CUDA0"),
    ]
    executions = []
    for role, backend in roles:
        normalized = _normalize_rows(rows_by_role[role])
        execution = v2.validate_execution(
            normalized,
            role,
            phase_id,
            model,
            backend,
        )
        calls = execution["call_shapes"]
        v2.require(
            all(call["n_seqs"] == 8 for call in calls),
            f"E_B8_GEOMETRY: {role}",
        )
        prefill = sum(call["n_tokens"] for call in calls if call["phase"] == "prefill")
        decode = sum(call["n_tokens"] for call in calls if call["phase"] == "decode")
        expected_prefill = sum(
            len(request["input_tokens"]) for request in execution["requests"].values()
        )
        expected_decode = sum(
            len(request["continuation_tokens"])
            for request in execution["requests"].values()
        )
        v2.exact(prefill, expected_prefill, f"E_B8_PREFILL_ROWS: {role}")
        v2.exact(decode, expected_decode, f"E_B8_DECODE_ROWS: {role}")
        v2.require(
            all(
                call["n_tokens"] == 8
                for call in calls
                if call["phase"] == "decode"
            ),
            f"E_B8_DECODE_CALL: {role}",
        )
        executions.append(execution)
    phone, tested, oracle = executions
    for request_id in range(8):
        v2.exact(
            len(phone["requests"][request_id]["continuation_tokens"]),
            len(tested["requests"][request_id]["continuation_tokens"]),
            f"E_ORACLE_LENGTH: {model['model_id']}[{request_id}]",
        )
        v2.exact(
            len(tested["requests"][request_id]["continuation_tokens"]),
            len(oracle["requests"][request_id]["continuation_tokens"]),
            f"E_ORACLE_LENGTH: {model['model_id']}[{request_id}]",
        )
    return phone, tested, oracle


def _derive_memory(
    rows: list[dict[str, Any]],
    role: str,
    phase_id: str,
    model: dict[str, Any],
    parent: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    extras = {"process_pid", "process_used_bytes", "sample_id", "sampler_sha256"}
    v2.require(len(rows) == 3, f"E_ROWS: {role}")
    sampler = None
    normalized = _normalize_rows(
        rows,
        {kind: extras for kind in ("before", "ready", "after")},
    )
    for index, (row, base) in enumerate(zip(rows, normalized)):
        field = f"{role}[{index}]"
        v2.exact(row["event_ns"], row["timestamp_ns"], f"{field}.event")
        v2.exact(
            row["used_bytes"] + row["free_bytes"],
            row["memory_total_bytes"],
            f"E_MEMORY_EXACT: {field}",
        )
        v2.string(row["sample_id"], f"{field}.sample_id")
        current_sampler = v2.digest(row["sampler_sha256"], f"{field}.sampler_sha256")
        sampler = current_sampler if sampler is None else sampler
        v2.exact(current_sampler, sampler, f"{field}.sampler")
        pid = v2.integer(row["process_pid"], f"{field}.process_pid")
        process_used = v2.integer(
            row["process_used_bytes"],
            f"{field}.process_used_bytes",
        )
        if row["kind"] == "ready":
            v2.require(pid > 0, f"E_PROCESS: {field}")
            v2.require(
                row["model_buffer_bytes"] + row["kv_buffer_bytes"]
                <= process_used
                <= row["used_bytes"],
                f"E_PROCESS_MEMORY: {field}",
            )
        else:
            v2.exact(pid, 0, f"{field}.process_pid")
            v2.exact(process_used, 0, f"{field}.process_used_bytes")
        del base
    allocation = v2.derive_cuda_memory(
        normalized,
        role,
        phase_id,
        model,
        parent,
    )
    return allocation, normalized[1]


def _derive_quality(
    rows_by_role: dict[str, list[dict[str, Any]]],
    artifact_digests: dict[str, str],
    model: dict[str, Any],
    phase_id: str,
    task_suite: dict[str, Any],
    parent: dict[str, Any],
) -> tuple[dict[str, int], dict[int, dict[str, Any]]]:
    corpus_role = "quality.corpus"
    corpus_rows = _normalize_rows(rows_by_role[corpus_role])
    corpus = v2.derive_corpus(
        corpus_rows,
        corpus_role,
        phase_id,
        task_suite,
        parent,
    )
    corpus_sha = artifact_digests[corpus_role]
    normalized_outputs = {}
    extras = {"corpus_item_sha256", "corpus_sha256"}
    for suffix in ("cuda", "phone"):
        role = f"model.{model['model_id']}.quality.{suffix}"
        normalized = _normalize_rows(rows_by_role[role], {"output": extras})
        for index, row in enumerate(rows_by_role[role]):
            v2.exact(row["corpus_sha256"], corpus_sha, f"{role}[{index}].corpus")
            v2.exact(
                row["corpus_item_sha256"],
                v2.digest_json(corpus_rows[index]),
                f"{role}[{index}].corpus_item",
            )
        normalized_outputs[suffix] = normalized
    quality = v2.derive_quality(
        normalized_outputs["cuda"],
        normalized_outputs["phone"],
        f"model.{model['model_id']}.quality.cuda",
        f"model.{model['model_id']}.quality.phone",
        phase_id,
        model,
        corpus,
        parent,
    )
    return quality, corpus


def _derive_bridge(
    rows: list[dict[str, Any]],
    role: str,
    phase_id: str,
    model: dict[str, Any],
    phone_execution: dict[str, Any],
    phone_rows: list[dict[str, Any]],
    cuda_ready: dict[str, Any],
    clock_id: str,
    parent: dict[str, Any],
) -> dict[str, int]:
    extras = {
        "phone_publication_received": {"phone_request_sha256"},
        "cuda_ready": {"cuda_memory_ready_sha256"},
    }
    normalized = _normalize_rows(rows, extras)
    normalized_phone_rows = _normalize_rows(phone_rows)
    phone_request_rows = {
        row["request_id"]: row
        for row in normalized_phone_rows
        if row["kind"] == "request"
    }
    publications = {}
    for raw in rows:
        if raw["kind"] == "phone_publication_received":
            request_id = raw["request_id"]
            v2.require(request_id not in publications, f"E_BRIDGE_REQUEST_REUSE: {role}")
            request = phone_execution["requests"][request_id]
            v2.exact(
                raw["phone_request_sha256"],
                v2.digest_json(phone_request_rows[request_id]),
                f"E_BRIDGE_LINK: {role}[{request_id}]",
            )
            v2.exact(
                raw["token_ids"],
                request["continuation_tokens"],
                f"E_BRIDGE_TOKENS: {role}[{request_id}]",
            )
            publications[request_id] = raw
        elif raw["kind"] == "cuda_ready":
            v2.exact(
                raw["cuda_memory_ready_sha256"],
                v2.digest_json(cuda_ready),
                f"E_BRIDGE_READY_LINK: {role}",
            )
            v2.exact(
                raw["timestamp_ns"],
                cuda_ready["timestamp_ns"],
                f"E_BRIDGE_READY_TIME: {role}",
            )
    v2.exact(sorted(publications), list(range(8)), f"E_BRIDGE_REQUESTS: {role}")
    return v2.derive_bridge(
        normalized,
        role,
        phase_id,
        model,
        clock_id,
        parent,
    )


def _derive_transfer(
    rows: list[dict[str, Any]],
    role: str,
    phase_id: str,
    model: dict[str, Any],
    lock: dict[str, Any],
    phone_execution: dict[str, Any],
) -> dict[str, int]:
    normalized = _normalize_rows(
        rows,
        {"transfer": {"call_index", "row_count"}},
    )
    transfers = [row for row in rows if row["kind"] == "transfer"]
    calls = phone_execution["call_shapes"]
    v2.exact(len(transfers), len(calls), f"E_TRANSFER_CALLS: {role}")
    total = 0
    for index, (row, call) in enumerate(zip(transfers, calls)):
        v2.exact(row["call_index"], index, f"{role}[{index}].call_index")
        v2.exact(row["row_count"], call["n_tokens"], f"{role}[{index}].row_count")
        expected = (
            call["n_tokens"]
            * lock["hidden_size"]
            * lock["activation_element_bytes"]
        )
        v2.exact(row["payload_bytes"], expected, f"E_TRANSFER_SIZE: {role}[{index}]")
        total += expected
    derived = v2.derive_transfer(
        normalized,
        role,
        phase_id,
        model,
        lock,
    )
    v2.exact(derived["direct_payload_bytes"], total, f"E_TRANSFER_TOTAL: {role}")
    return derived


def evaluate_model_phase(
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate: dict[str, Any],
    candidate_raw: bytes,
    parent: dict[str, Any],
    manifest: dict[str, Any],
    manifest_raw: bytes,
    rows_by_role: dict[str, list[dict[str, Any]]],
    artifact_digests: dict[str, str],
    prior_results: list[dict[str, Any]],
) -> dict[str, Any]:
    phase = manifest["phase"]
    v2.require(phase in PHASE_SLOT, f"E_PHASE: {phase}")
    slot = PHASE_SLOT[phase]
    model = _model_by_slot(candidate, slot)
    prior_hashes = [_phase_result_sha256(result) for result in prior_results]
    expected_prior = [] if slot == "A" else prior_hashes
    if slot == "B":
        v2.require(
            len(prior_results) == 1
            and prior_results[0]["status"] == "MODEL_A_QUALIFICATION_PASS",
            "E_PHASE_CHAIN: B requires A",
        )
        v2.require(
            prior_results[0]["phase_closed_ns"] < manifest["phase_opened_ns"],
            "E_PHASE_ORDER: A before B",
        )
    else:
        v2.require(not prior_results, "E_PHASE_CHAIN: A has no predecessor")
    phase_lock = validate_phase_lock(
        rows_by_role["phase.lock"],
        manifest,
        artifact_digests,
        contract,
        contract_raw,
        candidate_raw,
        expected_prior,
    )
    prefix = f"model.{model['model_id']}"
    lock = validate_route_lock(
        rows_by_role[f"{prefix}.route_lock"],
        f"{prefix}.route_lock",
        manifest,
        model,
        contract,
        parent,
    )
    v2.require(
        max(
            max(row["event_ns"] for row in rows_by_role["quality.corpus"]),
            rows_by_role[f"{prefix}.route_lock"][0]["event_ns"],
        )
        <= phase_lock["event_ns"],
        "E_LOCK_ORDER: model inputs after phase lock",
    )
    v2.require(
        phase_lock["event_ns"]
        < min(row["event_ns"] for row in rows_by_role["phase.preflight"]),
        "E_LOCK_ORDER: phase lock must precede preflight",
    )
    preflight = validate_preflight(
        rows_by_role["phase.preflight"],
        manifest,
        contract,
        [(model, lock)],
    )
    task_suite = candidate["task_suite"]
    quality, _ = _derive_quality(
        rows_by_role,
        artifact_digests,
        model,
        manifest["phase_id"],
        task_suite,
        parent,
    )
    phone, _, _ = _validate_execution_geometry(
        rows_by_role,
        model,
        manifest["phase_id"],
    )
    normalized_all = {
        role: _normalize_rows(rows)
        for role, rows in rows_by_role.items()
        if role.startswith(prefix)
    }
    oracle = v2.derive_oracle(normalized_all, model, manifest["phase_id"])
    allocation, cuda_ready = _derive_memory(
        rows_by_role[f"{prefix}.cuda_memory"],
        f"{prefix}.cuda_memory",
        manifest["phase_id"],
        model,
        parent,
    )
    bridge = _derive_bridge(
        rows_by_role[f"{prefix}.bridge"],
        f"{prefix}.bridge",
        manifest["phase_id"],
        model,
        phone,
        rows_by_role[f"{prefix}.mechanics.phone"],
        cuda_ready,
        manifest["clock_id"],
        parent,
    )
    placement = {}
    for phone_name in ("op15", "op12"):
        role = f"{prefix}.placement.{phone_name}"
        placement[phone_name] = v2.derive_placement(
            _normalize_rows(rows_by_role[role]),
            role,
            manifest["phase_id"],
            model,
            phone_name,
            lock,
            parent,
        )
    transfer = _derive_transfer(
        rows_by_role[f"{prefix}.route_transfer"],
        f"{prefix}.route_transfer",
        manifest["phase_id"],
        model,
        lock,
        phone,
    )
    return {
        "bundle_manifest_sha256": v2.sha256_bytes(manifest_raw),
        "candidate_sha256": v2.sha256_bytes(candidate_raw),
        "contract_sha256": v2.sha256_bytes(contract_raw),
        "derived": {
            "model": {
                "allocation": allocation,
                "bridge": bridge,
                "cuda_ready": cuda_ready,
                "model_id": model["model_id"],
                "oracle": oracle,
                "placement": placement,
                "quality": quality,
                "route_lock": lock,
                "transfer": transfer,
            },
            "preflight": preflight,
        },
        "phase": phase,
        "phase_closed_ns": manifest["phase_closed_ns"],
        "phase_id": manifest["phase_id"],
        "phase_opened_ns": manifest["phase_opened_ns"],
        "schema": "s39-cp0-r1-evidence-result-v2.1",
        "status": contract["claim_boundary"]["phase_status"][phase],
    }


def _normalize_pair_memory(
    rows: list[dict[str, Any]],
    prior_hashes: list[str],
) -> list[dict[str, Any]]:
    extras = {"sample_id", "sampler_sha256"}
    normalized = _normalize_rows(
        rows,
        {
            "before": extras,
            "after": extras,
            "attempt": extras
            | {
                "model_phase_result_sha256s",
                "process_pids",
                "process_used_bytes",
                "used_bytes",
            },
        },
    )
    sampler = None
    for index, row in enumerate(rows):
        field = f"pair.cuda_memory[{index}]"
        v2.exact(row["event_ns"], row["timestamp_ns"], f"{field}.event")
        v2.exact(
            row["used_bytes"] + row["free_bytes"],
            row["memory_total_bytes"],
            f"E_MEMORY_EXACT: {field}",
        )
        v2.string(row["sample_id"], f"{field}.sample_id")
        current = v2.digest(row["sampler_sha256"], f"{field}.sampler_sha256")
        sampler = current if sampler is None else sampler
        v2.exact(current, sampler, f"{field}.sampler")
        if row["kind"] == "attempt":
            v2.exact(
                row["model_phase_result_sha256s"],
                prior_hashes,
                "E_PAIR_RESULT_LINK",
            )
            pids = v2.int_list(row["process_pids"], f"{field}.process_pids", 2)
            used = v2.int_list(
                row["process_used_bytes"],
                f"{field}.process_used_bytes",
                2,
            )
            v2.exact(len(pids), 2, f"{field}.process_pids")
            v2.exact(len(used), 2, f"{field}.process_used_bytes")
            v2.require(all(pid > 0 for pid in pids), f"E_PROCESS: {field}")
            v2.require(sum(used) <= row["used_bytes"], f"E_PROCESS_MEMORY: {field}")
    return normalized


def _normalize_reprepare(
    rows: list[dict[str, Any]],
    role: str,
    to_lock: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized = _normalize_rows(
        rows,
        {
            "phone_before": {"local_shard_bytes"},
            "phone_ready": {"local_shard_bytes", "verified_shard_bytes_read"},
        },
    )
    before = {}
    ready = {}
    for row in rows:
        if row["kind"] == "start" or row["kind"] == "end":
            v2.exact(row["event_ns"], row["timestamp_ns"], f"{role}.{row['kind']}")
        elif row["kind"] == "phone_before":
            before[row["phone"]] = row
        elif row["kind"] == "phone_ready":
            ready[row["phone"]] = row
            v2.exact(
                row["event_ns"],
                row["host_received_ns"],
                f"{role}.{row['phone']}.event",
            )
    for phone in ("op15", "op12"):
        expected = to_lock[f"{phone}_shard_bytes"]
        v2.exact(before[phone]["local_shard_bytes"], expected, f"{role}.{phone}.before")
        v2.exact(ready[phone]["local_shard_bytes"], expected, f"{role}.{phone}.ready")
        v2.exact(
            ready[phone]["verified_shard_bytes_read"],
            expected,
            f"E_FULL_SHARD_UFS: {role}.{phone}",
        )
        v2.exact(
            ready[phone]["local_ufs_read_bytes"]
            - before[phone]["local_ufs_read_bytes"],
            expected,
            f"E_UFS_COUNTER: {role}.{phone}",
        )
    return normalized


def evaluate_pair_phase(
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate: dict[str, Any],
    candidate_raw: bytes,
    parent: dict[str, Any],
    manifest: dict[str, Any],
    manifest_raw: bytes,
    rows_by_role: dict[str, list[dict[str, Any]]],
    artifact_digests: dict[str, str],
    prior_results: list[dict[str, Any]],
) -> dict[str, Any]:
    v2.exact(manifest["phase"], "PAIR", "pair.phase")
    v2.require(len(prior_results) == 2, "E_PHASE_CHAIN: pair requires A and B")
    v2.exact(
        [result["status"] for result in prior_results],
        ["MODEL_A_QUALIFICATION_PASS", "MODEL_B_QUALIFICATION_PASS"],
        "E_PHASE_CHAIN: pair statuses",
    )
    v2.require(
        prior_results[0]["phase_closed_ns"]
        < prior_results[1]["phase_opened_ns"]
        and prior_results[1]["phase_closed_ns"] < manifest["phase_opened_ns"],
        "E_PHASE_ORDER: A then B then pair",
    )
    prior_hashes = [_phase_result_sha256(result) for result in prior_results]
    phase_lock = validate_phase_lock(
        rows_by_role["phase.lock"],
        manifest,
        artifact_digests,
        contract,
        contract_raw,
        candidate_raw,
        prior_hashes,
    )
    models = candidate["models"]
    locks = [
        result["derived"]["model"]["route_lock"] for result in prior_results
    ]
    v2.require(
        phase_lock["event_ns"]
        < min(row["event_ns"] for row in rows_by_role["phase.preflight"]),
        "E_LOCK_ORDER: phase lock must precede preflight",
    )
    validate_preflight(
        rows_by_role["phase.preflight"],
        manifest,
        contract,
        list(zip(models, locks)),
    )
    allocations = {
        model["model_id"]: result["derived"]["model"]["allocation"]
        for model, result in zip(models, prior_results)
    }
    pair_rows = _normalize_pair_memory(rows_by_role["pair.cuda_memory"], prior_hashes)
    pair = v2.derive_pair_memory(
        pair_rows,
        "pair.cuda_memory",
        manifest["phase_id"],
        models,
        allocations,
        parent,
    )
    reprepare = {}
    directions = [
        ("A_to_B", models[0], models[1], locks[1]),
        ("B_to_A", models[1], models[0], locks[0]),
    ]
    for direction, from_model, to_model, to_lock in directions:
        role = f"reprepare.{direction}"
        normalized = _normalize_reprepare(
            rows_by_role[role],
            role,
            to_lock,
        )
        reprepare[direction] = v2.derive_reprepare(
            normalized,
            role,
            manifest["phase_id"],
            from_model,
            to_model,
            to_lock,
            manifest["clock_id"],
            parent,
        )
    return {
        "bundle_manifest_sha256": v2.sha256_bytes(manifest_raw),
        "candidate_sha256": v2.sha256_bytes(candidate_raw),
        "contract_sha256": v2.sha256_bytes(contract_raw),
        "derived": {
            "model_phase_result_sha256s": prior_hashes,
            "pair_cuda_memory": pair,
            "reprepare": reprepare,
        },
        "phase": "PAIR",
        "phase_closed_ns": manifest["phase_closed_ns"],
        "phase_id": manifest["phase_id"],
        "phase_opened_ns": manifest["phase_opened_ns"],
        "schema": "s39-cp0-r1-evidence-result-v2.1",
        "status": contract["claim_boundary"]["phase_status"]["PAIR"],
    }


def evaluate_root(
    root: Path,
    manifest_name: str,
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate: dict[str, Any],
    candidate_raw: bytes,
    parent: dict[str, Any],
    prior_results: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest, manifest_raw, rows, digests = load_bundle(
        root,
        manifest_name,
        contract,
        contract_raw,
        candidate_raw,
    )
    if manifest["phase"] == "PAIR":
        return evaluate_pair_phase(
            contract,
            contract_raw,
            candidate,
            candidate_raw,
            parent,
            manifest,
            manifest_raw,
            rows,
            digests,
            prior_results,
        )
    return evaluate_model_phase(
        contract,
        contract_raw,
        candidate,
        candidate_raw,
        parent,
        manifest,
        manifest_raw,
        rows,
        digests,
        prior_results,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive phased CP0-R1 V2.1 eligibility from raw evidence"
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--manifest", default="EVIDENCE_BUNDLE.json")
    parser.add_argument("--a-bundle-root", type=Path)
    parser.add_argument("--b-bundle-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract, contract_raw, candidate, candidate_raw, parent = validate_inputs(
            args.contract,
            args.candidate,
        )
        if args.bundle_root is None:
            result = {
                "candidate_sha256": v2.sha256_bytes(candidate_raw),
                "contract_sha256": v2.sha256_bytes(contract_raw),
                "schema": "s39-cp0-r1-evidence-contract-check-v2.1",
                "status": "V2_1_PHASED_EVIDENCE_READY_ACQUISITION_NOT_RUN",
            }
        else:
            manifest_raw = v2.secure_read(args.bundle_root, args.manifest)
            manifest = v2.parse_json(manifest_raw, args.manifest)
            phase = manifest.get("phase")
            prior = []
            if phase in ("B_ONLY", "PAIR"):
                v2.require(args.a_bundle_root is not None, "E_PHASE_CHAIN: missing A")
                prior.append(
                    evaluate_root(
                        args.a_bundle_root,
                        args.manifest,
                        contract,
                        contract_raw,
                        candidate,
                        candidate_raw,
                        parent,
                        [],
                    )
                )
            if phase == "PAIR":
                v2.require(args.b_bundle_root is not None, "E_PHASE_CHAIN: missing B")
                prior.append(
                    evaluate_root(
                        args.b_bundle_root,
                        args.manifest,
                        contract,
                        contract_raw,
                        candidate,
                        candidate_raw,
                        parent,
                        prior[:1],
                    )
                )
            result = evaluate_root(
                args.bundle_root,
                args.manifest,
                contract,
                contract_raw,
                candidate,
                candidate_raw,
                parent,
                prior,
            )
        print(v2.canonical_bytes(result).decode("ascii"), end="")
        return 0
    except (v2.EvidenceError, OSError, KeyError, ValueError) as exc:
        print(f"CP0_R1_V2_1_REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
