#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "CP0_R1_EVIDENCE_CONTRACT_V2.json"
DEFAULT_CANDIDATE = HERE / "CP0_R1_CANDIDATE.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
MAX_INT = (1 << 63) - 1


class EvidenceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"E_DUPLICATE_KEY: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise EvidenceError(f"E_JSON_NUMBER: non-finite number {value}")


def parse_json(raw: bytes, field: str) -> Any:
    try:
        text = raw.decode("ascii")
        return json.loads(
            text,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"E_JSON: {field}: {exc}") from exc


def canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError) as exc:
        raise EvidenceError("E_CANONICAL: value is not canonical JSON") from exc


def canonical_line(value: Any) -> bytes:
    return canonical_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def is_int(value: Any) -> bool:
    return type(value) is int


def integer(value: Any, field: str, minimum: int = 0) -> int:
    require(is_int(value), f"E_TYPE: {field}: expected integer")
    require(minimum <= value <= MAX_INT, f"E_RANGE: {field}")
    return value


def string(value: Any, field: str, allow_empty: bool = False) -> str:
    require(type(value) is str, f"E_TYPE: {field}: expected string")
    require(allow_empty or bool(value), f"E_EMPTY: {field}")
    return value


def digest(value: Any, field: str) -> str:
    value = string(value, field)
    require(SHA256_RE.fullmatch(value) is not None, f"E_DIGEST: {field}")
    return value


def exact(value: Any, expected: Any, field: str) -> None:
    require(
        type(value) is type(expected) and value == expected,
        f"E_VALUE: {field}: expected {expected!r}, got {value!r}",
    )


def exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    require(type(value) is dict, f"E_TYPE: {field}: expected object")
    actual = set(value)
    require(
        actual == expected,
        f"E_KEYS: {field}: missing={sorted(expected - actual)}, "
        f"unknown={sorted(actual - expected)}",
    )
    return value


def int_list(value: Any, field: str, minimum_length: int = 1) -> list[int]:
    require(
        type(value) is list and len(value) >= minimum_length,
        f"E_TYPE: {field}: expected integer list",
    )
    return [integer(item, f"{field}[{index}]") for index, item in enumerate(value)]


def digest_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def _normalize_relative(path_text: Any) -> list[str]:
    path_text = string(path_text, "artifact.path")
    require("\\" not in path_text, f"E_PATH: {path_text!r}")
    candidate = PurePosixPath(path_text)
    require(not candidate.is_absolute(), f"E_PATH: {path_text!r}")
    parts = list(candidate.parts)
    require(
        bool(parts)
        and all(part not in ("", ".", "..") for part in parts)
        and str(candidate) == path_text,
        f"E_PATH: {path_text!r}",
    )
    return parts


def _open_root(root: Path) -> int:
    absolute = root.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    current = os.open("/", flags)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
        return current
    except Exception:
        os.close(current)
        raise


def secure_read(root: Path, path_text: str) -> bytes:
    parts = _normalize_relative(path_text)
    try:
        root_fd = _open_root(root)
    except OSError as exc:
        raise EvidenceError(f"E_PATH: trusted root is unavailable: {exc}") from exc
    parent = os.dup(root_fd)
    try:
        dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        root_dev = os.fstat(root_fd).st_dev
        for part in parts[:-1]:
            child = os.open(part, dir_flags, dir_fd=parent)
            info = os.fstat(child)
            require(info.st_dev == root_dev, f"E_PATH: mount crossing: {path_text}")
            os.close(parent)
            parent = child
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        fd = os.open(parts[-1], flags, dir_fd=parent)
        try:
            before = os.fstat(fd)
            require(stat.S_ISREG(before.st_mode), f"E_TYPE: {path_text}: not regular")
            require(before.st_nlink == 1, f"E_HARDLINK: {path_text}")
            chunks = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(fd)
            fingerprint = lambda value: (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_nlink,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
            require(
                fingerprint(before) == fingerprint(after),
                f"E_CHANGED: {path_text}",
            )
            return b"".join(chunks)
        finally:
            os.close(fd)
    except (OSError, EvidenceError) as exc:
        if isinstance(exc, EvidenceError):
            raise
        raise EvidenceError(f"E_PATH: {path_text}: {exc}") from exc
    finally:
        os.close(parent)
        os.close(root_fd)


def load_canonical_path(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = secure_read(path.parent, path.name)
    value = parse_json(raw, str(path))
    require(type(value) is dict, f"E_TYPE: {path}: expected object")
    require(canonical_bytes(value) == raw, f"E_CANONICAL: {path}")
    return value, raw


def parse_jsonl(raw: bytes, role: str, acquisition_id: str) -> list[dict[str, Any]]:
    require(raw.endswith(b"\n"), f"E_CANONICAL: {role}: missing final newline")
    lines = raw.splitlines(keepends=True)
    require(bool(lines), f"E_EMPTY: {role}")
    rows = []
    for index, line in enumerate(lines):
        require(line != b"\n", f"E_EMPTY: {role}[{index}]")
        row = parse_json(line, f"{role}[{index}]")
        require(type(row) is dict, f"E_TYPE: {role}[{index}]")
        require(canonical_line(row) == line, f"E_CANONICAL: {role}[{index}]")
        exact(row.get("role"), role, f"{role}[{index}].role")
        exact(
            row.get("acquisition_id"),
            acquisition_id,
            f"{role}[{index}].acquisition_id",
        )
        rows.append(row)
    return rows


def required_roles(model_ids: list[str]) -> list[str]:
    roles = ["quality.corpus"]
    for model_id in model_ids:
        prefix = f"model.{model_id}"
        roles.extend(
            [
                f"{prefix}.route_lock",
                f"{prefix}.mechanics.phone",
                f"{prefix}.oracle.cuda_route",
                f"{prefix}.oracle.cuda_monolithic",
                f"{prefix}.cuda_memory",
                f"{prefix}.quality.cuda",
                f"{prefix}.quality.phone",
                f"{prefix}.bridge",
                f"{prefix}.placement.op15",
                f"{prefix}.placement.op12",
                f"{prefix}.route_transfer",
            ]
        )
    return roles + [
        "pair.cuda_memory",
        "reprepare.A_to_B",
        "reprepare.B_to_A",
    ]


def validate_contract(contract: dict[str, Any]) -> None:
    exact_keys(
        contract,
        {
            "candidate_lock",
            "claim_boundary",
            "derivations",
            "devices",
            "gates",
            "parent",
            "raw_evidence",
            "schema",
            "scope",
            "serving_envelope",
            "status",
        },
        "contract",
    )
    exact(contract["schema"], "s39-cp0-r1-evidence-contract-v2", "contract.schema")
    exact(
        contract["status"],
        "FROZEN_BEFORE_CP0_R1_V2_MODEL_ACQUISITION",
        "contract.status",
    )
    exact(
        contract["scope"],
        "EVIDENCE_HARDENING_AND_NO_MODEL_PREFLIGHT_ONLY",
        "contract.scope",
    )
    parent = exact_keys(
        contract["parent"],
        {"candidate_sha256", "contract_sha256", "files", "manifest_sha256"},
        "contract.parent",
    )
    for field in ("candidate_sha256", "contract_sha256", "manifest_sha256"):
        digest(parent[field], f"contract.parent.{field}")
    exact(
        parent["candidate_sha256"],
        "ee3196ca660fa7eb7ea293260dc98dd6fdbf14571a4d4c5aece5343fb29b28d8",
        "contract.parent.candidate_sha256",
    )
    exact(
        parent["contract_sha256"],
        "ffb2abeb33e818477e8a7181e177a6d296ed3b021767bc83a8df0d2450e5d095",
        "contract.parent.contract_sha256",
    )
    exact(
        parent["manifest_sha256"],
        "480cf8836e0a83ea9102c19e82bf5ec01e2f78b0ea2a3d9061be71bed4b31275",
        "contract.parent.manifest_sha256",
    )
    files = parent["files"]
    require(type(files) is dict and bool(files), "E_TYPE: contract.parent.files")
    for path, value in files.items():
        string(path, "contract.parent.files.path")
        digest(value, f"contract.parent.files.{path}")

    lock = exact_keys(
        contract["candidate_lock"],
        {
            "candidate_attempt",
            "maximum_new_candidates",
            "models",
            "task_suite_sha256",
        },
        "contract.candidate_lock",
    )
    exact(lock["candidate_attempt"], 1, "candidate_lock.candidate_attempt")
    exact(lock["maximum_new_candidates"], 1, "candidate_lock.maximum_new_candidates")
    digest(lock["task_suite_sha256"], "candidate_lock.task_suite_sha256")
    models = lock["models"]
    require(type(models) is list and len(models) == 2, "E_TYPE: candidate_lock.models")
    for index, model in enumerate(models):
        exact_keys(
            model,
            {
                "architecture",
                "artifact_bytes",
                "artifact_sha256",
                "model_id",
                "n_layer",
                "quantization",
                "slot",
            },
            f"candidate_lock.models[{index}]",
        )
        exact(model["slot"], "AB"[index], f"candidate_lock.models[{index}].slot")
        exact(model["architecture"], "qwen3", f"candidate_lock.models[{index}].arch")
        string(model["model_id"], f"candidate_lock.models[{index}].model_id")
        string(model["quantization"], f"candidate_lock.models[{index}].quantization")
        integer(model["artifact_bytes"], f"candidate_lock.models[{index}].bytes", 1)
        digest(model["artifact_sha256"], f"candidate_lock.models[{index}].sha")
        integer(model["n_layer"], f"candidate_lock.models[{index}].n_layer", 1)

    devices = exact_keys(contract["devices"], {"cuda", "op12", "op15"}, "devices")
    cuda = exact_keys(
        devices["cuda"],
        {"host", "memory_total_bytes", "name", "uuid"},
        "devices.cuda",
    )
    exact(cuda["host"], "zhihao-Z690-C-ac", "devices.cuda.host")
    exact(cuda["name"], "NVIDIA GeForce RTX 4060 Ti", "devices.cuda.name")
    exact(
        cuda["uuid"],
        "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
        "devices.cuda.uuid",
    )
    exact(cuda["memory_total_bytes"], 17_175_674_880, "devices.cuda.memory")
    expected_phones = {
        "op12": ("5ae7a43d", "CPH2583", "CPH2583", "OP595DL1"),
        "op15": ("3C15AU002CL00000", "CPH2749", "CPH2749", "OP611FL1"),
    }
    for phone, expected in expected_phones.items():
        item = exact_keys(
            devices[phone],
            {"device", "model", "product", "serial"},
            f"devices.{phone}",
        )
        for key, value in zip(("serial", "model", "product", "device"), expected):
            exact(item[key], value, f"devices.{phone}.{key}")

    envelope = exact_keys(
        contract["serving_envelope"],
        {
            "batch",
            "kv_type_k",
            "kv_type_v",
            "max_streams",
            "n_batch",
            "n_ctx_seq",
            "n_ubatch",
            "sampler",
        },
        "serving_envelope",
    )
    exact(
        envelope,
        {
            "batch": 8,
            "kv_type_k": "f16",
            "kv_type_v": "f16",
            "max_streams": 8,
            "n_batch": 64,
            "n_ctx_seq": 256,
            "n_ubatch": 64,
            "sampler": "greedy",
        },
        "serving_envelope",
    )
    gates = exact_keys(
        contract["gates"],
        {
            "bridge_minimum_requests",
            "bridge_minimum_tokens",
            "cuda_minimum_free_bytes",
            "maximum_process_swap_bytes",
            "maximum_system_swap_growth_bytes",
            "phone_minimum_available_bytes",
            "quality_items",
            "quality_maximum_new_errors",
            "quality_maximum_score_regression_items",
            "reprepare_maximum_elapsed_us",
        },
        "gates",
    )
    for field, value in gates.items():
        integer(value, f"gates.{field}")
    exact(
        gates,
        {
            "bridge_minimum_requests": 8,
            "bridge_minimum_tokens": 8,
            "cuda_minimum_free_bytes": 536_870_912,
            "maximum_process_swap_bytes": 0,
            "maximum_system_swap_growth_bytes": 0,
            "phone_minimum_available_bytes": 536_870_912,
            "quality_items": 64,
            "quality_maximum_new_errors": 1,
            "quality_maximum_score_regression_items": 1,
            "reprepare_maximum_elapsed_us": 30_000_000,
        },
        "gates",
    )

    raw = exact_keys(
        contract["raw_evidence"],
        {
            "artifact_format",
            "common_clock",
            "maximum_artifact_bytes",
            "read_semantics",
            "required_roles",
            "role_count",
            "role_reuse",
            "summary_input",
        },
        "raw_evidence",
    )
    exact(raw["artifact_format"], "CANONICAL_ASCII_JSONL", "raw.format")
    exact(raw["common_clock"], "HOST_MONOTONIC_RAW", "raw.clock")
    exact(raw["read_semantics"], "OPEN_ONCE_HASH_AND_PARSE_SAME_BYTES", "raw.read")
    exact(raw["role_reuse"], "FORBIDDEN", "raw.role_reuse")
    exact(raw["summary_input"], "FORBIDDEN", "raw.summary")
    exact(raw["maximum_artifact_bytes"], 67_108_864, "raw.maximum_artifact_bytes")
    expected_roles = required_roles([model["model_id"] for model in models])
    exact(raw["required_roles"], expected_roles, "raw.required_roles")
    exact(raw["role_count"], len(expected_roles), "raw.role_count")

    exact(
        contract["derivations"],
        [
            "INDEPENDENT_PATH_MATCHED_CUDA_ORACLE",
            "CUDA_B8_MEMORY_AND_PAIR_NON_CORESIDENCY",
            "PER_ITEM_TASK_QUALITY",
            "COMMON_CLOCK_LIVE_BRIDGE",
            "REALIZED_PHONE_PLACEMENT",
            "DIRECT_PHONE_ACTIVATION",
            "FULL_LOCAL_UFS_REPREPARE",
        ],
        "contract.derivations",
    )
    claim = exact_keys(
        contract["claim_boundary"],
        {
            "eligibility_requires_complete_raw_bundle",
            "eligibility_status",
            "forbidden_before_eligibility",
            "mechanics_status",
            "preflight_pass_does_not_authorize_model_acquisition",
        },
        "claim_boundary",
    )
    exact(
        claim["mechanics_status"],
        "CP0_R1_V2_EVIDENCE_MECHANICS_PASS",
        "claim.mechanics_status",
    )
    exact(
        claim["eligibility_status"],
        "TWO_ROUTE_ELIGIBILITY_PASS",
        "claim.eligibility_status",
    )
    exact(claim["eligibility_requires_complete_raw_bundle"], True, "claim.complete")
    exact(
        claim["preflight_pass_does_not_authorize_model_acquisition"],
        True,
        "claim.preflight",
    )
    exact(
        claim["forbidden_before_eligibility"],
        [
            "MODEL_SWITCH_CYCLE",
            "TRACE_REPLAY",
            "CONTROLLER_INTEGRATION",
            "ENERGY_ACQUISITION",
        ],
        "claim.forbidden_before_eligibility",
    )


def validate_candidate(
    candidate: dict[str, Any],
    candidate_raw: bytes,
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lock = contract["candidate_lock"]
    exact(
        sha256_bytes(candidate_raw),
        contract["parent"]["candidate_sha256"],
        "candidate.sha256",
    )
    exact(candidate["candidate_attempt"], lock["candidate_attempt"], "candidate.attempt")
    exact(candidate["candidate_attempt_limit"], 1, "candidate.attempt_limit")
    models = candidate["models"]
    require(type(models) is list and len(models) == 2, "E_TYPE: candidate.models")
    for actual, expected in zip(models, lock["models"]):
        for field in ("architecture", "model_id", "n_layer", "quantization", "slot"):
            exact(actual[field], expected[field], f"candidate.{expected['slot']}.{field}")
        exact(
            actual["artifact"]["bytes"],
            expected["artifact_bytes"],
            f"candidate.{expected['slot']}.artifact.bytes",
        )
        exact(
            actual["artifact"]["sha256"],
            expected["artifact_sha256"],
            f"candidate.{expected['slot']}.artifact.sha256",
        )
    exact(
        digest_json(candidate["task_suite"]),
        lock["task_suite_sha256"],
        "candidate.task_suite",
    )
    return models, candidate["task_suite"]


def load_bundle(
    root: Path,
    manifest_name: str,
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate_raw: bytes,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest_raw = secure_read(root, manifest_name)
    manifest = parse_json(manifest_raw, manifest_name)
    require(type(manifest) is dict, "E_TYPE: bundle")
    require(canonical_bytes(manifest) == manifest_raw, "E_CANONICAL: bundle")
    exact_keys(
        manifest,
        {
            "acquisition_id",
            "acquisition_started_ns",
            "artifacts",
            "candidate_attempt",
            "candidate_sha256",
            "clock_id",
            "contract_sha256",
            "schema",
        },
        "bundle",
    )
    exact(manifest["schema"], "s39-cp0-r1-evidence-bundle-v2", "bundle.schema")
    acquisition_id = string(manifest["acquisition_id"], "bundle.acquisition_id")
    clock_id = string(manifest["clock_id"], "bundle.clock_id")
    exact(clock_id, contract["raw_evidence"]["common_clock"], "bundle.clock_id")
    integer(manifest["acquisition_started_ns"], "bundle.acquisition_started_ns", 1)
    exact(manifest["candidate_attempt"], 1, "bundle.candidate_attempt")
    exact(manifest["contract_sha256"], sha256_bytes(contract_raw), "bundle.contract")
    exact(manifest["candidate_sha256"], sha256_bytes(candidate_raw), "bundle.candidate")

    artifacts = manifest["artifacts"]
    require(type(artifacts) is list, "E_TYPE: bundle.artifacts")
    expected_roles = contract["raw_evidence"]["required_roles"]
    require(len(artifacts) == len(expected_roles), "E_ROLE_SET: artifact count")
    by_role: dict[str, dict[str, Any]] = {}
    paths = set()
    digests = set()
    for index, artifact in enumerate(artifacts):
        field = f"bundle.artifacts[{index}]"
        exact_keys(
            artifact,
            {"bytes", "format", "path", "role", "sha256"},
            field,
        )
        role = string(artifact["role"], f"{field}.role")
        require(role not in by_role, f"E_ROLE_REUSE: {role}")
        path = string(artifact["path"], f"{field}.path")
        _normalize_relative(path)
        require(path != manifest_name, f"E_PATH: manifest cannot be evidence: {path}")
        require(path not in paths, f"E_PATH_REUSE: {path}")
        declared_digest = digest(artifact["sha256"], f"{field}.sha256")
        require(declared_digest not in digests, f"E_DIGEST_REUSE: {role}")
        exact(artifact["format"], "CANONICAL_ASCII_JSONL", f"{field}.format")
        size = integer(artifact["bytes"], f"{field}.bytes", 1)
        require(
            size <= contract["raw_evidence"]["maximum_artifact_bytes"],
            f"E_SIZE: {role}",
        )
        by_role[role] = artifact
        paths.add(path)
        digests.add(declared_digest)
    exact(sorted(by_role), sorted(expected_roles), "bundle.role_set")

    rows_by_role = {}
    for role in expected_roles:
        artifact = by_role[role]
        raw = secure_read(root, artifact["path"])
        exact(len(raw), artifact["bytes"], f"{role}.bytes")
        exact(sha256_bytes(raw), artifact["sha256"], f"{role}.sha256")
        rows_by_role[role] = parse_jsonl(raw, role, acquisition_id)
    return manifest, rows_by_role


def common_keys(extra: set[str]) -> set[str]:
    return {"acquisition_id", "kind", "role"} | extra


def validate_row(
    row: Any,
    role: str,
    acquisition_id: str,
    kind: str,
    extra: set[str],
    field: str,
) -> dict[str, Any]:
    row = exact_keys(row, common_keys(extra), field)
    exact(row["role"], role, f"{field}.role")
    exact(row["acquisition_id"], acquisition_id, f"{field}.acquisition_id")
    exact(row["kind"], kind, f"{field}.kind")
    return row


def validate_route_lock(
    rows: list[dict[str, Any]],
    role: str,
    acquisition_id: str,
    acquisition_started_ns: int,
    clock_id: str,
    model: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    require(len(rows) == 1, f"E_ROWS: {role}: expected one row")
    row = validate_row(
        rows[0],
        role,
        acquisition_id,
        "route_lock",
        {
            "backend",
            "batch_config_sha256",
            "clock_id",
            "cut_layer",
            "frozen_ns",
            "model_id",
            "model_sha256",
            "n_layer",
            "op12_shard_sha256",
            "op12_stored_layers",
            "op15_shard_sha256",
            "op15_stored_layers",
        },
        role,
    )
    exact(row["clock_id"], clock_id, f"{role}.clock_id")
    require(
        integer(row["frozen_ns"], f"{role}.frozen_ns", 1) < acquisition_started_ns,
        f"E_ORDER: {role}: route was not frozen before acquisition",
    )
    exact(row["model_id"], model["model_id"], f"{role}.model_id")
    exact(row["model_sha256"], model["artifact"]["sha256"], f"{role}.model_sha256")
    exact(row["n_layer"], model["n_layer"], f"{role}.n_layer")
    exact(row["backend"], "GPUOpenCL", f"{role}.backend")
    exact(
        row["batch_config_sha256"],
        digest_json(contract["serving_envelope"]),
        f"{role}.batch_config_sha256",
    )
    cut = integer(row["cut_layer"], f"{role}.cut_layer", 1)
    require(cut < model["n_layer"], f"E_RANGE: {role}.cut_layer")
    for phone in ("op15", "op12"):
        layers = int_list(row[f"{phone}_stored_layers"], f"{role}.{phone}.stored", 2)
        require(len(layers) == 2 and layers[0] < layers[1], f"E_RANGE: {role}.{phone}")
        require(layers[1] <= model["n_layer"], f"E_RANGE: {role}.{phone}")
        digest(row[f"{phone}_shard_sha256"], f"{role}.{phone}.shard")
    require(
        row["op15_stored_layers"][0] == 0
        and row["op15_stored_layers"][1] >= cut,
        f"E_COVERAGE: {role}.op15",
    )
    require(
        row["op12_stored_layers"][0] <= cut
        and row["op12_stored_layers"][1] == model["n_layer"],
        f"E_COVERAGE: {role}.op12",
    )
    return row


def validate_call_shapes(value: Any, field: str) -> list[dict[str, Any]]:
    require(type(value) is list and bool(value), f"E_TYPE: {field}")
    result = []
    for index, item in enumerate(value):
        item = exact_keys(
            item,
            {"call_index", "n_seqs", "n_tokens", "phase"},
            f"{field}[{index}]",
        )
        exact(item["call_index"], index, f"{field}[{index}].call_index")
        integer(item["n_seqs"], f"{field}[{index}].n_seqs", 1)
        integer(item["n_tokens"], f"{field}[{index}].n_tokens", 1)
        require(item["phase"] in ("decode", "prefill"), f"E_VALUE: {field}.phase")
        result.append(item)
    return result


def validate_execution(
    rows: list[dict[str, Any]],
    role: str,
    acquisition_id: str,
    model: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    require(len(rows) == 9, f"E_ROWS: {role}: expected meta plus B8 rows")
    meta = validate_row(
        rows[0],
        role,
        acquisition_id,
        "meta",
        {
            "backend",
            "call_shapes",
            "model_id",
            "model_sha256",
            "program_sha256",
            "state_count_after",
            "state_count_before",
        },
        f"{role}[0]",
    )
    exact(meta["backend"], backend, f"{role}.backend")
    exact(meta["model_id"], model["model_id"], f"{role}.model_id")
    exact(meta["model_sha256"], model["artifact"]["sha256"], f"{role}.model_sha256")
    digest(meta["program_sha256"], f"{role}.program_sha256")
    exact(meta["state_count_before"], 0, f"{role}.state_count_before")
    exact(meta["state_count_after"], 0, f"{role}.state_count_after")
    calls = validate_call_shapes(meta["call_shapes"], f"{role}.call_shapes")

    requests = {}
    for index, raw in enumerate(rows[1:]):
        field = f"{role}[{index + 1}]"
        row = validate_row(
            raw,
            role,
            acquisition_id,
            "request",
            {
                "continuation_tokens",
                "input_tokens",
                "model_id",
                "model_sha256",
                "owner_after",
                "owner_before",
                "ownership_epoch_after",
                "ownership_epoch_before",
                "positions",
                "request_id",
            },
            field,
        )
        request_id = integer(row["request_id"], f"{field}.request_id")
        require(request_id not in requests, f"E_DUPLICATE_REQUEST: {role}:{request_id}")
        exact(row["model_id"], model["model_id"], f"{field}.model_id")
        exact(row["model_sha256"], model["artifact"]["sha256"], f"{field}.model_sha")
        inputs = int_list(row["input_tokens"], f"{field}.input_tokens")
        positions = int_list(row["positions"], f"{field}.positions")
        continuation = int_list(
            row["continuation_tokens"],
            f"{field}.continuation_tokens",
        )
        require(len(inputs) == len(positions), f"E_POSITION: {field}")
        require(
            all(right > left for left, right in zip(positions, positions[1:])),
            f"E_POSITION: {field}: not strictly increasing",
        )
        expected_owner = "PHONE" if backend == "PHONE_COLLECTIVE" else "CUDA"
        exact(row["owner_before"], expected_owner, f"{field}.owner_before")
        exact(row["owner_after"], "RELEASED", f"{field}.owner_after")
        before = integer(row["ownership_epoch_before"], f"{field}.epoch_before", 1)
        after = integer(row["ownership_epoch_after"], f"{field}.epoch_after", 1)
        require(after == before + 1, f"E_OWNERSHIP: {field}")
        requests[request_id] = {
            "continuation_tokens": continuation,
            "input_tokens": inputs,
            "positions": positions,
        }
    exact(sorted(requests), list(range(8)), f"{role}.request_ids")
    return {
        "call_shapes": calls,
        "program_sha256": meta["program_sha256"],
        "requests": requests,
    }


def derive_oracle(
    rows_by_role: dict[str, list[dict[str, Any]]],
    model: dict[str, Any],
    acquisition_id: str,
) -> dict[str, Any]:
    prefix = f"model.{model['model_id']}"
    phone = validate_execution(
        rows_by_role[f"{prefix}.mechanics.phone"],
        f"{prefix}.mechanics.phone",
        acquisition_id,
        model,
        "PHONE_COLLECTIVE",
    )
    tested = validate_execution(
        rows_by_role[f"{prefix}.oracle.cuda_route"],
        f"{prefix}.oracle.cuda_route",
        acquisition_id,
        model,
        "CUDA0",
    )
    oracle = validate_execution(
        rows_by_role[f"{prefix}.oracle.cuda_monolithic"],
        f"{prefix}.oracle.cuda_monolithic",
        acquisition_id,
        model,
        "CUDA0",
    )
    require(
        tested["program_sha256"] != oracle["program_sha256"],
        f"E_ORACLE_INDEPENDENCE: {model['model_id']}",
    )
    exact(
        tested["call_shapes"],
        oracle["call_shapes"],
        f"{model['model_id']}.oracle.call_shapes",
    )
    greedy_matches = 0
    greedy_total = 0
    for request_id in range(8):
        phone_row = phone["requests"][request_id]
        tested_row = tested["requests"][request_id]
        oracle_row = oracle["requests"][request_id]
        exact(
            tested_row["input_tokens"],
            oracle_row["input_tokens"],
            f"{model['model_id']}.oracle.input[{request_id}]",
        )
        exact(
            tested_row["positions"],
            oracle_row["positions"],
            f"{model['model_id']}.oracle.positions[{request_id}]",
        )
        exact(
            tested_row["continuation_tokens"],
            oracle_row["continuation_tokens"],
            f"{model['model_id']}.oracle.continuation[{request_id}]",
        )
        exact(
            phone_row["input_tokens"],
            tested_row["input_tokens"],
            f"{model['model_id']}.phone_history[{request_id}]",
        )
        exact(
            phone_row["positions"],
            tested_row["positions"],
            f"{model['model_id']}.phone_positions[{request_id}]",
        )
        for phone_token, cuda_token in zip(
            phone_row["continuation_tokens"],
            tested_row["continuation_tokens"],
        ):
            greedy_total += 1
            greedy_matches += int(phone_token == cuda_token)
    require(greedy_total > 0, f"E_EMPTY: {model['model_id']}.greedy")
    return {
        "cross_backend_greedy_matches": greedy_matches,
        "cross_backend_greedy_total": greedy_total,
        "cross_geometry_exact": phone["call_shapes"] == tested["call_shapes"],
    }


def validate_memory_row(
    row: dict[str, Any],
    role: str,
    acquisition_id: str,
    kind: str,
    model: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    row = validate_row(
        row,
        role,
        acquisition_id,
        kind,
        {
            "batch",
            "clock_id",
            "completed_requests",
            "config_sha256",
            "device_name",
            "device_uuid",
            "free_bytes",
            "host_swap_used_bytes",
            "kv_buffer_bytes",
            "memory_total_bytes",
            "model_buffer_bytes",
            "model_id",
            "model_sha256",
            "placement_compute_nodes",
            "state_count",
            "timestamp_ns",
            "used_bytes",
        },
        f"{role}.{kind}",
    )
    exact(row["clock_id"], contract["raw_evidence"]["common_clock"], f"{role}.clock")
    exact(row["device_name"], contract["devices"]["cuda"]["name"], f"{role}.name")
    exact(row["device_uuid"], contract["devices"]["cuda"]["uuid"], f"{role}.uuid")
    exact(
        row["memory_total_bytes"],
        contract["devices"]["cuda"]["memory_total_bytes"],
        f"{role}.total",
    )
    exact(row["model_id"], model["model_id"], f"{role}.model_id")
    exact(row["model_sha256"], model["artifact"]["sha256"], f"{role}.model_sha")
    for field in (
        "batch",
        "completed_requests",
        "free_bytes",
        "host_swap_used_bytes",
        "kv_buffer_bytes",
        "memory_total_bytes",
        "model_buffer_bytes",
        "placement_compute_nodes",
        "state_count",
        "timestamp_ns",
        "used_bytes",
    ):
        integer(row[field], f"{role}.{kind}.{field}")
    require(
        row["used_bytes"] + row["free_bytes"] <= row["memory_total_bytes"],
        f"E_MEMORY: {role}.{kind}",
    )
    if kind == "ready":
        exact(row["batch"], 8, f"{role}.ready.batch")
        exact(row["completed_requests"], 8, f"{role}.ready.completed")
        require(row["model_buffer_bytes"] > 0, f"E_MEMORY: {role}.model_buffer")
        require(row["kv_buffer_bytes"] > 0, f"E_MEMORY: {role}.kv_buffer")
        require(row["placement_compute_nodes"] > 0, f"E_PLACEMENT: {role}.cuda")
        exact(row["state_count"], 8, f"{role}.ready.state_count")
        exact(
            row["config_sha256"],
            digest_json(contract["serving_envelope"]),
            f"{role}.ready.config",
        )
    else:
        for field in (
            "batch",
            "completed_requests",
            "kv_buffer_bytes",
            "model_buffer_bytes",
            "placement_compute_nodes",
            "state_count",
        ):
            exact(row[field], 0, f"{role}.{kind}.{field}")
        exact(row["config_sha256"], "NONE", f"{role}.{kind}.config")
    return row


def derive_cuda_memory(
    rows: list[dict[str, Any]],
    role: str,
    acquisition_id: str,
    model: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, int]:
    require(len(rows) == 3, f"E_ROWS: {role}: expected before, ready, after")
    before = validate_memory_row(
        rows[0], role, acquisition_id, "before", model, contract
    )
    ready = validate_memory_row(
        rows[1], role, acquisition_id, "ready", model, contract
    )
    after = validate_memory_row(
        rows[2], role, acquisition_id, "after", model, contract
    )
    require(
        before["timestamp_ns"] < ready["timestamp_ns"] < after["timestamp_ns"],
        f"E_ORDER: {role}",
    )
    require(
        ready["free_bytes"] >= contract["gates"]["cuda_minimum_free_bytes"],
        f"E_HEADROOM: {role}",
    )
    require(
        after["host_swap_used_bytes"] - before["host_swap_used_bytes"]
        <= contract["gates"]["maximum_system_swap_growth_bytes"],
        f"E_SWAP: {role}",
    )
    return {
        "kv_buffer_bytes": ready["kv_buffer_bytes"],
        "model_buffer_bytes": ready["model_buffer_bytes"],
    }


def derive_pair_memory(
    rows: list[dict[str, Any]],
    role: str,
    acquisition_id: str,
    models: list[dict[str, Any]],
    allocations: dict[str, dict[str, int]],
    contract: dict[str, Any],
) -> dict[str, int]:
    require(len(rows) == 3, f"E_ROWS: {role}")
    simple_keys = {
        "clock_id",
        "device_uuid",
        "free_bytes",
        "host_swap_used_bytes",
        "memory_total_bytes",
        "timestamp_ns",
        "used_bytes",
    }
    before = validate_row(
        rows[0], role, acquisition_id, "before", simple_keys, f"{role}.before"
    )
    after = validate_row(
        rows[2], role, acquisition_id, "after", simple_keys, f"{role}.after"
    )
    attempt = validate_row(
        rows[1],
        role,
        acquisition_id,
        "attempt",
        {
            "clock_id",
            "config_sha256",
            "device_uuid",
            "exit_code",
            "free_bytes",
            "kv_buffer_bytes",
            "memory_total_bytes",
            "model_buffer_bytes",
            "model_ids",
            "model_sha256s",
            "outcome",
            "timestamp_ns",
        },
        f"{role}.attempt",
    )
    for record, label in ((before, "before"), (after, "after")):
        exact(record["clock_id"], contract["raw_evidence"]["common_clock"], f"{role}.{label}.clock")
        exact(record["device_uuid"], contract["devices"]["cuda"]["uuid"], f"{role}.{label}.uuid")
        exact(
            record["memory_total_bytes"],
            contract["devices"]["cuda"]["memory_total_bytes"],
            f"{role}.{label}.total",
        )
        for field in (
            "free_bytes",
            "host_swap_used_bytes",
            "memory_total_bytes",
            "timestamp_ns",
            "used_bytes",
        ):
            integer(record[field], f"{role}.{label}.{field}")
    exact(attempt["clock_id"], contract["raw_evidence"]["common_clock"], f"{role}.attempt.clock")
    exact(attempt["device_uuid"], contract["devices"]["cuda"]["uuid"], f"{role}.attempt.uuid")
    exact(
        attempt["memory_total_bytes"],
        contract["devices"]["cuda"]["memory_total_bytes"],
        f"{role}.attempt.total",
    )
    expected_ids = [model["model_id"] for model in models]
    expected_shas = [model["artifact"]["sha256"] for model in models]
    exact(attempt["model_ids"], expected_ids, f"{role}.attempt.model_ids")
    exact(attempt["model_sha256s"], expected_shas, f"{role}.attempt.model_shas")
    expected_model_bytes = [allocations[item]["model_buffer_bytes"] for item in expected_ids]
    expected_kv_bytes = [allocations[item]["kv_buffer_bytes"] for item in expected_ids]
    exact(attempt["model_buffer_bytes"], expected_model_bytes, f"{role}.model_bytes")
    exact(attempt["kv_buffer_bytes"], expected_kv_bytes, f"{role}.kv_bytes")
    exact(
        attempt["config_sha256"],
        digest_json(contract["serving_envelope"]),
        f"{role}.config",
    )
    exact(attempt["outcome"], "CUDA_OOM", f"{role}.outcome")
    require(integer(attempt["exit_code"], f"{role}.exit_code", 1) > 0, f"E_EXIT: {role}")
    integer(attempt["free_bytes"], f"{role}.attempt.free_bytes")
    integer(attempt["timestamp_ns"], f"{role}.attempt.timestamp_ns", 1)
    require(
        before["timestamp_ns"] < attempt["timestamp_ns"] < after["timestamp_ns"],
        f"E_ORDER: {role}",
    )
    require(
        after["host_swap_used_bytes"] - before["host_swap_used_bytes"]
        <= contract["gates"]["maximum_system_swap_growth_bytes"],
        f"E_SWAP: {role}",
    )
    lower_bound = (
        before["used_bytes"]
        + contract["gates"]["cuda_minimum_free_bytes"]
        + sum(expected_model_bytes)
        + sum(expected_kv_bytes)
    )
    require(
        lower_bound > contract["devices"]["cuda"]["memory_total_bytes"],
        f"E_CORESIDENCY: {role}: measured allocations fit",
    )
    return {"lower_bound_used_bytes": lower_bound}


def derive_corpus(
    rows: list[dict[str, Any]],
    role: str,
    acquisition_id: str,
    task_suite: dict[str, Any],
    contract: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    exact(len(rows), contract["gates"]["quality_items"], f"{role}.rows")
    result = {}
    seen_sources = set()
    mapping = ["A", "B", "C", "D"]
    for index, raw in enumerate(rows):
        field = f"{role}[{index}]"
        row = validate_row(
            raw,
            role,
            acquisition_id,
            "item",
            {
                "choices",
                "dataset",
                "dataset_revision",
                "expected_answer",
                "item_index",
                "question",
                "source_row",
                "subject",
            },
            field,
        )
        exact(row["item_index"], index, f"{field}.item_index")
        exact(row["dataset"], task_suite["dataset"], f"{field}.dataset")
        exact(row["dataset_revision"], task_suite["revision"], f"{field}.revision")
        subject = string(row["subject"], f"{field}.subject")
        source_row = integer(row["source_row"], f"{field}.source_row")
        require((subject, source_row) not in seen_sources, f"E_CORPUS_REUSE: {field}")
        seen_sources.add((subject, source_row))
        question = string(row["question"], f"{field}.question")
        choices = row["choices"]
        require(
            type(choices) is list
            and len(choices) == 4
            and all(type(choice) is str and bool(choice) for choice in choices),
            f"E_TYPE: {field}.choices",
        )
        expected = string(row["expected_answer"], f"{field}.expected_answer")
        require(expected in mapping, f"E_VALUE: {field}.expected_answer")
        prompt = task_suite["prompt_format"].format(
            question=question,
            choice0=choices[0],
            choice1=choices[1],
            choice2=choices[2],
            choice3=choices[3],
        )
        result[index] = {
            "expected": expected,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        }
    return result


def parse_answer(raw_output: str) -> str | None:
    match = re.match(r"^([A-D])(?:\b|$)", raw_output.lstrip())
    return match.group(1) if match else None


def derive_quality_output(
    rows: list[dict[str, Any]],
    role: str,
    acquisition_id: str,
    model: dict[str, Any],
    corpus: dict[int, dict[str, Any]],
) -> dict[int, bool]:
    exact(len(rows), len(corpus), f"{role}.rows")
    result = {}
    for index, raw in enumerate(rows):
        field = f"{role}[{index}]"
        row = validate_row(
            raw,
            role,
            acquisition_id,
            "output",
            {
                "item_index",
                "model_id",
                "model_sha256",
                "prompt_sha256",
                "raw_output",
            },
            field,
        )
        exact(row["item_index"], index, f"{field}.item_index")
        exact(row["model_id"], model["model_id"], f"{field}.model_id")
        exact(row["model_sha256"], model["artifact"]["sha256"], f"{field}.model_sha")
        exact(
            row["prompt_sha256"],
            corpus[index]["prompt_sha256"],
            f"{field}.prompt_sha256",
        )
        output = string(row["raw_output"], f"{field}.raw_output", allow_empty=True)
        answer = parse_answer(output)
        require(answer is not None, f"E_QUALITY_PARSE: {field}")
        result[index] = answer == corpus[index]["expected"]
    return result


def derive_quality(
    cuda_rows: list[dict[str, Any]],
    phone_rows: list[dict[str, Any]],
    cuda_role: str,
    phone_role: str,
    acquisition_id: str,
    model: dict[str, Any],
    corpus: dict[int, dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, int]:
    cuda = derive_quality_output(
        cuda_rows, cuda_role, acquisition_id, model, corpus
    )
    phone = derive_quality_output(
        phone_rows, phone_role, acquisition_id, model, corpus
    )
    new_errors = sum(cuda[index] and not phone[index] for index in corpus)
    recovered = sum(not cuda[index] and phone[index] for index in corpus)
    cuda_correct = sum(cuda.values())
    phone_correct = sum(phone.values())
    require(
        new_errors <= contract["gates"]["quality_maximum_new_errors"],
        f"E_QUALITY_NEW_ERRORS: {model['model_id']}",
    )
    require(
        cuda_correct - phone_correct
        <= contract["gates"]["quality_maximum_score_regression_items"],
        f"E_QUALITY_SCORE: {model['model_id']}",
    )
    return {
        "cuda_correct": cuda_correct,
        "phone_correct": phone_correct,
        "new_errors": new_errors,
        "recovered_errors": recovered,
    }


def derive_bridge(
    rows: list[dict[str, Any]],
    role: str,
    acquisition_id: str,
    model: dict[str, Any],
    clock_id: str,
    contract: dict[str, Any],
) -> dict[str, int]:
    require(len(rows) >= 10, f"E_ROWS: {role}")
    start = validate_row(
        rows[0],
        role,
        acquisition_id,
        "cuda_load_start",
        {"clock_id", "model_id", "model_sha256", "timestamp_ns"},
        f"{role}[0]",
    )
    ready = validate_row(
        rows[-1],
        role,
        acquisition_id,
        "cuda_ready",
        {"clock_id", "model_id", "model_sha256", "timestamp_ns"},
        f"{role}[{len(rows) - 1}]",
    )
    for record, label in ((start, "start"), (ready, "ready")):
        exact(record["clock_id"], clock_id, f"{role}.{label}.clock")
        exact(record["model_id"], model["model_id"], f"{role}.{label}.model")
        exact(record["model_sha256"], model["artifact"]["sha256"], f"{role}.{label}.sha")
        integer(record["timestamp_ns"], f"{role}.{label}.timestamp", 1)
    require(start["timestamp_ns"] < ready["timestamp_ns"], f"E_ORDER: {role}")
    requests = set()
    tokens = 0
    for index, raw in enumerate(rows[1:-1], start=1):
        field = f"{role}[{index}]"
        row = validate_row(
            raw,
            role,
            acquisition_id,
            "phone_publication_received",
            {
                "clock_id",
                "model_id",
                "model_sha256",
                "request_id",
                "timestamp_ns",
                "token_ids",
            },
            field,
        )
        exact(row["clock_id"], clock_id, f"{field}.clock")
        exact(row["model_id"], model["model_id"], f"{field}.model")
        exact(row["model_sha256"], model["artifact"]["sha256"], f"{field}.sha")
        request_id = integer(row["request_id"], f"{field}.request_id")
        require(request_id not in requests, f"E_BRIDGE_REQUEST_REUSE: {field}")
        requests.add(request_id)
        published = int_list(row["token_ids"], f"{field}.token_ids")
        timestamp = integer(row["timestamp_ns"], f"{field}.timestamp_ns", 1)
        require(
            start["timestamp_ns"] <= timestamp < ready["timestamp_ns"],
            f"E_BRIDGE_ORDER: {field}",
        )
        tokens += len(published)
    require(
        len(requests) >= contract["gates"]["bridge_minimum_requests"],
        f"E_BRIDGE_REQUESTS: {role}",
    )
    require(
        tokens >= contract["gates"]["bridge_minimum_tokens"],
        f"E_BRIDGE_TOKENS: {role}",
    )
    return {"requests": len(requests), "tokens": tokens}


def derive_placement(
    rows: list[dict[str, Any]],
    role: str,
    acquisition_id: str,
    model: dict[str, Any],
    phone: str,
    lock: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, int]:
    require(len(rows) >= 2, f"E_ROWS: {role}")
    meta = validate_row(
        rows[0],
        role,
        acquisition_id,
        "meta",
        {
            "available_after_bytes",
            "available_before_bytes",
            "batch",
            "boot_id",
            "device",
            "executed_layers",
            "model",
            "model_id",
            "model_sha256",
            "process_swap_bytes",
            "product",
            "serial",
            "shard_sha256",
            "stored_layers",
            "system_swap_after_bytes",
            "system_swap_before_bytes",
        },
        f"{role}[0]",
    )
    expected_device = contract["devices"][phone]
    for field in ("device", "model", "product", "serial"):
        exact(meta[field], expected_device[field], f"{role}.{field}")
    require(
        UUID_RE.fullmatch(string(meta["boot_id"], f"{role}.boot_id")) is not None,
        f"E_BOOT: {role}",
    )
    exact(meta["model_id"], model["model_id"], f"{role}.model_id")
    exact(meta["model_sha256"], model["artifact"]["sha256"], f"{role}.model_sha")
    exact(meta["batch"], 8, f"{role}.batch")
    exact(meta["stored_layers"], lock[f"{phone}_stored_layers"], f"{role}.stored")
    executed = (
        [0, lock["cut_layer"]]
        if phone == "op15"
        else [lock["cut_layer"], model["n_layer"]]
    )
    exact(meta["executed_layers"], executed, f"{role}.executed")
    exact(meta["shard_sha256"], lock[f"{phone}_shard_sha256"], f"{role}.shard")
    for field in (
        "available_after_bytes",
        "available_before_bytes",
        "process_swap_bytes",
        "system_swap_after_bytes",
        "system_swap_before_bytes",
    ):
        integer(meta[field], f"{role}.{field}")
    minimum = contract["gates"]["phone_minimum_available_bytes"]
    require(
        min(meta["available_before_bytes"], meta["available_after_bytes"]) >= minimum,
        f"E_PHONE_HEADROOM: {role}",
    )
    require(
        meta["process_swap_bytes"] <= contract["gates"]["maximum_process_swap_bytes"],
        f"E_PHONE_PROCESS_SWAP: {role}",
    )
    require(
        meta["system_swap_after_bytes"] - meta["system_swap_before_bytes"]
        <= contract["gates"]["maximum_system_swap_growth_bytes"],
        f"E_PHONE_SWAP: {role}",
    )
    node_ids = set()
    gpu_compute = 0
    for index, raw in enumerate(rows[1:], start=1):
        field = f"{role}[{index}]"
        row = validate_row(
            raw,
            role,
            acquisition_id,
            "node",
            {"backend", "compute", "missing_buffer", "node_id", "op"},
            field,
        )
        node_id = integer(row["node_id"], f"{field}.node_id")
        require(node_id not in node_ids, f"E_PLACEMENT_NODE_REUSE: {role}")
        node_ids.add(node_id)
        op = string(row["op"], f"{field}.op")
        require(type(row["compute"]) is bool, f"E_TYPE: {field}.compute")
        exact(row["missing_buffer"], False, f"{field}.missing_buffer")
        backend = string(row["backend"], f"{field}.backend")
        require(backend in ("GPUOpenCL", "CPU"), f"E_PLACEMENT_BACKEND: {field}")
        if backend == "CPU":
            require(op == "GET_ROWS", f"E_CPU_FALLBACK: {field}")
        if row["compute"] and backend == "GPUOpenCL":
            gpu_compute += 1
    require(gpu_compute > 0, f"E_PLACEMENT_EMPTY: {role}")
    return {"gpu_compute_nodes": gpu_compute, "nodes": len(node_ids)}


def derive_transfer(
    rows: list[dict[str, Any]],
    role: str,
    acquisition_id: str,
    model: dict[str, Any],
    lock: dict[str, Any],
) -> dict[str, int]:
    require(len(rows) >= 2, f"E_ROWS: {role}")
    meta = validate_row(
        rows[0],
        role,
        acquisition_id,
        "meta",
        {
            "batch",
            "cut_layer",
            "model_id",
            "model_sha256",
            "request_ids",
        },
        f"{role}[0]",
    )
    exact(meta["model_id"], model["model_id"], f"{role}.model_id")
    exact(meta["model_sha256"], model["artifact"]["sha256"], f"{role}.model_sha")
    exact(meta["batch"], 8, f"{role}.batch")
    exact(meta["cut_layer"], lock["cut_layer"], f"{role}.cut_layer")
    exact(meta["request_ids"], list(range(8)), f"{role}.request_ids")
    payload = 0
    for index, raw in enumerate(rows[1:], start=1):
        field = f"{role}[{index}]"
        row = validate_row(
            raw,
            role,
            acquisition_id,
            "transfer",
            {
                "host_payload_bytes",
                "path",
                "payload_bytes",
                "payload_sha256",
                "receiver",
                "sender",
            },
            field,
        )
        exact(row["sender"], "op15", f"{field}.sender")
        exact(row["receiver"], "op12", f"{field}.receiver")
        exact(row["path"], "WIFI_TCP_DIRECT", f"{field}.path")
        exact(row["host_payload_bytes"], 0, f"{field}.host_payload")
        payload += integer(row["payload_bytes"], f"{field}.payload_bytes", 1)
        digest(row["payload_sha256"], f"{field}.payload_sha256")
    return {"direct_payload_bytes": payload}


def derive_reprepare(
    rows: list[dict[str, Any]],
    role: str,
    acquisition_id: str,
    from_model: dict[str, Any],
    to_model: dict[str, Any],
    to_lock: dict[str, Any],
    clock_id: str,
    contract: dict[str, Any],
) -> dict[str, int]:
    require(len(rows) == 6, f"E_ROWS: {role}: expected six rows")
    start = validate_row(
        rows[0],
        role,
        acquisition_id,
        "start",
        {"clock_id", "from_model_id", "timestamp_ns", "to_model_id"},
        f"{role}[0]",
    )
    end = validate_row(
        rows[-1],
        role,
        acquisition_id,
        "end",
        {"clock_id", "from_model_id", "timestamp_ns", "to_model_id"},
        f"{role}[5]",
    )
    for record, label in ((start, "start"), (end, "end")):
        exact(record["clock_id"], clock_id, f"{role}.{label}.clock")
        exact(record["from_model_id"], from_model["model_id"], f"{role}.{label}.from")
        exact(record["to_model_id"], to_model["model_id"], f"{role}.{label}.to")
        integer(record["timestamp_ns"], f"{role}.{label}.timestamp", 1)
    require(start["timestamp_ns"] < end["timestamp_ns"], f"E_ORDER: {role}")

    before_by_phone = {}
    ready_by_phone = {}
    for index, raw in enumerate(rows[1:-1], start=1):
        field = f"{role}[{index}]"
        kind = raw.get("kind")
        require(kind in ("phone_before", "phone_ready"), f"E_KIND: {field}")
        if kind == "phone_before":
            row = validate_row(
                raw,
                role,
                acquisition_id,
                kind,
                {
                    "boot_id",
                    "local_ufs_read_bytes",
                    "network_weight_bytes",
                    "phone",
                    "ready_generation",
                    "serial",
                    "state_count",
                    "usb_weight_bytes",
                },
                field,
            )
            target = before_by_phone
        else:
            row = validate_row(
                raw,
                role,
                acquisition_id,
                kind,
                {
                    "boot_id",
                    "host_received_ns",
                    "local_ufs_read_bytes",
                    "model_sha256",
                    "network_weight_bytes",
                    "phone",
                    "ready_generation",
                    "serial",
                    "shard_sha256",
                    "state_count",
                    "usb_weight_bytes",
                },
                field,
            )
            target = ready_by_phone
        phone = string(row["phone"], f"{field}.phone")
        require(phone in ("op15", "op12"), f"E_PHONE: {field}")
        require(phone not in target, f"E_PHONE_REUSE: {field}")
        exact(row["serial"], contract["devices"][phone]["serial"], f"{field}.serial")
        require(
            UUID_RE.fullmatch(string(row["boot_id"], f"{field}.boot_id")) is not None,
            f"E_BOOT: {field}",
        )
        for counter in (
            "local_ufs_read_bytes",
            "network_weight_bytes",
            "ready_generation",
            "state_count",
            "usb_weight_bytes",
        ):
            integer(row[counter], f"{field}.{counter}")
        target[phone] = row
    exact(sorted(before_by_phone), ["op12", "op15"], f"{role}.before_phones")
    exact(sorted(ready_by_phone), ["op12", "op15"], f"{role}.ready_phones")
    ufs_delta = 0
    for phone in ("op15", "op12"):
        before = before_by_phone[phone]
        ready = ready_by_phone[phone]
        exact(ready["boot_id"], before["boot_id"], f"{role}.{phone}.boot_id")
        require(before["state_count"] > 0, f"E_RELEASE_UNEXERCISED: {role}.{phone}")
        exact(ready["state_count"], 0, f"{role}.{phone}.state_count")
        require(
            ready["ready_generation"] > before["ready_generation"],
            f"E_GENERATION: {role}.{phone}",
        )
        exact(ready["model_sha256"], to_model["artifact"]["sha256"], f"{role}.{phone}.model")
        exact(ready["shard_sha256"], to_lock[f"{phone}_shard_sha256"], f"{role}.{phone}.shard")
        host_received = integer(ready["host_received_ns"], f"{role}.{phone}.received", 1)
        require(
            start["timestamp_ns"] < host_received <= end["timestamp_ns"],
            f"E_REPREPARE_ORDER: {role}.{phone}",
        )
        current_ufs_delta = (
            ready["local_ufs_read_bytes"] - before["local_ufs_read_bytes"]
        )
        require(current_ufs_delta > 0, f"E_UFS: {role}.{phone}")
        ufs_delta += current_ufs_delta
        exact(
            ready["usb_weight_bytes"] - before["usb_weight_bytes"],
            0,
            f"{role}.{phone}.usb_weight_bytes",
        )
        exact(
            ready["network_weight_bytes"] - before["network_weight_bytes"],
            0,
            f"{role}.{phone}.network_weight_bytes",
        )
    elapsed_us = (end["timestamp_ns"] - start["timestamp_ns"] + 999) // 1000
    require(
        elapsed_us <= contract["gates"]["reprepare_maximum_elapsed_us"],
        f"E_REPREPARE_DWELL: {role}",
    )
    return {"elapsed_us": elapsed_us, "local_ufs_read_bytes": ufs_delta}


def evaluate(
    contract: dict[str, Any],
    contract_raw: bytes,
    candidate: dict[str, Any],
    candidate_raw: bytes,
    manifest: dict[str, Any],
    rows_by_role: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    validate_contract(contract)
    models, task_suite = validate_candidate(candidate, candidate_raw, contract)
    acquisition_id = manifest["acquisition_id"]
    clock_id = manifest["clock_id"]
    locks = {}
    for model in models:
        role = f"model.{model['model_id']}.route_lock"
        locks[model["model_id"]] = validate_route_lock(
            rows_by_role[role],
            role,
            acquisition_id,
            manifest["acquisition_started_ns"],
            clock_id,
            model,
            contract,
        )

    corpus = derive_corpus(
        rows_by_role["quality.corpus"],
        "quality.corpus",
        acquisition_id,
        task_suite,
        contract,
    )
    model_results = {}
    allocations = {}
    for model in models:
        model_id = model["model_id"]
        prefix = f"model.{model_id}"
        oracle = derive_oracle(rows_by_role, model, acquisition_id)
        allocations[model_id] = derive_cuda_memory(
            rows_by_role[f"{prefix}.cuda_memory"],
            f"{prefix}.cuda_memory",
            acquisition_id,
            model,
            contract,
        )
        quality = derive_quality(
            rows_by_role[f"{prefix}.quality.cuda"],
            rows_by_role[f"{prefix}.quality.phone"],
            f"{prefix}.quality.cuda",
            f"{prefix}.quality.phone",
            acquisition_id,
            model,
            corpus,
            contract,
        )
        bridge = derive_bridge(
            rows_by_role[f"{prefix}.bridge"],
            f"{prefix}.bridge",
            acquisition_id,
            model,
            clock_id,
            contract,
        )
        placement = {}
        for phone in ("op15", "op12"):
            role = f"{prefix}.placement.{phone}"
            placement[phone] = derive_placement(
                rows_by_role[role],
                role,
                acquisition_id,
                model,
                phone,
                locks[model_id],
                contract,
            )
        transfer = derive_transfer(
            rows_by_role[f"{prefix}.route_transfer"],
            f"{prefix}.route_transfer",
            acquisition_id,
            model,
            locks[model_id],
        )
        model_results[model_id] = {
            "bridge": bridge,
            "cuda_memory": allocations[model_id],
            "oracle": oracle,
            "placement": placement,
            "quality": quality,
            "transfer": transfer,
        }

    pair = derive_pair_memory(
        rows_by_role["pair.cuda_memory"],
        "pair.cuda_memory",
        acquisition_id,
        models,
        allocations,
        contract,
    )
    reprepare = {
        "A_to_B": derive_reprepare(
            rows_by_role["reprepare.A_to_B"],
            "reprepare.A_to_B",
            acquisition_id,
            models[0],
            models[1],
            locks[models[1]["model_id"]],
            clock_id,
            contract,
        ),
        "B_to_A": derive_reprepare(
            rows_by_role["reprepare.B_to_A"],
            "reprepare.B_to_A",
            acquisition_id,
            models[1],
            models[0],
            locks[models[0]["model_id"]],
            clock_id,
            contract,
        ),
    }
    return {
        "candidate_sha256": sha256_bytes(candidate_raw),
        "contract_sha256": sha256_bytes(contract_raw),
        "derived": {
            "models": model_results,
            "pair_cuda_memory": pair,
            "reprepare": reprepare,
        },
        "schema": "s39-cp0-r1-evidence-result-v2",
        "status": contract["claim_boundary"]["eligibility_status"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive CP0-R1 eligibility from role-tagged raw evidence"
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--manifest", default="EVIDENCE_BUNDLE.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract, contract_raw = load_canonical_path(args.contract)
        candidate, candidate_raw = load_canonical_path(args.candidate)
        validate_contract(contract)
        validate_candidate(candidate, candidate_raw, contract)
        if args.bundle_root is None:
            result = {
                "candidate_sha256": sha256_bytes(candidate_raw),
                "contract_sha256": sha256_bytes(contract_raw),
                "schema": "s39-cp0-r1-evidence-contract-check-v2",
                "status": "RAW_EVIDENCE_CONTRACT_READY_ACQUISITION_NOT_RUN",
            }
        else:
            manifest, rows = load_bundle(
                args.bundle_root,
                args.manifest,
                contract,
                contract_raw,
                candidate_raw,
            )
            result = evaluate(
                contract,
                contract_raw,
                candidate,
                candidate_raw,
                manifest,
                rows,
            )
        print(canonical_bytes(result).decode("ascii"), end="")
        return 0
    except (EvidenceError, OSError, KeyError) as exc:
        print(f"CP0_R1_V2_REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
