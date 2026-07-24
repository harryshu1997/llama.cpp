#!/usr/bin/env python3

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_INTENTS = HERE / "bundle_frequent" / "replay_intents.jsonl"
DEFAULT_REPLAY_MANIFEST = HERE / "bundle_frequent" / "replay_manifest.json"
DEFAULT_READINESS = HERE / "CURRENT_ROUTE_READINESS.json"

SHA256_RE = re.compile(r"[0-9a-f]{64}")
MODEL_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")

INTENT_KEYS = {
    "cycle_index",
    "from_model_id",
    "intent_id",
    "kind",
    "schema_version",
    "sequence",
    "source_event_id",
    "t_us",
    "to_model_id",
    "window_id",
}
ROUTE_KEYS = {
    "backend",
    "cut_layer",
    "evidence",
    "model_sha256",
    "reason",
    "status",
}
READINESS_KEYS = {"routes", "schema_version"}
REPLAY_MANIFEST_KEYS = {
    "active_bundle",
    "active_profile",
    "builder_version",
    "canonical_models",
    "derived",
    "inputs",
    "output",
    "policy",
    "schema_version",
    "semantics",
}
REPLAY_INPUT_KEYS = {
    "active_trace_sha256",
    "bundle_manifest_sha256",
    "model_assignment_sha256",
    "replay_builder_sha256",
    "requests_sha256",
    "shard_manifest_sha256",
}
REPLAY_POLICY_KEYS = {
    "cold_trigger_gap_us",
    "minimum_request_tokens",
    "minimum_target_dwell_us",
    "source",
}
REPLAY_DERIVED_KEYS = {
    "first_intent_us",
    "last_intent_us",
    "promotion_windows",
    "target_changes",
    "trace_horizon_us",
}


class ControllerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ControllerError(message)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_int(value: Any, field: str, minimum: int = 0) -> int:
    require(is_int(value), f"{field}: expected integer")
    require(value >= minimum, f"{field}: expected >= {minimum}")
    return value


def require_string(value: Any, field: str) -> str:
    require(isinstance(value, str) and bool(value), f"{field}: expected string")
    return value


def require_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field}: expected object")
    actual = set(value)
    require(
        actual == expected,
        f"{field}: keys differ; missing={sorted(expected - actual)}, "
        f"unknown={sorted(actual - expected)}",
    )
    return value


def reject_constant(value: str) -> None:
    raise ControllerError(f"invalid JSON constant {value}")


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_json(raw: bytes, field: str) -> Any:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ControllerError(f"{field}: expected ASCII") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ControllerError(f"{field}: invalid JSON: {error}") from error


def read_bytes(path: Path, field: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ControllerError(f"{field}: cannot read {path}: {error}") from error


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def validate_digest(value: Any, field: str) -> str:
    value = require_string(value, field)
    require(SHA256_RE.fullmatch(value) is not None, f"{field}: invalid SHA-256")
    return value


def load_intents(
    intents_path: Path,
    replay_manifest_path: Path,
) -> list[dict[str, Any]]:
    manifest_raw = read_bytes(replay_manifest_path, "replay_manifest")
    manifest = require_keys(
        parse_json(manifest_raw, "replay_manifest"),
        REPLAY_MANIFEST_KEYS,
        "replay_manifest",
    )
    require(
        canonical_bytes(manifest) == manifest_raw,
        "replay_manifest: not canonical JSON",
    )
    require_int(manifest["schema_version"], "replay_manifest.schema_version")
    require(manifest["schema_version"] == 1, "replay_manifest: unsupported version")
    require(
        manifest["builder_version"] == "s39-active-trace-replay-v1",
        "replay_manifest: unsupported builder",
    )
    require(
        require_string(manifest["active_bundle"], "replay_manifest.active_bundle")
        == replay_manifest_path.parent.name,
        "replay_manifest: active bundle mismatch",
    )
    require(
        manifest["active_profile"] == "frequent_switch",
        "replay_manifest: active profile mismatch",
    )
    require(
        manifest["semantics"] == "policy_intents_only_no_readiness_or_execution_claim",
        "replay_manifest: invalid semantics",
    )

    canonical_models = require_keys(
        manifest["canonical_models"],
        {"cold", "hot"},
        "replay_manifest.canonical_models",
    )
    for role in ("cold", "hot"):
        model_id = require_string(
            canonical_models[role],
            f"replay_manifest.canonical_models.{role}",
        )
        require(
            MODEL_ID_RE.fullmatch(model_id) is not None,
            f"replay_manifest.canonical_models.{role}: invalid model ID",
        )
    require(
        canonical_models["cold"] != canonical_models["hot"],
        "replay_manifest: canonical models must differ",
    )

    inputs = require_keys(
        manifest["inputs"],
        REPLAY_INPUT_KEYS,
        "replay_manifest.inputs",
    )
    for name, digest in inputs.items():
        validate_digest(digest, f"replay_manifest.inputs.{name}")

    policy = require_keys(
        manifest["policy"],
        REPLAY_POLICY_KEYS,
        "replay_manifest.policy",
    )
    for field in (
        "cold_trigger_gap_us",
        "minimum_request_tokens",
        "minimum_target_dwell_us",
    ):
        require_int(policy[field], f"replay_manifest.policy.{field}")
    require_string(policy["source"], "replay_manifest.policy.source")

    derived = require_keys(
        manifest["derived"],
        REPLAY_DERIVED_KEYS,
        "replay_manifest.derived",
    )
    for field in REPLAY_DERIVED_KEYS:
        require_int(derived[field], f"replay_manifest.derived.{field}")

    output = require_keys(
        manifest["output"],
        {"bytes", "path", "records", "sha256"},
        "replay_manifest.output",
    )
    require(
        require_string(output["path"], "replay_manifest.output.path")
        == intents_path.name,
        "replay_manifest: output path mismatch",
    )
    expected_bytes = require_int(output["bytes"], "replay_manifest.output.bytes")
    expected_records = require_int(output["records"], "replay_manifest.output.records")
    expected_sha = validate_digest(output["sha256"], "replay_manifest.output.sha256")

    raw = read_bytes(intents_path, "intents")
    require(len(raw) == expected_bytes, "intents: byte count mismatch")
    require(sha256(raw) == expected_sha, "intents: SHA-256 mismatch")
    require(raw.endswith(b"\n"), "intents: missing final newline")
    require(raw.count(b"\n") == expected_records, "intents: record count mismatch")

    records = []
    previous_t = -1
    previous_sequence = -1
    for index, line in enumerate(raw.splitlines()):
        record = require_keys(parse_json(line, f"intent[{index}]"), INTENT_KEYS, f"intent[{index}]")
        require(canonical_bytes(record).rstrip(b"\n") == line, f"intent[{index}]: not canonical")
        require_int(record["schema_version"], f"intent[{index}].schema_version")
        require(record["schema_version"] == 1, f"intent[{index}]: unsupported version")
        sequence = require_int(record["sequence"], f"intent[{index}].sequence")
        require(sequence == previous_sequence + 1, f"intent[{index}]: sequence gap")
        t_us = require_int(record["t_us"], f"intent[{index}].t_us")
        require(t_us > previous_t, f"intent[{index}]: nonmonotonic time")
        require(record["kind"] in {"PROMOTE", "DEMOTE"}, f"intent[{index}]: invalid kind")
        for field in ("from_model_id", "to_model_id"):
            model_id = require_string(record[field], f"intent[{index}].{field}")
            require(MODEL_ID_RE.fullmatch(model_id) is not None, f"intent[{index}]: invalid model ID")
        require(
            record["from_model_id"] != record["to_model_id"],
            f"intent[{index}]: identical source and target",
        )
        previous_t = t_us
        previous_sequence = sequence
        records.append(record)

    require(bool(records), "intents: empty")
    require(
        len(records) == derived["target_changes"],
        "replay_manifest: target change count mismatch",
    )
    require(
        records[0]["t_us"] == derived["first_intent_us"],
        "replay_manifest: first intent mismatch",
    )
    require(
        records[-1]["t_us"] == derived["last_intent_us"],
        "replay_manifest: last intent mismatch",
    )
    require(
        sum(record["kind"] == "PROMOTE" for record in records)
        == derived["promotion_windows"],
        "replay_manifest: promotion count mismatch",
    )
    require(
        {record["from_model_id"] for record in records}
        | {record["to_model_id"] for record in records}
        == set(canonical_models.values()),
        "replay_manifest: canonical model mismatch",
    )
    require(
        records[0]["from_model_id"] == canonical_models["hot"]
        and records[0]["to_model_id"] == canonical_models["cold"],
        "replay_manifest: initial model roles mismatch",
    )
    return records


def load_readiness(path: Path) -> dict[str, dict[str, Any]]:
    root = require_keys(parse_json(read_bytes(path, "readiness"), "readiness"), READINESS_KEYS, "readiness")
    require_int(root["schema_version"], "readiness.schema_version")
    require(root["schema_version"] == 1, "readiness: unsupported version")
    routes = root["routes"]
    require(isinstance(routes, dict) and bool(routes), "readiness.routes: expected nonempty object")

    result = {}
    for model_id, value in routes.items():
        require(
            isinstance(model_id, str) and MODEL_ID_RE.fullmatch(model_id) is not None,
            f"readiness.routes: invalid model ID {model_id!r}",
        )
        route = require_keys(value, ROUTE_KEYS, f"readiness.routes.{model_id}")
        require(
            route["status"]
            in {
                "PASS",
                "PROVISIONAL_B1",
                "PROVISIONAL_BATCH",
                "FAIL_CORRECTNESS",
                "BLOCKED",
            },
            f"readiness.routes.{model_id}: invalid status",
        )
        validate_digest(route["model_sha256"], f"readiness.routes.{model_id}.model_sha256")
        require_string(route["backend"], f"readiness.routes.{model_id}.backend")
        require_int(route["cut_layer"], f"readiness.routes.{model_id}.cut_layer")
        require_string(route["reason"], f"readiness.routes.{model_id}.reason")
        evidence = route["evidence"]
        require(isinstance(evidence, dict) and bool(evidence), f"readiness.routes.{model_id}.evidence")
        for name, digest in evidence.items():
            require_string(name, f"readiness.routes.{model_id}.evidence key")
            validate_digest(digest, f"readiness.routes.{model_id}.evidence.{name}")
        result[model_id] = route
    return result


class WarmTierController:
    def __init__(
        self,
        *,
        gpu_model: str,
        edge_model: str,
        readiness: dict[str, dict[str, Any]],
    ) -> None:
        require(gpu_model != edge_model, "initial GPU and edge models must differ")
        require(gpu_model in readiness, "initial GPU model missing from readiness")
        require(edge_model in readiness, "initial edge model missing from readiness")
        self.gpu_model = gpu_model
        self.edge_model = edge_model
        self.readiness = copy.deepcopy(readiness)
        self.phase = "STABLE"
        self.transition_epoch = 0
        self.source_model: str | None = None
        self.target_model: str | None = None
        self.last_sequence = -1
        self.last_t_us = -1
        self.actions: list[dict[str, Any]] = []

    def snapshot(self) -> dict[str, Any]:
        return {
            "gpu_model": self.gpu_model,
            "edge_model": self.edge_model,
            "phase": self.phase,
            "transition_epoch": self.transition_epoch,
            "source_model": self.source_model,
            "target_model": self.target_model,
            "last_sequence": self.last_sequence,
            "last_t_us": self.last_t_us,
            "actions": copy.deepcopy(self.actions),
        }

    def _append(self, kind: str, **fields: Any) -> dict[str, Any]:
        action = {
            "action_index": len(self.actions),
            "kind": kind,
            "transition_epoch": self.transition_epoch,
            **fields,
        }
        self.actions.append(action)
        return action

    def submit_intent(self, intent: dict[str, Any]) -> dict[str, Any]:
        before = self.snapshot()
        try:
            require(self.phase == "STABLE", "E_TRANSITION_BUSY")
            require(intent["sequence"] == self.last_sequence + 1, "E_INTENT_SEQUENCE")
            require(intent["t_us"] > self.last_t_us, "E_INTENT_TIME")
            require(intent["from_model_id"] == self.gpu_model, "E_SOURCE_NOT_HOT")
            require(intent["to_model_id"] == self.edge_model, "E_TARGET_NOT_WARM")
            for model_id in (intent["from_model_id"], intent["to_model_id"]):
                route = self.readiness.get(model_id)
                require(route is not None, f"E_ROUTE_MISSING:{model_id}")
                require(route["status"] == "PASS", f"E_ROUTE_NOT_READY:{model_id}:{route['status']}")

            self.transition_epoch += 1
            self.source_model = intent["from_model_id"]
            self.target_model = intent["to_model_id"]
            self.last_sequence = intent["sequence"]
            self.last_t_us = intent["t_us"]
            self.phase = "DRAINING_GPU"
            return self._append(
                "DRAIN_GPU",
                intent_id=intent["intent_id"],
                model_id=self.source_model,
                edge_serving_model=self.target_model,
            )
        except Exception:
            self.__dict__.update(before)
            raise

    def gpu_drained(self) -> dict[str, Any]:
        require(self.phase == "DRAINING_GPU", "E_PHASE_GPU_DRAINED")
        self.phase = "LOADING_GPU"
        return self._append("LOAD_GPU", model_id=self.target_model)

    def gpu_ready(self) -> dict[str, Any]:
        require(self.phase == "LOADING_GPU", "E_PHASE_GPU_READY")
        self.phase = "CATCHING_UP"
        return self._append("BATCH_CATCHUP", model_id=self.target_model)

    def catchup_committed(self) -> dict[str, Any]:
        require(self.phase == "CATCHING_UP", "E_PHASE_CATCHUP")
        self.gpu_model = str(self.target_model)
        self.phase = "DRAINING_EDGE"
        return self._append("DRAIN_EDGE", model_id=self.target_model)

    def edge_drained(self) -> dict[str, Any]:
        require(self.phase == "DRAINING_EDGE", "E_PHASE_EDGE_DRAINED")
        self.phase = "PREPARING_EDGE"
        return self._append("PREPARE_EDGE", model_id=self.source_model)

    def edge_ready(self) -> dict[str, Any]:
        require(self.phase == "PREPARING_EDGE", "E_PHASE_EDGE_READY")
        self.edge_model = str(self.source_model)
        self.phase = "STABLE"
        action = self._append("PUBLISH_EDGE_READY", model_id=self.edge_model)
        self.source_model = None
        self.target_model = None
        return action


def check_trace_ready(
    intents: list[dict[str, Any]],
    readiness: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    models = sorted(
        {record[field] for record in intents for field in ("from_model_id", "to_model_id")}
    )
    blocked = {
        model_id: readiness.get(model_id, {}).get("status", "MISSING")
        for model_id in models
        if readiness.get(model_id, {}).get("status") != "PASS"
    }
    require(not blocked, "E_ROUTE_NOT_READY:" + ",".join(f"{key}={value}" for key, value in blocked.items()))
    return {
        "schema_version": 1,
        "status": "READY_FOR_PHYSICAL_REPLAY",
        "intent_count": len(intents),
        "models": models,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intents", type=Path, default=DEFAULT_INTENTS)
    parser.add_argument("--replay-manifest", type=Path, default=DEFAULT_REPLAY_MANIFEST)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    args = parser.parse_args()
    try:
        intents = load_intents(args.intents, args.replay_manifest)
        readiness = load_readiness(args.readiness)
        print(canonical_bytes(check_trace_ready(intents, readiness)).decode("ascii"), end="")
        return 0
    except ControllerError as error:
        print(f"warm_tier_controller: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
