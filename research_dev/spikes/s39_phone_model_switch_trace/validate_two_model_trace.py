#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPLAY_BUILDER = HERE / "build_replay_intents.py"
DEFAULT_OUTPUT = HERE / "TWO_MODEL_TRACE_CONTRACT.json"


class TraceContractError(RuntimeError):
    pass


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise TraceContractError(f"{code}: {message}")


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


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_replay_builder():
    spec = importlib.util.spec_from_file_location("s39_replay_builder", REPLAY_BUILDER)
    require(spec is not None and spec.loader is not None, "E_IMPORT", "replay builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_for_request(
    request: dict[str, Any],
    source_to_model: dict[str, str],
) -> str:
    source_model = request["source_fields"]["model"]
    require(
        source_model in source_to_model,
        "E_REQUEST_MODEL",
        f"unknown source model {source_model!r}",
    )
    return source_to_model[source_model]


def derive_contract(
    *,
    requests: list[dict[str, Any]],
    mappings: dict[str, Any],
    intents: list[dict[str, Any]],
    hot_model: str,
    cold_model: str,
    horizon_us: int,
    input_digests: dict[str, str],
) -> dict[str, Any]:
    require(hot_model != cold_model, "E_MODEL_SET", "hot and cold models match")
    canonical_models = {hot_model, cold_model}
    source_to_model = {
        source_model: mapping["model_id"]
        for source_model, mapping in mappings.items()
    }
    require(
        set(source_to_model.values()) == canonical_models,
        "E_MODEL_SET",
        "assignment does not map exactly two canonical models",
    )

    successful = [request for request in requests if request["output_tokens"] > 0]
    request_by_id = {request["event_id"]: request for request in successful}
    require(
        len(request_by_id) == len(successful),
        "E_REQUEST_ID",
        "successful request IDs are not unique",
    )

    model_requests = {
        model: [
            request
            for request in successful
            if model_for_request(request, source_to_model) == model
        ]
        for model in sorted(canonical_models)
    }
    require(
        all(model_requests[model] for model in canonical_models),
        "E_TWO_MODEL_REQUESTS",
        "both models need successful requests",
    )
    require(intents, "E_INTENTS", "switch intent list is empty")

    current_gpu = hot_model
    current_phone = cold_model
    phone_witnesses = {model: [] for model in canonical_models}
    cuda_witnesses = {model: [] for model in canonical_models}
    transitions = []

    for index, intent in enumerate(intents):
        require(
            intent["sequence"] == index,
            "E_INTENT_SEQUENCE",
            f"intent {index} has sequence {intent['sequence']!r}",
        )
        require(
            intent["from_model_id"] == current_gpu
            and intent["to_model_id"] == current_phone,
            "E_RESIDENCY_CHAIN",
            f"intent {index} does not match current hot/warm placement",
        )
        trigger_id = intent["source_event_id"]
        require(
            trigger_id in request_by_id,
            "E_TRIGGER_REQUEST",
            f"intent {index} trigger is not a successful request",
        )
        trigger = request_by_id[trigger_id]
        trigger_model = model_for_request(trigger, source_to_model)
        require(
            trigger_model == current_phone,
            "E_TRIGGER_MODEL",
            f"intent {index} trigger is not for the phone-warm model",
        )
        require(
            trigger["t_us"] == intent["t_us"],
            "E_TRIGGER_TIME",
            f"intent {index} time does not match its trigger request",
        )
        phone_witnesses[current_phone].append(trigger_id)

        next_intent_us = (
            intents[index + 1]["t_us"] if index + 1 < len(intents) else horizon_us
        )
        require(
            intent["t_us"] < next_intent_us <= horizon_us,
            "E_INTENT_WINDOW",
            f"intent {index} has an invalid target-hot window",
        )
        hot_candidates = [
            request
            for request in model_requests[current_phone]
            if intent["t_us"] < request["t_us"] < next_intent_us
        ]
        cuda_witness_id = hot_candidates[-1]["event_id"] if hot_candidates else None
        if cuda_witness_id is not None:
            cuda_witnesses[current_phone].append(cuda_witness_id)

        transitions.append(
            {
                "cuda_witness_event_id": cuda_witness_id,
                "from_gpu_model_id": current_gpu,
                "intent_id": intent["intent_id"],
                "kind": intent["kind"],
                "phone_trigger_event_id": trigger_id,
                "sequence": index,
                "t_us": intent["t_us"],
                "to_gpu_model_id": current_phone,
            }
        )
        current_gpu, current_phone = current_phone, current_gpu

    require(
        all(phone_witnesses[model] for model in canonical_models),
        "E_BIDIRECTIONAL_PHONE",
        "both models need a phone-side promotion trigger",
    )
    require(
        all(cuda_witnesses[model] for model in canonical_models),
        "E_BIDIRECTIONAL_CUDA",
        "both models need a post-cutover CUDA request witness",
    )
    require(
        {transition["kind"] for transition in transitions} == {"PROMOTE", "DEMOTE"},
        "E_BIDIRECTIONAL_SWITCH",
        "trace needs both promotion and demotion intents",
    )

    model_summary = {}
    for model in sorted(canonical_models):
        rows = model_requests[model]
        model_summary[model] = {
            "cuda_witness_event_ids": cuda_witnesses[model],
            "input_tokens": sum(request["input_tokens"] for request in rows),
            "output_tokens": sum(request["output_tokens"] for request in rows),
            "phone_trigger_event_ids": phone_witnesses[model],
            "successful_requests": len(rows),
        }

    return {
        "contract": "s39-two-model-physical-replay-v1",
        "event_order": "admit_trigger_to_current_phone_tier_before_same_time_intent",
        "final_planned_state": {
            "gpu_model_id": current_gpu,
            "phone_model_id": current_phone,
        },
        "initial_planned_state": {
            "gpu_model_id": hot_model,
            "phone_model_id": cold_model,
        },
        "inputs": dict(sorted(input_digests.items())),
        "model_summary": model_summary,
        "required_physical_gates": [
            "execute_every_successful_request_exactly_once",
            "execute_each_model_on_phones_and_cuda",
            "complete_every_residency_transition",
            "keep_phone_owner_during_cuda_load_and_catchup",
            "commit_each_ownership_change_at_one_token_frontier",
            "reject_stale_or_false_phone_readiness",
        ],
        "schema_version": 1,
        "status": "TRACE_CONTRACT_PASS_PHYSICAL_EXECUTION_NOT_RUN",
        "successful_requests": len(successful),
        "transition_count": len(transitions),
        "transitions": transitions,
    }


def build_contract(root: Path) -> dict[str, Any]:
    root = root.resolve()
    replay = load_replay_builder()
    selector_path = root / "ACTIVE_TRACE.json"
    shard_manifest_path = root / "SHARD_MANIFEST.json"

    with tempfile.TemporaryDirectory(prefix="s39_two_model_trace_") as directory:
        generated = Path(directory)
        replay_manifest = replay.build(
            root=root,
            selector_path=selector_path,
            shard_manifest_path=shard_manifest_path,
            output_dir=generated,
        )
        bundle = root / replay_manifest["active_bundle"]
        for name in ("replay_intents.jsonl", "replay_manifest.json"):
            expected = (bundle / name).read_bytes()
            actual = (generated / name).read_bytes()
            require(
                actual == expected,
                "E_FROZEN_REPLAY",
                f"{name} is stale relative to the active trace",
            )

    selector, selector_raw = replay.read_json(selector_path, "selector")
    selector = replay.validate_selector(selector)
    bundle = root / selector["active_bundle"]
    manifest, manifest_raw = replay.read_json(bundle / "manifest.json", "manifest")
    manifest = replay.validate_manifest(manifest, selector["active_profile"])
    _, requests_raw = replay.verify_output(
        bundle,
        manifest["outputs"]["requests"],
        "manifest.outputs.requests",
    )
    _, assignment_raw = replay.verify_output(
        bundle,
        manifest["outputs"]["model_assignment"],
        "manifest.outputs.model_assignment",
    )
    assignment = replay.parse_json_bytes(assignment_raw, "assignment")
    mappings = assignment["mappings"]
    requests = replay.parse_requests(
        requests_raw,
        mappings=mappings,
        duration_us=manifest["selection"]["window_us"],
    )
    intents = [
        replay.parse_json_bytes(line, f"intent[{index}]")
        for index, line in enumerate(
            (bundle / "replay_intents.jsonl").read_bytes().splitlines()
        )
    ]
    replay_manifest_raw = (bundle / "replay_manifest.json").read_bytes()
    shard_manifest_raw = shard_manifest_path.read_bytes()

    canonical_models = replay_manifest["canonical_models"]
    return derive_contract(
        requests=requests,
        mappings=mappings,
        intents=intents,
        hot_model=canonical_models["hot"],
        cold_model=canonical_models["cold"],
        horizon_us=replay_manifest["derived"]["trace_horizon_us"],
        input_digests={
            "active_trace_sha256": sha256(selector_raw),
            "assignment_sha256": sha256(assignment_raw),
            "manifest_sha256": sha256(manifest_raw),
            "replay_intents_sha256": sha256(
                (bundle / "replay_intents.jsonl").read_bytes()
            ),
            "replay_manifest_sha256": sha256(replay_manifest_raw),
            "requests_sha256": sha256(requests_raw),
            "shard_manifest_sha256": sha256(shard_manifest_raw),
            "validator_sha256": sha256(Path(__file__).read_bytes()),
        },
    )


def atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise TraceContractError(f"E_WRITE: cannot write {path}: {error}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the S39 two-model physical replay obligations"
    )
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract = build_contract(args.root)
        atomic_write(args.output, canonical_bytes(contract))
    except (OSError, TraceContractError) as error:
        print(f"S39_TWO_MODEL_TRACE_ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(contract, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
