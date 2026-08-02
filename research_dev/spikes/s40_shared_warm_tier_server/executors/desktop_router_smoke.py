#!/usr/bin/env python3
"""Run B1 and B8 one-token-quanta smoke tests through a native router."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import sys
import time
from typing import Any

from desktop_gateway import (
    DesktopExecutor,
    SubprocessCacheController,
    UrllibTransport,
    parse_desktop_smoke_config,
    read_warm_tier_internal_token_from_env,
)
from phone_gateway import (
    COMMAND_CLEANUP,
    COMMAND_EXECUTE,
    COMMAND_LOAD,
    GatewayError,
    canonical_bytes,
    exact_keys,
    integer,
    require,
    sha256_text,
    strict_json_loads,
    string,
    tokens,
)


REQUEST_KEYS = {
    "arrival_us",
    "event_id",
    "input_tokens",
    "model_id",
    "output_tokens",
    "prompt_tokens",
    "request_index",
    "schema",
    "slo_us",
    "source_input_tokens",
    "source_model",
    "source_output_tokens",
    "source_t_us",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompts(
    path: Path,
    expected_sha256: str,
    model_id: str,
) -> list[list[int]]:
    require(path.is_absolute() and path.is_file(), "smoke request path")
    require(
        file_sha256(path) == sha256_text(
            expected_sha256,
            "smoke request SHA-256",
        ),
        "smoke request artifact changed",
    )
    result = []
    with path.open("rb") as source:
        for index, raw in enumerate(source):
            require(raw.endswith(b"\n"), f"smoke request[{index}] framing")
            row = strict_json_loads(raw, f"smoke request[{index}]")
            require(canonical_bytes(row) == raw, "smoke request canonical bytes")
            row = exact_keys(row, REQUEST_KEYS, f"smoke request[{index}]")
            if row["model_id"] != model_id:
                continue
            prompt = tokens(row["prompt_tokens"], "smoke request tokens")
            require(
                integer(row["output_tokens"], "smoke output tokens", 1) == 8,
                "smoke output token budget",
            )
            result.append(prompt)
    require(len(result) >= 8, "smoke requires eight model requests")
    return result[:8]


def request_command(
    executor_id: str,
    executor_instance_id: str,
    command_id: int,
    model_id: str,
    request_id: str,
    prompt: list[int],
    committed: list[int],
) -> dict[str, Any]:
    return {
        "command_id": command_id,
        "controller_epoch": 1,
        "executor_id": executor_id,
        "executor_instance_id": executor_instance_id,
        "kind": COMMAND_EXECUTE,
        "max_output_tokens": 1,
        "model_id": model_id,
        "request": {
            "committed_output_tokens": list(committed),
            "model_id": model_id,
            "owner_id": executor_id,
            "ownership_epoch": 1,
            "position": len(prompt) + len(committed),
            "prompt_tokens": list(prompt),
            "publication_index": len(committed),
            "request_id": request_id,
            "state": 1,
        },
        "request_id": request_id,
        "schema": "llama-server-warm-tier-command-v3",
        "total_output_tokens": 8,
    }


def cleanup_command(
    executor_id: str,
    executor_instance_id: str,
    command_id: int,
    model_id: str,
    request_id: str,
    prompt: list[int],
    committed: list[int],
) -> dict[str, Any]:
    value = request_command(
        executor_id,
        executor_instance_id,
        command_id,
        model_id,
        request_id,
        prompt,
        committed,
    )
    value["kind"] = COMMAND_CLEANUP
    value["max_output_tokens"] = 0
    value["total_output_tokens"] = 0
    return value


def load_command(
    executor_id: str,
    executor_instance_id: str,
    command_id: int,
    model_id: str,
) -> dict[str, Any]:
    return {
        "command_id": command_id,
        "controller_epoch": 1,
        "executor_id": executor_id,
        "executor_instance_id": executor_instance_id,
        "kind": COMMAND_LOAD,
        "max_output_tokens": 0,
        "model_id": model_id,
        "request": {
            "committed_output_tokens": [],
            "model_id": "",
            "owner_id": "",
            "ownership_epoch": 0,
            "position": 0,
            "prompt_tokens": [],
            "publication_index": 0,
            "request_id": "",
            "state": 0,
        },
        "request_id": "",
        "schema": "llama-server-warm-tier-command-v3",
        "total_output_tokens": 0,
    }


def run_geometry(
    executor: DesktopExecutor,
    executor_instance_id: str,
    model_id: str,
    prompts: list[list[int]],
    batch_size: int,
    first_command_id: int,
) -> tuple[dict[str, Any], int]:
    require(batch_size in (1, 8), "smoke batch size")
    selected = prompts[:batch_size]
    request_ids = [
        f"smoke-b{batch_size}-r{index}"
        for index in range(batch_size)
    ]
    outputs = [[] for _ in range(batch_size)]
    command_id = first_command_id
    rounds = []
    started_ns = time.monotonic_ns()
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        for step in range(8):
            round_started_ns = time.monotonic_ns()
            commands = []
            for index in range(batch_size):
                commands.append(request_command(
                    executor.executor_id,
                    executor_instance_id,
                    command_id,
                    model_id,
                    request_ids[index],
                    selected[index],
                    outputs[index],
                ))
                command_id += 1
            futures = [
                pool.submit(executor.handle, command)
                for command in commands
            ]
            results = [future.result() for future in futures]
            round_completed_ns = time.monotonic_ns()
            for index, (command, result) in enumerate(zip(commands, results)):
                require(
                    result["success"] is True
                    and len(result["publications"]) == 1,
                    "smoke execute result",
                )
                publication = result["publications"][0]
                require(
                    publication["position"]
                    == len(selected[index]) + step,
                    "smoke publication position",
                )
                outputs[index].append(
                    integer(publication["token"], "smoke token")
                )
                evidence = executor.take_execute_evidence(
                    command["command_id"]
                )
                require(evidence is not None, "smoke execute evidence")
                require(
                    evidence["execute_quantum_tokens"] == 1
                    and evidence["full_history_per_token_reprefill"] is False
                    and evidence["publication_count"] == 1
                    and evidence["resident_session_reused"] is (step > 0),
                    "smoke one-token resident execution",
                )
            rounds.append({
                "completed_ns": round_completed_ns,
                "round": step,
                "started_ns": round_started_ns,
            })
    cleanup = []
    for index in range(batch_size):
        command = cleanup_command(
            executor.executor_id,
            executor_instance_id,
            command_id,
            model_id,
            request_ids[index],
            selected[index],
            outputs[index],
        )
        command_id += 1
        result = executor.handle(command)
        require(result["success"] is True, "smoke cleanup")
        cleanup.append({
            "command_id": command["command_id"],
            "request_id": request_ids[index],
        })
    completed_ns = time.monotonic_ns()
    return {
        "batch_size": batch_size,
        "cleanup": cleanup,
        "completed_ns": completed_ns,
        "output_tokens": outputs,
        "rounds": rounds,
        "started_ns": started_ns,
    }, command_id


def validate_cleanup_evidence(
    value: dict[str, Any],
    expected_model_id: str | None,
) -> dict[str, Any]:
    value = exact_keys(
        value,
        {
            "completed_ns",
            "initial_active_models",
            "initial_busy_requests",
            "initial_request_sessions",
            "problems",
            "remaining_active_models",
            "remaining_request_sessions",
            "schema",
            "started_ns",
            "success",
            "unloaded",
        },
        "smoke cleanup evidence",
    )
    require(
        value["schema"] == "s40-desktop-cleanup-evidence-v1"
        and value["success"] is True
        and value["problems"] == []
        and value["initial_busy_requests"] == []
        and value["initial_request_sessions"] == []
        and value["remaining_active_models"] == []
        and value["remaining_request_sessions"] == [],
        "smoke terminal cleanup failed",
    )
    started_ns = integer(
        value["started_ns"],
        "smoke cleanup start",
        1,
    )
    require(
        integer(
            value["completed_ns"],
            "smoke cleanup completion",
            started_ns,
        ) >= started_ns,
        "smoke cleanup interval",
    )
    initial_models = value["initial_active_models"]
    unloaded = value["unloaded"]
    require(
        type(initial_models) is list
        and all(type(model_id) is str and model_id for model_id in initial_models)
        and initial_models == sorted(set(initial_models))
        and type(unloaded) is list
        and len(unloaded) == len(initial_models),
        "smoke cleanup model accounting",
    )
    seen_models = set()
    for index, row in enumerate(unloaded):
        row = exact_keys(
            row,
            {
                "instance_id",
                "logical_model_id",
                "native_model_id",
                "process_exited",
                "process_id",
                "process_start_ticks",
            },
            f"smoke cleanup unloaded[{index}]",
        )
        logical_model_id = string(
            row["logical_model_id"],
            "smoke cleanup logical model",
        )
        require(
            logical_model_id in initial_models
            and logical_model_id not in seen_models
            and string(
                row["native_model_id"],
                "smoke cleanup native model",
            )
            and string(row["instance_id"], "smoke cleanup instance")
            and integer(row["process_id"], "smoke cleanup process", 1) > 0
            and integer(
                row["process_start_ticks"],
                "smoke cleanup process start",
                1,
            ) > 0
            and row["process_exited"] is True,
            "smoke cleanup process exit",
        )
        seen_models.add(logical_model_id)
    if expected_model_id is not None:
        require(
            initial_models == [expected_model_id]
            and seen_models == {expected_model_id},
            "smoke cleanup did not unload the tested model",
        )
    return value


def run_smoke_session(
    executor: DesktopExecutor,
    executor_instance_id: str,
    model_id: str,
    prompts: list[list[int]],
) -> dict[str, Any]:
    executor_instance_id = string(
        executor_instance_id,
        "smoke executor instance ID",
    )
    load_id = 1
    result: dict[str, Any] | None = None
    primary_error: BaseException | None = None
    try:
        loaded = executor.handle(
            load_command(
                executor.executor_id,
                executor_instance_id,
                load_id,
                model_id,
            )
        )
        require(
            type(loaded) is dict and loaded.get("success") is True,
            "smoke load failed",
        )
        load_evidence = executor.take_lifecycle_evidence(load_id)
        require(
            type(load_evidence) is dict
            and load_evidence.get("operation") == "LOAD",
            "smoke load evidence missing",
        )
        b1, next_id = run_geometry(
            executor,
            executor_instance_id,
            model_id,
            prompts,
            1,
            load_id + 1,
        )
        b8, next_id = run_geometry(
            executor,
            executor_instance_id,
            model_id,
            prompts,
            8,
            next_id,
        )
        result = {
            "geometries": [b1, b8],
            "last_command_id": next_id - 1,
            "load": {
                "command_id": load_id,
                "evidence": load_evidence,
                "result": loaded,
            },
            "runtime_inventory": executor.runtime_inventory(),
        }
    except BaseException as error:
        primary_error = error

    cleanup: dict[str, Any] | None = None
    cleanup_error: BaseException | None = None
    try:
        cleanup = executor.close()
        validate_cleanup_evidence(
            cleanup,
            model_id if result is not None else None,
        )
    except BaseException as error:
        cleanup_error = error

    if primary_error is not None:
        if cleanup_error is not None:
            raise GatewayError(
                "smoke execution and terminal cleanup both failed: "
                f"{type(primary_error).__name__}: {primary_error}; "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            ) from primary_error
        raise primary_error
    if cleanup_error is not None:
        raise cleanup_error
    require(
        result is not None and cleanup is not None,
        "smoke result missing",
    )
    result["cleanup_evidence"] = cleanup
    return result


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    require(path.is_absolute() and not path.exists(), "smoke output path")
    with path.open("xb", buffering=0) as sink:
        sink.write(canonical_bytes(value))
        sink.flush()
        os.fsync(sink.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--requests-sha256", required=True)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()
    try:
        require(
            args.config.is_absolute()
            and args.requests.is_absolute()
            and args.output.is_absolute()
            and args.timeout > 0,
            "smoke absolute paths and timeout",
        )
        (
            executor_id,
            role,
            mode,
            base_url,
            cache_regime,
            routes,
            profile_lock_sha256,
            nvidia_smi,
            config_sha256,
        ) = parse_desktop_smoke_config(args.config)
        require(
            role in ("GPU", "CPU")
            and mode == "SINGLE_ACTIVE"
            and profile_lock_sha256 is None
            and args.model_id in routes
            and len(routes[args.model_id].slots) >= 8,
            "smoke desktop route geometry",
        )
        prompts = load_prompts(
            args.requests,
            args.requests_sha256,
            args.model_id,
        )
        executor = DesktopExecutor(
            executor_id,
            role,
            mode,
            routes,
            UrllibTransport(
                base_url,
                args.timeout,
                read_warm_tier_internal_token_from_env(),
            ),
            (),
            SubprocessCacheController(args.timeout),
            cache_regime,
            lifecycle_timeout_s=args.timeout,
            nvidia_smi=nvidia_smi,
        )
        smoke = run_smoke_session(
            executor,
            f"desktop-smoke-{os.getpid()}-{time.monotonic_ns()}",
            args.model_id,
            prompts,
        )
        output = {
            "cleanup_evidence": smoke["cleanup_evidence"],
            "config_sha256": config_sha256,
            "executor_id": executor_id,
            "geometries": smoke["geometries"],
            "last_command_id": smoke["last_command_id"],
            "load": smoke["load"],
            "model_id": args.model_id,
            "model_sha256": routes[args.model_id].model_sha256,
            "requests_sha256": args.requests_sha256,
            "runtime_inventory": smoke["runtime_inventory"],
            "schema": "s40-desktop-router-smoke-v1",
            "status": "B1_B8_SMOKE_PASS",
        }
        write_exclusive(args.output, output)
        print(canonical_bytes(output).decode("ascii"), end="")
        return 0
    except (GatewayError, OSError, TimeoutError) as error:
        print(f"desktop router smoke failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
