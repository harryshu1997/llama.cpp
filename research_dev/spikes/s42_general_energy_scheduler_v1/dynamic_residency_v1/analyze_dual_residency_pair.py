#!/usr/bin/env python3
"""Validate one matched Qwen/Gemma dual-residency diagnostic pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CONTROL = (
    HERE / "results/RTX4060TI_QWEN18_GEMMA0_ENERGY_DIAGNOSTIC_V1"
)
DEFAULT_TREATMENT = (
    HERE / "results/RTX4060TI_QWEN15_GEMMA1_ENERGY_DIAGNOSTIC_V1"
)
EXPECTED_FILES = frozenset({
    "RESULT.json",
    "gemma-smoke.raw",
    "gemma-warm.raw",
    "gemma.stderr.log",
    "gemma.stdout.log",
    "qwen-smoke.raw",
    "qwen-warm.raw",
    "qwen.stderr.log",
    "qwen.stdout.log",
})
EXPECTED_CONFIGURATIONS = {
    "control": {
        "gemma_gpu_layers": 0,
        "gpu_reserve_bytes": 536_870_912,
        "qwen_gpu_layers": 18,
    },
    "treatment": {
        "gemma_gpu_layers": 1,
        "gpu_reserve_bytes": 536_870_912,
        "qwen_gpu_layers": 15,
    },
}
EXPECTED_INPUTS = {
    "gemma_model": {
        "sha256": (
            "ed76f2183d2d1d65091986033023e6c7"
            "8d27f6276c1b0c5826cc92acf73538cf"
        ),
        "size_bytes": 23_832_065_056,
    },
    "probe_source": {
        "sha256": (
            "d3be83d5910b9e6d8ca6b0f1553baf"
            "3165c34062114a83e99effbd9b115fdeab"
        ),
        "size_bytes": 26_698,
    },
    "qwen_model": {
        "sha256": (
            "d89e9e823744222e595e0b3c8fd5436c"
            "e5d3a6a446fa42492ebce6064dfa9718"
        ),
        "size_bytes": 29_543_423_360,
    },
    "server": {
        "sha256": (
            "d8433ed903da695fe8e884fd55f43442c"
            "f2417caec4363495da671b23dbda040"
        ),
        "size_bytes": 17_896,
    },
    "trace": {
        "sha256": (
            "b20a9ba66ee3558d835a0e19ed3cfa4"
            "c31a4a9e8b4f9c085b29a14f80250a0ff"
        ),
        "size_bytes": 204_556,
    },
}
EXPECTED_REQUESTS = {
    "gemma": {
        "input_tokens": 271,
        "model": "gemma4-12b-f16-proxy",
        "output_tokens": 41,
        "request_index": 50,
    },
    "qwen": {
        "input_tokens": 16,
        "model": "qwen3-14b-f16-proxy",
        "output_tokens": 9,
        "request_index": 52,
    },
}
GPU_NAME = "NVIDIA GeForce RTX 4060 Ti"
GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
GPU_TOTAL_BYTES = 17_175_674_880


class AnalysisError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def canonical(value: object) -> bytes:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise AnalysisError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    require(type(value) is dict, f"object: {path}")
    return value


def positive(name: str, value: object) -> float:
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and value > 0,
        f"positive {name}",
    )
    return float(value)


def validate_manifest(root: Path) -> dict[str, str]:
    path = root / "SHA256SUMS.txt"
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    entries: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        require(len(parts) == 2, f"manifest row: {path}")
        digest, name = parts
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and name not in entries
            and Path(name).name == name,
            f"manifest identity: {path}",
        )
        entries[name] = digest
    require(set(entries) == EXPECTED_FILES, f"manifest file set: {path}")
    for name, digest in entries.items():
        require(sha256(root / name) == digest, f"manifest digest: {name}")
    return entries


def validate_record_hash(result: dict[str, Any], path: Path) -> None:
    claimed = result.get("record_sha256")
    unsigned = dict(result)
    unsigned.pop("record_sha256", None)
    require(
        type(claimed) is str
        and hashlib.sha256(canonical(unsigned)).hexdigest() == claimed,
        f"canonical result hash: {path}",
    )


def validate_gpu(row: object, name: str) -> dict[str, Any]:
    require(type(row) is dict, f"GPU snapshot: {name}")
    require(
        row.get("name") == GPU_NAME
        and row.get("uuid") == GPU_UUID
        and row.get("memory_total_bytes") == GPU_TOTAL_BYTES,
        f"GPU identity: {name}",
    )
    positive(f"GPU free memory: {name}", row.get("memory_free_bytes"))
    return row


def command_value(command: object, option: str, name: str) -> str:
    require(
        type(command) is list
        and all(type(value) is str for value in command)
        and command.count(option) == 1,
        f"command option {option}: {name}",
    )
    index = command.index(option)
    require(index + 1 < len(command), f"command value {option}: {name}")
    return command[index + 1]


def parse_completion(
    path: Path,
    *,
    expected_input_tokens: int,
    expected_model: str,
    expected_output_tokens: int,
    expected_seed: int,
) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    tokens: list[int] = []
    finals: list[dict[str, Any]] = []
    chunks = 0
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"malformed completion chunk: {path}") from exc
        require(type(value) is dict, f"completion chunk: {path}")
        chunk = value.get("tokens", [])
        require(
            type(chunk) is list
            and all(type(token) is int for token in chunk),
            f"completion token row: {path}",
        )
        chunks += 1
        tokens.extend(chunk)
        if value.get("stop") is True:
            finals.append(value)
    require(
        chunks > 0
        and len(finals) == 1
        and len(tokens) == expected_output_tokens,
        f"completion geometry: {path}",
    )
    final = finals[0]
    timings = final.get("timings")
    settings = final.get("generation_settings")
    prompt = final.get("prompt")
    require(
        final.get("model") == expected_model
        and final.get("tokens_predicted") == expected_output_tokens
        and final.get("tokens_evaluated") == expected_input_tokens
        and type(timings) is dict
        and timings.get("prompt_n") == expected_input_tokens
        and timings.get("predicted_n") == expected_output_tokens
        and type(settings) is dict
        and settings.get("seed") == expected_seed
        and settings.get("max_tokens") == expected_output_tokens
        and settings.get("temperature") == 0.0
        and type(prompt) is str,
        f"completion binding: {path}",
    )
    return {
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "tokens": tokens,
        "tokens_sha256": hashlib.sha256(canonical(tokens)).hexdigest(),
    }


def validate_stage(
    stage: object,
    *,
    expected_layers: int,
    expected_total_layers: int,
    model_path: str,
    name: str,
) -> dict[str, Any]:
    require(type(stage) is dict, f"stage: {name}")
    allocation = stage.get("allocations")
    memory = stage.get("process_memory")
    require(
        type(allocation) is dict
        and allocation.get("offloaded_layers") == expected_layers
        and allocation.get("total_layers") == expected_total_layers
        and type(memory) is dict
        and positive(f"stage RSS: {name}", memory.get("rss_bytes")) > 0
        and memory.get("swap_bytes") == 0
        and positive(f"load time: {name}", stage.get("load_us")) > 0,
        f"stage placement: {name}",
    )
    command = stage.get("command")
    require(
        command_value(command, "--model", name) == model_path
        and int(command_value(command, "--n-gpu-layers", name))
            == expected_layers,
        f"stage command binding: {name}",
    )
    if expected_layers == 0:
        require(
            command_value(command, "--device", name) == "none",
            f"CPU-only device binding: {name}",
        )
    validate_gpu(stage.get("gpu"), name)
    return {
        "allocations": allocation,
        "load_us": stage["load_us"],
        "rss_bytes": memory["rss_bytes"],
        "swap_bytes": memory["swap_bytes"],
    }


def validate_arm(root: Path, arm: str) -> dict[str, Any]:
    manifest = validate_manifest(root)
    result_path = root / "RESULT.json"
    result = read_object(result_path)
    validate_record_hash(result, result_path)
    configuration = EXPECTED_CONFIGURATIONS[arm]
    require(
        result.get("schema") == "s42-dynamic-gpu-residency-capacity-v1"
        and result.get("status") == "EXECUTION_SMOKE_PASS"
        and result.get("measurement_scope")
            == "EXECUTION_SMOKE_NO_ENERGY_OR_PROFILE_CLAIM"
        and result.get("configuration") == configuration,
        f"result identity: {arm}",
    )
    inputs = result.get("inputs")
    require(type(inputs) is dict, f"input identity: {arm}")
    for name, expected in EXPECTED_INPUTS.items():
        row = inputs.get(name)
        require(
            type(row) is dict
            and row.get("sha256") == expected["sha256"]
            and row.get("size_bytes") == expected["size_bytes"],
            f"input identity {name}: {arm}",
        )
    require(
        sha256(HERE / "profile_gpu_residency.py")
            == EXPECTED_INPUTS["probe_source"]["sha256"],
        "checked-in profiler differs from measured source",
    )
    baseline = validate_gpu(result.get("baseline_gpu"), f"{arm} baseline")
    cleanup = result.get("cleanup")
    require(
        type(cleanup) is dict
        and cleanup.get("passed") is True
        and cleanup.get("compute_apps") == [],
        f"cleanup: {arm}",
    )
    validate_gpu(cleanup.get("gpu"), f"{arm} cleanup")
    stages = result.get("stages")
    require(type(stages) is list and len(stages) == 2, f"stages: {arm}")
    qwen_stage = validate_stage(
        stages[0],
        expected_layers=configuration["qwen_gpu_layers"],
        expected_total_layers=41,
        model_path=inputs["qwen_model"]["path"],
        name=f"{arm} qwen",
    )
    gemma_stage = validate_stage(
        stages[1],
        expected_layers=configuration["gemma_gpu_layers"],
        expected_total_layers=49,
        model_path=inputs["gemma_model"]["path"],
        name=f"{arm} gemma",
    )
    service = result.get("service_smoke")
    require(
        type(service) is dict
        and service.get("status") == "EXECUTION_SMOKE_PASS",
        f"service status: {arm}",
    )
    energy = service.get("energy")
    service_gpu = service.get("gpu")
    service_memory = service.get("process_memory")
    requests = service.get("requests")
    require(
        type(energy) is dict
        and energy.get("accounting_scope") == "CPU_PACKAGE_PLUS_GPU_BOARD"
        and energy.get("claim") == "DIAGNOSTIC_SINGLE_PROCESS_NOT_PROFILE"
        and type(service_gpu) is dict
        and service_gpu.get("minimum_free_bytes")
            >= configuration["gpu_reserve_bytes"]
        and type(service_gpu.get("samples")) is int
        and service_gpu["samples"] > 0
        and type(service_memory) is dict
        and set(service_memory) == {"gemma", "qwen"}
        and all(
            type(row) is dict
            and row.get("swap_bytes") == 0
            and positive(f"service RSS {name}: {arm}", row.get("rss_bytes"))
                > 0
            for name, row in service_memory.items()
        )
        and type(requests) is dict
        and set(requests) == {"gemma", "qwen"},
        f"service accounting: {arm}",
    )
    cpu_uj = int(positive(
        "CPU package energy", energy.get("cpu_package_uj")
    ))
    gpu_uj = int(positive(
        "GPU board energy", energy.get("gpu_board_uj")
    ))
    cpu_j = cpu_uj / 1e6
    gpu_j = gpu_uj / 1e6
    wall_s = positive("service wall", service.get("wall_service_us")) / 1e6
    completions: dict[str, dict[str, Any]] = {}
    started_ns: list[int] = []
    completed_ns: list[int] = []
    for model, expected in EXPECTED_REQUESTS.items():
        request = requests[model]
        require(
            request.get("request_index") == expected["request_index"]
            and request.get("output_tokens") == expected["output_tokens"]
            and type(request.get("started_monotonic_ns")) is int
            and type(request.get("completed_monotonic_ns")) is int
            and request["completed_monotonic_ns"]
                > request["started_monotonic_ns"]
            and positive(f"service time {model}: {arm}",
                         request.get("service_us")) > 0
            and positive(f"first token {model}: {arm}",
                         request.get("first_token_us")) > 0
            and request["first_token_us"] <= request["service_us"],
            f"request receipt {model}: {arm}",
        )
        smoke = parse_completion(
            root / f"{model}-smoke.raw",
            expected_input_tokens=expected["input_tokens"],
            expected_model=expected["model"],
            expected_output_tokens=expected["output_tokens"],
            expected_seed=expected["request_index"],
        )
        warm = parse_completion(
            root / f"{model}-warm.raw",
            expected_input_tokens=expected["input_tokens"],
            expected_model=expected["model"],
            expected_output_tokens=2,
            expected_seed=expected["request_index"],
        )
        require(
            smoke["prompt"] == warm["prompt"]
            and smoke["tokens_sha256"] == request.get("tokens_sha256"),
            f"raw completion binding {model}: {arm}",
        )
        started_ns.append(request["started_monotonic_ns"])
        completed_ns.append(request["completed_monotonic_ns"])
        completions[model] = {
            "first_token_s": request["first_token_us"] / 1e6,
            "output_tokens": request["output_tokens"],
            "prompt_sha256": smoke["prompt_sha256"],
            "request_index": request["request_index"],
            "service_s": request["service_us"] / 1e6,
            "tokens": smoke["tokens"],
            "tokens_sha256": smoke["tokens_sha256"],
        }
    active_span_s = (max(completed_ns) - min(started_ns)) / 1e9
    start_skew_s = (max(started_ns) - min(started_ns)) / 1e9
    require(
        start_skew_s <= 0.1
        and active_span_s <= wall_s
        and wall_s - active_span_s <= 0.1,
        f"concurrent service boundary: {arm}",
    )
    return {
        "artifact_sha256": {
            **manifest,
            "SHA256SUMS.txt": sha256(root / "SHA256SUMS.txt"),
        },
        "configuration": configuration,
        "energy": {
            "cpu_package_j": cpu_j,
            "gpu_board_j": gpu_j,
            "server_j": (cpu_uj + gpu_uj) / 1e6,
        },
        "gpu": {
            "baseline_free_bytes": baseline["memory_free_bytes"],
            "minimum_free_bytes": service_gpu["minimum_free_bytes"],
            "samples": service_gpu["samples"],
        },
        "inputs": inputs,
        "requests": completions,
        "service": {
            "active_span_s": active_span_s,
            "start_skew_s": start_skew_s,
            "wall_s": wall_s,
        },
        "stages": {"gemma": gemma_stage, "qwen": qwen_stage},
    }


def change_pct(control: float, treatment: float) -> float:
    return 100.0 * (treatment / control - 1.0)


def token_quality(control: list[int], treatment: list[int]) -> dict[str, Any]:
    require(len(control) == len(treatment) and control, "token comparison")
    matches = sum(left == right for left, right in zip(control, treatment))
    prefix = 0
    for left, right in zip(control, treatment):
        if left != right:
            break
        prefix += 1
    return {
        "common_prefix_tokens": prefix,
        "exact": control == treatment,
        "positional_agreement_pct": 100.0 * matches / len(control),
        "positional_matches": matches,
        "tokens": len(control),
    }


def analyze(control_root: Path, treatment_root: Path) -> dict[str, Any]:
    control = validate_arm(control_root, "control")
    treatment = validate_arm(treatment_root, "treatment")
    require(control["inputs"] == treatment["inputs"], "matched input epoch")
    for model in EXPECTED_REQUESTS:
        require(
            control["requests"][model]["prompt_sha256"]
                == treatment["requests"][model]["prompt_sha256"],
            f"matched prompt: {model}",
        )
    changes = {
        "cpu_package_energy_pct": change_pct(
            control["energy"]["cpu_package_j"],
            treatment["energy"]["cpu_package_j"],
        ),
        "gemma_first_token_pct": change_pct(
            control["requests"]["gemma"]["first_token_s"],
            treatment["requests"]["gemma"]["first_token_s"],
        ),
        "gemma_service_pct": change_pct(
            control["requests"]["gemma"]["service_s"],
            treatment["requests"]["gemma"]["service_s"],
        ),
        "gpu_board_energy_pct": change_pct(
            control["energy"]["gpu_board_j"],
            treatment["energy"]["gpu_board_j"],
        ),
        "qwen_first_token_pct": change_pct(
            control["requests"]["qwen"]["first_token_s"],
            treatment["requests"]["qwen"]["first_token_s"],
        ),
        "qwen_service_pct": change_pct(
            control["requests"]["qwen"]["service_s"],
            treatment["requests"]["qwen"]["service_s"],
        ),
        "server_energy_pct": change_pct(
            control["energy"]["server_j"],
            treatment["energy"]["server_j"],
        ),
        "wall_service_pct": change_pct(
            control["service"]["wall_s"], treatment["service"]["wall_s"]
        ),
    }
    quality = {
        model: token_quality(
            control["requests"][model].pop("tokens"),
            treatment["requests"][model].pop("tokens"),
        )
        for model in EXPECTED_REQUESTS
    }
    output: dict[str, Any] = {
        "admission": {
            "decision": "REPEAT_ABBA_BEFORE_ADMISSION",
            "dynamic_energy_claim": None,
            "eligible": False,
            "missing": [
                "ABBA_REPEATS",
                "FULL_TRACE_EQUAL_WORK",
                "LOAD_TRANSITION_AND_RESTORE_ENERGY",
                "OP15_SYNCHRONIZED_ENERGY",
                "PROTECTED_GPU_FENCE_RECEIPTS",
                "ATOMIC_TENSOR_SLICE_MANIFEST",
            ],
        },
        "arms": {"control": control, "treatment": treatment},
        "boundary": "cpu-package+gpu-board-concurrent-two-request-service-v1",
        "changes": changes,
        "claim_gates": {
            "abba_repeated": False,
            "atomic_tensor_slice_manifest": False,
            "full_trace_equal_work": False,
            "load_transition_restore_energy_included": False,
            "op15_energy_included": False,
            "protected_gpu_fence_receipts": False,
        },
        "observed_direction": (
            "LOWER_SERVER_ENERGY_AND_WALL_SINGLE_PAIR"
            if changes["server_energy_pct"] < 0
            and changes["wall_service_pct"] < 0
            else "NO_SINGLE_PAIR_JOINT_IMPROVEMENT"
        ),
        "quality": quality,
        "run_order": ["control", "treatment"],
        "schema": "s42-dual-residency-pair-diagnostic-v1",
        "status": "PROMISING_SINGLE_PAIR_NO_CLAIM",
        "validity_gates": {
            "artifact_manifests_verified": True,
            "cleanup_passed": True,
            "concurrent_same_requests": True,
            "exact_input_epoch": True,
            "gpu_reserve_preserved": True,
            "same_energy_boundary": True,
            "zero_observed_process_swap": True,
        },
        "workload": {
            "input_tokens": 287,
            "models": 2,
            "output_tokens": 50,
            "requests": 2,
            "trace_source_sha256": EXPECTED_INPUTS["trace"]["sha256"],
        },
    }
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--treatment", type=Path, default=DEFAULT_TREATMENT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an absolute new path")
    try:
        value = analyze(args.control, args.treatment)
        args.output.write_bytes(canonical(value))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"dual-residency pair analysis failed: {exc}\n")
    print(json.dumps({
        "decision": value["admission"]["decision"],
        "record_sha256": value["record_sha256"],
        "server_energy_change_pct": value["changes"]["server_energy_pct"],
        "status": value["status"],
        "wall_service_change_pct": value["changes"]["wall_service_pct"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
