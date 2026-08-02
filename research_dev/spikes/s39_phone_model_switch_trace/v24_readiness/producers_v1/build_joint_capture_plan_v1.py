#!/usr/bin/python3 -I
"""Build the exact V2.4 concurrent phone/CUDA capture plan."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"E_IMPORT: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


joint = load_module("s39_v24_joint", "joint_phone_cuda_v1.py")
phone = load_module("s39_v24_phone", "phone_route_v1.py")
cuda = load_module("s39_v24_cuda", "cuda_route_v1.py")


def file_record(path: Path, argv_index: int) -> dict:
    raw, _ = joint.read_regular(path, f"file[{argv_index}]")
    return {
        "argv_index": argv_index,
        "bytes": len(raw),
        "path": str(path),
        "sha256": joint.sha256_bytes(raw),
    }


def command(
    producer: Path,
    launch: Path,
    history: Path,
    mechanism_sha256: str,
    name: str,
    cwd: Path,
) -> dict:
    template = [
        str(producer),
        "--output",
        "{output_path}",
        "--phase-id",
        "{phase_id}",
        "--pre-dir",
        "{pre_dir}",
        "--started",
        "{acquisition_started_ns}",
        "--plan",
        "{command_plan_sha256}",
        "--mechanism-commands-sha256",
        mechanism_sha256,
        "--model-sha256",
        phone.MODEL_SHA256,
        "--launch-plan",
        str(launch),
    ]
    if name == "cuda":
        template.extend(["--histories", str(history)])
    template.extend([
        "--execute",
        "--confirm",
        (
            "RUN_V24_PHONE_ROUTE_A_ONLY"
            if name == "phone"
            else "RUN_V24_CUDA_ROUTE_A_ONLY"
        ),
    ])
    launch_index = template.index(str(launch))
    records = [
        file_record(producer, 0),
        file_record(launch, launch_index),
    ]
    if name == "cuda":
        records.append(file_record(history, template.index(str(history))))
    return {
        "argv_template": template,
        "cwd": str(cwd),
        "environment": {
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "executed_files": sorted(records, key=lambda value: value["argv_index"]),
        "launch_plan_argv_index": launch_index,
        "launch_plan_sha256": records[1]["sha256"],
        "producer_sha256": records[0]["sha256"],
        "result_filename": f"{name}.result.json",
        "timeout_seconds": 7200,
    }


def build(
    history_path: Path,
    phone_launch_path: Path,
    cuda_launch_path: Path,
    cwd: Path,
) -> dict:
    history_raw, _ = joint.read_regular(history_path, "history")
    phone_plan, _ = phone.load_plan(phone_launch_path)
    cuda_plan, _ = cuda.load_plan(
        cuda_launch_path,
        history_path,
        history_raw,
        cuda.read_canonical(cuda_launch_path)[0]["model_artifact"],
    )
    joint.exact(
        phone_plan["mechanism_commands"],
        cuda_plan["mechanism_commands"],
        "E_MECHANISM_MATRIX",
    )
    mechanism = phone_plan["mechanism_commands"]
    mechanism_sha256 = joint.sha256_bytes(joint.canonical_bytes(mechanism))
    phone_producer = ROOT / "phone_route_v1.py"
    cuda_producer = ROOT / "cuda_route_v1.py"
    return {
        "commands": {
            "cuda": command(
                cuda_producer,
                cuda_launch_path,
                history_path,
                mechanism_sha256,
                "cuda",
                cwd,
            ),
            "phone": command(
                phone_producer,
                phone_launch_path,
                history_path,
                mechanism_sha256,
                "phone",
                cwd,
            ),
        },
        "history": {
            "bytes": len(history_raw),
            "path": str(history_path),
            "sha256": joint.sha256_bytes(history_raw),
        },
        "mechanism_commands": mechanism,
        "model_id": joint.MODEL_ID,
        "model_sha256": phone.MODEL_SHA256,
        "phase": joint.PHASE,
        "schema": joint.PLAN_SCHEMA,
    }


def validate_before_publish(value: dict, output: Path) -> bytes:
    raw = joint.canonical_bytes(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".validate",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        validated, reopened = joint.load_plan(temporary)
        joint.exact(validated, value, "E_PLAN_REOPEN")
        joint.exact(reopened, raw, "E_PLAN_BYTES")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--phone-launch", type=Path, required=True)
    parser.add_argument("--cuda-launch", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        for field, path in vars(args).items():
            joint.require(path.is_absolute(), f"E_PATH: {field}")
        joint.require(args.cwd.is_dir(), "E_CWD")
        joint.require(not args.output.exists(), "E_OUTPUT_EXISTS")
        value = build(
            args.history,
            args.phone_launch,
            args.cuda_launch,
            args.cwd,
        )
        raw = validate_before_publish(value, args.output)
        joint.write_new(args.output, raw)
        return 0
    except (OSError, ValueError, phone.CaptureError, cuda.CaptureError) as error:
        print(f"V24_JOINT_PLAN_REFUSED: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
