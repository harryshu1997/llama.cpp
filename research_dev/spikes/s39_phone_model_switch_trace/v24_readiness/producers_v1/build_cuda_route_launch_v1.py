#!/usr/bin/python3 -I
"""Upgrade a frozen V2.3 CUDA-route plan to the exact V2.4 history geometry."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import tempfile


ROUTE_PATH = Path(__file__).resolve().with_name("cuda_route_v1.py")
SPEC = importlib.util.spec_from_file_location("s39_v24_cuda_route", ROUTE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("E_ROUTE_IMPORT")
route = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(route)

LEGACY_SCHEMA = "s39-cp0-r1-a-only-cuda-route-launch-v1"


def set_option(argv: list[str], option: str, value: str) -> None:
    route.require(argv.count(option) == 1, f"E_OPTION_COUNT: {option}")
    index = argv.index(option)
    route.require(index + 1 < len(argv), f"E_OPTION_VALUE: {option}")
    argv[index + 1] = value


def build(template_path: Path, history_path: Path) -> dict:
    template, _ = route.read_canonical(template_path)
    route.exact(set(template), route.PLAN_KEYS, "template.keys")
    route.require(
        template["schema"] in {LEGACY_SCHEMA, route.PLAN_SCHEMA},
        "E_TEMPLATE_SCHEMA",
    )
    history, history_raw = route.load_histories(history_path)
    old_worker = list(template["worker"]["argv"])
    worker = list(old_worker)
    for option, value in (
        ("--ctx-size", str(route.N_CTX_SEQ * route.BATCH)),
        ("--parallel", str(route.BATCH)),
        ("--batch-size", str(route.N_BATCH)),
        ("--ubatch-size", str(route.N_UBATCH)),
    ):
        set_option(worker, option, value)
    template["schema"] = route.PLAN_SCHEMA
    template["history_path"] = str(history_path)
    template["history_sha256"] = route.sha256(history_raw)
    template["quality_corpus_content_sha256"] = history["corpus_sha256"]
    template["expected_n_ctx_seq"] = route.N_CTX_SEQ
    template["expected_n_batch"] = route.N_BATCH
    template["expected_n_ubatch"] = route.N_UBATCH
    template["worker"]["argv"] = worker
    desktop = template["mechanism_commands"]["desktop"]
    route.exact(desktop[1], old_worker, "template.mechanism.worker")
    desktop[1] = list(worker)
    return template


def validate_before_publish(
    value: dict,
    history_path: Path,
    output: Path,
) -> bytes:
    raw = route.canonical_bytes(value)
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
        validated, reopened = route.load_plan(
            temporary,
            history_path,
            route.read_regular(history_path),
            value["model_artifact"],
        )
        route.exact(validated, value, "E_PLAN_REOPEN")
        route.exact(reopened, raw, "E_PLAN_BYTES")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        for field, path in (
            ("template", args.template),
            ("history", args.history),
            ("output", args.output),
        ):
            route.require(path.is_absolute(), f"E_PATH: {field}")
        route.require(not args.output.exists(), "E_OUTPUT_EXISTS")
        value = build(args.template, args.history)
        raw = validate_before_publish(value, args.history, args.output)
        route.durable_write_new(args.output, raw)
        return 0
    except (OSError, ValueError, route.CaptureError) as error:
        print(f"V24_CUDA_ROUTE_LAUNCH_REFUSED: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
