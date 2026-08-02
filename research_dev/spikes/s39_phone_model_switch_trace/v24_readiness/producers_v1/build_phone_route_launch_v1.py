#!/usr/bin/python3 -I
"""Upgrade a frozen V2.3 phone-route plan to the exact V2.4 history geometry."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import tempfile


PHONE_PATH = Path(__file__).resolve().with_name("phone_route_v1.py")
SPEC = importlib.util.spec_from_file_location("s39_v24_phone_route", PHONE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("E_PHONE_IMPORT")
phone = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phone)

LEGACY_SCHEMA = "s39-cp0-r1-a-only-phone-route-launch-v1"


def replace_option(argv: list[str], option: str, value: str) -> bool:
    count = argv.count(option)
    phone.require(count in (0, 1), f"E_OPTION_COUNT: {option}")
    if count == 0:
        return False
    index = argv.index(option)
    phone.require(index + 1 < len(argv), f"E_OPTION_VALUE: {option}")
    argv[index + 1] = value
    return True


def build(template_path: Path, history_path: Path) -> dict:
    template, _ = phone.read_canonical(template_path)
    phone.require(
        template["schema"] in {LEGACY_SCHEMA, phone.PLAN_SCHEMA},
        "E_TEMPLATE_SCHEMA",
    )
    history_raw = phone.read_regular(history_path)
    history, _ = phone.load_histories(
        history_path,
        phone.sha256(history_raw),
        phone.MODEL_SHA256,
    )
    old_commands = {
        name: list(spec["argv"])
        for name, spec in template["processes"].items()
    }
    changed = 0
    for spec in template["processes"].values():
        argv = list(spec["argv"])
        changed += replace_option(
            argv,
            "--ctx-size",
            str(phone.N_CTX_SEQ * phone.MAX_STREAMS),
        )
        replace_option(argv, "--parallel", str(phone.MAX_STREAMS))
        replace_option(argv, "--batch-size", str(phone.N_BATCH))
        replace_option(argv, "--ubatch-size", str(phone.N_UBATCH))
        spec["argv"] = argv
    phone.require(changed >= 2, "E_CONTEXT_OPTION_COVERAGE")
    template["schema"] = phone.PLAN_SCHEMA
    template["history_path"] = str(history_path)
    template["history_sha256"] = phone.sha256(history_raw)
    template["quality_corpus_content_sha256"] = history["corpus_sha256"]
    template["expected_n_ctx_seq"] = phone.N_CTX_SEQ
    template["expected_n_batch"] = phone.N_BATCH
    template["expected_n_ubatch"] = phone.N_UBATCH
    matrix = template["mechanism_commands"]
    for endpoint in ("op12", "op15"):
        for index, command in enumerate(matrix[endpoint]):
            for name, old in old_commands.items():
                if command == old:
                    matrix[endpoint][index] = list(
                        template["processes"][name]["argv"]
                    )
                    break
    return template


def validate_before_publish(
    value: dict,
    output: Path,
) -> bytes:
    raw = phone.canonical_bytes(value)
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
        validated, reopened = phone.load_plan(temporary)
        phone.exact(validated, value, "E_PLAN_REOPEN")
        phone.exact(reopened, raw, "E_PLAN_BYTES")
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
        for field, path in vars(args).items():
            phone.require(path.is_absolute(), f"E_PATH: {field}")
        phone.require(not args.output.exists(), "E_OUTPUT_EXISTS")
        value = build(args.template, args.history)
        raw = validate_before_publish(value, args.output)
        phone.durable_write_new(args.output, raw)
        return 0
    except (OSError, ValueError, phone.CaptureError) as error:
        print(f"V24_PHONE_ROUTE_LAUNCH_REFUSED: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
