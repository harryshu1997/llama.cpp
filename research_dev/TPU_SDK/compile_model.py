"""Compile one TFLite graph for Tensor G5 using the locally installed SDK."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from ai_edge_litert.aot import aot_compile
from ai_edge_litert.aot.vendors.google_tensor import target


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--precision", choices=("half", "bfloat16", "no_truncation", "auto"), default="half")
    parser.add_argument("--sharding", choices=("minimal", "moderate", "extensive", "maximum"), default="minimal")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sdk = Path(os.environ["GOOGLE_TENSOR_COMPILER_LIB"])
    record = dict(at=datetime.now(timezone.utc).isoformat(), input=str(args.model.resolve()),
                  input_sha256=digest(args.model), sdk_plugin_sha256=digest(sdk / "liblitert_plugin_compiler.so"),
                  target="Tensor_G5", precision=args.precision, sharding=args.sharding,
                  compiler_script_sha256=digest(Path(__file__)))
    started = time.monotonic()
    try:
        result = aot_compile.aot_compile(str(args.model.resolve()), output_dir=args.output / "intermediate",
            target=[target.Target(target.SocModel.TENSOR_G5)], keep_going=False,
            google_tensor_truncation_type=args.precision,
            google_tensor_sharding_intensity=args.sharding)
        result.export(args.output, model_name=args.model.stem)
        report = result.compilation_report()
        (args.output / "PARTITION_REPORT.txt").write_text(report)
        record.update(status="PASS", elapsed_s=time.monotonic()-started, partition_report=report,
                      outputs={p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in args.output.glob("*.tflite")})
    except Exception as error:
        record.update(status="FAIL", elapsed_s=time.monotonic()-started, error=repr(error))
        (args.output / "RESULT.json").write_text(json.dumps(record, indent=2) + "\n")
        raise
    (args.output / "RESULT.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
