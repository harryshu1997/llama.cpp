"""Compile one TFLite graph for Tensor G5 (SDK v2.0-20260902) with explicit Google Tensor options.

All options are passed through ai_edge_litert.aot.aot_compile as google_tensor_* flags of
apply_plugin_main; they are recorded verbatim in RESULT.json.
"""
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--precision", choices=("half", "bfloat16", "no_truncation", "auto"), default="half")
    ap.add_argument("--sharding", choices=("minimal", "moderate", "extensive", "maximum"), default="minimal")
    ap.add_argument("--performance-mode", default=None)
    ap.add_argument("--enable-4bit", action="store_true")
    ap.add_argument("--enable-drq", action="store_true")
    ap.add_argument("--large-model", action="store_true")
    ap.add_argument("--dump-op-timings", action="store_true")
    ap.add_argument("--extra-options", default=None)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    opts = dict(google_tensor_truncation_type=args.precision, google_tensor_sharding_intensity=args.sharding)
    if args.performance_mode:
        opts["google_tensor_performance_mode"] = args.performance_mode
    if args.enable_4bit:
        opts["google_tensor_enable_4bit_compilation"] = "true"
    if args.enable_drq:
        opts["google_tensor_enable_dynamic_range_quantization"] = "true"
    if args.large_model:
        opts["google_tensor_enable_large_model_support"] = "true"
    if args.dump_op_timings:
        opts["google_tensor_dump_op_timings"] = "true"
    if args.extra_options:
        opts["google_tensor_extra_options"] = args.extra_options
    sdk = Path(os.environ["GOOGLE_TENSOR_COMPILER_LIB"])
    rec = dict(at=datetime.now(timezone.utc).isoformat(), input=str(args.model.resolve()),
               input_sha256=digest(args.model), sdk_plugin_sha256=digest(sdk / "liblitert_plugin_compiler.so"),
               target="Tensor_G5", compiler_flags=opts, compiler_script_sha256=digest(Path(__file__)))
    started = time.monotonic()
    try:
        result = aot_compile.aot_compile(str(args.model.resolve()), output_dir=args.output / "intermediate",
                                         target=[target.Target(target.SocModel.TENSOR_G5)], keep_going=False, **opts)
        result.export(args.output, model_name=args.model.stem)
        report = result.compilation_report()
        (args.output / "PARTITION_REPORT.txt").write_text(report)
        rec.update(status="PASS", elapsed_s=time.monotonic() - started, partition_report=report,
                   outputs={p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in args.output.glob("*.tflite")})
    except Exception as error:
        rec.update(status="FAIL", elapsed_s=time.monotonic() - started, error=repr(error))
        (args.output / "RESULT.json").write_text(json.dumps(rec, indent=2) + "\n")
        raise
    (args.output / "RESULT.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
