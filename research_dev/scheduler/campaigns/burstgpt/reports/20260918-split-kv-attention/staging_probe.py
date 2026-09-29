"""Compare allocator-reserved GPU scratch for one and four split KV layers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from research_dev.scheduler.tests.tiny_llama_gguf import write_tiny_llama_gguf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = write_tiny_llama_gguf(args.output / "tiny.gguf", n_layer=4, head_dim=64)
    buffers = {}
    for name, layers in (("one", (3,)), ("four", range(4))):
        command = [str(args.probe.resolve()), "--model", str(model),
                   "--out", str(args.output / f"{name}.json"),
                   "--ctx-size", "32768", "--gpu-layers", "5", "--threads", "4",
                   "--flash-attn", "on", "--batch-size", "512", "--ubatch-size", "128",
                   "--max-tokens", "512", "--tokens", "2,3,4,5,6,7", "--decode", "1",
                   "--kv-device-cells", ",".join(f"{i}:8192" for i in layers)]
        (args.output / f"{name}.command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (args.output / f"{name}.stderr").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=log,
                           env=os.environ.copy(), timeout=120)
        record = json.loads((args.output / f"{name}.json").read_text())
        assert record["context_created"] and record["decode"]["status"] == 0, record
        lines = [line for line in record["log"] if "compute buffer size" in line]
        values = [float(match.group(1)) for line in lines
                  if (match := re.search(r"CUDA\d+ compute buffer size =\s*([0-9.]+) MiB", line))]
        assert len(values) == 1, lines
        buffers[name] = values[0]
        print(name, lines, flush=True)
    slice_mib = (32768 - 8192) * 2 * 2 * 64 * 2 / (1024**2)
    # Three extra split layers must not retain three extra host-slice copies on GPU.
    assert buffers["four"] - buffers["one"] < slice_mib, buffers
    summary = {"status": "PASS", "compute_buffer_mib": buffers,
               "one_host_kv_slice_mib": slice_mib,
               "scope": "allocator worst-case graph reservation, not a long-context decode"}
    (args.output / "RESULT.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
