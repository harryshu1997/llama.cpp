"""Build a private Android CPU/Vulkan streaming benchmark with checked reads."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    original = base / "software/pixel10pro-vulkan"
    output = args.output.resolve()
    output.mkdir()
    for name in ("pixel_bandwidth.cpp", "pixel_bandwidth.comp", "build_pixel_bandwidth.py"):
        (output / name).write_bytes((base / name).read_bytes())
    docker = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
              "-v", f"{original}:/work:ro", "-v", f"{output}:/diag", "snapdragon-toolchain-hostgcc:v0.3"]
    commands = []
    with (output / "BUILD.log").open("x") as log:
        for width, unroll in ((1, 1), (4, 1), (4, 4), (4, 8)):
            name = f"read_v{width}_u{unroll}.spv"
            command = docker + ["env", "LD_LIBRARY_PATH=/work/deps/sysroot/usr/lib/x86_64-linux-gnu",
                "/work/deps/sysroot/usr/bin/glslc", "-fshader-stage=compute", "--target-env=vulkan1.2",
                "-O", "-Werror", f"-DWIDTH={width}", f"-DUNROLL={unroll}",
                "/diag/pixel_bandwidth.comp", "-o", f"/diag/{name}"]
            commands.append(command)
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            command = [str(base / "software/spirv-tools-validation/root/usr/bin/spirv-val"),
                       "--target-env", "vulkan1.2", str(output / name)]
            commands.append(command)
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        command = docker + ["/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++",
            "-std=c++17", "-O3", "-DNDEBUG", "-Wall", "-Wextra", "-Werror", "-Wno-missing-field-initializers",
            "-static-libstdc++", "/diag/pixel_bandwidth.cpp", "-lvulkan", "-pthread", "-o", "/diag/pixel-bandwidth"]
        commands.append(command)
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    (output / "BUILD_COMMANDS.json").write_text(json.dumps(commands, indent=2) + "\n")
    files = [p for p in output.iterdir() if p.is_file()]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (output / "BUILD_PROVENANCE.json").write_text(json.dumps({"status": "PASS", "sha256": hashes}, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "binary_sha256": hashes["pixel-bandwidth"]}))


if __name__ == "__main__":
    main()
