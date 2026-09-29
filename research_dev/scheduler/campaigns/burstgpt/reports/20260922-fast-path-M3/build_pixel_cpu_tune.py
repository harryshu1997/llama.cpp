"""Build a private FFN worker using existing CPU threadpool and affinity APIs."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


INSERT = r'''
    std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> cpu_pool(nullptr, ggml_threadpool_free);
    if (cfg.backend == "CPU" && getenv("S42_PIXEL_CPU_THREADS")) {
        auto setting = [](const char * name, int fallback, int base) -> long {
            const char * value = getenv(name);
            if (!value) return fallback;
            char * end = nullptr;
            errno = 0;
            long result = strtol(value, &end, base);
            return errno || end == value || *end ? -1 : result;
        };
        long threads = setting("S42_PIXEL_CPU_THREADS", 4, 10);
        long mask = setting("S42_PIXEL_CPU_MASK", 0, 16);
        long persistent = setting("S42_PIXEL_CPU_POOL", 0, 10);
        if (threads < 1 || threads > 8 || mask < 0 || mask > 255 || persistent < 0 || persistent > 1 ||
            (mask && (!persistent || __builtin_popcount(unsigned(mask)) < threads))) {
            fprintf(stderr, "[ffn-worker] invalid private CPU tuning parameters\n");
            ggml_backend_free(backend);
            return 1;
        }
        ggml_backend_cpu_set_n_threads(backend, int(threads));
        if (persistent) {
            ggml_threadpool_params params = ggml_threadpool_params_default(int(threads));
            params.paused = true;
            params.poll = 0;
            params.strict_cpu = mask != 0;
            for (int i = 0; i < 8; ++i) params.cpumask[i] = (mask & (1 << i)) != 0;
            cpu_pool.reset(ggml_threadpool_new(&params));
            if (!cpu_pool) { ggml_backend_free(backend); return 1; }
            ggml_backend_cpu_set_threadpool(backend, cpu_pool.get());
        }
        fprintf(stderr, "S42PIXELCPU threads=%ld mask=%lx persistent=%ld poll=0\n", threads, mask, persistent);
    }
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--affinity-fix", action="store_true")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    original = base / "software/pixel10pro-ffn-coalesced"
    source = (original / "ffn-split-worker.cpp").read_bytes()
    if hashlib.sha256(source).hexdigest() != "e1be45350fae2c7c06f26ae985114108ef81e697983a20286a4a3f757e32de22":
        raise ValueError("qualified worker changed")
    old = source.decode()
    anchor = '    print_residency_phase(cfg, "HTP_INIT_READY");'
    if old.count(anchor) != 1:
        raise ValueError("backend initialization differs")
    updated = old.replace('#include "ggml-backend.h"', '#include "ggml-backend.h"\n#include "ggml-cpu.h"')
    updated = updated.replace(anchor, INSERT + "\n" + anchor)
    output = args.output.resolve()
    output.mkdir()
    (output / "ffn-split-worker.cpp").write_text(updated)
    (output / "CPU_TUNE.patch").write_text("".join(difflib.unified_diff(
        old.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile="qualified/ffn-split-worker.cpp", tofile="private/ffn-split-worker.cpp")))
    for header in original.glob("*.h"):
        (output / header.name).write_bytes(header.read_bytes())
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    previous = base / "software/pixel10pro-graph-tune"
    command = json.loads((previous / "BUILD_COMMAND.json").read_text())
    command = [str(output) + ":/diag" if item == str(previous) + ":/diag" else item for item in command]
    (output / "BUILD_COMMAND.json").write_text(json.dumps(command, indent=2) + "\n")
    with (output / "BUILD.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if args.affinity_fix:
            runtime = base / "software/pixel10pro-vulkan"
            cpu_source = runtime / "source/ggml/src/ggml-cpu/ggml-cpu.c"
            before = cpu_source.read_text()
            anchor = "#elif defined(__gnu_linux__)\n// TODO: this may not work on BSD, to be verified"
            if before.count(anchor) != 1:
                raise ValueError("affinity platform guard differs")
            after = before.replace(anchor, anchor.replace("defined(__gnu_linux__)", "defined(__gnu_linux__) || defined(__ANDROID__)"))
            (output / "ggml-cpu.c").write_text(after)
            (output / "ANDROID_AFFINITY.patch").write_text("".join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile="qualified/ggml-cpu.c", tofile="private/ggml-cpu.c")))
            compiled = next(row for row in json.loads((runtime / "build/compile_commands.json").read_text())
                            if row["file"].endswith("/ggml-cpu.c"))
            compile_command = shlex.split(compiled["command"])
            compile_command = ["/diag/ggml-cpu.c" if item == compiled["file"] else item for item in compile_command]
            compile_command[compile_command.index("-o") + 1] = "/diag/ggml-cpu.c.o"
            link_text = subprocess.check_output(["ninja", "-C", str(runtime / "build"), "-t", "commands", "bin/libggml-cpu.so"], text=True).splitlines()[-1]
            link_command = shlex.split(link_text)[2:-2]
            link_command = ["/diag/ggml-cpu.c.o" if item == compiled["output"] else
                            "/diag/libggml-cpu.so" if item == "bin/libggml-cpu.so" else
                            "/work/build/" + item if not item.startswith("-") and item.endswith((".o", ".so")) else item for item in link_command]
            compiler_index = next(i for i, item in enumerate(command) if item.endswith("aarch64-linux-android28-clang++"))
            docker = command[:compiler_index]
            commands = [docker + compile_command, docker + link_command]
            (output / "CPU_LIBRARY_COMMANDS.json").write_text(json.dumps(commands, indent=2) + "\n")
            for cmd in commands:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            object_paths = [runtime / "build" / item.removeprefix("/work/build/") for item in link_command
                            if item.startswith("/work/build/") and item.endswith((".o", ".so"))]
            (output / "REUSED_OBJECTS.json").write_text(json.dumps(
                {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in object_paths}, indent=2) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output / "BUILD_PROVENANCE.json").write_text(json.dumps({"status": "PASS", "sha256": hashes}, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "worker_sha256": hashes["llama-ffn-split-worker"]}))


if __name__ == "__main__":
    main()
