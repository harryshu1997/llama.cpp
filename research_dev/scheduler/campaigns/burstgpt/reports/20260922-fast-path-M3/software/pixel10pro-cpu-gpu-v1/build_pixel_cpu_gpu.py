"""Build an isolated whole-block Pixel CPU/GPU variant of the dual worker."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

from build_pixel_cpu_tune import INSERT


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"source anchor differs: {before[:100]}")
    return source.replace(before, after, 1)


def transform(source):
    source = replace_once(source, '#include "ggml-backend.h"',
                          '#include "ggml-backend.h"\n#include "ggml-cpu.h"')
    source = replace_once(source, '    print_residency_phase(cfg, "HTP_INIT_READY");',
                          INSERT + '\n    print_residency_phase(cfg, "HTP_INIT_READY");')
    source = replace_once(source, '    double fraction = 0.0;',
                          '    double fraction = 0.0;\n    unsigned block_mask = 0;')
    source = replace_once(source, 'return !backend.empty() && fraction > 0.0;',
                          'return !backend.empty() && (fraction > 0.0 || block_mask != 0);')
    source = replace_once(source, '    result.backend = backend;', r'''
    result.backend = backend;
    if (const char * mask = getenv("S42_PIXEL_GPU_BLOCK_MASK")) {
        char * end = nullptr;
        errno = 0;
        const unsigned long value = strtoul(mask, &end, 16);
        if (errno || end == mask || *end || value == 0 || value >= 15) {
            error = "S42_PIXEL_GPU_BLOCK_MASK must select some of four blocks";
            return false;
        }
        result.block_mask = static_cast<unsigned>(value);
    }''')
    source = replace_once(source, '    const bool dual = secondary.enabled();', r'''
    const bool dual = secondary.enabled();
    if (secondary.block_mask && (cfg.backend != "CPU" || secondary.backend != "Vulkan0" ||
            cfg.columns != 17408 || cfg.column_quantum != 4352 || cfg.alternate_columns != 0 ||
            !cfg.ffs_root.empty() || secondary.fraction != 0 || secondary.max_tokens != 0)) {
        fprintf(stderr, "[ffn-worker] incompatible private whole-block configuration\n");
        return 2;
    }''')
    source = replace_once(source, '            if (dual) {\n                const int64_t align = secondary.align;', r'''
            if (dual && secondary.block_mask) {
                if (secondary.block_mask & (1u << state.blocks.size())) {
                    weight_block.secondary_columns = block_columns;
                    weight_block.secondary_gate = std::move(weight_block.gate);
                    weight_block.secondary_up = std::move(weight_block.up);
                    weight_block.secondary_down = std::move(weight_block.down);
                }
            } else if (dual) {
                const int64_t align = secondary.align;''')
    source = replace_once(source, '        ggml_init_params weight_params = {', r'''
        bool has_primary = false;
        for (size_t i = first_block; i < last_block; ++i) {
            has_primary |= states.front().blocks[i].primary_columns() > 0;
        }
        if (!has_primary) {
            continue;
        }
        ggml_init_params weight_params = {''')
    anchor = '                layer_state::block & block = state.blocks[block_index];'
    if source.count(anchor) != 2:
        raise ValueError("primary weight allocation loops differ")
    source = source.replace(anchor, anchor + '\n                if (block.primary_columns() == 0) continue;')
    source = replace_once(source, '            layer_state::block & block = state.blocks[index];\n            ggml_tensor * gate',
                          '            layer_state::block & block = state.blocks[index];\n'
                          '            if (block.primary_columns() == 0) continue;\n            ggml_tensor * gate')
    source = replace_once(source, '        if (sum == nullptr) {\n            return false;\n        }\n        // dual mode',
                          '        if (sum == nullptr) {\n            instance = graph_instance();\n'
                          '            return dual;\n        }\n        // dual mode')
    source = replace_once(source, '        uint64_t secondary_us = 0;',
                          '        uint64_t secondary_us = 0;\n        uint64_t secondary_started = 0;\n'
                          '        uint64_t secondary_done = 0;')
    source = replace_once(source, '                const uint64_t secondary_started = now_us();',
                          '                secondary_started = now_us();')
    source = replace_once(source, '                secondary_us = now_us() - secondary_started;',
                          '                secondary_done = now_us();\n'
                          '                secondary_us = secondary_done - secondary_started;')
    source = replace_once(source, '        graph_instance execution;\n        bool primary_ok = build_graph',
                          '        const uint64_t primary_started = now_us();\n'
                          '        graph_instance execution;\n        bool primary_ok = build_graph')
    source = replace_once(source, '        if (primary_ok) {\n            ggml_backend_tensor_set(',
                          '        if (primary_ok && execution.graph != nullptr) {\n            ggml_backend_tensor_set(')
    source = replace_once(source, '        if (primary_ok) {\n            dual_primary_output.resize(elements);',
                          '        dual_primary_output.assign(elements, 0.0f);\n'
                          '        if (primary_ok && execution.graph != nullptr) {')
    source = replace_once(source, '                    "merge_us=%llu total_us=%llu\\n",',
                          '                    "merge_us=%llu total_us=%llu primary_start_us=%llu "\n'
                          '                    "secondary_start_us=%llu secondary_end_us=%llu overlap_us=%llu\\n",')
    source = replace_once(source, '                    static_cast<unsigned long long>(compute_us));', r'''
                    static_cast<unsigned long long>(compute_us),
                    static_cast<unsigned long long>(primary_started - started),
                    static_cast<unsigned long long>(secondary_started ? secondary_started - started : 0),
                    static_cast<unsigned long long>(secondary_done ? secondary_done - started : 0),
                    static_cast<unsigned long long>(secondary_done > primary_started && primary_done > secondary_started ?
                        std::min(primary_done, secondary_done) - std::max(primary_started, secondary_started) : 0));''')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    old = args.source.read_text()
    new = transform(old)
    output = args.output.resolve()
    output.mkdir()
    (output / "BASE_SOURCE.cpp").write_text(old)
    (output / "ffn-split-worker.cpp").write_text(new)
    (output / "CPU_GPU.patch").write_text("".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile="existing-dual/ffn-split-worker.cpp", tofile="private-pixel/ffn-split-worker.cpp")))
    for header in (base / "software/pixel10pro-ffn-coalesced").glob("*.h"):
        (output / header.name).write_bytes(header.read_bytes())
    for path in (Path(__file__), base / "build_pixel_cpu_tune.py"):
        (output / path.name).write_bytes(path.read_bytes())
    previous = base / "software/pixel10pro-graph-tune"
    command = json.loads((previous / "BUILD_COMMAND.json").read_text())
    command = [str(output) + ":/diag" if item == str(previous) + ":/diag" else item for item in command]
    (output / "BUILD_COMMAND.json").write_text(json.dumps(command, indent=2) + "\n")
    with (output / "BUILD.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output / "BUILD_PROVENANCE.json").write_text(json.dumps({
        "status": "PASS", "base_source": str(args.source.resolve()), "sha256": hashes}, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "worker_sha256": hashes["llama-ffn-split-worker"]}))


if __name__ == "__main__":
    main()
