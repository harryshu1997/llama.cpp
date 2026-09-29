"""Extend the preserved fused CPU worker with row queues and paired NEON dots."""

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

from build_pixel_cpu_gpu import replace_once


BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(source, header):
    source = '#include <sys/auxv.h>\n#include <asm/hwcap.h>\n' + source
    source = replace_once(source, '    struct graph_instance {',
        '    struct graph_instance {\n        std::vector<std::unique_ptr<pixel_packed_task>> packed_tasks;')
    begin = source.index('    auto build_graph = [&]')
    end = source.index('    for (size_t arena_index = 0;', begin)
    graph = source[begin:end]
    graph = replace_once(graph, '        ggml_reset(graph_ctx);',
                         '        instance.packed_tasks.clear();\n        ggml_reset(graph_ctx);')
    if graph.count(', pixel_correction);') != 3:
        raise ValueError('primary projection anchors differ')
    graph = graph.replace(', pixel_correction);', ', pixel_correction, instance.packed_tasks);')
    source = source[:begin] + graph + source[end:]
    source = replace_once(source, '    pixel_gpu_layout.rows = pixel_layout_integer("S42_PIXEL_GPU_LAYOUT_ROWS", 0);', r'''
    pixel_cpu_row_chunk = pixel_layout_integer("S42_PIXEL_CPU_ROW_CHUNK", 0);
    pixel_cpu_pair_mode = pixel_layout_integer("S42_PIXEL_CPU_PAIR_DOT", 0);
    pixel_cpu_row_profile = pixel_layout_integer("S42_PIXEL_CPU_ROW_PROFILE", 0) == 1;
    GGML_ASSERT(pixel_cpu_pair_mode <= 2);
    GGML_ASSERT(pixel_cpu_row_chunk == 0 || pixel_cpu_row_chunk == 16 || pixel_cpu_row_chunk == 32 ||
                pixel_cpu_row_chunk == 64 || pixel_cpu_row_chunk == 128 || pixel_cpu_row_chunk == 256);
    if (pixel_cpu_row_chunk || pixel_cpu_pair_mode || pixel_cpu_row_profile) {
        GGML_ASSERT(pixel_fused_residual && cfg.backend == "CPU" && !secondary.enabled());
        if (pixel_cpu_pair_mode) GGML_ASSERT(getauxval(AT_HWCAP) & HWCAP_ASIMDDP);
        fprintf(stderr, "S42PIXELCPUPACK chunk=%d mode=%d profile=%d\n",
                pixel_cpu_row_chunk, pixel_cpu_pair_mode, int(pixel_cpu_row_profile));
    }
    pixel_gpu_layout.rows = pixel_layout_integer("S42_PIXEL_GPU_LAYOUT_ROWS", 0);''')
    source = replace_once(source, '                fprintf(stderr, "S42PIXELNEONCALLS projections=%llu\\n",', r'''
                fprintf(stderr, "S42PIXELCPUPACKCALLS q4=%llu q6=%llu\n",
                        static_cast<unsigned long long>(pixel_cpu_q4_calls.load()),
                        static_cast<unsigned long long>(pixel_cpu_q6_calls.load()));
                if (pixel_cpu_row_profile) {
                    for (int i = 0; i < 8; ++i) {
                        const auto & sample = pixel_row_totals[i];
                        if (!sample.calls) continue;
                        fprintf(stderr, "S42PIXELROWPROFILE thread=%d calls=%llu rows=%llu work_us=%llu slack_us=%llu\n", i,
                                static_cast<unsigned long long>(sample.calls), static_cast<unsigned long long>(sample.rows),
                                static_cast<unsigned long long>(sample.work_us), static_cast<unsigned long long>(sample.slack_us));
                    }
                }
                fprintf(stderr, "S42PIXELNEONCALLS projections=%llu\n",''')
    header = replace_once(header, 'static std::atomic<uint64_t> pixel_residual_calls{0};',
        'static std::atomic<uint64_t> pixel_residual_calls{0};\n#include "pixel_packed_cpu.h"')
    header = replace_once(header,
        'static ggml_tensor * pixel_fused_residual_projection(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * input) {',
        'static ggml_tensor * pixel_fused_residual_projection(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * input,\n'
        '        std::vector<std::unique_ptr<pixel_packed_task>> & tasks) {')
    header = replace_once(header, '    ggml_tensor * dot_args[] = {weights, pair};', '''    ggml_tensor * dot_args[] = {weights, pair};
    ggml_custom_op_t function = pixel_packed_pair_dot;
    void * userdata = nullptr;
    if (pixel_cpu_row_chunk || pixel_cpu_pair_mode || pixel_cpu_row_profile) {
        tasks.emplace_back(new pixel_packed_task());
        function = pixel_tuned_packed_dot;
        userdata = tasks.back().get();
    }''')
    header = replace_once(header, 'dot_args, 2, pixel_packed_pair_dot, GGML_N_TASKS_MAX, nullptr);',
                          'dot_args, 2, function, GGML_N_TASKS_MAX, userdata);')
    header = replace_once(header,
        'static ggml_tensor * pixel_neon_projection(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * input, bool correction) {',
        'static ggml_tensor * pixel_neon_projection(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * input, bool correction,\n'
        '        std::vector<std::unique_ptr<pixel_packed_task>> & tasks) {')
    header = replace_once(header, 'return pixel_fused_residual_projection(ctx, weights, input);',
                          'return pixel_fused_residual_projection(ctx, weights, input, tasks);')
    return source, header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    previous = BASE / 'software/pixel10pro-layout-worker-v5'
    provenance = json.loads((previous / 'BUILD_PROVENANCE.json').read_text())
    for name in ('ffn-split-worker.cpp', 'pixel_f16_layout.h', 'BUILD_COMMAND.json'):
        if sha(previous / name) != provenance['sha256'][name]:
            raise ValueError('qualified source differs: ' + name)
    output = args.output.resolve()
    output.mkdir()
    for path in previous.glob('*.h'):
        (output / path.name).write_bytes(path.read_bytes())
    for path in (Path(__file__), BASE / 'pixel_packed_cpu.h'):
        (output / path.name).write_bytes(path.read_bytes())
    source, header = transform((previous / 'ffn-split-worker.cpp').read_text(), (previous / 'pixel_f16_layout.h').read_text())
    (output / 'ffn-split-worker.cpp').write_text(source)
    (output / 'pixel_f16_layout.h').write_text(header)
    patch = []
    for name in ('ffn-split-worker.cpp', 'pixel_f16_layout.h'):
        patch.extend(difflib.unified_diff((previous / name).read_text().splitlines(keepends=True),
            (output / name).read_text().splitlines(keepends=True), fromfile='qualified/' + name, tofile='tuned/' + name))
    (output / 'PACKED_CPU.patch').write_text(''.join(patch))
    command = json.loads((previous / 'BUILD_COMMAND.json').read_text())
    command = [str(output) + ':/diag' if item == str(previous) + ':/diag' else item for item in command]
    (output / 'BUILD_COMMAND.json').write_text(json.dumps(command, indent=2) + '\n')
    with (output / 'BUILD.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    compiler = command.index('/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++')
    disassembly = command[:compiler] + ['/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-objdump',
                                      '-d', '--demangle', '/diag/llama-ffn-split-worker']
    with (output / 'DISASSEMBLY.txt').open('x') as stream:
        subprocess.run(disassembly, stdout=stream, check=True)
    (output / 'BUILD_PROVENANCE.json').write_text(json.dumps(dict(status='PASS', at=datetime.now(timezone.utc).isoformat(),
        base_worker_sha256=provenance['sha256']['llama-ffn-split-worker'], physical_validation='NOT_RUN',
        sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()}), indent=2) + '\n')
    print('BUILD PASS', sha(output / 'llama-ffn-split-worker'), flush=True)


if __name__ == '__main__':
    main()
