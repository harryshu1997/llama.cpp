"""Build private Pixel Q4_K/Q6_K kernels on the qualified layout backend."""

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from build_pixel_cpu_gpu import replace_once


BASE = Path(__file__).resolve().parent
VARIANTS = ("vec4", "pair8", "block16", "block16_shuffle")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(source):
    source = '#include "pixel_packed_shader_data.hpp"\n' + source
    source = replace_once(source, "    static constexpr uint32_t mul_mat_vec_num_bindings = 5;", r'''
    uint32_t pixel_q_variant = 0;
    uint32_t pixel_q_wg = 128, pixel_q_rows = 4, pixel_q_subgroup = 128;
    if (const char * value = getenv("S42_PIXEL_Q_SHADER")) {
        for (uint32_t i = 0; i < sizeof(pixel_q_names) / sizeof(pixel_q_names[0]); ++i) {
            if (std::string(value) == pixel_q_names[i]) pixel_q_variant = i + 1;
        }
        GGML_ASSERT(pixel_q_variant != 0);
        GGML_ASSERT(std::string(device->properties.deviceName.data()).find("PowerVR") == 0);
        auto read_setting = [](const char * name, uint32_t fallback) {
            const char * value = getenv(name);
            if (!value) return fallback;
            char * end = nullptr;
            const unsigned long result = strtoul(value, &end, 10);
            GGML_ASSERT(*value && !*end && result <= 512);
            return uint32_t(result);
        };
        pixel_q_wg = read_setting("S42_PIXEL_Q_WG", 128);
        pixel_q_rows = read_setting("S42_PIXEL_Q_ROWS", 4);
        pixel_q_subgroup = read_setting("S42_PIXEL_Q_SUBGROUP", 128);
        GGML_ASSERT(pixel_q_wg == 128 || pixel_q_wg == 256);
        GGML_ASSERT(pixel_q_rows == 2 || pixel_q_rows == 4 || pixel_q_rows == 8);
        GGML_ASSERT(pixel_q_subgroup == 32 || pixel_q_subgroup == 64 || pixel_q_subgroup == 128);
        GGML_ASSERT(device->subgroup_size_control && pixel_q_subgroup >= device->subgroup_min_size &&
                    pixel_q_subgroup <= device->subgroup_max_size);
        GGML_ASSERT(pixel_q_wg <= device->properties.limits.maxComputeWorkGroupInvocations);
        GGML_ASSERT(pixel_q_wg <= device->properties.limits.maxComputeWorkGroupSize[0]);
        if (std::string(value) == "block16_shuffle") {
            GGML_ASSERT(device->subgroup_shuffle && pixel_q_wg / pixel_q_rows <= pixel_q_subgroup);
        }
        std::cerr << "S42PIXELQKERNEL variant=" << value << " wg=" << pixel_q_wg
                  << " rows=" << pixel_q_rows << " subgroup=" << pixel_q_subgroup << " precision=f32" << std::endl;
    }
    static constexpr uint32_t mul_mat_vec_num_bindings = 5;''')
    for index, kind in enumerate(("Q4_K", "Q6_K")):
        anchor = next(line for line in source.splitlines() if
                      f'pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_{kind}][i], "mul_mat_vec_' in line)
        replacement = f'''            if (pixel_q_variant && w == DMMV_WG_SIZE_SUBGROUP && (i == 0 || i == 1 || i == 3 || i == 7)) {{
                const auto code = pixel_q_code[pixel_q_variant - 1][{index}];
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_{kind}][i],
                    "pixel_packed_{kind.lower()}", code.len, code.data, "main", mul_mat_vec_num_bindings,
                    sizeof(vk_mat_vec_push_constants), {{pixel_q_rows, 1, 1}}, {{pixel_q_wg, pixel_q_rows, i+1}},
                    1, true, true, pixel_q_subgroup);
            }} else {{
{anchor}
            }}'''
        source = replace_once(source, anchor, replacement)
    source = replace_once(source, "    uint64_t pixel_packed_dispatches[8] = {};", '''    uint64_t pixel_packed_dispatches[8] = {};
    uint64_t pixel_q4_dispatches[8] = {};
    uint64_t pixel_q6_dispatches[8] = {};
    const bool pixel_q_enabled = getenv("S42_PIXEL_Q_SHADER") != nullptr;''')
    start = source.index("static void ggml_vk_mul_mat_vec_q_f16(")
    end = source.index("static void ggml_vk_mul_mat_vec_p021_f16_f32(", start)
    part = source[start:end]
    part = replace_once(part, "    vk_pipeline to_fp16_vk_0 = nullptr;", '''    const bool pixel_q_selected = ctx->pixel_q_enabled &&
        (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q6_K) &&
        (ne11 == 1 || ne11 == 2 || ne11 == 4 || ne11 == 8);
    if (pixel_q_selected) {
        GGML_ASSERT(!x_non_contig && !y_non_contig && f16_f32_kernel && ne00 % 256 == 0);
        GGML_ASSERT(ggml_nbytes(src0) <= ctx->device->properties.limits.maxStorageBufferRange);
        quantize_y = false;
    }
    vk_pipeline to_fp16_vk_0 = nullptr;''')
    part = replace_once(part, "        base_work_group_y += groups_y;", '''        if (pixel_q_selected) {
            GGML_ASSERT(dmmv == ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[DMMV_WG_SIZE_SUBGROUP][src0->type][ne11-1]);
            if (src0->type == GGML_TYPE_Q4_K) ++ctx->pixel_q4_dispatches[ne11 - 1];
            else ++ctx->pixel_q6_dispatches[ne11 - 1];
        }
        base_work_group_y += groups_y;''')
    source = source[:start] + part + source[end:]
    source = replace_once(source, "        if (ctx->pixel_f16_dispatches[i] || ctx->pixel_packed_dispatches[i]) {", '''        if (ctx->pixel_q4_dispatches[i] || ctx->pixel_q6_dispatches[i]) {
            std::cerr << "S42PIXELQDISPATCH rows=" << i+1 << " q4=" << ctx->pixel_q4_dispatches[i]
                      << " q6=" << ctx->pixel_q6_dispatches[i] << std::endl;
        }
        if (ctx->pixel_f16_dispatches[i] || ctx->pixel_packed_dispatches[i]) {''')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    previous = BASE / "software/pixel10pro-dense-layout-v2"
    provenance = json.loads((previous / "BUILD_PROVENANCE.json").read_text())
    if sha(previous / "ggml-vulkan.cpp") != provenance["source_sha256"]:
        raise ValueError("qualified source differs")
    output = args.output.resolve()
    output.mkdir()
    for path in (Path(__file__), BASE / "pixel_packed_gemv.comp", previous / "pixel_shader_data.hpp"):
        (output / path.name).write_bytes(path.read_bytes())
    commands = json.loads((previous / "BUILD_COMMANDS.json").read_text())
    commands = [[str(output) + ":/diag" if item == str(previous) + ":/diag" else item for item in cmd] for cmd in commands]
    compiler = commands[0].index("/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++")
    docker = commands[0][:compiler]
    header = ["#pragma once\n#include <cstdint>\n"]
    shader_commands = []
    validations = []
    validator = BASE / "software/spirv-tools-validation/root/usr/bin/spirv-val"
    with (output / "BUILD.log").open("x") as log:
        for variant in VARIANTS:
            for kind in ("q4_k", "q6_k"):
                name = f"{variant}_{kind}"
                definitions = {f"DATA_A_{kind.upper()}": 1, "B_TYPE": "float", "B_TYPEV4": "vec4",
                               "D_TYPE": "float", "FLOAT_TYPE": "float", "FLOAT_TYPEV2": "vec2",
                               "PIXEL_Q_PAIR": int(variant == "pair8"), "PIXEL_Q_SUPER": int(variant.startswith("block16")),
                               "PIXEL_Q_SHUFFLE": int(variant == "block16_shuffle")}
                cmd = docker + ["env", "LD_LIBRARY_PATH=/work/deps/sysroot/usr/lib/x86_64-linux-gnu",
                    "/work/deps/sysroot/usr/bin/glslc", "-fshader-stage=compute", "--target-env=vulkan1.2", "-O", "-Werror",
                    "-I/work/source/ggml/src/ggml-vulkan/vulkan-shaders", *[f"-D{k}={v}" for k, v in definitions.items()],
                    "/diag/pixel_packed_gemv.comp", "-o", f"/diag/{name}.spv"]
                shader_commands.append(cmd)
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
                blob = (output / f"{name}.spv").read_bytes()
                words = struct.unpack(f"<{len(blob)//4}I", blob)
                if words[0] != 0x07230203:
                    raise ValueError("invalid SPIR-V")
                header.append(f"static const uint32_t pixel_q_{name}[] = {{\n")
                for i in range(0, len(words), 8):
                    header.append("    " + ", ".join(f"0x{v:08x}" for v in words[i:i+8]) + ",\n")
                header.append("};\n")
                check = subprocess.run([str(validator), "--target-env", "vulkan1.2", str(output / f"{name}.spv")], capture_output=True, text=True)
                validations.append(dict(module=name,exit_code=check.returncode,output=check.stdout+check.stderr))
                if check.returncode:
                    raise ValueError(validations[-1])
        header.append('struct pixel_q_blob { const void * data; uint64_t len; };\n')
        header.append('static const char * pixel_q_names[] = {' + ', '.join(json.dumps(v) for v in VARIANTS) + '};\n')
        header.append('static const pixel_q_blob pixel_q_code[][2] = {\n')
        for variant in VARIANTS:
            header.append('    {' + ', '.join(f'{{pixel_q_{variant}_{kind}, sizeof(pixel_q_{variant}_{kind})}}' for kind in ("q4_k", "q6_k")) + '},\n')
        header.append('};\n')
        (output / "pixel_packed_shader_data.hpp").write_text(''.join(header))
        source = (previous / "ggml-vulkan.cpp").read_text()
        updated = transform(source)
        (output / "ggml-vulkan.cpp").write_text(updated)
        (output / "PACKED_BACKEND.patch").write_text(''.join(difflib.unified_diff(source.splitlines(keepends=True),updated.splitlines(keepends=True),fromfile="qualified/ggml-vulkan.cpp",tofile="packed/ggml-vulkan.cpp")))
        (output / "BUILD_COMMANDS.json").write_text(json.dumps(commands,indent=2)+'\n')
        (output / "SHADER_COMMANDS.json").write_text(json.dumps(shader_commands,indent=2)+'\n')
        (output / "SPIRV_VALIDATION.json").write_text(json.dumps(validations,indent=2)+'\n')
        for cmd in commands:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest=dict(status="PASS",at=datetime.now(timezone.utc).isoformat(),qualified_backend_sha256=provenance["source_sha256"],physical_validation="NOT_RUN",sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output / "BUILD_PROVENANCE.json").write_text(json.dumps(manifest,indent=2)+'\n')
    print("BUILD PASS",sha(output / "libggml-vulkan.so"),flush=True)


if __name__ == "__main__":
    main()
