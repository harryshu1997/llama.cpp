"""Build isolated shader-body experiments using the qualified Vulkan backend."""

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import struct
import subprocess


VARIANTS = {
    "pair4_u4": (4, 4, 1, 1, 0),
    "vec4_u1": (4, 1, 1, 0, 0),
    "vec4_u2": (4, 2, 1, 0, 0),
    "vec4_u4": (4, 4, 1, 0, 0),
    "vec4_u8": (4, 8, 1, 0, 0),
    "vec4_u4_a2": (4, 4, 2, 0, 0),
    "vec4_u4_a4": (4, 4, 4, 0, 0),
    "vec8_u2": (8, 2, 1, 0, 0),
    "striped4_u4": (4, 4, 1, 0, 1),
}
MACROS = ("PIXEL_WIDTH", "PIXEL_UNROLL", "PIXEL_ACCUM", "PIXEL_PAIRED", "PIXEL_STRIPED")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fuse-swiglu", action="store_true")
    parser.add_argument("--aligned-k", action="store_true", help="enable dense kernels for 64-aligned K up to8704")
    parser.add_argument("--maximum-k", type=int, default=8704)
    parser.add_argument("--batch-rows", action="store_true", help="reuse F16 weights across 2/4/8 input rows")
    parser.add_argument("--dispatch-counts", action="store_true")
    parser.add_argument("--batch-shared", action="store_true")
    parser.add_argument("--f16-layout", action="store_true")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    original = base / "software/pixel10pro-vulkan"
    previous = base / "software/pixel10pro-gemv-tune-subgroup"
    provenance = json.loads((previous / "BUILD_PROVENANCE.json").read_text())
    if sha(previous / "ggml-vulkan.cpp") != provenance["source_sha256"]:
        raise RuntimeError("qualified backend source changed")
    output = args.output.resolve()
    output.mkdir()
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (output / "pixel_dense_gemv.glsl").write_bytes((base / "pixel_dense_gemv.glsl").read_bytes())
    snippet = (base / "pixel_dense_gemv.glsl").read_text()
    if args.f16_layout:
        if args.fuse_swiglu or args.batch_shared:
            raise ValueError("layout shaders require unfused projections")
        snippet += "\n" + (base / "pixel_dense_layout.glsl").read_text()
        (output / "pixel_dense_layout.glsl").write_bytes((base / "pixel_dense_layout.glsl").read_bytes())
    if args.batch_rows:
        snippet += "\n" + (base / "pixel_dense_batch.glsl").read_text()
        (output / "pixel_dense_batch.glsl").write_bytes((base / "pixel_dense_batch.glsl").read_bytes())
    if args.batch_shared:
        if not args.batch_rows or args.fuse_swiglu:
            raise ValueError("shared tiling requires unfused batch kernels")
        snippet += "\n" + (base / "pixel_dense_batch_shared.glsl").read_text()
        (output / "pixel_dense_batch_shared.glsl").write_bytes((base / "pixel_dense_batch_shared.glsl").read_bytes())
    shaders = original / "source/ggml/src/ggml-vulkan/vulkan-shaders"
    if args.fuse_swiglu:
        import pixel_swiglu_fusion
        base_source = (shaders / "mul_mat_vec_base.glsl").read_text()
        fused_base = pixel_swiglu_fusion.shader_base(base_source)
        (output / "mul_mat_vec_base.glsl").write_text(fused_base)
        (output / "pixel_swiglu_fusion.py").write_bytes((base / "pixel_swiglu_fusion.py").read_bytes())
        (output / "SWIGLU_SHADER.patch").write_text("".join(difflib.unified_diff(
            base_source.splitlines(keepends=True), fused_base.splitlines(keepends=True),
            fromfile="qualified/mul_mat_vec_base.glsl", tofile="fused/mul_mat_vec_base.glsl")))
    source = (shaders / "mul_mat_vec.comp").read_text()
    anchor = "void compute_outputs(const uint32_t first_row, const uint32_t num_rows) {"
    if source.count(anchor) != 1:
        raise RuntimeError("shader entry differs")
    updated = source.replace(anchor, snippet + "\n" + anchor)
    anchor = "    get_offsets(a_offset, b_offset, d_offset);\n"
    guard = """    if (NUM_COLS == 1 && (p.ncols == 5120 || p.ncols == 4352) &&
        a_offset % 4 == 0 && b_offset % 4 == 0) {
        pixel_dense_outputs(first_row, num_rows);
        return;
    }
"""
    if args.aligned_k:
        guard = guard.replace("(p.ncols == 5120 || p.ncols == 4352)",
                              f"(p.ncols >= 64 && p.ncols <= {args.maximum_k} && p.ncols % 64 == 0)")
    if args.batch_rows:
        batch_guard = guard.replace("NUM_COLS == 1", "(NUM_COLS == 2 || NUM_COLS == 4 || NUM_COLS == 8)").replace(
            "pixel_dense_outputs", "pixel_dense_batch_outputs")
        if args.batch_shared:
            guard += batch_guard.replace("if (", "if (BLOCK_SIZE == 128 && NUM_ROWS == 8 && ", 1).replace(
                "pixel_dense_batch_outputs", "pixel_dense_batch_shared_outputs")
        guard += batch_guard
    if args.f16_layout:
        guard = """#if PIXEL_LAYOUT_K
    pixel_layout_outputs(first_row, num_rows);
    return;
#endif
""" + guard
    if updated.count(anchor) != 1:
        raise RuntimeError("shader offset setup differs")
    updated = updated.replace(anchor, anchor + guard)
    (output / "mul_mat_vec.comp").write_text(updated)
    (output / "DENSE_SHADER.patch").write_text("".join(difflib.unified_diff(
        source.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile="qualified/mul_mat_vec.comp", tofile="dense/mul_mat_vec.comp")))
    commands = json.loads((previous / "BUILD_COMMANDS.json").read_text())
    commands = [[str(output) + ":/diag" if item == str(previous) + ":/diag" else item
                 for item in command] for command in commands]
    compiler_index = commands[0].index("/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++")
    docker = commands[0][:compiler_index]
    shader_commands = []
    definitions = {"DATA_A_F16": "1", "A_TYPEV4": "f16vec4", "B_TYPE": "float",
                   "B_TYPEV2": "vec2", "B_TYPEV4": "vec4", "D_TYPE": "float",
                   "FLOAT_TYPE": "float", "FLOAT_TYPEV2": "vec2"}
    variants = {}
    variant_values = dict(VARIANTS)
    if args.f16_layout:
        variant_values.update(layout8k4=VARIANTS["vec4_u1"], layout8k16=VARIANTS["vec4_u1"])
    header = ["#pragma once\n"]
    with (output / "BUILD.log").open("x") as log:
        for name, values in variant_values.items():
            variants[name] = dict(zip(MACROS, values))
            variants[name]["PIXEL_LAYOUT_K"] = int(name.removeprefix("layout8k")) if name.startswith("layout8k") else 0
            for reduction, flag in enumerate((None, "USE_SUBGROUP_ADD", "USE_SUBGROUP_ADD_NO_SHMEM")):
                defs = {**definitions, **variants[name]}
                if flag:
                    defs[flag] = "1"
                command = docker + ["env", "LD_LIBRARY_PATH=/work/deps/sysroot/usr/lib/x86_64-linux-gnu",
                    "/work/deps/sysroot/usr/bin/glslc", "-fshader-stage=compute", "--target-env=vulkan1.2",
                    "-O", "-Werror", "-I/work/source/ggml/src/ggml-vulkan/vulkan-shaders",
                    *[f"-D{k}={v}" for k, v in defs.items()], "/diag/mul_mat_vec.comp",
                    "-o", f"/diag/{name}-{reduction}.spv"]
                shader_commands.append(command)
                subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
                blob = (output / f"{name}-{reduction}.spv").read_bytes()
                words = struct.unpack(f"<{len(blob) // 4}I", blob)
                if words[0] != 0x07230203:
                    raise RuntimeError("invalid SPIR-V header")
                header.append(f"static const uint32_t pixel_{name}_{reduction}[] = {{\n")
                for index in range(0, len(words), 8):
                    header.append("    " + ", ".join(f"0x{x:08x}" for x in words[index:index + 8]) + ",\n")
                header.append("};\n")
        header.append("struct pixel_shader_blob { const void * data; uint64_t len; };\n")
        header.append("static const char * pixel_shader_names[] = {" + ", ".join(f'"{n}"' for n in variant_values) + "};\n")
        header.append("static const pixel_shader_blob pixel_shader_code[][3] = {\n")
        for name in variant_values:
            header.append("    {" + ", ".join(f"{{pixel_{name}_{r}, sizeof(pixel_{name}_{r})}}" for r in range(3)) + "},\n")
        header.append("};\n")
        (output / "pixel_shader_data.hpp").write_text("".join(header))
        source = (previous / "ggml-vulkan.cpp").read_text()
        if args.batch_rows:
            source = source.replace("w == DMMV_WG_SIZE_SUBGROUP && i == 0",
                                    "w == DMMV_WG_SIZE_SUBGROUP && (i == 0 || i == 1 || i == 3 || i == 7)")
        updated = '#include <cstdint>\n#include "pixel_shader_data.hpp"\n' + source
        anchor = "    static constexpr uint32_t mul_mat_vec_num_bindings = 5;\n"
        selection = """    uint32_t pixel_f16_shader = 0;
    if (const char * name = getenv("S42_PIXEL_F16_SHADER")) {
        GGML_ASSERT(pixel_f16_wg != 0);
        for (uint32_t i = 0; i < sizeof(pixel_shader_names) / sizeof(pixel_shader_names[0]); ++i) {
            if (std::string(name) == pixel_shader_names[i]) {
                pixel_f16_shader = i + 1;
            }
        }
        GGML_ASSERT(pixel_f16_shader != 0);
        std::cerr << "S42PIXELSHADER name=" << name << " precision=f32 cols=1" << std::endl;
    }
"""
        if updated.count(anchor) != 1:
            raise RuntimeError("backend setup differs")
        updated = updated.replace(anchor, selection + anchor)
        anchor = '            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32", arr_dmmv_f16_f32_f32_len[f16_reduc], arr_dmmv_f16_f32_f32_data[f16_reduc],'
        replacement = """            const bool pixel_dense = pixel_f16 && pixel_f16_shader != 0;
            const auto f16_code = pixel_dense ? pixel_shader_code[pixel_f16_shader - 1][f16_reduc].data : arr_dmmv_f16_f32_f32_data[f16_reduc];
            const auto f16_len = pixel_dense ? pixel_shader_code[pixel_f16_shader - 1][f16_reduc].len : arr_dmmv_f16_f32_f32_len[f16_reduc];
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32", f16_len, f16_code,"""
        if updated.count(anchor) != 1:
            raise RuntimeError("backend pipeline differs")
        updated = updated.replace(anchor, replacement)
        if args.fuse_swiglu:
            before_fusion = updated
            updated = pixel_swiglu_fusion.backend(updated)
            (output / "SWIGLU_BACKEND.patch").write_text("".join(difflib.unified_diff(
                before_fusion.splitlines(keepends=True), updated.splitlines(keepends=True),
                fromfile="dense/ggml-vulkan.cpp", tofile="fused/ggml-vulkan.cpp")))
        if args.dispatch_counts:
            from pixel_dispatch_counts import transform
            updated = transform(updated, args.batch_rows, args.maximum_k)
            (output / "pixel_dispatch_counts.py").write_bytes((base / "pixel_dispatch_counts.py").read_bytes())
        (output / "ggml-vulkan.cpp").write_text(updated)
        (output / "DENSE_BACKEND.patch").write_text("".join(difflib.unified_diff(
            source.splitlines(keepends=True), updated.splitlines(keepends=True),
            fromfile="qualified-tuned/ggml-vulkan.cpp", tofile="dense/ggml-vulkan.cpp")))
        save(output / "SHADER_COMMANDS.json", shader_commands)
        save(output / "BUILD_COMMANDS.json", commands)
        for command in commands:
            subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    invocation = [str(Path(__file__)), "--output", str(output)]
    if args.fuse_swiglu:
        invocation.append("--fuse-swiglu")
    if args.aligned_k:
        invocation.append("--aligned-k")
    invocation += ["--maximum-k", str(args.maximum_k)]
    if args.batch_rows:
        invocation.append("--batch-rows")
    if args.dispatch_counts:
        invocation.append("--dispatch-counts")
    if args.batch_shared:
        invocation.append("--batch-shared")
    if args.f16_layout:
        invocation.append("--f16-layout")
    save(output / "BUILD_PROVENANCE.json", {
        "status": "PASS", "at": datetime.now(timezone.utc).isoformat(), "variants": variants,
        "source_sha256": sha(output / "ggml-vulkan.cpp"), "shader_sha256": sha(output / "mul_mat_vec.comp"),
        "snippet_sha256": sha(base / "pixel_dense_gemv.glsl"), "builder_sha256": sha(Path(__file__)),
        "library_sha256": sha(output / "libggml-vulkan.so"),
        "swiglu_fusion": args.fuse_swiglu,
        "dense_aligned_k": args.aligned_k,
        "maximum_k": args.maximum_k, "batch_rows": args.batch_rows,
        "dispatch_counts": args.dispatch_counts,
        "batch_shared": args.batch_shared,
        "f16_layout": args.f16_layout,
        "swiglu_patch_sha256": sha(output / "pixel_swiglu_fusion.py") if args.fuse_swiglu else None,
        "base_shader_sha256": sha(output / "mul_mat_vec_base.glsl") if args.fuse_swiglu else None,
        "spirv_sha256": {p.name: sha(p) for p in output.glob("*.spv")},
        "compiler_version": "shaderc 2023.8-1build1; SPIR-V Tools 2023.6; glslang 14.0.0",
        "qualified_backend_sha256": provenance["source_sha256"],
        "reused_shader_objects": 132, "precision": "F16 weights, F32 input and accumulation",
        "physical_validation": "NOT_RUN", "invocation": shlex.join(invocation)})
    print("BUILD PASS", sha(output / "libggml-vulkan.so"), flush=True)


if __name__ == "__main__":
    main()
