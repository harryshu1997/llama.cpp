"""Build an isolated F16 GEMV specialization sweep from the qualified snapshot."""

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


EXPECTED_SOURCE = "5586ee49e4dff91ba31b6f6f58f999122db9abe42e036a3e851847af02699990"
SETTINGS = """
    uint32_t pixel_f16_wg = 0;
    uint32_t pixel_f16_rows = 2;
    uint32_t pixel_f16_subgroup = subgroup_size;
    const char * pixel_wg_env = getenv("S42_PIXEL_F16_WG");
    const char * pixel_rows_env = getenv("S42_PIXEL_F16_ROWS");
    const char * pixel_subgroup_env = getenv("S42_PIXEL_F16_SUBGROUP");
    if (pixel_wg_env || pixel_rows_env || pixel_subgroup_env) {
        GGML_ASSERT(pixel_wg_env && pixel_rows_env);
        GGML_ASSERT(std::string(device->properties.deviceName.data()).find("PowerVR") == 0);
        GGML_ASSERT(use_subgroups && subgroup_size == 128);
        char * wg_end = nullptr;
        char * rows_end = nullptr;
        const unsigned long wg = strtoul(pixel_wg_env, &wg_end, 10);
        const unsigned long rows = strtoul(pixel_rows_env, &rows_end, 10);
        GGML_ASSERT(*pixel_wg_env && *pixel_rows_env && !*wg_end && !*rows_end);
        if (pixel_subgroup_env) {
            char * subgroup_end = nullptr;
            const unsigned long sg = strtoul(pixel_subgroup_env, &subgroup_end, 10);
            GGML_ASSERT(*pixel_subgroup_env && !*subgroup_end);
            GGML_ASSERT(sg == 32 || sg == 64 || sg == 128);
            GGML_ASSERT(device->subgroup_size_control);
            GGML_ASSERT(sg >= device->subgroup_min_size && sg <= device->subgroup_max_size);
            pixel_f16_subgroup = uint32_t(sg);
        }
        GGML_ASSERT(wg == 32 || wg == 64 || wg == 128 || wg == 256 || wg == 512);
        GGML_ASSERT(wg >= pixel_f16_subgroup && wg % pixel_f16_subgroup == 0);
        GGML_ASSERT(rows == 1 || rows == 2 || rows == 4 || rows == 8);
        GGML_ASSERT(wg <= device->properties.limits.maxComputeWorkGroupInvocations);
        GGML_ASSERT(wg <= device->properties.limits.maxComputeWorkGroupSize[0]);
        GGML_ASSERT(wg * rows * sizeof(float) <= device->properties.limits.maxComputeSharedMemorySize);
        pixel_f16_wg = uint32_t(wg);
        pixel_f16_rows = uint32_t(rows);
        std::cerr << "S42PIXELGEMV wg=" << pixel_f16_wg << " rows=" << pixel_f16_rows
                  << " subgroup=" << pixel_f16_subgroup << " cols=1" << std::endl;
    }
"""
PIPELINE = """            const bool pixel_f16 = pixel_f16_wg && w == DMMV_WG_SIZE_SUBGROUP && i == 0;
            const uint32_t f16_wg = pixel_f16 ? pixel_f16_wg : wg_size_subgroup;
            const uint32_t f16_rows = pixel_f16 ? pixel_f16_rows : 2;
            const uint32_t f16_subgroup = pixel_f16 ? pixel_f16_subgroup : force_subgroup_size;
            const shader_reduction_mode f16_reduc = pixel_f16 ?
                (f16_wg == f16_subgroup ? SHADER_REDUCTION_MODE_SUBGROUP : SHADER_REDUCTION_MODE_HYBRID) : reduc;
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32", arr_dmmv_f16_f32_f32_len[f16_reduc], arr_dmmv_f16_f32_f32_data[f16_reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {f16_rows, 1, 1}, {f16_wg, f16_rows, i+1}, 1, false, use_subgroups, f16_subgroup);
"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    original = base / "software/pixel10pro-vulkan"
    previous = base / "software/pixel10pro-named-profile"
    source_path = original / "source/ggml/src/ggml-vulkan/ggml-vulkan.cpp"
    if sha(source_path) != EXPECTED_SOURCE:
        raise RuntimeError("qualified source snapshot changed")
    source = source_path.read_text()
    marker = "    static constexpr uint32_t mul_mat_vec_num_bindings = 5;\n"
    if source.count(marker) != 1:
        raise RuntimeError("pipeline setup anchor differs")
    updated = source.replace(marker, SETTINGS + marker)
    matches = [line for line in updated.splitlines(keepends=True)
               if "pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i]," in line]
    if len(matches) != 1:
        raise RuntimeError("F16 pipeline anchor differs")
    updated = updated.replace(matches[0], PIPELINE)
    output = args.output.resolve()
    output.mkdir()
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (output / "ggml-vulkan.cpp").write_text(updated)
    (output / "GEMV_TUNE.patch").write_text("".join(difflib.unified_diff(
        source.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile="qualified/ggml-vulkan.cpp", tofile="tuned/ggml-vulkan.cpp")))
    commands = json.loads((previous / "BUILD_COMMANDS.json").read_text())
    commands = [[str(output) + ":/diag" if item == str(previous) + ":/diag" else item
                 for item in command] for command in commands]
    if not all(str(output) + ":/diag" in command for command in commands):
        raise RuntimeError("isolated build mount differs")
    write_json(output / "BUILD_COMMANDS.json", commands)
    (output / "REUSED_OBJECT_SHA256.json").write_bytes((previous / "REUSED_OBJECT_SHA256.json").read_bytes())
    with (output / "BUILD.log").open("x") as log:
        for command in commands:
            subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    write_json(output / "BUILD_PROVENANCE.json", {
        "status": "PASS", "finished_at": datetime.now(timezone.utc).isoformat(),
        "qualified_source_sha256": EXPECTED_SOURCE,
        "source_sha256": sha(output / "ggml-vulkan.cpp"),
        "library_sha256": sha(output / "libggml-vulkan.so"),
        "builder_sha256": sha(Path(__file__)),
        "precision": "F16 weights, F32 input and accumulator",
        "scope": "PowerVR opt-in, F16/F32 GEMV with one input column only",
        "shader_objects_reused": True, "physical_validation": "NOT_RUN"})
    print(output, "BUILD PASS", sha(output / "libggml-vulkan.so"), flush=True)


if __name__ == "__main__":
    main()
