"""Count actual matvec dispatches by input rows in a private Vulkan build."""

from build_pixel_cpu_gpu import replace_once


def transform(source, batch, maximum_k):
    source = replace_once(source, "    int num_additional_fused_ops {};", """    int num_additional_fused_ops {};
    uint64_t pixel_f16_dispatches[8] = {};
    uint64_t pixel_packed_dispatches[8] = {};
    uint64_t pixel_tiled_dispatches[8] = {};
    const bool pixel_custom_shader = getenv("S42_PIXEL_F16_SHADER") != nullptr;""")
    start = source.index("static void ggml_vk_mul_mat_vec_q_f16(")
    end = source.index("static void ggml_vk_mul_mat_vec_p021_f16_f32(", start)
    part = source[start:end]
    counter = f"""        if (src0->type == GGML_TYPE_F16) ++ctx->pixel_f16_dispatches[ne11 - 1];
        if (ggml_is_quantized(src0->type)) ++ctx->pixel_packed_dispatches[ne11 - 1];
        if ({str(batch).lower()} && ctx->pixel_custom_shader && src0->type == GGML_TYPE_F16 &&
            (ne11 == 2 || ne11 == 4 || ne11 == 8) && ne00 >= 64 && ne00 <= {maximum_k} && ne00 % 64 == 0 &&
            !x_non_contig && !y_non_contig && f16_f32_kernel && !quantize_y &&
            dmmv == ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[DMMV_WG_SIZE_SUBGROUP][GGML_TYPE_F16][ne11-1]) {{
            ++ctx->pixel_tiled_dispatches[ne11 - 1];
        }}
        base_work_group_y += groups_y;"""
    part = replace_once(part, "        base_work_group_y += groups_y;", counter)
    source = source[:start] + part + source[end:]
    start = source.index("static void ggml_backend_vk_free(ggml_backend_t backend) {")
    end = source.index("\n}", start)
    part = source[start:end]
    part = replace_once(part, "    ggml_vk_cleanup(ctx);", """    for (int i = 0; i < 8; ++i) {
        if (ctx->pixel_f16_dispatches[i] || ctx->pixel_packed_dispatches[i]) {
            std::cerr << "S42PIXELDISPATCH rows=" << i+1 << " f16=" << ctx->pixel_f16_dispatches[i]
                      << " packed=" << ctx->pixel_packed_dispatches[i] << " tiled=" << ctx->pixel_tiled_dispatches[i] << std::endl;
        }
    }
    ggml_vk_cleanup(ctx);""")
    return source[:start] + part + source[end:]
