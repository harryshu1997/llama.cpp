"""Correct native CPU Q8_K activation rounding using a second packed dot."""

from build_pixel_cpu_gpu import replace_once


KERNEL = r'''
void pixel_q8_residual(ggml_tensor * dst, const ggml_tensor * input, int ith, int nth, void *) {
    GGML_ASSERT(input->type == GGML_TYPE_F32 && ggml_is_contiguous(input));
    GGML_ASSERT(ggml_nelements(input) % 256 == 0);
    const auto quantize = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    const auto dequantize = ggml_get_type_traits(GGML_TYPE_Q8_K)->to_float;
    alignas(64) uint8_t packed[512];
    alignas(64) float decoded[256];
    GGML_ASSERT(ggml_type_size(GGML_TYPE_Q8_K) <= sizeof(packed));
    const float * source = static_cast<const float *>(input->data);
    float * output = static_cast<float *>(dst->data);
    for (int64_t block = ith; block < ggml_nelements(input) / 256; block += nth) {
        const float * row = source + block * 256;
        quantize(row, packed, 256);
        dequantize(packed, decoded, 256);
        for (int i = 0; i < 256; ++i) output[block * 256 + i] = row[i] - decoded[i];
    }
}

ggml_tensor * pixel_residual_matmul(ggml_context * ctx, ggml_tensor * weights,
                                   ggml_tensor * input, bool correction) {
    ggml_tensor * result = ggml_mul_mat(ctx, weights, input);
    if (!correction) return result;
    GGML_ASSERT(weights->type == GGML_TYPE_Q4_K || weights->type == GGML_TYPE_Q6_K);
    ggml_tensor * residual = ggml_map_custom1(ctx, input, pixel_q8_residual, GGML_N_TASKS_MAX, nullptr);
    return ggml_add(ctx, result, ggml_mul_mat(ctx, weights, residual));
}
'''


def transform(source):
    source = replace_once(source, "host_tensor pixel_join(", KERNEL + "\nhost_tensor pixel_join(")
    source = replace_once(source, '    const bool pixel_packed = getenv("S42_PIXEL_PACKED_WEIGHTS") != nullptr;', r'''
    const bool pixel_packed = getenv("S42_PIXEL_PACKED_WEIGHTS") != nullptr;
    const bool pixel_correction = getenv("S42_PIXEL_CPU_QUANT_RESIDUAL") != nullptr;
    if (pixel_correction && (!pixel_packed || cfg.backend != "CPU")) {
        fprintf(stderr, "[ffn-worker] residual correction requires packed CPU weights\n");
        return 2;
    }
    if (pixel_correction) fprintf(stderr, "S42PIXELRESIDUAL activation=q8_K correction=second_dot\n");''')
    source = replace_once(source, "    for (int layer = 0; layer < 64; ++layer) {", r'''
    if (getenv("S42_PIXEL_CPU_PAIR_BLOCKS")) {
        if (dual || cfg.backend != "CPU" || !pixel_packed || cfg.columns != 17408 || cfg.column_quantum != 4352) {
            fprintf(stderr, "[ffn-worker] CPU block pairing requires packed CPU-only mode\n");
            return 2;
        }
        runtime_columns = {8704, 17408};
        partition_columns = {8704, 8704};
        fprintf(stderr, "S42PIXELCPUPAIR full_projections=6 half_projections=3\n");
    }
    for (int layer = 0; layer < 64; ++layer) {''')
    start = source.index("    auto build_graph = [&]")
    end = source.index("    for (size_t arena_index = 0;", start)
    part = source[start:end]
    for name, activation in (("gate", "input"), ("up", "input"), ("down", "activation")):
        old = f"ggml_mul_mat(\n                    graph_ctx, block.{name}_weight, {activation})"
        part = replace_once(part, old, f"pixel_residual_matmul(\n                    graph_ctx, block.{name}_weight, {activation}, pixel_correction)")
    return source[:start] + part + source[end:]
