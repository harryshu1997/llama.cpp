"""Apply a bounded F16 matvec/SwiGLU fusion to the isolated Pixel backend."""


def replace(source, before, after, count=1):
    if source.count(before) != count:
        raise RuntimeError("fusion patch anchor differs: " + before[:100])
    return source.replace(before, after)


def shader_base(source):
    anchor = "layout (constant_id = 0) const uint BLOCK_SIZE = 32;"
    helper = """float pixel_swiglu_epilogue(float value, uint index) {
    if ((p.fusion_flags & 0x10u) != 0u) {
        const float gate = float(data_fuse0[index]);
        return gate / (1.0f + exp(-gate)) * value;
    }
    return value;
}

"""
    source = replace(source, anchor, helper + anchor)
    index = "j*p.batch_stride_d + d_offset + first_row + n"
    for value, count in (("temp[j][n]", 2), ("tmpsh[j][n][0]", 1)):
        source = replace(source, f"data_d[{index}] = D_TYPE({value});",
                         f"data_d[{index}] = D_TYPE(pixel_swiglu_epilogue({value}, {index}));", count)
    return source


def backend(source):
    source = replace(source, "    bool disable_fusion;",
                     "    bool disable_fusion;\n    bool pixel_fuse_swiglu = false;")
    source = replace(source, "    int num_additional_fused_ops {};",
                     "    int num_additional_fused_ops {};\n    uint64_t pixel_swiglu_dispatches = 0;")
    anchor = "    static constexpr uint32_t mul_mat_vec_num_bindings = 5;"
    setup = """    if (const char * enabled = getenv("S42_PIXEL_FUSE_SWIGLU")) {
        GGML_ASSERT(std::string(enabled) == "0" || std::string(enabled) == "1");
        device->pixel_fuse_swiglu = std::string(enabled) == "1";
        GGML_ASSERT(!device->pixel_fuse_swiglu || pixel_f16_shader != 0);
    }
"""
    source = replace(source, anchor, setup + anchor)
    source = replace(source, "#define MAT_VEC_FUSION_FLAGS_SCALE1 0x8",
                     "#define MAT_VEC_FUSION_FLAGS_SCALE1 0x8\n#define MAT_VEC_FUSION_FLAGS_SWIGLU 0x10")
    anchor = "    auto const &mm_add_ok = [&](const ggml_tensor *mul, const ggml_tensor *add) {"
    guard = """    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_GLU) {
        if (!ctx->device->pixel_fuse_swiglu) {
            return false;
        }
        const ggml_tensor * mul = cgraph->nodes[node_idx];
        const ggml_tensor * glu = cgraph->nodes[node_idx + 1];
        const ggml_tensor * gate = glu->src[0];
        const ggml_tensor * weight = mul->src[0];
        const ggml_tensor * input = mul->src[1];
        if (glu->src[1] != mul || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
            weight->type != GGML_TYPE_F16 || input->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32 ||
            weight->ne[0] != 5120 || weight->ne[1] != 4352 || weight->ne[2] != 1 || weight->ne[3] != 1 ||
            input->ne[0] != 5120 || ggml_nrows(input) != 1 || ggml_nrows(mul) != 1 ||
            !ggml_are_same_shape(mul, gate) || !ggml_are_same_shape(mul, glu)) {
            return false;
        }
        for (const ggml_tensor * tensor : { weight, input, mul, gate, glu }) {
            if (!ggml_is_contiguous(tensor) || get_misalign_bytes(ctx, tensor) != 0) {
                return false;
            }
        }
        return true;
    }
"""
    source = replace(source, anchor, guard + anchor)
    anchor = """            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_ADD })) {"""
    pattern = """            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_GLU })) {
                ctx->num_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_SWIGLU";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
"""
    source = replace(source, anchor, pattern + anchor)
    start = source.index("static void ggml_vk_mul_mat_vec_q_f16(")
    end = source.index("static void ggml_vk_mul_mat_vec_p021_f16_f32(", start)
    part = source[start:end]
    anchor = """    vk_subbuffer d_F0 = d_D;
    if (ctx->num_additional_fused_ops > 0) {"""
    dispatch = """    vk_subbuffer d_F0 = d_D;
    if (ctx->num_additional_fused_ops == 1 && cgraph->nodes[node_idx + 1]->op == GGML_OP_GLU) {
        GGML_ASSERT(ctx->device->pixel_fuse_swiglu && !quantize_y && f16_f32_kernel);
        d_F0 = ggml_vk_tensor_subbuffer(ctx, cgraph->nodes[node_idx + 1]->src[0]);
        fusion_flags |= MAT_VEC_FUSION_FLAGS_SWIGLU;
        ++ctx->pixel_swiglu_dispatches;
    } else if (ctx->num_additional_fused_ops > 0) {"""
    part = replace(part, anchor, dispatch)
    source = source[:start] + part + source[end:]
    start = source.index("static void ggml_backend_vk_free(ggml_backend_t backend) {")
    end = source.index("\n}", start)
    part = source[start:end]
    part = replace(part, "    ggml_vk_cleanup(ctx);", """    if (ctx->device->pixel_fuse_swiglu) {
        std::cerr << "S42PIXELFUSION op=MUL_MAT_SWIGLU dispatches=" << ctx->pixel_swiglu_dispatches << std::endl;
    }
    ggml_vk_cleanup(ctx);""")
    return source[:start] + part + source[end:]


def worker(source):
    anchor = """        size_t first_block = state.blocks.size();"""
    return replace(source, anchor, """        ggml_set_output(input);

""" + anchor)
