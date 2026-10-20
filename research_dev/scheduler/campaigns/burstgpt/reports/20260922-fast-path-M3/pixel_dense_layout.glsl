#if PIXEL_LAYOUT_K
shared float pixel_layout_sums[BLOCK_SIZE];

void pixel_layout_outputs(const uint first_row, const uint num_rows) {
    const uint tid = gl_LocalInvocationID.x;
    const uint row = tid % 8;
    const uint lane = tid / 8;
    for (uint batch = 0; batch < NUM_COLS; ++batch) {
        float sum = 0.0;
        for (uint col = lane * 4; col < p.ncols; col += (BLOCK_SIZE / 8) * 4) {
            const uint offset = a_offset + first_row * p.ncols +
                (col / PIXEL_LAYOUT_K) * 8 * PIXEL_LAYOUT_K + row * PIXEL_LAYOUT_K + col % PIXEL_LAYOUT_K;
            const vec4 weights = vec4(data_a_v4[offset / 4]);
            const vec4 activation = data_b_v4[(batch * p.batch_stride_b + b_offset + col) / 4];
            sum += dot(weights, activation);
        }
        pixel_layout_sums[tid] = sum;
        barrier();
        if (tid < num_rows) {
            float total = pixel_layout_sums[tid];
            for (uint index = tid + 8; index < BLOCK_SIZE; index += 8) total += pixel_layout_sums[index];
            if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) total += float(data_fuse0[batch * p.batch_stride_d + d_offset + first_row + tid]);
            if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) total += float(data_fuse1[batch * p.batch_stride_d + d_offset + first_row + tid]);
            data_d[batch * p.batch_stride_d + d_offset + first_row + tid] = D_TYPE(total);
        }
        barrier();
    }
}
#endif
