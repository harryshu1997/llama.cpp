shared vec4 pixel_input_tile[NUM_COLS][64];
shared float pixel_partial[NUM_COLS][128];

void pixel_dense_batch_shared_outputs(const uint first_row, const uint num_rows) {
    const uint tid = gl_LocalInvocationID.x;
    const uint row = tid / 16;
    const uint lane = tid % 16;
    float accum[NUM_COLS];
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) accum[j] = 0.0;

    for (uint base = 0; base < p.ncols; base += 256) {
        for (uint index = tid; index < NUM_COLS * 64; index += 128) {
            const uint j = index / 64;
            const uint k = index % 64;
            pixel_input_tile[j][k] = base + k * 4 < p.ncols ?
                data_b_v4[(b_offset + j * p.batch_stride_b + base) / 4 + k] : vec4(0.0);
        }
        barrier();
        if (row < num_rows) {
            [[unroll]] for (uint step = 0; step < 4; ++step) {
                const uint k = lane + 16 * step;
                const uint col = base + k * 4;
                if (col < p.ncols) {
                    const vec4 a = vec4(data_a_v4[(a_offset + (first_row + row) * p.ncols + col) / 4]);
                    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                        accum[j] += dot(a, pixel_input_tile[j][k]);
                    }
                }
            }
        }
        barrier();
    }
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) pixel_partial[j][tid] = accum[j];
    barrier();
    [[unroll]] for (uint stride = 8; stride > 0; stride >>= 1) {
        if (lane < stride) {
            [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                pixel_partial[j][tid] += pixel_partial[j][tid + stride];
            }
        }
        barrier();
    }
    if (lane == 0 && row < num_rows) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            const uint index = j * p.batch_stride_d + d_offset + first_row + row;
            float value = pixel_partial[j][tid];
            if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) value += float(data_fuse0[index]);
            if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) value += float(data_fuse1[index]);
            data_d[index] = D_TYPE(value);
        }
    }
}
