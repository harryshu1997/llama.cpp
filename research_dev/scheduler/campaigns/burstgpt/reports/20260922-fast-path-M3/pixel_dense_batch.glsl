void pixel_dense_batch_outputs(const uint first_row, const uint num_rows) {
    const uint tid = gl_LocalInvocationID.x;
    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS];
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < NUM_ROWS; ++n) temp[j][n] = 0.0;
    }
    for (uint base = 0; base < p.ncols; base += BLOCK_SIZE * 4) {
        const uint col = base + tid * 4;
        if (col < p.ncols) {
            vec4 b[NUM_COLS];
            [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                b[j] = data_b_v4[(b_offset + j * p.batch_stride_b + col) / 4];
            }
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                const vec4 a = vec4(data_a_v4[(a_offset + (first_row + n) * p.ncols + col) / 4]);
                [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) temp[j][n] += dot(a, b[j]);
            }
        }
    }
    reduce_result(temp, d_offset, first_row, num_rows, tid);
}
