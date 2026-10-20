void pixel_dense_outputs(const uint first_row, const uint num_rows) {
    const uint tid = gl_LocalInvocationID.x;
    float accum[PIXEL_ACCUM][NUM_ROWS];
    [[unroll]] for (uint a = 0; a < PIXEL_ACCUM; ++a) {
        [[unroll]] for (uint n = 0; n < NUM_ROWS; ++n) {
            accum[a][n] = 0.0;
        }
    }

    for (uint base = 0; base < p.ncols; base += BLOCK_SIZE * PIXEL_WIDTH * PIXEL_UNROLL) {
        [[unroll]] for (uint u = 0; u < PIXEL_UNROLL; ++u) {
            const uint block = base + u * BLOCK_SIZE * PIXEL_WIDTH;
#if PIXEL_STRIPED
            vec4 b = vec4(0.0);
            [[unroll]] for (uint c = 0; c < 4; ++c) {
                const uint col = block + tid + c * BLOCK_SIZE;
                if (col < p.ncols) {
                    b[c] = data_b[b_offset + col];
                }
            }
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                vec4 a = vec4(0.0);
                [[unroll]] for (uint c = 0; c < 4; ++c) {
                    const uint col = block + tid + c * BLOCK_SIZE;
                    if (col < p.ncols) {
                        a[c] = float(data_a[a_offset + (first_row + n) * p.ncols + col]);
                    }
                }
                accum[u % PIXEL_ACCUM][n] += dot(a, b);
            }
#else
            const uint col = block + tid * PIXEL_WIDTH;
            if (col < p.ncols) {
                vec4 b[PIXEL_WIDTH / 4];
                [[unroll]] for (uint v = 0; v < PIXEL_WIDTH / 4; ++v) {
                    b[v] = data_b_v4[(b_offset + col) / 4 + v];
                }
                [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                    const uint offset = a_offset + (first_row + n) * p.ncols + col;
                    [[unroll]] for (uint v = 0; v < PIXEL_WIDTH / 4; ++v) {
#if PIXEL_PAIRED
                        const vec4 a = vec4(data_a_packed32[offset / 2 + 2 * v],
                                           data_a_packed32[offset / 2 + 2 * v + 1]);
#else
                        const vec4 a = vec4(data_a_v4[offset / 4 + v]);
#endif
                        accum[u % PIXEL_ACCUM][n] += dot(a, b[v]);
                    }
                }
            }
#endif
        }
    }

    FLOAT_TYPE temp[NUM_COLS][NUM_ROWS];
    [[unroll]] for (uint n = 0; n < num_rows; ++n) {
        temp[0][n] = accum[0][n];
        [[unroll]] for (uint a = 1; a < PIXEL_ACCUM; ++a) {
            temp[0][n] += accum[a][n];
        }
    }
    reduce_result(temp, d_offset, first_row, num_rows, tid);
}
