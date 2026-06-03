// gemm_noshuffle_q4_0_f32_emit.cl — LazyVLM design-delta V3.2 (in-GEMM fusion).
//
// = kernel_gemm_noshuffle_q4_0_f32 with the q4x4x2 emit FUSED into the matmul, so
// the weight is read ONCE (the emit reuses the ushorts the GEMM already loaded —
// no separate-pass re-read, which is what cost V3.1 its margin).
//
// The K loop is reordered to process, per t in [0,32), the TWO halves of a 256-elem
// super-block together: low-half K-group (sb*64+t) and high-half (sb*64+32+t). That
// puts both bits-vectors in-register at the same iteration, so q4x4x2's pairing
// q[j]=(nib[j+128]<<4)|nib[j] needs NO 32-deep buffer. The matmul accumulates the
// same set of K contributions (sum reordered only) -> mathematically equivalent GEMM.
//
// Only gy==0 work-items emit (one N-tile covers all M rows; avoids redundant writes).
// q4x4x2 row layout (== ggml-hexagon repack_row_q4x4x2): per super-block 128 quant
// bytes then 8 fp16 scales; row stride = K/2 + (K/256)*16.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_gemm_noshuffle_q4_0_f32_emit(
        global const ushort * src0_q,       // quantized A (noshuffle + transposed)
        global const half  * src0_d,        // A scales (transposed)
        __read_only image1d_buffer_t src1,  // B (1d image)
        global float * dst,                 // C
        int m,                              // M
        int n,                              // N with padding
        int k,                              // K
        int n_no_padding,                   // N without padding
        global uchar * wq,                  // out: q4x4x2 weight (V3.2)
        int row_stride                      // q4x4x2 row stride bytes = K/2 + (K/256)*16
) {
    int n_4 = n >> 2;

    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0; // 8x4 output elements
    half8 B;
    half4 dqw;
    __global const ushort* weight_ptr = src0_q + gx_2;
    __global const half*   scale_ptr  = src0_d + gx_2;

    int do_emit = (gy == 0);
    int nsb = k >> 8;  // k / 256

    for (int sb = 0; sb < nsb; ++sb) {
        int gbase = sb * 64;                 // first K-group of this super-block
        for (int t = 0; t < 32; ++t) {
            int glo = gbase + t;             // low-half K-group  (i_lo = glo*4)
            int ghi = gbase + 32 + t;        // high-half K-group (i_hi = ghi*4)
            int ilo = glo << 2;
            int ihi = ghi << 2;

            ushort4 blo = vload4(0, weight_ptr + (long)glo * m);
            ushort4 bhi = vload4(0, weight_ptr + (long)ghi * m);
            half4   slo = vload4(0, scale_ptr  + (long)(ilo >> 5) * m);
            half4   shi = vload4(0, scale_ptr  + (long)(ihi >> 5) * m);

            // ---- matmul: low-half 4 K-elements ----
            #define MM(BITS, SC, IK, MASK, SH)                                  \
                B.s0123 = read_imageh(src1, gy*2 + (IK)*n_4);                    \
                B.s4567 = read_imageh(src1, gy*2 + (IK)*n_4 + 1);               \
                dqw.s0 = ((((BITS).s0 & (MASK)) >> (SH)) - 8) * (SC).s0;        \
                dqw.s1 = ((((BITS).s1 & (MASK)) >> (SH)) - 8) * (SC).s1;        \
                dqw.s2 = ((((BITS).s2 & (MASK)) >> (SH)) - 8) * (SC).s2;        \
                dqw.s3 = ((((BITS).s3 & (MASK)) >> (SH)) - 8) * (SC).s3;        \
                c0 += B * dqw.s0; c1 += B * dqw.s1; c2 += B * dqw.s2; c3 += B * dqw.s3;

            MM(blo, slo, ilo + 0, 0x000F,  0);
            MM(blo, slo, ilo + 1, 0x00F0,  4);
            MM(blo, slo, ilo + 2, 0x0F00,  8);
            MM(blo, slo, ilo + 3, 0xF000, 12);
            MM(bhi, shi, ihi + 0, 0x000F,  0);
            MM(bhi, shi, ihi + 1, 0x00F0,  4);
            MM(bhi, shi, ihi + 2, 0x0F00,  8);
            MM(bhi, shi, ihi + 3, 0xF000, 12);
            #undef MM

            // ---- emit q4x4x2 (gy==0 only): q[sb*128 + t*4 + p] for each of 4 rows ----
            if (do_emit) {
                int qoff = sb * 128 + t * 4;
                #define EMITQ(LANE, ROW)                                                       \
                    {                                                                          \
                        ushort lo = blo.LANE, hi = bhi.LANE;                                   \
                        __global uchar* qp = wq + (long)(gx_2 + (ROW)) * row_stride + qoff;    \
                        uchar4 ov;                                                             \
                        ov.s0 = (uchar)(((((uint)hi >>  0) & 0xF) << 4) | (((uint)lo >>  0) & 0xF)); \
                        ov.s1 = (uchar)(((((uint)hi >>  4) & 0xF) << 4) | (((uint)lo >>  4) & 0xF)); \
                        ov.s2 = (uchar)(((((uint)hi >>  8) & 0xF) << 4) | (((uint)lo >>  8) & 0xF)); \
                        ov.s3 = (uchar)(((((uint)hi >> 12) & 0xF) << 4) | (((uint)lo >> 12) & 0xF)); \
                        vstore4(ov, 0, qp);                                                    \
                    }
                EMITQ(s0, 0); EMITQ(s1, 1); EMITQ(s2, 2); EMITQ(s3, 3);
                #undef EMITQ

                // scales: at t%8==0, slo is block (sb*8 + t/8), shi is block (sb*8+4 + t/8)
                if ((t & 7) == 0) {
                    int blk_lo = sb * 8 + (t >> 3);
                    int blk_hi = sb * 8 + 4 + (t >> 3);
                    #define EMITS(SC, BLK, LANE, ROW)                                                \
                        {                                                                            \
                            ushort hh = as_ushort((SC).LANE);                                        \
                            __global uchar* dp = wq + (long)(gx_2 + (ROW)) * row_stride + (k >> 1) + (BLK) * 2; \
                            dp[0] = (uchar)(hh & 0xFF); dp[1] = (uchar)(hh >> 8);                     \
                        }
                    EMITS(slo, blk_lo, s0, 0); EMITS(slo, blk_lo, s1, 1); EMITS(slo, blk_lo, s2, 2); EMITS(slo, blk_lo, s3, 3);
                    EMITS(shi, blk_hi, s0, 0); EMITS(shi, blk_hi, s1, 1); EMITS(shi, blk_hi, s2, 2); EMITS(shi, blk_hi, s3, 3);
                    #undef EMITS
                }
            }
        }
    }

    int idx = (gy<<3)*m + (gx<<2); // vectorized store 16 elements
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx); idx += m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx); idx += m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx); idx += m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx); idx += m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx); idx += m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx); idx += m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx); idx += m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx); }
}
