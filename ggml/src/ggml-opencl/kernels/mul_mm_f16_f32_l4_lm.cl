#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LOAD_VEC_A 4
#define LOAD_VEC_B 4

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 8

// src0 (weights, fp16) is bound as a CL_RGBA / CL_HALF_FLOAT image1d_buffer so the
// inner-loop weight loads hit the Adreno texture cache (much higher hit rate than
// the buffer cache, since the same A tile is reused across BN/N tiles within a
// prefill kernel call). Element layout is identical to the original half4 buffer
// — each texel is one half4. offset0 is in bytes (matches buffer convention) and
// is converted to a texel offset inside the kernel.
kernel void kernel_mul_mm_f16_f32_l4_lm(
    read_only image1d_buffer_t src0_img,
    ulong offset0,
    global float4 * src1,
    ulong offset1,
    global float * dst,
    ulong offsetd,

    int ne00,
    int ne01,
    int ne02,
    int ne11,
    int ne12,

    int stride_a,
    int stride_b,
    int stride_d,

    int batch_stride_a,
    int batch_stride_b,
    int batch_stride_d,

    int r2,
    int r3
) {
    src1 = (global float4*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);
    // image1d_buffer is element-indexed (texel = half4 = 8 bytes here), so convert
    // the byte offset once up-front.
    const int src0_off_t = (int)(offset0 / (sizeof(half) * LOAD_VEC_A));

    local half  buf_a[BM * BK];
    local float buf_b[BN * BK];

    const int batch_idx = get_global_id(2);

    const int i13 = batch_idx / ne12;
    const int i12 = batch_idx % ne12;

    const int i03 = i13 / r3;
    const int i02 = i12 / r2;

    const int batch_idx_a = i03 * ne02 + i02;

    const int ir = get_group_id(0);
    const int ic = get_group_id(1);

    const int tid = get_local_id(0);
    const int th_r  = tid % (BM / TM);
    const int th_c  = tid / (BM / TM);

    const int loadr_a = get_local_id(0) % (BK / LOAD_VEC_A);
    const int loadc_a = get_local_id(0) / (BK / LOAD_VEC_A);
    const int loadr_b = get_local_id(0) % (BK / LOAD_VEC_B);
    const int loadc_b = get_local_id(0) / (BK / LOAD_VEC_B);

    const int loadstride_a = get_local_size(0) * LOAD_VEC_A / BK;
    const int loadstride_b = get_local_size(0) * LOAD_VEC_B / BK;

    int pos_a = (batch_idx_a * batch_stride_a + ir * BM * stride_a) / LOAD_VEC_A;
    int pos_b = (batch_idx   * batch_stride_b + ic * BN * stride_b) / LOAD_VEC_B;

    // sums laid out as TN rows of TM columns, matching the original cc*TM + cr indexing.
    float sums[TM * TN];
    for (int i = 0; i < TM * TN; i++) {
        sums[i] = 0.0f;
    }

    for (int block = 0; block < ne00; block += BK) {
        for (int l = 0; l < BM; l += loadstride_a) {
            if (ir*BM + loadc_a + l < ne01) {
                const int idx = pos_a + (loadc_a + l) * stride_a / LOAD_VEC_A + loadr_a;
                const half4 t = read_imageh(src0_img, src0_off_t + idx);
                buf_a[(loadr_a * LOAD_VEC_A + 0) * BM + loadc_a + l] = t.s0;
                buf_a[(loadr_a * LOAD_VEC_A + 1) * BM + loadc_a + l] = t.s1;
                buf_a[(loadr_a * LOAD_VEC_A + 2) * BM + loadc_a + l] = t.s2;
                buf_a[(loadr_a * LOAD_VEC_A + 3) * BM + loadc_a + l] = t.s3;
            } else {
                buf_a[(loadr_a * LOAD_VEC_A + 0) * BM + loadc_a + l] = 0.0h;
                buf_a[(loadr_a * LOAD_VEC_A + 1) * BM + loadc_a + l] = 0.0h;
                buf_a[(loadr_a * LOAD_VEC_A + 2) * BM + loadc_a + l] = 0.0h;
                buf_a[(loadr_a * LOAD_VEC_A + 3) * BM + loadc_a + l] = 0.0h;
            }
        }

        for (int l = 0; l < BN; l += loadstride_b) {
            if (ic*BN + loadc_b + l < ne11) {
                const int idx = pos_b + (loadc_b + l) * stride_b / LOAD_VEC_B + loadr_b;
                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = src1[idx].s0;
                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = src1[idx].s1;
                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = src1[idx].s2;
                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = src1[idx].s3;
            } else {
                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = 0.0h;
                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = 0.0h;
                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = 0.0h;
                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = 0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        pos_a += BK / LOAD_VEC_A;
        pos_b += BK / LOAD_VEC_B;

        // Inner-K loop. With TM=4 we fetch a half4 in one LDS load per K step, and
        // with TN=8 we fetch two float4s. convert_float is hoisted out of the mad so
        // the 4 fp16->fp32 conversions are amortised across the 32 mads.
        for (int i = 0; i < BK; i++) {
            const int row_a_off = i * BM + th_r * TM;
            const int row_b_off = i * BN + th_c * TN;

            // 1× half4 LDS load -> 4 fp32 values for A
            const half4  ca_h = vload4(0, &buf_a[row_a_off]);
            const float4 ca   = convert_float4(ca_h);
            // 2× float4 LDS loads for B (TN=8)
            const float4 cb0 = vload4(0, &buf_b[row_b_off]);
            const float4 cb1 = vload4(0, &buf_b[row_b_off + 4]);

            // Manually unrolled 4×8 outer product: each of the 8 B values multiplies
            // the same float4 of A and accumulates into 4 contiguous sums entries.
            sums[0*TM + 0] = mad(ca.s0, cb0.s0, sums[0*TM + 0]);
            sums[0*TM + 1] = mad(ca.s1, cb0.s0, sums[0*TM + 1]);
            sums[0*TM + 2] = mad(ca.s2, cb0.s0, sums[0*TM + 2]);
            sums[0*TM + 3] = mad(ca.s3, cb0.s0, sums[0*TM + 3]);

            sums[1*TM + 0] = mad(ca.s0, cb0.s1, sums[1*TM + 0]);
            sums[1*TM + 1] = mad(ca.s1, cb0.s1, sums[1*TM + 1]);
            sums[1*TM + 2] = mad(ca.s2, cb0.s1, sums[1*TM + 2]);
            sums[1*TM + 3] = mad(ca.s3, cb0.s1, sums[1*TM + 3]);

            sums[2*TM + 0] = mad(ca.s0, cb0.s2, sums[2*TM + 0]);
            sums[2*TM + 1] = mad(ca.s1, cb0.s2, sums[2*TM + 1]);
            sums[2*TM + 2] = mad(ca.s2, cb0.s2, sums[2*TM + 2]);
            sums[2*TM + 3] = mad(ca.s3, cb0.s2, sums[2*TM + 3]);

            sums[3*TM + 0] = mad(ca.s0, cb0.s3, sums[3*TM + 0]);
            sums[3*TM + 1] = mad(ca.s1, cb0.s3, sums[3*TM + 1]);
            sums[3*TM + 2] = mad(ca.s2, cb0.s3, sums[3*TM + 2]);
            sums[3*TM + 3] = mad(ca.s3, cb0.s3, sums[3*TM + 3]);

            sums[4*TM + 0] = mad(ca.s0, cb1.s0, sums[4*TM + 0]);
            sums[4*TM + 1] = mad(ca.s1, cb1.s0, sums[4*TM + 1]);
            sums[4*TM + 2] = mad(ca.s2, cb1.s0, sums[4*TM + 2]);
            sums[4*TM + 3] = mad(ca.s3, cb1.s0, sums[4*TM + 3]);

            sums[5*TM + 0] = mad(ca.s0, cb1.s1, sums[5*TM + 0]);
            sums[5*TM + 1] = mad(ca.s1, cb1.s1, sums[5*TM + 1]);
            sums[5*TM + 2] = mad(ca.s2, cb1.s1, sums[5*TM + 2]);
            sums[5*TM + 3] = mad(ca.s3, cb1.s1, sums[5*TM + 3]);

            sums[6*TM + 0] = mad(ca.s0, cb1.s2, sums[6*TM + 0]);
            sums[6*TM + 1] = mad(ca.s1, cb1.s2, sums[6*TM + 1]);
            sums[6*TM + 2] = mad(ca.s2, cb1.s2, sums[6*TM + 2]);
            sums[6*TM + 3] = mad(ca.s3, cb1.s2, sums[6*TM + 3]);

            sums[7*TM + 0] = mad(ca.s0, cb1.s3, sums[7*TM + 0]);
            sums[7*TM + 1] = mad(ca.s1, cb1.s3, sums[7*TM + 1]);
            sums[7*TM + 2] = mad(ca.s2, cb1.s3, sums[7*TM + 2]);
            sums[7*TM + 3] = mad(ca.s3, cb1.s3, sums[7*TM + 3]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int dr = ir * BM + th_r * TM;
    const int dc = ic * BN + th_c * TN;

    const int offsets = batch_idx * batch_stride_d;

    for (int cc = 0; cc < TN; cc++) {
        for (int cr = 0; cr < TM; cr++) {
            if (dr + cr < ne01 && dc + cc < ne11) {
                dst[offsets + (dc + cc) * stride_d + dr + cr] = sums[cc * TM + cr];
            }
        }
    }
}
