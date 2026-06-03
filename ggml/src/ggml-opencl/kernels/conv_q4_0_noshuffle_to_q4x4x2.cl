// conv_q4_0_noshuffle_to_q4x4x2.cl — LazyVLM design-delta V3.1 (δ-fast).
//
// Emits a Q4_0 weight into Hexagon's q4x4x2 layout, reading directly from the
// Adreno "noshuffle" weight SoA — the EXACT layout consumed by
// kernel_gemm_noshuffle_q4_0_f32 (produced by kernel_convert_block_q4_0_noshuffle
// followed by transpose_2d_as_16b(q, K/4, M) / (d, K/32, M)). This keeps the fast
// vendor GEMM as the prefill base; the emit just runs after it.
//
// Layout (derived from the GEMM's own reads, gemm_noshuffle_q4_0_f32.cl:51-96):
//   for logical weight row m in [0,M) and K element kk in [0,K):
//     canonical nibble = (src0_q[(kk/4)*M + m] >> (4*(kk&3))) & 0xF   (ushort buffer)
//     per-32-block scale = src0_d[(kk/32)*M + m]                       (half buffer)
//   i.e. column-major (M-major), 4 K-elements packed per ushort in nibble order.
//
// q4x4x2 row (mirrors ggml-hexagon repack_row_q4x4x2, byte-identical to the
// canonical conv_q4_0_to_q4x4x2): per 256-elem super-block -> 128 quant bytes
//   q[j] = (nib[j+128] << 4) | nib[j], j in [0,128) -> then 8 fp16 block scales.
//
// Efficiency: q[g*4+p] pairs element (sb*256 + g*4 + p) with (sb*256 + 128 + g*4 + p),
// which live in ushorts (sb*64 + g) and (sb*64 + 32 + g) at the SAME nibble slot p.
// So each super-block reads exactly 64 source ushorts ONCE (32 low-half + 32
// high-half), extracts 4 nibble-pairs per ushort-pair, and writes a uchar4 — the
// minimum traffic (~K/4 ushorts read + K/2 bytes written per row, no re-reads).
// One work-item per weight row; consecutive rows read consecutive addresses
// (the source is M-major), so the wavefront is coalesced.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// One work-item per weight row m in [0,M); consecutive rows read consecutive
// addresses (source is M-major) so the wavefront is coalesced.
__kernel void kernel_conv_q4_0_noshuffle_to_q4x4x2(
    __global const ushort* src0_q,   // noshuffle + transposed quants
    __global const half*   src0_d,   // noshuffle + transposed scales
    __global uchar*        wq,       // out: q4x4x2 rows
    const int K, const int M)
{
    int m = get_global_id(0);
    if (m >= M) return;

    int nsb = K / 256, row = (K / 2) + nsb * 16;
    __global uchar* yq = wq + (long)m * row;

    for (int sb = 0; sb < nsb; ++sb) {
        long ubase = (long)(sb * 64) * M + m;   // ushort idx of group 0, low half
        __global uchar* q = yq + sb * 128;
        for (int g = 0; g < 32; ++g) {
            ushort ulo = src0_q[ubase + (long)g        * M];  // elems sb*256 + g*4 + {0..3}
            ushort uhi = src0_q[ubase + (long)(g + 32) * M];  // elems sb*256 + 128 + g*4 + {0..3}
            uchar4 out;
            out.s0 = (uchar)(((((uint)uhi >>  0) & 0xF) << 4) | (((uint)ulo >>  0) & 0xF));
            out.s1 = (uchar)(((((uint)uhi >>  4) & 0xF) << 4) | (((uint)ulo >>  4) & 0xF));
            out.s2 = (uchar)(((((uint)uhi >>  8) & 0xF) << 4) | (((uint)ulo >>  8) & 0xF));
            out.s3 = (uchar)(((((uint)uhi >> 12) & 0xF) << 4) | (((uint)ulo >> 12) & 0xF));
            vstore4(out, 0, q + g * 4);
        }
        __global uchar* d = yq + (K / 2) + sb * 16;
        for (int bi = 0; bi < 8; ++bi) {
            ushort h = as_ushort(src0_d[(long)(sb * 8 + bi) * M + m]);
            d[bi * 2 + 0] = (uchar)(h & 0xFF);
            d[bi * 2 + 1] = (uchar)(h >> 8);
        }
    }
}
