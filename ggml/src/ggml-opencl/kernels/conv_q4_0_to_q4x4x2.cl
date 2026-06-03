// conv_q4_0_to_q4x4x2.cl — LazyVLM design-delta (V3).
//
// Emits a Q4_0 weight (canonical struct-of-arrays as produced by
// kernel_convert_block_q4_0: src0_q = raw Q4_0 quant bytes, 16 bytes/block,
// block-contiguous, NO deshuffle; src0_d = 1 fp16 scale/block) into Hexagon's
// q4x4x2 layout in a companion buffer, one work-item per weight row.
//
// q4x4x2 row layout (mirrors ggml-hexagon.cpp repack_row_q4x4x2):
//   per 256-elem super-block (= 8 Q4_0 blocks): 128 quant bytes where
//     q[j] = (nib[j+128] << 4) | nib[j]   for j in [0,128)
//   followed by 8 fp16 block scales. Row stride = K/2 + (K/256)*16 bytes;
//   quants first, scales at offset K/2.
//
// Byte-identical to the V1/V2 emit (phase0_smoke/emit_q4x4x2_only.cl), which was
// validated against repack_row_q4x4x2 on op15. Used only when GGML_OPENCL_FUSED_REPACK
// is set; the default Adreno noshuffle path never calls it.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Canonical Q4_0 nibble: byte (e&31 within block, low 16 -> low nibble, high 16 ->
// high nibble), block index = e>>5. `base` is the first block index of this row.
inline uchar nib_of(__global const uchar* sq, int base, int e) {
    int bi = e >> 5, wi = e & 31;
    __global const uchar* qb = sq + (long)(base + bi) * 16;
    return (wi < 16) ? (uchar)(qb[wi] & 0x0F) : (uchar)(qb[wi - 16] >> 4);
}

__kernel void kernel_conv_q4_0_to_q4x4x2(
    __global const uchar* src0_q,   // canonical Q4_0 quants, 16 bytes/block
    __global const half*  src0_d,   // 1 fp16 scale/block
    __global uchar*       wq,       // out: q4x4x2 rows
    const int K, const int M)
{
    int m = get_global_id(0);
    if (m >= M) return;

    int nblk = K / 32, nsb = K / 256, row = (K / 2) + nsb * 16;
    __global uchar* yq = wq + (long)m * row;
    __global uchar* yd = yq + (K / 2);

    for (int sb = 0; sb < nsb; ++sb) {
        int base = m * nblk + sb * 8;            // first Q4_0 block of this super-block
        __global uchar* q = yq + sb * 128;
        for (int j = 0; j < 128; ++j) {
            q[j] = (uchar)((nib_of(src0_q, base, j + 128) << 4) | nib_of(src0_q, base, j));
        }
        __global uchar* d = yd + sb * 16;
        for (int bi = 0; bi < 8; ++bi) {
            ushort h = as_ushort(src0_d[base + bi]);
            d[bi * 2 + 0] = (uchar)(h & 0xFF);
            d[bi * 2 + 1] = (uchar)(h >> 8);
        }
    }
}
