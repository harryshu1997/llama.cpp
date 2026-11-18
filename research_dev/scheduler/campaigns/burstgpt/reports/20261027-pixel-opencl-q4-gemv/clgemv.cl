// Pixel 10 Pro (PowerVR DXT) GEMV microkernels: y[N] = W[N][K] * x[K].
// Build options: -D R=<rows per work-group> -D WG=<work-group size>.  L = WG/R lanes share one row.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#ifndef R
#define R 8
#endif
#ifndef WG
#define WG 128
#endif
#define L (WG / R)

// ---- F16 weights, row-major [N][K]; f32 activations; lanes of one row read contiguous 16 B chunks.
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void f16_rows(__global const half8 * w, __constant const float * x, __global float * y, const int K) {
    const int lid = get_local_id(0), g = get_group_id(0);
    const int r = lid / L, lane = lid % L;
    const int n8 = K / 8;
    __global const half8 * row = w + (size_t)(g * R + r) * n8;
    float acc = 0.f;
    for (int i = lane; i < n8; i += L) {
        float8 f = convert_float8(row[i]);
        float8 xv = vload8(i, x);
        acc += dot(f.lo, xv.lo) + dot(f.hi, xv.hi);
    }
    __local float red[WG];
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) { float t = 0.f; for (int l = 0; l < L; ++l) t += red[r * L + l]; y[g * R + r] = t; }
}

// ---- Q4 tile layout in global memory.
// Weights: for row-group g and block b (32 weights): R consecutive uint4 (16 B each): bytes 0..15 = low nibbles of w[0..15], high nibbles = w[16..31], value = nibble - 8.
// Scales: half, index (g*nb + b)*R + r.  Activations: int8 per block of 32 with f32 scale xs[b]: xq = K/16 uint4.
inline float q4_block(uint4 q, uint4 a0, uint4 a1) {
    uint4 lo = q & (uint4)(0x0F0F0F0Fu), hi = (q >> 4) & (uint4)(0x0F0F0F0Fu);
    const char4 eight = (char4)(8);
    int s = dot_acc_sat(as_char4(lo.x) - eight, as_char4(a0.x), 0);
    s = dot_acc_sat(as_char4(lo.y) - eight, as_char4(a0.y), s);
    s = dot_acc_sat(as_char4(lo.z) - eight, as_char4(a0.z), s);
    s = dot_acc_sat(as_char4(lo.w) - eight, as_char4(a0.w), s);
    s = dot_acc_sat(as_char4(hi.x) - eight, as_char4(a1.x), s);
    s = dot_acc_sat(as_char4(hi.y) - eight, as_char4(a1.y), s);
    s = dot_acc_sat(as_char4(hi.z) - eight, as_char4(a1.z), s);
    s = dot_acc_sat(as_char4(hi.w) - eight, as_char4(a1.w), s);
    return (float) s;
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void q4_global(__global const uint4 * w, __global const half * sc, __constant const uint4 * xq, __constant const float * xs, __global float * y, const int K) {
    const int lid = get_local_id(0), g = get_group_id(0);
    const int r = lid % R, lane = lid / R;          // consecutive lids -> consecutive rows of one block: contiguous 16*R bytes
    const int nb = K / 32;
    float acc = 0.f;
    for (int b = lane; b < nb; b += L) {
        const int t = (g * nb + b) * R + r;
        float s = q4_block(w[t], xq[2 * b], xq[2 * b + 1]);
        acc += vload_half(t, sc) * xs[b] * s;
    }
    __local float red[WG];
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) { float tt = 0.f; for (int l = 0; l < L; ++l) tt += red[l * R + r]; y[g * R + r] = tt; }
}

// ---- Same tile layout, weights bound as a 2D RGBA32UI image (one texel = 16 B), activations in __constant memory.
#ifndef IMGW
#define IMGW 8192
#endif
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void q4_image(__read_only image2d_t w, __global const half * sc, __constant const uint4 * xq, __constant const float * xs, __global float * y, const int K) {
    const int lid = get_local_id(0), g = get_group_id(0);
    const int r = lid % R, lane = lid / R;
    const int nb = K / 32;
    float acc = 0.f;
    for (int b = lane; b < nb; b += L) {
        const int t = (g * nb + b) * R + r;
        uint4 q = read_imageui(w, (int2)(t % IMGW, t / IMGW));
        float s = q4_block(q, xq[2 * b], xq[2 * b + 1]);
        acc += vload_half(t, sc) * xs[b] * s;
    }
    __local float red[WG];
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) { float tt = 0.f; for (int l = 0; l < L; ++l) tt += red[l * R + r]; y[g * R + r] = tt; }
}

// ---- Tile layout with the scale interleaved into the stream: R uint4 of nibbles followed by R halfs (2R bytes) per block -> one contiguous read stream per work-group.
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void q4_global_interleaved(__global const uchar * w, __constant const uint4 * xq, __constant const float * xs, __global float * y, const int K) {
    const int lid = get_local_id(0), g = get_group_id(0);
    const int r = lid % R, lane = lid / R;
    const int nb = K / 32;
    const int stride = 16 * R + 2 * R;               // bytes per (g, b)
    float acc = 0.f;
    for (int b = lane; b < nb; b += L) {
        __global const uchar * base = w + (size_t)(g * nb + b) * stride;
        uint4 q = vload4(0, (__global const uint *)(base + 16 * r));
        float d = vload_half(r, (__global const half *)(base + 16 * R));
        acc += d * xs[b] * q4_block(q, xq[2 * b], xq[2 * b + 1]);
    }
    __local float red[WG];
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) { float tt = 0.f; for (int l = 0; l < L; ++l) tt += red[l * R + r]; y[g * R + r] = tt; }
}

// ---- M activation rows share the weight stream (decode batch M): y[m][N]. xq: [M][K/16] uint4, xs: [M][nb].
#ifndef M
#define M 2
#endif
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void q4_global_m(__global const uint4 * w, __global const half * sc, __constant const uint4 * xq, __constant const float * xs, __global float * y, const int K, const int N) {
    const int lid = get_local_id(0), g = get_group_id(0);
    const int r = lid % R, lane = lid / R;
    const int nb = K / 32;
    float acc[M];
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    for (int b = lane; b < nb; b += L) {
        const int t = (g * nb + b) * R + r;
        uint4 q = w[t];
        float d = vload_half(t, sc);
        uint4 lo = q & (uint4)(0x0F0F0F0Fu), hi = (q >> 4) & (uint4)(0x0F0F0F0Fu);
        const char4 eight = (char4)(8);
        char4 w0 = as_char4(lo.x) - eight, w1 = as_char4(lo.y) - eight, w2 = as_char4(lo.z) - eight, w3 = as_char4(lo.w) - eight;
        char4 w4 = as_char4(hi.x) - eight, w5 = as_char4(hi.y) - eight, w6 = as_char4(hi.z) - eight, w7 = as_char4(hi.w) - eight;
        for (int m = 0; m < M; ++m) {
            uint4 a0 = xq[m * (K / 16) + 2 * b], a1 = xq[m * (K / 16) + 2 * b + 1];
            int s = dot_acc_sat(w0, as_char4(a0.x), 0); s = dot_acc_sat(w1, as_char4(a0.y), s); s = dot_acc_sat(w2, as_char4(a0.z), s); s = dot_acc_sat(w3, as_char4(a0.w), s);
            s = dot_acc_sat(w4, as_char4(a1.x), s); s = dot_acc_sat(w5, as_char4(a1.y), s); s = dot_acc_sat(w6, as_char4(a1.z), s); s = dot_acc_sat(w7, as_char4(a1.w), s);
            acc[m] += d * xs[m * nb + b] * (float) s;
        }
    }
    __local float red[M][WG];
    for (int m = 0; m < M; ++m) red[m][lid] = acc[m];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) for (int m = 0; m < M; ++m) { float tt = 0.f; for (int l = 0; l < L; ++l) tt += red[m][l * R + r]; y[m * N + g * R + r] = tt; }
}

// ---- pure streaming read of the weight buffer (xor reduce): the GPU's raw read ceiling under OpenCL.
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void stream_read(__global const uint4 * w, __global uint * y, const int count) {
    const int gid = get_global_id(0), gs = get_global_size(0);
    uint4 a = (uint4)(0), b = (uint4)(0);
    int i = gid;
    for (; i + gs < count; i += 2 * gs) { a ^= w[i]; b ^= w[i + gs]; }
    if (i < count) a ^= w[i];
    a ^= b; y[gid] = a.x ^ a.y ^ a.z ^ a.w;
}

// ---- q4_global with two blocks per iteration (more loads in flight per work-item).
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void q4_global_u2(__global const uint4 * w, __global const half * sc, __constant const uint4 * xq, __constant const float * xs, __global float * y, const int K) {
    const int lid = get_local_id(0), g = get_group_id(0);
    const int r = lid % R, lane = lid / R;
    const int nb = K / 32;
    float acc = 0.f;
    int b = lane;
    for (; b + L < nb; b += 2 * L) {
        const int t0 = (g * nb + b) * R + r, t1 = (g * nb + b + L) * R + r;
        uint4 q0 = w[t0], q1 = w[t1];
        float d0 = vload_half(t0, sc), d1 = vload_half(t1, sc);
        acc += d0 * xs[b] * q4_block(q0, xq[2 * b], xq[2 * b + 1]);
        acc += d1 * xs[b + L] * q4_block(q1, xq[2 * (b + L)], xq[2 * (b + L) + 1]);
    }
    if (b < nb) { const int t = (g * nb + b) * R + r; acc += vload_half(t, sc) * xs[b] * q4_block(w[t], xq[2 * b], xq[2 * b + 1]); }
    __local float red[WG];
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) { float tt = 0.f; for (int l = 0; l < L; ++l) tt += red[l * R + r]; y[g * R + r] = tt; }
}

// ---- v2: unsigned nibbles with the -8 folded into a per-block activation sum (8 fewer ops per block), scales in the stream as
// 16 B-aligned chunks after the R nibble rows, loads for the next block issued before the current block is computed.
// Stream per (g,b): R*16 B nibbles + SCB bytes of scales (SCB = R*2 rounded up to 16).  xsum[b] = sum of the 32 int8 activations.
#ifndef SCB
#define SCB (((R * 2) + 15) / 16 * 16)
#endif
#define STRIDE_V2 (R * 16 + SCB)
inline int q4_dot_u(uint4 q, uint4 a0, uint4 a1) {
    uint4 lo = q & (uint4)(0x0F0F0F0Fu), hi = (q >> 4) & (uint4)(0x0F0F0F0Fu);
    int s = dot_acc_sat(as_uchar4(lo.x), as_char4(a0.x), 0);
    s = dot_acc_sat(as_uchar4(lo.y), as_char4(a0.y), s);
    s = dot_acc_sat(as_uchar4(lo.z), as_char4(a0.z), s);
    s = dot_acc_sat(as_uchar4(lo.w), as_char4(a0.w), s);
    s = dot_acc_sat(as_uchar4(hi.x), as_char4(a1.x), s);
    s = dot_acc_sat(as_uchar4(hi.y), as_char4(a1.y), s);
    s = dot_acc_sat(as_uchar4(hi.z), as_char4(a1.z), s);
    s = dot_acc_sat(as_uchar4(hi.w), as_char4(a1.w), s);
    return s;
}
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void q4_v2(__global const uchar * w, __constant const uint4 * xq, __constant const float * xs, __constant const int * xsum, __global float * y, const int K) {
    const int lid = get_local_id(0), g = get_group_id(0);
    const int r = lid % R, lane = lid / R;
    const int nb = K / 32;
    float acc = 0.f;
    int b = lane;
    __global const uchar * base = w + ((size_t) g * nb + b) * STRIDE_V2;
    uint4 q = (b < nb) ? vload4(0, (__global const uint *)(base + 16 * r)) : (uint4)(0);
    half d = (b < nb) ? *((__global const half *)(base + R * 16) + r) : (half) 0;
    for (; b < nb; b += L) {
        const int bn = b + L;
        __global const uchar * basen = base + (size_t) L * STRIDE_V2;
        uint4 qn = (bn < nb) ? vload4(0, (__global const uint *)(basen + 16 * r)) : (uint4)(0);
        half dn = (bn < nb) ? *((__global const half *)(basen + R * 16) + r) : (half) 0;
        int s = q4_dot_u(q, xq[2 * b], xq[2 * b + 1]) - 8 * xsum[b];
        acc += (float) d * xs[b] * (float) s;
        q = qn; d = dn; base = basen;
    }
    __local float red[WG];
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) { float tt = 0.f; for (int l = 0; l < L; ++l) tt += red[l * R + r]; y[g * R + r] = tt; }
}

// ---- async_work_group_copy (bulk copies into local memory, double-buffered) as the GPU's "DMA-like" read path.
#ifndef CHUNK
#define CHUNK 1024   // uint4 per chunk = 16 KB; two buffers = 32 KB of the 256 KB local memory
#endif
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void stream_async(__global const uint4 * w, __global uint * y, const int count) {
    __local uint4 buf[2][CHUNK];
    const int lid = get_local_id(0), g = get_group_id(0), ng = get_num_groups(0);
    const int nchunks = count / CHUNK;
    uint4 acc = (uint4)(0);
    int c = g; int cur = 0;
    event_t ev = async_work_group_copy(buf[0], w + (size_t) c * CHUNK, CHUNK, 0);
    for (; c < nchunks; c += ng) {
        const int cn = c + ng;
        const int cs = (cn < nchunks) ? cn : c;   // event_t cannot be conditional: always issue the next copy (a repeat of the current chunk at the tail)
        event_t evn = async_work_group_copy(buf[cur ^ 1], w + (size_t) cs * CHUNK, CHUNK, 0);
        wait_group_events(1, &ev);
        for (int i = lid; i < CHUNK; i += WG) acc ^= buf[cur][i];
        barrier(CLK_LOCAL_MEM_FENCE);
        ev = evn; cur ^= 1;
    }
    wait_group_events(1, &ev);
    y[get_global_id(0)] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}

// ---- instruction-form variants of the Q4 block dot (same tile layout and arguments as q4_global).
// packed form: dot_acc_sat_4x8packed_ss_int(uint, uint, int)
inline float q4_block_packed(uint4 q, uint4 a0, uint4 a1) {
    uint4 lo = q & (uint4)(0x0F0F0F0Fu), hi = (q >> 4) & (uint4)(0x0F0F0F0Fu);
    const uint eight = 0x08080808u;   // subtract 8 per byte: bytes are 0..15 so no borrow across lanes
    int s = dot_acc_sat_4x8packed_ss_int(lo.x - eight, a0.x, 0);
    s = dot_acc_sat_4x8packed_ss_int(lo.y - eight, a0.y, s);
    s = dot_acc_sat_4x8packed_ss_int(lo.z - eight, a0.z, s);
    s = dot_acc_sat_4x8packed_ss_int(lo.w - eight, a0.w, s);
    s = dot_acc_sat_4x8packed_ss_int(hi.x - eight, a1.x, s);
    s = dot_acc_sat_4x8packed_ss_int(hi.y - eight, a1.y, s);
    s = dot_acc_sat_4x8packed_ss_int(hi.z - eight, a1.z, s);
    s = dot_acc_sat_4x8packed_ss_int(hi.w - eight, a1.w, s);
    return (float) s;
}
// no-dot baseline: unpack to short/float and multiply-add (what a driver without dot acceleration would do)
inline float q4_block_nodot(uint4 q, uint4 a0, uint4 a1) {
    uint4 lo = q & (uint4)(0x0F0F0F0Fu), hi = (q >> 4) & (uint4)(0x0F0F0F0Fu);
    char16 wl = as_char16(lo) - (char16)(8), wh = as_char16(hi) - (char16)(8);
    short16 pl = convert_short16(wl) * convert_short16(as_char16(a0)), ph = convert_short16(wh) * convert_short16(as_char16(a1));
    short16 p = pl + ph;
    short8 p8 = p.lo + p.hi; short4 p4 = p8.lo + p8.hi;
    return (float)(p4.x + p4.y + p4.z + p4.w);
}
// fp16 math: weights unpacked to half and dotted with half activations (activations passed as the int8 vector, converted)
inline float q4_block_half(uint4 q, uint4 a0, uint4 a1) {
    uint4 lo = q & (uint4)(0x0F0F0F0Fu), hi = (q >> 4) & (uint4)(0x0F0F0F0Fu);
    half16 wl = convert_half16(as_char16(lo) - (char16)(8)), wh = convert_half16(as_char16(hi) - (char16)(8));
    half16 al = convert_half16(as_char16(a0)), ah = convert_half16(as_char16(a1));
    half16 p = wl * al + wh * ah;
    half8 p8 = p.lo + p.hi; half4 p4 = p8.lo + p8.hi;
    return (float)(p4.x + p4.y) + (float)(p4.z + p4.w);
}
#define Q4_KERNEL(NAME, BLOCK) \
__kernel __attribute__((reqd_work_group_size(WG, 1, 1))) \
void NAME(__global const uint4 * w, __global const half * sc, __constant const uint4 * xq, __constant const float * xs, __global float * y, const int K) { \
    const int lid = get_local_id(0), g = get_group_id(0); const int r = lid % R, lane = lid / R; const int nb = K / 32; float acc = 0.f; \
    for (int b = lane; b < nb; b += L) { const int t = (g * nb + b) * R + r; acc += vload_half(t, sc) * xs[b] * BLOCK(w[t], xq[2 * b], xq[2 * b + 1]); } \
    __local float red[WG]; red[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE); \
    if (lane == 0) { float tt = 0.f; for (int l = 0; l < L; ++l) tt += red[l * R + r]; y[g * R + r] = tt; } }
Q4_KERNEL(q4_dotpacked, q4_block_packed)
Q4_KERNEL(q4_nodot, q4_block_nodot)
Q4_KERNEL(q4_half, q4_block_half)

// sub-group reduction instead of the local-memory loop: lanes of one row must be consecutive lids (mapping r = lid / L).
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void q4_subgroup(__global const uint4 * w, __global const half * sc, __constant const uint4 * xq, __constant const float * xs, __global float * y, const int K) {
    const int lid = get_local_id(0), g = get_group_id(0);
    const int r = lid / L, lane = lid % L;
    const int nb = K / 32;
    float acc = 0.f;
    for (int b = lane; b < nb; b += L) { const int t = (g * nb + b) * R + r; acc += vload_half(t, sc) * xs[b] * q4_block(w[t], xq[2 * b], xq[2 * b + 1]); }
    float tt = sub_group_reduce_add(acc);            // valid when L == sub-group size (set WG = R * subgroup)
    if (lane == 0) y[g * R + r] = tt;
}

// ---- q4_uu: the only hardware-accelerated 8-bit dot on this driver is unsigned x unsigned (packed), so both operands are made
// unsigned: weights stay 0..15 nibbles, activations are shifted by +128; the corrections use a precomputed per-(row,block) nibble
// sum (ushort) stored in the stream next to the scale, and a per-block activation sum.  Stream per (g,b): R*16 B nibbles, R halfs, R ushorts.
#define STRIDE_UU (R * 20)
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void q4_uu(__global const uchar * w, __constant const uint4 * xqu, __constant const float * xs, __constant const int * asum, __global float * y, const int K) {
    const int lid = get_local_id(0), g = get_group_id(0);
    const int r = lid % R, lane = lid / R;
    const int nb = K / 32;
    float acc = 0.f;
    for (int b = lane; b < nb; b += L) {
        __global const uchar * base = w + ((size_t) g * nb + b) * STRIDE_UU;
        uint4 q = vload4(0, (__global const uint *)(base + 16 * r));
        float d = vload_half(r, (__global const half *)(base + 16 * R));
        int wsum = *((__global const ushort *)(base + 18 * R) + r);
        uint4 a0 = xqu[2 * b], a1 = xqu[2 * b + 1];
        uint4 lo = q & (uint4)(0x0F0F0F0Fu), hi = (q >> 4) & (uint4)(0x0F0F0F0Fu);
        uint s = dot_acc_sat_4x8packed_uu_uint(lo.x, a0.x, 0u);
        s = dot_acc_sat_4x8packed_uu_uint(lo.y, a0.y, s); s = dot_acc_sat_4x8packed_uu_uint(lo.z, a0.z, s); s = dot_acc_sat_4x8packed_uu_uint(lo.w, a0.w, s);
        s = dot_acc_sat_4x8packed_uu_uint(hi.x, a1.x, s); s = dot_acc_sat_4x8packed_uu_uint(hi.y, a1.y, s); s = dot_acc_sat_4x8packed_uu_uint(hi.z, a1.z, s); s = dot_acc_sat_4x8packed_uu_uint(hi.w, a1.w, s);
        acc += d * xs[b] * (float)((int) s - 128 * wsum - 8 * asum[b]);
    }
    __local float red[WG];
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) { float tt = 0.f; for (int l = 0; l < L; ++l) tt += red[l * R + r]; y[g * R + r] = tt; }
}

// ---- Q8_0: 32 int8 weights + f16 scale. Stream per (g,b): R*32 B then R halfs (padded to 16 B).
#define SCB8 (((R * 2) + 15) / 16 * 16)
#define STRIDE_Q8 (R * 32 + SCB8)
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void q8_tile(__global const uchar * w, __constant const uint4 * xq, __constant const float * xs, __global float * y, const int K) {
    const int lid = get_local_id(0), g = get_group_id(0);
    const int r = lid % R, lane = lid / R;
    const int nb = K / 32;
    float acc = 0.f;
    for (int b = lane; b < nb; b += L) {
        __global const uchar * base = w + ((size_t) g * nb + b) * STRIDE_Q8;
        uint4 q0 = vload4(0, (__global const uint *)(base + 32 * r)), q1 = vload4(1, (__global const uint *)(base + 32 * r));
        float d = vload_half(r, (__global const half *)(base + 32 * R));
        uint4 a0 = xq[2 * b], a1 = xq[2 * b + 1];
        int s = dot_acc_sat(as_char4(q0.x), as_char4(a0.x), 0); s = dot_acc_sat(as_char4(q0.y), as_char4(a0.y), s); s = dot_acc_sat(as_char4(q0.z), as_char4(a0.z), s); s = dot_acc_sat(as_char4(q0.w), as_char4(a0.w), s);
        s = dot_acc_sat(as_char4(q1.x), as_char4(a1.x), s); s = dot_acc_sat(as_char4(q1.y), as_char4(a1.y), s); s = dot_acc_sat(as_char4(q1.z), as_char4(a1.z), s); s = dot_acc_sat(as_char4(q1.w), as_char4(a1.w), s);
        acc += d * xs[b] * (float) s;
    }
    __local float red[WG];
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) { float tt = 0.f; for (int l = 0; l < L; ++l) tt += red[l * R + r]; y[g * R + r] = tt; }
}

// ---- Q4_K: super-block of 256 = 8 sub-blocks of 32; w = d*sc_j*q - dmin*m_j (q 0..15, sc/m 6-bit, ggml packing).
// Stream per (g, sb): R*128 B nibbles (chunk i of 32 B: sub-block 2i in low nibbles, 2i+1 in high), then R*16 B meta (d, dmin halfs, 12 B scales).
#define STRIDE_Q4K (R * 144)
inline int dot_u(uint4 q, uint4 a0, uint4 unused) {   // nibbles in q (unsigned 0..15) . 32 int8 activations
    int s = dot_acc_sat(as_uchar4(q.x), as_char4(a0.x), 0); s = dot_acc_sat(as_uchar4(q.y), as_char4(a0.y), s); s = dot_acc_sat(as_uchar4(q.z), as_char4(a0.z), s); s = dot_acc_sat(as_uchar4(q.w), as_char4(a0.w), s);
    return s;
}
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void q4k_tile(__global const uchar * w, __constant const uint4 * xq, __constant const float * xs, __constant const int * asum, __global float * y, const int K) {
    const int lid = get_local_id(0), g = get_group_id(0);
    const int r = lid % R, lane = lid / R;
    const int nsb = K / 256;
    float acc = 0.f;
    for (int sb = lane; sb < nsb; sb += L) {
        __global const uchar * base = w + ((size_t) g * nsb + sb) * STRIDE_Q4K;
        __global const uint * qs = (__global const uint *)(base + 128 * r);
        __global const uchar * meta = base + 128 * R + 16 * r;
        const float d = vload_half(0, (__global const half *) meta), dmin = vload_half(1, (__global const half *) meta);
        uchar sc[12]; for (int i = 0; i < 12; ++i) sc[i] = meta[4 + i];
        float sum_qs = 0.f, sum_m = 0.f;
        for (int i = 0; i < 4; ++i) {
            uint4 c0 = vload4(2 * i, qs), c1 = vload4(2 * i + 1, qs);          // 32 bytes of chunk i
            uint4 lo0 = c0 & (uint4)(0x0F0F0F0Fu), lo1 = c1 & (uint4)(0x0F0F0F0Fu);
            uint4 hi0 = (c0 >> 4) & (uint4)(0x0F0F0F0Fu), hi1 = (c1 >> 4) & (uint4)(0x0F0F0F0Fu);
            const int j0 = 2 * i, j1 = 2 * i + 1, b0 = sb * 8 + j0, b1 = b0 + 1;
            int s0 = dot_u(lo0, xq[2 * b0], (uint4)(0)) + dot_u(lo1, xq[2 * b0 + 1], (uint4)(0));
            int s1 = dot_u(hi0, xq[2 * b1], (uint4)(0)) + dot_u(hi1, xq[2 * b1 + 1], (uint4)(0));
            int scd0, scm0, scd1, scm1;
            if (j0 < 4) { scd0 = sc[j0] & 63; scm0 = sc[j0 + 4] & 63; scd1 = sc[j1] & 63; scm1 = sc[j1 + 4] & 63; }
            else { scd0 = (sc[j0 + 4] & 0xF) | ((sc[j0 - 4] >> 6) << 4); scm0 = (sc[j0 + 4] >> 4) | ((sc[j0] >> 6) << 4); scd1 = (sc[j1 + 4] & 0xF) | ((sc[j1 - 4] >> 6) << 4); scm1 = (sc[j1 + 4] >> 4) | ((sc[j1] >> 6) << 4); }
            sum_qs += (float)(scd0 * s0) * xs[b0] + (float)(scd1 * s1) * xs[b1];
            sum_m += (float)(scm0 * asum[b0]) * xs[b0] + (float)(scm1 * asum[b1]) * xs[b1];
        }
        acc += d * sum_qs - dmin * sum_m;
    }
    __local float red[WG];
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) { float tt = 0.f; for (int l = 0; l < L; ++l) tt += red[l * R + r]; y[g * R + r] = tt; }
}

// ---- Q4_K v2: the 128 B of nibbles per (row, super-block) are stored as eight 16 B pieces interleaved across the R rows
// (piece i of row r at (i*R + r)*16), so every load instruction of a work-group is contiguous; meta read as one uint4.
inline float half_from_bits(uint h) { return vload_half(0, (__private const half *) &h); }
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void q4k_tile2(__global const uchar * w, __constant const uint4 * xq, __constant const float * xs, __constant const int * asum, __global float * y, const int K) {
    const int lid = get_local_id(0), g = get_group_id(0);
    const int r = lid % R, lane = lid / R;
    const int nsb = K / 256;
    float acc = 0.f;
    for (int sb = lane; sb < nsb; sb += L) {
        __global const uchar * base = w + ((size_t) g * nsb + sb) * STRIDE_Q4K;
        __global const uint4 * pieces = (__global const uint4 *) base;
        const uint4 meta = *((__global const uint4 *)(base + 128 * R) + r);
        const ushort dh = (ushort)(meta.x & 0xFFFFu), dmh = (ushort)(meta.x >> 16);
        const float d = vload_half(0, (__private const half *) &dh), dmin = vload_half(0, (__private const half *) &dmh);
        // 12 scale bytes = meta.y, meta.z, meta.w
        const uint s0 = meta.y, s1 = meta.z, s2 = meta.w;
        float sum_qs = 0.f, sum_m = 0.f;
        for (int i = 0; i < 4; ++i) {
            uint4 c0 = pieces[(2 * i) * R + r], c1 = pieces[(2 * i + 1) * R + r];
            uint4 lo0 = c0 & (uint4)(0x0F0F0F0Fu), lo1 = c1 & (uint4)(0x0F0F0F0Fu);
            uint4 hi0 = (c0 >> 4) & (uint4)(0x0F0F0F0Fu), hi1 = (c1 >> 4) & (uint4)(0x0F0F0F0Fu);
            const int j0 = 2 * i, j1 = 2 * i + 1, b0 = sb * 8 + j0, b1 = b0 + 1;
            int q0 = dot_u(lo0, xq[2 * b0], (uint4)(0)) + dot_u(lo1, xq[2 * b0 + 1], (uint4)(0));
            int q1 = dot_u(hi0, xq[2 * b1], (uint4)(0)) + dot_u(hi1, xq[2 * b1 + 1], (uint4)(0));
            int scd0, scm0, scd1, scm1;
            if (i < 2) {   // j0, j1 < 4: sc[j] & 63, sc[j+4] & 63 ; sc[0..3] in s0 bytes, sc[4..7] in s1 bytes
                scd0 = (s0 >> (8 * j0)) & 63; scm0 = (s1 >> (8 * j0)) & 63; scd1 = (s0 >> (8 * j1)) & 63; scm1 = (s1 >> (8 * j1)) & 63;
            } else {       // j >= 4: d = (sc[j+4] & 0xF) | ((sc[j-4] >> 6) << 4) ; m = (sc[j+4] >> 4) | ((sc[j] >> 6) << 4) ; sc[8..11] in s2
                const int k0 = j0 - 4, k1 = j1 - 4;
                scd0 = ((s2 >> (8 * k0)) & 0xF) | ((((s0 >> (8 * k0)) & 0xFF) >> 6) << 4); scm0 = (((s2 >> (8 * k0)) & 0xFF) >> 4) | ((((s1 >> (8 * k0)) & 0xFF) >> 6) << 4);
                scd1 = ((s2 >> (8 * k1)) & 0xF) | ((((s0 >> (8 * k1)) & 0xFF) >> 6) << 4); scm1 = (((s2 >> (8 * k1)) & 0xFF) >> 4) | ((((s1 >> (8 * k1)) & 0xFF) >> 6) << 4);
            }
            sum_qs += (float)(scd0 * q0) * xs[b0] + (float)(scd1 * q1) * xs[b1];
            sum_m += (float)(scm0 * asum[b0]) * xs[b0] + (float)(scm1 * asum[b1]) * xs[b1];
        }
        acc += d * sum_qs - dmin * sum_m;
    }
    __local float red[WG];
    red[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane == 0) { float tt = 0.f; for (int l = 0; l < L; ++l) tt += red[l * R + r]; y[g * R + r] = tt; }
}

// ---- wider loads / more streams per work-item, to test whether the read ceiling is the GPU's own request rate.
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void stream_read16(__global const uint16 * w, __global uint * y, const int count4) {   // count4 = number of uint4 -> uint16 count = count4/4
    const int gid = get_global_id(0), gs = get_global_size(0); const int count = count4 / 4;
    uint16 a = (uint16)(0), b = (uint16)(0), c = (uint16)(0), d = (uint16)(0);
    int i = gid;
    for (; i + 3 * gs < count; i += 4 * gs) { a ^= w[i]; b ^= w[i + gs]; c ^= w[i + 2 * gs]; d ^= w[i + 3 * gs]; }
    for (; i < count; i += gs) a ^= w[i];
    a ^= b ^ c ^ d; uint8 e = a.lo ^ a.hi; uint4 f = e.lo ^ e.hi; y[gid] = f.x ^ f.y ^ f.z ^ f.w;
}
