#ifndef HTP_FFN_FUSED_OPS_H
#define HTP_FFN_FUSED_OPS_H

#include <stdbool.h>
#include <stdint.h>
#include <limits.h>

struct htp_ffn_fused_params {
    uint32_t m_chunk, n_chunk, n_threads, glu;
    uint32_t raw, weights[2], input, input_tmp, projections[3], glu_tmp, scales;
    uint32_t vtcm_size;
};

// Shared by host admission and DSP validation. Every HMX region is tile-aligned.
static inline bool htp_ffn_fused_layout(uint32_t k, uint32_t mc, uint32_t nc, uint32_t threads,
                                      struct htp_ffn_fused_params * p) {
    if (!k || !mc || !nc || k > INT_MAX || mc > INT_MAX || nc > INT_MAX ||
        k % 32 || mc % 32 || nc % 32 || !threads || threads > 10) {
        return false;
    }
    const uint64_t weights = (uint64_t) k * nc * 2;
    const uint64_t projection = (uint64_t) mc * nc * 2;
    const uint64_t sizes[] = { weights, weights, weights, (uint64_t) mc * k * 2,
        (uint64_t) threads * 4 * k * 4, projection, projection, projection,
        (uint64_t) threads * 3 * 2 * nc * 4, 256 };
    uint32_t * offsets[] = { &p->raw, &p->weights[0], &p->weights[1], &p->input,
        &p->input_tmp, &p->projections[0], &p->projections[1], &p->projections[2], &p->glu_tmp, &p->scales };
    uint64_t total = 0;
    for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        const uint64_t bytes = (sizes[i] + 2047) / 2048 * 2048;
        if (bytes > INT_MAX - total) {
            return false;
        }
        *offsets[i] = (uint32_t) total;
        total += bytes;
    }
    p->m_chunk = mc;
    p->n_chunk = nc;
    p->n_threads = threads;
    p->vtcm_size = (uint32_t) total;
    return true;
}

struct htp_ops_context;
int op_ffn_fused(struct htp_ops_context * octx);

#endif
