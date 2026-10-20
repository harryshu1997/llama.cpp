#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "ffn-fused-ops.h"
#include "matmul-ops.h"
#include "act-ops.h"
#include "hmx-queue.h"
#include "hvx-utils.h"

struct ffn_glu_job {
    struct htp_context * ctx;
    const struct htp_ffn_fused_params * p;
    float * dst;
    const void * gate;
    const void * up;
    uint32_t rows, cols, stride;
};

static void ffn_glu_rows(unsigned nth, unsigned ith, void * data) {
    const struct ffn_glu_job * job = data;
    const struct htp_ffn_fused_params * p = job->p;
    uint8_t * base = job->ctx->vtcm_base;
    float * gate = (float *) (base + p->glu_tmp) + ith * 6 * p->n_chunk;
    float * up = gate + 2 * p->n_chunk;
    float * out = up + 2 * p->n_chunk;
    struct htp_thread_trace * trace = &job->ctx->trace[ith];
    htp_trace_event_start(trace, HTP_TRACE_EVT_HVX_COMP, ith);
    for (uint32_t row = 2 * ith; row < job->rows; row += 2 * nth) {
        const uint32_t rows = MIN(2, job->rows - row);
        htp_mm_f16_rows(gate, job->gate, row, rows, job->cols);
        htp_mm_f16_rows(up, job->up, row, rows, job->cols);
        htp_glu_f32((uint8_t *) out, (const uint8_t *) gate, (const uint8_t *) up,
                    rows * job->cols, p->glu);
        for (uint32_t r = 0; r < rows; ++r) {
            hvx_copy_uu((uint8_t *) (job->dst + (row + r) * job->stride),
                        (const uint8_t *) (out + r * job->cols), job->cols, sizeof(float));
        }
    }
    htp_trace_event_stop(trace, HTP_TRACE_EVT_HVX_COMP, ith);
}

static bool tensor_2d(const struct htp_tensor * t, uint32_t type, uint32_t cols, uint32_t rows) {
    if (!t || t->type != type || t->ne[0] != cols || t->ne[1] != rows ||
        t->ne[2] != 1 || t->ne[3] != 1 || !t->data || t->data % 128) {
        return false;
    }
    const uint32_t bytes = type == HTP_TYPE_F16 ? 2 : 4;
    return t->nb[0] == bytes && t->nb[1] >= (uint64_t) cols * bytes && t->nb[1] % 128 == 0 &&
           (uint64_t) (rows - 1) * t->nb[1] + (uint64_t) cols * bytes <= t->size;
}

static void prefetch_projection(struct htp_context * ctx, const struct htp_ffn_fused_params * p,
                                const struct htp_tensor * const weights[2], uint32_t index, uint32_t n, uint32_t k) {
    const uint32_t col = (index / 2) * p->n_chunk;
    const struct htp_tensor * weight = weights[index % 2];
    dma_queue_push(ctx->dma[0], dma_make_ptr(ctx->vtcm_base + p->raw,
                    (const void *) (weight->data + col * weight->nb[1])),
                    k * 2, weight->nb[1], k * 2, MIN(n - col, p->n_chunk));
}

static void finish_glu(struct htp_context * ctx, const struct htp_ffn_fused_params * p,
                        const struct htp_tensor * dst, uint32_t index, uint32_t row, uint32_t rows, uint32_t n) {
    const uint32_t col = (index / 2) * p->n_chunk;
    struct ffn_glu_job job = { ctx, p, (float *) (dst->data + row * dst->nb[1]) + col,
        ctx->vtcm_base + p->projections[index % 3], ctx->vtcm_base + p->projections[(index + 1) % 3],
        rows, MIN(n - col, p->n_chunk), dst->nb[1] / 4 };
    worker_pool_run_func(ctx->worker_pool, ffn_glu_rows, &job, p->n_threads);
}

int op_ffn_fused(struct htp_ops_context * octx) {
    struct htp_context * ctx = octx->ctx;
    const struct htp_ffn_fused_params * p = (const void *) octx->kernel_params;
    const struct htp_tensor * gate = octx->src[0];
    const struct htp_tensor * input = octx->src[1];
    const struct htp_tensor * up = octx->src[2];
    const struct htp_tensor * dst = octx->dst;
    if (!gate || !input || !up || !dst || !ctx->hmx_enabled || !ctx->hmx_queue) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    const uint32_t k = input->ne[0], m = input->ne[1], n = gate->ne[1];
    if (!k || !n || k % 32 || n % 32 || m <= HTP_MM_HMX_MIN_NROWS || m > INT_MAX - 31 ||
        !tensor_2d(gate, HTP_TYPE_F16, k, n) || !tensor_2d(up, HTP_TYPE_F16, k, n) ||
        !tensor_2d(input, HTP_TYPE_F32, k, m) || !tensor_2d(dst, HTP_TYPE_F32, n, m) ||
        (p->glu != HTP_OP_GLU_GEGLU && p->glu != HTP_OP_GLU_SWIGLU) ||
        p->n_threads > octx->n_threads || p->m_chunk > hex_round_up(m, 32) || p->n_chunk > n) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    struct htp_ffn_fused_params expected = {0};
    expected.glu = p->glu;
    if (!htp_ffn_fused_layout(k, p->m_chunk, p->n_chunk, p->n_threads, &expected) ||
        memcmp(p, &expected, sizeof(expected))) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (p->vtcm_size > ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }
    const struct htp_tensor * weights[] = {gate, up};
    const uint32_t count = 2 * hmx_ceil_div(n, p->n_chunk);
    hmx_matmul_job_t jobs[2];
    for (uint32_t row = 0; row < m; row += p->m_chunk) {
        const uint32_t rows = MIN(m - row, p->m_chunk);
        htp_mm_f16_input(ctx, p, (const float *) (input->data + row * input->nb[1]), rows, k, input->nb[1] / 4);
        prefetch_projection(ctx, p, weights, 0, n, k);
        for (uint32_t index = 0; index < count; ++index) {
            const uint32_t cols = MIN(n - (index / 2) * p->n_chunk, p->n_chunk);
            dma_queue_pop(ctx->dma[0]);
            htp_mm_f16_weights(ctx, p, index % 2, cols, k);
            if (index + 1 < count) {
                prefetch_projection(ctx, p, weights, index + 1, n, k);
            }
            if (!htp_mm_f16_submit(ctx, p, &jobs[index % 2], index, rows, cols, k)) {
                if (index) {
                    hmx_queue_pop(ctx->hmx_queue);
                }
                dma_queue_flush(ctx->dma[0]);
                hmx_queue_suspend(ctx->hmx_queue);
                return HTP_STATUS_INTERNAL_ERR;
            }
            if (index) {
                hmx_queue_pop(ctx->hmx_queue);
                if (index % 2 == 0) {
                    // Three output tiles retain this pair while HMX computes the next gate.
                    finish_glu(ctx, p, dst, index - 2, row, rows, n);
                }
            }
        }
        hmx_queue_pop(ctx->hmx_queue);
        finish_glu(ctx, p, dst, count - 2, row, rows, n);
    }
    hmx_queue_suspend(ctx->hmx_queue);
    return HTP_STATUS_OK;
}
