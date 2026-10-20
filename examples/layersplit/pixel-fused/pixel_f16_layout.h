struct pixel_layout_options {
    int rows = 0;
    int chunk = 0;
    int unroll = 1;
    int prefetch = 0;
    int fmlal = 0;
};

static pixel_layout_options pixel_cpu_layout;
static pixel_layout_options pixel_gpu_layout;
static std::atomic<uint64_t> pixel_neon_calls{0};
static uint64_t pixel_layout_bytes = 0;
static uint64_t pixel_layout_us = 0;
static bool pixel_fused_residual = false;
static std::atomic<uint64_t> pixel_residual_calls{0};
#include "pixel_packed_cpu.h"

static void pixel_quantize_pair(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * input = dst->src[0];
    GGML_ASSERT(input->type == GGML_TYPE_F32 && ggml_is_contiguous(input));
    GGML_ASSERT(dst->type == GGML_TYPE_Q8_K && input->ne[0] % 256 == 0);
    const auto quantize = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
    const auto * x = static_cast<const float *>(input->data);
    auto * q = static_cast<block_q8_K *>(dst->data);
    const int64_t blocks = ggml_nelements(input) / 256;
    alignas(64) float residual[256];
    for (int64_t block = ith; block < blocks; block += nth) {
        quantize(x + block * 256, q + block, 256);
        dequantize_row_q8_K(q + block, residual, 256);
        for (int i = 0; i < 256; ++i) residual[i] = x[block * 256 + i] - residual[i];
        quantize(residual, q + blocks + block, 256);
    }
}

static void pixel_packed_pair_dot(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * pair = dst->src[1];
    GGML_ASSERT(weights->type == GGML_TYPE_Q4_K || weights->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(pair->type == GGML_TYPE_Q8_K && ggml_is_contiguous(weights));
    const auto dot = ggml_get_type_traits_cpu(weights->type)->vec_dot;
    const int64_t rows = weights->ne[1];
    const int64_t batch = pair->ne[1] / 2;
    const auto * w = static_cast<const uint8_t *>(weights->data);
    const auto * q = static_cast<const uint8_t *>(pair->data);
    auto * output = static_cast<float *>(dst->data);
    if (ith == 0) pixel_residual_calls.fetch_add(1, std::memory_order_relaxed);
    for (int64_t row = rows * ith / nth; row < rows * (ith + 1) / nth; ++row) {
        for (int64_t b = 0; b < batch; ++b) {
            float first = 0, residual = 0;
            const void * weight_row = w + row * weights->nb[1];
            dot(int(weights->ne[0]), &first, 0, weight_row, 0, q + b * pair->nb[1], 0, 1);
            dot(int(weights->ne[0]), &residual, 0, weight_row, 0, q + (b + batch) * pair->nb[1], 0, 1);
            output[b * rows + row] = first + residual;
        }
    }
}

static ggml_tensor * pixel_fused_residual_projection(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * input,
        std::vector<std::unique_ptr<pixel_packed_task>> & tasks) {
    ggml_tensor * quant_args[] = {input};
    ggml_tensor * pair = ggml_custom_4d(ctx, GGML_TYPE_Q8_K, input->ne[0], 2 * input->ne[1], 1, 1,
                                       quant_args, 1, pixel_quantize_pair, GGML_N_TASKS_MAX, nullptr);
    ggml_tensor * dot_args[] = {weights, pair};
    ggml_custom_op_t function = pixel_packed_pair_dot;
    void * userdata = nullptr;
    if (pixel_cpu_row_chunk || pixel_cpu_pair_mode || pixel_cpu_row_profile) {
        tasks.emplace_back(new pixel_packed_task());
        function = pixel_tuned_packed_dot;
        userdata = tasks.back().get();
    }
    return ggml_custom_4d(ctx, GGML_TYPE_F32, weights->ne[1], input->ne[1], 1, 1,
                          dot_args, 2, function, GGML_N_TASKS_MAX, userdata);
}

static void pixel_expand_gpu_weights(host_tensor & tensor) {
    if (tensor.bytes.empty()) return;
    GGML_ASSERT(tensor.type == GGML_TYPE_Q4_K || tensor.type == GGML_TYPE_Q6_K);
    const auto decode = ggml_get_type_traits(tensor.type)->to_float;
    GGML_ASSERT(decode != nullptr);
    const size_t source_row = ggml_row_size(tensor.type, tensor.ne0);
    const size_t target_row = size_t(tensor.ne0) * sizeof(ggml_fp16_t);
    std::vector<uint8_t> expanded(size_t(tensor.ne1) * target_row);
    std::vector<float> row(size_t(tensor.ne0));
    for (int64_t r = 0; r < tensor.ne1; ++r) {
        decode(tensor.bytes.data() + r * source_row, row.data(), tensor.ne0);
        ggml_fp32_to_fp16_row(row.data(), reinterpret_cast<ggml_fp16_t *>(expanded.data() + r * target_row), tensor.ne0);
    }
    tensor.bytes.swap(expanded);
    tensor.type = GGML_TYPE_F16;
}

static int pixel_layout_integer(const char * name, int fallback) {
    const char * value = getenv(name);
    if (!value) return fallback;
    char * end = nullptr;
    const long result = strtol(value, &end, 10);
    GGML_ASSERT(end != value && *end == 0 && result >= 0 && result <= 4096);
    return int(result);
}

static void pixel_pack_f16(host_tensor & tensor, const pixel_layout_options & layout) {
    if (!layout.rows || !layout.chunk || tensor.bytes.empty()) return;
    GGML_ASSERT(tensor.type == GGML_TYPE_F16);
    GGML_ASSERT(tensor.ne0 % layout.chunk == 0 && tensor.ne1 % layout.rows == 0);
    const uint64_t started = now_us();
    std::vector<uint8_t> packed(tensor.bytes.size());
    const size_t chunk_bytes = size_t(layout.chunk) * sizeof(ggml_fp16_t);
    for (int64_t row = 0; row < tensor.ne1; row += layout.rows) {
        for (int64_t k = 0; k < tensor.ne0; k += layout.chunk) {
            for (int r = 0; r < layout.rows; ++r) {
                const size_t source = size_t((row + r) * tensor.ne0 + k) * 2;
                const size_t target = size_t(row * tensor.ne0 + k * layout.rows + r * layout.chunk) * 2;
                memcpy(packed.data() + target, tensor.bytes.data() + source, chunk_bytes);
            }
        }
    }
    tensor.bytes.swap(packed);
    pixel_layout_bytes += tensor.bytes.size();
    pixel_layout_us += now_us() - started;
}

template<int R, int U, bool H>
static void pixel_neon_matvec(ggml_tensor * dst, int ith, int nth, void *) {
#if defined(__aarch64__)
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * input = dst->src[1];
    const int64_t kmax = weights->ne[0];
    const int64_t rows = weights->ne[1];
    const int chunk = pixel_cpu_layout.chunk;
    const int prefetch = pixel_cpu_layout.prefetch;
    GGML_ASSERT(weights->type == GGML_TYPE_F16 && input->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(weights) && ggml_is_contiguous(input));
    GGML_ASSERT(kmax % (8 * U) == 0 && rows % R == 0);
    if (ith == 0) pixel_neon_calls.fetch_add(1, std::memory_order_relaxed);
    const auto * w = static_cast<const float16_t *>(weights->data);
    const auto * x = static_cast<const float *>(input->data);
    auto * y = static_cast<float *>(dst->data);
    const int64_t tiles = rows / R;
    const int64_t begin = tiles * ith / nth;
    const int64_t end = tiles * (ith + 1) / nth;
    for (int64_t batch = 0; batch < input->ne[1]; ++batch) {
        for (int64_t tile = begin; tile < end; ++tile) {
            float32x4_t sums[R][U][2];
            for (int r = 0; r < R; ++r) {
                for (int u = 0; u < U; ++u) {
                    sums[r][u][0] = vdupq_n_f32(0);
                    sums[r][u][1] = vdupq_n_f32(0);
                }
            }
            for (int64_t k = 0; k < kmax; k += 8 * U) {
                for (int u = 0; u < U; ++u) {
                    const int64_t col = k + 8 * u;
                    const float32x4_t lo = vld1q_f32(x + batch * kmax + col);
                    const float32x4_t hi = vld1q_f32(x + batch * kmax + col + 4);
                    for (int r = 0; r < R; ++r) {
                        const int64_t offset = chunk ? tile * R * kmax + (col & ~(int64_t(chunk) - 1)) * R + r * chunk + (col & (chunk - 1)) :
                            (tile * R + r) * kmax + col;
                        const float16_t * address = w + offset;
                        if (prefetch && offset + prefetch * R < rows * kmax) __builtin_prefetch(address + prefetch * R, 0, 3);
                        const float16x8_t value = vld1q_f16(address);
                        if constexpr (H) {
                            const float16x8_t activation = vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
                            sums[r][u][0] = vfmlalq_low_f16(sums[r][u][0], value, activation);
                            sums[r][u][1] = vfmlalq_high_f16(sums[r][u][1], value, activation);
                        } else {
                            sums[r][u][0] = vfmaq_f32(sums[r][u][0], vcvt_f32_f16(vget_low_f16(value)), lo);
                            sums[r][u][1] = vfmaq_f32(sums[r][u][1], vcvt_high_f32_f16(value), hi);
                        }
                    }
                }
            }
            for (int r = 0; r < R; ++r) {
                float32x4_t sum = vaddq_f32(sums[r][0][0], sums[r][0][1]);
                for (int u = 1; u < U; ++u) sum = vaddq_f32(sum, vaddq_f32(sums[r][u][0], sums[r][u][1]));
                y[batch * rows + tile * R + r] = vaddvq_f32(sum);
            }
        }
    }
#else
    (void) dst; (void) ith; (void) nth;
    GGML_ABORT("NEON matvec needs aarch64");
#endif
}

static ggml_tensor * pixel_neon_projection(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * input, bool correction,
        std::vector<std::unique_ptr<pixel_packed_task>> & tasks) {
    if (pixel_fused_residual) {
        GGML_ASSERT(correction && !pixel_cpu_layout.rows);
        return pixel_fused_residual_projection(ctx, weights, input, tasks);
    }
    if (!pixel_cpu_layout.rows) return pixel_residual_matmul(ctx, weights, input, correction);
    ggml_custom_op_t function = nullptr;
#define PIXEL_NEON_SELECT(R, U) if (pixel_cpu_layout.rows == R && pixel_cpu_layout.unroll == U) function = pixel_cpu_layout.fmlal ? pixel_neon_matvec<R, U, true> : pixel_neon_matvec<R, U, false>;
    PIXEL_NEON_SELECT(2, 1)
    PIXEL_NEON_SELECT(2, 2)
    PIXEL_NEON_SELECT(4, 1)
    PIXEL_NEON_SELECT(4, 2)
    PIXEL_NEON_SELECT(8, 1)
#undef PIXEL_NEON_SELECT
    GGML_ASSERT(function != nullptr && !correction);
    ggml_tensor * arguments[] = {weights, input};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, weights->ne[1], input->ne[1], 1, 1,
                          arguments, 2, function, GGML_N_TASKS_MAX, nullptr);
}
