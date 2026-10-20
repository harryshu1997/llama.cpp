static int pixel_cpu_row_chunk = 0;
static int pixel_cpu_pair_mode = 0;
static bool pixel_cpu_row_profile = false;
static std::atomic<uint64_t> pixel_cpu_q4_calls{0}, pixel_cpu_q6_calls{0};

struct alignas(64) pixel_row_sample {
    uint64_t begin = 0, end = 0, rows = 0;
};

struct pixel_packed_task {
    alignas(64) std::atomic<int64_t> next{0};
    alignas(64) std::atomic<int> finished{0};
    pixel_row_sample samples[8];
};

struct pixel_row_total {
    uint64_t calls = 0, rows = 0, work_us = 0, slack_us = 0;
};

static pixel_row_total pixel_row_totals[8];

template<int V>
__attribute__((target("dotprod")))
static void pixel_q4_pair(int n, float * result, const block_q4_K * w, const block_q8_K * const * inputs) {
    const uint8x16_t mask = vdupq_n_u8(15);
    const int32x4_t zero = vdupq_n_s32(0);
    float sums[V] = {};
    for (int i = 0; i < n / 256; ++i) {
        uint32_t scales[4] = {};
        memcpy(scales, w[i].scales, 12);
        uint32x2_t mins8 = vdup_n_u32(0);
        mins8 = vset_lane_u32(scales[1] & 0x3f3f3f3f, mins8, 0);
        mins8 = vset_lane_u32(((scales[2] >> 4) & 0x0f0f0f0f) |
                              (((scales[1] >> 6) & 0x03030303) << 4), mins8, 1);
        scales[1] = (scales[2] & 0x0f0f0f0f) | (((scales[0] >> 6) & 0x03030303) << 4);
        scales[0] &= 0x3f3f3f3f;
        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const auto * scale = reinterpret_cast<const uint8_t *>(scales);
        const float wd = GGML_FP16_TO_FP32(w[i].d);
        const float wm = GGML_FP16_TO_FP32(w[i].dmin);
        int32_t lo[V] = {}, hi[V] = {};
        for (int v = 0; v < V; ++v) {
            const auto & x = inputs[v][i];
            const int16x8_t bsums = vpaddq_s16(vld1q_s16(x.bsums), vld1q_s16(x.bsums + 8));
            const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16(bsums), vget_low_s16(mins)),
                                            vmull_s16(vget_high_s16(bsums), vget_high_s16(mins)));
            sums[v] -= (x.d * wm) * vaddvq_s32(prod);
        }
        for (int j = 0; j < 4; ++j) {
            const uint8x16_t bits0 = vld1q_u8(w[i].qs + 32 * j);
            const uint8x16_t bits1 = vld1q_u8(w[i].qs + 32 * j + 16);
            const int8x16_t low0 = vreinterpretq_s8_u8(vandq_u8(bits0, mask));
            const int8x16_t low1 = vreinterpretq_s8_u8(vandq_u8(bits1, mask));
            const int8x16_t high0 = vreinterpretq_s8_u8(vshrq_n_u8(bits0, 4));
            const int8x16_t high1 = vreinterpretq_s8_u8(vshrq_n_u8(bits1, 4));
            for (int v = 0; v < V; ++v) {
                const int8_t * x = inputs[v][i].qs + 64 * j;
                lo[v] += vaddvq_s32(vdotq_s32(vdotq_s32(zero, low0, vld1q_s8(x)), low1, vld1q_s8(x + 16))) * scale[2*j];
                hi[v] += vaddvq_s32(vdotq_s32(vdotq_s32(zero, high0, vld1q_s8(x + 32)), high1, vld1q_s8(x + 48))) * scale[2*j + 1];
            }
        }
        for (int v = 0; v < V; ++v) sums[v] += (inputs[v][i].d * wd) * (lo[v] + hi[v]);
    }
    for (int v = 0; v < V; ++v) result[v] = sums[v];
}

template<int V>
__attribute__((target("dotprod")))
static void pixel_q6_pair(int n, float * result, const block_q6_K * w, const block_q8_K * const * inputs) {
    const uint8x16_t mask = vdupq_n_u8(15), mask2 = vdupq_n_u8(3);
    const int32x4_t zero = vdupq_n_s32(0);
    float sums[V] = {};
    for (int i = 0; i < n / 256; ++i) {
        const int8x16_t scales = vld1q_s8(w[i].scales);
        const int16x8_t scale0 = vmovl_s8(vget_low_s8(scales));
        const int16x8_t scale1 = vmovl_s8(vget_high_s8(scales));
        int32_t correction[V] = {}, total[V] = {};
        for (int v = 0; v < V; ++v) {
            const int16x8_t sum0 = vld1q_s16(inputs[v][i].bsums);
            const int16x8_t sum1 = vld1q_s16(inputs[v][i].bsums + 8);
            correction[v] = vaddvq_s32(vaddq_s32(
                vaddq_s32(vmull_s16(vget_low_s16(sum0), vget_low_s16(scale0)), vmull_s16(vget_high_s16(sum0), vget_high_s16(scale0))),
                vaddq_s32(vmull_s16(vget_low_s16(sum1), vget_low_s16(scale1)), vmull_s16(vget_high_s16(sum1), vget_high_s16(scale1)))));
        }
        for (int j = 0; j < 2; ++j) {
            const uint8_t * lo = w[i].ql + j * 64;
            const uint8_t * hi = w[i].qh + j * 32;
            const uint8x16_t h0 = vld1q_u8(hi), h1 = vld1q_u8(hi + 16);
            const uint8x16_t l0 = vld1q_u8(lo), l1 = vld1q_u8(lo + 16);
            const uint8x16_t l2 = vld1q_u8(lo + 32), l3 = vld1q_u8(lo + 48);
            const int8x16_t values[8] = {
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l0, mask), vshlq_n_u8(vandq_u8(h0, mask2), 4))),
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l1, mask), vshlq_n_u8(vandq_u8(h1, mask2), 4))),
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l2, mask), vshlq_n_u8(vandq_u8(vshrq_n_u8(h0, 2), mask2), 4))),
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l3, mask), vshlq_n_u8(vandq_u8(vshrq_n_u8(h1, 2), mask2), 4))),
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h0, 4), mask2), 4))),
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l1, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h1, 4), mask2), 4))),
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l2, 4), vshlq_n_u8(vshrq_n_u8(h0, 6), 4))),
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l3, 4), vshlq_n_u8(vshrq_n_u8(h1, 6), 4))),
            };
            for (int g = 0; g < 8; ++g) {
                for (int v = 0; v < V; ++v) {
                    total[v] += vaddvq_s32(vdotq_s32(zero, values[g], vld1q_s8(inputs[v][i].qs + j * 128 + g * 16))) * w[i].scales[j * 8 + g];
                }
            }
        }
        const float wd = GGML_FP16_TO_FP32(w[i].d);
        for (int v = 0; v < V; ++v) sums[v] += (wd * inputs[v][i].d) * (total[v] - 32 * correction[v]);
    }
    for (int v = 0; v < V; ++v) result[v] = sums[v];
}

static void pixel_tuned_packed_dot(ggml_tensor * dst, int ith, int nth, void * userdata) {
    auto & task = *static_cast<pixel_packed_task *>(userdata);
    const auto * weights = dst->src[0];
    const auto * pair = dst->src[1];
    const int64_t rows = weights->ne[1], batch = pair->ne[1] / 2;
    GGML_ASSERT(nth <= 8 && pair->type == GGML_TYPE_Q8_K && weights->ne[0] % 256 == 0);
    GGML_ASSERT(ggml_is_contiguous(weights) && ggml_is_contiguous(pair));
    GGML_ASSERT(weights->type == GGML_TYPE_Q4_K || weights->type == GGML_TYPE_Q6_K);
    const auto dot = ggml_get_type_traits_cpu(weights->type)->vec_dot;
    const auto * w = static_cast<const uint8_t *>(weights->data);
    const auto * q = static_cast<const uint8_t *>(pair->data);
    auto * output = static_cast<float *>(dst->data);
    const uint64_t started = pixel_cpu_row_profile ? now_us() : 0;
    uint64_t completed = 0;
    if (ith == 0) {
        pixel_residual_calls.fetch_add(1, std::memory_order_relaxed);
        (weights->type == GGML_TYPE_Q4_K ? pixel_cpu_q4_calls : pixel_cpu_q6_calls).fetch_add(1, std::memory_order_relaxed);
    }
    auto compute = [&](int64_t begin, int64_t end) {
        completed += uint64_t(end - begin);
        for (int64_t row = begin; row < end; ++row) {
            const void * weight_row = w + row * weights->nb[1];
            for (int64_t b = 0; b < batch; ++b) {
                const block_q8_K * inputs[2] = {
                    reinterpret_cast<const block_q8_K *>(q + b * pair->nb[1]),
                    reinterpret_cast<const block_q8_K *>(q + (b + batch) * pair->nb[1]),
                };
                float values[2] = {};
                if (pixel_cpu_pair_mode == 1) {
                    if (weights->type == GGML_TYPE_Q4_K) pixel_q4_pair<2>(int(weights->ne[0]), values, static_cast<const block_q4_K *>(weight_row), inputs);
                    else pixel_q6_pair<2>(int(weights->ne[0]), values, static_cast<const block_q6_K *>(weight_row), inputs);
                } else if (pixel_cpu_pair_mode == 2) {
                    for (int v = 0; v < 2; ++v) {
                        if (weights->type == GGML_TYPE_Q4_K) pixel_q4_pair<1>(int(weights->ne[0]), values + v, static_cast<const block_q4_K *>(weight_row), inputs + v);
                        else pixel_q6_pair<1>(int(weights->ne[0]), values + v, static_cast<const block_q6_K *>(weight_row), inputs + v);
                    }
                } else {
                    for (int v = 0; v < 2; ++v) dot(int(weights->ne[0]), values + v, 0, weight_row, 0, inputs[v], 0, 1);
                }
                output[b * rows + row] = values[0] + values[1];
            }
        }
    };
    if (pixel_cpu_row_chunk) {
        for (;;) {
            const int64_t begin = task.next.fetch_add(pixel_cpu_row_chunk, std::memory_order_relaxed);
            if (begin >= rows) break;
            compute(begin, std::min(rows, begin + pixel_cpu_row_chunk));
        }
    } else {
        compute(rows * ith / nth, rows * (ith + 1) / nth);
    }
    if (pixel_cpu_row_profile) task.samples[ith] = {started, now_us(), completed};
    if (task.finished.fetch_add(1, std::memory_order_acq_rel) == nth - 1) {
        if (pixel_cpu_row_profile) {
            uint64_t end = 0;
            for (int i = 0; i < nth; ++i) end = std::max(end, task.samples[i].end);
            for (int i = 0; i < nth; ++i) {
                const auto & sample = task.samples[i];
                auto & total = pixel_row_totals[i];
                ++total.calls;
                total.rows += sample.rows;
                total.work_us += sample.end - sample.begin;
                total.slack_us += end - sample.end;
            }
        }
        // The backend graph barrier precedes the next execution of this node.
        task.next.store(0, std::memory_order_relaxed);
        task.finished.store(0, std::memory_order_release);
    }
}
