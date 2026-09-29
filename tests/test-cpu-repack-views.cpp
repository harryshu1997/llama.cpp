#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static ggml_context * make_context() {
    ggml_init_params params = {ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx);
    return ctx;
}

static std::vector<float> evaluate(ggml_backend_t backend, ggml_backend_buffer_type_t buft,
        const std::vector<uint8_t> & weights, int64_t k, int64_t rows, int64_t tokens,
        ggml_type type = GGML_TYPE_Q4_0, int64_t full_k = 256, int64_t full_rows = 64) {
    ggml_context * weights_ctx = make_context();
    ggml_tensor * full = ggml_new_tensor_2d(weights_ctx, type, full_k, full_rows);
    ggml_backend_buffer_t weights_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights_ctx, buft);
    GGML_ASSERT(weights_buffer);
    ggml_backend_tensor_set(full, weights.data(), 0, weights.size());
    const size_t allocated_bytes = ggml_backend_buffer_get_size(weights_buffer);

    ggml_context * ctx = make_context();
    ggml_tensor * prefix = ggml_view_2d(ctx, full, k, rows, full->nb[1], 0);
    GGML_ASSERT(ggml_backend_view_init(prefix) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(prefix->buffer == full->buffer && prefix->data == full->data);
    GGML_ASSERT(ggml_backend_buffer_get_size(weights_buffer) == allocated_bytes);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tokens);
    ggml_tensor * output = ggml_mul_mat(ctx, prefix, input);
    GGML_ASSERT(ggml_backend_supports_op(backend, output));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    std::vector<float> values(k * tokens);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = std::sin(float(i) * 0.37f) + std::cos(float(i) * 0.13f);
    }
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(rows * tokens);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(weights_ctx);
    return result;
}

static std::vector<uint8_t> make_weights(ggml_type type, int64_t k, int64_t rows) {
    std::vector<float> source(k * rows);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = std::sin(float(i) * 0.071f) * (1.0f + float(i / k) * 0.003f);
    }
    std::vector<uint8_t> weights(ggml_row_size(type, k) * rows);
    GGML_ASSERT(ggml_quantize_chunk(type, source.data(), weights.data(), 0, rows, k, nullptr) == weights.size());
    return weights;
}

static bool compare(ggml_backend_t backend, ggml_backend_buffer_type_t packed,
        const std::vector<uint8_t> & weights, int64_t k, int64_t rows, int64_t tokens,
        ggml_type type, int64_t full_k, int64_t full_rows) {
    const auto reference = evaluate(backend, ggml_backend_get_default_buffer_type(backend), weights,
                                    k, rows, tokens, type, full_k, full_rows);
    const auto actual = evaluate(backend, packed, weights, k, rows, tokens, type, full_k, full_rows);
    double error = 0, magnitude = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        GGML_ASSERT(std::isfinite(actual[i]) && std::isfinite(reference[i]));
        error += std::pow(double(actual[i]) - reference[i], 2);
        magnitude += std::pow(double(reference[i]), 2);
    }
    const double nmse = error / std::max(magnitude, 1e-30);
    std::printf("type=%s M=%lld K=%lld N=%lld parent=%lldx%lld nmse=%.9g %s\n", ggml_type_name(type),
                (long long) tokens, (long long) k, (long long) rows, (long long) full_k,
                (long long) full_rows, nmse, nmse <= 5e-4 ? "PASS" : "FAIL");
    return nmse <= 5e-4;
}

static bool rejects_unsupported_views(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    ggml_context * ctx = make_context();
    ggml_tensor * full = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    GGML_ASSERT(buffer);
    bool ok = true;
    for (int64_t rows : {int64_t(3), int64_t(32)}) {
        const size_t offset = rows == 3 ? 0 : full->nb[1];
        ggml_tensor * view = ggml_view_2d(ctx, full, 128, rows, full->nb[1], offset);
        const ggml_status status = ggml_backend_view_init(view);
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 1);
        ggml_tensor * output = ggml_mul_mat(ctx, view, input);
        const bool supported = ggml_backend_supports_op(backend, output);
        ok = ok && status != GGML_STATUS_SUCCESS && !supported;
        std::printf("rejected rows=%lld offset=%zu init=%d supported=%d\n",
                    (long long) rows, offset, int(status), int(supported));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

static bool supports_type(ggml_backend_buffer_type_t buft, ggml_type type) {
    ggml_context * ctx = make_context();
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, ggml_blck_size(type) * 8, 64);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    const bool supported = buffer && weight->extra;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return supported;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(device);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    auto extras = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    ggml_backend_buffer_type_t packed = nullptr;
    for (auto buft = extras ? extras(device) : nullptr; buft && *buft; ++buft) {
        if (std::strcmp(ggml_backend_buft_name(*buft), "CPU_REPACK") == 0) {
            packed = *buft;
        }
    }
    if (!packed) {
        std::puts("SKIP: CPU_REPACK unavailable");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    GGML_ASSERT(backend && set_threads);
    if (!supports_type(packed, GGML_TYPE_Q4_0)) {
        ggml_backend_free(backend);
        std::puts("SKIP: Q4_0 repacking unavailable on this CPU");
        return 77;
    }
    int failed = 0;
    int cases = 0;
    for (int threads : {1, 4}) {
        set_threads(backend, threads);
        std::printf("threads=%d\n", threads);
        for (ggml_type type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K}) {
            if (!supports_type(packed, type)) {
                std::printf("SKIP: %s repacking unavailable on this CPU\n", ggml_type_name(type));
                continue;
            }
            const int64_t block = ggml_blck_size(type);
            const auto weights = make_weights(type, block * 8, 64);
            for (int64_t tokens : {1, 2, 3, 4, 5, 16, 31}) {
                for (int64_t blocks : {2, 3, 4, 6, 8}) {
                    for (int64_t rows : {32, 64}) {
                        failed += !compare(backend, packed, weights, block * blocks, rows, tokens, type, block * 8, 64);
                        ++cases;
                    }
                }
            }
        }
    }
    for (bool down : {false, true}) {
        const int64_t full_k = down ? 8192 : 2048;
        const int64_t full_rows = down ? 2048 : 8192;
        const auto weights = make_weights(GGML_TYPE_Q4_0, full_k, full_rows);
        for (int64_t tokens : {1, 5}) {
            for (int64_t quarters : {1, 2, 3, 4}) {
                failed += !compare(backend, packed, weights, down ? full_k * quarters / 4 : full_k,
                                   down ? full_rows : full_rows * quarters / 4, tokens,
                                   GGML_TYPE_Q4_0, full_k, full_rows);
                ++cases;
            }
        }
    }
    failed += !rejects_unsupported_views(backend, packed);
    ggml_backend_free(backend);
    std::printf("packed prefix cases=%d failures=%d\n", cases, failed);
    return failed ? 1 : 0;
}
