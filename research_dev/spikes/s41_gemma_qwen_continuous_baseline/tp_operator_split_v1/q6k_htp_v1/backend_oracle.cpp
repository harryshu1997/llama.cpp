#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    const int64_t m = argc > 1 ? atoll(argv[1]) : 16;
    const int64_t n = argc > 2 ? atoll(argv[2]) : 1;
    const int64_t k = argc > 3 ? atoll(argv[3]) : 256;
    const char * type_name = argc > 4 ? argv[4] : "q6_K";

    enum ggml_type type = GGML_TYPE_COUNT;
    if (strcmp(type_name, "q4_0") == 0) {
        type = GGML_TYPE_Q4_0;
    } else if (strcmp(type_name, "q6_K") == 0) {
        type = GGML_TYPE_Q6_K;
    } else if (strcmp(type_name, "q8_0") == 0) {
        type = GGML_TYPE_Q8_0;
    }

    const int64_t block_size = type == GGML_TYPE_Q6_K ? 256 : 32;
    if (m <= 0 || n <= 0 || k <= 0 || type == GGML_TYPE_COUNT || k % block_size != 0) {
        fprintf(stderr, "usage: %s [m n k [q4_0|q6_K|q8_0]]\n", argv[0]);
        return 2;
    }

    ggml_backend_load_all();
    ggml_backend_dev_t device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t candidate = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_dev_name(candidate), "HTP0") == 0) {
            device = candidate;
            break;
        }
    }
    if (device == nullptr) {
        fprintf(stderr, "HTP0 not found\n");
        return 2;
    }

    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr || cpu == nullptr) {
        fprintf(stderr, "backend initialization failed\n");
        return 2;
    }

    const ggml_init_params params = {
        8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_context * cpu_ctx = ggml_init(params);
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, type, k, m);
    ggml_tensor * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * output = ggml_mul_mat(ctx, tensor, activation);
    ggml_tensor * cpu_tensor = ggml_new_tensor_2d(cpu_ctx, type, k, m);
    ggml_tensor * cpu_activation = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * cpu_output = ggml_mul_mat(cpu_ctx, cpu_tensor, cpu_activation);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, cpu);

    std::mt19937 gen(0x51396bU);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> input(static_cast<size_t>(k * m));
    std::vector<float> act_data(static_cast<size_t>(k * n));
    for (float & value : input) {
        value = distribution(gen);
    }
    for (float & value : act_data) {
        value = distribution(gen);
    }

    std::vector<uint8_t> encoded(ggml_nbytes(tensor));
    std::vector<uint8_t> decoded(ggml_nbytes(tensor));
    std::vector<float> imatrix(static_cast<size_t>(k), 1.0f);
    ggml_quantize_chunk(type, input.data(), encoded.data(), 0, m, k, imatrix.data());
    ggml_backend_tensor_set(tensor, encoded.data(), 0, encoded.size());
    ggml_backend_tensor_set(activation, act_data.data(), 0, act_data.size() * sizeof(float));
    ggml_backend_tensor_get(tensor, decoded.data(), 0, decoded.size());
    ggml_backend_tensor_set(cpu_tensor, decoded.data(), 0, decoded.size());
    ggml_backend_tensor_set(cpu_activation, act_data.data(), 0, act_data.size() * sizeof(float));

    size_t mismatches = 0;
    for (size_t i = 0; i < encoded.size(); ++i) {
        mismatches += encoded[i] != decoded[i];
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    ggml_build_forward_expand(graph, output);
    ggml_build_forward_expand(cpu_graph, cpu_output);
    const ggml_status htp_status = ggml_backend_graph_compute(backend, graph);
    const ggml_status cpu_status = ggml_backend_graph_compute(cpu, cpu_graph);

    std::vector<float> htp_result(static_cast<size_t>(m * n));
    std::vector<float> cpu_result(static_cast<size_t>(m * n));
    ggml_backend_tensor_get(output, htp_result.data(), 0, htp_result.size() * sizeof(float));
    ggml_backend_tensor_get(cpu_output, cpu_result.data(), 0, cpu_result.size() * sizeof(float));

    double diff2 = 0.0;
    double ref2 = 0.0;
    double max_abs = 0.0;
    bool finite = true;
    for (size_t i = 0; i < htp_result.size(); ++i) {
        const double diff = static_cast<double>(htp_result[i]) - cpu_result[i];
        diff2 += diff * diff;
        ref2 += static_cast<double>(cpu_result[i]) * cpu_result[i];
        max_abs = std::max(max_abs, std::abs(diff));
        finite = finite && std::isfinite(htp_result[i]) && std::isfinite(cpu_result[i]);
    }
    const double nmse = ref2 > 0.0 ? diff2 / ref2 : diff2;
    printf("type=%s shape=[%lld,%lld]x[%lld,%lld] bytes=%zu mismatches=%zu "
           "htp_status=%d cpu_status=%d nmse=%.9g max_abs=%.9g finite=%d\n",
           type_name, static_cast<long long>(k), static_cast<long long>(m),
           static_cast<long long>(k), static_cast<long long>(n), encoded.size(),
           mismatches, htp_status, cpu_status, nmse, max_abs, finite ? 1 : 0);

    ggml_backend_buffer_free(cpu_buffer);
    ggml_backend_buffer_free(buffer);
    ggml_free(cpu_ctx);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    ggml_backend_free(backend);
    ggml_quantize_free();
    return mismatches == 0 && htp_status == GGML_STATUS_SUCCESS &&
                   cpu_status == GGML_STATUS_SUCCESS && finite && nmse <= 5e-4 ? 0 : 1;
}
