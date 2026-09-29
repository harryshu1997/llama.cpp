#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using steady_clock = std::chrono::steady_clock;

double now_ms() {
    return std::chrono::duration<double, std::milli>(
            steady_clock::now().time_since_epoch()).count();
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(
            fraction * static_cast<double>(values.size() - 1));
    return values[index];
}

bool parse_i64(const char * text, int64_t & value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char * end = nullptr;
    value = strtoll(text, &end, 10);
    return end != text && *end == '\0';
}

ggml_type parse_type(const char * name) {
    if (strcmp(name, "f16") == 0) {
        return GGML_TYPE_F16;
    }
    if (strcmp(name, "q4_0") == 0) {
        return GGML_TYPE_Q4_0;
    }
    if (strcmp(name, "q8_0") == 0) {
        return GGML_TYPE_Q8_0;
    }
    return GGML_TYPE_COUNT;
}

bool encode_matrix(
        ggml_type type,
        int64_t columns,
        int64_t rows,
        std::vector<uint8_t> & encoded) {
    const size_t row_bytes = ggml_row_size(type, columns);
    std::vector<float> source(static_cast<size_t>(columns));
    for (int64_t index = 0; index < columns; ++index) {
        source[static_cast<size_t>(index)] =
                std::sin(static_cast<float>(index) * 0.017f) * 0.0625f;
    }
    std::vector<float> imatrix(static_cast<size_t>(columns), 1.0f);
    std::vector<uint8_t> row(row_bytes);
    if (ggml_quantize_chunk(
                type, source.data(), row.data(), 0, 1, columns,
                imatrix.data()) != row_bytes) {
        return false;
    }
    encoded.resize(row_bytes * static_cast<size_t>(rows));
    for (int64_t index = 0; index < rows; ++index) {
        memcpy(encoded.data() + static_cast<size_t>(index) * row_bytes,
               row.data(), row_bytes);
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 11) {
        fprintf(stderr,
                "usage: %s BACKEND TYPE K N M ACTIVATION WARMUP ITERATIONS THREADS IO\n",
                argv[0]);
        return 2;
    }

    const std::string backend_name = argv[1];
    const ggml_type type = parse_type(argv[2]);
    int64_t k = 0;
    int64_t n = 0;
    int64_t m = 0;
    int64_t warmup = 0;
    int64_t iterations = 0;
    int64_t threads = 0;
    const bool swiglu = strcmp(argv[6], "swiglu") == 0;
    const bool geglu = strcmp(argv[6], "geglu") == 0;
    const bool f16_io = strcmp(argv[10], "f16") == 0;
    if (type == GGML_TYPE_COUNT ||
        !parse_i64(argv[3], k) || !parse_i64(argv[4], n) ||
        !parse_i64(argv[5], m) || !parse_i64(argv[7], warmup) ||
        !parse_i64(argv[8], iterations) || !parse_i64(argv[9], threads) ||
        k <= 0 || n <= 0 || m <= 0 || warmup < 0 || iterations <= 0 ||
        threads <= 0 || k % ggml_blck_size(type) != 0 ||
        n % ggml_blck_size(type) != 0 || (!swiglu && !geglu) ||
        (!f16_io && strcmp(argv[10], "f32") != 0)) {
        fprintf(stderr, "invalid FFN benchmark configuration\n");
        return 2;
    }

    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name(backend_name.c_str());
    if (device == nullptr) {
        fprintf(stderr, "backend not found: %s\n", backend_name.c_str());
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "backend initialization failed\n");
        return 1;
    }
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, static_cast<int>(threads));
    }

    const ggml_init_params weight_params = {
        ggml_tensor_overhead() * 8,
        nullptr,
        true,
    };
    ggml_context * weight_ctx = ggml_init(weight_params);
    ggml_tensor * gate_weight = ggml_new_tensor_2d(weight_ctx, type, k, n);
    ggml_tensor * up_weight = ggml_new_tensor_2d(weight_ctx, type, k, n);
    ggml_tensor * down_weight = ggml_new_tensor_2d(weight_ctx, type, n, k);

    ggml_backend_buffer_type_t weight_buft =
            ggml_backend_get_default_buffer_type(backend);
    if (ggml_is_quantized(type) && !ggml_backend_is_cpu(backend)) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        auto get_extra_bufts = reinterpret_cast<
                ggml_backend_dev_get_extra_bufts_t>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_dev_get_extra_bufts"));
        ggml_backend_buffer_type_t * extra =
                get_extra_bufts != nullptr ? get_extra_bufts(device) : nullptr;
        if (extra != nullptr && extra[0] != nullptr) {
            weight_buft = extra[0];
        }
    }
    ggml_backend_buffer_t weight_buffer =
            ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, weight_buft);
    if (weight_buffer == nullptr) {
        fprintf(stderr, "FFN weight allocation failed\n");
        return 1;
    }

    std::vector<uint8_t> encoded;
    if (!encode_matrix(type, k, n, encoded)) {
        fprintf(stderr, "gate/up weight encoding failed\n");
        return 1;
    }
    ggml_backend_tensor_set(gate_weight, encoded.data(), 0, encoded.size());
    ggml_backend_tensor_set(up_weight, encoded.data(), 0, encoded.size());
    if (!encode_matrix(type, n, k, encoded)) {
        fprintf(stderr, "down weight encoding failed\n");
        return 1;
    }
    ggml_backend_tensor_set(down_weight, encoded.data(), 0, encoded.size());
    encoded.clear();
    encoded.shrink_to_fit();
    ggml_backend_synchronize(backend);

    const size_t graph_nodes = 32;
    const ggml_init_params graph_params = {
        ggml_tensor_overhead() * graph_nodes +
                ggml_graph_overhead_custom(graph_nodes, false),
        nullptr,
        true,
    };
    ggml_context * graph_ctx = ggml_init(graph_params);
    ggml_tensor * input_wire = ggml_new_tensor_2d(
            graph_ctx, f16_io ? GGML_TYPE_F16 : GGML_TYPE_F32, k, m);
    ggml_set_input(input_wire);
    ggml_tensor * input = f16_io ?
            ggml_cast(graph_ctx, input_wire, GGML_TYPE_F32) : input_wire;
    ggml_tensor * gate = ggml_mul_mat(graph_ctx, gate_weight, input);
    ggml_tensor * up = ggml_mul_mat(graph_ctx, up_weight, input);
    ggml_tensor * activation = swiglu ?
            ggml_swiglu_split(graph_ctx, gate, up) :
            ggml_geglu_split(graph_ctx, gate, up);
    ggml_tensor * output = ggml_mul_mat(graph_ctx, down_weight, activation);
    ggml_tensor * output_wire = f16_io ?
            ggml_cast(graph_ctx, output, GGML_TYPE_F16) : output;
    ggml_set_output(output_wire);
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, graph_nodes, false);
    ggml_build_forward_expand(graph, output_wire);
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        ggml_tensor * node = ggml_graph_node(graph, index);
        if (!ggml_backend_supports_op(backend, node)) {
            fprintf(stderr, "unsupported operation: %s\n", ggml_op_name(node->op));
            return 1;
        }
    }
    ggml_backend_buffer_t graph_buffer =
            ggml_backend_alloc_ctx_tensors(graph_ctx, backend);
    if (graph_buffer == nullptr) {
        fprintf(stderr, "FFN graph allocation failed\n");
        return 1;
    }

    const size_t input_elements = static_cast<size_t>(k * m);
    const size_t output_elements = input_elements;
    std::vector<float> input_data(input_elements);
    for (size_t index = 0; index < input_data.size(); ++index) {
        input_data[index] =
                std::cos(static_cast<float>(index) * 0.013f) * 0.03125f;
    }
    std::vector<ggml_fp16_t> input_f16(f16_io ? input_elements : 0);
    std::vector<float> output_data(output_elements);
    std::vector<ggml_fp16_t> output_f16(f16_io ? output_elements : 0);
    std::vector<double> compute_samples;
    std::vector<double> total_samples;
    const int64_t total_runs = warmup + iterations;
    double checksum = 0.0;
    for (int64_t run = 0; run < total_runs; ++run) {
        input_data[0] = static_cast<float>(run + 1) * 0.0001f;
        if (f16_io) {
            ggml_fp32_to_fp16_row(
                    input_data.data(), input_f16.data(), input_elements);
        }
        const double total_started = now_ms();
        ggml_backend_tensor_set(
                input_wire,
                f16_io ? static_cast<const void *>(input_f16.data()) :
                         static_cast<const void *>(input_data.data()),
                0,
                input_elements * (f16_io ? sizeof(ggml_fp16_t) : sizeof(float)));
        const double compute_started = now_ms();
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        const double compute_ms = now_ms() - compute_started;
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FFN graph compute failed\n");
            return 1;
        }
        if (f16_io) {
            ggml_backend_tensor_get(
                    output_wire, output_f16.data(), 0,
                    output_elements * sizeof(ggml_fp16_t));
            ggml_fp16_to_fp32_row(
                    output_f16.data(), output_data.data(), output_elements);
        } else {
            ggml_backend_tensor_get(
                    output_wire, output_data.data(), 0,
                    output_elements * sizeof(float));
        }
        if (!std::isfinite(output_data[static_cast<size_t>(run) % output_elements])) {
            fprintf(stderr, "non-finite FFN output\n");
            return 1;
        }
        checksum += output_data[static_cast<size_t>(run) % output_elements];
        if (run >= warmup) {
            compute_samples.push_back(compute_ms);
            total_samples.push_back(now_ms() - total_started);
        }
    }

    printf(
            "{\"backend\":\"%s\",\"description\":\"%s\","
            "\"type\":\"%s\",\"activation\":\"%s\",\"io\":\"%s\","
            "\"k\":%lld,\"n\":%lld,\"m\":%lld,\"threads\":%lld,"
            "\"warmup\":%lld,\"iterations\":%lld,"
            "\"logical_weight_bytes\":%zu,\"backend_weight_bytes\":%zu,"
            "\"wire_bytes_each_way\":%zu,\"compute_p50_ms\":%.6f,"
            "\"compute_p90_ms\":%.6f,\"total_p50_ms\":%.6f,"
            "\"total_p90_ms\":%.6f,\"checksum\":%.9g}\n",
            backend_name.c_str(), ggml_backend_dev_description(device),
            ggml_type_name(type), swiglu ? "swiglu" : "geglu",
            f16_io ? "f16" : "f32", static_cast<long long>(k),
            static_cast<long long>(n), static_cast<long long>(m),
            static_cast<long long>(threads), static_cast<long long>(warmup),
            static_cast<long long>(iterations),
            3 * ggml_row_size(type, k) * static_cast<size_t>(n),
            ggml_backend_buffer_get_size(weight_buffer),
            input_elements * (f16_io ? sizeof(ggml_fp16_t) : sizeof(float)),
            percentile(compute_samples, 0.50),
            percentile(compute_samples, 0.90),
            percentile(total_samples, 0.50),
            percentile(total_samples, 0.90), checksum);

    ggml_backend_buffer_free(graph_buffer);
    ggml_free(graph_ctx);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(weight_ctx);
    ggml_backend_free(backend);
    ggml_quantize_free();
    return 0;
}
