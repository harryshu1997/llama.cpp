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
    if (values.empty()) {
        return 0.0;
    }
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
    if (strcmp(name, "q6_K") == 0) {
        return GGML_TYPE_Q6_K;
    }
    if (strcmp(name, "q8_0") == 0) {
        return GGML_TYPE_Q8_0;
    }
    return GGML_TYPE_COUNT;
}

bool all_finite(const std::vector<float> & values) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 11) {
        fprintf(stderr,
                "usage: %s BACKEND TYPE K N M MATRICES WARMUP ITERATIONS THREADS IO\n",
                argv[0]);
        return 2;
    }

    const std::string backend_name = argv[1];
    const ggml_type type = parse_type(argv[2]);
    int64_t k = 0;
    int64_t n = 0;
    int64_t m = 0;
    int64_t matrix_count = 0;
    int64_t warmup = 0;
    int64_t iterations = 0;
    int64_t threads = 0;
    const bool f16_io = strcmp(argv[10], "f16") == 0;
    if (type == GGML_TYPE_COUNT ||
        !parse_i64(argv[3], k) || !parse_i64(argv[4], n) ||
        !parse_i64(argv[5], m) || !parse_i64(argv[6], matrix_count) ||
        !parse_i64(argv[7], warmup) || !parse_i64(argv[8], iterations) ||
        !parse_i64(argv[9], threads) ||
        k <= 0 || n <= 0 || m <= 0 || matrix_count <= 0 ||
        matrix_count > 8 || warmup < 0 || iterations <= 0 || threads <= 0 ||
        k % ggml_blck_size(type) != 0 ||
        static_cast<uint64_t>(k) * static_cast<uint64_t>(n) >
                std::numeric_limits<size_t>::max() ||
        (!f16_io && strcmp(argv[10], "f32") != 0)) {
        fprintf(stderr, "invalid benchmark configuration\n");
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
        fprintf(stderr, "backend initialization failed: %s\n", backend_name.c_str());
        return 1;
    }
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, static_cast<int>(threads));
    }

    const ggml_init_params weight_params = {
        ggml_tensor_overhead() * static_cast<size_t>(matrix_count + 4),
        nullptr,
        true,
    };
    ggml_context * weight_ctx = ggml_init(weight_params);
    std::vector<ggml_tensor *> weights;
    weights.reserve(static_cast<size_t>(matrix_count));
    for (int64_t index = 0; index < matrix_count; ++index) {
        weights.push_back(ggml_new_tensor_2d(weight_ctx, type, k, n));
    }

    ggml_backend_buffer_type_t weight_buft =
            ggml_backend_get_default_buffer_type(backend);
    if (ggml_is_quantized(type) &&
        ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        auto get_extra_bufts = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_dev_get_extra_bufts"));
        ggml_backend_buffer_type_t * extra_bufts =
                get_extra_bufts != nullptr ? get_extra_bufts(device) : nullptr;
        if (extra_bufts != nullptr && extra_bufts[0] != nullptr) {
            weight_buft = extra_bufts[0];
        }
    }
    ggml_backend_buffer_t weight_buffer =
            ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, weight_buft);
    if (weight_buffer == nullptr) {
        fprintf(stderr, "weight allocation failed\n");
        return 1;
    }

    const size_t row_bytes = ggml_row_size(type, k);
    const size_t weight_bytes = row_bytes * static_cast<size_t>(n);
    std::vector<float> source_row(static_cast<size_t>(k));
    for (int64_t index = 0; index < k; ++index) {
        source_row[static_cast<size_t>(index)] =
                std::sin(static_cast<float>(index) * 0.017f) * 0.125f;
    }
    std::vector<uint8_t> encoded_row(row_bytes);
    std::vector<float> imatrix(static_cast<size_t>(k), 1.0f);
    const size_t quantized = ggml_quantize_chunk(
            type, source_row.data(), encoded_row.data(), 0, 1, k,
            imatrix.data());
    if (quantized != row_bytes) {
        fprintf(stderr, "row quantization failed: %zu != %zu\n", quantized, row_bytes);
        return 1;
    }
    std::vector<uint8_t> encoded_weight(weight_bytes);
    for (int64_t row = 0; row < n; ++row) {
        memcpy(encoded_weight.data() + static_cast<size_t>(row) * row_bytes,
                encoded_row.data(), row_bytes);
    }
    for (ggml_tensor * weight : weights) {
        ggml_backend_tensor_set(weight, encoded_weight.data(), 0, weight_bytes);
    }
    encoded_weight.clear();
    encoded_weight.shrink_to_fit();
    ggml_backend_synchronize(backend);

    const size_t graph_nodes = static_cast<size_t>(matrix_count * 4 + 16);
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
    std::vector<ggml_tensor *> outputs;
    outputs.reserve(static_cast<size_t>(matrix_count));
    for (ggml_tensor * weight : weights) {
        ggml_tensor * output = ggml_mul_mat(graph_ctx, weight, input);
        ggml_tensor * output_wire = f16_io ?
                ggml_cast(graph_ctx, output, GGML_TYPE_F16) : output;
        ggml_set_output(output_wire);
        outputs.push_back(output_wire);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, graph_nodes, false);
    for (ggml_tensor * output : outputs) {
        ggml_build_forward_expand(graph, output);
    }
    for (int node_index = 0; node_index < ggml_graph_n_nodes(graph); ++node_index) {
        ggml_tensor * node = ggml_graph_node(graph, node_index);
        if (!ggml_backend_supports_op(backend, node)) {
            fprintf(stderr, "unsupported operation: %s\n", ggml_op_name(node->op));
            return 1;
        }
    }
    ggml_backend_buffer_t graph_buffer =
            ggml_backend_alloc_ctx_tensors(graph_ctx, backend);
    if (graph_buffer == nullptr) {
        fprintf(stderr, "graph allocation failed\n");
        return 1;
    }
    ggml_backend_buffer_set_usage(graph_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    std::vector<float> input_data(static_cast<size_t>(k * m));
    for (size_t index = 0; index < input_data.size(); ++index) {
        input_data[index] = std::cos(static_cast<float>(index) * 0.013f) * 0.09375f;
    }
    std::vector<ggml_fp16_t> input_f16(
            f16_io ? input_data.size() : 0);
    std::vector<float> output_data(static_cast<size_t>(n * m));
    std::vector<ggml_fp16_t> output_f16(
            f16_io ? output_data.size() : 0);
    std::vector<double> set_samples;
    std::vector<double> compute_samples;
    std::vector<double> get_samples;
    std::vector<double> total_samples;
    set_samples.reserve(static_cast<size_t>(iterations));
    compute_samples.reserve(static_cast<size_t>(iterations));
    get_samples.reserve(static_cast<size_t>(iterations));
    total_samples.reserve(static_cast<size_t>(iterations));

    double checksum = 0.0;
    const int64_t total_runs = warmup + iterations;
    for (int64_t run = 0; run < total_runs; ++run) {
        input_data[0] = static_cast<float>(run + 1) * 0.0001f;
        if (f16_io) {
            ggml_fp32_to_fp16_row(
                    input_data.data(), input_f16.data(), input_data.size());
        }
        const double total_start = now_ms();
        double started = now_ms();
        ggml_backend_tensor_set(
                input_wire,
                f16_io ? static_cast<const void *>(input_f16.data()) :
                         static_cast<const void *>(input_data.data()),
                0, input_data.size() *
                        (f16_io ? sizeof(ggml_fp16_t) : sizeof(float)));
        const double set_ms = now_ms() - started;

        started = now_ms();
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        const double compute_ms = now_ms() - started;
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed: %d\n", static_cast<int>(status));
            return 1;
        }

        started = now_ms();
        for (ggml_tensor * output : outputs) {
            if (f16_io) {
                ggml_backend_tensor_get(
                        output, output_f16.data(), 0,
                        output_f16.size() * sizeof(output_f16[0]));
                ggml_fp16_to_fp32_row(
                        output_f16.data(), output_data.data(), output_data.size());
            } else {
                ggml_backend_tensor_get(
                        output, output_data.data(), 0,
                        output_data.size() * sizeof(output_data[0]));
            }
            if (!all_finite(output_data)) {
                fprintf(stderr, "non-finite output\n");
                return 1;
            }
            checksum += output_data[static_cast<size_t>(run) % output_data.size()];
        }
        const double get_ms = now_ms() - started;
        const double total_ms = now_ms() - total_start;
        if (run >= warmup) {
            set_samples.push_back(set_ms);
            compute_samples.push_back(compute_ms);
            get_samples.push_back(get_ms);
            total_samples.push_back(total_ms);
        }
    }

    const double compute_p50 = percentile(compute_samples, 0.50);
    const double operations = 2.0 * static_cast<double>(k) *
            static_cast<double>(n) * static_cast<double>(m) *
            static_cast<double>(matrix_count);
    const double gops = compute_p50 > 0.0 ? operations / (compute_p50 * 1e6) : 0.0;
    printf(
            "{\"backend\":\"%s\",\"description\":\"%s\","
            "\"type\":\"%s\",\"io\":\"%s\",\"k\":%lld,\"n\":%lld,\"m\":%lld,"
            "\"matrices\":%lld,\"threads\":%lld,\"warmup\":%lld,"
            "\"iterations\":%lld,\"logical_weight_bytes\":%zu,"
            "\"backend_weight_bytes\":%zu,\"wire_input_f16_bytes\":%zu,"
            "\"wire_output_f16_bytes\":%zu,\"set_p50_ms\":%.6f,"
            "\"compute_p50_ms\":%.6f,\"compute_p90_ms\":%.6f,"
            "\"get_p50_ms\":%.6f,\"total_p50_ms\":%.6f,"
            "\"total_p90_ms\":%.6f,\"gops_p50\":%.6f,"
            "\"checksum\":%.9g}\n",
            backend_name.c_str(), ggml_backend_dev_description(device),
            ggml_type_name(type), f16_io ? "f16" : "f32",
            static_cast<long long>(k),
            static_cast<long long>(n), static_cast<long long>(m),
            static_cast<long long>(matrix_count), static_cast<long long>(threads),
            static_cast<long long>(warmup), static_cast<long long>(iterations),
            weight_bytes * static_cast<size_t>(matrix_count),
            ggml_backend_buffer_get_size(weight_buffer),
            static_cast<size_t>(k * m) * sizeof(ggml_fp16_t),
            static_cast<size_t>(n * m * matrix_count) * sizeof(ggml_fp16_t),
            percentile(set_samples, 0.50), compute_p50,
            percentile(compute_samples, 0.90), percentile(get_samples, 0.50),
            percentile(total_samples, 0.50), percentile(total_samples, 0.90),
            gops, checksum);

    ggml_backend_buffer_free(graph_buffer);
    ggml_free(graph_ctx);
    ggml_backend_buffer_free(weight_buffer);
    ggml_free(weight_ctx);
    ggml_backend_free(backend);
    ggml_quantize_free();
    return 0;
}
