// Causal Qwen/Gemma-shaped layer probe.
//
// This is a synthetic layer benchmark, not a llama.cpp decoder. It enforces the
// transformer dependency that the FFN input is produced by attention before
// the host and phone execute complementary FFN slices. Both sides generate the
// same quantized master weights, and every treatment is checked against a
// monolithic host reference before timing is reported.
//
// usage: causal_layer_host <n_kv> <iters> <phone_cols> [weight_type] [mode] [shape]
// mode: all (default), monolithic, two_phase, or split
// shape: qwen3_14b (default) or gemma4_12b

#include "causal_ffn_protocol.h"
#include "causal_quantized_weights.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern "C" {
struct libusb_context;
struct libusb_device;
struct libusb_device_handle;

struct libusb_device_descriptor {
    uint8_t  bLength;
    uint8_t  bDescriptorType;
    uint16_t bcdUSB;
    uint8_t  bDeviceClass;
    uint8_t  bDeviceSubClass;
    uint8_t  bDeviceProtocol;
    uint8_t  bMaxPacketSize0;
    uint16_t idVendor;
    uint16_t idProduct;
    uint16_t bcdDevice;
    uint8_t  iManufacturer;
    uint8_t  iProduct;
    uint8_t  iSerialNumber;
    uint8_t  bNumConfigurations;
};

int libusb_init(libusb_context **);
void libusb_exit(libusb_context *);
ssize_t libusb_get_device_list(libusb_context *, libusb_device ***);
void libusb_free_device_list(libusb_device **, int);
uint8_t libusb_get_bus_number(libusb_device *);
uint8_t libusb_get_device_address(libusb_device *);
int libusb_get_device_descriptor(libusb_device *, libusb_device_descriptor *);
int libusb_open(libusb_device *, libusb_device_handle **);
void libusb_close(libusb_device_handle *);
int libusb_claim_interface(libusb_device_handle *, int);
int libusb_release_interface(libusb_device_handle *, int);
int libusb_detach_kernel_driver(libusb_device_handle *, int);
int libusb_bulk_transfer(
        libusb_device_handle *, unsigned char, unsigned char *, int, int *, unsigned int);
const char * libusb_error_name(int);
}

using steady_clock = std::chrono::steady_clock;

static double elapsed_ms(steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(
            steady_clock::now() - started).count();
}

static int64_t unix_time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

static bool parse_type(const char * name, ggml_type & result) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const ggml_type type = (ggml_type) i;
        const char * type_name = ggml_type_name(type);
        if (type_name != nullptr && name == std::string(type_name)) {
            result = type;
            return true;
        }
    }
    return false;
}

static bool is_accessory_pid(uint16_t pid) {
    return pid == 0x2d00 || pid == 0x2d01 || pid == 0x2d04 || pid == 0x2d05;
}

static libusb_device_handle * open_accessory(
        libusb_context * context, int wanted_bus, int wanted_address) {
    libusb_device ** devices = nullptr;
    const ssize_t count = libusb_get_device_list(context, &devices);
    if (count < 0) {
        fprintf(stderr, "[causal-host] libusb device list failed: %zd\n", count);
        return nullptr;
    }

    libusb_device * selected = nullptr;
    int candidates = 0;
    int selected_bus = -1;
    int selected_address = -1;
    for (ssize_t i = 0; i < count; ++i) {
        libusb_device_descriptor descriptor = {};
        if (libusb_get_device_descriptor(devices[i], &descriptor) != 0 ||
            descriptor.idVendor != 0x18d1 || !is_accessory_pid(descriptor.idProduct)) {
            continue;
        }
        const int bus = libusb_get_bus_number(devices[i]);
        const int address = libusb_get_device_address(devices[i]);
        if ((wanted_bus >= 0 && bus != wanted_bus) ||
            (wanted_address >= 0 && address != wanted_address)) {
            continue;
        }
        ++candidates;
        selected = devices[i];
        selected_bus = bus;
        selected_address = address;
    }

    libusb_device_handle * handle = nullptr;
    if (candidates == 1) {
        const int status = libusb_open(selected, &handle);
        if (status != 0) {
            fprintf(stderr, "[causal-host] open accessory failed: %s\n",
                    libusb_error_name(status));
            handle = nullptr;
        } else {
            fprintf(stderr, "[causal-host] accessory bus=%d address=%d\n",
                    selected_bus, selected_address);
        }
    } else if (candidates == 0) {
        fprintf(stderr, "[causal-host] no matching accessory found\n");
    } else {
        fprintf(stderr,
                "[causal-host] %d accessories match; set S41_AOA_BUS and S41_AOA_ADDRESS\n",
                candidates);
    }

    libusb_free_device_list(devices, 1);
    return handle;
}

static bool usb_transfer(
        libusb_device_handle * handle, uint8_t endpoint, void * data, size_t size) {
    uint8_t * ptr = (uint8_t *) data;
    size_t completed = 0;
    while (completed < size) {
        int transferred = 0;
        const size_t remaining = size - completed;
        if (remaining > (size_t) INT32_MAX) {
            return false;
        }
        const int status = libusb_bulk_transfer(
                handle, endpoint, ptr + completed, (int) remaining,
                &transferred, 4000);
        if (status != 0) {
            fprintf(stderr, "[causal-host] USB endpoint 0x%02x failed: %s\n",
                    endpoint, libusb_error_name(status));
            return false;
        }
        if (transferred <= 0) {
            fprintf(stderr, "[causal-host] USB endpoint 0x%02x made no progress\n",
                    endpoint);
            return false;
        }
        completed += (size_t) transferred;
    }
    return true;
}

static bool initialize_weight(
        ggml_tensor * tensor,
        ggml_type type,
        uint32_t matrix,
        uint32_t row_offset,
        uint32_t column_offset,
        uint64_t * weight_hash = nullptr) {
    const int64_t columns = tensor->ne[0];
    const int64_t rows = tensor->ne[1];
    std::vector<uint8_t> quantized(ggml_nbytes(tensor));
    if (!s41_fill_quantized_weight(
                type, matrix, row_offset, column_offset,
                (uint32_t) columns, (uint32_t) rows,
                quantized.data(), quantized.size())) {
        fprintf(stderr, "[causal-host] deterministic weight fill failed\n");
        return false;
    }
    if (weight_hash != nullptr) {
        *weight_hash = s41_weight_hash_update(
                *weight_hash, matrix, row_offset, column_offset,
                (uint32_t) columns, (uint32_t) rows,
                quantized.data(), quantized.size());
    }
    ggml_backend_tensor_set(tensor, quantized.data(), 0, quantized.size());
    return true;
}

static bool phone_ffn(
        libusb_device_handle * handle,
        uint32_t request_id,
        ggml_type type,
        uint32_t k,
        uint32_t n_ff,
        uint32_t offset,
        uint32_t count,
        uint64_t weight_hash,
        bool f16_io,
        const std::vector<float> & input,
        std::vector<float> & output,
        std::vector<ggml_fp16_t> & encoded_output,
        double & duration_ms) {
    std::vector<ggml_fp16_t> encoded_input(
            f16_io ? (size_t) k : 0);
    const void * input_payload = input.data();
    if (f16_io) {
        ggml_fp32_to_fp16_row(
                input.data(), encoded_input.data(), encoded_input.size());
        input_payload = encoded_input.data();
    }
    s41_ffn_request request = {};
    request.magic = S41_FFN_REQUEST_MAGIC;
    request.version = S41_FFN_PROTOCOL_VERSION;
    request.reserved = s41_ffn_encode_flags(
            S41_FFN_FLAG_FAST_HASH |
            (f16_io ? S41_FFN_FLAG_F16_INPUT : 0) |
            (f16_io ? S41_FFN_FLAG_F16_OUTPUT : 0) |
            S41_FFN_FLAG_RESIDUAL_OUTPUT,
            1);
    request.request_id = request_id;
    request.type = (uint32_t) type;
    request.k = k;
    request.n_ff = n_ff;
    request.offset = offset;
    request.count = count;
    request.input_bytes = (f16_io ? sizeof(ggml_fp16_t) : sizeof(float)) * k;
    request.input_hash =
            s41_fast_hash_bytes(input_payload, request.input_bytes);
    request.weight_hash = weight_hash;

    std::vector<uint8_t> request_data(sizeof(request) + request.input_bytes);
    memcpy(request_data.data(), &request, sizeof(request));
    memcpy(
            request_data.data() + sizeof(request),
            input_payload, request.input_bytes);

    const size_t expected_output_bytes =
            (f16_io ? sizeof(ggml_fp16_t) : sizeof(float)) * (size_t) k;
    std::vector<uint8_t> response_data(
            sizeof(s41_ffn_response) + expected_output_bytes);
    const auto started = steady_clock::now();
    if (!usb_transfer(handle, 0x01, request_data.data(), request_data.size()) ||
        !usb_transfer(handle, 0x81, response_data.data(), response_data.size())) {
        return false;
    }
    duration_ms = elapsed_ms(started);

    s41_ffn_response response = {};
    memcpy(&response, response_data.data(), sizeof(response));
    if (response.magic != S41_FFN_RESPONSE_MAGIC ||
        response.version != S41_FFN_PROTOCOL_VERSION ||
        response.status != 0 ||
        response.request_id != request_id ||
        response.k != k ||
        response.output_bytes != expected_output_bytes ||
        response.weight_hash != weight_hash) {
        fprintf(stderr, "[causal-host] invalid phone response header\n");
        return false;
    }
    const void * output_payload = response_data.data() + sizeof(response);
    if (response.output_hash !=
        s41_fast_hash_bytes(output_payload, expected_output_bytes)) {
        fprintf(stderr, "[causal-host] phone response hash mismatch\n");
        return false;
    }
    if (f16_io) {
        memcpy(
                encoded_output.data(), output_payload,
                expected_output_bytes);
        ggml_fp16_to_fp32_row(
                encoded_output.data(), output.data(), output.size());
    } else {
        memcpy(output.data(), output_payload, expected_output_bytes);
    }
    return true;
}

struct error_metrics {
    double relative_l2;
    double max_absolute;
    size_t reference_argmax;
    size_t candidate_argmax;
    size_t non_finite;
};

static error_metrics compare(
        const std::vector<float> & reference,
        const std::vector<float> & candidate) {
    double error_squared = 0.0;
    double reference_squared = 0.0;
    double max_absolute = 0.0;
    size_t non_finite = 0;
    size_t reference_argmax = 0;
    size_t candidate_argmax = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i])) {
            ++non_finite;
            continue;
        }
        const double difference = (double) candidate[i] - reference[i];
        error_squared += difference * difference;
        reference_squared += (double) reference[i] * reference[i];
        max_absolute = std::max(max_absolute, std::abs(difference));
        if (reference[i] > reference[reference_argmax]) {
            reference_argmax = i;
        }
        if (candidate[i] > candidate[candidate_argmax]) {
            candidate_argmax = i;
        }
    }
    const double relative_l2 =
            reference_squared > 0.0 ? std::sqrt(error_squared / reference_squared) : INFINITY;
    return {relative_l2, max_absolute, reference_argmax, candidate_argmax, non_finite};
}

static double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = std::min(
            values.size() - 1, (size_t) std::floor(fraction * values.size()));
    return values[index];
}

static void add_nodes(ggml_cgraph * graph, std::initializer_list<ggml_tensor *> nodes) {
    for (ggml_tensor * node : nodes) {
        ggml_graph_add_node(graph, node);
    }
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <n_kv> <iters> <phone_cols> [weight_type] "
                "[mode] [shape]\n",
                argv[0]);
        return 2;
    }

    const int64_t n_kv = atoll(argv[1]);
    const int iterations = atoi(argv[2]);
    const int64_t phone_columns = atoll(argv[3]);
    const std::string mode = argc > 5 ? argv[5] : "all";
    const std::string shape = argc > 6 ? argv[6] : "qwen3_14b";
    int64_t k = 5120;
    int64_t n_heads = 40;
    int64_t n_kv_heads = 8;
    int64_t head_dimension = 128;
    int64_t n_ff = 17408;
    if (shape == "gemma4_12b") {
        k = 3840;
        n_heads = 16;
        n_kv_heads = 8;
        head_dimension = 256;
        n_ff = 15360;
    } else if (shape != "qwen3_14b") {
        fprintf(stderr, "[causal-host] unknown shape '%s'\n", shape.c_str());
        return 2;
    }
    const int64_t gpu_columns = n_ff - phone_columns;

    const char * phone_io_env = getenv("S41_PHONE_IO");
    const bool phone_f16_io = phone_io_env == nullptr ||
            strcmp(phone_io_env, "f16") == 0;
    if (!phone_f16_io && strcmp(phone_io_env, "f32") != 0) {
        fprintf(stderr, "[causal-host] invalid S41_PHONE_IO\n");
        return 2;
    }

    ggml_type weight_type = GGML_TYPE_Q4_0;
    if (argc > 4 && !parse_type(argv[4], weight_type)) {
        fprintf(stderr, "[causal-host] unknown weight type '%s'\n", argv[4]);
        return 2;
    }
    if (weight_type != GGML_TYPE_Q4_0 && weight_type != GGML_TYPE_Q8_0) {
        fprintf(stderr, "[causal-host] only q4_0 and q8_0 are supported\n");
        return 2;
    }
    if (mode != "all" && mode != "monolithic" && mode != "two_phase" &&
        mode != "split") {
        fprintf(stderr, "[causal-host] unknown mode '%s'\n", mode.c_str());
        return 2;
    }
    const int64_t block = ggml_blck_size(weight_type);
    if (n_kv <= 0 || iterations <= 0 || phone_columns < 0 || gpu_columns <= 0 ||
        k % block != 0 || n_ff % block != 0 ||
        phone_columns % block != 0 || gpu_columns % block != 0 ||
        (mode == "split" && phone_columns == 0)) {
        fprintf(stderr,
                "[causal-host] invalid dimensions n_kv=%lld iterations=%d "
                "phone_columns=%lld block=%lld\n",
                (long long) n_kv, iterations, (long long) phone_columns,
                (long long) block);
        return 2;
    }

    const char * host_backend_env = getenv("S41_HOST_BACKEND");
    const bool host_is_cpu = host_backend_env != nullptr &&
            (strcmp(host_backend_env, "cpu") == 0 ||
             strcmp(host_backend_env, "CPU") == 0);
    if (host_backend_env != nullptr && !host_is_cpu &&
        strcmp(host_backend_env, "gpu") != 0 &&
        strcmp(host_backend_env, "GPU") != 0 &&
        strcmp(host_backend_env, "cuda") != 0 &&
        strcmp(host_backend_env, "CUDA") != 0) {
        fprintf(stderr, "[causal-host] invalid S41_HOST_BACKEND\n");
        return 2;
    }
    const enum ggml_backend_dev_type wanted_host_type = host_is_cpu
            ? GGML_BACKEND_DEVICE_TYPE_CPU
            : GGML_BACKEND_DEVICE_TYPE_GPU;
    ggml_backend_dev_t host_device = nullptr;
    ggml_backend_t backend = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) == wanted_host_type) {
            host_device = device;
            backend = ggml_backend_dev_init(device, nullptr);
            break;
        }
    }
    if (backend == nullptr) {
        fprintf(stderr, "[causal-host] requested backend not found\n");
        return 1;
    }
    int cpu_threads = 0;
    if (host_is_cpu) {
        const char * thread_env = getenv("S41_CPU_THREADS");
        cpu_threads = GGML_DEFAULT_N_THREADS;
        if (thread_env != nullptr) {
            char * end = nullptr;
            const long parsed = strtol(thread_env, &end, 10);
            if (end == thread_env || *end != '\0' ||
                parsed <= 0 || parsed > 1024) {
                fprintf(stderr, "[causal-host] invalid S41_CPU_THREADS\n");
                return 2;
            }
            cpu_threads = (int) parsed;
        }
        ggml_backend_cpu_set_n_threads(backend, cpu_threads);
    }
    fprintf(stderr,
            "[causal-host] host_backend=%s description=%s threads=%d\n",
            host_is_cpu ? "CPU" : "GPU",
            ggml_backend_dev_description(host_device),
            host_is_cpu ? cpu_threads : 0);
    const char * host_tag = host_is_cpu ? "cpu" : "cuda";

    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead() * 8;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return 1;
    }

    const int64_t n_q = n_heads * head_dimension;
    const int64_t n_kv_projection = n_kv_heads * head_dimension;

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    ggml_tensor * attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
    ggml_tensor * ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
    ggml_tensor * wq = ggml_new_tensor_2d(ctx, weight_type, k, n_q);
    ggml_tensor * wk = ggml_new_tensor_2d(ctx, weight_type, k, n_kv_projection);
    ggml_tensor * wv = ggml_new_tensor_2d(ctx, weight_type, k, n_kv_projection);
    ggml_tensor * wo = ggml_new_tensor_2d(ctx, weight_type, n_q, k);
    ggml_tensor * key_cache = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F16, head_dimension, n_kv, n_kv_heads);
    ggml_tensor * value_cache = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F16, n_kv, head_dimension, n_kv_heads);

    ggml_tensor * normed_input =
            ggml_mul(ctx, ggml_rms_norm(ctx, input, 1e-6f), attention_norm);
    ggml_tensor * q = ggml_mul_mat(ctx, wq, normed_input);
    ggml_tensor * projected_k = ggml_mul_mat(ctx, wk, normed_input);
    ggml_tensor * projected_v = ggml_mul_mat(ctx, wv, normed_input);
    ggml_tensor * q_heads = ggml_permute(
            ctx, ggml_reshape_3d(ctx, q, head_dimension, n_heads, 1), 0, 2, 1, 3);
    ggml_tensor * scores = ggml_mul_mat(ctx, key_cache, q_heads);
    scores = ggml_soft_max_ext(
            ctx, scores, nullptr, 1.0f / std::sqrt((float) head_dimension), 0.0f);
    ggml_tensor * attended = ggml_mul_mat(ctx, value_cache, scores);
    ggml_tensor * merged = ggml_cont_2d(
            ctx, ggml_permute(ctx, attended, 0, 2, 1, 3), n_q, 1);
    ggml_tensor * attention_output = ggml_mul_mat(ctx, wo, merged);
    ggml_tensor * residual = ggml_add(ctx, input, attention_output);
    residual = ggml_add(
            ctx, residual,
            ggml_scale(
                    ctx,
                    ggml_pad(
                            ctx, ggml_add(ctx, projected_k, projected_v),
                            k - n_kv_projection, 0, 0, 0),
                    0.0f));
    ggml_tensor * ffn_input =
            ggml_mul(ctx, ggml_rms_norm(ctx, residual, 1e-6f), ffn_norm);

    ggml_tensor * full_gate =
            ggml_new_tensor_2d(ctx, weight_type, k, n_ff);
    ggml_tensor * full_up =
            ggml_new_tensor_2d(ctx, weight_type, k, n_ff);
    ggml_tensor * full_down =
            ggml_new_tensor_2d(ctx, weight_type, n_ff, k);
    ggml_tensor * full_gate_output = ggml_mul_mat(ctx, full_gate, ffn_input);
    ggml_tensor * full_up_output = ggml_mul_mat(ctx, full_up, ffn_input);
    ggml_tensor * full_silu = ggml_silu(ctx, full_gate_output);
    ggml_tensor * full_activation = ggml_mul(ctx, full_silu, full_up_output);
    ggml_tensor * full_down_output = ggml_mul_mat(ctx, full_down, full_activation);
    ggml_tensor * full_output = ggml_add(ctx, residual, full_down_output);

    ggml_tensor * gpu_gate = nullptr;
    ggml_tensor * gpu_up = nullptr;
    ggml_tensor * gpu_down = nullptr;
    ggml_tensor * gpu_gate_output = nullptr;
    ggml_tensor * gpu_up_output = nullptr;
    ggml_tensor * gpu_silu = nullptr;
    ggml_tensor * gpu_activation = nullptr;
    ggml_tensor * gpu_down_output = nullptr;
    ggml_tensor * gpu_output = nullptr;
    ggml_tensor * oracle_gate = nullptr;
    ggml_tensor * oracle_up = nullptr;
    ggml_tensor * oracle_down = nullptr;
    ggml_tensor * oracle_gate_output = nullptr;
    ggml_tensor * oracle_up_output = nullptr;
    ggml_tensor * oracle_silu = nullptr;
    ggml_tensor * oracle_activation = nullptr;
    ggml_tensor * oracle_output = nullptr;
    ggml_tensor * returned_phone_residual = nullptr;
    ggml_tensor * combined_split_output = nullptr;
    if (phone_columns > 0) {
        gpu_gate = ggml_new_tensor_2d(ctx, weight_type, k, gpu_columns);
        gpu_up = ggml_new_tensor_2d(ctx, weight_type, k, gpu_columns);
        gpu_down = ggml_new_tensor_2d(ctx, weight_type, gpu_columns, k);
        gpu_gate_output = ggml_mul_mat(ctx, gpu_gate, ffn_input);
        gpu_up_output = ggml_mul_mat(ctx, gpu_up, ffn_input);
        gpu_silu = ggml_silu(ctx, gpu_gate_output);
        gpu_activation = ggml_mul(ctx, gpu_silu, gpu_up_output);
        gpu_down_output = ggml_mul_mat(ctx, gpu_down, gpu_activation);
        gpu_output = ggml_add(ctx, residual, gpu_down_output);

        oracle_gate = ggml_new_tensor_2d(ctx, weight_type, k, phone_columns);
        oracle_up = ggml_new_tensor_2d(ctx, weight_type, k, phone_columns);
        oracle_down = ggml_new_tensor_2d(ctx, weight_type, phone_columns, k);
        oracle_gate_output = ggml_mul_mat(ctx, oracle_gate, ffn_input);
        oracle_up_output = ggml_mul_mat(ctx, oracle_up, ffn_input);
        oracle_silu = ggml_silu(ctx, oracle_gate_output);
        oracle_activation = ggml_mul(ctx, oracle_silu, oracle_up_output);
        oracle_output = ggml_mul_mat(ctx, oracle_down, oracle_activation);
        returned_phone_residual =
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
        combined_split_output = ggml_add(
                ctx, gpu_output, returned_phone_residual);
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        fprintf(stderr, "[causal-host] backend allocation failed\n");
        return 1;
    }

    fprintf(stderr, "[causal-host] initializing deterministic weights\n");
    uint64_t phone_weight_hash = S41_HASH64_OFFSET;
    if (!initialize_weight(wq, weight_type, S41_WEIGHT_Q, 0, 0) ||
        !initialize_weight(wk, weight_type, S41_WEIGHT_K, 0, 0) ||
        !initialize_weight(wv, weight_type, S41_WEIGHT_V, 0, 0) ||
        !initialize_weight(wo, weight_type, S41_WEIGHT_O, 0, 0) ||
        !initialize_weight(full_gate, weight_type, S41_WEIGHT_GATE, 0, 0) ||
        !initialize_weight(full_up, weight_type, S41_WEIGHT_UP, 0, 0) ||
        !initialize_weight(full_down, weight_type, S41_WEIGHT_DOWN, 0, 0) ||
        (phone_columns > 0 &&
         (!initialize_weight(gpu_gate, weight_type, S41_WEIGHT_GATE, 0, 0) ||
          !initialize_weight(gpu_up, weight_type, S41_WEIGHT_UP, 0, 0) ||
          !initialize_weight(gpu_down, weight_type, S41_WEIGHT_DOWN, 0, 0) ||
          !initialize_weight(
                  oracle_gate, weight_type, S41_WEIGHT_GATE,
                  (uint32_t) gpu_columns, 0, &phone_weight_hash) ||
          !initialize_weight(
                  oracle_up, weight_type, S41_WEIGHT_UP,
                  (uint32_t) gpu_columns, 0, &phone_weight_hash) ||
          !initialize_weight(
                  oracle_down, weight_type, S41_WEIGHT_DOWN,
                  0, (uint32_t) gpu_columns, &phone_weight_hash)))) {
        return 1;
    }
    if (phone_columns > 0) {
        fprintf(stderr, "[causal-host] phone_weight_hash=%016llx\n",
                (unsigned long long) phone_weight_hash);
    }

    std::vector<float> norm((size_t) k, 1.0f);
    std::vector<float> input_data((size_t) k);
    for (uint32_t i = 0; i < (uint32_t) k; ++i) {
        input_data[i] = s41_activation_value(i);
    }
    std::vector<uint16_t> key_data(ggml_nelements(key_cache), 0x3400);
    std::vector<uint16_t> value_data(ggml_nelements(value_cache), 0x3400);
    ggml_backend_tensor_set(attention_norm, norm.data(), 0, norm.size() * sizeof(float));
    ggml_backend_tensor_set(ffn_norm, norm.data(), 0, norm.size() * sizeof(float));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(key_cache, key_data.data(), 0, key_data.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(
            value_cache, value_data.data(), 0, value_data.size() * sizeof(uint16_t));
    auto set_input_sample = [&](uint32_t sample) {
        const uint32_t shift = 17 * (sample + 1);
        for (uint32_t i = 0; i < (uint32_t) k; ++i) {
            input_data[i] = s41_activation_value(i + shift);
        }
        ggml_backend_tensor_set(
                input, input_data.data(), 0,
                input_data.size() * sizeof(float));
    };

    size_t cache_flush_mib = 0;
    const char * cache_flush_env = getenv("S41_HOST_CACHE_FLUSH_MIB");
    if (cache_flush_env != nullptr) {
        char * end = nullptr;
        const unsigned long long parsed =
                strtoull(cache_flush_env, &end, 10);
        if (end == cache_flush_env || *end != '\0' || parsed > 4096) {
            fprintf(stderr,
                    "[causal-host] invalid S41_HOST_CACHE_FLUSH_MIB\n");
            return 2;
        }
        cache_flush_mib = (size_t) parsed;
    }
    std::vector<uint8_t> cache_flush(
            cache_flush_mib * 1024ULL * 1024ULL, 1);
    volatile uint64_t cache_flush_sink = 0;
    auto evict_host_cache = [&]() {
        for (size_t offset = 0; offset < cache_flush.size(); offset += 64) {
            cache_flush_sink += cache_flush[offset];
        }
    };

    ggml_cgraph * monolithic_graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(monolithic_graph, full_output);
    ggml_cgraph * phase_a_graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(phase_a_graph, ffn_input);
    ggml_cgraph * full_phase_b_graph = ggml_new_graph_custom(ctx, 16, false);
    add_nodes(full_phase_b_graph, {
            full_gate_output, full_up_output, full_silu,
            full_activation, full_down_output, full_output});
    ggml_cgraph * split_phase_b_graph = nullptr;
    ggml_cgraph * oracle_phone_graph = nullptr;
    ggml_cgraph * recursive_split_graph = nullptr;
    ggml_cgraph * recursive_oracle_graph = nullptr;
    ggml_cgraph * late_merge_graph = nullptr;
    if (phone_columns > 0) {
        split_phase_b_graph = ggml_new_graph_custom(ctx, 16, false);
        add_nodes(split_phase_b_graph, {
                gpu_gate_output, gpu_up_output, gpu_silu,
                gpu_activation, gpu_down_output, gpu_output});
        oracle_phone_graph = ggml_new_graph_custom(ctx, 16, false);
        add_nodes(oracle_phone_graph, {
                oracle_gate_output, oracle_up_output, oracle_silu,
                oracle_activation, oracle_output});
        recursive_split_graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(recursive_split_graph, gpu_output);
        recursive_oracle_graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(recursive_oracle_graph, oracle_output);
        late_merge_graph = ggml_new_graph_custom(ctx, 8, false);
        const enum ggml_op gpu_output_op = gpu_output->op;
        ggml_tensor * gpu_output_sources[GGML_MAX_SRC];
        memcpy(
                gpu_output_sources, gpu_output->src,
                sizeof(gpu_output_sources));
        gpu_output->op = GGML_OP_NONE;
        memset(gpu_output->src, 0, sizeof(gpu_output->src));
        ggml_build_forward_expand(late_merge_graph, combined_split_output);
        gpu_output->op = gpu_output_op;
        memcpy(
                gpu_output->src, gpu_output_sources,
                sizeof(gpu_output_sources));
    }

    fprintf(stderr,
            "[causal-host] shape=%s n_kv=%lld type=%s phone_columns=%lld "
            "phone_io=%s nodes(monolithic=%d phase_a=%d)\n",
            shape.c_str(), (long long) n_kv, ggml_type_name(weight_type),
            (long long) phone_columns, phone_f16_io ? "f16" : "f32",
            ggml_graph_n_nodes(monolithic_graph),
            ggml_graph_n_nodes(phase_a_graph));

    libusb_context * usb_context = nullptr;
    libusb_device_handle * usb = nullptr;
    if (phone_columns > 0) {
        if (libusb_init(&usb_context) != 0) {
            fprintf(stderr, "[causal-host] libusb initialization failed\n");
            return 1;
        }
        const char * bus_env = getenv("S41_AOA_BUS");
        const char * address_env = getenv("S41_AOA_ADDRESS");
        const int wanted_bus = bus_env != nullptr ? atoi(bus_env) : -1;
        const int wanted_address = address_env != nullptr ? atoi(address_env) : -1;
        usb = open_accessory(usb_context, wanted_bus, wanted_address);
        if (usb == nullptr) {
            return 1;
        }
        libusb_detach_kernel_driver(usb, 0);
        const int status = libusb_claim_interface(usb, 0);
        if (status != 0) {
            fprintf(stderr, "[causal-host] claim interface failed: %s\n",
                    libusb_error_name(status));
            return 1;
        }
    }

    std::vector<float> activation_host((size_t) k);
    std::vector<float> monolithic_output((size_t) k);
    std::vector<float> two_phase_output((size_t) k);
    std::vector<float> split_output((size_t) k);
    std::vector<float> local_split_output((size_t) k);
    std::vector<float> recursive_split_output((size_t) k);
    std::vector<float> phone_output((size_t) k);
    std::vector<ggml_fp16_t> phone_output_f16((size_t) k);
    std::vector<float> oracle_phone_output((size_t) k);
    std::vector<float> rounded_phone_oracle_output((size_t) k);
    std::vector<float> rounded_activation((size_t) k);
    std::vector<ggml_fp16_t> rounded_activation_f16((size_t) k);
    uint32_t request_id = 1;

    auto compute = [&](ggml_cgraph * graph) {
        const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[causal-host] graph compute failed: %d\n", (int) status);
            return false;
        }
        return true;
    };

    auto run_monolithic = [&](std::vector<float> & output) {
        evict_host_cache();
        const auto started = steady_clock::now();
        if (!compute(monolithic_graph)) {
            return -1.0;
        }
        ggml_backend_tensor_get(
                full_output, output.data(), 0, output.size() * sizeof(float));
        return elapsed_ms(started);
    };

    auto run_two_phase = [&](std::vector<float> & output) {
        evict_host_cache();
        const auto started = steady_clock::now();
        if (!compute(phase_a_graph)) {
            return -1.0;
        }
        ggml_backend_tensor_get(
                ffn_input, activation_host.data(), 0,
                activation_host.size() * sizeof(float));
        if (!compute(full_phase_b_graph)) {
            return -1.0;
        }
        ggml_backend_tensor_get(
                full_output, output.data(), 0, output.size() * sizeof(float));
        return elapsed_ms(started);
    };

    double last_phone_ms = 0.0;
    double last_phase_a_ms = 0.0;
    double last_overlap_span_ms = 0.0;
    double last_late_merge_ms = 0.0;
    double last_late_upload_ms = 0.0;
    double last_late_compute_ms = 0.0;
    double last_late_download_ms = 0.0;
    double last_device_ready_ms = 0.0;
    auto run_split = [&](std::vector<float> & output) {
        evict_host_cache();
        const auto started = steady_clock::now();
        if (!compute(phase_a_graph)) {
            return -1.0;
        }
        ggml_backend_tensor_get(
                ffn_input, activation_host.data(), 0,
                activation_host.size() * sizeof(float));
        const auto phase_a_finished = steady_clock::now();

        double phone_ms = 0.0;
        const uint32_t this_request = request_id++;
        const auto host_phase_b_started = steady_clock::now();
        enum ggml_status launch_status = GGML_STATUS_FAILED;
        bool phone_ok = false;
        if (host_is_cpu) {
            std::thread phone_thread([&]() {
                phone_ok = phone_ffn(
                        usb, this_request, weight_type,
                        (uint32_t) k, (uint32_t) n_ff,
                        (uint32_t) gpu_columns, (uint32_t) phone_columns,
                        phone_weight_hash, phone_f16_io,
                        activation_host, phone_output,
                        phone_output_f16, phone_ms);
            });
            launch_status =
                    ggml_backend_graph_compute(backend, split_phase_b_graph);
            phone_thread.join();
        } else {
            launch_status = ggml_backend_graph_compute_async(
                    backend, split_phase_b_graph);
            phone_ok = phone_ffn(
                    usb, this_request, weight_type,
                    (uint32_t) k, (uint32_t) n_ff,
                    (uint32_t) gpu_columns, (uint32_t) phone_columns,
                    phone_weight_hash, phone_f16_io,
                    activation_host, phone_output,
                    phone_output_f16, phone_ms);
            ggml_backend_synchronize(backend);
        }
        const auto host_phase_b_finished = steady_clock::now();
        if (launch_status != GGML_STATUS_SUCCESS || !phone_ok) {
            return -1.0;
        }
        const auto late_merge_started = steady_clock::now();
        const auto upload_started = steady_clock::now();
        ggml_backend_tensor_set_async(
                backend, returned_phone_residual,
                phone_output.data(), 0,
                phone_output.size() * sizeof(float));
        const auto upload_finished = steady_clock::now();
        const auto compute_started = steady_clock::now();
        if (!compute(late_merge_graph)) {
            return -1.0;
        }
        const auto compute_finished = steady_clock::now();
        const auto download_started = steady_clock::now();
        ggml_backend_tensor_get(
                combined_split_output, output.data(), 0,
                output.size() * sizeof(float));
        const auto late_merge_finished = steady_clock::now();
        last_phone_ms = phone_ms;
        last_phase_a_ms = std::chrono::duration<double, std::milli>(
                phase_a_finished - started).count();
        last_overlap_span_ms = std::chrono::duration<double, std::milli>(
                host_phase_b_finished - host_phase_b_started).count();
        last_late_merge_ms = std::chrono::duration<double, std::milli>(
                late_merge_finished - late_merge_started).count();
        last_late_upload_ms = std::chrono::duration<double, std::milli>(
                upload_finished - upload_started).count();
        last_late_compute_ms = std::chrono::duration<double, std::milli>(
                compute_finished - compute_started).count();
        last_late_download_ms = std::chrono::duration<double, std::milli>(
                late_merge_finished - download_started).count();
        last_device_ready_ms = std::chrono::duration<double, std::milli>(
                compute_finished - started).count();
        return elapsed_ms(started);
    };

    auto run_local_split = [&](std::vector<float> & output) {
        if (!compute(phase_a_graph) ||
            !compute(split_phase_b_graph) ||
            !compute(oracle_phone_graph)) {
            return false;
        }
        ggml_backend_tensor_get(
                gpu_output, output.data(), 0, output.size() * sizeof(float));
        ggml_backend_tensor_get(
                oracle_output, oracle_phone_output.data(), 0,
                oracle_phone_output.size() * sizeof(float));
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] += oracle_phone_output[i];
        }
        return true;
    };

    auto run_recursive_split = [&](std::vector<float> & output) {
        if (!compute(recursive_split_graph) ||
            !compute(recursive_oracle_graph)) {
            return false;
        }
        ggml_backend_tensor_get(
                gpu_output, output.data(), 0, output.size() * sizeof(float));
        ggml_backend_tensor_get(
                oracle_output, oracle_phone_output.data(), 0,
                oracle_phone_output.size() * sizeof(float));
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] += oracle_phone_output[i];
        }
        return true;
    };

    auto cleanup = [&]() {
        if (usb != nullptr) {
            libusb_release_interface(usb, 0);
            libusb_close(usb);
            libusb_exit(usb_context);
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        ggml_quantize_free();
    };

    if (mode == "monolithic" || mode == "two_phase") {
        std::vector<double> times;
        times.reserve((size_t) iterations);
        for (int i = 0; i < 10; ++i) {
            const double duration = mode == "monolithic"
                    ? run_monolithic(monolithic_output)
                    : run_two_phase(two_phase_output);
            if (duration < 0.0) {
                cleanup();
                return 3;
            }
        }
        printf("ENERGY_WINDOW_START unix_ns=%lld mode=%s n=%d\n",
                (long long) unix_time_ns(), mode.c_str(), iterations);
        fflush(stdout);
        for (int i = 0; i < iterations; ++i) {
            set_input_sample((uint32_t) i);
            const double duration = mode == "monolithic"
                    ? run_monolithic(monolithic_output)
                    : run_two_phase(two_phase_output);
            if (duration < 0.0) {
                cleanup();
                return 3;
            }
            times.push_back(duration);
        }
        printf("ENERGY_WINDOW_END unix_ns=%lld mode=%s n=%d\n",
                (long long) unix_time_ns(), mode.c_str(), iterations);
        printf("RESULT host_backend=%s cache=%s cache_flush_mib=%zu "
               "control=%s_%s n=%d median_ms=%.6f p90_ms=%.6f\n",
                host_is_cpu ? "CPU" : "GPU",
                cache_flush.empty() ? "warm" : "cold", cache_flush_mib,
                mode.c_str(), host_tag, iterations, percentile(times, 0.5),
                percentile(times, 0.9));
        cleanup();
        return 0;
    }

    for (int i = 0; i < 5; ++i) {
        if (run_monolithic(monolithic_output) < 0.0 ||
            run_two_phase(two_phase_output) < 0.0 ||
            (phone_columns > 0 && run_split(split_output) < 0.0)) {
            return 3;
        }
    }

    run_monolithic(monolithic_output);
    run_two_phase(two_phase_output);
    const error_metrics two_phase_metrics =
            compare(monolithic_output, two_phase_output);
    fprintf(stderr,
            "CORRECTNESS control=two_phase_%s rel_l2=%.9g max_abs=%.9g "
            "argmax=%zu/%zu non_finite=%zu\n",
            host_tag, two_phase_metrics.relative_l2,
            two_phase_metrics.max_absolute,
            two_phase_metrics.reference_argmax, two_phase_metrics.candidate_argmax,
            two_phase_metrics.non_finite);

    error_metrics split_metrics = {};
    if (phone_columns > 0) {
        if (!run_recursive_split(recursive_split_output)) {
            return 3;
        }
        const error_metrics recursive_split_metrics =
                compare(monolithic_output, recursive_split_output);
        fprintf(stderr,
                "CORRECTNESS control=recursive_%s_partition rel_l2=%.9g "
                "max_abs=%.9g argmax=%zu/%zu non_finite=%zu\n",
                host_tag, recursive_split_metrics.relative_l2,
                recursive_split_metrics.max_absolute,
                recursive_split_metrics.reference_argmax,
                recursive_split_metrics.candidate_argmax,
                recursive_split_metrics.non_finite);
        if (!run_local_split(local_split_output)) {
            return 3;
        }
        const error_metrics local_split_metrics =
                compare(monolithic_output, local_split_output);
        fprintf(stderr,
                "CORRECTNESS control=local_%s_partition rel_l2=%.9g "
                "max_abs=%.9g argmax=%zu/%zu non_finite=%zu\n",
                host_tag, local_split_metrics.relative_l2,
                local_split_metrics.max_absolute,
                local_split_metrics.reference_argmax,
                local_split_metrics.candidate_argmax,
                local_split_metrics.non_finite);
        if (run_split(split_output) < 0.0) {
            return 3;
        }
        const error_metrics phone_partial_metrics =
                compare(oracle_phone_output, phone_output);
        fprintf(stderr,
                "CORRECTNESS component=phone_vs_%s_slice rel_l2=%.9g "
                "max_abs=%.9g argmax=%zu/%zu non_finite=%zu\n",
                host_tag, phone_partial_metrics.relative_l2,
                phone_partial_metrics.max_absolute,
                phone_partial_metrics.reference_argmax,
                phone_partial_metrics.candidate_argmax,
                phone_partial_metrics.non_finite);
        ggml_fp32_to_fp16_row(
                activation_host.data(), rounded_activation_f16.data(),
                rounded_activation_f16.size());
        ggml_fp16_to_fp32_row(
                rounded_activation_f16.data(), rounded_activation.data(),
                rounded_activation.size());
        ggml_backend_tensor_set(
                ffn_input, rounded_activation.data(), 0,
                rounded_activation.size() * sizeof(float));
        if (!compute(oracle_phone_graph)) {
            return 3;
        }
        ggml_backend_tensor_get(
                oracle_output, rounded_phone_oracle_output.data(), 0,
                rounded_phone_oracle_output.size() * sizeof(float));
        const error_metrics rounded_phone_metrics =
                compare(rounded_phone_oracle_output, phone_output);
        double activation_l2 = 0.0;
        double activation_max_abs = 0.0;
        for (float value : activation_host) {
            activation_l2 += (double) value * (double) value;
            activation_max_abs = std::max(
                    activation_max_abs, std::abs((double) value));
        }
        fprintf(stderr,
                "CORRECTNESS diagnostic=phone_vs_f16_rounded_%s_slice "
                "rel_l2=%.9g max_abs=%.9g argmax=%zu/%zu "
                "non_finite=%zu activation_l2=%.9g "
                "activation_max_abs=%.9g\n",
                host_tag, rounded_phone_metrics.relative_l2,
                rounded_phone_metrics.max_absolute,
                rounded_phone_metrics.reference_argmax,
                rounded_phone_metrics.candidate_argmax,
                rounded_phone_metrics.non_finite,
                std::sqrt(activation_l2), activation_max_abs);
        split_metrics = compare(monolithic_output, split_output);
        fprintf(stderr,
                "CORRECTNESS treatment=%s_phone rel_l2=%.9g max_abs=%.9g "
                "argmax=%zu/%zu non_finite=%zu\n",
                host_tag, split_metrics.relative_l2,
                split_metrics.max_absolute,
                split_metrics.reference_argmax, split_metrics.candidate_argmax,
                split_metrics.non_finite);
        const double max_relative_l2 = getenv("S41_MAX_REL_L2") != nullptr
                ? atof(getenv("S41_MAX_REL_L2")) : 5e-3;
        const double max_phone_relative_l2 =
                getenv("S41_MAX_PHONE_REL_L2") != nullptr
                ? atof(getenv("S41_MAX_PHONE_REL_L2")) : 3e-2;
        if (split_metrics.non_finite != 0 ||
            split_metrics.relative_l2 > max_relative_l2 ||
            split_metrics.reference_argmax != split_metrics.candidate_argmax ||
            phone_partial_metrics.non_finite != 0 ||
            phone_partial_metrics.relative_l2 > max_phone_relative_l2 ||
            phone_partial_metrics.reference_argmax !=
                    phone_partial_metrics.candidate_argmax ||
            local_split_metrics.non_finite != 0 ||
            local_split_metrics.relative_l2 > max_relative_l2 ||
            local_split_metrics.reference_argmax !=
                    local_split_metrics.candidate_argmax ||
            recursive_split_metrics.non_finite != 0 ||
            recursive_split_metrics.relative_l2 > max_relative_l2 ||
            recursive_split_metrics.reference_argmax !=
                    recursive_split_metrics.candidate_argmax) {
            fprintf(stderr, "[causal-host] correctness gate failed\n");
            return 4;
        }
    }

    if (mode == "split") {
        std::vector<double> split_times;
        std::vector<double> phone_times;
        std::vector<double> phase_a_times;
        std::vector<double> overlap_span_times;
        std::vector<double> late_merge_times;
        std::vector<double> late_upload_times;
        std::vector<double> late_compute_times;
        std::vector<double> late_download_times;
        std::vector<double> device_ready_times;
        split_times.reserve((size_t) iterations);
        phone_times.reserve((size_t) iterations);
        phase_a_times.reserve((size_t) iterations);
        overlap_span_times.reserve((size_t) iterations);
        late_merge_times.reserve((size_t) iterations);
        late_upload_times.reserve((size_t) iterations);
        late_compute_times.reserve((size_t) iterations);
        late_download_times.reserve((size_t) iterations);
        device_ready_times.reserve((size_t) iterations);
        printf("ENERGY_WINDOW_START unix_ns=%lld mode=%s n=%d\n",
                (long long) unix_time_ns(), mode.c_str(), iterations);
        fflush(stdout);
        for (int i = 0; i < iterations; ++i) {
            set_input_sample((uint32_t) i);
            const double split_ms = run_split(split_output);
            if (split_ms < 0.0) {
                cleanup();
                return 3;
            }
            split_times.push_back(split_ms);
            phone_times.push_back(last_phone_ms);
            phase_a_times.push_back(last_phase_a_ms);
            overlap_span_times.push_back(last_overlap_span_ms);
            late_merge_times.push_back(last_late_merge_ms);
            late_upload_times.push_back(last_late_upload_ms);
            late_compute_times.push_back(last_late_compute_ms);
            late_download_times.push_back(last_late_download_ms);
            device_ready_times.push_back(last_device_ready_ms);
        }
        printf("ENERGY_WINDOW_END unix_ns=%lld mode=%s n=%d\n",
                (long long) unix_time_ns(), mode.c_str(), iterations);
        printf("RESULT host_backend=%s cache=%s cache_flush_mib=%zu "
               "treatment=%s_phone_isolated n=%d median_ms=%.6f "
               "p90_ms=%.6f phase_a_median_ms=%.6f "
               "overlap_span_median_ms=%.6f phone_median_ms=%.6f "
               "late_merge_median_ms=%.6f device_ready_median_ms=%.6f "
               "result_residency=%s\n",
                host_is_cpu ? "CPU" : "GPU",
                cache_flush.empty() ? "warm" : "cold", cache_flush_mib,
                host_tag, iterations, percentile(split_times, 0.5),
                percentile(split_times, 0.9),
                percentile(phase_a_times, 0.5),
                percentile(overlap_span_times, 0.5),
                percentile(phone_times, 0.5),
                percentile(late_merge_times, 0.5),
                percentile(device_ready_times, 0.5), host_tag);
        printf("LATE_BREAKDOWN upload_median_ms=%.6f "
               "compute_median_ms=%.6f download_median_ms=%.6f\n",
                percentile(late_upload_times, 0.5),
                percentile(late_compute_times, 0.5),
                percentile(late_download_times, 0.5));
        cleanup();
        return 0;
    }

    std::vector<double> monolithic_times;
    std::vector<double> two_phase_times;
    std::vector<double> split_times;
    std::vector<double> phone_times;
    std::vector<double> phase_a_times;
    std::vector<double> overlap_span_times;
    std::vector<double> late_merge_times;
    std::vector<double> device_ready_times;
    monolithic_times.reserve((size_t) iterations);
    two_phase_times.reserve((size_t) iterations);
    split_times.reserve((size_t) iterations);
    phone_times.reserve((size_t) iterations);
    phase_a_times.reserve((size_t) iterations);
    overlap_span_times.reserve((size_t) iterations);
    late_merge_times.reserve((size_t) iterations);
    device_ready_times.reserve((size_t) iterations);
    std::vector<uint32_t> monolithic_hashes;
    std::vector<uint32_t> split_hashes;
    double max_dynamic_relative_l2 = 0.0;
    size_t dynamic_argmax_mismatches = 0;
    const bool require_dynamic_argmax =
            getenv("S41_REQUIRE_DYNAMIC_ARGMAX") == nullptr ||
            atoi(getenv("S41_REQUIRE_DYNAMIC_ARGMAX")) != 0;

    for (int i = 0; i < iterations; ++i) {
        set_input_sample((uint32_t) i);
        double monolithic_ms;
        double two_phase_ms;
        double split_ms = 0.0;
        if ((i & 1) == 0) {
            monolithic_ms = run_monolithic(monolithic_output);
            two_phase_ms = run_two_phase(two_phase_output);
            if (phone_columns > 0) {
                split_ms = run_split(split_output);
            }
        } else {
            if (phone_columns > 0) {
                split_ms = run_split(split_output);
            }
            two_phase_ms = run_two_phase(two_phase_output);
            monolithic_ms = run_monolithic(monolithic_output);
        }
        if (monolithic_ms < 0.0 || two_phase_ms < 0.0 ||
            (phone_columns > 0 && split_ms < 0.0)) {
            return 3;
        }
        monolithic_times.push_back(monolithic_ms);
        two_phase_times.push_back(two_phase_ms);
        monolithic_hashes.push_back(s41_fast_hash_bytes(
                monolithic_output.data(),
                monolithic_output.size() * sizeof(float)));
        if (phone_columns > 0) {
            const error_metrics dynamic_metrics =
                    compare(monolithic_output, split_output);
            max_dynamic_relative_l2 = std::max(
                    max_dynamic_relative_l2,
                    dynamic_metrics.relative_l2);
            dynamic_argmax_mismatches +=
                    dynamic_metrics.reference_argmax ==
                            dynamic_metrics.candidate_argmax ? 0 : 1;
            split_hashes.push_back(s41_fast_hash_bytes(
                    split_output.data(),
                    split_output.size() * sizeof(float)));
            const double max_relative_l2 =
                    getenv("S41_MAX_REL_L2") != nullptr
                    ? atof(getenv("S41_MAX_REL_L2")) : 5e-3;
            if (dynamic_metrics.non_finite != 0 ||
                dynamic_metrics.relative_l2 > max_relative_l2 ||
                (require_dynamic_argmax &&
                 dynamic_metrics.reference_argmax !=
                         dynamic_metrics.candidate_argmax)) {
                fprintf(stderr,
                        "[causal-host] dynamic correctness gate failed "
                        "iteration=%d rel_l2=%.9g argmax=%zu/%zu "
                        "non_finite=%zu\n",
                        i, dynamic_metrics.relative_l2,
                        dynamic_metrics.reference_argmax,
                        dynamic_metrics.candidate_argmax,
                        dynamic_metrics.non_finite);
                cleanup();
                return 4;
            }
            split_times.push_back(split_ms);
            phone_times.push_back(last_phone_ms);
            phase_a_times.push_back(last_phase_a_ms);
            overlap_span_times.push_back(last_overlap_span_ms);
            late_merge_times.push_back(last_late_merge_ms);
            device_ready_times.push_back(last_device_ready_ms);
        }
    }

    const double monolithic_median = percentile(monolithic_times, 0.5);
    const double two_phase_median = percentile(two_phase_times, 0.5);
    std::sort(monolithic_hashes.begin(), monolithic_hashes.end());
    const size_t distinct_monolithic_hashes = (size_t) std::distance(
            monolithic_hashes.begin(),
            std::unique(monolithic_hashes.begin(),
                        monolithic_hashes.end()));
    printf("RESULT host_backend=%s cache=%s cache_flush_mib=%zu "
           "control=monolithic_%s n=%d median_ms=%.6f p90_ms=%.6f\n",
            host_is_cpu ? "CPU" : "GPU",
            cache_flush.empty() ? "warm" : "cold", cache_flush_mib,
            host_tag, iterations, monolithic_median,
            percentile(monolithic_times, 0.9));
    printf("RESULT host_backend=%s cache=%s cache_flush_mib=%zu "
           "control=two_phase_%s n=%d median_ms=%.6f p90_ms=%.6f "
           "ratio_vs_monolithic=%.6f\n",
            host_is_cpu ? "CPU" : "GPU",
            cache_flush.empty() ? "warm" : "cold", cache_flush_mib,
            host_tag, iterations, two_phase_median,
            percentile(two_phase_times, 0.9),
            two_phase_median / monolithic_median);
    if (phone_columns > 0) {
        std::sort(split_hashes.begin(), split_hashes.end());
        const size_t distinct_split_hashes = (size_t) std::distance(
                split_hashes.begin(),
                std::unique(split_hashes.begin(), split_hashes.end()));
        const double split_median = percentile(split_times, 0.5);
        printf("RESULT host_backend=%s cache=%s cache_flush_mib=%zu "
               "treatment=%s_phone n=%d median_ms=%.6f p90_ms=%.6f "
               "phase_a_median_ms=%.6f overlap_span_median_ms=%.6f "
               "phone_median_ms=%.6f late_merge_median_ms=%.6f "
               "device_ready_median_ms=%.6f result_residency=%s "
               "speedup_vs_monolithic=%.6f "
               "speedup_vs_two_phase=%.6f\n",
                host_is_cpu ? "CPU" : "GPU",
                cache_flush.empty() ? "warm" : "cold", cache_flush_mib,
                host_tag, iterations, split_median,
                percentile(split_times, 0.9),
                percentile(phase_a_times, 0.5),
                percentile(overlap_span_times, 0.5),
                percentile(phone_times, 0.5),
                percentile(late_merge_times, 0.5),
                percentile(device_ready_times, 0.5), host_tag,
                monolithic_median / split_median,
                two_phase_median / split_median);
        printf("DYNAMIC_CORRECTNESS result=%s max_rel_l2=%.9g "
               "require_argmax=%s "
               "argmax_mismatches=%zu/%d "
               "distinct_monolithic_hashes=%zu/%d "
               "distinct_split_hashes=%zu/%d\n",
                dynamic_argmax_mismatches == 0
                        ? "PASS" : "RELATIVE_GATE_ONLY",
                max_dynamic_relative_l2,
                require_dynamic_argmax ? "yes" : "no",
                dynamic_argmax_mismatches,
                iterations, distinct_monolithic_hashes, iterations,
                distinct_split_hashes, iterations);
    }

    cleanup();
    return 0;
}
