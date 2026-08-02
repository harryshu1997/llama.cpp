// CUDA/HTP Qwen GQA-group attention diagnostic.
//
// This bounded probe writes the current K/V projections into the last cache
// slot and checks whether complete GQA groups can run concurrently.
//
// usage:
//   causal_attention_host <n_kv> <iters> <phone_groups> [weight_type]
//       [mode]

#include "causal_attention_protocol.h"
#include "causal_quantized_weights.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/types.h>
#include <vector>

extern "C" {
struct libusb_context;
struct libusb_device;
struct libusb_device_handle;

struct libusb_device_descriptor {
    uint8_t bLength;
    uint8_t bDescriptorType;
    uint16_t bcdUSB;
    uint8_t bDeviceClass;
    uint8_t bDeviceSubClass;
    uint8_t bDeviceProtocol;
    uint8_t bMaxPacketSize0;
    uint16_t idVendor;
    uint16_t idProduct;
    uint16_t bcdDevice;
    uint8_t iManufacturer;
    uint8_t iProduct;
    uint8_t iSerialNumber;
    uint8_t bNumConfigurations;
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
        libusb_device_handle *, unsigned char, unsigned char *, int, int *,
        unsigned int);
const char * libusb_error_name(int);
}

using steady_clock = std::chrono::steady_clock;

static constexpr int64_t S41_QWEN_K = 5120;
static constexpr int64_t S41_QWEN_HEAD_DIM = 128;
static constexpr int64_t S41_QWEN_HEADS = 40;
static constexpr int64_t S41_QWEN_KV_HEADS = 8;
static constexpr int64_t S41_QWEN_GQA = 5;

static double duration_ms(
        steady_clock::time_point started, steady_clock::time_point finished) {
    return std::chrono::duration<double, std::milli>(
            finished - started).count();
}

static double elapsed_ms(steady_clock::time_point started) {
    return duration_ms(started, steady_clock::now());
}

static double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = std::min(
            values.size() - 1,
            (size_t) std::floor(fraction * (values.size() - 1)));
    return values[index];
}

static void print_metrics(
        const char * name, const std::vector<double> & values) {
    printf("%s_median_ms=%.6f %s_p90_ms=%.6f ",
            name, percentile(values, 0.5),
            name, percentile(values, 0.9));
}

static bool parse_type(const char * name, ggml_type & result) {
    for (int index = 0; index < GGML_TYPE_COUNT; ++index) {
        const ggml_type type = (ggml_type) index;
        const char * type_name = ggml_type_name(type);
        if (type_name != nullptr && name == std::string(type_name)) {
            result = type;
            return true;
        }
    }
    return false;
}

static bool is_accessory_pid(uint16_t pid) {
    return pid == 0x2d00 || pid == 0x2d01 ||
            pid == 0x2d04 || pid == 0x2d05;
}

static libusb_device_handle * open_accessory(
        libusb_context * context, int wanted_bus, int wanted_address) {
    libusb_device ** devices = nullptr;
    const ssize_t count = libusb_get_device_list(context, &devices);
    if (count < 0) {
        fprintf(stderr, "[attention-host] libusb list failed: %zd\n", count);
        return nullptr;
    }

    libusb_device * selected = nullptr;
    int candidates = 0;
    int selected_bus = -1;
    int selected_address = -1;
    for (ssize_t index = 0; index < count; ++index) {
        libusb_device_descriptor descriptor = {};
        if (libusb_get_device_descriptor(devices[index], &descriptor) != 0 ||
            descriptor.idVendor != 0x18d1 ||
            !is_accessory_pid(descriptor.idProduct)) {
            continue;
        }
        const int bus = libusb_get_bus_number(devices[index]);
        const int address = libusb_get_device_address(devices[index]);
        if ((wanted_bus >= 0 && bus != wanted_bus) ||
            (wanted_address >= 0 && address != wanted_address)) {
            continue;
        }
        ++candidates;
        selected = devices[index];
        selected_bus = bus;
        selected_address = address;
    }

    libusb_device_handle * handle = nullptr;
    if (candidates == 1) {
        const int status = libusb_open(selected, &handle);
        if (status != 0) {
            fprintf(stderr, "[attention-host] accessory open failed: %s\n",
                    libusb_error_name(status));
            handle = nullptr;
        } else {
            fprintf(stderr, "[attention-host] accessory bus=%d address=%d\n",
                    selected_bus, selected_address);
        }
    } else if (candidates == 0) {
        fprintf(stderr, "[attention-host] no matching accessory found\n");
    } else {
        fprintf(stderr,
                "[attention-host] %d accessories match; set S41_AOA_BUS "
                "and S41_AOA_ADDRESS\n",
                candidates);
    }
    libusb_free_device_list(devices, 1);
    return handle;
}

static bool usb_transfer(
        libusb_device_handle * handle,
        uint8_t endpoint,
        void * data,
        size_t size) {
    const char * timeout_text = getenv("S41_AOA_TIMEOUT_MS");
    const long timeout_value =
            timeout_text != nullptr ? atol(timeout_text) : 4000;
    if (timeout_value <= 0 || timeout_value > 60000) {
        fprintf(stderr, "[attention-host] invalid AOA timeout\n");
        return false;
    }
    uint8_t * pointer = (uint8_t *) data;
    size_t completed = 0;
    while (completed < size) {
        const size_t remaining = size - completed;
        int transferred = 0;
        const int status = libusb_bulk_transfer(
                handle, endpoint, pointer + completed, (int) remaining,
                &transferred, (unsigned int) timeout_value);
        if (status != 0) {
            fprintf(stderr,
                    "[attention-host] USB endpoint 0x%02x failed: %s\n",
                    endpoint, libusb_error_name(status));
            return false;
        }
        if (transferred <= 0) {
            fprintf(stderr,
                    "[attention-host] USB endpoint 0x%02x made no progress\n",
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
    const uint32_t columns = (uint32_t) tensor->ne[0];
    const uint32_t rows = (uint32_t) tensor->ne[1];
    std::vector<uint8_t> quantized(ggml_nbytes(tensor));
    if (!s41_fill_quantized_weight(
                type, matrix, row_offset, column_offset, columns, rows,
                quantized.data(), quantized.size())) {
        return false;
    }
    if (weight_hash != nullptr) {
        *weight_hash = s41_weight_hash_update(
                *weight_hash, matrix, row_offset, column_offset,
                columns, rows, quantized.data(), quantized.size());
    }
    ggml_backend_tensor_set(tensor, quantized.data(), 0, quantized.size());
    return true;
}

static void initialize_cache(
        ggml_tensor * tensor,
        uint32_t matrix,
        uint32_t group_offset,
        uint32_t group_count,
        uint32_t n_kv) {
    const size_t elements = ggml_nelements(tensor);
    std::vector<float> data_f32(elements);
    for (uint32_t group = 0; group < group_count; ++group) {
        for (uint32_t token = 0; token < n_kv; ++token) {
            for (uint32_t dimension = 0;
                 dimension < S41_QWEN_HEAD_DIM; ++dimension) {
                const size_t index =
                        ((size_t) group * n_kv + token) *
                                S41_QWEN_HEAD_DIM + dimension;
                data_f32[index] = s41_kv_value(
                        matrix, group_offset + group, token, dimension);
            }
        }
    }
    std::vector<ggml_fp16_t> data_f16(elements);
    ggml_fp32_to_fp16_row(data_f32.data(), data_f16.data(), elements);
    ggml_backend_tensor_set(
            tensor, data_f16.data(), 0,
            data_f16.size() * sizeof(ggml_fp16_t));
}

struct attention_path {
    int64_t group_offset;
    int64_t group_count;
    int64_t q_width;
    int64_t kv_width;
    ggml_tensor * wq;
    ggml_tensor * wk;
    ggml_tensor * wv;
    ggml_tensor * wo;
    ggml_tensor * key_cache;
    ggml_tensor * value_cache;
    ggml_tensor * q_output;
    ggml_tensor * k_output;
    ggml_tensor * v_output;
    ggml_tensor * key_store;
    ggml_tensor * value_store;
    ggml_tensor * output;
    ggml_cgraph * graph;
};

static attention_path build_path(
        ggml_context * context,
        ggml_type weight_type,
        ggml_tensor * input,
        int64_t n_kv,
        int64_t group_offset,
        int64_t group_count) {
    attention_path path = {};
    path.group_offset = group_offset;
    path.group_count = group_count;
    const int64_t q_heads = group_count * S41_QWEN_GQA;
    path.q_width = q_heads * S41_QWEN_HEAD_DIM;
    path.kv_width = group_count * S41_QWEN_HEAD_DIM;

    path.wq = ggml_new_tensor_2d(
            context, weight_type, S41_QWEN_K, path.q_width);
    path.wk = ggml_new_tensor_2d(
            context, weight_type, S41_QWEN_K, path.kv_width);
    path.wv = ggml_new_tensor_2d(
            context, weight_type, S41_QWEN_K, path.kv_width);
    path.wo = ggml_new_tensor_2d(
            context, weight_type, path.q_width, S41_QWEN_K);
    path.key_cache = ggml_new_tensor_4d(
            context, GGML_TYPE_F16,
            S41_QWEN_HEAD_DIM, n_kv, group_count, 1);
    path.value_cache = ggml_new_tensor_4d(
            context, GGML_TYPE_F16,
            S41_QWEN_HEAD_DIM, n_kv, group_count, 1);

    path.q_output = ggml_mul_mat(context, path.wq, input);
    path.k_output = ggml_mul_mat(context, path.wk, input);
    path.v_output = ggml_mul_mat(context, path.wv, input);
    ggml_tensor * k_current = ggml_reshape_3d(
            context, path.k_output,
            S41_QWEN_HEAD_DIM, 1, group_count);
    ggml_tensor * v_current = ggml_reshape_3d(
            context, path.v_output,
            S41_QWEN_HEAD_DIM, 1, group_count);
    ggml_tensor * key_slot = ggml_view_3d(
            context, path.key_cache,
            S41_QWEN_HEAD_DIM, 1, group_count,
            path.key_cache->nb[1], path.key_cache->nb[2],
            (size_t) (n_kv - 1) * path.key_cache->nb[1]);
    ggml_tensor * value_slot = ggml_view_3d(
            context, path.value_cache,
            S41_QWEN_HEAD_DIM, 1, group_count,
            path.value_cache->nb[1], path.value_cache->nb[2],
            (size_t) (n_kv - 1) * path.value_cache->nb[1]);
    path.key_store = ggml_cpy(context, k_current, key_slot);
    path.value_store = ggml_cpy(context, v_current, value_slot);
    ggml_tensor * q_heads_tensor = ggml_permute(
            context,
            ggml_reshape_4d(
                    context, path.q_output,
                    S41_QWEN_HEAD_DIM, q_heads, 1, 1),
            0, 2, 1, 3);
    ggml_tensor * attended = ggml_flash_attn_ext(
            context, q_heads_tensor,
            path.key_cache, path.value_cache, nullptr,
            1.0f / std::sqrt((float) S41_QWEN_HEAD_DIM), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
    ggml_tensor * merged =
            ggml_reshape_2d(context, attended, path.q_width, 1);
    path.output = ggml_mul_mat(context, path.wo, merged);
    path.graph = nullptr;
    return path;
}

static bool initialize_path(
        attention_path & path,
        ggml_type weight_type,
        int64_t n_kv,
        uint64_t * weight_hash) {
    const uint32_t q_row_offset = (uint32_t) (
            path.group_offset * S41_QWEN_GQA * S41_QWEN_HEAD_DIM);
    const uint32_t kv_row_offset = (uint32_t) (
            path.group_offset * S41_QWEN_HEAD_DIM);
    if (!initialize_weight(
                path.wq, weight_type, S41_WEIGHT_Q,
                q_row_offset, 0, weight_hash) ||
        !initialize_weight(
                path.wk, weight_type, S41_WEIGHT_K,
                kv_row_offset, 0, weight_hash) ||
        !initialize_weight(
                path.wv, weight_type, S41_WEIGHT_V,
                kv_row_offset, 0, weight_hash) ||
        !initialize_weight(
                path.wo, weight_type, S41_WEIGHT_O,
                0, q_row_offset, weight_hash)) {
        return false;
    }
    initialize_cache(
            path.key_cache, S41_WEIGHT_K,
            (uint32_t) path.group_offset,
            (uint32_t) path.group_count, (uint32_t) n_kv);
    initialize_cache(
            path.value_cache, S41_WEIGHT_V,
            (uint32_t) path.group_offset,
            (uint32_t) path.group_count, (uint32_t) n_kv);
    return true;
}

struct phone_metrics {
    double total_ms = 0.0;
    double prepare_ms = 0.0;
    double usb_out_ms = 0.0;
    double usb_in_ms = 0.0;
    double validate_ms = 0.0;
    double set_ms = 0.0;
    double compute_ms = 0.0;
    double get_ms = 0.0;
};

class phone_client {
public:
    phone_client(
            libusb_device_handle * handle,
            ggml_type type,
            uint32_t n_kv,
            uint32_t group_offset,
            uint32_t group_count,
            uint64_t weight_hash)
        : handle_(handle),
          type_(type),
          n_kv_(n_kv),
          group_offset_(group_offset),
          group_count_(group_count),
          weight_hash_(weight_hash),
          input_f16_(S41_QWEN_K),
          output_f16_(S41_QWEN_K),
          request_data_(
                  sizeof(s41_attention_request) +
                  sizeof(ggml_fp16_t) * S41_QWEN_K),
          response_data_(
                  sizeof(s41_attention_response) +
                  sizeof(ggml_fp16_t) * S41_QWEN_K) {
    }

    bool run(
            uint32_t request_id,
            const std::vector<float> & input,
            std::vector<float> & output,
            phone_metrics & metrics) {
        const auto submitted = steady_clock::now();
        const auto prepare_started = steady_clock::now();
        ggml_fp32_to_fp16_row(
                input.data(), input_f16_.data(), S41_QWEN_K);
        const size_t input_bytes =
                input_f16_.size() * sizeof(ggml_fp16_t);
        s41_attention_request request = {};
        request.magic = S41_ATTN_REQUEST_MAGIC;
        request.version = S41_ATTN_PROTOCOL_VERSION;
        request.flags = S41_ATTN_FLAG_FAST_HASH |
                S41_ATTN_FLAG_F16_INPUT |
                S41_ATTN_FLAG_F16_OUTPUT |
                S41_ATTN_FLAG_LAST_SLOT_UPDATE;
        request.request_id = request_id;
        request.type = (uint32_t) type_;
        request.k = S41_QWEN_K;
        request.n_kv = n_kv_;
        request.n_heads = S41_QWEN_HEADS;
        request.n_kv_heads = S41_QWEN_KV_HEADS;
        request.group_offset = group_offset_;
        request.group_count = group_count_;
        request.input_bytes = (uint32_t) input_bytes;
        request.input_hash =
                s41_fast_hash_bytes(input_f16_.data(), input_bytes);
        request.weight_hash = weight_hash_;
        memcpy(request_data_.data(), &request, sizeof(request));
        memcpy(
                request_data_.data() + sizeof(request),
                input_f16_.data(), input_bytes);
        metrics.prepare_ms = elapsed_ms(prepare_started);

        auto started = steady_clock::now();
        if (!usb_transfer(
                    handle_, 0x01,
                    request_data_.data(), request_data_.size())) {
            return false;
        }
        metrics.usb_out_ms = elapsed_ms(started);
        started = steady_clock::now();
        if (!usb_transfer(
                    handle_, 0x81,
                    response_data_.data(), response_data_.size())) {
            return false;
        }
        metrics.usb_in_ms = elapsed_ms(started);

        const auto validate_started = steady_clock::now();
        s41_attention_response response = {};
        memcpy(&response, response_data_.data(), sizeof(response));
        const size_t output_bytes =
                output_f16_.size() * sizeof(ggml_fp16_t);
        const void * response_output =
                response_data_.data() + sizeof(response);
        if (response.magic != S41_ATTN_RESPONSE_MAGIC ||
            response.version != S41_ATTN_PROTOCOL_VERSION ||
            response.status != 0 ||
            response.request_id != request_id ||
            response.output_elements != S41_QWEN_K ||
            response.output_bytes != output_bytes ||
            response.weight_hash != weight_hash_ ||
            response.output_hash !=
                    s41_fast_hash_bytes(response_output, output_bytes)) {
            fprintf(stderr, "[attention-host] invalid phone response\n");
            return false;
        }
        memcpy(output_f16_.data(), response_output, output_bytes);
        ggml_fp16_to_fp32_row(
                output_f16_.data(), output.data(), S41_QWEN_K);
        metrics.validate_ms = elapsed_ms(validate_started);
        metrics.total_ms = elapsed_ms(submitted);
        metrics.set_ms = response.set_us / 1000.0;
        metrics.compute_ms = response.compute_us / 1000.0;
        metrics.get_ms = response.get_us / 1000.0;
        return true;
    }

private:
    libusb_device_handle * handle_;
    ggml_type type_;
    uint32_t n_kv_;
    uint32_t group_offset_;
    uint32_t group_count_;
    uint64_t weight_hash_;
    std::vector<ggml_fp16_t> input_f16_;
    std::vector<ggml_fp16_t> output_f16_;
    std::vector<uint8_t> request_data_;
    std::vector<uint8_t> response_data_;
};

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
    size_t reference_argmax = 0;
    size_t candidate_argmax = 0;
    size_t non_finite = 0;
    for (size_t index = 0; index < reference.size(); ++index) {
        if (!std::isfinite(reference[index]) ||
            !std::isfinite(candidate[index])) {
            ++non_finite;
            continue;
        }
        const double difference =
                (double) candidate[index] - reference[index];
        error_squared += difference * difference;
        reference_squared +=
                (double) reference[index] * reference[index];
        max_absolute = std::max(max_absolute, std::abs(difference));
        if (reference[index] > reference[reference_argmax]) {
            reference_argmax = index;
        }
        if (candidate[index] > candidate[candidate_argmax]) {
            candidate_argmax = index;
        }
    }
    return {
        reference_squared > 0.0
                ? std::sqrt(error_squared / reference_squared)
                : INFINITY,
        max_absolute,
        reference_argmax,
        candidate_argmax,
        non_finite,
    };
}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 6) {
        fprintf(stderr,
                "usage: %s <n_kv> <iters> <phone_groups> [weight_type] "
                "[mode]\n",
                argv[0]);
        return 2;
    }

    const int64_t n_kv = atoll(argv[1]);
    const int iterations = atoi(argv[2]);
    const int64_t phone_groups = atoll(argv[3]);
    ggml_type weight_type = GGML_TYPE_Q8_0;
    if (argc > 4 && !parse_type(argv[4], weight_type)) {
        fprintf(stderr, "[attention-host] unknown type '%s'\n", argv[4]);
        return 2;
    }
    const std::string mode = argc > 5 ? argv[5] : "steady_corun";
    if (mode != "steady_corun" && mode != "cuda_only") {
        fprintf(stderr, "[attention-host] unknown mode '%s'\n", mode.c_str());
        return 2;
    }
    const int64_t gpu_groups = S41_QWEN_KV_HEADS - phone_groups;
    if (n_kv <= 0 || iterations <= 0 ||
        phone_groups <= 0 || gpu_groups <= 0 ||
        weight_type != GGML_TYPE_Q8_0) {
        fprintf(stderr, "[attention-host] invalid configuration\n");
        return 2;
    }

    ggml_backend_t backend = nullptr;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (ggml_backend_dev_type(device) ==
            GGML_BACKEND_DEVICE_TYPE_GPU) {
            backend = ggml_backend_dev_init(device, nullptr);
            fprintf(stderr, "[attention-host] GPU=%s\n",
                    ggml_backend_dev_description(device));
            break;
        }
    }
    if (backend == nullptr) {
        fprintf(stderr, "[attention-host] no GPU backend found\n");
        return 1;
    }

    ggml_init_params parameters = {};
    parameters.mem_size =
            ggml_tensor_overhead() * 160 + ggml_graph_overhead() * 6;
    parameters.no_alloc = true;
    ggml_context * context = ggml_init(parameters);
    if (context == nullptr) {
        return 1;
    }
    ggml_tensor * input = ggml_new_tensor_2d(
            context, GGML_TYPE_F32, S41_QWEN_K, 1);
    attention_path full = build_path(
            context, weight_type, input, n_kv, 0, S41_QWEN_KV_HEADS);
    attention_path gpu = build_path(
            context, weight_type, input, n_kv, 0, gpu_groups);
    attention_path oracle = build_path(
            context, weight_type, input, n_kv,
            gpu_groups, phone_groups);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        fprintf(stderr, "[attention-host] allocation failed\n");
        return 1;
    }
    fprintf(stderr, "[attention-host] initializing weights and KV\n");
    uint64_t phone_weight_hash = S41_HASH64_OFFSET;
    if (!initialize_path(full, weight_type, n_kv, nullptr) ||
        !initialize_path(gpu, weight_type, n_kv, nullptr) ||
        !initialize_path(
                oracle, weight_type, n_kv, &phone_weight_hash)) {
        fprintf(stderr, "[attention-host] initialization failed\n");
        return 1;
    }

    std::vector<float> input_data(S41_QWEN_K);
    for (size_t index = 0; index < input_data.size(); ++index) {
        input_data[index] = s41_activation_value((uint32_t) index);
    }
    ggml_backend_tensor_set(
            input, input_data.data(), 0,
            input_data.size() * sizeof(float));

    auto finish_graph = [&](attention_path & path) {
        path.graph = ggml_new_graph_custom(context, 32, false);
        ggml_build_forward_expand(path.graph, path.key_store);
        ggml_build_forward_expand(path.graph, path.value_store);
        ggml_build_forward_expand(path.graph, path.output);
    };
    finish_graph(full);
    finish_graph(gpu);
    finish_graph(oracle);
    fprintf(stderr,
            "[attention-host] n_kv=%lld groups=%lld+%lld type=%s "
            "phone_weight_hash=%016llx nodes=%d/%d/%d "
            "cache_mode=last_slot_update\n",
            (long long) n_kv, (long long) gpu_groups,
            (long long) phone_groups, ggml_type_name(weight_type),
            (unsigned long long) phone_weight_hash,
            ggml_graph_n_nodes(full.graph),
            ggml_graph_n_nodes(gpu.graph),
            ggml_graph_n_nodes(oracle.graph));

    auto compute = [&](ggml_cgraph * graph) {
        const enum ggml_status status =
                ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[attention-host] compute failed: %d\n",
                    (int) status);
            return false;
        }
        return true;
    };
    auto run_path = [&](
            attention_path & path,
            std::vector<float> & output,
            double & total_ms) {
        const auto started = steady_clock::now();
        if (!compute(path.graph)) {
            return false;
        }
        ggml_backend_tensor_get(
                path.output, output.data(), 0,
                output.size() * sizeof(float));
        total_ms = elapsed_ms(started);
        return true;
    };

    std::vector<float> full_output(S41_QWEN_K);
    std::vector<float> gpu_output(S41_QWEN_K);
    std::vector<float> oracle_output(S41_QWEN_K);
    std::vector<float> phone_output(S41_QWEN_K);
    std::vector<float> split_output(S41_QWEN_K);
    double ignored_ms = 0.0;
    for (int index = 0; index < 20; ++index) {
        if (!run_path(full, full_output, ignored_ms) ||
            !run_path(gpu, gpu_output, ignored_ms) ||
            !run_path(oracle, oracle_output, ignored_ms)) {
            return 3;
        }
    }
    if (mode == "cuda_only") {
        std::vector<double> full_times;
        for (int index = 0; index < iterations; ++index) {
            double total_ms = 0.0;
            if (!run_path(full, full_output, total_ms)) {
                return 3;
            }
            full_times.push_back(total_ms);
        }
        printf("ATTENTION_CUDA_ONLY n_kv=%lld type=%s n=%d ",
                (long long) n_kv, ggml_type_name(weight_type), iterations);
        print_metrics("cuda_full", full_times);
        printf("sample=%.9g cache_mode=last_slot_update\n", full_output[0]);
        return 0;
    }

    libusb_context * usb_context = nullptr;
    if (libusb_init(&usb_context) != 0) {
        fprintf(stderr, "[attention-host] libusb initialization failed\n");
        return 1;
    }
    const char * bus_env = getenv("S41_AOA_BUS");
    const char * address_env = getenv("S41_AOA_ADDRESS");
    const int wanted_bus = bus_env != nullptr ? atoi(bus_env) : -1;
    const int wanted_address =
            address_env != nullptr ? atoi(address_env) : -1;
    libusb_device_handle * usb =
            open_accessory(usb_context, wanted_bus, wanted_address);
    if (usb == nullptr) {
        return 1;
    }
    libusb_detach_kernel_driver(usb, 0);
    const int claim_status = libusb_claim_interface(usb, 0);
    if (claim_status != 0) {
        fprintf(stderr, "[attention-host] claim failed: %s\n",
                libusb_error_name(claim_status));
        return 1;
    }

    phone_client client(
            usb, weight_type, (uint32_t) n_kv,
            (uint32_t) gpu_groups, (uint32_t) phone_groups,
            phone_weight_hash);
    uint32_t request_id = 1;
    phone_metrics correctness_phone = {};
    if (!client.run(
                request_id++, input_data, phone_output,
                correctness_phone)) {
        return 3;
    }
    for (size_t index = 0; index < split_output.size(); ++index) {
        split_output[index] = gpu_output[index] + phone_output[index];
    }
    std::vector<float> local_output(S41_QWEN_K);
    for (size_t index = 0; index < local_output.size(); ++index) {
        local_output[index] = gpu_output[index] + oracle_output[index];
    }
    const error_metrics local_error =
            compare(full_output, local_output);
    const error_metrics phone_error =
            compare(oracle_output, phone_output);
    const error_metrics split_error =
            compare(full_output, split_output);
    printf("ATTENTION_CORRECTNESS control=local_partition "
           "rel_l2=%.9g argmax=%zu/%zu non_finite=%zu\n",
            local_error.relative_l2,
            local_error.reference_argmax,
            local_error.candidate_argmax,
            local_error.non_finite);
    printf("ATTENTION_CORRECTNESS component=phone "
           "rel_l2=%.9g argmax=%zu/%zu non_finite=%zu\n",
            phone_error.relative_l2,
            phone_error.reference_argmax,
            phone_error.candidate_argmax,
            phone_error.non_finite);
    printf("ATTENTION_CORRECTNESS treatment=split "
           "rel_l2=%.9g argmax=%zu/%zu non_finite=%zu\n",
            split_error.relative_l2,
            split_error.reference_argmax,
            split_error.candidate_argmax,
            split_error.non_finite);
    const bool correctness_pass =
            local_error.non_finite == 0 &&
            local_error.relative_l2 <= 0.0001 &&
            local_error.reference_argmax ==
                    local_error.candidate_argmax &&
            phone_error.non_finite == 0 &&
            phone_error.relative_l2 <= 0.03 &&
            phone_error.reference_argmax ==
                    phone_error.candidate_argmax &&
            split_error.non_finite == 0 &&
            split_error.relative_l2 <= 0.005 &&
            split_error.reference_argmax ==
                    split_error.candidate_argmax;
    if (!correctness_pass) {
        fprintf(stderr, "[attention-host] correctness gate failed\n");
        return 4;
    }

    for (int index = 0; index < 20; ++index) {
        phone_metrics metrics = {};
        if (!client.run(
                    request_id++, input_data, phone_output, metrics)) {
            return 3;
        }
    }
    std::vector<double> full_times;
    std::vector<double> gpu_times;
    for (int index = 0; index < iterations; ++index) {
        double total_ms = 0.0;
        if (!run_path(full, full_output, total_ms)) {
            return 3;
        }
        full_times.push_back(total_ms);
    }
    for (int index = 0; index < iterations; ++index) {
        double total_ms = 0.0;
        if (!run_path(gpu, gpu_output, total_ms)) {
            return 3;
        }
        gpu_times.push_back(total_ms);
    }
    const double gpu_deadline_ms = percentile(gpu_times, 0.5);

    std::vector<double> split_times;
    std::vector<double> phone_times;
    std::vector<double> exposed_wait_times;
    std::vector<double> merge_times;
    std::vector<double> prepare_times;
    std::vector<double> usb_out_times;
    std::vector<double> usb_in_times;
    std::vector<double> htp_compute_times;
    std::vector<double> htp_get_times;
    size_t hidden_count = 0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto total_started = steady_clock::now();
        const enum ggml_status launch_status =
                ggml_backend_graph_compute_async(backend, gpu.graph);
        if (launch_status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[attention-host] async launch failed: %d\n",
                    (int) launch_status);
            return 3;
        }
        phone_metrics metrics = {};
        if (!client.run(
                    request_id++, input_data, phone_output, metrics)) {
            return 3;
        }
        ggml_backend_synchronize(backend);
        const auto merge_started = steady_clock::now();
        ggml_backend_tensor_get(
                gpu.output, gpu_output.data(), 0,
                gpu_output.size() * sizeof(float));
        for (size_t index = 0; index < split_output.size(); ++index) {
            split_output[index] =
                    gpu_output[index] + phone_output[index];
        }
        merge_times.push_back(elapsed_ms(merge_started));
        split_times.push_back(elapsed_ms(total_started));
        phone_times.push_back(metrics.total_ms);
        exposed_wait_times.push_back(
                std::max(0.0, metrics.total_ms - gpu_deadline_ms));
        prepare_times.push_back(metrics.prepare_ms);
        usb_out_times.push_back(metrics.usb_out_ms);
        usb_in_times.push_back(metrics.usb_in_ms);
        htp_compute_times.push_back(metrics.compute_ms);
        htp_get_times.push_back(metrics.get_ms);
        hidden_count += metrics.total_ms <= gpu_deadline_ms ? 1 : 0;
        const error_metrics treatment_error =
                compare(full_output, split_output);
        if (treatment_error.non_finite != 0 ||
            treatment_error.relative_l2 > 0.005 ||
            treatment_error.reference_argmax !=
                    treatment_error.candidate_argmax) {
            fprintf(stderr, "[attention-host] treatment output changed\n");
            return 4;
        }
    }

    printf("ATTENTION_RESULT n_kv=%lld type=%s groups=%lld+%lld n=%d ",
            (long long) n_kv, ggml_type_name(weight_type),
            (long long) gpu_groups, (long long) phone_groups, iterations);
    print_metrics("cuda_full", full_times);
    print_metrics("cuda_shard", gpu_times);
    print_metrics("phone_total", phone_times);
    print_metrics("split_total", split_times);
    print_metrics("exposed_wait", exposed_wait_times);
    print_metrics("merge", merge_times);
    printf("hidden=%zu/%d hidden_fraction=%.6f "
           "cache_mode=last_slot_update\n",
            hidden_count, iterations,
            (double) hidden_count / (double) iterations);
    printf("ATTENTION_PHONE_BREAKDOWN ");
    print_metrics("prepare", prepare_times);
    print_metrics("usb_out", usb_out_times);
    print_metrics("usb_in_compute", usb_in_times);
    print_metrics("htp_compute", htp_compute_times);
    print_metrics("htp_get", htp_get_times);
    printf("\n");
    const double full_median = percentile(full_times, 0.5);
    const double split_median = percentile(split_times, 0.5);
    printf("ATTENTION_VERDICT correctness=PASS split_beats_cuda=%s "
           "latency_change_percent=%.3f\n",
            split_median < full_median ? "PASS" : "FAIL",
            100.0 * (split_median / full_median - 1.0));

    libusb_release_interface(usb, 0);
    libusb_close(usb);
    libusb_exit(usb_context);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    ggml_quantize_free();
    return 0;
}
