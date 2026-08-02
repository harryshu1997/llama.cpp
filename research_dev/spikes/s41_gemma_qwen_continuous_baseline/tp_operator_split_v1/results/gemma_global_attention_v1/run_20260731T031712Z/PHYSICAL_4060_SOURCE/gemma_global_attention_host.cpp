// CUDA plus OP15 Gemma global-attention context-shard probe.
//
// The input is already-projected Q and the K/V cache is resident on each
// device. This isolates the context-dependent global-attention core.
//
// usage:
//   gemma_global_attention_host <total_kv> <phone_kv> <iters>
//       <cuda_only|corun>

#include "gemma_global_attention_protocol.h"

#include "ggml.h"
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

static double elapsed_ms(steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(
            steady_clock::now() - started).count();
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

static bool is_accessory_pid(uint16_t pid) {
    return pid == 0x2d00 || pid == 0x2d01 ||
            pid == 0x2d04 || pid == 0x2d05;
}

static libusb_device_handle * open_accessory(
        libusb_context * context, int wanted_bus, int wanted_address) {
    libusb_device ** devices = nullptr;
    const ssize_t count = libusb_get_device_list(context, &devices);
    if (count < 0) {
        fprintf(stderr,
                "[gemma-attention-host] libusb list failed: %zd\n", count);
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
            fprintf(stderr,
                    "[gemma-attention-host] accessory open failed: %s\n",
                    libusb_error_name(status));
            handle = nullptr;
        } else {
            fprintf(stderr,
                    "[gemma-attention-host] accessory bus=%d address=%d\n",
                    selected_bus, selected_address);
        }
    } else {
        fprintf(stderr,
                "[gemma-attention-host] accessory candidates=%d\n",
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
            timeout_text != nullptr ? atol(timeout_text) : 10000;
    if (timeout_value <= 0 || timeout_value > 60000) {
        return false;
    }
    uint8_t * pointer = static_cast<uint8_t *>(data);
    size_t completed = 0;
    while (completed < size) {
        int transferred = 0;
        const int status = libusb_bulk_transfer(
                handle, endpoint, pointer + completed,
                (int) (size - completed), &transferred,
                (unsigned int) timeout_value);
        if (status != 0) {
            fprintf(stderr,
                    "[gemma-attention-host] endpoint 0x%02x failed: %s\n",
                    endpoint, libusb_error_name(status));
            return false;
        }
        if (transferred <= 0) {
            return false;
        }
        completed += (size_t) transferred;
    }
    return true;
}

struct attention_path {
    uint32_t offset;
    uint32_t tokens;
    ggml_tensor * key;
    ggml_tensor * value;
    ggml_tensor * packed;
    ggml_cgraph * graph;
};

static attention_path build_attention_path(
        ggml_context * context,
        ggml_tensor * q,
        uint32_t offset,
        uint32_t tokens) {
    attention_path result = {};
    result.offset = offset;
    result.tokens = tokens;
    result.key = ggml_new_tensor_2d(
            context, GGML_TYPE_F16,
            S41_GEMMA_ATTN_HEAD_DIM, tokens);
    result.value = ggml_new_tensor_2d(
            context, GGML_TYPE_F16,
            tokens, S41_GEMMA_ATTN_HEAD_DIM);
    ggml_tensor * scores = ggml_mul_mat(context, result.key, q);
    ggml_tensor * probabilities = ggml_soft_max_ext(
            context, scores, nullptr,
            1.0f / std::sqrt((float) S41_GEMMA_ATTN_HEAD_DIM), 0.0f);
    ggml_tensor * state =
            ggml_mul_mat(context, result.value, probabilities);
    ggml_tensor * score_anchor = ggml_cont(
            context,
            ggml_view_2d(
                    context, scores, 1, S41_GEMMA_ATTN_HEADS,
                    scores->nb[1], 0));
    ggml_tensor * probability_anchor = ggml_cont(
            context,
            ggml_view_2d(
                    context, probabilities, 1,
                    S41_GEMMA_ATTN_HEADS,
                    probabilities->nb[1], 0));
    result.packed = ggml_concat(
            context,
            ggml_concat(context, state, score_anchor, 0),
            probability_anchor, 0);
    result.graph = ggml_new_graph_custom(context, 32, false);
    ggml_build_forward_expand(result.graph, result.packed);
    return result;
}

static uint64_t initialize_kv(attention_path & path) {
    const size_t elements =
            (size_t) path.tokens * S41_GEMMA_ATTN_HEAD_DIM;
    std::vector<ggml_fp16_t> data(elements);
    for (uint32_t token = 0; token < path.tokens; ++token) {
        for (uint32_t dimension = 0;
             dimension < S41_GEMMA_ATTN_HEAD_DIM; ++dimension) {
            data[(size_t) token * S41_GEMMA_ATTN_HEAD_DIM + dimension] =
                    ggml_fp32_to_fp16(s41_gemma_k_value(
                            path.offset + token, dimension));
        }
    }
    uint64_t hash = s41_gemma_hash_bytes(
            data.data(), data.size() * sizeof(data[0]));
    ggml_backend_tensor_set(
            path.key, data.data(), 0, data.size() * sizeof(data[0]));
    for (uint32_t dimension = 0;
         dimension < S41_GEMMA_ATTN_HEAD_DIM; ++dimension) {
        for (uint32_t token = 0; token < path.tokens; ++token) {
            data[(size_t) dimension * path.tokens + token] =
                    ggml_fp32_to_fp16(s41_gemma_v_value(
                            path.offset + token, dimension));
        }
    }
    hash = s41_gemma_hash_bytes(
            data.data(), data.size() * sizeof(data[0]), hash);
    ggml_backend_tensor_set(
            path.value, data.data(), 0, data.size() * sizeof(data[0]));
    return hash;
}

struct shard_output {
    std::vector<float> state =
            std::vector<float>(S41_GEMMA_ATTN_Q_ELEMENTS);
    std::vector<float> log_z =
            std::vector<float>(S41_GEMMA_ATTN_HEADS);
};

static bool unpack_output(
        const std::vector<float> & packed,
        shard_output & output) {
    for (uint32_t head = 0; head < S41_GEMMA_ATTN_HEADS; ++head) {
        const size_t packed_base =
                (size_t) head * S41_GEMMA_ATTN_PACKED_STRIDE;
        const size_t state_base =
                (size_t) head * S41_GEMMA_ATTN_HEAD_DIM;
        memcpy(
                output.state.data() + state_base,
                packed.data() + packed_base,
                S41_GEMMA_ATTN_HEAD_DIM * sizeof(float));
        const float anchor_score =
                packed[packed_base + S41_GEMMA_ATTN_HEAD_DIM];
        const float anchor_probability =
                packed[packed_base + S41_GEMMA_ATTN_HEAD_DIM + 1];
        if (!std::isfinite(anchor_score) ||
            !std::isfinite(anchor_probability) ||
            anchor_probability <= 0.0f) {
            return false;
        }
        output.log_z[head] =
                anchor_score /
                        std::sqrt((float) S41_GEMMA_ATTN_HEAD_DIM) -
                std::log(anchor_probability);
    }
    return true;
}

static bool merge_outputs(
        const shard_output & first,
        const shard_output & second,
        std::vector<float> & merged) {
    for (uint32_t head = 0; head < S41_GEMMA_ATTN_HEADS; ++head) {
        const double maximum = std::max(
                (double) first.log_z[head],
                (double) second.log_z[head]);
        const double first_weight =
                std::exp((double) first.log_z[head] - maximum);
        const double second_weight =
                std::exp((double) second.log_z[head] - maximum);
        const double denominator = first_weight + second_weight;
        if (!std::isfinite(denominator) || denominator <= 0.0) {
            return false;
        }
        const size_t base =
                (size_t) head * S41_GEMMA_ATTN_HEAD_DIM;
        for (uint32_t dimension = 0;
             dimension < S41_GEMMA_ATTN_HEAD_DIM; ++dimension) {
            merged[base + dimension] = (float) (
                    (first_weight * first.state[base + dimension] +
                     second_weight * second.state[base + dimension]) /
                    denominator);
        }
    }
    return true;
}

struct error_metrics {
    double relative_l2 = INFINITY;
    double max_absolute = INFINITY;
    size_t non_finite = 0;
};

static error_metrics compare(
        const std::vector<float> & reference,
        const std::vector<float> & candidate) {
    double error_squared = 0.0;
    double reference_squared = 0.0;
    double max_absolute = 0.0;
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
    }
    error_metrics result = {};
    result.relative_l2 = reference_squared > 0.0
            ? std::sqrt(error_squared / reference_squared)
            : INFINITY;
    result.max_absolute = max_absolute;
    result.non_finite = non_finite;
    return result;
}

struct phone_metrics {
    double total_ms = 0.0;
    double prepare_ms = 0.0;
    double usb_out_ms = 0.0;
    double usb_in_ms = 0.0;
    double decode_ms = 0.0;
    double set_ms = 0.0;
    double compute_ms = 0.0;
    double get_ms = 0.0;
    double encode_ms = 0.0;
};

class phone_client {
public:
    phone_client(
            libusb_device_handle * handle,
            uint32_t total_kv,
            uint32_t segment_offset,
            uint32_t segment_tokens,
            uint64_t kv_hash,
            const std::vector<ggml_fp16_t> & q)
        : handle_(handle),
          total_kv_(total_kv),
          segment_offset_(segment_offset),
          segment_tokens_(segment_tokens),
          kv_hash_(kv_hash),
          q_(q),
          request_data_(
                  sizeof(s41_gemma_attention_request) +
                  S41_GEMMA_ATTN_STATE_BYTES),
          response_data_(
                  sizeof(s41_gemma_attention_response) +
                  S41_GEMMA_ATTN_STATE_BYTES +
                  S41_GEMMA_ATTN_ANCHOR_BYTES) {
    }

    bool run(
            uint32_t request_id,
            shard_output & output,
            phone_metrics & metrics) {
        const auto total_started = steady_clock::now();
        const auto prepare_started = steady_clock::now();
        s41_gemma_attention_request request = {};
        request.magic = S41_GEMMA_ATTN_REQUEST_MAGIC;
        request.version = S41_GEMMA_ATTN_PROTOCOL_VERSION;
        request.request_id = request_id;
        request.total_kv = total_kv_;
        request.segment_offset = segment_offset_;
        request.segment_tokens = segment_tokens_;
        request.q_bytes = S41_GEMMA_ATTN_STATE_BYTES;
        request.q_hash = s41_gemma_hash_bytes(
                q_.data(), S41_GEMMA_ATTN_STATE_BYTES);
        request.kv_hash = kv_hash_;
        memcpy(request_data_.data(), &request, sizeof(request));
        memcpy(
                request_data_.data() + sizeof(request),
                q_.data(), S41_GEMMA_ATTN_STATE_BYTES);
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

        const auto decode_started = steady_clock::now();
        s41_gemma_attention_response response = {};
        memcpy(&response, response_data_.data(), sizeof(response));
        const uint8_t * state_data =
                response_data_.data() + sizeof(response);
        const uint8_t * anchor_data =
                state_data + S41_GEMMA_ATTN_STATE_BYTES;
        uint64_t output_hash = s41_gemma_hash_bytes(
                state_data, S41_GEMMA_ATTN_STATE_BYTES);
        output_hash = s41_gemma_hash_bytes(
                anchor_data, S41_GEMMA_ATTN_ANCHOR_BYTES,
                output_hash);
        if (response.magic != S41_GEMMA_ATTN_RESPONSE_MAGIC ||
            response.version != S41_GEMMA_ATTN_PROTOCOL_VERSION ||
            response.status != 0 ||
            response.request_id != request_id ||
            response.state_bytes != S41_GEMMA_ATTN_STATE_BYTES ||
            response.anchor_bytes != S41_GEMMA_ATTN_ANCHOR_BYTES ||
            response.output_hash != output_hash ||
            response.kv_hash != kv_hash_) {
            fprintf(stderr,
                    "[gemma-attention-host] invalid phone response\n");
            return false;
        }
        const ggml_fp16_t * state_f16 =
                reinterpret_cast<const ggml_fp16_t *>(state_data);
        ggml_fp16_to_fp32_row(
                state_f16, output.state.data(), output.state.size());
        const float * anchors =
                reinterpret_cast<const float *>(anchor_data);
        for (uint32_t head = 0;
             head < S41_GEMMA_ATTN_HEADS; ++head) {
            const float anchor_score = anchors[head];
            const float anchor_probability =
                    anchors[S41_GEMMA_ATTN_HEADS + head];
            if (!std::isfinite(anchor_score) ||
                !std::isfinite(anchor_probability) ||
                anchor_probability <= 0.0f) {
                return false;
            }
            output.log_z[head] =
                    anchor_score /
                            std::sqrt(
                                    (float) S41_GEMMA_ATTN_HEAD_DIM) -
                    std::log(anchor_probability);
        }
        metrics.decode_ms = elapsed_ms(decode_started);
        metrics.total_ms = elapsed_ms(total_started);
        metrics.set_ms = response.set_us / 1000.0;
        metrics.compute_ms = response.compute_us / 1000.0;
        metrics.get_ms = response.get_us / 1000.0;
        metrics.encode_ms = response.encode_us / 1000.0;
        return true;
    }

private:
    libusb_device_handle * handle_;
    uint32_t total_kv_;
    uint32_t segment_offset_;
    uint32_t segment_tokens_;
    uint64_t kv_hash_;
    const std::vector<ggml_fp16_t> & q_;
    std::vector<uint8_t> request_data_;
    std::vector<uint8_t> response_data_;
};

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr,
                "usage: %s <total_kv> <phone_kv> <iters> "
                "<cuda_only|corun>\n",
                argv[0]);
        return 2;
    }
    const int64_t total_kv = atoll(argv[1]);
    const int64_t phone_kv = atoll(argv[2]);
    const int iterations = atoi(argv[3]);
    const std::string mode = argv[4];
    if (total_kv <= 0 || phone_kv <= 0 || phone_kv >= total_kv ||
        total_kv % 256 != 0 || phone_kv % 256 != 0 ||
        iterations <= 0 ||
        (mode != "cuda_only" && mode != "corun")) {
        fprintf(stderr, "[gemma-attention-host] invalid configuration\n");
        return 2;
    }
    const uint32_t gpu_kv = (uint32_t) (total_kv - phone_kv);

    ggml_backend_t backend = nullptr;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (ggml_backend_dev_type(device) ==
            GGML_BACKEND_DEVICE_TYPE_GPU) {
            backend = ggml_backend_dev_init(device, nullptr);
            fprintf(stderr,
                    "[gemma-attention-host] GPU=%s\n",
                    ggml_backend_dev_description(device));
            break;
        }
    }
    if (backend == nullptr) {
        fprintf(stderr, "[gemma-attention-host] no GPU backend\n");
        return 1;
    }

    ggml_init_params parameters = {};
    parameters.mem_size =
            ggml_tensor_overhead() * 80 + ggml_graph_overhead() * 8;
    parameters.no_alloc = true;
    ggml_context * context = ggml_init(parameters);
    if (context == nullptr) {
        return 1;
    }
    ggml_tensor * q = ggml_new_tensor_2d(
            context, GGML_TYPE_F32,
            S41_GEMMA_ATTN_HEAD_DIM, S41_GEMMA_ATTN_HEADS);
    attention_path full = build_attention_path(
            context, q, 0, (uint32_t) total_kv);
    attention_path gpu = build_attention_path(
            context, q, 0, gpu_kv);
    attention_path oracle = build_attention_path(
            context, q, gpu_kv, (uint32_t) phone_kv);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        fprintf(stderr, "[gemma-attention-host] allocation failed\n");
        return 1;
    }

    std::vector<float> q_source(S41_GEMMA_ATTN_Q_ELEMENTS);
    std::vector<ggml_fp16_t> q_f16(S41_GEMMA_ATTN_Q_ELEMENTS);
    std::vector<float> q_f32(S41_GEMMA_ATTN_Q_ELEMENTS);
    for (uint32_t head = 0; head < S41_GEMMA_ATTN_HEADS; ++head) {
        for (uint32_t dimension = 0;
             dimension < S41_GEMMA_ATTN_HEAD_DIM; ++dimension) {
            const size_t index =
                    (size_t) head * S41_GEMMA_ATTN_HEAD_DIM + dimension;
            q_source[index] = s41_gemma_q_value(head, dimension);
        }
    }
    ggml_fp32_to_fp16_row(
            q_source.data(), q_f16.data(), q_f16.size());
    ggml_fp16_to_fp32_row(
            q_f16.data(), q_f32.data(), q_f32.size());
    ggml_backend_tensor_set(
            q, q_f32.data(), 0, q_f32.size() * sizeof(float));

    fprintf(stderr, "[gemma-attention-host] initializing CUDA KV paths\n");
    const uint64_t full_hash = initialize_kv(full);
    const uint64_t gpu_hash = initialize_kv(gpu);
    const uint64_t phone_hash = initialize_kv(oracle);
    fprintf(stderr,
            "[gemma-attention-host] total_kv=%lld split=%u+%lld "
            "hashes=%016llx/%016llx/%016llx nodes=%d/%d/%d\n",
            (long long) total_kv, gpu_kv, (long long) phone_kv,
            (unsigned long long) full_hash,
            (unsigned long long) gpu_hash,
            (unsigned long long) phone_hash,
            ggml_graph_n_nodes(full.graph),
            ggml_graph_n_nodes(gpu.graph),
            ggml_graph_n_nodes(oracle.graph));

    const size_t packed_elements =
            S41_GEMMA_ATTN_PACKED_STRIDE *
            S41_GEMMA_ATTN_HEADS;
    std::vector<float> full_packed(packed_elements);
    std::vector<float> gpu_packed(packed_elements);
    std::vector<float> oracle_packed(packed_elements);
    shard_output full_output;
    shard_output gpu_output;
    shard_output oracle_output;

    auto compute = [&](attention_path & path) {
        const enum ggml_status status =
                ggml_backend_graph_compute(backend, path.graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr,
                    "[gemma-attention-host] compute failed: %d\n",
                    (int) status);
            return false;
        }
        return true;
    };
    auto run_path = [&](
            attention_path & path,
            std::vector<float> & packed,
            shard_output & output,
            double & total_ms) {
        const auto started = steady_clock::now();
        if (!compute(path)) {
            return false;
        }
        ggml_backend_tensor_get(
                path.packed, packed.data(), 0,
                packed.size() * sizeof(float));
        total_ms = elapsed_ms(started);
        return unpack_output(packed, output);
    };

    double ignored_ms = 0.0;
    for (int index = 0; index < 10; ++index) {
        if (!run_path(
                    full, full_packed, full_output, ignored_ms)) {
            return 3;
        }
    }
    if (!run_path(gpu, gpu_packed, gpu_output, ignored_ms) ||
        !run_path(
                oracle, oracle_packed, oracle_output, ignored_ms)) {
        return 3;
    }
    std::vector<float> local_merged(S41_GEMMA_ATTN_Q_ELEMENTS);
    if (!merge_outputs(gpu_output, oracle_output, local_merged)) {
        return 3;
    }
    const error_metrics local_error =
            compare(full_output.state, local_merged);
    printf("GEMMA_GLOBAL_ATTENTION_CORRECTNESS "
           "control=local_context_partition rel_l2=%.9g "
           "max_abs=%.9g non_finite=%zu\n",
            local_error.relative_l2,
            local_error.max_absolute,
            local_error.non_finite);
    if (local_error.non_finite != 0 ||
        local_error.relative_l2 > 0.005) {
        fprintf(stderr,
                "[gemma-attention-host] local correctness failed\n");
        return 4;
    }

    if (mode == "cuda_only") {
        std::vector<double> full_times;
        for (int index = 0; index < iterations; ++index) {
            double total_ms = 0.0;
            if (!run_path(
                        full, full_packed,
                        full_output, total_ms)) {
                return 3;
            }
            full_times.push_back(total_ms);
        }
        printf("GEMMA_GLOBAL_ATTENTION_CUDA_ONLY "
               "total_kv=%lld phone_kv_shape=%lld n=%d ",
                (long long) total_kv,
                (long long) phone_kv, iterations);
        print_metrics("cuda_full", full_times);
        printf("scope=projected_q_resident_kv_attention_core\n");
        return 0;
    }

    libusb_context * usb_context = nullptr;
    if (libusb_init(&usb_context) != 0) {
        return 1;
    }
    const char * bus_text = getenv("S41_AOA_BUS");
    const char * address_text = getenv("S41_AOA_ADDRESS");
    const int wanted_bus = bus_text != nullptr ? atoi(bus_text) : -1;
    const int wanted_address =
            address_text != nullptr ? atoi(address_text) : -1;
    libusb_device_handle * usb =
            open_accessory(
                    usb_context, wanted_bus, wanted_address);
    if (usb == nullptr) {
        return 1;
    }
    libusb_detach_kernel_driver(usb, 0);
    const int claim_status = libusb_claim_interface(usb, 0);
    if (claim_status != 0) {
        fprintf(stderr,
                "[gemma-attention-host] claim failed: %s\n",
                libusb_error_name(claim_status));
        return 1;
    }

    phone_client client(
            usb, (uint32_t) total_kv, gpu_kv,
            (uint32_t) phone_kv, phone_hash, q_f16);
    uint32_t request_id = 1;
    shard_output phone_output;
    phone_metrics correctness_phone = {};
    if (!client.run(
                request_id++, phone_output,
                correctness_phone)) {
        return 3;
    }
    const error_metrics phone_error =
            compare(oracle_output.state, phone_output.state);
    double log_z_max_abs = 0.0;
    for (uint32_t head = 0;
         head < S41_GEMMA_ATTN_HEADS; ++head) {
        log_z_max_abs = std::max(
                log_z_max_abs,
                std::abs(
                        (double) phone_output.log_z[head] -
                        oracle_output.log_z[head]));
    }
    std::vector<float> treatment_output(
            S41_GEMMA_ATTN_Q_ELEMENTS);
    if (!merge_outputs(
                gpu_output, phone_output,
                treatment_output)) {
        return 3;
    }
    const error_metrics treatment_error =
            compare(full_output.state, treatment_output);
    printf("GEMMA_GLOBAL_ATTENTION_CORRECTNESS "
           "component=phone rel_l2=%.9g max_abs=%.9g "
           "log_z_max_abs=%.9g non_finite=%zu\n",
            phone_error.relative_l2,
            phone_error.max_absolute,
            log_z_max_abs,
            phone_error.non_finite);
    printf("GEMMA_GLOBAL_ATTENTION_CORRECTNESS "
           "treatment=cuda_phone_context_partition rel_l2=%.9g "
           "max_abs=%.9g non_finite=%zu\n",
            treatment_error.relative_l2,
            treatment_error.max_absolute,
            treatment_error.non_finite);
    if (phone_error.non_finite != 0 ||
        phone_error.relative_l2 > 0.03 ||
        log_z_max_abs > 0.02 ||
        treatment_error.non_finite != 0 ||
        treatment_error.relative_l2 > 0.01) {
        fprintf(stderr,
                "[gemma-attention-host] phone correctness failed\n");
        return 4;
    }

    for (int index = 0; index < 10; ++index) {
        phone_metrics warmup = {};
        if (!client.run(
                    request_id++, phone_output, warmup)) {
            return 3;
        }
    }

    std::vector<double> full_times;
    std::vector<double> gpu_times;
    std::vector<double> split_times;
    std::vector<double> phone_times;
    std::vector<double> merge_times;
    std::vector<double> prepare_times;
    std::vector<double> usb_out_times;
    std::vector<double> usb_in_times;
    std::vector<double> decode_times;
    std::vector<double> htp_set_times;
    std::vector<double> htp_compute_times;
    std::vector<double> htp_get_times;
    std::vector<double> htp_encode_times;

    auto run_full_timed = [&]() {
        double total_ms = 0.0;
        const bool ok = run_path(
                full, full_packed,
                full_output, total_ms);
        if (ok) {
            full_times.push_back(total_ms);
        }
        return ok;
    };
    auto run_split_timed = [&]() {
        const auto total_started = steady_clock::now();
        const enum ggml_status launch_status =
                ggml_backend_graph_compute_async(
                        backend, gpu.graph);
        if (launch_status != GGML_STATUS_SUCCESS) {
            return false;
        }
        phone_metrics metrics = {};
        if (!client.run(
                    request_id++, phone_output, metrics)) {
            return false;
        }
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(
                gpu.packed, gpu_packed.data(), 0,
                gpu_packed.size() * sizeof(float));
        if (!unpack_output(gpu_packed, gpu_output)) {
            return false;
        }
        const auto merge_started = steady_clock::now();
        if (!merge_outputs(
                    gpu_output, phone_output,
                    treatment_output)) {
            return false;
        }
        merge_times.push_back(elapsed_ms(merge_started));
        split_times.push_back(elapsed_ms(total_started));
        phone_times.push_back(metrics.total_ms);
        prepare_times.push_back(metrics.prepare_ms);
        usb_out_times.push_back(metrics.usb_out_ms);
        usb_in_times.push_back(metrics.usb_in_ms);
        decode_times.push_back(metrics.decode_ms);
        htp_set_times.push_back(metrics.set_ms);
        htp_compute_times.push_back(metrics.compute_ms);
        htp_get_times.push_back(metrics.get_ms);
        htp_encode_times.push_back(metrics.encode_ms);
        const error_metrics iteration_error =
                compare(full_output.state, treatment_output);
        return iteration_error.non_finite == 0 &&
                iteration_error.relative_l2 <= 0.01;
    };

    for (int index = 0; index < iterations; ++index) {
        const bool ok = index % 2 == 0
                ? (run_full_timed() && run_split_timed())
                : (run_split_timed() && run_full_timed());
        if (!ok) {
            fprintf(stderr,
                    "[gemma-attention-host] timed iteration failed\n");
            return 3;
        }
    }
    for (int index = 0; index < iterations; ++index) {
        double total_ms = 0.0;
        if (!run_path(
                    gpu, gpu_packed,
                    gpu_output, total_ms)) {
            return 3;
        }
        gpu_times.push_back(total_ms);
    }

    printf("GEMMA_GLOBAL_ATTENTION_RESULT "
           "total_kv=%lld gpu_kv=%u phone_kv=%lld n=%d ",
            (long long) total_kv, gpu_kv,
            (long long) phone_kv, iterations);
    print_metrics("cuda_full", full_times);
    print_metrics("cuda_shard", gpu_times);
    print_metrics("phone_total", phone_times);
    print_metrics("split_total", split_times);
    print_metrics("merge", merge_times);
    printf("request_bytes=%zu response_bytes=%zu "
           "scope=projected_q_resident_kv_attention_core\n",
            sizeof(s41_gemma_attention_request) +
                    (size_t) S41_GEMMA_ATTN_STATE_BYTES,
            sizeof(s41_gemma_attention_response) +
                    (size_t) S41_GEMMA_ATTN_STATE_BYTES +
                    S41_GEMMA_ATTN_ANCHOR_BYTES);
    printf("GEMMA_GLOBAL_ATTENTION_PHONE_BREAKDOWN ");
    print_metrics("prepare", prepare_times);
    print_metrics("usb_out", usb_out_times);
    print_metrics("usb_in", usb_in_times);
    print_metrics("decode", decode_times);
    print_metrics("htp_set", htp_set_times);
    print_metrics("htp_compute", htp_compute_times);
    print_metrics("htp_get", htp_get_times);
    print_metrics("htp_encode", htp_encode_times);
    printf("\n");
    const double full_median = percentile(full_times, 0.5);
    const double split_median = percentile(split_times, 0.5);
    printf("GEMMA_GLOBAL_ATTENTION_VERDICT "
           "correctness=PASS split_beats_cuda=%s "
           "latency_change_percent=%.3f\n",
            split_median < full_median ? "PASS" : "FAIL",
            100.0 * (split_median / full_median - 1.0));

    libusb_release_interface(usb, 0);
    libusb_close(usb);
    libusb_exit(usb_context);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    return 0;
}
