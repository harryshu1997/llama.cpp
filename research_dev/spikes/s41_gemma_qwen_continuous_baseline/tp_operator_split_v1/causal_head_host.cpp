// Host/phone sharded vocabulary-head diagnostic.
//
// The host computes a prefix of vocabulary rows while the phone computes the
// suffix and reduces it to one token and score before returning over AOA.
//
// usage:
//   causal_head_host <iters> <phone_vocab> [weight_type] [shape]
//       [input_type] [mode]

#include "causal_head_protocol.h"
#include "causal_quantized_weights.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <sys/types.h>
#include <thread>
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
        fprintf(stderr, "[head-host] libusb list failed: %zd\n", count);
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
            fprintf(stderr, "[head-host] accessory open failed: %s\n",
                    libusb_error_name(status));
            handle = nullptr;
        } else {
            fprintf(stderr, "[head-host] accessory bus=%d address=%d\n",
                    selected_bus, selected_address);
        }
    } else if (candidates == 0) {
        fprintf(stderr, "[head-host] no matching accessory found\n");
    } else {
        fprintf(stderr,
                "[head-host] %d accessories match; set S41_AOA_BUS and "
                "S41_AOA_ADDRESS\n",
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
    uint8_t * pointer = (uint8_t *) data;
    size_t completed = 0;
    while (completed < size) {
        const size_t remaining = size - completed;
        if (remaining > (size_t) INT32_MAX) {
            return false;
        }
        int transferred = 0;
        const int status = libusb_bulk_transfer(
                handle, endpoint, pointer + completed, (int) remaining,
                &transferred, 4000);
        if (status != 0) {
            fprintf(stderr,
                    "[head-host] USB endpoint 0x%02x failed: %s\n",
                    endpoint, libusb_error_name(status));
            return false;
        }
        if (transferred <= 0) {
            fprintf(stderr,
                    "[head-host] USB endpoint 0x%02x made no progress\n",
                    endpoint);
            return false;
        }
        completed += (size_t) transferred;
    }
    return true;
}

static bool initialize_full_head(
        ggml_tensor * tensor,
        ggml_type type,
        uint32_t phone_offset,
        uint32_t phone_count,
        uint64_t & phone_weight_hash) {
    const uint32_t columns = (uint32_t) tensor->ne[0];
    const uint32_t rows = (uint32_t) tensor->ne[1];
    std::vector<uint8_t> quantized(ggml_nbytes(tensor));
    if (!s41_fill_quantized_weight(
                type, S41_WEIGHT_HEAD, 0, 0, columns, rows,
                quantized.data(), quantized.size())) {
        return false;
    }
    const size_t row_bytes = ggml_row_size(type, columns);
    const size_t phone_bytes = row_bytes * phone_count;
    phone_weight_hash = s41_weight_hash_update(
            S41_HASH64_OFFSET, S41_WEIGHT_HEAD, phone_offset, 0,
            columns, phone_count,
            quantized.data() + row_bytes * phone_offset, phone_bytes);
    ggml_backend_tensor_set(tensor, quantized.data(), 0, quantized.size());
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
    double reduce_ms = 0.0;
};

class phone_client {
public:
    phone_client(
            libusb_device_handle * handle,
            ggml_type type,
            uint32_t k,
            uint32_t vocab,
            uint32_t offset,
            uint32_t count,
            uint64_t weight_hash,
            const std::string & input_type)
        : handle_(handle),
          type_(type),
          k_(k),
          vocab_(vocab),
          offset_(offset),
          count_(count),
          weight_hash_(weight_hash),
          f16_input_(input_type == "f16"),
          i8_input_(input_type == "i8"),
          flags_(S41_HEAD_FLAG_FAST_HASH |
                  (f16_input_ ? S41_HEAD_FLAG_F16_INPUT : 0) |
                  (i8_input_ ? S41_HEAD_FLAG_I8_INPUT : 0) |
                  S41_HEAD_FLAG_TOP_K),
          input_bytes_(i8_input_
                  ? sizeof(float) + k
                  : (f16_input_
                          ? sizeof(ggml_fp16_t)
                          : sizeof(float)) * k),
          request_data_(sizeof(s41_head_request) + input_bytes_),
          response_data_(
                  sizeof(s41_head_response) +
                  sizeof(s41_head_candidate) * S41_HEAD_TOP_K),
          input_f16_(f16_input_ ? k : 0),
          input_i8_(i8_input_ ? input_bytes_ : 0) {
    }

    bool run(
            uint32_t request_id,
            const std::vector<float> & input,
            std::array<s41_head_candidate, S41_HEAD_TOP_K> & candidates,
            phone_metrics & metrics) {
        const auto submitted = steady_clock::now();
        const auto prepare_started = steady_clock::now();
        const void * encoded_input = input.data();
        if (f16_input_) {
            ggml_fp32_to_fp16_row(
                    input.data(), input_f16_.data(), (size_t) k_);
            encoded_input = input_f16_.data();
        } else if (i8_input_) {
            float maximum = 0.0f;
            for (float value : input) {
                maximum = std::max(maximum, std::abs(value));
            }
            const float scale = std::max(maximum / 127.0f, 1e-12f);
            const float inverse_scale = 1.0f / scale;
            memcpy(input_i8_.data(), &scale, sizeof(scale));
            int8_t * values =
                    (int8_t *) (input_i8_.data() + sizeof(scale));
            for (size_t index = 0; index < input.size(); ++index) {
                const float scaled = input[index] * inverse_scale;
                const int quantized = std::max(
                        -127, std::min(
                                127,
                                (int) (scaled +
                                        (scaled >= 0.0f ? 0.5f : -0.5f))));
                values[index] = (int8_t) quantized;
            }
            encoded_input = input_i8_.data();
        }

        s41_head_request request = {};
        request.magic = S41_HEAD_REQUEST_MAGIC;
        request.version = S41_HEAD_PROTOCOL_VERSION;
        request.flags = flags_;
        request.request_id = request_id;
        request.type = (uint32_t) type_;
        request.k = k_;
        request.vocab = vocab_;
        request.offset = offset_;
        request.count = count_;
        request.input_bytes = (uint32_t) input_bytes_;
        request.input_hash =
                s41_fast_hash_bytes(encoded_input, input_bytes_);
        request.weight_hash = weight_hash_;
        memcpy(request_data_.data(), &request, sizeof(request));
        memcpy(
                request_data_.data() + sizeof(request),
                encoded_input, input_bytes_);
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
        s41_head_response response = {};
        memcpy(&response, response_data_.data(), sizeof(response));
        const void * payload =
                response_data_.data() + sizeof(response);
        memcpy(candidates.data(), payload, sizeof(candidates));
        if (response.magic != S41_HEAD_RESPONSE_MAGIC ||
            response.version != S41_HEAD_PROTOCOL_VERSION ||
            response.status != 0 ||
            response.request_id != request_id ||
            response.candidate_count != S41_HEAD_TOP_K ||
            response.output_bytes != sizeof(candidates) ||
            response.weight_hash != weight_hash_ ||
            response.output_hash !=
                    s41_fast_hash_bytes(payload, sizeof(candidates))) {
            fprintf(stderr, "[head-host] invalid phone response\n");
            return false;
        }
        for (size_t index = 0; index < candidates.size(); ++index) {
            const s41_head_candidate & candidate = candidates[index];
            if (candidate.token < offset_ ||
                candidate.token >= offset_ + count_ ||
                !std::isfinite(candidate.score) ||
                (index > 0 &&
                 (candidate.score > candidates[index - 1].score ||
                  (candidate.score == candidates[index - 1].score &&
                   candidate.token < candidates[index - 1].token)))) {
                fprintf(stderr,
                        "[head-host] invalid phone candidate ordering\n");
                return false;
            }
        }
        metrics.validate_ms = elapsed_ms(validate_started);
        metrics.total_ms = elapsed_ms(submitted);
        metrics.set_ms = response.set_us / 1000.0;
        metrics.compute_ms = response.compute_us / 1000.0;
        metrics.get_ms = response.get_us / 1000.0;
        metrics.reduce_ms = response.reduce_us / 1000.0;
        return true;
    }

private:
    libusb_device_handle * handle_;
    ggml_type type_;
    uint32_t k_;
    uint32_t vocab_;
    uint32_t offset_;
    uint32_t count_;
    uint64_t weight_hash_;
    bool f16_input_;
    bool i8_input_;
    uint16_t flags_;
    size_t input_bytes_;
    std::vector<uint8_t> request_data_;
    std::vector<uint8_t> response_data_;
    std::vector<ggml_fp16_t> input_f16_;
    std::vector<uint8_t> input_i8_;
};

class spin_phone_client {
public:
    explicit spin_phone_client(phone_client & client)
        : client_(client), worker_(&spin_phone_client::work_loop, this) {
    }

    ~spin_phone_client() {
        while (state_.load(std::memory_order_acquire) != 0) {
            pause();
        }
        state_.store(3, std::memory_order_release);
        worker_.join();
    }

    steady_clock::time_point submit(
            uint32_t request_id,
            const std::vector<float> & input,
            std::array<s41_head_candidate, S41_HEAD_TOP_K> & candidates) {
        while (state_.load(std::memory_order_acquire) != 0) {
            pause();
        }
        request_id_ = request_id;
        input_ = &input;
        candidates_ = &candidates;
        const auto submitted = steady_clock::now();
        state_.store(1, std::memory_order_release);
        return submitted;
    }

    bool wait(phone_metrics & metrics) {
        while (state_.load(std::memory_order_acquire) != 2) {
            pause();
        }
        metrics = metrics_;
        const bool ok = ok_;
        state_.store(0, std::memory_order_release);
        return ok;
    }

private:
    static void pause() {
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#else
        std::this_thread::yield();
#endif
    }

    void work_loop() {
        for (;;) {
            const int state = state_.load(std::memory_order_acquire);
            if (state == 0 || state == 2) {
                pause();
                continue;
            }
            if (state == 3) {
                return;
            }
            ok_ = client_.run(
                    request_id_, *input_, *candidates_, metrics_);
            state_.store(2, std::memory_order_release);
        }
    }

    phone_client & client_;
    std::atomic<int> state_ = 0;
    uint32_t request_id_ = 0;
    const std::vector<float> * input_ = nullptr;
    std::array<s41_head_candidate, S41_HEAD_TOP_K> * candidates_ = nullptr;
    phone_metrics metrics_;
    bool ok_ = false;
    std::thread worker_;
};

static s41_head_top1 merge_top1(
        const s41_head_top1 & first,
        const s41_head_top1 & second) {
    if (second.score > first.score ||
        (second.score == first.score && second.token < first.token)) {
        return second;
    }
    return first;
}

static bool candidate_better(
        const s41_head_candidate & first,
        const s41_head_candidate & second) {
    return first.score > second.score ||
            (first.score == second.score && first.token < second.token);
}

static std::array<s41_head_candidate, S41_HEAD_TOP_K> top_candidates(
        const std::vector<float> & logits,
        uint32_t token_offset) {
    std::array<s41_head_candidate, S41_HEAD_TOP_K> result;
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = {UINT32_MAX, -INFINITY};
    }
    for (size_t index = 0; index < logits.size(); ++index) {
        s41_head_candidate candidate = {
            token_offset + (uint32_t) index,
            logits[index],
        };
        size_t position = result.size();
        while (position > 0 &&
               candidate_better(candidate, result[position - 1])) {
            --position;
        }
        if (position == result.size()) {
            continue;
        }
        for (size_t move = result.size() - 1;
             move > position; --move) {
            result[move] = result[move - 1];
        }
        result[position] = candidate;
    }
    return result;
}

static std::array<s41_head_candidate, S41_HEAD_TOP_K> merge_candidates(
        const std::array<s41_head_candidate, S41_HEAD_TOP_K> & first,
        const std::array<s41_head_candidate, S41_HEAD_TOP_K> & second) {
    std::array<s41_head_candidate, S41_HEAD_TOP_K> merged;
    size_t first_index = 0;
    size_t second_index = 0;
    for (size_t index = 0; index < merged.size(); ++index) {
        if (second_index >= second.size() ||
            (first_index < first.size() &&
             candidate_better(
                     first[first_index], second[second_index]))) {
            merged[index] = first[first_index++];
        } else {
            merged[index] = second[second_index++];
        }
    }
    return merged;
}

static bool same_candidate_tokens(
        const std::array<s41_head_candidate, S41_HEAD_TOP_K> & first,
        const std::array<s41_head_candidate, S41_HEAD_TOP_K> & second) {
    for (size_t index = 0; index < first.size(); ++index) {
        if (first[index].token != second[index].token) {
            return false;
        }
    }
    return true;
}

static size_t matching_candidate_tokens(
        const std::array<s41_head_candidate, S41_HEAD_TOP_K> & first,
        const std::array<s41_head_candidate, S41_HEAD_TOP_K> & second) {
    size_t matches = 0;
    for (size_t index = 0; index < first.size(); ++index) {
        matches += first[index].token == second[index].token ? 1 : 0;
    }
    return matches;
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 7) {
        fprintf(stderr,
                "usage: %s <iters> <phone_vocab> [weight_type] [shape] "
                "[input_type] [mode]\n",
                argv[0]);
        return 2;
    }

    const int iterations = atoi(argv[1]);
    const int64_t phone_vocab = atoll(argv[2]);
    ggml_type weight_type = GGML_TYPE_Q8_0;
    if (argc > 3 && !parse_type(argv[3], weight_type)) {
        fprintf(stderr, "[head-host] unknown type '%s'\n", argv[3]);
        return 2;
    }
    const std::string shape = argc > 4 ? argv[4] : "qwen3_14b";
    int64_t k = 5120;
    int64_t vocab = 151936;
    if (shape == "gemma4_12b") {
        k = 3840;
        vocab = 262144;
    } else if (shape != "qwen3_14b") {
        fprintf(stderr, "[head-host] unknown shape '%s'\n", shape.c_str());
        return 2;
    }
    const std::string input_type = argc > 5 ? argv[5] : "f32";
    if (input_type != "f32" &&
        input_type != "f16" &&
        input_type != "i8") {
        fprintf(stderr, "[head-host] unknown input type '%s'\n",
                input_type.c_str());
        return 2;
    }
    const std::string mode = argc > 6 ? argv[6] : "steady_corun";
    if (mode != "steady_corun" &&
        mode != "spin_corun" &&
        mode != "host_only" &&
        mode != "host_profile" &&
        mode != "cuda_only" &&
        mode != "phone_only" &&
        mode != "dynamic_check") {
        fprintf(stderr, "[head-host] unknown mode '%s'\n", mode.c_str());
        return 2;
    }

    const int64_t gpu_vocab = vocab - phone_vocab;
    const int64_t block = ggml_blck_size(weight_type);
    if (iterations <= 0 || phone_vocab <= 0 || gpu_vocab <= 0 ||
        (weight_type != GGML_TYPE_Q4_0 &&
         weight_type != GGML_TYPE_Q8_0) ||
        k % block != 0) {
        fprintf(stderr, "[head-host] invalid configuration\n");
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
        fprintf(stderr, "[head-host] invalid S41_HOST_BACKEND\n");
        return 2;
    }
    if (host_is_cpu &&
        mode != "spin_corun" &&
        mode != "host_only" &&
        mode != "host_profile" &&
        mode != "phone_only" &&
        mode != "dynamic_check") {
        fprintf(stderr,
                "[head-host] CPU supports spin_corun, host_only, "
                "host_profile, or phone_only\n");
        return 2;
    }
    const enum ggml_backend_dev_type wanted_host_type = host_is_cpu
            ? GGML_BACKEND_DEVICE_TYPE_CPU
            : GGML_BACKEND_DEVICE_TYPE_GPU;
    ggml_backend_dev_t host_device = nullptr;
    ggml_backend_t backend = nullptr;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (ggml_backend_dev_type(device) == wanted_host_type) {
            host_device = device;
            backend = ggml_backend_dev_init(device, nullptr);
            break;
        }
    }
    if (backend == nullptr) {
        fprintf(stderr, "[head-host] requested backend not found\n");
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
                fprintf(stderr, "[head-host] invalid S41_CPU_THREADS\n");
                return 2;
            }
            cpu_threads = (int) parsed;
        }
        ggml_backend_cpu_set_n_threads(backend, cpu_threads);
    }
    fprintf(stderr,
            "[head-host] host_backend=%s description=%s threads=%d\n",
            host_is_cpu ? "CPU" : "GPU",
            ggml_backend_dev_description(host_device),
            host_is_cpu ? cpu_threads : 0);

    ggml_init_params parameters = {};
    parameters.mem_size =
            ggml_tensor_overhead() * 48 + ggml_graph_overhead() * 4;
    parameters.no_alloc = true;
    ggml_context * context = ggml_init(parameters);
    if (context == nullptr) {
        return 1;
    }

    ggml_tensor * input =
            ggml_new_tensor_2d(context, GGML_TYPE_F32, k, 1);
    ggml_tensor * full_head =
            ggml_new_tensor_2d(context, weight_type, k, vocab);
    ggml_tensor * gpu_head = ggml_view_2d(
            context, full_head, k, gpu_vocab, full_head->nb[1], 0);
    ggml_tensor * phone_head = ggml_view_2d(
            context, full_head, k, phone_vocab, full_head->nb[1],
            (size_t) gpu_vocab * full_head->nb[1]);

    ggml_tensor * full_logits = ggml_mul_mat(context, full_head, input);
    ggml_tensor * full_argmax = ggml_argmax(context, full_logits);
    ggml_tensor * full_rows =
            ggml_reshape_2d(context, full_logits, 1, vocab);
    ggml_tensor * full_score =
            ggml_get_rows(context, full_rows, full_argmax);

    ggml_tensor * gpu_logits = ggml_mul_mat(context, gpu_head, input);
    ggml_tensor * gpu_argmax = ggml_argmax(context, gpu_logits);
    ggml_tensor * gpu_rows =
            ggml_reshape_2d(context, gpu_logits, 1, gpu_vocab);
    ggml_tensor * gpu_score =
            ggml_get_rows(context, gpu_rows, gpu_argmax);

    ggml_tensor * phone_logits = ggml_mul_mat(context, phone_head, input);
    ggml_tensor * phone_argmax = ggml_argmax(context, phone_logits);
    ggml_tensor * phone_rows =
            ggml_reshape_2d(context, phone_logits, 1, phone_vocab);
    ggml_tensor * phone_score =
            ggml_get_rows(context, phone_rows, phone_argmax);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        fprintf(stderr, "[head-host] allocation failed\n");
        return 1;
    }

    fprintf(stderr, "[head-host] initializing full head\n");
    uint64_t phone_weight_hash = S41_HASH64_OFFSET;
    if (!initialize_full_head(
                full_head, weight_type, (uint32_t) gpu_vocab,
                (uint32_t) phone_vocab, phone_weight_hash)) {
        fprintf(stderr, "[head-host] weight initialization failed\n");
        return 1;
    }

    std::vector<float> input_data((size_t) k);
    for (size_t index = 0; index < input_data.size(); ++index) {
        input_data[index] = s41_activation_value((uint32_t) index);
    }
    ggml_backend_tensor_set(
            input, input_data.data(), 0,
            input_data.size() * sizeof(float));
    auto set_input_sample = [&](uint32_t sample) {
        const uint32_t shift = 17 * (sample + 1);
        for (size_t index = 0; index < input_data.size(); ++index) {
            input_data[index] = s41_activation_value(
                    (uint32_t) index + shift);
        }
        ggml_backend_tensor_set(
                input, input_data.data(), 0,
                input_data.size() * sizeof(float));
    };

    ggml_cgraph * full_graph =
            ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(full_graph, full_score);
    ggml_cgraph * gpu_graph =
            ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(gpu_graph, gpu_score);
    ggml_cgraph * phone_oracle_graph =
            ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(phone_oracle_graph, phone_score);

    fprintf(stderr,
            "[head-host] shape=%s K=%lld vocab=%lld split=%lld+%lld "
            "type=%s input_type=%s phone_weight_hash=%016llx "
            "nodes=%d/%d/%d\n",
            shape.c_str(), (long long) k, (long long) vocab,
            (long long) gpu_vocab, (long long) phone_vocab,
            ggml_type_name(weight_type), input_type.c_str(),
            (unsigned long long) phone_weight_hash,
            ggml_graph_n_nodes(full_graph),
            ggml_graph_n_nodes(gpu_graph),
            ggml_graph_n_nodes(phone_oracle_graph));

    auto compute = [&](ggml_cgraph * graph) {
        const enum ggml_status status =
                ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[head-host] compute failed: %d\n",
                    (int) status);
            return false;
        }
        return true;
    };
    auto read_top = [&](
            ggml_tensor * argmax,
            ggml_tensor * score,
            uint32_t token_offset,
            s41_head_top1 & top1) {
        int32_t token = -1;
        ggml_backend_tensor_get(argmax, &token, 0, sizeof(token));
        ggml_backend_tensor_get(score, &top1.score, 0, sizeof(top1.score));
        if (token < 0 || !std::isfinite(top1.score)) {
            return false;
        }
        top1.token = token_offset + (uint32_t) token;
        return true;
    };
    auto run_gpu = [&](
            ggml_cgraph * graph,
            ggml_tensor * argmax,
            ggml_tensor * score,
            uint32_t token_offset,
            s41_head_top1 & top1,
            double & total_ms) {
        const auto started = steady_clock::now();
        if (!compute(graph) ||
            !read_top(argmax, score, token_offset, top1)) {
            return false;
        }
        total_ms = elapsed_ms(started);
        return true;
    };

    s41_head_top1 full_top = {};
    s41_head_top1 gpu_top = {};
    s41_head_top1 phone_oracle_top = {};
    double ignored_ms = 0.0;
    for (int index = 0; index < 20; ++index) {
        if (!run_gpu(
                    full_graph, full_argmax, full_score, 0,
                    full_top, ignored_ms) ||
            !run_gpu(
                    gpu_graph, gpu_argmax, gpu_score, 0,
                    gpu_top, ignored_ms) ||
            !run_gpu(
                    phone_oracle_graph, phone_argmax, phone_score,
                    (uint32_t) gpu_vocab, phone_oracle_top, ignored_ms)) {
            return 3;
        }
    }
    std::vector<float> full_logits_data((size_t) vocab);
    std::vector<float> gpu_logits_data((size_t) gpu_vocab);
    std::vector<float> phone_logits_data((size_t) phone_vocab);
    ggml_backend_tensor_get(
            full_logits, full_logits_data.data(), 0,
            full_logits_data.size() * sizeof(float));
    ggml_backend_tensor_get(
            gpu_logits, gpu_logits_data.data(), 0,
            gpu_logits_data.size() * sizeof(float));
    ggml_backend_tensor_get(
            phone_logits, phone_logits_data.data(), 0,
            phone_logits_data.size() * sizeof(float));
    const auto full_candidates =
            top_candidates(full_logits_data, 0);
    const auto gpu_candidates =
            top_candidates(gpu_logits_data, 0);
    const auto phone_oracle_candidates =
            top_candidates(phone_logits_data, (uint32_t) gpu_vocab);
    const auto cuda_partition_candidates =
            merge_candidates(gpu_candidates, phone_oracle_candidates);

    if (mode == "host_only" || mode == "cuda_only") {
        std::vector<double> full_times;
        for (int index = 0; index < iterations; ++index) {
            double total_ms = 0.0;
            if (!run_gpu(
                        full_graph, full_argmax, full_score, 0,
                        full_top, total_ms)) {
                return 3;
            }
            full_times.push_back(total_ms);
        }
        printf("HEAD_HOST_ONLY host_backend=%s shape=%s type=%s "
               "vocab=%lld n=%d ",
                host_is_cpu ? "CPU" : "GPU",
                shape.c_str(), ggml_type_name(weight_type),
                (long long) vocab, iterations);
        print_metrics(host_is_cpu ? "cpu_full" : "cuda_full", full_times);
        printf("token=%u score=%.9g\n", full_top.token, full_top.score);
        return 0;
    }

    if (mode == "host_profile") {
        std::vector<double> full_times;
        std::vector<double> shard_times;
        for (int index = 0; index < iterations; ++index) {
            double total_ms = 0.0;
            if (!run_gpu(
                        full_graph, full_argmax, full_score, 0,
                        full_top, total_ms)) {
                return 3;
            }
            full_times.push_back(total_ms);
            if (!run_gpu(
                        gpu_graph, gpu_argmax, gpu_score, 0,
                        gpu_top, total_ms)) {
                return 3;
            }
            shard_times.push_back(total_ms);
        }
        printf("HEAD_HOST_PROFILE host_backend=%s shape=%s type=%s "
               "vocab_split=%lld+%lld n=%d ",
                host_is_cpu ? "CPU" : "GPU", shape.c_str(),
                ggml_type_name(weight_type),
                (long long) gpu_vocab, (long long) phone_vocab,
                iterations);
        print_metrics(host_is_cpu ? "cpu_full" : "cuda_full", full_times);
        print_metrics(host_is_cpu ? "cpu_shard" : "cuda_shard",
                      shard_times);
        printf("full_token=%u shard_token=%u\n",
                full_top.token, gpu_top.token);
        return 0;
    }

    libusb_context * usb_context = nullptr;
    if (libusb_init(&usb_context) != 0) {
        fprintf(stderr, "[head-host] libusb initialization failed\n");
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
        fprintf(stderr, "[head-host] claim failed: %s\n",
                libusb_error_name(claim_status));
        return 1;
    }

    phone_client client(
            usb, weight_type, (uint32_t) k, (uint32_t) vocab,
            (uint32_t) gpu_vocab, (uint32_t) phone_vocab,
            phone_weight_hash, input_type);
    uint32_t request_id = 1;
    std::array<s41_head_candidate, S41_HEAD_TOP_K> phone_candidates;
    phone_metrics correctness_phone = {};
    if (!client.run(
                request_id++, input_data,
                phone_candidates, correctness_phone)) {
        return 3;
    }
    const s41_head_top1 phone_top = phone_candidates[0];
    const s41_head_top1 cuda_partition_top =
            merge_top1(gpu_top, phone_oracle_top);
    const s41_head_top1 split_top =
            merge_top1(gpu_top, phone_top);
    const auto split_candidates =
            merge_candidates(gpu_candidates, phone_candidates);
    const double phone_score_error =
            std::abs((double) phone_top.score - phone_oracle_top.score);
    const double phone_score_relative = phone_score_error /
            std::max(1e-12, std::abs((double) phone_oracle_top.score));
    printf("HEAD_CORRECTNESS full_token=%u gpu_token=%u "
           "phone_oracle_token=%u phone_token=%u "
           "cuda_partition_token=%u split_token=%u "
           "phone_score_rel=%.9g phone_score_abs=%.9g "
           "phone_local_topk_positions=%zu/%u "
           "cuda_partition_topk=%s split_topk=%s k=%u\n",
            full_top.token, gpu_top.token,
            phone_oracle_top.token, phone_top.token,
            cuda_partition_top.token, split_top.token,
            phone_score_relative, phone_score_error,
            matching_candidate_tokens(
                    phone_oracle_candidates, phone_candidates),
            S41_HEAD_TOP_K,
            same_candidate_tokens(
                    full_candidates, cuda_partition_candidates)
                    ? "PASS" : "FAIL",
            same_candidate_tokens(full_candidates, split_candidates)
                    ? "PASS" : "FAIL",
            S41_HEAD_TOP_K);
    const bool correctness_pass =
            phone_top.token == phone_oracle_top.token &&
            cuda_partition_top.token == full_top.token &&
            split_top.token == full_top.token &&
            same_candidate_tokens(
                    full_candidates, cuda_partition_candidates) &&
            same_candidate_tokens(full_candidates, split_candidates) &&
            std::isfinite(phone_score_relative);
    if (!correctness_pass) {
        fprintf(stderr, "[head-host] correctness gate failed\n");
        return 4;
    }

    if (mode == "dynamic_check") {
        size_t phone_top1_passes = 0;
        size_t phone_topk_passes = 0;
        size_t phone_topk_position_matches = 0;
        size_t split_top1_passes = 0;
        size_t split_topk_passes = 0;
        double max_phone_top1_relative = 0.0;
        std::vector<uint32_t> input_hashes;
        input_hashes.reserve((size_t) iterations);
        for (int index = 0; index < iterations; ++index) {
            set_input_sample((uint32_t) index);
            input_hashes.push_back(s41_fast_hash_bytes(
                    input_data.data(), input_data.size() * sizeof(float)));
            if (!run_gpu(
                        full_graph, full_argmax, full_score, 0,
                        full_top, ignored_ms) ||
                !run_gpu(
                        gpu_graph, gpu_argmax, gpu_score, 0,
                        gpu_top, ignored_ms) ||
                !run_gpu(
                        phone_oracle_graph, phone_argmax, phone_score,
                        (uint32_t) gpu_vocab, phone_oracle_top,
                        ignored_ms)) {
                return 3;
            }
            ggml_backend_tensor_get(
                    full_logits, full_logits_data.data(), 0,
                    full_logits_data.size() * sizeof(float));
            ggml_backend_tensor_get(
                    gpu_logits, gpu_logits_data.data(), 0,
                    gpu_logits_data.size() * sizeof(float));
            ggml_backend_tensor_get(
                    phone_logits, phone_logits_data.data(), 0,
                    phone_logits_data.size() * sizeof(float));
            const auto sample_full_candidates =
                    top_candidates(full_logits_data, 0);
            const auto sample_gpu_candidates =
                    top_candidates(gpu_logits_data, 0);
            const auto sample_phone_oracle_candidates = top_candidates(
                    phone_logits_data, (uint32_t) gpu_vocab);
            phone_metrics sample_metrics = {};
            if (!client.run(
                        request_id++, input_data,
                        phone_candidates, sample_metrics)) {
                return 3;
            }
            const auto sample_split_candidates = merge_candidates(
                    sample_gpu_candidates, phone_candidates);
            const bool phone_top1_pass =
                    sample_phone_oracle_candidates[0].token ==
                    phone_candidates[0].token;
            const bool phone_topk_pass = same_candidate_tokens(
                    sample_phone_oracle_candidates, phone_candidates);
            const bool split_top1_pass =
                    sample_full_candidates[0].token ==
                    sample_split_candidates[0].token;
            const bool split_topk_pass = same_candidate_tokens(
                    sample_full_candidates, sample_split_candidates);
            phone_top1_passes += phone_top1_pass ? 1 : 0;
            phone_topk_passes += phone_topk_pass ? 1 : 0;
            phone_topk_position_matches += matching_candidate_tokens(
                    sample_phone_oracle_candidates, phone_candidates);
            split_top1_passes += split_top1_pass ? 1 : 0;
            split_topk_passes += split_topk_pass ? 1 : 0;
            if (phone_top1_pass) {
                const double relative = std::abs(
                        (double) phone_candidates[0].score -
                        sample_phone_oracle_candidates[0].score) /
                        std::max(
                                1e-12,
                                std::abs((double)
                                        sample_phone_oracle_candidates[0]
                                                .score));
                max_phone_top1_relative = std::max(
                        max_phone_top1_relative, relative);
            }
            if (!phone_top1_pass || !phone_topk_pass ||
                !split_top1_pass || !split_topk_pass) {
                fprintf(stderr,
                        "[head-host] dynamic sample=%d phone_top1=%s "
                        "phone_topk=%zu/%u split_top1=%s "
                        "split_topk=%s\n",
                        index, phone_top1_pass ? "PASS" : "FAIL",
                        matching_candidate_tokens(
                                sample_phone_oracle_candidates,
                                phone_candidates),
                        S41_HEAD_TOP_K,
                        split_top1_pass ? "PASS" : "FAIL",
                        split_topk_pass ? "PASS" : "FAIL");
            }
        }
        std::sort(input_hashes.begin(), input_hashes.end());
        const size_t distinct_inputs = (size_t) std::distance(
                input_hashes.begin(),
                std::unique(input_hashes.begin(), input_hashes.end()));
        printf("HEAD_DYNAMIC_CHECK shape=%s type=%s input_type=%s "
               "samples=%d distinct_inputs=%zu phone_top1=%zu/%d "
               "phone_topk=%zu/%d phone_topk_positions=%zu/%zu "
               "split_top1=%zu/%d split_topk=%zu/%d "
               "max_phone_top1_rel=%.9g "
               "graph_cache_mode=worker_reported\n",
                shape.c_str(), ggml_type_name(weight_type),
                input_type.c_str(), iterations, distinct_inputs,
                phone_top1_passes, iterations,
                phone_topk_passes, iterations,
                phone_topk_position_matches,
                (size_t) iterations * S41_HEAD_TOP_K,
                split_top1_passes, iterations,
                split_topk_passes, iterations,
                max_phone_top1_relative);
        libusb_release_interface(usb, 0);
        libusb_close(usb);
        libusb_exit(usb_context);
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        ggml_backend_free(backend);
        ggml_quantize_free();
        return distinct_inputs == (size_t) iterations &&
                split_top1_passes == (size_t) iterations &&
                split_topk_passes == (size_t) iterations ? 0 : 4;
    }

    for (int index = 0; index < 20; ++index) {
        phone_metrics metrics = {};
        if (!client.run(
                    request_id++, input_data,
                    phone_candidates, metrics)) {
            return 3;
        }
    }
    if (mode == "phone_only") {
        std::vector<double> phone_times;
        std::vector<double> compute_times;
        std::vector<double> get_times;
        std::vector<double> reduce_times;
        for (int index = 0; index < iterations; ++index) {
            phone_metrics metrics = {};
            if (!client.run(
                        request_id++, input_data,
                        phone_candidates, metrics)) {
                return 3;
            }
            phone_times.push_back(metrics.total_ms);
            compute_times.push_back(metrics.compute_ms);
            get_times.push_back(metrics.get_ms);
            reduce_times.push_back(metrics.reduce_ms);
        }
        printf("HEAD_PHONE_ONLY shape=%s type=%s input_type=%s "
               "phone_vocab=%lld n=%d ",
                shape.c_str(), ggml_type_name(weight_type),
                input_type.c_str(), (long long) phone_vocab, iterations);
        print_metrics("phone_total", phone_times);
        print_metrics("htp_compute", compute_times);
        print_metrics("htp_get", get_times);
        print_metrics("arm_reduce", reduce_times);
        printf("token=%u score=%.9g candidates=%u response_bytes=%zu\n",
                phone_candidates[0].token, phone_candidates[0].score,
                S41_HEAD_TOP_K,
                sizeof(s41_head_candidate) * phone_candidates.size());
        return 0;
    }

    std::vector<double> full_times;
    std::vector<double> gpu_times;
    for (int index = 0; index < iterations; ++index) {
        double total_ms = 0.0;
        if (!run_gpu(
                    full_graph, full_argmax, full_score, 0,
                    full_top, total_ms)) {
            return 3;
        }
        full_times.push_back(total_ms);
    }
    for (int index = 0; index < iterations; ++index) {
        double total_ms = 0.0;
        if (!run_gpu(
                    gpu_graph, gpu_argmax, gpu_score, 0,
                    gpu_top, total_ms)) {
            return 3;
        }
        gpu_times.push_back(total_ms);
    }
    const double gpu_deadline_ms = percentile(gpu_times, 0.5);
    std::unique_ptr<spin_phone_client> spin_client;
    if (mode == "spin_corun") {
        spin_client = std::make_unique<spin_phone_client>(client);
    }

    std::vector<double> split_times;
    std::vector<double> phone_times;
    std::vector<double> exposed_wait_times;
    std::vector<double> merge_times;
    std::vector<double> prepare_times;
    std::vector<double> usb_out_times;
    std::vector<double> usb_in_times;
    std::vector<double> set_times;
    std::vector<double> compute_times;
    std::vector<double> get_times;
    std::vector<double> reduce_times;
    size_t hidden_count = 0;
    for (int index = 0; index < iterations; ++index) {
        auto total_started = steady_clock::now();
        phone_metrics metrics = {};
        if (spin_client != nullptr) {
            total_started = spin_client->submit(
                    request_id++, input_data, phone_candidates);
            if (!compute(gpu_graph) || !spin_client->wait(metrics)) {
                return 3;
            }
        } else {
            const enum ggml_status launch_status =
                    ggml_backend_graph_compute_async(backend, gpu_graph);
            if (launch_status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[head-host] async launch failed: %d\n",
                        (int) launch_status);
                return 3;
            }
            if (!client.run(
                        request_id++, input_data,
                        phone_candidates, metrics)) {
                return 3;
            }
            ggml_backend_synchronize(backend);
        }
        const auto merge_started = steady_clock::now();
        if (!read_top(gpu_argmax, gpu_score, 0, gpu_top)) {
            return 3;
        }
        const s41_head_top1 treatment_top =
                merge_top1(gpu_top, phone_candidates[0]);
        merge_times.push_back(elapsed_ms(merge_started));
        split_times.push_back(elapsed_ms(total_started));
        phone_times.push_back(metrics.total_ms);
        exposed_wait_times.push_back(
                std::max(0.0, metrics.total_ms - gpu_deadline_ms));
        prepare_times.push_back(metrics.prepare_ms);
        usb_out_times.push_back(metrics.usb_out_ms);
        usb_in_times.push_back(metrics.usb_in_ms);
        set_times.push_back(metrics.set_ms);
        compute_times.push_back(metrics.compute_ms);
        get_times.push_back(metrics.get_ms);
        reduce_times.push_back(metrics.reduce_ms);
        hidden_count += metrics.total_ms <= gpu_deadline_ms ? 1 : 0;
        if (treatment_top.token != full_top.token) {
            fprintf(stderr, "[head-host] treatment token changed\n");
            return 4;
        }
    }

    printf("HEAD_RESULT host_backend=%s shape=%s type=%s input_type=%s "
           "vocab_split=%lld+%lld n=%d ",
            host_is_cpu ? "CPU" : "GPU",
            shape.c_str(), ggml_type_name(weight_type), input_type.c_str(),
            (long long) gpu_vocab, (long long) phone_vocab, iterations);
    print_metrics(host_is_cpu ? "cpu_full" : "cuda_full", full_times);
    print_metrics(host_is_cpu ? "cpu_shard" : "cuda_shard", gpu_times);
    print_metrics("phone_total", phone_times);
    print_metrics("split_total", split_times);
    print_metrics("exposed_wait", exposed_wait_times);
    print_metrics("merge", merge_times);
    printf("hidden=%zu/%d hidden_fraction=%.6f\n",
            hidden_count, iterations,
            (double) hidden_count / (double) iterations);
    printf("HEAD_PHONE_BREAKDOWN ");
    print_metrics("prepare", prepare_times);
    print_metrics("usb_out", usb_out_times);
    print_metrics("usb_in_compute", usb_in_times);
    print_metrics("htp_set", set_times);
    print_metrics("htp_compute", compute_times);
    print_metrics("htp_get", get_times);
    print_metrics("arm_reduce", reduce_times);
    printf("\n");

    const double full_median = percentile(full_times, 0.5);
    const double split_median = percentile(split_times, 0.5);
    printf("HEAD_VERDICT comparison_scope=same_process_same_config "
           "phone_top1=PASS split_top1=PASS "
           "phone_topk=PASS split_topk=PASS candidates=%u "
           "response_bytes=%zu "
           "split_beats_host=%s latency_change_percent=%.3f\n",
            S41_HEAD_TOP_K,
            sizeof(s41_head_candidate) * phone_candidates.size(),
            split_median < full_median ? "PASS" : "FAIL",
            100.0 * (split_median / full_median - 1.0));

    spin_client.reset();
    libusb_release_interface(usb, 0);
    libusb_close(usb);
    libusb_exit(usb_context);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    ggml_quantize_free();
    return 0;
}
