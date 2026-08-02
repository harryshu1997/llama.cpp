// CUDA/HTP FFN-slice diagnostic.
//
// HTP computes a suffix of gate, up, SiLU, and multiply. CUDA computes the
// complementary FFN prefix. The phone returns either the gated activation for
// a CUDA suffix-down continuation or a completed residual for a host-side sum.
//
// usage:
//   causal_activation_host <iters> <phone_cols> [weight_type] [shape]
//       [input_type] [mode] [output_type] [late_path] [batch] [result]

#include "causal_ffn_protocol.h"
#include "causal_quantized_weights.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
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
        libusb_device_handle *, unsigned char, unsigned char *, int, int *,
        unsigned int);
const char * libusb_error_name(int);
}

using steady_clock = std::chrono::steady_clock;

static double duration_ms(
        steady_clock::time_point started, steady_clock::time_point finished) {
    return std::chrono::duration<double, std::milli>(finished - started).count();
}

static double elapsed_ms(steady_clock::time_point started) {
    return duration_ms(started, steady_clock::now());
}

static uint64_t unix_time_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

static void print_energy_marker(
        const char * boundary, const char * mode, int iterations) {
    printf("ENERGY_WINDOW_%s unix_ns=%llu mode=%s n=%d\n",
            boundary, (unsigned long long) unix_time_ns(), mode, iterations);
    fflush(stdout);
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
        fprintf(stderr, "[activation-host] libusb list failed: %zd\n", count);
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
            fprintf(stderr, "[activation-host] accessory open failed: %s\n",
                    libusb_error_name(status));
            handle = nullptr;
        } else {
            fprintf(stderr, "[activation-host] accessory bus=%d address=%d\n",
                    selected_bus, selected_address);
        }
    } else if (candidates == 0) {
        fprintf(stderr, "[activation-host] no matching accessory found\n");
    } else {
        fprintf(stderr,
                "[activation-host] %d accessories match; set S41_AOA_BUS "
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
                    "[activation-host] USB endpoint 0x%02x failed: %s\n",
                    endpoint, libusb_error_name(status));
            return false;
        }
        if (transferred <= 0) {
            fprintf(stderr,
                    "[activation-host] USB endpoint 0x%02x made no progress\n",
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
        fprintf(stderr, "[activation-host] deterministic weight fill failed\n");
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

struct phone_metrics {
    double total_ms = 0.0;
    double prepare_ms = 0.0;
    double usb_out_ms = 0.0;
    double usb_in_ms = 0.0;
    double validate_ms = 0.0;
};

class phone_client {
public:
    phone_client(
            libusb_device_handle * handle,
            ggml_type type,
            uint32_t k,
            uint32_t n_ff,
            uint32_t offset,
            uint32_t count,
            uint64_t weight_hash,
            int input_encoding,
            bool f16_output,
            uint16_t batch,
            bool residual_output)
        : handle_(handle),
          type_(type),
          k_(k),
          n_ff_(n_ff),
          offset_(offset),
          count_(count),
          batch_(batch),
          weight_hash_(weight_hash),
          input_encoding_(input_encoding),
          f16_output_(f16_output),
          residual_output_(residual_output),
          input_elements_((size_t) k * batch),
          output_elements_((size_t) (
                  residual_output ? k : count) * batch),
          input_bytes_(input_encoding == 2
                  ? sizeof(float) + input_elements_
                  : (input_encoding == 1
                          ? sizeof(ggml_fp16_t)
                          : sizeof(float)) * input_elements_),
          request_data_(
                  sizeof(s41_ffn_request) + input_bytes_),
          response_data_(
                  sizeof(s41_ffn_response) +
                          (f16_output
                                  ? sizeof(ggml_fp16_t)
                                  : sizeof(float)) * output_elements_),
          input_f16_(input_encoding == 1 ? input_elements_ : 0),
          input_i8_(input_encoding == 2 ? input_bytes_ : 0) {
    }

    bool run(
            uint32_t request_id,
            const std::vector<float> & input,
            std::vector<float> & output,
            phone_metrics & metrics,
            std::atomic<bool> * output_started = nullptr) {
        return begin(request_id, input, metrics, output_started) &&
                finish(request_id, output, metrics);
    }

    bool begin(
            uint32_t request_id,
            const std::vector<float> & input,
            phone_metrics & metrics,
            std::atomic<bool> * output_started = nullptr) {
        submitted_ = steady_clock::now();
        const auto prepare_started = steady_clock::now();
        s41_ffn_request request = {};
        request.magic = S41_FFN_REQUEST_MAGIC;
        request.version = S41_FFN_PROTOCOL_VERSION;
        const uint16_t base_flags = S41_FFN_FLAG_FAST_HASH |
                (input_encoding_ == 1
                        ? S41_FFN_FLAG_F16_INPUT : 0) |
                (input_encoding_ == 2
                        ? S41_FFN_FLAG_I8_INPUT : 0) |
                (f16_output_ ? S41_FFN_FLAG_F16_OUTPUT : 0) |
                (residual_output_
                        ? S41_FFN_FLAG_RESIDUAL_OUTPUT : 0);
        request.reserved = s41_ffn_encode_flags(base_flags, batch_);
        request.request_id = request_id;
        request.type = (uint32_t) type_;
        request.k = k_;
        request.n_ff = n_ff_;
        request.offset = offset_;
        request.count = count_;
        request.input_bytes = (uint32_t) input_bytes_;
        const void * input_data = input.data();
        if (input_encoding_ == 1) {
            ggml_fp32_to_fp16_row(
                    input.data(), input_f16_.data(), input_elements_);
            input_data = input_f16_.data();
        } else if (input_encoding_ == 2) {
            float maximum = 0.0f;
            for (float value : input) {
                maximum = std::max(maximum, std::abs(value));
            }
            const float scale = std::max(maximum / 127.0f, 1e-12f);
            const float inverse_scale = 1.0f / scale;
            memcpy(input_i8_.data(), &scale, sizeof(scale));
            int8_t * values = (int8_t *) (
                    input_i8_.data() + sizeof(scale));
            for (size_t index = 0; index < input_elements_; ++index) {
                const float scaled = input[index] * inverse_scale;
                const int quantized = std::max(
                        -127, std::min(
                                127,
                                (int) (scaled +
                                        (scaled >= 0.0f ? 0.5f : -0.5f))));
                values[index] = (int8_t) quantized;
            }
            input_data = input_i8_.data();
        }
        request.input_hash =
                s41_fast_hash_bytes(input_data, request.input_bytes);
        request.weight_hash = weight_hash_;
        memcpy(request_data_.data(), &request, sizeof(request));
        memcpy(
                request_data_.data() + sizeof(request),
                input_data, request.input_bytes);
        metrics.prepare_ms = elapsed_ms(prepare_started);

        auto started = steady_clock::now();
        bool ok = usb_transfer(
                handle_, 0x01, request_data_.data(), request_data_.size());
        metrics.usb_out_ms = elapsed_ms(started);
        if (output_started != nullptr) {
            output_started->store(true, std::memory_order_release);
        }
        return ok;
    }

    bool finish(
            uint32_t request_id,
            std::vector<float> & output,
            phone_metrics & metrics) {
        auto started = steady_clock::now();
        bool ok = usb_transfer(
                handle_, 0x81,
                response_data_.data(), response_data_.size());
        metrics.usb_in_ms = elapsed_ms(started);
        if (!ok) {
            metrics.total_ms = elapsed_ms(submitted_);
            return false;
        }

        const auto validate_started = steady_clock::now();
        s41_ffn_response response = {};
        memcpy(&response, response_data_.data(), sizeof(response));
        const size_t output_bytes =
                (f16_output_ ? sizeof(ggml_fp16_t) : sizeof(float)) *
                output_elements_;
        if (response.magic != S41_FFN_RESPONSE_MAGIC ||
            response.version != S41_FFN_PROTOCOL_VERSION ||
            response.status != 0 ||
            response.request_id != request_id ||
            response.k != output_elements_ ||
            response.output_bytes != output_bytes ||
            response.weight_hash != weight_hash_) {
            fprintf(stderr,
                    "[activation-host] invalid phone response\n");
            ok = false;
        } else {
            const void * response_output =
                    response_data_.data() + sizeof(response);
            if (response.output_hash !=
                s41_fast_hash_bytes(response_output, output_bytes)) {
                fprintf(stderr,
                        "[activation-host] response hash mismatch\n");
                ok = false;
            } else if (f16_output_) {
                ggml_fp16_to_fp32_row(
                        (const ggml_fp16_t *) response_output,
                        output.data(), output_elements_);
            } else {
                memcpy(output.data(), response_output, output_bytes);
            }
        }
        metrics.validate_ms = elapsed_ms(validate_started);
        metrics.total_ms = elapsed_ms(submitted_);
        return ok;
    }

    const void * encoded_output_data() const {
        return response_data_.data() + sizeof(s41_ffn_response);
    }

    size_t encoded_output_bytes() const {
        return (f16_output_ ? sizeof(ggml_fp16_t) : sizeof(float)) *
                output_elements_;
    }

private:
    libusb_device_handle * handle_;
    ggml_type type_;
    uint32_t k_;
    uint32_t n_ff_;
    uint32_t offset_;
    uint32_t count_;
    uint16_t batch_;
    uint64_t weight_hash_;
    int input_encoding_;
    bool f16_output_;
    bool residual_output_;
    size_t input_elements_;
    size_t output_elements_;
    size_t input_bytes_;
    std::vector<uint8_t> request_data_;
    std::vector<uint8_t> response_data_;
    std::vector<ggml_fp16_t> input_f16_;
    std::vector<uint8_t> input_i8_;
    steady_clock::time_point submitted_;
};

class spin_phone_client {
public:
    explicit spin_phone_client(phone_client & client)
        : client_(client),
          worker_(&spin_phone_client::work_loop, this) {
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
            std::vector<float> & output) {
        while (state_.load(std::memory_order_acquire) != 0) {
            pause();
        }
        request_id_ = request_id;
        input_ = &input;
        output_ = &output;
        output_started_.store(false, std::memory_order_relaxed);
        const auto submitted = steady_clock::now();
        state_.store(1, std::memory_order_release);
        return submitted;
    }

    void wait_output_started() {
        while (!output_started_.load(std::memory_order_acquire)) {
            pause();
        }
    }

    bool wait(
            phone_metrics & metrics,
            steady_clock::time_point & completed) {
        while (state_.load(std::memory_order_acquire) != 2) {
            pause();
        }
        metrics = metrics_;
        completed = completed_;
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
            if (state == 0) {
                pause();
                continue;
            }
            if (state == 3) {
                return;
            }
            if (state != 1) {
                pause();
                continue;
            }
            ok_ = client_.run(
                    request_id_, *input_, *output_, metrics_,
                    &output_started_);
            completed_ = steady_clock::now();
            state_.store(2, std::memory_order_release);
        }
    }

    phone_client & client_;
    std::atomic<int> state_ = 0;
    std::atomic<bool> output_started_ = false;
    uint32_t request_id_ = 0;
    const std::vector<float> * input_ = nullptr;
    std::vector<float> * output_ = nullptr;
    phone_metrics metrics_;
    steady_clock::time_point completed_;
    bool ok_ = false;
    std::thread worker_;
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
    const double relative_l2 = reference_squared > 0.0
            ? std::sqrt(error_squared / reference_squared)
            : INFINITY;
    return {
        relative_l2,
        max_absolute,
        reference_argmax,
        candidate_argmax,
        non_finite,
    };
}

static size_t matching_row_argmax(
        const std::vector<float> & reference,
        const std::vector<float> & candidate,
        size_t width,
        size_t rows) {
    if (reference.size() != candidate.size() ||
        reference.size() != width * rows) {
        return 0;
    }
    size_t matches = 0;
    for (size_t row = 0; row < rows; ++row) {
        const size_t begin = row * width;
        size_t reference_argmax = begin;
        size_t candidate_argmax = begin;
        for (size_t index = begin + 1; index < begin + width; ++index) {
            if (reference[index] > reference[reference_argmax]) {
                reference_argmax = index;
            }
            if (candidate[index] > candidate[candidate_argmax]) {
                candidate_argmax = index;
            }
        }
        matches += reference_argmax == candidate_argmax ? 1 : 0;
    }
    return matches;
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

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 11) {
        fprintf(stderr,
                "usage: %s <iters> <phone_cols> [weight_type] [shape] "
                "[input_type] [mode] [output_type] [late_path] [batch] "
                "[result]\n",
                argv[0]);
        return 2;
    }

    const int iterations = atoi(argv[1]);
    const int64_t phone_columns = atoll(argv[2]);
    ggml_type weight_type = GGML_TYPE_Q8_0;
    if (argc > 3 && !parse_type(argv[3], weight_type)) {
        fprintf(stderr, "[activation-host] unknown type '%s'\n", argv[3]);
        return 2;
    }
    const std::string shape = argc > 4 ? argv[4] : "qwen3_14b";
    int64_t k = 5120;
    int64_t n_ff = 17408;
    if (shape == "gemma4_12b") {
        k = 3840;
        n_ff = 15360;
    } else if (shape != "qwen3_14b") {
        fprintf(stderr, "[activation-host] unknown shape '%s'\n",
                shape.c_str());
        return 2;
    }
    const std::string input_type = argc > 5 ? argv[5] : "f32";
    if (input_type != "f32" &&
        input_type != "f16" &&
        input_type != "i8") {
        fprintf(stderr, "[activation-host] unknown input type '%s'\n",
                input_type.c_str());
        return 2;
    }
    const int input_encoding =
            input_type == "f16" ? 1 : (input_type == "i8" ? 2 : 0);
    const std::string mode = argc > 6 ? argv[6] : "corun";
    if (mode != "corun" &&
        mode != "spin_corun" &&
        mode != "phone_first_corun" &&
        mode != "steady_corun" &&
        mode != "cuda_only" &&
        mode != "cuda_profile" &&
        mode != "cuda_energy" &&
        mode != "split_energy" &&
        mode != "phone_only" &&
        mode != "phone_only_gap") {
        fprintf(stderr, "[activation-host] unknown mode '%s'\n",
                mode.c_str());
        return 2;
    }
    const std::string output_type = argc > 7 ? argv[7] : "f32";
    if (output_type != "f32" && output_type != "f16") {
        fprintf(stderr, "[activation-host] unknown output type '%s'\n",
                output_type.c_str());
        return 2;
    }
    const std::string late_path =
            argc > 8 ? argv[8] : "f32_async";
    const int batch = argc > 9 ? atoi(argv[9]) : 1;
    const std::string result_name =
            argc > 10 ? argv[10] : "activation";
    if (result_name != "activation" && result_name != "residual") {
        fprintf(stderr, "[activation-host] unknown result '%s'\n",
                result_name.c_str());
        return 2;
    }
    const bool residual_output = result_name == "residual";
    if (late_path != "f32_sync" &&
        late_path != "f32_async" &&
        late_path != "f32_dual" &&
        late_path != "f16_sync" &&
        late_path != "f16_async" &&
        late_path != "host_sum") {
        fprintf(stderr, "[activation-host] unknown late path '%s'\n",
                late_path.c_str());
        return 2;
    }
    if ((residual_output && late_path != "host_sum") ||
        (!residual_output && late_path == "host_sum")) {
        fprintf(stderr,
                "[activation-host] result and late path are incompatible\n");
        return 2;
    }
    const bool late_dual = late_path == "f32_dual";
    const bool late_f16 = late_path.rfind("f16_", 0) == 0;
    const bool late_async =
            late_dual ||
            (late_path.size() >= 6 &&
             late_path.compare(late_path.size() - 6, 6, "_async") == 0);
    if (late_f16 && output_type != "f16") {
        fprintf(stderr,
                "[activation-host] f16 late path requires f16 output\n");
        return 2;
    }
    if (late_dual &&
        mode != "steady_corun" &&
        mode != "split_energy" &&
        mode != "cuda_profile") {
        fprintf(stderr,
                "[activation-host] f32_dual requires steady_corun, "
                "split_energy, or cuda_profile\n");
        return 2;
    }
    if (residual_output && mode == "cuda_profile") {
        fprintf(stderr,
                "[activation-host] residual cuda_profile is unsupported\n");
        return 2;
    }
    const int64_t gpu_columns = n_ff - phone_columns;
    const int64_t block = ggml_blck_size(weight_type);
    if (iterations <= 0 || phone_columns <= 0 || gpu_columns <= 0 ||
        batch <= 0 || batch > S41_FFN_MAX_BATCH ||
        (weight_type != GGML_TYPE_Q4_0 &&
         weight_type != GGML_TYPE_Q8_0) ||
        k % block != 0 || n_ff % block != 0 ||
        gpu_columns % block != 0 || phone_columns % block != 0) {
        fprintf(stderr, "[activation-host] invalid configuration\n");
        return 2;
    }
    const size_t batch_size = (size_t) batch;
    const size_t output_elements = (size_t) k * batch_size;
    const size_t phone_elements = (size_t) phone_columns * batch_size;
    const size_t phone_result_elements =
            residual_output ? output_elements : phone_elements;

    ggml_backend_dev_t gpu_device = nullptr;
    ggml_backend_t backend = nullptr;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (ggml_backend_dev_type(device) ==
            GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_device = device;
            backend = ggml_backend_dev_init(device, nullptr);
            fprintf(stderr, "[activation-host] GPU=%s\n",
                    ggml_backend_dev_description(device));
            break;
        }
    }
    if (backend == nullptr) {
        fprintf(stderr, "[activation-host] no GPU backend found\n");
        return 1;
    }

    ggml_init_params parameters = {};
    parameters.mem_size =
            ggml_tensor_overhead() * 96 + ggml_graph_overhead() * 8;
    parameters.no_alloc = true;
    ggml_context * context = ggml_init(parameters);
    if (context == nullptr) {
        return 1;
    }

    ggml_tensor * input =
            ggml_new_tensor_2d(context, GGML_TYPE_F32, k, batch);

    ggml_tensor * full_gate =
            ggml_new_tensor_2d(context, weight_type, k, n_ff);
    ggml_tensor * full_up =
            ggml_new_tensor_2d(context, weight_type, k, n_ff);
    ggml_tensor * full_down =
            ggml_new_tensor_2d(context, weight_type, n_ff, k);
    ggml_tensor * full_gate_output =
            ggml_mul_mat(context, full_gate, input);
    ggml_tensor * full_up_output =
            ggml_mul_mat(context, full_up, input);
    ggml_tensor * full_activation = ggml_mul(
            context, ggml_silu(context, full_gate_output), full_up_output);
    ggml_tensor * full_output =
            ggml_mul_mat(context, full_down, full_activation);

    ggml_tensor * prefix_gate =
            ggml_new_tensor_2d(context, weight_type, k, gpu_columns);
    ggml_tensor * prefix_up =
            ggml_new_tensor_2d(context, weight_type, k, gpu_columns);
    ggml_tensor * prefix_down =
            ggml_new_tensor_2d(context, weight_type, gpu_columns, k);
    ggml_tensor * prefix_gate_output =
            ggml_mul_mat(context, prefix_gate, input);
    ggml_tensor * prefix_up_output =
            ggml_mul_mat(context, prefix_up, input);
    ggml_tensor * prefix_activation = ggml_mul(
            context, ggml_silu(context, prefix_gate_output),
            prefix_up_output);
    ggml_tensor * prefix_output =
            ggml_mul_mat(context, prefix_down, prefix_activation);

    ggml_tensor * oracle_gate =
            ggml_new_tensor_2d(context, weight_type, k, phone_columns);
    ggml_tensor * oracle_up =
            ggml_new_tensor_2d(context, weight_type, k, phone_columns);
    ggml_tensor * oracle_gate_output =
            ggml_mul_mat(context, oracle_gate, input);
    ggml_tensor * oracle_up_output =
            ggml_mul_mat(context, oracle_up, input);
    ggml_tensor * oracle_activation = ggml_mul(
            context, ggml_silu(context, oracle_gate_output),
            oracle_up_output);

    ggml_tensor * returned_activation = ggml_new_tensor_2d(
            context, late_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32,
            phone_columns, batch);
    ggml_tensor * suffix_down =
            ggml_new_tensor_2d(context, weight_type, phone_columns, k);
    ggml_tensor * returned_down_output =
            ggml_mul_mat(context, suffix_down, returned_activation);
    ggml_tensor * combined_output =
            ggml_add(context, returned_down_output, prefix_output);

    ggml_tensor * oracle_down_output =
            ggml_mul_mat(context, suffix_down, oracle_activation);
    ggml_tensor * local_partition_output =
            ggml_add(context, prefix_output, oracle_down_output);
    ggml_tensor * oracle_phone_output =
            residual_output ? oracle_down_output : oracle_activation;

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        fprintf(stderr, "[activation-host] backend allocation failed\n");
        return 1;
    }

    fprintf(stderr, "[activation-host] initializing weights\n");
    uint64_t phone_weight_hash = S41_HASH64_OFFSET;
    if (!initialize_weight(
                full_gate, weight_type, S41_WEIGHT_GATE, 0, 0) ||
        !initialize_weight(
                full_up, weight_type, S41_WEIGHT_UP, 0, 0) ||
        !initialize_weight(
                full_down, weight_type, S41_WEIGHT_DOWN, 0, 0) ||
        !initialize_weight(
                prefix_gate, weight_type, S41_WEIGHT_GATE, 0, 0) ||
        !initialize_weight(
                prefix_up, weight_type, S41_WEIGHT_UP, 0, 0) ||
        !initialize_weight(
                prefix_down, weight_type, S41_WEIGHT_DOWN, 0, 0) ||
        !initialize_weight(
                oracle_gate, weight_type, S41_WEIGHT_GATE,
                (uint32_t) gpu_columns, 0, &phone_weight_hash) ||
        !initialize_weight(
                oracle_up, weight_type, S41_WEIGHT_UP,
                (uint32_t) gpu_columns, 0, &phone_weight_hash) ||
        !initialize_weight(
                suffix_down, weight_type, S41_WEIGHT_DOWN,
                0, (uint32_t) gpu_columns,
                residual_output ? &phone_weight_hash : nullptr)) {
        return 1;
    }

    std::vector<float> input_data(output_elements);
    for (size_t index = 0; index < input_data.size(); ++index) {
        input_data[index] = s41_activation_value((uint32_t) index);
    }
    ggml_backend_tensor_set(
            input, input_data.data(), 0,
            input_data.size() * sizeof(float));

    ggml_cgraph * full_graph =
            ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(full_graph, full_output);
    ggml_cgraph * prefix_graph =
            ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(prefix_graph, prefix_output);
    ggml_cgraph * oracle_phone_graph =
            ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(oracle_phone_graph, oracle_phone_output);
    ggml_cgraph * local_partition_graph =
            ggml_new_graph_custom(context, 32, false);
    ggml_build_forward_expand(
            local_partition_graph, local_partition_output);
    ggml_cgraph * late_graph =
            ggml_new_graph_custom(context, 8, false);
    // Build the completed prefix as an external input to the late graph.
    const enum ggml_op prefix_output_op = prefix_output->op;
    ggml_tensor * prefix_output_sources[GGML_MAX_SRC];
    memcpy(
            prefix_output_sources, prefix_output->src,
            sizeof(prefix_output_sources));
    prefix_output->op = GGML_OP_NONE;
    memset(prefix_output->src, 0, sizeof(prefix_output->src));
    ggml_build_forward_expand(late_graph, combined_output);
    prefix_output->op = prefix_output_op;
    memcpy(
            prefix_output->src, prefix_output_sources,
            sizeof(prefix_output_sources));

    ggml_backend_t dual_backend = nullptr;
    ggml_context * dual_context = nullptr;
    ggml_backend_buffer_t dual_buffer = nullptr;
    ggml_backend_buffer_t dual_host_buffer = nullptr;
    float * dual_prefix_host = nullptr;
    float * dual_suffix_host = nullptr;
    ggml_tensor * dual_returned_activation = nullptr;
    ggml_tensor * dual_suffix_down = nullptr;
    ggml_tensor * dual_suffix_output = nullptr;
    ggml_cgraph * dual_graph = nullptr;
    if (late_dual) {
        dual_backend = ggml_backend_dev_init(gpu_device, nullptr);
        if (dual_backend == nullptr) {
            fprintf(stderr,
                    "[activation-host] second GPU backend unavailable\n");
            return 1;
        }
        ggml_init_params dual_parameters = {};
        dual_parameters.mem_size =
                ggml_tensor_overhead() * 8 + ggml_graph_overhead() * 2;
        dual_parameters.no_alloc = true;
        dual_context = ggml_init(dual_parameters);
        if (dual_context == nullptr) {
            return 1;
        }
        dual_returned_activation = ggml_new_tensor_2d(
                dual_context, GGML_TYPE_F32, phone_columns, batch);
        dual_suffix_down = ggml_new_tensor_2d(
                dual_context, weight_type, phone_columns, k);
        dual_suffix_output = ggml_mul_mat(
                dual_context, dual_suffix_down, dual_returned_activation);
        dual_buffer =
                ggml_backend_alloc_ctx_tensors(dual_context, dual_backend);
        if (dual_buffer == nullptr) {
            fprintf(stderr,
                    "[activation-host] second GPU allocation failed\n");
            return 1;
        }
        ggml_backend_buffer_type_t host_buffer_type =
                ggml_backend_dev_host_buffer_type(gpu_device);
        if (host_buffer_type == nullptr) {
            fprintf(stderr,
                    "[activation-host] pinned host buffer unavailable\n");
            return 1;
        }
        dual_host_buffer = ggml_backend_buft_alloc_buffer(
                host_buffer_type, 2 * output_elements * sizeof(float));
        if (dual_host_buffer == nullptr) {
            fprintf(stderr,
                    "[activation-host] pinned host allocation failed\n");
            return 1;
        }
        dual_prefix_host =
                (float *) ggml_backend_buffer_get_base(dual_host_buffer);
        dual_suffix_host = dual_prefix_host + output_elements;
        if (!initialize_weight(
                    dual_suffix_down, weight_type, S41_WEIGHT_DOWN,
                    0, (uint32_t) gpu_columns)) {
            return 1;
        }
        dual_graph = ggml_new_graph_custom(dual_context, 8, false);
        ggml_build_forward_expand(dual_graph, dual_suffix_output);
    }

    fprintf(stderr,
            "[activation-host] shape=%s K=%lld NFF=%lld split=%lld+%lld "
            "batch=%d "
            "type=%s input_type=%s phone_weight_hash=%016llx "
            "output_type=%s result=%s late_path=%s nodes=%d/%d/%d\n",
            shape.c_str(), (long long) k, (long long) n_ff,
            (long long) gpu_columns, (long long) phone_columns,
            batch,
            ggml_type_name(weight_type), input_type.c_str(),
            (unsigned long long) phone_weight_hash,
            output_type.c_str(), result_name.c_str(), late_path.c_str(),
            ggml_graph_n_nodes(full_graph),
            ggml_graph_n_nodes(prefix_graph),
            ggml_graph_n_nodes(late_graph));

    if (mode == "cuda_only" || mode == "cuda_energy") {
        std::vector<float> cuda_only_result(output_elements);
        std::vector<double> cuda_only_times;
        for (int index = 0; index < 20; ++index) {
            const enum ggml_status status =
                    ggml_backend_graph_compute(backend, full_graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr,
                        "[activation-host] compute failed: %d\n",
                        (int) status);
                return 3;
            }
            ggml_backend_tensor_get(
                    full_output, cuda_only_result.data(), 0,
                    cuda_only_result.size() * sizeof(float));
        }
        if (mode == "cuda_energy") {
            print_energy_marker("START", "cuda", iterations);
        }
        for (int index = 0; index < iterations; ++index) {
            const auto started = steady_clock::now();
            const enum ggml_status status =
                    ggml_backend_graph_compute(backend, full_graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr,
                        "[activation-host] compute failed: %d\n",
                        (int) status);
                return 3;
            }
            ggml_backend_tensor_get(
                    full_output, cuda_only_result.data(), 0,
                    cuda_only_result.size() * sizeof(float));
            cuda_only_times.push_back(elapsed_ms(started));
        }
        if (mode == "cuda_energy") {
            print_energy_marker("END", "cuda", iterations);
        }
        printf("CUDA_ONLY shape=%s type=%s batch=%d rows=%lld n=%d ",
                shape.c_str(), ggml_type_name(weight_type), batch,
                (long long) iterations * batch, iterations);
        print_metrics("cuda_full", cuda_only_times);
        printf("sample=%.9g non_finite=%zu\n",
                cuda_only_result[0],
                std::count_if(
                        cuda_only_result.begin(), cuda_only_result.end(),
                        [](float value) { return !std::isfinite(value); }));
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        ggml_backend_free(backend);
        ggml_quantize_free();
        return 0;
    }

    if (mode == "cuda_profile") {
        std::vector<float> oracle_data(phone_elements);
        std::vector<float> full_result(output_elements);
        std::vector<float> split_result(output_elements);
        const enum ggml_status oracle_status =
                ggml_backend_graph_compute(
                        backend, oracle_phone_graph);
        if (oracle_status != GGML_STATUS_SUCCESS) {
            fprintf(stderr,
                    "[activation-host] oracle compute failed: %d\n",
                    (int) oracle_status);
            return 3;
        }
        ggml_backend_tensor_get(
                oracle_activation, oracle_data.data(), 0,
                oracle_data.size() * sizeof(float));
        if (late_dual) {
            const char * delay_text = getenv("S41_PROFILE_PHONE_US");
            const long profile_delay_us =
                    delay_text != nullptr ? atol(delay_text) : 0;
            if (profile_delay_us < 0 || profile_delay_us > 10000000) {
                fprintf(stderr,
                        "[activation-host] invalid profile phone delay\n");
                return 2;
            }
            ggml_backend_tensor_set(
                    dual_returned_activation, oracle_data.data(), 0,
                    oracle_data.size() * sizeof(float));

            auto compute_backend = [](
                    ggml_backend_t profile_backend,
                    ggml_cgraph * graph) {
                const enum ggml_status status =
                        ggml_backend_graph_compute(
                                profile_backend, graph);
                if (status != GGML_STATUS_SUCCESS) {
                    fprintf(stderr,
                            "[activation-host] profile compute failed: %d\n",
                            (int) status);
                    return false;
                }
                return true;
            };
            for (int index = 0; index < 20; ++index) {
                if (!compute_backend(backend, full_graph) ||
                    !compute_backend(backend, prefix_graph) ||
                    !compute_backend(dual_backend, dual_graph)) {
                    return 3;
                }
            }

            std::vector<float> suffix_result(output_elements);
            std::vector<double> full_times;
            std::vector<double> prefix_times;
            std::vector<double> late_times;
            std::vector<double> delayed_dual_times;
            std::vector<double> launch_delay_times;
            for (int index = 0; index < iterations; ++index) {
                auto started = steady_clock::now();
                if (!compute_backend(backend, full_graph)) {
                    return 3;
                }
                ggml_backend_tensor_get(
                        full_output, full_result.data(), 0,
                        full_result.size() * sizeof(float));
                full_times.push_back(elapsed_ms(started));

                started = steady_clock::now();
                if (!compute_backend(backend, prefix_graph)) {
                    return 3;
                }
                prefix_times.push_back(elapsed_ms(started));

                started = steady_clock::now();
                ggml_backend_tensor_set_async(
                        dual_backend, dual_returned_activation,
                        oracle_data.data(), 0,
                        oracle_data.size() * sizeof(float));
                if (!compute_backend(dual_backend, dual_graph)) {
                    return 3;
                }
                ggml_backend_tensor_get(
                        dual_suffix_output, suffix_result.data(), 0,
                        suffix_result.size() * sizeof(float));
                late_times.push_back(elapsed_ms(started));

                started = steady_clock::now();
                const enum ggml_status prefix_status =
                        ggml_backend_graph_compute_async(
                                backend, prefix_graph);
                if (prefix_status != GGML_STATUS_SUCCESS) {
                    fprintf(stderr,
                            "[activation-host] profile prefix launch "
                            "failed: %d\n",
                            (int) prefix_status);
                    return 3;
                }
                for (;;) {
                    const double elapsed_us =
                            elapsed_ms(started) * 1000.0;
                    if (elapsed_us >= (double) profile_delay_us) {
                        break;
                    }
                }
                launch_delay_times.push_back(elapsed_ms(started));
                ggml_backend_tensor_set_async(
                        dual_backend, dual_returned_activation,
                        oracle_data.data(), 0,
                        oracle_data.size() * sizeof(float));
                const enum ggml_status suffix_status =
                        ggml_backend_graph_compute_async(
                                dual_backend, dual_graph);
                if (suffix_status != GGML_STATUS_SUCCESS) {
                    fprintf(stderr,
                            "[activation-host] profile suffix launch "
                            "failed: %d\n",
                            (int) suffix_status);
                    return 3;
                }
                ggml_backend_synchronize(backend);
                ggml_backend_synchronize(dual_backend);
                ggml_backend_tensor_get_async(
                        backend, prefix_output, dual_prefix_host, 0,
                        output_elements * sizeof(float));
                ggml_backend_tensor_get_async(
                        dual_backend, dual_suffix_output,
                        dual_suffix_host, 0,
                        output_elements * sizeof(float));
                ggml_backend_synchronize(backend);
                ggml_backend_synchronize(dual_backend);
                for (size_t element = 0;
                     element < output_elements; ++element) {
                    split_result[element] =
                            dual_prefix_host[element] +
                            dual_suffix_host[element];
                }
                delayed_dual_times.push_back(elapsed_ms(started));
            }
            const error_metrics profile_error =
                    compare(full_result, split_result);
            const size_t row_argmax_matches = matching_row_argmax(
                    full_result, split_result, (size_t) k, batch_size);
            printf("CUDA_DUAL_PROFILE shape=%s type=%s batch=%d rows=%lld "
                   "phone_cols=%lld n=%d profile_delay_us=%ld ",
                    shape.c_str(), ggml_type_name(weight_type), batch,
                    (long long) iterations * batch,
                    (long long) phone_columns, iterations,
                    profile_delay_us);
            print_metrics("cuda_full", full_times);
            print_metrics("cuda_prefix", prefix_times);
            print_metrics("cuda_late", late_times);
            print_metrics("launch_delay", launch_delay_times);
            print_metrics("delayed_dual", delayed_dual_times);
            printf("rel_l2=%.9g row_argmax=%zu/%d non_finite=%zu\n",
                    profile_error.relative_l2, row_argmax_matches, batch,
                    profile_error.non_finite);
            const bool profile_pass =
                    profile_error.non_finite == 0 &&
                    profile_error.relative_l2 <= 0.0001 &&
                    row_argmax_matches == batch_size;
            ggml_backend_buffer_free(dual_host_buffer);
            ggml_backend_buffer_free(dual_buffer);
            ggml_free(dual_context);
            ggml_backend_free(dual_backend);
            ggml_backend_buffer_free(buffer);
            ggml_free(context);
            ggml_backend_free(backend);
            ggml_quantize_free();
            return profile_pass ? 0 : 4;
        }
        if (late_f16) {
            std::vector<ggml_fp16_t> oracle_f16(phone_elements);
            ggml_fp32_to_fp16_row(
                    oracle_data.data(), oracle_f16.data(), phone_elements);
            ggml_backend_tensor_set(
                    returned_activation, oracle_f16.data(), 0,
                    oracle_f16.size() * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(
                    returned_activation, oracle_data.data(), 0,
                    oracle_data.size() * sizeof(float));
        }

        auto compute_profile = [&](ggml_cgraph * graph) {
            const enum ggml_status status =
                    ggml_backend_graph_compute(backend, graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr,
                        "[activation-host] profile compute failed: %d\n",
                        (int) status);
                return false;
            }
            return true;
        };
        for (int index = 0; index < 20; ++index) {
            if (!compute_profile(full_graph) ||
                !compute_profile(prefix_graph) ||
                !compute_profile(late_graph)) {
                return 3;
            }
        }

        std::vector<double> full_times;
        std::vector<double> prefix_times;
        std::vector<double> late_times;
        std::vector<double> sequential_split_times;
        for (int index = 0; index < iterations; ++index) {
            auto started = steady_clock::now();
            if (!compute_profile(full_graph)) {
                return 3;
            }
            ggml_backend_tensor_get(
                    full_output, full_result.data(), 0,
                    full_result.size() * sizeof(float));
            full_times.push_back(elapsed_ms(started));

            started = steady_clock::now();
            if (!compute_profile(prefix_graph)) {
                return 3;
            }
            prefix_times.push_back(elapsed_ms(started));

            started = steady_clock::now();
            if (!compute_profile(late_graph)) {
                return 3;
            }
            ggml_backend_tensor_get(
                    combined_output, split_result.data(), 0,
                    split_result.size() * sizeof(float));
            late_times.push_back(elapsed_ms(started));

            started = steady_clock::now();
            if (!compute_profile(prefix_graph) ||
                !compute_profile(late_graph)) {
                return 3;
            }
            ggml_backend_tensor_get(
                    combined_output, split_result.data(), 0,
                    split_result.size() * sizeof(float));
            sequential_split_times.push_back(elapsed_ms(started));
        }
        const error_metrics profile_error =
                compare(full_result, split_result);
        const size_t row_argmax_matches = matching_row_argmax(
                full_result, split_result, (size_t) k, batch_size);
        printf("CUDA_PROFILE shape=%s type=%s batch=%d rows=%lld "
               "phone_cols=%lld n=%d ",
                shape.c_str(), ggml_type_name(weight_type), batch,
                (long long) iterations * batch,
                (long long) phone_columns, iterations);
        print_metrics("cuda_full", full_times);
        print_metrics("cuda_prefix", prefix_times);
        print_metrics("cuda_late", late_times);
        print_metrics("sequential_split", sequential_split_times);
        printf("rel_l2=%.9g row_argmax=%zu/%d non_finite=%zu\n",
                profile_error.relative_l2, row_argmax_matches, batch,
                profile_error.non_finite);
        const bool profile_pass =
                profile_error.non_finite == 0 &&
                profile_error.relative_l2 <= 0.0001 &&
                row_argmax_matches == batch_size;
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        ggml_backend_free(backend);
        ggml_quantize_free();
        return profile_pass ? 0 : 4;
    }

    libusb_context * usb_context = nullptr;
    if (libusb_init(&usb_context) != 0) {
        fprintf(stderr, "[activation-host] libusb initialization failed\n");
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
        fprintf(stderr, "[activation-host] claim failed: %s\n",
                libusb_error_name(claim_status));
        return 1;
    }

    auto compute = [&](ggml_cgraph * graph) {
        const enum ggml_status status =
                ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[activation-host] compute failed: %d\n",
                    (int) status);
            return false;
        }
        return true;
    };

    std::vector<float> full_result(output_elements);
    std::vector<float> local_result(output_elements);
    std::vector<float> treatment_result(output_elements);
    std::vector<float> dual_suffix_result(output_elements);
    std::vector<float> oracle_phone_data(phone_result_elements);
    std::vector<float> phone_result_data(phone_result_elements);

    if (!compute(full_graph) ||
        !compute(local_partition_graph) ||
        !compute(oracle_phone_graph)) {
        return 3;
    }
    ggml_backend_tensor_get(
            full_output, full_result.data(), 0,
            full_result.size() * sizeof(float));
    ggml_backend_tensor_get(
            local_partition_output, local_result.data(), 0,
            local_result.size() * sizeof(float));
    ggml_backend_tensor_get(
            oracle_phone_output, oracle_phone_data.data(), 0,
            oracle_phone_data.size() * sizeof(float));

    phone_client client(
            usb, weight_type, (uint32_t) k, (uint32_t) n_ff,
            (uint32_t) gpu_columns, (uint32_t) phone_columns,
            phone_weight_hash, input_encoding, output_type == "f16",
            (uint16_t) batch, residual_output);
    std::unique_ptr<spin_phone_client> spin_client;
    if (mode == "spin_corun") {
        spin_client = std::make_unique<spin_phone_client>(client);
    }
    uint32_t request_id = 1;

    auto run_full = [&](double & total_ms) {
        const auto started = steady_clock::now();
        if (!compute(full_graph)) {
            return false;
        }
        ggml_backend_tensor_get(
                full_output, full_result.data(), 0,
                full_result.size() * sizeof(float));
        total_ms = elapsed_ms(started);
        return true;
    };

    auto run_prefix_control = [&](double & total_ms) {
        const auto started = steady_clock::now();
        if (!compute(prefix_graph)) {
            return false;
        }
        total_ms = elapsed_ms(started);
        return true;
    };

    struct treatment_metrics {
        double total_ms;
        double prefix_ms;
        double deadline_ms;
        double exposed_wait_ms;
        double main_wait_ms;
        double late_ms;
        double late_upload_ms;
        double late_compute_ms;
        double late_download_ms;
        phone_metrics phone;
        bool hidden;
    };

    auto run_treatment = [&](
            treatment_metrics & metrics, double prefix_reference_ms) {
        const auto total_started = steady_clock::now();
        steady_clock::time_point phone_finished;
        steady_clock::time_point wait_started;
        steady_clock::time_point wait_finished;
        steady_clock::time_point overlap_started = total_started;
        double deadline_ms = prefix_reference_ms;
        if (mode == "phone_first_corun") {
            const uint32_t current_request_id = request_id++;
            if (!client.begin(
                        current_request_id, input_data, metrics.phone)) {
                return false;
            }
            const auto cuda_started = steady_clock::now();
            const enum ggml_status launch_status =
                    ggml_backend_graph_compute_async(backend, prefix_graph);
            if (launch_status != GGML_STATUS_SUCCESS) {
                fprintf(stderr,
                        "[activation-host] async launch failed: %d\n",
                        (int) launch_status);
                return false;
            }
            if (!client.finish(
                        current_request_id, phone_result_data,
                        metrics.phone)) {
                return false;
            }
            phone_finished = steady_clock::now();
            wait_started = steady_clock::now();
            ggml_backend_synchronize(backend);
            wait_finished = steady_clock::now();
            metrics.prefix_ms = prefix_reference_ms;
            deadline_ms =
                    duration_ms(total_started, cuda_started) +
                    prefix_reference_ms;
        } else if (spin_client != nullptr) {
            const auto submitted = spin_client->submit(
                    request_id++, input_data, phone_result_data);
            overlap_started = submitted;
            const auto cuda_started = steady_clock::now();
            if (!compute(prefix_graph)) {
                return false;
            }
            const auto prefix_finished = steady_clock::now();
            wait_started = steady_clock::now();
            if (!spin_client->wait(metrics.phone, phone_finished)) {
                return false;
            }
            wait_finished = steady_clock::now();
            deadline_ms = duration_ms(submitted, prefix_finished);
            metrics.prefix_ms = duration_ms(cuda_started, prefix_finished);
        } else {
            const enum ggml_status launch_status =
                    ggml_backend_graph_compute_async(backend, prefix_graph);
            if (launch_status != GGML_STATUS_SUCCESS) {
                fprintf(stderr,
                        "[activation-host] async launch failed: %d\n",
                        (int) launch_status);
                return false;
            }
            if (!client.run(
                        request_id++, input_data, phone_result_data,
                        metrics.phone)) {
                return false;
            }
            phone_finished = steady_clock::now();
            wait_started = steady_clock::now();
            if (!late_dual) {
                ggml_backend_synchronize(backend);
            }
            wait_finished = steady_clock::now();
        }

        const auto late_started = steady_clock::now();
        if (residual_output) {
            const auto download_started = steady_clock::now();
            ggml_backend_tensor_get(
                    prefix_output, treatment_result.data(), 0,
                    treatment_result.size() * sizeof(float));
            for (size_t index = 0; index < treatment_result.size(); ++index) {
                treatment_result[index] += phone_result_data[index];
            }
            metrics.late_download_ms = elapsed_ms(download_started);
        } else {
            const auto upload_started = steady_clock::now();
            const void * late_data = late_f16
                    ? client.encoded_output_data()
                    : phone_result_data.data();
            const size_t late_bytes = late_f16
                    ? client.encoded_output_bytes()
                    : phone_result_data.size() * sizeof(float);
            if (late_dual) {
                ggml_backend_tensor_set_async(
                        dual_backend, dual_returned_activation,
                        late_data, 0, late_bytes);
            } else if (late_async) {
                ggml_backend_tensor_set_async(
                        backend, returned_activation, late_data, 0, late_bytes);
            } else {
                ggml_backend_tensor_set(
                        returned_activation, late_data, 0, late_bytes);
            }
            metrics.late_upload_ms = elapsed_ms(upload_started);
            const auto compute_started = steady_clock::now();
            if (late_dual) {
                const enum ggml_status launch_status =
                        ggml_backend_graph_compute_async(
                                dual_backend, dual_graph);
                if (launch_status != GGML_STATUS_SUCCESS) {
                    fprintf(stderr,
                            "[activation-host] second async launch failed: %d\n",
                            (int) launch_status);
                    return false;
                }
                ggml_backend_synchronize(backend);
                ggml_backend_synchronize(dual_backend);
                metrics.late_compute_ms = elapsed_ms(compute_started);
                const auto download_started = steady_clock::now();
                ggml_backend_tensor_get_async(
                        backend, prefix_output, dual_prefix_host, 0,
                        treatment_result.size() * sizeof(float));
                ggml_backend_tensor_get_async(
                        dual_backend, dual_suffix_output, dual_suffix_host, 0,
                        dual_suffix_result.size() * sizeof(float));
                ggml_backend_synchronize(backend);
                ggml_backend_synchronize(dual_backend);
                for (size_t index = 0;
                     index < treatment_result.size(); ++index) {
                    dual_suffix_result[index] = dual_suffix_host[index];
                    treatment_result[index] =
                            dual_prefix_host[index] + dual_suffix_host[index];
                }
                metrics.late_download_ms = elapsed_ms(download_started);
            } else {
                if (!compute(late_graph)) {
                    return false;
                }
                metrics.late_compute_ms = elapsed_ms(compute_started);
                const auto download_started = steady_clock::now();
                ggml_backend_tensor_get(
                        combined_output, treatment_result.data(), 0,
                        treatment_result.size() * sizeof(float));
                metrics.late_download_ms = elapsed_ms(download_started);
            }
        }
        const auto late_finished = steady_clock::now();

        if (spin_client == nullptr && mode != "phone_first_corun") {
            metrics.prefix_ms = prefix_reference_ms;
        }
        metrics.deadline_ms = deadline_ms;
        const double phone_completion_ms =
                duration_ms(overlap_started, phone_finished);
        metrics.exposed_wait_ms = std::max(
                0.0, phone_completion_ms - deadline_ms);
        metrics.main_wait_ms =
                duration_ms(wait_started, wait_finished);
        metrics.late_ms =
                duration_ms(late_started, late_finished);
        metrics.total_ms = duration_ms(total_started, late_finished);
        metrics.hidden = phone_completion_ms <= metrics.deadline_ms;
        return true;
    };

    for (int index = 0; index < 20; ++index) {
        double full_ms = 0.0;
        double prefix_ms = 0.0;
        treatment_metrics treatment = {};
        if (!run_full(full_ms) ||
            !run_prefix_control(prefix_ms) ||
            !run_treatment(treatment, prefix_ms)) {
            return 3;
        }
    }

    double correctness_prefix_ms = 0.0;
    treatment_metrics correctness_treatment = {};
    if (!run_prefix_control(correctness_prefix_ms) ||
        !run_treatment(correctness_treatment, correctness_prefix_ms)) {
        return 3;
    }
    std::vector<float> prefix_debug(output_elements);
    std::vector<float> suffix_debug(output_elements);
    std::vector<float> returned_debug(phone_result_elements);
    ggml_backend_tensor_get(
            prefix_output, prefix_debug.data(), 0,
            prefix_debug.size() * sizeof(float));
    if (residual_output) {
        suffix_debug = phone_result_data;
        returned_debug = phone_result_data;
    } else if (late_dual) {
        suffix_debug = dual_suffix_result;
        returned_debug = phone_result_data;
    } else if (late_f16) {
        ggml_backend_tensor_get(
                returned_down_output, suffix_debug.data(), 0,
                suffix_debug.size() * sizeof(float));
        std::vector<ggml_fp16_t> returned_debug_f16(
                phone_elements);
        ggml_backend_tensor_get(
                returned_activation, returned_debug_f16.data(), 0,
                returned_debug_f16.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(
                returned_debug_f16.data(), returned_debug.data(),
                phone_elements);
    } else {
        ggml_backend_tensor_get(
                returned_down_output, suffix_debug.data(), 0,
                suffix_debug.size() * sizeof(float));
        ggml_backend_tensor_get(
                returned_activation, returned_debug.data(), 0,
                returned_debug.size() * sizeof(float));
    }
    const error_metrics local_error =
            compare(full_result, local_result);
    const error_metrics activation_error =
            compare(oracle_phone_data, phone_result_data);
    const error_metrics treatment_error =
            compare(full_result, treatment_result);
    const size_t local_argmax_matches = matching_row_argmax(
            full_result, local_result, (size_t) k, batch_size);
    const size_t activation_argmax_matches = matching_row_argmax(
            oracle_phone_data, phone_result_data,
            residual_output ? (size_t) k : (size_t) phone_columns,
            batch_size);
    const size_t treatment_argmax_matches = matching_row_argmax(
            full_result, treatment_result, (size_t) k, batch_size);
    printf("CORRECTNESS control=local_partition rel_l2=%.9g max_abs=%.9g "
           "argmax=%zu/%zu row_argmax=%zu/%d non_finite=%zu\n",
            local_error.relative_l2, local_error.max_absolute,
            local_error.reference_argmax, local_error.candidate_argmax,
            local_argmax_matches, batch,
            local_error.non_finite);
    printf("CORRECTNESS component=phone_%s rel_l2=%.9g "
           "max_abs=%.9g argmax=%zu/%zu row_argmax=%zu/%d "
           "non_finite=%zu\n",
            result_name.c_str(),
            activation_error.relative_l2, activation_error.max_absolute,
            activation_error.reference_argmax,
            activation_error.candidate_argmax,
            activation_argmax_matches, batch,
            activation_error.non_finite);
    printf("CORRECTNESS treatment=%s_return rel_l2=%.9g "
           "max_abs=%.9g argmax=%zu/%zu row_argmax=%zu/%d "
           "non_finite=%zu\n",
            result_name.c_str(),
            treatment_error.relative_l2, treatment_error.max_absolute,
            treatment_error.reference_argmax,
            treatment_error.candidate_argmax,
            treatment_argmax_matches, batch,
            treatment_error.non_finite);
    printf("BOUNDARY phone_sample=%.9g tensor_sample=%.9g "
           "oracle_sample=%.9g prefix_sample=%.9g suffix_sample=%.9g "
           "combined_sample=%.9g reference_sample=%.9g\n",
            phone_result_data[0], returned_debug[0],
            oracle_phone_data[0], prefix_debug[0], suffix_debug[0],
            treatment_result[0], full_result[0]);
    const bool correctness_pass =
            local_error.non_finite == 0 &&
            local_error.relative_l2 <= 0.0001 &&
            local_argmax_matches == batch_size &&
            activation_error.non_finite == 0 &&
            activation_error.relative_l2 <= 0.03 &&
            activation_argmax_matches == batch_size &&
            treatment_error.non_finite == 0 &&
            treatment_error.relative_l2 <= 0.005 &&
            treatment_argmax_matches == batch_size;
    if (!correctness_pass) {
        fprintf(stderr, "[activation-host] correctness gate failed\n");
        return 4;
    }

    if (mode == "phone_only" || mode == "phone_only_gap") {
        for (int index = 0; index < 20; ++index) {
            if (mode == "phone_only_gap") {
                std::this_thread::sleep_for(
                        std::chrono::microseconds(1500));
            }
            phone_metrics metrics;
            if (!client.run(
                        request_id++, input_data, phone_result_data,
                        metrics)) {
                return 3;
            }
        }
        std::vector<double> total_times;
        std::vector<double> prepare_times;
        std::vector<double> usb_out_times;
        std::vector<double> usb_in_times;
        std::vector<double> validate_times;
        for (int index = 0; index < iterations; ++index) {
            if (mode == "phone_only_gap") {
                std::this_thread::sleep_for(
                        std::chrono::microseconds(1500));
            }
            phone_metrics metrics;
            if (!client.run(
                        request_id++, input_data, phone_result_data,
                        metrics)) {
                return 3;
            }
            total_times.push_back(metrics.total_ms);
            prepare_times.push_back(metrics.prepare_ms);
            usb_out_times.push_back(metrics.usb_out_ms);
            usb_in_times.push_back(metrics.usb_in_ms);
            validate_times.push_back(metrics.validate_ms);
        }
        printf("PHONE_ONLY shape=%s type=%s input_type=%s output_type=%s "
               "result=%s mode=%s batch=%d rows=%lld "
               "phone_cols=%lld n=%d ",
                shape.c_str(), ggml_type_name(weight_type),
                input_type.c_str(), output_type.c_str(),
                result_name.c_str(), mode.c_str(),
                batch, (long long) iterations * batch,
                (long long) phone_columns, iterations);
        print_metrics("phone_total", total_times);
        print_metrics("prepare", prepare_times);
        print_metrics("usb_out", usb_out_times);
        print_metrics("usb_in_compute", usb_in_times);
        print_metrics("validate", validate_times);
        printf("\n");

        libusb_release_interface(usb, 0);
        libusb_close(usb);
        libusb_exit(usb_context);
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        ggml_backend_free(backend);
        ggml_quantize_free();
        return 0;
    }

    std::vector<double> full_times;
    std::vector<double> total_times;
    std::vector<double> prefix_times;
    std::vector<double> deadline_times;
    std::vector<double> exposed_wait_times;
    std::vector<double> cuda_tail_wait_times;
    std::vector<double> late_times;
    std::vector<double> late_upload_times;
    std::vector<double> late_compute_times;
    std::vector<double> late_download_times;
    std::vector<double> phone_times;
    std::vector<double> prepare_times;
    std::vector<double> usb_out_times;
    std::vector<double> usb_in_times;
    std::vector<double> validate_times;
    size_t hidden_count = 0;
    if (mode == "split_energy") {
        std::vector<double> reference_times;
        for (int index = 0; index < 100; ++index) {
            double prefix_ms = 0.0;
            if (!run_prefix_control(prefix_ms)) {
                return 3;
            }
            reference_times.push_back(prefix_ms);
        }
        const double prefix_reference_ms =
                percentile(reference_times, 0.5);
        print_energy_marker("START", "split", iterations);
        for (int index = 0; index < iterations; ++index) {
            treatment_metrics treatment = {};
            if (!run_treatment(treatment, prefix_reference_ms)) {
                return 3;
            }
            total_times.push_back(treatment.total_ms);
            deadline_times.push_back(treatment.deadline_ms);
            exposed_wait_times.push_back(treatment.exposed_wait_ms);
            late_times.push_back(treatment.late_ms);
            phone_times.push_back(treatment.phone.total_ms);
            hidden_count += treatment.hidden ? 1 : 0;
        }
        print_energy_marker("END", "split", iterations);
        printf("SPLIT_ENERGY shape=%s type=%s result=%s batch=%d "
               "rows=%lld phone_cols=%lld n=%d ",
                shape.c_str(), ggml_type_name(weight_type),
                result_name.c_str(), batch,
                (long long) iterations * batch,
                (long long) phone_columns, iterations);
        print_metrics("split_total", total_times);
        print_metrics("phone_total", phone_times);
        print_metrics("exposed_wait", exposed_wait_times);
        print_metrics("late_cuda", late_times);
        printf("hidden=%zu/%d hidden_fraction=%.6f\n",
                hidden_count, iterations,
                (double) hidden_count / (double) iterations);

        spin_client.reset();
        libusb_release_interface(usb, 0);
        libusb_close(usb);
        libusb_exit(usb_context);
        if (dual_buffer != nullptr) {
            ggml_backend_buffer_free(dual_host_buffer);
            ggml_backend_buffer_free(dual_buffer);
            ggml_free(dual_context);
            ggml_backend_free(dual_backend);
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        ggml_backend_free(backend);
        ggml_quantize_free();
        return 0;
    } else if (mode == "steady_corun") {
        for (int index = 0; index < iterations; ++index) {
            double full_ms = 0.0;
            if (!run_full(full_ms)) {
                return 3;
            }
            full_times.push_back(full_ms);
        }
        for (int index = 0; index < iterations; ++index) {
            double prefix_ms = 0.0;
            if (!run_prefix_control(prefix_ms)) {
                return 3;
            }
            prefix_times.push_back(prefix_ms);
        }
        const double prefix_reference_ms =
                percentile(prefix_times, 0.5);
        for (int index = 0; index < iterations; ++index) {
            treatment_metrics treatment = {};
            if (!run_treatment(treatment, prefix_reference_ms)) {
                return 3;
            }
            total_times.push_back(treatment.total_ms);
            deadline_times.push_back(treatment.deadline_ms);
            exposed_wait_times.push_back(treatment.exposed_wait_ms);
            cuda_tail_wait_times.push_back(treatment.main_wait_ms);
            late_times.push_back(treatment.late_ms);
            late_upload_times.push_back(treatment.late_upload_ms);
            late_compute_times.push_back(treatment.late_compute_ms);
            late_download_times.push_back(treatment.late_download_ms);
            phone_times.push_back(treatment.phone.total_ms);
            prepare_times.push_back(treatment.phone.prepare_ms);
            usb_out_times.push_back(treatment.phone.usb_out_ms);
            usb_in_times.push_back(treatment.phone.usb_in_ms);
            validate_times.push_back(treatment.phone.validate_ms);
            hidden_count += treatment.hidden ? 1 : 0;
        }
    } else {
        for (int index = 0; index < iterations; ++index) {
            double full_ms = 0.0;
            double prefix_ms = 0.0;
            treatment_metrics treatment = {};
            const bool prefix_ok = run_prefix_control(prefix_ms);
            const bool ok = prefix_ok && (index % 2 == 0
                    ? run_full(full_ms) &&
                            run_treatment(treatment, prefix_ms)
                    : run_treatment(treatment, prefix_ms) &&
                            run_full(full_ms));
            if (!ok) {
                return 3;
            }
            full_times.push_back(full_ms);
            total_times.push_back(treatment.total_ms);
            prefix_times.push_back(treatment.prefix_ms);
            deadline_times.push_back(treatment.deadline_ms);
            exposed_wait_times.push_back(treatment.exposed_wait_ms);
            cuda_tail_wait_times.push_back(treatment.main_wait_ms);
            late_times.push_back(treatment.late_ms);
            late_upload_times.push_back(treatment.late_upload_ms);
            late_compute_times.push_back(treatment.late_compute_ms);
            late_download_times.push_back(treatment.late_download_ms);
            phone_times.push_back(treatment.phone.total_ms);
            prepare_times.push_back(treatment.phone.prepare_ms);
            usb_out_times.push_back(treatment.phone.usb_out_ms);
            usb_in_times.push_back(treatment.phone.usb_in_ms);
            validate_times.push_back(treatment.phone.validate_ms);
            hidden_count += treatment.hidden ? 1 : 0;
        }
    }

    printf("RESULT shape=%s type=%s input_type=%s output_type=%s "
           "result=%s mode=%s late_path=%s batch=%d rows=%lld "
           "phone_cols=%lld n=%d ",
            shape.c_str(), ggml_type_name(weight_type), input_type.c_str(),
            output_type.c_str(), result_name.c_str(),
            mode.c_str(), late_path.c_str(), batch,
            (long long) iterations * batch,
            (long long) phone_columns, iterations);
    print_metrics("cuda_full", full_times);
    print_metrics("split_total", total_times);
    print_metrics("cuda_prefix", prefix_times);
    print_metrics("phone_deadline", deadline_times);
    print_metrics("phone_total", phone_times);
    print_metrics("exposed_wait", exposed_wait_times);
    print_metrics("late_cuda", late_times);
    printf("hidden=%zu/%d hidden_fraction=%.6f\n",
            hidden_count, iterations,
            (double) hidden_count / (double) iterations);

    printf("PHONE_BREAKDOWN ");
    print_metrics("prepare", prepare_times);
    print_metrics("usb_out", usb_out_times);
    print_metrics("usb_in_compute", usb_in_times);
    print_metrics("validate", validate_times);
    print_metrics("overlap_tail_wait", cuda_tail_wait_times);
    printf("\n");

    printf("LATE_BREAKDOWN ");
    print_metrics("upload_submit", late_upload_times);
    print_metrics("compute_with_dependency", late_compute_times);
    print_metrics("download", late_download_times);
    printf("\n");

    const double full_median = percentile(full_times, 0.5);
    const double split_median = percentile(total_times, 0.5);
    printf("VERDICT no_wait_median=%s no_wait_all=%s "
           "split_beats_cuda=%s latency_change_percent=%.3f\n",
            percentile(exposed_wait_times, 0.5) == 0.0 ? "PASS" : "FAIL",
            hidden_count == (size_t) iterations ? "PASS" : "FAIL",
            split_median < full_median ? "PASS" : "FAIL",
            100.0 * (split_median / full_median - 1.0));

    spin_client.reset();
    libusb_release_interface(usb, 0);
    libusb_close(usb);
    libusb_exit(usb_context);
    if (dual_buffer != nullptr) {
        ggml_backend_buffer_free(dual_host_buffer);
        ggml_backend_buffer_free(dual_buffer);
        ggml_free(dual_context);
        ggml_backend_free(dual_backend);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    ggml_quantize_free();
    return 0;
}
