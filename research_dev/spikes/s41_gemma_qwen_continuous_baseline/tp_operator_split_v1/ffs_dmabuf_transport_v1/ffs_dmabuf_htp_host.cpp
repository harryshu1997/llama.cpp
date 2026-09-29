#include "../aoa_async_transport_v1/libusb_abi.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" unsigned char * libusb_dev_mem_alloc(
        libusb_device_handle * handle, size_t length);
extern "C" int libusb_dev_mem_free(libusb_device_handle * handle,
        unsigned char * buffer, size_t length);

static constexpr uint16_t VENDOR_ID = 0x18d1;
static constexpr uint16_t PRODUCT_ID = 0x2d00;
static constexpr unsigned char OUT_ENDPOINT = 0x01;
static constexpr unsigned char IN_ENDPOINT = 0x81;
static constexpr unsigned int TRANSFER_TIMEOUT_MS = 10000;

class transfer_buffer {
public:
    transfer_buffer(libusb_device_handle * handle, size_t size,
            bool persistent) : handle_(handle), size_(size),
            persistent_(persistent) {
        if (persistent_) {
            data_ = libusb_dev_mem_alloc(handle_, size_);
            if (data_ == nullptr) {
                throw std::runtime_error(
                        "libusb persistent DMA allocation failed");
            }
        } else {
            owned_.resize(size_);
            data_ = owned_.data();
        }
    }

    ~transfer_buffer() {
        if (persistent_ && data_ != nullptr) {
            libusb_dev_mem_free(handle_, data_, size_);
        }
    }

    transfer_buffer(const transfer_buffer &) = delete;
    transfer_buffer & operator=(const transfer_buffer &) = delete;

    unsigned char * data() { return data_; }
    const unsigned char * data() const { return data_; }
    size_t size() const { return size_; }

private:
    libusb_device_handle * handle_;
    size_t size_;
    bool persistent_;
    unsigned char * data_ = nullptr;
    std::vector<unsigned char> owned_;
};

struct sample {
    double response_ms;
    double out_ms;
    double tail_ms;
};

static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double elapsed_ms(uint64_t started, uint64_t completed) {
    return (completed - started) / 1e6;
}

static uint64_t parse_u64(const char * text, const char * name) {
    errno = 0;
    char * end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return value;
}

static float input_value(uint64_t sequence, size_t index) {
    const int32_t encoded = static_cast<int32_t>(
            (sequence * 17 + index * 13) % 257) - 128;
    return encoded / 32.0f;
}

static void fill_request(transfer_buffer & buffer, uint64_t sequence,
        size_t elements) {
    for (size_t index = 0; index < elements; ++index) {
        const float value = input_value(sequence, index);
        std::memcpy(buffer.data() + index * sizeof(value),
                &value, sizeof(value));
    }
}

static double validate_response(const transfer_buffer & buffer,
        uint64_t sequence, size_t elements) {
    double maximum_error = 0.0;
    for (size_t index = 0; index < elements; ++index) {
        float output = 0.0f;
        std::memcpy(&output, buffer.data() + index * sizeof(output),
                sizeof(output));
        const float input = input_value(sequence, index);
        const float expected = input * input;
        const double error = std::abs(
                static_cast<double>(output) - expected);
        maximum_error = std::max(maximum_error, error);
        const double tolerance = 1e-5 +
                std::abs(static_cast<double>(expected)) * 1e-5;
        if (!std::isfinite(output) || error > tolerance) {
            throw std::runtime_error(
                    "HTP response mismatch at sequence " +
                    std::to_string(sequence) + " index " +
                    std::to_string(index));
        }
    }
    return maximum_error;
}

static void transfer_exact(libusb_device_handle * handle,
        unsigned char endpoint, unsigned char * data, size_t size) {
    size_t completed = 0;
    while (completed < size) {
        int transferred = 0;
        const int status = libusb_bulk_transfer(handle, endpoint,
                data + completed, static_cast<int>(size - completed),
                &transferred, TRANSFER_TIMEOUT_MS);
        if (status != 0) {
            throw std::runtime_error(std::string("bulk transfer failed: ") +
                    libusb_error_name(status));
        }
        if (transferred <= 0) {
            throw std::runtime_error("bulk transfer made no progress");
        }
        completed += static_cast<size_t>(transferred);
    }
}

static double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = std::min(values.size() - 1,
            static_cast<size_t>(fraction * (values.size() - 1)));
    return values[index];
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0
            ? (values[middle - 1] + values[middle]) / 2.0
            : values[middle];
}

static void write_array(std::ofstream & output, const char * name,
        const std::vector<double> & values, bool trailing_comma) {
    output << "  \"" << name << "\": [\n";
    for (size_t index = 0; index < values.size(); ++index) {
        output << "    " << values[index];
        if (index + 1 != values.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "  ]" << (trailing_comma ? "," : "") << "\n";
}

static void write_result(const char * path, const std::string & allocator,
        size_t elements, uint64_t warmup, uint64_t iterations,
        const std::vector<sample> & samples, double maximum_error,
        uint64_t first_started, uint64_t last_completed) {
    std::vector<double> response_values;
    std::vector<double> out_values;
    std::vector<double> tail_values;
    for (const sample & value : samples) {
        response_values.push_back(value.response_ms);
        out_values.push_back(value.out_ms);
        tail_values.push_back(value.tail_ms);
    }
    const size_t bytes = elements * sizeof(float);
    const double seconds = (last_completed - first_started) / 1e9;
    const double requests_per_second = iterations / seconds;
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot open output file");
    }
    output << std::setprecision(12);
    output << "{\n";
    output << "  \"schema\": \"s41_ffs_dmabuf_htp_v1\",\n";
    output << "  \"operator\": \"sqr_f32\",\n";
    output << "  \"host_allocator\": \"" << allocator << "\",\n";
    output << "  \"elements\": " << elements << ",\n";
    output << "  \"request_bytes\": " << bytes << ",\n";
    output << "  \"response_bytes\": " << bytes << ",\n";
    output << "  \"warmup\": " << warmup << ",\n";
    output << "  \"iterations\": " << iterations << ",\n";
    output << "  \"response_median_ms\": "
           << median(response_values) << ",\n";
    output << "  \"response_p90_ms\": "
           << percentile(response_values, 0.90) << ",\n";
    output << "  \"response_p99_ms\": "
           << percentile(response_values, 0.99) << ",\n";
    output << "  \"out_median_ms\": " << median(out_values) << ",\n";
    output << "  \"post_out_tail_median_ms\": "
           << median(tail_values) << ",\n";
    output << "  \"maximum_absolute_error\": " << maximum_error << ",\n";
    output << "  \"requests_per_second\": " << requests_per_second << ",\n";
    output << "  \"aggregate_payload_MBps\": "
           << requests_per_second * 2.0 * bytes / 1e6 << ",\n";
    write_array(output, "response_samples_ms", response_values, true);
    write_array(output, "out_samples_ms", out_values, true);
    write_array(output, "post_out_tail_samples_ms", tail_values, false);
    output << "}\n";
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
                "usage: %s <malloc|devmem> <elements> <warmup> "
                "<iterations> <output.json>\n",
                argv[0]);
        return 2;
    }
    try {
        const std::string allocator = argv[1];
        const uint64_t elements_value = parse_u64(argv[2], "elements");
        const uint64_t warmup = parse_u64(argv[3], "warmup");
        const uint64_t iterations = parse_u64(argv[4], "iterations");
        if ((allocator != "malloc" && allocator != "devmem") ||
                elements_value == 0 || elements_value > 4U * 1024U * 1024U ||
                iterations == 0) {
            throw std::runtime_error("invalid arguments");
        }
        const size_t elements = static_cast<size_t>(elements_value);
        const size_t bytes = elements * sizeof(float);

        libusb_context * context = nullptr;
        int status = libusb_init(&context);
        if (status != 0) {
            throw std::runtime_error(std::string("libusb_init failed: ") +
                    libusb_error_name(status));
        }
        libusb_device_handle * handle = libusb_open_device_with_vid_pid(
                context, VENDOR_ID, PRODUCT_ID);
        if (handle == nullptr) {
            libusb_exit(context);
            throw std::runtime_error("S41 FunctionFS device not found");
        }
        libusb_detach_kernel_driver(handle, 0);
        status = libusb_claim_interface(handle, 0);
        if (status != 0) {
            libusb_close(handle);
            libusb_exit(context);
            throw std::runtime_error(std::string("claim failed: ") +
                    libusb_error_name(status));
        }

        const bool persistent = allocator == "devmem";
        transfer_buffer request(handle, bytes, persistent);
        transfer_buffer response(handle, bytes, persistent);
        std::vector<sample> samples;
        samples.reserve(static_cast<size_t>(iterations));
        double maximum_error = 0.0;
        uint64_t first_started = 0;
        uint64_t last_completed = 0;
        const uint64_t total = warmup + iterations;
        for (uint64_t sequence = 1; sequence <= total; ++sequence) {
            fill_request(request, sequence, elements);
            std::memset(response.data(), 0xcd, response.size());
            const uint64_t started = now_ns();
            transfer_exact(handle, OUT_ENDPOINT,
                    request.data(), request.size());
            const uint64_t out_completed = now_ns();
            transfer_exact(handle, IN_ENDPOINT,
                    response.data(), response.size());
            const uint64_t completed = now_ns();
            maximum_error = std::max(maximum_error,
                    validate_response(response, sequence, elements));
            if (sequence > warmup) {
                if (samples.empty()) {
                    first_started = started;
                }
                last_completed = completed;
                samples.push_back({
                    elapsed_ms(started, completed),
                    elapsed_ms(started, out_completed),
                    elapsed_ms(out_completed, completed),
                });
            }
        }
        write_result(argv[5], allocator, elements, warmup, iterations,
                samples, maximum_error, first_started, last_completed);
        std::printf(
                "operator=sqr_f32 allocator=%s elements=%zu median=%.6f "
                "p90=%.6f max_error=%.9g\n",
                allocator.c_str(), elements,
                median([&samples]() {
                    std::vector<double> values;
                    for (const sample & value : samples) {
                        values.push_back(value.response_ms);
                    }
                    return values;
                }()),
                percentile([&samples]() {
                    std::vector<double> values;
                    for (const sample & value : samples) {
                        values.push_back(value.response_ms);
                    }
                    return values;
                }(), 0.90),
                maximum_error);
        libusb_release_interface(handle, 0);
        libusb_close(handle);
        libusb_exit(context);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
