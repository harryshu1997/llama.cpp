#include "aoa_async_protocol.h"
#include "libusb_abi.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint16_t AOA_VENDOR_ID = 0x18d1;
static constexpr uint16_t AOA_PRODUCT_IDS[] = {
    0x2d00, 0x2d01, 0x2d04, 0x2d05,
};
static constexpr unsigned char OUT_ENDPOINT = 0x01;
static constexpr unsigned char IN_ENDPOINT = 0x81;
static constexpr unsigned int TRANSFER_TIMEOUT_MS = 10000;

struct sample {
    uint64_t sequence;
    uint64_t started_ns;
    uint64_t out_completed_ns;
    uint64_t response_completed_ns;
    double out_completion_ms;
    double post_out_tail_ms;
    double response_ready_ms;
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

static void fill_request(std::vector<unsigned char> & data,
        uint64_t sequence, uint32_t request_bytes, uint32_t response_bytes) {
    std::fill(data.begin(), data.end(), 0xa5);
    const s41_aoa_frame_header header = {
        S41_AOA_REQUEST_MAGIC,
        sequence,
        request_bytes,
        response_bytes,
        s41_aoa_sentinel(sequence),
    };
    std::memcpy(data.data(), &header, sizeof(header));
    const uint64_t trailing = s41_aoa_sentinel(sequence);
    std::memcpy(data.data() + data.size() - sizeof(trailing),
            &trailing, sizeof(trailing));
}

static void validate_response(const std::vector<unsigned char> & data,
        uint64_t sequence, uint32_t request_bytes, uint32_t response_bytes) {
    s41_aoa_frame_header header;
    std::memcpy(&header, data.data(), sizeof(header));
    uint64_t trailing;
    std::memcpy(&trailing, data.data() + data.size() - sizeof(trailing),
            sizeof(trailing));
    if (header.magic != S41_AOA_RESPONSE_MAGIC ||
            header.sequence != sequence ||
            header.request_bytes != request_bytes ||
            header.response_bytes != response_bytes ||
            header.sentinel != s41_aoa_sentinel(sequence) ||
            trailing != s41_aoa_sentinel(sequence)) {
        throw std::runtime_error("response validation failed for sequence " +
                std::to_string(sequence));
    }
}

static libusb_device_handle * open_accessory(libusb_context * context,
        uint16_t * selected_product) {
    for (const uint16_t product : AOA_PRODUCT_IDS) {
        libusb_device_handle * handle = libusb_open_device_with_vid_pid(
                context, AOA_VENDOR_ID, product);
        if (handle != nullptr) {
            *selected_product = product;
            return handle;
        }
    }
    return nullptr;
}

static void transfer_exact(libusb_device_handle * handle,
        unsigned char endpoint, unsigned char * data, size_t size) {
    size_t completed = 0;
    while (completed < size) {
        const size_t remaining = size - completed;
        const int chunk = remaining > (size_t) std::numeric_limits<int>::max()
            ? std::numeric_limits<int>::max() : (int) remaining;
        int transferred = 0;
        const int status = libusb_bulk_transfer(handle, endpoint,
                data + completed, chunk, &transferred, TRANSFER_TIMEOUT_MS);
        if (status != 0) {
            throw std::runtime_error(std::string("bulk transfer failed: ") +
                    libusb_error_name(status));
        }
        if (transferred <= 0) {
            throw std::runtime_error("bulk transfer made no progress");
        }
        completed += (size_t) transferred;
    }
}

static std::vector<sample> run_sync(libusb_device_handle * handle,
        uint32_t request_bytes, uint32_t response_bytes, uint64_t warmup,
        uint64_t iterations) {
    std::vector<unsigned char> request(request_bytes);
    std::vector<unsigned char> response(response_bytes);
    std::vector<sample> samples;
    samples.reserve((size_t) iterations);
    const uint64_t total = warmup + iterations;
    for (uint64_t sequence = 1; sequence <= total; ++sequence) {
        fill_request(request, sequence, request_bytes, response_bytes);
        const uint64_t started = now_ns();
        transfer_exact(handle, OUT_ENDPOINT, request.data(), request.size());
        const uint64_t out_completed = now_ns();
        transfer_exact(handle, IN_ENDPOINT, response.data(), response.size());
        const uint64_t response_completed = now_ns();
        validate_response(response, sequence, request_bytes, response_bytes);
        if (sequence > warmup) {
            samples.push_back({
                sequence,
                started,
                out_completed,
                response_completed,
                elapsed_ms(started, out_completed),
                elapsed_ms(out_completed, response_completed),
                elapsed_ms(started, response_completed),
            });
        }
    }
    return samples;
}

struct async_slot;

struct transfer_tag {
    async_slot * slot;
    bool input;
};

struct async_slot {
    std::vector<unsigned char> request;
    std::vector<unsigned char> response;
    libusb_transfer * input_transfer = nullptr;
    libusb_transfer * output_transfer = nullptr;
    transfer_tag input_tag = {this, true};
    transfer_tag output_tag = {this, false};
    uint64_t sequence = 0;
    uint64_t started_ns = 0;
    uint64_t out_completed_ns = 0;
    uint64_t response_completed_ns = 0;
    bool active = false;
    bool input_done = false;
    bool output_done = false;
    bool failed = false;
    int failure_status = 0;

    async_slot(uint32_t request_bytes, uint32_t response_bytes) :
        request(request_bytes), response(response_bytes) {
    }
};

static void transfer_completed(libusb_transfer * transfer) {
    transfer_tag * tag = static_cast<transfer_tag *>(transfer->user_data);
    async_slot * slot = tag->slot;
    const uint64_t completed = now_ns();
    if (transfer->status != LIBUSB_TRANSFER_COMPLETED ||
            transfer->actual_length != transfer->length) {
        slot->failed = true;
        slot->failure_status = transfer->status;
    }
    if (tag->input) {
        slot->input_done = true;
        slot->response_completed_ns = completed;
    } else {
        slot->output_done = true;
        slot->out_completed_ns = completed;
    }
}

static void configure_transfer(libusb_transfer * transfer,
        libusb_device_handle * handle, unsigned char endpoint,
        std::vector<unsigned char> & buffer, transfer_tag * tag) {
    transfer->dev_handle = handle;
    transfer->flags = 0;
    transfer->endpoint = endpoint;
    transfer->type = LIBUSB_TRANSFER_TYPE_BULK;
    transfer->timeout = TRANSFER_TIMEOUT_MS;
    transfer->length = (int) buffer.size();
    transfer->callback = transfer_completed;
    transfer->user_data = tag;
    transfer->buffer = buffer.data();
    transfer->num_iso_packets = 0;
}

static void submit_slot(async_slot & slot, uint64_t sequence,
        uint32_t request_bytes, uint32_t response_bytes) {
    fill_request(slot.request, sequence, request_bytes, response_bytes);
    slot.sequence = sequence;
    slot.out_completed_ns = 0;
    slot.response_completed_ns = 0;
    slot.input_done = false;
    slot.output_done = false;
    slot.failed = false;
    slot.failure_status = 0;
    slot.active = true;
    slot.started_ns = now_ns();

    int status = libusb_submit_transfer(slot.input_transfer);
    if (status != 0) {
        slot.active = false;
        throw std::runtime_error(std::string("submit IN failed: ") +
                libusb_error_name(status));
    }
    status = libusb_submit_transfer(slot.output_transfer);
    if (status != 0) {
        libusb_cancel_transfer(slot.input_transfer);
        throw std::runtime_error(std::string("submit OUT failed: ") +
                libusb_error_name(status));
    }
}

static void cancel_active(libusb_context * context,
        std::vector<std::unique_ptr<async_slot>> & slots) {
    bool pending = false;
    for (const auto & item : slots) {
        if (!item->active) {
            continue;
        }
        if (!item->input_done) {
            libusb_cancel_transfer(item->input_transfer);
            pending = true;
        }
        if (!item->output_done) {
            libusb_cancel_transfer(item->output_transfer);
            pending = true;
        }
    }
    while (pending) {
        pending = false;
        libusb_handle_events(context);
        for (const auto & item : slots) {
            if (item->active && (!item->input_done || !item->output_done)) {
                pending = true;
            }
        }
    }
}

static std::vector<sample> run_async(libusb_context * context,
        libusb_device_handle * handle, uint32_t request_bytes,
        uint32_t response_bytes, uint64_t warmup, uint64_t iterations,
        unsigned int queue_depth) {
    const uint64_t total = warmup + iterations;
    const unsigned int active_depth = (unsigned int) std::min<uint64_t>(
            total, queue_depth);
    std::vector<std::unique_ptr<async_slot>> slots;
    slots.reserve(active_depth);
    for (unsigned int index = 0; index < active_depth; ++index) {
        auto slot = std::make_unique<async_slot>(request_bytes, response_bytes);
        slot->input_transfer = libusb_alloc_transfer(0);
        slot->output_transfer = libusb_alloc_transfer(0);
        if (slot->input_transfer == nullptr || slot->output_transfer == nullptr) {
            throw std::runtime_error("libusb transfer allocation failed");
        }
        configure_transfer(slot->input_transfer, handle, IN_ENDPOINT,
                slot->response, &slot->input_tag);
        configure_transfer(slot->output_transfer, handle, OUT_ENDPOINT,
                slot->request, &slot->output_tag);
        slots.push_back(std::move(slot));
    }

    std::vector<sample> samples;
    samples.reserve((size_t) iterations);
    uint64_t next_sequence = 1;
    uint64_t completed_count = 0;
    try {
        for (auto & slot : slots) {
            submit_slot(*slot, next_sequence++, request_bytes, response_bytes);
        }
        while (completed_count < total) {
            const int event_status = libusb_handle_events(context);
            if (event_status != 0) {
                throw std::runtime_error(std::string("event handling failed: ") +
                        libusb_error_name(event_status));
            }
            for (auto & slot : slots) {
                if (!slot->active || !slot->input_done || !slot->output_done) {
                    continue;
                }
                if (slot->failed) {
                    throw std::runtime_error("asynchronous transfer failed "
                            "with status " +
                            std::to_string(slot->failure_status));
                }
                validate_response(slot->response, slot->sequence,
                        request_bytes, response_bytes);
                if (slot->sequence > warmup) {
                    const uint64_t tail_start = std::min(
                            slot->out_completed_ns,
                            slot->response_completed_ns);
                    samples.push_back({
                        slot->sequence,
                        slot->started_ns,
                        slot->out_completed_ns,
                        slot->response_completed_ns,
                        elapsed_ms(slot->started_ns,
                                slot->out_completed_ns),
                        elapsed_ms(tail_start,
                                slot->response_completed_ns),
                        elapsed_ms(slot->started_ns,
                                slot->response_completed_ns),
                    });
                }
                slot->active = false;
                ++completed_count;
                if (next_sequence <= total) {
                    submit_slot(*slot, next_sequence++, request_bytes,
                            response_bytes);
                }
            }
        }
    } catch (...) {
        cancel_active(context, slots);
        for (auto & slot : slots) {
            libusb_free_transfer(slot->output_transfer);
            libusb_free_transfer(slot->input_transfer);
        }
        throw;
    }
    for (auto & slot : slots) {
        libusb_free_transfer(slot->output_transfer);
        libusb_free_transfer(slot->input_transfer);
    }
    return samples;
}

static double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = std::min(values.size() - 1,
            (size_t) (fraction * (values.size() - 1)));
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

static void write_result(const char * output_path, const std::string & mode,
        const char * phone_mode, const char * case_name,
        const char * workload, uint64_t repetition, uint16_t product_id,
        uint32_t request_bytes, uint32_t response_bytes, uint64_t warmup,
        uint64_t iterations, unsigned int queue_depth,
        const std::vector<sample> & samples) {
    std::vector<double> response_values;
    std::vector<double> out_values;
    std::vector<double> tail_values;
    response_values.reserve(samples.size());
    out_values.reserve(samples.size());
    tail_values.reserve(samples.size());
    uint64_t first_started = UINT64_MAX;
    uint64_t last_completed = 0;
    for (const sample & value : samples) {
        response_values.push_back(value.response_ready_ms);
        out_values.push_back(value.out_completion_ms);
        tail_values.push_back(value.post_out_tail_ms);
        first_started = std::min(first_started, value.started_ns);
        last_completed = std::max(last_completed, value.response_completed_ns);
    }
    const double campaign_seconds =
        (last_completed - first_started) / 1e9;
    const double requests_per_second = iterations / campaign_seconds;
    const double aggregate_MBps =
        ((double) request_bytes + response_bytes) * iterations /
        campaign_seconds / 1e6;

    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("cannot open output file");
    }
    output << std::setprecision(12);
    output << "{\n";
    output << "  \"schema\": \"s41_aoa_async_transport_v1\",\n";
    output << "  \"case_name\": \"" << case_name << "\",\n";
    output << "  \"workload\": \"" << workload << "\",\n";
    output << "  \"repetition\": " << repetition << ",\n";
    output << "  \"phone_mode\": \"" << phone_mode << "\",\n";
    output << "  \"mode\": \"" << mode << "\",\n";
    output << "  \"preposted_in\": "
           << (mode == "async" ? "true" : "false") << ",\n";
    output << "  \"aoa_product_id\": \"0x" << std::hex
           << std::setw(4) << std::setfill('0') << product_id << std::dec
           << std::setfill(' ') << "\",\n";
    output << "  \"request_bytes\": " << request_bytes << ",\n";
    output << "  \"response_bytes\": " << response_bytes << ",\n";
    output << "  \"warmup\": " << warmup << ",\n";
    output << "  \"iterations\": " << iterations << ",\n";
    output << "  \"queue_depth\": " << queue_depth << ",\n";
    output << "  \"response_ready_median_ms\": "
           << median(response_values) << ",\n";
    output << "  \"response_ready_p90_ms\": "
           << percentile(response_values, 0.90) << ",\n";
    output << "  \"response_ready_p99_ms\": "
           << percentile(response_values, 0.99) << ",\n";
    output << "  \"out_completion_median_ms\": "
           << median(out_values) << ",\n";
    output << "  \"post_out_tail_median_ms\": "
           << median(tail_values) << ",\n";
    output << "  \"campaign_seconds\": " << campaign_seconds << ",\n";
    output << "  \"requests_per_second\": "
           << requests_per_second << ",\n";
    output << "  \"aggregate_payload_MBps\": "
           << aggregate_MBps << ",\n";
    write_array(output, "response_ready_samples_ms", response_values, true);
    write_array(output, "out_completion_samples_ms", out_values, true);
    write_array(output, "post_out_tail_samples_ms", tail_values, false);
    output << "}\n";

    std::printf("mode=%s depth=%u request=%u response=%u median=%.6f "
                "p90=%.6f p99=%.6f requests_per_second=%.3f "
                "aggregate_MBps=%.3f\n",
            mode.c_str(), queue_depth, request_bytes, response_bytes,
            median(response_values), percentile(response_values, 0.90),
            percentile(response_values, 0.99), requests_per_second,
            aggregate_MBps);
}

int main(int argc, char ** argv) {
    if (argc != 8) {
        std::fprintf(stderr,
                "usage: %s <sync|async> <request_bytes> <response_bytes> "
                "<warmup> <iterations> <queue_depth> <output.json>\n",
                argv[0]);
        return 2;
    }
    try {
        const std::string mode = argv[1];
        const uint64_t request_value = parse_u64(argv[2], "request_bytes");
        const uint64_t response_value = parse_u64(argv[3], "response_bytes");
        const uint64_t warmup = parse_u64(argv[4], "warmup");
        const uint64_t iterations = parse_u64(argv[5], "iterations");
        const uint64_t depth_value = parse_u64(argv[6], "queue_depth");
        if ((mode != "sync" && mode != "async") ||
                request_value < sizeof(s41_aoa_frame_header) ||
                response_value < sizeof(s41_aoa_frame_header) ||
                request_value > 16U * 1024U * 1024U ||
                response_value > 16U * 1024U * 1024U || iterations == 0 ||
                depth_value == 0 || depth_value > 32 ||
                (mode == "sync" && depth_value != 1)) {
            throw std::runtime_error("invalid arguments");
        }
        const uint32_t request_bytes = (uint32_t) request_value;
        const uint32_t response_bytes = (uint32_t) response_value;
        const unsigned int queue_depth = (unsigned int) depth_value;

        libusb_context * context = nullptr;
        int status = libusb_init(&context);
        if (status != 0) {
            throw std::runtime_error(std::string("libusb_init failed: ") +
                    libusb_error_name(status));
        }
        uint16_t product_id = 0;
        libusb_device_handle * handle = open_accessory(context, &product_id);
        if (handle == nullptr) {
            libusb_exit(context);
            throw std::runtime_error("no AOA device found");
        }
        libusb_detach_kernel_driver(handle, 0);
        status = libusb_claim_interface(handle, 0);
        if (status != 0) {
            libusb_close(handle);
            libusb_exit(context);
            throw std::runtime_error(std::string("claim failed: ") +
                    libusb_error_name(status));
        }

        std::vector<sample> samples;
        try {
            samples = mode == "sync"
                ? run_sync(handle, request_bytes, response_bytes,
                        warmup, iterations)
                : run_async(context, handle, request_bytes, response_bytes,
                        warmup, iterations, queue_depth);
            const char * phone_mode = std::getenv("S41_PHONE_MODE");
            const char * case_name = std::getenv("S41_CASE_NAME");
            const char * workload = std::getenv("S41_WORKLOAD");
            const char * repetition_text = std::getenv("S41_REPETITION");
            const uint64_t repetition = repetition_text == nullptr
                ? 0 : parse_u64(repetition_text, "S41_REPETITION");
            write_result(argv[7], mode,
                    phone_mode == nullptr ? "unknown" : phone_mode,
                    case_name == nullptr ? "unnamed" : case_name,
                    workload == nullptr ? "unknown" : workload,
                    repetition,
                    product_id, request_bytes, response_bytes, warmup,
                    iterations, queue_depth, samples);
        } catch (...) {
            libusb_release_interface(handle, 0);
            libusb_close(handle);
            libusb_exit(context);
            throw;
        }
        libusb_release_interface(handle, 0);
        libusb_close(handle);
        libusb_exit(context);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
