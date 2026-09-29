#include "../aoa_async_transport_v1/aoa_async_protocol.h"

#include "../../../../../examples/layersplit/ffn-split-usb-client.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static constexpr uint16_t vendor_id = 0x18d1;
static constexpr uint16_t product_id = 0x2d00;
static constexpr uint8_t out_endpoint = 0x01;
static constexpr uint8_t in_endpoint = 0x81;
static constexpr size_t slot_safety_bytes = 64U * 1024U;
static constexpr const char * transport_generation =
        "functionfs-dmabuf-async-ring-v2";

struct sample {
    uint64_t sequence;
    uint64_t started_ns;
    uint64_t out_completed_ns;
    uint64_t response_completed_ns;
    double out_completion_ms;
    double post_out_tail_ms;
    double response_ready_ms;
};

uint64_t parse_u64(const char * text, const char * name) {
    errno = 0;
    char * end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return value;
}

double elapsed_ms(uint64_t started, uint64_t completed) {
    return static_cast<double>(completed - started) / 1e6;
}

void fill_request(
        const ffn_split::usb_slot_buffers & slot, uint64_t sequence,
        uint32_t request_bytes, uint32_t response_bytes) {
    if (slot.host_to_device_capacity < request_bytes) {
        throw std::runtime_error("USB request slot is too small");
    }
    std::memset(slot.host_to_device, 0xa5, request_bytes);
    const s41_aoa_frame_header header = {
        S41_AOA_REQUEST_MAGIC,
        sequence,
        request_bytes,
        response_bytes,
        s41_aoa_sentinel(sequence),
    };
    std::memcpy(slot.host_to_device, &header, sizeof(header));
    const uint64_t trailing = s41_aoa_sentinel(sequence);
    std::memcpy(
            slot.host_to_device + request_bytes - sizeof(trailing),
            &trailing, sizeof(trailing));
}

void validate_response(
        const ffn_split::usb_slot_buffers & slot, uint64_t sequence,
        uint32_t request_bytes, uint32_t response_bytes) {
    if (slot.device_to_host_capacity < response_bytes) {
        throw std::runtime_error("USB response slot is too small");
    }
    s41_aoa_frame_header header = {};
    uint64_t trailing = 0;
    std::memcpy(&header, slot.device_to_host, sizeof(header));
    std::memcpy(
            &trailing,
            slot.device_to_host + response_bytes - sizeof(trailing),
            sizeof(trailing));
    if (header.magic != S41_AOA_RESPONSE_MAGIC ||
        header.sequence != sequence ||
        header.request_bytes != request_bytes ||
        header.response_bytes != response_bytes ||
        header.sentinel != s41_aoa_sentinel(sequence) ||
        trailing != s41_aoa_sentinel(sequence)) {
        throw std::runtime_error(
                "response validation failed for sequence " +
                std::to_string(sequence));
    }
}

void submit(
        ffn_split::usb_client & client,
        const ffn_split::usb_slot_buffers & slot,
        uint64_t sequence, uint32_t request_bytes,
        uint32_t response_bytes) {
    fill_request(slot, sequence, request_bytes, response_bytes);
    std::string error;
    if (!client.submit_acquired(
                slot, request_bytes, response_bytes,
                {sequence, 0, 0}, error)) {
        throw std::runtime_error(error);
    }
}

std::vector<sample> run(
        ffn_split::usb_client & client,
        uint32_t request_bytes, uint32_t response_bytes,
        uint64_t warmup, uint64_t iterations) {
    const uint64_t total = warmup + iterations;
    const unsigned int active_depth = static_cast<unsigned int>(
            std::min<uint64_t>(total, client.queue_depth()));
    std::vector<ffn_split::usb_slot_buffers> slots(active_depth);
    std::vector<uint64_t> sequences(active_depth, 0);
    std::string error;
    for (unsigned int index = 0; index < active_depth; ++index) {
        if (!client.acquire(slots[index], error)) {
            throw std::runtime_error(error);
        }
        sequences[index] = index + 1;
        submit(
                client, slots[index], sequences[index],
                request_bytes, response_bytes);
    }

    std::vector<sample> samples;
    samples.reserve(static_cast<size_t>(iterations));
    uint64_t next_sequence = active_depth + 1;
    uint64_t completed = 0;
    try {
        while (completed < total) {
            const unsigned int index = static_cast<unsigned int>(
                    completed % active_depth);
            ffn_split::usb_transfer_record record;
            if (!client.wait_acquired(slots[index], record, error)) {
                throw std::runtime_error(error);
            }
            const uint64_t sequence = sequences[index];
            if (record.identity.request_id != sequence) {
                throw std::runtime_error("USB transfer identity mismatch");
            }
            validate_response(
                    slots[index], sequence, request_bytes, response_bytes);
            if (sequence > warmup) {
                samples.push_back({
                    sequence,
                    record.started_ns,
                    record.host_to_device_completed_ns,
                    record.device_to_host_completed_ns,
                    elapsed_ms(
                            record.started_ns,
                            record.host_to_device_completed_ns),
                    record.device_to_host_completed_ns >
                                    record.host_to_device_completed_ns ?
                            elapsed_ms(
                                    record.host_to_device_completed_ns,
                                    record.device_to_host_completed_ns) : 0.0,
                    elapsed_ms(
                            record.started_ns,
                            record.device_to_host_completed_ns),
                });
            }
            ++completed;
            if (next_sequence <= total) {
                sequences[index] = next_sequence++;
                submit(
                        client, slots[index], sequences[index],
                        request_bytes, response_bytes);
            }
        }
    } catch (...) {
        client.close();
        throw;
    }
    for (const auto & slot : slots) {
        if (!client.release(slot, error)) {
            throw std::runtime_error(error);
        }
    }
    return samples;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = std::min(
            values.size() - 1,
            static_cast<size_t>(fraction * (values.size() - 1)));
    return values[index];
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ?
            (values[middle - 1] + values[middle]) / 2.0 : values[middle];
}

void write_array(
        std::ofstream & output, const char * name,
        const std::vector<double> & values, bool comma) {
    output << "  \"" << name << "\": [\n";
    for (size_t index = 0; index < values.size(); ++index) {
        output << "    " << values[index]
               << (index + 1 == values.size() ? "" : ",") << '\n';
    }
    output << "  ]" << (comma ? "," : "") << "\n";
}

void write_result(
        const char * output_path, const std::string & mode,
        const std::string & allocator, const char * device_mode,
        const char * case_name, uint32_t request_bytes,
        uint32_t response_bytes, uint64_t warmup, uint64_t iterations,
        unsigned int configured_depth, unsigned int active_depth,
        const std::vector<sample> & samples) {
    std::vector<double> response_values;
    std::vector<double> out_values;
    std::vector<double> tail_values;
    uint64_t first_started = std::numeric_limits<uint64_t>::max();
    uint64_t last_completed = 0;
    for (const sample & value : samples) {
        response_values.push_back(value.response_ready_ms);
        out_values.push_back(value.out_completion_ms);
        tail_values.push_back(value.post_out_tail_ms);
        first_started = std::min(first_started, value.started_ns);
        last_completed = std::max(
                last_completed, value.response_completed_ns);
    }
    const double seconds = static_cast<double>(
            last_completed - first_started) / 1e9;
    const double h2d_MBps = static_cast<double>(request_bytes) *
            iterations / seconds / 1e6;
    const double d2h_MBps = static_cast<double>(response_bytes) *
            iterations / seconds / 1e6;
    const double aggregate_MBps = h2d_MBps + d2h_MBps;

    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("cannot open output file");
    }
    output << std::setprecision(12);
    output << "{\n";
    output << "  \"schema\": \"s41_ffs_dmabuf_transport_v2\",\n";
    output << "  \"case_name\": \"" << case_name << "\",\n";
    output << "  \"device_mode\": \"" << device_mode << "\",\n";
    output << "  \"host_mode\": \"" << mode << "\",\n";
    output << "  \"host_allocator\": \"" << allocator << "\",\n";
    output << "  \"transport_generation\": \""
           << transport_generation << "\",\n";
    output << "  \"request_bytes\": " << request_bytes << ",\n";
    output << "  \"response_bytes\": " << response_bytes << ",\n";
    output << "  \"warmup\": " << warmup << ",\n";
    output << "  \"iterations\": " << iterations << ",\n";
    output << "  \"configured_queue_depth\": "
           << configured_depth << ",\n";
    output << "  \"queue_depth\": " << active_depth << ",\n";
    output << "  \"usbfs_available_bytes\": "
           << ffn_split::usbfs_memory_bytes() << ",\n";
    output << "  \"slot_safety_bytes\": "
           << slot_safety_bytes << ",\n";
    output << "  \"campaign_start_monotonic_ns\": "
           << first_started << ",\n";
    output << "  \"campaign_end_monotonic_ns\": "
           << last_completed << ",\n";
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
    output << "  \"campaign_seconds\": " << seconds << ",\n";
    output << "  \"requests_per_second\": "
           << iterations / seconds << ",\n";
    output << "  \"h2d_payload_MBps\": " << h2d_MBps << ",\n";
    output << "  \"d2h_payload_MBps\": " << d2h_MBps << ",\n";
    output << "  \"aggregate_payload_MBps\": "
           << aggregate_MBps << ",\n";
    output << "  \"reset_recoveries\": 0,\n";
    write_array(output, "response_ready_samples_ms", response_values, true);
    write_array(output, "out_completion_samples_ms", out_values, true);
    write_array(output, "post_out_tail_samples_ms", tail_values, false);
    output << "}\n";

    std::printf(
            "device=%s host=%s allocator=%s depth=%u request=%u "
            "response=%u median=%.6f p90=%.6f h2d_MBps=%.3f "
            "d2h_MBps=%.3f aggregate_MBps=%.3f\n",
            device_mode, mode.c_str(), allocator.c_str(), active_depth,
            request_bytes, response_bytes, median(response_values),
            percentile(response_values, 0.90), h2d_MBps, d2h_MBps,
            aggregate_MBps);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 9) {
        std::fprintf(
                stderr,
                "usage: %s <sync|async> <malloc|devmem> <request-bytes> "
                "<response-bytes> <warmup> <iterations> <depth> "
                "<output.json>\n",
                argv[0]);
        return 2;
    }
    try {
        const std::string mode = argv[1];
        const std::string allocator_name = argv[2];
        const uint64_t request_value = parse_u64(argv[3], "request_bytes");
        const uint64_t response_value = parse_u64(argv[4], "response_bytes");
        const uint64_t warmup = parse_u64(argv[5], "warmup");
        const uint64_t iterations = parse_u64(argv[6], "iterations");
        const uint64_t depth_value = parse_u64(argv[7], "depth");
        if ((mode != "sync" && mode != "async") ||
            (allocator_name != "malloc" && allocator_name != "devmem") ||
            request_value < sizeof(s41_aoa_frame_header) + sizeof(uint64_t) ||
            response_value < sizeof(s41_aoa_frame_header) + sizeof(uint64_t) ||
            request_value > 16U * 1024U * 1024U ||
            response_value > 16U * 1024U * 1024U || iterations == 0 ||
            depth_value == 0 || depth_value > 16 ||
            (mode == "sync" && depth_value != 1)) {
            throw std::runtime_error("invalid arguments");
        }
        ffn_split::usb_host_allocator allocator;
        if (!ffn_split::parse_usb_host_allocator(
                    allocator_name, allocator)) {
            throw std::runtime_error("invalid host allocator");
        }
        ffn_split::usb_client_config config;
        config.vendor_id = vendor_id;
        config.product_id = product_id;
        config.host_to_device_endpoint = out_endpoint;
        config.device_to_host_endpoint = in_endpoint;
        config.host_to_device_slot_bytes = request_value;
        config.device_to_host_slot_bytes = response_value;
        config.slot_safety_bytes = slot_safety_bytes;
        config.configured_max_queue_depth =
                static_cast<unsigned int>(depth_value);
        config.timeout_ms = 10000;
        config.allocator = allocator;
        config.transport_generation = transport_generation;
        ffn_split::usb_client client(std::move(config));
        std::string error;
        if (!client.connect(error)) {
            throw std::runtime_error(error);
        }
        const unsigned int active_depth = client.queue_depth();
        const auto samples = run(
                client, static_cast<uint32_t>(request_value),
                static_cast<uint32_t>(response_value), warmup, iterations);
        const char * device_mode = std::getenv("S41_DEVICE_MODE");
        const char * case_name = std::getenv("S41_CASE_NAME");
        write_result(
                argv[8], mode, allocator_name,
                device_mode == nullptr ? "unknown" : device_mode,
                case_name == nullptr ? "unnamed" : case_name,
                static_cast<uint32_t>(request_value),
                static_cast<uint32_t>(response_value), warmup, iterations,
                static_cast<unsigned int>(depth_value), active_depth,
                samples);
        client.close();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
