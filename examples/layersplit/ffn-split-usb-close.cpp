#include "ffn-split-dmabuf.h"
#include "ffn-split-protocol.h"
#include "ffn-split-usb-client.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

uint64_t parse_unsigned(const char * name, const char * text, uint64_t maximum) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || value > maximum) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<uint64_t>(value);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 11) {
        fprintf(stderr,
                "usage: %s <vendor-id> <product-id> <layer-mask> <n-embd> "
                "<columns> <max-tokens> <flags> <usbfs-bytes> "
                "<transport-generation> <artifact-sha256>\n",
                argv[0]);
        return 2;
    }

    try {
        const uint16_t vendor_id = static_cast<uint16_t>(parse_unsigned(
                "vendor id", argv[1], UINT16_MAX));
        const uint16_t product_id = static_cast<uint16_t>(parse_unsigned(
                "product id", argv[2], UINT16_MAX));
        const uint64_t layer_mask = parse_unsigned(
                "layer mask", argv[3], UINT64_MAX);
        const uint32_t n_embd = static_cast<uint32_t>(parse_unsigned(
                "embedding width", argv[4], UINT32_MAX));
        const uint32_t columns = static_cast<uint32_t>(parse_unsigned(
                "columns", argv[5], UINT32_MAX));
        const uint16_t max_tokens = static_cast<uint16_t>(parse_unsigned(
                "maximum tokens", argv[6], UINT16_MAX));
        const uint16_t flags = static_cast<uint16_t>(parse_unsigned(
                "flags", argv[7], UINT16_MAX));
        const size_t usbfs_bytes = static_cast<size_t>(parse_unsigned(
                "usbfs bytes", argv[8], std::numeric_limits<size_t>::max()));
        const std::string generation = argv[9];
        const std::string artifact = argv[10];
        uint8_t artifact_sha256[32] = {};
        const size_t element_bytes = flags & ffn_split::flag_f16_io ? 2 : 4;
        if (vendor_id == 0 || product_id == 0 || layer_mask == 0 ||
            n_embd == 0 || columns == 0 || max_tokens == 0 ||
            generation.empty() ||
            !ffn_split::parse_artifact_sha256(
                    artifact, artifact_sha256) ||
            n_embd > (SIZE_MAX - ffn_split::dmabuf_payload_offset) /
                    max_tokens / element_bytes) {
            throw std::runtime_error("invalid direct close contract");
        }

        const size_t slot_bytes = ffn_split::dmabuf_payload_offset +
                static_cast<size_t>(n_embd) * max_tokens * element_bytes;
        ffn_split::usb_client_config config;
        config.vendor_id = vendor_id;
        config.product_id = product_id;
        config.host_to_device_endpoint = ffn_split::dmabuf_out_endpoint;
        config.device_to_host_endpoint = ffn_split::dmabuf_in_endpoint;
        config.host_to_device_slot_bytes = slot_bytes;
        config.device_to_host_slot_bytes = slot_bytes;
        config.usbfs_available_bytes = usbfs_bytes;
        config.slot_safety_bytes = 64U * 1024U;
        config.configured_max_queue_depth = 1;
        config.timeout_ms = 10000;
        config.allocator = ffn_split::usb_host_allocator::malloc_buffer;
        config.transport_generation = generation;

        ffn_split::usb_client client(std::move(config));
        std::string error;
        if (!client.connect(error)) {
            throw std::runtime_error(error);
        }

        ffn_split::hello_request hello = {};
        hello.magic = ffn_split::protocol_magic;
        hello.version = ffn_split::protocol_version;
        hello.message = static_cast<uint16_t>(
                ffn_split::message_type::hello_request);
        hello.layer_mask = layer_mask;
        hello.n_embd = n_embd;
        hello.max_columns = columns;
        hello.flags = flags;
        hello.max_tokens = max_tokens;
        memcpy(
                hello.artifact_sha256, artifact_sha256,
                sizeof(artifact_sha256));
        ffn_split::hello_response response = {};
        ffn_split::usb_transfer_record record;
        if (!client.exchange(
                    &hello, sizeof(hello), &response, sizeof(response), {},
                    record, error)) {
            throw std::runtime_error("HELLO failed: " + error);
        }
        if (response.magic != ffn_split::protocol_magic ||
            response.version != ffn_split::protocol_version ||
            response.message != static_cast<uint16_t>(
                    ffn_split::message_type::hello_response) ||
            response.status != 0 || response.layer_mask != layer_mask ||
            response.n_embd != n_embd || response.max_columns != columns ||
            response.max_tokens != max_tokens || response.flags != flags ||
            !ffn_split::same_artifact_sha256(
                    response.artifact_sha256, hello.artifact_sha256)) {
            throw std::runtime_error("HELLO identity mismatch");
        }

        ffn_split::execute_request shutdown = {};
        shutdown.magic = ffn_split::protocol_magic;
        shutdown.version = ffn_split::protocol_version;
        shutdown.message = static_cast<uint16_t>(
                ffn_split::message_type::execute_request);
        shutdown.layer = -1;
        if (!client.exchange(
                    &shutdown, sizeof(shutdown), nullptr, 0, {}, record,
                    error)) {
            throw std::runtime_error("shutdown failed: " + error);
        }
        fprintf(stdout,
                "FFNUSB_CLOSE status=ok queue_depth=%u generation=%s\n",
                client.queue_depth(), generation.c_str());
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "FFNUSB_CLOSE status=error detail=%s\n", error.what());
        return 1;
    }
}
