#pragma once

#include "ffn-split-protocol.h"

#include <cstddef>
#include <cstdint>

namespace ffn_split {

static constexpr uint16_t dmabuf_vendor_id = 0x18d1;
static constexpr uint16_t dmabuf_product_id = 0x2d00;
static constexpr uint8_t dmabuf_out_endpoint = 0x01;
static constexpr uint8_t dmabuf_in_endpoint = 0x81;
static constexpr size_t dmabuf_payload_offset = 128;
static constexpr uint32_t dmabuf_payload_ready_magic = 0x44524d41U;

static_assert(sizeof(execute_request) <= dmabuf_payload_offset,
        "FFN request header exceeds DMA-BUF prefix");
static_assert(sizeof(execute_response) <= dmabuf_payload_offset,
        "FFN response header exceeds DMA-BUF prefix");

} // namespace ffn_split
