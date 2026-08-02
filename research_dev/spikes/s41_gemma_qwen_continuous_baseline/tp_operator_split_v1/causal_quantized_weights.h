#ifndef S41_CAUSAL_QUANTIZED_WEIGHTS_H
#define S41_CAUSAL_QUANTIZED_WEIGHTS_H

#include "causal_ffn_protocol.h"

#include "ggml.h"
#include "ggml-quants.h"

#include <cstddef>
#include <cstdint>

static inline uint64_t s41_weight_bits(
        uint32_t matrix, uint32_t row, uint32_t column) {
    uint64_t key = (uint64_t) matrix << 56;
    key ^= (uint64_t) row * 0xd6e8feb86659fd93ULL;
    key ^= (uint64_t) column * 0xa5a3564e27f8862fULL;
    return s41_mix64(key);
}

static inline bool s41_fill_quantized_weight(
        ggml_type type,
        uint32_t matrix,
        uint32_t row_offset,
        uint32_t column_offset,
        uint32_t columns,
        uint32_t rows,
        void * destination,
        size_t destination_size) {
    const size_t row_bytes = ggml_row_size(type, columns);
    if (row_bytes * rows != destination_size || columns % 32 != 0) {
        return false;
    }

    uint8_t * bytes = (uint8_t *) destination;
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t global_row = row_offset + row;
        if (type == GGML_TYPE_Q4_0) {
            block_q4_0 * blocks =
                    (block_q4_0 *) (bytes + (size_t) row * row_bytes);
            for (uint32_t block = 0; block < columns / QK4_0; ++block) {
                blocks[block].d = (ggml_half) 0x1800;
                const uint32_t base = column_offset + block * QK4_0;
                for (uint32_t index = 0; index < QK4_0 / 2; ++index) {
                    const uint8_t low = (uint8_t) (
                            s41_weight_bits(
                                    matrix, global_row, base + index) & 0x0f);
                    const uint8_t high = (uint8_t) (
                            s41_weight_bits(
                                    matrix, global_row,
                                    base + index + QK4_0 / 2) & 0x0f);
                    blocks[block].qs[index] = low | (uint8_t) (high << 4);
                }
            }
        } else if (type == GGML_TYPE_Q8_0) {
            block_q8_0 * blocks =
                    (block_q8_0 *) (bytes + (size_t) row * row_bytes);
            for (uint32_t block = 0; block < columns / QK8_0; ++block) {
                blocks[block].d = (ggml_half) 0x0800;
                const uint32_t base = column_offset + block * QK8_0;
                for (uint32_t index = 0; index < QK8_0; ++index) {
                    const uint32_t value = (uint32_t) (
                            s41_weight_bits(
                                    matrix, global_row, base + index) % 255);
                    blocks[block].qs[index] = (int8_t) ((int) value - 127);
                }
            }
        } else {
            return false;
        }
    }
    return true;
}

#endif
