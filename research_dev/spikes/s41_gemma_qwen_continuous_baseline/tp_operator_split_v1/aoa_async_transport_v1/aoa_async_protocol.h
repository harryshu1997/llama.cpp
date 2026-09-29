#ifndef S41_AOA_ASYNC_PROTOCOL_H
#define S41_AOA_ASYNC_PROTOCOL_H

#include <stdint.h>

#define S41_AOA_REQUEST_MAGIC UINT64_C(0x5134414f41524551)
#define S41_AOA_RESPONSE_MAGIC UINT64_C(0x5134414f41525350)
#define S41_AOA_SENTINEL_MAGIC UINT64_C(0x9e3779b97f4a7c15)

struct s41_aoa_frame_header {
    uint64_t magic;
    uint64_t sequence;
    uint32_t request_bytes;
    uint32_t response_bytes;
    uint64_t sentinel;
};

#ifdef __cplusplus
static_assert(sizeof(s41_aoa_frame_header) == 32,
        "unexpected AOA frame header size");
#else
_Static_assert(sizeof(struct s41_aoa_frame_header) == 32,
        "unexpected AOA frame header size");
#endif

static inline uint64_t s41_aoa_sentinel(uint64_t sequence) {
    return sequence ^ S41_AOA_SENTINEL_MAGIC;
}

#endif
