// Phone-side Gemma global-attention context-shard probe.
//
// usage:
//   gemma_global_attention_worker <total_kv> <segment_offset>
//       <segment_tokens> <backend>

#include "gemma_global_attention_protocol.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

static bool read_exact(int fd, void * destination, size_t size) {
    uint8_t * pointer = static_cast<uint8_t *>(destination);
    while (size > 0) {
        const ssize_t count = read(fd, pointer, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        pointer += count;
        size -= (size_t) count;
    }
    return true;
}

static bool write_exact(int fd, const void * source, size_t size) {
    const uint8_t * pointer = static_cast<const uint8_t *>(source);
    while (size > 0) {
        const ssize_t count = write(fd, pointer, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        pointer += count;
        size -= (size_t) count;
    }
    return true;
}

static double now_us() {
    timespec value = {};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double) value.tv_sec * 1e6 + (double) value.tv_nsec / 1e3;
}

static uint32_t elapsed_us(double started) {
    return (uint32_t) std::min(
            now_us() - started, (double) UINT32_MAX);
}

struct attention_graph {
    ggml_tensor * q;
    ggml_tensor * key;
    ggml_tensor * value;
    ggml_tensor * packed;
    ggml_cgraph * graph;
};

static attention_graph build_attention_graph(
        ggml_context * context,
        ggml_context * input_context,
        int64_t n_kv) {
    attention_graph result = {};
    result.q = ggml_new_tensor_2d(
            input_context, GGML_TYPE_F32,
            S41_GEMMA_ATTN_HEAD_DIM, S41_GEMMA_ATTN_HEADS);
    ggml_set_input(result.q);
    result.key = ggml_new_tensor_2d(
            context, GGML_TYPE_F16,
            S41_GEMMA_ATTN_HEAD_DIM, n_kv);
    result.value = ggml_new_tensor_2d(
            context, GGML_TYPE_F16,
            n_kv, S41_GEMMA_ATTN_HEAD_DIM);

    ggml_tensor * scores = ggml_mul_mat(context, result.key, result.q);
    ggml_tensor * probabilities = ggml_soft_max_ext(
            context, scores, nullptr,
            1.0f / std::sqrt((float) S41_GEMMA_ATTN_HEAD_DIM), 0.0f);
    ggml_tensor * state =
            ggml_mul_mat(context, result.value, probabilities);
    ggml_tensor * score_anchor_view = ggml_view_2d(
            context, scores, 1, S41_GEMMA_ATTN_HEADS,
            scores->nb[1], 0);
    ggml_tensor * probability_anchor_view = ggml_view_2d(
            context, probabilities, 1, S41_GEMMA_ATTN_HEADS,
            probabilities->nb[1], 0);
    ggml_tensor * score_anchor =
            ggml_cont(context, score_anchor_view);
    ggml_tensor * probability_anchor =
            ggml_cont(context, probability_anchor_view);
    result.packed = ggml_concat(
            context,
            ggml_concat(context, state, score_anchor, 0),
            probability_anchor, 0);
    result.graph = ggml_new_graph_custom(context, 32, false);
    ggml_build_forward_expand(result.graph, result.packed);
    return result;
}

static uint64_t initialize_kv(
        ggml_tensor * key,
        ggml_tensor * value,
        uint32_t offset,
        uint32_t tokens) {
    const size_t elements =
            (size_t) tokens * S41_GEMMA_ATTN_HEAD_DIM;
    std::vector<ggml_fp16_t> data(elements);
    for (uint32_t token = 0; token < tokens; ++token) {
        const uint32_t global_token = offset + token;
        for (uint32_t dimension = 0;
             dimension < S41_GEMMA_ATTN_HEAD_DIM; ++dimension) {
            data[(size_t) token * S41_GEMMA_ATTN_HEAD_DIM + dimension] =
                    ggml_fp32_to_fp16(
                            s41_gemma_k_value(global_token, dimension));
        }
    }
    uint64_t hash = s41_gemma_hash_bytes(data.data(),
            data.size() * sizeof(data[0]));
    ggml_backend_tensor_set(
            key, data.data(), 0, data.size() * sizeof(data[0]));

    for (uint32_t dimension = 0;
         dimension < S41_GEMMA_ATTN_HEAD_DIM; ++dimension) {
        for (uint32_t token = 0; token < tokens; ++token) {
            data[(size_t) dimension * tokens + token] =
                    ggml_fp32_to_fp16(s41_gemma_v_value(
                            offset + token, dimension));
        }
    }
    hash = s41_gemma_hash_bytes(
            data.data(), data.size() * sizeof(data[0]), hash);
    ggml_backend_tensor_set(
            value, data.data(), 0, data.size() * sizeof(data[0]));
    return hash;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr,
                "usage: %s <total_kv> <segment_offset> "
                "<segment_tokens> <backend>\n",
                argv[0]);
        return 2;
    }
    const int64_t total_kv = atoll(argv[1]);
    const int64_t segment_offset = atoll(argv[2]);
    const int64_t segment_tokens = atoll(argv[3]);
    const std::string backend_name = argv[4];
    if (total_kv <= 0 || segment_offset < 0 || segment_tokens <= 0 ||
        segment_offset + segment_tokens != total_kv ||
        segment_tokens % 256 != 0) {
        fprintf(stderr, "[gemma-attention-worker] invalid context shard\n");
        return 2;
    }
    const char * max_requests_text =
            getenv("S41_GEMMA_ATTN_MAX_REQUESTS");
    const long max_requests = max_requests_text != nullptr
            ? atol(max_requests_text) : 0;
    if (max_requests < 0) {
        fprintf(stderr, "[gemma-attention-worker] invalid request limit\n");
        return 2;
    }

    ggml_backend_t backend = nullptr;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (std::string(ggml_backend_dev_name(device)).find(backend_name) !=
            std::string::npos) {
            backend = ggml_backend_dev_init(device, nullptr);
            fprintf(stderr,
                    "[gemma-attention-worker] backend=%s description=%s\n",
                    ggml_backend_dev_name(device),
                    ggml_backend_dev_description(device));
            break;
        }
    }
    if (backend == nullptr) {
        fprintf(stderr, "[gemma-attention-worker] backend not found\n");
        return 1;
    }

    ggml_init_params parameters = {};
    parameters.mem_size =
            ggml_tensor_overhead() * 24 + ggml_graph_overhead() * 2;
    parameters.no_alloc = true;
    ggml_context * context = ggml_init(parameters);
    ggml_init_params input_parameters = {};
    input_parameters.mem_size = ggml_tensor_overhead() * 2;
    input_parameters.no_alloc = true;
    ggml_context * input_context = ggml_init(input_parameters);
    if (context == nullptr || input_context == nullptr) {
        return 1;
    }

    attention_graph attention = build_attention_graph(
            context, input_context, segment_tokens);
    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(context, backend);
    ggml_backend_buffer_t input_buffer =
            ggml_backend_alloc_ctx_tensors(input_context, backend);
    if (buffer == nullptr || input_buffer == nullptr) {
        fprintf(stderr, "[gemma-attention-worker] allocation failed\n");
        return 1;
    }
    ggml_backend_buffer_set_usage(
            input_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    fprintf(stderr, "[gemma-attention-worker] initializing KV shard\n");
    const uint64_t kv_hash = initialize_kv(
            attention.key, attention.value,
            (uint32_t) segment_offset, (uint32_t) segment_tokens);
    fprintf(stderr,
            "[gemma-attention-worker] ready total_kv=%lld "
            "segment=[%lld,%lld) kv_hash=%016llx nodes=%d\n",
            (long long) total_kv,
            (long long) segment_offset,
            (long long) (segment_offset + segment_tokens),
            (unsigned long long) kv_hash,
            ggml_graph_n_nodes(attention.graph));
    fflush(stderr);

    const size_t q_bytes =
            S41_GEMMA_ATTN_Q_ELEMENTS * sizeof(ggml_fp16_t);
    const size_t request_bytes =
            sizeof(s41_gemma_attention_request) + q_bytes;
    const size_t response_bytes =
            sizeof(s41_gemma_attention_response) +
            S41_GEMMA_ATTN_STATE_BYTES +
            S41_GEMMA_ATTN_ANCHOR_BYTES;
    std::vector<uint8_t> request_data(request_bytes);
    std::vector<uint8_t> response_data(response_bytes);
    std::vector<ggml_fp16_t> q_f16(S41_GEMMA_ATTN_Q_ELEMENTS);
    std::vector<float> q_f32(S41_GEMMA_ATTN_Q_ELEMENTS);
    std::vector<float> packed(
            S41_GEMMA_ATTN_PACKED_STRIDE *
            S41_GEMMA_ATTN_HEADS);
    std::vector<ggml_fp16_t> state(S41_GEMMA_ATTN_Q_ELEMENTS);
    std::vector<float> anchors(2 * S41_GEMMA_ATTN_HEADS);

    long completed_requests = 0;
    bool finished = false;
    while (!finished) {
        const int descriptor = open("/dev/usb_accessory", O_RDWR);
        if (descriptor < 0) {
            fprintf(stderr,
                    "[gemma-attention-worker] accessory open: %s\n",
                    strerror(errno));
            sleep(1);
            continue;
        }
        fprintf(stderr, "[gemma-attention-worker] endpoint open\n");
        fflush(stderr);

        for (;;) {
            if (!read_exact(
                        descriptor, request_data.data(),
                        request_data.size())) {
                break;
            }
            s41_gemma_attention_request request = {};
            memcpy(&request, request_data.data(), sizeof(request));
            const void * q_data =
                    request_data.data() + sizeof(request);
            if (request.magic != S41_GEMMA_ATTN_REQUEST_MAGIC ||
                request.version != S41_GEMMA_ATTN_PROTOCOL_VERSION ||
                request.flags != 0 ||
                request.total_kv != (uint32_t) total_kv ||
                request.segment_offset != (uint32_t) segment_offset ||
                request.segment_tokens != (uint32_t) segment_tokens ||
                request.q_bytes != q_bytes ||
                request.q_hash != s41_gemma_hash_bytes(q_data, q_bytes) ||
                request.kv_hash != kv_hash) {
                fprintf(stderr, "[gemma-attention-worker] invalid request\n");
                break;
            }
            memcpy(q_f16.data(), q_data, q_bytes);
            ggml_fp16_to_fp32_row(
                    q_f16.data(), q_f32.data(), q_f32.size());

            double started = now_us();
            ggml_backend_tensor_set(
                    attention.q, q_f32.data(), 0,
                    q_f32.size() * sizeof(float));
            const uint32_t set_us = elapsed_us(started);

            started = now_us();
            const enum ggml_status status =
                    ggml_backend_graph_compute(backend, attention.graph);
            const uint32_t compute_us = elapsed_us(started);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr,
                        "[gemma-attention-worker] compute failed: %d\n",
                        (int) status);
                break;
            }

            started = now_us();
            ggml_backend_tensor_get(
                    attention.packed, packed.data(), 0,
                    packed.size() * sizeof(float));
            const uint32_t get_us = elapsed_us(started);

            started = now_us();
            for (uint32_t head = 0;
                 head < S41_GEMMA_ATTN_HEADS; ++head) {
                const size_t packed_base =
                        (size_t) head * S41_GEMMA_ATTN_PACKED_STRIDE;
                const size_t state_base =
                        (size_t) head * S41_GEMMA_ATTN_HEAD_DIM;
                ggml_fp32_to_fp16_row(
                        packed.data() + packed_base,
                        state.data() + state_base,
                        S41_GEMMA_ATTN_HEAD_DIM);
                anchors[head] =
                        packed[packed_base + S41_GEMMA_ATTN_HEAD_DIM];
                anchors[S41_GEMMA_ATTN_HEADS + head] =
                        packed[packed_base + S41_GEMMA_ATTN_HEAD_DIM + 1];
            }
            const uint32_t encode_us = elapsed_us(started);

            uint8_t * state_output =
                    response_data.data() +
                    sizeof(s41_gemma_attention_response);
            uint8_t * anchor_output =
                    state_output + S41_GEMMA_ATTN_STATE_BYTES;
            memcpy(state_output, state.data(),
                    S41_GEMMA_ATTN_STATE_BYTES);
            memcpy(anchor_output, anchors.data(),
                    S41_GEMMA_ATTN_ANCHOR_BYTES);
            uint64_t output_hash = s41_gemma_hash_bytes(
                    state_output, S41_GEMMA_ATTN_STATE_BYTES);
            output_hash = s41_gemma_hash_bytes(
                    anchor_output, S41_GEMMA_ATTN_ANCHOR_BYTES,
                    output_hash);

            s41_gemma_attention_response response = {};
            response.magic = S41_GEMMA_ATTN_RESPONSE_MAGIC;
            response.version = S41_GEMMA_ATTN_PROTOCOL_VERSION;
            response.request_id = request.request_id;
            response.state_bytes = S41_GEMMA_ATTN_STATE_BYTES;
            response.anchor_bytes = S41_GEMMA_ATTN_ANCHOR_BYTES;
            response.output_hash = output_hash;
            response.kv_hash = kv_hash;
            response.set_us = set_us;
            response.compute_us = compute_us;
            response.get_us = get_us;
            response.encode_us = encode_us;
            memcpy(response_data.data(), &response, sizeof(response));
            if (!write_exact(
                        descriptor,
                        response_data.data(), response_data.size())) {
                break;
            }
            ++completed_requests;
            finished = max_requests > 0 &&
                    completed_requests >= max_requests;
            if (finished) {
                break;
            }
        }
        close(descriptor);
    }

    ggml_backend_buffer_free(input_buffer);
    ggml_backend_buffer_free(buffer);
    ggml_free(input_context);
    ggml_free(context);
    ggml_backend_free(backend);
    return 0;
}
