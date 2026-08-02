// Phone-side Qwen GQA-group attention worker with a resident KV slice.
//
// The bounded probe writes the projected K and V for the current token into
// the last cache slot before running attention.
//
// usage:
//   causal_attention_worker <K> <n_kv> <group_offset> <group_count>
//       <backend> <type>

#include "causal_attention_protocol.h"
#include "causal_quantized_weights.h"

#include "ggml.h"
#include "ggml-alloc.h"
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

static constexpr int64_t S41_QWEN_HEAD_DIM = 128;
static constexpr int64_t S41_QWEN_HEADS = 40;
static constexpr int64_t S41_QWEN_KV_HEADS = 8;
static constexpr int64_t S41_QWEN_GQA = 5;

static bool read_exact(int fd, void * destination, size_t size) {
    uint8_t * pointer = (uint8_t *) destination;
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
    const uint8_t * pointer = (const uint8_t *) source;
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

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
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

static bool initialize_qkv(
        ggml_tensor * tensor,
        ggml_type type,
        uint32_t q_row_offset,
        uint32_t q_rows,
        uint32_t kv_row_offset,
        uint32_t kv_rows,
        uint64_t & weight_hash) {
    const uint32_t columns = (uint32_t) tensor->ne[0];
    const size_t q_bytes = ggml_row_size(type, columns) * q_rows;
    const size_t kv_bytes = ggml_row_size(type, columns) * kv_rows;
    std::vector<uint8_t> quantized(ggml_nbytes(tensor));
    if (!s41_fill_quantized_weight(
                type, S41_WEIGHT_Q, q_row_offset, 0, columns, q_rows,
                quantized.data(), q_bytes) ||
        !s41_fill_quantized_weight(
                type, S41_WEIGHT_K, kv_row_offset, 0, columns, kv_rows,
                quantized.data() + q_bytes, kv_bytes) ||
        !s41_fill_quantized_weight(
                type, S41_WEIGHT_V, kv_row_offset, 0, columns, kv_rows,
                quantized.data() + q_bytes + kv_bytes, kv_bytes)) {
        return false;
    }
    weight_hash = s41_weight_hash_update(
            weight_hash, S41_WEIGHT_Q, q_row_offset, 0, columns, q_rows,
            quantized.data(), q_bytes);
    weight_hash = s41_weight_hash_update(
            weight_hash, S41_WEIGHT_K, kv_row_offset, 0, columns, kv_rows,
            quantized.data() + q_bytes, kv_bytes);
    weight_hash = s41_weight_hash_update(
            weight_hash, S41_WEIGHT_V, kv_row_offset, 0, columns, kv_rows,
            quantized.data() + q_bytes + kv_bytes, kv_bytes);
    ggml_backend_tensor_set(tensor, quantized.data(), 0, quantized.size());
    return true;
}

static bool initialize_output(
        ggml_tensor * tensor,
        ggml_type type,
        uint32_t column_offset,
        uint64_t & weight_hash) {
    const uint32_t columns = (uint32_t) tensor->ne[0];
    const uint32_t rows = (uint32_t) tensor->ne[1];
    std::vector<uint8_t> quantized(ggml_nbytes(tensor));
    if (!s41_fill_quantized_weight(
                type, S41_WEIGHT_O, 0, column_offset, columns, rows,
                quantized.data(), quantized.size())) {
        return false;
    }
    weight_hash = s41_weight_hash_update(
            weight_hash, S41_WEIGHT_O, 0, column_offset, columns, rows,
            quantized.data(), quantized.size());
    ggml_backend_tensor_set(tensor, quantized.data(), 0, quantized.size());
    return true;
}

static void initialize_cache(
        ggml_tensor * tensor,
        uint32_t matrix,
        uint32_t group_offset,
        uint32_t group_count,
        uint32_t n_kv) {
    const size_t elements = ggml_nelements(tensor);
    std::vector<float> data_f32(elements);
    for (uint32_t group = 0; group < group_count; ++group) {
        for (uint32_t token = 0; token < n_kv; ++token) {
            for (uint32_t dimension = 0;
                 dimension < S41_QWEN_HEAD_DIM; ++dimension) {
                const size_t index =
                        ((size_t) group * n_kv + token) *
                                S41_QWEN_HEAD_DIM + dimension;
                data_f32[index] = s41_kv_value(
                        matrix, group_offset + group, token, dimension);
            }
        }
    }
    std::vector<ggml_fp16_t> data_f16(elements);
    ggml_fp32_to_fp16_row(data_f32.data(), data_f16.data(), elements);
    ggml_backend_tensor_set(
            tensor, data_f16.data(), 0,
            data_f16.size() * sizeof(ggml_fp16_t));
}

int main(int argc, char ** argv) {
    if (argc != 7) {
        fprintf(stderr,
                "usage: %s <K> <n_kv> <group_offset> <group_count> "
                "<backend> <type>\n",
                argv[0]);
        return 2;
    }

    const int64_t k = atoll(argv[1]);
    const int64_t n_kv = atoll(argv[2]);
    const int64_t group_offset = atoll(argv[3]);
    const int64_t group_count = atoll(argv[4]);
    const std::string backend_name = argv[5];
    ggml_type weight_type = GGML_TYPE_COUNT;
    if (!parse_type(argv[6], weight_type) ||
        weight_type != GGML_TYPE_Q8_0) {
        fprintf(stderr, "[attention-worker] only q8_0 is supported\n");
        return 2;
    }
    if (k != 5120 || n_kv <= 0 || group_offset < 0 ||
        group_count <= 0 ||
        group_offset + group_count != S41_QWEN_KV_HEADS) {
        fprintf(stderr, "[attention-worker] invalid dimensions\n");
        return 2;
    }

    const int64_t q_heads = group_count * S41_QWEN_GQA;
    const int64_t q_width = q_heads * S41_QWEN_HEAD_DIM;
    const int64_t kv_width = group_count * S41_QWEN_HEAD_DIM;
    const int64_t q_row_offset =
            group_offset * S41_QWEN_GQA * S41_QWEN_HEAD_DIM;
    const int64_t kv_row_offset =
            group_offset * S41_QWEN_HEAD_DIM;
    const int64_t qkv_width = q_width + 2 * kv_width;
    const size_t input_bytes = sizeof(ggml_fp16_t) * (size_t) k;
    const size_t output_bytes = sizeof(ggml_fp16_t) * (size_t) k;
    const uint16_t flags = S41_ATTN_FLAG_FAST_HASH |
            S41_ATTN_FLAG_F16_INPUT |
            S41_ATTN_FLAG_F16_OUTPUT |
            S41_ATTN_FLAG_LAST_SLOT_UPDATE;

    ggml_backend_t backend = nullptr;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (std::string(ggml_backend_dev_name(device)).find(backend_name) !=
            std::string::npos) {
            backend = ggml_backend_dev_init(device, nullptr);
            fprintf(stderr,
                    "[attention-worker] backend=%s description=%s\n",
                    ggml_backend_dev_name(device),
                    ggml_backend_dev_description(device));
            break;
        }
    }
    if (backend == nullptr) {
        fprintf(stderr, "[attention-worker] backend not found\n");
        return 1;
    }

    ggml_init_params parameters = {};
    parameters.mem_size =
            ggml_tensor_overhead() * 40 + ggml_graph_overhead() * 2;
    parameters.no_alloc = true;
    ggml_context * context = ggml_init(parameters);
    if (context == nullptr) {
        return 1;
    }

    ggml_tensor * input =
            ggml_new_tensor_2d(context, GGML_TYPE_F32, k, 1);
    ggml_tensor * qkv =
            ggml_new_tensor_2d(context, weight_type, k, qkv_width);
    ggml_tensor * output_weight =
            ggml_new_tensor_2d(context, weight_type, q_width, k);
    ggml_tensor * key_cache = ggml_new_tensor_4d(
            context, GGML_TYPE_F16,
            S41_QWEN_HEAD_DIM, n_kv, group_count, 1);
    ggml_tensor * value_cache = ggml_new_tensor_4d(
            context, GGML_TYPE_F16,
            S41_QWEN_HEAD_DIM, n_kv, group_count, 1);

    ggml_tensor * qkv_output = ggml_mul_mat(context, qkv, input);
    ggml_tensor * q_output =
            ggml_view_1d(context, qkv_output, q_width, 0);
    ggml_tensor * k_output = ggml_view_1d(
            context, qkv_output, kv_width,
            sizeof(float) * (size_t) q_width);
    ggml_tensor * v_output = ggml_view_1d(
            context, qkv_output, kv_width,
            sizeof(float) * (size_t) (q_width + kv_width));
    ggml_tensor * k_current = ggml_reshape_3d(
            context, k_output,
            S41_QWEN_HEAD_DIM, 1, group_count);
    ggml_tensor * v_current = ggml_reshape_3d(
            context, v_output,
            S41_QWEN_HEAD_DIM, 1, group_count);
    ggml_tensor * key_slot = ggml_view_3d(
            context, key_cache,
            S41_QWEN_HEAD_DIM, 1, group_count,
            key_cache->nb[1], key_cache->nb[2],
            (size_t) (n_kv - 1) * key_cache->nb[1]);
    ggml_tensor * value_slot = ggml_view_3d(
            context, value_cache,
            S41_QWEN_HEAD_DIM, 1, group_count,
            value_cache->nb[1], value_cache->nb[2],
            (size_t) (n_kv - 1) * value_cache->nb[1]);
    ggml_tensor * key_store = ggml_cpy(context, k_current, key_slot);
    ggml_tensor * value_store = ggml_cpy(context, v_current, value_slot);
    ggml_tensor * q_heads_tensor = ggml_permute(
            context,
            ggml_reshape_4d(
                    context, q_output,
                    S41_QWEN_HEAD_DIM, q_heads, 1, 1),
            0, 2, 1, 3);
    ggml_tensor * attended = ggml_flash_attn_ext(
            context, q_heads_tensor, key_cache, value_cache, nullptr,
            1.0f / std::sqrt((float) S41_QWEN_HEAD_DIM), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
    ggml_tensor * merged =
            ggml_reshape_2d(context, attended, q_width, 1);
    ggml_tensor * output =
            ggml_mul_mat(context, output_weight, merged);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        fprintf(stderr, "[attention-worker] allocation failed\n");
        return 1;
    }

    uint64_t weight_hash = S41_HASH64_OFFSET;
    if (!initialize_qkv(
                qkv, weight_type,
                (uint32_t) q_row_offset, (uint32_t) q_width,
                (uint32_t) kv_row_offset, (uint32_t) kv_width,
                weight_hash) ||
        !initialize_output(
                output_weight, weight_type,
                (uint32_t) q_row_offset, weight_hash)) {
        fprintf(stderr, "[attention-worker] weight initialization failed\n");
        return 1;
    }
    initialize_cache(
            key_cache, S41_WEIGHT_K, (uint32_t) group_offset,
            (uint32_t) group_count, (uint32_t) n_kv);
    initialize_cache(
            value_cache, S41_WEIGHT_V, (uint32_t) group_offset,
            (uint32_t) group_count, (uint32_t) n_kv);

    ggml_cgraph * graph = ggml_new_graph_custom(context, 32, false);
    ggml_build_forward_expand(graph, key_store);
    ggml_build_forward_expand(graph, value_store);
    ggml_build_forward_expand(graph, output);
    fprintf(stderr,
            "[attention-worker] ready K=%lld n_kv=%lld groups=[%lld,%lld) "
            "q_heads=%lld type=%s weight_hash=%016llx nodes=%d "
            "cache_mode=last_slot_update\n",
            (long long) k, (long long) n_kv,
            (long long) group_offset,
            (long long) (group_offset + group_count),
            (long long) q_heads, ggml_type_name(weight_type),
            (unsigned long long) weight_hash,
            ggml_graph_n_nodes(graph));
    fflush(stderr);

    std::vector<uint8_t> request_data(
            sizeof(s41_attention_request) + input_bytes);
    std::vector<ggml_fp16_t> input_f16((size_t) k);
    std::vector<float> input_f32((size_t) k);
    std::vector<float> output_f32((size_t) k);
    std::vector<ggml_fp16_t> output_f16((size_t) k);
    std::vector<uint8_t> response_data(
            sizeof(s41_attention_response) + output_bytes);
    std::vector<double> compute_times;
    std::vector<double> request_times;

    for (;;) {
        const int descriptor = open("/dev/usb_accessory", O_RDWR);
        if (descriptor < 0) {
            fprintf(stderr, "[attention-worker] accessory open: %s\n",
                    strerror(errno));
            sleep(1);
            continue;
        }
        fprintf(stderr, "[attention-worker] endpoint open\n");
        fflush(stderr);

        for (;;) {
            if (!read_exact(
                        descriptor, request_data.data(),
                        request_data.size())) {
                break;
            }
            const double request_started = now_us();
            s41_attention_request request = {};
            memcpy(&request, request_data.data(), sizeof(request));
            if (request.magic != S41_ATTN_REQUEST_MAGIC ||
                request.version != S41_ATTN_PROTOCOL_VERSION ||
                request.flags != flags ||
                request.type != (uint32_t) weight_type ||
                request.k != (uint32_t) k ||
                request.n_kv != (uint32_t) n_kv ||
                request.n_heads != S41_QWEN_HEADS ||
                request.n_kv_heads != S41_QWEN_KV_HEADS ||
                request.group_offset != (uint32_t) group_offset ||
                request.group_count != (uint32_t) group_count ||
                request.input_bytes != input_bytes ||
                request.weight_hash != weight_hash) {
                fprintf(stderr, "[attention-worker] invalid request\n");
                break;
            }
            memcpy(
                    input_f16.data(),
                    request_data.data() + sizeof(request), input_bytes);
            if (s41_fast_hash_bytes(input_f16.data(), input_bytes) !=
                request.input_hash) {
                fprintf(stderr, "[attention-worker] input hash mismatch\n");
                break;
            }
            ggml_fp16_to_fp32_row(
                    input_f16.data(), input_f32.data(), (size_t) k);

            double started = now_us();
            ggml_backend_tensor_set(
                    input, input_f32.data(), 0,
                    input_f32.size() * sizeof(float));
            const uint32_t set_us = elapsed_us(started);

            started = now_us();
            const enum ggml_status status =
                    ggml_backend_graph_compute(backend, graph);
            const uint32_t compute_us = elapsed_us(started);
            compute_times.push_back((double) compute_us);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[attention-worker] compute failed: %d\n",
                        (int) status);
                break;
            }

            started = now_us();
            ggml_backend_tensor_get(
                    output, output_f32.data(), 0,
                    output_f32.size() * sizeof(float));
            const uint32_t get_us = elapsed_us(started);
            ggml_fp32_to_fp16_row(
                    output_f32.data(), output_f16.data(), (size_t) k);

            s41_attention_response response = {};
            response.magic = S41_ATTN_RESPONSE_MAGIC;
            response.version = S41_ATTN_PROTOCOL_VERSION;
            response.request_id = request.request_id;
            response.output_elements = (uint32_t) k;
            response.output_bytes = (uint32_t) output_bytes;
            response.output_hash =
                    s41_fast_hash_bytes(output_f16.data(), output_bytes);
            response.weight_hash = weight_hash;
            response.set_us = set_us;
            response.compute_us = compute_us;
            response.get_us = get_us;
            memcpy(response_data.data(), &response, sizeof(response));
            memcpy(
                    response_data.data() + sizeof(response),
                    output_f16.data(), output_bytes);
            if (!write_exact(
                        descriptor, response_data.data(),
                        response_data.size())) {
                break;
            }
            request_times.push_back(now_us() - request_started);
            if (request_times.size() % 100 == 0) {
                fprintf(stderr,
                        "[attention-worker] requests=%zu compute_us=%.0f "
                        "request_us=%.0f\n",
                        request_times.size(), median(compute_times),
                        median(request_times));
                fflush(stderr);
            }
        }
        close(descriptor);
    }
}
