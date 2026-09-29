// GGML graph control for Qwen3-14B-shaped phone operators.
//
// usage:
//   model_ops_graph_worker <rmsnorm|swiglu|attention> <backend>
//       <n_kv> <requests>

#include "model_ops_protocol.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <signal.h>
#include <string>
#include <unistd.h>
#include <vector>
#include <zlib.h>

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
        size -= static_cast<size_t>(count);
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
        size -= static_cast<size_t>(count);
    }
    return true;
}

static uint64_t now_ns() {
    timespec value = {};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL +
            static_cast<uint64_t>(value.tv_nsec);
}

static uint32_t crc32_bytes(const void * data, size_t size) {
    return static_cast<uint32_t>(crc32(
            0, static_cast<const Bytef *>(data), static_cast<uInt>(size)));
}

static bool parse_op(const std::string & name, s41_model_op & op) {
    if (name == "rmsnorm") {
        op = S41_MODEL_OP_RMSNORM;
        return true;
    }
    if (name == "swiglu") {
        op = S41_MODEL_OP_SWIGLU;
        return true;
    }
    if (name == "attention") {
        op = S41_MODEL_OP_ATTENTION;
        return true;
    }
    return false;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr,
                "usage: %s <rmsnorm|swiglu|attention> <backend> "
                "<n_kv> <requests>\n",
                argv[0]);
        return 2;
    }

    s41_model_op op = S41_MODEL_OP_RMSNORM;
    const std::string op_name = argv[1];
    const std::string backend_name = argv[2];
    const int64_t n_kv = atoll(argv[3]);
    const int max_requests = atoi(argv[4]);
    if (!parse_op(op_name, op) || n_kv <= 0 || n_kv > (1 << 20) ||
        max_requests <= 0) {
        fprintf(stderr, "[model-graph] invalid configuration\n");
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);

    ggml_backend_t backend = nullptr;
    const char * selected_name = nullptr;
    const char * selected_description = nullptr;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const char * device_name = ggml_backend_dev_name(device);
        if (std::string(device_name).find(backend_name) != std::string::npos) {
            backend = ggml_backend_dev_init(device, nullptr);
            selected_name = device_name;
            selected_description = ggml_backend_dev_description(device);
            break;
        }
    }
    if (backend == nullptr) {
        fprintf(stderr, "[model-graph] backend '%s' not found\n",
                backend_name.c_str());
        return 1;
    }

    ggml_init_params parameters = {};
    parameters.mem_size = ggml_tensor_overhead() * 24 +
            ggml_graph_overhead_custom(24, false);
    parameters.no_alloc = true;
    ggml_context * context = ggml_init(parameters);
    if (context == nullptr) {
        return 1;
    }

    ggml_tensor * input0 = nullptr;
    ggml_tensor * input1 = nullptr;
    ggml_tensor * norm_weight = nullptr;
    ggml_tensor * key = nullptr;
    ggml_tensor * value = nullptr;
    ggml_tensor * output = nullptr;

    if (op == S41_MODEL_OP_RMSNORM) {
        input0 = ggml_new_tensor_1d(
                context, GGML_TYPE_F32, S41_QWEN3_14B_HIDDEN);
        norm_weight = ggml_new_tensor_1d(
                context, GGML_TYPE_F32, S41_QWEN3_14B_HIDDEN);
        output = ggml_mul(
                context, ggml_rms_norm(context, input0, 1e-6f),
                norm_weight);
    } else if (op == S41_MODEL_OP_SWIGLU) {
        input0 = ggml_new_tensor_1d(
                context, GGML_TYPE_F32, S41_QWEN3_14B_INTERMEDIATE);
        input1 = ggml_new_tensor_1d(
                context, GGML_TYPE_F32, S41_QWEN3_14B_INTERMEDIATE);
        output = ggml_swiglu_split(context, input0, input1);
    } else {
        input0 = ggml_new_tensor_4d(
                context, GGML_TYPE_F32, S41_QWEN3_14B_HEAD_DIM, 1,
                S41_QWEN3_14B_GQA, 1);
        key = ggml_new_tensor_4d(
                context, GGML_TYPE_F16, S41_QWEN3_14B_HEAD_DIM, n_kv,
                1, 1);
        value = ggml_new_tensor_4d(
                context, GGML_TYPE_F16, S41_QWEN3_14B_HEAD_DIM, n_kv,
                1, 1);
        output = ggml_flash_attn_ext(
                context, input0, key, value, nullptr,
                1.0f / std::sqrt((float) S41_QWEN3_14B_HEAD_DIM),
                0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    }
    ggml_set_input(input0);
    if (input1 != nullptr) {
        ggml_set_input(input1);
    }
    ggml_set_output(output);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        fprintf(stderr, "[model-graph] backend allocation failed\n");
        return 1;
    }
    ggml_backend_buffer_set_usage(
            buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    if (norm_weight != nullptr) {
        std::vector<float> weights(S41_QWEN3_14B_HIDDEN);
        std::vector<ggml_fp16_t> rounded(S41_QWEN3_14B_HIDDEN);
        for (uint32_t index = 0; index < S41_QWEN3_14B_HIDDEN; ++index) {
            weights[index] = s41_model_norm_weight(index);
        }
        ggml_fp32_to_fp16_row(weights.data(), rounded.data(), weights.size());
        ggml_fp16_to_fp32_row(rounded.data(), weights.data(), weights.size());
        ggml_backend_tensor_set(
                norm_weight, weights.data(), 0,
                weights.size() * sizeof(float));
    }
    if (key != nullptr) {
        const size_t cache_elements = static_cast<size_t>(n_kv) *
                S41_QWEN3_14B_HEAD_DIM;
        std::vector<float> cache_f32(cache_elements);
        std::vector<ggml_fp16_t> cache_f16(cache_elements);
        for (int64_t token = 0; token < n_kv; ++token) {
            for (uint32_t dimension = 0;
                 dimension < S41_QWEN3_14B_HEAD_DIM; ++dimension) {
                const size_t index = static_cast<size_t>(token) *
                        S41_QWEN3_14B_HEAD_DIM + dimension;
                cache_f32[index] = s41_model_key_value(
                        static_cast<uint32_t>(token), dimension);
            }
        }
        ggml_fp32_to_fp16_row(
                cache_f32.data(), cache_f16.data(), cache_elements);
        ggml_backend_tensor_set(
                key, cache_f16.data(), 0,
                cache_f16.size() * sizeof(ggml_fp16_t));
        for (int64_t token = 0; token < n_kv; ++token) {
            for (uint32_t dimension = 0;
                 dimension < S41_QWEN3_14B_HEAD_DIM; ++dimension) {
                const size_t index = static_cast<size_t>(token) *
                        S41_QWEN3_14B_HEAD_DIM + dimension;
                cache_f32[index] = s41_model_value_value(
                        static_cast<uint32_t>(token), dimension);
            }
        }
        ggml_fp32_to_fp16_row(
                cache_f32.data(), cache_f16.data(), cache_elements);
        ggml_backend_tensor_set(
                value, cache_f16.data(), 0,
                cache_f16.size() * sizeof(ggml_fp16_t));
    }

    ggml_cgraph * graph = ggml_new_graph_custom(context, 24, false);
    ggml_build_forward_expand(graph, output);

    const uint32_t input_elements = s41_model_op_input_elements(op);
    const uint32_t output_elements = s41_model_op_output_elements(op);
    const size_t input_bytes = static_cast<size_t>(input_elements) *
            sizeof(ggml_fp16_t);
    const size_t output_bytes = static_cast<size_t>(output_elements) *
            sizeof(ggml_fp16_t);
    std::vector<uint8_t> request_bytes(
            sizeof(s41_model_op_request) + input_bytes);
    std::vector<uint8_t> response_bytes(
            sizeof(s41_model_op_response) + output_bytes);
    std::vector<float> input_f32(input_elements);
    std::vector<float> output_f32(output_elements);
    std::vector<ggml_fp16_t> output_f16(output_elements);

    fprintf(stderr,
            "[model-graph] ready op=%s backend=%s description=%s "
            "n_kv=%lld nodes=%d input_elements=%u output_elements=%u "
            "requests=%d\n",
            op_name.c_str(), selected_name, selected_description,
            static_cast<long long>(n_kv), ggml_graph_n_nodes(graph),
            input_elements, output_elements, max_requests);
    fflush(stderr);

    uint64_t previous_write_ns = 0;
    int served = 0;
    for (;;) {
        const int descriptor = open("/dev/usb_accessory", O_RDWR);
        if (descriptor < 0) {
            fprintf(stderr, "[model-graph] endpoint open: %s\n",
                    strerror(errno));
            sleep(1);
            continue;
        }
        fprintf(stderr, "[model-graph] endpoint open\n");
        fflush(stderr);

        while (served < max_requests && read_exact(
                    descriptor, request_bytes.data(), request_bytes.size())) {
            const uint64_t request_started = now_ns();
            uint64_t started = request_started;
            s41_model_op_request request = {};
            memcpy(&request, request_bytes.data(), sizeof(request));
            const uint8_t * encoded_input =
                    request_bytes.data() + sizeof(request);
            const bool request_ok =
                    request.magic == S41_MODEL_OP_REQUEST_MAGIC &&
                    request.version == S41_MODEL_OP_PROTOCOL_VERSION &&
                    request.op == static_cast<uint16_t>(op) &&
                    request.request_id != 0 &&
                    request.request_id != 0xffffffffU &&
                    request.input_elements == input_elements &&
                    request.output_elements == output_elements &&
                    request.input_bytes == input_bytes &&
                    request.input_crc32 ==
                            crc32_bytes(encoded_input, input_bytes);
            const uint64_t validate_ns = now_ns() - started;
            if (!request_ok) {
                fprintf(stderr, "[model-graph] invalid request\n");
                close(descriptor);
                return 3;
            }

            started = now_ns();
            ggml_fp16_to_fp32_row(
                    reinterpret_cast<const ggml_fp16_t *>(encoded_input),
                    input_f32.data(), input_elements);
            const uint64_t decode_ns = now_ns() - started;

            started = now_ns();
            const size_t first_elements = op == S41_MODEL_OP_SWIGLU
                    ? S41_QWEN3_14B_INTERMEDIATE : input_elements;
            ggml_backend_tensor_set(
                    input0, input_f32.data(), 0,
                    first_elements * sizeof(float));
            if (input1 != nullptr) {
                ggml_backend_tensor_set(
                        input1, input_f32.data() + first_elements, 0,
                        first_elements * sizeof(float));
            }
            const uint64_t set_ns = now_ns() - started;

            started = now_ns();
            const enum ggml_status status =
                    ggml_backend_graph_compute_async(backend, graph);
            const uint64_t submit_ns = now_ns() - started;
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[model-graph] compute failed: %d\n",
                        static_cast<int>(status));
                close(descriptor);
                return 3;
            }

            started = now_ns();
            ggml_backend_synchronize(backend);
            const uint64_t sync_ns = now_ns() - started;

            started = now_ns();
            ggml_backend_tensor_get(
                    output, output_f32.data(), 0,
                    output_elements * sizeof(float));
            const uint64_t get_ns = now_ns() - started;

            started = now_ns();
            ggml_fp32_to_fp16_row(
                    output_f32.data(), output_f16.data(), output_elements);
            const uint32_t output_crc =
                    crc32_bytes(output_f16.data(), output_bytes);
            const uint64_t encode_hash_ns = now_ns() - started;
            const uint64_t prewrite_ns = now_ns() - request_started;

            s41_model_op_response response = {};
            response.magic = S41_MODEL_OP_RESPONSE_MAGIC;
            response.version = S41_MODEL_OP_PROTOCOL_VERSION;
            response.request_id = request.request_id;
            response.op = static_cast<uint32_t>(op);
            response.output_elements = output_elements;
            response.output_bytes = output_bytes;
            response.output_crc32 = output_crc;
            response.validate_ns = validate_ns;
            response.decode_ns = decode_ns;
            response.set_ns = set_ns;
            response.submit_ns = submit_ns;
            response.sync_ns = sync_ns;
            response.get_ns = get_ns;
            response.encode_hash_ns = encode_hash_ns;
            response.prewrite_ns = prewrite_ns;
            response.previous_write_ns = previous_write_ns;
            memcpy(response_bytes.data(), &response, sizeof(response));
            memcpy(response_bytes.data() + sizeof(response),
                    output_f16.data(), output_bytes);

            started = now_ns();
            if (!write_exact(
                        descriptor, response_bytes.data(),
                        response_bytes.size())) {
                break;
            }
            previous_write_ns = now_ns() - started;
            ++served;
        }
        close(descriptor);
        if (served >= max_requests) {
            break;
        }
    }

    fprintf(stderr, "[model-graph] complete requests=%d\n", served);
    fflush(stderr);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    return 0;
}
