// Phone-side sharded vocabulary head with local greedy reduction.
//
// usage:
//   causal_head_worker <K> <vocab> <offset> <count> <backend> <type>
//       [input_type]

#include "causal_head_protocol.h"
#include "causal_quantized_weights.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
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
    const double value = now_us() - started;
    return (uint32_t) std::min(value, (double) UINT32_MAX);
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

static bool initialize_head(
        ggml_tensor * tensor,
        ggml_type type,
        uint32_t row_offset,
        uint64_t & weight_hash) {
    const uint32_t columns = (uint32_t) tensor->ne[0];
    const uint32_t rows = (uint32_t) tensor->ne[1];
    std::vector<uint8_t> quantized(ggml_nbytes(tensor));
    if (!s41_fill_quantized_weight(
                type, S41_WEIGHT_HEAD, row_offset, 0, columns, rows,
                quantized.data(), quantized.size())) {
        return false;
    }
    weight_hash = s41_weight_hash_update(
            weight_hash, S41_WEIGHT_HEAD, row_offset, 0, columns, rows,
            quantized.data(), quantized.size());
    ggml_backend_tensor_set(tensor, quantized.data(), 0, quantized.size());
    return true;
}

static bool candidate_better(
        const s41_head_candidate & first,
        const s41_head_candidate & second) {
    return first.score > second.score ||
            (first.score == second.score && first.token < second.token);
}

int main(int argc, char ** argv) {
    if (argc < 7 || argc > 8) {
        fprintf(stderr,
                "usage: %s <K> <vocab> <offset> <count> <backend> <type> "
                "[input_type]\n",
                argv[0]);
        return 2;
    }

    const int64_t k = atoll(argv[1]);
    const int64_t vocab = atoll(argv[2]);
    const int64_t offset = atoll(argv[3]);
    const int64_t count = atoll(argv[4]);
    const std::string backend_name = argv[5];
    ggml_type weight_type = GGML_TYPE_COUNT;
    if (!parse_type(argv[6], weight_type) ||
        (weight_type != GGML_TYPE_Q4_0 &&
         weight_type != GGML_TYPE_Q8_0)) {
        fprintf(stderr, "[head-worker] unsupported type '%s'\n", argv[6]);
        return 2;
    }

    const std::string input_name = argc > 7 ? argv[7] : "f32";
    const bool f16_input = input_name == "f16";
    const bool i8_input = input_name == "i8";
    if (input_name != "f32" && !f16_input && !i8_input) {
        fprintf(stderr, "[head-worker] unsupported input '%s'\n",
                input_name.c_str());
        return 2;
    }
    const uint16_t request_flags = S41_HEAD_FLAG_FAST_HASH |
            (f16_input ? S41_HEAD_FLAG_F16_INPUT : 0) |
            (i8_input ? S41_HEAD_FLAG_I8_INPUT : 0) |
            S41_HEAD_FLAG_TOP_K;
    const size_t input_bytes = i8_input
            ? sizeof(float) + (size_t) k
            : (f16_input ? sizeof(ggml_fp16_t) : sizeof(float)) *
                    (size_t) k;

    const int64_t block = ggml_blck_size(weight_type);
    if (k <= 0 || vocab <= 0 || offset < 0 || count <= 0 ||
        offset + count != vocab || k % block != 0) {
        fprintf(stderr, "[head-worker] invalid dimensions\n");
        return 2;
    }

    ggml_backend_t backend = nullptr;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (std::string(ggml_backend_dev_name(device)).find(backend_name) !=
            std::string::npos) {
            backend = ggml_backend_dev_init(device, nullptr);
            fprintf(stderr, "[head-worker] backend=%s description=%s\n",
                    ggml_backend_dev_name(device),
                    ggml_backend_dev_description(device));
            break;
        }
    }
    if (backend == nullptr) {
        fprintf(stderr, "[head-worker] backend not found\n");
        return 1;
    }

    ggml_init_params parameters = {};
    parameters.mem_size =
            ggml_tensor_overhead() * 12 + ggml_graph_overhead();
    parameters.no_alloc = true;
    ggml_context * context = ggml_init(parameters);
    if (context == nullptr) {
        return 1;
    }

    ggml_init_params input_parameters = {};
    input_parameters.mem_size = ggml_tensor_overhead();
    input_parameters.no_alloc = true;
    ggml_context * input_context = ggml_init(input_parameters);
    if (input_context == nullptr) {
        return 1;
    }

    ggml_tensor * input =
            ggml_new_tensor_2d(input_context, GGML_TYPE_F32, k, 1);
    ggml_set_input(input);
    ggml_tensor * head =
            ggml_new_tensor_2d(context, weight_type, k, count);
    ggml_tensor * logits = ggml_mul_mat(context, head, input);
    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        fprintf(stderr, "[head-worker] allocation failed\n");
        return 1;
    }
    ggml_backend_buffer_t input_buffer =
            ggml_backend_alloc_ctx_tensors(input_context, backend);
    if (input_buffer == nullptr) {
        fprintf(stderr, "[head-worker] input allocation failed\n");
        return 1;
    }
    ggml_backend_buffer_set_usage(
            input_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint64_t weight_hash = S41_HASH64_OFFSET;
    if (!initialize_head(
                head, weight_type, (uint32_t) offset, weight_hash)) {
        fprintf(stderr, "[head-worker] weight initialization failed\n");
        return 1;
    }
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, logits);
    fprintf(stderr,
            "[head-worker] ready K=%lld vocab=%lld slice=[%lld,%lld) "
            "type=%s input_type=%s weight_hash=%016llx nodes=%d\n",
            (long long) k, (long long) vocab, (long long) offset,
            (long long) (offset + count), ggml_type_name(weight_type),
            input_name.c_str(), (unsigned long long) weight_hash,
            ggml_graph_n_nodes(graph));
    fflush(stderr);

    std::vector<uint8_t> request_bytes(
            sizeof(s41_head_request) + input_bytes);
    std::vector<uint8_t> input_data(input_bytes);
    std::vector<float> input_f32(f16_input || i8_input ? (size_t) k : 0);
    std::vector<float> output_data((size_t) count);
    std::vector<double> compute_times;
    std::vector<double> get_times;
    std::vector<double> reduce_times;
    std::vector<double> request_times;
    std::vector<uint8_t> response_bytes(
            sizeof(s41_head_response) +
            sizeof(s41_head_candidate) * S41_HEAD_TOP_K);

    for (;;) {
        const int descriptor = open("/dev/usb_accessory", O_RDWR);
        if (descriptor < 0) {
            fprintf(stderr, "[head-worker] accessory open: %s\n",
                    strerror(errno));
            sleep(1);
            continue;
        }
        fprintf(stderr, "[head-worker] endpoint open\n");
        fflush(stderr);

        for (;;) {
            if (!read_exact(
                        descriptor, request_bytes.data(),
                        request_bytes.size())) {
                break;
            }
            const double request_started = now_us();
            s41_head_request request = {};
            memcpy(&request, request_bytes.data(), sizeof(request));
            if (request.magic != S41_HEAD_REQUEST_MAGIC ||
                request.version != S41_HEAD_PROTOCOL_VERSION ||
                request.flags != request_flags ||
                request.type != (uint32_t) weight_type ||
                request.k != (uint32_t) k ||
                request.vocab != (uint32_t) vocab ||
                request.offset != (uint32_t) offset ||
                request.count != (uint32_t) count ||
                request.input_bytes != input_bytes ||
                request.weight_hash != weight_hash) {
                fprintf(stderr, "[head-worker] invalid request\n");
                break;
            }

            memcpy(
                    input_data.data(),
                    request_bytes.data() + sizeof(request), input_bytes);
            if (s41_fast_hash_bytes(input_data.data(), input_bytes) !=
                request.input_hash) {
                fprintf(stderr, "[head-worker] input hash mismatch\n");
                break;
            }

            const void * tensor_input = input_data.data();
            if (f16_input) {
                ggml_fp16_to_fp32_row(
                        (const ggml_fp16_t *) input_data.data(),
                        input_f32.data(), (size_t) k);
                tensor_input = input_f32.data();
            } else if (i8_input) {
                float scale = 0.0f;
                memcpy(&scale, input_data.data(), sizeof(scale));
                if (!std::isfinite(scale) || scale <= 0.0f) {
                    fprintf(stderr, "[head-worker] invalid input scale\n");
                    break;
                }
                const int8_t * values = (const int8_t *) (
                        input_data.data() + sizeof(scale));
                for (size_t index = 0; index < (size_t) k; ++index) {
                    input_f32[index] = scale * (float) values[index];
                }
                tensor_input = input_f32.data();
            }

            double started = now_us();
            ggml_backend_tensor_set(
                    input, tensor_input, 0, sizeof(float) * (size_t) k);
            const uint32_t set_us = elapsed_us(started);

            started = now_us();
            const enum ggml_status status =
                    ggml_backend_graph_compute(backend, graph);
            const uint32_t compute_us = elapsed_us(started);
            compute_times.push_back((double) compute_us);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[head-worker] compute failed: %d\n",
                        (int) status);
                break;
            }

            started = now_us();
            ggml_backend_tensor_get(
                    logits, output_data.data(), 0,
                    output_data.size() * sizeof(float));
            const uint32_t get_us = elapsed_us(started);
            get_times.push_back((double) get_us);

            started = now_us();
            std::array<s41_head_candidate, S41_HEAD_TOP_K> candidates;
            for (size_t index = 0; index < candidates.size(); ++index) {
                candidates[index] = {UINT32_MAX, -INFINITY};
            }
            for (size_t index = 0; index < output_data.size(); ++index) {
                s41_head_candidate candidate = {
                    (uint32_t) offset + (uint32_t) index,
                    output_data[index],
                };
                size_t position = candidates.size();
                while (position > 0 &&
                       candidate_better(
                               candidate, candidates[position - 1])) {
                    --position;
                }
                if (position == candidates.size()) {
                    continue;
                }
                for (size_t move = candidates.size() - 1;
                     move > position; --move) {
                    candidates[move] = candidates[move - 1];
                }
                candidates[position] = candidate;
            }
            const uint32_t reduce_us = elapsed_us(started);
            reduce_times.push_back((double) reduce_us);

            s41_head_response response = {};
            response.magic = S41_HEAD_RESPONSE_MAGIC;
            response.version = S41_HEAD_PROTOCOL_VERSION;
            response.request_id = request.request_id;
            response.candidate_count = S41_HEAD_TOP_K;
            response.output_bytes = sizeof(candidates);
            response.output_hash =
                    s41_fast_hash_bytes(
                            candidates.data(), sizeof(candidates));
            response.weight_hash = weight_hash;
            response.set_us = set_us;
            response.compute_us = compute_us;
            response.get_us = get_us;
            response.reduce_us = reduce_us;
            memcpy(response_bytes.data(), &response, sizeof(response));
            memcpy(
                    response_bytes.data() + sizeof(response),
                    candidates.data(), sizeof(candidates));
            if (!write_exact(
                        descriptor,
                        response_bytes.data(), response_bytes.size())) {
                break;
            }
            request_times.push_back(now_us() - request_started);

            if (request_times.size() % 100 == 0) {
                fprintf(stderr,
                        "[head-worker] requests=%zu compute_us=%.0f "
                        "get_us=%.0f reduce_us=%.0f request_us=%.0f\n",
                        request_times.size(), median(compute_times),
                        median(get_times), median(reduce_times),
                        median(request_times));
                fflush(stderr);
            }
        }
        close(descriptor);
    }
}
