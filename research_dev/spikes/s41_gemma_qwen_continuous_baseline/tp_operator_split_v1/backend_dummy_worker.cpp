// Phone-side backend latency probe with fixed FP16 activation boundaries.
//
// usage:
//   backend_dummy_worker <elements> <backend> <noop|sqr> <repeats> <requests>

#include "backend_dummy_protocol.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cerrno>
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

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr,
                "usage: %s <elements> <backend> <noop|sqr> "
                "<repeats> <requests>\n",
                argv[0]);
        return 2;
    }

    const int64_t elements = atoll(argv[1]);
    const std::string backend_name = argv[2];
    const std::string op_name = argv[3];
    const int repeats = atoi(argv[4]);
    const int max_requests = atoi(argv[5]);
    if (elements <= 0 || elements > (1 << 24) ||
        (op_name != "noop" && op_name != "sqr") ||
        (op_name == "noop" && repeats != 0) ||
        (op_name == "sqr" && (repeats <= 0 || repeats > 256)) ||
        max_requests <= 0) {
        fprintf(stderr, "[dummy-worker] invalid configuration\n");
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
        fprintf(stderr, "[dummy-worker] backend '%s' not found\n",
                backend_name.c_str());
        return 1;
    }

    ggml_init_params parameters = {};
    parameters.mem_size = ggml_tensor_overhead() *
            static_cast<size_t>(repeats + 4) + ggml_graph_overhead();
    parameters.no_alloc = true;
    ggml_context * context = ggml_init(parameters);
    if (context == nullptr) {
        return 1;
    }

    ggml_tensor * input = ggml_new_tensor_1d(
            context, GGML_TYPE_F32, elements);
    ggml_set_input(input);
    ggml_tensor * result = input;
    for (int index = 0; index < repeats; ++index) {
        result = ggml_sqr(context, result);
    }
    ggml_set_output(result);

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        fprintf(stderr, "[dummy-worker] backend allocation failed\n");
        return 1;
    }
    ggml_backend_buffer_set_usage(
            buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, result);

    const size_t encoded_bytes =
            static_cast<size_t>(elements) * sizeof(ggml_fp16_t);
    const size_t decoded_bytes =
            static_cast<size_t>(elements) * sizeof(float);
    std::vector<uint8_t> request_bytes(
            sizeof(s41_dummy_request) + encoded_bytes);
    std::vector<uint8_t> response_bytes(
            sizeof(s41_dummy_response) + encoded_bytes);
    std::vector<float> input_f32(static_cast<size_t>(elements));
    std::vector<float> output_f32(static_cast<size_t>(elements));
    std::vector<ggml_fp16_t> output_f16(static_cast<size_t>(elements));

    fprintf(stderr,
            "[dummy-worker] ready backend=%s description=%s elements=%lld "
            "op=%s repeats=%d nodes=%d requests=%d request_bytes=%zu "
            "response_bytes=%zu\n",
            selected_name, selected_description,
            static_cast<long long>(elements), op_name.c_str(), repeats,
            ggml_graph_n_nodes(graph), max_requests, request_bytes.size(),
            response_bytes.size());
    fflush(stderr);

    uint64_t previous_write_ns = 0;
    int served = 0;
    for (;;) {
        const int descriptor = open("/dev/usb_accessory", O_RDWR);
        if (descriptor < 0) {
            fprintf(stderr, "[dummy-worker] endpoint open: %s\n",
                    strerror(errno));
            sleep(1);
            continue;
        }
        fprintf(stderr, "[dummy-worker] endpoint open\n");
        fflush(stderr);

        while (served < max_requests && read_exact(
                    descriptor, request_bytes.data(), request_bytes.size())) {
            const uint64_t request_started = now_ns();
            uint64_t started = request_started;
            s41_dummy_request request = {};
            memcpy(&request, request_bytes.data(), sizeof(request));
            const uint8_t * encoded_input =
                    request_bytes.data() + sizeof(request);
            const bool request_ok =
                    request.magic == S41_DUMMY_REQUEST_MAGIC &&
                    request.version == S41_DUMMY_PROTOCOL_VERSION &&
                    request.reserved == 0 &&
                    request.elements == static_cast<uint32_t>(elements) &&
                    request.input_bytes == encoded_bytes &&
                    request.input_crc32 ==
                            crc32_bytes(encoded_input, encoded_bytes);
            const uint64_t validate_ns = now_ns() - started;
            if (!request_ok) {
                fprintf(stderr, "[dummy-worker] invalid request\n");
                close(descriptor);
                return 3;
            }

            started = now_ns();
            ggml_fp16_to_fp32_row(
                    reinterpret_cast<const ggml_fp16_t *>(encoded_input),
                    input_f32.data(), elements);
            const uint64_t decode_ns = now_ns() - started;

            started = now_ns();
            ggml_backend_tensor_set(
                    input, input_f32.data(), 0, decoded_bytes);
            const uint64_t set_ns = now_ns() - started;

            started = now_ns();
            const enum ggml_status status =
                    ggml_backend_graph_compute_async(backend, graph);
            const uint64_t submit_ns = now_ns() - started;
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[dummy-worker] compute failed: %d\n",
                        static_cast<int>(status));
                close(descriptor);
                return 3;
            }

            started = now_ns();
            ggml_backend_synchronize(backend);
            const uint64_t sync_ns = now_ns() - started;

            started = now_ns();
            ggml_backend_tensor_get(
                    result, output_f32.data(), 0, decoded_bytes);
            const uint64_t get_ns = now_ns() - started;

            started = now_ns();
            ggml_fp32_to_fp16_row(
                    output_f32.data(), output_f16.data(), elements);
            const uint32_t output_crc =
                    crc32_bytes(output_f16.data(), encoded_bytes);
            const uint64_t encode_hash_ns = now_ns() - started;
            const uint64_t prewrite_ns = now_ns() - request_started;

            s41_dummy_response response = {};
            response.magic = S41_DUMMY_RESPONSE_MAGIC;
            response.version = S41_DUMMY_PROTOCOL_VERSION;
            response.request_id = request.request_id;
            response.elements = static_cast<uint32_t>(elements);
            response.output_bytes = static_cast<uint32_t>(encoded_bytes);
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
                    output_f16.data(), encoded_bytes);

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

    fprintf(stderr, "[dummy-worker] complete requests=%d\n", served);
    fflush(stderr);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    return 0;
}
