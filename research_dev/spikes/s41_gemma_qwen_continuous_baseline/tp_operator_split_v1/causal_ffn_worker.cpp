#include "causal_ffn_protocol.h"
#include "causal_quantized_weights.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

static bool read_exact(int fd, void * dst, size_t size) {
    uint8_t * ptr = (uint8_t *) dst;
    while (size > 0) {
        const ssize_t n = read(fd, ptr, size);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        ptr += n;
        size -= (size_t) n;
    }
    return true;
}

static bool write_exact(int fd, const void * src, size_t size) {
    const uint8_t * ptr = (const uint8_t *) src;
    while (size > 0) {
        const ssize_t n = write(fd, ptr, size);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        ptr += n;
        size -= (size_t) n;
    }
    return true;
}

static double now_us() {
    timespec ts = {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e6 + (double) ts.tv_nsec / 1e3;
}

static bool parse_type(const char * name, ggml_type & result) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const ggml_type type = (ggml_type) i;
        const char * type_name = ggml_type_name(type);
        if (type_name != nullptr && name == std::string(type_name)) {
            result = type;
            return true;
        }
    }
    return false;
}

static bool initialize_weight(
        ggml_tensor * tensor,
        ggml_type type,
        uint32_t matrix,
        uint32_t row_offset,
        uint32_t column_offset,
        uint64_t * weight_hash = nullptr) {
    const int64_t columns = tensor->ne[0];
    const int64_t rows = tensor->ne[1];
    std::vector<uint8_t> quantized(ggml_nbytes(tensor));
    if (!s41_fill_quantized_weight(
                type, matrix, row_offset, column_offset,
                (uint32_t) columns, (uint32_t) rows,
                quantized.data(), quantized.size())) {
        fprintf(stderr, "[causal-worker] deterministic weight fill failed\n");
        return false;
    }
    if (weight_hash != nullptr) {
        *weight_hash = s41_weight_hash_update(
                *weight_hash, matrix, row_offset, column_offset,
                (uint32_t) columns, (uint32_t) rows,
                quantized.data(), quantized.size());
    }
    ggml_backend_tensor_set(tensor, quantized.data(), 0, quantized.size());
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 8) {
        fprintf(stderr,
                "usage: %s <port> <K> <NFF> <offset> <count> <backend> <type> [max_requests]\n",
                argv[0]);
        return 2;
    }

    const int port = atoi(argv[1]);
    const int64_t k = atoll(argv[2]);
    const int64_t n_ff = atoll(argv[3]);
    const int64_t offset = atoll(argv[4]);
    const int64_t count = atoll(argv[5]);
    const std::string backend_name = argv[6];
    const long max_requests = argc > 8 ? atol(argv[8]) : 0;
    const bool aoa = getenv("S41_FFN_AOA") != nullptr;

    ggml_type weight_type;
    if (!parse_type(argv[7], weight_type)) {
        fprintf(stderr, "[causal-worker] unknown weight type '%s'\n", argv[7]);
        return 2;
    }
    if (weight_type != GGML_TYPE_Q4_0 && weight_type != GGML_TYPE_Q8_0) {
        fprintf(stderr, "[causal-worker] only q4_0 and q8_0 are supported\n");
        return 2;
    }

    const int64_t block = ggml_blck_size(weight_type);
    if (k <= 0 || n_ff <= 0 || offset < 0 || count <= 0 || offset + count != n_ff ||
        k % block != 0 || n_ff % block != 0 || offset % block != 0 || count % block != 0) {
        fprintf(stderr,
                "[causal-worker] invalid dimensions K=%lld NFF=%lld offset=%lld count=%lld block=%lld\n",
                (long long) k, (long long) n_ff, (long long) offset,
                (long long) count, (long long) block);
        return 2;
    }

    ggml_backend_t backend = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (std::string(ggml_backend_dev_name(device)).find(backend_name) != std::string::npos) {
            backend = ggml_backend_dev_init(device, nullptr);
            fprintf(stderr, "[causal-worker] backend=%s description=%s\n",
                    ggml_backend_dev_name(device), ggml_backend_dev_description(device));
            break;
        }
    }
    if (backend == nullptr) {
        fprintf(stderr, "[causal-worker] backend '%s' not found\n", backend_name.c_str());
        return 1;
    }

    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return 1;
    }

    ggml_tensor * gate = ggml_new_tensor_2d(ctx, weight_type, k, count);
    ggml_tensor * up = ggml_new_tensor_2d(ctx, weight_type, k, count);
    ggml_tensor * down = ggml_new_tensor_2d(ctx, weight_type, count, k);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);

    ggml_tensor * gate_out = ggml_mul_mat(ctx, gate, input);
    ggml_tensor * up_out = ggml_mul_mat(ctx, up, input);
    ggml_tensor * silu = ggml_silu(ctx, gate_out);
    ggml_tensor * activation = ggml_mul(ctx, silu, up_out);
    ggml_tensor * output = ggml_mul_mat(ctx, down, activation);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        fprintf(stderr, "[causal-worker] backend allocation failed\n");
        return 1;
    }

    uint64_t weight_hash = S41_HASH64_OFFSET;
    if (!initialize_weight(
                gate, weight_type, S41_WEIGHT_GATE, (uint32_t) offset, 0,
                &weight_hash) ||
        !initialize_weight(
                up, weight_type, S41_WEIGHT_UP, (uint32_t) offset, 0,
                &weight_hash) ||
        !initialize_weight(
                down, weight_type, S41_WEIGHT_DOWN, 0, (uint32_t) offset,
                &weight_hash)) {
        return 1;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    fprintf(stderr,
            "[causal-worker] ready K=%lld NFF=%lld slice=[%lld,%lld) type=%s nodes=%d mode=%s\n",
            (long long) k, (long long) n_ff, (long long) offset,
            (long long) (offset + count), ggml_type_name(weight_type),
            ggml_graph_n_nodes(graph), aoa ? "aoa" : "socket");
    fprintf(stderr, "[causal-worker] weight_hash=%016llx\n",
            (unsigned long long) weight_hash);
    fflush(stderr);

    int listen_fd = -1;
    if (!aoa) {
        listen_fd = socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        if (listen_fd < 0 ||
            setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0) {
            perror("[causal-worker] socket");
            return 1;
        }
        sockaddr_in address = {};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port);
        if (bind(listen_fd, (sockaddr *) &address, sizeof(address)) != 0 ||
            listen(listen_fd, 4) != 0) {
            perror("[causal-worker] bind/listen");
            return 1;
        }
    }

    std::vector<float> input_data((size_t) k);
    std::vector<float> output_data((size_t) k);
    std::vector<uint8_t> request_data(
            sizeof(s41_ffn_request) + sizeof(float) * (size_t) k);
    std::vector<uint8_t> response(sizeof(s41_ffn_response) + sizeof(float) * (size_t) k);
    long served = 0;
    std::vector<double> samples;

    for (;;) {
        int fd = -1;
        if (aoa) {
            fd = open("/dev/usb_accessory", O_RDWR);
            if (fd < 0) {
                fprintf(stderr, "[causal-worker] accessory open: %s\n", strerror(errno));
                sleep(1);
                continue;
            }
        } else {
            fd = accept(listen_fd, nullptr, nullptr);
            if (fd >= 0) {
                int one = 1;
                setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            }
        }
        if (fd < 0) {
            continue;
        }
        fprintf(stderr, "[causal-worker] endpoint open\n");
        fflush(stderr);

        for (;;) {
            s41_ffn_request request = {};
            // The Android accessory driver posts a USB request sized to read().
            // Read the full wire packet at once so a host bulk transfer larger
            // than the header cannot overflow a short gadget-side request.
            if (!read_exact(fd, request_data.data(), request_data.size())) {
                break;
            }
            memcpy(&request, request_data.data(), sizeof(request));
            const bool valid =
                    request.magic == S41_FFN_REQUEST_MAGIC &&
                    request.version == S41_FFN_PROTOCOL_VERSION &&
                    request.type == (uint32_t) weight_type &&
                    request.k == (uint32_t) k &&
                    request.n_ff == (uint32_t) n_ff &&
                    request.offset == (uint32_t) offset &&
                    request.count == (uint32_t) count &&
                    request.input_bytes == sizeof(float) * (uint32_t) k &&
                    request.weight_hash == weight_hash;
            if (!valid) {
                fprintf(stderr, "[causal-worker] invalid request\n");
                break;
            }
            memcpy(input_data.data(), request_data.data() + sizeof(request),
                   sizeof(float) * (size_t) k);
            if (request.input_hash !=
                s41_hash_bytes(input_data.data(), sizeof(float) * (size_t) k)) {
                fprintf(stderr, "[causal-worker] input hash mismatch\n");
                break;
            }

            const double started = now_us();
            ggml_backend_tensor_set(input, input_data.data(), 0, sizeof(float) * (size_t) k);
            const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[causal-worker] graph compute failed: %d\n", (int) status);
                break;
            }
            ggml_backend_tensor_get(output, output_data.data(), 0, sizeof(float) * (size_t) k);
            samples.push_back(now_us() - started);

            s41_ffn_response header = {};
            header.magic = S41_FFN_RESPONSE_MAGIC;
            header.version = S41_FFN_PROTOCOL_VERSION;
            header.status = 0;
            header.request_id = request.request_id;
            header.k = (uint32_t) k;
            header.output_bytes = sizeof(float) * (uint32_t) k;
            header.output_hash =
                    s41_hash_bytes(output_data.data(), sizeof(float) * (size_t) k);
            header.weight_hash = weight_hash;
            memcpy(response.data(), &header, sizeof(header));
            memcpy(response.data() + sizeof(header), output_data.data(),
                   sizeof(float) * (size_t) k);
            if (!write_exact(fd, response.data(), response.size())) {
                break;
            }

            ++served;
            if (max_requests > 0 && served >= max_requests) {
                close(fd);
                ggml_backend_buffer_free(buffer);
                ggml_free(ctx);
                ggml_backend_free(backend);
                return 0;
            }
            if (samples.size() % 50 == 0) {
                std::vector<double> sorted = samples;
                std::sort(sorted.begin(), sorted.end());
                fprintf(stderr, "[causal-worker] requests=%zu device_median_us=%.0f\n",
                        samples.size(), sorted[sorted.size() / 2]);
                fflush(stderr);
            }
        }
        close(fd);
    }
}
