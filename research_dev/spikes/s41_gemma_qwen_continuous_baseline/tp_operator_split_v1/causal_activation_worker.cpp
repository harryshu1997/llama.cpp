// Phone-side FFN-slice worker for the activation and residual diagnostics.
//
// The host sends the causal FFN input. HTP computes gate, up, SiLU, and
// multiply for one suffix slice. It returns either the slice activation or
// the completed down-projection residual.
//
// usage:
//   causal_activation_worker <K> <NFF> <offset> <count> <backend> <type>
//       [input_type] [output_type] [batch] [result]

#include "causal_ffn_protocol.h"
#include "causal_quantized_weights.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
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

static int open_tcp_listener(int port) {
    int descriptor = socket(AF_INET, SOCK_STREAM, 0);
    if (descriptor < 0) {
        return -1;
    }
    int one = 1;
    setsockopt(descriptor, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons((uint16_t) port);
    if (bind(descriptor, (sockaddr *) &address, sizeof(address)) != 0 ||
        listen(descriptor, 1) != 0) {
        close(descriptor);
        return -1;
    }
    return descriptor;
}

static bool initialize_gate_up(
        ggml_tensor * tensor,
        ggml_type type,
        uint32_t row_offset,
        uint32_t count,
        uint64_t & weight_hash) {
    const uint32_t columns = (uint32_t) tensor->ne[0];
    const uint32_t rows = (uint32_t) tensor->ne[1];
    if (rows != 2 * count) {
        return false;
    }
    const size_t matrix_bytes = ggml_row_size(type, columns) * count;
    std::vector<uint8_t> quantized(ggml_nbytes(tensor));
    if (!s41_fill_quantized_weight(
                type, S41_WEIGHT_GATE, row_offset, 0, columns, count,
                quantized.data(), matrix_bytes) ||
        !s41_fill_quantized_weight(
                type, S41_WEIGHT_UP, row_offset, 0, columns, count,
                quantized.data() + matrix_bytes, matrix_bytes)) {
        return false;
    }
    weight_hash = s41_weight_hash_update(
            weight_hash, S41_WEIGHT_GATE, row_offset, 0, columns, count,
            quantized.data(), matrix_bytes);
    weight_hash = s41_weight_hash_update(
            weight_hash, S41_WEIGHT_UP, row_offset, 0, columns, count,
            quantized.data() + matrix_bytes, matrix_bytes);
    ggml_backend_tensor_set(tensor, quantized.data(), 0, quantized.size());
    return true;
}

static bool initialize_weight(
        ggml_tensor * tensor,
        ggml_type type,
        uint32_t matrix,
        uint32_t row_offset,
        uint32_t column_offset,
        uint64_t & weight_hash) {
    const uint32_t columns = (uint32_t) tensor->ne[0];
    const uint32_t rows = (uint32_t) tensor->ne[1];
    std::vector<uint8_t> quantized(ggml_nbytes(tensor));
    if (!s41_fill_quantized_weight(
                type, matrix, row_offset, column_offset, columns, rows,
                quantized.data(), quantized.size())) {
        return false;
    }
    weight_hash = s41_weight_hash_update(
            weight_hash, matrix, row_offset, column_offset, columns, rows,
            quantized.data(), quantized.size());
    ggml_backend_tensor_set(tensor, quantized.data(), 0, quantized.size());
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 7 || argc > 11) {
        fprintf(stderr,
                "usage: %s <K> <NFF> <offset> <count> <backend> <type> "
                "[input_type] [output_type] [batch] [result]\n",
                argv[0]);
        return 2;
    }

    const int64_t k = atoll(argv[1]);
    const int64_t n_ff = atoll(argv[2]);
    const int64_t offset = atoll(argv[3]);
    const int64_t count = atoll(argv[4]);
    const std::string backend_name = argv[5];
    ggml_type weight_type = GGML_TYPE_COUNT;
    if (!parse_type(argv[6], weight_type) ||
        (weight_type != GGML_TYPE_Q4_0 && weight_type != GGML_TYPE_Q8_0)) {
        fprintf(stderr, "[activation-worker] unsupported type '%s'\n", argv[6]);
        return 2;
    }
    ggml_type input_type = GGML_TYPE_F32;
    bool i8_input = false;
    if (argc >= 8) {
        if (std::string(argv[7]) == "f16") {
            input_type = GGML_TYPE_F16;
        } else if (std::string(argv[7]) == "i8") {
            i8_input = true;
        } else if (std::string(argv[7]) != "f32") {
            fprintf(stderr,
                    "[activation-worker] unsupported input type '%s'\n",
                    argv[7]);
            return 2;
        }
    }
    const std::string output_name = argc > 8 ? argv[8] : "f32";
    const int64_t batch = argc > 9 ? atoll(argv[9]) : 1;
    if (output_name != "f32" && output_name != "f16") {
        fprintf(stderr,
                "[activation-worker] unsupported output type '%s'\n",
                output_name.c_str());
        return 2;
    }
    const bool f16_output = output_name == "f16";
    const std::string result_name = argc > 10 ? argv[10] : "activation";
    if (result_name != "activation" && result_name != "residual") {
        fprintf(stderr,
                "[activation-worker] unsupported result '%s'\n",
                result_name.c_str());
        return 2;
    }
    const bool residual_output = result_name == "residual";
    const uint16_t base_request_flags = S41_FFN_FLAG_FAST_HASH |
            (input_type == GGML_TYPE_F16 ? S41_FFN_FLAG_F16_INPUT : 0) |
            (i8_input ? S41_FFN_FLAG_I8_INPUT : 0) |
            (f16_output ? S41_FFN_FLAG_F16_OUTPUT : 0) |
            (residual_output ? S41_FFN_FLAG_RESIDUAL_OUTPUT : 0);
    const uint16_t request_flags = s41_ffn_encode_flags(
            base_request_flags, (uint16_t) batch);
    const size_t input_elements = (size_t) k * (size_t) batch;
    const size_t output_elements = (size_t) (
            residual_output ? k : count) * (size_t) batch;
    const size_t input_bytes = i8_input
            ? sizeof(float) + input_elements
            : ggml_type_size(input_type) * input_elements;
    const char * input_name =
            i8_input ? "i8" : ggml_type_name(input_type);

    const int64_t block = ggml_blck_size(weight_type);
    if (k <= 0 || n_ff <= 0 || offset < 0 || count <= 0 ||
        batch <= 0 || batch > S41_FFN_MAX_BATCH ||
        offset + count > n_ff || k % block != 0 ||
        offset % block != 0 || count % block != 0) {
        fprintf(stderr, "[activation-worker] invalid dimensions\n");
        return 2;
    }

    ggml_backend_t backend = nullptr;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (std::string(ggml_backend_dev_name(device)).find(backend_name) !=
            std::string::npos) {
            backend = ggml_backend_dev_init(device, nullptr);
            fprintf(stderr, "[activation-worker] backend=%s description=%s\n",
                    ggml_backend_dev_name(device),
                    ggml_backend_dev_description(device));
            break;
        }
    }
    if (backend == nullptr) {
        fprintf(stderr,
                "[activation-worker] backend '%s' not found; devices:",
                backend_name.c_str());
        for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
            fprintf(stderr, " %s", ggml_backend_dev_name(
                    ggml_backend_dev_get(index)));
        }
        fprintf(stderr, "\n");
        return 1;
    }

    ggml_init_params parameters = {};
    parameters.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
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
            ggml_new_tensor_2d(input_context, GGML_TYPE_F32, k, batch);
    ggml_set_input(input);
    ggml_tensor * gate_up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate_output = nullptr;
    ggml_tensor * up_output = nullptr;
    if (batch == 1) {
        gate_up =
                ggml_new_tensor_2d(context, weight_type, k, 2 * count);
        ggml_tensor * gate_up_output =
                ggml_mul_mat(context, gate_up, input);
        gate_output =
                ggml_view_1d(context, gate_up_output, count, 0);
        up_output = ggml_view_1d(
                context, gate_up_output, count,
                sizeof(float) * (size_t) count);
    } else {
        gate = ggml_new_tensor_2d(context, weight_type, k, count);
        up = ggml_new_tensor_2d(context, weight_type, k, count);
        gate_output = ggml_mul_mat(context, gate, input);
        up_output = ggml_mul_mat(context, up, input);
    }
    ggml_tensor * activation =
            ggml_mul(context, ggml_silu(context, gate_output), up_output);
    ggml_tensor * down = nullptr;
    ggml_tensor * result = activation;
    if (residual_output) {
        down = ggml_new_tensor_2d(context, weight_type, count, k);
        result = ggml_mul_mat(context, down, activation);
    }

    ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        fprintf(stderr, "[activation-worker] allocation failed\n");
        return 1;
    }
    ggml_backend_buffer_t input_buffer =
            ggml_backend_alloc_ctx_tensors(input_context, backend);
    if (input_buffer == nullptr) {
        fprintf(stderr, "[activation-worker] input allocation failed\n");
        return 1;
    }
    ggml_backend_buffer_set_usage(
            input_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    uint64_t weight_hash = S41_HASH64_OFFSET;
    const bool weights_ok = batch == 1
            ? initialize_gate_up(
                    gate_up, weight_type, (uint32_t) offset,
                    (uint32_t) count, weight_hash)
            : initialize_weight(
                    gate, weight_type, S41_WEIGHT_GATE,
                    (uint32_t) offset, 0, weight_hash) &&
                    initialize_weight(
                            up, weight_type, S41_WEIGHT_UP,
                            (uint32_t) offset, 0,
                            weight_hash);
    const bool down_ok = !residual_output ||
            initialize_weight(
                    down, weight_type, S41_WEIGHT_DOWN,
                    0, (uint32_t) offset, weight_hash);
    if (!weights_ok || !down_ok) {
        fprintf(stderr, "[activation-worker] weight initialization failed\n");
        return 1;
    }

    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, result);
    if (getenv("S41_DISABLE_GRAPH_CACHE") != nullptr) {
        graph->uid = 0;
    }
    fprintf(stderr,
            "[activation-worker] ready K=%lld NFF=%lld slice=[%lld,%lld) "
            "batch=%lld type=%s input_type=%s output_type=%s "
            "result=%s weight_hash=%016llx nodes=%d\n",
            (long long) k, (long long) n_ff, (long long) offset,
            (long long) (offset + count), (long long) batch,
            ggml_type_name(weight_type),
            input_name,
            output_name.c_str(),
            result_name.c_str(),
            (unsigned long long) weight_hash, ggml_graph_n_nodes(graph));
    if (graph->uid == 0) {
        fprintf(stderr, "[activation-worker] graph cache disabled\n");
    }
    fflush(stderr);

    std::vector<uint8_t> request_bytes(
            sizeof(s41_ffn_request) + input_bytes);
    std::vector<uint8_t> response_bytes(
            sizeof(s41_ffn_response) +
                    (f16_output
                            ? sizeof(ggml_fp16_t)
                            : sizeof(float)) * output_elements);
    std::vector<uint8_t> input_data(input_bytes);
    std::vector<float> input_f32(
            input_type == GGML_TYPE_F16 || i8_input ? input_elements : 0);
    std::vector<float> output_data(output_elements);
    std::vector<ggml_fp16_t> output_f16(
            f16_output ? output_elements : 0);
    std::vector<double> input_hash_times;
    std::vector<double> convert_times;
    std::vector<double> set_times;
    std::vector<double> compute_times;
    std::vector<double> get_times;
    std::vector<double> output_hash_times;
    std::vector<double> write_times;
    std::vector<double> request_times;

    const char * tcp_port_text = getenv("S41_TCP_PORT");
    int listener = -1;
    if (tcp_port_text != nullptr) {
        const long tcp_port = atol(tcp_port_text);
        if (tcp_port <= 0 || tcp_port > 65535) {
            fprintf(stderr, "[activation-worker] invalid TCP port\n");
            return 2;
        }
        listener = open_tcp_listener((int) tcp_port);
        if (listener < 0) {
            fprintf(stderr, "[activation-worker] TCP listen failed: %s\n",
                    strerror(errno));
            return 1;
        }
        fprintf(stderr, "[activation-worker] TCP port=%ld\n", tcp_port);
        fflush(stderr);
    }

    for (;;) {
        const int descriptor = listener >= 0
                ? accept(listener, nullptr, nullptr)
                : open("/dev/usb_accessory", O_RDWR);
        if (descriptor < 0) {
            fprintf(stderr, "[activation-worker] endpoint open: %s\n",
                    strerror(errno));
            sleep(1);
            continue;
        }
        if (listener >= 0) {
            int one = 1;
            setsockopt(
                    descriptor, IPPROTO_TCP, TCP_NODELAY,
                    &one, sizeof(one));
        }
        fprintf(stderr, "[activation-worker] endpoint open\n");
        fflush(stderr);

        for (;;) {
            if (!read_exact(
                        descriptor, request_bytes.data(), request_bytes.size())) {
                break;
            }
            const double request_started = now_us();
            s41_ffn_request request = {};
            memcpy(&request, request_bytes.data(), sizeof(request));
            if (request.magic != S41_FFN_REQUEST_MAGIC ||
                request.version != S41_FFN_PROTOCOL_VERSION ||
                request.reserved != request_flags ||
                s41_ffn_decode_batch(request.reserved) != batch ||
                request.type != (uint32_t) weight_type ||
                request.k != (uint32_t) k ||
                request.n_ff != (uint32_t) n_ff ||
                request.offset != (uint32_t) offset ||
                request.count != (uint32_t) count ||
                request.input_bytes != input_bytes ||
                request.weight_hash != weight_hash) {
                fprintf(stderr, "[activation-worker] invalid request\n");
                break;
            }

            memcpy(
                    input_data.data(),
                    request_bytes.data() + sizeof(request),
                    input_bytes);
            double started = now_us();
            const uint32_t input_hash =
                    s41_fast_hash_bytes(input_data.data(), input_bytes);
            input_hash_times.push_back(now_us() - started);
            if (input_hash != request.input_hash) {
                fprintf(stderr, "[activation-worker] input hash mismatch\n");
                break;
            }

            started = now_us();
            const void * tensor_input = input_data.data();
            if (input_type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row(
                        (const ggml_fp16_t *) input_data.data(),
                        input_f32.data(), input_elements);
                tensor_input = input_f32.data();
            } else if (i8_input) {
                float scale = 0.0f;
                memcpy(&scale, input_data.data(), sizeof(scale));
                if (!std::isfinite(scale) || scale <= 0.0f) {
                    fprintf(stderr, "[activation-worker] invalid input scale\n");
                    break;
                }
                const int8_t * values = (const int8_t *) (
                        input_data.data() + sizeof(scale));
                for (size_t index = 0; index < input_elements; ++index) {
                    input_f32[index] =
                            scale * (float) values[index];
                }
                tensor_input = input_f32.data();
            }
            convert_times.push_back(now_us() - started);

            started = now_us();
            ggml_backend_tensor_set(
                    input, tensor_input, 0,
                    sizeof(float) * input_elements);
            set_times.push_back(now_us() - started);

            started = now_us();
            const enum ggml_status status =
                    ggml_backend_graph_compute(backend, graph);
            compute_times.push_back(now_us() - started);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "[activation-worker] compute failed: %d\n",
                        (int) status);
                break;
            }

            started = now_us();
            ggml_backend_tensor_get(
                    result, output_data.data(), 0,
                    sizeof(float) * output_elements);
            get_times.push_back(now_us() - started);

            started = now_us();
            const void * response_output = output_data.data();
            size_t output_bytes = sizeof(float) * output_elements;
            if (f16_output) {
                ggml_fp32_to_fp16_row(
                        output_data.data(), output_f16.data(),
                        output_elements);
                response_output = output_f16.data();
                output_bytes =
                        sizeof(ggml_fp16_t) * output_elements;
            }
            const uint32_t output_hash =
                    s41_fast_hash_bytes(
                            response_output, output_bytes);
            output_hash_times.push_back(now_us() - started);

            s41_ffn_response response = {};
            response.magic = S41_FFN_RESPONSE_MAGIC;
            response.version = S41_FFN_PROTOCOL_VERSION;
            response.request_id = request.request_id;
            response.k = (uint32_t) output_elements;
            response.output_bytes = (uint32_t) output_bytes;
            response.output_hash = output_hash;
            response.weight_hash = weight_hash;
            memcpy(response_bytes.data(), &response, sizeof(response));
            memcpy(
                    response_bytes.data() + sizeof(response),
                    response_output, output_bytes);
            started = now_us();
            const bool write_ok = write_exact(
                    descriptor, response_bytes.data(), response_bytes.size());
            write_times.push_back(now_us() - started);
            request_times.push_back(now_us() - request_started);
            if (!write_ok) {
                break;
            }

            const size_t requests = compute_times.size();
            if (requests % 500 == 0) {
                fprintf(stderr,
                        "[activation-worker] requests=%zu hash_in_us=%.0f "
                        "convert_us=%.0f set_us=%.0f compute_us=%.0f "
                        "get_us=%.0f "
                        "hash_out_us=%.0f write_us=%.0f request_us=%.0f\n",
                        requests, median(input_hash_times),
                        median(convert_times), median(set_times),
                        median(compute_times), median(get_times),
                        median(output_hash_times), median(write_times),
                        median(request_times));
                fflush(stderr);
            }
        }
        close(descriptor);
    }
}
