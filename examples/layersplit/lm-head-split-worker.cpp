#include "lm-head-split-protocol.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "gguf.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <numeric>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

using steady_clock = std::chrono::steady_clock;

struct config {
    std::string model;
    std::string backend = "CPU";
    std::string bind = "127.0.0.1";
    int port = 0;
    int64_t rows = 0;
    int64_t top_k = 0;
    bool f16_io = false;
    long max_requests = 0;
};

struct host_tensor {
    std::vector<uint8_t> bytes;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    ggml_type type = GGML_TYPE_COUNT;
};

struct gguf_reader {
    int fd = -1;
    ggml_context * metadata = nullptr;
    gguf_context * gguf = nullptr;

    ~gguf_reader() {
        if (gguf != nullptr) {
            gguf_free(gguf);
        }
        if (metadata != nullptr) {
            ggml_free(metadata);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
};

uint64_t now_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            steady_clock::now().time_since_epoch()).count());
}

bool parse_i64(const char * text, int64_t & value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const long long parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    value = parsed;
    return true;
}

bool parse_config(int argc, char ** argv, config & result) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            result.model = argv[++i];
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            result.backend = argv[++i];
        } else if (strcmp(argv[i], "--bind") == 0 && i + 1 < argc) {
            result.bind = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            int64_t value = 0;
            if (!parse_i64(argv[++i], value) || value <= 0 || value > 65535) {
                return false;
            }
            result.port = static_cast<int>(value);
        } else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            if (!parse_i64(argv[++i], result.rows) || result.rows <= 0) {
                return false;
            }
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!parse_i64(argv[++i], result.top_k) || result.top_k <= 0) {
                return false;
            }
        } else if (strcmp(argv[i], "--f16-io") == 0) {
            result.f16_io = true;
        } else if (strcmp(argv[i], "--max-requests") == 0 && i + 1 < argc) {
            int64_t value = 0;
            if (!parse_i64(argv[++i], value) || value < 0 || value > LONG_MAX) {
                return false;
            }
            result.max_requests = static_cast<long>(value);
        } else {
            return false;
        }
    }
    return !result.model.empty() && !result.backend.empty() &&
           !result.bind.empty() && result.port > 0 && result.rows > 0 &&
           result.top_k > 0 && result.top_k <= result.rows;
}

bool read_exact(int fd, void * data, size_t size) {
    uint8_t * ptr = static_cast<uint8_t *>(data);
    while (size > 0) {
        const ssize_t count = read(fd, ptr, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        ptr += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

bool write_exact(int fd, const void * data, size_t size) {
    const uint8_t * ptr = static_cast<const uint8_t *>(data);
    while (size > 0) {
        const ssize_t count = write(fd, ptr, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        ptr += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

bool open_gguf(const std::string & path, gguf_reader & reader, std::string & error) {
    reader.fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    struct stat file_stat = {};
    if (reader.fd < 0 || fstat(reader.fd, &file_stat) != 0 ||
        !S_ISREG(file_stat.st_mode)) {
        error = "cannot open model GGUF";
        return false;
    }
    const std::string fd_path = "/proc/self/fd/" + std::to_string(reader.fd);
    gguf_init_params params = { true, &reader.metadata };
    reader.gguf = gguf_init_from_file(fd_path.c_str(), params);
    if (reader.gguf == nullptr || reader.metadata == nullptr) {
        error = "cannot parse model GGUF";
        return false;
    }
    const int64_t architecture_key = gguf_find_key(reader.gguf, "general.architecture");
    if (architecture_key < 0 ||
        gguf_get_kv_type(reader.gguf, architecture_key) != GGUF_TYPE_STRING ||
        strcmp(gguf_get_val_str(reader.gguf, architecture_key), "gemma4") != 0) {
        error = "LM-head worker requires a Gemma4 GGUF";
        return false;
    }
    return true;
}

bool read_tensor(
        gguf_reader & reader,
        const std::string & name,
        host_tensor & result,
        std::string & error) {
    const int64_t tensor_id = gguf_find_tensor(reader.gguf, name.c_str());
    ggml_tensor * tensor = tensor_id >= 0 ?
            ggml_get_tensor(reader.metadata, name.c_str()) : nullptr;
    if (tensor_id < 0 || tensor == nullptr || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        error = "missing or non-matrix tensor: " + name;
        return false;
    }
    result.ne0 = tensor->ne[0];
    result.ne1 = tensor->ne[1];
    result.type = tensor->type;
    result.bytes.resize(ggml_nbytes(tensor));
    const size_t data_offset = gguf_get_data_offset(reader.gguf);
    const size_t tensor_offset = gguf_get_tensor_offset(reader.gguf, tensor_id);
    if (tensor_offset > std::numeric_limits<size_t>::max() - data_offset) {
        error = "tensor offset overflow: " + name;
        return false;
    }
    const off_t offset = static_cast<off_t>(data_offset + tensor_offset);
    size_t completed = 0;
    while (completed < result.bytes.size()) {
        const ssize_t count = pread(
                reader.fd, result.bytes.data() + completed,
                result.bytes.size() - completed, offset + completed);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            break;
        }
        completed += static_cast<size_t>(count);
    }
    if (completed != result.bytes.size()) {
        error = "short tensor read: " + name;
        return false;
    }
    return true;
}

bool slice_rows(
        const host_tensor & source,
        int64_t row_offset,
        int64_t row_count,
        host_tensor & result) {
    if (row_offset < 0 || row_count <= 0 || row_offset + row_count > source.ne1) {
        return false;
    }
    const size_t row_bytes = ggml_row_size(source.type, source.ne0);
    if (row_bytes == 0 || source.bytes.size() != row_bytes * static_cast<size_t>(source.ne1)) {
        return false;
    }
    result.ne0 = source.ne0;
    result.ne1 = row_count;
    result.type = source.type;
    result.bytes.resize(row_bytes * static_cast<size_t>(row_count));
    memcpy(
            result.bytes.data(),
            source.bytes.data() + row_bytes * static_cast<size_t>(row_offset),
            result.bytes.size());
    return true;
}

bool all_finite(const std::vector<float> & values) {
    for (float value : values) {
        uint32_t bits = 0;
        memcpy(&bits, &value, sizeof(bits));
        if ((bits & 0x7f800000U) == 0x7f800000U) {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    config cfg;
    if (!parse_config(argc, argv, cfg)) {
        fprintf(stderr,
                "usage: %s -m MODEL --rows N --top-k N --backend DEVICE --port P "
                "[--bind ADDRESS] [--f16-io] [--max-requests N]\n",
                argv[0]);
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);

    gguf_reader reader;
    std::string error;
    if (!open_gguf(cfg.model, reader, error)) {
        fprintf(stderr, "[lm-head-worker] %s\n", error.c_str());
        return 1;
    }
    host_tensor full_weight;
    if (!read_tensor(reader, "output.weight", full_weight, error)) {
        error.clear();
        if (!read_tensor(reader, "token_embd.weight", full_weight, error)) {
            fprintf(stderr, "[lm-head-worker] %s\n", error.c_str());
            return 1;
        }
    }
    if (full_weight.ne0 <= 0 || full_weight.ne1 <= cfg.rows) {
        fprintf(stderr, "[lm-head-worker] invalid LM-head tensor shape\n");
        return 1;
    }
    const int64_t n_embd = full_weight.ne0;
    const int64_t n_vocab = full_weight.ne1;
    const int64_t offset = n_vocab - cfg.rows;
    host_tensor weight_data;
    if (!slice_rows(full_weight, offset, cfg.rows, weight_data)) {
        fprintf(stderr, "[lm-head-worker] cannot extract vocabulary suffix\n");
        return 1;
    }
    full_weight.bytes.clear();

    uint64_t weight_hash = 14695981039346656037ULL;
    const uint64_t metadata[] = {
        static_cast<uint64_t>(n_embd), static_cast<uint64_t>(n_vocab),
        static_cast<uint64_t>(offset), static_cast<uint64_t>(cfg.rows),
        static_cast<uint64_t>(weight_data.type),
    };
    weight_hash = lm_head_split::hash64_update(weight_hash, metadata, sizeof(metadata));
    weight_hash = lm_head_split::hash64_update(
            weight_hash, weight_data.bytes.data(), weight_data.bytes.size());

    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name(cfg.backend.c_str());
    if (device == nullptr) {
        fprintf(stderr, "[lm-head-worker] backend not found: %s\n", cfg.backend.c_str());
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "[lm-head-worker] backend initialization failed\n");
        return 1;
    }

    struct shard_state {
        int64_t row_offset = 0;
        int64_t rows = 0;
        ggml_context * weight_ctx = nullptr;
        ggml_tensor * weight = nullptr;
        ggml_backend_buffer_t weight_buffer = nullptr;
        ggml_context * graph_ctx = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * logits = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_backend_buffer_t graph_buffer = nullptr;
    };

    // Quantized HTP MUL_MAT currently accepts at most 32768 output rows.
    const int64_t max_shard_rows =
            cfg.backend.rfind("HTP", 0) == 0 ? 32768 : cfg.rows;
    const size_t weight_row_bytes = ggml_row_size(weight_data.type, n_embd);
    ggml_backend_buffer_type_t weight_buft =
            ggml_backend_get_default_buffer_type(backend);
    if (ggml_is_quantized(weight_data.type) &&
        ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        auto get_extra_bufts = reinterpret_cast<
                ggml_backend_dev_get_extra_bufts_t>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_dev_get_extra_bufts"));
        ggml_backend_buffer_type_t * extra_bufts =
                get_extra_bufts != nullptr ? get_extra_bufts(device) : nullptr;
        if (extra_bufts != nullptr && extra_bufts[0] != nullptr) {
            weight_buft = extra_bufts[0];
        }
    }
    std::vector<shard_state> shards;
    shards.reserve(static_cast<size_t>(
            (cfg.rows + max_shard_rows - 1) / max_shard_rows));
    size_t weight_buffer_bytes = 0;
    for (int64_t row = 0; row < cfg.rows; row += max_shard_rows) {
        shard_state shard;
        shard.row_offset = row;
        shard.rows = std::min(max_shard_rows, cfg.rows - row);

        const ggml_init_params weight_params = {
            ggml_tensor_overhead() * 4, nullptr, true,
        };
        shard.weight_ctx = ggml_init(weight_params);
        shard.weight = ggml_new_tensor_2d(
                shard.weight_ctx, weight_data.type, n_embd, shard.rows);
        shard.weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
                shard.weight_ctx, weight_buft);
        const size_t shard_weight_bytes =
                weight_row_bytes * static_cast<size_t>(shard.rows);
        if (shard.weight_buffer == nullptr ||
            ggml_nbytes(shard.weight) != shard_weight_bytes) {
            fprintf(stderr, "[lm-head-worker] backend weight allocation failed\n");
            return 1;
        }
        ggml_backend_tensor_set(
                shard.weight,
                weight_data.bytes.data() +
                        weight_row_bytes * static_cast<size_t>(row),
                0, shard_weight_bytes);
        weight_buffer_bytes += ggml_backend_buffer_get_size(shard.weight_buffer);

        const ggml_init_params graph_params = {
            ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true,
        };
        shard.graph_ctx = ggml_init(graph_params);
        shard.input = ggml_new_tensor_2d(
                shard.graph_ctx, GGML_TYPE_F32, n_embd, 1);
        ggml_set_input(shard.input);
        shard.logits = ggml_mul_mat(shard.graph_ctx, shard.weight, shard.input);
        shard.graph = ggml_new_graph(shard.graph_ctx);
        ggml_build_forward_expand(shard.graph, shard.logits);
        if (getenv("S41_DISABLE_GRAPH_CACHE") != nullptr) {
            shard.graph->uid = 0;
        }
        for (int i = 0; i < ggml_graph_n_nodes(shard.graph); ++i) {
            ggml_tensor * node = ggml_graph_node(shard.graph, i);
            if (!ggml_backend_supports_op(backend, node)) {
                fprintf(stderr,
                        "[lm-head-worker] backend does not support %s for %lld rows\n",
                        ggml_op_name(node->op), static_cast<long long>(shard.rows));
                return 1;
            }
        }
        shard.graph_buffer =
                ggml_backend_alloc_ctx_tensors(shard.graph_ctx, backend);
        if (shard.graph_buffer == nullptr) {
            fprintf(stderr, "[lm-head-worker] backend graph allocation failed\n");
            return 1;
        }
        ggml_backend_buffer_set_usage(
                shard.graph_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        shards.push_back(shard);
    }
    ggml_backend_synchronize(backend);
    weight_data.bytes.clear();

    std::vector<float> input_data(static_cast<size_t>(n_embd), 0.0f);
    std::vector<float> logits_data(static_cast<size_t>(cfg.rows));
    std::vector<uint32_t> indices(static_cast<size_t>(cfg.rows));
    for (shard_state & shard : shards) {
        ggml_backend_tensor_set(
                shard.input, input_data.data(), 0,
                input_data.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, shard.graph) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[lm-head-worker] warmup failed\n");
            return 1;
        }
        ggml_backend_tensor_get(
                shard.logits,
                logits_data.data() + static_cast<size_t>(shard.row_offset), 0,
                static_cast<size_t>(shard.rows) * sizeof(float));
    }

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(cfg.port));
    if (listen_fd < 0 ||
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0 ||
        inet_pton(AF_INET, cfg.bind.c_str(), &address.sin_addr) != 1 ||
        bind(listen_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
        listen(listen_fd, 4) != 0) {
        fprintf(stderr, "[lm-head-worker] listen setup failed: %s\n", strerror(errno));
        return 1;
    }
    fprintf(stderr,
            "[lm-head-worker] ready backend=%s K=%lld vocab=%lld rows=[%lld,%lld) "
            "top_k=%lld type=%s io=%s shards=%zu weights=%.2f MiB hash=%016llx\n",
            ggml_backend_dev_description(device), static_cast<long long>(n_embd),
            static_cast<long long>(n_vocab), static_cast<long long>(offset),
            static_cast<long long>(n_vocab), static_cast<long long>(cfg.top_k),
            ggml_type_name(weight_data.type), cfg.f16_io ? "f16" : "f32",
            shards.size(),
            static_cast<double>(weight_buffer_bytes) / (1024.0 * 1024.0),
            static_cast<unsigned long long>(weight_hash));
    fflush(stderr);

    const size_t input_bytes = static_cast<size_t>(n_embd) *
            (cfg.f16_io ? sizeof(ggml_fp16_t) : sizeof(float));
    const size_t candidate_bytes = static_cast<size_t>(cfg.top_k) *
            sizeof(lm_head_split::candidate);
    std::vector<uint8_t> request_packet(
            sizeof(lm_head_split::execute_request) + input_bytes);
    std::vector<uint8_t> response_packet(
            sizeof(lm_head_split::execute_response) + candidate_bytes);
    std::vector<ggml_fp16_t> input_f16(
            cfg.f16_io ? static_cast<size_t>(n_embd) : 0);
    long served = 0;

    for (;;) {
        const int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            continue;
        }
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        lm_head_split::hello_request hello = {};
        if (!read_exact(client_fd, &hello, sizeof(hello))) {
            close(client_fd);
            continue;
        }
        const uint16_t expected_flags = cfg.f16_io ? lm_head_split::flag_f16_io : 0;
        const bool hello_ok =
                hello.magic == lm_head_split::protocol_magic &&
                hello.version == lm_head_split::protocol_version &&
                hello.message == static_cast<uint16_t>(lm_head_split::message_type::hello_request) &&
                hello.n_embd == n_embd && hello.rows == cfg.rows &&
                hello.top_k == cfg.top_k && hello.flags == expected_flags;
        lm_head_split::hello_response hello_response = {};
        hello_response.magic = lm_head_split::protocol_magic;
        hello_response.version = lm_head_split::protocol_version;
        hello_response.message = static_cast<uint16_t>(lm_head_split::message_type::hello_response);
        hello_response.status = hello_ok ? 0 : 1;
        hello_response.flags = expected_flags;
        hello_response.n_embd = static_cast<uint32_t>(n_embd);
        hello_response.n_vocab = static_cast<uint32_t>(n_vocab);
        hello_response.offset = static_cast<uint32_t>(offset);
        hello_response.rows = static_cast<uint32_t>(cfg.rows);
        hello_response.weight_type = static_cast<uint32_t>(weight_data.type);
        hello_response.top_k = static_cast<uint32_t>(cfg.top_k);
        hello_response.weight_hash = weight_hash;
        if (!write_exact(client_fd, &hello_response, sizeof(hello_response)) || !hello_ok) {
            close(client_fd);
            continue;
        }
        fprintf(stderr, "[lm-head-worker] client connected\n");

        while (read_exact(client_fd, request_packet.data(), request_packet.size())) {
            lm_head_split::execute_request request = {};
            memcpy(&request, request_packet.data(), sizeof(request));
            const uint8_t * input_payload = request_packet.data() + sizeof(request);
            const bool request_ok =
                    request.magic == lm_head_split::protocol_magic &&
                    request.version == lm_head_split::protocol_version &&
                    request.message == static_cast<uint16_t>(lm_head_split::message_type::execute_request) &&
                    request.request_id != 0 && request.elements == n_embd &&
                    request.payload_bytes == input_bytes &&
                    request.payload_hash == lm_head_split::hash_bytes(input_payload, input_bytes);
            if (!request_ok) {
                fprintf(stderr, "[lm-head-worker] invalid execute request\n");
                break;
            }
            if (cfg.f16_io) {
                memcpy(input_f16.data(), input_payload, input_bytes);
                ggml_fp16_to_fp32_row(input_f16.data(), input_data.data(), input_data.size());
            } else {
                memcpy(input_data.data(), input_payload, input_bytes);
            }
            if (!all_finite(input_data)) {
                fprintf(stderr, "[lm-head-worker] non-finite input\n");
                break;
            }

            const uint64_t compute_start = now_us();
            ggml_status status = GGML_STATUS_SUCCESS;
            for (shard_state & shard : shards) {
                ggml_backend_tensor_set(
                        shard.input, input_data.data(), 0,
                        input_data.size() * sizeof(float));
                status = ggml_backend_graph_compute(backend, shard.graph);
                if (status != GGML_STATUS_SUCCESS) {
                    break;
                }
                ggml_backend_tensor_get(
                        shard.logits,
                        logits_data.data() +
                                static_cast<size_t>(shard.row_offset), 0,
                        static_cast<size_t>(shard.rows) * sizeof(float));
            }
            const uint64_t compute_us = now_us() - compute_start;
            if (status != GGML_STATUS_SUCCESS || !all_finite(logits_data)) {
                fprintf(stderr, "[lm-head-worker] graph execution failed\n");
                break;
            }

            const uint64_t reduce_start = now_us();
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(
                    indices.begin(), indices.begin() + cfg.top_k, indices.end(),
                    [&](uint32_t left, uint32_t right) {
                        if (logits_data[left] != logits_data[right]) {
                            return logits_data[left] > logits_data[right];
                        }
                        return left < right;
                    });
            auto * candidates = reinterpret_cast<lm_head_split::candidate *>(
                    response_packet.data() + sizeof(lm_head_split::execute_response));
            for (int64_t i = 0; i < cfg.top_k; ++i) {
                candidates[i].token_id = static_cast<uint32_t>(offset) + indices[i];
                candidates[i].score = logits_data[indices[i]];
            }
            const uint64_t reduce_us = now_us() - reduce_start;

            lm_head_split::execute_response response = {};
            response.magic = lm_head_split::protocol_magic;
            response.version = lm_head_split::protocol_version;
            response.message = static_cast<uint16_t>(lm_head_split::message_type::execute_response);
            response.status = 0;
            response.request_id = request.request_id;
            response.count = static_cast<uint32_t>(cfg.top_k);
            response.payload_bytes = static_cast<uint32_t>(candidate_bytes);
            response.payload_hash = lm_head_split::hash_bytes(candidates, candidate_bytes);
            response.compute_us = compute_us;
            response.reduce_us = reduce_us;
            memcpy(response_packet.data(), &response, sizeof(response));
            if (!write_exact(client_fd, response_packet.data(), response_packet.size())) {
                break;
            }
            ++served;
            if (served % 32 == 0) {
                fprintf(stderr, "[lm-head-worker] requests=%ld\n", served);
                fflush(stderr);
            }
            if (cfg.max_requests > 0 && served >= cfg.max_requests) {
                close(client_fd);
                close(listen_fd);
                for (shard_state & shard : shards) {
                    ggml_backend_buffer_free(shard.graph_buffer);
                    ggml_free(shard.graph_ctx);
                    ggml_backend_buffer_free(shard.weight_buffer);
                    ggml_free(shard.weight_ctx);
                }
                ggml_backend_free(backend);
                return 0;
            }
        }
        close(client_fd);
        fprintf(stderr, "[lm-head-worker] client disconnected\n");
    }
}
