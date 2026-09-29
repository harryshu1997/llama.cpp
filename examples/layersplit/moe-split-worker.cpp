#include "moe-split-protocol.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "gguf.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <netinet/in.h>
#include <netinet/tcp.h>
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
    uint64_t layer_mask = 0;
    bool f16_io = false;
    long max_requests = 0;
};

struct host_tensor {
    std::vector<uint8_t> bytes;
    int64_t ne[4] = { 1, 1, 1, 1 };
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

struct layer_state {
    int layer = -1;
    host_tensor pre_norm;
    host_tensor post_norm;
    host_tensor router;
    host_tensor router_scale;
    host_tensor gate_up;
    host_tensor down;
    host_tensor down_scale;

    ggml_tensor * pre_norm_weight = nullptr;
    ggml_tensor * post_norm_weight = nullptr;
    ggml_tensor * router_weight = nullptr;
    ggml_tensor * router_scale_weight = nullptr;
    ggml_tensor * gate_up_weight = nullptr;
    ggml_tensor * down_weight = nullptr;
    ggml_tensor * down_scale_weight = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
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

bool parse_layer_spec(const char * text, uint64_t & mask) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    uint64_t parsed_mask = 0;
    std::string remaining(text);
    while (!remaining.empty()) {
        const size_t comma = remaining.find(',');
        const std::string item = remaining.substr(0, comma);
        if (item.empty()) {
            return false;
        }
        const size_t dash = item.find('-');
        int64_t first = 0;
        int64_t last = 0;
        if (dash == std::string::npos) {
            if (!parse_i64(item.c_str(), first)) {
                return false;
            }
            last = first;
        } else if (item.find('-', dash + 1) != std::string::npos ||
                   !parse_i64(item.substr(0, dash).c_str(), first) ||
                   !parse_i64(item.substr(dash + 1).c_str(), last)) {
            return false;
        }
        if (first < 0 || last < first || last >= 64) {
            return false;
        }
        for (int64_t layer = first; layer <= last; ++layer) {
            parsed_mask |= UINT64_C(1) << layer;
        }
        if (comma == std::string::npos) {
            break;
        }
        remaining.erase(0, comma + 1);
    }
    mask = parsed_mask;
    return mask != 0;
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
        } else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            int64_t value = 0;
            if (result.layer_mask != 0 || !parse_i64(argv[++i], value) ||
                value < 0 || value >= 64) {
                return false;
            }
            result.layer_mask = UINT64_C(1) << value;
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            if (result.layer_mask != 0 ||
                !parse_layer_spec(argv[++i], result.layer_mask)) {
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
           !result.bind.empty() && result.port > 0 && result.layer_mask != 0;
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
    if (reader.fd < 0 || fstat(reader.fd, &file_stat) != 0 || !S_ISREG(file_stat.st_mode)) {
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
        error = "MoE split worker requires a Gemma4 GGUF";
        return false;
    }
    return true;
}

bool read_u32(gguf_reader & reader, const char * name, uint32_t & value) {
    const int64_t key = gguf_find_key(reader.gguf, name);
    if (key < 0 || gguf_get_kv_type(reader.gguf, key) != GGUF_TYPE_UINT32) {
        return false;
    }
    value = gguf_get_val_u32(reader.gguf, key);
    return true;
}

bool read_f32(gguf_reader & reader, const char * name, float & value) {
    const int64_t key = gguf_find_key(reader.gguf, name);
    if (key < 0 || gguf_get_kv_type(reader.gguf, key) != GGUF_TYPE_FLOAT32) {
        return false;
    }
    value = gguf_get_val_f32(reader.gguf, key);
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
    if (tensor_id < 0 || tensor == nullptr) {
        error = "missing tensor: " + name;
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        result.ne[i] = tensor->ne[i];
    }
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

bool shape_is(const host_tensor & tensor, int64_t ne0, int64_t ne1, int64_t ne2 = 1) {
    return tensor.ne[0] == ne0 && tensor.ne[1] == ne1 && tensor.ne[2] == ne2 &&
           tensor.ne[3] == 1;
}

bool all_finite(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    });
}

} // namespace

int main(int argc, char ** argv) {
    config cfg;
    if (!parse_config(argc, argv, cfg)) {
        fprintf(stderr,
                "usage: %s -m MODEL (--layer N | --layers SPEC) "
                "--backend DEVICE --port P [--bind ADDRESS] [--f16-io] "
                "[--max-requests N]\n",
                argv[0]);
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);

    gguf_reader reader;
    std::string error;
    if (!open_gguf(cfg.model, reader, error)) {
        fprintf(stderr, "[moe-worker] %s\n", error.c_str());
        return 1;
    }

    uint32_t n_embd = 0;
    uint32_t n_ff_exp = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_layer = 0;
    float rms_eps = 0.0f;
    if (!read_u32(reader, "gemma4.embedding_length", n_embd) ||
        !read_u32(reader, "gemma4.expert_feed_forward_length", n_ff_exp) ||
        !read_u32(reader, "gemma4.expert_count", n_expert) ||
        !read_u32(reader, "gemma4.expert_used_count", n_expert_used) ||
        !read_u32(reader, "gemma4.block_count", n_layer) ||
        !read_f32(reader, "gemma4.attention.layer_norm_rms_epsilon", rms_eps) ||
        n_embd == 0 || n_ff_exp == 0 || n_expert == 0 || n_expert_used == 0 ||
        n_expert_used > n_expert || n_layer == 0 || n_layer > 64 || rms_eps <= 0.0f ||
        (cfg.layer_mask >> n_layer) != 0) {
        fprintf(stderr, "[moe-worker] invalid Gemma4 MoE metadata or layer selection\n");
        return 1;
    }

    std::vector<layer_state> states;
    ggml_type expert_type = GGML_TYPE_COUNT;
    uint64_t weight_hash = 14695981039346656037ULL;
    for (uint32_t layer = 0; layer < n_layer; ++layer) {
        if ((cfg.layer_mask & (UINT64_C(1) << layer)) == 0) {
            continue;
        }
        layer_state state;
        state.layer = static_cast<int>(layer);
        const std::string prefix = "blk." + std::to_string(layer);
        if (!read_tensor(reader, prefix + ".pre_ffw_norm_2.weight", state.pre_norm, error) ||
            !read_tensor(reader, prefix + ".post_ffw_norm_2.weight", state.post_norm, error) ||
            !read_tensor(reader, prefix + ".ffn_gate_inp.weight", state.router, error) ||
            !read_tensor(reader, prefix + ".ffn_gate_inp.scale", state.router_scale, error) ||
            !read_tensor(reader, prefix + ".ffn_gate_up_exps.weight", state.gate_up, error) ||
            !read_tensor(reader, prefix + ".ffn_down_exps.weight", state.down, error) ||
            !read_tensor(reader, prefix + ".ffn_down_exps.scale", state.down_scale, error)) {
            fprintf(stderr, "[moe-worker] %s\n", error.c_str());
            return 1;
        }
        const bool shapes_ok =
                shape_is(state.pre_norm, n_embd, 1) &&
                shape_is(state.post_norm, n_embd, 1) &&
                shape_is(state.router, n_embd, n_expert) &&
                shape_is(state.router_scale, n_embd, 1) &&
                shape_is(state.gate_up, n_embd, 2 * n_ff_exp, n_expert) &&
                shape_is(state.down, n_ff_exp, n_embd, n_expert) &&
                shape_is(state.down_scale, n_expert, 1) &&
                state.pre_norm.type == GGML_TYPE_F32 &&
                state.post_norm.type == GGML_TYPE_F32 &&
                state.router.type == GGML_TYPE_F32 &&
                state.router_scale.type == GGML_TYPE_F32 &&
                state.down_scale.type == GGML_TYPE_F32 &&
                state.gate_up.type == state.down.type;
        if (!shapes_ok || (!states.empty() && state.gate_up.type != expert_type)) {
            fprintf(stderr, "[moe-worker] layer %u tensor type or shape mismatch\n", layer);
            return 1;
        }
        expert_type = state.gate_up.type;
        const uint64_t metadata[] = {
            layer, n_embd, n_ff_exp, n_expert, n_expert_used,
            static_cast<uint64_t>(expert_type),
        };
        weight_hash = moe_split::hash64_update(weight_hash, metadata, sizeof(metadata));
        const host_tensor * tensors[] = {
            &state.pre_norm, &state.post_norm, &state.router, &state.router_scale,
            &state.gate_up, &state.down, &state.down_scale,
        };
        for (const host_tensor * tensor : tensors) {
            weight_hash = moe_split::hash64_update(
                    weight_hash, tensor->bytes.data(), tensor->bytes.size());
        }
        states.push_back(std::move(state));
    }
    if (states.empty()) {
        fprintf(stderr, "[moe-worker] no layers selected\n");
        return 1;
    }

    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name(cfg.backend.c_str());
    if (device == nullptr) {
        fprintf(stderr, "[moe-worker] backend not found: %s\n", cfg.backend.c_str());
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "[moe-worker] backend initialization failed\n");
        return 1;
    }

    ggml_init_params weight_params = {
        ggml_tensor_overhead() * (states.size() * 7 + 8), nullptr, true,
    };
    ggml_context * weight_ctx = ggml_init(weight_params);
    for (layer_state & state : states) {
        state.pre_norm_weight = ggml_new_tensor_1d(weight_ctx, GGML_TYPE_F32, n_embd);
        state.post_norm_weight = ggml_new_tensor_1d(weight_ctx, GGML_TYPE_F32, n_embd);
        state.router_weight = ggml_new_tensor_2d(weight_ctx, GGML_TYPE_F32, n_embd, n_expert);
        state.router_scale_weight = ggml_new_tensor_1d(weight_ctx, GGML_TYPE_F32, n_embd);
        state.gate_up_weight = ggml_new_tensor_3d(
                weight_ctx, expert_type, n_embd, 2 * n_ff_exp, n_expert);
        state.down_weight = ggml_new_tensor_3d(
                weight_ctx, expert_type, n_ff_exp, n_embd, n_expert);
        state.down_scale_weight = ggml_new_tensor_1d(weight_ctx, GGML_TYPE_F32, n_expert);
    }
    ggml_backend_buffer_t weight_buffer =
            ggml_backend_alloc_ctx_tensors(weight_ctx, backend);
    if (weight_buffer == nullptr) {
        fprintf(stderr, "[moe-worker] backend weight allocation failed\n");
        return 1;
    }
    for (layer_state & state : states) {
        struct upload {
            ggml_tensor * destination;
            host_tensor * source;
        } uploads[] = {
            { state.pre_norm_weight, &state.pre_norm },
            { state.post_norm_weight, &state.post_norm },
            { state.router_weight, &state.router },
            { state.router_scale_weight, &state.router_scale },
            { state.gate_up_weight, &state.gate_up },
            { state.down_weight, &state.down },
            { state.down_scale_weight, &state.down_scale },
        };
        for (upload & item : uploads) {
            if (ggml_nbytes(item.destination) != item.source->bytes.size()) {
                fprintf(stderr, "[moe-worker] layer %d weight allocation mismatch\n", state.layer);
                return 1;
            }
            ggml_backend_tensor_set(
                    item.destination, item.source->bytes.data(), 0, item.source->bytes.size());
            item.source->bytes.clear();
        }
    }
    ggml_backend_synchronize(backend);

    ggml_init_params graph_params = {
        states.size() * (ggml_tensor_overhead() * 192 +
                         ggml_graph_overhead_custom(256, false)),
        nullptr, true,
    };
    ggml_context * graph_ctx = ggml_init(graph_params);
    for (layer_state & state : states) {
        state.input = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, n_embd, 1);
        ggml_tensor * normalized = ggml_mul(
                graph_ctx, ggml_rms_norm(graph_ctx, state.input, rms_eps),
                state.pre_norm_weight);

        ggml_tensor * router_input = ggml_rms_norm(graph_ctx, state.input, rms_eps);
        router_input = ggml_scale(graph_ctx, router_input, 1.0f / sqrtf(n_embd));
        router_input = ggml_mul(graph_ctx, router_input, state.router_scale_weight);
        ggml_tensor * logits = ggml_mul_mat(graph_ctx, state.router_weight, router_input);
        ggml_tensor * probabilities = ggml_soft_max(graph_ctx, logits);
        ggml_tensor * selected = ggml_argsort_top_k(
                graph_ctx, probabilities, n_expert_used);

        probabilities = ggml_reshape_3d(graph_ctx, probabilities, 1, n_expert, 1);
        ggml_tensor * weights = ggml_get_rows(graph_ctx, probabilities, selected);
        weights = ggml_reshape_2d(graph_ctx, weights, n_expert_used, 1);
        ggml_tensor * weight_sum = ggml_sum_rows(graph_ctx, weights);
        weight_sum = ggml_clamp(graph_ctx, weight_sum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(graph_ctx, weights, weight_sum);
        weights = ggml_reshape_3d(graph_ctx, weights, 1, n_expert_used, 1);

        normalized = ggml_reshape_3d(graph_ctx, normalized, n_embd, 1, 1);
        ggml_tensor * gate_up = ggml_mul_mat_id(
                graph_ctx, state.gate_up_weight, normalized, selected);
        ggml_tensor * gate = ggml_view_3d(
                graph_ctx, gate_up, n_ff_exp, n_expert_used, 1,
                gate_up->nb[1], gate_up->nb[2], 0);
        ggml_tensor * up = ggml_view_3d(
                graph_ctx, gate_up, n_ff_exp, n_expert_used, 1,
                gate_up->nb[1], gate_up->nb[2], n_ff_exp * gate_up->nb[0]);
        ggml_tensor * activation = ggml_geglu_split(graph_ctx, gate, up);
        ggml_tensor * experts = ggml_mul_mat_id(
                graph_ctx, state.down_weight, activation, selected);

        ggml_tensor * scales = ggml_reshape_3d(
                graph_ctx, state.down_scale_weight, 1, n_expert, 1);
        scales = ggml_repeat_4d(graph_ctx, scales, 1, n_expert, 1, 1);
        scales = ggml_get_rows(graph_ctx, scales, selected);
        experts = ggml_mul(graph_ctx, experts, scales);
        experts = ggml_mul(graph_ctx, experts, weights);

        ggml_tensor * combined = ggml_view_2d(
                graph_ctx, experts, n_embd, 1, experts->nb[2], 0);
        for (uint32_t expert = 1; expert < n_expert_used; ++expert) {
            ggml_tensor * current = ggml_view_2d(
                    graph_ctx, experts, n_embd, 1, experts->nb[2],
                    expert * experts->nb[1]);
            combined = ggml_add(graph_ctx, combined, current);
        }
        state.output = ggml_mul(
                graph_ctx, ggml_rms_norm(graph_ctx, combined, rms_eps),
                state.post_norm_weight);
        state.graph = ggml_new_graph_custom(graph_ctx, 256, false);
        ggml_build_forward_expand(state.graph, state.output);
        if (getenv("S41_DISABLE_GRAPH_CACHE") != nullptr) {
            state.graph->uid = 0;
        }
        for (int node_index = 0;
             node_index < ggml_graph_n_nodes(state.graph); ++node_index) {
            ggml_tensor * node = ggml_graph_node(state.graph, node_index);
            if (!ggml_backend_supports_op(backend, node)) {
                fprintf(stderr,
                        "[moe-worker] layer %d backend does not support %s\n",
                        state.layer, ggml_op_name(node->op));
                return 1;
            }
        }
    }
    ggml_backend_buffer_t graph_buffer =
            ggml_backend_alloc_ctx_tensors(graph_ctx, backend);
    if (graph_buffer == nullptr) {
        fprintf(stderr, "[moe-worker] backend graph allocation failed\n");
        return 1;
    }

    std::vector<float> input_data(n_embd, 0.0f);
    std::vector<float> output_data(n_embd);
    for (layer_state & state : states) {
        ggml_backend_tensor_set(
                state.input, input_data.data(), 0, input_data.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, state.graph) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "[moe-worker] layer %d warmup failed\n", state.layer);
            return 1;
        }
        ggml_backend_tensor_get(
                state.output, output_data.data(), 0, output_data.size() * sizeof(float));
        if (!all_finite(output_data)) {
            fprintf(stderr, "[moe-worker] layer %d warmup produced non-finite output\n", state.layer);
            return 1;
        }
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
        fprintf(stderr, "[moe-worker] listen setup failed: %s\n", strerror(errno));
        return 1;
    }
    fprintf(stderr,
            "[moe-worker] ready backend=%s layers=%zu mask=%016llx K=%u NFF=%u "
            "experts=%u top_k=%u type=%s io=%s weights=%.2f MiB hash=%016llx\n",
            ggml_backend_dev_description(device), states.size(),
            static_cast<unsigned long long>(cfg.layer_mask), n_embd, n_ff_exp,
            n_expert, n_expert_used, ggml_type_name(expert_type),
            cfg.f16_io ? "f16" : "f32",
            static_cast<double>(ggml_backend_buffer_get_size(weight_buffer)) / (1024.0 * 1024.0),
            static_cast<unsigned long long>(weight_hash));
    fflush(stderr);

    const size_t payload_bytes = static_cast<size_t>(n_embd) *
            (cfg.f16_io ? sizeof(ggml_fp16_t) : sizeof(float));
    std::vector<uint8_t> request_packet(
            sizeof(moe_split::execute_request) + payload_bytes);
    std::vector<uint8_t> response_packet(
            sizeof(moe_split::execute_response) + payload_bytes);
    std::vector<ggml_fp16_t> input_f16(cfg.f16_io ? n_embd : 0);
    std::vector<ggml_fp16_t> output_f16(cfg.f16_io ? n_embd : 0);
    long served = 0;
    std::vector<double> compute_samples;

    for (;;) {
        const int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            continue;
        }
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

        moe_split::hello_request hello = {};
        if (!read_exact(client_fd, &hello, sizeof(hello))) {
            close(client_fd);
            continue;
        }
        const uint16_t expected_flags = cfg.f16_io ? moe_split::flag_f16_io : 0;
        const bool hello_ok =
                hello.magic == moe_split::protocol_magic &&
                hello.version == moe_split::protocol_version &&
                hello.message == static_cast<uint16_t>(moe_split::message_type::hello_request) &&
                hello.layer_mask == cfg.layer_mask && hello.n_embd == n_embd &&
                hello.flags == expected_flags;
        moe_split::hello_response hello_response = {};
        hello_response.magic = moe_split::protocol_magic;
        hello_response.version = moe_split::protocol_version;
        hello_response.message = static_cast<uint16_t>(moe_split::message_type::hello_response);
        hello_response.status = hello_ok ? 0 : 1;
        hello_response.flags = expected_flags;
        hello_response.n_embd = n_embd;
        hello_response.n_ff_exp = n_ff_exp;
        hello_response.n_expert = n_expert;
        hello_response.n_expert_used = n_expert_used;
        hello_response.weight_type = static_cast<uint32_t>(expert_type);
        hello_response.layer_count = static_cast<uint32_t>(states.size());
        hello_response.layer_mask = cfg.layer_mask;
        hello_response.weight_hash = weight_hash;
        if (!write_exact(client_fd, &hello_response, sizeof(hello_response)) || !hello_ok) {
            close(client_fd);
            continue;
        }
        fprintf(stderr, "[moe-worker] client connected\n");

        while (read_exact(client_fd, request_packet.data(), request_packet.size())) {
            moe_split::execute_request request = {};
            memcpy(&request, request_packet.data(), sizeof(request));
            const uint8_t * input_payload = request_packet.data() + sizeof(request);
            const bool request_ok =
                    request.magic == moe_split::protocol_magic &&
                    request.version == moe_split::protocol_version &&
                    request.message == static_cast<uint16_t>(moe_split::message_type::execute_request) &&
                    request.request_id != 0 && request.layer >= 0 && request.layer < 64 &&
                    (cfg.layer_mask & (UINT64_C(1) << request.layer)) != 0 &&
                    request.elements == n_embd && request.payload_bytes == payload_bytes &&
                    request.payload_hash == moe_split::hash_bytes(input_payload, payload_bytes);
            if (!request_ok) {
                fprintf(stderr, "[moe-worker] invalid execute request\n");
                break;
            }
            if (cfg.f16_io) {
                memcpy(input_f16.data(), input_payload, payload_bytes);
                ggml_fp16_to_fp32_row(input_f16.data(), input_data.data(), input_data.size());
            } else {
                memcpy(input_data.data(), input_payload, payload_bytes);
            }
            if (!all_finite(input_data)) {
                fprintf(stderr, "[moe-worker] non-finite input\n");
                break;
            }

            const auto state_it = std::lower_bound(
                    states.begin(), states.end(), request.layer,
                    [](const layer_state & state, int layer) { return state.layer < layer; });
            if (state_it == states.end() || state_it->layer != request.layer) {
                fprintf(stderr, "[moe-worker] layer lookup failed\n");
                break;
            }
            layer_state & state = *state_it;
            const uint64_t started = now_us();
            ggml_backend_tensor_set(
                    state.input, input_data.data(), 0, input_data.size() * sizeof(float));
            const ggml_status status = ggml_backend_graph_compute(backend, state.graph);
            ggml_backend_tensor_get(
                    state.output, output_data.data(), 0, output_data.size() * sizeof(float));
            const uint64_t compute_us = now_us() - started;
            if (status != GGML_STATUS_SUCCESS || !all_finite(output_data)) {
                fprintf(stderr, "[moe-worker] graph execution failed\n");
                break;
            }

            void * output_payload = response_packet.data() + sizeof(moe_split::execute_response);
            if (cfg.f16_io) {
                ggml_fp32_to_fp16_row(output_data.data(), output_f16.data(), output_f16.size());
                memcpy(output_payload, output_f16.data(), payload_bytes);
            } else {
                memcpy(output_payload, output_data.data(), payload_bytes);
            }
            moe_split::execute_response response = {};
            response.magic = moe_split::protocol_magic;
            response.version = moe_split::protocol_version;
            response.message = static_cast<uint16_t>(moe_split::message_type::execute_response);
            response.status = 0;
            response.request_id = request.request_id;
            response.layer = request.layer;
            response.elements = n_embd;
            response.payload_bytes = static_cast<uint32_t>(payload_bytes);
            response.payload_hash = moe_split::hash_bytes(output_payload, payload_bytes);
            response.compute_us = compute_us;
            memcpy(response_packet.data(), &response, sizeof(response));
            if (!write_exact(client_fd, response_packet.data(), response_packet.size())) {
                break;
            }
            compute_samples.push_back(static_cast<double>(compute_us));
            ++served;
            if (served % 32 == 0) {
                std::vector<double> sorted = compute_samples;
                std::sort(sorted.begin(), sorted.end());
                fprintf(stderr,
                        "[moe-worker] requests=%ld compute_p50_us=%.0f\n",
                        served, sorted[sorted.size() / 2]);
                fflush(stderr);
            }
            if (cfg.max_requests > 0 && served >= cfg.max_requests) {
                close(client_fd);
                close(listen_fd);
                ggml_backend_buffer_free(graph_buffer);
                ggml_free(graph_ctx);
                ggml_backend_buffer_free(weight_buffer);
                ggml_free(weight_ctx);
                ggml_backend_free(backend);
                return 0;
            }
        }
        close(client_fd);
        fprintf(stderr, "[moe-worker] client disconnected\n");
    }
}
