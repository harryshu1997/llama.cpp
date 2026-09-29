#include "llama.h"
#include "ggml-backend.h"
#include "../../src/llama-cparams.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct ubatch_graph_options {
    std::string model;
    std::string device = "CPU";
    std::string output;
    int ubatch = 64;
    int prefill_tokens = 128;
    int decode_requests = 4;
    int decode_context = 32;
    int gpu_layers = 99;
};

struct ubatch_graph_row {
    llama_token token;
    llama_pos pos;
    llama_seq_id seq_id;
    bool output;
};

struct ubatch_graph_model_owner {
    llama_model * ptr = nullptr;

    ~ubatch_graph_model_owner() {
        if (ptr != nullptr) {
            llama_model_free(ptr);
        }
    }
};

struct ubatch_graph_context_owner {
    llama_context * ptr = nullptr;

    ~ubatch_graph_context_owner() {
        if (ptr != nullptr) {
            llama_free(ptr);
        }
    }
};

struct ubatch_graph_batch_owner {
    llama_batch value = {};

    explicit ubatch_graph_batch_owner(int32_t rows) : value(llama_batch_init(rows, 0, 1)) {}

    ~ubatch_graph_batch_owner() {
        llama_batch_free(value);
    }
};

static bool parse_i32(const char * text, int & value) {
    if (text == nullptr || text[0] == '\0') {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

static void usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s -m MODEL --output FILE [options]\n"
            "  --device NAME          execution device (default CPU)\n"
            "  --ubatch N             physical token limit (default 64)\n"
            "  --prefill N            prefill rows in the mixed batch (default 128)\n"
            "  --decode-requests N    decode rows in the mixed batch (default 4)\n"
            "  --decode-context N     seeded KV rows per decode request (default 32)\n"
            "  -ngl N                 layers assigned to the device (default 99)\n",
            argv0);
}

static bool parse_options(int argc, char ** argv, ubatch_graph_options & options) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        auto parse_next = [&](int & value) {
            return i + 1 < argc && parse_i32(argv[++i], value);
        };
        if ((std::strcmp(arg, "-m") == 0 || std::strcmp(arg, "--model") == 0) && i + 1 < argc) {
            options.model = argv[++i];
        } else if (std::strcmp(arg, "--output") == 0 && i + 1 < argc) {
            options.output = argv[++i];
        } else if (std::strcmp(arg, "--device") == 0 && i + 1 < argc) {
            options.device = argv[++i];
        } else if (std::strcmp(arg, "--ubatch") == 0) {
            if (!parse_next(options.ubatch)) {
                return false;
            }
        } else if (std::strcmp(arg, "--prefill") == 0) {
            if (!parse_next(options.prefill_tokens)) {
                return false;
            }
        } else if (std::strcmp(arg, "--decode-requests") == 0) {
            if (!parse_next(options.decode_requests)) {
                return false;
            }
        } else if (std::strcmp(arg, "--decode-context") == 0) {
            if (!parse_next(options.decode_context)) {
                return false;
            }
        } else if (std::strcmp(arg, "-ngl") == 0) {
            if (!parse_next(options.gpu_layers)) {
                return false;
            }
        } else if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }
    return !options.model.empty() && !options.output.empty() &&
            options.ubatch >= 2 && options.prefill_tokens >= 1 &&
            options.decode_requests >= 1 && options.decode_context >= 1 &&
            options.decode_requests + 1 <= options.ubatch &&
            options.decode_requests + 1 <= LLAMA_MAX_SEQ && options.gpu_layers >= 0;
}

static llama_model * load_model(const ubatch_graph_options & options) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = options.device == "CPU" ? 0 : options.gpu_layers;

    std::vector<ggml_backend_dev_t> devices;
    if (options.device != "CPU") {
        ggml_backend_dev_t selected = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t device = ggml_backend_dev_get(i);
            if (options.device == ggml_backend_dev_name(device)) {
                selected = device;
                break;
            }
        }
        if (selected == nullptr) {
            std::fprintf(stderr, "error: device '%s' not found\n", options.device.c_str());
            return nullptr;
        }
        devices.push_back(selected);
        devices.push_back(nullptr);
        params.devices = devices.data();
    }
    return llama_model_load_from_file(options.model.c_str(), params);
}

static nlohmann::ordered_json tensor_shape(const ggml_tensor * tensor) {
    nlohmann::ordered_json shape = nlohmann::ordered_json::array();
    for (int i = 0; i < ggml_n_dims(tensor); ++i) {
        shape.push_back(tensor->ne[i]);
    }
    return shape;
}

static nlohmann::ordered_json tensor_strides(const ggml_tensor * tensor) {
    nlohmann::ordered_json strides = nlohmann::ordered_json::array();
    for (int i = 0; i < ggml_n_dims(tensor); ++i) {
        strides.push_back(tensor->nb[i]);
    }
    return strides;
}

static std::string tensor_buffer_type(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return "NONE";
    }
    const char * name = ggml_backend_buft_name(ggml_backend_buffer_get_type(tensor->buffer));
    return name == nullptr ? "NONE" : name;
}

static const ggml_tensor * tensor_root(const ggml_tensor * tensor, size_t & offset) {
    offset = 0;
    while (tensor != nullptr && tensor->view_src != nullptr) {
        offset += tensor->view_offs;
        tensor = tensor->view_src;
    }
    return tensor;
}

struct ubatch_graph_capture {
    explicit ubatch_graph_capture(int64_t n_embd) : n_embd(n_embd) {}

    int64_t n_embd;
    bool enabled = false;
    int graph_index = 0;
    int tensor_index = 0;
    int64_t observed_tokens = -1;
    std::map<const ggml_tensor *, std::string> tensor_ids;
    nlohmann::ordered_json nodes = nlohmann::ordered_json::array();
    nlohmann::ordered_json graphs = nlohmann::ordered_json::array();

    std::string tensor_id(const ggml_tensor * tensor) {
        auto found = tensor_ids.find(tensor);
        if (found != tensor_ids.end()) {
            return found->second;
        }
        const std::string id = "t" + std::to_string(tensor_index++);
        tensor_ids.emplace(tensor, id);
        return id;
    }

    nlohmann::ordered_json source_json(const ggml_tensor * tensor) {
        size_t view_offset = 0;
        const ggml_tensor * root = tensor_root(tensor, view_offset);
        return {
            {"tensor_id", tensor_id(tensor)},
            {"name", ggml_get_name(tensor)},
            {"op", ggml_op_name(tensor->op)},
            {"type", ggml_type_name(tensor->type)},
            {"shape", tensor_shape(tensor)},
            {"strides", tensor_strides(tensor)},
            {"nbytes", ggml_nbytes(tensor)},
            {"buffer_type", tensor_buffer_type(tensor)},
            {"block_size", ggml_blck_size(tensor->type)},
            {"root_tensor_id", tensor_id(root)},
            {"root_name", ggml_get_name(root)},
            {"root_type", ggml_type_name(root->type)},
            {"root_shape", tensor_shape(root)},
            {"root_strides", tensor_strides(root)},
            {"root_nbytes", ggml_nbytes(root)},
            {"root_buffer_type", tensor_buffer_type(root)},
            {"root_block_size", ggml_blck_size(root->type)},
            {"view_offset", view_offset},
        };
    }

    void finish_graph(int64_t observed_outputs) {
        graphs.push_back({
            {"index", graph_index++},
            {"observed_tokens", observed_tokens},
            {"observed_outputs", observed_outputs},
            {"nodes", std::move(nodes)},
        });
        tensor_index = 0;
        observed_tokens = -1;
        tensor_ids.clear();
        nodes = nlohmann::ordered_json::array();
    }

    void record(ggml_tensor * tensor) {
        const char * name = ggml_get_name(tensor);
        nlohmann::ordered_json sources = nlohmann::ordered_json::array();
        for (int i = 0; i < GGML_MAX_SRC && tensor->src[i] != nullptr; ++i) {
            sources.push_back(source_json(tensor->src[i]));
        }
        nodes.push_back({
            {"tensor_id", tensor_id(tensor)},
            {"name", name},
            {"op", ggml_op_name(tensor->op)},
            {"type", ggml_type_name(tensor->type)},
            {"shape", tensor_shape(tensor)},
            {"strides", tensor_strides(tensor)},
            {"nbytes", ggml_nbytes(tensor)},
            {"buffer_type", tensor_buffer_type(tensor)},
            {"sources", std::move(sources)},
        });

        if ((std::strcmp(name, "attn_norm-0") == 0 || std::strcmp(name, "ffn_norm-0") == 0) &&
                tensor->ne[0] == n_embd) {
            observed_tokens = ggml_nelements(tensor) / n_embd;
        }
        if (std::strcmp(name, "result_output") == 0) {
            int64_t outputs = 1;
            for (int i = 1; i < ggml_n_dims(tensor); ++i) {
                outputs *= tensor->ne[i];
            }
            finish_graph(outputs);
        }
    }
};

static bool graph_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * capture = static_cast<ubatch_graph_capture *>(user_data);
    if (capture == nullptr || !capture->enabled || !ask || tensor == nullptr) {
        return false;
    }
    capture->record(tensor);
    return false;
}

static llama_token token_at(int index, int salt, int n_vocab, llama_token first) {
    if (index == 0 && first >= 0 && first < n_vocab) {
        return first;
    }
    return static_cast<llama_token>((static_cast<int64_t>(index) * 131 + salt) % n_vocab);
}

static bool decode_rows(llama_context * context, const std::vector<ubatch_graph_row> & rows) {
    ubatch_graph_batch_owner owner(static_cast<int32_t>(rows.size()));
    llama_batch & batch = owner.value;
    batch.n_tokens = static_cast<int32_t>(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        batch.token[i] = rows[i].token;
        batch.pos[i] = rows[i].pos;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = rows[i].seq_id;
        batch.logits[i] = rows[i].output ? 1 : 0;
    }
    const int result = llama_decode(context, batch);
    if (result != 0) {
        std::fprintf(stderr, "error: llama_decode returned %d\n", result);
        return false;
    }
    llama_synchronize(context);
    return true;
}

static nlohmann::ordered_json expected_ubatches(
        const ubatch_graph_options & options,
        const nlohmann::ordered_json & observed) {
    nlohmann::ordered_json result = nlohmann::ordered_json::array();
    std::vector<ubatch_graph_row> rows;
    rows.reserve(static_cast<size_t>(options.decode_requests + options.prefill_tokens));
    for (int seq = 0; seq < options.decode_requests; ++seq) {
        rows.push_back({0, options.decode_context, static_cast<llama_seq_id>(seq), true});
    }
    for (int pos = 0; pos < options.prefill_tokens; ++pos) {
        rows.push_back({
            0,
            pos,
            static_cast<llama_seq_id>(options.decode_requests),
            pos + 1 == options.prefill_tokens,
        });
    }

    size_t row = 0;
    for (size_t index = 0; index < observed.size(); ++index) {
        const int count = observed[index].value("observed_tokens", -1);
        nlohmann::ordered_json spans = nlohmann::ordered_json::array();
        int requested_outputs = 0;
        if (count <= 0 || count > options.ubatch || row + count > rows.size()) {
            result.push_back({
                {"index", index},
                {"n_tokens", count},
                {"valid", false},
                {"spans", std::move(spans)},
            });
            return result;
        }

        size_t end = row + static_cast<size_t>(count);
        while (row < end) {
            const ubatch_graph_row & first = rows[row];
            size_t next = row + 1;
            while (next < end && rows[next].seq_id == first.seq_id &&
                    rows[next].pos == rows[next - 1].pos + 1 &&
                    rows[next].output == first.output) {
                ++next;
            }
            int span_outputs = 0;
            for (size_t i = row; i < next; ++i) {
                span_outputs += rows[i].output ? 1 : 0;
            }
            requested_outputs += span_outputs;
            spans.push_back({
                {"role", first.seq_id < options.decode_requests ? "decode" : "prefill"},
                {"seq_id", first.seq_id},
                {"pos_begin", first.pos},
                {"pos_end", rows[next - 1].pos + 1},
                {"tokens", next - row},
                {"requested_outputs", span_outputs},
            });
            row = next;
        }

        result.push_back({
            {"index", index},
            {"n_tokens", count},
            {"requested_outputs", requested_outputs},
            {"graph_outputs", std::max(1, requested_outputs)},
            {"graph_output_floor_applied", requested_outputs == 0},
            {"valid", true},
            {"spans", std::move(spans)},
        });
    }
    if (row != rows.size()) {
        result.push_back({
            {"index", result.size()},
            {"n_tokens", rows.size() - row},
            {"valid", false},
            {"spans", nlohmann::ordered_json::array()},
        });
    }
    return result;
}

static std::string model_metadata(const llama_model * model, const char * key) {
    char value[256] = {};
    return llama_model_meta_val_str(model, key, value, sizeof(value)) >= 0 ? value : "unknown";
}

static nlohmann::ordered_json available_devices() {
    nlohmann::ordered_json result = nlohmann::ordered_json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        result.push_back(ggml_backend_dev_name(ggml_backend_dev_get(i)));
    }
    return result;
}

int main(int argc, char ** argv) {
    ubatch_graph_options options;
    if (!parse_options(argc, argv, options)) {
        usage(argv[0]);
        return 1;
    }
    {
        std::ifstream existing(options.output, std::ios::binary);
        if (existing.good()) {
            std::fprintf(stderr, "error: output already exists: '%s'\n", options.output.c_str());
            return 1;
        }
    }

    ggml_backend_load_all();
    llama_backend_init();

    ubatch_graph_model_owner model;
    model.ptr = load_model(options);
    if (model.ptr == nullptr) {
        return 2;
    }

    const int n_embd = llama_model_n_embd(model.ptr);
    const int n_layer = llama_model_n_layer(model.ptr);
    const llama_vocab * vocab = llama_model_get_vocab(model.ptr);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (n_embd <= 0 || n_layer <= 0 || n_vocab <= 0) {
        std::fprintf(stderr, "error: invalid model dimensions\n");
        return 2;
    }

    ubatch_graph_capture capture(n_embd);
    llama_context_params params = llama_context_default_params();
    params.n_ctx = static_cast<uint32_t>(
            options.decode_requests * (options.decode_context + 2) + options.prefill_tokens + 8);
    params.n_batch = static_cast<uint32_t>(std::max(
            options.decode_requests * options.decode_context,
            options.decode_requests + options.prefill_tokens));
    params.n_ubatch = static_cast<uint32_t>(options.ubatch);
    params.n_seq_max = static_cast<uint32_t>(options.decode_requests + 1);
    params.n_outputs_max = static_cast<uint32_t>(options.decode_requests + 1);
    params.kv_unified = true;
    params.no_perf = true;
    params.cb_eval = graph_callback;
    params.cb_eval_user_data = &capture;

    ubatch_graph_context_owner context;
    context.ptr = llama_init_from_model(model.ptr, params);
    if (context.ptr == nullptr || static_cast<int>(llama_n_ubatch(context.ptr)) != options.ubatch) {
        std::fprintf(stderr, "error: failed to create the requested context\n");
        return 2;
    }

    llama_token first = llama_vocab_bos(vocab);
    if (first < 0 || first >= n_vocab) {
        first = 0;
    }
    std::vector<ubatch_graph_row> seed;
    seed.reserve(static_cast<size_t>(options.decode_requests * options.decode_context));
    for (int seq = 0; seq < options.decode_requests; ++seq) {
        for (int pos = 0; pos < options.decode_context; ++pos) {
            seed.push_back({
                token_at(pos, 7 + seq * 17, n_vocab, first),
                pos,
                static_cast<llama_seq_id>(seq),
                pos + 1 == options.decode_context,
            });
        }
    }
    if (!decode_rows(context.ptr, seed)) {
        return 3;
    }

    std::vector<ubatch_graph_row> mixed;
    mixed.reserve(static_cast<size_t>(options.decode_requests + options.prefill_tokens));
    for (int seq = 0; seq < options.decode_requests; ++seq) {
        mixed.push_back({
            token_at(options.decode_context, 7 + seq * 17, n_vocab, first),
            options.decode_context,
            static_cast<llama_seq_id>(seq),
            true,
        });
    }
    for (int pos = 0; pos < options.prefill_tokens; ++pos) {
        mixed.push_back({
            token_at(pos, 41, n_vocab, first),
            pos,
            static_cast<llama_seq_id>(options.decode_requests),
            pos + 1 == options.prefill_tokens,
        });
    }

    capture.enabled = true;
    const bool decoded = decode_rows(context.ptr, mixed);
    capture.enabled = false;

    nlohmann::ordered_json expected = expected_ubatches(options, capture.graphs);
    std::string status = decoded ? "PASS" : "FAIL";
    std::string error;
    if (!capture.nodes.empty()) {
        status = "FAIL";
        error = "incomplete physical graph capture";
    } else if (capture.graphs.size() != expected.size()) {
        status = "FAIL";
        error = "physical graph count mismatch";
    } else {
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!expected[i].value("valid", false) ||
                    capture.graphs[i]["observed_tokens"] != expected[i]["n_tokens"] ||
                    capture.graphs[i]["observed_outputs"] != expected[i]["graph_outputs"]) {
                status = "FAIL";
                error = "physical ubatch shape mismatch";
                break;
            }
            capture.graphs[i]["expected"] = expected[i];
        }
    }

    char description[256] = {};
    llama_model_desc(model.ptr, description, sizeof(description));
    const std::string architecture = model_metadata(model.ptr, "general.architecture");
    nlohmann::ordered_json result = {
        {"schema", "s42-llama-ubatch-graph-v1"},
        {"status", status},
        {"error", error.empty() ? nlohmann::ordered_json(nullptr) : nlohmann::ordered_json(error)},
        {"model", {
            {"path", options.model},
            {"architecture", architecture},
            {"description", description},
            {"n_embd", n_embd},
            {"n_layer", n_layer},
            {"n_vocab", n_vocab},
            {"tensor_bytes", llama_model_size(model.ptr)},
            {"parameter_count", llama_model_n_params(model.ptr)},
        }},
        {"runtime", {
            {"requested_device", options.device},
            {"available_devices", available_devices()},
            {"gpu_layers", options.gpu_layers},
            {"system_info", llama_print_system_info()},
        }},
        {"context", {
            {"n_ctx", params.n_ctx},
            {"n_batch", params.n_batch},
            {"n_ubatch", params.n_ubatch},
            {"n_seq_max", params.n_seq_max},
            {"kv_unified", params.kv_unified},
        }},
        {"logical_batch", {
            {"decode_requests", options.decode_requests},
            {"decode_context", options.decode_context},
            {"prefill_tokens", options.prefill_tokens},
            {"rows", options.decode_requests + options.prefill_tokens},
        }},
        {"kv_ownership", {
            {"owner", "llama_context"},
            {"before", nlohmann::ordered_json::array()},
            {"after", nlohmann::ordered_json::array()},
        }},
        {"physical_ubatches", std::move(capture.graphs)},
    };
    for (int seq = 0; seq < options.decode_requests; ++seq) {
        result["kv_ownership"]["before"].push_back({
            {"seq_id", seq}, {"pos_begin", 0}, {"pos_end", options.decode_context},
        });
        result["kv_ownership"]["after"].push_back({
            {"seq_id", seq}, {"pos_begin", 0}, {"pos_end", options.decode_context + 1},
        });
    }
    result["kv_ownership"]["after"].push_back({
        {"seq_id", options.decode_requests},
        {"pos_begin", 0},
        {"pos_end", options.prefill_tokens},
    });

    std::ofstream stream(options.output, std::ios::binary | std::ios::trunc);
    if (!stream) {
        std::fprintf(stderr, "error: cannot open output '%s'\n", options.output.c_str());
        return 4;
    }
    stream << result.dump(-1, ' ', true) << '\n';
    stream.close();
    std::fprintf(stdout,
            "UBATCH_GRAPH status=%s physical_ubatches=%zu output=%s\n",
            status.c_str(), expected.size(), options.output.c_str());
    return status == "PASS" ? 0 : 4;
}
