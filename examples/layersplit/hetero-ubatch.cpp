#include "llama.h"
#include "ggml-backend.h"
#include "../../src/llama-cparams.h"
#include "../../src/llama-ext.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

static constexpr double HETERO_SAME_BACKEND_RELATIVE_L2_LIMIT = 1e-6;
static constexpr double HETERO_CROSS_BACKEND_RELATIVE_L2_LIMIT = 0.05;
static constexpr int HETERO_DEFAULT_VALIDATION_STEPS = 4;

struct hetero_placement_tally {
    int64_t metadata_nodes = 0;
    int64_t compute_nodes = 0;
    int64_t copy_nodes = 0;
    int64_t missing_buffer_compute_nodes = 0;
    std::map<std::string, int64_t> metadata_by_op;
    std::map<std::string, int64_t> copy_by_op;
    std::map<std::string, int64_t> compute_by_buffer_type;
    std::map<std::string, int64_t> compute_by_op;
    std::map<std::string, std::map<std::string, int64_t>> compute_by_buffer_and_op;
};

static double hetero_now_ms();

struct hetero_wavefront_trace {
    double epoch_ms = 0.0;
    std::vector<double> gpu_tile_done_ms;
    std::vector<double> gpu_wait_npu_ms;
    std::vector<double> transfer_start_ms;
    std::vector<double> transfer_done_ms;
    std::vector<double> gpu_tile_release_ms;
    std::vector<double> npu_wait_start_ms;
    std::vector<double> npu_tile_release_ms;
    std::vector<size_t> transfer_bytes;
    double npu_first_ubatch_done_ms = 0.0;
};

struct hetero_wavefront {
    int n_layer = 0;
    int tile_layers = 0;
    int n_tiles = 0;
    llama_context * gpu_ctx = nullptr;
    llama_context * npu_ctx = nullptr;
    llama_seq_id seq_id = 1;
    llama_pos p0 = 0;
    llama_pos p1 = 0;

    std::atomic<bool> active {false};
    std::atomic<bool> abort {false};
    std::atomic<int> error_code {0};
    std::atomic<int> npu_waiting_tile {-1};
    std::atomic<int> ready_tiles {0};

    int gpu_next_tile = 0;
    int npu_next_tile = 0;
    bool npu_first_ubatch_done = false;
    hetero_wavefront_trace trace;

    void reset(
            int layers,
            int tile,
            llama_context * gpu,
            llama_context * npu,
            llama_seq_id sequence,
            llama_pos pos0,
            llama_pos pos1,
            double epoch_ms) {
        n_layer = layers;
        tile_layers = tile;
        n_tiles = (n_layer + tile_layers - 1) / tile_layers;
        gpu_ctx = gpu;
        npu_ctx = npu;
        seq_id = sequence;
        p0 = pos0;
        p1 = pos1;
        gpu_next_tile = 0;
        npu_next_tile = 0;
        npu_first_ubatch_done = false;
        abort.store(false, std::memory_order_relaxed);
        error_code.store(0, std::memory_order_relaxed);
        npu_waiting_tile.store(-1, std::memory_order_relaxed);
        ready_tiles.store(0, std::memory_order_relaxed);

        trace = {};
        trace.epoch_ms = epoch_ms;
        trace.gpu_tile_done_ms.resize((size_t) n_tiles);
        trace.gpu_wait_npu_ms.resize((size_t) n_tiles);
        trace.transfer_start_ms.resize((size_t) n_tiles);
        trace.transfer_done_ms.resize((size_t) n_tiles);
        trace.gpu_tile_release_ms.resize((size_t) n_tiles);
        trace.npu_wait_start_ms.resize((size_t) n_tiles);
        trace.npu_tile_release_ms.resize((size_t) n_tiles);
        trace.transfer_bytes.resize((size_t) n_tiles);
        active.store(true, std::memory_order_release);
    }

    void fail(int code) {
        int expected = 0;
        error_code.compare_exchange_strong(expected, code, std::memory_order_relaxed);
        abort.store(true, std::memory_order_release);
    }

    bool wait_until(const std::atomic<int> & value, int target, int timeout_code) {
        const double deadline_ms = hetero_now_ms() + 120000.0;
        while (value.load(std::memory_order_acquire) < target) {
            if (abort.load(std::memory_order_acquire)) {
                return false;
            }
            if (hetero_now_ms() >= deadline_ms) {
                fail(timeout_code);
                return false;
            }
            std::this_thread::yield();
        }
        return true;
    }

    static bool node_is(const ggml_tensor * tensor, const char * prefix, int il) {
        char expected[GGML_MAX_NAME];
        snprintf(expected, sizeof(expected), "%s-%d", prefix, il);
        return strcmp(ggml_get_name(tensor), expected) == 0;
    }

    bool gpu_callback(ggml_tensor * tensor, bool ask) {
        if (!active.load(std::memory_order_acquire) || gpu_next_tile >= n_tiles) {
            return false;
        }

        const int tile = gpu_next_tile;
        const int il0 = tile * tile_layers;
        const int il1 = std::min(n_layer, il0 + tile_layers);
        if (!node_is(tensor, "l_out", il1 - 1)) {
            return false;
        }
        if (ask) {
            return true;
        }

        trace.gpu_tile_done_ms[(size_t) tile] = hetero_now_ms() - trace.epoch_ms;
        const double wait_start_ms = hetero_now_ms();
        if (!wait_until(npu_waiting_tile, tile, 1)) {
            return false;
        }
        trace.gpu_wait_npu_ms[(size_t) tile] = hetero_now_ms() - wait_start_ms;

        trace.transfer_start_ms[(size_t) tile] = hetero_now_ms() - trace.epoch_ms;
        size_t bytes = 0;
        if (!llama_kv_cache_wavefront_copy(
                    npu_ctx, gpu_ctx, seq_id, seq_id, p0, p1,
                    (uint32_t) il0, (uint32_t) il1, &bytes)) {
            fail(3);
            return false;
        }
        trace.transfer_done_ms[(size_t) tile] = hetero_now_ms() - trace.epoch_ms;
        trace.transfer_bytes[(size_t) tile] = bytes;
        trace.gpu_tile_release_ms[(size_t) tile] = hetero_now_ms() - trace.epoch_ms;

        ++gpu_next_tile;
        ready_tiles.store(gpu_next_tile, std::memory_order_release);
        return true;
    }

    bool npu_callback(ggml_tensor * tensor, bool ask) {
        if (!active.load(std::memory_order_acquire) || npu_first_ubatch_done) {
            return false;
        }

        if (npu_next_tile < n_tiles) {
            const int tile = npu_next_tile;
            const int il0 = tile * tile_layers;
            if (!node_is(tensor, "attn_norm", il0)) {
                return false;
            }
            if (ask) {
                return true;
            }

            trace.npu_wait_start_ms[(size_t) tile] = hetero_now_ms() - trace.epoch_ms;
            npu_waiting_tile.store(tile, std::memory_order_release);
            if (!wait_until(ready_tiles, tile + 1, 2)) {
                return false;
            }
            trace.npu_tile_release_ms[(size_t) tile] = hetero_now_ms() - trace.epoch_ms;
            ++npu_next_tile;
            return true;
        }

        if (strcmp(ggml_get_name(tensor), "result_output") != 0) {
            return false;
        }
        if (ask) {
            return true;
        }

        trace.npu_first_ubatch_done_ms = hetero_now_ms() - trace.epoch_ms;
        npu_first_ubatch_done = true;
        return true;
    }
};

struct hetero_eval_state {
    hetero_placement_tally * placement = nullptr;
    hetero_wavefront * wavefront = nullptr;
    bool gpu_lane = false;
};

static bool hetero_metadata_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

static bool hetero_placement_cb(struct ggml_tensor * tensor, bool ask, void * user_data) {
    if (!ask || tensor == nullptr || user_data == nullptr) {
        return false;
    }

    hetero_placement_tally * tally = (hetero_placement_tally *) user_data;
    const std::string op_name = ggml_op_name(tensor->op);
    if (hetero_metadata_op(tensor->op)) {
        ++tally->metadata_nodes;
        ++tally->metadata_by_op[op_name];
        return false;
    }
    if (tensor->op == GGML_OP_DUP || tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_CONT) {
        ++tally->copy_nodes;
        ++tally->copy_by_op[op_name];
        return false;
    }

    ++tally->compute_nodes;
    ++tally->compute_by_op[op_name];
    std::string buffer_name = "NONE";
    if (tensor->buffer == nullptr) {
        ++tally->missing_buffer_compute_nodes;
    } else {
        const char * name = ggml_backend_buft_name(ggml_backend_buffer_get_type(tensor->buffer));
        buffer_name = name ? name : "NONE";
    }
    ++tally->compute_by_buffer_type[buffer_name];
    ++tally->compute_by_buffer_and_op[buffer_name][op_name];
    return false;
}

static bool hetero_eval_cb(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto * state = (hetero_eval_state *) user_data;
    if (!state) {
        return false;
    }
    if (state->placement) {
        hetero_placement_cb(tensor, ask, state->placement);
    }
    if (!state->wavefront) {
        return false;
    }
    return state->gpu_lane ?
        state->wavefront->gpu_callback(tensor, ask) :
        state->wavefront->npu_callback(tensor, ask);
}

static nlohmann::ordered_json hetero_placement_json(const hetero_placement_tally & tally) {
    return {
        {"metadata_nodes", tally.metadata_nodes},
        {"compute_nodes", tally.compute_nodes},
        {"copy_nodes", tally.copy_nodes},
        {"missing_buffer_compute_nodes", tally.missing_buffer_compute_nodes},
        {"metadata_by_op", tally.metadata_by_op},
        {"copy_by_op", tally.copy_by_op},
        {"compute_by_buffer_type", tally.compute_by_buffer_type},
        {"compute_by_op", tally.compute_by_op},
        {"compute_by_buffer_and_op", tally.compute_by_buffer_and_op},
    };
}

static int64_t hetero_unexpected_compute_nodes(
        const hetero_placement_tally & tally,
        const std::string & expected_buffer) {
    int64_t unexpected = 0;
    for (const auto & buffer : tally.compute_by_buffer_and_op) {
        if (buffer.first == expected_buffer) {
            continue;
        }
        for (const auto & op : buffer.second) {
            unexpected += op.second;
        }
    }
    return unexpected;
}

static nlohmann::ordered_json hetero_unexpected_compute_json(
        const hetero_placement_tally & tally,
        const std::string & expected_buffer) {
    nlohmann::ordered_json result = nlohmann::ordered_json::object();
    for (const auto & buffer : tally.compute_by_buffer_and_op) {
        if (buffer.first == expected_buffer) {
            continue;
        }
        result[buffer.first] = buffer.second;
    }
    return result;
}

static bool hetero_placement_certified(
        const hetero_placement_tally & tally,
        const std::string & expected_buffer) {
    return tally.compute_nodes > 0 &&
        tally.missing_buffer_compute_nodes == 0 &&
        hetero_unexpected_compute_nodes(tally, expected_buffer) == 0;
}

static std::string hetero_expected_buffer(const std::string & device_name) {
    if (device_name == "GPUOpenCL") {
        return "OpenCL";
    }
    return device_name;
}

static double hetero_placement_fraction(
        const hetero_placement_tally & tally,
        const std::string & buffer_name) {
    const auto found = tally.compute_by_buffer_type.find(buffer_name);
    const int64_t on_target = found == tally.compute_by_buffer_type.end() ? 0 : found->second;
    return tally.compute_nodes > 0 ? (double) on_target / (double) tally.compute_nodes : 0.0;
}

static double hetero_now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool hetero_parse_i32(const char * text, int & value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    char * end = nullptr;
    const long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX) {
        return false;
    }
    value = (int) parsed;
    return true;
}

static void hetero_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m MODEL [options]\n"
        "  --dev-gpu NAME       first mixed ubatch and decode lane (default GPUOpenCL)\n"
        "  --dev-npu NAME       remaining prefill lane (default HTP0)\n"
        "  --ubatch N, -b N     physical ubatch rows (default 64)\n"
        "  --prompt-len N       prefill request length, must be >= ubatch (default 256)\n"
        "  --decode-ctx N       existing decode request length (default 32)\n"
        "  --decode-requests N  decode rows placed first in the mixed ubatch (default 1)\n"
        "  --decode-steps N     maximum GPU steps overlapped with NPU prefill (default 32)\n"
        "  --validation-steps N transferred-prefill continuation checks (default 4)\n"
        "  --wavefront-layers N layers per GPU/NPU wavefront tile; 0 uses full-state handoff (default 8)\n"
        "  --no-warmup          skip untimed GPU and NPU shape warmups\n"
        "  --placement-audit    collect scheduled-node placement (changes timing)\n"
        "  --single-device-control  run only the original mixed batch on --dev-gpu\n"
        "  -ngl N               layers assigned to the selected device (default 99)\n",
        argv0);
}

static llama_model * hetero_load_model(
        const std::string & model_path,
        const std::string & device_name,
        int n_gpu_layers) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = device_name == "CPU" ? 0 : n_gpu_layers;

    std::vector<ggml_backend_dev_t> devices;
    if (device_name != "CPU") {
        ggml_backend_dev_t selected = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t device = ggml_backend_dev_get(i);
            if (device_name == ggml_backend_dev_name(device)) {
                selected = device;
                break;
            }
        }
        if (selected == nullptr) {
            fprintf(stderr, "error: device '%s' not found; available devices:\n", device_name.c_str());
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                fprintf(stderr, "  %s\n", ggml_backend_dev_name(ggml_backend_dev_get(i)));
            }
            return nullptr;
        }
        devices.push_back(selected);
        devices.push_back(nullptr);
        params.devices = devices.data();
    }

    return llama_model_load_from_file(model_path.c_str(), params);
}

struct hetero_model_owner {
    llama_model * ptr = nullptr;
    ~hetero_model_owner() {
        if (ptr != nullptr) {
            llama_model_free(ptr);
        }
    }
};

struct hetero_context_owner {
    llama_context * ptr = nullptr;
    ~hetero_context_owner() {
        if (ptr != nullptr) {
            llama_free(ptr);
        }
    }
};

struct hetero_row {
    llama_token token;
    llama_pos pos;
    llama_seq_id seq_id;
    bool output;
};

struct hetero_decode_timing {
    double backend_ms = 0.0;
    double output_copy_ms = 0.0;
    double total_ms = 0.0;
    double backend_done_at_ms = 0.0;
};

static bool hetero_decode(
        llama_context * ctx,
        const std::vector<hetero_row> & rows,
        int n_vocab,
        std::vector<std::vector<float>> & output_logits,
        double & elapsed_ms,
        bool synchronize = true,
        hetero_decode_timing * timing = nullptr) {
    output_logits.clear();
    output_logits.resize(rows.size());
    if (rows.empty()) {
        return false;
    }

    llama_batch batch = llama_batch_init((int32_t) rows.size(), 0, 1);
    batch.n_tokens = (int32_t) rows.size();
    for (size_t i = 0; i < rows.size(); ++i) {
        batch.token[i] = rows[i].token;
        batch.pos[i] = rows[i].pos;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = rows[i].seq_id;
        batch.logits[i] = rows[i].output ? 1 : 0;
    }

    const double start_ms = hetero_now_ms();
    const int decode_rc = llama_decode(ctx, batch);
    bool ok = decode_rc == 0;
    if (!ok) {
        fprintf(stderr, "error: llama_decode returned %d for %zu rows\n", decode_rc, rows.size());
    }
    if (ok && synchronize) {
        llama_synchronize(ctx); // synchronized calls are scheduler boundaries
    }
    const double backend_done_at_ms = hetero_now_ms();
    if (ok) {
        for (size_t i = 0; i < rows.size(); ++i) {
            if (!rows[i].output) {
                continue;
            }
            const float * logits = llama_get_logits_ith(ctx, (int32_t) i);
            if (logits == nullptr) {
                fprintf(stderr, "error: logits are null for output row %zu\n", i);
                ok = false;
                break;
            }
            output_logits[i].assign(logits, logits + n_vocab);
        }
    }
    const double output_done_at_ms = hetero_now_ms();
    elapsed_ms = output_done_at_ms - start_ms;
    if (timing != nullptr) {
        timing->backend_ms = backend_done_at_ms - start_ms;
        timing->output_copy_ms = output_done_at_ms - backend_done_at_ms;
        timing->total_ms = elapsed_ms;
        timing->backend_done_at_ms = backend_done_at_ms;
    }
    llama_batch_free(batch);
    return ok;
}

static llama_token hetero_argmax(const std::vector<float> & logits) {
    if (logits.empty() || !std::isfinite(logits.front())) {
        return -1;
    }
    size_t best = 0;
    for (size_t i = 1; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i])) {
            return -1;
        }
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return (llama_token) best;
}

struct hetero_diff {
    double relative_l2 = 0.0;
    double max_abs = 0.0;
    size_t nonfinite_values = 0;
};

static hetero_diff hetero_logits_diff(
        const std::vector<float> & reference,
        const std::vector<float> & candidate) {
    hetero_diff result;
    if (reference.size() != candidate.size() || reference.empty()) {
        result.relative_l2 = INFINITY;
        result.max_abs = INFINITY;
        return result;
    }

    double sum_diff_2 = 0.0;
    double sum_ref_2 = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i])) {
            ++result.nonfinite_values;
            continue;
        }
        const double diff = (double) candidate[i] - (double) reference[i];
        sum_diff_2 += diff * diff;
        sum_ref_2 += (double) reference[i] * (double) reference[i];
        result.max_abs = std::max(result.max_abs, std::fabs(diff));
    }
    if (result.nonfinite_values > 0) {
        result.relative_l2 = INFINITY;
        result.max_abs = INFINITY;
    } else {
        result.relative_l2 = sum_ref_2 > 0.0 ?
            std::sqrt(sum_diff_2 / sum_ref_2) : std::sqrt(sum_diff_2);
    }
    return result;
}

static bool hetero_diff_within(const hetero_diff & diff, double relative_l2_limit) {
    return diff.nonfinite_values == 0 &&
        std::isfinite(diff.relative_l2) &&
        std::isfinite(diff.max_abs) &&
        diff.relative_l2 <= relative_l2_limit;
}

static constexpr int hetero_result_code(
        bool placement_audit,
        bool placement_ok,
        bool correctness_ok) {
    return placement_audit && !placement_ok ? 5 : (correctness_ok ? 0 : 4);
}

static_assert(hetero_result_code(false, true, false) == 4);
static_assert(hetero_result_code(true, false, true) == 5);

struct hetero_state_blob {
    std::vector<uint8_t> data;
    double export_ms = 0.0;
};

static bool hetero_export_state(
        llama_context * ctx,
        llama_seq_id seq_id,
        hetero_state_blob & blob) {
    const double start_ms = hetero_now_ms();
    const size_t size = llama_state_seq_get_size(ctx, seq_id);
    if (size == 0) {
        fprintf(stderr, "error: state size is zero for sequence %d\n", seq_id);
        return false;
    }
    blob.data.resize(size);
    const size_t written = llama_state_seq_get_data(ctx, blob.data.data(), blob.data.size(), seq_id);
    blob.export_ms = hetero_now_ms() - start_ms;
    if (written == 0 || written > blob.data.size()) {
        fprintf(stderr, "error: failed to export sequence %d state (%zu/%zu bytes)\n",
                seq_id, written, blob.data.size());
        return false;
    }
    blob.data.resize(written);
    return true;
}

static bool hetero_import_state(
        llama_context * ctx,
        llama_seq_id seq_id,
        const hetero_state_blob & blob,
        double & import_ms) {
    llama_memory_t memory = llama_get_memory(ctx);
    if (!llama_memory_seq_rm(memory, seq_id, -1, -1)) {
        fprintf(stderr, "error: failed to clear destination sequence %d\n", seq_id);
        return false;
    }

    const double start_ms = hetero_now_ms();
    const size_t read = llama_state_seq_set_data(ctx, blob.data.data(), blob.data.size(), seq_id);
    import_ms = hetero_now_ms() - start_ms;
    if (read == 0) {
        fprintf(stderr, "error: failed to import %zu bytes into sequence %d\n", blob.data.size(), seq_id);
        return false;
    }
    if (read != blob.data.size()) {
        fprintf(stderr, "error: partial state import for sequence %d (%zu/%zu bytes)\n",
                seq_id, read, blob.data.size());
        return false;
    }
    return true;
}

static bool hetero_restore_states(
        llama_context * ctx,
        const std::vector<hetero_state_blob> & states,
        double & import_ms) {
    import_ms = 0.0;
    for (size_t i = 0; i < states.size(); ++i) {
        double current_ms = 0.0;
        if (!hetero_import_state(ctx, (llama_seq_id) i, states[i], current_ms)) {
            return false;
        }
        import_ms += current_ms;
    }
    return true;
}

static llama_token hetero_token_at(int index, int salt, int n_vocab, llama_token first) {
    if (index == 0 && first >= 0 && first < n_vocab) {
        return first;
    }
    const int64_t value = (int64_t) index * 131 + salt;
    return (llama_token) (value % n_vocab);
}

static double hetero_median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5 * (values[middle - 1] + values[middle]);
    }
    return values[middle];
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string gpu_device = "GPUOpenCL";
    std::string npu_device = "HTP0";
    int ubatch = 64;
    int prompt_len = 256;
    int decode_ctx = 32;
    int decode_requests = 1;
    int decode_steps = 32;
    int validation_steps = HETERO_DEFAULT_VALIDATION_STEPS;
    int wavefront_layers = 8;
    int n_gpu_layers = 99;
    bool warmup = true;
    bool placement_audit = false;
    bool single_device_control = false;
    bool wavefront_layers_set = false;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        auto parse_next = [&](int & value) {
            return i + 1 < argc && hetero_parse_i32(argv[++i], value);
        };
        if (strcmp(arg, "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((strcmp(arg, "--dev-gpu") == 0 || strcmp(arg, "--dev-decode") == 0) && i + 1 < argc) {
            gpu_device = argv[++i];
        } else if ((strcmp(arg, "--dev-npu") == 0 || strcmp(arg, "--dev-prefill") == 0) && i + 1 < argc) {
            npu_device = argv[++i];
        } else if (strcmp(arg, "--ubatch") == 0 || strcmp(arg, "-b") == 0) {
            if (!parse_next(ubatch)) {
                fprintf(stderr, "error: invalid ubatch\n");
                return 1;
            }
        } else if (strcmp(arg, "--prompt-len") == 0) {
            if (!parse_next(prompt_len)) {
                fprintf(stderr, "error: invalid prompt length\n");
                return 1;
            }
        } else if (strcmp(arg, "--decode-ctx") == 0) {
            if (!parse_next(decode_ctx)) {
                fprintf(stderr, "error: invalid decode context\n");
                return 1;
            }
        } else if (strcmp(arg, "--decode-requests") == 0) {
            if (!parse_next(decode_requests)) {
                fprintf(stderr, "error: invalid decode request count\n");
                return 1;
            }
        } else if (strcmp(arg, "--decode-steps") == 0 || strcmp(arg, "-n") == 0) {
            if (!parse_next(decode_steps)) {
                fprintf(stderr, "error: invalid decode steps\n");
                return 1;
            }
        } else if (strcmp(arg, "--validation-steps") == 0) {
            if (!parse_next(validation_steps)) {
                fprintf(stderr, "error: invalid validation steps\n");
                return 1;
            }
        } else if (strcmp(arg, "--wavefront-layers") == 0) {
            if (!parse_next(wavefront_layers)) {
                fprintf(stderr, "error: invalid wavefront layer count\n");
                return 1;
            }
            wavefront_layers_set = true;
        } else if (strcmp(arg, "-ngl") == 0) {
            if (!parse_next(n_gpu_layers)) {
                fprintf(stderr, "error: invalid -ngl\n");
                return 1;
            }
        } else if (strcmp(arg, "--no-warmup") == 0) {
            warmup = false;
        } else if (strcmp(arg, "--placement-audit") == 0) {
            placement_audit = true;
        } else if (strcmp(arg, "--single-device-control") == 0) {
            single_device_control = true;
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            hetero_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown or incomplete argument '%s'\n", arg);
            hetero_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || ubatch < 2 || prompt_len < ubatch ||
        decode_ctx < 1 || decode_requests < 1 || decode_requests >= ubatch ||
        decode_requests + 1 > LLAMA_MAX_SEQ || decode_steps < 1 || validation_steps < 1 ||
        wavefront_layers < 0 || n_gpu_layers < 0) {
        fprintf(stderr,
            "error: require a model, ubatch >= 2, prompt-len >= ubatch, decode-ctx >= 1, "
            "1 <= decode-requests < ubatch, decode-steps >= 1, validation-steps >= 1, "
            "and wavefront-layers >= 0\n");
        hetero_usage(argv[0]);
        return 1;
    }

    const char * layer_start = getenv("LLAMA_LAYER_START");
    if (layer_start != nullptr && atoi(layer_start) != 0) {
        fprintf(stderr, "error: hetero ubatch requires a full model (LLAMA_LAYER_START must be zero)\n");
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();

    if (single_device_control) {
        fprintf(stderr,
            "[hetero] loading one full model copy: device=%s ubatch=%d prompt=%d decode_ctx=%d decode_requests=%d\n",
            gpu_device.c_str(), ubatch, prompt_len, decode_ctx, decode_requests);
    } else {
        fprintf(stderr,
            "[hetero] loading two full model copies: gpu=%s npu=%s ubatch=%d prompt=%d decode_ctx=%d decode_requests=%d max_decode_steps=%d wavefront_layers=%d\n",
            gpu_device.c_str(), npu_device.c_str(), ubatch, prompt_len, decode_ctx,
            decode_requests, decode_steps, wavefront_layers);
    }

    hetero_model_owner gpu_model;
    gpu_model.ptr = hetero_load_model(model_path, gpu_device, n_gpu_layers);
    if (gpu_model.ptr == nullptr) {
        return 2;
    }
    hetero_model_owner npu_model;
    if (!single_device_control) {
        npu_model.ptr = hetero_load_model(model_path, npu_device, n_gpu_layers);
        if (npu_model.ptr == nullptr) {
            return 2;
        }
    }

    const int n_layer = (int) llama_model_n_layer(gpu_model.ptr);
    const bool wavefront_enabled = !single_device_control && wavefront_layers > 0;
    if (wavefront_enabled && wavefront_layers > n_layer) {
        if (wavefront_layers_set) {
            fprintf(stderr, "error: wavefront tile %d exceeds model layer count %d\n",
                    wavefront_layers, n_layer);
            return 1;
        }
        wavefront_layers = n_layer;
        fprintf(stderr, "[hetero] clamping wavefront tile to model layer count %d\n", n_layer);
    }
    if (wavefront_enabled) {
        char architecture[64] = {};
        if (llama_model_meta_val_str(
                    gpu_model.ptr, "general.architecture",
                    architecture, sizeof(architecture)) < 0 ||
            strcmp(architecture, "qwen2") != 0) {
            fprintf(stderr,
                "error: experimental wavefront currently requires general.architecture=qwen2 (found '%s')\n",
                architecture);
            return 1;
        }
    }
    const char * layer_end = getenv("LLAMA_LAYER_END");
    if (layer_end != nullptr && atoi(layer_end) != n_layer) {
        fprintf(stderr, "error: hetero ubatch requires a full model (LLAMA_LAYER_END=%s, model layers=%d)\n",
                layer_end, n_layer);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(gpu_model.ptr);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0 ||
        (!single_device_control && llama_vocab_n_tokens(llama_model_get_vocab(npu_model.ptr)) != n_vocab)) {
        fprintf(stderr, "error: model vocabulary mismatch\n");
        return 2;
    }

    hetero_placement_tally gpu_placement;
    hetero_placement_tally npu_placement;
    hetero_wavefront wavefront;
    hetero_eval_state gpu_eval = {
        placement_audit ? &gpu_placement : nullptr,
        wavefront_enabled ? &wavefront : nullptr,
        true,
    };
    hetero_eval_state npu_eval = {
        placement_audit ? &npu_placement : nullptr,
        wavefront_enabled ? &wavefront : nullptr,
        false,
    };
    const llama_seq_id prefill_seq_id = (llama_seq_id) decode_requests;
    const uint32_t context_size = (uint32_t) (
            decode_requests * (decode_ctx + decode_steps + 2) +
            prompt_len + validation_steps + 8);
    const uint32_t batch_size = (uint32_t) std::max(
            decode_requests * decode_ctx, prompt_len + decode_requests);
    llama_context_params context_params = llama_context_default_params();
    context_params.n_seq_max = (uint32_t) decode_requests + 1;
    context_params.n_ctx = context_size;
    context_params.n_batch = batch_size;
    context_params.n_ubatch = (uint32_t) ubatch;
    context_params.n_outputs_max = (uint32_t) decode_requests + 1;
    context_params.kv_unified = true;
    context_params.no_perf = true;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (placement_audit || wavefront_enabled) {
        context_params.cb_eval = hetero_eval_cb;
        context_params.cb_eval_user_data = &gpu_eval;
    }

    hetero_context_owner gpu_context;
    gpu_context.ptr = llama_init_from_model(gpu_model.ptr, context_params);
    hetero_context_owner npu_context;
    if (!single_device_control) {
        if (placement_audit || wavefront_enabled) {
            context_params.cb_eval_user_data = &npu_eval;
        }
        npu_context.ptr = llama_init_from_model(npu_model.ptr, context_params);
    }
    if (gpu_context.ptr == nullptr || (!single_device_control && npu_context.ptr == nullptr)) {
        fprintf(stderr, "error: failed to create one or both contexts\n");
        return 2;
    }

    if ((int) llama_n_ubatch(gpu_context.ptr) != ubatch ||
        (!single_device_control && (int) llama_n_ubatch(npu_context.ptr) != ubatch)) {
        fprintf(stderr, "error: requested ubatch %d was not retained\n", ubatch);
        return 2;
    }

    const llama_token bos = llama_vocab_bos(vocab);
    std::vector<std::vector<llama_token>> a_prompts(
            (size_t) decode_requests, std::vector<llama_token>((size_t) decode_ctx));
    std::vector<llama_token> b_prompt((size_t) prompt_len);
    for (int seq = 0; seq < decode_requests; ++seq) {
        for (int i = 0; i < decode_ctx; ++i) {
            a_prompts[(size_t) seq][(size_t) i] =
                hetero_token_at(i, 7 + seq * 17, n_vocab, bos);
        }
    }
    for (int i = 0; i < prompt_len; ++i) {
        b_prompt[(size_t) i] = hetero_token_at(i, 41, n_vocab, bos);
    }

    std::vector<hetero_row> seed_rows;
    seed_rows.reserve((size_t) decode_requests * decode_ctx);
    for (int seq = 0; seq < decode_requests; ++seq) {
        for (int i = 0; i < decode_ctx; ++i) {
            seed_rows.push_back({
                a_prompts[(size_t) seq][(size_t) i], i,
                (llama_seq_id) seq, i == decode_ctx - 1,
            });
        }
    }
    std::vector<std::vector<float>> seed_logits;
    double seed_ms = 0.0;
    if (!hetero_decode(gpu_context.ptr, seed_rows, n_vocab, seed_logits, seed_ms)) {
        return 3;
    }
    std::vector<llama_token> a_inputs((size_t) decode_requests);
    std::vector<hetero_state_blob> a_seed_states((size_t) decode_requests);
    size_t seed_state_bytes = 0;
    double seed_state_export_ms = 0.0;
    for (int seq = 0; seq < decode_requests; ++seq) {
        const size_t output_row = (size_t) (seq + 1) * decode_ctx - 1;
        a_inputs[(size_t) seq] = hetero_argmax(seed_logits[output_row]);
        if (!hetero_export_state(gpu_context.ptr, (llama_seq_id) seq, a_seed_states[(size_t) seq])) {
            return 3;
        }
        seed_state_bytes += a_seed_states[(size_t) seq].data.size();
        seed_state_export_ms += a_seed_states[(size_t) seq].export_ms;
    }

    std::vector<hetero_row> control_rows;
    control_rows.reserve((size_t) prompt_len + decode_requests);
    for (int seq = 0; seq < decode_requests; ++seq) {
        control_rows.push_back({a_inputs[(size_t) seq], decode_ctx, (llama_seq_id) seq, true});
    }
    for (int i = 0; i < prompt_len; ++i) {
        control_rows.push_back({b_prompt[(size_t) i], i, prefill_seq_id, i == prompt_len - 1});
    }

    if (single_device_control) {
        double warmup_ms = 0.0;
        double restore_ms = 0.0;
        if (warmup) {
            std::vector<std::vector<float>> warmup_logits;
            if (!hetero_decode(gpu_context.ptr, control_rows, n_vocab, warmup_logits, warmup_ms)) {
                return 3;
            }
            llama_memory_clear(llama_get_memory(gpu_context.ptr), true);
            if (!hetero_restore_states(gpu_context.ptr, a_seed_states, restore_ms)) {
                return 3;
            }
        }

        std::vector<std::vector<float>> control_logits;
        double control_ms = 0.0;
        hetero_decode_timing control_timing;
        if (!hetero_decode(
                    gpu_context.ptr, control_rows, n_vocab, control_logits,
                    control_ms, true, &control_timing)) {
            return 3;
        }
        const double decode_host_start_ms = hetero_now_ms();
        std::vector<llama_token> decode_tokens((size_t) decode_requests);
        for (int seq = 0; seq < decode_requests; ++seq) {
            decode_tokens[(size_t) seq] = hetero_argmax(control_logits[(size_t) seq]);
        }
        const double decode_host_ms = hetero_now_ms() - decode_host_start_ms;
        const double decode_tokens_ready_ms = control_ms + decode_host_ms;
        const llama_token prefill_token = hetero_argmax(control_logits.back());
        const bool logits_finite = prefill_token >= 0 && std::all_of(
            decode_tokens.begin(), decode_tokens.end(), [](llama_token token) { return token >= 0; });
        const std::string expected_buffer = hetero_expected_buffer(gpu_device);
        const double placement_fraction = hetero_placement_fraction(gpu_placement, expected_buffer);
        const int64_t unexpected_compute_nodes =
            hetero_unexpected_compute_nodes(gpu_placement, expected_buffer);
        const bool placement_ok = !placement_audit ||
            hetero_placement_certified(gpu_placement, expected_buffer);
        const std::string status = !placement_ok ? "placement_fallback" :
            (logits_finite ? "ok" : "nonfinite_logits");

        nlohmann::ordered_json result = {
            {"schema", "hetero-single-device-control-v3-strict"},
            {"status", status},
            {"model", model_path},
            {"device", gpu_device},
            {"n_layer", n_layer},
            {"n_vocab", n_vocab},
            {"ubatch", ubatch},
            {"logical_batch_rows", (int) control_rows.size()},
            {"physical_ubatches", ((int) control_rows.size() + ubatch - 1) / ubatch},
            {"decode_rows", decode_requests},
            {"prefill_rows", prompt_len},
            {"decode_ctx", decode_ctx},
            {"validation_steps", validation_steps},
            {"kv_unified", true},
            {"control", {
                {"mixed_full_ms", control_ms},
                {"backend_ms", control_timing.backend_ms},
                {"output_copy_ms", control_timing.output_copy_ms},
                {"decode_host_ms", decode_host_ms},
                {"decode_tokens_ready_ms", decode_tokens_ready_ms},
                {"decode_token", decode_tokens.front()},
                {"decode_tokens", decode_tokens},
                {"prefill_token", prefill_token},
            }},
            {"setup", {
                {"warmup_enabled", warmup},
                {"placement_audit", placement_audit},
                {"warmup_ms", warmup_ms},
                {"seed_ms", seed_ms},
                {"seed_state_bytes", seed_state_bytes},
                {"seed_state_export_ms", seed_state_export_ms},
                {"seed_state_restore_ms", restore_ms},
            }},
            {"placement", {
                {"evidence", placement_audit ? "strict_scheduled_graph_node_buffers" : "disabled"},
                {"enabled", placement_audit},
                {"valid", placement_audit ? nlohmann::ordered_json(placement_ok) : nlohmann::ordered_json(nullptr)},
                {"all_compute_nodes_must_use_requested_buffer", true},
                {"expected_buffer", expected_buffer},
                {"target_fraction", placement_fraction},
                {"unexpected_compute_nodes", unexpected_compute_nodes},
                {"unexpected_compute", hetero_unexpected_compute_json(gpu_placement, expected_buffer)},
                {"graph", hetero_placement_json(gpu_placement)},
            }},
        };

        fprintf(stdout, "HETEROJSON %s\n", result.dump(-1, ' ', true).c_str());
        nlohmann::ordered_json summary = {
            {"mode", "single_device"},
            {"device", gpu_device},
            {"status", status},
            {"mixed_full_ms", control_ms},
            {"backend_ms", control_timing.backend_ms},
            {"output_copy_ms", control_timing.output_copy_ms},
            {"decode_tokens_ready_ms", decode_tokens_ready_ms},
            {"decode_requests", decode_requests},
            {"decode_tokens", decode_tokens},
            {"prefill_token", prefill_token},
            {"logits_finite", logits_finite},
        };
        fprintf(stdout, "HETEROSUMMARY %s\n", summary.dump(-1, ' ', true).c_str());
        fflush(stdout);
        fprintf(stderr, "[hetero] single-device mixed batch %.2f ms on %s; status=%s\n",
                control_ms, gpu_device.c_str(), status.c_str());
        return hetero_result_code(placement_audit, placement_ok, logits_finite);
    }

    const int prefix_len = ubatch - decode_requests;
    const int npu_remainder_rows = prompt_len - prefix_len;
    const int final_npu_rows = 1 + (npu_remainder_rows - 1) % ubatch;
    const int final_npu_begin = prompt_len - final_npu_rows;
    const int npu_physical_ubatches = (npu_remainder_rows + ubatch - 1) / ubatch;

    double gpu_warmup_ms = 0.0;
    double npu_warmup_ms = 0.0;
    if (warmup) {
        std::vector<std::vector<float>> warmup_logits;
        if (!hetero_decode(gpu_context.ptr, control_rows, n_vocab, warmup_logits, gpu_warmup_ms)) {
            return 3;
        }
        if (!llama_memory_seq_rm(
                    llama_get_memory(gpu_context.ptr), prefill_seq_id, prefix_len, -1)) {
            fprintf(stderr, "error: failed to trim warmup prefill sequence\n");
            return 3;
        }
        hetero_state_blob warmup_prefix;
        if (!hetero_export_state(gpu_context.ptr, prefill_seq_id, warmup_prefix)) {
            return 3;
        }
        double warmup_prefix_import_ms = 0.0;
        if (!hetero_import_state(
                    npu_context.ptr, prefill_seq_id, warmup_prefix, warmup_prefix_import_ms)) {
            return 3;
        }
        llama_memory_clear(llama_get_memory(gpu_context.ptr), true);

        const double npu_warmup_start_ms = hetero_now_ms();
        if (prefix_len < final_npu_begin) {
            std::vector<hetero_row> rows;
            rows.reserve((size_t) (final_npu_begin - prefix_len));
            for (int i = prefix_len; i < final_npu_begin; ++i) {
                rows.push_back({b_prompt[(size_t) i], i, prefill_seq_id, false});
            }
            std::vector<std::vector<float>> grouped_logits;
            double submit_ms = 0.0;
            if (!hetero_decode(npu_context.ptr, rows, n_vocab, grouped_logits, submit_ms, false)) {
                return 3;
            }
            llama_synchronize(npu_context.ptr);
        }
        std::vector<hetero_row> final_rows;
        final_rows.reserve((size_t) final_npu_rows);
        for (int i = final_npu_begin; i < prompt_len; ++i) {
            final_rows.push_back({
                b_prompt[(size_t) i], i, prefill_seq_id, i == prompt_len - 1,
            });
        }
        std::vector<std::vector<float>> final_logits;
        double final_ms = 0.0;
        if (!hetero_decode(npu_context.ptr, final_rows, n_vocab, final_logits, final_ms)) {
            return 3;
        }
        npu_warmup_ms = hetero_now_ms() - npu_warmup_start_ms;

        hetero_state_blob warmup_full;
        if (!hetero_export_state(npu_context.ptr, prefill_seq_id, warmup_full)) {
            return 3;
        }
        double warmup_full_import_ms = 0.0;
        if (!hetero_import_state(
                    gpu_context.ptr, prefill_seq_id, warmup_full, warmup_full_import_ms)) {
            return 3;
        }
        llama_memory_clear(llama_get_memory(gpu_context.ptr), true);
        llama_memory_clear(llama_get_memory(npu_context.ptr), true);
        double warmup_restore_ms = 0.0;
        if (!hetero_restore_states(gpu_context.ptr, a_seed_states, warmup_restore_ms)) {
            return 3;
        }
    }

    std::vector<std::vector<float>> control_logits;
    double control_ms = 0.0;
    hetero_decode_timing control_timing;
    if (!hetero_decode(
                gpu_context.ptr, control_rows, n_vocab, control_logits,
                control_ms, true, &control_timing)) {
        return 3;
    }
    std::vector<std::vector<float>> control_a_logits((size_t) decode_requests);
    std::vector<llama_token> control_a_tokens((size_t) decode_requests);
    const double control_a_host_start_ms = hetero_now_ms();
    for (int seq = 0; seq < decode_requests; ++seq) {
        control_a_logits[(size_t) seq] = std::move(control_logits[(size_t) seq]);
        control_a_tokens[(size_t) seq] = hetero_argmax(control_a_logits[(size_t) seq]);
    }
    const double control_a_host_ms = hetero_now_ms() - control_a_host_start_ms;
    const double control_decode_ready_ms = control_ms + control_a_host_ms;
    const std::vector<float> control_b_logits = std::move(control_logits.back());
    const llama_token control_b_token = hetero_argmax(control_b_logits);

    llama_memory_clear(llama_get_memory(gpu_context.ptr), true);
    llama_memory_clear(llama_get_memory(npu_context.ptr), true);
    double a_restore_ms = 0.0;
    if (!hetero_restore_states(gpu_context.ptr, a_seed_states, a_restore_ms)) {
        return 3;
    }

    std::vector<hetero_row> first_ubatch_rows;
    first_ubatch_rows.reserve((size_t) ubatch);
    for (int seq = 0; seq < decode_requests; ++seq) {
        first_ubatch_rows.push_back({
            a_inputs[(size_t) seq], decode_ctx, (llama_seq_id) seq, true,
        });
    }
    for (int i = 0; i < prefix_len; ++i) {
        first_ubatch_rows.push_back({b_prompt[(size_t) i], i, prefill_seq_id, false});
    }

    if (wavefront_enabled && !llama_kv_cache_wavefront_reserve(
                npu_context.ptr, prefill_seq_id, 0, prefix_len)) {
        fprintf(stderr, "error: failed to reserve NPU prefix KV cells for wavefront\n");
        return 3;
    }

    enum : uint32_t {
        HETERO_NPU_READY = 1u << 0,
        HETERO_START = 1u << 1,
        HETERO_GPU_U0_DONE = 1u << 2,
        HETERO_GPU_PREFIX_EXPORTED = 1u << 3,
        HETERO_NPU_FINAL_UBATCH = 1u << 4,
        HETERO_NPU_DONE = 1u << 5,
        HETERO_GPU_DONE = 1u << 6,
        HETERO_ABORT = 1u << 7,
    };
    std::atomic<uint32_t> phase_bits {0};
    bool npu_ok = false;
    bool handoff_ok = false;
    double npu_remainder_ms = 0.0;
    double npu_done_ms = 0.0;
    double npu_start_ms = 0.0;
    double handoff_start_ms = 0.0;
    double gpu_prefix_export_done_ms = 0.0;
    double handoff_done_ms = 0.0;
    std::vector<double> npu_submission_ms;
    std::vector<std::vector<float>> npu_logits;
    hetero_state_blob gpu_to_npu;
    double gpu_to_npu_import_ms = 0.0;
    double treatment_start_ms = 0.0;

    std::thread npu_worker([&] {
        phase_bits.fetch_or(HETERO_NPU_READY, std::memory_order_release);
        while ((phase_bits.load(std::memory_order_acquire) & (HETERO_START | HETERO_ABORT)) == 0) {
            std::this_thread::yield();
        }
        if ((phase_bits.load(std::memory_order_acquire) & HETERO_ABORT) != 0) {
            phase_bits.fetch_or(HETERO_NPU_DONE, std::memory_order_release);
            return;
        }

        if (!wavefront_enabled) {
            while ((phase_bits.load(std::memory_order_acquire) &
                    (HETERO_GPU_U0_DONE | HETERO_ABORT)) == 0) {
                std::this_thread::yield();
            }
            if ((phase_bits.load(std::memory_order_acquire) & HETERO_ABORT) != 0) {
                phase_bits.fetch_or(HETERO_NPU_DONE, std::memory_order_release);
                return;
            }

            handoff_start_ms = hetero_now_ms();
            if (!hetero_export_state(gpu_context.ptr, prefill_seq_id, gpu_to_npu)) {
                phase_bits.fetch_or(
                    HETERO_GPU_PREFIX_EXPORTED | HETERO_NPU_DONE | HETERO_ABORT,
                    std::memory_order_release);
                return;
            }
            gpu_prefix_export_done_ms = hetero_now_ms();
            phase_bits.fetch_or(HETERO_GPU_PREFIX_EXPORTED, std::memory_order_release);

            if (!hetero_import_state(
                        npu_context.ptr, prefill_seq_id, gpu_to_npu, gpu_to_npu_import_ms)) {
                phase_bits.fetch_or(HETERO_NPU_DONE | HETERO_ABORT, std::memory_order_release);
                return;
            }
            handoff_done_ms = hetero_now_ms();
            handoff_ok = true;
        }
        npu_start_ms = hetero_now_ms();
        const double remainder_start_ms = hetero_now_ms();
        npu_ok = true;
        if (prefix_len < final_npu_begin) {
            std::vector<hetero_row> rows;
            rows.reserve((size_t) (final_npu_begin - prefix_len));
            for (int i = prefix_len; i < final_npu_begin; ++i) {
                rows.push_back({b_prompt[(size_t) i], i, prefill_seq_id, false});
            }
            std::vector<std::vector<float>> grouped_logits;
            double submit_ms = 0.0;
            const double grouped_start_ms = hetero_now_ms();
            if (!hetero_decode(npu_context.ptr, rows, n_vocab, grouped_logits, submit_ms, false)) {
                npu_ok = false;
            } else {
                llama_synchronize(npu_context.ptr);
                npu_submission_ms.push_back(hetero_now_ms() - grouped_start_ms);
            }
            if (wavefront_enabled && wavefront.abort.load(std::memory_order_acquire)) {
                npu_ok = false;
            }
        }
        if (npu_ok) {
            phase_bits.fetch_or(HETERO_NPU_FINAL_UBATCH, std::memory_order_release);
            std::vector<hetero_row> final_rows;
            final_rows.reserve((size_t) final_npu_rows);
            for (int i = final_npu_begin; i < prompt_len; ++i) {
                final_rows.push_back({
                    b_prompt[(size_t) i], i, prefill_seq_id, i == prompt_len - 1,
                });
            }
            double final_ms = 0.0;
            if (!hetero_decode(npu_context.ptr, final_rows, n_vocab, npu_logits, final_ms)) {
                npu_ok = false;
            } else {
                npu_submission_ms.push_back(final_ms);
            }
            if (wavefront_enabled && wavefront.abort.load(std::memory_order_acquire)) {
                npu_ok = false;
            }
        }
        npu_remainder_ms = hetero_now_ms() - remainder_start_ms;
        npu_done_ms = hetero_now_ms();
        if (wavefront_enabled && !npu_ok) {
            wavefront.fail(4);
            phase_bits.fetch_or(HETERO_ABORT, std::memory_order_release);
        }
        phase_bits.fetch_or(HETERO_NPU_DONE, std::memory_order_release);
    });

    auto abort_workers = [&] {
        phase_bits.fetch_or(HETERO_ABORT, std::memory_order_release);
        if (npu_worker.joinable()) {
            npu_worker.join();
        }
    };

    const double ready_deadline_ms = hetero_now_ms() + 30000.0;
    while ((phase_bits.load(std::memory_order_acquire) & HETERO_NPU_READY) == 0 &&
           hetero_now_ms() < ready_deadline_ms) {
        std::this_thread::yield();
    }
    if ((phase_bits.load(std::memory_order_acquire) & HETERO_NPU_READY) == 0) {
        fprintf(stderr, "error: heterogeneous NPU worker did not become ready\n");
        abort_workers();
        return 3;
    }

    treatment_start_ms = hetero_now_ms();
    if (wavefront_enabled) {
        wavefront.reset(
            n_layer, wavefront_layers, gpu_context.ptr, npu_context.ptr,
            prefill_seq_id, 0, prefix_len, treatment_start_ms);
    }
    phase_bits.fetch_or(HETERO_START, std::memory_order_release);
    std::vector<std::vector<float>> first_ubatch_logits;
    double first_ubatch_ms = 0.0;
    hetero_decode_timing first_ubatch_timing;
    if (!hetero_decode(
                gpu_context.ptr, first_ubatch_rows, n_vocab, first_ubatch_logits,
                first_ubatch_ms, true, &first_ubatch_timing)) {
        if (wavefront_enabled) {
            wavefront.fail(5);
        }
        abort_workers();
        return 3;
    }
    if (wavefront_enabled && wavefront.abort.load(std::memory_order_acquire)) {
        abort_workers();
        fprintf(stderr, "error: wavefront callback failed (code=%d)\n",
                wavefront.error_code.load(std::memory_order_relaxed));
        return 3;
    }
    const double decode_logits_ready_at_ms = hetero_now_ms();
    const double decode_backend_ready_at_ms = first_ubatch_timing.backend_done_at_ms;
    phase_bits.fetch_or(HETERO_GPU_U0_DONE, std::memory_order_release);
    if (wavefront_enabled) {
        handoff_ok = true;
        phase_bits.fetch_or(HETERO_GPU_PREFIX_EXPORTED, std::memory_order_release);
    }

    const double a_host_start_ms = hetero_now_ms();
    std::vector<std::vector<float>> treatment_a_logits((size_t) decode_requests);
    std::vector<llama_token> treatment_a_tokens((size_t) decode_requests);
    for (int seq = 0; seq < decode_requests; ++seq) {
        treatment_a_logits[(size_t) seq] = std::move(first_ubatch_logits[(size_t) seq]);
        treatment_a_tokens[(size_t) seq] = hetero_argmax(treatment_a_logits[(size_t) seq]);
    }
    const double a_host_done_ms = hetero_now_ms();
    const double a_host_ms = a_host_done_ms - a_host_start_ms;

    while ((phase_bits.load(std::memory_order_acquire) &
            (HETERO_GPU_PREFIX_EXPORTED | HETERO_ABORT)) == 0) {
        std::this_thread::yield();
    }
    if ((phase_bits.load(std::memory_order_acquire) & HETERO_ABORT) != 0) {
        abort_workers();
        return 3;
    }

    std::vector<llama_token> gpu_next = treatment_a_tokens;
    std::vector<double> gpu_step_ms;
    std::vector<double> gpu_step_done_ms;
    bool gpu_ok = true;
    bool gpu_paused_for_npu_final = false;
    const double gpu_continuation_start_ms = hetero_now_ms();
    for (int step = 0; step < decode_steps; ++step) {
        const uint32_t bits = phase_bits.load(std::memory_order_acquire);
        if ((bits & HETERO_ABORT) != 0) {
            gpu_ok = false;
            break;
        }
        if ((bits & (HETERO_NPU_FINAL_UBATCH | HETERO_NPU_DONE)) != 0) {
            gpu_paused_for_npu_final = (bits & HETERO_NPU_DONE) == 0;
            break;
        }
        std::vector<hetero_row> step_rows;
        step_rows.reserve((size_t) decode_requests);
        for (int seq = 0; seq < decode_requests; ++seq) {
            step_rows.push_back({
                gpu_next[(size_t) seq], decode_ctx + 1 + step, (llama_seq_id) seq, true,
            });
        }
        std::vector<std::vector<float>> step_logits;
        double step_ms = 0.0;
        if (!hetero_decode(gpu_context.ptr, step_rows, n_vocab, step_logits, step_ms)) {
            gpu_ok = false;
            break;
        }
        for (int seq = 0; seq < decode_requests; ++seq) {
            gpu_next[(size_t) seq] = hetero_argmax(step_logits[(size_t) seq]);
        }
        gpu_step_ms.push_back(step_ms);
        gpu_step_done_ms.push_back(hetero_now_ms());
    }
    phase_bits.fetch_or(HETERO_GPU_DONE, std::memory_order_release);
    npu_worker.join();
    if (wavefront_enabled) {
        wavefront.active.store(false, std::memory_order_release);
    }
    if (!gpu_ok || !handoff_ok || !npu_ok ||
        (wavefront_enabled && wavefront.abort.load(std::memory_order_acquire))) {
        fprintf(stderr, "error: prefix handoff, concurrent GPU decode, or NPU prefill failed\n");
        return 3;
    }

    const llama_token npu_b_token = hetero_argmax(npu_logits.back());
    int gpu_steps_before_npu_done = 0;
    for (double step_done_ms : gpu_step_done_ms) {
        if (step_done_ms <= npu_done_ms) {
            ++gpu_steps_before_npu_done;
        }
    }

    const double npu_to_gpu_start_ms = hetero_now_ms();
    hetero_state_blob npu_to_gpu;
    if (!hetero_export_state(npu_context.ptr, prefill_seq_id, npu_to_gpu)) {
        return 3;
    }
    double npu_to_gpu_import_ms = 0.0;
    if (!hetero_import_state(
                gpu_context.ptr, prefill_seq_id, npu_to_gpu, npu_to_gpu_import_ms)) {
        return 3;
    }
    const double b_gpu_ready_ms = hetero_now_ms();

    const bool same_backend = gpu_device == npu_device;
    const double cross_backend_relative_l2_limit = same_backend ?
        HETERO_SAME_BACKEND_RELATIVE_L2_LIMIT : HETERO_CROSS_BACKEND_RELATIVE_L2_LIMIT;
    const hetero_diff b_prefill_diff = hetero_logits_diff(control_b_logits, npu_logits.back());
    const bool b_prefill_match = control_b_token == npu_b_token;
    const bool b_prefill_numeric_ok = hetero_diff_within(
        b_prefill_diff, cross_backend_relative_l2_limit);

    nlohmann::ordered_json continuation_checks = nlohmann::ordered_json::array();
    std::vector<llama_token> npu_continuation_tokens;
    std::vector<llama_token> gpu_continuation_tokens;
    std::vector<double> continuation_relative_l2;
    int continuation_match_count = 0;
    int continuation_numeric_count = 0;
    size_t continuation_nonfinite_values = 0;
    double continuation_relative_l2_max = 0.0;
    double continuation_max_abs = 0.0;
    double npu_next_ms = 0.0;
    double gpu_next_ms = 0.0;
    llama_token continuation_input = npu_b_token;
    for (int step = 0; step < validation_steps; ++step) {
        if (continuation_input < 0) {
            break;
        }
        std::vector<hetero_row> continuation_row = {
            {continuation_input, prompt_len + step, prefill_seq_id, true},
        };
        std::vector<std::vector<float>> npu_step_logits;
        std::vector<std::vector<float>> gpu_step_logits;
        double npu_step_ms = 0.0;
        double gpu_step_ms = 0.0;
        if (!hetero_decode(
                    npu_context.ptr, continuation_row, n_vocab, npu_step_logits, npu_step_ms) ||
            !hetero_decode(
                    gpu_context.ptr, continuation_row, n_vocab, gpu_step_logits, gpu_step_ms)) {
            return 3;
        }
        npu_next_ms += npu_step_ms;
        gpu_next_ms += gpu_step_ms;

        const llama_token npu_token = hetero_argmax(npu_step_logits.front());
        const llama_token gpu_token = hetero_argmax(gpu_step_logits.front());
        const hetero_diff diff = hetero_logits_diff(
            npu_step_logits.front(), gpu_step_logits.front());
        const bool argmax_match = npu_token == gpu_token && npu_token >= 0;
        const bool numeric_ok = hetero_diff_within(diff, cross_backend_relative_l2_limit);
        continuation_match_count += argmax_match ? 1 : 0;
        continuation_numeric_count += numeric_ok ? 1 : 0;
        continuation_nonfinite_values += diff.nonfinite_values;
        continuation_relative_l2.push_back(diff.relative_l2);
        continuation_relative_l2_max = std::max(
            continuation_relative_l2_max, diff.relative_l2);
        continuation_max_abs = std::max(continuation_max_abs, diff.max_abs);
        npu_continuation_tokens.push_back(npu_token);
        gpu_continuation_tokens.push_back(gpu_token);
        continuation_checks.push_back({
            {"step", step},
            {"position", prompt_len + step},
            {"input_token", continuation_input},
            {"npu_token", npu_token},
            {"gpu_token", gpu_token},
            {"argmax_match", argmax_match},
            {"relative_l2", diff.relative_l2},
            {"relative_l2_within_limit", numeric_ok},
            {"max_abs", diff.max_abs},
            {"nonfinite_values", diff.nonfinite_values},
        });
        continuation_input = npu_token;
    }

    const bool b_continue_match = continuation_match_count == validation_steps;
    const bool b_continue_numeric_ok = continuation_numeric_count == validation_steps;
    const llama_token npu_next_token = npu_continuation_tokens.empty() ?
        -1 : npu_continuation_tokens.front();
    const llama_token gpu_next_token = gpu_continuation_tokens.empty() ?
        -1 : gpu_continuation_tokens.front();
    const double continuation_relative_l2_p50 = hetero_median(continuation_relative_l2);

    nlohmann::ordered_json decode_checks = nlohmann::ordered_json::array();
    std::vector<double> decode_relative_l2;
    decode_relative_l2.reserve((size_t) decode_requests);
    int decode_match_count = 0;
    int decode_numeric_count = 0;
    size_t decode_nonfinite_values = 0;
    double decode_relative_l2_max = 0.0;
    double decode_max_abs = 0.0;
    for (int seq = 0; seq < decode_requests; ++seq) {
        const hetero_diff diff = hetero_logits_diff(
            control_a_logits[(size_t) seq], treatment_a_logits[(size_t) seq]);
        const bool match = control_a_tokens[(size_t) seq] == treatment_a_tokens[(size_t) seq];
        const bool numeric_ok = hetero_diff_within(
            diff, HETERO_SAME_BACKEND_RELATIVE_L2_LIMIT);
        decode_match_count += match ? 1 : 0;
        decode_numeric_count += numeric_ok ? 1 : 0;
        decode_nonfinite_values += diff.nonfinite_values;
        decode_relative_l2.push_back(diff.relative_l2);
        decode_relative_l2_max = std::max(decode_relative_l2_max, diff.relative_l2);
        decode_max_abs = std::max(decode_max_abs, diff.max_abs);
        decode_checks.push_back({
            {"sequence", seq},
            {"control_token", control_a_tokens[(size_t) seq]},
            {"treatment_token", treatment_a_tokens[(size_t) seq]},
            {"argmax_match", match},
            {"relative_l2", diff.relative_l2},
            {"relative_l2_within_limit", numeric_ok},
            {"max_abs", diff.max_abs},
            {"nonfinite_values", diff.nonfinite_values},
        });
    }
    const bool a_match = decode_match_count == decode_requests;
    const bool a_numeric_ok = decode_numeric_count == decode_requests;
    const double decode_relative_l2_p50 = hetero_median(decode_relative_l2);

    const double gpu_step_p50 = hetero_median(gpu_step_ms);
    const double gpu_step_max = gpu_step_ms.empty() ? 0.0 :
        *std::max_element(gpu_step_ms.begin(), gpu_step_ms.end());

    nlohmann::ordered_json wavefront_tiles = nlohmann::ordered_json::array();
    size_t wavefront_bytes = 0;
    double wavefront_transfer_ms = 0.0;
    double wavefront_gpu_wait_ms = 0.0;
    double wavefront_npu_wait_ms = 0.0;
    double wavefront_compute_overlap_ms = 0.0;
    if (wavefront_enabled) {
        for (int tile = 0; tile < wavefront.n_tiles; ++tile) {
            const size_t i = (size_t) tile;
            const int il0 = tile * wavefront_layers;
            const int il1 = std::min(n_layer, il0 + wavefront_layers);
            const double transfer_ms =
                wavefront.trace.transfer_done_ms[i] - wavefront.trace.transfer_start_ms[i];
            const double npu_wait_ms =
                wavefront.trace.npu_tile_release_ms[i] - wavefront.trace.npu_wait_start_ms[i];
            const double gpu_compute_start_ms = tile == 0 ? 0.0 :
                wavefront.trace.gpu_tile_release_ms[i - 1];
            const double gpu_compute_done_ms = wavefront.trace.gpu_tile_done_ms[i];
            const double npu_compute_start_ms = wavefront.trace.npu_tile_release_ms[i];
            const double npu_compute_done_ms = tile + 1 < wavefront.n_tiles ?
                wavefront.trace.npu_wait_start_ms[i + 1] :
                wavefront.trace.npu_first_ubatch_done_ms;
            double overlap_ms = 0.0;
            for (int gpu_tile = 0; gpu_tile < wavefront.n_tiles; ++gpu_tile) {
                const size_t gi = (size_t) gpu_tile;
                const double gpu_start_ms = gpu_tile == 0 ? 0.0 :
                    wavefront.trace.gpu_tile_release_ms[gi - 1];
                const double gpu_done_ms = wavefront.trace.gpu_tile_done_ms[gi];
                overlap_ms += std::max(0.0,
                    std::min(gpu_done_ms, npu_compute_done_ms) -
                    std::max(gpu_start_ms, npu_compute_start_ms));
            }

            wavefront_bytes += wavefront.trace.transfer_bytes[i];
            wavefront_transfer_ms += transfer_ms;
            wavefront_gpu_wait_ms += wavefront.trace.gpu_wait_npu_ms[i];
            wavefront_npu_wait_ms += npu_wait_ms;
            wavefront_compute_overlap_ms += overlap_ms;

            wavefront_tiles.push_back({
                {"tile", tile},
                {"layer_begin", il0},
                {"layer_end", il1},
                {"gpu_compute_start_ms", gpu_compute_start_ms},
                {"gpu_tile_done_ms", gpu_compute_done_ms},
                {"gpu_wait_npu_ms", wavefront.trace.gpu_wait_npu_ms[i]},
                {"transfer_start_ms", wavefront.trace.transfer_start_ms[i]},
                {"transfer_done_ms", wavefront.trace.transfer_done_ms[i]},
                {"transfer_ms", transfer_ms},
                {"transfer_bytes", wavefront.trace.transfer_bytes[i]},
                {"npu_wait_start_ms", wavefront.trace.npu_wait_start_ms[i]},
                {"npu_tile_release_ms", wavefront.trace.npu_tile_release_ms[i]},
                {"npu_wait_gpu_ms", npu_wait_ms},
                {"npu_compute_done_ms", npu_compute_done_ms},
                {"compute_overlap_ms", overlap_ms},
            });
        }

        handoff_start_ms = treatment_start_ms + wavefront.trace.transfer_start_ms.front();
        gpu_prefix_export_done_ms = treatment_start_ms + wavefront.trace.transfer_done_ms.back();
        handoff_done_ms = gpu_prefix_export_done_ms;
    }

    const double treatment_decode_backend_ready_ms =
        decode_backend_ready_at_ms - treatment_start_ms;
    const double treatment_decode_logits_ready_ms =
        decode_logits_ready_at_ms - treatment_start_ms;
    const double treatment_decode_ready_ms = a_host_done_ms - treatment_start_ms;
    const double treatment_b_prefill_ready_ms = npu_done_ms - treatment_start_ms;
    const double treatment_b_gpu_ready_ms = b_gpu_ready_ms - treatment_start_ms;
    const double prefix_export_overlap_a_host_ms = std::max(0.0,
        std::min(gpu_prefix_export_done_ms, a_host_done_ms) -
        std::max(handoff_start_ms, a_host_start_ms));
    const double decode_backend_ready_speedup = treatment_decode_backend_ready_ms > 0.0 ?
        control_timing.backend_ms / treatment_decode_backend_ready_ms : 0.0;
    const double decode_logits_ready_speedup = treatment_decode_logits_ready_ms > 0.0 ?
        control_ms / treatment_decode_logits_ready_ms : 0.0;
    const double decode_ready_speedup = treatment_decode_ready_ms > 0.0 ?
        control_decode_ready_ms / treatment_decode_ready_ms : 0.0;

    const std::string gpu_buffer = hetero_expected_buffer(gpu_device);
    const std::string npu_buffer = hetero_expected_buffer(npu_device);
    const double gpu_placement_fraction = hetero_placement_fraction(gpu_placement, gpu_buffer);
    const double npu_placement_fraction = hetero_placement_fraction(npu_placement, npu_buffer);
    const int64_t gpu_unexpected_compute_nodes =
        hetero_unexpected_compute_nodes(gpu_placement, gpu_buffer);
    const int64_t npu_unexpected_compute_nodes =
        hetero_unexpected_compute_nodes(npu_placement, npu_buffer);
    const bool placement_ok = !placement_audit ||
        (hetero_placement_certified(gpu_placement, gpu_buffer) &&
         hetero_placement_certified(npu_placement, npu_buffer));
    const bool has_nonfinite_logits = decode_nonfinite_values > 0 ||
        b_prefill_diff.nonfinite_values > 0 || continuation_nonfinite_values > 0;
    const bool correctness_ok = a_match && a_numeric_ok &&
        b_prefill_match && b_prefill_numeric_ok &&
        b_continue_match && b_continue_numeric_ok;
    std::string status = "ok";
    if (placement_audit && !placement_ok) {
        status = "placement_fallback";
    } else if (has_nonfinite_logits) {
        status = "nonfinite_logits";
    } else if (!a_match) {
        status = "gpu_first_ubatch_mismatch";
    } else if (!a_numeric_ok) {
        status = "gpu_first_ubatch_numeric_mismatch";
    } else if (same_backend &&
               (!b_prefill_match || !b_prefill_numeric_ok ||
                !b_continue_match || !b_continue_numeric_ok)) {
        status = "same_backend_state_mismatch";
    } else if (!b_prefill_match || !b_prefill_numeric_ok ||
               !b_continue_match || !b_continue_numeric_ok) {
        status = "cross_backend_numeric_drift";
    }

    nlohmann::ordered_json result = {
        {"schema", "hetero-ubatch-v9-fail-closed-validation"},
        {"status", status},
        {"model", model_path},
        {"gpu_device", gpu_device},
        {"npu_device", npu_device},
        {"n_layer", n_layer},
        {"n_vocab", n_vocab},
        {"ubatch", ubatch},
        {"first_ubatch_decode_rows", decode_requests},
        {"first_ubatch_prefill_rows", prefix_len},
        {"npu_remainder_rows", prompt_len - prefix_len},
        {"prompt_len", prompt_len},
        {"decode_ctx", decode_ctx},
        {"decode_requests", decode_requests},
        {"decode_steps_limit", decode_steps},
        {"validation_steps", validation_steps},
        {"kv_unified", true},
        {"execution_mode", wavefront_enabled ? "layer_wavefront" : "full_state_handoff"},
        {"control", {
            {"gpu_mixed_full_ms", control_ms},
            {"gpu_mixed_backend_ms", control_timing.backend_ms},
            {"gpu_mixed_output_copy_ms", control_timing.output_copy_ms},
            {"decode_host_ms", control_a_host_ms},
            {"decode_tokens_ready_ms", control_decode_ready_ms},
            {"decode_token", control_a_tokens.front()},
            {"decode_tokens", control_a_tokens},
            {"prefill_token", control_b_token},
        }},
        {"treatment", {
            {"workers_ready_before_treatment", true},
            {"gpu_first_ubatch_ms", first_ubatch_ms},
            {"gpu_first_ubatch_backend_ms", first_ubatch_timing.backend_ms},
            {"gpu_first_ubatch_output_copy_ms", first_ubatch_timing.output_copy_ms},
            {"decode_backend_ready_ms", treatment_decode_backend_ready_ms},
            {"decode_backend_ready_speedup", decode_backend_ready_speedup},
            {"decode_logits_ready_ms", treatment_decode_logits_ready_ms},
            {"decode_logits_ready_speedup", decode_logits_ready_speedup},
            {"decode_ready_ms", treatment_decode_ready_ms},
            {"decode_ready_saved_ms", control_decode_ready_ms - treatment_decode_ready_ms},
            {"decode_ready_speedup", decode_ready_speedup},
            {"a_host_ms", a_host_ms},
            {"gpu_continuation_start_ms", gpu_continuation_start_ms - treatment_start_ms},
            {"npu_start_ms", npu_start_ms - treatment_start_ms},
            {"npu_remainder_ms", npu_remainder_ms},
            {"npu_physical_ubatches", npu_physical_ubatches},
            {"npu_submissions", (int) npu_submission_ms.size()},
            {"npu_submission_ms", npu_submission_ms},
            {"b_prefill_ready_ms", treatment_b_prefill_ready_ms},
            {"b_gpu_ready_ms", treatment_b_gpu_ready_ms},
            {"gpu_decode_steps", (int) gpu_step_ms.size()},
            {"gpu_decode_rows_per_step", decode_requests},
            {"gpu_decode_tokens", (int) gpu_step_ms.size() * decode_requests},
            {"gpu_steps_before_npu_done", gpu_steps_before_npu_done},
            {"gpu_paused_for_npu_final", gpu_paused_for_npu_final},
            {"gpu_decode_step_p50_ms", gpu_step_p50},
            {"gpu_decode_step_max_ms", gpu_step_max},
            {"gpu_decode_step_ms", gpu_step_ms},
            {"overlap_start_to_npu_done_ms", npu_done_ms - npu_start_ms},
            {"gpu_boundary_wait_after_npu_ms", std::max(0.0, npu_to_gpu_start_ms - npu_done_ms)},
        }},
        {"handoff", {
            {"worker", wavefront_enabled ? "gpu_eval_callback" : "npu"},
            {"gpu_u0_done_to_handoff_start_ms", handoff_start_ms - decode_backend_ready_at_ms},
            {"gpu_u0_done_to_prefix_export_ms", gpu_prefix_export_done_ms - decode_backend_ready_at_ms},
            {"prefix_export_overlap_a_host_ms", prefix_export_overlap_a_host_ms},
            {"prefix_handoff_total_ms", handoff_done_ms - handoff_start_ms},
            {"npu_release_delay_ms", wavefront_enabled ?
                wavefront.trace.npu_tile_release_ms.front() - wavefront.trace.transfer_done_ms.front() :
                npu_start_ms - handoff_done_ms},
            {"gpu_to_npu_bytes", wavefront_enabled ? wavefront_bytes : gpu_to_npu.data.size()},
            {"gpu_to_npu_export_ms", wavefront_enabled ? 0.0 : gpu_to_npu.export_ms},
            {"gpu_to_npu_import_ms", wavefront_enabled ? 0.0 : gpu_to_npu_import_ms},
            {"npu_to_gpu_bytes", npu_to_gpu.data.size()},
            {"npu_to_gpu_export_ms", npu_to_gpu.export_ms},
            {"npu_to_gpu_import_ms", npu_to_gpu_import_ms},
        }},
        {"wavefront", {
            {"enabled", wavefront_enabled},
            {"tile_layers", wavefront_enabled ? wavefront_layers : 0},
            {"tiles", wavefront_enabled ? wavefront.n_tiles : 0},
            {"first_npu_ubatch_done_ms", wavefront_enabled ?
                wavefront.trace.npu_first_ubatch_done_ms : 0.0},
            {"transfer_bytes", wavefront_bytes},
            {"transfer_total_ms", wavefront_transfer_ms},
            {"gpu_wait_npu_total_ms", wavefront_gpu_wait_ms},
            {"npu_wait_gpu_total_ms", wavefront_npu_wait_ms},
            {"compute_overlap_ms", wavefront_compute_overlap_ms},
            {"callback_error_code", wavefront_enabled ?
                wavefront.error_code.load(std::memory_order_relaxed) : 0},
            {"timeline", wavefront_tiles},
        }},
        {"correctness", {
            {"passed", correctness_ok},
            {"same_backend_relative_l2_limit", HETERO_SAME_BACKEND_RELATIVE_L2_LIMIT},
            {"cross_backend_relative_l2_limit", HETERO_CROSS_BACKEND_RELATIVE_L2_LIMIT},
            {"gpu_first_ubatch_argmax_match", a_match},
            {"gpu_first_ubatch_argmax_match_count", decode_match_count},
            {"gpu_first_ubatch_numeric_match", a_numeric_ok},
            {"gpu_first_ubatch_numeric_match_count", decode_numeric_count},
            {"gpu_first_ubatch_nonfinite_values", decode_nonfinite_values},
            {"gpu_first_ubatch_relative_l2_p50", decode_relative_l2_p50},
            {"gpu_first_ubatch_relative_l2_max", decode_relative_l2_max},
            {"gpu_first_ubatch_max_abs", decode_max_abs},
            {"gpu_first_ubatch_per_request", decode_checks},
            {"gpu_control_prefill_token", control_b_token},
            {"hetero_prefill_token", npu_b_token},
            {"prefill_argmax_match", b_prefill_match},
            {"prefill_numeric_match", b_prefill_numeric_ok},
            {"prefill_relative_l2", b_prefill_diff.relative_l2},
            {"prefill_max_abs", b_prefill_diff.max_abs},
            {"prefill_nonfinite_values", b_prefill_diff.nonfinite_values},
            {"npu_continuation_token", npu_next_token},
            {"gpu_continuation_token", gpu_next_token},
            {"continuation_argmax_match", b_continue_match},
            {"continuation_argmax_match_count", continuation_match_count},
            {"continuation_numeric_match", b_continue_numeric_ok},
            {"continuation_numeric_match_count", continuation_numeric_count},
            {"continuation_steps_requested", validation_steps},
            {"continuation_steps_completed", (int) continuation_checks.size()},
            {"continuation_relative_l2_p50", continuation_relative_l2_p50},
            {"continuation_relative_l2_max", continuation_relative_l2_max},
            {"continuation_max_abs", continuation_max_abs},
            {"continuation_nonfinite_values", continuation_nonfinite_values},
            {"npu_continuation_tokens", npu_continuation_tokens},
            {"gpu_continuation_tokens", gpu_continuation_tokens},
            {"continuation_checks", continuation_checks},
        }},
        {"setup", {
            {"warmup_enabled", warmup},
            {"handoff_warmup", warmup},
            {"placement_audit", placement_audit},
            {"gpu_warmup_ms", gpu_warmup_ms},
            {"npu_warmup_ms", npu_warmup_ms},
            {"seed_ms", seed_ms},
            {"seed_state_bytes", seed_state_bytes},
            {"seed_state_export_ms", seed_state_export_ms},
            {"seed_state_restore_ms", a_restore_ms},
            {"npu_continuation_decode_ms", npu_next_ms},
            {"gpu_continuation_decode_ms", gpu_next_ms},
        }},
        {"placement", {
            {"evidence", placement_audit ? "strict_scheduled_graph_node_buffers" : "disabled"},
            {"enabled", placement_audit},
            {"valid", placement_audit ? nlohmann::ordered_json(placement_ok) : nlohmann::ordered_json(nullptr)},
            {"policy", {
                {"all_compute_nodes_must_use_requested_buffer", true},
                {"permitted_metadata_ops", {
                    "NONE", "RESHAPE", "VIEW", "PERMUTE", "TRANSPOSE",
                }},
                {"permitted_transfer_ops", {"DUP", "CPY", "CONT"}},
            }},
            {"gpu_expected_buffer", gpu_buffer},
            {"gpu_target_fraction", gpu_placement_fraction},
            {"gpu_unexpected_compute_nodes", gpu_unexpected_compute_nodes},
            {"gpu_unexpected_compute", hetero_unexpected_compute_json(gpu_placement, gpu_buffer)},
            {"npu_expected_buffer", npu_buffer},
            {"npu_target_fraction", npu_placement_fraction},
            {"npu_unexpected_compute_nodes", npu_unexpected_compute_nodes},
            {"npu_unexpected_compute", hetero_unexpected_compute_json(npu_placement, npu_buffer)},
            {"gpu", hetero_placement_json(gpu_placement)},
            {"npu", hetero_placement_json(npu_placement)},
        }},
    };

    fprintf(stdout, "HETEROJSON %s\n", result.dump(-1, ' ', true).c_str());
    nlohmann::ordered_json summary = {
        {"mode", wavefront_enabled ? "layer_wavefront" : "full_state_handoff"},
        {"tile_layers", wavefront_enabled ? wavefront_layers : 0},
        {"status", status},
        {"decode_requests", decode_requests},
        {"first_ubatch_prefill_rows", prefix_len},
        {"control_ms", control_ms},
        {"control_backend_ms", control_timing.backend_ms},
        {"control_decode_ready_ms", control_decode_ready_ms},
        {"decode_backend_ready_ms", treatment_decode_backend_ready_ms},
        {"decode_logits_ready_ms", treatment_decode_logits_ready_ms},
        {"decode_ready_ms", treatment_decode_ready_ms},
        {"decode_ready_speedup", decode_ready_speedup},
        {"prefill_ready_ms", treatment_b_prefill_ready_ms},
        {"prefill_gpu_ready_ms", treatment_b_gpu_ready_ms},
        {"transfer_ms", wavefront_enabled ? wavefront_transfer_ms :
            gpu_to_npu.export_ms + gpu_to_npu_import_ms},
        {"compute_overlap_ms", wavefront_compute_overlap_ms},
        {"gpu_decode_steps", (int) gpu_step_ms.size()},
        {"gpu_decode_step_p50_ms", gpu_step_p50},
        {"transfer_bytes", wavefront_enabled ? wavefront_bytes : gpu_to_npu.data.size()},
        {"decode_argmax_match", a_match},
        {"decode_argmax_match_count", decode_match_count},
        {"correctness_passed", correctness_ok},
        {"prefill_argmax_match", b_prefill_match},
        {"continuation_argmax_match", b_continue_match},
        {"continuation_steps", (int) continuation_checks.size()},
    };
    fprintf(stdout, "HETEROSUMMARY %s\n", summary.dump(-1, ' ', true).c_str());
    fflush(stdout);
    fprintf(stderr,
        "[hetero] decode tokens ready %.2f -> %.2f ms (%.2fx), backend %.2f -> %.2f ms; "
        "B ready on NPU %.2f ms, on GPU %.2f ms; status=%s\n",
        control_decode_ready_ms, treatment_decode_ready_ms, decode_ready_speedup,
        control_timing.backend_ms, treatment_decode_backend_ready_ms,
        treatment_b_prefill_ready_ms, treatment_b_gpu_ready_ms, status.c_str());

    return hetero_result_code(placement_audit, placement_ok, correctness_ok);
}
