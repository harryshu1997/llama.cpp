#include "causal_quantized_weights.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

namespace {

using steady_clock = std::chrono::steady_clock;

enum class gpu_weight_mode {
    native,
    f16,
    f16_xmem,
};

enum class benchmark_mode {
    all,
    htp_control,
    htp_shard,
    gpu_shard,
    dual,
    repack_htp_control,
    repack_gpu_shard,
    reconstruct_gpu_shard,
    prepare_gpu_xmem,
};

struct config {
    int64_t k = 3840;
    int64_t n_ff = 15360;
    int64_t htp_columns = 8192;
    int64_t gpu_columns = 1472;
    int64_t batch = 1;
    int warmup = 20;
    int iterations = 100;
    ggml_type weight_type = GGML_TYPE_Q4_0;
    gpu_weight_mode gpu_mode = gpu_weight_mode::native;
    benchmark_mode benchmark = benchmark_mode::all;
    std::string htp_backend = "HTP";
    std::string gpu_backend = "OpenCL";
};

struct backend_state {
    ggml_backend_dev_t device = nullptr;
    ggml_backend_t backend = nullptr;
};

struct shared_io {
    ggml_context * htp_ctx = nullptr;
    ggml_backend_buffer_t htp_buffer = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * control_output = nullptr;
    ggml_tensor * htp_output = nullptr;
    ggml_tensor * gpu_output = nullptr;

    ggml_context * gpu_ctx = nullptr;
    ggml_backend_buffer_t gpu_buffer = nullptr;
    ggml_tensor * gpu_input_alias = nullptr;
    ggml_tensor * gpu_output_alias = nullptr;
};

struct ffn_engine {
    std::string label;
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    int64_t offset = 0;
    int64_t columns = 0;

    ggml_context * weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_tensor * gate_weight = nullptr;
    ggml_tensor * up_weight = nullptr;
    ggml_tensor * down_weight = nullptr;
    std::vector<uint8_t> gate_source;
    std::vector<uint8_t> up_source;
    std::vector<uint8_t> down_source;

    ggml_context * graph_ctx = nullptr;
    ggml_gallocr_t graph_alloc = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;

    double pack_ms = 0.0;
    double host_prepare_ms = 0.0;
    double prime_ms = 0.0;
    size_t weight_bytes = 0;
    ggml_type resident_type = GGML_TYPE_COUNT;
};

struct dual_sample {
    double compute_ms = 0.0;
    double merge_ms = 0.0;
    double total_ms = 0.0;
    double htp_ms = 0.0;
    double gpu_ms = 0.0;
    double launch_skew_ms = 0.0;
    bool ok = false;
};

double elapsed_ms(steady_clock::time_point started) {
    return std::chrono::duration<double, std::milli>(
            steady_clock::now() - started).count();
}

uint64_t boot_time_ns() {
    struct timespec value = {};
    clock_gettime(CLOCK_BOOTTIME, &value);
    return static_cast<uint64_t>(value.tv_sec) * UINT64_C(1000000000) +
            static_cast<uint64_t>(value.tv_nsec);
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

bool parse_int(const char * text, int & value) {
    int64_t parsed = 0;
    if (!parse_i64(text, parsed) || parsed < 0 || parsed > INT32_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool parse_type(const char * text, ggml_type & type) {
    if (strcmp(text, "q4_0") == 0) {
        type = GGML_TYPE_Q4_0;
        return true;
    }
    if (strcmp(text, "q8_0") == 0) {
        type = GGML_TYPE_Q8_0;
        return true;
    }
    return false;
}

bool parse_gpu_mode(const char * text, gpu_weight_mode & mode) {
    if (strcmp(text, "native") == 0) {
        mode = gpu_weight_mode::native;
        return true;
    }
    if (strcmp(text, "f16") == 0) {
        mode = gpu_weight_mode::f16;
        return true;
    }
    if (strcmp(text, "f16-xmem") == 0) {
        mode = gpu_weight_mode::f16_xmem;
        return true;
    }
    return false;
}

bool parse_benchmark_mode(const char * text, benchmark_mode & mode) {
    if (strcmp(text, "all") == 0) {
        mode = benchmark_mode::all;
        return true;
    }
    if (strcmp(text, "htp-control") == 0) {
        mode = benchmark_mode::htp_control;
        return true;
    }
    if (strcmp(text, "htp-shard") == 0) {
        mode = benchmark_mode::htp_shard;
        return true;
    }
    if (strcmp(text, "gpu-shard") == 0) {
        mode = benchmark_mode::gpu_shard;
        return true;
    }
    if (strcmp(text, "dual") == 0) {
        mode = benchmark_mode::dual;
        return true;
    }
    if (strcmp(text, "repack-htp-control") == 0) {
        mode = benchmark_mode::repack_htp_control;
        return true;
    }
    if (strcmp(text, "repack-gpu-shard") == 0) {
        mode = benchmark_mode::repack_gpu_shard;
        return true;
    }
    if (strcmp(text, "reconstruct-gpu-shard") == 0) {
        mode = benchmark_mode::reconstruct_gpu_shard;
        return true;
    }
    if (strcmp(text, "prepare-gpu-xmem") == 0) {
        mode = benchmark_mode::prepare_gpu_xmem;
        return true;
    }
    return false;
}

const char * benchmark_mode_name(benchmark_mode mode) {
    switch (mode) {
        case benchmark_mode::all:
            return "all";
        case benchmark_mode::htp_control:
            return "htp-control";
        case benchmark_mode::htp_shard:
            return "htp-shard";
        case benchmark_mode::gpu_shard:
            return "gpu-shard";
        case benchmark_mode::dual:
            return "dual";
        case benchmark_mode::repack_htp_control:
            return "repack-htp-control";
        case benchmark_mode::repack_gpu_shard:
            return "repack-gpu-shard";
        case benchmark_mode::reconstruct_gpu_shard:
            return "reconstruct-gpu-shard";
        case benchmark_mode::prepare_gpu_xmem:
            return "prepare-gpu-xmem";
    }
    return "unknown";
}

const char * gpu_mode_name(gpu_weight_mode mode) {
    switch (mode) {
        case gpu_weight_mode::native:
            return "native";
        case gpu_weight_mode::f16:
            return "f16";
        case gpu_weight_mode::f16_xmem:
            return "f16-xmem";
    }
    return "unknown";
}

bool parse_config(int argc, char ** argv, config & cfg) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            if (!parse_i64(argv[++i], cfg.k)) {
                return false;
            }
        } else if (strcmp(argv[i], "--n-ff") == 0 && i + 1 < argc) {
            if (!parse_i64(argv[++i], cfg.n_ff)) {
                return false;
            }
        } else if (strcmp(argv[i], "--htp-columns") == 0 && i + 1 < argc) {
            if (!parse_i64(argv[++i], cfg.htp_columns)) {
                return false;
            }
        } else if (strcmp(argv[i], "--gpu-columns") == 0 && i + 1 < argc) {
            if (!parse_i64(argv[++i], cfg.gpu_columns)) {
                return false;
            }
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            if (!parse_i64(argv[++i], cfg.batch)) {
                return false;
            }
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], cfg.warmup)) {
                return false;
            }
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], cfg.iterations)) {
                return false;
            }
        } else if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            if (!parse_type(argv[++i], cfg.weight_type)) {
                return false;
            }
        } else if (strcmp(argv[i], "--gpu-mode") == 0 && i + 1 < argc) {
            if (!parse_gpu_mode(argv[++i], cfg.gpu_mode)) {
                return false;
            }
        } else if (strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc) {
            if (!parse_benchmark_mode(argv[++i], cfg.benchmark)) {
                return false;
            }
        } else if (strcmp(argv[i], "--htp-backend") == 0 && i + 1 < argc) {
            cfg.htp_backend = argv[++i];
        } else if (strcmp(argv[i], "--gpu-backend") == 0 && i + 1 < argc) {
            cfg.gpu_backend = argv[++i];
        } else {
            return false;
        }
    }

    const int64_t block = ggml_blck_size(cfg.weight_type);
    const int64_t total_columns = cfg.htp_columns + cfg.gpu_columns;
    return cfg.k > 0 && cfg.n_ff > 0 && cfg.htp_columns > 0 &&
           cfg.gpu_columns > 0 && total_columns <= cfg.n_ff && cfg.batch > 0 &&
           cfg.batch <= 512 && cfg.warmup >= 0 && cfg.iterations > 0 &&
           cfg.k % block == 0 && cfg.n_ff % block == 0 &&
           cfg.htp_columns % block == 0 && cfg.gpu_columns % block == 0 &&
           total_columns % block == 0 && !cfg.htp_backend.empty() &&
           !cfg.gpu_backend.empty();
}

bool xmem_matmul_shape_eligible(int64_t k, int64_t rows, int64_t batch) {
    return batch > 1 && rows >= 64 && batch >= 16 && k >= 64 && k % 8 == 0;
}

int expected_xmem_matmuls(const config & cfg) {
    if (cfg.gpu_mode != gpu_weight_mode::f16_xmem) {
        return 0;
    }
    int count = 0;
    count += xmem_matmul_shape_eligible(
            cfg.k, cfg.gpu_columns, cfg.batch) ? 2 : 0;
    count += xmem_matmul_shape_eligible(
            cfg.gpu_columns, cfg.k, cfg.batch) ? 1 : 0;
    return count;
}

backend_state initialize_backend(const std::string & needle) {
    backend_state result;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(device);
        const char * description = ggml_backend_dev_description(device);
        if ((name != nullptr && std::string(name).find(needle) != std::string::npos) ||
            (description != nullptr &&
             std::string(description).find(needle) != std::string::npos)) {
            result.device = device;
            result.backend = ggml_backend_dev_init(device, nullptr);
            break;
        }
    }
    return result;
}

ggml_backend_buffer_type_t weight_buffer_type(
        ggml_backend_dev_t device, ggml_backend_t backend, ggml_type type) {
    ggml_backend_buffer_type_t result =
            ggml_backend_get_default_buffer_type(backend);
    if (!ggml_is_quantized(type)) {
        return result;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    auto get_extra_bufts = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_dev_get_extra_bufts"));
    ggml_backend_buffer_type_t * extra =
            get_extra_bufts != nullptr ? get_extra_bufts(device) : nullptr;
    if (extra != nullptr && extra[0] != nullptr) {
        result = extra[0];
    }
    return result;
}

bool initialize_shared_io(
        const config & cfg,
        const backend_state & htp,
        const backend_state & gpu,
        shared_io & io) {
    ggml_init_params htp_params = {
        ggml_tensor_overhead() * 16,
        nullptr,
        true,
    };
    io.htp_ctx = ggml_init(htp_params);
    if (io.htp_ctx == nullptr) {
        return false;
    }
    io.input = ggml_new_tensor_2d(
            io.htp_ctx, GGML_TYPE_F32, cfg.k, cfg.batch);
    io.control_output = ggml_new_tensor_2d(
            io.htp_ctx, GGML_TYPE_F32, cfg.k, cfg.batch);
    io.htp_output = ggml_new_tensor_2d(
            io.htp_ctx, GGML_TYPE_F32, cfg.k, cfg.batch);
    io.gpu_output = ggml_new_tensor_2d(
            io.htp_ctx, GGML_TYPE_F32, cfg.k, cfg.batch);
    ggml_set_name(io.input, "dual.input.weight");
    ggml_set_name(io.control_output, "dual.control.output.weight");
    ggml_set_name(io.htp_output, "dual.htp.output.weight");
    ggml_set_name(io.gpu_output, "dual.gpu.output.weight");
    ggml_set_input(io.input);
    ggml_set_output(io.control_output);
    ggml_set_output(io.htp_output);
    ggml_set_output(io.gpu_output);

    io.htp_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            io.htp_ctx, ggml_backend_get_default_buffer_type(htp.backend));
    if (io.htp_buffer == nullptr) {
        return false;
    }

    ggml_init_params gpu_params = {
        ggml_tensor_overhead() * 8,
        nullptr,
        true,
    };
    io.gpu_ctx = ggml_init(gpu_params);
    if (io.gpu_ctx == nullptr) {
        return false;
    }
    io.gpu_input_alias = ggml_new_tensor_2d(
            io.gpu_ctx, GGML_TYPE_F32, cfg.k, cfg.batch);
    io.gpu_output_alias = ggml_new_tensor_2d(
            io.gpu_ctx, GGML_TYPE_F32, cfg.k, cfg.batch);
    ggml_set_name(io.gpu_input_alias, "dual.input.weight");
    ggml_set_name(io.gpu_output_alias, "dual.gpu.output.weight");
    ggml_set_input(io.gpu_input_alias);
    ggml_set_output(io.gpu_output_alias);

    io.gpu_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            io.gpu_ctx, ggml_backend_get_default_buffer_type(gpu.backend));
    return io.gpu_buffer != nullptr;
}

bool fill_quantized_weight(
        ggml_tensor * tensor,
        ggml_type type,
        uint32_t matrix,
        uint32_t row_offset,
        uint32_t column_offset,
        std::vector<uint8_t> & bytes) {
    bytes.resize(ggml_nbytes(tensor));
    return s41_fill_quantized_weight(
            type, matrix, row_offset, column_offset,
            static_cast<uint32_t>(tensor->ne[0]),
            static_cast<uint32_t>(tensor->ne[1]),
            bytes.data(), bytes.size());
}

bool fill_f16_from_quantized_weight(
        ggml_tensor * tensor,
        ggml_type source_type,
        uint32_t matrix,
        uint32_t row_offset,
        uint32_t column_offset,
        std::vector<uint8_t> & bytes) {
    if (tensor->type != GGML_TYPE_F16 ||
        (source_type != GGML_TYPE_Q4_0 && source_type != GGML_TYPE_Q8_0)) {
        return false;
    }

    const int64_t columns = tensor->ne[0];
    const int64_t rows = tensor->ne[1];
    const size_t quantized_row_bytes = ggml_row_size(source_type, columns);
    std::vector<uint8_t> quantized(quantized_row_bytes * rows);
    if (!s41_fill_quantized_weight(
                source_type, matrix, row_offset, column_offset,
                static_cast<uint32_t>(columns), static_cast<uint32_t>(rows),
                quantized.data(), quantized.size())) {
        return false;
    }

    std::vector<float> row_f32(columns);
    std::vector<ggml_fp16_t> values(
            static_cast<size_t>(columns) * static_cast<size_t>(rows));
    for (int64_t row = 0; row < rows; ++row) {
        const uint8_t * source =
                quantized.data() + static_cast<size_t>(row) * quantized_row_bytes;
        if (source_type == GGML_TYPE_Q4_0) {
            dequantize_row_q4_0(
                    reinterpret_cast<const block_q4_0 *>(source),
                    row_f32.data(), columns);
        } else {
            dequantize_row_q8_0(
                    reinterpret_cast<const block_q8_0 *>(source),
                    row_f32.data(), columns);
        }
        ggml_fp32_to_fp16_row(
                row_f32.data(), values.data() + row * columns, columns);
    }
    bytes.resize(values.size() * sizeof(values[0]));
    memcpy(bytes.data(), values.data(), bytes.size());
    return bytes.size() == ggml_nbytes(tensor);
}

bool fill_weight(
        ggml_tensor * tensor,
        ggml_type source_type,
        uint32_t matrix,
        uint32_t row_offset,
        uint32_t column_offset,
        std::vector<uint8_t> & bytes) {
    if (tensor->type == source_type) {
        return fill_quantized_weight(
                tensor, source_type, matrix, row_offset, column_offset, bytes);
    }
    return fill_f16_from_quantized_weight(
            tensor, source_type, matrix, row_offset, column_offset, bytes);
}

bool initialize_engine(
        const config & cfg,
        const backend_state & backend,
        const char * label,
        int64_t offset,
        int64_t columns,
        ggml_type resident_type,
        ggml_tensor * input,
        ggml_tensor * output,
        ffn_engine & engine) {
    engine.label = label;
    engine.backend = backend.backend;
    engine.device = backend.device;
    engine.offset = offset;
    engine.columns = columns;
    engine.resident_type = resident_type;
    engine.input = input;
    engine.output = output;

    ggml_init_params weight_params = {
        ggml_tensor_overhead() * 8,
        nullptr,
        true,
    };
    engine.weight_ctx = ggml_init(weight_params);
    if (engine.weight_ctx == nullptr) {
        return false;
    }
    engine.gate_weight = ggml_new_tensor_2d(
            engine.weight_ctx, resident_type, cfg.k, columns);
    engine.up_weight = ggml_new_tensor_2d(
            engine.weight_ctx, resident_type, cfg.k, columns);
    engine.down_weight = ggml_new_tensor_2d(
            engine.weight_ctx, resident_type, columns, cfg.k);
    ggml_set_name(engine.gate_weight, (engine.label + ".gate").c_str());
    ggml_set_name(engine.up_weight, (engine.label + ".up").c_str());
    ggml_set_name(engine.down_weight, (engine.label + ".down").c_str());

    ggml_backend_buffer_type_t weight_buft = weight_buffer_type(
            backend.device, backend.backend, resident_type);
    engine.weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            engine.weight_ctx, weight_buft);
    if (engine.weight_buffer == nullptr) {
        fprintf(stderr, "[dual-repack] %s weight allocation failed\n", label);
        return false;
    }
    engine.weight_bytes = ggml_backend_buffer_get_size(engine.weight_buffer);

    const auto host_prepare_started = steady_clock::now();
    if (!fill_weight(
                engine.gate_weight, cfg.weight_type, S41_WEIGHT_GATE,
                static_cast<uint32_t>(offset), 0, engine.gate_source) ||
        !fill_weight(
                engine.up_weight, cfg.weight_type, S41_WEIGHT_UP,
                static_cast<uint32_t>(offset), 0, engine.up_source) ||
        !fill_weight(
                engine.down_weight, cfg.weight_type, S41_WEIGHT_DOWN, 0,
                static_cast<uint32_t>(offset), engine.down_source)) {
        fprintf(stderr, "[dual-repack] %s deterministic weight fill failed\n", label);
        return false;
    }
    engine.host_prepare_ms = elapsed_ms(host_prepare_started);

    const auto pack_started = steady_clock::now();
    ggml_backend_tensor_set(
            engine.gate_weight, engine.gate_source.data(), 0,
            engine.gate_source.size());
    ggml_backend_tensor_set(
            engine.up_weight, engine.up_source.data(), 0,
            engine.up_source.size());
    ggml_backend_tensor_set(
            engine.down_weight, engine.down_source.data(), 0,
            engine.down_source.size());
    ggml_backend_synchronize(backend.backend);
    engine.pack_ms = elapsed_ms(pack_started);

    ggml_init_params graph_params = {
        ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
        nullptr,
        true,
    };
    engine.graph_ctx = ggml_init(graph_params);
    if (engine.graph_ctx == nullptr) {
        return false;
    }
    ggml_tensor * gate = ggml_mul_mat(
            engine.graph_ctx, engine.gate_weight, input);
    ggml_tensor * up = ggml_mul_mat(
            engine.graph_ctx, engine.up_weight, input);
    ggml_tensor * activation = ggml_geglu_split(engine.graph_ctx, gate, up);
    ggml_tensor * partial = ggml_mul_mat(
            engine.graph_ctx, engine.down_weight, activation);
    ggml_tensor * copied = ggml_cpy(engine.graph_ctx, partial, output);
    engine.graph = ggml_new_graph(engine.graph_ctx);
    ggml_build_forward_expand(engine.graph, copied);

    for (int i = 0; i < ggml_graph_n_nodes(engine.graph); ++i) {
        ggml_tensor * node = ggml_graph_node(engine.graph, i);
        if (!ggml_backend_supports_op(backend.backend, node)) {
            fprintf(stderr,
                    "[dual-repack] %s unsupported op=%s node=%s\n",
                    label, ggml_op_name(node->op), node->name);
            return false;
        }
    }
    engine.graph_alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend.backend));
    if (engine.graph_alloc == nullptr ||
        !ggml_gallocr_alloc_graph(engine.graph_alloc, engine.graph)) {
        fprintf(stderr, "[dual-repack] %s graph allocation failed\n", label);
        return false;
    }
    return true;
}

void destroy_engine(ffn_engine & engine) {
    if (engine.graph_alloc != nullptr) {
        ggml_gallocr_free(engine.graph_alloc);
    }
    if (engine.graph_ctx != nullptr) {
        ggml_free(engine.graph_ctx);
    }
    if (engine.weight_buffer != nullptr) {
        ggml_backend_buffer_free(engine.weight_buffer);
    }
    if (engine.weight_ctx != nullptr) {
        ggml_free(engine.weight_ctx);
    }
    engine = {};
}

bool fresh_repack(
        const config & cfg,
        const backend_state & backend,
        const ffn_engine & source,
        double & pack_ms,
        double & total_ms) {
    const auto total_started = steady_clock::now();
    ffn_engine target;
    target.backend = backend.backend;
    target.device = backend.device;
    target.resident_type = source.resident_type;
    ggml_init_params params = {
        ggml_tensor_overhead() * 8,
        nullptr,
        true,
    };
    target.weight_ctx = ggml_init(params);
    if (target.weight_ctx == nullptr) {
        return false;
    }
    target.gate_weight = ggml_new_tensor_2d(
            target.weight_ctx, source.resident_type, cfg.k, source.columns);
    target.up_weight = ggml_new_tensor_2d(
            target.weight_ctx, source.resident_type, cfg.k, source.columns);
    target.down_weight = ggml_new_tensor_2d(
            target.weight_ctx, source.resident_type, source.columns, cfg.k);
    ggml_backend_buffer_type_t buft = weight_buffer_type(
            backend.device, backend.backend, source.resident_type);
    target.weight_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            target.weight_ctx, buft);
    if (target.weight_buffer == nullptr) {
        destroy_engine(target);
        return false;
    }
    const auto pack_started = steady_clock::now();
    ggml_backend_tensor_set(
            target.gate_weight, source.gate_source.data(), 0,
            source.gate_source.size());
    ggml_backend_tensor_set(
            target.up_weight, source.up_source.data(), 0,
            source.up_source.size());
    ggml_backend_tensor_set(
            target.down_weight, source.down_source.data(), 0,
            source.down_source.size());
    ggml_backend_synchronize(backend.backend);
    pack_ms = elapsed_ms(pack_started);
    destroy_engine(target);
    total_ms = elapsed_ms(total_started);
    return true;
}

bool compute(ffn_engine & engine) {
    const ggml_status status =
            ggml_backend_graph_compute_async(engine.backend, engine.graph);
    ggml_backend_synchronize(engine.backend);
    return status == GGML_STATUS_SUCCESS;
}

bool fresh_xmem_prepare(
        const config & cfg,
        const backend_state & backend,
        ggml_tensor * input,
        ggml_tensor * output,
        int64_t offset,
        ffn_engine & target,
        double & host_prepare_ms,
        double & pack_ms,
        double & prime_ms,
        double & total_ms,
        size_t & weight_bytes) {
    const auto total_started = steady_clock::now();
    if (!initialize_engine(
                cfg, backend, "gpu-xmem-fresh", offset, cfg.gpu_columns,
                GGML_TYPE_F16, input, output, target)) {
        destroy_engine(target);
        return false;
    }
    const auto prime_started = steady_clock::now();
    const bool ok = compute(target);
    prime_ms = elapsed_ms(prime_started);
    host_prepare_ms = target.host_prepare_ms;
    pack_ms = target.pack_ms;
    weight_bytes = target.weight_bytes;
    total_ms = elapsed_ms(total_started);
    return ok;
}

double reconstruct(const config & cfg, ffn_engine & engine) {
    const auto started = steady_clock::now();
    if (!fill_weight(
                engine.gate_weight, cfg.weight_type, S41_WEIGHT_GATE,
                static_cast<uint32_t>(engine.offset), 0,
                engine.gate_source) ||
        !fill_weight(
                engine.up_weight, cfg.weight_type, S41_WEIGHT_UP,
                static_cast<uint32_t>(engine.offset), 0,
                engine.up_source) ||
        !fill_weight(
                engine.down_weight, cfg.weight_type, S41_WEIGHT_DOWN, 0,
                static_cast<uint32_t>(engine.offset),
                engine.down_source)) {
        return -1.0;
    }
    return elapsed_ms(started);
}

void fill_input(std::vector<float> & values, uint64_t sequence) {
    for (size_t i = 0; i < values.size(); ++i) {
        const uint64_t mixed = s41_mix64(
                sequence * UINT64_C(0x9e3779b97f4a7c15) + i);
        const int32_t centered = static_cast<int32_t>(mixed % 2001) - 1000;
        values[i] = static_cast<float>(centered) / 1000.0f;
    }
}

bool all_finite(const float * values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        memcpy(&bits, values + i, sizeof(bits));
        if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
            return false;
        }
    }
    return true;
}

double percentile(std::vector<double> values, double quantile) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t index = std::min(
            values.size() - 1,
            static_cast<size_t>(std::ceil(quantile * values.size()) - 1));
    return values[index];
}

class dual_executor {
public:
    dual_executor(ffn_engine & htp, ffn_engine & gpu)
        : htp_(htp), gpu_(gpu), htp_thread_(&dual_executor::worker, this, true),
          gpu_thread_(&dual_executor::worker, this, false) {
    }

    ~dual_executor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++epoch_;
        }
        start_cv_.notify_all();
        htp_thread_.join();
        gpu_thread_.join();
    }

    dual_sample run(
            const float * htp_output,
            const float * gpu_output,
            float * merged,
            size_t count) {
        dual_sample sample;
        steady_clock::time_point wall_started;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            completed_ = 0;
            htp_ok_ = false;
            gpu_ok_ = false;
            wall_started = steady_clock::now();
            ++epoch_;
        }
        start_cv_.notify_all();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [&]() { return completed_ == 2; });
            sample.htp_ms = htp_ms_;
            sample.gpu_ms = gpu_ms_;
            sample.launch_skew_ms = std::chrono::duration<double, std::milli>(
                    htp_started_ > gpu_started_ ?
                            htp_started_ - gpu_started_ :
                            gpu_started_ - htp_started_).count();
            sample.ok = htp_ok_ && gpu_ok_;
        }
        const auto merge_started = steady_clock::now();
        sample.compute_ms = std::chrono::duration<double, std::milli>(
                merge_started - wall_started).count();
        for (size_t i = 0; i < count; ++i) {
            merged[i] = htp_output[i] + gpu_output[i];
        }
        sample.merge_ms = elapsed_ms(merge_started);
        sample.total_ms = elapsed_ms(wall_started);
        return sample;
    }

private:
    void worker(bool htp_leg) {
        uint64_t local_epoch = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_cv_.wait(lock, [&]() {
                    return stop_ || epoch_ != local_epoch;
                });
                if (stop_) {
                    return;
                }
                local_epoch = epoch_;
                const auto started = steady_clock::now();
                if (htp_leg) {
                    htp_started_ = started;
                } else {
                    gpu_started_ = started;
                }
            }

            const auto started = steady_clock::now();
            const bool ok = compute(htp_leg ? htp_ : gpu_);
            const double duration_ms = elapsed_ms(started);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (htp_leg) {
                    htp_ms_ = duration_ms;
                    htp_ok_ = ok;
                } else {
                    gpu_ms_ = duration_ms;
                    gpu_ok_ = ok;
                }
                ++completed_;
            }
            done_cv_.notify_one();
        }
    }

    ffn_engine & htp_;
    ffn_engine & gpu_;
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    uint64_t epoch_ = 0;
    int completed_ = 0;
    bool stop_ = false;
    bool htp_ok_ = false;
    bool gpu_ok_ = false;
    double htp_ms_ = 0.0;
    double gpu_ms_ = 0.0;
    steady_clock::time_point htp_started_;
    steady_clock::time_point gpu_started_;
    std::thread htp_thread_;
    std::thread gpu_thread_;
};

std::vector<double> benchmark_solo(
        ffn_engine & engine,
        ggml_tensor * shared_input,
        std::vector<float> & input_data,
        int warmup,
        int iterations,
        uint64_t sequence_base,
        const char * energy_label,
        bool & ok) {
    std::vector<double> samples;
    ok = true;
    for (int i = 0; i < warmup + iterations; ++i) {
        if (i == warmup && energy_label != nullptr) {
            printf("PHONE_ENERGY_WINDOW_START phone_uptime_ns=%llu label=%s\n",
                    static_cast<unsigned long long>(boot_time_ns()),
                    energy_label);
            fflush(stdout);
        }
        fill_input(input_data, sequence_base + static_cast<uint64_t>(i));
        ggml_backend_tensor_set(
                shared_input, input_data.data(), 0,
                input_data.size() * sizeof(float));
        const auto started = steady_clock::now();
        const bool compute_ok = compute(engine);
        const double duration_ms = elapsed_ms(started);
        if (!compute_ok) {
            ok = false;
            break;
        }
        if (i >= warmup) {
            samples.push_back(duration_ms);
        }
    }
    if (energy_label != nullptr) {
        printf("PHONE_ENERGY_WINDOW_END phone_uptime_ns=%llu label=%s\n",
                static_cast<unsigned long long>(boot_time_ns()),
                energy_label);
        fflush(stdout);
    }
    return samples;
}

std::vector<dual_sample> benchmark_dual(
        dual_executor & executor,
        ggml_tensor * shared_input,
        const float * htp_output,
        const float * gpu_output,
        std::vector<float> & input_data,
        std::vector<float> & merged,
        int warmup,
        int iterations,
        uint64_t sequence_base,
        const char * energy_label,
        bool & ok) {
    std::vector<dual_sample> samples;
    ok = true;
    for (int i = 0; i < warmup + iterations; ++i) {
        if (i == warmup && energy_label != nullptr) {
            printf("PHONE_ENERGY_WINDOW_START phone_uptime_ns=%llu label=%s\n",
                    static_cast<unsigned long long>(boot_time_ns()),
                    energy_label);
            fflush(stdout);
        }
        fill_input(input_data, sequence_base + static_cast<uint64_t>(i));
        ggml_backend_tensor_set(
                shared_input, input_data.data(), 0,
                input_data.size() * sizeof(float));
        dual_sample sample = executor.run(
                htp_output, gpu_output, merged.data(), merged.size());
        if (!sample.ok || !all_finite(merged.data(), merged.size())) {
            ok = false;
            break;
        }
        if (i >= warmup) {
            samples.push_back(sample);
        }
    }
    if (energy_label != nullptr) {
        printf("PHONE_ENERGY_WINDOW_END phone_uptime_ns=%llu label=%s\n",
                static_cast<unsigned long long>(boot_time_ns()),
                energy_label);
        fflush(stdout);
    }
    return samples;
}

void compare_outputs(
        const float * reference,
        const float * actual,
        size_t count,
        double & rel_l2,
        double & max_abs) {
    long double error_sum = 0.0;
    long double reference_sum = 0.0;
    max_abs = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double difference =
                static_cast<double>(actual[i]) - reference[i];
        error_sum += static_cast<long double>(difference) * difference;
        reference_sum += static_cast<long double>(reference[i]) * reference[i];
        max_abs = std::max(max_abs, std::abs(difference));
    }
    rel_l2 = reference_sum > 0.0 ?
            std::sqrt(static_cast<double>(error_sum / reference_sum)) :
            std::sqrt(static_cast<double>(error_sum));
}

} // namespace

int main(int argc, char ** argv) {
    config cfg;
    if (!parse_config(argc, argv, cfg)) {
        fprintf(stderr,
                "usage: %s [--k N] [--n-ff N] [--htp-columns N] "
                "[--gpu-columns N] [--batch N] [--type q4_0|q8_0] "
                "[--gpu-mode native|f16|f16-xmem] "
                "[--benchmark all|htp-control|htp-shard|gpu-shard|dual|"
                "repack-htp-control|repack-gpu-shard|"
                "reconstruct-gpu-shard|prepare-gpu-xmem] "
                "[--warmup N] [--iterations N] "
                "[--htp-backend NAME] [--gpu-backend NAME]\n",
                argv[0]);
        return 2;
    }

    setenv("GGML_PHONE_SHARE_PUBLISH", "1", 1);
    setenv("GGML_PHONE_SHARE_IMPORT", "1", 1);
    if (cfg.gpu_mode == gpu_weight_mode::f16_xmem) {
        setenv("GGML_OPENCL_ADRENO_XMEM_GEMM", "1", 1);
        setenv("GGML_OPENCL_XMEM_PREPACK_CACHE", "1", 1);
        setenv("GGML_OPENCL_XMEM_IMAGE_CACHE", "1", 1);
    } else {
        unsetenv("GGML_OPENCL_ADRENO_XMEM_GEMM");
        unsetenv("GGML_OPENCL_XMEM_PREPACK_CACHE");
        unsetenv("GGML_OPENCL_XMEM_IMAGE_CACHE");
    }
    ggml_backend_load_all();

    backend_state htp = initialize_backend(cfg.htp_backend);
    backend_state gpu = initialize_backend(cfg.gpu_backend);
    if (htp.backend == nullptr || gpu.backend == nullptr ||
        htp.backend == gpu.backend) {
        fprintf(stderr,
                "[dual-repack] backend initialization failed htp=%s gpu=%s\n",
                cfg.htp_backend.c_str(), cfg.gpu_backend.c_str());
        return 1;
    }
    fprintf(stderr,
            "[dual-repack] HTP=%s (%s) GPU=%s (%s)\n",
            ggml_backend_dev_name(htp.device),
            ggml_backend_dev_description(htp.device),
            ggml_backend_dev_name(gpu.device),
            ggml_backend_dev_description(gpu.device));

    shared_io io;
    if (!initialize_shared_io(cfg, htp, gpu, io)) {
        fprintf(stderr, "[dual-repack] shared DMA-BUF I/O setup failed\n");
        return 1;
    }

    const int64_t total_columns = cfg.htp_columns + cfg.gpu_columns;
    const int64_t total_offset = cfg.n_ff - total_columns;
    const int64_t gpu_offset = total_offset;
    const int64_t htp_offset = gpu_offset + cfg.gpu_columns;
    const ggml_type gpu_resident_type =
            cfg.gpu_mode == gpu_weight_mode::native ?
                    cfg.weight_type : GGML_TYPE_F16;
    const int xmem_matmuls = expected_xmem_matmuls(cfg);

    ffn_engine htp_control;
    ffn_engine htp_shard;
    ffn_engine gpu_shard;
    if (!initialize_engine(
                cfg, htp, "htp-control", total_offset, total_columns,
                cfg.weight_type, io.input, io.control_output, htp_control) ||
        !initialize_engine(
                cfg, htp, "htp-shard", htp_offset, cfg.htp_columns,
                cfg.weight_type, io.input, io.htp_output, htp_shard) ||
        !initialize_engine(
                cfg, gpu, "gpu-shard", gpu_offset, cfg.gpu_columns,
                gpu_resident_type, io.gpu_input_alias, io.gpu_output_alias,
                gpu_shard)) {
        fprintf(stderr, "[dual-repack] FFN graph preparation failed\n");
        return 1;
    }

    const size_t elements = static_cast<size_t>(cfg.k) * cfg.batch;
    std::vector<float> input_data(elements);
    std::vector<float> merged(elements);
    bool prime_ok = true;
    if (cfg.gpu_mode == gpu_weight_mode::f16_xmem && xmem_matmuls == 3) {
        fill_input(input_data, UINT64_C(0x786d656d));
        ggml_backend_tensor_set(
                io.input, input_data.data(), 0,
                input_data.size() * sizeof(float));
        const auto prime_started = steady_clock::now();
        prime_ok = compute(gpu_shard);
        gpu_shard.prime_ms = elapsed_ms(prime_started);
    }
    if (!prime_ok) {
        fprintf(stderr, "[dual-repack] GPU xmem prime failed\n");
        return 1;
    }

    const size_t xmem_cache_bytes = xmem_matmuls == 3 ?
            ggml_nbytes(gpu_shard.gate_weight) +
                    ggml_nbytes(gpu_shard.up_weight) +
                    ggml_nbytes(gpu_shard.down_weight) : 0;
    const size_t dual_resident_bytes = htp_shard.weight_bytes +
            gpu_shard.weight_bytes + xmem_cache_bytes;

    fprintf(stderr,
            "[dual-repack] prepared_once source_type=%s gpu_mode=%s "
            "gpu_type=%s xmem_matmuls=%d K=%lld M=%lld NFF=%lld "
            "control=[%lld,%lld) gpu=[%lld,%lld) htp=[%lld,%lld) "
            "host_prepare_ms={control:%.3f,htp:%.3f,gpu:%.3f} "
            "backend_pack_ms={control:%.3f,htp:%.3f,gpu:%.3f} "
            "xmem_prime_ms=%.3f "
            "weight_mib={control:%.2f,htp:%.2f,gpu:%.2f,"
            "xmem_cache:%.2f,dual:%.2f}\n",
            ggml_type_name(cfg.weight_type),
            gpu_mode_name(cfg.gpu_mode),
            ggml_type_name(gpu_resident_type),
            xmem_matmuls,
            static_cast<long long>(cfg.k),
            static_cast<long long>(cfg.batch),
            static_cast<long long>(cfg.n_ff),
            static_cast<long long>(total_offset),
            static_cast<long long>(cfg.n_ff),
            static_cast<long long>(gpu_offset),
            static_cast<long long>(gpu_offset + cfg.gpu_columns),
            static_cast<long long>(htp_offset),
            static_cast<long long>(htp_offset + cfg.htp_columns),
            htp_control.host_prepare_ms, htp_shard.host_prepare_ms,
            gpu_shard.host_prepare_ms,
            htp_control.pack_ms, htp_shard.pack_ms, gpu_shard.pack_ms,
            gpu_shard.prime_ms,
            static_cast<double>(htp_control.weight_bytes) / (1024.0 * 1024.0),
            static_cast<double>(htp_shard.weight_bytes) / (1024.0 * 1024.0),
            static_cast<double>(gpu_shard.weight_bytes) / (1024.0 * 1024.0),
            static_cast<double>(xmem_cache_bytes) / (1024.0 * 1024.0),
            static_cast<double>(dual_resident_bytes) / (1024.0 * 1024.0));

    auto cleanup = [&]() {
        destroy_engine(gpu_shard);
        destroy_engine(htp_shard);
        destroy_engine(htp_control);
        if (io.gpu_buffer != nullptr) {
            ggml_backend_buffer_free(io.gpu_buffer);
        }
        if (io.gpu_ctx != nullptr) {
            ggml_free(io.gpu_ctx);
        }
        if (io.htp_buffer != nullptr) {
            ggml_backend_buffer_free(io.htp_buffer);
        }
        if (io.htp_ctx != nullptr) {
            ggml_free(io.htp_ctx);
        }
        ggml_backend_free(gpu.backend);
        ggml_backend_free(htp.backend);
    };

    if (cfg.benchmark != benchmark_mode::all) {
        bool selected_ok = false;
        std::vector<double> selected_samples;
        std::vector<double> pack_samples;
        const char * label = benchmark_mode_name(cfg.benchmark);
        const bool repack_mode =
                cfg.benchmark == benchmark_mode::repack_htp_control ||
                cfg.benchmark == benchmark_mode::repack_gpu_shard;
        const bool reconstruct_mode =
                cfg.benchmark == benchmark_mode::reconstruct_gpu_shard;
        const bool prepare_mode =
                cfg.benchmark == benchmark_mode::prepare_gpu_xmem;
        if (prepare_mode) {
            selected_ok = cfg.gpu_mode == gpu_weight_mode::f16_xmem &&
                    xmem_matmuls == 3;
            std::vector<double> host_prepare_samples;
            std::vector<double> prime_samples;
            std::vector<ffn_engine> prepared_engines;
            prepared_engines.reserve(
                    static_cast<size_t>(cfg.warmup + cfg.iterations));
            size_t prepared_bytes = 0;
            for (int i = 0; selected_ok && i < cfg.warmup; ++i) {
                double host_prepare_ms = 0.0;
                double pack_ms = 0.0;
                double prime_ms = 0.0;
                double total_ms = 0.0;
                prepared_engines.emplace_back();
                selected_ok = fresh_xmem_prepare(
                        cfg, gpu, io.gpu_input_alias, io.gpu_output_alias,
                        gpu_offset, prepared_engines.back(), host_prepare_ms,
                        pack_ms, prime_ms, total_ms, prepared_bytes);
            }
            printf("PHONE_ENERGY_WINDOW_START phone_uptime_ns=%llu label=%s\n",
                    static_cast<unsigned long long>(boot_time_ns()), label);
            fflush(stdout);
            for (int i = 0; selected_ok && i < cfg.iterations; ++i) {
                double host_prepare_ms = 0.0;
                double pack_ms = 0.0;
                double prime_ms = 0.0;
                double total_ms = 0.0;
                prepared_engines.emplace_back();
                selected_ok = fresh_xmem_prepare(
                        cfg, gpu, io.gpu_input_alias, io.gpu_output_alias,
                        gpu_offset, prepared_engines.back(), host_prepare_ms,
                        pack_ms, prime_ms, total_ms, prepared_bytes);
                if (selected_ok) {
                    host_prepare_samples.push_back(host_prepare_ms);
                    pack_samples.push_back(pack_ms);
                    prime_samples.push_back(prime_ms);
                    selected_samples.push_back(total_ms);
                }
            }
            printf("PHONE_ENERGY_WINDOW_END phone_uptime_ns=%llu label=%s\n",
                    static_cast<unsigned long long>(boot_time_ns()), label);
            printf(
                    "PHONE_PREPARE_RESULT status=%s benchmark=%s "
                    "gpu_mode=%s resident_type=f16 bytes=%zu iterations=%d "
                    "total_p50_ms=%.6f total_p90_ms=%.6f "
                    "host_prepare_p50_ms=%.6f pack_p50_ms=%.6f "
                    "prime_p50_ms=%.6f prime_p90_ms=%.6f\n",
                    selected_ok ? "PASS" : "FAIL", label,
                    gpu_mode_name(cfg.gpu_mode), prepared_bytes,
                    cfg.iterations,
                    percentile(selected_samples, 0.50),
                    percentile(selected_samples, 0.90),
                    percentile(host_prepare_samples, 0.50),
                    percentile(pack_samples, 0.50),
                    percentile(prime_samples, 0.50),
                    percentile(prime_samples, 0.90));
            for (ffn_engine & engine : prepared_engines) {
                destroy_engine(engine);
            }
        } else if (repack_mode || reconstruct_mode) {
            ffn_engine & engine =
                    cfg.benchmark == benchmark_mode::repack_htp_control
                    ? htp_control : gpu_shard;
            const backend_state & selected_backend =
                    cfg.benchmark == benchmark_mode::repack_htp_control
                    ? htp : gpu;
            selected_ok = true;
            for (int i = 0; i < cfg.warmup; ++i) {
                if (reconstruct_mode) {
                    selected_ok = reconstruct(cfg, engine) >= 0.0;
                } else {
                    double pack_ms = 0.0;
                    double total_ms = 0.0;
                    selected_ok = fresh_repack(
                            cfg, selected_backend, engine, pack_ms, total_ms);
                }
                if (!selected_ok) {
                    break;
                }
            }
            printf("PHONE_ENERGY_WINDOW_START phone_uptime_ns=%llu label=%s\n",
                    static_cast<unsigned long long>(boot_time_ns()), label);
            fflush(stdout);
            for (int i = 0; i < cfg.iterations; ++i) {
                double elapsed = 0.0;
                double pack_ms = 0.0;
                if (reconstruct_mode) {
                    elapsed = reconstruct(cfg, engine);
                } else if (!fresh_repack(
                            cfg, selected_backend, engine, pack_ms, elapsed)) {
                    elapsed = -1.0;
                }
                if (elapsed < 0.0) {
                    selected_ok = false;
                    break;
                }
                selected_samples.push_back(elapsed);
                if (!reconstruct_mode) {
                    pack_samples.push_back(pack_ms);
                }
            }
            printf("PHONE_ENERGY_WINDOW_END phone_uptime_ns=%llu label=%s\n",
                    static_cast<unsigned long long>(boot_time_ns()), label);
            printf(
                    "PHONE_REPACK_RESULT status=%s benchmark=%s "
                    "gpu_mode=%s resident_type=%s bytes=%zu iterations=%d "
                    "total_p50_ms=%.6f total_p90_ms=%.6f "
                    "pack_p50_ms=%.6f pack_p90_ms=%.6f\n",
                    selected_ok ? "PASS" : "FAIL", label,
                    gpu_mode_name(cfg.gpu_mode),
                    ggml_type_name(engine.resident_type), engine.weight_bytes,
                    cfg.iterations,
                    selected_samples.empty()
                            ? 0.0 : percentile(selected_samples, 0.50),
                    selected_samples.empty()
                            ? 0.0 : percentile(selected_samples, 0.90),
                    pack_samples.empty()
                            ? 0.0 : percentile(pack_samples, 0.50),
                    pack_samples.empty()
                            ? 0.0 : percentile(pack_samples, 0.90));
        } else if (cfg.benchmark == benchmark_mode::htp_control) {
            selected_samples = benchmark_solo(
                    htp_control, io.input, input_data, cfg.warmup,
                    cfg.iterations, UINT64_C(100000), label, selected_ok);
        } else if (cfg.benchmark == benchmark_mode::htp_shard) {
            selected_samples = benchmark_solo(
                    htp_shard, io.input, input_data, cfg.warmup,
                    cfg.iterations, UINT64_C(200000), label, selected_ok);
        } else if (cfg.benchmark == benchmark_mode::gpu_shard) {
            selected_samples = benchmark_solo(
                    gpu_shard, io.input, input_data, cfg.warmup,
                    cfg.iterations, UINT64_C(300000), label, selected_ok);
        } else {
            const float * htp_output =
                    static_cast<const float *>(io.htp_output->data);
            const float * gpu_output =
                    static_cast<const float *>(io.gpu_output->data);
            dual_executor executor(htp_shard, gpu_shard);
            const std::vector<dual_sample> samples = benchmark_dual(
                    executor, io.input, htp_output, gpu_output, input_data,
                    merged, cfg.warmup, cfg.iterations, UINT64_C(400000),
                    label, selected_ok);
            for (const dual_sample & sample : samples) {
                selected_samples.push_back(sample.total_ms);
            }
        }
        const bool pass = selected_ok &&
                selected_samples.size() == static_cast<size_t>(cfg.iterations);
        if (!repack_mode && !reconstruct_mode && !prepare_mode) {
            printf(
                    "PHONE_ENGINE_RESULT status=%s benchmark=%s gpu_mode=%s "
                    "K=%lld M=%lld NFF=%lld total_cols=%lld htp_cols=%lld "
                    "gpu_cols=%lld iterations=%d p50_ms=%.6f p90_ms=%.6f "
                    "pack_once_control_ms=%.6f pack_once_htp_ms=%.6f "
                    "pack_once_gpu_ms=%.6f xmem_prime_ms=%.6f\n",
                    pass ? "PASS" : "FAIL", label,
                    gpu_mode_name(cfg.gpu_mode),
                    static_cast<long long>(cfg.k),
                    static_cast<long long>(cfg.batch),
                    static_cast<long long>(cfg.n_ff),
                    static_cast<long long>(total_columns),
                    static_cast<long long>(cfg.htp_columns),
                    static_cast<long long>(cfg.gpu_columns), cfg.iterations,
                    pass ? percentile(selected_samples, 0.50) : 0.0,
                    pass ? percentile(selected_samples, 0.90) : 0.0,
                    htp_control.pack_ms, htp_shard.pack_ms, gpu_shard.pack_ms,
                    gpu_shard.prime_ms);
        }
        fflush(stdout);
        cleanup();
        return pass ? 0 : 1;
    }

    bool control_ok = false;
    bool htp_solo_ok = false;
    bool gpu_solo_ok = false;
    bool dual_ok = false;

    std::vector<double> control_samples = benchmark_solo(
            htp_control, io.input, input_data, cfg.warmup, cfg.iterations,
            UINT64_C(100000), nullptr, control_ok);
    std::vector<double> htp_solo_samples = benchmark_solo(
            htp_shard, io.input, input_data, cfg.warmup, cfg.iterations,
            UINT64_C(200000), nullptr, htp_solo_ok);
    std::vector<double> gpu_solo_samples = benchmark_solo(
            gpu_shard, io.input, input_data, cfg.warmup, cfg.iterations,
            UINT64_C(300000), nullptr, gpu_solo_ok);

    const float * htp_output = static_cast<const float *>(io.htp_output->data);
    const float * gpu_output = static_cast<const float *>(io.gpu_output->data);
    const float * control_output =
            static_cast<const float *>(io.control_output->data);
    std::vector<dual_sample> dual_samples;
    std::vector<float> reference;
    dual_sample correctness_sample;
    bool correctness_control_ok = false;
    {
        dual_executor executor(htp_shard, gpu_shard);
        dual_samples = benchmark_dual(
                executor, io.input, htp_output, gpu_output, input_data, merged,
                cfg.warmup, cfg.iterations, UINT64_C(400000), nullptr,
                dual_ok);

        fill_input(input_data, UINT64_C(0x5a17));
        ggml_backend_tensor_set(
                io.input, input_data.data(), 0,
                input_data.size() * sizeof(float));
        correctness_control_ok = compute(htp_control);
        reference.assign(control_output, control_output + elements);
        correctness_sample = executor.run(
                htp_output, gpu_output, merged.data(), merged.size());
    }
    double rel_l2 = 0.0;
    double max_abs = 0.0;
    compare_outputs(
            reference.data(), merged.data(), merged.size(), rel_l2, max_abs);
    const bool correctness_ok = correctness_control_ok && correctness_sample.ok &&
            all_finite(reference.data(), reference.size()) &&
            all_finite(merged.data(), merged.size()) && rel_l2 <= 0.005;

    std::vector<double> dual_compute_samples;
    std::vector<double> dual_merge_samples;
    std::vector<double> dual_total_samples;
    std::vector<double> dual_htp_samples;
    std::vector<double> dual_gpu_samples;
    std::vector<double> launch_skew_samples;
    for (const dual_sample & sample : dual_samples) {
        dual_compute_samples.push_back(sample.compute_ms);
        dual_merge_samples.push_back(sample.merge_ms);
        dual_total_samples.push_back(sample.total_ms);
        dual_htp_samples.push_back(sample.htp_ms);
        dual_gpu_samples.push_back(sample.gpu_ms);
        launch_skew_samples.push_back(sample.launch_skew_ms);
    }

    const double control_p50 = percentile(control_samples, 0.50);
    const double control_p90 = percentile(control_samples, 0.90);
    const double htp_solo_p50 = percentile(htp_solo_samples, 0.50);
    const double gpu_solo_p50 = percentile(gpu_solo_samples, 0.50);
    const double dual_compute_p50 = percentile(dual_compute_samples, 0.50);
    const double dual_merge_p50 = percentile(dual_merge_samples, 0.50);
    const double dual_total_p50 = percentile(dual_total_samples, 0.50);
    const double dual_total_p90 = percentile(dual_total_samples, 0.90);
    const double dual_htp_p50 = percentile(dual_htp_samples, 0.50);
    const double dual_gpu_p50 = percentile(dual_gpu_samples, 0.50);
    const double launch_skew_p50 = percentile(launch_skew_samples, 0.50);
    const double speedup = dual_total_p50 > 0.0 ?
            control_p50 / dual_total_p50 : 0.0;
    const bool xmem_request_eligible =
            cfg.gpu_mode != gpu_weight_mode::f16_xmem || xmem_matmuls == 3;
    const bool functional_ok = control_ok && htp_solo_ok && gpu_solo_ok &&
            dual_ok && correctness_ok && xmem_request_eligible &&
            control_samples.size() ==
                    static_cast<size_t>(cfg.iterations) &&
            dual_samples.size() == static_cast<size_t>(cfg.iterations);
    const char * verdict = !xmem_request_eligible ? "XMEM_INELIGIBLE" :
            !functional_ok ? "FAIL" :
            speedup > 1.0 ? "VALID_SPEEDUP" : "VALID_NO_SPEEDUP";

    printf(
            "DUAL_REPACK_RESULT verdict=%s source_type=%s gpu_mode=%s "
            "gpu_type=%s xmem_matmuls=%d K=%lld M=%lld NFF=%lld "
            "total_cols=%lld htp_cols=%lld gpu_cols=%lld "
            "host_prepare_control_ms=%.6f host_prepare_htp_ms=%.6f "
            "host_prepare_gpu_ms=%.6f "
            "pack_once_control_ms=%.6f pack_once_htp_ms=%.6f "
            "pack_once_gpu_ms=%.6f xmem_prime_ms=%.6f "
            "htp_control_weight_mib=%.6f htp_weight_mib=%.6f "
            "gpu_weight_mib=%.6f xmem_cache_mib=%.6f "
            "dual_resident_mib=%.6f "
            "htp_control_p50_ms=%.6f htp_control_p90_ms=%.6f "
            "htp_shard_solo_p50_ms=%.6f gpu_shard_solo_p50_ms=%.6f "
            "dual_htp_p50_ms=%.6f dual_gpu_p50_ms=%.6f "
            "dual_compute_p50_ms=%.6f phone_merge_p50_ms=%.6f "
            "dual_total_p50_ms=%.6f dual_total_p90_ms=%.6f "
            "launch_skew_p50_ms=%.6f speedup_vs_htp=%.6f "
            "rel_l2=%.9g max_abs=%.9g pack_calls_htp=6 "
            "pack_calls_gpu=3 iterations=%d\n",
            verdict, ggml_type_name(cfg.weight_type),
            gpu_mode_name(cfg.gpu_mode), ggml_type_name(gpu_resident_type),
            xmem_matmuls,
            static_cast<long long>(cfg.k),
            static_cast<long long>(cfg.batch),
            static_cast<long long>(cfg.n_ff),
            static_cast<long long>(total_columns),
            static_cast<long long>(cfg.htp_columns),
            static_cast<long long>(cfg.gpu_columns),
            htp_control.host_prepare_ms, htp_shard.host_prepare_ms,
            gpu_shard.host_prepare_ms,
            htp_control.pack_ms, htp_shard.pack_ms, gpu_shard.pack_ms,
            gpu_shard.prime_ms,
            static_cast<double>(htp_control.weight_bytes) / (1024.0 * 1024.0),
            static_cast<double>(htp_shard.weight_bytes) / (1024.0 * 1024.0),
            static_cast<double>(gpu_shard.weight_bytes) / (1024.0 * 1024.0),
            static_cast<double>(xmem_cache_bytes) / (1024.0 * 1024.0),
            static_cast<double>(dual_resident_bytes) / (1024.0 * 1024.0),
            control_p50, control_p90, htp_solo_p50, gpu_solo_p50,
            dual_htp_p50, dual_gpu_p50, dual_compute_p50, dual_merge_p50,
            dual_total_p50, dual_total_p90, launch_skew_p50, speedup,
            rel_l2, max_abs, cfg.iterations);
    fflush(stdout);

    cleanup();
    return functional_ok ? 0 : 1;
}
