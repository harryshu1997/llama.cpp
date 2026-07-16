#include "phone_pim_ffn.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

extern "C" {
#include "sha256.h"
}

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace phone_pim {
namespace {

uint64_t now_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool checked_product(uint64_t a, uint64_t b, uint64_t & result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

struct HostTensor {
    std::vector<uint8_t> bytes;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    ggml_type type = GGML_TYPE_COUNT;
};

struct GgufReader {
    int fd = -1;
    ggml_context * metadata = nullptr;
    gguf_context * gguf = nullptr;
    struct stat opened_stat = {};

    ~GgufReader() {
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

    bool open_and_verify(const FfnSpec & spec, uint64_t & verify_us, std::string & error) {
        const uint64_t start = now_us();
        fd = spec.model_fd >= 0
                ? fcntl(spec.model_fd, F_DUPFD_CLOEXEC, 0)
                : open(spec.model_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0 || fstat(fd, &opened_stat) != 0 || opened_stat.st_size < 0) {
            error = "cannot open model for verified FFN load: " + std::string(std::strerror(errno));
            return false;
        }
        if (static_cast<uint64_t>(opened_stat.st_size) != spec.expected_model_bytes) {
            error = "model byte size does not match PREPARE identity";
            return false;
        }
        if (spec.model_fd_verified) {
            if (spec.model_fd < 0) {
                error = "verified model source requires an open descriptor";
                return false;
            }
            verify_us = now_us() - start;
            return true;
        }

        sha256_t hash;
        sha256_init(&hash);
        std::array<uint8_t, 1024 * 1024> buffer = {};
        uint64_t offset = 0;
        while (offset < spec.expected_model_bytes) {
            const size_t wanted = static_cast<size_t>(std::min<uint64_t>(
                    buffer.size(), spec.expected_model_bytes - offset));
            const ssize_t count = pread(fd, buffer.data(), wanted, static_cast<off_t>(offset));
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count <= 0) {
                error = "short model read during SHA-256 verification";
                return false;
            }
            sha256_update(&hash, buffer.data(), static_cast<size_t>(count));
            offset += static_cast<uint64_t>(count);
        }
        std::array<uint8_t, 32> actual_sha256 = {};
        sha256_final(&hash, actual_sha256.data());
        verify_us = now_us() - start;
        if (actual_sha256 != spec.expected_model_sha256) {
            error = "model SHA-256 does not match PREPARE identity";
            return false;
        }
        return true;
    }

    bool load_metadata(std::string & error) {
        const std::string fd_path = "/proc/self/fd/" + std::to_string(fd);
        gguf_init_params params = { true, &metadata };
        gguf = gguf_init_from_file(fd_path.c_str(), params);
        if (gguf == nullptr || metadata == nullptr) {
            error = "cannot parse verified GGUF metadata";
            return false;
        }
        struct stat current = {};
        if (fstat(fd, &current) != 0 || current.st_dev != opened_stat.st_dev ||
            current.st_ino != opened_stat.st_ino || current.st_size != opened_stat.st_size ||
            current.st_mtim.tv_sec != opened_stat.st_mtim.tv_sec ||
            current.st_mtim.tv_nsec != opened_stat.st_mtim.tv_nsec) {
            error = "model inode changed while loading metadata";
            return false;
        }
        return true;
    }

    bool unchanged(std::string & error) const {
        struct stat current = {};
        if (fstat(fd, &current) != 0 || current.st_dev != opened_stat.st_dev ||
            current.st_ino != opened_stat.st_ino || current.st_size != opened_stat.st_size ||
            current.st_mtim.tv_sec != opened_stat.st_mtim.tv_sec ||
            current.st_mtim.tv_nsec != opened_stat.st_mtim.tv_nsec) {
            error = "model inode changed while loading tensors";
            return false;
        }
        return true;
    }
};

bool read_tensor(GgufReader & reader, const std::string & name, HostTensor & out, std::string & error) {
    const int64_t tensor_id = gguf_find_tensor(reader.gguf, name.c_str());
    ggml_tensor * tensor = tensor_id >= 0 ? ggml_get_tensor(reader.metadata, name.c_str()) : nullptr;
    if (tensor_id < 0 || tensor == nullptr) {
        error = "tensor not found: " + name;
        return false;
    }
    out.ne0 = tensor->ne[0];
    out.ne1 = tensor->ne[1];
    out.type = tensor->type;
    const size_t byte_count = ggml_nbytes(tensor);
    const size_t data_offset = gguf_get_data_offset(reader.gguf);
    const size_t tensor_offset = gguf_get_tensor_offset(reader.gguf, tensor_id);
    if (tensor_offset > std::numeric_limits<size_t>::max() - data_offset) {
        error = "tensor file offset overflow: " + name;
        return false;
    }
    out.bytes.resize(byte_count);
    size_t completed = 0;
    const off_t offset = static_cast<off_t>(data_offset + tensor_offset);
    while (completed < byte_count) {
        const ssize_t count = pread(
                reader.fd, out.bytes.data() + completed, byte_count - completed, offset + completed);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            break;
        }
        completed += static_cast<size_t>(count);
    }
    if (completed != byte_count) {
        error = "short tensor read: " + name;
        return false;
    }
    return true;
}

bool norm_as_f32(const HostTensor & tensor, int64_t count, std::vector<float> & output, std::string & error) {
    if (count <= 0 || tensor.ne0 != count || tensor.ne1 != 1) {
        error = "invalid norm tensor shape";
        return false;
    }
    output.resize(static_cast<size_t>(count));
    if (tensor.type == GGML_TYPE_F32) {
        std::memcpy(output.data(), tensor.bytes.data(), output.size() * sizeof(float));
        return true;
    }
    if (tensor.type == GGML_TYPE_F16) {
        const ggml_fp16_t * source = reinterpret_cast<const ggml_fp16_t *>(tensor.bytes.data());
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] = ggml_fp16_to_fp32(source[i]);
        }
        return true;
    }
    error = "norm tensor must be F16 or F32";
    return false;
}

} // namespace

struct FfnIsland::Impl {
    FfnSpec spec;
    std::string backend_description;
    ggml_backend_t backend = nullptr;

    ggml_context * weight_context = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_tensor * gate_weight = nullptr;
    ggml_tensor * up_weight = nullptr;
    ggml_tensor * down_weight = nullptr;
    ggml_tensor * norm_weight = nullptr;
    ggml_tensor * post_norm_weight = nullptr;

    ggml_context * graph_context = nullptr;
    ggml_backend_buffer_t graph_buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;

    int64_t n_embd = 0;
    int64_t n_ff = 0;
    uint64_t elements = 0;
    std::vector<float> norm_gain;

    ~Impl() {
        if (graph_buffer != nullptr) {
            ggml_backend_buffer_free(graph_buffer);
        }
        if (graph_context != nullptr) {
            ggml_free(graph_context);
        }
        if (weight_buffer != nullptr) {
            ggml_backend_buffer_free(weight_buffer);
        }
        if (weight_context != nullptr) {
            ggml_free(weight_context);
        }
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

FfnIsland::FfnIsland() = default;
FfnIsland::~FfnIsland() = default;

bool FfnIsland::prepare(const FfnSpec & spec, FfnPrepareMetrics & metrics, std::string & error) {
    metrics = {};
    reset();
    if ((spec.model_fd < 0 && spec.model_path.empty()) || spec.tensor_prefix.empty() ||
        spec.backend_name.empty() ||
        spec.token_count == 0 || spec.expected_model_bytes == 0) {
        error = "incomplete FFN specification";
        return false;
    }

    auto candidate = std::make_unique<Impl>();
    candidate->spec = spec;
    const std::vector<std::string> names = {
        spec.tensor_prefix + ".ffn_gate.weight",
        spec.tensor_prefix + ".ffn_up.weight",
        spec.tensor_prefix + ".ffn_down.weight",
        spec.tensor_prefix + ".ffn_norm.weight",
        spec.tensor_prefix + ".post_ffw_norm.weight",
    };

    GgufReader reader;
    if (!reader.open_and_verify(spec, metrics.verify_us, error)) {
        return false;
    }
    const uint64_t load_start = now_us();
    if (!reader.load_metadata(error)) {
        return false;
    }
    const int64_t architecture_key = gguf_find_key(reader.gguf, "general.architecture");
    const int64_t epsilon_key = gguf_find_key(reader.gguf, "gemma4.attention.layer_norm_rms_epsilon");
    if (architecture_key < 0 || gguf_get_kv_type(reader.gguf, architecture_key) != GGUF_TYPE_STRING ||
        std::strcmp(gguf_get_val_str(reader.gguf, architecture_key), "gemma4") != 0 ||
        epsilon_key < 0 || gguf_get_kv_type(reader.gguf, epsilon_key) != GGUF_TYPE_FLOAT32 ||
        std::fabs(gguf_get_val_f32(reader.gguf, epsilon_key) - 1e-6f) > 1e-12f) {
        error = "FFN capability requires Gemma4 with RMS epsilon 1e-6";
        return false;
    }
    if (gguf_find_tensor(reader.gguf, (spec.tensor_prefix + ".ffn_gate_inp.weight").c_str()) >= 0) {
        error = "FFN capability does not support a Gemma4 MoE layer";
        return false;
    }
    HostTensor gate;
    HostTensor up;
    HostTensor down;
    HostTensor norm;
    HostTensor post_norm;
    if (!read_tensor(reader, names[0], gate, error) ||
        !read_tensor(reader, names[1], up, error) ||
        !read_tensor(reader, names[2], down, error) ||
        !read_tensor(reader, names[3], norm, error) ||
        !read_tensor(reader, names[4], post_norm, error) ||
        !reader.unchanged(error)) {
        return false;
    }
    metrics.load_us = now_us() - load_start;

    if (gate.type != GGML_TYPE_F16 || up.type != GGML_TYPE_F16 || down.type != GGML_TYPE_F16 ||
        gate.ne0 <= 0 || gate.ne1 <= 0 || up.ne0 != gate.ne0 || up.ne1 != gate.ne1 ||
        down.ne0 != gate.ne1 || down.ne1 != gate.ne0) {
        error = "FFN projection tensor type or shape mismatch";
        return false;
    }
    candidate->n_embd = gate.ne0;
    candidate->n_ff = gate.ne1;
    if (!norm_as_f32(norm, candidate->n_embd, candidate->norm_gain, error)) {
        return false;
    }
    std::vector<float> post_norm_f32;
    if (!norm_as_f32(post_norm, candidate->n_embd, post_norm_f32, error)) {
        return false;
    }
    if (!all_finite(candidate->norm_gain.data(), candidate->norm_gain.size()) ||
        !all_finite(post_norm_f32.data(), post_norm_f32.size())) {
        error = "non-finite norm tensor";
        return false;
    }
    if (!checked_product(static_cast<uint64_t>(candidate->n_embd), spec.token_count, candidate->elements) ||
        candidate->elements > SIZE_MAX / sizeof(float)) {
        error = "FFN activation shape overflow";
        return false;
    }

    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name(spec.backend_name.c_str());
    if (device == nullptr) {
        error = "backend not found: " + spec.backend_name;
        return false;
    }
    candidate->backend = ggml_backend_dev_init(device, nullptr);
    if (candidate->backend == nullptr) {
        error = "backend initialization failed: " + spec.backend_name;
        return false;
    }
    candidate->backend_description = ggml_backend_dev_description(device);

    const uint64_t upload_start = now_us();
    ggml_init_params weight_params = { ggml_tensor_overhead() * 8, nullptr, true };
    candidate->weight_context = ggml_init(weight_params);
    if (candidate->weight_context == nullptr) {
        error = "weight metadata allocation failed";
        return false;
    }
    candidate->gate_weight = ggml_new_tensor_2d(candidate->weight_context, GGML_TYPE_F16, candidate->n_embd, candidate->n_ff);
    candidate->up_weight = ggml_new_tensor_2d(candidate->weight_context, GGML_TYPE_F16, candidate->n_embd, candidate->n_ff);
    candidate->down_weight = ggml_new_tensor_2d(candidate->weight_context, GGML_TYPE_F16, candidate->n_ff, candidate->n_embd);
    candidate->norm_weight = ggml_new_tensor_1d(candidate->weight_context, GGML_TYPE_F32, candidate->n_embd);
    candidate->post_norm_weight = ggml_new_tensor_1d(candidate->weight_context, GGML_TYPE_F32, candidate->n_embd);
    ggml_set_name(candidate->gate_weight, names[0].c_str());
    ggml_set_name(candidate->up_weight, names[1].c_str());
    ggml_set_name(candidate->down_weight, names[2].c_str());
    ggml_set_name(candidate->norm_weight, names[3].c_str());
    ggml_set_name(candidate->post_norm_weight, names[4].c_str());
    candidate->weight_buffer = ggml_backend_alloc_ctx_tensors(candidate->weight_context, candidate->backend);
    if (candidate->weight_buffer == nullptr) {
        error = "backend weight allocation failed";
        return false;
    }
    ggml_backend_tensor_set(candidate->gate_weight, gate.bytes.data(), 0, gate.bytes.size());
    ggml_backend_tensor_set(candidate->up_weight, up.bytes.data(), 0, up.bytes.size());
    ggml_backend_tensor_set(candidate->down_weight, down.bytes.data(), 0, down.bytes.size());
    ggml_backend_tensor_set(candidate->norm_weight, candidate->norm_gain.data(), 0, candidate->norm_gain.size() * sizeof(float));
    ggml_backend_tensor_set(candidate->post_norm_weight, post_norm_f32.data(), 0, post_norm_f32.size() * sizeof(float));
    ggml_backend_synchronize(candidate->backend);

    ggml_init_params graph_params = { ggml_tensor_overhead() * 40 + ggml_graph_overhead(), nullptr, true };
    candidate->graph_context = ggml_init(graph_params);
    if (candidate->graph_context == nullptr) {
        error = "graph metadata allocation failed";
        return false;
    }
    candidate->input = ggml_new_tensor_2d(
            candidate->graph_context, GGML_TYPE_F32, candidate->n_embd, spec.token_count);
    ggml_set_name(candidate->input, "phone_pim_input");
    ggml_tensor * normalized = ggml_mul(
            candidate->graph_context,
            ggml_rms_norm(candidate->graph_context, candidate->input, 1e-6f),
            candidate->norm_weight);
    ggml_tensor * gate_projection = ggml_mul_mat(candidate->graph_context, candidate->gate_weight, normalized);
    ggml_tensor * up_projection = ggml_mul_mat(candidate->graph_context, candidate->up_weight, normalized);
    ggml_tensor * activation = ggml_geglu_split(candidate->graph_context, gate_projection, up_projection);
    ggml_tensor * down_projection = ggml_mul_mat(candidate->graph_context, candidate->down_weight, activation);
    ggml_tensor * post_normalized = ggml_mul(
            candidate->graph_context,
            ggml_rms_norm(candidate->graph_context, down_projection, 1e-6f),
            candidate->post_norm_weight);
    candidate->output = ggml_add(candidate->graph_context, candidate->input, post_normalized);
    ggml_set_name(candidate->output, "phone_pim_output");
    candidate->graph = ggml_new_graph(candidate->graph_context);
    ggml_build_forward_expand(candidate->graph, candidate->output);
    for (int i = 0; i < ggml_graph_n_nodes(candidate->graph); ++i) {
        ggml_tensor * node = ggml_graph_node(candidate->graph, i);
        if (!ggml_backend_supports_op(candidate->backend, node)) {
            error = std::string("backend does not support FFN op: ") + ggml_op_name(node->op);
            return false;
        }
    }
    candidate->graph_buffer = ggml_backend_alloc_ctx_tensors(candidate->graph_context, candidate->backend);
    if (candidate->graph_buffer == nullptr) {
        error = "backend activation allocation failed";
        return false;
    }
    metrics.upload_us = now_us() - upload_start;
    metrics.weight_bytes = gate.bytes.size() + up.bytes.size() + down.bytes.size() +
                           candidate->norm_gain.size() * sizeof(float) + post_norm_f32.size() * sizeof(float);
    metrics.resident_buffer_bytes = ggml_backend_buffer_get_size(candidate->weight_buffer) +
                                    ggml_backend_buffer_get_size(candidate->graph_buffer);

    impl_ = std::move(candidate);
    const std::vector<float> warm_input = make_test_input(0x70696d31U);
    std::vector<float> warm_output;
    FfnExecuteMetrics warm_metrics;
    const uint64_t warm_start = now_us();
    if (!execute(warm_input, warm_output, warm_metrics, error)) {
        reset();
        return false;
    }
    metrics.warmup_us = now_us() - warm_start;
    if (!all_finite(warm_output.data(), warm_output.size())) {
        error = "warmup produced non-finite output";
        reset();
        return false;
    }
    return true;
}

bool FfnIsland::execute(
        const std::vector<float> & input,
        std::vector<float> & output,
        FfnExecuteMetrics & metrics,
        std::string & error) {
    metrics = {};
    if (!ready()) {
        error = "FFN island is not ready";
        return false;
    }
    if (input.size() != impl_->elements || !all_finite(input.data(), input.size())) {
        error = "invalid FFN input";
        return false;
    }
    output.resize(input.size());
    uint64_t start = now_us();
    ggml_backend_tensor_set(impl_->input, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_synchronize(impl_->backend);
    metrics.input_set_us = now_us() - start;

    start = now_us();
    const ggml_status status = ggml_backend_graph_compute(impl_->backend, impl_->graph);
    ggml_backend_synchronize(impl_->backend);
    metrics.compute_us = now_us() - start;
    if (status != GGML_STATUS_SUCCESS) {
        error = "backend graph compute failed with status " + std::to_string(static_cast<int>(status));
        return false;
    }

    start = now_us();
    ggml_backend_tensor_get(impl_->output, output.data(), 0, output.size() * sizeof(float));
    metrics.output_get_us = now_us() - start;
    if (!all_finite(output.data(), output.size())) {
        error = "FFN output is non-finite";
        return false;
    }
    return true;
}

void FfnIsland::reset() {
    impl_.reset();
}

bool FfnIsland::ready() const {
    return impl_ != nullptr && impl_->graph != nullptr && impl_->graph_buffer != nullptr;
}

uint64_t FfnIsland::input_elements() const {
    return ready() ? impl_->elements : 0;
}

uint64_t FfnIsland::output_elements() const {
    return input_elements();
}

uint64_t FfnIsland::n_embd() const {
    return ready() ? static_cast<uint64_t>(impl_->n_embd) : 0;
}

uint64_t FfnIsland::n_ff() const {
    return ready() ? static_cast<uint64_t>(impl_->n_ff) : 0;
}

uint32_t FfnIsland::token_count() const {
    return ready() ? impl_->spec.token_count : 0;
}

const std::string & FfnIsland::backend_name() const {
    static const std::string empty;
    return ready() ? impl_->spec.backend_name : empty;
}

const std::string & FfnIsland::backend_description() const {
    static const std::string empty;
    return ready() ? impl_->backend_description : empty;
}

const std::string & FfnIsland::tensor_prefix() const {
    static const std::string empty;
    return ready() ? impl_->spec.tensor_prefix : empty;
}

std::vector<float> FfnIsland::make_test_input(uint32_t seed) const {
    if (!ready()) {
        return {};
    }
    std::vector<float> input(static_cast<size_t>(impl_->elements));
    for (uint32_t token = 0; token < impl_->spec.token_count; ++token) {
        const uint32_t token_seed = seed + token * 2654435761U;
        for (int64_t i = 0; i < impl_->n_embd; ++i) {
            const float base = 0.5f * static_cast<float>((static_cast<int64_t>(i + token_seed) % 17) - 8) / 8.0f;
            const float gain = std::max(std::fabs(impl_->norm_gain[static_cast<size_t>(i)]), 0.05f);
            input[static_cast<size_t>(token) * impl_->n_embd + i] = base / gain;
        }
    }
    return input;
}

bool all_finite(const float * data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, data + i, sizeof(bits));
        if ((bits & 0x7f800000U) == 0x7f800000U) {
            return false;
        }
    }
    return true;
}

double relative_l2(const std::vector<float> & actual, const std::vector<float> & expected, bool & finite) {
    finite = actual.size() == expected.size() &&
             all_finite(actual.data(), actual.size()) &&
             all_finite(expected.data(), expected.size());
    if (!finite || actual.empty()) {
        return 1.0;
    }
    double error_squared = 0.0;
    double reference_squared = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double difference = static_cast<double>(actual[i]) - expected[i];
        error_squared += difference * difference;
        reference_squared += static_cast<double>(expected[i]) * expected[i];
    }
    if (reference_squared == 0.0) {
        return error_squared == 0.0 ? 0.0 : 1.0;
    }
    return std::sqrt(error_squared / reference_squared);
}

} // namespace phone_pim
