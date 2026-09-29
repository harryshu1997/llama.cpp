#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-stream.h"
#include "server-tools.h"
#include "server-warm-tier-runtime.h"

#if defined(S41_SERVER_FFN_SPLIT)
#include "examples/layersplit/ffn-split-client.h"
#include "src/llama-ffn-split-policy.h"
#endif

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <clocale>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <signal.h>
#include <string>
#include <thread> // for std::thread::hardware_concurrency
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

#if defined(S41_SERVER_FFN_SPLIT)
namespace {

bool s41_parse_u64_env(const char * name, uint64_t & value) {
    const char * text = std::getenv(name);
    if (text == nullptr || text[0] == '\0') {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    value = static_cast<uint64_t>(parsed);
    return true;
}

bool s41_set_env(const char * name, const std::string & value) {
#if defined(_WIN32)
    return _putenv_s(name, value.c_str()) == 0;
#else
    return setenv(name, value.c_str(), 1) == 0;
#endif
}

// transport of one FFN helper phone: <prefix>TRANSPORT plus <prefix>HOST/<prefix>PORT (tcp) or
// the <prefix>USB_* contract (functionfs-usb); the legacy single-helper prefix is S41_SERVER_FFN_
struct s41_ffn_transport_env {
    bool functionfs_usb = false;
    std::string host;
    uint64_t port = 0;
    const char * usb_allocator = nullptr;
    const char * usb_generation = nullptr;
    const char * usb_batch_plan = nullptr;
    uint64_t usb_queue_depth = 0;
    uint64_t usbfs_available_bytes = 0;
    uint64_t usb_slot_safety_bytes = 0;
    uint64_t usb_max_payload_bytes = 0;
    uint64_t usb_vendor_id = 0;
    uint64_t usb_product_id = 0;
    uint64_t usb_split_h2d = 0;
    uint64_t usb_full_duplex = 0;
};

bool s41_parse_transport_env(const std::string & prefix, s41_ffn_transport_env & out, std::string & error) {
    const auto env = [&prefix](const char * suffix) { return std::getenv((prefix + suffix).c_str()); };
    const auto u64 = [&prefix](const char * suffix, uint64_t & value) {
        return s41_parse_u64_env((prefix + suffix).c_str(), value);
    };
    const char * transport_text = env("TRANSPORT");
    const char * host = env("HOST");
    out.functionfs_usb = transport_text != nullptr &&
            std::strcmp(transport_text, "functionfs-usb") == 0;
    const bool tcp = transport_text == nullptr ||
            std::strcmp(transport_text, "tcp") == 0;
    if ((!tcp && !out.functionfs_usb) ||
        (tcp && (host == nullptr || host[0] == '\0'))) {
        error = "invalid or conflicting server FFN split configuration";
        return false;
    }
    out.host = host == nullptr ? "" : host;
    if (tcp && (!u64("PORT", out.port) || out.port == 0 || out.port > 65535)) {
        error = "invalid server FFN split environment";
        return false;
    }
    if (!out.functionfs_usb) {
        return true;
    }
    out.usb_allocator = env("USB_ALLOCATOR");
    out.usb_generation = env("USB_TRANSPORT_GENERATION");
    out.usb_batch_plan = env("USB_BATCH_PLAN");
    if (out.usb_allocator == nullptr ||
        (std::strcmp(out.usb_allocator, "malloc") != 0 &&
         std::strcmp(out.usb_allocator, "devmem") != 0) ||
        out.usb_generation == nullptr || out.usb_generation[0] == '\0' ||
        out.usb_batch_plan == nullptr ||
        (std::strcmp(out.usb_batch_plan, "coalesced-batch") != 0 &&
         std::strcmp(out.usb_batch_plan, "split-row") != 0) ||
        !u64("USB_QUEUE_DEPTH", out.usb_queue_depth) ||
        !u64("USBFS_AVAILABLE_BYTES", out.usbfs_available_bytes) ||
        !u64("USB_SLOT_SAFETY_BYTES", out.usb_slot_safety_bytes) ||
        !u64("USB_MAX_PAYLOAD_BYTES", out.usb_max_payload_bytes) ||
        !u64("USB_VENDOR_ID", out.usb_vendor_id) ||
        !u64("USB_PRODUCT_ID", out.usb_product_id) ||
        !u64("USB_SPLIT_H2D", out.usb_split_h2d) ||
        !u64("USB_FULL_DUPLEX", out.usb_full_duplex) ||
        out.usb_queue_depth == 0 || out.usb_queue_depth > 64 ||
        out.usb_slot_safety_bytes == 0 ||
        out.usb_max_payload_bytes == 0 ||
        out.usb_vendor_id == 0 || out.usb_vendor_id > UINT16_MAX ||
        out.usb_product_id == 0 || out.usb_product_id > UINT16_MAX ||
        out.usb_split_h2d > 1 || out.usb_full_duplex > 1) {
        error = "invalid server FFN split USB environment";
        return false;
    }
    return true;
}

// layer of an FFN split graph marker (ffn_norm-<il>, ffn_phone_partial-<il>)
bool s41_ffn_marker_layer(const char * name, int & layer) {
    for (const char * prefix : { "ffn_norm-", "ffn_phone_partial-" }) {
        const size_t size = std::strlen(prefix);
        if (name == nullptr || std::strncmp(name, prefix, size) != 0) {
            continue;
        }
        errno = 0;
        char * end = nullptr;
        const long parsed = std::strtol(name + size, &end, 10);
        if (errno != 0 || end == name + size || *end != '\0' || parsed < 0 || parsed >= 64) {
            return false;
        }
        layer = static_cast<int>(parsed);
        return true;
    }
    return false;
}

bool s41_valid_helper_label(const char * text) {
    const size_t size = text == nullptr ? 0 : std::strlen(text);
    return size > 0 && size <= 32 && std::all_of(text, text + size, [](char value) {
        return std::isalnum(static_cast<unsigned char>(value)) || value == '-' || value == '_';
    });
}

// one FFN split client per helper phone; every helper owns a disjoint subset of the layer mask
struct s41_ffn_helper {
    std::string label;
    bool functionfs_usb = false;
    uint64_t layer_mask = 0;
    std::unique_ptr<ffn_split::client> client;
    bool deferred = false;
    // sub-policy accepted last, restored when a later helper rejects a new union policy
    uint64_t applied_layer_mask = 0;
    uint32_t applied_columns = 0;
};

class s41_server_ffn_runtime {
public:
    ~s41_server_ffn_runtime() {
        finish();
    }

    bool init(common_params & params, std::string & error) {
        const char * host = std::getenv("S41_SERVER_FFN_HOST");
        const char * transport_text =
                std::getenv("S41_SERVER_FFN_TRANSPORT");
        const char * helpers_text = std::getenv("S41_SERVER_FFN_HELPERS");
        const char * artifact_sha256 =
                std::getenv("S41_SERVER_FFN_ARTIFACT_SHA256");
        if (host == nullptr && transport_text == nullptr && helpers_text == nullptr) {
            if (std::getenv("S41_SERVER_FFN_DORMANT_DROP_CACHE") || std::getenv("S41_SERVER_FFN_DORMANT_POPULATE") ||
                std::getenv("S41_SERVER_FFN_ROW_DIAGNOSTIC_STEPS")) {
                error = "dormant host share policy requires an FFN transport";
                return false;
            }
            return true;
        }
        // S41_SERVER_FFN_HELPERS=<n> configures one client per helper phone from the
        // S41_SERVER_FFN_HELPER<k>_{LABEL,LAYER_MASK,TRANSPORT,HOST,PORT,USB_*} variables
        const bool multi_format = helpers_text != nullptr;
        uint64_t helper_count = 1;
        if (multi_format &&
            (!s41_parse_u64_env("S41_SERVER_FFN_HELPERS", helper_count) ||
             helper_count == 0 || helper_count > 8 || host != nullptr ||
             transport_text != nullptr || std::getenv("S41_SERVER_FFN_PORT") != nullptr)) {
            error = "invalid server FFN helper count or a legacy transport next to S41_SERVER_FFN_HELPERS";
            return false;
        }
        std::vector<s41_ffn_transport_env> transports(helper_count);
        std::vector<std::string> labels;
        std::vector<std::string> prefixes;
        for (uint64_t index = 0; index < helper_count; ++index) {
            prefixes.push_back(multi_format ?
                    "S41_SERVER_FFN_HELPER" + std::to_string(index) + "_" : std::string("S41_SERVER_FFN_"));
            if (!s41_parse_transport_env(prefixes.back(), transports[index], error)) {
                return false;
            }
            const char * label = std::getenv((prefixes.back() + "LABEL").c_str());
            if (label != nullptr && (!s41_valid_helper_label(label) ||
                                     std::find(labels.begin(), labels.end(), label) != labels.end())) {
                error = "invalid or duplicate server FFN helper label";
                return false;
            }
            labels.push_back(label != nullptr ? label : "helper" + std::to_string(index));
        }
        // the host USB client opens the first device with the FunctionFS vendor and product id
        if (std::count_if(transports.begin(), transports.end(),
                    [](const s41_ffn_transport_env & value) { return value.functionfs_usb; }) > 1) {
            error = "at most one server FFN helper can use the functionfs-usb transport";
            return false;
        }
        if (artifact_sha256 == nullptr || artifact_sha256[0] == '\0' ||
            params.cb_eval != nullptr) {
            error = "invalid or conflicting server FFN split configuration";
            return false;
        }

        uint64_t n_embd = 0;
        uint64_t layer_mask = 0;
        uint64_t columns = 0;
        if (!s41_parse_u64_env("S41_SERVER_FFN_N_EMBD", n_embd) ||
            !s41_parse_u64_env("S41_SERVER_FFN_LAYER_MASK", layer_mask) ||
            !s41_parse_u64_env("S41_SERVER_FFN_COLUMNS", columns) ||
            n_embd == 0 ||
            n_embd > std::numeric_limits<uint32_t>::max() ||
            layer_mask == 0 || columns == 0 ||
            columns > std::numeric_limits<uint32_t>::max() ||
            params.n_ubatch <= 0) {
            error = "invalid server FFN split environment";
            return false;
        }
        std::vector<uint64_t> helper_masks(helper_count, layer_mask);
        if (multi_format) {
            uint64_t covered = 0;
            for (uint64_t index = 0; index < helper_count; ++index) {
                if (!s41_parse_u64_env((prefixes[index] + "LAYER_MASK").c_str(), helper_masks[index]) ||
                    helper_masks[index] == 0 || (helper_masks[index] & ~layer_mask) != 0 ||
                    (helper_masks[index] & covered) != 0) {
                    error = "server FFN helper layer masks must be nonempty, disjoint subsets of the layer mask";
                    return false;
                }
                covered |= helper_masks[index];
            }
            if (covered != layer_mask) {
                error = "server FFN helper layer masks do not cover the layer mask";
                return false;
            }
        }

        const bool has_shape_policy =
                std::getenv("S41_SERVER_FFN_M1_COLUMNS") != nullptr ||
                std::getenv("S41_SERVER_FFN_SMALL_M_COLUMNS") != nullptr ||
                std::getenv("S41_SERVER_FFN_LARGE_M_COLUMNS") != nullptr ||
                std::getenv("S41_SERVER_FFN_SMALL_M_MAX") != nullptr;
        const char * table_policy_text =
                std::getenv("S41_SERVER_FFN_POLICY");
        const bool has_table_policy =
                table_policy_text != nullptr && table_policy_text[0] != '\0';
        if (has_shape_policy && has_table_policy) {
            error = "conflicting server FFN split policies";
            return false;
        }
        uint64_t m1_columns = columns;
        uint64_t small_columns = columns;
        uint64_t large_columns = columns;
        uint64_t small_m_max = 0;
        if (has_shape_policy &&
            (!s41_parse_u64_env("S41_SERVER_FFN_M1_COLUMNS", m1_columns) ||
             !s41_parse_u64_env("S41_SERVER_FFN_SMALL_M_COLUMNS", small_columns) ||
             !s41_parse_u64_env("S41_SERVER_FFN_LARGE_M_COLUMNS", large_columns) ||
             !s41_parse_u64_env("S41_SERVER_FFN_SMALL_M_MAX", small_m_max) ||
             m1_columns > columns || small_columns > columns ||
             large_columns > columns || small_m_max == 0 ||
             small_m_max > static_cast<uint64_t>(params.n_ubatch))) {
            error = "invalid server FFN split shape policy";
            return false;
        }
        bool f16_io = false;
        if (const char * value = std::getenv("S41_SERVER_FFN_F16_IO")) {
            if (std::strcmp(value, "0") != 0 && std::strcmp(value, "1") != 0) {
                error = "S41_SERVER_FFN_F16_IO must be 0 or 1";
                return false;
            }
            f16_io = std::strcmp(value, "1") == 0;
        }
        bool swiglu = false;
        if (const char * value = std::getenv("S41_SERVER_FFN_ACTIVATION")) {
            if (std::strcmp(value, "geglu") == 0) {
                swiglu = false;
            } else if (std::strcmp(value, "swiglu") == 0) {
                swiglu = true;
            } else {
                error = "S41_SERVER_FFN_ACTIVATION must be geglu or swiglu";
                return false;
            }
        }
        bool runtime_control = false;
        if (const char * value = std::getenv(
                    "S41_SERVER_FFN_RUNTIME_CONTROL")) {
            if (std::strcmp(value, "0") != 0 &&
                std::strcmp(value, "1") != 0) {
                error = "S41_SERVER_FFN_RUNTIME_CONTROL must be 0 or 1";
                return false;
            }
            runtime_control = std::strcmp(value, "1") == 0;
        }
        bool dormant_host_share = false;
        if (const char * value = std::getenv("S41_SERVER_FFN_DORMANT_HOST_SHARE")) {
            if (std::strcmp(value, "0") != 0 && std::strcmp(value, "1") != 0) {
                error = "S41_SERVER_FFN_DORMANT_HOST_SHARE must be 0 or 1";
                return false;
            }
            dormant_host_share = std::strcmp(value, "1") == 0;
            if (dormant_host_share && !runtime_control) {
                error = "S41_SERVER_FFN_DORMANT_HOST_SHARE requires S41_SERVER_FFN_RUNTIME_CONTROL=1";
                return false;
            }
            if (dormant_host_share && !params.use_mmap) {
                error = "S41_SERVER_FFN_DORMANT_HOST_SHARE requires memory-mapped weights";
                return false;
            }
        }
        auto dormant_flag = [&](const char * name, bool & target) {
            const char * value = std::getenv(name);
            if (value == nullptr) {
                return true;
            }
            if (!dormant_host_share || (std::strcmp(value, "0") != 0 && std::strcmp(value, "1") != 0)) {
                error = std::string(name) + " requires dormant host share and a value of 0 or 1";
                return false;
            }
            target = std::strcmp(value, "1") == 0;
            return true;
        };
        if (!dormant_flag("S41_SERVER_FFN_DORMANT_DROP_CACHE", dormant_drop_cache_) ||
            !dormant_flag("S41_SERVER_FFN_DORMANT_POPULATE", dormant_populate_)) {
            return false;
        }
        uint64_t row_diagnostic_steps = 0;
        if (std::getenv("S41_SERVER_FFN_ROW_DIAGNOSTIC_STEPS") &&
            (!s41_parse_u64_env("S41_SERVER_FFN_ROW_DIAGNOSTIC_STEPS", row_diagnostic_steps) ||
             (row_diagnostic_steps != 0 && ((row_diagnostic_steps != 5 && row_diagnostic_steps != 64) || !dormant_host_share)))) {
            error = "FFN row diagnostic requires dormant runtime control and five or 64 steps";
            return false;
        }
        row_diagnostic_steps_ = static_cast<uint32_t>(row_diagnostic_steps);
        const uint64_t default_max_tokens = static_cast<uint64_t>(
                std::min(params.n_ubatch, 512));
        uint64_t max_tokens = default_max_tokens;
        if (std::getenv("S41_SERVER_FFN_MAX_TOKENS") != nullptr &&
            (!s41_parse_u64_env(
                     "S41_SERVER_FFN_MAX_TOKENS", max_tokens) ||
             max_tokens == 0 || max_tokens > default_max_tokens ||
             (!runtime_control && max_tokens != default_max_tokens))) {
            error = "invalid server FFN split maximum tokens";
            return false;
        }
        uint64_t remote_resident_layer_mask = 0;
        if (std::getenv("S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK") != nullptr &&
            !s41_parse_u64_env("S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK",
                    remote_resident_layer_mask)) {
            error = "invalid server FFN remote-resident layer mask";
            return false;
        }
        if ((remote_resident_layer_mask & ~layer_mask) != 0) {
            error = "server FFN remote-resident layers exceed the resident layer mask";
            return false;
        }
        if (dormant_host_share && remote_resident_layer_mask != 0) {
            error = "S41_SERVER_FFN_DORMANT_HOST_SHARE cannot be combined with remote-resident layers";
            return false;
        }
        if (remote_resident_layer_mask != 0 &&
            (max_tokens != default_max_tokens ||
             static_cast<uint64_t>(params.n_ubatch) > default_max_tokens ||
             has_shape_policy || has_table_policy)) {
            // prefill also runs on the phone for these layers: every ubatch shape must fit
            error = "server FFN remote-resident layers require max_tokens to cover the whole ubatch and no column policy";
            return false;
        }
        std::vector<llama_ffn_split_policy::point> table_policy;
        const uint64_t table_policy_max_tokens = runtime_control ?
                default_max_tokens : max_tokens;
        if (has_table_policy && !llama_ffn_split_policy::parse(
                    table_policy_text,
                    static_cast<uint32_t>(table_policy_max_tokens),
                    static_cast<uint32_t>(columns), table_policy, error)) {
            return false;
        }
        uint64_t timeout_ms = ffn_split::client_config{}.timeout_ms;
        if (std::getenv("S41_SERVER_FFN_TIMEOUT_MS") != nullptr &&
            (!s41_parse_u64_env("S41_SERVER_FFN_TIMEOUT_MS", timeout_ms) ||
             timeout_ms == 0 || timeout_ms > 600000)) {
            error = "invalid server FFN split timeout";
            return false;
        }
        const char * tail_fence_socket =
                std::getenv("S41_SERVER_FFN_TAIL_FENCE_SOCKET");
        const bool has_tail_fence =
                tail_fence_socket != nullptr && tail_fence_socket[0] != '\0';
        uint64_t tail_fence_layer = 0;
        const char * tail_fence_join_text =
                std::getenv("S41_SERVER_FFN_TAIL_FENCE_JOIN_LAYER");
        const bool has_tail_fence_join = tail_fence_join_text != nullptr &&
                tail_fence_join_text[0] != '\0';
        uint64_t tail_fence_join_layer = 0;
        if ((std::getenv("S41_SERVER_FFN_TAIL_FENCE_LAYER") != nullptr) !=
                    has_tail_fence ||
            (tail_fence_join_text != nullptr && !has_tail_fence_join) ||
            (has_tail_fence &&
             (!s41_parse_u64_env(
                      "S41_SERVER_FFN_TAIL_FENCE_LAYER", tail_fence_layer) ||
              tail_fence_socket[0] != '/' || tail_fence_layer >= 64 ||
              (layer_mask & (UINT64_C(1) << tail_fence_layer)) == 0)) ||
            (has_tail_fence_join &&
             (!has_tail_fence ||
              !s41_parse_u64_env(
                      "S41_SERVER_FFN_TAIL_FENCE_JOIN_LAYER",
                      tail_fence_join_layer) ||
              tail_fence_join_layer >= 64 ||
              tail_fence_join_layer <= tail_fence_layer))) {
            error = "invalid server FFN split tail fence";
            return false;
        }
        if (helper_count > 1 &&
            (remote_resident_layer_mask != 0 || has_tail_fence || row_diagnostic_steps != 0)) {
            error = "server FFN helpers cannot be combined with remote-resident layers, "
                    "a tail fence or the row diagnostic";
            return false;
        }

        if (!s41_set_env("LLAMA_FFN_SPLIT_LAYER_MASK",
                    std::to_string(layer_mask)) ||
            !s41_set_env("LLAMA_FFN_SPLIT_COLUMNS", std::to_string(columns)) ||
            !s41_set_env("LLAMA_FFN_SPLIT_VIEW_SAFE_WEIGHTS", "1")) {
            error = "cannot configure the dense FFN split graph";
            return false;
        }
        if (remote_resident_layer_mask != 0 &&
            !s41_set_env("LLAMA_FFN_REMOTE_RESIDENT_LAYER_MASK",
                    std::to_string(remote_resident_layer_mask))) {
            error = "cannot configure the remote-resident FFN loader";
            return false;
        }
        if (has_shape_policy &&
            (!s41_set_env("LLAMA_FFN_SPLIT_M1_COLUMNS", std::to_string(m1_columns)) ||
             !s41_set_env("LLAMA_FFN_SPLIT_SMALL_M_COLUMNS", std::to_string(small_columns)) ||
             !s41_set_env("LLAMA_FFN_SPLIT_LARGE_M_COLUMNS", std::to_string(large_columns)) ||
             !s41_set_env("LLAMA_FFN_SPLIT_SMALL_M_MAX", std::to_string(small_m_max)))) {
            error = "cannot configure the dense FFN split shape policy";
            return false;
        }
        if (has_table_policy &&
            !s41_set_env("LLAMA_FFN_SPLIT_POLICY", table_policy_text)) {
            error = "cannot configure the dense FFN split policy table";
            return false;
        }

        for (uint64_t index = 0; index < helper_count; ++index) {
            const s41_ffn_transport_env & transport = transports[index];
            ffn_split::client_config config;
            config.artifact_sha256 = artifact_sha256;
            config.transport = transport.functionfs_usb ?
                    ffn_split::client_transport::functionfs_usb :
                    ffn_split::client_transport::tcp;
            config.host = transport.host;
            config.port = static_cast<int>(transport.port);
            config.layer_mask = helper_masks[index];
            config.remote_resident_layer_mask = remote_resident_layer_mask;
            config.max_columns = static_cast<uint32_t>(columns);
            config.n_embd = static_cast<uint32_t>(n_embd);
            config.max_tokens = static_cast<uint16_t>(max_tokens);
            config.f16_io = f16_io;
            config.swiglu = swiglu;
            config.runtime_control = runtime_control;
            config.row_diagnostic_steps = row_diagnostic_steps_;
            config.timeout_ms = static_cast<int>(timeout_ms);
            // disjoint request-id ranges keep S41SERVERFFNCALL proofs unique across helpers
            config.first_request_id = 1 + static_cast<uint32_t>(index) * (UINT32_C(1) << 24);
            if (transport.functionfs_usb) {
                config.usb_allocator = transport.usb_allocator;
                config.usb_queue_depth = static_cast<unsigned int>(
                        transport.usb_queue_depth);
                config.usbfs_available_bytes = static_cast<size_t>(
                        transport.usbfs_available_bytes);
                config.usb_slot_safety_bytes = static_cast<size_t>(
                        transport.usb_slot_safety_bytes);
                config.usb_max_payload_bytes = static_cast<size_t>(
                        transport.usb_max_payload_bytes);
                config.usb_transport_generation = transport.usb_generation;
                config.usb_vendor_id = static_cast<uint16_t>(transport.usb_vendor_id);
                config.usb_product_id = static_cast<uint16_t>(transport.usb_product_id);
                config.usb_split_h2d = transport.usb_split_h2d != 0;
                config.usb_full_duplex = transport.usb_full_duplex != 0;
                config.usb_batch_plan = transport.usb_batch_plan;
            }
            if (has_tail_fence) {
                config.tail_fence_socket = tail_fence_socket;
                config.tail_fence_layer = static_cast<int>(tail_fence_layer);
                if (has_tail_fence_join) {
                    config.tail_fence_join_layer =
                            static_cast<int>(tail_fence_join_layer);
                }
            }

            s41_ffn_helper helper;
            helper.label = labels[index];
            helper.functionfs_usb = transport.functionfs_usb;
            helper.layer_mask = helper_masks[index];
            helper.client = std::make_unique<ffn_split::client>(std::move(config));
            if (runtime_control && remote_resident_layer_mask == 0) {
                helper.deferred = true;
            } else if (!helper.client->connect(error,
                        runtime_control ? remote_resident_layer_mask : 0)) {
                // remote-resident layers are connected before the model loads so that the
                // warm-up already validates phone ownership of the omitted weights
                error = helper_count > 1 ? "helper " + helper.label + ": " + error : error;
                finish_clients();
                return false;
            }
            helpers_.push_back(std::move(helper));
        }
        layer_mask_ = layer_mask;
        remote_resident_layer_mask_ = remote_resident_layer_mask;
        runtime_control_ = runtime_control;
        dormant_host_share_ = dormant_host_share;
        auto supported_columns = [this](uint64_t value) {
            return std::all_of(helpers_.begin(), helpers_.end(), [value](const s41_ffn_helper & helper) {
                return helper.deferred || value == 0 ||
                        value == helper.client->max_columns() ||
                        value == helper.client->alternate_columns() ||
                        (helper.client->column_quantum() != 0 &&
                         value % helper.client->column_quantum() == 0);
            });
        };
        for (const auto & helper : helpers_) {
            if (!helper.deferred && !consistent_geometry(helper, error)) {
                finish_clients();
                return false;
            }
        }
        if (has_shape_policy &&
            (!supported_columns(m1_columns) ||
             !supported_columns(small_columns) ||
             !supported_columns(large_columns))) {
            error = "FFN split shape policy is unsupported by the phone worker";
            finish_clients();
            return false;
        }
        if (has_table_policy && std::any_of(
                    table_policy.begin(), table_policy.end(),
                    [&supported_columns](const auto & value) {
                        return !supported_columns(value.columns);
                    })) {
            error = "FFN split policy table is unsupported by the phone worker";
            finish_clients();
            return false;
        }
        params.cb_eval = eval_cb;
        params.cb_eval_user_data = this;
        const char * endpoint_host = transports.front().functionfs_usb ? "usb" : transports.front().host.c_str();
        const unsigned long long endpoint_port = transports.front().functionfs_usb ?
                0 : static_cast<unsigned long long>(transports.front().port);
        if (helpers_.size() > 1) {
            for (const auto & helper : helpers_) {
                const s41_ffn_transport_env & transport = transports[&helper - helpers_.data()];
                std::fprintf(stderr,
                        "S41SERVERFFNHELPER label=%s layer_mask=%llu transport=%s host=%s port=%llu connection=%s\n",
                        helper.label.c_str(), static_cast<unsigned long long>(helper.layer_mask),
                        transport.functionfs_usb ? "functionfs-usb" : "tcp",
                        transport.functionfs_usb ? "usb" : transport.host.c_str(),
                        static_cast<unsigned long long>(transport.port),
                        helper.deferred ? "deferred" : "connected");
            }
        }
        if (runtime_control && remote_resident_layer_mask == 0) {
            // a lost helper's session is closed and its layers stay on the host; a TCP helper reconnects
            // when a policy owns it again, a USB helper only after its worker session was relaunched
            std::fprintf(stderr, "S41SERVERFFNCAPS helper_mask_out=1 helper_reconnect_tcp=1\n");
        }
        if (runtime_control) {
            std::fprintf(stderr,
                    "S41SERVERFFN ready host=%s port=%llu columns=%llu "
                    "max_tokens=%llu io=%s activation=%s weights=view_safe "
                    "runtime_control=enabled initial_mask=0 initial_columns=0 "
                    "connection=%s%s\n",
                    endpoint_host, endpoint_port,
                    static_cast<unsigned long long>(columns),
                    static_cast<unsigned long long>(max_tokens),
                    f16_io ? "f16" : "f32",
                    swiglu ? "swiglu" : "geglu",
                    helpers_.front().deferred ? "deferred" : "connected",
                    helpers_.size() > 1 ? (" helpers=" + std::to_string(helpers_.size())).c_str() : "");
            std::fflush(stderr);
            return true;
        }
        if (has_table_policy) {
            std::fprintf(stderr,
                    "S41SERVERFFN ready host=%s port=%llu columns=%llu "
                    "max_tokens=%u io=%s activation=%s weights=view_safe "
                    "tail_fence=%s tail_fence_layer=%d "
                    "tail_fence_join_layer=%d policy=%s\n",
                    endpoint_host, endpoint_port,
                    static_cast<unsigned long long>(columns),
                    static_cast<unsigned>(std::min(params.n_ubatch, 512)),
                    f16_io ? "f16" : "f32",
                    swiglu ? "swiglu" : "geglu",
                    has_tail_fence ? "enabled" : "disabled",
                    has_tail_fence ? static_cast<int>(tail_fence_layer) : -1,
                    has_tail_fence_join ?
                            static_cast<int>(tail_fence_join_layer) : -1,
                    table_policy_text);
            std::fflush(stderr);
            return true;
        }
        std::fprintf(stderr,
                "S41SERVERFFN ready host=%s port=%llu columns=%llu "
                "max_tokens=%u io=%s activation=%s weights=view_safe policy=%llu:%llu,%llu:%llu,%u:%llu\n",
                endpoint_host, endpoint_port,
                static_cast<unsigned long long>(columns),
                static_cast<unsigned>(std::min(params.n_ubatch, 512)),
                f16_io ? "f16" : "f32",
                swiglu ? "swiglu" : "geglu",
                static_cast<unsigned long long>(1),
                static_cast<unsigned long long>(m1_columns),
                static_cast<unsigned long long>(small_m_max),
                static_cast<unsigned long long>(small_columns),
                static_cast<unsigned>(std::min(params.n_ubatch, 512)),
                static_cast<unsigned long long>(large_columns));
        std::fflush(stderr);
        return true;
    }

    void finish() {
        if (helpers_.empty() || finished_) {
            return;
        }
        for (const auto & helper : helpers_) {
            finish_helper(helper);
        }
        std::fflush(stderr);
        finished_ = true;
    }

    // with several helpers every summary row starts with the helper label and its layer mask
    void finish_helper(const s41_ffn_helper & helper) const {
        helper.client->finish();
        const ffn_split::client_summary summary = helper.client->summary();
        const std::string owner = helpers_.size() < 2 ? std::string() :
                "\"helper\":\"" + helper.label + "\",\"layer_mask\":" + std::to_string(helper.layer_mask) + ",";
        std::fprintf(stderr,
                "S41SERVERFFN {%s\"status\":\"%s\",\"calls\":%zu,"
                "\"transport\":\"%s\",\"allocator\":\"%s\","
                "\"transport_generation\":\"%s\","
                "\"batch_plan\":\"%s\","
                "\"queue_depth\":%u,\"maximum_active_slots\":%u,"
                "\"maximum_outstanding_transfers\":%u,"
                "\"batched_calls\":%zu,\"transfer_subrequests\":%zu,"
                "\"maximum_payload_bytes\":%zu,"
                "\"full_duplex\":%s,"
                "\"decode_calls\":%zu,\"prefill_calls\":%zu,"
                "\"input_rows\":%llu,\"maximum_tokens\":%u,"
                "\"upload_bytes\":%llu,\"download_bytes\":%llu,"
                "\"h2d_us\":%llu,\"d2h_exposed_us\":%llu,"
                "\"rpc_mean_ms\":%.6f,"
                "\"rpc_p50_ms\":%.6f,\"rpc_p90_ms\":%.6f,"
                "\"h2d_payload_MBps\":%.6f,"
                "\"d2h_exposed_payload_MBps\":%.6f,"
                "\"compute_mean_ms\":%.6f,\"compute_p50_ms\":%.6f,"
                "\"host_mean_ms\":%.6f,\"host_p50_ms\":%.6f,"
                "\"wait_mean_ms\":%.6f,\"wait_min_ms\":%.6f,"
                "\"wait_p10_ms\":%.6f,\"wait_p50_ms\":%.6f,"
                "\"wait_p90_ms\":%.6f,\"wait_max_ms\":%.6f,"
                "\"overlap_mean_ms\":%.6f,\"overlap_p50_ms\":%.6f,"
                "\"useful_overlap_mean_ms\":%.6f,"
                "\"phone_tail_mean_ms\":%.6f,\"phone_tail_min_ms\":%.6f,"
                "\"phone_tail_p10_ms\":%.6f,\"phone_tail_p50_ms\":%.6f,"
                "\"phone_tail_p90_ms\":%.6f,\"phone_tail_max_ms\":%.6f,"
                "\"tail_fence_opportunities\":%zu,\"tail_fence_calls\":%zu,"
                "\"tail_fence_skipped_complete\":%zu,"
                "\"tail_fence_mean_ms\":%.6f,\"tail_fence_p50_ms\":%.6f,"
                "\"tail_fence_p90_ms\":%.6f,\"tail_fence_max_ms\":%.6f,"
                "\"tail_fence_overlap_mean_ms\":%.6f,"
                "\"tail_fence_overlap_p50_ms\":%.6f,"
                "\"tail_fence_overrun_mean_ms\":%.6f,"
                "\"tail_fence_overrun_max_ms\":%.6f,"
                "\"tail_fence_macro_windows\":%zu,"
                "\"tail_fence_window_mean_ms\":%.6f,"
                "\"tail_fence_window_min_ms\":%.6f,"
                "\"tail_fence_window_p10_ms\":%.6f,"
                "\"tail_fence_window_p50_ms\":%.6f,"
                "\"tail_fence_window_p90_ms\":%.6f,"
                "\"tail_fence_window_max_ms\":%.6f,"
                "\"tail_fence_join_wait_mean_ms\":%.6f,"
                "\"tail_fence_join_wait_p50_ms\":%.6f,"
                "\"tail_fence_join_wait_p90_ms\":%.6f,"
                "\"tail_fence_join_wait_max_ms\":%.6f}\n",
                owner.c_str(), helper.client->failed() ? "error" : "ok", summary.calls,
                summary.transport.c_str(), summary.allocator.c_str(),
                summary.transport_generation.c_str(),
                summary.batch_plan.c_str(), summary.queue_depth,
                summary.maximum_active_slots,
                summary.maximum_outstanding_transfers,
                summary.batched_calls, summary.transfer_subrequests,
                summary.maximum_payload_bytes,
                summary.full_duplex ? "true" : "false",
                summary.decode_calls, summary.prefill_calls,
                static_cast<unsigned long long>(summary.input_rows),
                summary.maximum_tokens,
                static_cast<unsigned long long>(summary.upload_bytes),
                static_cast<unsigned long long>(summary.download_bytes),
                static_cast<unsigned long long>(summary.h2d_us),
                static_cast<unsigned long long>(summary.d2h_exposed_us),
                summary.rpc_mean_ms, summary.rpc_p50_ms, summary.rpc_p90_ms,
                summary.h2d_payload_MBps,
                summary.d2h_exposed_payload_MBps,
                summary.compute_mean_ms, summary.compute_p50_ms,
                summary.host_branch_mean_ms, summary.host_branch_p50_ms,
                summary.wait_mean_ms, summary.wait_min_ms,
                summary.wait_p10_ms, summary.wait_p50_ms,
                summary.wait_p90_ms, summary.wait_max_ms,
                summary.overlap_mean_ms, summary.overlap_p50_ms,
                summary.useful_overlap_mean_ms,
                summary.phone_tail_mean_ms, summary.phone_tail_min_ms,
                summary.phone_tail_p10_ms, summary.phone_tail_p50_ms,
                summary.phone_tail_p90_ms, summary.phone_tail_max_ms,
                summary.tail_fence_opportunities, summary.tail_fence_calls,
                summary.tail_fence_skipped_complete,
                summary.tail_fence_mean_ms, summary.tail_fence_p50_ms,
                summary.tail_fence_p90_ms, summary.tail_fence_max_ms,
                summary.tail_fence_overlap_mean_ms,
                summary.tail_fence_overlap_p50_ms,
                summary.tail_fence_overrun_mean_ms,
                summary.tail_fence_overrun_max_ms,
                summary.tail_fence_macro_windows,
                summary.tail_fence_window_mean_ms,
                summary.tail_fence_window_min_ms,
                summary.tail_fence_window_p10_ms,
                summary.tail_fence_window_p50_ms,
                summary.tail_fence_window_p90_ms,
                summary.tail_fence_window_max_ms,
                summary.tail_fence_join_wait_mean_ms,
                summary.tail_fence_join_wait_p50_ms,
                summary.tail_fence_join_wait_p90_ms,
                summary.tail_fence_join_wait_max_ms);
        if (helper.client->failed()) {
            std::fprintf(stderr, "S41SERVERFFNERROR %sdetail=%s\n",
                    helpers_.size() < 2 ? "" : ("helper=" + helper.label + " ").c_str(),
                    helper.client->error().c_str());
        }
        if (helper.client->reset_count() != 0) {
            std::fprintf(stderr, "S41SERVERFFNRESET %ssummary resets=%zu last_error=%s\n",
                    helpers_.size() < 2 ? "" : ("helper=" + helper.label + " ").c_str(),
                    helper.client->reset_count(), helper.client->last_reset_error().c_str());
        }
        for (const auto & shape : summary.shapes) {
            std::fprintf(stderr,
                    "S41SERVERFFNSHAPE {%s\"tokens\":%u,\"columns\":%u,"
                    "\"calls\":%zu,\"rpc_mean_ms\":%.6f,"
                    "\"rpc_p50_ms\":%.6f,\"compute_mean_ms\":%.6f,"
                    "\"compute_p50_ms\":%.6f,\"host_mean_ms\":%.6f,"
                    "\"wait_mean_ms\":%.6f,\"overlap_mean_ms\":%.6f,"
                    "\"overlap_p50_ms\":%.6f}\n",
                    owner.c_str(), shape.tokens, shape.columns, shape.calls,
                    shape.rpc_mean_ms, shape.rpc_p50_ms,
                    shape.compute_mean_ms, shape.compute_p50_ms,
                    shape.host_branch_mean_ms, shape.wait_mean_ms,
                    shape.overlap_mean_ms,
                    shape.overlap_p50_ms);
        }
    }

    bool failed() const {
        return std::any_of(helpers_.begin(), helpers_.end(),
                [](const s41_ffn_helper & helper) { return helper.client->failed(); });
    }

    std::string error() const {
        for (const auto & helper : helpers_) {
            if (helper.client->failed()) {
                return helpers_.size() < 2 ? helper.client->error() :
                        "helper " + helper.label + ": " + helper.client->error();
            }
        }
        return std::string();
    }

    bool enabled() const {
        return !helpers_.empty();
    }

    uint64_t layer_mask() const {
        return layer_mask_;
    }

    uint64_t remote_resident_layer_mask() const {
        return remote_resident_layer_mask_;
    }

    uint32_t max_columns() const {
        return helpers_.empty() ? 0 : helpers_.front().client->max_columns();
    }

    // one quantum for the union policy: the least common multiple of the helpers' quanta,
    // 1 while a connection is deferred (the clients then validate the width at apply time)
    uint32_t column_quantum() const {
        if (helpers_.empty()) {
            return 0;
        }
        uint64_t quantum = 1;
        for (const auto & helper : helpers_) {
            if (helper.deferred) {
                return 1;
            }
            const uint64_t value = helper.client->column_quantum();
            quantum = value == 0 ? quantum : quantum / std::gcd(quantum, value) * value;
        }
        return quantum > std::numeric_limits<uint32_t>::max() ? 1 : static_cast<uint32_t>(quantum);
    }

    ffn_split::client_summary summary() const {
        return aggregate_summary([](const ffn_split::client & client) {
            return client.summary();
        });
    }

    ffn_split::client_summary summary(
            const std::vector<std::string> & request_ids) const {
        return aggregate_summary([&request_ids](const ffn_split::client & client) {
            return client.summary(request_ids);
        });
    }

    bool runtime_control() const {
        return runtime_control_;
    }

    bool dormant_host_share() const {
        return dormant_host_share_;
    }

    bool dormant_drop_cache() const {
        return dormant_drop_cache_;
    }

    uint32_t row_diagnostic_steps() const {
        return row_diagnostic_steps_;
    }

    bool dormant_populate() const {
        return dormant_populate_;
    }

    // Applies one union (layer_mask, columns) policy: each helper receives the owned subset.
    // Every owner of an active layer connects before any helper switches, and a helper that
    // rejects its subset restores the helpers switched before it.
    bool apply_policy(
            uint64_t layer_mask, uint32_t columns, std::string & error) {
        if (helpers_.empty()) {
            error = "FFN split client is absent";
            return false;
        }
        std::lock_guard<std::mutex> lock(policy_mutex_);
        const auto owned = [layer_mask, columns](const s41_ffn_helper & helper) {
            return columns == 0 ? UINT64_C(0) : layer_mask & helper.layer_mask;
        };
        for (auto & helper : helpers_) {
            // a failed session (or, owned again, an idle TCP session whose worker closed it) is closed
            // here; its layers stay on the host until a policy owns the helper and it connects afresh
            const bool failed = helper.client->failed();
            if (remote_resident_layer_mask_ == 0 && (failed || (owned(helper) != 0 &&
                    helper.applied_layer_mask == 0 && helper.client->peer_closed()))) {
                const bool usb_session = helper.functionfs_usb && helper.client->ready();
                helper.client->reset_session();
                helper.applied_layer_mask = 0;
                helper.applied_columns = 0;
                std::fprintf(stderr, "S41SERVERFFNRESET %scause=%s resets=%zu\n",
                        helpers_.size() < 2 ? "" : ("helper=" + helper.label + " ").c_str(),
                        failed ? "failed" : "peer_closed", helper.client->reset_count());
                std::fflush(stderr);
                if (usb_session && owned(helper) != 0) {
                    // the FunctionFS worker reads a new HELLO only after its session was relaunched
                    error = "FFN split USB session was reset; its worker must be relaunched before it is owned again";
                    error = helpers_.size() < 2 ? error : "helper " + helper.label + ": " + error;
                    return false;
                }
            }
            if (owned(helper) != 0 && !helper.client->ready()) {
                if (!helper.client->connect(error, owned(helper)) ||
                    !consistent_geometry(helper, error)) {
                    if (helper.client->reset_count() != 0) {
                        helper.client->latch_error(error);
                    }
                    error = helpers_.size() < 2 ? error : "helper " + helper.label + ": " + error;
                    return false;
                }
                helper.deferred = false;
            }
        }
        for (size_t index = 0; index < helpers_.size(); ++index) {
            s41_ffn_helper & helper = helpers_[index];
            const uint64_t mask = owned(helper);
            if (!helper.client->set_runtime_policy(mask, mask == 0 ? 0 : columns, error)) {
                error = helpers_.size() < 2 ? error : "helper " + helper.label + ": " + error;
                for (size_t previous = 0; previous < index; ++previous) {
                    std::string ignored;
                    helpers_[previous].client->set_runtime_policy(
                            helpers_[previous].applied_layer_mask,
                            helpers_[previous].applied_columns, ignored);
                }
                return false;
            }
        }
        for (auto & helper : helpers_) {
            helper.applied_layer_mask = owned(helper);
            helper.applied_columns = helper.applied_layer_mask == 0 ? 0 : columns;
        }
        return true;
    }

    bool apply_context(
            const std::vector<server_ffn_split_runtime_context> & entries,
            std::string & error) {
        if (helpers_.empty()) {
            error = "FFN split client is absent";
            return false;
        }
        std::vector<ffn_split::client_runtime_context_entry> converted;
        converted.reserve(entries.size());
        for (const auto & entry : entries) {
            converted.push_back({
                entry.request_id,
                entry.slot_id,
                entry.rows,
                entry.plan_generation,
                entry.ubatch_rows, entry.positions, entry.decoded_token_index, entry.applied_token_index,
            });
        }
        for (auto & helper : helpers_) {
            if (!helper.client->set_runtime_context(converted, error)) {
                error = helpers_.size() < 2 ? error : "helper " + helper.label + ": " + error;
                return false;
            }
        }
        return true;
    }

private:
    // with one helper its client observes every tensor as before; with several the FFN marker's
    // layer selects the owning client and no other tensor is observed
    static bool eval_cb(ggml_tensor * tensor, bool ask, void * user_data) {
        auto * runtime = static_cast<s41_server_ffn_runtime *>(user_data);
        if (runtime == nullptr || runtime->helpers_.empty() || tensor == nullptr) {
            return false;
        }
        if (runtime->helpers_.size() == 1) {
            return runtime->eval_helper(runtime->helpers_.front(), tensor, ask);
        }
        int layer = -1;
        if (s41_ffn_marker_layer(tensor->name, layer)) {
            for (auto & helper : runtime->helpers_) {
                if ((helper.layer_mask & (UINT64_C(1) << layer)) != 0) {
                    return runtime->eval_helper(helper, tensor, ask);
                }
            }
        }
        return !ask;
    }

    // names a helper the moment its session fails (the shutdown summary repeats a lasting failure)
    bool eval_helper(s41_ffn_helper & helper, ggml_tensor * tensor, bool ask) const {
        const bool failed = helper.client->failed();
        const bool result = helper.client->eval(tensor, ask);
        if (!failed && helper.client->failed()) {
            std::fprintf(stderr, "S41SERVERFFNERROR %sdetail=%s\n",
                    helpers_.size() < 2 ? "" : ("helper=" + helper.label + " ").c_str(),
                    helper.client->error().c_str());
            std::fflush(stderr);
        }
        return result;
    }

    void finish_clients() {
        for (auto & helper : helpers_) {
            helper.client->finish();
        }
        helpers_.clear();
    }

    // one union column policy needs every connected helper to serve the same FFN suffix
    bool consistent_geometry(const s41_ffn_helper & helper, std::string & error) const {
        for (const auto & other : helpers_) {
            if (&other != &helper && other.client->ready() &&
                (other.client->n_ff() != helper.client->n_ff() ||
                 other.client->offset() != helper.client->offset())) {
                error = "helper " + helper.label + " serves another FFN geometry than helper " + other.label;
                return false;
            }
        }
        return true;
    }

    // totals over all helpers; the per-call means are weighted by each helper's call count
    template <typename Summarize>
    ffn_split::client_summary aggregate_summary(Summarize summarize) const {
        if (helpers_.size() < 2) {
            return helpers_.empty() ? ffn_split::client_summary{} : summarize(*helpers_.front().client);
        }
        ffn_split::client_summary total;
        double rpc_ms = 0.0;
        double compute_ms = 0.0;
        double host_ms = 0.0;
        double wait_ms = 0.0;
        double useful_overlap_ms = 0.0;
        for (const auto & helper : helpers_) {
            const ffn_split::client_summary row = summarize(*helper.client);
            total.queue_depth = std::max(total.queue_depth, row.queue_depth);
            total.maximum_active_slots = std::max(total.maximum_active_slots, row.maximum_active_slots);
            total.maximum_outstanding_transfers = std::max(
                    total.maximum_outstanding_transfers, row.maximum_outstanding_transfers);
            total.maximum_payload_bytes = std::max(total.maximum_payload_bytes, row.maximum_payload_bytes);
            total.maximum_tokens = std::max(total.maximum_tokens, row.maximum_tokens);
            total.batched_calls += row.batched_calls;
            total.transfer_subrequests += row.transfer_subrequests;
            total.calls += row.calls;
            total.decode_calls += row.decode_calls;
            total.prefill_calls += row.prefill_calls;
            total.input_rows += row.input_rows;
            total.upload_bytes += row.upload_bytes;
            total.download_bytes += row.download_bytes;
            total.h2d_us += row.h2d_us;
            total.d2h_exposed_us += row.d2h_exposed_us;
            rpc_ms += row.rpc_mean_ms * row.calls;
            compute_ms += row.compute_mean_ms * row.calls;
            host_ms += row.host_branch_mean_ms * row.calls;
            wait_ms += row.wait_mean_ms * row.calls;
            useful_overlap_ms += row.useful_overlap_mean_ms * row.calls;
        }
        if (total.calls > 0) {
            total.rpc_mean_ms = rpc_ms / total.calls;
            total.compute_mean_ms = compute_ms / total.calls;
            total.host_branch_mean_ms = host_ms / total.calls;
            total.wait_mean_ms = wait_ms / total.calls;
            total.useful_overlap_mean_ms = useful_overlap_ms / total.calls;
        }
        total.h2d_payload_MBps = total.h2d_us == 0 ? 0.0 :
                static_cast<double>(total.upload_bytes) / static_cast<double>(total.h2d_us);
        total.d2h_exposed_payload_MBps = total.d2h_exposed_us == 0 ? 0.0 :
                static_cast<double>(total.download_bytes) / static_cast<double>(total.d2h_exposed_us);
        return total;
    }

    std::vector<s41_ffn_helper> helpers_;
    uint64_t layer_mask_ = 0;
    uint64_t remote_resident_layer_mask_ = 0;
    bool runtime_control_ = false;
    bool dormant_host_share_ = false;
    bool dormant_drop_cache_ = true;
    bool dormant_populate_ = true;
    uint32_t row_diagnostic_steps_ = 0;
    bool finished_ = false;
    std::mutex policy_mutex_;
};

} // namespace
#endif

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

// satisfies -Wmissing-declarations
int llama_server(int argc, char ** argv);

int llama_server(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // own arguments required by this example
    common_params params;

#if defined(S41_SERVER_FFN_SPLIT)
    s41_server_ffn_runtime ffn_runtime;
#endif

    common_init();

    // start the stream session manager GC right after common init, before any HTTP route can
    // touch it. lifecycle is symmetric, stop_gc() runs in clean_up() before backend free
    g_stream_sessions.start_gc();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    common_models_handler models_handler;
    try {
        models_handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
        if (common_models_handler_is_preset_repo(models_handler)) {
            // apply the preset and start the server in router mode
            common_models_handler_apply(models_handler, params);
        }
    } catch (const std::exception & e) {
        SRV_ERR("failed to fetch model metadata: %s\n", e.what());
        return 1;
    }

    // router server never loads a model and must not touch the GPU
    const bool is_router_server = params.model.path.empty()
                               && params.model.hf_repo.empty();

    std::string warm_tier_internal_token;
    if (is_router_server) {
        const char * warm_tier_config =
            std::getenv("LLAMA_SERVER_WARM_TIER_CONFIG");
        if (warm_tier_config != nullptr && warm_tier_config[0] != '\0') {
            try {
                warm_tier_internal_token =
                    server_warm_tier_internal_token_from_env();
            } catch (const std::exception & e) {
                SRV_ERR(
                    "failed to load warm-tier internal capability: %s\n",
                    e.what());
                return 1;
            }
        }
    }

    // skip device enumeration so the CUDA primary context stays uncreated
    common_params_print_info(params, !is_router_server);

    if (!is_router_server) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            SRV_TRC("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    auto model_name = params.model.get_name();
    if (params.model_alias.empty() && !model_name.empty()) {
        params.model_alias.insert(model_name);
    }

    // struct that contains llama context and inference
    server_context ctx_server;

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    //
    // Router
    //

    // register API routes
    server_child child; // only used in non-router mode
    server_routes routes(params, ctx_server);
    server_tools tools;

    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(
                params, argc, argv, warm_tier_internal_token);
        } catch (const std::exception & e) {
            SRV_ERR("failed to initialize router models: %s\n", e.what());
            return 1;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_control                = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.post_chat_completions_tok   = models_routes->proxy_post;
        routes.post_responses_tok_oai      = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        // custom routes for router
        routes.get_props                   = models_routes->get_router_props;
        routes.get_models                  = models_routes->get_router_models;

        ctx_http.post("/models",               ex_wrapper(models_routes->post_router_models));
        ctx_http.post("/models/load",          ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload",        ex_wrapper(models_routes->post_router_models_unload));
        ctx_http.get ("/models/sse",           ex_wrapper(models_routes->get_router_models_sse));
        ctx_http.del ("/models",               ex_wrapper(models_routes->del_router_models));
        if (models_routes->warm_tier_enabled()) {
            ctx_http.post(
                "/experimental/warm-tier/activate",
                ex_wrapper(models_routes->post_warm_tier_activate));
            ctx_http.get(
                "/experimental/warm-tier/activate",
                ex_wrapper(models_routes->get_warm_tier_activate));
            ctx_http.post(
                "/experimental/warm-tier/completion",
                ex_wrapper(models_routes->post_warm_tier_completion));
            ctx_http.post(
                "/experimental/warm-tier/requests",
                ex_wrapper(models_routes->post_warm_tier_request));
            ctx_http.get(
                "/experimental/warm-tier/requests/:request_id",
                ex_wrapper(models_routes->get_warm_tier_request));
            ctx_http.post(
                "/experimental/warm-tier/finalize",
                ex_wrapper(models_routes->post_warm_tier_finalize));
            ctx_http.get(
                "/experimental/warm-tier/finalize",
                ex_wrapper(models_routes->get_warm_tier_finalize));
            ctx_http.post(
                "/experimental/warm-tier/switch",
                ex_wrapper(models_routes->post_warm_tier_switch));
        }
    }

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    // token counting
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));

    // resumable streaming, the conversation_id is the session identity end to end. router and
    // child wire different handlers under the same paths: a child binds the local g_stream_sessions
    // backed factories, the router binds proxies that resolve the owning child through the
    // conv_id -> model map
    server_http_context::handler_t stream_get_h;
    server_http_context::handler_t streams_lookup_h;
    server_http_context::handler_t stream_delete_h;
    if (is_router_server) {
        stream_get_h     = models_routes->router_stream_get;
        streams_lookup_h = models_routes->router_streams_lookup;
        stream_delete_h  = models_routes->router_stream_delete;
    } else {
        stream_get_h     = make_stream_get_handler();
        streams_lookup_h = make_streams_lookup_handler();
        stream_delete_h  = make_stream_delete_handler();
    }
    ctx_http.get ("/v1/stream/:conv_id",       ex_wrapper(stream_get_h));
    // POST /v1/streams/lookup with body {"conversation_ids": [...]}. you can only ask for ids
    // you already own (the WebUI passes the convs visible in its sidebar). the server never
    // lists ids it has not been asked about, so a random caller cannot enumerate live sessions
    ctx_http.post("/v1/streams/lookup",        ex_wrapper(streams_lookup_h));
    ctx_http.del ("/v1/stream/:conv_id",       ex_wrapper(stream_delete_h));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();

    // return 403 for disabled features
    server_http_context::handler_t res_403 = [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->status = 403;
        res->data = safe_json_to_str({
            {"error", {
                {"message", "this feature is disabled"},
                {"type", "feature_disabled"},
            }}
        });
        return res;
    };

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    if (params.ui_mcp_proxy) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "CORS proxy is enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be removed or changed in future versions\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
    } else {
        ctx_http.get ("/cors-proxy",      ex_wrapper(res_403));
        ctx_http.post("/cors-proxy",      ex_wrapper(res_403));
    }

    // EXPERIMENTAL built-in tools
    if (!params.server_tools.empty()) {
        try {
            tools.setup(params.server_tools);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return 1;
        }
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "Built-in tools are enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be changed in the future\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
    } else {
        ctx_http.get ("/tools",           ex_wrapper(res_403));
        ctx_http.post("/tools",           ex_wrapper(res_403));
    }

    //
    // Handle downloading model
    //

    if (child.is_child() && child.get_mode() == SERVER_CHILD_MODE_DOWNLOAD) {
        return child.run_download(params);
    } else if (!is_router_server) {
        // single-model mode (NOT spawned by router)
        try {
            common_models_handler_apply(models_handler, params);
        } catch (const std::exception & e) {
            SRV_ERR("failed to download model: %s\n", e.what());
            return 1;
        }
    }

#if defined(S41_SERVER_FFN_SPLIT)
    if (!is_router_server) {
        std::string error;
        if (!ffn_runtime.init(params, error)) {
            SRV_ERR("failed to initialize server FFN split: %s\n", error.c_str());
            return 1;
        }
        if (ffn_runtime.enabled()) {
            ctx_server.configure_ffn_remote_resident(
                    ffn_runtime.remote_resident_layer_mask());
            ctx_server.configure_ffn_dormant_host_share(
                    ffn_runtime.dormant_host_share(), ffn_runtime.dormant_drop_cache(), ffn_runtime.dormant_populate(),
                    ffn_runtime.row_diagnostic_steps());
        }
        if (ffn_runtime.enabled() && ffn_runtime.runtime_control()) {
            ctx_server.configure_ffn_split_runtime(
                    ffn_runtime.layer_mask(),
                    ffn_runtime.max_columns(),
                    ffn_runtime.column_quantum(),
                    [&ffn_runtime](
                            uint64_t layer_mask, uint32_t columns,
                            std::string & error) {
                        return ffn_runtime.apply_policy(
                                layer_mask, columns, error);
                    },
                    [&ffn_runtime](
                            const std::vector<
                                    server_ffn_split_runtime_context> & entries,
                            std::string & error) {
                        return ffn_runtime.apply_context(entries, error);
                    },
                    [&ffn_runtime, &ctx_server](
                            const std::vector<std::string> & request_ids) {
                        const ffn_split::client_summary value =
                                ffn_runtime.summary(request_ids);
                        const server_ffn_dormant_state dormant = ctx_server.ffn_dormant_state();
                        const auto total_us = [calls = value.calls](
                                double mean_ms) -> uint64_t {
                            return static_cast<uint64_t>(std::llround(
                                    mean_ms * static_cast<double>(calls) *
                                    1000.0));
                        };
                        const uint64_t h2d_us =
                                value.h2d_payload_MBps <= 0.0 ? 0 :
                                static_cast<uint64_t>(std::llround(
                                    static_cast<double>(value.upload_bytes) /
                                    value.h2d_payload_MBps));
                        const uint64_t d2h_us =
                                value.d2h_exposed_payload_MBps <= 0.0 ? 0 :
                                static_cast<uint64_t>(std::llround(
                                    static_cast<double>(value.download_bytes) /
                                    value.d2h_exposed_payload_MBps));
                        return json {
                            { "configured_queue_depth", value.queue_depth },
                            { "maximum_active_slots", value.maximum_active_slots },
                            { "maximum_outstanding_transfers",
                              value.maximum_outstanding_transfers },
                            { "batched_calls", value.batched_calls },
                            { "transfer_subrequests", value.transfer_subrequests },
                            { "maximum_tokens", value.maximum_tokens },
                            { "calls", value.calls },
                            { "input_rows", value.input_rows },
                            { "download_bytes", value.download_bytes },
                            { "exposed_tail_us", total_us(value.wait_mean_ms) },
                            { "phone_compute_us", total_us(value.compute_mean_ms) },
                            { "rpc_us", total_us(value.rpc_mean_ms) },
                            { "upload_bytes", value.upload_bytes },
                            { "usb_transfer_us", h2d_us + d2h_us },
                            { "usb_h2d_us", value.h2d_us },
                            { "usb_d2h_us", value.d2h_exposed_us },
                            { "desktop_compute_us", total_us(value.host_branch_mean_ms) },
                            { "useful_overlap_us", total_us(value.useful_overlap_mean_ms) },
                            // decode-only relocation: the scheduler credits this release once per generation
                            { "dormant_host_share", dormant.enabled ? 1 : 0 },
                            { "dormant_layer_mask", dormant.layer_mask },
                            { "dormant_host_columns", dormant.host_columns },
                            { "dormant_release_generation", dormant.release_generation },
                            { "dormant_released_bytes", dormant.released_bytes },
                            { "dormant_release_elapsed_us", dormant.release_elapsed_us },
                        };
                    });
        }
    }
#endif

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        SRV_INF("%s", "starting server in router mode. models will be automatically loaded on-demand\n");

        clean_up = [&models_routes]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            g_stream_sessions.stop_gc();
            if (models_routes.has_value()) {
                models_routes->stopping.store(true); // maybe redundant, but just to be safe
                models_routes->models.unload_all();
            }
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }
        ctx_http.is_ready.store(true);

        shutdown_handler = [&](int) {
            if (models_routes.has_value()) {
                // important to disconnect any SSE clients
                models_routes->stopping.store(true);
            }
            ctx_http.stop();
        };

    } else {
        // setup clean up function, to be called before exit
        clean_up = [&ctx_http, &ctx_server]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            g_stream_sessions.stop_gc();
            ctx_http.stop();
            ctx_server.terminate();
            llama_backend_free();
        };

        // start the HTTP server before loading the model to be able to serve /health requests
        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }

        // setup communication child --> router if necessary
        if (child.is_child()) {
            ctx_server.set_state_callback([&](server_state state, json payload) {
                child.notify_to_router(server_state_to_str(state), payload);
            });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            SRV_ERR("%s", "exiting due to model loading error\n");
            return 1;
        }

#if defined(S41_SERVER_FFN_SPLIT)
        if (ffn_runtime.failed()) {
            const std::string error = ffn_runtime.error();
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            SRV_ERR("FFN split failed during model loading: %s\n", error.c_str());
            return 1;
        }
        {
            const llama_model * loaded_model = llama_get_model(
                    ctx_server.get_llama_context());
            const uint64_t loaded_remote_mask = loaded_model == nullptr ?
                    0 : llama_model_remote_resident_ffn_layer_mask(loaded_model);
            if (loaded_remote_mask != ffn_runtime.remote_resident_layer_mask()) {
                clean_up();
                if (ctx_http.thread.joinable()) {
                    ctx_http.thread.join();
                }
                SRV_ERR("remote-resident FFN mask differs: loader=0x%llx runtime=0x%llx\n",
                        (unsigned long long) loaded_remote_mask,
                        (unsigned long long) ffn_runtime.remote_resident_layer_mask());
                return 1;
            }
            if (loaded_remote_mask != 0) {
                std::fprintf(stderr,
                        "S41SERVERFFN remote_resident mask=%llu omitted_bytes=%llu "
                        "unmapped_bytes=%llu warmup=%s\n",
                        (unsigned long long) loaded_remote_mask,
                        (unsigned long long) llama_model_remote_resident_ffn_bytes(loaded_model),
                        (unsigned long long) llama_model_remote_resident_ffn_unmapped_bytes(loaded_model),
                        params.warmup ? "validated" : "skipped");
                std::fflush(stderr);
            }
        }
#endif

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);

        SRV_INF("%s", "model loaded\n");

        shutdown_handler = [&](int) {
            // this will unblock start_loop()
            ctx_server.terminate();
        };
    }

    // TODO: refactor in common/console
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    SRV_INF("listening on %s\n", ctx_http.listening_address.c_str());

    if (is_router_server) {
        SRV_WRN("%s", "NOTE: router mode is experimental\n");
        SRV_WRN("%s", "      it is not recommended to use this mode in untrusted environments\n");

        if (!params.models_preset_hf.empty()) {
            SRV_WRN(      "NOTE: using preset.ini from HF repo '%s'\n", params.models_preset_hf.c_str());
            SRV_WRN("%s", "      please only use presets that you can trust! Unknown presets may be unsafe\n");
        }

        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join(); // keep the main thread alive
        }

        // when the HTTP server stops, clean up and exit
        clean_up();
    } else {
        // optionally, notify router server that this instance is ready
        std::thread monitor_thread;
        if (child.is_child()) {
            monitor_thread = child.setup(shutdown_handler);
            child.notify_to_router(server_state_to_str(SERVER_STATE_READY), routes.get_model_info());
        }

        // this call blocks the main thread until queue_tasks.terminate() is called
        ctx_server.start_loop();

#if defined(S41_SERVER_FFN_SPLIT)
        ffn_runtime.finish();
#endif

        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            common_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}
