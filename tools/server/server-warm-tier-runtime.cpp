#include "server-warm-tier-runtime.h"

#include "server-warm-tier-executors.h"

#include <nlohmann/json.hpp>

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#include <openssl/sha.h>
#endif

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

using json = nlohmann::json;

constexpr size_t MAX_CONFIG_BYTES = 1024 * 1024;

struct runtime_executor_config {
    std::string executor_id;
    std::string executor_instance_id;
    server_warm_tier_executor_role role = SERVER_WARM_TIER_EXECUTOR_GPU;
    uint32_t order = 0;
    uint32_t credits = 0;
    std::string socket_path;
    int64_t expected_peer_pid = 0;
    uint64_t expected_peer_start_time_ticks = 0;
    int32_t timeout_ms = 0;
    size_t execute_concurrency = 0;
    size_t queue_capacity = 0;
    size_t output_limit_bytes = 0;
};

struct runtime_route_config {
    std::string model_id;
    std::string executor_id;
    server_warm_tier_model_state state = SERVER_WARM_TIER_MODEL_ABSENT;
};

struct runtime_config {
    std::string configuration;
    std::string run_id;
    std::string event_log_path;
    std::string config_sha256;
    bool promotion_enabled = false;
    std::vector<runtime_executor_config> executors;
    std::vector<runtime_route_config> initial_models;
};

struct runtime_callback_state {
    std::mutex mutex;
    server_warm_tier_controller * controller = nullptr;
};

bool exact_keys(
        const json & value,
        std::initializer_list<const char *> expected) {
    if (!value.is_object() || value.size() != expected.size()) {
        return false;
    }
    for (const char * key : expected) {
        if (!value.contains(key)) {
            return false;
        }
    }
    return true;
}

bool is_sha256(const std::string & value) {
    if (value.size() != 64) {
        return false;
    }
    for (char character : value) {
        if (!((character >= '0' && character <= '9')
                    || (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

std::string string_field(
        const json & value,
        const char * key,
        size_t maximum_length = 4096) {
    const auto found = value.find(key);
    if (found == value.end() || !found->is_string()) {
        throw std::runtime_error(std::string("invalid string field: ") + key);
    }
    const std::string result = found->get<std::string>();
    if (result.empty() || result.size() > maximum_length) {
        throw std::runtime_error(std::string("string field out of range: ") + key);
    }
    for (unsigned char character : result) {
        if (character < 0x20 || character > 0x7e) {
            throw std::runtime_error(std::string("non-ASCII string field: ") + key);
        }
    }
    return result;
}

uint64_t integer_field(
        const json & value,
        const char * key,
        uint64_t minimum,
        uint64_t maximum) {
    const auto found = value.find(key);
    if (found == value.end()
            || !(found->is_number_integer() || found->is_number_unsigned())) {
        throw std::runtime_error(std::string("invalid integer field: ") + key);
    }
    if (!found->is_number_unsigned() && found->get<int64_t>() < 0) {
        throw std::runtime_error(std::string("integer field out of range: ") + key);
    }
    uint64_t result = 0;
    try {
        result = found->get<uint64_t>();
    } catch (const json::exception &) {
        throw std::runtime_error(std::string("integer field out of range: ") + key);
    }
    if (result < minimum || result > maximum) {
        throw std::runtime_error(std::string("integer field out of range: ") + key);
    }
    return result;
}

bool boolean_field(const json & value, const char * key) {
    const auto found = value.find(key);
    if (found == value.end() || !found->is_boolean()) {
        throw std::runtime_error(std::string("invalid boolean field: ") + key);
    }
    return found->get<bool>();
}

server_warm_tier_model_state parse_state(const std::string & value) {
    if (value == "ABSENT") {
        return SERVER_WARM_TIER_MODEL_ABSENT;
    }
    if (value == "READY") {
        return SERVER_WARM_TIER_MODEL_READY;
    }
    throw std::runtime_error("invalid initial model state");
}

server_warm_tier_executor_role parse_role(const std::string & value) {
    if (value == "GPU") {
        return SERVER_WARM_TIER_EXECUTOR_GPU;
    }
    if (value == "CPU") {
        return SERVER_WARM_TIER_EXECUTOR_CPU;
    }
    if (value == "PHONE") {
        return SERVER_WARM_TIER_EXECUTOR_PHONE;
    }
    throw std::runtime_error("invalid executor role");
}

std::string read_config(const std::string & path) {
    std::ifstream source(path, std::ios::binary);
    if (!source) {
        throw std::runtime_error("cannot open config");
    }
    source.seekg(0, std::ios::end);
    const std::streamoff length = source.tellg();
    if (length <= 0 || static_cast<uint64_t>(length) > MAX_CONFIG_BYTES) {
        throw std::runtime_error("config size is invalid");
    }
    source.seekg(0, std::ios::beg);
    std::string raw(static_cast<size_t>(length), '\0');
    if (!source.read(raw.data(), length)) {
        throw std::runtime_error("cannot read config");
    }
    return raw;
}

std::string sha256_bytes(const unsigned char * data, size_t size) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    unsigned char digest[SHA256_DIGEST_LENGTH];
    if (SHA256(data, size, digest) == nullptr) {
        throw std::runtime_error("SHA-256 failed");
    }
    static const char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(SHA256_DIGEST_LENGTH * 2);
    for (unsigned char byte : digest) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 0x0f]);
    }
    return result;
#else
    (void) data;
    (void) size;
    throw std::runtime_error("warm-tier runtime requires LLAMA_OPENSSL");
#endif
}

class duplicate_key_scanner {
public:
    explicit duplicate_key_scanner(std::string_view input) : input(input) {}

    bool scan() {
        skip_space();
        parse_value();
        skip_space();
        if (position != input.size()) {
            throw std::runtime_error("invalid trailing JSON");
        }
        return duplicate;
    }

private:
    void skip_space() {
        while (position < input.size()
                && (input[position] == ' '
                    || input[position] == '\n'
                    || input[position] == '\r'
                    || input[position] == '\t')) {
            ++position;
        }
    }

    void consume(char expected) {
        skip_space();
        if (position >= input.size() || input[position] != expected) {
            throw std::runtime_error("invalid JSON structure");
        }
        ++position;
    }

    std::string parse_string() {
        skip_space();
        const size_t start = position;
        consume('"');
        bool escaped = false;
        while (position < input.size()) {
            const char character = input[position++];
            if (escaped) {
                escaped = false;
            } else if (character == '\\') {
                escaped = true;
            } else if (character == '"') {
                try {
                    return json::parse(
                        input.substr(start, position - start)).get<std::string>();
                } catch (const json::exception &) {
                    throw std::runtime_error("invalid JSON string");
                }
            }
        }
        throw std::runtime_error("unterminated JSON string");
    }

    void parse_object() {
        consume('{');
        skip_space();
        if (position < input.size() && input[position] == '}') {
            ++position;
            return;
        }
        std::set<std::string> keys;
        while (true) {
            const std::string key = parse_string();
            if (!keys.insert(key).second) {
                duplicate = true;
            }
            consume(':');
            parse_value();
            skip_space();
            if (position < input.size() && input[position] == '}') {
                ++position;
                return;
            }
            consume(',');
        }
    }

    void parse_array() {
        consume('[');
        skip_space();
        if (position < input.size() && input[position] == ']') {
            ++position;
            return;
        }
        while (true) {
            parse_value();
            skip_space();
            if (position < input.size() && input[position] == ']') {
                ++position;
                return;
            }
            consume(',');
        }
    }

    void parse_scalar() {
        skip_space();
        const size_t start = position;
        while (position < input.size()
                && input[position] != ','
                && input[position] != ']'
                && input[position] != '}'
                && input[position] != ' '
                && input[position] != '\n'
                && input[position] != '\r'
                && input[position] != '\t') {
            ++position;
        }
        if (position == start) {
            throw std::runtime_error("invalid JSON scalar");
        }
    }

    void parse_value() {
        skip_space();
        if (position >= input.size()) {
            throw std::runtime_error("missing JSON value");
        }
        switch (input[position]) {
            case '{': parse_object(); return;
            case '[': parse_array(); return;
            case '"': (void) parse_string(); return;
            default: parse_scalar(); return;
        }
    }

    std::string_view input;
    size_t position = 0;
    bool duplicate = false;
};

runtime_config parse_config(const std::string & path) {
    const std::string raw = read_config(path);
    json root;
    try {
        root = json::parse(raw);
    } catch (const json::exception & exc) {
        throw std::runtime_error(std::string("invalid config JSON: ") + exc.what());
    }
    if (duplicate_key_scanner(raw).scan()) {
        throw std::runtime_error("config contains a duplicate key");
    }
    if (root.dump() + "\n" != raw) {
        throw std::runtime_error("config is not canonical JSON");
    }
    if (!exact_keys(
            root,
            {
                "c3_profile_lock_sha256",
                "configuration",
                "evidence_root_sha256",
                "event_log_path",
                "executors",
                "initial_models",
                "promotion_enabled",
                "run_id",
                "runtime_plan_sha256",
                "schema",
            })) {
        throw std::runtime_error("config fields do not match schema");
    }
    if (string_field(root, "schema", 64)
            != "llama-server-warm-tier-runtime-v4") {
        throw std::runtime_error("unsupported config schema");
    }
    const std::string configuration =
        string_field(root, "configuration", 64);
    static const std::set<std::string> configurations = {
        "C1_GPU_ONLY_OPTIMIZED",
        "C2_GPU_PLUS_CPU_WARM_EXECUTOR",
        "C3_DUAL_PARTIAL_OFFLOAD",
        "T1_PHONE_WARM_TIER",
        "T2_PHONE_NO_PROMOTION",
    };
    if (configurations.count(configuration) == 0) {
        throw std::runtime_error("invalid configuration");
    }
    const std::string runtime_plan_sha256 =
        string_field(root, "runtime_plan_sha256", 64);
    const std::string evidence_root_sha256 =
        string_field(root, "evidence_root_sha256", 64);
    if (!is_sha256(runtime_plan_sha256)
            || !is_sha256(evidence_root_sha256)) {
        throw std::runtime_error("invalid evidence digest");
    }
    const auto profile_lock = root.find("c3_profile_lock_sha256");
    if (configuration == "C3_DUAL_PARTIAL_OFFLOAD") {
        if (profile_lock == root.end()
                || !profile_lock->is_string()
                || !is_sha256(profile_lock->get<std::string>())) {
            throw std::runtime_error("C3 requires a profile lock digest");
        }
    } else if (profile_lock == root.end() || !profile_lock->is_null()) {
        throw std::runtime_error("non-C3 configuration has a profile lock");
    }

    runtime_config result;
    result.configuration = configuration;
    result.config_sha256 = sha256_bytes(
        reinterpret_cast<const unsigned char *>(raw.data()),
        raw.size());
    result.run_id = string_field(root, "run_id", 128);
    result.event_log_path = string_field(root, "event_log_path");
    result.promotion_enabled = boolean_field(root, "promotion_enabled");
    if (result.event_log_path.front() != '/') {
        throw std::runtime_error("event log path must be absolute");
    }

    const json & executors = root["executors"];
    if (!executors.is_array() || executors.empty() || executors.size() > 16) {
        throw std::runtime_error("executor count is invalid");
    }
    std::set<std::string> executor_ids;
    std::set<std::string> executor_instance_ids;
    std::set<uint32_t> executor_orders;
    for (const json & value : executors) {
        if (!exact_keys(
                value,
                {
                    "credits",
                    "execute_concurrency",
                    "executor_id",
                    "executor_instance_id",
                    "expected_peer_pid",
                    "expected_peer_start_time_ticks",
                    "order",
                    "output_limit_bytes",
                    "queue_capacity",
                    "role",
                    "socket_path",
                    "timeout_ms",
                    "transport",
                })) {
            throw std::runtime_error("executor fields do not match schema");
        }
        runtime_executor_config executor;
        executor.executor_id = string_field(value, "executor_id", 128);
        executor.executor_instance_id =
            string_field(value, "executor_instance_id", 256);
        if (!executor_ids.insert(executor.executor_id).second) {
            throw std::runtime_error("duplicate executor id");
        }
        if (!executor_instance_ids.insert(
                    executor.executor_instance_id).second) {
            throw std::runtime_error("duplicate executor instance id");
        }
        executor.role = parse_role(string_field(value, "role", 16));
        executor.order = static_cast<uint32_t>(
            integer_field(
                value,
                "order",
                0,
                std::numeric_limits<uint32_t>::max()));
        if (!executor_orders.insert(executor.order).second) {
            throw std::runtime_error("duplicate executor order");
        }
        if (string_field(value, "transport", 32) != "UNIX_SOCKET") {
            throw std::runtime_error("executor transport is invalid");
        }
        executor.socket_path = string_field(value, "socket_path", 103);
        if (executor.socket_path.front() != '/') {
            throw std::runtime_error("executor socket path must be absolute");
        }
        executor.timeout_ms = static_cast<int32_t>(
            integer_field(value, "timeout_ms", 1, 3'600'000));
        executor.expected_peer_pid = static_cast<int64_t>(
            integer_field(
                value,
                "expected_peer_pid",
                1,
                std::numeric_limits<int32_t>::max()));
        executor.expected_peer_start_time_ticks = integer_field(
            value,
            "expected_peer_start_time_ticks",
            1,
            std::numeric_limits<uint64_t>::max());
        executor.execute_concurrency = static_cast<size_t>(
            integer_field(value, "execute_concurrency", 1, 128));
        executor.credits = static_cast<uint32_t>(
            integer_field(value, "credits", 1, 128));
        if (executor.credits > executor.execute_concurrency) {
            throw std::runtime_error(
                "executor credits exceed execute concurrency");
        }
        executor.queue_capacity = static_cast<size_t>(
            integer_field(value, "queue_capacity", 1, 4096));
        if (executor.queue_capacity < executor.credits) {
            throw std::runtime_error(
                "executor queue capacity is below credits");
        }
        executor.output_limit_bytes = static_cast<size_t>(
            integer_field(value, "output_limit_bytes", 1024, 64 * 1024 * 1024));
        result.executors.push_back(std::move(executor));
    }

    const json & routes = root["initial_models"];
    if (!routes.is_array() || routes.empty() || routes.size() > 256) {
        throw std::runtime_error("initial model route count is invalid");
    }
    std::set<std::pair<std::string, std::string>> route_keys;
    for (const json & value : routes) {
        if (!exact_keys(value, {"executor_id", "model_id", "state"})) {
            throw std::runtime_error("initial model fields do not match schema");
        }
        runtime_route_config route;
        route.model_id = string_field(value, "model_id", 256);
        route.executor_id = string_field(value, "executor_id", 128);
        route.state = parse_state(string_field(value, "state", 16));
        if (executor_ids.count(route.executor_id) == 0) {
            throw std::runtime_error("initial model names an unknown executor");
        }
        if (!route_keys.emplace(route.model_id, route.executor_id).second) {
            throw std::runtime_error("duplicate initial model route");
        }
        result.initial_models.push_back(std::move(route));
    }

    std::vector<server_warm_tier_executor_role> expected_roles;
    bool expected_promotion = true;
    if (configuration == "C1_GPU_ONLY_OPTIMIZED") {
        expected_roles = {SERVER_WARM_TIER_EXECUTOR_GPU};
    } else if (configuration == "C2_GPU_PLUS_CPU_WARM_EXECUTOR") {
        expected_roles = {
            SERVER_WARM_TIER_EXECUTOR_GPU,
            SERVER_WARM_TIER_EXECUTOR_CPU,
        };
    } else if (configuration == "C3_DUAL_PARTIAL_OFFLOAD") {
        expected_roles = {SERVER_WARM_TIER_EXECUTOR_GPU};
        expected_promotion = false;
    } else {
        expected_roles = {
            SERVER_WARM_TIER_EXECUTOR_GPU,
            SERVER_WARM_TIER_EXECUTOR_PHONE,
        };
        expected_promotion = configuration == "T1_PHONE_WARM_TIER";
    }
    if (result.executors.size() != expected_roles.size()
            || result.promotion_enabled != expected_promotion) {
        throw std::runtime_error("configuration policy mismatch");
    }
    for (size_t index = 0; index < result.executors.size(); ++index) {
        if (result.executors[index].order != index
                || result.executors[index].role != expected_roles[index]) {
            throw std::runtime_error("configuration executor topology mismatch");
        }
    }
    for (size_t index = 1; index < result.executors.size(); ++index) {
        if (result.executors[0].credits < result.executors[index].credits) {
            throw std::runtime_error(
                "GPU credits are below warm executor credits");
        }
    }

    std::set<std::string> model_ids;
    for (const auto & route : result.initial_models) {
        model_ids.insert(route.model_id);
    }
    if (model_ids.size() != 2
            || result.initial_models.size()
                != model_ids.size() * result.executors.size()) {
        throw std::runtime_error("configuration model matrix mismatch");
    }

    if (configuration == "C3_DUAL_PARTIAL_OFFLOAD") {
        for (const auto & route : result.initial_models) {
            if (route.state != SERVER_WARM_TIER_MODEL_READY) {
                throw std::runtime_error("configuration C3 residency mismatch");
            }
        }
    } else {
        std::string hot_model_id;
        for (const auto & route : result.initial_models) {
            if (route.executor_id == result.executors[0].executor_id
                    && route.state == SERVER_WARM_TIER_MODEL_READY) {
                if (!hot_model_id.empty()) {
                    throw std::runtime_error(
                        "configuration GPU residency mismatch");
                }
                hot_model_id = route.model_id;
            }
        }
        if (hot_model_id.empty()) {
            throw std::runtime_error("configuration GPU residency mismatch");
        }
        for (const auto & route : result.initial_models) {
            const bool is_gpu =
                route.executor_id == result.executors[0].executor_id;
            const bool expected_ready =
                is_gpu == (route.model_id == hot_model_id);
            if ((route.state == SERVER_WARM_TIER_MODEL_READY)
                    != expected_ready) {
                throw std::runtime_error(
                    "configuration initial residency mismatch");
            }
        }
    }
    return result;
}

struct event_log {
    explicit event_log(const std::string & path) {
        file = std::fopen(path.c_str(), "wx");
        if (file == nullptr) {
            throw std::runtime_error("cannot exclusively create event log");
        }
    }

    ~event_log() {
        if (file != nullptr) {
            std::fclose(file);
        }
    }

    void append(const server_warm_tier_event & event) {
        const std::string row = server_warm_tier_event_jsonl(event);
        std::lock_guard<std::mutex> lock(mutex);
        if (std::fwrite(row.data(), 1, row.size(), file) != row.size()
                || std::fflush(file) != 0) {
            throw std::runtime_error("cannot append event log");
        }
        const int descriptor =
#ifdef _WIN32
            _fileno(file);
        if (descriptor < 0 || _commit(descriptor) != 0) {
#else
            fileno(file);
        if (descriptor < 0 || fsync(descriptor) != 0) {
#endif
            throw std::runtime_error("cannot sync event log");
        }
    }

    FILE * file = nullptr;
    std::mutex mutex;
};

void append_u64_le(std::vector<unsigned char> & bytes, uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<unsigned char>((value >> shift) & 0xff));
    }
}

void append_tokens(
        std::vector<unsigned char> & bytes,
        const std::vector<llama_token> & tokens) {
    append_u64_le(bytes, tokens.size());
    for (llama_token token : tokens) {
        const uint32_t value = static_cast<uint32_t>(token);
        for (unsigned int shift = 0; shift < 32; shift += 8) {
            bytes.push_back(
                static_cast<unsigned char>((value >> shift) & 0xff));
        }
    }
}

std::string history_sha256_impl(
        const std::vector<llama_token> & prompt_tokens,
        const std::vector<llama_token> & committed_output_tokens) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    static const char domain[] = "s40-token-history-v1";
    std::vector<unsigned char> bytes(
        reinterpret_cast<const unsigned char *>(domain),
        reinterpret_cast<const unsigned char *>(domain) + sizeof(domain) - 1);
    append_tokens(bytes, prompt_tokens);
    append_tokens(bytes, committed_output_tokens);
    unsigned char digest[SHA256_DIGEST_LENGTH];
    if (SHA256(bytes.data(), bytes.size(), digest) == nullptr) {
        throw std::runtime_error("SHA-256 failed");
    }
    static const char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(SHA256_DIGEST_LENGTH * 2);
    for (unsigned char byte : digest) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 0x0f]);
    }
    return result;
#else
    (void) prompt_tokens;
    (void) committed_output_tokens;
    throw std::runtime_error("warm-tier runtime requires LLAMA_OPENSSL");
#endif
}

int64_t monotonic_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

std::string server_warm_tier_history_sha256(
        const std::vector<llama_token> & prompt_tokens,
        const std::vector<llama_token> & committed_output_tokens) {
    return history_sha256_impl(prompt_tokens, committed_output_tokens);
}

std::string server_warm_tier_internal_token_from_env() {
    const char * path =
        std::getenv("LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE");
    if (path == nullptr || path[0] == '\0') {
        throw std::runtime_error(
            "warm-tier internal token file is not configured");
    }
#ifdef _WIN32
    throw std::runtime_error(
        "warm-tier internal token files require a Unix host");
#else
    const int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        throw std::runtime_error(
            "failed to open warm-tier internal token file");
    }

    struct stat before {};
    if (fstat(fd, &before) != 0
            || !S_ISREG(before.st_mode)
            || before.st_uid != geteuid()
            || before.st_nlink != 1
            || (before.st_mode & 0077) != 0
            || (before.st_mode & S_IRUSR) == 0
            || before.st_size != 65) {
        close(fd);
        throw std::runtime_error(
            "warm-tier internal token file has unsafe metadata");
    }

    std::string raw(65, '\0');
    size_t offset = 0;
    while (offset < raw.size()) {
        const ssize_t count =
            read(fd, raw.data() + offset, raw.size() - offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            close(fd);
            throw std::runtime_error(
                "warm-tier internal token file is truncated");
        }
        offset += static_cast<size_t>(count);
    }
    char extra = '\0';
    ssize_t extra_count;
    do {
        extra_count = read(fd, &extra, 1);
    } while (extra_count < 0 && errno == EINTR);

    struct stat after {};
    const bool stable =
        fstat(fd, &after) == 0
        && before.st_dev == after.st_dev
        && before.st_ino == after.st_ino
        && before.st_mode == after.st_mode
        && before.st_uid == after.st_uid
        && before.st_nlink == after.st_nlink
        && before.st_size == after.st_size;
    close(fd);
    if (extra_count != 0 || !stable || raw.back() != '\n') {
        throw std::runtime_error(
            "warm-tier internal token file changed while reading");
    }
    raw.pop_back();
    if (!is_sha256(raw)) {
        throw std::runtime_error(
            "warm-tier internal token is not lowercase hexadecimal");
    }
    return raw;
#endif
}

std::shared_ptr<server_warm_tier_controller>
server_warm_tier_create_runtime_from_env() {
    const char * path = std::getenv("LLAMA_SERVER_WARM_TIER_CONFIG");
    if (path == nullptr || path[0] == '\0') {
        return nullptr;
    }

    try {
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
        throw std::runtime_error("warm-tier runtime requires LLAMA_OPENSSL");
#endif
#ifdef _WIN32
        throw std::runtime_error(
            "warm-tier runtime requires Unix-domain sockets");
#endif
        const runtime_config config = parse_config(path);
        auto log = std::make_shared<event_log>(config.event_log_path);
        server_warm_tier_options options;
        options.enabled = true;
        options.run_id = config.run_id;
        options.runtime_config_sha256 = config.config_sha256;
        options.clock = monotonic_ns;
        options.history_digest = server_warm_tier_history_sha256;
        options.event_sink = [log](const server_warm_tier_event & event) {
            log->append(event);
        };
        auto callbacks = std::make_shared<runtime_callback_state>();
        auto controller = std::shared_ptr<server_warm_tier_controller>(
            new server_warm_tier_controller(std::move(options)),
            [callbacks](server_warm_tier_controller * value) {
                {
                    std::lock_guard<std::mutex> lock(callbacks->mutex);
                    callbacks->controller = nullptr;
                }
                delete value;
            });
        callbacks->controller = controller.get();
        if (!controller->set_promotion_enabled(config.promotion_enabled)) {
            throw std::runtime_error("cannot install promotion policy");
        }
        for (const runtime_executor_config & value : config.executors) {
            server_warm_tier_unix_executor_options executor_options;
            executor_options.executor_id = value.executor_id;
            executor_options.executor_instance_id =
                value.executor_instance_id;
            executor_options.socket_path = value.socket_path;
            executor_options.expected_peer_pid = value.expected_peer_pid;
            executor_options.expected_peer_start_time_ticks =
                value.expected_peer_start_time_ticks;
            executor_options.timeout_ms = value.timeout_ms;
            executor_options.execute_concurrency = value.execute_concurrency;
            executor_options.queue_capacity = value.queue_capacity;
            executor_options.output_limit_bytes = value.output_limit_bytes;
            executor_options.result_sink =
                [callbacks](server_warm_tier_result result) {
                    std::lock_guard<std::mutex> lock(callbacks->mutex);
                    if (callbacks->controller == nullptr) {
                        return;
                    }
                    if (!callbacks->controller->handle_executor_result(result)) {
                        std::fprintf(
                            stderr,
                            "warm-tier controller rejected executor result: %s\n",
                            callbacks->controller->last_error().c_str());
                    }
                };
            auto executor = server_warm_tier_create_unix_executor(
                std::move(executor_options));
            if (!executor || !controller->register_executor(std::move(executor))) {
                throw std::runtime_error("cannot register executor");
            }
            server_warm_tier_executor_policy policy;
            policy.executor_id = value.executor_id;
            policy.role = value.role;
            policy.order = value.order;
            policy.credits = value.credits;
            if (!controller->set_executor_policy(std::move(policy))) {
                throw std::runtime_error("cannot install executor policy");
            }
        }
        for (const runtime_route_config & route : config.initial_models) {
            if (!controller->set_initial_model_state(
                    route.model_id,
                    route.executor_id,
                    SERVER_WARM_TIER_MODEL_ABSENT)) {
                throw std::runtime_error("cannot install initial model route");
            }
            if (route.state == SERVER_WARM_TIER_MODEL_READY
                    && !controller->set_bootstrap_model_ready(
                        route.model_id,
                        route.executor_id)) {
                throw std::runtime_error("cannot install bootstrap model route");
            }
        }
        if (!controller->start()) {
            throw std::runtime_error(
                "cannot start warm-tier controller: "
                + controller->last_error());
        }
        return controller;
    } catch (const std::exception & exc) {
        std::fprintf(
            stderr,
            "warm-tier runtime refused config %s: %s\n",
            path,
            exc.what());
        return nullptr;
    }
}
