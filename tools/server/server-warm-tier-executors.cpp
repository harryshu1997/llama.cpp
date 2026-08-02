#include "server-warm-tier-executors.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <type_traits>
#include <utility>

#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif


namespace {

using json = nlohmann::json;

constexpr size_t MAX_COMMAND_BYTES = 4 * 1024 * 1024;
#ifdef MSG_NOSIGNAL
constexpr int SOCKET_SEND_FLAGS = MSG_NOSIGNAL;
#else
constexpr int SOCKET_SEND_FLAGS = 0;
#endif

struct transport_result {
    int exit_code = -1;
    bool timed_out = false;
    bool truncated = false;
    std::string failure;
    std::string output;
};

static json request_to_json(const server_warm_tier_request_snapshot & request) {
    return {
        {"committed_output_tokens", request.committed_output_tokens},
        {"model_id", request.model_id},
        {"owner_id", request.owner_id},
        {"ownership_epoch", request.ownership_epoch},
        {"position", request.position},
        {"prompt_tokens", request.prompt_tokens},
        {"publication_index", request.publication_index},
        {"request_id", request.request_id},
        {"state", static_cast<int>(request.state)},
    };
}

static json command_to_json(const server_warm_tier_command & command) {
    return {
        {"command_id", command.command_id},
        {"controller_epoch", command.controller_epoch},
        {"executor_id", command.executor_id},
        {"executor_instance_id", command.executor_instance_id},
        {"kind", static_cast<int>(command.kind)},
        {"max_output_tokens", command.max_output_tokens},
        {"model_id", command.model_id},
        {"request", request_to_json(command.request)},
        {"request_id", command.request_id},
        {"schema", "llama-server-warm-tier-command-v3"},
        {"total_output_tokens", command.total_output_tokens},
    };
}

static bool exact_keys(
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

template<typename T>
static bool get_exact(const json & value, const char * key, T & output) {
    const auto found = value.find(key);
    if (found == value.end()) {
        return false;
    }
    if constexpr (std::is_same_v<T, bool>) {
        if (!found->is_boolean()) {
            return false;
        }
    } else if constexpr (std::is_integral_v<T>) {
        if (!found->is_number_integer()) {
            return false;
        }
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (!found->is_string()) {
            return false;
        }
    }
    try {
        output = found->get<T>();
        return true;
    } catch (const json::exception &) {
        return false;
    }
}

static bool get_token_vector(
        const json & value,
        const char * key,
        std::vector<llama_token> & output) {
    const auto found = value.find(key);
    if (found == value.end() || !found->is_array()) {
        return false;
    }
    output.clear();
    output.reserve(found->size());
    for (const json & item : *found) {
        if (!item.is_number_integer()) {
            return false;
        }
        try {
            const llama_token token = item.get<llama_token>();
            if (token < 0) {
                return false;
            }
            output.push_back(token);
        } catch (const json::exception &) {
            return false;
        }
    }
    return true;
}

static bool parse_request(
        const json & value,
        server_warm_tier_request_snapshot & request) {
    if (!exact_keys(
            value,
            {
                "committed_output_tokens",
                "model_id",
                "owner_id",
                "ownership_epoch",
                "position",
                "prompt_tokens",
                "publication_index",
                "request_id",
                "state",
            })) {
        return false;
    }
    int state = -1;
    if (!get_exact(value, "request_id", request.request_id)
            || !get_exact(value, "model_id", request.model_id)
            || !get_token_vector(value, "prompt_tokens", request.prompt_tokens)
            || !get_token_vector(
                value,
                "committed_output_tokens",
                request.committed_output_tokens)
            || !get_exact(value, "position", request.position)
            || !get_exact(value, "owner_id", request.owner_id)
            || !get_exact(value, "ownership_epoch", request.ownership_epoch)
            || !get_exact(value, "publication_index", request.publication_index)
            || !get_exact(value, "state", state)
            || state < SERVER_WARM_TIER_REQUEST_QUEUED
            || state > SERVER_WARM_TIER_REQUEST_STRANDED) {
        return false;
    }
    request.state = static_cast<server_warm_tier_request_state>(state);
    return true;
}

static bool parse_publication(
        const json & value,
        server_warm_tier_publication & publication) {
    if (!exact_keys(
            value,
            {
                "owner_id",
                "ownership_epoch",
                "position",
                "publication_index",
                "token",
            })) {
        return false;
    }
    int64_t token = 0;
    if (!get_exact(value, "owner_id", publication.owner_id)
            || !get_exact(value, "ownership_epoch", publication.ownership_epoch)
            || !get_exact(value, "position", publication.position)
            || !get_exact(value, "publication_index", publication.publication_index)
            || !get_exact(value, "token", token)
            || token < std::numeric_limits<llama_token>::min()
            || token > std::numeric_limits<llama_token>::max()) {
        return false;
    }
    publication.token = static_cast<llama_token>(token);
    return true;
}

static bool parse_result(
        const std::string & raw,
        const server_warm_tier_command & command,
        server_warm_tier_result & result,
        std::string & error) {
    json value;
    try {
        value = json::parse(raw);
    } catch (const json::exception & exc) {
        error = std::string("executor returned invalid JSON: ") + exc.what();
        return false;
    }
    if (value.dump() + "\n" != raw) {
        error = "executor result is not canonical JSON";
        return false;
    }
    if (!exact_keys(
            value,
            {
                "command_id",
                "controller_epoch",
                "detail",
                "executor_id",
                "executor_instance_id",
                "has_replay_snapshot",
                "kind",
                "model_id",
                "publications",
                "replay_snapshot",
                "request_complete",
                "request_id",
                "schema",
                "success",
            })) {
        error = "executor result fields do not match schema";
        return false;
    }

    std::string schema;
    int kind = -1;
    if (!get_exact(value, "schema", schema)
            || schema != "llama-server-warm-tier-result-v2"
            || !get_exact(value, "command_id", result.command_id)
            || !get_exact(value, "controller_epoch", result.controller_epoch)
            || !get_exact(value, "kind", kind)
            || !get_exact(value, "model_id", result.model_id)
            || !get_exact(value, "request_id", result.request_id)
            || !get_exact(value, "executor_id", result.executor_id)
            || !get_exact(
                value,
                "executor_instance_id",
                result.executor_instance_id)
            || !get_exact(value, "success", result.success)
            || !get_exact(value, "detail", result.detail)
            || !get_exact(
                value,
                "has_replay_snapshot",
                result.has_replay_snapshot)
            || !get_exact(value, "request_complete", result.request_complete)
            || kind < SERVER_WARM_TIER_COMMAND_EXECUTE
            || kind > SERVER_WARM_TIER_COMMAND_CLEANUP) {
        error = "executor result has invalid field types";
        return false;
    }
    result.kind = static_cast<server_warm_tier_command_kind>(kind);
    if (result.command_id != command.command_id
            || result.controller_epoch != command.controller_epoch
            || result.kind != command.kind
            || result.model_id != command.model_id
            || result.request_id != command.request_id
            || result.executor_id != command.executor_id
            || result.executor_instance_id
                != command.executor_instance_id) {
        error = "executor result does not match submitted command";
        return false;
    }
    const json & publications = value["publications"];
    if (!publications.is_array()) {
        error = "executor result publications are invalid";
        return false;
    }
    for (const json & item : publications) {
        server_warm_tier_publication publication;
        if (!parse_publication(item, publication)) {
            error = "executor result contains an invalid publication";
            return false;
        }
        result.publications.push_back(std::move(publication));
    }
    const json & replay = value["replay_snapshot"];
    if (result.has_replay_snapshot) {
        if (!parse_request(replay, result.replay_snapshot)) {
            error = "executor result contains an invalid replay snapshot";
            return false;
        }
    } else if (!replay.is_null()) {
        error = "executor returned an unmarked replay snapshot";
        return false;
    }
    if (command.kind == SERVER_WARM_TIER_COMMAND_REPLAY
            && result.success
            && !result.has_replay_snapshot) {
        error = "replay result omitted its exact snapshot";
        return false;
    }
    if (command.kind != SERVER_WARM_TIER_COMMAND_REPLAY
            && result.has_replay_snapshot) {
        error = "non-replay result contains a replay snapshot";
        return false;
    }
    if (command.kind != SERVER_WARM_TIER_COMMAND_EXECUTE
            && (!result.publications.empty() || result.request_complete)) {
        error = "non-execute executor result contains request output";
        return false;
    }
    if (command.kind == SERVER_WARM_TIER_COMMAND_EXECUTE
            && result.success
            && static_cast<int32_t>(result.publications.size())
                    > command.max_output_tokens) {
        error = "executor returned too many publications";
        return false;
    }
    return true;
}

#ifndef _WIN32

class unix_socket_transport {
public:
    explicit unix_socket_transport(
            std::string socket_path,
            int64_t expected_peer_pid,
            uint64_t expected_peer_start_time_ticks)
        : socket_path(std::move(socket_path)),
          expected_peer_pid(expected_peer_pid),
          expected_peer_start_time_ticks(expected_peer_start_time_ticks) {
    }

    transport_result run(
            const std::string & input,
            size_t output_limit,
            int32_t timeout_ms) {
        transport_result result;
        if (input.size() > MAX_COMMAND_BYTES) {
            result.failure = "command exceeded transport limit";
            return result;
        }

        const int descriptor = socket(AF_UNIX, SOCK_STREAM, 0);
        if (descriptor < 0) {
            result.failure = socket_error("socket");
            return result;
        }
#ifdef SO_NOSIGPIPE
        const int no_sigpipe = 1;
        if (setsockopt(
                descriptor,
                SOL_SOCKET,
                SO_NOSIGPIPE,
                &no_sigpipe,
                sizeof(no_sigpipe)) < 0) {
            result.failure = socket_error("setsockopt");
            close(descriptor);
            return result;
        }
#endif
        descriptor_guard guard{*this, descriptor};
        if (!register_descriptor(descriptor)) {
            result.failure = "transport is stopping";
            return result;
        }

        const int descriptor_flags = fcntl(descriptor, F_GETFD, 0);
        const int status_flags = fcntl(descriptor, F_GETFL, 0);
        if (descriptor_flags < 0
                || status_flags < 0
                || fcntl(
                    descriptor,
                    F_SETFD,
                    descriptor_flags | FD_CLOEXEC) < 0
                || fcntl(
                    descriptor,
                    F_SETFL,
                    status_flags | O_NONBLOCK) < 0) {
            result.failure = socket_error("fcntl");
            return result;
        }

        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::memcpy(
            address.sun_path,
            socket_path.c_str(),
            socket_path.size() + 1);
        const auto deadline = std::chrono::steady_clock::now()
                            + std::chrono::milliseconds(timeout_ms);
        if (connect(
                descriptor,
                reinterpret_cast<const sockaddr *>(&address),
                sizeof(address)) < 0) {
            if (errno != EINPROGRESS) {
                result.failure = socket_error("connect");
                return result;
            }
            if (!wait_for(descriptor, POLLOUT, deadline, result)) {
                return result;
            }
            int socket_status = 0;
            socklen_t status_size = sizeof(socket_status);
            if (getsockopt(
                    descriptor,
                    SOL_SOCKET,
                    SO_ERROR,
                    &socket_status,
                    &status_size) < 0) {
                result.failure = socket_error("getsockopt");
                return result;
            }
            if (socket_status != 0) {
                errno = socket_status;
                result.failure = socket_error("connect");
                return result;
            }
        }
        if (!verify_peer(descriptor, result)) {
            return result;
        }

        size_t sent = 0;
        while (sent < input.size()) {
            const ssize_t count = send(
                descriptor,
                input.data() + sent,
                input.size() - sent,
                SOCKET_SEND_FLAGS);
            if (count > 0) {
                sent += static_cast<size_t>(count);
                continue;
            }
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                if (!wait_for(descriptor, POLLOUT, deadline, result)) {
                    return result;
                }
                continue;
            }
            result.failure = socket_error("send");
            return result;
        }
        if (shutdown(descriptor, SHUT_WR) < 0) {
            result.failure = socket_error("shutdown");
            return result;
        }

        char buffer[4096];
        while (true) {
            const ssize_t count = recv(descriptor, buffer, sizeof(buffer), 0);
            if (count > 0) {
                const size_t length = static_cast<size_t>(count);
                if (result.output.size() + length > output_limit) {
                    const size_t remaining =
                        output_limit - result.output.size();
                    result.output.append(buffer, remaining);
                    result.truncated = true;
                    return result;
                }
                result.output.append(buffer, length);
                continue;
            }
            if (count == 0) {
                result.exit_code = 0;
                return result;
            }
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                if (!wait_for(descriptor, POLLIN, deadline, result)) {
                    return result;
                }
                continue;
            }
            result.failure = socket_error("recv");
            return result;
        }
    }

    void cancel() {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
        for (int descriptor : descriptors) {
            shutdown(descriptor, SHUT_RDWR);
        }
    }

private:
    struct descriptor_guard {
        unix_socket_transport & transport;
        int descriptor;

        ~descriptor_guard() {
            transport.unregister_descriptor(descriptor);
            close(descriptor);
        }
    };

    static std::string socket_error(const char * operation) {
        return std::string(operation) + " failed: " + std::strerror(errno);
    }

    static bool read_process_start_time(
            int64_t process_id,
            uint64_t & start_time_ticks) {
        std::ifstream source(
            "/proc/" + std::to_string(process_id) + "/stat",
            std::ios::binary);
        std::string raw;
        if (!source || !std::getline(source, raw) || raw.size() > 64 * 1024) {
            return false;
        }
        const size_t close = raw.rfind(')');
        if (close == std::string::npos
                || close + 2 >= raw.size()
                || raw[close + 1] != ' ') {
            return false;
        }
        std::istringstream fields(raw.substr(close + 2));
        std::string field;
        for (int number = 3; number <= 22; ++number) {
            if (!(fields >> field)) {
                return false;
            }
            if (number == 22) {
                try {
                    size_t consumed = 0;
                    start_time_ticks = std::stoull(field, &consumed, 10);
                    return consumed == field.size()
                        && start_time_ticks != 0;
                } catch (const std::exception &) {
                    return false;
                }
            }
        }
        return false;
    }

    bool verify_peer(int descriptor, transport_result & result) const {
#if defined(__linux__) && defined(SO_PEERCRED)
        struct peer_credentials {
            pid_t pid;
            uid_t uid;
            gid_t gid;
        };
        peer_credentials credentials{};
        socklen_t credentials_size = sizeof(credentials);
        if (getsockopt(
                descriptor,
                SOL_SOCKET,
                SO_PEERCRED,
                &credentials,
                &credentials_size) < 0
                || credentials_size != sizeof(credentials)) {
            result.failure = socket_error("peer credentials");
            return false;
        }
        if (static_cast<int64_t>(credentials.pid) != expected_peer_pid) {
            result.failure = "Unix socket peer PID changed";
            return false;
        }
        uint64_t start_time_ticks = 0;
        if (!read_process_start_time(
                    expected_peer_pid,
                    start_time_ticks)
                || start_time_ticks != expected_peer_start_time_ticks) {
            result.failure = "Unix socket peer start identity changed";
            return false;
        }
        return true;
#else
        (void) descriptor;
        result.failure = "Unix socket peer identity is unsupported";
        return false;
#endif
    }

    static bool wait_for(
            int descriptor,
            short events,
            const std::chrono::steady_clock::time_point & deadline,
            transport_result & result) {
        while (true) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                result.timed_out = true;
                return false;
            }
            const auto remaining = std::chrono::duration_cast<
                std::chrono::milliseconds>(deadline - now);
            const int timeout = static_cast<int>(
                std::max<int64_t>(1, remaining.count()));
            pollfd item{};
            item.fd = descriptor;
            item.events = events;
            const int status = poll(&item, 1, timeout);
            if (status > 0) {
                if ((item.revents & POLLNVAL) != 0) {
                    result.failure = "socket became invalid";
                    return false;
                }
                return true;
            }
            if (status == 0) {
                result.timed_out = true;
                return false;
            }
            if (errno != EINTR) {
                result.failure = socket_error("poll");
                return false;
            }
        }
    }

    bool register_descriptor(int descriptor) {
        std::lock_guard<std::mutex> lock(mutex);
        if (stopping) {
            return false;
        }
        descriptors.insert(descriptor);
        return true;
    }

    void unregister_descriptor(int descriptor) {
        std::lock_guard<std::mutex> lock(mutex);
        descriptors.erase(descriptor);
    }

    std::string socket_path;
    int64_t expected_peer_pid = 0;
    uint64_t expected_peer_start_time_ticks = 0;
    std::mutex mutex;
    std::set<int> descriptors;
    bool stopping = false;
};

#endif

struct queued_executor_options {
    std::string executor_id;
    std::string executor_instance_id;
    server_warm_tier_executor_result_sink result_sink;
    int32_t timeout_ms = 0;
    size_t execute_concurrency = 0;
    size_t queue_capacity = 0;
    size_t output_limit_bytes = 0;
    std::string transport_name;
    std::function<transport_result(
        const std::string &, size_t, int32_t)> run_transport;
    std::function<void()> cancel_transport;
};

class queued_executor final : public server_warm_tier_executor {
public:
    explicit queued_executor(queued_executor_options options)
        : state(std::make_shared<worker_state>(std::move(options))) {
        for (size_t i = 0; i < state->options.execute_concurrency; ++i) {
            workers.emplace_back(&queued_executor::run, state);
        }
    }

    ~queued_executor() override {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->stopping = true;
            state->commands.clear();
        }
        state->options.cancel_transport();
        state->condition.notify_all();
        for (std::thread & worker : workers) {
            if (worker.get_id() == std::this_thread::get_id()) {
                worker.detach();
            } else {
                worker.join();
            }
        }
    }

    const std::string & id() const override {
        return state->options.executor_id;
    }

    const std::string & instance_id() const override {
        return state->options.executor_instance_id;
    }

    bool submit(
            const server_warm_tier_command & command,
            std::string & error) override {
        if (command.executor_id != state->options.executor_id
                || command.executor_instance_id
                    != state->options.executor_instance_id
                || command.command_id == 0
                || command.kind < SERVER_WARM_TIER_COMMAND_EXECUTE
                || command.kind > SERVER_WARM_TIER_COMMAND_CLEANUP
                || (command.kind == SERVER_WARM_TIER_COMMAND_EXECUTE
                    && (command.max_output_tokens != 1
                        || command.total_output_tokens <= 0
                        || command.request.committed_output_tokens.size()
                            >= static_cast<size_t>(command.total_output_tokens)))
                || (command.kind != SERVER_WARM_TIER_COMMAND_EXECUTE
                    && (command.max_output_tokens != 0
                        || command.total_output_tokens != 0))) {
            error = "invalid executor command";
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->stopping) {
                error = "executor is stopping";
                return false;
            }
            if (state->commands.size() >= state->options.queue_capacity) {
                error = "executor queue is full";
                return false;
            }
            state->commands.push_back(command);
        }
        state->condition.notify_one();
        return true;
    }

private:
    struct worker_state {
        explicit worker_state(queued_executor_options options)
            : options(std::move(options)) {
        }

        queued_executor_options options;
        std::mutex mutex;
        std::condition_variable condition;
        std::deque<server_warm_tier_command> commands;
        size_t active_execute = 0;
        bool lifecycle_active = false;
        bool stopping = false;
    };

    struct finish_guard {
        std::shared_ptr<worker_state> state;
        server_warm_tier_command_kind kind;

        ~finish_guard() {
            queued_executor::finish_command(state, kind);
        }
    };

    static bool can_run_front(const worker_state & state) {
        if (state.commands.empty() || state.lifecycle_active) {
            return false;
        }
        if (state.commands.front().kind == SERVER_WARM_TIER_COMMAND_EXECUTE) {
            return state.active_execute < state.options.execute_concurrency;
        }
        return state.active_execute == 0;
    }

    static bool take_command(
            const std::shared_ptr<worker_state> & state,
            server_warm_tier_command & command) {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->condition.wait(lock, [&]() {
            return state->stopping || can_run_front(*state);
        });
        if (state->stopping) {
            return false;
        }
        command = std::move(state->commands.front());
        state->commands.pop_front();
        if (command.kind == SERVER_WARM_TIER_COMMAND_EXECUTE) {
            state->active_execute++;
        } else {
            state->lifecycle_active = true;
        }
        return true;
    }

    static void finish_command(
            const std::shared_ptr<worker_state> & state,
            server_warm_tier_command_kind kind) {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (kind == SERVER_WARM_TIER_COMMAND_EXECUTE) {
                state->active_execute--;
            } else {
                state->lifecycle_active = false;
            }
        }
        state->condition.notify_all();
    }

    static void run(std::shared_ptr<worker_state> state) {
        while (true) {
            server_warm_tier_command command;
            if (!take_command(state, command)) {
                return;
            }
            finish_guard guard{state, command.kind};
            server_warm_tier_result result;
            result.command_id = command.command_id;
            result.controller_epoch = command.controller_epoch;
            result.kind = command.kind;
            result.model_id = command.model_id;
            result.request_id = command.request_id;
            result.executor_id = command.executor_id;
            result.executor_instance_id = command.executor_instance_id;

            const transport_result transport = state->options.run_transport(
                command_to_json(command).dump() + "\n",
                state->options.output_limit_bytes,
                state->options.timeout_ms);
            std::string error;
            if (transport.timed_out) {
                result.detail = state->options.transport_name + " timed out";
            } else if (transport.truncated) {
                result.detail =
                    state->options.transport_name + " output exceeded limit";
            } else if (!transport.failure.empty()) {
                result.detail =
                    state->options.transport_name + ": " + transport.failure;
            } else if (transport.exit_code != 0) {
                result.detail = state->options.transport_name
                              + " exited with code "
                              + std::to_string(transport.exit_code);
            } else if (!parse_result(
                    transport.output,
                    command,
                    result,
                    error)) {
                result.success = false;
                result.publications.clear();
                result.request_complete = false;
                result.detail = std::move(error);
            }
            try {
                state->options.result_sink(std::move(result));
            } catch (const std::exception & exc) {
                std::fprintf(
                    stderr,
                    "warm-tier executor result sink failed: %s\n",
                    exc.what());
            } catch (...) {
                std::fprintf(stderr, "warm-tier executor result sink failed\n");
            }
        }
    }

    std::shared_ptr<worker_state> state;
    std::vector<std::thread> workers;
};

} // namespace

std::shared_ptr<server_warm_tier_executor>
server_warm_tier_create_unix_executor(
        server_warm_tier_unix_executor_options options) {
#ifdef _WIN32
    (void) options;
    return nullptr;
#else
    bool printable_path = true;
    for (unsigned char character : options.socket_path) {
        if (character < 0x20 || character > 0x7e) {
            printable_path = false;
            break;
        }
    }
    if (options.executor_id.empty()
            || options.executor_instance_id.empty()
            || options.socket_path.empty()
            || options.socket_path.front() != '/'
            || options.socket_path.size() >= sizeof(sockaddr_un{}.sun_path)
            || options.socket_path.find('\0') != std::string::npos
            || !printable_path
            || !options.result_sink
            || options.expected_peer_pid <= 0
            || options.expected_peer_start_time_ticks == 0
            || options.timeout_ms <= 0
            || options.execute_concurrency == 0
            || options.queue_capacity == 0
            || options.output_limit_bytes == 0) {
        return nullptr;
    }
    auto transport = std::make_shared<unix_socket_transport>(
        std::move(options.socket_path),
        options.expected_peer_pid,
        options.expected_peer_start_time_ticks);
    queued_executor_options queued;
    queued.executor_id = std::move(options.executor_id);
    queued.executor_instance_id =
        std::move(options.executor_instance_id);
    queued.result_sink = std::move(options.result_sink);
    queued.timeout_ms = options.timeout_ms;
    queued.execute_concurrency = options.execute_concurrency;
    queued.queue_capacity = options.queue_capacity;
    queued.output_limit_bytes = options.output_limit_bytes;
    queued.transport_name = "Unix socket";
    queued.run_transport =
        [transport](
                const std::string & input,
                size_t output_limit,
                int32_t timeout_ms) {
            return transport->run(input, output_limit, timeout_ms);
        };
    queued.cancel_transport = [transport]() {
        transport->cancel();
    };
    return std::make_shared<queued_executor>(std::move(queued));
#endif
}
