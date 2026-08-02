#include "server-warm-tier-executors.h"
#include "server-models.h"
#include "server-warm-tier-runtime.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#include <openssl/sha.h>
#endif

#ifdef MSG_NOSIGNAL
static constexpr int TEST_SOCKET_SEND_FLAGS = MSG_NOSIGNAL;
#else
static constexpr int TEST_SOCKET_SEND_FLAGS = 0;
#endif

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void test_child_environment_drops_warm_tier_capability() {
    const std::vector<std::string> input = {
        "A=1",
        "LLAMA_SERVER_WARM_TIER_CONFIG=/tmp/config",
        "LLAMA_SERVER_WARM_TIER_CONFIG_EXTRA=keep",
        "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE=/tmp/token",
        "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE_EXTRA=keep",
        "Z=2",
    };
    const std::vector<std::string> expected = {
        "A=1",
        "LLAMA_SERVER_WARM_TIER_CONFIG_EXTRA=keep",
        "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE_EXTRA=keep",
        "Z=2",
    };
    require(
        server_models_sanitize_child_environment(input) == expected,
        "warm-tier capability leaked into model child environment");
}

struct result_log {
    std::mutex mutex;
    std::condition_variable condition;
    std::vector<server_warm_tier_result> results;

    void add(server_warm_tier_result result) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            results.push_back(std::move(result));
        }
        condition.notify_all();
    }

    void wait(size_t count) {
        std::unique_lock<std::mutex> lock(mutex);
        const bool ready = condition.wait_for(
            lock,
            std::chrono::seconds(10),
            [&]() { return results.size() >= count; });
        require(ready, "timed out waiting for executor result");
    }
};

static server_warm_tier_command command(
        uint64_t command_id,
        server_warm_tier_command_kind kind,
        const std::string & model_id) {
    server_warm_tier_command result;
    result.command_id = command_id;
    result.controller_epoch = 0;
    result.kind = kind;
    result.model_id = model_id;
    result.request_id = kind == SERVER_WARM_TIER_COMMAND_EXECUTE ? "r1" : "";
    result.executor_id = "GPU";
    result.executor_instance_id = "GPU-instance";
    result.max_output_tokens =
        kind == SERVER_WARM_TIER_COMMAND_EXECUTE ? 1 : 0;
    result.total_output_tokens =
        kind == SERVER_WARM_TIER_COMMAND_EXECUTE ? 1 : 0;
    result.request.request_id = result.request_id;
    result.request.model_id = model_id;
    result.request.prompt_tokens = {1, 2, 3};
    result.request.position = 3;
    result.request.owner_id = "GPU";
    result.request.ownership_epoch = 1;
    result.request.state = SERVER_WARM_TIER_REQUEST_ACTIVE;
    return result;
}

static nlohmann::json command_json(
        const server_warm_tier_command & value) {
    return {
        {"command_id", value.command_id},
        {"controller_epoch", value.controller_epoch},
        {"executor_id", value.executor_id},
        {"executor_instance_id", value.executor_instance_id},
        {"kind", static_cast<int>(value.kind)},
        {"max_output_tokens", value.max_output_tokens},
        {"model_id", value.model_id},
        {"request", {
            {"committed_output_tokens",
                value.request.committed_output_tokens},
            {"model_id", value.request.model_id},
            {"owner_id", value.request.owner_id},
            {"ownership_epoch", value.request.ownership_epoch},
            {"position", value.request.position},
            {"prompt_tokens", value.request.prompt_tokens},
            {"publication_index", value.request.publication_index},
            {"request_id", value.request.request_id},
            {"state", static_cast<int>(value.request.state)},
        }},
        {"request_id", value.request_id},
        {"schema", "llama-server-warm-tier-command-v3"},
        {"total_output_tokens", value.total_output_tokens},
    };
}

static nlohmann::json result_json(
        const server_warm_tier_result & value) {
    nlohmann::json publications = nlohmann::json::array();
    for (const auto & publication : value.publications) {
        publications.push_back({
            {"owner_id", publication.owner_id},
            {"ownership_epoch", publication.ownership_epoch},
            {"position", publication.position},
            {"publication_index", publication.publication_index},
            {"token", publication.token},
        });
    }
    nlohmann::json replay = nullptr;
    if (value.has_replay_snapshot) {
        replay = {
            {"committed_output_tokens",
                value.replay_snapshot.committed_output_tokens},
            {"model_id", value.replay_snapshot.model_id},
            {"owner_id", value.replay_snapshot.owner_id},
            {"ownership_epoch", value.replay_snapshot.ownership_epoch},
            {"position", value.replay_snapshot.position},
            {"prompt_tokens", value.replay_snapshot.prompt_tokens},
            {"publication_index", value.replay_snapshot.publication_index},
            {"request_id", value.replay_snapshot.request_id},
            {"state", static_cast<int>(value.replay_snapshot.state)},
        };
    }
    return {
        {"command_id", value.command_id},
        {"controller_epoch", value.controller_epoch},
        {"detail", value.detail},
        {"executor_id", value.executor_id},
        {"executor_instance_id", value.executor_instance_id},
        {"has_replay_snapshot", value.has_replay_snapshot},
        {"kind", static_cast<int>(value.kind)},
        {"model_id", value.model_id},
        {"publications", std::move(publications)},
        {"replay_snapshot", std::move(replay)},
        {"request_complete", value.request_complete},
        {"request_id", value.request_id},
        {"schema", "llama-server-warm-tier-result-v2"},
        {"success", value.success},
    };
}

class test_executor final : public server_warm_tier_executor {
public:
    test_executor(
            std::string executor_id,
            server_warm_tier_executor_result_sink result_sink)
        : state(std::make_shared<shared_state>(
            std::move(executor_id),
            std::move(result_sink))) {
    }

    ~test_executor() override {
        std::vector<std::thread> threads;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->stopping = true;
            threads.swap(state->threads);
        }
        for (std::thread & thread : threads) {
            if (thread.get_id() == std::this_thread::get_id()) {
                thread.detach();
            } else {
                thread.join();
            }
        }
    }

    const std::string & id() const override {
        return state->executor_id;
    }

    const std::string & instance_id() const override {
        return state->executor_instance_id;
    }

    bool submit(
            const server_warm_tier_command & value,
            std::string & error) override {
        if (value.executor_id != state->executor_id
                || value.executor_instance_id
                    != state->executor_instance_id) {
            error = "wrong executor";
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->stopping) {
                error = "executor is stopping";
                return false;
            }
        }
        auto shared = state;
        std::thread thread([shared, value]() {
            server_warm_tier_result result;
            result.command_id = value.command_id;
            result.controller_epoch = value.controller_epoch;
            result.kind = value.kind;
            result.model_id = value.model_id;
            result.request_id = value.request_id;
            result.executor_id = value.executor_id;
            result.executor_instance_id = value.executor_instance_id;
            result.success = true;
            if (value.kind == SERVER_WARM_TIER_COMMAND_REPLAY) {
                result.has_replay_snapshot = true;
                result.replay_snapshot = value.request;
            } else if (value.kind == SERVER_WARM_TIER_COMMAND_EXECUTE) {
                server_warm_tier_publication publication;
                publication.token = 100;
                publication.position = value.request.position;
                publication.publication_index =
                    value.request.publication_index;
                publication.ownership_epoch =
                    value.request.ownership_epoch;
                publication.owner_id = value.executor_id;
                result.publications.push_back(std::move(publication));
                result.request_complete = true;
            }
            shared->result_sink(std::move(result));
        });
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->stopping) {
                error = "executor is stopping";
                thread.join();
                return false;
            }
            state->threads.push_back(std::move(thread));
        }
        return true;
    }

private:
    struct shared_state {
        shared_state(
                std::string executor_id,
                server_warm_tier_executor_result_sink result_sink)
            : executor_id(std::move(executor_id)),
              executor_instance_id(this->executor_id + "-instance"),
              result_sink(std::move(result_sink)) {
        }

        std::string executor_id;
        std::string executor_instance_id;
        server_warm_tier_executor_result_sink result_sink;
        std::mutex mutex;
        std::vector<std::thread> threads;
        bool stopping = false;
    };

    std::shared_ptr<shared_state> state;
};

static std::shared_ptr<server_warm_tier_executor> make_test_executor(
        server_warm_tier_executor_result_sink sink) {
    return std::make_shared<test_executor>("GPU", std::move(sink));
}

struct callback_destroy_state {
    std::mutex mutex;
    std::condition_variable condition;
    std::unique_ptr<server_warm_tier_controller> controller;
    bool callback_entered = false;
    bool allow_destroy = false;
    bool callback_finished = false;
};

static void test_controller_last_owner_destroyed_in_result_callback() {
    auto state = std::make_shared<callback_destroy_state>();

    auto executor = make_test_executor([state](server_warm_tier_result) {
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->callback_entered = true;
            state->condition.notify_all();
            require(
                state->condition.wait_for(
                    lock,
                    std::chrono::seconds(10),
                    [&]() { return state->allow_destroy; }),
                "timed out before callback destruction");
        }

        state->controller.reset();

        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->callback_finished = true;
        }
        state->condition.notify_all();
    });
    require(executor != nullptr, "failed to create callback executor");
    std::weak_ptr<server_warm_tier_executor> weak_executor = executor;

    server_warm_tier_options options;
    options.enabled = true;
    options.run_id = "callback-destroy";
    options.runtime_config_sha256 = std::string(64, 'c');
    options.clock = []() { return int64_t{1}; };
    options.history_digest = [](const std::vector<llama_token> &,
                                const std::vector<llama_token> &) {
        return std::string(64, 'a');
    };
    state->controller =
        std::make_unique<server_warm_tier_controller>(std::move(options));
    require(
        state->controller->register_executor(executor),
        "failed to register callback executor");

    std::string error;
    require(
        executor->submit(
            command(23, SERVER_WARM_TIER_COMMAND_EXECUTE, "model"),
            error),
        "failed to submit callback destruction command");
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        require(
            state->condition.wait_for(
                lock,
                std::chrono::seconds(10),
                [&]() { return state->callback_entered; }),
            "result callback did not start");
    }

    executor.reset();
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->allow_destroy = true;
    }
    state->condition.notify_all();
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        require(
            state->condition.wait_for(
                lock,
                std::chrono::seconds(10),
                [&]() { return state->callback_finished; }),
            "controller destruction deadlocked in result callback");
    }
    require(
        weak_executor.expired(),
        "controller did not release its last executor owner");
}

static void test_controller_execute_path() {
    std::mutex mutex;
    std::condition_variable condition;
    bool completed = false;
    bool cleaned = false;
    std::string result_error;
    server_warm_tier_options options;
    options.enabled = true;
    options.run_id = "executor-integration";
    options.runtime_config_sha256 = std::string(64, 'c');
    options.clock = []() { return int64_t{1}; };
    options.history_digest = [](const std::vector<llama_token> &,
                                const std::vector<llama_token> &) {
        return std::string(64, 'a');
    };
    options.event_sink = [&](const server_warm_tier_event & event) {
        if (event.kind == SERVER_WARM_TIER_EVENT_REQUEST_COMPLETED) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                completed = true;
            }
            condition.notify_all();
        }
        if (event.kind == SERVER_WARM_TIER_EVENT_CLEANUP_END
                && event.request_id == "r1"
                && event.success) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                cleaned = true;
            }
            condition.notify_all();
        }
    };
    server_warm_tier_controller controller(std::move(options));

    bool result_rejected = false;
    auto executor = make_test_executor([&](server_warm_tier_result result) {
        if (!controller.handle_executor_result(result)) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                result_rejected = true;
                result_error = controller.last_error();
            }
            condition.notify_all();
        }
    });
    require(executor != nullptr, "failed to create controller executor");
    require(controller.register_executor(executor), "failed to register executor");
    server_warm_tier_executor_policy policy;
    policy.executor_id = "GPU";
    policy.role = SERVER_WARM_TIER_EXECUTOR_GPU;
    policy.order = 0;
    policy.credits = 2;
    require(
        controller.set_executor_policy(std::move(policy)),
        "failed to set executor policy");
    require(controller.set_initial_model_state(
        "model",
        "GPU",
        SERVER_WARM_TIER_MODEL_READY), "failed to set initial model state");
    require(controller.start(), "failed to start controller");
    require(
        controller.enqueue_scheduled_request(
            "r1", "model", {1, 2, 3}, 0, 1),
        "failed to enqueue request");
    {
        std::unique_lock<std::mutex> lock(mutex);
        require(condition.wait_for(
            lock,
            std::chrono::seconds(10),
            [&]() { return (completed && cleaned) || result_rejected; }),
            "timed out waiting for controller completion");
    }
    if (result_rejected) {
        throw std::runtime_error(
            "controller rejected executor result: " + result_error);
    }
    server_warm_tier_request_snapshot request;
    require(controller.get_request("r1", request), "request snapshot missing");
    require(
        request.state == SERVER_WARM_TIER_REQUEST_COMPLETED,
        "request did not complete");
    require(
        request.committed_output_tokens
            == std::vector<llama_token>({100}),
        "committed tokens do not match");
    if (!controller.stop()) {
        throw std::runtime_error(
            "failed to stop controller: " + controller.last_error());
    }
}

static std::filesystem::path unique_test_directory() {
    const auto suffix = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path()
        / ("llama-warm-tier-runtime-" + std::to_string(suffix));
    require(
        std::filesystem::create_directory(path),
        "failed to create runtime test directory");
    return path;
}

#ifndef _WIN32

class test_unix_gateway {
public:
    explicit test_unix_gateway(std::filesystem::path path)
        : path(std::move(path)) {
        descriptor = socket(AF_UNIX, SOCK_STREAM, 0);
        require(descriptor >= 0, "failed to create test Unix socket");
        require(
            fcntl(descriptor, F_SETFD, FD_CLOEXEC) == 0,
            "failed to set test socket close-on-exec");
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        const std::string value = this->path.string();
        require(
            value.size() < sizeof(address.sun_path),
            "test Unix socket path is too long");
        std::memcpy(address.sun_path, value.c_str(), value.size() + 1);
        require(
            bind(
                descriptor,
                reinterpret_cast<const sockaddr *>(&address),
                sizeof(address)) == 0,
            "failed to bind test Unix socket");
        require(listen(descriptor, 16) == 0, "failed to listen on test socket");
        accept_thread = std::thread([this]() { accept_loop(); });
    }

    ~test_unix_gateway() {
        stopping.store(true);
        shutdown(descriptor, SHUT_RDWR);
        close(descriptor);
        if (accept_thread.joinable()) {
            accept_thread.join();
        }
        for (std::thread & thread : handlers) {
            thread.join();
        }
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }

private:
    static std::string read_command(int client) {
        std::string raw;
        char buffer[4096];
        while (true) {
            const ssize_t count = recv(client, buffer, sizeof(buffer), 0);
            if (count > 0) {
                raw.append(buffer, static_cast<size_t>(count));
                if (raw.size() > 4 * 1024 * 1024) {
                    return {};
                }
                continue;
            }
            if (count == 0) {
                return raw;
            }
            if (errno != EINTR) {
                return {};
            }
        }
    }

    static nlohmann::json make_result(const nlohmann::json & command) {
        const int kind = command.at("kind").get<int>();
        nlohmann::json result = {
            {"command_id", command.at("command_id")},
            {"controller_epoch", command.at("controller_epoch")},
            {"detail", "ok"},
            {"executor_id", command.at("executor_id")},
            {"executor_instance_id", command.at("executor_instance_id")},
            {"has_replay_snapshot",
                kind == SERVER_WARM_TIER_COMMAND_REPLAY},
            {"kind", kind},
            {"model_id", command.at("model_id")},
            {"publications", nlohmann::json::array()},
            {"replay_snapshot",
                kind == SERVER_WARM_TIER_COMMAND_REPLAY
                    ? command.at("request")
                    : nlohmann::json(nullptr)},
            {"request_complete", false},
            {"request_id", command.at("request_id")},
            {"schema", "llama-server-warm-tier-result-v2"},
            {"success", true},
        };
        if (kind == SERVER_WARM_TIER_COMMAND_EXECUTE) {
            const auto & request = command.at("request");
            result["request_complete"] = true;
            result["publications"].push_back({
                {"owner_id", command.at("executor_id")},
                {"ownership_epoch", request.at("ownership_epoch")},
                {"position", request.at("position")},
                {"publication_index", request.at("publication_index")},
                {"token", 100},
            });
        }
        return result;
    }

    static void serve_one(int client) {
        const std::string raw = read_command(client);
        if (raw.empty()) {
            close(client);
            return;
        }
        try {
            const nlohmann::json command = nlohmann::json::parse(raw);
            const std::string model_id =
                command.at("model_id").get<std::string>();
            if (model_id == "socket-slow") {
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
            } else if (model_id == "socket-callback") {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            } else if (model_id == "socket-hang") {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                close(client);
                return;
            } else if (model_id == "socket-malformed") {
                const std::string output = "{}\n";
                (void) send(
                    client,
                    output.data(),
                    output.size(),
                    TEST_SOCKET_SEND_FLAGS);
                close(client);
                return;
            } else if (model_id == "socket-oversize") {
                const std::string output(8192, 'x');
                (void) send(
                    client,
                    output.data(),
                    output.size(),
                    TEST_SOCKET_SEND_FLAGS);
                close(client);
                return;
            }
            nlohmann::json result = make_result(command);
            if (model_id == "socket-bad-float") {
                result["command_id"] =
                    static_cast<double>(command.at("command_id").get<uint64_t>());
            } else if (model_id == "socket-bad-negative-replay") {
                result["replay_snapshot"]["prompt_tokens"][0] = -1;
            }
            std::string output = result.dump() + "\n";
            if (model_id == "socket-nul") {
                output.append(1, '\0');
                output += "hidden";
            }
            size_t sent = 0;
            while (sent < output.size()) {
                const ssize_t count = send(
                    client,
                    output.data() + sent,
                    output.size() - sent,
                    TEST_SOCKET_SEND_FLAGS);
                if (count > 0) {
                    sent += static_cast<size_t>(count);
                } else if (count < 0 && errno == EINTR) {
                    continue;
                } else {
                    break;
                }
            }
        } catch (const nlohmann::json::exception &) {
        }
        close(client);
    }

    void accept_loop() {
        while (!stopping.load()) {
            const int client = accept(descriptor, nullptr, nullptr);
            if (client >= 0) {
                (void) fcntl(client, F_SETFD, FD_CLOEXEC);
                handlers.emplace_back([client]() { serve_one(client); });
            } else if (errno != EINTR) {
                return;
            }
        }
    }

    std::filesystem::path path;
    int descriptor = -1;
    std::atomic<bool> stopping{false};
    std::thread accept_thread;
    std::vector<std::thread> handlers;
};

static server_warm_tier_result find_result(
        const result_log & log,
        uint64_t command_id) {
    for (const auto & result : log.results) {
        if (result.command_id == command_id) {
            return result;
        }
    }
    throw std::runtime_error("Unix executor result is missing");
}

static uint64_t current_process_start_time_ticks() {
    std::ifstream source("/proc/self/stat", std::ios::binary);
    std::string raw;
    require(
        source && std::getline(source, raw) && raw.size() <= 64 * 1024,
        "cannot read current process start identity");
    const size_t close = raw.rfind(')');
    require(close != std::string::npos && close + 2 < raw.size(),
            "current process stat is invalid");
    std::istringstream fields(raw.substr(close + 2));
    std::string field;
    for (int number = 3; number <= 22; ++number) {
        require(
            static_cast<bool>(fields >> field),
            "current process stat is truncated");
        if (number == 22) {
            size_t consumed = 0;
            const uint64_t ticks = std::stoull(field, &consumed, 10);
            require(consumed == field.size() && ticks != 0,
                    "current process start identity is invalid");
            return ticks;
        }
    }
    throw std::runtime_error("current process start identity is missing");
}

static std::shared_ptr<server_warm_tier_executor> make_unix_executor(
        const std::filesystem::path & path,
        result_log & log,
        int32_t timeout_ms = 1000,
        size_t output_limit = 4096,
        int64_t expected_peer_pid = -1,
        uint64_t expected_peer_start_time_ticks = 0) {
    server_warm_tier_unix_executor_options options;
    options.executor_id = "GPU";
    options.executor_instance_id = "GPU-instance";
    options.socket_path = path.string();
    options.expected_peer_pid = expected_peer_pid > 0
        ? expected_peer_pid
        : static_cast<int64_t>(getpid());
    options.expected_peer_start_time_ticks =
        expected_peer_start_time_ticks != 0
            ? expected_peer_start_time_ticks
            : current_process_start_time_ticks();
    options.result_sink = [&](server_warm_tier_result result) {
        log.add(std::move(result));
    };
    options.timeout_ms = timeout_ms;
    options.execute_concurrency = 2;
    options.output_limit_bytes = output_limit;
    return server_warm_tier_create_unix_executor(std::move(options));
}

static void test_unix_executor_rejects_rebound_peer() {
    const auto directory = unique_test_directory();
    const auto socket_path = directory / "gateway.sock";
    test_unix_gateway gateway(socket_path);

    result_log pid_log;
    auto wrong_pid = make_unix_executor(
        socket_path,
        pid_log,
        1000,
        4096,
        static_cast<int64_t>(getpid()) + 1,
        current_process_start_time_ticks());
    require(wrong_pid != nullptr, "failed to create wrong-PID executor");
    std::string error;
    require(
        wrong_pid->submit(
            command(70, SERVER_WARM_TIER_COMMAND_EXECUTE, "model"),
            error),
        "failed to submit wrong-PID command");
    pid_log.wait(1);
    const auto pid_result = find_result(pid_log, 70);
    require(!pid_result.success, "wrong Unix peer PID was accepted");
    require(
        pid_result.detail.find("peer PID changed") != std::string::npos,
        "wrong Unix peer PID diagnostic");
    wrong_pid.reset();

    result_log start_log;
    auto wrong_start = make_unix_executor(
        socket_path,
        start_log,
        1000,
        4096,
        static_cast<int64_t>(getpid()),
        current_process_start_time_ticks() + 1);
    require(wrong_start != nullptr, "failed to create wrong-start executor");
    require(
        wrong_start->submit(
            command(71, SERVER_WARM_TIER_COMMAND_EXECUTE, "model"),
            error),
        "failed to submit wrong-start command");
    start_log.wait(1);
    const auto start_result = find_result(start_log, 71);
    require(!start_result.success, "wrong Unix peer start was accepted");
    require(
        start_result.detail.find("peer start identity changed")
            != std::string::npos,
        "wrong Unix peer start diagnostic");
}

static void test_unix_executor_transport() {
    const auto directory = unique_test_directory();
    const auto socket_path = directory / "gateway.sock";
    {
        test_unix_gateway gateway(socket_path);
        result_log log;
        auto executor = make_unix_executor(socket_path, log);
        require(executor != nullptr, "failed to create Unix executor");
        std::string error;
        const auto started = std::chrono::steady_clock::now();
        require(
            executor->submit(
                command(30, SERVER_WARM_TIER_COMMAND_EXECUTE, "socket-slow"),
                error),
            "failed to submit first Unix command");
        require(
            executor->submit(
                command(31, SERVER_WARM_TIER_COMMAND_EXECUTE, "socket-slow"),
                error),
            "failed to submit second Unix command");
        log.wait(2);
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started);
        require(elapsed.count() < 280, "Unix commands did not overlap");
        require(find_result(log, 30).success, "first Unix command failed");
        require(find_result(log, 31).success, "second Unix command failed");

        require(
            executor->submit(
                command(
                    32,
                    SERVER_WARM_TIER_COMMAND_EXECUTE,
                    "socket-malformed"),
                error),
            "failed to submit malformed Unix command");
        log.wait(3);
        require(
            !find_result(log, 32).success,
            "malformed Unix result was accepted");

        require(
            executor->submit(
                command(
                    37,
                    SERVER_WARM_TIER_COMMAND_REPLAY,
                    "socket-replay"),
                error),
            "failed to submit Unix replay command");
        require(
            executor->submit(
                command(
                    38,
                    SERVER_WARM_TIER_COMMAND_EXECUTE,
                    "socket-bad-float"),
                error),
            "failed to submit non-integral Unix result");
        log.wait(5);
        const auto replay = find_result(log, 37);
        require(replay.success, "valid Unix replay failed");
        require(replay.has_replay_snapshot, "Unix replay snapshot is missing");
        require(
            replay.replay_snapshot.prompt_tokens
                == std::vector<llama_token>({1, 2, 3}),
            "Unix replay snapshot changed");
        require(
            !find_result(log, 38).success,
            "non-integral Unix command id was accepted");

        require(
            executor->submit(
                command(
                    39,
                    SERVER_WARM_TIER_COMMAND_REPLAY,
                    "socket-bad-negative-replay"),
                error),
            "failed to submit invalid Unix replay result");
        require(
            executor->submit(
                command(
                    40,
                    SERVER_WARM_TIER_COMMAND_EXECUTE,
                    "socket-nul"),
                error),
            "failed to submit Unix result with trailing bytes");
        log.wait(7);
        require(
            !find_result(log, 39).success,
            "negative Unix replay token was accepted");
        require(
            !find_result(log, 40).success,
            "Unix result with trailing bytes was accepted");
    }
    {
        test_unix_gateway gateway(socket_path);
        result_log log;
        auto executor = make_unix_executor(socket_path, log);
        std::string error;
        require(
            executor->submit(
                command(
                    41,
                    SERVER_WARM_TIER_COMMAND_EXECUTE,
                    "socket-slow"),
                error),
            "failed to submit Unix execute before lifecycle");
        require(
            executor->submit(
                command(
                    42,
                    SERVER_WARM_TIER_COMMAND_LOAD,
                    "socket-lifecycle"),
                error),
            "failed to submit Unix lifecycle command");
        require(
            executor->submit(
                command(
                    43,
                    SERVER_WARM_TIER_COMMAND_EXECUTE,
                    "socket-after-lifecycle"),
                error),
            "failed to submit Unix execute after lifecycle");
        log.wait(3);
        require(
            log.results[0].command_id == 41
                && log.results[1].command_id == 42
                && log.results[2].command_id == 43,
            "Unix lifecycle command did not form an execution barrier");
    }
    {
        test_unix_gateway gateway(socket_path);
        result_log log;
        auto executor = make_unix_executor(socket_path, log, 1000, 1024);
        std::string error;
        require(
            executor->submit(
                command(
                    33,
                    SERVER_WARM_TIER_COMMAND_EXECUTE,
                    "socket-oversize"),
                error),
            "failed to submit oversized Unix command");
        log.wait(1);
        require(
            find_result(log, 33).detail.find("exceeded limit")
                != std::string::npos,
            "oversized Unix result was not rejected");
    }
    {
        result_log log;
        auto executor = make_unix_executor(socket_path, log);
        std::string error;
        require(
            executor->submit(
                command(34, SERVER_WARM_TIER_COMMAND_EXECUTE, "model"),
                error),
            "failed to submit unavailable Unix command");
        log.wait(1);
        require(
            !find_result(log, 34).success,
            "unavailable Unix gateway was accepted");
    }
    {
        test_unix_gateway gateway(socket_path);
        result_log log;
        auto executor = make_unix_executor(socket_path, log, 50);
        std::string error;
        require(
            executor->submit(
                command(
                    35,
                    SERVER_WARM_TIER_COMMAND_EXECUTE,
                    "socket-hang"),
                error),
            "failed to submit timed Unix command");
        log.wait(1);
        require(
            find_result(log, 35).detail.find("timed out")
                != std::string::npos,
            "Unix timeout was not enforced");
    }
    {
        test_unix_gateway gateway(socket_path);
        result_log log;
        auto executor = make_unix_executor(socket_path, log, 5000);
        std::string error;
        require(
            executor->submit(
                command(
                    36,
                    SERVER_WARM_TIER_COMMAND_EXECUTE,
                    "socket-hang"),
                error),
            "failed to submit cancellable Unix command");
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        const auto started = std::chrono::steady_clock::now();
        executor.reset();
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started);
        require(
            elapsed.count() < 500,
            "Unix executor destruction did not cancel active I/O");
    }

    server_warm_tier_unix_executor_options invalid;
    invalid.executor_id = "GPU";
    invalid.socket_path = "relative.sock";
    invalid.result_sink = [](server_warm_tier_result) {};
    require(
        server_warm_tier_create_unix_executor(std::move(invalid)) == nullptr,
        "relative Unix socket path was accepted");
    std::filesystem::remove_all(directory);
}

struct unix_callback_destroy_state {
    std::mutex mutex;
    std::condition_variable condition;
    std::shared_ptr<server_warm_tier_executor> executor;
    bool callback_finished = false;
};

static void test_unix_executor_destroyed_in_result_callback() {
    const auto directory = unique_test_directory();
    const auto socket_path = directory / "callback.sock";
    test_unix_gateway gateway(socket_path);
    auto state = std::make_shared<unix_callback_destroy_state>();

    server_warm_tier_unix_executor_options options;
    options.executor_id = "GPU";
    options.executor_instance_id = "GPU-instance";
    options.socket_path = socket_path.string();
    options.expected_peer_pid = static_cast<int64_t>(getpid());
    options.expected_peer_start_time_ticks =
        current_process_start_time_ticks();
    options.timeout_ms = 5000;
    options.execute_concurrency = 2;
    options.result_sink = [state](server_warm_tier_result result) {
        if (result.command_id != 44) {
            return;
        }
        state->executor.reset();
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->callback_finished = true;
        }
        state->condition.notify_all();
    };
    state->executor =
        server_warm_tier_create_unix_executor(std::move(options));
    require(state->executor != nullptr, "failed to create callback Unix executor");
    std::weak_ptr<server_warm_tier_executor> weak_executor = state->executor;

    std::string error;
    require(
        state->executor->submit(
            command(
                45,
                SERVER_WARM_TIER_COMMAND_EXECUTE,
                "socket-hang"),
            error),
        "failed to submit blocked callback Unix command");
    require(
        state->executor->submit(
            command(
                44,
                SERVER_WARM_TIER_COMMAND_EXECUTE,
                "socket-callback"),
            error),
        "failed to submit callback Unix command");

    {
        std::unique_lock<std::mutex> lock(state->mutex);
        require(
            state->condition.wait_for(
                lock,
                std::chrono::seconds(2),
                [&]() { return state->callback_finished; }),
            "Unix executor destruction deadlocked in result callback");
    }
    require(
        weak_executor.expired(),
        "Unix callback did not release the last executor owner");
    std::filesystem::remove_all(directory);
}

#endif

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
static std::string sha256_string(const std::string & value) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    require(
        SHA256(
            reinterpret_cast<const unsigned char *>(value.data()),
            value.size(),
            digest) != nullptr,
        "test SHA-256 failed");
    static const char hex[] = "0123456789abcdef";
    std::string result;
    for (unsigned char byte : digest) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 0x0f]);
    }
    return result;
}
#endif

static void test_runtime_factory() {
    unsetenv("LLAMA_SERVER_WARM_TIER_CONFIG");
    unsetenv("LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE");
    require(
        server_warm_tier_create_runtime_from_env() == nullptr,
        "runtime was enabled without the environment variable");

    const auto directory = unique_test_directory();
    const auto config_path = directory / "config.json";
    const auto event_path = directory / "events.jsonl";
    const auto socket_path = directory / "runtime.sock";
#if defined(CPPHTTPLIB_OPENSSL_SUPPORT) && !defined(_WIN32)
    bool missing_token_refused = false;
    try {
        (void) server_warm_tier_internal_token_from_env();
    } catch (const std::exception &) {
        missing_token_refused = true;
    }
    require(
        missing_token_refused,
        "missing internal capability was accepted");

    const auto token_path = directory / "internal-token";
    const std::string token(64, 'a');
    {
        std::ofstream output(token_path, std::ios::binary);
        require(bool(output), "failed to create internal capability");
        output << token << '\n';
        require(bool(output), "failed to write internal capability");
    }
    require(
        chmod(token_path.c_str(), S_IRUSR | S_IWUSR) == 0,
        "failed to protect internal capability");
    require(
        setenv(
            "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE",
            token_path.c_str(),
            1) == 0,
        "failed to set internal capability environment");
    require(
        server_warm_tier_internal_token_from_env() == token,
        "internal capability bytes changed");

    require(
        chmod(token_path.c_str(), S_IRUSR | S_IWUSR | S_IROTH) == 0,
        "failed to weaken internal capability mode");
    bool unsafe_mode_refused = false;
    try {
        (void) server_warm_tier_internal_token_from_env();
    } catch (const std::exception &) {
        unsafe_mode_refused = true;
    }
    require(unsafe_mode_refused, "public internal capability was accepted");
    require(
        chmod(token_path.c_str(), S_IRUSR | S_IWUSR) == 0,
        "failed to restore internal capability mode");

    const auto token_link = directory / "internal-token-link";
    std::filesystem::create_hard_link(token_path, token_link);
    bool linked_token_refused = false;
    try {
        (void) server_warm_tier_internal_token_from_env();
    } catch (const std::exception &) {
        linked_token_refused = true;
    }
    require(linked_token_refused, "linked internal capability was accepted");
    std::filesystem::remove(token_link);

    const auto token_symlink = directory / "internal-token-symlink";
    std::filesystem::create_symlink(token_path, token_symlink);
    require(
        setenv(
            "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE",
            token_symlink.c_str(),
            1) == 0,
        "failed to set linked capability environment");
    bool symlink_refused = false;
    try {
        (void) server_warm_tier_internal_token_from_env();
    } catch (const std::exception &) {
        symlink_refused = true;
    }
    require(symlink_refused, "symlinked internal capability was accepted");
    require(
        setenv(
            "LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE",
            token_path.c_str(),
            1) == 0,
        "failed to restore internal capability environment");
#endif
    int64_t expected_peer_pid = 1;
    uint64_t expected_peer_start_time_ticks = 1;
#ifndef _WIN32
    auto gateway = std::make_unique<test_unix_gateway>(socket_path);
    expected_peer_pid = static_cast<int64_t>(getpid());
    expected_peer_start_time_ticks = current_process_start_time_ticks();
#endif
    nlohmann::json config = {
        {"c3_profile_lock_sha256", nullptr},
        {"configuration", "C1_GPU_ONLY_OPTIMIZED"},
        {"evidence_root_sha256", std::string(64, 'd')},
        {"event_log_path", event_path.string()},
        {"executors", nlohmann::json::array({
            {
                {"credits", 2},
                {"execute_concurrency", 2},
                {"executor_id", "GPU"},
                {"executor_instance_id", "GPU-instance"},
                {"expected_peer_pid", expected_peer_pid},
                {"expected_peer_start_time_ticks",
                    expected_peer_start_time_ticks},
                {"order", 0},
                {"output_limit_bytes", 1048576},
                {"queue_capacity", 8},
                {"role", "GPU"},
                {"socket_path", socket_path.string()},
                {"timeout_ms", 5000},
                {"transport", "UNIX_SOCKET"},
            },
        })},
        {"initial_models", nlohmann::json::array({
            {
                {"executor_id", "GPU"},
                {"model_id", "model-a"},
                {"state", "READY"},
            },
            {
                {"executor_id", "GPU"},
                {"model_id", "model-b"},
                {"state", "ABSENT"},
            },
        })},
        {"promotion_enabled", true},
        {"run_id", "runtime-test"},
        {"runtime_plan_sha256", std::string(64, 'e')},
        {"schema", "llama-server-warm-tier-runtime-v4"},
    };
    {
        std::ofstream output(config_path, std::ios::binary);
        require(bool(output), "failed to create runtime config");
        output << config.dump() << '\n';
        require(bool(output), "failed to write runtime config");
    }
    require(
        setenv(
            "LLAMA_SERVER_WARM_TIER_CONFIG",
            config_path.c_str(),
            1) == 0,
        "failed to set runtime environment");
#if defined(CPPHTTPLIB_OPENSSL_SUPPORT) && !defined(_WIN32)
    const std::string config_raw = config.dump() + "\n";
    auto controller = server_warm_tier_create_runtime_from_env();
    require(controller != nullptr, "valid runtime config was refused");
    require(
        controller->activation_state() == SERVER_WARM_TIER_ACTIVATION_WAITING,
        "runtime did not wait for explicit activation");
    require(controller->activate(), "runtime activation was refused");
    const auto activation_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (controller->activation_state() == SERVER_WARM_TIER_ACTIVATION_PREPARING
            && std::chrono::steady_clock::now() < activation_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    require(
        controller->activation_state() == SERVER_WARM_TIER_ACTIVATION_READY,
        "runtime activation did not complete");
    require(controller->stop(), "runtime controller did not stop");
    controller.reset();

    std::ifstream events(event_path, std::ios::binary);
    const std::string raw{
        std::istreambuf_iterator<char>(events),
        std::istreambuf_iterator<char>()};
    size_t rows = 0;
    for (char character : raw) {
        rows += character == '\n';
    }
    require(rows == 8, "runtime event log has the wrong row count");
    {
        const size_t newline = raw.find('\n');
        require(newline != std::string::npos, "runtime event log is truncated");
        const auto event = nlohmann::json::parse(raw.substr(0, newline));
        require(
            event.at("runtime_config_sha256").get<std::string>()
                == sha256_string(config_raw),
            "runtime event did not bind exact config bytes");
    }

    const auto duplicate_path = directory / "duplicate.json";
    const auto duplicate_event = directory / "duplicate-events.jsonl";
    nlohmann::json duplicate_config = config;
    duplicate_config["event_log_path"] = duplicate_event.string();
    std::string duplicate = duplicate_config.dump();
    const std::string marker = "\"credits\":2";
    const size_t marker_pos = duplicate.find(marker);
    require(
        marker_pos != std::string::npos,
        "runtime nested duplicate marker changed");
    duplicate.replace(
        marker_pos,
        marker.size(),
        "\"credits\":2,\"credits\":2");
    {
        std::ofstream output(duplicate_path, std::ios::binary);
        output << duplicate << '\n';
        require(bool(output), "failed to write duplicate-key config");
    }
    require(
        setenv(
            "LLAMA_SERVER_WARM_TIER_CONFIG",
            duplicate_path.c_str(),
            1) == 0,
        "failed to set duplicate config");
    require(
        server_warm_tier_create_runtime_from_env() == nullptr,
        "duplicate-key runtime config was accepted");
    require(
        !std::filesystem::exists(duplicate_event),
        "duplicate-key config created evidence");

    const auto invalid_config = [&directory, &config](
            const char * name,
            const std::function<void(nlohmann::json &)> & mutate) {
        nlohmann::json candidate = config;
        const auto path = directory / (std::string(name) + ".json");
        const auto log_path = directory / (std::string(name) + "-events.jsonl");
        candidate["event_log_path"] = log_path.string();
        mutate(candidate);
        {
            std::ofstream output(path, std::ios::binary);
            output << candidate.dump() << '\n';
            require(bool(output), "failed to write invalid runtime config");
        }
        require(
            setenv("LLAMA_SERVER_WARM_TIER_CONFIG", path.c_str(), 1) == 0,
            "failed to set invalid runtime config");
        require(
            server_warm_tier_create_runtime_from_env() == nullptr,
            "invalid runtime config was accepted");
        require(
            !std::filesystem::exists(log_path),
            "invalid runtime config created evidence");
    };
    invalid_config("legacy-schema", [](nlohmann::json & candidate) {
        candidate["schema"] = "llama-server-warm-tier-runtime-v2";
    });
    invalid_config("process-transport", [](nlohmann::json & candidate) {
        candidate["executors"][0]["transport"] = "PROCESS";
    });
    invalid_config("relative-socket", [](nlohmann::json & candidate) {
        candidate["executors"][0]["socket_path"] = "gateway.sock";
    });
    invalid_config("unknown-configuration", [](nlohmann::json & candidate) {
        candidate["configuration"] = "UNKNOWN";
    });
    invalid_config("bad-plan-digest", [](nlohmann::json & candidate) {
        candidate["runtime_plan_sha256"] = std::string(64, 'A');
    });
    invalid_config("bad-evidence-digest", [](nlohmann::json & candidate) {
        candidate["evidence_root_sha256"] = std::string(63, 'f');
    });
    invalid_config("non-c3-profile-lock", [](nlohmann::json & candidate) {
        candidate["c3_profile_lock_sha256"] = std::string(64, 'f');
    });
    invalid_config("c3-without-profile-lock", [](nlohmann::json & candidate) {
        candidate["configuration"] = "C3_DUAL_PARTIAL_OFFLOAD";
    });
    invalid_config("c1-with-cpu", [
            &socket_path,
            expected_peer_pid,
            expected_peer_start_time_ticks](nlohmann::json & candidate) {
        candidate["executors"].push_back({
            {"credits", 1},
            {"execute_concurrency", 1},
            {"executor_id", "CPU"},
            {"executor_instance_id", "CPU-instance"},
            {"expected_peer_pid", expected_peer_pid},
            {"expected_peer_start_time_ticks",
                expected_peer_start_time_ticks},
            {"order", 1},
            {"output_limit_bytes", 1048576},
            {"queue_capacity", 8},
            {"role", "CPU"},
            {"socket_path", socket_path.string()},
            {"timeout_ms", 5000},
            {"transport", "UNIX_SOCKET"},
        });
        candidate["initial_models"].push_back({
            {"executor_id", "CPU"},
            {"model_id", "model-a"},
            {"state", "ABSENT"},
        });
        candidate["initial_models"].push_back({
            {"executor_id", "CPU"},
            {"model_id", "model-b"},
            {"state", "READY"},
        });
    });
    invalid_config("c1-without-promotion", [](nlohmann::json & candidate) {
        candidate["promotion_enabled"] = false;
    });
    invalid_config("incomplete-model-matrix", [](nlohmann::json & candidate) {
        candidate["initial_models"].erase(candidate["initial_models"].begin());
    });
    invalid_config("two-gpu-ready", [](nlohmann::json & candidate) {
        candidate["initial_models"][1]["state"] = "READY";
    });
    invalid_config("executor-order-gap", [](nlohmann::json & candidate) {
        candidate["executors"][0]["order"] = 1;
    });
    invalid_config("queue-below-credits", [](nlohmann::json & candidate) {
        candidate["executors"][0]["queue_capacity"] = 1;
    });
    invalid_config("missing-peer-pid", [](nlohmann::json & candidate) {
        candidate["executors"][0]["expected_peer_pid"] = 0;
    });
    invalid_config("missing-peer-start", [](nlohmann::json & candidate) {
        candidate["executors"][0]["expected_peer_start_time_ticks"] = 0;
    });
    invalid_config("negative-peer-start", [](nlohmann::json & candidate) {
        candidate["executors"][0]["expected_peer_start_time_ticks"] = -1;
    });
    invalid_config("warm-credits-exceed-gpu", [
            &socket_path,
            expected_peer_pid,
            expected_peer_start_time_ticks](nlohmann::json & candidate) {
        candidate["configuration"] = "C2_GPU_PLUS_CPU_WARM_EXECUTOR";
        candidate["executors"].push_back({
            {"credits", 3},
            {"execute_concurrency", 3},
            {"executor_id", "CPU"},
            {"executor_instance_id", "CPU-instance"},
            {"expected_peer_pid", expected_peer_pid},
            {"expected_peer_start_time_ticks",
                expected_peer_start_time_ticks},
            {"order", 1},
            {"output_limit_bytes", 1048576},
            {"queue_capacity", 8},
            {"role", "CPU"},
            {"socket_path", socket_path.string()},
            {"timeout_ms", 5000},
            {"transport", "UNIX_SOCKET"},
        });
        candidate["initial_models"].push_back({
            {"executor_id", "CPU"},
            {"model_id", "model-a"},
            {"state", "ABSENT"},
        });
        candidate["initial_models"].push_back({
            {"executor_id", "CPU"},
            {"model_id", "model-b"},
            {"state", "READY"},
        });
    });

    const auto maximum_uint_path = directory / "maximum-uint.json";
    const auto maximum_uint_event =
        directory / "maximum-uint-events.jsonl";
    nlohmann::json maximum_uint_config = config;
    maximum_uint_config["event_log_path"] =
        maximum_uint_event.string();
    maximum_uint_config["executors"][0][
        "expected_peer_start_time_ticks"] =
            std::numeric_limits<uint64_t>::max();
    {
        std::ofstream output(maximum_uint_path, std::ios::binary);
        output << maximum_uint_config.dump() << '\n';
        require(bool(output), "failed to write maximum uint config");
    }
    require(
        setenv(
            "LLAMA_SERVER_WARM_TIER_CONFIG",
            maximum_uint_path.c_str(),
            1) == 0,
        "failed to set maximum uint config");
    auto maximum_uint_controller =
        server_warm_tier_create_runtime_from_env();
    require(maximum_uint_controller != nullptr,
            "maximum uint runtime config was refused");
    require(maximum_uint_controller->stop(),
            "maximum uint runtime controller did not stop");
    maximum_uint_controller.reset();

    const auto c3_path = directory / "valid-c3.json";
    const auto c3_event = directory / "valid-c3-events.jsonl";
    nlohmann::json c3_config = config;
    c3_config["configuration"] = "C3_DUAL_PARTIAL_OFFLOAD";
    c3_config["c3_profile_lock_sha256"] = std::string(64, 'f');
    c3_config["promotion_enabled"] = false;
    c3_config["initial_models"][1]["state"] = "READY";
    c3_config["event_log_path"] = c3_event.string();
    {
        std::ofstream output(c3_path, std::ios::binary);
        output << c3_config.dump() << '\n';
        require(bool(output), "failed to write valid C3 runtime config");
    }
    require(
        setenv("LLAMA_SERVER_WARM_TIER_CONFIG", c3_path.c_str(), 1) == 0,
        "failed to set valid C3 runtime config");
    auto c3_controller = server_warm_tier_create_runtime_from_env();
    require(c3_controller != nullptr, "valid C3 runtime config was refused");
    require(c3_controller->stop(), "C3 runtime controller did not stop");
    c3_controller.reset();

    require(
        setenv(
            "LLAMA_SERVER_WARM_TIER_CONFIG",
            config_path.c_str(),
            1) == 0,
        "failed to reset runtime environment");
    require(
        server_warm_tier_create_runtime_from_env() == nullptr,
        "runtime overwrote an existing event log");
#else
    require(
        server_warm_tier_create_runtime_from_env() == nullptr,
        "runtime started without a supported secure transport");
    require(
        !std::filesystem::exists(event_path),
        "unsupported runtime created evidence");
#endif
    unsetenv("LLAMA_SERVER_WARM_TIER_CONFIG");
    unsetenv("LLAMA_SERVER_WARM_TIER_INTERNAL_TOKEN_FILE");
    std::filesystem::remove_all(directory);
}

static void test_history_sha256_vector() {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    require(
        server_warm_tier_history_sha256({1, 2, 3}, {100, 101})
            == "b2c2611b132cd66f6ecc086dbea88824e5f1435f5aa2a8f026cf2a65e3285694",
        "history SHA-256 encoding changed");
#else
    bool refused = false;
    try {
        (void) server_warm_tier_history_sha256({1, 2, 3}, {100, 101});
    } catch (const std::runtime_error &) {
        refused = true;
    }
    require(refused, "history SHA-256 ran without OpenSSL");
#endif
}

#ifndef _WIN32

static int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static size_t parse_benchmark_size(
        const char * value,
        const char * name) {
    if (value == nullptr || value[0] == '\0') {
        throw std::runtime_error(std::string("missing ") + name);
    }
    size_t result = 0;
    for (const unsigned char character : std::string(value)) {
        if (character < '0' || character > '9') {
            throw std::runtime_error(std::string("invalid ") + name);
        }
        const size_t digit = static_cast<size_t>(character - '0');
        if (result > (std::numeric_limits<size_t>::max() - digit) / 10) {
            throw std::runtime_error(std::string("oversized ") + name);
        }
        result = result * 10 + digit;
    }
    return result;
}

struct benchmark_completion {
    int64_t started_ns = 0;
    int64_t completed_ns = 0;
    server_warm_tier_result result;
};

static int run_unix_benchmark(
        const std::string & socket_path,
        size_t sample_count,
        size_t fanout,
        int64_t expected_peer_pid,
        uint64_t expected_peer_start_time_ticks) {
    if (sample_count < 50
            || sample_count > 100000
            || (fanout != 1 && fanout != 8)
            || sample_count % fanout != 0) {
        throw std::runtime_error("invalid Unix benchmark dimensions");
    }

    std::mutex mutex;
    std::condition_variable condition;
    std::map<uint64_t, benchmark_completion> completions;
    server_warm_tier_unix_executor_options options;
    options.executor_id = "noop";
    options.executor_instance_id = "noop-instance";
    options.socket_path = socket_path;
    options.expected_peer_pid = expected_peer_pid;
    options.expected_peer_start_time_ticks =
        expected_peer_start_time_ticks;
    options.timeout_ms = 10000;
    options.execute_concurrency = fanout;
    options.queue_capacity = sample_count;
    options.output_limit_bytes = 4 * 1024 * 1024;
    options.result_sink = [&](server_warm_tier_result result) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto found = completions.find(result.command_id);
            if (found == completions.end()
                    || found->second.completed_ns != 0) {
                return;
            }
            found->second.completed_ns = steady_ns();
            found->second.result = std::move(result);
        }
        condition.notify_all();
    };
    auto executor = server_warm_tier_create_unix_executor(std::move(options));
    if (executor == nullptr) {
        throw std::runtime_error("cannot create Unix benchmark executor");
    }

    nlohmann::json rows = nlohmann::json::array();
    uint64_t command_id = 1;
    for (size_t batch = 0; batch < sample_count / fanout; ++batch) {
        std::vector<server_warm_tier_command> commands;
        commands.reserve(fanout);
        for (size_t item = 0; item < fanout; ++item) {
            auto value = command(
                command_id++,
                SERVER_WARM_TIER_COMMAND_EXECUTE,
                "model");
            value.total_output_tokens = 8;
            value.executor_id = "noop";
            value.executor_instance_id = "noop-instance";
            value.request_id = "request-" + std::to_string(value.command_id);
            value.request.request_id = value.request_id;
            value.request.prompt_tokens.clear();
            for (llama_token token = 0; token < 128; ++token) {
                value.request.prompt_tokens.push_back(token);
            }
            value.request.position = 128;
            value.request.owner_id = "noop";
            commands.push_back(std::move(value));
        }

        const int64_t batch_started_ns = steady_ns();
        std::string error;
        for (auto & value : commands) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                benchmark_completion completion;
                completion.started_ns = steady_ns();
                completions.emplace(value.command_id, std::move(completion));
            }
            if (!executor->submit(value, error)) {
                throw std::runtime_error(
                    "Unix benchmark submit failed: " + error);
            }
        }
        {
            std::unique_lock<std::mutex> lock(mutex);
            const uint64_t last_id = commands.back().command_id;
            if (!condition.wait_for(
                    lock,
                    std::chrono::seconds(30),
                    [&]() {
                        for (const auto & value : commands) {
                            const auto found =
                                completions.find(value.command_id);
                            if (found == completions.end()
                                    || found->second.completed_ns == 0) {
                                return false;
                            }
                        }
                        return completions.find(last_id)
                            != completions.end();
                    })) {
                throw std::runtime_error(
                    "Unix benchmark result timed out");
            }
        }

        int64_t batch_completed_ns = batch_started_ns;
        for (const auto & value : commands) {
            const benchmark_completion & completion =
                completions.at(value.command_id);
            const server_warm_tier_result & result = completion.result;
            if (!result.success
                    || result.command_id != value.command_id
                    || result.kind != SERVER_WARM_TIER_COMMAND_EXECUTE
                    || result.executor_id != value.executor_id
                    || result.executor_instance_id
                        != value.executor_instance_id
                    || result.model_id != value.model_id
                    || result.request_id != value.request_id
                    || result.publications.size() != 1
                    || result.has_replay_snapshot
                    || result.request_complete
                    || result.publications[0].owner_id != value.executor_id
                    || result.publications[0].ownership_epoch
                        != value.request.ownership_epoch
                    || result.publications[0].position
                        != value.request.position
                    || result.publications[0].publication_index
                        != value.request.publication_index) {
                throw std::runtime_error(
                    "Unix benchmark result did not match command");
            }
            batch_completed_ns =
                std::max(batch_completed_ns, completion.completed_ns);
        }

        for (size_t item = 0; item < commands.size(); ++item) {
            const auto & value = commands[item];
            const benchmark_completion & completion =
                completions.at(value.command_id);
            const std::string command_raw = command_json(value).dump() + "\n";
            const std::string result_raw =
                result_json(completion.result).dump() + "\n";
            rows.push_back({
                {"batch_index", batch},
                {"batch_makespan_ns",
                    batch_completed_ns - batch_started_ns},
                {"command_bytes", command_raw.size()},
                {"command_id", value.command_id},
                {"completed_ns", completion.completed_ns},
                {"item_index", item},
                {"latency_ns",
                    completion.completed_ns - completion.started_ns},
                {"result_bytes", result_raw.size()},
                {"started_ns", completion.started_ns},
            });
        }
    }

    const nlohmann::json output = {
        {"fanout", fanout},
        {"rows", std::move(rows)},
        {"sample_count", sample_count},
        {"schema", "s40-native-unix-executor-bench-v1"},
        {"socket_path", socket_path},
        {"transport", "UNIX_SOCKET"},
    };
    std::cout << output.dump() << '\n';
    return EXIT_SUCCESS;
}

#endif

int main(int argc, char ** argv) {
    try {
#ifndef _WIN32
        if (argc == 7 && std::string(argv[1]) == "--unix-bench") {
            return run_unix_benchmark(
                argv[2],
                parse_benchmark_size(argv[3], "sample count"),
                parse_benchmark_size(argv[4], "fanout"),
                static_cast<int64_t>(
                    parse_benchmark_size(argv[5], "peer PID")),
                parse_benchmark_size(argv[6], "peer start time"));
        }
#endif
        require(argc == 1, "invalid test arguments");
        test_controller_last_owner_destroyed_in_result_callback();
        test_controller_execute_path();
#ifndef _WIN32
        test_unix_executor_transport();
        test_unix_executor_rejects_rebound_peer();
        test_unix_executor_destroyed_in_result_callback();
#endif
        test_runtime_factory();
        test_history_sha256_vector();
        test_child_environment_drops_warm_tier_capability();
        std::puts("test-warm-tier-executors: PASS");
        return EXIT_SUCCESS;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "test-warm-tier-executors: FAIL: %s\n", exc.what());
        return EXIT_FAILURE;
    }
}
