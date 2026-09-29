#include "ffn-split-session-manifest.h"

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
#include <map>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using steady_clock = std::chrono::steady_clock;

struct config {
    std::string manifest;
    std::string model;
    std::string worker;
    int64_t max_tokens = 1;
    int64_t column_quantum = 32;
    long max_requests = 0;
    int control_port = 0;
    bool f16_io = false;
};

struct worker_state {
    ffn_split::resident_session_shard shard;
    pid_t pid = -1;
    long load_count = 0;
    int64_t column_quantum = 0;
    int64_t max_tokens = 0;
};

volatile sig_atomic_t stop_requested = 0;

void handle_stop(int) {
    stop_requested = 1;
}

uint64_t monotonic_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            steady_clock::now().time_since_epoch()).count());
}

uint64_t epoch_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

void print_phase(
        const char * phase,
        const ffn_split::resident_session_shard & shard) {
    fprintf(stderr,
            "RESIDENTPHASE {\"schema\":\"s42-phone-residency-phase-v1\"," 
            "\"component\":\"resident-manager\"," 
            "\"phase\":\"%s\",\"session_id\":\"%s\"," 
            "\"artifact_sha256\":\"%s\",\"session_generation\":%llu," 
            "\"monotonic_us\":%llu,\"epoch_us\":%llu}\n",
            phase, shard.session_id.c_str(), shard.artifact_sha256.c_str(),
            static_cast<unsigned long long>(shard.session_generation),
            static_cast<unsigned long long>(monotonic_us()),
            static_cast<unsigned long long>(epoch_us()));
    fflush(stderr);
}

bool parse_number(const char * text, int64_t minimum, int64_t maximum, int64_t & value) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const long long parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed < minimum || parsed > maximum) {
        return false;
    }
    value = parsed;
    return true;
}

bool parse_config(int argc, char ** argv, config & result) {
    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--manifest") == 0 && index + 1 < argc) {
            result.manifest = argv[++index];
        } else if (strcmp(argv[index], "--worker") == 0 && index + 1 < argc) {
            result.worker = argv[++index];
        } else if (strcmp(argv[index], "--model") == 0 && index + 1 < argc) {
            result.model = argv[++index];
        } else if (strcmp(argv[index], "--max-tokens") == 0 && index + 1 < argc) {
            if (!parse_number(argv[++index], 1, UINT16_MAX, result.max_tokens)) {
                return false;
            }
        } else if (strcmp(argv[index], "--column-quantum") == 0 && index + 1 < argc) {
            if (!parse_number(argv[++index], 1, INT_MAX, result.column_quantum)) {
                return false;
            }
        } else if (strcmp(argv[index], "--max-requests") == 0 && index + 1 < argc) {
            int64_t value = 0;
            if (!parse_number(argv[++index], 0, LONG_MAX, value)) {
                return false;
            }
            result.max_requests = static_cast<long>(value);
        } else if (strcmp(argv[index], "--control-port") == 0 && index + 1 < argc) {
            int64_t value = 0;
            if (!parse_number(argv[++index], 0, 65535, value)) {
                return false;
            }
            result.control_port = static_cast<int>(value);
        } else if (strcmp(argv[index], "--f16-io") == 0) {
            result.f16_io = true;
        } else {
            return false;
        }
    }
    if (result.worker.empty()) {
        char executable[PATH_MAX + 1] = {};
        const ssize_t count = readlink(
                "/proc/self/exe", executable, sizeof(executable) - 1);
        if (count <= 0 || count >= static_cast<ssize_t>(sizeof(executable))) {
            return false;
        }
        executable[count] = '\0';
        char * separator = strrchr(executable, '/');
        if (separator == nullptr) {
            return false;
        }
        *separator = '\0';
        result.worker = std::string(executable) + "/llama-ffn-split-worker";
    }
    if (result.control_port == 0) {
        const char * diagnostic_port = getenv("S42_DIAGNOSTIC_PORT");
        int64_t value = 0;
        if (diagnostic_port != nullptr &&
            parse_number(diagnostic_port, 1, 65534, value)) {
            result.control_port = static_cast<int>(value + 1);
        }
    }
    return !result.manifest.empty() && result.worker[0] == '/' &&
            access(result.worker.c_str(), X_OK) == 0;
}

bool same_shard(
        const ffn_split::resident_session_shard & left,
        const ffn_split::resident_session_shard & right) {
    return left.session_id == right.session_id && left.backend == right.backend &&
            left.artifact_sha256 == right.artifact_sha256 &&
            left.model == right.model && left.layers == right.layers &&
            left.layer_mask == right.layer_mask && left.columns == right.columns &&
            left.port == right.port && left.endpoint_sha256 == right.endpoint_sha256 &&
            left.resident_bytes == right.resident_bytes &&
            left.resident_geometry_sha256 == right.resident_geometry_sha256 &&
            left.operator_plan_sha256 == right.operator_plan_sha256 &&
            left.session_generation == right.session_generation;
}

bool port_ready(int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    const bool ready = fd >= 0 &&
            connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0;
    if (fd >= 0) {
        close(fd);
    }
    return ready;
}

pid_t launch_worker(
        const config & cfg, const ffn_split::resident_session_shard & shard,
        int64_t column_quantum, int64_t max_tokens) {
    const pid_t pid = fork();
    if (pid != 0) {
        return pid;
    }
    const std::string generation = std::to_string(shard.session_generation);
    setenv("S42_RESIDENCY_SESSION_ID", shard.session_id.c_str(), 1);
    setenv("S42_RESIDENCY_SESSION_GENERATION", generation.c_str(), 1);
    std::vector<std::string> arguments = {
        cfg.worker,
        "-m", shard.model.empty() ? cfg.model : shard.model,
        "--artifact-sha256", shard.artifact_sha256,
        "--layers", shard.layers,
        "--columns", std::to_string(shard.columns),
        "--backend", shard.backend,
        "--bind", "127.0.0.1",
        "--port", std::to_string(shard.port),
        "--max-tokens", std::to_string(max_tokens),
        "--column-quantum", std::to_string(column_quantum),
        "--max-requests", std::to_string(cfg.max_requests),
    };
    if (cfg.f16_io) {
        arguments.push_back("--f16-io");
    }
    std::vector<char *> pointers;
    pointers.reserve(arguments.size() + 1);
    for (std::string & value : arguments) {
        pointers.push_back(value.data());
    }
    pointers.push_back(nullptr);
    execv(cfg.worker.c_str(), pointers.data());
    fprintf(stderr, "[resident-workers] exec failed: %s\n", strerror(errno));
    _exit(127);
}

bool wait_ready(int port, pid_t pid) {
    const auto deadline = steady_clock::now() + std::chrono::minutes(8);
    while (steady_clock::now() < deadline && !stop_requested) {
        if (port_ready(port)) {
            return true;
        }
        int status = 0;
        if (waitpid(pid, &status, WNOHANG) == pid) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

void stop_worker(pid_t pid) {
    if (pid <= 0) {
        return;
    }
    kill(pid, SIGTERM);
    const auto deadline = steady_clock::now() + std::chrono::seconds(5);
    int status = 0;
    while (steady_clock::now() < deadline) {
        const pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid || (result < 0 && errno == ECHILD)) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    kill(pid, SIGKILL);
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
}

void print_load(const worker_state & state) {
    const auto & shard = state.shard;
    fprintf(stderr,
            "RESIDENTSHARD {\"status\":\"WARM\","
            "\"session_id\":\"%s\",\"backend\":\"%s\","
            "\"artifact_sha256\":\"%s\","
            "\"layer_mask\":\"%016llx\",\"columns\":%lld,"
            "\"max_tokens\":%lld,\"resident_bytes\":%zu,"
            "\"load_count\":%ld,"
            "\"endpoint_sha256\":\"%s\","
            "\"resident_geometry_sha256\":\"%s\","
            "\"operator_plan_sha256\":\"%s\","
            "\"session_generation\":%llu}\n",
            shard.session_id.c_str(), shard.backend.c_str(),
            shard.artifact_sha256.c_str(),
            static_cast<unsigned long long>(shard.layer_mask),
            static_cast<long long>(shard.columns),
            static_cast<long long>(state.max_tokens), shard.resident_bytes,
            state.load_count, shard.endpoint_sha256.c_str(),
            shard.resident_geometry_sha256.c_str(),
            shard.operator_plan_sha256.c_str(),
            static_cast<unsigned long long>(shard.session_generation));
    fflush(stderr);
}

bool receive_exact(int fd, void * destination, size_t size) {
    char * cursor = static_cast<char *>(destination);
    while (size > 0) {
        const ssize_t count = recv(fd, cursor, size, 0);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        cursor += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

bool send_text(int fd, const std::string & value) {
    const char * cursor = value.data();
    size_t size = value.size();
    while (size > 0) {
        const ssize_t count = send(fd, cursor, size, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        cursor += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

bool receive_line(int fd, std::string & line) {
    line.clear();
    while (line.size() <= 4096) {
        char value = 0;
        const ssize_t count = recv(fd, &value, 1, 0);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count != 1) {
            return false;
        }
        if (value == '\n') {
            return true;
        }
        if (value == '\r' || value == '\0') {
            return false;
        }
        line.push_back(value);
    }
    return false;
}

int create_listener(int port) {
    if (port == 0) {
        return -1;
    }
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    const int one = 1;
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    if (fd < 0 ||
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0 ||
        bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
        listen(fd, 4) != 0) {
        if (fd >= 0) {
            close(fd);
        }
        return -1;
    }
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

bool write_manifest(const std::string & path, const std::string & payload) {
    const std::string temporary = path + ".next";
    FILE * output = fopen(temporary.c_str(), "wb");
    bool ok = output != nullptr;
    if (ok) {
        ok = fwrite(payload.data(), 1, payload.size(), output) == payload.size() &&
                fflush(output) == 0 && fsync(fileno(output)) == 0;
        ok = fclose(output) == 0 && ok;
        output = nullptr;
    }
    if (!ok) {
        if (output != nullptr) {
            fclose(output);
        }
        unlink(temporary.c_str());
        return false;
    }
    if (rename(temporary.c_str(), path.c_str()) != 0) {
        unlink(temporary.c_str());
        return false;
    }
    return true;
}

bool replace_one_session(
        const config & cfg, const std::string & payload,
        const std::string & manifest_sha256, int64_t column_quantum,
        int64_t max_tokens, bool report_column_quantum,
        bool report_max_tokens,
        std::map<std::string, worker_state> & workers,
        std::string & response) {
    const std::string received = cfg.manifest + ".received";
    if (!write_manifest(received, payload)) {
        response = "S42RESIDENCY_ERROR write_received\n";
        return false;
    }
    std::vector<ffn_split::resident_session_shard> target;
    std::string error;
    const bool loaded = ffn_split::load_resident_session_manifest(
            received, target, error);
    unlink(received.c_str());
    if (!loaded) {
        response = "S42RESIDENCY_ERROR invalid_manifest\n";
        return false;
    }
    std::map<std::string, ffn_split::resident_session_shard> target_by_session;
    for (const auto & shard : target) {
        target_by_session.emplace(shard.session_id, shard);
    }
    std::vector<std::string> changed;
    for (const auto & row : workers) {
        const auto found = target_by_session.find(row.first);
        if (found == target_by_session.end() ||
            !same_shard(row.second.shard, found->second)) {
            changed.push_back(row.first);
        }
    }
    for (const auto & row : target_by_session) {
        if (workers.find(row.first) == workers.end()) {
            changed.push_back(row.first);
        }
    }
    std::sort(changed.begin(), changed.end());
    changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
    if (changed.size() != 1) {
        response = "S42RESIDENCY_ERROR changed_session_count\n";
        return false;
    }
    const std::string session_id = changed.front();
    const auto next = target_by_session.find(session_id);
    if (next != target_by_session.end() &&
        next->second.columns % column_quantum != 0) {
        response = "S42RESIDENCY_ERROR column_quantum\n";
        return false;
    }
    const auto old = workers.find(session_id);
    const bool had_old = old != workers.end();
    worker_state previous;
    if (next != target_by_session.end()) {
        print_phase("LOAD_AUTHORIZED", next->second);
    } else if (had_old) {
        print_phase("EVICTION_AUTHORIZED", old->second.shard);
    }
    if (had_old) {
        previous = old->second;
        stop_worker(previous.pid);
        workers.erase(old);
    }
    worker_state replacement;
    bool replacement_ready = next == target_by_session.end();
    if (next != target_by_session.end()) {
        replacement.shard = next->second;
        replacement.column_quantum = column_quantum;
        replacement.max_tokens = max_tokens;
        replacement.load_count = had_old ? previous.load_count + 1 : 1;
        replacement.pid = launch_worker(
                cfg, replacement.shard, replacement.column_quantum,
                replacement.max_tokens);
        replacement_ready = replacement.pid > 0 &&
                wait_ready(replacement.shard.port, replacement.pid);
        if (replacement_ready) {
            print_phase("VERIFIED", replacement.shard);
        }
    }
    if (!replacement_ready || !write_manifest(cfg.manifest, payload)) {
        if (next != target_by_session.end()) {
            print_phase("LOAD_FAILED", next->second);
        }
        if (replacement.pid > 0) {
            stop_worker(replacement.pid);
        }
        if (had_old) {
            print_phase("ROLLBACK_BEGIN", previous.shard);
            previous.pid = launch_worker(
                    cfg, previous.shard, previous.column_quantum,
                    previous.max_tokens);
            if (previous.pid <= 0 || !wait_ready(previous.shard.port, previous.pid)) {
                print_phase("ROLLBACK_FAILED", previous.shard);
                response = "S42RESIDENCY_ERROR rollback_failed\n";
                stop_requested = 1;
                return false;
            }
            ++previous.load_count;
            workers.emplace(session_id, previous);
            print_load(previous);
            print_phase("ROLLBACK_READY", previous.shard);
        }
        response = "S42RESIDENCY_ERROR replacement_failed\n";
        return false;
    }
    if (next != target_by_session.end()) {
        workers.emplace(session_id, replacement);
        print_load(replacement);
        print_phase("READY", replacement.shard);
    }
    const long load_count = next == target_by_session.end() ?
            0 : replacement.load_count;
    const uint64_t session_generation = next == target_by_session.end() ?
            0 : replacement.shard.session_generation;
    response = "S42RESIDENCY_READY " + manifest_sha256 + " " +
            session_id + " " + std::to_string(load_count) +
            (report_column_quantum ?
                    " " + std::to_string(column_quantum) : "") +
            (report_max_tokens ?
                    " " + std::to_string(max_tokens) +
                    " " + std::to_string(session_generation) : "") + "\n";
    fprintf(stderr,
            "RESIDENTRECONFIG {\"status\":\"READY\","
            "\"session_id\":\"%s\",\"manifest_sha256\":\"%s\","
            "\"load_count\":%ld,\"column_quantum\":%lld,"
            "\"max_tokens\":%lld,\"session_generation\":%llu}\n",
            session_id.c_str(), manifest_sha256.c_str(), load_count,
            static_cast<long long>(column_quantum),
            static_cast<long long>(max_tokens),
            static_cast<unsigned long long>(session_generation));
    fflush(stderr);
    return true;
}

void serve_control(
        int listener, const config & cfg,
        std::map<std::string, worker_state> & workers) {
    const int client = accept(listener, nullptr, nullptr);
    if (client < 0) {
        return;
    }
    timeval timeout = {30, 0};
    setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    std::string header;
    std::string response = "S42RESIDENCY_ERROR invalid_request\n";
    if (receive_line(client, header)) {
        char schema[32] = {};
        char digest[80] = {};
        char quantum_text[32] = {};
        char max_tokens_text[32] = {};
        size_t size = 0;
        char extra = 0;
        int64_t column_quantum = cfg.column_quantum;
        int64_t max_tokens = cfg.max_tokens;
        const bool legacy =
                sscanf(header.c_str(), "%31s %zu %79s %c",
                        schema, &size, digest, &extra) == 3 &&
                strcmp(schema, "S42RESIDENCY_V1") == 0;
        const bool current =
                sscanf(header.c_str(), "%31s %zu %79s %31s %c",
                        schema, &size, digest, quantum_text, &extra) == 4 &&
                strcmp(schema, "S42RESIDENCY_V2") == 0 &&
                parse_number(quantum_text, 1, INT_MAX, column_quantum);
        const bool shaped =
                sscanf(header.c_str(), "%31s %zu %79s %31s %31s %c",
                        schema, &size, digest, quantum_text,
                        max_tokens_text, &extra) == 5 &&
                strcmp(schema, "S42RESIDENCY_V3") == 0 &&
                parse_number(quantum_text, 1, INT_MAX, column_quantum) &&
                parse_number(max_tokens_text, 1, UINT16_MAX, max_tokens);
        if ((legacy || current || shaped) && size > 0 && size <= 65536 &&
            ffn_split::sha256_text(digest)) {
            std::string payload(size, '\0');
            if (receive_exact(client, payload.data(), payload.size())) {
                replace_one_session(
                        cfg, payload, digest, column_quantum,
                        max_tokens, current || shaped, shaped,
                        workers, response);
            }
        }
    }
    send_text(client, response);
    close(client);
}

} // namespace

int main(int argc, char ** argv) {
    config cfg;
    if (!parse_config(argc, argv, cfg)) {
        fprintf(stderr,
                "usage: %s --manifest PATH --worker PATH --model GGUF "
                "[--max-tokens N] [--column-quantum N] "
                "[--max-requests N] [--control-port N] [--f16-io]\n",
                argv[0]);
        return 2;
    }
    std::vector<ffn_split::resident_session_shard> shards;
    std::string error;
    if (!ffn_split::load_resident_session_manifest(cfg.manifest, shards, error)) {
        fprintf(stderr, "[resident-workers] %s\n", error.c_str());
        return 2;
    }
    if (std::any_of(shards.begin(), shards.end(), [&](const auto & shard) {
            return shard.columns % cfg.column_quantum != 0;
        })) {
        fprintf(stderr, "[resident-workers] shard width is not quantum aligned\n");
        return 2;
    }
    signal(SIGINT, handle_stop);
    signal(SIGTERM, handle_stop);
    signal(SIGPIPE, SIG_IGN);

    std::map<std::string, worker_state> workers;
    for (const auto & shard : shards) {
        worker_state state;
        state.shard = shard;
        state.column_quantum = cfg.column_quantum;
        state.max_tokens = cfg.max_tokens;
        state.load_count = 1;
        print_phase("LOAD_AUTHORIZED", shard);
        state.pid = launch_worker(
                cfg, shard, state.column_quantum, state.max_tokens);
        if (state.pid <= 0 || !wait_ready(shard.port, state.pid)) {
            print_phase("LOAD_FAILED", shard);
            fprintf(stderr,
                    "[resident-workers] session=%s failed to become warm\n",
                    shard.session_id.c_str());
            stop_requested = 1;
            break;
        }
        print_phase("VERIFIED", shard);
        workers.emplace(shard.session_id, state);
        print_load(state);
        print_phase("READY", shard);
    }
    if (stop_requested) {
        for (const auto & row : workers) {
            stop_worker(row.second.pid);
        }
        return 1;
    }
    const int listener = create_listener(cfg.control_port);
    if (cfg.control_port != 0 && listener < 0) {
        fprintf(stderr, "[resident-workers] control listener failed\n");
        for (const auto & row : workers) {
            stop_worker(row.second.pid);
        }
        return 1;
    }
    fprintf(stderr,
            "RESIDENTSHARDS {\"status\":\"WARM\","
            "\"session_count\":%zu,\"control_port\":%d}\n",
            workers.size(), cfg.control_port);
    fflush(stderr);

    int result = 0;
    while (!stop_requested) {
        for (const auto & row : workers) {
            int status = 0;
            if (waitpid(row.second.pid, &status, WNOHANG) == row.second.pid) {
                fprintf(stderr,
                        "[resident-workers] session=%s exited unexpectedly\n",
                        row.first.c_str());
                result = 1;
                stop_requested = 1;
                break;
            }
        }
        if (stop_requested) {
            break;
        }
        if (listener >= 0) {
            serve_control(listener, cfg, workers);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (listener >= 0) {
        close(listener);
    }
    for (const auto & row : workers) {
        stop_worker(row.second.pid);
    }
    return result;
}
