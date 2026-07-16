#include "phone_pim_protocol.h"
#include "phone_pim_socket.h"
#include "phone_pim_store.h"

#if !defined(__linux__)

#include <cstdio>

int main() {
    std::fprintf(stderr, "phone PIM stream integration test requires Linux\n");
    return 1;
}

#else

#include <arpa/inet.h>
#include <dirent.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace pp = phone_pim;

namespace {

constexpr uint64_t k_route_epoch = 7;
constexpr uint64_t k_generation = 11;
constexpr uint64_t k_max_payload = 1024 * 1024;

int checks = 0;

void require(bool condition, const char * name) {
    ++checks;
    std::printf("  %s %s\n", condition ? "PASS" : "FAIL", name);
    if (!condition) {
        throw std::runtime_error(name);
    }
}

bool write_all(int fd, const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    size_t completed = 0;
    while (completed < size) {
        const ssize_t count = write(fd, bytes + completed, size - completed);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        completed += static_cast<size_t>(count);
    }
    return true;
}

void remove_tree(const std::string & path) {
    struct stat st = {};
    if (lstat(path.c_str(), &st) != 0) return;
    if (!S_ISDIR(st.st_mode)) {
        unlink(path.c_str());
        return;
    }
    DIR * dir = opendir(path.c_str());
    if (dir != nullptr) {
        while (dirent * entry = readdir(dir)) {
            if (std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            remove_tree(path + "/" + entry->d_name);
        }
        closedir(dir);
    }
    rmdir(path.c_str());
}

uint16_t reserve_loopback_port() {
    pp::Fd socket_fd(socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
    if (!socket_fd.valid()) return 0;
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (bind(socket_fd.get(), reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0) {
        return 0;
    }
    socklen_t size = sizeof(address);
    if (getsockname(socket_fd.get(), reinterpret_cast<sockaddr *>(&address), &size) != 0) {
        return 0;
    }
    return ntohs(address.sin_port);
}

class WorkerHarness {
public:
    explicit WorkerHarness(const char * worker) {
        char resolved[PATH_MAX] = {};
        if (worker == nullptr || realpath(worker, resolved) == nullptr || access(resolved, X_OK) != 0) {
            throw std::runtime_error("worker binary is not executable");
        }
        worker_ = resolved;
        char root_template[] = "/tmp/phone_pim_stream_XXXXXX";
        char * root = mkdtemp(root_template);
        if (root == nullptr) {
            throw std::runtime_error("cannot create temporary test directory");
        }
        root_ = root;
        store_ = root_ + "/store";
        dummy_ = root_ + "/dummy";
        log_ = root_ + "/worker.log";
        pp::Fd file(open(dummy_.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600));
        static constexpr char dummy_bytes[] = "not-a-gguf";
        if (!file.valid() || !write_all(file.get(), dummy_bytes, sizeof(dummy_bytes) - 1) ||
            fsync(file.get()) != 0) {
            file.reset();
            remove_tree(root_);
            root_.clear();
            throw std::runtime_error("cannot create dummy pre-staged model");
        }
    }

    ~WorkerHarness() {
        stop();
        if (!root_.empty()) remove_tree(root_);
    }

    WorkerHarness(const WorkerHarness &) = delete;
    WorkerHarness & operator=(const WorkerHarness &) = delete;

    void start(uint16_t port) {
        if (pid_ > 0) throw std::runtime_error("worker is already running");
        const std::string port_text = std::to_string(port);
        pid_ = fork();
        if (pid_ < 0) throw std::runtime_error("cannot fork worker");
        if (pid_ == 0) {
            prctl(PR_SET_PDEATHSIG, SIGKILL);
            const int log_fd = open(log_.c_str(), O_CREAT | O_WRONLY | O_APPEND | O_CLOEXEC, 0600);
            if (log_fd >= 0) {
                dup2(log_fd, STDOUT_FILENO);
                dup2(log_fd, STDERR_FILENO);
                close(log_fd);
            }
            execl(worker_.c_str(), worker_.c_str(),
                  "--model", dummy_.c_str(),
                  "--store-dir", store_.c_str(),
                  "--backend", "CPU",
                  "--bind", "127.0.0.1",
                  "--port", port_text.c_str(),
                  "--route-epoch", "7",
                  "--generation", "11",
                  "--max-payload-mib", "1",
                  "--max-store-mib", "1",
                  "--max-model-mib", "1",
                  "--min-free-mib", "0",
                  "--timeout-ms", "1000",
                  static_cast<char *>(nullptr));
            _exit(127);
        }
    }

    bool kill_and_wait() {
        if (pid_ <= 0) return false;
        if (kill(pid_, SIGKILL) != 0 && errno != ESRCH) return false;
        int status = 0;
        while (waitpid(pid_, &status, 0) < 0) {
            if (errno != EINTR) return false;
        }
        pid_ = -1;
        return WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL;
    }

    bool wait_clean(int timeout_ms) {
        if (pid_ <= 0) return false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        int status = 0;
        while (std::chrono::steady_clock::now() < deadline) {
            const pid_t result = waitpid(pid_, &status, WNOHANG);
            if (result == pid_) {
                pid_ = -1;
                return WIFEXITED(status) && WEXITSTATUS(status) == 0;
            }
            if (result < 0 && errno != EINTR) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    }

    void stop() {
        if (pid_ <= 0) return;
        kill(pid_, SIGKILL);
        while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {}
        pid_ = -1;
    }

    void dump_log() const {
        FILE * file = std::fopen(log_.c_str(), "rb");
        if (file == nullptr) return;
        std::fprintf(stderr, "--- worker log ---\n");
        std::array<char, 4096> buffer = {};
        while (const size_t count = std::fread(buffer.data(), 1, buffer.size(), file)) {
            std::fwrite(buffer.data(), 1, count, stderr);
        }
        std::fprintf(stderr, "--- end worker log ---\n");
        std::fclose(file);
    }

    const std::string & store() const { return store_; }

private:
    std::string worker_;
    std::string root_;
    std::string store_;
    std::string dummy_;
    std::string log_;
    pid_t pid_ = -1;
};

struct HelloInfo {
    uint32_t version = 0;
    uint64_t session_epoch = 0;
    uint64_t max_payload = 0;
    std::string backend;
    uint64_t feature_bits = 0;
    uint64_t max_model_bytes = 0;
    uint64_t max_chunk_bytes = 0;
    uint32_t max_chunks = 0;
    std::string capability;
};

class WireClient {
public:
    bool connect_to(uint16_t port, std::string & error) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            error.clear();
            socket_ = pp::connect_tcp("127.0.0.1", port, 250, error);
            if (socket_.valid()) {
                return pp::set_socket_deadlines(socket_.get(), 5000, error);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        return false;
    }

    bool hello(HelloInfo & info, std::string & error) {
        pp::Frame response;
        if (!call_ok(pp::Opcode::hello, {}, 0, response, error)) return false;
        pp::Reader reader(response.payload);
        if (!reader.u32(info.version) || !reader.u64(info.session_epoch) ||
            !reader.u64(info.max_payload) || !reader.string(info.backend, 1024) ||
            !reader.u64(info.feature_bits) || !reader.u64(info.max_model_bytes) ||
            !reader.u64(info.max_chunk_bytes) || !reader.u32(info.max_chunks) ||
            !reader.string(info.capability, 1024) || !reader.done() || info.session_epoch == 0) {
            error = "malformed HELLO response";
            return false;
        }
        session_epoch_ = info.session_epoch;
        return true;
    }

    bool call_ok(
            pp::Opcode opcode,
            const std::vector<uint8_t> & payload,
            uint64_t generation,
            pp::Frame & response,
            std::string & error) {
        if (!exchange(opcode, payload, generation, response, error)) return false;
        if (response.header.opcode != opcode || response.header.flags != pp::flag_response) {
            error = "expected successful response";
            return false;
        }
        return true;
    }

    bool call_error(
            pp::Opcode opcode,
            const std::vector<uint8_t> & payload,
            uint64_t generation,
            pp::ErrorCode expected,
            std::string & remote_message,
            std::string & error) {
        pp::Frame response;
        if (!exchange(opcode, payload, generation, response, error)) return false;
        pp::ErrorCode code = pp::ErrorCode::internal;
        if (response.header.opcode != pp::Opcode::error ||
            response.header.flags != (pp::flag_response | pp::flag_error) ||
            !pp::parse_error_payload(response.payload, code, remote_message) || code != expected) {
            error = "unexpected remote error response";
            return false;
        }
        return true;
    }

    bool send_truncated(
            pp::Opcode opcode,
            const std::vector<uint8_t> & payload,
            uint64_t generation,
            size_t payload_prefix,
            std::string & error) {
        if (!socket_.valid() || payload_prefix >= payload.size()) {
            error = "invalid truncated-frame request";
            return false;
        }
        pp::Header header = next_header(opcode, generation);
        header.payload_bytes = payload.size();
        header.payload_sha256 = pp::sha256(payload.data(), payload.size());
        const auto encoded = pp::encode_header(header);
        if (!pp::send_bytes(socket_.get(), encoded.data(), encoded.size(), error) ||
            !pp::send_bytes(socket_.get(), payload.data(), payload_prefix, error)) {
            return false;
        }
        shutdown(socket_.get(), SHUT_RDWR);
        socket_.reset();
        return true;
    }

    void close() { socket_.reset(); }

private:
    pp::Header next_header(pp::Opcode opcode, uint64_t generation) {
        pp::Header header;
        header.opcode = opcode;
        header.request_id = ++request_id_;
        header.command_seq = ++command_seq_;
        header.session_epoch = opcode == pp::Opcode::hello ? 0 : session_epoch_;
        header.route_epoch = k_route_epoch;
        header.residency_generation = opcode == pp::Opcode::hello ? 0 : generation;
        return header;
    }

    bool exchange(
            pp::Opcode opcode,
            const std::vector<uint8_t> & payload,
            uint64_t generation,
            pp::Frame & response,
            std::string & error) {
        if (!socket_.valid() || payload.size() > k_max_payload ||
            payload.size() > pp::opcode_payload_limit(opcode)) {
            error = "invalid client request";
            return false;
        }
        pp::Header header = next_header(opcode, generation);
        if (!pp::send_frame(socket_.get(), header, payload, error) ||
            pp::receive_frame(socket_.get(), response, k_max_payload, error) != pp::ReceiveResult::ok) {
            return false;
        }
        if (response.header.request_id != header.request_id ||
            response.header.command_seq != header.command_seq ||
            response.header.session_epoch != header.session_epoch ||
            response.header.route_epoch != header.route_epoch ||
            response.header.residency_generation != header.residency_generation) {
            error = "response header does not bind the request";
            return false;
        }
        return true;
    }

    pp::Fd socket_;
    uint64_t request_id_ = 0;
    uint64_t command_seq_ = 0;
    uint64_t session_epoch_ = 0;
};

pp::StageObjectSpec make_spec(const std::vector<uint8_t> & data, uint32_t chunk_bytes) {
    pp::StageObjectSpec spec;
    spec.bytes = data.size();
    spec.sha256 = pp::sha256(data.data(), data.size());
    spec.chunk_bytes = chunk_bytes;
    uint64_t offset = 0;
    while (offset < data.size()) {
        pp::StageChunkSpec chunk;
        chunk.index = static_cast<uint32_t>(spec.chunks.size());
        chunk.offset = offset;
        chunk.bytes = static_cast<uint32_t>(std::min<uint64_t>(chunk_bytes, data.size() - offset));
        chunk.sha256 = pp::sha256(data.data() + offset, chunk.bytes);
        spec.chunks.push_back(chunk);
        offset += chunk.bytes;
    }
    spec.manifest_sha256 = pp::stage_manifest_sha256(spec);
    return spec;
}

std::vector<uint8_t> begin_payload(uint64_t ticket, const pp::StageObjectSpec & spec) {
    pp::Writer writer;
    writer.u64(ticket);
    writer.u64(spec.bytes);
    writer.bytes(spec.sha256.data(), spec.sha256.size());
    writer.u32(spec.chunk_bytes);
    writer.u32(static_cast<uint32_t>(spec.chunks.size()));
    writer.bytes(spec.manifest_sha256.data(), spec.manifest_sha256.size());
    for (const pp::StageChunkSpec & chunk : spec.chunks) {
        writer.u32(chunk.index);
        writer.u64(chunk.offset);
        writer.u32(chunk.bytes);
        writer.bytes(chunk.sha256.data(), chunk.sha256.size());
    }
    return writer.take();
}

std::vector<uint8_t> chunk_payload(
        uint64_t ticket,
        const pp::StageObjectSpec & spec,
        const std::vector<uint8_t> & data,
        uint32_t index,
        bool corrupt) {
    const pp::StageChunkSpec & chunk = spec.chunks.at(index);
    std::vector<uint8_t> bytes(
            data.begin() + static_cast<ptrdiff_t>(chunk.offset),
            data.begin() + static_cast<ptrdiff_t>(chunk.offset + chunk.bytes));
    if (corrupt) bytes[bytes.size() / 2] ^= 0x80;
    pp::Writer writer;
    writer.u64(ticket);
    writer.u32(chunk.index);
    writer.u64(chunk.offset);
    writer.u32(chunk.bytes);
    writer.bytes(chunk.sha256.data(), chunk.sha256.size());
    writer.bytes(bytes.data(), bytes.size());
    return writer.take();
}

bool decode_progress(pp::Reader & reader, pp::StageProgress & progress) {
    uint32_t state = 0;
    if (!reader.u32(state) ||
        (state != static_cast<uint32_t>(pp::StageState::receiving) &&
         state != static_cast<uint32_t>(pp::StageState::published)) ||
        !reader.u64(progress.ticket_id) || !reader.u64(progress.residency_generation) ||
        !reader.u64(progress.verified_bytes) || !reader.u32(progress.next_chunk) ||
        !reader.bytes(progress.prefix_sha256.data(), progress.prefix_sha256.size()) ||
        !reader.bytes(progress.manifest_sha256.data(), progress.manifest_sha256.size()) ||
        !reader.u64(progress.resume_scan_us)) {
        return false;
    }
    progress.state = static_cast<pp::StageState>(state);
    return true;
}

bool decode_progress(const std::vector<uint8_t> & payload, pp::StageProgress & progress) {
    pp::Reader reader(payload);
    return decode_progress(reader, progress) && reader.done();
}

bool begin_and_decode(
        WireClient & client,
        uint64_t ticket,
        const pp::StageObjectSpec & spec,
        pp::StageProgress & progress,
        std::string & error) {
    pp::Frame response;
    if (!client.call_ok(pp::Opcode::stage_begin, begin_payload(ticket, spec),
                        k_generation, response, error)) {
        return false;
    }
    pp::Reader reader(response.payload);
    uint32_t state = 0;
    if (!reader.u32(state) ||
        (state != static_cast<uint32_t>(pp::StageState::receiving) &&
         state != static_cast<uint32_t>(pp::StageState::published)) ||
        !reader.u64(progress.ticket_id) || !reader.u64(progress.residency_generation) ||
        !reader.u64(progress.verified_bytes) || !reader.u32(progress.next_chunk) ||
        !reader.bytes(progress.prefix_sha256.data(), progress.prefix_sha256.size()) ||
        !reader.bytes(progress.manifest_sha256.data(), progress.manifest_sha256.size()) ||
        !reader.u64(progress.resume_scan_us) || !reader.done()) {
        error = "malformed STAGE_BEGIN response";
        return false;
    }
    progress.state = static_cast<pp::StageState>(state);
    return true;
}

bool put_and_decode(
        WireClient & client,
        uint64_t ticket,
        const pp::StageObjectSpec & spec,
        const std::vector<uint8_t> & data,
        uint32_t index,
        pp::StageProgress & progress,
        std::string & error) {
    pp::Frame response;
    if (!client.call_ok(pp::Opcode::stage_chunk, chunk_payload(ticket, spec, data, index, false),
                        k_generation, response, error) ||
        !decode_progress(response.payload, progress)) {
        if (error.empty()) error = "malformed STAGE_CHUNK response";
        return false;
    }
    return true;
}

bool published_bytes_match(
        const std::string & store,
        const pp::StageObjectSpec & spec,
        const std::vector<uint8_t> & expected) {
    const std::string path = store + "/" + pp::hex_sha256(spec.sha256) + ".gguf";
    pp::Fd file(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    struct stat st = {};
    if (!file.valid() || fstat(file.get(), &st) != 0 || !S_ISREG(st.st_mode) ||
        st.st_size < 0 || static_cast<uint64_t>(st.st_size) != spec.bytes) {
        return false;
    }
    std::vector<uint8_t> actual(expected.size());
    size_t completed = 0;
    while (completed < actual.size()) {
        const ssize_t count = pread(file.get(), actual.data() + completed,
                                    actual.size() - completed, static_cast<off_t>(completed));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return false;
        completed += static_cast<size_t>(count);
    }
    return actual == expected;
}

void validate_hello(const HelloInfo & hello) {
    require(hello.version == pp::k_version, "HELLO reports protocol v3");
    require(hello.max_payload == k_max_payload, "HELLO reports the configured payload cap");
    require(hello.max_chunk_bytes + pp::k_stage_chunk_envelope_bytes == hello.max_payload,
            "HELLO chunk cap accounts for the 56-byte envelope");
    require(hello.feature_bits == 3, "HELLO reports dynamic and pre-staged sources");
    require(hello.max_model_bytes == k_max_payload, "HELLO reports the configured model cap");
    require(hello.max_chunks == pp::k_max_stage_chunks, "HELLO reports the chunk-count cap");
    require(hello.backend == "CPU" &&
                    hello.capability == "sequential_dynamic_gemma4_dense_ffn_v3",
            "HELLO binds backend and dynamic capability");
}

void run_test(WorkerHarness & worker) {
    std::vector<uint8_t> data(20011);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 37 + i / 7 + 11) & 0xff);
    }
    const pp::StageObjectSpec spec = make_spec(data, 4096);
    require(spec.chunks.size() == 5, "deterministic object spans five chunks");
    const std::array<uint8_t, 32> empty_digest = pp::sha256(data.data(), 0);
    std::string error;

    uint16_t port = reserve_loopback_port();
    require(port != 0, "reserved initial loopback port");
    worker.start(port);
    WireClient first;
    require(first.connect_to(port, error), "connected to the real worker");
    HelloInfo hello;
    require(first.hello(hello, error), "completed initial HELLO");
    validate_hello(hello);

    constexpr uint64_t ticket1 = 1001;
    std::string remote_message;
    require(first.call_error(pp::Opcode::stage_begin, begin_payload(ticket1, spec),
                             k_generation - 1, pp::ErrorCode::stale_epoch,
                             remote_message, error) &&
                    remote_message.find("generation") != std::string::npos,
            "stale-generation STAGE_BEGIN is rejected");

    pp::StageProgress progress;
    require(begin_and_decode(first, ticket1, spec, progress, error) &&
                    progress.state == pp::StageState::receiving &&
                    progress.ticket_id == ticket1 &&
                    progress.residency_generation == k_generation &&
                    progress.verified_bytes == 0 && progress.next_chunk == 0 &&
                    progress.prefix_sha256 == empty_digest &&
                    progress.manifest_sha256 == spec.manifest_sha256,
            "valid STAGE_BEGIN starts at the exact empty prefix");

    require(first.call_error(pp::Opcode::stage_chunk,
                             chunk_payload(ticket1, spec, data, 0, true),
                             k_generation, pp::ErrorCode::hash_mismatch,
                             remote_message, error),
            "application-corrupt chunk with a valid outer frame is rejected");
    require(begin_and_decode(first, ticket1, spec, progress, error) &&
                    progress.verified_bytes == 0 && progress.next_chunk == 0 &&
                    progress.prefix_sha256 == empty_digest,
            "rejected application payload does not advance the durable prefix");

    require(put_and_decode(first, ticket1, spec, data, 0, progress, error) &&
                    progress.verified_bytes == spec.chunks[0].bytes && progress.next_chunk == 1 &&
                    progress.prefix_sha256 == pp::sha256(data.data(), spec.chunks[0].bytes),
            "valid chunk ACK carries the exact durable prefix hash");
    const pp::StageProgress first_ack = progress;
    require(put_and_decode(first, ticket1, spec, data, 0, progress, error) &&
                    progress.verified_bytes == first_ack.verified_bytes &&
                    progress.next_chunk == first_ack.next_chunk &&
                    progress.prefix_sha256 == first_ack.prefix_sha256,
            "exact duplicate chunk receives an idempotent ACK");

    const std::vector<uint8_t> second_payload = chunk_payload(ticket1, spec, data, 1, false);
    require(first.send_truncated(pp::Opcode::stage_chunk, second_payload, k_generation,
                                 second_payload.size() / 2, error),
            "connection is cut in the middle of a valid frame");

    WireClient reconnect;
    require(reconnect.connect_to(port, error), "reconnected after truncated frame");
    HelloInfo reconnect_hello;
    require(reconnect.hello(reconnect_hello, error), "completed reconnect HELLO");
    constexpr uint64_t ticket2 = 1002;
    require(begin_and_decode(reconnect, ticket2, spec, progress, error) &&
                    progress.verified_bytes == spec.chunks[0].bytes && progress.next_chunk == 1 &&
                    progress.prefix_sha256 == pp::sha256(data.data(), spec.chunks[0].bytes),
            "truncated frame reconnect recovers only the exact durable prefix");
    require(put_and_decode(reconnect, ticket2, spec, data, 1, progress, error) &&
                    progress.verified_bytes == spec.chunks[0].bytes + spec.chunks[1].bytes &&
                    progress.next_chunk == 2 &&
                    progress.prefix_sha256 == pp::sha256(data.data(), progress.verified_bytes),
            "second chunk is durable before hard restart");
    const uint64_t restart_prefix = progress.verified_bytes;
    reconnect.close();
    require(worker.kill_and_wait(), "worker is terminated with SIGKILL");

    port = reserve_loopback_port();
    require(port != 0, "reserved restart loopback port");
    worker.start(port);
    WireClient restarted;
    require(restarted.connect_to(port, error), "connected to restarted worker");
    HelloInfo restart_hello;
    require(restarted.hello(restart_hello, error), "completed post-restart HELLO");
    constexpr uint64_t ticket3 = 1003;
    require(begin_and_decode(restarted, ticket3, spec, progress, error) &&
                    progress.state == pp::StageState::receiving &&
                    progress.verified_bytes == restart_prefix && progress.next_chunk == 2 &&
                    progress.prefix_sha256 == pp::sha256(data.data(), restart_prefix),
            "worker restart reconstructs the exact durable prefix");

    for (uint32_t i = progress.next_chunk; i < spec.chunks.size(); ++i) {
        require(put_and_decode(restarted, ticket3, spec, data, i, progress, error),
                "remaining ordered chunk is acknowledged");
    }
    pp::Writer commit_writer;
    commit_writer.u64(ticket3);
    commit_writer.bytes(spec.manifest_sha256.data(), spec.manifest_sha256.size());
    pp::Frame commit_response;
    require(restarted.call_ok(pp::Opcode::stage_commit, commit_writer.data(),
                              k_generation, commit_response, error),
            "complete object commits through the real worker");
    pp::Reader commit_reader(commit_response.payload);
    uint32_t state = 0;
    pp::StageProgress committed;
    committed.state = pp::StageState::receiving;
    require(commit_reader.u32(state) && state == static_cast<uint32_t>(pp::StageState::published) &&
                    commit_reader.u64(committed.ticket_id) &&
                    commit_reader.u64(committed.residency_generation) &&
                    commit_reader.u64(committed.verified_bytes) &&
                    commit_reader.u32(committed.next_chunk) &&
                    commit_reader.bytes(committed.prefix_sha256.data(), committed.prefix_sha256.size()) &&
                    commit_reader.bytes(committed.manifest_sha256.data(), committed.manifest_sha256.size()) &&
                    commit_reader.u64(committed.resume_scan_us) &&
                    committed.ticket_id == ticket3 && committed.residency_generation == k_generation &&
                    committed.verified_bytes == spec.bytes &&
                    committed.next_chunk == spec.chunks.size() &&
                    committed.prefix_sha256 == spec.sha256 &&
                    committed.manifest_sha256 == spec.manifest_sha256,
            "STAGE_COMMIT returns the published identity");
    uint64_t metric = 0;
    for (int i = 0; i < 9; ++i) {
        require(commit_reader.u64(metric), "STAGE_COMMIT carries every metric field");
    }
    require(commit_reader.done() && published_bytes_match(worker.store(), spec, data),
            "atomic publication exposes exactly the committed bytes");

    static constexpr char absent_label[] = "absent-published-object";
    const auto absent_digest = pp::sha256(absent_label, sizeof(absent_label) - 1);
    pp::Writer prepare_writer;
    prepare_writer.u64(77);
    prepare_writer.u32(1);
    prepare_writer.u32(static_cast<uint32_t>(pp::ModelSource::published_store));
    prepare_writer.string("blk.2");
    prepare_writer.u64(spec.bytes);
    prepare_writer.bytes(absent_digest.data(), absent_digest.size());
    require(restarted.call_error(pp::Opcode::prepare, prepare_writer.data(),
                                 k_generation, pp::ErrorCode::invalid_state,
                                 remote_message, error) &&
                    remote_message.find("durably published") != std::string::npos,
            "published-store PREPARE never falls back to the existing pre-staged path");

    pp::Frame shutdown_response;
    require(restarted.call_ok(pp::Opcode::shutdown, {}, k_generation,
                              shutdown_response, error),
            "worker accepts a generation-bound shutdown");
    restarted.close();
    require(worker.wait_clean(5000), "worker exits cleanly after shutdown");
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s /path/to/llama-phone-pim-worker\n", argv[0]);
        return 2;
    }
    try {
        WorkerHarness worker(argv[1]);
        try {
            run_test(worker);
        } catch (...) {
            worker.dump_log();
            throw;
        }
        std::printf("stream integration tests: %d checks, 0 failures\n", checks);
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "stream integration test failed: %s\n", exception.what());
        return 1;
    }
}

#endif
