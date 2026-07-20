#include "phone_pim_client.h"
#include "phone_pim_json.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <array>
#include <cstring>
#include <cstdio>
#include <functional>
#include <string>
#include <thread>
#include <utility>

namespace pp = phone_pim;

namespace {

int checks = 0;
int failures = 0;

void check(bool condition, const char * name) {
    ++checks;
    std::printf("  %s %s\n", condition ? "PASS" : "FAIL", name);
    if (!condition) ++failures;
}

class ScriptedServer {
public:
    using Handler = std::function<void(int, std::string &)>;

    explicit ScriptedServer(Handler handler) {
        pp::Fd listener(socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
        const int one = 1;
        if (!listener.valid() ||
            setsockopt(listener.get(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0) {
            error_ = "cannot create test listener";
            return;
        }
        sockaddr_in address = {};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = 0;
        socklen_t address_size = sizeof(address);
        if (bind(listener.get(), reinterpret_cast<const sockaddr *>(&address), address_size) != 0 ||
            listen(listener.get(), 1) != 0 ||
            getsockname(listener.get(), reinterpret_cast<sockaddr *>(&address), &address_size) != 0) {
            error_ = "cannot bind test listener";
            return;
        }
        port_ = ntohs(address.sin_port);
        listener_ = std::move(listener);
        thread_ = std::thread([this, handler = std::move(handler)] {
            std::string accept_error;
            pp::Fd client = pp::accept_tcp(listener_.get(), 2000, accept_error);
            if (!client.valid()) {
                error_ = "accept: " + accept_error;
                return;
            }
            handler(client.get(), error_);
        });
    }

    ~ScriptedServer() {
        join();
    }

    ScriptedServer(const ScriptedServer &) = delete;
    ScriptedServer & operator=(const ScriptedServer &) = delete;

    uint16_t port() const { return port_; }

    bool join() {
        if (thread_.joinable()) thread_.join();
        return error_.empty();
    }

private:
    pp::Fd listener_;
    std::thread thread_;
    uint16_t port_ = 0;
    std::string error_;
};

bool receive_request(int fd, pp::Opcode opcode, pp::Frame & request, std::string & error) {
    if (pp::receive_frame(fd, request, pp::k_default_max_payload, error) != pp::ReceiveResult::ok ||
        request.header.opcode != opcode || request.header.flags != 0) {
        if (error.empty()) error = "unexpected request";
        return false;
    }
    return true;
}

bool send_response(
        int fd,
        const pp::Frame & request,
        pp::Opcode opcode,
        const std::vector<uint8_t> & payload,
        std::string & error,
        uint64_t request_id_delta = 0) {
    pp::Header header;
    header.opcode = opcode;
    header.flags = pp::flag_response;
    header.request_id = request.header.request_id + request_id_delta;
    header.command_seq = request.header.command_seq;
    header.session_epoch = request.header.session_epoch;
    header.route_epoch = request.header.route_epoch;
    header.residency_generation = request.header.residency_generation;
    return pp::send_frame(fd, header, payload, error);
}

std::vector<uint8_t> hello_payload(uint64_t session_epoch) {
    pp::Writer writer;
    writer.u32(pp::k_version);
    writer.u64(session_epoch);
    writer.u64(pp::k_default_max_payload);
    writer.string("HTP0");
    writer.u64(2);
    writer.u64(1024ULL * 1024 * 1024);
    writer.u64(pp::k_max_stage_chunk_bytes);
    writer.u32(pp::k_max_stage_chunks);
    writer.string("prestaged_gemma4_dense_ffn_v3");
    return writer.take();
}

std::vector<uint8_t> empty_status_payload(uint64_t generation, bool inconsistent) {
    pp::Writer writer;
    writer.u32(0);
    writer.u64(inconsistent ? 1 : 0);
    writer.u64(generation);
    writer.u64(0);
    writer.u64(0);
    writer.u32(0);
    writer.string("");
    writer.u32(0);
    writer.u32(0);
    writer.u64(0);
    writer.u64(0);
    writer.u32(0);
    const std::array<uint8_t, 32> zero = {};
    writer.bytes(zero.data(), zero.size());
    return writer.take();
}

std::array<uint8_t, 32> model_digest() {
    std::array<uint8_t, 32> digest = {};
    digest[0] = 1;
    return digest;
}

pp::PrepareRequest prepare_request() {
    pp::PrepareRequest request;
    request.island_id = 1;
    request.token_count = 1;
    request.prefix = "blk.2";
    request.model_bytes = 1;
    request.model_sha256 = model_digest();
    return request;
}

std::vector<uint8_t> prepared_payload(uint64_t island_id) {
    pp::Writer writer;
    writer.u64(island_id);
    writer.u64(1);
    writer.u64(1);
    writer.u32(1);
    writer.u64(1);
    const auto digest = model_digest();
    writer.bytes(digest.data(), digest.size());
    writer.u32(static_cast<uint32_t>(pp::ModelSource::prestaged));
    writer.u64(1);
    writer.u64(1);
    writer.u64(1);
    writer.u64(1);
    writer.u64(1);
    writer.u64(1);
    writer.string("fake backend");
    return writer.take();
}

std::vector<uint8_t> execute_payload(uint64_t input_set_us, uint64_t compute_us, uint64_t output_get_us) {
    pp::Writer writer;
    writer.u64(1);
    writer.u64(1);
    writer.u64(input_set_us);
    writer.u64(compute_us);
    writer.u64(output_get_us);
    const float output = 0.0f;
    uint32_t bits = 0;
    std::memcpy(&bits, &output, sizeof(bits));
    writer.u32(bits);
    return writer.take();
}

pp::ClientOptions options(uint16_t port, uint64_t generation_hint = 99) {
    pp::ClientOptions result;
    result.port = port;
    result.route_epoch = 7;
    result.residency_generation = generation_hint;
    result.timeout_ms = 2000;
    return result;
}

} // namespace

int main() {
    {
        const std::string first = pp::json_uint64(7611920510101663989ULL);
        const std::string second = pp::json_uint64(7611920510101663990ULL);
        check(first == "\"7611920510101663989\"" &&
              second == "\"7611920510101663990\"" && first != second,
              "JSON preserves adjacent uint64 protocol identities as strings");
    }

    {
        constexpr uint64_t session_epoch = 1234;
        constexpr uint64_t current_generation = 7;
        ScriptedServer server([](int fd, std::string & error) {
            pp::Frame request;
            if (!receive_request(fd, pp::Opcode::hello, request, error) ||
                !send_response(fd, request, pp::Opcode::hello, hello_payload(session_epoch), error) ||
                !receive_request(fd, pp::Opcode::status, request, error) ||
                !send_response(fd, request, pp::Opcode::status,
                               empty_status_payload(current_generation, false), error) ||
                !receive_request(fd, pp::Opcode::close, request, error)) {
                return;
            }
            if (request.header.session_epoch != session_epoch ||
                request.header.residency_generation != current_generation) {
                error = "CLOSE did not use discovered epochs";
                return;
            }
            send_response(fd, request, pp::Opcode::close, {}, error);
        });
        pp::ClientSession client(options(server.port()));
        pp::HelloInfo hello;
        pp::StatusInfo status;
        std::string error;
        const bool ok = client.connect(hello, error) && client.status(status, error) &&
                        status.residency_generation == current_generation &&
                        client.residency_generation() == current_generation &&
                        client.close(error) && server.join();
        check(ok, "STATUS replaces a stale generation hint before mutation");
    }

    {
        ScriptedServer server([](int fd, std::string & error) {
            pp::Frame request;
            if (receive_request(fd, pp::Opcode::hello, request, error)) {
                send_response(fd, request, pp::Opcode::hello, hello_payload(22), error);
            }
        });
        pp::ClientSession client(options(server.port()));
        pp::HelloInfo hello;
        std::string error;
        pp::PrepareRequest request = prepare_request();
        pp::PreparedInfo prepared;
        const bool connected = client.connect(hello, error);
        error.clear();
        const bool rejected = connected && !client.prepare(request, prepared, error) &&
                              error == "invalid PREPARE request";
        client.close(error);
        check(rejected && server.join(), "mutating commands are rejected before STATUS synchronization");
    }

    {
        bool all_rejected = true;
        for (int mutation = 0; mutation < 7; ++mutation) {
            ScriptedServer server([mutation](int fd, std::string & error) {
                pp::Frame request;
                if (!receive_request(fd, pp::Opcode::hello, request, error)) return;
                pp::Header header;
                header.opcode = mutation == 5 ? pp::Opcode::status : pp::Opcode::hello;
                header.flags = mutation == 6 ? pp::flag_response | 0x80U : pp::flag_response;
                header.request_id = request.header.request_id + (mutation == 0 ? 1 : 0);
                header.command_seq = request.header.command_seq + (mutation == 1 ? 1 : 0);
                header.session_epoch = request.header.session_epoch + (mutation == 2 ? 1 : 0);
                header.route_epoch = request.header.route_epoch + (mutation == 3 ? 1 : 0);
                header.residency_generation = request.header.residency_generation + (mutation == 4 ? 1 : 0);
                pp::send_frame(fd, header, hello_payload(33), error);
            });
            pp::ClientSession client(options(server.port()));
            pp::HelloInfo hello;
            std::string error;
            all_rejected = all_rejected && !client.connect(hello, error) &&
                           !client.connected() && server.join();
        }
        check(all_rejected, "response correlation rejects every mismatched header field");
    }

    {
        ScriptedServer server([](int fd, std::string & error) {
            pp::Frame request;
            if (!receive_request(fd, pp::Opcode::hello, request, error) ||
                !send_response(fd, request, pp::Opcode::hello, hello_payload(44), error) ||
                !receive_request(fd, pp::Opcode::status, request, error)) {
                return;
            }
            send_response(fd, request, pp::Opcode::status, empty_status_payload(8, true), error);
        });
        pp::ClientSession client(options(server.port()));
        pp::HelloInfo hello;
        pp::StatusInfo status;
        std::string error;
        const bool rejected = client.connect(hello, error) && !client.status(status, error) &&
                              error == "malformed STATUS response" && !client.connected();
        client.close(error);
        check(rejected && server.join(), "STATUS rejects inconsistent empty residency fields");
    }

    {
        ScriptedServer server([](int fd, std::string & error) {
            pp::Frame request;
            if (!receive_request(fd, pp::Opcode::hello, request, error) ||
                !send_response(fd, request, pp::Opcode::hello, hello_payload(45), error) ||
                !receive_request(fd, pp::Opcode::status, request, error) ||
                !send_response(fd, request, pp::Opcode::status, empty_status_payload(9, false), error) ||
                !receive_request(fd, pp::Opcode::prepare, request, error)) {
                return;
            }
            send_response(fd, request, pp::Opcode::prepare, prepared_payload(2), error);
        });
        pp::ClientSession client(options(server.port()));
        pp::HelloInfo hello;
        pp::StatusInfo status;
        pp::PreparedInfo prepared;
        prepared.island_id = 777;
        std::string error;
        const bool rejected = client.connect(hello, error) && client.status(status, error) &&
                              !client.prepare(prepare_request(), prepared, error) &&
                              error == "PREPARE response identity mismatch" &&
                              prepared.island_id == 777 && !client.connected();
        check(rejected && server.join(), "PREPARE identity mismatch poisons without publishing output");
    }

    {
        ScriptedServer server([](int fd, std::string & error) {
            pp::Frame request;
            if (!receive_request(fd, pp::Opcode::hello, request, error) ||
                !send_response(fd, request, pp::Opcode::hello, hello_payload(46), error) ||
                !receive_request(fd, pp::Opcode::status, request, error) ||
                !send_response(fd, request, pp::Opcode::status, empty_status_payload(10, false), error) ||
                !receive_request(fd, pp::Opcode::close, request, error)) {
                return;
            }
            send_response(fd, request, pp::Opcode::close, {}, error);
        });
        pp::ClientSession client(options(server.port()));
        pp::HelloInfo hello;
        pp::StatusInfo status;
        pp::ExecutionInfo execution;
        std::string error;
        const std::vector<float> input = {0.0f};
        const bool rejected = client.connect(hello, error) && client.status(status, error) &&
                              !client.execute(1, input, execution, error) &&
                              error == "invalid EXECUTE request" && client.close(error);
        check(rejected && server.join(), "EXECUTE requires an exact successful PREPARE binding");
    }

    {
        ScriptedServer server([](int fd, std::string & error) {
            pp::Frame request;
            if (!receive_request(fd, pp::Opcode::hello, request, error) ||
                !send_response(fd, request, pp::Opcode::hello, hello_payload(47), error) ||
                !receive_request(fd, pp::Opcode::status, request, error) ||
                !send_response(fd, request, pp::Opcode::status, empty_status_payload(11, false), error) ||
                !receive_request(fd, pp::Opcode::prepare, request, error) ||
                !send_response(fd, request, pp::Opcode::prepare, prepared_payload(1), error) ||
                !receive_request(fd, pp::Opcode::execute, request, error)) {
                return;
            }
            send_response(fd, request, pp::Opcode::execute,
                    execute_payload(UINT64_MAX, 1, 1), error);
        });
        pp::ClientSession client(options(server.port()));
        pp::HelloInfo hello;
        pp::StatusInfo status;
        pp::PreparedInfo prepared;
        pp::ExecutionInfo execution;
        std::string error;
        const std::vector<float> input = {0.0f};
        const bool rejected = client.connect(hello, error) && client.status(status, error) &&
                              client.prepare(prepare_request(), prepared, error) &&
                              !client.execute(1, input, execution, error) &&
                              error == "EXECUTE timing fields overflow" && !client.connected();
        check(rejected && server.join(), "EXECUTE rejects overflowing worker timing and poisons the session");
    }

    {
        ScriptedServer server([](int fd, std::string & error) {
            pp::Frame request;
            if (!receive_request(fd, pp::Opcode::hello, request, error) ||
                !send_response(fd, request, pp::Opcode::hello, hello_payload(50), error) ||
                !receive_request(fd, pp::Opcode::status, request, error) ||
                !send_response(fd, request, pp::Opcode::status, empty_status_payload(14, false), error) ||
                !receive_request(fd, pp::Opcode::prepare, request, error) ||
                !send_response(fd, request, pp::Opcode::prepare, prepared_payload(1), error) ||
                !receive_request(fd, pp::Opcode::execute, request, error)) {
                return;
            }
            send_response(fd, request, pp::Opcode::execute,
                    execute_payload(10000000, 0, 0), error);
        });
        pp::ClientSession client(options(server.port()));
        pp::HelloInfo hello;
        pp::StatusInfo status;
        pp::PreparedInfo prepared;
        pp::ExecutionInfo execution;
        std::string error;
        const std::vector<float> input = {0.0f};
        const bool rejected = client.connect(hello, error) && client.status(status, error) &&
                              client.prepare(prepare_request(), prepared, error) &&
                              !client.execute(1, input, execution, error) &&
                              error == "EXECUTE worker timing exceeds host E2E time" &&
                              !client.connected();
        check(rejected && server.join(),
              "EXECUTE rejects finite worker timing greater than client operation wall time");
    }

    {
        ScriptedServer server([](int fd, std::string & error) {
            pp::Frame request;
            if (!receive_request(fd, pp::Opcode::hello, request, error) ||
                !send_response(fd, request, pp::Opcode::hello, hello_payload(49), error) ||
                !receive_request(fd, pp::Opcode::status, request, error) ||
                !send_response(fd, request, pp::Opcode::status, empty_status_payload(13, false), error) ||
                !receive_request(fd, pp::Opcode::prepare, request, error) ||
                !send_response(fd, request, pp::Opcode::prepare, prepared_payload(1), error) ||
                !receive_request(fd, pp::Opcode::execute, request, error) ||
                !send_response(fd, request, pp::Opcode::execute, execute_payload(0, 0, 0), error) ||
                !receive_request(fd, pp::Opcode::release, request, error)) {
                return;
            }
            pp::Writer release_writer;
            release_writer.u64(14);
            if (!send_response(fd, request, pp::Opcode::release, release_writer.data(), error) ||
                !receive_request(fd, pp::Opcode::close, request, error)) {
                return;
            }
            if (request.header.residency_generation != 14) {
                error = "CLOSE did not use the released generation";
                return;
            }
            send_response(fd, request, pp::Opcode::close, {}, error);
        });
        pp::ClientSession client(options(server.port()));
        pp::HelloInfo hello;
        pp::StatusInfo status;
        pp::PreparedInfo prepared;
        pp::ExecutionInfo execution;
        std::string error;
        uint64_t next_generation = 0;
        const std::vector<float> input = {0.0f};
        const bool ok = client.connect(hello, error) && client.status(status, error) &&
                        client.prepare(prepare_request(), prepared, error) &&
                        client.execute(1, input, execution, error) &&
                        execution.output.size() == 1 &&
                        client.release(1, next_generation, error) && next_generation == 14 &&
                        client.close(error) && server.join();
        check(ok, "valid PREPARE EXECUTE RELEASE CLOSE state machine");
    }

    {
        ScriptedServer server([](int fd, std::string & error) {
            pp::Frame request;
            if (!receive_request(fd, pp::Opcode::hello, request, error) ||
                !send_response(fd, request, pp::Opcode::hello, hello_payload(48), error) ||
                !receive_request(fd, pp::Opcode::status, request, error) ||
                !send_response(fd, request, pp::Opcode::status, empty_status_payload(12, false), error) ||
                !receive_request(fd, pp::Opcode::prepare, request, error) ||
                !send_response(fd, request, pp::Opcode::prepare, prepared_payload(1), error) ||
                !receive_request(fd, pp::Opcode::release, request, error)) {
                return;
            }
            pp::Writer writer;
            writer.u64(14);
            send_response(fd, request, pp::Opcode::release, writer.data(), error);
        });
        pp::ClientSession client(options(server.port()));
        pp::HelloInfo hello;
        pp::StatusInfo status;
        pp::PreparedInfo prepared;
        std::string error;
        uint64_t next_generation = 999;
        const bool rejected = client.connect(hello, error) && client.status(status, error) &&
                              client.prepare(prepare_request(), prepared, error) &&
                              !client.release(1, next_generation, error) &&
                              error == "malformed RELEASE response" &&
                              next_generation == 999 && !client.connected();
        check(rejected && server.join(), "RELEASE requires an exact one-generation advance");
    }

    {
        ScriptedServer server([](int fd, std::string & error) {
            pp::Frame request;
            if (!receive_request(fd, pp::Opcode::hello, request, error) ||
                !send_response(fd, request, pp::Opcode::hello, hello_payload(55), error) ||
                !receive_request(fd, pp::Opcode::status, request, error)) {
                return;
            }
            pp::Header header;
            header.opcode = pp::Opcode::error;
            header.flags = pp::flag_response | pp::flag_error;
            header.request_id = request.header.request_id;
            header.command_seq = request.header.command_seq;
            header.session_epoch = request.header.session_epoch;
            header.route_epoch = request.header.route_epoch;
            header.residency_generation = request.header.residency_generation;
            pp::send_frame(fd, header,
                    pp::make_error_payload(pp::ErrorCode::invalid_argument, "status refused"), error);
        });
        pp::ClientSession client(options(server.port()));
        pp::HelloInfo hello;
        pp::StatusInfo status;
        std::string error;
        const bool rejected = client.connect(hello, error) && !client.status(status, error) &&
                              error.find("remote error 5: status refused") != std::string::npos;
        client.close(error);
        check(rejected && server.join(), "typed remote errors are preserved");
    }

    std::printf("client tests: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
