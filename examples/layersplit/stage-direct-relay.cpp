#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <vector>

namespace {

enum : int32_t {
    STAGE_STOP           = -1,
    STAGE_RESET          = -2,
    STAGE_DETACH         = -7,
    STAGE_V3_HELLO       = -8,
    STAGE_V3_BATCH       = -9,
    STAGE_V3_SEQ_REMOVE  = -10,
    STAGE_V3_STATUS      = -11,
    STAGE_V3_DRAIN       = -12,
    STAGE_V3_IDENTITY    = -13,
    STAGE_V3_RANGE_BATCH = -14,
};

enum : int32_t {
    STAGE_V3_MAGIC         = 0x4c535633,
    STAGE_V3_VERSION       = 3,
    STAGE_IDENTITY_MAGIC   = 0x4c534944,
    STAGE_IDENTITY_VERSION = 1,
};

enum : int32_t {
    STAGE_V3_CAP_BATCH      = 1 << 0,
    STAGE_V3_CAP_SEQ_REMOVE = 1 << 1,
    STAGE_V3_CAP_STATUS     = 1 << 2,
    STAGE_V3_CAP_DRAIN      = 1 << 3,
    STAGE_V3_CAP_TERMINAL   = 1 << 4,
    STAGE_V3_CAP_IDENTITY   = 1 << 5,
    STAGE_V3_CAP_RANGE      = 1 << 6,
};

constexpr int32_t REQUIRED_CAPABILITIES =
    STAGE_V3_CAP_BATCH |
    STAGE_V3_CAP_SEQ_REMOVE |
    STAGE_V3_CAP_STATUS |
    STAGE_V3_CAP_DRAIN |
    STAGE_V3_CAP_IDENTITY;
constexpr int32_t MAX_ROWS = 4096;
constexpr int32_t MAX_EMBEDDING = 65536;

struct Endpoint {
    std::string host;
    int port = 0;
};

struct Hello {
    int32_t layer_start = 0;
    int32_t layer_end = 0;
    int32_t n_layer = 0;
    int32_t n_embd = 0;
    int32_t max_streams = 0;
    int32_t n_ctx_seq = 0;
    int32_t n_batch = 0;
    int32_t n_ubatch = 0;
    int32_t capabilities = 0;
};

struct Identity {
    int32_t file_type = -1;
    std::array<uint8_t, 32> sha256 = {};
};

struct Status {
    int32_t code = -1;
    int32_t active_sequences = 0;
    int32_t max_streams = 0;
    int32_t draining = 0;
};

struct BatchFrame {
    int32_t n_rows = 0;
    int32_t hidden_width = 0;
    std::vector<int64_t> request_ids;
    std::vector<int64_t> route_epochs;
    std::vector<int32_t> seq_ids;
    std::vector<int32_t> positions;
    std::vector<int32_t> tokens;
    std::vector<float> hidden;
};

struct Counters {
    int64_t batches = 0;
    int64_t rows = 0;
    int64_t activation_bytes = 0;
};

bool valid_status(const Status & status) {
    return status.code == 0 &&
        status.active_sequences >= 0 &&
        status.max_streams > 0 &&
        status.active_sequences <= status.max_streams &&
        (status.draining == 0 || status.draining == 1);
}

bool send_all(int fd, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    while (size > 0) {
        const ssize_t sent = send(fd, bytes, size, MSG_NOSIGNAL);
        if (sent < 0 && errno == EINTR) {
            continue;
        }
        if (sent <= 0) {
            return false;
        }
        bytes += sent;
        size -= static_cast<size_t>(sent);
    }
    return true;
}

bool recv_all(int fd, void * data, size_t size) {
    auto * bytes = static_cast<uint8_t *>(data);
    while (size > 0) {
        const ssize_t received = recv(fd, bytes, size, 0);
        if (received < 0 && errno == EINTR) {
            continue;
        }
        if (received <= 0) {
            return false;
        }
        bytes += received;
        size -= static_cast<size_t>(received);
    }
    return true;
}

template <typename T>
bool send_vector(int fd, const std::vector<T> & values) {
    return values.empty() ||
        send_all(fd, values.data(), values.size() * sizeof(T));
}

template <typename T>
bool recv_vector(int fd, std::vector<T> & values) {
    return values.empty() ||
        recv_all(fd, values.data(), values.size() * sizeof(T));
}

void set_nodelay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

bool parse_port(const char * text, int & port) {
    char * end = nullptr;
    errno = 0;
    const long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value <= 0 || value > 65535) {
        return false;
    }
    port = static_cast<int>(value);
    return true;
}

bool parse_endpoint(const char * text, Endpoint & endpoint) {
    const std::string value(text);
    const size_t separator = value.rfind(':');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 == value.size()) {
        return false;
    }
    endpoint.host = value.substr(0, separator);
    return parse_port(value.c_str() + separator + 1, endpoint.port);
}

int connect_to(const Endpoint & endpoint) {
    addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo * addresses = nullptr;
    const std::string port = std::to_string(endpoint.port);
    if (getaddrinfo(
            endpoint.host.c_str(), port.c_str(), &hints, &addresses) != 0) {
        return -1;
    }
    int fd = -1;
    for (addrinfo * address = addresses; address != nullptr;
         address = address->ai_next) {
        fd = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (connect(fd, address->ai_addr, address->ai_addrlen) == 0) {
            set_nodelay(fd);
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(addresses);
    return fd;
}

int listen_on(int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0 ||
        listen(fd, 1) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

bool request_hello(int fd, Hello & hello) {
    const int32_t command = STAGE_V3_HELLO;
    int32_t words[11] = {};
    if (!send_all(fd, &command, sizeof(command)) ||
        !recv_all(fd, words, sizeof(words)) ||
        words[0] != STAGE_V3_MAGIC ||
        words[1] != STAGE_V3_VERSION) {
        return false;
    }
    hello = {
        words[2], words[3], words[4], words[5], words[6],
        words[7], words[8], words[9], words[10],
    };
    return hello.layer_start >= 0 &&
        hello.layer_start < hello.layer_end &&
        hello.layer_end <= hello.n_layer &&
        hello.n_embd > 0 &&
        hello.n_embd <= MAX_EMBEDDING &&
        hello.max_streams > 0 &&
        hello.max_streams <= MAX_ROWS &&
        hello.n_ctx_seq > 0 &&
        hello.n_batch > 0 &&
        hello.n_ubatch > 0 &&
        (hello.capabilities & REQUIRED_CAPABILITIES) == REQUIRED_CAPABILITIES;
}

bool request_identity(int fd, Identity & identity) {
    const int32_t command = STAGE_V3_IDENTITY;
    int32_t words[3] = {};
    if (!send_all(fd, &command, sizeof(command)) ||
        !recv_all(fd, words, sizeof(words)) ||
        words[0] != STAGE_IDENTITY_MAGIC ||
        words[1] != STAGE_IDENTITY_VERSION ||
        words[2] < 0 ||
        !recv_all(fd, identity.sha256.data(), identity.sha256.size())) {
        return false;
    }
    identity.file_type = words[2];
    return true;
}

bool request_status(int fd, int32_t command, Status & status) {
    const int32_t words[] = {command, STAGE_V3_VERSION};
    int32_t response[5] = {};
    if (!send_all(fd, words, sizeof(words)) ||
        !recv_all(fd, response, sizeof(response)) ||
        response[1] != STAGE_V3_VERSION) {
        return false;
    }
    status = {response[0], response[2], response[3], response[4]};
    return valid_status(status);
}

bool send_status(int fd, const Status & status, int32_t max_streams) {
    const int32_t words[] = {
        status.code,
        STAGE_V3_VERSION,
        status.active_sequences,
        max_streams,
        status.draining,
    };
    return send_all(fd, words, sizeof(words));
}

bool checked_elements(int32_t rows, int32_t width, size_t element_size, size_t & bytes) {
    if (rows <= 0 || rows > MAX_ROWS || width < 0 || width > MAX_EMBEDDING) {
        return false;
    }
    const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(width);
    if (width != 0 && count / static_cast<size_t>(width) != static_cast<size_t>(rows)) {
        return false;
    }
    if (count > std::numeric_limits<size_t>::max() / element_size) {
        return false;
    }
    bytes = count * element_size;
    return true;
}

bool read_batch_payload(
        int fd,
        int32_t n_rows,
        int32_t hidden_width,
        BatchFrame & frame) {
    size_t hidden_bytes = 0;
    if (!checked_elements(n_rows, hidden_width, sizeof(float), hidden_bytes)) {
        return false;
    }
    frame.n_rows = n_rows;
    frame.hidden_width = hidden_width;
    frame.request_ids.resize(static_cast<size_t>(n_rows));
    frame.route_epochs.resize(static_cast<size_t>(n_rows));
    frame.seq_ids.resize(static_cast<size_t>(n_rows));
    frame.positions.resize(static_cast<size_t>(n_rows));
    frame.tokens.resize(static_cast<size_t>(n_rows));
    frame.hidden.resize(hidden_bytes / sizeof(float));
    return recv_vector(fd, frame.request_ids) &&
        recv_vector(fd, frame.route_epochs) &&
        recv_vector(fd, frame.seq_ids) &&
        recv_vector(fd, frame.positions) &&
        recv_vector(fd, frame.tokens) &&
        recv_vector(fd, frame.hidden);
}

bool send_batch_request(int fd, const BatchFrame & frame) {
    const int32_t header[] = {
        STAGE_V3_BATCH,
        STAGE_V3_VERSION,
        frame.n_rows,
        frame.hidden_width,
    };
    return send_all(fd, header, sizeof(header)) &&
        send_vector(fd, frame.request_ids) &&
        send_vector(fd, frame.route_epochs) &&
        send_vector(fd, frame.seq_ids) &&
        send_vector(fd, frame.positions) &&
        send_vector(fd, frame.tokens) &&
        send_vector(fd, frame.hidden);
}

bool same_lineage(const BatchFrame & left, const BatchFrame & right) {
    return left.n_rows == right.n_rows &&
        left.request_ids == right.request_ids &&
        left.route_epochs == right.route_epochs &&
        left.seq_ids == right.seq_ids &&
        left.positions == right.positions;
}

bool receive_batch_response(
        int fd,
        int32_t expected_rows,
        int32_t expected_width,
        BatchFrame & frame) {
    int32_t header[3] = {};
    if (!recv_all(fd, header, sizeof(header)) ||
        header[0] != 0 ||
        header[1] != expected_rows ||
        header[2] != expected_width) {
        return false;
    }
    frame.n_rows = header[1];
    frame.hidden_width = header[2];
    frame.request_ids.resize(static_cast<size_t>(frame.n_rows));
    frame.route_epochs.resize(static_cast<size_t>(frame.n_rows));
    frame.seq_ids.resize(static_cast<size_t>(frame.n_rows));
    frame.positions.resize(static_cast<size_t>(frame.n_rows));
    frame.tokens.clear();
    frame.hidden.clear();
    if (!recv_vector(fd, frame.request_ids) ||
        !recv_vector(fd, frame.route_epochs) ||
        !recv_vector(fd, frame.seq_ids) ||
        !recv_vector(fd, frame.positions)) {
        return false;
    }
    if (expected_width == 0) {
        frame.tokens.resize(static_cast<size_t>(frame.n_rows));
        return recv_vector(fd, frame.tokens);
    }
    size_t hidden_bytes = 0;
    if (!checked_elements(frame.n_rows, expected_width, sizeof(float), hidden_bytes)) {
        return false;
    }
    frame.hidden.resize(hidden_bytes / sizeof(float));
    return recv_vector(fd, frame.hidden);
}

bool send_terminal_response(int fd, const BatchFrame & frame) {
    if (frame.hidden_width != 0 ||
        frame.tokens.size() != static_cast<size_t>(frame.n_rows)) {
        return false;
    }
    const int32_t header[] = {0, frame.n_rows, 0};
    return send_all(fd, header, sizeof(header)) &&
        send_vector(fd, frame.request_ids) &&
        send_vector(fd, frame.route_epochs) &&
        send_vector(fd, frame.seq_ids) &&
        send_vector(fd, frame.positions) &&
        send_vector(fd, frame.tokens);
}

bool forward_remove(
        int fd,
        int32_t seq_id,
        int64_t request_id,
        int64_t route_epoch,
        Status & status) {
    const int32_t header[] = {
        STAGE_V3_SEQ_REMOVE,
        STAGE_V3_VERSION,
        seq_id,
    };
    int32_t response[5] = {};
    return send_all(fd, header, sizeof(header)) &&
        send_all(fd, &request_id, sizeof(request_id)) &&
        send_all(fd, &route_epoch, sizeof(route_epoch)) &&
        recv_all(fd, response, sizeof(response)) &&
        response[0] == 0 &&
        response[1] == STAGE_V3_VERSION &&
        (status = {response[0], response[2], response[3], response[4]},
         valid_status(status));
}

bool forward_simple_with_ack(int fd, int32_t command) {
    int32_t response = -1;
    return send_all(fd, &command, sizeof(command)) &&
        recv_all(fd, &response, sizeof(response)) &&
        response == 0;
}

std::string digest_hex(const Identity & identity) {
    static const char digits[] = "0123456789abcdef";
    std::string result(identity.sha256.size() * 2, '0');
    for (size_t i = 0; i < identity.sha256.size(); ++i) {
        result[2 * i] = digits[identity.sha256[i] >> 4];
        result[2 * i + 1] = digits[identity.sha256[i] & 0x0f];
    }
    return result;
}

bool validate_chain(
        const Hello & head,
        const Hello & tail,
        const Identity & head_identity,
        const Identity & tail_identity) {
    return head.layer_start == 0 &&
        !(head.capabilities & STAGE_V3_CAP_TERMINAL) &&
        (tail.capabilities & STAGE_V3_CAP_TERMINAL) &&
        head.layer_end == tail.layer_start &&
        tail.layer_end == tail.n_layer &&
        head.n_layer == tail.n_layer &&
        head.n_embd == tail.n_embd &&
        head.max_streams == tail.max_streams &&
        head.n_ctx_seq == tail.n_ctx_seq &&
        head_identity.file_type == tail_identity.file_type &&
        head_identity.sha256 == tail_identity.sha256;
}

void emit_certificate(
        const Hello & head,
        const Identity & identity,
        const Endpoint & head_endpoint,
        const Endpoint & tail_endpoint,
        const Counters & counters,
        int rc) {
    const std::string digest = digest_hex(identity);
    fprintf(
        stderr,
        "DIRECTCERT {\"schema\":\"ls-stage-direct-relay-v1\","
        "\"status\":\"%s\",\"run_rc\":%d,"
        "\"head_endpoint\":\"%s:%d\",\"tail_endpoint\":\"%s:%d\","
        "\"layer_start\":%d,\"cut_layer\":%d,\"layer_end\":%d,"
        "\"n_layer\":%d,\"n_embd\":%d,\"file_type\":%d,"
        "\"model_sha256\":\"%s\",\"batches\":%lld,\"rows\":%lld,"
        "\"activation_payload_bytes\":%lld,"
        "\"host_activation_payload_bytes\":0}\n",
        rc == 0 ? "DIRECT_RELAY_OK" : "DIRECT_RELAY_ERROR",
        rc,
        head_endpoint.host.c_str(),
        head_endpoint.port,
        tail_endpoint.host.c_str(),
        tail_endpoint.port,
        head.layer_start,
        head.layer_end,
        head.n_layer,
        head.n_layer,
        head.n_embd,
        identity.file_type,
        digest.c_str(),
        static_cast<long long>(counters.batches),
        static_cast<long long>(counters.rows),
        static_cast<long long>(counters.activation_bytes));
}

int run_relay(
        int client,
        int head_fd,
        int tail_fd,
        const Hello & head,
        const Hello & tail,
        const Identity & identity,
        Counters & counters) {
    const int32_t max_streams = std::min(head.max_streams, tail.max_streams);
    while (true) {
        int32_t command = 0;
        if (!recv_all(client, &command, sizeof(command))) {
            fprintf(stderr, "error: direct relay host disconnected\n");
            return 3;
        }
        if (command == STAGE_V3_HELLO) {
            const int32_t words[] = {
                STAGE_V3_MAGIC,
                STAGE_V3_VERSION,
                head.layer_start,
                tail.layer_end,
                head.n_layer,
                head.n_embd,
                max_streams,
                std::min(head.n_ctx_seq, tail.n_ctx_seq),
                std::min(head.n_batch, tail.n_batch),
                std::min(head.n_ubatch, tail.n_ubatch),
                REQUIRED_CAPABILITIES | STAGE_V3_CAP_TERMINAL,
            };
            if (!send_all(client, words, sizeof(words))) {
                return 3;
            }
            continue;
        }
        if (command == STAGE_V3_IDENTITY) {
            const int32_t words[] = {
                STAGE_IDENTITY_MAGIC,
                STAGE_IDENTITY_VERSION,
                identity.file_type,
            };
            if (!send_all(client, words, sizeof(words)) ||
                !send_all(client, identity.sha256.data(), identity.sha256.size())) {
                return 3;
            }
            continue;
        }
        if (command == STAGE_V3_STATUS || command == STAGE_V3_DRAIN) {
            int32_t version = 0;
            if (!recv_all(client, &version, sizeof(version)) ||
                version != STAGE_V3_VERSION) {
                return 3;
            }
            Status head_status;
            Status tail_status;
            const bool ok = command == STAGE_V3_DRAIN ?
                request_status(tail_fd, command, tail_status) &&
                    request_status(head_fd, command, head_status) :
                request_status(head_fd, command, head_status) &&
                    request_status(tail_fd, command, tail_status);
            if (!ok ||
                head_status.max_streams != head.max_streams ||
                tail_status.max_streams != tail.max_streams ||
                head_status.active_sequences != tail_status.active_sequences ||
                head_status.draining != tail_status.draining) {
                fprintf(stderr, "error: direct relay status mismatch\n");
                return 3;
            }
            if (!send_status(client, head_status, max_streams)) {
                return 3;
            }
            continue;
        }
        if (command == STAGE_V3_SEQ_REMOVE) {
            int32_t version = 0;
            int32_t seq_id = -1;
            int64_t request_id = 0;
            int64_t route_epoch = 0;
            if (!recv_all(client, &version, sizeof(version)) ||
                !recv_all(client, &seq_id, sizeof(seq_id)) ||
                !recv_all(client, &request_id, sizeof(request_id)) ||
                !recv_all(client, &route_epoch, sizeof(route_epoch)) ||
                version != STAGE_V3_VERSION) {
                return 3;
            }
            Status tail_status;
            Status head_status;
            if (!forward_remove(
                    tail_fd, seq_id, request_id, route_epoch, tail_status) ||
                !forward_remove(
                    head_fd, seq_id, request_id, route_epoch, head_status) ||
                head_status.max_streams != head.max_streams ||
                tail_status.max_streams != tail.max_streams ||
                head_status.active_sequences != tail_status.active_sequences ||
                head_status.draining != tail_status.draining ||
                !send_status(client, head_status, max_streams)) {
                fprintf(stderr, "error: direct relay remove mismatch\n");
                return 3;
            }
            continue;
        }
        if (command == STAGE_RESET || command == STAGE_DETACH) {
            if (!forward_simple_with_ack(tail_fd, command) ||
                !forward_simple_with_ack(head_fd, command)) {
                return 3;
            }
            const int32_t status = 0;
            if (!send_all(client, &status, sizeof(status))) {
                return 3;
            }
            if (command == STAGE_DETACH) {
                return 0;
            }
            continue;
        }
        if (command == STAGE_STOP) {
            if (!send_all(tail_fd, &command, sizeof(command)) ||
                !send_all(head_fd, &command, sizeof(command))) {
                return 3;
            }
            return 0;
        }
        if (command == STAGE_V3_RANGE_BATCH) {
            fprintf(stderr, "error: direct relay does not support range batches\n");
            return 3;
        }
        if (command != STAGE_V3_BATCH) {
            fprintf(stderr, "error: direct relay unknown command %d\n", command);
            return 3;
        }

        int32_t header[3] = {};
        if (!recv_all(client, header, sizeof(header)) ||
            header[0] != STAGE_V3_VERSION ||
            header[1] <= 0 ||
            header[1] > std::min({
                head.n_batch,
                tail.n_batch,
                head.n_ubatch,
                tail.n_ubatch,
            }) ||
            header[2] != 0) {
            fprintf(stderr, "error: invalid direct relay batch header\n");
            return 3;
        }
        BatchFrame input;
        if (!read_batch_payload(client, header[1], header[2], input) ||
            !send_batch_request(head_fd, input)) {
            return 3;
        }
        BatchFrame activation;
        if (!receive_batch_response(
                head_fd, input.n_rows, head.n_embd, activation) ||
            !same_lineage(input, activation)) {
            fprintf(stderr, "error: direct relay head lineage mismatch\n");
            return 3;
        }
        activation.tokens = input.tokens;
        if (!send_batch_request(tail_fd, activation)) {
            return 3;
        }
        BatchFrame terminal;
        if (!receive_batch_response(tail_fd, input.n_rows, 0, terminal) ||
            !same_lineage(input, terminal) ||
            !send_terminal_response(client, terminal)) {
            fprintf(stderr, "error: direct relay tail lineage mismatch\n");
            return 3;
        }
        counters.batches += 1;
        counters.rows += input.n_rows;
        counters.activation_bytes +=
            static_cast<int64_t>(input.n_rows) *
            static_cast<int64_t>(head.n_embd) *
            static_cast<int64_t>(sizeof(float));
    }
}

void usage(const char * program) {
    fprintf(
        stderr,
        "usage: %s --listen PORT --head HOST:PORT --tail HOST:PORT\n",
        program);
}

} // namespace

int main(int argc, char ** argv) {
    int listen_port = 0;
    Endpoint head_endpoint;
    Endpoint tail_endpoint;
    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--listen") == 0 && index + 1 < argc) {
            if (!parse_port(argv[++index], listen_port)) {
                fprintf(stderr, "error: invalid --listen\n");
                return 1;
            }
        } else if (strcmp(argv[index], "--head") == 0 && index + 1 < argc) {
            if (!parse_endpoint(argv[++index], head_endpoint)) {
                fprintf(stderr, "error: invalid --head\n");
                return 1;
            }
        } else if (strcmp(argv[index], "--tail") == 0 && index + 1 < argc) {
            if (!parse_endpoint(argv[++index], tail_endpoint)) {
                fprintf(stderr, "error: invalid --tail\n");
                return 1;
            }
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (listen_port == 0 || head_endpoint.port == 0 || tail_endpoint.port == 0) {
        usage(argv[0]);
        return 1;
    }

    const int head_fd = connect_to(head_endpoint);
    if (head_fd < 0) {
        fprintf(stderr, "error: cannot connect head %s:%d\n",
                head_endpoint.host.c_str(), head_endpoint.port);
        return 2;
    }
    const int tail_fd = connect_to(tail_endpoint);
    if (tail_fd < 0) {
        fprintf(stderr, "error: cannot connect tail %s:%d\n",
                tail_endpoint.host.c_str(), tail_endpoint.port);
        close(head_fd);
        return 2;
    }

    Hello head;
    Hello tail;
    Identity head_identity;
    Identity tail_identity;
    if (!request_hello(head_fd, head) ||
        !request_hello(tail_fd, tail) ||
        !request_identity(head_fd, head_identity) ||
        !request_identity(tail_fd, tail_identity) ||
        !validate_chain(head, tail, head_identity, tail_identity)) {
        fprintf(stderr, "error: incompatible direct relay chain\n");
        close(tail_fd);
        close(head_fd);
        return 2;
    }

    const int server = listen_on(listen_port);
    if (server < 0) {
        fprintf(stderr, "error: cannot listen on port %d: %s\n",
                listen_port, strerror(errno));
        close(tail_fd);
        close(head_fd);
        return 2;
    }
    fprintf(
        stderr,
        "[direct-relay] listening on 0.0.0.0:%d head=%s:%d tail=%s:%d\n",
        listen_port,
        head_endpoint.host.c_str(),
        head_endpoint.port,
        tail_endpoint.host.c_str(),
        tail_endpoint.port);
    const int client = accept(server, nullptr, nullptr);
    if (client < 0) {
        fprintf(stderr, "error: direct relay accept: %s\n", strerror(errno));
        close(server);
        close(tail_fd);
        close(head_fd);
        return 2;
    }
    set_nodelay(client);
    Counters counters;
    const int rc = run_relay(
        client, head_fd, tail_fd, head, tail, head_identity, counters);
    emit_certificate(
        head, head_identity, head_endpoint, tail_endpoint, counters, rc);
    close(client);
    close(server);
    close(tail_fd);
    close(head_fd);
    return rc;
}
