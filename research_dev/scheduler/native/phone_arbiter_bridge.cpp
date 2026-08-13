#define FFN_DMABUF_BRIDGE_NO_MAIN
#include "../../spikes/s41_gemma_qwen_continuous_baseline/tp_operator_split_v1/ffs_dmabuf_transport_v1/ffn_dmabuf_bridge.cpp"

#include <array>
#include <limits>
#include <map>
#include <poll.h>

namespace {

static constexpr int protected_client_index = 0;
static constexpr int filler_client_index = 1;

struct arbiter_client {
    const char * name = nullptr;
    int listener = -1;
    int fd = -1;
    bool connected = false;
    bool initialized = false;
    ffn_split::hello_request hello = {};
    ffn_split::hello_response response = {};
    uint64_t calls = 0;
};

struct completed_phone_call {
    ffn_split::execute_response response = {};
    std::vector<unsigned char> payload;
};

uint64_t arbiter_now_us() {
    return static_cast<uint64_t>(now_ns() / 1000);
}

uint64_t parse_positive_us(const char * text, const char * name) {
    errno = 0;
    char * end = nullptr;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<uint64_t>(value);
}

uint64_t parse_nonnegative_us(const char * text, const char * name) {
    errno = 0;
    char * end = nullptr;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<uint64_t>(value);
}

int parse_layer_bound(const char * text, const char * name) {
    errno = 0;
    char * end = nullptr;
    const long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < 0 || value >= 64) {
        throw std::runtime_error(std::string("invalid ") + name);
    }
    return static_cast<int>(value);
}

int accept_client(int listener) {
    int fd;
    do {
        fd = accept(listener, nullptr, nullptr);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        throw std::runtime_error(std::string("TCP accept failed: ") +
                strerror(errno));
    }
    const int one = 1;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0) {
        close(fd);
        throw std::runtime_error(std::string("TCP_NODELAY failed: ") +
                strerror(errno));
    }
    return fd;
}

bool valid_phone_hello(
        const ffn_split::hello_request & request,
        const ffn_split::hello_response & response) {
    return request.magic == ffn_split::protocol_magic &&
            request.version == ffn_split::protocol_version &&
            request.message == static_cast<uint16_t>(
                    ffn_split::message_type::hello_request) &&
            request.layer_mask != 0 && request.n_embd != 0 &&
            request.max_columns != 0 && request.max_tokens != 0 &&
            response.magic == ffn_split::protocol_magic &&
            response.version == ffn_split::protocol_version &&
            response.message == static_cast<uint16_t>(
                    ffn_split::message_type::hello_response) &&
            valid_hello_response(request, response);
}

bool valid_local_request(
        const arbiter_client & client,
        const ffn_split::execute_request & request,
        size_t max_payload_bytes) {
    const size_t element_size =
            client.hello.flags & ffn_split::flag_f16_io ?
                    sizeof(uint16_t) : sizeof(float);
    return request.magic == ffn_split::protocol_magic &&
            request.version == ffn_split::protocol_version &&
            request.message == static_cast<uint16_t>(
                    ffn_split::message_type::execute_request) &&
            request.request_id != 0 && request.layer >= 0 &&
            request.layer < 64 &&
            (client.response.layer_mask &
                    (UINT64_C(1) << request.layer)) != 0 &&
            request.tokens > 0 &&
            request.tokens <= client.response.max_tokens &&
            request.elements ==
                    static_cast<uint64_t>(client.response.n_embd) *
                            request.tokens &&
            request.payload_bytes == request.elements * element_size &&
            request.payload_bytes <= max_payload_bytes &&
            request.columns > 0 &&
            request.columns <= client.response.max_columns &&
            (request.columns == client.response.max_columns ||
             request.columns == static_cast<uint32_t>(
                     client.response.alternate_columns_32) * 32 ||
             request.columns % client.response.column_quantum == 0);
}

bool valid_phone_response(
        const ffn_split::execute_request & request,
        const ffn_split::execute_response & response,
        const unsigned char * payload) {
    return response.magic == ffn_split::protocol_magic &&
            response.version == ffn_split::protocol_version &&
            response.message == static_cast<uint16_t>(
                    ffn_split::message_type::execute_response) &&
            response.status == 0 &&
            response.request_id == request.request_id &&
            response.layer == request.layer &&
            response.elements == request.elements &&
            response.payload_bytes == request.payload_bytes &&
            response.columns == request.columns &&
            response.tokens == request.tokens &&
            response.payload_hash == ffn_split::hash_bytes(
                    payload, request.payload_bytes);
}

bool socket_ready(int fd) {
    pollfd item = {};
    item.fd = fd;
    item.events = POLLIN;
    int status;
    do {
        status = poll(&item, 1, 0);
    } while (status < 0 && errno == EINTR);
    return status > 0 &&
            (item.revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL)) != 0;
}

double minimum(const std::vector<double> & values) {
    return values.empty() ? 0.0 :
            *std::min_element(values.begin(), values.end());
}

double maximum(const std::vector<double> & values) {
    return values.empty() ? 0.0 :
            *std::max_element(values.begin(), values.end());
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 10) {
        fprintf(stderr,
                "usage: %s <bind-address> <protected-port> <filler-port> "
                "<malloc-split|devmem-split> <protected-first-layer> "
                "<protected-last-layer> <idle-lower-us> "
                "<filler-sandwich-upper-us> <guard-us>\n",
                argv[0]);
        return 2;
    }

    try {
        const int protected_port = parse_port(argv[2]);
        const int filler_port = parse_port(argv[3]);
        if (protected_port == filler_port) {
            throw std::runtime_error("arbiter client ports must differ");
        }
        const std::string allocator = argv[4];
        if (allocator != "malloc-split" && allocator != "devmem-split") {
            throw std::runtime_error(
                    "phone arbiter requires a split host allocator");
        }
        const bool persistent = allocator == "devmem-split";
        const int protected_first_layer = parse_layer_bound(
                argv[5], "protected first layer");
        const int protected_last_layer = parse_layer_bound(
                argv[6], "protected last layer");
        if (protected_first_layer > protected_last_layer) {
            throw std::runtime_error("protected layer group is invalid");
        }
        const uint64_t idle_lower_us = parse_positive_us(
                argv[7], "phone idle lower bound");
        const uint64_t filler_upper_us = parse_positive_us(
                argv[8], "filler sandwich upper bound");
        const uint64_t guard_us = parse_nonnegative_us(
                argv[9], "phone guard");
        if (filler_upper_us >
                std::numeric_limits<uint64_t>::max() - guard_us ||
            filler_upper_us + guard_us > idle_lower_us) {
            throw std::runtime_error(
                    "filler upper bound and guard do not fit the idle bound");
        }

        signal(SIGPIPE, SIG_IGN);
        prefetch_fence_client prefetch_fence;
        std::string protected_done_file;
        if (const char * value = getenv("S42_PHONE_PROTECTED_DONE_FILE")) {
            if (*value == '\0' || *value != '/') {
                throw std::runtime_error(
                        "protected completion receipt path is invalid");
            }
            protected_done_file = value;
        }
        const auto protected_is_done = [&]() {
            return !protected_done_file.empty() &&
                    access(protected_done_file.c_str(), F_OK) == 0;
        };
        bool defer_filler_until_protected_done = false;
        if (const char * value = getenv(
                    "S42_PHONE_DEFER_FILLER_UNTIL_PROTECTED_DONE")) {
            if (strcmp(value, "0") == 0) {
                defer_filler_until_protected_done = false;
            } else if (strcmp(value, "1") == 0) {
                defer_filler_until_protected_done = true;
            } else {
                throw std::runtime_error(
                        "invalid filler protected-done fence");
            }
        }
        if (defer_filler_until_protected_done &&
            protected_done_file.empty()) {
            throw std::runtime_error(
                    "filler protected-done fence has no receipt path");
        }
        libusb_context * usb_context = nullptr;
        const int usb_status = libusb_init(&usb_context);
        if (usb_status != 0) {
            throw std::runtime_error(std::string("libusb_init failed: ") +
                    libusb_error_name(usb_status));
        }
        libusb_device_handle * handle = open_claimed_device(
                usb_context, recovery_timeout_ms);

        std::array<arbiter_client, 2> clients = {{
            {"protected"},
            {"filler"},
        }};
        clients[protected_client_index].listener = open_listener(
                argv[1], protected_port);
        clients[filler_client_index].listener = open_listener(
                argv[1], filler_port);
        fprintf(stderr,
                "[phone-arbiter-bridge] ready protected=%s:%d "
                "filler=%s:%d allocator=%s idle_lower_us=%llu "
                "filler_upper_us=%llu guard_us=%llu "
                "defer_filler_until_protected_done=%s\n",
                argv[1], protected_port, argv[1], filler_port,
                allocator.c_str(),
                static_cast<unsigned long long>(idle_lower_us),
                static_cast<unsigned long long>(filler_upper_us),
                static_cast<unsigned long long>(guard_us),
                defer_filler_until_protected_done ? "true" : "false");
        fflush(stderr);

        int active_client = -1;
        uint64_t session_switches = 0;
        std::vector<double> switch_samples;

        const auto send_shutdown = [&](bool terminate) {
            ffn_split::execute_request shutdown = {};
            shutdown.magic = ffn_split::protocol_magic;
            shutdown.version = ffn_split::protocol_version;
            shutdown.message = static_cast<uint16_t>(
                    ffn_split::message_type::execute_request);
            shutdown.layer = terminate ? -1 : 0;
            transfer_exact(handle, ffn_split::dmabuf_out_endpoint,
                    reinterpret_cast<unsigned char *>(&shutdown),
                    sizeof(shutdown));
        };

        const auto arm_direct = [&](int index) {
            arbiter_client & client = clients[index];
            ffn_split::hello_response response = {};
            transfer_exact(handle, ffn_split::dmabuf_out_endpoint,
                    reinterpret_cast<unsigned char *>(&client.hello),
                    sizeof(client.hello));
            transfer_exact(handle, ffn_split::dmabuf_in_endpoint,
                    reinterpret_cast<unsigned char *>(&response),
                    sizeof(response));
            if (!valid_phone_hello(client.hello, response)) {
                throw std::runtime_error(std::string("phone rejected ") +
                        client.name + " HELLO");
            }
            if (client.initialized &&
                memcmp(&response, &client.response, sizeof(response)) != 0) {
                throw std::runtime_error(std::string(client.name) +
                        " phone identity changed");
            }
            client.response = response;
            client.initialized = true;
            active_client = index;
        };

        const auto switch_to = [&](int index) {
            if (active_client == index) {
                return;
            }
            const uint64_t started_us = arbiter_now_us();
            if (active_client >= 0) {
                send_shutdown(false);
            }
            arm_direct(index);
            switch_samples.push_back(static_cast<double>(
                    arbiter_now_us() - started_us));
            ++session_switches;
        };

        for (int index : {protected_client_index, filler_client_index}) {
            arbiter_client & client = clients[index];
            client.fd = accept_client(client.listener);
            client.connected = true;
            if (!receive_exact(client.fd, &client.hello,
                        sizeof(client.hello))) {
                throw std::runtime_error(std::string(client.name) +
                        " local HELLO read failed");
            }
            switch_to(index);
            if (!send_exact(client.fd, &client.response,
                        sizeof(client.response))) {
                throw std::runtime_error(std::string(client.name) +
                        " local HELLO write failed");
            }
            fprintf(stderr,
                    "[phone-arbiter-bridge] client=%s armed "
                    "n_embd=%u columns=%u layers=%016llx hash=%016llx\n",
                    client.name, client.response.n_embd,
                    client.response.max_columns,
                    static_cast<unsigned long long>(
                            client.response.layer_mask),
                    static_cast<unsigned long long>(
                            client.response.weight_hash));
            fflush(stderr);
        }
        switch_to(protected_client_index);

        size_t max_payload_bytes = 0;
        for (const arbiter_client & client : clients) {
            const size_t element_size =
                    client.hello.flags & ffn_split::flag_f16_io ?
                            sizeof(uint16_t) : sizeof(float);
            const size_t payload_bytes =
                    static_cast<size_t>(client.response.n_embd) *
                    client.response.max_tokens * element_size;
            max_payload_bytes = std::max(max_payload_bytes, payload_bytes);
        }
        const size_t max_wire_bytes =
                ffn_split::dmabuf_payload_offset + max_payload_bytes;
        std::unique_ptr<transfer_buffer> request_buffer =
                std::make_unique<transfer_buffer>(
                        handle, max_wire_bytes, persistent);
        std::unique_ptr<transfer_buffer> response_buffer =
                std::make_unique<transfer_buffer>(
                        handle, max_wire_bytes, persistent);

        uint64_t reset_recoveries = 0;
        const auto reconnect = [&](int desired_client) {
            response_buffer.reset();
            request_buffer.reset();
            close_claimed_device(handle);
            handle = nullptr;
            handle = open_claimed_device(usb_context, recovery_timeout_ms);
            request_buffer = std::make_unique<transfer_buffer>(
                    handle, max_wire_bytes, persistent);
            response_buffer = std::make_unique<transfer_buffer>(
                    handle, max_wire_bytes, persistent);
            active_client = -1;
            arm_direct(desired_client);
        };

        const auto recover_to = [&](int desired_client) {
            if (reset_recoveries >= maximum_reset_recoveries) {
                throw std::runtime_error(
                        "phone arbiter exceeded USB reset recovery bound");
            }
            ++reset_recoveries;
            fprintf(stderr,
                    "[phone-arbiter-bridge] USB recovery=%llu "
                    "desired=%s\n",
                    static_cast<unsigned long long>(reset_recoveries),
                    clients[desired_client].name);
            fflush(stderr);
            reconnect(desired_client);
        };

        const auto ensure_session = [&](int index) {
            for (;;) {
                try {
                    switch_to(index);
                    return;
                } catch (const std::exception & error) {
                    fprintf(stderr,
                            "[phone-arbiter-bridge] session switch failed "
                            "target=%s error=%s\n",
                            clients[index].name, error.what());
                    fflush(stderr);
                    recover_to(index);
                }
            }
        };

        const auto read_request = [&](int index,
                                      ffn_split::execute_request & request) {
            arbiter_client & client = clients[index];
            if (!receive_exact(client.fd, &request, sizeof(request))) {
                return false;
            }
            if (!valid_local_request(
                        client, request, max_payload_bytes)) {
                throw std::runtime_error(std::string("invalid ") +
                        client.name + " local execute request");
            }
            memset(request_buffer->data(), 0,
                    ffn_split::dmabuf_payload_offset);
            memcpy(request_buffer->data(), &request, sizeof(request));
            unsigned char * payload = request_buffer->data() +
                    ffn_split::dmabuf_payload_offset;
            if (!receive_exact(client.fd, payload, request.payload_bytes)) {
                throw std::runtime_error(std::string(client.name) +
                        " local execute payload read failed");
            }
            if (request.payload_hash != ffn_split::hash_bytes(
                        payload, request.payload_bytes)) {
                throw std::runtime_error(std::string(client.name) +
                        " local execute payload hash mismatch");
            }
            return true;
        };

        const auto execute_phone = [&](int index,
                                       const ffn_split::execute_request & request) {
            const size_t payload_bytes = request.payload_bytes;
            const size_t wire_bytes =
                    ffn_split::dmabuf_payload_offset + payload_bytes;
            const std::vector<unsigned char> replay(
                    request_buffer->data(),
                    request_buffer->data() + wire_bytes);
            for (;;) {
                try {
                    ensure_session(index);
                    transfer_exact(handle, ffn_split::dmabuf_out_endpoint,
                            reinterpret_cast<unsigned char *>(
                                    const_cast<ffn_split::execute_request *>(
                                            &request)),
                            sizeof(request));
                    uint32_t payload_ready = 0;
                    transfer_exact(handle, ffn_split::dmabuf_in_endpoint,
                            reinterpret_cast<unsigned char *>(&payload_ready),
                            sizeof(payload_ready));
                    if (payload_ready !=
                            (ffn_split::dmabuf_payload_ready_magic ^
                             request.request_id)) {
                        throw std::runtime_error(
                                "invalid phone payload-ready response");
                    }
                    transfer_exact(handle, ffn_split::dmabuf_out_endpoint,
                            request_buffer->data() +
                                    ffn_split::dmabuf_payload_offset,
                            payload_bytes);
                    transfer_exact(handle, ffn_split::dmabuf_in_endpoint,
                            response_buffer->data(), wire_bytes);
                    completed_phone_call completed;
                    memcpy(&completed.response, response_buffer->data(),
                            sizeof(completed.response));
                    const unsigned char * output = response_buffer->data() +
                            ffn_split::dmabuf_payload_offset;
                    if (!valid_phone_response(
                                request, completed.response, output)) {
                        throw std::runtime_error(
                                "invalid phone execute response");
                    }
                    completed.payload.assign(
                            output, output + payload_bytes);
                    return completed;
                } catch (const std::exception & error) {
                    fprintf(stderr,
                            "[phone-arbiter-bridge] execute recovery "
                            "client=%s request_id=%u error=%s\n",
                            clients[index].name, request.request_id,
                            error.what());
                    fflush(stderr);
                    recover_to(index);
                    memcpy(request_buffer->data(), replay.data(), wire_bytes);
                }
            }
        };

        const auto send_completed = [&](int index,
                                        const completed_phone_call & completed) {
            return send_exact(clients[index].fd, &completed.response,
                            sizeof(completed.response)) &&
                    send_exact(clients[index].fd, completed.payload.data(),
                            completed.payload.size());
        };

        uint64_t protected_group_starts = 0;
        uint64_t protected_group_ends = 0;
        uint64_t idle_opened_us = 0;
        uint64_t filler_admitted = 0;
        uint64_t filler_deferred = 0;
        uint64_t filler_before_protected_done = 0;
        uint64_t filler_after_protected_done = 0;
        uint64_t filler_upper_violations = 0;
        uint64_t filler_upper_violations_before_protected_done = 0;
        uint64_t filler_upper_violations_after_protected_done = 0;
        uint64_t guard_violations = 0;
        uint64_t idle_lower_violations = 0;
        uint64_t protected_pending_after_filler = 0;
        bool protected_done_observed = false;
        bool deferred_in_window = false;
        std::vector<double> protected_rpc_samples;
        std::vector<double> filler_sandwich_samples;
        std::vector<double> filler_before_protected_done_samples;
        std::vector<double> filler_after_protected_done_samples;
        std::map<uint32_t, std::vector<double>> filler_shape_samples;
        std::vector<double> observed_idle_samples;

        const auto close_client = [&](int index) {
            arbiter_client & client = clients[index];
            if (client.fd >= 0) {
                close(client.fd);
                client.fd = -1;
            }
            client.connected = false;
            fprintf(stderr,
                    "[phone-arbiter-bridge] client=%s disconnected\n",
                    client.name);
            fflush(stderr);
        };

        while (clients[protected_client_index].connected ||
               clients[filler_client_index].connected) {
            std::array<pollfd, 2> items = {};
            for (int index = 0; index < 2; ++index) {
                items[index].fd = clients[index].connected ?
                        clients[index].fd : -1;
                items[index].events = POLLIN;
            }
            int poll_status;
            do {
                poll_status = poll(items.data(), items.size(), -1);
            } while (poll_status < 0 && errno == EINTR);
            if (poll_status < 0) {
                throw std::runtime_error(std::string("arbiter poll failed: ") +
                        strerror(errno));
            }

            int selected = -1;
            const short protected_events =
                    items[protected_client_index].revents;
            const short filler_events = items[filler_client_index].revents;
            if (protected_events &
                    (POLLIN | POLLHUP | POLLERR | POLLNVAL)) {
                selected = protected_client_index;
            } else if (
                    (filler_events & (POLLHUP | POLLERR | POLLNVAL)) &&
                    !(filler_events & POLLIN)) {
                selected = filler_client_index;
            } else if (filler_events & POLLIN) {
                const uint64_t current_us = arbiter_now_us();
                const bool protected_done =
                        !clients[protected_client_index].connected ||
                        protected_is_done();
                protected_done_observed =
                        protected_done_observed || protected_is_done();
                const bool filler_fits = idle_opened_us != 0 &&
                        current_us <= idle_opened_us + idle_lower_us &&
                        filler_upper_us + guard_us <=
                                idle_opened_us + idle_lower_us - current_us;
                if (protected_done ||
                    (!defer_filler_until_protected_done && filler_fits)) {
                    selected = filler_client_index;
                } else {
                    if (!deferred_in_window) {
                        ++filler_deferred;
                        deferred_in_window = true;
                    }
                    pollfd protected_item = {};
                    protected_item.fd = clients[protected_client_index].fd;
                    protected_item.events = POLLIN;
                    do {
                        poll_status = poll(&protected_item, 1, 10);
                    } while (poll_status < 0 && errno == EINTR);
                    if (poll_status < 0) {
                        throw std::runtime_error(
                                "protected wait after filler deferral failed");
                    }
                    continue;
                }
            }
            if (selected < 0) {
                continue;
            }

            const uint64_t request_arrival_us = arbiter_now_us();
            ffn_split::execute_request request = {};
            if (!read_request(selected, request)) {
                close_client(selected);
                if (selected == protected_client_index) {
                    idle_opened_us = 0;
                }
                continue;
            }

            if (selected == protected_client_index) {
                if (idle_opened_us != 0) {
                    const uint64_t observed_idle_us =
                            request_arrival_us >= idle_opened_us ?
                                    request_arrival_us - idle_opened_us : 0;
                    observed_idle_samples.push_back(
                            static_cast<double>(observed_idle_us));
                    if (observed_idle_us < idle_lower_us) {
                        ++idle_lower_violations;
                    }
                    idle_opened_us = 0;
                }
                deferred_in_window = false;
                if (request.layer == protected_first_layer) {
                    ++protected_group_starts;
                }
                const uint64_t started_us = arbiter_now_us();
                prefetch_fence.begin(request);
                completed_phone_call completed = execute_phone(
                        protected_client_index, request);
                prefetch_fence.finish(request, now_ms());
                const uint64_t completed_us = arbiter_now_us();
                protected_rpc_samples.push_back(static_cast<double>(
                        completed_us - started_us));
                ++clients[protected_client_index].calls;
                if (!send_completed(protected_client_index, completed)) {
                    close_client(protected_client_index);
                    idle_opened_us = 0;
                    continue;
                }
                if (request.layer == protected_last_layer) {
                    ++protected_group_ends;
                    idle_opened_us = arbiter_now_us();
                }
                continue;
            }

            const uint64_t filler_started_us = request_arrival_us;
            const bool protected_done_at_start = protected_is_done();
            const uint64_t protected_deadline_us =
                    clients[protected_client_index].connected &&
                            !protected_is_done() &&
                            idle_opened_us != 0 ?
                            idle_opened_us + idle_lower_us : 0;
            completed_phone_call completed = execute_phone(
                    filler_client_index, request);
            protected_done_observed =
                    protected_done_observed || protected_is_done();
            if (clients[protected_client_index].connected &&
                !protected_done_observed) {
                ensure_session(protected_client_index);
            }
            const uint64_t filler_completed_us = arbiter_now_us();
            const uint64_t filler_duration_us =
                    filler_completed_us - filler_started_us;
            filler_sandwich_samples.push_back(
                    static_cast<double>(filler_duration_us));
            filler_shape_samples[request.tokens].push_back(
                    static_cast<double>(filler_duration_us));
            ++filler_admitted;
            ++clients[filler_client_index].calls;
            if (protected_done_at_start) {
                ++filler_after_protected_done;
                filler_after_protected_done_samples.push_back(
                        static_cast<double>(filler_duration_us));
            } else {
                ++filler_before_protected_done;
                filler_before_protected_done_samples.push_back(
                        static_cast<double>(filler_duration_us));
            }
            if (filler_duration_us > filler_upper_us) {
                ++filler_upper_violations;
                if (protected_done_at_start) {
                    ++filler_upper_violations_after_protected_done;
                } else {
                    ++filler_upper_violations_before_protected_done;
                }
            }
            if (protected_deadline_us != 0 &&
                (filler_completed_us > protected_deadline_us ||
                 guard_us > protected_deadline_us - filler_completed_us)) {
                ++guard_violations;
            }
            if (clients[protected_client_index].connected &&
                !protected_done_observed &&
                socket_ready(clients[protected_client_index].fd)) {
                ++protected_pending_after_filler;
            }
            if (!send_completed(filler_client_index, completed)) {
                close_client(filler_client_index);
            }
        }

        prefetch_fence.require_complete();
        if (active_client >= 0) {
            send_shutdown(true);
        }
        for (const auto & shape : filler_shape_samples) {
            fprintf(stderr,
                    "PHONEARBITERSHAPE {\"status\":\"PASS\","
                    "\"energy_claim_eligible\":false,"
                    "\"tokens\":%u,\"calls\":%zu,"
                    "\"sandwich_p50_us\":%.0f,"
                    "\"sandwich_max_us\":%.0f}\n",
                    shape.first, shape.second.size(),
                    percentile(shape.second, 0.50),
                    maximum(shape.second));
        }
        fprintf(stderr,
                "PHONEARBITER {\"status\":\"MECHANICS_ONLY\","
                "\"energy_claim_eligible\":false,"
                "\"allocator\":\"%s\",\"max_wire_bytes\":%zu,"
                "\"protected_calls\":%llu,\"filler_calls\":%llu,"
                "\"protected_group_starts\":%llu,"
                "\"protected_group_ends\":%llu,"
                "\"filler_admitted\":%llu,\"filler_deferred\":%llu,"
                "\"defer_filler_until_protected_done\":%s,"
                "\"filler_before_protected_done\":%llu,"
                "\"filler_after_protected_done\":%llu,"
                "\"session_switches\":%llu,\"reset_recoveries\":%llu,"
                "\"prefetch_fence_enabled\":%s,"
                "\"prefetch_fence_calls\":%llu,"
                "\"prefetch_group_first_layer\":%d,"
                "\"prefetch_group_last_layer\":%d,"
                "\"prefetch_decode_window_min_ms\":%.6f,"
                "\"prefetch_overrun_max_ms\":%.6f,"
                "\"idle_lower_us\":%llu,\"filler_upper_us\":%llu,"
                "\"guard_us\":%llu,\"idle_samples\":%zu,"
                "\"observed_idle_min_us\":%.0f,"
                "\"observed_idle_p10_us\":%.0f,"
                "\"observed_idle_p50_us\":%.0f,"
                "\"protected_rpc_p50_us\":%.0f,"
                "\"filler_sandwich_p50_us\":%.0f,"
                "\"filler_sandwich_p90_us\":%.0f,"
                "\"filler_sandwich_max_us\":%.0f,"
                "\"filler_before_protected_done_max_us\":%.0f,"
                "\"filler_after_protected_done_max_us\":%.0f,"
                "\"switch_p50_us\":%.0f,\"switch_p90_us\":%.0f,"
                "\"protected_done_observed\":%s,"
                "\"filler_upper_violations\":%llu,"
                "\"filler_upper_violations_before_protected_done\":%llu,"
                "\"filler_upper_violations_after_protected_done\":%llu,"
                "\"guard_violations\":%llu,"
                "\"idle_lower_violations\":%llu,"
                "\"protected_pending_after_filler\":%llu}\n",
                allocator.c_str(), max_wire_bytes,
                static_cast<unsigned long long>(
                        clients[protected_client_index].calls),
                static_cast<unsigned long long>(
                        clients[filler_client_index].calls),
                static_cast<unsigned long long>(protected_group_starts),
                static_cast<unsigned long long>(protected_group_ends),
                static_cast<unsigned long long>(filler_admitted),
                static_cast<unsigned long long>(filler_deferred),
                defer_filler_until_protected_done ? "true" : "false",
                static_cast<unsigned long long>(
                        filler_before_protected_done),
                static_cast<unsigned long long>(
                        filler_after_protected_done),
                static_cast<unsigned long long>(session_switches),
                static_cast<unsigned long long>(reset_recoveries),
                prefetch_fence.configured() ? "true" : "false",
                static_cast<unsigned long long>(prefetch_fence.calls()),
                prefetch_fence.group_first_layer(),
                prefetch_fence.group_last_layer(),
                prefetch_fence.decode_window_min_ms(),
                prefetch_fence.overrun_max_ms(),
                static_cast<unsigned long long>(idle_lower_us),
                static_cast<unsigned long long>(filler_upper_us),
                static_cast<unsigned long long>(guard_us),
                observed_idle_samples.size(),
                minimum(observed_idle_samples),
                percentile(observed_idle_samples, 0.10),
                percentile(observed_idle_samples, 0.50),
                percentile(protected_rpc_samples, 0.50),
                percentile(filler_sandwich_samples, 0.50),
                percentile(filler_sandwich_samples, 0.90),
                maximum(filler_sandwich_samples),
                maximum(filler_before_protected_done_samples),
                maximum(filler_after_protected_done_samples),
                percentile(switch_samples, 0.50),
                percentile(switch_samples, 0.90),
                protected_done_observed ? "true" : "false",
                static_cast<unsigned long long>(filler_upper_violations),
                static_cast<unsigned long long>(
                        filler_upper_violations_before_protected_done),
                static_cast<unsigned long long>(
                        filler_upper_violations_after_protected_done),
                static_cast<unsigned long long>(guard_violations),
                static_cast<unsigned long long>(idle_lower_violations),
                static_cast<unsigned long long>(
                        protected_pending_after_filler));
        fflush(stderr);

        for (arbiter_client & client : clients) {
            if (client.fd >= 0) {
                close(client.fd);
            }
            close(client.listener);
        }
        response_buffer.reset();
        request_buffer.reset();
        close_claimed_device(handle);
        libusb_exit(usb_context);
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "[phone-arbiter-bridge] %s\n", error.what());
        return 1;
    }
}
