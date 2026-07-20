#pragma once

#include "phone_pim_protocol.h"
#include "phone_pim_socket.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace phone_pim {

struct ClientOptions {
    std::string host = "127.0.0.1";
    uint16_t port = 9090;
    uint64_t route_epoch = 1;
    uint64_t residency_generation = 1;
    uint64_t max_payload = k_default_max_payload;
    int timeout_ms = 600000;
};

struct HelloInfo {
    uint32_t protocol_version = 0;
    uint64_t session_epoch = 0;
    uint64_t max_payload = 0;
    std::string backend;
    uint64_t feature_bits = 0;
    uint64_t max_model_bytes = 0;
    uint64_t max_chunk_bytes = 0;
    uint32_t max_chunks = 0;
    std::string capability;
};

struct StatusInfo {
    bool ready = false;
    uint64_t island_id = 0;
    uint64_t residency_generation = 0;
    uint64_t n_embd = 0;
    uint64_t n_ff = 0;
    uint32_t token_count = 0;
    std::string prefix;
    bool store_enabled = false;
    bool staging_active = false;
    uint64_t staging_ticket_id = 0;
    uint64_t staging_verified_bytes = 0;
    uint32_t staging_next_chunk = 0;
    std::array<uint8_t, 32> staging_manifest_sha256 = {};
};

struct PrepareRequest {
    uint64_t island_id = 0;
    uint32_t token_count = 0;
    ModelSource model_source = ModelSource::prestaged;
    std::string prefix;
    uint64_t model_bytes = 0;
    std::array<uint8_t, 32> model_sha256 = {};
};

struct PreparedInfo {
    uint64_t island_id = 0;
    uint64_t n_embd = 0;
    uint64_t n_ff = 0;
    uint32_t token_count = 0;
    uint64_t model_bytes = 0;
    std::array<uint8_t, 32> model_sha256 = {};
    ModelSource model_source = ModelSource::prestaged;
    uint64_t verify_us = 0;
    uint64_t weight_bytes = 0;
    uint64_t resident_bytes = 0;
    uint64_t load_us = 0;
    uint64_t upload_us = 0;
    uint64_t warmup_us = 0;
    std::string backend_description;
};

struct ExecutionInfo {
    std::vector<float> output;
    uint64_t input_set_us = 0;
    uint64_t compute_us = 0;
    uint64_t output_get_us = 0;
    uint64_t e2e_us = 0;
};

class ClientSession {
public:
    explicit ClientSession(ClientOptions options);

    ClientSession(const ClientSession &) = delete;
    ClientSession & operator=(const ClientSession &) = delete;

    bool connect(HelloInfo & hello, std::string & error);
    bool status(StatusInfo & status, std::string & error);
    bool prepare(const PrepareRequest & request, PreparedInfo & prepared, std::string & error);
    bool execute(uint64_t island_id, const std::vector<float> & input, ExecutionInfo & execution, std::string & error);
    bool release(uint64_t island_id, uint64_t & next_generation, std::string & error);
    bool close(std::string & error);

    bool connected() const { return socket_.valid() && session_epoch_ != 0; }
    uint64_t session_epoch() const { return session_epoch_; }
    uint64_t residency_generation() const { return options_.residency_generation; }
    const ClientOptions & options() const { return options_; }

private:
    bool call(Opcode opcode, const std::vector<uint8_t> & payload, Frame & response, std::string & error);
    void poison();

    ClientOptions options_;
    Fd socket_;
    uint64_t command_seq_ = 0;
    uint64_t request_id_ = 0;
    uint64_t session_epoch_ = 0;
    bool generation_synchronized_ = false;
    uint64_t prepared_island_id_ = 0;
    uint64_t prepared_elements_ = 0;
};

} // namespace phone_pim
