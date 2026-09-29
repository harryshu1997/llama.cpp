#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

struct ggml_tensor;

namespace moe_split {

struct client_config {
    std::string host;
    int port = 0;
    uint64_t layer_mask = 0;
    uint32_t n_embd = 0;
    bool f16_io = false;
    int timeout_ms = 5000;
};

struct client_summary {
    size_t calls = 0;
    double rpc_p50_ms = 0.0;
    double rpc_p90_ms = 0.0;
    double compute_p50_ms = 0.0;
    double host_branch_p50_ms = 0.0;
    double wait_p50_ms = 0.0;
    double overlap_p50_ms = 0.0;
};

class client {
public:
    explicit client(client_config config);
    ~client();

    client(const client &) = delete;
    client & operator=(const client &) = delete;

    bool connect(std::string & error);
    bool eval(ggml_tensor * tensor, bool ask);
    void finish();

    bool failed() const;
    const std::string & error() const;
    client_summary summary() const;

    uint32_t n_ff_exp() const;
    uint32_t n_expert() const;
    uint32_t n_expert_used() const;
    uint32_t layer_count() const;
    uint64_t layer_mask() const;
    uint64_t weight_hash() const;

private:
    bool exchange(uint32_t request_id, int layer, std::string & error);
    bool parse_named_layer(const char * name, const char * prefix, int & layer) const;
    void set_error(const std::string & error);

    client_config config_;
    int fd_ = -1;
    uint32_t n_ff_exp_ = 0;
    uint32_t n_expert_ = 0;
    uint32_t n_expert_used_ = 0;
    uint32_t layer_count_ = 0;
    uint64_t weight_hash_ = 0;
    uint32_t next_request_id_ = 1;

    bool pending_ = false;
    int pending_layer_ = -1;
    bool thread_ok_ = false;
    bool failed_ = false;
    std::string error_;
    std::string thread_error_;
    std::thread thread_;
    std::vector<float> input_;
    std::vector<float> output_;
    double launch_ms_ = 0.0;
    double rpc_ms_ = 0.0;
    double compute_ms_ = 0.0;

    std::vector<double> rpc_samples_;
    std::vector<double> compute_samples_;
    std::vector<double> host_branch_samples_;
    std::vector<double> wait_samples_;
    std::vector<double> overlap_samples_;
};

} // namespace moe_split
