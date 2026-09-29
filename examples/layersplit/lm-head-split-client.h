#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

struct ggml_tensor;

namespace lm_head_split {

struct client_config {
    std::string host;
    int port = 0;
    uint32_t rows = 0;
    uint32_t top_k = 0;
    uint32_t n_embd = 0;
    uint32_t n_vocab = 0;
    bool f16_io = false;
    int timeout_ms = 5000;
};

struct client_summary {
    size_t calls = 0;
    double rpc_p50_ms = 0.0;
    double rpc_p90_ms = 0.0;
    double compute_p50_ms = 0.0;
    double reduce_p50_ms = 0.0;
    double host_branch_p50_ms = 0.0;
    double wait_p50_ms = 0.0;
    double rescore_p50_ms = 0.0;
    double score_error_max = 0.0;
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

    uint32_t offset() const;
    uint64_t weight_hash() const;

private:
    bool exchange(uint32_t request_id, std::string & error);
    void set_error(const std::string & error);
    void reset_request();

    client_config config_;
    int fd_ = -1;
    uint32_t offset_ = 0;
    uint64_t weight_hash_ = 0;
    uint32_t next_request_id_ = 1;

    bool request_active_ = false;
    bool rpc_pending_ = false;
    bool thread_ok_ = false;
    bool scores_ready_ = false;
    bool failed_ = false;
    std::string error_;
    std::string thread_error_;
    std::thread thread_;
    std::vector<float> input_;
    std::vector<uint32_t> candidate_ids_;
    std::vector<float> approximate_scores_;
    std::vector<float> exact_scores_;
    double launch_ms_ = 0.0;
    double joined_ms_ = 0.0;
    double rpc_ms_ = 0.0;
    double compute_ms_ = 0.0;
    double reduce_ms_ = 0.0;

    std::vector<double> rpc_samples_;
    std::vector<double> compute_samples_;
    std::vector<double> reduce_samples_;
    std::vector<double> host_branch_samples_;
    std::vector<double> wait_samples_;
    std::vector<double> rescore_samples_;
    double score_error_max_ = 0.0;
};

} // namespace lm_head_split
