#pragma once

#include "../../src/llama-ffn-split-policy.h"

#if defined(FFN_SPLIT_USB_TRANSPORT)
#include "ffn-split-usb-client.h"
#endif

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct ggml_tensor;

namespace ffn_split {

enum class client_transport {
    tcp,
    functionfs_usb,
};

struct client_config {
    client_transport transport = client_transport::tcp;
    std::string artifact_sha256;
    std::string host;
    int port = 0;
    uint64_t layer_mask = 0;
    // subset of layer_mask whose gate/up/down weights exist only on the phone: always
    // dispatched at the full stored width, for every batch, independent of the runtime policy
    uint64_t remote_resident_layer_mask = 0;
    uint32_t max_columns = 0;
    uint32_t n_embd = 0;
    uint16_t max_tokens = 1;
    bool f16_io = false;
    bool swiglu = false;
    bool runtime_control = false;
    uint32_t row_diagnostic_steps = 0;
    int timeout_ms = 5000;
    std::string tail_fence_socket;
    int tail_fence_layer = -1;
    int tail_fence_join_layer = -1;
    std::string usb_allocator = "malloc";
    unsigned int usb_queue_depth = 1;
    size_t usbfs_available_bytes = 0;
    size_t usb_slot_safety_bytes = 64U * 1024U;
    size_t usb_max_payload_bytes = 0;
    std::string usb_transport_generation;
    uint16_t usb_vendor_id = 0;
    uint16_t usb_product_id = 0;
    bool usb_split_h2d = false;
    bool usb_full_duplex = false;
    std::string usb_batch_plan = "split-row";
    // request id of the first call; disjoint ranges keep call proofs unique across clients
    uint32_t first_request_id = 1;
};

struct client_shape_summary {
    uint32_t columns = 0;
    uint32_t tokens = 0;
    size_t calls = 0;
    double rpc_mean_ms = 0.0;
    double rpc_p50_ms = 0.0;
    double compute_mean_ms = 0.0;
    double compute_p50_ms = 0.0;
    double host_branch_mean_ms = 0.0;
    double wait_mean_ms = 0.0;
    double overlap_mean_ms = 0.0;
    double overlap_p50_ms = 0.0;
    double useful_overlap_mean_ms = 0.0;
};

struct client_summary {
    std::string transport;
    std::string allocator;
    std::string transport_generation;
    std::string batch_plan;
    unsigned int queue_depth = 0;
    unsigned int maximum_active_slots = 0;
    unsigned int maximum_outstanding_transfers = 0;
    size_t batched_calls = 0;
    size_t transfer_subrequests = 0;
    size_t maximum_payload_bytes = 0;
    bool full_duplex = false;
    size_t calls = 0;
    size_t decode_calls = 0;
    size_t prefill_calls = 0;
    uint64_t input_rows = 0;
    uint32_t maximum_tokens = 0;
    uint64_t upload_bytes = 0;
    uint64_t download_bytes = 0;
    uint64_t h2d_us = 0;
    uint64_t d2h_exposed_us = 0;
    double rpc_mean_ms = 0.0;
    double rpc_p50_ms = 0.0;
    double rpc_p90_ms = 0.0;
    double h2d_payload_MBps = 0.0;
    double d2h_exposed_payload_MBps = 0.0;
    double compute_mean_ms = 0.0;
    double compute_p50_ms = 0.0;
    double host_branch_mean_ms = 0.0;
    double host_branch_p50_ms = 0.0;
    double wait_mean_ms = 0.0;
    double wait_min_ms = 0.0;
    double wait_p10_ms = 0.0;
    double wait_p50_ms = 0.0;
    double wait_p90_ms = 0.0;
    double wait_max_ms = 0.0;
    double overlap_mean_ms = 0.0;
    double overlap_p50_ms = 0.0;
    double useful_overlap_mean_ms = 0.0;
    double phone_tail_mean_ms = 0.0;
    double phone_tail_min_ms = 0.0;
    double phone_tail_p10_ms = 0.0;
    double phone_tail_p50_ms = 0.0;
    double phone_tail_p90_ms = 0.0;
    double phone_tail_max_ms = 0.0;
    size_t tail_fence_opportunities = 0;
    size_t tail_fence_calls = 0;
    size_t tail_fence_skipped_complete = 0;
    double tail_fence_mean_ms = 0.0;
    double tail_fence_p50_ms = 0.0;
    double tail_fence_p90_ms = 0.0;
    double tail_fence_max_ms = 0.0;
    double tail_fence_overlap_mean_ms = 0.0;
    double tail_fence_overlap_p50_ms = 0.0;
    double tail_fence_overrun_mean_ms = 0.0;
    double tail_fence_overrun_max_ms = 0.0;
    size_t tail_fence_macro_windows = 0;
    double tail_fence_window_mean_ms = 0.0;
    double tail_fence_window_min_ms = 0.0;
    double tail_fence_window_p10_ms = 0.0;
    double tail_fence_window_p50_ms = 0.0;
    double tail_fence_window_p90_ms = 0.0;
    double tail_fence_window_max_ms = 0.0;
    double tail_fence_join_wait_mean_ms = 0.0;
    double tail_fence_join_wait_p50_ms = 0.0;
    double tail_fence_join_wait_p90_ms = 0.0;
    double tail_fence_join_wait_max_ms = 0.0;
    double decode_rpc_p50_ms = 0.0;
    double decode_compute_p50_ms = 0.0;
    double decode_overlap_p50_ms = 0.0;
    double prefill_rpc_p50_ms = 0.0;
    double prefill_compute_p50_ms = 0.0;
    double prefill_overlap_p50_ms = 0.0;
    std::vector<client_shape_summary> shapes;
};

struct client_runtime_context_entry {
    std::string request_id;
    int32_t slot_id = -1;
    uint32_t rows = 0;
    uint64_t plan_generation = 0;
    std::vector<uint32_t> ubatch_rows;
    std::vector<int32_t> positions;
    int32_t decoded_token_index = -1;
    int32_t applied_token_index = -1;
};

class client {
public:
    explicit client(client_config config);
    ~client();

    client(const client &) = delete;
    client & operator=(const client &) = delete;

    bool connect(std::string & error, uint64_t layer_mask = 0);
    bool ready() const;
    bool set_runtime_policy(
            uint64_t layer_mask, uint32_t columns, std::string & error);
    bool set_runtime_context(
            const std::vector<client_runtime_context_entry> & entries,
            std::string & error);
    bool eval(ggml_tensor * tensor, bool ask);
    void finish();
    // drops a failed or closed session (the call statistics are kept) so that connect() starts a new one;
    // a live USB session is shut down first
    void reset_session();
    // an idle TCP session whose worker closed the connection
    bool peer_closed() const;
    // keeps a failed reconnect visible (status and summary) after a reset cleared the old error
    void latch_error(const std::string & error);
    size_t reset_count() const;
    const std::string & last_reset_error() const;

    bool failed() const;
    const std::string & error() const;
    client_summary summary() const;
    client_summary summary(
            const std::vector<std::string> & request_ids) const;

    uint32_t n_ff() const;
    uint32_t offset() const;
    uint32_t layer_count() const;
    uint32_t max_columns() const;
    uint32_t column_quantum() const;
    uint32_t alternate_columns() const;
    uint16_t max_tokens() const;
    uint64_t layer_mask() const;
    uint64_t remote_resident_layer_mask() const;
    bool remote_resident_layer(int layer) const;
    uint64_t weight_hash() const;

private:
    struct runtime_summary_accumulator {
        size_t calls = 0;
        size_t batched_calls = 0;
        size_t transfer_subrequests = 0;
        uint64_t input_rows = 0;
        uint32_t maximum_tokens = 0;
        uint64_t upload_bytes = 0;
        uint64_t download_bytes = 0;
        uint64_t h2d_us = 0;
        uint64_t d2h_exposed_us = 0;
        double rpc_total_ms = 0.0;
        double compute_total_ms = 0.0;
        double host_branch_total_ms = 0.0;
        double wait_total_ms = 0.0;
        double useful_overlap_total_ms = 0.0;
    };

    static std::string runtime_context_key(
            const std::vector<std::string> & request_ids);
    void record_runtime_summary(
            const std::vector<client_runtime_context_entry> & entries,
            uint32_t tokens, size_t payload_bytes,
            uint32_t transfer_subrequests, double h2d_ms,
            double d2h_exposed_ms, double rpc_ms, double compute_ms,
            double host_branch_ms, double wait_ms,
            double useful_overlap_ms);
    bool exchange(
            uint32_t request_id, int layer, uint32_t elements,
            uint32_t columns, uint32_t tokens, std::string & error);
#if defined(FFN_SPLIT_USB_TRANSPORT)
    struct usb_pending_part {
        usb_slot_buffers buffers;
        uint32_t request_id = 0;
        uint32_t elements = 0;
        uint32_t tokens = 0;
        uint32_t token_offset = 0;
        size_t payload_bytes = 0;
        size_t host_to_device_bytes = 0;
        size_t device_to_host_bytes = 0;
        usb_transfer_record transfer;
        uint64_t compute_us = 0;
    };

    bool start_usb_exchange(
            uint32_t request_id, int layer, uint32_t elements,
            uint32_t columns, uint32_t tokens, ggml_tensor * tensor,
            unsigned int & part_count, std::string & error);
    bool publish_usb_output(ggml_tensor * tensor, std::string & error);
    void release_usb_exchange();
#endif
    void log_diagnostic_rows(const char * stage, ggml_tensor * tensor, uint32_t call, int layer);
    // local-shadow rows of the pending call, keyed by ubatch row, for the returned-row distance
    std::map<uint32_t, std::vector<float>> diagnostic_local_rows_;
    bool parse_named_layer(const char * name, const char * prefix, int & layer) const;
    bool runtime_columns(
            uint32_t tokens, uint64_t & layer_mask, uint32_t & columns,
            std::string & error) const;
    bool exchange_tail_fence(
            uint64_t sequence, uint32_t request_id, int layer,
            uint32_t tokens,
            std::string & error);
    bool start_async_tail_fence(
            uint32_t request_id, int layer, uint32_t tokens,
            std::string & error);
    bool finish_async_tail_fence(
            int64_t join_ready_ns, std::string & error);
    bool rearm_runtime_connection(
            uint64_t layer_mask, std::string & error);
    bool parse_named_layer_any(
            const char * name, const char * prefix, int & layer) const;
    void set_error(const std::string & error);
    bool send_usb_shutdown(std::string & error);
    void worker_loop();
    bool connected() const;

    client_config config_;
    int fd_ = -1;
#if defined(FFN_SPLIT_USB_TRANSPORT)
    std::unique_ptr<usb_client> usb_;
    std::vector<usb_pending_part> usb_parts_;
#endif
    int tail_fence_fd_ = -1;
    uint32_t n_ff_ = 0;
    uint32_t offset_ = 0;
    uint32_t layer_count_ = 0;
    uint32_t column_quantum_ = 0;
    uint32_t alternate_columns_ = 0;
    uint64_t connected_layer_mask_ = 0;
    uint64_t weight_hash_ = 0;
    mutable std::mutex runtime_policy_mutex_;
    std::atomic<uint64_t> runtime_layer_mask_ { 0 };
    std::atomic<uint32_t> runtime_columns_ { 0 };
    mutable std::mutex runtime_context_mutex_;
    std::vector<client_runtime_context_entry> runtime_context_;
    std::vector<client_runtime_context_entry> pending_runtime_context_;
    std::map<std::string, runtime_summary_accumulator>
            runtime_summary_by_context_;
    std::vector<llama_ffn_split_policy::point> policy_;
    uint32_t next_request_id_ = 1;

    bool pending_ = false;
    uint32_t pending_request_id_ = 0;
    int pending_layer_ = -1;
    uint32_t pending_elements_ = 0;
    uint32_t pending_columns_ = 0;
    uint32_t pending_tokens_ = 0;
    bool thread_ok_ = false;
    bool thread_stop_ = false;
    bool thread_has_job_ = false;
    bool thread_done_ = false;
    int64_t thread_done_ns_ = 0;
    uint32_t thread_request_id_ = 0;
    int thread_layer_ = -1;
    uint32_t thread_elements_ = 0;
    uint32_t thread_columns_ = 0;
    uint32_t thread_tokens_ = 0;
    bool failed_ = false;
    std::string error_;
    // sticky across resets: a reconnect must reach the same worker weights
    uint64_t expected_weight_hash_ = 0;
    size_t reset_count_ = 0;
    std::string last_reset_error_;
    std::string thread_error_;
    std::thread thread_;
    std::thread tail_fence_thread_;
    std::mutex thread_mutex_;
    std::mutex tail_fence_mutex_;
    std::condition_variable thread_job_cv_;
    std::condition_variable thread_done_cv_;
    std::condition_variable tail_fence_done_cv_;
    std::vector<float> input_;
    std::vector<float> output_;
    std::vector<uint16_t> encoded_input_;
    std::vector<uint8_t> request_packet_;
    std::vector<uint8_t> response_payload_;
    bool usb_shutdown_sent_ = false;
    double launch_ms_ = 0.0;
    double rpc_ms_ = 0.0;
    double compute_ms_ = 0.0;
    uint64_t next_tail_fence_sequence_ = 1;
    bool tail_fence_pending_ = false;
    bool tail_fence_done_ = false;
    bool tail_fence_ok_ = false;
    int64_t tail_fence_started_ns_ = 0;
    int64_t tail_fence_done_ns_ = 0;
    std::string tail_fence_error_;

    std::vector<double> rpc_samples_;
    std::vector<double> h2d_samples_;
    std::vector<double> d2h_exposed_samples_;
    std::vector<double> compute_samples_;
    std::vector<double> host_branch_samples_;
    std::vector<double> wait_samples_;
    std::vector<double> overlap_samples_;
    std::vector<double> useful_overlap_samples_;
    std::vector<double> phone_tail_samples_;
    std::vector<double> tail_fence_samples_;
    std::vector<double> tail_fence_overlap_samples_;
    std::vector<double> tail_fence_overrun_samples_;
    std::vector<double> tail_fence_window_samples_;
    std::vector<double> tail_fence_join_wait_samples_;
    size_t tail_fence_opportunities_ = 0;
    size_t tail_fence_skipped_complete_ = 0;
    std::vector<uint32_t> columns_samples_;
    std::vector<uint32_t> tokens_samples_;
    std::vector<uint8_t> decode_samples_;
    std::vector<uint32_t> payload_samples_;
    std::vector<uint32_t> transfer_parts_samples_;
};

} // namespace ffn_split
