#pragma once

#include "llama.h"

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

enum server_warm_tier_model_state {
    SERVER_WARM_TIER_MODEL_ABSENT,
    SERVER_WARM_TIER_MODEL_LOADING,
    SERVER_WARM_TIER_MODEL_READY,
    SERVER_WARM_TIER_MODEL_DRAINING,
    SERVER_WARM_TIER_MODEL_REPLAYING,
    SERVER_WARM_TIER_MODEL_FAILED,
};

enum server_warm_tier_request_state {
    SERVER_WARM_TIER_REQUEST_QUEUED,
    SERVER_WARM_TIER_REQUEST_ACTIVE,
    SERVER_WARM_TIER_REQUEST_COMPLETED,
    SERVER_WARM_TIER_REQUEST_STRANDED,
};

enum server_warm_tier_executor_role {
    SERVER_WARM_TIER_EXECUTOR_GPU,
    SERVER_WARM_TIER_EXECUTOR_CPU,
    SERVER_WARM_TIER_EXECUTOR_PHONE,
};

enum server_warm_tier_finalization_state {
    SERVER_WARM_TIER_FINALIZATION_OPEN,
    SERVER_WARM_TIER_FINALIZATION_DRAINING,
    SERVER_WARM_TIER_FINALIZATION_FINALIZED,
    SERVER_WARM_TIER_FINALIZATION_FAILED,
};

enum server_warm_tier_activation_state {
    SERVER_WARM_TIER_ACTIVATION_WAITING,
    SERVER_WARM_TIER_ACTIVATION_PREPARING,
    SERVER_WARM_TIER_ACTIVATION_READY,
    SERVER_WARM_TIER_ACTIVATION_FAILED,
};

struct server_warm_tier_executor_policy {
    std::string executor_id;
    server_warm_tier_executor_role role = SERVER_WARM_TIER_EXECUTOR_GPU;
    uint32_t order = 0;
    uint32_t credits = 1;
};

enum server_warm_tier_command_kind {
    SERVER_WARM_TIER_COMMAND_EXECUTE,
    SERVER_WARM_TIER_COMMAND_DRAIN,
    SERVER_WARM_TIER_COMMAND_UNLOAD,
    SERVER_WARM_TIER_COMMAND_LOAD,
    SERVER_WARM_TIER_COMMAND_REPLAY,
    SERVER_WARM_TIER_COMMAND_DISCARD,
    SERVER_WARM_TIER_COMMAND_CLEANUP,
};

enum server_warm_tier_event_kind {
    SERVER_WARM_TIER_EVENT_RUN_START,
    SERVER_WARM_TIER_EVENT_RUN_END,
    SERVER_WARM_TIER_EVENT_REQUEST_ARRIVED,
    SERVER_WARM_TIER_EVENT_REQUEST_DISPATCHED,
    SERVER_WARM_TIER_EVENT_EXECUTE_END,
    SERVER_WARM_TIER_EVENT_TOKEN_COMMITTED,
    SERVER_WARM_TIER_EVENT_REQUEST_COMPLETED,
    SERVER_WARM_TIER_EVENT_REQUEST_STRANDED,
    SERVER_WARM_TIER_EVENT_MODEL_STATE_CHANGED,
    SERVER_WARM_TIER_EVENT_SWITCH_INTENT_SUBMITTED,
    SERVER_WARM_TIER_EVENT_SWITCH_INTENT_QUEUED,
    SERVER_WARM_TIER_EVENT_SWITCH_INTENT_COALESCED,
    SERVER_WARM_TIER_EVENT_DRAIN_BEGIN,
    SERVER_WARM_TIER_EVENT_DRAIN_END,
    SERVER_WARM_TIER_EVENT_UNLOAD_BEGIN,
    SERVER_WARM_TIER_EVENT_UNLOAD_END,
    SERVER_WARM_TIER_EVENT_LOAD_BEGIN,
    SERVER_WARM_TIER_EVENT_LOAD_END,
    SERVER_WARM_TIER_EVENT_REPLAY_BEGIN,
    SERVER_WARM_TIER_EVENT_REPLAY_END,
    SERVER_WARM_TIER_EVENT_OWNERSHIP_COMMIT,
    SERVER_WARM_TIER_EVENT_OWNERSHIP_COMMIT_COMPLETE,
    SERVER_WARM_TIER_EVENT_DISCARD_BEGIN,
    SERVER_WARM_TIER_EVENT_DISCARD_END,
    SERVER_WARM_TIER_EVENT_CLEANUP_BEGIN,
    SERVER_WARM_TIER_EVENT_CLEANUP_END,
    SERVER_WARM_TIER_EVENT_EXECUTOR_FAILED,
    SERVER_WARM_TIER_EVENT_RESOURCE_SAMPLE,
    SERVER_WARM_TIER_EVENT_PHONE_TELEMETRY,
};

struct server_warm_tier_publication {
    uint64_t publication_index = 0;
    int64_t position = 0;
    llama_token token = LLAMA_TOKEN_NULL;
    std::string owner_id;
    uint64_t ownership_epoch = 0;
};

struct server_warm_tier_request_snapshot {
    std::string request_id;
    std::string model_id;
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> committed_output_tokens;
    int64_t position = 0;
    std::string owner_id;
    uint64_t ownership_epoch = 0;
    uint64_t publication_index = 0;
    server_warm_tier_request_state state = SERVER_WARM_TIER_REQUEST_QUEUED;
};

struct server_warm_tier_command {
    uint64_t command_id = 0;
    uint64_t controller_epoch = 0;
    server_warm_tier_command_kind kind = SERVER_WARM_TIER_COMMAND_DRAIN;
    std::string model_id;
    std::string request_id;
    std::string executor_id;
    std::string executor_instance_id;
    int32_t max_output_tokens = 0;
    int32_t total_output_tokens = 0;
    server_warm_tier_request_snapshot request;
};

struct server_warm_tier_result {
    uint64_t command_id = 0;
    uint64_t controller_epoch = 0;
    server_warm_tier_command_kind kind = SERVER_WARM_TIER_COMMAND_DRAIN;
    std::string model_id;
    std::string request_id;
    std::string executor_id;
    std::string executor_instance_id;
    bool success = false;
    std::string detail;
    std::vector<server_warm_tier_publication> publications;
    bool request_complete = false;
    bool has_replay_snapshot = false;
    server_warm_tier_request_snapshot replay_snapshot;
};

struct server_warm_tier_event {
    uint32_t schema_version = 3;
    std::string run_id;
    std::string runtime_config_sha256;
    uint64_t sequence = 0;
    int64_t t_monotonic_ns = 0;
    server_warm_tier_event_kind kind = SERVER_WARM_TIER_EVENT_RUN_START;
    uint64_t controller_epoch = 0;
    bool has_command = false;
    uint64_t command_id = 0;
    server_warm_tier_command_kind command_kind = SERVER_WARM_TIER_COMMAND_DRAIN;
    std::string command_disposition;
    std::vector<server_warm_tier_publication> result_publications;
    bool result_request_complete = false;
    std::string model_id;
    std::string request_id;
    std::string executor_id;
    server_warm_tier_model_state state_before = SERVER_WARM_TIER_MODEL_ABSENT;
    server_warm_tier_model_state state_after = SERVER_WARM_TIER_MODEL_ABSENT;
    bool has_request = false;
    server_warm_tier_request_snapshot request;
    std::string old_owner_id;
    std::string new_owner_id;
    uint64_t old_ownership_epoch = 0;
    uint64_t new_ownership_epoch = 0;
    uint64_t publication_index = 0;
    std::string committed_history_digest;
    bool success = true;
    std::string detail;
};

struct server_warm_tier_switch_intent {
    uint64_t sequence = 0;
    std::string intent_id;
    std::string source_model_id;
    std::string target_model_id;
    std::string gpu_executor_id;
    std::string warm_executor_id;
};

struct server_warm_tier_executor {
    virtual ~server_warm_tier_executor() = default;

    virtual const std::string & id() const = 0;
    virtual const std::string & instance_id() const = 0;
    // submit must enqueue the command and return without calling the controller inline.
    virtual bool submit(const server_warm_tier_command & command, std::string & error) = 0;
};

using server_warm_tier_clock =
    std::function<int64_t(void)>;
using server_warm_tier_history_digest =
    std::function<std::string(
            const std::vector<llama_token> & prompt_tokens,
            const std::vector<llama_token> & committed_output_tokens)>;
using server_warm_tier_event_sink =
    std::function<void(const server_warm_tier_event & event)>;

struct server_warm_tier_options {
    bool enabled = false;
    std::string run_id;
    server_warm_tier_clock clock;
    server_warm_tier_history_digest history_digest;
    server_warm_tier_event_sink event_sink;
    std::string runtime_config_sha256;
};

struct server_warm_tier_controller {
    explicit server_warm_tier_controller(server_warm_tier_options options);
    ~server_warm_tier_controller();

    server_warm_tier_controller(const server_warm_tier_controller &) = delete;
    server_warm_tier_controller & operator=(const server_warm_tier_controller &) = delete;

    bool enabled() const;
    std::string last_error() const;

    bool register_executor(std::shared_ptr<server_warm_tier_executor> executor);
    bool set_executor_policy(server_warm_tier_executor_policy policy);
    bool set_promotion_enabled(bool enabled);
    bool set_initial_model_state(
            const std::string & model_id,
            const std::string & executor_id,
            server_warm_tier_model_state state);
    bool set_bootstrap_model_ready(
            const std::string & model_id,
            const std::string & executor_id);
    bool start();
    bool activate();
    bool stop();
    bool finalize_queued(const std::string & reason);
    server_warm_tier_activation_state activation_state() const;
    server_warm_tier_finalization_state finalization_state() const;

    bool enqueue_request(
            const std::string & request_id,
            const std::string & model_id,
            std::vector<llama_token> prompt_tokens);
    bool enqueue_scheduled_request(
            const std::string & request_id,
            const std::string & model_id,
            std::vector<llama_token> prompt_tokens,
            uint64_t arrival_order,
            int32_t max_output_tokens);
    bool dispatch_request(
            const std::string & request_id,
            const std::string & executor_id,
            int32_t max_output_tokens);
    bool publish_token(
            const std::string & request_id,
            const std::string & executor_id,
            uint64_t ownership_epoch,
            uint64_t publication_index,
            int64_t position,
            llama_token token);
    bool complete_request(
            const std::string & request_id,
            const std::string & executor_id,
            uint64_t ownership_epoch);
    bool strand_request(const std::string & request_id, const std::string & reason);

    bool submit_switch_intent(const server_warm_tier_switch_intent & intent);
    bool submit_model_switch(
            uint64_t sequence,
            const std::string & intent_id,
            const std::string & source_model_id,
            const std::string & target_model_id);
    bool coalesce_switch_intent(
            const server_warm_tier_switch_intent & obsolete,
            const server_warm_tier_switch_intent & replacement);
    bool handle_executor_result(const server_warm_tier_result & result);

    bool emit_resource_sample(
            const std::string & model_id,
            const std::string & executor_id,
            const std::string & detail);
    bool emit_phone_telemetry(
            const std::string & model_id,
            const std::string & executor_id,
            const std::string & detail);

    bool get_model_state(
            const std::string & model_id,
            const std::string & executor_id,
            server_warm_tier_model_state & state) const;
    bool get_request(
            const std::string & request_id,
            server_warm_tier_request_snapshot & request) const;
    bool has_active_transition() const;
    uint64_t controller_epoch() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

const char * server_warm_tier_model_state_name(server_warm_tier_model_state state);
const char * server_warm_tier_request_state_name(server_warm_tier_request_state state);
const char * server_warm_tier_activation_state_name(
        server_warm_tier_activation_state state);
const char * server_warm_tier_finalization_state_name(
        server_warm_tier_finalization_state state);
const char * server_warm_tier_command_kind_name(server_warm_tier_command_kind kind);
const char * server_warm_tier_event_kind_name(server_warm_tier_event_kind kind);
std::string server_warm_tier_event_jsonl(const server_warm_tier_event & event);
bool server_warm_tier_parse_json_strict(
        const std::string & input,
        nlohmann::json & output,
        std::string & error);
