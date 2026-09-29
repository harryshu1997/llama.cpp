#pragma once

#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <set>

struct server_context_impl; // private implementation

struct server_ffn_dormant_state {
    bool     enabled = false;
    uint64_t layer_mask = 0;
    uint32_t host_columns = 0;
    uint64_t release_generation = 0;
    size_t   released_bytes = 0;
    int64_t  release_elapsed_us = 0;
};

struct server_context_meta {
    std::string build_info;
    std::string model_name;
    std::set<std::string> model_aliases;
    std::set<std::string> model_tags;
    std::string model_path;
    bool has_mtmd;
    bool has_inp_image;
    bool has_inp_audio;
    bool has_inp_video;
    json json_ui_settings;
    int slot_n_ctx;
    enum llama_pooling_type pooling_type;

    // chat params
    server_chat_params & chat_params;
    std::map<std::string, bool> chat_template_caps;

    // tokens
    std::string bos_token_str;
    std::string eos_token_str;
    llama_token fim_pre_token;
    llama_token fim_sub_token;
    llama_token fim_mid_token;
    llama_token fim_pad_token;
    llama_token fim_rep_token;
    llama_token fim_sep_token;

    // sampling
    std::vector<llama_logit_bias> logit_bias_eog;

    // model meta
    enum llama_vocab_type model_vocab_type;
    int32_t model_vocab_n_tokens;
    int32_t model_n_ctx_train;
    int32_t model_n_embd_inp;
    uint64_t model_n_params;
    uint64_t model_size;
};

enum server_state {
    SERVER_STATE_DOWNLOADING,
    SERVER_STATE_LOADING,
    SERVER_STATE_READY,
    SERVER_STATE_SLEEPING,
};

static std::string server_state_to_str(server_state state) {
    switch (state) {
        case SERVER_STATE_DOWNLOADING: return "downloading";
        case SERVER_STATE_LOADING:     return "loading";
        case SERVER_STATE_READY:       return "ready";
        case SERVER_STATE_SLEEPING:    return "sleeping";
        default: GGML_ASSERT(false && "invalid server_state");
    }
}

static server_state server_state_from_str(const std::string & str) {
    if (str == "downloading") return SERVER_STATE_DOWNLOADING;
    if (str == "loading")     return SERVER_STATE_LOADING;
    if (str == "ready")       return SERVER_STATE_READY;
    if (str == "sleeping")    return SERVER_STATE_SLEEPING;
    GGML_ASSERT(false && "invalid server_state string");
}

using server_state_callback_t = std::function<void(server_state, json /* payload */)>;
using server_ffn_split_apply_callback_t = std::function<bool(
        uint64_t, uint32_t, std::string &)>;
struct server_ffn_split_runtime_context {
    std::string request_id;
    int32_t slot_id = -1;
    uint32_t rows = 0;
    uint64_t plan_generation = 0;
    std::vector<uint32_t> ubatch_rows;
    std::vector<int32_t> positions;
    int32_t decoded_token_index = -1;
    int32_t applied_token_index = -1;
};
using server_ffn_split_context_callback_t = std::function<bool(
        const std::vector<server_ffn_split_runtime_context> &,
        std::string &)>;
using server_ffn_split_stats_callback_t = std::function<json(
        const std::vector<std::string> &)>;

struct server_context {
    std::unique_ptr<server_context_impl> impl;

    server_context();
    ~server_context();

    // load the model and initialize llama_context
    // returns true on success
    bool load_model(common_params & params);

    // this function will block main thread until termination
    void start_loop();

    // terminate main loop (will unblock start_loop)
    void terminate();

    // get the underlaying llama_context, can return nullptr if sleeping
    // not thread-safe, should only be used from the main thread
    llama_context * get_llama_context() const;

    // get a new response reader, used by CLI application
    server_response_reader get_response_reader();

    // get server metadata (read-only), can only be called after load_model()
    // not thread-safe, should only be used from the main thread
    server_context_meta get_meta() const;

    // note: must be set before load_model() is called
    void set_state_callback(server_state_callback_t callback);

    // Layers whose FFN weights exist only on the phone: controls may never target them.
    void configure_ffn_remote_resident(uint64_t remote_resident_layer_mask);
    // S42 dormant host share: release the phone-executed FFN column suffix while every processing
    // slot decodes, populate it again before any prompt processing (needs runtime control).
    void configure_ffn_dormant_host_share(bool enabled, bool drop_cache = true, bool populate = true, uint32_t row_diagnostic_steps = 0);
    // current dormant host share state: the release generation counts decode-phase releases since
    // launch, released_bytes is 0 while the share is resident (populated)
    server_ffn_dormant_state ffn_dormant_state() const;

    // Configure the maximum preloaded dense FFN helper slice.
    void configure_ffn_split_runtime(
            uint64_t layer_mask,
            uint32_t max_columns,
            uint32_t column_quantum,
            server_ffn_split_apply_callback_t apply_callback,
            server_ffn_split_context_callback_t context_callback,
            server_ffn_split_stats_callback_t stats_callback);
};


// forward declarations
struct server_res_generator;

struct server_routes {
    server_routes(const common_params & params, server_context & ctx_server);

    void init_routes();

    // note: this is not thread-safe and can only when ctx_http.is_ready is false
    void update_meta(const server_context & ctx_server) {
        this->meta = std::make_unique<server_context_meta>(ctx_server.get_meta());
    }

    // handlers using lambda function, so that they can capture `this` without `std::bind`
    // they won't be called until ctx_http.is_ready is set to true
    server_http_context::handler_t get_health;
    server_http_context::handler_t get_metrics;
    server_http_context::handler_t get_slots;
    server_http_context::handler_t post_slots;
    server_http_context::handler_t get_props;
    server_http_context::handler_t post_props;
    server_http_context::handler_t post_infill;
    server_http_context::handler_t post_completions;
    server_http_context::handler_t post_completions_oai;
    server_http_context::handler_t post_chat_completions;
    server_http_context::handler_t post_chat_completions_tok;
    server_http_context::handler_t post_control;
    server_http_context::handler_t post_responses_oai;
    server_http_context::handler_t post_responses_tok_oai;
    server_http_context::handler_t post_transcriptions_oai;
    server_http_context::handler_t post_anthropic_messages;
    server_http_context::handler_t post_anthropic_count_tokens;
    server_http_context::handler_t post_apply_template;
    server_http_context::handler_t get_models;
    server_http_context::handler_t post_tokenize;
    server_http_context::handler_t post_detokenize;
    server_http_context::handler_t post_embeddings;
    server_http_context::handler_t post_embeddings_oai;
    server_http_context::handler_t post_rerank;
    server_http_context::handler_t get_lora_adapters;
    server_http_context::handler_t post_lora_adapters;

    // to be used in router mode
    json get_model_info() const;

private:
    std::unique_ptr<server_res_generator> handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_slots_save(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_erase(const server_http_req &, int id_slot);
    std::unique_ptr<server_res_generator> handle_embeddings_impl(const server_http_req & req, task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const server_http_req & req, task_response_type res_type);

    // using unique_ptr to allow late initialization of const
    std::unique_ptr<const server_context_meta> meta;

    const common_params & params;
    const server_context_impl & ctx_server;

    server_queue & queue_tasks;
    server_response & queue_results;
    std::unique_ptr<server_res_generator> create_response(bool bypass_sleep = false);
};
