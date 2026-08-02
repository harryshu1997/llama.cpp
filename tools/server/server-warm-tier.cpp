#include "server-warm-tier.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <utility>

namespace {

using route_key = std::pair<std::string, std::string>;

enum transition_phase {
    TRANSITION_NONE,
    TRANSITION_DRAINING,
    TRANSITION_UNLOADING,
    TRANSITION_LOADING,
    TRANSITION_REPLAYING,
    TRANSITION_CLEANUP,
    TRANSITION_ROLLBACK,
    TRANSITION_WARM_DRAINING,
    TRANSITION_WARM_UNLOADING,
    TRANSITION_WARM_LOADING,
};

struct request_record {
    server_warm_tier_request_snapshot snapshot;
    std::vector<server_warm_tier_publication> publications;
    uint64_t arrival_sequence = 0;
    int32_t total_output_tokens = 0;
    bool scheduled = false;
};

struct transition_record {
    bool active = false;
    uint64_t epoch = 0;
    transition_phase phase = TRANSITION_NONE;
    server_warm_tier_switch_intent intent;
    std::set<std::string> replay_waiting;
    std::set<std::string> cleanup_waiting;
    std::map<std::string, std::string> old_owners;
    std::string failure;
};

server_warm_tier_event_kind event_begin(server_warm_tier_command_kind kind) {
    switch (kind) {
        case SERVER_WARM_TIER_COMMAND_EXECUTE: return SERVER_WARM_TIER_EVENT_REQUEST_DISPATCHED;
        case SERVER_WARM_TIER_COMMAND_DRAIN:   return SERVER_WARM_TIER_EVENT_DRAIN_BEGIN;
        case SERVER_WARM_TIER_COMMAND_UNLOAD:  return SERVER_WARM_TIER_EVENT_UNLOAD_BEGIN;
        case SERVER_WARM_TIER_COMMAND_LOAD:    return SERVER_WARM_TIER_EVENT_LOAD_BEGIN;
        case SERVER_WARM_TIER_COMMAND_REPLAY:  return SERVER_WARM_TIER_EVENT_REPLAY_BEGIN;
        case SERVER_WARM_TIER_COMMAND_DISCARD: return SERVER_WARM_TIER_EVENT_DISCARD_BEGIN;
        case SERVER_WARM_TIER_COMMAND_CLEANUP: return SERVER_WARM_TIER_EVENT_CLEANUP_BEGIN;
    }
    return SERVER_WARM_TIER_EVENT_EXECUTOR_FAILED;
}

server_warm_tier_event_kind event_end(server_warm_tier_command_kind kind) {
    switch (kind) {
        case SERVER_WARM_TIER_COMMAND_EXECUTE: return SERVER_WARM_TIER_EVENT_EXECUTE_END;
        case SERVER_WARM_TIER_COMMAND_DRAIN:   return SERVER_WARM_TIER_EVENT_DRAIN_END;
        case SERVER_WARM_TIER_COMMAND_UNLOAD:  return SERVER_WARM_TIER_EVENT_UNLOAD_END;
        case SERVER_WARM_TIER_COMMAND_LOAD:    return SERVER_WARM_TIER_EVENT_LOAD_END;
        case SERVER_WARM_TIER_COMMAND_REPLAY:  return SERVER_WARM_TIER_EVENT_REPLAY_END;
        case SERVER_WARM_TIER_COMMAND_DISCARD: return SERVER_WARM_TIER_EVENT_DISCARD_END;
        case SERVER_WARM_TIER_COMMAND_CLEANUP: return SERVER_WARM_TIER_EVENT_CLEANUP_END;
    }
    return SERVER_WARM_TIER_EVENT_EXECUTOR_FAILED;
}

bool same_replay_frontier(
        const server_warm_tier_request_snapshot & expected,
        const server_warm_tier_request_snapshot & actual) {
    return expected.request_id == actual.request_id
        && expected.model_id == actual.model_id
        && expected.prompt_tokens == actual.prompt_tokens
        && expected.committed_output_tokens == actual.committed_output_tokens
        && expected.position == actual.position
        && expected.owner_id == actual.owner_id
        && expected.ownership_epoch == actual.ownership_epoch
        && expected.publication_index == actual.publication_index
        && expected.state == actual.state;
}

bool is_sha256(const std::string & value) {
    if (value.size() != 64) {
        return false;
    }
    for (char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            return false;
        }
    }
    return true;
}

}

struct server_warm_tier_controller::impl {
    server_warm_tier_options options;
    bool started = false;
    bool stopped = false;
    bool poisoned = false;
    bool event_sink_failed = false;
    bool finalizing = false;
    bool activation_requested = false;
    bool activated = false;
    bool activation_failed = false;
    std::string finalize_reason;
    std::string error;
    uint64_t event_sequence = 0;
    uint64_t next_command_id = 1;
    uint64_t epoch = 0;
    uint64_t last_intent_sequence = 0;
    bool has_intent_sequence = false;
    uint64_t last_accepted_intent_sequence = 0;
    bool has_accepted_intent_sequence = false;
    uint64_t next_arrival_sequence = 0;
    uint64_t last_scheduled_arrival = 0;
    bool has_scheduled_arrival = false;
    bool promotion_enabled = false;
    int64_t last_event_ns = -1;

    std::map<std::string, std::shared_ptr<server_warm_tier_executor>> executors;
    std::map<std::string, server_warm_tier_executor_policy> executor_policies;
    std::map<route_key, server_warm_tier_model_state> routes;
    std::map<std::string, request_record> requests;
    std::map<std::string, std::deque<std::string>> pending_by_model;
    std::deque<route_key> bootstrap_waiting;
    std::optional<route_key> bootstrap_active;
    std::optional<server_warm_tier_switch_intent> pending_intent;
    std::map<uint64_t, server_warm_tier_command> commands;
    std::map<uint64_t, std::string> quarantined_commands;
    transition_record transition;
    mutable std::mutex mutex;

    explicit impl(server_warm_tier_options options)
        : options(std::move(options)) {
    }

    bool fail(const std::string & message) {
        error = message;
        return false;
    }

    bool operational() {
        if (!options.enabled) {
            return fail("E_DISABLED");
        }
        if (poisoned) {
            return fail("E_POISONED");
        }
        if (!started || stopped) {
            return fail("E_NOT_RUNNING");
        }
        return true;
    }

    bool admitting() {
        if (!operational()) {
            return false;
        }
        if (!activated) {
            return fail("E_NOT_ACTIVATED");
        }
        return true;
    }

    server_warm_tier_request_snapshot snapshot(const request_record & request) const {
        server_warm_tier_request_snapshot result = request.snapshot;
        result.publication_index = request.publications.size();
        return result;
    }

    std::string digest(const server_warm_tier_request_snapshot & request) const {
        if (!options.history_digest) {
            return {};
        }
        return options.history_digest(
                request.prompt_tokens,
                request.committed_output_tokens);
    }

    bool emit(server_warm_tier_event event) {
        event.schema_version = 3;
        event.run_id = options.run_id;
        event.runtime_config_sha256 = options.runtime_config_sha256;
        event.sequence = event_sequence++;
        event.controller_epoch = epoch;
        if (options.clock) {
            event.t_monotonic_ns = options.clock();
        }
        if (event.t_monotonic_ns < last_event_ns) {
            return fail("E_CLOCK_NONMONOTONIC");
        }
        last_event_ns = event.t_monotonic_ns;
        if (event.has_request) {
            event.committed_history_digest = digest(event.request);
            if (!is_sha256(event.committed_history_digest)) {
                poisoned = true;
                return fail("E_HISTORY_SHA256");
            }
        }
        if (options.event_sink) {
            try {
                options.event_sink(event);
            } catch (...) {
                event_sink_failed = true;
                poisoned = true;
                return fail("E_EVENT_SINK");
            }
        }
        return true;
    }

    bool set_route_state(
            const std::string & model_id,
            const std::string & executor_id,
            server_warm_tier_model_state state) {
        route_key key{model_id, executor_id};
        auto found = routes.find(key);
        const bool existed = found != routes.end();
        const server_warm_tier_model_state before =
            !existed ? SERVER_WARM_TIER_MODEL_ABSENT : found->second;
        routes[key] = state;

        server_warm_tier_event event;
        event.kind = SERVER_WARM_TIER_EVENT_MODEL_STATE_CHANGED;
        event.model_id = model_id;
        event.executor_id = executor_id;
        event.state_before = before;
        event.state_after = state;
        if (emit(std::move(event))) {
            return true;
        }
        if (existed) {
            routes[key] = before;
        } else {
            routes.erase(key);
        }
        return false;
    }

    bool route_is(
            const std::string & model_id,
            const std::string & executor_id,
            server_warm_tier_model_state state) const {
        auto found = routes.find({model_id, executor_id});
        return found != routes.end() && found->second == state;
    }

    bool emit_command_event(
            const server_warm_tier_command & command,
            server_warm_tier_event_kind kind,
            bool success,
            const std::string & detail,
            const std::string & disposition = {},
            const std::vector<server_warm_tier_publication> & publications = {},
            bool request_complete = false) {
        server_warm_tier_event event;
        event.kind = kind;
        event.model_id = command.model_id;
        event.request_id = command.request_id;
        event.executor_id = command.executor_id;
        event.has_command = true;
        event.command_id = command.command_id;
        event.command_kind = command.kind;
        event.command_disposition = disposition;
        event.result_publications = publications;
        event.result_request_complete = request_complete;
        event.success = success;
        event.detail = detail;
        if (!command.request_id.empty()) {
            event.has_request = true;
            event.request = command.request;
        }
        return emit(std::move(event));
    }

    bool maybe_finish_finalization() {
        if (!finalizing || stopped) {
            return true;
        }
        if (transition.active || !commands.empty()) {
            return true;
        }
        for (const auto & item : requests) {
            if (item.second.snapshot.state == SERVER_WARM_TIER_REQUEST_ACTIVE
                    || item.second.snapshot.state == SERVER_WARM_TIER_REQUEST_QUEUED) {
                return true;
            }
        }
        server_warm_tier_event event;
        event.kind = SERVER_WARM_TIER_EVENT_RUN_END;
        event.detail = finalize_reason;
        if (!emit(std::move(event))) {
            poisoned = true;
            return false;
        }
        stopped = true;
        return true;
    }

    bool terminate_transition(const std::string & failure) {
        transition.active = false;
        transition.phase = TRANSITION_NONE;
        transition.replay_waiting.clear();
        transition.cleanup_waiting.clear();
        transition.old_owners.clear();
        transition.failure = failure;
        return failure.empty() ? true : fail(failure);
    }

    bool command_matches(
            const server_warm_tier_command & command,
            const server_warm_tier_result & result) const {
        return command.command_id == result.command_id
            && command.controller_epoch == result.controller_epoch
            && command.kind == result.kind
            && command.model_id == result.model_id
            && command.request_id == result.request_id
            && command.executor_id == result.executor_id
            && command.executor_instance_id == result.executor_instance_id;
    }

    bool fail_unstarted_command(
            const server_warm_tier_command & command,
            const std::string & reason) {
        poisoned = true;
        switch (command.kind) {
            case SERVER_WARM_TIER_COMMAND_DRAIN:
            case SERVER_WARM_TIER_COMMAND_UNLOAD:
                return start_emergency_discard(command, reason);
            case SERVER_WARM_TIER_COMMAND_REPLAY:
                if (transition.active) {
                    return begin_rollback(reason, false);
                }
                routes[{command.model_id, command.executor_id}] =
                    SERVER_WARM_TIER_MODEL_FAILED;
                return fail(reason);
            case SERVER_WARM_TIER_COMMAND_EXECUTE:
                quarantine_all_commands(reason);
                for (auto & item : requests) {
                    auto & request = item.second.snapshot;
                    if (request.model_id == command.model_id
                            && request.owner_id == command.executor_id
                            && request.state
                                == SERVER_WARM_TIER_REQUEST_ACTIVE) {
                        request.state = SERVER_WARM_TIER_REQUEST_STRANDED;
                    }
                }
                routes[{command.model_id, command.executor_id}] =
                    SERVER_WARM_TIER_MODEL_FAILED;
                if (transition.active) {
                    for (const auto & owner : transition.old_owners) {
                        routes[{
                            transition.intent.target_model_id,
                            owner.second,
                        }] = SERVER_WARM_TIER_MODEL_FAILED;
                    }
                    return terminate_transition(reason);
                }
                return fail(reason);
            case SERVER_WARM_TIER_COMMAND_CLEANUP:
                quarantine_all_commands(reason);
                routes[{command.model_id, command.executor_id}] =
                    SERVER_WARM_TIER_MODEL_FAILED;
                if (transition.active) {
                    transition.cleanup_waiting.clear();
                    return terminate_transition(reason);
                }
                return fail(reason);
            case SERVER_WARM_TIER_COMMAND_LOAD:
                quarantine_all_commands(reason);
                routes[{command.model_id, command.executor_id}] =
                    SERVER_WARM_TIER_MODEL_FAILED;
                if (bootstrap_active.has_value()
                        && *bootstrap_active
                            == route_key{
                                command.model_id,
                                command.executor_id,
                            }) {
                    bootstrap_active.reset();
                    bootstrap_waiting.clear();
                    activation_failed = true;
                }
                if (transition.active) {
                    return terminate_transition(reason);
                }
                return fail(reason);
            case SERVER_WARM_TIER_COMMAND_DISCARD:
                routes[{command.model_id, command.executor_id}] =
                    SERVER_WARM_TIER_MODEL_FAILED;
                if (transition.active) {
                    return terminate_transition(reason);
                }
                return fail(reason);
        }
        return fail(reason);
    }

    uint64_t issue(
            server_warm_tier_command_kind kind,
            const std::string & model_id,
            const std::string & request_id,
            const std::string & executor_id,
            const server_warm_tier_request_snapshot * request = nullptr,
            int32_t max_output_tokens = 0) {
        auto found = executors.find(executor_id);
        if (found == executors.end()) {
            fail("E_EXECUTOR_UNKNOWN:" + executor_id);
            return 0;
        }

        server_warm_tier_command command;
        command.command_id = next_command_id++;
        command.controller_epoch = epoch;
        command.kind = kind;
        command.model_id = model_id;
        command.request_id = request_id;
        command.executor_id = executor_id;
        command.executor_instance_id = found->second->instance_id();
        command.max_output_tokens = max_output_tokens;
        if (request != nullptr) {
            command.request = *request;
        }
        if (kind == SERVER_WARM_TIER_COMMAND_EXECUTE) {
            auto request_record = requests.find(request_id);
            if (request_record == requests.end()
                    || request_record->second.total_output_tokens <= 0) {
                fail("E_OUTPUT_BUDGET");
                return 0;
            }
            command.total_output_tokens =
                request_record->second.total_output_tokens;
        }
        commands.emplace(command.command_id, command);
        bool suppress_evidence =
            kind == SERVER_WARM_TIER_COMMAND_DISCARD && event_sink_failed;
        if (!suppress_evidence
                && !emit_command_event(command, event_begin(kind), true, {})) {
            suppress_evidence =
                kind == SERVER_WARM_TIER_COMMAND_DISCARD && event_sink_failed;
            if (!suppress_evidence) {
                commands.erase(command.command_id);
                const std::string failure =
                    error.empty() ? "E_COMMAND_BEGIN" : error;
                fail_unstarted_command(command, failure);
                return 0;
            }
        }

        std::string submit_error;
        if (!found->second->submit(command, submit_error)) {
            commands.erase(command.command_id);
            if (!suppress_evidence) {
                emit_command_event(
                    command, event_end(kind), false, submit_error, "RECEIVED");
            }
            executor_failure(
                    command,
                    submit_error.empty() ? "E_EXECUTOR_REJECTED" : submit_error,
                    !suppress_evidence);
            return 0;
        }
        return command.command_id;
    }

    bool issue_next_bootstrap() {
        if (bootstrap_active.has_value()) {
            return true;
        }
        if (bootstrap_waiting.empty()) {
            activated = true;
            return true;
        }
        const route_key key = bootstrap_waiting.front();
        if (!route_is(
                    key.first,
                    key.second,
                    SERVER_WARM_TIER_MODEL_ABSENT)) {
            activation_failed = true;
            poisoned = true;
            return fail("E_BOOTSTRAP_ROUTE_STATE");
        }
        if (!set_route_state(
                    key.first,
                    key.second,
                    SERVER_WARM_TIER_MODEL_LOADING)) {
            activation_failed = true;
            poisoned = true;
            return false;
        }
        bootstrap_active = key;
        if (issue(
                    SERVER_WARM_TIER_COMMAND_LOAD,
                    key.first,
                    {},
                    key.second) == 0) {
            activation_failed = true;
            poisoned = true;
            return false;
        }
        return true;
    }

    void quarantine_epoch_commands(
            uint64_t canceled_epoch,
            const std::string & reason) {
        for (const auto & item : commands) {
            if (item.second.controller_epoch == canceled_epoch) {
                quarantined_commands.emplace(item.first, reason);
            }
        }
    }

    void quarantine_all_commands(const std::string & reason) {
        for (const auto & item : commands) {
            quarantined_commands.emplace(item.first, reason);
        }
    }

    bool has_quarantined_epoch_commands(uint64_t canceled_epoch) const {
        for (const auto & item : quarantined_commands) {
            const auto command = commands.find(item.first);
            if (command != commands.end()
                    && command->second.controller_epoch == canceled_epoch) {
                return true;
            }
        }
        return false;
    }

    bool start_rollback_discard() {
        if (!transition.active || transition.phase != TRANSITION_ROLLBACK) {
            return true;
        }
        if (has_quarantined_epoch_commands(transition.epoch)) {
            return true;
        }
        const auto & intent = transition.intent;
        for (const auto & item : commands) {
            if (item.second.kind == SERVER_WARM_TIER_COMMAND_DISCARD
                    && item.second.controller_epoch == transition.epoch
                    && item.second.model_id == intent.target_model_id
                    && item.second.executor_id == intent.gpu_executor_id) {
                return true;
            }
        }
        if (issue(
                    SERVER_WARM_TIER_COMMAND_DISCARD,
                    intent.target_model_id,
                    {},
                    intent.gpu_executor_id) == 0) {
            terminate_transition(transition.failure + ":E_DISCARD_SUBMIT");
            poisoned = true;
            return false;
        }
        return true;
    }

    bool begin_rollback(
            const std::string & reason,
            bool publish_state_events = true) {
        poisoned = true;
        const uint64_t failed_epoch = transition.epoch;
        quarantine_epoch_commands(failed_epoch, reason);
        transition.replay_waiting.clear();
        transition.cleanup_waiting.clear();
        transition.old_owners.clear();
        transition.failure = reason;
        transition.phase = TRANSITION_ROLLBACK;

        const auto & intent = transition.intent;
        if (!publish_state_events) {
            routes[{intent.target_model_id, intent.warm_executor_id}] =
                SERVER_WARM_TIER_MODEL_READY;
            routes[{intent.target_model_id, intent.gpu_executor_id}] =
                SERVER_WARM_TIER_MODEL_FAILED;
        } else if (route_is(
                       intent.target_model_id,
                       intent.warm_executor_id,
                       SERVER_WARM_TIER_MODEL_DRAINING)) {
            set_route_state(
                    intent.target_model_id,
                    intent.warm_executor_id,
                    SERVER_WARM_TIER_MODEL_READY);
        }
        if (publish_state_events) {
            set_route_state(
                    intent.target_model_id,
                    intent.gpu_executor_id,
                    SERVER_WARM_TIER_MODEL_FAILED);
        }
        if (!start_rollback_discard()) {
            return false;
        }
        return fail(reason);
    }

    bool start_emergency_discard(
            const server_warm_tier_command & command,
            const std::string & reason) {
        quarantine_all_commands(reason);
        routes[{command.model_id, command.executor_id}] =
            SERVER_WARM_TIER_MODEL_FAILED;
        if (bootstrap_active.has_value()
                && *bootstrap_active
                    == route_key{command.model_id, command.executor_id}) {
            bootstrap_active.reset();
            bootstrap_waiting.clear();
            activation_failed = true;
        }
        if (transition.active) {
            transition.failure = reason;
            transition.phase = TRANSITION_ROLLBACK;
            transition.replay_waiting.clear();
            transition.cleanup_waiting.clear();
            transition.old_owners.clear();
        }
        if (issue(
                    SERVER_WARM_TIER_COMMAND_DISCARD,
                    command.model_id,
                    {},
                    command.executor_id) == 0) {
            if (transition.active) {
                terminate_transition(reason + ":E_DISCARD_SUBMIT");
            }
            poisoned = true;
            return fail(reason + ":E_DISCARD_SUBMIT");
        }
        poisoned = true;
        return fail(reason);
    }

    bool terminate_after_unload_event_failure(
            const server_warm_tier_command & command,
            const std::string & reason) {
        quarantine_all_commands(reason);
        routes[{command.model_id, command.executor_id}] =
            SERVER_WARM_TIER_MODEL_FAILED;
        if (transition.active) {
            terminate_transition(reason);
        }
        poisoned = true;
        return fail(reason);
    }

    bool emit_executor_failure(
            const server_warm_tier_command & command,
            const std::string & detail) {
        server_warm_tier_event event;
        event.kind = SERVER_WARM_TIER_EVENT_EXECUTOR_FAILED;
        event.model_id = command.model_id;
        event.request_id = command.request_id;
        event.executor_id = command.executor_id;
        event.has_command = true;
        event.command_id = command.command_id;
        event.command_kind = command.kind;
        event.success = false;
        event.detail = detail;
        return emit(std::move(event));
    }

    bool strand_active_request(
            request_record & record,
            const std::string & executor_id,
            const std::string & reason) {
        record.snapshot.state = SERVER_WARM_TIER_REQUEST_STRANDED;
        server_warm_tier_event event;
        event.kind = SERVER_WARM_TIER_EVENT_REQUEST_STRANDED;
        event.model_id = record.snapshot.model_id;
        event.request_id = record.snapshot.request_id;
        event.executor_id = executor_id;
        event.has_request = true;
        event.request = snapshot(record);
        event.success = false;
        event.detail = reason;
        if (!emit(std::move(event))) {
            return false;
        }
        return fail(reason);
    }

    bool strand_execute_at_frontier(
            request_record & record,
            const std::string & executor_id,
            const std::string & reason) {
        strand_active_request(record, executor_id, reason);
        if (error != reason) {
            return false;
        }
        poisoned = true;
        const route_key failed_route{
            record.snapshot.model_id,
            executor_id,
        };
        if (transition.active
                && transition.phase == TRANSITION_REPLAYING
                && transition.intent.target_model_id == record.snapshot.model_id) {
            const bool result = begin_rollback(reason);
            routes[failed_route] = SERVER_WARM_TIER_MODEL_FAILED;
            return result;
        }
        if (transition.active
                && transition.phase == TRANSITION_DRAINING
                && transition.intent.source_model_id == record.snapshot.model_id) {
            set_route_state(
                    transition.intent.source_model_id,
                    transition.intent.gpu_executor_id,
                    SERVER_WARM_TIER_MODEL_FAILED);
            return terminate_transition(reason);
        }
        if (transition.active
                && transition.phase == TRANSITION_CLEANUP
                && transition.intent.target_model_id == record.snapshot.model_id
                && transition.intent.gpu_executor_id == executor_id) {
            const uint64_t failed_epoch = transition.epoch;
            quarantine_epoch_commands(failed_epoch, reason);
            for (auto & item : requests) {
                auto & sibling = item.second;
                if (sibling.snapshot.request_id != record.snapshot.request_id
                        && sibling.snapshot.model_id
                            == transition.intent.target_model_id
                        && sibling.snapshot.owner_id
                            == transition.intent.gpu_executor_id
                        && sibling.snapshot.state
                            == SERVER_WARM_TIER_REQUEST_ACTIVE) {
                    strand_active_request(sibling, executor_id, reason);
                }
            }
            set_route_state(
                    transition.intent.target_model_id,
                    transition.intent.gpu_executor_id,
                    SERVER_WARM_TIER_MODEL_FAILED);
            if (route_is(
                        transition.intent.target_model_id,
                        transition.intent.warm_executor_id,
                        SERVER_WARM_TIER_MODEL_DRAINING)) {
                set_route_state(
                        transition.intent.target_model_id,
                        transition.intent.warm_executor_id,
                        SERVER_WARM_TIER_MODEL_READY);
            }
            return terminate_transition(reason);
        }
        if (event_sink_failed) {
            routes[failed_route] = SERVER_WARM_TIER_MODEL_FAILED;
        } else if (!set_route_state(
                       record.snapshot.model_id,
                       executor_id,
                       SERVER_WARM_TIER_MODEL_FAILED)) {
            routes[failed_route] = SERVER_WARM_TIER_MODEL_FAILED;
        }
        return fail(reason);
    }

    bool executor_failure(
            const server_warm_tier_command & command,
            const std::string & detail,
            bool publish_event = true) {
        if (publish_event) {
            emit_executor_failure(command, detail);
        }
        poisoned = true;
        const std::string reason =
            std::string("E_EXECUTOR_FAILED:") + server_warm_tier_command_kind_name(command.kind)
            + ":" + detail;
        quarantine_all_commands(reason);

        if (command.kind == SERVER_WARM_TIER_COMMAND_LOAD
                && bootstrap_active.has_value()
                && *bootstrap_active
                    == route_key{command.model_id, command.executor_id}) {
            set_route_state(
                    command.model_id,
                    command.executor_id,
                    SERVER_WARM_TIER_MODEL_FAILED);
            bootstrap_active.reset();
            bootstrap_waiting.clear();
            activation_failed = true;
            return fail(reason);
        }

        switch (command.kind) {
            case SERVER_WARM_TIER_COMMAND_EXECUTE:
                {
                    auto found = requests.find(command.request_id);
                    if (found != requests.end()
                            && found->second.snapshot.state == SERVER_WARM_TIER_REQUEST_ACTIVE) {
                        return strand_execute_at_frontier(
                                found->second, command.executor_id, reason);
                    }
                    return fail(reason);
                }
            case SERVER_WARM_TIER_COMMAND_DRAIN:
            case SERVER_WARM_TIER_COMMAND_UNLOAD:
                if (transition.phase == TRANSITION_WARM_DRAINING
                        || transition.phase == TRANSITION_WARM_UNLOADING) {
                    set_route_state(
                            transition.intent.target_model_id,
                            transition.intent.warm_executor_id,
                            SERVER_WARM_TIER_MODEL_FAILED);
                    return terminate_transition(reason);
                }
                set_route_state(
                        transition.intent.source_model_id,
                        transition.intent.gpu_executor_id,
                        SERVER_WARM_TIER_MODEL_FAILED);
                return terminate_transition(reason);
            case SERVER_WARM_TIER_COMMAND_LOAD:
                if (transition.phase == TRANSITION_WARM_LOADING) {
                    set_route_state(
                            transition.intent.source_model_id,
                            transition.intent.warm_executor_id,
                            SERVER_WARM_TIER_MODEL_FAILED);
                    return terminate_transition(reason);
                }
                return begin_rollback(reason);
            case SERVER_WARM_TIER_COMMAND_REPLAY:
                return begin_rollback(reason);
            case SERVER_WARM_TIER_COMMAND_DISCARD:
                if (publish_event) {
                    set_route_state(
                            command.model_id,
                            command.executor_id,
                            SERVER_WARM_TIER_MODEL_FAILED);
                } else {
                    routes[{command.model_id, command.executor_id}] =
                        SERVER_WARM_TIER_MODEL_FAILED;
                }
                if (transition.active) {
                    terminate_transition(reason);
                }
                poisoned = true;
                return false;
            case SERVER_WARM_TIER_COMMAND_CLEANUP:
                if (!transition.active
                        || transition.phase != TRANSITION_CLEANUP
                        || transition.cleanup_waiting.count(command.request_id) == 0
                        || transition.old_owners.count(command.request_id) == 0
                        || transition.old_owners.at(command.request_id)
                            != command.executor_id) {
                    quarantine_epoch_commands(command.controller_epoch, reason);
                    set_route_state(
                            command.model_id,
                            command.executor_id,
                            SERVER_WARM_TIER_MODEL_FAILED);
                    if (transition.active) {
                        terminate_transition(reason);
                    }
                    poisoned = true;
                    return fail(reason);
                }
                if (transition.failure.empty()) {
                    transition.failure = reason;
                }
                std::set<std::string> failed_owners;
                for (const auto & owner : transition.old_owners) {
                    failed_owners.insert(owner.second);
                }
                for (const auto & owner : failed_owners) {
                    if (!set_route_state(
                                transition.intent.target_model_id,
                                owner,
                                SERVER_WARM_TIER_MODEL_FAILED)) {
                        routes[{
                            transition.intent.target_model_id,
                            owner,
                        }] = SERVER_WARM_TIER_MODEL_FAILED;
                    }
                }
                transition.cleanup_waiting.clear();
                return terminate_transition(transition.failure);
        }
        return fail(reason);
    }

    bool start_load() {
        transition.phase = TRANSITION_LOADING;
        const auto & intent = transition.intent;
        if (!set_route_state(
                    intent.target_model_id,
                    intent.gpu_executor_id,
                    SERVER_WARM_TIER_MODEL_LOADING)) {
            const std::string failure =
                error.empty() ? "E_LOAD_STATE" : error;
            routes[{intent.target_model_id, intent.gpu_executor_id}] =
                SERVER_WARM_TIER_MODEL_FAILED;
            poisoned = true;
            return terminate_transition(failure);
        }
        return issue(
                SERVER_WARM_TIER_COMMAND_LOAD,
                intent.target_model_id,
                {},
                intent.gpu_executor_id) != 0;
    }

    bool has_execute_for_model(const std::string & model_id) const {
        for (const auto & item : commands) {
            if (item.second.kind == SERVER_WARM_TIER_COMMAND_EXECUTE
                    && item.second.model_id == model_id) {
                return true;
            }
        }
        return false;
    }

    bool has_cleanup_for_model(const std::string & model_id) const {
        for (const auto & item : commands) {
            if (item.second.kind == SERVER_WARM_TIER_COMMAND_CLEANUP
                    && item.second.model_id == model_id) {
                return true;
            }
        }
        return false;
    }

    bool has_active_request_for_model(const std::string & model_id) const {
        for (const auto & item : requests) {
            if (item.second.snapshot.model_id == model_id
                    && item.second.snapshot.state == SERVER_WARM_TIER_REQUEST_ACTIVE) {
                return true;
            }
        }
        return false;
    }

    bool maybe_start_source_drain() {
        if (!transition.active || transition.phase != TRANSITION_DRAINING) {
            return true;
        }
        const auto & intent = transition.intent;
        if (has_active_request_for_model(intent.source_model_id)
                || has_execute_for_model(intent.source_model_id)
                || has_cleanup_for_model(intent.source_model_id)) {
            return true;
        }
        return issue(
                SERVER_WARM_TIER_COMMAND_DRAIN,
                intent.source_model_id,
                {},
                intent.gpu_executor_id) != 0;
    }

    size_t occupied_credit_count(const std::string & executor_id) const {
        size_t result = 0;
        for (const auto & item : commands) {
            if ((item.second.kind == SERVER_WARM_TIER_COMMAND_EXECUTE
                    || item.second.kind == SERVER_WARM_TIER_COMMAND_REPLAY
                    || item.second.kind == SERVER_WARM_TIER_COMMAND_CLEANUP)
                    && item.second.executor_id == executor_id) {
                result++;
            }
        }
        return result;
    }

    bool has_executor_credit(
            const std::string & executor_id,
            size_t requested = 1) const {
        const auto policy = executor_policies.find(executor_id);
        if (policy == executor_policies.end()) {
            return false;
        }
        const size_t occupied = occupied_credit_count(executor_id);
        return occupied <= policy->second.credits
            && requested <= policy->second.credits - occupied;
    }

    std::optional<std::string> select_ready_executor(
            const std::string & model_id) const {
        const server_warm_tier_executor_policy * selected = nullptr;
        for (const auto & item : executor_policies) {
            const auto & policy = item.second;
            if (!route_is(
                        model_id,
                        policy.executor_id,
                        SERVER_WARM_TIER_MODEL_READY)
                    || occupied_credit_count(policy.executor_id) >= policy.credits) {
                continue;
            }
            if (selected == nullptr
                    || policy.order < selected->order
                    || (policy.order == selected->order
                        && policy.executor_id < selected->executor_id)) {
                selected = &policy;
            }
        }
        if (selected == nullptr) {
            return std::nullopt;
        }
        return selected->executor_id;
    }

    bool has_ready_compatible_route(const std::string & model_id) const {
        for (const auto & item : executor_policies) {
            if (route_is(
                        model_id,
                        item.first,
                        SERVER_WARM_TIER_MODEL_READY)) {
                return true;
            }
        }
        return false;
    }

    bool has_non_gpu_compatible_route(const std::string & model_id) const {
        for (const auto & item : executor_policies) {
            if (item.second.role != SERVER_WARM_TIER_EXECUTOR_GPU
                    && routes.count({model_id, item.first}) != 0) {
                return true;
            }
        }
        return false;
    }

    bool remove_pending_request(
            const std::string & model_id,
            const std::string & request_id) {
        auto found = pending_by_model.find(model_id);
        if (found == pending_by_model.end()) {
            return false;
        }
        auto & queue = found->second;
        auto request = std::find(queue.begin(), queue.end(), request_id);
        if (request == queue.end()) {
            return false;
        }
        queue.erase(request);
        if (queue.empty()) {
            pending_by_model.erase(found);
        }
        return true;
    }

    bool dispatch_queued_request(
            request_record & record,
            const std::string & executor_id,
            int32_t total_output_tokens) {
        if (finalizing) {
            return fail("E_FINALIZING");
        }
        auto & request = record.snapshot;
        if (request.state != SERVER_WARM_TIER_REQUEST_QUEUED) {
            return fail("E_REQUEST_NOT_QUEUED");
        }
        if (total_output_tokens <= 0) {
            return fail("E_OUTPUT_BUDGET");
        }
        if (!has_executor_credit(executor_id)) {
            return fail("E_EXECUTOR_CREDIT");
        }
        if (!route_is(
                    request.model_id,
                    executor_id,
                    SERVER_WARM_TIER_MODEL_READY)) {
            return fail("E_ROUTE_NOT_READY");
        }
        request.owner_id = executor_id;
        request.ownership_epoch = 1;
        request.state = SERVER_WARM_TIER_REQUEST_ACTIVE;
        record.total_output_tokens = total_output_tokens;

        const auto request_snapshot = snapshot(record);
        if (issue(
                    SERVER_WARM_TIER_COMMAND_EXECUTE,
                    request.model_id,
                    request.request_id,
                    executor_id,
                    &request_snapshot,
                    1) == 0) {
            if (request.state != SERVER_WARM_TIER_REQUEST_ACTIVE) {
                return false;
            }
            request.owner_id.clear();
            request.ownership_epoch = 0;
            request.state = SERVER_WARM_TIER_REQUEST_QUEUED;
            return false;
        }
        return true;
    }

    bool begin_switch(
            const server_warm_tier_switch_intent & intent,
            bool emit_submission = true) {
        if (finalizing) {
            return fail("E_FINALIZING");
        }
        if (transition.active) {
            return fail("E_TRANSITION_BUSY");
        }
        if (intent.intent_id.empty()
                || intent.source_model_id.empty()
                || intent.target_model_id.empty()
                || intent.source_model_id == intent.target_model_id
                || intent.gpu_executor_id.empty()) {
            return fail("E_INTENT_INVALID");
        }
        if (has_intent_sequence
                && intent.sequence <= last_intent_sequence) {
            return fail("E_INTENT_SEQUENCE");
        }
        if (!route_is(
                    intent.source_model_id,
                    intent.gpu_executor_id,
                    SERVER_WARM_TIER_MODEL_READY)
                || !route_is(
                    intent.target_model_id,
                    intent.gpu_executor_id,
                    SERVER_WARM_TIER_MODEL_ABSENT)) {
            return fail("E_SWITCH_ROUTE_STATE");
        }
        if (intent.warm_executor_id.empty()) {
            if (has_active_request_for_model(intent.target_model_id)) {
                return fail("E_NO_WARM_ACTIVE_TARGET");
            }
        } else if (!route_is(
                           intent.target_model_id,
                           intent.warm_executor_id,
                           SERVER_WARM_TIER_MODEL_READY)
                || !route_is(
                           intent.source_model_id,
                           intent.warm_executor_id,
                           SERVER_WARM_TIER_MODEL_ABSENT)) {
            return fail("E_SWITCH_ROUTE_STATE");
        }

        epoch++;
        transition = {};
        transition.active = true;
        transition.epoch = epoch;
        transition.phase = TRANSITION_DRAINING;
        transition.intent = intent;
        last_intent_sequence = intent.sequence;
        has_intent_sequence = true;

        if (emit_submission) {
            server_warm_tier_event event;
            event.kind = SERVER_WARM_TIER_EVENT_SWITCH_INTENT_SUBMITTED;
            event.model_id = intent.target_model_id;
            event.executor_id = intent.gpu_executor_id;
            event.detail = intent.intent_id;
            if (!emit(std::move(event))) {
                transition = {};
                return false;
            }
        }
        if (!set_route_state(
                    intent.source_model_id,
                    intent.gpu_executor_id,
                    SERVER_WARM_TIER_MODEL_DRAINING)) {
            transition = {};
            return false;
        }
        return maybe_start_source_drain();
    }

    bool derive_and_begin_switch(
            uint64_t sequence,
            const std::string & intent_id,
            const std::string & source_model_id,
            const std::string & target_model_id,
            bool emit_submission = true) {
        const server_warm_tier_executor_policy * gpu = nullptr;
        std::string derived_source_model_id;
        for (const auto & item : executor_policies) {
            const auto & policy = item.second;
            if (policy.role != SERVER_WARM_TIER_EXECUTOR_GPU
                    || routes.count({target_model_id, policy.executor_id}) == 0
                    || !route_is(
                        target_model_id,
                        policy.executor_id,
                        SERVER_WARM_TIER_MODEL_ABSENT)) {
                continue;
            }
            std::string candidate_source = source_model_id;
            if (candidate_source.empty()) {
                for (const auto & route : routes) {
                    if (route.first.second == policy.executor_id
                            && route.first.first != target_model_id
                            && route.second == SERVER_WARM_TIER_MODEL_READY) {
                        candidate_source = route.first.first;
                        break;
                    }
                }
            }
            if (candidate_source.empty()
                    || !route_is(
                        candidate_source,
                        policy.executor_id,
                        SERVER_WARM_TIER_MODEL_READY)) {
                continue;
            }
            if (gpu == nullptr || policy.order < gpu->order) {
                gpu = &policy;
                derived_source_model_id = candidate_source;
            }
        }
        if (gpu == nullptr) {
            return fail("E_SWITCH_GPU_ROUTE");
        }

        bool has_warm_compatibility = false;
        const server_warm_tier_executor_policy * warm = nullptr;
        for (const auto & item : executor_policies) {
            const auto & policy = item.second;
            if (policy.role == SERVER_WARM_TIER_EXECUTOR_GPU
                    || routes.count({target_model_id, policy.executor_id}) == 0) {
                continue;
            }
            has_warm_compatibility = true;
            if (routes.count({derived_source_model_id, policy.executor_id}) == 0
                    || !route_is(
                        target_model_id,
                        policy.executor_id,
                        SERVER_WARM_TIER_MODEL_READY)
                    || !route_is(
                        derived_source_model_id,
                        policy.executor_id,
                        SERVER_WARM_TIER_MODEL_ABSENT)) {
                continue;
            }
            if (warm == nullptr || policy.order < warm->order) {
                warm = &policy;
            }
        }
        if (has_warm_compatibility && warm == nullptr) {
            return fail("E_SWITCH_WARM_ROUTE");
        }

        server_warm_tier_switch_intent intent;
        intent.sequence = sequence;
        intent.intent_id = intent_id;
        intent.source_model_id = derived_source_model_id;
        intent.target_model_id = target_model_id;
        intent.gpu_executor_id = gpu->executor_id;
        if (warm != nullptr) {
            intent.warm_executor_id = warm->executor_id;
        }
        return begin_switch(intent, emit_submission);
    }

    bool has_demand_for_model(const std::string & model_id) const {
        for (const auto & item : requests) {
            if (item.second.snapshot.model_id == model_id
                    && (item.second.snapshot.state
                        == SERVER_WARM_TIER_REQUEST_QUEUED
                        || item.second.snapshot.state
                        == SERVER_WARM_TIER_REQUEST_ACTIVE)) {
                return true;
            }
        }
        return false;
    }

    uint64_t next_intent_sequence() const {
        return has_accepted_intent_sequence
            ? last_accepted_intent_sequence + 1 : 0;
    }

    bool accept_model_switch(
            uint64_t sequence,
            const std::string & intent_id,
            const std::string & source_model_id,
            const std::string & target_model_id) {
        if (finalizing) {
            return fail("E_FINALIZING");
        }
        if (!promotion_enabled) {
            return fail("E_PROMOTION_DISABLED");
        }
        if (intent_id.empty() || target_model_id.empty()
                || source_model_id == target_model_id
                || (has_accepted_intent_sequence
                    && sequence <= last_accepted_intent_sequence)) {
            return fail("E_INTENT_INVALID");
        }

        server_warm_tier_switch_intent requested;
        requested.sequence = sequence;
        requested.intent_id = intent_id;
        requested.source_model_id = source_model_id;
        requested.target_model_id = target_model_id;

        if (transition.active) {
            server_warm_tier_event event;
            event.model_id = target_model_id;
            event.detail = intent_id;
            if (!pending_intent.has_value()) {
                event.kind = SERVER_WARM_TIER_EVENT_SWITCH_INTENT_QUEUED;
                pending_intent = requested;
            } else if (!has_demand_for_model(
                           pending_intent->target_model_id)) {
                event.kind = SERVER_WARM_TIER_EVENT_SWITCH_INTENT_COALESCED;
                event.detail =
                    pending_intent->intent_id + ":" + requested.intent_id;
                pending_intent = requested;
            } else {
                event.kind = SERVER_WARM_TIER_EVENT_SWITCH_INTENT_COALESCED;
                event.detail =
                    requested.intent_id + ":preserved:" + pending_intent->intent_id;
            }
            if (!emit(std::move(event))) {
                return false;
            }
        } else if (!derive_and_begin_switch(
                       sequence,
                       intent_id,
                       source_model_id,
                       target_model_id)) {
            return false;
        }
        has_accepted_intent_sequence = true;
        last_accepted_intent_sequence = sequence;
        return true;
    }

    bool maybe_propose_promotion(const std::string & target_model_id) {
        if (!promotion_enabled) {
            return true;
        }
        if (transition.active
                && transition.intent.target_model_id == target_model_id) {
            return true;
        }
        if (pending_intent.has_value()
                && pending_intent->target_model_id == target_model_id) {
            return true;
        }
        for (const auto & item : executor_policies) {
            if (item.second.role == SERVER_WARM_TIER_EXECUTOR_GPU
                    && route_is(
                        target_model_id,
                        item.first,
                        SERVER_WARM_TIER_MODEL_READY)) {
                return true;
            }
        }
        const uint64_t sequence = next_intent_sequence();
        return accept_model_switch(
            sequence,
            "auto-" + std::to_string(sequence) + "-" + target_model_id,
            {},
            target_model_id);
    }

    bool enqueue_record(
            const std::string & request_id,
            const std::string & model_id,
            std::vector<llama_token> prompt_tokens,
            uint64_t arrival_order,
            int32_t max_output_tokens,
            bool scheduled) {
        if (finalizing) {
            return fail("E_FINALIZING");
        }
        if (request_id.empty() || model_id.empty() || prompt_tokens.empty()
                || prompt_tokens.size() > 1024 * 1024
                || std::any_of(
                    prompt_tokens.begin(),
                    prompt_tokens.end(),
                    [](llama_token token) { return token < 0; })
                || (scheduled && max_output_tokens <= 0)) {
            return fail("E_REQUEST_INVALID");
        }
        if (scheduled
                && ((!has_scheduled_arrival && arrival_order != 0)
                    || (has_scheduled_arrival
                        && arrival_order != last_scheduled_arrival + 1))) {
            return fail("E_ARRIVAL_ORDER");
        }
        if (requests.count(request_id) != 0) {
            return fail("E_REQUEST_DUPLICATE");
        }
        if (scheduled) {
            bool compatible = false;
            for (const auto & item : executor_policies) {
                compatible |= routes.count({model_id, item.first}) != 0;
            }
            if (!compatible) {
                return fail("E_REQUEST_NO_COMPATIBLE_ROUTE");
            }
        }

        request_record record;
        record.snapshot.request_id = request_id;
        record.snapshot.model_id = model_id;
        record.snapshot.prompt_tokens = std::move(prompt_tokens);
        record.snapshot.position = record.snapshot.prompt_tokens.size();
        record.snapshot.state = SERVER_WARM_TIER_REQUEST_QUEUED;
        record.arrival_sequence =
            scheduled ? arrival_order : next_arrival_sequence++;
        record.total_output_tokens = max_output_tokens;
        record.scheduled = scheduled;
        auto inserted = requests.emplace(request_id, std::move(record));

        server_warm_tier_event event;
        event.kind = SERVER_WARM_TIER_EVENT_REQUEST_ARRIVED;
        event.model_id = model_id;
        event.request_id = request_id;
        event.has_request = true;
        event.request = snapshot(inserted.first->second);
        if (!emit(std::move(event))) {
            requests.erase(inserted.first);
            return false;
        }
        if (scheduled) {
            has_scheduled_arrival = true;
            last_scheduled_arrival = arrival_order;
            pending_by_model[model_id].push_back(request_id);
            if (!maybe_propose_promotion(model_id)
                    || !schedule_pending()) {
                const std::string schedule_error = error;
                auto found = requests.find(request_id);
                if (found != requests.end()
                        && found->second.snapshot.state
                            == SERVER_WARM_TIER_REQUEST_QUEUED) {
                    remove_pending_request(model_id, request_id);
                    found->second.snapshot.state =
                        SERVER_WARM_TIER_REQUEST_STRANDED;
                    server_warm_tier_event stranded;
                    stranded.kind = SERVER_WARM_TIER_EVENT_REQUEST_STRANDED;
                    stranded.model_id = model_id;
                    stranded.request_id = request_id;
                    stranded.has_request = true;
                    stranded.request = snapshot(found->second);
                    stranded.success = false;
                    stranded.detail = schedule_error;
                    if (!emit(std::move(stranded))) {
                        return false;
                    }
                }
                error = schedule_error;
                return false;
            }
        }
        return true;
    }

    bool schedule_pending() {
        if (!activated) {
            return true;
        }
        if (finalizing) {
            return maybe_finish_finalization();
        }
        while (true) {
            request_record * selected_request = nullptr;
            std::string selected_model;
            std::string selected_executor;
            for (auto & item : pending_by_model) {
                if (item.second.empty()) {
                    continue;
                }
                auto found = requests.find(item.second.front());
                if (found == requests.end()
                        || found->second.snapshot.state
                            != SERVER_WARM_TIER_REQUEST_QUEUED) {
                    return fail("E_PENDING_STATE");
                }
                const auto executor =
                    select_ready_executor(found->second.snapshot.model_id);
                if (!executor.has_value()) {
                    continue;
                }
                if (selected_request == nullptr
                        || found->second.arrival_sequence
                            < selected_request->arrival_sequence) {
                    selected_request = &found->second;
                    selected_model = item.first;
                    selected_executor = *executor;
                }
            }
            if (selected_request == nullptr) {
                break;
            }
            const std::string request_id = selected_request->snapshot.request_id;
            const int32_t budget = selected_request->total_output_tokens;
            if (!dispatch_queued_request(
                        *selected_request, selected_executor, budget)) {
                remove_pending_request(selected_model, request_id);
                return false;
            }
            remove_pending_request(selected_model, request_id);
        }

        if (!transition.active && pending_intent.has_value()) {
            const auto intent = *pending_intent;
            if (has_demand_for_model(intent.target_model_id)) {
                if (!derive_and_begin_switch(
                            intent.sequence,
                            intent.intent_id,
                            intent.source_model_id,
                            intent.target_model_id,
                            false)) {
                    return false;
                }
            } else {
                server_warm_tier_event event;
                event.kind = SERVER_WARM_TIER_EVENT_SWITCH_INTENT_COALESCED;
                event.model_id = intent.target_model_id;
                event.detail = intent.intent_id + ":dropped-no-demand";
                if (!emit(std::move(event))) {
                    return false;
                }
            }
            pending_intent.reset();
        }
        if (transition.active) {
            return true;
        }

        const request_record * oldest = nullptr;
        for (const auto & item : pending_by_model) {
            if (item.second.empty()) {
                continue;
            }
            auto found = requests.find(item.second.front());
            if (found != requests.end()
                    && (oldest == nullptr
                        || found->second.arrival_sequence
                            < oldest->arrival_sequence)) {
                oldest = &found->second;
            }
        }
        if (oldest != nullptr
                && !has_ready_compatible_route(oldest->snapshot.model_id)) {
            return maybe_propose_promotion(oldest->snapshot.model_id);
        }
        return true;
    }

    bool issue_replays_at_frontier() {
        const auto & intent = transition.intent;
        std::vector<std::string> replay_ids;
        for (const auto & item : requests) {
            const auto & request = item.second.snapshot;
            if (request.state == SERVER_WARM_TIER_REQUEST_ACTIVE
                    && request.model_id == intent.target_model_id
                    && request.owner_id != intent.gpu_executor_id) {
                replay_ids.push_back(item.first);
            }
        }

        if (replay_ids.empty()) {
            if (!set_route_state(
                        intent.target_model_id,
                        intent.gpu_executor_id,
                        SERVER_WARM_TIER_MODEL_READY)) {
                const std::string failure =
                    error.empty() ? "E_REPLAY_READY_STATE" : error;
                return begin_rollback(failure, !event_sink_failed);
            }
            return start_warm_rotation();
        }

        if (!has_executor_credit(
                    intent.gpu_executor_id,
                    replay_ids.size())) {
            return begin_rollback("E_REPLAY_CREDIT");
        }

        for (const auto & request_id : replay_ids) {
            const auto & request = requests.at(request_id);
            const auto request_snapshot = snapshot(request);
            transition.replay_waiting.insert(request_id);
            transition.old_owners.emplace(request_id, request_snapshot.owner_id);
            if (issue(
                        SERVER_WARM_TIER_COMMAND_REPLAY,
                        intent.target_model_id,
                        request_id,
                        intent.gpu_executor_id,
                        &request_snapshot) == 0) {
                return false;
            }
        }
        return true;
    }

    bool start_replay() {
        transition.phase = TRANSITION_REPLAYING;
        const auto & intent = transition.intent;
        if (!set_route_state(
                    intent.target_model_id,
                    intent.gpu_executor_id,
                    SERVER_WARM_TIER_MODEL_REPLAYING)) {
            const std::string failure =
                error.empty() ? "E_REPLAY_STATE" : error;
            return begin_rollback(failure, !event_sink_failed);
        }
        if (!set_route_state(
                    intent.target_model_id,
                    intent.warm_executor_id,
                    SERVER_WARM_TIER_MODEL_DRAINING)) {
            const std::string failure =
                error.empty() ? "E_WARM_DRAIN_STATE" : error;
            return begin_rollback(failure, !event_sink_failed);
        }
        if (has_execute_for_model(intent.target_model_id)
                || has_cleanup_for_model(intent.target_model_id)) {
            return true;
        }
        return issue_replays_at_frontier();
    }

    bool commit_replays() {
        const auto & intent = transition.intent;
        struct ownership_before {
            std::string request_id;
            std::string owner_id;
            uint64_t epoch;
        };
        std::vector<ownership_before> old;
        old.reserve(transition.old_owners.size());

        const auto abort_commit = [&]() {
            const std::string failure =
                error.empty() ? "E_OWNERSHIP_COMMIT" : error;
            for (const auto & before : old) {
                auto found = requests.find(before.request_id);
                if (found != requests.end()) {
                    found->second.snapshot.owner_id = before.owner_id;
                    found->second.snapshot.ownership_epoch = before.epoch;
                }
            }
            const bool result =
                begin_rollback(failure, !event_sink_failed);
            routes[{intent.target_model_id, intent.gpu_executor_id}] =
                SERVER_WARM_TIER_MODEL_FAILED;
            routes[{intent.target_model_id, intent.warm_executor_id}] =
                SERVER_WARM_TIER_MODEL_READY;
            return result;
        };

        for (const auto & item : transition.old_owners) {
            auto found = requests.find(item.first);
            if (found == requests.end()
                    || found->second.snapshot.state != SERVER_WARM_TIER_REQUEST_ACTIVE
                    || found->second.snapshot.owner_id != item.second
                    || found->second.total_output_tokens <= 0
                    || found->second.snapshot.committed_output_tokens.size()
                        >= static_cast<size_t>(found->second.total_output_tokens)) {
                return begin_rollback("E_REPLAY_FRONTIER_CHANGED:" + item.first);
            }
            old.push_back({
                item.first,
                found->second.snapshot.owner_id,
                found->second.snapshot.ownership_epoch,
            });
        }

        for (const auto & before : old) {
            auto & request = requests.at(before.request_id).snapshot;
            request.owner_id = intent.gpu_executor_id;
            request.ownership_epoch++;
        }

        if (!set_route_state(
                    intent.target_model_id,
                    intent.gpu_executor_id,
                    SERVER_WARM_TIER_MODEL_READY)) {
            return abort_commit();
        }

        for (const auto & before : old) {
            const auto request_snapshot = snapshot(requests.at(before.request_id));
            server_warm_tier_event event;
            event.kind = SERVER_WARM_TIER_EVENT_OWNERSHIP_COMMIT;
            event.model_id = request_snapshot.model_id;
            event.request_id = request_snapshot.request_id;
            event.executor_id = intent.gpu_executor_id;
            event.has_request = true;
            event.request = request_snapshot;
            event.old_owner_id = before.owner_id;
            event.new_owner_id = request_snapshot.owner_id;
            event.old_ownership_epoch = before.epoch;
            event.new_ownership_epoch = request_snapshot.ownership_epoch;
            event.publication_index = request_snapshot.publication_index;
            if (!emit(std::move(event))) {
                return abort_commit();
            }
        }
        {
            server_warm_tier_event event;
            event.kind = SERVER_WARM_TIER_EVENT_OWNERSHIP_COMMIT_COMPLETE;
            event.model_id = intent.target_model_id;
            event.executor_id = intent.gpu_executor_id;
            event.detail = std::to_string(old.size());
            if (!emit(std::move(event))) {
                return abort_commit();
            }
        }
        transition.phase = TRANSITION_CLEANUP;
        for (const auto & before : old) {
            transition.cleanup_waiting.insert(before.request_id);
        }
        for (const auto & before : old) {
            auto & record = requests.at(before.request_id);
            const auto request_snapshot = snapshot(record);
            if (issue(
                        SERVER_WARM_TIER_COMMAND_EXECUTE,
                        request_snapshot.model_id,
                        request_snapshot.request_id,
                        intent.gpu_executor_id,
                        &request_snapshot,
                        1) == 0) {
                return false;
            }
        }

        return start_next_cleanup();
    }

    bool start_warm_rotation() {
        const auto & intent = transition.intent;
        transition.phase = TRANSITION_WARM_DRAINING;
        return issue(
                SERVER_WARM_TIER_COMMAND_DRAIN,
                intent.target_model_id,
                {},
                intent.warm_executor_id) != 0;
    }

    bool start_next_cleanup() {
        if (transition.cleanup_waiting.empty()) {
            if (!transition.failure.empty()) {
                return terminate_transition(transition.failure);
            }
            return start_warm_rotation();
        }
        const std::string request_id = *transition.cleanup_waiting.begin();
        const auto owner = transition.old_owners.find(request_id);
        const auto request = requests.find(request_id);
        if (owner == transition.old_owners.end() || request == requests.end()) {
            return terminate_transition("E_CLEANUP_REQUEST:" + request_id);
        }
        const auto request_snapshot = snapshot(request->second);
        return issue(
                SERVER_WARM_TIER_COMMAND_CLEANUP,
                transition.intent.target_model_id,
                request_id,
                owner->second,
                &request_snapshot) != 0;
    }

    bool replay_done(
            const server_warm_tier_command & command,
            const server_warm_tier_result & result) {
        if (!result.has_replay_snapshot) {
            return begin_rollback("E_REPLAY_SNAPSHOT_MISSING:" + command.request_id);
        }
        auto found = requests.find(command.request_id);
        if (found == requests.end()) {
            return begin_rollback("E_REPLAY_REQUEST_MISSING:" + command.request_id);
        }
        const auto current = snapshot(found->second);
        if (!same_replay_frontier(current, command.request)
                || !same_replay_frontier(current, result.replay_snapshot)) {
            return begin_rollback("E_REPLAY_FRONTIER_MISMATCH:" + command.request_id);
        }
        transition.replay_waiting.erase(command.request_id);
        if (transition.replay_waiting.empty()) {
            return commit_replays();
        }
        return true;
    }

    bool command_success(
            const server_warm_tier_command & command,
            const server_warm_tier_result & result) {
        if (command.kind == SERVER_WARM_TIER_COMMAND_LOAD
                && bootstrap_active.has_value()
                && *bootstrap_active
                    == route_key{command.model_id, command.executor_id}) {
            if (!set_route_state(
                        command.model_id,
                        command.executor_id,
                        SERVER_WARM_TIER_MODEL_READY)) {
                const std::string failure =
                    error.empty() ? "E_BOOTSTRAP_READY_STATE" : error;
                return start_emergency_discard(command, failure);
            }
            bootstrap_waiting.pop_front();
            bootstrap_active.reset();
            return issue_next_bootstrap();
        }

        const auto & intent = transition.intent;
        switch (command.kind) {
            case SERVER_WARM_TIER_COMMAND_EXECUTE:
                {
                    auto found = requests.find(command.request_id);
                    if (found == requests.end()
                            || found->second.snapshot.state != SERVER_WARM_TIER_REQUEST_ACTIVE) {
                        return fail("E_EXECUTE_REQUEST");
                    }
                    auto & record = found->second;
                    if (result.publications.empty()
                            || result.publications.size()
                                > static_cast<size_t>(command.max_output_tokens)
                            || result.publications.size()
                                != static_cast<size_t>(command.max_output_tokens)) {
                        return strand_execute_at_frontier(
                                record, command.executor_id, "E_EXECUTE_BUDGET");
                    }
                    for (size_t i = 0; i < result.publications.size(); ++i) {
                        const auto & publication = result.publications[i];
                        if (publication.owner_id != command.executor_id
                                || publication.ownership_epoch != record.snapshot.ownership_epoch
                                || publication.publication_index != record.publications.size() + i
                                || publication.position
                                    != record.snapshot.position + static_cast<int64_t>(i)
                                || publication.token < 0) {
                            return strand_execute_at_frontier(
                                    record, command.executor_id, "E_EXECUTE_PUBLICATION");
                        }
                    }
                    if (record.total_output_tokens <= 0
                            || record.snapshot.committed_output_tokens.size()
                                + result.publications.size()
                                > static_cast<size_t>(record.total_output_tokens)
                            || result.request_complete
                                != (record.snapshot.committed_output_tokens.size()
                                    + result.publications.size()
                                    == static_cast<size_t>(
                                        record.total_output_tokens))) {
                        return strand_execute_at_frontier(
                            record, command.executor_id, "E_EXECUTE_COMPLETION");
                    }
                    const size_t publications_before = record.publications.size();
                    const size_t output_before =
                        record.snapshot.committed_output_tokens.size();
                    const int64_t position_before = record.snapshot.position;
                    const auto rollback_publications = [&]() {
                        record.publications.resize(publications_before);
                        record.snapshot.committed_output_tokens.resize(output_before);
                        record.snapshot.position = position_before;
                        record.snapshot.state = SERVER_WARM_TIER_REQUEST_ACTIVE;
                    };
                    for (const auto & publication : result.publications) {
                        record.publications.push_back(publication);
                        record.snapshot.committed_output_tokens.push_back(publication.token);
                        record.snapshot.position++;

                        server_warm_tier_event event;
                        event.kind = SERVER_WARM_TIER_EVENT_TOKEN_COMMITTED;
                        event.model_id = command.model_id;
                        event.request_id = command.request_id;
                        event.executor_id = command.executor_id;
                        event.has_command = true;
                        event.command_id = command.command_id;
                        event.command_kind = command.kind;
                        event.has_request = true;
                        event.request = snapshot(record);
                        event.publication_index = publication.publication_index;
                        if (!emit(std::move(event))) {
                            rollback_publications();
                            return false;
                        }
                    }
                    if (result.request_complete) {
                        record.snapshot.state = SERVER_WARM_TIER_REQUEST_COMPLETED;
                        server_warm_tier_event event;
                        event.kind = SERVER_WARM_TIER_EVENT_REQUEST_COMPLETED;
                        event.model_id = command.model_id;
                        event.request_id = command.request_id;
                        event.executor_id = command.executor_id;
                        event.has_command = true;
                        event.command_id = command.command_id;
                        event.command_kind = command.kind;
                        event.has_request = true;
                        event.request = snapshot(record);
                        if (!emit(std::move(event))) {
                            rollback_publications();
                            return false;
                        }
                        const auto completed = snapshot(record);
                        return issue(
                                SERVER_WARM_TIER_COMMAND_CLEANUP,
                                command.model_id,
                                command.request_id,
                                command.executor_id,
                                &completed) != 0;
                    }
                    if (transition.active
                            && transition.phase == TRANSITION_REPLAYING
                            && transition.intent.target_model_id == command.model_id) {
                        if (!has_execute_for_model(command.model_id)) {
                            return issue_replays_at_frontier();
                        }
                        return true;
                    }
                    const auto next = snapshot(record);
                    return issue(
                            SERVER_WARM_TIER_COMMAND_EXECUTE,
                            command.model_id,
                            command.request_id,
                            command.executor_id,
                            &next,
                            1) != 0;
                }
            case SERVER_WARM_TIER_COMMAND_DRAIN:
                if (transition.phase == TRANSITION_WARM_DRAINING) {
                    transition.phase = TRANSITION_WARM_UNLOADING;
                    return issue(
                            SERVER_WARM_TIER_COMMAND_UNLOAD,
                            intent.target_model_id,
                            {},
                            intent.warm_executor_id) != 0;
                }
                transition.phase = TRANSITION_UNLOADING;
                return issue(
                        SERVER_WARM_TIER_COMMAND_UNLOAD,
                        intent.source_model_id,
                        {},
                        intent.gpu_executor_id) != 0;
            case SERVER_WARM_TIER_COMMAND_UNLOAD:
                if (transition.phase == TRANSITION_WARM_UNLOADING) {
                    if (!set_route_state(
                                intent.target_model_id,
                                intent.warm_executor_id,
                                SERVER_WARM_TIER_MODEL_ABSENT)) {
                        const std::string failure =
                            error.empty() ? "E_WARM_UNLOAD_STATE" : error;
                        return terminate_after_unload_event_failure(
                            command, failure);
                    }
                    transition.phase = TRANSITION_WARM_LOADING;
                    if (!set_route_state(
                                intent.source_model_id,
                                intent.warm_executor_id,
                                SERVER_WARM_TIER_MODEL_LOADING)) {
                        const std::string failure =
                            error.empty() ? "E_WARM_LOAD_STATE" : error;
                        routes[{
                            intent.source_model_id,
                            intent.warm_executor_id,
                        }] = SERVER_WARM_TIER_MODEL_FAILED;
                        poisoned = true;
                        return terminate_transition(failure);
                    }
                    return issue(
                            SERVER_WARM_TIER_COMMAND_LOAD,
                            intent.source_model_id,
                            {},
                            intent.warm_executor_id) != 0;
                }
                if (!set_route_state(
                            intent.source_model_id,
                            intent.gpu_executor_id,
                            SERVER_WARM_TIER_MODEL_ABSENT)) {
                    const std::string failure =
                        error.empty() ? "E_UNLOAD_STATE" : error;
                    return terminate_after_unload_event_failure(
                        command, failure);
                }
                return start_load();
            case SERVER_WARM_TIER_COMMAND_LOAD:
                if (transition.phase == TRANSITION_WARM_LOADING) {
                    if (!set_route_state(
                                intent.source_model_id,
                                intent.warm_executor_id,
                                SERVER_WARM_TIER_MODEL_READY)) {
                        const std::string failure =
                            error.empty() ? "E_WARM_READY_STATE" : error;
                        return start_emergency_discard(command, failure);
                    }
                    return terminate_transition(transition.failure);
                }
                if (intent.warm_executor_id.empty()) {
                    if (!set_route_state(
                                intent.target_model_id,
                                intent.gpu_executor_id,
                                SERVER_WARM_TIER_MODEL_READY)) {
                        const std::string failure =
                            error.empty() ? "E_GPU_READY_STATE" : error;
                        return start_emergency_discard(command, failure);
                    }
                    return terminate_transition({});
                }
                return start_replay();
            case SERVER_WARM_TIER_COMMAND_REPLAY:
                return replay_done(command, result);
            case SERVER_WARM_TIER_COMMAND_DISCARD:
                if (event_sink_failed) {
                    routes[{command.model_id, command.executor_id}] =
                        SERVER_WARM_TIER_MODEL_FAILED;
                } else {
                    set_route_state(
                            command.model_id,
                            command.executor_id,
                            SERVER_WARM_TIER_MODEL_FAILED);
                }
                if (bootstrap_active.has_value()
                        && *bootstrap_active
                            == route_key{
                                command.model_id,
                                command.executor_id,
                            }) {
                    bootstrap_active.reset();
                    bootstrap_waiting.clear();
                    activation_failed = true;
                }
                if (transition.active) {
                    terminate_transition(transition.failure);
                }
                poisoned = true;
                return false;
            case SERVER_WARM_TIER_COMMAND_CLEANUP:
                if (!transition.active
                        || transition.phase != TRANSITION_CLEANUP
                        || transition.cleanup_waiting.count(command.request_id) == 0
                        || transition.old_owners.count(command.request_id) == 0
                        || transition.old_owners.at(command.request_id)
                            != command.executor_id) {
                    auto request = requests.find(command.request_id);
                    if (request == requests.end()
                            || request->second.snapshot.state
                                != SERVER_WARM_TIER_REQUEST_COMPLETED) {
                        return fail("E_CLEANUP_REQUEST:" + command.request_id);
                    }
                    if (transition.active
                            && transition.phase == TRANSITION_DRAINING
                            && transition.intent.source_model_id == command.model_id) {
                        return maybe_start_source_drain();
                    }
                    if (transition.active
                            && transition.phase == TRANSITION_REPLAYING
                            && transition.intent.target_model_id == command.model_id
                            && !has_execute_for_model(command.model_id)
                            && !has_cleanup_for_model(command.model_id)) {
                        return issue_replays_at_frontier();
                    }
                    return true;
                }
                transition.cleanup_waiting.erase(command.request_id);
                if (transition.cleanup_waiting.empty()) {
                    if (!transition.failure.empty()) {
                        return terminate_transition(transition.failure);
                    }
                    return start_warm_rotation();
                }
                return start_next_cleanup();
        }
        return fail("E_COMMAND_KIND");
    }
};

server_warm_tier_controller::server_warm_tier_controller(server_warm_tier_options options)
    : pimpl(std::make_unique<impl>(std::move(options))) {
}

server_warm_tier_controller::~server_warm_tier_controller() {
    std::vector<std::shared_ptr<server_warm_tier_executor>> executors;
    {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        pimpl->stopped = true;
        pimpl->poisoned = true;
        executors.reserve(pimpl->executors.size());
        for (auto & item : pimpl->executors) {
            executors.push_back(std::move(item.second));
        }
        pimpl->executors.clear();
    }
    executors.clear();
}

bool server_warm_tier_controller::enabled() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->options.enabled;
}

std::string server_warm_tier_controller::last_error() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->error;
}

bool server_warm_tier_controller::register_executor(
        std::shared_ptr<server_warm_tier_executor> executor) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->started) {
        return pimpl->fail("E_ALREADY_STARTED");
    }
    if (!executor
            || executor->id().empty()
            || executor->instance_id().empty()) {
        return pimpl->fail("E_EXECUTOR_INVALID");
    }
    for (const auto & item : pimpl->executors) {
        if (item.second->instance_id() == executor->instance_id()) {
            return pimpl->fail("E_EXECUTOR_INSTANCE_DUPLICATE");
        }
    }
    if (!pimpl->executors.emplace(executor->id(), std::move(executor)).second) {
        return pimpl->fail("E_EXECUTOR_DUPLICATE");
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::set_executor_policy(
        server_warm_tier_executor_policy policy) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->started) {
        return pimpl->fail("E_ALREADY_STARTED");
    }
    if (policy.executor_id.empty()
            || pimpl->executors.count(policy.executor_id) == 0
            || policy.credits == 0
            || (policy.role != SERVER_WARM_TIER_EXECUTOR_GPU
                && policy.role != SERVER_WARM_TIER_EXECUTOR_CPU
                && policy.role != SERVER_WARM_TIER_EXECUTOR_PHONE)) {
        return pimpl->fail("E_EXECUTOR_POLICY");
    }
    for (const auto & item : pimpl->executor_policies) {
        if (item.second.order == policy.order) {
            return pimpl->fail("E_EXECUTOR_ORDER_DUPLICATE");
        }
    }
    if (!pimpl->executor_policies.emplace(
                policy.executor_id, std::move(policy)).second) {
        return pimpl->fail("E_EXECUTOR_POLICY_DUPLICATE");
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::set_promotion_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->started) {
        return pimpl->fail("E_ALREADY_STARTED");
    }
    pimpl->promotion_enabled = enabled;
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::set_initial_model_state(
        const std::string & model_id,
        const std::string & executor_id,
        server_warm_tier_model_state state) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->started) {
        return pimpl->fail("E_ALREADY_STARTED");
    }
    if (model_id.empty() || pimpl->executors.count(executor_id) == 0) {
        return pimpl->fail("E_ROUTE_INVALID");
    }
    if (state != SERVER_WARM_TIER_MODEL_ABSENT
            && state != SERVER_WARM_TIER_MODEL_READY
            && state != SERVER_WARM_TIER_MODEL_FAILED) {
        return pimpl->fail("E_INITIAL_STATE");
    }
    pimpl->routes[{model_id, executor_id}] = state;
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::set_bootstrap_model_ready(
        const std::string & model_id,
        const std::string & executor_id) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->started) {
        return pimpl->fail("E_ALREADY_STARTED");
    }
    const route_key key{model_id, executor_id};
    if (!pimpl->route_is(
                model_id,
                executor_id,
                SERVER_WARM_TIER_MODEL_ABSENT)) {
        return pimpl->fail("E_BOOTSTRAP_ROUTE_STATE");
    }
    if (std::find(
                pimpl->bootstrap_waiting.begin(),
                pimpl->bootstrap_waiting.end(),
                key) != pimpl->bootstrap_waiting.end()) {
        return pimpl->fail("E_BOOTSTRAP_DUPLICATE");
    }
    pimpl->bootstrap_waiting.push_back(key);
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::start() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->options.enabled) {
        return pimpl->fail("E_DISABLED");
    }
    if (pimpl->started) {
        return pimpl->fail("E_ALREADY_STARTED");
    }
    if (pimpl->options.run_id.empty()
            || !is_sha256(pimpl->options.runtime_config_sha256)
            || !pimpl->options.clock
            || !pimpl->options.history_digest
            || !pimpl->options.event_sink) {
        return pimpl->fail("E_OPTIONS_INCOMPLETE");
    }
    if (pimpl->executors.empty()) {
        return pimpl->fail("E_EXECUTORS_EMPTY");
    }
    pimpl->started = true;
    pimpl->stopped = false;
    pimpl->activated = pimpl->bootstrap_waiting.empty();
    pimpl->activation_requested = pimpl->activated;
    server_warm_tier_event event;
    event.kind = SERVER_WARM_TIER_EVENT_RUN_START;
    if (!pimpl->emit(std::move(event))) {
        pimpl->started = false;
        return false;
    }
    for (const auto & item : pimpl->routes) {
        server_warm_tier_event route_event;
        route_event.kind = SERVER_WARM_TIER_EVENT_MODEL_STATE_CHANGED;
        route_event.model_id = item.first.first;
        route_event.executor_id = item.first.second;
        route_event.state_before = SERVER_WARM_TIER_MODEL_ABSENT;
        route_event.state_after = item.second;
        if (!pimpl->emit(std::move(route_event))) {
            pimpl->started = false;
            return false;
        }
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::activate() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->operational()) {
        return false;
    }
    if (pimpl->activated) {
        pimpl->error.clear();
        return true;
    }
    if (pimpl->activation_failed) {
        return pimpl->fail("E_ACTIVATION_FAILED");
    }
    if (pimpl->activation_requested) {
        pimpl->error.clear();
        return true;
    }
    pimpl->activation_requested = true;
    if (!pimpl->issue_next_bootstrap()) {
        return false;
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::stop() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->operational()) {
        return false;
    }
    if (pimpl->transition.active || !pimpl->commands.empty()) {
        return pimpl->fail("E_STOP_BUSY");
    }
    for (const auto & item : pimpl->requests) {
        const auto state = item.second.snapshot.state;
        if (state == SERVER_WARM_TIER_REQUEST_QUEUED
                || state == SERVER_WARM_TIER_REQUEST_ACTIVE) {
            return pimpl->fail("E_STOP_REQUESTS");
        }
    }
    server_warm_tier_event event;
    event.kind = SERVER_WARM_TIER_EVENT_RUN_END;
    if (!pimpl->emit(std::move(event))) {
        return false;
    }
    pimpl->stopped = true;
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::finalize_queued(const std::string & reason) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->admitting()) {
        return false;
    }
    if (pimpl->finalizing) {
        return pimpl->fail("E_FINALIZE_STARTED");
    }
    if (reason != "HORIZON_REACHED" && reason != "TRACE_COMPLETE") {
        return pimpl->fail("E_FINALIZE_REASON");
    }
    if (reason == "TRACE_COMPLETE") {
        if (pimpl->transition.active
                || pimpl->pending_intent.has_value()
                || !pimpl->commands.empty()) {
            return pimpl->fail("E_FINALIZE_BUSY");
        }
        for (const auto & item : pimpl->requests) {
            if (item.second.snapshot.state
                        == SERVER_WARM_TIER_REQUEST_ACTIVE
                    || item.second.snapshot.state
                        == SERVER_WARM_TIER_REQUEST_QUEUED) {
                return pimpl->fail("E_FINALIZE_REQUESTS");
            }
        }
    }
    pimpl->finalizing = true;
    pimpl->finalize_reason = reason;
    pimpl->pending_intent.reset();

    for (auto & item : pimpl->requests) {
        auto & record = item.second;
        if (record.snapshot.state != SERVER_WARM_TIER_REQUEST_QUEUED) {
            continue;
        }
        record.snapshot.state = SERVER_WARM_TIER_REQUEST_STRANDED;
        server_warm_tier_event event;
        event.kind = SERVER_WARM_TIER_EVENT_REQUEST_STRANDED;
        event.model_id = record.snapshot.model_id;
        event.request_id = record.snapshot.request_id;
        event.has_request = true;
        event.request = pimpl->snapshot(record);
        event.success = false;
        event.detail = reason;
        if (!pimpl->emit(std::move(event))) {
            pimpl->poisoned = true;
            return false;
        }
    }
    pimpl->pending_by_model.clear();
    if (!pimpl->maybe_finish_finalization()) {
        return false;
    }
    pimpl->error.clear();
    return true;
}

server_warm_tier_activation_state
server_warm_tier_controller::activation_state() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->activation_failed || pimpl->poisoned) {
        return SERVER_WARM_TIER_ACTIVATION_FAILED;
    }
    if (pimpl->activated) {
        return SERVER_WARM_TIER_ACTIVATION_READY;
    }
    if (pimpl->activation_requested) {
        return SERVER_WARM_TIER_ACTIVATION_PREPARING;
    }
    return SERVER_WARM_TIER_ACTIVATION_WAITING;
}

server_warm_tier_finalization_state
server_warm_tier_controller::finalization_state() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->poisoned) {
        return SERVER_WARM_TIER_FINALIZATION_FAILED;
    }
    if (pimpl->finalizing && pimpl->stopped) {
        return SERVER_WARM_TIER_FINALIZATION_FINALIZED;
    }
    if (pimpl->finalizing) {
        return SERVER_WARM_TIER_FINALIZATION_DRAINING;
    }
    return SERVER_WARM_TIER_FINALIZATION_OPEN;
}

bool server_warm_tier_controller::enqueue_request(
        const std::string & request_id,
        const std::string & model_id,
        std::vector<llama_token> prompt_tokens) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->admitting()) {
        return false;
    }
    if (!pimpl->enqueue_record(
                request_id, model_id, std::move(prompt_tokens), 0, 0, false)) {
        return false;
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::enqueue_scheduled_request(
        const std::string & request_id,
        const std::string & model_id,
        std::vector<llama_token> prompt_tokens,
        uint64_t arrival_order,
        int32_t max_output_tokens) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->admitting()) {
        return false;
    }
    if (!pimpl->enqueue_record(
                request_id,
                model_id,
                std::move(prompt_tokens),
                arrival_order,
                max_output_tokens,
                true)) {
        return false;
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::dispatch_request(
        const std::string & request_id,
        const std::string & executor_id,
        int32_t max_output_tokens) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->admitting()) {
        return false;
    }
    auto found = pimpl->requests.find(request_id);
    if (found == pimpl->requests.end()
            || found->second.snapshot.state != SERVER_WARM_TIER_REQUEST_QUEUED) {
        return pimpl->fail("E_REQUEST_NOT_QUEUED");
    }
    if (!pimpl->dispatch_queued_request(
                found->second, executor_id, max_output_tokens)) {
        return false;
    }
    if (found->second.scheduled) {
        pimpl->remove_pending_request(
            found->second.snapshot.model_id, request_id);
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::publish_token(
        const std::string & request_id,
        const std::string & executor_id,
        uint64_t ownership_epoch,
        uint64_t publication_index,
        int64_t position,
        llama_token token) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->admitting()) {
        return false;
    }
    auto found = pimpl->requests.find(request_id);
    if (found == pimpl->requests.end()
            || found->second.snapshot.state != SERVER_WARM_TIER_REQUEST_ACTIVE) {
        return pimpl->fail("E_REQUEST_NOT_ACTIVE");
    }
    auto & record = found->second;
    auto & request = record.snapshot;
    if (request.owner_id != executor_id || request.ownership_epoch != ownership_epoch) {
        return pimpl->fail("E_OWNER_STALE");
    }
    if (record.total_output_tokens <= 0
            || request.committed_output_tokens.size()
                >= static_cast<size_t>(record.total_output_tokens)) {
        return pimpl->fail("E_OUTPUT_BUDGET");
    }
    if (publication_index != record.publications.size()) {
        return pimpl->fail("E_PUBLICATION_DUPLICATE_OR_GAP");
    }
    if (position != request.position || token < 0) {
        return pimpl->fail("E_TOKEN_POSITION");
    }

    record.publications.push_back({
        publication_index,
        position,
        token,
        executor_id,
        ownership_epoch,
    });
    request.committed_output_tokens.push_back(token);
    request.position++;

    server_warm_tier_event event;
    event.kind = SERVER_WARM_TIER_EVENT_TOKEN_COMMITTED;
    event.model_id = request.model_id;
    event.request_id = request_id;
    event.executor_id = executor_id;
    event.has_request = true;
    event.request = pimpl->snapshot(record);
    event.publication_index = publication_index;
    if (!pimpl->emit(std::move(event))) {
        request.position--;
        request.committed_output_tokens.pop_back();
        record.publications.pop_back();
        return false;
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::complete_request(
        const std::string & request_id,
        const std::string & executor_id,
        uint64_t ownership_epoch) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->admitting()) {
        return false;
    }
    auto found = pimpl->requests.find(request_id);
    if (found == pimpl->requests.end()
            || found->second.snapshot.state != SERVER_WARM_TIER_REQUEST_ACTIVE) {
        return pimpl->fail("E_REQUEST_NOT_ACTIVE");
    }
    auto & request = found->second.snapshot;
    if (request.owner_id != executor_id || request.ownership_epoch != ownership_epoch) {
        return pimpl->fail("E_OWNER_STALE");
    }
    if (found->second.total_output_tokens <= 0
            || request.committed_output_tokens.size()
                != static_cast<size_t>(found->second.total_output_tokens)) {
        return pimpl->fail("E_OUTPUT_BUDGET");
    }
    request.state = SERVER_WARM_TIER_REQUEST_COMPLETED;

    server_warm_tier_event event;
    event.kind = SERVER_WARM_TIER_EVENT_REQUEST_COMPLETED;
    event.model_id = request.model_id;
    event.request_id = request_id;
    event.executor_id = executor_id;
    event.has_request = true;
    event.request = pimpl->snapshot(found->second);
    if (!pimpl->emit(std::move(event))) {
        request.state = SERVER_WARM_TIER_REQUEST_ACTIVE;
        return false;
    }
    if (pimpl->transition.active
            && pimpl->transition.phase == TRANSITION_DRAINING
            && pimpl->transition.intent.source_model_id == request.model_id
            && !pimpl->maybe_start_source_drain()) {
        return false;
    }
    if (!pimpl->schedule_pending()) {
        return false;
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::strand_request(
        const std::string & request_id,
        const std::string & reason) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->admitting()) {
        return false;
    }
    auto found = pimpl->requests.find(request_id);
    if (found == pimpl->requests.end()
            || (found->second.snapshot.state != SERVER_WARM_TIER_REQUEST_QUEUED
                && found->second.snapshot.state != SERVER_WARM_TIER_REQUEST_ACTIVE)) {
        return pimpl->fail("E_REQUEST_TERMINAL");
    }
    if (found->second.snapshot.state == SERVER_WARM_TIER_REQUEST_ACTIVE) {
        return pimpl->fail("E_REQUEST_INFLIGHT");
    }
    const auto before = found->second.snapshot.state;
    found->second.snapshot.state = SERVER_WARM_TIER_REQUEST_STRANDED;

    server_warm_tier_event event;
    event.kind = SERVER_WARM_TIER_EVENT_REQUEST_STRANDED;
    event.model_id = found->second.snapshot.model_id;
    event.request_id = request_id;
    event.executor_id = found->second.snapshot.owner_id;
    event.has_request = true;
    event.request = pimpl->snapshot(found->second);
    event.success = false;
    event.detail = reason;
    if (!pimpl->emit(std::move(event))) {
        found->second.snapshot.state = before;
        return false;
    }
    if (found->second.scheduled) {
        pimpl->remove_pending_request(
            found->second.snapshot.model_id, request_id);
    }
    if (!pimpl->schedule_pending()) {
        return false;
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::submit_switch_intent(
        const server_warm_tier_switch_intent & intent) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->admitting()) {
        return false;
    }
    if (!pimpl->begin_switch(intent)) {
        return false;
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::submit_model_switch(
        uint64_t sequence,
        const std::string & intent_id,
        const std::string & source_model_id,
        const std::string & target_model_id) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->admitting()) {
        return false;
    }
    if (!pimpl->accept_model_switch(
                sequence, intent_id, source_model_id, target_model_id)) {
        return false;
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::coalesce_switch_intent(
        const server_warm_tier_switch_intent & obsolete,
        const server_warm_tier_switch_intent & replacement) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->admitting()) {
        return false;
    }
    if (pimpl->finalizing) {
        return pimpl->fail("E_FINALIZING");
    }
    if (obsolete.intent_id.empty()
            || replacement.intent_id.empty()
            || obsolete.intent_id == replacement.intent_id
            || replacement.sequence <= obsolete.sequence) {
        return pimpl->fail("E_COALESCE_INVALID");
    }
    server_warm_tier_event event;
    event.kind = SERVER_WARM_TIER_EVENT_SWITCH_INTENT_COALESCED;
    event.model_id = replacement.target_model_id;
    event.executor_id = replacement.gpu_executor_id;
    event.detail = obsolete.intent_id + ":" + replacement.intent_id;
    if (!pimpl->emit(std::move(event))) {
        return false;
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::handle_executor_result(
        const server_warm_tier_result & result) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->options.enabled) {
        return pimpl->fail("E_DISABLED");
    }
    if (!pimpl->started || pimpl->stopped) {
        return pimpl->fail("E_NOT_RUNNING");
    }
    auto found = pimpl->commands.find(result.command_id);
    const auto quarantined =
        pimpl->quarantined_commands.find(result.command_id);
    if (pimpl->poisoned
            && (found == pimpl->commands.end()
                || (found->second.kind != SERVER_WARM_TIER_COMMAND_DISCARD
                    && quarantined == pimpl->quarantined_commands.end()))) {
        return pimpl->fail("E_POISONED");
    }
    if (!pimpl->poisoned && !pimpl->operational()) {
        return false;
    }
    if (found == pimpl->commands.end()) {
        return pimpl->fail("E_RESULT_UNKNOWN");
    }
    const auto command = found->second;
    if (!pimpl->command_matches(command, result)) {
        return pimpl->fail("E_RESULT_IDENTITY");
    }
    pimpl->commands.erase(found);
    const bool is_quarantined =
        quarantined != pimpl->quarantined_commands.end();
    const std::string quarantine_reason =
        is_quarantined ? quarantined->second : std::string();
    if (!pimpl->emit_command_event(
                command,
                event_end(command.kind),
                result.success,
                is_quarantined
                    ? "E_COMMAND_QUARANTINED:" + quarantine_reason
                    : result.detail,
                is_quarantined ? "QUARANTINED" : "RECEIVED",
                result.publications,
                result.request_complete)) {
        const std::string failure =
            pimpl->error.empty() ? "E_EVENT_SINK" : pimpl->error;
        if (is_quarantined) {
            pimpl->quarantined_commands.erase(quarantined);
            pimpl->start_rollback_discard();
            return false;
        }
        if (command.kind == SERVER_WARM_TIER_COMMAND_DISCARD) {
            pimpl->routes[{command.model_id, command.executor_id}] =
                SERVER_WARM_TIER_MODEL_FAILED;
            if (pimpl->transition.active) {
                pimpl->terminate_transition(
                    pimpl->transition.failure.empty()
                        ? failure : pimpl->transition.failure);
            }
            pimpl->poisoned = true;
            return false;
        }
        if (pimpl->transition.active
                && (command.kind == SERVER_WARM_TIER_COMMAND_LOAD
                    || command.kind == SERVER_WARM_TIER_COMMAND_REPLAY)
                && command.model_id
                    == pimpl->transition.intent.target_model_id
                && command.executor_id
                    == pimpl->transition.intent.gpu_executor_id) {
            return pimpl->begin_rollback(failure, false);
        }
        if (command.kind == SERVER_WARM_TIER_COMMAND_DRAIN
                || command.kind == SERVER_WARM_TIER_COMMAND_LOAD) {
            return pimpl->start_emergency_discard(command, failure);
        }
        if (command.kind == SERVER_WARM_TIER_COMMAND_UNLOAD) {
            return pimpl->terminate_after_unload_event_failure(
                command, failure);
        }
        return false;
    }
    if (is_quarantined) {
        pimpl->quarantined_commands.erase(quarantined);
        if (!pimpl->start_rollback_discard()) {
            return false;
        }
        return true;
    }
    if (!result.success) {
        return pimpl->executor_failure(command, result.detail);
    }
    const bool ok = pimpl->command_success(command, result);
    if (ok) {
        if (!pimpl->schedule_pending()) {
            return false;
        }
        pimpl->error.clear();
    }
    return ok;
}

bool server_warm_tier_controller::emit_resource_sample(
        const std::string & model_id,
        const std::string & executor_id,
        const std::string & detail) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->operational()) {
        return false;
    }
    if (pimpl->executors.count(executor_id) == 0 || detail.empty()) {
        return pimpl->fail("E_RESOURCE_SAMPLE");
    }
    server_warm_tier_event event;
    event.kind = SERVER_WARM_TIER_EVENT_RESOURCE_SAMPLE;
    event.model_id = model_id;
    event.executor_id = executor_id;
    event.detail = detail;
    if (!pimpl->emit(std::move(event))) {
        return false;
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::emit_phone_telemetry(
        const std::string & model_id,
        const std::string & executor_id,
        const std::string & detail) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->operational()) {
        return false;
    }
    if (pimpl->executors.count(executor_id) == 0 || detail.empty()) {
        return pimpl->fail("E_PHONE_TELEMETRY");
    }
    server_warm_tier_event event;
    event.kind = SERVER_WARM_TIER_EVENT_PHONE_TELEMETRY;
    event.model_id = model_id;
    event.executor_id = executor_id;
    event.detail = detail;
    if (!pimpl->emit(std::move(event))) {
        return false;
    }
    pimpl->error.clear();
    return true;
}

bool server_warm_tier_controller::get_model_state(
        const std::string & model_id,
        const std::string & executor_id,
        server_warm_tier_model_state & state) const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    auto found = pimpl->routes.find({model_id, executor_id});
    if (found == pimpl->routes.end()) {
        return false;
    }
    state = found->second;
    return true;
}

bool server_warm_tier_controller::get_request(
        const std::string & request_id,
        server_warm_tier_request_snapshot & request) const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    auto found = pimpl->requests.find(request_id);
    if (found == pimpl->requests.end()) {
        return false;
    }
    request = pimpl->snapshot(found->second);
    return true;
}

bool server_warm_tier_controller::has_active_transition() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->transition.active;
}

uint64_t server_warm_tier_controller::controller_epoch() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->epoch;
}

const char * server_warm_tier_model_state_name(server_warm_tier_model_state state) {
    switch (state) {
        case SERVER_WARM_TIER_MODEL_ABSENT:    return "ABSENT";
        case SERVER_WARM_TIER_MODEL_LOADING:   return "LOADING";
        case SERVER_WARM_TIER_MODEL_READY:     return "READY";
        case SERVER_WARM_TIER_MODEL_DRAINING:  return "DRAINING";
        case SERVER_WARM_TIER_MODEL_REPLAYING: return "REPLAYING";
        case SERVER_WARM_TIER_MODEL_FAILED:    return "FAILED";
    }
    return "UNKNOWN";
}

const char * server_warm_tier_request_state_name(server_warm_tier_request_state state) {
    switch (state) {
        case SERVER_WARM_TIER_REQUEST_QUEUED:    return "QUEUED";
        case SERVER_WARM_TIER_REQUEST_ACTIVE:    return "ACTIVE";
        case SERVER_WARM_TIER_REQUEST_COMPLETED: return "COMPLETED";
        case SERVER_WARM_TIER_REQUEST_STRANDED:  return "STRANDED";
    }
    return "UNKNOWN";
}

const char * server_warm_tier_activation_state_name(
        server_warm_tier_activation_state state) {
    switch (state) {
        case SERVER_WARM_TIER_ACTIVATION_WAITING:   return "WAITING";
        case SERVER_WARM_TIER_ACTIVATION_PREPARING: return "PREPARING";
        case SERVER_WARM_TIER_ACTIVATION_READY:     return "READY";
        case SERVER_WARM_TIER_ACTIVATION_FAILED:    return "FAILED";
    }
    return "UNKNOWN";
}

const char * server_warm_tier_command_kind_name(server_warm_tier_command_kind kind) {
    switch (kind) {
        case SERVER_WARM_TIER_COMMAND_EXECUTE: return "EXECUTE";
        case SERVER_WARM_TIER_COMMAND_DRAIN:   return "DRAIN";
        case SERVER_WARM_TIER_COMMAND_UNLOAD:  return "UNLOAD";
        case SERVER_WARM_TIER_COMMAND_LOAD:    return "LOAD";
        case SERVER_WARM_TIER_COMMAND_REPLAY:  return "REPLAY";
        case SERVER_WARM_TIER_COMMAND_DISCARD: return "DISCARD";
        case SERVER_WARM_TIER_COMMAND_CLEANUP: return "CLEANUP";
    }
    return "UNKNOWN";
}

const char * server_warm_tier_finalization_state_name(
        server_warm_tier_finalization_state state) {
    switch (state) {
        case SERVER_WARM_TIER_FINALIZATION_OPEN:      return "OPEN";
        case SERVER_WARM_TIER_FINALIZATION_DRAINING:  return "DRAINING";
        case SERVER_WARM_TIER_FINALIZATION_FINALIZED: return "FINALIZED";
        case SERVER_WARM_TIER_FINALIZATION_FAILED:    return "FAILED";
    }
    return "UNKNOWN";
}

const char * server_warm_tier_event_kind_name(server_warm_tier_event_kind kind) {
    switch (kind) {
        case SERVER_WARM_TIER_EVENT_RUN_START:               return "run_start";
        case SERVER_WARM_TIER_EVENT_RUN_END:                 return "run_end";
        case SERVER_WARM_TIER_EVENT_REQUEST_ARRIVED:         return "request_arrived";
        case SERVER_WARM_TIER_EVENT_REQUEST_DISPATCHED:      return "request_dispatched";
        case SERVER_WARM_TIER_EVENT_EXECUTE_END:             return "execute_end";
        case SERVER_WARM_TIER_EVENT_TOKEN_COMMITTED:         return "token_committed";
        case SERVER_WARM_TIER_EVENT_REQUEST_COMPLETED:       return "request_completed";
        case SERVER_WARM_TIER_EVENT_REQUEST_STRANDED:        return "request_stranded";
        case SERVER_WARM_TIER_EVENT_MODEL_STATE_CHANGED:     return "model_state_changed";
        case SERVER_WARM_TIER_EVENT_SWITCH_INTENT_SUBMITTED: return "switch_intent_submitted";
        case SERVER_WARM_TIER_EVENT_SWITCH_INTENT_QUEUED:    return "switch_intent_queued";
        case SERVER_WARM_TIER_EVENT_SWITCH_INTENT_COALESCED: return "switch_intent_coalesced";
        case SERVER_WARM_TIER_EVENT_DRAIN_BEGIN:             return "drain_begin";
        case SERVER_WARM_TIER_EVENT_DRAIN_END:               return "drain_end";
        case SERVER_WARM_TIER_EVENT_UNLOAD_BEGIN:            return "unload_begin";
        case SERVER_WARM_TIER_EVENT_UNLOAD_END:              return "unload_end";
        case SERVER_WARM_TIER_EVENT_LOAD_BEGIN:              return "load_begin";
        case SERVER_WARM_TIER_EVENT_LOAD_END:                return "load_end";
        case SERVER_WARM_TIER_EVENT_REPLAY_BEGIN:            return "replay_begin";
        case SERVER_WARM_TIER_EVENT_REPLAY_END:              return "replay_end";
        case SERVER_WARM_TIER_EVENT_OWNERSHIP_COMMIT:        return "ownership_commit";
        case SERVER_WARM_TIER_EVENT_OWNERSHIP_COMMIT_COMPLETE:
            return "ownership_commit_complete";
        case SERVER_WARM_TIER_EVENT_DISCARD_BEGIN:             return "discard_begin";
        case SERVER_WARM_TIER_EVENT_DISCARD_END:               return "discard_end";
        case SERVER_WARM_TIER_EVENT_CLEANUP_BEGIN:           return "cleanup_begin";
        case SERVER_WARM_TIER_EVENT_CLEANUP_END:             return "cleanup_end";
        case SERVER_WARM_TIER_EVENT_EXECUTOR_FAILED:         return "executor_failed";
        case SERVER_WARM_TIER_EVENT_RESOURCE_SAMPLE:         return "resource_sample";
        case SERVER_WARM_TIER_EVENT_PHONE_TELEMETRY:         return "phone_telemetry";
    }
    return "unknown";
}

std::string server_warm_tier_event_jsonl(const server_warm_tier_event & event) {
    using json = nlohmann::ordered_json;

    const auto nullable_string = [](const std::string & value) -> json {
        return value.empty() ? json(nullptr) : json(value);
    };

    json request = nullptr;
    if (event.has_request) {
        request = {
            {"request_id",              event.request.request_id},
            {"model_id",                event.request.model_id},
            {"prompt_tokens",           event.request.prompt_tokens},
            {"committed_output_tokens", event.request.committed_output_tokens},
            {"position",                event.request.position},
            {"owner_id",                nullable_string(event.request.owner_id)},
            {"ownership_epoch",         event.request.ownership_epoch},
            {"publication_index",       event.request.publication_index},
            {"state",                   server_warm_tier_request_state_name(event.request.state)},
        };
    }

    const bool has_model_state =
        event.kind == SERVER_WARM_TIER_EVENT_MODEL_STATE_CHANGED;
    const bool has_publication =
        event.kind == SERVER_WARM_TIER_EVENT_TOKEN_COMMITTED
        || event.kind == SERVER_WARM_TIER_EVENT_OWNERSHIP_COMMIT;

    json result_publications = json::array();
    for (const auto & publication : event.result_publications) {
        result_publications.push_back({
            {"owner_id",          publication.owner_id},
            {"ownership_epoch",   publication.ownership_epoch},
            {"position",          publication.position},
            {"publication_index", publication.publication_index},
            {"token",             publication.token},
        });
    }

    json root = {
        {"schema",                  "s40-warm-tier-event-v3"},
        {"schema_version",          event.schema_version},
        {"run_id",                  event.run_id},
        {"runtime_config_sha256",   nullable_string(event.runtime_config_sha256)},
        {"sequence",                event.sequence},
        {"t_ns",                    event.t_monotonic_ns},
        {"controller_epoch",        event.controller_epoch},
        {"command_id",              event.has_command
            ? json(event.command_id) : json(nullptr)},
        {"command_kind",            event.has_command
            ? json(static_cast<uint32_t>(event.command_kind)) : json(nullptr)},
        {"command_disposition",     nullable_string(event.command_disposition)},
        {"result_publications",     std::move(result_publications)},
        {"result_request_complete", event.result_request_complete},
        {"kind",                    server_warm_tier_event_kind_name(event.kind)},
        {"model_id",                nullable_string(event.model_id)},
        {"request_id",              nullable_string(event.request_id)},
        {"executor_id",             nullable_string(event.executor_id)},
        {"state_before",            has_model_state
            ? json(server_warm_tier_model_state_name(event.state_before)) : json(nullptr)},
        {"state_after",             has_model_state
            ? json(server_warm_tier_model_state_name(event.state_after)) : json(nullptr)},
        {"request",                 std::move(request)},
        {"old_owner",               nullable_string(event.old_owner_id)},
        {"new_owner",               nullable_string(event.new_owner_id)},
        {"old_ownership_epoch",     event.old_owner_id.empty()
            ? json(nullptr) : json(event.old_ownership_epoch)},
        {"new_ownership_epoch",     event.new_owner_id.empty()
            ? json(nullptr) : json(event.new_ownership_epoch)},
        {"publication_index",       has_publication
            ? json(event.publication_index) : json(nullptr)},
        {"history_sha256",          nullable_string(event.committed_history_digest)},
        {"success",                 event.success},
        {"detail",                  event.detail},
    };
    return root.dump() + "\n";
}

bool server_warm_tier_parse_json_strict(
        const std::string & input,
        nlohmann::json & output,
        std::string & error) {
    std::vector<std::set<std::string>> object_keys;
    bool duplicate = false;
    try {
        output = nlohmann::json::parse(
            input,
            [&object_keys, &duplicate](
                    int,
                    nlohmann::json::parse_event_t event,
                    nlohmann::json & parsed) {
                if (event == nlohmann::json::parse_event_t::object_start) {
                    object_keys.emplace_back();
                } else if (event == nlohmann::json::parse_event_t::key) {
                    if (object_keys.empty()
                            || !object_keys.back().insert(
                                parsed.get<std::string>()).second) {
                        duplicate = true;
                    }
                } else if (event == nlohmann::json::parse_event_t::object_end) {
                    if (object_keys.empty()) {
                        duplicate = true;
                    } else {
                        object_keys.pop_back();
                    }
                }
                return true;
            });
    } catch (const nlohmann::json::exception & exception) {
        error = exception.what();
        return false;
    }
    if (duplicate || !object_keys.empty()) {
        error = "duplicate JSON key";
        return false;
    }
    error.clear();
    return true;
}
