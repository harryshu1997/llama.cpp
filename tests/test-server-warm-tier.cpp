#include "server-warm-tier.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

void check(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "test-server-warm-tier: " << message << "\n";
        std::exit(1);
    }
}

struct fake_executor : server_warm_tier_executor {
    std::string name;
    std::string instance;
    std::vector<server_warm_tier_command> commands;
    bool reject_next = false;

    explicit fake_executor(std::string name)
        : name(std::move(name)),
          instance(this->name + "-instance") {
    }

    const std::string & id() const override {
        return name;
    }

    const std::string & instance_id() const override {
        return instance;
    }

    bool submit(const server_warm_tier_command & command, std::string & error) override {
        if (reject_next) {
            reject_next = false;
            error = "injected submit failure";
            return false;
        }
        commands.push_back(command);
        return true;
    }

    server_warm_tier_command take(server_warm_tier_command_kind kind) {
        for (auto it = commands.begin(); it != commands.end(); ++it) {
            if (it->kind == kind) {
                auto command = *it;
                commands.erase(it);
                return command;
            }
        }
        check(false, "expected command not found");
        return {};
    }
};

struct slow_callback_executor : server_warm_tier_executor {
    std::string name = "slow";
    std::string instance = "slow-instance";
    std::function<void(server_warm_tier_result)> sink;
    std::thread worker;

    ~slow_callback_executor() override {
        if (worker.joinable()) {
            worker.join();
        }
    }

    const std::string & id() const override {
        return name;
    }

    const std::string & instance_id() const override {
        return instance;
    }

    bool submit(const server_warm_tier_command & command, std::string &) override {
        worker = std::thread([this, command]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            server_warm_tier_result result;
            result.command_id = command.command_id;
            result.controller_epoch = command.controller_epoch;
            result.kind = command.kind;
            result.model_id = command.model_id;
            result.request_id = command.request_id;
            result.executor_id = command.executor_id;
            result.executor_instance_id = command.executor_instance_id;
            result.success = false;
            result.detail = "late callback";
            sink(std::move(result));
        });
        return true;
    }
};

struct fixture {
    std::atomic<int64_t> clock_ns{0};
    std::vector<server_warm_tier_event> events;
    std::shared_ptr<fake_executor> gpu = std::make_shared<fake_executor>("gpu");
    std::shared_ptr<fake_executor> warm = std::make_shared<fake_executor>("phone");
    int fail_event_kind = -1;
    int fail_event_occurrence = 1;
    int fail_event_seen = 0;
    server_warm_tier_controller controller;

    fixture(uint32_t gpu_credits, uint32_t warm_credits)
        : fixture(
            [](const std::vector<llama_token> &,
               const std::vector<llama_token> &) {
                return std::string(64, 'a');
            },
            -1,
            1,
            gpu_credits,
            warm_credits) {
    }

    explicit fixture(
            server_warm_tier_history_digest history_digest =
                [](const std::vector<llama_token> &, const std::vector<llama_token> &) {
                    return std::string(64, 'a');
                },
            int fail_event_kind = -1,
            int fail_event_occurrence = 1,
            uint32_t gpu_credits = 2,
            uint32_t warm_credits = 2)
        : fail_event_kind(fail_event_kind),
          fail_event_occurrence(fail_event_occurrence),
          controller(server_warm_tier_options{
            true,
            "test-run",
            [this]() { return clock_ns.fetch_add(1); },
            std::move(history_digest),
            [this](const server_warm_tier_event & event) {
                if (static_cast<int>(event.kind) == this->fail_event_kind) {
                    this->fail_event_seen++;
                    if (this->fail_event_seen == this->fail_event_occurrence) {
                        throw std::runtime_error("injected event sink failure");
                    }
                }
                events.push_back(event);
            },
            std::string(64, 'b'),
        }) {
        check(controller.register_executor(gpu), "register gpu");
        check(controller.register_executor(warm), "register warm");
        check(controller.set_executor_policy({
                    "gpu", SERVER_WARM_TIER_EXECUTOR_GPU, 0, gpu_credits}),
                "gpu policy");
        check(controller.set_executor_policy({
                    "phone", SERVER_WARM_TIER_EXECUTOR_PHONE, 1, warm_credits}),
                "phone policy");
        check(controller.set_initial_model_state(
                    "model-a", "gpu", SERVER_WARM_TIER_MODEL_READY), "a gpu ready");
        check(controller.set_initial_model_state(
                    "model-a", "phone", SERVER_WARM_TIER_MODEL_ABSENT), "a phone absent");
        check(controller.set_initial_model_state(
                    "model-b", "gpu", SERVER_WARM_TIER_MODEL_ABSENT), "b gpu absent");
        check(controller.set_initial_model_state(
                    "model-b", "phone", SERVER_WARM_TIER_MODEL_READY), "b phone ready");
        check(controller.start(), "start");
    }

    server_warm_tier_result success(const server_warm_tier_command & command) {
        server_warm_tier_result result;
        result.command_id = command.command_id;
        result.controller_epoch = command.controller_epoch;
        result.kind = command.kind;
        result.model_id = command.model_id;
        result.request_id = command.request_id;
        result.executor_id = command.executor_id;
        result.executor_instance_id = command.executor_instance_id;
        result.success = true;
        return result;
    }

    void complete(const server_warm_tier_command & command) {
        if (!controller.handle_executor_result(success(command))) {
            std::cerr << "failed command "
                      << server_warm_tier_command_kind_name(command.kind)
                      << ": " << controller.last_error() << "\n";
            check(false, "complete command");
        }
    }

    server_warm_tier_switch_intent intent() {
        return {
            0,
            "switch-0",
            "model-a",
            "model-b",
            "gpu",
            "phone",
        };
    }
};

server_warm_tier_publication publication(
        const server_warm_tier_command & command,
        uint64_t index,
        llama_token token) {
    return {
        command.request.publication_index + index,
        command.request.position + static_cast<int64_t>(index),
        token,
        command.executor_id,
        command.request.ownership_epoch,
    };
}

void test_execute_invokes_selected_executor() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1, 2, 3}), "enqueue");
    check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
    auto first = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(first.request.prompt_tokens == std::vector<llama_token>({1, 2, 3}), "execute prompt");
    check(first.max_output_tokens == 1, "one-token execute quantum");
    check(first.total_output_tokens == 2, "immutable total output budget");

    auto first_result = f.success(first);
    first_result.publications = {publication(first, 0, 10)};
    check(f.controller.handle_executor_result(first_result), "first execute result");
    auto second = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(second.max_output_tokens == 1, "second one-token quantum");
    auto second_result = f.success(second);
    second_result.publications = {publication(second, 0, 11)};
    second_result.request_complete = true;
    check(f.controller.handle_executor_result(second_result), "second execute result");

    server_warm_tier_request_snapshot request;
    check(f.controller.get_request("r0", request), "request snapshot");
    check(request.state == SERVER_WARM_TIER_REQUEST_COMPLETED, "request complete");
    check(request.committed_output_tokens == std::vector<llama_token>({10, 11}), "tokens committed");
    check(!f.controller.handle_executor_result(second_result), "duplicate result rejected");
    check(f.controller.last_error() == "E_RESULT_UNKNOWN", "duplicate diagnostic");
}

void test_incomplete_execute_has_followup() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
    check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
    auto first = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    auto result = f.success(first);
    result.publications = {publication(first, 0, 10)};
    result.request_complete = false;
    check(f.controller.handle_executor_result(result), "incomplete quantum");
    auto second = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(second.request.committed_output_tokens == std::vector<llama_token>({10}), "followup history");
    check(second.request.position == 2, "followup position");
}

void test_executor_instance_mismatch_is_rejected() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
    check(f.controller.dispatch_request("r0", "phone", 1), "dispatch");
    const auto execute =
        f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    auto result = f.success(execute);
    result.executor_instance_id = "rebound-instance";
    result.publications = {publication(execute, 0, 10)};
    result.request_complete = true;
    check(!f.controller.handle_executor_result(result),
            "rebound executor instance rejected");
    check(f.controller.last_error() == "E_RESULT_IDENTITY",
            "executor instance diagnostic");
    server_warm_tier_request_snapshot request;
    check(f.controller.get_request("r0", request)
                && request.state == SERVER_WARM_TIER_REQUEST_ACTIVE
                && request.committed_output_tokens.empty()
                && request.owner_id == "phone",
            "rebound result cannot publish");
}

void test_scheduled_fifo_respects_executor_credit() {
    fixture f;
    uint64_t arrival_order = 0;
    for (const char * request_id : {"r0", "r1", "r2"}) {
        check(f.controller.enqueue_scheduled_request(
                    request_id, "model-b", {1}, arrival_order++, 1),
                "scheduled enqueue");
    }
    auto first = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    auto second = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(first.request_id == "r0" && second.request_id == "r1",
            "per-model FIFO dispatch");
    server_warm_tier_request_snapshot queued;
    check(f.controller.get_request("r2", queued)
            && queued.state == SERVER_WARM_TIER_REQUEST_QUEUED,
            "credit keeps third request queued");

    auto result = f.success(first);
    result.publications = {publication(first, 0, 10)};
    result.request_complete = true;
    check(f.controller.handle_executor_result(result), "complete first scheduled request");
    check(f.controller.get_request("r2", queued)
                && queued.state == SERVER_WARM_TIER_REQUEST_QUEUED,
            "cleanup retains executor credit");
    f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_CLEANUP));
    check(f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE).request_id == "r2",
            "cleanup releases credit to FIFO head");
}

void test_direct_dispatch_respects_executor_credit() {
    fixture f;
    for (const char * request_id : {"r0", "r1", "r2"}) {
        check(f.controller.enqueue_request(request_id, "model-b", {1}),
                "enqueue direct request");
    }
    check(f.controller.dispatch_request("r0", "phone", 1),
            "dispatch first direct request");
    check(f.controller.dispatch_request("r1", "phone", 1),
            "dispatch second direct request");
    check(!f.controller.dispatch_request("r2", "phone", 1),
            "third direct request exceeds credits");
    check(f.controller.last_error() == "E_EXECUTOR_CREDIT",
            "direct credit diagnostic");
    server_warm_tier_request_snapshot request;
    check(f.controller.get_request("r2", request)
                && request.state == SERVER_WARM_TIER_REQUEST_QUEUED
                && request.owner_id.empty()
                && request.ownership_epoch == 0,
            "credit rejection preserves queued request");
}

void test_replay_refuses_insufficient_target_credit() {
    fixture f(1, 2);
    for (const char * request_id : {"r0", "r1"}) {
        check(f.controller.enqueue_request(request_id, "model-b", {1}),
                "enqueue warm request");
        check(f.controller.dispatch_request(request_id, "phone", 2),
                "dispatch warm request");
    }
    const auto execute0 = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    const auto execute1 = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);

    check(f.controller.submit_switch_intent(f.intent()), "submit switch");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));

    auto result0 = f.success(execute0);
    result0.publications = {publication(execute0, 0, 10)};
    check(f.controller.handle_executor_result(result0),
            "first warm quantum reaches replay frontier");

    auto result1 = f.success(execute1);
    result1.publications = {publication(execute1, 0, 11)};
    check(!f.controller.handle_executor_result(result1),
            "replay refuses insufficient GPU credits");
    check(f.controller.last_error() == "E_REPLAY_CREDIT",
            "replay credit diagnostic");
    check(std::none_of(
                f.gpu->commands.begin(),
                f.gpu->commands.end(),
                [](const server_warm_tier_command & command) {
                    return command.kind == SERVER_WARM_TIER_COMMAND_REPLAY
                        || command.kind == SERVER_WARM_TIER_COMMAND_EXECUTE;
                }),
            "no destination work issued over credit");
    check(f.gpu->take(SERVER_WARM_TIER_COMMAND_DISCARD).model_id == "model-b",
            "failed destination is discarded");
    for (const char * request_id : {"r0", "r1"}) {
        server_warm_tier_request_snapshot request;
        check(f.controller.get_request(request_id, request)
                    && request.state == SERVER_WARM_TIER_REQUEST_ACTIVE
                    && request.owner_id == "phone"
                    && request.ownership_epoch == 1,
                "replay credit failure preserves old owner");
    }
}

void test_scheduled_request_starts_gpu_only_switch() {
    std::atomic<int64_t> clock_ns{0};
    auto gpu = std::make_shared<fake_executor>("gpu");
    server_warm_tier_controller controller({
        true,
        "auto-switch",
        [&clock_ns]() { return clock_ns.fetch_add(1); },
        [](const std::vector<llama_token> &, const std::vector<llama_token> &) {
            return std::string(64, 'a');
        },
        [](const server_warm_tier_event &) {},
        std::string(64, 'b'),
    });
    check(controller.register_executor(gpu), "register gpu");
    check(controller.set_executor_policy({
                "gpu", SERVER_WARM_TIER_EXECUTOR_GPU, 0, 1}), "gpu policy");
    check(controller.set_promotion_enabled(true), "enable promotion");
    check(controller.set_initial_model_state(
                "model-a", "gpu", SERVER_WARM_TIER_MODEL_READY), "a ready");
    check(controller.set_initial_model_state(
                "model-b", "gpu", SERVER_WARM_TIER_MODEL_ABSENT), "b absent");
    check(controller.start(), "start");
    check(controller.enqueue_scheduled_request(
                "r0", "model-b", {1}, 0, 1), "enqueue blocked target");

    const auto complete = [&controller](const server_warm_tier_command & command) {
        server_warm_tier_result result;
        result.command_id = command.command_id;
        result.controller_epoch = command.controller_epoch;
        result.kind = command.kind;
        result.model_id = command.model_id;
        result.request_id = command.request_id;
        result.executor_id = command.executor_id;
        result.executor_instance_id = command.executor_instance_id;
        result.success = true;
        check(controller.handle_executor_result(result), "complete auto-switch command");
    };
    complete(gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    complete(gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    complete(gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
    const auto execute = gpu->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(execute.model_id == "model-b" && execute.request_id == "r0",
            "queued target dispatched after automatic switch");
}

void test_overlapping_demand_queues_reverse_promotion() {
    std::atomic<int64_t> clock_ns{0};
    auto gpu = std::make_shared<fake_executor>("gpu");
    auto phone = std::make_shared<fake_executor>("phone");
    server_warm_tier_controller controller({
        true,
        "coalesced-switch",
        [&clock_ns]() { return clock_ns.fetch_add(1); },
        [](const std::vector<llama_token> &, const std::vector<llama_token> &) {
            return std::string(64, 'a');
        },
        [](const server_warm_tier_event &) {},
        std::string(64, 'b'),
    });
    check(controller.register_executor(gpu), "register gpu");
    check(controller.register_executor(phone), "register phone");
    check(controller.set_executor_policy({
                "gpu", SERVER_WARM_TIER_EXECUTOR_GPU, 0, 2}), "gpu policy");
    check(controller.set_executor_policy({
                "phone", SERVER_WARM_TIER_EXECUTOR_PHONE, 1, 2}), "phone policy");
    check(controller.set_promotion_enabled(true), "enable promotion");
    check(controller.set_initial_model_state(
                "model-a", "gpu", SERVER_WARM_TIER_MODEL_READY), "a gpu ready");
    check(controller.set_initial_model_state(
                "model-a", "phone", SERVER_WARM_TIER_MODEL_ABSENT), "a phone absent");
    check(controller.set_initial_model_state(
                "model-b", "gpu", SERVER_WARM_TIER_MODEL_ABSENT), "b gpu absent");
    check(controller.set_initial_model_state(
                "model-b", "phone", SERVER_WARM_TIER_MODEL_READY), "b phone ready");
    check(controller.start(), "start");

    check(controller.enqueue_scheduled_request(
                "rb", "model-b", {1}, 0, 1), "enqueue b");
    const auto execute_b = phone->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(controller.enqueue_scheduled_request(
                "ra", "model-a", {2}, 1, 1), "enqueue reverse demand");

    const auto complete = [&controller](const server_warm_tier_command & command) {
        server_warm_tier_result result;
        result.command_id = command.command_id;
        result.controller_epoch = command.controller_epoch;
        result.kind = command.kind;
        result.model_id = command.model_id;
        result.request_id = command.request_id;
        result.executor_id = command.executor_id;
        result.executor_instance_id = command.executor_instance_id;
        result.success = true;
        check(controller.handle_executor_result(result), "complete switch command");
    };
    complete(gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    complete(gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    complete(gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));

    server_warm_tier_result execute_result;
    execute_result.command_id = execute_b.command_id;
    execute_result.controller_epoch = execute_b.controller_epoch;
    execute_result.kind = execute_b.kind;
    execute_result.model_id = execute_b.model_id;
    execute_result.request_id = execute_b.request_id;
    execute_result.executor_id = execute_b.executor_id;
    execute_result.executor_instance_id = execute_b.executor_instance_id;
    execute_result.success = true;
    execute_result.publications = {publication(execute_b, 0, 10)};
    execute_result.request_complete = true;
    check(controller.handle_executor_result(execute_result), "finish b phone request");

    complete(phone->take(SERVER_WARM_TIER_COMMAND_CLEANUP));
    complete(phone->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    complete(phone->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    complete(phone->take(SERVER_WARM_TIER_COMMAND_LOAD));
    check(phone->take(SERVER_WARM_TIER_COMMAND_EXECUTE).request_id == "ra",
            "continuous source demand dispatched on re-prepared phone route");
    const auto reverse_drain = gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN);
    check(reverse_drain.model_id == "model-b",
            "queued reverse promotion starts after forward transition");
}

void test_negative_prompt_token_is_rejected() {
    fixture f;
    check(!f.controller.enqueue_request("r0", "model-b", {-1}),
            "negative token rejected");
    check(f.controller.last_error() == "E_REQUEST_INVALID",
            "negative token diagnostic");
}

void test_execute_budget_failures() {
    {
        fixture f;
        check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
        check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
        auto command = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
        auto result = f.success(command);
        result.publications = {
            publication(command, 0, 10),
            publication(command, 1, 11),
            publication(command, 2, 12),
        };
        result.request_complete = true;
        check(!f.controller.handle_executor_result(result), "over budget rejected");
        check(f.controller.last_error() == "E_EXECUTE_BUDGET", "over budget diagnostic");
        check(std::any_of(
                    f.events.begin(),
                    f.events.end(),
                    [](const server_warm_tier_event & event) {
                        return event.kind
                            == SERVER_WARM_TIER_EVENT_REQUEST_STRANDED;
                    }),
                "over budget stranded event");
        server_warm_tier_model_state state;
        check(f.controller.get_model_state("model-b", "phone", state)
                    && state == SERVER_WARM_TIER_MODEL_FAILED,
                "over budget result fails route");
        check(!f.controller.enqueue_request("late", "model-b", {1}),
                "over budget result poisons run");
        check(f.controller.last_error() == "E_POISONED",
                "over budget poison diagnostic");
    }
    {
        fixture f;
        check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
        check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
        auto command = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
        auto result = f.success(command);
        result.publications = {publication(command, 0, 10)};
        result.request_complete = true;
        check(!f.controller.handle_executor_result(result), "under budget complete rejected");
        check(f.controller.last_error() == "E_EXECUTE_COMPLETION", "under budget diagnostic");
        server_warm_tier_model_state state;
        check(f.controller.get_model_state("model-b", "phone", state)
                    && state == SERVER_WARM_TIER_MODEL_FAILED,
                "under budget result fails route");
    }
}

void test_rotation_replays_and_rearms_warm_tier() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1, 2}), "enqueue");
    check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
    auto execute = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);

    check(f.controller.submit_switch_intent(f.intent()), "switch");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
    check(f.gpu->commands.empty(), "replay waits for inflight warm quantum");
    check(f.controller.enqueue_request("late", "model-b", {3}), "late enqueue");
    check(!f.controller.dispatch_request("late", "phone", 1), "closed warm frontier");
    check(f.controller.last_error() == "E_ROUTE_NOT_READY", "closed frontier diagnostic");
    check(f.controller.strand_request("late", "frontier closed"), "strand late request");

    auto execute_result = f.success(execute);
    execute_result.publications = {publication(execute, 0, 10)};
    execute_result.request_complete = false;
    check(f.controller.handle_executor_result(execute_result), "finish warm quantum");

    auto replay = f.gpu->take(SERVER_WARM_TIER_COMMAND_REPLAY);
    auto replay_result = f.success(replay);
    replay_result.has_replay_snapshot = true;
    replay_result.replay_snapshot = replay.request;
    check(f.controller.handle_executor_result(replay_result), "replay");

    server_warm_tier_request_snapshot request;
    check(f.controller.get_request("r0", request), "request after commit");
    check(request.owner_id == "gpu" && request.ownership_epoch == 2, "atomic owner commit");
    bool commit_complete = false;
    for (const auto & event : f.events) {
        commit_complete |= event.kind == SERVER_WARM_TIER_EVENT_OWNERSHIP_COMMIT_COMPLETE;
    }
    check(commit_complete, "ownership transaction complete event");

    f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_CLEANUP));
    f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_LOAD));
    check(!f.controller.has_active_transition(), "transition complete");

    server_warm_tier_model_state state;
    check(f.controller.get_model_state("model-b", "gpu", state)
            && state == SERVER_WARM_TIER_MODEL_READY, "b gpu ready");
    check(f.controller.get_model_state("model-b", "phone", state)
            && state == SERVER_WARM_TIER_MODEL_ABSENT, "b phone absent");
    check(f.controller.get_model_state("model-a", "phone", state)
            && state == SERVER_WARM_TIER_MODEL_READY, "a phone ready");
}

void test_completed_phone_quantum_does_not_stall_replay() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
    check(f.controller.dispatch_request("r0", "phone", 1), "dispatch");
    auto execute = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(f.controller.submit_switch_intent(f.intent()), "switch");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));

    check(!f.controller.strand_request("r0", "client timeout"),
            "inflight request cannot be stranded");
    check(f.controller.last_error() == "E_REQUEST_INFLIGHT",
            "inflight strand diagnostic");
    auto result = f.success(execute);
    result.publications = {publication(execute, 0, 10)};
    result.request_complete = true;
    check(f.controller.handle_executor_result(result), "complete phone quantum");
    f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_CLEANUP));
    check(f.warm->take(SERVER_WARM_TIER_COMMAND_DRAIN).model_id == "model-b",
            "completed frontier advances to warm drain");
}

void test_gpu_only_switch_has_no_warm_lifecycle() {
    fixture f;
    check(f.controller.enqueue_request("queued", "model-b", {1}), "enqueue target");
    auto intent = f.intent();
    intent.warm_executor_id.clear();
    check(f.controller.submit_switch_intent(intent), "gpu-only switch");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
    check(!f.controller.has_active_transition(), "gpu-only transition complete");
    check(f.warm->commands.empty(), "gpu-only switch has no warm commands");

    server_warm_tier_model_state state;
    check(f.controller.get_model_state("model-a", "gpu", state)
            && state == SERVER_WARM_TIER_MODEL_ABSENT, "gpu-only source absent");
    check(f.controller.get_model_state("model-b", "gpu", state)
            && state == SERVER_WARM_TIER_MODEL_READY, "gpu-only target ready");
    check(f.controller.dispatch_request("queued", "gpu", 1), "dispatch queued target");
}

void test_source_quanta_finish_before_drain_and_unload() {
    fixture f;
    check(f.controller.enqueue_request("active", "model-a", {1}), "enqueue source");
    check(f.controller.dispatch_request("active", "gpu", 3), "dispatch source");
    auto execute = f.gpu->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(f.controller.submit_switch_intent(f.intent()), "switch");
    check(f.controller.enqueue_request("queued", "model-a", {2}), "queue source");
    check(!f.controller.dispatch_request("queued", "gpu", 1),
            "draining route rejects new dispatch");
    check(f.controller.last_error() == "E_ROUTE_NOT_READY",
            "draining route diagnostic");
    check(f.gpu->commands.empty(), "drain deferred behind active source");

    for (int token = 10; token < 13; ++token) {
        auto result = f.success(execute);
        result.publications = {publication(execute, 0, token)};
        result.request_complete = token == 12;
        check(f.controller.handle_executor_result(result), "source quantum");
        if (token < 12) {
            check(f.gpu->commands.size() == 1
                        && f.gpu->commands.front().kind
                            == SERVER_WARM_TIER_COMMAND_EXECUTE,
                    "next source quantum precedes drain");
            execute = f.gpu->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
        }
    }

    check(f.gpu->commands.size() == 1
                && f.gpu->commands.front().kind == SERVER_WARM_TIER_COMMAND_CLEANUP,
            "cleanup follows terminal source quantum");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_CLEANUP));
    check(f.gpu->commands.size() == 1
                && f.gpu->commands.front().kind == SERVER_WARM_TIER_COMMAND_DRAIN,
            "drain follows terminal cleanup");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    check(f.gpu->commands.size() == 1
                && f.gpu->commands.front().kind == SERVER_WARM_TIER_COMMAND_UNLOAD,
            "unload follows drain");
    for (const auto & command : f.gpu->commands) {
        check(command.kind != SERVER_WARM_TIER_COMMAND_EXECUTE,
                "no execute after unload begins");
    }
}

void test_source_execute_failure_aborts_drain_transition() {
    fixture f;
    check(f.controller.enqueue_request("active", "model-a", {1}), "enqueue source");
    check(f.controller.dispatch_request("active", "gpu", 3), "dispatch source");
    auto execute = f.gpu->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(f.controller.submit_switch_intent(f.intent()), "switch");
    auto failed = f.success(execute);
    failed.success = false;
    failed.detail = "injected execute failure";
    check(!f.controller.handle_executor_result(failed), "source execute failure");
    check(!f.controller.has_active_transition(), "failed source drain is terminal");
    server_warm_tier_model_state state;
    check(f.controller.get_model_state("model-a", "gpu", state)
                && state == SERVER_WARM_TIER_MODEL_FAILED,
            "failed source route is not reused");
    check(f.gpu->commands.empty(), "failure does not issue drain or unload");
    check(!f.controller.enqueue_request("late", "model-b", {2}),
            "executor failure poisons later admission");
    check(f.controller.last_error() == "E_POISONED",
            "executor failure poison diagnostic");
}

void test_lifecycle_failure_poisoned_with_rollback_cleanup() {
    fixture f;
    check(f.controller.submit_switch_intent(f.intent()), "switch");

    auto drain = f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN);
    auto failed_drain = f.success(drain);
    failed_drain.success = false;
    failed_drain.detail = "injected drain failure";
    check(!f.controller.handle_executor_result(failed_drain), "drain failure");
    check(!f.controller.enqueue_request("late", "model-b", {2}),
            "drain failure poisons later admission");
    check(f.controller.last_error() == "E_POISONED",
            "drain failure poison diagnostic");

    fixture rollback;
    check(rollback.controller.submit_switch_intent(rollback.intent()), "rollback switch");
    rollback.complete(rollback.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    rollback.complete(rollback.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    auto load = rollback.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD);
    auto failed_load = rollback.success(load);
    failed_load.success = false;
    failed_load.detail = "injected load failure";
    check(!rollback.controller.handle_executor_result(failed_load), "load failure");
    check(!rollback.controller.enqueue_request("late", "model-b", {2}),
            "load failure poisons later admission");
    check(rollback.controller.last_error() == "E_POISONED",
            "load failure poison diagnostic");
    auto discard = rollback.gpu->take(SERVER_WARM_TIER_COMMAND_DISCARD);
    check(!rollback.controller.handle_executor_result(rollback.success(discard)),
            "poisoned controller accepts rollback discard only");
    check(!rollback.controller.has_active_transition(),
            "rollback discard terminates poisoned transition");
}

void test_terminal_cleanup_failure_poisoned() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
    check(f.controller.dispatch_request("r0", "phone", 1), "dispatch");
    auto execute = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    auto result = f.success(execute);
    result.publications = {publication(execute, 0, 10)};
    result.request_complete = true;
    check(f.controller.handle_executor_result(result), "complete request");
    auto cleanup = f.warm->take(SERVER_WARM_TIER_COMMAND_CLEANUP);
    auto failed = f.success(cleanup);
    failed.success = false;
    failed.detail = "injected terminal cleanup failure";
    check(!f.controller.handle_executor_result(failed), "cleanup failure");
    server_warm_tier_request_snapshot request;
    check(f.controller.get_request("r0", request)
                && request.state == SERVER_WARM_TIER_REQUEST_COMPLETED,
            "cleanup failure preserves completion ledger");
    server_warm_tier_model_state state;
    check(f.controller.get_model_state("model-b", "phone", state)
                && state == SERVER_WARM_TIER_MODEL_FAILED,
            "cleanup failure invalidates route");
    check(!f.controller.enqueue_request("r1", "model-b", {1}),
            "cleanup failure poisons run");
    check(f.controller.last_error() == "E_POISONED", "poison diagnostic");
}

void test_cleanup_advances_across_all_replayed_requests() {
    fixture f;
    for (const char * request_id : {"r0", "r1"}) {
        check(f.controller.enqueue_request(request_id, "model-b", {1}), "enqueue");
        check(f.controller.dispatch_request(request_id, "phone", 2), "dispatch");
    }
    auto execute0 = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    auto execute1 = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(f.controller.submit_switch_intent(f.intent()), "switch");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));

    for (const auto & execute : {execute0, execute1}) {
        auto result = f.success(execute);
        result.publications = {publication(execute, 0, 10)};
        check(f.controller.handle_executor_result(result), "finish phone quantum");
    }
    for (int i = 0; i < 2; ++i) {
        auto replay = f.gpu->take(SERVER_WARM_TIER_COMMAND_REPLAY);
        auto result = f.success(replay);
        result.has_replay_snapshot = true;
        result.replay_snapshot = replay.request;
        check(f.controller.handle_executor_result(result), "replay");
    }
    const auto cleanup0 = f.warm->take(SERVER_WARM_TIER_COMMAND_CLEANUP);
    f.complete(cleanup0);
    const auto cleanup1 = f.warm->take(SERVER_WARM_TIER_COMMAND_CLEANUP);
    check(cleanup0.request_id != cleanup1.request_id, "cleanup requests are distinct");
    f.complete(cleanup1);
    check(f.warm->take(SERVER_WARM_TIER_COMMAND_DRAIN).model_id == "model-b",
            "all cleanup commands completed before warm drain");
}

void test_target_cleanup_does_not_satisfy_old_owner_cleanup() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
    check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
    auto phone_execute = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(f.controller.submit_switch_intent(f.intent()), "switch");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));

    auto phone_result = f.success(phone_execute);
    phone_result.publications = {publication(phone_execute, 0, 10)};
    check(f.controller.handle_executor_result(phone_result), "phone quantum");

    auto replay = f.gpu->take(SERVER_WARM_TIER_COMMAND_REPLAY);
    auto replay_result = f.success(replay);
    replay_result.has_replay_snapshot = true;
    replay_result.replay_snapshot = replay.request;
    check(f.controller.handle_executor_result(replay_result), "replay");

    auto gpu_execute = f.gpu->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    auto gpu_result = f.success(gpu_execute);
    gpu_result.publications = {publication(gpu_execute, 0, 11)};
    gpu_result.request_complete = true;
    check(f.controller.handle_executor_result(gpu_result), "gpu quantum");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_CLEANUP));

    check(f.warm->commands.size() == 1
                && f.warm->commands.front().kind
                    == SERVER_WARM_TIER_COMMAND_CLEANUP,
            "target cleanup leaves old-owner cleanup pending");
    f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_CLEANUP));
    check(f.warm->commands.size() == 1
                && f.warm->commands.front().kind
                    == SERVER_WARM_TIER_COMMAND_DRAIN,
            "old-owner cleanup advances warm rotation");
}

void test_target_cleanup_blocks_warm_drain_after_gpu_load() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
    check(f.controller.dispatch_request("r0", "phone", 1), "dispatch");
    const auto execute = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(f.controller.submit_switch_intent(f.intent()), "switch");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    const auto load = f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD);

    auto execute_result = f.success(execute);
    execute_result.publications = {publication(execute, 0, 10)};
    execute_result.request_complete = true;
    check(f.controller.handle_executor_result(execute_result),
            "phone request completes before gpu load");
    const auto cleanup = f.warm->take(SERVER_WARM_TIER_COMMAND_CLEANUP);

    f.complete(load);
    for (const auto & command : f.warm->commands) {
        check(command.kind != SERVER_WARM_TIER_COMMAND_DRAIN,
                "warm drain waits for target cleanup");
    }

    f.complete(cleanup);
    check(f.warm->take(SERVER_WARM_TIER_COMMAND_DRAIN).model_id == "model-b",
            "warm drain starts after target cleanup");
}

void test_incomplete_ownership_transaction_preserves_old_owner() {
    const auto run = [](server_warm_tier_event_kind kind, int occurrence) {
        fixture f;
        std::vector<server_warm_tier_command> executes;
        for (const char * request_id : {"r0", "r1"}) {
            check(f.controller.enqueue_request(request_id, "model-b", {1}), "enqueue");
            check(f.controller.dispatch_request(request_id, "phone", 2), "dispatch");
            executes.push_back(f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE));
        }
        check(f.controller.submit_switch_intent(f.intent()), "switch");
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
        for (const auto & execute : executes) {
            auto result = f.success(execute);
            result.publications = {publication(execute, 0, 10)};
            check(f.controller.handle_executor_result(result), "phone quantum");
        }
        auto replay0 = f.gpu->take(SERVER_WARM_TIER_COMMAND_REPLAY);
        auto replay1 = f.gpu->take(SERVER_WARM_TIER_COMMAND_REPLAY);
        for (int i = 0; i < 2; ++i) {
            const auto & replay = i == 0 ? replay0 : replay1;
            if (i == 1) {
                f.fail_event_kind = kind;
                f.fail_event_occurrence = occurrence;
                f.fail_event_seen = 0;
            }
            auto result = f.success(replay);
            result.has_replay_snapshot = true;
            result.replay_snapshot = replay.request;
            const bool expected = i == 0;
            check(f.controller.handle_executor_result(result) == expected,
                    "event failure commit result");
        }

        for (const char * request_id : {"r0", "r1"}) {
            server_warm_tier_request_snapshot request;
            check(f.controller.get_request(request_id, request), "request snapshot");
            check(request.owner_id == "phone" && request.ownership_epoch == 1,
                    "incomplete transaction preserves old owner");
        }
        server_warm_tier_model_state state;
        check(f.controller.get_model_state("model-b", "gpu", state)
                    && state == SERVER_WARM_TIER_MODEL_FAILED,
                "incomplete transaction leaves target failed");
        check(f.controller.get_model_state("model-b", "phone", state)
                    && state == SERVER_WARM_TIER_MODEL_READY,
                "incomplete transaction restores warm route");
        check(f.controller.has_active_transition(),
                "incomplete transaction waits for target discard");
        const auto discard = f.gpu->take(SERVER_WARM_TIER_COMMAND_DISCARD);
        check(discard.model_id == "model-b" && discard.executor_id == "gpu",
                "incomplete transaction discards replay target");
        check(!f.controller.handle_executor_result(f.success(discard)),
                "discard closes failed ownership transaction");
        check(!f.controller.has_active_transition(),
                "discard terminates failed ownership transaction");
        bool complete = false;
        for (const auto & event : f.events) {
            complete |= event.kind == SERVER_WARM_TIER_EVENT_OWNERSHIP_COMMIT_COMPLETE;
        }
        check(!complete, "failed ownership transaction has no completion marker");
        check(f.controller.last_error() == "E_EVENT_SINK", "event sink diagnostic");
    };

    run(SERVER_WARM_TIER_EVENT_MODEL_STATE_CHANGED, 1);
    run(SERVER_WARM_TIER_EVENT_OWNERSHIP_COMMIT, 1);
    run(SERVER_WARM_TIER_EVENT_OWNERSHIP_COMMIT, 2);
    run(SERVER_WARM_TIER_EVENT_OWNERSHIP_COMMIT_COMPLETE, 1);
}

void test_precommit_end_event_failure_discards_target() {
    const auto verify_rollback = [](fixture & f) {
        server_warm_tier_request_snapshot request;
        check(f.controller.get_request("r0", request), "request snapshot");
        check(request.owner_id == "phone" && request.ownership_epoch == 1,
                "precommit event failure preserves old owner");
        const auto discard =
            f.gpu->take(SERVER_WARM_TIER_COMMAND_DISCARD);
        check(discard.model_id == "model-b"
                    && discard.executor_id == "gpu",
                "precommit event failure discards target");
        check(!f.controller.handle_executor_result(f.success(discard)),
                "precommit discard closes failed transition");
        check(!f.controller.has_active_transition(),
                "precommit discard terminates transition");
        check(f.controller.last_error() == "E_EVENT_SINK",
                "precommit event failure diagnostic");
    };

    {
        fixture f;
        check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
        check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
        f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
        check(f.controller.submit_switch_intent(f.intent()), "switch");
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
        const auto load = f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD);
        f.fail_event_kind = SERVER_WARM_TIER_EVENT_LOAD_END;
        check(!f.controller.handle_executor_result(f.success(load)),
                "load end event failure");
        verify_rollback(f);
    }

    {
        fixture f;
        check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
        check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
        const auto execute =
            f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
        check(f.controller.submit_switch_intent(f.intent()), "switch");
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
        auto execute_result = f.success(execute);
        execute_result.publications = {publication(execute, 0, 10)};
        check(f.controller.handle_executor_result(execute_result),
                "phone quantum");
        const auto replay =
            f.gpu->take(SERVER_WARM_TIER_COMMAND_REPLAY);
        auto replay_result = f.success(replay);
        replay_result.has_replay_snapshot = true;
        replay_result.replay_snapshot = replay.request;
        f.fail_event_kind = SERVER_WARM_TIER_EVENT_REPLAY_END;
        check(!f.controller.handle_executor_result(replay_result),
                "replay end event failure");
        verify_rollback(f);
    }
}

void test_replay_state_event_failure_discards_target() {
    for (int occurrence : {1, 2, 3}) {
        fixture f;
        check(f.controller.submit_switch_intent(f.intent()), "switch");
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
        const auto load = f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD);
        f.fail_event_kind = SERVER_WARM_TIER_EVENT_MODEL_STATE_CHANGED;
        f.fail_event_occurrence = occurrence;
        f.fail_event_seen = 0;
        check(!f.controller.handle_executor_result(f.success(load)),
                "replay state event failure");
        const auto discard =
            f.gpu->take(SERVER_WARM_TIER_COMMAND_DISCARD);
        check(discard.model_id == "model-b"
                    && discard.executor_id == "gpu",
                "replay state failure discards target");
        check(!f.controller.handle_executor_result(f.success(discard)),
                "replay state discard closes transition");
        check(!f.controller.has_active_transition(),
                "replay state failure terminates transition");
        check(f.controller.last_error() == "E_EVENT_SINK",
                "replay state failure diagnostic");
    }
}

void test_lifecycle_end_event_failure_is_terminal() {
    {
        fixture f;
        check(f.controller.submit_switch_intent(f.intent()), "switch");
        const auto drain =
            f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN);
        f.fail_event_kind = SERVER_WARM_TIER_EVENT_DRAIN_END;
        check(!f.controller.handle_executor_result(f.success(drain)),
                "drain end event failure");
        const auto discard =
            f.gpu->take(SERVER_WARM_TIER_COMMAND_DISCARD);
        check(discard.model_id == "model-a"
                    && discard.executor_id == "gpu",
                "drain failure discards loaded source");
        check(!f.controller.handle_executor_result(f.success(discard)),
                "drain discard closes transition");
        check(!f.controller.has_active_transition(),
                "drain event failure terminates transition");
    }
    {
        fixture f;
        check(f.controller.submit_switch_intent(f.intent()), "switch");
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        const auto unload =
            f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD);
        f.fail_event_kind = SERVER_WARM_TIER_EVENT_UNLOAD_END;
        check(!f.controller.handle_executor_result(f.success(unload)),
                "unload end event failure");
        check(!f.controller.has_active_transition(),
                "unload event failure terminates transition");
        server_warm_tier_model_state state;
        check(f.controller.get_model_state("model-a", "gpu", state)
                    && state == SERVER_WARM_TIER_MODEL_FAILED,
                "unload event failure invalidates source");
        check(f.gpu->commands.empty(),
                "unloaded source needs no cleanup command");
    }
    {
        fixture f;
        check(f.controller.submit_switch_intent(f.intent()), "switch");
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
        f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
        const auto load =
            f.warm->take(SERVER_WARM_TIER_COMMAND_LOAD);
        f.fail_event_kind = SERVER_WARM_TIER_EVENT_LOAD_END;
        check(!f.controller.handle_executor_result(f.success(load)),
                "warm load end event failure");
        const auto discard =
            f.warm->take(SERVER_WARM_TIER_COMMAND_DISCARD);
        check(discard.model_id == "model-a"
                    && discard.executor_id == "phone",
                "warm load failure discards physical route");
        check(!f.controller.handle_executor_result(f.success(discard)),
                "warm load discard closes transition");
        check(!f.controller.has_active_transition(),
                "warm load event failure terminates transition");
    }
}

void test_post_lifecycle_state_event_failure_is_terminal() {
    for (int occurrence : {1, 2}) {
        fixture f;
        check(f.controller.submit_switch_intent(f.intent()), "switch");
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        const auto unload =
            f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD);
        f.fail_event_kind = SERVER_WARM_TIER_EVENT_MODEL_STATE_CHANGED;
        f.fail_event_occurrence = occurrence;
        f.fail_event_seen = 0;
        check(!f.controller.handle_executor_result(f.success(unload)),
                "source unload state event failure");
        check(!f.controller.has_active_transition(),
                "source unload state failure terminates transition");
        server_warm_tier_model_state state;
        const std::string failed_model =
            occurrence == 1 ? "model-a" : "model-b";
        check(f.controller.get_model_state(failed_model, "gpu", state)
                    && state == SERVER_WARM_TIER_MODEL_FAILED,
                "post-unload state failure invalidates route");
        check(f.gpu->commands.empty(),
                "source unload state failure queues no command");
    }
    {
        fixture f;
        check(f.controller.submit_switch_intent(f.intent()), "switch");
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
        f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        const auto unload =
            f.warm->take(SERVER_WARM_TIER_COMMAND_UNLOAD);
        f.fail_event_kind = SERVER_WARM_TIER_EVENT_MODEL_STATE_CHANGED;
        f.fail_event_occurrence = 2;
        f.fail_event_seen = 0;
        check(!f.controller.handle_executor_result(f.success(unload)),
                "warm load state event failure");
        check(!f.controller.has_active_transition(),
                "warm load state failure terminates transition");
        server_warm_tier_model_state state;
        check(f.controller.get_model_state("model-a", "phone", state)
                    && state == SERVER_WARM_TIER_MODEL_FAILED,
                "warm load state failure invalidates route");
        check(f.warm->commands.empty(),
                "warm load state failure queues no physical load");
    }
    {
        fixture f;
        check(f.controller.submit_switch_intent(f.intent()), "switch");
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
        f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
        const auto load =
            f.warm->take(SERVER_WARM_TIER_COMMAND_LOAD);
        f.fail_event_kind = SERVER_WARM_TIER_EVENT_MODEL_STATE_CHANGED;
        f.fail_event_occurrence = 1;
        f.fail_event_seen = 0;
        check(!f.controller.handle_executor_result(f.success(load)),
                "warm ready state event failure");
        const auto discard =
            f.warm->take(SERVER_WARM_TIER_COMMAND_DISCARD);
        check(discard.model_id == "model-a"
                    && discard.executor_id == "phone",
                "warm ready state failure discards loaded route");
        check(!f.controller.handle_executor_result(f.success(discard)),
                "warm ready discard closes transition");
        check(!f.controller.has_active_transition(),
                "warm ready state failure terminates transition");
    }
    {
        fixture f;
        auto intent = f.intent();
        intent.warm_executor_id.clear();
        check(f.controller.submit_switch_intent(intent), "gpu-only switch");
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
        const auto load =
            f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD);
        f.fail_event_kind = SERVER_WARM_TIER_EVENT_MODEL_STATE_CHANGED;
        f.fail_event_occurrence = 1;
        f.fail_event_seen = 0;
        check(!f.controller.handle_executor_result(f.success(load)),
                "gpu ready state event failure");
        const auto discard =
            f.gpu->take(SERVER_WARM_TIER_COMMAND_DISCARD);
        check(discard.model_id == "model-b"
                    && discard.executor_id == "gpu",
                "gpu ready state failure discards loaded route");
        check(!f.controller.handle_executor_result(f.success(discard)),
                "gpu ready discard closes transition");
        check(!f.controller.has_active_transition(),
                "gpu ready state failure terminates transition");
    }
}

void test_chained_begin_event_failure_is_terminal() {
    for (const auto fail_kind : {
            SERVER_WARM_TIER_EVENT_DRAIN_BEGIN,
            SERVER_WARM_TIER_EVENT_UNLOAD_BEGIN}) {
        fixture f;
        if (fail_kind == SERVER_WARM_TIER_EVENT_DRAIN_BEGIN) {
            f.fail_event_kind = fail_kind;
            check(!f.controller.submit_switch_intent(f.intent()),
                    "drain begin event failure");
        } else {
            check(f.controller.submit_switch_intent(f.intent()), "switch");
            const auto drain =
                f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN);
            f.fail_event_kind = fail_kind;
            check(!f.controller.handle_executor_result(f.success(drain)),
                    "unload begin event failure");
        }
        const auto discard =
            f.gpu->take(SERVER_WARM_TIER_COMMAND_DISCARD);
        check(discard.model_id == "model-a"
                    && discard.executor_id == "gpu",
                "failed destructive begin discards loaded source");
        check(!f.controller.handle_executor_result(f.success(discard)),
                "failed destructive begin closes transition");
        check(!f.controller.has_active_transition(),
                "failed destructive begin is terminal");
    }
    {
        fixture f;
        check(f.controller.submit_switch_intent(f.intent()), "switch");
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        const auto unload =
            f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD);
        f.fail_event_kind = SERVER_WARM_TIER_EVENT_LOAD_BEGIN;
        check(!f.controller.handle_executor_result(f.success(unload)),
                "load begin event failure");
        check(!f.controller.has_active_transition(),
                "load begin event failure is terminal");
        server_warm_tier_model_state state;
        check(f.controller.get_model_state("model-b", "gpu", state)
                    && state == SERVER_WARM_TIER_MODEL_FAILED,
                "unstarted load invalidates target route");
        check(f.gpu->commands.empty(),
                "unstarted load queues no physical command");
    }
    {
        fixture f;
        check(f.controller.enqueue_request("r0", "model-b", {1}),
                "enqueue");
        check(f.controller.dispatch_request("r0", "phone", 2),
                "dispatch");
        const auto execute =
            f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
        check(f.controller.submit_switch_intent(f.intent()), "switch");
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
        f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
        auto execute_result = f.success(execute);
        execute_result.publications = {publication(execute, 0, 10)};
        f.fail_event_kind = SERVER_WARM_TIER_EVENT_REPLAY_BEGIN;
        check(!f.controller.handle_executor_result(execute_result),
                "replay begin event failure");
        const auto discard =
            f.gpu->take(SERVER_WARM_TIER_COMMAND_DISCARD);
        check(discard.model_id == "model-b"
                    && discard.executor_id == "gpu",
                "replay begin failure discards target");
        check(!f.controller.handle_executor_result(f.success(discard)),
                "replay begin discard closes transition");
        check(!f.controller.has_active_transition(),
                "replay begin failure is terminal");
        server_warm_tier_request_snapshot request;
        check(f.controller.get_request("r0", request)
                    && request.owner_id == "phone",
                "replay begin failure preserves owner");
    }
    {
        fixture f;
        check(f.controller.enqueue_request("r0", "model-b", {1}),
                "enqueue");
        check(f.controller.dispatch_request("r0", "phone", 1),
                "dispatch");
        const auto execute =
            f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
        auto result = f.success(execute);
        result.publications = {publication(execute, 0, 10)};
        result.request_complete = true;
        f.fail_event_kind = SERVER_WARM_TIER_EVENT_CLEANUP_BEGIN;
        check(!f.controller.handle_executor_result(result),
                "cleanup begin event failure");
        server_warm_tier_model_state state;
        check(f.controller.get_model_state("model-b", "phone", state)
                    && state == SERVER_WARM_TIER_MODEL_FAILED,
                "cleanup begin failure invalidates route");
        check(f.warm->commands.empty(),
                "cleanup begin failure queues no physical command");
    }
    {
        fixture f;
        check(f.controller.enqueue_request("r0", "model-b", {1}),
                "enqueue");
        check(f.controller.dispatch_request("r0", "phone", 2),
                "dispatch");
        const auto execute =
            f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
        auto result = f.success(execute);
        result.publications = {publication(execute, 0, 10)};
        f.fail_event_kind = SERVER_WARM_TIER_EVENT_REQUEST_DISPATCHED;
        check(!f.controller.handle_executor_result(result),
                "execute begin event failure");
        server_warm_tier_request_snapshot request;
        check(f.controller.get_request("r0", request)
                    && request.state == SERVER_WARM_TIER_REQUEST_STRANDED,
                "execute begin failure strands request");
        server_warm_tier_model_state state;
        check(f.controller.get_model_state("model-b", "phone", state)
                    && state == SERVER_WARM_TIER_MODEL_FAILED,
                "execute begin failure invalidates route");
    }
}

void test_replay_mismatch_rolls_back_owner() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
    check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
    auto execute = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(f.controller.submit_switch_intent(f.intent()), "switch");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));

    auto execute_result = f.success(execute);
    execute_result.publications = {publication(execute, 0, 10)};
    check(f.controller.handle_executor_result(execute_result), "warm quantum");
    auto replay = f.gpu->take(SERVER_WARM_TIER_COMMAND_REPLAY);
    auto replay_result = f.success(replay);
    replay_result.has_replay_snapshot = true;
    replay_result.replay_snapshot = replay.request;
    replay_result.replay_snapshot.position++;
    check(!f.controller.handle_executor_result(replay_result), "mismatch rejected");
    check(!f.controller.enqueue_request("during-rollback", "model-b", {2}),
            "rollback closes admission before discard");
    check(f.controller.last_error() == "E_POISONED",
            "rollback admission diagnostic");
    check(f.warm->commands.empty(),
            "rollback admits no new warm execute");
    auto discard = f.gpu->take(SERVER_WARM_TIER_COMMAND_DISCARD);
    check(!f.controller.handle_executor_result(f.success(discard)), "rollback remains failed");
    check(!f.controller.has_active_transition(), "rollback complete");

    server_warm_tier_request_snapshot request;
    check(f.controller.get_request("r0", request), "request after rollback");
    check(request.owner_id == "phone" && request.ownership_epoch == 1, "old owner preserved");
    server_warm_tier_model_state state;
    check(f.controller.get_model_state("model-b", "gpu", state)
            && state == SERVER_WARM_TIER_MODEL_FAILED, "target remains failed");
    check(!f.controller.enqueue_request("late", "model-b", {2}),
            "rollback poisons later admission");
    check(f.controller.last_error() == "E_POISONED", "rollback poison diagnostic");
}

void test_rollback_quarantines_concurrent_replay() {
    fixture f;
    std::vector<server_warm_tier_command> executes;
    for (const char * request_id : {"r0", "r1"}) {
        check(f.controller.enqueue_request(request_id, "model-b", {1}), "enqueue");
        check(f.controller.dispatch_request(request_id, "phone", 2), "dispatch");
        executes.push_back(f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE));
    }
    check(f.controller.submit_switch_intent(f.intent()), "switch");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
    for (const auto & execute : executes) {
        auto result = f.success(execute);
        result.publications = {publication(execute, 0, 10)};
        check(f.controller.handle_executor_result(result), "phone quantum");
    }

    const auto replay0 = f.gpu->take(SERVER_WARM_TIER_COMMAND_REPLAY);
    const auto replay1 = f.gpu->take(SERVER_WARM_TIER_COMMAND_REPLAY);
    auto mismatched = f.success(replay0);
    mismatched.has_replay_snapshot = true;
    mismatched.replay_snapshot = replay0.request;
    mismatched.replay_snapshot.position++;
    check(!f.controller.handle_executor_result(mismatched),
            "first replay triggers rollback");
    for (const auto & command : f.gpu->commands) {
        check(command.kind != SERVER_WARM_TIER_COMMAND_DISCARD,
                "discard waits for quarantined replay");
    }

    auto late = f.success(replay1);
    late.has_replay_snapshot = true;
    late.replay_snapshot = replay1.request;
    check(f.controller.handle_executor_result(late),
            "quarantined replay result consumed");
    const auto discard = f.gpu->take(SERVER_WARM_TIER_COMMAND_DISCARD);

    for (const auto & replay : {replay0, replay1}) {
        int begin_count = 0;
        int end_count = 0;
        for (const auto & event : f.events) {
            if (!event.has_command || event.command_id != replay.command_id) {
                continue;
            }
            begin_count += event.kind == SERVER_WARM_TIER_EVENT_REPLAY_BEGIN;
            end_count += event.kind == SERVER_WARM_TIER_EVENT_REPLAY_END;
            if (event.kind == SERVER_WARM_TIER_EVENT_REPLAY_END
                    && replay.command_id == replay1.command_id) {
                check(event.command_disposition == "QUARANTINED",
                        "late replay disposition");
                check(event.success, "late replay raw success preserved");
            }
        }
        check(begin_count == 1 && end_count == 1,
                "replay command has one begin and one end");
    }
    check(!f.controller.handle_executor_result(f.success(discard)),
            "rollback discard remains terminal failure");
}

void test_poison_quarantines_concurrent_execute() {
    fixture f;
    for (const char * request_id : {"r0", "r1"}) {
        check(f.controller.enqueue_request(request_id, "model-b", {1}), "enqueue");
        check(f.controller.dispatch_request(request_id, "phone", 1), "dispatch");
    }
    const auto execute0 = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    const auto execute1 = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    auto failed = f.success(execute0);
    failed.success = false;
    failed.detail = "injected execute failure";
    check(!f.controller.handle_executor_result(failed), "execute failure poisons");

    auto late = f.success(execute1);
    late.publications = {publication(execute1, 0, 11)};
    late.request_complete = true;
    check(f.controller.handle_executor_result(late),
            "quarantined execute result consumed");
    server_warm_tier_request_snapshot request;
    check(f.controller.get_request("r1", request), "late request snapshot");
    check(request.committed_output_tokens.empty(),
            "quarantined execute publishes no token");
    int begin_count = 0;
    int end_count = 0;
    bool quarantined = false;
    for (const auto & event : f.events) {
        if (!event.has_command || event.command_id != execute1.command_id) {
            continue;
        }
        begin_count += event.kind == SERVER_WARM_TIER_EVENT_REQUEST_DISPATCHED;
        end_count += event.kind == SERVER_WARM_TIER_EVENT_EXECUTE_END;
        quarantined |= event.kind == SERVER_WARM_TIER_EVENT_EXECUTE_END
                    && event.command_disposition == "QUARANTINED"
                    && event.result_publications.size() == 1
                    && event.result_request_complete;
    }
    check(begin_count == 1 && end_count == 1 && quarantined,
            "quarantined execute has exact raw terminal evidence");
}

void test_cleanup_failure_is_latched() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
    check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
    auto execute = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(f.controller.submit_switch_intent(f.intent()), "switch");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
    auto execute_result = f.success(execute);
    execute_result.publications = {publication(execute, 0, 10)};
    check(f.controller.handle_executor_result(execute_result), "warm quantum");
    auto replay = f.gpu->take(SERVER_WARM_TIER_COMMAND_REPLAY);
    auto replay_result = f.success(replay);
    replay_result.has_replay_snapshot = true;
    replay_result.replay_snapshot = replay.request;
    check(f.controller.handle_executor_result(replay_result), "replay");

    auto cleanup = f.warm->take(SERVER_WARM_TIER_COMMAND_CLEANUP);
    auto failed = f.success(cleanup);
    failed.success = false;
    failed.detail = "injected cleanup failure";
    check(!f.controller.handle_executor_result(failed), "cleanup failure");
    check(!f.controller.has_active_transition(), "failed cleanup terminal");
    check(f.controller.last_error().find("E_EXECUTOR_FAILED:CLEANUP") != std::string::npos,
            "cleanup failure latched");
}

void test_multi_request_cleanup_failure_is_terminal() {
    fixture f;
    std::vector<server_warm_tier_command> executes;
    for (const char * request_id : {"r0", "r1"}) {
        check(f.controller.enqueue_request(request_id, "model-b", {1}),
                "enqueue");
        check(f.controller.dispatch_request(request_id, "phone", 2),
                "dispatch");
        executes.push_back(
            f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE));
    }
    check(f.controller.submit_switch_intent(f.intent()), "switch");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
    for (const auto & execute : executes) {
        auto result = f.success(execute);
        result.publications = {publication(execute, 0, 10)};
        check(f.controller.handle_executor_result(result), "phone quantum");
    }
    for (int i = 0; i < 2; ++i) {
        const auto replay =
            f.gpu->take(SERVER_WARM_TIER_COMMAND_REPLAY);
        auto result = f.success(replay);
        result.has_replay_snapshot = true;
        result.replay_snapshot = replay.request;
        check(f.controller.handle_executor_result(result), "replay");
    }
    const auto cleanup =
        f.warm->take(SERVER_WARM_TIER_COMMAND_CLEANUP);
    auto failed = f.success(cleanup);
    failed.success = false;
    failed.detail = "injected cleanup failure";
    check(!f.controller.handle_executor_result(failed),
            "first cleanup failure");
    check(!f.controller.has_active_transition(),
            "multi-request cleanup failure is terminal");
    check(f.warm->commands.empty(),
            "no unprocessable cleanup is queued");
    server_warm_tier_model_state state;
    check(f.controller.get_model_state("model-b", "phone", state)
                && state == SERVER_WARM_TIER_MODEL_FAILED,
            "all old-owner state is invalidated");
}

void test_postcommit_execute_failure_is_terminal() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
    check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
    auto execute = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(f.controller.submit_switch_intent(f.intent()), "switch");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
    auto execute_result = f.success(execute);
    execute_result.publications = {publication(execute, 0, 10)};
    check(f.controller.handle_executor_result(execute_result), "warm quantum");
    auto replay = f.gpu->take(SERVER_WARM_TIER_COMMAND_REPLAY);
    auto replay_result = f.success(replay);
    replay_result.has_replay_snapshot = true;
    replay_result.replay_snapshot = replay.request;
    check(f.controller.handle_executor_result(replay_result), "replay");

    auto gpu_execute = f.gpu->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    auto failed = f.success(gpu_execute);
    failed.success = false;
    failed.detail = "injected promoted execute failure";
    check(!f.controller.handle_executor_result(failed), "promoted execute failure");
    check(!f.controller.has_active_transition(), "postcommit failure is terminal");
    server_warm_tier_request_snapshot request;
    check(f.controller.get_request("r0", request)
                && request.owner_id == "gpu"
                && request.ownership_epoch == 2
                && request.state == SERVER_WARM_TIER_REQUEST_STRANDED,
            "committed ownership is not rolled back");
    server_warm_tier_model_state state;
    check(f.controller.get_model_state("model-b", "gpu", state)
                && state == SERVER_WARM_TIER_MODEL_FAILED,
            "promoted route failed");
    check(f.controller.get_model_state("model-b", "phone", state)
                && state == SERVER_WARM_TIER_MODEL_READY,
            "warm route retained after postcommit failure");
}

void test_stop_rejects_live_request() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
    check(!f.controller.stop(), "stop with queued request");
    check(f.controller.last_error() == "E_STOP_REQUESTS", "stop diagnostic");
}

void test_finalize_strands_queued_requests_and_ends_run() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-a", {1}), "enqueue r0");
    check(f.controller.enqueue_request("r1", "model-b", {2}), "enqueue r1");
    check(f.controller.finalize_queued("HORIZON_REACHED"), "finalize");
    for (const char * request_id : {"r0", "r1"}) {
        server_warm_tier_request_snapshot request;
        check(f.controller.get_request(request_id, request)
                    && request.state == SERVER_WARM_TIER_REQUEST_STRANDED,
                "finalize strands queued request");
    }
    check(f.events.back().kind == SERVER_WARM_TIER_EVENT_RUN_END
                && f.events.back().detail == "HORIZON_REACHED",
            "finalize emits run end");
    check(!f.controller.finalize_queued("HORIZON_REACHED"),
            "finalize is one shot");
    check(f.controller.last_error() == "E_NOT_RUNNING", "second finalize diagnostic");
}

void test_trace_complete_requires_quiescence() {
    {
        fixture f;
        check(f.controller.finalize_queued("TRACE_COMPLETE"),
                "quiescent trace completion");
        check(f.controller.finalization_state()
                    == SERVER_WARM_TIER_FINALIZATION_FINALIZED,
                "trace completion final state");
        check(f.events.back().kind == SERVER_WARM_TIER_EVENT_RUN_END
                    && f.events.back().detail == "TRACE_COMPLETE",
                "trace completion run end");
    }
    {
        fixture f;
        check(f.controller.enqueue_request("queued", "model-b", {1}), "enqueue");
        check(!f.controller.finalize_queued("TRACE_COMPLETE"),
                "trace completion rejects live demand");
        check(f.controller.last_error() == "E_FINALIZE_REQUESTS",
                "trace completion live-demand diagnostic");
    }
}

void test_finalize_drains_only_active_work() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
    check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
    auto first = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(f.controller.enqueue_request("queued", "model-b", {2}), "queued request");
    check(f.controller.finalize_queued("HORIZON_REACHED"), "begin finalize");
    check(f.controller.finalization_state()
                == SERVER_WARM_TIER_FINALIZATION_DRAINING,
            "finalize drains active request");
    server_warm_tier_request_snapshot queued;
    check(f.controller.get_request("queued", queued)
                && queued.state == SERVER_WARM_TIER_REQUEST_STRANDED,
            "horizon strands queued request");
    check(!f.controller.enqueue_request("late", "model-b", {3}),
            "horizon closes admission");
    check(f.controller.last_error() == "E_FINALIZING", "closed admission diagnostic");
    check(!f.controller.submit_switch_intent(f.intent()), "horizon closes switching");
    check(f.controller.last_error() == "E_FINALIZING", "closed switch diagnostic");
    check(!f.controller.submit_model_switch(
                0, "auto", "model-a", "model-b"),
            "horizon closes policy switching");
    check(f.controller.last_error() == "E_FINALIZING",
            "closed policy switch diagnostic");
    auto obsolete = f.intent();
    auto replacement = obsolete;
    replacement.sequence = 1;
    replacement.intent_id = "switch-1";
    check(!f.controller.coalesce_switch_intent(obsolete, replacement),
            "horizon closes coalescing");
    check(f.controller.last_error() == "E_FINALIZING",
            "closed coalesce diagnostic");

    auto first_result = f.success(first);
    first_result.publications = {publication(first, 0, 10)};
    check(f.controller.handle_executor_result(first_result), "first draining quantum");
    auto second = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    auto second_result = f.success(second);
    second_result.publications = {publication(second, 0, 11)};
    second_result.request_complete = true;
    check(f.controller.handle_executor_result(second_result), "last draining quantum");
    f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_CLEANUP));
    check(f.controller.finalization_state()
                == SERVER_WARM_TIER_FINALIZATION_FINALIZED,
            "cleanup completes finalization");
    check(f.events.back().kind == SERVER_WARM_TIER_EVENT_RUN_END,
            "drained finalization emits run end");
}

void test_stale_owner_and_publication_are_rejected() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
    check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
    check(!f.controller.publish_token("r0", "phone", 0, 0, 1, 10), "stale epoch");
    check(f.controller.publish_token("r0", "phone", 1, 0, 1, 10), "publish");
    check(!f.controller.publish_token("r0", "phone", 1, 0, 2, 11), "duplicate publication");
    check(f.controller.last_error() == "E_PUBLICATION_DUPLICATE_OR_GAP",
            "duplicate diagnostic");
    check(f.controller.publish_token("r0", "phone", 1, 1, 2, 11), "second publish");
    check(f.controller.complete_request("r0", "phone", 1), "complete exact budget");
}

void test_direct_publication_requires_exact_budget() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
    check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
    check(f.controller.publish_token("r0", "phone", 1, 0, 1, 10), "first publish");
    check(!f.controller.complete_request("r0", "phone", 1), "early completion rejected");
    check(f.controller.last_error() == "E_OUTPUT_BUDGET", "early completion diagnostic");
    check(f.controller.publish_token("r0", "phone", 1, 1, 2, 11), "second publish");
    check(!f.controller.publish_token("r0", "phone", 1, 2, 3, 12), "overrun rejected");
    check(f.controller.last_error() == "E_OUTPUT_BUDGET", "overrun diagnostic");
    check(f.controller.complete_request("r0", "phone", 1), "exact completion");
}

void test_concurrent_arrivals_are_serialized() {
    fixture f;
    std::vector<std::thread> threads;
    for (int i = 0; i < 32; i++) {
        threads.emplace_back([&f, i]() {
            check(f.controller.enqueue_request(
                        "r" + std::to_string(i), "model-b", {i + 1}), "concurrent enqueue");
        });
    }
    for (auto & thread : threads) {
        thread.join();
    }
    for (int i = 0; i < 32; i++) {
        server_warm_tier_request_snapshot request;
        check(f.controller.get_request("r" + std::to_string(i), request), "concurrent request exists");
    }
}

void test_invalid_digest_poisoned() {
    std::atomic<int64_t> clock_ns{0};
    auto executor = std::make_shared<fake_executor>("gpu");
    server_warm_tier_controller controller({
        true,
        "bad-digest",
        [&clock_ns]() { return clock_ns.fetch_add(1); },
        [](const std::vector<llama_token> &, const std::vector<llama_token> &) {
            return std::string("not-sha256");
        },
        [](const server_warm_tier_event &) {},
        std::string(64, 'b'),
    });
    check(controller.register_executor(executor), "register");
    check(controller.set_initial_model_state(
                "model-a", "gpu", SERVER_WARM_TIER_MODEL_READY), "route");
    check(controller.start(), "start");
    check(!controller.enqueue_request("r0", "model-a", {1}), "invalid digest rejected");
    check(controller.last_error() == "E_HISTORY_SHA256", "invalid digest diagnostic");
    check(!controller.enqueue_request("r1", "model-a", {1}), "poisoned controller");
    check(controller.last_error() == "E_POISONED", "poison diagnostic");
}

void test_execute_event_failure_rolls_back_result() {
    for (int occurrence : {1, 2}) {
        fixture f(
            [](const std::vector<llama_token> &, const std::vector<llama_token> &) {
                return std::string(64, 'a');
            },
            SERVER_WARM_TIER_EVENT_TOKEN_COMMITTED,
            occurrence);
        check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
        check(f.controller.dispatch_request("r0", "phone", 2), "dispatch");
        auto command = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
        auto result = f.success(command);
        result.publications = {publication(command, 0, 10)};
        if (occurrence == 2) {
            check(f.controller.handle_executor_result(result), "first token event");
            command = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
            result = f.success(command);
            result.publications = {publication(command, 0, 11)};
            result.request_complete = true;
        }
        check(!f.controller.handle_executor_result(result), "token event failure");
        server_warm_tier_request_snapshot request;
        check(f.controller.get_request("r0", request), "request snapshot");
        check(request.state == SERVER_WARM_TIER_REQUEST_ACTIVE,
                "failed result remains active");
        const std::vector<llama_token> expected =
            occurrence == 1 ? std::vector<llama_token>{}
                            : std::vector<llama_token>{10};
        check(request.committed_output_tokens == expected,
                "failed quantum tokens rolled back");
        check(request.publication_index == expected.size()
                    && request.position == 1 + static_cast<int64_t>(expected.size()),
                "failed result frontier rolled back");
        check(f.controller.last_error() == "E_EVENT_SINK", "event sink diagnostic");
    }
}

void test_destructor_quiesces_owned_executor() {
    std::atomic<bool> callback_finished{false};
    auto controller = std::make_unique<server_warm_tier_controller>(
        server_warm_tier_options{
            true,
            "destructor",
            []() {
                return std::chrono::steady_clock::now().time_since_epoch().count();
            },
            [](const std::vector<llama_token> &, const std::vector<llama_token> &) {
                return std::string(64, 'a');
            },
            [](const server_warm_tier_event &) {},
            std::string(64, 'b'),
        });
    auto executor = std::make_shared<slow_callback_executor>();
    executor->sink = [&callback_finished, ptr = controller.get()](
            server_warm_tier_result result) {
        check(!ptr->handle_executor_result(result), "late callback rejected");
        callback_finished.store(true);
    };
    check(controller->register_executor(executor), "register slow executor");
    check(controller->set_executor_policy({
                "slow", SERVER_WARM_TIER_EXECUTOR_GPU, 0, 1}),
            "slow executor policy");
    check(controller->set_initial_model_state(
                "model", "slow", SERVER_WARM_TIER_MODEL_READY), "slow route");
    check(controller->start(), "start slow controller");
    check(controller->enqueue_request("r0", "model", {1}), "enqueue slow request");
    check(controller->dispatch_request("r0", "slow", 1), "dispatch slow request");
    executor.reset();
    controller.reset();
    check(callback_finished.load(), "controller destructor joined executor callback");
}

void test_bootstrap_activation_is_fail_closed() {
    std::atomic<int64_t> clock_ns{0};
    std::vector<server_warm_tier_event> events;
    server_warm_tier_controller controller(server_warm_tier_options{
        true,
        "bootstrap",
        [&clock_ns]() { return clock_ns.fetch_add(1); },
        [](const std::vector<llama_token> &, const std::vector<llama_token> &) {
            return std::string(64, 'a');
        },
        [&events](const server_warm_tier_event & event) {
            events.push_back(event);
        },
        std::string(64, 'b'),
    });
    auto gpu = std::make_shared<fake_executor>("gpu");
    auto phone = std::make_shared<fake_executor>("phone");
    check(controller.register_executor(gpu), "bootstrap register gpu");
    check(controller.register_executor(phone), "bootstrap register phone");
    check(controller.set_executor_policy({
                "gpu", SERVER_WARM_TIER_EXECUTOR_GPU, 0, 1}), "bootstrap gpu policy");
    check(controller.set_executor_policy({
                "phone", SERVER_WARM_TIER_EXECUTOR_PHONE, 1, 1}), "bootstrap phone policy");
    check(controller.set_initial_model_state(
                "model-a", "gpu", SERVER_WARM_TIER_MODEL_ABSENT),
            "bootstrap gpu route");
    check(controller.set_initial_model_state(
                "model-b", "phone", SERVER_WARM_TIER_MODEL_ABSENT),
            "bootstrap phone route");
    check(controller.set_bootstrap_model_ready("model-a", "gpu"),
            "bootstrap gpu target");
    check(controller.set_bootstrap_model_ready("model-b", "phone"),
            "bootstrap phone target");
    check(controller.start(), "bootstrap start");
    check(controller.activation_state() == SERVER_WARM_TIER_ACTIVATION_WAITING,
            "bootstrap initially waiting");
    check(!controller.enqueue_request("early", "model-a", {1}),
            "bootstrap rejects early admission");
    check(controller.last_error() == "E_NOT_ACTIVATED",
            "bootstrap early admission diagnostic");
    check(controller.activate(), "bootstrap activate");
    check(controller.activation_state() == SERVER_WARM_TIER_ACTIVATION_PREPARING,
            "bootstrap preparing");
    check(gpu->commands.size() == 1 && phone->commands.empty(),
            "bootstrap loads serially");
    check(controller.activate(), "bootstrap activate idempotent");
    check(gpu->commands.size() == 1, "bootstrap no duplicate load");

    auto gpu_load = gpu->take(SERVER_WARM_TIER_COMMAND_LOAD);
    server_warm_tier_result result;
    result.command_id = gpu_load.command_id;
    result.controller_epoch = gpu_load.controller_epoch;
    result.kind = gpu_load.kind;
    result.model_id = gpu_load.model_id;
    result.request_id = gpu_load.request_id;
    result.executor_id = gpu_load.executor_id;
    result.executor_instance_id = gpu_load.executor_instance_id;
    result.success = true;
    check(controller.handle_executor_result(result), "bootstrap gpu loaded");
    check(phone->commands.size() == 1, "bootstrap phone load issued");
    auto phone_load = phone->take(SERVER_WARM_TIER_COMMAND_LOAD);
    result.command_id = phone_load.command_id;
    result.controller_epoch = phone_load.controller_epoch;
    result.kind = phone_load.kind;
    result.model_id = phone_load.model_id;
    result.request_id = phone_load.request_id;
    result.executor_id = phone_load.executor_id;
    result.executor_instance_id = phone_load.executor_instance_id;
    check(controller.handle_executor_result(result), "bootstrap phone loaded");
    check(controller.activation_state() == SERVER_WARM_TIER_ACTIVATION_READY,
            "bootstrap ready");
    check(controller.enqueue_request("ready", "model-a", {1}),
            "bootstrap admits after readiness");

    bool saw_command_id = false;
    for (const auto & event : events) {
        if (event.kind == SERVER_WARM_TIER_EVENT_LOAD_BEGIN) {
            saw_command_id = saw_command_id
                || (event.has_command
                    && event.command_id > 0
                    && event.command_kind == SERVER_WARM_TIER_COMMAND_LOAD);
        }
    }
    check(saw_command_id, "bootstrap command identity persisted");
}

void test_bootstrap_failure_poisoned() {
    std::atomic<int64_t> clock_ns{0};
    server_warm_tier_controller controller(server_warm_tier_options{
        true,
        "bootstrap-failure",
        [&clock_ns]() { return clock_ns.fetch_add(1); },
        [](const std::vector<llama_token> &, const std::vector<llama_token> &) {
            return std::string(64, 'a');
        },
        [](const server_warm_tier_event &) {},
        std::string(64, 'b'),
    });
    auto gpu = std::make_shared<fake_executor>("gpu");
    check(controller.register_executor(gpu), "failed bootstrap register");
    check(controller.set_executor_policy({
                "gpu", SERVER_WARM_TIER_EXECUTOR_GPU, 0, 1}),
            "failed bootstrap policy");
    check(controller.set_initial_model_state(
                "model-a", "gpu", SERVER_WARM_TIER_MODEL_ABSENT),
            "failed bootstrap route");
    check(controller.set_bootstrap_model_ready("model-a", "gpu"),
            "failed bootstrap target");
    check(controller.start(), "failed bootstrap start");
    gpu->reject_next = true;
    check(!controller.activate(), "failed bootstrap rejected");
    check(controller.activation_state() == SERVER_WARM_TIER_ACTIVATION_FAILED,
            "failed bootstrap state");
    server_warm_tier_model_state state;
    check(controller.get_model_state("model-a", "gpu", state)
                && state == SERVER_WARM_TIER_MODEL_FAILED,
            "failed bootstrap route state");
}

void test_bootstrap_begin_event_failure_is_terminal() {
    std::atomic<int64_t> clock_ns{0};
    bool fail_enabled = false;
    server_warm_tier_controller controller(server_warm_tier_options{
        true,
        "bootstrap-begin-failure",
        [&clock_ns]() { return clock_ns.fetch_add(1); },
        [](const std::vector<llama_token> &, const std::vector<llama_token> &) {
            return std::string(64, 'a');
        },
        [&fail_enabled](const server_warm_tier_event & event) {
            if (fail_enabled
                    && event.kind == SERVER_WARM_TIER_EVENT_LOAD_BEGIN) {
                throw std::runtime_error("injected event sink failure");
            }
        },
        std::string(64, 'b'),
    });
    auto gpu = std::make_shared<fake_executor>("gpu");
    check(controller.register_executor(gpu), "bootstrap begin register");
    check(controller.set_executor_policy({
                "gpu", SERVER_WARM_TIER_EXECUTOR_GPU, 0, 1}),
            "bootstrap begin policy");
    check(controller.set_initial_model_state(
                "model-a", "gpu", SERVER_WARM_TIER_MODEL_ABSENT),
            "bootstrap begin route");
    check(controller.set_bootstrap_model_ready("model-a", "gpu"),
            "bootstrap begin target");
    check(controller.start(), "bootstrap begin start");
    fail_enabled = true;
    check(!controller.activate(), "bootstrap load begin failure");
    check(controller.activation_state() == SERVER_WARM_TIER_ACTIVATION_FAILED,
            "bootstrap begin failure is terminal");
    server_warm_tier_model_state state;
    check(controller.get_model_state("model-a", "gpu", state)
                && state == SERVER_WARM_TIER_MODEL_FAILED,
            "bootstrap begin failure invalidates route");
    check(gpu->commands.empty(),
            "bootstrap begin failure starts no physical load");
}

void test_bootstrap_end_event_failure_discards_loaded_route() {
    for (const auto fail_kind : {
            SERVER_WARM_TIER_EVENT_LOAD_END,
            SERVER_WARM_TIER_EVENT_MODEL_STATE_CHANGED}) {
        std::atomic<int64_t> clock_ns{0};
        bool fail_enabled = false;
        server_warm_tier_controller controller(server_warm_tier_options{
            true,
            "bootstrap-event-failure",
            [&clock_ns]() { return clock_ns.fetch_add(1); },
            [](const std::vector<llama_token> &, const std::vector<llama_token> &) {
                return std::string(64, 'a');
            },
            [&fail_enabled, fail_kind](const server_warm_tier_event & event) {
                if (fail_enabled && event.kind == fail_kind) {
                    throw std::runtime_error("injected event sink failure");
                }
            },
            std::string(64, 'b'),
        });
        auto gpu = std::make_shared<fake_executor>("gpu");
        check(controller.register_executor(gpu), "bootstrap event register");
        check(controller.set_executor_policy({
                    "gpu", SERVER_WARM_TIER_EXECUTOR_GPU, 0, 1}),
                "bootstrap event policy");
        check(controller.set_initial_model_state(
                    "model-a", "gpu", SERVER_WARM_TIER_MODEL_ABSENT),
                "bootstrap event route");
        check(controller.set_bootstrap_model_ready("model-a", "gpu"),
                "bootstrap event target");
        check(controller.start(), "bootstrap event start");
        check(controller.activate(), "bootstrap event activate");
        const auto load =
            gpu->take(SERVER_WARM_TIER_COMMAND_LOAD);
        fail_enabled = true;
        server_warm_tier_result load_result;
        load_result.command_id = load.command_id;
        load_result.controller_epoch = load.controller_epoch;
        load_result.kind = load.kind;
        load_result.model_id = load.model_id;
        load_result.request_id = load.request_id;
        load_result.executor_id = load.executor_id;
        load_result.executor_instance_id = load.executor_instance_id;
        load_result.success = true;
        check(!controller.handle_executor_result(load_result),
                "bootstrap load event failure");
        const auto discard =
            gpu->take(SERVER_WARM_TIER_COMMAND_DISCARD);
        check(discard.model_id == "model-a"
                    && discard.executor_id == "gpu",
                "bootstrap event failure discards loaded route");
        server_warm_tier_result discard_result;
        discard_result.command_id = discard.command_id;
        discard_result.controller_epoch = discard.controller_epoch;
        discard_result.kind = discard.kind;
        discard_result.model_id = discard.model_id;
        discard_result.request_id = discard.request_id;
        discard_result.executor_id = discard.executor_id;
        discard_result.executor_instance_id = discard.executor_instance_id;
        discard_result.success = true;
        check(!controller.handle_executor_result(discard_result),
                "bootstrap discard closes failed activation");
        check(controller.activation_state() == SERVER_WARM_TIER_ACTIVATION_FAILED,
                "bootstrap event failure is terminal");
        server_warm_tier_model_state state;
        check(controller.get_model_state("model-a", "gpu", state)
                    && state == SERVER_WARM_TIER_MODEL_FAILED,
                "bootstrap event failure invalidates route");
    }
}

void test_event_json_shape() {
    fixture f;
    check(f.controller.enqueue_request("r0", "model-b", {1}), "enqueue");
    const auto line = server_warm_tier_event_jsonl(f.events.back());
    check(line.find("\"schema\":\"s40-warm-tier-event-v3\"") != std::string::npos,
            "event schema");
    check(line.find("\"command_id\":null") != std::string::npos,
            "non-command event command id null");
    check(line.find("\"owner_id\":null") != std::string::npos, "queued owner null");
    check(line.find("\"history_sha256\":\"") != std::string::npos, "history sha");
    check(!line.empty() && line.back() == '\n', "event newline");
}

void test_strict_http_json_rejects_duplicate_keys() {
    nlohmann::json parsed;
    std::string error;
    check(server_warm_tier_parse_json_strict(
                R"({"items":[{"x":1},{"x":2}],"schema":"valid"})",
                parsed,
                error),
            "nested sibling objects are valid");
    const std::vector<std::string> duplicate_endpoint_bodies = {
        R"({"schema":"llama-server-warm-tier-request-v1","request_id":"r0","request_id":"r1"})",
        R"({"schema":"llama-server-warm-tier-activate-v1","schema":"other"})",
        R"({"schema":"llama-server-warm-tier-completion-v1","model":"a","model":"b"})",
        R"({"schema":"llama-server-warm-tier-switch-v1","sequence":0,"sequence":1})",
        R"({"schema":"llama-server-warm-tier-finalize-v1","reason":"HORIZON_REACHED","reason":"OTHER"})",
    };
    for (const auto & body : duplicate_endpoint_bodies) {
        check(!server_warm_tier_parse_json_strict(body, parsed, error),
                "duplicate endpoint key rejected");
        check(error == "duplicate JSON key", "duplicate endpoint diagnostic");
    }
    check(!server_warm_tier_parse_json_strict("{", parsed, error),
            "malformed endpoint JSON rejected");
}

void write_event_fixture(const std::string & path) {
    fixture f([](
            const std::vector<llama_token> & prompt,
            const std::vector<llama_token> & committed) {
        check(prompt == std::vector<llama_token>({1, 2}), "fixture digest prompt");
        if (committed.empty()) {
            return std::string("8dfbabf09c2c7e151b88da7aedc50bb2d6c2c1d67920a881cad42d1c63fb9a2e");
        }
        if (committed == std::vector<llama_token>({10})) {
            return std::string("040f1ff5b92548a0df1c03aad9295146161ef14026d8dd59294df6549aa87307");
        }
        check(committed == std::vector<llama_token>({10, 11}), "fixture digest output");
        return std::string("6a79c356999e9c25bff982678799068b356b7ebc5b3091125d94b263ed2fed53");
    });
    check(f.controller.enqueue_request("r0", "model-b", {1, 2}), "fixture enqueue");
    check(f.controller.dispatch_request("r0", "phone", 2), "fixture dispatch");
    auto execute = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(f.controller.submit_switch_intent(f.intent()), "fixture switch");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
    auto execute_result = f.success(execute);
    execute_result.publications = {publication(execute, 0, 10)};
    check(f.controller.handle_executor_result(execute_result), "fixture execute");
    auto replay = f.gpu->take(SERVER_WARM_TIER_COMMAND_REPLAY);
    auto replay_result = f.success(replay);
    replay_result.has_replay_snapshot = true;
    replay_result.replay_snapshot = replay.request;
    check(f.controller.handle_executor_result(replay_result), "fixture replay");
    auto gpu_execute = f.gpu->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    auto gpu_result = f.success(gpu_execute);
    gpu_result.publications = {publication(gpu_execute, 0, 11)};
    gpu_result.request_complete = true;
    check(f.controller.handle_executor_result(gpu_result), "fixture gpu continuation");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_CLEANUP));
    f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_CLEANUP));
    f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.warm->take(SERVER_WARM_TIER_COMMAND_LOAD));
    check(f.controller.emit_phone_telemetry(
                "model-a", "phone", "{\"thermal_status\":0}"), "fixture telemetry");
    check(f.controller.stop(), "fixture stop");

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    check(output.good(), "open event fixture");
    for (const auto & event : f.events) {
        output << server_warm_tier_event_jsonl(event);
    }
    output.close();
    check(output.good(), "write event fixture");
}

void write_failed_precommit_event_fixture(const std::string & path) {
    fixture f([](
            const std::vector<llama_token> & prompt,
            const std::vector<llama_token> & committed) {
        check(prompt == std::vector<llama_token>({1, 2}), "failed fixture prompt");
        if (committed.empty()) {
            return std::string("8dfbabf09c2c7e151b88da7aedc50bb2d6c2c1d67920a881cad42d1c63fb9a2e");
        }
        if (committed == std::vector<llama_token>({10})) {
            return std::string("040f1ff5b92548a0df1c03aad9295146161ef14026d8dd59294df6549aa87307");
        }
        check(committed == std::vector<llama_token>({10, 11}), "failed fixture output");
        return std::string("6a79c356999e9c25bff982678799068b356b7ebc5b3091125d94b263ed2fed53");
    });
    check(f.controller.enqueue_request("r0", "model-b", {1, 2}), "failed fixture enqueue");
    check(f.controller.dispatch_request("r0", "phone", 2), "failed fixture dispatch");
    auto execute = f.warm->take(SERVER_WARM_TIER_COMMAND_EXECUTE);
    check(f.controller.submit_switch_intent(f.intent()), "failed fixture switch");
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_DRAIN));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_UNLOAD));
    f.complete(f.gpu->take(SERVER_WARM_TIER_COMMAND_LOAD));
    auto execute_result = f.success(execute);
    execute_result.publications = {publication(execute, 0, 10)};
    check(f.controller.handle_executor_result(execute_result), "failed fixture execute");
    auto replay = f.gpu->take(SERVER_WARM_TIER_COMMAND_REPLAY);
    auto replay_result = f.success(replay);
    replay_result.has_replay_snapshot = true;
    replay_result.replay_snapshot = replay.request;
    replay_result.replay_snapshot.position++;
    check(!f.controller.handle_executor_result(replay_result), "failed fixture mismatch");
    auto discard = f.gpu->take(SERVER_WARM_TIER_COMMAND_DISCARD);
    check(!f.controller.handle_executor_result(f.success(discard)), "failed fixture discard");

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    check(output.good(), "open failed event fixture");
    for (const auto & event : f.events) {
        output << server_warm_tier_event_jsonl(event);
    }
    output.close();
    check(output.good(), "write failed event fixture");
}

}

int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "--dump-events") {
        write_event_fixture(argv[2]);
        return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--dump-failed-precommit-events") {
        write_failed_precommit_event_fixture(argv[2]);
        return 0;
    }
    check(argc == 1, "invalid arguments");
    test_execute_invokes_selected_executor();
    test_incomplete_execute_has_followup();
    test_executor_instance_mismatch_is_rejected();
    test_scheduled_fifo_respects_executor_credit();
    test_direct_dispatch_respects_executor_credit();
    test_replay_refuses_insufficient_target_credit();
    test_scheduled_request_starts_gpu_only_switch();
    test_overlapping_demand_queues_reverse_promotion();
    test_negative_prompt_token_is_rejected();
    test_execute_budget_failures();
    test_rotation_replays_and_rearms_warm_tier();
    test_completed_phone_quantum_does_not_stall_replay();
    test_gpu_only_switch_has_no_warm_lifecycle();
    test_source_quanta_finish_before_drain_and_unload();
    test_source_execute_failure_aborts_drain_transition();
    test_lifecycle_failure_poisoned_with_rollback_cleanup();
    test_terminal_cleanup_failure_poisoned();
    test_cleanup_advances_across_all_replayed_requests();
    test_target_cleanup_does_not_satisfy_old_owner_cleanup();
    test_target_cleanup_blocks_warm_drain_after_gpu_load();
    test_incomplete_ownership_transaction_preserves_old_owner();
    test_precommit_end_event_failure_discards_target();
    test_replay_state_event_failure_discards_target();
    test_lifecycle_end_event_failure_is_terminal();
    test_post_lifecycle_state_event_failure_is_terminal();
    test_chained_begin_event_failure_is_terminal();
    test_replay_mismatch_rolls_back_owner();
    test_rollback_quarantines_concurrent_replay();
    test_poison_quarantines_concurrent_execute();
    test_cleanup_failure_is_latched();
    test_multi_request_cleanup_failure_is_terminal();
    test_postcommit_execute_failure_is_terminal();
    test_stop_rejects_live_request();
    test_finalize_strands_queued_requests_and_ends_run();
    test_trace_complete_requires_quiescence();
    test_finalize_drains_only_active_work();
    test_stale_owner_and_publication_are_rejected();
    test_direct_publication_requires_exact_budget();
    test_concurrent_arrivals_are_serialized();
    test_invalid_digest_poisoned();
    test_execute_event_failure_rolls_back_result();
    test_destructor_quiesces_owned_executor();
    test_bootstrap_activation_is_fail_closed();
    test_bootstrap_failure_poisoned();
    test_bootstrap_begin_event_failure_is_terminal();
    test_bootstrap_end_event_failure_discards_loaded_route();
    test_event_json_shape();
    test_strict_http_json_rejects_duplicate_keys();
    return 0;
}
