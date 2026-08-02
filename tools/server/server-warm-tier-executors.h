#pragma once

#include "server-warm-tier.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

using server_warm_tier_executor_result_sink =
    std::function<void(server_warm_tier_result result)>;

struct server_warm_tier_unix_executor_options {
    std::string executor_id;
    std::string executor_instance_id;
    std::string socket_path;
    int64_t expected_peer_pid = 0;
    uint64_t expected_peer_start_time_ticks = 0;
    server_warm_tier_executor_result_sink result_sink;
    int32_t timeout_ms = 300000;
    size_t execute_concurrency = 4;
    size_t queue_capacity = 64;
    size_t output_limit_bytes = 4 * 1024 * 1024;
};

std::shared_ptr<server_warm_tier_executor>
server_warm_tier_create_unix_executor(
        server_warm_tier_unix_executor_options options);
