#pragma once

#include "server-warm-tier.h"

#include <memory>
#include <string>
#include <vector>


std::shared_ptr<server_warm_tier_controller>
server_warm_tier_create_runtime_from_env();

std::string server_warm_tier_internal_token_from_env();

std::string server_warm_tier_history_sha256(
        const std::vector<llama_token> & prompt_tokens,
        const std::vector<llama_token> & committed_output_tokens);
