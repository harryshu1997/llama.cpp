#include "common.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr size_t MAX_LINE_BYTES = 1024 * 1024;
constexpr size_t MAX_TEXT_BYTES = 256 * 1024;
constexpr size_t MAX_TOKENS = 16 * 1024;

bool is_digest(const std::string & value) {
    if (value.size() != 64) {
        return false;
    }
    for (char character : value) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

bool read_bounded_line(std::istream & input, std::string & line) {
    line.clear();
    char character = 0;
    while (input.get(character)) {
        if (character == '\n') {
            return true;
        }
        if (line.size() >= MAX_LINE_BYTES) {
            return false;
        }
        line.push_back(character);
    }
    return input.eof() && line.empty();
}

bool parse_request(
        const std::string & line,
        int64_t previous_id,
        nlohmann::json & request,
        std::string & error) {
    if (line.empty() || line.size() > MAX_LINE_BYTES) {
        error = "request line has an invalid byte length";
        return false;
    }
    try {
        std::set<std::string> keys;
        auto callback = [&](int, nlohmann::json::parse_event_t event,
                            nlohmann::json & parsed) {
            if (event == nlohmann::json::parse_event_t::key) {
                const std::string key = parsed.get<std::string>();
                if (!keys.insert(key).second) {
                    throw std::runtime_error("duplicate request key");
                }
            }
            return true;
        };
        request = nlohmann::json::parse(line, callback, true, false);
        if (!request.is_object() ||
            request.dump(-1, ' ', true) != line ||
            !request.contains("schema") ||
            !request["schema"].is_string() ||
            request["schema"].get<std::string>() !=
                "layersplit-token-codec-request-v1" ||
            !request.contains("op") ||
            !request["op"].is_string() ||
            !request.contains("request_id") ||
            (!request["request_id"].is_number_integer() &&
             !request["request_id"].is_number_unsigned())) {
            throw std::runtime_error("invalid request envelope");
        }
        const int64_t request_id = request["request_id"].get<int64_t>();
        if (request_id <= previous_id) {
            throw std::runtime_error("request_id is stale or duplicated");
        }
        const std::string op = request["op"].get<std::string>();
        if (op == "tokenize") {
            const std::set<std::string> expected = {
                "op", "request_id", "schema", "text",
            };
            if (keys != expected || request.size() != expected.size() ||
                !request["text"].is_string() ||
                request["text"].get_ref<const std::string &>().size() >
                    MAX_TEXT_BYTES) {
                throw std::runtime_error("invalid tokenize request");
            }
        } else if (op == "detokenize") {
            const std::set<std::string> expected = {
                "op", "request_id", "schema", "tokens",
            };
            if (keys != expected || request.size() != expected.size() ||
                !request["tokens"].is_array() ||
                request["tokens"].empty() ||
                request["tokens"].size() > MAX_TOKENS) {
                throw std::runtime_error("invalid detokenize request");
            }
            for (const auto & token : request["tokens"]) {
                if ((!token.is_number_integer() &&
                     !token.is_number_unsigned()) ||
                    token.get<int64_t>() < 0 ||
                    token.get<int64_t>() >
                        std::numeric_limits<llama_token>::max()) {
                    throw std::runtime_error("invalid token id");
                }
            }
        } else {
            throw std::runtime_error("unknown codec operation");
        }
    } catch (const std::exception & exception) {
        error = exception.what();
        return false;
    }
    return true;
}

void print_usage(const char * program) {
    std::cerr
        << "usage: " << program
        << " --model <absolute.gguf> --model-sha256 <lowercase-hex>\n";
}

} // namespace

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    std::string model_sha256;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--model" && index + 1 < argc) {
            model_path = argv[++index];
        } else if (argument == "--model-sha256" && index + 1 < argc) {
            model_sha256 = argv[++index];
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }
    if (model_path == nullptr || model_path[0] != '/' ||
        !is_digest(model_sha256)) {
        print_usage(argv[0]);
        return 2;
    }

    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path, params);
    if (model == nullptr) {
        std::cerr << "token-codec: model load failed\n";
        llama_backend_free();
        return 3;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (vocab == nullptr || llama_vocab_n_tokens(vocab) <= 0) {
        std::cerr << "token-codec: vocabulary is unavailable\n";
        llama_model_free(model);
        llama_backend_free();
        return 3;
    }

    int64_t previous_id = -1;
    int result = 0;
    size_t request_count = 0;
    while (true) {
        std::string line;
        if (!read_bounded_line(std::cin, line)) {
            std::cerr << "token-codec: request line exceeds limit\n";
            result = 4;
            break;
        }
        if (std::cin.eof() && line.empty()) {
            break;
        }

        nlohmann::json request;
        std::string error;
        if (!parse_request(line, previous_id, request, error)) {
            std::cerr << "token-codec: " << error << "\n";
            result = 4;
            break;
        }
        const int64_t request_id = request["request_id"].get<int64_t>();
        const std::string op = request["op"].get<std::string>();
        nlohmann::json response = {
            {"model_sha256", model_sha256},
            {"op", op},
            {"request_id", request_id},
            {"schema", "layersplit-token-codec-response-v1"},
        };
        if (op == "tokenize") {
            const std::string text = request["text"].get<std::string>();
            const std::vector<llama_token> tokens =
                common_tokenize(vocab, text, true, true);
            if (tokens.empty() || tokens.size() > MAX_TOKENS) {
                std::cerr << "token-codec: tokenization failed or exceeds limit\n";
                result = 4;
                break;
            }
            response["tokens"] = tokens;
        } else {
            const std::vector<llama_token> tokens =
                request["tokens"].get<std::vector<llama_token>>();
            for (llama_token token : tokens) {
                if (token < 0 || token >= llama_vocab_n_tokens(vocab)) {
                    std::cerr << "token-codec: token id is outside vocabulary\n";
                    result = 4;
                    break;
                }
            }
            if (result != 0) {
                break;
            }
            response["text"] = common_detokenize(vocab, tokens, false);
        }
        std::cout << response.dump(-1, ' ', true) << '\n';
        std::cout.flush();
        if (!std::cout.good()) {
            std::cerr << "token-codec: response write failed\n";
            result = 4;
            break;
        }
        previous_id = request_id;
        ++request_count;
    }
    if (result == 0 && request_count == 0) {
        std::cerr << "token-codec: no requests received\n";
        result = 4;
    }

    llama_model_free(model);
    llama_backend_free();
    return result;
}
