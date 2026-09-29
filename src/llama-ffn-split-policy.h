#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace llama_ffn_split_policy {

struct point {
    uint32_t max_tokens;
    uint32_t columns;
};

inline bool parse(
        const char * text,
        uint32_t max_tokens,
        uint32_t max_columns,
        std::vector<point> & result,
        std::string & error) {
    result.clear();
    if (text == nullptr || text[0] == '\0' || max_tokens == 0) {
        error = "empty FFN split policy";
        return false;
    }

    const char * cursor = text;
    uint32_t previous = 0;
    while (*cursor != '\0') {
        errno = 0;
        char * end = nullptr;
        const unsigned long long threshold = std::strtoull(cursor, &end, 10);
        if (errno != 0 || end == cursor || *end != ':' ||
            threshold <= previous || threshold > max_tokens) {
            error = "invalid FFN split policy threshold";
            return false;
        }

        cursor = end + 1;
        errno = 0;
        const unsigned long long columns = std::strtoull(cursor, &end, 10);
        if (errno != 0 || end == cursor || columns > max_columns ||
            (*end != ',' && *end != '\0')) {
            error = "invalid FFN split policy width";
            return false;
        }

        result.push_back({
            static_cast<uint32_t>(threshold),
            static_cast<uint32_t>(columns),
        });
        previous = static_cast<uint32_t>(threshold);
        if (*end == '\0') {
            break;
        }
        cursor = end + 1;
    }

    if (result.empty() || result.back().max_tokens != max_tokens) {
        error = "FFN split policy does not cover the maximum batch";
        return false;
    }
    return true;
}

inline uint32_t select(const std::vector<point> & policy, uint32_t tokens) {
    for (const point & value : policy) {
        if (tokens <= value.max_tokens) {
            return value.columns;
        }
    }
    return 0;
}

} // namespace llama_ffn_split_policy
