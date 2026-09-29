#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace ffn_split {

struct resident_session_shard {
    std::string session_id;
    std::string backend;
    std::string artifact_sha256;
    std::string model;
    std::string layers;
    uint64_t layer_mask = 0;
    int64_t columns = 0;
    int port = 0;
    std::string endpoint_sha256;
    size_t resident_bytes = 0;
    std::string resident_geometry_sha256;
    std::string operator_plan_sha256;
    uint64_t session_generation = 1;
};

inline bool session_text(const std::string & value) {
    if (value.empty()) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return (character >= 'a' && character <= 'z') ||
                (character >= 'A' && character <= 'Z') ||
                (character >= '0' && character <= '9') ||
                character == '.' || character == '_' || character == '-';
    });
}

inline bool sha256_text(const std::string & value) {
    if (value.size() != 71 || value.compare(0, 7, "sha256:") != 0) {
        return false;
    }
    return std::all_of(value.begin() + 7, value.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f');
    });
}

inline bool path_text(const std::string & value) {
    return !value.empty() && value[0] == '/' &&
            std::all_of(value.begin(), value.end(), [](unsigned char character) {
                return character >= 0x21 && character <= 0x7e && character != ',';
            });
}

inline bool parse_unsigned(
        const std::string & text, uint64_t minimum, uint64_t maximum,
        uint64_t & value) {
    if (text.empty()) {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' ||
        parsed < minimum || parsed > maximum) {
        return false;
    }
    value = static_cast<uint64_t>(parsed);
    return true;
}

inline bool parse_layer_spec(const std::string & text, uint64_t & mask) {
    if (text.empty()) {
        return false;
    }
    uint64_t result = 0;
    size_t offset = 0;
    while (offset < text.size()) {
        const size_t comma = text.find(',', offset);
        const std::string item = text.substr(
                offset, comma == std::string::npos ? std::string::npos : comma - offset);
        const size_t dash = item.find('-');
        if (item.empty() || (dash != std::string::npos &&
                            item.find('-', dash + 1) != std::string::npos)) {
            return false;
        }
        uint64_t first = 0;
        uint64_t last = 0;
        if (!parse_unsigned(item.substr(0, dash), 0, 63, first) ||
            !parse_unsigned(
                    dash == std::string::npos ? item : item.substr(dash + 1),
                    first, 63, last)) {
            return false;
        }
        for (uint64_t layer = first; layer <= last; ++layer) {
            result |= UINT64_C(1) << layer;
        }
        if (comma == std::string::npos) {
            break;
        }
        offset = comma + 1;
    }
    mask = result;
    return mask != 0;
}

inline bool load_resident_session_manifest(
        const std::string & path, std::vector<resident_session_shard> & rows,
        std::string & error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open resident session manifest";
        return false;
    }
    std::string line;
    std::map<std::string, uint64_t> covered_layers_by_artifact;
    std::set<std::string> session_ids;
    std::set<int> ports;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields;
        size_t offset = 0;
        for (;;) {
            const size_t comma = line.find(',', offset);
            fields.push_back(line.substr(
                    offset,
                    comma == std::string::npos ? std::string::npos : comma - offset));
            if (comma == std::string::npos) {
                break;
            }
            offset = comma + 1;
        }
        resident_session_shard row;
        uint64_t columns = 0;
        uint64_t port = 0;
        uint64_t resident_bytes = 0;
        const bool fenced = fields.size() == 12;
        const bool current = fields.size() == 11 || fenced;
        const size_t layer_index = current ? 4 : 2;
        const size_t column_index = current ? 5 : 3;
        const size_t port_index = current ? 6 : 4;
        const size_t endpoint_index = current ? 7 : 5;
        const size_t bytes_index = current ? 8 : 6;
        const size_t geometry_index = current ? 9 : 7;
        const size_t plan_index = current ? 10 : 8;
        uint64_t session_generation = 1;
        const std::string artifact = current ?
                fields[2] : "legacy-single-artifact";
        if ((fields.size() != 9 && !current) ||
            !session_text(fields[0]) || !session_text(fields[1]) ||
            (current && (!sha256_text(fields[2]) || !path_text(fields[3]))) ||
            !parse_layer_spec(fields[layer_index], row.layer_mask) ||
            !parse_unsigned(fields[column_index], 1, INT64_MAX, columns) ||
            !parse_unsigned(fields[port_index], 1, 65535, port) ||
            !sha256_text(fields[endpoint_index]) ||
            !parse_unsigned(fields[bytes_index], 1, std::numeric_limits<size_t>::max(),
                            resident_bytes) ||
            !sha256_text(fields[geometry_index]) ||
            !sha256_text(fields[plan_index]) ||
            (fenced && !parse_unsigned(
                    fields[11], 1, std::numeric_limits<uint64_t>::max(),
                    session_generation)) ||
            !session_ids.insert(fields[0]).second ||
            !ports.insert(static_cast<int>(port)).second ||
            (covered_layers_by_artifact[artifact] & row.layer_mask) != 0) {
            error = "resident session manifest row is invalid";
            return false;
        }
        row.session_id = fields[0];
        row.backend = fields[1];
        row.artifact_sha256 = current ? fields[2] : "";
        row.model = current ? fields[3] : "";
        row.layers = fields[layer_index];
        row.columns = static_cast<int64_t>(columns);
        row.port = static_cast<int>(port);
        row.endpoint_sha256 = fields[endpoint_index];
        row.resident_bytes = static_cast<size_t>(resident_bytes);
        row.resident_geometry_sha256 = fields[geometry_index];
        row.operator_plan_sha256 = fields[plan_index];
        row.session_generation = session_generation;
        covered_layers_by_artifact[artifact] |= row.layer_mask;
        rows.push_back(std::move(row));
    }
    if (rows.empty()) {
        error = "resident session manifest is empty";
        return false;
    }
    std::sort(rows.begin(), rows.end(), [](const auto & left, const auto & right) {
        return left.session_id < right.session_id;
    });
    return true;
}

} // namespace ffn_split
