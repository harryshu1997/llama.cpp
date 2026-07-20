#pragma once

#include <cstdint>
#include <string>

namespace phone_pim {

inline std::string json_uint64(uint64_t value) {
    return "\"" + std::to_string(value) + "\"";
}

} // namespace phone_pim
