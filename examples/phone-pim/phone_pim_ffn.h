#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace phone_pim {

struct FfnSpec {
    std::string model_path;
    int model_fd = -1;
    bool model_fd_verified = false;
    std::string tensor_prefix;
    std::string backend_name;
    uint32_t token_count = 0;
    uint64_t expected_model_bytes = 0;
    std::array<uint8_t, 32> expected_model_sha256 = {};
};

struct FfnPrepareMetrics {
    uint64_t verify_us = 0;
    uint64_t load_us = 0;
    uint64_t upload_us = 0;
    uint64_t warmup_us = 0;
    uint64_t weight_bytes = 0;
    uint64_t resident_buffer_bytes = 0;
};

struct FfnExecuteMetrics {
    uint64_t input_set_us = 0;
    uint64_t compute_us = 0;
    uint64_t output_get_us = 0;
};

class FfnIsland {
public:
    FfnIsland();
    ~FfnIsland();

    FfnIsland(const FfnIsland &) = delete;
    FfnIsland & operator=(const FfnIsland &) = delete;

    bool prepare(const FfnSpec & spec, FfnPrepareMetrics & metrics, std::string & error);
    bool execute(
            const std::vector<float> & input,
            std::vector<float> & output,
            FfnExecuteMetrics & metrics,
            std::string & error);
    void reset();

    bool ready() const;
    uint64_t input_elements() const;
    uint64_t output_elements() const;
    uint64_t n_embd() const;
    uint64_t n_ff() const;
    uint32_t token_count() const;
    const std::string & backend_name() const;
    const std::string & backend_description() const;
    const std::string & tensor_prefix() const;

    std::vector<float> make_test_input(uint32_t seed) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool all_finite(const float * data, size_t size);
double relative_l2(const std::vector<float> & actual, const std::vector<float> & expected, bool & finite);

} // namespace phone_pim
