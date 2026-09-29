#pragma once

#include <cstddef>
#include <memory>

struct ggml_tensor;

namespace llama_fenced_tensor {

bool configured();
bool matches(const char * tensor_name);
bool eligible(const ggml_tensor * tensor);

class stage {
public:
    stage(
            ggml_tensor * tensor,
            const void * source,
            size_t source_offset,
            size_t size);
    ~stage();

    stage(const stage &) = delete;
    stage & operator=(const stage &) = delete;

    void execute();
    size_t size() const;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

} // namespace llama_fenced_tensor
