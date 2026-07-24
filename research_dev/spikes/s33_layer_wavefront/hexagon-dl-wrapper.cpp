#include "ggml-backend.h"

#include <dlfcn.h>

using hexagon_reg_fn = ggml_backend_reg_t (*)(void);

extern "C" int ggml_backend_score(void) {
    return 1;
}

extern "C" ggml_backend_reg_t ggml_backend_init(void) {
    static void * handle = dlopen("libggml-hexagon-core.so", RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        return nullptr;
    }

    auto reg = reinterpret_cast<hexagon_reg_fn>(dlsym(handle, "ggml_backend_hexagon_reg"));
    return reg ? reg() : nullptr;
}
