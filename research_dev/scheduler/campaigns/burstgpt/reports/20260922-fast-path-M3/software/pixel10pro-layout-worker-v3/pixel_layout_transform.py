"""Lossless, private F16 layouts using the existing custom-op and worker paths."""

from build_pixel_cpu_gpu import replace_once


def transform(source):
    source = '#include <arm_neon.h>\n#include <atomic>\n' + source
    source = replace_once(source, "host_tensor pixel_join(", '#include "pixel_f16_layout.h"\n\nhost_tensor pixel_join(')
    source = replace_once(source, '    print_residency_phase(cfg, "WEIGHT_UPLOAD_BEGIN");', r'''
    pixel_cpu_layout.rows = pixel_layout_integer("S42_PIXEL_NEON_ROWS", 0);
    pixel_cpu_layout.chunk = pixel_layout_integer("S42_PIXEL_NEON_CHUNK", 0);
    pixel_cpu_layout.unroll = pixel_layout_integer("S42_PIXEL_NEON_UNROLL", 1);
    pixel_cpu_layout.prefetch = pixel_layout_integer("S42_PIXEL_NEON_PREFETCH", 0);
    pixel_cpu_layout.fmlal = pixel_layout_integer("S42_PIXEL_NEON_FMLAL", 0);
    GGML_ASSERT(pixel_cpu_layout.fmlal <= 1);
    pixel_gpu_layout.rows = pixel_layout_integer("S42_PIXEL_GPU_LAYOUT_ROWS", 0);
    pixel_gpu_layout.chunk = pixel_layout_integer("S42_PIXEL_GPU_LAYOUT_CHUNK", 0);
    const bool pixel_gpu_f16 = pixel_layout_integer("S42_PIXEL_GPU_EXPAND_F16", 0) == 1;
    if (pixel_gpu_f16) {
        GGML_ASSERT(pixel_packed && !secondary.max_tokens && !pixel_coalesce);
        GGML_ASSERT(cfg.backend == "Vulkan0" || secondary.backend == "Vulkan0");
        fprintf(stderr, "S42PIXELDEVICEWEIGHTS cpu=original_packed gpu=decoded_f16 requantized=0\n");
    }
    if (pixel_cpu_layout.rows) {
        GGML_ASSERT(cfg.backend == "CPU" && !pixel_packed && !pixel_coalesce && !secondary.max_tokens);
        GGML_ASSERT(pixel_cpu_layout.rows == 2 || pixel_cpu_layout.rows == 4 || pixel_cpu_layout.rows == 8);
        GGML_ASSERT(pixel_cpu_layout.chunk == 0 || pixel_cpu_layout.chunk == 8 || pixel_cpu_layout.chunk == 32);
        GGML_ASSERT(pixel_cpu_layout.unroll == 1 || (pixel_cpu_layout.unroll == 2 && pixel_cpu_layout.rows <= 4));
        fprintf(stderr, "S42PIXELNEON rows=%d chunk=%d unroll=%d prefetch=%d precision=f32\n",
                pixel_cpu_layout.rows, pixel_cpu_layout.chunk, pixel_cpu_layout.unroll, pixel_cpu_layout.prefetch);
        fprintf(stderr, "S42PIXELNEONMATH activation=%s accumulate=f32\n", pixel_cpu_layout.fmlal ? "f16" : "f32");
    } else {
        GGML_ASSERT(pixel_cpu_layout.chunk == 0);
    }
    if (pixel_gpu_layout.rows) {
        GGML_ASSERT((!pixel_packed || pixel_gpu_f16) && !pixel_coalesce && !secondary.max_tokens);
        GGML_ASSERT(pixel_gpu_layout.rows == 8 && (pixel_gpu_layout.chunk == 4 || pixel_gpu_layout.chunk == 16));
        GGML_ASSERT(cfg.backend == "Vulkan0" || secondary.backend == "Vulkan0");
        const std::string shader = "layout8k" + std::to_string(pixel_gpu_layout.chunk);
        GGML_ASSERT(getenv("S42_PIXEL_F16_SHADER") && shader == getenv("S42_PIXEL_F16_SHADER"));
        GGML_ASSERT(pixel_layout_integer("S42_PIXEL_F16_ROWS", 0) == 8);
        GGML_ASSERT(pixel_layout_integer("S42_PIXEL_F16_WG", 0) == 128);
        fprintf(stderr, "S42PIXELGPULAYOUT rows=8 chunk=%d precision=f32\n", pixel_gpu_layout.chunk);
    } else {
        const char * shader = getenv("S42_PIXEL_F16_SHADER");
        GGML_ASSERT(!shader || std::string(shader).find("layout") != 0);
    }
    for (layer_state & state : states) {
        for (layer_state::block & block : state.blocks) {
            const auto & primary = cfg.backend == "CPU" ? pixel_cpu_layout : pixel_gpu_layout;
            for (host_tensor * tensor : {&block.gate, &block.up, &block.down}) {
                if (pixel_gpu_f16 && cfg.backend == "Vulkan0") pixel_expand_gpu_weights(*tensor);
                pixel_pack_f16(*tensor, primary);
            }
            for (host_tensor * tensor : {&block.secondary_gate, &block.secondary_up, &block.secondary_down}) {
                if (pixel_gpu_f16) pixel_expand_gpu_weights(*tensor);
                pixel_pack_f16(*tensor, pixel_gpu_layout);
            }
        }
    }
    fprintf(stderr, "S42PIXELPACKLAYOUT bytes=%llu startup_us=%llu extra_resident_bytes=0\n",
            static_cast<unsigned long long>(pixel_layout_bytes), static_cast<unsigned long long>(pixel_layout_us));
    print_residency_phase(cfg, "WEIGHT_UPLOAD_BEGIN");''')
    start = source.index("    auto build_graph = [&]")
    end = source.index("    for (size_t arena_index = 0;", start)
    graph = source[start:end]
    if graph.count("pixel_residual_matmul(") != 3:
        raise ValueError("primary projection anchors differ")
    source = source[:start] + graph.replace("pixel_residual_matmul(", "pixel_neon_projection(") + source[end:]
    source = replace_once(source, "                free_secondary();\n                ggml_backend_free(backend);\n                return 0;", r'''
                free_secondary();
                ggml_backend_free(backend);
                fprintf(stderr, "S42PIXELNEONCALLS projections=%llu\n",
                        static_cast<unsigned long long>(pixel_neon_calls.load()));
                return 0;''')
    return source
