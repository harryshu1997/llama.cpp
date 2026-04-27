#pragma once

#include "ggml.h"

#include <CL/cl_platform.h>
#include <clblast.h>

#ifdef GGML_OPENCL_USE_CLBLAST
// Diagnostic counters - controlled by GGML_OPENCL_CLBLAST_DEBUG=1 env var.
// Counts hits / falls-through across the whole process; dumped on backend free.
#include <atomic>
struct ggml_clblast_stats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> fallthrough{0};
    std::atomic<uint64_t> errors{0};
};
inline ggml_clblast_stats & ggml_clblast_get_stats() {
    static ggml_clblast_stats s;
    return s;
}
inline bool ggml_clblast_debug_enabled() {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("GGML_OPENCL_CLBLAST_DEBUG");
        v = (e && atoi(e) != 0) ? 1 : 0;
    }
    return v != 0;
}
// CLBlast dispatch is OPT-IN: GGML_OPENCL_CLBLAST_ENABLE=1 turns on the matmul
// override; otherwise all dispatch sites fall through to the native Adreno
// kernels (which empirically beat CLBlast on Adreno 750 for our workload).
// The integration is kept compiled in for future use on devices where CLBlast
// is competitive (ROCm, Intel, etc.) or after a deeper Adreno-specific tuner
// pass; flipping the env var lets us re-evaluate without rebuilding.
inline bool ggml_clblast_dispatch_enabled() {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("GGML_OPENCL_CLBLAST_ENABLE");
        v = (e && atoi(e) != 0) ? 1 : 0;
    }
    return v != 0;
}

#ifdef GGML_OPENCL_PROFILING
#define CLBLAST_GEMM(kernel, M, N, K, ...) \
    do { \
        cl_event evt; \
        int ret = kernel(&backend_ctx->queue, &evt, M, N, K, __VA_ARGS__); \
        if (ret == 0) { \
            ggml_clblast_get_stats().hits.fetch_add(1, std::memory_order_relaxed); \
            if (ggml_clblast_debug_enabled()) { \
                fprintf(stderr, "[clblast HIT] %s M=%d N=%d K=%d\n", #kernel, (int)(M), (int)(N), (int)(K)); \
            } \
            std::string kernel_name = #kernel; \
            size_t global_size[3] = {0, 0, 0}; \
            size_t local_size[3] = {0, 0, 0}; \
            backend_ctx->profiling_info.emplace_back(); \
            populateProfilingInfo(backend_ctx->profiling_info.back(), evt, kernel_name, 3, global_size, local_size, dst); \
            return; \
        } else { \
            ggml_clblast_get_stats().fallthrough.fetch_add(1, std::memory_order_relaxed); \
            if (ggml_clblast_debug_enabled()) { \
                fprintf(stderr, "[clblast FALL] %s M=%d N=%d K=%d ret=%d\n", #kernel, (int)(M), (int)(N), (int)(K), ret); \
            } \
        } \
    } while(0)
#else
#define CLBLAST_GEMM(kernel, M, N, K, ...) \
    do { \
        cl_event evt = NULL; \
        int ret = kernel(&backend_ctx->queue, &evt, M, N, K, __VA_ARGS__); \
        if (ret == 0) { \
            ggml_clblast_get_stats().hits.fetch_add(1, std::memory_order_relaxed); \
            if (ggml_clblast_debug_enabled()) { \
                fprintf(stderr, "[clblast HIT] %s M=%d N=%d K=%d\n", #kernel, (int)(M), (int)(N), (int)(K)); \
            } \
            return; \
        } else { \
            ggml_clblast_get_stats().fallthrough.fetch_add(1, std::memory_order_relaxed); \
            if (ggml_clblast_debug_enabled()) { \
                fprintf(stderr, "[clblast FALL] %s M=%d N=%d K=%d ret=%d\n", #kernel, (int)(M), (int)(N), (int)(K), ret); \
            } \
        } \
    } while(0)
#endif
#endif

inline bool ggml_is_strictly_contiguous(const struct ggml_tensor * tensor, int n) {
    size_t next_nb = ggml_type_size(tensor->type);
    if (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        return false;
    }
    next_nb *= tensor->ne[0] / ggml_blck_size(tensor->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (i > n) {
            if (tensor->nb[i] != next_nb) {
                return false;
            }
            next_nb *= tensor->ne[i];
        } else {
            // this dimension does not need to be contiguous
            next_nb = tensor->ne[i] * tensor->nb[i];
        }
    }
    return true;
}

template <typename TA, typename TB, typename TC, typename TScalar>
inline int clblast_gemv_wrapper(cl_command_queue * queue,
                                cl_event *         evt,
                                const int          M,
                                const int          N,
                                const int          batch_dim1,      // lower batch for B and C
                                const int          batch_dim2,      // upper batch for B and C
                                const int          repeat_a_dim1,   // repeat for A (lower batch)
                                const int          repeat_a_dim2,   // repeat for A (upper batch)
                                cl_mem             buf_a,           // TA
                                const int          offset_a,        // in bytes
                                const int          ld_a,            // in elements
                                const int          batch_stride_a,  // in elements
                                cl_mem             buf_b,           // TB
                                const int          offset_b,        // in bytes
                                const int          ld_b,            // in elements
                                const int          batch_stride_b,  // in elements
                                cl_mem             buf_c,           // TC
                                const int          offset_c,        // in bytes
                                const int          ld_c,            // in elements
                                const int          batch_stride_c) {         // in elements
    // Check conditions
    if (!(offset_a % sizeof(TA) == 0 && offset_b % sizeof(TB) == 0 && offset_c % sizeof(TC) == 0)) {
        return -1;
    }

    clblast::StatusCode status;
    if (batch_dim1 == 1 && batch_dim2 == 1 && repeat_a_dim1 == 1 && repeat_a_dim2 == 1) {
        status = clblast::Gemv<TA, TB, TC, TScalar>(clblast::Layout::kRowMajor, clblast::Transpose::kNo, M, N, 1, buf_a,
                                                    offset_a / sizeof(TA), ld_a, buf_b, offset_b / sizeof(TB), ld_b, 0,
                                                    buf_c, offset_c / sizeof(TC), ld_c, queue, evt);
    } else {
        return -1;
    }
    if (status != clblast::StatusCode::kSuccess) {
        // Print all parameters
        printf("Error in clblast_gemv_wrapper: clblast::Gemv failed with status %d\n", status);
        printf("Error in clblast_gemv_wrapper: M = %d, N = %d\n", M, N);
        printf("Error in clblast_gemv_wrapper: batch_dim1 = %d, batch_dim2 = %d\n", batch_dim1, batch_dim2);
        printf("Error in clblast_gemv_wrapper: repeat_a_dim1 = %d, repeat_a_dim2 = %d\n", repeat_a_dim1, repeat_a_dim2);
        printf("Error in clblast_gemv_wrapper: offset_a = %d, offset_b = %d, offset_c = %d\n", offset_a, offset_b,
               offset_c);
        printf("Error in clblast_gemm_wrapper: ld_a = %d, ld_b = %d, ld_c = %d\n", ld_a, ld_b, ld_c);
        printf("Error in clblast_gemm_wrapper: batch_stride_a = %d, batch_stride_b = %d, batch_stride_c = %d\n",
               batch_stride_a, batch_stride_b, batch_stride_c);
        fflush(stdout);
        return -1;
    }
    // GGML_ASSERT(status == clblast::StatusCode::kSuccess);
    return 0;
}

template <typename TA, typename TB, typename TC, typename TScalar>
inline int clblast_gemm_wrapper(cl_command_queue * queue,
                                cl_event *         evt,
                                const int          M,
                                const int          N,
                                const int          K,
                                const int          batch_dim1,      // lower batch for B and C
                                const int          batch_dim2,      // upper batch for B and C
                                const int          repeat_a_dim1,   // repeat for A (lower batch)
                                const int          repeat_a_dim2,   // repeat for A (upper batch)
                                cl_mem             buf_a,           // TA
                                const int          offset_a,        // in bytes
                                const int          ld_a,            // in elements
                                const int          batch_stride_a,  // in elements
                                cl_mem             buf_b,           // TB
                                const int          offset_b,        // in bytes
                                const int          ld_b,            // in elements
                                const int          batch_stride_b,  // in elements
                                cl_mem             buf_c,           // TC
                                const int          offset_c,        // in bytes
                                const int          ld_c,            // in elements
                                const int          batch_stride_c) {         // in elements
    // Check conditions
    if (!(offset_a % sizeof(TA) == 0 && offset_b % sizeof(TB) == 0 && offset_c % sizeof(TC) == 0)) {
        return -1;
    }

    if (N == 1) {
        // const int b_inc = 1;
        // const int c_inc = 1;
        // int ret = clblast_gemv_wrapper<TA, TB, TC, TScalar>(
        //     queue, evt, M, K, batch_dim1, batch_dim2, repeat_a_dim1, repeat_a_dim2, buf_a, offset_a, ld_a,
        //     batch_stride_a, buf_b, offset_b, b_inc, batch_stride_b, buf_c, offset_c, c_inc, batch_stride_c);
        // return ret;
        return -1;
    }

    clblast::StatusCode status;
    if (batch_dim1 == 1 && batch_dim2 == 1 && repeat_a_dim1 == 1 && repeat_a_dim2 == 1) {
        status = clblast::Gemm<TA, TB, TC, TScalar>(clblast::Layout::kColMajor, clblast::Transpose::kYes,
                                                    clblast::Transpose::kNo, M, N, K,
                                                    1,                                   // alpha
                                                    buf_a, offset_a / sizeof(TA), ld_a,  // A is [K, M]
                                                    buf_b, offset_b / sizeof(TB), ld_b,  // B is [K, N]
                                                    0,                                   // beta
                                                    buf_c, offset_c / sizeof(TC), ld_c,  // C is [M, N]
                                                    queue, evt);
    } else if (repeat_a_dim1 == 1 && repeat_a_dim2 == 1) {
        status = clblast::GemmStridedBatched<TA, TB, TC, TScalar>(
            clblast::Layout::kColMajor, clblast::Transpose::kYes, clblast::Transpose::kNo, M, N, K,
            1,                                                   // alpha
            buf_a, offset_a / sizeof(TA), ld_a, batch_stride_a,  // A is [K, M]
            buf_b, offset_b / sizeof(TB), ld_b, batch_stride_b,  // B is [K, N]
            0,                                                   // beta
            buf_c, offset_c / sizeof(TC), ld_c, batch_stride_c,  // C is [M, N]
            batch_dim1 * batch_dim2, queue, evt);
    } else {
        int                  batch_a_dim1 = batch_dim1 / repeat_a_dim1;
        size_t               batch_count  = batch_dim1 * batch_dim2;
        std::vector<size_t>  a_offsets(batch_count);
        std::vector<size_t>  b_offsets(batch_count);
        std::vector<size_t>  c_offsets(batch_count);
        std::vector<TScalar> alphas(batch_count, 1.0f);
        std::vector<TScalar> betas(batch_count, 0.0f);

        for (int i2 = 0; i2 < batch_dim2; ++i2) {
            for (int i1 = 0; i1 < batch_dim1; ++i1) {
                int ai1 = i1 / repeat_a_dim1;
                int ai2 = i2 / repeat_a_dim2;

                size_t idx = i2 * batch_dim1 + i1;

                a_offsets[idx] = offset_a / sizeof(TA) + ai1 * batch_stride_a + ai2 * (batch_stride_a * batch_a_dim1);
                b_offsets[idx] = offset_b / sizeof(TB) + i1 * batch_stride_b + i2 * (batch_stride_b * batch_dim1);
                c_offsets[idx] = offset_c / sizeof(TC) + i1 * batch_stride_c + i2 * (batch_stride_c * batch_dim1);
            }
        }

        status = clblast::GemmBatched<TA, TB, TC, TScalar>(clblast::Layout::kColMajor, clblast::Transpose::kYes,
                                                           clblast::Transpose::kNo,        //
                                                           M, N, K, alphas.data(),         //
                                                           buf_a, a_offsets.data(), ld_a,  //
                                                           buf_b, b_offsets.data(), ld_b,  //
                                                           betas.data(),                   //
                                                           buf_c, c_offsets.data(), ld_c,  //
                                                           batch_count, queue, evt);
    }
    if (status != clblast::StatusCode::kSuccess) {
        // Print all parameters
        printf("Error in clblast_gemm_wrapper: clblast::GemmBatched failed with status %d\n", status);
        printf("Error in clblast_gemm_wrapper: M = %d, N = %d, K = %d\n", M, N, K);
        printf("Error in clblast_gemm_wrapper: batch_dim1 = %d, batch_dim2 = %d\n", batch_dim1, batch_dim2);
        printf("Error in clblast_gemm_wrapper: repeat_a_dim1 = %d, repeat_a_dim2 = %d\n", repeat_a_dim1, repeat_a_dim2);
        printf("Error in clblast_gemm_wrapper: offset_a = %d, offset_b = %d, offset_c = %d\n", offset_a, offset_b,
               offset_c);
        printf("Error in clblast_gemm_wrapper: ld_a = %d, ld_b = %d, ld_c = %d\n", ld_a, ld_b, ld_c);
        printf("Error in clblast_gemm_wrapper: batch_stride_a = %d, batch_stride_b = %d, batch_stride_c = %d\n",
               batch_stride_a, batch_stride_b, batch_stride_c);
        fflush(stdout);
        return -1;
    }
    // GGML_ASSERT(status == clblast::StatusCode::kSuccess);
    return 0;
}

inline int clblast_gemm_f32_f32_f32(cl_command_queue * queue,
                                    cl_event *         evt,
                                    const int          M,
                                    const int          N,
                                    const int          K,
                                    const int          batch_dim1,      // lower batch for B and C
                                    const int          batch_dim2,      // upper batch for B and C
                                    const int          repeat_a_dim1,   // repeat for A (lower batch)
                                    const int          repeat_a_dim2,   // repeat for A (upper batch)
                                    cl_mem             buf_a,           // float32
                                    const int          offset_a,        // in bytes
                                    const int          ld_a,            // in elements
                                    const int          batch_stride_a,  // in elements
                                    cl_mem             buf_b,           // float32
                                    const int          offset_b,        // in bytes
                                    const int          ld_b,            // in elements
                                    const int          batch_stride_b,  // in elements
                                    cl_mem             buf_c,           // float32
                                    const int          offset_c,        // in bytes
                                    const int          ld_c,            // in elements
                                    const int          batch_stride_c) {         // in elements
    return clblast_gemm_wrapper<cl_float, cl_float, cl_float, cl_float>(
        queue, evt, M, N, K, batch_dim1, batch_dim2, repeat_a_dim1, repeat_a_dim2, buf_a, offset_a, ld_a,
        batch_stride_a, buf_b, offset_b, ld_b, batch_stride_b, buf_c, offset_c, ld_c, batch_stride_c);
}

inline int clblast_gemm_f16_f32_f32(cl_command_queue * queue,
                                    cl_event *         evt,
                                    const int          M,
                                    const int          N,
                                    const int          K,
                                    const int          batch_dim1,      // lower batch for B and C
                                    const int          batch_dim2,      // upper batch for B and C
                                    const int          repeat_a_dim1,   // repeat for A (lower batch)
                                    const int          repeat_a_dim2,   // repeat for A (upper batch)
                                    cl_mem             buf_a,           // float16
                                    const int          offset_a,        // in bytes
                                    const int          ld_a,            // in elements
                                    const int          batch_stride_a,  // in elements
                                    cl_mem             buf_b,           // float32
                                    const int          offset_b,        // in bytes
                                    const int          ld_b,            // in elements
                                    const int          batch_stride_b,  // in elements
                                    cl_mem             buf_c,           // float32
                                    const int          offset_c,        // in bytes
                                    const int          ld_c,            // in elements
                                    const int          batch_stride_c) {         // in elements
    return clblast_gemm_wrapper<cl_half, cl_float, cl_float, cl_float>(
        queue, evt, M, N, K, batch_dim1, batch_dim2, repeat_a_dim1, repeat_a_dim2, buf_a, offset_a, ld_a,
        batch_stride_a, buf_b, offset_b, ld_b, batch_stride_b, buf_c, offset_c, ld_c, batch_stride_c);
}

inline int clblast_gemm_f16_f16_f32(cl_command_queue * queue,
                                    cl_event *         evt,
                                    const int          M,
                                    const int          N,
                                    const int          K,
                                    const int          batch_dim1,      // lower batch for B and C
                                    const int          batch_dim2,      // upper batch for B and C
                                    const int          repeat_a_dim1,   // repeat for A (lower batch)
                                    const int          repeat_a_dim2,   // repeat for A (upper batch)
                                    cl_mem             buf_a,           // float16
                                    const int          offset_a,        // in bytes
                                    const int          ld_a,            // in elements
                                    const int          batch_stride_a,  // in elements
                                    cl_mem             buf_b,           // float16
                                    const int          offset_b,        // in bytes
                                    const int          ld_b,            // in elements
                                    const int          batch_stride_b,  // in elements
                                    cl_mem             buf_c,           // float32
                                    const int          offset_c,        // in bytes
                                    const int          ld_c,            // in elements
                                    const int          batch_stride_c) {         // in elements
    return clblast_gemm_wrapper<cl_half, cl_half, cl_float, cl_float>(
        queue, evt, M, N, K, batch_dim1, batch_dim2, repeat_a_dim1, repeat_a_dim2, buf_a, offset_a, ld_a,
        batch_stride_a, buf_b, offset_b, ld_b, batch_stride_b, buf_c, offset_c, ld_c, batch_stride_c);
}