// clgemv: OpenCL GEMV microbenchmark for the Pixel 10 Pro. usage: clgemv <kernel.cl> <variant> N K repeats [R] [WG]
// variants: f16_rows | q4_global | q4_image | q4_global_interleaved
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#define F(name) static decltype(&name) p_##name;
F(clGetPlatformIDs) F(clGetDeviceIDs) F(clGetDeviceInfo) F(clCreateContext) F(clCreateCommandQueueWithProperties) F(clCreateProgramWithSource)
F(clBuildProgram) F(clGetProgramBuildInfo) F(clCreateKernel) F(clSetKernelArg) F(clCreateBuffer) F(clCreateImage) F(clEnqueueNDRangeKernel)
F(clEnqueueReadBuffer) F(clFinish) F(clGetEventProfilingInfo) F(clReleaseEvent) F(clReleaseMemObject) F(clReleaseKernel) F(clReleaseProgram)
F(clReleaseCommandQueue) F(clReleaseContext) F(clGetKernelWorkGroupInfo)
static void * lib;
template <class T> static void load(T & p, const char * n) { p = (T) dlsym(lib, n); if (!p) { std::fprintf(stderr, "FAIL dlsym %s\n", n); std::exit(2); } }
static void check(cl_int e, const char * what) { if (e != CL_SUCCESS) { std::fprintf(stderr, "FAIL %s: %d\n", what, e); std::exit(3); } }
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static uint32_t rng_state = 12345u;
static float frand() { rng_state = rng_state * 1664525u + 1013904223u; return ((rng_state >> 8) & 0xFFFF) / 65535.f * 2.f - 1.f; }
static uint16_t f2h(float f) { uint32_t x; std::memcpy(&x, &f, 4); uint32_t sign = (x >> 16) & 0x8000; int32_t exp = int32_t((x >> 23) & 0xFF) - 127 + 15; uint32_t man = x & 0x7FFFFF; if (exp <= 0) return uint16_t(sign); if (exp >= 31) return uint16_t(sign | 0x7C00); return uint16_t(sign | (uint32_t(exp) << 10) | (man >> 13)); }
static float h2f(uint16_t h) { uint32_t sign = (h & 0x8000u) << 16; uint32_t exp = (h >> 10) & 0x1F; uint32_t man = h & 0x3FF; uint32_t x; if (exp == 0) x = sign; else x = sign | ((exp + 112) << 23) | (man << 13); float f; std::memcpy(&f, &x, 4); return f; }

int main(int argc, char ** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: clgemv kernel.cl variant N K repeats [R] [WG]\n"); return 1; }
    std::string clpath = argv[1], variant = argv[2]; int N = std::atoi(argv[3]), K = std::atoi(argv[4]), repeats = std::atoi(argv[5]);
    int R = argc > 6 ? std::atoi(argv[6]) : 8, WG = argc > 7 ? std::atoi(argv[7]) : 128, M = argc > 8 ? std::atoi(argv[8]) : 1;
    if (K % 32 || N % R || WG % R) { std::fprintf(stderr, "FAIL geometry\n"); return 1; }
    lib = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW); if (!lib) lib = dlopen("libOpenCL.so", RTLD_NOW); if (!lib) { std::fprintf(stderr, "FAIL dlopen\n"); return 2; }
#define L(n) load(p_##n, #n);
    L(clGetPlatformIDs) L(clGetDeviceIDs) L(clGetDeviceInfo) L(clCreateContext) L(clCreateCommandQueueWithProperties) L(clCreateProgramWithSource) L(clBuildProgram) L(clGetProgramBuildInfo)
    L(clCreateKernel) L(clSetKernelArg) L(clCreateBuffer) L(clCreateImage) L(clEnqueueNDRangeKernel) L(clEnqueueReadBuffer) L(clFinish) L(clGetEventProfilingInfo) L(clReleaseEvent) L(clReleaseMemObject)
    L(clReleaseKernel) L(clReleaseProgram) L(clReleaseCommandQueue) L(clReleaseContext) L(clGetKernelWorkGroupInfo)
    cl_platform_id plat; cl_device_id dev; cl_int err;
    check(p_clGetPlatformIDs(1, &plat, nullptr), "platform"); check(p_clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr), "device");
    cl_context ctx = p_clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); check(err, "context");
    const char * prio = std::getenv("CLG_PRIO");
    cl_queue_properties qp_plain[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_queue_properties qp_prio[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0x1096 /*CL_QUEUE_PRIORITY_KHR*/, cl_queue_properties(prio && std::string(prio) == "high" ? (1 << 0) : (1 << 2)), 0};
    cl_queue_properties * qp = prio ? qp_prio : qp_plain;
    cl_command_queue q = p_clCreateCommandQueueWithProperties(ctx, dev, qp, &err); check(err, "queue");
    // ---- data: exact f32 weights, Q4_0 quantized copy, f32 activations and their Q8 blocks
    const int nb = K / 32, ng = N / R;
    std::vector<float> x(K), W(size_t(N) * K);
    for (auto & v : x) v = frand();
    for (auto & v : W) v = frand();
    std::vector<uint8_t> tiles(size_t(ng) * nb * R * 16);  // tile layout nibbles
    std::vector<uint16_t> scales(size_t(ng) * nb * R);
    std::vector<uint16_t> wh(size_t(N) * K);                 // f16 row-major
    std::vector<float> Wq(size_t(N) * K);                    // dequantized reference
    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < nb; ++b) {
            float amax = 0.f, maxv = 0.f;
            for (int i = 0; i < 32; ++i) { float v = W[size_t(n) * K + b * 32 + i]; if (std::fabs(v) > amax) { amax = std::fabs(v); maxv = v; } }
            float d = maxv / -8.f, id = d ? 1.f / d : 0.f; uint16_t dh = f2h(d); float dq = h2f(dh);
            int g = n / R, r = n % R; size_t t = (size_t(g) * nb + b) * R + r; scales[t] = dh;
            for (int i = 0; i < 16; ++i) {
                float v0 = W[size_t(n) * K + b * 32 + i] * id, v1 = W[size_t(n) * K + b * 32 + 16 + i] * id;
                uint8_t q0 = uint8_t(std::min(15.f, std::max(0.f, v0 + 8.5f))), q1 = uint8_t(std::min(15.f, std::max(0.f, v1 + 8.5f)));
                tiles[t * 16 + i] = uint8_t(q0 | (q1 << 4));
                Wq[size_t(n) * K + b * 32 + i] = (int(q0) - 8) * dq; Wq[size_t(n) * K + b * 32 + 16 + i] = (int(q1) - 8) * dq;
            }
        }
        for (int k = 0; k < K; ++k) wh[size_t(n) * K + k] = f2h(W[size_t(n) * K + k]);
    }
    std::vector<int8_t> xq(K); std::vector<float> xs(nb), xdq(K);
    for (int b = 0; b < nb; ++b) { float amax = 0.f; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i])); float s = amax / 127.f, is = s ? 1.f / s : 0.f; xs[b] = s; for (int i = 0; i < 32; ++i) { int v = int(std::lround(x[b * 32 + i] * is)); xq[b * 32 + i] = int8_t(v); xdq[b * 32 + i] = v * s; } }
    std::vector<int8_t> xqm(size_t(M) * K); std::vector<float> xsm(size_t(M) * nb);
    for (int m = 0; m < M; ++m) { for (int k = 0; k < K; ++k) xqm[size_t(m) * K + k] = int8_t(std::max(-127, std::min(127, int(xq[k]) + (m ? (k % 7) - 3 : 0)))); for (int b = 0; b < nb; ++b) xsm[size_t(m) * nb + b] = xs[b]; }
    std::vector<uint8_t> inter;  // interleaved layout: per (g,b): 16*R nibble bytes then R halfs
    if (variant == "q4_global_interleaved") { inter.resize(size_t(ng) * nb * (18 * R)); for (size_t gb = 0; gb < size_t(ng) * nb; ++gb) { std::memcpy(&inter[gb * 18 * R], &tiles[gb * R * 16], 16 * R); std::memcpy(&inter[gb * 18 * R + 16 * R], &scales[gb * R], 2 * R); } }
    // ---- references
    std::vector<float> y_exact(N), y_q(N);
    for (int n = 0; n < N; ++n) { double a = 0, b2 = 0; for (int k = 0; k < K; ++k) { a += double(W[size_t(n) * K + k]) * x[k]; b2 += double(Wq[size_t(n) * K + k]) * xdq[k]; } y_exact[n] = float(a); y_q[n] = float(b2); }
    // ---- program
    std::ifstream f(clpath); std::stringstream ss; ss << f.rdbuf(); std::string src = ss.str(); const char * srcp = src.c_str();
    cl_program prog = p_clCreateProgramWithSource(ctx, 1, &srcp, nullptr, &err); check(err, "program");
    const char * extra = std::getenv("CLG_OPTS");
    char opts[512]; std::snprintf(opts, sizeof opts, "-cl-std=CL3.0 -cl-mad-enable -D R=%d -D WG=%d -D IMGW=8192 -D M=%d %s", R, WG, M, extra ? extra : "");
    err = p_clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr);
    { size_t n = 0; p_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n); std::string log(n, 0); p_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr); if (err || log.find("error") != std::string::npos) { std::fprintf(stderr, "BUILD %d\n%s\n", err, log.c_str()); if (err) return 4; } }
    if (variant == "cache_sweep") {
        cl_mem src = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tiles.size(), tiles.data(), &err); check(err, "src");
        cl_kernel k16 = p_clCreateKernel(prog, "stream_read16", &err); check(err, "k16");
        cl_mem out = p_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, size_t(1) << 20, nullptr, &err); check(err, "out");
        for (size_t mb : {size_t(1) / 2, size_t(1), size_t(2), size_t(3), size_t(4), size_t(6), size_t(8), size_t(12), size_t(16), size_t(24), size_t(32), size_t(44)}) {
            size_t bytes = mb ? mb << 20 : size_t(512) << 10; if (bytes > tiles.size()) break;
            int count4 = int(bytes / 16); size_t gl = size_t(WG) * (mb >= 4 ? 64 : 16), lc = WG;
            check(p_clSetKernelArg(k16, 0, sizeof src, &src), "a0"); check(p_clSetKernelArg(k16, 1, sizeof out, &out), "a1"); check(p_clSetKernelArg(k16, 2, sizeof count4, &count4), "a2");
            for (int i = 0; i < 5; ++i) check(p_clEnqueueNDRangeKernel(q, k16, 1, nullptr, &gl, &lc, 0, nullptr, nullptr), "e"); check(p_clFinish(q), "f");
            std::vector<cl_event> evs(60);
            for (auto & e : evs) check(p_clEnqueueNDRangeKernel(q, k16, 1, nullptr, &gl, &lc, 0, nullptr, &e), "e");
            check(p_clFinish(q), "f");
            std::vector<double> ms; for (auto & e : evs) { cl_ulong a = 0, b = 0; p_clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof a, &a, nullptr); p_clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof b, &b, nullptr); ms.push_back((b - a) * 1e-6); p_clReleaseEvent(e); }
            std::sort(ms.begin(), ms.end()); double med = ms[ms.size() / 2], mn = ms.front();
            std::printf("{\"cache_sweep_MB\":%.1f,\"device_ms_med\":%.4f,\"device_ms_min\":%.4f,\"GBps_med\":%.1f,\"GBps_min\":%.1f}\n", bytes / 1048576.0, med, mn, bytes / (med * 1e-3) / 1e9, bytes / (mn * 1e-3) / 1e9);
        }
        return 0;
    }
    if (variant == "copy_engine") {
        auto p_clEnqueueCopyBuffer = (decltype(&clEnqueueCopyBuffer)) dlsym(lib, "clEnqueueCopyBuffer");
        cl_mem src = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tiles.size(), tiles.data(), &err); check(err, "src");
        cl_mem dst = p_clCreateBuffer(ctx, CL_MEM_READ_WRITE, tiles.size(), nullptr, &err); check(err, "dst");
        for (int i = 0; i < 3; ++i) { check(p_clEnqueueCopyBuffer(q, src, dst, 0, 0, tiles.size(), 0, nullptr, nullptr), "copy"); } check(p_clFinish(q), "finish");
        std::vector<cl_event> evs(repeats); double t0 = now();
        for (int i = 0; i < repeats; ++i) check(p_clEnqueueCopyBuffer(q, src, dst, 0, 0, tiles.size(), 0, nullptr, &evs[i]), "copy");
        check(p_clFinish(q), "finish"); double wall = (now() - t0) / repeats * 1e3;
        std::vector<double> ms; for (auto & e : evs) { cl_ulong a = 0, b = 0; p_clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof a, &a, nullptr); p_clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof b, &b, nullptr); ms.push_back((b - a) * 1e-6); p_clReleaseEvent(e); }
        std::sort(ms.begin(), ms.end()); double med = ms[ms.size() / 2];
        std::printf("{\"variant\":\"copy_engine\",\"bytes_MB\":%.1f,\"device_ms_med\":%.3f,\"read_GBps\":%.2f,\"read_plus_write_GBps\":%.2f,\"wall_ms_batched\":%.3f}\n", tiles.size() / 1e6, med, tiles.size() / (med * 1e-3) / 1e9, 2 * tiles.size() / (med * 1e-3) / 1e9, wall);
        return 0;
    }
    cl_kernel kern = p_clCreateKernel(prog, variant.c_str(), &err); check(err, "kernel");
    size_t pref = 0; p_clGetKernelWorkGroupInfo(kern, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof pref, &pref, nullptr);
    // ---- buffers
    const char * alloc = std::getenv("CLG_ALLOC"); std::string alloc_mode = alloc ? alloc : "copy";
    std::vector<void *> pinned;
    auto mk = [&](const void * p, size_t bytes, cl_mem_flags fl) {
        cl_mem m;
        if (alloc_mode.rfind("img_", 0) == 0) {   // Imagination allocation flags via clCreateBufferWithProperties / cl_mem_flags bits
            static auto p_clCreateBufferWithProperties = (decltype(&clCreateBufferWithProperties)) dlsym(lib, "clCreateBufferWithProperties");
            cl_mem_flags extra = 0; cl_ulong img = 0;
            if (alloc_mode == "img_uncached_cpu") extra = (cl_mem_flags) 1 << 26; else if (alloc_mode == "img_cached_cpu") extra = (cl_mem_flags) 1 << 27;
            else if (alloc_mode == "img_gpu_wc") img = 1 << 1; else if (alloc_mode == "img_gpu_cached") img = 1 << 2; else if (alloc_mode == "img_cpu_local") img = 1 << 3; else if (alloc_mode == "img_gpu_local") img = 1 << 4; else if (alloc_mode == "img_relax") img = 1 << 0;
            if (img) { cl_mem_properties props[] = {0x40D7 /*CL_MEM_ALLOC_FLAGS_IMG*/, (cl_mem_properties) img, 0}; m = p_clCreateBufferWithProperties(ctx, props, fl | CL_MEM_COPY_HOST_PTR, bytes, const_cast<void *>(p), &err); }
            else m = p_clCreateBuffer(ctx, fl | extra | CL_MEM_COPY_HOST_PTR, bytes, const_cast<void *>(p), &err);
            if (err) { std::fprintf(stderr, "alloc %s failed: %d\n", alloc_mode.c_str(), err); std::exit(5); }
        } else if (alloc_mode == "use_host") { void * h = nullptr; if (posix_memalign(&h, 4096, bytes)) std::abort(); std::memcpy(h, p, bytes); pinned.push_back(h); m = p_clCreateBuffer(ctx, fl | CL_MEM_USE_HOST_PTR, bytes, h, &err); }
        else if (alloc_mode == "alloc_host") { m = p_clCreateBuffer(ctx, fl | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR, bytes, const_cast<void *>(p), &err); }
        else m = p_clCreateBuffer(ctx, fl | CL_MEM_COPY_HOST_PTR, bytes, const_cast<void *>(p), &err);
        check(err, "buffer"); return m; };
    cl_mem b_y = p_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, size_t(M) * N * 4, nullptr, &err); check(err, "ybuf");
    cl_mem b_w = nullptr, b_sc = nullptr, b_x = nullptr, b_xq = nullptr, b_xs = nullptr; size_t weight_bytes = 0;
    if (variant == "f16_rows") {
        b_w = mk(wh.data(), wh.size() * 2, CL_MEM_READ_ONLY); b_x = mk(x.data(), x.size() * 4, CL_MEM_READ_ONLY); weight_bytes = wh.size() * 2;
        check(p_clSetKernelArg(kern, 0, sizeof b_w, &b_w), "a0"); check(p_clSetKernelArg(kern, 1, sizeof b_x, &b_x), "a1"); check(p_clSetKernelArg(kern, 2, sizeof b_y, &b_y), "a2"); check(p_clSetKernelArg(kern, 3, sizeof K, &K), "a3");
    } else {
        b_xq = mk(xqm.data(), xqm.size(), CL_MEM_READ_ONLY); b_xs = mk(xsm.data(), xsm.size() * 4, CL_MEM_READ_ONLY);
        if (variant == "q8_tile") {
            const int scb = ((R * 2) + 15) / 16 * 16, stride = R * 32 + scb; std::vector<uint8_t> q8(size_t(ng) * nb * stride, 0); std::vector<float> Wq8(size_t(N) * K);
            for (int n = 0; n < N; ++n) for (int b = 0; b < nb; ++b) { float amax = 0.f; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(W[size_t(n) * K + b * 32 + i])); float d = amax / 127.f, id = d ? 1.f / d : 0.f; uint16_t dh = f2h(d); float dq = h2f(dh);
                size_t gb = size_t(n / R) * nb + b; int rr = n % R; std::memcpy(&q8[gb * stride + 32 * R + 2 * rr], &dh, 2);
                for (int i = 0; i < 32; ++i) { int v = int(std::lround(W[size_t(n) * K + b * 32 + i] * id)); v = std::max(-127, std::min(127, v)); q8[gb * stride + 32 * rr + i] = uint8_t(int8_t(v)); Wq8[size_t(n) * K + b * 32 + i] = v * dq; } }
            for (int n = 0; n < N; ++n) { double a = 0; for (int k = 0; k < K; ++k) a += double(Wq8[size_t(n) * K + k]) * xdq[k]; y_q[n] = float(a); }
            b_w = mk(q8.data(), q8.size(), CL_MEM_READ_ONLY); weight_bytes = q8.size();
            check(p_clSetKernelArg(kern, 0, sizeof b_w, &b_w), "a0"); check(p_clSetKernelArg(kern, 1, sizeof b_xq, &b_xq), "a1"); check(p_clSetKernelArg(kern, 2, sizeof b_xs, &b_xs), "a2"); check(p_clSetKernelArg(kern, 3, sizeof b_y, &b_y), "a3"); check(p_clSetKernelArg(kern, 4, sizeof K, &K), "a4");
        } else if (variant == "q4k_tile" || variant == "q4k_tile2") {
            if (K % 256) { std::fprintf(stderr, "FAIL K%%256\n"); return 1; }
            const int nsb = K / 256, stride = R * 144; std::vector<uint8_t> q4k(size_t(ng) * nsb * stride, 0); std::vector<float> Wk(size_t(N) * K);
            for (int n = 0; n < N; ++n) for (int sb = 0; sb < nsb; ++sb) {
                float scales[8], mins[8];
                for (int j = 0; j < 8; ++j) { float mn = 1e30f, mx = -1e30f; for (int i = 0; i < 32; ++i) { float v = W[size_t(n) * K + sb * 256 + j * 32 + i]; mn = std::min(mn, v); mx = std::max(mx, v); } scales[j] = (mx - mn) / 15.f; mins[j] = -mn; }
                float smax = 0.f, mmax = 0.f; for (int j = 0; j < 8; ++j) { smax = std::max(smax, scales[j]); mmax = std::max(mmax, mins[j]); }
                uint16_t dh = f2h(smax / 63.f), dmh = f2h(mmax / 63.f); float d = h2f(dh), dmin = h2f(dmh);
                int sc6[8], m6[8]; for (int j = 0; j < 8; ++j) { sc6[j] = std::max(0, std::min(63, int(std::lround(scales[j] / d)))); m6[j] = std::max(0, std::min(63, int(std::lround(mins[j] / dmin)))); }
                size_t gsb = size_t(n / R) * nsb + sb; int rr = n % R; uint8_t * meta = &q4k[gsb * stride + 128 * R + 16 * rr]; std::memcpy(meta, &dh, 2); std::memcpy(meta + 2, &dmh, 2);
                for (int j = 0; j < 4; ++j) { meta[4 + j] = uint8_t(sc6[j] | ((sc6[j + 4] >> 4) << 6)); meta[8 + j] = uint8_t(m6[j] | ((m6[j + 4] >> 4) << 6)); meta[12 + j] = uint8_t((sc6[j + 4] & 0xF) | ((m6[j + 4] & 0xF) << 4)); }
                uint8_t * qs = &q4k[gsb * stride + 128 * rr];
                for (int j = 0; j < 8; ++j) { float sj = d * sc6[j], mj = dmin * m6[j]; for (int i = 0; i < 32; ++i) { float v = W[size_t(n) * K + sb * 256 + j * 32 + i]; int q = sj ? int(std::lround((v + mj) / sj)) : 0; q = std::max(0, std::min(15, q)); Wk[size_t(n) * K + sb * 256 + j * 32 + i] = sj * q - mj; int chunk = j / 2; if (j % 2 == 0) qs[chunk * 32 + i] = uint8_t((qs[chunk * 32 + i] & 0xF0) | q); else qs[chunk * 32 + i] = uint8_t((qs[chunk * 32 + i] & 0x0F) | (q << 4)); } }
            }
            if (variant == "q4k_tile2") {   // re-layout nibbles: piece i (16 B) of row r at (i*R + r)*16 within the 128*R block
                std::vector<uint8_t> re(q4k.size()); for (size_t gsb = 0; gsb < size_t(ng) * nsb; ++gsb) { const uint8_t * src = &q4k[gsb * stride]; uint8_t * dst = &re[gsb * stride];
                    for (int rr = 0; rr < R; ++rr) for (int i = 0; i < 8; ++i) std::memcpy(dst + (size_t(i) * R + rr) * 16, src + 128 * rr + 16 * i, 16); std::memcpy(dst + 128 * R, src + 128 * R, 16 * R); }
                q4k.swap(re); }
            for (int n = 0; n < N; ++n) { double a = 0; for (int k = 0; k < K; ++k) a += double(Wk[size_t(n) * K + k]) * xdq[k]; y_q[n] = float(a); }
            std::vector<int32_t> asum(nb); for (int b = 0; b < nb; ++b) { int t = 0; for (int i = 0; i < 32; ++i) t += xq[b * 32 + i]; asum[b] = t; }
            b_w = mk(q4k.data(), q4k.size(), CL_MEM_READ_ONLY); b_sc = mk(asum.data(), asum.size() * 4, CL_MEM_READ_ONLY); weight_bytes = q4k.size();
            check(p_clSetKernelArg(kern, 0, sizeof b_w, &b_w), "a0"); check(p_clSetKernelArg(kern, 1, sizeof b_xq, &b_xq), "a1"); check(p_clSetKernelArg(kern, 2, sizeof b_xs, &b_xs), "a2"); check(p_clSetKernelArg(kern, 3, sizeof b_sc, &b_sc), "a3"); check(p_clSetKernelArg(kern, 4, sizeof b_y, &b_y), "a4"); check(p_clSetKernelArg(kern, 5, sizeof K, &K), "a5");
        } else if (variant == "q4_uu") {
            const int stride = R * 20; std::vector<uint8_t> uu(size_t(ng) * nb * stride, 0);
            for (size_t gb = 0; gb < size_t(ng) * nb; ++gb) {
                std::memcpy(&uu[gb * stride], &tiles[gb * R * 16], 16 * R); std::memcpy(&uu[gb * stride + 16 * R], &scales[gb * R], 2 * R);
                for (int rr = 0; rr < R; ++rr) { uint16_t ws = 0; for (int i = 0; i < 16; ++i) { uint8_t byte = tiles[(gb * R + rr) * 16 + i]; ws += (byte & 15) + (byte >> 4); } std::memcpy(&uu[gb * stride + 18 * R + 2 * rr], &ws, 2); }
            }
            std::vector<uint8_t> xqu(K); for (int k = 0; k < K; ++k) xqu[k] = uint8_t(int(xq[k]) + 128);
            std::vector<int32_t> asum(nb); for (int b = 0; b < nb; ++b) { int t = 0; for (int i = 0; i < 32; ++i) t += xq[b * 32 + i]; asum[b] = t; }
            b_w = mk(uu.data(), uu.size(), CL_MEM_READ_ONLY); b_xq = mk(xqu.data(), xqu.size(), CL_MEM_READ_ONLY); b_sc = mk(asum.data(), asum.size() * 4, CL_MEM_READ_ONLY); weight_bytes = uu.size();
            check(p_clSetKernelArg(kern, 0, sizeof b_w, &b_w), "a0"); check(p_clSetKernelArg(kern, 1, sizeof b_xq, &b_xq), "a1"); check(p_clSetKernelArg(kern, 2, sizeof b_xs, &b_xs), "a2"); check(p_clSetKernelArg(kern, 3, sizeof b_sc, &b_sc), "a3"); check(p_clSetKernelArg(kern, 4, sizeof b_y, &b_y), "a4"); check(p_clSetKernelArg(kern, 5, sizeof K, &K), "a5");
        } else if (variant == "q4_v2") {
            const int scb = ((R * 2) + 15) / 16 * 16, stride = R * 16 + scb; std::vector<uint8_t> v2(size_t(ng) * nb * stride, 0);
            for (size_t gb = 0; gb < size_t(ng) * nb; ++gb) { std::memcpy(&v2[gb * stride], &tiles[gb * R * 16], 16 * R); std::memcpy(&v2[gb * stride + 16 * R], &scales[gb * R], 2 * R); }
            std::vector<int32_t> xsum(nb); for (int b = 0; b < nb; ++b) { int t = 0; for (int i = 0; i < 32; ++i) t += xq[b * 32 + i]; xsum[b] = t; }
            b_w = mk(v2.data(), v2.size(), CL_MEM_READ_ONLY); b_sc = mk(xsum.data(), xsum.size() * 4, CL_MEM_READ_ONLY); weight_bytes = v2.size();
            check(p_clSetKernelArg(kern, 0, sizeof b_w, &b_w), "a0"); check(p_clSetKernelArg(kern, 1, sizeof b_xq, &b_xq), "a1"); check(p_clSetKernelArg(kern, 2, sizeof b_xs, &b_xs), "a2"); check(p_clSetKernelArg(kern, 3, sizeof b_sc, &b_sc), "a3"); check(p_clSetKernelArg(kern, 4, sizeof b_y, &b_y), "a4"); check(p_clSetKernelArg(kern, 5, sizeof K, &K), "a5");
        } else if (variant == "stream_read" || variant == "stream_async" || variant == "stream_read16") {
            b_w = mk(tiles.data(), tiles.size(), CL_MEM_READ_ONLY); weight_bytes = tiles.size(); int count = int(tiles.size() / 16);
            check(p_clSetKernelArg(kern, 0, sizeof b_w, &b_w), "a0"); check(p_clSetKernelArg(kern, 1, sizeof b_y, &b_y), "a1"); check(p_clSetKernelArg(kern, 2, sizeof count, &count), "a2");
        } else if (variant == "q4_global_m") {
            b_w = mk(tiles.data(), tiles.size(), CL_MEM_READ_ONLY); b_sc = mk(scales.data(), scales.size() * 2, CL_MEM_READ_ONLY); weight_bytes = tiles.size() + scales.size() * 2;
            check(p_clSetKernelArg(kern, 0, sizeof b_w, &b_w), "a0"); check(p_clSetKernelArg(kern, 1, sizeof b_sc, &b_sc), "a1"); check(p_clSetKernelArg(kern, 2, sizeof b_xq, &b_xq), "a2"); check(p_clSetKernelArg(kern, 3, sizeof b_xs, &b_xs), "a3"); check(p_clSetKernelArg(kern, 4, sizeof b_y, &b_y), "a4"); check(p_clSetKernelArg(kern, 5, sizeof K, &K), "a5"); check(p_clSetKernelArg(kern, 6, sizeof N, &N), "a6");
        } else if (variant == "q4_global_interleaved") {
            b_w = mk(inter.data(), inter.size(), CL_MEM_READ_ONLY); weight_bytes = inter.size();
            check(p_clSetKernelArg(kern, 0, sizeof b_w, &b_w), "a0"); check(p_clSetKernelArg(kern, 1, sizeof b_xq, &b_xq), "a1"); check(p_clSetKernelArg(kern, 2, sizeof b_xs, &b_xs), "a2"); check(p_clSetKernelArg(kern, 3, sizeof b_y, &b_y), "a3"); check(p_clSetKernelArg(kern, 4, sizeof K, &K), "a4");
        } else {
            weight_bytes = tiles.size() + scales.size() * 2;
            if (variant == "q4_image") {
                size_t texels = tiles.size() / 16, width = 8192, height = (texels + width - 1) / width;
                std::vector<uint8_t> padded(width * height * 16, 0); std::memcpy(padded.data(), tiles.data(), tiles.size());
                cl_image_format fmt{CL_RGBA, CL_UNSIGNED_INT32}; cl_image_desc desc{}; desc.image_type = CL_MEM_OBJECT_IMAGE2D; desc.image_width = width; desc.image_height = height;
                b_w = p_clCreateImage(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &fmt, &desc, padded.data(), &err); check(err, "image");
            } else b_w = mk(tiles.data(), tiles.size(), CL_MEM_READ_ONLY);
            b_sc = mk(scales.data(), scales.size() * 2, CL_MEM_READ_ONLY);
            check(p_clSetKernelArg(kern, 0, sizeof b_w, &b_w), "a0"); check(p_clSetKernelArg(kern, 1, sizeof b_sc, &b_sc), "a1"); check(p_clSetKernelArg(kern, 2, sizeof b_xq, &b_xq), "a2"); check(p_clSetKernelArg(kern, 3, sizeof b_xs, &b_xs), "a3"); check(p_clSetKernelArg(kern, 4, sizeof b_y, &b_y), "a4"); check(p_clSetKernelArg(kern, 5, sizeof K, &K), "a5");
        }
    }
    size_t global = size_t(ng) * WG, local = WG;
    // ---- warm-up + correctness
    cl_event ev; check(p_clEnqueueNDRangeKernel(q, kern, 1, nullptr, &global, &local, 0, nullptr, &ev), "enqueue"); check(p_clFinish(q), "finish"); p_clReleaseEvent(ev);
    std::vector<float> y(N); check(p_clEnqueueReadBuffer(q, b_y, CL_TRUE, 0, size_t(N) * 4, y.data(), 0, nullptr, nullptr), "read");
    double e_exact = 0, e_q = 0, nrm = 0; for (int n = 0; n < N; ++n) { e_exact += double(y[n] - y_exact[n]) * (y[n] - y_exact[n]); e_q += double(y[n] - y_q[n]) * (y[n] - y_q[n]); nrm += double(y_exact[n]) * y_exact[n]; }
    // ---- optional: CLG_NBUF=n cycles the weight argument over n device copies (tests allocation-size / TLB effects of a many-buffer resident set)
    const int nbuf = std::getenv("CLG_NBUF") ? std::max(1, std::atoi(std::getenv("CLG_NBUF"))) : 1;
    std::vector<cl_mem> wbufs{b_w};
    auto p_copy_buffer = (decltype(&clEnqueueCopyBuffer)) dlsym(lib, "clEnqueueCopyBuffer");
    for (int i = 1; i < nbuf; ++i) { cl_mem c = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY, weight_bytes, nullptr, &err); check(err, "nbuf alloc");
        check(p_copy_buffer(q, b_w, c, 0, 0, weight_bytes, 0, nullptr, nullptr), "nbuf copy"); wbufs.push_back(c); }
    check(p_clFinish(q), "nbuf finish");
    // ---- timing: enqueue back-to-back with events, then wall per single dispatch
    std::vector<cl_event> evs(repeats); double t0 = now();
    for (int i = 0; i < repeats; ++i) { if (nbuf > 1) check(p_clSetKernelArg(kern, 0, sizeof(cl_mem), &wbufs[i % nbuf]), "a0n"); check(p_clEnqueueNDRangeKernel(q, kern, 1, nullptr, &global, &local, 0, nullptr, &evs[i]), "enqueue"); }
    check(p_clFinish(q), "finish"); double wall_batch = (now() - t0) / repeats;
    std::vector<double> dev_ms;
    for (auto & e : evs) { cl_ulong s = 0, en = 0; p_clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof s, &s, nullptr); p_clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof en, &en, nullptr); dev_ms.push_back((en - s) * 1e-6); p_clReleaseEvent(e); }
    std::vector<double> single; for (int i = 0; i < std::min(repeats, 20); ++i) { if (nbuf > 1) check(p_clSetKernelArg(kern, 0, sizeof(cl_mem), &wbufs[i % nbuf]), "a0n"); double t = now(); check(p_clEnqueueNDRangeKernel(q, kern, 1, nullptr, &global, &local, 0, nullptr, nullptr), "enqueue"); check(p_clFinish(q), "finish"); single.push_back((now() - t) * 1e3); }
    std::sort(dev_ms.begin(), dev_ms.end()); std::sort(single.begin(), single.end());
    double med = dev_ms[dev_ms.size() / 2], p90 = dev_ms[size_t(dev_ms.size() * 0.9)], mn = dev_ms.front();
    std::printf("{\"variant\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"R\":%d,\"WG\":%d,\"pref_wg_multiple\":%zu,\"weight_MB\":%.1f,\"device_ms_min\":%.3f,\"device_ms_med\":%.3f,\"device_ms_p90\":%.3f,\"GBps_med\":%.2f,\"wall_ms_batched\":%.3f,\"wall_ms_single_med\":%.3f,\"rel_l2_vs_exact\":%.3e,\"rel_l2_vs_quantized_ref\":%.3e}\n",
        variant.c_str(), M, N, K, R, WG, pref, weight_bytes / 1e6, mn, med, p90, weight_bytes / (med * 1e-3) / 1e9, wall_batch * 1e3, single[single.size() / 2], std::sqrt(e_exact / nrm), std::sqrt(e_q / nrm));
    for (cl_mem m : {b_y, b_w, b_sc, b_x, b_xq, b_xs}) if (m) p_clReleaseMemObject(m);
    p_clReleaseKernel(kern); p_clReleaseProgram(prog); p_clReleaseCommandQueue(q); p_clReleaseContext(ctx);
    return 0;
}
