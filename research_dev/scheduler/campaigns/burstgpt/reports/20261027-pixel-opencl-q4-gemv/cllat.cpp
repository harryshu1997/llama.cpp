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
#include <thread>
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


struct Mat { int N, K, nb, ng; cl_mem w, sc, xq, xs, y; cl_kernel k; size_t global, local; };
static Mat make(cl_context ctx, cl_program prog, int N, int K, int R, int WG, float fill) {
    cl_int err; Mat m{N, K, K / 32, N / R};
    std::vector<uint8_t> tiles(size_t(m.ng) * m.nb * R * 16); std::vector<uint16_t> scales(size_t(m.ng) * m.nb * R, f2h(0.01f));
    for (auto & b : tiles) b = uint8_t(rng_state = rng_state * 1664525u + 1013904223u);
    std::vector<int8_t> xq(K, int8_t(fill)); std::vector<float> xs(m.nb, 0.02f);
    auto mk = [&](const void * p, size_t bytes) { cl_mem o = p_clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, const_cast<void *>(p), &err); check(err, "buffer"); return o; };
    m.w = mk(tiles.data(), tiles.size()); m.sc = mk(scales.data(), scales.size() * 2); m.xq = mk(xq.data(), xq.size()); m.xs = mk(xs.data(), xs.size() * 4);
    m.y = p_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, size_t(N) * 4, nullptr, &err); check(err, "ybuf");
    m.k = p_clCreateKernel(prog, "q4_global", &err); check(err, "kernel");
    check(p_clSetKernelArg(m.k, 0, sizeof m.w, &m.w), "a0"); check(p_clSetKernelArg(m.k, 1, sizeof m.sc, &m.sc), "a1"); check(p_clSetKernelArg(m.k, 2, sizeof m.xq, &m.xq), "a2"); check(p_clSetKernelArg(m.k, 3, sizeof m.xs, &m.xs), "a3"); check(p_clSetKernelArg(m.k, 4, sizeof m.y, &m.y), "a4"); check(p_clSetKernelArg(m.k, 5, sizeof K, &K), "a5");
    m.global = size_t(m.ng) * WG; m.local = WG; return m;
}
int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: cllat kernel.cl [R] [WG] [profiling 0/1] [layer_gap_ms token_gap_ms tokens]\n"); return 1; }
    int R = argc > 2 ? std::atoi(argv[2]) : 32, WG = argc > 3 ? std::atoi(argv[3]) : 256; bool prof = argc > 4 ? std::atoi(argv[4]) : 0;
    double layer_gap_ms = argc > 5 ? std::atof(argv[5]) : -1, token_gap_ms = argc > 6 ? std::atof(argv[6]) : 150; int tokens = argc > 7 ? std::atoi(argv[7]) : 20; double keepalive_ms = argc > 8 ? std::atof(argv[8]) : 0;
    lib = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW); if (!lib) { std::fprintf(stderr, "FAIL dlopen\n"); return 2; }
#define L(n) load(p_##n, #n);
    L(clGetPlatformIDs) L(clGetDeviceIDs) L(clGetDeviceInfo) L(clCreateContext) L(clCreateCommandQueueWithProperties) L(clCreateProgramWithSource) L(clBuildProgram) L(clGetProgramBuildInfo)
    L(clCreateKernel) L(clSetKernelArg) L(clCreateBuffer) L(clCreateImage) L(clEnqueueNDRangeKernel) L(clEnqueueReadBuffer) L(clFinish) L(clGetEventProfilingInfo) L(clReleaseEvent) L(clReleaseMemObject)
    L(clReleaseKernel) L(clReleaseProgram) L(clReleaseCommandQueue) L(clReleaseContext) L(clGetKernelWorkGroupInfo)
    auto p_clWaitForEvents = (decltype(&clWaitForEvents)) dlsym(lib, "clWaitForEvents"); 
    auto p_clEnqueueWriteBuffer = (decltype(&clEnqueueWriteBuffer)) dlsym(lib, "clEnqueueWriteBuffer");
    cl_platform_id plat; cl_device_id dev; cl_int err;
    check(p_clGetPlatformIDs(1, &plat, nullptr), "platform"); check(p_clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr), "device");
    cl_context ctx = p_clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); check(err, "context");
    cl_queue_properties qp[] = {CL_QUEUE_PROPERTIES, cl_queue_properties(prof ? CL_QUEUE_PROFILING_ENABLE : 0), 0};
    cl_command_queue q = p_clCreateCommandQueueWithProperties(ctx, dev, qp, &err); check(err, "queue");
    std::ifstream f(argv[1]); std::stringstream ss; ss << f.rdbuf(); std::string src = ss.str(); const char * srcp = src.c_str();
    cl_program prog = p_clCreateProgramWithSource(ctx, 1, &srcp, nullptr, &err); check(err, "program");
    char opts[256]; std::snprintf(opts, sizeof opts, "-cl-std=CL3.0 -cl-mad-enable -D R=%d -D WG=%d -D IMGW=8192", R, WG);
    check(p_clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr), "build");
    Mat gate = make(ctx, prog, 17408, 5120, R, WG, 3), up = make(ctx, prog, 17408, 5120, R, WG, 5), down = make(ctx, prog, 5120, 17408, R, WG, 2);
    std::vector<int8_t> act(5120, 1);
    auto layer = [&]() { check(p_clEnqueueNDRangeKernel(q, gate.k, 1, nullptr, &gate.global, &gate.local, 0, nullptr, nullptr), "e"); check(p_clEnqueueNDRangeKernel(q, up.k, 1, nullptr, &up.global, &up.local, 0, nullptr, nullptr), "e"); check(p_clEnqueueNDRangeKernel(q, down.k, 1, nullptr, &down.global, &down.local, 0, nullptr, nullptr), "e"); };
    for (int i = 0; i < 5; ++i) { layer(); check(p_clFinish(q), "f"); }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    auto p90 = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[size_t(v.size() * 0.9)]; };
    Mat tiny = make(ctx, prog, R, 32, R, WG, 1);
    if (layer_gap_ms >= 0) {   // production cadence: 23 layer calls per token with host gaps between them, then the rest of the token
        std::vector<double> per_layer; std::vector<double> first_layer;
        for (int tk = 0; tk < tokens; ++tk) {
            for (int l = 0; l < 23; ++l) {
                double t = now(); layer(); check(p_clFinish(q), "f"); double ms = (now() - t) * 1e3; per_layer.push_back(ms); if (l == 0) first_layer.push_back(ms);
                double until = now() + layer_gap_ms * 1e-3; while (now() < until) {}   // host attention: busy wait keeps the host thread's own clock up, like the worker's spin poll
            }
            if (keepalive_ms > 0) {   // during the token gap, poke the GPU with a one-work-group kernel every keepalive_ms so it does not power down
                double until = now() + token_gap_ms * 1e-3;
                while (now() < until) { std::this_thread::sleep_for(std::chrono::microseconds(long(keepalive_ms * 1e3))); check(p_clEnqueueNDRangeKernel(q, tiny.k, 1, nullptr, &tiny.global, &tiny.local, 0, nullptr, nullptr), "ka"); check(p_clFinish(q), "kf"); }
            } else std::this_thread::sleep_for(std::chrono::milliseconds(long(token_gap_ms)));
        }
        std::sort(per_layer.begin(), per_layer.end()); std::sort(first_layer.begin(), first_layer.end());
        std::printf("{\"cadence\":{\"layer_gap_ms\":%.1f,\"token_gap_ms\":%.0f,\"tokens\":%d},\"layer_wall_ms\":{\"med\":%.3f,\"p90\":%.3f,\"max\":%.3f},\"first_layer_of_token_ms\":{\"med\":%.3f,\"p90\":%.3f}}\n", layer_gap_ms, token_gap_ms, tokens, per_layer[per_layer.size() / 2], per_layer[size_t(per_layer.size() * 0.9)], per_layer.back(), first_layer[first_layer.size() / 2], first_layer[size_t(first_layer.size() * 0.9)]);
        return 0;
    }
    std::vector<double> t_finish, t_event, t_stream, t_tiny, t_write;
    for (int i = 0; i < 40; ++i) { double t = now(); layer(); check(p_clFinish(q), "f"); t_finish.push_back((now() - t) * 1e3); }
    for (int i = 0; i < 40; ++i) { double t = now(); check(p_clEnqueueNDRangeKernel(q, gate.k, 1, nullptr, &gate.global, &gate.local, 0, nullptr, nullptr), "e"); check(p_clEnqueueNDRangeKernel(q, up.k, 1, nullptr, &up.global, &up.local, 0, nullptr, nullptr), "e"); cl_event ev; check(p_clEnqueueNDRangeKernel(q, down.k, 1, nullptr, &down.global, &down.local, 0, nullptr, &ev), "e"); check(p_clWaitForEvents(1, &ev), "w"); p_clReleaseEvent(ev); t_event.push_back((now() - t) * 1e3); }
    { double t = now(); for (int i = 0; i < 40; ++i) layer(); check(p_clFinish(q), "f"); t_stream.push_back((now() - t) * 1e3 / 40); }
    // realistic call: write the activation vector (5 KB) to the device, run the layer, read 5120 floats back
    std::vector<float> out(5120);
    for (int i = 0; i < 40; ++i) { double t = now(); check(p_clEnqueueWriteBuffer(q, gate.xq, CL_FALSE, 0, act.size(), act.data(), 0, nullptr, nullptr), "w"); layer(); check(p_clEnqueueReadBuffer(q, down.y, CL_TRUE, 0, out.size() * 4, out.data(), 0, nullptr, nullptr), "r"); t_write.push_back((now() - t) * 1e3); }
    // tiny kernel round trip: 1 work-group
    for (int i = 0; i < 40; ++i) { double t = now(); check(p_clEnqueueNDRangeKernel(q, tiny.k, 1, nullptr, &tiny.global, &tiny.local, 0, nullptr, nullptr), "e"); check(p_clFinish(q), "f"); t_tiny.push_back((now() - t) * 1e3); }
    std::printf("{\"R\":%d,\"WG\":%d,\"profiling\":%d,\"layer_wall_ms_finish\":{\"med\":%.3f,\"p90\":%.3f},\"layer_wall_ms_event\":{\"med\":%.3f,\"p90\":%.3f},\"layer_wall_ms_streaming\":%.3f,\"layer_wall_ms_write_run_read\":{\"med\":%.3f,\"p90\":%.3f},\"tiny_kernel_roundtrip_ms\":{\"med\":%.3f,\"p90\":%.3f}}\n",
        R, WG, prof, med(t_finish), p90(t_finish), med(t_event), p90(t_event), t_stream[0], med(t_write), p90(t_write), med(t_tiny), p90(t_tiny));
    return 0;
}
