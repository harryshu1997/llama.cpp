#include <algorithm>
// clprobe: OpenCL device capability probe for the Pixel 10 Pro (dlopen vendor loader, no link-time dependency).
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define F(name) static decltype(&name) p_##name;
F(clGetPlatformIDs) F(clGetPlatformInfo) F(clGetDeviceIDs) F(clGetDeviceInfo) F(clCreateContext) F(clCreateProgramWithSource)
F(clBuildProgram) F(clGetProgramBuildInfo) F(clReleaseProgram) F(clReleaseContext) F(clGetSupportedImageFormats)
static void * lib;
template <class T> static void load(T & p, const char * n) { p = (T) dlsym(lib, n); if (!p) { std::fprintf(stderr, "FAIL dlsym %s\n", n); std::exit(2); } }
static std::string dinfo(cl_device_id d, cl_device_info k) { size_t n = 0; p_clGetDeviceInfo(d, k, 0, nullptr, &n); std::string s(n, 0); p_clGetDeviceInfo(d, k, n, s.data(), nullptr); while (!s.empty() && s.back() == 0) s.pop_back(); return s; }
template <class T> static T dval(cl_device_id d, cl_device_info k) { T v{}; p_clGetDeviceInfo(d, k, sizeof(v), &v, nullptr); return v; }
static bool try_build(cl_context ctx, cl_device_id dev, const char * tag, const char * src, const char * opts) {
    cl_int err; const char * s = src; cl_program prog = p_clCreateProgramWithSource(ctx, 1, &s, nullptr, &err);
    if (err) { std::printf("{\"build\":\"%s\",\"status\":\"create_failed\",\"err\":%d}\n", tag, err); return false; }
    err = p_clBuildProgram(prog, 1, &dev, opts, nullptr, nullptr);
    size_t n = 0; p_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n); std::string log(n, 0); p_clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
    for (auto & c : log) if (c == '\n' || c == '"') c = ' ';
    std::printf("{\"build\":\"%s\",\"status\":\"%s\",\"err\":%d,\"log\":\"%.400s\"}\n", tag, err ? "FAIL" : "OK", err, log.c_str());
    p_clReleaseProgram(prog); return err == 0;
}
int main() {
    const char * paths[] = {"/vendor/lib64/libOpenCL.so", "libOpenCL.so", "/vendor/lib64/libPVROCL.so"};
    for (auto p : paths) { lib = dlopen(p, RTLD_NOW); std::printf("{\"dlopen\":\"%s\",\"ok\":%d,\"err\":\"%s\"}\n", p, lib != nullptr, lib ? "" : dlerror()); if (lib) break; }
    if (!lib) return 2;
#define L(n) load(p_##n, #n);
    L(clGetPlatformIDs) L(clGetPlatformInfo) L(clGetDeviceIDs) L(clGetDeviceInfo) L(clCreateContext) L(clCreateProgramWithSource) L(clBuildProgram) L(clGetProgramBuildInfo) L(clReleaseProgram) L(clReleaseContext) L(clGetSupportedImageFormats)
    cl_uint np = 0; cl_int err = p_clGetPlatformIDs(0, nullptr, &np); std::printf("{\"platforms\":%u,\"err\":%d}\n", np, err); if (err || !np) return 3;
    std::vector<cl_platform_id> plats(np); p_clGetPlatformIDs(np, plats.data(), nullptr);
    for (auto pl : plats) {
        char name[256] = {0}, ver[256] = {0}; p_clGetPlatformInfo(pl, CL_PLATFORM_NAME, sizeof name, name, nullptr); p_clGetPlatformInfo(pl, CL_PLATFORM_VERSION, sizeof ver, ver, nullptr);
        cl_uint nd = 0; p_clGetDeviceIDs(pl, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd); std::vector<cl_device_id> devs(nd); p_clGetDeviceIDs(pl, CL_DEVICE_TYPE_ALL, nd, devs.data(), nullptr);
        std::printf("{\"platform\":\"%s\",\"version\":\"%s\",\"devices\":%u}\n", name, ver, nd);
        for (auto d : devs) {
            std::printf("{\"device\":\"%s\",\"version\":\"%s\",\"driver\":\"%s\",\"cl_c\":\"%s\",\"cu\":%u,\"clock_mhz\":%u,\"max_wg\":%zu,\"local_kb\":%llu,\"const_kb\":%llu,\"global_mb\":%llu,\"max_alloc_mb\":%llu,\"image\":%d,\"img2d_w\":%zu,\"img2d_h\":%zu,\"img1d_buf\":%zu,\"fp16_pref_w\":%u,\"char_pref_w\":%u,\"unified\":%d}\n",
                dinfo(d, CL_DEVICE_NAME).c_str(), dinfo(d, CL_DEVICE_VERSION).c_str(), dinfo(d, CL_DRIVER_VERSION).c_str(), dinfo(d, CL_DEVICE_OPENCL_C_VERSION).c_str(),
                dval<cl_uint>(d, CL_DEVICE_MAX_COMPUTE_UNITS), dval<cl_uint>(d, CL_DEVICE_MAX_CLOCK_FREQUENCY), dval<size_t>(d, CL_DEVICE_MAX_WORK_GROUP_SIZE),
                (unsigned long long) dval<cl_ulong>(d, CL_DEVICE_LOCAL_MEM_SIZE) / 1024, (unsigned long long) dval<cl_ulong>(d, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) / 1024,
                (unsigned long long) dval<cl_ulong>(d, CL_DEVICE_GLOBAL_MEM_SIZE) >> 20, (unsigned long long) dval<cl_ulong>(d, CL_DEVICE_MAX_MEM_ALLOC_SIZE) >> 20,
                (int) dval<cl_bool>(d, CL_DEVICE_IMAGE_SUPPORT), dval<size_t>(d, CL_DEVICE_IMAGE2D_MAX_WIDTH), dval<size_t>(d, CL_DEVICE_IMAGE2D_MAX_HEIGHT), dval<size_t>(d, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE),
                dval<cl_uint>(d, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF), dval<cl_uint>(d, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR), (int) dval<cl_bool>(d, CL_DEVICE_HOST_UNIFIED_MEMORY));
            std::string ext = dinfo(d, CL_DEVICE_EXTENSIONS); std::printf("{\"extensions\":\"%s\"}\n", ext.c_str());
            cl_context ctx = p_clCreateContext(nullptr, 1, &d, nullptr, nullptr, &err); if (err) { std::printf("{\"context_err\":%d}\n", err); continue; }
            cl_uint nf = 0; p_clGetSupportedImageFormats(ctx, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE1D_BUFFER, 0, nullptr, &nf); std::vector<cl_image_format> fm(nf); p_clGetSupportedImageFormats(ctx, CL_MEM_READ_ONLY, CL_MEM_OBJECT_IMAGE1D_BUFFER, nf, fm.data(), nullptr);
            std::printf("{\"image1d_buffer_formats\":%u,\"rgba_u32\":%d,\"rgba_half\":%d,\"rgba_u8\":%d}\n", nf,
                (int) std::any_of(fm.begin(), fm.end(), [](cl_image_format f){ return f.image_channel_order == CL_RGBA && f.image_channel_data_type == CL_UNSIGNED_INT32; }),
                (int) std::any_of(fm.begin(), fm.end(), [](cl_image_format f){ return f.image_channel_order == CL_RGBA && f.image_channel_data_type == CL_HALF_FLOAT; }),
                (int) std::any_of(fm.begin(), fm.end(), [](cl_image_format f){ return f.image_channel_order == CL_RGBA && f.image_channel_data_type == CL_UNSIGNED_INT8; }));
            try_build(ctx, d, "fp16", "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n__kernel void k(__global const half8 * w, __global float * o) { half8 v = w[get_global_id(0)]; o[get_global_id(0)] = (float)(v.s0 + v.s7); }", "");
            try_build(ctx, d, "integer_dot_product", "#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable\n__kernel void k(__global const uint * a, __global const uint * b, __global int * o) { uint i = get_global_id(0); o[i] = dot_acc_sat(as_char4(a[i]), as_char4(b[i]), 0); }", "-cl-std=CL3.0");
            try_build(ctx, d, "integer_dot_product_packed", "#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable\n__kernel void k(__global const uint * a, __global const uint * b, __global int * o) { uint i = get_global_id(0); o[i] = dot_4x8packed_ss_int(a[i], b[i]); }", "-cl-std=CL3.0");
            try_build(ctx, d, "arm_dot", "#pragma OPENCL EXTENSION cl_arm_integer_dot_product_int8 : enable\n__kernel void k(__global const uint * a, __global const uint * b, __global int * o) { uint i = get_global_id(0); o[i] = arm_dot(as_char4(a[i]), as_char4(b[i])); }", "");
            try_build(ctx, d, "image1d_buffer_constant", "__constant sampler_t s = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n__kernel void k(__read_only image1d_buffer_t w, __constant uint4 * act, __global uint * o) { int i = get_global_id(0); uint4 v = read_imageui(w, i); uint4 a = act[i & 255]; o[i] = v.x ^ a.x ^ v.w ^ a.w; }", "");
            try_build(ctx, d, "subgroups", "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n__kernel void k(__global float * o) { o[get_global_id(0)] = sub_group_reduce_add((float) get_sub_group_local_id()) + get_sub_group_size(); }", "-cl-std=CL2.0");
            p_clReleaseContext(ctx);
        }
    }
    std::puts("{\"probe\":\"done\"}"); return 0;
}
