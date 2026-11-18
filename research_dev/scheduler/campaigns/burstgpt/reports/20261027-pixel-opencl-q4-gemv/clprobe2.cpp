#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
int main() {
    void * lib = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW); if (!lib) return 2;
    auto gp = (decltype(&clGetPlatformIDs)) dlsym(lib, "clGetPlatformIDs"); auto gd = (decltype(&clGetDeviceIDs)) dlsym(lib, "clGetDeviceIDs"); auto gi = (decltype(&clGetDeviceInfo)) dlsym(lib, "clGetDeviceInfo");
    cl_platform_id pl; cl_device_id d; gp(1, &pl, nullptr); gd(pl, CL_DEVICE_TYPE_GPU, 1, &d, nullptr);
    struct { cl_bool a, b, c, d, e, f; } p8{}, p4{};
    cl_int e1 = gi(d, 0x1074, sizeof p8, &p8, nullptr), e2 = gi(d, 0x1075, sizeof p4, &p4, nullptr);
    cl_device_svm_capabilities svm = 0; gi(d, CL_DEVICE_SVM_CAPABILITIES, sizeof svm, &svm, nullptr);
    cl_uint maxsg = 0; gi(d, 0x105C /*CL_DEVICE_MAX_NUM_SUB_GROUPS*/, sizeof maxsg, &maxsg, nullptr);
    cl_bool devq = 0; cl_uint devqmax = 0; gi(d, CL_DEVICE_MAX_ON_DEVICE_QUEUES, sizeof devqmax, &devqmax, nullptr);
    cl_uint pw16 = 0; gi(d, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof pw16, &pw16, nullptr);
    cl_bool nonuniform = 0; gi(d, 0x1067 /*CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT*/, sizeof nonuniform, &nonuniform, nullptr);
    std::printf("{\"dot8bit\":{\"err\":%d,\"signed\":%u,\"unsigned\":%u,\"mixed\":%u,\"acc_signed\":%u,\"acc_unsigned\":%u,\"acc_mixed\":%u},\"dot4x8packed\":{\"err\":%d,\"signed\":%u,\"unsigned\":%u,\"mixed\":%u,\"acc_signed\":%u,\"acc_unsigned\":%u,\"acc_mixed\":%u},\"svm_caps\":%llu,\"svm_coarse\":%d,\"svm_fine_buffer\":%d,\"svm_fine_system\":%d,\"svm_atomics\":%d,\"max_sub_groups\":%u,\"on_device_queues\":%u,\"pref_short_w\":%u,\"nonuniform_wg\":%u}\n",
        e1, p8.a, p8.b, p8.c, p8.d, p8.e, p8.f, e2, p4.a, p4.b, p4.c, p4.d, p4.e, p4.f, (unsigned long long) svm, int(bool(svm & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)), int(bool(svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)), int(bool(svm & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)), int(bool(svm & CL_DEVICE_SVM_ATOMICS)), maxsg, devqmax, pw16, nonuniform);
    (void) devq; return 0;
}
