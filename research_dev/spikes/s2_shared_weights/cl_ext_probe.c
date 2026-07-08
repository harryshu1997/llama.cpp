// S2 precursor: dump CL_DEVICE_EXTENSIONS + key caps on-device via dlopen of the
// vendor libOpenCL — no OpenCL link lib needed. Tells us whether the Adreno driver
// advertises cl_arm_import_memory / cl_qcom_dmabuf_host_ptr / cl_qcom_ion_host_ptr
// (the enablers for a single shared f16 weight copy across HTP0 + GPUOpenCL).
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int          cl_int;
typedef unsigned int cl_uint;
typedef uint64_t     cl_ulong;
typedef uint64_t     cl_bitfield;
typedef cl_bitfield  cl_device_type;
typedef void *       cl_platform_id;
typedef void *       cl_device_id;
typedef cl_uint      cl_platform_info;
typedef cl_uint      cl_device_info;

#define CL_DEVICE_TYPE_ALL        0xFFFFFFFFFFFFFFFFULL
#define CL_PLATFORM_NAME          0x0902
#define CL_PLATFORM_EXTENSIONS    0x0904
#define CL_DEVICE_NAME            0x102B
#define CL_DEVICE_VERSION         0x102F
#define CL_DRIVER_VERSION         0x102D
#define CL_DEVICE_EXTENSIONS      0x1030

typedef cl_int (*pfn_getPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*pfn_getPlatformInfo)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
typedef cl_int (*pfn_getDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_int (*pfn_getDeviceInfo)(cl_device_id, cl_device_info, size_t, void *, size_t *);

static const char * CANDIDATES[] = {
    "cl_arm_import_memory", "cl_arm_import_memory_dma_buf", "cl_arm_import_memory_host",
    "cl_qcom_dmabuf_host_ptr", "cl_qcom_ion_host_ptr", "cl_qcom_ext_host_ptr",
    "cl_qcom_android_native_buffer_host_ptr", "cl_img_import_memory",
};

int main(void) {
    const char * paths[] = { "libOpenCL.so", "/vendor/lib64/libOpenCL.so", "/system/lib64/libOpenCL.so", "/vendor/lib64/egl/libGLES_mali.so" };
    void * lib = NULL; const char * used = NULL;
    for (unsigned i = 0; i < sizeof(paths)/sizeof(paths[0]); ++i) { lib = dlopen(paths[i], RTLD_NOW); if (lib) { used = paths[i]; break; } }
    if (!lib) { printf("FAIL: could not dlopen libOpenCL (%s)\n", dlerror()); return 1; }
    printf("dlopen OK: %s\n", used);

    pfn_getPlatformIDs  clGetPlatformIDs  = (pfn_getPlatformIDs)  dlsym(lib, "clGetPlatformIDs");
    pfn_getPlatformInfo clGetPlatformInfo = (pfn_getPlatformInfo) dlsym(lib, "clGetPlatformInfo");
    pfn_getDeviceIDs    clGetDeviceIDs    = (pfn_getDeviceIDs)    dlsym(lib, "clGetDeviceIDs");
    pfn_getDeviceInfo   clGetDeviceInfo   = (pfn_getDeviceInfo)   dlsym(lib, "clGetDeviceInfo");
    void * imp = dlsym(lib, "clImportMemoryARM");
    printf("clImportMemoryARM symbol: %s\n", imp ? "PRESENT" : "absent");
    if (!clGetPlatformIDs || !clGetDeviceIDs || !clGetDeviceInfo) { printf("FAIL: missing core syms\n"); return 1; }

    cl_platform_id plats[8]; cl_uint nplat = 0;
    if (clGetPlatformIDs(8, plats, &nplat) != 0 || nplat == 0) { printf("FAIL: no platforms\n"); return 1; }
    char buf[8192];
    for (cl_uint p = 0; p < nplat; ++p) {
        if (clGetPlatformInfo) { clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sizeof(buf), buf, NULL); printf("platform[%u]: %s\n", p, buf); }
        cl_device_id devs[8]; cl_uint ndev = 0;
        cl_int derr = clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 8, devs, &ndev);
        printf("  clGetDeviceIDs(ALL) rc=%d ndev=%u\n", derr, ndev);
        if (derr != 0 || ndev == 0) { derr = clGetDeviceIDs(plats[p], 0x4 /*GPU*/, 8, devs, &ndev); printf("  clGetDeviceIDs(GPU) rc=%d ndev=%u\n", derr, ndev); }
        if (derr != 0 || ndev == 0) continue;
        for (cl_uint d = 0; d < ndev; ++d) {
            clGetDeviceInfo(devs[d], CL_DEVICE_NAME,    sizeof(buf), buf, NULL); printf("  device[%u]: %s\n", d, buf);
            clGetDeviceInfo(devs[d], CL_DEVICE_VERSION, sizeof(buf), buf, NULL); printf("    version: %s\n", buf);
            if (clGetDeviceInfo(devs[d], CL_DRIVER_VERSION, sizeof(buf), buf, NULL) == 0) printf("    driver : %s\n", buf);
            size_t ext_sz = 0; clGetDeviceInfo(devs[d], CL_DEVICE_EXTENSIONS, 0, NULL, &ext_sz);
            char * ext = (char *) malloc(ext_sz + 1); ext[0] = 0;
            clGetDeviceInfo(devs[d], CL_DEVICE_EXTENSIONS, ext_sz, ext, NULL); ext[ext_sz ? ext_sz : 0] = 0;
            printf("    >>> import-relevant extensions:\n");
            int any = 0;
            for (unsigned c = 0; c < sizeof(CANDIDATES)/sizeof(CANDIDATES[0]); ++c)
                if (strstr(ext, CANDIDATES[c])) { printf("        [YES] %s\n", CANDIDATES[c]); any = 1; }
            if (!any) printf("        (none of the ARM/QCOM import extensions advertised)\n");
            printf("    --- full extension string ---\n%s\n", ext);
            free(ext);
        }
    }
    return 0;
}
