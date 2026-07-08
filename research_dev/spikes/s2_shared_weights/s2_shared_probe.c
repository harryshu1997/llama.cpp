// S2 spike (Design-A requirement 3): prove ONE physical rpcmem/dmabuf copy is
// readable by the Adreno GPU via the QCOM ext-host-ptr import — the DSP/HMX side
// already reads rpcmem natively (that is how ggml-hexagon loads weights), so if
// the GPU can read the SAME fd's bytes, one f16 weight copy serves both engines.
//
// Flow: rpcmem_alloc2 (Hexagon system dmabuf) -> write known byte pattern (stands
// in for f16 weights) -> rpcmem_to_fd -> clCreateBuffer(CL_MEM_EXT_HOST_PTR_QCOM,
// cl_mem_ion_host_ptr{fd,base}) -> GPU kernel copies it out -> bit-compare +
// VmRSS delta (import must NOT add ~size bytes = no second copy).
//
// Everything is dlopen'd (libcdsprpc.so for rpcmem, libOpenCL.so for CL) so no
// vendor link libs are needed at build time.
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- minimal OpenCL typedefs ---- */
typedef int8_t   cl_char; typedef uint8_t cl_uchar;
typedef int32_t  cl_int;  typedef uint32_t cl_uint;
typedef uint64_t cl_ulong, cl_bitfield, cl_device_type, cl_mem_flags, cl_command_queue_properties;
typedef size_t   cl_size;
typedef void *cl_platform_id, *cl_device_id, *cl_context, *cl_command_queue, *cl_mem, *cl_program, *cl_kernel;
typedef intptr_t cl_context_properties;
typedef cl_uint  cl_device_info;

#define CL_DEVICE_TYPE_GPU        (1<<2)
#define CL_DEVICE_EXTENSIONS      0x1030
#define CL_MEM_READ_ONLY          (1<<2)
#define CL_MEM_WRITE_ONLY         (1<<1)
#define CL_MEM_EXT_HOST_PTR_QCOM  (1<<29)
#define CL_MEM_ION_HOST_PTR_QCOM  0x40A8
#define CL_MEM_HOST_UNCACHED_QCOM 0x40A4
#define CL_MEM_HOST_IOCOHERENT_QCOM 0x40A9
#define CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM 0x40A0
#define CL_DEVICE_PAGE_SIZE_QCOM  0x40A1
#define CL_TRUE                   1

typedef struct _cl_mem_ext_host_ptr { cl_uint allocation_type; cl_uint host_cache_policy; } cl_mem_ext_host_ptr;
typedef struct _cl_mem_ion_host_ptr { cl_mem_ext_host_ptr ext_host_ptr; int ion_filedesc; void* ion_hostptr; } cl_mem_ion_host_ptr;
/* cl_qcom_dmabuf_host_ptr — not in the NDK header; from the Adreno OpenCL SDK */
#define CL_MEM_DMABUF_HOST_PTR_QCOM 0x40C7
typedef struct _cl_mem_dmabuf_host_ptr { cl_mem_ext_host_ptr ext_host_ptr; int dmabuf_filedesc; void* dmabuf_hostptr; } cl_mem_dmabuf_host_ptr;

/* ---- CL fn pointers ---- */
typedef cl_int (*p_ids)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*p_pids)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*p_dinfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);
typedef cl_context (*p_ctx)(const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (*p_cq)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_mem (*p_buf)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_program (*p_prog)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*p_build)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_int (*p_proginfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_kernel (*p_krn)(cl_program, const char*, cl_int*);
typedef cl_int (*p_arg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*p_ndr)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
typedef cl_int (*p_rd)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const void*, void*);
typedef cl_int (*p_fin)(cl_command_queue);
#define CL_PROGRAM_BUILD_LOG 0x1183

static long vmrss_kb(void) {
    FILE* f = fopen("/proc/self/status", "r"); if (!f) return -1;
    char line[256]; long kb = -1;
    while (fgets(line, sizeof line, f)) if (!strncmp(line, "VmRSS:", 6)) { sscanf(line+6, "%ld", &kb); break; }
    fclose(f); return kb;
}

int main(void) {
    const int N = 1<<20;                 // 1 Mi uint32 = 4 MiB payload
    const size_t bytes = (size_t)N * 4;

    /* rpcmem via libcdsprpc.so */
    void* rl = dlopen("libcdsprpc.so", RTLD_NOW); if (!rl) rl = dlopen("/vendor/lib64/libcdsprpc.so", RTLD_NOW);
    if (!rl) { printf("FAIL dlopen libcdsprpc: %s\n", dlerror()); return 1; }
    void* (*rpcmem_alloc2)(int,uint32_t,size_t) = (void*(*)(int,uint32_t,size_t)) dlsym(rl, "rpcmem_alloc2");
    void* (*rpcmem_alloc)(int,uint32_t,int)     = (void*(*)(int,uint32_t,int))    dlsym(rl, "rpcmem_alloc");
    int   (*rpcmem_to_fd)(void*)                = (int(*)(void*))                 dlsym(rl, "rpcmem_to_fd");
    void  (*rpcmem_free)(void*)                 = (void(*)(void*))                dlsym(rl, "rpcmem_free");
    void  (*rpcmem_init)(void)                  = (void(*)(void))                 dlsym(rl, "rpcmem_init");
    if (rpcmem_init) rpcmem_init();
    if ((!rpcmem_alloc2 && !rpcmem_alloc) || !rpcmem_to_fd) { printf("FAIL: rpcmem syms missing\n"); return 1; }

    long rss0 = vmrss_kb();
    const int HEAP_SYSTEM = 25; const uint32_t FLAGS = 1;   // RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS
    size_t alloc_bytes = bytes + (1<<16);                   // + slack for QCOM padding/page
    uint8_t* base = rpcmem_alloc2 ? (uint8_t*)rpcmem_alloc2(HEAP_SYSTEM, FLAGS, alloc_bytes)
                                  : (uint8_t*)rpcmem_alloc (HEAP_SYSTEM, FLAGS, (int)alloc_bytes);
    if (!base) { printf("FAIL: rpcmem_alloc returned NULL\n"); return 1; }
    int fd = rpcmem_to_fd(base);
    printf("rpcmem: base=%p fd=%d alloc=%zu bytes (%s)\n", (void*)base, fd, alloc_bytes, rpcmem_alloc2?"alloc2":"alloc");
    long rss1 = vmrss_kb();

    /* known pattern (as if HMX f16 weights were resident here) */
    uint32_t* w = (uint32_t*)base;
    for (int i = 0; i < N; ++i) w[i] = (uint32_t)(i*2654435761u + 0x9e3779b9u);

    /* OpenCL via libOpenCL.so */
    void* cl = dlopen("libOpenCL.so", RTLD_NOW); if (!cl) cl = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW);
    if (!cl) { printf("FAIL dlopen libOpenCL: %s\n", dlerror()); return 1; }
    p_pids   clGetPlatformIDs = (p_pids)  dlsym(cl,"clGetPlatformIDs");
    p_ids    clGetDeviceIDs   = (p_ids)   dlsym(cl,"clGetDeviceIDs");
    p_dinfo  clGetDeviceInfo  = (p_dinfo) dlsym(cl,"clGetDeviceInfo");
    p_ctx    clCreateContext  = (p_ctx)   dlsym(cl,"clCreateContext");
    p_cq     clCreateCommandQueue=(p_cq)  dlsym(cl,"clCreateCommandQueue");
    p_buf    clCreateBuffer   = (p_buf)   dlsym(cl,"clCreateBuffer");
    p_prog   clCreateProgramWithSource=(p_prog)dlsym(cl,"clCreateProgramWithSource");
    p_build  clBuildProgram   = (p_build) dlsym(cl,"clBuildProgram");
    p_proginfo clGetProgramBuildInfo=(p_proginfo)dlsym(cl,"clGetProgramBuildInfo");
    p_krn    clCreateKernel   = (p_krn)   dlsym(cl,"clCreateKernel");
    p_arg    clSetKernelArg   = (p_arg)   dlsym(cl,"clSetKernelArg");
    p_ndr    clEnqueueNDRangeKernel=(p_ndr)dlsym(cl,"clEnqueueNDRangeKernel");
    p_rd     clEnqueueReadBuffer=(p_rd)   dlsym(cl,"clEnqueueReadBuffer");
    p_fin    clFinish         = (p_fin)   dlsym(cl,"clFinish");
    if (!clCreateContext||!clCreateBuffer||!clEnqueueNDRangeKernel) { printf("FAIL: CL syms missing\n"); return 1; }

    cl_platform_id plat; cl_uint np=0; clGetPlatformIDs(1,&plat,&np);
    cl_device_id dev; cl_uint nd=0;
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &nd)!=0 || nd==0) { printf("FAIL: no GPU device\n"); return 1; }
    cl_uint pad=0, page=0; cl_int rpad, rpage;
    rpad  = clGetDeviceInfo(dev, CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, sizeof(pad), &pad, NULL);
    rpage = clGetDeviceInfo(dev, CL_DEVICE_PAGE_SIZE_QCOM, sizeof(page), &page, NULL);
    printf("QCOM caps: ext_mem_padding=%u (rc=%d) page_size=%u (rc=%d)\n", pad, rpad, page, rpage);

    cl_int err=0;
    cl_context ctx = clCreateContext(NULL,1,&dev,NULL,NULL,&err); if(err){printf("FAIL ctx err=%d\n",err);return 1;}
    cl_command_queue q = clCreateCommandQueue(ctx,dev,0,&err); if(err){printf("FAIL cq err=%d\n",err);return 1;}

    /* sanity: a plain device buffer must succeed (rules out context/signature issues) */
    { cl_int se=0; cl_mem sb = clCreateBuffer(ctx, 1/*RW*/, bytes, NULL, &se);
      printf("sanity plain clCreateBuffer(RW,%zu): err=%d (%s)\n", bytes, se, sb?"OK":"NULL"); }

    /* THE IMPORT: alias the rpcmem dmabuf fd as a cl_mem (no copy).
       Sweep {alloc_type} x {cache_policy} x {mem_flags} to find the combo the driver accepts. */
    const cl_uint POLS[2]   = { 0x40A4,0x40A5 };   // uncached, writeback
    const char*   POLN[2]   = { "uncached","writeback" };
    const cl_uint ATYPE[2]  = { CL_MEM_DMABUF_HOST_PTR_QCOM, CL_MEM_ION_HOST_PTR_QCOM };
    const char*   ATYN[2]   = { "dmabuf","ion" };
    #define QC CL_MEM_EXT_HOST_PTR_QCOM
    const cl_mem_flags MF[4]= { QC, QC|(1<<2)/*RO*/, QC|(1<<3)/*USE_HOST_PTR*/, QC|(1<<3)|(1<<0)/*USE|RW*/ };
    const char*   MFN[4]    = { "EXT","EXT|RO","EXT|USEHOST","EXT|USEHOST|RW" };
    cl_mem wbuf = NULL; char how[64] = {0};
    for (int a=0;a<2 && !wbuf;++a) for (int p=0;p<2 && !wbuf;++p) for (int m=0;m<4 && !wbuf;++m) {
        /* dmabuf and ion structs share the same layout: {ext_host_ptr, int fd, void* hostptr} */
        cl_mem_dmabuf_host_ptr hp; memset(&hp,0,sizeof hp);
        hp.ext_host_ptr.allocation_type  = ATYPE[a];
        hp.ext_host_ptr.host_cache_policy = POLS[p];
        hp.dmabuf_filedesc = fd; hp.dmabuf_hostptr = base;
        cl_int e = 0;
        cl_mem b = clCreateBuffer(ctx, MF[m], bytes, &hp, &e);
        if (b && !e) { wbuf = b; snprintf(how,sizeof how,"%s/%s/%s", ATYN[a],POLN[p],MFN[m]); break; }
        printf("  try %s/%s/%s -> err=%d\n", ATYN[a],POLN[p],MFN[m], e);
    }
    if (!wbuf) { printf("FAIL: all QCOM import attempts failed\n"); rpcmem_free(base); return 2; }
    long rss2 = vmrss_kb();
    printf("IMPORT OK (%s): cl_mem aliases rpcmem fd=%d\n", how, fd);

    cl_mem obuf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    const char* src = "__kernel void cp(__global const uint* w,__global uint* o){int i=get_global_id(0);o[i]=w[i];}";
    size_t slen = strlen(src);
    cl_program prog = clCreateProgramWithSource(ctx,1,&src,&slen,&err);
    if (clBuildProgram(prog,1,&dev,"",NULL,NULL)!=0) {
        char log[4096]={0}; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,sizeof log,log,NULL);
        printf("FAIL build:\n%s\n", log); return 2;
    }
    cl_kernel k = clCreateKernel(prog,"cp",&err);
    clSetKernelArg(k,0,sizeof(cl_mem),&wbuf);
    clSetKernelArg(k,1,sizeof(cl_mem),&obuf);
    size_t gws = N;
    if (clEnqueueNDRangeKernel(q,k,1,NULL,&gws,NULL,0,NULL,NULL)!=0){printf("FAIL enqueue\n");return 2;}
    clFinish(q);

    uint32_t* out = (uint32_t*)malloc(bytes);
    clEnqueueReadBuffer(q,obuf,CL_TRUE,0,bytes,out,0,NULL,NULL);

    /* bit-compare: GPU-read bytes must equal what the CPU wrote into rpcmem */
    long mism=0; int first=-1;
    for (int i=0;i<N;++i) if (out[i]!=w[i]) { if(first<0)first=i; mism++; }
    printf("BITCOMPARE: %ld / %d mismatches%s%s\n", mism, N,
           first>=0?" first@":"", first>=0?"":" -> exact match");
    if (first>=0) printf("  first mismatch @%d: gpu=%08x cpu=%08x\n", first, out[first], w[first]);

    printf("RSS(kB): before_alloc=%ld after_alloc+fill=%ld after_import=%ld\n", rss0, rss1, rss2);
    printf("  alloc added %ld kB (payload=%zu kB); IMPORT added %ld kB (want ~0 => no 2nd copy)\n",
           rss1-rss0, bytes/1024, rss2-rss1);

    printf(mism==0 ? "\nS2 RESULT: PASS — GPU reads the SAME physical rpcmem dmabuf as the DSP would (one copy).\n"
                   : "\nS2 RESULT: FAIL — GPU read differs from CPU-written rpcmem bytes.\n");
    free(out); rpcmem_free(base);
    return mism==0?0:2;
}
