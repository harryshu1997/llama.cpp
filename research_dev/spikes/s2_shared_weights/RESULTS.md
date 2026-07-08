# S2 spike — single-copy weight sharing (NPU ↔ GPU) — PASS on both phones

*2026-07-08 · gates Design-A requirement (3): one physical f16 weight copy read by both the Hexagon NPU (HMX) and the Adreno GPU (xmem).*

## Verdict: **PASS on op12 (Adreno 750) and op15 (Adreno 840).**

A Hexagon `rpcmem` dmabuf, written by the CPU (standing in for resident f16 weights), is imported into OpenCL and read **bit-exact** by an Adreno GPU kernel: **0 / 1048576 uint32 mismatches** on both phones. Since `ggml-hexagon` already reads `rpcmem` natively for HMX, the GPU reading the *same fd* closes the loop — **one copy serves both engines.**

## The mechanism that works (both devices, identical)

- **Allocate** (Hexagon side): `rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM=25, RPCMEM_DEFAULT_FLAGS=1, size)` → CPU `base`; `rpcmem_to_fd(base)` → dmabuf fd. (`libcdsprpc.so`, dlopen-able.)
- **Import** (OpenCL side): the winning combo is
  ```c
  cl_mem_ion_host_ptr h = {
      .ext_host_ptr = { .allocation_type = CL_MEM_ION_HOST_PTR_QCOM /*0x40A8*/,
                        .host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM /*0x40A4*/ },
      .ion_filedesc = fd, .ion_hostptr = base };
  clCreateBuffer(ctx, CL_MEM_EXT_HOST_PTR_QCOM | CL_MEM_USE_HOST_PTR, size, &h, &err);
  ```
  i.e. **ion allocation_type + UNCACHED + `EXT_HOST_PTR_QCOM | USE_HOST_PTR`.**

## What failed (and the lesson)

- Without `CL_MEM_USE_HOST_PTR` → **-30 (CL_INVALID_VALUE)** for every cache policy / alloc type. `USE_HOST_PTR` is mandatory alongside `EXT_HOST_PTR_QCOM`.
- `dmabuf` allocation_type (`CL_MEM_DMABUF_HOST_PTR_QCOM` 0x40C7) + `USE_HOST_PTR` → **-59 (CL_INVALID_OPERATION)**. Even though `CL_DEVICE_EXTENSIONS` advertises `cl_qcom_dmabuf_host_ptr`, the rpcmem fd imports as **ION**, not the dmabuf struct. Use `cl_mem_ion_host_ptr`.
- Cache policy: `UNCACHED` works. (rpcmem was allocated CACHED; consider matching WRITEBACK + explicit cache flush for the real integration to avoid per-read coherency cost — TBD in Build 3.)

## Device facts (via `cl_ext_probe.c`)

- op12: QUALCOMM Adreno 750, OpenCL 3.0, driver 0762.34. op15: Adreno 840, OpenCL 3.0, driver 0842.21.
- Both advertise `cl_qcom_dmabuf_host_ptr`, `cl_qcom_ext_host_ptr`, `cl_qcom_ext_host_ptr_iocoherent`, `cl_qcom_android_native_buffer_host_ptr`. **`clImportMemoryARM` is absent** (ARM ext unsupported — use the QCOM path).
- `CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM=0`, `CL_DEVICE_PAGE_SIZE_QCOM=4096`. rpcmem allocations are page-aligned.
- `clGetDeviceIDs(CL_DEVICE_TYPE_ALL)` returns **-31** on these ICDs — must pass `CL_DEVICE_TYPE_GPU`.

## Caveats / open for Build 3

- **RSS**: the probe's RSS delta is confounded by lazy OpenCL runtime init + a 4 MB sanity buffer allocated in between, so it is not a clean "0 added" measurement (op12 +14 MB, op15 +19 MB for a 4 MB payload — mostly CL init, not a weight copy). Re-measure in Build 3 by importing with no other CL allocation in the window, and via `/proc/pid/smaps` dmabuf accounting.
- **Cache coherency**: UNCACHED import avoids stale reads but costs GPU read bandwidth. For the real weight path, prefer CACHED rpcmem + WRITEBACK import + a one-time `rpcmem` cache flush after weight load (weights are read-only at inference → no per-step coherency needed).
- The os8 xmem prepack still derives a small GPU-private tile from the shared linear copy — sharing removes the **linear** f16 duplication (the big one), not the tile.

## Build

```
NDK=<r29>; $NDK/.../aarch64-linux-android31-clang s2_shared_probe.c -o s2_shared_probe
adb push s2_shared_probe /data/local/tmp && adb shell /data/local/tmp/s2_shared_probe
```
No vendor link libs needed — `libcdsprpc.so` (rpcmem) and `libOpenCL.so` are dlopen'd at runtime.
