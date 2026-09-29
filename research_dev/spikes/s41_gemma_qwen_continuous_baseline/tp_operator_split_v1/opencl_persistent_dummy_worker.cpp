// Phone-side OpenCL doorbell latency probe with fine-grained SVM.
//
// usage:
//   opencl_persistent_dummy_worker <elements> <backend> <noop|sqr>
//       <repeats> <requests>

// This is an experimental queue control. OpenCLPersistent keeps one workgroup
// resident. OpenCLDoorbellDispatch relaunches that same workgroup per request.

#define CL_TARGET_OPENCL_VERSION 300

#include "backend_dummy_protocol.h"
#include "ggml.h"

#include <CL/cl.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <zlib.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

static bool read_exact(int fd, void * destination, size_t size) {
    uint8_t * pointer = static_cast<uint8_t *>(destination);
    while (size > 0) {
        const ssize_t count = read(fd, pointer, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        pointer += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

static bool write_exact(int fd, const void * source, size_t size) {
    const uint8_t * pointer = static_cast<const uint8_t *>(source);
    while (size > 0) {
        const ssize_t count = write(fd, pointer, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        pointer += count;
        size -= static_cast<size_t>(count);
    }
    return true;
}

static uint64_t now_ns() {
    timespec value = {};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL + static_cast<uint64_t>(value.tv_nsec);
}

static uint32_t crc32_bytes(const void * data, size_t size) {
    return static_cast<uint32_t>(crc32(0, static_cast<const Bytef *>(data), static_cast<uInt>(size)));
}

static void print_build_log(cl_program program, cl_device_id device) {
    size_t size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
    std::vector<char> log(size + 1, 0);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
    fprintf(stderr, "%s\n", log.data());
}

static const char * persistent_source = R"CLC(
__kernel void persistent_sqr(
        volatile __global atomic_uint * request_seq,
        volatile __global atomic_uint * done_seq,
        volatile __global atomic_uint * stop,
        volatile __global atomic_uint * ready,
        __global float * data,
        const uint elements,
        const uint repeats) {
    const uint index = get_local_id(0);
    const uint count = get_local_size(0);
    __local uint command;
    uint seen = 0;

    if (index == 0) {
        atomic_store_explicit(
                ready, 1, memory_order_release,
                memory_scope_all_svm_devices);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (;;) {
        if (index == 0) {
            if (atomic_load_explicit(
                        stop, memory_order_acquire,
                        memory_scope_all_svm_devices) != 0) {
                command = 0xffffffffU;
            } else {
                command = atomic_load_explicit(
                        request_seq, memory_order_acquire,
                        memory_scope_all_svm_devices);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (command == 0xffffffffU) {
            break;
        }
        if (command != seen) {
            for (uint repeat = 0; repeat < repeats; ++repeat) {
                for (uint element = index;
                     element < elements; element += count) {
                    const float value = data[element];
                    data[element] = value * value;
                }
                barrier(CLK_GLOBAL_MEM_FENCE);
            }
            if (index == 0) {
                atomic_store_explicit(
                        done_seq, command, memory_order_release,
                        memory_scope_all_svm_devices);
            }
            seen = command;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
)CLC";

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr,
                "usage: %s <elements> <backend> <noop|sqr> "
                "<repeats> <requests>\n",
                argv[0]);
        return 2;
    }

    const int64_t     elements     = atoll(argv[1]);
    const std::string backend_name = argv[2];
    const std::string op_name      = argv[3];
    const int         repeats      = atoi(argv[4]);
    const int         max_requests = atoi(argv[5]);
    const bool        persistent   = backend_name == "OpenCLPersistent";
    const bool        dispatched   = backend_name == "OpenCLDoorbellDispatch";
    if (elements <= 0 || elements > (1 << 24) || (!persistent && !dispatched) || op_name != "sqr" || repeats <= 0 ||
        repeats > 256 || max_requests <= 0) {
        fprintf(stderr, "[persistent-worker] invalid configuration\n");
        return 2;
    }

    signal(SIGPIPE, SIG_IGN);

    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0) {
        fprintf(stderr, "[persistent-worker] no OpenCL platform\n");
        return 1;
    }
    std::vector<cl_platform_id> platforms(platform_count);
    clGetPlatformIDs(platform_count, platforms.data(), nullptr);

    cl_device_id device = nullptr;
    for (cl_platform_id platform : platforms) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count) != CL_SUCCESS ||
            device_count == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(device_count);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr);
        for (cl_device_id candidate : devices) {
            char name[256] = {};
            clGetDeviceInfo(candidate, CL_DEVICE_NAME, sizeof(name), name, nullptr);
            if (strstr(name, "Adreno") != nullptr) {
                device = candidate;
                break;
            }
        }
        if (device != nullptr) {
            break;
        }
    }
    if (device == nullptr) {
        fprintf(stderr, "[persistent-worker] no Adreno GPU\n");
        return 1;
    }

    cl_device_svm_capabilities svm_caps = 0;
    clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(svm_caps), &svm_caps, nullptr);
    const cl_device_svm_capabilities required_svm = CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_ATOMICS;
    if ((svm_caps & required_svm) != required_svm) {
        fprintf(stderr, "[persistent-worker] fine-grained SVM atomics unavailable\n");
        return 1;
    }

    cl_int     error   = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "[persistent-worker] context: %d\n", error);
        return 1;
    }
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "[persistent-worker] queue: %d\n", error);
        return 1;
    }

    const size_t source_size = strlen(persistent_source);
    cl_program   program     = clCreateProgramWithSource(context, 1, &persistent_source, &source_size, &error);
    if (error != CL_SUCCESS || clBuildProgram(program, 1, &device, "-cl-std=CL2.0", nullptr, nullptr) != CL_SUCCESS) {
        fprintf(stderr, "[persistent-worker] program build failed\n");
        print_build_log(program, device);
        return 1;
    }
    cl_kernel kernel = clCreateKernel(program, "persistent_sqr", &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "[persistent-worker] kernel: %d\n", error);
        return 1;
    }

    const cl_svm_mem_flags control_flags = CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS;
    const cl_svm_mem_flags data_flags    = CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER;
    uint32_t * control = static_cast<uint32_t *>(clSVMAlloc(context, control_flags, 4 * sizeof(uint32_t), 64));
    float *    shared_data =
        static_cast<float *>(clSVMAlloc(context, data_flags, static_cast<size_t>(elements) * sizeof(float), 128));
    if (control == nullptr || shared_data == nullptr) {
        fprintf(stderr, "[persistent-worker] SVM allocation failed\n");
        return 1;
    }
    memset(control, 0, 4 * sizeof(uint32_t));
    memset(shared_data, 0, static_cast<size_t>(elements) * sizeof(float));

    uint32_t *    request_seq = control;
    uint32_t *    done_seq    = control + 1;
    uint32_t *    stop        = control + 2;
    uint32_t *    ready       = control + 3;
    const cl_uint element_arg = static_cast<cl_uint>(elements);
    const cl_uint repeat_arg  = static_cast<cl_uint>(repeats);
    clSetKernelArgSVMPointer(kernel, 0, request_seq);
    clSetKernelArgSVMPointer(kernel, 1, done_seq);
    clSetKernelArgSVMPointer(kernel, 2, stop);
    clSetKernelArgSVMPointer(kernel, 3, ready);
    clSetKernelArgSVMPointer(kernel, 4, shared_data);
    clSetKernelArg(kernel, 5, sizeof(element_arg), &element_arg);
    clSetKernelArg(kernel, 6, sizeof(repeat_arg), &repeat_arg);

    const size_t work_items       = 256;
    cl_event     persistent_event = nullptr;
    uint64_t     launch_ready_ns  = 0;
    if (persistent) {
        const uint64_t launch_started = now_ns();
        error =
            clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &work_items, &work_items, 0, nullptr, &persistent_event);
        if (error != CL_SUCCESS || clFlush(queue) != CL_SUCCESS) {
            fprintf(stderr, "[persistent-worker] launch: %d\n", error);
            return 1;
        }
        const uint64_t ready_deadline = launch_started + 5000000000ULL;
        while (__atomic_load_n(ready, __ATOMIC_ACQUIRE) == 0 && now_ns() < ready_deadline) {
        }
        launch_ready_ns = now_ns() - launch_started;
        if (__atomic_load_n(ready, __ATOMIC_ACQUIRE) == 0) {
            fprintf(stderr, "[persistent-worker] launch readiness timeout\n");
            return 1;
        }
    }

    const size_t          encoded_bytes = static_cast<size_t>(elements) * sizeof(uint16_t);
    const size_t          decoded_bytes = static_cast<size_t>(elements) * sizeof(float);
    std::vector<uint8_t>  request_bytes(sizeof(s41_dummy_request) + encoded_bytes);
    std::vector<uint8_t>  response_bytes(sizeof(s41_dummy_response) + encoded_bytes);
    std::vector<float>    input_f32(static_cast<size_t>(elements));
    std::vector<float>    output_f32(static_cast<size_t>(elements));
    std::vector<uint16_t> output_f16(static_cast<size_t>(elements));

    fprintf(stderr,
            "[persistent-worker] ready backend=%s "
            "elements=%lld op=%s repeats=%d requests=%d "
            "request_bytes=%zu response_bytes=%zu launch_ready_us=%.1f "
            "svm_caps=0x%llx\n",
            backend_name.c_str(), static_cast<long long>(elements), op_name.c_str(), repeats, max_requests,
            request_bytes.size(), response_bytes.size(), launch_ready_ns / 1000.0,
            static_cast<unsigned long long>(svm_caps));
    fflush(stderr);

    uint64_t previous_write_ns = 0;
    int      served            = 0;
    for (;;) {
        const int descriptor = open("/dev/usb_accessory", O_RDWR);
        if (descriptor < 0) {
            fprintf(stderr, "[persistent-worker] endpoint open: %s\n", strerror(errno));
            sleep(1);
            continue;
        }
        fprintf(stderr, "[persistent-worker] endpoint open\n");
        fflush(stderr);

        while (served < max_requests && read_exact(descriptor, request_bytes.data(), request_bytes.size())) {
            const uint64_t    request_started = now_ns();
            uint64_t          started         = request_started;
            s41_dummy_request request         = {};
            memcpy(&request, request_bytes.data(), sizeof(request));
            const uint8_t * encoded_input = request_bytes.data() + sizeof(request);
            const bool      request_ok    = request.magic == S41_DUMMY_REQUEST_MAGIC &&
                                    request.version == S41_DUMMY_PROTOCOL_VERSION && request.reserved == 0 &&
                                    request.elements == static_cast<uint32_t>(elements) &&
                                    request.input_bytes == encoded_bytes &&
                                    request.input_crc32 == crc32_bytes(encoded_input, encoded_bytes) &&
                                    request.request_id != 0 && request.request_id != 0xffffffffU;
            const uint64_t validate_ns = now_ns() - started;
            if (!request_ok) {
                fprintf(stderr, "[persistent-worker] invalid request\n");
                close(descriptor);
                return 3;
            }

            started = now_ns();
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(encoded_input), input_f32.data(), elements);
            const uint64_t decode_ns = now_ns() - started;

            started = now_ns();
            memcpy(shared_data, input_f32.data(), decoded_bytes);
            const uint64_t set_ns = now_ns() - started;

            uint64_t submit_ns = 0;
            uint64_t sync_ns   = 0;
            if (persistent) {
                started = now_ns();
                __atomic_store_n(request_seq, request.request_id, __ATOMIC_RELEASE);
                submit_ns = now_ns() - started;

                started                         = now_ns();
                const uint64_t request_deadline = started + 4000000000ULL;
                while (__atomic_load_n(done_seq, __ATOMIC_ACQUIRE) != request.request_id &&
                       now_ns() < request_deadline) {
                }
                sync_ns = now_ns() - started;
                if (__atomic_load_n(done_seq, __ATOMIC_ACQUIRE) != request.request_id) {
                    fprintf(stderr, "[persistent-worker] request completion timeout\n");
                    close(descriptor);
                    return 3;
                }
            } else {
                __atomic_store_n(done_seq, 0U, __ATOMIC_RELAXED);
                __atomic_store_n(stop, 0U, __ATOMIC_RELAXED);
                __atomic_store_n(ready, 0U, __ATOMIC_RELAXED);
                __atomic_store_n(request_seq, request.request_id, __ATOMIC_RELEASE);

                cl_event dispatch_event = nullptr;
                started                 = now_ns();
                error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &work_items, &work_items, 0, nullptr,
                                               &dispatch_event);
                if (error == CL_SUCCESS) {
                    error = clFlush(queue);
                }
                submit_ns = now_ns() - started;
                if (error != CL_SUCCESS) {
                    fprintf(stderr, "[persistent-worker] dispatch launch: %d\n", error);
                    close(descriptor);
                    return 3;
                }

                started                         = now_ns();
                const uint64_t request_deadline = started + 4000000000ULL;
                while (__atomic_load_n(done_seq, __ATOMIC_ACQUIRE) != request.request_id &&
                       now_ns() < request_deadline) {
                }
                const bool completed = __atomic_load_n(done_seq, __ATOMIC_ACQUIRE) == request.request_id;
                __atomic_store_n(stop, 1U, __ATOMIC_RELEASE);
                const cl_int wait_status = clWaitForEvents(1, &dispatch_event);
                sync_ns                  = now_ns() - started;
                if (!completed || wait_status != CL_SUCCESS) {
                    fprintf(stderr, "[persistent-worker] dispatch completion failed: %d\n", wait_status);
                    close(descriptor);
                    return 3;
                }

                cl_ulong queued         = 0;
                cl_ulong submitted      = 0;
                cl_ulong device_started = 0;
                cl_ulong device_ended   = 0;
                clGetEventProfilingInfo(dispatch_event, CL_PROFILING_COMMAND_QUEUED, sizeof(queued), &queued, nullptr);
                clGetEventProfilingInfo(dispatch_event, CL_PROFILING_COMMAND_SUBMIT, sizeof(submitted), &submitted,
                                        nullptr);
                clGetEventProfilingInfo(dispatch_event, CL_PROFILING_COMMAND_START, sizeof(device_started),
                                        &device_started, nullptr);
                clGetEventProfilingInfo(dispatch_event, CL_PROFILING_COMMAND_END, sizeof(device_ended), &device_ended,
                                        nullptr);
                fprintf(stderr,
                        "[dispatch-profile] request=%u queued_us=%.1f "
                        "start_us=%.1f exec_us=%.1f\n",
                        request.request_id, (submitted - queued) / 1000.0, (device_started - submitted) / 1000.0,
                        (device_ended - device_started) / 1000.0);
                clReleaseEvent(dispatch_event);
            }

            started = now_ns();
            memcpy(output_f32.data(), shared_data, decoded_bytes);
            const uint64_t get_ns = now_ns() - started;

            started = now_ns();
            ggml_fp32_to_fp16_row(output_f32.data(), reinterpret_cast<ggml_fp16_t *>(output_f16.data()), elements);
            const uint32_t output_crc     = crc32_bytes(output_f16.data(), encoded_bytes);
            const uint64_t encode_hash_ns = now_ns() - started;
            const uint64_t prewrite_ns    = now_ns() - request_started;

            s41_dummy_response response = {};
            response.magic              = S41_DUMMY_RESPONSE_MAGIC;
            response.version            = S41_DUMMY_PROTOCOL_VERSION;
            response.request_id         = request.request_id;
            response.elements           = static_cast<uint32_t>(elements);
            response.output_bytes       = static_cast<uint32_t>(encoded_bytes);
            response.output_crc32       = output_crc;
            response.validate_ns        = validate_ns;
            response.decode_ns          = decode_ns;
            response.set_ns             = set_ns;
            response.submit_ns          = submit_ns;
            response.sync_ns            = sync_ns;
            response.get_ns             = get_ns;
            response.encode_hash_ns     = encode_hash_ns;
            response.prewrite_ns        = prewrite_ns;
            response.previous_write_ns  = previous_write_ns;
            memcpy(response_bytes.data(), &response, sizeof(response));
            memcpy(response_bytes.data() + sizeof(response), output_f16.data(), encoded_bytes);

            started = now_ns();
            if (!write_exact(descriptor, response_bytes.data(), response_bytes.size())) {
                break;
            }
            previous_write_ns = now_ns() - started;
            ++served;
        }
        close(descriptor);
        if (served >= max_requests) {
            break;
        }
    }

    if (persistent) {
        __atomic_store_n(stop, 1U, __ATOMIC_RELEASE);
        error                   = clWaitForEvents(1, &persistent_event);
        cl_ulong queued         = 0;
        cl_ulong submitted      = 0;
        cl_ulong device_started = 0;
        cl_ulong device_ended   = 0;
        clGetEventProfilingInfo(persistent_event, CL_PROFILING_COMMAND_QUEUED, sizeof(queued), &queued, nullptr);
        clGetEventProfilingInfo(persistent_event, CL_PROFILING_COMMAND_SUBMIT, sizeof(submitted), &submitted, nullptr);
        clGetEventProfilingInfo(persistent_event, CL_PROFILING_COMMAND_START, sizeof(device_started), &device_started,
                                nullptr);
        clGetEventProfilingInfo(persistent_event, CL_PROFILING_COMMAND_END, sizeof(device_ended), &device_ended,
                                nullptr);
        fprintf(stderr,
                "[persistent-worker] complete requests=%d wait_status=%d "
                "launch_queue_us=%.1f launch_start_us=%.1f resident_ms=%.3f\n",
                served, error, (submitted - queued) / 1000.0, (device_started - submitted) / 1000.0,
                (device_ended - device_started) / 1000000.0);
        clReleaseEvent(persistent_event);
    } else {
        fprintf(stderr, "[persistent-worker] complete requests=%d wait_status=0\n", served);
    }
    fflush(stderr);

    clSVMFree(context, shared_data);
    clSVMFree(context, control);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return error == CL_SUCCESS ? 0 : 3;
}
