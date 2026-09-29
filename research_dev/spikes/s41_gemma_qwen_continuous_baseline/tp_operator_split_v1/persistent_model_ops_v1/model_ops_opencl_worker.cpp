// Direct OpenCL controls for Qwen3-14B-shaped phone operators.
//
// usage:
//   model_ops_opencl_worker <rmsnorm|swiglu|attention>
//       <OpenCLDispatch|OpenCLPersistent> <n_kv> <requests>

#define CL_TARGET_OPENCL_VERSION 300

#include "model_ops_protocol.h"

#include "ggml.h"

#include <CL/cl.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <signal.h>
#include <string>
#include <unistd.h>
#include <vector>
#include <zlib.h>

static constexpr uint32_t S41_MODEL_OP_GROUPS = S41_QWEN3_14B_GQA;
static constexpr size_t S41_MODEL_OP_LOCAL_SIZE = S41_QWEN3_14B_HEAD_DIM;

enum s41_control_index : uint32_t {
    S41_CONTROL_REQUEST_SEQ = 0,
    S41_CONTROL_DONE_SEQ = 1,
    S41_CONTROL_STOP = 2,
    S41_CONTROL_READY_GROUPS = 3,
    S41_CONTROL_COMPLETE_GROUPS = 4,
    S41_CONTROL_OPCODE = 5,
    S41_CONTROL_COUNT = 6,
};

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
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL +
            static_cast<uint64_t>(value.tv_nsec);
}

static uint32_t crc32_bytes(const void * data, size_t size) {
    return static_cast<uint32_t>(crc32(
            0, static_cast<const Bytef *>(data), static_cast<uInt>(size)));
}

static bool parse_op(const std::string & name, s41_model_op & op) {
    if (name == "rmsnorm") {
        op = S41_MODEL_OP_RMSNORM;
        return true;
    }
    if (name == "swiglu") {
        op = S41_MODEL_OP_SWIGLU;
        return true;
    }
    if (name == "attention") {
        op = S41_MODEL_OP_ATTENTION;
        return true;
    }
    return false;
}

static void print_build_log(cl_program program, cl_device_id device) {
    size_t size = 0;
    clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
    std::vector<char> log(size + 1, 0);
    clGetProgramBuildInfo(
            program, device, CL_PROGRAM_BUILD_LOG, size, log.data(), nullptr);
    fprintf(stderr, "%s\n", log.data());
}

static const char * model_ops_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define OP_RMSNORM 1U
#define OP_SWIGLU 2U
#define OP_ATTENTION 3U
#define HIDDEN 5120U
#define INTERMEDIATE 17408U
#define HEAD_DIM 128U
#define GROUPS 5U

__kernel void model_ops_loop(
        volatile __global atomic_uint * request_seq,
        volatile __global atomic_uint * done_seq,
        volatile __global atomic_uint * stop,
        volatile __global atomic_uint * ready_groups,
        volatile __global atomic_uint * complete_groups,
        volatile __global atomic_uint * opcode,
        __global const half * input,
        __global half * output,
        __global const half * norm_weight,
        __global const half * key,
        __global const half * value,
        __global float * scores,
        const uint n_kv) {
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);
    __local uint command;
    __local uint selected_op;
    __local float scratch[HEAD_DIM];
    uint seen = 0;

    if (lid == 0) {
        atomic_fetch_add_explicit(
                ready_groups, 1U, memory_order_release,
                memory_scope_all_svm_devices);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (;;) {
        if (lid == 0) {
            if (atomic_load_explicit(
                        stop, memory_order_acquire,
                        memory_scope_all_svm_devices) != 0U) {
                command = 0xffffffffU;
            } else {
                command = atomic_load_explicit(
                        request_seq, memory_order_acquire,
                        memory_scope_all_svm_devices);
                selected_op = atomic_load_explicit(
                        opcode, memory_order_relaxed,
                        memory_scope_all_svm_devices);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (command == 0xffffffffU) {
            break;
        }
        if (command == 0U || command == seen) {
            continue;
        }

        if (selected_op == OP_RMSNORM) {
            if (group == 0U) {
                float sum = 0.0f;
                for (uint index = lid; index < HIDDEN; index += HEAD_DIM) {
                    const float item = convert_float(input[index]);
                    sum = fma(item, item, sum);
                }
                scratch[lid] = sum;
                barrier(CLK_LOCAL_MEM_FENCE);
                for (uint stride = HEAD_DIM / 2U; stride > 0U;
                     stride >>= 1U) {
                    if (lid < stride) {
                        scratch[lid] += scratch[lid + stride];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                const float scale = native_rsqrt(
                        scratch[0] * (1.0f / (float) HIDDEN) + 1.0e-6f);
                for (uint index = lid; index < HIDDEN; index += HEAD_DIM) {
                    output[index] = convert_half_rte(
                            convert_float(input[index]) * scale *
                            convert_float(norm_weight[index]));
                }
            }
        } else if (selected_op == OP_SWIGLU) {
            const uint global_lane = group * HEAD_DIM + lid;
            const uint global_stride = GROUPS * HEAD_DIM;
            for (uint index = global_lane; index < INTERMEDIATE;
                 index += global_stride) {
                const float gate = convert_float(input[index]);
                const float up = convert_float(input[INTERMEDIATE + index]);
                const float silu = gate / (1.0f + native_exp(-gate));
                output[index] = convert_half_rte(silu * up);
            }
        } else if (selected_op == OP_ATTENTION) {
            const uint head = group;
            const uint query_base = head * HEAD_DIM;
            const uint score_base = head * n_kv;
            float local_max = -INFINITY;
            for (uint token = lid; token < n_kv; token += HEAD_DIM) {
                const uint cache_base = token * HEAD_DIM;
                float dot_product = 0.0f;
                for (uint dimension = 0; dimension < HEAD_DIM;
                     dimension += 4U) {
                    const float4 query = convert_float4(vload4(
                            0, input + query_base + dimension));
                    const float4 key_value = convert_float4(vload4(
                            0, key + cache_base + dimension));
                    dot_product += dot(query, key_value);
                }
                dot_product *= 0.08838834764831845f;
                scores[score_base + token] = dot_product;
                local_max = fmax(local_max, dot_product);
            }
            scratch[lid] = local_max;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint stride = HEAD_DIM / 2U; stride > 0U;
                 stride >>= 1U) {
                if (lid < stride) {
                    scratch[lid] = fmax(
                            scratch[lid], scratch[lid + stride]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            const float maximum = scratch[0];

            float local_sum = 0.0f;
            for (uint token = lid; token < n_kv; token += HEAD_DIM) {
                const float probability = native_exp(
                        scores[score_base + token] - maximum);
                scores[score_base + token] = probability;
                local_sum += probability;
            }
            scratch[lid] = local_sum;
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint stride = HEAD_DIM / 2U; stride > 0U;
                 stride >>= 1U) {
                if (lid < stride) {
                    scratch[lid] += scratch[lid + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            const float inverse_sum = 1.0f / scratch[0];

            float result = 0.0f;
            for (uint token = 0; token < n_kv; ++token) {
                result = fma(
                        scores[score_base + token],
                        convert_float(value[token * HEAD_DIM + lid]), result);
            }
            output[query_base + lid] = convert_half_rte(
                    result * inverse_sum);
        }

        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
        if (lid == 0) {
            atomic_work_item_fence(
                    CLK_GLOBAL_MEM_FENCE, memory_order_release,
                    memory_scope_all_svm_devices);
            const uint completed = atomic_fetch_add_explicit(
                    complete_groups, 1U, memory_order_acq_rel,
                    memory_scope_all_svm_devices) + 1U;
            if (completed == GROUPS) {
                atomic_store_explicit(
                        done_seq, command, memory_order_release,
                        memory_scope_all_svm_devices);
            }
        }
        seen = command;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
)CLC";

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr,
                "usage: %s <rmsnorm|swiglu|attention> "
                "<OpenCLDispatch|OpenCLPersistent> <n_kv> <requests>\n",
                argv[0]);
        return 2;
    }

    s41_model_op op = S41_MODEL_OP_RMSNORM;
    const std::string op_name = argv[1];
    const std::string mode_name = argv[2];
    const int64_t n_kv = atoll(argv[3]);
    const int max_requests = atoi(argv[4]);
    const bool persistent = mode_name == "OpenCLPersistent";
    const bool dispatched = mode_name == "OpenCLDispatch";
    if (!parse_op(op_name, op) || (!persistent && !dispatched) ||
        n_kv <= 0 || n_kv > (1 << 20) || max_requests <= 0) {
        fprintf(stderr, "[model-opencl] invalid configuration\n");
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);

    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, nullptr, &platform_count) != CL_SUCCESS ||
        platform_count == 0) {
        fprintf(stderr, "[model-opencl] no OpenCL platform\n");
        return 1;
    }
    std::vector<cl_platform_id> platforms(platform_count);
    clGetPlatformIDs(platform_count, platforms.data(), nullptr);
    cl_device_id device = nullptr;
    for (cl_platform_id platform : platforms) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(
                    platform, CL_DEVICE_TYPE_GPU, 0, nullptr,
                    &device_count) != CL_SUCCESS || device_count == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(device_count);
        clGetDeviceIDs(
                platform, CL_DEVICE_TYPE_GPU, device_count,
                devices.data(), nullptr);
        for (cl_device_id candidate : devices) {
            char name[256] = {};
            clGetDeviceInfo(
                    candidate, CL_DEVICE_NAME, sizeof(name), name, nullptr);
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
        fprintf(stderr, "[model-opencl] no Adreno GPU\n");
        return 1;
    }

    cl_device_svm_capabilities svm_caps = 0;
    clGetDeviceInfo(
            device, CL_DEVICE_SVM_CAPABILITIES,
            sizeof(svm_caps), &svm_caps, nullptr);
    const cl_device_svm_capabilities required_svm =
            CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_ATOMICS;
    if ((svm_caps & required_svm) != required_svm) {
        fprintf(stderr,
                "[model-opencl] fine-grained SVM atomics unavailable\n");
        return 1;
    }

    cl_int error = CL_SUCCESS;
    cl_context context = clCreateContext(
            nullptr, 1, &device, nullptr, nullptr, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "[model-opencl] context: %d\n", error);
        return 1;
    }
    cl_command_queue queue = clCreateCommandQueue(
            context, device, CL_QUEUE_PROFILING_ENABLE, &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "[model-opencl] queue: %d\n", error);
        return 1;
    }
    const size_t source_size = strlen(model_ops_source);
    cl_program program = clCreateProgramWithSource(
            context, 1, &model_ops_source, &source_size, &error);
    if (error != CL_SUCCESS ||
        clBuildProgram(
                program, 1, &device,
                "-cl-std=CL2.0 -cl-fast-relaxed-math -cl-mad-enable",
                nullptr, nullptr) != CL_SUCCESS) {
        fprintf(stderr, "[model-opencl] program build failed\n");
        print_build_log(program, device);
        return 1;
    }
    cl_kernel kernel = clCreateKernel(program, "model_ops_loop", &error);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "[model-opencl] kernel: %d\n", error);
        return 1;
    }

    const uint32_t input_elements = s41_model_op_input_elements(op);
    const uint32_t output_elements = s41_model_op_output_elements(op);
    const size_t input_bytes = static_cast<size_t>(input_elements) *
            sizeof(ggml_fp16_t);
    const size_t output_bytes = static_cast<size_t>(output_elements) *
            sizeof(ggml_fp16_t);
    const size_t cache_elements = op == S41_MODEL_OP_ATTENTION
            ? static_cast<size_t>(n_kv) * S41_QWEN3_14B_HEAD_DIM : 1;
    const size_t score_elements = op == S41_MODEL_OP_ATTENTION
            ? static_cast<size_t>(n_kv) * S41_QWEN3_14B_GQA : 1;
    const cl_svm_mem_flags control_flags = CL_MEM_READ_WRITE |
            CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS;
    const cl_svm_mem_flags data_flags =
            CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER;
    uint32_t * control = static_cast<uint32_t *>(clSVMAlloc(
            context, control_flags,
            S41_CONTROL_COUNT * sizeof(uint32_t), 64));
    ggml_fp16_t * shared_input = static_cast<ggml_fp16_t *>(clSVMAlloc(
            context, data_flags, input_bytes, 128));
    ggml_fp16_t * shared_output = static_cast<ggml_fp16_t *>(clSVMAlloc(
            context, data_flags, output_bytes, 128));
    ggml_fp16_t * norm_weight = static_cast<ggml_fp16_t *>(clSVMAlloc(
            context, data_flags,
            S41_QWEN3_14B_HIDDEN * sizeof(ggml_fp16_t), 128));
    ggml_fp16_t * key = static_cast<ggml_fp16_t *>(clSVMAlloc(
            context, data_flags,
            cache_elements * sizeof(ggml_fp16_t), 128));
    ggml_fp16_t * value = static_cast<ggml_fp16_t *>(clSVMAlloc(
            context, data_flags,
            cache_elements * sizeof(ggml_fp16_t), 128));
    float * scores = static_cast<float *>(clSVMAlloc(
            context, data_flags, score_elements * sizeof(float), 128));
    if (control == nullptr || shared_input == nullptr ||
        shared_output == nullptr || norm_weight == nullptr ||
        key == nullptr || value == nullptr || scores == nullptr) {
        fprintf(stderr, "[model-opencl] SVM allocation failed\n");
        return 1;
    }
    memset(control, 0, S41_CONTROL_COUNT * sizeof(uint32_t));
    memset(shared_input, 0, input_bytes);
    memset(shared_output, 0, output_bytes);
    memset(scores, 0, score_elements * sizeof(float));

    std::vector<float> initialize_f32(std::max(
            static_cast<size_t>(S41_QWEN3_14B_HIDDEN), cache_elements));
    for (uint32_t index = 0; index < S41_QWEN3_14B_HIDDEN; ++index) {
        initialize_f32[index] = s41_model_norm_weight(index);
    }
    ggml_fp32_to_fp16_row(
            initialize_f32.data(), norm_weight, S41_QWEN3_14B_HIDDEN);
    if (op == S41_MODEL_OP_ATTENTION) {
        for (int64_t token = 0; token < n_kv; ++token) {
            for (uint32_t dimension = 0;
                 dimension < S41_QWEN3_14B_HEAD_DIM; ++dimension) {
                const size_t index = static_cast<size_t>(token) *
                        S41_QWEN3_14B_HEAD_DIM + dimension;
                initialize_f32[index] = s41_model_key_value(
                        static_cast<uint32_t>(token), dimension);
            }
        }
        ggml_fp32_to_fp16_row(
                initialize_f32.data(), key, cache_elements);
        for (int64_t token = 0; token < n_kv; ++token) {
            for (uint32_t dimension = 0;
                 dimension < S41_QWEN3_14B_HEAD_DIM; ++dimension) {
                const size_t index = static_cast<size_t>(token) *
                        S41_QWEN3_14B_HEAD_DIM + dimension;
                initialize_f32[index] = s41_model_value_value(
                        static_cast<uint32_t>(token), dimension);
            }
        }
        ggml_fp32_to_fp16_row(
                initialize_f32.data(), value, cache_elements);
    }

    clSetKernelArgSVMPointer(
            kernel, 0, control + S41_CONTROL_REQUEST_SEQ);
    clSetKernelArgSVMPointer(
            kernel, 1, control + S41_CONTROL_DONE_SEQ);
    clSetKernelArgSVMPointer(kernel, 2, control + S41_CONTROL_STOP);
    clSetKernelArgSVMPointer(
            kernel, 3, control + S41_CONTROL_READY_GROUPS);
    clSetKernelArgSVMPointer(
            kernel, 4, control + S41_CONTROL_COMPLETE_GROUPS);
    clSetKernelArgSVMPointer(kernel, 5, control + S41_CONTROL_OPCODE);
    clSetKernelArgSVMPointer(kernel, 6, shared_input);
    clSetKernelArgSVMPointer(kernel, 7, shared_output);
    clSetKernelArgSVMPointer(kernel, 8, norm_weight);
    clSetKernelArgSVMPointer(kernel, 9, key);
    clSetKernelArgSVMPointer(kernel, 10, value);
    clSetKernelArgSVMPointer(kernel, 11, scores);
    const cl_uint n_kv_arg = static_cast<cl_uint>(n_kv);
    clSetKernelArg(kernel, 12, sizeof(n_kv_arg), &n_kv_arg);

    const size_t global_size =
            S41_MODEL_OP_GROUPS * S41_MODEL_OP_LOCAL_SIZE;
    const size_t local_size = S41_MODEL_OP_LOCAL_SIZE;
    cl_event persistent_event = nullptr;
    uint64_t launch_ready_ns = 0;
    if (persistent) {
        const uint64_t launch_started = now_ns();
        error = clEnqueueNDRangeKernel(
                queue, kernel, 1, nullptr, &global_size, &local_size,
                0, nullptr, &persistent_event);
        if (error != CL_SUCCESS || clFlush(queue) != CL_SUCCESS) {
            fprintf(stderr, "[model-opencl] launch: %d\n", error);
            return 1;
        }
        const uint64_t deadline = launch_started + 5000000000ULL;
        while (__atomic_load_n(
                       control + S41_CONTROL_READY_GROUPS,
                       __ATOMIC_ACQUIRE) != S41_MODEL_OP_GROUPS &&
               now_ns() < deadline) {
        }
        launch_ready_ns = now_ns() - launch_started;
        if (__atomic_load_n(
                    control + S41_CONTROL_READY_GROUPS,
                    __ATOMIC_ACQUIRE) != S41_MODEL_OP_GROUPS) {
            fprintf(stderr, "[model-opencl] launch readiness timeout\n");
            return 1;
        }
    }

    std::vector<uint8_t> request_bytes(
            sizeof(s41_model_op_request) + input_bytes);
    std::vector<uint8_t> response_bytes(
            sizeof(s41_model_op_response) + output_bytes);
    std::vector<ggml_fp16_t> output_copy(output_elements);

    fprintf(stderr,
            "[model-opencl] ready op=%s mode=%s n_kv=%lld "
            "groups=%u local=%zu input_elements=%u output_elements=%u "
            "requests=%d launch_ready_us=%.1f svm_caps=0x%llx\n",
            op_name.c_str(), mode_name.c_str(),
            static_cast<long long>(n_kv), S41_MODEL_OP_GROUPS, local_size,
            input_elements, output_elements, max_requests,
            launch_ready_ns / 1000.0,
            static_cast<unsigned long long>(svm_caps));
    fflush(stderr);

    uint64_t previous_write_ns = 0;
    int served = 0;
    for (;;) {
        const int descriptor = open("/dev/usb_accessory", O_RDWR);
        if (descriptor < 0) {
            fprintf(stderr, "[model-opencl] endpoint open: %s\n",
                    strerror(errno));
            sleep(1);
            continue;
        }
        fprintf(stderr, "[model-opencl] endpoint open\n");
        fflush(stderr);

        while (served < max_requests && read_exact(
                    descriptor, request_bytes.data(), request_bytes.size())) {
            const uint64_t request_started = now_ns();
            uint64_t started = request_started;
            s41_model_op_request request = {};
            memcpy(&request, request_bytes.data(), sizeof(request));
            const uint8_t * encoded_input =
                    request_bytes.data() + sizeof(request);
            const bool request_ok =
                    request.magic == S41_MODEL_OP_REQUEST_MAGIC &&
                    request.version == S41_MODEL_OP_PROTOCOL_VERSION &&
                    request.op == static_cast<uint16_t>(op) &&
                    request.request_id != 0 &&
                    request.request_id != 0xffffffffU &&
                    request.input_elements == input_elements &&
                    request.output_elements == output_elements &&
                    request.input_bytes == input_bytes &&
                    request.input_crc32 ==
                            crc32_bytes(encoded_input, input_bytes);
            const uint64_t validate_ns = now_ns() - started;
            if (!request_ok) {
                fprintf(stderr, "[model-opencl] invalid request\n");
                close(descriptor);
                return 3;
            }

            const uint64_t decode_ns = 0;
            started = now_ns();
            memcpy(shared_input, encoded_input, input_bytes);
            const uint64_t set_ns = now_ns() - started;

            __atomic_store_n(
                    control + S41_CONTROL_DONE_SEQ, 0U, __ATOMIC_RELAXED);
            __atomic_store_n(
                    control + S41_CONTROL_COMPLETE_GROUPS, 0U,
                    __ATOMIC_RELAXED);
            __atomic_store_n(
                    control + S41_CONTROL_OPCODE,
                    static_cast<uint32_t>(op), __ATOMIC_RELAXED);

            uint64_t submit_ns = 0;
            uint64_t sync_ns = 0;
            cl_event dispatch_event = nullptr;
            if (persistent) {
                started = now_ns();
                __atomic_store_n(
                        control + S41_CONTROL_REQUEST_SEQ,
                        request.request_id, __ATOMIC_RELEASE);
                submit_ns = now_ns() - started;
            } else {
                __atomic_store_n(
                        control + S41_CONTROL_STOP, 0U, __ATOMIC_RELAXED);
                __atomic_store_n(
                        control + S41_CONTROL_READY_GROUPS, 0U,
                        __ATOMIC_RELAXED);
                __atomic_store_n(
                        control + S41_CONTROL_REQUEST_SEQ,
                        request.request_id, __ATOMIC_RELEASE);
                started = now_ns();
                error = clEnqueueNDRangeKernel(
                        queue, kernel, 1, nullptr, &global_size, &local_size,
                        0, nullptr, &dispatch_event);
                if (error == CL_SUCCESS) {
                    error = clFlush(queue);
                }
                submit_ns = now_ns() - started;
                if (error != CL_SUCCESS) {
                    fprintf(stderr,
                            "[model-opencl] dispatch launch: %d\n", error);
                    close(descriptor);
                    return 3;
                }
            }

            started = now_ns();
            const uint64_t deadline = started + 10000000000ULL;
            while (__atomic_load_n(
                           control + S41_CONTROL_DONE_SEQ,
                           __ATOMIC_ACQUIRE) != request.request_id &&
                   now_ns() < deadline) {
            }
            const bool completed = __atomic_load_n(
                    control + S41_CONTROL_DONE_SEQ,
                    __ATOMIC_ACQUIRE) == request.request_id;
            if (!persistent) {
                __atomic_store_n(
                        control + S41_CONTROL_STOP, 1U, __ATOMIC_RELEASE);
                const cl_int wait_status =
                        clWaitForEvents(1, &dispatch_event);
                if (wait_status != CL_SUCCESS) {
                    fprintf(stderr,
                            "[model-opencl] dispatch wait: %d\n",
                            wait_status);
                    close(descriptor);
                    return 3;
                }
            }
            sync_ns = now_ns() - started;
            if (!completed) {
                fprintf(stderr, "[model-opencl] completion timeout\n");
                close(descriptor);
                return 3;
            }

            if (dispatch_event != nullptr) {
                cl_ulong queued = 0;
                cl_ulong submitted = 0;
                cl_ulong device_started = 0;
                cl_ulong device_ended = 0;
                clGetEventProfilingInfo(
                        dispatch_event, CL_PROFILING_COMMAND_QUEUED,
                        sizeof(queued), &queued, nullptr);
                clGetEventProfilingInfo(
                        dispatch_event, CL_PROFILING_COMMAND_SUBMIT,
                        sizeof(submitted), &submitted, nullptr);
                clGetEventProfilingInfo(
                        dispatch_event, CL_PROFILING_COMMAND_START,
                        sizeof(device_started), &device_started, nullptr);
                clGetEventProfilingInfo(
                        dispatch_event, CL_PROFILING_COMMAND_END,
                        sizeof(device_ended), &device_ended, nullptr);
                fprintf(stderr,
                        "[model-dispatch-profile] request=%u "
                        "queued_us=%.1f start_us=%.1f event_us=%.1f\n",
                        request.request_id,
                        (submitted - queued) / 1000.0,
                        (device_started - submitted) / 1000.0,
                        (device_ended - device_started) / 1000.0);
                clReleaseEvent(dispatch_event);
            }

            started = now_ns();
            memcpy(output_copy.data(), shared_output, output_bytes);
            const uint64_t get_ns = now_ns() - started;

            started = now_ns();
            const uint32_t output_crc =
                    crc32_bytes(output_copy.data(), output_bytes);
            const uint64_t encode_hash_ns = now_ns() - started;
            const uint64_t prewrite_ns = now_ns() - request_started;

            s41_model_op_response response = {};
            response.magic = S41_MODEL_OP_RESPONSE_MAGIC;
            response.version = S41_MODEL_OP_PROTOCOL_VERSION;
            response.request_id = request.request_id;
            response.op = static_cast<uint32_t>(op);
            response.output_elements = output_elements;
            response.output_bytes = output_bytes;
            response.output_crc32 = output_crc;
            response.validate_ns = validate_ns;
            response.decode_ns = decode_ns;
            response.set_ns = set_ns;
            response.submit_ns = submit_ns;
            response.sync_ns = sync_ns;
            response.get_ns = get_ns;
            response.encode_hash_ns = encode_hash_ns;
            response.prewrite_ns = prewrite_ns;
            response.previous_write_ns = previous_write_ns;
            memcpy(response_bytes.data(), &response, sizeof(response));
            memcpy(response_bytes.data() + sizeof(response),
                    output_copy.data(), output_bytes);

            started = now_ns();
            if (!write_exact(
                        descriptor, response_bytes.data(),
                        response_bytes.size())) {
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
        __atomic_store_n(
                control + S41_CONTROL_STOP, 1U, __ATOMIC_RELEASE);
        error = clWaitForEvents(1, &persistent_event);
        cl_ulong queued = 0;
        cl_ulong submitted = 0;
        cl_ulong device_started = 0;
        cl_ulong device_ended = 0;
        clGetEventProfilingInfo(
                persistent_event, CL_PROFILING_COMMAND_QUEUED,
                sizeof(queued), &queued, nullptr);
        clGetEventProfilingInfo(
                persistent_event, CL_PROFILING_COMMAND_SUBMIT,
                sizeof(submitted), &submitted, nullptr);
        clGetEventProfilingInfo(
                persistent_event, CL_PROFILING_COMMAND_START,
                sizeof(device_started), &device_started, nullptr);
        clGetEventProfilingInfo(
                persistent_event, CL_PROFILING_COMMAND_END,
                sizeof(device_ended), &device_ended, nullptr);
        fprintf(stderr,
                "[model-opencl] complete requests=%d status=%d "
                "launch_queue_us=%.1f launch_start_us=%.1f "
                "resident_ms=%.3f\n",
                served, error, (submitted - queued) / 1000.0,
                (device_started - submitted) / 1000.0,
                (device_ended - device_started) / 1000000.0);
        clReleaseEvent(persistent_event);
    } else {
        fprintf(stderr,
                "[model-opencl] complete requests=%d status=0\n", served);
    }
    fflush(stderr);

    clSVMFree(context, scores);
    clSVMFree(context, value);
    clSVMFree(context, key);
    clSVMFree(context, norm_weight);
    clSVMFree(context, shared_output);
    clSVMFree(context, shared_input);
    clSVMFree(context, control);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return error == CL_SUCCESS ? 0 : 3;
}
