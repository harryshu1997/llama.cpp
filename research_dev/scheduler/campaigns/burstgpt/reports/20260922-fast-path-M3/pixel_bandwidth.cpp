#include <vulkan/vulkan.h>
#include <arm_neon.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

static void require(bool ok, const char * msg) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", msg); std::exit(2); }
}
static void vkcheck(VkResult r) {
    if (r != VK_SUCCESS) { std::fprintf(stderr, "FAIL Vulkan %d\n", int(r)); std::exit(3); }
}
static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static uint32_t pattern(uint32_t x) {
    x += 0x9e3779b9u; x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
static uint32_t checksum(const uint32_t * p, size_t n, size_t prefetch) {
    uint32x4_t a = vdupq_n_u32(0), b = a, c = a, d = a;
    for (size_t i = 0; i < n; i += 16) {
        if (prefetch && i + prefetch < n) __builtin_prefetch(p + i + prefetch, 0, 3);
        a = veorq_u32(a, vld1q_u32(p + i)); b = veorq_u32(b, vld1q_u32(p + i + 4));
        c = veorq_u32(c, vld1q_u32(p + i + 8)); d = veorq_u32(d, vld1q_u32(p + i + 12));
    }
    a = veorq_u32(veorq_u32(a, b), veorq_u32(c, d));
    return vgetq_lane_u32(a, 0) ^ vgetq_lane_u32(a, 1) ^ vgetq_lane_u32(a, 2) ^ vgetq_lane_u32(a, 3);
}
struct CPU {
    struct alignas(64) Result { uint32_t got = 0, expected = 0; size_t words_read = 0; };
    size_t bytes, words, prefetch, chunk_words;
    uint32_t * data;
    int count;
    bool stop = false;
    pthread_barrier_t begin, end;
    std::vector<Result> results;
    std::vector<uint32_t> chunk_got, chunk_expected;
    std::atomic<size_t> next_chunk{0};
    std::vector<std::thread> threads;
    CPU(size_t size, int nt, unsigned mask, size_t pf, size_t chunk) : bytes(size), words(size / 4), prefetch(pf / 4), chunk_words(chunk / 4), count(nt), results(nt) {
        require(nt > 0 && nt <= 8 && size % 64 == 0, "CPU configuration");
        require(!chunk || (chunk % 64 == 0 && size % chunk == 0), "CPU chunk geometry");
        data = static_cast<uint32_t *>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        require(data != MAP_FAILED, "CPU allocation");
        for (size_t i = 0; i < words; ++i) data[i] = pattern(uint32_t(i));
        if (chunk_words) {
            chunk_got.resize(words / chunk_words, 0); chunk_expected.resize(chunk_got.size(), 0);
            for (size_t i = 0; i < words; ++i) chunk_expected[i / chunk_words] ^= data[i];
        }
        require(pthread_barrier_init(&begin, nullptr, nt + 1) == 0, "CPU begin barrier");
        require(pthread_barrier_init(&end, nullptr, nt + 1) == 0, "CPU end barrier");
        std::vector<int> cores;
        for (int i = 0; i < 8; ++i) if (mask & (1u << i)) cores.push_back(i);
        require(!mask || int(cores.size()) >= nt, "CPU affinity capacity");
        for (int t = 0; t < nt; ++t) {
            size_t first = (words / 16 * size_t(t) / nt) * 16;
            size_t last = (words / 16 * size_t(t + 1) / nt) * 16;
            for (size_t i = first; i < last; ++i) results[t].expected ^= data[i];
            threads.emplace_back([this, t, first, last, cores]() {
                if (!cores.empty()) {
                    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cores[t], &set);
                    require(sched_setaffinity(0, sizeof(set), &set) == 0, "CPU affinity");
                }
                for (;;) {
                    pthread_barrier_wait(&begin);
                    if (stop) break;
                    results[t].words_read = 0;
                    if (chunk_words) {
                        for (;;) {
                            size_t c = next_chunk.fetch_add(1, std::memory_order_relaxed);
                            if (c >= chunk_got.size()) break;
                            chunk_got[c] = checksum(data + c * chunk_words, chunk_words, prefetch);
                            results[t].words_read += chunk_words;
                        }
                    } else {
                        results[t].got = checksum(data + first, last - first, prefetch);
                        results[t].words_read = last - first;
                    }
                    pthread_barrier_wait(&end);
                }
            });
        }
    }
    void step() { next_chunk.store(0, std::memory_order_relaxed); pthread_barrier_wait(&begin); pthread_barrier_wait(&end); }
    void verify() {
        size_t total = 0; for (auto & r : results) total += r.words_read;
        require(total == words, "CPU read coverage");
        if (chunk_words) require(chunk_got == chunk_expected, "CPU every-chunk checksum");
        else for (auto & r : results) require(r.got == r.expected, "CPU checksum");
    }
    void print() {
        std::printf("{\"type\":\"cpu_partition\",\"chunk_bytes\":%zu,\"last_pass_thread_bytes\":[", chunk_words * 4);
        for (int t = 0; t < count; ++t) std::printf("%s%zu", t ? "," : "", results[t].words_read * 4);
        std::puts("]}");
    }
    ~CPU() {
        stop = true; pthread_barrier_wait(&begin);
        for (auto & t : threads) t.join();
        pthread_barrier_destroy(&begin); pthread_barrier_destroy(&end); munmap(data, bytes);
    }
};
struct GPU {
    struct Buffer { VkBuffer buffer; VkDeviceMemory memory; void * ptr; uint32_t memory_type; };
    VkInstance instance;
    VkPhysicalDevice physical;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem;
    VkDevice device;
    VkQueue queue;
    VkCommandPool pool;
    VkCommandBuffer command;
    VkFence fence;
    VkQueryPool queries = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    Buffer input{}, output{};
    size_t bytes, chunk, output_bytes;
    uint32_t width, lanes, chunks, timestamp_bits;
    bool copy;
    std::vector<uint32_t> expected;
    Buffer allocate(size_t size) {
        Buffer b{};
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkcheck(vkCreateBuffer(device, &bi, nullptr, &b.buffer));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device, b.buffer, &mr);
        int best = -1, score = -1;
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
            auto flags = mem.memoryTypes[i].propertyFlags;
            VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            if (!(mr.memoryTypeBits & (1u << i)) || (flags & required) != required) continue;
            int s = int(bool(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) * 2 + int(bool(flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT));
            if (s > score) { best = int(i); score = s; }
        }
        require(best >= 0, "host coherent Vulkan memory"); b.memory_type = uint32_t(best);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize = mr.size; ai.memoryTypeIndex = uint32_t(best);
        vkcheck(vkAllocateMemory(device, &ai, nullptr, &b.memory));
        vkcheck(vkBindBufferMemory(device, b.buffer, b.memory, 0));
        vkcheck(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.ptr));
        return b;
    }
    GPU(size_t size, const char * path, uint32_t w, uint32_t wg, uint32_t threads, bool is_copy) : bytes(size), width(w), lanes(threads), copy(is_copy) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ii.pApplicationInfo = &app;
        vkcheck(vkCreateInstance(&ii, nullptr, &instance));
        uint32_t n = 0; vkcheck(vkEnumeratePhysicalDevices(instance, &n, nullptr)); require(n == 1, "one Vulkan device");
        vkcheck(vkEnumeratePhysicalDevices(instance, &n, &physical));
        vkGetPhysicalDeviceProperties(physical, &props); vkGetPhysicalDeviceMemoryProperties(physical, &mem);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &n, nullptr);
        std::vector<VkQueueFamilyProperties> families(n); vkGetPhysicalDeviceQueueFamilyProperties(physical, &n, families.data());
        uint32_t family = 0; while (family < n && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;
        require(family < n, "compute queue"); timestamp_bits = families[family].timestampValidBits;
        float priority = 1;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex = family; qi.queueCount = 1; qi.pQueuePriorities = &priority;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
        vkcheck(vkCreateDevice(physical, &di, nullptr, &device)); vkGetDeviceQueue(device, family, 0, &queue);
        chunk = std::min(size_t(128) << 20, size_t(props.limits.maxStorageBufferRange));
        require(chunk && bytes % chunk == 0 && chunk % props.limits.minStorageBufferOffsetAlignment == 0, "GPU chunk alignment");
        chunks = uint32_t(bytes / chunk); output_bytes = copy ? bytes : size_t(chunks) * lanes * 4;
        require((width == 1 || width == 4) && lanes % wg == 0 && wg <= props.limits.maxComputeWorkGroupInvocations, "GPU geometry");
        require(lanes / wg <= props.limits.maxComputeWorkGroupCount[0], "GPU dispatch size");
        input = allocate(bytes); output = allocate(output_bytes);
        auto * src = static_cast<uint32_t *>(input.ptr);
        for (size_t i = 0; i < bytes / 4; ++i) src[i] = pattern(uint32_t(i));
        std::memset(output.ptr, 0x5a, output_bytes);
        if (!copy) {
            expected.resize(output_bytes / 4, 0);
            for (uint32_t c = 0; c < chunks; ++c) {
                size_t base = size_t(c) * chunk / 4;
                for (size_t i = 0; i < chunk / 4; ++i) expected[size_t(c) * lanes + (i / width) % lanes] ^= src[base + i];
            }
            VkDescriptorSetLayoutBinding bindings[2]{};
            for (uint32_t i = 0; i < 2; ++i) { bindings[i].binding = i; bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bindings[i].descriptorCount = 1; bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
            VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; li.bindingCount = 2; li.pBindings = bindings;
            vkcheck(vkCreateDescriptorSetLayout(device, &li, nullptr, &layout));
            VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
            VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &layout; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &push;
            vkcheck(vkCreatePipelineLayout(device, &pli, nullptr, &pipeline_layout));
            std::ifstream file(path, std::ios::binary | std::ios::ate); require(bool(file), "shader file");
            size_t length = size_t(file.tellg()); require(length && length % 4 == 0, "SPIR-V length");
            std::vector<uint32_t> code(length / 4); file.seekg(0); file.read(reinterpret_cast<char *>(code.data()), length); require(bool(file), "shader read");
            VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; si.codeSize = length; si.pCode = code.data();
            vkcheck(vkCreateShaderModule(device, &si, nullptr, &shader));
            VkSpecializationMapEntry entry{0, 0, sizeof(wg)}; VkSpecializationInfo spec{1, &entry, sizeof(wg), &wg};
            VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; pi.layout = pipeline_layout;
            pi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pi.stage.module = shader; pi.stage.pName = "main"; pi.stage.pSpecializationInfo = &spec;
            vkcheck(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline));
        }
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex = family;
        vkcheck(vkCreateCommandPool(device, &pci, nullptr, &pool));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        vkcheck(vkAllocateCommandBuffers(device, &cai, &command));
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; vkcheck(vkCreateFence(device, &fi, nullptr, &fence));
        if (timestamp_bits) {
            VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO}; qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 2;
            vkcheck(vkCreateQueryPool(device, &qpi, nullptr, &queries));
        }
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkcheck(vkBeginCommandBuffer(command, &cbi));
        if (queries) { vkCmdResetQueryPool(command, queries, 0, 2); vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0); }
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        if (copy) {
            VkBufferCopy region{0, 0, bytes}; vkCmdCopyBuffer(command, input.buffer, output.buffer, 1, &region);
        } else {
            VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, chunks * 2};
            VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = chunks; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
            vkcheck(vkCreateDescriptorPool(device, &dpi, nullptr, &descriptors));
            std::vector<VkDescriptorSetLayout> layouts(chunks, layout); std::vector<VkDescriptorSet> sets(chunks);
            VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = descriptors; dai.descriptorSetCount = chunks; dai.pSetLayouts = layouts.data();
            vkcheck(vkAllocateDescriptorSets(device, &dai, sets.data()));
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            for (uint32_t c = 0; c < chunks; ++c) {
                VkDescriptorBufferInfo buffers[2] = {{input.buffer, size_t(c) * chunk, chunk}, {output.buffer, size_t(c) * lanes * 4, size_t(lanes) * 4}};
                VkWriteDescriptorSet writes[2]{};
                for (uint32_t i = 0; i < 2; ++i) { writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = sets[c]; writes[i].dstBinding = i; writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &buffers[i]; }
                vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
                uint32_t params[2] = {uint32_t(chunk / (4 * width)), lanes};
                vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &sets[c], 0, nullptr);
                vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, params);
                vkCmdDispatch(command, lanes / wg, 1, 1);
            }
        }
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        if (queries) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
        vkcheck(vkEndCommandBuffer(command));
        std::printf("{\"type\":\"device\",\"name\":\"%s\",\"max_storage_range\":%u,\"chunk_bytes\":%zu,\"memory_type\":%u,\"memory_flags\":%u,\"timestamp_bits\":%u,\"timestamp_period_ns\":%.9g,\"output_bytes\":%zu}\n", props.deviceName, props.limits.maxStorageBufferRange, chunk, input.memory_type, mem.memoryTypes[input.memory_type].propertyFlags, timestamp_bits, props.limits.timestampPeriod, output_bytes);
    }
    double step() {
        vkcheck(vkResetFences(device, 1, &fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &command;
        vkcheck(vkQueueSubmit(queue, 1, &si, fence)); vkcheck(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        if (!queries) return 0;
        uint64_t times[2]; vkcheck(vkGetQueryPoolResults(device, queries, 0, 2, sizeof(times), times, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
        uint64_t mask = timestamp_bits == 64 ? UINT64_MAX : ((uint64_t(1) << timestamp_bits) - 1);
        return double((times[1] - times[0]) & mask) * props.limits.timestampPeriod * 1e-9;
    }
    void verify() {
        require(std::memcmp(output.ptr, copy ? input.ptr : expected.data(), output_bytes) == 0, "GPU every-output checksum/copy");
    }
    ~GPU() {
        vkcheck(vkDeviceWaitIdle(device));
        for (Buffer b : {input, output}) { vkUnmapMemory(device, b.memory); vkDestroyBuffer(device, b.buffer, nullptr); vkFreeMemory(device, b.memory, nullptr); }
        if (queries) vkDestroyQueryPool(device, queries, nullptr);
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (shader) vkDestroyShaderModule(device, shader, nullptr);
        if (descriptors) vkDestroyDescriptorPool(device, descriptors, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        if (layout) vkDestroyDescriptorSetLayout(device, layout, nullptr);
        vkDestroyFence(device, fence, nullptr); vkDestroyCommandPool(device, pool, nullptr);
        vkDestroyDevice(device, nullptr); vkDestroyInstance(instance, nullptr);
    }
};
struct Timing {
    double start = 0, end = 0;
    std::vector<double> wall, device;
    void print(const char * name, size_t per_pass) {
        double sum = 0; for (double x : device) sum += x;
        std::printf("{\"type\":\"result\",\"engine\":\"%s\",\"passes\":%zu,\"bytes_per_pass\":%zu,\"elapsed_s\":%.9f,\"GBps\":%.9f,\"device_GBps\":%.9f,\"wall_ms\":[", name, wall.size(), per_pass, end - start, wall.size() * double(per_pass) / (end - start) / 1e9, sum ? wall.size() * double(per_pass) / sum / 1e9 : 0);
        for (size_t i = 0; i < wall.size(); ++i) std::printf("%s%.6f", i ? "," : "", wall[i] * 1000);
        std::printf("],\"device_ms\":[");
        for (size_t i = 0; i < device.size(); ++i) std::printf("%s%.6f", i ? "," : "", device[i] * 1000);
        std::printf("]}\n");
    }
};
int main(int argc, char ** argv) {
    require(argc == 11 || argc == 12, "usage: mode shader width wg lanes cpu_threads cpu_mask_hex prefetch_bytes seconds MiB [cpu_chunk_KiB]");
    std::string mode(argv[1]);
    bool use_cpu = mode == "cpu" || mode == "both", use_gpu = mode == "gpu" || mode == "copy" || mode == "both";
    require(use_cpu || use_gpu, "mode");
    size_t bytes = size_t(std::stoul(argv[10])) << 20; double duration = std::stod(argv[9]);
    require(bytes >= (size_t(128) << 20) && bytes <= (size_t(1024) << 20) && duration > 0 && duration <= 10, "bounded run");
    std::unique_ptr<CPU> cpu; std::unique_ptr<GPU> gpu;
    if (use_cpu) cpu = std::make_unique<CPU>(bytes, std::stoi(argv[6]), unsigned(std::stoul(argv[7], nullptr, 16)), std::stoul(argv[8]), argc == 12 ? std::stoul(argv[11]) * 1024 : 0);
    if (use_gpu) gpu = std::make_unique<GPU>(bytes, argv[2], std::stoul(argv[3]), std::stoul(argv[4]), std::stoul(argv[5]), mode == "copy");
    for (int i = 0; i < 8; ++i) { if (cpu) cpu->step(); if (gpu) gpu->step(); }
    if (cpu) cpu->verify(); if (gpu) gpu->verify();
    Timing ct, gt; ct.wall.reserve(2000); gt.wall.reserve(2000); gt.device.reserve(2000);
    pthread_barrier_t start; require(pthread_barrier_init(&start, nullptr, use_cpu && use_gpu ? 2 : 1) == 0, "joint barrier");
    auto crun = [&]() { pthread_barrier_wait(&start); ct.start = now(); do { double t = now(); cpu->step(); ct.wall.push_back(now() - t); } while (now() < ct.start + duration); ct.end = now(); };
    auto grun = [&]() { pthread_barrier_wait(&start); gt.start = now(); do { double t = now(); double d = gpu->step(); gt.wall.push_back(now() - t); gt.device.push_back(d); } while (now() < gt.start + duration); gt.end = now(); };
    if (cpu && gpu) { std::thread c(crun); grun(); c.join(); }
    else if (cpu) crun(); else grun();
    pthread_barrier_destroy(&start);
    if (cpu) { cpu->verify(); ct.print("cpu_read", bytes); cpu->print(); }
    if (gpu) { gpu->verify(); gt.print(mode == "copy" ? "gpu_copy_read_plus_write" : "gpu_read", bytes * (mode == "copy" ? 2 : 1)); }
    if (cpu && gpu) {
        double elapsed = std::max(ct.end, gt.end) - std::min(ct.start, gt.start);
        double overlap = std::min(ct.end, gt.end) - std::max(ct.start, gt.start);
        std::printf("{\"type\":\"joint\",\"GBps\":%.9f,\"overlap_fraction\":%.9f,\"elapsed_s\":%.9f,\"start_skew_ms\":%.6f}\n", (ct.wall.size() + gt.wall.size()) * double(bytes) / elapsed / 1e9, overlap / elapsed, elapsed, std::abs(ct.start - gt.start) * 1000);
    }
    std::puts("{\"type\":\"audit\",\"status\":\"PASS\",\"verification\":\"all CPU slice checksums and every GPU output, before and after timing\"}");
    return 0;
}
