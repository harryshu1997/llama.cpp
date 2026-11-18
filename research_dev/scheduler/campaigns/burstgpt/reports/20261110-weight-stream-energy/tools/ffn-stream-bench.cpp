// Weight-stream energy benchmark, host legs: the FFN (gate/up mul_mat -> swiglu/geglu -> down mul_mat) of the
// first N layers of an f16 FFN shard, batch 1, as ONE ggml graph per decode call on one ggml backend: the CPU
// backend (default) or a GPU device such as CUDA0 with the weights resident in its memory (the kernels
// llama-server runs for host and GPU layers). Calls repeat back to back; a GPU call ends when its last output
// has been read back.
//   ffn-stream-bench --shard S.ffn.gguf --layers N --threads T [--device CUDA0] --idle-s I --warmup-s W --seconds S --out OUT.jsonl
// Lines: {"kind":"meta"..} {"kind":"idle","t0_ns","t1_ns"} {"kind":"call","phase","t_start_ns","t_end_ns"}
//        {"kind":"measure","t0_ns","t1_ns","calls"}.  Times are CLOCK_MONOTONIC.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

void sleep_s(double seconds) {
    if (seconds > 0) std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
}

bool load_tensor(int fd, gguf_context * gguf, const std::string & name, ggml_tensor * dst) {
    const int64_t id = gguf_find_tensor(gguf, name.c_str());
    if (id < 0) return false;
    const size_t bytes = ggml_nbytes(dst);
    const off_t base = (off_t) (gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, id));
    std::vector<uint8_t> host(bytes);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t n = pread(fd, host.data() + done, bytes - done, base + (off_t) done);
        if (n <= 0) return false;
        done += (size_t) n;
    }
    ggml_backend_tensor_set(dst, host.data(), 0, bytes);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::string shard, out_path, device_name = "CPU";
    int n_layers = 0, threads = 8;
    double idle_s = 10, warmup_s = 3, seconds = 30;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        const char * v = argv[i + 1];
        if      (a == "--shard")    shard = v;
        else if (a == "--layers")   n_layers = atoi(v);
        else if (a == "--threads")  threads = atoi(v);
        else if (a == "--idle-s")   idle_s = atof(v);
        else if (a == "--warmup-s") warmup_s = atof(v);
        else if (a == "--seconds")  seconds = atof(v);
        else if (a == "--out")      out_path = v;
        else if (a == "--device")   device_name = v;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (shard.empty() || out_path.empty() || n_layers <= 0 || threads <= 0) {
        fprintf(stderr, "usage: %s --shard S.gguf --layers N --threads T --idle-s I --warmup-s W --seconds S --out O\n",
                argv[0]);
        return 2;
    }
    const int fd = open(shard.c_str(), O_RDONLY | O_CLOEXEC);
    ggml_context * meta = nullptr;
    gguf_init_params gp = { true, &meta };
    gguf_context * gguf = fd >= 0 ? gguf_init_from_file(shard.c_str(), gp) : nullptr;
    if (gguf == nullptr) { fprintf(stderr, "cannot open %s\n", shard.c_str()); return 1; }
    const int64_t arch_id = gguf_find_key(gguf, "general.architecture");
    const std::string arch = arch_id >= 0 ? gguf_get_val_str(gguf, arch_id) : "";
    const bool geglu = arch.rfind("gemma", 0) == 0;

    ggml_backend_load_all();
    const bool on_cpu = device_name == "CPU";
    ggml_backend_dev_t device = ggml_backend_dev_by_name(device_name.c_str());
    ggml_backend_t backend = device != nullptr ? ggml_backend_dev_init(device, nullptr) : nullptr;
    if (backend == nullptr) { fprintf(stderr, "%s backend not found\n", device_name.c_str()); return 1; }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_threads == nullptr && on_cpu) { fprintf(stderr, "cannot set threads\n"); return 1; }
    if (set_threads != nullptr) set_threads(backend, threads);

    static const char * suffix[3] = { "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight" };
    ggml_init_params wp = { ggml_tensor_overhead() * (size_t) (3 * n_layers + 4), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    std::vector<ggml_tensor *> weights;
    size_t bytes = 0;
    for (int l = 0; l < n_layers; ++l) {
        for (const char * s : suffix) {
            const std::string name = "blk." + std::to_string(l) + "." + s;
            ggml_tensor * m = ggml_get_tensor(meta, name.c_str());
            if (m == nullptr || m->type != GGML_TYPE_F16) { fprintf(stderr, "missing f16 tensor %s\n", name.c_str()); return 1; }
            ggml_tensor * w = ggml_new_tensor_2d(wctx, m->type, m->ne[0], m->ne[1]);
            ggml_set_name(w, name.c_str());
            weights.push_back(w);
            bytes += ggml_nbytes(m);
        }
    }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (wbuf == nullptr) { fprintf(stderr, "weight allocation failed\n"); return 1; }
    for (ggml_tensor * w : weights) {
        if (!load_tensor(fd, gguf, ggml_get_name(w), w)) { fprintf(stderr, "read failed %s\n", ggml_get_name(w)); return 1; }
    }
    const int64_t n_embd = weights[0]->ne[0];
    const int64_t n_ff = weights[0]->ne[1];

    ggml_init_params gpar = { ggml_tensor_overhead() * (size_t) (8 * n_layers + 8) + ggml_graph_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(gpar);
    ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(x);
    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_tensor * last_y = nullptr;
    for (int l = 0; l < n_layers; ++l) {
        ggml_tensor * g = ggml_mul_mat(gctx, weights[(size_t) l * 3 + 0], x);
        ggml_tensor * u = ggml_mul_mat(gctx, weights[(size_t) l * 3 + 1], x);
        ggml_tensor * a = geglu ? ggml_geglu_split(gctx, g, u) : ggml_swiglu_split(gctx, g, u);
        ggml_tensor * y = ggml_mul_mat(gctx, weights[(size_t) l * 3 + 2], a);
        ggml_set_output(y);
        last_y = y;
        ggml_build_forward_expand(graph, y);
    }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, graph)) { fprintf(stderr, "graph allocation failed\n"); return 1; }
    std::mt19937 rng(1234);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> input((size_t) n_embd);
    for (float & v : input) v = normal(rng);
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));

    std::vector<float> readback((size_t) ggml_nelements(last_y));

    FILE * out = fopen(out_path.c_str(), "w");
    if (out == nullptr) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    fprintf(out, "{\"kind\":\"meta\",\"shard\":\"%s\",\"arch\":\"%s\",\"geglu\":%s,\"layers\":%d,\"threads\":%d,"
                 "\"n_embd\":%lld,\"n_ff\":%lld,\"weight_bytes\":%zu,\"device\":\"%s\",\"backend\":\"%s\"}\n",
            shard.c_str(), arch.c_str(), geglu ? "true" : "false", n_layers, threads, (long long) n_embd,
            (long long) n_ff, bytes, ggml_backend_dev_description(device), device_name.c_str());
    fflush(out);
    const int64_t idle0 = now_ns();
    sleep_s(idle_s);
    const int64_t idle1 = now_ns();
    fprintf(out, "{\"kind\":\"idle\",\"t0_ns\":%lld,\"t1_ns\":%lld}\n", (long long) idle0, (long long) idle1);

    const int64_t run0 = now_ns();
    const int64_t measure0 = run0 + (int64_t) (warmup_s * 1e9);
    const int64_t run1 = measure0 + (int64_t) (seconds * 1e9);
    long calls = 0;
    int64_t first = 0, last = 0;
    std::vector<int64_t> starts, ends;
    starts.reserve(1 << 20); ends.reserve(1 << 20);
    for (;;) {
        const int64_t t0 = now_ns();
        if (t0 >= run1) break;
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        if (!on_cpu) ggml_backend_tensor_get(last_y, readback.data(), 0, ggml_nbytes(last_y));
        const int64_t t1 = now_ns();
        starts.push_back(t0); ends.push_back(t1);
        if (t0 >= measure0) { if (first == 0) first = t0; last = t1; ++calls; }
    }
    for (size_t i = 0; i < starts.size(); ++i) {
        fprintf(out, "{\"kind\":\"call\",\"phase\":\"%s\",\"t_start_ns\":%lld,\"t_end_ns\":%lld}\n",
                starts[i] >= measure0 ? "measure" : "warmup", (long long) starts[i], (long long) ends[i]);
    }
    fprintf(out, "{\"kind\":\"measure\",\"t0_ns\":%lld,\"t1_ns\":%lld,\"calls\":%ld}\n", (long long) first, (long long) last, calls);
    fclose(out);
    return 0;
}
