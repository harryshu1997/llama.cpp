// Dual-engine FFN benchmark: split each layer's intermediate columns between
// an HTP (NPU) backend and an OpenCL (GPU) backend, run both legs concurrently
// and sum the partial [n_embd, M] outputs on the CPU. Weights are real GGUF
// layers streamed across L layers so the working set exceeds the system cache.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct config {
    std::string model;
    std::string htp_name = "HTP";
    std::string gpu_name = "OpenCL";
    int         layer_start = 10;
    int         layers = 6;
    double      frac = 1.0;
    int64_t     align = 256;
    std::vector<int> batches = {1};
    int         sweeps = 50;
    int         warmup_sweeps = 3;
    std::string dump_prefix;
    std::string samples_csv;
    bool        cpu_ref = true;
    uint32_t    seed = 1234;
};

std::vector<int> parse_list(const char * s) {
    std::vector<int> out;
    std::string str(s);
    size_t pos = 0;
    while (pos < str.size()) {
        size_t next = str.find(',', pos);
        if (next == std::string::npos) next = str.size();
        out.push_back(atoi(str.substr(pos, next - pos).c_str()));
        pos = next + 1;
    }
    return out;
}

bool parse_args(int argc, char ** argv, config & cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : nullptr; };
        const char * v = nullptr;
        if      (a == "--model"        && (v = next())) cfg.model = v;
        else if (a == "--htp"          && (v = next())) cfg.htp_name = v;
        else if (a == "--gpu"          && (v = next())) cfg.gpu_name = v;
        else if (a == "--layer-start"  && (v = next())) cfg.layer_start = atoi(v);
        else if (a == "--layers"       && (v = next())) cfg.layers = atoi(v);
        else if (a == "--frac"         && (v = next())) cfg.frac = atof(v);
        else if (a == "--align"        && (v = next())) cfg.align = atoll(v);
        else if (a == "--batches"      && (v = next())) cfg.batches = parse_list(v);
        else if (a == "--sweeps"       && (v = next())) cfg.sweeps = atoi(v);
        else if (a == "--warmup"       && (v = next())) cfg.warmup_sweeps = atoi(v);
        else if (a == "--dump"         && (v = next())) cfg.dump_prefix = v;
        else if (a == "--samples"      && (v = next())) cfg.samples_csv = v;
        else if (a == "--seed"         && (v = next())) cfg.seed = (uint32_t) strtoul(v, nullptr, 0);
        else if (a == "--no-cpu-ref") cfg.cpu_ref = false;
        else {
            fprintf(stderr, "unknown or incomplete argument: %s\n", a.c_str());
            return false;
        }
    }
    return !cfg.model.empty() && cfg.layers > 0 && cfg.frac >= 0.0 && cfg.frac <= 1.0 && cfg.align > 0;
}

struct host_matrix {
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    std::vector<ggml_fp16_t> data;
};

bool read_f16(int fd, gguf_context * gguf, ggml_context * meta, const std::string & name, host_matrix & out) {
    const int64_t id = gguf_find_tensor(gguf, name.c_str());
    ggml_tensor * t = id >= 0 ? ggml_get_tensor(meta, name.c_str()) : nullptr;
    if (t == nullptr || t->type != GGML_TYPE_F16 || t->ne[2] != 1 || t->ne[3] != 1) {
        fprintf(stderr, "missing or non-F16 matrix: %s\n", name.c_str());
        return false;
    }
    out.ne0 = t->ne[0];
    out.ne1 = t->ne[1];
    out.data.resize((size_t) out.ne0 * out.ne1);
    const size_t bytes = ggml_nbytes(t);
    const off_t  base  = (off_t) (gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, id));
    size_t done = 0;
    auto * dst = (uint8_t *) out.data.data();
    while (done < bytes) {
        ssize_t n = pread(fd, dst + done, bytes - done, base + (off_t) done);
        if (n <= 0) {
            fprintf(stderr, "short read: %s\n", name.c_str());
            return false;
        }
        done += (size_t) n;
    }
    posix_fadvise(fd, base, (off_t) bytes, POSIX_FADV_DONTNEED);
    return true;
}

struct layer_graph {
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   graph = nullptr;
    ggml_tensor *   input = nullptr;
    ggml_tensor *   output = nullptr;
    ggml_gallocr_t  galloc = nullptr;
};

struct leg {
    std::string        label;
    ggml_backend_t     backend = nullptr;
    int64_t            offset = 0;
    int64_t            cols = 0;
    std::vector<ggml_context *>         wctx;
    std::vector<ggml_backend_buffer_t>  wbuf;
    std::vector<ggml_tensor *>          gate, up, down;
    std::vector<layer_graph>            graphs;
    std::vector<float>                  out;
    double last_ms = 0.0;
    double last_compute_ms = 0.0;
    bool   ok = true;

    bool active() const { return cols > 0; }
};

void free_graphs(leg & l) {
    for (auto & g : l.graphs) {
        if (g.galloc) ggml_gallocr_free(g.galloc);
        if (g.ctx) ggml_free(g.ctx);
    }
    l.graphs.clear();
}

bool build_graphs(leg & l, int64_t n_embd, int M, bool swiglu) {
    free_graphs(l);
    l.out.assign((size_t) n_embd * M, 0.0f);
    for (size_t i = 0; i < l.gate.size(); ++i) {
        layer_graph g;
        ggml_init_params p = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        g.ctx = ggml_init(p);
        g.input = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, M);
        ggml_set_input(g.input);
        ggml_tensor * gate = ggml_mul_mat(g.ctx, l.gate[i], g.input);
        ggml_tensor * up   = ggml_mul_mat(g.ctx, l.up[i], g.input);
        ggml_tensor * act  = swiglu ? ggml_swiglu_split(g.ctx, gate, up) : ggml_geglu_split(g.ctx, gate, up);
        g.output = ggml_mul_mat(g.ctx, l.down[i], act);
        ggml_set_output(g.output);
        g.graph = ggml_new_graph(g.ctx);
        ggml_build_forward_expand(g.graph, g.output);
        for (int n = 0; n < ggml_graph_n_nodes(g.graph); ++n) {
            ggml_tensor * node = ggml_graph_node(g.graph, n);
            if (!ggml_backend_supports_op(l.backend, node)) {
                fprintf(stderr, "[%s] backend does not support %s (%s) at M=%d\n",
                        l.label.c_str(), ggml_op_desc(node), node->name, M);
                return false;
            }
        }
        g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(l.backend));
        if (!ggml_gallocr_alloc_graph(g.galloc, g.graph)) {
            fprintf(stderr, "[%s] graph allocation failed\n", l.label.c_str());
            return false;
        }
        l.graphs.push_back(g);
    }
    return true;
}

void run_leg(leg & l, int layer, const float * input) {
    const auto t0 = clk::now();
    layer_graph & g = l.graphs[layer];
    ggml_backend_tensor_set(g.input, input, 0, ggml_nbytes(g.input));
    const auto t1 = clk::now();
    l.ok = ggml_backend_graph_compute(l.backend, g.graph) == GGML_STATUS_SUCCESS;
    l.last_compute_ms = ms_since(t1);
    ggml_backend_tensor_get(g.output, l.out.data(), 0, ggml_nbytes(g.output));
    l.last_ms = ms_since(t0);
}

// persistent GPU-leg worker; the main thread runs the HTP leg
class gpu_worker {
public:
    explicit gpu_worker(leg & l) : leg_(l), thread_([this]() { loop(); }) {}
    ~gpu_worker() {
        stop_.store(true, std::memory_order_release);
        go_.fetch_add(1, std::memory_order_acq_rel);
        thread_.join();
    }
    void start(int layer, const float * input) {
        layer_ = layer;
        input_ = input;
        go_.fetch_add(1, std::memory_order_acq_rel);
    }
    void wait() {
        const uint64_t target = go_.load(std::memory_order_acquire);
        while (done_.load(std::memory_order_acquire) != target) { }
    }
    clk::time_point started_at() const { return started_; }
private:
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            uint64_t cur;
            while ((cur = go_.load(std::memory_order_acquire)) == seen) { }
            seen = cur;
            if (stop_.load(std::memory_order_acquire)) return;
            started_ = clk::now();
            run_leg(leg_, layer_, input_);
            done_.store(seen, std::memory_order_release);
        }
    }
    leg & leg_;
    std::atomic<uint64_t> go_{0};
    std::atomic<uint64_t> done_{0};
    std::atomic<bool>     stop_{false};
    int layer_ = 0;
    const float * input_ = nullptr;
    clk::time_point started_;
    std::thread thread_;
};

struct stats {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    double mean() const { double s = 0; for (double x : v) s += x; return v.empty() ? 0 : s / v.size(); }
    double q(double p) const {
        if (v.empty()) return 0;
        std::vector<double> s = v;
        std::sort(s.begin(), s.end());
        size_t i = (size_t) std::min<double>(s.size() - 1, std::max(0.0, std::ceil(p * s.size()) - 1));
        return s[i];
    }
};

struct mode_stats {
    stats wall, npu, gpu, npu_compute, gpu_compute, sync, merge, skew;
};

ggml_backend_t init_backend(const std::string & needle, std::string & name_out) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        std::string name = ggml_backend_dev_name(d);
        std::string desc = ggml_backend_dev_description(d);
        if (name.find(needle) != std::string::npos || desc.find(needle) != std::string::npos) {
            name_out = name + " (" + desc + ")";
            return ggml_backend_dev_init(d, nullptr);
        }
    }
    return nullptr;
}

void cpu_reference(const host_matrix & g, const host_matrix & u, const host_matrix & d,
                   const std::vector<float> & x, int M, bool swiglu, std::vector<float> & out) {
    const int64_t E = g.ne0, C = g.ne1;
    out.assign((size_t) E * M, 0.0f);
    std::vector<float> act((size_t) C);
    std::vector<float> row((size_t) std::max(E, C));
    for (int m = 0; m < M; ++m) {
        const float * xm = x.data() + (size_t) m * E;
        for (int64_t c = 0; c < C; ++c) {
            double sg = 0, su = 0;
            const ggml_fp16_t * gr = g.data.data() + (size_t) c * E;
            const ggml_fp16_t * ur = u.data.data() + (size_t) c * E;
            for (int64_t e = 0; e < E; ++e) {
                sg += (double) ggml_fp16_to_fp32(gr[e]) * xm[e];
                su += (double) ggml_fp16_to_fp32(ur[e]) * xm[e];
            }
            double a;
            if (swiglu) {
                a = sg / (1.0 + std::exp(-sg)) * su;
            } else {
                a = 0.5 * sg * (1.0 + std::tanh(0.7978845608028654 * (sg + 0.044715 * sg * sg * sg))) * su;
            }
            act[c] = (float) a;
        }
        for (int64_t e = 0; e < E; ++e) {
            const ggml_fp16_t * dr = d.data.data() + (size_t) e * C;
            double s = 0;
            for (int64_t c = 0; c < C; ++c) s += (double) ggml_fp16_to_fp32(dr[c]) * act[c];
            out[(size_t) m * E + e] = (float) s;
        }
    }
}

void compare(const char * tag, const std::vector<float> & a, const std::vector<float> & ref) {
    double max_abs = 0, num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = (double) a[i] - ref[i];
        max_abs = std::max(max_abs, std::fabs(diff));
        num += diff * diff;
        den += (double) ref[i] * ref[i];
    }
    printf("DUALBENCH_CHECK %s max_abs=%.6g rel_l2=%.6g ref_l2=%.6g finite=%d\n", tag, max_abs,
           std::sqrt(num / std::max(den, 1e-30)), std::sqrt(den),
           (int) std::all_of(a.begin(), a.end(), [](float v) { return std::isfinite(v); }));
}

} // namespace

int main(int argc, char ** argv) {
    config cfg;
    if (!parse_args(argc, argv, cfg)) {
        fprintf(stderr,
                "usage: %s --model F16.gguf [--frac f] [--layers L] [--layer-start S] [--batches 1,4,8]\n"
                "          [--sweeps N] [--warmup N] [--align A] [--dump prefix] [--samples csv] [--no-cpu-ref]\n",
                argv[0]);
        return 2;
    }

    const int fd = open(cfg.model.c_str(), O_RDONLY | O_CLOEXEC);
    ggml_context * meta = nullptr;
    gguf_init_params gp = { true, &meta };
    gguf_context * gguf = fd >= 0 ? gguf_init_from_file(cfg.model.c_str(), gp) : nullptr;
    if (gguf == nullptr) {
        fprintf(stderr, "cannot open %s\n", cfg.model.c_str());
        return 1;
    }
    const int64_t arch_id = gguf_find_key(gguf, "general.architecture");
    const std::string arch = arch_id >= 0 ? gguf_get_val_str(gguf, arch_id) : "";
    const bool swiglu = arch != "gemma4";

    ggml_backend_load_all();

    ggml_tensor * g0 = ggml_get_tensor(meta, ("blk." + std::to_string(cfg.layer_start) + ".ffn_gate.weight").c_str());
    if (g0 == nullptr) {
        fprintf(stderr, "layer %d not found\n", cfg.layer_start);
        return 1;
    }
    const int64_t n_embd = g0->ne[0];
    const int64_t n_ff   = g0->ne[1];
    int64_t split = (int64_t) std::llround(cfg.frac * n_ff / cfg.align) * cfg.align;
    split = std::clamp<int64_t>(split, 0, n_ff);

    leg npu, gpu;
    npu.label = "npu"; npu.offset = 0;     npu.cols = split;
    gpu.label = "gpu"; gpu.offset = split; gpu.cols = n_ff - split;
    std::string npu_dev = "-", gpu_dev = "-";
    if (npu.active() && (npu.backend = init_backend(cfg.htp_name, npu_dev)) == nullptr) {
        fprintf(stderr, "HTP backend '%s' not found\n", cfg.htp_name.c_str());
        return 1;
    }
    if (gpu.active() && (gpu.backend = init_backend(cfg.gpu_name, gpu_dev)) == nullptr) {
        fprintf(stderr, "GPU backend '%s' not found\n", cfg.gpu_name.c_str());
        return 1;
    }

    const size_t layer_bytes = (size_t) 3 * n_embd * n_ff * sizeof(ggml_fp16_t);
    printf("DUALBENCH_CONFIG arch=%s n_embd=%lld n_ff=%lld layers=[%d,%d) frac_req=%.3f npu_cols=%lld gpu_cols=%lld "
           "frac_act=%.4f layer_bytes=%zu npu_dev=\"%s\" gpu_dev=\"%s\"\n",
           arch.c_str(), (long long) n_embd, (long long) n_ff, cfg.layer_start, cfg.layer_start + cfg.layers,
           cfg.frac, (long long) npu.cols, (long long) gpu.cols, (double) npu.cols / n_ff, layer_bytes,
           npu_dev.c_str(), gpu_dev.c_str());
    fflush(stdout);

    std::mt19937 rng(cfg.seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int max_m = *std::max_element(cfg.batches.begin(), cfg.batches.end());
    std::vector<float> input((size_t) n_embd * max_m);
    for (float & v : input) v = dist(rng);

    // first-layer reference for max_m tokens; smaller M use the prefix columns
    std::vector<float> ref_full;

    const auto load_t0 = clk::now();
    std::vector<ggml_fp16_t> down_slice;
    for (int li = 0; li < cfg.layers; ++li) {
        const std::string pfx = "blk." + std::to_string(cfg.layer_start + li);
        host_matrix hg, hu, hd;
        if (!read_f16(fd, gguf, meta, pfx + ".ffn_gate.weight", hg) ||
            !read_f16(fd, gguf, meta, pfx + ".ffn_up.weight", hu) ||
            !read_f16(fd, gguf, meta, pfx + ".ffn_down.weight", hd)) {
            return 1;
        }
        if (hg.ne0 != n_embd || hg.ne1 != n_ff || hu.ne0 != n_embd || hu.ne1 != n_ff ||
            hd.ne0 != n_ff || hd.ne1 != n_embd) {
            fprintf(stderr, "%s: unexpected FFN shapes\n", pfx.c_str());
            return 1;
        }
        if (li == 0 && cfg.cpu_ref) {
            const auto t_ref = clk::now();
            cpu_reference(hg, hu, hd, input, max_m, swiglu, ref_full);
            printf("DUALBENCH_CPUREF seconds=%.2f\n", ms_since(t_ref) / 1000.0);
        }
        for (leg * l : { &npu, &gpu }) {
            if (!l->active()) continue;
            ggml_init_params p = { ggml_tensor_overhead() * 4, nullptr, true };
            ggml_context * ctx = ggml_init(p);
            ggml_tensor * tg = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, l->cols);
            ggml_tensor * tu = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, l->cols);
            ggml_tensor * td = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, l->cols, n_embd);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(
                    ctx, ggml_backend_get_default_buffer_type(l->backend));
            if (buf == nullptr) {
                fprintf(stderr, "[%s] weight allocation failed at layer %d\n", l->label.c_str(), li);
                return 1;
            }
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            const size_t row = (size_t) n_embd;
            ggml_backend_tensor_set(tg, hg.data.data() + l->offset * row, 0, ggml_nbytes(tg));
            ggml_backend_tensor_set(tu, hu.data.data() + l->offset * row, 0, ggml_nbytes(tu));
            down_slice.resize((size_t) l->cols * n_embd);
            for (int64_t e = 0; e < n_embd; ++e) {
                memcpy(down_slice.data() + (size_t) e * l->cols, hd.data.data() + (size_t) e * n_ff + l->offset,
                       (size_t) l->cols * sizeof(ggml_fp16_t));
            }
            ggml_backend_tensor_set(td, down_slice.data(), 0, ggml_nbytes(td));
            l->wctx.push_back(ctx);
            l->wbuf.push_back(buf);
            l->gate.push_back(tg);
            l->up.push_back(tu);
            l->down.push_back(td);
        }
    }
    std::vector<ggml_fp16_t>().swap(down_slice);
    printf("DUALBENCH_LOAD seconds=%.2f npu_weight_bytes=%zu gpu_weight_bytes=%zu\n", ms_since(load_t0) / 1000.0,
           (size_t) 3 * n_embd * npu.cols * 2 * cfg.layers, (size_t) 3 * n_embd * gpu.cols * 2 * cfg.layers);
    fflush(stdout);

    FILE * samples = cfg.samples_csv.empty() ? nullptr : fopen(cfg.samples_csv.c_str(), "w");
    if (samples) fprintf(samples, "M,mode,sweep,layer,wall_ms,npu_ms,gpu_ms,npu_compute_ms,gpu_compute_ms,merge_ms\n");

    std::vector<float> merged;
    for (size_t bi = 0; bi < cfg.batches.size(); ++bi) {
        const int M = cfg.batches[bi];
        for (leg * l : { &npu, &gpu }) {
            if (l->active() && !build_graphs(*l, n_embd, M, swiglu)) return 1;
        }
        merged.assign((size_t) n_embd * M, 0.0f);
        const bool dual = npu.active() && gpu.active();

        std::vector<std::string> modes;
        if (dual) modes = { "dual", "npu_solo", "gpu_solo" };
        else      modes = { npu.active() ? "npu_solo" : "gpu_solo" };
        std::vector<mode_stats> ms(modes.size());

        gpu_worker * worker = dual ? new gpu_worker(gpu) : nullptr;
        bool all_ok = true;
        for (int sweep = -cfg.warmup_sweeps; sweep < cfg.sweeps; ++sweep) {
            for (size_t mi = 0; mi < modes.size(); ++mi) {
                const std::string & mode = modes[mi];
                for (int li = 0; li < cfg.layers; ++li) {
                    double wall = 0, merge = 0, skew = 0;
                    const auto t0 = clk::now();
                    if (mode == "dual") {
                        worker->start(li, input.data());
                        const auto npu_start = clk::now();
                        run_leg(npu, li, input.data());
                        worker->wait();
                        skew = std::chrono::duration<double, std::milli>(worker->started_at() - npu_start).count();
                        const auto tm = clk::now();
                        for (size_t i = 0; i < merged.size(); ++i) merged[i] = npu.out[i] + gpu.out[i];
                        merge = ms_since(tm);
                        wall = ms_since(t0);
                    } else {
                        leg & l = mode == "npu_solo" ? npu : gpu;
                        run_leg(l, li, input.data());
                        wall = ms_since(t0);
                        if (!dual) merged = l.out;
                    }
                    all_ok = all_ok && npu.ok && gpu.ok;
                    if (sweep < 0) continue;
                    mode_stats & s = ms[mi];
                    s.wall.add(wall);
                    s.merge.add(merge);
                    s.skew.add(skew);
                    const bool un = mode == "dual" || mode == "npu_solo";
                    const bool ug = mode == "dual" || mode == "gpu_solo";
                    if (un) { s.npu.add(npu.last_ms); s.npu_compute.add(npu.last_compute_ms); }
                    if (ug) { s.gpu.add(gpu.last_ms); s.gpu_compute.add(gpu.last_compute_ms); }
                    if (mode == "dual") s.sync.add(wall - merge - std::max(npu.last_ms, gpu.last_ms));
                    if (samples) {
                        fprintf(samples, "%d,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", M, mode.c_str(), sweep, li, wall,
                                un ? npu.last_ms : 0.0, ug ? gpu.last_ms : 0.0, un ? npu.last_compute_ms : 0.0,
                                ug ? gpu.last_compute_ms : 0.0, merge);
                    }
                }
                // correctness snapshot: layer 0 output of the final (dual or single-leg) sweep
                if (sweep == cfg.sweeps - 1 && mi == 0) {
                    // re-run layer 0 to capture its output
                    if (mode == "dual") {
                        worker->start(0, input.data());
                        run_leg(npu, 0, input.data());
                        worker->wait();
                        for (size_t i = 0; i < merged.size(); ++i) merged[i] = npu.out[i] + gpu.out[i];
                    } else {
                        leg & l = npu.active() ? npu : gpu;
                        run_leg(l, 0, input.data());
                        merged = l.out;
                    }
                    if (cfg.cpu_ref) {
                        std::vector<float> ref(ref_full.begin(), ref_full.begin() + (size_t) n_embd * M);
                        compare(("M=" + std::to_string(M) + " " + mode + "_vs_cpu_ref").c_str(), merged, ref);
                    }
                    if (!cfg.dump_prefix.empty()) {
                        const std::string path = cfg.dump_prefix + "_M" + std::to_string(M) + ".f32";
                        FILE * f = fopen(path.c_str(), "wb");
                        if (f) { fwrite(merged.data(), sizeof(float), merged.size(), f); fclose(f); }
                    }
                }
            }
        }
        delete worker;

        for (size_t mi = 0; mi < modes.size(); ++mi) {
            const mode_stats & s = ms[mi];
            const double gbps = layer_bytes / (s.wall.mean() * 1e6);
            const double npu_leg_bytes = (double) 3 * n_embd * npu.cols * 2;
            const double gpu_leg_bytes = (double) 3 * n_embd * gpu.cols * 2;
            printf("DUALBENCH_RESULT M=%d mode=%s frac=%.4f npu_cols=%lld gpu_cols=%lld n=%zu "
                   "wall_mean=%.3f wall_p50=%.3f wall_p90=%.3f wall_min=%.3f "
                   "npu_mean=%.3f npu_p50=%.3f npu_compute_p50=%.3f gpu_mean=%.3f gpu_p50=%.3f gpu_compute_p50=%.3f "
                   "sync_mean=%.4f sync_p50=%.4f merge_mean=%.4f skew_mean=%.4f "
                   "agg_gbps=%.2f npu_leg_gbps=%.2f gpu_leg_gbps=%.2f ok=%d\n",
                   M, modes[mi].c_str(), (double) npu.cols / n_ff, (long long) npu.cols, (long long) gpu.cols,
                   s.wall.v.size(), s.wall.mean(), s.wall.q(0.5), s.wall.q(0.9), s.wall.q(0.0),
                   s.npu.mean(), s.npu.q(0.5), s.npu_compute.q(0.5), s.gpu.mean(), s.gpu.q(0.5), s.gpu_compute.q(0.5),
                   s.sync.mean(), s.sync.q(0.5), s.merge.mean(), s.skew.mean(), gbps,
                   s.npu.v.empty() ? 0.0 : npu_leg_bytes / (s.npu.mean() * 1e6),
                   s.gpu.v.empty() ? 0.0 : gpu_leg_bytes / (s.gpu.mean() * 1e6), (int) all_ok);
        }
        fflush(stdout);
    }
    if (samples) fclose(samples);

    for (leg * l : { &npu, &gpu }) {
        free_graphs(*l);
        for (auto b : l->wbuf) ggml_backend_buffer_free(b);
        for (auto c : l->wctx) ggml_free(c);
        if (l->backend) ggml_backend_free(l->backend);
    }
    gguf_free(gguf);
    ggml_free(meta);
    close(fd);
    return 0;
}
