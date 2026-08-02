// Host coordinator for the operator-split test.
//
// Splits ONE matmul [K,N] x [K,1] across the local CUDA GPU and N phone
// workers reached over adb-forwarded USB sockets. Row counts are chosen so
// every device is predicted to finish at the same instant:
//
//     t = n_gpu / r_gpu = n_i / r_i + W_i        (r = rows/us, W = wire RTT)
//
// Each iteration issues the phone sends first (so their transfer overlaps GPU
// compute), computes the GPU slice, then joins. Reports wall time against a
// GPU-only run of the same full matmul.
//
// usage: tp_host <N> <K> <iters> [port:rate:wire_us ...]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct Phone {
    int      port;
    double   rate;      // rows per microsecond, measured
    double   wire_us;   // measured round-trip floor
    int64_t  rows = 0;
    int      fd   = -1;
    std::vector<float> out;
    double   last_ms = 0;
};

static bool read_exact(int fd, void * dst, size_t n) {
    auto * p = static_cast<char *>(dst);
    while (n > 0) { ssize_t r = read(fd, p, n); if (r <= 0) return false; p += r; n -= (size_t) r; }
    return true;
}
static bool write_exact(int fd, const void * src, size_t n) {
    const auto * p = static_cast<const char *>(src);
    while (n > 0) { ssize_t w = write(fd, p, n); if (w <= 0) return false; p += w; n -= (size_t) w; }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <N> <K> <iters> [port:rate:wire_us ...]\n", argv[0]);
        return 2;
    }
    const int64_t N     = atoll(argv[1]);
    const int64_t K     = atoll(argv[2]);
    const int     iters = atoi(argv[3]);
    signal(SIGPIPE, SIG_IGN);

    // spec: port:rate:wire_us[:rows]. An explicit row count pins the split to
    // the slice the running worker was started with (its graph is static).
    bool pinned = false;
    std::vector<Phone> phones;
    for (int i = 4; i < argc; i++) {
        Phone p{};
        long long rows = 0;
        const int got = sscanf(argv[i], "%d:%lf:%lf:%lld", &p.port, &p.rate, &p.wire_us, &rows);
        if (got < 3) { fprintf(stderr, "bad spec '%s'\n", argv[i]); return 2; }
        if (got == 4) { p.rows = rows; pinned = true; }
        phones.push_back(p);
    }

    ggml_backend_t backend = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            backend = ggml_backend_dev_init(dev, nullptr);
            fprintf(stderr, "[host] gpu: %s\n", ggml_backend_dev_description(dev));
            break;
        }
    }
    if (!backend) { fprintf(stderr, "[host] no GPU backend\n"); return 1; }

    // ---- measure the GPU's own rate on this exact shape (full matmul) ----
    auto build = [&](int64_t rows, ggml_context ** pctx, ggml_tensor ** pa,
                     ggml_tensor ** pout, ggml_backend_buffer_t * pbuf,
                     ggml_cgraph ** pgf) {
        ggml_init_params ip = {};
        ip.mem_size   = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
        ip.no_alloc   = true;
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, rows);
        ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, 1);
        ggml_tensor * out = ggml_mul_mat(ctx, w, a);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { fprintf(stderr, "[host] alloc failed rows=%lld\n", (long long) rows); exit(1); }
        std::vector<float> src((size_t) K * rows);
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (auto & v : src) v = dist(rng);
        std::vector<char> q(ggml_nbytes(w));
        ggml_quantize_chunk(GGML_TYPE_Q8_0, src.data(), q.data(), 0, rows, K, nullptr);
        ggml_backend_tensor_set(w, q.data(), 0, ggml_nbytes(w));
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        *pctx = ctx; *pa = a; *pout = out; *pbuf = buf; *pgf = gf;
    };

    std::vector<float> act((size_t) K);
    { std::mt19937 rng(7); std::uniform_real_distribution<float> d(-1.f, 1.f);
      for (auto & v : act) v = d(rng); }

    double gpu_only_ms = 0;
    {
        ggml_context * ctx; ggml_tensor *a, *out; ggml_backend_buffer_t buf; ggml_cgraph * gf;
        build(N, &ctx, &a, &out, &buf, &gf);
        std::vector<float> res((size_t) N);
        for (int i = 0; i < 5; i++) {
            ggml_backend_tensor_set(a, act.data(), 0, sizeof(float) * K);
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_tensor_get(out, res.data(), 0, sizeof(float) * N);
        }
        std::vector<double> t;
        for (int i = 0; i < iters; i++) {
            auto t0 = clk::now();
            ggml_backend_tensor_set(a, act.data(), 0, sizeof(float) * K);
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_tensor_get(out, res.data(), 0, sizeof(float) * N);
            t.push_back(ms_since(t0));
        }
        std::sort(t.begin(), t.end());
        gpu_only_ms = t[t.size() / 2];
        ggml_backend_buffer_free(buf); ggml_free(ctx);
    }
    const double r_gpu = (double) N / (gpu_only_ms * 1000.0);   // rows per us
    printf("GPU-only     : %8.3f ms   (%.1f rows/us)\n", gpu_only_ms, r_gpu);

    // ---- balanced split: n_gpu/r_gpu = n_i/r_i + W_i ----
    int64_t n_gpu;
    double  t_pred;
    if (pinned) {
        int64_t phone_rows = 0;
        for (auto & p : phones) phone_rows += p.rows;
        if (phone_rows >= N) { printf("pinned rows exceed N\n"); return 2; }
        n_gpu  = N - phone_rows;
        t_pred = n_gpu / r_gpu;
    } else {
        double sum_r = 0, sum_rw = 0;
        for (auto & p : phones) { sum_r += p.rate; sum_rw += p.rate * p.wire_us; }
        const double n_gpu_f = ((double) N + sum_rw) / (1.0 + sum_r / r_gpu);
        if (n_gpu_f >= (double) N) {
            printf("NO VALID SPLIT: balance needs n_gpu=%.0f > N=%lld "
                   "(wire exceeds the whole GPU matmul)\n", n_gpu_f, (long long) N);
            printf("               phones would finish later than the GPU even with zero rows.\n");
            return 0;
        }
        n_gpu  = (int64_t) n_gpu_f;
        t_pred = n_gpu / r_gpu;
        int64_t assigned = n_gpu;
        for (auto & p : phones) {
            p.rows = (int64_t) std::max(0.0, p.rate * (t_pred - p.wire_us));
            assigned += p.rows;
        }
        n_gpu += (N - assigned);   // remainder to the GPU
    }
    printf("split        : gpu %lld rows (%.1f%%)", (long long) n_gpu, 100.0 * n_gpu / N);
    for (auto & p : phones) printf(", :%d %lld rows (%.1f%%)", p.port, (long long) p.rows, 100.0 * p.rows / N);
    printf("\npredicted    : %8.3f ms\n", t_pred / 1000.0);

    // ---- connect phones ----
    for (auto & p : phones) {
        p.fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in sa = {}; sa.sin_family = AF_INET;
        sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK); sa.sin_port = htons((uint16_t) p.port);
        if (connect(p.fd, (sockaddr *) &sa, sizeof sa) < 0) {
            fprintf(stderr, "[host] connect :%d failed\n", p.port); return 1;
        }
        int one = 1; setsockopt(p.fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        p.out.resize((size_t) p.rows);
    }

    ggml_context * ctx; ggml_tensor *a, *out; ggml_backend_buffer_t buf; ggml_cgraph * gf;
    build(n_gpu, &ctx, &a, &out, &buf, &gf);
    std::vector<float> gres((size_t) n_gpu);

    auto one_iter = [&]() {
        std::vector<std::thread> th;
        for (auto & p : phones) {
            th.emplace_back([&p, &act, K]() {
                auto t0 = clk::now();
                if (!write_exact(p.fd, act.data(), sizeof(float) * K)) { p.last_ms = -1; return; }
                if (!read_exact(p.fd, p.out.data(), sizeof(float) * p.out.size())) { p.last_ms = -1; return; }
                p.last_ms = ms_since(t0);
            });
        }
        ggml_backend_tensor_set(a, act.data(), 0, sizeof(float) * K);
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_tensor_get(out, gres.data(), 0, sizeof(float) * n_gpu);
        double g = ms_since(clk::now());   // placeholder, gpu timed outside
        (void) g;
        for (auto & t : th) t.join();
    };

    for (int i = 0; i < 3; i++) one_iter();
    std::vector<double> wall;
    std::vector<std::vector<double>> pms(phones.size());
    for (int i = 0; i < iters; i++) {
        auto t0 = clk::now();
        one_iter();
        wall.push_back(ms_since(t0));
        for (size_t j = 0; j < phones.size(); j++) pms[j].push_back(phones[j].last_ms);
    }
    std::sort(wall.begin(), wall.end());
    const double w_med = wall[wall.size() / 2];
    printf("3-device wall: %8.3f ms   (min %.3f, p95 %.3f)\n",
           w_med, wall.front(), wall[(size_t) (0.95 * wall.size())]);
    for (size_t j = 0; j < phones.size(); j++) {
        auto v = pms[j]; std::sort(v.begin(), v.end());
        printf("  phone :%d   : %8.3f ms  (%lld rows)\n",
               phones[j].port, v[v.size() / 2], (long long) phones[j].rows);
    }
    printf("SPEEDUP      : %.3fx  %s\n", gpu_only_ms / w_med,
           w_med < gpu_only_ms ? "(collaboration wins)" : "(GPU alone is faster)");

    for (auto & p : phones) close(p.fd);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    return 0;
}
