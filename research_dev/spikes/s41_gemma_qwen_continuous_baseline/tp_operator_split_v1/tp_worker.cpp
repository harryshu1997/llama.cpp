// Phone-side worker for the operator-split test.
//
// Holds a real [K, N_slice] q8_0 weight slice resident on the phone GPU and
// serves one request at a time: receive a K-wide f32 activation, run a real
// ggml mul_mat, return the N_slice f32 partial. This is the phone half of a
// single matmul split across an A6000 and two phones.
//
// usage: tp_worker <port> <N_slice> <K> <backend-substring>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <ctime>
#include <unistd.h>

static bool read_exact(int fd, void * dst, size_t n) {
    auto * p = static_cast<char *>(dst);
    while (n > 0) {
        ssize_t r = read(fd, p, n);
        if (r <= 0) return false;
        p += r; n -= (size_t) r;
    }
    return true;
}

static bool write_exact(int fd, const void * src, size_t n) {
    const auto * p = static_cast<const char *>(src);
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) return false;
        p += w; n -= (size_t) w;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <port> <N_slice> <K> <backend>\n", argv[0]);
        return 2;
    }
    const int     port = atoi(argv[1]);
    const int64_t N    = atoll(argv[2]);
    const int64_t K    = atoll(argv[3]);
    const std::string want = argv[4];
    signal(SIGPIPE, SIG_IGN);

    ggml_backend_t backend = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::string(ggml_backend_dev_name(dev)).find(want) != std::string::npos) {
            backend = ggml_backend_dev_init(dev, nullptr);
            fprintf(stderr, "[worker] backend %s (%s)\n",
                    ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
            break;
        }
    }
    if (!backend) { fprintf(stderr, "[worker] no backend matching '%s'\n", want.c_str()); return 1; }

    // context holds w, a, out plus the graph
    ggml_init_params ip = {};
    ip.mem_size   = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(4200, false);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);

    // argv[7]: weight type name (default q8_0). HTP prefers q4_0.
    ggml_type wtype = GGML_TYPE_Q8_0;
    if (argc > 7) {
        for (int t = 0; t < GGML_TYPE_COUNT; t++) {
            const char * nm = ggml_type_name((ggml_type) t);
            if (nm && std::string(nm) == argv[7]) { wtype = (ggml_type) t; break; }
        }
    }
    fprintf(stderr, "[worker] weight type %s\n", ggml_type_name(wtype));
    ggml_tensor * w   = ggml_new_tensor_2d(ctx, wtype, K, N);
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, 1);
    ggml_tensor * out = ggml_mul_mat(ctx, w, a);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "[worker] alloc failed (%lld x %lld)\n", (long long) K, (long long) N); return 1; }

    // real quantized weights
    {
        std::vector<float> src((size_t) K * N);
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto & v : src) v = dist(rng);
        std::vector<char> q(ggml_nbytes(w));
        ggml_quantize_chunk(wtype, src.data(), q.data(), 0, N, K, nullptr);
        ggml_backend_tensor_set(w, q.data(), 0, ggml_nbytes(w));
    }

    // optional argv[5]: duplicate the matmul REPS times in ONE graph. Lets us
    // separate the fixed dispatch latency (paid once per graph_compute) from
    // the marginal per-kernel cost (paid per node).
    const int reps = argc > 5 ? atoi(argv[5]) : 1;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, reps + 8, false);
    ggml_build_forward_expand(gf, out);
    for (int r = 1; r < reps; r++) ggml_graph_add_node(gf, out);
    fprintf(stderr, "[worker] graph nodes=%d\n", ggml_graph_n_nodes(gf));

    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t) port);
    if (bind(ls, (sockaddr *) &addr, sizeof addr) || listen(ls, 4)) { perror("bind"); return 1; }
    fprintf(stderr, "[worker] ready: N=%lld K=%lld port=%d\n", (long long) N, (long long) K, port);

    // per-stage timing so the fixed cost can be attributed to wire vs ggml
    auto now_us = []() {
        timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double) ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
    };

    // argv[6]: exit cleanly after this many requests so the OpenCL backend is
    // freed and GGML_OPENCL_PROFILING flushes cl_profiling.csv. 0 = serve forever.
    const long max_reqs = argc > 6 ? atol(argv[6]) : 0;
    long served = 0;

    std::vector<float> in((size_t) K), res((size_t) N);
    for (;;) {
        int cs = accept(ls, nullptr, nullptr);
        if (cs < 0) continue;
        setsockopt(cs, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        std::vector<double> t_set, t_cmp, t_get, t_snd, t_enq, t_wait;
        while (read_exact(cs, in.data(), sizeof(float) * K)) {
            const double t0 = now_us();
            ggml_backend_tensor_set(a, in.data(), 0, sizeof(float) * K);
            const double t1 = now_us();
            // split the dispatch: enqueue (submission) vs synchronize (wait for
            // completion). Only the second part could be replaced by polling a
            // flag, so this attributes the 586 us between the two.
            ggml_backend_graph_compute_async(backend, gf);
            const double t1b = now_us();
            ggml_backend_synchronize(backend);
            const double t2 = now_us();
            t_enq.push_back(t1b - t1); t_wait.push_back(t2 - t1b);
            ggml_backend_tensor_get(out, res.data(), 0, sizeof(float) * N);
            const double t3 = now_us();
            if (!write_exact(cs, res.data(), sizeof(float) * N)) break;
            const double t4 = now_us();
            if (max_reqs && ++served >= max_reqs) {
                fprintf(stderr, "[worker] served %ld, freeing backend for profile dump\n", served);
                close(cs);
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
                ggml_backend_free(backend);
                return 0;
            }
            t_set.push_back(t1 - t0); t_cmp.push_back(t2 - t1);
            t_get.push_back(t3 - t2); t_snd.push_back(t4 - t3);
            if (t_set.size() % 50 == 0) {
                auto med = [](std::vector<double> v) {
                    std::sort(v.begin(), v.end()); return v[v.size() / 2];
                };
                fprintf(stderr, "[worker] n=%zu rows=%lld | set %.0f | ENQUEUE %.0f | WAIT %.0f | get %.0f | send %.0f us\n",
                        t_set.size(), (long long) N, med(t_set), med(t_enq), med(t_wait), med(t_get), med(t_snd));
                fflush(stderr);
            }
        }
        close(cs);
    }
}
