// Fused-FFN worker: runs a whole FFN block slice in ONE dispatch.
//
// The Hexagon backend enqueues every graph node into one op-batch and flushes
// once, so a chain of DEPENDENT ops costs a single CPU<->DSP round trip. This
// worker builds the Megatron-style FFN column slice
//
//     gate = W_gate_slice @ x        [Nff_slice]
//     up   = W_up_slice   @ x        [Nff_slice]
//     act  = silu(gate) * up
//     out  = W_down_slice @ act      [K]   (row-split partial)
//
// so one activation round trip covers 3 matmuls + 2 elementwise ops. Compare
// its batch-dur against a single matmul's to measure dispatch amortization.
//
// usage: ffn_worker <port> <K> <Nff_slice> <backend> <type> [max_reqs]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <ctime>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

static bool read_exact(int fd, void * d, size_t n) {
    char * p = (char *) d;
    while (n) { ssize_t r = read(fd, p, n); if (r <= 0) return false; p += r; n -= r; }
    return true;
}
static bool write_exact(int fd, const void * s, size_t n) {
    const char * p = (const char *) s;
    while (n) { ssize_t w = write(fd, p, n); if (w <= 0) return false; p += w; n -= w; }
    return true;
}
static double now_us() {
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <port> <K> <Nff_slice> <backend> <type> [max_reqs]\n", argv[0]);
        return 2;
    }
    const int     port = atoi(argv[1]);
    const int64_t K    = atoll(argv[2]);
    const int64_t NF   = atoll(argv[3]);
    const std::string want = argv[4];
    const long max_reqs = argc > 6 ? atol(argv[6]) : 0;
    signal(SIGPIPE, SIG_IGN);

    ggml_type wt = GGML_TYPE_Q4_0;
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * nm = ggml_type_name((ggml_type) t);
        if (nm && std::string(nm) == argv[5]) { wt = (ggml_type) t; break; }
    }

    ggml_backend_t backend = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (std::string(ggml_backend_dev_name(d)).find(want) != std::string::npos) {
            backend = ggml_backend_dev_init(d, nullptr);
            fprintf(stderr, "[ffn] backend %s (%s) type %s\n", ggml_backend_dev_name(d),
                    ggml_backend_dev_description(d), ggml_type_name(wt));
            break;
        }
    }
    if (!backend) { fprintf(stderr, "[ffn] no backend '%s'\n", want.c_str()); return 1; }

    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * wg = ggml_new_tensor_2d(ctx, wt, K,  NF);   // gate slice
    ggml_tensor * wu = ggml_new_tensor_2d(ctx, wt, K,  NF);   // up slice
    ggml_tensor * wd = ggml_new_tensor_2d(ctx, wt, NF, K);    // down slice (row-split)
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);

    ggml_tensor * g   = ggml_mul_mat(ctx, wg, x);
    ggml_tensor * u   = ggml_mul_mat(ctx, wu, x);
    ggml_tensor * act = ggml_mul(ctx, ggml_silu(ctx, g), u);
    ggml_tensor * out = ggml_mul_mat(ctx, wd, act);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "[ffn] alloc failed\n"); return 1; }

    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        for (ggml_tensor * w : {wg, wu, wd}) {
            const int64_t rows = w->ne[1], cols = w->ne[0];
            std::vector<float> src((size_t) rows * cols);
            for (auto & v : src) v = dist(rng);
            std::vector<char> q(ggml_nbytes(w));
            ggml_quantize_chunk(wt, src.data(), q.data(), 0, rows, cols, nullptr);
            ggml_backend_tensor_set(w, q.data(), 0, ggml_nbytes(w));
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    fprintf(stderr, "[ffn] graph nodes=%d (3 matmuls + silu + mul)\n", ggml_graph_n_nodes(gf));

    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a = {}; a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(port);
    if (bind(ls, (sockaddr *) &a, sizeof a) || listen(ls, 4)) { perror("bind"); return 1; }
    fprintf(stderr, "[ffn] ready: K=%lld Nff_slice=%lld port=%d\n",
            (long long) K, (long long) NF, port);

    std::vector<float> in(K), res(K);
    long served = 0;
    // AOA mode: the accessory endpoint replaces the socket entirely, removing
    // the adb server, adbd and both TCP loopback hops.
    const bool aoa = getenv("FFN_AOA") != nullptr;
    for (;;) {
        int cs;
        if (aoa) {
            cs = open("/dev/usb_accessory", O_RDWR);
            if (cs < 0) { fprintf(stderr, "[ffn] accessory open: %s\n", strerror(errno)); sleep(1); continue; }
            fprintf(stderr, "[ffn] AOA endpoint open\n"); fflush(stderr);
        } else {
            cs = accept(ls, nullptr, nullptr);
        }
        if (cs < 0) continue;
        if (!aoa) setsockopt(cs, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        std::vector<double> t_all;
        while (read_exact(cs, in.data(), sizeof(float) * K)) {
            const double t0 = now_us();
            ggml_backend_tensor_set(x, in.data(), 0, sizeof(float) * K);
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_tensor_get(out, res.data(), 0, sizeof(float) * K);
            t_all.push_back(now_us() - t0);
            if (!write_exact(cs, res.data(), sizeof(float) * K)) break;
            if (max_reqs && ++served >= max_reqs) {
                ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(backend);
                return 0;
            }
            if (t_all.size() % 50 == 0) {
                std::vector<double> v = t_all; std::sort(v.begin(), v.end());
                fprintf(stderr, "[ffn] n=%zu  device-side median %.0f us\n",
                        t_all.size(), v[v.size() / 2]);
                fflush(stderr);
            }
        }
        close(cs);
    }
}
