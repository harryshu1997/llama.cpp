// One-transformer-layer latency: all-server vs FFN-offloaded-to-phones.
//
// Builds a real Qwen3-14B decode layer (M=1) on the local CUDA GPU:
//   attention: q[Nq,K] k[Nkv,K] v[Nkv,K] o[K,Nq]
//   ffn:       gate[Nff,K] up[Nff,K] down[K,Nff]  with silu*mul
//
// In split mode the GPU keeps only (Nff - sum(phone slices)) FFN columns; each
// phone owns a column slice and returns a row-split `down` partial that the
// host sums. Phone sends are issued before the GPU graph runs so the transfer
// overlaps GPU compute -- the same structure as tp_host.
//
// usage: layer_host <K> <Nq> <Nkv> <Nff> <iters> [port:rows ...]
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
#include <thread>
#include <vector>
#include <unistd.h>

// libusb, declared by hand (no -dev headers on this host). AOA mode talks to
// the accessory bulk endpoints directly: no adb server, no adbd, no loopback.
extern "C" {
struct libusb_context;
int  libusb_init(libusb_context **);
void libusb_exit(libusb_context *);
void * libusb_open_device_with_vid_pid(libusb_context *, unsigned short, unsigned short);
void libusb_close(void *);
int  libusb_claim_interface(void *, int);
int  libusb_release_interface(void *, int);
int  libusb_detach_kernel_driver(void *, int);
int  libusb_bulk_transfer(void *, unsigned char, unsigned char *, int, int *, unsigned int);
}

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

struct Phone { int port; int64_t rows; int fd = -1; std::vector<float> out; double last = 0;
               void * usb = nullptr; };   // usb != null => AOA bulk transport

static bool usb_xfer(void * h, unsigned char ep, void * data, int len) {
    int done = 0;
    while (done < len) {
        int n = 0;
        if (libusb_bulk_transfer(h, ep, (unsigned char *) data + done, len - done, &n, 3000) != 0)
            return false;
        done += n;
    }
    return true;
}

static bool rd(int fd, void * d, size_t n) {
    char * p = (char *) d;
    while (n) { ssize_t r = read(fd, p, n); if (r <= 0) return false; p += r; n -= r; }
    return true;
}
static bool wr(int fd, const void * s, size_t n) {
    const char * p = (const char *) s;
    while (n) { ssize_t w = write(fd, p, n); if (w <= 0) return false; p += w; n -= w; }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <K> <Nq> <Nkv> <Nff> <iters> [port:rows ...]\n", argv[0]);
        return 2;
    }
    const int64_t K = atoll(argv[1]), NQ = atoll(argv[2]);
    const int64_t NKV = atoll(argv[3]), NFF = atoll(argv[4]);
    const int iters = atoi(argv[5]);
    signal(SIGPIPE, SIG_IGN);

    std::vector<Phone> phones;
    for (int i = 6; i < argc; i++) {
        Phone p{}; long long r = 0;
        if (sscanf(argv[i], "%d:%lld", &p.port, &r) != 2) { fprintf(stderr, "bad %s\n", argv[i]); return 2; }
        p.rows = r; phones.push_back(p);
    }
    int64_t offloaded = 0;
    for (auto & p : phones) offloaded += p.rows;
    const int64_t NFF_GPU = NFF - offloaded;

    ggml_backend_t be = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            be = ggml_backend_dev_init(d, nullptr);
            fprintf(stderr, "[layer] gpu: %s\n", ggml_backend_dev_description(d));
            break;
        }
    }
    if (!be) { fprintf(stderr, "no gpu\n"); return 1; }

    ggml_init_params ip = {}; ip.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead() * 2;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);

    const ggml_type WT = GGML_TYPE_Q4_K;
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * wq = ggml_new_tensor_2d(ctx, WT, K, NQ);
    ggml_tensor * wk = ggml_new_tensor_2d(ctx, WT, K, NKV);
    ggml_tensor * wv = ggml_new_tensor_2d(ctx, WT, K, NKV);
    ggml_tensor * wo = ggml_new_tensor_2d(ctx, WT, NQ, K);
    ggml_tensor * wg = ggml_new_tensor_2d(ctx, WT, K, NFF_GPU);
    ggml_tensor * wu = ggml_new_tensor_2d(ctx, WT, K, NFF_GPU);
    ggml_tensor * wd = ggml_new_tensor_2d(ctx, WT, NFF_GPU, K);

    ggml_tensor * q = ggml_mul_mat(ctx, wq, x);
    ggml_tensor * k = ggml_mul_mat(ctx, wk, x);
    ggml_tensor * v = ggml_mul_mat(ctx, wv, x);
    ggml_tensor * ao = ggml_mul_mat(ctx, wo, q);          // stands in for attn output proj
    ggml_tensor * h  = ggml_add(ctx, x, ao);
    ggml_tensor * g  = ggml_mul_mat(ctx, wg, h);
    ggml_tensor * u  = ggml_mul_mat(ctx, wu, h);
    ggml_tensor * act = ggml_mul(ctx, ggml_silu(ctx, g), u);
    ggml_tensor * dn = ggml_mul_mat(ctx, wd, act);
    ggml_tensor * out = ggml_add(ctx, h, dn);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "[layer] alloc failed\n"); return 1; }
    {
        std::mt19937 rng(3); std::uniform_real_distribution<float> d(-1.f, 1.f);
        for (ggml_tensor * w : {wq, wk, wv, wo, wg, wu, wd}) {
            const int64_t rows = w->ne[1], cols = w->ne[0];
            std::vector<float> src((size_t) rows * cols);
            for (auto & z : src) z = d(rng);
            std::vector<char> qd(ggml_nbytes(w));
            ggml_quantize_chunk(WT, src.data(), qd.data(), 0, rows, cols, nullptr);
            ggml_backend_tensor_set(w, qd.data(), 0, ggml_nbytes(w));
        }
        std::vector<float> xv(K); for (auto & z : xv) z = d(rng);
        ggml_backend_tensor_set(x, xv.data(), 0, sizeof(float) * K);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    (void) k; (void) v;

    // port 0 selects AOA: open the accessory device and claim interface 0
    libusb_context * uctx = nullptr;
    for (auto & p : phones) {
        if (p.port == 0) {
            if (!uctx && libusb_init(&uctx) != 0) { fprintf(stderr, "libusb_init failed\n"); return 1; }
            for (unsigned short pid : {0x2d01, 0x2d00, 0x2d05, 0x2d04}) {
                p.usb = libusb_open_device_with_vid_pid(uctx, 0x18d1, pid);
                if (p.usb) { fprintf(stderr, "[layer] AOA device 18d1:%04x\n", pid); break; }
            }
            if (!p.usb) { fprintf(stderr, "[layer] no accessory device\n"); return 1; }
            libusb_detach_kernel_driver(p.usb, 0);
            if (libusb_claim_interface(p.usb, 0) != 0) { fprintf(stderr, "[layer] claim failed\n"); return 1; }
            p.out.resize(K);
            continue;
        }
        p.fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in sa = {}; sa.sin_family = AF_INET;
        sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK); sa.sin_port = htons(p.port);
        if (connect(p.fd, (sockaddr *) &sa, sizeof sa) < 0) {
            fprintf(stderr, "[layer] connect :%d failed\n", p.port); return 1;
        }
        int one = 1; setsockopt(p.fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        p.out.resize(K);
    }

    std::vector<float> xhost(K, 0.1f), res(K);
    auto iter = [&]() {
        std::vector<std::thread> th;
        for (auto & p : phones) {
            th.emplace_back([&p, &xhost, K]() {
                auto t0 = clk::now();
                if (p.usb) {
                    if (!usb_xfer(p.usb, 0x01, (void *) xhost.data(), sizeof(float) * K)) { p.last = -1; return; }
                    if (!usb_xfer(p.usb, 0x81, p.out.data(), sizeof(float) * K)) { p.last = -1; return; }
                } else {
                    if (!wr(p.fd, xhost.data(), sizeof(float) * K)) { p.last = -1; return; }
                    if (!rd(p.fd, p.out.data(), sizeof(float) * K)) { p.last = -1; return; }
                }
                p.last = ms_since(t0);
            });
        }
        ggml_backend_graph_compute(be, gf);
        ggml_backend_tensor_get(out, res.data(), 0, sizeof(float) * K);
        for (auto & t : th) t.join();
        for (auto & p : phones)            // all-reduce the row-split down partials
            for (int64_t i = 0; i < K; i++) res[i] += p.out[i];
    };

    for (int i = 0; i < 10; i++) iter();
    std::vector<double> w;
    for (int i = 0; i < iters; i++) { auto t0 = clk::now(); iter(); w.push_back(ms_since(t0)); }
    std::sort(w.begin(), w.end());
    printf("%-42s %8.3f ms  (min %.3f, p90 %.3f)\n",
           phones.empty() ? "ALL-SERVER one layer" : "SPLIT: attn+partial FFN on GPU",
           w[w.size() / 2], w.front(), w[(size_t) (0.9 * w.size())]);
    if (!phones.empty()) {
        printf("   GPU keeps %lld/%lld FFN cols (%.0f%%); phones:", (long long) NFF_GPU,
               (long long) NFF, 100.0 * NFF_GPU / NFF);
        for (auto & p : phones) printf("  :%d %lld rows %.3fms", p.port, (long long) p.rows, p.last);
        printf("\n");
    }
    for (auto & p : phones) {
        if (p.usb) { libusb_release_interface(p.usb, 0); libusb_close(p.usb); }
        else close(p.fd);
    }
    if (uctx) libusb_exit(uctx);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    return 0;
}
