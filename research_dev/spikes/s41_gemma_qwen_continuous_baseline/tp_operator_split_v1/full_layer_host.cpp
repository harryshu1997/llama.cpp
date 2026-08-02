// LEGACY INVALID BENCHMARK. Do not use its latency or energy results.
//
// The phone starts from a constant input before CUDA produces the causal FFN
// input, and the two devices initialize unrelated weights. Use
// causal_layer_host.cpp and causal_ffn_worker.cpp instead.
//
// COMPLETE Qwen3-14B decode layer: all-server vs FFN-offloaded-to-OP15.
//
// Unlike layer_host (which measured only the matmul skeleton and silently
// dropped k/v), this builds every weight-touching op of a real layer:
//
//   attn_norm -> q,k,v proj -> GQA attention over a [n_kv] KV cache -> o proj
//   -> residual -> ffn_norm -> gate,up -> silu*mul -> down -> residual
//
// n_head=40, n_head_kv=8 (GQA), head_dim=128, n_embd=5120, n_ff=17408.
// RoPE is omitted: it touches no weights and is a fixed ~microsecond cost on
// both sides, so it cannot change the offload balance.
//
// usage: full_layer_host <n_kv> <iters> [0:cols] [K NH NHKV HD NFF type]
//   defaults are Qwen3-14B; pass the 6 extra args for Gemma-4-12B:
//     5120 40 8 128 17408 q4_K   (qwen3-14b)
//     3840 16 8 256 15360 q8_0   (gemma-4-12b standard layer)
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
#include <string>
#include <unistd.h>

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
static bool usb_xfer(void * h, unsigned char ep, void * d, int len) {
    int done = 0;
    while (done < len) {
        int n = 0;
        if (libusb_bulk_transfer(h, ep, (unsigned char *) d + done, len - done, &n, 4000) != 0) return false;
        done += n;
    }
    return true;
}

int main(int argc, char ** argv) {
    const int64_t n_kv  = argc > 1 ? atoll(argv[1]) : 512;
    const int     iters = argc > 2 ? atoi(argv[2]) : 100;
    int64_t phone_cols  = 0;
    if (argc > 3) { long long c = 0; sscanf(argv[3], "0:%lld", &c); phone_cols = c; }
    signal(SIGPIPE, SIG_IGN);

    int64_t K = 5120, NH = 40, NHKV = 8, HD = 128, NFF = 17408;
    ggml_type WT = GGML_TYPE_Q4_K;
    if (argc > 9) {
        K = atoll(argv[4]); NH = atoll(argv[5]); NHKV = atoll(argv[6]);
        HD = atoll(argv[7]); NFF = atoll(argv[8]);
        for (int t = 0; t < GGML_TYPE_COUNT; t++) {
            const char * nm = ggml_type_name((ggml_type) t);
            if (nm && std::string(nm) == argv[9]) { WT = (ggml_type) t; break; }
        }
    }
    const int64_t NQ = NH * HD, NKV_P = NHKV * HD;
    const int64_t NFF_GPU = NFF - phone_cols;

    ggml_backend_t be = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            be = ggml_backend_dev_init(d, nullptr);
            fprintf(stderr, "[full] gpu: %s\n", ggml_backend_dev_description(d));
            break;
        }
    }
    if (!be) return 1;

    ggml_init_params ip = {};
    ip.mem_size = ggml_tensor_overhead() * 128 + ggml_graph_overhead() * 2;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * an  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    ggml_tensor * fn  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    ggml_tensor * wq  = ggml_new_tensor_2d(ctx, WT, K, NQ);
    ggml_tensor * wk  = ggml_new_tensor_2d(ctx, WT, K, NKV_P);
    ggml_tensor * wv  = ggml_new_tensor_2d(ctx, WT, K, NKV_P);
    ggml_tensor * wo  = ggml_new_tensor_2d(ctx, WT, NQ, K);
    ggml_tensor * wg  = ggml_new_tensor_2d(ctx, WT, K, NFF_GPU);
    ggml_tensor * wu  = ggml_new_tensor_2d(ctx, WT, K, NFF_GPU);
    ggml_tensor * wd  = ggml_new_tensor_2d(ctx, WT, NFF_GPU, K);
    // KV cache for this layer, f16, [head_dim, n_kv, n_head_kv]
    ggml_tensor * kc  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, HD, n_kv, NHKV);
    ggml_tensor * vc  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_kv, HD, NHKV);

    // ---- attention ----
    ggml_tensor * xn = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), an);
    ggml_tensor * q  = ggml_mul_mat(ctx, wq, xn);            // [NQ,1]
    ggml_tensor * kk = ggml_mul_mat(ctx, wk, xn);            // [NKV_P,1]
    ggml_tensor * vv = ggml_mul_mat(ctx, wv, xn);            // [NKV_P,1]
    // q -> [HD, 1, NH] so mul_mat broadcasts n_head_kv -> n_head (GQA)
    ggml_tensor * qh = ggml_permute(ctx, ggml_reshape_3d(ctx, q, HD, NH, 1), 0, 2, 1, 3);
    ggml_tensor * kq = ggml_mul_mat(ctx, kc, qh);            // [n_kv, 1, NH]
    kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / sqrtf((float) HD), 0.0f);
    ggml_tensor * kqv = ggml_mul_mat(ctx, vc, kq);           // [HD, 1, NH]
    ggml_tensor * am  = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), NQ, 1);
    ggml_tensor * ao  = ggml_mul_mat(ctx, wo, am);
    // keep k/v projections live so they are actually executed
    ggml_tensor * h   = ggml_add(ctx, x, ao);
    h = ggml_add(ctx, h, ggml_scale(ctx, ggml_pad(ctx, ggml_add(ctx, kk, vv), K - NKV_P, 0, 0, 0), 0.0f));

    // ---- ffn ----
    ggml_tensor * hn  = ggml_mul(ctx, ggml_rms_norm(ctx, h, 1e-6f), fn);
    ggml_tensor * g   = ggml_mul_mat(ctx, wg, hn);
    ggml_tensor * u   = ggml_mul_mat(ctx, wu, hn);
    ggml_tensor * act = ggml_mul(ctx, ggml_silu(ctx, g), u);
    ggml_tensor * dn  = ggml_mul_mat(ctx, wd, act);
    ggml_tensor * out = ggml_add(ctx, h, dn);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "[full] alloc failed\n"); return 1; }
    {
        std::mt19937 rng(5); std::uniform_real_distribution<float> d(-1.f, 1.f);
        for (ggml_tensor * w : {wq, wk, wv, wo, wg, wu, wd}) {
            const int64_t rows = w->ne[1], cols = w->ne[0];
            std::vector<float> src((size_t) rows * cols);
            for (auto & z : src) z = d(rng);
            std::vector<char> qd(ggml_nbytes(w));
            ggml_quantize_chunk(WT, src.data(), qd.data(), 0, rows, cols, nullptr);
            ggml_backend_tensor_set(w, qd.data(), 0, ggml_nbytes(w));
        }
        std::vector<float> ones(K, 1.0f);
        ggml_backend_tensor_set(an, ones.data(), 0, sizeof(float) * K);
        ggml_backend_tensor_set(fn, ones.data(), 0, sizeof(float) * K);
        std::vector<float> xv(K); for (auto & z : xv) z = d(rng);
        ggml_backend_tensor_set(x, xv.data(), 0, sizeof(float) * K);
        std::vector<uint16_t> kz(ggml_nelements(kc), 0x3400), vz(ggml_nelements(vc), 0x3400);
        ggml_backend_tensor_set(kc, kz.data(), 0, ggml_nbytes(kc));
        ggml_backend_tensor_set(vc, vz.data(), 0, ggml_nbytes(vc));
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    fprintf(stderr, "[full] graph nodes=%d n_kv=%lld K=%lld NFF=%lld type=%s gpu_ffn_cols=%lld/%lld\n",
            ggml_graph_n_nodes(gf), (long long) n_kv, (long long) K, (long long) NFF,
            ggml_type_name(WT), (long long) NFF_GPU, (long long) NFF);

    void * usb = nullptr; libusb_context * uctx = nullptr;
    if (phone_cols > 0) {
        if (libusb_init(&uctx) != 0) return 1;
        for (unsigned short pid : {0x2d01, 0x2d00, 0x2d05, 0x2d04}) {
            usb = libusb_open_device_with_vid_pid(uctx, 0x18d1, pid);
            if (usb) { fprintf(stderr, "[full] AOA 18d1:%04x\n", pid); break; }
        }
        if (!usb) { fprintf(stderr, "[full] no accessory device\n"); return 1; }
        libusb_detach_kernel_driver(usb, 0);
        if (libusb_claim_interface(usb, 0) != 0) { fprintf(stderr, "[full] claim failed\n"); return 1; }
    }

    std::vector<float> xhost(K, 0.05f), res(K), pout(K);
    double plast = 0;
    auto iter = [&]() {
        std::thread th;
        if (usb) th = std::thread([&]() {
            auto t0 = clk::now();
            // A failed bulk transfer must be loud: silently ignoring it makes a
            // dead phone look like a 0.01 ms round trip and invalidates the run.
            if (!usb_xfer(usb, 0x01, (void *) xhost.data(), sizeof(float) * K) ||
                !usb_xfer(usb, 0x81, pout.data(), sizeof(float) * K)) {
                fprintf(stderr, "[full] FATAL: phone bulk transfer failed\n");
                _exit(3);
            }
            plast = ms_since(t0);
        });
        ggml_backend_graph_compute(be, gf);
        ggml_backend_tensor_get(out, res.data(), 0, sizeof(float) * K);
        if (th.joinable()) th.join();
        if (usb) for (int64_t i = 0; i < K; i++) res[i] += pout[i];
    };

    for (int i = 0; i < 10; i++) iter();
    std::vector<double> w;
    for (int i = 0; i < iters; i++) { auto t0 = clk::now(); iter(); w.push_back(ms_since(t0)); }
    std::sort(w.begin(), w.end());
    printf("%-34s %8.3f ms  (min %.3f, p90 %.3f)",
           phone_cols ? "FULL LAYER split (GPU+OP15)" : "FULL LAYER all-server",
           w[w.size() / 2], w.front(), w[(size_t) (0.9 * w.size())]);
    if (phone_cols) printf("   phone %lld cols %.3f ms", (long long) phone_cols, plast);
    printf("\n");

    if (usb) { libusb_release_interface(usb, 0); libusb_close(usb); libusb_exit(uctx); }
    ggml_backend_buffer_free(buf); ggml_free(ctx);
    return 0;
}
