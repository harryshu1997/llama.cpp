// mtmd-sched: the VQ scheduler on-device. A CPU dispatcher pulls requests from a
// VIRTUAL QUEUE and PLANS each one onto a backend, then admits it only while that
// backend's in-flight count < K (CONWIP WIP-cap). One worker thread per backend =
// one in-order hardware queue; a monitor logs the queue depths -> vq_sched_series.csv.
//
// Planning (SCHED_HEFT=1, default): a request that supports >1 backend is assigned
// by HEFT — min estimated-finish-time EFT(b) = max(now, busy_until[b]) + cost[kind][b],
// over supported backends that have a free slot; cost is an online EMA of measured
// run times (same idea as vq_dispatcher.heft_schedule + CostModel). SCHED_HEFT=0 =
// fixed baseline (vision->GPU, decode->NPU).
//
// NPU-CAPACITY ARBITRATION (the point of this harness):
//   vision-encode supports {GPU(Adreno 913ms), NPU(HTP 769ms — FASTER, commit 1180672db)}
//   text-decode   supports {NPU(HTP)}
//   The NPU is ONE shared in-order queue: busy_until[NPU] accumulates BOTH vision and
//   decode, so vision@NPU and decode@NPU SERIALISE on it (physically correct — one
//   Hexagon). HEFT then arbitrates with NO new policy: when decode keeps the NPU busy,
//   EFT(vision@GPU) < EFT(vision@NPU) -> vision routed to GPU (Config A, the 1.85x
//   vision∥decode overlap); when the NPU has headroom, vision@NPU wins (Config B, the
//   faster encode). decode-heavy => A, vision-heavy/decode-light => B; the crossover is
//   emergent from the measured costs.
//
// Single-residency: a clip context is pinned to one backend at load (a 2nd segfaults).
// The clip here is GPU-resident, so vision@GPU is REAL (mtmd_encode_chunk) and
// vision@NPU is MODELED (a measured-cost sleep on the NPU worker, so it still contends
// with real decode on the shared NPU queue). decode@NPU is REAL. The HEFT routing
// DECISION and the NPU contention are real; only the non-resident vision execution is modeled.
// The first vision@NPU job also pays a one-time REPACK tax (clip weights Q4_0->q4x4x2).
//
// Env: SCHED_NENC(6) SCHED_NDEC(40) SCHED_KGPU(1) SCHED_KNPU(2) SCHED_HEFT(1)
//      SCHED_PARTITION(1) SCHED_OUT(vq_sched_series.csv)
//      SCHED_GPU_VIS_MS(913) SCHED_NPU_VIS_MS(769) SCHED_DEC_MS(95) SCHED_REPACK_MS(300)
// MTMD_BACKEND_DEVICE=GPUOpenCL picks the GPU vision backend; --device HTP0 the text.
#include "arg.h"
#include "log.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <vector>
#include <deque>
#include <array>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#if defined(__unix__) || defined(__ANDROID__)
#include <sched.h>
#endif

static double now_ms() { return (double) ggml_time_us() / 1000.0; }
static void pin(int lo, int hi) {
#if defined(__unix__) || defined(__ANDROID__)
    cpu_set_t s; CPU_ZERO(&s); for (int c = lo; c <= hi; ++c) CPU_SET(c, &s);
    sched_setaffinity(0, sizeof(s), &s);
#else
    (void) lo; (void) hi;
#endif
}

enum BK { BK_GPU = 0, BK_NPU = 1, BK_N = 2 };
static const char * BKNAME[BK_N] = { "GPU", "NPU" };
enum Kind { ENCODE = 0, DECODE = 1 };
struct Req { int id; Kind kind; };

struct BackendQ {
    std::mutex m; std::condition_variable cv;
    std::deque<Req> jobs;
    std::atomic<int> inflight{0};
    int K = 1;
    bool stop = false;
    std::atomic<int> enc_done{0};   // vision encodes completed on this backend
    std::atomic<int> dec_done{0};   // decodes completed on this backend
};

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD)) return 1;
    if (params.mmproj.path.empty() || params.image.empty()) { LOG_ERR("need -m --mmproj --image\n"); return 1; }
    auto envi = [](const char* k, int d){ const char* e=std::getenv(k); return e?atoi(e):d; };
    const int  N_ENC = envi("SCHED_NENC", 6),  N_DEC = envi("SCHED_NDEC", 40);
    const int  K_GPU = envi("SCHED_KGPU", 1),  K_NPU = envi("SCHED_KNPU", 2);
    const bool HEFT  = envi("SCHED_HEFT", 1) != 0;
    const bool PART  = envi("SCHED_PARTITION", 1) != 0;
    const int  GPU_VIS_MS = envi("SCHED_GPU_VIS_MS", 913);  // measured vision@Adreno (same-boot 2026-06-09)
    const int  NPU_VIS_MS = envi("SCHED_NPU_VIS_MS", 769);  // measured vision@HTP   (same-boot, commit 1180672db)
    const int  DEC_MS     = envi("SCHED_DEC_MS", 95);       // ~per-decode-step prior (10.5 tok/s; real EMA refines)
    const int  REPACK_MS  = envi("SCHED_REPACK_MS", 300);   // one-time clip-weight Q4_0->q4x4x2 (CPU-concurrent, hideable)
    const char * OUT = std::getenv("SCHED_OUT") ? std::getenv("SCHED_OUT") : "vq_sched_series.csv";

    ggml_time_init(); common_init(); ggml_backend_load_all();
    common_init_result_ptr init = common_init_from_params(params);
    llama_model   * model = init ? init->model()   : nullptr;
    llama_context * lctx  = init ? init->context() : nullptr;
    if (!model || !lctx) { LOG_ERR("model load failed\n"); return 1; }

    auto mkv = [&](bool gpu){
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu = gpu; mp.print_timings = false; mp.n_threads = params.cpuparams.n_threads;
        mp.flash_attn_type = params.flash_attn_type; mp.warmup = params.warmup;
        return mtmd_init_from_file(params.mmproj.path.c_str(), model, mp);
    };
    mtmd_context * vgpu = mkv(true);                       // Adreno clip (single-residency)
    if (!vgpu) { LOG_ERR("vision load failed\n"); return 1; }

    auto tok = [&](mtmd_context * v) -> const mtmd_input_chunk * {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(v, params.image[0].c_str()));
        if (!bmp.ptr) return nullptr;
        std::vector<const mtmd_bitmap *> bmps = { bmp.ptr.get() };
        std::string prompt = std::string(mtmd_default_marker()) + "Describe this image.";
        mtmd_input_text text{ prompt.c_str(), true, true };
        auto * ch = mtmd_input_chunks_init();
        if (mtmd_tokenize(v, ch, &text, bmps.data(), bmps.size())) return nullptr;
        for (size_t i = 0; i < mtmd_input_chunks_size(ch); ++i) {
            const mtmd_input_chunk * c = mtmd_input_chunks_get(ch, i);
            if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) return c;
        }
        return nullptr;
    };
    const mtmd_input_chunk * img_gpu = tok(vgpu);
    if (!img_gpu) { LOG_ERR("tokenize failed\n"); return 1; }
    { llama_batch b = llama_batch_init(1,0,1); b.n_tokens=1; b.token[0]=llama_vocab_bos(llama_model_get_vocab(model));
      b.pos[0]=0; b.n_seq_id[0]=1; b.seq_id[0][0]=0; b.logits[0]=true; llama_decode(lctx,b); llama_batch_free(b); }
    std::atomic<llama_pos> dpos{1};

    // virtual queue: interleave encode + decode requests, all present at t=0 (burst)
    std::deque<Req> vq; std::mutex vqm;
    { int ie=0, idc=0, rid=0; while (ie<N_ENC || idc<N_DEC) {
        if (ie<N_ENC) { vq.push_back({rid++, ENCODE}); ie++; }
        for (int k=0;k<7 && idc<N_DEC; ++k){ vq.push_back({rid++, DECODE}); idc++; } } }
    const int TOTAL = (int) vq.size();

    BackendQ bk[BK_N]; bk[BK_GPU].K=K_GPU; bk[BK_NPU].K=K_NPU;
    std::atomic<int> done{0};
    std::condition_variable sched_cv; std::mutex sched_m;
    std::atomic<bool> npu_vis_warm{false};   // first vision@NPU pays the one-time repack tax
    // online cost model (ms) per (kind, backend); seeded with measured priors, refined by EMA
    double cost[2][BK_N];
    cost[ENCODE][BK_GPU] = GPU_VIS_MS; cost[ENCODE][BK_NPU] = NPU_VIS_MS;
    cost[DECODE][BK_GPU] = 1e9;        cost[DECODE][BK_NPU] = DEC_MS;   // decode is NPU-only
    double busy_until[BK_N] = {0,0};
    std::mutex costm;

    auto run_job = [&](BK b, Kind k){
        double t = now_ms();
        if (k == DECODE) {                                                  // REAL text decode @ NPU
            llama_batch bt = llama_batch_init(1,0,1); bt.n_tokens=1;
            bt.token[0]=llama_vocab_bos(llama_model_get_vocab(model));
            bt.pos[0]=dpos.fetch_add(1); bt.n_seq_id[0]=1; bt.seq_id[0][0]=0; bt.logits[0]=true;
            llama_decode(lctx, bt); llama_batch_free(bt);
        } else if (b == BK_GPU) {                                           // REAL vision encode @ Adreno
            mtmd_encode_chunk(vgpu, img_gpu);
        } else {                                                            // MODELED vision encode @ HTP
            int ms = NPU_VIS_MS;                                            // (clip is GPU-resident; single-residency)
            if (!npu_vis_warm.exchange(true)) ms += REPACK_MS;              // one-time clip-weight repack
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        }
        double dur = now_ms() - t;
        { std::lock_guard<std::mutex> lk(costm); cost[k][b] = 0.7*cost[k][b] + 0.3*dur; }  // EMA
    };
    auto worker = [&](BK b, int lo, int hi){
        if (PART) pin(lo, hi);
        for (;;) {
            Req job;
            { std::unique_lock<std::mutex> lk(bk[b].m);
              bk[b].cv.wait(lk, [&]{ return !bk[b].jobs.empty() || bk[b].stop; });
              if (bk[b].jobs.empty() && bk[b].stop) return;
              job = bk[b].jobs.front(); bk[b].jobs.pop_front(); }
            run_job(b, job.kind);
            (job.kind == ENCODE ? bk[b].enc_done : bk[b].dec_done).fetch_add(1);
            bk[b].inflight.fetch_sub(1); done.fetch_add(1);
            { std::lock_guard<std::mutex> lk(sched_m); } sched_cv.notify_one();
        }
    };
    // GPU vision on the compute half (cores 4-7); NPU FastRPC on 0-3
    fprintf(stderr, "[sched] prefill done, starting 2 workers + dispatch (%d reqs)\n", TOTAL); fflush(stderr);
    std::thread tg(worker, BK_GPU, 4, 7);
    std::thread tn(worker, BK_NPU, 0, 3);

    std::atomic<bool> run{true};
    std::vector<std::array<double,4>> series;   // t, virtual, gpu, npu
    double t0 = now_ms();
    std::thread mon([&]{
        while (run.load()) {
            size_t v; { std::lock_guard<std::mutex> lk(vqm); v = vq.size(); }
            series.push_back({ now_ms()-t0, (double)v,
                (double)bk[BK_GPU].inflight.load(), (double)bk[BK_NPU].inflight.load() });
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    // supported backends per kind
    auto supported = [&](Kind k) -> std::vector<BK> {
        if (k == DECODE) return { BK_NPU };
        return HEFT ? std::vector<BK>{ BK_GPU, BK_NPU } : std::vector<BK>{ BK_GPU };  // vision: GPU vs NPU
    };

    // dispatcher: HEFT-pick a backend with a free slot, by min EFT; admit it
    while (done.load() < TOTAL) {
        bool progress = false;
        { std::lock_guard<std::mutex> lk(vqm);
          for (auto it = vq.begin(); it != vq.end(); ) {
            BK best = BK_N; double best_eft = 1e18;
            double tnow = now_ms();
            { std::lock_guard<std::mutex> cl(costm);
              for (BK b : supported(it->kind)) {
                if (bk[b].inflight.load() >= bk[b].K) continue;            // no free slot -> can't admit now
                double c = cost[it->kind][b];
                if (it->kind == ENCODE && b == BK_NPU && !npu_vis_warm.load()) c += REPACK_MS;  // one-time tax
                double eft = std::max(tnow, busy_until[b]) + c;            // HEFT earliest-finish (shared NPU)
                if (eft < best_eft) { best_eft = eft; best = b; }
              }
              if (best != BK_N) {
                double c = cost[it->kind][best];
                if (it->kind == ENCODE && best == BK_NPU && !npu_vis_warm.load()) c += REPACK_MS;
                busy_until[best] = std::max(tnow, busy_until[best]) + c;
              }
            }
            if (best != BK_N) {
                bk[best].inflight.fetch_add(1);
                { std::lock_guard<std::mutex> wl(bk[best].m); bk[best].jobs.push_back(*it); } bk[best].cv.notify_one();
                it = vq.erase(it); progress = true;
            } else ++it;
          } }
        if (!progress) { std::unique_lock<std::mutex> lk(sched_m);
            sched_cv.wait_for(lk, std::chrono::milliseconds(5), [&]{ return done.load() >= TOTAL; }); }
    }
    run.store(false); mon.join();
    for (int b=0;b<BK_N;++b){ { std::lock_guard<std::mutex> lk(bk[b].m); bk[b].stop=true; } bk[b].cv.notify_all(); }
    tg.join(); tn.join();

    double mx[3]={0,0,0};
    FILE * f = fopen(OUT, "w");
    if (f) fprintf(f, "t_ms,virtual,gpu_real,npu_real\n");
    for (auto & r : series) { if (f) fprintf(f, "%.1f,%.0f,%.0f,%.0f\n", r[0],r[1],r[2],r[3]);
        for (int j=0;j<3;++j) mx[j]=std::max(mx[j], r[1+j]); }
    if (f) fclose(f);
    const int enc_gpu = bk[BK_GPU].enc_done.load(), enc_npu = bk[BK_NPU].enc_done.load();
    LOG("\n============ vq scheduler — NPU-capacity arbitration (%s) ============\n", HEFT?"HEFT":"fixed");
    LOG("requests     : %d (%d encode + %d decode)\n", TOTAL, N_ENC, N_DEC);
    LOG("caps         : K_gpu=%d K_npu=%d  partition=%d\n", K_GPU,K_NPU,(int)PART);
    LOG("costs(ms)    : vision@GPU=%d vision@NPU=%d decode@NPU=%d repack(1x)=%d\n", GPU_VIS_MS,NPU_VIS_MS,DEC_MS,REPACK_MS);
    LOG("note         : vision@GPU + decode@NPU are REAL; vision@NPU is MODELED (single-residency); NPU queue shared\n");
    LOG("virtual peak : %.0f\n", mx[0]);
    LOG("VISION SPLIT : GPU=%d  NPU=%d   (decode: NPU=%d)\n", enc_gpu, enc_npu, bk[BK_NPU].dec_done.load());
    LOG("  -> %s\n", enc_npu==0 ? "all vision on GPU (Config A: vision∥decode overlap — NPU saturated by decode)"
                   : enc_gpu==0 ? "all vision on NPU (Config B: faster encode — NPU had headroom)"
                                : "MIXED — scheduler spilled vision across both as NPU pressure varied");
    for (int b=0;b<BK_N;++b)
        LOG("%s real max   : %.0f (cap %d) %s\n", BKNAME[b], mx[1+b], bk[b].K, mx[1+b]<=bk[b].K?"OK":"OVER");
    LOG("learned cost : vision@GPU=%.0f vision@NPU=%.0f decode@NPU=%.0f ms\n",
        cost[ENCODE][BK_GPU], cost[ENCODE][BK_NPU], cost[DECODE][BK_NPU]);
    LOG("wall         : %.0f ms,  series -> %s (%zu samples)\n", now_ms()-t0, OUT, series.size());
    LOG("====================================================================\n");
    mtmd_free(vgpu);
    return 0;
}
