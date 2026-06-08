// mtmd-sched: the VQ scheduler on-device. A CPU dispatcher pulls requests from a
// VIRTUAL QUEUE and PLANS each one onto a backend, then admits it only while that
// backend's in-flight count < K (CONWIP WIP-cap). One worker thread per backend =
// one in-order hardware queue; a monitor logs the queue depths -> vq_sched_series.csv.
//
// Planning (SCHED_HEFT=1, default): a request that supports >1 backend is assigned
// by HEFT — min estimated-finish-time EFT(b) = max(now, busy_until[b]) + cost[b],
// over supported backends that have a free slot; cost[b] is an online EMA of
// measured run times (same idea as vq_dispatcher.heft_schedule + CostModel). With
// SCHED_HEFT=0 the assignment is fixed (encode->GPU, decode->NPU) as a baseline.
//
// Here vision-encode supports {GPU(Adreno), CPU}; text-decode supports {NPU(HTP)}.
// HEFT load-balances encode across GPU and CPU when the GPU queue backs up. NOTE:
// a 2nd (CPU) clip context segfaults on this stack (a llama context is single-
// residency), so CPU encode is MODELED (a measured-cost sleep). The HEFT routing
// DECISION is real (real queue depths + real GPU timings); only CPU *execution* is modeled.
//
// Env: SCHED_NENC(6) SCHED_NDEC(40) SCHED_KGPU(1) SCHED_KCPU(1) SCHED_KNPU(2)
//      SCHED_HEFT(1) SCHED_PARTITION(1) SCHED_OUT(vq_sched_series.csv)
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

enum BK { BK_GPU = 0, BK_CPU = 1, BK_NPU = 2, BK_N = 3 };
static const char * BKNAME[BK_N] = { "GPU", "CPU", "NPU" };
enum Kind { ENCODE, DECODE };
struct Req { int id; Kind kind; };

struct BackendQ {
    std::mutex m; std::condition_variable cv;
    std::deque<int> jobs;
    std::atomic<int> inflight{0};
    int K = 1;
    bool stop = false;
    std::atomic<int> done_cnt{0};   // jobs completed on this backend (for the report)
};

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD)) return 1;
    if (params.mmproj.path.empty() || params.image.empty()) { LOG_ERR("need -m --mmproj --image\n"); return 1; }
    auto envi = [](const char* k, int d){ const char* e=std::getenv(k); return e?atoi(e):d; };
    const int  N_ENC = envi("SCHED_NENC", 6),  N_DEC = envi("SCHED_NDEC", 40);
    const int  K_GPU = envi("SCHED_KGPU", 1),  K_CPU = envi("SCHED_KCPU", 1), K_NPU = envi("SCHED_KNPU", 2);
    const bool HEFT  = envi("SCHED_HEFT", 1) != 0;
    const bool PART  = envi("SCHED_PARTITION", 1) != 0;
    const int  CPU_MS = envi("SCHED_CPU_MS", 2500);   // MODELED CPU-encode latency (2nd clip ctx segfaults on this stack)
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
    mtmd_context * vgpu = mkv(true);                       // Adreno (MTMD_BACKEND_DEVICE=GPUOpenCL)
    if (!vgpu) { LOG_ERR("vision load failed\n"); return 1; }
    // NOTE: a 2nd (CPU) clip context segfaults on this stack (single-residency); under
    // HEFT the CPU-encode backend is therefore MODELED (a measured-cost sleep), while the
    // HEFT routing decision itself runs on real queue state + real GPU timings.

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

    BackendQ bk[BK_N]; bk[BK_GPU].K=K_GPU; bk[BK_CPU].K=K_CPU; bk[BK_NPU].K=K_NPU;
    std::atomic<int> done{0};
    std::condition_variable sched_cv; std::mutex sched_m;
    // online cost model (ms) + projected free time, seeded with priors
    double cost[BK_N] = { 900.0, (double) CPU_MS, 25.0 };
    double busy_until[BK_N] = {0,0,0};
    std::mutex costm;

    auto run_job = [&](BK b){
        double t = now_ms();
        if (b == BK_NPU) {
            llama_batch bt = llama_batch_init(1,0,1); bt.n_tokens=1;
            bt.token[0]=llama_vocab_bos(llama_model_get_vocab(model));
            bt.pos[0]=dpos.fetch_add(1); bt.n_seq_id[0]=1; bt.seq_id[0][0]=0; bt.logits[0]=true;
            llama_decode(lctx, bt); llama_batch_free(bt);
        } else if (b == BK_CPU) {
            std::this_thread::sleep_for(std::chrono::milliseconds(CPU_MS));   // MODELED CPU encode
        } else {
            mtmd_encode_chunk(vgpu, img_gpu);                                 // REAL vision encode @ GPU
        }
        double dur = now_ms() - t;
        { std::lock_guard<std::mutex> lk(costm); cost[b] = 0.7*cost[b] + 0.3*dur; }  // EMA
    };
    auto worker = [&](BK b, int lo, int hi){
        if (PART) pin(lo, hi);
        for (;;) {
            int job;
            { std::unique_lock<std::mutex> lk(bk[b].m);
              bk[b].cv.wait(lk, [&]{ return !bk[b].jobs.empty() || bk[b].stop; });
              if (bk[b].jobs.empty() && bk[b].stop) return;
              job = bk[b].jobs.front(); bk[b].jobs.pop_front(); }
            run_job(b); (void) job;
            bk[b].inflight.fetch_sub(1); bk[b].done_cnt.fetch_add(1); done.fetch_add(1);
            { std::lock_guard<std::mutex> lk(sched_m); } sched_cv.notify_one();
        }
    };
    // GPU vision + CPU vision share the compute half (cores 4-7); NPU FastRPC on 0-3
    fprintf(stderr, "[sched] prefill done, starting %d workers + dispatch (%d reqs)\n", HEFT?3:2, TOTAL); fflush(stderr);
    std::thread tg(worker, BK_GPU, 4, 7);
    std::thread tc = HEFT ? std::thread(worker, BK_CPU, 4, 7) : std::thread();
    std::thread tn(worker, BK_NPU, 0, 3);

    std::atomic<bool> run{true};
    std::vector<std::array<double,5>> series;   // t, virtual, gpu, cpu, npu
    double t0 = now_ms();
    std::thread mon([&]{
        while (run.load()) {
            size_t v; { std::lock_guard<std::mutex> lk(vqm); v = vq.size(); }
            series.push_back({ now_ms()-t0, (double)v,
                (double)bk[BK_GPU].inflight.load(), (double)bk[BK_CPU].inflight.load(), (double)bk[BK_NPU].inflight.load() });
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    // supported backends per kind
    auto supported = [&](Kind k) -> std::vector<BK> {
        if (k == DECODE) return { BK_NPU };
        return HEFT ? std::vector<BK>{ BK_GPU, BK_CPU } : std::vector<BK>{ BK_GPU };
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
                if (bk[b].inflight.load() >= bk[b].K) continue;       // no free slot -> can't admit now
                double eft = std::max(tnow, busy_until[b]) + cost[b];  // HEFT earliest-finish
                if (eft < best_eft) { best_eft = eft; best = b; }
              }
              if (best != BK_N) busy_until[best] = std::max(tnow, busy_until[best]) + cost[best];
            }
            if (best != BK_N) {
                bk[best].inflight.fetch_add(1);
                { std::lock_guard<std::mutex> wl(bk[best].m); bk[best].jobs.push_back(it->id); } bk[best].cv.notify_one();
                it = vq.erase(it); progress = true;
            } else ++it;
          } }
        if (!progress) { std::unique_lock<std::mutex> lk(sched_m);
            sched_cv.wait_for(lk, std::chrono::milliseconds(5), [&]{ return done.load() >= TOTAL; }); }
    }
    run.store(false); mon.join();
    for (int b=0;b<BK_N;++b){ { std::lock_guard<std::mutex> lk(bk[b].m); bk[b].stop=true; } bk[b].cv.notify_all(); }
    tg.join(); if (tc.joinable()) tc.join(); tn.join();

    double mx[4]={0,0,0,0};
    FILE * f = fopen(OUT, "w");
    if (f) fprintf(f, "t_ms,virtual,gpu_real,cpu_real,npu_real\n");
    for (auto & r : series) { if (f) fprintf(f, "%.1f,%.0f,%.0f,%.0f,%.0f\n", r[0],r[1],r[2],r[3],r[4]);
        for (int j=0;j<4;++j) mx[j]=std::max(mx[j], r[1+j]); }
    if (f) fclose(f);
    LOG("\n================ vq scheduler (%s) ================\n", HEFT?"HEFT":"fixed");
    LOG("requests     : %d (%d encode + %d decode)\n", TOTAL, N_ENC, N_DEC);
    LOG("caps         : K_gpu=%d K_cpu=%d K_npu=%d  partition=%d\n", K_GPU,K_CPU,K_NPU,(int)PART);
    LOG("note         : GPU encode + NPU decode are REAL; CPU encode is MODELED (%d ms sleep)\n", CPU_MS);
    LOG("virtual peak : %.0f\n", mx[0]);
    LOG("encode split : GPU=%d  CPU=%d   decode NPU=%d\n",
        bk[BK_GPU].done_cnt.load(), bk[BK_CPU].done_cnt.load(), bk[BK_NPU].done_cnt.load());
    for (int b=0;b<BK_N;++b)
        LOG("%s real max   : %.0f (cap %d) %s   learned cost=%.0f ms\n", BKNAME[b], mx[1+b], bk[b].K,
            mx[1+b]<=bk[b].K?"OK":"OVER", cost[b]);
    LOG("wall         : %.0f ms,  series -> %s (%zu samples)\n", now_ms()-t0, OUT, series.size());
    LOG("=================================================\n");
    mtmd_free(vgpu);
    return 0;
}
