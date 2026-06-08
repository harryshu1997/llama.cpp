// mtmd-sched: the VQ scheduler running for real on-device. A CPU dispatcher pulls
// requests from a VIRTUAL QUEUE and admits them to a backend (vision-encode@GPU /
// text-decode@NPU) only while that backend's in-flight count < K (the CONWIP WIP
// cap, same policy as ggml_vq_admit / vq_dispatcher.simulate_bounded). One worker
// thread per backend = one in-order hardware queue. A monitor logs the three queue
// depths (virtual / GPU / NPU) every few ms -> vq_sched_series.csv, which vq_view
// renders. This makes the three-queue picture MEASURED, not modeled.
//
// Env: SCHED_NENC (encode reqs, def 6), SCHED_NDEC (decode reqs, def 40),
//      SCHED_KGPU (def 1), SCHED_KNPU (def 2), SCHED_PARTITION (4/4 affinity, def 1),
//      SCHED_OUT (csv path, def vq_sched_series.csv).
// MTMD_BACKEND_DEVICE=GPUOpenCL picks vision backend; --device HTP0 picks text.
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

enum Kind { ENCODE, DECODE };
struct Req { int id; Kind kind; };

// one in-order backend = a worker thread draining an admitted-job deque
struct BackendQ {
    std::mutex m;
    std::condition_variable cv;
    std::deque<int> jobs;          // request ids admitted (waiting + the one running)
    std::atomic<int> inflight{0};  // admitted but not finished == real queue depth (<= K)
    int K = 1;
    bool stop = false;
};

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD)) return 1;
    if (params.mmproj.path.empty() || params.image.empty()) { LOG_ERR("need -m --mmproj --image\n"); return 1; }
    const int  N_ENC = std::getenv("SCHED_NENC") ? atoi(std::getenv("SCHED_NENC")) : 6;
    const int  N_DEC = std::getenv("SCHED_NDEC") ? atoi(std::getenv("SCHED_NDEC")) : 40;
    const int  K_GPU = std::getenv("SCHED_KGPU") ? atoi(std::getenv("SCHED_KGPU")) : 1;
    const int  K_NPU = std::getenv("SCHED_KNPU") ? atoi(std::getenv("SCHED_KNPU")) : 2;
    const bool PART  = std::getenv("SCHED_PARTITION") ? atoi(std::getenv("SCHED_PARTITION")) != 0 : true;
    const char * OUT = std::getenv("SCHED_OUT") ? std::getenv("SCHED_OUT") : "vq_sched_series.csv";

    ggml_time_init(); common_init(); ggml_backend_load_all();
    common_init_result_ptr init = common_init_from_params(params);
    llama_model   * model = init ? init->model()   : nullptr;
    llama_context * lctx  = init ? init->context() : nullptr;
    if (!model || !lctx) { LOG_ERR("model load failed\n"); return 1; }

    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu = params.mmproj_use_gpu; mp.print_timings = false;
    mp.n_threads = params.cpuparams.n_threads; mp.flash_attn_type = params.flash_attn_type; mp.warmup = params.warmup;
    mtmd_context * vctx = mtmd_init_from_file(params.mmproj.path.c_str(), model, mp);
    if (!vctx) { LOG_ERR("vision load failed\n"); return 1; }

    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(vctx, params.image[0].c_str()));
    if (!bmp.ptr) { LOG_ERR("image load failed\n"); return 1; }
    std::vector<const mtmd_bitmap *> bmps = { bmp.ptr.get() };
    std::string prompt = std::string(mtmd_default_marker()) + "Describe this image.";
    mtmd_input_text text{ prompt.c_str(), true, true };
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    if (mtmd_tokenize(vctx, chunks.ptr.get(), &text, bmps.data(), bmps.size())) { LOG_ERR("tokenize failed\n"); return 1; }
    const mtmd_input_chunk * img = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks.ptr.get()); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks.ptr.get(), i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) { img = c; break; }
    }
    if (!img) { LOG_ERR("no image chunk\n"); return 1; }
    // prefill so decode has a valid KV state
    { llama_batch b = llama_batch_init(1,0,1); b.n_tokens=1; b.token[0]=llama_vocab_bos(llama_model_get_vocab(model));
      b.pos[0]=0; b.n_seq_id[0]=1; b.seq_id[0][0]=0; b.logits[0]=true; llama_decode(lctx,b); llama_batch_free(b); }
    std::atomic<llama_pos> dpos{1};

    // ---- virtual queue (burst: all requests arrive at t=0), interleaved enc/dec ----
    std::deque<Req> vq;
    { int ie=0, id=0, rid=0;
      while (ie < N_ENC || id < N_DEC) {
        if (ie < N_ENC) { vq.push_back({rid++, ENCODE}); ie++; }
        for (int k=0; k<7 && id<N_DEC; ++k) { vq.push_back({rid++, DECODE}); id++; }  // ~7 decodes per encode
      } }
    const int TOTAL = (int) vq.size();
    std::mutex vqm;

    BackendQ gpu, npu; gpu.K = K_GPU; npu.K = K_NPU;
    BackendQ * bq[2] = { &gpu, &npu };  // index by Kind
    std::atomic<int> done{0};
    std::condition_variable sched_cv; std::mutex sched_m;  // dispatcher wakeup on completion

    auto worker = [&](BackendQ * q, Kind kind, int lo, int hi) {
        if (PART) pin(lo, hi);
        for (;;) {
            int job;
            { std::unique_lock<std::mutex> lk(q->m);
              q->cv.wait(lk, [&]{ return !q->jobs.empty() || q->stop; });
              if (q->jobs.empty() && q->stop) return;
              job = q->jobs.front(); q->jobs.pop_front(); }
            if (kind == ENCODE) {
                mtmd_encode_chunk(vctx, img);                         // real vision encode @ GPU
            } else {
                llama_batch b = llama_batch_init(1,0,1); b.n_tokens=1;
                b.token[0]=llama_vocab_bos(llama_model_get_vocab(model));
                b.pos[0]=dpos.fetch_add(1); b.n_seq_id[0]=1; b.seq_id[0][0]=0; b.logits[0]=true;
                llama_decode(lctx, b); llama_batch_free(b);           // real decode @ NPU
            }
            (void) job;
            q->inflight.fetch_sub(1);
            done.fetch_add(1);
            { std::lock_guard<std::mutex> lk(sched_m); } sched_cv.notify_one();  // a slot freed
        }
    };
    std::thread tg(worker, &gpu, ENCODE, 4, 7);
    std::thread tn(worker, &npu, DECODE, 0, 3);

    // ---- monitor: log the three queue depths every 2 ms ----
    std::atomic<bool> run{true};
    std::vector<std::array<double,4>> series;  // t_ms, virtual, gpu_real, npu_real
    double t0 = now_ms();
    std::thread mon([&]{
        while (run.load()) {
            size_t v; { std::lock_guard<std::mutex> lk(vqm); v = vq.size(); }
            series.push_back({ now_ms()-t0, (double)v, (double)gpu.inflight.load(), (double)npu.inflight.load() });
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    // ---- dispatcher: admit from the virtual queue while a backend has room (< K) ----
    while (done.load() < TOTAL) {
        bool progress = false;
        { std::lock_guard<std::mutex> lk(vqm);
          for (auto it = vq.begin(); it != vq.end(); ) {              // per-backend FIFO, no head-of-line block
            BackendQ * q = bq[it->kind];
            if (q->inflight.load() < q->K) {
                q->inflight.fetch_add(1);
                { std::lock_guard<std::mutex> wl(q->m); q->jobs.push_back(it->id); } q->cv.notify_one();
                it = vq.erase(it); progress = true;
            } else ++it;
          } }
        if (!progress) {                                             // nothing admittable -> wait for a completion
            std::unique_lock<std::mutex> lk(sched_m);
            sched_cv.wait_for(lk, std::chrono::milliseconds(5), [&]{ return done.load() >= TOTAL; });
        }
    }
    run.store(false); mon.join();
    { std::lock_guard<std::mutex> lk(gpu.m); gpu.stop = true; } gpu.cv.notify_all();
    { std::lock_guard<std::mutex> lk(npu.m); npu.stop = true; } npu.cv.notify_all();
    tg.join(); tn.join();

    // ---- write series + summary ----
    double maxv=0, maxg=0, maxn=0;
    FILE * f = fopen(OUT, "w");
    if (f) fprintf(f, "t_ms,virtual,gpu_real,npu_real\n");
    for (auto & r : series) {
        if (f) fprintf(f, "%.1f,%.0f,%.0f,%.0f\n", r[0], r[1], r[2], r[3]);
        maxv=std::max(maxv,r[1]); maxg=std::max(maxg,r[2]); maxn=std::max(maxn,r[3]);
    }
    if (f) fclose(f);
    LOG("\n================ vq scheduler ================\n");
    LOG("requests     : %d (%d encode@GPU + %d decode@NPU)\n", TOTAL, N_ENC, N_DEC);
    LOG("caps         : K_gpu=%d  K_npu=%d   partition=%d\n", K_GPU, K_NPU, (int)PART);
    LOG("virtual peak : %.0f\n", maxv);
    LOG("GPU real max : %.0f  (cap %d)  -> %s\n", maxg, K_GPU, maxg<=K_GPU?"bounded OK":"OVER CAP");
    LOG("NPU real max : %.0f  (cap %d)  -> %s\n", maxn, K_NPU, maxn<=K_NPU?"bounded OK":"OVER CAP");
    LOG("wall         : %.0f ms,  series -> %s (%zu samples)\n", now_ms()-t0, OUT, series.size());
    LOG("==============================================\n");
    mtmd_free(vctx);
    return 0;
}
