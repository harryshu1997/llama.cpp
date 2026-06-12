// mtmd-stream: REQ-003 continuous-arrival serving harness. Open-loop: one FRAME arrives
// every STREAM_T_MS; each frame = 1 real vision encode + STREAM_D decode steps released
// when its encode completes (caption follows frame). Reports per-frame latency
// (arrival -> last decode done), steady-state p50/p95, fps, queue lengths.
//
// Policies via env (same as REQ-002): P1 all-NPU = MTMD_BACKEND_DEVICE=HTP0;
// P2 vision@GPU || decode@HTP = MTMD_BACKEND_DEVICE=GPUOpenCL. --device HTP0 -ngl 99 = decode.
// Env: STREAM_D(7) STREAM_T_MS(1100) STREAM_NFRAMES(20) STREAM_PARTITION(1).
//
// REQ-011 mixed-size mode: STREAM_BIG=1 -> -m is a TEXT-ONLY big decoder (no mmproj/vision);
// big free-runs 1-tok decode steps for MIX_WALL_MS. MIX_K>0 = big yields after K steps until
// an M2 slab lands (size-aware admission). MIX_GATE=0 = M2/M3 ignore the busy flag (FIFO arm).
// STREAM_M3* = third model stream (same coalesce/round-robin semantics as M2).
#include "arg.h"
#include "log.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#if defined(__unix__) || defined(__ANDROID__)
#include <sched.h>
#endif

extern "C" void ggml_backend_fair_thread_weight(double w);   // REQ-012 (GGML_FAIR=stride)
extern "C" void ggml_backend_chunk_thread(int c);            // REQ-015 (per-thread submission grain)

static void pin_cores(int lo, int hi) {
#if defined(__unix__) || defined(__ANDROID__)
    cpu_set_t s; CPU_ZERO(&s);
    for (int c = lo; c <= hi; ++c) CPU_SET(c, &s);
    sched_setaffinity(0, sizeof(s), &s);
#else
    (void) lo; (void) hi;
#endif
}
static double now_ms() { return (double) ggml_time_us() / 1000.0; }

struct Frame { int id; double t_arr=0, t_enc0=0, t_enc1=0, t_dec1=0; };

struct XS {                                              // extra coalesced decode stream (M2/M3)
    llama_model * mdl = nullptr; llama_context * ctx = nullptr;
    std::string dev; int B = 1, S = 1; double w = 1.0;
    std::atomic<long> toks{0}, slabs{0};
    std::vector<double> itl; std::vector<long> seq_toks;
    std::thread th;
};

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD)) return 1;
    auto envi = [](const char * k, int d){ const char * e = std::getenv(k); return e ? atoi(e) : d; };
    const bool BIG  = envi("STREAM_BIG", 0) != 0;       // REQ-011: -m is a text-only big decoder
    if (!BIG && (params.mmproj.path.empty() || params.image.empty())) { LOG_ERR("need -m --mmproj --image\n"); return 1; }
    const int  D    = envi("STREAM_D", 7);
    const int  T_MS = envi("STREAM_T_MS", 1100);
    const int  NF   = envi("STREAM_NFRAMES", 20);
    const bool PART = envi("STREAM_PARTITION", 1) != 0;
    const int  WARM = 2;   // frames excluded from stats
    const int  WALL_MS = envi("MIX_WALL_MS", 45000);    // big-mode run length
    const int  KCAP    = envi("MIX_K", 0);              // big yields after K steps (0 = free-run)
    const int  BIGCHUNK= envi("MIX_CHUNK", 0);          // REQ-015: big-model graph submission grain (0/1=whole)
    const int  PFN     = envi("STREAM_PREFILL", 0);     // REQ-020: big = repeated PFN-token PREFILL blocker (0=decode)
    const bool GATE    = envi("MIX_GATE", 1) != 0;      // M2/M3 respect the busy flag
    auto envd = [](const char * k, double d){ const char * e = std::getenv(k); return e ? atof(e) : d; };
    const double W_BIG = envd("MIX_W_BIG", 1.0);        // stride weights (GGML_FAIR=stride)
    const double W_M2  = envd("MIX_W_M2", 1.0);
    const double W_M3  = envd("MIX_W_M3", 1.0);

    ggml_time_init(); common_init();
    common_init_result_ptr init = common_init_from_params(params);
    llama_model   * model = init ? init->model()   : nullptr;
    llama_context * lctx  = init ? init->context() : nullptr;
    if (!model || !lctx) { LOG_ERR("model load failed\n"); return 1; }

    const char * ARB = std::getenv("STREAM_ARB");            // REQ-014: "ours" | "band" (dual-clip EFT routing)
    const bool arb_band = ARB && std::string(ARB) == "band"; // band = concurrency-BLIND (busy_until OFF)
    mtmd_context * vctx = nullptr;                           // primary clip ctx (GPU by convention)
    mtmd_context * vctx_npu = nullptr;                       // 2nd clip ctx on NPU (arbiter routes between them)
    mtmd::bitmap bmp;
    mtmd::input_chunks chunks;
    const mtmd_input_chunk * img = nullptr;
    if (!BIG) {
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu = params.mmproj_use_gpu; mp.print_timings = false;
        mp.n_threads = params.cpuparams.n_threads; mp.flash_attn_type = params.flash_attn_type;
        mp.warmup = params.warmup;
        vctx = mtmd_init_from_file(params.mmproj.path.c_str(), model, mp);
        if (!vctx) { LOG_ERR("vision load failed\n"); return 1; }
        bmp.ptr.reset(mtmd_helper_bitmap_init_from_file(vctx, params.image[0].c_str()));
        if (!bmp.ptr) { LOG_ERR("image load failed\n"); return 1; }
        std::vector<const mtmd_bitmap *> bmps = { bmp.ptr.get() };
        std::string prompt = std::string(mtmd_default_marker()) + "Describe this image.";
        mtmd_input_text text{ prompt.c_str(), true, true };
        chunks.ptr.reset(mtmd_input_chunks_init());
        if (mtmd_tokenize(vctx, chunks.ptr.get(), &text, bmps.data(), bmps.size())) { LOG_ERR("tokenize failed\n"); return 1; }
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.ptr.get()); ++i) {
            const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks.ptr.get(), i);
            if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) { img = c; break; }
        }
        if (!img) { LOG_ERR("no image chunk\n"); return 1; }

        // REQ-003 Step-0 / REQ-014 gate: can ONE process hold TWO clip contexts (GPU + HTP)?
        // load a 2nd vision ctx on the OPPOSITE backend, encode one frame on each.
        if (std::getenv("STREAM_DUAL_CLIP")) {
            const char * a = std::getenv("MTMD_BACKEND_DEVICE");
            std::string devA = a ? a : "GPUOpenCL";
            std::string devB = (devA == "HTP0") ? "GPUOpenCL" : "HTP0";
            setenv("MTMD_BACKEND_DEVICE", devB.c_str(), 1);
            mtmd_context_params mp2 = mtmd_context_params_default();
            mp2.use_gpu = params.mmproj_use_gpu; mp2.print_timings = false;
            mp2.n_threads = params.cpuparams.n_threads; mp2.flash_attn_type = params.flash_attn_type;
            mp2.warmup = params.warmup;
            LOG("[step0] loading 2nd clip ctx on %s (1st on %s)...\n", devB.c_str(), devA.c_str());
            mtmd_context * vctx2 = mtmd_init_from_file(params.mmproj.path.c_str(), model, mp2);
            if (!vctx2) { LOG("[step0] FAIL: 2nd clip ctx load returned null\n"); return 2; }
            double ta0 = now_ms(); mtmd_encode_chunk(vctx,  img); double ta1 = now_ms();
            double tb0 = now_ms(); mtmd_encode_chunk(vctx2, img); double tb1 = now_ms();
            LOG("[step0] OK dual-residency: encode A(%s)=%.0fms  B(%s)=%.0fms\n",
                devA.c_str(), ta1-ta0, devB.c_str(), tb1-tb0);
            mtmd_free(vctx2);
            setenv("MTMD_BACKEND_DEVICE", devA.c_str(), 1);
            mtmd_free(vctx);
            return 0;
        }

        // REQ-014: arbiter mode — keep BOTH clip contexts alive (primary=GPU, 2nd=NPU);
        // the encode worker routes each frame by min-EFT. Dual-residency proven in Step-0.
        if (ARB) {
            setenv("MTMD_BACKEND_DEVICE", "HTP0", 1);
            mtmd_context_params mpn = mtmd_context_params_default();
            mpn.use_gpu = params.mmproj_use_gpu; mpn.print_timings = false;
            mpn.n_threads = params.cpuparams.n_threads; mpn.flash_attn_type = params.flash_attn_type;
            mpn.warmup = params.warmup;
            vctx_npu = mtmd_init_from_file(params.mmproj.path.c_str(), model, mpn);
            setenv("MTMD_BACKEND_DEVICE", "GPUOpenCL", 1);
            if (!vctx_npu) { LOG_ERR("[arb] 2nd (NPU) clip ctx load failed\n"); return 2; }
            LOG("[arb] mode=%s dual clip ready (primary=GPU, alt=NPU)\n", ARB);
        }
    }
    { llama_batch b = llama_batch_init(1,0,1); b.n_tokens=1; b.token[0]=llama_vocab_bos(llama_model_get_vocab(model));
      b.pos[0]=0; b.n_seq_id[0]=1; b.seq_id[0][0]=0; b.logits[0]=true; llama_decode(lctx,b); llama_batch_free(b); }

    // ---- extra models (M2 / M3): residency by load-time HEFT (utilisation EFT) ----
    auto load_extra = [&](XS & x, const char * path, const char * devenv, int B, int S) {
        x.B = std::max(1, B); x.S = std::max(x.B, std::max(1, S)); x.seq_toks.assign(x.S, 0);
        x.dev = devenv ? devenv : "auto";
        if (x.dev == "auto") {
            double util_gpu = BIG ? 0.0           : 911.0 / T_MS;   // vision encode occupies GPU
            double util_npu = BIG ? 1.0           : D * 65.0 / T_MS; // big/E2B decode occupies NPU
            x.dev = util_gpu < util_npu ? "GPUOpenCL" : "HTP0";
            LOG("[heft-load] util_gpu=%.2f util_npu=%.2f -> %s @ %s\n", util_gpu, util_npu, path, x.dev.c_str());
        }
        llama_model_params mp2 = llama_model_default_params();
        ggml_backend_dev_t devs[3]; size_t nd = 0;
        for (size_t i = 0; i < ggml_backend_dev_count() && nd + 1 < 3; ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (x.dev == ggml_backend_dev_name(d)) devs[nd++] = d;
        }
        for (size_t i = 0; i < ggml_backend_dev_count() && nd + 1 < 3; ++i) {     // CPU fallback
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) devs[nd++] = d;
        }
        devs[nd] = nullptr; mp2.devices = devs; mp2.n_gpu_layers = 99;
        x.mdl = llama_model_load_from_file(path, mp2);
        if (x.mdl) { llama_context_params cp2 = llama_context_default_params();
                     cp2.n_ctx = 2048 * x.S; cp2.n_seq_max = x.S; cp2.n_batch = std::max(32, x.B);
                     x.ctx = llama_init_from_model(x.mdl, cp2); }
        if (!x.ctx) { LOG_ERR("extra model load failed: %s\n", path); exit(1); }
    };
    XS x2, x3;
    if (std::getenv("STREAM_M2")) { load_extra(x2, std::getenv("STREAM_M2"), std::getenv("STREAM_M2_DEV"),
                                               envi("STREAM_M2_B", 1), envi("STREAM_M2_SEQS", 0)); x2.w = W_M2; }
    if (std::getenv("STREAM_M3")) { load_extra(x3, std::getenv("STREAM_M3"), std::getenv("STREAM_M3_DEV"),
                                               envi("STREAM_M3_B", 1), envi("STREAM_M3_SEQS", 0)); x3.w = W_M3; }

    std::vector<Frame> fr(NF);
    std::deque<int> encq, decq;
    std::atomic<bool> e2b_busy{false};                       // main-model burst in flight -> hold M2/M3 injection
    std::mutex m; std::condition_variable cv;
    std::atomic<bool> stop{false};
    std::atomic<int> done{0};
    std::atomic<llama_pos> dpos{1};
    size_t enc_peak = 0, dec_peak = 0;

    // ---- REQ-014 arbiter state (EFT with cross-stream busy_until) ----
    std::mutex arb_mtx;
    double busy_npu = 0, busy_gpu = 0;            // backend free-at timestamps (ms, absolute now_ms scale)
    double cost_enc_gpu = 1000, cost_enc_npu = 795, cost_dec_npu = 65;  // EMA seeds (measured)
    long route_gpu = 0, route_npu = 0;
    std::vector<double> arb_abserr, arb_relerr;   // |predicted finish - actual| per frame
    std::vector<double> arb_ignored;              // NPU backlog (busy_npu-now) present at decision (band discards it)
    auto ema = [](double & c, double s){ c = 0.7 * c + 0.3 * s; };

    // ---- REQ-019 admission/aging ----
    std::atomic<int> p0_pending{0};                  // P0 (M2) has a decode in flight/waiting
    const int  ADMIT_MS = envi("MIX_ADMIT_MS", 0);   // defer P2 (vision) up to this long while P0 pending (0=off)
    const int  M3_SLEEP = envi("MIX_M3_SLEEP_MS", 0); // rate-limit P1 (M3) for the feasible regime R1
    std::vector<double> vis_defer;                   // measured P2 deferral per encode (max-defer / starvation check)

    // ---- main workload threads ----
    std::thread tenc, tdec, tbig;
    std::atomic<long> big_toks{0};
    std::vector<double> big_itl;
    std::vector<double> pf_ms; std::atomic<long> pf_n{0};   // REQ-020: per-prefill wall-times
    if (!BIG) {
        tenc = std::thread([&]{                              // vision worker (+ REQ-014 EFT routing)
            if (PART) pin_cores(4, 7);
            { int vc = envi("VIS_CHUNK", 0); if (vc > 1) ggml_backend_chunk_thread(vc); }  // REQ-017: chunk the monolithic encode
            ggml_backend_fair_thread_weight(envd("VIS_W", 1.0));  // REQ-018: vision priority (low) for GGML_FAIR=prio/stride
            for (;;) {
                int id;
                { std::unique_lock<std::mutex> lk(m);
                  cv.wait(lk, [&]{ return !encq.empty() || stop; });
                  if (encq.empty()) return;
                  id = encq.front(); encq.pop_front(); }
                if (ADMIT_MS > 0) {                          // REQ-019 admission: defer P2 while P0 pending, P2 ages (bounded)
                    double dstart = now_ms();
                    while (!stop.load() && p0_pending.load() > 0 && now_ms() - dstart < ADMIT_MS)
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    vis_defer.push_back(now_ms() - dstart);
                }
                mtmd_context * use = vctx; bool on_npu = false; double predicted = 0;
                if (ARB) {                                   // pick backend by min-EFT
                    std::lock_guard<std::mutex> lk(arb_mtx);
                    double now = now_ms();
                    arb_ignored.push_back(std::max(0.0, busy_npu - now));  // NPU backlog at decision (decode burst)
                    // band = concurrency-BLIND: ignore busy_until (isolated static profile)
                    double eft_gpu = (arb_band ? now : std::max(now, busy_gpu)) + cost_enc_gpu;
                    double eft_npu = (arb_band ? now : std::max(now, busy_npu)) + cost_enc_npu;
                    on_npu = eft_npu <= eft_gpu;
                    use = on_npu ? vctx_npu : vctx;
                    predicted = on_npu ? eft_npu : eft_gpu;
                    double start = std::max(now, on_npu ? busy_npu : busy_gpu);
                    if (on_npu) busy_npu = start + cost_enc_npu; else busy_gpu = start + cost_enc_gpu;
                    (on_npu ? route_npu : route_gpu)++;
                }
                fr[id].t_enc0 = now_ms();
                mtmd_encode_chunk(use, img);                 // REAL encode on chosen backend
                fr[id].t_enc1 = now_ms();
                { static std::atomic<int> enc_n{0}; int en = enc_n.fetch_add(1) + 1; if (en % 4 == 0) LOG("[visbeat] enc=%d enc_ms=%.0f\n", en, fr[id].t_enc1 - fr[id].t_enc0); }
                if (ARB) {
                    std::lock_guard<std::mutex> lk(arb_mtx);
                    ema(on_npu ? cost_enc_npu : cost_enc_gpu, fr[id].t_enc1 - fr[id].t_enc0);
                    double aerr = std::abs(predicted - fr[id].t_enc1);   // predicted vs ACTUAL finish
                    arb_abserr.push_back(aerr);
                    arb_relerr.push_back(aerr / std::max(1.0, fr[id].t_enc1 - fr[id].t_enc0));
                }
                { std::lock_guard<std::mutex> lk(m); decq.push_back(id); dec_peak = std::max(dec_peak, decq.size()); }
                cv.notify_all();
            }
        });
        tdec = std::thread([&]{                              // decode worker (HTP)
            if (PART) pin_cores(0, 3);
            ggml_backend_fair_thread_weight(W_BIG);
            llama_batch b = llama_batch_init(1,0,1);
            for (;;) {
                int id;
                { std::unique_lock<std::mutex> lk(m);
                  cv.wait(lk, [&]{ return !decq.empty() || stop; });
                  if (decq.empty()) { if (stop) break; else continue; }
                  id = decq.front(); decq.pop_front(); }
                e2b_busy.store(true);
                if (ARB) {                                   // decode is NPU-pinned: book the burst on busy_npu
                    std::lock_guard<std::mutex> lk(arb_mtx);
                    double now = now_ms();
                    busy_npu = std::max(now, busy_npu) + D * cost_dec_npu;
                }
                double td0 = now_ms();
                for (int k = 0; k < D; ++k) {
                    b.n_tokens=1; b.token[0]=llama_vocab_bos(llama_model_get_vocab(model));
                    b.pos[0]=dpos.fetch_add(1); b.n_seq_id[0]=1; b.seq_id[0][0]=0; b.logits[0]=true;
                    llama_decode(lctx, b);
                }
                if (ARB && D > 0) { std::lock_guard<std::mutex> lk(arb_mtx); ema(cost_dec_npu, (now_ms() - td0) / D); }
                e2b_busy.store(false);
                fr[id].t_dec1 = now_ms(); done.fetch_add(1);
            }
            llama_batch_free(b);
        });
    } else {
        tbig = std::thread([&]{                              // REQ-011 big decoder: free-run 1-tok steps
            if (PART) pin_cores(PFN > 0 ? 4 : 0, PFN > 0 ? 7 : 3);  // REQ-020: prefill blocker on producer half (cores4-7), like vision
            ggml_backend_fair_thread_weight(W_BIG);
            if (BIGCHUNK > 1) ggml_backend_chunk_thread(BIGCHUNK);  // REQ-015/020: chunk the big graph (decode or prefill)
            const bool BIG_OFF = std::getenv("MIX_BIG_OFF") != nullptr;  // REQ-015 solo ref: big idle
            llama_batch b = llama_batch_init(1,0,1);
            llama_batch bp = llama_batch_init(PFN > 0 ? PFN : 1, 0, 1);  // REQ-020 prefill batch
            int k = 0;
            double tb0 = now_ms();
            while (!stop.load()) {
                if (BIG_OFF) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); continue; }
                if (PFN > 0) {   // REQ-020: monolithic prefill blocker — KV-reset BEFORE each iter = a fresh PFN-token prefill graph at pos 0
                    llama_memory_seq_rm(llama_get_memory(lctx), 0, 0, -1);
                    double tp0 = now_ms();
                    bp.n_tokens = PFN;
                    for (int i = 0; i < PFN; ++i) { bp.token[i] = (llama_token)(1 + (i % 3000)); bp.pos[i] = i; bp.n_seq_id[i] = 1; bp.seq_id[i][0] = 0; bp.logits[i] = (i == PFN - 1); }
                    if (llama_decode(lctx, bp)) break;
                    double pms = now_ms() - tp0; pf_ms.push_back(pms); big_toks.fetch_add(PFN);
                    long np = pf_n.fetch_add(1) + 1;
                    LOG("[pfbeat] prefill#%ld ms=%.0f tok/s=%.1f\n", np, pms, 1000.0 * PFN / pms);
                    continue;
                }
                if (KCAP > 0) e2b_busy.store(true);
                double t0 = now_ms();
                b.n_tokens=1; b.token[0]=llama_vocab_bos(llama_model_get_vocab(model));
                b.pos[0]=dpos.fetch_add(1); b.n_seq_id[0]=1; b.seq_id[0][0]=0; b.logits[0]=true;
                if (llama_decode(lctx, b)) break;
                big_itl.push_back(now_ms() - t0); big_toks.fetch_add(1);
                { long bt = big_toks.load(); if (bt % 64 == 0) LOG("[bigbeat] toks=%ld tok/s=%.2f\n", bt, 1000.0 * bt / (now_ms() - tb0)); }
                if (dpos.load() >= 1800) { llama_memory_seq_rm(llama_get_memory(lctx), 0, 0, -1); dpos.store(1); }
                if (KCAP > 0 && ++k >= KCAP) {               // size-aware yield: open a window for M2/M3
                    k = 0; e2b_busy.store(false);
                    long s0 = x2.slabs.load() + x3.slabs.load();
                    double tw = now_ms();
                    while (!stop.load() && x2.slabs.load() + x3.slabs.load() == s0 && now_ms() - tw < 100)
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            e2b_busy.store(false);
            llama_batch_free(b);
            llama_batch_free(bp);
        });
    }

    const bool M2_LO = std::getenv("STREAM_M2_CORE0") != nullptr;  // REQ-017: pin victim to cores0-3 (off the vision producer half)
    auto run_xs = [&](XS & x, bool is_p0, int sleep_ms){     // coalesced round-robin decode loop (M2=P0 / M3=P1)
        if (PART) { if (M2_LO) pin_cores(0, 3); else pin_cores(4, 7); }  // default cores4-7 (off FastRPC); M2_LO=cores0-3 to free the release window
        ggml_backend_fair_thread_weight(x.w);
        llama_batch b2 = llama_batch_init(x.B,0,x.B);
        llama_token t = llama_vocab_bos(llama_model_get_vocab(x.mdl));
        std::vector<llama_pos> sp(x.S, 0);
        int rr = 0;
        double xt0 = now_ms(); long xlast = 0;
        while (!stop.load()) {
            if (GATE && e2b_busy.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); continue; }  // admission: yield to main burst
            if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));  // REQ-019 R1: rate-limit P1
            double t0 = now_ms();
            b2.n_tokens = x.B;
            for (int i = 0; i < x.B; ++i) { int s = (rr + i) % x.S;
                b2.token[i]=t; b2.pos[i]=sp[s]; b2.n_seq_id[i]=1; b2.seq_id[i][0]=s; b2.logits[i]=(i==x.B-1); }
            if (is_p0) p0_pending.fetch_add(1);              // REQ-019: P0 wants the NPU (drives admission)
            int rc = llama_decode(x.ctx, b2);
            if (is_p0) p0_pending.fetch_sub(1);
            if (rc) break;
            for (int i = 0; i < x.B; ++i) { int s = (rr + i) % x.S; sp[s]++; x.seq_toks[s]++;
                if (sp[s] >= 1800) { llama_memory_seq_rm(llama_get_memory(x.ctx), s, 0, -1); sp[s] = 0; } }
            rr = (rr + x.B) % x.S;
            x.itl.push_back((now_ms() - t0) / x.B); x.toks.fetch_add(x.B); x.slabs.fetch_add(1);
            { long tk = x.toks.load(); if (tk - xlast >= 64) { xlast = tk; LOG("[xsbeat:%s] toks=%ld tok/s=%.2f B=%d dev=%s\n", is_p0?"P0":"P1", tk, 1000.0*tk/(now_ms()-xt0), x.B, x.dev.c_str()); } }
        }
        llama_batch_free(b2);
    };
    if (x2.ctx) x2.th = std::thread([&]{ run_xs(x2, true,  0); });          // M2 = P0 (high pri, drives admission)
    if (x3.ctx) x3.th = std::thread([&]{ run_xs(x3, false, M3_SLEEP); });   // M3 = P1 (mid, rate-limited in R1)

    LOG("\n[stream] %s D=%d T=%dms NF=%d wall=%dms K=%d gate=%d policy=%s m2=%s@%s m3=%s@%s\n",
        BIG ? "BIGMODE" : "vision", D, T_MS, NF, WALL_MS, KCAP, (int)GATE,
        std::getenv("MTMD_BACKEND_DEVICE") ? std::getenv("MTMD_BACKEND_DEVICE") : "default",
        x2.ctx ? std::getenv("STREAM_M2") : "none", x2.ctx ? x2.dev.c_str() : "-",
        x3.ctx ? std::getenv("STREAM_M3") : "none", x3.ctx ? x3.dev.c_str() : "-");
    double t0 = now_ms();
    if (!BIG) {
        for (int i = 0; i < NF; ++i) {                       // open-loop generator
            double due = t0 + (double) i * T_MS;
            while (now_ms() < due) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            fr[i].t_arr = now_ms();
            { std::lock_guard<std::mutex> lk(m); encq.push_back(i); enc_peak = std::max(enc_peak, encq.size()); }
            cv.notify_all();
        }
        while (done.load() < NF) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } else {
        while (now_ms() - t0 < WALL_MS) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    stop = true; cv.notify_all();
    if (tenc.joinable()) tenc.join();
    if (tdec.joinable()) tdec.join();
    if (tbig.joinable()) tbig.join();
    if (x2.th.joinable()) x2.th.join();
    if (x3.th.joinable()) x3.th.join();
    double wall = now_ms() - t0;

    auto jfi = [](const std::vector<long> & v){              // Jain fairness index over per-stream tokens
        double s = 0, s2 = 0; for (long x : v) { s += x; s2 += (double)x * x; }
        return (!v.empty() && s2 > 0) ? s*s / (v.size()*s2) : 0.0; };
    auto report_xs = [&](const char * tag, XS & x){
        if (!x.ctx || x.itl.empty()) return;
        std::sort(x.itl.begin(), x.itl.end());
        long mn = x.seq_toks[0], mx = x.seq_toks[0];
        for (long v : x.seq_toks) { mn = std::min(mn, v); mx = std::max(mx, v); }
        LOG("%s  toks=%ld  tok/s=%.2f  ITL p50=%.1f p95=%.1f MAX=%.1f  dev=%s  seqs=%d slotB=%d  per-stream min=%ld max=%ld JFI=%.3f\n",
            tag, x.toks.load(), 1000.0 * x.toks.load() / wall,
            x.itl[x.itl.size()/2], x.itl[(size_t)(0.95*x.itl.size())], x.itl.back(), x.dev.c_str(), x.S, x.B, mn, mx, jfi(x.seq_toks));
    };

    if (!BIG) {
        std::vector<double> lat, enc_ms;
        for (int i = WARM; i < NF; ++i) { lat.push_back(fr[i].t_dec1 - fr[i].t_arr); enc_ms.push_back(fr[i].t_enc1 - fr[i].t_enc0); }
        std::sort(lat.begin(), lat.end());
        auto pct = [&](double p){ return lat[std::min(lat.size()-1, (size_t)(p*lat.size()))]; };
        double el = 0; for (double e : enc_ms) el += e; el /= enc_ms.size();
        LOG("\n========== stream result (D=%d T=%d NF=%d, warm=%d skipped) ==========\n", D, T_MS, NF, WARM);
        for (int i = 0; i < NF; ++i)
            LOG("frame %2d  arr=%7.0f  encq_wait=%6.0f  enc=%5.0f  dec_done=+%7.0f  lat=%7.0f\n", i,
                fr[i].t_arr - t0, fr[i].t_enc0 - fr[i].t_arr, fr[i].t_enc1 - fr[i].t_enc0,
                fr[i].t_dec1 - fr[i].t_arr, fr[i].t_dec1 - fr[i].t_arr);
        LOG("LAT p50=%.0f  p95=%.0f  mean_enc=%.0f  fps=%.2f  enc_peak=%zu dec_peak=%zu  wall=%.0f\n",
            pct(0.50), pct(0.95), el, 1000.0 * NF / wall, enc_peak, dec_peak, wall);
        if (!vis_defer.empty()) {                            // REQ-019: P2 deferral (max-defer / starvation bound)
            std::sort(vis_defer.begin(), vis_defer.end());
            LOG("P2(vision) defer: p50=%.0f p95=%.0f MAX=%.0f ms (admit_ms=%d) — bounded => no P2 starvation\n",
                vis_defer[vis_defer.size()/2], vis_defer[(size_t)(0.95*vis_defer.size())], vis_defer.back(), ADMIT_MS);
        }
        if (ARB && !arb_abserr.empty()) {
            std::sort(arb_abserr.begin(), arb_abserr.end());
            double me = 0; for (double e : arb_relerr) me += e; me /= arb_relerr.size();
            std::sort(arb_ignored.begin(), arb_ignored.end());
            double ig50 = arb_ignored[arb_ignored.size()/2], ig95 = arb_ignored[(size_t)(0.95*arb_ignored.size())];
            LOG("ARB mode=%s route GPU=%ld NPU=%ld  enc-finish-miss p50=%.0fms p95=%.0fms (%.1f%%)  NPU-backlog-at-decision p50=%.0fms p95=%.0fms %s\n",
                ARB, route_gpu, route_npu, arb_abserr[arb_abserr.size()/2],
                arb_abserr[(size_t)(0.95*arb_abserr.size())], 100.0 * me, ig50, ig95,
                arb_band ? "(DISCARDED by band)" : "(used by ours)");
        }
    } else {
        LOG("\n========== mix result (wall=%.0f K=%d gate=%d) ==========\n", wall, KCAP, (int)GATE);
        if (!big_itl.empty()) {
            std::sort(big_itl.begin(), big_itl.end());
            LOG("BIG toks=%ld  tok/s=%.2f  slab(ITL) p50=%.1f p95=%.1f ms/step\n",
                big_toks.load(), 1000.0 * big_toks.load() / wall,
                big_itl[big_itl.size()/2], big_itl[(size_t)(0.95*big_itl.size())]);
        }
        if (!pf_ms.empty()) {                                // REQ-020: prefill blocker stats
            std::vector<double> p = pf_ms; std::sort(p.begin(), p.end());
            double sum = 0; for (double v : p) sum += v; double mean = sum / p.size();
            LOG("PREFILL n=%zu PFN=%d  prefill_ms p50=%.0f p95=%.0f mean=%.0f  eff_tok/s=%.1f  per-chunk_ms(@C=%d)=%.0f\n",
                p.size(), PFN, p[p.size()/2], p[(size_t)(0.95*p.size())], mean,
                1000.0 * PFN / mean, std::max(1, BIGCHUNK), mean / std::max(1, BIGCHUNK));
        }
    }
    report_xs("M2", x2);
    report_xs("M3", x3);
    if (vctx_npu) mtmd_free(vctx_npu);
    if (vctx) mtmd_free(vctx);
    return 0;
}
