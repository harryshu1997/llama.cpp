// mtmd-coexec: measure free overlap of vision-encode@GPU || text-decode@NPU in
// ONE coordinated process, and prove live concurrency via the VQ byte table's
// busy mask. Phase-Adaptive Cross-Backend Scheduler, Phase 1b (vq_scheduler_design.md).
//
// Three phases per run (one device invocation => full comparison):
//   1. solo-vision : N_v real gemma-4 vision encodes on the mmproj backend (GPU).
//   2. solo-decode : N_d real text decodes on the text backend (NPU/HTP).
//   3. coexec      : both loops concurrently on two threads (4/4 CPU partition).
// Speedup = (solo_vision_wall + solo_decode_wall) / coexec_wall. ~2x => free overlap.
//
// MTMD_BACKEND_DEVICE picks the vision backend (GPUOpenCL); --device picks text (HTP0).
// Env: COEXEC_NV (default 4), COEXEC_ND (default 64), COEXEC_PARTITION=1 (4/4 affinity).
#include "arg.h"
#include "log.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <vector>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <climits>
#if defined(__unix__) || defined(__ANDROID__)
#include <sched.h>
#include <pthread.h>
#endif

// VQ byte table (libggml-base, ggml-backend-impl.h) — declared here to avoid the internal header.
extern "C" {
    bool     ggml_vq_enabled(void);
    int      ggml_vq_depth(const char *);
    unsigned ggml_vq_busy_mask(void);
    void     ggml_vq_set_model(int);
    void     ggml_vq_session_begin(int);
    int      ggml_vq_admit(const char *, int);
}

static void pin_cores(int lo, int hi) {
#if defined(__unix__) || defined(__ANDROID__)
    cpu_set_t set; CPU_ZERO(&set);
    for (int c = lo; c <= hi; ++c) CPU_SET(c, &set);
    sched_setaffinity(0, sizeof(set), &set);
#else
    (void) lo; (void) hi;
#endif
}

static double now_ms() { return (double) ggml_time_us() / 1000.0; }

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD)) return 1;
    if (params.mmproj.path.empty() || params.image.empty()) {
        LOG_ERR("need -m, --mmproj and --image\n"); return 1;
    }
    const int    N_V  = std::getenv("COEXEC_NV") ? atoi(std::getenv("COEXEC_NV")) : 4;
    const int    N_D  = std::getenv("COEXEC_ND") ? atoi(std::getenv("COEXEC_ND")) : 64;
    const bool   PART = std::getenv("COEXEC_PARTITION") ? atoi(std::getenv("COEXEC_PARTITION")) != 0 : true;

    ggml_time_init();
    common_init();
    ggml_backend_load_all();

    // ---- load text model (text backend, e.g. HTP0 via --device) + vision (mmproj backend) ----
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init ? llama_init->model()   : nullptr;
    llama_context * lctx  = llama_init ? llama_init->context() : nullptr;
    if (!model || !lctx) { LOG_ERR("model load failed\n"); return 1; }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu        = params.mmproj_use_gpu;
    mparams.print_timings  = false;
    mparams.n_threads      = params.cpuparams.n_threads;
    mparams.flash_attn_type= params.flash_attn_type;
    mparams.warmup         = params.warmup;
    mtmd_context * vctx = mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams);
    if (!vctx) { LOG_ERR("vision load failed\n"); return 1; }

    // ---- load image, tokenize, extract the IMAGE chunk ----
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(vctx, params.image[0].c_str()));
    if (!bmp.ptr) { LOG_ERR("image load failed\n"); return 1; }
    std::vector<const mtmd_bitmap *> bmps = { bmp.ptr.get() };

    std::string prompt = std::string(mtmd_default_marker()) + "Describe this image.";
    mtmd_input_text text{ prompt.c_str(), /*add_special*/ true, /*parse_special*/ true };
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    if (mtmd_tokenize(vctx, chunks.ptr.get(), &text, bmps.data(), bmps.size()) != 0) {
        LOG_ERR("tokenize failed\n"); return 1;
    }
    const mtmd_input_chunk * img_chunk = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks.ptr.get()); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks.ptr.get(), i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) { img_chunk = c; break; }
    }
    if (!img_chunk) { LOG_ERR("no image chunk\n"); return 1; }

    if (ggml_vq_enabled()) ggml_vq_session_begin(2);
    LOG_INF("coexec: N_V=%d encodes, N_D=%d decodes, partition=%d, vq=%d\n",
            N_V, N_D, (int) PART, (int) ggml_vq_enabled());

    // ---- workers ----
    auto vision_loop = [&](int n, int k_gpu) {
        if (PART) pin_cores(4, 7);
        ggml_vq_set_model(0);
        for (int i = 0; i < n; ++i) {
            if (k_gpu > 0) ggml_vq_admit("GPU", k_gpu);
            int rc = mtmd_encode_chunk(vctx, img_chunk);    // real vision encode on the mmproj backend
            if (rc != 0) { LOG_ERR("encode %d failed\n", i); break; }
        }
    };
    // prefill once so decode has a valid KV state, then spin single-token decodes.
    auto decode_prefill = [&]() {
        llama_pos np = 0;
        llama_batch tb = llama_batch_init(8, 0, 1);
        tb.n_tokens = 1; tb.token[0] = llama_vocab_bos(llama_model_get_vocab(model));
        tb.pos[0] = np++; tb.n_seq_id[0] = 1; tb.seq_id[0][0] = 0; tb.logits[0] = true;
        llama_decode(lctx, tb);
        llama_batch_free(tb);
        return np;
    };
    auto decode_loop = [&](int n, int k_npu, llama_pos start_pos) {
        if (PART) pin_cores(0, 3);
        ggml_vq_set_model(1);
        llama_pos np = start_pos;
        llama_batch tb = llama_batch_init(1, 0, 1);
        llama_token tok = llama_vocab_bos(llama_model_get_vocab(model));
        for (int i = 0; i < n; ++i) {
            if (k_npu > 0) ggml_vq_admit("NPU", k_npu);
            tb.n_tokens = 1; tb.token[0] = tok; tb.pos[0] = np++;
            tb.n_seq_id[0] = 1; tb.seq_id[0][0] = 0; tb.logits[0] = true;
            if (llama_decode(lctx, tb)) { LOG_ERR("decode %d failed\n", i); break; }
        }
        llama_batch_free(tb);
    };

    // ---- Phase 1: solo vision ----
    double t0 = now_ms();
    vision_loop(N_V, 0);
    double solo_v = now_ms() - t0;

    // ---- Phase 2: solo decode ----
    llama_pos pos0 = decode_prefill();
    t0 = now_ms();
    decode_loop(N_D, 0, pos0);
    double solo_d = now_ms() - t0;

    // reset KV so coexec decode starts clean
    llama_memory_clear(llama_get_memory(lctx), true);
    llama_pos pos1 = decode_prefill();

    // ---- Phase 3: coexec (both backends concurrent) + busy-mask monitor ----
    std::atomic<bool> running{true};
    std::atomic<int>  both_busy{0}, samples{0};
    std::thread mon([&]{
        while (running.load()) {
            unsigned m = ggml_vq_busy_mask();      // bit1=GPU bit2=NPU
            samples.fetch_add(1);
            if ((m & (1u<<1)) && (m & (1u<<2))) both_busy.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    t0 = now_ms();
    std::thread tv([&]{ vision_loop(N_V, 0); });
    std::thread td([&]{ decode_loop(N_D, 0, pos1); });
    tv.join(); td.join();
    double coexec = now_ms() - t0;
    running = false; mon.join();

    // ---- report ----
    double serial = solo_v + solo_d;
    double pct_both = samples.load() ? 100.0 * both_busy.load() / samples.load() : 0.0;
    LOG("\n================ coexec result ================\n");
    LOG("solo vision  : %8.1f ms  (%d encodes, %.1f ms/enc)\n", solo_v, N_V, solo_v / N_V);
    LOG("solo decode  : %8.1f ms  (%d decodes, %.2f tok/s)\n", solo_d, N_D, 1000.0 * N_D / solo_d);
    LOG("serial sum   : %8.1f ms\n", serial);
    LOG("coexec wall  : %8.1f ms\n", coexec);
    LOG("speedup      : %8.2fx  (serial / coexec)\n", serial / coexec);
    LOG("overlap proof: %.1f%% of %d samples saw GPU&NPU both busy\n", pct_both, samples.load());
    LOG("===============================================\n");

    mtmd_free(vctx);
    return 0;
}
