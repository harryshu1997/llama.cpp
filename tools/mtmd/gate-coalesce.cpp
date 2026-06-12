// gate-coalesce: REQ-006 gates. ONE llama_context (n_seq_max=16).
//  G1 break-even: per-step time of ONE batched decode of B seqs (1 token/seq) vs B
//     sequential 1-token decodes. PASS = batched(B=4) <= 0.6x sequential.
//  G2 correctness: logits of each seq in the batched step ~= the same step decoded
//     sequentially (B=1 calls), max-abs eps. Env: GATE_STEPS(50) GATE_PREFILL(8).
#include "arg.h"
#include "log.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double now_ms() { return (double) ggml_time_us() / 1000.0; }

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    auto envi = [](const char * k, int d){ const char * e = getenv(k); return e ? atoi(e) : d; };
    const int STEPS = envi("GATE_STEPS", 50), PRE = envi("GATE_PREFILL", 8);
    const int BS[] = {1, 2, 4, 8, 16}; const int NB = 5, MAXB = 16;

    ggml_time_init(); common_init();
    params.n_parallel = MAXB; params.n_ctx = 2048 * 2;
    common_init_result_ptr init = common_init_from_params(params);
    llama_model * model = init ? init->model() : nullptr;
    llama_context * ctx = init ? init->context() : nullptr;
    if (!model || !ctx) { LOG_ERR("load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_token tok = llama_vocab_bos(vocab);
    llama_memory_t mem = llama_get_memory(ctx);

    llama_batch b = llama_batch_init(MAXB, 0, MAXB);
    std::vector<llama_pos> sp(MAXB, 0);                       // per-seq next position
    auto step_batched = [&](int B, bool logits) {             // 1 token per seq, advances B seqs
        b.n_tokens = B;
        for (int s = 0; s < B; ++s) { b.token[s]=tok; b.pos[s]=sp[s]; b.n_seq_id[s]=1; b.seq_id[s][0]=s; b.logits[s]=logits; }
        int rc = llama_decode(ctx, b);
        if (!rc) for (int s = 0; s < B; ++s) sp[s]++;
        return rc;
    };
    auto step_one = [&](int s, bool logits) {
        b.n_tokens = 1; b.token[0]=tok; b.pos[0]=sp[s]; b.n_seq_id[0]=1; b.seq_id[0][0]=s; b.logits[0]=logits;
        int rc = llama_decode(ctx, b);
        if (!rc) sp[s]++;
        return rc;
    };

    LOG("n_ctx=%d n_seq_max=%d\n", llama_n_ctx(ctx), llama_n_seq_max(ctx));
    for (int p = 0; p < PRE; ++p)
        if (int rc = step_batched(MAXB, false)) { LOG_ERR("prefill rc=%d at p=%d\n", rc, p); return 1; }

    LOG("\n==== G1 break-even (%d steps each; ms/STEP and ms/SEQ-TOKEN) ====\n", STEPS);
    double seq1 = 0;
    for (int bi = 0; bi < NB; ++bi) {
        int B = BS[bi];
        int rcb = 0, rcs = 0;
        double t0 = now_ms();
        for (int i = 0; i < STEPS && !rcb; ++i) rcb = step_batched(B, false);
        double batched = (now_ms() - t0) / STEPS;
        t0 = now_ms();
        for (int i = 0; i < STEPS && !rcs; ++i) for (int s = 0; s < B && !rcs; ++s) rcs = step_one(s, false);
        double seq = (now_ms() - t0) / STEPS;
        if (rcb || rcs) { LOG("B=%2d  FAILED rc_batched=%d rc_seq=%d -> UNSUPPORTED\n", B, rcb, rcs); continue; }
        if (B == 1) seq1 = seq;
        LOG("B=%2d  batched=%7.2f ms/step (%6.2f/tok)  sequential=%7.2f (%6.2f/tok)  ratio=%.3f  vsB1=%.3f\n",
            B, batched, batched/B, seq, seq/B, batched/seq, batched/(B*seq1));
    }

    LOG("\n==== G2 correctness (B=4 batched vs sequential, same per-seq pos) ====\n");
    std::vector<llama_pos> save(sp);
    if (step_batched(4, true)) { LOG_ERR("G2 batched failed\n"); return 1; }
    std::vector<float> lb(4 * n_vocab);
    for (int s = 0; s < 4; ++s) memcpy(lb.data() + s*n_vocab, llama_get_logits_ith(ctx, s), n_vocab*sizeof(float));
    for (int s = 0; s < 4; ++s) { llama_memory_seq_rm(mem, s, save[s], -1); sp[s] = save[s]; }   // rewind
    double mx = 0;
    for (int s = 0; s < 4; ++s) {
        if (step_one(s, true)) { LOG_ERR("G2 seq %d failed\n", s); return 1; }
        const float * l = llama_get_logits_ith(ctx, 0);
        for (int i = 0; i < n_vocab; ++i) mx = std::fmax(mx, std::fabs(l[i] - lb[s*n_vocab+i]));
    }
    LOG("G2 max|logit diff| = %.5f  (%s)\n", mx, mx < 0.1 ? "PASS" : "CHECK");
    llama_batch_free(b);
    return 0;
}
