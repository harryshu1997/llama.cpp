// llama-layersplit: offline correctness oracle for the cross-device LAYER-SPLIT decode
// feature patched into src/models/gemma4.cpp.
//
// The gemma4 graph reads env LLAMA_LAYER_START (ls, default 0) and LLAMA_LAYER_END
// (le, default n_layer). It computes only transformer layers [ls, le).
//   - le < n_layer  => head-less "head" stage: exposes the hidden state at the cut via
//                      res->t_h_nextn (read with llama_get_embeddings_nextn), no final norm / lm_head.
//   - ls > 0        => "tail" stage: input is an injected activation fed via batch.embd
//                      (batch.token == NULL).
//
// This tool is a SINGLE-TOKEN oracle (no prompt / KV complexity):
//   mode=mono : full model on one token, print argmax + top-5.
//   mode=head : run head stage (LLAMA_LAYER_END=k), dump the cut activation to --act-file.
//   mode=tail : run tail stage (LLAMA_LAYER_START=k) feeding the activation, print argmax + top-5.
// Oracle passes when mono top-1 == tail top-1.

#include "llama.h"
#include "ggml-backend.h"         // ggml_backend_dev_* for the HTP+Adreno device split
#include "common.h"               // common_tokenize / common_token_to_piece
#include "chat.h"                 // [plan-a port] common_chat_templates_* for --chat prompt formatting
#include "../../src/llama-ext.h" // staging header: llama_set/get_embeddings_nextn

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <clocale>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// socket relay (tailnet / headnet)
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

static void print_usage(int, char ** argv) {
    fprintf(stderr,
        "\nusage: %s -m <model> --mode <mono|head|mid|tail|tailnet|headnet> [opts]\n"
        "  mono    : full model on the prompt, print last-token top-1 + top-5\n"
        "  head    : run layers [0,LLAMA_LAYER_END), dump N cut activations + token ids to --act-file\n"
        "  mid     : env LLAMA_LAYER_START=k2 LLAMA_LAYER_END=k3 ; inject --act-file, run [k2,k3) head-less, relay to --act-out\n"
        "  tail    : run layers [LLAMA_LAYER_START,n_layer) on injected activations, print last-token top-1 + top-5\n"
        "  tailnet : --port P ; listen 0.0.0.0:P, own KV[k,48)+sampling (env LLAMA_LAYER_START=k)\n"
        "  headnet : --host H --port P -p PROMPT -n NGEN ; drive decode (env LLAMA_LAYER_END=k)\n"
        "  tailbench: -b STREAMS -n STEPS ; BATCHED tail decode, STREAMS seqs/forward (env LLAMA_LAYER_START=k)\n"
        "  stagenet  : --port P ; PERSISTENT head-less stage server (env LLAMA_LAYER_START/END), KV-resident\n"
        "  pipedriver: --host H --port A --port2 B -p PROMPT -n NGEN ; host tail (env LLAMA_LAYER_START=k3) drives\n"
        "              stage A (op15 [0,k2)) + stage B (op12 [k2,k3)) over TCP/USB, incremental decode\n"
        "  prefill: mono/head take -p PROMPT (multi-token) or --tok <int> (single); tail/mid read N from --act-file\n"
        "  common opts: [-p <prompt>] [--tok <int>] [--act-file <path>] [--act-out <path>] [-ngl <int>]\n\n",
        argv[0]);
}

// Print argmax token id + piece + top-5 (id:logit) for logits over the vocab.
static llama_token report_logits(const llama_vocab * vocab, const float * logits, int n_vocab) {
    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[i] = i;

    const int k = std::min(5, n_vocab);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    const llama_token top1 = idx[0];

    auto piece = [&](llama_token id) {
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) n = 0;
        return std::string(buf, n);
    };

    printf("ARGMAX: id=%d logit=%.6f piece='%s'\n", top1, logits[top1], piece(top1).c_str());
    printf("TOP5:\n");
    for (int i = 0; i < k; ++i) {
        printf("  %d  id=%d  logit=%.6f  piece='%s'\n",
               i, idx[i], logits[idx[i]], piece(idx[i]).c_str());
    }
    return top1;
}

// ===========================================================================
// DISAGG de-risk (REQ-045): does a KV state blob produced on one backend (CUDA server)
// load + continue correctly on another (Adreno op15)?  Both sides run the WHOLE 48L model.
//   kvsave: prefill prompt_len tokens -> snapshot seq-0 KV via llama_state_seq_get_data ->
//           write {size, first_token(argmax of prefill), blob} -> greedy-decode n_gen (ref).
//   kvload: read the blob -> llama_state_seq_set_data -> greedy-decode n_gen from first_token.
// PASS (same hw) = bit-identical gen; cross-hw = caption-faithful (Q4-kernel drift).
// ===========================================================================
static int run_kv(llama_context * ctx, const llama_vocab * vocab, int n_vocab,
                  bool save, int prompt_len, int n_gen, const std::string & blob_file) {
    const llama_token bos = llama_vocab_bos(vocab);
    int rc = 0;
    llama_token first = 0;

    if (save) {
        llama_batch pf = llama_batch_init(prompt_len, 0, 1);
        pf.n_tokens = prompt_len;
        for (int i = 0; i < prompt_len; ++i) {
            pf.token[i]  = (i == 0) ? bos : (llama_token)((i * 131 + 7) % n_vocab);
            pf.pos[i]    = i; pf.n_seq_id[i] = 1; pf.seq_id[i][0] = 0;
            pf.logits[i] = (i == prompt_len - 1);
        }
        if (llama_decode(ctx, pf) != 0) { fprintf(stderr, "error: prefill decode\n"); llama_batch_free(pf); return 2; }
        const float * lg = llama_get_logits_ith(ctx, prompt_len - 1);
        if (!lg) { fprintf(stderr, "error: prefill logits NULL\n"); llama_batch_free(pf); return 2; }
        { float bv = lg[0]; for (int i = 1; i < n_vocab; ++i) if (lg[i] > bv) { bv = lg[i]; first = i; } }
        llama_batch_free(pf);
        size_t sz = llama_state_seq_get_size(ctx, 0);
        std::vector<uint8_t> buf(sz);
        size_t w = llama_state_seq_get_data(ctx, buf.data(), sz, 0);
        FILE * f = fopen(blob_file.c_str(), "wb");
        if (!f) { fprintf(stderr, "error: open blob '%s'\n", blob_file.c_str()); return 2; }
        fwrite(&w, sizeof(size_t), 1, f); fwrite(&first, sizeof(llama_token), 1, f); fwrite(buf.data(), 1, w, f);
        fclose(f);
        fprintf(stderr, "[kvsave] prompt_len=%d KV blob=%zu bytes (%.1f MB) first_tok=%d\n",
                prompt_len, w, w/1e6, first);
    } else {
        size_t w = 0;
        FILE * f = fopen(blob_file.c_str(), "rb");
        if (!f) { fprintf(stderr, "error: open blob '%s'\n", blob_file.c_str()); return 2; }
        if (fread(&w, sizeof(size_t), 1, f) != 1 || fread(&first, sizeof(llama_token), 1, f) != 1) { fclose(f); return 2; }
        std::vector<uint8_t> buf(w);
        if (fread(buf.data(), 1, w, f) != w) { fprintf(stderr, "error: short blob read\n"); fclose(f); return 2; }
        fclose(f);
        size_t r = llama_state_seq_set_data(ctx, buf.data(), w, 0);
        if (r == 0) { fprintf(stderr, "error: llama_state_seq_set_data FAILED (cross-backend KV incompatible?)\n"); return 3; }
        fprintf(stderr, "[kvload] restored %zu bytes KV (set_data ret=%zu) first_tok=%d\n", w, r, first);
    }

    llama_batch b = llama_batch_init(1, 0, 1);
    llama_token tok = first;
    printf(save ? "KVSAVE_GEN %d" : "KVLOAD_GEN %d", first);
    for (int g = 1; g < n_gen; ++g) {
        b.n_tokens = 1; b.token[0] = tok; b.pos[0] = prompt_len + (g - 1);
        b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "error: gen decode g=%d\n", g); rc = 2; break; }
        const float * lg = llama_get_logits_ith(ctx, 0); if (!lg) { rc = 2; break; }
        int best = 0; float bv = lg[0]; for (int i = 1; i < n_vocab; ++i) if (lg[i] > bv) { bv = lg[i]; best = i; }
        printf(" %d", best); tok = best;
    }
    printf("\n"); fflush(stdout);
    llama_batch_free(b);
    return rc;
}

// [plan-a port] act-file v2: { int32 n_embd, int32 n_tokens(N), int32 tokens[N], float residual[N*n_embd] }.
// Carries the relayed input token ids (needed at EVERY stage to rebuild gemma-3n per-layer token
// embeddings) alongside the N cut-hidden residuals, one per prefill position.
static bool write_actfile(const std::string & path, int n_embd,
                          const std::vector<llama_token> & toks, const std::vector<float> & residual) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "error: cannot open act-file '%s' for writing\n", path.c_str()); return false; }
    const int32_t ne = (int32_t) n_embd, N = (int32_t) toks.size();
    std::vector<int32_t> t32(toks.begin(), toks.end());
    bool ok = fwrite(&ne, sizeof(int32_t), 1, f) == 1
           && fwrite(&N,  sizeof(int32_t), 1, f) == 1
           && fwrite(t32.data(), sizeof(int32_t), (size_t) N, f) == (size_t) N
           && fwrite(residual.data(), sizeof(float), (size_t) N * n_embd, f) == (size_t) N * n_embd;
    fclose(f);
    if (!ok) fprintf(stderr, "error: short write to act-file '%s'\n", path.c_str());
    return ok;
}

static bool read_actfile(const std::string & path, int n_embd,
                         std::vector<llama_token> & toks, std::vector<float> & residual) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "error: cannot open act-file '%s' for reading\n", path.c_str()); return false; }
    int32_t ne = 0, N = 0;
    if (fread(&ne, sizeof(int32_t), 1, f) != 1 || fread(&N, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "error: failed to read act-file header\n"); fclose(f); return false;
    }
    if (ne != n_embd) { fprintf(stderr, "error: act-file n_embd=%d != model n_embd=%d\n", ne, n_embd); fclose(f); return false; }
    if (N <= 0)       { fprintf(stderr, "error: act-file n_tokens=%d invalid\n", N); fclose(f); return false; }
    std::vector<int32_t> t32((size_t) N);
    if (fread(t32.data(), sizeof(int32_t), (size_t) N, f) != (size_t) N) {
        fprintf(stderr, "error: failed to read %d token ids from act-file\n", N); fclose(f); return false;
    }
    toks.assign(t32.begin(), t32.end());
    residual.resize((size_t) N * n_embd);
    if (fread(residual.data(), sizeof(float), (size_t) N * n_embd, f) != (size_t) N * n_embd) {
        fprintf(stderr, "error: failed to read %dx%d activation floats from act-file\n", N, n_embd); fclose(f); return false;
    }
    fclose(f);
    return true;
}

// [plan-a port] read just the n_tokens header (so tail/mid can size their context before decoding).
static int peek_actfile_ntokens(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return -1;
    int32_t ne = 0, N = 0;
    const bool ok = fread(&ne, sizeof(int32_t), 1, f) == 1 && fread(&N, sizeof(int32_t), 1, f) == 1;
    fclose(f);
    return ok ? (int) N : -1;
}

// monogen: whole-model greedy reference for the correctness oracle. Generates n_gen tokens
// from `tok` (greedy argmax, feeding back) and prints the token-id sequence. Run with NO
// LLAMA_LAYER_START/END (full 48 L) → this is the ground truth the head∥tail relay must match.
static int run_monogen(llama_context * ctx, const llama_vocab * vocab,
                       int n_vocab, llama_token tok, int n_gen) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    printf("MONOGEN");
    int rc = 0;
    for (int g = 0; g < n_gen; ++g) {
        batch.n_tokens = 1; batch.token[0] = tok; batch.pos[0] = g;
        batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "error: decode (monogen g=%d)\n", g); rc = 2; break; }
        const float * lg = llama_get_logits_ith(ctx, 0);
        if (!lg) { rc = 2; break; }
        int best = 0; float bv = lg[0];
        for (int i = 1; i < n_vocab; ++i) if (lg[i] > bv) { bv = lg[i]; best = i; }
        printf(" %d", best); tok = best;
    }
    printf("\n"); fflush(stdout);
    llama_batch_free(batch);
    return rc;
}

// mono / head share the literal-token prefill path (N tokens at pos 0..N-1).
//   mono : full model -> report the last token's next-token logits.
//   head : run layers [0,LLAMA_LAYER_END) -> dump all N cut residuals + the N token ids (act-file v2).
static int run_mono_or_head(llama_context * ctx, const llama_vocab * vocab,
                            int n_embd, int n_vocab, const std::vector<llama_token> & toks,
                            bool is_head, const std::string & act_file) {
    if (is_head) {
        // expose the cut hidden state for ALL token positions (masked=false dumps every row).
        llama_set_embeddings_nextn(ctx, true, false);
    }

    const int N = (int) toks.size();
    llama_batch batch = llama_batch_init(N, 0, 1);
    batch.n_tokens = N;
    for (int i = 0; i < N; ++i) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == N - 1); // only the last token predicts the next
    }

    fprintf(stderr, "[%s] %d-token prefill (n_embd=%d, n_vocab=%d)\n",
            is_head ? "head" : "mono", N, n_embd, n_vocab);

    int rc = 0;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "error: llama_decode failed\n");
        rc = 2;
    } else if (!is_head) {
        const float * logits = llama_get_logits_ith(ctx, N - 1);
        if (!logits) { fprintf(stderr, "error: llama_get_logits_ith returned NULL\n"); rc = 2; }
        else         { report_logits(vocab, logits, n_vocab); }
    } else {
        // collect all N cut-hidden rows (t_h_nextn) in token order, relay them + the token ids.
        std::vector<float> residual((size_t) N * n_embd);
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            const float * h = llama_get_embeddings_nextn_ith(ctx, i);
            if (!h) { fprintf(stderr, "error: llama_get_embeddings_nextn_ith(%d) NULL (LLAMA_LAYER_END<n_layer?)\n", i); ok = false; break; }
            memcpy(residual.data() + (size_t) i * n_embd, h, (size_t) n_embd * sizeof(float));
        }
        if (!ok || !write_actfile(act_file, n_embd, toks, residual)) rc = 2;
        else {
            const float * h0 = residual.data();
            double s = 0.0, smax = -1e30, smin = 1e30;
            for (int i = 0; i < n_embd; ++i) { s += h0[i]; smax = std::max(smax,(double)h0[i]); smin = std::min(smin,(double)h0[i]); }
            fprintf(stderr, "[head] wrote %dx%d residual + %d token ids to %s (row0 mean=%.5f min=%.5f max=%.5f)\n",
                    N, n_embd, N, act_file.c_str(), s / n_embd, smin, smax);
        }
    }

    llama_batch_free(batch);
    return rc;
}

// [plan-a port] Build a DUAL batch of N tokens: token[i] = relayed id (rebuilds gemma-3n per-layer
// token embeddings + scaled token embedding), embd[i] = injected residual (becomes inpL for the
// stage's layers). llama_batch_init(.,n_embd,.) allocates only embd, so allocate token ourselves
// (llama_batch_free releases it). Only the last token is flagged for output.
static llama_batch make_inject_batch(int n_embd, const std::vector<llama_token> & toks,
                                     const std::vector<float> & act) {
    const int N = (int) toks.size();
    llama_batch batch = llama_batch_init(N, n_embd, 1);
    batch.n_tokens = N;
    batch.token    = (llama_token *) malloc((size_t) N * sizeof(llama_token));
    for (int i = 0; i < N; ++i) {
        batch.token[i]     = toks[i];
        memcpy((float *) batch.embd + (size_t) i * n_embd, act.data() + (size_t) i * n_embd, (size_t) n_embd * sizeof(float));
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == N - 1);
    }
    return batch;
}

static int run_tail(llama_context * ctx, const llama_vocab * vocab,
                    int n_embd, int n_vocab, const std::string & act_file) {
    std::vector<llama_token> toks;
    std::vector<float> act;
    if (!read_actfile(act_file, n_embd, toks, act)) return 2;
    const int N = (int) toks.size();

    llama_batch batch = make_inject_batch(n_embd, toks, act);
    fprintf(stderr, "[tail] injecting %d activations + relayed token ids (dual batch)\n", N);

    int rc = 0;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "error: llama_decode failed (mode=tail)\n");
        rc = 2;
    } else {
        const float * logits = llama_get_logits_ith(ctx, N - 1);
        if (!logits) { fprintf(stderr, "error: llama_get_logits_ith returned NULL (tail)\n"); rc = 2; }
        else         { report_logits(vocab, logits, n_vocab); }
    }

    llama_batch_free(batch);
    return rc;
}

// [plan-a port] mid: a MIDDLE cross-device stage (env LLAMA_LAYER_START=k2, LLAMA_LAYER_END=k3, both
// strictly inside (0,n_layer)). Injects the relayed {tokens, residual} from the previous stage as a
// dual batch, runs layers [k2,k3) HEAD-LESS (le<n_layer), then relays the new {tokens, residual}
// onward. Reads --act-file (in), writes --act-out (out). The gemma4 graph already supports
// ls>0 && le<n_layer with no source change.
static int run_mid(llama_context * ctx, int n_embd,
                   const std::string & act_in, const std::string & act_out) {
    std::vector<llama_token> toks;
    std::vector<float> act;
    if (!read_actfile(act_in, n_embd, toks, act)) return 2;
    const int N = (int) toks.size();

    llama_set_embeddings_nextn(ctx, true, false); // expose the cut hidden for all N rows
    llama_batch batch = make_inject_batch(n_embd, toks, act);
    fprintf(stderr, "[mid] inject %d activations + relayed tokens, run head-less, relay onward\n", N);

    int rc = 0;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "error: llama_decode failed (mode=mid)\n");
        rc = 2;
    } else {
        std::vector<float> residual((size_t) N * n_embd);
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            const float * h = llama_get_embeddings_nextn_ith(ctx, i);
            if (!h) { fprintf(stderr, "error: llama_get_embeddings_nextn_ith(%d) NULL (mid)\n", i); ok = false; break; }
            memcpy(residual.data() + (size_t) i * n_embd, h, (size_t) n_embd * sizeof(float));
        }
        if (!ok || !write_actfile(act_out, n_embd, toks, residual)) rc = 2;
        else fprintf(stderr, "[mid] wrote %dx%d residual + %d token ids to %s\n", N, n_embd, N, act_out.c_str());
    }

    llama_batch_free(batch);
    return rc;
}

// ---------------------------------------------------------------------------
// tailbench: BATCHED tail decode. One forward pass over layers [LLAMA_LAYER_START,
// n_layer) carries n_streams tokens (one injected activation per sequence) -> the
// 40-layer weight load is amortized across all streams (continuous-batch serving).
// Measures the saturated batched-tail throughput + (with external power) J/tok.
// Activation VALUES are irrelevant to energy/throughput, so we inject a fixed dummy.
// ---------------------------------------------------------------------------
static double now_epoch() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static int run_tailbench(llama_context * ctx, int n_embd, int n_streams, int n_steps, double gap_ms) {
    llama_batch batch = llama_batch_init(n_streams, n_embd, 1);

    std::vector<float> hidden((size_t) n_embd);
    for (int i = 0; i < n_embd; ++i) hidden[i] = 0.001f * ((i % 17) - 8);  // non-denormal dummy

    fprintf(stderr, "[tailbench] n_streams=%d n_steps=%d n_embd=%d gap=%.1fms (3 warmup steps untimed)\n",
            n_streams, n_steps, n_embd, gap_ms);

    // gap_ms models the per-token cross-device head-wait: with n_streams PARALLEL phone heads,
    // the tail idles ~gap_ms each round waiting for all heads, then batch-decodes B=n_streams.
    // The idle is REAL idle-burn (GPU doesn't downclock in ~31ms) -> NVML measures it.
    struct timespec gts{ (time_t)(gap_ms/1000.0), (long)((gap_ms - 1000.0*(long)(gap_ms/1000.0))*1e6) };

    double t0 = 0.0;
    const int total = 3 + n_steps;
    for (int s = 0; s < total; ++s) {
        if (gap_ms > 0.0 && s >= 3) nanosleep(&gts, nullptr);   // head-wait before each timed round
        batch.n_tokens = n_streams;
        for (int j = 0; j < n_streams; ++j) {
            memcpy(batch.embd + (size_t) j * n_embd, hidden.data(), (size_t) n_embd * sizeof(float));
            batch.pos[j]       = s;          // each seq advances 1 pos/step
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = j;          // distinct KV per stream
            batch.logits[j]    = 1;          // realistic: lm_head + sample each token
        }
        if (s == 3) { t0 = now_epoch(); printf("BENCH_START %.3f streams=%d\n", t0, n_streams); fflush(stdout); }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "error: llama_decode failed (tailbench, step=%d)\n", s);
            llama_batch_free(batch);
            return 2;
        }
    }
    double t1 = now_epoch();
    long toks = (long) n_streams * n_steps;
    printf("BENCH_END %.3f tokens=%ld throughput=%.2f tok/s\n", t1, toks, toks / (t1 - t0));
    fflush(stdout);

    llama_batch_free(batch);
    return 0;
}

// ---------------------------------------------------------------------------
// socket relay helpers (raw TCP, blocking, one client, fixed-size frames)
// ---------------------------------------------------------------------------

// Send exactly n bytes, looping over partial writes. Returns true on success.
static bool send_all(int fd, const void * buf, size_t n) {
    const char * p = (const char *) buf;
    size_t sent = 0;
    while (sent < n) {
        ssize_t k = send(fd, p + sent, n - sent, 0);
        if (k <= 0) {
            if (k < 0 && (errno == EINTR)) continue;
            return false;
        }
        sent += (size_t) k;
    }
    return true;
}

// Recv exactly n bytes, looping over partial reads. Returns true on success.
// (false on clean EOF or error -- caller treats both as fatal.)
static bool recv_all(int fd, void * buf, size_t n) {
    char * p = (char *) buf;
    size_t got = 0;
    while (got < n) {
        ssize_t k = recv(fd, p + got, n - got, 0);
        if (k == 0) return false;            // peer closed
        if (k < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        got += (size_t) k;
    }
    return true;
}

static void set_nodelay(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

// tailnet: own layers [LLAMA_LAYER_START, n_layer) + sampling.
// listen on 0.0.0.0:port, accept one client, then per frame:
//   recv { int32 pos; int32 n_embd; n_embd*float32 hidden }   (pos<0 => exit)
//   inject hidden via batch.embd (token NULL), decode, greedy argmax over logits,
//   send back { int32 token }.
static int run_tailnet(llama_context * ctx, const llama_vocab * vocab,
                       int n_embd, int n_vocab, int port) {
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { fprintf(stderr, "error: socket() failed: %s\n", strerror(errno)); return 3; }
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons((uint16_t) port);

    if (bind(srv, (sockaddr *) &addr, sizeof(addr)) < 0) {
        fprintf(stderr, "error: bind(0.0.0.0:%d) failed: %s\n", port, strerror(errno));
        close(srv); return 3;
    }
    if (listen(srv, 1) < 0) {
        fprintf(stderr, "error: listen() failed: %s\n", strerror(errno));
        close(srv); return 3;
    }
    fprintf(stderr, "[tailnet] listening on 0.0.0.0:%d (n_embd=%d, n_vocab=%d)\n", port, n_embd, n_vocab);

    int cli = accept(srv, nullptr, nullptr);
    if (cli < 0) {
        fprintf(stderr, "error: accept() failed: %s\n", strerror(errno));
        close(srv); return 3;
    }
    set_nodelay(cli);
    fprintf(stderr, "[tailnet] client connected\n");

    // reusable 1-token EMBD batch (token NULL, embd injected each frame).
    llama_batch batch = llama_batch_init(1, n_embd, 1);

    std::vector<float> hidden((size_t) n_embd);
    int rc = 0;
    long n_steps = 0;
    while (true) {
        int32_t pos = 0, ne = 0;
        if (!recv_all(cli, &pos, sizeof(pos))) {
            fprintf(stderr, "error: recv(pos) failed/EOF after %ld steps\n", n_steps);
            rc = 3; break;
        }
        if (pos < 0) {
            fprintf(stderr, "[tailnet] sentinel pos=%d received, exiting after %ld steps\n", pos, n_steps);
            break;
        }
        if (!recv_all(cli, &ne, sizeof(ne))) {
            fprintf(stderr, "error: recv(n_embd) failed/EOF\n"); rc = 3; break;
        }
        if (ne != n_embd) {
            fprintf(stderr, "error: frame n_embd=%d != model n_embd=%d\n", ne, n_embd);
            rc = 3; break;
        }
        if (!recv_all(cli, hidden.data(), (size_t) n_embd * sizeof(float))) {
            fprintf(stderr, "error: recv(hidden) failed/EOF\n"); rc = 3; break;
        }

        batch.n_tokens     = 1;
        // batch.token stays NULL (gemma4 applies no embd scale when token==NULL).
        memcpy(batch.embd, hidden.data(), (size_t) n_embd * sizeof(float));
        batch.pos[0]       = pos;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "error: llama_decode failed (tailnet, pos=%d)\n", pos);
            rc = 3; break;
        }
        const float * logits = llama_get_logits_ith(ctx, 0);
        if (!logits) { fprintf(stderr, "error: llama_get_logits_ith NULL (tailnet)\n"); rc = 3; break; }

        // greedy argmax
        int32_t best = 0;
        float   bestv = logits[0];
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > bestv) { bestv = logits[i]; best = i; }
        }

        if (!send_all(cli, &best, sizeof(best))) {
            fprintf(stderr, "error: send(token) failed (pos=%d)\n", pos); rc = 3; break;
        }
        n_steps++;
    }

    llama_batch_free(batch);
    close(cli);
    close(srv);
    return rc;
}

// headnet: own layers [0, LLAMA_LAYER_END). connect host:port, then drive the loop.
// For each prompt token, then each of n_gen generated tokens:
//   decode literal token over [0,k), read cut hidden state, send {pos,n_embd,h},
//   recv {token}. The recv'd token becomes the next input token (ignored for all
//   prompt tokens except the last; after the prompt it is the generated token).
static int run_headnet(llama_context * ctx, const llama_vocab * vocab,
                       int n_embd, const std::string & host, int port,
                       const std::string & prompt, int n_gen) {
    // resolve + connect
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { fprintf(stderr, "error: socket() failed: %s\n", strerror(errno)); return 3; }

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t) port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        // fall back to DNS resolution
        struct addrinfo hints, * res = nullptr;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family   = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) {
            fprintf(stderr, "error: cannot resolve host '%s'\n", host.c_str());
            close(fd); return 3;
        }
        addr.sin_addr = ((sockaddr_in *) res->ai_addr)->sin_addr;
        freeaddrinfo(res);
    }
    if (connect(fd, (sockaddr *) &addr, sizeof(addr)) < 0) {
        fprintf(stderr, "error: connect(%s:%d) failed: %s\n", host.c_str(), port, strerror(errno));
        close(fd); return 3;
    }
    set_nodelay(fd);
    fprintf(stderr, "[headnet] connected to %s:%d\n", host.c_str(), port);

    // expose the cut hidden state (masked=false => all rows; single row anyway).
    llama_set_embeddings_nextn(ctx, true, false);

    // tokenize prompt with BOS.
    std::vector<llama_token> prompt_tokens = common_tokenize(vocab, prompt, true, true);
    if (prompt_tokens.empty()) {
        fprintf(stderr, "error: prompt tokenized to 0 tokens\n");
        close(fd); return 3;
    }
    fprintf(stderr, "[headnet] prompt='%s' (%zu tokens), n_gen=%d\n",
            prompt.c_str(), prompt_tokens.size(), n_gen);

    llama_batch batch = llama_batch_init(1, 0, 1);

    auto exchange = [&](int32_t pos, llama_token tok, llama_token & out_tok) -> bool {
        batch.n_tokens     = 1;
        batch.token[0]     = tok;
        batch.pos[0]       = pos;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "error: llama_decode failed (headnet, pos=%d)\n", pos);
            return false;
        }
        const float * h = llama_get_embeddings_nextn(ctx);
        if (!h) {
            fprintf(stderr, "error: llama_get_embeddings_nextn NULL "
                            "(is LLAMA_LAYER_END < n_layer set?)\n");
            return false;
        }
        int32_t ne = (int32_t) n_embd;
        if (!send_all(fd, &pos, sizeof(pos)) ||
            !send_all(fd, &ne,  sizeof(ne))  ||
            !send_all(fd, h, (size_t) n_embd * sizeof(float))) {
            fprintf(stderr, "error: send(frame) failed (pos=%d)\n", pos);
            return false;
        }
        if (!recv_all(fd, &out_tok, sizeof(out_tok))) {
            fprintf(stderr, "error: recv(token) failed/EOF (pos=%d)\n", pos);
            return false;
        }
        return true;
    };

    int rc = 0;
    int32_t pos = 0;
    llama_token next_tok = 0;
    std::string generated;

    // 1) feed the prompt tokens; ignore returned token until the LAST prompt token.
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        llama_token recv_tok = 0;
        if (!exchange(pos, prompt_tokens[i], recv_tok)) { rc = 3; goto done; }
        if (i + 1 == prompt_tokens.size()) {
            next_tok = recv_tok;   // first generated token
        }
        pos++;
    }

    // 2) generation: the token recv'd after the prompt is the generated token.
    printf("\n=== GENERATED ===\n%s", prompt.c_str());
    fflush(stdout);
    for (int g = 0; g < n_gen; ++g) {
        std::string piece = common_token_to_piece(ctx, next_tok, true);
        generated += piece;
        printf("%s", piece.c_str());
        fflush(stdout);
        if (llama_vocab_is_eog(vocab, next_tok)) {
            fprintf(stderr, "\n[headnet] EOG token at gen step %d, stopping\n", g);
            break;
        }
        if (g + 1 >= n_gen) break;   // last token already printed; no need to feed again
        llama_token recv_tok = 0;
        if (!exchange(pos, next_tok, recv_tok)) { rc = 3; goto done; }
        next_tok = recv_tok;
        pos++;
    }
    printf("\n=================\n");
    fflush(stdout);

done:
    // sentinel: tell tail to exit.
    {
        int32_t sentinel = -1;
        send_all(fd, &sentinel, sizeof(sentinel));
    }
    llama_batch_free(batch);
    close(fd);
    return rc;
}

// ---------------------------------------------------------------------------
// [plan-a port] Persistent 3-stage pipeline over TCP (adb-forwarded USB).
// HUB-AND-SPOKE (phones can't peer): each phone runs a `stagenet` — a persistent
// HEAD-LESS layer-range server holding its own KV; the host runs `pipedriver`,
// which owns the terminal tail stage inline and drives the loop, calling stage A
// (op15 [0,k2)) then stage B (op12 [k2,k3)) then the local tail [k3,n) per token.
// Unlike tailnet/headnet, the frame carries the token id and each stage injects a
// DUAL batch (relayed token + injected residual) — required for gemma-3n per-layer
// embeddings. KV persists across steps, so this decodes incrementally (no re-prefill).
// ---------------------------------------------------------------------------
static int connect_to(const std::string & host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    sockaddr_in addr; memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET; addr.sin_port = htons((uint16_t) port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        struct addrinfo hints, * res = nullptr; memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) { close(fd); return -1; }
        addr.sin_addr = ((sockaddr_in *) res->ai_addr)->sin_addr; freeaddrinfo(res);
    }
    if (connect(fd, (sockaddr *) &addr, sizeof(addr)) < 0) { close(fd); return -1; }
    set_nodelay(fd);
    return fd;
}

// stagenet: persistent HEAD-LESS stage server. env LLAMA_LAYER_START/END bound [ls,le) (le<n_layer).
//   request: { i32 pos; i32 tok; i32 nh; nh*f32 hidden }   (pos<0 => exit)
//   reply:   { i32 ne; ne*f32 hidden }                     (cut residual via nextn)
// ls==0 (head): token batch (token embedding). ls>0 (mid): DUAL batch {relayed token + residual}.
static int run_stagenet(llama_context * ctx, int n_embd, int port) {
    llama_set_embeddings_nextn(ctx, true, false);
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { fprintf(stderr, "error: socket(): %s\n", strerror(errno)); return 3; }
    int one = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr; memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_ANY); addr.sin_port = htons((uint16_t) port);
    if (bind(srv, (sockaddr *) &addr, sizeof(addr)) < 0) { fprintf(stderr, "error: bind(:%d): %s\n", port, strerror(errno)); close(srv); return 3; }
    if (listen(srv, 1) < 0) { fprintf(stderr, "error: listen(): %s\n", strerror(errno)); close(srv); return 3; }
    fprintf(stderr, "[stagenet] listening on 0.0.0.0:%d (n_embd=%d)\n", port, n_embd);
    int cli = accept(srv, nullptr, nullptr);
    if (cli < 0) { fprintf(stderr, "error: accept(): %s\n", strerror(errno)); close(srv); return 3; }
    set_nodelay(cli);
    fprintf(stderr, "[stagenet] client connected\n");

    llama_batch tb = llama_batch_init(1, 0, 1);              // token-only (head, ls==0)
    llama_batch db = llama_batch_init(1, n_embd, 1);         // dual (mid, ls>0): embd + our token
    db.token = (llama_token *) malloc(sizeof(llama_token));  // freed by llama_batch_free
    std::vector<float> hidden((size_t) n_embd);
    int rc = 0; long steps = 0;
    while (true) {
        int32_t pos = 0, tok = 0, nh = 0;
        if (!recv_all(cli, &pos, sizeof(pos))) { fprintf(stderr, "[stagenet] EOF after %ld steps\n", steps); break; }
        if (pos < 0) { fprintf(stderr, "[stagenet] exit after %ld steps\n", steps); break; }
        if (!recv_all(cli, &tok, sizeof(tok)) || !recv_all(cli, &nh, sizeof(nh))) { rc = 3; break; }
        if (nh > 0 && !recv_all(cli, hidden.data(), (size_t) nh * sizeof(float))) { rc = 3; break; }

        llama_batch * b;
        if (nh > 0) {
            db.n_tokens = 1; db.token[0] = tok; memcpy(db.embd, hidden.data(), (size_t) n_embd * sizeof(float));
            db.pos[0] = pos; db.n_seq_id[0] = 1; db.seq_id[0][0] = 0; db.logits[0] = 1; b = &db;
        } else {
            tb.n_tokens = 1; tb.token[0] = tok; tb.pos[0] = pos;
            tb.n_seq_id[0] = 1; tb.seq_id[0][0] = 0; tb.logits[0] = 1; b = &tb;
        }
        if (llama_decode(ctx, *b) != 0) { fprintf(stderr, "error: decode (stagenet pos=%d)\n", pos); rc = 3; break; }
        const float * h = llama_get_embeddings_nextn(ctx);
        if (!h) { fprintf(stderr, "error: nextn NULL (stagenet; LLAMA_LAYER_END<n_layer?)\n"); rc = 3; break; }
        int32_t ne = (int32_t) n_embd;
        if (!send_all(cli, &ne, sizeof(ne)) || !send_all(cli, h, (size_t) n_embd * sizeof(float))) { rc = 3; break; }
        steps++;
    }
    llama_batch_free(tb); llama_batch_free(db);
    close(cli); close(srv);
    return rc;
}

// pipedriver: host orchestrator + inline TAIL (env LLAMA_LAYER_START=k3). Connects to stage A
// (op15 [0,k2)) and stage B (op12 [k2,k3)) over TCP (adb-forwarded USB), drives incremental decode.
// [plan-a port] format a single user message with the model's chat template (Jinja). gemma-4-it
// needs its channel-based template — raw prompts are out-of-distribution and generate degenerately.
static std::string apply_chat_template(const llama_model * model, const std::string & user_msg) {
    common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
    common_chat_templates_inputs in;
    common_chat_msg m; m.role = "user"; m.content = user_msg;
    in.messages = { m };
    in.add_generation_prompt = true;
    in.use_jinja = true;
    in.add_bos   = false; // BOS is added at tokenization (common_tokenize add_special=true)
    return common_chat_templates_apply(tmpls.get(), in).prompt;
}

static int run_pipedriver(llama_context * ctx, const llama_vocab * vocab, int n_embd, int n_vocab,
                          const std::string & host, int portA, int portB,
                          const std::string & prompt, int n_gen, bool chat_mode) {
    int fdA = connect_to(host, portA);
    if (fdA < 0) { fprintf(stderr, "error: connect A %s:%d failed\n", host.c_str(), portA); return 3; }
    int fdB = connect_to(host, portB);
    if (fdB < 0) { fprintf(stderr, "error: connect B %s:%d failed\n", host.c_str(), portB); close(fdA); return 3; }
    fprintf(stderr, "[pipedriver] connected A=%s:%d B=%s:%d\n", host.c_str(), portA, host.c_str(), portB);

    const std::string eff = chat_mode ? apply_chat_template(llama_get_model(ctx), prompt) : prompt;
    // add BOS (gemma requires it) + parse the template's special tokens (<|channel> etc.)
    std::vector<llama_token> ptoks = common_tokenize(vocab, eff, true, true);
    if (ptoks.empty()) { fprintf(stderr, "error: prompt -> 0 tokens\n"); close(fdA); close(fdB); return 3; }
    fprintf(stderr, "[pipedriver] chat=%d prompt='%s' (%zu tok), n_gen=%d\n", (int)chat_mode, prompt.c_str(), ptoks.size(), n_gen);

    llama_batch tail = llama_batch_init(1, n_embd, 1);
    tail.token = (llama_token *) malloc(sizeof(llama_token));
    std::vector<float> hh((size_t) n_embd), hm((size_t) n_embd);

    // [plan-a port] per-hop timing: wall-clock spent in stage A (op15 = RTT+compute),
    // stage B (op12 = RTT+compute), and the inline host tail (A6000 CUDA compute).
    // Only the generation-phase steps are timed (decode), not prompt prefill.
    using clk = std::chrono::steady_clock;
    double us_A = 0, us_B = 0, us_T = 0; long timed = 0; bool do_time = false;
    auto now_us = [&]() { return std::chrono::duration<double, std::micro>(clk::now().time_since_epoch()).count(); };

    auto stage = [&](int fd, int32_t pos, int32_t tok, const float * hin, int32_t nh, float * hout) -> bool {
        if (!send_all(fd, &pos, sizeof(pos)) || !send_all(fd, &tok, sizeof(tok)) || !send_all(fd, &nh, sizeof(nh))) return false;
        if (nh > 0 && !send_all(fd, hin, (size_t) nh * sizeof(float))) return false;
        int32_t ne = 0;
        if (!recv_all(fd, &ne, sizeof(ne)) || ne != n_embd) return false;
        return recv_all(fd, hout, (size_t) n_embd * sizeof(float));
    };
    auto step = [&](int32_t pos, llama_token tok, llama_token & out) -> bool {
        double a0 = do_time ? now_us() : 0;
        if (!stage(fdA, pos, tok, nullptr, 0, hh.data()))       { fprintf(stderr, "error: stage A (pos=%d)\n", pos); return false; }
        double a1 = do_time ? now_us() : 0;
        if (!stage(fdB, pos, tok, hh.data(), n_embd, hm.data())) { fprintf(stderr, "error: stage B (pos=%d)\n", pos); return false; }
        double a2 = do_time ? now_us() : 0;
        tail.n_tokens = 1; tail.token[0] = tok; memcpy(tail.embd, hm.data(), (size_t) n_embd * sizeof(float));
        tail.pos[0] = pos; tail.n_seq_id[0] = 1; tail.seq_id[0][0] = 0; tail.logits[0] = 1;
        if (llama_decode(ctx, tail) != 0) { fprintf(stderr, "error: tail decode (pos=%d)\n", pos); return false; }
        const float * lg = llama_get_logits_ith(ctx, 0);
        if (!lg) return false;
        int32_t best = 0; float bv = lg[0];
        for (int i = 1; i < n_vocab; ++i) if (lg[i] > bv) { bv = lg[i]; best = i; }
        if (do_time) { double a3 = now_us(); us_A += a1 - a0; us_B += a2 - a1; us_T += a3 - a2; timed++; }
        out = best; return true;
    };

    int rc = 0; int32_t pos = 0; llama_token next = 0;
    for (size_t i = 0; i < ptoks.size(); ++i) {
        llama_token o = 0;
        if (!step(pos, ptoks[i], o)) { rc = 3; goto done; }
        if (i + 1 == ptoks.size()) next = o;
        pos++;
    }
    printf("\n=== GENERATED ===\n%s", prompt.c_str()); fflush(stdout);
    do_time = true;   // time only the generation-phase steps (steady-state decode)
    for (int g = 0; g < n_gen; ++g) {
        std::string pc = common_token_to_piece(ctx, next, true);
        printf("%s", pc.c_str()); fflush(stdout);
        if (llama_vocab_is_eog(vocab, next)) { fprintf(stderr, "\n[pipedriver] EOG at gen %d\n", g); break; }
        if (g + 1 >= n_gen) break;
        llama_token o = 0;
        if (!step(pos, next, o)) { rc = 3; goto done; }
        next = o; pos++;
    }
    printf("\n=================\n"); fflush(stdout);
    if (timed > 0) {
        double tot = (us_A + us_B + us_T) / timed / 1000.0;
        fprintf(stderr, "[pipedriver] per-tok decode breakdown over %ld steps (ms):\n"
                        "    stageA op15 (RTT+compute) = %.2f\n"
                        "    stageB op12 (RTT+compute) = %.2f\n"
                        "    tail   A6000 (CUDA)       = %.2f\n"
                        "    total decode/tok          = %.2f\n",
                timed, us_A / timed / 1000.0, us_B / timed / 1000.0, us_T / timed / 1000.0, tot);
    }
done:
    { int32_t s = -1; send_all(fdA, &s, sizeof(s)); send_all(fdB, &s, sizeof(s)); }
    llama_batch_free(tail);
    close(fdA); close(fdB);
    return rc;
}

// ===========================================================================
// STREAMING multi-caption relay (tailstream / headstream) — replays a trace.
// Opcodes (int32, head->tail):  RESET=3 clear KV + ack ; DECODE=2 {pos,n_embd,h}
//   -> reply {token} ; SHUTDOWN=4 exit. Each caption = N decode steps then RESET.
// Reuses the validated single-token relay; decode runs from a short context (KV-size
// affects decode energy <2% vs the 6.5GB/tok weight stream — prefill measured separately).
// ===========================================================================
enum { OP_DECODE = 2, OP_RESET = 3, OP_SHUTDOWN = 4 };

static int run_tailstream(llama_context * ctx, const llama_vocab * vocab,
                          int n_embd, int n_vocab, int port) {
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { fprintf(stderr, "error: socket(): %s\n", strerror(errno)); return 3; }
    int one = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr; memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_ANY); addr.sin_port = htons((uint16_t) port);
    if (bind(srv, (sockaddr *) &addr, sizeof(addr)) < 0) { fprintf(stderr, "error: bind(%d): %s\n", port, strerror(errno)); close(srv); return 3; }
    if (listen(srv, 1) < 0) { fprintf(stderr, "error: listen(): %s\n", strerror(errno)); close(srv); return 3; }
    fprintf(stderr, "[tailstream] listening 0.0.0.0:%d (n_embd=%d)\n", port, n_embd);
    int cli = accept(srv, nullptr, nullptr);
    if (cli < 0) { fprintf(stderr, "error: accept(): %s\n", strerror(errno)); close(srv); return 3; }
    set_nodelay(cli);
    fprintf(stderr, "[tailstream] client connected\n");

    llama_memory_t mem = llama_get_memory(ctx);
    llama_batch batch = llama_batch_init(1, n_embd, 1);
    std::vector<float> hidden((size_t) n_embd);
    long ncap = 0, nsteps = 0; int rc = 0;
    while (true) {
        int32_t op = 0;
        if (!recv_all(cli, &op, sizeof(op))) { fprintf(stderr, "[tailstream] EOF after %ld caps\n", ncap); break; }
        if (op == OP_SHUTDOWN) { fprintf(stderr, "[tailstream] shutdown (%ld caps, %ld steps)\n", ncap, nsteps); break; }
        if (op == OP_RESET) { llama_memory_clear(mem, true); ncap++; int32_t ack = 0; if (!send_all(cli, &ack, sizeof(ack))) { rc = 3; break; } continue; }
        if (op == OP_DECODE) {
            int32_t pos = 0, ne = 0;
            if (!recv_all(cli, &pos, sizeof(pos)) || !recv_all(cli, &ne, sizeof(ne))) { rc = 3; break; }
            if (ne != n_embd) { fprintf(stderr, "error: n_embd %d!=%d\n", ne, n_embd); rc = 3; break; }
            if (!recv_all(cli, hidden.data(), (size_t) n_embd * sizeof(float))) { rc = 3; break; }
            batch.n_tokens = 1; memcpy(batch.embd, hidden.data(), (size_t) n_embd * sizeof(float));
            batch.pos[0] = pos; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "error: decode (tailstream pos=%d)\n", pos); rc = 3; break; }
            const float * logits = llama_get_logits_ith(ctx, 0);
            if (!logits) { rc = 3; break; }
            int32_t best = 0; float bv = logits[0];
            for (int i = 1; i < n_vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
            if (!send_all(cli, &best, sizeof(best))) { rc = 3; break; }
            nsteps++;
        } else { fprintf(stderr, "error: bad opcode %d\n", op); rc = 3; break; }
    }
    llama_batch_free(batch); close(cli); close(srv);
    return rc;
}

// headstream: own [0,k). Reads a schedule file (lines: "arr_ms decode_len"), connects,
// replays each caption at its arrival time, logs per-caption latency. KV reset per caption.
static int run_headstream(llama_context * ctx, int n_embd,
                          const std::string & host, int port, const std::string & sched_file) {
    // load schedule
    std::vector<std::pair<double,int>> sched;
    FILE * sf = fopen(sched_file.c_str(), "r");
    if (!sf) { fprintf(stderr, "error: cannot open sched '%s'\n", sched_file.c_str()); return 3; }
    { double a; int l; while (fscanf(sf, "%lf %d", &a, &l) == 2) sched.push_back({a, l}); }
    fclose(sf);
    if (sched.empty()) { fprintf(stderr, "error: empty schedule\n"); return 3; }
    fprintf(stderr, "[headstream] %zu captions scheduled\n", sched.size());

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { fprintf(stderr, "error: socket(): %s\n", strerror(errno)); return 3; }
    sockaddr_in addr; memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET; addr.sin_port = htons((uint16_t) port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        struct addrinfo hints, * res = nullptr; memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) { fprintf(stderr, "error: resolve '%s'\n", host.c_str()); close(fd); return 3; }
        addr.sin_addr = ((sockaddr_in *) res->ai_addr)->sin_addr; freeaddrinfo(res);
    }
    if (connect(fd, (sockaddr *) &addr, sizeof(addr)) < 0) { fprintf(stderr, "error: connect %s:%d: %s\n", host.c_str(), port, strerror(errno)); close(fd); return 3; }
    set_nodelay(fd);
    fprintf(stderr, "[headstream] connected %s:%d\n", host.c_str(), port);

    llama_set_embeddings_nextn(ctx, true, false);
    llama_memory_t mem = llama_get_memory(ctx);
    llama_batch batch = llama_batch_init(1, 0, 1);
    const llama_token bos = llama_vocab_bos(llama_model_get_vocab(llama_get_model(ctx)));

    auto exchange = [&](int32_t pos, llama_token tok, llama_token & out) -> bool {
        batch.n_tokens = 1; batch.token[0] = tok; batch.pos[0] = pos;
        batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
        if (llama_decode(ctx, batch) != 0) return false;
        const float * h = llama_get_embeddings_nextn(ctx);
        if (!h) return false;
        int32_t op = OP_DECODE, ne = (int32_t) n_embd;
        if (!send_all(fd, &op, sizeof(op)) || !send_all(fd, &pos, sizeof(pos)) ||
            !send_all(fd, &ne, sizeof(ne)) || !send_all(fd, h, (size_t) n_embd * sizeof(float))) return false;
        return recv_all(fd, &out, sizeof(out));
    };

    auto t_start = std::chrono::steady_clock::now();
    auto wall_ms = [&]{ return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count(); };
    printf("CAP idx arr_ms start_ms done_ms lat_ms toks\n"); fflush(stdout);
    int rc = 0;
    for (size_t i = 0; i < sched.size(); ++i) {
        double arr = sched[i].first; int len = sched[i].second;
        while (wall_ms() < arr) { struct timespec ts{0, 500000}; nanosleep(&ts, nullptr); }   // sleep to arrival
        double t0 = wall_ms();
        llama_token tok = bos, out = 0;
        std::string gen_ids;            // token-id trace for the correctness oracle
        for (int g = 0; g < len; ++g) {
            if (!exchange(g, tok, out)) { fprintf(stderr, "error: exchange cap=%zu g=%d\n", i, g); rc = 3; goto done; }
            tok = out;
            gen_ids += " " + std::to_string(out);
        }
        double t1 = wall_ms();
        // reset both KVs for next caption
        llama_memory_clear(mem, true);
        { int32_t op = OP_RESET, ack = 0; if (!send_all(fd, &op, sizeof(op)) || !recv_all(fd, &ack, sizeof(ack))) { rc = 3; goto done; } }
        printf("CAP %zu %.1f %.1f %.1f %.1f %d\n", i, arr, t0, t1, t1 - arr, len); fflush(stdout);
        printf("GEN %zu%s\n", i, gen_ids.c_str()); fflush(stdout);
    }
done:
    { int32_t op = OP_SHUTDOWN; send_all(fd, &op, sizeof(op)); }
    llama_batch_free(batch); close(fd);
    return rc;
}

// ===========================================================================
// DISAGG end-to-end relay (REQ-045): server PREFILLS (whole 48L) + ships resident KV;
// op15 DECODES (whole 48L) from the injected KV; server is RELEASED (prefills next).
//   kvserver (host A6000): per caption: prefill prompt_len tokens -> state_seq_get_data ->
//            send {first_token, size, blob} -> recv ack -> clear seq KV -> next.
//   kvclient (op15 Adreno): recv {first,size,blob} -> state_seq_set_data -> decode n_gen ->
//            send ack -> reset; logs per-caption recv(KV-ship) + decode latency.
// ===========================================================================
static int run_kvserver(llama_context * ctx, const llama_vocab * vocab, int n_vocab,
                        int port, int prompt_len, int n_caps) {
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { fprintf(stderr, "error: socket()\n"); return 3; }
    int one = 1; setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr; memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_ANY); addr.sin_port = htons((uint16_t) port);
    if (bind(srv,(sockaddr*)&addr,sizeof(addr))<0){fprintf(stderr,"error: bind %d\n",port);close(srv);return 3;}
    if (listen(srv,1)<0){close(srv);return 3;}
    fprintf(stderr,"[kvserver] listening 0.0.0.0:%d (prompt_len=%d, %d caps)\n",port,prompt_len,n_caps);
    int cli = accept(srv,nullptr,nullptr); if (cli<0){close(srv);return 3;} set_nodelay(cli);
    fprintf(stderr,"[kvserver] client connected\n");

    llama_memory_t mem = llama_get_memory(ctx);
    const llama_token bos = llama_vocab_bos(vocab);
    std::vector<uint8_t> buf; int rc = 0;
    for (int c = 0; c < n_caps; ++c) {
        llama_batch pf = llama_batch_init(prompt_len, 0, 1);
        pf.n_tokens = prompt_len;
        for (int i=0;i<prompt_len;++i){ pf.token[i]=(i==0)?bos:(llama_token)((i*131+7+c)%n_vocab);
            pf.pos[i]=i; pf.n_seq_id[i]=1; pf.seq_id[i][0]=0; pf.logits[i]=(i==prompt_len-1); }
        if (llama_decode(ctx,pf)!=0){fprintf(stderr,"error: prefill c=%d\n",c);llama_batch_free(pf);rc=3;break;}
        const float* lg=llama_get_logits_ith(ctx,prompt_len-1); int32_t first=0;
        { float bv=lg[0]; for(int i=1;i<n_vocab;++i) if(lg[i]>bv){bv=lg[i];first=i;} }
        llama_batch_free(pf);
        size_t sz=llama_state_seq_get_size(ctx,0); buf.resize(sz);
        size_t w=llama_state_seq_get_data(ctx,buf.data(),sz,0);
        int64_t w64=(int64_t)w;
        if (!send_all(cli,&first,sizeof(first))||!send_all(cli,&w64,sizeof(w64))||!send_all(cli,buf.data(),w)){rc=3;break;}
        int32_t ack=0; if(!recv_all(cli,&ack,sizeof(ack))){rc=3;break;}   // wait for op15 decode done
        llama_memory_clear(mem,true);
    }
    int32_t done=-1; send_all(cli,&done,sizeof(done));
    close(cli); close(srv);
    fprintf(stderr,"[kvserver] done (%d caps)\n",n_caps);
    return rc;
}

static int run_kvclient(llama_context * ctx, int n_vocab,
                        const std::string & host, int port, int prompt_len, int n_gen) {
    int fd = socket(AF_INET, SOCK_STREAM, 0); if (fd<0){return 3;}
    sockaddr_in addr; memset(&addr,0,sizeof(addr)); addr.sin_family=AF_INET; addr.sin_port=htons((uint16_t)port);
    if (inet_pton(AF_INET,host.c_str(),&addr.sin_addr)!=1){
        struct addrinfo hints,*res=nullptr; memset(&hints,0,sizeof(hints)); hints.ai_family=AF_INET; hints.ai_socktype=SOCK_STREAM;
        if (getaddrinfo(host.c_str(),nullptr,&hints,&res)!=0||!res){fprintf(stderr,"error: resolve %s\n",host.c_str());close(fd);return 3;}
        addr.sin_addr=((sockaddr_in*)res->ai_addr)->sin_addr; freeaddrinfo(res);
    }
    if (connect(fd,(sockaddr*)&addr,sizeof(addr))<0){fprintf(stderr,"error: connect %s:%d\n",host.c_str(),port);close(fd);return 3;}
    set_nodelay(fd); fprintf(stderr,"[kvclient] connected %s:%d\n",host.c_str(),port);

    llama_memory_t mem = llama_get_memory(ctx);
    llama_batch b = llama_batch_init(1,0,1);
    std::vector<uint8_t> buf; int rc=0, idx=0;
    auto now=[]{ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    printf("CAP idx recv_ms decode_ms total_ms toks\n"); fflush(stdout);
    while (true) {
        int32_t first=0; if(!recv_all(fd,&first,sizeof(first))){rc=3;break;}
        if (first<0){ fprintf(stderr,"[kvclient] shutdown after %d caps\n",idx); break; }
        int64_t w64=0; if(!recv_all(fd,&w64,sizeof(w64))){rc=3;break;}
        buf.resize((size_t)w64);
        double t0=now();
        if(!recv_all(fd,buf.data(),(size_t)w64)){rc=3;break;}   // KV-ship over USB
        double t_recv=now();
        size_t r=llama_state_seq_set_data(ctx,buf.data(),(size_t)w64,0);
        if(r==0){fprintf(stderr,"error: set_data failed cap=%d\n",idx);rc=3;break;}
        llama_token tok=first;
        for(int g=1;g<n_gen;++g){ b.n_tokens=1;b.token[0]=tok;b.pos[0]=prompt_len+(g-1);
            b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=1;
            if(llama_decode(ctx,b)!=0){rc=3;goto done2;}
            const float* lg=llama_get_logits_ith(ctx,0); if(!lg){rc=3;goto done2;}
            int best=0;float bv=lg[0];for(int i=1;i<n_vocab;++i)if(lg[i]>bv){bv=lg[i];best=i;} tok=best; }
        double t_dec=now();
        { int32_t ack=0; if(!send_all(fd,&ack,sizeof(ack))){rc=3;break;} }
        llama_memory_clear(mem,true);
        printf("CAP %d %.0f %.0f %.0f %d\n", idx, t_recv-t0, t_dec-t_recv, t_dec-t0, n_gen); fflush(stdout);
        idx++;
    }
done2:
    llama_batch_free(b); close(fd);
    return rc;
}

// ===========================================================================
// [plan-a M4] dualengine: ONE process, TWO backends, ONE session (requirement 1).
// Loads the SAME layer shard on two devices — a PREFILL engine (GPU/xmem, runs prompts
// one-by-one) and a DECODE engine (NPU/HMX, static-batched) — and runs them on two threads
// concurrently. The decode engine ACCUMULATES B injected requests then fires ONE B-way forward
// (requirement 2: static batch, clears the HMX B>=5 gate); prefill stays one-request-at-a-time.
// Weights are loaded per-engine for now (2x shard RAM); requirement 3 collapses that to a single
// shared copy once the shared-dmabuf buffer-type lands (Build 3 / spike S2) — NOT here.
//
// This is a self-contained correctness + concurrency harness (no sockets, host-testable):
//   (A) CORRECTNESS: proves the B-way batched decode is bit-for-bit identical to B single-seq
//       decodes on the SAME engine (no cross-seq KV bleed, correct per-seq block-diagonal mask).
//   (B) CONCURRENCY: times the prefill engine running in parallel with the decode engine and
//       reports the overlap (wall < prefill_alone + decode_alone => the two backends run at once).
// Device strings are CLI args, so host runs it with e.g. --dev-prefill CUDA0 --dev-decode CPU and
// the phone runs the identical code with --dev-prefill GPUOpenCL --dev-decode HTP0.
// ===========================================================================

// Load `model_path` with weights pinned to a single device CSV (e.g. "HTP0" or "GPUOpenCL"),
// honoring -ngl and the LLAMA_LAYER_START/END partial-load env. Returns nullptr on failure.
static llama_model * load_shard_on_device(const std::string & model_path, const std::string & dev_csv, int ngl) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;

    // device list must outlive llama_model_load_from_file; the load consumes it synchronously.
    std::vector<ggml_backend_dev_t> devs;
    if (!dev_csv.empty() && dev_csv != "CPU") {
        std::vector<std::string> names;
        { std::string s = dev_csv; size_t p; while ((p = s.find(',')) != std::string::npos) { names.push_back(s.substr(0,p)); s = s.substr(p+1); } names.push_back(s); }
        for (auto & nm : names) {
            ggml_backend_dev_t d = nullptr;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) { auto dd = ggml_backend_dev_get(i); if (nm == ggml_backend_dev_name(dd)) { d = dd; break; } }
            if (!d) { fprintf(stderr, "error: device '%s' not found. available:\n", nm.c_str());
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) fprintf(stderr, "  %s\n", ggml_backend_dev_name(ggml_backend_dev_get(i)));
                return nullptr; }
            devs.push_back(d);
        }
        devs.push_back(nullptr);
        mp.devices = devs.data();
    }
    return llama_model_load_from_file(model_path.c_str(), mp);
}

// Fill a B-way batch: B injected requests (or token ids when is_head), each its own sequence at `pos`.
// Every row is flagged for output so nextn exposes each stream's cut residual.
static void fill_batch_Bway(llama_batch & b, int n_embd, int B, int32_t pos, bool is_head,
                            const std::vector<llama_token> & toks, const std::vector<float> & residual) {
    b.n_tokens = B;
    for (int j = 0; j < B; ++j) {
        b.token[j] = toks[j];
        if (!is_head) memcpy((float *) b.embd + (size_t) j * n_embd, residual.data() + (size_t) j * n_embd, (size_t) n_embd * sizeof(float));
        b.pos[j]       = pos;
        b.n_seq_id[j]  = 1;
        b.seq_id[j][0] = j;      // distinct KV stream per request
        b.logits[j]    = 1;
    }
}

static int run_dualengine(const std::string & model_path, const std::string & dev_prefill,
                          const std::string & dev_decode, int B, int rounds, int prefill_tokens, int ngl) {
    const char * e_ls = getenv("LLAMA_LAYER_START");
    const char * e_le = getenv("LLAMA_LAYER_END");
    const bool is_head = (e_ls == nullptr || atoi(e_ls) == 0);
    fprintf(stderr, "[dualengine] prefill=%s decode=%s  B=%d rounds=%d prefill_tokens=%d  shard[ls=%s,le=%s] is_head=%d\n",
            dev_prefill.c_str(), dev_decode.c_str(), B, rounds, prefill_tokens,
            e_ls ? e_ls : "0", e_le ? e_le : "n_layer", (int) is_head);

    // --- load the SAME shard on each engine's device (requirement 1: two backends, one process) ---
    llama_model * m_dec = load_shard_on_device(model_path, dev_decode, ngl);
    if (!m_dec) { fprintf(stderr, "error: decode-engine model load failed\n"); return 1; }
    llama_model * m_pre = load_shard_on_device(model_path, dev_prefill, ngl);
    if (!m_pre) { fprintf(stderr, "error: prefill-engine model load failed\n"); llama_model_free(m_dec); return 1; }

    const int n_embd = llama_model_n_embd(m_dec);
    const llama_vocab * vocab = llama_model_get_vocab(m_dec);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token bos = llama_vocab_bos(vocab);

    // decode context: B sequences, each advancing `rounds` positions.
    llama_context_params dp = llama_context_default_params();
    dp.n_seq_max = B;
    dp.n_ctx     = (uint32_t) (B * (rounds + 4) + 64);
    dp.n_batch   = (uint32_t) std::max(B, 8);
    dp.n_ubatch  = (uint32_t) std::max(B, 8);
    dp.no_perf   = true;
    llama_context * ctx_dec = llama_init_from_model(m_dec, dp);

    // prefill context: one request of up to prefill_tokens at a time.
    llama_context_params pp = llama_context_default_params();
    pp.n_seq_max = 1;
    pp.n_ctx     = (uint32_t) std::max(prefill_tokens + 8, 64);
    pp.n_batch   = (uint32_t) std::max(prefill_tokens, 8);
    pp.n_ubatch  = (uint32_t) std::max(prefill_tokens, 8);
    pp.no_perf   = true;
    llama_context * ctx_pre = llama_init_from_model(m_pre, pp);

    if (!ctx_dec || !ctx_pre) {
        fprintf(stderr, "error: context creation failed\n");
        if (ctx_dec) llama_free(ctx_dec);
        if (ctx_pre) llama_free(ctx_pre);
        llama_model_free(m_pre); llama_model_free(m_dec);
        return 1;
    }

    const bool head_less = (e_le != nullptr && atoi(e_le) < (int) llama_model_n_layer(m_dec));
    if (head_less) { llama_set_embeddings_nextn(ctx_dec, true, false); llama_set_embeddings_nextn(ctx_pre, true, false); }

    // synthetic per-request inputs: distinct token id + distinct non-denormal residual per stream.
    std::vector<llama_token> toks(B);
    std::vector<float> resid((size_t) B * n_embd);
    for (int j = 0; j < B; ++j) {
        toks[j] = (llama_token) ((bos + 1 + j * 131) % n_vocab);
        for (int i = 0; i < n_embd; ++i) resid[(size_t) j * n_embd + i] = 0.001f * (((i + j) % 17) - 8);
    }

    auto now_ms = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    int rc = 0;

    // ----------------------------- (A) CORRECTNESS -----------------------------
    // Batched: one B-way forward at pos 0, capture each stream's cut residual (head-less) or argmax.
    llama_batch bb = llama_batch_init(B, is_head ? 0 : n_embd, 1);  // embd allocated iff we inject
    bb.token = (llama_token *) malloc((size_t) B * sizeof(llama_token));
    fill_batch_Bway(bb, n_embd, B, /*pos*/0, is_head, toks, resid);
    if (llama_decode(ctx_dec, bb) != 0) { fprintf(stderr, "error: batched decode failed\n"); rc = 2; goto cleanup; }

    {
        std::vector<std::vector<float>> h_batch(B);
        std::vector<llama_token> arg_batch(B, -1);
        for (int j = 0; j < B; ++j) {
            if (head_less) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_dec, j);
                if (!h) { fprintf(stderr, "error: nextn NULL for stream %d (need LLAMA_LAYER_END<n_layer)\n", j); rc = 2; goto cleanup; }
                h_batch[j].assign(h, h + n_embd);
            } else {
                const float * lg = llama_get_logits_ith(ctx_dec, j);
                if (!lg) { rc = 2; goto cleanup; }
                int best = 0; float bv = lg[0]; for (int i = 1; i < n_vocab; ++i) if (lg[i] > bv) { bv = lg[i]; best = i; }
                arg_batch[j] = best;
            }
        }

        // Reference: clear KV, replay each stream as its OWN single-token decode at pos 0.
        // Batched and single run on the SAME engine, so any diff is batched-vs-serial fp noise
        // (16-wide GEMM tiles differently than 1-wide GEMV) UNLESS there is real cross-seq bleed.
        // Distinguish with an L2-RELATIVE metric: bleed scales with B and dominates the ref norm;
        // fp noise stays ~1e-3. Also report argmax agreement of the cut residual as a robust check.
        llama_memory_clear(llama_get_memory(ctx_dec), true);
        double max_abs = 0.0, sum_d2 = 0.0, sum_r2 = 0.0; int mismatches = 0, argmax_mismatch = 0;
        llama_batch sb = llama_batch_init(1, is_head ? 0 : n_embd, 1);
        sb.token = (llama_token *) malloc(sizeof(llama_token));
        for (int j = 0; j < B; ++j) {
            llama_memory_clear(llama_get_memory(ctx_dec), true);
            sb.n_tokens = 1; sb.token[0] = toks[j];
            if (!is_head) memcpy(sb.embd, resid.data() + (size_t) j * n_embd, (size_t) n_embd * sizeof(float));
            sb.pos[0] = 0; sb.n_seq_id[0] = 1; sb.seq_id[0][0] = 0; sb.logits[0] = 1;
            if (llama_decode(ctx_dec, sb) != 0) { fprintf(stderr, "error: single decode stream %d\n", j); rc = 2; break; }
            if (head_less) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_dec, 0);
                if (!h) { rc = 2; break; }
                int am_ref = 0, am_bat = 0;
                for (int i = 0; i < n_embd; ++i) {
                    const double d = (double) h[i] - (double) h_batch[j][i];
                    max_abs = std::max(max_abs, std::fabs(d));
                    sum_d2 += d * d; sum_r2 += (double) h[i] * (double) h[i];
                    if (h[i] > h[am_ref]) am_ref = i;
                    if (h_batch[j][i] > h_batch[j][am_bat]) am_bat = i;
                }
                if (am_ref != am_bat) argmax_mismatch++;
            } else {
                const float * lg = llama_get_logits_ith(ctx_dec, 0);
                if (!lg) { rc = 2; break; }
                int best = 0; float bv = lg[0]; for (int i = 1; i < n_vocab; ++i) if (lg[i] > bv) { bv = lg[i]; best = i; }
                if (best != arg_batch[j]) mismatches++;
            }
        }
        llama_batch_free(sb);
        if (rc == 0) {
            if (head_less) {
                const double rel_l2 = sum_r2 > 0 ? std::sqrt(sum_d2 / sum_r2) : 0.0;
                const bool pass = rel_l2 < 1e-2 && argmax_mismatch == 0;   // 1% L2 tolerates fp noise; bleed would blow past it
                fprintf(stderr, "[dualengine] CORRECTNESS: batched(B=%d) vs single  rel_L2=%.3e  max|Δ|=%.3e  argmax_mismatch=%d/%d => %s\n",
                        B, rel_l2, max_abs, argmax_mismatch, B, pass ? "PASS" : "FAIL (cross-seq bleed?)");
            } else {
                fprintf(stderr, "[dualengine] CORRECTNESS: batched(B=%d) vs single argmax mismatches = %d/%d => %s\n",
                        B, mismatches, B, (mismatches == 0) ? "PASS" : "FAIL");
            }
        }
    }
    if (rc != 0) goto cleanup;

    // ----------------------------- (B) CONCURRENCY -----------------------------
    llama_memory_clear(llama_get_memory(ctx_dec), true);
    {
        std::atomic<double> t_dec_ms{0}, t_pre_ms{0};

        // DECODE thread: `rounds` static B-way batched forwards (all B streams advance 1 pos/round).
        auto decode_worker = [&]() {
            double t0 = now_ms();
            for (int r = 0; r < rounds; ++r) {
                fill_batch_Bway(bb, n_embd, B, /*pos*/r, is_head, toks, resid);
                if (llama_decode(ctx_dec, bb) != 0) { fprintf(stderr, "error: decode round %d\n", r); break; }
            }
            t_dec_ms = now_ms() - t0;
        };
        // PREFILL thread: `rounds` prompts, one at a time (prefill_tokens each, single sequence).
        auto prefill_worker = [&]() {
            llama_batch pb = llama_batch_init(prefill_tokens, is_head ? 0 : n_embd, 1);
            if (is_head) pb.token = (llama_token *) malloc((size_t) prefill_tokens * sizeof(llama_token));
            double t0 = now_ms();
            for (int r = 0; r < rounds; ++r) {
                llama_memory_clear(llama_get_memory(ctx_pre), true);
                pb.n_tokens = prefill_tokens;
                for (int i = 0; i < prefill_tokens; ++i) {
                    if (is_head) pb.token[i] = (llama_token) ((bos + 1 + (r + i) * 97) % n_vocab);
                    else memcpy((float *) pb.embd + (size_t) i * n_embd, resid.data() + (size_t) (i % B) * n_embd, (size_t) n_embd * sizeof(float));
                    pb.pos[i] = i; pb.n_seq_id[i] = 1; pb.seq_id[i][0] = 0; pb.logits[i] = (i == prefill_tokens - 1);
                }
                if (llama_decode(ctx_pre, pb) != 0) { fprintf(stderr, "error: prefill round %d\n", r); break; }
            }
            t_pre_ms = now_ms() - t0;
            llama_batch_free(pb);
        };

        double w0 = now_ms();
        std::thread td(decode_worker), tp(prefill_worker);
        td.join(); tp.join();
        double wall = now_ms() - w0;
        double sum  = t_dec_ms.load() + t_pre_ms.load();
        fprintf(stderr, "[dualengine] CONCURRENCY over %d rounds:\n"
                        "    decode  engine (%s, B=%d) alone = %.1f ms  (%.2f ms/round)\n"
                        "    prefill engine (%s, T=%d) alone = %.1f ms  (%.2f ms/round)\n"
                        "    wall (overlapped)               = %.1f ms\n"
                        "    serial sum                      = %.1f ms  => overlap saved %.1f ms (%.0f%%), speedup %.2fx\n",
                rounds, dev_decode.c_str(), B, t_dec_ms.load(), t_dec_ms.load() / rounds,
                dev_prefill.c_str(), prefill_tokens, t_pre_ms.load(), t_pre_ms.load() / rounds,
                wall, sum, sum - wall, sum > 0 ? 100.0 * (sum - wall) / sum : 0.0, wall > 0 ? sum / wall : 0.0);
    }

cleanup:
    llama_batch_free(bb);
    llama_free(ctx_pre); llama_free(ctx_dec);
    llama_model_free(m_pre); llama_model_free(m_dec);
    return rc;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string mode;
    std::string act_file;
    std::string act_out;     // mid: output act-file (relayed onward)
    std::string tokens_file; // head/mono: exact token-id list (pipeline generation loop)
    std::string host;        // headnet
    std::string prompt;      // headnet
    std::string sched_file;  // headstream
    int  port    = 0;        // tailnet/headnet/stagenet ; pipedriver stage-A port
    int  port2   = 0;        // pipedriver stage-B port
    int  n_gen   = 16;       // headnet / tailbench (steps)
    int  n_streams = 1;      // tailbench batch size
    double gap_ms  = 0.0;    // tailbench: per-round head-wait (models cross-device relay gap)
    int  prompt_len = 16;    // kvsave/kvload: prefill length (KV positions to snapshot/ship)
    std::string dev_csv;     // --devices HTP0,GPUOpenCL : split layers across these backends
    std::string tsplit_csv;  // --tsplit 24,24          : layers (or ratio) per device
    std::string dev_prefill; // dualengine: prefill-engine device (e.g. GPUOpenCL / CUDA0)
    std::string dev_decode;  // dualengine: decode-engine device  (e.g. HTP0 / CPU)
    int  ngl     = 99;
    int  tok     = -1;     // -1 => use model BOS
    bool tok_set = false;
    bool chat_mode = false; // pipedriver --chat: apply the model's chat template to -p

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--tok") == 0 && i + 1 < argc) {
            tok = atoi(argv[++i]); tok_set = true;
        } else if (strcmp(argv[i], "--act-file") == 0 && i + 1 < argc) {
            act_file = argv[++i];
        } else if (strcmp(argv[i], "--act-out") == 0 && i + 1 < argc) {
            act_out = argv[++i];
        } else if (strcmp(argv[i], "--tokens-file") == 0 && i + 1 < argc) {
            tokens_file = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--port2") == 0 && i + 1 < argc) {
            port2 = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--chat") == 0) {
            chat_mode = true;
        } else if (strcmp(argv[i], "--sched") == 0 && i + 1 < argc) {
            sched_file = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_gen = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            n_streams = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--devices") == 0 && i + 1 < argc) {
            dev_csv = argv[++i];
        } else if (strcmp(argv[i], "--tsplit") == 0 && i + 1 < argc) {
            tsplit_csv = argv[++i];
        } else if (strcmp(argv[i], "--dev-prefill") == 0 && i + 1 < argc) {
            dev_prefill = argv[++i];
        } else if (strcmp(argv[i], "--dev-decode") == 0 && i + 1 < argc) {
            dev_decode = argv[++i];
        } else if (strcmp(argv[i], "--prompt-len") == 0 && i + 1 < argc) {
            prompt_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gap-ms") == 0 && i + 1 < argc) {
            gap_ms = atof(argv[++i]);
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            ngl = atoi(argv[++i]);
        } else {
            fprintf(stderr, "unknown / incomplete arg: %s\n", argv[i]);
            print_usage(argc, argv);
            return 1;
        }
    }

    if (model_path.empty() || mode.empty()) {
        print_usage(argc, argv);
        return 1;
    }
    const bool is_mono     = (mode == "mono");
    const bool is_monogen  = (mode == "monogen");
    const bool is_kvsave   = (mode == "kvsave");
    const bool is_kvload   = (mode == "kvload");
    const bool is_kvserver = (mode == "kvserver");
    const bool is_kvclient = (mode == "kvclient");
    const bool is_head     = (mode == "head");
    const bool is_tail     = (mode == "tail");
    const bool is_mid      = (mode == "mid");
    const bool is_tailnet  = (mode == "tailnet");
    const bool is_headnet  = (mode == "headnet");
    const bool is_stagenet   = (mode == "stagenet");
    const bool is_pipedriver = (mode == "pipedriver");
    const bool is_tailbench = (mode == "tailbench");
    const bool is_tailstream = (mode == "tailstream");
    const bool is_headstream = (mode == "headstream");
    const bool is_dualengine = (mode == "dualengine");
    if (is_kvsave && is_kvload) { fprintf(stderr, "error: pick one of kvsave/kvload\n"); return 1; }
    if ((is_kvsave || is_kvload) && act_file.empty()) { fprintf(stderr, "error: --act-file (blob path) required for %s\n", mode.c_str()); return 1; }
    if (is_kvserver && port <= 0) { fprintf(stderr, "error: --port required for kvserver\n"); return 1; }
    if (is_kvclient && (host.empty() || port <= 0)) { fprintf(stderr, "error: --host --port required for kvclient\n"); return 1; }
    if (!is_mono && !is_monogen && !is_kvsave && !is_kvload && !is_kvserver && !is_kvclient && !is_head && !is_tail && !is_mid && !is_tailnet && !is_headnet && !is_tailbench
        && !is_tailstream && !is_headstream && !is_stagenet && !is_pipedriver && !is_dualengine) {
        fprintf(stderr, "error: --mode must be mono|head|tail|mid|stagenet|pipedriver|tailnet|headnet|tailbench|tailstream|headstream|dualengine (got '%s')\n", mode.c_str());
        return 1;
    }
    if (is_mid && (act_file.empty() || act_out.empty())) {
        fprintf(stderr, "error: --act-file (in) and --act-out (out) are required for mode=mid\n");
        return 1;
    }
    if (is_stagenet && port <= 0) { fprintf(stderr, "error: --port required for stagenet\n"); return 1; }
    if (is_pipedriver && (host.empty() || port <= 0 || port2 <= 0 || prompt.empty())) {
        fprintf(stderr, "error: --host --port (stage A) --port2 (stage B) -p PROMPT required for pipedriver\n"); return 1;
    }
    if (is_tailstream && port <= 0) { fprintf(stderr, "error: --port required for tailstream\n"); return 1; }
    if (is_headstream && (host.empty() || port <= 0 || sched_file.empty())) {
        fprintf(stderr, "error: --host --port --sched required for headstream\n"); return 1;
    }
    if ((is_head || is_tail) && act_file.empty()) {
        fprintf(stderr, "error: --act-file is required for mode=%s\n", mode.c_str());
        return 1;
    }
    if ((is_tailnet || is_headnet) && port <= 0) {
        fprintf(stderr, "error: --port is required for mode=%s\n", mode.c_str());
        return 1;
    }
    if (is_headnet && (host.empty() || prompt.empty())) {
        fprintf(stderr, "error: --host and -p PROMPT are required for mode=headnet\n");
        return 1;
    }

    ggml_backend_load_all();

    // [plan-a M4] dualengine owns its own two models + two contexts (one per engine/device);
    // dispatch it here, before the single-model/single-context path below.
    if (is_dualengine) {
        const std::string dp = dev_prefill.empty() ? "GPUOpenCL" : dev_prefill;
        const std::string dd = dev_decode.empty()  ? "HTP0"      : dev_decode;
        const int B       = std::max(n_streams, 1);              // -b : static decode batch
        const int rounds  = std::max(n_gen, 1);                  // -n : decode rounds
        const int pf_toks = std::max(prompt_len, 1);             // --prompt-len : tokens/prefill request
        return run_dualengine(model_path, dp, dd, B, rounds, pf_toks, ngl);
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    // optional device split (e.g. op15 HTP+Adreno): --devices HTP0,GPUOpenCL --tsplit 24,24
    static std::vector<ggml_backend_dev_t> devs;
    static std::vector<float> tsplit;
    if (!dev_csv.empty()) {
        std::vector<std::string> names;
        { std::string s=dev_csv; size_t p; while((p=s.find(','))!=std::string::npos){names.push_back(s.substr(0,p));s=s.substr(p+1);} names.push_back(s); }
        for (auto & nm : names) {
            ggml_backend_dev_t d = nullptr;
            for (size_t i=0;i<ggml_backend_dev_count();++i){ auto dd=ggml_backend_dev_get(i); if (nm==ggml_backend_dev_name(dd)){d=dd;break;} }
            if (!d) { fprintf(stderr,"error: device '%s' not found. available:\n",nm.c_str());
                for (size_t i=0;i<ggml_backend_dev_count();++i) fprintf(stderr,"  %s\n",ggml_backend_dev_name(ggml_backend_dev_get(i)));
                return 1; }
            devs.push_back(d);
        }
        devs.push_back(nullptr);                 // NULL-terminated
        model_params.devices = devs.data();
        tsplit.assign(names.size(), 1.0f);       // default equal
        if (!tsplit_csv.empty()) { std::string s=tsplit_csv; size_t idx=0,p;
            while (idx<tsplit.size()) { p=s.find(','); std::string t=(p==std::string::npos)?s:s.substr(0,p);
                tsplit[idx++]=(float)atof(t.c_str()); if(p==std::string::npos)break; s=s.substr(p+1); } }
        model_params.tensor_split = tsplit.data();
        fprintf(stderr,"[devsplit]");
        for (size_t i=0;i<names.size();++i) fprintf(stderr," %s(%.0f)",names[i].c_str(),tsplit[i]);
        fprintf(stderr,"\n");
    }

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "error: unable to load model '%s'\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);

    if (!tok_set) {
        tok = (int) llama_vocab_bos(vocab);
    }

    // [plan-a port] multi-token prefill source for mono/head: --tokens-file (exact ids, for the
    // pipeline generation loop) > -p PROMPT (tokenized, +BOS) > single --tok / BOS. tail/mid derive
    // N from the act-file they receive (peek below).
    std::vector<llama_token> toks;
    if ((is_mono || is_head) && !tokens_file.empty()) {
        std::ifstream tf(tokens_file);
        if (!tf) { fprintf(stderr, "error: cannot open --tokens-file '%s'\n", tokens_file.c_str()); llama_model_free(model); return 1; }
        for (long v; tf >> v; ) toks.push_back((llama_token) v);
        if (toks.empty()) { fprintf(stderr, "error: --tokens-file '%s' had 0 ids\n", tokens_file.c_str()); llama_model_free(model); return 1; }
    } else if ((is_mono || is_head) && !prompt.empty()) {
        toks = common_tokenize(vocab, prompt, /*add_special*/ true, /*parse_special*/ true);
        if (toks.empty()) { fprintf(stderr, "error: prompt tokenized to 0 tokens\n"); llama_model_free(model); return 1; }
    } else {
        toks = { (llama_token) tok };
    }
    int n_prefill = (int) toks.size();
    if (is_tail || is_mid) {
        n_prefill = peek_actfile_ntokens(act_file);
        if (n_prefill <= 0) { fprintf(stderr, "error: cannot read n_tokens from act-file '%s'\n", act_file.c_str()); llama_model_free(model); return 1; }
    }

    llama_context_params ctx_params = llama_context_default_params();
    // net modes advance KV one position per token (prompt + n_gen); size generously.
    ctx_params.n_ctx   = (is_tailnet || is_headnet || is_tailstream || is_headstream || is_stagenet || is_pipedriver) ? 4096 :
                         (is_kvsave || is_kvload || is_kvserver || is_kvclient) ? (uint32_t)(prompt_len + n_gen + 64) : 64;
    ctx_params.n_batch = 8;
    if (is_kvsave || is_kvload || is_kvserver || is_kvclient) {   // prefill the whole prompt in one batch
        ctx_params.n_batch  = (uint32_t) std::max(prompt_len, 8);
        ctx_params.n_ubatch = (uint32_t) std::max(prompt_len, 8);
    }
    ctx_params.no_perf = true;
    if (is_mono || is_head || is_tail || is_mid) {   // [plan-a port] size for an N-token prefill
        const uint32_t N = (uint32_t) std::max(n_prefill, 1);
        ctx_params.n_ctx    = std::max<uint32_t>(N + 8, 64);
        ctx_params.n_batch  = std::max<uint32_t>(N, 8);
        ctx_params.n_ubatch = std::max<uint32_t>(N, 8);
    }
    if (is_tailbench) {
        // B sequences, each advancing n_gen positions -> need B*n_gen KV cells.
        ctx_params.n_seq_max = n_streams;
        ctx_params.n_ctx     = (uint32_t) (n_streams * (n_gen + 4) + 64);
        ctx_params.n_batch   = (uint32_t) std::max(n_streams, 8);
        ctx_params.n_ubatch  = (uint32_t) std::max(n_streams, 8);
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    int rc;
    if (is_tailstream) {
        rc = run_tailstream(ctx, vocab, n_embd, n_vocab, port);
    } else if (is_kvsave || is_kvload) {
        rc = run_kv(ctx, vocab, n_vocab, is_kvsave, prompt_len, n_gen, act_file);
    } else if (is_kvserver) {
        rc = run_kvserver(ctx, vocab, n_vocab, port, prompt_len, std::max(n_streams, 1));
    } else if (is_kvclient) {
        rc = run_kvclient(ctx, n_vocab, host, port, prompt_len, n_gen);
    } else if (is_monogen) {
        rc = run_monogen(ctx, vocab, n_vocab, (llama_token) tok, n_gen);
    } else if (is_headstream) {
        rc = run_headstream(ctx, n_embd, host, port, sched_file);
    } else if (is_tailbench) {
        rc = run_tailbench(ctx, n_embd, n_streams, n_gen, gap_ms);
    } else if (is_stagenet) {
        rc = run_stagenet(ctx, n_embd, port);
    } else if (is_pipedriver) {
        rc = run_pipedriver(ctx, vocab, n_embd, n_vocab, host, port, port2, prompt, n_gen, chat_mode);
    } else if (is_tailnet) {
        rc = run_tailnet(ctx, vocab, n_embd, n_vocab, port);
    } else if (is_headnet) {
        rc = run_headnet(ctx, vocab, n_embd, host, port, prompt, n_gen);
    } else if (is_mid) {
        rc = run_mid(ctx, n_embd, act_file, act_out);
    } else if (is_tail) {
        rc = run_tail(ctx, vocab, n_embd, n_vocab, act_file);
    } else {
        rc = run_mono_or_head(ctx, vocab, n_embd, n_vocab, toks, is_head, act_file);
    }

    llama_free(ctx);
    llama_model_free(model);
    return rc;
}
