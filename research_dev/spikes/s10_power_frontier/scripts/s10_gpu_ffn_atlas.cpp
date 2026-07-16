// S10-V0 CP2 measurement harness (opt-in, standalone, no protected-path edits).
// Runs the IDENTICAL certified Gemma4 dense-FFN island (examples/phone-pim/
// phone_pim_ffn.cpp) on a CUDA A6000 across a batch (token_count = M) sweep and
// reports measured on-device compute latency. A separate --sustain mode loops
// EXECUTE for a fixed wall time so an external nvidia-smi sampler can attribute
// board power to the load. Measurement only; it changes no scheduler behavior.
#include "phone_pim_ffn.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

using phone_pim::FfnIsland;
using phone_pim::FfnSpec;
using phone_pim::FfnPrepareMetrics;
using phone_pim::FfnExecuteMetrics;

static uint64_t now_us() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool parse_hex32(const std::string & hex, std::array<uint8_t,32> & out) {
    if (hex.size() != 64) return false;
    for (int i = 0; i < 32; ++i) {
        auto nib = [](char c)->int{
            if (c>='0'&&c<='9') return c-'0';
            if (c>='a'&&c<='f') return c-'a'+10;
            if (c>='A'&&c<='F') return c-'A'+10;
            return -1; };
        int hi = nib(hex[2*i]), lo = nib(hex[2*i+1]);
        if (hi<0||lo<0) return false;
        out[i] = (uint8_t)((hi<<4)|lo);
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model, prefix = "blk.2", backend = "CUDA0", sha_hex;
    uint64_t model_bytes = 0;
    std::vector<int> sweep;
    int reps = 300;
    int sustain_m = 0, sustain_seconds = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]()->std::string{ return (i+1<argc)?argv[++i]:std::string(); };
        if (a=="--model") model = next();
        else if (a=="--prefix") prefix = next();
        else if (a=="--backend") backend = next();
        else if (a=="--sha256") sha_hex = next();
        else if (a=="--model-bytes") model_bytes = std::stoull(next());
        else if (a=="--reps") reps = std::stoi(next());
        else if (a=="--sustain-m") sustain_m = std::stoi(next());
        else if (a=="--sustain-seconds") sustain_seconds = std::stoi(next());
        else if (a=="--sweep") { std::string s=next(); size_t p=0; while(p<s.size()){ size_t c=s.find(',',p); sweep.push_back(std::stoi(s.substr(p,c-p))); if(c==std::string::npos)break; p=c+1; } }
    }
    if (model.empty() || model_bytes==0 || sha_hex.empty()) {
        std::fprintf(stderr, "usage: --model M --model-bytes N --sha256 HEX [--prefix blk.2] [--backend CUDA0] [--sweep 1,2,4,..] [--reps 300] [--sustain-m M --sustain-seconds S]\n");
        return 2;
    }
    std::array<uint8_t,32> sha{};
    if (!parse_hex32(sha_hex, sha)) { std::fprintf(stderr, "bad sha256 hex\n"); return 2; }
    if (sweep.empty()) sweep = {1,2,4,8,16,32,64,128,256,512};

    auto run_one = [&](int M, bool sustain)->int {
        FfnSpec spec;
        spec.model_path = model;
        spec.tensor_prefix = prefix;
        spec.backend_name = backend;
        spec.token_count = (uint32_t)M;
        spec.expected_model_bytes = model_bytes;
        spec.expected_model_sha256 = sha;
        FfnIsland island;
        FfnPrepareMetrics pm; std::string err;
        if (!island.prepare(spec, pm, err)) {
            std::fprintf(stderr, "prepare M=%d failed: %s\n", M, err.c_str());
            return 1;
        }
        std::vector<float> in = island.make_test_input(0x70696d31U + (uint32_t)M);
        std::vector<float> out;
        FfnExecuteMetrics em;
        // warmup
        for (int w=0; w<10; ++w) { if(!island.execute(in,out,em,err)){ std::fprintf(stderr,"exec warmup fail: %s\n",err.c_str()); return 1; } }
        if (sustain) {
            uint64_t t_end = now_us() + (uint64_t)sustain_seconds*1000000ULL;
            uint64_t iters=0; uint64_t comp_sum=0;
            while (now_us() < t_end) { if(!island.execute(in,out,em,err)){ std::fprintf(stderr,"exec fail: %s\n",err.c_str()); return 1;} comp_sum+=em.compute_us; ++iters; }
            std::printf("{\"mode\":\"sustain\",\"backend\":\"%s\",\"M\":%d,\"iters\":%llu,\"compute_us_mean\":%.3f,\"weight_bytes\":%llu,\"resident_bytes\":%llu}\n",
                backend.c_str(), M, (unsigned long long)iters, iters? (double)comp_sum/iters:0.0,
                (unsigned long long)pm.weight_bytes, (unsigned long long)pm.resident_buffer_bytes);
            return 0;
        }
        std::vector<uint64_t> comp, inset, outget, e2e;
        comp.reserve(reps); inset.reserve(reps); outget.reserve(reps); e2e.reserve(reps);
        for (int r=0; r<reps; ++r) {
            uint64_t s0=now_us();
            if(!island.execute(in,out,em,err)){ std::fprintf(stderr,"exec fail M=%d: %s\n",M,err.c_str()); return 1; }
            e2e.push_back(now_us()-s0);
            comp.push_back(em.compute_us); inset.push_back(em.input_set_us); outget.push_back(em.output_get_us);
        }
        auto pct = [](std::vector<uint64_t> v, double p)->uint64_t{ std::sort(v.begin(),v.end()); size_t idx=(size_t)std::floor(p*(v.size()-1)); return v[idx]; };
        uint64_t in_bytes = (uint64_t)M * island.n_embd() * sizeof(float);
        uint64_t out_bytes = in_bytes;
        std::printf("{\"mode\":\"latency\",\"backend\":\"%s\",\"prefix\":\"%s\",\"M\":%d,\"n_embd\":%llu,\"n_ff\":%llu,\"reps\":%d,"
            "\"compute_us_p50\":%llu,\"compute_us_p05\":%llu,\"compute_us_p95\":%llu,"
            "\"inset_us_p50\":%llu,\"outget_us_p50\":%llu,\"e2e_us_p50\":%llu,"
            "\"weight_bytes\":%llu,\"resident_bytes\":%llu,\"input_bytes\":%llu,\"output_bytes\":%llu}\n",
            backend.c_str(), prefix.c_str(), M,
            (unsigned long long)island.n_embd(), (unsigned long long)island.n_ff(), reps,
            (unsigned long long)pct(comp,0.5),(unsigned long long)pct(comp,0.05),(unsigned long long)pct(comp,0.95),
            (unsigned long long)pct(inset,0.5),(unsigned long long)pct(outget,0.5),(unsigned long long)pct(e2e,0.5),
            (unsigned long long)pm.weight_bytes,(unsigned long long)pm.resident_buffer_bytes,
            (unsigned long long)in_bytes,(unsigned long long)out_bytes);
        std::fflush(stdout);
        return 0;
    };

    if (sustain_m>0 && sustain_seconds>0) return run_one(sustain_m, true);
    int rc=0;
    for (int M : sweep) { int r=run_one(M,false); if(r){rc=r;} }
    return rc;
}
