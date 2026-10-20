// Minimal TCP driver for llama-ffn-split-worker: streams deterministic inputs
// across layers and records per-call worker compute_us plus the outputs.
#include "ffn-split-protocol.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool read_exact(int fd, void * data, size_t size) {
    uint8_t * p = static_cast<uint8_t *>(data);
    while (size > 0) {
        ssize_t n = read(fd, p, size);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}

static bool write_exact(int fd, const void * data, size_t size) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    while (size > 0) {
        ssize_t n = write(fd, p, size);
        if (n <= 0) {
            return false;
        }
        p += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x800000;
        const int shift = 14 - exp;
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1);
        const uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) {
            ++half;
        }
        return static_cast<uint16_t>(sign | half);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) {
        ++half;
    }
    return static_cast<uint16_t>(half);
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ff;
            x = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000u | (mant << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

int main(int argc, char ** argv) {
    std::string host = "127.0.0.1";
    int port = 0;
    std::string sha;
    std::vector<int> layers;
    uint32_t n_embd = 0;
    uint32_t columns = 0;
    uint32_t request_columns = 0;
    uint32_t tokens = 1;
    uint32_t max_tokens = 1;
    int iters = 16;
    int warmup = 2;
    bool swiglu = true;
    std::string out_path;
    std::string csv_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--host") host = next();
        else if (a == "--port") port = atoi(next().c_str());
        else if (a == "--sha") sha = next();
        else if (a == "--layers") {
            std::string s = next();
            size_t pos = 0;
            while (pos < s.size()) {
                size_t c = s.find(',', pos);
                layers.push_back(atoi(s.substr(pos, c - pos).c_str()));
                if (c == std::string::npos) break;
                pos = c + 1;
            }
        }
        else if (a == "--n-embd") n_embd = static_cast<uint32_t>(atoi(next().c_str()));
        else if (a == "--columns") columns = static_cast<uint32_t>(atoi(next().c_str()));
        else if (a == "--request-columns") request_columns = static_cast<uint32_t>(atoi(next().c_str()));
        else if (a == "--tokens") tokens = static_cast<uint32_t>(atoi(next().c_str()));
        else if (a == "--max-tokens") max_tokens = static_cast<uint32_t>(atoi(next().c_str()));
        else if (a == "--iters") iters = atoi(next().c_str());
        else if (a == "--warmup") warmup = atoi(next().c_str());
        else if (a == "--geglu") swiglu = false;
        else if (a == "--out") out_path = next();
        else if (a == "--csv") csv_path = next();
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (request_columns == 0) {
        request_columns = columns;
    }
    uint8_t digest[32] = {};
    if (port <= 0 || layers.empty() || n_embd == 0 || columns == 0 ||
        !ffn_split::parse_artifact_sha256(sha, digest)) {
        fprintf(stderr, "usage: --port P --sha sha256:.. --layers a,b --n-embd K --columns C "
                        "[--request-columns C] [--tokens T --max-tokens M] [--iters N] [--out F] [--csv F]\n");
        return 2;
    }
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
    if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        perror("connect");
        return 1;
    }
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    ffn_split::hello_request hello = {};
    hello.magic = ffn_split::protocol_magic;
    hello.version = ffn_split::protocol_version;
    hello.message = static_cast<uint16_t>(ffn_split::message_type::hello_request);
    for (int l : layers) {
        hello.layer_mask |= UINT64_C(1) << l;
    }
    hello.n_embd = n_embd;
    hello.max_columns = columns;
    hello.flags = ffn_split::flag_f16_io | (swiglu ? ffn_split::flag_swiglu : 0);
    hello.max_tokens = static_cast<uint16_t>(max_tokens);
    memcpy(hello.artifact_sha256, digest, 32);
    ffn_split::hello_response hr = {};
    if (!write_exact(fd, &hello, sizeof(hello)) || !read_exact(fd, &hr, sizeof(hr)) || hr.status != 0) {
        fprintf(stderr, "hello failed status=%u\n", hr.status);
        return 1;
    }
    fprintf(stderr, "hello ok n_ff=%u offset=%u weight_hash=%016llx\n", hr.n_ff, hr.offset,
            static_cast<unsigned long long>(hr.weight_hash));

    const size_t elements = static_cast<size_t>(n_embd) * tokens;
    std::vector<uint16_t> input(elements);
    std::vector<uint16_t> output(elements);
    std::vector<float> all_outputs;
    FILE * csv = csv_path.empty() ? nullptr : fopen(csv_path.c_str(), "w");
    if (csv) {
        fprintf(csv, "iter,layer,tokens,columns,compute_us,rtt_us\n");
    }
    std::vector<double> compute_samples;
    std::vector<double> rtt_samples;
    uint32_t request_id = 1;
    for (int iter = -warmup; iter < iters; ++iter) {
        for (int layer : layers) {
            // inputs depend on (iter, layer) only, so runs are comparable
            uint64_t state = 0x9e3779b97f4a7c15ULL ^ (static_cast<uint64_t>(iter + 1000) << 20) ^
                    static_cast<uint64_t>(layer);
            for (size_t i = 0; i < elements; ++i) {
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                double u1 = ((state >> 11) + 1.0) / 9007199254740993.0;
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                double u2 = (state >> 11) / 9007199254740992.0;
                float g = static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
                input[i] = f32_to_f16(g);
            }
            ffn_split::execute_request req = {};
            req.magic = ffn_split::protocol_magic;
            req.version = ffn_split::protocol_version;
            req.message = static_cast<uint16_t>(ffn_split::message_type::execute_request);
            req.request_id = request_id++;
            req.layer = layer;
            req.elements = static_cast<uint32_t>(elements);
            req.payload_bytes = static_cast<uint32_t>(elements * 2);
            req.payload_hash = ffn_split::hash_bytes(input.data(), elements * 2);
            req.columns = request_columns;
            req.tokens = tokens;
            const auto t0 = std::chrono::steady_clock::now();
            ffn_split::execute_response resp = {};
            if (!write_exact(fd, &req, sizeof(req)) || !write_exact(fd, input.data(), elements * 2) ||
                !read_exact(fd, &resp, sizeof(resp)) || resp.status != 0 ||
                resp.payload_bytes != elements * 2 || !read_exact(fd, output.data(), elements * 2)) {
                fprintf(stderr, "execute failed iter=%d layer=%d\n", iter, layer);
                return 1;
            }
            const double rtt = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - t0).count();
            if (resp.payload_hash != ffn_split::hash_bytes(output.data(), elements * 2)) {
                fprintf(stderr, "response hash mismatch\n");
                return 1;
            }
            if (iter < 0) {
                continue;
            }
            compute_samples.push_back(static_cast<double>(resp.compute_us));
            rtt_samples.push_back(rtt);
            if (csv) {
                fprintf(csv, "%d,%d,%u,%u,%llu,%.0f\n", iter, layer, tokens, request_columns,
                        static_cast<unsigned long long>(resp.compute_us), rtt);
            }
            for (size_t i = 0; i < elements; ++i) {
                all_outputs.push_back(f16_to_f32(output[i]));
            }
        }
    }
    if (csv) {
        fclose(csv);
    }
    if (!out_path.empty()) {
        FILE * f = fopen(out_path.c_str(), "wb");
        fwrite(all_outputs.data(), sizeof(float), all_outputs.size(), f);
        fclose(f);
    }
    auto pct = [](std::vector<double> v, double q) {
        std::sort(v.begin(), v.end());
        return v[static_cast<size_t>(q * (v.size() - 1))];
    };
    double mean = 0;
    for (double v : compute_samples) mean += v;
    mean /= compute_samples.size();
    printf("DRIVER calls=%zu tokens=%u columns=%u compute_p50_us=%.0f compute_mean_us=%.0f "
           "compute_p10_us=%.0f compute_p90_us=%.0f rtt_p50_us=%.0f\n",
           compute_samples.size(), tokens, request_columns, pct(compute_samples, 0.5), mean,
           pct(compute_samples, 0.1), pct(compute_samples, 0.9), pct(rtt_samples, 0.5));
    close(fd);
    return 0;
}
