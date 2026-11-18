// Weight-stream energy benchmark: device-local client of the FFN split worker (loopback, protocol v6).
// One "decode call" = one EXECUTE per owned layer, batch 1, back to back; calls repeat with no gap.
//   ffn-stream-client --group PORT:0,1,.. [--group PORT2:4,5,..] --artifact sha256:HEX --n-embd D --columns NFF
//                     [--geglu] --idle-s I --warmup-s W --seconds S --out FILE.csv [--seed N]
// A group is one worker (one NPU session); a call visits the groups in order.
// Timeline (CLOCK_BOOTTIME, the clock of /proc/uptime): HELLO, idle window (weights resident, no calls),
// warm-up calls, measured calls. CSV: phase,call,t_start_us,t_end_us,compute_us,<per-layer compute_us...>
#include "ffn-split-protocol.h"

#include <arpa/inet.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <random>
#include <string>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {

uint64_t now_us() {
    timespec ts;
    clock_gettime(CLOCK_BOOTTIME, &ts);
    return uint64_t(ts.tv_sec) * 1000000ULL + uint64_t(ts.tv_nsec) / 1000ULL;
}

void sleep_us(uint64_t us) {
    timespec ts = { time_t(us / 1000000ULL), long(us % 1000000ULL) * 1000L };
    nanosleep(&ts, nullptr);
}

bool write_exact(int fd, const void * data, size_t size) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    while (size > 0) {
        const ssize_t n = send(fd, p, size, MSG_NOSIGNAL);
        if (n <= 0) return false;
        p += n; size -= size_t(n);
    }
    return true;
}

bool read_exact(int fd, void * data, size_t size) {
    uint8_t * p = static_cast<uint8_t *>(data);
    while (size > 0) {
        const ssize_t n = recv(fd, p, size, 0);
        if (n <= 0) return false;
        p += n; size -= size_t(n);
    }
    return true;
}

uint16_t f32_to_f16(float value) {
    uint32_t x;
    memcpy(&x, &value, 4);
    const uint32_t sign = (x >> 16) & 0x8000U;
    const int32_t exponent = int32_t((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = x & 0x7FFFFFU;
    if (exponent <= 0) return uint16_t(sign);
    if (exponent >= 31) return uint16_t(sign | 0x7C00U);
    mantissa = (mantissa + 0x1000U) >> 13;
    return uint16_t((sign | (uint32_t(exponent) << 10)) + mantissa);
}

std::vector<int> parse_list(const std::string & s) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t next = s.find(',', pos);
        if (next == std::string::npos) next = s.size();
        out.push_back(atoi(s.substr(pos, next - pos).c_str()));
        pos = next + 1;
    }
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    int n_embd = 0, columns = 0;
    std::vector<int> ports;          // one per group
    std::vector<int> layer_group;    // group index of each layer
    bool geglu = false;
    double idle_s = 10, warmup_s = 2, seconds = 20;
    uint32_t seed = 1234;
    std::string artifact, out;
    std::vector<int> layers;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--group") {
            const std::string g = next();
            const size_t colon = g.find(':');
            if (colon == std::string::npos) { fprintf(stderr, "bad group\n"); return 2; }
            ports.push_back(atoi(g.substr(0, colon).c_str()));
            for (int layer : parse_list(g.substr(colon + 1))) {
                layers.push_back(layer);
                layer_group.push_back(int(ports.size()) - 1);
            }
        }
        else if (a == "--artifact") artifact = next();
        else if (a == "--n-embd") n_embd = atoi(next().c_str());
        else if (a == "--columns") columns = atoi(next().c_str());
        else if (a == "--geglu") geglu = true;
        else if (a == "--idle-s") idle_s = atof(next().c_str());
        else if (a == "--warmup-s") warmup_s = atof(next().c_str());
        else if (a == "--seconds") seconds = atof(next().c_str());
        else if (a == "--seed") seed = uint32_t(strtoul(next().c_str(), nullptr, 10));
        else if (a == "--out") out = next();
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    uint8_t sha[32];
    if (ports.empty() || layers.empty() || n_embd <= 0 || columns <= 0 || out.empty() ||
            !ffn_split::parse_artifact_sha256(artifact, sha)) {
        fprintf(stderr, "missing arguments\n");
        return 2;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<std::vector<uint16_t>> inputs(layers.size(), std::vector<uint16_t>(size_t(n_embd)));
    for (auto & row : inputs) for (uint16_t & v : row) v = f32_to_f16(normal(rng));

    std::vector<int> fds;
    std::vector<ffn_split::hello_response> hellos;
    for (size_t g = 0; g < ports.size(); ++g) {
        const int fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in address = {};
        address.sin_family = AF_INET;
        address.sin_port = htons(uint16_t(ports[g]));
        inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
        bool connected = false;
        for (int attempt = 0; attempt < 600 && !connected; ++attempt) {
            connected = connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0;
            if (!connected) sleep_us(100000);
        }
        if (!connected) { fprintf(stderr, "connect failed\n"); return 1; }
        const int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        ffn_split::hello_request hello = {};
        hello.magic = ffn_split::protocol_magic;
        hello.version = ffn_split::protocol_version;
        hello.message = uint16_t(ffn_split::message_type::hello_request);
        for (size_t l = 0; l < layers.size(); ++l) {
            if (layer_group[l] == int(g)) hello.layer_mask |= UINT64_C(1) << layers[l];
        }
        hello.n_embd = uint32_t(n_embd);
        hello.max_columns = uint32_t(columns);
        hello.flags = uint16_t(ffn_split::flag_f16_io | (geglu ? 0 : ffn_split::flag_swiglu));
        hello.max_tokens = 1;
        memcpy(hello.artifact_sha256, sha, 32);
        ffn_split::hello_response hr = {};
        if (!write_exact(fd, &hello, sizeof(hello)) || !read_exact(fd, &hr, sizeof(hr)) || hr.status != 0) {
            fprintf(stderr, "hello failed group %zu status=%u\n", g, unsigned(hr.status));
            return 1;
        }
        fds.push_back(fd);
        hellos.push_back(hr);
    }
    FILE * csv = fopen(out.c_str(), "w");
    if (!csv) return 1;
    for (const ffn_split::hello_response & hr : hellos) {
        fprintf(csv, "# hello n_embd=%u n_ff=%u max_columns=%u weight_type=%u layers=%u weight_hash=%016llx t_us=%llu\n",
                hr.n_embd, hr.n_ff, hr.max_columns, hr.weight_type, hr.layer_count,
                static_cast<unsigned long long>(hr.weight_hash), static_cast<unsigned long long>(now_us()));
    }
    const uint64_t idle0 = now_us();
    sleep_us(uint64_t(idle_s * 1e6));
    const uint64_t idle1 = now_us();
    fprintf(csv, "# idle t0_us=%llu t1_us=%llu\n", static_cast<unsigned long long>(idle0),
            static_cast<unsigned long long>(idle1));
    fprintf(csv, "phase,call,t_start_us,t_end_us,compute_us");
    for (int layer : layers) fprintf(csv, ",l%d_us", layer);
    fprintf(csv, "\n");

    const uint64_t run0 = now_us();
    const uint64_t measure0 = run0 + uint64_t(warmup_s * 1e6);
    const uint64_t run1 = measure0 + uint64_t(seconds * 1e6);
    uint32_t request_id = 1;
    std::vector<uint8_t> response_payload;
    std::vector<uint64_t> per_layer(layers.size());
    long call = 0;
    uint64_t first_measured = 0, last_end = 0;
    while (true) {
        const uint64_t started = now_us();
        if (started >= run1) break;
        uint64_t compute = 0;
        for (size_t l = 0; l < layers.size(); ++l) {
            const int fd = fds[size_t(layer_group[l])];
            ffn_split::execute_request request = {};
            request.magic = ffn_split::protocol_magic;
            request.version = ffn_split::protocol_version;
            request.message = uint16_t(ffn_split::message_type::execute_request);
            request.request_id = request_id++;
            request.layer = layers[l];
            request.elements = uint32_t(n_embd);
            request.payload_bytes = uint32_t(n_embd) * 2;
            request.payload_hash = ffn_split::hash_bytes(inputs[l].data(), request.payload_bytes);
            request.columns = uint32_t(columns);
            request.tokens = 1;
            ffn_split::execute_response response = {};
            if (!write_exact(fd, &request, sizeof(request)) ||
                    !write_exact(fd, inputs[l].data(), request.payload_bytes) ||
                    !read_exact(fd, &response, sizeof(response))) {
                fprintf(stderr, "rpc failed at request %u\n", request.request_id);
                return 1;
            }
            response_payload.resize(response.payload_bytes);
            if (!read_exact(fd, response_payload.data(), response_payload.size())) return 1;
            if (response.status != 0 || response.request_id != request.request_id ||
                    response.payload_bytes != request.payload_bytes) {
                fprintf(stderr, "bad response at request %u status=%u\n", request.request_id, unsigned(response.status));
                return 1;
            }
            per_layer[l] = response.compute_us;
            compute += response.compute_us;
        }
        const uint64_t finished = now_us();
        const bool measured = started >= measure0;
        if (measured && first_measured == 0) first_measured = started;
        if (measured) last_end = finished;
        fprintf(csv, "%s,%ld,%llu,%llu,%llu", measured ? "measure" : "warmup", call++,
                static_cast<unsigned long long>(started), static_cast<unsigned long long>(finished),
                static_cast<unsigned long long>(compute));
        for (uint64_t v : per_layer) fprintf(csv, ",%llu", static_cast<unsigned long long>(v));
        fprintf(csv, "\n");
    }
    fprintf(csv, "# measure t0_us=%llu t1_us=%llu calls=%ld\n", static_cast<unsigned long long>(first_measured),
            static_cast<unsigned long long>(last_end), call);
    fclose(csv);
    for (int fd : fds) close(fd);
    printf("done calls=%ld requests=%u\n", call, request_id - 1);
    return 0;
}
