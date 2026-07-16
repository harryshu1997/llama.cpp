// Measurement-only transport probe for S9-V1A (Checkpoint 1).
//
// This binary is NOT part of the phone-PIM production protocol. It never writes,
// fsyncs, hashes-to-disk, or publishes anything: it measures the raw host<->phone
// data path (the adb-forwarded USB TCP socket) so we can attribute the sequential
// dynamic-provisioning wall time. It deliberately uses its own minimal fixed frame
// header (not protocol v3) so it can never affect the frozen v3 wire evidence.
//
// Roles:
//   --role sink    (phone) receive frames into memory, optional payload SHA, ack; report device-side timers.
//   --role source  (phone) stream frames from memory to the host (D2H).
//   --role host    (PC)    drive an experiment and emit a result JSON line.
//
// Experiments (host):
//   h2d  : send `count` payload frames, per-frame ack, bounded outstanding `window`
//          (window=1 == production stop-and-wait). --final-ack streams then waits one ack.
//   d2h  : ask the source to stream `count` frames back; measure inbound bandwidth.
//
// Nothing here touches the production worker/host/store or the v3 protocol.

#include "phone_pim_protocol.h"
#include "phone_pim_socket.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace pp = phone_pim;

namespace {

uint64_t now_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

constexpr uint32_t k_bench_magic = 0x4e454250U; // "PBEN"

enum BenchType : uint32_t {
    type_data      = 0,
    type_ack       = 1,
    type_start_d2h = 2,
    type_shutdown  = 3,
    type_summary   = 4, // final frame: 16-byte payload [frames:u64][bytes:u64] for agreement
};

constexpr uint32_t k_summary_bytes = 16;

// 16-byte fixed header, little-endian, sent raw.
struct BenchHdr {
    uint32_t magic;
    uint32_t type;
    uint32_t seq;
    uint32_t payload_bytes;
};

void put_hdr(uint8_t * b, const BenchHdr & h) {
    std::memcpy(b + 0, &h.magic, 4);
    std::memcpy(b + 4, &h.type, 4);
    std::memcpy(b + 8, &h.seq, 4);
    std::memcpy(b + 12, &h.payload_bytes, 4);
}

BenchHdr get_hdr(const uint8_t * b) {
    BenchHdr h;
    std::memcpy(&h.magic, b + 0, 4);
    std::memcpy(&h.type, b + 4, 4);
    std::memcpy(&h.seq, b + 8, 4);
    std::memcpy(&h.payload_bytes, b + 12, 4);
    return h;
}

void put_summary(uint8_t * b, uint64_t frames, uint64_t bytes) {
    std::memcpy(b + 0, &frames, 8);
    std::memcpy(b + 8, &bytes, 8);
}
void get_summary(const uint8_t * b, uint64_t & frames, uint64_t & bytes) {
    std::memcpy(&frames, b + 0, 8);
    std::memcpy(&bytes, b + 8, 8);
}

struct Config {
    std::string role;
    std::string host = "127.0.0.1";
    std::string bind = "127.0.0.1";
    uint16_t port = 0;
    std::string experiment = "h2d"; // host only
    uint32_t payload_bytes = 4U * 1024 * 1024;
    uint32_t count = 16;            // 16 * 4 MiB = 64 MiB
    uint32_t window = 1;            // host h2d only
    int frame_sha = 0;              // 0/1: recompute+verify payload SHA on both ends
    int final_ack = 0;             // host h2d: stream then one ack (streaming ceiling)
    int timeout_ms = 600000;
    std::string json_path;
    std::string label;
};

bool parse_u32(const char * t, uint32_t & v) {
    if (!t || !*t) return false;
    char * e = nullptr; errno = 0;
    unsigned long p = std::strtoul(t, &e, 10);
    if (errno || e == t || *e) return false;
    v = static_cast<uint32_t>(p);
    return true;
}

int fail(const std::string & msg) {
    std::fprintf(stderr, "phone-pim-bench: %s\n", msg.c_str());
    return 1;
}

// ---- device roles ------------------------------------------------------------

int run_sink(const Config & cfg) {
    std::string err;
    pp::Fd listener = pp::listen_tcp(cfg.bind, cfg.port, 1, err);
    if (!listener.valid()) return fail("listen: " + err);
    pp::Fd conn = pp::accept_tcp(listener.get(), cfg.timeout_ms, err);
    if (!conn.valid()) return fail("accept: " + err);

    std::vector<uint8_t> hdr(16);
    std::vector<uint8_t> payload(cfg.payload_bytes);
    std::array<uint8_t, 32> sha_wire = {};
    uint64_t recv_us = 0, sha_us = 0, ack_us = 0;
    uint64_t frames = 0, bytes = 0;
    uint32_t expected_seq = 0;

    for (;;) {
        uint64_t t = now_us();
        pp::ReceiveResult rr = pp::receive_bytes(conn.get(), hdr.data(), hdr.size(), err);
        recv_us += now_us() - t;
        if (rr != pp::ReceiveResult::ok) return fail("recv header: " + err);
        BenchHdr h = get_hdr(hdr.data());
        if (h.magic != k_bench_magic) return fail("bad magic");
        if (h.type == type_shutdown) break;
        if (h.type != type_data) return fail("unexpected frame type");
        if (h.seq != expected_seq) return fail("out-of-order data seq");
        if (h.payload_bytes != cfg.payload_bytes) return fail("unexpected payload size");

        t = now_us();
        rr = pp::receive_bytes(conn.get(), payload.data(), h.payload_bytes, err);
        recv_us += now_us() - t;
        if (rr != pp::ReceiveResult::ok) return fail("recv payload: " + err);

        if (cfg.frame_sha) {
            t = now_us();
            rr = pp::receive_bytes(conn.get(), sha_wire.data(), sha_wire.size(), err);
            recv_us += now_us() - t;
            if (rr != pp::ReceiveResult::ok) return fail("recv sha: " + err);
            t = now_us();
            std::array<uint8_t, 32> got = pp::sha256(payload.data(), h.payload_bytes);
            sha_us += now_us() - t;
            if (got != sha_wire) return fail("payload sha mismatch");
        }

        ++frames;
        bytes += h.payload_bytes;
        ++expected_seq;

        if (!cfg.final_ack) {
            BenchHdr ack{ k_bench_magic, type_ack, h.seq, 0 };
            put_hdr(hdr.data(), ack);
            t = now_us();
            if (!pp::send_bytes(conn.get(), hdr.data(), hdr.size(), err)) return fail("ack: " + err);
            ack_us += now_us() - t;
        }
    }

    // Final summary for host/device count + byte agreement.
    std::vector<uint8_t> summary(k_summary_bytes);
    BenchHdr sum{ k_bench_magic, type_summary, static_cast<uint32_t>(frames), k_summary_bytes };
    put_hdr(hdr.data(), sum);
    put_summary(summary.data(), frames, bytes);
    if (!pp::send_bytes(conn.get(), hdr.data(), hdr.size(), err) ||
        !pp::send_bytes(conn.get(), summary.data(), summary.size(), err)) {
        return fail("summary: " + err);
    }

    std::printf(
            "{\"role\":\"sink\",\"frames\":%llu,\"bytes\":%llu,\"frame_sha\":%d,"
            "\"final_ack\":%d,\"dev_recv_ms\":%.3f,\"dev_sha_ms\":%.3f,\"dev_ack_ms\":%.3f}\n",
            static_cast<unsigned long long>(frames), static_cast<unsigned long long>(bytes),
            cfg.frame_sha, cfg.final_ack, recv_us / 1000.0, sha_us / 1000.0, ack_us / 1000.0);
    return 0;
}

int run_source(const Config & cfg) {
    std::string err;
    pp::Fd listener = pp::listen_tcp(cfg.bind, cfg.port, 1, err);
    if (!listener.valid()) return fail("listen: " + err);
    pp::Fd conn = pp::accept_tcp(listener.get(), cfg.timeout_ms, err);
    if (!conn.valid()) return fail("accept: " + err);

    std::vector<uint8_t> hdr(16);
    pp::ReceiveResult rr = pp::receive_bytes(conn.get(), hdr.data(), hdr.size(), err);
    if (rr != pp::ReceiveResult::ok) return fail("recv start: " + err);
    BenchHdr start = get_hdr(hdr.data());
    if (start.magic != k_bench_magic || start.type != type_start_d2h) return fail("bad start");
    const uint32_t count = start.seq;
    const uint32_t payload_bytes = start.payload_bytes;

    std::vector<uint8_t> payload(payload_bytes, 0xA5);
    std::array<uint8_t, 32> sha = cfg.frame_sha ? pp::sha256(payload.data(), payload_bytes)
                                                : std::array<uint8_t, 32>{};
    uint64_t send_us = 0, bytes = 0;
    for (uint32_t i = 0; i < count; ++i) {
        BenchHdr h{ k_bench_magic, type_data, i, payload_bytes };
        put_hdr(hdr.data(), h);
        uint64_t t = now_us();
        if (!pp::send_bytes(conn.get(), hdr.data(), hdr.size(), err) ||
            !pp::send_bytes(conn.get(), payload.data(), payload_bytes, err) ||
            (cfg.frame_sha && !pp::send_bytes(conn.get(), sha.data(), sha.size(), err))) {
            return fail("send: " + err);
        }
        send_us += now_us() - t;
        bytes += payload_bytes;
    }
    // Final summary for host/device count + byte agreement.
    std::vector<uint8_t> summary(k_summary_bytes);
    BenchHdr sum{ k_bench_magic, type_summary, count, k_summary_bytes };
    put_hdr(hdr.data(), sum);
    put_summary(summary.data(), count, bytes);
    if (!pp::send_bytes(conn.get(), hdr.data(), hdr.size(), err) ||
        !pp::send_bytes(conn.get(), summary.data(), summary.size(), err)) {
        return fail("summary: " + err);
    }
    std::printf("{\"role\":\"source\",\"frames\":%u,\"bytes\":%llu,\"dev_send_ms\":%.3f}\n",
            count, static_cast<unsigned long long>(bytes), send_us / 1000.0);
    return 0;
}

// ---- host role ---------------------------------------------------------------

void emit_host_json(const Config & cfg, uint64_t wall_us, uint64_t send_us, uint64_t recv_us,
                    uint64_t sha_us, uint64_t bytes) {
    const double mib = bytes / 1048576.0;
    const double goodput = wall_us == 0 ? 0.0 : mib * 1000000.0 / wall_us;
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
            "{\"role\":\"host\",\"experiment\":\"%s\",\"label\":\"%s\",\"payload_bytes\":%u,"
            "\"count\":%u,\"window\":%u,\"frame_sha\":%d,\"final_ack\":%d,\"bytes\":%llu,"
            "\"wall_ms\":%.3f,\"host_send_ms\":%.3f,\"host_recv_ms\":%.3f,\"host_sha_ms\":%.3f,"
            "\"goodput_mib_s\":%.3f}\n",
            cfg.experiment.c_str(), cfg.label.c_str(), cfg.payload_bytes, cfg.count, cfg.window,
            cfg.frame_sha, cfg.final_ack, static_cast<unsigned long long>(bytes), wall_us / 1000.0,
            send_us / 1000.0, recv_us / 1000.0, sha_us / 1000.0, goodput);
    std::fputs(buf, stdout);
    if (!cfg.json_path.empty()) {
        FILE * f = std::fopen(cfg.json_path.c_str(), "we");
        if (f) { std::fputs(buf, f); std::fclose(f); }
    }
}

int run_host_h2d(const Config & cfg, pp::Fd & conn) {
    std::string err;
    std::vector<uint8_t> hdr(16);
    std::vector<uint8_t> ackhdr(16);
    std::vector<uint8_t> payload(cfg.payload_bytes, 0x5A);
    std::array<uint8_t, 32> sha = cfg.frame_sha ? pp::sha256(payload.data(), cfg.payload_bytes)
                                                : std::array<uint8_t, 32>{};
    const uint32_t window = std::max<uint32_t>(1, cfg.window);
    uint64_t send_us = 0, recv_us = 0, sha_us = 0, bytes = 0;
    uint32_t next = 0, acked = 0;
    std::deque<uint32_t> outstanding; // seqs sent but not yet ACKed (window mode)

    auto send_data = [&](uint32_t seq) -> bool {
        BenchHdr h{ k_bench_magic, type_data, seq, cfg.payload_bytes };
        put_hdr(hdr.data(), h);
        if (cfg.frame_sha) { uint64_t s = now_us(); sha = pp::sha256(payload.data(), cfg.payload_bytes); sha_us += now_us() - s; }
        uint64_t t = now_us();
        const bool ok = pp::send_bytes(conn.get(), hdr.data(), hdr.size(), err) &&
                        pp::send_bytes(conn.get(), payload.data(), cfg.payload_bytes, err) &&
                        (!cfg.frame_sha || pp::send_bytes(conn.get(), sha.data(), sha.size(), err));
        send_us += now_us() - t;
        if (ok) bytes += cfg.payload_bytes;
        return ok;
    };

    const uint64_t start = now_us();
    if (cfg.final_ack) {
        for (; next < cfg.count; ++next) {
            if (!send_data(next)) return fail("send: " + err);
        }
        acked = cfg.count;
    } else {
        while (acked < cfg.count) {
            while (outstanding.size() < window && next < cfg.count) {
                if (!send_data(next)) return fail("send: " + err);
                outstanding.push_back(next);
                ++next;
            }
            uint64_t t = now_us();
            if (pp::receive_bytes(conn.get(), ackhdr.data(), ackhdr.size(), err) != pp::ReceiveResult::ok)
                return fail("ack: " + err);
            recv_us += now_us() - t;
            BenchHdr a = get_hdr(ackhdr.data());
            if (a.magic != k_bench_magic || a.type != type_ack) return fail("bad ack");
            if (outstanding.empty() || a.seq != outstanding.front()) return fail("ack seq mismatch");
            outstanding.pop_front();
            ++acked;
        }
    }
    const uint64_t wall = now_us() - start;

    // Shut down, then read the device summary and verify count + byte agreement.
    BenchHdr sd{ k_bench_magic, type_shutdown, 0, 0 };
    put_hdr(hdr.data(), sd);
    if (!pp::send_bytes(conn.get(), hdr.data(), hdr.size(), err)) return fail("shutdown: " + err);
    if (pp::receive_bytes(conn.get(), hdr.data(), hdr.size(), err) != pp::ReceiveResult::ok)
        return fail("summary header: " + err);
    BenchHdr sh = get_hdr(hdr.data());
    if (sh.magic != k_bench_magic || sh.type != type_summary || sh.payload_bytes != k_summary_bytes)
        return fail("bad summary");
    std::vector<uint8_t> sbuf(k_summary_bytes);
    if (pp::receive_bytes(conn.get(), sbuf.data(), sbuf.size(), err) != pp::ReceiveResult::ok)
        return fail("summary payload: " + err);
    uint64_t dev_frames = 0, dev_bytes = 0;
    get_summary(sbuf.data(), dev_frames, dev_bytes);
    if (dev_frames != cfg.count) return fail("device/host frame count disagreement");
    if (dev_bytes != bytes) return fail("device/host byte count disagreement");
    emit_host_json(cfg, wall, send_us, recv_us, sha_us, bytes);
    return 0;
}

int run_host_d2h(const Config & cfg, pp::Fd & conn) {
    std::string err;
    std::vector<uint8_t> hdr(16);
    std::vector<uint8_t> payload(cfg.payload_bytes);
    std::array<uint8_t, 32> sha_wire = {};

    BenchHdr go{ k_bench_magic, type_start_d2h, cfg.count, cfg.payload_bytes };
    put_hdr(hdr.data(), go);
    if (!pp::send_bytes(conn.get(), hdr.data(), hdr.size(), err)) return fail("start: " + err);

    uint64_t recv_us = 0, sha_us = 0, bytes = 0;
    const uint64_t start = now_us();
    for (uint32_t i = 0; i < cfg.count; ++i) {
        uint64_t t = now_us();
        if (pp::receive_bytes(conn.get(), hdr.data(), hdr.size(), err) != pp::ReceiveResult::ok)
            return fail("recv header: " + err);
        recv_us += now_us() - t;
        BenchHdr h = get_hdr(hdr.data());
        if (h.magic != k_bench_magic || h.type != type_data || h.seq != i ||
            h.payload_bytes != cfg.payload_bytes)
            return fail("bad data frame");
        t = now_us();
        if (pp::receive_bytes(conn.get(), payload.data(), h.payload_bytes, err) != pp::ReceiveResult::ok)
            return fail("recv payload: " + err);
        recv_us += now_us() - t;
        if (cfg.frame_sha) {
            t = now_us();
            if (pp::receive_bytes(conn.get(), sha_wire.data(), sha_wire.size(), err) != pp::ReceiveResult::ok)
                return fail("recv sha: " + err);
            recv_us += now_us() - t;
            t = now_us();
            std::array<uint8_t, 32> got = pp::sha256(payload.data(), h.payload_bytes);
            sha_us += now_us() - t;
            if (got != sha_wire) return fail("d2h sha mismatch");
        }
        bytes += h.payload_bytes;
    }
    const uint64_t wall = now_us() - start;
    // Read the device summary and verify count + byte agreement.
    if (pp::receive_bytes(conn.get(), hdr.data(), hdr.size(), err) != pp::ReceiveResult::ok)
        return fail("summary header: " + err);
    BenchHdr sh = get_hdr(hdr.data());
    if (sh.magic != k_bench_magic || sh.type != type_summary || sh.payload_bytes != k_summary_bytes)
        return fail("bad summary");
    std::vector<uint8_t> sbuf(k_summary_bytes);
    if (pp::receive_bytes(conn.get(), sbuf.data(), sbuf.size(), err) != pp::ReceiveResult::ok)
        return fail("summary payload: " + err);
    uint64_t dev_frames = 0, dev_bytes = 0;
    get_summary(sbuf.data(), dev_frames, dev_bytes);
    if (dev_frames != cfg.count) return fail("device/host frame count disagreement");
    if (dev_bytes != bytes) return fail("device/host byte count disagreement");
    emit_host_json(cfg, wall, 0, recv_us, sha_us, bytes);
    return 0;
}

int run_host(const Config & cfg) {
    std::string err;
    pp::Fd conn = pp::connect_tcp(cfg.host, cfg.port, cfg.timeout_ms, err);
    if (!conn.valid()) return fail("connect: " + err);
    if (cfg.experiment == "h2d") return run_host_h2d(cfg, conn);
    if (cfg.experiment == "d2h") return run_host_d2h(cfg, conn);
    return fail("unknown experiment: " + cfg.experiment);
}

} // namespace

int main(int argc, char ** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
            return argv[++i];
        };
        uint32_t u = 0;
        if (a == "--role") cfg.role = next("--role");
        else if (a == "--host") cfg.host = next("--host");
        else if (a == "--bind") cfg.bind = next("--bind");
        else if (a == "--port") { if (!parse_u32(next("--port"), u)) return fail("bad --port"); cfg.port = static_cast<uint16_t>(u); }
        else if (a == "--experiment") cfg.experiment = next("--experiment");
        else if (a == "--payload-bytes") { if (!parse_u32(next("--payload-bytes"), cfg.payload_bytes)) return fail("bad --payload-bytes"); }
        else if (a == "--count") { if (!parse_u32(next("--count"), cfg.count)) return fail("bad --count"); }
        else if (a == "--window") { if (!parse_u32(next("--window"), cfg.window)) return fail("bad --window"); }
        else if (a == "--frame-sha") { if (!parse_u32(next("--frame-sha"), u)) return fail("bad --frame-sha"); cfg.frame_sha = u ? 1 : 0; }
        else if (a == "--final-ack") cfg.final_ack = 1;
        else if (a == "--timeout-ms") { if (!parse_u32(next("--timeout-ms"), u)) return fail("bad --timeout-ms"); cfg.timeout_ms = static_cast<int>(u); }
        else if (a == "--json") cfg.json_path = next("--json");
        else if (a == "--label") cfg.label = next("--label");
        else return fail("unknown arg: " + a);
    }
    if (cfg.port == 0) return fail("--port required");
    if (cfg.payload_bytes == 0) return fail("--payload-bytes must be > 0");
    if (cfg.role == "sink") return run_sink(cfg);
    if (cfg.role == "source") return run_source(cfg);
    if (cfg.role == "host") return run_host(cfg);
    return fail("--role must be sink|source|host");
}
