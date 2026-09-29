#!/usr/bin/env python3
"""Round-trip latency of phone FFN-sized messages over a network link (WiFi or adb-forwarded TCP).

Phone side (rooted or not, stock toybox): an echo server
    adb shell "toybox nc -L -p 7070 cat"          # or: nc -L -p 7070 cat
Host side:
    python3 wifi_rtt.py --host <phone-ip> --port 7070 --bytes 10240 --calls 5000 --gap-ms 8 --out rtt.json

Each call sends `--bytes` (default 10,240 = one Qwen3-14B row in f16) and waits for the same number of bytes back,
like the synchronous per-layer FFN exchange; `--gap-ms` spaces calls like the decode cadence (the phone computes
~8-9 ms between exchanges). Reports p50/p90/p99/p99.9/max and the fraction of calls over thresholds, and the
per-token cost for N synchronous calls per token (the tail matters: a token waits on every one of its calls).
"""
import argparse, json, socket, statistics, time


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=7070)
    ap.add_argument("--bytes", type=int, default=10240)
    ap.add_argument("--calls", type=int, default=5000)
    ap.add_argument("--gap-ms", type=float, default=8.0)
    ap.add_argument("--calls-per-token", type=int, default=24)
    ap.add_argument("--out")
    ap.add_argument("--tos", type=lambda x: int(x, 0), default=0,
                    help="IP TOS byte for our packets, e.g. 0xb8 = DSCP EF (WMM voice queue)")
    a = ap.parse_args()
    payload = bytes(range(256)) * (a.bytes // 256) + bytes(a.bytes % 256)
    s = socket.create_connection((a.host, a.port), timeout=5)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if a.tos:
        s.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, a.tos)
    rtts = []
    for i in range(a.calls):
        t0 = time.perf_counter_ns()
        s.sendall(payload)
        got = 0
        while got < a.bytes:
            chunk = s.recv(a.bytes - got)
            if not chunk:
                raise SystemExit("connection closed after %d calls" % i)
            got += len(chunk)
        rtts.append((time.perf_counter_ns() - t0) / 1e6)
        if a.gap_ms > 0:
            time.sleep(a.gap_ms / 1e3)
    s.close()
    rtts_sorted = sorted(rtts)
    n = a.calls_per_token
    # per-token transport = sum of n consecutive calls (synchronous chain)
    tokens = [sum(rtts[i:i + n]) for i in range(0, len(rtts) - n + 1, n)]
    res = {
        "host": a.host, "port": a.port, "tos": a.tos, "bytes": a.bytes, "calls": a.calls, "gap_ms": a.gap_ms,
        "rtt_ms": {"p50": statistics.median(rtts), "p90": pct(rtts, 0.90), "p99": pct(rtts, 0.99),
                   "p999": pct(rtts, 0.999), "max": rtts_sorted[-1], "mean": statistics.mean(rtts)},
        "frac_over_ms": {str(t): sum(1 for x in rtts if x > t) / len(rtts) for t in (2, 5, 10, 20, 50)},
        "per_token_transport_ms": {"calls_per_token": n, "p50": statistics.median(tokens), "p99": pct(tokens, 0.99),
                                   "max": max(tokens)},
    }
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump({**res, "samples_ms": rtts}, open(a.out, "w"))


if __name__ == "__main__":
    main()
