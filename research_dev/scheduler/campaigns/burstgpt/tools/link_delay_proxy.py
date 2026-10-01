#!/usr/bin/env python3
"""Host-side TCP delay proxy for a helper phone's adb-forward link (link-latency sweep).

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.link_delay_proxy \\
        --listen 127.0.0.1:26992 --upstream 127.0.0.1:26991 --rtt-ms 3.5 \\
        [--up-delay-ms D --down-delay-ms D] [--up-jitter-ms J] [--down-jitter-ms J] [--seed 1] \\
        [--ready-file PATH] [--status PATH] [--status-interval-s 30]

llama-server dials --listen (rig ``helper_phones[].link_delay_proxy_port``); every accepted connection
opens its own upstream connection to --upstream (the adb forward, ``helper_phones[].forward_port``), so the
phone worker still sees exactly one client per server connection and a disconnect when it closes. Each
chunk read from one side is written to the other side ``delay + U(0, jitter)`` after it was read, never
before an earlier chunk of the same direction (ordering is preserved; a burst queues behind its head).
"up" is server -> phone, "down" is phone -> server; --rtt-ms sets both delays to half of it. EOF is
delayed like data. Bandwidth is not limited: an FFN call carries 10-40 KB, so only latency matters.

Timers: asyncio sleeps until 2 ms before the release time, a pool thread sleeps the rest with
``time.sleep`` (sub-millisecond precision; epoll timeouts alone round up to whole milliseconds) and ends
early by a per-direction bias learned from the observed wake-up lateness, so the applied delay is unbiased
(release lateness is reported signed). Nothing spins, so the proxy adds no measurable host CPU load. The
proxy hop itself adds ~0.1-0.3 ms per round trip on loopback: run the 0-delay point through the proxy too. Stats (per direction: bytes, chunks, applied delay
and release lateness percentiles) go to --status at exit (SIGTERM/SIGINT) and every --status-interval-s.
The proxy never touches adb, the phone or the scheduler.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import signal
import socket
import threading
import time
from typing import Any

SCHEMA = "ws6-link-delay-proxy-v1"
CHUNK_BYTES = 256 * 1024
QUEUE_CHUNKS = 1024
COARSE_MARGIN_S = 0.002
SAMPLE_LIMIT = 200_000
# the precise sleep ends this much before the release time, learned per direction from the observed
# wake-up lateness (thread -> event loop handoff, ~0.2-0.4 ms), so the applied delay is unbiased
BIAS_GAIN = 0.1
BIAS_MAX_S = 0.001


@dataclass(frozen=True)
class DirectionConfig:
    delay_s: float = 0.0
    jitter_s: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.delay_s <= 10.0 and 0.0 <= self.jitter_s <= 10.0):
            raise ValueError("delay and jitter must be within [0, 10] s")

    def to_json(self) -> dict[str, float]:
        return {"delay_ms": self.delay_s * 1e3, "jitter_ms": self.jitter_s * 1e3}


@dataclass(frozen=True)
class ProxyConfig:
    listen_host: str
    listen_port: int
    upstream_host: str
    upstream_port: int
    up: DirectionConfig = DirectionConfig()
    down: DirectionConfig = DirectionConfig()
    seed: int = 1

    def to_json(self) -> dict[str, Any]:
        return {"listen": f"{self.listen_host}:{self.listen_port}",
                "upstream": f"{self.upstream_host}:{self.upstream_port}",
                "up": self.up.to_json(), "down": self.down.to_json(), "seed": self.seed}


@dataclass
class DirectionStats:
    chunks: int = 0
    bytes: int = 0
    delays_s: list[float] = field(default_factory=list)
    lateness_s: list[float] = field(default_factory=list)
    wake_bias_s: float = 0.0

    def record(self, size: int, delay_s: float, lateness_s: float) -> None:
        self.chunks += 1
        self.bytes += size
        if len(self.delays_s) < SAMPLE_LIMIT:
            self.delays_s.append(delay_s)
            self.lateness_s.append(lateness_s)

    def to_json(self) -> dict[str, Any]:
        def quantiles(values: list[float]) -> dict[str, float] | None:
            if not values:
                return None
            ordered = sorted(values)

            def pick(q: float) -> float:
                return ordered[min(len(ordered) - 1, int(q * len(ordered)))] * 1e3
            return {"p50_ms": pick(0.5), "p90_ms": pick(0.9), "p99_ms": pick(0.99), "max_ms": ordered[-1] * 1e3,
                    "mean_ms": sum(ordered) / len(ordered) * 1e3}
        return {"chunks": self.chunks, "bytes": self.bytes, "sampled_chunks": len(self.delays_s),
                "applied_delay": quantiles(self.delays_s), "release_lateness": quantiles(self.lateness_s),
                "wake_bias_ms": self.wake_bias_s * 1e3}


def _precise_sleep_until(deadline: float) -> None:
    while (remaining := deadline - time.monotonic()) > 0:
        time.sleep(remaining)


def _no_delay(writer: asyncio.StreamWriter) -> None:
    sock = writer.get_extra_info("socket")
    if sock is not None:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass


class LinkDelayProxy:
    """One listening socket; per accepted connection one upstream connection and two delayed pumps."""

    def __init__(self, config: ProxyConfig) -> None:
        self.config = config
        self._random = random.Random(config.seed)
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="link-delay")
        self._server: asyncio.base_events.Server | None = None
        self._tasks: set[asyncio.Task] = set()
        self.port: int | None = None
        self.started_at = time.time()
        self.stats = {"up": DirectionStats(), "down": DirectionStats()}
        self.connections = {"accepted": 0, "closed": 0, "upstream_failures": 0, "errors": 0}

    async def start(self) -> int:
        self._server = await asyncio.start_server(self._accept, self.config.listen_host, self.config.listen_port)
        self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._executor.shutdown(wait=True)

    def status(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "pid": os.getpid(), "config": self.config.to_json(), "port": self.port,
                "started_at_unix_s": self.started_at, "written_at_unix_s": time.time(),
                "connections": dict(self.connections),
                "directions": {name: row.to_json() for name, row in self.stats.items()}}

    def _sample(self, direction: DirectionConfig) -> float:
        return direction.delay_s + (self._random.uniform(0.0, direction.jitter_s) if direction.jitter_s else 0.0)

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._tasks.add(task)
        self.connections["accepted"] += 1
        try:
            try:
                up_reader, up_writer = await asyncio.open_connection(self.config.upstream_host, self.config.upstream_port)
            except OSError:
                self.connections["upstream_failures"] += 1
                writer.close()
                return
            _no_delay(writer)
            _no_delay(up_writer)
            pumps = {asyncio.create_task(self._pump(reader, up_writer, self.config.up, self.stats["up"])),
                     asyncio.create_task(self._pump(up_reader, writer, self.config.down, self.stats["down"]))}
            try:
                done, pending = await asyncio.wait(pumps, return_when=asyncio.FIRST_EXCEPTION)
                failed = [row for row in done if not row.cancelled() and row.exception() is not None]
                if failed:
                    self.connections["errors"] += 1
                    for row in pending:
                        row.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
            finally:
                for row in pumps:
                    row.cancel()
                for side in (up_writer, writer):
                    side.close()
                for side in (up_writer, writer):
                    try:
                        await side.wait_closed()
                    except (OSError, asyncio.CancelledError):
                        pass
        finally:
            self.connections["closed"] += 1
            self._tasks.discard(task)

    async def _pump(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, direction: DirectionConfig,
                    stats: DirectionStats) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_CHUNKS)
        delayed = bool(direction.delay_s or direction.jitter_s)

        async def receive() -> None:
            # a read error is queued like EOF, behind the data already read, and raised by send()
            last_release = 0.0
            while True:
                error: BaseException | None = None
                try:
                    data = await reader.read(CHUNK_BYTES)
                except (ConnectionError, OSError) as failure:
                    data, error = b"", failure
                arrival = loop.time()
                release = max(last_release, arrival + self._sample(direction)) if delayed else arrival
                last_release = release
                await queue.put((arrival, release, data, error))
                if not data:
                    return

        async def send() -> None:
            while True:
                arrival, release, data, error = await queue.get()
                if error is not None:
                    raise error
                slept = False
                if delayed:
                    wake = release - stats.wake_bias_s
                    remaining = wake - loop.time()
                    slept = remaining > 0
                    if remaining > COARSE_MARGIN_S:
                        await asyncio.sleep(remaining - COARSE_MARGIN_S)
                    if wake > time.monotonic():
                        await loop.run_in_executor(self._executor, _precise_sleep_until, wake)
                if not data:
                    if writer.can_write_eof():
                        writer.write_eof()
                    return
                sent = loop.time()
                writer.write(data)
                await writer.drain()
                if slept:
                    stats.wake_bias_s = min(BIAS_MAX_S, max(0.0, stats.wake_bias_s + BIAS_GAIN * (sent - release)))
                stats.record(len(data), sent - arrival, sent - release)

        receiver = asyncio.create_task(receive())
        try:
            await send()
            await receiver
        finally:
            receiver.cancel()


def _address(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit() or not 0 <= int(port) <= 65535:
        raise argparse.ArgumentTypeError("address must be HOST:PORT")
    return host, int(port)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class ProxyThread:
    """Run a proxy on its own event loop thread (tests and Python harnesses): ``with ProxyThread(c) as p``."""

    def __init__(self, config: ProxyConfig) -> None:
        self.proxy = LinkDelayProxy(config)
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="link-delay-proxy", daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self.proxy.start())
        self._ready.set()
        self._loop.run_forever()
        self._loop.run_until_complete(self.proxy.close())
        self._loop.close()

    def __enter__(self) -> "ProxyThread":
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("link delay proxy did not start")
        return self

    def __exit__(self, *exc: object) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)

    @property
    def port(self) -> int:
        assert self.proxy.port is not None
        return self.proxy.port


async def _serve(config: ProxyConfig, ready_file: Path | None, status_file: Path | None, interval_s: float) -> None:
    proxy = LinkDelayProxy(config)
    port = await proxy.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for number in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(number, stop.set)
    ready = {"schema": SCHEMA, "pid": os.getpid(), "port": port, "config": config.to_json()}
    if ready_file is not None:
        _write_json(ready_file, ready)
    print("LINK_DELAY_PROXY_READY " + json.dumps(ready, sort_keys=True), flush=True)
    try:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_s)
            except asyncio.TimeoutError:
                if status_file is not None:
                    _write_json(status_file, proxy.status())
    finally:
        await proxy.close()
        if status_file is not None:
            _write_json(status_file, {**proxy.status(), "stopped": True})


def parse_config(argv: list[str] | None = None) -> tuple[ProxyConfig, argparse.Namespace]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", type=_address, required=True, help="HOST:PORT llama-server dials (0 = any free port)")
    ap.add_argument("--upstream", type=_address, required=True, help="HOST:PORT of the adb forward")
    ap.add_argument("--rtt-ms", type=float, default=None, help="added round trip; sets both one-way delays to half")
    ap.add_argument("--up-delay-ms", type=float, default=None, help="server -> phone one-way delay")
    ap.add_argument("--down-delay-ms", type=float, default=None, help="phone -> server one-way delay")
    ap.add_argument("--up-jitter-ms", type=float, default=0.0, help="extra U(0, J) per chunk, server -> phone")
    ap.add_argument("--down-jitter-ms", type=float, default=0.0, help="extra U(0, J) per chunk, phone -> server")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ready-file", type=Path, default=None)
    ap.add_argument("--status", type=Path, default=None)
    ap.add_argument("--status-interval-s", type=float, default=30.0)
    args = ap.parse_args(argv)
    if args.rtt_ms is not None and (args.up_delay_ms is not None or args.down_delay_ms is not None):
        ap.error("--rtt-ms excludes --up-delay-ms/--down-delay-ms")
    half = (args.rtt_ms or 0.0) / 2
    up = args.up_delay_ms if args.up_delay_ms is not None else half
    down = args.down_delay_ms if args.down_delay_ms is not None else half
    config = ProxyConfig(args.listen[0], args.listen[1], args.upstream[0], args.upstream[1],
                         DirectionConfig(up / 1e3, args.up_jitter_ms / 1e3),
                         DirectionConfig(down / 1e3, args.down_jitter_ms / 1e3), args.seed)
    return config, args


def main(argv: list[str] | None = None) -> int:
    config, args = parse_config(argv)
    asyncio.run(_serve(config, args.ready_file, args.status, args.status_interval_s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
