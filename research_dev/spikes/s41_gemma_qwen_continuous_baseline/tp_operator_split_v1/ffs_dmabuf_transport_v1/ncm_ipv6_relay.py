#!/usr/bin/env python3

import argparse
import select
import socket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--target", default="fe80::2")
    parser.add_argument("--target-port", type=int, required=True)
    return parser.parse_args()


def relay(left: socket.socket, right: socket.socket) -> tuple[int, int]:
    peers = {left: right, right: left}
    counts = {left: 0, right: 0}
    active = {left, right}
    while active:
        readable, _, _ = select.select(list(active), [], [])
        for source in readable:
            data = source.recv(1024 * 1024)
            destination = peers[source]
            if data:
                destination.sendall(data)
                counts[source] += len(data)
                continue
            active.remove(source)
            destination.shutdown(socket.SHUT_WR)
    return counts[left], counts[right]


def main() -> int:
    args = parse_args()
    scope_id = socket.if_nametoindex(args.interface)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.bind, args.port))
        listener.listen(1)
        print(
            f"[ncm-relay] ready bind={args.bind}:{args.port} "
            f"target=[{args.target}%{args.interface}]:{args.target_port}",
            flush=True,
        )
        client, _ = listener.accept()

    with client, socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as phone:
        phone.connect((args.target, args.target_port, 0, scope_id))
        print("[ncm-relay] connected", flush=True)
        uploaded, downloaded = relay(client, phone)
        print(
            f"[ncm-relay] complete upload_bytes={uploaded} "
            f"download_bytes={downloaded}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
