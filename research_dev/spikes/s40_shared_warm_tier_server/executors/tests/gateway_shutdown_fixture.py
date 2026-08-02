#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys


HERE = Path(__file__).resolve().parent
EXECUTORS = HERE.parent
if str(EXECUTORS) not in sys.path:
    sys.path.insert(0, str(EXECUTORS))

from phone_gateway import GatewayServer, canonical_bytes
from runtime_binding import RuntimeBinding, process_start_time_ticks


class FakeExecutor:
    def __init__(self, active_path: Path, route_evidence: Path):
        self.active_path = active_path
        self.route_evidence = route_evidence
        self.executor_id = "PHONE"
        self.route_specs = {"model-a": object()}
        self.timeout_s = 2.0

    def take_execute_evidence(self, command_id):
        del command_id
        return None

    def handle(self, command):
        raise AssertionError(command)

    def close(self):
        with self.route_evidence.open("xb", buffering=0) as output:
            output.write(canonical_bytes({
                "model_id": "model-a",
                "route_instance_id": "route-a",
                "schema": "s40-phone-route-unload-v1",
                "success": True,
            }))
            output.flush()
            os.fsync(output.fileno())
        self.active_path.unlink()
        descriptor = os.open(
            self.active_path.parent,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class FakeAuthenticator:
    @staticmethod
    def authenticate(connection):
        del connection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--route-evidence", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()
    server = GatewayServer(
        args.socket,
        FakeExecutor(args.active, args.route_evidence),
        args.evidence,
        controller_authenticator=FakeAuthenticator(),
        route_config_sha256="1" * 64,
        run_id="shutdown-test",
        runtime_binding=RuntimeBinding(
            executor_id="PHONE",
            executor_instance_id="instance-phone",
            gateway_pid=os.getpid(),
            gateway_start_time_ticks=process_start_time_ticks(),
            runtime_config_path=str(args.active.parent / "runtime.json"),
            runtime_config_sha256="2" * 64,
            runtime_config_device=1,
            runtime_config_inode=1,
        ),
    )
    previous = signal.signal(
        signal.SIGTERM,
        lambda _signum, _frame: server.request_stop(),
    )
    try:
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
