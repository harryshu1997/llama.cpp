#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time


HERE = Path(__file__).resolve().parent
EXECUTORS = HERE.parent
if str(EXECUTORS) not in sys.path:
    sys.path.insert(0, str(EXECUTORS))

from desktop_gateway import DesktopGateway
from runtime_binding import RuntimeBinding, process_start_time_ticks


class FakeExecutor:
    cache_regime = "WARM_CACHE"
    executor_id = "GPU"
    mode = "SINGLE_ACTIVE"
    role = "GPU"

    @staticmethod
    def runtime_inventory():
        return []

    @staticmethod
    def close():
        now = time.monotonic_ns()
        return {
            "completed_ns": now,
            "initial_active_models": [],
            "initial_busy_requests": [],
            "initial_request_sessions": [],
            "problems": [],
            "remaining_active_models": [],
            "remaining_request_sessions": [],
            "schema": "s40-desktop-cleanup-evidence-v1",
            "started_ns": now,
            "success": True,
            "unloaded": [],
        }


class FakeAuthenticator:
    @staticmethod
    def authenticate(connection):
        del connection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    args = parser.parse_args()
    gateway = DesktopGateway(
        args.socket,
        args.evidence,
        FakeExecutor(),
        2.0,
        controller_authenticator=FakeAuthenticator(),
        executor_config_sha256="1" * 64,
        profile_lock_sha256=None,
        run_id="shutdown-test",
        runtime_binding=RuntimeBinding(
            executor_id="GPU",
            executor_instance_id="instance-gpu",
            gateway_pid=os.getpid(),
            gateway_start_time_ticks=process_start_time_ticks(),
            runtime_config_path=str(args.evidence.parent / "runtime.json"),
            runtime_config_sha256="2" * 64,
            runtime_config_device=1,
            runtime_config_inode=1,
        ),
    )
    previous = signal.signal(
        signal.SIGTERM,
        lambda _signum, _frame: gateway.request_stop(),
    )
    try:
        gateway.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
