"""Check resident worker reconnect and refusal to stop a connected client."""

import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, os.environ["S42_UNIFIED_REPO_ROOT"])
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError  # noqa: E402
from research_dev.scheduler.adapters.phone_tcp_session import (  # noqa: E402
    AdbTcpPhoneWorkerSession, HELLO_REQUEST, HELLO_RESPONSE, PROTOCOL_MAGIC, PROTOCOL_VERSION,
)
from research_dev.scheduler.campaigns.burstgpt.helper_phone_evidence import load_helper_evidence  # noqa: E402


def main():
    evidence = load_helper_evidence(Path(sys.argv[1]))
    output = Path(sys.argv[2])
    output.mkdir()
    session = AdbTcpPhoneWorkerSession(evidence.worker)
    result = {"status": "FAIL", "preflight": evidence.live_preflight()}
    try:
        result["launch"] = session.start(output / "worker.log").to_json()
        assert len(result["launch"]["worker_pids"]) == 1
        configuration = evidence.worker
        result["connections"] = []
        for index in range(2):
            with socket.create_connection(("127.0.0.1", configuration.forward_port), timeout=60) as stream:
                stream.sendall(HELLO_REQUEST.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, 1,
                    configuration.layer_mask, configuration.n_embd, configuration.columns,
                    3, configuration.max_tokens, bytes.fromhex(configuration.artifact_sha256[7:])))
                hello = HELLO_RESPONSE.unpack(session._receive(stream, HELLO_RESPONSE.size))
                assert hello[:4] == (PROTOCOL_MAGIC, PROTOCOL_VERSION, 2, 0)
                try:
                    session.stop(allow_idle_signal=True, timeout_s=1)
                except PhysicalAdapterError as error:
                    assert "no client" in str(error)
                    result["connections"].append({"index": index, "connected_stop_refused": str(error)})
                else:
                    raise RuntimeError("worker accepted stop with a connected client")
            session._wait_disconnected(10)
        result["stop"] = session.stop(allow_idle_signal=True).to_json()
        assert result["stop"]["boot_unchanged"] and result["stop"]["forward_removed"]
        assert result["stop"]["signalled"]
        assert not result["stop"]["worker_pids_after"]
        result["status"] = "PASS"
    finally:
        if session.active:
            session._wait_disconnected(10)
            result["cleanup"] = session.stop(allow_idle_signal=True).to_json()
        (output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
