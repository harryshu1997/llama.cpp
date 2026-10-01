#!/usr/bin/env python3
"""WS10: Pixel-only llama-server token identity over a chosen helper transport (adb-tcp or aoa-bridge).

    cd <repo root> && python3 -m research_dev.scheduler.campaigns.burstgpt.tools.qualify_pixel_server_transport \\
        --config SERVER_IDENTITY_CONFIG.json --output OUT --transport aoa-bridge --aoa-bridge AOA_BRIDGE.json \\
        --forward-port 26991

Same experiment and output files as ``reports/20260925-two-phone-eval/tools/qualify_pixel_server.py`` (desktop /
Pixel 8,704 / Pixel 17,408 / desktop, identical tokens required; config derived by make_server_identity_config.py),
so ``prepare_campaign_eval.server_identity()`` validates it unchanged. Differences: the worker (finite budget) and
its host endpoint come from the scheduler's own session class of the transport under test, the budget remainder is
drained through that transport by the session stop, and IDENTITY.json / RESULT.json also record the transport
(for aoa-bridge: relay, host bridge and keep-awake option pins). Run under the rig lock; OP15 is never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from research_dev.scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeControl, AdaptiveDecodePolicy
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler.adapters.phone_aoa_session import AoaBridgeConfiguration, AoaBridgePhoneWorkerSession
from research_dev.scheduler.adapters.phone_tcp_session import AdbTcpPhoneWorkerSession, AdbTcpWorkerConfiguration


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def hex_digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def request_json(port: int, method: str, path: str, body: object | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.request(method, path, body=None if body is None else json.dumps(body),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read()
        return json.loads(data) if response.status == 200 else None
    except (OSError, ValueError):
        return None
    finally:
        connection.close()


def wait_ready(process, check, seconds: float = 240) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited with {process.returncode}")
        if check():
            return
        time.sleep(0.5)
    raise TimeoutError("readiness")


def worker_configuration(config: dict, forward_port: int) -> AdbTcpWorkerConfiguration:
    phone = config["phone"]
    pins = {path: "sha256:" + value.removeprefix("sha256:") for path, value in config["expected_phone_sha256"].items()}
    return AdbTcpWorkerConfiguration(
        device_id="pixel10pro-phone", serial=config["serial"], adb_port=5037,
        adb_path=Path(config.get("adb_path", "/usr/bin/adb")), worker_path=config["phone_worker"],
        library_directories=(config["phone_library_dir"],), shard_path=config["phone_model"],
        artifact_sha256=phone["artifact_sha256"], layer_mask=sum(1 << layer for layer in phone["layers"]),
        n_embd=phone["n_embd"], columns=phone["columns"], column_quantum=phone["quantum"], max_tokens=4,
        swiglu=True, backend=config.get("phone_backend", "CPU"), phone_port=config["phone_port"],
        forward_port=forward_port, max_requests=config["output_tokens"] * 2 * len(phone["layers"]),
        worker_environment=config.get("phone_environment", {}), expected_sha256_by_path=pins,
        as_root=bool(config.get("phone_root", False)),
        phone_lock_path="/data/local/tmp/.s42-pixel-ffn-kernels.lock" if config.get("phone_root") else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=("adb-tcp", "aoa-bridge"), required=True)
    parser.add_argument("--aoa-bridge", type=Path, default=None)
    parser.add_argument("--forward-port", type=int, default=26991)
    args = parser.parse_args(argv)
    from research_dev.scheduler.adapters import HostEnergySampler, LlamaCppHttpClient, default_host_metric_callbacks
    from research_dev.scheduler.adapters.http_backend import LlamaCppCompletionPayload
    from research_dev.scheduler.campaigns.burstgpt.remote_resident_gate import _energy

    config = json.loads(args.config.read_text())
    root = args.output
    root.mkdir()
    (root / "qualify_pixel_server_transport.py").write_bytes(Path(__file__).read_bytes())
    save(root / "CONFIG.json", config)
    worker = worker_configuration(config, args.forward_port)
    bridge = None
    if args.transport == "aoa-bridge":
        if args.aoa_bridge is None:
            parser.error("--aoa-bridge is required for the aoa-bridge transport")
        bridge = AoaBridgeConfiguration.from_json(json.loads(args.aoa_bridge.read_text()))
        session = AoaBridgePhoneWorkerSession(worker, bridge)
    else:
        session = AdbTcpPhoneWorkerSession(worker)
    phone = config["phone"]
    server = None
    sampler = HostEnergySampler(default_host_metric_callbacks(), interval_s=0.1)
    records = []
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", config["server_port"]))
        preflight = session.preflight().to_json()
        save(root / "PHONE_PREFLIGHT.json", preflight)
        observed = preflight["observed_sha256_by_path"]
        phone_hashes = "".join(f"{value[7:]}  {path}\n" for path, value in sorted(observed.items()))
        identity = {"boot_id": preflight["boot_id"], "phone_hashes": phone_hashes,
                    "server_sha256": hex_digest(config["server"]), "harness_sha256": hex_digest(Path(__file__)),
                    "server_libraries": {str(p): hex_digest(p) for p in Path(config["server"]).parent.glob("*.so")},
                    "parent_stat": list(Path(config["model"]).stat()),
                    "parent_sha256_from_previous_verification": phone["artifact_sha256"],
                    "transport": args.transport}
        if bridge is not None:
            identity["aoa_bridge"] = {"configuration": bridge.to_json(), "options_sha256": bridge.options_sha256,
                                      "preflight": preflight.get("aoa_bridge")}
        save(root / "IDENTITY.json", identity)
        launch = session.start(root / "worker.log").to_json()
        save(root / "LAUNCH.json", launch)
        save(root / "WORKER_COMMAND.json", launch["command"])
        port = session.transport_contract().control_port
        env = {key: value for key, value in os.environ.items() if not key.startswith("S41_SERVER_FFN_")}
        env.update(config["server_environment"])
        env["S41_SERVER_FFN_PORT"] = str(port)
        command = [config["server"], *config["server_arguments"]]
        save(root / "SERVER_COMMAND.json", {"argv": command, "environment": config["server_environment"],
                                            "S41_SERVER_FFN_PORT": str(port), "transport": args.transport})
        with (root / "server.log").open("x") as server_log:
            server = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=server_log,
                                      stderr=subprocess.STDOUT)
        wait_ready(server, lambda: request_json(config["server_port"], "GET", "/health"))
        tokens = request_json(config["server_port"], "POST", "/tokenize",
                              {"content": config["prompt_text"], "add_special": True})["tokens"]
        tokens = tuple(tokens[:config["prompt_tokens"]])
        if len(tokens) != config["prompt_tokens"]:
            raise RuntimeError("prompt is too short")
        save(root / "PROMPT.json", list(tokens))
        endpoint = f"http://127.0.0.1:{config['server_port']}"
        client = LlamaCppHttpClient()
        sampler.start()
        wait_ready(server, lambda: len(sampler.rows()) >= 4, 30)   # as qualify_pixel_server.py: coverage before the first request
        for index, columns in enumerate((0, 8704, 17408, 0)):
            rid = f"pixel-ffn-{index}-columns{columns}"
            first, progress, controls = [], [], []

            def on_first(t_ns, rid=rid, columns=columns, index=index, first=first, controls=controls):
                first.append(t_ns)
                if columns:
                    policy = AdaptiveDecodePolicy(
                        route_id=rid, executor_id="pixel-transport-qualification",
                        operator_plan_sha256=canonical_sha256({"config": config, "columns": columns,
                                                               "transport": args.transport}),
                        desktop_parent_route_id="pixel-qualification-parent",
                        desktop_placement_sha256=canonical_sha256(command),
                        layer_indices=tuple(phone["layers"]), layer_mask=sum(1 << x for x in phone["layers"]),
                        columns=columns, split_fraction_ppm=columns * 1000000 // phone["columns"],
                        resource_ids=("desktop-cpu", "desktop-cuda", "pixel-cpu"))
                    control = AdaptiveDecodeControl(rid, 0, index + 1, policy)
                    ack, received = client.apply_ffn_control(endpoint, control, timeout_s=180)
                    controls.append({"ack": ack, "received_ns": received, "control": control.to_json()})

            payload = LlamaCppCompletionPayload(
                request_id=rid, expected_model_alias=config["alias"], input_tokens=len(tokens),
                output_tokens=config["output_tokens"], prompt_tokens=tokens, seed=17,
                stream_path=root / f"{rid}.raw", on_first_token=on_first,
                on_decode_progress=lambda *values, progress=progress: progress.append(values),
                quality_mode="accounting-only", timeout_s=600)
            started = time.monotonic_ns()
            result = client.complete(endpoint, payload, lambda: None)
            finished = time.monotonic_ns()
            if len(result["tokens"]) != config["output_tokens"] or not first:
                raise RuntimeError("incomplete generated tokens")
            row = {"request_id": rid, "columns": columns, "started_ns": started, "first_token_ns": first[0],
                   "finished_ns": finished, "request_s": (finished - started) / 1e9,
                   "decode_s": (finished - first[0]) / 1e9,
                   "request_host_energy": _energy(sampler, started, finished),
                   "decode_host_energy": _energy(sampler, first[0], finished),
                   "controls": controls, "execution": result}
            save(root / f"REQUEST-{index}.json", row)
            records.append(row)
            print("REQUEST_DONE", index, columns, row["request_s"], flush=True)
        server.send_signal(signal.SIGINT)
        server_exit = server.wait(timeout=60)
        if server_exit != 0:
            raise RuntimeError(f"server shutdown status {server_exit}")
        log = (root / "server.log").read_text(errors="replace").splitlines()
        proof_lines = [line for line in log if "S41SERVERFFNCALL " in line]
        shapes = [line for line in log if "S41SERVERFFNSHAPE " in line]
        calls = len(proof_lines)
        if not 0 < calls < worker.max_requests:
            raise RuntimeError(f"unexpected server call count {calls}")
        save(root / "SERVER_CALLS.json", proof_lines)
        save(root / "SERVER_FFN_SHAPES.json", shapes)
        stop = session.stop(served_calls=calls).to_json()
        save(root / "STOP.json", stop)
        expected = records[0]["execution"]["tokens"]
        matches = [row["execution"]["tokens"] == expected for row in records]
        summary = {"status": "PASS" if all(matches) else "FAIL", "token_identity": matches,
                   "output_tokens_each": len(expected), "phone_calls": calls,
                   "call_columns": {str(c): sum(bool(re.search(rf"\bcolumns={c}\b", line)) for line in proof_lines)
                                    for c in (8704, 17408)},
                   "server_exit": server_exit, "worker_exit": stop["exit_code"],
                   "drained_calls_outside_measurement": stop["drained_calls"], "transport": args.transport,
                   "boot_unchanged": stop["boot_unchanged"], "phone_energy_measured": False,
                   "requests": [{k: v for k, v in r.items() if k != "execution"} for r in records],
                   "finished_epoch_s": time.time()}
        save(root / "RESULT.json", summary)
        return 0 if all(matches) else 1
    except BaseException as error:
        save(root / "FAILURE.json", {"error": repr(error), "server_status": server.poll() if server else None,
                                     "session_active": session.active, "processes_not_killed": True,
                                     "finished_epoch_s": time.time()})
        raise
    finally:
        sampler.stop()
        save(root / "POWER_SAMPLES.json", sampler.rows())
        save(root / "POWER_DIAGNOSTICS.json", sampler.diagnostics())


if __name__ == "__main__":
    sys.exit(main())
