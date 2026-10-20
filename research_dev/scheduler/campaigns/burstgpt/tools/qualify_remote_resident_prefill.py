#!/usr/bin/env python3
"""Phone prefill for remote-resident FFN layers: desktop-only server versus the same server with a phone owning
the dense FFN weights of some layers (the desktop never loads them; the phone runs them in prefill AND decode).

    cd <repo root> && python3 -m research_dev.scheduler.campaigns.burstgpt.tools.qualify_remote_resident_prefill \\
        --config SERVER_TEST_op15.json --output OUT --forward-port 27314 --prompt-lengths 32,512,2048

The config is a ``qualify_pixel_server_transport`` config (server binary and arguments, worker pins, phone layers;
2026-10-08: Qwen3-32B Q4_0 with OnePlus 15 HTP layers 0-10 over adb-tcp). Both arms serve the same prompts (one
per length, distinct prefixes, cache_prompt off, temperature 0) after one short warm-up request, and record:

* prefill time (server ``prompt_ms``), time to first token, decode time and host energy per request;
* memory after the requests: process RSS, resident and mapped bytes of the model file, the loader and server
  omission lines (``REMOTE_RESIDENT_FFN`` / ``S41SERVERFFN remote_resident``) and the CPU buffer line;
* generated tokens; token identity against the desktop arm is reported, not required (HTP Q4_0 is approximate).

The remote arm sets ``S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK`` = the phone layers with ``MAX_TOKENS`` = the
ubatch, full columns and no runtime control, so every ubatch row of those layers runs on the phone. adb-tcp
is a test transport: its per-call cost is not the FunctionFS cost, which is why worker compute is recorded too.

``--co-helper CONFIG`` adds the arm ``both``: the same remote-resident owner plus a decode co-helper (e.g. the Pixel
on layers 11-20, max_tokens 4) in the server's multi-helper format with runtime control; each request's decode
policy (co-helper layers, full columns) is applied at its first token, so prefill of those layers stays on the
desktop (needs the per-helper MAX_TOKENS server of 2026-10-08).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from research_dev.scheduler.campaigns.burstgpt.tools.qualify_pixel_server_transport import (
    hex_digest, request_json, save, wait_ready, worker_configuration,
)
from research_dev.scheduler.adapters.phone_tcp_session import AdbTcpPhoneWorkerSession

DROPPED_ENVIRONMENT = ("S41_SERVER_FFN_DORMANT_", "S41_SERVER_FFN_RUNTIME_CONTROL")


def remote_environment(config: dict, mask: int, ubatch: int) -> dict[str, str]:
    """The server FFN environment of the config with remote-resident layers (no dormant share, no control)."""
    env = {key: value for key, value in config["server_environment"].items()
           if not key.startswith(DROPPED_ENVIRONMENT)}
    env.update({"S41_SERVER_FFN_LAYER_MASK": str(mask), "S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK": str(mask),
                "S41_SERVER_FFN_MAX_TOKENS": str(ubatch), "S41_SERVER_FFN_COLUMNS": str(config["phone"]["columns"])})
    return env


def both_environment(config: dict, co_config: dict, mask: int, co_mask: int, ubatch: int,
                     port: int, co_port: int) -> dict[str, str]:
    """Multi-helper environment: helper0 owns the remote-resident layers (whole ubatches), helper1 decodes."""
    env = {key: value for key, value in remote_environment(config, mask, ubatch).items()
           if key not in ("S41_SERVER_FFN_HOST", "S41_SERVER_FFN_TRANSPORT", "S41_SERVER_FFN_PORT")}
    env.update({"S41_SERVER_FFN_LAYER_MASK": str(mask | co_mask), "S41_SERVER_FFN_RUNTIME_CONTROL": "1",
                "S41_SERVER_FFN_HELPERS": "2"})
    for index, (label, layers, helper_port, tokens) in enumerate(
            (("owner", mask, port, ubatch), ("cohelper", co_mask, co_port, co_config.get("co_helper_max_tokens", 4)))):
        prefix = f"S41_SERVER_FFN_HELPER{index}_"
        env.update({prefix + "LABEL": label, prefix + "LAYER_MASK": str(layers), prefix + "TRANSPORT": "tcp",
                    prefix + "HOST": "127.0.0.1", prefix + "PORT": str(helper_port), prefix + "MAX_TOKENS": str(tokens)})
    return env


def ubatch_size(arguments: list[str]) -> int:
    for flag in ("-ub", "--ubatch-size"):
        if flag in arguments:
            return int(arguments[arguments.index(flag) + 1])
    return 512


def memory_record(pid: int, model: Path, log_path: Path) -> dict[str, object]:
    from research_dev.scheduler.campaigns.burstgpt.remote_resident_gate import _host_memory, _smaps_for_file, _vm_hwm, _vm_rss
    lines = log_path.read_text(errors="replace").splitlines()
    return {"vm_rss_bytes": _vm_rss(pid), "vm_hwm_bytes": _vm_hwm(pid), "host_memory": _host_memory(),
            "model_file": _smaps_for_file(pid, model),
            "loader_lines": [line for line in lines if "REMOTE_RESIDENT_FFN" in line or "model buffer size" in line
                             or "S41SERVERFFN remote_resident" in line or "unmap_remote_resident" in line]}


def prompts(port: int, config: dict, lengths: list[int]) -> dict[int, tuple[int, ...]]:
    """One token list per length; each starts at a different offset of the repeated text (no shared prefix)."""
    text = config["prompt_text"]
    base = request_json(port, "POST", "/tokenize", {"content": text, "add_special": False})["tokens"]
    bos = request_json(port, "POST", "/tokenize", {"content": "", "add_special": True})["tokens"]
    rows = {}
    for index, length in enumerate(lengths):
        body = base[7 * index + 1:]
        while len(body) < length:
            body = body + base
        rows[length] = tuple(bos + body[:length - len(bos)])
    return rows


def run_arm(name: str, config: dict, args, environment: dict[str, str], prompt_rows, sampler,
            control_layers: tuple[int, ...] = (), control_columns: int = 0) -> dict[str, object]:
    from research_dev.scheduler.adapters import LlamaCppHttpClient
    from research_dev.scheduler.adapters.http_backend import LlamaCppCompletionPayload
    from research_dev.scheduler.campaigns.burstgpt.remote_resident_gate import _energy

    root = args.output / name
    root.mkdir()
    port = config["server_port"]
    env = {key: value for key, value in os.environ.items() if not key.startswith("S41_SERVER_FFN_")}
    env.update(environment)
    command = [config["server"], *config["server_arguments"]]
    save(root / "SERVER_COMMAND.json", {"argv": command, "environment": environment})
    log_path = root / "server.log"
    with log_path.open("x") as log:
        server = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    try:
        started = time.monotonic()
        wait_ready(server, lambda: request_json(port, "GET", "/health"), 900)
        load_s = time.monotonic() - started
        rows = prompt_rows if prompt_rows is not None else prompts(port, config, args.prompt_lengths)
        endpoint = f"http://127.0.0.1:{port}"
        client = LlamaCppHttpClient()
        records = []
        plan = [("warmup", rows[min(rows)], 4)] + [(f"len{length}", rows[length], args.output_tokens)
                                                   for length in sorted(rows)]
        for generation, (label, tokens, outputs) in enumerate(plan, start=1):
            first, controls = [], []

            def on_first(t_ns, rid=f"{name}-{label}", generation=generation, first=first, controls=controls):
                first.append(t_ns)
                if control_layers:
                    from research_dev.scheduler._internal.adaptive_decode_contracts import (
                        AdaptiveDecodeControl, AdaptiveDecodePolicy)
                    from research_dev.scheduler._internal.types import canonical_sha256
                    mask = sum(1 << layer for layer in control_layers)
                    policy = AdaptiveDecodePolicy(
                        route_id=rid, executor_id="remote-resident-prefill-qualification",
                        operator_plan_sha256=canonical_sha256({"environment": environment, "label": rid}),
                        desktop_parent_route_id="remote-resident-qualification-parent",
                        desktop_placement_sha256=canonical_sha256(command), layer_indices=tuple(control_layers),
                        layer_mask=mask, columns=control_columns, split_fraction_ppm=1000000,
                        resource_ids=("desktop-cpu", "desktop-cuda", "co-helper"))
                    control = AdaptiveDecodeControl(rid, 0, generation, policy)
                    ack, received = client.apply_ffn_control(endpoint, control, timeout_s=180)
                    controls.append({"ack": ack, "received_ns": received})

            payload = LlamaCppCompletionPayload(
                request_id=f"{name}-{label}", expected_model_alias=config["alias"], input_tokens=len(tokens),
                output_tokens=outputs, prompt_tokens=tokens, seed=17, stream_path=root / f"{label}.raw",
                on_first_token=on_first, quality_mode="accounting-only", timeout_s=1800)
            begin = time.monotonic_ns()
            result = client.complete(endpoint, payload, lambda: None)
            end = time.monotonic_ns()
            if len(result["tokens"]) != outputs or not first:
                raise RuntimeError(f"{name} {label}: incomplete generated tokens")
            row = {"label": label, "input_tokens": len(tokens), "output_tokens": outputs,
                   "prompt_ms": result.get("prompt_ms"), "predicted_ms": result.get("predicted_ms"),
                   "ttft_s": (first[0] - begin) / 1e9, "request_s": (end - begin) / 1e9,
                   "decode_s": (end - first[0]) / 1e9, "tokens": result["tokens"], "controls": controls,
                   "prefill_host_energy": _energy(sampler, begin, first[0]),
                   "request_host_energy": _energy(sampler, begin, end)}
            records.append(row)
            print(name, label, len(tokens), "ttft", round(row["ttft_s"], 2), "prompt_ms", row["prompt_ms"],
                  "decode", round(row["decode_s"], 2), flush=True)
        memory = memory_record(server.pid, Path(config["model"]), log_path)
        server.send_signal(signal.SIGINT)
        exit_code = server.wait(timeout=120)
        lines = log_path.read_text(errors="replace").splitlines()
        result = {"arm": name, "load_s": load_s, "server_exit": exit_code, "memory": memory, "requests": records,
                  "phone_calls": sum("S41SERVERFFNCALL " in line for line in lines),
                  "helper_lines": [line for line in lines if "S41SERVERFFNHELPER " in line],
                  "summary_lines": [line[:400] for line in lines if 'S41SERVERFFN {"status"' in line],
                  "phone_call_tokens": sorted({int(part.split("=")[1]) for line in lines
                                               if "S41SERVERFFNCALL " in line
                                               for part in line.split() if part.startswith("tokens=")})}
        save(root / "ARM.json", result)
        return result | {"prompt_rows": rows}
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forward-port", type=int, required=True)
    parser.add_argument("--prompt-lengths", type=lambda text: [int(v) for v in text.split(",")], default=[32, 512, 2048])
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--arms", default="desktop,remote")
    parser.add_argument("--co-helper", type=Path, default=None, help="qualify config of the decode co-helper")
    parser.add_argument("--co-helper-forward-port", type=int, default=0)
    parser.add_argument("--co-helper-no-control", action="store_true",
                        help="never apply the co-helper decode policy (its layers stay on the desktop)")
    args = parser.parse_args(argv)
    from research_dev.scheduler.adapters import HostEnergySampler, default_host_metric_callbacks

    config = json.loads(args.config.read_text())
    args.output.mkdir()
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(args.output / "CONFIG.json", config)
    phone = config["phone"]
    mask = sum(1 << layer for layer in phone["layers"])
    ubatch = ubatch_size(config["server_arguments"])
    arms = args.arms.split(",")
    lengths = sorted(args.prompt_lengths)
    calls_per_layer = sum(math.ceil(n / ubatch) for n in [lengths[0]] + lengths) + 4 + args.output_tokens * len(lengths)
    worker = worker_configuration(config, args.forward_port)
    worker = replace(worker, max_tokens=ubatch, max_requests=(calls_per_layer + 16) * len(phone["layers"]) * 2)
    session = AdbTcpPhoneWorkerSession(worker)
    sampler = HostEnergySampler(default_host_metric_callbacks(), interval_s=0.1)
    results = {}
    prompt_rows = None
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", config["server_port"]))
        save(args.output / "IDENTITY.json", {"server_sha256": hex_digest(config["server"]),
                                             "server_libraries": {str(p): hex_digest(p) for p in
                                                                  Path(config["server"]).parent.glob("*.so")},
                                             "harness_sha256": hex_digest(Path(__file__)),
                                             "parent_sha256": phone["artifact_sha256"], "remote_mask": mask,
                                             "ubatch": ubatch, "worker_max_requests": worker.max_requests})
        sampler.start()
        time.sleep(1.0)
        if "desktop" in arms:
            results["desktop"] = run_arm("desktop", config, args, {}, None, sampler)
            prompt_rows = results["desktop"]["prompt_rows"]
        if "remote" in arms:
            save(args.output / "PHONE_PREFLIGHT.json", session.preflight().to_json())
            launch = session.start(args.output / "worker.log").to_json()
            save(args.output / "LAUNCH.json", launch)
            env = remote_environment(config, mask, ubatch)
            env["S41_SERVER_FFN_PORT"] = str(session.transport_contract().control_port)
            results["remote"] = run_arm("remote", config, args, env, prompt_rows, sampler)
            stop = session.stop(served_calls=results["remote"]["phone_calls"]).to_json()
            save(args.output / "STOP.json", stop)
        if "both" in arms:
            co_config = json.loads(args.co_helper.read_text())
            co_layers = tuple(co_config["phone"]["layers"])
            co_mask = sum(1 << layer for layer in co_layers)
            if co_mask & mask:
                raise RuntimeError("co-helper layers overlap the remote-resident layers")
            owner = AdbTcpPhoneWorkerSession(worker)
            co_worker = worker_configuration(co_config, args.co_helper_forward_port)
            co_tokens = int(co_config.get("co_helper_max_tokens", 4))
            co_worker = replace(co_worker, max_tokens=co_tokens,
                                max_requests=(args.output_tokens * len(lengths) + 8) * len(co_layers) * 2)
            co_session = AdbTcpPhoneWorkerSession(co_worker)
            save(args.output / "PHONE_PREFLIGHT_BOTH.json", {"owner": owner.preflight().to_json(),
                                                             "co_helper": co_session.preflight().to_json()})
            save(args.output / "LAUNCH_BOTH.json", {"owner": owner.start(args.output / "worker-both.log").to_json(),
                                                    "co_helper": co_session.start(args.output / "co-worker.log").to_json()})
            env = both_environment(config, co_config, mask, co_mask, ubatch,
                                   owner.transport_contract().control_port, co_session.transport_contract().control_port)
            results["both"] = run_arm("both", config, args, env, prompt_rows, sampler,
                                      () if args.co_helper_no_control else co_layers,
                                      int(co_config["phone"]["columns"]))
            lines = (args.output / "both" / "server.log").read_text(errors="replace").splitlines()
            owner_calls = sum(1 for line in lines if "S41SERVERFFNCALL " in line
                              and int(line.split(" layer=")[1].split()[0]) in phone["layers"])
            co_calls = sum(1 for line in lines if "S41SERVERFFNCALL " in line
                           and int(line.split(" layer=")[1].split()[0]) in co_layers)
            results["both"]["owner_calls"], results["both"]["co_helper_calls"] = owner_calls, co_calls
            save(args.output / "STOP_BOTH.json", {"owner": owner.stop(served_calls=owner_calls).to_json(),
                                                  "co_helper": co_session.stop(served_calls=co_calls).to_json()})
        summary = {"arms": {name: {k: v for k, v in result.items() if k != "prompt_rows"}
                            for name, result in results.items()}}
        for arm in ("remote", "both"):
            if "desktop" in results and arm in results:
                pairs = zip(results["desktop"]["requests"], results[arm]["requests"])
                key = "" if arm == "remote" else "_" + arm
                summary["token_identity" + key] = {a["label"]: a["tokens"] == b["tokens"] for a, b in pairs}
                summary["model_file_rss_freed_bytes" + key] = (
                    results["desktop"]["memory"]["model_file"]["rss_bytes"]
                    - results[arm]["memory"]["model_file"]["rss_bytes"])
                summary["vm_rss_freed_bytes" + key] = (results["desktop"]["memory"]["vm_rss_bytes"]
                                                       - results[arm]["memory"]["vm_rss_bytes"])
        summary["finished_epoch_s"] = time.time()
        save(args.output / "RESULT.json", summary)
        return 0
    except BaseException as error:
        save(args.output / "FAILURE.json", {"error": repr(error), "session_active": session.active,
                                            "finished_epoch_s": time.time()})
        raise
    finally:
        sampler.stop()
        save(args.output / "POWER_SAMPLES.json", sampler.rows())


if __name__ == "__main__":
    sys.exit(main())
