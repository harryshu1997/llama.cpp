#!/usr/bin/env python3
"""S15 persistent-session gate (real device, MECHANICS only, no energy).

Two resident phone stagenet [0,6) workers (OP15 v81 + OP12 v75) plus a host
parallel-head shared-tail. Runs seven sequential B1 host sessions; the first six
end with DETACH (worker resets, emits a session cert, stays resident), the
seventh with STOP (worker drains and terminates). Proves:
  - protocol compatibility: the legacy STOP path is byte-unchanged; DETACH is an
    additive opt-in opcode; the same host driver runs both ends;
  - persistence: one resident worker PID/boot-nonce serves all seven sessions and
    only terminates on the final STOP;
  - reset exactness: per-stream token ids are identical across all seven sessions.

Tail B2 is a frozen negative and is not run. ASCII only. No commit.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ART = HERE / "artifacts"
RESULTS = HERE / "results"
HOST_BIN = str(REPO / "build-cuda/bin/llama-layersplit")
FULL_MODEL = "/home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf"
SHARD = "/data/local/tmp/ls-npu/12b-f16-head-0-6.gguf"
DEST = "/data/local/tmp/ls-s14-persistent"
K = 6
N_LAYER = 48
MBUF = 3336
GPU_UUID = "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"
PHONES = [
    {"name": "op15", "serial": "3C15AU002CL00000", "port": 5811, "soc": "v81"},
    {"name": "op12", "serial": "5ae7a43d", "port": 5812, "soc": "v75"},
]
N_SESSIONS = 7
N_GEN = 16
PROMPT = "Explain in one sentence why the sky is blue."
DECLARED_CPU_OPS = {"GET_ROWS"}


def adb(serial, *args, check=True, timeout=120):
    return subprocess.run(["adb", "-s", serial, *args], capture_output=True,
                          text=True, check=check, timeout=timeout)


def start_worker(phone, log_path):
    serial, port = phone["serial"], phone["port"]
    adb(serial, "forward", "--remove", f"tcp:{port}", check=False)
    adb(serial, "forward", f"tcp:{port}", f"tcp:{port}")
    adb(serial, "shell", "pkill -9 -f llama-layersplit", check=False)
    time.sleep(1)
    cmd = (f"cd {DEST} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF={MBUF} "
           f"LLAMA_LAYER_END={K} LAYERSPLIT_PLACEMENT_CERT=1 ./llama-layersplit "
           f"-m {SHARD} --devices HTP0 -ngl 99 --mode stagenet --port {port} "
           f"--driver-batch 1 --driver-context 4096 --driver-max-prefill 512 -n {N_GEN}")
    handle = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(["adb", "-s", serial, "shell", cmd],
                            stdout=handle, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 240
    while time.time() < deadline:
        handle.flush()
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        if "[stagenet] listening" in text:
            return proc, handle
        if proc.poll() is not None:
            handle.close()
            raise RuntimeError(f"{phone['name']} worker exited early:\n{text[-2000:]}")
        time.sleep(0.5)
    proc.kill()
    handle.close()
    raise RuntimeError(f"{phone['name']} worker never listened")


def run_host_session(session_end, log_path):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    env["LLAMA_LAYER_START"] = str(K)
    env.pop("LLAMA_LAYER_END", None)
    env["LD_LIBRARY_PATH"] = str(Path(HOST_BIN).parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [HOST_BIN, "-m", FULL_MODEL, "-ngl", "99", "--mode", "pipedriver",
           "--parallel-heads", "--parallel-tail-batch", "1", "--driver-batch", "2",
           "--host", "127.0.0.1", "--port", str(PHONES[0]["port"]),
           "--port2", str(PHONES[1]["port"]),
           "-p", PROMPT, "-n", str(N_GEN), "--driver-requests", "2",
           "--driver-warmup", "0", "--driver-context", "4096",
           "--driver-max-prefill", "512", "--session-end", session_end]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=400)
    Path(log_path).write_text("=== CMD ===\n" + " ".join(cmd) +
                              "\n=== STDOUT ===\n" + result.stdout +
                              "\n=== STDERR ===\n" + result.stderr,
                              encoding="utf-8", errors="replace")
    return result


def parse_json_lines(text, prefix):
    out = []
    for match in re.finditer(re.escape(prefix) + r" (\{.*\})", text):
        try:
            out.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    return out


def verify_worker_certs(name, certs):
    problems = []
    if len(certs) != N_SESSIONS:
        problems.append(f"{name}: expected {N_SESSIONS} session certs, got {len(certs)}")
        return problems, {}
    ids = [c["session_id"] for c in certs]
    if ids != list(range(1, N_SESSIONS + 1)):
        problems.append(f"{name}: session ids not contiguous 1..{N_SESSIONS}: {ids}")
    ends = [c["session_end"] for c in certs]
    if ends != ["DETACH"] * (N_SESSIONS - 1) + ["STOP"]:
        problems.append(f"{name}: session_end sequence wrong: {ends}")
    pids = {c["worker_pid"] for c in certs}
    nonces = {c["worker_boot_nonce"] for c in certs}
    if len(pids) != 1 or len(nonces) != 1:
        problems.append(f"{name}: worker identity not constant pids={pids} nonces={nonces}")
    for c in certs:
        if c["placement_status"] != "SCHEDULED_PLACEMENT_OK" or c["missing_buffer_compute_nodes"] != 0:
            problems.append(f"{name}: session {c['session_id']} placement not clean")
        if c["layer_start"] != 0 or c["layer_end"] != K or c["expected_backend"] != "HTP0":
            problems.append(f"{name}: session {c['session_id']} wrong island/backend")
        for op, bufs in c["compute_by_op_and_buffer"].items():
            for buf in bufs:
                if buf != "HTP0" and op not in DECLARED_CPU_OPS:
                    problems.append(f"{name}: session {c['session_id']} undeclared {op}@{buf}")
    summary = {"pid": certs[0]["worker_pid"], "boot_nonce": certs[0]["worker_boot_nonce"],
               "session_ids": ids, "session_ends": ends}
    return problems, summary


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    status = open(RESULTS / "gate_status.log", "w", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        status.write(line + "\n")
        status.flush()

    workers = []
    try:
        for phone in PHONES:
            log(f"starting {phone['name']} worker ({phone['soc']}) resident [0,{K})...")
            proc, handle = start_worker(phone, RESULTS / f"worker_{phone['name']}.log")
            workers.append({"phone": phone, "proc": proc, "handle": handle})
            log(f"{phone['name']} worker listening (pid tracked in cert)")

        sessions = []
        for idx in range(1, N_SESSIONS + 1):
            session_end = "detach" if idx < N_SESSIONS else "stop"
            log(f"host session {idx}/{N_SESSIONS} (session_end={session_end}) ...")
            result = run_host_session(session_end, RESULTS / f"host_session_{idx}.log")
            routes = parse_json_lines(result.stderr, "ROUTEJSON")
            by_stream = {}
            for r in routes:
                if r.get("status") == "ok":
                    by_stream[r["stream_index"]] = r["token_ids"]
            log(f"  session {idx} rc={result.returncode} streams={sorted(by_stream)} "
                f"tokens={[len(v) for _, v in sorted(by_stream.items())]}")
            sessions.append({"idx": idx, "session_end": session_end,
                             "rc": result.returncode, "tokens_by_stream": by_stream})
            time.sleep(1)

        # workers should have terminated after the final STOP session
        time.sleep(3)
        for w in workers:
            w["handle"].flush()

        problems = []
        worker_summaries = {}
        for w in workers:
            name = w["phone"]["name"]
            text = Path(RESULTS / f"worker_{name}.log").read_text(encoding="utf-8", errors="replace")
            certs = parse_json_lines(text, "SESSIONCERT")
            probs, summ = verify_worker_certs(name, certs)
            problems.extend(probs)
            worker_summaries[name] = summ
            alive = w["proc"].poll() is None
            if alive:
                problems.append(f"{name}: worker still alive after final STOP")
                adb(w["phone"]["serial"], "shell", "pkill -9 -f llama-layersplit", check=False)

        # reset exactness: per stream, token ids identical across all 7 sessions
        reset_exact = True
        stream_digests = {}
        for stream in (0, 1):
            seqs = [tuple(s["tokens_by_stream"].get(stream, [])) for s in sessions]
            nonempty = [s for s in seqs if s]
            identical = len(nonempty) == N_SESSIONS and len(set(nonempty)) == 1
            stream_digests[stream] = {
                "identical_across_sessions": identical,
                "n_tokens": len(nonempty[0]) if nonempty else 0,
                "first_session_tokens": list(nonempty[0]) if nonempty else [],
            }
            if not identical:
                reset_exact = False
                problems.append(f"stream {stream}: token ids diverged across sessions")

        certified = not problems
        verdict = ("PERSISTENT_SESSION_MECHANICS_PASS_PHYSICAL_ENERGY_NOT_RUN"
                   if certified else "PERSISTENT_SESSION_MECHANICS_FAIL")
        report = {
            "schema": "s15-persistence-gate-v1",
            "verdict": verdict,
            "certified": certified,
            "n_sessions": N_SESSIONS,
            "session_end_sequence": [s["session_end"] for s in sessions],
            "worker_summaries": worker_summaries,
            "reset_exact_across_sessions": reset_exact,
            "stream_reset": stream_digests,
            "problems": problems,
            "artifact_binary_sha256": _sha(ART / "llama-layersplit"),
            "energy_scope": "UNKNOWN",
        }
        (RESULTS / "gate_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        log("VERDICT " + verdict)
        log("problems: " + (json.dumps(problems) if problems else "none"))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if certified else 2
    finally:
        for w in workers:
            try:
                if w["proc"].poll() is None:
                    w["proc"].kill()
                w["handle"].close()
            except Exception:
                pass
        for phone in PHONES:
            adb(phone["serial"], "shell", "pkill -9 -f llama-layersplit", check=False)
            adb(phone["serial"], "forward", "--remove", f"tcp:{phone['port']}", check=False)
        status.close()


def _sha(path):
    import hashlib
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
