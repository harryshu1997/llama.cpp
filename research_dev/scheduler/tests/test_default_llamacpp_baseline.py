#!/usr/bin/env python3
"""WS7 out-of-the-box llama.cpp baseline harness (tools/default_llamacpp_baseline.py): the replay plan equals the
campaign's work, the request body equals the campaign client's body plus the router's ``model`` field, the client
release gate, log parsing of the fitted placement, host energy with the campaign's integrators, the RESULT subset
read unchanged by fleet_energy.py / latency_report.py, an end-to-end run against a fake router binary, and the
desktop launcher (dry run)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from research_dev.scheduler.adapters import http_backend
from research_dev.scheduler.adapters.host_runtime import server_energy_summary
from research_dev.scheduler.campaigns.burstgpt.common import canonical
from research_dev.scheduler.campaigns.burstgpt.tools import default_llamacpp_baseline as baseline
from research_dev.scheduler.campaigns.burstgpt.tools import fleet_energy, latency_report

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "research_dev/scheduler/campaigns/burstgpt/paper_config_v1"
TRACE = PAPER / "trace/longtail_eval_v2"
LAUNCHER = REPO / "research_dev/scheduler/campaigns/burstgpt/tools/rig/launch_default_baseline.sh"


def paper_plan() -> dict:
    return baseline.plan_from_trace(TRACE / "REQUESTS_SEMANTIC_SOURCE.jsonl", TRACE / "REQUESTS_OVERLAY.jsonl",
                                    TRACE / "TRACE_MANIFEST.json", TRACE / "burstgpt_longtail_eval_v2.json",
                                    json.loads((PAPER / "template/models.json").read_text()))


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class PlanTest(unittest.TestCase):
    def test_paper_plan_is_the_campaign_replay(self):
        plan = paper_plan()
        baseline.validate_plan(plan)
        rows = plan["requests"]
        self.assertEqual(len(rows), 14)
        order = "".join(row["model_id"][0] for row in rows)  # g(emma) q(wen) l(lama) in combined order
        self.assertEqual(order, "gqgqqqgqggglqq")
        self.assertEqual([row["seed"] for row in rows], list(range(14)))
        self.assertEqual([row["combined_request_index"] for row in rows], list(range(14)))
        self.assertEqual(rows[0]["replay_arrival_us"], 1_000_000)
        self.assertEqual(rows[-1]["replay_arrival_us"], 1_676_000_000)
        self.assertEqual(sum(row["output_tokens"] for row in rows), 3604)
        self.assertEqual(sum(row["input_tokens"] for row in rows), 5385)
        self.assertEqual(rows[11]["request_id"], "burstgpt_longtail_eval_v2:llama-3.2-1b-overlay:00")
        for row in rows:
            self.assertEqual(row["prompt_sha256"],
                             "sha256:" + hashlib.sha256(canonical(row["prompt_tokens"])).hexdigest())
        # prompt hash of request 000 as recorded by the reference run s2a (RESULT.request_results[0].prompt_sha256)
        self.assertEqual(rows[0]["prompt_sha256"],
                         "sha256:4b43ec5599c95f06abfdda1974d667c9b974b76ca7e04c3aef3a6f99fa6e310b")
        self.assertEqual({row["model_id"]: row["artifact_bytes"] for row in plan["models"]},
                         {"qwen3-14b-q4km-dequant-f16": 29543423360, "gemma-4-12b-q40-dequant-f16": 23832065056,
                          "llama-3.2-1b-instruct-q4_0": 770928288})

    def test_smoke_plan_keeps_order_and_uses_one_small_model(self):
        plan = baseline.smoke_plan(paper_plan(), "/models/small.gguf", time_scale=0.01, max_prompt_tokens=8,
                                   max_output_tokens=5)
        baseline.validate_plan(plan)
        self.assertEqual({row["path"] for row in plan["models"]}, {"/models/small.gguf"})
        self.assertEqual([row["model_id"] for row in plan["requests"]][:3], ["smoke-cold", "smoke-hot", "smoke-cold"])
        self.assertTrue(all(row["input_tokens"] <= 8 and row["output_tokens"] <= 5 for row in plan["requests"]))
        self.assertEqual(plan["requests"][-1]["replay_arrival_us"], 1_000_000 + 16_750_000)
        # every smoke prompt is a Qwen-tokenized prompt (Qwen3 vocabulary: <|im_start|> = 151644)
        self.assertTrue(all(row["prompt_tokens"][0] == 151644 for row in plan["requests"]))

    def test_request_body_is_the_campaign_body_plus_model(self):
        request = paper_plan()["requests"][1]
        captured = {}

        class FakeConnection:
            def __init__(self, host, port, timeout):
                captured["address"] = (host, port)

            def connect(self):
                pass

            def request(self, method, path, body, headers):
                captured.update(method=method, path=path, body=body, headers=headers)

            def getresponse(self):
                raise OSError("captured")

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(http_backend.http.client, "HTTPConnection", FakeConnection):
            payload = http_backend.LlamaCppCompletionPayload(
                request_id=request["request_id"], expected_model_alias="alias", input_tokens=request["input_tokens"],
                output_tokens=request["output_tokens"], prompt_tokens=tuple(request["prompt_tokens"]),
                seed=request["seed"], stream_path=Path(root) / "s.raw", on_first_token=lambda _ns: None)
            with self.assertRaises(OSError):
                http_backend.LlamaCppHttpClient().complete("http://127.0.0.1:1", payload, lambda: None)
        ours = baseline.completion_body(request, request["model_id"])
        self.assertEqual(ours.pop("model"), request["model_id"])
        self.assertEqual(canonical(ours), captured["body"])  # byte-identical campaign body
        self.assertEqual((captured["method"], captured["path"]), ("POST", "/completion"))
        self.assertEqual(captured["headers"]["X-Scheduler-Request-ID"], request["request_id"])

    def test_preset_names_only_model_paths(self):
        text = baseline.preset_text([{"model_id": "a-1", "path": "/m/a.gguf"}, {"model_id": "b.2", "path": "/m/b"}])
        options = [line for line in text.splitlines() if "=" in line and not line.startswith(";")]
        self.assertEqual(options, ["model = /m/a.gguf", "model = /m/b"])
        self.assertIn("[a-1]", text)
        self.assertNotIn("version", text)  # a top-level key would create a nameless preset


class GateTest(unittest.TestCase):
    @staticmethod
    def req(name, model):
        return {"request_id": name, "model_id": model}

    def test_fifo_drain_holds_other_models_and_keeps_order(self):
        gate = baseline.SwitchGate("fifo-drain")

        def ids(released):
            return [row["request_id"] for row in released]

        a, b, c, d, e = (self.req("a", "gemma"), self.req("b", "qwen"), self.req("c", "gemma"),
                         self.req("d", "qwen"), self.req("e", "qwen"))
        self.assertEqual(ids(gate.arrive(a)), ["a"])
        self.assertEqual(ids(gate.arrive(b)), [])          # other model: waits for a
        self.assertEqual(ids(gate.arrive(c)), [])          # same model as a, but behind b (strict FIFO)
        self.assertEqual(ids(gate.complete(a)), ["b"])     # drained -> switch to qwen
        self.assertEqual(ids(gate.arrive(d)), [])          # behind c
        self.assertEqual(ids(gate.complete(b)), ["c"])     # c (gemma) next; d waits for c
        self.assertEqual(ids(gate.complete(c)), ["d"])
        self.assertEqual(ids(gate.arrive(e)), ["e"])       # joins d (same model, head of queue)
        self.assertEqual(gate.in_flight, 2)

    def test_none_releases_every_arrival(self):
        gate = baseline.SwitchGate("none")
        self.assertEqual(len(gate.arrive(self.req("a", "x"))) + len(gate.arrive(self.req("b", "y"))), 2)
        with self.assertRaises(ValueError):
            baseline.SwitchGate("affinity")


ROUTER_LOG = """\
0.00.208.866 I srv   load_models: Loaded 3 custom model presets from preset.ini
0.00.210.627 I srv  llama_server: listening on http://127.0.0.1:18600
0.01.000.000 I srv  ensure_model: model name=gemma is not loaded, loading...
0.01.000.100 I srv          load: spawning server instance with name=gemma on port 41000
0.01.000.110 I srv          load: spawning server instance with args:
0.01.000.111 I srv          load:   /bin/llama-server
0.01.000.112 I srv          load:   --alias
0.01.000.113 I srv          load:   gemma
0.01.000.200 I srv  ensure_model: waiting until model name=gemma is fully loaded...
[41000] 0.00.100.000 I cmn  common_param:   - CUDA0   : NVIDIA GeForce RTX 4060 Ti (16380 MiB, 15890 MiB free)
[41000] 0.00.110.000 I cmn  common_param: system_info: n_threads = 8 (n_threads_batch = 8) / 24 | CUDA : ARCHS = 890
[41000] 0.00.120.000 I srv  llama_server: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true
[41000] 0.00.130.000 I common_params_fit_impl: projected to use 26000 MiB of device memory vs. 15890 MiB of free device memory
[41000] 0.00.140.000 I common_params_fit_impl: context size reduced from 262144 to 4096 -> need 9000 MiB less memory in total
[41000] 0.00.150.000 I common_fit_params: successfully fit params to free device memory
[41000] 0.00.160.000 I print_info: n_ctx_train           = 262144
[41000] 0.00.161.000 I print_info: n_layer               = 48
[41000] 0.00.170.000 I load_tensors: offloaded 25/49 layers to GPU
[41000] 0.00.171.000 I load_tensors:   CPU_Mapped model buffer size = 11000.00 MiB
[41000] 0.00.172.000 I load_tensors:        CUDA0 model buffer size = 13000.00 MiB
[41000] 0.00.180.000 I llama_context: n_seq_max     = 4
[41000] 0.00.180.100 I llama_context: n_ctx         = 4096
[41000] 0.00.180.200 I llama_context: n_ctx_seq     = 4096
[41000] 0.00.180.300 I llama_context: n_batch       = 2048
[41000] 0.00.180.400 I llama_context: n_ubatch      = 512
[41000] 0.00.180.500 I llama_context: flash_attn    = auto
[41000] 0.00.180.600 I llama_context: kv_unified    = true
[41000] 0.00.181.000 I llama_context: Flash Attention was auto, set to enabled
[41000] 0.00.190.000 I llama_kv_cache:      CUDA0 KV buffer size =   400.00 MiB
[41000] 0.00.191.000 I sched_reserve:      CUDA0 compute buffer size =   900.00 MiB
[41000] 0.00.200.000 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 4096, kv_unified = 'true'
[41000] cmd_child_to_router:state:{"state":"ready","payload":{"id":"gemma","meta":{"n_ctx":4096,"n_ctx_train":262144}}}
0.40.000.000 I srv    unload_lru: models_max limit reached, removing LRU name=gemma
0.40.000.001 I srv        unload: stopping model instance name=gemma
0.40.300.000 I srv    operator(): instance name=gemma exited with status 0
0.40.300.100 I srv          load: spawning server instance with name=qwen on port 41001
[41001] 0.00.170.000 I load_tensors: offloaded 18/41 layers to GPU
[41001] cmd_child_to_router:state:{"state":"ready","payload":{"id":"qwen","meta":{"n_ctx":4096}}}
"""


def log_lines(text: str, epoch_ns: int, step_ns: int = 1_000_000) -> list[baseline.LogLine]:
    """One line per millisecond after the epoch, except lines stamped 0.40.* (40 s later)."""
    out = []
    for index, line in enumerate(text.splitlines()):
        offset = 40_000_000_000 if line.startswith("0.40.") or line.startswith("[41001]") else 0
        out.append(baseline.LogLine(epoch_ns + offset + index * step_ns, line))
    return out


class LogParseTest(unittest.TestCase):
    def test_instances_lifecycle_and_fitted_placement(self):
        epoch = 10**12
        instances = baseline.parse_instances(log_lines(ROUTER_LOG, epoch), epoch)
        self.assertEqual([row["model_id"] for row in instances], ["gemma", "qwen"])
        gemma, qwen = instances
        self.assertEqual(gemma["args"], ["/bin/llama-server", "--alias", "gemma"])
        self.assertEqual(gemma["exit_status"], 0)
        self.assertIsNotNone(gemma["evicted_lru_us"])
        self.assertGreater(gemma["ready_us"], gemma["spawn_us"])
        self.assertEqual(gemma["ready_info"]["n_ctx"], 4096)
        placement = gemma["placement"]
        self.assertEqual((placement["gpu_layers"], placement["gpu_layers_of"]), (25, 49))
        self.assertEqual((placement["n_ctx"], placement["n_ctx_seq"], placement["n_seq_max"]), (4096, 4096, 4))
        self.assertEqual((placement["n_slots"], placement["n_ctx_slot"], placement["slots_kv_unified"]),
                         (4, 4096, True))
        self.assertEqual((placement["n_threads"], placement["n_threads_batch"]), (8, 8))
        self.assertEqual(placement["fit_status"], "fitted")
        self.assertEqual((placement["fit_context_reduced_from"], placement["fit_context_reduced_to"]), (262144, 4096))
        self.assertEqual(placement["flash_attn_resolved"], "enabled")
        self.assertEqual(placement["buffers_mib"]["model"], {"CPU_Mapped": 11000.0, "CUDA0": 13000.0})
        self.assertEqual(placement["devices_at_start"][0]["free_mib"], 15890)
        self.assertEqual(placement["n_parallel_auto"], 4)
        self.assertEqual(qwen["placement"]["gpu_layers"], 18)
        self.assertIsNone(qwen["exit_us"])
        self.assertIs(baseline.serving_instance(instances, "qwen", qwen["spawn_us"] + 5), qwen)
        self.assertIsNone(baseline.serving_instance(instances, "qwen", qwen["spawn_us"] - 5))
        by_model = baseline.placements_by_model(instances)
        self.assertEqual(by_model["gemma"][0]["instances"], [0])
        self.assertEqual(by_model["gemma"][0]["gpu_layers"], 25)


def samples(start_ns, end_ns, *, period_ns=200_000_000, gpu_w=50.0, cpu_w=30.0, rapl=True, gap=None,
            wrap_uj=None):
    rows = []
    t = start_ns
    energy = 0
    maximum = wrap_uj or 262_143_328_850
    while t <= end_ns:
        if gap is None or not gap[0] < t < gap[1]:
            row = {"gpu": {"power_mw": int(gpu_w * 1000), "sample_t_ns": t}, "t_ns": t}
            row["rapl_package"] = ({"energy_uj": energy % maximum, "max_energy_range_uj": maximum,
                                    "name": "package-0", "sample_t_ns": t} if rapl else None)
            rows.append(row)
        t += period_ns
        energy += int(cpu_w * period_ns / 1000)
    return rows


class EnergyTest(unittest.TestCase):
    def test_matches_the_campaign_summary(self):
        rows = samples(0, 20_000_000_000, wrap_uj=150_000_000)  # 30 W for 20 s wraps a 150 J counter
        energy = baseline.host_energy(rows, 1_000_000_000, 11_000_000_000)
        reference = server_energy_summary(rows, 1_000_000_000, 11_000_000_000)
        self.assertAlmostEqual(energy["gpu_board_energy_j"], reference["gpu_board_energy_j"])
        self.assertAlmostEqual(energy["cpu_package_energy_j"], reference["cpu_package_energy_j"])
        self.assertAlmostEqual(energy["gpu_board_energy_j"], 500.0, places=6)
        self.assertAlmostEqual(energy["cpu_package_energy_j"], 300.0, places=3)
        block, evidence = baseline.trace_energy_block(energy, None, "nvidia")
        self.assertEqual(evidence, "MEASURED")
        self.assertEqual(block["fleet_energy_uj_by_domain"], {"cpu-package": 300_000_000, "gpu-board": 500_000_000})
        self.assertEqual(block["measurement_evidence_ids"], ["physical:rapl-package-0", "physical:nvml-board-power"])

    def test_rapl_absent_gives_gpu_only_partial(self):
        energy = baseline.host_energy(samples(0, 5_000_000_000, rapl=False), 1_000_000_000, 4_000_000_000)
        self.assertIsNone(energy["cpu_package_energy_j"])
        self.assertAlmostEqual(energy["gpu_board_energy_j"], 150.0, places=6)
        block, evidence = baseline.trace_energy_block(energy, "PermissionError: root only", "nvidia")
        self.assertEqual((evidence, list(block["fleet_energy_uj_by_domain"])), ("PARTIAL", ["gpu-board"]))
        self.assertIn("absent: PermissionError", block["estimation_metadata"]["rapl_package"])

    def test_gap_or_short_coverage_is_reported_not_integrated(self):
        gap = baseline.host_energy(samples(0, 20_000_000_000, gap=(5_000_000_000, 11_000_000_000)),
                                   1_000_000_000, 19_000_000_000)
        self.assertIsNone(gap["gpu_board_energy_j"])
        self.assertFalse(gap["coverage"]["gpu"]["contiguous"])
        short = baseline.host_energy(samples(0, 5_000_000_000), 1_000_000_000, 9_000_000_000)
        self.assertFalse(short["coverage"]["gpu"]["covered"])
        self.assertEqual(baseline.trace_energy_block(short, None, "x")[1], "ABSENT")


class EnvironmentTest(unittest.TestCase):
    def test_behaviour_changing_variables_are_removed(self):
        base = {"PATH": "/bin", "LLAMA_ARG_N_GPU_LAYERS": "99", "GGML_CUDA_DISABLE_GRAPHS": "1",
                "S41_SERVER_FFN_SPLIT": "1", "CUDA_VISIBLE_DEVICES": "1", "LD_LIBRARY_PATH": "/old",
                "LLAMA_SERVER_WARM_TIER_CONFIG": "x"}
        environment, overrides, removed = baseline.router_environment(base, ["/cuda/lib"], Path("/srv/bin"), None)
        self.assertEqual(removed, ["CUDA_VISIBLE_DEVICES", "GGML_CUDA_DISABLE_GRAPHS", "LLAMA_ARG_N_GPU_LAYERS",
                                   "LLAMA_SERVER_WARM_TIER_CONFIG", "S41_SERVER_FFN_SPLIT"])
        self.assertEqual(environment["LD_LIBRARY_PATH"], "/cuda/lib:/srv/bin:/old")
        self.assertEqual(environment["PATH"], "/bin")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
        self.assertEqual(set(overrides), {"LD_LIBRARY_PATH"})
        local, overrides, _ = baseline.router_environment(base, [], Path("/srv/bin"), "0")
        self.assertEqual(local["CUDA_VISIBLE_DEVICES"], "0")
        self.assertIn("local test only", overrides["CUDA_VISIBLE_DEVICES"])

    def test_router_command_has_only_the_documented_flags(self):
        command = baseline.router_command(Path("/b/llama-server"), Path("/o/p.ini"), "127.0.0.1", 18600, 1, 4)
        self.assertEqual(command, ["/b/llama-server", "--models-preset", "/o/p.ini", "--models-max", "1",
                                   "--host", "127.0.0.1", "--port", "18600", "-lv", "4"])
        self.assertNotIn("-lv", baseline.router_command(Path("/b"), Path("/p"), "h", 1, 1, None))


FAKE_ROUTER = r'''
import json, os, signal, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
args = sys.argv[1:]
if "--help" in args:
    print("--models-preset PATH  --models-max N  -fit, --fit [on|off]"); sys.exit(0)
port = int(args[args.index("--port") + 1]); models_max = int(args[args.index("--models-max") + 1])
names = [l.strip()[1:-1] for l in open(args[args.index("--models-preset") + 1]) if l.startswith("[")]
lock = threading.Lock(); state = {"loaded": [], "port": 42000}
def log(line):
    sys.stdout.write(line + "\n"); sys.stdout.flush()
def ensure(name):
    with lock:
        if name in state["loaded"]:
            return
        if len(state["loaded"]) >= models_max:
            old = state["loaded"].pop(0)
            log("0.00.000.000 I srv    unload_lru: models_max limit reached, removing LRU name=" + old)
            log("0.00.000.000 I srv    operator(): instance name=%s exited with status 0" % old)
        state["port"] += 1; p = state["port"]
        log("0.00.000.000 I srv          load: spawning server instance with name=%s on port %d" % (name, p))
        log("0.00.000.000 I srv          load: spawning server instance with args:")
        log("0.00.000.000 I srv          load:   --alias")
        log("0.00.000.000 I srv          load:   " + name)
        time.sleep(0.05)
        log("[%d] 0.00.000.001 I load_tensors: offloaded 7/9 layers to GPU" % p)
        log("[%d] 0.00.000.002 I llama_context: n_ctx         = 4096" % p)
        log("[%d] 0.00.000.003 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 4096, kv_unified = 'true'" % p)
        log('[%d] cmd_child_to_router:state:{"state":"ready","payload":{"meta":{"n_ctx":4096}}}' % p)
        state["loaded"].append(name)
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike".split()
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):
        pass
    def do_GET(self):
        body = json.dumps({"status": "ok"} if self.path == "/health" else {"data": [{"id": n} for n in names]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        name = req["model"]
        if name not in names:
            body = b'{"error":"unknown model"}'
            self.send_response(400); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        ensure(name)
        log("0.00.000.000 I srv  proxy_reques: proxying request to model %s" % name)
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Connection", "close"); self.end_headers()
        n = req["n_predict"]
        for i in range(n):
            tok = 1000 + (i * 7 + req["seed"]) % 97
            chunk = {"content": WORDS[tok % len(WORDS)] + " ", "tokens": [tok], "tokens_predicted": i + 1, "stop": False, "id_slot": 0}
            self.wfile.write(("data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n").encode()); self.wfile.flush(); time.sleep(0.002)
        final = {"content": "", "tokens": [], "stop": True, "model": name, "id_slot": 0,
                 "timings": {"prompt_n": len(req["prompt"]), "prompt_ms": 1.0, "predicted_n": n, "predicted_ms": 2.0 * n, "predicted_per_token_ms": 2.0}}
        self.wfile.write(("data: " + json.dumps(final, separators=(",", ":")) + "\n\n").encode()); self.wfile.flush(); self.close_connection = True
server = ThreadingHTTPServer(("127.0.0.1", port), H)
signal.signal(signal.SIGTERM, lambda *a: threading.Thread(target=server.shutdown).start())
log("0.00.000.000 I srv  llama_server: listening on http://127.0.0.1:%d" % port)
server.serve_forever()
'''


class FakeSampler:
    """HostEnergySampler stand-in: one row per 0.2 s of wall clock, 40 W GPU, 20 W RAPL."""

    def __init__(self, _callbacks):
        self.started = None
        self.stopped = False

    def start(self):
        self.started = time.monotonic_ns() - 400_000_000

    def stop(self):
        self.stopped = True

    def rows(self):
        now = time.monotonic_ns()
        out, t = [], self.started
        while t <= now:
            out.append({"gpu": {"power_mw": 40_000, "sample_t_ns": t},
                        "rapl_package": {"energy_uj": int((t - self.started) * 20 / 1000), "name": "package-0",
                                         "max_energy_range_uj": 262_143_328_850, "sample_t_ns": t},
                        "t_ns": t})
            t += 200_000_000
        return tuple(out)

    def latest_rows(self):
        return self.rows()[-4:]

    def diagnostics(self):
        return {"sample_count": len(self.rows()), "events": [], "fatal_error": None}


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.server = self.root / "bin/llama-server"
        self.server.parent.mkdir()
        self.server.write_text("#!" + sys.executable + "\n" + FAKE_ROUTER)
        self.server.chmod(self.server.stat().st_mode | stat.S_IXUSR)
        model = self.root / "small.gguf"
        model.write_bytes(b"GGUF" + bytes(60))
        plan = baseline.smoke_plan(paper_plan(), str(model), time_scale=0.0005, max_prompt_tokens=6,
                                   max_output_tokens=12)
        plan["requests"] = plan["requests"][:6]  # cold, hot, cold, hot, hot, hot
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_bytes(canonical(plan))

    def tearDown(self):
        self.tmp.cleanup()

    def run_arm(self, *extra):
        out = self.root / "run"
        args = baseline.build_parser().parse_args([
            "run", "--plan", str(self.plan_path), "--server", str(self.server), "--out", str(out),
            "--port", str(free_port()), "--max-duration-s", "60", *extra])
        code = baseline.run(args, sampler_factory=FakeSampler, rapl_check=lambda: None)
        return code, out, json.loads((out / "RESULT.json").read_text())

    def test_fifo_drain_run_is_exact_and_readable_by_the_tools(self):
        code, out, result = self.run_arm()
        self.assertEqual(code, 0, result["status_reasons"])
        self.assertEqual((result["status"], result["energy_evidence"]), ("PASS", "MEASURED"))
        self.assertEqual(result["counts"]["completed"], 6)
        self.assertEqual(result["switch_gate"], "fifo-drain")
        flags = [row["flag"] for row in result["deviations_from_defaults"]]
        self.assertEqual(flags, ["--models-max 1", "client FIFO drain-before-switch (--switch-gate fifo-drain)",
                                 "-lv 4", "request body field \"model\""])
        self.assertIn("--models-max", result["server"]["router_command"])
        # every model switch drained the previous model first: releases never overlap two models
        rows = sorted(result["request_results"], key=lambda row: row["client"]["released_us"])
        for row in rows:
            others = [o for o in rows if o["model_id"] != row["model_id"]
                      and o["client"]["released_us"] < row["client"]["released_us"] < o["completion"]["actual_end_us"]]
            self.assertEqual(others, [], row["request_id"])
        # loads: one router autoload per model switch, charged to the triggering request
        loads = [receipt for row in result["request_results"] for receipt in row["terminal_ticket"]["transition_receipts"]]
        self.assertEqual(len(loads), len(result["model_instances"]))
        self.assertEqual(result["model_placements"]["smoke-cold"][0]["gpu_layers"], 7)
        self.assertTrue((out / "streams/request-000.raw").is_file())
        self.assertEqual((out / "server.log").read_text().count("spawning server instance with name="), 4)
        # the accountants read the output unchanged
        fleet = fleet_energy.account_run("default", out)
        self.assertEqual(fleet["desktop"]["evidence"], fleet_energy.MEASURED)
        self.assertAlmostEqual(fleet["desktop"]["host_j"],
                               (result["trace_energy"]["fleet_energy_uj_by_domain"]["cpu-package"]
                                + result["trace_energy"]["fleet_energy_uj_by_domain"]["gpu-board"]) / 1e6, places=5)
        self.assertLess(abs(fleet["desktop"]["cross_check"]["host_delta_j"]), 1.0)
        self.assertAlmostEqual(fleet["desktop"]["gpu_board_j"] / fleet["duration_s"], 40.0, places=3)
        latency = latency_report.report_run("default", out)
        self.assertEqual(latency["overall"]["n"], 6)
        first = latency["requests"][0]
        self.assertGreater(first["load_s"], 0)
        self.assertEqual(first["server_predicted_n"], first["output_tokens"])
        self.assertIsNotNone(first["queue_s"])

    def test_pure_default_client_is_recorded_as_such(self):
        code, _, result = self.run_arm("--switch-gate", "none", "--models-max", "4", "--log-verbosity", "default")
        self.assertEqual(code, 0, result["status_reasons"])
        self.assertEqual([row["flag"] for row in result["deviations_from_defaults"]], ["request body field \"model\""])
        self.assertNotIn("-lv", result["server"]["router_command"])

    def test_refuses_a_used_output_directory(self):
        (self.root / "run").mkdir()
        (self.root / "run/RESULT.json").write_text("{}")
        with self.assertRaises(SystemExit):
            self.run_arm()


class LauncherTest(unittest.TestCase):
    def test_dry_run_command_and_refusals(self):
        with tempfile.TemporaryDirectory() as root:
            template = Path(root) / "template-eval2-s2"
            template.mkdir()
            env = {**os.environ, "EVAL_ROOT": root, "EVAL_LOCK": root + "/lock", "EVAL_DRY_RUN": "1",
                   "DEFAULT_BASELINE_PORT": "18777"}
            missing = subprocess.run(["bash", str(LAUNCHER), "t1"], env=env, capture_output=True, text=True)
            self.assertEqual(missing.returncode, 2)
            for name in ("campaign.json", "models.json", "rig.json"):
                (template / name).write_text("{}")
            ok = subprocess.run(["bash", str(LAUNCHER), "t1"], env=env, capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("flock -w 7200 " + root + "/lock", ok.stdout)
            self.assertIn("default_llamacpp_baseline run --campaign '" + str(template / "campaign.json"), ok.stdout)
            self.assertIn("--out '" + root + "/inputs-default-t1/run-eval/run' --port 18777", ok.stdout)
            self.assertIn("cd '" + str(REPO) + "'", ok.stdout)
            (Path(root) / "inputs-default-t1/run-eval").mkdir(parents=True)
            again = subprocess.run(["bash", str(LAUNCHER), "t1"], env=env, capture_output=True, text=True)
            self.assertEqual(again.returncode, 13)


if __name__ == "__main__":
    unittest.main()
