#!/usr/bin/env python3
"""WS18 stock llama.cpp RPC whole-layer offload baseline (tools/rpc_llamacpp_baseline.py, tools/rpc_phone_servers.py,
tools/rig/launch_rpc_baseline.sh): placement math against llama.cpp's layer order, GGUF layer bytes, router preset
options, the RPC wire probe against a fake server, the phone lifecycle with a fake adb, an end-to-end run against a
fake router binary with fake rpc servers, and the desktop launchers (dry run)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import unittest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from research_dev.scheduler.campaigns.burstgpt.common import canonical  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import default_llamacpp_baseline as baseline  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import fleet_energy, latency_report  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import rpc_llamacpp_baseline as rpc  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import rpc_phone_servers as phones  # noqa: E402

import test_default_llamacpp_baseline as ws7  # noqa: E402

REPO = TESTS_DIR.parents[2]
RIG = REPO / "research_dev/scheduler/campaigns/burstgpt/tools/rig"
SHA = "a" * 64


def geometry(n_layer: int, layer_bytes: int, kv: int = 0) -> dict:
    return {"architecture": "qwen3", "n_layer": n_layer, "non_layer_bytes": 0, "ctx": 4096,
            "layers": [{"layer": i, "weight_bytes": layer_bytes, "kv_bytes": kv, "bytes": layer_bytes + kv}
                       for i in range(n_layer)]}


def phone_rows(ports=((50052, 50152), (50053, 50153))) -> dict:
    names = ("op15", "pixel")
    return {"schema": phones.PHONES_SCHEMA, "phones": [
        {"name": names[i], "serial": "SER%d" % i, "phone_dir": "/data/local/tmp/s43-rpc-v1", "phone_port": phone,
         "host_port": host, "expected_sha256": {"ggml-rpc-server": SHA, "libggml.so": "b" * 64}}
        for i, (phone, host) in enumerate(ports)]}


class PlacementTest(unittest.TestCase):
    def test_phones_take_cpu_layers_nearest_the_gpu_block_in_llama_cpp_order(self):
        # 40 layers, 18 on the GPU -> CPU-resident 0..21. Pixel (RPC1, next to CUDA) fills first, then the OP15.
        plan = rpc.plan_model(geometry(40, 100), 18, [("op15", 750), ("pixel", 450)], margin_bytes=50)
        self.assertEqual(plan["phones"]["pixel"]["layers"], 4)        # (450 - 50) // 100
        self.assertEqual(plan["phones"]["op15"]["layers"], 7)         # (750 - 50) // 100
        self.assertEqual(plan["phones"]["pixel"]["range"], [18, 22])
        self.assertEqual(plan["phones"]["op15"]["range"], [11, 18])
        self.assertEqual(plan["cpu_layers"], 11)
        self.assertEqual(plan["n_gpu_layers"], 4 + 7 + 18 + 1)
        self.assertEqual(plan["tensor_split"], [7, 4, 19])
        # replay llama.cpp's split points (src/llama-model.cpp) and check every layer's device
        ngl, split = plan["n_gpu_layers"], plan["tensor_split"]
        start = 40 + 1 - ngl
        cumulative, total = [], float(sum(split))
        acc = 0.0
        for value in split:
            acc += value
            cumulative.append(acc / total)
        owner = []
        for layer in range(41):
            if layer < start:
                owner.append("cpu")
                continue
            fraction = float(layer - start) / ngl
            index = next(i for i, c in enumerate(cumulative) if c > fraction)
            owner.append(("op15", "pixel", "cuda")[index])
        self.assertEqual(owner[:11], ["cpu"] * 11)
        self.assertEqual(owner[11:18], ["op15"] * 7)
        self.assertEqual(owner[18:22], ["pixel"] * 4)
        self.assertEqual(owner[22:], ["cuda"] * 19)                    # 18 layers + output

    def test_all_cpu_layers_fit_and_no_phone_layer_means_no_options(self):
        full = rpc.plan_model(geometry(10, 100), 6, [("op15", 10_000), ("pixel", 10_000)], margin_bytes=0)
        self.assertEqual((full["cpu_layers"], full["phones"]["pixel"]["layers"], full["phones"]["op15"]["layers"]),
                         (0, 4, 0))
        self.assertEqual(full["tensor_split"], [0, 4, 7])
        none = rpc.plan_model(geometry(10, 100), 10, [("op15", 10_000)], margin_bytes=0)
        self.assertEqual(none["phone_layers"], 0)
        self.assertEqual(rpc.preset_options(none, ["127.0.0.1:1"], 4096), {})
        tight = rpc.plan_model(geometry(10, 100), 2, [("op15", 90)], margin_bytes=0)
        self.assertEqual(tight["phone_layers"], 0)
        with self.assertRaises(ValueError):
            rpc.plan_model(geometry(10, 100), 11, [("op15", 1)], margin_bytes=0)

    def test_required_phone_memory_is_the_largest_planned_footprint(self):
        placement = {"models": {
            "q": {"phones": {"op15": {"planned_bytes": 900, "margin_bytes": 100}, "pixel": {"planned_bytes": 50,
                                                                                             "margin_bytes": 100}}},
            "g": {"phones": {"op15": {"planned_bytes": 700, "margin_bytes": 100}}}, "l": {"phone_layers": 0}}}
        self.assertEqual(rpc.required_bytes(placement), {"op15": 1000, "pixel": 150})
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "PHONES.json").write_text(json.dumps(phone_rows()))
            phone_list = phones.load_phones(Path(root) / "PHONES.json")
            servers = rpc.PhoneServers(phone_list, Path(root), runner=FakeAdb(mem_kib=1),
                                       required={"op15": 10 ** 9})
            self.assertEqual([p.min_mem_available_bytes for p in servers.phones], [10 ** 9, 0])
            with self.assertRaisesRegex(phones.RpcPhoneError, "MemAvailable"):
                servers.__enter__()

    def test_gpu_layers_from_an_out_of_the_box_run(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "MODEL_PLACEMENTS.json"
            path.write_text(json.dumps({"by_model": {"q": [{"gpu_layers": 19}, {"gpu_layers": 19}],
                                                     "g": [{"gpu_layers": 25}]}}))
            self.assertEqual(rpc.gpu_layers_from_reference(path), {"q": 18, "g": 24})
            path.write_text(json.dumps({"by_model": {"q": [{"gpu_layers": 19}, {"gpu_layers": 18}]}}))
            with self.assertRaises(ValueError):
                rpc.gpu_layers_from_reference(path)

    def test_gguf_layer_bytes_and_kv(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "tiny.gguf"
            write_tiny_gguf(path)
            layers = rpc.model_layers(path, 4096)
        self.assertEqual(layers["n_layer"], 2)
        # per layer: attn_q 8x8 f16 (128 B) + ffn_up 8x16 f16 (256 B); kv: 4096 ctx x 2 heads x (4 + 4) x 2 B
        self.assertEqual([row["weight_bytes"] for row in layers["layers"]], [384, 384])
        self.assertEqual(layers["layers"][0]["kv_bytes"], 4096 * 2 * 8 * 2)
        self.assertEqual(layers["non_layer_bytes"], 8 * 32 * 2)


def _gguf_string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_tiny_gguf(path: Path) -> None:
    kv = [("general.architecture", 8, _gguf_string("qwen3")), ("qwen3.block_count", 4, struct.pack("<I", 2)),
          ("qwen3.attention.head_count", 4, struct.pack("<I", 2)),
          ("qwen3.attention.head_count_kv", 4, struct.pack("<I", 2)),
          ("qwen3.embedding_length", 4, struct.pack("<I", 8)),
          ("qwen3.attention.key_length", 4, struct.pack("<I", 4)),
          ("qwen3.attention.value_length", 4, struct.pack("<I", 4))]
    tensors = [("token_embd.weight", (8, 32)), ("blk.0.attn_q.weight", (8, 8)), ("blk.0.ffn_up.weight", (8, 16)),
               ("blk.1.attn_q.weight", (8, 8)), ("blk.1.ffn_up.weight", (8, 16))]
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, kind, value in kv:
        out += _gguf_string(key) + struct.pack("<I", kind) + value
    offset = 0
    for name, shape in tensors:
        out += _gguf_string(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
        out += struct.pack("<IQ", 1, offset)                       # F16
        offset += shape[0] * shape[1] * 2
    path.write_bytes(out + bytes(offset))


class PresetTest(unittest.TestCase):
    def test_options_become_preset_lines_and_bad_options_are_refused(self):
        models = [{"model_id": "q", "path": "/m/q.gguf", "preset_options": {"rpc": "127.0.0.1:1,127.0.0.1:2",
                                                                             "n-gpu-layers": "30"}},
                  {"model_id": "l", "path": "/m/l.gguf"}]
        text = baseline.preset_text(models)
        self.assertIn("[q]\nmodel = /m/q.gguf\nn-gpu-layers = 30\nrpc = 127.0.0.1:1,127.0.0.1:2\n", text)
        self.assertIn("[l]\nmodel = /m/l.gguf\n", text)
        self.assertTrue(text.startswith("; derived arm"))
        plan = ws7.paper_plan()
        for bad in ({"model": "/x"}, {"rpc": "a\nb"}, {"Rpc": "1"}, {"rpc": "x;y"}, {"rpc": ""}):
            broken = json.loads(json.dumps(plan))
            broken["models"][0]["preset_options"] = bad
            with self.assertRaises(ValueError):
                baseline.validate_plan(broken)
        baseline.validate_plan(plan)  # the default plan carries no extension

    def test_plan_binds_placement_phone_order_and_deviations(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "PHONES.json").write_text(json.dumps(phone_rows()))
            phone_list = phones.load_phones(Path(root) / "PHONES.json")
        trace_plan = ws7.paper_plan()
        placement = {"ctx": 4096, "phones": [{"name": "op15", "budget_bytes": 1}, {"name": "pixel", "budget_bytes": 1}],
                     "models": {row["model_id"]: {"phone_layers": 0} for row in trace_plan["models"]}}
        hot = trace_plan["models"][0]["model_id"]
        placement["models"][hot] = {"phone_layers": 11, "n_gpu_layers": 30, "tensor_split": [7, 4, 19]}
        plan = rpc.build_plan(trace_plan, placement, phone_list)
        self.assertEqual(plan["models"][0]["preset_options"],
                         {"ctx-size": "4096", "n-gpu-layers": "30", "rpc": "127.0.0.1:50152,127.0.0.1:50153",
                          "tensor-split": "7,4,19"})
        self.assertNotIn("preset_options", plan["models"][1])
        self.assertEqual(plan["arm_extension"]["arm"], rpc.ARM)
        self.assertEqual(plan["arm_extension"]["required_server_flags"], ["--rpc", "--tensor-split"])
        self.assertEqual(plan["requests"], trace_plan["requests"])      # the work is unchanged
        placement["phones"].reverse()
        with self.assertRaises(ValueError):
            rpc.build_plan(trace_plan, placement, phone_list)


class FakeRpcServer:
    """Minimal ggml-rpc server: HELLO (24 caps), DEVICE_COUNT, GET_DEVICE_MEMORY; one client at a time."""

    def __init__(self, devices=((8 << 30, 16 << 30),), version=(4, 0, 1)):
        self.devices, self.version = devices, version
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.connections = 0
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                client, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1
            with client:
                try:
                    while True:
                        head = phones._recv_exact(client, 9)
                        cmd, size = head[0], struct.unpack("<Q", head[1:])[0]
                        payload = phones._recv_exact(client, size) if size else b""
                        if cmd == phones.CMD_HELLO:
                            body = bytes(self.version) + b"\0" + bytes(24)
                        elif cmd == phones.CMD_DEVICE_COUNT:
                            body = struct.pack("<I", len(self.devices))
                        elif cmd == phones.CMD_GET_DEVICE_MEMORY:
                            body = struct.pack("<QQ", *self.devices[struct.unpack("<I", payload)[0]])
                        else:
                            break
                        client.sendall(struct.pack("<Q", len(body)) + body)
                except phones.RpcPhoneError:
                    pass

    def close(self):
        self.sock.close()


class ProbeTest(unittest.TestCase):
    def test_hello_devices_and_memory(self):
        server = FakeRpcServer(devices=((1, 2), (3, 4)))
        try:
            result = phones.probe("127.0.0.1:%d" % server.port)
        finally:
            server.close()
        self.assertEqual(result["protocol"], "4.0.1")
        self.assertEqual(result["devices"], [{"device": 0, "free_bytes": 1, "total_bytes": 2},
                                             {"device": 1, "free_bytes": 3, "total_bytes": 4}])

    def test_wait_for_probe_times_out(self):
        with self.assertRaises(phones.RpcPhoneError):
            phones.wait_for_probe("127.0.0.1:%d" % ws7.free_port(), 0.6, interval_s=0.1)


class FakeAdb:
    """Records adb argv and answers like a phone with the pinned files installed."""

    def __init__(self, sha=SHA, state="device", leftover="", mem_kib=12_000_000, start_pid="4242"):
        self.calls, self.sha, self.state, self.leftover = [], sha, state, leftover
        self.mem_kib, self.start_pid, self.alive = mem_kib, start_pid, True

    def __call__(self, argv, timeout_s):
        self.calls.append(list(argv))
        text = " ".join(argv)
        out, rc = "", 0
        if argv[-1] == "get-state":
            out = self.state + "\n"
        elif "sha256sum" in text:
            out = "%s  ggml-rpc-server\n%s  libggml.so\n" % (self.sha, "b" * 64)
        elif "setsid ./ggml-rpc-server" in text:   # the start line ends with pidof: match it first
            out = self.start_pid + "\n"
        elif "pidof ggml-rpc-server" in text:
            out = self.leftover if self.alive is False or self.leftover else ""
        elif "/proc/meminfo" in text:
            out = "MemTotal: 15000000 kB\nMemAvailable: %d kB\n" % self.mem_kib
        elif "kill -TERM" in text:
            self.alive = False
        elif "kill -0" in text:
            rc = 0 if self.alive else 1
        elif "cat /data/local/tmp/s43-rpc-v1/rpc-server.log" in text:
            out = "Starting RPC server v4.0.1\n"
        return subprocess.CompletedProcess(argv, rc, out, "")


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "PHONES.json").write_text(json.dumps(phone_rows()))
        self.op15 = phones.load_phones(self.root / "PHONES.json")[0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_preflight_refusals(self):
        self.assertEqual(phones.preflight(self.op15, FakeAdb())["mem_available_bytes"], 12_000_000 * 1024)
        for fake, message in ((FakeAdb(state="unauthorized"), "adb state"), (FakeAdb(sha="c" * 64), "differ"),
                              (FakeAdb(leftover="99"), "already runs")):
            with self.assertRaisesRegex(phones.RpcPhoneError, message):
                phones.preflight(self.op15, fake)
        hungry = phones.PhoneRpcServer(**{**self.op15.__dict__, "min_mem_available_bytes": 20 * 10 ** 9})
        with self.assertRaisesRegex(phones.RpcPhoneError, "MemAvailable"):
            phones.preflight(hungry, FakeAdb())

    def test_start_forwards_probes_and_stop_terminates_only_its_pid(self):
        fake = FakeAdb()
        started = phones.start(self.op15, runner=fake, probe_fn=lambda endpoint: {"endpoint": endpoint,
                                                                                    "protocol": "4.0.1"})
        self.assertEqual(started.pid, 4242)
        launch = next(call for call in fake.calls if "setsid ./ggml-rpc-server" in " ".join(call))
        self.assertEqual(launch[:6], ["adb", "-P", "5037", "-s", "SER0", "shell"])
        self.assertIn("su -c ", launch[6])
        self.assertIn("./ggml-rpc-server -H 127.0.0.1 -p 50052 -c", launch[6])
        self.assertIn("LLAMA_CACHE=/data/local/tmp/s43-rpc-v1/cache", launch[6])
        self.assertIn(["adb", "-P", "5037", "-s", "SER0", "forward", "tcp:50152", "tcp:50052"], fake.calls)
        receipt = phones.stop(started, runner=fake, out_dir=self.root)
        self.assertTrue(receipt["clean"])
        kills = [call for call in fake.calls if "kill -TERM" in " ".join(call)]
        self.assertEqual(len(kills), 1)
        self.assertIn("kill -TERM 4242", kills[0][-1])
        self.assertNotIn("-9", " ".join(kills[0]))
        self.assertIn(["adb", "-P", "5037", "-s", "SER0", "forward", "--remove", "tcp:50152"], fake.calls)
        self.assertIn("Starting RPC server", (self.root / "rpc-server-op15.log").read_text())

    def test_a_failed_probe_stops_the_started_server(self):
        fake = FakeAdb()

        def refuse(_endpoint):
            raise phones.RpcPhoneError("no answer")

        with self.assertRaises(phones.RpcPhoneError):
            phones.start(self.op15, runner=fake, probe_fn=refuse, ready_timeout_s=0.3)
        self.assertTrue(any("kill -TERM 4242" in " ".join(call) for call in fake.calls))

    def test_configuration_refusals(self):
        bad = phone_rows()
        bad["phones"][0]["phone_dir"] = "/sdcard/x"
        (self.root / "bad.json").write_text(json.dumps(bad))
        with self.assertRaises(phones.RpcPhoneError):
            phones.load_phones(self.root / "bad.json")
        bad = phone_rows(((50052, 50152), (50053, 50152)))
        (self.root / "bad.json").write_text(json.dumps(bad))
        with self.assertRaises(phones.RpcPhoneError):
            phones.load_phones(self.root / "bad.json")
        bad = phone_rows()
        bad["phones"][1]["transport"] = "aoa-bridge"
        (self.root / "bad.json").write_text(json.dumps(bad))
        with self.assertRaisesRegex(phones.RpcPhoneError, "aoa"):
            phones.load_phones(self.root / "bad.json")


FAKE_RPC_ROUTER = ws7.FAKE_ROUTER.replace(
    'print("--models-preset PATH  --models-max N  -fit, --fit [on|off]"); sys.exit(0)',
    'print("--models-preset PATH  --models-max N  -fit, --fit [on|off] --rpc SERVERS -ts, --tensor-split N0,N1"); '
    'sys.exit(0)')


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.server = self.root / "bin/llama-server"
        self.server.parent.mkdir()
        self.server.write_text("#!" + sys.executable + "\n" + FAKE_RPC_ROUTER)
        self.server.chmod(self.server.stat().st_mode | stat.S_IXUSR)
        model = self.root / "small.gguf"
        model.write_bytes(b"GGUF" + bytes(60))
        self.rpc_servers = [FakeRpcServer(), FakeRpcServer()]
        rows = phone_rows(((50052, self.rpc_servers[0].port), (50053, self.rpc_servers[1].port)))
        (self.root / "PHONES.json").write_text(json.dumps(rows))
        plan = baseline.smoke_plan(ws7.paper_plan(), str(model), time_scale=0.0005, max_prompt_tokens=6,
                                   max_output_tokens=12)
        plan["requests"] = plan["requests"][:4]
        placement = {"ctx": 4096, "phones": [{"name": "op15", "budget_bytes": 1}, {"name": "pixel", "budget_bytes": 1}],
                     "models": {row["model_id"]: {"phone_layers": 0} for row in plan["models"]}}
        for row in plan["models"][:2]:
            placement["models"][row["model_id"]] = {"phone_layers": 9, "n_gpu_layers": 25, "tensor_split": [5, 4, 16]}
        self.plan = rpc.build_plan(plan, placement, phones.load_phones(self.root / "PHONES.json"))
        (self.root / "PLAN.json").write_bytes(canonical(self.plan))

    def tearDown(self):
        for server in self.rpc_servers:
            server.close()
        self.tmp.cleanup()

    def test_run_writes_a_result_the_accountants_read_and_records_the_arm(self):
        out = self.root / "run"
        args = rpc.build_parser().parse_args([
            "run", "--plan", str(self.root / "PLAN.json"), "--phones", str(self.root / "PHONES.json"),
            "--server", str(self.server), "--out", str(out), "--port", str(ws7.free_port()),
            "--max-duration-s", "60", "--phones-already-running"])
        code = rpc.run(args, sampler_factory=ws7.FakeSampler, rapl_check=lambda: None, repo=REPO)
        result = json.loads((out / "RESULT.json").read_text())
        self.assertEqual(code, 0, result["status_reasons"])
        self.assertEqual(result["arm"], rpc.ARM)
        flags = [row["flag"] for row in result["deviations_from_defaults"]]
        self.assertEqual(flags[-3:], [row["flag"] for row in rpc.DEVIATIONS])
        self.assertIn("rpc = 127.0.0.1:%d,127.0.0.1:%d" % (self.rpc_servers[0].port, self.rpc_servers[1].port),
                      (out / "models-preset.ini").read_text())
        self.assertTrue(result["rpc_phone_servers"]["already_running"])
        self.assertEqual([row["protocol"] for row in result["rpc_phone_servers"]["started"]], ["4.0.1", "4.0.1"])
        self.assertTrue((out / "streams/request-000.raw").is_file() and (out / "RPC_PHONES.json").is_file())
        self.assertFalse((out / "router").exists())
        fleet = fleet_energy.account_run("rpc", out)
        self.assertEqual(fleet["desktop"]["evidence"], fleet_energy.MEASURED)
        self.assertEqual(latency_report.report_run("rpc", out)["overall"]["n"], 4)

    def test_protocol_major_mismatch_is_refused(self):
        self.rpc_servers[1].version = (3, 0, 0)
        args = rpc.build_parser().parse_args([
            "run", "--plan", str(self.root / "PLAN.json"), "--phones", str(self.root / "PHONES.json"),
            "--server", str(self.server), "--out", str(self.root / "run2"), "--port", str(ws7.free_port()),
            "--phones-already-running"])
        with self.assertRaisesRegex(SystemExit, "protocol"):
            rpc.run(args, sampler_factory=ws7.FakeSampler, rapl_check=lambda: None, repo=REPO)


class LauncherTest(unittest.TestCase):
    def test_dry_run_and_refusals(self):
        with tempfile.TemporaryDirectory() as root:
            template = Path(root) / "template-eval2-s2"
            template.mkdir()
            base = {**os.environ, "EVAL_ROOT": root, "EVAL_LOCK": root + "/lock", "EVAL_DRY_RUN": "1",
                    "RPC_BASELINE_PORT": "18777", "RPC_SERVER_BIN": root + "/llama-server-rpc"}
            launcher = RIG / "launch_rpc_baseline.sh"
            missing = subprocess.run(["bash", str(launcher), "t1"], env=base, capture_output=True, text=True)
            self.assertEqual(missing.returncode, 2)
            for name in ("campaign.json", "models.json"):
                (template / name).write_text("{}")
            (template / "rig.json").write_text(json.dumps({"library_directories": {"cuda": "/opt/cuda/lib"}}))
            (Path(root) / "rpc-baseline").mkdir()
            for name in ("PHONES.json", "PLACEMENT.json"):
                (Path(root) / "rpc-baseline" / name).write_text("{}")
            ok = subprocess.run(["bash", str(launcher), "t1"], env=base, capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("flock -w 7200 " + root + "/lock", ok.stdout)
            self.assertIn("rpc_llamacpp_baseline plan --campaign '" + str(template / "campaign.json"), ok.stdout)
            self.assertIn("rpc_llamacpp_baseline run --plan '" + root + "/inputs-rpc-t1/PLAN.json'", ok.stdout)
            self.assertIn("--library-dir /opt/cuda/lib", ok.stdout)
            self.assertIn("--out '" + root + "/inputs-rpc-t1/run-eval/run' --port 18777", ok.stdout)
            (Path(root) / "inputs-rpc-t1/run-eval").mkdir(parents=True)
            again = subprocess.run(["bash", str(launcher), "t1"], env=base, capture_output=True, text=True)
            self.assertEqual(again.returncode, 13)
            # launch_one_arm.sh KIND=rpc hands the arm to the rpc launcher
            (Path(root) / "inputs-rpc-t1/run-eval").rmdir()
            (Path(root) / "source").symlink_to(REPO)  # launch_one_arm hands EVAL_REPO=$EVAL_DEPLOY/source over
            via = subprocess.run(["bash", str(RIG / "launch_one_arm.sh"), "t1", "template-eval2-s2", "rpc"],
                                 env={**base, "EVAL_DEPLOY": root}, capture_output=True, text=True)
            self.assertEqual(via.returncode, 0, via.stderr)
            self.assertIn("rpc_llamacpp_baseline run", via.stdout)
            bad = subprocess.run(["bash", str(RIG / "launch_one_arm.sh"), "t1", "template-eval2-s2", "rpcx"],
                                 env=base, capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
