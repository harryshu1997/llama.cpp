#!/usr/bin/env python3
"""Unit tests for the WiFi-server harness pure functions.  Run from this directory:

    python3 -m unittest -v
"""
import copy
import json
import os
import stat
import struct
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import analyze
import benchlib as bl
import server_bench as sb

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE / "wifi_config.example.json"
SHA = "sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718"

# verbatim lines from the desktop reference run (server-prep/reference/ref_qwen_server_stderr_head.txt)
REF_HELPER = ("S41SERVERFFNHELPER label=pixel10pro layer_mask=16515072 transport=tcp host=127.0.0.1 port=26991 "
              "connection=deferred")
REF_CAPS = "S41SERVERFFNCAPS helper_mask_out=1 helper_reconnect_tcp=1"
REF_READY = ("S41SERVERFFN ready host=usb port=0 columns=17408 max_tokens=4 io=f16 activation=swiglu weights=view_safe "
             "runtime_control=enabled initial_mask=0 initial_columns=0 connection=deferred helpers=2")
REF_USB = ("S41SERVERFFNUSB request=1 layer=0 tokens=2 columns=17408 slot=1 h2d_bytes=20608 d2h_bytes=20608 "
           "started_ns=1492022136757336 h2d_completed_ns=1492022136872775 d2h_completed_ns=1492022151498590 "
           "compute_us=11822")
REF_CALL = ("S41SERVERFFNCALL context=6275727374:1:1:1,6275727375:2:1:1 request=1 layer=0 tokens=2 columns=17408 "
            "payload_bytes=20480")
REF_CONTROL = ("0.09.131.473 I srv  process_sing: FFNCONTROL request=wifibaf2db-phone-c1-w0-s0 slot=0 generation=1 "
               "token=2 mask=16777215 columns=17408 policy=sha256:6684")


def example():
    with open(EXAMPLE) as handle:
        return json.load(handle)


class LayerAndShareTests(unittest.TestCase):
    def test_layer_spec_roundtrip(self):
        self.assertEqual(bl.parse_layer_spec("0-5"), 63)
        self.assertEqual(bl.parse_layer_spec("6-11"), 4032)
        self.assertEqual(bl.parse_layer_spec("12-17"), 258048)
        self.assertEqual(bl.parse_layer_spec("18-23"), 16515072)
        self.assertEqual(bl.parse_layer_spec("0,2,4-5"), 0b110101)
        for mask in (63, 16515072, 0b110101, 1 << 63):
            self.assertEqual(bl.parse_layer_spec(bl.layer_spec(mask)), mask)
        self.assertEqual(bl.layer_spec(16777215), "0-23")
        for bad in ("", "5-3", "1-2-3", "64", "a", "1,,2"):
            with self.assertRaises(ValueError):
                bl.parse_layer_spec(bad)

    def test_share_to_columns(self):
        self.assertEqual([bl.share_to_columns(s, 17408, 4352) for s in (0.25, 0.5, 0.75, 1.0)],
                         [4352, 8704, 13056, 17408])
        self.assertEqual(bl.share_to_columns(0.125, 17408, 2176), 2176)
        with self.assertRaises(ValueError):
            bl.share_to_columns(0.3, 17408, 4352)
        with self.assertRaises(ValueError):
            bl.share_to_columns(0.5, 17408, 5000)
        with self.assertRaises(ValueError):
            bl.share_to_columns(1.5, 17408, 4352)

    def test_parse_arm(self):
        self.assertEqual(bl.parse_arm("cpu")["helpers"], False)
        self.assertEqual(bl.parse_arm("gpu")["kind"], "gpu")
        self.assertEqual(bl.parse_arm("cpu-helpers")["helpers"], True)
        self.assertEqual(bl.parse_arm("cpu-helpers")["columns"], 0)
        self.assertEqual(bl.parse_arm("phone")["columns"], 17408)
        split = bl.parse_arm("split-75")
        self.assertEqual((split["kind"], split["columns"], split["share"]), ("split", 13056, 0.75))
        self.assertEqual(bl.parse_arm("split-100")["kind"], "phone")
        for bad in ("split-30", "split-0", "fast", "split-"):
            with self.assertRaises(ValueError):
                bl.parse_arm(bad)
        self.assertEqual(bl.lcm(2176, 4352), 4352)


class ConfigTests(unittest.TestCase):
    def test_example_config(self):
        config = bl.load_config(str(EXAMPLE))
        labels = [helper["label"] for helper in config["helpers"]]
        self.assertEqual(labels, ["op15-htp0", "op15-htp1", "op15-htp2", "pixel"])
        self.assertEqual(config["phone_layer_mask"], 16777215)
        self.assertEqual(config["union_quantum"], 4352)
        self.assertEqual([h["port"] for h in config["helpers"]], [7071, 7072, 7073, 7074])
        self.assertEqual(config["helpers"][3]["host"], "192.168.77.52")
        self.assertEqual(config["helpers"][1]["backend"], "HTP1")

    def test_overlap_and_limits(self):
        raw = example()
        raw["phones"][1]["workers"][0]["layers"] = "17-23"
        with self.assertRaisesRegex(ValueError, "overlaps"):
            bl.load_config(raw)
        raw = example()
        raw["phones"][1]["workers"][0]["label"] = "op15-htp0"
        with self.assertRaisesRegex(ValueError, "duplicate worker label"):
            bl.load_config(raw)
        raw = example()
        raw["phones"][0]["workers"] = [
            {"label": "w%d" % i, "layers": str(i), "port": 7100 + i, "backend": "HTP0"} for i in range(9)]
        raw["phones"][1]["workers"] = []
        with self.assertRaisesRegex(ValueError, "at most 8"):
            bl.load_config(raw)
        raw = example()
        raw["phones"][0]["lock_path"] = "/data/local/tmp/lock"
        with self.assertRaisesRegex(ValueError, "single worker"):
            bl.load_config(raw)
        raw = example()
        raw["phones"][1]["columns"] = 8704
        with self.assertRaisesRegex(ValueError, "full width"):
            bl.load_config(raw)
        raw = example()
        raw["server"]["column_quantum"] = 2176
        with self.assertRaisesRegex(ValueError, "LCM quantum"):
            bl.load_config(raw)
        raw = example()
        raw["phones"][1]["workers"][0]["port"] = 7071
        raw["phones"][1]["wlan_ip"] = raw["phones"][0]["wlan_ip"]
        with self.assertRaisesRegex(ValueError, "duplicate endpoint"):
            bl.load_config(raw)


class ServerLaunchTests(unittest.TestCase):
    def setUp(self):
        self.config = bl.load_config(str(EXAMPLE))

    def test_multi_helper_environment(self):
        env = bl.server_ffn_environment(self.config, bl.parse_arm("split-25"))
        self.assertEqual(env["S41_SERVER_FFN_HELPERS"], "4")
        self.assertEqual(env["S41_SERVER_FFN_LAYER_MASK"], "16777215")
        self.assertEqual(env["S41_SERVER_FFN_COLUMNS"], "17408")
        self.assertEqual(env["S41_SERVER_FFN_RUNTIME_CONTROL"], "1")
        self.assertEqual(env["S41_SERVER_FFN_MAX_TOKENS"], "4")
        self.assertEqual(env["S41_SERVER_FFN_HELPER2_LAYER_MASK"], "258048")
        self.assertEqual(env["S41_SERVER_FFN_HELPER3_HOST"], "192.168.77.52")
        self.assertEqual(env["S41_SERVER_FFN_HELPER3_PORT"], "7074")
        self.assertTrue(all(env["S41_SERVER_FFN_HELPER%d_TRANSPORT" % i] == "tcp" for i in range(4)))
        self.assertNotIn("S41_SERVER_FFN_HOST", env)
        masks = [int(env["S41_SERVER_FFN_HELPER%d_LAYER_MASK" % i]) for i in range(4)]
        self.assertEqual(sum(masks), 16777215)
        self.assertEqual(bl.server_ffn_environment(self.config, bl.parse_arm("cpu")), {})

    def test_single_helper_legacy_environment(self):
        helper = self.config["helpers"][:1]
        env = bl.server_ffn_environment(self.config, bl.parse_arm("phone"), helper)
        self.assertEqual((env["S41_SERVER_FFN_TRANSPORT"], env["S41_SERVER_FFN_HOST"], env["S41_SERVER_FFN_PORT"]),
                         ("tcp", "192.168.77.51", "7071"))
        self.assertNotIn("S41_SERVER_FFN_HELPERS", env)

    def test_tap_rewrites_endpoints(self):
        tapped = bl.helpers_for_tap(self.config["helpers"], 7170)
        self.assertEqual([(h["host"], h["port"]) for h in tapped][:2], [("127.0.0.1", 7170), ("127.0.0.1", 7171)])
        self.assertEqual((tapped[3]["target_host"], tapped[3]["target_port"]), ("192.168.77.52", 7074))

    def test_argv_and_process_env(self):
        argv = bl.server_argv(self.config, bl.parse_arm("cpu"), "/x/llama-server")
        self.assertEqual(argv[argv.index("--n-gpu-layers") + 1], "16")
        self.assertIn("--device", argv)
        self.assertEqual(argv[argv.index("--threads") + 1], "48")
        gpu = bl.server_argv(self.config, bl.parse_arm("gpu"))
        self.assertEqual(gpu[gpu.index("--n-gpu-layers") + 1], "999")
        env = bl.server_process_environment(self.config, bl.parse_arm("cpu"),
                                            base={"S41_SERVER_FFN_HOST": "stale", "LLAMA_FFN_SPLIT_COLUMNS": "9",
                                                  "PATH": "/bin"})
        self.assertNotIn("S41_SERVER_FFN_HOST", env)
        self.assertNotIn("LLAMA_FFN_SPLIT_COLUMNS", env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["GGML_CUDA_DISABLE_GRAPHS"], "1")
        self.assertEqual(bl.cpu_resident_layers(40, 16), list(range(24)))

    def test_control_payloads(self):
        single = sb.control_payloads([("r0", 0)], 16777215, 8704, "split-50")
        self.assertEqual(single[0]["action"], "ffn_split")
        self.assertEqual((single[0]["slot_id"], single[0]["columns"], single[0]["enabled"]), (0, 8704, True))
        self.assertTrue(bl.SHA256_RE.match(single[0]["policy_hash"]))
        cohort = sb.control_payloads([("r0", 0), ("r1", 1)], 16777215, 8704, "split-50")
        self.assertEqual(cohort[0]["action"], "ffn_split_cohort")
        self.assertEqual(cohort[0]["members"], [{"request_id": "r0", "slot_id": 0}, {"request_id": "r1", "slot_id": 1}])
        log = [{"action": "ffn_split_cohort", "request_ids": ["r0", "r1"],
                "response": {"success": True, "cohort_members": [
                    {"request_id": "r0", "applied_token_index": 2}, {"request_id": "r1", "applied_token_index": 3}]}}]
        self.assertEqual(sb.applied_index(log, "r1"), 3)
        self.assertIsNone(sb.applied_index(log, "r9"))
        log = [{"action": "ffn_split", "request_ids": ["r0"], "response": {"success": False}},
               {"action": "ffn_split", "request_ids": ["r0"], "response": {"success": True, "applied_token_index": 1}}]
        self.assertEqual(sb.applied_index(log, "r0"), 1)


class LineParsingTests(unittest.TestCase):
    def test_reference_lines(self):
        helper = bl.parse_ffn_line(REF_HELPER)
        self.assertEqual((helper["kind"], helper["label"], helper["layer_mask"], helper["port"]),
                         ("helper", "pixel10pro", 16515072, 26991))
        self.assertEqual(bl.parse_ffn_line(REF_CAPS)["helper_reconnect_tcp"], 1)
        ready = bl.parse_ffn_line(REF_READY)
        self.assertEqual((ready["kind"], ready["columns"], ready["max_tokens"], ready["helpers"]),
                         ("ready", 17408, 4, 2))
        usb = bl.parse_ffn_line(REF_USB)
        self.assertEqual(usb["kind"], "usb_call")
        self.assertAlmostEqual(usb["rpc_ms"], 14.741254, places=5)
        self.assertAlmostEqual(usb["h2d_ms"], 0.115439, places=5)
        self.assertAlmostEqual(usb["compute_ms"], 11.822)
        self.assertAlmostEqual(usb["transport_ms"], 14.741254 - 11.822, places=5)
        call = bl.parse_ffn_line(REF_CALL)
        self.assertEqual((call["kind"], call["layer"], call["tokens"], call["payload_bytes"]), ("call", 0, 2, 20480))
        control = bl.parse_ffn_line(REF_CONTROL)
        self.assertEqual((control["kind"], control["slot"], control["columns"]), ("control", 0, 17408))
        self.assertIsNone(bl.parse_ffn_line("0.37.759.309 I load_tensors: offloaded 16/41 layers to GPU"))

    def test_summary_and_shape_json(self):
        summary = bl.parse_ffn_line('S41SERVERFFN {"helper":"pixel","layer_mask":16515072,"status":"ok","calls":696,'
                                    '"rpc_p50_ms":8.79,"compute_p50_ms":8.67,"host_p50_ms":0.01}')
        self.assertEqual((summary["kind"], summary["helper"], summary["calls"]), ("summary", "pixel", 696))
        shape = bl.parse_ffn_line('S41SERVERFFNSHAPE {"tokens":1,"columns":8704,"calls":10,"rpc_mean_ms":8.0}')
        self.assertEqual((shape["kind"], shape["columns"]), ("shape", 8704))
        error = bl.parse_ffn_line("S41SERVERFFNERROR helper=pixel detail=cannot connect to FFN split worker")
        self.assertEqual(error["kind"], "error")
        self.assertIn("cannot connect", error["text"])
        self.assertEqual(bl.parse_ffn_line("S41SERVERFFN dormant_policy drop_cache=0 populate=1")["kind"], "info")

    def test_call_summaries(self):
        rows = [bl.usb_call_timing({"started_ns": 0, "h2d_completed_ns": 100_000, "d2h_completed_ns": ms * 1_000_000,
                                    "compute_us": 8000}) for ms in (10, 11, 12, 30)]
        for layer, row in enumerate(rows):
            row["layer"] = layer % 2
        summary = bl.summarize_calls(rows)
        self.assertEqual(summary["calls"], 4)
        self.assertEqual(summary["rpc_ms"]["p50"], 11)
        self.assertEqual(summary["rpc_ms"]["max"], 30)
        self.assertEqual(summary["transport_ms"]["p50"], 3)
        self.assertEqual(summary["rpc_p50_ms_by_layer"], {"0": 10, "1": 11})
        self.assertIsNone(bl.percentile([], 0.5))
        self.assertEqual(bl.percentile(list(range(1, 101)), 0.99), 99)
        self.assertEqual(bl.percentile(list(range(1, 1001)), 0.999), 999)


class CounterAndTimingTests(unittest.TestCase):
    def test_counters(self):
        self.assertEqual(bl.counter_delta(10, 25, None), 15)
        self.assertEqual(bl.counter_delta(90, 5, 100), 15)
        with self.assertRaises(ValueError):
            bl.counter_delta(90, 5, None)
        series = [(0.0, 0.0), (1.0, 100.0), (2.0, 300.0)]
        self.assertAlmostEqual(bl.energy_between(series, 0.5, 1.5), 150.0)
        self.assertIsNone(bl.energy_between(series, 0.5, 2.5))
        self.assertIsNone(bl.energy_between(series[:1], 0.0, 0.0))
        self.assertAlmostEqual(bl.mean_between([(0, 10), (1, 20), (2, 60)], 0, 1), 15)

    def test_steady_period_and_identity(self):
        times = [0.0, 1.0] + [1.0 + 0.25 * i for i in range(1, 9)]
        self.assertAlmostEqual(bl.steady_period_ms(times, 2), 250.0)
        self.assertIsNone(bl.steady_period_ms(times[:3], 2))
        self.assertEqual(bl.compare_tokens([1, 2, 3], [1, 2, 3])["identical"], True)
        diverged = bl.compare_tokens([1, 2, 3], [1, 9, 3])
        self.assertEqual((diverged["identical"], diverged["first_divergence"]), (False, 1))
        self.assertIsNone(bl.compare_tokens(None, [1])["identical"])


class ProtocolTests(unittest.TestCase):
    def test_sizes_match_protocol_header(self):
        self.assertEqual((bl.HELLO_REQUEST.size, bl.HELLO_RESPONSE.size, bl.EXECUTE_REQUEST.size,
                          bl.EXECUTE_RESPONSE.size), (64, 96, 36, 48))

    def test_fnv_and_hello(self):
        self.assertEqual(bl.fnv1a32(b""), 2166136261)
        self.assertEqual(bl.fnv1a32(b"a"), 0xE40C292C)
        hello = bl.encode_hello(63, 5120, 17408, 4, SHA)
        magic, version, message, mask, n_embd, columns, flags, max_tokens, digest = bl.HELLO_REQUEST.unpack(hello)
        self.assertEqual((magic, version, message, mask, n_embd, columns, flags, max_tokens),
                         (0x46534631, 6, 1, 63, 5120, 17408, 3, 4))
        self.assertEqual(digest.hex(), SHA[7:])
        response = bl.HELLO_RESPONSE.pack(0x46534631, 6, 2, 0, 3, 5120, 17408, 0, 17408, 1, 6, 63, 0xabc, 2176, 4, 0,
                                          bytes.fromhex(SHA[7:]))
        decoded = bl.decode_hello_response(response)
        self.assertTrue(decoded["ok"])
        self.assertEqual((decoded["n_ff"], decoded["offset"], decoded["column_quantum"], decoded["artifact_sha256"]),
                         (17408, 0, 2176, SHA))
        with self.assertRaises(ValueError):
            bl.artifact_digest("sha256:xyz")

    def test_execute_roundtrip(self):
        payload = bl.probe_payload(8, 2, seed=3)
        self.assertEqual(len(payload), 8 * 2 * 2)
        header = bl.encode_execute(7, 5, 2, 8, 4352, payload)
        request = bl.decode_execute_request(header)
        self.assertTrue(request["ok"])
        self.assertEqual((request["request_id"], request["layer"], request["elements"], request["payload_bytes"],
                          request["payload_hash"], request["columns"], request["tokens"]),
                         (7, 5, 16, 32, bl.fnv1a32(payload), 4352, 2))
        with self.assertRaises(ValueError):
            bl.encode_execute(7, 5, 3, 8, 4352, payload)
        response = bl.EXECUTE_RESPONSE.pack(0x46534631, 6, 4, 0, 0, 7, 5, 16, 32, 1, 4352, 2, 8912)
        self.assertEqual(bl.decode_execute_response(response)["compute_us"], 8912)
        values = struct.unpack("<16e", payload)
        self.assertTrue(all(abs(v) <= 0.125 for v in values))


class ModelTests(unittest.TestCase):
    def test_tensor_bytes(self):
        self.assertEqual(bl.tensor_nbytes([5120, 17408], 1), 5120 * 17408 * 2)
        self.assertEqual(bl.tensor_nbytes([5120, 17408], 12), 5120 * 17408 // 256 * 144)
        with self.assertRaises(ValueError):
            bl.tensor_nbytes([100], 12)
        table = bl.layer_byte_table({"blk.0.ffn_up.weight": 10, "blk.0.ffn_gate.weight": 10, "blk.0.ffn_down.weight": 10,
                                     "blk.0.attn_q.weight": 5, "blk.0.attn_norm.weight": 1, "output.weight": 7})
        self.assertEqual(table["layers"][0], {"ffn": 30, "attn": 5, "other": 1})
        self.assertEqual(table["output"], 7)
        fallback = bl.model_bytes("/nonexistent.gguf")
        self.assertEqual(fallback["layers"][0]["ffn"], 3 * 5120 * 17408 * 2)

    def test_gguf_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.gguf")
            with open(path, "wb") as handle:
                def string(text):
                    data = text.encode()
                    return struct.pack("<Q", len(data)) + data
                handle.write(b"GGUF" + struct.pack("<IQQ", 3, 2, 3))
                handle.write(string("general.architecture") + struct.pack("<I", 8) + string("qwen3"))
                handle.write(string("qwen3.block_count") + struct.pack("<II", 4, 1))
                handle.write(string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 100)
                             + b"".join(string("t%d" % i) for i in range(100)))
                handle.write(string("blk.0.ffn_up.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 5120, 17408)
                             + struct.pack("<IQ", 1, 0))
                handle.write(string("output.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 5120, 10)
                             + struct.pack("<IQ", 0, 0))
            metadata, tensors = bl.read_gguf_tensors(path)
            self.assertEqual(metadata["general.architecture"], "qwen3")
            self.assertEqual(tensors, {"blk.0.ffn_up.weight": 5120 * 17408 * 2, "output.weight": 5120 * 10 * 4})

    def test_time_models(self):
        table = {"layers": {0: {"ffn": 100e6, "attn": 0, "other": 0}, 1: {"ffn": 100e6, "attn": 0, "other": 0}},
                 "output": 0}
        phones = {"a": {"bw_gbs": 100.0, "rtt_ms": 0.1}, "b": {"bw_gbs": 100.0, "rtt_ms": 0.1}}
        owners = {0: "a", 1: "b"}
        cpu = bl.predict_cpu_resident_ms(table, [0, 1], 0.0, owners, 100.0, phones)
        self.assertAlmostEqual(cpu["ms"], 2.0)
        overlap = bl.predict_cpu_resident_ms(table, [0, 1], 0.5, owners, 100.0, phones, "overlap")
        self.assertAlmostEqual(overlap["ms"], 1.2)          # 2 x max(0.5, 0.5 + 0.1)
        self.assertEqual(overlap["calls"], 2)
        aggregate = bl.predict_cpu_resident_ms(table, [0, 1], 0.5, owners, 100.0, phones, "aggregate")
        self.assertAlmostEqual(aggregate["ms"], 1.2)        # max(1.0, 0.5) + 2 x 0.1
        skewed = bl.predict_cpu_resident_ms(table, [0, 1], 0.25, owners, 100.0, phones, "overlap")
        self.assertAlmostEqual(skewed["ms"], 1.5)           # 2 x max(0.75, 0.35)
        one_owner = bl.predict_cpu_resident_ms(table, [0, 1], 2 / 3, {0: "a", 1: "a"}, 100.0, phones, "aggregate")
        self.assertAlmostEqual(one_owner["ms"], 200e6 * 2 / 3 / 100e9 * 1e3 + 0.2)
        share, best = bl.best_share(table, [0, 1], owners, 100.0, phones)
        self.assertEqual(share, 0.5)
        with self.assertRaises(ValueError):
            bl.predict_cpu_resident_ms(table, [0], 0.5, owners, 100.0, phones, "bogus")


class PhoneCommandTests(unittest.TestCase):
    def setUp(self):
        self.config = bl.load_config(str(EXAMPLE))

    def test_worker_argv(self):
        op15 = self.config["phones"][0]
        argv = bl.worker_phone_argv(op15, op15["workers"][1], self.config["server"])
        text = " ".join(argv)
        for fragment in ("--backend HTP1", "--layers 6-11", "--port 7072", "--bind 0.0.0.0", "--column-quantum 2176",
                         "--max-tokens 4", "--f16-io", "GGML_HEXAGON_NDEV=3", "S42_RESIDENCY_SESSION_ID=HTP1",
                         "LD_LIBRARY_PATH=/data/local/tmp/s42-ffn-shards-20260904-v1-bin", "qwen/HTP1.ffn.gguf"):
            self.assertIn(fragment, text)
        pixel = self.config["phones"][1]
        script = bl.worker_start_script(pixel, pixel["workers"][0], self.config["server"])
        self.assertIn("flock -n 9", script)
        self.assertIn("S43_PIXEL_UCLAMP_MIN=1024", script)
        self.assertNotIn("'", script)

    def test_commands_quote_cleanly(self):
        lines = bl.worker_commands(self.config, "start")
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith("adb -s 3C15AU002CL00000 shell -T \"su -c '"))
        self.assertTrue(lines[3].startswith("adb -P 5037 -s 5A040DLCH004ES shell -T"))
        self.assertIn("\\$!", lines[0])
        echo = bl.worker_commands(self.config, "echo-start")
        self.assertEqual(len(echo), 2)
        self.assertIn("toybox nc -L -p 7070 cat", echo[0])
        local = bl.worker_commands(self.config, "start", local=True, local_dir="/tmp/x")
        self.assertIn("--bind 127.0.0.1", local[0])
        self.assertIn("--column-quantum 2176", local[0])
        with self.assertRaises(ValueError):
            bl._words(["has space"])


def _write_exec(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class PhoneWorkersScriptTest(unittest.TestCase):
    """phone_workers.sh against a fake adb/su/toybox: exercises the real quoting and the lifecycle."""

    def test_start_status_stop(self):
        with tempfile.TemporaryDirectory(prefix="wifiharness") as tmp:
            tmp = Path(tmp)
            bin_dir = tmp / "bin"
            bin_dir.mkdir()
            _write_exec(bin_dir / "adb", textwrap.dedent("""\
                #!/bin/sh
                while [ "$1" != shell ]; do shift; done; shift
                [ "$1" = -T ] && shift
                exec sh -c "$*"
                """))
            _write_exec(bin_dir / "su", "#!/bin/sh\n[ \"$1\" = -c ] && shift\nexec sh -c \"$1\"\n")
            _write_exec(bin_dir / "toybox", "#!/bin/sh\nexec sleep 30\n")
            worker = tmp / "fake-worker"
            _write_exec(worker, "#!/bin/sh\necho \"[ffn-worker] ready backend=FAKE $*\" >&2\nexec sleep 30\n")
            raw = example()
            for phone in raw["phones"]:
                phone["adb"] = ["adb", "-s", "FAKE-" + phone["name"]]
                phone["log_dir"] = str(tmp / ("logs-" + phone["name"]))
                phone["worker_binary"] = str(worker)
                if phone.get("lock_path"):
                    phone["lock_path"] = str(tmp / ("lock-" + phone["name"]))
            config_path = tmp / "config.json"
            config_path.write_text(json.dumps(raw))
            env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
            script = str(HERE / "phone_workers.sh")

            def run(action):
                done = subprocess.run([script, "--config", str(config_path), action], env=env, capture_output=True,
                                      text=True, timeout=60)
                # drop the "+ <command>" echo lines, keep what the phone side printed
                done.stdout = "\n".join(line for line in done.stdout.splitlines() if not line.startswith("+ "))
                return done
            try:
                started = run("start")
                self.assertEqual(started.returncode, 0, started.stdout + started.stderr)
                self.assertEqual(started.stdout.count("ready backend=FAKE"), 4)
                self.assertIn("--layers 18-23", started.stdout)
                self.assertIn("--backend HTP2", started.stdout)
                again = run("start")
                self.assertEqual(again.stdout.count("already running"), 4)
                status = run("status")
                self.assertEqual(status.stdout.count(" running pid "), 4, status.stdout)
                echo = run("echo-start")
                self.assertEqual(echo.stdout.count("echo server on :7070"), 2, echo.stdout + echo.stderr)
            finally:
                stopped = run("stop")
                run("echo-stop")
            self.assertEqual(stopped.stdout.count(" stopped"), 4, stopped.stdout)
            self.assertEqual(run("status").stdout.count("not running"), 4)


class AnalyzeTests(unittest.TestCase):
    def _result(self, arm, step, joules, helpers=None):
        return {"arm": bl.parse_arm(arm), "power_sources": {"rapl": {"reason": "denied"}},
                "ffn": {"summaries": helpers or []},
                "levels": [{"concurrency": 1, "metrics": {
                    "concurrency": 1, "step_period_ms": step, "ms_per_token": step, "decode_tok_s": 1000 / step,
                    "gpu_decode_mean_w": 100.0, "gpu_decode_j_per_token": joules, "cpu_pkg_decode_j_per_token": None,
                    "identity_vs_cpu": None if arm == "cpu" else {"compared": 5, "identical": 5}}, "waves": []}]}

    def test_report(self):
        config = bl.load_config(str(EXAMPLE))
        config["server"]["model"] = "/nonexistent.gguf"
        summaries = [{"helper": h["label"], "calls": 100, "rpc_p50_ms": 5.0, "compute_p50_ms": 4.5,
                      "host_p50_ms": 5.5, "wait_p50_ms": 0.2} for h in config["helpers"]]
        with tempfile.TemporaryDirectory() as tmp:
            for arm, step, joules, helpers in (("cpu", 270.0, 27.0, None), ("split-25", 225.0, 22.5, summaries)):
                with open(os.path.join(tmp, "RESULT-%s.json" % arm), "w") as handle:
                    json.dump(self._result(arm, step, joules, helpers), handle)
            report, results = analyze.analyze(tmp, config, 600.0, {})
        self.assertIn("| split-25 | 1 | 225.0 |", report)
        self.assertIn("-16.7%", report)
        self.assertIn("5/5", report)
        self.assertIn("overlap", report)
        predicted = analyze.predict(config, [1.5], 69.0, 17.0)
        self.assertIn("| 1.5 | split-50 |", predicted)


if __name__ == "__main__":
    unittest.main()
