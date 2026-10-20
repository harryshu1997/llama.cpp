"""WS21: fused attention-projection path of the OP15 (Hexagon HTP) FFN worker -- shards, contracts, environment,
qualification tool, and (when a host worker is built) the worker's fused numerics on a tiny model."""
from __future__ import annotations

import hashlib
import json
import os
import select
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[2]
for entry in (REPO_ROOT, REPO_ROOT / "gguf-py", REPO_ROOT / "research_dev" / "scheduler" / "native"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import gguf  # noqa: E402
from gguf.constants import GGMLQuantizationType as Q  # noqa: E402

import ffn_shard_gguf as shard_tool  # noqa: E402
import fused_numerics  # noqa: E402

from research_dev.scheduler._internal.plan_contracts.co_helpers import (  # noqa: E402
    RuntimeCoHelperDeclaration,
    RuntimeCoHelperPhone,
    RuntimePlanError,
    phone_helper_rows,
    phone_helpers_support,
    phone_helpers_with_primary_layers,
)
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError  # noqa: E402
from research_dev.scheduler.adapters.ffn_shards import FfnShardIndex, FfnShardIndexError  # noqa: E402
from research_dev.scheduler.adapters.phone_helpers import PhoneHelperBinding, helper_server_environment  # noqa: E402
from research_dev.scheduler.adapters.phone_transport import phone_transport_contract  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import qualify_op15_fused as qualify  # noqa: E402

# opt-in only: a stale host build (e.g. build-cpu/ from before the fused protocol) must never be picked up by default;
# 2026-10-02 it was, and the end-to-end test blocked forever on a worker that does not know the fused request kinds
WORKER = Path(os.environ["LLAMA_FFN_SPLIT_WORKER"]) if os.environ.get("LLAMA_FFN_SPLIT_WORKER") else None
N_LAYER, N_EMBD, N_FF, N_Q, N_KV = 4, 64, 1024, 64, 32
RMS_EPS = 1e-6


def _weights(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(N_EMBD)
    result = {}
    for layer in range(N_LAYER):
        # numpy shape is (ne1, ne0)
        result[f"blk.{layer}.ffn_gate.weight"] = (rng.standard_normal((N_FF, N_EMBD)) * scale).astype(np.float32)
        result[f"blk.{layer}.ffn_up.weight"] = (rng.standard_normal((N_FF, N_EMBD)) * scale).astype(np.float32)
        result[f"blk.{layer}.ffn_down.weight"] = (rng.standard_normal((N_EMBD, N_FF)) * scale).astype(np.float32)
        result[f"blk.{layer}.attn_output.weight"] = (rng.standard_normal((N_EMBD, N_Q)) * scale).astype(np.float32)
        result[f"blk.{layer}.attn_q.weight"] = (rng.standard_normal((N_Q, N_EMBD)) * scale).astype(np.float32)
        result[f"blk.{layer}.attn_k.weight"] = (rng.standard_normal((N_KV, N_EMBD)) * scale).astype(np.float32)
        result[f"blk.{layer}.attn_v.weight"] = (rng.standard_normal((N_KV, N_EMBD)) * scale).astype(np.float32)
        result[f"blk.{layer}.attn_norm.weight"] = (1.0 + 0.1 * rng.standard_normal(N_EMBD)).astype(np.float32)
        result[f"blk.{layer}.ffn_norm.weight"] = (1.0 + 0.1 * rng.standard_normal(N_EMBD)).astype(np.float32)
    result["token_embd.weight"] = rng.standard_normal((128, N_EMBD)).astype(np.float32)
    return result


def _write_model(path: Path, weights: dict[str, np.ndarray]) -> None:
    writer = gguf.GGUFWriter(str(path), "qwen3")
    writer.add_name("tiny-qwen3-ws21")
    writer.add_block_count(N_LAYER)
    writer.add_embedding_length(N_EMBD)
    writer.add_feed_forward_length(N_FF)
    writer.add_layer_norm_rms_eps(RMS_EPS)
    for name, data in weights.items():
        if name.endswith("_norm.weight"):
            writer.add_tensor(name, data.astype(np.float32), raw_dtype=Q.F32)
        else:
            writer.add_tensor(name, data.astype(np.float16), raw_dtype=Q.F16)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


FUNCTIONFS = {
    "ffn_transport": "functionfs-usb", "usb_allocator": "devmem", "usb_queue_depth": 4,
    "usb_concurrent_streams": 4, "usb_max_payload_bytes": 40960, "usb_full_duplex": 1, "usb_split_h2d": 0,
    "usb_slot_safety_bytes": 65536, "usb_vendor_id": 0x18D1, "usb_product_id": 0x4EE7,
    "usb_transport_generation": "functionfs-dmabuf-async-ring-v2", "usb_transport_profile_id": "op15-profile",
    "usbfs_available_bytes": 1 << 30,
}
SHARED = {"S41_SERVER_FFN_LAYER_MASK": str(0b1110), "S41_SERVER_FFN_MAX_TOKENS": "4", "S41_SERVER_FFN_N_EMBD": "5120"}


class FusedShardToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.model = root / "tiny.gguf"
        _write_model(cls.model, _weights())
        cls.parent = _sha256(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def write(self, name, shards, **options):
        out = Path(self.directory.name) / name
        return shard_tool.write_shards(self.model, self.parent, out, [shard_tool.ShardRequest.parse(s) for s in shards],
                                       **options), out

    def test_fused_shard_carries_projections_norms_and_metadata(self):
        index, out = self.write("fused", ["HTP0=1-2:1024"], fused_layers=(1, 2))
        record = index["shards"][0]
        self.assertEqual((record["kernel"], record["fused_layer_mask"], record["fused_next_mask"]),
                         ("attn_proj:fused", f"{0b110:016x}", f"{0b110:016x}"))
        self.assertEqual((record["fused"]["n_q"], record["fused"]["n_kv"]), (N_Q, N_KV))
        self.assertAlmostEqual(record["fused"]["rms_eps"], RMS_EPS)      # float32 in the GGUF
        reader = gguf.GGUFReader(str(out / "HTP0.ffn.gguf"))
        fields = {key: field.contents() for key, field in reader.fields.items() if key.startswith("s43.fused.")}
        self.assertEqual(fields["s43.fused.version"], 1)
        self.assertEqual((fields["s43.fused.layer_mask"], fields["s43.fused.next_mask"]), (0b110, 0b110))
        self.assertEqual((fields["s43.fused.n_q"], fields["s43.fused.n_kv"]), (N_Q, N_KV))
        self.assertAlmostEqual(fields["s43.fused.rms_eps"], RMS_EPS)
        names = {tensor.name for tensor in reader.tensors}
        expected = {f"blk.{l}.{s}.weight" for l in (1, 2) for s in ("ffn_gate", "ffn_up", "ffn_down", "attn_output", "ffn_norm")}
        expected |= {f"blk.{l}.{s}.weight" for l in (2, 3) for s in ("attn_norm", "attn_q", "attn_k", "attn_v")}
        self.assertEqual(names, expected)
        weights = _weights()
        for name in expected:
            tensor = next(t for t in reader.tensors if t.name == name)
            source = weights[name].astype(np.float32 if name.endswith("_norm.weight") else np.float16)
            np.testing.assert_array_equal(np.asarray(tensor.data).reshape(source.shape), source)
        # the index parses with geometry, and the planner spec round-trips through the qualification tool
        loaded = FfnShardIndex.load(out / "FFN_SHARDS.json", "/data/local/tmp/x")
        self.assertEqual((loaded.records[0].fused_n_q, loaded.records[0].fused_n_kv), (N_Q, N_KV))
        self.assertEqual(loaded.records[0].fused_row_elements(N_EMBD), N_EMBD + N_Q + 2 * N_KV)
        self.assertEqual(qualify.shard_fused_masks(out / "FFN_SHARDS.json")[:2], (0b110, 0b110))

    def test_next_layers_default_and_explicit(self):
        index, _ = self.write("last", ["HTP0=2-3:1024"], fused_layers=(2, 3))
        self.assertEqual(index["shards"][0]["fused"]["fused_next_layers"], [2])     # layer 3 has no next layer
        index, out = self.write("plain-next", ["HTP0=1-2:1024"], fused_layers=(1, 2), fused_next_layers=())
        self.assertEqual(index["shards"][0]["fused_next_mask"], f"{0:016x}")
        names = {tensor.name for tensor in gguf.GGUFReader(str(out / "HTP0.ffn.gguf")).tensors}
        self.assertFalse(any(s in n for n in names for s in ("attn_q", "attn_k", "attn_v", "attn_norm")))

    def test_ffn_only_output_has_no_fused_trace_and_refusals(self):
        index, out = self.write("ffn", ["HTP0=1-2:1024", "HTP1=3:512"])
        for record in index["shards"]:
            self.assertNotIn("fused", record)
            self.assertNotIn("kernel", record)
        reader = gguf.GGUFReader(str(out / "HTP0.ffn.gguf"))
        self.assertFalse(any(key.startswith("s43.") for key in reader.fields))
        self.assertTrue(all(".ffn_" in tensor.name and "norm" not in tensor.name for tensor in reader.tensors))
        with self.assertRaisesRegex(shard_tool.FfnShardError, "full FFN width"):
            self.write("half", ["HTP0=1-2:512"], fused_layers=(1,))
        with self.assertRaisesRegex(shard_tool.FfnShardError, "not owned"):
            self.write("foreign", ["HTP0=1-2:1024"], fused_layers=(3,))
        with self.assertRaisesRegex(shard_tool.FfnShardError, "no next layer"):
            self.write("edge", ["HTP0=3:1024"], fused_layers=(3,), fused_next_layers=(3,))
        with self.assertRaisesRegex(shard_tool.FfnShardError, "must be fused"):
            self.write("next-only", ["HTP0=1-2:1024"], fused_layers=(1,), fused_next_layers=(2,))
        self.assertEqual(shard_tool.main([str(self.model), "--parent-sha256", self.parent, "--out-dir",
                                          str(Path(self.directory.name) / "cli"), "--shard", "HTP0=1:1024",
                                          "--fused-next-layers", "1"]), 1)
        # a Gemma-style block (post norms) or a layer with another n_kv is not the WS15 fused math
        weights = _weights()
        weights["blk.1.post_ffw_norm.weight"] = np.ones(N_EMBD, np.float32)
        weights["blk.3.attn_k.weight"] = weights["blk.3.attn_k.weight"][: N_KV // 2]
        other = Path(self.directory.name) / "gemma-like.gguf"
        _write_model(other, weights)
        for layers, message in (((1,), "pre-norm block"), ((2,), "n_kv differs")):
            with self.assertRaisesRegex(shard_tool.FfnShardError, message):
                shard_tool.write_shards(other, _sha256(other), Path(self.directory.name) / f"g{layers[0]}",
                                        [shard_tool.ShardRequest.parse("HTP0=1-2:1024")], fused_layers=layers)

    @unittest.skipUnless(WORKER is not None and WORKER.exists(),
                         "set LLAMA_FFN_SPLIT_WORKER to a fused-capable host llama-ffn-split-worker")
    def test_host_worker_serves_fused_calls_within_tolerance(self):
        _, out = self.write("worker", ["HTP0=1-2:1024"], fused_layers=(1, 2))
        port = _free_port()
        process = subprocess.Popen(
            [str(WORKER), "-m", str(out / "HTP0.ffn.gguf"), "--artifact-sha256", self.parent, "--layers", "1-2",
             "--columns", "1024", "--column-quantum", "256", "--backend", "CPU", "--port", str(port), "--f16-io",
             "--max-tokens", "4"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        try:
            lines, deadline = [], time.monotonic() + 60
            while time.monotonic() < deadline:
                # never block on a worker that stays silent: wait on the pipe with the remaining deadline
                ready, _, _ = select.select([process.stderr], [], [], max(0.0, deadline - time.monotonic()))
                if not ready:
                    break
                line = process.stderr.readline()
                if line:
                    lines.append(line)
                    if "[ffn-worker] fused layers=" in line:
                        break
                elif process.poll() is not None:
                    break
            self.assertTrue(any("S43FUSEDWEIGHTS layers=0000000000000006 next=0000000000000006" in l for l in lines),
                            "".join(lines)[-3000:])
            result = Path(self.directory.name) / "NUMERICS.json"
            code = fused_numerics.main(["--port", str(port), "--proxy", str(self.model), "--artifact-sha256", self.parent,
                                        "--layers", "1-2", "--next-layers", "1-2", "--worker-layers", "1-2",
                                        "--rows", "1,2,4", "--n-embd", str(N_EMBD), "--max-columns", str(N_FF),
                                        "--out", str(result)])
        finally:
            process.kill()
            process.wait(timeout=10)
            process.stderr.close()
        self.assertEqual(code, 0)
        numerics = json.loads(result.read_text())
        self.assertEqual((numerics["status"], numerics["calls"]), ("PASS", 12))
        self.assertLessEqual(numerics["worst_rel_l2"], 1e-3)
        self.assertEqual((numerics["fused_hello"]["n_q"], numerics["fused_hello"]["n_kv"]), (N_Q, N_KV))


class ShardIndexTests(unittest.TestCase):
    def index(self, **extra):
        record = {"path": "HTP0.ffn.gguf", "session_id": "HTP0", "shard_sha256": "sha256:" + "a" * 64,
                  "parent_sha256": "sha256:" + "b" * 64, "layer_mask": f"{0b110:016x}", "columns": 1024, "n_ff": 1024,
                  "shard_bytes": 10, "weight_type": "F16", **extra}
        return {"schema": "s42-ffn-shard-index-v1", "parent_sha256": "sha256:" + "b" * 64, "shards": [record]}

    def test_fused_record_needs_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "FFN_SHARDS.json"
            fused = {"kernel": "attn_proj:fused", "fused_layer_mask": f"{0b110:016x}", "fused_next_mask": f"{0b10:016x}"}
            path.write_text(json.dumps(self.index(**fused, fused={"n_q": 5120, "n_kv": 1024})))
            record = FfnShardIndex.load(path, "/data/local/tmp/x").records[0]
            self.assertEqual((record.fused_n_q, record.fused_n_kv, record.fused_row_elements(5120)), (5120, 1024, 12288))
            path.write_text(json.dumps(self.index(**fused)))      # a Pixel (TCP) record: no geometry
            record = FfnShardIndex.load(path, "/data/local/tmp/x").records[0]
            self.assertEqual((record.fused_n_q, record.fused_row_elements(5120)), (0, 0))
            path.write_text(json.dumps(self.index(**fused, fused={"n_q": 5120})))
            with self.assertRaises(FfnShardIndexError):
                FfnShardIndex.load(path, "/data/local/tmp/x")
            path.write_text(json.dumps(self.index()))
            plain = FfnShardIndex.load(path, "/data/local/tmp/x").records[0]
            self.assertEqual((plain.fused_n_q, plain.fused_row_elements(5120)), (0, 5120))


class DeclarationAndBindingTests(unittest.TestCase):
    def co_helper(self):
        return RuntimeCoHelperPhone(
            device_id="pixel10pro-phone", serial="PX", label="pixel", session_id="PIXEL0", layer_mask=0b110000,
            column_quantum=17408, max_tokens=4, shard_sha256="sha256:" + "d" * 64, resident_bytes=10,
            transport_parameters={"adb_port": 5037, "adb_serial": "PX", "ffn_transport": "adb-tcp",
                                  "ffn_worker_host": "127.0.0.1", "ffn_worker_port": 26991, "phone_worker_port": 27195})

    def test_primary_fused_masks_round_trip_and_reach_the_binding(self):
        plain = RuntimeCoHelperDeclaration("op15", "OP", (self.co_helper(),))
        self.assertEqual(set(plain.to_json()), {"helpers", "primary_label", "primary_serial", "schema"})
        self.assertEqual(RuntimeCoHelperDeclaration.from_json(plain.to_json()), plain)
        fused = RuntimeCoHelperDeclaration("op15", "OP", (self.co_helper(),), primary_fused_layer_mask=0b1110,
                                           primary_fused_next_mask=0b110, primary_fused_n_q=5120, primary_fused_n_kv=1024)
        self.assertEqual(RuntimeCoHelperDeclaration.from_json(json.loads(json.dumps(fused.to_json()))), fused)
        rows = phone_helper_rows(fused.phone_helpers("op15-phone", 0b1111))
        self.assertEqual((rows[0]["fused_layer_mask"], rows[0]["fused_next_mask"], rows[0]["fused_n_q"], rows[0]["fused_n_kv"]),
                         (0b1110, 0b110, 5120, 1024))
        self.assertNotIn("fused_layer_mask", rows[1])
        # the primary's fused layers are clipped to the layers it serves this time
        rows = phone_helper_rows(fused.phone_helpers("op15-phone", 0b11))
        self.assertEqual((rows[0]["fused_layer_mask"], rows[0]["fused_next_mask"]), (0b10, 0b10))
        self.assertEqual(json.loads(plain.phone_helpers("op15-phone", 0b1111))[0],
                         {"device_id": "op15-phone", "label": "op15", "layer_mask": 0b1111, "serial": "OP",
                          "transport_parameters": {}})
        with self.assertRaises(RuntimePlanError):
            RuntimeCoHelperDeclaration("op15", "OP", (self.co_helper(),), primary_fused_layer_mask=0b10)
        with self.assertRaises(RuntimePlanError):
            RuntimeCoHelperDeclaration("op15", "OP", (self.co_helper(),), primary_fused_layer_mask=0b10,
                                       primary_fused_next_mask=0b100, primary_fused_n_q=1, primary_fused_n_kv=1)
        partial = dict(fused.to_json())
        del partial["primary_fused_n_kv"]
        with self.assertRaises(RuntimePlanError):
            RuntimeCoHelperDeclaration.from_json(partial)
        # a live server serves a request only with the fused-ness it was launched with, clipped to the request
        live = fused.phone_helpers("op15-phone", 0b1111)
        self.assertTrue(phone_helpers_support(live, fused.phone_helpers("op15-phone", 0b11)))
        self.assertFalse(phone_helpers_support(live, plain.phone_helpers("op15-phone", 0b11)))
        self.assertFalse(phone_helpers_support(plain.phone_helpers("op15-phone", 0b1111),
                                               fused.phone_helpers("op15-phone", 0b11)))
        clipped = phone_helper_rows(phone_helpers_with_primary_layers(live, 0b1001))
        self.assertEqual((clipped[0]["layer_mask"], clipped[0]["fused_layer_mask"], clipped[0]["fused_next_mask"]),
                         (0b1001, 0b1000, 0))
        self.assertNotIn("fused_layer_mask", phone_helper_rows(phone_helpers_with_primary_layers(live, 0b1))[0])

    def test_binding_geometry_and_json(self):
        binding = PhoneHelperBinding("op15-phone", "OP", 0b1110, "op15", fused_layer_mask=0b110, fused_next_mask=0b10,
                                     fused_n_q=5120, fused_n_kv=1024)
        self.assertEqual(PhoneHelperBinding.from_json(binding.to_json()), binding)
        self.assertEqual(binding.fused_row_elements(5120), 12288)
        no_next = PhoneHelperBinding("op15-phone", "OP", 0b1110, "op15", fused_layer_mask=0b110, fused_n_q=5120, fused_n_kv=1024)
        self.assertEqual(no_next.fused_row_elements(5120), 10240)
        self.assertNotIn("fused_n_q", PhoneHelperBinding("op15-phone", "OP", 0b1110, "op15").to_json())
        with self.assertRaises(PhysicalAdapterError):
            PhoneHelperBinding("op15-phone", "OP", 0b1110, "op15", fused_n_q=5120)
        with self.assertRaises(PhysicalAdapterError):
            PhoneHelperBinding("op15-phone", "OP", 0b1110, "op15", fused_layer_mask=0b110, fused_n_q=5120)


class EnvironmentTests(unittest.TestCase):
    def environment(self, payload: int, fused: bool = True, next_mask: int = 0b10):
        contract = phone_transport_contract({**FUNCTIONFS, "usb_max_payload_bytes": payload})
        extra = {"fused_layer_mask": 0b110, "fused_next_mask": next_mask, "fused_n_q": 5120, "fused_n_kv": 1024} if fused else {}
        binding = PhoneHelperBinding("op15-phone", "OP", 0b1110, "op15", **extra)
        return helper_server_environment((binding,), SHARED, (contract,))

    def test_fused_primary_over_functionfs_needs_slots_for_the_fused_rows(self):
        env = self.environment(12288 * 4 * 2)
        self.assertEqual((env["S41_SERVER_FFN_FUSED_LAYER_MASK"], env["S41_SERVER_FFN_FUSED_NEXT_MASK"]), ("6", "2"))
        self.assertEqual(env["S41_SERVER_FFN_TRANSPORT"], "functionfs-usb")
        with self.assertRaisesRegex(PhysicalAdapterError, "hold 40960 bytes, fused calls need 98304"):
            self.environment(40960)
        self.assertIn("S41_SERVER_FFN_FUSED_LAYER_MASK", self.environment(10240 * 4 * 2, next_mask=0))
        with self.assertRaisesRegex(PhysicalAdapterError, "need the decode shape"):
            binding = PhoneHelperBinding("op15-phone", "OP", 0b1110, "op15", fused_layer_mask=0b110)
            helper_server_environment((binding,), SHARED, (phone_transport_contract(FUNCTIONFS),))
        plain = self.environment(40960, fused=False)
        self.assertFalse(any("FUSED" in key for key in plain))
        with self.assertRaisesRegex(PhysicalAdapterError, "TCP or FunctionFS"):
            contract = phone_transport_contract({"ffn_transport": "tcp", "bridge_allocator": "malloc",
                                                 "ffn_bridge_host": "h", "ffn_bridge_port": 1, "bridge_queue_depth": 1})
            binding = PhoneHelperBinding("op15-phone", "OP", 0b1110, "op15", fused_layer_mask=0b110, fused_next_mask=0b10,
                                         fused_n_q=5120, fused_n_kv=1024)
            helper_server_environment((binding,), SHARED, (contract,))


class QualificationToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.model = self.root / "tiny.gguf"
        _write_model(self.model, _weights())
        self.parent = _sha256(self.model)
        shard_tool.write_shards(self.model, self.parent, self.root / "shards",
                                [shard_tool.ShardRequest.parse("HTP0=1-2:1024")], fused_layers=(1, 2))
        self.index = self.root / "shards" / "FFN_SHARDS.json"
        rows, layers = [1, 2, 4], [1, 2]
        cases = [{"layer": l, "next_qkv": n, "rows": r, "repeat": 0, "rpc_ms": 1.0, "compute_ms": 0.5,
                  "segments": {"delta": {"finite": True, "rel_l2": 2e-4}}}
                 for l in layers for n in (True, False) for r in rows]
        self.numerics = {"schema": "s43-fused-numerics-v1", "status": "PASS", "worst_rel_l2": 4e-4, "rows": rows,
                         "layers": layers, "next_layers": layers, "calls": 12, "cases": cases, "io": "f16", "repeats": 1,
                         "fused_hello": {"fused_layer_mask": f"{0b110:016x}", "next_qkv_mask": f"{0b110:016x}",
                                         "n_q": N_Q, "n_kv": N_KV, "fused_weight_hash": "0" * 16}}
        self.tokens = {"schema": "s43-token-identity-v1",
                       "runs": [{"prompt_sha256": f"{i:064x}", "tokens": list(range(i, i + 20))} for i in range(4)]}
        self.stderr = f"S41SERVERFFNFUSED layers={0b110:016x} next={0b110:016x}\nS41SERVERFFNFUSEDCALL layer=1 tokens=1\n"

    def tearDown(self):
        self.directory.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return str(path)

    def run_tool(self, numerics=None, fused_tokens=None, stderr=None, output="OUT.json"):
        return qualify.main(["--shard-index", str(self.index), "--model", "tiny", "--numerics",
                             self.write("NUMERICS.json", numerics or self.numerics),
                             "--fused-tokens", self.write("fused.json", fused_tokens or self.tokens),
                             "--plain-tokens", self.write("plain.json", self.tokens),
                             "--fused-server-stderr", self.write("server.stderr", stderr or self.stderr),
                             "--output", str(self.root / output)])

    def test_receipt_and_refusals(self):
        self.assertEqual(self.run_tool(), 0)
        receipt = json.loads((self.root / "OUT.json").read_text())
        self.assertEqual((receipt["status"], receipt["qualified_layers"], receipt["fused_next_mask"]), ("PASS", "1-2", f"{0b110:016x}"))
        self.assertEqual(receipt["planner_overrides"], ["--kernels op15-phone=ffn,attn_proj:fused",
                                                        "--qualified-kernel op15-phone:tiny:attn_proj:fused=1-2"])
        self.assertEqual(receipt["token_identity"]["fused_calls_logged"], 1)
        self.assertEqual(self.run_tool(), 1)                                  # output exists
        self.assertEqual(self.run_tool({**self.numerics, "status": "FAIL"}, output="a.json"), 1)
        self.assertEqual(self.run_tool({**self.numerics, "layers": [1], "calls": 9, "cases": self.numerics["cases"][:9]},
                                       output="b.json"), 1)
        different = json.loads(json.dumps(self.tokens))
        different["runs"][2]["tokens"][5] += 1
        self.assertEqual(self.run_tool(fused_tokens=different, output="c.json"), 1)
        self.assertEqual(self.run_tool(stderr=f"S41SERVERFFNFUSED layers={0b10:016x} next={0b10:016x}\nS41SERVERFFNFUSEDCALL x\n",
                                       output="d.json"), 1)
        self.assertEqual(self.run_tool(stderr=f"S41SERVERFFNFUSED layers={0b110:016x} next={0b110:016x}\n", output="e.json"), 1)
        self.assertEqual(qualify.layer_spec(0b1011011), "0-1,3-4,6")


if __name__ == "__main__":
    unittest.main()
