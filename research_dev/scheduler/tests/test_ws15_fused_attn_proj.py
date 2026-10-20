"""WS15 fused attention projections: shards, planner kernel qualification, contracts, evidence, numerics tool.

The C++ path (worker fused call, server graph, client) is exercised end to end on hardware-free hosts by
``fused_numerics.py`` and the host e2e harness; these tests pin the Python side: what a fused shard carries, how the
planner restricts and prices the kernel, how fused masks reach the server environment, and what evidence a
fused helper needs.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import MappingProxyType

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[2]
for path in (str(REPO_ROOT), str(TESTS_DIR), str(REPO_ROOT / "gguf-py"),
             str(REPO_ROOT / "research_dev" / "scheduler" / "native")):
    if path not in sys.path:
        sys.path.insert(0, path)

import gguf  # noqa: E402
from gguf.constants import GGMLQuantizationType as Q  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402

import fused_numerics  # noqa: E402
import pixel_packed_shard as shard_tool  # noqa: E402
from research_dev.scheduler._internal.layer_placement import (  # noqa: E402
    PlacementProblem, restrict_to_executable,
)
from research_dev.scheduler._internal import layer_placement_opgroups as og  # noqa: E402
from research_dev.scheduler._internal.layer_placement_io import (  # noqa: E402
    device_from_json, device_to_json, fused_shape_observations, parse_fused_shape_line,
)
from research_dev.scheduler._internal.plan_contracts.co_helpers import (  # noqa: E402
    RuntimeCoHelperDeclaration, RuntimeCoHelperPhone, RuntimePlanError, phone_helper_rows,
)
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError  # noqa: E402
from research_dev.scheduler.adapters.ffn_shards import FfnShardIndex, FfnShardIndexError  # noqa: E402
from research_dev.scheduler.adapters.phone_helpers import (  # noqa: E402
    PhoneHelperBinding, helper_server_environment,
)
from research_dev.scheduler.adapters.phone_transport import phone_transport_contract  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.helper_phone_evidence import load_helper_evidence  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import layer_placement as placement_cli  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import prepare_aoa_evidence as aoa_tool  # noqa: E402

import test_layer_placement_opgroups as opg  # noqa: E402
import test_ws12_aoa_rebind_composition as ws12  # noqa: E402

PARENT = "sha256:" + "ab" * 32


class FusedShardTests(unittest.TestCase):
    """pixel_packed_shard.py --fused-layers: own O + ffn_norm, next attn_norm/Q/K/V, exact copies, metadata."""
    N_LAYER, N_EMBD, N_FF, N_KV = 4, 256, 512, 64

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(15)
        origin, proxy = {}, {}

        def add(name, shape, qtype):
            values = rng.standard_normal(shape).astype(np.float32)
            if qtype == Q.F32:
                origin[name] = (values, Q.F32)
                proxy[name] = (values, Q.F32)
                return
            packed = quantize(values, qtype)
            origin[name] = (packed, qtype)
            proxy[name] = (dequantize(packed, qtype).astype(np.float16), Q.F16)

        for layer in range(self.N_LAYER):
            add(f"blk.{layer}.ffn_gate.weight", (self.N_FF, self.N_EMBD), Q.Q4_0)
            add(f"blk.{layer}.ffn_up.weight", (self.N_FF, self.N_EMBD), Q.Q4_0)
            add(f"blk.{layer}.ffn_down.weight", (self.N_EMBD, self.N_FF), Q.Q8_0)
            add(f"blk.{layer}.attn_q.weight", (self.N_EMBD, self.N_EMBD), Q.Q4_0)
            add(f"blk.{layer}.attn_k.weight", (self.N_KV, self.N_EMBD), Q.Q4_0)
            add(f"blk.{layer}.attn_v.weight", (self.N_KV, self.N_EMBD), Q.Q8_0)
            add(f"blk.{layer}.attn_output.weight", (self.N_EMBD, self.N_EMBD), Q.Q4_0)
            add(f"blk.{layer}.attn_norm.weight", (self.N_EMBD,), Q.F32)
            add(f"blk.{layer}.ffn_norm.weight", (self.N_EMBD,), Q.F32)
        self.origin, self.proxy = self.tmp / "origin.gguf", self.tmp / "proxy.gguf"
        for path, tensors in ((self.origin, origin), (self.proxy, proxy)):
            writer = gguf.GGUFWriter(str(path), "qwen3")
            writer.add_block_count(self.N_LAYER)
            writer.add_embedding_length(self.N_EMBD)
            writer.add_feed_forward_length(self.N_FF)
            writer.add_float32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
            for name, (data, qtype) in tensors.items():
                writer.add_tensor(name, data, raw_dtype=qtype)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file(progress=False)
            writer.close()

    def test_fused_shard_carries_own_o_and_next_qkv_with_metadata(self):
        out = self.tmp / "shard" / "F.ffn.gguf"
        manifest = shard_tool.build(self.origin, self.proxy, (1, 2), out, parent_sha256=PARENT,
                                    allowed_types=("Q4_0", "Q8_0"), fused_layers=(1, 2), fused_next_layers=(1,))
        self.assertEqual(manifest["status"], "PASS")
        names = {tensor.name for tensor in gguf.GGUFReader(str(out)).tensors}
        self.assertEqual(names - {f"blk.{l}.ffn_{n}.weight" for l in (1, 2) for n in ("gate", "up", "down")}, {
            "blk.1.attn_output.weight", "blk.1.ffn_norm.weight", "blk.2.attn_output.weight", "blk.2.ffn_norm.weight",
            "blk.2.attn_norm.weight", "blk.2.attn_q.weight", "blk.2.attn_k.weight", "blk.2.attn_v.weight"})
        fields = gguf.GGUFReader(str(out)).fields
        value = lambda key: fields[key].contents()  # noqa: E731
        self.assertEqual((value("s43.fused.layer_mask"), value("s43.fused.next_mask"), value("s43.fused.n_q"),
                          value("s43.fused.n_kv")), (0b110, 0b10, self.N_EMBD, self.N_KV))
        self.assertAlmostEqual(value("s43.fused.rms_eps"), 1e-6)
        record = json.loads(out.with_name(out.name + ".index-record.json").read_text())
        self.assertEqual((record["kernel"], record["fused_layer_mask"], record["fused_next_mask"]),
                         ("attn_proj:fused", f"{0b110:016x}", f"{0b10:016x}"))
        self.assertTrue(all(row["exact_dequantization"]["mismatches"] == 0 for row in manifest["tensors"]))
        index = FfnShardIndex.load(out.with_name("PIXEL_FFN_SHARDS.json"), "/phone")
        self.assertEqual((index.records[0].kernel, index.records[0].fused_layer_mask, index.records[0].fused_next_mask),
                         ("attn_proj:fused", 0b110, 0b10))

    def test_ffn_only_shard_is_unchanged_and_refusals(self):
        plain = shard_tool.build(self.origin, self.proxy, (1,), self.tmp / "p" / "P.ffn.gguf", parent_sha256=PARENT,
                                 allowed_types=("Q4_0", "Q8_0"))
        self.assertNotIn("fused", plain)
        record = json.loads((self.tmp / "p" / "P.ffn.gguf.index-record.json").read_text())
        self.assertNotIn("kernel", record)
        with self.assertRaisesRegex(shard_tool.PackedShardError, "subset"):
            shard_tool.build(self.origin, self.proxy, (1,), self.tmp / "x" / "X.ffn.gguf", fused_layers=(2,),
                             parent_sha256=PARENT, allowed_types=("Q4_0", "Q8_0"))
        with self.assertRaisesRegex(shard_tool.PackedShardError, "below the last block"):
            shard_tool.build(self.origin, self.proxy, (3,), self.tmp / "y" / "Y.ffn.gguf", fused_layers=(3,),
                             fused_next_layers=(3,), parent_sha256=PARENT, allowed_types=("Q4_0", "Q8_0"))


class ShardIndexTests(unittest.TestCase):
    def index(self, **record):
        row = {"path": "S.ffn.gguf", "parent_sha256": PARENT, "shard_sha256": "sha256:" + "c" * 64,
               "layer_mask": f"{0b1110:016x}", "columns": 128, "n_ff": 128, "shard_bytes": 10,
               "weight_type": "Q4_K", "session_id": "PIXEL0", **record}
        return FfnShardIndex.from_index({"schema": "s42-ffn-shard-index-v1", "parent_sha256": PARENT,
                                         "shards": [row]}, "/phone")

    def test_fused_records_parse_and_inconsistent_ones_are_refused(self):
        self.assertEqual(self.index().records[0].kernel, "ffn")
        fused = self.index(kernel="attn_proj:fused", fused_layer_mask=f"{0b110:016x}",
                           fused_next_mask=f"{0b10:016x}").records[0]
        self.assertEqual((fused.fused_layer_mask, fused.fused_next_mask), (0b110, 0b10))
        for change in ({"kernel": "attn_proj:fused"},                                 # kernel without layers
                       {"fused_layer_mask": f"{0b10:016x}"},                           # layers without kernel
                       {"kernel": "attn_proj:fused", "fused_layer_mask": f"{0b10000:016x}"},   # not its layers
                       {"kernel": "attn_proj:fused", "fused_layer_mask": f"{0b10:016x}",
                        "fused_next_mask": f"{0b100:016x}"},                             # next outside fused
                       {"kernel": "attn_proj:fused", "fused_layer_mask": f"{0b10:016x}", "columns": 64},
                       {"kernel": "lm_head"}):
            with self.subTest(change=change), self.assertRaises(FfnShardIndexError):
                self.index(**change)


class PlannerKernelTests(unittest.TestCase):
    def setUp(self):
        self.profile = opg.profile(phone=opg.priors())
        self.inputs = opg.inputs()

    def problem(self, **device_extra):
        devices = (opg.device("phone", kernels=frozenset({"ffn", "attn_proj:fused"}),
                              stored_layers=MappingProxyType({"m": 0b1111}),
                              qualified_layers=MappingProxyType({"m": 0b1111}), **device_extra),)
        return PlacementProblem((opg.model(),), devices, self.profile)

    def fused_layers(self, problem):
        adjusted, _ = og._adjusted(problem, self.inputs)
        return {layer for layer in range(4)
                if any(o.variant == "proj-fused" for o in og.opgroup_options(adjusted, self.inputs,
                                                                               adjusted.models[0], layer, 1.0))}

    def test_executable_fused_needs_per_kernel_qualification(self):
        self.assertEqual(self.fused_layers(self.problem()), {0, 1, 2, 3})               # ideal: any layer
        self.assertEqual(self.fused_layers(restrict_to_executable(self.problem())), set())   # no fused evidence
        qualified = MappingProxyType({"attn_proj:fused": MappingProxyType({"m": 0b0110})})
        executable = restrict_to_executable(self.problem(kernel_qualified_layers=qualified))
        self.assertEqual(self.fused_layers(executable), {1, 2})
        # the FFN itself stays executable on every stored+qualified layer
        self.assertEqual(executable.devices[0].allowed_layers["m"], 0b1111)

    def test_identity_tool_counts_a_fused_call_once(self):
        """2026-10-05: the rebuilt server prints a fused call as a proof line (fused_width=) AND a pricing
        line; the identity tool must not count it twice (it did: 5704 for 2852 calls)."""
        from research_dev.scheduler.campaigns.burstgpt.tools.qualify_pixel_server_transport import served_call_count
        proof_new = ["S41SERVERFFNCALL context=70:0:1:2 request=1 layer=1 tokens=1 columns=17408 payload_bytes=20480 fused_width=10240"] * 3
        pricing = ["S41SERVERFFNFUSEDCALL request=1 layer=1 tokens=1 next_qkv=1 rpc_ms=8.9 compute_ms=7.6"] * 3
        plain = ["S41SERVERFFNCALL request=9 layer=0 tokens=1 columns=17408 payload_bytes=10240"] * 2
        self.assertEqual(served_call_count(proof_new, pricing), 3)            # rebuilt server: proof lines only
        self.assertEqual(served_call_count(plain + proof_new, pricing), 5)
        self.assertEqual(served_call_count(plain, pricing), 5)                # older server: plain proofs + pricing lines
        self.assertEqual(served_call_count([], pricing), 3)

    def test_measured_fused_call_prices_the_unit(self):
        problem, _ = og._adjusted(self.problem(), self.inputs)
        option = lambda: next(o for o in og.opgroup_options(problem, self.inputs, problem.models[0], 1, 1.0)  # noqa: E731
                              if o.variant == "proj-fused")
        scaled = option()
        host = og._host_costs(problem, self.inputs, problem.models[0], 1)
        # layer 1's unit carries O(1) and Q/K/V(2): stored under the FFN format with the +next suffix
        self.profile.observe_fused_call_summary("phone", "m", "f16+next", "usb", 1, 500, 21.0, 19.5, "e2e")
        self.profile.observe_fused_call_summary("phone", "m", "f16+next", "usb", 2, 500, 25.0, 23.0, "e2e")
        measured = option()
        self.assertAlmostEqual(measured.ms_by_rows[1], 21.0 + host.kv_ms[1])
        self.assertNotAlmostEqual(measured.ms_by_rows[1], scaled.ms_by_rows[1])
        self.assertEqual(self.profile.fused_call("phone", "m", "f16+next", "usb", 1).rpc_ms, 21.0)

    def test_profile_and_device_json_round_trip_and_stay_byte_identical_without_ws15(self):
        plain = self.profile.to_json()
        self.assertNotIn("fused_calls", plain)
        self.profile.observe_fused_call_summary("phone", "m", "f16", "usb", 1, 3, 9.0, 8.0)
        copy = type(self.profile).from_json(self.profile.to_json())
        self.assertEqual(copy.fused_call("phone", "m", "f16", "usb", 1).compute_ms, 8.0)
        device = self.problem().devices[0]
        self.assertNotIn("kernel_qualified_layers", device_to_json(device))
        qualified = self.problem(kernel_qualified_layers=MappingProxyType(
            {"attn_proj:fused": MappingProxyType({"m": 0b0110})})).devices[0]
        row = device_to_json(qualified)
        self.assertEqual(row["kernel_qualified_layers"], {"attn_proj:fused": {"m": "1-2"}})
        self.assertEqual(device_from_json(row).kernel_qualified_layers["attn_proj:fused"]["m"], 0b0110)

    def test_fused_shape_lines_and_inventory_overrides(self):
        lines = ['x S41SERVERFFNFUSEDSHAPE {"helper":"pixel","kind":"attn_proj:fused","tokens":1,"next_qkv":true,'
                 '"calls":10,"rpc_mean_ms":8.0,"rpc_p50_ms":8.0,"compute_mean_ms":7.0,"compute_p50_ms":7.0}',
                 'S41SERVERFFNFUSEDSHAPE {"helper":"pixel","kind":"attn_proj:fused","tokens":2,"next_qkv":false,'
                 '"calls":4,"rpc_mean_ms":9.0,"rpc_p50_ms":9.0,"compute_mean_ms":8.0,"compute_p50_ms":8.0}',
                 'S41SERVERFFNSHAPE {"tokens":1,"columns":17408,"calls":3,"rpc_mean_ms":1,"compute_mean_ms":1}']
        rows = fused_shape_observations(lines, model_id="m", helpers={
            "pixel": {"device_id": "phone", "format": "pk", "transport": "aoa-bridge"}})
        self.assertEqual([(row["shard_format"], row["rows"], row["calls"]) for row in rows],
                         [("pk+next", 1, 10), ("pk", 2, 4)])
        self.assertIsNone(parse_fused_shape_line(lines[2]))
        self.assertEqual(placement_cli.add_fused_shape_rows(self.profile, rows), 2)
        inventory = {"devices": [device_to_json(self.problem().devices[0])]}
        changed = placement_cli.override_inventory(inventory, kernels=["phone=ffn,attn_proj:fused"],
                                                   qualified_kernel=["phone:m:attn_proj:fused=1-2"])
        self.assertEqual(changed["devices"][0]["kernel_qualified_layers"], {"attn_proj:fused": {"m": "1-2"}})
        with self.assertRaises(ValueError):
            placement_cli.override_inventory(inventory, qualified_kernel=["phone:m:ffn=1-2"])


class ContractAndEnvironmentTests(unittest.TestCase):
    def co_helper(self, **fused):
        return RuntimeCoHelperPhone(
            device_id="pixel10pro-phone", serial="PX", label="pixel", session_id="PIXEL0", layer_mask=0b1110,
            column_quantum=17408, max_tokens=4, shard_sha256="sha256:" + "d" * 64, resident_bytes=10,
            transport_parameters={"adb_port": 5037, "adb_serial": "PX", "ffn_transport": "adb-tcp",
                                  "ffn_worker_host": "127.0.0.1", "ffn_worker_port": 26991,
                                  "phone_worker_port": 27195}, **fused)

    def test_co_helper_and_binding_json_carry_fused_masks_only_when_set(self):
        plain = self.co_helper()
        self.assertNotIn("fused_layer_mask", plain.to_json())
        fused = self.co_helper(fused_layer_mask=0b110, fused_next_mask=0b10)
        self.assertEqual(RuntimeCoHelperPhone.from_json(fused.to_json()), fused)
        binding = RuntimeCoHelperDeclaration("op15", "OP", (fused,)).phone_helpers("op15-phone", 0b1)
        rows = phone_helper_rows(binding)
        self.assertEqual((rows[1]["fused_layer_mask"], rows[1]["fused_next_mask"]), (0b110, 0b10))
        self.assertNotIn("fused_layer_mask", json.loads(
            RuntimeCoHelperDeclaration("op15", "OP", (plain,)).phone_helpers("op15-phone", 0b1))[1])
        with self.assertRaises(RuntimePlanError):
            self.co_helper(fused_layer_mask=0b10000)
        with self.assertRaises(RuntimePlanError):
            self.co_helper(fused_layer_mask=0b10, fused_next_mask=0b100)
        bad = json.loads(binding)
        bad[1]["fused_layer_mask"] = 0b10000
        with self.assertRaises(RuntimePlanError):
            phone_helper_rows(json.dumps(bad))

    def test_server_environment_has_per_helper_fused_keys(self):
        parameters = dict(self.co_helper().transport_parameters)
        helpers = (PhoneHelperBinding("op15-phone", "OP", 0b1, "op15"),
                   PhoneHelperBinding("pixel10pro-phone", "PX", 0b1110, "pixel", parameters,
                                      fused_layer_mask=0b110, fused_next_mask=0b10))
        contracts = (phone_transport_contract({"ffn_transport": "adb-tcp", "adb_serial": "OP", "adb_port": 5037,
                                               "ffn_worker_host": "127.0.0.1", "ffn_worker_port": 26990,
                                               "phone_worker_port": 27195}),
                     phone_transport_contract(parameters))
        env = helper_server_environment(helpers, {"S41_SERVER_FFN_LAYER_MASK": str(0b1111)}, contracts)
        self.assertEqual((env["S41_SERVER_FFN_HELPER1_FUSED_LAYER_MASK"], env["S41_SERVER_FFN_HELPER1_FUSED_NEXT_MASK"]),
                         ("6", "2"))
        self.assertFalse(any("FUSED" in key for key in env if key.startswith("S41_SERVER_FFN_HELPER0_")))
        plain = helper_server_environment(helpers[:1] + (PhoneHelperBinding(
            "pixel10pro-phone", "PX", 0b1110, "pixel", parameters),), {"S41_SERVER_FFN_LAYER_MASK": "15"}, contracts)
        self.assertFalse(any("FUSED" in key for key in plain))
        self.assertEqual(PhoneHelperBinding.from_json(helpers[1].to_json()), helpers[1])


class FusedRebindTests(unittest.TestCase):
    """rebind-shard on a fused shard: fused numerics from the identity run, re-pinned server, fused evidence."""

    def setUp(self):
        self.base = ws12.RebindAndComposeTests(methodName="test_rebind_swaps_only_the_shard_and_its_receipts")
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        root = self.base.root
        self.fused, self.next = ws12.NEW_MASK, ws12.NEW_MASK & ~(1 << max(ws12.NEW_LAYERS))
        index = json.loads((root / "PIXEL_FFN_SHARDS.json").read_text())
        index["shards"][0].update(kernel="attn_proj:fused", fused_layer_mask=f"{self.fused:016x}",
                                  fused_next_mask=f"{self.next:016x}")
        (root / "PIXEL_FFN_SHARDS.json").write_text(json.dumps(index))
        server = Path(self.base.args["server_identity"])
        result = json.loads((server / "RESULT.json").read_text())
        result.update(call_columns={"128": 0}, fused_calls=40)
        (server / "RESULT.json").write_text(json.dumps(result))
        self.numerics = {
            "schema": "s43-fused-numerics-v1", "status": "PASS", "worst_rel_l2": 4e-4, "rows": [1, 2, 4],
            "layers": list(ws12.NEW_LAYERS), "next_layers": [l for l in ws12.NEW_LAYERS if self.next >> l & 1],
            "calls": 27, "inputs": "synthetic seed=1",
            "fused_hello": {"fused_layer_mask": f"{self.fused:016x}", "fused_weight_hash": "0123456789abcdef"}}
        (server / "NUMERICS.json").write_text(json.dumps(self.numerics))
        self.server = server

    def rebind(self, **change):
        return self.base.rebind(**{"numerical_dir": None, "carry_numerical": False, "host_software": [], **change})

    def test_fused_rebind_writes_fused_evidence(self):
        evidence = load_helper_evidence(self.rebind())
        self.assertEqual(evidence.fused, {"kernel": "attn_proj:fused", "fused_layer_mask": self.fused,
                                          "fused_next_mask": self.next, "fused_weight_hash": "0123456789abcdef"})
        raw = json.loads((self.base.root / "rebound" / "PIXEL_EVIDENCE_AOA.json").read_text())
        receipt = json.loads(Path(raw["receipt_paths"]["numerical-rows-1-2-4"]).read_text())
        self.assertEqual((receipt["kernel"], receipt["fused_layer_mask"]), ("attn_proj:fused", self.fused))
        self.assertEqual(raw["rebound_from"]["numerical"], "fused-numerics")

    def test_fused_rebind_repins_the_fused_worker_binary(self):
        """2026-10-02 rig: the fused kernel ships in a new worker binary (57577797...); --worker re-pins it to the
        file the identity run used and verified, and the fused numerics of that same run back it."""
        new_worker, new_sha = "/phone/fused/llama-ffn-split-worker", "sha256:" + "d" * 64
        config = json.loads((self.server / "CONFIG.json").read_text())
        (self.server / "CONFIG.json").write_text(json.dumps({**config, "phone_worker": new_worker}))
        identity = json.loads((self.server / "IDENTITY.json").read_text())
        (self.server / "IDENTITY.json").write_text(json.dumps(
            {**identity, "phone_hashes": identity["phone_hashes"] + f"{new_sha[7:]}  {new_worker}\n"}))
        evidence = load_helper_evidence(self.rebind(worker=f"{new_worker}={new_sha}"))
        self.assertEqual((evidence.worker.worker_path, evidence.worker.expected_sha256_by_path[new_worker]),
                         (new_worker, new_sha))
        self.assertEqual(evidence.fused["kernel"], "attn_proj:fused")

    def test_fused_rebind_refusals(self):
        with self.assertRaisesRegex(PhysicalAdapterError, "cannot carry"):
            self.rebind(carry_numerical=True)
        for change, message in (({"layers": [ws12.NEW_LAYERS[0]]}, "numerical rows"),
                                ({"status": "FAIL"}, "numerical rows"),
                                ({"worst_rel_l2": 2e-3}, "numerical rows")):
            with self.subTest(change=change):
                (self.server / "NUMERICS.json").write_text(json.dumps({**self.numerics, **change}))
                with self.assertRaisesRegex(PhysicalAdapterError, message):
                    self.rebind()
                self.assertFalse((self.base.root / "rebound").exists())
        (self.server / "NUMERICS.json").write_text(json.dumps(self.numerics))
        result = json.loads((self.server / "RESULT.json").read_text())
        (self.server / "RESULT.json").write_text(json.dumps({**result, "fused_calls": 0}))
        with self.assertRaisesRegex(PhysicalAdapterError, "no fused Pixel calls"):
            self.rebind()

    def test_a_rebuilt_server_at_the_same_path_is_re_hashed(self):
        """2026-10-04: the deploy's llama-server was rebuilt in place (same path, new libraries); rebind-shard kept the
        bundle's old digest because only a changed PATH was re-hashed, and load_helper_evidence refused the bundle."""
        root = self.base.root
        raw = json.loads(self.base.evidence.read_text())
        impl = Path(raw["host_software_paths"]["host_impl_sha256"])
        impl.write_bytes(impl.read_bytes() + b"rebuilt")          # same path, new content
        server = Path(raw["host_software_paths"]["host_binary_sha256"])
        (self.server / "IDENTITY.json").write_text(json.dumps({
            **json.loads((self.server / "IDENTITY.json").read_text()), "server_sha256": ws12.digest(server)[7:]}))
        # the fixture pins every host key to one file, so the in-place rebuild touches all three pinned paths
        repin = [f"{key}={raw['host_software_paths'][key]}" for key in raw["host_software_paths"]]
        path = self.rebind(host_software=repin, output_dir=str(root / "rebuilt"))
        bundle = json.loads(path.read_text())
        self.assertEqual(bundle["transport_identity"]["software_identity"]["host_impl_sha256"], ws12.digest(impl))
        self.assertEqual(bundle["host_software_paths"]["host_impl_sha256"], str(impl))
        load_helper_evidence(path)

    def test_fused_server_is_re_pinned(self):
        new_server = self.base.root / "server-fused"
        new_server.write_bytes(b"fused llama-server")
        identity = json.loads((self.server / "IDENTITY.json").read_text())
        identity["server_sha256"] = ws12.digest(new_server)[7:]
        (self.server / "IDENTITY.json").write_text(json.dumps(identity))
        with self.assertRaisesRegex(PhysicalAdapterError, "another server"):
            self.rebind()
        path = self.rebind(host_software=[f"host_binary_sha256={new_server}"])
        evidence = load_helper_evidence(path)
        self.assertEqual(evidence.identity.software_identity["host_binary_sha256"], ws12.digest(new_server))
        with self.assertRaisesRegex(PhysicalAdapterError, "pinned host file"):
            self.base.rebind(numerical_dir=None, carry_numerical=False, output_dir=str(self.base.root / "other"),
                             host_software=["not_pinned=/x"])

    def test_identity_config_turns_on_fused_legs_and_numerics(self):
        base = json.loads((self.server / "CONFIG.json").read_text())
        base.update(expected_phone_sha256={}, phone_model="/phone/old.gguf", server_environment={"A": "1"})
        base["phone"].update(columns=128)
        (self.base.root / "BASE.json").write_text(json.dumps(base))
        out = aoa_tool.identity_config(argparse.Namespace(
            base=str(self.base.root / "BASE.json"), shard_index=str(self.base.root / "PIXEL_FFN_SHARDS.json"),
            phone_shard_path=ws12.NEW_PATH, output=str(self.base.root / "IDENTITY_CONFIG.json")))
        config = json.loads(out.read_text())
        self.assertEqual(config["legs_columns"], [0, 128, 128, 0])
        self.assertEqual(config["server_environment"]["S41_SERVER_FFN_FUSED_LAYER_MASK"], str(self.fused))
        self.assertEqual(config["fused_numerics"]["layers"], ",".join(map(str, ws12.NEW_LAYERS)))


class NumericsToolTests(unittest.TestCase):
    """fused_numerics.py against a fake worker that computes the reference itself (protocol framing + verdict)."""

    def test_framing_and_verdict(self):
        n_embd, n_q, n_kv, n_ff = 256, 256, 64, 512
        rng = np.random.default_rng(3)
        weights = {"blk.1.attn_output.weight": rng.normal(0, 0.05, (n_embd, n_q)),
                   "blk.1.ffn_norm.weight": rng.normal(1, 0.1, n_embd),
                   "blk.1.ffn_gate.weight": rng.normal(0, 0.05, (n_ff, n_embd)),
                   "blk.1.ffn_up.weight": rng.normal(0, 0.05, (n_ff, n_embd)),
                   "blk.1.ffn_down.weight": rng.normal(0, 0.05, (n_embd, n_ff)),
                   "blk.2.attn_norm.weight": rng.normal(1, 0.1, n_embd),
                   "blk.2.attn_q.weight": rng.normal(0, 0.05, (n_q, n_embd)),
                   "blk.2.attn_k.weight": rng.normal(0, 0.05, (n_kv, n_embd)),
                   "blk.2.attn_v.weight": rng.normal(0, 0.05, (n_kv, n_embd))}
        proxy = type("Proxy", (), {"rms_eps": 1e-6, "get": lambda self, name: weights[name].astype(np.float32)})()
        tmp = Path(tempfile.mkdtemp())
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        corrupt = {"on": False}

        def serve():
            conn, _ = listener.accept()
            self.addCleanup(conn.close)
            recv = lambda size: fused_numerics._receive(conn, size)  # noqa: E731
            hello = fused_numerics.HELLO_REQUEST.unpack(recv(64))
            conn.sendall(fused_numerics.HELLO_RESPONSE.pack(
                fused_numerics.MAGIC, 6, 2, 0, 3, n_embd, n_ff, 0, n_ff, 12, 1, hello[3], 7, n_ff, 4, 0, hello[8]))
            while True:
                try:
                    header = fused_numerics.EXECUTE_REQUEST.unpack(recv(36))
                except fused_numerics.FusedNumericsError:
                    return
                body = recv(header[6])
                if header[2] == fused_numerics.MSG_FUSED_HELLO:
                    payload = fused_numerics.FUSED_HELLO.pack(0b10, 0b10, n_q, n_kv, 0xABC, 1e-6, 0)
                    conn.sendall(fused_numerics.EXECUTE_RESPONSE.pack(
                        fused_numerics.MAGIC, 6, 10, 0, 0, 0, -1, 0, len(payload), fused_numerics.fnv32(payload),
                        0, 0, 0) + payload)
                    continue
                tokens, nxt = header[9], header[2] == fused_numerics.MSG_FUSED_NEXT
                rows = np.frombuffer(body, dtype=np.float16).astype(np.float32).reshape(tokens, n_q + n_embd)
                ref = fused_numerics.reference(proxy, header[4], rows[:, :n_q], rows[:, n_q:], nxt)
                out = np.concatenate([ref[k] for k in ("delta", "q", "k", "v") if k in ref], axis=1)
                if corrupt["on"]:
                    out = out * 1.01
                data = out.astype(np.float16).tobytes()
                conn.sendall(fused_numerics.EXECUTE_RESPONSE.pack(
                    fused_numerics.MAGIC, 6, 13, 0, 0, header[3], header[4], out.size, len(data),
                    fused_numerics.fnv32(data), header[8], tokens, 1000) + data)

        original = fused_numerics.ProxyWeights
        fused_numerics.ProxyWeights = lambda path: proxy  # noqa: E731
        self.addCleanup(setattr, fused_numerics, "ProxyWeights", original)
        args = argparse.Namespace(host="127.0.0.1", port=port, proxy=tmp / "p.gguf", proxy_sha256=None,
                                  artifact_sha256=PARENT, layers="1", next_layers="1", worker_layers="1", rows="1,2,4",
                                  repeats=1, seed=1, n_embd=n_embd, max_columns=n_ff, max_tokens=4, f32_io=False,
                                  threshold=1e-3)
        for corrupted, status in ((False, "PASS"), (True, "FAIL")):
            corrupt["on"] = corrupted
            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            result = fused_numerics.run(args)
            thread.join(5)
            self.assertEqual(result["status"], status)
            self.assertEqual(result["calls"], 6)
        listener.close()


if __name__ == "__main__":
    unittest.main()
