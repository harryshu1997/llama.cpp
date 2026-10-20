"""2026-10-08: a qualified AOA bundle re-qualified for another model (``prepare_aoa_evidence retarget-model``).

The Qwen3-32B Pixel co-helper keeps the phone, USB port, relay, host bridge, server and worker binary of the 14B
bundle; the artifact, width, shard, libraries and environment change. Fresh receipts: server-token-identity (half and
full width of the new model) and numerical-rows-1-2-4 (plain FFN rows, ggml-cpu reference); the link and session
receipts are carried over with an annex; the FFN kernel rate is re-measured. The result must load with the
scheduler's own loader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from research_dev.scheduler.adapters.contracts import PhysicalAdapterError  # noqa: E402
from research_dev.scheduler.adapters.phone_aoa_session import AoaBridgeConfiguration  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.helper_phone_evidence import digest, load_helper_evidence  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import prepare_aoa_evidence as aoa_tool  # noqa: E402

import test_aoa_bridge_wiring as wiring  # noqa: E402

LAYERS = (21, 22, 23)
MASK = sum(1 << layer for layer in LAYERS)
ARTIFACT = "sha256:" + "a" * 64
LIBRARIES = "/phone/q8"
SHARD = LIBRARIES + "/Q8_21_23.ffn.gguf"
SHARD_SHA = "sha256:" + "c" * 64
COLUMNS = 256
LAYER_BYTES = 3 * 64 * COLUMNS * 34 // 32


class RetargetModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.evidence = aoa_tool.build_bundle(wiring.PrepareAoaEvidenceTests().fixture(self.directory.name))
        raw = json.loads(self.evidence.read_text())
        self.worker = raw["worker"]
        aoa = AoaBridgeConfiguration.from_json(raw["aoa_bridge"])
        worker_sha = self.worker["expected_sha256_by_path"][self.worker["worker_path"]]
        (self.root / "PIXEL_FFN_SHARDS.json").write_text(json.dumps({
            "schema": "s42-ffn-shard-index-v1", "parent_sha256": ARTIFACT, "shards": [{
                "path": Path(SHARD).name, "parent_sha256": ARTIFACT, "shard_sha256": SHARD_SHA,
                "layer_mask": f"{MASK:016x}", "columns": COLUMNS, "n_ff": COLUMNS,
                "shard_bytes": len(LAYERS) * LAYER_BYTES, "weight_type": "Q8_0", "session_id": "PIXEL0"}]}))
        self.pins = {self.worker["worker_path"]: worker_sha, LIBRARIES + "/libggml.so": "sha256:" + "1" * 64,
                     LIBRARIES + "/libggml-cpu.so": "sha256:" + "2" * 64, SHARD: SHARD_SHA}
        self.server = self.root / "server-identity-q8"
        self.server.mkdir()
        self.config = {
            "phone_worker": self.worker["worker_path"], "phone_model": SHARD, "phone_library_dir": LIBRARIES,
            "phone_environment": {"S42_PIXEL_CPU_THREADS": "5", "S42_PIXEL_CPU_MASK": "7c"},
            "expected_phone_sha256": {path: pin[7:] for path, pin in self.pins.items()},
            "legs_columns": [0, COLUMNS // 2, COLUMNS, 0],
            "phone": {"layers": list(LAYERS), "artifact_sha256": ARTIFACT, "columns": COLUMNS, "n_embd": 64,
                      "quantum": COLUMNS // 2}}
        self.write("CONFIG.json", self.config)
        self.write("RESULT.json", {
            "status": "PASS", "token_identity": [True] * 4, "transport": "aoa-bridge", "server_exit": 0,
            "worker_exit": 0, "phone_calls": 40, "boot_unchanged": True, "output_tokens_each": 64,
            "call_columns": {str(COLUMNS // 2): 20, str(COLUMNS): 20}})
        self.write("IDENTITY.json", {
            "phone_hashes": "".join(f"{pin[7:]}  {path}\n" for path, pin in self.pins.items()),
            "server_sha256": digest(self.root / "server")[7:],
            "aoa_bridge": {"options_sha256": aoa.options_sha256, "configuration": aoa.to_json()}})
        self.numerics = {"schema": "s43-ffn-numerics-v1", "status": "PASS", "worst_rel_l2": 2.5e-4,
                         "worst_rel_l2_f32_reference": 2.3e-2, "rows": [1, 2, 4], "layers": list(LAYERS),
                         "artifact_sha256": ARTIFACT, "calls": 9, "inputs": "synthetic seed=20261008",
                         "reference": "ggml-cpu (Q8_0 activation rounding, float64 products) from the parent GGUF",
                         "weight_types": {"ffn_gate": "Q8_0", "ffn_up": "Q8_0", "ffn_down": "Q8_0"}}
        self.write("FFN_NUMERICS.json", self.numerics)
        self.write("SERVER_FFN_SHAPES.json", [
            'S41SERVERFFNSHAPE {"tokens":1,"columns":%d,"calls":20,"rpc_mean_ms":12.0,"compute_mean_ms":8.0}' % COLUMNS,
            'S41SERVERFFNSHAPE {"tokens":1,"columns":%d,"calls":20,"rpc_mean_ms":7.0,"compute_mean_ms":4.0}'
            % (COLUMNS // 2)])

    def write(self, name: str, value: object) -> None:
        (self.server / name).write_text(json.dumps(value))

    def retarget(self, output: str = "retargeted", host_software: tuple[str, ...] = ()) -> Path:
        return aoa_tool.retarget_model(argparse.Namespace(
            evidence=str(self.evidence), shard_index=str(self.root / "PIXEL_FFN_SHARDS.json"),
            phone_shard_path=SHARD, server_identity=str(self.server), output_dir=str(self.root / output),
            host_software=list(host_software)))

    def test_retarget_requalifies_the_model_and_carries_only_the_link_receipts(self) -> None:
        old = load_helper_evidence(self.evidence)
        path = self.retarget()
        new = load_helper_evidence(path)
        self.assertEqual((new.worker.artifact_sha256, new.worker.columns, new.worker.layer_mask),
                         (ARTIFACT, COLUMNS, MASK))
        self.assertEqual(new.worker.library_directories, (LIBRARIES,))
        self.assertEqual(dict(new.worker.worker_environment), self.config["phone_environment"])
        self.assertEqual(new.worker.expected_sha256_by_path[SHARD], SHARD_SHA)
        self.assertEqual(new.worker.column_quantum, COLUMNS // 2)
        self.assertEqual(new.aoa_bridge, old.aoa_bridge)
        software = new.identity.software_identity
        self.assertEqual({key for key in software if key.startswith("phone_library_sha256:")},
                         {"phone_library_sha256:" + LIBRARIES + "/libggml.so",
                          "phone_library_sha256:" + LIBRARIES + "/libggml-cpu.so"})
        self.assertEqual(software["phone_worker_sha256"], old.identity.software_identity["phone_worker_sha256"])
        changed = {kind for kind in old.identity.receipts if old.identity.receipts[kind] != new.identity.receipts[kind]}
        self.assertEqual(changed, {"server-token-identity", "numerical-rows-1-2-4"})
        raw = json.loads(path.read_text())
        self.assertEqual(set(raw["retargeted_from"]["carried_receipts"]), set(aoa_tool.CARRIED_RECEIPTS))
        numerical = json.loads(Path(raw["receipt_paths"]["numerical-rows-1-2-4"]).read_text())
        self.assertEqual((numerical["kernel"], numerical["layers"]), ("ffn", list(LAYERS)))
        kernel = [row for row in raw["profile_fragment"]["kernels"] if row["kernel_id"].endswith(":ffn")][0]
        self.assertEqual(kernel["effective_bytes_per_s"], round(LAYER_BYTES / 0.008))
        self.assertNotIn("fused", raw)

    def test_retarget_refusals(self) -> None:
        cases = (
            ("FFN numerical rows failed", "FFN_NUMERICS.json", {**self.numerics, "status": "FAIL"}),
            ("FFN numerical rows failed", "FFN_NUMERICS.json", {**self.numerics, "layers": [21, 22]}),
            ("FFN numerical rows failed", "FFN_NUMERICS.json", {**self.numerics, "worst_rel_l2": 2e-3}),
            ("half and full of the new width", "CONFIG.json", {**self.config, "legs_columns": [0, 64, 256, 0]}),
            ("keeps the qualified worker binary", "CONFIG.json", {
                **self.config, "expected_phone_sha256": {**self.config["expected_phone_sha256"],
                                                         self.worker["worker_path"]: "d" * 64}}),
            ("served another shard", "CONFIG.json", {**self.config, "phone_model": LIBRARIES + "/other.gguf"}),
            ("served other layers", "CONFIG.json", {**self.config, "phone": {**self.config["phone"], "layers": [21]}}),
        )
        for index, (message, name, value) in enumerate(cases):
            original = (self.server / name).read_text()
            self.write(name, value)
            with self.subTest(message=message, index=index):
                with self.assertRaisesRegex(PhysicalAdapterError, message):
                    self.retarget(output=f"refused-{index}")
                self.assertFalse((self.root / f"refused-{index}").exists())
            (self.server / name).write_text(original)


    def test_retarget_repins_another_server_build_only_when_the_run_loaded_it(self) -> None:
        # 2026-10-08: the prompt-anchor server = the same llama-server binary next to a rebuilt server library in
        # another directory; the identity run must launch that copy with its directory first on LD_LIBRARY_PATH
        raw = json.loads(self.evidence.read_text())
        build = self.root / "anchor-build"
        build.mkdir()
        server = build / "llama-server"
        server.write_bytes(Path(raw["host_software_paths"]["host_binary_sha256"]).read_bytes())
        impl = build / "libllama-server-impl.so"
        impl.write_bytes(b"prompt-anchor server library")
        repin = (f"host_binary_sha256={server}", f"host_impl_sha256={impl}")
        self.write("CONFIG.json", {**self.config, "server": str(server),
                                   "server_environment": {"LD_LIBRARY_PATH": f"{build}:/cuda/lib"}})
        new = load_helper_evidence(self.retarget(host_software=repin))
        bundle = json.loads((self.root / "retargeted" / "PIXEL_EVIDENCE_AOA.json").read_text())
        self.assertEqual(bundle["host_software_paths"]["host_impl_sha256"], str(impl.resolve()))
        self.assertEqual(new.identity.software_identity["host_impl_sha256"], digest(impl))
        self.assertEqual(new.identity.software_identity["host_binary_sha256"], digest(server))
        self.assertEqual(set(bundle["retargeted_from"]["host_software"]), {"host_binary_sha256", "host_impl_sha256"})
        refusals = (
            ("did not load the re-pinned server library", {"LD_LIBRARY_PATH": f"/cuda/lib:{build}"}, str(server)),
            ("launched another server", {"LD_LIBRARY_PATH": f"{build}:/cuda/lib"},
             raw["host_software_paths"]["host_binary_sha256"]),
        )
        for index, (message, environment, launched) in enumerate(refusals):
            self.write("CONFIG.json", {**self.config, "server": launched, "server_environment": environment})
            with self.subTest(message=message):
                with self.assertRaisesRegex(PhysicalAdapterError, message):
                    self.retarget(output=f"repin-refused-{index}", host_software=repin)
                self.assertFalse((self.root / f"repin-refused-{index}").exists())


if __name__ == "__main__":
    unittest.main()
