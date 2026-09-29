"""Native split-KV storage, boundary, and unsplit-equivalence checks through the probe."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

import numpy as np

from .tiny_llama_gguf import N_HEAD_KV, write_tiny_llama_gguf

ROOT = Path(__file__).resolve().parents[3]
BIN = Path(os.environ.get("S42_LLAMA_BUILD_BIN", ROOT / "build-cpu/bin"))
PROBE = BIN / "llama-ffn-remote-resident-probe"
GPU_LAYERS = os.environ.get("S42_SPLIT_KV_GPU_LAYERS", "0")


@unittest.skipUnless(PROBE.exists(), "native probe is not built")
class SplitKvAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="split-kv-")
        cls.model = write_tiny_llama_gguf(Path(cls.tmp.name) / "tiny.gguf", n_layer=4, head_dim=64)
        cls.serial = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def probe(self, extra=(), count=300, context=1024, decode=64, success=True):
        type(self).serial += 1
        stem = Path(self.tmp.name) / str(self.serial)
        rng = random.Random(7)
        tokens = [rng.randrange(2, 128) for _ in range(count)]
        command = [str(PROBE), "--model", str(self.model), "--out", str(stem.with_suffix(".json")),
                   "--logits", str(stem.with_suffix(".bin")), "--ctx-size", str(context), "--flash-attn", "on",
                   "--gpu-layers", GPU_LAYERS, "--threads", "4", "--max-tokens", "512",
                   "--batch-size", "512", "--ubatch-size", "512", "--decode", str(decode),
                   "--tokens", ",".join(map(str, tokens)), *extra]
        run = subprocess.run(command, capture_output=True, text=True, timeout=300,
                             env={**os.environ, "LLAMA_KV_CACHE_EAGER_CLEAR": "0"})
        if not success:
            self.assertGreaterEqual(run.returncode, 0, run.stderr[-4000:])
            if run.returncode == 0:
                self.assertFalse(json.loads(stem.with_suffix(".json").read_text())["context_created"])
            return
        self.assertEqual(run.returncode, 0, run.stderr[-4000:])
        record = json.loads(stem.with_suffix(".json").read_text())
        self.assertEqual(record["decode"]["status"], 0, record)
        return record, np.fromfile(stem.with_suffix(".bin"), dtype=np.float32)

    def test_extremes_are_bit_exact(self):
        for prefix, control in ((0, ("--kv-cpu-layers", "0,1,2,3")), (1024, ())):
            before, a = self.probe(control)
            after, b = self.probe(("--kv-device-cells", ",".join(f"{i}:{prefix}" for i in range(4))))
            self.assertEqual(before["decode"]["argmax"], after["decode"]["argmax"])
            self.assertTrue(np.array_equal(a, b))

    def test_boundary_and_straddling_ubatch(self):
        for count in (256, 300):
            before, a = self.probe(count=count)
            after, b = self.probe(("--kv-device-cells", "0:256,1:256,2:256,3:256",
                                   "--decode-tokens", ",".join(map(str, before["decode"]["argmax"][:-1]))), count=count)
            nmse = float(np.sum((a.astype(np.float64) - b)**2) / np.sum(a.astype(np.float64)**2))
            self.assertLessEqual(nmse, 5e-4)
            aa, bb = a.reshape(-1, 128), b.reshape(-1, 128)
            disagreements = 0
            for row, (x, y) in enumerate(zip(before["decode"]["argmax"], after["decode"]["argmax"])):
                if x != y:
                    disagreements += 1
                    # Greedy ties may differ; the winning margin must lie within the measured logit error.
                    self.assertLessEqual(aa[row, x] - aa[row, y], 2 * np.max(np.abs(aa[row] - bb[row])))
            print(f"split KV prompt={count} max_abs={np.max(np.abs(a-b)):.9g} NMSE={nmse:.9g} "
                  f"near_tie_argmax_differences={disagreements}", flush=True)

    def test_clear_reuse_and_touch_across_boundary(self):
        record, _ = self.probe(("--kv-device-cells", "0:256,1:256,2:256,3:256", "--clear-reuse",
                                "--kv-touch-tokens", "4096"), context=8192, decode=4)
        self.assertTrue(record["clear_reuse"]["logits_identical"], record["clear_reuse"])
        self.assertEqual(record["kv_touch"]["bytes"], 4 * 2 * N_HEAD_KV * 64 * 2 * 4096)

    def test_invalid_contracts_are_rejected(self):
        for spec in ("0:257", "0:1280", "0:256,0:512", "4:256"):
            self.probe(("--kv-device-cells", spec), count=6, decode=1, success=False)
        self.probe(("--kv-device-cells", "0:256", "--flash-attn", "off"), count=6, success=False)
        self.probe(("--kv-device-cells", "0:256", "--kv-cpu-layers", "0"), count=6, success=False)

    def test_state_roundtrip_and_two_streams(self):
        record, _ = self.probe(("--kv-device-cells", "0:256,1:512,2:256,3:512", "--state-reuse"),
                               count=400, decode=4)
        self.assertTrue(record["state_reuse"]["logits_identical"], record["state_reuse"])
        # Two streams put both the scratch rows and host slices at nonzero stream offsets.
        extra = ("--sequence-count", "2", "--batch-size", "1024", "--max-tokens", "1024")
        before, a = self.probe((*extra, "--state-reuse"), count=600, context=2048, decode=4)
        self.assertTrue(before["state_reuse"]["logits_identical"], before["state_reuse"])
        after, b = self.probe((*extra, "--kv-device-cells", "0:256,1:256,2:256,3:256",
                               "--state-reuse", "--decode-tokens",
                               ",".join(map(str, before["decode"]["argmax"][:-1]))),
                              count=600, context=2048, decode=4)
        self.assertTrue(after["state_reuse"]["logits_identical"], after["state_reuse"])
        self.assertLess(float(np.sum((a.astype(np.float64)-b)**2) / np.sum(a.astype(np.float64)**2)), 5e-4)


if __name__ == "__main__":
    unittest.main()
