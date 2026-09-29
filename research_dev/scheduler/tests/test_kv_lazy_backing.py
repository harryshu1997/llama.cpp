"""Page-granular KV backing: a plain host KV buffer must not be resident before it is written.

``llama_kv_cache`` zeroes plain CPU buffers with ``MADV_DONTNEED`` instead of ``memset``, so the
resident footprint tracks written cells; ``LLAMA_KV_CACHE_EAGER_CLEAR=1`` restores the eager
behaviour. The native probe reports anonymous RSS right after context creation and the greedy
argmax trace, so lazy and eager runs of the same tiny llama GGUF can be compared directly.

Binaries come from ``build-cpu/bin`` (override with ``S42_LLAMA_BUILD_BIN``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[3]
BIN_DIR = Path(os.environ.get("S42_LLAMA_BUILD_BIN", REPO_ROOT / "build-cpu" / "bin"))
PROBE = BIN_DIR / "llama-ffn-remote-resident-probe"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_llama_gguf import HEAD_DIM, N_FF, N_HEAD_KV, N_VOCAB, write_tiny_llama_gguf  # noqa: E402

N_LAYER = 4
LARGE_CONTEXT = 262144
SMALL_CONTEXT = 256
PROMPT = (1, 5, 9, 13, 21, 34)
KV_BYTES = 2 * N_LAYER * N_HEAD_KV * HEAD_DIM * 2 * LARGE_CONTEXT  # f16 K and V


@unittest.skipUnless(PROBE.exists(), f"{PROBE} is not built")
@unittest.skipUnless(sys.platform.startswith("linux"), "lazy zeroing is Linux only")
class KvLazyBackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="kv-lazy-")
        cls.model = write_tiny_llama_gguf(Path(cls.tmp.name) / "tiny.gguf", n_layer=N_LAYER)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _probe(self, name: str, *, eager: bool, context: int, model: Path | None = None,
               tokens: tuple[int, ...] = PROMPT, extra: tuple[str, ...] = ()) -> dict:
        out = Path(self.tmp.name) / f"{name}.json"
        env = {**os.environ, "LLAMA_KV_CACHE_EAGER_CLEAR": "1" if eager else "0",
               "LD_LIBRARY_PATH": str(BIN_DIR) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")}
        command = [str(PROBE), "--model", str(model or self.model), "--out", str(out), "--ctx-size", str(context),
                   "--gpu-layers", "0", "--threads", "2", "--tokens", ",".join(map(str, tokens)), "--decode", "4", *extra]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300, env=env)
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        record = json.loads(out.read_text())
        self.assertTrue(record["context_created"])
        self.assertEqual(record["decode"]["status"], 0, record["decode"].get("error"))
        record["lazy_lines"] = [line for line in record["log"] if "KV buffer zeroed lazily" in line]
        return record

    def test_lazy_host_kv_is_not_resident_until_written(self):
        lazy = self._probe("lazy", eager=False, context=LARGE_CONTEXT)
        eager = self._probe("eager", eager=True, context=LARGE_CONTEXT)
        small = self._probe("small", eager=False, context=SMALL_CONTEXT)
        self.assertGreaterEqual(KV_BYTES, 64 * 1024 * 1024)
        self.assertEqual(len(lazy["lazy_lines"]), 1, lazy["log"][-5:])
        self.assertEqual(eager["lazy_lines"], [])
        lazy_rss, eager_rss, small_rss = (r["context_rss_anon_bytes"] for r in (lazy, eager, small))
        # the eager context carries the whole KV allocation in anonymous RSS. The lazy one grows over the
        # small context only by context-scaled compute buffers (KQ scratch, masks), which stay well below
        # half of the eager growth for this shape.
        self.assertGreaterEqual(eager_rss - lazy_rss, KV_BYTES * 9 // 10, (eager_rss, lazy_rss, KV_BYTES))
        self.assertLess(lazy_rss - small_rss, (eager_rss - small_rss) // 2, (lazy_rss, small_rss, eager_rss))
        # greedy decode is unaffected: zero-fill-on-demand pages read as the zeros memset wrote before
        self.assertEqual(lazy["decode"]["argmax"], eager["decode"]["argmax"])
        # the trace holds the prompt's last-position argmax plus the four decode steps
        self.assertEqual(len(lazy["decode"]["argmax"]), 5)

    def test_clear_returns_touched_kv_pages_and_reuse_decodes_identically(self):
        """clear(true) on a lazily zeroed cache gives the written pages back to the kernel; a second decode of
        the same prompt on the same context reproduces the logits. The eager mode keeps the pages resident."""
        import random  # noqa: PLC0415
        model = write_tiny_llama_gguf(Path(self.tmp.name) / "tiny128.gguf", n_layer=4, head_dim=128)
        prompt = tuple(random.Random(5).randrange(2, N_VOCAB) for _ in range(4096))
        touched = 2 * N_LAYER * N_HEAD_KV * 128 * 2 * len(prompt)  # f16 K and V actually written: 16 MiB
        extra = ("--clear-reuse", "--batch-size", "4096", "--ubatch-size", "512", "--max-tokens", "4096")
        lazy = self._probe("clear-lazy", eager=False, context=8192, model=model, tokens=prompt, extra=extra)["clear_reuse"]
        eager = self._probe("clear-eager", eager=True, context=8192, model=model, tokens=prompt, extra=extra)["clear_reuse"]
        for record in (lazy, eager):
            self.assertTrue(record["ran"])
            self.assertEqual(record["decode_status"], 0)
            self.assertTrue(record["logits_identical"], record)
            self.assertEqual(record["max_abs_diff"], 0.0)
        self.assertGreaterEqual(lazy["rss_anon_after_decode"] - lazy["rss_anon_after_clear"], touched * 3 // 4, lazy)
        self.assertLess(eager["rss_anon_after_decode"] - eager["rss_anon_after_clear"], touched // 8, eager)

    def test_kv_touch_occupies_exactly_the_cache_pages(self):
        """The test hook writes the first N cells of every layer's K and V: anonymous RSS grows by the host-tier
        bytes, weights stay resident, decode is unaffected."""
        model = write_tiny_llama_gguf(Path(self.tmp.name) / "tiny128-touch.gguf", n_layer=4, head_dim=128)
        record = self._probe("touch", eager=False, context=8192, model=model, extra=("--kv-touch-tokens", "4000"))
        touch = record["kv_touch"]
        expected = 2 * N_LAYER * N_HEAD_KV * 128 * 2 * 4000
        self.assertTrue(touch["ran"])
        self.assertEqual(touch["bytes"], expected)
        self.assertGreaterEqual(touch["rss_anon_after"] - touch["rss_anon_before"], expected * 9 // 10)
        self.assertEqual(touch["file_rss_before"], touch["file_rss_after"])
        self.assertEqual(touch["majflt_before"], touch["majflt_after"])
        self.assertEqual(record["decode"]["status"], 0)

    def test_restore_after_released_room_is_consumed(self):
        """Release the phone share, occupy exactly the released bytes with touched anonymous memory (KV growth
        stand-in), decode locally and populate the share again: byte-exact restore and identical logits. Under a
        hard memory budget the same sequence would fault or be OOM-killed, which is why prompt admission must
        re-reserve the share before the next prompt (see decode_split_selection.DecodeReleaseAccountant)."""
        model = write_tiny_llama_gguf(Path(self.tmp.name) / "tiny128-consume.gguf", n_layer=4, head_dim=128)
        record = self._probe("consume", eager=False, context=512, model=model,
                             extra=("--dormant-mask", "6", "--dormant-host-columns", str(N_FF // 2), "--dormant-consume"))
        dormant = record["dormant"]
        self.assertTrue(dormant["ran"])
        self.assertGreater(dormant["released_bytes"], 0)
        self.assertEqual(dormant["consumed_bytes"], dormant["released_bytes"])
        # the consumer is touched page by page; RSS may already cover it from the heap, so only monotonicity is asserted
        self.assertGreaterEqual(dormant["rss_anon_after_consume"], dormant["rss_anon_before_consume"])
        self.assertEqual(dormant["restored_bytes"], dormant["released_bytes"])
        self.assertEqual(dormant["decode_status"], 0)
        self.assertTrue(dormant["logits_identical"], dormant)
        self.assertEqual(dormant["max_abs_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()
