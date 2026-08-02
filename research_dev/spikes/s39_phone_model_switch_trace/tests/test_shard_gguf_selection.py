import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[4]
SHARD_PATH = ROOT / "research_dev" / "shard_gguf.py"
SPEC = importlib.util.spec_from_file_location("shard_gguf", SHARD_PATH)
assert SPEC is not None and SPEC.loader is not None
SHARD_GGUF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARD_GGUF)


class ShardSelectionTests(unittest.TestCase):
    def test_qwen2_untied_tail_omits_token_embedding(self):
        self.assertFalse(
            SHARD_GGUF.want_tensor(
                "token_embd.weight",
                30,
                48,
                48,
                arch="qwen2",
                has_output_weight=True,
            )
        )

    def test_qwen2_tied_tail_keeps_token_embedding(self):
        self.assertTrue(
            SHARD_GGUF.want_tensor(
                "token_embd.weight",
                30,
                48,
                48,
                arch="qwen2",
                has_output_weight=False,
            )
        )

    def test_other_arch_tail_behavior_is_unchanged(self):
        self.assertTrue(
            SHARD_GGUF.want_tensor(
                "token_embd.weight",
                30,
                48,
                48,
                arch="qwen3",
                has_output_weight=True,
            )
        )

    def test_head_keeps_token_embedding(self):
        self.assertTrue(
            SHARD_GGUF.want_tensor(
                "token_embd.weight",
                0,
                30,
                48,
                arch="qwen2",
                has_output_weight=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
