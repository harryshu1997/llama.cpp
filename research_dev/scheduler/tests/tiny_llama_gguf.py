"""Write a tiny real-architecture (llama) GGUF that llama.cpp can load.

Used by native loader tests (remote-resident FFN omission) and by scheduler
tests that need a model llama.cpp itself accepts. By default no tokenizer is
written; server completion tests can request a small SPM vocabulary.

usage: tiny_llama_gguf.py <path> [--layers N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "gguf-py") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

import numpy as np  # noqa: E402
import gguf  # noqa: E402

N_EMBD, N_FF, N_HEAD, N_HEAD_KV, N_VOCAB, HEAD_DIM = 64, 256, 4, 2, 128, 16


def ffn_tensor_bytes_per_layer() -> int:
    """gate + up + down in f16."""
    return 3 * N_FF * N_EMBD * 2


def write_tiny_llama_gguf(path: Path, *, n_layer: int = 4, seed: int = 0, head_dim: int = HEAD_DIM,
                        with_tokenizer: bool = False) -> Path:
    path = Path(path)
    if path.exists():
        path.unlink()
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_context_length(256)
    n_embd = N_HEAD * head_dim
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(N_FF)
    writer.add_head_count(N_HEAD)
    writer.add_head_count_kv(N_HEAD_KV)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_freq_base(10000.0)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_rope_dimension_count(head_dim)
    writer.add_vocab_size(N_VOCAB)
    writer.add_tokenizer_model("llama" if with_tokenizer else "no_vocab")
    if with_tokenizer:
        writer.add_token_list(["<unk>", "<s>", "</s>", "<0x0A>"] + [f"t{i}" for i in range(4, N_VOCAB)])
        writer.add_token_types([2, 3, 3, 6] + [1] * (N_VOCAB - 4))
        writer.add_token_scores([0.0] * N_VOCAB)
        writer.add_unk_token_id(0)
        writer.add_bos_token_id(1)
        writer.add_eos_token_id(2)
        writer.add_add_bos_token(False)
    rng = np.random.default_rng(seed)

    def tensor(name: str, shape: tuple[int, ...]) -> None:
        values = rng.standard_normal(shape) * 0.02
        if len(shape) == 1:
            # norm weights stay f32 like every converter output; f16 norms are unsupported by ggml_mul
            writer.add_tensor(name, (1.0 + values).astype(np.float32))
        else:
            writer.add_tensor(name, values.astype(np.float16))

    tensor("token_embd.weight", (N_VOCAB, n_embd))
    tensor("output_norm.weight", (n_embd,))
    tensor("output.weight", (N_VOCAB, n_embd))
    for index in range(n_layer):
        block = f"blk.{index}."
        tensor(block + "attn_norm.weight", (n_embd,))
        tensor(block + "attn_q.weight", (N_HEAD * head_dim, n_embd))
        tensor(block + "attn_k.weight", (N_HEAD_KV * head_dim, n_embd))
        tensor(block + "attn_v.weight", (N_HEAD_KV * head_dim, n_embd))
        tensor(block + "attn_output.weight", (n_embd, N_HEAD * head_dim))
        tensor(block + "ffn_norm.weight", (n_embd,))
        tensor(block + "ffn_gate.weight", (N_FF, n_embd))
        tensor(block + "ffn_up.weight", (N_FF, n_embd))
        tensor(block + "ffn_down.weight", (n_embd, N_FF))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()
    written = write_tiny_llama_gguf(args.path, n_layer=args.layers)
    print(written, written.stat().st_size, "bytes;", ffn_tensor_bytes_per_layer(), "ffn bytes per layer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
