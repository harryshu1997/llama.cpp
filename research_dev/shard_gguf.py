#!/usr/bin/env python3
# [plan-a port] Shard a gguf into a per-stage slice so a phone STORES ONLY ITS LAYERS.
#
# Keeps ORIGINAL block indices (blk.17.* stays blk.17.*) and copies ALL metadata verbatim
# (block_count, sliding_window_pattern array, rope, tokenizer, ...), so the SWA/rope per-layer
# indexing is identical to the full model. The loader (gemma4.cpp load_arch_tensors, env
# LLAMA_LAYER_START/END) then creates exactly this slice - no "tensor not found", no wasted RAM.
#
# Tensor selection for range [start, end) of n_layer (= block_count):
#   blk.<N>.*        keep iff start <= N < end
#   token_embd.*     keep for the head and for a terminal stage with a tied lm_head
#   output(.weight)  keep iff end==n_layer (terminal lm_head)
#   output_norm.*    keep iff end==n_layer
#   everything else  keep (global tables, e.g. gemma-3n per_layer_* - needed by every stage)
#
# Usage:
#   shard_gguf.py IN.gguf OUT.gguf --start 17 --end 35
#   # then on device:  LLAMA_LAYER_START=17 llama-layersplit -m OUT.gguf ...
import argparse, logging, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
import gguf  # noqa: E402

logger = logging.getLogger("shard_gguf")

BLK_RE = re.compile(r"^blk\.(\d+)\.")


def want_tensor(
    name: str,
    start: int,
    end: int,
    n_layer: int,
    *,
    arch: str | None = None,
    has_output_weight: bool = False,
) -> bool:
    m = BLK_RE.match(name)
    if m:
        return start <= int(m.group(1)) < end
    if name.startswith("token_embd"):
        qwen2_has_untied_output = arch == "qwen2" and has_output_weight
        return start == 0 or (end == n_layer and not qwen2_has_untied_output)
    if name == "output.weight" or name.startswith("output_norm"):
        return end == n_layer
    # global (non per-layer) tensors: keep in every shard
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Shard a gguf to a layer slice [start,end).")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--start", type=int, required=True, help="first layer to keep (inclusive)")
    ap.add_argument("--end", type=int, required=True, help="one past last layer to keep")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    reader = gguf.GGUFReader(args.input, "r")

    arch_field = reader.fields["general.architecture"]
    arch = arch_field.contents()
    bc_field = reader.fields.get("%s.block_count" % arch)
    n_layer = int(bc_field.contents()) if bc_field else None
    if n_layer is None:
        raise SystemExit("could not read %s.block_count" % arch)
    if not (0 <= args.start < args.end <= n_layer):
        raise SystemExit(f"bad range [{args.start},{args.end}) for n_layer={n_layer}")

    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)

    # --- copy ALL key/value metadata verbatim (block_count stays n_layer) ---
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue  # written by GGUFWriter itself
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if val is not None:
            writer.add_key_value(field.name, val, val_type, sub_type=sub_type)

    # --- select tensors ---
    has_output_weight = any(t.name == "output.weight" for t in reader.tensors)
    kept = [
        t
        for t in reader.tensors
        if want_tensor(
            t.name,
            args.start,
            args.end,
            n_layer,
            arch=arch,
            has_output_weight=has_output_weight,
        )
    ]
    dropped = len(reader.tensors) - len(kept)
    kept_bytes = sum(t.n_bytes for t in kept)
    total_bytes = sum(t.n_bytes for t in reader.tensors)
    logger.info("arch=%s n_layer=%d  keep layers [%d,%d)", arch, n_layer, args.start, args.end)
    logger.info("tensors: keep %d / %d (drop %d)", len(kept), len(reader.tensors), dropped)
    logger.info("weight bytes: %.2f GiB / %.2f GiB (%.1f%%)",
                kept_bytes / 2**30, total_bytes / 2**30, 100.0 * kept_bytes / max(total_bytes, 1))

    for t in kept:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in kept:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
