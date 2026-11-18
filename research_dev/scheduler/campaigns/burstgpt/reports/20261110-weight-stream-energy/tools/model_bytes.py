#!/usr/bin/env python3
"""Weight bytes streamed per decoded token by the full model, at two bytes per weight (f16).

    model_bytes.py NAME=MODEL.gguf ... --json OUT

Streamed = every weight matrix a decode step multiplies in full: the attention projections and FFN matrices of
every block, any other 2-D block matrix, and the output head (the token embedding when the head is tied).
Not streamed = embedding tables that are indexed (token_embd when untied, per-layer token embeddings) and 1-D
tensors (norms, biases; counted separately, negligible).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(REPO / "gguf-py"))
import gguf  # noqa: E402

LOOKUP = ("token_embd.weight", "per_layer_token_embd.weight", "position_embd.weight")


def inventory(path: Path) -> dict:
    reader = gguf.GGUFReader(str(path))
    arch = reader.fields["general.architecture"].contents()
    names = {t.name for t in reader.tensors}
    tied = "output.weight" not in names
    groups = defaultdict(int)
    for t in reader.tensors:
        shape = [int(v) for v in t.shape]
        elements = 1
        for v in shape:
            elements *= v
        nbytes = elements * 2
        if len([v for v in shape if v > 1]) < 2:
            groups["one_d"] += nbytes
        elif t.name == "token_embd.weight":
            groups["output_head" if tied else "lookup"] += nbytes
        elif t.name in LOOKUP or "per_layer_token_embd" in t.name:
            groups["lookup"] += nbytes
        elif t.name == "output.weight":
            groups["output_head"] += nbytes
        elif ".ffn_" in t.name and t.name.startswith("blk."):
            groups["ffn"] += nbytes
        elif ".attn_" in t.name and t.name.startswith("blk."):
            groups["attention"] += nbytes
        else:
            groups["other_2d"] += nbytes
            groups.setdefault("_other_names", 0)
    other = sorted({t.name.split(".", 2)[-1] if t.name.startswith("blk.") else t.name for t in reader.tensors
                    if len([v for v in t.shape if int(v) > 1]) >= 2 and ".ffn_" not in t.name
                    and ".attn_" not in t.name and t.name not in ("output.weight", "token_embd.weight")
                    and "per_layer_token_embd" not in t.name})
    groups.pop("_other_names", None)
    streamed = groups["ffn"] + groups["attention"] + groups["output_head"] + groups["other_2d"]
    return {"path": str(path), "architecture": arch, "blocks": int(reader.fields[arch + ".block_count"].contents()),
            "tied_head": tied, "bytes": dict(groups), "streamed_bytes": streamed,
            "ffn_share": groups["ffn"] / streamed, "other_2d_tensors": other}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out = {}
    for item in args.models:
        name, _, path = item.partition("=")
        out[name] = inventory(Path(path))
        r = out[name]
        b = r["bytes"]
        print(f"{name:<12} {r['architecture']:<7} blocks {r['blocks']:3d} streamed {r['streamed_bytes']/1e9:7.3f} GB "
              f"(ffn {b.get('ffn',0)/1e9:.3f} attn {b.get('attention',0)/1e9:.3f} head {b.get('output_head',0)/1e9:.3f} "
              f"other {b.get('other_2d',0)/1e9:.3f}) lookup {b.get('lookup',0)/1e9:.3f} tied={r['tied_head']} other={r['other_2d_tensors'][:8]}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
