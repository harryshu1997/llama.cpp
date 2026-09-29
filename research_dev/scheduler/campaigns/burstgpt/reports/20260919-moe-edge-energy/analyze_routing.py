#!/usr/bin/env python3
"""Coverage curve of a resident-expert tier from a routing histogram.

Input: the JSON written by ``llama-moe-routing-histogram`` (per-layer expert selection counts) and
the GGUF (for exact per-expert bytes per layer). Output: for a set of tier sizes in bytes, the
fraction of layer-token expert selections the tier would serve if it held, per layer, the most
frequently selected experts, with the tier's bytes split across layers greedily by marginal hits
per byte. Also prints the skew of the routing distribution (how far it is from uniform), because a
near-uniform router caps every tier's coverage at tier_bytes / bank_bytes.
"""
import argparse
import json
import pathlib
import sys


def expert_bytes_per_layer(gguf_path, n_layer, gguf_py=None):
    """Bytes of one expert per layer, from the GGUF tensor sizes of the ffn_*_exps tensors."""
    sys.path.insert(0, gguf_py or str(pathlib.Path(__file__).resolve().parents[5] / "gguf-py"))
    from gguf import GGUFReader  # type: ignore
    reader = GGUFReader(gguf_path)
    per_layer = {}
    n_expert = None
    for t in reader.tensors:
        name = t.name
        if not name.startswith("blk.") or "_exps" not in name:
            continue
        il = int(name.split(".")[1])
        nbytes = int(t.n_bytes)
        # shape [n_ff, n_embd, n_expert] (or [n_embd, n_ff, n_expert]); the last dim is n_expert
        ne = list(t.shape)
        n_expert = int(ne[-1])
        per_layer[il] = per_layer.get(il, 0) + nbytes // n_expert
    if len(per_layer) < n_layer:
        raise SystemExit(f"expected {n_layer} MoE layers, found {len(per_layer)}")
    return per_layer, n_expert


def coverage_for_budget(counts, bytes_per_expert, budget):
    """Greedy: repeatedly add the (layer, next-most-frequent expert) with the best hits per byte."""
    layers = sorted(counts)
    sorted_counts = {il: sorted(counts[il], reverse=True) for il in layers}
    pos = {il: 0 for il in layers}
    total = sum(sum(c) for c in counts.values())
    spent = 0
    hits = 0
    resident = {il: 0 for il in layers}
    import heapq
    heap = []
    for il in layers:
        if sorted_counts[il]:
            heapq.heappush(heap, (-sorted_counts[il][0] / bytes_per_expert[il], il))
    while heap:
        _, il = heapq.heappop(heap)
        if spent + bytes_per_expert[il] > budget:
            continue
        spent += bytes_per_expert[il]
        hits += sorted_counts[il][pos[il]]
        pos[il] += 1
        resident[il] += 1
        if pos[il] < len(sorted_counts[il]):
            heapq.heappush(heap, (-sorted_counts[il][pos[il]] / bytes_per_expert[il], il))
    return {"budget_bytes": budget, "spent_bytes": spent, "coverage": hits / total if total else 0.0,
            "experts_resident": sum(resident.values()), "resident_per_layer_min": min(resident.values()),
            "resident_per_layer_max": max(resident.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--histogram", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tiers-gib", default="1,2,3.75,6,8,12")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf-py", default=None)
    args = ap.parse_args()

    h = json.load(open(args.histogram))
    counts = {int(k): v for k, v in h["counts"].items()}
    n_layer = len(counts)
    per_layer_bytes, n_expert = expert_bytes_per_layer(args.gguf, n_layer, args.gguf_py)
    bank_bytes = sum(per_layer_bytes[il] * n_expert for il in counts)
    total_sel = sum(sum(c) for c in counts.values())
    tokens = h["observed_rows"] / n_layer
    per_token_bytes = sum(per_layer_bytes[il] * h["n_expert_used"] for il in counts)

    # Skew: per layer, the coverage of the top-k experts vs the uniform k/n.
    skew = {}
    for k in (8, 16, 32, 64):
        cov = []
        for il, c in counts.items():
            s = sorted(c, reverse=True)
            cov.append(sum(s[:k]) / max(sum(s), 1))
        skew[f"top{k}_mean_coverage"] = sum(cov) / len(cov)
        skew[f"top{k}_uniform"] = k / n_expert
    # Entropy-based effective number of experts per layer.
    import math
    eff = []
    for c in counts.values():
        s = sum(c)
        p = [x / s for x in c if x > 0]
        eff.append(math.exp(-sum(x * math.log(x) for x in p)))
    skew["effective_experts_mean"] = sum(eff) / len(eff)
    skew["effective_experts_min"] = min(eff)
    skew["effective_experts_max"] = max(eff)

    tiers = []
    for g in args.tiers_gib.split(","):
        b = int(float(g) * 2**30)
        r = coverage_for_budget(counts, per_layer_bytes, b)
        r["tier_gib"] = float(g)
        r["uniform_coverage"] = min(1.0, b / bank_bytes)
        tiers.append(r)

    # Dense curve for interpolation by downstream scripts (coverage as a function of resident bytes).
    curve = []
    step = int(0.25 * 2**30)
    b = 0
    while b <= bank_bytes + step:
        curve.append({"bytes": b, "coverage": coverage_for_budget(counts, per_layer_bytes, b)["coverage"] if b else 0.0})
        b += step

    out = {
        "curve": curve,
        "histogram": args.histogram, "gguf": args.gguf, "tokens_observed": tokens, "selections": total_sel,
        "n_layer": n_layer, "n_expert": n_expert, "n_expert_used": h["n_expert_used"],
        "expert_bytes_per_layer_mean": sum(per_layer_bytes.values()) / n_layer,
        "bank_bytes": bank_bytes, "per_token_expert_bytes": per_token_bytes,
        "skew": skew, "tiers": tiers,
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"tokens {tokens:.0f}, bank {bank_bytes/2**30:.2f} GiB, per-token expert bytes {per_token_bytes/2**20:.1f} MiB, "
          f"effective experts/layer {skew['effective_experts_mean']:.1f} of {n_expert}")
    for t in tiers:
        print(f"tier {t['tier_gib']:>5} GiB: coverage {t['coverage']*100:5.1f} % (uniform {t['uniform_coverage']*100:5.1f} %), "
              f"{t['experts_resident']} experts, per layer {t['resident_per_layer_min']}-{t['resident_per_layer_max']}")


if __name__ == "__main__":
    main()
