#!/usr/bin/env python3
"""Host energy (RAPL + NVML) over the union of one model's adaptive windows at one active batch size.

    python3 phase_energy_batch.py label=<run> ...    (needs ../20260924-coherent-policy-coalesced/phase_energy.py)

Same integration as phase_energy.py; J/token divides by that phase's window tokens (all slots)."""
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "20260924-coherent-policy-coalesced"))
from phase_energy import energy, load, union  # noqa: E402


def main(label, run):
    result, rapl, gpu = load(run)
    t0 = result["paid_start_ns"]
    model = {row["request_id"]: row["model_id"].split("-")[0] for row in result["request_results"]}
    spans, tokens = collections.defaultdict(list), collections.Counter()
    for group in json.load(open(pathlib.Path(run) / "ADAPTIVE_DECODE_OBSERVATIONS.json"))["groups"]:
        if group["request_id"] not in model:
            continue
        for w in group["windows"]:
            key = (model[group["request_id"]], w.get("active_batch") or 1)
            spans[key].append((t0 + w["started_at_us"] * 1000, t0 + w["finished_at_us"] * 1000))
            tokens[key] += w["token_end"] - w["token_start"]
    out = {"label": label}
    for (name, batch), intervals in sorted(spans.items()):
        joined = union(intervals)
        joules = sum(sum(energy(rapl, gpu, a, b)) for a, b in joined)
        out[f"{name}_b{batch}"] = {"s": round(sum(b - a for a, b in joined) / 1e9, 1), "kj": round(joules / 1e3, 2),
                                   "tokens": tokens[(name, batch)],
                                   "j_per_token": round(joules / tokens[(name, batch)], 1)}
    print(json.dumps(out))


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        label, _, path = spec.partition("=")
        main(label, path)
