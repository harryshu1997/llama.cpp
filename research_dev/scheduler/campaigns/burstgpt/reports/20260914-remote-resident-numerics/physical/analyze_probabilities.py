"""Compare pre-sampling probabilities only at identical autoregressive prefixes."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_stream(path):
    tokens, probabilities, terminal = [], [], None
    for line in path.read_text().splitlines():
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        row = json.loads(line[5:])
        assert "error" not in row, (path, row)
        if row.get("stop"):
            terminal = row
            continue
        chunk = row.get("tokens", [])
        if not chunk:
            continue
        probs = row.get("completion_probabilities", [])
        assert len(probs) == len(chunk), (path, "missing diagnostic observations")
        for token, prob in zip(chunk, probs, strict=True):
            assert token == prob["id"]
            top = prob["top_logprobs"]
            assert len(top) == 32 and len({item["id"] for item in top}) == 32
            assert all(math.isfinite(item["logprob"]) for item in top)
            probabilities.append(prob)
        tokens.extend(chunk)
    assert terminal is not None
    settings = terminal["generation_settings"]
    assert settings["n_probs"] == 32 and settings["post_sampling_probs"] is False
    assert settings["backend_sampling"] is False
    assert settings["seed"] == 42 and settings["temperature"] == 0.0
    assert len(tokens) == terminal["tokens_predicted"]
    return tokens, probabilities, terminal


def compare(full_path, reduced_path, expected_full, expected_reduced):
    full_tokens, full_probs, full_final = read_stream(full_path)
    reduced_tokens, reduced_probs, reduced_final = read_stream(reduced_path)
    assert full_tokens == expected_full["tokens"]
    assert reduced_tokens == expected_reduced["tokens"]
    assert full_final["prompt"] == reduced_final["prompt"]
    assert full_final["tokens_evaluated"] == reduced_final["tokens_evaluated"]
    assert full_final["generation_settings"] == reduced_final["generation_settings"]
    assert len(full_tokens) == len(reduced_tokens)
    first = next((i for i, (a, b) in enumerate(zip(full_tokens, reduced_tokens, strict=True)) if a != b), None)
    count = len(full_tokens) if first is None else first + 1
    positions = []
    for index in range(count):
        a = {row["id"]: row for row in full_probs[index]["top_logprobs"]}
        b = {row["id"]: row for row in reduced_probs[index]["top_logprobs"]}
        common = sorted(a.keys() & b.keys())
        deltas = [b[key]["logprob"] - a[key]["logprob"] for key in common]
        selected = sorted({full_tokens[index], reduced_tokens[index]})
        assert all(key in a and key in b for key in selected), "selected candidate outside observed top-32"
        positions.append({
            "output_position": index + 1,
            "same_prefix": True,
            "context_tokens": full_final["tokens_evaluated"] + index,
            "full_selected": full_tokens[index], "reduced_selected": reduced_tokens[index],
            "shared_top32_tokens": len(common),
            "max_abs_shared_logprob_delta": max(abs(value) for value in deltas),
            "rms_shared_logprob_delta": math.sqrt(sum(value * value for value in deltas) / len(deltas)),
            "full_top5": full_probs[index]["top_logprobs"][:5],
            "reduced_top5": reduced_probs[index]["top_logprobs"][:5],
        })
    divergence = None
    if first is not None:
        left, right = full_tokens[first], reduced_tokens[first]
        a = {row["id"]: row for row in full_probs[first]["top_logprobs"]}
        b = {row["id"]: row for row in reduced_probs[first]["top_logprobs"]}
        full_gap = a[left]["logprob"] - a[right]["logprob"]
        reduced_gap = b[left]["logprob"] - b[right]["logprob"]
        divergence = {
            "output_position": first + 1, "context_tokens": full_final["tokens_evaluated"] + first,
            "full_selected": a[left], "reduced_selected": b[right],
            "full_probability_of_full_choice": math.exp(a[left]["logprob"]),
            "full_probability_of_reduced_choice": math.exp(a[right]["logprob"]),
            "reduced_probability_of_full_choice": math.exp(b[left]["logprob"]),
            "reduced_probability_of_reduced_choice": math.exp(b[right]["logprob"]),
            "full_choice_log_odds": full_gap, "reduced_choice_log_odds": reduced_gap,
            "choice_log_odds_delta": reduced_gap - full_gap,
        }
    return {
        "request_index": expected_full["request_index"],
        "full_stream_sha256": digest(full_path), "reduced_stream_sha256": digest(reduced_path),
        "output_tokens": len(full_tokens), "identical": first is None,
        "same_prefix_comparisons": count,
        "excluded_after_divergence": len(full_tokens) - count,
        "first_divergence": divergence,
        "maximum_same_prefix_shared_logprob_delta": max(row["max_abs_shared_logprob_delta"] for row in positions),
        "positions": positions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current_path = args.run / "REMOTE_RESIDENT_GATE.json"
    result = json.loads(current_path.read_text())
    previous = json.loads(args.previous.read_text())
    for key in ("artifact_sha256", "runtime_binary_sha256", "runtime_libraries_sha256",
                "remote_layer_mask", "resident_layer_mask", "shard", "launch_contracts", "requests"):
        assert result[key] == previous[key], ("comparison input changed", key)
    rows = []
    for index, (full, reduced) in enumerate(zip(result["arms"]["full"]["requests"],
                                              result["arms"]["reduced"]["requests"], strict=True)):
        row = compare(args.run / (full["request_id"] + ".raw"),
                      args.run / "phone/streams" / (reduced["request_id"] + ".raw"), full, reduced)
        row["matches_original_full_tokens"] = full["tokens"] == previous["arms"]["full"]["requests"][index]["tokens"]
        row["matches_original_reduced_tokens"] = reduced["tokens"] == previous["arms"]["reduced"]["requests"][index]["tokens"]
        rows.append(row)
    summary = {
        "schema": "s42-remote-resident-same-prefix-probabilities-v1",
        "status": "MEASURED", "gate_verdict_unchanged": result["status"],
        "result_sha256": digest(current_path), "previous_result_sha256": digest(args.previous),
        "comparisons": rows,
        "limits": ["Top-32 normalized logits only, not full-vocabulary KL or per-layer numeric errors",
                   "Differences after the first divergent prediction excluded because their prefixes differ",
                   "No qualification or energy/latency savings inferred from diagnostic runs"],
    }
    with args.output.open("x") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": summary["status"], "comparisons": [
        {key: value for key, value in row.items() if key != "positions"} for row in rows
    ]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
