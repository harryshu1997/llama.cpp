"""Audit raw server tokens, FFN call coverage and sampled energy for a Pixel arm."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics

from research_dev.scheduler.adapters.host_runtime import server_energy_summary


def analyze(root):
    config = json.loads((root / "CONFIG.json").read_text())
    result = json.loads((root / "RESULT.json").read_text())
    power = json.loads((root / "POWER_SAMPLES.json").read_text())
    requests = [json.loads((root / f"REQUEST-{i}.json").read_text()) for i in range(4)]
    calls = Counter()
    call_ids = []
    log = (root / "server.log").read_text()
    for match in re.finditer(r"S41SERVERFFNCALL context=(\S+) request=(\d+) layer=(\d+) "
                            r"tokens=(\d+) columns=(\d+) payload_bytes=(\d+)", log):
        context, ident, layer, tokens, columns, size = match.groups()
        rid = bytes.fromhex(context.split(":")[0]).decode("ascii")
        assert int(tokens) == 1 and int(size) == 10240
        calls[rid, int(layer), int(columns)] += 1
        call_ids.append(int(ident))
    assert call_ids == list(range(1, result["phone_calls"] + 1))
    baseline_tokens = requests[0]["execution"]["tokens"]
    token_checks = []
    for row in requests:
        raw_tokens = []
        terminal = None
        for line in (root / (row["request_id"] + ".raw")).read_text().splitlines():
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                event = json.loads(line[5:])
                raw_tokens.extend(event.get("tokens", []))
                if event.get("stop"):
                    terminal = event
        assert raw_tokens == row["execution"]["tokens"]
        assert len(raw_tokens) == config["output_tokens"]
        assert terminal is not None and terminal["truncated"] is False
        assert terminal["timings"]["prompt_n"] == config["prompt_tokens"]
        token_checks.append({"request_id": row["request_id"], "exact": raw_tokens == baseline_tokens,
                             "tokens": len(raw_tokens), "first_mismatch": next(
                                 (i for i, (a, b) in enumerate(zip(raw_tokens, baseline_tokens)) if a != b), None)})
        for field, start in (("request_host_energy", row["started_ns"]),
                             ("decode_host_energy", row["first_token_ns"])):
            recomputed = server_energy_summary(power, start, row["finished_ns"])
            for key in ("server_compute_device_energy_j", "cpu_package_energy_j", "gpu_board_energy_j"):
                assert abs(recomputed[key] - row[field][key]) < 1e-6
        if row["columns"]:
            assert len(row["controls"]) == 1
            applied = row["controls"][0]["ack"]["applied_token_index"]
            for layer in config["phone"]["layers"]:
                assert calls[row["request_id"], layer, row["columns"]] == config["output_tokens"] - applied
        else:
            assert not any(key[0] == row["request_id"] for key in calls)
    comparisons = []
    for row in requests[1:3]:
        comparison = {"columns": row["columns"], "request_s": row["request_s"],
                      "decode_s": row["decode_s"]}
        for field in ("request_host_energy", "decode_host_energy"):
            controls = [r[field]["server_compute_device_energy_j"] for r in (requests[0], requests[3])]
            energy = row[field]["server_compute_device_energy_j"]
            comparison[field] = {"joules": energy, "control_joules": controls,
                                 "saving_pct_vs_each": [(1 - energy / b) * 100 for b in controls],
                                 "saving_pct_vs_control_mean": (1 - energy / statistics.mean(controls)) * 100}
        comparison["decode_slowdown_pct_vs_each"] = [
            (row["decode_s"] / r["decode_s"] - 1) * 100 for r in (requests[0], requests[3])]
        comparisons.append(comparison)
    assert result["server_exit"] == result["worker_exit"] == 0
    shapes = [json.loads(line.split("S41SERVERFFNSHAPE ", 1)[1]) for line in log.splitlines()
              if "S41SERVERFFNSHAPE " in line]
    return {"status": "PASS" if all(x["exact"] for x in token_checks) else "FAIL",
            "raw_token_checks": token_checks, "power_recomputation": "PASS",
            "per_layer_call_coverage": "PASS", "phone_calls": len(call_ids),
            "layers": config["phone"]["layers"], "comparisons": comparisons,
            "shape_timing": shapes, "phone_energy_measured": False,
            "note": "One sample per split, bracketed by two controls; not a trace result or statistical estimate."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.arm), indent=2, allow_nan=False))
