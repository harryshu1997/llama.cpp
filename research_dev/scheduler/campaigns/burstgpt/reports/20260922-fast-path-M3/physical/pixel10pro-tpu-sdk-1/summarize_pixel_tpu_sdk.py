"""Recheck saved TPU FFN outputs and summarize warm physical measurements."""

import json
from pathlib import Path
import statistics

import numpy as np


def main():
    base = Path(__file__).resolve().parent
    probes = base.parents[4] / "TPU_SDK/probes"
    physical = base / "physical/pixel10pro-tpu-sdk-1"
    groups = {
        "tail512_fp16_b1": (512, 1, "ffn512_b1_v1", ["run-ffn512-2"]),
        "full_fp16_b1": (17408, 1, "ffn17408_b1_v1", ["run-ffn17408b1-1", "run-ffn17408b1-2"]),
        "full_fp16_b4": (17408, 4, "ffn17408_b4_v1", ["run-ffn17408b4-1", "run-ffn17408b4-2"]),
        "full_bf16_b1": (17408, 1, "ffn17408_b1_v1", ["run-ffn17408b1bf16-1"]),
    }
    summary = {}
    for name, (width, batch, vectors_dir, runs) in groups.items():
        vectors = np.load(probes / vectors_dir / "vectors.npz")
        rows, errors, output_hashes = [], [], {}
        results = []
        for run in runs:
            folder = physical / run
            result = json.loads((folder / "RESULT.json").read_text())
            results.append(result)
            records = [json.loads(line) for line in (folder / "CALLS.jsonl").read_text().splitlines()]
            if result["status"] != "PASS" or len(records) != result["calls"]:
                raise RuntimeError(f"incomplete/failed run: {run}")
            for record in records:
                expected = vectors["reference"][record["case"]].astype(np.float64)
                actual = np.fromfile(folder / f'output-{record["id"]:03d}.f32', dtype="<f4").reshape(batch, 5120)
                if not np.isfinite(actual).all():
                    raise RuntimeError(f"nonfinite output: {run}")
                error = np.linalg.norm(actual-expected, axis=1) / np.linalg.norm(expected, axis=1)
                errors.extend(error.tolist())
                output_hashes.setdefault(record["case"], set()).add(record["output_sha256"])
                if not record["warmup"]:
                    rows.append(record)
        timing = {}
        for field in ("invoke_ns", "worker_ns", "rpc_ns"):
            values = [row[field]/1e6 for row in rows]
            timing[field.removesuffix("_ns")] = dict(mean_ms=statistics.mean(values),
                median_ms=statistics.median(values), p90_ms=float(np.percentile(values, 90)),
                p99_ms=float(np.percentile(values, 99)), max_ms=max(values))
        timing["buffer_handling_median_ms"] = statistics.median((r["worker_ns"]-r["invoke_ns"])/1e6 for r in rows)
        timing["outside_worker_median_ms"] = statistics.median((r["rpc_ns"]-r["worker_ns"])/1e6 for r in rows)
        repeats_exact = all(len(hashes) == 1 for hashes in output_hashes.values())
        summary[name] = dict(status="PASS" if max(errors) <= 0.01 and repeats_exact else "FAIL",
            width=width, batch=batch, runs=runs, calls=sum(r["calls"] for r in results), warm_calls=len(rows),
            max_relative_l2=max(errors), repeats_exact_across_processes=repeats_exact, timing=timing,
            useful_matmul_gflops=6*5120*width*batch/(timing["invoke"]["median_ms"]*1e6),
            nominal_weight_GB_per_s=3*5120*width*2/(timing["invoke"]["median_ms"]*1e6),
            worker_ms_per_row=timing["worker"]["median_ms"]/batch,
            individual_worker_medians_ms=[r["timing"]["worker"]["median_ms"] for r in results])
    result = dict(status="PASS" if all(r["status"] == "PASS" for r in summary.values()) else "FAIL",
                  scope="numerical qualification and latency, real layer18 weights, synthetic inputs",
                  ffn_calls=sum(r["calls"] for r in summary.values()),
                  ffn_rows=sum(r["calls"]*r["batch"] for r in summary.values()),
                  add_calls=24, groups=summary, energy_measured=False, server_integrated=False,
                  full_model_tokens_verified=False, matched_cpu_comparison=False,
                  physical_peak_measured=False)
    (base / "PIXEL_TPU_SDK_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError("saved-output qualification failed")


if __name__ == "__main__":
    main()
