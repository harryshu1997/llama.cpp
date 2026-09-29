#!/usr/bin/env python3
"""Per-arm accounting for the Pixel integration-2 campaigns (reads run artifacts only).

    python3 analyze_int2.py --arm label=<run dir> ... [--baseline label] --out X.json

Per arm: completion, duration, measured host energy (RAPL package + NVML board), every phone energy
domain (ASSUMED 4.5 W active / 0.875 W idle models, never measured), FFN calls per phone session and per
device (execution proofs), per-model assisted requests and fraction counts, helper-preparation failures,
the phone layout (re-provisioning) timeline, and the Pixel worker lifecycle. With --baseline, exact saved
output tokens of every arm are compared against that arm (analyze_longdecode_pair).
"""
import argparse
import collections
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "20260921-fast-path-trace-v2a"))
from analyze_longdecode_pair import analyze  # noqa: E402

TIMELINE_KINDS = ("PREPARING", "SESSION_LOADING", "SESSION_VERIFIED", "SESSION_READY", "READY",
                  "TRANSITION_FAILED", "SESSION_FAILED", "PROPOSAL_REJECTED")


def model_key(model_id):
    return "qwen" if "qwen" in model_id else "gemma" if "gemma" in model_id else "llama"


def arm_summary(run: Path):
    result = json.loads((run / "RESULT.json").read_text())
    domains = result["trace_energy"]["fleet_energy_uj_by_domain"]
    host = (domains.get("cpu-package", 0) + domains.get("gpu-board", 0)) / 1e9
    phones = {key: value / 1e9 for key, value in domains.items() if key not in ("cpu-package", "gpu-board")}
    calls_by_session = collections.Counter()
    rows_by_session = collections.Counter()
    calls_by_model_device = collections.Counter()
    assisted = collections.defaultdict(set)
    for proof in result.get("physical_execution_proofs", {}).values():
        model = next((model_key(row["model_id"]) for row in result["request_results"]
                      if row.get("physical_execution_proof", {}) and
                      row["physical_execution_proof"].get("proof_sha256") == proof.get("proof_sha256")), None)
        for entry in proof.get("phone_calls_by_session", []):
            session = entry["session_id"]
            device = "pixel" if session.startswith("PIXEL") else "op15"
            calls_by_session[session] += entry.get("calls", 0)
            rows_by_session[session] += entry.get("rows", 0)
            calls_by_model_device[(model or "?") + ":" + device] += entry.get("calls", 0)
    for row in result["request_results"]:
        proof = row.get("physical_execution_proof") or {}
        if proof.get("phone_call_count"):
            assisted[model_key(row["model_id"])].add(row["request_id"].split(":")[-1])
    helper_events = collections.Counter(event["kind"] for event in result.get("request_helper_events", []))
    failures = collections.Counter(event.get("reason") for event in result.get("request_helper_events", [])
                                   if event["kind"] == "PREPARATION_FAILED")
    layout = collections.Counter(event["kind"] for event in result.get("phone_residency_events", []))
    timeline = []
    artifacts = {row["artifact_sha256"]: model_key(model_id)
                 for model_id, row in (result.get("model_artifacts") or {}).items()}
    for row in result["request_results"]:
        proof = row.get("physical_execution_proof") or {}
        if proof.get("artifact_sha256"):
            artifacts.setdefault(proof["artifact_sha256"], model_key(row["model_id"]))
    for event in result.get("phone_residency_events", []):
        if event["kind"] not in TIMELINE_KINDS:
            continue
        session = event.get("session") or {}
        shards = (event.get("layout") or {}).get("shards") or []
        row = {"at_s": round(event["observed_at_us"] / 1e6, 1), "kind": event["kind"],
               "generation": event.get("layout_generation", event.get("generation", event.get("failed_generation")))}
        if session:
            row["session"] = session.get("session_id")
            row["model"] = artifacts.get(session.get("resident_artifact_sha256"), "?")
        if shards:
            row["layout"] = {shard.get("session_id"): artifacts.get(shard.get("artifact_sha256"), "?") + ":"
                             + format(int(shard.get("layer_mask", 0)), "x") for shard in shards}
        if event.get("reason"):
            row["reason"] = event["reason"][:120]
        timeline.append(row)
    lifecycle_path = run / "CO_HELPER_LIFECYCLE.json"
    lifecycle = json.loads(lifecycle_path.read_text()) if lifecycle_path.exists() else None
    per_request = []
    for row in sorted(result["request_results"], key=lambda item: item["replay_arrival_us"]):
        receipt = (row.get("completion") or {}).get("execution_receipt") or {}
        proof = row.get("physical_execution_proof") or {}
        per_request.append({
            "request": row["request_id"].split(":")[-1], "model": model_key(row["model_id"]),
            "arrival_s": round(row["replay_arrival_us"] / 1e6, 1),
            "start_s": None if not receipt.get("started_us") else round(receipt["started_us"] / 1e6, 1),
            "end_s": None if not receipt.get("finished_us") else round(receipt["finished_us"] / 1e6, 1),
            "output_tokens": row.get("output_tokens"), "phone_calls": proof.get("phone_call_count", 0),
            "phone_executed_fractions_ppm": (row.get("fraction_history") or {}).get(
                "phone_executed_split_fractions_ppm"),
        })
    return {
        "status": result.get("status"), "duration_s": result["duration_us"] / 1e6, "counts": result.get("counts"),
        "host_kj_measured": host, "cpu_kj": domains.get("cpu-package", 0) / 1e9,
        "gpu_kj": domains.get("gpu-board", 0) / 1e9, "phone_kj_assumed_by_domain": phones,
        "fleet_kj_modeled": host + sum(phones.values()),
        "ffn_calls_by_session": dict(sorted(calls_by_session.items())),
        "ffn_rows_by_session": dict(sorted(rows_by_session.items())),
        "ffn_calls_by_model_device": dict(sorted(calls_by_model_device.items())),
        "assisted_requests_by_model": {key: sorted(value) for key, value in sorted(assisted.items())},
        "placement_summary": {key: (result.get("placement_summary") or {}).get(key) for key in (
            "model_reload_count", "transition_count", "executed_fraction_counts", "adaptive_fraction_counts")},
        "request_helper_event_counts": dict(helper_events.most_common()),
        "preparation_failure_reasons": dict(failures.most_common()),
        "phone_layout_event_counts": dict(layout.most_common()),
        "phone_layout_timeline": timeline,
        "co_helper_lifecycle": lifecycle,
        "requests": per_request,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="label=run dir (containing RESULT.json)")
    parser.add_argument("--baseline")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = dict(spec.split("=", 1) for spec in args.arm)
    report = {"energy_note": "Host = RAPL package + NVML board (measured). Phone domains use the assumed "
                             "4.5 W active / 0.875 W idle model; fleet_kj_modeled is not a measured total. "
                             "Single run per arm.",
              "arms": {label: arm_summary(Path(path)) for label, path in arms.items()}}
    if args.baseline:
        base = report["arms"][args.baseline]
        for label, path in arms.items():
            arm = report["arms"][label]
            arm["host_vs_baseline_pct"] = 100 * (arm["host_kj_measured"] / base["host_kj_measured"] - 1)
            arm["duration_vs_baseline_pct"] = 100 * (arm["duration_s"] / base["duration_s"] - 1)
            if label != args.baseline:
                pair = analyze(Path(arms[args.baseline]), Path(path))
                arm["tokens_vs_baseline"] = {key: pair.get(key) for key in (
                    "status", "checks", "identical_outputs", "output_differences", "missing_request_ids",
                    "extra_request_ids", "input_differences", "treatment_by_model", "output_tokens")}
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for label, arm in report["arms"].items():
        tokens = arm.get("tokens_vs_baseline") or {}
        print(f"{label:24s} {arm['status']:6s} {arm['duration_s']:8.1f}s host {arm['host_kj_measured']:8.3f} kJ "
              f"phones* {sum(arm['phone_kj_assumed_by_domain'].values()):6.3f} kJ calls {arm['ffn_calls_by_model_device']} "
              f"identical {tokens.get('identical_outputs', '-')} "
              f"prepfail {sum(arm['preparation_failure_reasons'].values())}")


if __name__ == "__main__":
    main()
