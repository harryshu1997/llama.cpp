#!/usr/bin/env python3
"""Deterministically generate the synthetic E2 mechanics fixtures.

EVERY artifact here is SYNTHETIC. Per CONTRACT.md section 8 a synthetic timeline
exercises mechanics and emits NO physical result -- the comparator forces
MEASUREMENT_INVALID / SYNTHETIC_NO_PHYSICAL_CLAIM regardless of the arithmetic.
Nothing below is a measurement.

The synthetic pair is built so the mechanics are checkable by hand:
  control   : 300000 mW held for 20 s  -> 6_000_000_000_000 nJ
  treatment : 240000 mW held for 20 s  -> 4_800_000_000_000 nJ
which is exactly a 20 percent reduction, so the 10 percent gate must pass while
the label stays MEASUREMENT_INVALID because the evidence is synthetic.

Run with --check to assert the on-disk fixtures match a fresh generation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import e2_canon as canon  # noqa: E402
import integrator  # noqa: E402
import comparator  # noqa: E402

OUT = ROOT / "fixtures"
RAW = OUT / "raw"
CAPABILITY = OUT / "capability"
EXECUTION = OUT / "execution"

WORKLOAD = canon.digest({"workload": "s10_e2_synthetic"})
TRACE = canon.digest({"trace": "s10_e2_synthetic"})
SLO = canon.digest({"slo_policy": "s10_e2_synthetic"})
SCHEDULE = canon.digest({"schedule": "s10_e2_synthetic"})
POLICY_CONTROL = canon.digest({"policy": "OPTIMIZED_SERVER_ONLY_CONTROL"})
POLICY_TREATMENT = canon.digest({"policy": "Q_PIM_TREATMENT"})
NORMALIZER = canon.digest({"normalizer": "e2.zoh.left_edge.v1"})
CLOCK_EPOCH = "boot-synthetic-0000"
BOARD = "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"


def make_samples(power_mw, window_us, period_us, updates):
    """A synthetic timeline with an exactly known integral.

    Power alternates by +/-1 mW every `period_us` so the sample stream has real
    independent updates, then returns to the base value; the mean is arranged so
    the zero-order-hold integral is exactly power_mw * window_us.
    """
    samples = []
    timestamp = 0
    toggle = 0
    while timestamp < window_us:
        # Pairs of (+1, -1) cancel exactly over two periods, keeping the integral
        # exactly power_mw * window_us while producing genuine value changes.
        delta = 1 if toggle % 2 == 0 else -1
        samples.append([timestamp, power_mw + delta])
        timestamp += period_us
        toggle += 1
    samples.append([window_us, power_mw])
    return samples


def timeline(tid, role, policy, power_mw, raw_name, samples, window_us=20_000_000,
             period_us=100_000, uncertainty_mw=5000, offered=1000, met=1000,
             provenance="SYNTHETIC", instrument_kind="SYNTHETIC",
             scope="GPU_BOARD", boards=None, sync="MONOTONIC_SHARED",
             instrument_identity="synthetic-meter-0"):
    normalized = integrator.normalize_samples(samples)
    energy = integrator.integrate(normalized, 0, window_us)
    updates, gap = integrator.window_quality(normalized, 0, window_us)
    pstates = ["P0"] * len(samples)
    execution_name = raw_name
    record = {
        "schema_version": 1,
        "kind": "RealizedTimeline",
        "timeline_id": tid,
        "run_nonce": f"run.{tid}",
        "role": role,
        "provenance": provenance,
        "policy_digest": policy,
        "workload_digest": WORKLOAD,
        "trace_digest": TRACE,
        "slo_policy_digest": SLO,
        "route_schedule_digest": SCHEDULE,
        "offered_work": offered,
        "completed_work": met,
        "outcomes": {"met": met, "tardy": 0, "rejected": offered - met,
                     "canceled": 0},
        "clock_epoch_id": CLOCK_EPOCH,
        "marker_start_us": 0,
        "marker_end_us": window_us,
        "window_start_us": 0,
        "window_end_us": window_us,
        "scope": scope,
        "instrument_kind": instrument_kind,
        "instrument_identity": instrument_identity,
        "instrument_label": "synthetic mechanics meter",
        "synchronization": sync,
        "board_uuids": boards if boards is not None else (
            [BOARD] if scope == "GPU_BOARD" else []),
        "included_rails": ([f"{BOARD}/BOARD"] if scope == "GPU_BOARD"
                           else ["SERVER/WALL"]),
        "excluded_rails": ["SERVER/USB_VBUS", "PHONE/BATTERY"],
        "execution_artifact_id": f"execution.{tid}",
        "execution_artifact_sha256": "",
        "execution_artifact_path": f"execution/{execution_name}",
        "raw_artifact_id": raw_name,
        "raw_artifact_sha256": "",
        "raw_artifact_path": f"raw/{raw_name}",
        "paid_payload_sha256": comparator.paid_payload_digest(
            normalized, pstates, 0, window_us),
        "normalizer_digest": NORMALIZER,
        "sample_count": len(normalized),
        "independent_updates": updates,
        "max_gap_us": gap,
        "energy_nj": energy,
        "uncertainty_nj": uncertainty_mw * window_us,
        "source_revision": "933c722f6",
        "build_revision": "synthetic-e2-build-0",
        "device_identity": BOARD if scope == "GPU_BOARD" else "server-0",
        "status": "OK",
        "reason_code": "SYNTHETIC_MECHANICS_FIXTURE",
        "record_sha256": "",
    }
    raw = {"schema": "e2.raw.v2", "pstates": pstates, "samples": samples}
    raw.update({field: record[field] for field in comparator.RAW_BOUND_FIELDS})
    raw_text = json.dumps(raw, indent=2, sort_keys=True) + "\n"
    record["raw_artifact_sha256"] = canon.sha256_bytes(raw_text.encode("ascii"))

    execution = {"schema": "e2.execution.v2"}
    execution.update({field: record[field]
                      for field in comparator.EXECUTION_BOUND_FIELDS})
    execution_text = json.dumps(execution, indent=2, sort_keys=True) + "\n"
    record["execution_artifact_sha256"] = canon.sha256_bytes(
        execution_text.encode("ascii"))
    return canon.seal(record), raw_text, execution_text


def repetition_set(control, treatment):
    pairs = []
    order = []
    for index in range(8):
        first = ("OPTIMIZED_SERVER_ONLY_CONTROL" if index % 2 == 0
                 else "Q_PIM_TREATMENT")
        second = ("Q_PIM_TREATMENT" if index % 2 == 0
                  else "OPTIMIZED_SERVER_ONLY_CONTROL")
        order.extend([first, second])
        control_id = f"tl.control.{index}"
        treatment_id = f"tl.treatment.{index}"
        control_digest = (control["record_sha256"] if index == 0 else
                          canon.digest({"timeline_id": control_id}))
        treatment_digest = (treatment["record_sha256"] if index == 0 else
                            canon.digest({"timeline_id": treatment_id}))
        pairs.append({
            "pair_index": index,
            "control_timeline_id": control_id,
            "control_record_sha256": control_digest,
            "treatment_timeline_id": treatment_id,
            "treatment_record_sha256": treatment_digest,
            "first_executed_role": first,
            "status": "OK",
            "reason_code": "SYNTHETIC_MECHANICS_FIXTURE",
        })
    record = {
        "schema_version": 1,
        "kind": "RepetitionSet",
        "set_id": "set.synthetic.0",
        "declared_order": order,
        "order_digest": canon.digest(order),
        "aggregate_method": "SUM_ALL_PAIRS_V1",
        "warmup_timelines": [],
        "attempted_pairs": len(pairs),
        "pairs": pairs,
        "scope": "GPU_BOARD",
        "instrument_kind": "SYNTHETIC",
        "aggregate_result_label": "MEASUREMENT_INVALID",
        "reason_code": "SYNTHETIC_NO_PHYSICAL_CLAIM",
        "record_sha256": "",
    }
    return canon.seal(record)


def wall_coverage_proof(complete=True, uncertainty_floor_mw=1,
                        provenance="SYNTHETIC"):
    return {
        "schema": "e2.wall-capability-proof.v1",
        "capability_id": "cap.synthetic.wall",
        "coverage_proof_artifact_id": "cov.synthetic",
        "provenance": provenance,
        "instrument_kind": "EXTERNAL_WALL_METER",
        "instrument_identity": "synthetic-wall-meter-0",
        "covers_cpu": True,
        "covers_dram": True,
        "covers_gpu": True,
        "covers_storage": complete,
        "covers_psu_losses": complete,
        "covers_fans": complete,
        "uncertainty_floor_mw": uncertainty_floor_mw,
    }


def wall_capability(complete=True, uncertainty_floor_mw=1,
                    provenance="SYNTHETIC"):
    proof = wall_coverage_proof(complete, uncertainty_floor_mw, provenance)
    proof_text = json.dumps(proof, indent=2, sort_keys=True) + "\n"
    record = {
        "schema_version": 1,
        "kind": "ServerWallCapability",
        "capability_id": "cap.synthetic.wall",
        "provenance": provenance,
        "instrument_kind": "EXTERNAL_WALL_METER",
        "instrument_identity": "synthetic-wall-meter-0",
        "covers_cpu": True,
        "covers_dram": True,
        "covers_gpu": True,
        "covers_storage": complete,
        "covers_psu_losses": complete,
        "covers_fans": complete,
        "coverage_proof_artifact_id": "cov.synthetic",
        "coverage_proof_sha256": canon.sha256_bytes(proof_text.encode("ascii")),
        "coverage_proof_artifact_path": "capability/wall.json",
        "uncertainty_floor_mw": uncertainty_floor_mw,
        "status": "OK",
        "reason_code": "SYNTHETIC_MECHANICS_FIXTURE",
        "record_sha256": "",
    }
    return canon.seal(record)


def generate():
    files = []
    control_samples = make_samples(300_000, 20_000_000, 100_000, 200)
    treatment_samples = make_samples(240_000, 20_000_000, 100_000, 200)

    control, control_raw, control_execution = timeline(
        "tl.control.0", "OPTIMIZED_SERVER_ONLY_CONTROL",
        POLICY_CONTROL, 300_000, "control.json", control_samples)
    treatment, treatment_raw, treatment_execution = timeline(
        "tl.treatment.0", "Q_PIM_TREATMENT", POLICY_TREATMENT,
        240_000, "treatment.json", treatment_samples)

    files.append((RAW / "control.json", control_raw))
    files.append((RAW / "treatment.json", treatment_raw))

    files.append((EXECUTION / "control.json", control_execution))
    files.append((EXECUTION / "treatment.json", treatment_execution))

    files.append((OUT / "control_timeline.json",
                  json.dumps(control, indent=2, sort_keys=True) + "\n"))
    files.append((OUT / "treatment_timeline.json",
                  json.dumps(treatment, indent=2, sort_keys=True) + "\n"))
    files.append((OUT / "repetition_set.json",
                  json.dumps(repetition_set(control, treatment), indent=2,
                             sort_keys=True) + "\n"))
    proof_text = json.dumps(wall_coverage_proof(), indent=2, sort_keys=True) + "\n"
    files.append((CAPABILITY / "wall.json", proof_text))
    files.append((OUT / "wall_capability.json",
                  json.dumps(wall_capability(), indent=2, sort_keys=True) + "\n"))
    return files


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    RAW.mkdir(parents=True, exist_ok=True)
    CAPABILITY.mkdir(parents=True, exist_ok=True)
    EXECUTION.mkdir(parents=True, exist_ok=True)
    for path, text in generate():
        if args.check:
            if not path.exists() or path.read_text(encoding="ascii") != text:
                print(f"E2_FIXTURE_DRIFT: {path} differs from a fresh generation",
                      file=sys.stderr)
                return 1
        else:
            path.write_text(text, encoding="ascii")
    print("S10_E2_FIXTURES_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
