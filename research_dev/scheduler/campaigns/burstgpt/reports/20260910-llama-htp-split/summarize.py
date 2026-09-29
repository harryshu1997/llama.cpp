"""Audit the bounded Llama calibration records without changing their evidence."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import mean


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measurements(rows):
    def values(key):
        return [row[key] for row in rows]

    return {
        "requests": len(rows),
        "wall_s_mean": mean(values("wall_s")),
        "wall_s_range": [min(values("wall_s")), max(values("wall_s"))],
        "decode_tokens_per_s_mean": mean(values("decode_tokens_per_s")),
        "decode_tokens_per_s_range": [min(values("decode_tokens_per_s")), max(values("decode_tokens_per_s"))],
        "prefill_s_mean": mean(row["execution"]["prompt_ms"] / 1000 for row in rows),
        "cpu_j_mean": mean(row["energy"]["cpu_package_energy_j"] for row in rows),
        "gpu_j_mean": mean(row["energy"]["gpu_board_energy_j"] for row in rows),
        "fleet_j_by_phone_w_mean": {
            str(power): mean(row["energy"]["fleet_j_by_phone_w"][str(power)] for row in rows)
            for power in (3, 4.5, 6)
        },
    }


def audit(directory):
    result, spec = read(directory / "RESULT.json"), read(directory / "SPEC.json")
    assert result["status"] == "PASS" and result["scheduler_qualified"] is False
    assert read(directory / "CLEANUP.json")["status"] == "PASS"
    assert result["phone_weight_loads"] == 1
    native = "\n".join(path.read_text() for path in sorted(directory.glob("*/server.stderr")))
    calls = Counter()
    for match in re.finditer(r"S41SERVERFFNCALL context=([0-9a-f]+):\d+:\d+:\d+ request=\d+ layer=(\d+) tokens=(\d+) columns=(\d+)", native):
        request_id, layer, tokens, columns = match.groups()
        assert int(layer) in spec["shard"]["layers"] and int(tokens) == 1
        calls[(bytes.fromhex(request_id).decode(), int(columns))] += 1
    terminal = [json.loads(line.split("S41SERVERFFN ", 1)[1]) for line in native.splitlines()
                if "S41SERVERFFN {" in line]
    assert all(row["status"] == "ok" and row["prefill_calls"] == 0 for row in terminal)
    assert sum(row["decode_calls"] for row in terminal) == sum(calls.values())
    phone_log = read(directory / "PHONE_LOG.json")
    phases = [json.loads(line.removeprefix("RESIDENTPHASE ")) for line in phone_log.splitlines()
              if line.startswith("RESIDENTPHASE ")]
    assert all(row["session_generation"] == 1 and row["artifact_sha256"] == spec["artifact_sha256"]
               and row["session_id"] == spec["shard"]["session_id"] for row in phases)
    phase_times = {row["phase"]: row["monotonic_us"] for row in phases}
    assert Counter(row["phase"] for row in phases)["WEIGHT_READ_BEGIN"] == 1
    assert "FFN shard parent=" + spec["artifact_sha256"] in phone_log
    if "functionfs" in spec["argv"]:
        assert "descriptors_ready backend=Hexagon" in phone_log
        assert "recoveries=0 status=0" in phone_log
    ready = read(directory / "READY.json")
    launch = read(directory / "WORKER_LAUNCH.json")
    native_quantum = launch["environment"].get("S41_FFN_COLUMN_QUANTUM")
    if native_quantum is None:
        native_quantum = launch["arguments"][launch["arguments"].index("--column-quantum") + 1]
    if "column_quantum" in spec:
        assert int(native_quantum) == spec["column_quantum"]
    fractions = {}
    all_controls = []
    for fraction in sorted({row["fraction"] for row in result["runs"]}):
        rows = [row for row in result["runs"] if row["fraction"] == fraction]
        statistics = measurements(rows)
        row_calls, assisted_positions = 0, 0
        for row in rows:
            output = row["execution"]
            assert output["output_quality"]["accepted"] and len(output["tokens"]) == spec["tested_output_tokens"]
            assert output["runtime_prompt_tokens"] == spec["request"]["input_tokens"]
            if fraction == 0:
                assert not row["controls"]
                continue
            control, = row["controls"]
            issued, ack = control["control"], control["ack"]
            assert ack["success"] and ack["policy_hash"] == issued["policy_hash"]
            assert ack["plan_generation"] == issued["plan_generation"] >= 1
            assert issued["columns"] == spec["shard"]["n_ff"] * fraction // 100
            assert issued["policy"]["layer_indices"] == spec["shard"]["layers"]
            actual = calls[(issued["request_id"], issued["columns"])]
            expected_positions = spec["tested_output_tokens"] - ack["applied_token_index"]
            assert actual == expected_positions * len(spec["shard"]["layers"])
            row_calls += actual
            assisted_positions += expected_positions
            all_controls.append({"request_id": issued["request_id"], "policy_hash": issued["policy_hash"],
                                 "parent_hash": issued["policy"]["desktop_placement_sha256"],
                                 "operator_plan_hash": issued["policy"]["operator_plan_sha256"],
                                 "ack_token": ack["applied_token_index"], "calls": actual})
        statistics.update(
            phone_calls=row_calls,
            assisted_token_percentage=100 * assisted_positions / (len(rows) * spec["tested_output_tokens"]),
            fraction_weighted_percentage=fraction * assisted_positions / (len(rows) * spec["tested_output_tokens"]),
        )
        fractions[str(fraction)] = statistics
    for value in fractions.values():
        value["saving_pct_vs_same_run_cpu_by_phone_w"] = {
            power: 100 * (1 - joules / fractions["0"]["fleet_j_by_phone_w_mean"][power])
            for power, joules in value["fleet_j_by_phone_w_mean"].items()
        }
    assert sum(value["phone_calls"] for value in fractions.values()) == sum(calls.values())
    preparation = []
    for row in result["preparation"]:
        seconds = (row["end_ns"] - row["start_ns"]) / 1e9
        preparation.append({
            "kind": row["kind"], "seconds": seconds, "server_energy": row["server_energy"],
            "fleet_j_by_phone_w": {
                str(power): row["server_energy"]["server_compute_device_energy_j"] + seconds *
                (power if row["kind"] == "phone_preload" else 0.875) for power in (3, 4.5, 6)
            },
        })
    return {
        "status": "NATIVE_CALIBRATION_PASS_NOT_SCHEDULER_QUALIFIED",
        "result_sha256": digest(directory / "RESULT.json"), "spec_sha256": digest(directory / "SPEC.json"),
        "spec": {key: spec[key] for key in ("cpu_parent", "server_sha256", "libraries",
                                             "artifact_sha256", "index_sha256", "tested_output_tokens")},
        "column_quantum": int(native_quantum),
        "fractions": fractions, "controls": all_controls,
        "phone_ready_s": (ready["ready_ns"] - ready["started_ns"]) / 1e9,
        "phone_phases": phases,
        "weight_read_s": (phase_times["WEIGHT_READ_READY"] - phase_times["WEIGHT_READ_BEGIN"]) / 1e6,
        "htp_init_s": (phase_times["HTP_INIT_READY"] - phase_times["HTP_INIT_BEGIN"]) / 1e6,
        "weight_upload_s": (phase_times["WEIGHT_UPLOAD_READY"] - phase_times["WEIGHT_UPLOAD_BEGIN"]) / 1e6,
        "preparation": preparation,
        "native_shape_statistics": [json.loads(line.split("S41SERVERFFNSHAPE ", 1)[1])
                                    for line in native.splitlines() if "S41SERVERFFNSHAPE {" in line],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent / "physical")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = {"schema": "llama-native-split-calibration-v1", "completed": {}, "failed": {},
               "qualification": "diagnostic only; no scheduler tickets or lease qualification",
               "whole_model_npu_reference": "not tested; 100 percent means FFN-only offload",
               "energy_boundary": "request service including prefill and decode; preparation reported separately"}
    for directory in sorted(args.root.iterdir()):
        if (directory / "RESULT.json").exists():
            summary["completed"][directory.name] = audit(directory)
        elif (directory / "FAILURE.json").exists():
            failure = read(directory / "FAILURE.json")
            summary["failed"][directory.name] = {
                "error": failure["error"], "failure_sha256": digest(directory / "FAILURE.json"),
                "completed_requests_before_failure": len(failure.get("runs", [])),
                "cleanup": read(directory / "CLEANUP.json"),
            }
    with args.output.open("x") as stream:
        json.dump(summary, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print("Audited", len(summary["completed"]), "completed configurations and", len(summary["failed"]), "preserved failures")
    print("Completed requests:", sum(sum(row["requests"] for row in result["fractions"].values())
                                     for result in summary["completed"].values()))
    print("Summary SHA256:", digest(args.output))


if __name__ == "__main__":
    main()
