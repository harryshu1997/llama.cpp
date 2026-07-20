#!/usr/bin/env python3

import contextlib
import importlib.util
import io
import pathlib
import tempfile
import types
import unittest


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "fixed_route", HERE / "run_fixed_route.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def record(route, tokens, request_index=0, batch_index=0, batch_size=1, stream_index=0):
    return {
        "status": "ok",
        "route": route,
        "request_index": request_index,
        "batch_index": batch_index,
        "batch_size": batch_size,
        "stream_index": stream_index,
        "prompt_tokens": 4,
        "requested_tokens": len(tokens),
        "generated_tokens": len(tokens),
        "eog": 0,
        "prefill_us": 10,
        "decode_us": 20,
        "request_wall_us": 31,
        "stage_a_us": 0,
        "stage_b_us": 0,
        "host_us": 30,
        "token_ids": tokens,
    }


def record_text(item, done=None):
    if done is None:
        done = {
            "status": "ok",
            "route": item["route"],
            "requests": 1,
        }
    return (
        "ROUTEJSON " + MOD.json.dumps(item) + "\n" +
        "DRIVER_DONE " + MOD.json.dumps(done) + "\n")


def run_result(route, measurement_valid=False, memory_mib=100):
    return {
        "route": route,
        "records": [record(route, [1])],
        "measurement": {
            "quality": {"valid": measurement_valid},
            "energy_nj": 100 if measurement_valid else None,
            "uncertainty_nj": 10 if measurement_valid else None,
        },
        "gpu_ready": {"memory_used_mib": memory_mib},
        "batch_metrics": {
            "median_useful_requests_per_s": 10.0,
            "p95_group_wall_us": 10,
        },
    }


def aggregate_pair(index, **overrides):
    item = {
        "pair_index": index,
        "exact_work": True,
        "completed_requests": 1,
        "generated_tokens": 1,
        "measurement_valid": True,
        "reintegrated": True,
        "control_energy_nj": 200,
        "control_uncertainty_nj": 10,
        "treatment_energy_nj": 100,
        "treatment_uncertainty_nj": 10,
        "control_power_artifact": {
            "board_uuid": "GPU-test",
            "path": f"/tmp/control-{index}.jsonl",
            "record_count": 101,
            "sha256": f"{2 * index + 1:064x}",
            "window_start_us": 0,
            "window_end_us": 10_000_000,
            "power_limit_mw": 300_000,
        },
        "treatment_power_artifact": {
            "board_uuid": "GPU-test",
            "path": f"/tmp/treatment-{index}.jsonl",
            "record_count": 101,
            "sha256": f"{2 * index + 2:064x}",
            "window_start_us": 0,
            "window_end_us": 10_000_000,
            "power_limit_mw": 300_000,
        },
        "control_power_limit_observed_mw": 300_000,
        "treatment_power_limit_observed_mw": 300_000,
        "control_process_artifact": {
            "board_uuid": "GPU-test",
            "path": f"/tmp/control-proc-{index}.jsonl",
            "record_count": 8,
            "sha256": f"{2 * index + 101:064x}",
            "window_start_us": 0,
            "window_end_us": 10_000_000,
            "driver_pid": 4242,
        },
        "treatment_process_artifact": {
            "board_uuid": "GPU-test",
            "path": f"/tmp/treatment-proc-{index}.jsonl",
            "record_count": 8,
            "sha256": f"{2 * index + 102:064x}",
            "window_start_us": 0,
            "window_end_us": 10_000_000,
            "driver_pid": 4242,
        },
        "control_thermal_binding": "NOT_APPLICABLE",
        "treatment_thermal_binding": "TREATMENT",
        "treatment_thermal_artifact": {
            "serial": "3C15AU002CL00000",
            "path": f"/tmp/treatment-therm-{index}.jsonl",
            "record_count": 10,
            "sha256": f"{index + 201:064x}",
            "window_start_us": 0,
            "window_end_us": 10_000_000,
        },
        "control_placement_artifact": {
            "path": f"/tmp/control-place-{index}.jsonl",
            "record_count": 1,
            "sha256": f"{index + 301:064x}",
        },
        "treatment_placement_artifact": {
            "path": f"/tmp/treatment-place-{index}.jsonl",
            "record_count": 2,
            "sha256": f"{index + 401:064x}",
        },
        "control_placement_expectations": control_expectations(),
        "treatment_placement_expectations": treatment_expectations(),
        "gpu_ready_memory_relief_mib": 1000,
        "control_batch_metrics": {
            "median_useful_requests_per_s": 10.0,
            "p95_group_wall_us": 10,
        },
        "treatment_batch_metrics": {
            "median_useful_requests_per_s": 12.0,
            "p95_group_wall_us": 20,
        },
    }
    item.update(overrides)
    return item


def psample(timestamp_us, power_mw, pstate="P2", utilization_pct=0,
            memory_used_mib=0, power_limit_mw=300_000,
            board_uuid="GPU-test"):
    """One raw NVML power row with the full frozen six-field schema."""
    return {
        "kind": "sample",
        "board_uuid": board_uuid,
        "timestamp_us": timestamp_us,
        "power_mw": power_mw,
        "pstate": pstate,
        "utilization_pct": utilization_pct,
        "memory_used_mib": memory_used_mib,
        "power_limit_mw": power_limit_mw,
    }


def write_power_jsonl(path, count=120, constant=True, power_limit_mw=300_000):
    """Write a real power.jsonl the reintegration path can reopen.

    Rows carry the full six-field raw schema (including power_limit_mw). A
    constant stream is structurally valid but quality-invalid (zero in-window
    updates), which is exactly the INVALID_TIMELINE case the exit-3 test needs:
    reintegration reopens it without raising and reports it invalid.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as handle:
        for index in range(count):
            sample = psample(
                index * 100_000,
                10 if constant else 10 + index,
                power_limit_mw=power_limit_mw)
            handle.write(MOD.json.dumps(
                sample, sort_keys=True, separators=(",", ":")) + "\n")
    data = path.read_bytes()
    return {
        "board_uuid": "GPU-test",
        "path": str(path.resolve()),
        "record_count": count,
        "sha256": MOD.sha256_bytes(data),
        "window_start_us": 0,
        "window_end_us": (count - 1) * 100_000,
        "power_limit_mw": power_limit_mw,
    }


def write_process_jsonl(path, driver_pid=4242, count=6, foreign=False,
                        error=False, gap_at=None, board_uuid="GPU-test"):
    """Write a real gpu_processes.jsonl the process reintegration can reopen.

    Records use the kind-tagged schema the monitor persists: sample records with
    probe timestamps and observed processes, plus optional error records. A clean
    driver-only stream is valid; foreign/gap/error variants recompute as invalid.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [{
        "kind": "boundary", "board_uuid": board_uuid, "phase": "READY",
        "probe_start_us": 0, "probe_end_us": 10, "timestamp_us": 10,
        "processes": [{"pid": driver_pid, "name": "driver",
                       "memory_used_mib": 100}],
    }]
    ts = 20
    for index in range(count):
        processes = [{"pid": driver_pid, "name": "driver",
                      "memory_used_mib": 100}]
        if foreign and index == count // 2:
            processes.append({"pid": 9999, "name": "intruder",
                              "memory_used_mib": 50})
        records.append({
            "kind": "sample",
            "board_uuid": board_uuid,
            "probe_start_us": ts,
            "probe_end_us": ts + 10,
            "timestamp_us": ts + 10,
            "processes": processes,
        })
        step = 400_000 if gap_at is not None and index == gap_at else 100_000
        ts += step
    if error:
        records.append({
            "kind": "error", "board_uuid": board_uuid,
            "probe_start_us": ts, "probe_end_us": ts + 10,
            "message": "probe exceeded bound"})
        ts += 20
    records.append({
        "kind": "boundary", "board_uuid": board_uuid, "phase": "DONE",
        "probe_start_us": ts, "probe_end_us": ts + 10,
        "timestamp_us": ts + 10,
        "processes": [{"pid": driver_pid, "name": "driver",
                       "memory_used_mib": 100}],
    })
    with open(path, "w", encoding="ascii") as handle:
        for record in records:
            handle.write(MOD.json.dumps(
                record, sort_keys=True, separators=(",", ":")) + "\n")
    data = path.read_bytes()
    sample_ts = [r["timestamp_us"] for r in records if r["kind"] == "sample"]
    return {
        "board_uuid": board_uuid,
        "path": str(path.resolve()),
        "record_count": len(records),
        "sha256": MOD.sha256_bytes(data),
        "window_start_us": sample_ts[0],
        "window_end_us": sample_ts[-1],
        "driver_pid": driver_pid,
    }


def _test_boot_uuid(value):
    if MOD.BOOT_UUID_PATTERN.fullmatch(value):
        return value
    raw = MOD.hashlib.sha256(value.encode("ascii")).hexdigest()[:32]
    return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"


def write_thermal_jsonl(path, serial="3C15AU002CL00000", boot="epoch1000",
                        count=8, status=0, empty_sensors=False, error=False,
                        gap_at=None):
    """Write a real phone_thermal.jsonl the thermal reintegration can reopen."""
    path.parent.mkdir(parents=True, exist_ok=True)
    boot = _test_boot_uuid(boot)
    records = [{"kind": "header", "boot_id": boot, "serial": serial}]
    ts = 0
    for index in range(count):
        temps = [] if empty_sensors else [
            {"name": "cpu-0-0", "temp_millic": 42000 + index * 100}]
        records.append({
            "kind": "sample", "timestamp_us": ts,
            "thermal_status": status, "temperatures": temps})
        step = 3_000_000 if gap_at is not None and index == gap_at else 500_000
        ts += step
    if error:
        records.append({
            "kind": "error", "timestamp_us": ts, "message": "no thermal status"})
        ts += 500_000
    records.append({
        "kind": "footer", "boot_id": boot, "serial": serial,
        "timestamp_us": ts})
    with open(path, "w", encoding="ascii") as handle:
        for record in records:
            handle.write(MOD.json.dumps(
                record, sort_keys=True, separators=(",", ":")) + "\n")
    data = path.read_bytes()
    sample_ts = [r["timestamp_us"] for r in records if r["kind"] == "sample"]
    return {
        "serial": serial,
        "path": str(path.resolve()),
        "record_count": len(records),
        "sha256": MOD.sha256_bytes(data),
        "window_start_us": sample_ts[0],
        "window_end_us": sample_ts[-1],
    }


def placement_cert(role="phone_stage", pid=1000, status="PLACEMENT_OK",
                   compute_nodes=200, cpu_fallback_nodes=0,
                   observed_backends=None, expected_backend="HTP0",
                   layer_start=0, layer_end=2, n_layer=48, route="stagenet",
                   other_nodes=120, run_rc=0):
    if observed_backends is None:
        observed_backends = ["HTP0"]
    buffer_counts = {}
    remaining = compute_nodes
    if cpu_fallback_nodes:
        buffer_counts["CPU"] = cpu_fallback_nodes
        remaining -= cpu_fallback_nodes
    if observed_backends:
        each = remaining // len(observed_backends) if observed_backends else 0
        for index, backend in enumerate(observed_backends):
            amount = each if index + 1 < len(observed_backends) else remaining - each * index
            buffer_counts[backend] = buffer_counts.get(backend, 0) + amount
    return {
        "schema": "layersplit-scheduled-placement-v2",
        "role": role, "mode": route,
        "layer_start": layer_start, "layer_end": layer_end, "n_layer": n_layer,
        "pid": pid, "run_rc": run_rc, "compute_nodes": compute_nodes,
        "copy_nodes": 0, "metadata_nodes": other_nodes,
        "missing_buffer_compute_nodes": 0,
        "compute_by_buffer_type": buffer_counts,
        "compute_by_op": {"MUL_MAT": compute_nodes},
        "compute_by_op_and_buffer": {"MUL_MAT": buffer_counts},
        "copy_by_buffer_type": {}, "status": (
            "SCHEDULED_PLACEMENT_OK" if status == "PLACEMENT_OK" else status),
    }


def write_placement_jsonl(path, certs, sources=None):
    """Write a real placement_certs.jsonl the placement reintegration reopens."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if sources is None:
        sources = [
            "host" if cert["role"] in ("monodriver", "host_tail")
            else "phone:3C15AU002CL00000" for cert in certs]
    records = sorted(
        [{"source": source, "certificate": cert}
         for source, cert in zip(sources, certs)],
        key=lambda item: item["source"])
    with open(path, "w", encoding="ascii") as handle:
        for record_obj in records:
            handle.write(MOD.json.dumps(
                record_obj, sort_keys=True, separators=(",", ":")) + "\n")
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "record_count": len(records),
        "sha256": MOD.sha256_bytes(data),
    }


def treatment_expectations(pid=1000):
    return [{
        "source": "host", "role": "host_tail", "mode": "pipedriver",
        "layer_start": 2, "layer_end": 48, "n_layer": 48,
        "default_compute_buffer_type": "CUDA0",
        "compute_buffer_overrides": {
            "GET_ROWS": ["CUDA0", "CUDA_Host"]}, "pid": pid,
    }, {
        "source": "phone:3C15AU002CL00000", "role": "phone_stage",
        "mode": "stagenet", "layer_start": 0, "layer_end": 2,
        "n_layer": 48, "default_compute_buffer_type": "HTP0",
        "compute_buffer_overrides": {"GET_ROWS": ["CPU"]}, "pid": None,
    }]


def control_expectations(pid=2000):
    return [{
        "source": "host", "role": "monodriver", "mode": "monodriver",
        "layer_start": 0, "layer_end": 48, "n_layer": 48,
        "default_compute_buffer_type": "CUDA0",
        "compute_buffer_overrides": {
            "GET_ROWS": ["CUDA0", "CUDA_Host"]}, "pid": pid,
    }]


def treatment_placement(path, pid=1000):
    return write_placement_jsonl(path, [
        placement_cert(role="host_tail", pid=pid, expected_backend="CUDA",
                       observed_backends=["CUDA0"], layer_start=2, layer_end=48,
                       route="pipedriver"),
        placement_cert(role="phone_stage", pid=pid + 1,
                       expected_backend="HTP0", observed_backends=["HTP0"],
                       layer_start=0, layer_end=2, route="stagenet"),
    ])


def control_placement(path, pid=2000):
    return write_placement_jsonl(path, [
        placement_cert(role="monodriver", pid=pid, expected_backend="CUDA",
                       observed_backends=["CUDA0"], layer_start=0, layer_end=48,
                       route="monodriver"),
    ])


def build_measured_pairs(temp, count=8):
    """`count` pairs whose four evidence types are real, valid, on-disk bytes.

    Stored energy/uncertainty are set to the recomputed values so a subsequent
    reverify_pairs of an untouched pair agrees (no disagreement raise). Flip any
    one artifact to an invalid variant to prove that evidence type is load-bearing.
    """
    temp = pathlib.Path(temp)
    pairs = [aggregate_pair(index) for index in range(count)]
    for index, pair in enumerate(pairs):
        for offset, side in enumerate(("control", "treatment")):
            power = write_power_jsonl(
                temp / f"pw-{index}-{side}.jsonl",
                count=140 + index * 2 + offset, constant=False)
            pair[f"{side}_power_artifact"] = power
            recomputed = MOD.reintegrate_power_artifact(power, "p")
            pair[f"{side}_energy_nj"] = recomputed["energy_nj"]
            pair[f"{side}_uncertainty_nj"] = recomputed["uncertainty_nj"]
            pair[f"{side}_process_artifact"] = write_process_jsonl(
                temp / f"pr-{index}-{side}.jsonl",
                driver_pid=4242 + index * 2 + offset)
        pair["treatment_thermal_artifact"] = write_thermal_jsonl(
            temp / f"th-{index}.jsonl", boot=f"epoch{index}")
        pair["control_placement_artifact"] = control_placement(
            temp / f"cp-{index}.jsonl", pid=2000 + index * 10)
        pair["control_placement_expectations"] = control_expectations(
            pid=2000 + index * 10)
        pair["treatment_placement_artifact"] = treatment_placement(
            temp / f"tp-{index}.jsonl", pid=3000 + index * 10)
        pair["treatment_placement_expectations"] = treatment_expectations(
            pid=3000 + index * 10)
        pair.pop("reintegrated", None)
    return pairs


def measure_argv(overrides=None):
    """A frozen measurement CLI that parses, plus per-field overrides.

    Each key is a flag; a string value replaces it, True marks a store_true flag,
    and None removes the flag entirely (used to test a missing required flag).
    """
    base = {
        "--host-bin": "/tmp/host-bin",
        "--host-model": "/tmp/model.gguf",
        "--gpu-uuid": MOD.S11_E0_GPU_UUID,
        "--output": "/tmp/output",
        "--route": "op15",
        "--batch-size": str(MOD.S11_E0_BATCH_SIZE),
        "--driver-context": str(MOD.S11_E0_DRIVER_CONTEXT),
        "--driver-max-prefill": str(MOD.S11_E0_DRIVER_MAX_PREFILL),
        "--op15-end": str(MOD.S11_E0_OP15_END),
        "--warmups": "2",
        "--pairs": str(MOD.MIN_MEASUREMENT_PAIRS),
        "--n-gen": str(MOD.S11_E0_GENERATED_TOKENS),
        "--requests": "8",
        "--slo-p95-us": "100",
        "--host-ngl": "99",
        "--phone-ngl": "99",
        "--phone-backend": "HTP0",
        "--phone-decode-no-fa": True,
        "--chat": True,
        "--phone-thermal-logger": True,
        "--placement-cert": True,
        "--measure": True,
    }
    for flag, value in (overrides or {}).items():
        if value is None:
            base.pop(flag, None)
        else:
            base[flag] = value
    argv = []
    for flag, value in base.items():
        if value is True:
            argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return argv


class FixedRouteTests(unittest.TestCase):
    def test_v3_record_versions_are_distinct(self):
        self.assertEqual(MOD.PLAN_SCHEMA, "s11-fixed-route-poc-v4")
        self.assertEqual(MOD.RESULT_SCHEMA, "s11-fixed-route-poc-result-v4")
        self.assertEqual(MOD.FAILURE_SCHEMA, "s11-fixed-route-poc-failure-v4")
        self.assertEqual(MOD.ROUTE_RECORD_SCHEMA, "layersplit-route-v2")

    def test_order_is_abba(self):
        order = MOD.paired_order(2, MOD.ROUTE_OP15)
        self.assertEqual(
            [slot["route"] for slot in order],
            [MOD.ROUTE_CONTROL, MOD.ROUTE_OP15,
             MOD.ROUTE_OP15, MOD.ROUTE_CONTROL])

    def test_same_work_exact(self):
        ok, reason = MOD.same_work(
            [record(MOD.ROUTE_CONTROL, [1, 2])],
            [record(MOD.ROUTE_OP15, [1, 2])])
        self.assertTrue(ok)
        self.assertEqual(reason, "exact")

    def test_same_work_rejects_token_difference(self):
        ok, reason = MOD.same_work(
            [record(MOD.ROUTE_CONTROL, [1, 2])],
            [record(MOD.ROUTE_OP15, [1, 3])])
        self.assertFalse(ok)
        self.assertEqual(reason, "request_0:token_ids")

    def test_integrate_zoh(self):
        samples = [psample(0, 10), psample(10, 20), psample(20, 30)]
        self.assertEqual(MOD.integrate_zoh(samples, 5, 15), 150)

    def test_nvml_uncertainty_includes_average_boundaries(self):
        result = MOD.nvml_uncertainty(10_000_000, 300_000)
        self.assertEqual(result["nvml_accuracy"], 50_000_000_000)
        self.assertEqual(result["one_second_average_boundaries"], 600_000_000_000)
        self.assertEqual(result["total"], 650_000_000_000)

    def test_quality_rejects_oversampled_constant_sensor(self):
        samples = [psample(index * 100_000, 10) for index in range(120)]
        quality = MOD.measurement_quality(
            samples, samples[0]["timestamp_us"], samples[-1]["timestamp_us"])
        self.assertFalse(quality["valid"])
        self.assertIn("E_UPDATES:0", quality["reasons"])

    def test_quality_accepts_changing_sensor(self):
        samples = [psample(index * 100_000, 10 + index) for index in range(105)]
        quality = MOD.measurement_quality(
            samples, samples[0]["timestamp_us"], samples[-1]["timestamp_us"])
        self.assertTrue(quality["valid"])
        self.assertEqual(quality["independent_updates"], 103)

    def test_quality_records_pstate_transitions_as_outcome(self):
        samples = [psample(0, 10, pstate="P8")]
        samples.extend(
            psample(index * 100_000, 10 + index, pstate="P0")
            for index in range(1, 105))
        quality = MOD.measurement_quality(
            samples, 50_000, samples[-2]["timestamp_us"])
        self.assertTrue(quality["valid"])
        self.assertEqual(quality["pstates"], ["P0", "P8"])
        self.assertEqual(quality["pstate_transitions"], 1)
        self.assertEqual(
            quality["pstate_policy"],
            "OBSERVED_OUTCOME_TRANSITIONS_ALLOWED")

    def test_quality_ignores_transition_at_trailing_bracket(self):
        samples = [
            psample(0, 10, pstate="P0"),
            psample(10, 11, pstate="P0"),
            psample(20, 12, pstate="P8"),
        ]
        original = MOD.MIN_INDEPENDENT_UPDATES
        try:
            MOD.MIN_INDEPENDENT_UPDATES = 0
            quality = MOD.measurement_quality(samples, 0, 20)
        finally:
            MOD.MIN_INDEPENDENT_UPDATES = original
        self.assertTrue(quality["valid"])
        self.assertEqual(quality["pstates"], ["P0"])
        self.assertEqual(quality["pstate_transitions"], 0)

    def test_phone_boundary_quality_is_fail_closed(self):
        valid_state = {
            "thermal_status": 0,
            "current_temperatures": [{
                "name": "skin",
                "type": 3,
                "status": 0,
                "value_millic": 30000,
            }],
        }
        quality = MOD.phone_boundary_quality(
            ["phone"], {"phone": valid_state}, {"phone": valid_state})
        self.assertTrue(quality["valid"])
        self.assertEqual(quality["coverage"], "BOUNDARY_ONLY")

        invalid_before = dict(valid_state)
        invalid_before["thermal_status"] = False
        invalid_after = dict(valid_state)
        invalid_after["current_temperatures"] = []
        quality = MOD.phone_boundary_quality(
            ["phone"],
            {"phone": invalid_before},
            {"phone": invalid_after})
        self.assertFalse(quality["valid"])
        self.assertIn(
            "E_PHONE_THERMAL_STATUS_BEFORE:phone", quality["reasons"])
        self.assertIn(
            "E_PHONE_TEMPERATURES_AFTER:phone", quality["reasons"])

        quality = MOD.phone_boundary_quality(
            ["phone"], {}, {"phone": valid_state})
        self.assertFalse(quality["valid"])
        self.assertIn("E_PHONE_STATE_BEFORE:phone", quality["reasons"])

    def test_power_samples_fail_closed(self):
        valid = [psample(0, 10), psample(10, 20), psample(20, 30)]
        mutations = []
        item = [dict(sample) for sample in valid]
        item[1]["timestamp_us"] = 0
        mutations.append(item)
        item = [dict(sample) for sample in valid]
        item[1]["timestamp_us"] = True
        mutations.append(item)
        item = [dict(sample) for sample in valid]
        item[1]["power_mw"] = -1
        mutations.append(item)
        item = [dict(sample) for sample in valid]
        item[1]["pstate"] = "P16"
        mutations.append(item)
        # exact-integer-type rejections for the newly required raw fields
        item = [dict(sample) for sample in valid]
        item[1]["utilization_pct"] = 101
        mutations.append(item)
        item = [dict(sample) for sample in valid]
        item[1]["utilization_pct"] = True
        mutations.append(item)
        item = [dict(sample) for sample in valid]
        item[1]["memory_used_mib"] = -1
        mutations.append(item)
        item = [dict(sample) for sample in valid]
        item[1]["power_limit_mw"] = 0
        mutations.append(item)
        item = [dict(sample) for sample in valid]
        item[1]["power_limit_mw"] = 300000.0
        mutations.append(item)
        item = [dict(sample) for sample in valid]
        del item[1]["power_limit_mw"]
        mutations.append(item)
        item = [dict(sample) for sample in valid]
        item[1]["extra_field"] = 1
        mutations.append(item)
        for samples in mutations:
            with self.subTest(samples=samples):
                with self.assertRaises(MOD.ExperimentError):
                    MOD.integrate_zoh(samples, 5, 15)

    def test_parse_records_requires_done(self):
        line = "ROUTEJSON " + MOD.json.dumps(record(MOD.ROUTE_CONTROL, [1])) + "\n"
        with self.assertRaises(MOD.ExperimentError):
            MOD.parse_route_records(line)

    def test_parse_records_rejects_negative_timing(self):
        item = record(MOD.ROUTE_CONTROL, [1])
        item["decode_us"] = -1
        text = (
            "ROUTEJSON " + MOD.json.dumps(item) + "\n" +
            "DRIVER_DONE " + MOD.json.dumps({
                "status": "ok",
                "route": MOD.ROUTE_CONTROL,
                "requests": 1,
            }) + "\n")
        with self.assertRaises(MOD.ExperimentError):
            MOD.parse_route_records(text)

    def test_parse_records_rejects_duplicate_keys(self):
        item = record(MOD.ROUTE_CONTROL, [1])
        encoded = MOD.json.dumps(item)
        duplicate_record = encoded.replace(
            '{"status": "ok"', '{"status": "bad", "status": "ok"', 1)
        text = (
            "ROUTEJSON " + duplicate_record + "\n" +
            'DRIVER_DONE {"status":"ok","route":"SERVER_ONLY",'
            '"requests":1}\n')
        with self.assertRaises(MOD.ExperimentError):
            MOD.parse_route_records(text)

        text = (
            "ROUTEJSON " + encoded + "\n" +
            'DRIVER_DONE {"status":"ok","route":"SERVER_ONLY",'
            '"requests":1,"requests":1}\n')
        with self.assertRaises(MOD.ExperimentError):
            MOD.parse_route_records(text)

    def test_parse_records_rejects_invalid_done_schema(self):
        item = record(MOD.ROUTE_CONTROL, [1])
        invalid_done = [
            {"status": "ok", "route": MOD.ROUTE_CONTROL, "requests": True},
            {"status": "ok", "route": MOD.ROUTE_CONTROL, "requests": 0},
            {
                "status": "ok",
                "route": MOD.ROUTE_CONTROL,
                "requests": 1,
                "extra": 1,
            },
        ]
        for done in invalid_done:
            with self.subTest(done=done):
                with self.assertRaises(MOD.ExperimentError):
                    MOD.parse_route_records(record_text(item, done))

    def test_batch_metrics_uses_group_timing_once(self):
        records = [
            record(MOD.ROUTE_CONTROL, [1], 0, 0, 2, 0),
            record(MOD.ROUTE_CONTROL, [1], 1, 0, 2, 1),
            record(MOD.ROUTE_CONTROL, [1], 2, 1, 2, 0),
            record(MOD.ROUTE_CONTROL, [1], 3, 1, 2, 1),
        ]
        records[0]["request_wall_us"] = records[1]["request_wall_us"] = 100
        records[2]["request_wall_us"] = records[3]["request_wall_us"] = 300
        metrics = MOD.batch_metrics(records)
        self.assertEqual(metrics["group_count"], 2)
        self.assertEqual(metrics["median_group_wall_us"], 200)
        self.assertEqual(metrics["p95_group_wall_us"], 300)
        self.assertEqual(metrics["median_useful_requests_per_s"], 10000.0)

    def test_batch_metrics_rejects_split_group_timing(self):
        records = [
            record(MOD.ROUTE_CONTROL, [1], 0, 0, 2, 0),
            record(MOD.ROUTE_CONTROL, [1], 1, 0, 2, 1),
        ]
        records[1]["request_wall_us"] += 1
        with self.assertRaises(MOD.ExperimentError):
            MOD.batch_metrics(records)

    def test_summarize_pairs_accepts_complete_order(self):
        order = MOD.paired_order(2, MOD.ROUTE_OP15)
        runs = [run_result(slot["route"]) for slot in order]
        pairs = MOD.summarize_pairs(order, runs)
        self.assertEqual([pair["pair_index"] for pair in pairs], [0, 1])

    def test_summarize_pairs_rejects_length_or_route_mismatch(self):
        order = MOD.paired_order(1, MOD.ROUTE_OP15)
        with self.assertRaises(MOD.ExperimentError):
            MOD.summarize_pairs(order, [run_result(MOD.ROUTE_CONTROL)])
        runs = [
            run_result(MOD.ROUTE_OP15),
            run_result(MOD.ROUTE_CONTROL),
        ]
        with self.assertRaises(MOD.ExperimentError):
            MOD.summarize_pairs(order, runs)

    def test_summarize_pairs_rejects_duplicate_or_noncontiguous_slots(self):
        duplicate = [
            {"pair_index": 0, "pair_position": 0, "route": MOD.ROUTE_CONTROL},
            {"pair_index": 0, "pair_position": 0, "route": MOD.ROUTE_OP15},
        ]
        runs = [run_result(slot["route"]) for slot in duplicate]
        with self.assertRaises(MOD.ExperimentError):
            MOD.summarize_pairs(duplicate, runs)

        noncontiguous = [
            {"pair_index": 1, "pair_position": 0, "route": MOD.ROUTE_CONTROL},
            {"pair_index": 1, "pair_position": 1, "route": MOD.ROUTE_OP15},
        ]
        runs = [run_result(slot["route"]) for slot in noncontiguous]
        with self.assertRaises(MOD.ExperimentError):
            MOD.summarize_pairs(noncontiguous, runs)

    def test_aggregate_never_authorizes_formal_claim(self):
        pairs = [aggregate_pair(index) for index in range(8)]
        result = MOD.aggregate_result(pairs, True, 100)
        self.assertEqual(result["formal_claim"], "NONE")
        self.assertTrue(result["board_relief_observed"])
        self.assertTrue(result["board_relief_ge_10pct"])
        self.assertEqual(
            result["label"],
            "GPU_BOARD_DIAGNOSTIC_10PCT_OBSERVED_TOTAL_ENERGY_UNKNOWN")
        self.assertEqual(result["completed_requests_sum"], 8)
        self.assertEqual(result["generated_tokens_sum"], 8)
        self.assertEqual(result["gpu_board_uuid"], "GPU-test")
        self.assertEqual(result["phone_energy_status"], "UNKNOWN")
        self.assertEqual(result["total_system_energy_status"], "UNKNOWN")

    def test_aggregate_distinguishes_small_relief_and_failure(self):
        pairs = [aggregate_pair(index, treatment_energy_nj=170) for index in range(8)]
        below_gate = MOD.aggregate_result(pairs, True, 100)
        self.assertTrue(below_gate["board_relief_observed"])
        self.assertFalse(below_gate["board_relief_ge_10pct"])
        self.assertEqual(
            below_gate["label"],
            "GPU_BOARD_DIAGNOSTIC_BELOW_10PCT_TOTAL_ENERGY_UNKNOWN")

        pairs = [aggregate_pair(index, treatment_energy_nj=200) for index in range(8)]
        failed = MOD.aggregate_result(pairs, True, 100)
        self.assertFalse(failed["board_relief_observed"])
        self.assertFalse(failed["board_relief_ge_10pct"])
        self.assertEqual(failed["label"], "GPU_BOARD_DIAGNOSTIC_RELIEF_FAIL")

    def test_aggregate_reports_slo_failure_separately(self):
        pairs = [aggregate_pair(index) for index in range(8)]
        result = MOD.aggregate_result(pairs, True, 15)
        self.assertFalse(result["slo_met_all_pairs"])
        self.assertEqual(result["label"], "GPU_BOARD_DIAGNOSTIC_SLO_FAIL")

    def test_invalid_measurement_is_fail_closed(self):
        pairs = [aggregate_pair(index) for index in range(8)]
        pairs[-1]["measurement_valid"] = False
        result = MOD.aggregate_result(pairs, True, 100)
        self.assertEqual(result["measurement_status"], "INVALID_TIMELINE")
        self.assertEqual(result["label"], "MEASUREMENT_INVALID")
        self.assertEqual(result["phone_energy_status"], "UNKNOWN")
        self.assertEqual(result["total_system_energy_status"], "UNKNOWN")
        self.assertEqual(result["formal_claim"], "NONE")

    def test_aggregate_rejects_empty_pairs(self):
        with self.assertRaises(MOD.ExperimentError):
            MOD.aggregate_result([], False)

    def test_aggregate_rejects_mixed_gpu_board_uuids(self):
        pairs = [aggregate_pair(index) for index in range(8)]
        pairs[-1]["treatment_power_artifact"]["board_uuid"] = "GPU-other"
        with self.assertRaisesRegex(
                MOD.ExperimentError, "measurement slots mix GPU board UUIDs"):
            MOD.aggregate_result(pairs, True, 100)

    def test_aggregate_rejects_invalid_pair_claim_inputs(self):
        mutations = [
            {0: {"exact_work": False}},
            {0: {"measurement_valid": 1}},
            {0: {"treatment_energy_nj": -1}},
            {0: {"gpu_ready_memory_relief_mib": -1}},
            {0: {
                "treatment_batch_metrics": {
                    "median_useful_requests_per_s": 0.0,
                },
            }},
            {0: {
                "treatment_batch_metrics": {
                    "median_useful_requests_per_s": 1.0,
                    "p95_group_wall_us": 1.0,
                },
            }},
        ]
        for changes in mutations:
            pairs = [aggregate_pair(index) for index in range(8)]
            for index, fields in changes.items():
                pairs[index].update(fields)
            with self.subTest(changes=changes):
                with self.assertRaises(MOD.ExperimentError):
                    MOD.aggregate_result(pairs, True, 100)

    def test_aggregate_rejects_duplicate_or_noncontiguous_indexes(self):
        pairs = [aggregate_pair(index) for index in range(8)]
        pairs[-1]["pair_index"] = 6
        with self.assertRaises(MOD.ExperimentError):
            MOD.aggregate_result(pairs, True, 100)

    def test_aggregate_rejects_reused_power_artifact(self):
        pairs = [aggregate_pair(index) for index in range(8)]
        pairs[1]["control_power_artifact"] = pairs[0]["control_power_artifact"]
        with self.assertRaisesRegex(MOD.ExperimentError, "reused"):
            MOD.aggregate_result(pairs, True, 100)

        pairs = [aggregate_pair(index) for index in range(8)]
        pairs[-1]["pair_index"] = 8
        with self.assertRaises(MOD.ExperimentError):
            MOD.aggregate_result(pairs, True, 100)

    def test_aggregate_accepts_functional_no_measurement(self):
        order = MOD.paired_order(1, MOD.ROUTE_OP15)
        runs = [
            run_result(
                slot["route"],
                memory_mib=100 if slot["route"] == MOD.ROUTE_CONTROL else 90)
            for slot in order
        ]
        result = MOD.aggregate_result(
            MOD.summarize_pairs(order, runs), False)
        self.assertEqual(result["measurement_status"], "NOT_RUN")
        self.assertEqual(result["gpu_ready_memory_relief_mib_min"], 10)

    def test_effective_config_binds_measurement_gates(self):
        args = types.SimpleNamespace(
            output="/tmp/out",
            host_bin="relative-bin",
            host_model="relative-model",
            deploy_phone_bin=None,
            sample_ms=100,
        )
        config = MOD.build_effective_config(args, MOD.ROUTE_OP15)
        self.assertNotIn("output", config)
        self.assertEqual(config["resolved_treatment_route"], MOD.ROUTE_OP15)
        self.assertEqual(
            config["measurement_gates"]["minimum_pairs"],
            MOD.MIN_MEASUREMENT_PAIRS)
        self.assertEqual(
            config["measurement_gates"]["pstate_format"], "P0..P15")

    def test_measurement_cli_requires_frozen_work_and_slo(self):
        base = [
            "--host-bin", "/tmp/host-bin",
            "--host-model", "/tmp/model.gguf",
            "--gpu-uuid", MOD.S11_E0_GPU_UUID,
            "--output", "/tmp/output",
            "--measure",
            "--route", "op15",
            "--chat",
            "--batch-size", "8",
            "--driver-context", "96",
            "--driver-max-prefill", "64",
            "--warmups", "2",
            "--phone-thermal-logger",
            "--placement-cert",
            "--phone-decode-no-fa",
        ]
        invalid_suffixes = [
            ["--pairs", "7", "--n-gen", "32", "--slo-p95-us", "100"],
            ["--pairs", "8", "--n-gen", "31", "--slo-p95-us", "100"],
            ["--pairs", "8", "--n-gen", "32", "--requests", "4097",
             "--slo-p95-us", "100"],
            ["--pairs", "8", "--n-gen", "32"],
        ]
        for suffix in invalid_suffixes:
            with self.subTest(suffix=suffix):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        MOD.parse_args(base + suffix)

        valid = MOD.parse_args(
            base + ["--pairs", "8", "--n-gen", "32", "--requests", "8",
                    "--slo-p95-us", "100"])
        self.assertEqual(valid.route, "op15")
        self.assertEqual(valid.batch_size, MOD.S11_E0_BATCH_SIZE)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                MOD.parse_args([
                    "--host-bin", "/tmp/host-bin",
                    "--host-model", "/tmp/model.gguf",
                    "--gpu-uuid", "GPU-test",
                    "--output", "/tmp/output",
                    "--slo-p95-us", "100",
                ])

    def test_main_binds_v2_models_source_config_and_measurement(self):
        original_connected = MOD.connected_serials
        original_query_gpu = MOD.query_gpu
        original_remote_file_info = MOD.remote_file_info
        original_run_route = MOD.run_route
        remote_hash_requests = []
        try:
            MOD.connected_serials = lambda: {"3C15AU002CL00000"}
            MOD.query_gpu = lambda _uuid: {
                "uuid": "GPU-test",
                "pstate": "P8",
                "power_mw": 10,
                "power_limit_mw": 300000,
                "utilization_pct": 0,
                "memory_used_mib": 0,
            }

            def remote_file_info(_serial, path, include_hash):
                remote_hash_requests.append((path, include_hash))
                return {
                    "path": path,
                    "bytes": 1,
                    "sha256": "a" * 64 if include_hash else None,
                }

            MOD.remote_file_info = remote_file_info

            def run_route(_args, route, _run_dir):
                memory = 100 if route == MOD.ROUTE_CONTROL else 90
                result = run_result(route, memory_mib=memory)
                result.update({
                    "schema": MOD.RUN_SCHEMA,
                    "route_record_schema": MOD.ROUTE_RECORD_SCHEMA,
                    "token_sha256": "b" * 64,
                })
                return result

            MOD.run_route = run_route
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = pathlib.Path(temp_dir)
                host_bin = temp / "host-bin"
                host_model = temp / "model.gguf"
                output = temp / "output"
                host_bin.write_bytes(b"bin")
                host_model.write_bytes(b"model")
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = MOD.main([
                        "--host-bin", str(host_bin),
                        "--host-model", str(host_model),
                        "--gpu-uuid", "GPU-test",
                        "--output", str(output),
                        "--sample-ms", "77",
                    ])
                self.assertEqual(rc, 0)
                plan = MOD.json.loads((output / "plan.json").read_text())
                summary = MOD.json.loads((output / "summary.json").read_text())
                self.assertEqual(plan["schema"], MOD.PLAN_SCHEMA)
                self.assertEqual(
                    plan["route_record_schema"], MOD.ROUTE_RECORD_SCHEMA)
                self.assertEqual(summary["schema"], MOD.RESULT_SCHEMA)
                self.assertEqual(
                    summary["route_record_schema"], MOD.ROUTE_RECORD_SCHEMA)
                self.assertEqual(
                    plan["host"]["model_sha256"],
                    MOD.sha256_file(host_model))
                self.assertTrue(remote_hash_requests)
                self.assertTrue(all(item[1] for item in remote_hash_requests))
                self.assertEqual(
                    plan["implementation"]["runner_sha256"],
                    MOD.sha256_file(MOD.RUNNER_PATH))
                self.assertEqual(
                    plan["implementation"]["layersplit_source_sha256"],
                    MOD.sha256_file(MOD.LAYERSPLIT_SOURCE_PATH))
                self.assertEqual(
                    plan["implementation"]["llama_model_source_sha256"],
                    MOD.sha256_file(MOD.LLAMA_MODEL_SOURCE_PATH))
                self.assertEqual(
                    plan["implementation"]["effective_config_sha256"],
                    MOD.sha256_bytes(
                        MOD.canonical_bytes(plan["effective_config"])))
                self.assertEqual(
                    plan["measurement_config"]["sample_interval_ms"], 77)
                self.assertEqual(
                    plan["measurement_config"]["phone_thermal_coverage"],
                    "TREATMENT_CONTINUOUS_ON_DEVICE")
        finally:
            MOD.connected_serials = original_connected
            MOD.query_gpu = original_query_gpu
            MOD.remote_file_info = original_remote_file_info
            MOD.run_route = original_run_route

    def _valid_power_artifact(self, path, count=140):
        # a changing sensor so recomputed quality is valid and energy is real
        artifact = write_power_jsonl(path, count=count, constant=False)
        return artifact

    def test_reintegration_recomputes_energy_from_hashed_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "power.jsonl"
            artifact = self._valid_power_artifact(path)
            recomputed = MOD.reintegrate_power_artifact(artifact, "probe")
            self.assertTrue(recomputed["quality_valid"])
            self.assertEqual(recomputed["recomputed_sample_count"],
                             artifact["record_count"])
            self.assertEqual(recomputed["recomputed_sha256"], artifact["sha256"])
            # matches a direct integration of the same on-disk samples
            samples = [
                MOD.strict_json_loads(line, "sample")
                for line in path.read_text().split("\n") if line
            ]
            self.assertEqual(
                recomputed["energy_nj"],
                MOD.integrate_zoh(samples, artifact["window_start_us"],
                                  artifact["window_end_us"]))

    def test_reintegration_fails_closed_on_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "power.jsonl"
            artifact = self._valid_power_artifact(path)
            artifact["sha256"] = "0" * 64
            with self.assertRaisesRegex(MOD.ExperimentError, "sha256 mismatch"):
                MOD.reintegrate_power_artifact(artifact, "probe")

    def test_reintegration_fails_closed_on_sample_count_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "power.jsonl"
            artifact = self._valid_power_artifact(path)
            artifact["record_count"] += 1
            with self.assertRaisesRegex(
                    MOD.ExperimentError, "record-count mismatch"):
                MOD.reintegrate_power_artifact(artifact, "probe")

    def test_reintegration_fails_closed_on_tampered_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "power.jsonl"
            artifact = self._valid_power_artifact(path)
            lines = path.read_text().split("\n")
            lines[0] = lines[0].replace(
                '"memory_used_mib":0,',
                '"memory_used_mib":0,"memory_used_mib":1,', 1)
            path.write_text("\n".join(lines), encoding="ascii")
            artifact["sha256"] = MOD.sha256_bytes(path.read_bytes())
            with self.assertRaisesRegex(MOD.ExperimentError, "duplicate JSON key"):
                MOD.reintegrate_power_artifact(artifact, "probe")

    def test_reintegration_reports_invalid_stream_without_raising(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "power.jsonl"
            artifact = write_power_jsonl(path, count=120, constant=True)
            recomputed = MOD.reintegrate_power_artifact(artifact, "probe")
            self.assertFalse(recomputed["quality_valid"])
            self.assertIsNone(recomputed["energy_nj"])
            self.assertIn("E_UPDATES:0", recomputed["quality_reasons"])

    def test_reverify_pairs_overwrites_valid_energy_with_recomputed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            pairs = [aggregate_pair(index) for index in range(2)]
            for index, pair in enumerate(pairs):
                for side in ("control", "treatment"):
                    path = temp / f"{side}-{index}.jsonl"
                    artifact = self._valid_power_artifact(path)
                    pair[f"{side}_power_artifact"] = artifact
                    pair[f"{side}_process_artifact"] = write_process_jsonl(
                        temp / f"{side}-proc-{index}.jsonl",
                        driver_pid=4242 + index)
                    recomputed = MOD.reintegrate_power_artifact(artifact, "p")
                    pair[f"{side}_energy_nj"] = recomputed["energy_nj"]
                    pair[f"{side}_uncertainty_nj"] = recomputed["uncertainty_nj"]
                pair["treatment_thermal_artifact"] = write_thermal_jsonl(
                    temp / f"therm-{index}.jsonl", boot=f"epoch{index}")
                pair["control_placement_artifact"] = control_placement(
                    temp / f"cplace-{index}.jsonl", pid=2000 + index * 10)
                pair["control_placement_expectations"] = control_expectations(
                    pid=2000 + index * 10)
                pair["treatment_placement_artifact"] = treatment_placement(
                    temp / f"tplace-{index}.jsonl", pid=3000 + index * 10)
                pair["treatment_placement_expectations"] = treatment_expectations(
                    pid=3000 + index * 10)
                pair.pop("reintegrated", None)
            verified = MOD.reverify_pairs(pairs)
            for pair in verified:
                self.assertTrue(pair["reintegrated"])
                self.assertTrue(pair["measurement_valid"])
                self.assertIn("reintegration", pair)
                self.assertEqual(pair["control_power_limit_observed_mw"], 300_000)

    def test_reverify_pairs_fails_closed_when_valid_energy_disagrees(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            pair = aggregate_pair(0)
            for side in ("control", "treatment"):
                path = temp / f"{side}.jsonl"
                pair[f"{side}_power_artifact"] = self._valid_power_artifact(path)
                pair[f"{side}_process_artifact"] = write_process_jsonl(
                    temp / f"{side}-proc.jsonl", driver_pid=4242)
            pair["treatment_thermal_artifact"] = write_thermal_jsonl(
                temp / "therm.jsonl")
            pair["control_placement_artifact"] = control_placement(
                temp / "cplace.jsonl")
            pair["treatment_placement_artifact"] = treatment_placement(
                temp / "tplace.jsonl")
            # stored energy left at the aggregate_pair default (200/100), which
            # does not match the recomputed value from the on-disk stream
            pair.pop("reintegrated", None)
            with self.assertRaisesRegex(MOD.ExperimentError, "disagreement"):
                MOD.reverify_pairs([pair])

    def test_aggregate_rejects_unreintegrated_measured_pair(self):
        pairs = [aggregate_pair(index) for index in range(8)]
        pairs[3].pop("reintegrated")
        with self.assertRaisesRegex(
                MOD.ExperimentError, "without reintegration"):
            MOD.aggregate_result(pairs, True, 100)

    def _process_samples(self, foreign=False, gap=False, count=6):
        samples = []
        for index in range(count):
            step = 100_000 if not (gap and index == count - 1) else 400_000
            timestamp = (samples[-1]["timestamp_us"] + step) if samples else 0
            processes = [{"pid": 4242, "name": "driver", "memory_used_mib": 100}]
            if foreign and index == count // 2:
                processes.append(
                    {"pid": 9999, "name": "intruder", "memory_used_mib": 50})
            samples.append({"timestamp_us": timestamp, "processes": processes})
        return samples

    def test_evaluate_process_telemetry_accepts_driver_only(self):
        quality = MOD.evaluate_process_telemetry(
            self._process_samples(), driver_pid=4242)
        self.assertTrue(quality["valid"])
        self.assertEqual(quality["foreign_pids"], [])

    def test_evaluate_process_telemetry_detects_foreign_pid(self):
        quality = MOD.evaluate_process_telemetry(
            self._process_samples(foreign=True), driver_pid=4242)
        self.assertFalse(quality["valid"])
        self.assertEqual(quality["foreign_pids"], [9999])
        self.assertIn("E_GPU_CONTAMINATION:[9999]", quality["reasons"])

    def test_evaluate_process_telemetry_detects_coverage_gap(self):
        quality = MOD.evaluate_process_telemetry(
            self._process_samples(gap=True), driver_pid=4242)
        self.assertFalse(quality["valid"])
        self.assertTrue(any(r.startswith("E_PROCESS_GAP")
                            for r in quality["reasons"]))

    def test_evaluate_process_telemetry_requires_coverage(self):
        quality = MOD.evaluate_process_telemetry(
            self._process_samples(count=1), driver_pid=4242)
        self.assertFalse(quality["valid"])
        self.assertIn("E_PROCESS_MONITOR_UPDATES:1", quality["reasons"])

    def test_evaluate_process_telemetry_fails_on_probe_error(self):
        quality = MOD.evaluate_process_telemetry(
            self._process_samples(), driver_pid=4242, errors=["boom"])
        self.assertFalse(quality["valid"])
        self.assertIn("E_PROCESS_MONITOR_ERROR:1", quality["reasons"])

    def test_gpu_process_monitor_collects_hashes_and_evaluates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "gpu_processes.jsonl"
            monitor = MOD.GpuProcessMonitor(
                "GPU-test", path, interval_ms=10,
                probe=lambda: [{"pid": 4242, "name": "driver",
                                "memory_used_mib": 100}])
            monitor.start()
            MOD.time.sleep(0.08)
            telemetry = monitor.stop()
            self.assertGreaterEqual(telemetry["record_count"], 2)
            self.assertGreaterEqual(len(telemetry["samples"]), 2)
            self.assertEqual(telemetry["sha256"], MOD.sha256_file(path))
            self.assertEqual(telemetry["errors"], [])
            quality = MOD.evaluate_process_telemetry(
                telemetry["samples"], driver_pid=4242)
            self.assertTrue(quality["valid"])
            for sample in telemetry["samples"]:
                self.assertEqual(sample["kind"], "sample")
                self.assertEqual(sample["timestamp_us"], sample["probe_end_us"])
                self.assertGreaterEqual(
                    sample["probe_end_us"], sample["probe_start_us"])

    def test_parse_power_sample_binds_power_limit(self):
        sample = MOD.parse_power_sample(
            "GPU-test, 120.00, P2, 55, 24000, 300.00", "GPU-test", 999)
        self.assertEqual(sample["power_mw"], 120_000)
        self.assertEqual(sample["power_limit_mw"], 300_000)
        self.assertEqual(sample["timestamp_us"], 999)
        for bad in ("GPU-other, 1, P2, 0, 0, 300", "GPU-test, 1, P2, 0, 0"):
            with self.subTest(bad=bad):
                with self.assertRaises(MOD.ExperimentError):
                    MOD.parse_power_sample(bad, "GPU-test", 0)

    def test_power_limit_invariant_cases(self):
        self.assertTrue(
            MOD.power_limit_invariant([300_000, 300_000], 300_000)["valid"])
        changed = MOD.power_limit_invariant([300_000, 250_000], 300_000)
        self.assertFalse(changed["valid"])
        self.assertTrue(any(r.startswith("E_POWER_LIMIT_CHANGED")
                            for r in changed["reasons"]))
        mismatch = MOD.power_limit_invariant([250_000], 300_000)
        self.assertFalse(mismatch["valid"])
        self.assertTrue(any(r.startswith("E_POWER_LIMIT_MISMATCH")
                            for r in mismatch["reasons"]))
        empty = MOD.power_limit_invariant([], 300_000)
        self.assertFalse(empty["valid"])
        self.assertIn("E_POWER_LIMIT_UNOBSERVED", empty["reasons"])

    def test_aggregate_rejects_mixed_power_limit_across_slots(self):
        # the 16-slot limit check now reads the limit recomputed from each
        # stream's bytes, not the artifact metadata
        pairs = [aggregate_pair(index) for index in range(8)]
        pairs[-1]["treatment_power_limit_observed_mw"] = 250_000
        with self.assertRaisesRegex(
                MOD.ExperimentError, "do not share one GPU power limit"):
            MOD.aggregate_result(pairs, True, 100)

    def test_measurement_cli_parses_the_frozen_baseline(self):
        args = MOD.parse_args(measure_argv())
        self.assertTrue(args.measure)
        self.assertEqual(args.route, "op15")
        self.assertEqual(args.gpu_uuid, MOD.S11_E0_GPU_UUID)
        self.assertEqual(args.batch_size, MOD.S11_E0_BATCH_SIZE)
        self.assertEqual(args.n_gen, MOD.S11_E0_GENERATED_TOKENS)
        self.assertEqual(args.driver_context, MOD.S11_E0_DRIVER_CONTEXT)
        self.assertEqual(args.driver_max_prefill, MOD.S11_E0_DRIVER_MAX_PREFILL)
        self.assertEqual(args.op15_end, MOD.S11_E0_OP15_END)
        self.assertEqual(args.slo_p95_us, 100)

    def test_measurement_cli_rejects_every_frozen_field(self):
        """A negative case for each measurement-frozen field (CP0).

        Every override starts from a CLI that parses and changes exactly one
        frozen field to a value the general (non-measure) rules would accept, so
        the rejection is proven to come from the measurement freeze and not from
        an unrelated bound.
        """
        rejects = {
            "route_not_op15": {"--route": "two-phone"},
            "gpu_uuid_not_selected": {"--gpu-uuid": "GPU-some-other-board"},
            "batch_size_not_8": {"--batch-size": "4"},
            "n_gen_not_32": {"--n-gen": "16"},
            "driver_context_not_96": {"--driver-context": "128"},
            "driver_max_prefill_not_64": {"--driver-max-prefill": "32"},
            "op15_end_not_2": {"--op15-end": "1"},
            "host_ngl_not_99": {"--host-ngl": "98"},
            "phone_ngl_not_99": {"--phone-ngl": "98"},
            "phone_backend_not_htp0": {"--phone-backend": "CPU"},
            "requests_over_max": {"--requests": "4104"},
            "pairs_below_min": {"--pairs": "7"},
            "slo_missing": {"--slo-p95-us": None},
            "slo_nonpositive": {"--slo-p95-us": "0"},
            "chat_missing": {"--chat": None},
            "warmups_below_two": {"--warmups": "1"},
            "thermal_logger_missing": {"--phone-thermal-logger": None},
            "placement_cert_missing": {"--placement-cert": None},
        }
        for name, override in rejects.items():
            with self.subTest(field=name):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        MOD.parse_args(measure_argv(override))

    def test_main_measure_invalid_timeline_returns_exit_3(self):
        """An invalid measured run must return exit code 3 (CP0).

        The mocked slots produce exact work but invalid measurement quality, so
        the aggregate is INVALID_TIMELINE / MEASUREMENT_INVALID. main() must exit
        3 (not 0 and not the 2 reserved for exact-work failure).
        """
        originals = (MOD.connected_serials, MOD.query_gpu,
                     MOD.remote_file_info, MOD.run_route)
        counter = [0]
        try:
            MOD.connected_serials = lambda: {"3C15AU002CL00000"}
            MOD.query_gpu = lambda _uuid: {
                "uuid": MOD.S11_E0_GPU_UUID, "pstate": "P8", "power_mw": 10,
                "power_limit_mw": 300000, "utilization_pct": 0,
                "memory_used_mib": 0,
            }
            MOD.remote_file_info = lambda _serial, path, include_hash: {
                "path": path, "bytes": 1,
                "sha256": "a" * 64 if include_hash else None,
            }

            def run_route(_args, route, run_dir):
                counter[0] += 1
                memory = 100 if route == MOD.ROUTE_CONTROL else 90
                # constant power stream -> quality-invalid (E_UPDATES:0); the
                # process stream is structurally valid so reintegration runs and
                # the invalidity comes only from the recomputed power quality
                artifact = write_power_jsonl(
                    pathlib.Path(run_dir) / "power.jsonl",
                    count=120 + counter[0])
                process_artifact = write_process_jsonl(
                    pathlib.Path(run_dir) / "gpu_processes.jsonl",
                    driver_pid=4242, count=6 + counter[0])
                is_treatment = route != MOD.ROUTE_CONTROL
                thermal_artifact = None
                if is_treatment:
                    thermal_artifact = write_thermal_jsonl(
                        pathlib.Path(run_dir) / "phone_thermal.jsonl",
                        boot=f"epoch{counter[0]}")
                    placement = treatment_placement(
                        pathlib.Path(run_dir) / "placement_certs.jsonl",
                        pid=1000 + counter[0] * 10)
                    placement_expected = treatment_expectations(
                        pid=1000 + counter[0] * 10)
                else:
                    placement = control_placement(
                        pathlib.Path(run_dir) / "placement_certs.jsonl",
                        pid=5000 + counter[0] * 10)
                    placement_expected = control_expectations(
                        pid=5000 + counter[0] * 10)
                return {
                    "schema": MOD.RUN_SCHEMA,
                    "route_record_schema": MOD.ROUTE_RECORD_SCHEMA,
                    "route": route,
                    "records": [record(route, [1])],
                    "token_sha256": "b" * 64,
                    "gpu_ready": {"memory_used_mib": memory},
                    "batch_metrics": {
                        "median_useful_requests_per_s": 10.0,
                        "p95_group_wall_us": 10,
                    },
                    "measurement": {
                        "quality": {"valid": False, "reasons": ["E_UPDATES:0"]},
                        "energy_nj": None,
                        "uncertainty_nj": None,
                        "power_artifact": artifact,
                        "process_monitor_artifact": process_artifact,
                        "phone_thermal_artifact": thermal_artifact,
                        "phone_thermal_binding": (
                            "TREATMENT" if is_treatment else "NOT_APPLICABLE"),
                        "placement_artifact": placement,
                        "placement_expectations": placement_expected,
                    },
                }

            MOD.run_route = run_route
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = pathlib.Path(temp_dir)
                host_bin = temp / "host-bin"
                host_bin.write_bytes(b"bin")
                host_model = temp / "model.gguf"
                host_model.write_bytes(b"model")
                output = temp / "output"
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = MOD.main(measure_argv({
                        "--host-bin": str(host_bin),
                        "--host-model": str(host_model),
                        "--output": str(output),
                    }))
                self.assertEqual(rc, 3)
                summary = MOD.json.loads((output / "summary.json").read_text())
                aggregate = summary["aggregate"]
                self.assertEqual(
                    aggregate["measurement_status"], "INVALID_TIMELINE")
                self.assertEqual(aggregate["label"], "MEASUREMENT_INVALID")
                self.assertEqual(aggregate["formal_claim"], "NONE")
                self.assertEqual(aggregate["phone_energy_status"], "UNKNOWN")
                self.assertEqual(
                    aggregate["total_system_energy_status"], "UNKNOWN")
        finally:
            (MOD.connected_serials, MOD.query_gpu,
             MOD.remote_file_info, MOD.run_route) = originals

    # ---- Finding 1: process-monitor evidence -------------------------------

    def test_process_monitor_stop_requires_thread_join(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "gpu_processes.jsonl"
            monitor = MOD.GpuProcessMonitor(
                "GPU-test", path, probe=lambda: [])

            class Stuck:
                def join(self, timeout=None):
                    return None

                def is_alive(self):
                    return True

            monitor._thread = Stuck()
            with self.assertRaisesRegex(
                    MOD.ExperimentError, "did not terminate"):
                monitor.stop()

    def test_process_monitor_bounds_slow_probe_and_persists_error(self):
        def slow_probe():
            raise MOD.subprocess.TimeoutExpired(["nvidia-smi"], 0.03)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "gpu_processes.jsonl"
            monitor = MOD.GpuProcessMonitor(
                "GPU-test", path, interval_ms=10, probe=slow_probe,
                probe_timeout_s=0.03)
            monitor.start()
            MOD.time.sleep(0.15)
            telemetry = monitor.stop()
            # the bound turned every hung probe into a persisted error record
            self.assertTrue(telemetry["errors"])
            records = [
                MOD.strict_json_loads(line, "record")
                for line in path.read_text().split("\n") if line
            ]
            self.assertTrue(records)
            self.assertTrue(all(r["kind"] == "error" for r in records))
            self.assertTrue(all("bound" in r["message"] for r in records))
            for record in records:
                self.assertGreaterEqual(
                    record["probe_end_us"], record["probe_start_us"])
            # quality derives entirely from the persisted bytes: reintegrating
            # the hashed artifact recovers the errors and fails closed
            artifact = {
                "board_uuid": "GPU-test",
                "path": str(path.resolve()),
                "record_count": len(records),
                "sha256": MOD.sha256_file(path),
                "window_start_us": 0,
                "window_end_us": 1,
                "driver_pid": 4242,
            }
            recomputed = MOD.reintegrate_process_artifact(artifact, "p")
            self.assertFalse(recomputed["quality_valid"])
            self.assertGreaterEqual(recomputed["error_count"], 1)

    def test_evaluate_process_requires_window_bracket(self):
        samples = [
            {"timestamp_us": (index + 1) * 100_000,
             "processes": [{"pid": 4242, "name": "driver",
                            "memory_used_mib": 100}]}
            for index in range(6)
        ]
        unbracketed = MOD.evaluate_process_telemetry(
            samples, 4242, window_start_us=0, window_end_us=10_000_000)
        self.assertFalse(unbracketed["valid"])
        self.assertIn("E_PROCESS_NO_LEADING_SAMPLE", unbracketed["reasons"])
        self.assertIn("E_PROCESS_NO_TRAILING_SAMPLE", unbracketed["reasons"])
        bracketed = MOD.evaluate_process_telemetry(
            samples, 4242, window_start_us=150_000, window_end_us=550_000)
        self.assertTrue(bracketed["valid"])

    # ---- Finding 2: process-artifact reintegration -------------------------

    def test_reintegrate_process_recomputes_from_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "gpu_processes.jsonl"
            artifact = write_process_jsonl(path, driver_pid=4242, count=6)
            recomputed = MOD.reintegrate_process_artifact(artifact, "p")
            self.assertTrue(recomputed["quality_valid"])
            self.assertEqual(recomputed["recomputed_record_count"], 8)
            self.assertEqual(recomputed["sample_count"], 6)
            self.assertEqual(recomputed["boundary_count"], 2)
            self.assertEqual(recomputed["error_count"], 0)
            self.assertEqual(recomputed["foreign_pids"], [])

    def test_reintegrate_process_fails_closed_on_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "gpu_processes.jsonl"
            artifact = write_process_jsonl(path)
            artifact["sha256"] = "0" * 64
            with self.assertRaisesRegex(MOD.ExperimentError, "sha256 mismatch"):
                MOD.reintegrate_process_artifact(artifact, "p")

    def test_reintegrate_process_fails_closed_on_count_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "gpu_processes.jsonl"
            artifact = write_process_jsonl(path)
            artifact["record_count"] += 1
            with self.assertRaisesRegex(
                    MOD.ExperimentError, "record-count mismatch"):
                MOD.reintegrate_process_artifact(artifact, "p")

    def test_reintegrate_process_reports_foreign_pid_without_raising(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "gpu_processes.jsonl"
            artifact = write_process_jsonl(path, driver_pid=4242, foreign=True)
            recomputed = MOD.reintegrate_process_artifact(artifact, "p")
            self.assertFalse(recomputed["quality_valid"])
            self.assertEqual(recomputed["foreign_pids"], [9999])

    def test_reintegrate_process_rejects_tampered_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "gpu_processes.jsonl"
            artifact = write_process_jsonl(path)
            lines = path.read_text().split("\n")
            index = next(i for i, line in enumerate(lines)
                         if '"kind":"sample"' in line)
            lines[index] = lines[index].replace(
                '"kind":"sample"', '"kind":"sample","kind":"error"', 1)
            path.write_text("\n".join(lines), encoding="ascii")
            artifact["sha256"] = MOD.sha256_bytes(path.read_bytes())
            with self.assertRaisesRegex(
                    MOD.ExperimentError, "duplicate JSON key"):
                MOD.reintegrate_process_artifact(artifact, "p")

    def test_aggregate_consumes_recomputed_process_validity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            pairs = [aggregate_pair(index) for index in range(8)]
            for index, pair in enumerate(pairs):
                for offset, side in enumerate(("control", "treatment")):
                    power_path = temp / f"pw-{index}-{side}.jsonl"
                    power = write_power_jsonl(
                        power_path, count=140 + index * 2 + offset,
                        constant=False)
                    pair[f"{side}_power_artifact"] = power
                    recomputed = MOD.reintegrate_power_artifact(power, "p")
                    pair[f"{side}_energy_nj"] = recomputed["energy_nj"]
                    pair[f"{side}_uncertainty_nj"] = recomputed["uncertainty_nj"]
                    # pair 0's control process stream is contaminated
                    pair[f"{side}_process_artifact"] = write_process_jsonl(
                        temp / f"pr-{index}-{side}.jsonl",
                        driver_pid=4242 + index * 2 + offset,
                        foreign=(index == 0 and side == "control"))
                pair["treatment_thermal_artifact"] = write_thermal_jsonl(
                    temp / f"th-{index}.jsonl", boot=f"epoch{index}")
                pair["control_placement_artifact"] = control_placement(
                    temp / f"cp-{index}.jsonl", pid=2000 + index * 10)
                pair["control_placement_expectations"] = control_expectations(
                    pid=2000 + index * 10)
                pair["treatment_placement_artifact"] = treatment_placement(
                    temp / f"tp-{index}.jsonl", pid=3000 + index * 10)
                pair["treatment_placement_expectations"] = treatment_expectations(
                    pid=3000 + index * 10)
                pair.pop("reintegrated", None)
            verified = MOD.reverify_pairs(pairs)
            # the stored measurement_valid was True; reintegration of the bytes
            # flips pair 0 to invalid purely from the recomputed process evidence
            self.assertFalse(verified[0]["measurement_valid"])
            self.assertTrue(all(p["measurement_valid"] for p in verified[1:]))
            result = MOD.aggregate_result(verified, True, 100)
            self.assertEqual(result["measurement_status"], "INVALID_TIMELINE")
            self.assertEqual(result["label"], "MEASUREMENT_INVALID")
            self.assertEqual(result["formal_claim"], "NONE")

    def test_aggregate_rejects_reused_process_artifact(self):
        pairs = [aggregate_pair(index) for index in range(8)]
        pairs[1]["control_process_artifact"] = pairs[0][
            "control_process_artifact"]
        with self.assertRaisesRegex(
                MOD.ExperimentError, "process artifact is reused"):
            MOD.aggregate_result(pairs, True, 100)

    def test_aggregate_requires_process_artifact_under_measurement(self):
        pairs = [aggregate_pair(index) for index in range(8)]
        pairs[2]["treatment_process_artifact"] = None
        with self.assertRaisesRegex(
                MOD.ExperimentError, "complete process artifact"):
            MOD.aggregate_result(pairs, True, 100)

    # ---- Finding 3: power-limit evidence -----------------------------------

    def test_reintegrate_power_rejects_metadata_only_limit_forgery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "power.jsonl"
            artifact = write_power_jsonl(
                path, count=140, constant=False, power_limit_mw=300_000)
            # forge only the metadata declaration; the bytes (and their hash)
            # still carry the real 300 W limit, so the mismatch is caught
            artifact["power_limit_mw"] = 250_000
            recomputed = MOD.reintegrate_power_artifact(artifact, "p")
            self.assertFalse(recomputed["quality_valid"])
            self.assertTrue(any(
                r.startswith("E_POWER_LIMIT_MISMATCH")
                for r in recomputed["quality_reasons"]))
            self.assertIsNone(recomputed["energy_nj"])
            self.assertEqual(recomputed["observed_power_limit_mw"], 300_000)

    def test_reintegrate_power_uncertainty_uses_recomputed_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "power.jsonl"
            artifact = write_power_jsonl(
                path, count=140, constant=False, power_limit_mw=280_000)
            recomputed = MOD.reintegrate_power_artifact(artifact, "p")
            self.assertTrue(recomputed["quality_valid"])
            self.assertEqual(recomputed["observed_power_limit_mw"], 280_000)
            window = artifact["window_end_us"] - artifact["window_start_us"]
            self.assertEqual(
                recomputed["uncertainty_nj"],
                MOD.nvml_uncertainty(window, 280_000)["total"])

    # ---- CP1.3: on-device phone thermal logger -----------------------------

    def test_thermal_logger_script_runs_on_device_only(self):
        script = MOD._thermal_logger_script(
            "/data/local/tmp/ls-npu/t.jsonl",
            "/data/local/tmp/ls-npu/t.stop", "3C15AU002CL00000")
        self.assertIn('"kind":"header"', script)
        self.assertIn('"kind":"sample"', script)
        self.assertIn('"kind":"footer"', script)
        self.assertIn("/proc/sys/kernel/random/boot_id", script)
        self.assertIn("/proc/uptime", script)
        self.assertIn("thermal_zone", script)
        self.assertIn("dumpsys thermalservice", script)
        self.assertIn("sleep 0.5", script)
        # the loop runs entirely in the phone shell: it issues no host adb calls
        self.assertNotIn("adb", script)

    def test_reintegrate_thermal_recomputes_from_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "phone_thermal.jsonl"
            artifact = write_thermal_jsonl(path, count=8)
            recomputed = MOD.reintegrate_thermal_artifact(artifact, "t")
            self.assertTrue(recomputed["quality_valid"])
            self.assertEqual(recomputed["sample_count"], 8)
            self.assertEqual(recomputed["error_count"], 0)
            self.assertEqual(recomputed["boot_id"], _test_boot_uuid("epoch1000"))

    def test_reintegrate_thermal_fails_closed_on_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "phone_thermal.jsonl"
            artifact = write_thermal_jsonl(path)
            artifact["sha256"] = "0" * 64
            with self.assertRaisesRegex(MOD.ExperimentError, "sha256 mismatch"):
                MOD.reintegrate_thermal_artifact(artifact, "t")

    def test_reintegrate_thermal_reports_soft_invalids_without_raising(self):
        cases = {
            "throttled": (dict(status=5), "E_THERMAL_STATUS_NONZERO"),
            "sensorless": (dict(empty_sensors=True), "E_THERMAL_SENSORS_EMPTY"),
            "logger_error": (dict(error=True), "E_THERMAL_LOGGER_ERROR:1"),
            "gap": (dict(gap_at=2), "E_THERMAL_GAP"),
        }
        for name, (kwargs, reason_prefix) in cases.items():
            with self.subTest(case=name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = pathlib.Path(temp_dir) / "phone_thermal.jsonl"
                    artifact = write_thermal_jsonl(path, **kwargs)
                    recomputed = MOD.reintegrate_thermal_artifact(artifact, "t")
                    self.assertFalse(recomputed["quality_valid"])
                    self.assertTrue(any(
                        r.startswith(reason_prefix)
                        for r in recomputed["quality_reasons"]))

    def test_reintegrate_thermal_rejects_wrong_phone(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "phone_thermal.jsonl"
            artifact = write_thermal_jsonl(path, serial="OTHERPHONE")
            artifact["serial"] = "3C15AU002CL00000"
            recomputed = MOD.reintegrate_thermal_artifact(artifact, "t")
            self.assertFalse(recomputed["quality_valid"])
            self.assertTrue(any(
                r.startswith("E_THERMAL_SERIAL")
                for r in recomputed["quality_reasons"]))

    def test_evaluate_thermal_requires_window_bracket(self):
        boot = _test_boot_uuid("b")
        headers = [{"kind": "header", "boot_id": boot, "serial": "s"}]
        footers = [{"kind": "footer", "boot_id": boot, "serial": "s",
                    "timestamp_us": 4_000_000}]
        samples = [
            {"kind": "sample", "timestamp_us": (i + 1) * 500_000,
             "thermal_status": 0,
             "temperatures": [{"name": "z", "temp_millic": 40000}]}
            for i in range(6)
        ]
        unbracketed = MOD.evaluate_thermal_telemetry(
            headers, footers, samples, [], "s", 0, 10_000_000)
        self.assertFalse(unbracketed["valid"])
        self.assertIn("E_THERMAL_NO_LEADING_SAMPLE", unbracketed["reasons"])
        self.assertIn("E_THERMAL_NO_TRAILING_SAMPLE", unbracketed["reasons"])
        bracketed = MOD.evaluate_thermal_telemetry(
            headers, footers, samples, [], "s", 700_000, 2_900_000)
        self.assertTrue(bracketed["valid"])

    def test_evaluate_thermal_no_header_does_not_crash(self):
        # >= min_updates samples but zero headers must not IndexError on the main
        # return path (it must fail closed with boot_id None)
        samples = [
            {"kind": "sample", "timestamp_us": (i + 1) * 500_000,
             "thermal_status": 0,
             "temperatures": [{"name": "z", "temp_millic": 40000}]}
            for i in range(4)
        ]
        outcome = MOD.evaluate_thermal_telemetry(
            [], [{"kind": "footer", "boot_id": _test_boot_uuid("b"),
                  "serial": "s", "timestamp_us": 3_000_000}],
            samples, [], "s", 400_000, 2_100_000)
        self.assertFalse(outcome["valid"])
        self.assertIsNone(outcome["boot_id"])
        self.assertTrue(any(r.startswith("E_THERMAL_HEADER")
                            for r in outcome["reasons"]))

    def test_aggregate_enforces_treatment_thermal_binding(self):
        mutations = [
            ({"treatment_thermal_binding": "NOT_APPLICABLE"},
             "treatment thermal binding is not TREATMENT"),
            ({"control_thermal_binding": "TREATMENT"},
             "control thermal binding is not NOT_APPLICABLE"),
            ({"treatment_thermal_artifact": None},
             "complete thermal artifact"),
        ]
        for fields, regex in mutations:
            with self.subTest(fields=fields):
                pairs = [aggregate_pair(index) for index in range(8)]
                pairs[2].update(fields)
                with self.assertRaisesRegex(MOD.ExperimentError, regex):
                    MOD.aggregate_result(pairs, True, 100)

    def test_aggregate_rejects_reused_thermal_artifact(self):
        pairs = [aggregate_pair(index) for index in range(8)]
        pairs[1]["treatment_thermal_artifact"] = pairs[0][
            "treatment_thermal_artifact"]
        with self.assertRaisesRegex(
                MOD.ExperimentError, "thermal artifact is reused"):
            MOD.aggregate_result(pairs, True, 100)

    # ---- CP1.5: executed backend-placement certificate ---------------------

    def test_parse_placement_certs_strict(self):
        good = "PLACEMENTCERT " + MOD.json.dumps(placement_cert()) + "\n"
        certs = MOD.parse_placement_certs(good, "host")
        self.assertEqual(len(certs), 1)
        self.assertEqual(certs[0]["role"], "phone_stage")
        # a missing field must be rejected
        bad = placement_cert()
        del bad["status"]
        with self.assertRaises(MOD.ExperimentError):
            MOD.parse_placement_certs(
                "PLACEMENTCERT " + MOD.json.dumps(bad) + "\n", "host")

    def test_evaluate_placement_cert_gates(self):
        expected = treatment_expectations()[1]
        self.assertTrue(MOD.evaluate_placement_cert(
            placement_cert(), expected)["valid"])
        cases = {
            "fallback": dict(cpu_fallback_nodes=3, status="PLACEMENT_CPU_FALLBACK"),
            "zero_compute": dict(compute_nodes=0, status="PLACEMENT_NO_COMPUTE"),
            "cpu_backend": dict(observed_backends=["CPU"],
                                status="PLACEMENT_CPU_FALLBACK"),
            "multi_backend": dict(observed_backends=["HTP0", "CPU"],
                                  status="PLACEMENT_MULTI_BACKEND"),
            "bad_status": dict(status="PLACEMENT_MULTI_BACKEND"),
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                self.assertFalse(
                    MOD.evaluate_placement_cert(
                        placement_cert(**kwargs), expected)["valid"])

    def test_reintegrate_placement_recomputes_from_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "placement.jsonl"
            artifact = treatment_placement(path)
            recomputed = MOD.reintegrate_placement_artifact(
                artifact, treatment_expectations(), "p")
            self.assertTrue(recomputed["quality_valid"])
            self.assertEqual(
                recomputed["sources"], ["host", "phone:3C15AU002CL00000"])

    def test_reintegrate_placement_fails_closed_on_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "placement.jsonl"
            artifact = treatment_placement(path)
            artifact["sha256"] = "0" * 64
            with self.assertRaisesRegex(MOD.ExperimentError, "sha256 mismatch"):
                MOD.reintegrate_placement_artifact(
                    artifact, treatment_expectations(), "p")

    def test_reintegrate_placement_reports_missing_role(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "placement.jsonl"
            # only the host_tail cert present, but both roles expected
            artifact = write_placement_jsonl(path, [
                placement_cert(role="host_tail", observed_backends=["CUDA0"],
                               expected_backend="CUDA", route="pipedriver",
                               layer_start=2, layer_end=48)])
            recomputed = MOD.reintegrate_placement_artifact(
                artifact, treatment_expectations(), "p")
            self.assertFalse(recomputed["quality_valid"])
            self.assertIn(
                "E_PLACEMENT_MISSING:phone:3C15AU002CL00000",
                recomputed["quality_reasons"])

    def test_reintegrate_placement_reports_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "placement.jsonl"
            artifact = write_placement_jsonl(path, [
                placement_cert(role="phone_stage", cpu_fallback_nodes=4,
                               status="PLACEMENT_CPU_FALLBACK")])
            recomputed = MOD.reintegrate_placement_artifact(
                artifact, [treatment_expectations()[1]], "p")
            self.assertFalse(recomputed["quality_valid"])
            self.assertTrue(any(
                "E_PLACEMENT_" in r
                for r in recomputed["quality_reasons"]))

    def test_reintegrate_placement_reports_duplicate_and_unexpected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "placement.jsonl"
            artifact = write_placement_jsonl(path, [
                placement_cert(role="phone_stage", pid=1),
                placement_cert(role="phone_stage", pid=2)])
            recomputed = MOD.reintegrate_placement_artifact(
                artifact, [treatment_expectations()[1]], "p")
            self.assertFalse(recomputed["quality_valid"])
            self.assertIn(
                "E_PLACEMENT_DUPLICATE:phone:3C15AU002CL00000",
                recomputed["quality_reasons"])

    def test_reintegrate_placement_rejects_tampered_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "placement.jsonl"
            artifact = treatment_placement(path)
            lines = path.read_text().split("\n")
            lines[0] = lines[0].replace(
                '"compute_nodes":', '"compute_nodes":1,"compute_nodes":', 1)
            path.write_text("\n".join(lines), encoding="ascii")
            artifact["sha256"] = MOD.sha256_bytes(path.read_bytes())
            with self.assertRaisesRegex(
                    MOD.ExperimentError, "duplicate JSON key"):
                MOD.reintegrate_placement_artifact(
                    artifact, treatment_expectations(), "p")

    def test_aggregate_consumes_recomputed_placement_validity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            pairs = [aggregate_pair(index) for index in range(8)]
            for index, pair in enumerate(pairs):
                for offset, side in enumerate(("control", "treatment")):
                    power = write_power_jsonl(
                        temp / f"pw-{index}-{side}.jsonl",
                        count=140 + index * 2 + offset, constant=False)
                    pair[f"{side}_power_artifact"] = power
                    recomputed = MOD.reintegrate_power_artifact(power, "p")
                    pair[f"{side}_energy_nj"] = recomputed["energy_nj"]
                    pair[f"{side}_uncertainty_nj"] = recomputed["uncertainty_nj"]
                    pair[f"{side}_process_artifact"] = write_process_jsonl(
                        temp / f"pr-{index}-{side}.jsonl",
                        driver_pid=4242 + index * 2 + offset)
                pair["treatment_thermal_artifact"] = write_thermal_jsonl(
                    temp / f"th-{index}.jsonl", boot=f"epoch{index}")
                pair["control_placement_artifact"] = control_placement(
                    temp / f"cp-{index}.jsonl", pid=2000 + index * 10)
                pair["control_placement_expectations"] = control_expectations(
                    pid=2000 + index * 10)
                # pair 0's phone stage certificate shows a CPU fallback
                if index == 0:
                    pair["treatment_placement_artifact"] = write_placement_jsonl(
                        temp / f"tp-{index}.jsonl", [
                            placement_cert(role="host_tail", pid=10,
                                           observed_backends=["CUDA0"],
                                           expected_backend="CUDA"),
                            placement_cert(role="phone_stage", pid=11,
                                           cpu_fallback_nodes=5,
                                           status="PLACEMENT_CPU_FALLBACK")])
                    pair["treatment_placement_expectations"] = [{
                        **treatment_expectations(pid=10)[0], "pid": 10,
                    }, {
                        **treatment_expectations(pid=10)[1], "pid": None,
                    }]
                else:
                    pair["treatment_placement_artifact"] = treatment_placement(
                        temp / f"tp-{index}.jsonl", pid=3000 + index * 10)
                    pair["treatment_placement_expectations"] = treatment_expectations(
                        pid=3000 + index * 10)
                pair.pop("reintegrated", None)
            verified = MOD.reverify_pairs(pairs)
            self.assertFalse(verified[0]["measurement_valid"])
            self.assertTrue(all(p["measurement_valid"] for p in verified[1:]))
            result = MOD.aggregate_result(verified, True, 100)
            self.assertEqual(result["measurement_status"], "INVALID_TIMELINE")
            self.assertEqual(result["label"], "MEASUREMENT_INVALID")

    def test_aggregate_requires_placement_artifact_and_rejects_reuse(self):
        pairs = [aggregate_pair(index) for index in range(8)]
        pairs[2]["treatment_placement_artifact"] = None
        with self.assertRaisesRegex(
                MOD.ExperimentError, "complete placement artifact"):
            MOD.aggregate_result(pairs, True, 100)

        pairs = [aggregate_pair(index) for index in range(8)]
        pairs[1]["control_placement_artifact"] = pairs[0][
            "control_placement_artifact"]
        with self.assertRaisesRegex(
                MOD.ExperimentError, "placement artifact is reused"):
            MOD.aggregate_result(pairs, True, 100)

    # ---- Review findings [0]/[4]: byte-recomputed validity is load-bearing ----

    def test_aggregate_consumes_recomputed_thermal_validity(self):
        # mirror of the process/placement consumption tests for CP1.3 thermal
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            pairs = build_measured_pairs(temp)
            # pair 0's treatment thermal recomputes throttled (status != 0)
            pairs[0]["treatment_thermal_artifact"] = write_thermal_jsonl(
                temp / "th-throttled.jsonl", boot="epochbad", status=5)
            verified = MOD.reverify_pairs(pairs)
            self.assertFalse(verified[0]["measurement_valid"])
            self.assertTrue(all(p["measurement_valid"] for p in verified[1:]))
            result = MOD.aggregate_result(verified, True, 100)
            self.assertEqual(result["measurement_status"], "INVALID_TIMELINE")
            self.assertEqual(result["label"], "MEASUREMENT_INVALID")

    def test_end_to_end_relief_label_and_each_evidence_flip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            base = MOD.reverify_pairs(build_measured_pairs(temp / "base"))
            agg = MOD.aggregate_result(base, True, 100)
            # all four evidence types recomputed valid for all 8 pairs -> the
            # aggregate reaches the exploratory label stage from real bytes
            self.assertEqual(agg["measurement_status"], "EXPLORATORY_COMPLETE")
            self.assertTrue(agg["label"].startswith("GPU_BOARD_DIAGNOSTIC"))
            self.assertEqual(agg["formal_claim"], "NONE")
            self.assertEqual(agg["phone_energy_status"], "UNKNOWN")
            self.assertEqual(agg["total_system_energy_status"], "UNKNOWN")

            def flip_power(pair, sub):
                pair["control_power_artifact"] = write_power_jsonl(
                    sub / "flip-pw.jsonl", count=200, constant=True)

            def flip_process(pair, sub):
                pair["treatment_process_artifact"] = write_process_jsonl(
                    sub / "flip-pr.jsonl", driver_pid=4242, foreign=True)

            def flip_thermal(pair, sub):
                pair["treatment_thermal_artifact"] = write_thermal_jsonl(
                    sub / "flip-th.jsonl", status=5)

            def flip_placement(pair, sub):
                pair["treatment_placement_artifact"] = write_placement_jsonl(
                    sub / "flip-pl.jsonl", [
                        placement_cert(role="host_tail",
                                       observed_backends=["CUDA0"],
                                       expected_backend="CUDA"),
                        placement_cert(role="phone_stage", cpu_fallback_nodes=5,
                                       status="PLACEMENT_CPU_FALLBACK")])

            for name, flip in (("power", flip_power), ("process", flip_process),
                               ("thermal", flip_thermal),
                               ("placement", flip_placement)):
                with self.subTest(evidence=name):
                    sub = temp / name
                    pairs = build_measured_pairs(sub)
                    flip(pairs[0], sub)
                    verified = MOD.reverify_pairs(pairs)
                    self.assertFalse(verified[0]["measurement_valid"])
                    result = MOD.aggregate_result(verified, True, 100)
                    self.assertEqual(
                        result["measurement_status"], "INVALID_TIMELINE", name)
                    self.assertEqual(result["label"], "MEASUREMENT_INVALID", name)

    def test_reverify_does_not_trust_stored_validity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            pairs = build_measured_pairs(temp)
            pairs[0]["measurement_valid"] = False
            verified = MOD.reverify_pairs(pairs)
            self.assertTrue(verified[0]["measurement_valid"])
            self.assertTrue(verified[0]["stored_validity_disagrees"])

    # ---- Review finding [2]: exit-code contract through main() ---------------

    def _measure_env(self):
        MOD.connected_serials = lambda: {"3C15AU002CL00000"}
        MOD.query_gpu = lambda _uuid: {
            "uuid": MOD.S11_E0_GPU_UUID, "pstate": "P8", "power_mw": 10,
            "power_limit_mw": 300000, "utilization_pct": 0, "memory_used_mib": 0}
        MOD.remote_file_info = lambda _serial, path, include_hash: {
            "path": path, "bytes": 1,
            "sha256": "a" * 64 if include_hash else None}

    def test_main_measure_valid_timeline_returns_exit_0(self):
        originals = (MOD.connected_serials, MOD.query_gpu,
                     MOD.remote_file_info, MOD.run_route)
        counter = [0]
        try:
            self._measure_env()

            def run_route(_args, route, run_dir):
                counter[0] += 1
                is_treatment = route != MOD.ROUTE_CONTROL
                run_dir = pathlib.Path(run_dir)
                power = write_power_jsonl(
                    run_dir / "power.jsonl", count=140 + counter[0],
                    constant=False)
                prec = MOD.reintegrate_power_artifact(power, "p")
                process = write_process_jsonl(
                    run_dir / "gpu_processes.jsonl", driver_pid=4242 + counter[0])
                thermal = write_thermal_jsonl(
                    run_dir / "phone_thermal.jsonl",
                    boot=f"epoch{counter[0]}") if is_treatment else None
                placement = (treatment_placement(
                    run_dir / "placement_certs.jsonl", pid=1000 + counter[0] * 10)
                    if is_treatment else control_placement(
                        run_dir / "placement_certs.jsonl",
                        pid=5000 + counter[0] * 10))
                placement_expected = (
                    treatment_expectations(pid=1000 + counter[0] * 10)
                    if is_treatment else
                    control_expectations(pid=5000 + counter[0] * 10))
                return {
                    "schema": MOD.RUN_SCHEMA,
                    "route_record_schema": MOD.ROUTE_RECORD_SCHEMA,
                    "route": route, "records": [record(route, [1])],
                    "token_sha256": "b" * 64,
                    "gpu_ready": {"memory_used_mib": 100 if not is_treatment else 90},
                    "batch_metrics": {"median_useful_requests_per_s": 10.0,
                                      "p95_group_wall_us": 10},
                    "measurement": {
                        "quality": {"valid": True, "reasons": []},
                        "energy_nj": prec["energy_nj"],
                        "uncertainty_nj": prec["uncertainty_nj"],
                        "power_artifact": power,
                        "process_monitor_artifact": process,
                        "phone_thermal_artifact": thermal,
                        "phone_thermal_binding": (
                            "TREATMENT" if is_treatment else "NOT_APPLICABLE"),
                        "placement_artifact": placement,
                        "placement_expectations": placement_expected,
                    },
                }

            MOD.run_route = run_route
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = pathlib.Path(temp_dir)
                host_bin = temp / "host-bin"; host_bin.write_bytes(b"bin")
                host_model = temp / "model.gguf"; host_model.write_bytes(b"model")
                output = temp / "output"
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = MOD.main(measure_argv({
                        "--host-bin": str(host_bin),
                        "--host-model": str(host_model),
                        "--output": str(output),
                    }))
                self.assertEqual(rc, 0)
                summary = MOD.json.loads((output / "summary.json").read_text())
                agg = summary["aggregate"]
                self.assertEqual(agg["measurement_status"], "EXPLORATORY_COMPLETE")
                self.assertTrue(agg["label"].startswith("GPU_BOARD_DIAGNOSTIC"))
                self.assertEqual(agg["formal_claim"], "NONE")
        finally:
            (MOD.connected_serials, MOD.query_gpu,
             MOD.remote_file_info, MOD.run_route) = originals

    def test_main_exit_2_on_token_divergence(self):
        originals = (MOD.connected_serials, MOD.query_gpu,
                     MOD.remote_file_info, MOD.run_route)
        try:
            self._measure_env()

            def run_route(_args, route, _run_dir):
                tokens = [1] if route == MOD.ROUTE_CONTROL else [2]
                return {
                    "schema": MOD.RUN_SCHEMA,
                    "route_record_schema": MOD.ROUTE_RECORD_SCHEMA,
                    "route": route, "records": [record(route, tokens)],
                    "token_sha256": ("c" if route == MOD.ROUTE_CONTROL else "d") * 64,
                    "gpu_ready": {"memory_used_mib": 100},
                    "batch_metrics": {"median_useful_requests_per_s": 10.0,
                                      "p95_group_wall_us": 10},
                    "measurement": {
                        "quality": {"valid": False,
                                    "reasons": ["MEASUREMENT_NOT_REQUESTED"]},
                        "energy_nj": None, "uncertainty_nj": None},
                }

            MOD.run_route = run_route
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = pathlib.Path(temp_dir)
                host_bin = temp / "host-bin"; host_bin.write_bytes(b"bin")
                host_model = temp / "model.gguf"; host_model.write_bytes(b"model")
                output = temp / "output"
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(
                            MOD.ExperimentError, "exact-output mismatch"):
                        MOD.main([
                            "--host-bin", str(host_bin),
                            "--host-model", str(host_model),
                            "--gpu-uuid", "GPU-test", "--output", str(output),
                            "--route", "op15"])
                failure = MOD.json.loads((output / "failure.json").read_text())
                self.assertEqual(failure["schema"], MOD.FAILURE_SCHEMA)
                self.assertNotEqual(
                    failure["control_token_sha256"],
                    failure["treatment_token_sha256"])
        finally:
            (MOD.connected_serials, MOD.query_gpu,
             MOD.remote_file_info, MOD.run_route) = originals

    # ---- Review finding [5]: strict per-record schema for thermal/process ----

    def test_thermal_record_schema_rejections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "th.jsonl"
            artifact = write_thermal_jsonl(path)
            lines = path.read_text().split("\n")
            idx = next(i for i, l in enumerate(lines)
                       if l.startswith('{"kind":"sample"'))
            lines[idx] = lines[idx].replace(
                '"thermal_status":0', '"thermal_status":0,"thermal_status":5', 1)
            path.write_text("\n".join(lines), encoding="ascii")
            artifact["sha256"] = MOD.sha256_bytes(path.read_bytes())
            with self.assertRaisesRegex(MOD.ExperimentError, "duplicate JSON key"):
                MOD.reintegrate_thermal_artifact(artifact, "t")
        rejects = [
            {"kind": "sample", "timestamp_us": 0, "thermal_status": 0.0,
             "temperatures": []},                       # float status
            {"kind": "sample", "timestamp_us": 0, "thermal_status": 0,
             "temperatures": [], "extra": 1},           # extra field
            {"kind": "sample", "timestamp_us": 0, "thermal_status": 0},  # missing
            {"kind": "sample", "timestamp_us": 0, "thermal_status": 0,
             "temperatures": [{"name": "z", "temp_millic": 1.0}]},  # float temp
            {"kind": "sample", "timestamp_us": 0, "thermal_status": 0,
             "temperatures": [{"name": "z"}]},          # missing temp field
            {"kind": "bogus"},                          # unknown kind
        ]
        for record_obj in rejects:
            with self.subTest(record=record_obj):
                with self.assertRaises(MOD.ExperimentError):
                    MOD._validate_thermal_record(record_obj, "t")

    def test_process_record_schema_rejections(self):
        base = {"kind": "sample", "board_uuid": "GPU-test",
                "probe_start_us": 0, "probe_end_us": 1,
                "timestamp_us": 1,
                "processes": [{"pid": 1, "name": "x", "memory_used_mib": 1}]}
        rejects = [
            {**base, "processes": [{"pid": 1, "name": "x",
                                    "memory_used_mib": 1.0}]},  # float mem
            {**base, "processes": [{"pid": 1, "name": "x"}]},   # missing entry key
            {**base, "extra": 1},                               # extra field
            {"kind": "sample", "board_uuid": "GPU-test",
             "probe_start_us": 0, "probe_end_us": 1,
             "timestamp_us": 1},                                # missing processes
            {"kind": "error", "board_uuid": "GPU-test",
             "probe_start_us": 0, "probe_end_us": 1,
             "message": ""},                                   # empty message
            {"kind": "bogus"},                                  # unknown kind
        ]
        for record_obj in rejects:
            with self.subTest(record=record_obj):
                with self.assertRaises(MOD.ExperimentError):
                    MOD._validate_process_record(record_obj, "p")
        # end-to-end duplicate-key tamper reaching reintegrate
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "pr.jsonl"
            artifact = write_process_jsonl(path)
            lines = path.read_text().split("\n")
            lines[0] = lines[0].replace(
                '"timestamp_us":', '"timestamp_us":0,"timestamp_us":', 1)
            path.write_text("\n".join(lines), encoding="ascii")
            artifact["sha256"] = MOD.sha256_bytes(path.read_bytes())
            with self.assertRaisesRegex(MOD.ExperimentError, "duplicate JSON key"):
                MOD.reintegrate_process_artifact(artifact, "p")


class ReviewRegressionTests(unittest.TestCase):
    def test_placement_rejects_backend_different_from_expectation(self):
        outcome = MOD.evaluate_placement_cert(placement_cert(
            expected_backend="HTP0", observed_backends=["GPUOpenCL"]),
            treatment_expectations()[1])
        self.assertFalse(outcome["valid"])

    def test_thermal_rejects_boot_metadata_relabel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "thermal.jsonl"
            artifact = write_thermal_jsonl(path, boot="epoch1000")
            lines = path.read_text().splitlines()
            footer = MOD.strict_json_loads(lines[-1], "footer")
            footer["boot_id"] = _test_boot_uuid("epoch2000")
            lines[-1] = MOD.json.dumps(
                footer, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n", encoding="ascii")
            artifact["sha256"] = MOD.sha256_file(path)
            recomputed = MOD.reintegrate_thermal_artifact(artifact, "t")
            self.assertFalse(recomputed["quality_valid"])

    def test_thermal_rejects_source_order_inversion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "thermal.jsonl"
            artifact = write_thermal_jsonl(path)
            lines = [line for line in path.read_text().splitlines() if line]
            lines[1], lines[2] = lines[2], lines[1]
            path.write_text("\n".join(lines) + "\n", encoding="ascii")
            artifact["sha256"] = MOD.sha256_file(path)
            recomputed = MOD.reintegrate_thermal_artifact(artifact, "t")
            self.assertFalse(recomputed["quality_valid"])

    def test_process_record_rejects_impossible_chronology(self):
        record_obj = {
            "kind": "sample", "board_uuid": "GPU-test",
            "probe_start_us": 20, "probe_end_us": 10,
            "timestamp_us": 9,
            "processes": [{"pid": 1, "name": "x", "memory_used_mib": 1}],
        }
        with self.assertRaises(MOD.ExperimentError):
            MOD._validate_process_record(record_obj, "p")

    def test_power_artifact_rejects_board_uuid_relabel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = write_power_jsonl(
                pathlib.Path(temp_dir) / "power.jsonl", constant=False)
            artifact["board_uuid"] = "GPU-relabeled"
            with self.assertRaises(MOD.ExperimentError):
                MOD.reintegrate_power_artifact(artifact, "p")

    def test_process_artifact_rejects_board_uuid_relabel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = write_process_jsonl(
                pathlib.Path(temp_dir) / "process.jsonl")
            artifact["board_uuid"] = "GPU-relabeled"
            with self.assertRaises(MOD.ExperimentError):
                MOD.reintegrate_process_artifact(artifact, "p")

    def test_power_sampler_error_is_load_bearing_from_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "power.jsonl"
            artifact = write_power_jsonl(path, constant=False)
            lines = path.read_text().splitlines()
            error = {
                "kind": "error", "board_uuid": "GPU-test",
                "timestamp_us": 50_000, "message": "malformed NVML row"}
            lines.insert(1, MOD.json.dumps(
                error, sort_keys=True, separators=(",", ":")))
            path.write_text("\n".join(lines) + "\n", encoding="ascii")
            artifact["record_count"] += 1
            artifact["sha256"] = MOD.sha256_file(path)
            recomputed = MOD.reintegrate_power_artifact(artifact, "p")
            self.assertFalse(recomputed["quality_valid"])
            self.assertIn("E_SAMPLER:1", recomputed["quality_reasons"])

    def test_process_done_boundary_foreign_pid_is_load_bearing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = pathlib.Path(temp_dir) / "process.jsonl"
            artifact = write_process_jsonl(path)
            lines = path.read_text().splitlines()
            done = MOD.strict_json_loads(lines[-1], "done")
            done["processes"].append({
                "pid": 9999, "name": "foreign", "memory_used_mib": 1})
            lines[-1] = MOD.json.dumps(
                done, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n", encoding="ascii")
            artifact["sha256"] = MOD.sha256_file(path)
            recomputed = MOD.reintegrate_process_artifact(artifact, "p")
            self.assertFalse(recomputed["quality_valid"])
            self.assertEqual(recomputed["foreign_pids"], [9999])

    def test_placement_descriptor_fields_are_independent(self):
        cert = placement_cert()
        base = treatment_expectations()[1]
        mutations = {
            "mode": {**base, "mode": "pipedriver"},
            "range": {**base, "layer_end": 3},
            "layers": {**base, "n_layer": 49},
            "buffer": {**base, "default_compute_buffer_type": "GPUOpenCL"},
        }
        for name, descriptor in mutations.items():
            with self.subTest(name=name):
                self.assertFalse(MOD.evaluate_placement_cert(
                    cert, descriptor)["valid"])
        failed = placement_cert(run_rc=2, status="PLACEMENT_OK")
        self.assertFalse(MOD.evaluate_placement_cert(failed, base)["valid"])

    def test_readiness_reintegrates_without_energy_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            pair = aggregate_pair(0)
            pair.update({
                "control_placement_artifact": control_placement(
                    temp / "control-placement.jsonl", pid=2000),
                "control_placement_expectations": control_expectations(2000),
                "treatment_placement_artifact": treatment_placement(
                    temp / "treatment-placement.jsonl", pid=3000),
                "treatment_placement_expectations": treatment_expectations(3000),
                "treatment_thermal_artifact": write_thermal_jsonl(
                    temp / "thermal.jsonl"),
                "control_readiness_valid": True,
                "treatment_readiness_valid": True,
                "control_driver_pid": 2000,
                "treatment_driver_pid": 3000,
                "control_gpu_processes_ready": [],
                "control_gpu_processes_done": [],
                "treatment_gpu_processes_ready": [],
                "treatment_gpu_processes_done": [],
            })
            verified = MOD.reverify_readiness_pairs([pair])
            result = MOD.aggregate_readiness_result(verified)
            self.assertEqual(result["readiness_status"], "PASS")
            self.assertEqual(result["energy_status"], "NOT_MEASURED")
            self.assertEqual(result["formal_claim"], "NONE")

    def test_readiness_cli_is_distinct_from_measurement(self):
        argv = measure_argv({"--measure": None, "--slo-p95-us": None,
                             "--readiness": True, "--pairs": "1"})
        args = MOD.parse_args(argv)
        self.assertTrue(args.readiness)
        self.assertFalse(args.measure)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                MOD.parse_args(measure_argv({"--readiness": True}))
            with self.assertRaises(SystemExit):
                MOD.parse_args(measure_argv({"--phone-decode-no-fa": None}))

    def test_wait_for_gpu_idle_requires_consecutive_samples(self):
        samples = [
            {"utilization_pct": 0, "memory_used_mib": 10},
            {"utilization_pct": 0, "memory_used_mib": 10},
            {"utilization_pct": 90, "memory_used_mib": 10},
            {"utilization_pct": 0, "memory_used_mib": 10},
            {"utilization_pct": 0, "memory_used_mib": 10},
            {"utilization_pct": 0, "memory_used_mib": 10},
        ]
        calls = []
        original = MOD.query_gpu
        try:
            def fake_query(_uuid):
                calls.append(_uuid)
                return samples[len(calls) - 1]

            MOD.query_gpu = fake_query
            result = MOD.wait_for_gpu_idle(
                "GPU-test", 5, 20, timeout_s=1, stable_samples=3,
                interval_s=0)
        finally:
            MOD.query_gpu = original
        self.assertEqual(result, samples[-1])
        self.assertEqual(len(calls), 6)


if __name__ == "__main__":
    unittest.main()
