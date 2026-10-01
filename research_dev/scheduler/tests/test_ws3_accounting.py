#!/usr/bin/env python3
"""WS3 accounting tools: fleet energy (desktop + both phones), latency report (one percentile method), energy at
matched latency, and the desktop-side two-phone meter script (driven against a fake adb)."""

from __future__ import annotations

import fcntl
import gzip
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

from research_dev.scheduler.campaigns.burstgpt.tools import energy_latency, fleet_energy, latency_report

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "research_dev/scheduler/campaigns/burstgpt/tools/rig"
OP15, PIXEL = "3C15AU002CL00000", "5A040DLCH004ES"
START_S, END_S = 1000.0, 1100.0  # paid window on the host CLOCK_MONOTONIC


def _request(index, arrival, start, first, end, tokens, **extra):
    row = {"request_id": "trace:%03d" % index, "model_id": ("qwen3-14b", "gemma-4-12b", "llama-3.2-1b")[index % 3],
           "combined_request_index": index, "input_tokens": 10, "output_tokens": tokens,
           "replay_arrival_us": int(arrival * 1e6), "trace_arrival_us": int(arrival * 1e6),
           "first_token_ns": None if first is None else int(START_S * 1e9 + first * 1e9),
           "completion": {"actual_end_us": int(end * 1e6), "execution_receipt": {"started_us": int(start * 1e6),
                                                                                "executor_id": "physical:hot:desktop"}},
           "terminal_ticket": {"transition_receipts": []}, "attempt_ticket_ids": ["a"], "recoveries": []}
    row.update(extra)
    return row


def _result(cpu_j=5000.0, gpu_j=3000.0, requests=None):
    return {"status": "PASS", "duration_us": int((END_S - START_S) * 1e6),
            "paid_start_ns": int(START_S * 1e9), "paid_end_ns": int(END_S * 1e9),
            "trace_energy": {"fleet_energy_uj_by_domain": {"cpu-package": int(cpu_j * 1e6), "gpu-board": int(gpu_j * 1e6),
                                                           "phone-system": 450_000_000,
                                                           "pixel10pro-phone-system": 87_500_000},
                             "measurement_evidence_ids": ["ASSUMED_4P5W", "physical:nvml-board-power",
                                                          "physical:rapl-package-0"],
                             "estimation_metadata": {"phone_estimation_version": "assumed-phone-power-v1"}},
            "request_results": requests if requests is not None else [_request(0, 1, 2, 5, 25, 11)]}


def _anchor(serial, name):
    # phone uptime u <-> host monotonic 990 + (u - 10): the paid window is uptime [20, 120]
    return {"serial": serial, "local": "/nonexistent/" + name, "anchor_host_epoch_s": 5990.0,
            "anchor_host_monotonic_s": 990.0, "anchor_phone_uptime_s": 10.0, "anchor_uncertainty_s": 0.01,
            "end_host_epoch_s": 6120.0, "end_phone_uptime_s": 140.0, "samples": 131, "stopped": True}


def _op15_log(usb_ua=500_000, battery_ma=360, discharging=True):
    """2.5 W USB; counter -2,000 uAh every 10 s at 4 V (2.88 W) and current_now at half scale like the OP15."""
    lines = ["# start uptime 10.0 pid 1"]
    for u in range(10, 141):
        steps = (u - 10) // 10
        counter = 5_000_000 - 2000 * steps if discharging else 5_000_000
        lines.append("%d.00 battery/current_now=%d battery/voltage_now=4000000 battery/charge_counter=%d "
                     "usb/current_now=%d usb/voltage_now=5000000 usb/input_current_limit=500000"
                     % (u, battery_ma if discharging else 0, counter, usb_ua))
    lines.append("# stop uptime 140.0 samples 131")
    return "\n".join(lines) + "\n"


def _pixel_log(charging=True):
    """Charging: 5 W USB, 4 W into the battery (+1,000,000 uA at 4 V; counter +2,500 uAh every 9 s)."""
    lines = []
    for u in range(10, 141):
        if charging:
            counter, current, usb = 3_000_000 + 2500 * ((u - 10) // 9), 1_000_000, 1_000_000
        else:
            counter, current, usb = 3_000_000, 0, 100_000
        lines.append("%d.00 battery/current_now=%d battery/voltage_now=4000000 battery/charge_counter=%d "
                     "usb/current_now=%d usb/voltage_now=5000000 usb/input_current_limit=900000"
                     % (u, current, counter, usb))
    return "\n".join(lines) + "\n"


def _samples(gpu_w=30.0, cpu_w=50.0):
    maximum = 262_143_328_850
    rows = []
    energy = maximum - 1_000_000_000  # wraps inside the window
    t = 995.0
    while t <= 1105.0:
        rows.append({"t_ns": int(t * 1e9), "gpu": {"sample_t_ns": int(t * 1e9), "power_mw": int(gpu_w * 1000)},
                     "rapl_package": {"sample_t_ns": int(t * 1e9), "energy_uj": int(energy % maximum),
                                      "max_energy_range_uj": maximum}})
        energy += int(cpu_w * 0.5 * 1e6)
        t += 0.5
    return rows


def write_run(directory: Path, *, op15=None, pixel=None, result=None, samples=True, streams=None, meter_name=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "RESULT.json").write_text(json.dumps(result or _result()))
    meter = {}
    if op15 is not None:
        (directory / "x-run-POWER-op15.txt").write_text(op15)
        meter["op15"] = _anchor(OP15, "x-run-POWER-op15.txt")
    if pixel is not None:
        (directory / "x-run-POWER-pixel.txt").write_text(pixel)
        meter["pixel"] = _anchor(PIXEL, "x-run-POWER-pixel.txt")
    if meter:
        (directory / (meter_name or "x-run-POWER.json")).write_text(json.dumps(meter))
    if samples:
        with gzip.open(directory / "resource-samples.jsonl.gz", "wt") as stream:
            for row in _samples():
                stream.write(json.dumps(row) + "\n")
    for index, timings in (streams or {}).items():
        (directory / "streams").mkdir(exist_ok=True)
        (directory / "streams" / ("request-%03d.raw" % index)).write_text(
            'data: {"content":"x"}\n\ndata: {"stop":true,"timings":%s}\n\n' % json.dumps(timings))
    return directory


class FleetEnergyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_charging_disabled_op15_is_usb_plus_counter_discharge(self):
        run = write_run(self.root / "pe", op15=_op15_log(), pixel=_pixel_log(charging=False))
        report = fleet_energy.account_run("pe", run)
        op15 = report["phones"]["op15"]
        self.assertAlmostEqual(op15["usb_in_j"], 250.0, places=6)            # 2.5 W x 100 s, window-clipped
        self.assertAlmostEqual(op15["battery_net_discharge_j"], 288.0, places=6)  # 10 steps x 2 mAh x 4 V
        self.assertEqual(op15["battery_method"], "coulomb_counter")
        self.assertAlmostEqual(op15["system_j"], 538.0, places=6)
        self.assertAlmostEqual(op15["battery_current_to_counter_ratio"], 0.5, places=2)
        self.assertNotIn("BATTERY_CROSS_CHECK_RATIO", op15["flags"])  # OP15 current reads half scale by profile
        self.assertNotIn("CHARGING_IN_WINDOW", op15["flags"])
        self.assertEqual(op15["usb_at_limit_share"], 1.0)
        self.assertTrue(op15["coverage"]["complete"])
        self.assertEqual(report["phones_evidence"], fleet_energy.MEASURED)

    def test_charging_enabled_pixel_nets_out_stored_energy_without_double_counting(self):
        run = write_run(self.root / "chg", op15=_op15_log(), pixel=_pixel_log(charging=True))
        pixel = fleet_energy.account_run("chg", run, charge_efficiency=0.8)["phones"]["pixel"]
        self.assertAlmostEqual(pixel["usb_in_j"], 500.0, places=6)
        # counter +11 steps of 2.5 mAh at 4 V inside [uptime 20, 120] = 396 J stored
        self.assertAlmostEqual(pixel["battery_net_discharge_j"], -396.0, places=6)
        self.assertAlmostEqual(pixel["system_j"], 104.0, places=6)          # USB - stored, never USB + |battery|
        self.assertIn("CHARGING_IN_WINDOW", pixel["flags"])
        self.assertAlmostEqual(pixel["battery_current_cross_check_j"], -400.0, places=6)
        self.assertAlmostEqual(pixel["port_equivalent_j"], 500.0 - 396.0 / 0.8, places=6)
        self.assertEqual(pixel["port_equivalent_evidence"], fleet_energy.MODELED)

    def test_cross_check_flags_ratio_and_sign(self):
        wrong_scale = fleet_energy.account_run("r", write_run(self.root / "r", op15=_op15_log(battery_ma=720)))
        self.assertIn("BATTERY_CROSS_CHECK_RATIO", wrong_scale["phones"]["op15"]["flags"])
        wrong_sign = fleet_energy.account_run("s", write_run(self.root / "s", op15=_op15_log(battery_ma=-360)))
        self.assertIn("BATTERY_SIGN_MISMATCH", wrong_sign["phones"]["op15"]["flags"])

    def test_zero_battery_current_falls_back_to_counter_and_is_flagged(self):
        text = _op15_log().replace("battery/current_now=360", "battery/current_now=0")
        op15 = fleet_energy.account_run("z", write_run(self.root / "z", op15=text))["phones"]["op15"]
        self.assertIn("BATTERY_CURRENT_READS_ZERO", op15["flags"])
        self.assertAlmostEqual(op15["battery_net_discharge_j"], 288.0, places=6)

    def test_window_not_covered_is_flagged(self):
        short = "\n".join(line for line in _op15_log().splitlines() if not line[:1].isdigit() or float(line.split()[0]) < 90)
        op15 = fleet_energy.account_run("w", write_run(self.root / "w", op15=short + "\n"))["phones"]["op15"]
        self.assertIn("WINDOW_NOT_COVERED", op15["flags"])
        self.assertFalse(op15["coverage"]["complete"])

    def test_counter_reset_and_missing_clock_anchor_are_flagged_not_trusted(self):
        text = _op15_log().replace("battery/charge_counter=4990000", "battery/charge_counter=4000000", 1)
        op15 = fleet_energy.account_run("j", write_run(self.root / "j", op15=text))["phones"]["op15"]
        self.assertIn("COUNTER_JUMP", op15["flags"])
        run = write_run(self.root / "a", op15=_op15_log())
        meter = json.loads((run / "x-run-POWER.json").read_text())
        del meter["op15"]["anchor_host_monotonic_s"]
        (run / "x-run-POWER.json").write_text(json.dumps(meter))
        report = fleet_energy.account_run("a", run)
        self.assertEqual(report["phones"]["op15"]["flags"], ["CLOCK_ANCHOR_INVALID"])
        self.assertIsNone(report["fleet_j"])
        self.assertEqual(report["phones_evidence"], fleet_energy.ABSENT)

    def test_desktop_domains_cross_check_and_modeled_phone(self):
        report = fleet_energy.account_run("d", write_run(self.root / "d", op15=_op15_log(), pixel=_pixel_log()))
        desktop = report["desktop"]
        self.assertAlmostEqual(desktop["host_j"], 8000.0)
        self.assertEqual(desktop["evidence"], fleet_energy.MEASURED)
        self.assertAlmostEqual(desktop["cross_check"]["gpu_board_j"], 3000.0, places=6)
        self.assertAlmostEqual(desktop["cross_check"]["cpu_package_j"], 5000.0, places=3)  # across a RAPL wrap
        self.assertAlmostEqual(desktop["cross_check"]["host_delta_j"], 0.0, places=3)
        self.assertEqual(report["modeled_phone"], {"op15_j": 450.0, "pixel_j": 87.5, "evidence": fleet_energy.MODELED,
                                                   "model": "assumed-phone-power-v1"})
        self.assertAlmostEqual(report["fleet_j"], 8000.0 + 538.0 + 104.0, places=6)
        self.assertIsNone(report["fleet_wall_j"])
        self.assertEqual(desktop["wall_evidence"], fleet_energy.ABSENT)

    def test_wall_meter_fleet_adds_only_the_battery_terms(self):
        run = write_run(self.root / "wall", op15=_op15_log(), pixel=_pixel_log())
        wall = self.root / "wall.csv"
        wall.write_text("host_monotonic_s,watts\n" + "".join("%d,120\n" % t for t in range(990, 1111)))
        report = fleet_energy.account_run("wall", run, wall_path=wall)
        self.assertAlmostEqual(report["desktop"]["wall_j"], 12000.0, places=6)
        self.assertAlmostEqual(report["fleet_wall_j"], 12000.0 + 288.0 - 396.0, places=6)

    def test_meter_json_elsewhere_wins(self):
        run = write_run(self.root / "m", op15=_op15_log(), pixel=_pixel_log())
        other = self.root / "meter"
        other.mkdir()
        (other / "x-run-POWER-op15.txt").write_text(_op15_log(usb_ua=100_000))
        (other / "METER.json").write_text(json.dumps({"op15": _anchor(OP15, "x-run-POWER-op15.txt")}))
        report = fleet_energy.account_run("m", run, meter_path=other / "METER.json")
        self.assertEqual(sorted(report["phones"]), ["op15"])
        self.assertAlmostEqual(report["phones"]["op15"]["usb_in_j"], 50.0, places=6)

    def test_compare_conservative_attached_and_shift(self):
        base = fleet_energy.account_run("base", write_run(
            self.root / "b", op15=_op15_log(usb_ua=100_000, discharging=False), pixel=_pixel_log(charging=False),
            result=_result(cpu_j=11000.0, gpu_j=5000.0)))
        treat = fleet_energy.account_run("t", write_run(self.root / "t", op15=_op15_log(), pixel=_pixel_log(charging=False)))
        rows = fleet_energy.compare({"base": base, "t": treat}, "base")
        self.assertAlmostEqual(rows["t"]["host_saving"], 0.5)
        phones_t, phones_b = treat["phones_j"], base["phones_j"]
        self.assertAlmostEqual(phones_b, 50.0 + 50.0, places=6)
        self.assertAlmostEqual(rows["t"]["fleet_saving_conservative"], 1 - (8000.0 + phones_t) / 16000.0)
        self.assertAlmostEqual(rows["t"]["fleet_saving_attached"], 1 - (8000.0 + phones_t) / (16000.0 + phones_b))
        self.assertAlmostEqual(rows["t"]["shift_ratio"], (phones_t - phones_b) / 8000.0)
        text = fleet_energy.markdown({"base": base, "t": treat}, rows)
        self.assertIn("| t | 100 |", text)

    def test_cli_writes_json(self):
        run = write_run(self.root / "cli", op15=_op15_log(), pixel=_pixel_log())
        out = self.root / "out.json"
        with open(os.devnull, "w") as sink:
            stdout, sys.stdout = sys.stdout, sink
            try:
                code = fleet_energy.main(["--run", "a=" + str(run), "--baseline", "a", "--json", str(out)])
            finally:
                sys.stdout = stdout
        self.assertEqual(code, 0)
        data = json.loads(out.read_text())
        self.assertEqual(data["schema"], fleet_energy.SCHEMA)
        self.assertAlmostEqual(data["comparison"]["a"]["host_saving"], 0.0)

    def test_sampler_parser_skips_comments_and_keeps_absolute_nodes(self):
        rows = fleet_energy.parse_sampler("# start\n1.5 usb/current_now=10 /sys/x/y=1 bad=zz\n\n# stop\n")
        self.assertEqual(rows, [{"uptime_s": 1.5, "usb/current_now": 10.0, "/sys/x/y": 1.0}])


class LatencyTests(unittest.TestCase):
    # E2E seconds of runs s1c / s1d / s2a (longtail_eval_v2, 14 requests), sorted; values quoted in earlier docs
    S1C = [16.9, 25.3, 27.7, 82.3, 197.1, 219.0, 284.9, 339.6, 342.6, 409.5, 510.3, 591.5, 894.2, 923.3]
    S1D = [15.6, 19.3, 27.4, 114.4, 128.3, 148.8, 157.6, 213.9, 282.8, 289.1, 295.8, 312.6, 534.3, 692.9]
    S2A = [18.0, 26.5, 32.4, 72.3, 134.8, 172.1, 205.6, 273.6, 283.7, 289.6, 292.0, 327.8, 542.3, 740.9]

    def test_nearest_rank_reproduces_table2_p90_and_linear_the_readme(self):
        p = latency_report.percentile
        self.assertEqual([p(v, 90) for v in (self.S1C, self.S1D, self.S2A)], [894.2, 534.3, 542.3])
        self.assertEqual([round(p(v, 90, "linear")) for v in (self.S1C, self.S1D, self.S2A)], [803, 468, 478])
        self.assertEqual([round(p(v, 50, "linear")) for v in (self.S1C, self.S1D, self.S2A)], [312, 186, 240])
        self.assertEqual([round(p(v, 50, "index_floor")) for v in (self.S1C, self.S1D)], [340, 214])
        self.assertEqual([p(v, 50) for v in (self.S1C, self.S1D, self.S2A)], [284.9, 157.6, 205.6])

    def test_percentile_edges(self):
        p = latency_report.percentile
        self.assertEqual(p([3.0], 99), 3.0)
        self.assertEqual(p(list(range(1, 11)), 90), 9)        # 0.9 x 10 must not round up to rank 10
        self.assertEqual(p(list(range(1, 15)), 99), 14)
        self.assertEqual(p(list(range(1, 15)), 100), 14)
        self.assertIsNone(p([], 50))
        with self.assertRaises(ValueError):
            p([1.0], 0)
        with self.assertRaises(ValueError):
            p([1.0], 50, "nearest")

    def test_request_rows_ttft_tpot_queue_service(self):
        loads = {"terminal_ticket": {"transition_receipts": [{"started_us": 1_500_000, "finished_us": 1_900_000}]}}
        requests = [_request(0, 1, 2, 5, 25, 11, **loads), _request(1, 3, 3.5, 4, 4.5, 1),
                    _request(2, 5, 6, None, 9, 4, recoveries=[{"kind": "retire"}])]
        run = write_run(Path(self.tmp.name) / "lat", result=_result(requests=requests), samples=False,
                        streams={0: {"prompt_ms": 1000.0, "predicted_n": 11, "predicted_per_token_ms": 1990.0}})
        report = latency_report.report_run("lat", run)
        first, single, recovered = report["requests"]
        self.assertAlmostEqual(first["ttft_s"], 4.0)
        self.assertAlmostEqual(first["tpot_ms"], 2000.0)
        self.assertAlmostEqual(first["e2e_s"], 24.0)
        self.assertAlmostEqual(first["wait_s"], 1.0)
        self.assertAlmostEqual(first["load_s"], 0.4)
        self.assertAlmostEqual(first["prefill_s"], 3.0)
        self.assertAlmostEqual(first["server_prompt_s"], 1.0)
        self.assertAlmostEqual(first["server_queue_s"], 2.0)
        self.assertAlmostEqual(first["queue_s"], 3.0)
        self.assertAlmostEqual(first["service_s"], 21.0)
        self.assertEqual(first["server_tpot_ms"], 1990.0)
        self.assertIsNone(single["tpot_ms"])
        self.assertAlmostEqual(single["service_s"], 1.0)   # no stream: end - execution start
        self.assertIsNone(recovered["ttft_s"])
        self.assertTrue(recovered["recovered"])
        overall = report["overall"]
        self.assertEqual(overall["n"], 3)
        self.assertEqual(overall["e2e_s"]["p90"], 24.0)
        self.assertAlmostEqual(overall["token_weighted_tpot_ms"], 2000.0)
        self.assertEqual(sorted(report["by_model"]), ["gemma", "llama", "qwen"])
        self.assertIn("nearest rank", latency_report.markdown({"lat": report}))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()


class EnergyLatencyTests(unittest.TestCase):
    @staticmethod
    def pts(*rows):
        return [{"label": label, "family": "t", "host_kj": energy, "fleet_kj": energy, "p90_e2e_s": latency,
                 "mean_tpot_ms": latency} for label, latency, energy in rows]

    def test_step_never_uses_a_slower_run(self):
        points = self.pts(("a", 500, 100.0), ("b", 800, 80.0))
        self.assertIsNone(energy_latency.step_energy(points, "p90_e2e_s", "host_kj", 400))
        self.assertEqual(energy_latency.step_energy(points, "p90_e2e_s", "host_kj", 600)["run"], "a")
        self.assertEqual(energy_latency.step_energy(points, "p90_e2e_s", "host_kj", 900)["energy"], 80.0)

    def test_hull_interpolates_only_inside_and_labels_percentiles(self):
        points = self.pts(("a", 500, 100.0), ("b", 800, 70.0), ("c", 600, 99.0))
        self.assertEqual(energy_latency.lower_hull([(p["p90_e2e_s"], p["host_kj"]) for p in points]),
                         [(500, 100.0), (800, 70.0)])  # c lies above the chord
        mid = energy_latency.hull_energy(points, "p90_e2e_s", "host_kj", 650)
        self.assertAlmostEqual(mid["energy"], 85.0)
        self.assertEqual(mid["assumption"], "mix")
        self.assertIsNone(energy_latency.hull_energy(points, "p90_e2e_s", "host_kj", 499))
        self.assertAlmostEqual(energy_latency.hull_energy(points, "p90_e2e_s", "host_kj", 1000)["energy"], 70.0)
        self.assertEqual(energy_latency.hull_energy(points, "mean_tpot_ms", "host_kj", 650)["assumption"],
                         "time-shared mix (additive metric)")

    def test_compare_dominance_frozen_and_spread(self):
        points = self.pts(("a", 500, 100.0), ("b", 800, 70.0)) + [
            {"label": "ref", "family": "base", "host_kj": 200.0, "fleet_kj": 202.0, "p90_e2e_s": 700,
             "mean_tpot_ms": 700},
            {"label": "ref2", "family": "base", "host_kj": 220.0, "fleet_kj": 222.0, "p90_e2e_s": 900,
             "mean_tpot_ms": 900}]
        rows = energy_latency.compare(points, ["base"], ["t"], "b")
        row = next(r for r in rows if r["reference"] == "ref" and r["latency_metric"] == "p90_e2e_s"
                   and r["energy_metric"] == "host_kj")
        self.assertEqual(row["dominating_runs"], ["a"])
        self.assertAlmostEqual(row["step_saving"], 0.5)
        self.assertAlmostEqual(row["hull_energy"], 80.0)
        self.assertFalse(row["frozen"]["no_slower"])
        spread = energy_latency.spread(points, [["ref", "ref2"]])
        self.assertAlmostEqual(spread[0]["host_kj"]["range_over_mean"], 20.0 / 210.0)
        text = energy_latency.markdown(points, rows, spread)
        self.assertIn("| ref | p90_e2e_s | 700 | host_kj | 200.0 | t | 100.0 (a) | 50.0 % |", text)

    def test_point_from_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp) / "p", op15=_op15_log(), pixel=_pixel_log(),
                            result=_result(requests=[_request(0, 1, 2, 5, 25, 11), _request(1, 2, 3, 4, 10, 7)]))
            point = energy_latency.point("p", "t", run)
        self.assertAlmostEqual(point["host_kj"], 8.0)
        self.assertAlmostEqual(point["fleet_kj"], 8.642)
        self.assertEqual(point["p90_e2e_s"], 24.0)
        self.assertAlmostEqual(point["mean_tpot_ms"], (2000.0 + 1000.0) / 2)


FAKE_ADB = r'''#!/usr/bin/env python3
import os, re, shutil, subprocess, sys
root = os.environ["FAKE_PHONES"]
args = sys.argv[1:]
if args[:1] == ["-P"]:
    args = args[2:]
if args[:1] == ["-s"]:
    serial, args = args[1], args[2:]
base = os.path.join(root, serial)
def mapped(text):
    return text.replace("/sys/", base + "/sys/").replace("/data/local/tmp", base + "/tmp").replace("/proc/uptime", base + "/uptime")
if args[0] == "get-state":
    print("device"); sys.exit(0)
if args[0] == "push":
    dst = mapped(args[2])
    if dst.endswith("/"):
        dst = os.path.join(dst, os.path.basename(args[1]))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w").write(mapped(open(args[1]).read())); sys.exit(0)
if args[0] == "pull":
    shutil.copy(mapped(args[1]), args[2]); sys.exit(0)
line = " ".join(args[1:])
match = re.fullmatch(r"su -c '(.*)'", line, re.S)
line = match.group(1) if match else line
if line.startswith("dumpsys"):
    print("level: 80"); sys.exit(0)
sys.exit(subprocess.call(["/bin/sh", "-c", mapped(line)]))
'''


class MeterScriptTests(unittest.TestCase):
    NODES = {OP15: {"class/power_supply/usb/current_now": "494000", "class/power_supply/usb/voltage_now": "5100000",
                    "class/power_supply/usb/input_current_limit": "500000", "class/power_supply/battery/current_now": "300",
                    "class/power_supply/battery/voltage_now": "4100000",
                    "class/power_supply/battery/charge_counter": "5000000", "class/power_supply/battery/capacity": "80",
                    "class/oplus_chg/battery/mmi_charging_enable": "1"},
             PIXEL: {"class/power_supply/usb/current_now": "150000", "class/power_supply/usb/voltage_now": "5100000",
                     "class/power_supply/usb/input_current_limit": "900000",
                     "class/power_supply/battery/current_now": "-2000", "class/power_supply/battery/current_avg": "-1000",
                     "class/power_supply/battery/voltage_now": "4400000",
                     "class/power_supply/battery/charge_counter": "4000000", "class/power_supply/battery/capacity": "90",
                     "devices/platform/google,charger/charge_stop_level": "100"}}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for serial, nodes in self.NODES.items():
            for node, value in nodes.items():
                path = self.root / "phones" / serial / "sys" / node
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value + "\n")
            (self.root / "phones" / serial / "tmp").mkdir(parents=True)
            (self.root / "phones" / serial / "uptime").write_text("1000.00 5000.00\n")
        adb = self.root / "adb"
        adb.write_text(FAKE_ADB)
        adb.chmod(adb.stat().st_mode | stat.S_IEXEC)
        self.lock = self.root / "rig.lock"
        self.lock.write_text("")
        self.env = {**os.environ, "FAKE_PHONES": str(self.root / "phones"), "METER_ADB": "%s %s" % (sys.executable, adb),
                    "METER_LOCK": str(self.lock), "METER_SETTLE_S": "0"}

    def tearDown(self):
        self.tmp.cleanup()

    def node(self, serial, node):
        return (self.root / "phones" / serial / "sys" / node).read_text().strip()

    def run_meter(self, code=0, mode="controlled"):
        record = self.root / "during.txt"
        command = ("cat %s %s > %s; sleep 1; exit %d" % (
            self.root / "phones" / OP15 / "sys/class/oplus_chg/battery/mmi_charging_enable",
            self.root / "phones" / PIXEL / "sys/devices/platform/google,charger/charge_stop_level", record, code))
        out = self.root / ("meter-%s-%d" % (mode, code))
        completed = subprocess.run(["bash", str(SCRIPTS / "meter_phones.sh"), "--charging", mode, "--op15-period", "0.2",
                                    "--pixel-period", "0.1", str(out), "t1", "--", "bash", "-c", command],
                                   env=self.env, capture_output=True, text=True, timeout=120)
        return completed, out, record

    def test_syntax(self):
        for name in ("meter_phones.sh", "phone_power_sampler.sh"):
            self.assertEqual(subprocess.run(["bash", "-n", str(SCRIPTS / name)]).returncode, 0, name)

    def test_controlled_run_sets_and_restores_charging_and_writes_meter(self):
        completed, out, record = self.run_meter()
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(record.read_text().split(), ["0", "80"])        # during: OP15 mmi 0, Pixel stop = 90 - 10
        self.assertEqual(self.node(OP15, "class/oplus_chg/battery/mmi_charging_enable"), "1")
        self.assertEqual(self.node(PIXEL, "devices/platform/google,charger/charge_stop_level"), "100")
        meter = json.loads((out / "METER.json").read_text())
        self.assertEqual(sorted(meter), ["op15", "pixel"])
        for label, row in meter.items():
            self.assertTrue(row["stopped"], label)
            self.assertGreater(row["samples"], 0)
            self.assertEqual(row["charging_mode"], "controlled")
            self.assertIn("anchor_host_monotonic_s", row)
            rows = fleet_energy.parse_sampler(Path(row["local"]).read_text())
            self.assertIn("usb/current_now", rows[0])
            self.assertIn("battery/charge_counter", rows[0])
        self.assertIn("verified=1", (out / "t1-meter.log").read_text())

    def test_failing_command_still_restores_and_propagates_exit_code(self):
        completed, out, _ = self.run_meter(code=7)
        self.assertEqual(completed.returncode, 7, completed.stdout + completed.stderr)
        self.assertEqual(self.node(OP15, "class/oplus_chg/battery/mmi_charging_enable"), "1")
        self.assertEqual(self.node(PIXEL, "devices/platform/google,charger/charge_stop_level"), "100")

    def test_as_is_mode_changes_nothing(self):
        completed, _, record = self.run_meter(mode="as-is")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(record.read_text().split(), ["1", "100"])

    def test_charge_control_that_does_not_take_effect_aborts_and_restores(self):
        stop_node = self.root / "phones" / PIXEL / "sys/devices/platform/google,charger/charge_stop_level"
        stop_node.chmod(0o444)
        try:
            completed, out, record = self.run_meter()
        finally:
            stop_node.chmod(0o644)
        self.assertEqual(completed.returncode, 4, completed.stdout + completed.stderr)
        self.assertFalse(record.exists())
        self.assertEqual(self.node(OP15, "class/oplus_chg/battery/mmi_charging_enable"), "1")
        self.assertIn("did not take effect", (out / "t1-meter.log").read_text())

    def test_refuses_while_rig_lock_is_held(self):
        with open(self.lock) as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            completed, out, record = self.run_meter()
        self.assertEqual(completed.returncode, 3)
        self.assertFalse(record.exists())
        self.assertFalse((out / "METER.json").exists())
        self.assertEqual(self.node(OP15, "class/oplus_chg/battery/mmi_charging_enable"), "1")


if __name__ == "__main__":
    unittest.main()
