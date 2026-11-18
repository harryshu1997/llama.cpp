"""Portable path: phone power parsing, the run-directory loader and the reports, on synthetic runs."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler._internal.layer_placement import LayerPlacementError  # noqa: E402
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError  # noqa: E402
from research_dev.scheduler.campaigns.portable import report  # noqa: E402
from research_dev.scheduler.campaigns.portable.rundir import RunDirectory, RunDirectoryError  # noqa: E402
from research_dev.scheduler.drivers.meter import (  # noqa: E402
    PhoneClock,
    PhonePowerCalibration,
    parse_phone_power,
    phone_power_between,
)

import portable_fixtures as fx  # noqa: E402

S = fx.SECOND
GAUGE = PhonePowerCalibration("mA", 1, 0.5, "synthetic")


class PhonePowerTests(unittest.TestCase):
    def test_lines_parse_and_text_fields_drop(self):
        rows = parse_phone_power("\n".join(["# header", "", "10.5 usb_ua=400000 usb_uv=5000000 note=text bat_ma=-3",
                                             "broken", "x usb_ua=1"]))
        self.assertEqual(rows, [{"t": 10.5, "usb_ua": 400000.0, "usb_uv": 5000000.0, "bat_ma": -3.0}])

    def test_power_between_uses_the_clock_pairing_and_the_calibration(self):
        rows = parse_phone_power("\n".join(fx.phone_lines(100.0, 20, usb_w=2.0, battery_ma=100.0)))
        clock = PhoneClock(100.0, 50 * S)
        power = phone_power_between(rows, clock, GAUGE, 52 * S, 60 * S)
        self.assertEqual(power["n"], 9)
        self.assertAlmostEqual(power["usb_w"], 2.0)
        self.assertAlmostEqual(power["bat_w"], 0.1 * 4.0 / 0.5)
        self.assertAlmostEqual(power["w"], 2.8)
        self.assertEqual(power["battery_c"], (30.0, 30.0, 30.0))

    def test_microampere_gauge_with_charging_positive(self):
        calibration = PhonePowerCalibration("uA", -1, 1.0, "synthetic")
        self.assertAlmostEqual(calibration.battery_w(-250000.0, 4_000_000.0), 1.0)
        self.assertEqual(PhonePowerCalibration.from_json(calibration.to_json()), calibration)

    def test_absent_when_it_cannot_be_measured(self):
        rows = parse_phone_power("\n".join(fx.phone_lines(100.0, 20, usb_w=2.0)))
        clock = PhoneClock(100.0, 50 * S)
        self.assertIsNone(phone_power_between(rows, None, GAUGE, 52 * S, 60 * S))
        self.assertIsNone(phone_power_between(rows, clock, None, 52 * S, 60 * S))
        self.assertIsNone(phone_power_between(rows, clock, GAUGE, 52 * S, 53 * S))       # two samples
        self.assertIsNone(phone_power_between([{"t": 101.0}, {"t": 102.0}, {"t": 103.0}], clock, GAUGE, 50 * S, 60 * S))

    def test_calibration_is_validated(self):
        for arguments in (("A", 1, 1.0, "x"), ("mA", 0, 1.0, "x"), ("mA", 1, 0.0, "x"), ("mA", 1, 1.0, "")):
            with self.assertRaises(PhysicalAdapterError):
                PhonePowerCalibration(*arguments)


def seq_run(root: Path, *, phone="measured") -> Path:
    """Two requests, each alone (60 tokens in 6 s at 300 W) then shared (5 s at 280 W), 3 W phone."""
    phases, instant = [], 10 * S
    busy = []
    for index in range(2):
        for columns, seconds, watts in ((0, 6, (200.0, 100.0)), (3200, 5, (190.0, 90.0))):
            submitted, first = instant, instant + S
            finished = first + seconds * S
            busy.append((submitted, finished, watts))
            name = f"r{index:02d}c{columns}"
            phases.append(fx.phase(name, [fx.request(f"request-{index}", columns, submitted, first, finished, 61,
                                                     sha=f"sha-{index}")]))
            instant = finished + 2 * S
    warm = fx.phase("warm0", [fx.request("warm", 0, 2 * S, 3 * S, 5 * S, 11)])

    def power_at(t_ns):
        for start, end, watts in busy:
            if start <= t_ns <= end:      # the sample at the window's end still sees the busy power
                return watts
        return (100.0, 20.0)

    extra = {}
    if phone == "measured":
        extra = {"phone_lines": fx.phone_lines(1000.0, 80, usb_w=3.0), "phone_clock": (1000.0, 0),
                 "phone_power": {"calibration": GAUGE.to_json()}}
    elif phone == "assumed":
        extra = {"phone_power": {"assumed_active_w": 4.5, "assumed_idle_w": 0.875}}
    return fx.write_run(root, phases=[warm, *phases], samples=fx.power_samples(0, instant + 5 * S, power_at),
                        idle=(6 * S, 9 * S, 100.0, 20.0), **extra)


class SeqReportTests(unittest.TestCase):
    def test_per_token_saving_with_measured_phone(self):
        with tempfile.TemporaryDirectory() as directory:
            result = report.seq_report(RunDirectory(seq_run(Path(directory) / "run")))
        summary = result["summary"]
        self.assertEqual((summary["requests"], summary["decode_tokens"], summary["identical"]), (2, 120, 2))
        self.assertAlmostEqual(summary["alone"]["ms_token"], 100.0, places=6)
        self.assertAlmostEqual(summary["shared"]["ms_token"], 5000.0 / 60, places=6)
        self.assertAlmostEqual(summary["d_time"], -1.0 / 6, places=6)
        # alone 300 W x 6 s; shared (280 W + 3 W phone) x 5 s
        self.assertAlmostEqual(summary["alone"]["j_token"], 30.0, places=3)
        self.assertAlmostEqual(summary["shared"]["j_token"], 283.0 * 5 / 60, places=3)
        self.assertAlmostEqual(summary["shared"]["phone_j_token"], 0.25, places=6)
        self.assertEqual(summary["alone"]["phone_j_token"], 0.0)
        self.assertEqual(summary["phone_energy_quality"], "measured")
        self.assertEqual([row["same"] for row in result["requests"]], [True, True])
        self.assertIn("-16.7 %", report.format_seq(result))

    def test_assumed_and_absent_phone_energy_are_stated(self):
        with tempfile.TemporaryDirectory() as directory:
            assumed = report.seq_report(RunDirectory(seq_run(Path(directory) / "a", phone="assumed")))["summary"]
            absent = report.seq_report(RunDirectory(seq_run(Path(directory) / "b", phone="absent")))
        self.assertEqual(assumed["phone_energy_quality"], "assumed")
        self.assertAlmostEqual(assumed["shared"]["phone_j_token"], 4.5 * 5 / 60, places=6)
        self.assertEqual(absent["summary"]["phone_energy_quality"], "absent")
        self.assertEqual(absent["summary"]["shared"]["phone_j_token"], 0.0)
        self.assertIn("phone energy absent", report.format_seq(absent))

    def test_command_line_calibration_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            run = seq_run(Path(directory) / "run")
            (run / "PHONE_POWER.json").unlink()
            out = Path(directory) / "REPORT.json"
            with redirect_stdout(io.StringIO()) as text:
                self.assertEqual(report.main(["--gauge-ratio", "0.5", "--json", str(out), "seq", str(run)]), 0)
            self.assertIn("pooled over 2 requests", text.getvalue())
            self.assertEqual(json.loads(out.read_text())["summary"]["phone_energy_quality"], "measured")
            self.assertEqual(RunDirectory(run).phone_quality(), "absent")

    def test_unfinished_run_and_missing_pairs_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            run = seq_run(Path(directory) / "run")
            for name in ("PHASE-2.json", "PHASE-4.json"):
                (run / name).unlink()          # no request is left with both arms
            with self.assertRaisesRegex(LayerPlacementError, "both alone and shared"):
                report.seq_report(RunDirectory(run))
            (run / "RESULT.json").unlink()
            with self.assertRaisesRegex(RunDirectoryError, "did not finish"):
                RunDirectory(run)

    def test_disturbed_attempts_are_never_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            run = seq_run(Path(directory) / "run")
            (run / "PHASE-1.disturbed-0.json").write_text(json.dumps({"name": "r00c0", "cancelled": True}))
            self.assertEqual(len(RunDirectory(run).phases), 5)


def sweep_run(root: Path) -> Path:
    """Host-only brackets around three shares; the shape rows follow a known model."""
    legs = ((0, 10.0), (1600, 9.0), (0, 10.0), (3200, 8.0), (4800, 9.5), (0, 10.0))
    phases, busy, instant = [], [], 10 * S
    for index, (columns, seconds) in enumerate(legs):
        submitted, first = instant, instant + S
        finished = first + int(seconds * S)
        busy.append((first, finished, (200.0, 100.0)))
        phases.append(fx.phase(f"leg{index:02d}c{columns}",
                               [fx.request("probe", columns, submitted, first, finished, 101)]))
        instant = finished + 2 * S
    shapes = []
    for columns in (1600, 3200, 4800):
        share = columns / fx.N_FF
        host, rpc = 0.5 + (1 - share) * 8.0, 1.0 + share * 16.0
        shapes.append({"tokens": 1, "columns": columns, "calls": 200, "overlap_mean_ms": max(host, rpc),
                       "host_mean_ms": host, "rpc_mean_ms": rpc, "compute_mean_ms": share * 16.0})

    def power_at(t_ns):
        return next((watts for start, end, watts in busy if start <= t_ns <= end), (100.0, 20.0))

    return fx.write_run(root, phases=phases, samples=fx.power_samples(0, instant + 5 * S, power_at),
                        idle=(2 * S, 8 * S, 100.0, 20.0), shapes=shapes,
                        phone_power={"assumed_active_w": 3.0, "assumed_idle_w": 1.0})


class SweepReportTests(unittest.TestCase):
    def test_legs_fit_and_pooled_savings(self):
        with tempfile.TemporaryDirectory() as directory:
            run = RunDirectory(sweep_run(Path(directory) / "run"))
            legs = report.legs_report(run)
            sweep = report.sweep_report([run])
            text = report.format_sweep(sweep) + report.format_legs(legs)
        self.assertEqual([leg["columns"] for leg in legs["legs"]], [0, 1600, 0, 3200, 4800, 0])
        self.assertEqual(legs["token_identity"], [True])
        self.assertAlmostEqual(legs["legs"][3]["ms_token"], 80.0, places=6)
        self.assertAlmostEqual(legs["legs"][3]["d_time"], -0.2, places=6)
        self.assertEqual(legs["legs"][1]["phone_quality"], "assumed")
        self.assertAlmostEqual(legs["legs"][0]["phone_w"], 1.0)       # host-only leg: the phone idles
        self.assertAlmostEqual(legs["legs"][1]["phone_w"], 3.0)
        fit = sweep["sweep"]["fit"]
        self.assertAlmostEqual(fit["t_host_ms"], 8.0, places=9)
        self.assertAlmostEqual(fit["c_host_ms"], 0.5, places=9)
        self.assertAlmostEqual(fit["t_link_ms"], 1.0, places=9)
        self.assertAlmostEqual(fit["t_phone_ms"], 16.0, places=9)
        self.assertAlmostEqual(fit["f_star"], 7.5 / 24.0, places=9)
        pooled = sweep["pooled"]
        self.assertEqual(sorted(pooled), ["0.25", "0.5", "0.75"])
        self.assertAlmostEqual(pooled["0.5"]["time_saving_mean"], 20.0, places=4)
        # host J per token falls with the time at constant power: P_F = the host's decode power
        self.assertAlmostEqual(pooled["0.5"]["marginal_host_w"], 300.0, places=3)
        self.assertEqual(sweep["phone_energy_quality"], ["assumed"])
        self.assertIn("best share 31.2 %", text)
        self.assertIsNone(report.whole_layer_bound(sweep, "0.5"))     # no full-width leg in this sweep

    def test_run_without_host_only_leg_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            run = sweep_run(Path(directory) / "run")
            for number in (0, 2, 5):
                (run / f"PHASE-{number}.json").unlink()
            with self.assertRaisesRegex(LayerPlacementError, "host-only leg"):
                report.legs_report(RunDirectory(run))


def trace_arm(root: Path, mode: str, decode_s: float, watts) -> Path:
    """Two requests arriving 0 and 4 s after the origin; each decodes for ``decode_s`` after 1 s."""
    origin = 10 * S
    requests, busy = [], []
    for index, arrival in enumerate((0, 4)):
        target = origin + arrival * S
        first = target + S
        finished = first + int(decode_s * S)
        busy.append((target, finished))
        requests.append(fx.request(f"request-{index}", 0 if mode == "alone" else 3200, target, first, finished, 41,
                                   sha=f"sha-{index}", target_ns=target))

    def power_at(t_ns):
        return watts if any(start <= t_ns <= end for start, end in busy) else (100.0, 20.0)

    end = max(finished for _, finished in busy)
    extra = {"phone_lines": fx.phone_lines(500.0, 60, usb_w=2.0), "phone_clock": (500.0, 0),
             "phone_power": {"calibration": GAUGE.to_json()}} if mode == "shared" else {}
    # the meters stop two seconds after the last request: a longer common window is filled at idle power
    return fx.write_run(root, mode=mode, phases=[fx.phase("trace", requests)],
                        samples=fx.power_samples(0, end + 2 * S, power_at),
                        idle=(2 * S, 8 * S, 100.0, 20.0), tail_idle=(end + S, end + 2 * S, 100.0, 20.0), **extra)


class TraceReportTests(unittest.TestCase):
    def test_two_arms_over_the_common_window(self):
        with tempfile.TemporaryDirectory() as directory:
            alone = RunDirectory(trace_arm(Path(directory) / "alone", "alone", 5.0, (200.0, 100.0)))
            shared = RunDirectory(trace_arm(Path(directory) / "shared", "shared", 4.0, (200.0, 100.0)))
            result = report.trace_report(alone, shared)
            text = report.format_trace(result)
        self.assertAlmostEqual(result["window_s"], 10.0)
        self.assertEqual(result["identical_outputs"], [True, True])
        a, b = result["alone"], result["shared"]
        self.assertEqual((a["peak_active"], b["peak_active"]), (2, 2))
        self.assertAlmostEqual(a["busy_s"], 10.0)
        self.assertAlmostEqual(b["busy_s"], 9.0)
        # alone: busy the whole window at 300 W; shared: 9 s busy + 1 s idle (120 W), phone 2 W for 10 s
        # (a power step inside a window is resolved to one sample period: 0.02 kJ of slack)
        self.assertAlmostEqual(a["window"]["total_kj"], 3.0, delta=0.02)
        self.assertAlmostEqual(b["window"]["cpu_kj"] + b["window"]["gpu_kj"], 2.82, delta=0.02)
        self.assertAlmostEqual(b["window"]["phone_kj"], 0.02, places=4)
        self.assertEqual(a["window"]["phone_kj"], 0.0)
        self.assertAlmostEqual(result["saving_window_total"], 1 - 2.84 / 3.0, delta=0.007)
        self.assertAlmostEqual(result["d_e2e_mean"], -1.0 / 6, places=6)
        self.assertAlmostEqual(result["d_tpot_mean"], -0.2, places=6)
        # alone the two decodes overlap for one second (9 + 8 tokens); shared they only touch
        self.assertEqual(a["tokens_by_decoding_requests"], {1: 63, 2: 17})
        self.assertEqual(b["tokens_by_decoding_requests"], {1: 79, 2: 1})
        self.assertIn("identical outputs 2/2", text)

    def test_window_beyond_the_meters_is_filled_at_idle_power(self):
        with tempfile.TemporaryDirectory() as directory:
            alone = RunDirectory(trace_arm(Path(directory) / "alone", "alone", 9.0, (200.0, 100.0)))
            shared = RunDirectory(trace_arm(Path(directory) / "shared", "shared", 4.0, (200.0, 100.0)))
            result = report.trace_report(alone, shared)
        self.assertAlmostEqual(result["window_s"], 14.0)
        window = result["shared"]["window"]
        # the shared arm's meters end 11 s after the origin (minus the 0.2 s guard): about 3.2 s extrapolated
        self.assertAlmostEqual(window["extrapolated_s"], 3.2, places=1)
        self.assertAlmostEqual(window["cpu_kj"] + window["gpu_kj"], (9 * 300 + 5 * 120) / 1e3, delta=0.02)


class RowsReportTests(unittest.TestCase):
    def test_two_requests_decoding_together(self):
        with tempfile.TemporaryDirectory() as directory:
            origin = 10 * S
            phases, busy, instant = [], [], origin
            for name, columns, seconds in (("k2c0", 0, 8), ("k2c3200", 3200, 6)):
                requests = [fx.request(f"request-{index}", columns, instant, instant + S, instant + S + seconds * S, 81)
                            for index in range(2)]
                busy.append((instant, instant + S + seconds * S))
                phases.append(fx.phase(name, requests))
                instant += (seconds + 4) * S
            samples = fx.power_samples(0, instant, lambda t: (200.0, 100.0) if any(a <= t <= b for a, b in busy)
                                       else (100.0, 20.0))
            run = RunDirectory(fx.write_run(Path(directory) / "run", phases=phases, samples=samples,
                                            idle=(2 * S, 8 * S, 100.0, 20.0)))
            result = report.rows_report(run, skip=8)
            text = report.format_rows(result)
        alone, shared = result["rows"]
        self.assertEqual((alone["k"], shared["k"]), (2, 2))
        self.assertAlmostEqual(alone["ms_step"], 100.0, places=3)
        self.assertAlmostEqual(shared["ms_step"], 75.0, places=3)
        self.assertAlmostEqual(shared["d_time"], -0.25, places=4)
        self.assertAlmostEqual(alone["j_token"], 300.0 * 0.1 / 2, places=2)
        self.assertEqual((alone["d_time"], alone["columns"], shared["columns"]), (0.0, 0, 3200))
        self.assertIn("k2c3200", text)


if __name__ == "__main__":
    unittest.main()
