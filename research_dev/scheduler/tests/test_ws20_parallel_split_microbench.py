"""WS20: the two-phone concurrency microbenchmark (config, pair driving with a barrier, analysis verdict)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from research_dev.scheduler.campaigns.burstgpt.tools import parallel_split_microbench as psm  # noqa: E402
from research_dev.scheduler.campaigns.burstgpt.tools import weight_stream_microbench as wsm  # noqa: E402

from test_ws16_weight_stream_microbench import Clock, FakeWorker  # noqa: E402


def phone(name, device, transport, version):
    return {"name": name, "device": device, "transport": transport, "protocol_version": version,
            "activation": "swiglu", "layer": 18, "n_embd": 8, "n_ff": 64,
            "est_call_ms": {"1": 8.0, "2": 15.0, "4": 32.0}, "evidence": f"/e-{name}.json"}


def base_config(**changes):
    config = {
        "schema": psm.SCHEMA, "rows": [1, 2, 4], "segment_seconds": 10, "idle_s": 5, "cooldown_s": 0,
        "column_quantum": 32, "max_degradation": 1.10, "max_penalty_ms": 1.0,
        "cadence": {"calls_per_token": 3, "gap_call_ms": 5, "gap_token_ms": 300, "warmup_tokens": 2},
        "phones": [phone("op15", "op15", "adb-tcp", 5), phone("pixel", "pixel", "aoa-bridge", 6)],
    }
    config.update(changes)
    return config


class ConfigTests(unittest.TestCase):
    def test_validation_arms_widths_and_plan(self):
        config = base_config()
        psm.validate_config(config)
        self.assertEqual((psm.arm_width(config, "solo-half"), psm.arm_width(config, "pair-full")), (32, 64))
        planned = psm.plan(config)
        self.assertEqual([entry["arm"] for entry in planned["legs"]],
                         ["solo-full", "solo-full", "solo-half", "solo-half", "pair-half", "pair-full"])
        pair = next(entry for entry in planned["legs"] if entry["arm"] == "pair-half")
        self.assertEqual(len(pair["phones"]), 2)
        # both phones of a pair arm share one budget (the slower estimate decides the token count)
        legs = [psm.leg_for(config, p, "pair-half") for p in config["phones"]]
        self.assertEqual(wsm.phone_budget(config, legs[0]), wsm.phone_budget(config, legs[1]))
        for bad in (dict(base_config(), arms=["nope"]), dict(base_config(), column_quantum=48),
                    dict(base_config(), phones=base_config()["phones"][:1])):
            with self.assertRaises(SystemExit):
                psm.validate_config(bad)
        small = psm.smoke_config(config)
        self.assertEqual((small["rows"], small["arms"], small["idle_s"]), ([1], ["solo-full", "pair-half"], 2))


class DriverTests(unittest.TestCase):
    def test_half_width_calls_carry_the_width_and_exhaust_the_budget(self):
        config = base_config()
        leg = psm.leg_for(config, config["phones"][1], "solo-half")
        worker = FakeWorker(6)
        worker.start()
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "LEG.jsonl"
            sent = psm.drive_leg(worker.port, config, leg, "sha256:" + "ab" * 32, 4, 32, out, clock=clock, sleep=clock.sleep)
            records = wsm.read_jsonl(out)
        worker.join(5)
        self.assertEqual(sent, wsm.phone_budget(config, leg))
        self.assertEqual(len(worker.executes), sent)
        self.assertEqual({fields[8] for fields in worker.executes}, {32})          # columns = half width
        meta = next(r for r in records if r["kind"] == "meta")
        self.assertEqual((meta["columns"], meta["paired"], meta["logical_bytes"]), (32, False, 3 * 8 * 64 * 2 * 32 // 64))
        self.assertEqual([r["rows"] for r in records if r["kind"] == "segment"], [1, 2, 4])

    def test_pair_driving_releases_both_phones_calls_together(self):
        config = base_config()
        legs = [psm.leg_for(config, p, "pair-half") for p in config["phones"]]
        workers = [FakeWorker(5), FakeWorker(6)]
        for worker in workers:
            worker.start()
        barrier = threading.Barrier(2)
        sent = {}
        clock = Clock()
        lock = threading.Lock()

        def ticking():
            with lock:
                return clock()

        def sleep(seconds):
            with lock:
                clock.sleep(seconds)

        def work(index):
            with tempfile.TemporaryDirectory() as directory:
                sent[index] = psm.drive_leg(workers[index].port, config, legs[index], "sha256:" + "cd" * 32, 4, 32,
                                            Path(directory) / "LEG.jsonl", barrier=barrier.wait, clock=ticking, sleep=sleep)
        threads = [threading.Thread(target=work, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        for worker in workers:
            worker.join(5)
        self.assertEqual(sent[0], sent[1])
        self.assertEqual(len(workers[0].executes), len(workers[1].executes))
        self.assertFalse(barrier.broken)


def write_leg(directory: Path, arm: str, phone: str, device: str, columns: int, calls: dict, t0: int):
    """A synthetic LEG.jsonl: idle windows and one segment per row count with the given rpc/compute (ms)."""
    rows = []
    rows.append({"kind": "meta", "schema": "ws20-phone-leg-v1", "op": "ffn", "device": device, "layer": 18, "n_embd": 8,
                 "n_ff": 64, "columns": columns, "logical_bytes": 1000, "paired": arm.startswith("pair"), "arm": arm,
                 "phone": phone, "calls_per_token": 3, "gap_call_ms": 5, "gap_token_ms": 300, "warmup_tokens": 2,
                 "protocol_version": 6, "transport": "aoa-bridge"})
    t = t0
    rows.append({"kind": "idle", "phase": "before", "t0_ns": t, "t1_ns": t + 5_000_000_000}); t += 5_000_000_000
    for count, (rpc_ms, compute_ms) in calls.items():
        start = t
        for index in range(20):
            rows.append({"kind": "call", "rows": count, "warmup": False, "token": index, "position": 0, "t_start_ns": t,
                         "t_end_ns": t + int(rpc_ms * 1e6), "rpc_us": rpc_ms * 1000, "compute_us": compute_ms * 1000})
            t += int(rpc_ms * 1e6) + 300_000_000
        rows.append({"kind": "segment", "rows": count, "t0_ns": start, "t1_ns": t, "calls": 20, "warmup_t0_ns": start})
    rows.append({"kind": "idle", "phase": "after", "t0_ns": t, "t1_ns": t + 5_000_000_000})
    directory.mkdir(parents=True, exist_ok=True)
    name = "LEG.jsonl" if not arm.startswith("pair") else phone + "-LEG.jsonl"
    with (directory / name).open("w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")


class AnalysisTests(unittest.TestCase):
    def analyze(self, pair_half):
        config = base_config()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            out.mkdir()
            wsm.write_json(out / "CONFIG.json", config)
            solo_full = {"op15": {1: (10.0, 9.2), 2: (10.4, 9.5), 4: (11.0, 10.0)},
                         "pixel": {1: (9.4, 8.5), 2: (12.2, 11.0), 4: (20.0, 18.5)}}
            solo_half = {"op15": {1: (5.5, 4.7), 2: (5.8, 4.9), 4: (6.1, 5.2)},
                         "pixel": {1: (5.0, 4.3), 2: (6.6, 5.6), 4: (10.5, 9.3)}}
            t0 = 1_000_000_000_000
            for phone, device in (("op15", "op15"), ("pixel", "pixel")):
                write_leg(out / f"solo-full_{phone}", "solo-full", phone, device, 64, solo_full[phone], t0); t0 += 10 ** 12
                write_leg(out / f"solo-half_{phone}", "solo-half", phone, device, 32, solo_half[phone], t0); t0 += 10 ** 12
            for phone, device in (("op15", "op15"), ("pixel", "pixel")):
                write_leg(out / "pair-half", "pair-half", phone, device, 32, pair_half[phone], t0)
            return psm.analyze(out)

    def test_clean_concurrency_passes_with_a_small_penalty(self):
        result = self.analyze({"op15": {1: (5.6, 4.7), 2: (5.9, 4.9), 4: (6.2, 5.2)},
                               "pixel": {1: (5.1, 4.3), 2: (6.7, 5.6), 4: (10.6, 9.3)}})
        self.assertEqual((result["status"], result["flags"]), ("PASS", []))
        self.assertAlmostEqual(result["degradation"], 1.0, places=3)
        # rows 1: pair time 5.6 vs the slower solo half 5.5 -> penalty 0.1; rows 4: 10.6 vs 10.5
        self.assertAlmostEqual(result["penalty_ms"], 0.1, places=3)
        row = result["rows"][0]
        self.assertAlmostEqual(row["speedup_vs_single_phone"], 9.4 / 5.6, places=3)
        self.assertIn("| 1 | op15 |", psm.markdown(result))

    def test_interference_fails_the_verdict(self):
        result = self.analyze({"op15": {1: (7.5, 6.6), 2: (7.9, 6.9), 4: (8.4, 7.3)},
                               "pixel": {1: (6.9, 6.0), 2: (8.8, 7.6), 4: (13.5, 12.2)}})
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("DEGRADATION_ABOVE_LIMIT", result["flags"])
        self.assertIn("PENALTY_ABOVE_LIMIT", result["flags"])
        self.assertGreater(result["degradation"], 1.3)

    def test_missing_pair_arm_is_flagged(self):
        config = base_config()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            out.mkdir()
            wsm.write_json(out / "CONFIG.json", config)
            write_leg(out / "solo-full_op15", "solo-full", "op15", "op15", 64, {1: (10.0, 9.0)}, 10 ** 12)
            result = psm.analyze(out)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("PAIR_ARMS_MISSING", result["flags"])


class RunTests(unittest.TestCase):
    def test_run_records_every_unit_and_isolates_failures(self):
        class Sampler:
            def start(self): pass
            def stop(self): pass
            def rows(self): return [{"t_ns": 1}]

        def solo(config, arm, phone, directory):
            if arm == "solo-half" and phone["name"] == "pixel":
                raise RuntimeError("boom")
            return {"calls_sent": 3, "arm": arm, "phone": phone["name"]}

        def pair(config, arm, directory):
            return {"phones": {p["name"]: {"calls_sent": 3} for p in config["phones"]}}

        config = base_config()
        with tempfile.TemporaryDirectory() as directory:
            results = psm.run(config, Path(directory) / "out", sampler_factory=Sampler, solo_runner=solo,
                              pair_runner=pair, sleep=lambda s: None)
            names = [(r["name"], r["status"]) for r in results]
            self.assertEqual(names, [("solo-full:op15", "PASS"), ("solo-full:pixel", "PASS"), ("solo-half:op15", "PASS"),
                                     ("solo-half:pixel", "FAILED"), ("pair-half", "PASS"), ("pair-full", "PASS")])
            self.assertTrue((Path(directory) / "out" / "HOST_SAMPLES.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
