#!/usr/bin/env python3

import copy
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
S40 = HERE.parent
sys.path.insert(0, str(S40))
sys.path.insert(0, str(HERE))

from evidence_common import EvidenceError, canonical_bytes  # noqa: E402
from event_evidence import reduce_events  # noqa: E402
from fake_ledger import resources, successful_transition  # noqa: E402
from render_graphs import render, throughput_svg  # noqa: E402


class GraphTests(unittest.TestCase):
    def summary(self):
        events, expected = successful_transition()
        return reduce_events(
            events,
            expected,
            resources(events[0]["run_id"], events[-1]["t_ns"]),
        )

    def test_two_graphs_render_from_reduced_evidence(self):
        summary = self.summary()
        with tempfile.TemporaryDirectory(prefix="s40_graph_") as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            summary_path.write_bytes(canonical_bytes(summary))
            outputs = render(summary_path, root / "graphs")
            self.assertEqual(len(outputs), 2)
            for path in outputs:
                raw = path.read_bytes()
                self.assertTrue(raw.startswith(b"<svg "))
                self.assertGreater(len(raw), 1000)
            self.assertIn(
                b"Per-model throughput over time",
                outputs[0].read_bytes(),
            )
            self.assertIn(
                b"Not server-wall or total-system energy",
                outputs[1].read_bytes(),
            )

    def test_missing_timeline_fails_closed(self):
        summary = self.summary()
        summary.pop("timeline")
        with self.assertRaisesRegex(EvidenceError, "summary.timeline"):
            throughput_svg(summary)

    def test_energy_timeline_matches_integrated_total(self):
        summary = self.summary()
        self.assertEqual(
            summary["energy"]["timeline"][-1][
                "cumulative_gpu_energy_nj"
            ],
            summary["energy"]["gpu_energy_nj"],
        )

    def test_unknown_energy_scope_is_not_graphed(self):
        summary = self.summary()
        summary["energy"] = copy.deepcopy(summary["energy"])
        summary["energy"]["gpu_energy_scope"] = "SERVER_WALL"
        with tempfile.TemporaryDirectory(prefix="s40_graph_") as directory:
            path = Path(directory) / "summary.json"
            path.write_bytes(canonical_bytes(summary))
            with self.assertRaisesRegex(
                    EvidenceError, "selected-GPU samples required"):
                render(path, Path(directory) / "graphs")


if __name__ == "__main__":
    unittest.main()
