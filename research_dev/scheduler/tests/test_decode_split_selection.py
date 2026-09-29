"""Measured decode-split selection and decode-phase memory accounting (mechanics, no devices)."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from research_dev.scheduler._internal.capacity import DeviceMemoryCapacity
from research_dev.scheduler._internal.decode_split_selection import (
    DecodeReleaseAccountant, DecodeSplitAtlas, DecodeSplitEnvironment, DecodeSplitSelectionError, ShareBinding,
    adaptive_policy_for_selection, select_decode_split,
)
from research_dev.scheduler._internal.runtime_cost import RuntimeMemoryDemand
from research_dev.scheduler._internal.runtime_placement import RuntimePlacementSnapshot
from research_dev.scheduler._internal.runtime_resources import RuntimeHostShareReleaseProof, RuntimeMemoryLedger, RuntimeResourceError

ATLAS = Path(__file__).resolve().parents[1] / "campaigns" / "burstgpt" / "data" / "QWEN_DECODE_SPLIT_ATLAS.json"
QWEN = "sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718"
N_FF = 17408
GIB = 1024 ** 3
MASK = (1 << 18) - 1


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atlas = DecodeSplitAtlas.load(ATLAS)
        cls.environment = cls.atlas.rows[0].environment

    def select(self, **kwargs):
        args = dict(environment=self.environment, prompt_tokens=1158, output_tokens=96, feed_forward_length=N_FF, objective="latency")
        args.update(kwargs)
        return select_decode_split(self.atlas, **args)

    def test_atlas_is_measured_digest_pinned_and_environment_bound(self):
        self.assertEqual(len(self.atlas.rows), 12)
        self.assertTrue(all(row.result_sha256.startswith("sha256:") for row in self.atlas.rows))
        self.assertEqual({row.environment for row in self.atlas.rows}, {self.environment})
        self.assertEqual(self.environment.artifact_sha256, QWEN)
        self.assertEqual(dict(self.environment.session_masks), {"HTP0": 63, "HTP1": 4032, "HTP2": 258048})
        env_json = json.loads(json.dumps(self.environment.to_json()))
        self.assertEqual(DecodeSplitEnvironment.from_json(env_json), self.environment)

    def test_latency_picks_the_fastest_measured_split_in_range(self):
        short = self.select(objective="latency")
        self.assertEqual((short.split_fraction_ppm, short.host_columns, short.phone_columns), (500000, 8704, 8704))
        self.assertTrue(short.dormant_release)
        self.assertEqual(short.expected_release_bytes, 4435329024)
        self.assertEqual(short.validated_prompt_tokens, (868, 1447))
        long = self.select(objective="latency", prompt_tokens=9737, output_tokens=1024)
        self.assertEqual(long.split_fraction_ppm, 500000)
        self.assertAlmostEqual(long.decode_ms_per_token, 650.5, delta=0.1)
        self.assertEqual(long.source_run, "pair-v1/combined")

    def test_energy_picks_the_whole_phone_ffn(self):
        for prompt, output in ((1158, 96), (9737, 1024)):
            choice = self.select(objective="energy", prompt_tokens=prompt, output_tokens=output)
            self.assertEqual((choice.split_fraction_ppm, choice.host_columns, choice.phone_columns), (1000000, 0, N_FF))
            self.assertEqual(choice.expected_release_bytes, 9625706496)
        long = self.select(objective="energy", prompt_tokens=9737, output_tokens=1024)
        baseline = [row for row in self.atlas.rows if row.run == "pair-v1/control"][0]
        self.assertLess(long.energy_per_token_j, baseline.energy_per_token_j(4.5))
        self.assertEqual(long.source_run, "combined-h0-v1")

    def test_memory_objective_honours_required_release_and_latency_bound(self):
        biggest = self.select(objective="memory")
        self.assertEqual(biggest.split_fraction_ppm, 1000000)
        bounded = self.select(objective="memory", required_release_bytes=5 * GIB, max_decode_ms_per_token=530.0)
        self.assertEqual((bounded.split_fraction_ppm, bounded.host_columns), (750000, 4352))
        with self.assertRaises(DecodeSplitSelectionError):
            self.select(objective="memory", required_release_bytes=10 * GIB)
        with self.assertRaises(DecodeSplitSelectionError):
            self.select(objective="latency", max_decode_ms_per_token=100.0)

    def test_no_extrapolation_outside_validated_request_shapes(self):
        # a million-token prompt, a 5,000-token prompt in the unmeasured gap, and a generation longer than measured
        for prompt, output in ((1_000_000, 96), (5000, 96), (1158, 200), (9737, 2048), (16000, 64)):
            with self.assertRaises(DecodeSplitSelectionError, msg=(prompt, output)):
                self.select(prompt_tokens=prompt, output_tokens=output, max_decode_ms_per_token=700.0)
        with self.assertRaises(DecodeSplitSelectionError):
            self.select(prompt_tokens=32000, output_tokens=1024)  # does not fit the configured context

    def test_environment_must_match_exactly(self):
        for changed in (dict(gpu_layers=15), dict(context_cells=16384), dict(ubatch=256), dict(column_quantum=1088),
                        dict(runtime_sha256="sha256:" + "0" * 64), dict(kv_plan_sha256="sha256:" + "1" * 64),
                        dict(session_masks=(("HTP0", 63), ("HTP1", 4032))), dict(artifact_sha256="sha256:" + "2" * 64)):
            with self.assertRaises(DecodeSplitSelectionError, msg=changed):
                self.select(environment=replace(self.environment, **changed))
        with self.assertRaises(DecodeSplitSelectionError):
            self.select(objective="throughput")
        with self.assertRaises(DecodeSplitSelectionError):
            self.select(feed_forward_length=4096)

    def test_phone_unavailable_yields_the_baseline(self):
        choice = self.select(objective="energy", phone_available=False)
        self.assertEqual((choice.split_fraction_ppm, choice.host_columns, choice.expected_release_bytes), (0, N_FF, 0))
        self.assertFalse(choice.dormant_release)
        with self.assertRaises(DecodeSplitSelectionError):
            adaptive_policy_for_selection(choice, route_id="r", executor_id="e", operator_plan_sha256="sha256:" + "1" * 64,
                                          desktop_parent_route_id="p", desktop_placement_sha256="sha256:" + "2" * 64,
                                          layer_indices=range(18), layer_mask=MASK, resource_ids=("desktop-cpu",))

    def test_capped_rows_are_never_profiles(self):
        choice = self.select(objective="energy", prompt_tokens=9737, output_tokens=1024)
        self.assertNotIn("capped", choice.source_run)

    def test_policy_carries_the_selected_columns(self):
        choice = self.select(objective="latency")
        policy = adaptive_policy_for_selection(choice, route_id="decode-relocation", executor_id="native-calibration",
                                               operator_plan_sha256="sha256:" + "1" * 64, desktop_parent_route_id="cuda-parent",
                                               desktop_placement_sha256="sha256:" + "2" * 64, layer_indices=range(18),
                                               layer_mask=MASK, resource_ids=("desktop-cpu", "op15-htp"))
        self.assertEqual((policy.columns, policy.split_fraction_ppm, policy.layer_mask), (8704, 500000, MASK))


class AccountantTests(unittest.TestCase):
    POOL = "desktop-host"
    ENDPOINT = "http://127.0.0.1:41001"

    def setUp(self):
        self.ledger = RuntimeMemoryLedger()
        self.snapshot = RuntimePlacementSnapshot("snap", 0, 10_000_000_000,
                                                 {self.POOL: DeviceMemoryCapacity(self.POOL, 20 * GIB, 0, 0)})
        self.accountant = DecodeReleaseAccountant(self.ledger, host_pool=self.POOL)
        # the server's weights and KV, as reserve_layer_kv would charge them (minus the share)
        self.ledger.reserve("server-a", (RuntimeMemoryDemand("server-a:weights", self.POOL, "weights", 10 * GIB, 0, "request"),), self.snapshot)
        self.binding = ShareBinding(self.ENDPOINT, QWEN, MASK, 8704, 4 * GIB, column_quantum=2176)

    def proof(self, released, *, layer_mask=MASK, host_columns=8704, phase="decode"):
        return RuntimeHostShareReleaseProof(phase=phase, layer_mask=layer_mask, host_columns=host_columns,
                                            released_bytes=released, ranges=92196, elapsed_us=114052)

    def reserved(self):
        return self.ledger.snapshot()["by_resource_bytes"].get(self.POOL, 0)

    def test_release_credit_is_decode_phase_only_and_gates_the_next_prompt(self):
        self.accountant.reserve_share("server-a", self.binding, self.snapshot)
        self.assertEqual(self.reserved(), 14 * GIB)
        self.assertEqual(self.accountant.state("server-a"), "prefill-resident")
        with self.assertRaises(RuntimeResourceError):  # 6 GiB free < 7 GiB
            self.accountant.reserve_decode_growth("tenant", 7 * GIB, self.snapshot)
        credited = self.accountant.enter_decode("server-a", self.proof(4 * GIB), self.snapshot, endpoint=self.ENDPOINT, release_generation=1)
        self.assertEqual((credited, self.reserved()), (4 * GIB, 10 * GIB))
        self.assertEqual(self.accountant.decode_phase_headroom_bytes(self.snapshot), 10 * GIB)
        self.assertEqual(self.accountant.kv_growth_tokens(self.snapshot, 131072), 81920)
        self.accountant.reserve_decode_growth("tenant", 7 * GIB, self.snapshot)  # the freed room is usable while decoding
        self.assertFalse(self.accountant.preview_prompt_admission("server-a", self.snapshot))
        with self.assertRaises(RuntimeResourceError):
            self.accountant.restore_before_prompt("server-a", self.snapshot)
        self.assertEqual((self.accountant.state("server-a"), self.reserved()), ("restore-blocked", 17 * GIB))
        self.accountant.release_growth("tenant")
        self.assertTrue(self.accountant.preview_prompt_admission("server-a", self.snapshot))
        self.accountant.restore_before_prompt("server-a", self.snapshot)
        self.assertEqual((self.accountant.state("server-a"), self.reserved()), ("prefill-resident", 14 * GIB))
        self.accountant.forget("server-a")
        self.assertEqual(self.reserved(), 10 * GIB)

    def test_failed_restoration_preserves_the_retained_shortfall(self):
        """Bug 1: a 4 GiB reservation, 3 GiB physically released -> 1 GiB stays charged; a failed restore must
        leave that charge (and everything else) exactly as it was."""
        self.accountant.reserve_share("server-a", self.binding, self.snapshot)
        self.assertEqual(self.accountant.enter_decode("server-a", self.proof(3 * GIB), self.snapshot, endpoint=self.ENDPOINT, release_generation=1), 3 * GIB)
        self.assertEqual(self.reserved(), 11 * GIB)
        self.accountant.reserve_decode_growth("tenant", 8 * GIB, self.snapshot)  # 19 GiB reserved, 1 GiB free < 3 GiB missing
        before = self.ledger.snapshot()
        self.assertEqual(self.reserved(), 19 * GIB)
        self.assertFalse(self.accountant.preview_prompt_admission("server-a", self.snapshot))
        with self.assertRaises(RuntimeResourceError):
            self.accountant.restore_before_prompt("server-a", self.snapshot)
        after = self.ledger.snapshot()
        self.assertEqual(self.reserved(), 19 * GIB)
        self.assertEqual(before["reservations"], after["reservations"])
        self.assertIn("dormant-host-share-shortfall", {row["kind"] for row in after["reservations"]})
        self.accountant.release_growth("tenant")
        self.accountant.restore_before_prompt("server-a", self.snapshot)
        self.assertEqual(self.reserved(), 14 * GIB)
        self.assertEqual({row["kind"] for row in self.ledger.snapshot()["reservations"] if row["owner_id"].endswith("dormant-host-share")},
                         {"dormant-host-share"})

    def test_proofs_are_bound_to_their_allocation_and_consumed_once(self):
        """Bug 2: the same proof must not credit two owners, a proof from another endpoint or for other
        layers/columns must be refused, and a release event credits once."""
        self.accountant.reserve_share("server-a", self.binding, self.snapshot)
        other = ShareBinding("http://127.0.0.1:41002", QWEN, MASK, 8704, 4 * GIB, column_quantum=2176)
        self.accountant.reserve_share("server-b", other, self.snapshot)
        self.assertEqual(self.reserved(), 18 * GIB)
        proof = self.proof(4 * GIB)
        self.accountant.enter_decode("server-a", proof, self.snapshot, endpoint=self.ENDPOINT, release_generation=1)
        for kwargs in (dict(endpoint=self.ENDPOINT, release_generation=1),      # same event replayed
                       dict(endpoint=self.ENDPOINT, release_generation=2)):     # server-a's endpoint credited to server-b
            with self.assertRaises(DecodeSplitSelectionError, msg=kwargs):
                self.accountant.enter_decode("server-b", proof, self.snapshot, **kwargs)
        self.assertEqual((self.reserved(), self.accountant.state("server-b")), (14 * GIB, "prefill-resident"))
        with self.assertRaises(DecodeSplitSelectionError):  # layers outside the booking
            self.accountant.enter_decode("server-b", self.proof(4 * GIB, layer_mask=MASK | (1 << 40)), self.snapshot, endpoint=other.endpoint, release_generation=1)
        with self.assertRaises(DecodeSplitSelectionError):  # a larger release than booked
            self.accountant.enter_decode("server-b", self.proof(4 * GIB, host_columns=4352), self.snapshot, endpoint=other.endpoint, release_generation=1)
        with self.assertRaises(DecodeSplitSelectionError):  # not quantum aligned
            self.accountant.enter_decode("server-b", self.proof(4 * GIB, host_columns=8705), self.snapshot, endpoint=other.endpoint, release_generation=1)
        with self.assertRaises(DecodeSplitSelectionError):
            self.accountant.enter_decode("server-b", self.proof(4 * GIB, phase="local"), self.snapshot, endpoint=other.endpoint, release_generation=1)
        self.assertEqual(self.reserved(), 14 * GIB)
        self.accountant.enter_decode("server-b", self.proof(4 * GIB), self.snapshot, endpoint=other.endpoint, release_generation=1)
        self.assertEqual(self.reserved(), 10 * GIB)
        # after a restore the next release is a new generation; the old one stays consumed
        self.accountant.restore_before_prompt("server-a", self.snapshot)
        with self.assertRaises(DecodeSplitSelectionError):
            self.accountant.enter_decode("server-a", proof, self.snapshot, endpoint=self.ENDPOINT, release_generation=1)
        self.accountant.enter_decode("server-a", proof, self.snapshot, endpoint=self.ENDPOINT, release_generation=2)
        self.assertEqual(self.reserved(), 10 * GIB)

    def test_smaller_release_than_booked_keeps_the_remainder_charged(self):
        booked = ShareBinding(self.ENDPOINT, QWEN, MASK, 0, 8 * GIB, column_quantum=2176)  # up to the whole FFN
        self.accountant.reserve_share("server-a", booked, self.snapshot)
        self.assertEqual(self.reserved(), 18 * GIB)
        credited = self.accountant.enter_decode("server-a", self.proof(4 * GIB, host_columns=8704), self.snapshot,
                                                endpoint=self.ENDPOINT, release_generation=1)
        self.assertEqual((credited, self.reserved()), (4 * GIB, 14 * GIB))

    def test_re_release_while_released_resizes_the_shortfall(self):
        """The controller changed the fraction mid-request: generation 2 releases more while generation 1 is still
        credited; the ledger moves to the new shortfall without a prompt in between."""
        booked = ShareBinding(self.ENDPOINT, QWEN, MASK, 0, 8 * GIB, column_quantum=2176)
        self.accountant.reserve_share("server-a", booked, self.snapshot)
        self.assertEqual(self.accountant.enter_decode("server-a", self.proof(4 * GIB, host_columns=8704), self.snapshot,
                                                      endpoint=self.ENDPOINT, release_generation=1), 4 * GIB)
        self.assertEqual(self.reserved(), 14 * GIB)
        self.assertEqual(self.accountant.enter_decode("server-a", self.proof(8 * GIB, host_columns=0), self.snapshot,
                                                      endpoint=self.ENDPOINT, release_generation=2), 8 * GIB)
        self.assertEqual((self.reserved(), self.accountant.state("server-a")), (10 * GIB, "decode-released"))
        # a smaller re-release grows the shortfall back; when a tenant took the room it is refused and nothing changes
        self.accountant.reserve_decode_growth("tenant", 10 * GIB, self.snapshot)
        with self.assertRaises(RuntimeResourceError):
            self.accountant.enter_decode("server-a", self.proof(2 * GIB, host_columns=13056), self.snapshot,
                                         endpoint=self.ENDPOINT, release_generation=3)
        self.assertEqual(self.reserved(), 20 * GIB)
        self.accountant.release_growth("tenant")
        self.assertEqual(self.accountant.enter_decode("server-a", self.proof(2 * GIB, host_columns=13056), self.snapshot,
                                                      endpoint=self.ENDPOINT, release_generation=3), 2 * GIB)
        self.assertEqual(self.reserved(), 16 * GIB)

    def test_one_share_per_endpoint_and_admission_by_reservation(self):
        self.accountant.reserve_share("server-a", self.binding, self.snapshot)
        with self.assertRaises(DecodeSplitSelectionError):
            self.accountant.reserve_share("server-a-again", replace(self.binding, expected_release_bytes=GIB), self.snapshot)
        with self.assertRaises(DecodeSplitSelectionError):
            self.accountant.reserve_share("server-a", self.binding, self.snapshot)
        with self.assertRaises(RuntimeResourceError):  # 6 GiB free: a 12 GiB share cannot be resident for prefill
            self.accountant.reserve_share("server-c", ShareBinding("http://127.0.0.1:41003", QWEN, MASK, 0, 12 * GIB), self.snapshot)
        self.assertIsNone(self.accountant.state("server-c"))
        self.assertEqual(self.reserved(), 14 * GIB)


if __name__ == "__main__":
    unittest.main()
