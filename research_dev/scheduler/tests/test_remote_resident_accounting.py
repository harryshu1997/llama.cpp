"""Remote-resident accounting: credit after proof, recovery before admission, no double count."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler._internal.runtime_resources import (  # noqa: E402
    RuntimeRemoteResidentOmissionProof,
    RuntimeResourceError,
    remote_resident_accounting,
)

GiB = 1024**3
ARTIFACT = "sha256:" + "a" * 64
MASK = 0xFF
OMITTED = 2_831_155_200  # 8 layers x gate/up/down x 3840 x 15360 x f16


def _proof(**overrides) -> RuntimeRemoteResidentOmissionProof:
    values = {"layer_mask": MASK, "omitted_bytes": OMITTED, "unmapped_bytes": OMITTED - 4096,
              "warmup": "validated"}
    values.update(overrides)
    return RuntimeRemoteResidentOmissionProof(**values)


def _account(proof, *, fallback_mode="teardown", live_host=6 * GiB, required_host=None):
    return remote_resident_accounting(
        artifact_sha256=ARTIFACT,
        desktop_pool_id="host-ram",
        layer_mask=MASK,
        desktop_weights_full_bytes=9 * GiB,
        omitted_bytes_planned=OMITTED,
        proof=proof,
        phone_weights_bytes_by_session={"HTP0": OMITTED},
        phone_workspace_bytes=64 * 1024**2,
        kv_bytes_by_pool={"host-ram": 512 * 1024**2, "cuda0-vram": 1 * GiB},
        transition_peak_bytes_by_pool={"op15-ram": OMITTED + 64 * 1024**2},
        live_available_bytes_by_pool={"host-ram": live_host, "cuda0-vram": 2 * GiB},
        reduced_allocation_bytes_by_pool={"host-ram": 9 * GiB - OMITTED + 512 * 1024**2},
        recovery_required_bytes_by_pool={
            "host-ram": 9 * GiB + 512 * 1024**2 if required_host is None else required_host,
            "cuda0-vram": 1 * GiB,
        },
        fallback_mode=fallback_mode,
    )


class OmissionProofTests(unittest.TestCase):
    def test_parses_the_server_line_and_ignores_other_lines(self) -> None:
        line = (f"S41SERVERFFN remote_resident mask={MASK} omitted_bytes={OMITTED} "
                f"unmapped_bytes={OMITTED - 4096} warmup=validated")
        proof = RuntimeRemoteResidentOmissionProof.parse_line(line)
        self.assertEqual(proof, _proof())
        for prefix in ("\x1b[0m", "\x1b[31m\x1b[0m"):
            self.assertEqual(RuntimeRemoteResidentOmissionProof.parse_line(prefix + line), proof)
        self.assertIsNone(RuntimeRemoteResidentOmissionProof.parse_line("unrelated: " + line))
        self.assertIsNone(RuntimeRemoteResidentOmissionProof.parse_line("\x1b[2K" + line))
        self.assertIsNone(RuntimeRemoteResidentOmissionProof.parse_line("S41SERVERFFN ready host=usb"))
        self.assertEqual(RuntimeRemoteResidentOmissionProof.from_json(proof.to_json()), proof)
        with self.assertRaises(RuntimeResourceError):
            RuntimeRemoteResidentOmissionProof.parse_line("S41SERVERFFN remote_resident mask=1")
        with self.assertRaises(RuntimeResourceError):
            _proof(unmapped_bytes=OMITTED + 1)
        with self.assertRaises(RuntimeResourceError):
            _proof(warmup="maybe")


class AccountingTests(unittest.TestCase):
    def test_credit_only_after_a_matching_validated_proof(self) -> None:
        unverified = _account(None)
        self.assertFalse(unverified.verified)
        self.assertEqual(unverified.reclaimed_bytes, 0)
        self.assertEqual(unverified.kv_capacity_gain_bytes, 0)
        self.assertEqual(unverified.desktop_weights_allocated_bytes, 9 * GiB,
                         "an unverified omission is charged as full weights")
        for bad in (_proof(layer_mask=0x0F), _proof(omitted_bytes=OMITTED - 1), _proof(warmup="skipped")):
            with self.subTest(proof=bad.to_json()):
                self.assertFalse(_account(bad).verified)
        verified = _account(_proof())
        self.assertTrue(verified.verified)
        self.assertEqual(verified.omitted_bytes_verified, OMITTED)
        self.assertEqual(verified.desktop_weights_allocated_bytes, 9 * GiB - OMITTED)
        self.assertEqual(verified.reclaimed_bytes, OMITTED)

    def test_teardown_fallback_keeps_no_reserve_and_credits_all_reclaimed_to_kv(self) -> None:
        account = _account(_proof())
        self.assertEqual(account.fallback_mode, "teardown")
        self.assertEqual(account.fallback_reserve_bytes, 0)
        self.assertEqual(account.kv_capacity_gain_bytes, OMITTED)
        self.assertEqual(
            account.recovery_capacity_bytes_by_pool["host-ram"],
            6 * GiB + 9 * GiB - OMITTED + 512 * 1024**2,
        )
        self.assertTrue(account.recovery_feasible)

    def test_alongside_fallback_splits_reclaimed_between_reserve_and_kv(self) -> None:
        # live 6 GiB includes the reclaimed bytes; the full route needs 9.5 GiB next to us
        account = _account(_proof(), fallback_mode="alongside", live_host=6 * GiB,
                           required_host=6 * GiB - OMITTED // 2)
        self.assertEqual(account.fallback_reserve_bytes, OMITTED // 2)
        self.assertEqual(account.kv_capacity_gain_bytes, OMITTED - OMITTED // 2)
        self.assertEqual(account.fallback_reserve_bytes + account.kv_capacity_gain_bytes, OMITTED)
        self.assertTrue(account.recovery_feasible)
        infeasible = _account(_proof(), fallback_mode="alongside", live_host=6 * GiB,
                              required_host=7 * GiB)
        self.assertFalse(infeasible.recovery_feasible)
        self.assertEqual(infeasible.fallback_reserve_bytes, OMITTED,
                         "everything reclaimed is pledged when the fallback does not fit")
        self.assertEqual(infeasible.kv_capacity_gain_bytes, 0)

    def test_recovery_infeasible_when_capacity_is_short(self) -> None:
        account = _account(_proof(), live_host=0, required_host=20 * GiB)
        self.assertFalse(account.recovery_feasible)
        self.assertTrue(account.verified)

    def test_json_and_validation(self) -> None:
        payload = _account(_proof()).to_json()
        self.assertEqual(payload["schema"], "research-scheduler-remote-resident-accounting-v1")
        self.assertEqual(payload["proof"]["schema"], "scheduler-remote-resident-omission-v1")
        self.assertEqual(payload["phone_weights_bytes_by_session"], {"HTP0": OMITTED})
        with self.assertRaises(RuntimeResourceError):
            remote_resident_accounting(
                artifact_sha256=ARTIFACT, desktop_pool_id="host-ram", layer_mask=MASK,
                desktop_weights_full_bytes=1, omitted_bytes_planned=2, proof=None,
                phone_weights_bytes_by_session={"HTP0": 1}, phone_workspace_bytes=0,
                kv_bytes_by_pool={}, transition_peak_bytes_by_pool={},
                live_available_bytes_by_pool={"host-ram": 1},
                reduced_allocation_bytes_by_pool={}, recovery_required_bytes_by_pool={"host-ram": 1},
            )
        with self.assertRaises(RuntimeResourceError):
            _account(_proof(), fallback_mode="hope")


if __name__ == "__main__":
    unittest.main()
