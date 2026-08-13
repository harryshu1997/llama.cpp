from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import unittest


NATIVE_DIR = Path(__file__).resolve().parents[1] / "native"
sys.path.insert(0, str(NATIVE_DIR))

from analyze_phone_arbiter_probe import analyze  # noqa: E402


class PhoneArbiterProbeAnalysisTests(unittest.TestCase):
    def test_valid_probe_binds_all_three_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe = root / "probe.log"
            bridge = root / "bridge.log"
            router = root / "router.log"
            session = root / "session.log"
            workers = root / "workers.log"
            probe.write_text(
                'PHONEARBITERPROBE {"status":"PASS",'
                '"protected_groups":2,"protected_group_calls":24,'
                '"filler_calls":1,"configured_gap_us":350000,'
                '"samples":[{"tokens":16,"observed_gap_us":350100,'
                '"protected_group_rpc_us":1000,'
                '"filler_client_rpc_us":2000}],'
                '"validation_group_rpc_us":1000}\n',
                encoding="ascii",
            )
            bridge.write_text(
                'PHONEARBITERSHAPE {"status":"PASS",'
                '"energy_claim_eligible":false,"tokens":16,"calls":1,'
                '"sandwich_p50_us":2000,"sandwich_max_us":2000}\n'
                'PHONEARBITER {"status":"MECHANICS_ONLY",'
                '"energy_claim_eligible":false,"protected_calls":24,'
                '"filler_calls":1,"filler_admitted":1,'
                '"filler_before_protected_done":1,'
                '"filler_after_protected_done":0,'
                '"protected_group_starts":2,"protected_group_ends":2,'
                '"idle_samples":1,"idle_lower_us":330000,'
                '"filler_upper_us":250000,"guard_us":20000,'
                '"observed_idle_min_us":350100,'
                '"filler_sandwich_max_us":2000,"reset_recoveries":0,'
                '"filler_upper_violations":0,"guard_violations":0,'
                '"idle_lower_violations":0,'
                '"protected_pending_after_filler":0}\n',
                encoding="ascii",
            )
            router.write_text(
                'RESIDENTROUTER {"status":"ok","sessions":5,'
                '"requests":25,"terminate_requested":true}\n',
                encoding="ascii",
            )
            session.write_text(
                "[resident-session] HTP0+HTP1+HTP2 WARM "
                "mem_available_kib=2200000\n",
                encoding="ascii",
            )
            workers.write_text(
                "HTP0 op batching: n-bufs 16 vmem 3422552064\n"
                "HTP1 op batching: n-bufs 16 vmem 3422552064\n"
                "HTP2 op batching: n-bufs 16 vmem 3422552064\n",
                encoding="ascii",
            )
            result = analyze(Namespace(
                bridge_log=bridge,
                expected_filler_upper_us=250000,
                expected_gap_us=350000,
                expected_guard_us=20000,
                expected_idle_lower_us=330000,
                expected_tokens=(16,),
                expected_vmem_mib=3264,
                minimum_available_kib=2097152,
                probe_log=probe,
                router_log=router,
                session_log=session,
                workers_log=workers,
            ))
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["rows"][0]["tokens"], 16)


if __name__ == "__main__":
    unittest.main()
