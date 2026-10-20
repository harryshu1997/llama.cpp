"""Co-tenant leases: one server's requests share the lanes it multiplexes itself.

A remote-resident parent's server batches the rows of all its running requests into one
call per layer on the phone it drives, so its requests' leases on the phone lanes carry
one ``share_key`` and may overlap on a capacity-one lane; every other lease stays
exclusive with them (pB1q8oc 2026-10-10 ran its owner requests one at a time).
"""

from __future__ import annotations

from dataclasses import replace
import os
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from research_dev.scheduler._internal.policy_common import (  # noqa: E402
    LeaseDemand,
    ResourceProfile,
    SchedulerError,
)
from research_dev.scheduler._internal.resource_timeline import ResourceTimeline  # noqa: E402
from research_dev.scheduler._internal.runtime_queue import RuntimeDispatchQueue  # noqa: E402

KEY = "remote-resident-server:physical:hot:desktop:remote-owner@http://127.0.0.1:18484"


def timeline() -> ResourceTimeline:
    return ResourceTimeline({
        "phone": ResourceProfile("phone", "compute", 1, True, "phone"),
        "desktop": ResourceProfile("desktop", "compute", 4, True, "desktop"),
    })


def demand(lease_id: str, duration_us: int, share_key: str | None = None,
           resource_id: str = "phone") -> LeaseDemand:
    return LeaseDemand(lease_id=lease_id, resource_id=resource_id, slots=1, start_offset_us=0,
                       duration_us=duration_us, duration_upper_us=duration_us, share_key=share_key)


def commit(rows: ResourceTimeline, owner_id: str, arrival_us: int, duration_us: int,
           share_key: str | None = None):
    preview = rows.preview_leases((demand(owner_id, duration_us, share_key),), arrival_us,
                                  duration_us, duration_us)
    return preview, rows.commit_leases(preview, owner_id)


class CoTenantLeaseTests(unittest.TestCase):
    def test_same_key_leases_overlap_and_other_leases_wait(self) -> None:
        rows = timeline()
        first, leases = commit(rows, "a", 0, 100, KEY)
        self.assertEqual((first.start_us, leases[0].share_key), (0, KEY))
        joiner, _ = commit(rows, "b", 10, 20, KEY)
        self.assertEqual(joiner.start_us, 10)
        for key in (None, KEY + "-other"):
            with self.subTest(key=key):
                self.assertEqual(
                    rows.preview_leases((demand("c", 5, key),), 10, 5, 5).start_us, 100)

    def test_a_short_co_tenant_does_not_hide_a_long_one(self) -> None:
        # the joiner ends first; an exclusive lease at 40 must still see the leader until 100
        rows = timeline()
        commit(rows, "a", 0, 100, KEY)
        commit(rows, "b", 10, 20, KEY)
        self.assertEqual(rows.preview_leases((demand("c", 10),), 40, 10, 10).start_us, 100)
        with self.assertRaisesRegex(SchedulerError, "changed before commit"):
            stale = rows.preview_leases((demand("d", 10),), 100, 10, 10)
            rows.commit_leases(replace(stale, plans=tuple(
                replace(plan, start_us=40, reserved_until_us=50) for plan in stale.plans)), "d")

    def test_co_tenants_extend_over_each_other_but_not_over_other_work(self) -> None:
        rows = timeline()
        _, first = commit(rows, "a", 0, 100, KEY)
        _, second = commit(rows, "b", 10, 120, KEY)
        rows.extend(first[0].token, 140)
        rows.extend_many((second[0].token,), 150)
        commit(rows, "c", 150, 10)
        with self.assertRaisesRegex(SchedulerError, "overlaps committed work"):
            rows.extend(first[0].token, 155)

    def test_checkpoint_keeps_the_keys(self) -> None:
        rows = timeline()
        checkpoint = rows.checkpoint()
        commit(rows, "a", 0, 100, KEY)
        commit(rows, "b", 10, 20, KEY)
        state = rows.causal_state()
        rows.restore(checkpoint)
        self.assertEqual(rows.preview_leases((demand("c", 10),), 40, 10, 10).start_us, 40)
        commit(rows, "a", 0, 100, KEY)
        commit(rows, "b", 10, 20, KEY)
        self.assertEqual(rows.causal_state(), state)
        lanes = state["resources"]["phone"]["lanes"][0]
        self.assertEqual({row.get("share_key") for row in lanes}, {KEY})
        self.assertNotIn("share_key", state["resources"]["desktop"]["lanes"][0][0]
                         if state["resources"]["desktop"]["lanes"][0] else {})

    def test_the_dispatch_queue_lets_co_tenants_run_together(self) -> None:
        rows = timeline()
        decisions = []
        for owner_id, key in (("a", KEY), ("b", KEY), ("c", None)):
            preview, leases = commit(rows, owner_id, 0, 100, key)
            # the queue reads a decision's leases only (Decision carries them as committed)
            decisions.append(SimpleNamespace(request_id=owner_id, route_id="route-" + owner_id,
                                             leases=leases))
        a, b, c = decisions
        self.assertFalse(RuntimeDispatchQueue._decisions_conflict(a, b))
        self.assertTrue(RuntimeDispatchQueue._decisions_conflict(a, c))


class CoHelperCapacityTests(unittest.TestCase):
    """A decode route sized for a remote-resident owner's prefill (512 rows) keeps a decode
    co-helper at its own rows (the Pixel at 4) when they hold one decode step (target B preload,
    2026-10-10: every OnePlus+Pixel helper route was TRANSPORT_PROFILE_INCOMPLETE)."""

    @staticmethod
    def helpers(*rows: int):
        return SimpleNamespace(helpers=tuple(SimpleNamespace(max_tokens=row) for row in rows))

    def test_cases(self) -> None:
        from research_dev.scheduler._internal.route_generation.costing_parameters import co_helper_capacity
        cases = (
            # helpers, maximum, step, remote, decode -> (compatible, rebind)
            ((4,), 4, 4, False, True, (True, False)),      # plain decode route: equal rows
            ((4,), 512, 4, False, True, (True, True)),     # owner-sized decode route: capped co-helper
            ((2,), 512, 4, False, True, (False, False)),   # a co-helper below the decode step
            ((4,), 512, 512, False, False, (False, False)),  # prefill routes never cap
            ((4,), 512, 4, True, True, (True, False)),     # remote parent: its own binding caps rows
            ((512,), 512, 4, False, True, (True, False)),  # nothing below the maximum: no rebind
        )
        for rows, maximum, step, remote, decode, expected in cases:
            with self.subTest(rows=rows, maximum=maximum, step=step, remote=remote, decode=decode):
                self.assertEqual(co_helper_capacity(
                    self.helpers(*rows), maximum, step, remote_capped=remote, decode_phase=decode), expected)


if __name__ == "__main__":
    unittest.main()
