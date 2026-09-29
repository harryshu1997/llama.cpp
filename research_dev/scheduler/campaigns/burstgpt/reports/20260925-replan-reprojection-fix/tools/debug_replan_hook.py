"""Diagnostics for replans whose desktop baseline is memory-rejected (replay only, read-only).

install(UnifiedScheduler, say) wraps two scheduler methods in the replay process:
- _prepare_automated_replan: remembers the raw snapshot the replan started from;
- _replan_candidate_selection: when the baseline is in memory_rejections, prints the baseline's
  calendar/memory window, every ledger reservation overlapping it (owner, interval, whether the
  owner is a causal dependent of the replanned request) and whether re-projecting residency at the
  baseline's start from the raw snapshot changes the planning snapshot.
"""

from __future__ import annotations


def install(cls, say):
    original_prepare = cls._prepare_automated_replan
    original_selection = cls._replan_candidate_selection
    raw = {}

    def prepare(self, *, current, snapshot, **kwargs):
        raw[current.request.request_id] = snapshot
        return original_prepare(self, current=current, snapshot=snapshot, **kwargs)

    def selection(self, *, preparation, candidate_set, current, manifest, compiler,
                  observed_at_us, memory_rejections):
        baseline = candidate_set.baseline
        code = memory_rejections.get(baseline.candidate_id)
        if code is not None:
            rid = current.request.request_id
            plan_nb = self._runtime_plan_not_before_by_resource(baseline.plan)
            preview = self._preview_automated_resources(
                baseline,
                observed_at_us=self._causal_candidate_observed_at(
                    baseline, observed_at_us, plan_nb),
            )
            say("  DBG-REJECT", rid.split(":")[-1], "at", observed_at_us, "code", code,
                "baseline", baseline.candidate_id[-40:], "transitions", len(baseline.plan.transitions),
                "calendar", preview.start_us, preview.finish_us if hasattr(preview, "finish_us") else None,
                "memory_until", preview.finish_upper_us)
            queue = self._runtime_controller.queue
            ledger = self._runtime_memory.snapshot()["reservations"]
            owners = {}
            for row in ledger:
                if row["start_us"] < preview.finish_upper_us and preview.start_us < row["reserved_until_us"]:
                    owners.setdefault(row["owner_id"], []).append(row)
            for owner, rows in sorted(owners.items()):
                try:
                    dependent = queue._causally_depends_on(owner, rid)
                except KeyError:
                    dependent = None
                ticket = self.runtime_ticket(owner)
                say("     overlap", owner.split(":")[-1], ticket.dispatch_state, "dependent", dependent,
                    "start", min(r["start_us"] for r in rows),
                    "until", max(r["reserved_until_us"] for r in rows),
                    "replacement", sorted({(r["resource_id"], r.get("replaced_bytes", 0)) for r in rows
                                           if r.get("replacement_group")}),
                    "transitions", len(ticket.execution_plan.transitions) if ticket.execution_plan else None)
                try:
                    alone = self._runtime_memory.preview(
                        baseline.plan.memory_demands, preparation.context.snapshot.memory,
                        start_us=preview.start_us, reserved_until_us=preview.finish_upper_us,
                        transitions=baseline.plan.transitions,
                        residency=preparation.context.snapshot.residency,
                        exclusive_resource_by_device=self._runtime_exclusive_memory_resources(),
                        exclude_owner_id=owner,
                    )
                    say("       baseline admitted without this owner", bool(alone is not None))
                except Exception as exc:  # noqa: BLE001
                    say("       still rejected without this owner:", str(exc)[:120])
            source = raw.get(rid)
            if source is not None:
                try:
                    projected = self._automated_snapshot_for_request(
                        current.request, source, exclude_request_id=rid,
                        project_before_us=preview.start_us)
                    changed = (projected.residency != preparation.context.snapshot.residency
                               or projected.memory != preparation.context.snapshot.memory)
                    say("     reprojection at baseline start changes snapshot:", changed)
                except Exception as exc:  # noqa: BLE001
                    say("     reprojection failed:", type(exc).__name__, str(exc)[:160])
        return original_selection(
            self, preparation=preparation, candidate_set=candidate_set, current=current,
            manifest=manifest, compiler=compiler, observed_at_us=observed_at_us,
            memory_rejections=memory_rejections)

    cls._prepare_automated_replan = prepare
    cls._replan_candidate_selection = selection


def install_fix_probe(say):
    """Log every outcome of the fixed tree's _clear_rejected_replan_baseline (no-op on base)."""
    from research_dev.scheduler._unified.automated_requests_ops import replan_commit
    helper = getattr(replan_commit, "_clear_rejected_replan_baseline", None)
    if helper is None:
        return

    def probe(controller, *, preparation, current, candidate_set, memory_rejections, **kwargs):
        rid = current.request.request_id
        code = memory_rejections.get(candidate_set.baseline.candidate_id)
        runtime = controller._runtime_controller
        before = set(runtime.queued_causal_dependents(rid, 2**62))
        snapshot = preparation.context.snapshot
        result = helper(controller, preparation=preparation, current=current,
                        candidate_set=candidate_set, memory_rejections=memory_rejections, **kwargs)
        if code not in (None, "RESOURCE_CALENDAR_CURRENT"):
            after = set(runtime.queued_causal_dependents(rid, 2**62))
            say("  FIX", rid.split(":")[-1], "code", code, "retry", result,
                "deferred", sorted(x.split(":")[-1] for x in before - after),
                "reprojected", preparation.context.snapshot is not snapshot)
        return result

    replan_commit._clear_rejected_replan_baseline = probe
