"""Joint join: co-tenants run the host policy while a planner-joined request prefills.

Opt-in through ``dispatch_policy.joint_planner = {"mode": "active", ...}``; without a registration
nothing here runs (the registry stays empty and every check returns at its first line).

Why the holder must yield. llama-server batches two slots only when their FFN split policies are
equal (``server_slot::can_batch_with``), a new slot starts with the host policy, and an FFN control is
refused until the slot decodes. The batch anchor rotates among decoding slots only, and prompt
processing is skipped for every slot that cannot batch with the anchor. So while a co-tenant decodes
under a phone policy, a same-model joiner's prompt is never processed: it waits for the co-tenant to
finish (hardware s2a: Qwen 003 first token at 445.8 s = 001's end, 004 at 576.1 s = 003's end, Gemma
010 at 1503.0 s = 009's end), although the scheduler admitted it at once and phone rows are nearly free.

A registered prefill yield makes every co-tenant of the joiner's model and desktop parent run the
host policy until the joiner's decode session starts (its first token), the joiner finishes, the
registration is cleared, or it expires. The co-tenants' server policy group keeps its verdicts and
proposal (a yield window is a host window of an incomplete batch composition, so it is never
comparable, qualifies nothing and spends no probe budget). At its first token the joiner follows the
group's phone policy through the ordinary coherence path, the co-tenants return to it at their next
boundary, the slots batch into one forward and the phone call carries both rows; the joiner's helper
window shares the holder's lease tokens (``SERVER_HELPER_LEASES_SHARED``). Every phone control after a
yield is an ordinary control: helper readiness, identity, lease and thermal checks apply unchanged.

The registry is a physical fact about the server (which slot is in its prompt phase), kept outside
checkpoints like device membership: a rollback cannot un-start a prompt.
"""
from __future__ import annotations

from typing import Mapping

PREFILL_YIELD_REASON = "JOINT_PLANNER_PREFILL_YIELD"


def _started(controller, request_id: str) -> bool:
    return (request_id in controller._sessions or request_id in controller._sealed_sessions
            or request_id in controller._completed)


def _prefix(session) -> tuple[str, str]:
    return session.model_artifact_sha256, session.baseline.desktop_placement_sha256


def _event(controller, kind: str, joiner: str, at_us: int, **detail: object) -> None:
    controller._prefill_yield_events.append(
        {"kind": kind, "joiner_request_id": joiner, "observed_at_us": int(at_us), **detail})


def register(controller, joiner: str, *, model_artifact_sha256: str, desktop_placement_sha256: str,
             at_us: int, expires_at_us: int) -> tuple[str, tuple[str, ...]]:
    """(outcome, co-tenant request ids). Outcomes: REGISTERED, JOINER_ALREADY_DECODING,
    NO_CO_TENANT_SESSION (no adaptive session of that model and parent: nothing can block the prompt)."""
    if _started(controller, joiner):
        return "JOINER_ALREADY_DECODING", ()
    prefix = (model_artifact_sha256, desktop_placement_sha256)
    co_tenants = tuple(sorted(
        request_id for request_id, session in controller._sessions.items() if _prefix(session) == prefix))
    if not co_tenants:
        return "NO_CO_TENANT_SESSION", ()
    phone = tuple(
        request_id for request_id in co_tenants
        if not _running_policy_is_baseline(controller._sessions[request_id]))
    controller._prefill_yields[joiner] = {
        "prefix": prefix, "registered_at_us": int(at_us), "expires_at_us": int(expires_at_us),
        "applied_by": [],
    }
    _event(controller, "REGISTERED", joiner, at_us, co_tenant_request_ids=list(co_tenants),
           phone_policy_request_ids=list(phone), expires_at_us=int(expires_at_us))
    return "REGISTERED", co_tenants


def _running_policy_is_baseline(session) -> bool:
    policy = (session.awaiting_control.policy if session.awaiting_control is not None
              else session.current_policy)
    return policy is None or policy.baseline


def end(controller, joiner: str, at_us: int, reason: str) -> bool:
    row = controller._prefill_yields.pop(joiner, None)
    if row is None:
        return False
    _event(controller, "ENDED", joiner, at_us, reason=reason,
           yielded_request_ids=list(row["applied_by"]),
           held_us=max(0, int(at_us) - row["registered_at_us"]))
    return True


def pending(controller, session, at_us: int | None) -> str | None:
    """The joiner this session yields to, or None. Ends registrations whose joiner started (or
    finished) and expired ones (``at_us`` None: no expiry check)."""
    if not getattr(controller, "_prefill_yields", None):
        return None
    prefix = _prefix(session)
    for joiner in sorted(controller._prefill_yields):
        row = controller._prefill_yields[joiner]
        if row["prefix"] != prefix or joiner == session.request_id:
            continue
        if _started(controller, joiner):
            end(controller, joiner, row["registered_at_us"] if at_us is None else at_us, "JOINER_STARTED")
            continue
        if at_us is not None and at_us >= row["expires_at_us"]:
            end(controller, joiner, at_us, "EXPIRED")
            continue
        if session.request_id not in row["applied_by"]:
            row["applied_by"].append(session.request_id)
            _event(controller, "APPLIED", joiner, row["registered_at_us"] if at_us is None else at_us,
                   yielding_request_id=session.request_id)
        return joiner
    return None


def closes_window(controller, session, at_us: int) -> bool:
    """Whether the current window closes at this token (as a follower's window closes when the server
    policy changes): a phone window while a joined co-tenant prefills, so the host lands within one
    step; and the host window a yield opened, once the yield has ended, so the co-tenant returns to
    the group policy at once instead of alternating forwards with the joiner for a whole window.
    The second case consumes the yield marker, so it fires once per yield."""
    policy = session.current_policy
    if policy is None:
        return False
    if policy.baseline:
        if (session.zero_assistance_reason != PREFILL_YIELD_REASON
                or pending(controller, session, at_us) is not None):
            return False
        session.zero_assistance_reason = None
        return True
    return bool(getattr(controller, "_prefill_yields", None)) and pending(controller, session, at_us) is not None


def events(controller) -> tuple[Mapping[str, object], ...]:
    return tuple(dict(row) for row in controller._prefill_yield_events)
