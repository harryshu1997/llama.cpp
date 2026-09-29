#!/usr/bin/env python3
"""Verify one progressive session replacement gate from its RESULT.json.

The gate replaces exactly one dynamically selected phone session S from the
hot model at generation 1 to the cold model at generation 2 while the other
sessions keep serving the hot model. The same S must appear in the layout
plan, the physical reconfiguration command and receipt, and the published
READY layout. With ``--expect-fault-injection`` the run must also contain one
injected post-load failure whose physical and logical rollback restored the
source layout before the retry succeeded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


class GateFailure(Exception):
    pass


def check(condition: bool, message: str, checks: dict[str, bool]) -> None:
    checks[message] = bool(condition)


def _shards(event: dict) -> dict[str, dict]:
    return {row["session_id"]: row for row in event["layout"]["shards"]}


def _generations(event: dict) -> dict[str, int]:
    layout = event.get("layout") or {}
    if "session_generation_by_id" in layout:
        return {
            key: int(value)
            for key, value in layout["session_generation_by_id"].items()
        }
    return {
        row["session_id"]: int(row["session_generation"])
        for row in event.get("session_identities") or []
    }


def _model_artifacts(result: dict) -> tuple[str | None, str | None]:
    roles = result["model_roles"]
    artifacts = result.get("model_artifacts") or {}
    hot_artifact = (artifacts.get(roles["qwen"]) or {}).get("artifact_sha256")
    cold_artifact = (artifacts.get(roles["gemma"]) or {}).get("artifact_sha256")
    if hot_artifact is None or cold_artifact is None:
        # Fall back to the ticket bindings when the artifact map is absent.
        for row in result["request_results"]:
            artifact = row["terminal_ticket"]["model"]["artifact_sha256"]
            if row["model_id"] == roles["qwen"]:
                hot_artifact = artifact
            elif row["model_id"] == roles["gemma"]:
                cold_artifact = artifact
    return hot_artifact, cold_artifact


def _check_ready_layouts(
    checks: dict[str, bool],
    ready: list[dict],
    hot_artifact: str | None,
    cold_artifact: str | None,
    *,
    expect_fault: bool,
) -> tuple[str | None, dict[str, dict]]:
    first, last = ready[0], ready[-1]
    first_shards, last_shards = _shards(first), _shards(last)
    check(
        set(first_shards) == set(last_shards) and len(first_shards) == 3,
        "three sessions throughout",
        checks,
    )
    check(
        all(row["artifact_sha256"] == hot_artifact for row in first_shards.values()),
        "first READY layout is all hot model",
        checks,
    )
    changed = sorted(
        session_id for session_id, row in last_shards.items()
        if row["artifact_sha256"] != first_shards[session_id]["artifact_sha256"]
    )
    check(len(changed) == 1, "exactly one session changed", checks)
    selected = changed[0] if len(changed) == 1 else None
    plan_changed = list(last["layout"]["changed_session_ids"])
    check(
        selected is not None and plan_changed == [selected],
        "plan changed_session_ids names only S",
        checks,
    )
    check(
        selected is not None
        and last_shards[selected]["artifact_sha256"] == cold_artifact,
        "S holds the cold model in the published layout",
        checks,
    )
    first_generations, last_generations = _generations(first), _generations(last)
    final_session_generation = 4 if expect_fault else 2
    check(
        selected is not None
        and first_generations.get(selected) == 1
        and last_generations.get(selected) == final_session_generation,
        "S reaches the expected monotonic session generation",
        checks,
    )
    check(
        all(
            last_generations.get(session_id) == 1
            and last_shards[session_id]["artifact_sha256"] == hot_artifact
            and last_shards[session_id] == first_shards[session_id]
            for session_id in last_shards if session_id != selected
        ),
        "other sessions stay hot model generation 1 and identical",
        checks,
    )
    return selected, last_shards


def _check_reconfigurations(
    checks: dict[str, bool],
    reconfigurations: list[dict],
    selected: str | None,
    hot_artifact: str | None,
    cold_artifact: str | None,
    *,
    expect_fault: bool,
) -> None:
    check(
        bool(reconfigurations)
        and all(row["changed_session_id"] == selected for row in reconfigurations),
        "every physical reconfiguration receipt changes S",
        checks,
    )
    check(
        all(
            row.get("command_changed_session_ids") == [selected]
            for row in reconfigurations
        ),
        "every transition command changes only S",
        checks,
    )
    expected_sources = [1, 3] if expect_fault else [1]
    expected_targets = [2, 4] if expect_fault else [2]
    check(
        len(reconfigurations) == len(expected_targets)
        and all(
            row["target_shard"]["artifact_sha256"] == cold_artifact
            and row["previous_shard"]["artifact_sha256"] == hot_artifact
            and row["previous_shard"]["session_generation"] == source
            and row["target_shard"]["session_generation"] == target
            for row, source, target in zip(
                reconfigurations, expected_sources, expected_targets
            )
        ),
        "reconfiguration receipts advance only S monotonically",
        checks,
    )


def _check_fault_injection(
    checks: dict[str, bool],
    result: dict,
    *,
    ready: list[dict],
    failed: list[dict],
    helper_events: list[dict],
    reconfigurations: list[dict],
    rollbacks: list[dict],
    selected: str | None,
    hot_artifact: str | None,
) -> None:
    first, last = ready[0], ready[-1]
    injection = result.get("helper_preparation_fault_injection") or {}
    injected = injection.get("injected") or []
    check(len(injected) == 1, "exactly one fault injected", checks)
    check(
        len(reconfigurations) == 2,
        "two physical loads of S (failed attempt plus retry)",
        checks,
    )
    check(
        len(rollbacks) == 1
        and rollbacks[0].get("physical_change") is True
        and rollbacks[0].get("changed_session_id") == selected
        and rollbacks[0]["restored_shard"]["artifact_sha256"] == hot_artifact
        and rollbacks[0]["restored_shard"]["session_generation"] == 3,
        "one physical rollback restored hot gen3 on S",
        checks,
    )
    check(
        len(failed) == 1
        and "injected_helper_preparation_fault" in failed[0]["reason"]
        and failed[0].get("unavailable_session_ids") == [],
        "one logical TRANSITION_FAILED from the injected fault with no unavailable session",
        checks,
    )
    # Physical then logical ordering: the rollback receipt exists and the
    # controller republished the source layout as READY afterwards.
    ready_generations = [row["generation"] for row in ready]
    check(
        len(set(ready_generations)) == 2
        and ready_generations[0] == first["generation"]
        and ready_generations[-1] == last["generation"],
        "source layout stays authoritative until the retry publishes",
        checks,
    )
    retained = [
        row for row in helper_events
        if row.get("kind") == "REBIND_RETAINED_AFTER_ROLLBACK"
    ]
    retargeted = [
        row for row in helper_events if row.get("kind") == "REBIND_RETARGETED"
    ]
    check(
        bool(retained) and bool(retargeted),
        "quiesced hot helper retained through rollback and retargeted on retry",
        checks,
    )


def _check_helper_events(checks: dict[str, bool], helper_events: list[dict]) -> None:
    rebind_errors = [
        row for row in helper_events
        if isinstance(row.get("reason"), str)
        and "request helper rebind requires an active acquired helper"
            in row["reason"]
    ]
    check(not rebind_errors, "no helper rebind ownership error", checks)
    check(
        not [
            row for row in helper_events
            if row.get("kind") in {
                "HELPER_REMATERIALIZATION_FAILED",
                "READY_LAYOUT_REMATERIALIZATION_FAILED",
            }
        ],
        "no helper rematerialization failure",
        checks,
    )


def _phone_calls_by_session(
    proofs: dict,
    hot_artifact: str | None,
    cold_artifact: str | None,
) -> tuple[dict[str, int], dict[str, int]]:
    hot_calls_by_session: dict[str, int] = {}
    cold_calls_by_session: dict[str, int] = {}
    for row in proofs.values():
        target = (
            hot_calls_by_session
            if row.get("artifact_sha256") == hot_artifact
            else cold_calls_by_session
            if row.get("artifact_sha256") == cold_artifact
            else None
        )
        if target is None:
            continue
        for session in row.get("phone_calls_by_session") or []:
            target[session["session_id"]] = (
                target.get(session["session_id"], 0) + int(session["calls"])
            )
    return hot_calls_by_session, cold_calls_by_session


def _check_phone_calls(
    checks: dict[str, bool],
    hot_calls_by_session: dict[str, int],
    cold_calls_by_session: dict[str, int],
    selected: str | None,
    last_shards: dict[str, dict],
) -> None:
    check(
        selected is not None and cold_calls_by_session.get(selected, 0) > 0,
        "cold model produced nonzero calls on S",
        checks,
    )
    check(
        set(cold_calls_by_session) <= {selected},
        "cold model calls only on S",
        checks,
    )
    check(
        all(
            hot_calls_by_session.get(session_id, 0) > 0
            for session_id in last_shards if session_id != selected
        ),
        "hot model produced calls on every retained session",
        checks,
    )


def _check_request_outcomes(
    checks: dict[str, bool], result: dict, receipts: list[dict]
) -> None:
    stale = [
        row for row in result["request_results"]
        if row.get("recoveries")
    ]
    check(not stale, "zero execution recoveries or fallbacks", checks)
    check(
        all(
            row["terminal_ticket"]["dispatch_state"] == "COMPLETED"
            for row in result["request_results"]
        ),
        "every request completed",
        checks,
    )
    usb = result.get("usb_restore_receipts") or []
    check(len(usb) <= 1, "no mid-run USB reset (only the terminal restoration)", checks)
    terminal_resets = 0
    for row in receipts:
        terminal = row.get("terminal") or {}
        terminal_resets += int(terminal.get("reset_recoveries") or 0)
    check(terminal_resets == 0, "zero phone transport reset recoveries", checks)


def evaluate(result: dict, *, expect_fault: bool) -> dict:
    checks: dict[str, bool] = {}
    hot_artifact, cold_artifact = _model_artifacts(result)
    layout_events = result.get("phone_residency_events") or []
    helper_events = result.get("request_helper_events")
    if helper_events is None:
        helper_events = result.get("model_placement_events") or []
    receipts = result.get("direct_phone_receipts") or []
    proofs = result.get("physical_execution_proofs") or {}

    ready = [row for row in layout_events if row.get("kind") == "READY"]
    failed = [
        row for row in layout_events if row.get("kind") == "TRANSITION_FAILED"
    ]
    unavailable = [
        row for row in layout_events
        if row.get("kind") == "SESSION_UNAVAILABLE"
    ]

    check(len(ready) >= 2, "at least two READY layouts (QQQ then QQG)", checks)
    if len(ready) < 2:
        return {"checks": checks, "status": "FAIL"}
    selected, last_shards = _check_ready_layouts(
        checks, ready, hot_artifact, cold_artifact, expect_fault=expect_fault
    )

    reconfigurations = [
        row for row in receipts
        if row.get("kind") == "partial_phone_residency_reconfiguration"
    ]
    rollbacks = [
        row for row in receipts if row.get("kind") == "helper_transition_rollback"
    ]
    _check_reconfigurations(
        checks,
        reconfigurations,
        selected,
        hot_artifact,
        cold_artifact,
        expect_fault=expect_fault,
    )

    if expect_fault:
        _check_fault_injection(
            checks,
            result,
            ready=ready,
            failed=failed,
            helper_events=helper_events,
            reconfigurations=reconfigurations,
            rollbacks=rollbacks,
            selected=selected,
            hot_artifact=hot_artifact,
        )
    else:
        check(len(rollbacks) == 0, "no physical rollback", checks)
        check(len(failed) == 0, "no logical transition failure", checks)
    check(not unavailable, "no session marked UNAVAILABLE", checks)

    _check_helper_events(checks, helper_events)

    hot_calls_by_session, cold_calls_by_session = _phone_calls_by_session(
        proofs, hot_artifact, cold_artifact
    )
    _check_phone_calls(
        checks, hot_calls_by_session, cold_calls_by_session, selected, last_shards
    )

    _check_request_outcomes(checks, result, receipts)

    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "checks": checks,
        "cold_calls_by_session": cold_calls_by_session,
        "hot_calls_by_session": hot_calls_by_session,
        "physical_reconfigurations": len(reconfigurations),
        "physical_rollbacks": len(rollbacks),
        "ready_generations": [row["generation"] for row in ready],
        "schema": "s42-session-cow-gate-verdict-v1",
        "selected_session_id": selected,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--expect-fault-injection", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verdict = evaluate(load(args.result), expect_fault=args.expect_fault_injection)
    encoded = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="ascii")
    sys.stdout.write(encoded)
    return 0 if verdict["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
