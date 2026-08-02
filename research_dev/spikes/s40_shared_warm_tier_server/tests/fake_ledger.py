#!/usr/bin/env python3

import copy
from typing import Any

from event_evidence import token_history_digest


COMMAND_BEGIN = {
    "request_dispatched": 0,
    "drain_begin": 1,
    "unload_begin": 2,
    "load_begin": 3,
    "replay_begin": 4,
    "discard_begin": 5,
    "cleanup_begin": 6,
}
COMMAND_END = {
    "execute_end": 0,
    "drain_end": 1,
    "unload_end": 2,
    "load_end": 3,
    "replay_end": 4,
    "discard_end": 5,
    "cleanup_end": 6,
}


def snapshot(
        request_id: str,
        model_id: str,
        prompt: list[int],
        committed: list[int],
        owner: str | None,
        ownership_epoch: int,
        state: str) -> dict[str, Any]:
    return {
        "committed_output_tokens": list(committed),
        "model_id": model_id,
        "owner_id": owner,
        "ownership_epoch": ownership_epoch,
        "position": len(prompt) + len(committed),
        "prompt_tokens": list(prompt),
        "publication_index": len(committed),
        "request_id": request_id,
        "state": state,
    }


class Ledger:
    def __init__(self, run_id: str = "fake-run") -> None:
        self.run_id = run_id
        self.rows: list[dict[str, Any]] = []
        self.t_ns = 0
        self.next_command_id = 1
        self.open_commands: dict[
            tuple[int, str | None, str | None, str | None],
            int,
        ] = {}
        self.command_by_id: dict[int, dict[str, Any]] = {}
        self.latest_execute: dict[tuple[str, str], int] = {}
        self.request_state: dict[str, dict[str, Any]] = {}

    @staticmethod
    def command_key(
            command_kind: int,
            model_id: str | None,
            request_id: str | None,
            executor_id: str | None,
    ) -> tuple[int, str | None, str | None, str | None]:
        return command_kind, model_id, request_id, executor_id

    def append_row(
            self,
            kind: str,
            *,
            epoch: int,
            model_id: str | None,
            request_id: str | None,
            executor_id: str | None,
            state_before: str | None,
            state_after: str | None,
            request: dict[str, Any] | None,
            old_owner: str | None,
            new_owner: str | None,
            old_epoch: int | None,
            new_epoch: int | None,
            publication_index: int | None,
            success: bool,
            detail: str,
            command_id: int | None,
            command_kind: int | None,
            command_disposition: str | None,
            result_publications: list[dict[str, Any]],
            result_request_complete: bool) -> None:
        history = None
        if request is not None:
            history = token_history_digest(
                request["prompt_tokens"],
                request["committed_output_tokens"],
            )
        self.rows.append({
            "command_id": command_id,
            "command_disposition": command_disposition,
            "command_kind": command_kind,
            "controller_epoch": epoch,
            "detail": detail,
            "executor_id": executor_id,
            "history_sha256": history,
            "kind": kind,
            "model_id": model_id,
            "new_owner": new_owner,
            "new_ownership_epoch": new_epoch,
            "old_owner": old_owner,
            "old_ownership_epoch": old_epoch,
            "publication_index": publication_index,
            "request": copy.deepcopy(request),
            "request_id": request_id,
            "result_publications": copy.deepcopy(result_publications),
            "result_request_complete": result_request_complete,
            "run_id": self.run_id,
            "runtime_config_sha256": "0" * 64,
            "schema": "s40-warm-tier-event-v3",
            "schema_version": 3,
            "sequence": len(self.rows),
            "state_after": state_after,
            "state_before": state_before,
            "success": success,
            "t_ns": self.t_ns,
        })
        self.t_ns += 1_000_000
        if request is not None and request_id is not None:
            self.request_state[request_id] = copy.deepcopy(request)

    def add(
            self,
            kind: str,
            *,
            epoch: int = 0,
            model_id: str | None = None,
            request_id: str | None = None,
            executor_id: str | None = None,
            state_before: str | None = None,
            state_after: str | None = None,
            request: dict[str, Any] | None = None,
            old_owner: str | None = None,
            new_owner: str | None = None,
            old_epoch: int | None = None,
            new_epoch: int | None = None,
            publication_index: int | None = None,
            success: bool = True,
            detail: str = "",
            command_id: int | None = None,
            command_kind: int | None = None,
            command_disposition: str | None = None,
            result_publications: list[dict[str, Any]] | None = None,
            result_request_complete: bool = False) -> None:
        raw_publications = [] if result_publications is None \
            else copy.deepcopy(result_publications)
        if kind in {"token_committed", "request_completed"}:
            assert request_id is not None and executor_id is not None
            key = (request_id, executor_id)
            command_id = self.latest_execute.get(key)
            if command_id is None or (
                    kind == "token_committed"
                    and not self.command_by_id[command_id]["open"]
                    and self.command_by_id[command_id]["output_attached"]):
                previous = self.request_state[request_id]
                self.add(
                    "request_dispatched",
                    epoch=epoch,
                    model_id=model_id,
                    request_id=request_id,
                    executor_id=executor_id,
                    request=previous,
                )
                command_id = self.latest_execute[key]
            command = self.command_by_id[command_id]
            if command["open"]:
                assert request is not None
                if kind == "token_committed":
                    assert publication_index is not None
                    raw_publications = [{
                        "owner_id": request["owner_id"],
                        "ownership_epoch": request["ownership_epoch"],
                        "position": request["position"] - 1,
                        "publication_index": publication_index,
                        "token": request["committed_output_tokens"][
                            publication_index],
                    }]
                self.add(
                    "execute_end",
                    epoch=epoch,
                    model_id=model_id,
                    request_id=request_id,
                    executor_id=executor_id,
                    request=self.request_state[request_id],
                    success=success,
                    command_id=command_id,
                    command_kind=0,
                    result_publications=raw_publications,
                )
            elif kind == "token_committed":
                end_row = self.rows[command["end_row"]]
                if not end_row["result_publications"]:
                    assert request is not None and publication_index is not None
                    end_row["result_publications"] = [{
                        "owner_id": request["owner_id"],
                        "ownership_epoch": request["ownership_epoch"],
                        "position": request["position"] - 1,
                        "publication_index": publication_index,
                        "token": request["committed_output_tokens"][
                            publication_index],
                    }]
            if kind == "request_completed":
                self.rows[command["end_row"]][
                    "result_request_complete"] = True
            command["output_attached"] = True
            command_kind = 0
            command_disposition = None
            raw_publications = []
            result_request_complete = False
        elif kind == "executor_failed":
            assert request_id is not None and executor_id is not None
            command_id = self.latest_execute[(request_id, executor_id)]
            command = self.command_by_id[command_id]
            if command["open"]:
                self.add(
                    "execute_end",
                    epoch=epoch,
                    model_id=model_id,
                    request_id=request_id,
                    executor_id=executor_id,
                    request=self.request_state[request_id],
                    success=False,
                    detail=detail,
                    command_id=command_id,
                    command_kind=0,
                )
            command_kind = 0
        elif kind in COMMAND_BEGIN:
            command_kind = COMMAND_BEGIN[kind]
            if command_id is None:
                command_id = self.next_command_id
                self.next_command_id += 1
            key = self.command_key(
                command_kind, model_id, request_id, executor_id)
            assert key not in self.open_commands
            self.open_commands[key] = command_id
            self.command_by_id[command_id] = {
                "key": key,
                "open": True,
                "output_attached": False,
            }
            if command_kind == 0:
                assert request_id is not None and executor_id is not None
                self.latest_execute[(request_id, executor_id)] = command_id
        elif kind in COMMAND_END:
            inferred_kind = COMMAND_END[kind]
            if command_kind is None:
                command_kind = inferred_kind
            assert command_kind == inferred_kind
            key = self.command_key(
                command_kind, model_id, request_id, executor_id)
            if command_id is None:
                command_id = self.open_commands[key]
            assert self.open_commands.pop(key) == command_id
            self.command_by_id[command_id]["open"] = False
            self.command_by_id[command_id]["end_row"] = len(self.rows)
            if command_disposition is None:
                command_disposition = "RECEIVED"

        self.append_row(
            kind,
            epoch=epoch,
            model_id=model_id,
            request_id=request_id,
            executor_id=executor_id,
            state_before=state_before,
            state_after=state_after,
            request=request,
            old_owner=old_owner,
            new_owner=new_owner,
            old_epoch=old_epoch,
            new_epoch=new_epoch,
            publication_index=publication_index,
            success=success,
            detail=detail,
            command_id=command_id,
            command_kind=command_kind,
            command_disposition=command_disposition,
            result_publications=raw_publications,
            result_request_complete=result_request_complete,
        )


def expected_request(
        request_id: str = "r0",
        model_id: str = "B",
        arrival_us: int = 5_000) -> dict[str, Any]:
    return {
        "arrival_us": arrival_us,
        "event_id": request_id,
        "input_tokens": 2,
        "model_id": model_id,
        "output_tokens": 8,
        "prompt_tokens": [1, 2],
        "request_index": 0,
        "schema": "s39-cp0d-desktop-request-v1",
        "slo_us": 30_000_000,
        "source_input_tokens": 2,
        "source_model": "fixture",
        "source_output_tokens": 8,
        "source_t_us": 0,
    }


def successful_transition() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ledger = Ledger()
    prompt = [1, 2]
    committed: list[int] = []
    ledger.add("run_start")
    ledger.add(
        "model_state_changed",
        model_id="A",
        executor_id="GPU",
        state_before="ABSENT",
        state_after="READY",
    )
    ledger.add(
        "model_state_changed",
        model_id="B",
        executor_id="GPU",
        state_before="ABSENT",
        state_after="ABSENT",
    )
    ledger.add(
        "model_state_changed",
        model_id="B",
        executor_id="PHONE",
        state_before="ABSENT",
        state_after="READY",
    )
    ledger.add(
        "model_state_changed",
        model_id="A",
        executor_id="PHONE",
        state_before="ABSENT",
        state_after="ABSENT",
    )
    queued = snapshot("r0", "B", prompt, committed, None, 0, "QUEUED")
    ledger.add(
        "request_arrived",
        model_id="B",
        request_id="r0",
        request=queued,
    )
    active_phone = snapshot(
        "r0", "B", prompt, committed, "PHONE", 1, "ACTIVE")
    ledger.add(
        "request_dispatched",
        model_id="B",
        request_id="r0",
        executor_id="PHONE",
        request=active_phone,
    )
    committed.append(10)
    active_phone = snapshot(
        "r0", "B", prompt, committed, "PHONE", 1, "ACTIVE")
    ledger.add(
        "token_committed",
        model_id="B",
        request_id="r0",
        executor_id="PHONE",
        request=active_phone,
        publication_index=0,
    )
    ledger.add(
        "switch_intent_submitted",
        epoch=1,
        model_id="B",
        executor_id="GPU",
        detail="intent-0",
    )
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="A",
        executor_id="GPU",
        state_before="READY",
        state_after="DRAINING",
    )
    for kind in ("drain_begin", "drain_end", "unload_begin", "unload_end"):
        ledger.add(kind, epoch=1, model_id="A", executor_id="GPU")
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="A",
        executor_id="GPU",
        state_before="DRAINING",
        state_after="ABSENT",
    )
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="B",
        executor_id="GPU",
        state_before="ABSENT",
        state_after="LOADING",
    )
    ledger.add("load_begin", epoch=1, model_id="B", executor_id="GPU")
    ledger.add("load_end", epoch=1, model_id="B", executor_id="GPU")
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="B",
        executor_id="GPU",
        state_before="LOADING",
        state_after="REPLAYING",
    )
    ledger.add(
        "replay_begin",
        epoch=1,
        model_id="B",
        request_id="r0",
        executor_id="GPU",
        request=active_phone,
    )
    ledger.add(
        "replay_end",
        epoch=1,
        model_id="B",
        request_id="r0",
        executor_id="GPU",
        request=active_phone,
    )
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="B",
        executor_id="GPU",
        state_before="REPLAYING",
        state_after="READY",
    )
    active_gpu = snapshot(
        "r0", "B", prompt, committed, "GPU", 2, "ACTIVE")
    ledger.add(
        "ownership_commit",
        epoch=1,
        model_id="B",
        request_id="r0",
        executor_id="GPU",
        request=active_gpu,
        old_owner="PHONE",
        new_owner="GPU",
        old_epoch=1,
        new_epoch=2,
        publication_index=1,
    )
    ledger.add(
        "ownership_commit_complete",
        epoch=1,
        model_id="B",
        executor_id="GPU",
        detail="1",
    )
    ledger.add(
        "cleanup_begin",
        epoch=1,
        model_id="B",
        request_id="r0",
        executor_id="PHONE",
        request=active_gpu,
    )
    ledger.add(
        "cleanup_end",
        epoch=1,
        model_id="B",
        request_id="r0",
        executor_id="PHONE",
        request=active_gpu,
    )
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="B",
        executor_id="PHONE",
        state_before="READY",
        state_after="DRAINING",
    )
    for kind in ("drain_begin", "drain_end", "unload_begin", "unload_end"):
        ledger.add(kind, epoch=1, model_id="B", executor_id="PHONE")
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="B",
        executor_id="PHONE",
        state_before="DRAINING",
        state_after="ABSENT",
    )
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="A",
        executor_id="PHONE",
        state_before="ABSENT",
        state_after="LOADING",
    )
    ledger.add("load_begin", epoch=1, model_id="A", executor_id="PHONE")
    ledger.add("load_end", epoch=1, model_id="A", executor_id="PHONE")
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="A",
        executor_id="PHONE",
        state_before="LOADING",
        state_after="READY",
    )
    for token in range(11, 18):
        publication = len(committed)
        committed.append(token)
        active_gpu = snapshot(
            "r0", "B", prompt, committed, "GPU", 2, "ACTIVE")
        ledger.add(
            "token_committed",
            epoch=1,
            model_id="B",
            request_id="r0",
            executor_id="GPU",
            request=active_gpu,
            publication_index=publication,
        )
    complete = snapshot(
        "r0", "B", prompt, committed, "GPU", 2, "COMPLETED")
    ledger.add(
        "request_completed",
        epoch=1,
        model_id="B",
        request_id="r0",
        executor_id="GPU",
        request=complete,
    )
    ledger.add("run_end", epoch=1)
    return ledger.rows, [expected_request()]


def executor_failure() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ledger = Ledger("failure-run")
    prompt = [1, 2]
    ledger.add("run_start")
    ledger.add(
        "model_state_changed",
        model_id="B",
        executor_id="PHONE",
        state_before="ABSENT",
        state_after="READY",
    )
    queued = snapshot("r0", "B", prompt, [], None, 0, "QUEUED")
    ledger.add(
        "request_arrived",
        model_id="B",
        request_id="r0",
        request=queued,
    )
    active = snapshot("r0", "B", prompt, [], "PHONE", 1, "ACTIVE")
    ledger.add(
        "request_dispatched",
        model_id="B",
        request_id="r0",
        executor_id="PHONE",
        request=active,
    )
    ledger.add(
        "executor_failed",
        model_id="B",
        request_id="r0",
        executor_id="PHONE",
        success=False,
        detail="injected",
    )
    stranded = snapshot("r0", "B", prompt, [], "PHONE", 1, "STRANDED")
    ledger.add(
        "request_stranded",
        model_id="B",
        request_id="r0",
        executor_id="PHONE",
        request=stranded,
        success=False,
        detail="injected",
    )
    ledger.add("run_end")
    return ledger.rows, [expected_request(arrival_us=2_000)]


def precommit_rollback() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ledger = Ledger("rollback-run")
    prompt = [1, 2]
    committed: list[int] = []
    ledger.add("run_start")
    for model_id, executor_id, after in (
        ("A", "GPU", "READY"),
        ("B", "GPU", "ABSENT"),
        ("B", "PHONE", "READY"),
        ("A", "PHONE", "ABSENT"),
    ):
        ledger.add(
            "model_state_changed",
            model_id=model_id,
            executor_id=executor_id,
            state_before="ABSENT",
            state_after=after,
        )
    queued = snapshot("r0", "B", prompt, [], None, 0, "QUEUED")
    ledger.add(
        "request_arrived",
        model_id="B",
        request_id="r0",
        request=queued,
    )
    active = snapshot("r0", "B", prompt, [], "PHONE", 1, "ACTIVE")
    ledger.add(
        "request_dispatched",
        model_id="B",
        request_id="r0",
        executor_id="PHONE",
        request=active,
    )
    ledger.add(
        "switch_intent_submitted",
        epoch=1,
        model_id="B",
        executor_id="GPU",
        detail="intent-rollback",
    )
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="A",
        executor_id="GPU",
        state_before="READY",
        state_after="DRAINING",
    )
    for kind in ("drain_begin", "drain_end", "unload_begin", "unload_end"):
        ledger.add(kind, epoch=1, model_id="A", executor_id="GPU")
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="A",
        executor_id="GPU",
        state_before="DRAINING",
        state_after="ABSENT",
    )
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="B",
        executor_id="GPU",
        state_before="ABSENT",
        state_after="LOADING",
    )
    ledger.add("load_begin", epoch=1, model_id="B", executor_id="GPU")
    ledger.add("load_end", epoch=1, model_id="B", executor_id="GPU")
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="B",
        executor_id="GPU",
        state_before="LOADING",
        state_after="REPLAYING",
    )
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="B",
        executor_id="PHONE",
        state_before="READY",
        state_after="DRAINING",
    )
    ledger.add(
        "replay_begin",
        epoch=1,
        model_id="B",
        request_id="r0",
        executor_id="GPU",
        request=active,
    )
    ledger.add(
        "replay_end",
        epoch=1,
        model_id="B",
        request_id="r0",
        executor_id="GPU",
        request=active,
        detail="frontier mismatch",
    )
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="B",
        executor_id="PHONE",
        state_before="DRAINING",
        state_after="READY",
    )
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="B",
        executor_id="GPU",
        state_before="REPLAYING",
        state_after="FAILED",
    )
    ledger.add(
        "discard_begin",
        epoch=1,
        model_id="B",
        executor_id="GPU",
    )
    ledger.add(
        "discard_end",
        epoch=1,
        model_id="B",
        executor_id="GPU",
    )
    ledger.add(
        "model_state_changed",
        epoch=1,
        model_id="B",
        executor_id="GPU",
        state_before="FAILED",
        state_after="ABSENT",
    )
    for token in range(10, 18):
        publication = len(committed)
        committed.append(token)
        active = snapshot(
            "r0", "B", prompt, committed, "PHONE", 1, "ACTIVE")
        ledger.add(
            "token_committed",
            epoch=1,
            model_id="B",
            request_id="r0",
            executor_id="PHONE",
            request=active,
            publication_index=publication,
        )
    complete = snapshot(
        "r0", "B", prompt, committed, "PHONE", 1, "COMPLETED")
    ledger.add(
        "request_completed",
        epoch=1,
        model_id="B",
        request_id="r0",
        executor_id="PHONE",
        request=complete,
    )
    ledger.add("run_end", epoch=1)
    return ledger.rows, [expected_request()]


def resources(run_id: str, end_ns: int) -> list[dict[str, Any]]:
    rows = []
    for sequence, t_ns in enumerate((0, end_ns)):
        rows.append({
            "controller_pid": 1234,
            "controller_process_cpu_ticks": 100 + sequence,
            "controller_process_cpu_utilization_milli_pct": 5_000,
            "controller_process_rss_bytes": 8_000_000_000,
            "controller_process_start_ticks": 99,
            "controller_process_swap_bytes": 0,
            "cpu_utilization_milli_pct": 25_000,
            "gpu_memory_free_bytes": 4_000_000_000,
            "gpu_memory_used_bytes": 12_000_000_000,
            "gpu_power_mw": 100_000,
            "gpu_uuid": "GPU-fixture",
            "host_boot_id": "boot-fixture",
            "process_metric_scope": "CONTROLLER_PROCESS_ONLY",
            "run_id": run_id,
            "schema": "s40-selected-gpu-resource-v3",
            "sequence": sequence,
            "system_mem_available_bytes": 16_000_000_000,
            "system_swap_free_bytes": 1_000_000_000,
            "system_swap_total_bytes": 1_000_000_000,
            "t_ns": t_ns,
        })
    return rows
