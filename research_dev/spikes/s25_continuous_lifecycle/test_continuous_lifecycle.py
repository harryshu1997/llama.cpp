#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from continuous_lifecycle import (
    DEFAULT_REQUESTS,
    LifecycleError,
    LifecyclePlan,
    LifecycleStep,
    PlannedRow,
    RequestDefinition,
    build_lifecycle_plan,
    compare_outputs,
    execute_plan,
    execute_serial_oracle,
    output_mismatches,
    validate_lifecycle_plan,
    validate_stage_layout,
)

from stage_v3_client import (
    BatchResult,
    Hello,
    ProtocolError,
    STAGE_V3_BASE_CAPABILITIES,
    STAGE_V3_CAP_TERMINAL,
    Status,
)


def hello(start: int, end: int, terminal: bool = False) -> Hello:
    capabilities = STAGE_V3_BASE_CAPABILITIES
    if terminal:
        capabilities |= STAGE_V3_CAP_TERMINAL
    return Hello(start, end, 48, 3840, 4, 600, 4, 4, capabilities)


class FakeClient:
    def __init__(self, name: str, nonfinite: bool = False, batch_bias: bool = False):
        self.name = name
        self.nonfinite = nonfinite
        self.batch_bias = batch_bias
        self.active: dict[int, tuple[int, int, int]] = {}
        self.draining = False
        self.batch_sizes: list[int] = []

    def status(self) -> Status:
        return Status(len(self.active), 4, self.draining)

    def batch(self, rows):
        if self.draining:
            raise ProtocolError("batch after drain")
        if len({row.seq_id for row in rows}) != len(rows):
            raise ProtocolError("duplicate sequence in batch")
        self.batch_sizes.append(len(rows))
        results = []
        for row in rows:
            previous = self.active.get(row.seq_id)
            identity = (row.request_id, row.route_epoch)
            if previous is None:
                if row.position != 0:
                    raise ProtocolError("new sequence does not start at zero")
            elif previous[:2] != identity or previous[2] != row.position:
                raise ProtocolError("sequence identity or position mismatch")
            self.active[row.seq_id] = (*identity, row.position + 1)
            if self.name == "tail":
                bias = len(rows) if self.batch_bias else 0
                results.append(BatchResult(
                    row.request_id, row.route_epoch, row.seq_id, row.position,
                    None, row.token + row.position + 7 + bias,
                ))
            else:
                hidden = (float("nan"), 1.0) if self.nonfinite else (float(row.token), float(row.position))
                results.append(BatchResult(
                    row.request_id, row.route_epoch, row.seq_id, row.position,
                    hidden, None,
                ))
        return tuple(results)

    def remove(self, seq_id, request_id, route_epoch) -> Status:
        previous = self.active.get(seq_id)
        if previous is None or previous[:2] != (request_id, route_epoch):
            raise ProtocolError("remove identity mismatch")
        del self.active[seq_id]
        return self.status()

    def drain(self) -> Status:
        if self.active:
            raise ProtocolError("drain with active sequence")
        self.draining = True
        return self.status()


def fake_clients(**kwargs):
    return {
        "op12": FakeClient("op12", **kwargs),
        "op15": FakeClient("op15", **kwargs),
        "tail": FakeClient("tail", **kwargs),
    }


class LifecyclePlanTests(unittest.TestCase):
    def test_frozen_membership_and_reuse(self) -> None:
        plan = build_lifecycle_plan(DEFAULT_REQUESTS, 2)
        batches = [step for step in plan.steps if step.rows]
        self.assertEqual(
            [[row.request for row in step.rows] for step in batches],
            [["A", "B"], ["A", "B"], ["C", "B"], ["C", "B"], ["C", "D"], ["D"]],
        )
        self.assertEqual(
            [[row.seq_id for row in step.rows] for step in batches],
            [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1]],
        )
        self.assertEqual(plan.steps[2].removals, ("A",))
        self.assertEqual(plan.steps[2].admissions, ("C",))
        self.assertEqual(plan.steps[4].removals, ("B",))
        self.assertEqual(plan.steps[4].admissions, ("D",))
        self.assertEqual(plan.steps[-1].removals, ("D",))
        self.assertFalse(plan.steps[-1].rows)

    def test_reject_duplicate_names(self) -> None:
        with self.assertRaises(ValueError):
            build_lifecycle_plan((
                RequestDefinition("A", 0, 1, 2),
                RequestDefinition("A", 1, 1, 3),
            ), 2)

    def test_reject_bool_capacity(self) -> None:
        with self.assertRaises(ValueError):
            build_lifecycle_plan(DEFAULT_REQUESTS, True)

    def test_reject_fractional_request_fields(self) -> None:
        with self.assertRaises(ValueError):
            RequestDefinition("A", 0.5, 1, 2)
        with self.assertRaises(ValueError):
            RequestDefinition("A", 0, 1.5, 2)
        with self.assertRaises(ValueError):
            RequestDefinition("A", 0, 1, 2.5)

    def test_reject_early_removal(self) -> None:
        plan = build_lifecycle_plan(DEFAULT_REQUESTS, 2)
        steps = list(plan.steps)
        steps[1] = dataclasses.replace(steps[1], removals=("A",))
        with self.assertRaises(LifecycleError):
            validate_lifecycle_plan(LifecyclePlan(2, tuple(steps)), DEFAULT_REQUESTS)

    def test_reject_slot_alias(self) -> None:
        plan = build_lifecycle_plan(DEFAULT_REQUESTS, 2)
        steps = list(plan.steps)
        rows = list(steps[0].rows)
        rows[1] = PlannedRow("B", 0, 0)
        steps[0] = dataclasses.replace(steps[0], rows=tuple(rows))
        with self.assertRaises(LifecycleError):
            validate_lifecycle_plan(LifecyclePlan(2, tuple(steps)), DEFAULT_REQUESTS)

    def test_reject_missing_active_row(self) -> None:
        plan = build_lifecycle_plan(DEFAULT_REQUESTS, 2)
        steps = list(plan.steps)
        steps[1] = dataclasses.replace(steps[1], rows=(steps[1].rows[0],))
        with self.assertRaises(LifecycleError):
            validate_lifecycle_plan(LifecyclePlan(2, tuple(steps)), DEFAULT_REQUESTS)


class OutputTests(unittest.TestCase):
    def test_equal_outputs(self) -> None:
        compare_outputs({"A": [1, 2]}, {"A": (1, 2)})

    def test_reject_token_mismatch(self) -> None:
        with self.assertRaises(LifecycleError):
            compare_outputs({"A": [1, 2]}, {"A": [1, 3]})

    def test_reject_request_mismatch(self) -> None:
        with self.assertRaises(LifecycleError):
            compare_outputs({"A": [1]}, {"B": [1]})

    def test_mismatch_record_preserves_both_sequences(self) -> None:
        self.assertEqual(output_mismatches(
            {"A": [1, 2]}, {"A": [1, 3]},
        ), [{"request": "A", "oracle": [1, 2], "treatment": [1, 3]}])


class ExecutionTests(unittest.TestCase):
    def test_serial_and_continuous_execution(self) -> None:
        clients = fake_clients()
        serial = execute_serial_oracle(clients, DEFAULT_REQUESTS)
        dynamic = execute_plan(
            clients,
            build_lifecycle_plan(DEFAULT_REQUESTS, 2),
            DEFAULT_REQUESTS,
            2001,
            2,
            "dynamic",
        )
        compare_outputs(serial["outputs"], dynamic["outputs"])
        self.assertEqual(
            [event["members"] for event in dynamic["events"] if event["members"]],
            [["A", "B"], ["A", "B"], ["C", "B"], ["C", "B"], ["C", "D"], ["D"]],
        )
        for client in clients.values():
            self.assertEqual(client.status().active_sequences, 0)
            self.assertIn(2, client.batch_sizes)
            self.assertIn(1, client.batch_sizes)

    def test_nonfinite_activation_fails_closed(self) -> None:
        clients = fake_clients()
        clients["op12"].nonfinite = True
        definition = (RequestDefinition("A", 0, 1, 2),)
        with self.assertRaises(ProtocolError):
            execute_plan(
                clients, build_lifecycle_plan(definition, 1), definition,
                1001, 1, "nonfinite",
            )

    def test_batch_dependent_token_is_detected(self) -> None:
        clients = fake_clients()
        clients["tail"].batch_bias = True
        serial = execute_serial_oracle(clients, DEFAULT_REQUESTS)
        dynamic = execute_plan(
            clients,
            build_lifecycle_plan(DEFAULT_REQUESTS, 2),
            DEFAULT_REQUESTS,
            2001,
            2,
            "dynamic",
        )
        with self.assertRaises(LifecycleError):
            compare_outputs(serial["outputs"], dynamic["outputs"])


class LayoutTests(unittest.TestCase):
    def test_fixed_layout(self) -> None:
        validate_stage_layout({
            "op12": hello(0, 8),
            "op15": hello(8, 16),
            "tail": hello(16, 48, True),
        }, 2)

    def test_reject_wrong_boundary(self) -> None:
        with self.assertRaises(ProtocolError):
            validate_stage_layout({
                "op12": hello(0, 8),
                "op15": hello(8, 15),
                "tail": hello(16, 48, True),
            }, 2)

    def test_reject_small_capacity(self) -> None:
        small = dataclasses.replace(hello(0, 8), max_streams=1)
        with self.assertRaises(ProtocolError):
            validate_stage_layout({
                "op12": small,
                "op15": hello(8, 16),
                "tail": hello(16, 48, True),
            }, 2)


if __name__ == "__main__":
    unittest.main()
