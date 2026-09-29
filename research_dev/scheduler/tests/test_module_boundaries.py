#!/usr/bin/env python3
"""Compatibility and ownership checks for the modular scheduler implementation."""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import research_dev.scheduler as public
from research_dev.scheduler._internal.adaptive_decode import AdaptiveDecodeController
from research_dev.scheduler._internal.model_placement_controller import (
    ModelPlacementController,
)
from research_dev.scheduler._internal.runtime_controller import RuntimeController


PACKAGE = "research_dev.scheduler"
ROOT = Path(__file__).resolve().parents[1]
CONTRACT_FACADES = (
    "_internal.runtime_plan",
    "_internal.runtime_capabilities",
    "_internal.runtime_controller",
    "_internal.model_placement_controller",
    "_internal.adaptive_decode",
    "_unified.helper_preparation",
    "adapters.phone_session",
    "config",
)
CONTRACT_PARTS = frozenset({
    "plan_contracts", "capability_contracts", "request_contracts",
    "model_placement_contracts", "phone_session_contracts", "configuration",
    "adaptive_decode_state", "helper_preparation_checks",
})
OPERATIONS_OWNERS = (
    ("_internal.adaptive_decode", "adaptive_decode_ops"),
    ("_internal.model_placement_controller", "model_placement_ops"),
    ("_internal.runtime_controller", "runtime_controller_ops"),
    ("adapters.phone_session", "phone_session_ops"),
    ("_unified.phone_residency", "phone_residency_ops"),
    ("_unified.helper_envelopes", "helper_envelopes_ops"),
    ("_unified.helper_preparation", "helper_preparation_ops"),
    ("_unified.automated_requests", "automated_requests_ops"),
    ("_unified.automated_selection", "automated_selection_ops"),
    ("_unified.automated_candidates", "automated_candidates_ops"),
    ("_unified.placement_epochs", "placement_epochs_ops"),
)


class ModuleBoundaryTests(unittest.TestCase):
    def test_contract_reexports_keep_canonical_identity(self) -> None:
        checked = 0
        for name in CONTRACT_FACADES:
            facade = importlib.import_module(PACKAGE + "." + name)
            tree = ast.parse(Path(facade.__file__).read_text(encoding="ascii"))
            for node in tree.body:
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                if CONTRACT_PARTS.isdisjoint(node.module.split(".")):
                    continue
                origin_name = importlib.util.resolve_name(
                    "." * node.level + node.module, facade.__package__,
                )
                origin = importlib.import_module(origin_name)
                for alias in node.names:
                    bound = alias.asname or alias.name
                    with self.subTest(facade=name, symbol=bound):
                        self.assertIs(
                            getattr(facade, bound), getattr(origin, alias.name),
                        )
                    checked += 1
        self.assertGreater(checked, 100)

    def test_public_ticket_and_plan_use_canonical_contracts(self) -> None:
        from research_dev.scheduler._internal.plan_contracts.execution import (
            RuntimeExecutionPlan,
        )
        from research_dev.scheduler._internal.request_contracts.ticket import (
            RuntimeRequestTicket,
        )

        self.assertIs(public.RuntimeExecutionPlan, RuntimeExecutionPlan)
        self.assertIs(public.RuntimeRequestTicket, RuntimeRequestTicket)

    def test_request_delegation_passes_existing_owner(self) -> None:
        controller = RuntimeController()
        with mock.patch(
            PACKAGE + "._internal.runtime_controller_ops.admission.bind_decode_cohort",
            return_value=mock.sentinel.ticket,
        ) as operation:
            result = controller.bind_decode_cohort("request", mock.sentinel.binding)
        self.assertIs(result, mock.sentinel.ticket)
        operation.assert_called_once_with(
            controller, "request", mock.sentinel.binding,
        )

    def test_layout_delegation_preserves_keyword_contract(self) -> None:
        controller = ModelPlacementController()
        with mock.patch(
            PACKAGE + "._internal.model_placement_ops.planning.reject_phone_layout_proposal",
            return_value=mock.sentinel.layout,
        ) as operation:
            result = controller.reject_phone_layout_proposal(
                3, observed_at_us=10, reason="stale",
            )
        self.assertIs(result, mock.sentinel.layout)
        operation.assert_called_once_with(
            controller, 3, observed_at_us=10, reason="stale",
        )

    def test_request_checkpoint_restores_same_queue_owner(self) -> None:
        controller = RuntimeController()
        queue = controller.queue
        checkpoint = controller.checkpoint()
        controller._quarantined_routes.add("test-route")
        controller.restore(checkpoint)
        self.assertIs(controller.queue, queue)
        self.assertEqual(controller.checkpoint(), checkpoint)

    def test_adaptive_checkpoint_restores_same_lock_owner(self) -> None:
        controller = AdaptiveDecodeController()
        lock = controller._lock
        checkpoint = controller.checkpoint()
        controller._history_component_bindings["test"] = ("a", "b")
        controller.restore(checkpoint)
        self.assertIs(controller._lock, lock)
        self.assertEqual(controller.checkpoint(), checkpoint)

    def test_operations_do_not_import_their_facade(self) -> None:
        for owner, package in OPERATIONS_OWNERS:
            parent = owner.rsplit(".", 1)[0]
            folder = ROOT.joinpath(*parent.split("."), package)
            for path in sorted(folder.glob("*.py")):
                tree = ast.parse(path.read_text(encoding="ascii"))
                context = PACKAGE + "." + parent + "." + package
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    resolved = importlib.util.resolve_name(
                        "." * node.level + (node.module or ""), context,
                    )
                    imported = {
                        resolved, *(resolved + "." + a.name for a in node.names),
                    }
                    with self.subTest(owner=owner, path=path.name):
                        self.assertNotIn(PACKAGE + "." + owner, imported)
                        self.assertNotIn(PACKAGE + ".scheduler", imported)


if __name__ == "__main__":
    unittest.main()
