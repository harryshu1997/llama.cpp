#!/usr/bin/env python3
"""Speculative rows: draft-model speculation sized to the phone FFN row budget (campaign
``speculative_rows``). Configuration, catalog parameters, launch contract and arguments,
per-request policy and ledger, completion accounting, proof bounds, RESULT gating and the
analysis tool; every path stays byte-identical without the key."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from research_dev.scheduler import GGUFModelManifestLoader
from research_dev.scheduler.adapters import (
    CanonicalHttpExecutionBackend,
    LlamaCppCompletionPayload,
    LlamaCppHttpClient,
    LlamaServerProcessConfiguration,
    LlamaServerProcessLauncher,
    ManagedLlamaServer,
    PhysicalAdapterError,
    llama_server_launch_contract,
)
from research_dev.scheduler.adapters import http_backend
from research_dev.scheduler.adapters.llama_server_contracts import _launch_contract_supports_execution
from research_dev.scheduler.adapters.llama_server_ops.proofs import ManagedServerProofMixin
from research_dev.scheduler.adapters.speculative_rows import (
    DRAFT_BYTES_PARAMETER,
    DRAFT_GPU_LAYERS_PARAMETER,
    DRAFT_MAX_PARAMETER,
    DRAFT_MIN_PARAMETER,
    DRAFT_MODEL_PATH_PARAMETER,
    DRAFT_SHA256_PARAMETER,
    PATCHED_SERVER_PARAMETER,
    QUALIFIED_ROWS_PARAMETER,
    ROW_BUDGET_PARAMETER,
    LlamaServerSpeculativeContract,
    SpeculativeRequestContract,
    SpeculativeRowLedger,
    per_request_draft_max,
    reachable_call_rows,
    speculative_adapter_parameters,
    speculative_completion_statistics,
    speculative_launch_contract,
    speculative_request_contract,
    static_draft_max,
)
from research_dev.scheduler.campaigns.burstgpt import arguments, launch, preflight, runner
from research_dev.scheduler.campaigns.burstgpt.tools import speculative_rows_analysis
from research_dev.scheduler.config import (
    CAMPAIGN_MANIFEST_SCHEMA,
    CampaignManifest,
    SchedulerConfigurationError,
    SpeculativeRowsModelConfiguration,
    speculative_rows_configuration,
)

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
REPO_ROOT = TESTS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
from gguf import GGUFWriter  # noqa: E402
from test_gguf_cost import write_synthetic_gguf  # noqa: E402
from test_llama_server_adapter import execution_command  # noqa: E402

PIN = "sha256:" + "b" * 64
MODEL_ID = "qwen3-14b-q4km-dequant-f16"


def write_draft_gguf(path: Path, vocabulary: int = 64) -> None:
    """A tiny draft GGUF whose token embedding carries ``vocabulary`` rows."""
    writer = GGUFWriter(path, "synthetic_draft")
    writer.add_context_length(128)
    writer.add_embedding_length(16)
    writer.add_block_count(1)
    writer.add_feed_forward_length(32)
    writer.add_head_count(2)
    writer.add_tensor("token_embd.weight", np.zeros((vocabulary, 16), dtype=np.float16))
    writer.add_tensor("output_norm.weight", np.zeros((16,), dtype=np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def model_row(path: Path, **changes) -> dict:
    return {"draft_model_path": str(path), "draft_max": 3, **changes}


class ConfigurationTests(unittest.TestCase):
    ROW = {"schema": CAMPAIGN_MANIFEST_SCHEMA, "campaign_id": "speculative-rows-test",
           "rig_manifest_path": "rig.json", "models_manifest_path": "models.json",
           "evidence_manifest_path": "evidence.json", "selection_mode": "energy-aware",
           "maximum_latency_ppm": 1_100_000, "energy_attribution_kind": "diagnostic",
           "trace": {"large_requests_path": "large.json", "overlay_requests_path": "overlay.json",
                     "trace_manifest_path": "trace.json"}}

    def test_model_configuration_round_trips_and_omits_optionals(self):
        row = SpeculativeRowsModelConfiguration.from_json(model_row(Path("/m/draft.gguf")), Path("/inputs"))
        self.assertEqual((row.draft_max, row.draft_min, row.row_budget, row.qualified_rows, row.draft_gpu_layers,
                          row.patched_server_sha256), (3, 0, None, None, 0, None))
        self.assertEqual(row.to_json(), {"draft_gpu_layers": 0, "draft_max": 3, "draft_min": 0,
                                         "draft_model_path": "/m/draft.gguf"})
        full = SpeculativeRowsModelConfiguration.from_json(model_row(
            Path("draft.gguf"), draft_min=1, row_budget=4, qualified_rows=[4, 1, 2], draft_gpu_layers=99,
            patched_server_sha256=PIN), Path("/inputs"))
        self.assertEqual(full.draft_model_path, Path("/inputs/draft.gguf"))
        self.assertEqual(full.qualified_rows, (1, 2, 4))
        self.assertEqual(SpeculativeRowsModelConfiguration.from_json(full.to_json(), Path("/other")), full)

    def test_model_configuration_rejects_untyped_values(self):
        for changes in ({"draft_max": 0}, {"draft_max": 4}, {"draft_max": "3"}, {"draft_max": True},
                        {"draft_min": 4}, {"draft_min": -1}, {"row_budget": 0}, {"qualified_rows": []},
                        {"qualified_rows": [1, 1]}, {"qualified_rows": [0]}, {"qualified_rows": 4},
                        {"draft_gpu_layers": -1}, {"patched_server_sha256": "b" * 64},
                        {"patched_server_sha256": "sha256:xyz"}, {"unknown": 1}):
            with self.subTest(changes=changes), self.assertRaises(SchedulerConfigurationError):
                SpeculativeRowsModelConfiguration.from_json(model_row(Path("/m/d.gguf"), **changes), Path("/i"))
        for value in ({}, [], {"": model_row(Path("/m/d.gguf"))}, {MODEL_ID: {"draft_max": 3}}):
            with self.subTest(value=value), self.assertRaises(SchedulerConfigurationError):
                speculative_rows_configuration(value, Path("/i"))
        self.assertIsNone(speculative_rows_configuration(None, Path("/i")))

    def test_campaign_manifest_omits_the_key_by_default_and_round_trips_when_set(self):
        plain = CampaignManifest.from_json(dict(self.ROW), Path("/inputs"))
        self.assertIsNone(plain.speculative_rows)
        self.assertNotIn("speculative_rows", plain.to_json())
        value = {MODEL_ID: model_row(Path("/m/draft.gguf"), draft_min=1)}
        configured = CampaignManifest.from_json({**self.ROW, "speculative_rows": value}, Path("/inputs"))
        self.assertEqual(configured.to_json()["speculative_rows"],
                         {MODEL_ID: {"draft_gpu_layers": 0, "draft_max": 3, "draft_min": 1,
                                     "draft_model_path": "/m/draft.gguf"}})
        self.assertEqual(CampaignManifest.from_json(configured.to_json(), Path("/inputs")), configured)
        self.assertEqual(configured.to_json().keys() - plain.to_json().keys(), {"speculative_rows"})

    def test_runner_and_preflight_arguments_parse_the_same_object(self):
        value = {MODEL_ID: model_row(Path("/m/draft.gguf"), qualified_rows=[4, 2, 1])}
        parsed = arguments.speculative_rows_json(json.dumps(value))
        self.assertEqual(parsed, {MODEL_ID: {"draft_gpu_layers": 0, "draft_max": 3, "draft_min": 0,
                                             "draft_model_path": "/m/draft.gguf", "qualified_rows": [1, 2, 4]}})
        for text in ("null", "[]", "{", json.dumps({MODEL_ID: model_row(Path("relative.gguf"))}),
                     json.dumps({MODEL_ID: {"draft_max": 3}})):
            with self.subTest(text=text), self.assertRaises((argparse.ArgumentTypeError, ValueError)):
                arguments.speculative_rows_json(text)
        action = next(row for row in arguments._build_parser()._actions
                      if "--speculative-rows-json" in row.option_strings)
        self.assertEqual((action.dest, action.default), ("speculative_rows", None))
        source = Path(preflight.__file__).read_text(encoding="utf-8")
        self.assertIn('"--speculative-rows-json", dest="speculative_rows", type=speculative_rows_json', source)

    def test_launch_commands_carry_the_flag_only_when_declared(self):
        from test_two_phone_helpers import CampaignConfigurationTests, two_phone_rig_json
        from research_dev.scheduler.config import load_scheduler_configuration

        case = CampaignConfigurationTests()
        case.setUp()
        self.addCleanup(case.tearDown)
        path = case._write(two_phone_rig_json(), None)
        plain = load_scheduler_configuration(path, environ={})
        arguments_of = lambda configuration: launch.preflight_command(  # noqa: E731
            configuration, catalog_path=case.root / "c.json", normal_usb_receipt_path=case.root / "u.json",
            output_path=case.root / "out.json")
        self.assertNotIn("--speculative-rows-json", arguments_of(plain))
        campaign = json.loads(path.read_text())
        campaign["speculative_rows"] = {MODEL_ID: model_row(Path("/m/draft.gguf"))}
        path.write_text(json.dumps(campaign))
        command = arguments_of(load_scheduler_configuration(path, environ={}))
        value = json.loads(command[command.index("--speculative-rows-json") + 1])
        self.assertEqual(value, {MODEL_ID: {"draft_gpu_layers": 0, "draft_max": 3, "draft_min": 0,
                                            "draft_model_path": "/m/draft.gguf"}})
        self.assertEqual(Path(launch.__file__).read_text(encoding="utf-8").count('"--speculative-rows-json"'), 2)


class PolicyTests(unittest.TestCase):
    def test_reachable_rows_follow_the_draft_loop(self):
        self.assertEqual(sorted(reachable_call_rows(1, 3, 0)), [1, 2, 3, 4])
        self.assertEqual(sorted(reachable_call_rows(1, 3, 3)), [1, 4])
        self.assertEqual(sorted(reachable_call_rows(2, 1, 1)), [1, 2, 3, 4])
        self.assertEqual(sorted(reachable_call_rows(2, 3, 3)), [1, 2, 4, 5, 8])
        self.assertEqual(sorted(reachable_call_rows(2, 0, 0)), [1, 2])

    def test_static_bound_keeps_every_reachable_call_inside_the_phone_contract(self):
        self.assertEqual(static_draft_max(draft_max=3, draft_min=0, row_budget=4, parallel=2), (1, None))
        self.assertEqual(static_draft_max(draft_max=3, draft_min=0, row_budget=4, parallel=1), (3, None))
        bound, reason = static_draft_max(draft_max=3, draft_min=0, row_budget=4, parallel=4)
        self.assertEqual(bound, 0)
        self.assertIn("4 decode slots fill the 4-row phone budget", reason)
        # the Pixel qualified rows 1/2/4: two slots reach 3 rows, one slot with n_min 3 never does
        self.assertEqual(static_draft_max(draft_max=3, draft_min=0, row_budget=4, parallel=2,
                                          qualified_rows=(1, 2, 4))[0], 0)
        self.assertEqual(static_draft_max(draft_max=3, draft_min=3, row_budget=4, parallel=1,
                                          qualified_rows=(1, 2, 4)), (3, None))
        with self.assertRaises(PhysicalAdapterError):
            static_draft_max(draft_max=3, draft_min=0, row_budget=4, parallel=2, qualified_rows=(0,))

    def test_per_request_bound_reserves_a_plain_row_for_every_slot_that_can_fill(self):
        self.assertEqual(per_request_draft_max(draft_max=3, row_budget=4, parallel=1, reserved_rows=()), 3)
        self.assertEqual(per_request_draft_max(draft_max=3, row_budget=4, parallel=2, reserved_rows=()), 2)
        self.assertEqual(per_request_draft_max(draft_max=3, row_budget=4, parallel=2, reserved_rows=(3,)), 0)
        self.assertEqual(per_request_draft_max(draft_max=3, row_budget=4, parallel=2, reserved_rows=(1,)), 2)
        self.assertEqual(per_request_draft_max(draft_max=3, row_budget=4, parallel=4, reserved_rows=()), 0)
        self.assertEqual(per_request_draft_max(draft_max=3, row_budget=8, parallel=4, reserved_rows=()), 3)
        self.assertEqual(per_request_draft_max(draft_max=3, row_budget=4, parallel=2, reserved_rows=(1, 1)), 0)
        for kwargs in ({"draft_max": 4}, {"row_budget": 0}, {"parallel": 0}, {"reserved_rows": (0,)},
                       {"reserved_rows": ("1",)}, {"draft_max": 3.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(PhysicalAdapterError):
                per_request_draft_max(**{"draft_max": 3, "row_budget": 4, "parallel": 2,
                                         "reserved_rows": (), **kwargs})

    def test_ledger_admits_under_one_lock_and_releases(self):
        ledger = SpeculativeRowLedger()
        grants = []

        def admit(request_id):
            def grant(reserved):
                bound = per_request_draft_max(draft_max=3, row_budget=8, parallel=4, reserved_rows=reserved)
                grants.append((request_id, bound))
                return 1 + bound
            return ledger.admit("http://127.0.0.1:1", request_id, grant)

        threads = [threading.Thread(target=admit, args=(f"r{index}",)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        reserved = ledger.reserved_rows("http://127.0.0.1:1")
        self.assertEqual(len(reserved), 4)
        self.assertLessEqual(sum(reserved), 8)
        self.assertEqual(sorted(bound for _, bound in grants), [0, 0, 1, 3])
        ledger.release("http://127.0.0.1:1", "r0")
        self.assertEqual(len(ledger.reserved_rows("http://127.0.0.1:1")), 3)
        with self.assertRaises(PhysicalAdapterError):
            ledger.reserve("http://127.0.0.1:1", "r1", 1)
        self.assertIsNone(ledger.admit("http://127.0.0.1:2", "x", lambda reserved: None))
        self.assertEqual(ledger.reserved_rows("http://127.0.0.1:2"), ())


class DraftBindingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        target = self.root / "target.gguf"
        write_synthetic_gguf(target)
        self.manifest = GGUFModelManifestLoader.load("synthetic-model-id", target)
        self.draft = self.root / "draft.gguf"
        write_draft_gguf(self.draft)
        self.configuration = SpeculativeRowsModelConfiguration(self.draft, 3, draft_min=1)

    def tearDown(self):
        self.directory.cleanup()

    def parameters(self, **changes):
        return speculative_adapter_parameters(replace(self.configuration, **changes), self.manifest)

    def test_draft_is_bound_by_digest_size_and_vocabulary(self):
        parameters = self.parameters()
        self.assertEqual(parameters, {
            DRAFT_MODEL_PATH_PARAMETER: str(self.draft), DRAFT_SHA256_PARAMETER: file_sha256(self.draft),
            DRAFT_BYTES_PARAMETER: self.draft.stat().st_size, DRAFT_MAX_PARAMETER: 3, DRAFT_MIN_PARAMETER: 1,
            DRAFT_GPU_LAYERS_PARAMETER: 0,
        })
        full = self.parameters(row_budget=4, qualified_rows=(4, 1, 2), patched_server_sha256=PIN, draft_gpu_layers=99)
        self.assertEqual((full[ROW_BUDGET_PARAMETER], full[QUALIFIED_ROWS_PARAMETER], full[PATCHED_SERVER_PARAMETER],
                          full[DRAFT_GPU_LAYERS_PARAMETER]), (4, "1,2,4", PIN, 99))
        other = self.root / "other.gguf"
        write_draft_gguf(other, vocabulary=48)
        with self.assertRaisesRegex(PhysicalAdapterError, "vocabulary 48 differs from the target 64"):
            self.parameters(draft_model_path=other)
        with self.assertRaisesRegex(PhysicalAdapterError, "not a file"):
            self.parameters(draft_model_path=self.root / "missing.gguf")
        (self.root / "text.gguf").write_text("not a gguf")
        with self.assertRaisesRegex(PhysicalAdapterError, "unreadable"):
            self.parameters(draft_model_path=self.root / "text.gguf")


class LaunchContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        target = self.root / "model.gguf"
        write_synthetic_gguf(target)
        self.manifest = GGUFModelManifestLoader.load("synthetic-model-id", target)
        self.command = execution_command(self.manifest.artifact_sha256)
        self.draft = self.root / "draft.gguf"
        write_draft_gguf(self.draft)
        self.speculative = speculative_adapter_parameters(
            SpeculativeRowsModelConfiguration(self.draft, 3, draft_min=1), self.manifest)

    def tearDown(self):
        self.directory.cleanup()

    def with_parameters(self, command=None, **extra):
        command = self.command if command is None else command
        return replace(command, adapter_parameters={**command.adapter_parameters, **self.speculative, **extra})

    def desktop_command(self, **extra):
        parameters = {key: value for key, value in self.command.adapter_parameters.items()
                      if not key.startswith(("ffn_", "usb", "phone_"))}
        parameters.update({**self.speculative, **extra})
        return replace(self.command, adapter_parameters=parameters, operator_plan={
            **self.command.operator_plan,
            "operators": [{**row, "device_ids": ["accelerator-b"], "split_axis": "none", "split_fraction_ppm": 0}
                          for row in self.command.operator_plan["operators"]],
            "execution_contract": {**self.command.operator_plan["execution_contract"], "execution_mode": "desktop"},
        })

    def test_absent_parameters_launch_without_a_draft(self):
        self.assertIsNone(llama_server_launch_contract(self.command, self.manifest).speculative)
        self.assertIsNone(speculative_launch_contract(self.command.adapter_parameters, phone_attached=False,
                                                      phone_max_tokens=None))
        with self.assertRaisesRegex(PhysicalAdapterError, "incomplete"):
            speculative_launch_contract({**self.command.adapter_parameters, DRAFT_MAX_PARAMETER: 3},
                                        phone_attached=False, phone_max_tokens=None)

    def test_desktop_only_launch_drafts_the_cap_and_its_identity_carries_the_draft(self):
        contract = llama_server_launch_contract(self.desktop_command(), self.manifest)
        self.assertIsNone(contract.phone_device_id)
        self.assertEqual(contract.speculative, LlamaServerSpeculativeContract(
            draft_model_path=str(self.draft), draft_sha256=self.speculative[DRAFT_SHA256_PARAMETER],
            draft_bytes=self.speculative[DRAFT_BYTES_PARAMETER], draft_max=3, draft_min=1, draft_gpu_layers=0,
            row_budget=0, per_request_control=False))
        plain = replace(contract, speculative=None)
        self.assertNotEqual(contract, plain)
        self.assertTrue(_launch_contract_supports_execution(contract, contract))
        self.assertFalse(_launch_contract_supports_execution(plain, contract))
        self.assertEqual(contract.speculative.to_json()["per_request_control"], False)

    def test_phone_launch_drafts_only_under_the_patched_server_pin(self):
        stock = llama_server_launch_contract(self.with_parameters(), self.manifest)
        self.assertIsNotNone(stock.phone_device_id)
        self.assertIsNone(stock.speculative)
        pinned = llama_server_launch_contract(self.with_parameters(**{PATCHED_SERVER_PARAMETER: PIN}), self.manifest)
        self.assertEqual((pinned.speculative.per_request_control, pinned.speculative.row_budget,
                          pinned.speculative.draft_max, pinned.speculative.patched_server_sha256), (True, 4, 3, PIN))
        budgeted = llama_server_launch_contract(
            self.with_parameters(**{PATCHED_SERVER_PARAMETER: PIN, ROW_BUDGET_PARAMETER: 2}), self.manifest)
        self.assertEqual(budgeted.speculative.row_budget, 2)
        # four decode slots fill a four-row budget: even a lone request gets no draft row
        full = speculative_launch_contract({**self.with_parameters(**{PATCHED_SERVER_PARAMETER: PIN}).adapter_parameters,
                                            "parallel": 4}, phone_attached=True, phone_max_tokens=4)
        self.assertIsNone(full)
        self.assertIsNone(speculative_launch_contract(pinned and self.with_parameters(
            **{PATCHED_SERVER_PARAMETER: PIN}).adapter_parameters, phone_attached=True, phone_max_tokens=None))

    def launcher(self, device_names):
        server_path = self.root / "llama-server"
        server_path.write_text("#!/bin/sh\n", encoding="ascii")
        server_path.chmod(0o755)
        return LlamaServerProcessLauncher(LlamaServerProcessConfiguration(
            server_path=server_path,
            model_paths_by_artifact={self.manifest.artifact_sha256: self.root / "model.gguf"},
            library_paths_by_device={}, executable_device_names=device_names, output_directory=self.root,
        )), server_path

    def launch(self, launcher, contract):
        class FakeManagedServer:
            def __init__(self, command, environment, output_directory, label, launch_contract):
                self.command, self.environment = command, environment
                self.process = SimpleNamespace(poll=lambda: None)
                self.stderr_lines = [f"load_tensors: offloaded {launch_contract.gpu_layers}/2 layers to GPU"]
                if launch_contract.ffn_environment:
                    self.stderr_lines.append("S41SERVERFFN ready host=usb port=0 columns=64")

            def start(self):
                return None

            def stop(self):
                return None

        with patch("research_dev.scheduler.adapters.llama_server.ManagedLlamaServer", FakeManagedServer), \
                patch.object(launcher, "_healthy", return_value=True):
            return launcher._launch_contract("http://127.0.0.1:19000", contract, self.manifest,
                                             label="speculative", control_check=lambda: None)

    def test_launch_arguments_carry_the_draft_only_with_a_contract(self):
        launcher, server_path = self.launcher({"accelerator-b": "CUDA0"})
        contract = llama_server_launch_contract(self.desktop_command(), self.manifest)
        plain = self.launch(launcher, replace(contract, speculative=None)).command
        self.assertFalse(any(value.startswith("--spec") for value in plain))
        command = self.launch(launcher, contract).command
        self.assertEqual(command[:len(plain)], plain)
        self.assertEqual(list(command[len(plain):]), [
            "--spec-type", "draft-simple", "--spec-draft-model", str(self.draft),
            "--spec-draft-n-max", "3", "--spec-draft-n-min", "1", "--spec-draft-ngl", "0"])
        gpu = self.launch(launcher, replace(contract, speculative=replace(contract.speculative, draft_gpu_layers=99)))
        self.assertEqual(list(gpu.command[-4:]), ["--spec-draft-ngl", "99", "--spec-draft-device", "CUDA0"])
        with self.assertRaisesRegex(PhysicalAdapterError, "GPU parent"):
            self.launch(launcher, replace(contract, gpu_layers=0, speculative=replace(
                contract.speculative, draft_gpu_layers=99)))
        tampered = replace(contract, speculative=replace(contract.speculative, draft_bytes=1))
        with self.assertRaisesRegex(PhysicalAdapterError, "draft differs"):
            self.launch(launcher, tampered)
        pinned = llama_server_launch_contract(self.with_parameters(**{PATCHED_SERVER_PARAMETER: PIN}), self.manifest)
        with self.assertRaisesRegex(PhysicalAdapterError, "patched server pin"):
            self.launch(launcher, pinned)
        matching = replace(pinned, speculative=replace(pinned.speculative, patched_server_sha256=file_sha256(server_path)))
        self.assertIn("--spec-draft-model", self.launch(launcher, matching).command)
        with self.assertRaisesRegex(PhysicalAdapterError, "without the patched server pin"):
            self.launch(launcher, replace(pinned, speculative=replace(
                pinned.speculative, per_request_control=False, patched_server_sha256=None)))


class RequestContractTests(unittest.TestCase):
    PARAMETERS = {"parallel": 2, "ubatch_size": 512, DRAFT_MODEL_PATH_PARAMETER: "/m/draft.gguf",
                  DRAFT_SHA256_PARAMETER: "sha256:" + "c" * 64, DRAFT_BYTES_PARAMETER: 10, DRAFT_MAX_PARAMETER: 3,
                  DRAFT_MIN_PARAMETER: 1, DRAFT_GPU_LAYERS_PARAMETER: 0}

    def test_request_contract_follows_the_launch_mode(self):
        self.assertIsNone(speculative_request_contract({"parallel": 2}, ()))
        static = speculative_request_contract(self.PARAMETERS, ())
        self.assertEqual((static.n_max, static.per_request_control, static.row_budget, static.reserved_rows),
                         (None, False, 0, 1))
        self.assertEqual(static.body_fields(), {})
        desktop = speculative_request_contract({**self.PARAMETERS, PATCHED_SERVER_PARAMETER: PIN}, ())
        self.assertEqual((desktop.n_max, desktop.body_fields()), (3, {"speculative": {"n_max": 3, "n_min": 1}}))
        phone = {**self.PARAMETERS, PATCHED_SERVER_PARAMETER: PIN, "phone_device_id": "helper-c"}
        lone = speculative_request_contract(phone, ())
        self.assertEqual((lone.row_budget, lone.n_max, lone.reserved_rows), (2, 0, 1))
        wide = speculative_request_contract({**phone, "ffn_max_tokens": 4}, ())
        self.assertEqual((wide.row_budget, wide.n_max, wide.reserved_rows), (4, 2, 3))
        self.assertEqual(speculative_request_contract({**phone, "ffn_max_tokens": 4}, (3,)).n_max, 0)
        self.assertEqual(speculative_request_contract({**phone, "ffn_max_tokens": 4, ROW_BUDGET_PARAMETER: 3}, ()).n_max, 1)
        with self.assertRaises(PhysicalAdapterError):
            SpeculativeRequestContract(3, 0, 4, 2, False, n_max=1)
        with self.assertRaises(PhysicalAdapterError):
            SpeculativeRequestContract(3, 0, 4, 2, True, n_max=None)

    def test_completion_statistics_derive_steps_from_accepted_draft(self):
        request = speculative_request_contract(self.PARAMETERS, ())
        self.assertEqual(speculative_completion_statistics({"draft_n": 30, "draft_n_accepted": 20}, 50, request), {
            "acceptance_rate_ppm": 666_666, "draft_accepted": 20, "draft_n": 30, "n_max": None,
            "per_request_control": False, "row_budget": 0, "schema": "s42-speculative-rows-request-v1",
            "tokens_per_step_ppm": 1_666_666, "verif_steps": 30})
        self.assertEqual(speculative_completion_statistics({}, 5, request)["tokens_per_step_ppm"], 1_000_000)
        for timings in ({"draft_n": 3, "draft_n_accepted": 4}, {"draft_n": 9, "draft_n_accepted": 5},
                        {"draft_n": "3"}):
            with self.subTest(timings=timings), self.assertRaises(PhysicalAdapterError):
                speculative_completion_statistics(timings, 5, request)


class FakeConnection:
    """One streamed completion: records the request body, answers canned SSE chunks."""

    bodies: list[dict] = []
    timings_extra: dict = {}

    def __init__(self, host, port, timeout=None):
        self.lines = []

    def connect(self):
        return None

    def request(self, method, path, body=None, headers=None):
        request = json.loads(body)
        FakeConnection.bodies.append(request)
        count = request["n_predict"]
        chunks = [{"tokens": [100 + index], "content": f" w{index}", "tokens_predicted": index + 1,
                   "id_slot": 0, "stop": False} for index in range(count)]
        chunks.append({"tokens": [], "content": "", "tokens_predicted": count, "id_slot": 0, "stop": True,
                       "model": "synthetic-model", "timings": {
                           "prompt_n": len(request["prompt"]), "predicted_n": count, "predicted_ms": 5.0,
                           "prompt_ms": 1.0, **FakeConnection.timings_extra}})
        self.lines = [("data: " + json.dumps(chunk) + "\n").encode() for chunk in chunks]

    def getresponse(self):
        lines = iter(self.lines)
        return SimpleNamespace(status=200, readline=lambda: next(lines, b""))

    def close(self):
        return None


class CompletionClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        FakeConnection.bodies = []
        FakeConnection.timings_extra = {}

    def tearDown(self):
        self.directory.cleanup()

    def payload(self, name, speculative=None):
        return LlamaCppCompletionPayload(
            request_id="request-" + name, expected_model_alias="synthetic-model", input_tokens=2, output_tokens=4,
            prompt_tokens=(1, 2), seed=7, stream_path=self.root / (name + ".sse"), on_first_token=lambda _ns: None,
            quality_mode="accounting-only", speculative=speculative)

    def complete(self, payload):
        client = LlamaCppHttpClient(lambda *_args: [{"id": 0, "id_task": 7, "is_processing": True}])
        with patch.object(http_backend.http.client, "HTTPConnection", FakeConnection):
            return client.complete("http://127.0.0.1:19000", payload, lambda: None)

    def test_body_and_result_change_only_with_a_speculative_contract(self):
        plain = self.complete(self.payload("plain"))
        self.assertNotIn("speculative", plain)
        self.assertNotIn("speculative", FakeConnection.bodies[0])
        static = SpeculativeRequestContract(3, 1, 0, 2, False)
        FakeConnection.timings_extra = {"draft_n": 6, "draft_n_accepted": 2}
        result = self.complete(self.payload("static", static))
        self.assertEqual(FakeConnection.bodies[1], FakeConnection.bodies[0])
        self.assertEqual(result["speculative"]["draft_accepted"], 2)
        self.assertEqual(result["speculative"]["verif_steps"], 2)
        self.assertEqual({key: value for key, value in result.items() if key not in ("speculative", "stream_sha256")},
                         {key: value for key, value in plain.items() if key != "stream_sha256"})
        controlled = SpeculativeRequestContract(3, 1, 4, 2, True, n_max=2)
        self.complete(self.payload("controlled", controlled))
        self.assertEqual(FakeConnection.bodies[2]["speculative"], {"n_max": 2, "n_min": 1})
        self.assertEqual({key: value for key, value in FakeConnection.bodies[2].items() if key != "speculative"},
                         FakeConnection.bodies[0])
        with self.assertRaises(PhysicalAdapterError):
            self.payload("typed", speculative={"n_max": 1})


class BackendLedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        target = self.root / "model.gguf"
        write_synthetic_gguf(target)
        self.manifest = GGUFModelManifestLoader.load("synthetic-model-id", target)
        self.command = execution_command(self.manifest.artifact_sha256)
        self.backend = CanonicalHttpExecutionBackend(
            LlamaCppHttpClient(), SimpleNamespace(measure=lambda *_args: None), epoch_ns=0)

    def tearDown(self):
        self.directory.cleanup()

    def payload(self, name):
        return LlamaCppCompletionPayload(
            request_id=name, expected_model_alias="synthetic-model", input_tokens=2, output_tokens=4,
            prompt_tokens=(1, 2), seed=7, stream_path=self.root / (name + ".sse"), on_first_token=lambda _ns: None)

    def command_with(self, request_id, **extra):
        return replace(self.command, request_id=request_id, ticket_id=request_id + ":attempt:0",
                       adapter_parameters={**self.command.adapter_parameters, **extra})

    def test_backend_binds_contracts_from_the_endpoint_ledger(self):
        plain = self.payload("plain")
        self.assertIs(self.backend._speculative_payload(self.command, plain), plain)
        speculative = {DRAFT_MODEL_PATH_PARAMETER: "/m/draft.gguf", DRAFT_SHA256_PARAMETER: "sha256:" + "c" * 64,
                       DRAFT_BYTES_PARAMETER: 10, DRAFT_MAX_PARAMETER: 3, DRAFT_MIN_PARAMETER: 0,
                       DRAFT_GPU_LAYERS_PARAMETER: 0, PATCHED_SERVER_PARAMETER: PIN, "parallel": 2,
                       "ffn_max_tokens": 4}
        first = self.backend._speculative_payload(self.command_with("first", **speculative), self.payload("first"))
        self.assertEqual((first.speculative.n_max, first.speculative.reserved_rows), (2, 3))
        second = self.backend._speculative_payload(self.command_with("second", **speculative), self.payload("second"))
        self.assertEqual((second.speculative.n_max, second.speculative.reserved_rows), (0, 1))
        self.assertEqual(self.backend._speculative_ledger.reserved_rows(self.command.endpoint), (3, 1))
        self.backend._release_speculative_rows(self.command_with("first", **speculative), first)
        self.assertEqual(self.backend._speculative_ledger.reserved_rows(self.command.endpoint), (1,))
        third = self.backend._speculative_payload(self.command_with("third", **speculative), self.payload("third"))
        self.assertEqual(third.speculative.n_max, 2)
        static = self.backend._speculative_payload(
            self.command_with("static", **{key: value for key, value in speculative.items()
                                           if key != PATCHED_SERVER_PARAMETER}), self.payload("static"))
        self.assertEqual((static.speculative.n_max, static.speculative.per_request_control), (None, False))
        self.assertEqual(len(self.backend._speculative_ledger.reserved_rows(self.command.endpoint)), 2)
        with self.assertRaises(PhysicalAdapterError):
            self.backend._speculative_payload(self.command, first)


class ProofBoundTests(unittest.TestCase):
    def test_rows_bound_is_exact_without_speculation_and_a_range_with_it(self):
        within = ManagedServerProofMixin._rows_within_speculative_bound
        self.assertTrue(within({(0, 8): 10}, {(0, 8): 10}, 1))
        self.assertFalse(within({(0, 8): 11}, {(0, 8): 10}, 1))
        self.assertTrue(within({(0, 8): 11}, {(0, 8): 10}, 4))
        self.assertTrue(within({(0, 8): 40}, {(0, 8): 10}, 4))
        self.assertFalse(within({(0, 8): 41}, {(0, 8): 10}, 4))
        self.assertFalse(within({(0, 8): 9}, {(0, 8): 10}, 4))
        self.assertFalse(within({(0, 8): 10, (1, 8): 10}, {(0, 8): 10}, 4))

    def test_rows_per_token_follows_the_launched_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_synthetic_gguf(root / "model.gguf")
            manifest = GGUFModelManifestLoader.load("synthetic-model-id", root / "model.gguf")
            contract = llama_server_launch_contract(execution_command(manifest.artifact_sha256), manifest)
            server = ManagedLlamaServer(("synthetic-server",), {}, root, "synthetic", contract)
            self.assertEqual(server._speculative_rows_per_token(), 1)
            drafted = replace(contract, speculative=LlamaServerSpeculativeContract(
                "/m/draft.gguf", "sha256:" + "c" * 64, 10, 3, 0, 0, 4, True, PIN))
            self.assertEqual(ManagedLlamaServer(("s",), {}, root, "synthetic", drafted)._speculative_rows_per_token(), 4)
            self.assertEqual(ManagedServerProofMixin()._speculative_rows_per_token(), 1)


class ResultAndAnalysisTests(unittest.TestCase):
    def request(self, request_id, output_tokens, speculative=None):
        return {"request_id": request_id, "model_id": MODEL_ID, "output_tokens": output_tokens,
                **({} if speculative is None else {"speculative": speculative})}

    def statistics(self, draft_n, accepted, output_tokens):
        steps = output_tokens - accepted
        return {"acceptance_rate_ppm": 0 if not draft_n else accepted * 1_000_000 // draft_n,
                "draft_accepted": accepted, "draft_n": draft_n, "n_max": None, "per_request_control": False,
                "row_budget": 0, "schema": "s42-speculative-rows-request-v1",
                "tokens_per_step_ppm": output_tokens * 1_000_000 // steps, "verif_steps": steps}

    def test_result_carries_the_summary_only_under_the_key(self):
        rows = [self.request("a", 10, self.statistics(9, 6, 10)), self.request("b", 5, self.statistics(0, 0, 5)),
                self.request("c", 7)]
        self.assertEqual(runner._speculative_rows_result(SimpleNamespace(configuration=SimpleNamespace(
            speculative_rows=None)), rows), {})
        self.assertEqual(runner._speculative_rows_result(SimpleNamespace(), rows), {})
        configuration = speculative_rows_configuration({MODEL_ID: model_row(Path("/m/draft.gguf"))}, Path("/"))
        summary = runner._speculative_rows_result(SimpleNamespace(configuration=SimpleNamespace(
            speculative_rows=configuration)), rows)["speculative_rows"]
        self.assertEqual(summary, {
            "acceptance_rate_ppm": 666_666, "configuration": {MODEL_ID: configuration[MODEL_ID].to_json()},
            "draft_accepted": 6, "draft_n": 9, "requests_with_draft": 1, "requests_with_statistics": 2,
            "schema": "s42-speculative-rows-summary-v1", "tokens_per_step_ppm": 15 * 1_000_000 // 9,
            "verif_steps": 9})
        result = runner._assemble_result
        self.assertIn("_speculative_rows_result", Path(runner.__file__).read_text(encoding="utf-8"))
        self.assertTrue(callable(result))

    def test_analysis_parses_rows_per_call_and_request_statistics(self):
        lines = [
            "S41SERVERFFNUSB request=1 layer=2 tokens=4 columns=100 started_ns=1 h2d_completed_ns=2 d2h_completed_ns=3 compute_us=4",
            "S41SERVERFFNUSB request=2 layer=3 tokens=1 columns=100 started_ns=1 h2d_completed_ns=2 d2h_completed_ns=3 compute_us=4",
            "0.00.000.001 I S41SERVERFFNUSB request=3 layer=2 tokens=4 columns=100 started_ns=1 h2d_completed_ns=2 d2h_completed_ns=3 compute_us=4",
            "unrelated tokens=9",
        ]
        self.assertEqual(speculative_rows_analysis.usb_rows_histogram(lines), {1: 1, 4: 2})
        result = {"request_results": [self.request("a", 10, self.statistics(9, 6, 10)), self.request("c", 7)],
                  "speculative_rows": {"draft_n": 9}}
        rows = speculative_rows_analysis.request_speculative_rows(result)
        self.assertEqual([(row["request_id"], row["verif_steps"], row["tokens_per_step_ppm"]) for row in rows],
                         [("a", 4, 2_500_000)])
        self.assertEqual(speculative_rows_analysis.request_speculative_rows({"request_results": []}), [])
        with self.assertRaises(ValueError):
            speculative_rows_analysis.request_speculative_rows({"request_results": [
                self.request("x", 3, {"draft_n": "3"})]})
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "RESULT.json").write_text(json.dumps(result))
            (run / "large-model-qwen-desktop.stderr").write_text("\n".join(lines) + "\n")
            report = speculative_rows_analysis.summarize(run)
            self.assertEqual(report["rows_per_call"], {"large-model-qwen-desktop.stderr": {1: 1, 4: 2}})
            self.assertEqual(report["summary"], {"draft_n": 9})
            self.assertEqual(speculative_rows_analysis.main(["--run-dir", str(run), "--json"]), 0)


if __name__ == "__main__":
    unittest.main()
