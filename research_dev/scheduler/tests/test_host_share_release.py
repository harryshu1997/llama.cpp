"""Decode-only relocation contracts: release proof lines, planning lower bound, launch pass-through."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from research_dev.scheduler import GGUFModelManifestLoader, ModelManifest  # noqa: E402
from research_dev.scheduler._internal.runtime_resources import (  # noqa: E402
    RuntimeHostShareReleaseProof,
    RuntimeResourceError,
    host_share_release_lower_bound_bytes,
)
from research_dev.scheduler.adapters import (  # noqa: E402
    LlamaServerProcessConfiguration,
    LlamaServerProcessLauncher,
    PhysicalAdapterError,
    llama_server_launch_contract,
)
from research_dev.scheduler.adapters.llama_server_contracts import _launch_contract_supports_execution  # noqa: E402
from test_gguf_cost import write_synthetic_gguf  # noqa: E402
from test_llama_server_adapter import adaptive_execution_command, execution_command  # noqa: E402

GEMMA = ModelManifest.from_json(json.loads(
    (TESTS_DIR.parent / "campaigns/burstgpt/data/GEMMA_MANIFEST.json").read_text(encoding="ascii")
))
# exact page-inward release measured by the 2026-09-16 gate for layers 0-7 at host_columns 3840
GATE_EXACT_RELEASE_8_LAYERS = 1_981_743_104


class ReleaseProofTests(unittest.TestCase):
    def test_decode_and_local_lines_parse_with_the_log_prefix(self) -> None:
        decode = RuntimeHostShareReleaseProof.parse_line(
            "0.27.3 I srv apply_dormant: S41SERVERFFN dormant_host_share phase=decode layer_mask=255 "
            "host_columns=3840 released_bytes=1981743104 ranges=30736 elapsed_us=412000"
        )
        self.assertEqual((decode.phase, decode.layer_mask, decode.host_columns), ("decode", 255, 3840))
        self.assertEqual((decode.released_bytes, decode.ranges, decode.elapsed_us), (1_981_743_104, 30_736, 412_000))
        local = RuntimeHostShareReleaseProof.parse_line(
            "S41SERVERFFN dormant_host_share phase=local layer_mask=255 host_columns=3840 "
            "restored_bytes=1981743104 elapsed_us=2100000"
        )
        self.assertEqual((local.phase, local.released_bytes, local.ranges), ("local", 1_981_743_104, 0))
        self.assertIsNone(RuntimeHostShareReleaseProof.parse_line("S41SERVERFFN ready host=usb"))
        self.assertEqual(RuntimeHostShareReleaseProof.from_json(decode.to_json()), decode)

    def test_malformed_lines_fail_closed(self) -> None:
        for line in (
            "S41SERVERFFN dormant_host_share phase=decode layer_mask=255",
            "S41SERVERFFN dormant_host_share phase=idle layer_mask=255 host_columns=0 released_bytes=1 elapsed_us=1",
            "S41SERVERFFN dormant_host_share phase=decode layer_mask=0 host_columns=0 released_bytes=1 ranges=1 elapsed_us=1",
            "S41SERVERFFN dormant_host_share phase=decode phase=decode layer_mask=1 host_columns=0 released_bytes=1 elapsed_us=1",
        ):
            with self.subTest(line=line):
                with self.assertRaises(RuntimeResourceError):
                    RuntimeHostShareReleaseProof.parse_line(line)


class LowerBoundTests(unittest.TestCase):
    def test_bound_matches_the_page_geometry_and_never_exceeds_the_measured_release(self) -> None:
        bound = host_share_release_lower_bound_bytes(GEMMA, 0xFF, 3840)
        # gate/up: 16 tensors x (21,600 - 1) pages; down: 8 layers x 3,840 rows x (5 - 1) pages
        self.assertEqual(bound, 16 * 21_599 * 4096 + 8 * 3840 * 4 * 4096)
        self.assertLessEqual(bound, GATE_EXACT_RELEASE_8_LAYERS)
        self.assertGreater(bound, 0.95 * GATE_EXACT_RELEASE_8_LAYERS)

    def test_bound_is_monotonic_in_the_host_share(self) -> None:
        full = host_share_release_lower_bound_bytes(GEMMA, (1 << 24) - 1, 0)
        three_quarters = host_share_release_lower_bound_bytes(GEMMA, (1 << 24) - 1, 3840)
        half = host_share_release_lower_bound_bytes(GEMMA, (1 << 24) - 1, 7680)
        none = host_share_release_lower_bound_bytes(GEMMA, (1 << 24) - 1, GEMMA.feed_forward_length)
        self.assertGreater(full, three_quarters)
        self.assertGreater(three_quarters, half)
        self.assertEqual(none, 0)
        self.assertLess(full, sum(GEMMA.tensor_by_id[f"blk.{il}.ffn_{kind}.weight"].nbytes
                                  for il in range(24) for kind in ("gate", "up", "down")))

    def test_invalid_geometry_fails_closed(self) -> None:
        with self.assertRaises(RuntimeResourceError):
            host_share_release_lower_bound_bytes(GEMMA, 0xFF, GEMMA.feed_forward_length + 32)
        with self.assertRaises(RuntimeResourceError):
            host_share_release_lower_bound_bytes(GEMMA, 0xFF, 3840, page_size=0)


class LaunchPassThroughTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        model = Path(self.directory.name) / "model.gguf"
        write_synthetic_gguf(model)
        self.manifest = GGUFModelManifestLoader.load("synthetic-model-id", model)
        self.command = adaptive_execution_command(execution_command(self.manifest.artifact_sha256))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _with(self, **parameters):
        return replace(self.command, adapter_parameters={**dict(self.command.adapter_parameters), **parameters})

    def test_release_flag_reaches_the_server_environment_only_when_requested(self) -> None:
        plain = llama_server_launch_contract(self.command, self.manifest)
        self.assertNotIn("S41_SERVER_FFN_DORMANT_HOST_SHARE", plain.ffn_environment)
        released = llama_server_launch_contract(self._with(ffn_host_share_release=1), self.manifest)
        self.assertEqual(released.ffn_environment["S41_SERVER_FFN_DORMANT_HOST_SHARE"], "1")
        self.assertEqual(released.ffn_environment["S41_SERVER_FFN_RUNTIME_CONTROL"], "1")
        self.assertNotEqual(plain.ffn_environment, released.ffn_environment)

    def test_release_flag_requires_decode_runtime_control_and_a_valid_value(self) -> None:
        with self.assertRaisesRegex(PhysicalAdapterError, "flag is invalid"):
            llama_server_launch_contract(self._with(ffn_host_share_release="yes"), self.manifest)
        with self.assertRaisesRegex(PhysicalAdapterError, "flag is invalid"):
            llama_server_launch_contract(self._with(ffn_host_share_release=2), self.manifest)
        without_control = self._with(ffn_host_share_release=1, ffn_assistance_phase="all")
        with self.assertRaisesRegex(PhysicalAdapterError, "requires decode-phase runtime control|phase|batch"):
            llama_server_launch_contract(without_control, self.manifest)

    def test_cache_policy_requires_release_and_reaches_the_typed_launch(self) -> None:
        default = llama_server_launch_contract(self._with(ffn_host_share_release=1), self.manifest)
        self.assertEqual((default.ffn_host_share_drop_cache, default.ffn_host_share_populate), (1, 1))
        self.assertNotIn("S41_SERVER_FFN_DORMANT_DROP_CACHE", default.ffn_environment)
        lazy = llama_server_launch_contract(self._with(ffn_host_share_release=1,
            ffn_host_share_drop_cache=0, ffn_host_share_populate=0), self.manifest)
        self.assertEqual((lazy.ffn_host_share_drop_cache, lazy.ffn_host_share_populate), (0, 0))
        self.assertEqual(lazy.ffn_environment["S41_SERVER_FFN_DORMANT_DROP_CACHE"], "0")
        self.assertEqual(lazy.ffn_environment["S41_SERVER_FFN_DORMANT_POPULATE"], "0")
        for field in ("ffn_host_share_drop_cache", "ffn_host_share_populate"):
            with self.subTest(field=field), self.assertRaisesRegex(PhysicalAdapterError, "requires release"):
                llama_server_launch_contract(self._with(**{field: 0}), self.manifest)
            for value in (True, "0", -1, 2):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(PhysicalAdapterError, "flag is invalid"):
                    llama_server_launch_contract(self._with(ffn_host_share_release=1, **{field: value}), self.manifest)

    def test_runtime_reuse_checks_cache_policy_and_allows_plain_execution(self) -> None:
        launched = llama_server_launch_contract(self._with(ffn_host_share_release=1,
            ffn_host_share_drop_cache=0, ffn_host_share_populate=0), self.manifest)
        environment = dict(launched.ffn_environment)
        environment.pop("S41_SERVER_FFN_SHARDS", None)
        launched = replace(launched, ffn_environment=environment)
        plain = replace(launched, phone_device_id=None, ffn_environment={},
                        ffn_host_share_drop_cache=1, ffn_host_share_populate=1)
        self.assertTrue(_launch_contract_supports_execution(launched, plain))
        changed_environment = dict(environment)
        changed_environment.pop("S41_SERVER_FFN_DORMANT_POPULATE")
        changed = replace(launched, ffn_environment=changed_environment, ffn_host_share_populate=1)
        self.assertFalse(_launch_contract_supports_execution(launched, changed))

    def test_plain_desktop_parent_keeps_default_cache_policy_and_needs_no_confirmation(self) -> None:
        """The cache policy belongs to the dormant host share; a desktop parent without a phone has none.

        Regression for the 2026-09-21 trace run: the cold Qwen desktop parent (no phone, no FFN transport)
        was launched with the model's drop_cache=0 and then failed because it never logged a policy it
        could not apply. Phone routes without release still fail closed (test above).
        """
        from research_dev.scheduler._internal.runtime_plan import RuntimeExecutionContract
        desktop_contract = RuntimeExecutionContract(
            execution_mode="desktop", initial_split_fraction_ppm=0, allowed_adaptive_fractions_ppm=(),
            batch_plan="none", maximum_batch_size=1, queue_depth=1)
        parameters = {name: value for name, value in dict(self.command.adapter_parameters).items()
                      if name != "phone_device_id"}
        parameters.update({"ffn_host_share_release": 1, "ffn_host_share_drop_cache": 0, "ffn_host_share_populate": 0})

        def on_desktop(operator):
            row = dict(operator)
            if "helper-c" in row.get("device_ids", []):
                row.update(device_ids=[parameters["gpu_device_id"]], split_axis="none", split_fraction_ppm=0)
            return row

        operators = [on_desktop(row) for row in dict(self.command.operator_plan)["operators"]]
        desktop = replace(self.command, adapter_parameters=parameters, execution_contract=desktop_contract,
                          operator_plan={**dict(self.command.operator_plan), "operators": operators,
                                         "assisted_operator_kind": None,
                                         "execution_contract": desktop_contract.to_json()})
        plain = llama_server_launch_contract(desktop, self.manifest)
        self.assertIsNone(plain.phone_device_id)
        self.assertEqual(plain.ffn_environment, {})
        self.assertEqual((plain.ffn_host_share_drop_cache, plain.ffn_host_share_populate), (1, 1))
        # the launcher must not demand a policy line from such a server
        root = Path(self.directory.name)
        binary = root / "llama-server"
        binary.write_text("#!/bin/sh\n", encoding="ascii")
        binary.chmod(0o755)
        launcher = LlamaServerProcessLauncher(LlamaServerProcessConfiguration(
            server_path=binary, model_paths_by_artifact={self.manifest.artifact_sha256: root / "model.gguf"},
            library_paths_by_device={}, executable_device_names={plain.gpu_device_id: "CUDA0"}, output_directory=root))
        lines = [f"offloaded {plain.gpu_layers}/{self.manifest.block_count + 1} layers to GPU"]
        server = SimpleNamespace(process=SimpleNamespace(poll=lambda: None), stderr_lines=lines,
                                 start=Mock(), stop=Mock(), remote_resident_proof=lambda: None)
        with patch("research_dev.scheduler.adapters.llama_server.ManagedLlamaServer", return_value=server), \
                patch.object(launcher, "_healthy", return_value=True):
            self.assertIs(launcher._launch_contract("http://127.0.0.1:19000", plain, self.manifest,
                                                    label="plain", control_check=lambda: None), server)

    def test_server_must_confirm_nondefault_cache_policy(self) -> None:
        root = Path(self.directory.name)
        binary = root / "llama-server"
        binary.write_text("#!/bin/sh\n", encoding="ascii")
        binary.chmod(0o755)
        contract = llama_server_launch_contract(self._with(ffn_host_share_release=1,
            ffn_host_share_drop_cache=0, ffn_host_share_populate=0), self.manifest)
        launcher = LlamaServerProcessLauncher(LlamaServerProcessConfiguration(
            server_path=binary, model_paths_by_artifact={self.manifest.artifact_sha256: root / "model.gguf"},
            library_paths_by_device={}, executable_device_names={contract.gpu_device_id: "CUDA0"}, output_directory=root))
        for policy in (None, "drop_cache=1 populate=0", "drop_cache=0 populate=0"):
            with self.subTest(policy=policy):
                lines = [f"offloaded {contract.gpu_layers}/{self.manifest.block_count + 1} layers to GPU", "S41SERVERFFN ready "]
                if policy is not None:
                    lines.append("S41SERVERFFN dormant_policy " + policy)
                server = SimpleNamespace(process=SimpleNamespace(poll=lambda: None), stderr_lines=lines,
                                         start=Mock(), stop=Mock(), remote_resident_proof=lambda: None)
                with patch("research_dev.scheduler.adapters.llama_server.ManagedLlamaServer", return_value=server), \
                        patch.object(launcher, "_healthy", return_value=True):
                    if policy == "drop_cache=0 populate=0":
                        self.assertIs(launcher._launch_contract("http://127.0.0.1:19000", contract, self.manifest,
                            label="policy", control_check=lambda: None), server)
                    else:
                        with self.assertRaisesRegex(PhysicalAdapterError, "did not confirm"):
                            launcher._launch_contract("http://127.0.0.1:19000", contract, self.manifest,
                                label="policy", control_check=lambda: None)
                        server.stop.assert_called_once()

    def test_policy_line_arriving_after_health_is_still_confirmed(self) -> None:
        # The reader thread that fills stderr_lines can lag the health endpoint under load; a one-shot
        # check then aborts a correct server. Measured 2026-09-22 on arm m4a8b at 25 of 31 streams.
        root = Path(self.directory.name)
        binary = root / "llama-server"
        binary.write_text("#!/bin/sh\n", encoding="ascii")
        binary.chmod(0o755)
        contract = llama_server_launch_contract(self._with(ffn_host_share_release=1,
            ffn_host_share_drop_cache=0, ffn_host_share_populate=0), self.manifest)
        launcher = LlamaServerProcessLauncher(LlamaServerProcessConfiguration(
            server_path=binary, model_paths_by_artifact={self.manifest.artifact_sha256: root / "model.gguf"},
            library_paths_by_device={}, executable_device_names={contract.gpu_device_id: "CUDA0"}, output_directory=root))

        class LateLines(list):
            """Appends the policy line only on a later read, as the reader thread would."""

            def __init__(self, *rows):
                super().__init__(rows)
                self.reads = 0

            def __iter__(self):
                self.reads += 1
                if self.reads > 2 and "S41SERVERFFN dormant_policy drop_cache=0 populate=0" not in self:
                    self.append("S41SERVERFFN dormant_policy drop_cache=0 populate=0")
                return super().__iter__()

        lines = LateLines(f"offloaded {contract.gpu_layers}/{self.manifest.block_count + 1} layers to GPU",
                          "S41SERVERFFN ready ")
        server = SimpleNamespace(process=SimpleNamespace(poll=lambda: None), stderr_lines=lines,
                                 start=Mock(), stop=Mock(), remote_resident_proof=lambda: None)
        with patch("research_dev.scheduler.adapters.llama_server.ManagedLlamaServer", return_value=server), \
                patch.object(launcher, "_healthy", return_value=True):
            self.assertIs(launcher._launch_contract("http://127.0.0.1:19000", contract, self.manifest,
                label="policy", control_check=lambda: None), server)
        server.stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
