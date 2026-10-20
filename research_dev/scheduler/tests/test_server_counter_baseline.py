"""A request re-executed on the same llama-server inherits its FFN runtime counters.

llama-server keys ``ffn_split_stats`` by request id. Fault run p6g1flt (2026-10-04): request 003 was
recovered on the same server after the Pixel loss (fraction 0), and its first window reported the
failed attempt's thousands of phone calls, which the placement accounting rejects for a fraction-0
binding ("physical helper work lacks an active attachment") -> the run ended. The backend now
remembers which (endpoint, request id) pairs own counters on a server and treats the first snapshot
of a re-execution as the baseline, recording what it inherited.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for path in (str(REPO_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from research_dev.scheduler import GGUFModelManifestLoader  # noqa: E402
from research_dev.scheduler._internal.adaptive_decode_contracts import (  # noqa: E402
    AdaptiveDecodeDirective,
    AdaptiveDecodePolicy,
    AdaptiveDecodeWindowBoundary,
)
from research_dev.scheduler.adapters import (  # noqa: E402
    CanonicalHttpExecutionBackend,
    LlamaCppCompletionPayload,
)
import test_llama_server_adapter as base  # noqa: E402

RUNTIME_NAMES = ("batched_calls", "calls", "configured_queue_depth", "desktop_compute_us", "download_bytes",
                 "exposed_tail_us", "input_rows", "maximum_active_slots", "maximum_outstanding_transfers",
                 "maximum_tokens", "phone_compute_us", "rpc_us", "transfer_subrequests", "upload_bytes",
                 "usb_d2h_us", "usb_h2d_us", "usb_transfer_us", "useful_overlap_us")


def _baseline_policy():
    return AdaptiveDecodePolicy(
        route_id="synthetic:desktop-parent", executor_id="synthetic:desktop",
        operator_plan_sha256=base.PLAN_SHA256, desktop_parent_route_id="synthetic:desktop-parent",
        desktop_placement_sha256="sha256:" + "4" * 64, layer_indices=(), layer_mask=0, columns=0,
        split_fraction_ppm=0, resource_ids=("compute:accelerator-b",), baseline=True)


class _Client(base.FakeClient):
    """The server's counters for the request id are cumulative across executions, as in llama-server."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.calls = 0

    def complete(self, endpoint, payload, control_check, *, scheduler_headers=None):
        control_check()
        first_ns = time.monotonic_ns()
        payload.on_active_batch(2)
        payload.on_decode_progress(2, 1, first_ns, False)
        self._slots_probe = lambda *_args: [{"id": 2, "id_task": 1, "is_processing": True}]
        payload.on_decode_progress(2, 2, first_ns + 1_000_000, False)
        return {"stream_sha256": "3" * 64, "tokens": [1]}

    def read_ffn_stats(self, *_args, **_kwargs):
        stats = dict.fromkeys(RUNTIME_NAMES, 1)
        stats.update(calls=self.calls, configured_queue_depth=4, input_rows=self.calls)
        return ({"applied_token_index": 2, "plan_generation": 1, "policy_hash": self.policy.policy_hash,
                 "runtime_stats": stats, "slot_id": 2, "success": True}, time.monotonic_ns())


class _Scheduler:
    def __init__(self, policy):
        self.policy = policy
        self.started_at_us = None
        self.observation = None

    def start_adaptive_decode(self, _request_id, **values):
        self.started_at_us = values["at_us"]
        return AdaptiveDecodeDirective(state="BASELINE", reason="WINDOW_OPENED", target_token_index=2)

    def adaptive_decode_boundary(self, request_id, **values):
        boundary = AdaptiveDecodeWindowBoundary(
            request_id=request_id, slot_id=values["slot_id"], window_index=0, token_start=1,
            token_end=values["token_index"], started_at_us=self.started_at_us,
            finished_at_us=max(self.started_at_us + 1, values["at_us"]), policy=self.policy, applied_ack=None)
        return AdaptiveDecodeDirective(state="BASELINE", reason="WINDOW_MEASUREMENT_REQUIRED",
                                       target_token_index=None, boundary=boundary)

    def seal_adaptive_decode_tail(self, *_args, **_kwargs):
        return None

    def record_adaptive_decode_window(self, _request_id, _boundary, observation):
        self.observation = observation

    def acknowledge_adaptive_decode_control(self, *_args, **_kwargs):
        raise AssertionError("control acknowledgement was requested")

    def fail_adaptive_decode_control(self, *_args, **_kwargs):
        raise AssertionError("control failure was requested")

    def preview_adaptive_decode_completion(self, _request_id):
        return self.completed

    def complete_adaptive_decode(self, _request_id):
        return self.completed

    class completed:  # noqa: N801 - the adapter reads one attribute and to_json()
        grouped_observation_sha256 = "sha256:" + "5" * 64

        @staticmethod
        def to_json():
            return {"grouped_observation_sha256": "sha256:" + "5" * 64}


class ServerCounterBaselineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        model = root / "model.gguf"
        base.write_synthetic_gguf(model)
        manifest = GGUFModelManifestLoader.load("synthetic-model-id", model)
        self.command = base.adaptive_execution_command(base.execution_command(manifest.artifact_sha256))
        self.policy = _baseline_policy()
        self.client = _Client(self.policy)
        self.scheduler = _Scheduler(self.policy)
        self.backend = CanonicalHttpExecutionBackend(
            self.client, base.FakeMeter(), epoch_ns=time.monotonic_ns() - 1_000_000)
        self.backend.bind_scheduler(self.scheduler)
        self.root = root

    def execute(self, command, name, server_calls):
        self.client.calls = server_calls
        payload = LlamaCppCompletionPayload(
            request_id=command.request_id, expected_model_alias="synthetic-model", input_tokens=1,
            output_tokens=4, prompt_tokens=(1,), seed=1, stream_path=self.root / (name + ".raw"),
            on_first_token=lambda _value: None)
        result = self.backend.execute(command, payload, lambda: None)
        return self.scheduler.observation.completed_phone_calls, result.payload.get("server_counter_baselines")

    def test_a_re_execution_on_the_same_server_baselines_the_inherited_counters(self):
        # first attempt: 3 phone calls, counted
        self.assertEqual(self.execute(self.command, "first", 3), (3, None))
        # the attempt failed; the same request id runs again on the same server, which still reports its
        # 3 calls (plus none of its own, fraction 0): the first snapshot is the baseline, not work
        calls, baselines = self.execute(self.command, "second", 3)
        self.assertEqual(calls, 0)
        (record,) = baselines
        self.assertEqual((record["kind"], record["request_id"], record["ticket_id"],
                          record["inherited_counters"]["calls"], record["endpoint"]),
                         ("server_counters_baselined", self.command.request_id, self.command.ticket_id, 3,
                          self.command.endpoint))
        self.assertNotIn("configured_queue_depth", record["inherited_counters"])
        # a third execution on the same server baselines again (the counters are still there)
        self.assertEqual(self.execute(self.command, "third", 5)[0], 0)

    def test_another_server_starts_from_zero_counters(self):
        self.assertEqual(self.execute(self.command, "first", 2), (2, None))
        elsewhere = replace(self.command, endpoint="http://127.0.0.1:19999")
        self.assertEqual(self.execute(elsewhere, "second", 4), (4, None))


if __name__ == "__main__":
    unittest.main()
