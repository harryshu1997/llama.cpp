#!/usr/bin/env python3
"""Gate one resident Gemma LM-head worker through Qwen OP15 fences."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import socket
import struct
import sys
import threading
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    DeviceMemoryCapacity,
    DynamicResidencySnapshot,
    DynamicWeightPlacement,
    DynamicWeightPlacementSpec,
    GpuBackfillCandidate,
    GpuBubbleWindow,
    GpuReadyChunk,
    GpuWavefrontSnapshot,
    MetricEstimate,
    ProfileBundle,
    UnifiedScheduler,
    canonical_sha256,
)


PROFILE_SCHEMA = "s42-fp16-burstgpt-gpu-wavefront-profile-v1"
RESULT_SCHEMA = "s42-fp16-burstgpt-gpu-wavefront-gate-v1"
LM_MAGIC = 0x4C484431
LM_VERSION = 1
LM_HELLO_REQUEST = 1
LM_HELLO_RESPONSE = 2
LM_EXECUTE_REQUEST = 3
LM_EXECUTE_RESPONSE = 4
LM_FLAG_F16_IO = 1
FENCE_MAGIC = 0x53343250
FENCE_VERSION = 1
FENCE_BEGIN = 1
FENCE_DONE = 2

LM_HELLO_REQUEST_STRUCT = struct.Struct("<IHHIIIHH")
LM_HELLO_RESPONSE_STRUCT = struct.Struct("<IHHHHIIIIII4xQ")
LM_EXECUTE_REQUEST_STRUCT = struct.Struct("<IHHIIII")
LM_EXECUTE_RESPONSE_STRUCT = struct.Struct("<IHHHHIIII4xQQ")
FENCE_REQUEST_STRUCT = struct.Struct("<IHHQIIIIq")
FENCE_RESPONSE_STRUCT = struct.Struct("<IHHQIIQIIqq")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def require_ascii(value: object, name: str) -> str:
    require(type(value) is str and bool(value), f"invalid {name}")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise GateError(f"non-ASCII {name}") from exc
    return value


def require_integer(
    value: object, name: str, minimum: int = 0
) -> int:
    require(type(value) is int and value >= minimum, f"invalid {name}")
    return value


def require_sha256(value: object, name: str) -> str:
    text = require_ascii(value, name).removeprefix("sha256:")
    require(
        len(text) == 64
        and all(character in "0123456789abcdef" for character in text),
        f"invalid {name}",
    )
    return "sha256:" + text


def metric_from_json(value: object, name: str) -> MetricEstimate:
    require(type(value) is dict, f"invalid {name}")
    row = value
    try:
        return MetricEstimate(
            mean=row["mean"],
            upper=row["upper"],
            lower=row.get("lower"),
            sample_count=row["sample_count"],
            measured=row["measured"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GateError(f"invalid {name}") from exc


def metric_to_json(value: MetricEstimate) -> dict[str, object]:
    return {
        "lower": value.lower,
        "mean": value.mean,
        "measured": value.measured,
        "sample_count": value.sample_count,
        "upper": value.upper,
    }


@dataclass(frozen=True)
class WavefrontProfile:
    profile_id: str
    admission: str
    gemma_model_sha256: str
    lm_head_weight_sha256: str
    worker_weight_hash64: int
    gpu_uuid: str
    gpu_total_bytes: int
    gpu_free_bytes: int
    gpu_reserve_bytes: int
    lm_head_resident_bytes: int
    workspace_bytes: int
    n_embd: int
    n_vocab: int
    offset: int
    rows: int
    top_k: int
    flags: int
    protected_ready_lower_us: int
    guard_us: int
    bubble_runtime_verified: bool
    service_latency_us: MetricEstimate
    restore_latency_us: MetricEstimate
    avoided_energy_uj: MetricEstimate
    backfill_energy_uj: MetricEstimate
    energy_boundary_id: str
    accounting_scope: str
    minimum_energy_saving_ppm: int
    producer_resource_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    valid_for_us: int
    raw: dict[str, object]

    @classmethod
    def load(cls, path: Path) -> "WavefrontProfile":
        try:
            raw = json.loads(path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GateError("cannot read wavefront profile") from exc
        require(type(raw) is dict, "invalid wavefront profile")
        require(raw.get("schema") == PROFILE_SCHEMA, "profile schema mismatch")
        model = raw.get("model")
        memory = raw.get("memory")
        worker = raw.get("worker")
        bubble = raw.get("bubble")
        candidate = raw.get("candidate")
        require(
            all(type(value) is dict for value in (model, memory, worker, bubble, candidate)),
            "profile sections are invalid",
        )
        admission = require_ascii(raw.get("admission"), "admission")
        require(admission in {"mechanics", "qualified"}, "invalid admission")
        hash64_text = require_ascii(
            worker.get("weight_hash64"), "worker weight hash"
        )
        require(
            len(hash64_text) == 16
            and all(character in "0123456789abcdef" for character in hash64_text),
            "invalid worker weight hash",
        )
        producers = raw.get("producer_resource_ids")
        evidence = raw.get("evidence_ids")
        require(type(producers) is list and bool(producers), "invalid producers")
        require(type(evidence) is list and bool(evidence), "invalid evidence")
        producer_ids = tuple(
            require_ascii(value, "producer resource") for value in producers
        )
        evidence_ids = tuple(
            require_ascii(value, "evidence id") for value in evidence
        )
        require(
            len(producer_ids) == len(set(producer_ids))
            and len(evidence_ids) == len(set(evidence_ids)),
            "duplicate profile identity",
        )
        result = cls(
            profile_id=require_ascii(raw.get("profile_id"), "profile id"),
            admission=admission,
            gemma_model_sha256=require_sha256(
                model.get("sha256"), "Gemma model SHA-256"
            ),
            lm_head_weight_sha256=require_sha256(
                model.get("lm_head_weight_sha256"),
                "LM-head weight SHA-256",
            ),
            worker_weight_hash64=int(hash64_text, 16),
            gpu_uuid=require_ascii(memory.get("gpu_uuid"), "GPU UUID"),
            gpu_total_bytes=require_integer(
                memory.get("total_bytes"), "GPU total bytes", 1
            ),
            gpu_free_bytes=require_integer(
                memory.get("free_bytes"), "GPU free bytes"
            ),
            gpu_reserve_bytes=require_integer(
                memory.get("reserve_bytes"), "GPU reserve bytes", 1
            ),
            lm_head_resident_bytes=require_integer(
                model.get("lm_head_resident_bytes"),
                "LM-head resident bytes",
                1,
            ),
            workspace_bytes=require_integer(
                candidate.get("workspace_bytes"), "workspace bytes", 1
            ),
            n_embd=require_integer(worker.get("n_embd"), "n_embd", 1),
            n_vocab=require_integer(worker.get("n_vocab"), "n_vocab", 2),
            offset=require_integer(worker.get("offset"), "offset"),
            rows=require_integer(worker.get("rows"), "rows", 1),
            top_k=require_integer(worker.get("top_k"), "top_k", 1),
            flags=require_integer(worker.get("flags"), "worker flags"),
            protected_ready_lower_us=require_integer(
                bubble.get("protected_ready_lower_us"),
                "protected-ready lower bound",
                1,
            ),
            guard_us=require_integer(bubble.get("guard_us"), "guard", 1),
            bubble_runtime_verified=bubble.get("runtime_verified"),
            service_latency_us=metric_from_json(
                candidate.get("service_latency_us"), "service latency"
            ),
            restore_latency_us=metric_from_json(
                candidate.get("restore_latency_us"), "restore latency"
            ),
            avoided_energy_uj=metric_from_json(
                candidate.get("avoided_energy_uj"), "avoided energy"
            ),
            backfill_energy_uj=metric_from_json(
                candidate.get("backfill_energy_uj"), "backfill energy"
            ),
            energy_boundary_id=require_ascii(
                candidate.get("energy_boundary_id"), "energy boundary"
            ),
            accounting_scope=require_ascii(
                candidate.get("accounting_scope"), "accounting scope"
            ),
            minimum_energy_saving_ppm=require_integer(
                candidate.get("minimum_energy_saving_ppm"),
                "minimum energy saving",
            ),
            producer_resource_ids=producer_ids,
            evidence_ids=evidence_ids,
            valid_for_us=require_integer(
                raw.get("valid_for_us"), "profile validity", 1
            ),
            raw=raw,
        )
        result.validate()
        return result

    def validate(self) -> None:
        require(
            type(self.bubble_runtime_verified) is bool,
            "invalid bubble verification",
        )
        require(
            self.gpu_free_bytes <= self.gpu_total_bytes
            and self.gpu_free_bytes >= self.gpu_reserve_bytes,
            "GPU memory reserve is not available",
        )
        require(
            self.lm_head_resident_bytes
            <= self.gpu_total_bytes - self.gpu_free_bytes,
            "LM-head placement exceeds occupied VRAM",
        )
        require(
            self.workspace_bytes
            <= self.gpu_free_bytes - self.gpu_reserve_bytes,
            "LM-head workspace exceeds free VRAM",
        )
        require(
            self.offset + self.rows == self.n_vocab
            and self.offset > 0
            and self.top_k <= self.rows
            and self.top_k <= self.offset,
            "invalid partial LM-head geometry",
        )
        require(self.flags in {0, LM_FLAG_F16_IO}, "invalid LM-head flags")
        require(
            self.protected_ready_lower_us
            > self.guard_us
            + self.service_latency_us.upper
            + self.restore_latency_us.upper,
            "calibrated protected window cannot fit the candidate",
        )
        require(
            self.minimum_energy_saving_ppm < 1_000_000,
            "invalid energy saving gate",
        )
        if self.admission == "qualified":
            metrics = (
                self.service_latency_us,
                self.restore_latency_us,
                self.avoided_energy_uj,
                self.backfill_energy_uj,
            )
            require(self.bubble_runtime_verified, "qualified bubble is unverified")
            require(
                all(value.measured and value.sample_count > 0 for value in metrics),
                "qualified profile has unmeasured metrics",
            )

    @property
    def profile_sha256(self) -> str:
        return canonical_sha256(self.raw)


@dataclass
class PendingRequest:
    packet: bytes
    request_id: int
    ready_at_us: int
    input_sha256: str
    measured: bool
    response: bytes | None = None
    error: BaseException | None = None
    completed: threading.Event = field(default_factory=threading.Event)


def fnv32(payload: bytes) -> int:
    value = 2166136261
    for byte in payload:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def recv_exact(connection: socket.socket, size: int, *, eof_ok: bool = False) -> bytes | None:
    payload = bytearray()
    while len(payload) < size:
        chunk = connection.recv(size - len(payload))
        if not chunk:
            if eof_ok and not payload:
                return None
            raise GateError("short socket message")
        payload.extend(chunk)
    return bytes(payload)


def read_receipt(path: Path, name: str) -> int:
    try:
        value = int(path.read_text(encoding="ascii").strip())
    except (OSError, UnicodeError, ValueError) as exc:
        raise GateError(f"invalid {name} receipt") from exc
    now_ns = time.monotonic_ns()
    require(0 < value <= now_ns, f"{name} receipt is in the future")
    return value


def write_monotonic_receipt(path: Path, value_ns: int) -> None:
    temporary = path.with_name(
        f".{path.name}.tmp-{threading.get_ident()}"
    )
    try:
        temporary.write_text(f"{value_ns}\n", encoding="ascii")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def scheduler_profile(profile_id: str) -> ProfileBundle:
    resources = (
        ("cpu", "cpu"),
        ("gpu-compute", "gpu"),
        ("op15-htp", "phone"),
    )
    return ProfileBundle.from_json({
        "schema": "s42-general-scheduler-profile-v1",
        "profile_id": profile_id + "-runtime",
        "resources": [
            {
                "resource_id": resource_id,
                "kind": kind,
                "capacity": 1,
                "ready": True,
                "identity": resource_id,
            }
            for resource_id, kind in resources
        ],
        "routes": [{
            "route_id": "wavefront-runtime-anchor",
            "workload_id": "wavefront-runtime-anchor",
            "granularity": "task",
            "baseline": True,
            "resource_slots": {"cpu": 1},
            "latency": {
                "cost_us": {
                    "kind": "affine_tokens_v1",
                    "fixed": 1,
                    "input_token": 0,
                    "output_token": 0,
                },
                "ucb_add_us": 0,
                "sample_count": 1,
                "measured": True,
            },
            "energy": {
                "status": "unknown",
                "cost_uj": None,
                "lower_error_ppm": 0,
                "upper_error_ppm": 0,
            },
            "overlap": {"status": "not_applicable"},
            "quality_class": "exact",
            "placement_verified": True,
            "resident": True,
            "server_busy_ppm": 1_000_000,
            "server_memory_bytes": 1,
            "evidence_ids": [profile_id],
        }],
        "trace_workload_map": {
            "wavefront-runtime-anchor": "wavefront-runtime-anchor"
        },
        "policy": {
            "energy_saving_ppm": 0,
            "latency_limit_ppm": 2_000_000,
        },
    })


class WavefrontGate:
    def __init__(self, args: argparse.Namespace, profile: WavefrontProfile):
        self.args = args
        self.profile = profile
        self.worker: socket.socket | None = None
        self.client_listener: socket.socket | None = None
        self.client: socket.socket | None = None
        self.fence_listener: socket.socket | None = None
        self.fence: socket.socket | None = None
        self.client_thread: threading.Thread | None = None
        self.condition = threading.Condition()
        self.pending: PendingRequest | None = None
        self.client_done = False
        self.client_error: BaseException | None = None
        self.handshake: dict[str, int] | None = None
        self.scheduler: UnifiedScheduler | None = None
        self.completed_receipts: list[str] = []
        self.events: list[dict[str, object]] = []
        self.fence_calls = 0
        self.last_fence_sequence = 0
        self.backfills = 0
        self.warmup_calls = 0
        self.tail_calls = 0
        self.rejections = 0
        self.admission_disabled_reason: str | None = None
        self.qwen_complete_ns: int | None = None
        self.paid_tail_ns: int | None = None
        self.started_ns = time.monotonic_ns()
        self.last_progress_ns = self.started_ns

    def log_event(self, event: dict[str, object]) -> None:
        event = {
            "at_ns": time.monotonic_ns(),
            **event,
        }
        self.events.append(event)
        print(json.dumps(event, ensure_ascii=True, sort_keys=True), flush=True)
        self.last_progress_ns = event["at_ns"]

    def setup(self) -> None:
        self.worker = socket.create_connection(
            (self.args.worker_host, self.args.worker_port),
            timeout=self.args.connect_timeout_s,
        )
        self.worker.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.worker.settimeout(None)

        self.client_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client_listener.bind((self.args.listen_host, self.args.listen_port))
        self.client_listener.listen(1)
        self.client_listener.settimeout(0.05)

        self.fence_listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.fence_listener.bind(str(self.args.fence_socket))
        self.fence_listener.listen(1)
        self.fence_listener.settimeout(0.05)
        self.log_event({
            "event": "ready",
            "fence_socket": str(self.args.fence_socket),
            "listen_port": self.args.listen_port,
            "mode": self.args.mode,
            "profile_sha256": self.profile.profile_sha256,
        })

    def start_client(self) -> None:
        if self.client_thread is not None:
            return
        try:
            self.client, _ = self.client_listener.accept()
        except TimeoutError:
            return
        self.client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.client.settimeout(None)
        self.client_thread = threading.Thread(
            target=self.client_loop,
            name="gemma-lm-head-client",
            daemon=True,
        )
        self.client_thread.start()

    def client_loop(self) -> None:
        try:
            assert self.client is not None and self.worker is not None
            hello_packet = recv_exact(self.client, LM_HELLO_REQUEST_STRUCT.size)
            assert hello_packet is not None
            hello = LM_HELLO_REQUEST_STRUCT.unpack(hello_packet)
            require(
                hello[:3] == (LM_MAGIC, LM_VERSION, LM_HELLO_REQUEST),
                "invalid LM-head client hello",
            )
            require(
                hello[3] == self.profile.n_embd
                and hello[4] == self.profile.rows
                and hello[5] == self.profile.top_k
                and hello[6] == self.profile.flags
                and hello[7] == 0,
                "LM-head client geometry differs from the profile",
            )
            self.worker.sendall(hello_packet)
            response_packet = recv_exact(
                self.worker, LM_HELLO_RESPONSE_STRUCT.size
            )
            assert response_packet is not None
            response = LM_HELLO_RESPONSE_STRUCT.unpack(response_packet)
            require(
                response[:3] == (LM_MAGIC, LM_VERSION, LM_HELLO_RESPONSE)
                and response[3] == 0,
                "LM-head worker rejected the hello",
            )
            require(
                response[4] == self.profile.flags
                and response[5] == self.profile.n_embd
                and response[6] == self.profile.n_vocab
                and response[7] == self.profile.offset
                and response[8] == self.profile.rows
                and response[10] == self.profile.top_k
                and response[11] == self.profile.worker_weight_hash64,
                "LM-head worker identity differs from the profile",
            )
            self.handshake = {
                "flags": response[4],
                "n_embd": response[5],
                "n_vocab": response[6],
                "offset": response[7],
                "rows": response[8],
                "top_k": response[10],
                "weight_hash64": response[11],
                "weight_type": response[9],
            }
            self.build_scheduler()
            self.client.sendall(response_packet)
            self.log_event({"event": "client_connected", **self.handshake})

            input_bytes = self.profile.n_embd * (
                2 if self.profile.flags & LM_FLAG_F16_IO else 4
            )
            while True:
                header_packet = recv_exact(
                    self.client, LM_EXECUTE_REQUEST_STRUCT.size, eof_ok=True
                )
                if header_packet is None:
                    break
                header = LM_EXECUTE_REQUEST_STRUCT.unpack(header_packet)
                payload = recv_exact(self.client, input_bytes)
                assert payload is not None
                require(
                    header[:3] == (LM_MAGIC, LM_VERSION, LM_EXECUTE_REQUEST)
                    and header[3] > 0
                    and header[4] == self.profile.n_embd
                    and header[5] == input_bytes
                    and header[6] == fnv32(payload),
                    "invalid LM-head execute request",
                )
                pending = PendingRequest(
                    packet=header_packet + payload,
                    request_id=header[3],
                    ready_at_us=time.monotonic_ns() // 1000,
                    input_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
                    measured=self.args.arm_file.exists(),
                )
                with self.condition:
                    require(self.pending is None, "more than one LM-head input is pending")
                    self.pending = pending
                    self.condition.notify_all()
                pending.completed.wait()
                if pending.error is not None:
                    raise pending.error
                require(pending.response is not None, "LM-head response is absent")
                self.client.sendall(pending.response)
        except BaseException as exc:  # pylint: disable=broad-except
            self.client_error = exc
        finally:
            with self.condition:
                self.client_done = True
                self.condition.notify_all()

    def build_scheduler(self) -> None:
        require(self.handshake is not None, "worker handshake is absent")
        now_us = time.monotonic_ns() // 1000
        valid_until_us = now_us + self.profile.valid_for_us
        occupied_bytes = self.profile.gpu_total_bytes - self.profile.gpu_free_bytes
        spec = DynamicWeightPlacementSpec(
            placement_id="gemma-lm-head-vram",
            slice_id=f"gemma-lm-head-rows-{self.profile.offset}-{self.profile.n_vocab}",
            model_id="gemma4-12b-f16",
            model_hash=self.profile.gemma_model_sha256,
            weight_hash=self.profile.lm_head_weight_sha256,
            resource_id="gpu-vram",
            resident_bytes=self.profile.lm_head_resident_bytes,
            execution_resource_ids=("gpu-compute",),
            runtime_binding_ids=(
                f"lm-head-worker-fnv64-{self.profile.worker_weight_hash64:016x}",
                self.profile.gpu_uuid,
            ),
            evidence_ids=self.profile.evidence_ids,
        )
        placement = DynamicWeightPlacement(
            spec=spec,
            generation=1,
            resident_since_us=now_us,
            minimum_resident_until_us=valid_until_us,
        )
        identity = {
            "gpu_uuid": self.profile.gpu_uuid,
            "placement": spec.to_json(),
            "profile_sha256": self.profile.profile_sha256,
            "schema": "s42-gpu-wavefront-residency-epoch-v1",
        }
        snapshot = DynamicResidencySnapshot(
            snapshot_id=self.profile.profile_id + "-residency",
            epoch_key=canonical_sha256(identity),
            generation=1,
            captured_at_us=now_us,
            valid_until_us=valid_until_us,
            memory={
                "gpu-vram": DeviceMemoryCapacity(
                    resource_id="gpu-vram",
                    capacity_bytes=self.profile.gpu_total_bytes,
                    occupied_bytes=occupied_bytes,
                    reserve_bytes=self.profile.gpu_reserve_bytes,
                )
            },
            placements={spec.placement_id: placement},
        )
        self.scheduler = UnifiedScheduler(
            (scheduler_profile(self.profile.profile_id),),
            "control",
            dynamic_residency_snapshot=snapshot,
        )

    def accept_fence(self) -> None:
        if self.fence is not None:
            return
        try:
            self.fence, _ = self.fence_listener.accept()
        except TimeoutError:
            return
        self.fence.settimeout(0.05)
        self.log_event({"event": "fence_connected"})

    def take_pending(self) -> PendingRequest | None:
        with self.condition:
            value = self.pending
            self.pending = None
            return value

    def peek_pending(self) -> PendingRequest | None:
        with self.condition:
            return self.pending

    def finish_pending(
        self, pending: PendingRequest, response: bytes | None, error: BaseException | None
    ) -> None:
        pending.response = response
        pending.error = error
        pending.completed.set()

    def execute_worker(self, pending: PendingRequest) -> tuple[bytes, dict[str, int]]:
        assert self.worker is not None
        self.worker.sendall(pending.packet)
        response_bytes = LM_EXECUTE_RESPONSE_STRUCT.size + self.profile.top_k * 8
        response_packet = recv_exact(self.worker, response_bytes)
        assert response_packet is not None
        header = LM_EXECUTE_RESPONSE_STRUCT.unpack(
            response_packet[:LM_EXECUTE_RESPONSE_STRUCT.size]
        )
        candidates = response_packet[LM_EXECUTE_RESPONSE_STRUCT.size:]
        require(
            header[:3] == (LM_MAGIC, LM_VERSION, LM_EXECUTE_RESPONSE)
            and header[3] == 0
            and header[4] == 0
            and header[5] == pending.request_id
            and header[6] == self.profile.top_k
            and header[7] == len(candidates)
            and header[8] == fnv32(candidates),
            "invalid LM-head worker response",
        )
        return response_packet, {
            "compute_us": header[9],
            "reduce_us": header[10],
            "response_hash": header[8],
        }

    def execute_direct(self, pending: PendingRequest, route: str) -> tuple[int, int]:
        start_ns = time.monotonic_ns()
        try:
            response, metrics = self.execute_worker(pending)
        except BaseException as exc:  # pylint: disable=broad-except
            self.finish_pending(pending, None, exc)
            raise
        end_ns = time.monotonic_ns()
        self.finish_pending(pending, response, None)
        if route == "warmup":
            self.warmup_calls += 1
        else:
            self.tail_calls += 1
        self.log_event({
            "duration_us": (end_ns - start_ns) // 1000,
            "event": "worker_execute",
            "request_id": pending.request_id,
            "route": route,
            **metrics,
        })
        return start_ns, end_ns

    def schedule_pending(
        self, pending: PendingRequest, fence: tuple[int, ...]
    ) -> tuple[object, dict[str, object]]:
        assert self.scheduler is not None
        snapshot = self.scheduler.dynamic_residency_snapshot
        assert snapshot is not None
        now_us = time.monotonic_ns() // 1000
        protected_ready_us = fence[8] // 1000 + self.profile.protected_ready_lower_us
        bubble = GpuBubbleWindow(
            bubble_id=f"qwen-op15-fence-{fence[3]}",
            fence_receipt_id=f"qwen-op15-fence-receipt-{fence[3]}",
            source_snapshot_id=snapshot.snapshot_id,
            source_snapshot_sha256=canonical_sha256(snapshot.to_json()),
            source_generation=snapshot.generation,
            source_epoch_key=snapshot.epoch_key,
            gpu_resource_id="gpu-compute",
            protected_owner_id=f"qwen-request-{fence[4]}",
            protected_model_id="qwen3-14b-f16",
            captured_at_us=now_us,
            valid_until_us=protected_ready_us,
            protected_ready_lower_us=protected_ready_us,
            guard_us=self.profile.guard_us,
            runtime_verified=self.profile.bubble_runtime_verified,
            evidence_ids=self.profile.evidence_ids,
        )
        sequence_index = len(self.completed_receipts)
        chunk_id = (
            f"gemma-lm-head-{sequence_index}-{pending.request_id}-"
            f"{pending.input_sha256[-12:]}"
        )
        candidate = GpuBackfillCandidate(
            candidate_id=chunk_id,
            work_id=chunk_id + "-work",
            model_id="gemma4-12b-f16",
            gpu_resource_id="gpu-compute",
            workspace_resource_id="gpu-vram",
            workspace_bytes=self.profile.workspace_bytes,
            required_placement_ids=("gemma-lm-head-vram",),
            additional_resource_ids=self.profile.producer_resource_ids,
            deadline_us=protected_ready_us - self.profile.guard_us,
            energy_boundary_id=self.profile.energy_boundary_id,
            accounting_scope=self.profile.accounting_scope,
            service_latency_us=self.profile.service_latency_us,
            restore_latency_us=self.profile.restore_latency_us,
            avoided_energy_uj=self.profile.avoided_energy_uj,
            backfill_energy_uj=self.profile.backfill_energy_uj,
            evidence_ids=self.profile.evidence_ids,
        )
        predecessor = self.completed_receipts[-1] if self.completed_receipts else None
        chunk = GpuReadyChunk(
            chunk_id=chunk_id,
            pipeline_id=self.args.pipeline_id,
            model_id="gemma4-12b-f16",
            sequence_index=sequence_index,
            layer_start=48,
            layer_end=49,
            token_count=1,
            ready_receipt_id=chunk_id + "-ready",
            input_buffer_id=chunk_id + "-input",
            input_buffer_sha256=pending.input_sha256,
            predecessor_output_receipt_id=predecessor,
            ready_at_us=pending.ready_at_us,
            valid_until_us=protected_ready_us,
            producer_resource_ids=self.profile.producer_resource_ids,
            runtime_verified=pending.measured,
            candidate=candidate,
            evidence_ids=self.profile.evidence_ids,
        )
        wavefront = GpuWavefrontSnapshot(
            wavefront_id=f"wavefront-fence-{fence[3]}",
            source_snapshot_id=snapshot.snapshot_id,
            source_snapshot_sha256=canonical_sha256(snapshot.to_json()),
            source_generation=snapshot.generation,
            source_epoch_key=snapshot.epoch_key,
            captured_at_us=now_us,
            valid_until_us=protected_ready_us,
            next_sequence_by_pipeline={self.args.pipeline_id: sequence_index},
            completed_output_receipt_ids=tuple(self.completed_receipts),
            ready_chunks=(chunk,),
        )
        schedule = self.scheduler.schedule_gpu_wavefront_backfill(
            bubble,
            wavefront,
            now_us=now_us,
            minimum_energy_saving_ppm=self.profile.minimum_energy_saving_ppm,
            require_measured=self.args.mode == "qualified",
            objective="coverage_then_energy",
        )
        return schedule, {
            "bubble": bubble.to_json(),
            "decision": schedule.decision.to_json(),
            "wavefront": wavefront.to_json(),
        }

    def handle_fence(self, packet: bytes) -> None:
        require(len(packet) == FENCE_REQUEST_STRUCT.size, "short fence request")
        fence = FENCE_REQUEST_STRUCT.unpack(packet)
        require(
            fence[:3] == (FENCE_MAGIC, FENCE_VERSION, FENCE_BEGIN)
            and fence[3] == self.last_fence_sequence + 1
            and fence[4] > 0
            and fence[6] > 0
            and fence[7] == 0
            and 0 < fence[8] <= time.monotonic_ns(),
            "invalid fence request",
        )
        self.last_fence_sequence = fence[3]
        self.fence_calls += 1
        start_ns = time.monotonic_ns()
        selected = False
        schedule_receipt: dict[str, object] | None = None
        pending = self.peek_pending()
        eligible = (
            self.args.mode in {"mechanics", "qualified"}
            and self.qwen_complete_ns is None
            and self.admission_disabled_reason is None
            and pending is not None
            and pending.measured
            and (self.args.max_backfills == 0 or self.backfills < self.args.max_backfills)
        )
        if eligible:
            assert pending is not None
            try:
                schedule, schedule_receipt = self.schedule_pending(pending, fence)
            except BaseException as exc:  # pylint: disable=broad-except
                self.admission_disabled_reason = str(exc)
                self.log_event({
                    "error": str(exc),
                    "event": "wavefront_scheduler_failure",
                    "request_id": pending.request_id,
                })
                end_ns = time.monotonic_ns()
            else:
                if schedule.decision.chunk_id is not None:
                    require(self.take_pending() is pending, "pending request changed")
                    selected = True
                    worker_response = None
                    try:
                        worker_response, metrics = self.execute_worker(pending)
                        end_ns = time.monotonic_ns()
                        output_receipt = (
                            f"gemma-lm-head-output-{len(self.completed_receipts)}-"
                            f"{metrics['response_hash']:08x}"
                        )
                        self.scheduler.release_gpu_wavefront_backfill(
                            schedule, end_ns // 1000, output_receipt
                        )
                        self.completed_receipts.append(output_receipt)
                        self.backfills += 1
                        self.finish_pending(pending, worker_response, None)
                        self.log_event({
                            "decision_sha256": schedule.decision.decision_sha256,
                            "duration_us": (end_ns - start_ns) // 1000,
                            "event": "wavefront_execute",
                            "output_receipt_id": output_receipt,
                            "request_id": pending.request_id,
                            **metrics,
                        })
                    except BaseException as exc:  # pylint: disable=broad-except
                        self.admission_disabled_reason = str(exc)
                        self.finish_pending(
                            pending,
                            worker_response,
                            None if worker_response is not None else exc,
                        )
                        self.log_event({
                            "error": str(exc),
                            "event": "wavefront_contract_violation",
                            "request_id": pending.request_id,
                        })
                        end_ns = time.monotonic_ns()
                else:
                    self.rejections += 1
                    self.log_event({
                        "decision": schedule.decision.to_json(),
                        "event": "wavefront_rejected",
                        "request_id": pending.request_id,
                    })
                    end_ns = time.monotonic_ns()
        else:
            end_ns = time.monotonic_ns()

        assert self.fence is not None
        response = FENCE_RESPONSE_STRUCT.pack(
            FENCE_MAGIC,
            FENCE_VERSION,
            FENCE_DONE,
            fence[3],
            fence[4],
            0,
            0,
            0,
            0,
            start_ns,
            end_ns,
        )
        self.fence.sendall(response)
        self.log_event({
            "event": "fence_done",
            "fence_sequence": fence[3],
            "filler_selected": selected,
            "filler_us": (end_ns - start_ns) // 1000,
            "request_id": fence[4],
            "schedule": schedule_receipt,
            "tokens": fence[6],
        })

    def poll_fence(self) -> None:
        if self.fence is None:
            return
        try:
            packet = self.fence.recv(FENCE_REQUEST_STRUCT.size + 1)
        except TimeoutError:
            return
        if not packet:
            self.fence.close()
            self.fence = None
            self.log_event({"event": "fence_disconnected"})
            return
        self.handle_fence(packet)

    def poll_qwen_complete(self) -> None:
        if self.qwen_complete_ns is None and self.args.qwen_complete_file.exists():
            self.qwen_complete_ns = read_receipt(
                self.args.qwen_complete_file, "Qwen complete"
            )
            self.log_event({
                "event": "qwen_complete",
                "qwen_complete_ns": self.qwen_complete_ns,
            })

    def drain_ready(self) -> None:
        pending = self.peek_pending()
        if pending is None:
            return
        if not pending.measured:
            route = "warmup"
        elif self.qwen_complete_ns is not None:
            route = "tail"
        else:
            return
        require(self.take_pending() is pending, "pending request changed")
        self.execute_direct(pending, route)

    def complete_tail(self) -> bool:
        with self.condition:
            done = self.client_done and self.pending is None
        if self.qwen_complete_ns is None or not done:
            return False
        if self.client_error is not None:
            raise self.client_error
        self.paid_tail_ns = time.monotonic_ns()
        write_monotonic_receipt(
            self.args.paid_tail_file, self.paid_tail_ns
        )
        self.log_event({
            "event": "paid_tail_complete",
            "paid_tail_ns": self.paid_tail_ns,
        })
        return True

    def run(self) -> None:
        self.setup()
        timeout_ns = int(self.args.timeout_s * 1e9)
        while True:
            self.start_client()
            self.accept_fence()
            self.poll_qwen_complete()
            self.drain_ready()
            self.poll_fence()
            if self.complete_tail():
                return
            if self.client_error is not None:
                raise self.client_error
            if time.monotonic_ns() - self.last_progress_ns > timeout_ns:
                raise GateError("wavefront gate timed out")
            time.sleep(0.001)

    def result(self, status: str, error: str | None) -> dict[str, object]:
        return {
            "admission_disabled_reason": self.admission_disabled_reason,
            "backfills": self.backfills,
            "client_handshake": self.handshake,
            "energy_claim_eligible": False,
            "error": error,
            "events": self.events,
            "fence_calls": self.fence_calls,
            "finished_ns": time.monotonic_ns(),
            "mode": self.args.mode,
            "paid_tail_ns": self.paid_tail_ns,
            "profile_admission": self.profile.admission,
            "profile_sha256": self.profile.profile_sha256,
            "qwen_complete_ns": self.qwen_complete_ns,
            "rejections": self.rejections,
            "schema": RESULT_SCHEMA,
            "started_ns": self.started_ns,
            "status": status,
            "tail_calls": self.tail_calls,
            "warmup_calls": self.warmup_calls,
        }

    def close(self) -> None:
        for connection in (
            self.client,
            self.client_listener,
            self.fence,
            self.fence_listener,
            self.worker,
        ):
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
        try:
            self.args.fence_socket.unlink()
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "mechanics", "qualified"), required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--worker-host", default="127.0.0.1")
    parser.add_argument("--worker-port", type=int, required=True)
    parser.add_argument("--fence-socket", type=Path, required=True)
    parser.add_argument("--arm-file", type=Path, required=True)
    parser.add_argument("--qwen-complete-file", type=Path, required=True)
    parser.add_argument("--paid-tail-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline-id", default="gemma-request-50")
    parser.add_argument("--max-backfills", type=int, default=0)
    parser.add_argument("--connect-timeout-s", type=float, default=30.0)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    args = parser.parse_args()
    for name in (
        "profile",
        "fence_socket",
        "arm_file",
        "qwen_complete_file",
        "paid_tail_file",
        "output",
    ):
        path = getattr(args, name)
        if not path.is_absolute():
            parser.error(f"{name.replace('_', ' ')} must be absolute")
    if not args.profile.is_file():
        parser.error("profile must be an existing file")
    for path in (
        args.fence_socket,
        args.arm_file,
        args.qwen_complete_file,
        args.paid_tail_file,
        args.output,
    ):
        if path.exists() or not path.parent.is_dir():
            parser.error("runtime paths must be unused and have existing parents")
    if not (0 < args.listen_port <= 65535 and 0 < args.worker_port <= 65535):
        parser.error("ports must be in range")
    if args.max_backfills < 0 or args.connect_timeout_s <= 0 or args.timeout_s <= 0:
        parser.error("timeouts and maximum backfills are invalid")
    if args.mode == "mechanics" and args.max_backfills == 0:
        parser.error("mechanics mode requires a bounded maximum backfill count")
    return args


def write_result(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def main() -> int:
    args = parse_args()
    try:
        profile = WavefrontProfile.load(args.profile)
    except GateError as exc:
        print(f"wavefront profile error: {exc}", file=sys.stderr)
        return 2
    if args.mode == "qualified" and profile.admission != "qualified":
        print("qualified mode requires a qualified profile", file=sys.stderr)
        return 2
    gate = WavefrontGate(args, profile)
    status = "PASS"
    error = None
    result_code = 0
    try:
        gate.run()
        if gate.admission_disabled_reason is not None:
            raise GateError(gate.admission_disabled_reason)
    except BaseException as exc:  # pylint: disable=broad-except
        status = "FAIL"
        error = str(exc)
        result_code = 1
        print(f"wavefront gate failed: {exc}", file=sys.stderr)
    finally:
        write_result(args.output, gate.result(status, error))
        gate.close()
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
