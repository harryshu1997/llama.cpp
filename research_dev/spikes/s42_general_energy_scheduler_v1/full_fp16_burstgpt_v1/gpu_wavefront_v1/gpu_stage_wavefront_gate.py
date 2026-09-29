#!/usr/bin/env python3
"""Gate one resident Gemma layer stage through Qwen OP15 GPU fences."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
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
    UnifiedScheduler,
    canonical_sha256,
)
from gpu_wavefront_gate import (  # noqa: E402
    FENCE_BEGIN,
    FENCE_DONE,
    FENCE_MAGIC,
    FENCE_REQUEST_STRUCT,
    FENCE_RESPONSE_STRUCT,
    FENCE_VERSION,
    GateError,
    metric_from_json,
    read_receipt,
    recv_exact,
    require,
    require_ascii,
    require_integer,
    require_sha256,
    scheduler_profile,
    write_monotonic_receipt,
)


PROFILE_SCHEMA = "s42-fp16-burstgpt-gpu-stage-wavefront-profile-v1"
RESULT_SCHEMA = "s42-fp16-burstgpt-gpu-stage-wavefront-gate-v1"

STAGE_STOP = -1
STAGE_RESET = -2
STAGE_BATCH_PREFILL = -4
STAGE_BATCH_DECODE = -5
STAGE_HELLO = -6
STAGE_DETACH = -7
STAGE_V3_HELLO = -8
STAGE_V3_BATCH = -9
STAGE_V3_IDENTITY = -13
STAGE_V3_MAGIC = 0x4C535633
STAGE_V3_VERSION = 3
STAGE_IDENTITY_MAGIC = 0x4C534944
STAGE_IDENTITY_VERSION = 1
STAGE_V3_CAP_BATCH = 1 << 0
STAGE_V3_CAP_TERMINAL = 1 << 4
STAGE_V3_CAP_IDENTITY = 1 << 5

I32 = struct.Struct("<i")
LEGACY_HELLO_STRUCT = struct.Struct("<10i")
V3_HELLO_STRUCT = struct.Struct("<11i")
IDENTITY_STRUCT = struct.Struct("<3i")
PREFILL_HEADER_STRUCT = struct.Struct("<4i")
DECODE_HEADER_STRUCT = struct.Struct("<3i")
RESPONSE_HEADER_STRUCT = struct.Struct("<2i")
V3_BATCH_HEADER_STRUCT = struct.Struct("<4i")
V3_RESPONSE_HEADER_STRUCT = struct.Struct("<3i")


@dataclass(frozen=True)
class StageWavefrontProfile:
    profile_id: str
    admission: str
    gemma_model_sha256: str
    stage_weight_sha256: str
    gpu_uuid: str
    gpu_total_bytes: int
    gpu_free_bytes: int
    gpu_reserve_bytes: int
    resident_bytes: int
    workspace_bytes: int
    prefill_chunk_rows: int
    n_layer: int
    n_embd: int
    file_type: int
    layer_start: int
    layer_end: int
    protected_ready_lower_us: int
    guard_us: int
    bubble_runtime_verified: bool
    prefill_service_latency_us: MetricEstimate
    decode_service_latency_us: MetricEstimate
    restore_latency_us: MetricEstimate
    avoided_energy_uj: MetricEstimate
    backfill_energy_uj: MetricEstimate
    energy_boundary_id: str
    accounting_scope: str
    minimum_energy_saving_ppm: int
    producer_resource_ids: tuple[str, ...]
    protected_resource_ids: tuple[str, ...]
    fenced_resource_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    valid_for_us: int
    raw: dict[str, object]

    @classmethod
    def load(cls, path: Path) -> "StageWavefrontProfile":
        try:
            raw = json.loads(path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GateError("cannot read GPU-stage profile") from exc
        require(type(raw) is dict, "invalid GPU-stage profile")
        require(raw.get("schema") == PROFILE_SCHEMA, "profile schema mismatch")
        model = raw.get("model")
        memory = raw.get("memory")
        bubble = raw.get("bubble")
        candidate = raw.get("candidate")
        require(
            all(type(value) is dict for value in (model, memory, bubble, candidate)),
            "profile sections are invalid",
        )
        admission = require_ascii(raw.get("admission"), "admission")
        require(admission in {"mechanics", "qualified"}, "invalid admission")
        producers = raw.get("producer_resource_ids")
        protected = raw.get("protected_resource_ids")
        fenced = raw.get("fenced_resource_ids")
        evidence = raw.get("evidence_ids")
        require(type(producers) is list and bool(producers), "invalid producers")
        require(
            type(protected) is list and bool(protected),
            "invalid protected resources",
        )
        require(type(fenced) is list, "invalid fenced resources")
        require(type(evidence) is list and bool(evidence), "invalid evidence")
        producer_ids = tuple(
            require_ascii(value, "producer resource") for value in producers
        )
        evidence_ids = tuple(
            require_ascii(value, "evidence id") for value in evidence
        )
        protected_ids = tuple(
            require_ascii(value, "protected resource") for value in protected
        )
        fenced_ids = tuple(
            require_ascii(value, "fenced resource") for value in fenced
        )
        require(
            len(producer_ids) == len(set(producer_ids))
            and len(protected_ids) == len(set(protected_ids))
            and len(fenced_ids) == len(set(fenced_ids))
            and len(evidence_ids) == len(set(evidence_ids)),
            "duplicate profile identity",
        )
        result = cls(
            profile_id=require_ascii(raw.get("profile_id"), "profile id"),
            admission=admission,
            gemma_model_sha256=require_sha256(model.get("sha256"), "model SHA-256"),
            stage_weight_sha256=require_sha256(
                model.get("stage_weight_sha256"), "stage weight SHA-256"
            ),
            gpu_uuid=require_ascii(memory.get("gpu_uuid"), "GPU UUID"),
            gpu_total_bytes=require_integer(memory.get("total_bytes"), "GPU total", 1),
            gpu_free_bytes=require_integer(memory.get("free_bytes"), "GPU free"),
            gpu_reserve_bytes=require_integer(memory.get("reserve_bytes"), "GPU reserve", 1),
            resident_bytes=require_integer(model.get("resident_bytes"), "resident bytes", 1),
            workspace_bytes=require_integer(candidate.get("workspace_bytes"), "workspace", 1),
            prefill_chunk_rows=require_integer(
                candidate.get("prefill_chunk_rows"), "prefill chunk rows", 1
            ),
            n_layer=require_integer(model.get("n_layer"), "layer count", 1),
            n_embd=require_integer(model.get("n_embd"), "embedding width", 1),
            file_type=require_integer(model.get("file_type"), "file type"),
            layer_start=require_integer(model.get("layer_start"), "layer start"),
            layer_end=require_integer(model.get("layer_end"), "layer end", 1),
            protected_ready_lower_us=require_integer(
                bubble.get("protected_ready_lower_us"), "protected-ready lower", 1
            ),
            guard_us=require_integer(bubble.get("guard_us"), "guard", 1),
            bubble_runtime_verified=bubble.get("runtime_verified"),
            prefill_service_latency_us=metric_from_json(
                candidate.get("prefill_service_latency_us"), "prefill service latency"
            ),
            decode_service_latency_us=metric_from_json(
                candidate.get("decode_service_latency_us"), "decode service latency"
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
                candidate.get("minimum_energy_saving_ppm"), "minimum saving"
            ),
            producer_resource_ids=producer_ids,
            protected_resource_ids=protected_ids,
            fenced_resource_ids=fenced_ids,
            evidence_ids=evidence_ids,
            valid_for_us=require_integer(raw.get("valid_for_us"), "profile validity", 1),
            raw=raw,
        )
        result.validate()
        return result

    def validate(self) -> None:
        require(type(self.bubble_runtime_verified) is bool, "invalid bubble verification")
        require(
            self.layer_start < self.layer_end <= self.n_layer,
            "invalid resident layer range",
        )
        require(
            self.gpu_free_bytes <= self.gpu_total_bytes
            and self.gpu_free_bytes >= self.gpu_reserve_bytes,
            "GPU memory reserve is unavailable",
        )
        require(
            self.resident_bytes <= self.gpu_total_bytes - self.gpu_free_bytes,
            "stage placement exceeds occupied VRAM",
        )
        require(
            self.workspace_bytes <= self.gpu_free_bytes - self.gpu_reserve_bytes,
            "stage workspace exceeds free VRAM",
        )
        require(self.prefill_chunk_rows <= 512, "prefill chunk is too large")
        service_upper = max(
            self.prefill_service_latency_us.upper,
            self.decode_service_latency_us.upper,
        )
        require(
            self.protected_ready_lower_us
            > self.guard_us + service_upper + self.restore_latency_us.upper,
            "calibrated protected window cannot fit the stage",
        )
        require(self.minimum_energy_saving_ppm < 1_000_000, "invalid saving gate")
        runtime_resources = {"cpu", "gpu-compute", "op15-htp"}
        require(
            set(self.producer_resource_ids) <= runtime_resources
            and set(self.protected_resource_ids) <= runtime_resources
            and set(self.fenced_resource_ids) <= runtime_resources
            and "gpu-compute" not in self.producer_resource_ids
            and not set(self.producer_resource_ids) & set(self.fenced_resource_ids)
            and set(self.fenced_resource_ids) <= set(self.protected_resource_ids)
            and set(self.protected_resource_ids)
            <= set(self.producer_resource_ids) | set(self.fenced_resource_ids),
            "invalid causal resource topology",
        )
        if self.admission == "qualified":
            metrics = (
                self.prefill_service_latency_us,
                self.decode_service_latency_us,
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


@dataclass(frozen=True)
class StageIdentity:
    layer_start: int
    layer_end: int
    n_layer: int
    n_embd: int
    max_streams: int
    n_ctx_seq: int
    n_batch: int
    n_ubatch: int
    capabilities: int
    file_type: int
    model_sha256: str


@dataclass
class StageChunk:
    packet: bytes
    rows: int
    row_start: int
    shape: str
    input_sha256: str
    request_ids: tuple[int, ...]
    route_epochs: tuple[int, ...]
    seq_ids: tuple[int, ...]
    positions: tuple[int, ...]


@dataclass
class PendingStageCall:
    packet: bytes
    call_id: int
    command: int
    rows: int
    ready_at_us: int
    input_sha256: str
    measured: bool
    position_base: int
    chunks: tuple[StageChunk, ...]
    output: bytearray
    next_chunk: int = 0
    response: bytes | None = None
    error: BaseException | None = None
    completed: threading.Event = field(default_factory=threading.Event)


class StageWavefrontGate:
    def __init__(self, args: argparse.Namespace, profile: StageWavefrontProfile):
        self.args = args
        self.profile = profile
        self.worker: socket.socket | None = None
        self.worker_identity: StageIdentity | None = None
        self.client_listener: socket.socket | None = None
        self.client: socket.socket | None = None
        self.fence_listener: socket.socket | None = None
        self.fence: socket.socket | None = None
        self.client_thread: threading.Thread | None = None
        self.condition = threading.Condition()
        self.pending: PendingStageCall | None = None
        self.client_error: BaseException | None = None
        self.client_done = False
        self.final_stop = False
        self.scheduler: UnifiedScheduler | None = None
        self.session_index = 0
        self.session_completed_receipts: list[str] = []
        self.completed_receipts: list[str] = []
        self.events: list[dict[str, object]] = []
        self.call_count = 0
        self.chunk_calls = 0
        self.fence_calls = 0
        self.last_fence_sequence = 0
        self.backfills = 0
        self.warmup_calls = 0
        self.tail_calls = 0
        self.rejections = 0
        self.admission_disabled_reason: str | None = None
        self.circuit_breaker_reason: str | None = None
        self.circuit_breaker_at_backfill: int | None = None
        self.fallback_executions = 0
        self.protected_resource_leases: list[dict[str, object]] = []
        self.qwen_complete_ns: int | None = None
        self.paid_tail_ns: int | None = None
        self.candidate_ready_ns: int | None = None
        self.candidate_prepare_arm_ns: int | None = None
        self.candidate_prepared_ns: int | None = None
        self.prepare_calls = 0
        self.prepare_durations_us: list[int] = []
        self.prefill_position = 0
        self.started_ns = time.monotonic_ns()
        self.last_progress_ns = self.started_ns

    def log_event(self, event: dict[str, object]) -> None:
        value = {"at_ns": time.monotonic_ns(), **event}
        self.events.append(value)
        print(json.dumps(value, ensure_ascii=True, sort_keys=True), flush=True)
        self.last_progress_ns = int(value["at_ns"])

    def connect_worker(self) -> None:
        worker = socket.create_connection(
            (self.args.worker_host, self.args.worker_port),
            timeout=self.args.connect_timeout_s,
        )
        worker.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        worker.settimeout(None)
        worker.sendall(I32.pack(STAGE_V3_HELLO))
        hello_packet = recv_exact(worker, V3_HELLO_STRUCT.size)
        require(hello_packet is not None, "stage V3 hello is absent")
        hello = V3_HELLO_STRUCT.unpack(hello_packet)
        require(
            hello[0] == STAGE_V3_MAGIC
            and hello[1] == STAGE_V3_VERSION
            and (hello[10] & STAGE_V3_CAP_BATCH)
            and not (hello[10] & STAGE_V3_CAP_TERMINAL)
            and (hello[10] & STAGE_V3_CAP_IDENTITY),
            "invalid GPU-stage hello",
        )
        worker.sendall(I32.pack(STAGE_V3_IDENTITY))
        identity_packet = recv_exact(worker, IDENTITY_STRUCT.size)
        digest = recv_exact(worker, 32)
        require(identity_packet is not None and digest is not None, "stage identity is absent")
        identity_header = IDENTITY_STRUCT.unpack(identity_packet)
        require(
            identity_header[:2] == (STAGE_IDENTITY_MAGIC, STAGE_IDENTITY_VERSION),
            "invalid stage identity",
        )
        identity = StageIdentity(
            layer_start=hello[2],
            layer_end=hello[3],
            n_layer=hello[4],
            n_embd=hello[5],
            max_streams=hello[6],
            n_ctx_seq=hello[7],
            n_batch=hello[8],
            n_ubatch=hello[9],
            capabilities=hello[10],
            file_type=identity_header[2],
            model_sha256="sha256:" + digest.hex(),
        )
        require(
            identity.layer_start == self.profile.layer_start
            and identity.layer_end == self.profile.layer_end
            and identity.n_layer == self.profile.n_layer
            and identity.n_embd == self.profile.n_embd
            and identity.file_type == self.profile.file_type
            and identity.model_sha256 == self.profile.gemma_model_sha256
            and identity.max_streams > 0
            and identity.n_ctx_seq > 0
            and identity.n_batch > 0
            and identity.n_ubatch > 0,
            "GPU-stage identity differs from the profile",
        )
        require(
            self.profile.prefill_chunk_rows <= identity.n_ubatch,
            "prefill chunk exceeds the GPU-stage ubatch",
        )
        if self.worker_identity is not None:
            require(identity == self.worker_identity, "GPU-stage identity changed across sessions")
        self.worker = worker
        self.worker_identity = identity
        if self.scheduler is None:
            self.build_scheduler()
        self.log_event({
            "event": "worker_connected",
            "file_type": identity.file_type,
            "layer_end": identity.layer_end,
            "layer_start": identity.layer_start,
            "model_sha256": identity.model_sha256,
            "n_batch": identity.n_batch,
            "n_ubatch": identity.n_ubatch,
            "session_index": self.session_index,
        })

    def setup(self) -> None:
        self.connect_worker()
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

    def build_scheduler(self) -> None:
        now_us = time.monotonic_ns() // 1000
        valid_until_us = now_us + self.profile.valid_for_us
        occupied_bytes = self.profile.gpu_total_bytes - self.profile.gpu_free_bytes
        spec = DynamicWeightPlacementSpec(
            placement_id="gemma-gpu-stage-vram",
            slice_id=(
                f"gemma-layers-{self.profile.layer_start}-{self.profile.layer_end}"
            ),
            model_id="gemma4-12b-f16",
            model_hash=self.profile.gemma_model_sha256,
            weight_hash=self.profile.stage_weight_sha256,
            resource_id="gpu-vram",
            resident_bytes=self.profile.resident_bytes,
            execution_resource_ids=("gpu-compute",),
            runtime_binding_ids=(
                self.profile.gpu_uuid,
                f"layers-{self.profile.layer_start}-{self.profile.layer_end}",
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
            "schema": "s42-gpu-stage-residency-epoch-v1",
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
        if self.args.mode in {"adaptive", "qualified", "staged"}:
            for resource_id in self.profile.protected_resource_ids:
                leases = self.scheduler.reserve_external_resource(
                    resource_id,
                    f"protected-qwen-{resource_id}",
                    now_us,
                    valid_until_us,
                )
                self.protected_resource_leases.extend({
                    "lease_id": lease.lease_id,
                    "owner_id": lease.owner_id,
                    "predicted_end_us": lease.predicted_end_us,
                    "reserved_until_us": lease.reserved_until_us,
                    "resource_id": lease.resource_id,
                    "start_us": lease.start_us,
                } for lease in leases)

    def start_client(self) -> None:
        if self.client_thread is not None:
            if self.client_thread.is_alive():
                return
            self.client_thread.join()
            self.client_thread = None
            if self.client is not None:
                self.client.close()
                self.client = None
            if self.final_stop:
                return
            self.client_done = False
            self.session_index += 1
            self.session_completed_receipts.clear()
        try:
            client, _ = self.client_listener.accept()
        except TimeoutError:
            return
        if self.worker is None:
            self.connect_worker()
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client.settimeout(None)
        self.client = client
        self.client_thread = threading.Thread(
            target=self.client_loop,
            name="gemma-gpu-stage-client",
            daemon=True,
        )
        self.client_thread.start()

    @staticmethod
    def pack_values(code: str, values: tuple[int, ...]) -> bytes:
        return struct.pack(f"<{len(values)}{code}", *values)

    def stage_chunk(
        self,
        *,
        row_start: int,
        shape: str,
        request_ids: tuple[int, ...],
        route_epochs: tuple[int, ...],
        seq_ids: tuple[int, ...],
        positions: tuple[int, ...],
        tokens: tuple[int, ...],
    ) -> StageChunk:
        rows = len(tokens)
        require(
            rows > 0
            and all(
                len(values) == rows
                for values in (request_ids, route_epochs, seq_ids, positions)
            ),
            "invalid GPU-stage chunk vectors",
        )
        packet = V3_BATCH_HEADER_STRUCT.pack(
            STAGE_V3_BATCH, STAGE_V3_VERSION, rows, 0
        )
        packet += self.pack_values("q", request_ids)
        packet += self.pack_values("q", route_epochs)
        packet += self.pack_values("i", seq_ids)
        packet += self.pack_values("i", positions)
        packet += self.pack_values("i", tokens)
        return StageChunk(
            packet=packet,
            rows=rows,
            row_start=row_start,
            shape=shape,
            input_sha256="sha256:" + hashlib.sha256(packet).hexdigest(),
            request_ids=request_ids,
            route_epochs=route_epochs,
            seq_ids=seq_ids,
            positions=positions,
        )

    def read_stage_call(self, command_packet: bytes, command: int) -> PendingStageCall:
        assert self.client is not None
        identity = self.worker_identity
        assert identity is not None
        expected_width = 0 if identity.layer_start == 0 else identity.n_embd
        require(expected_width == 0, "GPU-prefix gate requires a head stage")
        if command == STAGE_BATCH_PREFILL:
            rest = recv_exact(self.client, PREFILL_HEADER_STRUCT.size - I32.size)
            require(rest is not None, "short stage prefill header")
            packet = command_packet + rest
            _, streams, tokens, width = PREFILL_HEADER_STRUCT.unpack(packet)
            require(streams > 0 and tokens > 0, "invalid stage prefill shape")
            rows = streams * tokens
            require(
                streams <= identity.max_streams
                and tokens <= 512
                and rows <= identity.n_batch
                and width == expected_width,
                "stage prefill exceeds worker geometry",
            )
            payload_size = rows * I32.size + rows * width * 4
        else:
            rest = recv_exact(self.client, DECODE_HEADER_STRUCT.size - I32.size)
            require(rest is not None, "short stage decode header")
            packet = command_packet + rest
            _, rows, width = DECODE_HEADER_STRUCT.unpack(packet)
            require(
                0 < rows <= identity.max_streams
                and rows <= identity.n_batch
                and width == expected_width,
                "stage decode exceeds worker geometry",
            )
            payload_size = rows * 3 * I32.size + rows * width * 4
        payload = recv_exact(self.client, payload_size)
        require(payload is not None, "short stage call payload")
        packet += payload
        session_identity = self.session_index + 1
        if command == STAGE_BATCH_PREFILL:
            require(streams == 1, "GPU-prefix prefill requires one stream")
            tokens_value = struct.unpack(f"<{rows}i", payload[: rows * I32.size])
            position_base = self.prefill_position
            chunks = []
            for row_start in range(0, rows, self.profile.prefill_chunk_rows):
                row_end = min(rows, row_start + self.profile.prefill_chunk_rows)
                chunk_rows = row_end - row_start
                chunks.append(self.stage_chunk(
                    row_start=row_start,
                    shape="prefill",
                    request_ids=(session_identity,) * chunk_rows,
                    route_epochs=(session_identity,) * chunk_rows,
                    seq_ids=(0,) * chunk_rows,
                    positions=tuple(range(
                        position_base + row_start,
                        position_base + row_end,
                    )),
                    tokens=tuple(tokens_value[row_start:row_end]),
                ))
        else:
            position_base = -1
            require(rows <= identity.n_ubatch, "GPU-prefix decode exceeds ubatch")
            vector_bytes = rows * I32.size
            seq_ids = struct.unpack(f"<{rows}i", payload[:vector_bytes])
            positions = struct.unpack(
                f"<{rows}i", payload[vector_bytes : 2 * vector_bytes]
            )
            tokens_value = struct.unpack(
                f"<{rows}i", payload[2 * vector_bytes : 3 * vector_bytes]
            )
            chunks = [self.stage_chunk(
                row_start=0,
                shape="decode",
                request_ids=(session_identity,) * rows,
                route_epochs=(session_identity,) * rows,
                seq_ids=tuple(seq_ids),
                positions=tuple(positions),
                tokens=tuple(tokens_value),
            )]
        self.call_count += 1
        return PendingStageCall(
            packet=packet,
            call_id=self.call_count,
            command=command,
            rows=rows,
            ready_at_us=time.monotonic_ns() // 1000,
            input_sha256="sha256:" + hashlib.sha256(packet).hexdigest(),
            measured=self.args.arm_file.exists(),
            position_base=position_base,
            chunks=tuple(chunks),
            output=bytearray(rows * identity.n_embd * 4),
        )

    def proxy_fixed(self, packet: bytes, response_size: int) -> bytes:
        assert self.worker is not None
        self.worker.sendall(packet)
        response = recv_exact(self.worker, response_size)
        require(response is not None, "short GPU-stage control response")
        return response

    def client_loop(self) -> None:
        try:
            assert self.client is not None and self.worker is not None
            while True:
                command_packet = recv_exact(self.client, I32.size, eof_ok=True)
                if command_packet is None:
                    break
                command = I32.unpack(command_packet)[0]
                if command == STAGE_HELLO:
                    self.client.sendall(self.proxy_fixed(command_packet, LEGACY_HELLO_STRUCT.size))
                    continue
                if command == STAGE_RESET:
                    response = self.proxy_fixed(command_packet, I32.size)
                    require(I32.unpack(response)[0] == 0, "GPU-stage reset failed")
                    self.prefill_position = 0
                    self.client.sendall(response)
                    continue
                if command in {STAGE_BATCH_PREFILL, STAGE_BATCH_DECODE}:
                    pending = self.read_stage_call(command_packet, command)
                    with self.condition:
                        require(self.pending is None, "more than one GPU-stage call is pending")
                        self.pending = pending
                        self.condition.notify_all()
                    if pending.measured and self.candidate_ready_ns is None:
                        self.candidate_ready_ns = time.monotonic_ns()
                        write_monotonic_receipt(
                            self.args.candidate_ready_file,
                            self.candidate_ready_ns,
                        )
                        self.log_event({
                            "call_id": pending.call_id,
                            "candidate_ready_ns": self.candidate_ready_ns,
                            "event": "candidate_ready",
                        })
                    pending.completed.wait()
                    if pending.error is not None:
                        raise pending.error
                    require(pending.response is not None, "GPU-stage response is absent")
                    self.client.sendall(pending.response)
                    continue
                if command == STAGE_DETACH:
                    response = self.proxy_fixed(command_packet, I32.size)
                    require(I32.unpack(response)[0] == 0, "GPU-stage detach failed")
                    self.client.sendall(response)
                    self.worker.close()
                    self.worker = None
                    break
                if command == STAGE_STOP:
                    self.worker.sendall(command_packet)
                    self.worker.close()
                    self.worker = None
                    self.final_stop = True
                    break
                raise GateError(f"unsupported GPU-stage command {command}")
        except BaseException as exc:  # pylint: disable=broad-except
            self.client_error = exc
        finally:
            with self.condition:
                self.client_done = True
                self.condition.notify_all()

    def accept_fence(self) -> None:
        if self.fence is not None:
            return
        try:
            self.fence, _ = self.fence_listener.accept()
        except TimeoutError:
            return
        self.fence.settimeout(0.05)
        self.log_event({"event": "fence_connected"})

    def peek_pending(self) -> PendingStageCall | None:
        with self.condition:
            return self.pending

    def take_pending(self) -> PendingStageCall | None:
        with self.condition:
            value = self.pending
            self.pending = None
            return value

    @staticmethod
    def finish_pending(
        pending: PendingStageCall,
        response: bytes | None,
        error: BaseException | None,
    ) -> None:
        pending.response = response
        pending.error = error
        pending.completed.set()

    def execute_worker(self, pending: PendingStageCall) -> tuple[bytes, dict[str, object]]:
        assert self.worker is not None
        identity = self.worker_identity
        assert identity is not None
        chunk = pending.chunks[pending.next_chunk]
        self.worker.sendall(chunk.packet)
        identity_bytes = chunk.rows * (8 + 8 + 4 + 4)
        response = recv_exact(
            self.worker,
            V3_RESPONSE_HEADER_STRUCT.size
            + identity_bytes
            + chunk.rows * identity.n_embd * 4,
        )
        require(response is not None, "short GPU-stage execution response")
        header = V3_RESPONSE_HEADER_STRUCT.unpack(
            response[:V3_RESPONSE_HEADER_STRUCT.size]
        )
        require(
            header == (0, chunk.rows, identity.n_embd),
            "invalid GPU-stage execution response",
        )
        offset = V3_RESPONSE_HEADER_STRUCT.size
        vectors: list[tuple[int, ...]] = []
        for code, width in (("q", 8), ("q", 8), ("i", 4), ("i", 4)):
            size = chunk.rows * width
            vectors.append(struct.unpack(
                f"<{chunk.rows}{code}", response[offset : offset + size]
            ))
            offset += size
        require(
            tuple(vectors)
            == (
                chunk.request_ids,
                chunk.route_epochs,
                chunk.seq_ids,
                chunk.positions,
            ),
            "GPU-stage response identity changed",
        )
        output = response[offset:]
        require(
            len(output) == chunk.rows * identity.n_embd * 4,
            "GPU-stage output size changed",
        )
        self.chunk_calls += 1
        return output, {
            "output_sha256": "sha256:" + hashlib.sha256(response).hexdigest(),
            "chunk_count": len(pending.chunks),
            "chunk_index": pending.next_chunk,
            "row_start": chunk.row_start,
            "position_start": chunk.positions[0],
            "position_end": chunk.positions[-1] + 1,
            "rows": chunk.rows,
            "shape": chunk.shape,
        }

    def advance_pending(self, pending: PendingStageCall, output: bytes) -> bool:
        identity = self.worker_identity
        assert identity is not None
        chunk = pending.chunks[pending.next_chunk]
        start = chunk.row_start * identity.n_embd * 4
        pending.output[start : start + len(output)] = output
        pending.next_chunk += 1
        pending.ready_at_us = time.monotonic_ns() // 1000
        if pending.next_chunk < len(pending.chunks):
            return False
        if pending.command == STAGE_BATCH_PREFILL:
            require(
                self.prefill_position == pending.position_base,
                "GPU-stage prefill position changed",
            )
            self.prefill_position += pending.rows
        require(self.take_pending() is pending, "pending stage call changed")
        response = RESPONSE_HEADER_STRUCT.pack(pending.rows, identity.n_embd)
        self.finish_pending(pending, response + bytes(pending.output), None)
        return True

    def execute_direct(self, pending: PendingStageCall, route: str) -> None:
        start_ns = time.monotonic_ns()
        try:
            output, metrics = self.execute_worker(pending)
        except BaseException as exc:  # pylint: disable=broad-except
            require(self.take_pending() is pending, "pending stage call changed")
            self.finish_pending(pending, None, exc)
            raise
        end_ns = time.monotonic_ns()
        self.advance_pending(pending, output)
        if route == "warmup":
            self.warmup_calls += 1
        else:
            self.tail_calls += 1
        self.log_event({
            "call_id": pending.call_id,
            "duration_us": (end_ns - start_ns) // 1000,
            "event": "worker_execute",
            "route": route,
            **metrics,
        })

    def schedule_pending(
        self, pending: PendingStageCall, fence: tuple[int, ...]
    ) -> tuple[object, dict[str, object]]:
        assert self.scheduler is not None
        chunk = pending.chunks[pending.next_chunk]
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
        sequence_index = len(self.session_completed_receipts)
        chunk_id = (
            f"gemma-stage-s{self.session_index}-c{sequence_index}-"
            f"{chunk.input_sha256[-12:]}"
        )
        service = (
            self.profile.prefill_service_latency_us
            if chunk.shape == "prefill"
            else self.profile.decode_service_latency_us
        )
        candidate = GpuBackfillCandidate(
            candidate_id=chunk_id,
            work_id=chunk_id + "-work",
            model_id="gemma4-12b-f16",
            gpu_resource_id="gpu-compute",
            workspace_resource_id="gpu-vram",
            workspace_bytes=self.profile.workspace_bytes,
            required_placement_ids=("gemma-gpu-stage-vram",),
            additional_resource_ids=self.profile.producer_resource_ids,
            deadline_us=protected_ready_us - self.profile.guard_us,
            energy_boundary_id=self.profile.energy_boundary_id,
            accounting_scope=self.profile.accounting_scope,
            service_latency_us=service,
            restore_latency_us=self.profile.restore_latency_us,
            avoided_energy_uj=self.profile.avoided_energy_uj,
            backfill_energy_uj=self.profile.backfill_energy_uj,
            evidence_ids=self.profile.evidence_ids,
        )
        predecessor = (
            self.session_completed_receipts[-1]
            if self.session_completed_receipts
            else None
        )
        chunk = GpuReadyChunk(
            chunk_id=chunk_id,
            pipeline_id=f"{self.args.pipeline_id}-session-{self.session_index}",
            model_id="gemma4-12b-f16",
            sequence_index=sequence_index,
            layer_start=self.profile.layer_start,
            layer_end=self.profile.layer_end,
            token_count=chunk.rows,
            ready_receipt_id=chunk_id + "-ready",
            input_buffer_id=chunk_id + "-input",
            input_buffer_sha256=chunk.input_sha256,
            predecessor_output_receipt_id=predecessor,
            ready_at_us=pending.ready_at_us,
            valid_until_us=protected_ready_us,
            producer_resource_ids=self.profile.producer_resource_ids,
            runtime_verified=pending.measured,
            candidate=candidate,
            evidence_ids=self.profile.evidence_ids,
        )
        pipeline_id = chunk.pipeline_id
        wavefront = GpuWavefrontSnapshot(
            wavefront_id=f"stage-wavefront-fence-{fence[3]}",
            source_snapshot_id=snapshot.snapshot_id,
            source_snapshot_sha256=canonical_sha256(snapshot.to_json()),
            source_generation=snapshot.generation,
            source_epoch_key=snapshot.epoch_key,
            captured_at_us=now_us,
            valid_until_us=protected_ready_us,
            next_sequence_by_pipeline={pipeline_id: sequence_index},
            completed_output_receipt_ids=tuple(self.session_completed_receipts),
            ready_chunks=(chunk,),
        )
        schedule = self.scheduler.schedule_gpu_wavefront_backfill(
            bubble,
            wavefront,
            now_us=now_us,
            minimum_energy_saving_ppm=self.profile.minimum_energy_saving_ppm,
            require_measured=self.args.mode in {"adaptive", "qualified"},
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
        end_ns = start_ns
        selected = False
        schedule_receipt: dict[str, object] | None = None
        pending = self.peek_pending()
        eligible = (
            self.args.mode in {"adaptive", "mechanics", "qualified", "staged"}
            and self.qwen_complete_ns is None
            and self.admission_disabled_reason is None
            and self.circuit_breaker_reason is None
            and pending is not None
            and pending.measured
            and (
                self.args.prepare_file is None
                or self.candidate_prepared_ns is not None
            )
            and (self.args.max_backfills == 0 or self.backfills < self.args.max_backfills)
        )
        if eligible:
            assert pending is not None
            try:
                schedule, schedule_receipt = self.schedule_pending(pending, fence)
            except BaseException as exc:  # pylint: disable=broad-except
                self.admission_disabled_reason = str(exc)
                self.log_event({
                    "call_id": pending.call_id,
                    "error": str(exc),
                    "event": "wavefront_scheduler_failure",
                })
                end_ns = time.monotonic_ns()
            else:
                if schedule.decision.chunk_id is None:
                    self.rejections += 1
                    self.log_event({
                        "call_id": pending.call_id,
                        "decision": schedule.decision.to_json(),
                        "event": "wavefront_rejected",
                    })
                    end_ns = time.monotonic_ns()
                else:
                    selected = True
                    output = None
                    sequence_index = len(self.session_completed_receipts)
                    try:
                        output, metrics = self.execute_worker(pending)
                        end_ns = time.monotonic_ns()
                    except BaseException as exc:  # pylint: disable=broad-except
                        self.admission_disabled_reason = str(exc)
                        require(
                            self.take_pending() is pending,
                            "pending stage call changed",
                        )
                        self.finish_pending(pending, None, exc)
                        self.log_event({
                            "call_id": pending.call_id,
                            "error": str(exc),
                            "event": "wavefront_contract_violation",
                        })
                        end_ns = time.monotonic_ns()
                    else:
                        output_receipt = (
                            f"gemma-stage-output-{self.session_index}-"
                            f"{len(self.session_completed_receipts)}-"
                            f"{metrics['output_sha256'][-12:]}"
                        )
                        try:
                            self.scheduler.release_gpu_wavefront_backfill(
                                schedule, end_ns // 1000, output_receipt
                            )
                        except BaseException as exc:  # pylint: disable=broad-except
                            try:
                                self.scheduler.abort_gpu_wavefront_backfill(
                                    schedule, end_ns // 1000
                                )
                            except BaseException as abort_exc:  # pylint: disable=broad-except
                                self.admission_disabled_reason = (
                                    f"{exc}; abort failed: {abort_exc}"
                                )
                            else:
                                self.circuit_breaker_reason = str(exc)
                                self.circuit_breaker_at_backfill = self.backfills
                                self.fallback_executions += 1
                            self.advance_pending(pending, output)
                            self.log_event({
                                "call_id": pending.call_id,
                                "completed_backfills": self.backfills,
                                "error": str(exc),
                                "event": "wavefront_circuit_breaker",
                                "fallback_executions": self.fallback_executions,
                            })
                        else:
                            self.session_completed_receipts.append(output_receipt)
                            self.completed_receipts.append(output_receipt)
                            self.backfills += 1
                            self.advance_pending(pending, output)
                            self.log_event({
                                "call_id": pending.call_id,
                                "decision_sha256": schedule.decision.decision_sha256,
                                "duration_us": (end_ns - start_ns) // 1000,
                                "event": "wavefront_execute",
                                "output_receipt_id": output_receipt,
                                "sequence_index": sequence_index,
                                **metrics,
                            })
        assert self.fence is not None
        self.fence.sendall(FENCE_RESPONSE_STRUCT.pack(
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
        ))
        response_sent_ns = time.monotonic_ns()
        self.log_event({
            "event": "fence_done",
            "fence_sequence": fence[3],
            "filler_selected": selected,
            "filler_us": (end_ns - start_ns) // 1000,
            "gate_handler_us": (response_sent_ns - start_ns) // 1000,
            "gate_queue_us": (start_ns - fence[8]) // 1000,
            "gate_start_ns": start_ns,
            "host_ready_ns": fence[8],
            "request_id": fence[4],
            "response_sent_ns": response_sent_ns,
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

    def poll_candidate_prepare(self) -> None:
        if (
            self.args.prepare_file is None
            or self.candidate_prepared_ns is not None
            or not self.args.prepare_file.exists()
            or self.qwen_complete_ns is not None
        ):
            return
        pending = self.peek_pending()
        if pending is None or not pending.measured:
            return
        prepare_arm_ns = read_receipt(
            self.args.prepare_file, "candidate prepare"
        )
        require(
            self.candidate_ready_ns is not None
            and self.candidate_ready_ns <= prepare_arm_ns,
            "candidate prepare preceded readiness",
        )
        start_ns = time.monotonic_ns()
        output_hashes: list[str] = []
        metrics: dict[str, object] | None = None
        stable = 0
        for _ in range(self.args.prepare_max_replays):
            replay_start_ns = time.monotonic_ns()
            _, metrics = self.execute_worker(pending)
            assert self.worker is not None
            self.worker.sendall(I32.pack(STAGE_RESET))
            response = recv_exact(self.worker, I32.size)
            require(
                response is not None and I32.unpack(response)[0] == 0,
                "GPU-stage prepare reset failed",
            )
            replay_duration_us = (time.monotonic_ns() - replay_start_ns) // 1000
            self.prepare_calls += 1
            self.prepare_durations_us.append(replay_duration_us)
            output_hashes.append(str(metrics["output_sha256"]))
            if replay_duration_us <= self.profile.prefill_service_latency_us.upper:
                stable += 1
            else:
                stable = 0
            if stable >= self.args.prepare_required_consecutive:
                break
        require(
            stable >= self.args.prepare_required_consecutive,
            "GPU-stage preparation did not reach the service bound",
        )
        require(
            len(set(output_hashes)) == 1,
            "GPU-stage preparation output changed across replays",
        )
        assert metrics is not None
        self.candidate_prepare_arm_ns = prepare_arm_ns
        self.candidate_prepared_ns = time.monotonic_ns()
        write_monotonic_receipt(
            self.args.prepared_file, self.candidate_prepared_ns
        )
        self.log_event({
            "call_id": pending.call_id,
            "candidate_prepare_arm_ns": prepare_arm_ns,
            "candidate_prepared_ns": self.candidate_prepared_ns,
            "duration_us": (self.candidate_prepared_ns - start_ns) // 1000,
            "event": "candidate_prepare",
            "prepare_output_sha256": output_hashes,
            "prepare_replay_durations_us": self.prepare_durations_us,
            "prepare_required_consecutive": self.args.prepare_required_consecutive,
            "prepare_service_upper_us": self.profile.prefill_service_latency_us.upper,
            "sequence_index": len(self.session_completed_receipts),
            **metrics,
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
        self.execute_direct(pending, route)

    def complete_tail(self) -> bool:
        with self.condition:
            done = self.final_stop and self.client_done and self.pending is None
        if self.qwen_complete_ns is None or not done:
            return False
        if self.client_error is not None:
            raise self.client_error
        self.paid_tail_ns = time.monotonic_ns()
        write_monotonic_receipt(self.args.paid_tail_file, self.paid_tail_ns)
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
            self.poll_candidate_prepare()
            self.drain_ready()
            self.poll_fence()
            if self.complete_tail():
                return
            if self.client_error is not None:
                raise self.client_error
            if time.monotonic_ns() - self.last_progress_ns > timeout_ns:
                raise GateError("GPU-stage wavefront gate timed out")
            time.sleep(0.001)

    def result(self, status: str, error: str | None) -> dict[str, object]:
        return {
            "admission_disabled_reason": self.admission_disabled_reason,
            "backfills": self.backfills,
            "calls": self.call_count,
            "candidate_ready_ns": self.candidate_ready_ns,
            "candidate_prepare_arm_ns": self.candidate_prepare_arm_ns,
            "candidate_prepared_ns": self.candidate_prepared_ns,
            "chunk_calls": self.chunk_calls,
            "circuit_breaker_at_backfill": self.circuit_breaker_at_backfill,
            "circuit_breaker_reason": self.circuit_breaker_reason,
            "completed_receipts": self.completed_receipts,
            "energy_claim_eligible": False,
            "error": error,
            "events": self.events,
            "fence_calls": self.fence_calls,
            "fallback_executions": self.fallback_executions,
            "finished_ns": time.monotonic_ns(),
            "mode": self.args.mode,
            "maximum_backfills": self.args.max_backfills,
            "paid_tail_ns": self.paid_tail_ns,
            "pid": os.getpid(),
            "profile_admission": self.profile.admission,
            "resource_snapshot": {
                "producer_resource_ids": list(self.profile.producer_resource_ids),
                "protected_resource_ids": list(self.profile.protected_resource_ids),
                "fenced_resource_ids": list(self.profile.fenced_resource_ids),
                "protected_resource_leases": self.protected_resource_leases,
                "protected_resources_reserved": self.args.mode
                in {"adaptive", "qualified", "staged"},
                "schema": "s42-gpu-stage-resource-snapshot-v1",
            },
            "prefill_chunk_rows": self.profile.prefill_chunk_rows,
            "profile_sha256": self.profile.profile_sha256,
            "prepare_calls": self.prepare_calls,
            "prepare_durations_us": self.prepare_durations_us,
            "prepare_max_replays": self.args.prepare_max_replays,
            "prepare_required_consecutive": self.args.prepare_required_consecutive,
            "qwen_complete_ns": self.qwen_complete_ns,
            "rejections": self.rejections,
            "schema": RESULT_SCHEMA,
            "started_ns": self.started_ns,
            "status": status,
            "tail_calls": self.tail_calls,
            "warmup_calls": self.warmup_calls,
            "worker_identity": (
                None if self.worker_identity is None else self.worker_identity.__dict__
            ),
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
    parser.add_argument(
        "--mode",
        choices=("adaptive", "control", "mechanics", "qualified", "staged"),
        required=True,
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--worker-host", default="127.0.0.1")
    parser.add_argument("--worker-port", type=int, required=True)
    parser.add_argument("--fence-socket", type=Path, required=True)
    parser.add_argument("--arm-file", type=Path, required=True)
    parser.add_argument("--qwen-complete-file", type=Path, required=True)
    parser.add_argument("--paid-tail-file", type=Path, required=True)
    parser.add_argument("--candidate-ready-file", type=Path, required=True)
    parser.add_argument("--prepare-file", type=Path)
    parser.add_argument("--prepared-file", type=Path)
    parser.add_argument("--prepare-max-replays", type=int, default=1)
    parser.add_argument("--prepare-required-consecutive", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline-id", default="gemma-gpu-prefix")
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
        "candidate_ready_file",
        "output",
    ):
        path = getattr(args, name)
        if not path.is_absolute():
            parser.error(f"{name.replace('_', ' ')} must be absolute")
    if not args.profile.is_file():
        parser.error("profile must be an existing file")
    if (args.prepare_file is None) != (args.prepared_file is None):
        parser.error("candidate prepare paths must be provided together")
    if args.prepare_file is not None and (
        not args.prepare_file.is_absolute()
        or not args.prepared_file.is_absolute()
    ):
        parser.error("candidate prepare paths must be absolute")
    for path in (
        args.fence_socket,
        args.arm_file,
        args.qwen_complete_file,
        args.paid_tail_file,
        args.candidate_ready_file,
        args.prepare_file,
        args.prepared_file,
        args.output,
    ):
        if path is None:
            continue
        if path.exists() or not path.parent.is_dir():
            parser.error("runtime paths must be unused and have existing parents")
    if not (0 < args.listen_port <= 65535 and 0 < args.worker_port <= 65535):
        parser.error("ports must be in range")
    if (
        args.max_backfills < 0
        or args.prepare_max_replays <= 0
        or args.prepare_required_consecutive <= 0
        or args.prepare_required_consecutive > args.prepare_max_replays
        or args.connect_timeout_s <= 0
        or args.timeout_s <= 0
    ):
        parser.error("runtime limits are invalid")
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
        profile = StageWavefrontProfile.load(args.profile)
    except GateError as exc:
        print(f"GPU-stage profile error: {exc}", file=sys.stderr)
        return 2
    if args.mode == "qualified" and profile.admission != "qualified":
        print("qualified mode requires a qualified profile", file=sys.stderr)
        return 2
    gate = StageWavefrontGate(args, profile)
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
        print(f"GPU-stage wavefront gate failed: {exc}", file=sys.stderr)
    finally:
        write_result(args.output, gate.result(status, error))
        gate.close()
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
