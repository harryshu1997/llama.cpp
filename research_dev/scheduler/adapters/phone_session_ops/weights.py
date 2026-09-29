"""DirectPhoneFfnSession weights operations on its existing owner."""

from __future__ import annotations

import hashlib
import json
import re

from ..._internal.runtime_plan import RuntimePhoneShard
from ..contracts import PhysicalAdapterError
from ..llama_server import PhoneFfnExecutionContract
from ..ffn_shards import FfnShardRecord
from ..phone_transport import PhoneTransportContract
from ..phone_session_contracts.events import DirectPhoneFfnTerminalReceipt, PhoneResidencyPhaseEvent
from ..phone_session_contracts.receipts import PhoneFfnWeightSource


def _worker_environment(
    execution: PhoneFfnExecutionContract,
    transport: PhoneTransportContract,
    column_quantum: int,
) -> tuple[str, ...]:
    return (
        "S41_FFN_F16_IO=1",
        "S41_FFN_STAGED_DMABUF=0",
        "S41_FFN_MAX_TOKENS=" + str(execution.max_tokens),
        "S41_FFN_COLUMN_QUANTUM=" + str(column_quantum),
        "S41_FFN_QUEUE_DEPTH=" + str(transport.queue_depth),
    )


def _layer_spec(mask: int) -> str:
    indices = tuple(
        index for index in range(64) if mask & (1 << index)
    )
    if not indices:
        raise PhysicalAdapterError("phone shard layer mask is empty")
    spans = []
    first = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        spans.append(
            str(first)
            if first == previous
            else str(first) + "-" + str(previous)
        )
        first = previous = index
    spans.append(
        str(first)
        if first == previous
        else str(first) + "-" + str(previous)
    )
    return ",".join(spans)


def _resolve_phone_weight_source(
    controller,
    shard: RuntimePhoneShard,
    phase_events: tuple[PhoneResidencyPhaseEvent, ...] = (),
) -> PhoneFfnWeightSource:
    artifact = shard.artifact_sha256
    if artifact is None:
        raise PhysicalAdapterError(
            "multi-session phone shard artifact is absent"
        )
    try:
        model_path = controller.configuration.model_paths_by_artifact[artifact]
    except KeyError as error:
        raise PhysicalAdapterError(
            "multi-session phone shard artifact is not deployed"
        ) from error
    index = getattr(
        controller.configuration, "ffn_shards_by_artifact", {}
    ).get(artifact)
    record: FfnShardRecord | None = None
    if index is not None:
        record = index.resolve(
            artifact,
            shard.layer_mask,
            shard.maximum_columns,
            session_id=shard.session_id,
        )
        if record is None:
            raise PhysicalAdapterError(
                "configured phone FFN shard index does not cover "
                "the scheduled assignment"
            )

    def phase_time(
        phase: str, component: str | None = None
    ) -> int | None:
        values = [
            row.epoch_us
            for row in phase_events
            if row.session_id == shard.session_id
            and row.session_generation == shard.session_generation
            and row.phase == phase
            and (component is None or row.component == component)
        ]
        return None if not values else min(values)

    return PhoneFfnWeightSource(
        session_id=shard.session_id,
        session_generation=shard.session_generation,
        weight_source=(
            "full_gguf" if record is None else "ffn_shard"
        ),
        source_path=model_path if record is None else record.remote_path,
        source_sha256=(
            artifact if record is None else record.shard_sha256
        ),
        index_sha256=(
            None if index is None else index.index_sha256
        ),
        parent_artifact_sha256=artifact,
        stored_layer_mask=(
            shard.layer_mask if record is None else record.layer_mask
        ),
        executed_layer_mask=shard.layer_mask,
        stored_columns=(
            shard.maximum_columns if record is None else record.columns
        ),
        active_columns=shard.maximum_columns,
        bytes_loaded=(
            shard.resident_bytes if record is None else record.shard_bytes
        ),
        load_started_epoch_us=(
            phase_time("WEIGHT_READ_BEGIN", "ffn-worker")
            or phase_time("LOAD_AUTHORIZED", "resident-manager")
        ),
        load_finished_epoch_us=phase_time(
            "WEIGHT_UPLOAD_READY", "ffn-worker"
        ),
        verified_epoch_us=phase_time(
            "VERIFIED", "resident-manager"
        ),
        ready_epoch_us=phase_time("READY", "resident-manager"),
    )


def _phone_weight_sources(
    controller,
    shards: tuple[RuntimePhoneShard, ...],
    phase_events: tuple[PhoneResidencyPhaseEvent, ...] = (),
) -> tuple[PhoneFfnWeightSource, ...]:
    return tuple(
        controller._resolve_phone_weight_source(shard, phase_events)
        for shard in sorted(shards, key=lambda row: row.session_id)
    )


def _multi_session_manifest(
    controller,
    shards: tuple[RuntimePhoneShard, ...],
) -> tuple[str, str]:
    base = controller.configuration.multi_session_port_base
    capacity = controller.configuration.multi_session_device_count
    if (
        not shards
        or base is None
        or capacity is None
        or len(shards) > capacity
        or base + capacity - 1 > 65535
    ):
        raise PhysicalAdapterError(
            "multi-session phone binding is unavailable"
        )
    port_by_id = dict(getattr(controller, "_multi_session_port_by_id", {}))
    used_ports = set(port_by_id.values())
    available_ports = iter(
        port for port in range(base, base + capacity)
        if port not in used_ports
    )
    for shard in sorted(shards, key=lambda row: row.session_id):
        if shard.session_id not in port_by_id:
            try:
                port_by_id[shard.session_id] = next(available_ports)
            except StopIteration as error:
                raise PhysicalAdapterError(
                    "multi-session phone binding is unavailable"
                ) from error
    rows = []
    for shard in shards:
        if (
            shard.artifact_sha256 is None
            or shard.session_generation < 1
            or
            not re.fullmatch(r"[A-Za-z0-9._-]+", shard.session_id)
            or not re.fullmatch(r"[A-Za-z0-9._:/-]+", shard.endpoint)
        ):
            raise PhysicalAdapterError(
                "multi-session phone shard identity is not executable"
            )
        endpoint_sha256 = "sha256:" + hashlib.sha256(
            shard.endpoint.encode("ascii")
        ).hexdigest()
        weight_source = controller._resolve_phone_weight_source(shard)
        model_path = weight_source.source_path
        if not re.fullmatch(r"/[A-Za-z0-9._:/-]+", model_path):
            raise PhysicalAdapterError(
                "multi-session phone model path is not executable"
            )
        rows.append(",".join((
            shard.session_id,
            shard.session_id,
            shard.artifact_sha256,
            model_path,
            controller._layer_spec(shard.layer_mask),
            str(shard.maximum_columns),
            str(port_by_id[shard.session_id]),
            endpoint_sha256,
            str(shard.resident_bytes),
            shard.resident_geometry_sha256,
            shard.operator_plan_sha256,
            str(shard.session_generation),
        )))
    manifest = ";".join(rows)
    encoded = ("\n".join(rows) + "\n").encode("ascii")
    controller._multi_session_port_by_id = port_by_id
    return manifest, "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_shard_loads(
    lines: tuple[str, ...] | list[str],
    shards: tuple[RuntimePhoneShard, ...],
) -> None:
    observed: dict[tuple[str, str | None, str | None], dict] = {}
    for raw in lines:
        stripped = raw.strip()
        if not stripped.startswith("RESIDENTSHARD "):
            continue
        try:
            row = json.loads(stripped.removeprefix("RESIDENTSHARD "))
        except json.JSONDecodeError as error:
            raise PhysicalAdapterError(
                "phone shard load receipt is invalid"
            ) from error
        if type(row) is not dict or type(row.get("session_id")) is not str:
            raise PhysicalAdapterError(
                "phone shard load receipt is invalid"
            )
        key = (
            row["session_id"],
            row.get("artifact_sha256"),
            row.get("resident_geometry_sha256"),
        )
        previous = observed.get(key)
        if (
            previous is None
            or int(row.get("load_count", 0))
                > int(previous.get("load_count", 0))
        ):
            observed[key] = row
    expected = {
        (
            row.session_id,
            row.artifact_sha256,
            row.resident_geometry_sha256,
        ): row
        for row in shards
    }
    if set(expected) - set(observed):
        raise PhysicalAdapterError(
            "phone shard load receipts are incomplete"
        )
    for key, shard in expected.items():
        row = observed[key]
        endpoint_sha256 = "sha256:" + hashlib.sha256(
            shard.endpoint.encode("ascii")
        ).hexdigest()
        if (
            row.get("status") != "WARM"
            or type(row.get("load_count")) is not int
            or row["load_count"] < 1
            or row.get("artifact_sha256")
                != shard.artifact_sha256
            or row.get("layer_mask")
                != format(shard.layer_mask, "016x")
            or row.get("columns") != shard.maximum_columns
            or row.get("resident_bytes") != shard.resident_bytes
            or row.get("endpoint_sha256") != endpoint_sha256
            or row.get("resident_geometry_sha256")
                != shard.resident_geometry_sha256
            or row.get("operator_plan_sha256")
                != shard.operator_plan_sha256
            or row.get("session_generation")
                != shard.session_generation
        ):
            raise PhysicalAdapterError(
                "phone shard load differs from the ticket"
            )


def _validate_shard_terminal(
    terminal: DirectPhoneFfnTerminalReceipt,
    shards: tuple[RuntimePhoneShard, ...],
    *,
    require_execution: bool = True,
    executed_shards: tuple[RuntimePhoneShard, ...] | None = None,
    historical_shards: tuple[RuntimePhoneShard, ...] = (),
) -> None:
    known = tuple(dict.fromkeys((*historical_shards, *shards)))
    known_by_base: dict[
        tuple[str, str | None, str], list[RuntimePhoneShard]
    ] = {}
    for shard in known:
        known_by_base.setdefault((
            shard.session_id,
            shard.artifact_sha256,
            shard.resident_geometry_sha256,
        ), []).append(shard)
    expected = {
        (
            row.session_id,
            row.artifact_sha256,
            row.resident_geometry_sha256,
            row.session_generation,
        ): row
        for row in known
    }
    current_keys = {
        (
            row.session_id,
            row.artifact_sha256,
            row.resident_geometry_sha256,
            row.session_generation,
        )
        for row in shards
    }
    observed = {}
    for proof in terminal.session_proofs:
        matches = known_by_base.get((
            proof.session_id,
            proof.artifact_sha256,
            proof.resident_geometry_sha256,
        ), [])
        if proof.session_generation is not None:
            matches = [
                row for row in matches
                if row.session_generation == proof.session_generation
            ]
        if len(matches) != 1:
            raise PhysicalAdapterError(
                "phone shard execution proof generation is ambiguous"
            )
        shard = matches[0]
        key = (
            shard.session_id,
            shard.artifact_sha256,
            shard.resident_geometry_sha256,
            shard.session_generation,
        )
        if key in observed:
            raise PhysicalAdapterError(
                "phone shard execution proof is duplicated"
            )
        observed[key] = proof
    if set(observed) - set(expected):
        raise PhysicalAdapterError(
            "phone shard execution proofs are incomplete"
        )
    required = (
        current_keys
        if executed_shards is None
        else {
            (
                row.session_id,
                row.artifact_sha256,
                row.resident_geometry_sha256,
                row.session_generation,
            )
            for row in executed_shards
        }
    )
    if not required.issubset(expected):
        raise PhysicalAdapterError(
            "phone executed shard proof is not resident"
        )
    if not required.issubset(observed):
        raise PhysicalAdapterError(
            "phone shard execution proofs are incomplete"
        )
    for key, proof in observed.items():
        shard = expected[key]
        endpoint_sha256 = "sha256:" + hashlib.sha256(
            shard.endpoint.encode("ascii")
        ).hexdigest()
        if (
            proof.endpoint_sha256 != endpoint_sha256
            or proof.artifact_sha256 != shard.artifact_sha256
            or proof.layer_mask != shard.layer_mask
            or proof.resident_geometry_sha256
                != shard.resident_geometry_sha256
            or proof.operator_plan_sha256
                != shard.operator_plan_sha256
            or (
                proof.session_generation is not None
                and proof.session_generation
                    != shard.session_generation
            )
            or (
                require_execution
                and key in required
                and (proof.calls <= 0 or proof.rows <= 0)
            )
        ):
            raise PhysicalAdapterError(
                "phone shard execution differs from the ticket"
            )
