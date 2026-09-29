"""DirectPhoneFfnSession replacement operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
import socket

from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_plan import (
    PhoneSessionReplacementAuthorization,
    RuntimePhoneShard,
    phone_session_map_sha256,
)
from ..contracts import PhysicalAdapterError
from ..llama_server import phone_ffn_resident_contract, primary_phone_ffn_contract
from ..phone_transport import PhoneTransportContract
from ..ticket import PhysicalTransitionCommand
from ..phone_session_contracts.events import parse_phone_residency_phase_events
from ..phone_session_contracts.receipts import DirectPhoneFfnReconfigurationReceipt
from .common import _ShardResidencyWindow


def _replace_resident_shard(
    controller,
    previous_shard: RuntimePhoneShard | None,
    target_shard: RuntimePhoneShard,
) -> None:
    """Advance physical proof windows for one added or replaced shard."""

    retiring = getattr(controller, "_residency_generation", 1)
    if previous_shard is not None:
        controller._shard_residency_windows = [
            replace(row, last_generation=retiring)
            if row.last_generation is None and row.shard == previous_shard
            else row
            for row in getattr(controller, "_shard_residency_windows", [])
        ]
    controller._residency_generation = retiring + 1
    controller._shard_residency_windows.append(_ShardResidencyWindow(
        shard=target_shard,
        first_generation=controller._residency_generation,
    ))
    if (
        previous_shard is not None
        and previous_shard in getattr(
            controller, "_executed_proof_shards", []
        )
    ):
        controller._executed_proof_shards.remove(previous_shard)
        controller._historical_execution_proofs = [
            *getattr(controller, "_historical_execution_proofs", []),
        ]
        controller._historical_execution_proofs.append({
            "artifact_sha256": previous_shard.artifact_sha256,
            "kind": "retired_executed_shard",
            "last_residency_generation": retiring,
            "operator_plan_sha256": previous_shard.operator_plan_sha256,
            "resident_geometry_sha256": (
                previous_shard.resident_geometry_sha256
            ),
            "session_id": previous_shard.session_id,
        })


def _changed_phone_sessions(
    current: tuple[RuntimePhoneShard, ...],
    target: tuple[RuntimePhoneShard, ...],
) -> tuple[str, ...]:
    current_by_session = {row.session_id: row for row in current}
    target_by_session = {row.session_id: row for row in target}
    return tuple(sorted(
        session_id
        for session_id in set(current_by_session) | set(target_by_session)
        if current_by_session.get(session_id)
            != target_by_session.get(session_id)
    ))


def _validate_partial_reconfiguration_authority(
    controller,
    command: PhysicalTransitionCommand,
) -> None:
    launch = controller._launch
    authorization = getattr(command, "replacement_authorization", None)
    changed = tuple(command.transition.changed_phone_session_ids)
    target = tuple(command.transition.phone_shards)
    if (
        launch is None
        or not isinstance(
            authorization, PhoneSessionReplacementAuthorization
        )
        or changed != (authorization.selected_session_id,)
    ):
        raise PhysicalAdapterError(
            "partial phone replacement authorization is absent"
        )
    selected = authorization.selected_session_id
    current_by_session = {
        row.session_id: row for row in launch.phone_shards
    }
    target_by_session = {row.session_id: row for row in target}
    current = current_by_session.get(selected)
    replacement = target_by_session.get(selected)
    if (
        phone_session_map_sha256(launch.phone_shards)
            != authorization.source_layout_hash
        or phone_session_map_sha256(target)
            != authorization.target_layout_hash
        or controller._changed_phone_sessions(launch.phone_shards, target)
            != (selected,)
        or replacement is None
        or (
            current is None
            and authorization.source_generation != 0
        )
        or (
            current is not None
            and current.session_generation
                != authorization.source_generation
        )
        or replacement.session_generation
            != authorization.target_generation
    ):
        raise PhysicalAdapterError(
            "stale phone session replacement authorization"
        )


def supports_partial_reconfiguration(
    controller,
    command: PhysicalTransitionCommand,
    manifest: ModelManifest,
    transport: PhoneTransportContract,
) -> bool:
    launch = controller._launch
    target = command.transition.phone_shards
    if (
        launch is not None
        and controller._remote_root is not None
        and len(command.transition.changed_phone_session_ids) == 1
    ):
        controller._validate_partial_reconfiguration_authority(command)
    if (
        launch is None
        or controller._remote_root is None
        or not launch.phone_shards
        or len(target) < 2
        or not launch.transport.shares_resident_session_with(transport)
        or controller.configuration.diagnostic_host is None
        or controller.configuration.diagnostic_port <= 0
        or controller.configuration.diagnostic_port >= 65535
        or command.artifact_sha256 != manifest.artifact_sha256
        or controller._changed_phone_sessions(launch.phone_shards, target)
            != tuple(command.transition.changed_phone_session_ids)
        or len(command.transition.changed_phone_session_ids) != 1
    ):
        return False
    model_path = controller.configuration.model_paths_by_artifact.get(
        manifest.artifact_sha256
    )
    if (
        model_path is None
        or controller._verified_remote_hash_by_path.get(model_path)
            != manifest.artifact_sha256
    ):
        return False
    execution = primary_phone_ffn_contract(
        command, phone_ffn_resident_contract(command, manifest)
    )
    model_shards = tuple(
        row for row in target
        if row.artifact_sha256 == manifest.artifact_sha256
    )
    return (
        bool(model_shards)
        and execution.layer_mask
            == sum(row.layer_mask for row in model_shards)
        and execution.columns
            == max(row.maximum_columns for row in model_shards)
    )


def reconfigure(
    controller,
    command: PhysicalTransitionCommand,
    manifest: ModelManifest,
    transport: PhoneTransportContract,
) -> DirectPhoneFfnReconfigurationReceipt:
    if not controller.supports_partial_reconfiguration(
        command, manifest, transport
    ):
        raise PhysicalAdapterError(
            "direct phone partial reconfiguration is unavailable"
        )
    launch = controller._launch
    assert launch is not None
    controller._validate_partial_reconfiguration_authority(command)
    target = command.transition.phone_shards
    authorization = command.replacement_authorization
    assert isinstance(
        authorization, PhoneSessionReplacementAuthorization
    )
    changed_session_id = authorization.selected_session_id
    previous_by_session = {
        row.session_id: row for row in launch.phone_shards
    }
    target_by_session = {row.session_id: row for row in target}
    previous_shard = previous_by_session.get(changed_session_id)
    target_shard = target_by_session.get(changed_session_id)
    if (
        target_shard is None
        or (
            previous_shard is None
            and target_shard.session_generation != 1
        )
        or (
            previous_shard is not None
            and target_shard.session_generation
                != previous_shard.session_generation + 1
        )
        or any(
            previous_by_session.get(session_id) != shard
            for session_id, shard in target_by_session.items()
            if session_id != changed_session_id
        )
    ):
        raise PhysicalAdapterError(
            "phone residency replacement session identity is invalid"
        )
    manifest_rows, target_sha256 = controller._multi_session_manifest(target)
    payload = (manifest_rows.replace(";", "\n") + "\n").encode(
        "ascii"
    )
    column_quantum = command.adapter_parameters.get(
        "ffn_column_quantum"
    )
    execution = primary_phone_ffn_contract(
        command, phone_ffn_resident_contract(command, manifest)
    )
    if (
        type(column_quantum) is not int
        or column_quantum <= 0
        or target_shard is None
        or target_shard.maximum_columns % column_quantum
    ):
        raise PhysicalAdapterError(
            "phone residency replacement quantum is invalid"
        )
    request = (
        "S42RESIDENCY_V3 "
        + str(len(payload))
        + " "
        + target_sha256
        + " "
        + str(column_quantum)
        + " "
        + str(execution.max_tokens)
        + "\n"
    ).encode("ascii") + payload
    host = controller.configuration.diagnostic_host
    assert host is not None
    response = b""
    try:
        with socket.create_connection(
            (host, controller.configuration.diagnostic_port + 1),
            timeout=controller.configuration.launch_timeout_s,
        ) as connection:
            connection.settimeout(
                controller.configuration.launch_timeout_s
            )
            connection.sendall(request)
            connection.shutdown(socket.SHUT_WR)
            while not response.endswith(b"\n"):
                block = connection.recv(4096)
                if not block or len(response) + len(block) > 4096:
                    raise PhysicalAdapterError(
                        "phone residency control response is invalid"
                    )
                response += block
    except OSError as error:
        raise PhysicalAdapterError(
            "phone residency control failed: " + str(error)
        ) from error
    fields = response.decode("ascii", "strict").strip().split()
    if (
        len(fields) != 7
        or fields[0] != "S42RESIDENCY_READY"
        or fields[1] != target_sha256
        or fields[2] != changed_session_id
        or not fields[3].isdigit()
        or not fields[4].isdigit()
        or int(fields[4]) != column_quantum
        or not fields[5].isdigit()
        or int(fields[5]) != execution.max_tokens
        or not fields[6].isdigit()
        or int(fields[6]) != target_shard.session_generation
    ):
        raise PhysicalAdapterError(
            "phone residency control differs from the ticket: "
            + response.decode("ascii", "backslashreplace").strip()
        )
    configured_ffn_shards = getattr(
        controller.configuration, "ffn_shards_by_artifact", {}
    )
    phase_events = (
        parse_phone_residency_phase_events(
            controller._read_diagnostic_file("residency.log").splitlines()
        )
        if configured_ffn_shards else ()
    )
    receipt = DirectPhoneFfnReconfigurationReceipt(
        ticket_id=command.ticket_id,
        previous_manifest_sha256=(
            launch.shard_manifest_sha256 or ""
        ),
        target_manifest_sha256=target_sha256,
        changed_session_id=changed_session_id,
        load_count=int(fields[3]),
        previous_load_count=controller._load_count_by_session.get(
            changed_session_id, 0
        ),
        column_quantum=column_quantum,
        previous_column_quantum=(
            None
            if previous_shard is None
            else controller._column_quantum_by_session.get(
                changed_session_id
            )
        ),
        max_tokens=execution.max_tokens,
        previous_max_tokens=controller._max_tokens_by_session.get(
            changed_session_id, execution.max_tokens
        ),
        previous_shard=previous_shard,
        target_shard=target_shard,
        target_shards=target,
        target_weight_source=controller._resolve_phone_weight_source(
            target_shard, phase_events
        ),
    )
    remote_hashes = dict(launch.remote_hashes)
    remote_hashes[
        "model:" + manifest.artifact_sha256
    ] = manifest.artifact_sha256
    controller._launch = replace(
        launch,
        remote_hashes=remote_hashes,
        phone_shards=target,
        shard_manifest_sha256=target_sha256,
        weight_sources=controller._phone_weight_sources(
            target, phase_events
        ),
    )
    controller._execution_by_artifact[manifest.artifact_sha256] = execution
    controller._transport_by_artifact[manifest.artifact_sha256] = transport
    controller._load_count_by_session[changed_session_id] = receipt.load_count
    controller._column_quantum_by_session[changed_session_id] = (
        receipt.column_quantum
    )
    controller._max_tokens_by_session[changed_session_id] = receipt.max_tokens
    controller._proof_shards = sorted(
        (
            target_shard
            if row.session_id == changed_session_id else row
            for row in controller._proof_shards
        ) if previous_shard is not None else (
            *controller._proof_shards, target_shard
        ),
        key=lambda row: row.session_id,
    )
    controller._replace_resident_shard(previous_shard, target_shard)
    if command.ticket_id not in controller._bound_ticket_ids:
        controller._bound_ticket_ids.append(command.ticket_id)
    bind_generations = dict(getattr(controller, "_ticket_bind_generations", {}))
    bind_generations.setdefault(
        command.ticket_id, controller._residency_generation
    )
    controller._ticket_bind_generations = bind_generations
    return receipt


def rollback_reconfiguration(
    controller,
    receipt: DirectPhoneFfnReconfigurationReceipt,
) -> None:
    launch = controller._launch
    if (
        launch is None
        or controller._remote_root is None
        or receipt.target_shard is None
    ):
        raise PhysicalAdapterError(
            "direct phone reconfiguration rollback is unavailable"
        )
    current_by_session = {
        row.session_id: row for row in launch.phone_shards
    }
    if (
        current_by_session.get(receipt.changed_session_id)
            != receipt.target_shard
    ):
        raise PhysicalAdapterError(
            "direct phone reconfiguration rollback source differs"
        )
    restored_shard = (
        None
        if receipt.previous_shard is None
        else replace(
            receipt.previous_shard,
            session_generation=(
                receipt.target_shard.session_generation + 1
            ),
        )
    )
    if restored_shard is None:
        current_by_session.pop(receipt.changed_session_id)
    else:
        current_by_session[receipt.changed_session_id] = restored_shard
    target = tuple(
        current_by_session[key] for key in sorted(current_by_session)
    )
    rollback_column_quantum = (
        receipt.column_quantum
        if restored_shard is None
        else receipt.previous_column_quantum
    )
    if (
        type(rollback_column_quantum) is not int
        or rollback_column_quantum <= 0
        or (
            restored_shard is not None
            and restored_shard.maximum_columns
                % rollback_column_quantum
        )
    ):
        raise PhysicalAdapterError(
            "phone residency rollback quantum is invalid"
        )
    manifest_rows, target_sha256 = controller._multi_session_manifest(target)
    payload = (manifest_rows.replace(";", "\n") + "\n").encode(
        "ascii"
    )
    request = (
        "S42RESIDENCY_V3 "
        + str(len(payload))
        + " "
        + target_sha256
        + " "
        + str(rollback_column_quantum)
        + " "
        + str(receipt.previous_max_tokens)
        + "\n"
    ).encode("ascii") + payload
    host = controller.configuration.diagnostic_host
    assert host is not None
    response = b""
    try:
        with socket.create_connection(
            (host, controller.configuration.diagnostic_port + 1),
            timeout=controller.configuration.launch_timeout_s,
        ) as connection:
            connection.settimeout(controller.configuration.launch_timeout_s)
            connection.sendall(request)
            connection.shutdown(socket.SHUT_WR)
            while not response.endswith(b"\n"):
                block = connection.recv(4096)
                if not block or len(response) + len(block) > 4096:
                    raise PhysicalAdapterError(
                        "phone residency rollback response is invalid"
                    )
                response += block
    except OSError as error:
        raise PhysicalAdapterError(
            "phone residency rollback failed: " + str(error)
        ) from error
    fields = response.decode("ascii", "strict").strip().split()
    if (
        len(fields) != 7
        or fields[0] != "S42RESIDENCY_READY"
        or fields[1] != target_sha256
        or fields[2] != receipt.changed_session_id
        or not fields[3].isdigit()
        or not fields[4].isdigit()
        or int(fields[4]) != rollback_column_quantum
        or not fields[5].isdigit()
        or int(fields[5]) != receipt.previous_max_tokens
        or not fields[6].isdigit()
        or int(fields[6]) != (
            0 if restored_shard is None else
            restored_shard.session_generation
        )
    ):
        raise PhysicalAdapterError(
            "phone residency rollback differs from the prior layout: "
            + response.decode("ascii", "backslashreplace").strip()
        )
    configured_ffn_shards = getattr(
        controller.configuration, "ffn_shards_by_artifact", {}
    )
    phase_events = (
        parse_phone_residency_phase_events(
            controller._read_diagnostic_file("residency.log").splitlines()
        )
        if configured_ffn_shards else ()
    )
    controller._launch = replace(
        launch,
        phone_shards=target,
        shard_manifest_sha256=target_sha256,
        weight_sources=controller._phone_weight_sources(
            target, phase_events
        ),
    )
    if restored_shard is None:
        controller._load_count_by_session.pop(receipt.changed_session_id, None)
        controller._column_quantum_by_session.pop(
            receipt.changed_session_id, None
        )
        controller._max_tokens_by_session.pop(receipt.changed_session_id, None)
        controller._proof_shards = [
            row for row in controller._proof_shards
            if row.session_id != receipt.changed_session_id
        ]
        retiring = controller._residency_generation
        controller._shard_residency_windows = [
            replace(row, last_generation=retiring)
            if row.last_generation is None
            and row.shard == receipt.target_shard else row
            for row in controller._shard_residency_windows
        ]
        controller._residency_generation = retiring + 1
    else:
        controller._load_count_by_session[receipt.changed_session_id] = int(
            fields[3]
        )
        controller._column_quantum_by_session[receipt.changed_session_id] = (
            rollback_column_quantum
        )
        controller._max_tokens_by_session[receipt.changed_session_id] = (
            receipt.previous_max_tokens
        )
        controller._proof_shards = [
            restored_shard
            if row.session_id == receipt.changed_session_id else row
            for row in controller._proof_shards
        ]
        controller._replace_resident_shard(
            receipt.target_shard, restored_shard
        )
