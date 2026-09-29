"""DirectPhoneFfnSession identity operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
import shlex
from types import MappingProxyType
from typing import Mapping

from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_plan import RuntimePhoneShard
from ..contracts import PhysicalAdapterError
from ..llama_server import (
    LlamaServerPhoneSessionProof,
    PhoneFfnExecutionContract,
    phone_ffn_resident_contract,
    primary_phone_ffn_contract,
)
from ..phone_transport import PhoneTransportContract
from ..remote_hash_cache import RemoteFileIdentity
from ..ticket import PhysicalTransitionCommand


def _remote_file_identities(
    controller,
    paths: tuple[str, ...],
    *,
    root: bool,
    timeout_s: int,
) -> Mapping[str, RemoteFileIdentity]:
    commands = [
        "printf 'BOOT '; cat /proc/sys/kernel/random/boot_id",
    ]
    for index, path in enumerate(paths):
        commands.append(
            "printf 'FILE"
            + str(index)
            + " '; stat -c '%d:%i:%s:%Y:%Z' "
            + shlex.quote(path)
        )
    command = "; ".join(commands)
    output = (
        controller._adb_root(command, timeout_s=timeout_s)
        if root else controller._adb(command, timeout_s=timeout_s)
    )
    values = {}
    boot_id = None
    for line in output.splitlines():
        if line.startswith("BOOT "):
            boot_id = line.removeprefix("BOOT ").strip()
            continue
        if not line.startswith("FILE"):
            continue
        label, separator, stat_identity = line.partition(" ")
        index_text = label.removeprefix("FILE")
        if (
            not separator
            or not index_text.isdigit()
            or int(index_text) >= len(paths)
        ):
            raise PhysicalAdapterError(
                "phone session file identities are invalid"
            )
        values[int(index_text)] = stat_identity.strip()
    if (
        type(boot_id) is not str
        or not boot_id
        or not boot_id.isascii()
        or set(values) != set(range(len(paths)))
    ):
        raise PhysicalAdapterError(
            "phone session file identities are incomplete"
        )
    try:
        return MappingProxyType({
            path: RemoteFileIdentity(
                boot_id=boot_id,
                stat_identity=values[index],
            )
            for index, path in enumerate(paths)
        })
    except ValueError as error:
        raise PhysicalAdapterError(str(error)) from error


def _current_close_identities(
    controller,
) -> tuple[
    tuple[
        PhoneFfnExecutionContract,
        PhoneTransportContract,
        str,
    ],
    ...,
]:
    launch = controller._launch
    if launch is None:
        raise PhysicalAdapterError("direct phone session is not active")
    phone_shards = tuple(getattr(launch, "phone_shards", ()))
    resident_artifacts = {
        shard.artifact_sha256 for shard in phone_shards
        if shard.artifact_sha256 is not None
    }
    execution_by_artifact = getattr(
        controller, "_execution_by_artifact", {}
    )
    transport_by_artifact = getattr(
        controller, "_transport_by_artifact", {}
    )
    max_tokens_by_session = getattr(
        controller, "_max_tokens_by_session", {}
    )
    identities = []
    for artifact_sha256 in reversed(
        tuple(execution_by_artifact)
    ):
        if artifact_sha256 not in resident_artifacts:
            continue
        shards = tuple(
            row for row in phone_shards
            if row.artifact_sha256 == artifact_sha256
        )
        if not shards:
            continue
        execution = execution_by_artifact[artifact_sha256]
        layer_mask = 0
        for shard in shards:
            layer_mask |= shard.layer_mask
        layer_indices = tuple(
            index for index in range(64)
            if layer_mask & (1 << index)
        )
        identities.append((
            replace(
                execution,
                layer_indices=layer_indices,
                layer_mask=layer_mask,
                columns=max(row.maximum_columns for row in shards),
                max_tokens=max(
                    max_tokens_by_session[row.session_id]
                    for row in shards
                ),
            ),
            transport_by_artifact[artifact_sha256],
            artifact_sha256,
        ))
    if identities:
        return tuple(identities)
    return ((
        launch.execution,
        launch.transport,
        getattr(
            launch,
            "artifact_sha256",
            "sha256:" + "0" * 64,
        ),
    ),)


def _current_close_identity(
    controller,
) -> tuple[
    PhoneFfnExecutionContract,
    PhoneTransportContract,
    str,
]:
    return controller._current_close_identities()[0]


def _close_current_usb(controller) -> None:
    errors = []
    for execution, transport, artifact_sha256 in (
        controller._current_close_identities()
    ):
        try:
            controller._close_direct_usb(
                execution, transport, artifact_sha256
            )
            return
        except PhysicalAdapterError as error:
            errors.append(artifact_sha256 + ": " + str(error))
    raise PhysicalAdapterError(
        "direct phone USB close rejected every current resident "
        "identity: " + "; ".join(errors)
    )


def supports(
    controller,
    command: PhysicalTransitionCommand,
    manifest: ModelManifest,
    transport: PhoneTransportContract,
) -> bool:
    launch = controller._launch
    if launch is None or controller._remote_root is None:
        return False
    if command.artifact_sha256 != manifest.artifact_sha256:
        raise PhysicalAdapterError(
            "direct phone reuse artifact differs from the ticket"
        )
    execution = primary_phone_ffn_contract(
        command, phone_ffn_resident_contract(command, manifest)
    )
    transition = getattr(command, "transition", None)
    residency_shards = (
        getattr(transition, "phone_shards", ())
        or command.execution_contract.phone_shards
    )
    remote = getattr(command.execution_contract, "remote_resident_ffn", None)
    if remote is not None:
        residency_shards = command.execution_contract.resident_phone_shards(
            manifest.feed_forward_length
        )
        sources = {row.session_id: row for row in launch.weight_sources}
        for owner in remote.sessions:
            source = sources.get(owner.session_id)
            if (
                source is None or source.weight_source != "ffn_shard"
                or source.source_sha256 != owner.shard_sha256
                or source.source_path != owner.remote_path
                or source.index_sha256 != remote.shard_index_sha256
                or source.parent_artifact_sha256 != remote.parent_artifact_sha256
                or source.session_generation != owner.session_generation
            ):
                return False
        if (
            not launch.transport.shares_resident_session_with(transport)
            or launch.remote_hashes.get("model:" + manifest.artifact_sha256)
                != manifest.artifact_sha256
            or any(shard not in launch.phone_shards for shard in residency_shards)
        ):
            return False
        return all(
            controller._max_tokens_by_session.get(shard.session_id, 0)
                >= execution.max_tokens
            for shard in residency_shards
        )
    if launch.phone_shards:
        if (
            not launch.transport.shares_resident_session_with(transport)
            or launch.phone_shards
                != residency_shards
            or launch.remote_hashes.get(
                "model:" + manifest.artifact_sha256
            ) != manifest.artifact_sha256
        ):
            return False
        model_shards = tuple(
            row for row in launch.phone_shards
            if row.artifact_sha256 == manifest.artifact_sha256
        )
        return (
            bool(model_shards)
            and execution.layer_mask
                == sum(row.layer_mask for row in model_shards)
            and execution.columns
                == max(row.maximum_columns for row in model_shards)
        )
    return (
        launch.remote_hashes.get("model") == manifest.artifact_sha256
        and launch.execution == execution
        and launch.transport == transport
        and launch.phone_shards
            == command.execution_contract.phone_shards
    )


def bind(
    controller,
    command: PhysicalTransitionCommand,
    manifest: ModelManifest,
    transport: PhoneTransportContract,
) -> None:
    if not controller.supports(command, manifest, transport):
        raise PhysicalAdapterError(
            "direct phone residency differs from the ticket"
        )
    if command.ticket_id not in controller._bound_ticket_ids:
        controller._bound_ticket_ids.append(command.ticket_id)
    bind_generations = dict(getattr(controller, "_ticket_bind_generations", {}))
    bind_generations.setdefault(
        command.ticket_id, getattr(controller, "_residency_generation", 1)
    )
    controller._ticket_bind_generations = bind_generations


def bind_ticket_generation(controller, ticket_id: str) -> int:
    """Pin a ticket's proofs to the residency generation it started in."""

    if type(ticket_id) is not str or not ticket_id:
        raise PhysicalAdapterError("phone ticket generation is invalid")
    generation = getattr(controller, "_residency_generation", 1)
    bind_generations = dict(getattr(controller, "_ticket_bind_generations", {}))
    bound = bind_generations.setdefault(ticket_id, generation)
    controller._ticket_bind_generations = bind_generations
    return bound


def _proof_key(
    row: RuntimePhoneShard | LlamaServerPhoneSessionProof,
) -> tuple[str, str | None, str, str, int]:
    return (
        row.session_id,
        row.artifact_sha256,
        row.resident_geometry_sha256,
        row.operator_plan_sha256,
        row.session_generation,
    )


def record_execution_proof(
    controller,
    ticket_id: str,
    artifact_sha256: str,
    proofs: tuple[LlamaServerPhoneSessionProof, ...],
) -> None:
    if (
        not controller.active
        or type(ticket_id) is not str
        or not ticket_id
        or type(artifact_sha256) is not str
        or not artifact_sha256.startswith("sha256:")
        or not proofs
        or any(
            not isinstance(row, LlamaServerPhoneSessionProof)
            or row.artifact_sha256 != artifact_sha256
            or row.calls <= 0
            or row.rows <= 0
            for row in proofs
        )
    ):
        raise PhysicalAdapterError(
            "phone execution proof binding is invalid"
        )
    loaded = {
        controller._proof_key(row): row for row in controller._proof_shards
    }
    if len(loaded) != len(controller._proof_shards):
        raise PhysicalAdapterError(
            "phone shard proof identity is duplicated"
        )
    current_generation = getattr(controller, "_residency_generation", 1)
    windows_history = getattr(controller, "_shard_residency_windows", [])
    bind_generations = getattr(controller, "_ticket_bind_generations", {})
    # A ticket bound before a replacement may legitimately report calls
    # on the shard that was resident when those calls happened.
    bound_generation = bind_generations.get(ticket_id, 1)
    selected = []
    historical = []
    for proof in proofs:
        key = controller._proof_key(proof)
        shard = loaded.get(key)
        if shard is not None and (
            shard.endpoint == proof.endpoint
            and proof.layer_mask > 0
            and not proof.layer_mask & ~shard.layer_mask
        ):
            selected.append(shard)
            continue
        windows = [
            row for row in windows_history
            if controller._proof_key(row.shard) == key
            and row.shard.endpoint == proof.endpoint
            and proof.layer_mask > 0
            and not proof.layer_mask & ~row.shard.layer_mask
            and row.last_generation is not None
            and row.first_generation <= current_generation
            and row.last_generation >= bound_generation
        ]
        if not windows:
            raise PhysicalAdapterError(
                "phone execution proof differs from loaded residency"
            )
        window = windows[-1]
        historical.append({
            "artifact_sha256": proof.artifact_sha256,
            "calls": proof.calls,
            "current_residency_generation": current_generation,
            "first_residency_generation": window.first_generation,
            "kind": "historical_session_execution",
            "last_residency_generation": window.last_generation,
            "layer_mask": proof.layer_mask,
            "operator_plan_sha256": proof.operator_plan_sha256,
            "resident_geometry_sha256": (
                proof.resident_geometry_sha256
            ),
            "rows": proof.rows,
            "session_id": proof.session_id,
            "ticket_bound_generation": bound_generation,
            "ticket_id": ticket_id,
        })
    for shard in selected:
        if shard not in controller._executed_proof_shards:
            controller._executed_proof_shards.append(shard)
    if historical:
        controller._historical_execution_proofs = [
            *getattr(controller, "_historical_execution_proofs", []),
            *historical,
        ]
    if ticket_id not in controller._bound_ticket_ids:
        controller._bound_ticket_ids.append(ticket_id)
    bind_generations.setdefault(ticket_id, 1)
    controller._ticket_bind_generations = bind_generations
