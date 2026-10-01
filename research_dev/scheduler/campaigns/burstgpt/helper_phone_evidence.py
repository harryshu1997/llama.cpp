"""Pinned evidence and lifecycle construction for static campaign co-helpers."""

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

from research_dev.scheduler import PlacementHardwareProfile, RuntimePhonePowerProfile
from research_dev.scheduler._internal.plan_contracts.co_helpers import co_helper_declaration
from research_dev.scheduler._internal.types import canonical_sha256
from research_dev.scheduler._internal.capability_contracts.common import _placement_profile_json
from research_dev.scheduler.adapters import apply_assumed_phone_power_profile
from research_dev.scheduler.adapters.co_helper_lifecycle import CoHelperLifecycle, IdleCoHelperStopPolicy
from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler.adapters.phone_helpers import PhoneHelperTransportIdentity, observe_usb_port
from research_dev.scheduler.adapters.phone_tcp_session import AdbTcpPhoneWorkerSession, AdbTcpWorkerConfiguration
from research_dev.scheduler.adapters.phone_aoa_session import (
    AOA_LINK_TRANSPORT, AoaBridgeConfiguration, AoaBridgePhoneWorkerSession,
)


def require(condition, message):
    if not condition:
        raise PhysicalAdapterError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


@dataclass(frozen=True)
class HelperPhoneEvidence:
    path: Path
    identity: PhoneHelperTransportIdentity
    worker: AdbTcpWorkerConfiguration
    profile_fragment: dict
    power: RuntimePhonePowerProfile
    # opt-in WS10: the helper's worker is reached through the host AOA bridge (transport identity aoa-bridge)
    aoa_bridge: AoaBridgeConfiguration | None = None

    def session(self):
        """A fresh worker session of the qualified transport."""
        if self.aoa_bridge is not None:
            return AoaBridgePhoneWorkerSession(self.worker, self.aoa_bridge)
        return AdbTcpPhoneWorkerSession(self.worker)

    def verify_identity(self, session=None):
        """The phone on the qualified USB port is the qualified phone (serial, speed, controller,
        ADB identity -- or, for aoa-bridge, its accessory+adb identity -- kernel release); returns the
        qualified transport identity."""
        observed = observe_usb_port(self.identity.hardware_identity["phone_usb_sysfs_device"])
        usb_identities = {self.identity.hardware_identity["adb_usb_identity"]}
        if self.aoa_bridge is not None:
            usb_identities.add(self.identity.hardware_identity["aoa_usb_identity"])
        require(observed.serial == self.worker.serial
                and observed.negotiated_speed_mbps >= self.identity.minimum_usb_speed_mbps
                and observed.host_controller == self.identity.hardware_identity["host_usb_controller"]
                and observed.vendor_product in usb_identities,
                "helper USB identity differs from qualification")
        session = self.session() if session is None else session
        require(session._shell("uname -r").strip() == self.identity.hardware_identity["phone_kernel_release"],
                "helper kernel differs from qualification")
        return self.identity.identity_sha256

    def live_preflight(self):
        session = self.session()
        self.verify_identity(session)
        return session.preflight().to_json()


def load_helper_evidence(path, *, server_path=None):
    path = Path(path)
    raw = json.loads(path.read_text())
    require(raw.get("schema") == "s42-static-helper-evidence-v1" and raw.get("status") == "PASS",
            "helper campaign evidence is not qualified")
    identity = PhoneHelperTransportIdentity.from_json(raw["transport_identity"])
    require(identity.qualified, "helper transport receipts are incomplete")
    for kind, receipt_hash in identity.receipts.items():
        receipt = Path(raw["receipt_paths"][kind])
        require(digest(receipt) == receipt_hash and json.loads(receipt.read_text()).get("status") == "PASS",
                "helper receipt differs or failed: " + kind)
    worker_raw = dict(raw["worker"])
    worker_raw["adb_path"] = Path(worker_raw["adb_path"])
    worker = AdbTcpWorkerConfiguration(**worker_raw)
    software = identity.software_identity
    require(worker.device_id == identity.device_id
            and worker.serial == identity.hardware_identity["phone_usb_serial"]
            and worker.adb_port == 5037 and worker.forward_port > 0 and worker.max_requests == 0,
            "helper campaign worker identity or lifecycle differs")
    require(software["worker_environment_sha256"] == canonical_sha256(dict(worker.worker_environment))
            and software["phone_worker_sha256"] == worker.expected_sha256_by_path[worker.worker_path]
            and software["phone_shard_sha256"] == worker.expected_sha256_by_path[worker.shard_path],
            "helper worker or shard differs from qualification")
    for name, expected in software.items():
        if name.startswith("phone_library_sha256:"):
            library = name.split(":", 1)[1]
            require(worker.expected_sha256_by_path.get(library) == expected,
                    "helper library differs from qualification: " + library)
    for name, host_path in raw["host_software_paths"].items():
        require(digest(host_path) == software[name], "helper host software differs: " + name)
    require({"host_binary_sha256", "host_impl_sha256", "transport_client_source_sha256"}
            <= set(raw["host_software_paths"]), "helper host software pins are incomplete")
    if server_path is not None:
        require(Path(raw["host_software_paths"]["host_binary_sha256"]).resolve() == Path(server_path).resolve(),
                "helper qualified a different server")
    power = RuntimePhonePowerProfile.from_json(raw["power"])
    require(power.device_id == worker.device_id and power.allow_assumed_for_scheduling,
            "helper scheduling power is absent")
    aoa = None
    if identity.transport == AOA_LINK_TRANSPORT:
        require("aoa_bridge" in raw, "aoa-bridge helper evidence lacks its bridge configuration")
        aoa = AoaBridgeConfiguration.from_json(raw["aoa_bridge"])
        require(software["phone_relay_sha256"] == aoa.relay_sha256
                and software["host_bridge_sha256"] == aoa.bridge_script_sha256
                and software["aoa_bridge_options_sha256"] == aoa.options_sha256
                and identity.hardware_identity["phone_usb_sysfs_device"] == aoa.usb_sysfs_device
                and worker.serial not in aoa.forbidden_serials,
                "helper AOA bridge differs from qualification")
    else:
        require("aoa_bridge" not in raw, "only aoa-bridge helper evidence carries a bridge configuration")
    fragment = raw["profile_fragment"]
    require([row["device_id"] for row in fragment["devices"]] == [worker.device_id]
            and all(row["device_id"] == worker.device_id for row in fragment["kernels"]),
            "helper cost profile modifies another device")
    return HelperPhoneEvidence(path, identity, worker, fragment, power, aoa)


def extend_helper_profile(profile, evidences):
    raw = _placement_profile_json(profile)
    for evidence in evidences.values():
        for key in ("devices", "memory_pools", "domains", "kernels", "links", "idle_charge_domains"):
            raw[key].extend(evidence.profile_fragment[key])
    profile = PlacementHardwareProfile.from_json(raw)
    for evidence in evidences.values():
        profile = apply_assumed_phone_power_profile(profile, evidence.power)
    return profile


def extend_helper_overlay(base, overlay, evidences):
    if not evidences:
        return overlay
    helpers = tuple(row for row in base.executors if row.device_id in evidences)
    resource_ids = {resource for row in helpers for resource in row.execution_resource_ids}
    resource_ids.update("link:" + row.link_id for row in base.placement_profile.links
                        if row.source_device in evidences or row.target_device in evidences)
    resources = dict(overlay.resources)
    resources.update({key: base.resources[key] for key in resource_ids if key not in resources})
    return replace(overlay, placement_profile=extend_helper_profile(overlay.placement_profile, evidences),
                   resources=resources, executors=overlay.executors + helpers,
                   phone_power_profiles=overlay.phone_power_profiles + tuple(row.power for row in evidences.values()))


def validate_helper_declaration(declaration, manifest, evidence, rig_row):
    helper = next(row for row in declaration.helpers if row.device_id == evidence.worker.device_id)
    worker = evidence.worker
    for name in ("device_id", "serial", "adb_port", "backend", "worker_path", "library_directories",
                 "column_quantum", "max_tokens", "forward_port", "max_requests", "worker_environment",
                 "as_root", "phone_lock_path"):
        require(getattr(worker, name) == getattr(rig_row, name), "helper rig differs: " + name)
    link = getattr(rig_row, "transport", "adb-tcp")
    require((link == AOA_LINK_TRANSPORT) == (evidence.aoa_bridge is not None)
            and helper.transport_parameters.get("ffn_link_transport") == (
                AOA_LINK_TRANSPORT if evidence.aoa_bridge is not None else None),
            "helper rig transport differs from its evidence")
    if evidence.aoa_bridge is not None:
        rig_bridge = AoaBridgeConfiguration.from_json(dict(getattr(rig_row, "aoa_bridge", None) or {}))
        require(rig_bridge.to_json() == evidence.aoa_bridge.to_json(), "helper rig AOA bridge differs from its evidence")
    require(worker.phone_port == rig_row.worker_port and worker.artifact_sha256 == manifest.artifact_sha256
            and worker.n_embd == manifest.embedding_length and worker.columns == manifest.feed_forward_length
            and worker.layer_mask == helper.layer_mask
            and worker.expected_sha256_by_path[worker.shard_path] == helper.shard_sha256,
            "helper shard or geometry differs from the model declaration")


def campaign_co_helper_lifecycles(paths, catalog, manifests, server_path):
    evidences = [load_helper_evidence(path, server_path=server_path) for path in paths]
    by_device = {row.worker.device_id: row for row in evidences}
    require(len(by_device) == len(evidences), "duplicate helper campaign evidence")
    declarations = {}
    for row in catalog.composite_executors:
        declaration = co_helper_declaration(row.adapter_parameters)
        if declaration is None:
            continue
        previous = declarations.setdefault(row.artifact_sha256, declaration)
        require(previous == declaration, "model has conflicting helper declarations")
    require({device for declaration in declarations.values() for device in declaration.device_ids} == set(by_device),
            "catalog and helper campaign evidence differ")
    by_artifact = {manifest.artifact_sha256: manifest for manifest in manifests.values()}
    result = {}
    used = set()
    for artifact, declaration in declarations.items():
        require(not used.intersection(declaration.device_ids), "static helper serves more than one artifact")
        used.update(declaration.device_ids)
        sessions = {}
        for device in declaration.device_ids:
            evidence = by_device[device]
            require(evidence.worker.artifact_sha256 == artifact and artifact in by_artifact,
                    "helper worker artifact differs from catalog")
            sessions[device] = evidence.session()
        # an elastic join re-verifies the pinned identity before the pinned-hash preflight
        result[artifact] = CoHelperLifecycle(
            declaration, sessions, IdleCoHelperStopPolicy(),
            identity_checks={device: by_device[device].verify_identity for device in declaration.device_ids})
    return result
