#!/usr/bin/env python3
"""WS10 Phase B: turn the AOA bridge qualification into a helper evidence bundle and a campaign arm.

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.prepare_aoa_evidence bundle \\
        --adb-evidence <qualified adb-tcp PIXEL_EVIDENCE.json> --ab-dir <pixel_transport_ab OUT (analysed)> \\
        --server-identity <qualify_pixel_server_transport OUT> --aoa-bridge AOA_BRIDGE.json \\
        --bridge-script <deployed adapters/aoa_bridge.py> --output-dir DIR
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.prepare_aoa_evidence derive-arm \\
        --inputs <inputs-two-phone-ATT> --evidence DIR/PIXEL_EVIDENCE_AOA.json --output <inputs-two-phone-ATT-aoa>

``bundle`` keeps the worker, shard, libraries, environment, power and kernel profile of the qualified adb-tcp
bundle (the worker binary is unchanged: its numerical receipt carries over) and replaces the transport: identity
``aoa-bridge`` (accessory+adb USB identity, relay / host bridge / keep-awake option pins), fresh receipts
(usb-link-speed in accessory mode, server-token-identity over the bridge, aoa-bridge-round-trip,
aoa-bridge-byte-identity, scheduler-launched-session) and the link rows of the cost profile from the measured
round trip (``--link-calibration adb`` keeps the adb-forward link rows instead: a transport-only full-system A/B
in which the scheduler's decisions see the same link costs). Every input must be a PASS and must match the AOA configuration being qualified; the result is
re-loaded with ``load_helper_evidence`` before it is written.

``derive-arm`` copies a materialized two-phone arm and changes only the Pixel's transport: rig
``helper_phones[].transport = "aoa-bridge"`` + ``aoa_bridge``, the helper evidence path and the campaign id.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

from research_dev.scheduler.adapters.contracts import PhysicalAdapterError
from research_dev.scheduler.adapters.phone_aoa_session import AoaBridgeConfiguration, file_sha256
from research_dev.scheduler.adapters.phone_helpers import IDENTITY_REQUIREMENTS
from research_dev.scheduler.adapters.phone_transport import AOA_BRIDGE_TRANSPORT_GENERATION
from research_dev.scheduler.campaigns.burstgpt.helper_phone_evidence import load_helper_evidence
from research_dev.scheduler.configuration.rig import RigManifest

AOA_USB_IDENTITY = "18d1:2d01"
LINK_PREFIX = "pixel-aoa-"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalAdapterError(message)


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write(path: Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def calibration_runs(ab_dir: Path, trip: dict) -> list[Path]:
    runs = [ab_dir / run for run in trip["runs"]]
    need(runs and all((run / "RESULT.json").exists() for run in runs), "round-trip runs are missing")
    return runs


def check_measured_configuration(ab_dir: Path, trip: dict, aoa: AoaBridgeConfiguration) -> dict:
    """The configuration qualified here is the one the calibration arm ran: relay and bridge pins by the
    session preflight, keep-awake options by the arm's merged options."""
    from research_dev.scheduler.campaigns.burstgpt.tools.pixel_transport_ab import bridge_configuration
    config = read(ab_dir / "CONFIG.json")
    for run in calibration_runs(ab_dir, trip):
        arm = read(run / "ARM.json")
        measured = bridge_configuration(config, arm)
        need(measured.options_sha256 == aoa.options_sha256, "the calibration arm ran other keep-awake options")
        preflight = read(run / "PREFLIGHT.json")["aoa_bridge"]
        need(preflight["relay_sha256"] == aoa.relay_sha256 and preflight["bridge_script_sha256"]
             == aoa.bridge_script_sha256, "the calibration arm ran another relay or host bridge")
    return config


def usb_receipt(ab_dir: Path, trip: dict, serial: str, sysfs_device: str, minimum_mbps: int) -> dict:
    run = calibration_runs(ab_dir, trip)[0]
    state = read(run / "HOST_USB_BEFORE.json")
    device = state["device"]
    need(device.get("serial") == serial and device.get("vendor_product") == AOA_USB_IDENTITY
         and device.get("sysfs_device") == sysfs_device and device.get("speed_mbps", 0) >= minimum_mbps,
         "the Pixel's accessory-mode USB link does not match the qualification")
    return {"status": "PASS", "source": str(run / "HOST_USB_BEFORE.json"), "device": device, "lpm": state.get("lpm"),
            "usbfs_memory_mb": state.get("usbfs_memory_mb")}


def server_receipt(run: Path, worker: dict, host_binary_sha256: str, aoa: AoaBridgeConfiguration) -> dict:
    need(not (run / "FAILURE.json").exists(), "server identity run failed")
    result = read(run / "RESULT.json")
    need(result["status"] == "PASS" and result["token_identity"] == [True, True, True, True]
         and result.get("transport") == "aoa-bridge" and result["server_exit"] == 0 and result["worker_exit"] == 0
         and result["phone_calls"] > 0 and result.get("boot_unchanged") is True, "server identity over the bridge failed")
    need(set(result["call_columns"]) == {"8704", "17408"} and all(int(v) > 0 for v in result["call_columns"].values()),
         "server identity run made no Pixel calls at both widths")
    config, identity = read(run / "CONFIG.json"), read(run / "IDENTITY.json")
    need(config["phone_worker"] == worker["worker_path"] and config["phone_model"] == worker["shard_path"]
         and [config["phone_library_dir"]] == worker["library_directories"]
         and config.get("phone_environment", {}) == worker["worker_environment"],
         "server identity run used another worker configuration")
    observed = {line.split(None, 1)[1].strip(): "sha256:" + line.split()[0]
                for line in identity["phone_hashes"].splitlines() if line.strip()}
    need(all(observed.get(path) == value for path, value in worker["expected_sha256_by_path"].items()),
         "server identity phone hashes differ from the pins")
    need("sha256:" + identity["server_sha256"] == host_binary_sha256, "server identity run used another server")
    need(identity.get("aoa_bridge", {}).get("options_sha256") == aoa.options_sha256
         and identity["aoa_bridge"]["configuration"]["relay_sha256"] == aoa.relay_sha256,
         "server identity run used another AOA bridge configuration")
    return {"status": "PASS", "run": str(run), "result_sha256": file_sha256(run / "RESULT.json"),
            "identity_sha256": file_sha256(run / "IDENTITY.json"), "phone_calls": result["phone_calls"],
            "call_columns": result["call_columns"], "outputs": 4, "tokens_each": result["output_tokens_each"],
            "note": "Pixel-only llama-server over the AOA bridge, configured worker, identical tokens"}


def aoa_links(fragment: dict, trip: dict, device: str) -> list[dict]:
    """The Pixel's host<->phone link rows re-measured over the bridge (same shape as the adb-forward rows)."""
    need("effective_transfer_bytes_per_s" in trip and "one_direction_fixed_us" in trip,
         "the round-trip receipt has no rows-1 and rows-4 calibration")
    inbound = [row for row in fragment["links"] if row["target_device"] == device]
    need(len(inbound) == 1, "the adb-tcp bundle needs exactly one host -> Pixel link row")
    template, host = inbound[0], inbound[0]["source_device"]
    return [{**template, "link_id": LINK_PREFIX + direction, "source_device": source, "target_device": target,
             "bandwidth_bytes_per_s": trip["effective_transfer_bytes_per_s"],
             "fixed_latency_us": trip["one_direction_fixed_us"], "status": "measured"}
            for direction, source, target in (("out", host, device), ("in", device, host))]


def build_bundle(args: argparse.Namespace) -> Path:
    adb_path = Path(args.adb_evidence)
    adb = read(adb_path)
    old = load_helper_evidence(adb_path)
    need(old.identity.transport == "adb-tcp" and old.aoa_bridge is None, "the source bundle must be adb-tcp")
    aoa = AoaBridgeConfiguration.from_json(read(args.aoa_bridge))
    need(file_sha256(args.bridge_script) == aoa.bridge_script_sha256, "the deployed host bridge differs from its pin")
    need(aoa.usb_sysfs_device == old.identity.hardware_identity["phone_usb_sysfs_device"],
         "the AOA bridge pins another USB port")
    ab_dir = Path(args.ab_dir)
    trip, identity_receipt = read(ab_dir / "aoa-bridge-round-trip.json"), read(ab_dir / "aoa-bridge-byte-identity.json")
    lifecycle = read(ab_dir / "aoa-bridge-scheduler-launched-session.json")
    need(trip["status"] == identity_receipt["status"] == lifecycle["status"] == "PASS", "an A/B receipt is not PASS")
    check_measured_configuration(ab_dir, trip, aoa)
    worker = adb["worker"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    receipts = {
        "usb-link-speed": usb_receipt(ab_dir, trip, worker["serial"], aoa.usb_sysfs_device,
                                      old.identity.minimum_usb_speed_mbps),
        "server-token-identity": server_receipt(Path(args.server_identity), worker,
                                                old.identity.software_identity["host_binary_sha256"], aoa),
        "aoa-bridge-round-trip": trip, "aoa-bridge-byte-identity": identity_receipt,
        "scheduler-launched-session": lifecycle,
    }
    paths = {}
    for kind, value in receipts.items():
        path = output / (kind + ".json")
        write(path, value)
        paths[kind] = str(path)
    paths["numerical-rows-1-2-4"] = adb["receipt_paths"]["numerical-rows-1-2-4"]  # same worker, same receipt
    need(set(paths) == set(IDENTITY_REQUIREMENTS["aoa-bridge"]["receipts"]), "receipt set differs")
    transport = copy.deepcopy(adb["transport_identity"])
    transport.update(transport="aoa-bridge", transport_generation=AOA_BRIDGE_TRANSPORT_GENERATION)
    transport["hardware_identity"]["aoa_usb_identity"] = AOA_USB_IDENTITY
    transport["software_identity"].update(phone_relay_sha256=aoa.relay_sha256,
                                          host_bridge_sha256=aoa.bridge_script_sha256,
                                          aoa_bridge_options_sha256=aoa.options_sha256)
    transport["receipts"] = {kind: file_sha256(Path(path)) for kind, path in paths.items()}
    fragment = copy.deepcopy(adb["profile_fragment"])
    device = worker["device_id"]
    if getattr(args, "link_calibration", "aoa") == "aoa":
        fragment["links"] = [row for row in fragment["links"]
                             if device not in (row["source_device"], row["target_device"])]
        fragment["links"] += aoa_links(adb["profile_fragment"], trip, device)
        for row in fragment["links"]:
            if row["link_id"].startswith(LINK_PREFIX):
                row["evidence_ids"] = [transport["receipts"]["aoa-bridge-round-trip"]]
    bundle = {**copy.deepcopy(adb), "transport_identity": transport, "receipt_paths": paths,
              "host_software_paths": {**adb["host_software_paths"], "host_bridge_sha256": str(Path(args.bridge_script))},
              "profile_fragment": fragment, "aoa_bridge": aoa.to_json(),
              "derived_from": {"adb_evidence": str(adb_path), "adb_evidence_sha256": file_sha256(adb_path),
                               "link_calibration": getattr(args, "link_calibration", "aoa")}}
    path = output / "PIXEL_EVIDENCE_AOA.json"
    write(path, bundle)
    load_helper_evidence(path)  # the scheduler's own loader must accept it
    return path


def derive_arm(args: argparse.Namespace) -> Path:
    source, target = Path(args.inputs), Path(args.output)
    evidence_path = Path(args.evidence).resolve()
    evidence = load_helper_evidence(evidence_path)
    need(evidence.aoa_bridge is not None, "the evidence is not an aoa-bridge bundle")
    shutil.copytree(source, target)
    rig, campaign, evidence_manifest = (read(target / name) for name in ("rig.json", "campaign.json", "evidence.json"))
    rows = [row for row in rig.get("helper_phones", []) if row["device_id"] == evidence.worker.device_id]
    need(len(rows) == 1, "the arm has no helper row for the evidence's device")
    rows[0]["transport"] = "aoa-bridge"
    rows[0]["aoa_bridge"] = evidence.aoa_bridge.to_json()
    need(rows[0]["forward_port"] == evidence.worker.forward_port, "the arm's forward port differs from the evidence")
    evidence_manifest["helper_phone_evidence_paths"][evidence.worker.device_id] = str(evidence_path)
    campaign["campaign_id"] = campaign["campaign_id"] + "-aoa"
    for name in ("rig", "models", "evidence"):
        key = name + "_manifest_path"
        if key in campaign:
            campaign[key] = str(target.resolve() / (name + ".json"))
    if "transport_qualification_identity_path" in evidence_manifest:
        evidence_manifest["transport_qualification_identity_path"] = str(
            target.resolve() / Path(evidence_manifest["transport_qualification_identity_path"]).name)
    write(target / "rig.json", rig)
    write(target / "campaign.json", campaign)
    write(target / "evidence.json", evidence_manifest)
    RigManifest.from_json(rig, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("bundle")
    for name in ("--adb-evidence", "--ab-dir", "--server-identity", "--aoa-bridge", "--bridge-script",
                 "--output-dir"):
        bundle.add_argument(name, required=True)
    bundle.add_argument("--link-calibration", choices=("aoa", "adb"), default="aoa")
    arm = commands.add_parser("derive-arm")
    for name in ("--inputs", "--evidence", "--output"):
        arm.add_argument(name, required=True)
    args = parser.parse_args(argv)
    path = build_bundle(args) if args.command == "bundle" else derive_arm(args)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
