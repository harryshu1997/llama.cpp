#!/usr/bin/env python3
"""WS10 Phase B: turn the AOA bridge qualification into a helper evidence bundle and a campaign arm.

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.prepare_aoa_evidence bundle \\
        --adb-evidence <qualified adb-tcp PIXEL_EVIDENCE.json> --ab-dir <pixel_transport_ab OUT (analysed)> \\
        --server-identity <qualify_pixel_server_transport OUT> --aoa-bridge AOA_BRIDGE.json \\
        --bridge-script <deployed adapters/aoa_bridge.py> --output-dir DIR
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.prepare_aoa_evidence derive-arm \\
        --inputs <inputs-two-phone-ATT> --evidence DIR/PIXEL_EVIDENCE_AOA.json --output <inputs-two-phone-ATT-aoa>
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.prepare_aoa_evidence rebind-shard \\
        --evidence DIR/PIXEL_EVIDENCE_AOA.json --shard-index NEW/PIXEL_FFN_SHARDS.json \\
        --phone-shard-path /data/local/tmp/.../QWEN_PACKED_1_24.ffn.gguf --server-identity <AOA run, new shard> \\
        (--numerical-dir <suite run, new shard> | --carry-numerical) --output-dir DIR2
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.prepare_aoa_evidence retarget-model \\
        --evidence DIR/PIXEL_EVIDENCE_AOA.json --shard-index NEW/PIXEL_FFN_SHARDS.json \\
        --phone-shard-path /data/local/tmp/.../Q8_21_40.ffn.gguf --server-identity <AOA run, new model> --output-dir DIR2
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.prepare_aoa_evidence identity-config \\
        --base <qualified server-identity CONFIG.json> --shard-index NEW/PIXEL_FFN_SHARDS.json \\
        --phone-shard-path /data/local/tmp/.../QWEN_PACKED_1_24.ffn.gguf --output NEW_CONFIG.json
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.prepare_aoa_evidence probe-config \\
        --base AB_CONFIG.json --shard-index NEW/PIXEL_FFN_SHARDS.json --phone-shard-path ... \\
        --input-dir NEW_INPUTS --tokens 1=1200,2=300 --output PROBE_CONFIG.json

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

``rebind-shard`` (WS12) re-qualifies a qualified aoa-bridge bundle for another layer mask of the same worker
(measured placement moves layers to the Pixel): new shard (path, sha256, mask from its one-record index) and
fresh receipts for what depends on it -- ``server-token-identity`` (a Pixel-only llama-server run over the
bridge with the new shard and mask) and ``numerical-rows-1-2-4`` (a numerical suite run covering exactly the
new layers, or with ``--carry-numerical`` the old receipt plus an annex naming the policy: same worker and
libraries, new layers vouched for end to end by the token identity -- an evidence-policy choice, never the
default). Transport identity, relay/bridge/options pins, the round-trip, byte-identity and launched-session
receipts and the cost profile are kept (they qualify the link and the worker binary, not the layers).
``retarget-model`` (2026-10-08) re-qualifies a bundle for another model on the same phone, port, relay, bridge, server
and worker binary: fresh server-token-identity and plain FFN numerical rows (``ffn_numerics`` in the identity run), the
link and session receipts carried over with an annex, the FFN kernel rate re-measured. ``identity-config`` derives the
``qualify_pixel_server_transport`` config for that run from the qualified one:
only the shard (path, pin) and the served layers change (tooling, no hand edits). ``probe-config`` derives a
``pixel_transport_ab`` config for the sustained-load placement probe (the calibration arm only, every new layer
called once per token at the serving cadence, long segments); its per-layer inputs are TIMING-ONLY copies of one
qualified layer's captures (``timing_inputs``), so its dumps are not a numerical receipt.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

from research_dev.scheduler._internal.types import canonical_sha256
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
    config, identity = read(run / "CONFIG.json"), read(run / "IDENTITY.json")
    if worker.get("fused") is None:
        # the run's helper widths (legs_columns; the 14B default 8,704 / 17,408)
        widths = {str(value) for value in config.get("legs_columns", (0, 8704, 17408, 0)) if value}
        need(len(widths) == 2 and set(result["call_columns"]) == widths
             and all(int(v) > 0 for v in result["call_columns"].values()),
             "server identity run made no Pixel calls at both widths")
    else:
        # WS15: a fused worker serves full width only; its helper legs are fused calls
        need(set(result["call_columns"]) == {str(worker["columns"])} and int(result.get("fused_calls", 0)) > 0,
             "server identity run made no fused Pixel calls")
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
            **({"fused_calls": result["fused_calls"]} if worker.get("fused") is not None else {}),
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


NUMERICAL_ARM = "04-sdot-pair-dynamic64"


def _mask_layers(mask: int) -> list[int]:
    return [layer for layer in range(64) if mask >> layer & 1]


def numerical_suite_receipt(suite_dir: Path, worker: dict, old_receipt: dict) -> dict:
    """A numerical suite run of the new shard: PASS, rel-L2 < 1e-3, rows 1/2/4, exactly the new layers."""
    suite = read(suite_dir / "SUITE_RESULT.json")
    arm = suite.get("arms", {}).get(NUMERICAL_ARM)
    need(suite.get("status") == "PASS" and arm is not None and arm.get("numerical_status") == "PASS",
         "the numerical suite failed or lacks " + NUMERICAL_ARM)
    need(arm["max_relative_l2"] < 0.001, "numerical error too large")
    shard_sha = worker["expected_sha256_by_path"][worker["shard_path"]]
    need(arm["arm"].get("model_path") == worker["shard_path"]
         and "sha256:" + str(arm["arm"].get("model_sha256", "")).removeprefix("sha256:") == shard_sha,
         "the numerical suite ran another shard")
    calls = read(suite_dir / "SUITE_CALLS.json")[NUMERICAL_ARM]
    need({1, 2, 4} <= {row["tokens"] for row in calls}, "numerical suite lacks rows 1, 2 or 4")
    need(sorted({row["layer"] for row in calls}) == _mask_layers(worker["layer_mask"]),
         "the numerical suite did not cover exactly the new layers")
    hashes = {"sha256:" + value.removeprefix("sha256:") for value in read(suite_dir / "EXPECTED_PHONE_HASHES.json").values()}
    libraries = {value for path, value in worker["expected_sha256_by_path"].items()
                 if path not in (worker["worker_path"], worker["shard_path"])}
    need(shard_sha in hashes and libraries <= hashes, "the numerical suite staged other libraries or shard")
    value = {"status": "PASS", "suite_sha256": file_sha256(suite_dir / "SUITE_RESULT.json"),
             "calls_sha256": file_sha256(suite_dir / "SUITE_CALLS.json"),
             "worker_hashes_sha256": file_sha256(suite_dir / "EXPECTED_PHONE_HASHES.json"),
             "max_relative_l2": arm["max_relative_l2"], "rows": sorted({row["tokens"] for row in calls}),
             "layers": _mask_layers(worker["layer_mask"])}
    if worker["expected_sha256_by_path"][worker["worker_path"]] not in hashes:
        # the configured worker inherits the suite's qualification through the old receipt's byte-identity bridge
        need("byte_identity_bridge" in old_receipt, "the configured worker is neither in the suite nor bridged")
        value["byte_identity_bridge"] = old_receipt["byte_identity_bridge"]
    return value


def _new_shard(index_path: str, phone_path: str, artifact: str, columns: int) -> tuple[dict, int]:
    index = read(index_path)
    need(index.get("schema") == "s42-ffn-shard-index-v1" and len(index.get("shards", ())) == 1,
         "the new shard index must hold one full-width record")
    record = index["shards"][0]
    need(index["parent_sha256"] == record["parent_sha256"] == artifact,
         "the new shard is not a shard of the qualified artifact")
    need(int(record["columns"]) == columns and Path(phone_path).name == record["path"],
         "the new shard's width or file name differs from the index")
    mask = int(record["layer_mask"], 16)
    need(0 < mask < 1 << 64, "the new shard has no layers")
    return record, mask


def _fused_record(record: dict) -> tuple[int, int] | None:
    """(fused layers, next layers) of a WS15 fused shard record, None for an FFN-only record."""
    if record.get("kernel", "ffn") == "ffn":
        need("fused_layer_mask" not in record, "an FFN-only shard record carries fused layers")
        return None
    need(record.get("kernel") == "attn_proj:fused", "unknown shard kernel " + str(record.get("kernel")))
    fused, following = int(record["fused_layer_mask"], 16), int(record.get("fused_next_mask", "0"), 16)
    need(0 < fused and not fused & ~int(record["layer_mask"], 16) and not following & ~fused,
         "fused shard record masks are inconsistent")
    return fused, following


def _spec(mask: int) -> str:
    return ",".join(str(layer) for layer in _mask_layers(mask))


def fused_numerics_receipt(run: Path, fused: int, following: int) -> dict:
    """The fused numerical rows recorded by the server identity run on its own pinned worker session."""
    numerics = read(run / "NUMERICS.json")
    need(numerics.get("schema") == "s43-fused-numerics-v1" and numerics.get("status") == "PASS"
         and numerics["worst_rel_l2"] <= 1e-3 and {1, 2, 4} <= set(numerics["rows"])
         and sum(1 << layer for layer in numerics["layers"]) == fused
         and sum(1 << layer for layer in numerics["next_layers"]) == following
         and int(numerics["fused_hello"]["fused_layer_mask"], 16) & fused == fused,
         "fused numerical rows failed or covered other layers")
    return {"status": "PASS", "kernel": "attn_proj:fused", "fused_layer_mask": fused, "fused_next_mask": following,
            "numerics": str(run / "NUMERICS.json"), "numerics_sha256": file_sha256(run / "NUMERICS.json"),
            "worst_rel_l2": numerics["worst_rel_l2"], "rows": numerics["rows"], "layers": numerics["layers"],
            "next_layers": numerics["next_layers"], "calls": numerics["calls"],
            "fused_weight_hash": numerics["fused_hello"]["fused_weight_hash"], "inputs": numerics["inputs"],
            "reference": "float32 numpy from the f16 proxy (research_dev/scheduler/native/fused_numerics.py)"}


def identity_config(args: argparse.Namespace) -> Path:
    base = read(args.base)
    phone = base["phone"]
    record, mask = _new_shard(args.shard_index, str(args.phone_shard_path), phone["artifact_sha256"], phone["columns"])
    config = copy.deepcopy(base)
    pins = {path: value for path, value in config["expected_phone_sha256"].items() if path != base["phone_model"]}
    pins[str(args.phone_shard_path)] = record["shard_sha256"].removeprefix("sha256:")
    config.update(phone_model=str(args.phone_shard_path), expected_phone_sha256=pins)
    config["phone"]["layers"] = _mask_layers(mask)
    # the server's split mask must name the new layers too, or the first helper control exceeds its resident slice
    environment = dict(config.get("server_environment") or {})
    if "S41_SERVER_FFN_LAYER_MASK" in environment:
        old = int(environment["S41_SERVER_FFN_LAYER_MASK"])
        need(old == sum(1 << layer for layer in phone["layers"]),
             "the base server layer mask differs from the base phone layers")
        environment["S41_SERVER_FFN_LAYER_MASK"] = str(mask)
        config["server_environment"] = environment
    fused = _fused_record(record)
    if fused is not None:
        # WS15: full-width helper legs run fused calls; numerical rows of the fused kernel run on the same session
        fused_mask, next_mask = fused
        config["server_environment"] = {**config["server_environment"],
                                        "S41_SERVER_FFN_FUSED_LAYER_MASK": str(fused_mask),
                                        "S41_SERVER_FFN_FUSED_NEXT_MASK": str(next_mask)}
        config["legs_columns"] = [0, phone["columns"], phone["columns"], 0]
        config["fused_numerics"] = {"layers": _spec(fused_mask), "next": _spec(next_mask) if next_mask else "",
                                    "calls": 3 * (bin(fused_mask).count("1") + bin(next_mask).count("1"))}
    output = Path(args.output)
    need(not output.exists(), "the output config exists")
    write(output, config)
    return output


def probe_config(args: argparse.Namespace) -> Path:
    base = read(args.base)
    phone = base["phone"]
    record, mask = _new_shard(args.shard_index, str(args.phone_shard_path), phone["artifact_sha256"], phone["columns"])
    layers = _mask_layers(mask)
    source = Path(base["input_dir"])
    template = phone["layers"][0]
    rows = {int(key): int(value) for key, value in (item.split("=") for item in args.tokens.split(","))}
    need(rows and all(rows_ >= 1 and tokens >= 24 for rows_, tokens in rows.items()), "--tokens needs ROWS=TOKENS >= 24")
    inputs = Path(args.input_dir)
    inputs.mkdir(parents=True, exist_ok=False)
    for layer in layers:
        for row in range(max(rows)):
            shutil.copyfile(source / f"input-layer{template}-row{row}.f16", inputs / f"input-layer{layer}-row{row}.f16")
    config = copy.deepcopy(base)
    pins = {path: value for path, value in phone["common_sha256"].items() if path != phone["shard_path"]}
    pins[str(args.phone_shard_path)] = record["shard_sha256"]
    config["phone"].update(shard_path=str(args.phone_shard_path), common_sha256=pins, layers=layers)
    config["cadence"] = {**base["cadence"], "calls_per_token": len(layers)}
    config["segments"] = [{"name": f"p{rows_}", "rows": rows_, "tokens": tokens, "warmup_tokens": 3}
                          for rows_, tokens in sorted(rows.items())]
    config["arms"] = [arm for arm in base["arms"] if arm["name"] == base["calibration_arm"]]
    need(len(config["arms"]) == 1, "the base config has no calibration arm")
    config["repeats"] = 1
    config["input_dir"] = str(inputs)
    config["timing_inputs"] = {"copied_from_layer": template, "source": str(source),
                               "note": "placement probe: timing only, not a numerical or byte-identity receipt"}
    output = Path(args.output)
    need(not output.exists(), "the output config exists")
    write(output, config)
    return output


def ffn_numerics_receipt(run: Path, worker: dict) -> dict:
    """The plain FFN numerical rows the server identity run recorded on its own pinned worker session
    (native/ffn_numerics.py, ggml-cpu reference from the parent GGUF): PASS, rel-L2 <= 1e-3, rows 1/2/4, exactly the
    worker's layers, the worker's artifact."""
    numerics = read(run / "FFN_NUMERICS.json")
    need(numerics.get("schema") == "s43-ffn-numerics-v1" and numerics.get("status") == "PASS"
         and numerics["worst_rel_l2"] <= 1e-3 and {1, 2, 4} <= set(numerics["rows"])
         and sum(1 << layer for layer in numerics["layers"]) == worker["layer_mask"]
         and numerics.get("artifact_sha256") == worker["artifact_sha256"],
         "FFN numerical rows failed or covered other layers")
    return {"status": "PASS", "kernel": "ffn", "numerics": str(run / "FFN_NUMERICS.json"),
            "numerics_sha256": file_sha256(run / "FFN_NUMERICS.json"), "worst_rel_l2": numerics["worst_rel_l2"],
            "worst_rel_l2_f32_reference": numerics.get("worst_rel_l2_f32_reference"), "rows": numerics["rows"],
            "layers": numerics["layers"], "calls": numerics["calls"], "weight_types": numerics.get("weight_types"),
            "inputs": numerics["inputs"], "reference": numerics["reference"] + " (research_dev/scheduler/native/"
            "ffn_numerics.py)"}


# link and session receipts: they qualify the USB port, the AOA relay / host bridge / keep-awake options and the
# scheduler's bridge session lifecycle of the worker binary -- not the model, shard, libraries or environment
CARRIED_RECEIPTS = ("usb-link-speed", "aoa-bridge-round-trip", "aoa-bridge-byte-identity", "scheduler-launched-session")


def _full_width_compute_ms(run: Path, columns: int) -> float:
    """Mean compute of the identity run's full-width rows-1 helper calls (server S41SERVERFFNSHAPE)."""
    from research_dev.scheduler._internal.layer_placement_io import parse_shape_line
    rows = [row for row in (parse_shape_line(line) for line in read(run / "SERVER_FFN_SHAPES.json")) if row
            and int(row["columns"]) == columns and int(row["tokens"]) == 1 and int(row["calls"]) > 0]
    need(len(rows) == 1, "the identity run has no full-width rows-1 helper call summary")
    return float(rows[0]["compute_mean_ms"])


def retarget_model(args: argparse.Namespace) -> Path:
    """A qualified aoa-bridge bundle re-qualified for ANOTHER model (artifact, width, shard, libraries, worker
    environment) on the same phone, USB port, relay, host bridge, server and worker binary (2026-10-08, Qwen3-32B).
    Fresh: server-token-identity (the new configuration over the bridge) and numerical-rows-1-2-4 (plain FFN rows of
    that run, ggml-cpu reference). Carried over with an annex: the link and session receipts (CARRIED_RECEIPTS),
    which do not depend on the model. The worker block and pins come from the identity run; the cost profile's
    FFN kernel rate is re-measured from its full-width calls (manifest-equivalent weight bytes per layer).
    --host-software NAME=PATH (as rebind-shard) re-pins a host file to another build, e.g. the prompt-anchor server
    library: the identity run must have launched exactly that server with the pinned files' directory first on its
    LD_LIBRARY_PATH (the server's RUNPATH may name the production build)."""
    source = Path(args.evidence).resolve()
    old = load_helper_evidence(source)
    need(old.aoa_bridge is not None, "retarget-model needs a qualified aoa-bridge bundle")
    raw = read(source)
    run = Path(args.server_identity)
    config = read(run / "CONFIG.json")
    phone = config["phone"]
    phone_path = str(args.phone_shard_path)
    record, mask = _new_shard(args.shard_index, phone_path, phone["artifact_sha256"], phone["columns"])
    need(_fused_record(record) is None, "retarget-model qualifies plain FFN shards only")
    pins = {path: "sha256:" + value.removeprefix("sha256:") for path, value in config["expected_phone_sha256"].items()}
    need(config["phone_model"] == phone_path and pins.get(phone_path) == record["shard_sha256"],
         "the identity run served another shard")
    need(sum(1 << layer for layer in phone["layers"]) == mask, "the identity run served other layers")
    old_worker = raw["worker"]
    worker_sha = pins.get(config["phone_worker"])
    need(worker_sha == old_worker["expected_sha256_by_path"][old_worker["worker_path"]],
         "retarget-model keeps the qualified worker binary (its session lifecycle receipt is carried over)")
    worker = copy.deepcopy(old_worker)
    worker.update(artifact_sha256=phone["artifact_sha256"], n_embd=int(phone["n_embd"]), columns=int(phone["columns"]),
                  # the identity run's worker quantum (qualify_pixel_server_transport: config phone.quantum)
                  column_quantum=int(phone.get("quantum", phone["columns"])), worker_path=config["phone_worker"],
                  library_directories=[config["phone_library_dir"]], shard_path=phone_path, layer_mask=mask,
                  worker_environment=dict(config.get("phone_environment", {})), expected_sha256_by_path=pins)
    need({int(value) for value in config.get("legs_columns", ()) if value} == {worker["columns"] // 2, worker["columns"]},
         "the identity run's helper legs must be half and full of the new width")
    host_paths = dict(raw["host_software_paths"])
    host_software = getattr(args, "host_software", None) or ()
    for text in host_software:
        name, _, value = text.partition("=")
        need(name in host_paths and value, "--host-software must replace a pinned host file: " + text)
        host_paths[name] = str(Path(value).resolve())
    if host_software:
        server_path = Path(config.get("server", "")).resolve()
        need(server_path == Path(host_paths["host_binary_sha256"]),
             "the identity run launched another server than --host-software host_binary_sha256")
        library_path = str(config.get("server_environment", {}).get("LD_LIBRARY_PATH", "")).split(":")
        need(all(Path(host_paths[name]).parent == server_path.parent for name in host_paths
                 if name in ("host_binary_sha256", "host_impl_sha256"))
             and Path(library_path[0]).resolve() == server_path.parent,
             "the identity run did not load the re-pinned server library (its directory must lead LD_LIBRARY_PATH)")
    host_binary = (file_sha256(Path(host_paths["host_binary_sha256"]))
                   if host_software else old.identity.software_identity["host_binary_sha256"])
    server = server_receipt(run, worker, host_binary, old.aoa_bridge)
    numerical = ffn_numerics_receipt(run, worker)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    paths = {kind: raw["receipt_paths"][kind] for kind in CARRIED_RECEIPTS}
    for kind, value in (("server-token-identity", server), ("numerical-rows-1-2-4", numerical)):
        path = output / (kind + ".json")
        write(path, value)
        paths[kind] = str(path)
    need(set(paths) == set(IDENTITY_REQUIREMENTS["aoa-bridge"]["receipts"]), "receipt set differs")
    transport = copy.deepcopy(raw["transport_identity"])
    software = {key: value for key, value in transport["software_identity"].items()
                if not key.startswith("phone_library_sha256:")}
    software.update(phone_worker_sha256=worker_sha, phone_shard_sha256=record["shard_sha256"],
                    worker_environment_sha256=canonical_sha256(worker["worker_environment"]),
                    **{"phone_library_sha256:" + path: value for path, value in pins.items()
                       if path not in (worker["worker_path"], phone_path)})
    for name, path in host_paths.items():
        if path != raw["host_software_paths"][name]:
            software[name] = file_sha256(Path(path))
    transport["software_identity"] = software
    transport["receipts"] = {kind: file_sha256(Path(path)) for kind, path in paths.items()}
    compute_ms = _full_width_compute_ms(run, worker["columns"])
    layer_bytes = int(record["shard_bytes"]) // bin(mask).count("1")
    fragment = copy.deepcopy(raw["profile_fragment"])
    kernels = [row for row in fragment["kernels"] if row["kernel_id"].endswith(":ffn")]
    need(len(kernels) == 1, "the source bundle needs exactly one FFN kernel row")
    kernels[0].update(effective_bytes_per_s=round(layer_bytes / (compute_ms / 1000.0)),
                      effective_ops_per_s=round(2 * 3 * worker["n_embd"] * worker["columns"] / (compute_ms / 1000.0)),
                      evidence_ids=[server["result_sha256"]] + [value for value in kernels[0]["evidence_ids"]
                                                                if not value.startswith("sha256:")])
    bundle = {key: value for key, value in copy.deepcopy(raw).items() if key not in ("fused", "rebound_from")}
    bundle.update(worker=worker, transport_identity=transport, receipt_paths=paths, profile_fragment=fragment,
                  host_software_paths=host_paths,
                  retargeted_from={
                      "evidence": str(source), "evidence_sha256": file_sha256(source),
                      "shard_index": str(Path(args.shard_index).resolve()),
                      "shard_index_sha256": file_sha256(Path(args.shard_index)),
                      "server_identity": str(run), "numerical": "fresh-ffn-numerics",
                      "carried_receipts": {kind: {"path": raw["receipt_paths"][kind],
                                                  "sha256": file_sha256(Path(raw["receipt_paths"][kind]))}
                                           for kind in CARRIED_RECEIPTS},
                      "policy": "same phone, USB port, AOA relay / host bridge / options, server and worker binary; "
                                "the model, shard, libraries and environment are qualified afresh by the server "
                                "token identity and the FFN numerical rows over the bridge; link and session "
                                "lifecycle receipts carried over (they do not depend on the model)",
                      "kernel_rate": {"full_width_compute_ms": compute_ms, "layer_bytes": layer_bytes},
                      **({"host_software": {name: path for name, path in host_paths.items()
                                            if path != raw["host_software_paths"][name]}} if host_software else {})})
    path = output / "PIXEL_EVIDENCE_AOA.json"
    write(path, bundle)
    load_helper_evidence(path)  # the scheduler's own loader must accept it
    return path


def rebind_shard(args: argparse.Namespace) -> Path:
    source = Path(args.evidence).resolve()
    repinned = {text.partition("=")[0] for text in (getattr(args, "host_software", None) or ())}
    old = load_helper_evidence(source, repinned_host_software=repinned)
    need(old.aoa_bridge is not None, "rebind-shard needs a qualified aoa-bridge bundle")
    raw = read(source)
    worker = copy.deepcopy(raw["worker"])
    phone_path = str(args.phone_shard_path)
    record, mask = _new_shard(args.shard_index, phone_path, worker["artifact_sha256"], worker["columns"])
    del worker["expected_sha256_by_path"][worker["shard_path"]]
    worker["shard_path"] = phone_path
    worker["layer_mask"] = mask
    worker["expected_sha256_by_path"][phone_path] = record["shard_sha256"]
    rebound_worker = None
    if getattr(args, "worker", None):
        # 2026-10-02 rig: the fused kernel ships in a new worker binary; re-pin it only to the exact file the server
        # identity run used and verified on the phone (its sha256sum listing in IDENTITY.json phone_hashes)
        worker_path, _, digest = args.worker.partition("=")
        need(worker_path.startswith("/") and digest.startswith("sha256:") and len(digest) == 71,
             "--worker needs PHONE_PATH=sha256:HEX")
        listing = read(Path(args.server_identity) / "IDENTITY.json").get("phone_hashes", "")
        verified = {parts[1]: parts[0] for parts in (line.split() for line in str(listing).splitlines()) if len(parts) == 2}
        need(verified.get(worker_path) == digest.removeprefix("sha256:"),
             "the server identity run did not verify this worker binary on the phone")
        need(not args.carry_numerical, "a new worker binary cannot carry the old numerical receipt over")
        rebound_worker = {"previous_path": worker["worker_path"],
                          "previous_sha256": worker["expected_sha256_by_path"].get(worker["worker_path"]),
                          "path": worker_path, "sha256": digest}
        worker["expected_sha256_by_path"].pop(worker["worker_path"], None)
        worker["worker_path"] = worker_path
        worker["expected_sha256_by_path"][worker_path] = digest
    fused = _fused_record(record)
    # WS15: a fused kernel needs a server with the fused path; --host-software NAME=PATH re-pins it
    host_paths = dict(raw["host_software_paths"])
    host_software = getattr(args, "host_software", None) or ()
    for text in host_software:
        name, _, value = text.partition("=")
        need(name in host_paths and value, "--host-software must replace a pinned host file: " + text)
        host_paths[name] = str(Path(value).resolve())
    host_binary = (file_sha256(Path(host_paths["host_binary_sha256"]))
                   if host_software else old.identity.software_identity["host_binary_sha256"])
    server = server_receipt(Path(args.server_identity), {**worker, "fused": fused}, host_binary, old.aoa_bridge)
    config = read(Path(args.server_identity) / "CONFIG.json")["phone"]
    need(sum(1 << layer for layer in config["layers"]) == mask and config["artifact_sha256"] == worker["artifact_sha256"]
         and config["columns"] == worker["columns"], "the server identity run served other layers")
    old_numerical_path = Path(raw["receipt_paths"]["numerical-rows-1-2-4"])
    old_numerical = read(old_numerical_path)
    if fused is not None:
        # the old receipt never covered the projections: a fused kernel always needs its own numerical rows
        need(not args.carry_numerical, "a fused shard cannot carry the FFN numerical receipt over")
        numerical = fused_numerics_receipt(Path(args.server_identity), *fused)
    elif args.carry_numerical:
        numerical = {**old_numerical, "status": "PASS", "carried_over": {
            "receipt": str(old_numerical_path), "receipt_sha256": file_sha256(old_numerical_path),
            "qualified_layer_mask": raw["worker"]["layer_mask"], "layer_mask": mask,
            "policy": "same worker binary, libraries and environment; the new layers are vouched for end to end by "
                      "server-token-identity over the new mask (--carry-numerical, an explicit evidence-policy choice)"}}
    else:
        need(args.numerical_dir is not None, "--numerical-dir or --carry-numerical is required")
        numerical = numerical_suite_receipt(Path(args.numerical_dir), worker, old_numerical)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    paths = dict(raw["receipt_paths"])
    for kind, value in (("server-token-identity", server), ("numerical-rows-1-2-4", numerical)):
        path = output / (kind + ".json")
        write(path, value)
        paths[kind] = str(path)
    transport = copy.deepcopy(raw["transport_identity"])
    transport["software_identity"]["phone_shard_sha256"] = record["shard_sha256"]
    if rebound_worker is not None:
        transport["software_identity"]["phone_worker_sha256"] = rebound_worker["sha256"]
    for name, path in host_paths.items():
        # a re-pinned host file is re-hashed even at an unchanged path: a server rebuilt in place (2026-10-04)
        # keeps its path, and the stale digest would fail the loader's host software check
        if path != raw["host_software_paths"][name] or name in repinned:
            transport["software_identity"][name] = file_sha256(Path(path))
    transport["receipts"] = {kind: file_sha256(Path(path)) for kind, path in paths.items()}
    bundle = {**copy.deepcopy(raw), "worker": worker, "transport_identity": transport, "receipt_paths": paths,
              "host_software_paths": host_paths,
              **({"fused": {"kernel": "attn_proj:fused", "fused_layer_mask": fused[0], "fused_next_mask": fused[1],
                            "fused_weight_hash": numerical["fused_weight_hash"]}} if fused is not None else {}),
              "rebound_from": {"evidence": str(source), "evidence_sha256": file_sha256(source),
                               "shard_index": str(Path(args.shard_index).resolve()),
                               "shard_index_sha256": file_sha256(Path(args.shard_index)),
                               "numerical": ("fused-numerics" if fused is not None else
                                             "carried-over" if args.carry_numerical else "fresh-suite"),
                               **({"worker": rebound_worker} if rebound_worker else {})}}
    path = output / "PIXEL_EVIDENCE_AOA.json"
    write(path, bundle)
    load_helper_evidence(path)  # the scheduler's own loader must accept it
    return path


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
    rebind = commands.add_parser("rebind-shard")
    for name in ("--evidence", "--shard-index", "--phone-shard-path", "--server-identity", "--output-dir"):
        rebind.add_argument(name, required=True)
    numerical = rebind.add_mutually_exclusive_group(required=False)
    numerical.add_argument("--numerical-dir")
    numerical.add_argument("--carry-numerical", action="store_true")
    rebind.add_argument("--host-software", action="append", default=[], metavar="NAME=PATH",
                        help="WS15: re-pin a host file (e.g. host_binary_sha256=<fused llama-server>)")
    rebind.add_argument("--worker", metavar="PHONE_PATH=sha256:HEX",
                        help="re-pin the phone worker binary (e.g. the fused worker); the server identity run must "
                             "have used and verified exactly this file")
    retarget = commands.add_parser("retarget-model")
    for name in ("--evidence", "--shard-index", "--phone-shard-path", "--server-identity", "--output-dir"):
        retarget.add_argument(name, required=True)
    retarget.add_argument("--host-software", action="append", default=[], metavar="NAME=PATH",
                          help="re-pin a host file (e.g. host_impl_sha256=<anchor build>/libllama-server-impl.so); "
                               "the identity run must have launched that server and loaded its directory first")
    config = commands.add_parser("identity-config")
    for name in ("--base", "--shard-index", "--phone-shard-path", "--output"):
        config.add_argument(name, required=True)
    probe = commands.add_parser("probe-config")
    for name in ("--base", "--shard-index", "--phone-shard-path", "--input-dir", "--tokens", "--output"):
        probe.add_argument(name, required=True)
    args = parser.parse_args(argv)
    path = {"bundle": build_bundle, "derive-arm": derive_arm, "rebind-shard": rebind_shard,
            "retarget-model": retarget_model, "identity-config": identity_config,
            "probe-config": probe_config}[args.command](args)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
