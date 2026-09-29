#!/usr/bin/env python3
"""Materialize the measured RTX 4060 Ti plus OP15 kernel-energy profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

import energy_common


SCHEMA = "s42-kernel-energy-profile-v1"
GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
PHONE_SERIAL = "3C15AU002CL00000"
K = 3840
N_FF = 15360
Q4_BLOCK = 32
Q4_BLOCK_BYTES = 18


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    energy_common.require(type(value) is dict, f"object: {path}")
    energy_common.require(value.get("status") == "PASS", f"status: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def evidence(root: Path, paths: Iterable[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path.relative_to(root)), "sha256": sha256(path)}
        for path in sorted(paths)
    ]


def summarize(values: Iterable[float]) -> dict[str, int]:
    rows = [float(value) for value in values]
    energy_common.require(len(rows) >= 3, "three physical repetitions")
    energy_common.require(all(value >= 0.0 for value in rows), "metric bounds")
    held_out_errors = []
    for index, actual in enumerate(rows):
        training = rows[:index] + rows[index + 1 :]
        prediction = statistics.median(training)
        denominator = max(actual, 1e-12)
        held_out_errors.append(abs(prediction - actual) / denominator)
    return {
        "max": round(max(rows)),
        "median": round(statistics.median(rows)),
        "min": round(min(rows)),
        "repeat_count": len(rows),
        "repeat_loo_max_error_ppm": round(max(held_out_errors) * 1_000_000),
    }


def exact_files(root: Path, stem: str) -> list[Path]:
    rows = sorted(root.glob(f"{stem}-r[0-9]*.json"))
    rows = [path for path in rows if ".invalid" not in path.name]
    energy_common.require(len(rows) == 3, f"three files for {stem}")
    return rows


def qualified_files(root: Path, pattern: str, label: str) -> list[Path]:
    rows = sorted(root.glob(pattern))
    rows = [path for path in rows if ".invalid" not in path.name]
    energy_common.require(len(rows) == 3, f"three files for {label}")
    return rows


def parse_key_values(text: str, prefix: str) -> dict[str, str]:
    matches = [line for line in text.splitlines() if line.startswith(prefix)]
    energy_common.require(len(matches) == 1, f"one {prefix} line")
    fields: dict[str, str] = {}
    for token in matches[0].split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    energy_common.require(fields.get("status", "PASS") == "PASS", prefix)
    return fields


def phone_capture_paths(result_path: Path) -> tuple[Path, Path, Path]:
    capture = result_path.with_suffix(".capture")
    return (
        capture / "capture.json",
        capture / "phone-samples.tsv",
        capture / "workload.log",
    )


def q4_bytes(k: int, n: int) -> int:
    energy_common.require((k * n) % Q4_BLOCK == 0, "Q4 block alignment")
    return 3 * k * n // Q4_BLOCK * Q4_BLOCK_BYTES


def ffn_ops(batch: int, k: int, n: int) -> int:
    return 6 * batch * k * n


def kernel_row(
    root: Path,
    stem: str,
    backend: str,
    batch: int,
    n: int,
    resident_type: str,
) -> dict[str, Any]:
    paths = exact_files(root, stem)
    measurements = []
    evidence_paths: list[Path] = []
    for path in paths:
        desktop = backend in {"cpu", "cuda"}
        data = load_json(path)
        if desktop:
            energy_common.require(data.get("gpu_uuid") == GPU_UUID, "GPU identity")
            fields = parse_key_values(data["stdout"], "RESULT ")
            p50_us = float(fields["median_ms"]) * 1000.0
            p90_us = float(fields["p90_ms"]) * 1000.0
            powers = {
                "cpu-package": float(data["cpu_package_average_power_w"]),
                "gpu-board": float(data["gpu_board_average_power_w"]),
            }
            evidence_paths.append(path)
        else:
            energy_common.require(data.get("serial") == PHONE_SERIAL, "phone identity")
            capture_paths = phone_capture_paths(path)
            log = capture_paths[2]
            fields = parse_key_values(log.read_text(encoding="ascii"), "PHONE_ENGINE_RESULT ")
            p50_us = float(fields["p50_ms"]) * 1000.0
            p90_us = float(fields["p90_ms"]) * 1000.0
            powers = {"phone-system": float(data["whole_phone_average_power_w"])}
            evidence_paths.extend((path, *capture_paths))
        energy_uj = sum(powers.values()) * p50_us
        measurements.append((p50_us, p90_us, powers, energy_uj))

    ops = ffn_ops(batch, K, n)
    p50_values = [row[0] for row in measurements]
    median_p50_us = statistics.median(p50_values)
    weight_bytes = q4_bytes(K, n) if resident_type == "q4_0" else 3 * K * n * 2
    if backend == "htp" and n == 9664:
        weight_bytes = 62_626_816
    active_power: dict[str, dict[str, int]] = {}
    for domain_id in measurements[0][2]:
        active_power[domain_id] = summarize(
            row[2][domain_id] * 1000.0 for row in measurements
        )
    return {
        "backend": backend,
        "compute_ops": ops,
        "effective_ops_per_s": round(ops * 1_000_000.0 / median_p50_us),
        "energy_uj_per_invocation": summarize(row[3] for row in measurements),
        "evidence": evidence(root, evidence_paths),
        "kernel_family": "gemma4-dense-ffn-swiglu",
        "latency_us": summarize(p50_values),
        "p90_latency_us": summarize(row[1] for row in measurements),
        "power_mw_by_domain": active_power,
        "profile_id": stem,
        "resident_bytes": weight_bytes,
        "resident_type": resident_type,
        "shape": {"k": K, "m": batch, "n": n},
        "status": "measured_shape_bucket",
    }


def idle_domain(
    root: Path, stem: str, field: str, domain_id: str
) -> dict[str, Any]:
    paths = exact_files(root, stem)
    values = [float(load_json(path)[field]) * 1000.0 for path in paths]
    return {
        "domain_id": domain_id,
        "evidence": evidence(root, paths),
        "idle_power_mw": summarize(values),
        "status": "measured",
    }


def pcie_row(root: Path, stem: str) -> dict[str, Any]:
    paths = exact_files(root, stem)
    rows = []
    for path in paths:
        data = load_json(path)
        fields = parse_key_values(data["stdout"], "PCIE_RESULT ")
        rows.append((
            fields,
            float(fields["p50_ms"]) * 1000.0,
            float(fields["p90_ms"]) * 1000.0,
            float(data["cpu_package_average_power_w"]),
            float(data["gpu_board_average_power_w"]),
        ))
    mode = rows[0][0]["mode"]
    payload_bytes = int(rows[0][0]["bytes"])
    if mode == "duplex":
        payload_bytes *= 2
    return {
        "direction": mode,
        "energy_uj_per_transaction": summarize(
            (row[3] + row[4]) * row[1] for row in rows
        ),
        "evidence": evidence(root, paths),
        "latency_us": summarize(row[1] for row in rows),
        "p90_latency_us": summarize(row[2] for row in rows),
        "payload_bytes": payload_bytes,
        "power_mw_by_domain": {
            "cpu-package": summarize(row[3] * 1000.0 for row in rows),
            "gpu-board": summarize(row[4] * 1000.0 for row in rows),
        },
        "profile_id": stem,
        "status": "measured_shape_bucket",
        "transport": "pcie-pinned",
    }


def usb_row(root: Path, stem: str) -> dict[str, Any]:
    desktop_paths = qualified_files(
        root, f"{stem}-r[0-9]*.desktop.json", f"{stem} desktop"
    )
    phone_paths = qualified_files(
        root, f"{stem}-r[0-9]*.phone.json", f"{stem} phone"
    )
    suffix = stem.removeprefix("usb-dmabuf-")
    direction = next(
        name for name in ("h2p", "p2h", "duplex") if suffix.startswith(name)
    )
    rows = []
    evidence_paths: list[Path] = []
    for desktop_path, phone_path in zip(desktop_paths, phone_paths):
        desktop = load_json(desktop_path)
        phone = load_json(phone_path)
        suffix = desktop_path.name.removesuffix(".desktop.json")
        transport_path = root / f"{suffix}.transport" / f"{suffix}.json"
        transport = json.loads(transport_path.read_text(encoding="ascii"))
        energy_common.require(transport.get("schema") == "s41_ffs_dmabuf_transport_v1", "USB schema")
        iterations = int(transport["iterations"])
        service_us = float(transport["campaign_seconds"]) * 1e6 / iterations
        rpc_us = float(transport["response_ready_median_ms"]) * 1000.0
        payload_bytes = int(transport["request_bytes"]) + int(transport["response_bytes"])
        rows.append((
            service_us,
            rpc_us,
            float(transport["aggregate_payload_MBps"]) * 1e6,
            float(desktop["cpu_package_average_power_w"]),
            float(desktop["gpu_board_average_power_w"]),
            float(phone["whole_phone_average_power_w"]),
            payload_bytes,
        ))
        capture_paths = phone_capture_paths(
            root / f"{suffix}.json"
        )
        phone_marker = root / f"{suffix}.transport" / f"{suffix}.phone.log"
        evidence_paths.extend((
            desktop_path,
            phone_path,
            transport_path,
            phone_marker,
            *capture_paths,
        ))
    return {
        "direction": direction,
        "energy_uj_per_transaction": summarize(
            (row[3] + row[4] + row[5]) * row[0] for row in rows
        ),
        "evidence": evidence(root, evidence_paths),
        "payload_bytes": rows[0][6],
        "pipeline_service_us": summarize(row[0] for row in rows),
        "power_mw_by_domain": {
            "cpu-package": summarize(row[3] * 1000.0 for row in rows),
            "gpu-board": summarize(row[4] * 1000.0 for row in rows),
            "phone-system": summarize(row[5] * 1000.0 for row in rows),
        },
        "profile_id": stem,
        "rpc_latency_us": summarize(row[1] for row in rows),
        "status": "measured_shape_bucket",
        "throughput_bytes_per_s": summarize(row[2] for row in rows),
        "transport": "usb-functionfs-dmabuf",
    }


def one_time_phone_row(
    root: Path,
    stem: str,
    kind: str,
    result_prefix: str = "PHONE_REPACK_RESULT ",
) -> dict[str, Any]:
    paths = exact_files(root, stem)
    rows = []
    evidence_paths: list[Path] = []
    for path in paths:
        data = load_json(path)
        capture_paths = phone_capture_paths(path)
        log = capture_paths[2]
        fields = parse_key_values(log.read_text(encoding="ascii"), result_prefix)
        total_us = float(fields["total_p50_ms"]) * 1000.0
        rows.append((
            total_us,
            float(fields["total_p90_ms"]) * 1000.0,
            float(data["whole_phone_average_power_w"]),
            int(fields["bytes"]),
        ))
        evidence_paths.extend((path, *capture_paths))
    return {
        "bytes": rows[0][3],
        "cost_id": stem,
        "energy_uj": summarize(row[0] * row[2] for row in rows),
        "evidence": evidence(root, evidence_paths),
        "kind": kind,
        "latency_us": summarize(row[0] for row in rows),
        "p90_latency_us": summarize(row[1] for row in rows),
        "power_mw_by_domain": {
            "phone-system": summarize(row[2] * 1000.0 for row in rows),
        },
        "status": "measured_one_time_cost",
    }


def model_load_row(root: Path, stem: str) -> dict[str, Any]:
    paths = exact_files(root, stem)
    rows = []
    model_name = ""
    model_bytes = 0
    layers = ""
    for path in paths:
        data = load_json(path)
        fields = parse_key_values(data["stdout"], "MODEL_LOAD_RESULT ")
        model_name = fields["model"]
        model_bytes = int(fields["bytes"])
        layers = fields["layers"]
        rows.append((
            float(fields["elapsed_ms"]) * 1000.0,
            float(data["cpu_package_energy_j"]) * 1e6,
            float(data["gpu_board_energy_j"]) * 1e6,
        ))
    return {
        "bytes": model_bytes,
        "cost_id": stem,
        "energy_uj_by_domain": {
            "cpu-package": summarize(row[1] for row in rows),
            "gpu-board": summarize(row[2] for row in rows),
        },
        "evidence": evidence(root, paths),
        "kind": "model-load-cpu-ram-to-cuda-vram",
        "latency_us": summarize(row[0] for row in rows),
        "model": model_name,
        "offloaded_layers": layers,
        "status": "measured_epoch_cost",
    }


def model_switch_row(root: Path, stem: str) -> dict[str, Any]:
    paths = exact_files(root, stem)
    rows = []
    source = ""
    target = ""
    target_bytes = 0
    source_layers = ""
    target_layers = ""
    for path in paths:
        data = load_json(path)
        fields = parse_key_values(data["stdout"], "MODEL_SWITCH_RESULT ")
        source = fields["source"]
        target = fields["target"]
        target_bytes = int(fields["target_bytes"])
        source_layers = fields["source_layers"]
        target_layers = fields["target_layers"]
        rows.append((
            float(fields["elapsed_ms"]) * 1000.0,
            float(data["cpu_package_energy_j"]) * 1e6,
            float(data["gpu_board_energy_j"]) * 1e6,
        ))
    return {
        "cost_id": stem,
        "energy_uj_by_domain": {
            "cpu-package": summarize(row[1] for row in rows),
            "gpu-board": summarize(row[2] for row in rows),
        },
        "evidence": evidence(root, paths),
        "kind": "model-switch-cuda-residency",
        "latency_us": summarize(row[0] for row in rows),
        "source": source,
        "source_layers": source_layers,
        "status": "measured_epoch_cost",
        "target": target,
        "target_bytes": target_bytes,
        "target_layers": target_layers,
    }


def validate_profile(profile: dict[str, Any]) -> None:
    energy_common.require(profile.get("schema") == SCHEMA, "profile schema")
    energy_common.require(len(profile["kernel_rows"]) == 16, "kernel row count")
    energy_common.require(len(profile["pcie_rows"]) == 7, "PCIe row count")
    energy_common.require(len(profile["usb_rows"]) == 6, "USB row count")
    energy_common.require(len(profile["one_time_cost_rows"]) == 9, "one-time row count")
    evidence_ids = []
    for section in ("energy_domains", "kernel_rows", "pcie_rows", "usb_rows", "one_time_cost_rows"):
        for row in profile[section]:
            evidence_ids.extend(item["sha256"] for item in row["evidence"])
    energy_common.require(all(item.startswith("sha256:") for item in evidence_ids), "evidence hashes")


def attach_dynamic_energy(
    profile: dict[str, Any], idle_power_mw: dict[str, int]
) -> None:
    for section, latency_field in (
        ("kernel_rows", "latency_us"),
        ("pcie_rows", "latency_us"),
        ("usb_rows", "pipeline_service_us"),
    ):
        for row in profile[section]:
            latency_us = row[latency_field]["median"]
            dynamic_mw = sum(
                max(0, values["median"] - idle_power_mw[domain_id])
                for domain_id, values in row["power_mw_by_domain"].items()
            )
            row["dynamic_energy_uj"] = round(dynamic_mw * latency_us / 1000)
            dynamic_mw_ucb = sum(
                max(0, values["max"] - idle_power_mw[domain_id])
                for domain_id, values in row["power_mw_by_domain"].items()
            )
            row["dynamic_energy_uj_ucb"] = round(
                dynamic_mw_ucb * row[latency_field]["max"] / 1000
            )
    for row in profile["one_time_cost_rows"]:
        latency_us = row["latency_us"]["median"]
        if "power_mw_by_domain" in row:
            dynamic_mw = sum(
                max(0, values["median"] - idle_power_mw[domain_id])
                for domain_id, values in row["power_mw_by_domain"].items()
            )
            row["dynamic_energy_uj"] = round(dynamic_mw * latency_us / 1000)
            dynamic_mw_ucb = sum(
                max(0, values["max"] - idle_power_mw[domain_id])
                for domain_id, values in row["power_mw_by_domain"].items()
            )
            row["dynamic_energy_uj_ucb"] = round(
                dynamic_mw_ucb * row["latency_us"]["max"] / 1000
            )
        else:
            row["dynamic_energy_uj_by_domain"] = {
                domain_id: max(
                    0,
                    values["median"]
                    - idle_power_mw[domain_id] * latency_us // 1000,
                )
                for domain_id, values in row["energy_uj_by_domain"].items()
            }


def fit_nonnegative_line(points: list[tuple[int, int]]) -> tuple[int, float]:
    energy_common.require(len(points) >= 2, "line-fit point count")
    x_mean = statistics.mean(point[0] for point in points)
    y_mean = statistics.mean(point[1] for point in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    energy_common.require(denominator > 0.0, "line-fit x range")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    intercept = y_mean - slope * x_mean
    if slope < 0.0:
        slope = 0.0
        intercept = max(y for _, y in points)
    elif intercept < 0.0:
        intercept = 0.0
        slope = sum(x * y for x, y in points) / sum(x * x for x, _ in points)
    return round(intercept), slope


def link_model(
    model_id: str,
    source_device: str,
    target_device: str,
    rows: list[dict[str, Any]],
    latency_field: str,
) -> dict[str, Any]:
    latency_points = [
        (row["payload_bytes"], row[latency_field]["max"]) for row in rows
    ]
    energy_points = [
        (row["payload_bytes"], row["dynamic_energy_uj_ucb"]) for row in rows
    ]
    fixed_latency_us, latency_slope = fit_nonnegative_line(latency_points)
    fixed_dynamic_uj, energy_slope = fit_nonnegative_line(energy_points)
    predicted = [
        fixed_latency_us + latency_slope * payload for payload, _ in latency_points
    ]
    max_error_ppm = max(
        round(abs(prediction - actual) * 1_000_000 / max(actual, 1))
        for prediction, (_, actual) in zip(predicted, latency_points)
    )
    return {
        "bandwidth_bytes_per_s": round(1_000_000 / max(latency_slope, 1e-12)),
        "dynamic_pj_per_byte": round(energy_slope * 1_000_000),
        "evidence_ids": sorted({
            item["sha256"] for row in rows for item in row["evidence"]
        }),
        "fit_max_error_ppm": max_error_ppm,
        "fixed_dynamic_uj": fixed_dynamic_uj,
        "fixed_latency_us": fixed_latency_us,
        "max_payload_bytes": max(point[0] for point in latency_points),
        "min_payload_bytes": min(point[0] for point in latency_points),
        "model_id": model_id,
        "source_device": source_device,
        "status": "derived_from_measured_shape_buckets",
        "target_device": target_device,
    }


def derive_link_models(profile: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for direction, source, target in (
        ("h2d", "cpu", "cuda"),
        ("d2h", "cuda", "cpu"),
    ):
        rows = [
            row for row in profile["pcie_rows"]
            if row["direction"] == direction
        ]
        result.append(link_model(
            f"pcie-{direction}-fit", source, target, rows, "latency_us"
        ))
    for direction, source, target in (
        ("h2p", "cpu", "phone-memory"),
        ("p2h", "phone-memory", "cpu"),
    ):
        rows = [
            row for row in profile["usb_rows"]
            if row["direction"] == direction
        ]
        result.append(link_model(
            f"usb-{direction}-fit",
            source,
            target,
            rows,
            "pipeline_service_us",
        ))
    return result


def materialize(root: Path) -> dict[str, Any]:
    energy_common.require(root.is_absolute() and root.is_dir(), "result root")
    domains = [
        idle_domain(root, "desktop-idle", "cpu_package_average_power_w", "cpu-package"),
        idle_domain(root, "desktop-idle", "gpu_board_average_power_w", "gpu-board"),
        idle_domain(root, "phone-idle", "whole_phone_average_power_w", "phone-system"),
    ]
    kernels = []
    for backend in ("cpu", "cuda"):
        for batch in (1, 8, 32, 128):
            stem = f"{backend}-gemma-q4-ffn-m{batch}"
            kernels.append(kernel_row(root, stem, backend, batch, N_FF, "q4_0"))
    for batch in (1, 8, 32, 128):
        stem = f"htp-gemma-q4-ffn-m{batch}"
        kernels.append(kernel_row(root, stem, "htp", batch, 9664, "q4_0"))
    kernels.append(kernel_row(root, "adreno-gemma-native-ffn512-m1", "adreno", 1, 512, "q4_0"))
    for batch in (16, 32, 128):
        stem = f"adreno-gemma-f16-xmem-ffn512-m{batch}"
        kernels.append(kernel_row(root, stem, "adreno", batch, 512, "f16"))

    pcie = []
    for mode in ("h2d", "d2h"):
        for size in ("8k", "1m", "4m"):
            pcie.append(pcie_row(root, f"pcie-{mode}-{size}"))
    pcie.append(pcie_row(root, "pcie-duplex-1m"))

    usb = [
        usb_row(root, stem)
        for stem in (
            "usb-dmabuf-duplex8k-sync",
            "usb-dmabuf-h2p1m",
            "usb-dmabuf-p2h1m",
            "usb-dmabuf-duplex1m",
            "usb-dmabuf-h2p4m",
            "usb-dmabuf-p2h4m",
        )
    ]
    one_time = [
        one_time_phone_row(root, "phone-htp-q4-repack-9664", "htp-q4-pack"),
        one_time_phone_row(root, "phone-adreno-q4-upload-512", "adreno-q4-upload"),
        one_time_phone_row(root, "phone-adreno-q4-to-f16-reconstruct-512", "adreno-q4-to-f16"),
        one_time_phone_row(root, "phone-adreno-f16-upload-512", "adreno-f16-upload"),
        one_time_phone_row(
            root,
            "phone-adreno-f16-xmem-prepare-512",
            "adreno-f16-xmem-full-prepare",
            "PHONE_PREPARE_RESULT ",
        ),
        model_load_row(root, "model-load-qwen3-14b-q4km"),
        model_load_row(root, "model-load-gemma4-12b-q8"),
        model_switch_row(root, "model-switch-gemma-to-qwen"),
        model_switch_row(root, "model-switch-qwen-to-gemma"),
    ]
    profile = {
        "devices": {
            "desktop_cpu": "12th Gen Intel(R) Core(TM) i9-12900K",
            "desktop_gpu": GPU_UUID,
            "phone": PHONE_SERIAL,
        },
        "energy_boundary": {
            "domains": ["cpu-package", "gpu-board", "phone-system"],
            "id": "4060ti-op15-connected-fleet-v1",
            "note": "Kernel rows measure the domains physically visible in each isolated acquisition. Composed routes require the held-out fleet validation stage.",
        },
        "energy_domains": domains,
        "kernel_rows": kernels,
        "one_time_cost_rows": one_time,
        "pcie_rows": pcie,
        "profile_id": "4060ti-op15-kernel-energy-v1",
        "qualification": {
            "composition_status": "pending_held_out_full_model_validation",
            "enforcement": "fail_closed",
            "repeat_gate": "three repetitions with leave-one-repeat-out error reported per metric",
            "shape_scope": "exact measured buckets only",
        },
        "schema": SCHEMA,
        "usb_rows": usb,
    }
    idle_power_mw = {
        row["domain_id"]: row["idle_power_mw"]["median"] for row in domains
    }
    attach_dynamic_energy(profile, idle_power_mw)
    profile["derived_link_models"] = derive_link_models(profile)
    validate_profile(profile)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    energy_common.require(args.output.is_absolute(), "absolute output")
    energy_common.require(not args.output.exists(), "new output")
    profile = materialize(args.result_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(energy_common.canonical(profile))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except energy_common.EnergyError as error:
        print(f"S42_PROFILE_ERROR: {error}")
        raise SystemExit(2)
