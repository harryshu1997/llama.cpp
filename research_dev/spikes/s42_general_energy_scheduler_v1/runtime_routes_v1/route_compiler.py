#!/usr/bin/env python3
"""Compile measured cohort certificates into epoch-bound runtime routes."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


PROFILE_SCHEMA = "s42-kernel-energy-profile-v1"
GRAPH_SCHEMA = "s42-llama-placement-graph-v1"
CERTIFICATE_SCHEMA = "s42-route-certificates-v1"
EPOCH_SCHEMA = "s42-runtime-epoch-v1"
OUTPUT_SCHEMA = "s42-epoch-route-bundle-v1"
QUALITY_CLASSES = {"approximate", "bounded_numeric", "exact"}


class RouteCompileError(ValueError):
    pass


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def object_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _object(name: str, value: object) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise RouteCompileError(f"{name} must be an object")
    return value


def _list(name: str, value: object, *, nonempty: bool = False) -> list[Any]:
    if type(value) is not list or (nonempty and not value):
        suffix = " a non-empty list" if nonempty else " a list"
        raise RouteCompileError(f"{name} must be{suffix}")
    return value


def _string(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise RouteCompileError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RouteCompileError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RouteCompileError(f"{name} must be an integer >= {minimum}")
    return value


def _number(name: str, value: object, minimum: float = 0.0) -> float:
    if type(value) not in (int, float):
        raise RouteCompileError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise RouteCompileError(f"{name} must be finite and >= {minimum}")
    return result


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise RouteCompileError(f"{name} must be bool")
    return value


def _sha(name: str, value: object) -> str:
    result = _string(name, value)
    digest = result.removeprefix("sha256:")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise RouteCompileError(f"{name} must be a lowercase SHA-256")
    return "sha256:" + digest


def _schema(name: str, value: Mapping[str, Any], expected: str) -> None:
    if value.get("schema") != expected:
        raise RouteCompileError(f"{name} schema mismatch")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise RouteCompileError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def load_json(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RouteCompileError(f"cannot load {path}: {exc}") from exc
    return _object(str(path), value), file_sha256(raw)


def _validate_epoch(epoch: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    _schema("runtime epoch", epoch, EPOCH_SCHEMA)
    _string("epoch label", epoch.get("epoch_label"))
    bindings = _object("epoch bindings", epoch.get("bindings"))
    for key in (
        "artifacts",
        "concurrency",
        "hardware",
        "models",
        "policy",
        "residency",
        "runtimes",
        "transport",
        "workload",
    ):
        _object(f"epoch bindings.{key}", bindings.get(key))

    hardware = _object("epoch hardware", bindings["hardware"])
    for key in (
        "desktop_cpu",
        "desktop_gpu_name",
        "desktop_gpu_uuid",
        "phone_model",
        "phone_serial",
    ):
        _string(f"epoch hardware.{key}", hardware.get(key))

    models = _object("epoch models", bindings["models"])
    for role in ("cold", "hot"):
        model = _object(f"epoch model {role}", models.get(role))
        _string(f"epoch model {role} architecture", model.get("architecture"))
        _string(f"epoch model {role} quantization", model.get("quantization"))
        _sha(f"epoch model {role} file_sha256", model.get("file_sha256"))
        for key in (
            "file_bytes",
            "n_embd",
            "n_layer",
            "n_vocab",
            "tensor_bytes",
        ):
            _integer(f"epoch model {role} {key}", model.get(key), 1)

    artifacts = _object("epoch artifacts", bindings["artifacts"])
    if not artifacts:
        raise RouteCompileError("epoch artifacts cannot be empty")
    for name, raw in artifacts.items():
        _string("epoch artifact name", name)
        artifact = _object(f"epoch artifact {name}", raw)
        _sha(f"epoch artifact {name} sha256", artifact.get("sha256"))

    runtimes = _object("epoch runtimes", bindings["runtimes"])
    for role in ("cold", "hot"):
        runtime = _object(f"epoch runtime {role}", runtimes.get(role))
        manifest = _list(
            f"epoch runtime {role} manifest",
            runtime.get("manifest"),
            nonempty=True,
        )
        seen_names: set[str] = set()
        for raw in manifest:
            row = _object(f"epoch runtime {role} artifact", raw)
            name = _string(f"epoch runtime {role} artifact name", row.get("name"))
            if name in seen_names:
                raise RouteCompileError(f"duplicate {role} runtime artifact name")
            seen_names.add(name)
            _sha(f"epoch runtime {role} artifact SHA-256", row.get("sha256"))
            _integer(
                f"epoch runtime {role} artifact size_bytes",
                row.get("size_bytes"),
                1,
            )

    workload = _object("epoch workload", bindings["workload"])
    _sha("epoch workload sha256", workload.get("sha256"))
    for key in ("cold_requests", "hot_requests", "input_tokens", "output_tokens", "requests"):
        _integer(f"epoch workload {key}", workload.get(key), 1)
    if workload["cold_requests"] + workload["hot_requests"] != workload["requests"]:
        raise RouteCompileError("epoch workload request counts do not reconcile")

    policy = _object("epoch policy", bindings["policy"])
    for key in ("id", "io", "table", "weight_layout"):
        _string(f"epoch policy {key}", policy.get(key))
    _integer("epoch policy column_quantum", policy.get("column_quantum"), 1)
    _integer("epoch policy max_columns", policy.get("max_columns"), 1)

    transport = _object("epoch transport", bindings["transport"])
    for key in (
        "allocator",
        "desktop_endpoint",
        "io_type",
        "phone_endpoint",
        "protocol",
        "reset_recovery",
    ):
        _string(f"epoch transport {key}", transport.get(key))

    concurrency = _object("epoch concurrency", bindings["concurrency"])
    for key in (
        "cold_parallel_slots",
        "cold_ubatch_size",
        "hot_parallel_slots",
        "request_count",
    ):
        _integer(f"epoch concurrency {key}", concurrency.get(key), 1)
    _boolean(
        "epoch concurrency hot_model_active", concurrency.get("hot_model_active")
    )
    return bindings, object_sha256(bindings)


def _validate_graph(
    name: str,
    graph: Mapping[str, Any],
    graph_sha256: str,
    certificate_row: Mapping[str, Any],
    epoch_bindings: Mapping[str, Any],
) -> None:
    _schema(f"graph {name}", graph, GRAPH_SCHEMA)
    if graph.get("status") != "PASS":
        raise RouteCompileError(f"graph {name} did not pass its adapter")
    expected_file = _sha(
        f"certificate graph {name} file_sha256",
        certificate_row.get("file_sha256"),
    )
    if graph_sha256 != expected_file:
        raise RouteCompileError(f"graph {name} file SHA-256 mismatch")
    expected_manifest = _sha(
        f"certificate graph {name} manifest_sha256",
        certificate_row.get("manifest_sha256"),
    )
    if _sha(f"graph {name} manifest_sha256", graph.get("manifest_sha256")) != expected_manifest:
        raise RouteCompileError(f"graph {name} raw manifest SHA-256 mismatch")

    model_binding = _string(
        f"certificate graph {name} model_binding",
        certificate_row.get("model_binding"),
    )
    epoch_models = _object("epoch models", epoch_bindings.get("models"))
    epoch_model = _object(
        f"epoch model {model_binding}", epoch_models.get(model_binding)
    )
    graph_model = _object(f"graph {name} model", graph.get("model"))
    for key in ("architecture", "n_embd", "n_layer", "n_vocab", "tensor_bytes"):
        if graph_model.get(key) != epoch_model.get(key):
            raise RouteCompileError(f"graph {name} model {key} mismatch")

    qualification = _object(
        f"graph {name} qualification", graph.get("qualification")
    )
    if qualification.get("graph_observed") is not True:
        raise RouteCompileError(f"graph {name} is not physically observed")
    for key in ("energy_bound", "placement_bound", "runtime_route"):
        if qualification.get(key) is not False:
            raise RouteCompileError(
                f"graph {name} unexpectedly claims prior {key} qualification"
            )


def _validate_profile(
    profile: Mapping[str, Any],
    profile_sha256: str,
    certificate: Mapping[str, Any],
    epoch_bindings: Mapping[str, Any],
) -> None:
    _schema("kernel profile", profile, PROFILE_SCHEMA)
    expected = _object("certificate profile", certificate.get("profile"))
    if _sha("certificate profile file_sha256", expected.get("file_sha256")) != profile_sha256:
        raise RouteCompileError("kernel profile file SHA-256 mismatch")
    if profile.get("profile_id") != expected.get("profile_id"):
        raise RouteCompileError("kernel profile id mismatch")
    qualification = _object("profile qualification", profile.get("qualification"))
    if qualification.get("enforcement") != "fail_closed":
        raise RouteCompileError("kernel profile must remain fail closed")
    if qualification.get("composition_status") == "qualified":
        raise RouteCompileError("unexpected broad profile composition claim")

    devices = _object("profile devices", profile.get("devices"))
    hardware = _object("epoch hardware", epoch_bindings.get("hardware"))
    if devices.get("desktop_gpu") != hardware.get("desktop_gpu_uuid"):
        raise RouteCompileError("profile GPU and runtime epoch differ")
    if devices.get("phone") != hardware.get("phone_serial"):
        raise RouteCompileError("profile phone and runtime epoch differ")


def _profile_matches(
    graph_name: str,
    graph: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> Mapping[str, Any]:
    model = _object(f"graph {graph_name} model", graph.get("model"))
    architecture = _string(
        f"graph {graph_name} model architecture", model.get("architecture")
    )
    operators = _list(f"graph {graph_name} operators", graph.get("operators"))
    rows = _list("kernel profile rows", profile.get("kernel_rows"), nonempty=True)
    one_time = _list("kernel profile one_time_cost_rows", profile.get("one_time_cost_rows"))
    preparation_ids = {
        _string("one-time cost id", _object("one-time cost row", row).get("cost_id"))
        for row in one_time
    }

    family_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    blocked_counts: Counter[str] = Counter()

    for raw_operator in operators:
        operator = _object("placement operator", raw_operator)
        family = _string("placement operator family", operator.get("family"))
        family_counts[family] += 1
        shape = _object("placement operator shape", operator.get("shape"))
        k = _integer("operator k", shape.get("k"), 1)
        m = _integer("operator m", shape.get("m"), 1)
        n = _integer("operator n", shape.get("n"), 1)
        split_options = _list(
            "placement operator split_options", operator.get("split_options")
        )

        for raw_row in rows:
            row = _object("kernel row", raw_row)
            if row.get("status") != "measured_shape_bucket":
                continue
            if architecture != "gemma4" or family != "dense_ffn":
                continue
            if row.get("kernel_family") != "gemma4-dense-ffn-swiglu":
                continue
            row_shape = _object("kernel row shape", row.get("shape"))
            if (row_shape.get("k"), row_shape.get("m")) != (k, m):
                continue
            row_n = _integer("kernel row n", row_shape.get("n"), 1)
            backend = _string("kernel row backend", row.get("backend"))
            profile_id = _string("kernel row profile_id", row.get("profile_id"))

            match_kind: str | None = None
            if backend in ("cpu", "cuda") and row_n == n:
                match_kind = "full_operator"
            elif backend in ("htp", "adreno") and row_n < n:
                for raw_split in split_options:
                    split = _object("operator split option", raw_split)
                    minimum = _integer("split minimum", split.get("minimum"), 1)
                    maximum = _integer("split maximum", split.get("maximum"), 1)
                    quantum = _integer("split quantum", split.get("quantum"), 1)
                    if minimum <= row_n <= maximum and row_n % quantum == 0:
                        match_kind = "isolated_split_kernel"
                        break
            if match_kind is None:
                continue

            backend_counts[backend] += 1
            profile_counts[profile_id] += 1
            if match_kind == "isolated_split_kernel":
                blocked_counts["profile_composition_fail_closed"] += 1
                blocked_counts["host_complement_kernel_unprofiled"] += 1
                blocked_counts["held_out_per_shape_route_missing"] += 1
                if backend == "htp":
                    blocked_counts["campaign_cut_differs_from_kernel_row"] += 1
                    blocked_counts["usb_payload_below_measured_range"] += 1
                if backend == "adreno":
                    blocked_counts["shared_memory_transition_unprofiled"] += 1
                    if row.get("resident_type") == "f16":
                        required = {
                            "phone-adreno-q4-to-f16-reconstruct-512",
                            "phone-adreno-f16-xmem-prepare-512",
                        }
                        if not required.issubset(preparation_ids):
                            raise RouteCompileError(
                                "Adreno f16 row lacks measured preparation rows"
                            )

    local_matches = backend_counts["cpu"] + backend_counts["cuda"]
    remote_matches = backend_counts["htp"] + backend_counts["adreno"]
    return {
        "architecture": architecture,
        "family_counts": dict(sorted(family_counts.items())),
        "graph": graph_name,
        "isolated_kernel_matches_by_backend": dict(sorted(backend_counts.items())),
        "isolated_kernel_matches_by_profile": dict(sorted(profile_counts.items())),
        "local_full_operator_matches": local_matches,
        "operator_count": len(operators),
        "remote_split_kernel_matches": remote_matches,
        "remote_composition_block_reasons": dict(sorted(blocked_counts.items())),
        "certified_general_operator_routes": 0,
    }


def _route_repeats(route: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = _list("route repeats", route.get("repeats"), nonempty=True)
    if len(rows) < 3:
        raise RouteCompileError("a cohort route needs at least three repeats")
    result: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    for raw in rows:
        row = _object("route repeat", raw)
        index = _integer("route repeat index", row.get("repeat_index"), 1)
        if index in seen:
            raise RouteCompileError("duplicate route repeat index")
        seen.add(index)
        _number("route repeat duration_s", row.get("duration_s"), 0.001)
        _number("route repeat fleet_j", row.get("fleet_j"), 0.001)
        _sha("route repeat result_sha256", row.get("result_sha256"))
        _sha("route repeat phone_energy_sha256", row.get("phone_energy_sha256"))
        result.append(row)
    return sorted(result, key=lambda row: int(row["repeat_index"]))


def _compile_certified_routes(
    certificate: Mapping[str, Any], epoch_key: str
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    campaign = _object("certificate campaign", certificate.get("campaign"))
    if campaign.get("verdict") != "PASS":
        raise RouteCompileError("campaign did not pass")
    if _integer("campaign repetitions", campaign.get("repetitions"), 3) < 3:
        raise RouteCompileError("campaign needs at least three repetitions")
    gates = _object("campaign gates", campaign.get("gates"))
    if not gates or any(_boolean(f"campaign gate {key}", value) is not True for key, value in gates.items()):
        raise RouteCompileError("not every campaign gate passed")

    raw_routes = _list("certificate routes", certificate.get("routes"), nonempty=True)
    routes = [_object("certificate route", row) for row in raw_routes]
    ids = [_string("certificate route id", row.get("route_id")) for row in routes]
    if len(ids) != len(set(ids)):
        raise RouteCompileError("duplicate certificate route id")

    by_role: dict[str, Mapping[str, Any]] = {}
    repeats_by_role: dict[str, list[Mapping[str, Any]]] = {}
    for route in routes:
        role = _string("certificate route role", route.get("role"))
        if role in by_role:
            raise RouteCompileError("duplicate certificate route role")
        by_role[role] = route
        repeats_by_role[role] = _route_repeats(route)
    if set(by_role) != {"control", "treatment"}:
        raise RouteCompileError("certificate requires control and treatment routes")

    control_rows = repeats_by_role["control"]
    treatment_rows = repeats_by_role["treatment"]
    if [row["repeat_index"] for row in control_rows] != [
        row["repeat_index"] for row in treatment_rows
    ]:
        raise RouteCompileError("control and treatment repeat indexes differ")
    if not all(
        _number("treatment duration", treatment["duration_s"])
        < _number("control duration", control["duration_s"])
        and _number("treatment fleet energy", treatment["fleet_j"])
        < _number("control fleet energy", control["fleet_j"])
        for control, treatment in zip(control_rows, treatment_rows, strict=True)
    ):
        raise RouteCompileError("not every treatment repeat beats its control")

    control_duration = [_number("control duration", row["duration_s"]) for row in control_rows]
    treatment_duration = [
        _number("treatment duration", row["duration_s"]) for row in treatment_rows
    ]
    control_energy = [_number("control fleet energy", row["fleet_j"]) for row in control_rows]
    treatment_energy = [
        _number("treatment fleet energy", row["fleet_j"]) for row in treatment_rows
    ]
    duration_change = 100.0 * (_mean(treatment_duration) / _mean(control_duration) - 1.0)
    energy_change = 100.0 * (_mean(treatment_energy) / _mean(control_energy) - 1.0)
    if duration_change >= 0.0 or energy_change > -10.0:
        raise RouteCompileError("campaign improvement gates do not recompute")

    phone_work = _object("campaign phone_work", campaign.get("phone_work"))
    if _integer("phone reset recoveries", phone_work.get("reset_recoveries")) != 0:
        raise RouteCompileError("campaign contains phone reset recoveries")
    if _number(
        "mean exposed join wait fraction",
        phone_work.get("mean_exposed_join_wait_fraction"),
    ) > 0.05:
        raise RouteCompileError("campaign join wait exceeds its gate")

    compiled: list[Mapping[str, Any]] = []
    for role in ("control", "treatment"):
        route = by_role[role]
        quality = _object(f"{role} route quality", route.get("quality"))
        quality_class = _string(f"{role} quality class", quality.get("class"))
        if quality_class not in QUALITY_CLASSES:
            raise RouteCompileError("unknown route quality class")
        route_id = _string(f"{role} route id", route.get("route_id"))
        route_rows = repeats_by_role[role]
        duration = [_number("route duration", row["duration_s"]) for row in route_rows]
        energy = [_number("route fleet energy", row["fleet_j"]) for row in route_rows]
        compiled.append(
            {
                "activation": {
                    "ready": False,
                    "status": "held_for_stage5_online_gates",
                },
                "certificate_scope": "exact_workload_cohort",
                "certification_status": "measured_cohort_certified",
                "dispatch_contract": _object(
                    f"{role} dispatch contract", route.get("dispatch_contract")
                ),
                "epoch_key": epoch_key,
                "fallback_route_id": route.get("fallback_route_id"),
                "metrics": {
                    "duration_s": {
                        "max_observed_bound": max(duration),
                        "mean": _mean(duration),
                        "min": min(duration),
                    },
                    "fleet_j": {
                        "max_observed_bound": max(energy),
                        "mean": _mean(energy),
                        "min": min(energy),
                    },
                    "repetitions": len(route_rows),
                },
                "quality": dict(quality),
                "role": role,
                "route_id": route_id,
                "scope": _string(f"{role} route scope", route.get("scope")),
            }
        )

    comparison = {
        "all_pairs_reduce_duration": True,
        "all_pairs_reduce_fleet_energy": True,
        "mean_duration_change_pct": duration_change,
        "mean_fleet_energy_change_pct": energy_change,
        "mean_exposed_join_wait_fraction": phone_work[
            "mean_exposed_join_wait_fraction"
        ],
        "reset_recoveries": phone_work["reset_recoveries"],
    }
    return compiled, comparison


def _validate_dispatch_contracts(
    routes: Sequence[Mapping[str, Any]], epoch_bindings: Mapping[str, Any]
) -> None:
    by_role = {row["role"]: row for row in routes}
    control = _object(
        "compiled control dispatch contract", by_role["control"]["dispatch_contract"]
    )
    treatment = _object(
        "compiled treatment dispatch contract",
        by_role["treatment"]["dispatch_contract"],
    )
    if control.get("route_mode") != "cpu-control":
        raise RouteCompileError("control dispatch mode mismatch")
    if control.get("cold_model") != "cpu" or control.get("hot_model") != "cuda":
        raise RouteCompileError("control model placement mismatch")
    if control.get("phone_dense_ffn_split") is not False:
        raise RouteCompileError("control unexpectedly enables phone splitting")

    policy = _object("epoch policy", epoch_bindings["policy"])
    if treatment.get("route_mode") != "cpu-htp-operator-split":
        raise RouteCompileError("treatment dispatch mode mismatch")
    if treatment.get("cold_model") != "cpu-plus-op15-htp":
        raise RouteCompileError("treatment cold model placement mismatch")
    if treatment.get("hot_model") != "cuda":
        raise RouteCompileError("treatment hot model placement mismatch")
    if treatment.get("phone_dense_ffn_split") is not True:
        raise RouteCompileError("treatment does not enable phone splitting")
    if treatment.get("operator_family") != "gemma4-dense-ffn-swiglu":
        raise RouteCompileError("treatment operator family mismatch")
    if treatment.get("policy_id") != policy.get("id"):
        raise RouteCompileError("treatment policy id differs from epoch")
    if treatment.get("split_table") != policy.get("table"):
        raise RouteCompileError("treatment split table differs from epoch")

    transport = _object("epoch transport", epoch_bindings["transport"])
    expected_transport = f"{transport.get('phone_endpoint')}-{transport.get('io_type')}"
    if treatment.get("transport") != expected_transport:
        raise RouteCompileError("treatment transport differs from epoch")


def compile_bundle(
    *,
    profile: Mapping[str, Any],
    profile_sha256: str,
    certificate: Mapping[str, Any],
    certificate_sha256: str,
    epoch: Mapping[str, Any],
    epoch_sha256: str,
    graphs: Mapping[str, tuple[Mapping[str, Any], str]],
) -> Mapping[str, Any]:
    _schema("certificate set", certificate, CERTIFICATE_SCHEMA)
    epoch_bindings, epoch_key = _validate_epoch(epoch)
    if _sha(
        "certificate required_epoch_binding_sha256",
        certificate.get("required_epoch_binding_sha256"),
    ) != epoch_key:
        raise RouteCompileError("runtime epoch binding SHA-256 mismatch")
    _validate_profile(profile, profile_sha256, certificate, epoch_bindings)

    certificate_graphs = _object("certificate graphs", certificate.get("graphs"))
    if set(graphs) != set(certificate_graphs):
        raise RouteCompileError("graph input set differs from certificate")
    graph_bindings: list[Mapping[str, Any]] = []
    match_summaries: list[Mapping[str, Any]] = []
    for name in sorted(graphs):
        graph, graph_sha256 = graphs[name]
        certificate_row = _object(
            f"certificate graph {name}", certificate_graphs[name]
        )
        _validate_graph(
            name,
            graph,
            graph_sha256,
            certificate_row,
            epoch_bindings,
        )
        graph_bindings.append(
            {
                "file_sha256": graph_sha256,
                "graph": name,
                "manifest_sha256": graph["manifest_sha256"],
                "model_binding": certificate_row["model_binding"],
            }
        )
        match_summaries.append(_profile_matches(name, graph, profile))

    compiled_routes, comparison = _compile_certified_routes(certificate, epoch_key)
    _validate_dispatch_contracts(compiled_routes, epoch_bindings)
    control_id = next(
        row["route_id"] for row in compiled_routes if row["role"] == "control"
    )
    for row in compiled_routes:
        fallback = row.get("fallback_route_id")
        if row["role"] == "control" and fallback is not None:
            raise RouteCompileError("control route cannot have a fallback")
        if row["role"] == "treatment" and fallback != control_id:
            raise RouteCompileError("treatment fallback must be the control route")

    campaign = _object("certificate campaign", certificate.get("campaign"))
    workload = _object("epoch workload", epoch_bindings.get("workload"))
    if _sha("epoch workload sha256", workload.get("sha256")) != _sha(
        "campaign trace sha256", campaign.get("trace_sha256")
    ):
        raise RouteCompileError("campaign trace and epoch workload differ")
    return {
        "binding_receipts": {
            "certificate_file_sha256": certificate_sha256,
            "epoch_file_sha256": epoch_sha256,
            "graphs": graph_bindings,
            "kernel_profile_file_sha256": profile_sha256,
        },
        "certificate_set_id": _string(
            "certificate set id", certificate.get("certificate_set_id")
        ),
        "cohort_comparison": comparison,
        "compiled_routes": compiled_routes,
        "epoch_key": epoch_key,
        "general_operator_enforcement": {
            "certified_route_count": 0,
            "status": "blocked_pending_composed_route_validation",
        },
        "operator_profile_matches": match_summaries,
        "physical_evidence": {
            "aggregate_file_sha256": _sha(
                "campaign aggregate file SHA-256",
                campaign.get("aggregate_file_sha256"),
            ),
            "aggregate_record_sha256": _sha(
                "campaign aggregate record SHA-256",
                campaign.get("aggregate_record_sha256"),
            ),
            "trace_sha256": _sha(
                "campaign trace SHA-256", campaign.get("trace_sha256")
            ),
        },
        "runtime_activation_ready": False,
        "schema": OUTPUT_SCHEMA,
        "status": "PASS",
    }


def _graph_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("graph must be NAME=PATH")
    try:
        name.encode("ascii")
    except UnicodeEncodeError as exc:
        raise argparse.ArgumentTypeError("graph name must be ASCII") from exc
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--certificates", type=Path, required=True)
    parser.add_argument("--epoch", type=Path, required=True)
    parser.add_argument(
        "--graph",
        action="append",
        type=_graph_argument,
        required=True,
        help="certificate graph binding as NAME=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise RouteCompileError(f"refusing to overwrite {args.output}")
    graph_paths: dict[str, Path] = {}
    for name, path in args.graph:
        if name in graph_paths:
            raise RouteCompileError(f"duplicate graph name {name}")
        graph_paths[name] = path

    profile, profile_sha256 = load_json(args.profile)
    certificate, certificate_sha256 = load_json(args.certificates)
    epoch, epoch_sha256 = load_json(args.epoch)
    graphs = {name: load_json(path) for name, path in graph_paths.items()}
    output = compile_bundle(
        profile=profile,
        profile_sha256=profile_sha256,
        certificate=certificate,
        certificate_sha256=certificate_sha256,
        epoch=epoch,
        epoch_sha256=epoch_sha256,
        graphs=graphs,
    )
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
