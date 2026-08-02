#!/usr/bin/env python3

from __future__ import annotations

import copy
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parents[1]
S39 = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_contract_v24 as builder
import cp0_r1_evidence_v24 as evidence
import v24_common as common


ZERO = "0" * 64
BOOT_IDS = {
    "cuda": "11111111-1111-1111-1111-111111111111",
    "op12": "22222222-2222-2222-2222-222222222222",
    "op15": "33333333-3333-3333-3333-333333333333",
}


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def stat_row(size: int, seed: int) -> dict:
    return {
        "ctime_ns": 10_000 + seed,
        "device_id": seed + 1,
        "inode": seed + 100,
        "mode": 0o100755,
        "mtime_ns": 20_000 + seed,
        "size": size,
    }


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.contract = builder.build_contract()
        self.contract_path = root / "contract.json"
        self.candidate_path = root / "candidate.json"
        self.contract_path.write_bytes(common.canonical_bytes(self.contract))
        self.candidate_path.write_bytes(builder.CANDIDATE.read_bytes())
        self.candidate = json.loads(self.candidate_path.read_text())
        self.contract_raw = self.contract_path.read_bytes()
        self.candidate_raw = self.candidate_path.read_bytes()
        self.tokenizer_plan = self._tokenizer_plan()
        self.tokenizer_plan_path = self.write(
            "tokenizer-plan.json",
            self.tokenizer_plan,
        )
        self.tokenizer_plan_raw = self.tokenizer_plan_path.read_bytes()
        self.plan = self._plan()
        self.plan_path = self.write("plan.json", self.plan)
        self.plan_raw = self.plan_path.read_bytes()
        self.plan_derived = evidence.validate_runtime_plan(
            self.plan,
            self.contract,
            self.contract_raw,
            self.candidate_raw,
        )
        self.history = self._history()
        self.history_path = self.write("token-history.json", self.history)
        self.history_raw = self.history_path.read_bytes()
        self.artifact_root = self._artifact_root()
        self.artifact_root_path = self.write("artifact-root.json", self.artifact_root)
        self.artifact_root_raw = self.artifact_root_path.read_bytes()
        self.preparation = self._preparation()
        self.preparation_path = self.write("preparation.json", self.preparation)
        self.preparation_raw = self.preparation_path.read_bytes()
        self.phase_lock = self._phase_lock()
        self.phase_lock_path = self.write("phase-lock.json", self.phase_lock)
        self.phase_lock_raw = self.phase_lock_path.read_bytes()
        self.fresh = self._fresh()
        self.fresh_path = self.write("fresh.json", self.fresh)
        self.fresh_raw = self.fresh_path.read_bytes()
        self.runtime = self._runtime()
        self.runtime_path = self.write("runtime.json", self.runtime)
        self.runtime_raw = self.runtime_path.read_bytes()
        self.acquisition = self._acquisition()
        self.acquisition_path = self.write("acquisition.json", self.acquisition)

    def write(self, name: str, value: dict) -> Path:
        path = self.root / name
        path.write_bytes(common.canonical_bytes(value))
        return path

    def rewrite(self, name: str, value: dict) -> None:
        path = getattr(self, f"{name}_path")
        path.write_bytes(common.canonical_bytes(value))

    def kwargs(self) -> dict:
        return {
            "contract_path": self.contract_path,
            "candidate_path": self.candidate_path,
            "runtime_plan_path": self.plan_path,
            "tokenizer_plan_path": self.tokenizer_plan_path,
            "token_history_path": self.history_path,
            "artifact_root_path": self.artifact_root_path,
            "preparation_path": self.preparation_path,
            "phase_lock_path": self.phase_lock_path,
            "fresh_path": self.fresh_path,
            "runtime_identity_path": self.runtime_path,
            "acquisition_path": self.acquisition_path,
        }

    def _tokenizer_plan(self) -> dict:
        executable_path = "/opt/s39/v24/cuda-monolithic/llama-layersplit"
        model = next(row for row in self.candidate["models"] if row["slot"] == "A")
        model_path = self.contract["model_geometry"][evidence.MODEL_ID][
            "cuda_model_path"
        ]
        return {
            "command_template": [
                executable_path,
                "-m",
                model_path,
                "--ids",
                "-f",
                "{PROMPT_FILE}",
                "--log-disable",
            ],
            "component_id": "cuda-mono",
            "cwd": "/opt/s39/v24/cuda-monolithic",
            "environment": {
                "LC_ALL": "C",
                "LD_LIBRARY_PATH": "/opt/s39/v24/cuda-monolithic",
            },
            "executable": {
                "bytes": 1002,
                "path": executable_path,
                "sha256": digest_text("cuda-mono"),
            },
            "model": {
                "bytes": model["artifact"]["bytes"],
                "model_id": evidence.MODEL_ID,
                "path": model_path,
                "sha256": model["artifact"]["sha256"],
                "vocab_size": 151936,
            },
            "protocol": {
                "add_bos": "MODEL_DEFAULT",
                "escape": True,
                "output_format": "BRACKETED_DECIMAL_IDS",
                "parse_special": True,
                "prompt_file_placeholder": "{PROMPT_FILE}",
            },
            "schema": "s39-cp0-r1-a-only-tokenizer-plan-v2",
            "timeout_seconds": 300,
        }

    def _plan(self) -> dict:
        roots = {
            "cuda_monolithic": "/opt/s39/v24/cuda-monolithic",
            "cuda_route": "/opt/s39/v24/cuda-route",
            "op12_stagenet": "/data/local/tmp/s39/v24/op12-stage",
            "op15_direct_relay": "/data/local/tmp/s39/v24/op15-relay",
            "op15_stagenet": "/data/local/tmp/s39/v24/op15-stage",
        }
        specs = [
            ("artifact-driver", "cuda_monolithic", "cuda", "artifact-root", "executable"),
            ("cuda-mono-capture", "cuda_monolithic", "cuda", "cuda-monolithic-v1.py", "executable"),
            ("cuda-mono", "cuda_monolithic", "cuda", "llama-layersplit", "executable"),
            ("cuda-route", "cuda_route", "cuda", "llama-layersplit", "executable"),
            ("fresh-driver", "cuda_route", "cuda", "fast-fresh", "executable"),
            ("joint-driver", "cuda_route", "cuda", "joint-phone-cuda", "executable"),
            ("op12-stage", "op12_stagenet", "op12", "llama-layersplit", "executable"),
            ("op15-relay", "op15_direct_relay", "op15", "stage-direct-relay", "executable"),
            ("op15-stage", "op15_stagenet", "op15", "llama-layersplit", "executable"),
        ]
        components = []
        for index, (component_id, bundle_id, endpoint, name, role) in enumerate(specs):
            producer_name = {
                "cuda-mono-capture": "cuda_monolithic",
                "joint-driver": "joint_phone_cuda",
            }.get(component_id)
            producer = (
                self.contract["producer_requirements"]["source_programs"][
                    producer_name
                ]
                if producer_name is not None
                else None
            )
            components.append(
                {
                    "bundle_id": bundle_id,
                    "bytes": (
                        producer["bytes"]
                        if producer is not None
                        else 1000 + index
                    ),
                    "component_id": component_id,
                    "endpoint": endpoint,
                    "path": f"{roots[bundle_id]}/{name}",
                    "role": role,
                    "sha256": (
                        producer["sha256"]
                        if producer is not None
                        else digest_text(component_id)
                    ),
                }
            )
        components.sort(key=lambda row: row["component_id"])
        launchers = {
            "cuda_monolithic": "cuda-mono",
            "cuda_route": "cuda-route",
            "op12_stagenet": "op12-stage",
            "op15_direct_relay": "op15-relay",
            "op15_stagenet": "op15-stage",
        }
        capture_components = {
            "artifact-driver",
            "cuda-mono-capture",
            "fresh-driver",
            "joint-driver",
        }
        bundles = []
        for bundle_id, (endpoint, process_role) in sorted(
            evidence.REQUIRED_BUNDLES.items()
        ):
            required = sorted(
                row["component_id"]
                for row in components
                if row["bundle_id"] == bundle_id
                and row["component_id"] not in capture_components
            )
            bundles.append(
                {
                    "bundle_id": bundle_id,
                    "endpoint": endpoint,
                    "launcher_component_id": launchers[bundle_id],
                    "process_role": process_role,
                    "required_component_ids": required,
                }
            )
        captures = [
            {
                "component_id": "artifact-driver",
                "execution_mode": "SELF_CONTAINED_PHYSICAL_CAPTURE",
                "kind": "artifact_root",
                "nested_capture_entrypoint_component_ids": [],
            },
            {
                "component_id": "cuda-mono-capture",
                "execution_mode": "SELF_CONTAINED_PHYSICAL_CAPTURE",
                "kind": "cuda_monolithic",
                "nested_capture_entrypoint_component_ids": ["cuda-mono"],
            },
            {
                "component_id": "fresh-driver",
                "execution_mode": "SELF_CONTAINED_PHYSICAL_CAPTURE",
                "kind": "fast_fresh_readiness",
                "nested_capture_entrypoint_component_ids": [],
            },
            {
                "component_id": "joint-driver",
                "execution_mode": "SELF_CONTAINED_PHYSICAL_CAPTURE",
                "kind": "joint_phone_cuda",
                "nested_capture_entrypoint_component_ids": [
                    "cuda-route",
                    "op12-stage",
                    "op15-relay",
                    "op15-stage",
                ],
            },
        ]
        model = next(row for row in self.candidate["models"] if row["slot"] == "A")
        protocol = self.contract["token_history_protocol"]
        root_component_ids = sorted(
            {
                *(row["component_id"] for row in components),
                "model.cuda",
                "model.op12_shard",
                "model.op15_shard",
                "token_history.mmlu64",
                "tokenizer.plan",
            }
        )
        component_by_id = {
            row["component_id"]: row for row in components
        }
        mono_bundle = next(
            row for row in bundles
            if row["bundle_id"] == "cuda_monolithic"
        )
        launch_components = [
            {
                "component_id": component_id,
                "path": component_by_id[component_id]["path"],
                "sha256": component_by_id[component_id]["sha256"],
                "stat": stat_row(
                    component_by_id[component_id]["bytes"],
                    root_component_ids.index(component_id),
                ),
            }
            for component_id in mono_bundle["required_component_ids"]
        ]
        model_path = self.contract["model_geometry"][evidence.MODEL_ID][
            "cuda_model_path"
        ]
        model_artifact = {
            "path": model_path,
            "sha256": model["artifact"]["sha256"],
            "stat": stat_row(
                model["artifact"]["bytes"],
                root_component_ids.index("model.cuda"),
            ),
        }
        bundle_sha256 = common.sha256_bytes(
            common.canonical_bytes(
                {
                    "bundle_id": "cuda_monolithic",
                    "components": launch_components,
                    "endpoint": "cuda",
                    "launcher_component_id": mono_bundle[
                        "launcher_component_id"
                    ],
                    "process_role": "cuda_monolithic",
                    "schema": (
                        "s39-cp0-r1-runtime-bundle-root-identity-v2.4"
                    ),
                }
            )
        )
        port = 39124
        cuda_launch = {
            "allowed_system_roots": [
                "/mnt/storage/s21_deps/cuda-13.2.1/lib/",
                "/usr/lib/x86_64-linux-gnu/",
            ],
            "bundle_id": "cuda_monolithic",
            "bundle_root": roots["cuda_monolithic"],
            "bundle_sha256": bundle_sha256,
            "command": [
                component_by_id["cuda-mono"]["path"],
                "-m",
                model_path,
                "--mode",
                "monov3",
                "--port",
                str(port),
                "--devices",
                "CUDA0",
                "--driver-batch",
                "8",
                "--driver-context",
                "512",
                "--driver-max-prefill",
                "8",
            ],
            "cwd": "/home/zhihao/llama.cpp-s40",
            "endpoint": "cuda",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "HOME": "/home/zhihao",
                "LAYERSPLIT_MEMORY_CERT": "1",
                "LAYERSPLIT_MODEL_SHA256": model["artifact"]["sha256"],
                "LAYERSPLIT_PLACEMENT_CERT": "1",
                "LC_ALL": "C",
                "LD_LIBRARY_PATH": roots["cuda_monolithic"],
            },
            "expected_capabilities": 0x3F,
            "expected_file_type": 15,
            "expected_max_streams": 8,
            "expected_n_batch": 64,
            "expected_n_ctx_seq": 512,
            "expected_n_embd": 5120,
            "expected_n_layer": 40,
            "expected_n_ubatch": 64,
            "host": "127.0.0.1",
            "io_timeout_ms": 300000,
            "launcher_component_id": "cuda-mono",
            "model_artifact": model_artifact,
            "model_id": evidence.MODEL_ID,
            "model_sha256": model["artifact"]["sha256"],
            "port": port,
            "required_components": launch_components,
            "route_epoch": 1,
            "schema": "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
            "shutdown_timeout_ms": 30000,
            "startup_timeout_ms": 300000,
        }
        return {
            "bundle_roots": roots,
            "bundles": bundles,
            "candidate_sha256": common.sha256_bytes(self.candidate_raw),
            "capture_entrypoints": captures,
            "components": components,
            "contract_sha256": common.sha256_bytes(self.contract_raw),
            "cuda_monolithic_launch": cuda_launch,
            "model_id": evidence.MODEL_ID,
            "phase": evidence.PHASE,
            "schema": "s39-cp0-r1-runtime-bundle-plan-v2.4",
            "token_history": {
                "artifact_path": "/opt/s39/v24/token-history.json",
                "batch": protocol["batch"],
                "continuation_tokens_per_request": protocol[
                    "continuation_tokens_per_request"
                ],
                "corpus_sha256": self.contract["quality_corpus"]["sha256"],
                "decode_calls_after_prefill": protocol[
                    "decode_calls_after_prefill"
                ],
                "model_sha256": model["artifact"]["sha256"],
                "n_batch": protocol["n_batch"],
                "n_ctx_seq": protocol["n_ctx_seq"],
                "n_ubatch": protocol["n_ubatch"],
                "prefill_chunking": protocol["prefill_chunking"],
                "prefill_row_order": protocol["prefill_row_order"],
                "mechanics_item_indices": protocol["mechanics_item_indices"],
                "quality_group_count": protocol["quality_group_count"],
                "quality_group_size": protocol["quality_group_size"],
                "quality_items": protocol["quality_items"],
                "tokenizer_component_id": "cuda-mono",
                "tokenizer_plan_bytes": len(self.tokenizer_plan_raw),
                "tokenizer_plan_path": "/opt/s39/v24/tokenizer-plan.json",
                "tokenizer_plan_sha256": common.sha256_bytes(
                    self.tokenizer_plan_raw
                ),
            },
        }

    def _history(self) -> dict:
        corpus = evidence._load_corpus(self.contract)
        requests = []
        for item_index in range(
            self.contract["token_history_protocol"]["quality_items"]
        ):
            prompt = evidence._prompt(corpus[item_index], self.candidate)
            tokens = [
                1000 + item_index * 100 + offset
                for offset in range(5 + item_index % 8)
            ]
            requests.append(
                {
                    "item_index": item_index,
                    "request_id": item_index % 8 + 1,
                    "seq_id": item_index % 8,
                    "prompt_utf8_base64": base64.b64encode(
                        prompt.encode("utf-8")
                    ).decode("ascii"),
                    "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "token_ids": tokens,
                }
            )

        def make_group(group_index: int) -> dict:
            selected = list(range(group_index * 8, group_index * 8 + 8))
            all_rows = []
            for item_index in selected:
                request = requests[item_index]
                for position, token_id in enumerate(request["token_ids"]):
                    all_rows.append(
                        (
                            position,
                            item_index,
                            request["request_id"],
                            request["seq_id"],
                            token_id,
                        )
                    )
            all_rows.sort(key=lambda row: (row[0], row[1]))
            waves = [
                [row for row in all_rows if row[0] == position]
                for position in sorted({row[0] for row in all_rows})
            ]
            partitions = []
            current = []
            for wave in waves:
                if current and len(current) + len(wave) > 64:
                    partitions.append(current)
                    current = []
                current.extend(wave)
            if current:
                partitions.append(current)
            prefill = [
                {
                    "call_index": call_index,
                    "rows": [
                        {
                            "item_index": item_index,
                            "position": position,
                            "request_id": request_id,
                            "seq_id": seq_id,
                            "token_id": token_id,
                        }
                        for position, item_index, request_id, seq_id, token_id
                        in partition
                    ],
                }
                for call_index, partition in enumerate(partitions)
            ]
            return {
                "decode_calls": [
                    {
                        "call_index": len(prefill) + call_index,
                        "continuation_input_ordinal": call_index,
                        "continuation_output_ordinal": call_index + 1,
                        "rows": [
                            {
                                "item_index": item_index,
                                "position": (
                                    len(requests[item_index]["token_ids"])
                                    + call_index
                                ),
                                "request_id": requests[item_index][
                                    "request_id"
                                ],
                                "seq_id": requests[item_index]["seq_id"],
                            }
                            for item_index in selected
                        ],
                    }
                    for call_index in range(7)
                ],
                "group_index": group_index,
                "item_indices": selected,
                "prefill_partitions": prefill,
            }

        groups = [make_group(group_index) for group_index in range(8)]
        protocol = self.contract["token_history_protocol"]
        model = next(row for row in self.candidate["models"] if row["slot"] == "A")
        tokenizer = next(
            row for row in self.plan["components"]
            if row["component_id"] == "cuda-mono"
        )
        return {
            "batch": 8,
            "candidate_sha256": common.sha256_bytes(self.candidate_raw),
            "continuation_tokens_per_request": 8,
            "corpus_sha256": self.contract["quality_corpus"]["sha256"],
            "mechanics_b8": groups[0],
            "model_id": evidence.MODEL_ID,
            "model_sha256": model["artifact"]["sha256"],
            "n_batch": 64,
            "n_ctx_seq": 512,
            "n_ubatch": 64,
            "prefill_chunking": "WHOLE_POSITION_WAVES_MAX_64_ROWS",
            "prefill_row_order": "POSITION_MAJOR_THEN_ITEM_INDEX",
            "quality_groups": groups,
            "requests": requests,
            "schema": "s39-cp0-r1-token-history-v2.4",
            "tokenizer": {
                "component_id": "cuda-mono",
                "path": tokenizer["path"],
                "plan_sha256": common.sha256_bytes(
                    self.tokenizer_plan_raw
                ),
                "sha256": tokenizer["sha256"],
            },
        }

    def _artifact_root(self) -> dict:
        model = next(row for row in self.candidate["models"] if row["slot"] == "A")
        geometry = self.contract["model_geometry"][evidence.MODEL_ID]
        rows = [
            {
                "bytes": model["artifact"]["bytes"],
                "component_id": "model.cuda",
                "endpoint": "cuda",
                "kind": "model_weight",
                "path": geometry["cuda_model_path"],
                "sha256": model["artifact"]["sha256"],
            },
            {
                "bytes": geometry["known_shards"]["op12"]["bytes"],
                "component_id": "model.op12_shard",
                "endpoint": "op12",
                "kind": "model_shard",
                "path": geometry["known_shards"]["op12"]["path"],
                "sha256": geometry["known_shards"]["op12"]["sha256"],
            },
            {
                "bytes": geometry["known_shards"]["op15"]["bytes"],
                "component_id": "model.op15_shard",
                "endpoint": "op15",
                "kind": "model_shard",
                "path": geometry["known_shards"]["op15"]["path"],
                "sha256": geometry["known_shards"]["op15"]["sha256"],
            },
            {
                "bytes": len(self.history_raw),
                "component_id": "token_history.mmlu64",
                "endpoint": "cuda",
                "kind": "token_history",
                "path": self.plan["token_history"]["artifact_path"],
                "sha256": common.sha256_bytes(self.history_raw),
            },
            {
                "bytes": len(self.tokenizer_plan_raw),
                "component_id": "tokenizer.plan",
                "endpoint": "cuda",
                "kind": "tokenizer_plan",
                "path": self.plan["token_history"]["tokenizer_plan_path"],
                "sha256": common.sha256_bytes(self.tokenizer_plan_raw),
            },
        ]
        for component in self.plan["components"]:
            rows.append(
                {
                    "bytes": component["bytes"],
                    "component_id": component["component_id"],
                    "endpoint": component["endpoint"],
                    "kind": "runtime_component",
                    "path": component["path"],
                    "sha256": component["sha256"],
                }
            )
        rows.sort(key=lambda row: row["component_id"])
        for index, row in enumerate(rows):
            row["stat"] = stat_row(row["bytes"], index)
        inventories = []
        for bundle in self.plan["bundles"]:
            inventories.append(
                {
                    "bundle_id": bundle["bundle_id"],
                    "endpoint": bundle["endpoint"],
                    "paths": sorted(
                        next(
                            component["path"]
                            for component in self.plan["components"]
                            if component["component_id"] == component_id
                        )
                        for component_id in bundle["required_component_ids"]
                    ),
                    "root": self.plan["bundle_roots"][bundle["bundle_id"]],
                }
            )
        return {
            "candidate_sha256": common.sha256_bytes(self.candidate_raw),
            "completed_ns": 200,
            "components": rows,
            "contract_sha256": common.sha256_bytes(self.contract_raw),
            "inventories": inventories,
            "model_id": evidence.MODEL_ID,
            "phase_scope": "PRE_REBOOT_OUTSIDE_PHASE",
            "runtime_bundle_plan_sha256": common.sha256_bytes(self.plan_raw),
            "schema": "s39-cp0-r1-artifact-root-v2.4",
            "started_ns": 100,
        }

    def _devices(self) -> dict:
        devices = self.contract["devices"]
        return {
            "cuda": {
                "gpu_uuid": devices["cuda"]["uuid"],
                "host": devices["cuda"]["host"],
                "host_boot_id": BOOT_IDS["cuda"],
                "pci_bus_id": "0000:01:00.0",
                "system_swap_used_bytes": 0,
            },
            "op12": {
                **devices["op12"],
                "available_bytes": 1_000_000_000,
                "boot_id": BOOT_IDS["op12"],
                "system_swap_used_bytes": 123_456,
                "thermal_status": 0,
            },
            "op15": {
                **devices["op15"],
                "available_bytes": 1_000_000_000,
                "boot_id": BOOT_IDS["op15"],
                "system_swap_used_bytes": 234_567,
                "thermal_status": 0,
            },
        }

    def _preparation(self) -> dict:
        devices = self._devices()
        devices["op12"].update({
            "interface": "wlan0",
            "local_ipv4": "10.0.0.12",
        })
        devices["op15"].update({
            "interface": "wlan0",
            "local_ipv4": "10.0.0.15",
        })
        return {
            "artifact_root_sha256": common.sha256_bytes(self.artifact_root_raw),
            "before_boot_ids": {
                "op12": "12121212-1212-4212-8212-121212121212",
                "op15": "15151515-1515-4515-8515-151515151515",
            },
            "completed_ns": 300,
            "devices": devices,
            "reboot_started_ns": 220,
            "runtime_bundle_plan_sha256": common.sha256_bytes(self.plan_raw),
            "schema": "s39-cp0-r1-reboot-preparation-v2.4",
            "started_ns": 210,
        }

    def _phase_lock(self) -> dict:
        return {
            "artifact_root_sha256": common.sha256_bytes(self.artifact_root_raw),
            "candidate_sha256": common.sha256_bytes(self.candidate_raw),
            "contract_sha256": common.sha256_bytes(self.contract_raw),
            "device_boot_ids": BOOT_IDS,
            "event_ns": 400,
            "model_id": evidence.MODEL_ID,
            "phase": "A_ONLY",
            "phase_id": "cp0-r1-v24-a-only-test",
            "preparation_sha256": common.sha256_bytes(self.preparation_raw),
            "quality_corpus_sha256": self.contract["quality_corpus"]["sha256"],
            "runtime_bundle_plan_sha256": common.sha256_bytes(self.plan_raw),
            "schema": "s39-cp0-r1-phase-lock-v2.4",
        }

    def _fresh(self) -> dict:
        return {
            "artifact_root_sha256": common.sha256_bytes(self.artifact_root_raw),
            "component_stats": [
                {
                    "component_id": row["component_id"],
                    "endpoint": row["endpoint"],
                    "path": row["path"],
                    "stat": copy.deepcopy(row["stat"]),
                }
                for row in self.artifact_root["components"]
            ],
            "completed_ns": 700,
            "devices": self._devices(),
            "inventories": copy.deepcopy(self.artifact_root["inventories"]),
            "phase": "A_ONLY",
            "phase_id": self.phase_lock["phase_id"],
            "phase_lock_sha256": common.sha256_bytes(self.phase_lock_raw),
            "preparation_sha256": common.sha256_bytes(self.preparation_raw),
            "runtime_bundle_plan_sha256": common.sha256_bytes(self.plan_raw),
            "schema": "s39-cp0-r1-fast-fresh-readiness-v2.4",
            "started_ns": 500,
        }

    def _dynamic_roles(self) -> list[str]:
        v22 = json.loads(evidence.V22_CONTRACT.read_text())
        roles = set(v22["phase_protocol"]["phase_roles"]["A_ONLY"])
        return sorted(
            (roles - evidence.PRE_ACQUISITION_ROLES)
            | evidence.CAPTURE_RECEIPT_ROLES
        )

    def _runtime(self) -> dict:
        root_components = {
            row["component_id"]: row for row in self.artifact_root["components"]
        }
        processes = []
        for index, bundle in enumerate(self.plan["bundles"]):
            bundle_id = bundle["bundle_id"]
            launcher = root_components[bundle["launcher_component_id"]]
            role = evidence.PROCESS_EVIDENCE_ROLES[bundle_id]
            processes.append(
                {
                    "boot_id": BOOT_IDS[bundle["endpoint"]],
                    "bundle_id": bundle_id,
                    "bundle_sha256": evidence._bundle_digest(
                        bundle_id,
                        self.plan_derived,
                        root_components,
                    ),
                    "endpoint": bundle["endpoint"],
                    "evidence_role": role,
                    "evidence_sha256": digest_text(role),
                    "launcher_component_id": bundle["launcher_component_id"],
                    "launcher_path": launcher["path"],
                    "loaded_repo_component_ids": bundle[
                        "required_component_ids"
                    ],
                    "model_mapping": (
                        {
                            "argv": self.plan["cuda_monolithic_launch"][
                                "command"
                            ],
                            "environment": self.plan[
                                "cuda_monolithic_launch"
                            ]["env"],
                            "model_mapping_rows": [
                                {
                                    "address_range": (
                                        f"{0x60000000 + index * 0x200000:x}-"
                                        f"{0x60100000 + index * 0x200000:x}"
                                    ),
                                    "device_major": os.major(
                                        root_components["model.cuda"]["stat"][
                                            "device_id"
                                        ]
                                    ),
                                    "device_minor": os.minor(
                                        root_components["model.cuda"]["stat"][
                                            "device_id"
                                        ]
                                    ),
                                    "inode": root_components["model.cuda"][
                                        "stat"
                                    ]["inode"],
                                    "offset_bytes": expected[
                                        "offset_bytes"
                                    ],
                                    "path": root_components["model.cuda"][
                                        "path"
                                    ],
                                    "permissions": expected["permissions"],
                                }
                                for index, expected in enumerate(
                                    self.contract[
                                        "cuda_monolithic_identity"
                                    ]["maps_exact_rows"]
                                )
                            ],
                            "model_file_type": 15,
                            "model_path": root_components["model.cuda"]["path"],
                            "model_sha256": root_components["model.cuda"][
                                "sha256"
                            ],
                            "other_gguf_mapping_paths": [],
                            "post_stat": root_components["model.cuda"]["stat"],
                            "pre_stat": root_components["model.cuda"]["stat"],
                        }
                        if bundle_id == "cuda_monolithic"
                        else None
                    ),
                    "observed_ns": 1200 + index,
                    "pid": 100 + index,
                    "process_swap_bytes": 0,
                    "start_ticks": 1000 + index,
                }
            )
        return {
            "artifact_root_sha256": common.sha256_bytes(self.artifact_root_raw),
            "completed_ns": 1900,
            "fresh_readiness_sha256": common.sha256_bytes(self.fresh_raw),
            "phase": "A_ONLY",
            "phase_id": self.phase_lock["phase_id"],
            "phone_after": {
                phone: {
                    "boot_id": BOOT_IDS[phone],
                    "observed_ns": 1800,
                    "system_swap_used_bytes": self.fresh["devices"][phone][
                        "system_swap_used_bytes"
                    ],
                }
                for phone in ("op12", "op15")
            },
            "processes": processes,
            "runtime_bundle_plan_sha256": common.sha256_bytes(self.plan_raw),
            "schema": "s39-cp0-r1-runtime-identity-v2.4",
            "started_ns": 1100,
        }

    def _acquisition(self) -> dict:
        return {
            "artifact_root_sha256": common.sha256_bytes(self.artifact_root_raw),
            "artifacts": [
                {
                    "bytes": 100 + index,
                    "path": f"raw/{index:02d}.jsonl",
                    "role": role,
                    "sha256": digest_text(role),
                }
                for index, role in enumerate(self._dynamic_roles())
            ],
            "candidate_sha256": common.sha256_bytes(self.candidate_raw),
            "completed_ns": 2000,
            "contract_sha256": common.sha256_bytes(self.contract_raw),
            "fresh_readiness_sha256": common.sha256_bytes(self.fresh_raw),
            "phase": "A_ONLY",
            "phase_id": self.phase_lock["phase_id"],
            "phase_lock_sha256": common.sha256_bytes(self.phase_lock_raw),
            "preparation_sha256": common.sha256_bytes(self.preparation_raw),
            "runtime_bundle_plan_sha256": common.sha256_bytes(self.plan_raw),
            "runtime_identity_sha256": common.sha256_bytes(self.runtime_raw),
            "raw_manifest_name": evidence.RAW_MANIFEST_NAME,
            "raw_manifest_sha256": ZERO,
            "raw_predicate_contract_sha256": self.contract[
                "raw_predicate_contract"
            ]["sha256"],
            "schema": "s39-cp0-r1-a-only-acquisition-v2.4",
            "started_ns": 1000,
            "status": "RAW_CAPTURE_COMPLETE_UNEVALUATED",
        }


class V24EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def assert_refused(self, code: str):
        with self.assertRaisesRegex(common.EvidenceError, code):
            evidence.validate_chain(**self.fixture.kwargs())

    def test_valid_chain_passes(self):
        result = evidence.validate_chain(**self.fixture.kwargs())
        self.assertEqual(
            result["status"],
            "V2_4_READINESS_CHAIN_PASS_RAW_QUALIFICATION_NOT_EVALUATED",
        )

    def test_stale_artifact_root_is_rejected(self):
        self.fixture.phase_lock["event_ns"] = (
            self.fixture.artifact_root["completed_ns"]
            + self.fixture.contract["gates"]["artifact_root_maximum_age_ns"]
            + 1
        )
        self.fixture.rewrite("phase_lock", self.fixture.phase_lock)
        self.assert_refused("E_ARTIFACT_ROOT_STALE")

    def test_post_hash_stat_mutation_is_rejected(self):
        self.fixture.fresh["component_stats"][0]["stat"]["mtime_ns"] += 1
        self.fixture.rewrite("fresh", self.fixture.fresh)
        self.assert_refused("E_POST_HASH_MUTATION")

    def test_root_completed_after_reboot_start_is_rejected(self):
        self.fixture.preparation["started_ns"] = 190
        self.fixture.preparation["reboot_started_ns"] = 199
        self.fixture.rewrite("preparation", self.fixture.preparation)
        self.assert_refused("E_PREPARATION_ORDER")

    def test_stale_pre_reboot_boot_id_is_rejected(self):
        self.fixture.preparation["before_boot_ids"]["op12"] = BOOT_IDS["op12"]
        self.fixture.rewrite("preparation", self.fixture.preparation)
        self.assert_refused("E_STALE_BOOT_REUSE")

    def test_reused_phone_ipv4_is_rejected(self):
        self.fixture.preparation["devices"]["op15"]["local_ipv4"] = (
            self.fixture.preparation["devices"]["op12"]["local_ipv4"]
        )
        self.fixture.rewrite("preparation", self.fixture.preparation)
        self.assert_refused("E_PHONE_IPV4_REUSE")

    def test_non_wifi_phone_interface_is_rejected(self):
        self.fixture.preparation["devices"]["op15"]["interface"] = "rmnet0"
        self.fixture.rewrite("preparation", self.fixture.preparation)
        self.assert_refused("preparation.devices.op15.interface")

    def test_embedded_identity_sentinel_is_rejected(self):
        with self.assertRaisesRegex(common.EvidenceError, "E_SENTINEL_LEAKAGE"):
            evidence._reject_unbound_identity(
                {"argv": ["--peer", "tcp://0.0.0.15:9000"]},
                "bound",
            )

    def test_phase_root_mismatch_is_rejected(self):
        self.fixture.phase_lock["artifact_root_sha256"] = ZERO
        self.fixture.rewrite("phase_lock", self.fixture.phase_lock)
        self.assert_refused("phase_lock.root")

    def test_phase_quality_corpus_mismatch_is_rejected(self):
        self.fixture.phase_lock["quality_corpus_sha256"] = ZERO
        self.fixture.rewrite("phase_lock", self.fixture.phase_lock)
        self.assert_refused("phase_lock.quality_corpus")

    def test_fresh_before_lock_is_rejected(self):
        self.fixture.fresh["started_ns"] = 399
        self.fixture.rewrite("fresh", self.fixture.fresh)
        self.assert_refused("E_FRESH_BEFORE_LOCK")

    def test_slow_fast_check_is_rejected(self):
        self.fixture.fresh["completed_ns"] = (
            self.fixture.fresh["started_ns"]
            + self.fixture.contract["gates"]["fast_check_maximum_duration_ns"]
            + 1
        )
        self.fixture.rewrite("fresh", self.fixture.fresh)
        self.assert_refused("E_FAST_CHECK_SLOW")

    def test_old_fresh_snapshot_is_rejected(self):
        self.fixture.acquisition["started_ns"] = (
            self.fixture.fresh["completed_ns"]
            + self.fixture.contract["gates"]["fresh_snapshot_maximum_age_ns"]
            + 1
        )
        self.fixture.rewrite("acquisition", self.fixture.acquisition)
        self.assert_refused("E_FRESH_STALE")

    def test_missing_process_identity_is_rejected(self):
        self.fixture.runtime["processes"].pop()
        self.fixture.rewrite("runtime", self.fixture.runtime)
        self.fixture.acquisition["runtime_identity_sha256"] = common.sha256_file(
            self.fixture.runtime_path
        )
        self.fixture.rewrite("acquisition", self.fixture.acquisition)
        self.assert_refused("E_RUNTIME_PROCESSES")

    def test_runtime_boot_change_is_rejected(self):
        self.fixture.runtime["processes"][0]["boot_id"] = (
            "44444444-4444-4444-4444-444444444444"
        )
        self.fixture.rewrite("runtime", self.fixture.runtime)
        self.fixture.acquisition["runtime_identity_sha256"] = common.sha256_file(
            self.fixture.runtime_path
        )
        self.fixture.rewrite("acquisition", self.fixture.acquisition)
        self.assert_refused("E_RUNTIME_BOOT")

    def test_runtime_bundle_digest_change_is_rejected(self):
        self.fixture.runtime["processes"][0]["bundle_sha256"] = ZERO
        self.fixture.rewrite("runtime", self.fixture.runtime)
        self.fixture.acquisition["runtime_identity_sha256"] = common.sha256_file(
            self.fixture.runtime_path
        )
        self.fixture.rewrite("acquisition", self.fixture.acquisition)
        self.assert_refused("E_RUNTIME_BUNDLE_DIGEST")

    def test_runtime_loaded_component_omission_is_rejected(self):
        self.fixture.runtime["processes"][0]["loaded_repo_component_ids"].pop()
        self.fixture.rewrite("runtime", self.fixture.runtime)
        self.fixture.acquisition["runtime_identity_sha256"] = common.sha256_file(
            self.fixture.runtime_path
        )
        self.fixture.rewrite("acquisition", self.fixture.acquisition)
        self.assert_refused("E_RUNTIME_COMPONENTS")

    def test_cuda_monolithic_missing_mapping_is_rejected(self):
        self.fixture.runtime["processes"][0]["model_mapping"] = None
        self.fixture.rewrite("runtime", self.fixture.runtime)
        self.fixture.acquisition["runtime_identity_sha256"] = common.sha256_file(
            self.fixture.runtime_path
        )
        self.fixture.rewrite("acquisition", self.fixture.acquisition)
        self.assert_refused("model_mapping")

    def test_cuda_monolithic_wrong_maps_inode_is_rejected(self):
        self.fixture.runtime["processes"][0]["model_mapping"][
            "model_mapping_rows"
        ][0]["inode"] += 1
        self.fixture.rewrite("runtime", self.fixture.runtime)
        self.fixture.acquisition["runtime_identity_sha256"] = common.sha256_file(
            self.fixture.runtime_path
        )
        self.fixture.rewrite("acquisition", self.fixture.acquisition)
        self.assert_refused("E_CUDA_MONOLITHIC_MAPS_INODE")

    def test_cuda_monolithic_missing_mapping_row_is_rejected(self):
        self.fixture.runtime["processes"][0]["model_mapping"][
            "model_mapping_rows"
        ].pop()
        self._rewrite_runtime_and_acquisition()
        self.assert_refused("E_CUDA_MONOLITHIC_MAPS_COUNT")

    def test_cuda_monolithic_extra_mapping_row_is_rejected(self):
        rows = self.fixture.runtime["processes"][0]["model_mapping"][
            "model_mapping_rows"
        ]
        extra = copy.deepcopy(rows[-1])
        extra["offset_bytes"] += 4096
        rows.append(extra)
        self._rewrite_runtime_and_acquisition()
        self.assert_refused("E_CUDA_MONOLITHIC_MAPS_COUNT")

    def test_cuda_monolithic_wrong_mapping_offset_is_rejected(self):
        self.fixture.runtime["processes"][0]["model_mapping"][
            "model_mapping_rows"
        ][0]["offset_bytes"] += 4096
        self._rewrite_runtime_and_acquisition()
        self.assert_refused("E_CUDA_MONOLITHIC_MAPS_ROWS")

    def test_cuda_monolithic_wrong_mapping_permissions_is_rejected(self):
        self.fixture.runtime["processes"][0]["model_mapping"][
            "model_mapping_rows"
        ][0]["permissions"] = "rw-s"
        self._rewrite_runtime_and_acquisition()
        self.assert_refused("E_CUDA_MONOLITHIC_MAPS_ROWS")

    def test_cuda_monolithic_other_gguf_mapping_is_rejected(self):
        self.fixture.runtime["processes"][0]["model_mapping"][
            "other_gguf_mapping_paths"
        ] = ["/models/other.gguf"]
        self._rewrite_runtime_and_acquisition()
        self.assert_refused("E_CUDA_MONOLITHIC_OTHER_GGUF")

    def test_cuda_monolithic_missing_launch_sha_is_rejected(self):
        self.fixture.runtime["processes"][0]["model_mapping"]["environment"][
            "LAYERSPLIT_MODEL_SHA256"
        ] = ZERO
        self.fixture.rewrite("runtime", self.fixture.runtime)
        self.fixture.acquisition["runtime_identity_sha256"] = common.sha256_file(
            self.fixture.runtime_path
        )
        self.fixture.rewrite("acquisition", self.fixture.acquisition)
        self.assert_refused("E_CUDA_MONOLITHIC_LIVE_ENV")

    def test_extra_inventory_path_is_rejected(self):
        self.fixture.fresh["inventories"][0]["paths"].append("/tmp/extra")
        self.fixture.rewrite("fresh", self.fixture.fresh)
        self.assert_refused("fresh.inventories")

    def _rewrite_runtime_and_acquisition(self):
        self.fixture.rewrite("runtime", self.fixture.runtime)
        self.fixture.acquisition["runtime_identity_sha256"] = common.sha256_file(
            self.fixture.runtime_path
        )
        self.fixture.rewrite("acquisition", self.fixture.acquisition)

    def test_positive_system_swap_growth_is_rejected(self):
        self.fixture.runtime["phone_after"]["op12"][
            "system_swap_used_bytes"
        ] += 1
        self._rewrite_runtime_and_acquisition()
        self.assert_refused("E_SYSTEM_SWAP_GROWTH")

    def test_system_swap_counter_reset_is_rejected(self):
        self.fixture.runtime["phone_after"]["op12"][
            "system_swap_used_bytes"
        ] -= 1
        self._rewrite_runtime_and_acquisition()
        self.assert_refused("E_SWAP_COUNTER_RESET")

    def test_system_swap_after_sample_is_required(self):
        del self.fixture.runtime["phone_after"]["op12"]
        self._rewrite_runtime_and_acquisition()
        self.assert_refused("runtime.phone_after")

    def test_system_swap_boot_change_is_rejected(self):
        self.fixture.runtime["phone_after"]["op12"]["boot_id"] = BOOT_IDS[
            "op15"
        ]
        self._rewrite_runtime_and_acquisition()
        self.assert_refused("E_SWAP_BOOT")

    def test_worker_process_swap_is_exact_zero(self):
        self.fixture.runtime["processes"][0]["process_swap_bytes"] = 1
        self._rewrite_runtime_and_acquisition()
        self.assert_refused("runtime.processes\\[0\\].swap")

    def test_tokenizer_identity_change_is_rejected(self):
        self.fixture.history["tokenizer"]["sha256"] = ZERO
        self.fixture.rewrite("history", self.fixture.history)
        self.assert_refused("token_history.tokenizer.sha256")

    def test_prefill_row_order_change_is_rejected(self):
        rows = self.fixture.history["quality_groups"][0][
            "prefill_partitions"
        ][0]["rows"]
        rows[0], rows[1] = rows[1], rows[0]
        self.fixture.rewrite("history", self.fixture.history)
        self.assert_refused("token_history.quality_groups")

    def test_wire_request_mapping_change_is_rejected(self):
        self.fixture.history["requests"][0]["request_id"] = 0
        self.fixture.rewrite("history", self.fixture.history)
        self.assert_refused("token_history.requests\\[0\\].request_id")

    def test_later_group_wire_mapping_change_is_rejected(self):
        self.fixture.history["requests"][8]["request_id"] = 9
        self.fixture.rewrite("history", self.fixture.history)
        self.assert_refused("token_history.requests\\[8\\].request_id")

    def test_mechanics_group_must_equal_quality_group_zero(self):
        self.fixture.history["mechanics_b8"] = copy.deepcopy(
            self.fixture.history["mechanics_b8"]
        )
        self.fixture.history["mechanics_b8"]["group_index"] = 1
        self.fixture.rewrite("history", self.fixture.history)
        self.assert_refused("token_history.mechanics_b8")

    def test_missing_quality_group_is_rejected(self):
        self.fixture.history["quality_groups"].pop()
        self.fixture.rewrite("history", self.fixture.history)
        self.assert_refused("token_history.quality_groups")

    def test_history_over_context_limit_is_rejected(self):
        self.fixture.history["requests"][0]["token_ids"] = list(
            range(
                self.fixture.contract["serving_envelope"]["n_ctx_seq"]
                - self.fixture.history["continuation_tokens_per_request"]
                + 1
            )
        )
        self.fixture.rewrite("history", self.fixture.history)
        self.assert_refused("E_TOKEN_IDS")

    def test_position_wave_split_is_rejected(self):
        partitions = self.fixture.history["quality_groups"][0][
            "prefill_partitions"
        ]
        first = partitions[0]["rows"]
        second = partitions[1]["rows"]
        first.extend(second[:2])
        del second[:2]
        self.fixture.rewrite("history", self.fixture.history)
        self.assert_refused("token_history.quality_groups")

    def test_missing_decode_call_is_rejected(self):
        self.fixture.history["quality_groups"][0]["decode_calls"].pop()
        self.fixture.rewrite("history", self.fixture.history)
        self.assert_refused("token_history.quality_groups")

    def test_fixed_two_token_history_is_rejected(self):
        for request in self.fixture.history["requests"]:
            request["token_ids"] = request["token_ids"][:2]
        self.fixture.rewrite("history", self.fixture.history)
        self.assert_refused("token_history.quality_groups")

    def test_legacy_contract_is_rejected(self):
        legacy = S39 / "v23_readiness" / "CP0_R1_EVIDENCE_CONTRACT_V2_3.json"
        self.fixture.contract_path.write_bytes(legacy.read_bytes())
        self.assert_refused("contract")

    def test_legacy_256_route_envelope_is_rejected(self):
        import cp0_r1_evidence_v2 as v2

        legacy, _ = common.read_canonical(evidence.V22_CONTRACT)
        legacy_digest = v2.digest_json(legacy["serving_envelope"])
        with self.assertRaisesRegex(common.EvidenceError, "E_V2_4_ROUTE_ENVELOPE"):
            evidence.validate_route_envelope_digest(
                legacy_digest,
                self.fixture.contract,
            )
        evidence.validate_route_envelope_digest(
            v2.digest_json(self.fixture.contract["serving_envelope"]),
            self.fixture.contract,
        )

    def test_raw_predicate_contract_is_512_and_bound(self):
        raw, raw_bytes, parent = evidence.raw_predicate_inputs(
            self.fixture.contract
        )
        self.assertEqual(raw["serving_envelope"]["n_ctx_seq"], 512)
        self.assertEqual(parent["serving_envelope"]["n_ctx_seq"], 512)
        self.assertEqual(
            common.sha256_bytes(raw_bytes),
            self.fixture.contract["raw_predicate_contract"]["sha256"],
        )

    def test_runtime_launch_shape_matches_real_producer_builder(self):
        builder_path = (
            HERE
            / "producers_v1"
            / "build_cuda_monolithic_launch_v1.py"
        )
        spec = importlib.util.spec_from_file_location(
            "v24_launch_builder_shape",
            builder_path,
        )
        self.assertIsNotNone(spec)
        launch_builder = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = launch_builder
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(launch_builder)
        root = self.fixture.root / "launch-shape"
        bundle = root / "bundle"
        bundle.mkdir(parents=True)
        launcher = bundle / "llama-layersplit"
        launcher.write_bytes(b"launcher")
        launcher.chmod(0o755)
        library = bundle / "libggml.so.0"
        library.write_bytes(b"library")
        model = root / "model.gguf"
        model.write_bytes(b"model")
        value = launch_builder.build_launch(
            launch_builder.LaunchConfig(
                allowed_system_roots=launch_builder.ALLOWED_SYSTEM_ROOTS,
                bundle_root=bundle,
                components=(
                    launch_builder.ComponentSpec(
                        "cuda-mono.bin",
                        launcher,
                        common.sha256_file(launcher),
                    ),
                    launch_builder.ComponentSpec(
                        "cuda-mono.lib",
                        library,
                        common.sha256_file(library),
                    ),
                ),
                cwd=root,
                environment=(
                    ("CUDA_VISIBLE_DEVICES", "0"),
                    ("HOME", "/home/zhihao"),
                    ("LAYERSPLIT_MEMORY_CERT", "1"),
                    (
                        "LAYERSPLIT_MODEL_SHA256",
                        common.sha256_file(model),
                    ),
                    ("LAYERSPLIT_PLACEMENT_CERT", "1"),
                    ("LC_ALL", "C"),
                    ("LD_LIBRARY_PATH", str(bundle)),
                ),
                expected_file_type=15,
                expected_n_embd=5120,
                launcher_component_id="cuda-mono.bin",
                model_bytes=model.stat().st_size,
                model_path=model,
                model_sha256=common.sha256_file(model),
                port=39124,
            ),
            17,
        )
        frozen = self.fixture.plan["cuda_monolithic_launch"]
        self.assertEqual(set(value), set(frozen))
        self.assertEqual(
            value["command"][3:],
            frozen["command"][3:],
        )
        self.assertEqual(set(value["env"]), set(frozen["env"]))
        for key in (
            "expected_capabilities",
            "expected_file_type",
            "expected_max_streams",
            "expected_n_batch",
            "expected_n_ctx_seq",
            "expected_n_embd",
            "expected_n_layer",
            "expected_n_ubatch",
            "io_timeout_ms",
            "shutdown_timeout_ms",
            "startup_timeout_ms",
        ):
            self.assertEqual(value[key], frozen[key])

    def test_raw_predicate_evaluator_accepts_512_route(self):
        tests_path = str(S39 / "tests")
        if tests_path not in sys.path:
            sys.path.insert(0, tests_path)
        import cp0_r1_evidence_v2 as v2
        import cp0_r1_evidence_v21 as v21
        import cp0_r1_evidence_v22 as v22
        from test_cp0_r1_evidence_v22 import V22PhaseFixture

        raw, raw_bytes, parent = evidence.raw_predicate_inputs(
            self.fixture.contract
        )
        frozen = v22.load_frozen_corpus(raw)
        raw_root = self.fixture.root / "raw-positive"
        phase = V22PhaseFixture(
            raw_root,
            "A_ONLY",
            raw,
            raw_bytes,
            parent,
            self.fixture.candidate,
            self.fixture.candidate_raw,
            [],
            [],
            frozen_corpus=frozen,
        )
        linked_history = self.fixture.history
        group = linked_history["mechanics_b8"]
        call_shapes = [
            {
                "call_index": item["call_index"],
                "n_seqs": len({row["seq_id"] for row in item["rows"]}),
                "n_tokens": len(item["rows"]),
                "phase": phase_name,
                "positions": [row["position"] for row in item["rows"]],
                "request_ids": [row["request_id"] for row in item["rows"]],
                "seq_ids": [row["seq_id"] for row in item["rows"]],
            }
            for phase_name, values in (
                ("prefill", group["prefill_partitions"]),
                ("decode", group["decode_calls"]),
            )
            for item in values
        ]
        prefix = f"model.{evidence.MODEL_ID}"
        execution_roles = (
            f"{prefix}.mechanics.phone",
            f"{prefix}.oracle.cuda_route",
            f"{prefix}.oracle.cuda_monolithic",
        )
        for role in execution_roles:
            phase.rows[role][0]["call_shapes"] = copy.deepcopy(call_shapes)
            request_rows = sorted(
                (
                    row for row in phase.rows[role]
                    if row["kind"] == "request"
                ),
                key=lambda row: row["request_id"],
            )
            for local_id, row in enumerate(request_rows):
                history_row = linked_history["requests"][local_id]
                row["request_id"] = local_id + 1
                row["input_tokens"] = history_row["token_ids"]
                row["positions"] = list(range(len(history_row["token_ids"])))

        mechanics_role = f"{prefix}.mechanics.phone"
        normalized_phone = {
            row["request_id"]: row
            for row in v21._normalize_rows(phase.rows[mechanics_role])
            if row["kind"] == "request"
        }
        for row in phase.rows[f"{prefix}.bridge"]:
            if row["kind"] != "phone_publication_received":
                continue
            row["request_id"] += 1
            row["phone_request_sha256"] = v2.digest_json(
                normalized_phone[row["request_id"]]
            )
        transfer_rows = [
            row for row in phase.rows[f"{prefix}.route_transfer"]
            if row["kind"] == "transfer"
        ]
        self.assertEqual(len(transfer_rows), len(call_shapes))
        for call_index, (row, call) in enumerate(
            zip(transfer_rows, call_shapes)
        ):
            row["call_index"] = call_index
            row["row_count"] = call["n_tokens"]
            row["payload_bytes"] = (
                call["n_tokens"]
                * phase.lock["hidden_size"]
                * phase.lock["activation_element_bytes"]
            )
        phase.build_phase_lock()
        phase.write()
        (phase.root / "EVIDENCE_BUNDLE.json").rename(
            phase.root / evidence.RAW_MANIFEST_NAME
        )
        result = evidence.evaluate_raw_predicates(
            phase.root,
            self.fixture.contract,
            self.fixture.candidate,
            self.fixture.candidate_raw,
            linked_history,
        )
        self.assertEqual(result["status"], "MODEL_A_QUALIFICATION_PASS")
        self.assertEqual(
            result["derived"]["model"]["route_lock"]["batch_config_sha256"],
            __import__("cp0_r1_evidence_v2").digest_json(
                self.fixture.contract["serving_envelope"]
            ),
        )

    def test_raw_predicate_contract_digest_change_is_rejected(self):
        self.fixture.acquisition["raw_predicate_contract_sha256"] = ZERO
        self.fixture.rewrite("acquisition", self.fixture.acquisition)
        self.assert_refused("acquisition.raw_predicate_contract_sha256")

    def _valid_path_rows(self):
        group = self.fixture.history["mechanics_b8"]
        calls = [
            {
                "call_index": item["call_index"],
                "n_seqs": len({row["seq_id"] for row in item["rows"]}),
                "n_tokens": len(item["rows"]),
                "phase": phase,
                "positions": [row["position"] for row in item["rows"]],
                "request_ids": [row["request_id"] for row in item["rows"]],
                "seq_ids": [row["seq_id"] for row in item["rows"]],
            }
            for phase, values in (
                ("prefill", group["prefill_partitions"]),
                ("decode", group["decode_calls"]),
            )
            for item in values
        ]
        rows = {}
        for suffix in (
            "mechanics.phone",
            "oracle.cuda_route",
            "oracle.cuda_monolithic",
        ):
            role = f"model.{evidence.MODEL_ID}.{suffix}"
            rows[role] = [{"call_shapes": calls, "kind": "meta"}]
            rows[role].extend(
                {
                    "continuation_tokens": list(range(8)),
                    "input_tokens": self.fixture.history["requests"][
                        item_index
                    ]["token_ids"],
                    "kind": "request",
                    "positions": list(
                        range(
                            len(
                                self.fixture.history["requests"][item_index][
                                    "token_ids"
                                ]
                            )
                        )
                    ),
                    "request_id": local_id + 1,
                }
                for local_id, item_index in enumerate(group["item_indices"])
            )
        return rows

    def test_path_input_not_matching_frozen_history_is_rejected(self):
        rows = self._valid_path_rows()
        evidence.validate_path_matched_history(rows, self.fixture.history)
        rows[f"model.{evidence.MODEL_ID}.oracle.cuda_route"][1][
            "input_tokens"
        ] = [999]
        with self.assertRaisesRegex(common.EvidenceError, "E_PATH_HISTORY_TOKENS"):
            evidence.validate_path_matched_history(rows, self.fixture.history)

    def test_legacy_zero_based_raw_wire_ids_are_rejected(self):
        rows = self._valid_path_rows()
        for values in rows.values():
            for row in values[1:]:
                row["request_id"] -= 1
        with self.assertRaisesRegex(common.EvidenceError, "E_PATH_HISTORY_REQUESTS"):
            evidence.validate_path_matched_history(rows, self.fixture.history)

    def test_permuted_raw_wire_id_binding_is_rejected(self):
        rows = self._valid_path_rows()
        for values in rows.values():
            values[1]["request_id"], values[2]["request_id"] = (
                values[2]["request_id"],
                values[1]["request_id"],
            )
        with self.assertRaisesRegex(common.EvidenceError, "E_PATH_HISTORY_TOKENS"):
            evidence.validate_path_matched_history(rows, self.fixture.history)

    def test_legacy_eighth_decode_call_is_rejected(self):
        rows = self._valid_path_rows()
        for values in rows.values():
            last = copy.deepcopy(values[0]["call_shapes"][-1])
            last["call_index"] += 1
            last["positions"] = [position + 1 for position in last["positions"]]
            values[0]["call_shapes"].append(last)
        with self.assertRaisesRegex(common.EvidenceError, "E_PATH_HISTORY_CALLS"):
            evidence.validate_path_matched_history(rows, self.fixture.history)

    def test_mutated_authority_support_pin_is_rejected(self):
        self.fixture.contract["exit_authority"]["support"]["common"][
            "sha256"
        ] = ZERO
        self.fixture.rewrite("contract", self.fixture.contract)
        self.assert_refused("contract")

    def test_contract_pins_orchestration_source_closure(self):
        requirements = self.fixture.contract["orchestration_requirements"]
        self.assertEqual(
            self.fixture.contract["phase_protocol"]["order"],
            [
                "ARTIFACT_ROOT",
                "REBOOT_PREPARATION",
                "PHASE_LOCK",
                "POST_REBOOT_IDENTITY_BINDING",
                "FAST_FRESH_READINESS",
                "ACQUISITION",
                "RUNTIME_IDENTITY",
                "V2_4_RAW_PREDICATE_REEVALUATION",
            ],
        )
        self.assertEqual(
            set(requirements["source_programs"]),
            {
                "fan_in",
                "identity_binding",
                "phase_lock",
                "preparation",
                "readiness_projection",
            },
        )
        for name, path in builder.ORCHESTRATION_PROGRAMS.items():
            raw = path.read_bytes()
            self.assertEqual(
                requirements["source_programs"][name],
                {
                    "bytes": len(raw),
                    "path": str(path.relative_to(evidence.S39)),
                    "sha256": common.sha256_bytes(raw),
                },
            )
        self.assertEqual(
            set(requirements["stage_support"]),
            set(requirements["source_programs"]),
        )
        self.assertIn("production_common", requirements["support"])
        self.assertIn("orchestration", requirements["support"])
        self.assertIn("authority", requirements["support"])
        for name, path in builder.ORCHESTRATION_SUPPORT.items():
            raw = path.read_bytes()
            self.assertEqual(
                requirements["support"][name],
                {
                    "bytes": len(raw),
                    "path": str(path.relative_to(evidence.S39)),
                    "sha256": common.sha256_bytes(raw),
                },
            )

    def test_mutated_orchestration_source_pin_is_rejected(self):
        self.fixture.contract["orchestration_requirements"]["source_programs"][
            "fan_in"
        ]["sha256"] = ZERO
        self.fixture.rewrite("contract", self.fixture.contract)
        self.assert_refused("contract")

    def test_evaluator_helper_source_mutation_is_rejected(self):
        contract = builder.build_contract()
        helper_keys = (
            "builder_v21",
            "builder_v22",
            "evaluator_v2",
            "evaluator_v21",
            "evaluator_v22",
            "mmlu_builder_v22",
        )
        copied_root = self.fixture.root / "verified-sources"
        for key in helper_keys:
            record = contract["exit_authority"]["support"][key]
            source = evidence.S39 / record["path"]
            target = copied_root / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        with mock.patch.object(evidence, "S39", copied_root):
            evidence._verified_helpers(contract)
            target = (
                copied_root
                / contract["exit_authority"]["support"]["evaluator_v2"]["path"]
            )
            target.write_bytes(target.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                common.EvidenceError,
                "authority.evaluator_v2.bytes",
            ):
                evidence._verified_helpers(contract)

    def test_verified_source_execution_ignores_valid_malicious_pyc(self):
        source = self.fixture.root / "cached_helper.py"
        source.write_text("VALUE = 2\n", encoding="ascii")
        timestamp = 1_700_000_000
        os.utime(source, (timestamp, timestamp))
        py_compile.compile(str(source), doraise=True)
        source.write_text("VALUE = 1\n", encoding="ascii")
        os.utime(source, (timestamp, timestamp))
        spec = importlib.util.spec_from_file_location(
            "cached_helper_probe",
            source,
        )
        self.assertIsNotNone(spec)
        cached = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cached)
        self.assertEqual(cached.VALUE, 2)
        verified, executed = evidence._execute_source(
            "verified_helper_probe",
            source,
        )
        self.assertEqual(executed, b"VALUE = 1\n")
        self.assertEqual(verified.VALUE, 1)

    def test_status_only_manifest_cannot_authorize(self):
        bundle = self.fixture.root / "bundle"
        bundle.mkdir()
        (bundle / evidence.RAW_MANIFEST_NAME).write_bytes(
            common.canonical_bytes(
                {
                    "phase": "A_ONLY",
                    "phase_id": self.fixture.phase_lock["phase_id"],
                    "status": "MODEL_A_QUALIFICATION_PASS",
                }
            )
        )
        self.fixture.acquisition["raw_manifest_sha256"] = common.sha256_file(
            bundle / evidence.RAW_MANIFEST_NAME
        )
        self.fixture.rewrite("acquisition", self.fixture.acquisition)
        with self.assertRaises(common.EvidenceError):
            evidence.authorize_a_only(
                bundle_root=bundle,
                chain_kwargs=self.fixture.kwargs(),
                evaluator=lambda _: {
                    "schema": "s39-cp0-r1-evidence-result-v2.2",
                    "status": "MODEL_A_QUALIFICATION_PASS",
                },
            )

    def test_contract_only_cli_reports_producer_block(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(HERE / "cp0_r1_evidence_v24.py"),
                "--contract",
                str(self.fixture.contract_path),
                "--candidate",
                str(self.fixture.candidate_path),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "V2_4_MECHANICS_CONTRACT_PRODUCERS_BLOCKED",
            completed.stdout,
        )

    def test_incomplete_cli_is_fail_closed_without_traceback(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(HERE / "cp0_r1_evidence_v24.py"),
                "--contract",
                str(self.fixture.contract_path),
                "--candidate",
                str(self.fixture.candidate_path),
                "--runtime-plan",
                str(self.fixture.plan_path),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("E_ARGUMENTS", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


if __name__ == "__main__":
    unittest.main()
