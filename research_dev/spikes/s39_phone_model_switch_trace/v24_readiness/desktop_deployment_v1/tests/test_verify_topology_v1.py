#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import base64
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "verify_topology_v1.py"
SPEC = importlib.util.spec_from_file_location("verify_topology_v1", SOURCE)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def completed(argv, stdout=b"", stderr=b"", returncode=0):
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


class FakeRunner:
    def __init__(self, rows):
        self.rows = {
            tuple(argv): completed(argv, stdout)
            for argv, stdout in rows
        }

    def run(self, argv, *, timeout):
        del timeout
        key = tuple(argv)
        if key not in self.rows:
            raise AssertionError(f"unexpected command: {argv}")
        return self.rows[key]


class Counter:
    def __init__(self, start=1000):
        self.value = start

    def __call__(self):
        self.value += 10
        return self.value


def contract(candidate_bytes=1, candidate_sha256="0" * 64):
    return {
        "candidate_lock": {
            "bytes": candidate_bytes,
            "model_id": verify.MODEL_ID,
            "sha256": candidate_sha256,
            "slot": "A",
        },
        "devices": {
            "cuda": {
                "host": verify.CONTROLLER_HOST,
                "memory_total_bytes": verify.CUDA_MEMORY_TOTAL_BYTES,
                "name": verify.CUDA_NAME,
                "uuid": verify.CUDA_UUID,
            },
            **copy.deepcopy(verify.EXPECTED_PHONES),
        },
        "quality_corpus": {
            "bytes": 1,
            "sha256": "0" * 64,
        },
        "schema": "s39-cp0-r1-evidence-contract-v2.4",
        "token_history_protocol": {
            "batch": 8,
            "continuation_tokens_per_request": 8,
            "decode_calls_after_prefill": 7,
            "mechanics_item_indices": list(range(8)),
            "n_batch": 64,
            "n_ctx_seq": 512,
            "n_ubatch": 64,
            "prefill_chunking": "WHOLE_POSITION_WAVES_MAX_64_ROWS",
            "prefill_row_order": "POSITION_MAJOR_THEN_ITEM_INDEX",
            "prompt_hash_encoding": "UTF-8",
            "quality_group_count": 8,
            "quality_group_size": 8,
            "quality_items": 64,
            "request_id_mapping": "GROUP_LOCAL_ONE_BASED",
            "seq_id_mapping": "GROUP_LOCAL_ZERO_BASED",
            "vocab_size": 151936,
        },
    }


PROMPT_FORMAT = (
    "Question: {question}\n"
    "A. {choice0}\n"
    "B. {choice1}\n"
    "C. {choice2}\n"
    "D. {choice3}\n"
    "Answer with exactly one uppercase letter: A, B, C, or D.\n"
    "Answer:"
)


def corpus_rows():
    return [
        {
            "choices": [f"A{index}", f"B{index}", f"C{index}", f"D{index}"],
            "dataset": "cais/mmlu",
            "dataset_revision": "b" * 40,
            "expected_answer": "A",
            "item_index": index,
            "question": f"Question {index}?",
            "source_row": index,
            "subject": f"subject-{index:02d}",
        }
        for index in range(64)
    ]


def history_request(item):
    prompt = PROMPT_FORMAT.format(
        question=item["question"],
        choice0=item["choices"][0],
        choice1=item["choices"][1],
        choice2=item["choices"][2],
        choice3=item["choices"][3],
    )
    raw = prompt.encode("utf-8")
    index = item["item_index"]
    return {
        "item_index": index,
        "prompt_sha256": verify.sha256_bytes(raw),
        "prompt_utf8_base64": base64.b64encode(raw).decode("ascii"),
        "prompt_utf8_bytes": len(raw),
        "request_id": index % 8 + 1,
        "seq_id": index % 8,
        "token_ids": [index + 1],
    }


def history_group(requests, group_index):
    item_indices = list(range(group_index * 8, group_index * 8 + 8))
    prefill_rows = [
        {
            "item_index": index,
            "position": 0,
            "request_id": requests[index]["request_id"],
            "seq_id": requests[index]["seq_id"],
            "token_id": requests[index]["token_ids"][0],
        }
        for index in item_indices
    ]
    return {
        "decode_calls": [
            {
                "call_index": call_index + 1,
                "continuation_input_ordinal": call_index,
                "continuation_output_ordinal": call_index + 1,
                "rows": [
                    {
                        "item_index": index,
                        "position": call_index + 1,
                        "request_id": requests[index]["request_id"],
                        "seq_id": requests[index]["seq_id"],
                    }
                    for index in item_indices
                ],
            }
            for call_index in range(7)
        ],
        "group_index": group_index,
        "item_indices": item_indices,
        "prefill_partitions": [{"call_index": 0, "rows": prefill_rows}],
    }


def phone_output(serial, boot_id, address):
    expected = next(
        value
        for value in verify.EXPECTED_PHONES.values()
        if value["serial"] == serial
    )
    return (
        f"PHYSICAL_SERIAL={serial}\n"
        f"PRODUCT={expected['product']}\n"
        f"MODEL={expected['model']}\n"
        f"DEVICE={expected['device']}\n"
        f"BOOT_ID={boot_id}\n"
        "INTERFACE=wlan0\n"
        f"WIFI_IPV4={address}\n"
    ).encode("ascii")


def ping_output(source, interface, target):
    return (
        f"PING {target} ({target}) from {source} {interface}: "
        "0(28) bytes of data.\n\n"
        f"--- {target} ping statistics ---\n"
        "3 packets transmitted, 3 received, 0% packet loss, time 2001ms\n\n"
    ).encode("ascii")


def runner_rows(*, wifi=True):
    controller = (
        b"HOST=zhihao-Z690-C-ac\n"
        b"BOOT_ID=2f68fcf5-54e1-4306-ba25-63359be572c2\n"
    )
    gpu = (
        b"NVIDIA GeForce RTX 4060 Ti, "
        b"GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08, 16380\n"
    )
    path_resolution = (
        b"ADB=/usr/bin/adb\n"
        + f"ADB_REAL={verify.ADB_PATH}\n".encode("ascii")
        + f"NVIDIA_SMI={verify.NVIDIA_SMI_PATH}\n".encode("ascii")
        + f"NVIDIA_SMI_REAL={verify.NVIDIA_SMI_PATH}\n".encode("ascii")
        + b"PYTHON=/usr/bin/python3\n"
        + f"PYTHON_REAL={verify.PYTHON_PATH}\n".encode("ascii")
        + f"SSH={verify.SSH_PATH}\n".encode("ascii")
    )
    self_ssh = controller + gpu
    device_rows = [
        "3C15AU002CL00000 device usb:8-3 product:CPH2749 "
        "model:CPH2749 device:OP611FL1 transport_id:5",
        "5ae7a43d device usb:6-2 product:CPH2583 "
        "model:CPH2583 device:OP595DL1 transport_id:6",
    ]
    selectors = {
        "3C15AU002CL00000": (
            "3C15AU002CL00000",
            "3eb99d7e-0b35-41a5-9e30-21867dc5dec7",
            "172.20.173.218",
        ),
        "5ae7a43d": (
            "5ae7a43d",
            "2ca4b7a3-c9c1-4614-a0d2-5746c57c8c4d",
            "172.20.59.72",
        ),
    }
    if wifi:
        device_rows.extend(
            [
                "172.20.173.218:5555 device product:CPH2749 "
                "model:CPH2749 device:OP611FL1 transport_id:7",
                "172.20.59.72:5555 device product:CPH2583 "
                "model:CPH2583 device:OP595DL1 transport_id:8",
            ]
        )
        selectors["172.20.173.218:5555"] = selectors["3C15AU002CL00000"]
        selectors["172.20.59.72:5555"] = selectors["5ae7a43d"]
    adb = (
        "List of devices attached\n"
        + "\n".join(device_rows)
        + "\n"
    ).encode("ascii")
    rows = [
        (verify.CONTROLLER_COMMAND, controller),
        (verify.GPU_COMMAND, gpu),
        (verify.PATH_RESOLUTION_COMMAND, path_resolution),
        (verify.SELF_SSH_COMMAND, self_ssh),
        (verify.ADB_DEVICES_COMMAND, adb),
    ]
    rows.extend(
        (
            verify.phone_command(selector),
            phone_output(*identity),
        )
        for selector, identity in selectors.items()
    )
    probe_phones = {
        "op12": {
            "usb_selector": "5ae7a43d",
            "wifi_ipv4": "172.20.59.72",
        },
        "op15": {
            "usb_selector": "3C15AU002CL00000",
            "wifi_ipv4": "172.20.173.218",
        },
    }
    rows.extend(
        (
            spec["argv"],
            ping_output(
                spec["source_ipv4"],
                spec["interface"],
                spec["target_ipv4"],
            ),
        )
        for spec in verify.connectivity_specs(probe_phones)
    )
    return rows


class OperatorFixture:
    def __init__(self, test):
        self.temporary = tempfile.TemporaryDirectory()
        test.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.files = {}
        for name in sorted(verify.FILE_KEYS):
            suffix = ".json" if name in verify.JSON_SCHEMAS else ".bin"
            if name == "cuda_runtime":
                path = self.root / "runtime" / "cuda_route" / "worker"
            elif name == "monolithic_runtime":
                path = self.root / "runtime" / "cuda_monolithic" / "worker"
            else:
                path = self.root / "files" / f"{name}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{name}\n".encode("ascii"))
            if name in verify.EXECUTABLE_FILE_KEYS:
                path.chmod(0o755)
            self.files[name] = path
        self.corpus = corpus_rows()
        self.files["quality_corpus"].write_bytes(
            b"".join(verify.canonical_bytes(item) for item in self.corpus)
        )
        self.model_sha256 = verify.sha256_bytes(self.files["model"].read_bytes())
        self.corpus_sha256 = verify.sha256_bytes(
            self.files["quality_corpus"].read_bytes()
        )
        candidate = {
            "candidate_attempt": 1,
            "candidate_attempt_limit": 1,
            "contract_sha256": "0" * 64,
            "historical_routes": {},
            "models": [
                {
                    "artifact": {
                        "bytes": self.files["model"].stat().st_size,
                        "sha256": self.model_sha256,
                    },
                    "model_id": verify.MODEL_ID,
                    "slot": "A",
                }
            ],
            "schema": "s39-cp0-r1-candidate-v1",
            "status": "TEST_ONLY",
            "task_suite": {"prompt_format": PROMPT_FORMAT},
        }
        self._json("candidate", candidate)
        candidate_raw = self.files["candidate"].read_bytes()
        contract_value = contract(
            len(candidate_raw),
            verify.sha256_bytes(candidate_raw),
        )
        contract_value["quality_corpus"] = {
            "bytes": self.files["quality_corpus"].stat().st_size,
            "sha256": self.corpus_sha256,
        }
        self._json("contract", contract_value)
        contract_sha256 = verify.sha256_bytes(
            self.files["contract"].read_bytes()
        )
        self.patches = mock.patch.multiple(
            verify,
            ADB_PATH=str(self.files["adb"]),
            NVIDIA_SMI_PATH=str(self.files["nvidia_smi"]),
            PYTHON_PATH=str(self.files["python"]),
            SSH_PATH=str(self.files["ssh"]),
            CONTRACT_SHA256=contract_sha256,
            ADB_DEVICES_COMMAND=[
                str(self.files["adb"]),
                "-P",
                str(verify.ADB_PORT),
                "devices",
                "-l",
            ],
            GPU_COMMAND=[
                str(self.files["nvidia_smi"]),
                "--query-gpu=name,uuid,memory.total",
                "--format=csv,noheader,nounits",
            ],
            SELF_SSH_COMMAND=[
                str(self.files["ssh"]),
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                "LogLevel=ERROR",
                verify.CUDA_SSH_TARGET,
                (
                    "set -eu; printf 'HOST='; hostname; "
                    "printf 'BOOT_ID='; "
                    "cat /proc/sys/kernel/random/boot_id; "
                    "nvidia-smi --query-gpu=name,uuid,memory.total "
                    "--format=csv,noheader,nounits"
                ),
            ],
        )
        self.patches.start()
        test.addCleanup(self.patches.stop)
        self._json(
            "tokenizer_plan",
            {
                "command_template": [
                    str(self.files["codec"]),
                    "-m",
                    str(self.files["model"]),
                    "--ids",
                    "-f",
                    "{PROMPT_FILE}",
                    "--log-disable",
                ],
                "component_id": "cuda-tokenize",
                "cwd": str(self.files["codec"].parent),
                "environment": {
                    "LC_ALL": "C",
                    "LD_LIBRARY_PATH": str(self.files["codec"].parent),
                },
                "executable": {
                    "bytes": self.files["codec"].stat().st_size,
                    "path": str(self.files["codec"]),
                    "sha256": verify.sha256_bytes(
                        self.files["codec"].read_bytes()
                    ),
                },
                "model": {
                    "bytes": self.files["model"].stat().st_size,
                    "model_id": verify.MODEL_ID,
                    "path": str(self.files["model"]),
                    "sha256": self.model_sha256,
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
            },
        )
        requests = [history_request(item) for item in self.corpus]
        groups = [
            history_group(requests, group_index)
            for group_index in range(8)
        ]
        self._json(
            "token_history",
            {
                "batch": 8,
                "candidate_sha256": verify.sha256_bytes(candidate_raw),
                "continuation_tokens_per_request": 8,
                "corpus_sha256": self.corpus_sha256,
                "mechanics_b8": groups[0],
                "model_id": verify.MODEL_ID,
                "model_sha256": self.model_sha256,
                "n_batch": 64,
                "n_ctx_seq": 512,
                "n_ubatch": 64,
                "prefill_chunking": "WHOLE_POSITION_WAVES_MAX_64_ROWS",
                "prefill_row_order": "POSITION_MAJOR_THEN_ITEM_INDEX",
                "quality_groups": groups,
                "requests": requests,
                "schema": "s39-cp0-r1-token-history-v2.4",
                "tokenizer": {
                    "component_id": "cuda-tokenize",
                    "path": str(self.files["codec"]),
                    "plan_sha256": verify.sha256_bytes(
                        self.files["tokenizer_plan"].read_bytes()
                    ),
                    "sha256": verify.sha256_bytes(
                        self.files["codec"].read_bytes()
                    ),
                },
            },
        )
        self.inventory = self._runtime_inventory()
        self._json("runtime_bundle_inventory", self.inventory)
        mono_components = [
            {
                key: value
                for key, value in component.items()
                if key
                in {"component_id", "path", "sha256", "stat"}
            }
            for component in self.inventory["components"]
            if component["bundle_id"] == "cuda_monolithic"
        ]
        monolithic_launch = {
                "allowed_system_roots": ["/usr/lib/"],
                "bundle_id": "cuda_monolithic",
                "bundle_root": self.inventory["bundle_roots"][
                    "cuda_monolithic"
                ],
                "bundle_sha256": "0" * 64,
                "command": [
                    str(self.files["monolithic_runtime"]),
                    "-m",
                    str(self.files["model"]),
                    "--mode",
                    "monov3",
                    "--port",
                    str(verify.CUDA_MONOLITHIC_PORT),
                    "--devices",
                    "CUDA0",
                    "--driver-batch",
                    "8",
                    "--driver-context",
                    "512",
                    "--driver-max-prefill",
                    "8",
                ],
                "cwd": str(self.root),
                "endpoint": "cuda",
                "env": {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "HOME": "/home/zhihao",
                    "LAYERSPLIT_MEMORY_CERT": "1",
                    "LAYERSPLIT_MODEL_SHA256": self.model_sha256,
                    "LAYERSPLIT_PLACEMENT_CERT": "1",
                    "LC_ALL": "C",
                    "LD_LIBRARY_PATH": self.inventory["bundle_roots"][
                        "cuda_monolithic"
                    ],
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
                "launcher_component_id": "cuda_monolithic.launcher",
                "model_artifact": {
                    "path": str(self.files["model"]),
                    "sha256": self.model_sha256,
                    "stat": verify.stat_record(
                        self.files["model"].stat(follow_symlinks=False)
                    ),
                },
                "model_id": verify.MODEL_ID,
                "model_sha256": self.model_sha256,
                "port": verify.CUDA_MONOLITHIC_PORT,
                "required_components": mono_components,
                "route_epoch": 1,
                "schema": "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
                "shutdown_timeout_ms": 30000,
                "startup_timeout_ms": 300000,
            }
        monolithic_launch["bundle_sha256"] = verify.sha256_bytes(
            verify.canonical_bytes(
                {
                    "bundle_id": "cuda_monolithic",
                    "components": mono_components,
                    "endpoint": "cuda",
                    "launcher_component_id": "cuda_monolithic.launcher",
                    "process_role": "cuda_monolithic",
                    "schema": "s39-cp0-r1-runtime-bundle-root-identity-v2.4",
                }
            )
        )
        self._json("cuda_monolithic_launch", monolithic_launch)
        self._json(
            "topology_receipt",
            verify.capture_topology(
                contract_value,
                runner=FakeRunner(runner_rows(wifi=False)),
                clock_ns=Counter(),
            ),
        )
        self.directories = {
            "cuda_bundle_root": self.inventory["bundle_roots"]["cuda_route"],
            "joint_cwd": str(self._directory("joint_cwd")),
        }
        self.value = self._value()

    def _directory(self, name):
        path = self.root / "directories" / name
        path.mkdir(parents=True)
        return path

    def _json(self, name, value):
        self.files[name].write_bytes(verify.canonical_bytes(value))

    def _component(self, bundle_id, endpoint, component_id, path, role):
        path = Path(path)
        if endpoint == "cuda":
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{component_id}\n".encode("ascii"))
                if role == "executable":
                    path.chmod(0o755)
            raw = path.read_bytes()
            metadata = verify.stat_record(path.stat(follow_symlinks=False))
        else:
            raw = f"{component_id}\n".encode("ascii")
            metadata = {
                "ctime_ns": 1,
                "device_id": 1,
                "inode": 1000 + len(component_id),
                "mode": 0o100755 if role == "executable" else 0o100644,
                "mtime_ns": 1,
                "size": len(raw),
            }
        return {
            "bundle_id": bundle_id,
            "bytes": len(raw),
            "component_id": component_id,
            "endpoint": endpoint,
            "path": str(path),
            "role": role,
            "sha256": verify.sha256_bytes(raw),
            "stat": metadata,
        }

    def _runtime_inventory(self):
        roots = {
            "cuda_monolithic": str(
                self.files["monolithic_runtime"].parent
            ),
            "cuda_route": str(self.files["cuda_runtime"].parent),
            "op12_stagenet": str(self.root / "remote" / "op12"),
            "op15_direct_relay": str(
                self.root / "remote" / "op15-relay"
            ),
            "op15_stagenet": str(self.root / "remote" / "op15-stage"),
        }
        launchers = {
            "cuda_monolithic": (
                "cuda",
                "cuda_monolithic.launcher",
                self.files["monolithic_runtime"],
            ),
            "cuda_route": (
                "cuda",
                "cuda_route.launcher",
                self.files["cuda_runtime"],
            ),
            "op12_stagenet": (
                "op12",
                "op12.launcher",
                Path(roots["op12_stagenet"]) / "worker",
            ),
            "op15_direct_relay": (
                "op15",
                "op15.relay",
                Path(roots["op15_direct_relay"]) / "relay",
            ),
            "op15_stagenet": (
                "op15",
                "op15.launcher",
                Path(roots["op15_stagenet"]) / "worker",
            ),
        }
        components = []
        bundles = []
        for bundle_id in sorted(verify.RUNTIME_BUNDLE_IDS):
            endpoint, launcher_id, launcher_path = launchers[bundle_id]
            if bundle_id == "op15_direct_relay":
                components.append(
                    self._component(
                        bundle_id,
                        endpoint,
                        launcher_id,
                        launcher_path,
                        "executable",
                    )
                )
                bundles.append(
                    {
                        "bundle_id": bundle_id,
                        "endpoint": endpoint,
                        "launcher_component_id": launcher_id,
                        "process_role": bundle_id,
                        "required_component_ids": [launcher_id],
                    }
                )
                continue
            library_id = f"{bundle_id}.library"
            library_path = Path(roots[bundle_id]) / "lib.so"
            components.extend(
                [
                    self._component(
                        bundle_id,
                        endpoint,
                        launcher_id,
                        launcher_path,
                        "executable",
                    ),
                    self._component(
                        bundle_id,
                        endpoint,
                        library_id,
                        library_path,
                        "shared_library",
                    ),
                ]
            )
            bundles.append(
                {
                    "bundle_id": bundle_id,
                    "endpoint": endpoint,
                    "launcher_component_id": launcher_id,
                    "process_role": bundle_id,
                    "required_component_ids": sorted(
                        [launcher_id, library_id]
                    ),
                }
            )
        return {
            "bundle_roots": roots,
            "bundles": bundles,
            "closure_complete": True,
            "components": components,
            "schema": "s39-cp0-r1-runtime-bundle-closure-input-v1",
        }

    def _pin(self, path):
        raw = path.read_bytes()
        return {
            "bytes": len(raw),
            "path": str(path),
            "sha256": verify.sha256_bytes(raw),
            "stat": verify.stat_record(path.stat(follow_symlinks=False)),
        }

    def _value(self):
        return {
            "controller_host": verify.CONTROLLER_HOST,
            "directories": self.directories,
            "files": {
                name: self._pin(path)
                for name, path in sorted(self.files.items())
            },
            "model_id": verify.MODEL_ID,
            "phase": verify.PHASE,
            "ports": {
                "adb_server": 5038,
                "cuda_monolithic": 39124,
                "cuda_route": 39125,
                "op12_stage": 39126,
                "op15_stage": 39127,
                "relay": 39128,
                "relay_tail_source": 39129,
            },
            "route_epoch": 1,
            "schema": verify.OPERATOR_INPUT_SCHEMA,
            "topology": {
                "adb_host": "127.0.0.1",
                "adb_port": 5038,
                "cuda_uuid": verify.CUDA_UUID,
                "physical_serials": {
                    "op12": "5ae7a43d",
                    "op15": "3C15AU002CL00000",
                },
            },
        }

    def rewrite_json(self, name, mutate):
        value = json.loads(self.files[name].read_bytes())
        mutate(value)
        self._json(name, value)
        self.value["files"][name] = self._pin(self.files[name])


class TopologyTests(unittest.TestCase):
    def setUp(self):
        value = contract()
        self.contract_digest = mock.patch.object(
            verify,
            "CONTRACT_SHA256",
            verify.sha256_bytes(verify.canonical_bytes(value)),
        )
        self.contract_digest.start()
        self.addCleanup(self.contract_digest.stop)

    def capture(self, *, wifi=True):
        return verify.capture_topology(
            contract(),
            runner=FakeRunner(runner_rows(wifi=wifi)),
            clock_ns=Counter(),
        )

    def test_usb_and_wifi_topology_passes(self):
        value = self.capture()
        observed = verify.validate_topology(value, contract())
        self.assertEqual(observed["cuda"]["uuid"], verify.CUDA_UUID)
        self.assertEqual(
            observed["phones"]["op15"]["wifi_selector"],
            "172.20.173.218:5555",
        )

    def test_usb_only_topology_passes(self):
        value = self.capture(wifi=False)
        self.assertIsNone(value["observed"]["phones"]["op12"]["wifi_selector"])
        self.assertEqual(len(value["observed"]["connectivity"]), 6)

    def test_connectivity_packet_loss_is_rejected(self):
        fake = FakeRunner(runner_rows(wifi=False))
        command = verify.desktop_ping_command("172.20.59.72")
        key = tuple(command)
        fake.rows[key] = completed(
            command,
            fake.rows[key].stdout.replace(
                b"3 packets transmitted, 3 received, 0% packet loss",
                b"3 packets transmitted, 2 received, 33% packet loss",
            ),
        )
        with self.assertRaisesRegex(
            verify.DeploymentError,
            "CONNECTIVITY_OUTPUT",
        ):
            verify.capture_topology(
                contract(),
                runner=fake,
                clock_ns=Counter(),
            )

    def test_connectivity_single_trailing_newline_is_rejected(self):
        fake = FakeRunner(runner_rows(wifi=False))
        command = verify.desktop_ping_command("172.20.59.72")
        key = tuple(command)
        fake.rows[key] = completed(
            command,
            fake.rows[key].stdout[:-1],
        )
        with self.assertRaisesRegex(
            verify.DeploymentError,
            "CONNECTIVITY_OUTPUT",
        ):
            verify.capture_topology(
                contract(),
                runner=fake,
                clock_ns=Counter(),
            )

    def test_connectivity_edge_omission_is_rejected(self):
        value = self.capture(wifi=False)
        value["captures"]["connectivity"]["edges"].pop()
        with self.assertRaisesRegex(
            verify.DeploymentError,
            "CONNECTIVITY_EDGES",
        ):
            verify.validate_topology(value, contract())

    def test_missing_usb_is_rejected(self):
        rows = runner_rows()
        rows[4] = (
            verify.ADB_DEVICES_COMMAND,
            rows[4][1].replace(
                b"5ae7a43d device usb:6-2 product:CPH2583 "
                b"model:CPH2583 device:OP595DL1 transport_id:6\n",
                b"",
            ),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "usb_selectors"):
            verify.capture_topology(
                contract(),
                runner=FakeRunner(rows),
                clock_ns=Counter(),
            )

    def test_one_wifi_alias_is_rejected(self):
        rows = runner_rows()
        rows[4] = (
            verify.ADB_DEVICES_COMMAND,
            rows[4][1].replace(
                b"172.20.59.72:5555 device product:CPH2583 "
                b"model:CPH2583 device:OP595DL1 transport_id:8\n",
                b"",
            ),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "WIFI_COUNT"):
            verify.capture_topology(
                contract(),
                runner=FakeRunner(rows),
                clock_ns=Counter(),
            )

    def test_wifi_identity_mismatch_is_rejected(self):
        rows = runner_rows()
        selector = "172.20.173.218:5555"
        key = tuple(verify.phone_command(selector))
        fake = FakeRunner(rows)
        fake.rows[key] = completed(
            list(key),
            phone_output(
                "3C15AU002CL00000",
                "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "172.20.173.218",
            ),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "USB_WIFI_IDENTITY"):
            verify.capture_topology(
                contract(),
                runner=fake,
                clock_ns=Counter(),
            )

    def test_wifi_row_identity_mismatch_is_rejected(self):
        rows = runner_rows()
        rows[4] = (
            verify.ADB_DEVICES_COMMAND,
            rows[4][1].replace(
                b"172.20.173.218:5555 device product:CPH2749 ",
                b"172.20.173.218:5555 device product:WRONG ",
            ),
        )
        with self.assertRaisesRegex(
            verify.DeploymentError,
            "ADB_SHELL_IDENTITY",
        ):
            verify.capture_topology(
                contract(),
                runner=FakeRunner(rows),
                clock_ns=Counter(),
            )

    def test_usb_selector_serial_mismatch_is_rejected(self):
        fake = FakeRunner(runner_rows(wifi=False))
        selector = "3C15AU002CL00000"
        key = tuple(verify.phone_command(selector))
        fake.rows[key] = completed(
            list(key),
            phone_output(
                "5ae7a43d",
                "2ca4b7a3-c9c1-4614-a0d2-5746c57c8c4d",
                "172.20.59.72",
            ),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "USB_SERIAL"):
            verify.capture_topology(
                contract(),
                runner=fake,
                clock_ns=Counter(),
            )

    def test_self_ssh_identity_mismatch_is_rejected(self):
        fake = FakeRunner(runner_rows())
        key = tuple(verify.SELF_SSH_COMMAND)
        fake.rows[key] = completed(
            list(key),
            fake.rows[key].stdout.replace(
                b"2f68fcf5-54e1-4306-ba25-63359be572c2",
                b"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            ),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "SELF_SSH_CONTROLLER"):
            verify.capture_topology(
                contract(),
                runner=fake,
                clock_ns=Counter(),
            )

    def test_self_ssh_uses_one_remote_command_argument(self):
        self.assertNotIn("sh", verify.SELF_SSH_COMMAND[-3:])
        self.assertNotIn("-c", verify.SELF_SSH_COMMAND[-3:])
        self.assertEqual(verify.SELF_SSH_COMMAND[-2], verify.CUDA_SSH_TARGET)
        self.assertIn("/usr/bin/nvidia-smi", verify.SELF_SSH_COMMAND[-1])

    def test_wrong_gpu_is_rejected(self):
        rows = runner_rows()
        rows[1] = (
            verify.GPU_COMMAND,
            rows[1][1].replace(verify.CUDA_UUID.encode(), b"GPU-wrong"),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "gpu.uuid"):
            verify.capture_topology(
                contract(),
                runner=FakeRunner(rows),
                clock_ns=Counter(),
            )

    def test_stored_observation_is_recomputed(self):
        value = self.capture()
        value["observed"]["phones"]["op12"]["wifi_ipv4"] = "192.0.2.1"
        with self.assertRaisesRegex(verify.DeploymentError, "topology.observed"):
            verify.validate_topology(value, contract())

    def test_capture_interval_is_enforced(self):
        value = self.capture()
        value["captures"]["gpu"]["completed_ns"] = value["completed_ns"] + 1
        with self.assertRaisesRegex(verify.DeploymentError, "TOPOLOGY_INTERVAL"):
            verify.validate_topology(value, contract())

    def test_capture_overlap_is_rejected(self):
        value = self.capture()
        value["captures"]["gpu"]["started_ns"] = (
            value["captures"]["controller_before"]["completed_ns"] - 1
        )
        with self.assertRaisesRegex(verify.DeploymentError, "TOPOLOGY_INTERVAL"):
            verify.validate_topology(value, contract())

    def test_contract_digest_is_enforced(self):
        value = contract()
        with mock.patch.object(verify, "CONTRACT_SHA256", "f" * 64):
            with self.assertRaisesRegex(
                verify.DeploymentError,
                "contract.sha256",
            ):
                verify.capture_topology(
                    value,
                    runner=FakeRunner(runner_rows()),
                    clock_ns=Counter(),
                )

    def test_cli_rejects_stale_topology(self):
        value = self.capture(wifi=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_path = root / "contract.json"
            topology_path = root / "topology.json"
            contract_path.write_bytes(verify.canonical_bytes(contract()))
            topology_path.write_bytes(verify.canonical_bytes(value))
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                result = verify.main(
                    [
                        "validate-topology",
                        "--contract",
                        str(contract_path),
                        "--input",
                        str(topology_path),
                    ]
                )
        self.assertEqual(result, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("E_TOPOLOGY_STALE", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_accepts_fresh_topology(self):
        value = self.capture(wifi=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_path = root / "contract.json"
            topology_path = root / "topology.json"
            contract_path.write_bytes(verify.canonical_bytes(contract()))
            topology_path.write_bytes(verify.canonical_bytes(value))
            with mock.patch.object(
                verify,
                "monotonic_raw_ns",
                return_value=value["completed_ns"] + 1,
            ):
                stdout = io.StringIO()
                stderr = io.StringIO()
                with (
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    result = verify.main(
                        [
                            "validate-topology",
                            "--contract",
                            str(contract_path),
                            "--input",
                            str(topology_path),
                        ]
                    )
        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue(), "V24_NO_MODEL_TOPOLOGY_PASS\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_cli_labels_stale_replay_without_pass(self):
        value = self.capture(wifi=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_path = root / "contract.json"
            topology_path = root / "topology.json"
            contract_path.write_bytes(verify.canonical_bytes(contract()))
            topology_path.write_bytes(verify.canonical_bytes(value))
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                result = verify.main(
                    [
                        "validate-topology-replay",
                        "--contract",
                        str(contract_path),
                        "--input",
                        str(topology_path),
                    ]
                )
        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue(), "V24_TOPOLOGY_REPLAY_VALID\n")
        self.assertNotIn("PASS", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")


class OperatorInputTests(unittest.TestCase):
    def test_valid_input_passes(self):
        fixture = OperatorFixture(self)
        result = verify.validate_operator_input(fixture.value)
        self.assertEqual(result["status"], verify.OPERATOR_INPUT_STATUS)

    def test_model_pin_mutation_is_rejected(self):
        fixture = OperatorFixture(self)
        fixture.files["model"].write_bytes(b"changed-model\n")
        with self.assertRaisesRegex(verify.DeploymentError, "files.model"):
            verify.validate_operator_input(fixture.value)

    def test_candidate_model_binding_is_enforced(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "candidate",
            lambda value: value["models"][0]["artifact"].update(
                {"sha256": "f" * 64}
            ),
        )
        contract_value = json.loads(fixture.files["contract"].read_bytes())
        contract_value["candidate_lock"].update(
            {
                "bytes": fixture.files["candidate"].stat().st_size,
                "sha256": verify.sha256_bytes(
                    fixture.files["candidate"].read_bytes()
                ),
            }
        )
        fixture._json("contract", contract_value)
        fixture.value["files"]["contract"] = fixture._pin(
            fixture.files["contract"]
        )
        verify.CONTRACT_SHA256 = verify.sha256_bytes(
            fixture.files["contract"].read_bytes()
        )
        with self.assertRaisesRegex(verify.DeploymentError, "model.sha256"):
            verify.validate_operator_input(fixture.value)

    def test_history_geometry_is_enforced(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "token_history",
            lambda value: value.update({"n_ctx_seq": 256}),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "history.n_ctx_seq"):
            verify.validate_operator_input(fixture.value)

    def test_corpus_binding_is_enforced(self):
        fixture = OperatorFixture(self)
        fixture.files["quality_corpus"].write_bytes(b"different\n")
        fixture.value["files"]["quality_corpus"] = fixture._pin(
            fixture.files["quality_corpus"]
        )
        with self.assertRaisesRegex(verify.DeploymentError, "quality_corpus"):
            verify.validate_operator_input(fixture.value)

    def test_port_reuse_is_rejected(self):
        fixture = OperatorFixture(self)
        fixture.value["ports"]["cuda_route"] = 39124
        with self.assertRaisesRegex(verify.DeploymentError, "PORT_REUSE"):
            verify.validate_operator_input(fixture.value)

    def test_skeletal_runtime_inventory_is_rejected(self):
        fixture = OperatorFixture(self)
        fixture._json(
            "runtime_bundle_inventory",
            {"schema": "s39-cp0-r1-runtime-bundle-closure-input-v1"},
        )
        fixture.value["files"]["runtime_bundle_inventory"] = fixture._pin(
            fixture.files["runtime_bundle_inventory"]
        )
        with self.assertRaisesRegex(verify.DeploymentError, "runtime_inventory"):
            verify.validate_operator_input(fixture.value)

    def test_direct_relay_with_unused_library_is_rejected(self):
        fixture = OperatorFixture(self)

        def add_unused_library(value):
            relay = next(
                item
                for item in value["bundles"]
                if item["bundle_id"] == "op15_direct_relay"
            )
            component = fixture._component(
                "op15_direct_relay",
                "op15",
                "op15.relay.unused",
                Path(value["bundle_roots"]["op15_direct_relay"])
                / "unused.so",
                "shared_library",
            )
            value["components"].append(component)
            relay["required_component_ids"].append(component["component_id"])
            relay["required_component_ids"].sort()

        fixture.rewrite_json("runtime_bundle_inventory", add_unused_library)
        with self.assertRaisesRegex(
            verify.DeploymentError,
            "E_RUNTIME_RELAY_CLOSURE",
        ):
            verify.validate_operator_input(fixture.value)

    def test_false_runtime_closure_is_rejected(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "runtime_bundle_inventory",
            lambda value: value.update({"closure_complete": False}),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "closure_complete"):
            verify.validate_operator_input(fixture.value)

    def test_wrong_runtime_process_role_is_rejected(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "runtime_bundle_inventory",
            lambda value: value["bundles"][0].update(
                {"process_role": "other"}
            ),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "process_role"):
            verify.validate_operator_input(fixture.value)

    def test_runtime_mixed_component_id_type_is_rejected(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "runtime_bundle_inventory",
            lambda value: value["bundles"][0][
                "required_component_ids"
            ].append(1),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "component_ids"):
            verify.validate_operator_input(fixture.value)

    def test_runtime_list_role_is_rejected(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "runtime_bundle_inventory",
            lambda value: value["components"][0].update({"role": []}),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "role"):
            verify.validate_operator_input(fixture.value)

    def test_runtime_launcher_substitution_is_rejected(self):
        fixture = OperatorFixture(self)

        def mutate(value):
            bundle = next(
                item
                for item in value["bundles"]
                if item["bundle_id"] == "cuda_route"
            )
            launcher_id = bundle["launcher_component_id"]
            component = next(
                item
                for item in value["components"]
                if item["component_id"] == launcher_id
            )
            replacement = (
                Path(value["bundle_roots"]["cuda_route"]) / "other-worker"
            )
            replacement.write_bytes(b"other\n")
            replacement.chmod(0o755)
            component["path"] = str(replacement)
            component["bytes"] = replacement.stat().st_size
            component["sha256"] = verify.sha256_bytes(
                replacement.read_bytes()
            )
            component["stat"] = verify.stat_record(
                replacement.stat(follow_symlinks=False)
            )

        fixture.rewrite_json("runtime_bundle_inventory", mutate)
        with self.assertRaisesRegex(verify.DeploymentError, "launcher.path"):
            verify.validate_operator_input(fixture.value)

    def test_skeletal_monolithic_launch_is_rejected(self):
        fixture = OperatorFixture(self)
        fixture._json(
            "cuda_monolithic_launch",
            {
                "model_sha256": fixture.model_sha256,
                "port": verify.CUDA_MONOLITHIC_PORT,
                "schema": "s39-cp0-r1-v24-cuda-monolithic-launch-v1",
            },
        )
        fixture.value["files"]["cuda_monolithic_launch"] = fixture._pin(
            fixture.files["cuda_monolithic_launch"]
        )
        with self.assertRaisesRegex(
            verify.DeploymentError,
            "cuda_monolithic_launch",
        ):
            verify.validate_operator_input(fixture.value)

    def test_monolithic_command_geometry_is_rejected(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "cuda_monolithic_launch",
            lambda value: value["command"].__setitem__(
                value["command"].index("--driver-batch") + 1,
                "1",
            ),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "driver-batch"):
            verify.validate_operator_input(fixture.value)

    def test_monolithic_extra_command_argument_is_rejected(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "cuda_monolithic_launch",
            lambda value: value["command"].append("--no-kv-offload"),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "command"):
            verify.validate_operator_input(fixture.value)

    def test_monolithic_extra_environment_is_rejected(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "cuda_monolithic_launch",
            lambda value: value["env"].update({"LD_PRELOAD": "/tmp/other.so"}),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "env"):
            verify.validate_operator_input(fixture.value)

    def test_monolithic_bundle_digest_is_recomputed(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "cuda_monolithic_launch",
            lambda value: value.update({"bundle_sha256": "f" * 64}),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "bundle_sha256"):
            verify.validate_operator_input(fixture.value)

    def test_monolithic_timeout_is_exact(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "cuda_monolithic_launch",
            lambda value: value.update({"io_timeout_ms": False}),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "io_timeout_ms"):
            verify.validate_operator_input(fixture.value)

    def test_history_out_of_vocab_token_is_rejected(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "token_history",
            lambda value: value["requests"][0].update(
                {"token_ids": [151936]}
            ),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "TOKEN_RANGE"):
            verify.validate_operator_input(fixture.value)

    def test_history_group_is_independently_derived(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "token_history",
            lambda value: value["quality_groups"][0].update(
                {"item_indices": list(reversed(range(8)))}
            ),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "quality_groups"):
            verify.validate_operator_input(fixture.value)

    def test_history_tokenizer_identity_is_bound(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "token_history",
            lambda value: value["tokenizer"].update(
                {"component_id": "other", "path": "/tmp/other"}
            ),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "component_id"):
            verify.validate_operator_input(fixture.value)

    def test_incomplete_token_history_is_rejected_without_traceback(self):
        fixture = OperatorFixture(self)
        fixture._json(
            "token_history",
            {"schema": "s39-cp0-r1-token-history-v2.4"},
        )
        fixture.value["files"]["token_history"] = fixture._pin(
            fixture.files["token_history"]
        )
        with self.assertRaisesRegex(verify.DeploymentError, "token_history"):
            verify.validate_operator_input(fixture.value)

    def test_topology_receipt_is_recomputed(self):
        fixture = OperatorFixture(self)
        fixture.rewrite_json(
            "topology_receipt",
            lambda value: value["observed"]["phones"]["op12"].update(
                {"wifi_ipv4": "192.0.2.1"}
            ),
        )
        with self.assertRaisesRegex(verify.DeploymentError, "topology.observed"):
            verify.validate_operator_input(fixture.value)

    def test_symlink_is_rejected(self):
        fixture = OperatorFixture(self)
        target = fixture.files["adb"]
        link = fixture.root / "adb-link"
        link.symlink_to(target)
        fixture.value["files"]["adb"]["path"] = str(link)
        with self.assertRaisesRegex(verify.DeploymentError, "SYMLINK"):
            verify.validate_operator_input(fixture.value)

    def test_reopen_mutation_is_rejected(self):
        fixture = OperatorFixture(self)

        def mutate(name):
            if name == "model":
                os.utime(fixture.files["model"], None)

        with self.assertRaisesRegex(
            verify.DeploymentError,
            "FILE_REOPEN_MUTATION",
        ):
            verify.validate_operator_input(
                fixture.value,
                after_file_read=mutate,
            )

    def test_large_file_hashing_does_not_retain_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "large.bin"
            with path.open("wb") as stream:
                stream.truncate(verify.MAX_RETAINED_FILE_BYTES + 1)
            digest_value, size, _, raw = verify.secure_hash(
                path,
                "large",
                retain=False,
            )
            self.assertEqual(len(digest_value), 64)
            self.assertEqual(size, verify.MAX_RETAINED_FILE_BYTES + 1)
            self.assertIsNone(raw)

    def test_json_float_is_rejected(self):
        with self.assertRaisesRegex(verify.DeploymentError, "JSON_FLOAT"):
            verify.parse_json(b'{"value":1.0}\n', "float")

    def test_duplicate_json_key_is_rejected(self):
        with self.assertRaisesRegex(verify.DeploymentError, "DUPLICATE_KEY"):
            verify.parse_json(b'{"value":1,"value":2}\n', "duplicate")

    def test_cli_refusal_has_no_traceback(self):
        fixture = OperatorFixture(self)
        fixture._json(
            "token_history",
            {"schema": "s39-cp0-r1-token-history-v2.4"},
        )
        fixture.value["files"]["token_history"] = fixture._pin(
            fixture.files["token_history"]
        )
        input_path = fixture.root / "operator-input.json"
        input_path.write_bytes(verify.canonical_bytes(fixture.value))
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = verify.main(
                ["validate-operator-input", "--input", str(input_path)]
            )
        self.assertEqual(result, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("V24_DESKTOP_DEPLOYMENT_REFUSED", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_malformed_nested_inputs_refuse_without_traceback(self):
        mutations = (
            (
                "tokenizer_plan",
                lambda value: value.pop("model"),
            ),
            (
                "runtime_bundle_inventory",
                lambda value: value["components"][0].update({"role": []}),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                fixture = OperatorFixture(self)
                fixture.rewrite_json(name, mutate)
                input_path = fixture.root / "operator-input.json"
                input_path.write_bytes(verify.canonical_bytes(fixture.value))
                stdout = io.StringIO()
                stderr = io.StringIO()
                with (
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    result = verify.main(
                        [
                            "validate-operator-input",
                            "--input",
                            str(input_path),
                        ]
                    )
                self.assertEqual(result, 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn(
                    "V24_DESKTOP_DEPLOYMENT_REFUSED",
                    stderr.getvalue(),
                )
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_occupied_cuda_port_is_rejected(self):
        fixture = OperatorFixture(self)
        with self.assertRaisesRegex(verify.DeploymentError, "cuda_route"):
            verify.validate_operator_input(
                fixture.value,
                port_available=lambda port: (
                    port != fixture.value["ports"]["cuda_route"]
                ),
            )


if __name__ == "__main__":
    unittest.main()
