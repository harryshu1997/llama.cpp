#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import datetime
import copy
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXECUTORS = HERE.parent
if str(EXECUTORS) not in sys.path:
    sys.path.insert(0, str(EXECUTORS))

from a6000_phone_route_control import (
    PhoneControl,
    argv_sha256,
    load_config,
)
from phone_gateway import canonical_bytes, strict_json_loads
from readiness_v23 import parse_artifact_certificate, parse_readiness_lock


def stat_record(seed: int, size: int) -> dict[str, int]:
    return {
        "ctime_ns": 1_700_000_000_000_000_000,
        "device_id": 100 + seed,
        "inode": 200 + seed,
        "mode": 0o100444,
        "mtime_ns": 1_600_000_000_000_000_000,
        "size": size,
    }


class FakePhoneControl(PhoneControl):
    def __init__(self, config):
        super().__init__(config)
        self.commands = []
        self.alive = {}
        self.next_pid = {"op15": 1001, "op12": 2001}
        self.pid_file = {}
        self.pid_process = {}
        self.tamper_stat = False
        self.tamper_pid = False
        self.tamper_runtime_digest = False

    def wait_for_process_exit(self, serial, pid, timeout_s):
        del timeout_s
        self.alive[(serial, pid)] = False
        return True

    def synchronous_observation(self, state):
        return {
            "direct_peer": {},
            "model_id": state["model_id"],
            "phones": {"op12": {}, "op15": {}},
            "route_instance_id": state["route_instance_id"],
            "schema": "s40-phone-route-observation-v1",
        }

    def adb(self, serial, *arguments, check=True, timeout=60.0):
        del timeout
        self.commands.append((serial, arguments))
        stdout = b""
        returncode = 0
        if arguments == ("get-state",):
            stdout = b"device\n"
        elif arguments[0] == "shell":
            command = arguments[1]
            if command.startswith("mkdir "):
                pass
            elif command.startswith("sh -c "):
                phone_name = "op15" if serial == "serial15" else "op12"
                pid = self.next_pid[phone_name]
                self.next_pid[phone_name] += 1
                self.alive[(serial, pid)] = True
                self.pid_file[serial] = pid
                matches = [
                    process
                    for process in self.config["routes"]["model-a"][
                        phone_name
                    ]["processes"]
                    if process["argv"][0] in command
                ]
                if len(matches) != 1:
                    raise AssertionError(command)
                process = copy.deepcopy(matches[0])
                log_match = re.search(
                    r"(/[A-Za-z0-9._/-]+/(?:stage|relay)\.log)",
                    command,
                )
                if log_match is None:
                    raise AssertionError(command)
                process["log"] = log_match.group(1)
                self.pid_process[(serial, pid)] = process
            elif command.startswith("grep -Fq "):
                pass
            elif command.startswith("kill -0 "):
                pid = int(command.split()[-1])
                returncode = 0 if self.alive.get((serial, pid), False) else 1
            elif command.startswith("kill "):
                pid = int(command.split()[-1])
                if self.alive.get((serial, pid), False):
                    self.alive[(serial, pid)] = False
                else:
                    returncode = 1
            else:
                raise AssertionError(command)
        else:
            raise AssertionError(arguments)
        result = subprocess.CompletedProcess(
            args=[],
            returncode=returncode,
            stdout=stdout,
            stderr=b"",
        )
        if check and returncode != 0:
            raise AssertionError(arguments)
        return result

    def text(self, serial, command):
        self.commands.append((serial, ("shell-text", command)))
        route = self.config["routes"]["model-a"]
        phone_name = "op15" if serial == "serial15" else "op12"
        phone = route[phone_name]
        if command == "cat /proc/sys/kernel/random/boot_id":
            return phone["boot_id"]
        if command.startswith("stat -c "):
            path = command.rsplit(" ", 1)[-1]
            path = path.strip("'")
            runtime = [
                row for row in phone["runtime_files"].values()
                if row["path"] == path
            ]
            if runtime:
                record = runtime[0]["stat"].copy()
            else:
                endpoint = (
                    phone_name
                    if path == phone["shard_path"]
                    else f"{phone_name}_worker"
                )
                record = route["artifact_certificate"]["artifacts"][
                    (endpoint, path)
                ]["stat"].copy()
            if self.tamper_stat:
                record["mtime_ns"] += 1
            def stamp(value):
                seconds, fraction = divmod(value, 1_000_000_000)
                date = datetime.datetime.fromtimestamp(
                    seconds,
                    datetime.timezone.utc,
                )
                return date.strftime("%Y-%m-%d %H:%M:%S") + (
                    f".{fraction:09d} +0000"
                )
            return (
                f"{record['device_id']}|{record['inode']}|{record['size']}|"
                f"{record['mode']:x}|{stamp(record['mtime_ns'])}|"
                f"{stamp(record['ctime_ns'])}"
            )
        if command.startswith("cat ") and command.endswith(".pid"):
            return str(self.pid_file[serial])
        if command.startswith("sha256sum /proc/"):
            pid = int(command.split("/proc/")[1].split("/")[0])
            process = self.pid_process[(serial, pid)]
            digest = argv_sha256(process["argv"])
            if self.tamper_pid:
                digest = "f" * 64
            return f"{digest}  /proc/{pid}/cmdline"
        if command.startswith("sha256sum "):
            path = command.split(" ", 1)[1]
            rows = [
                row for row in phone["runtime_files"].values()
                if row["path"] == path
            ]
            self.assert_runtime_row = rows
            if len(rows) != 1:
                raise AssertionError(command)
            digest = (
                "f" * 64
                if self.tamper_runtime_digest
                else rows[0]["sha256"]
            )
            return f"{digest}  {path}"
        if command.startswith("cat /proc/") and command.endswith("/stat"):
            pid = int(command.split("/proc/")[1].split("/")[0])
            fields = ["S"] + ["0"] * 18 + [str(100000 + pid)] + ["0"] * 8
            return f"{pid} (worker) " + " ".join(fields)
        if command.startswith("tail -n 64 "):
            log_path = command[len("tail -n 64 "):].strip("'")
            candidates = [
                (pid, process)
                for (candidate_serial, pid), process
                in self.pid_process.items()
                if candidate_serial == serial
                and process["kind"] in ("STAGE_HEAD", "STAGE_TAIL")
                and process.get("log") == log_path
            ]
            if len(candidates) != 1:
                raise AssertionError(command)
            pid, process = candidates[0]
            backend = process["backend"]
            compute = {"MUL_MAT": {backend: 1}}
            n_layer = self.config["routes"]["model-a"]["op12"][
                "processes"
            ][0]["layer_end"]
            session = {
                "compute_by_op_and_buffer": compute,
                "device_boot_id": phone["boot_id"],
                "expected_backend": backend,
                "layer_end": process["layer_end"],
                "layer_start": process["layer_start"],
                "missing_buffer_compute_nodes": 0,
                "n_layer": n_layer,
                "placement_status": "SCHEDULED_PLACEMENT_OK",
                "proto_version": 2,
                "reset_applied": False,
                "schema": "ls-stagenet-session-v2",
                "session_end": "EOF",
                "session_id": 1,
                "steps_session": 1,
                "steps_total": 1,
                "worker_boot_nonce": "nonce",
                "worker_pid": pid,
            }
            placement = {
                "compute_by_buffer_type": {backend: 1},
                "compute_by_op": {"MUL_MAT": 1},
                "compute_by_op_and_buffer": compute,
                "compute_nodes": 1,
                "copy_by_buffer_type": {},
                "copy_nodes": 0,
                "layer_end": process["layer_end"],
                "layer_start": process["layer_start"],
                "metadata_nodes": 0,
                "missing_buffer_compute_nodes": 0,
                "mode": (
                    "stagenet"
                    if process["kind"] == "STAGE_HEAD"
                    else "tailv3"
                ),
                "n_layer": n_layer,
                "pid": pid,
                "role": "phone_stage",
                "run_rc": 0,
                "schema": "layersplit-scheduled-placement-v2",
                "status": "SCHEDULED_PLACEMENT_OK",
            }
            return (
                "SESSIONCERT "
                + canonical_bytes(session).decode("ascii")
                + "PLACEMENTCERT "
                + canonical_bytes(placement).decode("ascii")
            ).strip()
        raise AssertionError(command)


class PersistentFakePhoneControl(FakePhoneControl):
    def __init__(self, config, remote_state, crash_at=None):
        super().__init__(config)
        self.remote_state = remote_state
        self.crash_at = crash_at
        if not remote_state.exists():
            self._save()

    def _load(self):
        value = strict_json_loads(
            self.remote_state.read_bytes(),
            "fake remote state",
        )
        self.alive = {
            (row[0], row[1]): True for row in value["alive"]
        }
        self.pid_file = {
            serial: pid for serial, pid in value["pid_file"].items()
        }
        self.pid_process = {
            (row["serial"], row["pid"]): row["process"]
            for row in value["pid_process"]
        }
        self.next_pid = value["next_pid"]

    def _save(self):
        value = {
            "alive": [
                [serial, pid]
                for (serial, pid), alive in sorted(self.alive.items())
                if alive
            ],
            "next_pid": self.next_pid,
            "pid_file": self.pid_file,
            "pid_process": [
                {
                    "pid": pid,
                    "process": process,
                    "serial": serial,
                }
                for (serial, pid), process in sorted(
                    self.pid_process.items()
                )
            ],
        }
        temporary = self.remote_state.with_suffix(".tmp")
        temporary.write_bytes(canonical_bytes(value))
        os.replace(temporary, self.remote_state)

    def adb(self, serial, *arguments, check=True, timeout=60.0):
        self._load()
        if (
            arguments[0] == "shell"
            and arguments[1].startswith("cat ")
            and arguments[1].endswith(".pid")
        ):
            pid = self.pid_file.get(serial)
            result = subprocess.CompletedProcess(
                args=[],
                returncode=0 if pid is not None else 1,
                stdout=(b"" if pid is None else f"{pid}\n".encode("ascii")),
                stderr=b"",
            )
            if check and result.returncode != 0:
                raise AssertionError(arguments)
            return result
        result = super().adb(
            serial,
            *arguments,
            check=check,
            timeout=timeout,
        )
        self._save()
        return result

    def text(self, serial, command):
        self._load()
        result = super().text(serial, command)
        self._save()
        return result

    def wait_for_process_exit(self, serial, pid, timeout_s):
        self._load()
        result = super().wait_for_process_exit(serial, pid, timeout_s)
        self._save()
        return result

    def checkpoint(self, name):
        if name == self.crash_at:
            os.kill(os.getpid(), signal.SIGKILL)


class A6000PhoneRouteControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        identity = self.root / "identity"
        identity.write_text("a6000\n", encoding="ascii")
        adb = self.root / "adb"
        adb.write_text("adb\n", encoding="ascii")
        adb.chmod(0o755)
        self.worker_sha = "3" * 64
        self.shard_sha = {"op15": "1" * 64, "op12": "2" * 64}
        self.stats = {
            ("op15", "/op15/weights.gguf"): stat_record(1, 6000),
            ("op12", "/op12/weights.gguf"): stat_record(2, 4000),
            (
                "op15_worker",
                "/op15/runtime/llama-layersplit",
            ): stat_record(3, 1000),
            (
                "op12_worker",
                "/op12/runtime/llama-layersplit",
            ): stat_record(4, 1000),
            ("cuda", "/desktop/model.gguf"): stat_record(5, 9000),
        }
        artifacts = []
        for (endpoint, path), stat in self.stats.items():
            digest = (
                self.worker_sha
                if endpoint.endswith("_worker")
                else (
                    self.shard_sha[endpoint]
                    if endpoint in self.shard_sha
                    else "a" * 64
                )
            )
            artifacts.append({
                "bytes": stat["size"],
                "endpoint": endpoint,
                "path": path,
                "sha256": digest,
                "stat": stat,
            })
        certificate = {
            "artifacts": artifacts,
            "completed_ns": 2,
            "model_id": "model-a",
            "phase": "A_ONLY",
            "route_lock_sha256": "b" * 64,
            "schema": "s39-cp0-r1-artifact-snapshot-v2.3",
            "slot": "A",
            "started_ns": 1,
        }
        self.certificate = certificate
        certificate_path = self.root / "certificate.json"
        certificate_path.write_bytes(canonical_bytes(certificate))
        readiness_lock = {
            "artifact_snapshot_sha256": hashlib.sha256(
                certificate_path.read_bytes()
            ).hexdigest(),
            "event_ns": 3,
            "phase": "A_ONLY",
            "phase_id": "phase-a",
            "schema": "s39-cp0-r1-readiness-lock-v2.3",
            "v2_2_phase_lock_sha256": "c" * 64,
        }
        readiness_lock_path = self.root / "readiness-lock.json"
        readiness_lock_path.write_bytes(canonical_bytes(readiness_lock))
        qualification = {
            "a_chain": None,
            "current": {
                "artifact_snapshot": {
                    "path": str(certificate_path),
                    "sha256": hashlib.sha256(
                        certificate_path.read_bytes()
                    ).hexdigest(),
                },
                "bundle_manifest_sha256": "d" * 64,
                "bundle_root": str(self.root),
                "fresh_snapshot": {
                    "path": str(self.root / "fresh.json"),
                    "sha256": "e" * 64,
                },
                "readiness_lock": {
                    "path": str(readiness_lock_path),
                    "sha256": hashlib.sha256(
                        readiness_lock_path.read_bytes()
                    ).hexdigest(),
                },
                "runtime_identity": {
                    "path": str(self.root / "runtime.json"),
                    "sha256": "f" * 64,
                },
            },
            "model_id": "model-a",
            "phase": "A_ONLY",
            "schema": "s40-route-qualification-authority-v1",
            "slot": "A",
        }
        routes = {}
        route = {
            "artifact_certificate_path": str(certificate_path),
            "artifact_certificate_sha256": hashlib.sha256(
                certificate_path.read_bytes()
            ).hexdigest(),
            "model_sha256": "a" * 64,
            "phase": "A_ONLY",
            "phase_lock_sha256": "c" * 64,
            "qualification": qualification,
            "readiness_lock_path": str(readiness_lock_path),
            "readiness_lock_sha256": hashlib.sha256(
                readiness_lock_path.read_bytes()
            ).hexdigest(),
            "route_lock_sha256": "b" * 64,
            "slot": "A",
            "worker_sha256": self.worker_sha,
            "direct_peer": {
                "op12_address": "192.168.1.12",
                "op12_port": 41000,
                "op15_address": "192.168.1.15",
                "op15_port": 5000,
                "schema": "s40-phone-direct-peer-v1",
            },
        }
        for phone_name, serial in (("op15", "serial15"), ("op12", "serial12")):
            runtime_root = f"/{phone_name}/runtime"
            worker_path = f"{runtime_root}/llama-layersplit"
            shard_path = f"/{phone_name}/weights.gguf"
            runtime_names = {
                "libcxx_shared": "libc++_shared.so",
                "libggml": "libggml.so",
                "libggml_base": "libggml-base.so",
                "libggml_cpu": "libggml-cpu.so",
                "libggml_hexagon": "libggml-hexagon.so",
                "libggml_opencl": "libggml-opencl.so",
                "libllama": "libllama.so",
                "libllama_common": "libllama-common.so",
                "worker": "llama-layersplit",
            }
            if phone_name == "op15":
                runtime_names["htp_skel_v81"] = "libggml-htp-v81.so"
                runtime_names["relay"] = "llama-stage-direct-relay"
            else:
                runtime_names["htp_skel_v75"] = "libggml-htp-v75.so"
            runtime_files = []
            for offset, (role, filename) in enumerate(
                sorted(runtime_names.items()),
                20 if phone_name == "op15" else 40,
            ):
                path = f"{runtime_root}/{filename}"
                stat = (
                    self.stats[(f"{phone_name}_worker", worker_path)]
                    if role == "worker"
                    else stat_record(offset, 1000 + offset)
                )
                digest = (
                    self.worker_sha
                    if role == "worker"
                    else f"{offset:064x}"
                )
                runtime_files.append({
                    "bytes": stat["size"],
                    "path": path,
                    "role": role,
                    "sha256": digest,
                    "stat": stat,
                })
            stage_port = 40000 if phone_name == "op15" else 41000
            stage_kind = "STAGE_HEAD" if phone_name == "op15" else "STAGE_TAIL"
            stage_mode = "stagenet" if phone_name == "op15" else "tailv3"
            stage_env = {
                "ADSP_LIBRARY_PATH": runtime_root,
                "LAYERSPLIT_MODEL_SHA256": "a" * 64,
                "LAYERSPLIT_PLACEMENT_CERT": "1",
                "LD_LIBRARY_PATH": runtime_root,
                "LLAMA_LAYER_END": "30" if phone_name == "op15" else "40",
                "LLAMA_LAYER_START": "0" if phone_name == "op15" else "30",
                "PATH": "/system/bin:/system/xbin",
            }
            processes = [{
                "argv": [
                    worker_path,
                    "-m",
                    shard_path,
                    "--mode",
                    stage_mode,
                    "--port",
                    str(stage_port),
                    "--driver-batch",
                    "8",
                    "--driver-context",
                    "64",
                    "--driver-max-prefill",
                    "8",
                    "--devices",
                    "GPUOpenCL",
                    "-ngl",
                    "99",
                ],
                "backend": "GPUOpenCL",
                "driver_batch": 8,
                "driver_context": 64,
                "driver_max_prefill": 8,
                "env": stage_env,
                "kind": stage_kind,
                "layer_end": 30 if phone_name == "op15" else 40,
                "layer_start": 0 if phone_name == "op15" else 30,
                "listen_port": stage_port,
                "name": "stage",
                "ready_marker":
                    f"[stagenet] listening on 0.0.0.0:{stage_port} (",
            }]
            if phone_name == "op15":
                relay_path = f"{runtime_root}/llama-stage-direct-relay"
                processes.append({
                    "argv": [
                        relay_path,
                        "--listen",
                        "42000",
                        "--head",
                        "127.0.0.1:40000",
                        "--tail",
                        "192.168.1.12:41000",
                        "--tail-source-port",
                        "5000",
                    ],
                    "env": {
                        "LD_LIBRARY_PATH": runtime_root,
                        "PATH": "/system/bin:/system/xbin",
                    },
                    "head_host": "127.0.0.1",
                    "head_port": 40000,
                    "kind": "DIRECT_RELAY",
                    "listen_port": 42000,
                    "name": "relay",
                    "ready_marker": (
                        "[direct-relay] listening on 0.0.0.0:42000 "
                        "head=127.0.0.1:40000 tail=192.168.1.12:41000"
                    ),
                    "tail_host": "192.168.1.12",
                    "tail_port": 41000,
                    "tail_source_port": 5000,
                })
            route[phone_name] = {
                "boot_id": f"{phone_name}-boot",
                "processes": processes,
                "runtime_files": runtime_files,
                "runtime_root": runtime_root,
                "serial": serial,
                "shard_path": shard_path,
                "shard_sha256": self.shard_sha[phone_name],
                "shard_size": 6000 if phone_name == "op15" else 4000,
                "worker_path": worker_path,
            }
        routes["model-a"] = route
        dependency_paths = {"adb": adb}
        for role in (
            "a6000_phone_observer",
            "a6000_phone_route_control",
            "phone_gateway",
            "python",
            "readiness_v23",
        ):
            dependency = self.root / role
            dependency.write_text(role + "\n", encoding="ascii")
            if role in (
                "a6000_phone_observer",
                "a6000_phone_route_control",
                "python",
            ):
                dependency.chmod(0o755)
            dependency_paths[role] = dependency
        dependency_files = [
            {
                "bytes": dependency.stat().st_size,
                "path": str(dependency),
                "role": role,
                "sha256": hashlib.sha256(
                    dependency.read_bytes()
                ).hexdigest(),
            }
            for role, dependency in sorted(dependency_paths.items())
        ]
        config = {
            "a6000_identity": hashlib.sha256(identity.read_bytes()).hexdigest(),
            "a6000_identity_path": str(identity),
            "adb_path": str(adb),
            "adb_port": 5038,
            "dependency_files": dependency_files,
            "host_boot_id": Path(
                "/proc/sys/kernel/random/boot_id"
            ).read_text(encoding="ascii").strip(),
            "routes": routes,
            "schema": "s40-a6000-phone-routes-v3",
            "state_dir": str(self.root / "state"),
        }
        config_path = self.root / "config.json"
        config_path.write_bytes(canonical_bytes(config))
        self.config = load_config(config_path)
        self.certificate_path = certificate_path
        self.raw_config = config
        self.raw_config_path = config_path

    def tearDown(self):
        self.temporary.cleanup()

    def load_mutation(self, mutate):
        value = copy.deepcopy(self.raw_config)
        mutate(value)
        path = self.root / f"mutation-{time.monotonic_ns()}.json"
        path.write_bytes(canonical_bytes(value))
        return load_config(path)

    def test_load_unload_uses_certified_stats_and_persists_evidence(self):
        control = FakePhoneControl(self.config)
        loaded = control.load("model-a")
        self.assertTrue(loaded["success"])
        self.assertFalse(
            any(
                "sha256sum /op15/weights.gguf" in argument
                or "sha256sum /op12/weights.gguf" in argument
                for _, command in control.commands
                for argument in command
            )
        )
        active = self.config["state_dir"] / "active.json"
        state = strict_json_loads(active.read_bytes(), "active")
        self.assertEqual(state["schema"], "s40-a6000-route-journal-v3")
        self.assertEqual(state["state"], "ACTIVE")
        unloaded = control.unload("model-a")
        self.assertEqual(
            unloaded["route_instance_id"],
            loaded["route_instance_id"],
        )
        self.assertEqual(
            set(unloaded["placements"]),
            {"op12", "op15"},
        )
        self.assertEqual(
            unloaded["placements"]["op15"]["placement"]["status"],
            "SCHEDULED_PLACEMENT_OK",
        )
        completed = (
            self.config["state_dir"]
            / f"completed-{loaded['route_instance_id']}.json"
        )
        record = strict_json_loads(completed.read_bytes(), "completed")
        self.assertEqual(record["schema"], "s40-a6000-completed-route-v3")
        self.assertEqual(record["release_kind"], "UNLOAD")
        self.assertFalse(active.exists())

    def test_phase_lock_cannot_be_replaced_by_route_lock(self):
        with self.assertRaisesRegex(RuntimeError, "phase root"):
            self.load_mutation(
                lambda value: value["routes"]["model-a"].__setitem__(
                    "phase_lock_sha256",
                    value["routes"]["model-a"]["route_lock_sha256"],
                )
            )

    def test_changed_artifact_stat_is_rejected_before_launch(self):
        control = FakePhoneControl(self.config)
        control.tamper_stat = True
        with self.assertRaisesRegex(Exception, "certified artifact stat changed"):
            control.load("model-a")
        self.assertFalse(control.alive)

    def test_runtime_digest_change_is_rejected_before_launch(self):
        control = FakePhoneControl(self.config)
        control.tamper_runtime_digest = True
        with self.assertRaisesRegex(Exception, "runtime digest changed"):
            control.load("model-a")
        self.assertFalse(control.alive)

    def test_process_argv_environment_and_chain_mutations_are_rejected(self):
        mutations = {
            "executable": lambda value: value["routes"]["model-a"]["op15"][
                "processes"
            ][0]["argv"].__setitem__(0, "/system/bin/true"),
            "model": lambda value: value["routes"]["model-a"]["op15"][
                "processes"
            ][0]["argv"].__setitem__(2, "/other/model.gguf"),
            "backend": lambda value: value["routes"]["model-a"]["op15"][
                "processes"
            ][0].__setitem__("backend", "HTP0"),
            "layer environment": lambda value: value["routes"]["model-a"][
                "op15"
            ]["processes"][0]["env"].__setitem__("LLAMA_LAYER_END", "29"),
            "extra flag": lambda value: value["routes"]["model-a"]["op15"][
                "processes"
            ][0]["argv"].append("--verbose"),
            "duplicate flag": lambda value: value["routes"]["model-a"]["op15"][
                "processes"
            ][0]["argv"].extend(["--port", "40000"]),
            "peer": lambda value: value["routes"]["model-a"]["op15"][
                "processes"
            ][1].__setitem__("tail_port", 41001),
            "chain": lambda value: value["routes"]["model-a"]["op12"][
                "processes"
            ][0].__setitem__("layer_start", 29),
            "ready marker": lambda value: value["routes"]["model-a"]["op12"][
                "processes"
            ][0].__setitem__("ready_marker", "READY"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(Exception):
                    self.load_mutation(mutate)

    def test_runtime_role_and_architecture_mutations_are_rejected(self):
        def remove_role(value):
            value["routes"]["model-a"]["op12"]["runtime_files"] = [
                row
                for row in value["routes"]["model-a"]["op12"][
                    "runtime_files"
                ]
                if row["role"] != "libggml_opencl"
            ]

        def wrong_skel(value):
            row = next(
                row
                for row in value["routes"]["model-a"]["op12"][
                    "runtime_files"
                ]
                if row["role"] == "htp_skel_v75"
            )
            row["path"] = "/op12/runtime/libggml-htp-v81.so"

        for name, mutate in (
            ("missing role", remove_role),
            ("wrong architecture skel", wrong_skel),
        ):
            with self.subTest(name=name):
                with self.assertRaises(Exception):
                    self.load_mutation(mutate)

    def test_reused_pid_is_not_killed(self):
        control = FakePhoneControl(self.config)
        control.load("model-a")
        control.tamper_pid = True
        with self.assertRaisesRegex(Exception, "reused phone PID"):
            control.unload("model-a")
        self.assertTrue(any(control.alive.values()))

    def _fork_crash(self, remote_state, event, action):
        pid = os.fork()
        if pid == 0:
            control = PersistentFakePhoneControl(
                self.config,
                remote_state,
                event,
            )
            if action == "load":
                control.load("model-a")
            else:
                control.unload("model-a")
            os._exit(70)
        waited, status = os.waitpid(pid, 0)
        self.assertEqual(waited, pid)
        self.assertTrue(os.WIFSIGNALED(status))
        self.assertEqual(os.WTERMSIG(status), signal.SIGKILL)

    def test_sigkill_restart_recovers_every_launch_boundary(self):
        for event in (
            "launch-planned:op12:stage",
            "launch-pid:op12:stage",
            "launch-ready:op12:stage",
        ):
            with self.subTest(event=event):
                state_dir = self.config["state_dir"]
                if state_dir.exists():
                    for path in state_dir.iterdir():
                        path.unlink()
                remote_state = self.root / (
                    "remote-" + event.replace(":", "-") + ".json"
                )
                self._fork_crash(remote_state, event, "load")
                journal = state_dir / "active.json"
                self.assertTrue(journal.is_file())
                recovered = PersistentFakePhoneControl(
                    self.config,
                    remote_state,
                )
                loaded = recovered.load("model-a")
                self.assertTrue(loaded["success"])
                recovered.unload("model-a")
                remote = strict_json_loads(
                    remote_state.read_bytes(),
                    "fake remote state",
                )
                self.assertEqual(remote["alive"], [])
                self.assertFalse(journal.exists())

    def test_sigkill_restart_recovers_every_release_boundary(self):
        for event in (
            "release-begin",
            "release-mark:op15:stage",
            "release-done:op15:stage",
            "release-all-done",
            "release-recorded",
        ):
            with self.subTest(event=event):
                state_dir = self.config["state_dir"]
                if state_dir.exists():
                    for path in state_dir.iterdir():
                        path.unlink()
                remote_state = self.root / (
                    "remote-" + event.replace(":", "-") + ".json"
                )
                control = PersistentFakePhoneControl(
                    self.config,
                    remote_state,
                )
                control.load("model-a")
                self._fork_crash(remote_state, event, "unload")
                recovered = PersistentFakePhoneControl(
                    self.config,
                    remote_state,
                )
                result = recovered.unload("model-a")
                self.assertTrue(result["success"])
                remote = strict_json_loads(
                    remote_state.read_bytes(),
                    "fake remote state",
                )
                self.assertEqual(remote["alive"], [])
                self.assertFalse((state_dir / "active.json").exists())

    def test_adb_is_always_pinned_to_port_5038(self):
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, b"device\n", b"")

        control = PhoneControl(self.config)
        original = subprocess.run
        try:
            subprocess.run = runner
            control.adb("serial15", "get-state")
        finally:
            subprocess.run = original
        self.assertEqual(calls[0][0][1:5], ["-P", "5038", "-s", "serial15"])
        self.assertEqual(
            calls[0][1]["env"],
            {
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/system/bin:/system/xbin",
            },
        )

    def test_dependency_mutation_during_adb_is_rejected(self):
        dependency = Path(
            next(
                row["path"]
                for row in self.raw_config["dependency_files"]
                if row["role"] == "phone_gateway"
            )
        )

        def runner(argv, **kwargs):
            del kwargs
            dependency.write_text("changed during ADB\n", encoding="ascii")
            return subprocess.CompletedProcess(argv, 0, b"device\n", b"")

        control = PhoneControl(self.config)
        original = subprocess.run
        try:
            subprocess.run = runner
            with self.assertRaisesRegex(Exception, "phone_gateway dependency"):
                control.adb("serial15", "get-state")
        finally:
            subprocess.run = original

    def test_non_experiment_adb_port_is_rejected(self):
        config_path = self.root / "config-wrong-port.json"
        value = copy.deepcopy(self.raw_config)
        value["adb_port"] = 5037
        value["state_dir"] = str(self.root / "wrong-port-state")
        config_path.write_bytes(canonical_bytes(value))
        with self.assertRaisesRegex(RuntimeError, "must be 5038"):
            load_config(config_path)

    def test_dependency_mutation_and_stale_boot_are_rejected(self):
        dependency = Path(
            next(
                row["path"]
                for row in self.raw_config["dependency_files"]
                if row["role"] == "readiness_v23"
            )
        )
        dependency.write_text("changed\n", encoding="ascii")
        with self.assertRaisesRegex(RuntimeError, "readiness_v23 dependency"):
            load_config(self.raw_config_path)

        dependency.write_text("readiness_v23\n", encoding="ascii")
        stale = copy.deepcopy(self.raw_config)
        stale["host_boot_id"] = "11111111-1111-4111-8111-111111111111"
        path = self.root / "stale-boot.json"
        path.write_bytes(canonical_bytes(stale))
        with self.assertRaisesRegex(RuntimeError, "host boot changed"):
            load_config(path)

    def test_missing_dependency_is_rejected(self):
        missing = copy.deepcopy(self.raw_config)
        missing["dependency_files"] = missing["dependency_files"][:-1]
        path = self.root / "missing-dependency.json"
        path.write_bytes(canonical_bytes(missing))
        with self.assertRaisesRegex(RuntimeError, "dependency files"):
            load_config(path)

    def test_certificate_phase_slot_root_and_interval_are_exact(self):
        mutations = (
            ("phase", "B_ONLY", "phase"),
            ("slot", "B", "slot"),
            ("route_lock_sha256", "e" * 64, "route lock"),
            ("started_ns", 2, "interval"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                certificate = copy.deepcopy(self.certificate)
                certificate[field] = value
                path = self.root / f"bad-{field}.json"
                path.write_bytes(canonical_bytes(certificate))
                with self.assertRaisesRegex(Exception, message):
                    parse_artifact_certificate(
                        path,
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                        "model-a",
                        "A_ONLY",
                        "A",
                        "b" * 64,
                    )

    def test_readiness_lock_cannot_precede_artifact_certificate(self):
        certificate = parse_artifact_certificate(
            self.certificate_path,
            hashlib.sha256(self.certificate_path.read_bytes()).hexdigest(),
            "model-a",
            "A_ONLY",
            "A",
            "b" * 64,
        )
        lock = {
            "artifact_snapshot_sha256": certificate["sha256"],
            "event_ns": 1,
            "phase": "A_ONLY",
            "phase_id": "phase-a",
            "schema": "s39-cp0-r1-readiness-lock-v2.3",
            "v2_2_phase_lock_sha256": "b" * 64,
        }
        path = self.root / "early-lock.json"
        path.write_bytes(canonical_bytes(lock))
        with self.assertRaisesRegex(Exception, "completed after"):
            parse_readiness_lock(
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                "A_ONLY",
                certificate,
                "b" * 64,
            )


if __name__ == "__main__":
    unittest.main()
