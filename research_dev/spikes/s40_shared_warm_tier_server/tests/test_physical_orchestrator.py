#!/usr/bin/env python3

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
S40 = HERE.parent
sys.path.insert(0, str(S40))

from evidence_common import (  # noqa: E402
    EvidenceError,
    canonical_bytes,
    digest_file,
)
from campaign_plan import CAMPAIGN_TOOL_PATHS, build_campaign  # noqa: E402
from gpu_isolation import canonical_lock_path  # noqa: E402
from physical_orchestrator import (  # noqa: E402
    create_internal_token_file,
    deterministic_launch_environment,
    expected_nul_cmdline,
    file_argument_bindings,
    finalize_runtime_identity,
    materialize_inputs,
    mkdir_new,
    publish_controller_identity,
    privileged_launch_environment,
    remove_internal_token_file,
    run_cleanup_actions,
    release_controller_publication_link,
    release_runtime_publication_link,
    recheck_gateway_processes,
    require_controller_port_available,
    validate_physical_plan,
    verify_argument_bindings,
    wait_for_controller_bindings,
    wait_for_gpu_observer,
    write_transport_descriptors,
)
from validate_inputs import DEFAULT_CONTRACT  # noqa: E402
from runtime_binding import process_start_time_ticks  # noqa: E402


GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
SERVING_ARGV = [
    "--ctx-size", "4096",
    "--parallel", "8",
    "--batch-size", "2048",
    "--ubatch-size", "512",
    "--flash-attn", "on",
    "--cont-batching",
    "--cache-type-k", "f16",
    "--cache-type-v", "f16",
    "--split-mode", "none",
]


class PhysicalOrchestratorTests(unittest.TestCase):
    def test_internal_token_is_private_and_ephemeral(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "warm-tier-internal.token"
            create_internal_token_file(path)
            metadata = path.stat(follow_symlinks=False)
            self.assertEqual(metadata.st_mode & 0o777, 0o600)
            self.assertEqual(metadata.st_nlink, 1)
            value = path.read_bytes()
            self.assertRegex(value, rb"^[0-9a-f]{64}\n$")
            remove_internal_token_file(path)
            self.assertFalse(path.exists())

            create_internal_token_file(path)
            path.chmod(0o644)
            with self.assertRaisesRegex(
                    Exception, "token cleanup metadata"):
                remove_internal_token_file(path)
            self.assertFalse(path.exists())

            target = Path(raw) / "missing-target"
            path.symlink_to(target)
            with self.assertRaisesRegex(
                    Exception, "warm-tier internal token path"):
                create_internal_token_file(path)
            with self.assertRaisesRegex(
                    Exception, "token cleanup metadata"):
                remove_internal_token_file(path)
            self.assertFalse(os.path.lexists(path))

    def test_cleanup_attempts_later_actions_after_first_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            token = Path(raw) / "warm-tier-internal.token"
            create_internal_token_file(token)
            stopped = []

            def fail_first():
                raise RuntimeError("injected cleanup failure")

            with self.assertRaisesRegex(Exception, "cleanup failed"):
                run_cleanup_actions([
                    ("first", fail_first),
                    ("desktop gateway", lambda: stopped.append("desktop")),
                    ("controller", lambda: stopped.append("controller")),
                    (
                        "warm-tier internal token",
                        lambda: remove_internal_token_file(token),
                    ),
                ])
            self.assertEqual(stopped, ["desktop", "controller"])
            self.assertFalse(os.path.lexists(token))

    def test_token_creation_failure_removes_directory_entry(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            token = directory / "warm-tier-internal.token"
            real_open = os.open

            def injected_open(path, flags, *args, **kwargs):
                if Path(path) == directory and flags & os.O_DIRECTORY:
                    raise OSError("injected directory fsync open failure")
                return real_open(path, flags, *args, **kwargs)

            with mock.patch(
                    "physical_orchestrator.os.open",
                    side_effect=injected_open):
                with self.assertRaisesRegex(
                        OSError, "injected directory"):
                    create_internal_token_file(token)
            self.assertFalse(os.path.lexists(token))

    def make_plan(self, root: Path) -> tuple[Path, dict]:
        contract_dir = root / "s40"
        contract_dir.mkdir()
        contract = contract_dir / "EXPERIMENT_CONTRACT.json"
        contract.write_bytes(DEFAULT_CONTRACT.read_bytes())
        (root / "s39_desktop_swap_baseline").symlink_to(
            S40.parent / "s39_desktop_swap_baseline",
            target_is_directory=True,
        )
        bridge_script = root / "gateway_bridge.py"
        bridge_script.write_text("# fixture\n", encoding="ascii")
        gateway_script = root / "desktop_gateway.py"
        gateway_script.write_text("# fixture\n", encoding="ascii")
        binary_directory = root / "bin"
        binary_directory.mkdir()
        controller = binary_directory / "llama-server"
        native_bench = binary_directory / "test-warm-tier-executors"
        nvidia_smi = binary_directory / "nvidia-smi"
        ldd = binary_directory / "ldd"
        for binary in (controller, native_bench, nvidia_smi, ldd):
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
            binary.chmod(0o700)
        nvidia_smi.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' "
            "'GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08,"
            "NVIDIA GeForce RTX 4060 Ti,00000000:01:00.0,0'\n",
            encoding="ascii",
        )
        config = root / "gateway-config.json"
        config.write_bytes(canonical_bytes({
            "cache_regime": "WARM_CACHE",
            "executor_id": "gpu0",
            "schema": "s40-desktop-executor-config-v4",
        }))
        evidence_root = root / "evidence-root-source.json"
        evidence_root.write_bytes(canonical_bytes({
            "c3_placement_artifacts": {},
            "configuration": "C1_GPU_ONLY_OPTIMIZED",
            "executor_configs": [{
                "executor_id": "gpu0",
                "gateway_config_path": str(config),
                "gateway_config_sha256": digest_file(config),
            }],
            "experiment_contract_sha256": digest_file(contract),
            "schema": "s40-runtime-evidence-root-v1",
        }))
        runtime_plan = root / "runtime-plan-source.json"
        runtime_plan.write_bytes(canonical_bytes({
            "c3_profile_lock_path": None,
            "c3_profile_lock_sha256": None,
            "evidence_root_path": str(evidence_root),
            "evidence_root_sha256": digest_file(evidence_root),
            "event_log_path": str(root / "source-events.jsonl"),
            "executors": [{
                "credits": 8,
                "execute_concurrency": 8,
                "executor_id": "gpu0",
                "kind": "GPU_PRIMARY",
                "output_limit_bytes": 65536,
                "queue_capacity": 74,
                "socket_path": str(root / "source.sock"),
                "timeout_ms": 300000,
                "transport": "UNIX_SOCKET",
            }],
            "hot_model_id": "qwen3-8b-q8_0",
            "mode": "C1_GPU_ONLY_OPTIMIZED",
            "run_id": "fixture-run",
            "schema": "s40-runtime-config-plan-v2",
        }))
        boot_id = Path(
            "/proc/sys/kernel/random/boot_id").read_text(
                encoding="ascii").strip()
        output = root / "physical-output"
        plan = {
            "cache_regime": "WARM_HOST_CACHE",
            "campaign_binding": None,
            "contract_path": str(contract.resolve()),
            "development": True,
            "devices": [{
                "boot_id": boot_id,
                "device_role": "GPU",
                "stable_id": GPU_UUID,
            }],
            "gateway_processes": [{
                "argv": [
                    sys.executable,
                    str(gateway_script),
                    "--socket",
                    str(root / "source.sock"),
                    "--config",
                    str(config),
                    "--evidence",
                    str(root / "source-command.jsonl"),
                    "--run-id",
                    "fixture-run",
                    "--runtime-config",
                    str(output / "runtime-config.json"),
                    "--controller-identity",
                    str(output / "controller-identity-lock.json"),
                    "--controller-binding-evidence",
                    str(output / "executor-controller-binding-gpu0.json"),
                    "--executor-instance-id",
                    "fixture-gpu0-instance",
                ],
                "executor_id": "gpu0",
                "socket_name": "gpu0.sock",
            }],
            "gpu_index": 0,
            "gpu_lock_path": str(canonical_lock_path(GPU_UUID)),
            "gpu_pci_bus_id": "00000000:01:00.0",
            "ldd_path": str(ldd),
            "mode": "C1_GPU_ONLY_OPTIMIZED",
            "nvidia_smi_path": str(nvidia_smi),
            "output_dir": str(output),
            "phone_identity_argv": None,
            "phone_telemetry_argv": None,
            "repeat_index": 0,
            "run_id": "fixture-run",
            "runtime_plan_template": str(runtime_plan),
            "schema": "s40-physical-run-plan-v2",
            "server_argv": [
                str(controller),
                *SERVING_ARGV,
                "--host",
                "127.0.0.1",
                "--port",
                "48991",
                "--threads-http",
                "128",
            ],
            "server_base_url": "http://127.0.0.1:48991",
        }
        path = root / "physical-plan.json"
        path.write_bytes(canonical_bytes(plan))
        return path, plan

    def rewrite(self, path: Path, plan: dict) -> None:
        path.write_bytes(canonical_bytes(plan))

    def start_fixture_gateway(
            self,
            output: Path,
            runtime_binaries: dict[str, Path],
            executor_bundle: dict,
            evidence_bundle: dict,
            gateway_argv_paths: dict[str, Path]) -> tuple[
                subprocess.Popen,
                dict[str, list[str]],
                dict[str, list[dict]],
                dict[str, str],
            ]:
        argv = {
            "gpu0": [
                str(runtime_binaries["python"]),
                "-I",
                "-S",
                "-B",
                "-c",
                "import time; time.sleep(60)",
                "--executor-instance-id",
                "fixture-gpu0-instance",
            ],
        }
        gateway_argv_paths["gpu0"].write_bytes(canonical_bytes({
            "argv": argv["gpu0"],
            "schema": "s40-gateway-argv-v4",
        }))
        bindings = {
            "gpu0": file_argument_bindings(
                argv["gpu0"], output, "gateway-gpu0"),
        }
        environment = deterministic_launch_environment(
            output,
            output / "captured-runtime" / "lib",
            output / "runtime-config.json",
            runtime_binaries["nvidia_smi"],
            GPU_UUID,
            executor_bundle["environment"],
            evidence_bundle["environment"],
        )
        environment = privileged_launch_environment(
            environment, output / "warm-tier-internal.token")
        process = subprocess.Popen(
            argv["gpu0"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
        )

        def stop() -> None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

        self.addCleanup(stop)
        expected = expected_nul_cmdline(argv["gpu0"])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            cmdline = Path(f"/proc/{process.pid}/cmdline")
            if cmdline.is_file() and cmdline.read_bytes() == expected:
                break
            if process.poll() is not None:
                self.fail("fixture gateway exited before identity capture")
            time.sleep(0.01)
        else:
            self.fail("fixture gateway did not publish its exact cmdline")
        return process, argv, bindings, environment

    def make_controller_binding_fixture(self, root: Path) -> dict:
        path, plan = self.make_plan(root)
        validated = validate_physical_plan(path)
        output = Path(plan["output_dir"])
        mkdir_new(output)
        (
            runtime_plan,
            runtime_path,
            argv,
            configs,
            gateway_argv,
            transport_descriptors,
            bundle,
            evidence_bundle,
            _,
            runtime_binaries,
            _,
            _,
            _,
            plan_value,
        ) = materialize_inputs(validated, output)

        (
            process,
            argv,
            gateway_file_bindings,
            launch_environment,
        ) = self.start_fixture_gateway(
            output,
            runtime_binaries,
            bundle,
            evidence_bundle,
            gateway_argv,
        )
        runtime, identities, runtime_publication = (
            finalize_runtime_identity(
                plan_value,
                runtime_plan,
                runtime_path,
                {"gpu0": process},
                argv,
                gateway_file_bindings,
                output,
                gateway_argv,
                configs,
                transport_descriptors,
                bundle,
                output / "executor-bundle",
                Path(plan["contract_path"]),
            )
        )
        boot_id = Path(
            "/proc/sys/kernel/random/boot_id"
        ).read_text(encoding="ascii").strip()
        identity_path = output / "controller-identity-lock.json"
        controller_identity, controller_publication = (
            publish_controller_identity(
                identity_path,
                process,
                runtime_publication["published_ns"],
                Path(argv["gpu0"][0]),
                plan["run_id"],
                boot_id,
                runtime_publication,
            )
        )
        binding_path = output / "executor-controller-binding-gpu0.json"
        identity = identities["gpu0"]
        binding_path.write_bytes(canonical_bytes({
            "authenticated_ns": time.monotonic_ns(),
            "controller_executable_path":
                controller_identity["controller_executable_path"],
            "controller_executable_sha256":
                controller_identity["controller_executable_sha256"],
            "controller_gid": controller_identity["controller_gid"],
            "controller_identity_device":
                controller_publication["device"],
            "controller_identity_inode":
                controller_publication["inode"],
            "controller_identity_path":
                controller_publication["path"],
            "controller_identity_sha256":
                controller_publication["sha256"],
            "controller_pid": controller_identity["controller_pid"],
            "controller_start_time_ticks":
                controller_identity["controller_start_time_ticks"],
            "controller_uid": controller_identity["controller_uid"],
            "executor_id": "gpu0",
            "executor_instance_id": identity["executor_instance_id"],
            "gateway_pid": identity["gateway_pid"],
            "gateway_start_time_ticks":
                identity["gateway_start_time_ticks"],
            "host_boot_id": boot_id,
            "peer_gid": controller_identity["controller_gid"],
            "peer_pid": controller_identity["controller_pid"],
            "peer_uid": controller_identity["controller_uid"],
            "run_id": plan["run_id"],
            "runtime_config_device": runtime_publication["device"],
            "runtime_config_inode": runtime_publication["inode"],
            "runtime_config_path": runtime_publication["path"],
            "runtime_config_sha256": runtime_publication["sha256"],
            "schema": "s40-executor-controller-binding-v1",
        }))
        return {
            "binding_path": binding_path,
            "bundle": bundle,
            "configs": configs,
            "controller_identity": controller_identity,
            "controller_publication": controller_publication,
            "gateway_argv": gateway_argv,
            "gateway_file_bindings": gateway_file_bindings,
            "gateway_process_argv": argv,
            "identities": identities,
            "launch_environment": launch_environment,
            "output": output,
            "process": process,
            "runtime": runtime,
            "runtime_publication": runtime_publication,
            "transport_descriptors": transport_descriptors,
        }

    def make_primary_plan(self, root: Path) -> tuple[Path, dict]:
        path, plan = self.make_plan(root)
        run_id = "fixture-000"
        runtime_plan_path = Path(plan["runtime_plan_template"])
        runtime_plan = __import__("json").loads(
            runtime_plan_path.read_text(encoding="ascii"))
        runtime_plan["run_id"] = run_id
        runtime_plan_path.write_bytes(canonical_bytes(runtime_plan))
        plan["gateway_processes"][0]["argv"][
            plan["gateway_processes"][0]["argv"].index("--run-id") + 1
        ] = run_id
        plan["run_id"] = run_id
        controller = Path(plan["server_argv"][0])
        campaign = build_campaign(
            1,
            campaign_id="fixture",
            experiment_contract_sha256=digest_file(
                Path(plan["contract_path"])),
            software_lock={
                **{
                    f"{name}_sha256": digest_file(tool)
                    for name, tool in CAMPAIGN_TOOL_PATHS.items()
                },
                "controller_binary_sha256": digest_file(controller),
                "evidence_bundle_manifest_sha256": "1" * 64,
                "executor_bundle_manifest_sha256": "2" * 64,
                "ldd_sha256": digest_file(Path(plan["ldd_path"])),
                "native_bench_binary_sha256": digest_file(
                    controller.parent / "test-warm-tier-executors"),
                "nvidia_smi_sha256": digest_file(
                    Path(plan["nvidia_smi_path"])),
                "python_sha256": digest_file(Path(sys.executable)),
                "schema": "s40-primary-software-lock-v2",
            },
        )
        campaign_path = root / "campaign.json"
        campaign_path.write_bytes(canonical_bytes(campaign))
        plan["campaign_binding"] = {
            "campaign_path": str(campaign_path.resolve()),
            "campaign_sha256": digest_file(campaign_path),
            "order": 0,
            "phase": "PRIMARY",
        }
        plan["development"] = False
        self.rewrite(path, plan)
        return path, plan

    def test_valid_preflight_and_materialization(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_plan(root)
            validated = validate_physical_plan(path)
            output = Path(plan["output_dir"])
            mkdir_new(output)
            (
                runtime_plan,
                runtime,
                argv,
                configs,
                gateway_argv,
                transport_descriptors,
                bundle,
                evidence_bundle,
                source_preflight,
                runtime_binaries,
                runtime_dependencies,
                phone_argv,
                phone_argv_paths,
                plan_value,
            ) = (
                materialize_inputs(validated, output))
            self.assertFalse(runtime_plan.exists())
            self.assertFalse(runtime.exists())
            self.assertEqual(set(argv), {"gpu0"})
            self.assertEqual(set(configs), {"gpu0"})
            self.assertEqual(
                __import__("json").loads(
                    configs["gpu0"].read_text(encoding="ascii")
                )["cache_regime"],
                "WARM_CACHE",
            )
            self.assertEqual(set(gateway_argv), {"gpu0"})
            self.assertEqual(set(transport_descriptors), {"gpu0"})
            self.assertEqual(bundle["status"], "PASS")
            self.assertEqual(evidence_bundle["status"], "PASS")
            self.assertTrue(source_preflight.is_file())
            self.assertTrue(runtime_dependencies.is_file())
            self.assertEqual(
                set(runtime_binaries),
                {"controller", "native_bench", "nvidia_smi", "python"},
            )
            self.assertEqual(phone_argv, {})
            self.assertEqual(phone_argv_paths, {})
            self.assertIn(
                '"role":"runtime_plan_template"',
                source_preflight.read_text(encoding="ascii"),
            )
            self.assertEqual(
                argv["gpu0"][1:4], ["-I", "-S", "-B"])
            self.assertEqual(
                argv["gpu0"][argv["gpu0"].index("--run-id") + 1],
                "fixture-run",
            )
            self.assertEqual(
                argv["gpu0"][
                    argv["gpu0"].index("--runtime-config") + 1],
                str(runtime),
            )
            self.assertEqual(
                argv["gpu0"][
                    argv["gpu0"].index("--executor-instance-id") + 1],
                "fixture-gpu0-instance",
            )
            self.assertEqual(
                argv["gpu0"][
                    argv["gpu0"].index("--controller-identity") + 1],
                str(output / "controller-identity-lock.json"),
            )
            self.assertEqual(
                argv["gpu0"][
                    argv["gpu0"].index("--controller-binding-evidence") + 1],
                str(output / "executor-controller-binding-gpu0.json"),
            )

            (
                process,
                argv,
                gateway_file_bindings,
                _,
            ) = self.start_fixture_gateway(
                output,
                runtime_binaries,
                bundle,
                evidence_bundle,
                gateway_argv,
            )

            runtime_value, identities, publication = (
                finalize_runtime_identity(
                    plan_value,
                    runtime_plan,
                    runtime,
                    {"gpu0": process},
                    argv,
                    gateway_file_bindings,
                    output,
                    gateway_argv,
                    configs,
                    transport_descriptors,
                    bundle,
                    output / "executor-bundle",
                    Path(plan["contract_path"]),
                )
            )
            self.assertTrue(runtime_plan.is_file())
            self.assertTrue(runtime.is_file())
            self.assertGreater(publication["published_ns"], 0)
            temporary = Path(publication["temporary_path"])
            self.assertTrue(temporary.is_file())
            self.assertEqual(
                temporary.stat(follow_symlinks=False).st_ino,
                runtime.stat(follow_symlinks=False).st_ino,
            )
            self.assertEqual(
                runtime_value["schema"],
                "llama-server-warm-tier-runtime-v4",
            )
            self.assertEqual(
                identities["gpu0"]["executor_instance_id"],
                "fixture-gpu0-instance",
            )
            self.assertFalse(transport_descriptors["gpu0"].exists())
            self.assertIn(
                str(output / "sockets" / "gpu0.sock"),
                runtime.read_text(encoding="ascii"),
            )
            release_runtime_publication_link(publication)
            self.assertFalse(temporary.exists())

    def test_gateway_identity_is_rechecked_immediately_before_publish(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_plan(root)
            validated = validate_physical_plan(path)
            output = Path(plan["output_dir"])
            mkdir_new(output)
            materialized = materialize_inputs(validated, output)
            (
                runtime_plan,
                runtime,
                argv,
                configs,
                gateway_argv,
                transport_descriptors,
                bundle,
                _,
                _,
                _,
                _,
                _,
                _,
                plan_value,
            ) = materialized

            (
                live_process,
                argv,
                gateway_file_bindings,
                _,
            ) = self.start_fixture_gateway(
                output,
                materialized[9],
                bundle,
                materialized[7],
                gateway_argv,
            )

            class ExitsBeforePublish:
                pid = live_process.pid
                calls = 0

                @classmethod
                def poll(cls):
                    cls.calls += 1
                    return None if cls.calls == 1 else 1

            with self.assertRaisesRegex(
                    EvidenceError, "exited before publish"):
                finalize_runtime_identity(
                    plan_value,
                    runtime_plan,
                    runtime,
                    {"gpu0": ExitsBeforePublish()},
                    argv,
                    gateway_file_bindings,
                    output,
                    gateway_argv,
                    configs,
                    transport_descriptors,
                    bundle,
                    output / "executor-bundle",
                    Path(plan["contract_path"]),
                )
            self.assertFalse(runtime.exists())
            self.assertFalse(any(output.glob(".runtime-config.json.*.tmp")))

    def test_controller_binding_drives_v4_transport_descriptor(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            fixture = self.make_controller_binding_fixture(Path(directory))
            ready = wait_for_controller_bindings(
                {"gpu0": fixture["process"]},
                {"gpu0": fixture["binding_path"]},
                fixture["runtime"],
                fixture["identities"],
                fixture["runtime_publication"],
                fixture["controller_identity"],
                fixture["controller_publication"],
                timeout_s=1,
            )
            self.assertEqual(
                [row["executor_id"] for row in ready], ["gpu0"])
            post_auth = recheck_gateway_processes(
                {"gpu0": fixture["process"]},
                fixture["gateway_process_argv"],
                {
                    "gpu0":
                        fixture["identities"]["gpu0"]["process_identity"],
                },
                fixture["gateway_file_bindings"],
                fixture["output"],
            )
            write_transport_descriptors(
                fixture["runtime"],
                fixture["identities"],
                post_auth,
                fixture["gateway_file_bindings"],
                {"gpu0": fixture["launch_environment"]},
                fixture["runtime_publication"],
                fixture["controller_identity"],
                fixture["controller_publication"],
                fixture["gateway_argv"],
                fixture["configs"],
                {"gpu0": fixture["binding_path"]},
                ready,
                fixture["transport_descriptors"],
                fixture["bundle"],
                fixture["output"] / "executor-bundle",
            )
            descriptor = __import__("json").loads(
                fixture["transport_descriptors"]["gpu0"].read_text(
                    encoding="ascii"))
            self.assertEqual(
                descriptor["schema"],
                "s40-executor-transport-descriptor-v4",
            )
            self.assertEqual(
                descriptor["controller_binding_inode"],
                fixture["binding_path"].stat(
                    follow_symlinks=False).st_ino,
            )
            self.assertEqual(
                descriptor["controller_identity_sha256"],
                fixture["controller_publication"]["sha256"],
            )
            release_controller_publication_link(
                fixture["controller_publication"])
            release_runtime_publication_link(
                fixture["runtime_publication"])

    def test_malformed_controller_binding_blocks_readiness(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            fixture = self.make_controller_binding_fixture(Path(directory))
            fixture["binding_path"].write_bytes(b"{}\n")
            with self.assertRaisesRegex(
                    EvidenceError, "controller binding evidence fields"):
                wait_for_controller_bindings(
                    {"gpu0": fixture["process"]},
                    {"gpu0": fixture["binding_path"]},
                    fixture["runtime"],
                    fixture["identities"],
                    fixture["runtime_publication"],
                    fixture["controller_identity"],
                    fixture["controller_publication"],
                    timeout_s=1,
                )

    def test_controller_binding_replacement_after_readiness_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            fixture = self.make_controller_binding_fixture(Path(directory))
            ready = wait_for_controller_bindings(
                {"gpu0": fixture["process"]},
                {"gpu0": fixture["binding_path"]},
                fixture["runtime"],
                fixture["identities"],
                fixture["runtime_publication"],
                fixture["controller_identity"],
                fixture["controller_publication"],
                timeout_s=1,
            )
            post_auth = recheck_gateway_processes(
                {"gpu0": fixture["process"]},
                fixture["gateway_process_argv"],
                {
                    "gpu0":
                        fixture["identities"]["gpu0"]["process_identity"],
                },
                fixture["gateway_file_bindings"],
                fixture["output"],
            )
            raw = fixture["binding_path"].read_bytes()
            replacement = fixture["binding_path"].with_name(
                "replacement-controller-binding.json")
            replacement.write_bytes(raw)
            os.replace(replacement, fixture["binding_path"])
            with self.assertRaisesRegex(
                    EvidenceError, "changed after readiness"):
                write_transport_descriptors(
                    fixture["runtime"],
                    fixture["identities"],
                    post_auth,
                    fixture["gateway_file_bindings"],
                    {"gpu0": fixture["launch_environment"]},
                    fixture["runtime_publication"],
                    fixture["controller_identity"],
                    fixture["controller_publication"],
                    fixture["gateway_argv"],
                    fixture["configs"],
                    {"gpu0": fixture["binding_path"]},
                    ready,
                    fixture["transport_descriptors"],
                    fixture["bundle"],
                    fixture["output"] / "executor-bundle",
                )

    def test_primary_plan_is_bound_to_exact_prospective_row(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_primary_plan(root)
            result = validate_physical_plan(path)
            self.assertEqual(result["campaign_binding"]["order"], 0)
            plan["repeat_index"] = 1
            self.rewrite(path, plan)
            with self.assertRaisesRegex(EvidenceError, "campaign row mismatch"):
                validate_physical_plan(path)

    def test_dependency_resolver_mutation_after_preflight_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_plan(root)
            validated = validate_physical_plan(path)
            Path(plan["ldd_path"]).write_bytes(
                b"#!/bin/sh\nprintf 'forged\\n'\n")
            with self.assertRaisesRegex(
                    EvidenceError, "source lock changed.*ldd"):
                materialize_inputs(validated, Path(plan["output_dir"]))

    def test_output_must_not_exist(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_plan(root)
            Path(plan["output_dir"]).mkdir()
            with self.assertRaisesRegex(EvidenceError, "already exists"):
                validate_physical_plan(path)

    def test_selected_gpu_mapping_is_exact(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_plan(root)
            plan["gpu_index"] = 1
            self.rewrite(path, plan)
            with self.assertRaisesRegex(
                    EvidenceError, "selected GPU mapping mismatch"):
                validate_physical_plan(path)
            plan["gpu_index"] = 0
            plan["gpu_pci_bus_id"] = "00000000:02:00.0"
            self.rewrite(path, plan)
            with self.assertRaisesRegex(
                    EvidenceError, "selected GPU mapping mismatch"):
                validate_physical_plan(path)

    def test_gpu_lock_path_must_be_local_regular_target(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_plan(root)
            plan["gpu_lock_path"] = str(root / "alternate.lock")
            self.rewrite(path, plan)
            with self.assertRaisesRegex(
                    EvidenceError, "noncanonical GPU lock path"):
                validate_physical_plan(path)

    def test_gpu_observer_early_exit_is_rejected(self):
        class Exited:
            @staticmethod
            def poll():
                return 2

        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                    EvidenceError, "exited before readiness"):
                wait_for_gpu_observer(
                    Exited(),
                    root / "observer.jsonl",
                    root / "selected-gpu.lock",
                    "fixture-run",
                    timeout_s=1,
                )

    def test_cold_nvme_reaches_desktop_gateway_config(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_plan(root)
            plan["cache_regime"] = "COLD_NVME"
            self.rewrite(path, plan)
            validated = validate_physical_plan(path)
            output = Path(plan["output_dir"])
            mkdir_new(output)
            materialized = materialize_inputs(validated, output)
            config = __import__("json").loads(
                materialized[3]["gpu0"].read_text(encoding="ascii"))
            self.assertEqual(config["cache_regime"], "COLD_NVME")

    def test_server_command_must_match_base_url(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_plan(root)
            port = plan["server_argv"].index("--port") + 1
            plan["server_argv"][port] = "48992"
            self.rewrite(path, plan)
            with self.assertRaisesRegex(EvidenceError, "base URL differ"):
                validate_physical_plan(path)

    def test_server_serving_envelope_is_exact(self):
        mutations = {
            "--ctx-size": "2048",
            "--parallel": "4",
            "--batch-size": "1024",
            "--ubatch-size": "256",
            "--flash-attn": "off",
            "--cache-type-k": "q8_0",
            "--cache-type-v": "q8_0",
            "--split-mode": "layer",
        }
        for flag, replacement in mutations.items():
            with self.subTest(flag=flag), tempfile.TemporaryDirectory(
                    prefix="s40_physical_") as directory:
                path, plan = self.make_plan(Path(directory))
                position = plan["server_argv"].index(flag) + 1
                plan["server_argv"][position] = replacement
                self.rewrite(path, plan)
                with self.assertRaisesRegex(
                        EvidenceError, "serving contract"):
                    validate_physical_plan(path)

        for mutation in ("missing", "duplicate", "disabled"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                    prefix="s40_physical_") as directory:
                path, plan = self.make_plan(Path(directory))
                position = plan["server_argv"].index("--cont-batching")
                if mutation == "missing":
                    del plan["server_argv"][position]
                elif mutation == "duplicate":
                    plan["server_argv"].insert(
                        position, "--cont-batching")
                else:
                    plan["server_argv"][position] = "--no-cont-batching"
                self.rewrite(path, plan)
                with self.assertRaisesRegex(
                        EvidenceError, "serving contract"):
                    validate_physical_plan(path)

        with tempfile.TemporaryDirectory(
                prefix="s40_physical_") as directory:
            path, plan = self.make_plan(Path(directory))
            plan["server_argv"].extend(["-c", "4096"])
            self.rewrite(path, plan)
            with self.assertRaisesRegex(EvidenceError, "serving contract"):
                validate_physical_plan(path)

    def test_server_rejects_proxy_agent_and_tool_features(self):
        mutations = (
            ["--ui-mcp-proxy"],
            ["--ui_mcp_proxy"],
            ["--webui-mcp-proxy"],
            ["--webui_mcp_proxy"],
            ["-ag"],
            ["--agent"],
            ["--tools", "exec_shell_command"],
            ["--ui-mcp-proxy=true"],
            ["--tools=exec_shell_command"],
        )
        for extra_argv in mutations:
            with self.subTest(extra_argv=extra_argv), tempfile.TemporaryDirectory(
                    prefix="s40_physical_") as directory:
                path, plan = self.make_plan(Path(directory))
                plan["server_argv"].extend(extra_argv)
                self.rewrite(path, plan)
                with self.assertRaisesRegex(
                        EvidenceError, "forbidden server feature"):
                    validate_physical_plan(path)

    def test_gateway_set_is_exact(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_plan(root)
            plan["gateway_processes"] = []
            self.rewrite(path, plan)
            with self.assertRaisesRegex(EvidenceError, "gateway set mismatch"):
                validate_physical_plan(path)

    def test_gateway_run_identity_is_exact(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_plan(root)
            argv = plan["gateway_processes"][0]["argv"]
            argv[argv.index("--run-id") + 1] = "other"
            self.rewrite(path, plan)
            with self.assertRaisesRegex(EvidenceError, "run ID mismatch"):
                validate_physical_plan(path)

    def test_gateway_runtime_path_and_instance_are_exact(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_plan(root)
            argv = plan["gateway_processes"][0]["argv"]
            argv[argv.index("--runtime-config") + 1] = str(
                root / "other-runtime.json")
            self.rewrite(path, plan)
            with self.assertRaisesRegex(
                    EvidenceError, "runtime config path mismatch"):
                validate_physical_plan(path)
            argv[argv.index("--runtime-config") + 1] = str(
                Path(plan["output_dir"]) / "runtime-config.json")
            argv[argv.index("--executor-instance-id") + 1] = ""
            self.rewrite(path, plan)
            with self.assertRaisesRegex(
                    EvidenceError, "invalid command array|instance"):
                validate_physical_plan(path)

    def test_device_set_is_exact(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_plan(root)
            plan["devices"].append({
                "boot_id": "phone-boot",
                "device_role": "OP15",
                "stable_id": "phone",
            })
            self.rewrite(path, plan)
            with self.assertRaisesRegex(EvidenceError, "device role set"):
                validate_physical_plan(path)

    def test_preexisting_listener_is_rejected(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        with self.assertRaisesRegex(EvidenceError, "pre-existing listener"):
            require_controller_port_available(
                f"http://127.0.0.1:{port}")

    def test_bindings_exclude_output_and_detect_input_mutation(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            executable = root / "gateway"
            config = root / "config.json"
            executable.write_bytes(b"binary\n")
            config.write_bytes(b"{}\n")
            output = root / "output"
            mkdir_new(output)
            rows = file_argument_bindings(
                [
                    str(executable),
                    "--config",
                    str(config),
                    "--evidence",
                    str(output / "future.jsonl"),
                ],
                output,
                "gateway",
            )
            self.assertEqual(
                [row["argv_index"] for row in rows], [0, 2])
            verify_argument_bindings(rows, output)
            config.write_bytes(b'{"changed":true}\n')
            with self.assertRaisesRegex(EvidenceError, "changed"):
                verify_argument_bindings(rows, output)

    def test_runtime_template_mutation_after_validation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_plan(root)
            validated = validate_physical_plan(path)
            runtime = Path(plan["runtime_plan_template"])
            value = __import__("json").loads(
                runtime.read_text(encoding="ascii"))
            value["executors"][0]["credits"] = 1
            runtime.write_bytes(canonical_bytes(value))
            output = Path(plan["output_dir"])
            mkdir_new(output)
            with self.assertRaisesRegex(
                    EvidenceError,
                    "source lock changed: runtime_plan_template"):
                materialize_inputs(validated, output)

    def test_each_primary_source_is_locked_after_validation(self):
        mutations = (
            ("physical_plan", lambda root, plan, path: path),
            (
                "experiment_contract",
                lambda root, plan, path: Path(plan["contract_path"]),
            ),
            (
                "evidence_root",
                lambda root, plan, path:
                    root / "evidence-root-source.json",
            ),
            (
                "runtime_binary::controller",
                lambda root, plan, path:
                    Path(plan["server_argv"][0]),
            ),
            (
                "runtime_binary::native_bench",
                lambda root, plan, path:
                    Path(plan["server_argv"][0]).parent
                    / "test-warm-tier-executors",
            ),
        )
        for role, select in mutations:
            with self.subTest(role=role), tempfile.TemporaryDirectory(
                    prefix="s40_physical_") as directory:
                root = Path(directory)
                path, plan = self.make_plan(root)
                validated = validate_physical_plan(path)
                source = select(root, plan, path)
                source.write_bytes(source.read_bytes() + b" ")
                output = Path(plan["output_dir"])
                mkdir_new(output)
                with self.assertRaisesRegex(
                        EvidenceError, f"source lock changed: {role}"):
                    materialize_inputs(validated, output)

    def test_gateway_config_mutation_after_validation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            root = Path(directory)
            path, plan = self.make_plan(root)
            validated = validate_physical_plan(path)
            config = root / "gateway-config.json"
            value = __import__("json").loads(
                config.read_text(encoding="ascii"))
            value["cache_regime"] = "COLD_NVME"
            config.write_bytes(canonical_bytes(value))
            output = Path(plan["output_dir"])
            mkdir_new(output)
            with self.assertRaisesRegex(
                    EvidenceError,
                    "source lock changed: gateway_config::gpu0"):
                materialize_inputs(validated, output)

    def test_malformed_plan_cli_fails_without_traceback(self):
        with tempfile.TemporaryDirectory(prefix="s40_physical_") as directory:
            malformed = Path(directory) / "physical-plan.json"
            malformed.write_bytes(b"0\n")
            process = subprocess.run(
                [
                    sys.executable,
                    str(S40 / "physical_orchestrator.py"),
                    "--plan",
                    str(malformed),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
            combined = process.stdout + process.stderr
            self.assertEqual(process.returncode, 2)
            self.assertEqual(
                combined,
                b"ERROR: EvidenceError: physical_plan: expected object\n",
            )
            self.assertNotIn(b"Traceback", combined)


if __name__ == "__main__":
    unittest.main()
