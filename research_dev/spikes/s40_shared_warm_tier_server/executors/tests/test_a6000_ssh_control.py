#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXECUTORS = HERE.parent
if str(EXECUTORS) not in sys.path:
    sys.path.insert(0, str(EXECUTORS))

from a6000_ssh_control import (
    load_config,
    public_key_fingerprint,
    run_control,
)
from phone_gateway import canonical_bytes, strict_json_loads


class A6000SshControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        known_hosts = root / "known_hosts"
        known_hosts.write_text("a6000 ssh-ed25519 AAAATEST\n", encoding="ascii")
        ssh = root / "ssh"
        shutil.copyfile("/usr/bin/ssh", ssh)
        ssh.chmod(0o755)
        identity_file = root / "id_ed25519"
        subprocess.run(
            [
                "/usr/bin/ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-f",
                str(identity_file),
            ],
            check=True,
        )
        identity_file.chmod(0o600)
        public_key = root / "id_ed25519.pub"
        fingerprint = public_key_fingerprint(public_key)
        ssh_keygen = root / "ssh-keygen"
        shutil.copyfile("/usr/bin/ssh-keygen", ssh_keygen)
        ssh_keygen.chmod(0o755)
        self.route = {
            "a6000_identity": "4" * 64,
            "artifact_certificate_sha256": "5" * 64,
            "model_sha256": "a" * 64,
            "op12_boot_id": "op12-boot",
            "op12_shard_sha256": "1" * 64,
            "op15_boot_id": "op15-boot",
            "op15_shard_sha256": "2" * 64,
            "qualification_sha256": "7" * 64,
            "readiness_lock_sha256": "6" * 64,
            "readiness_phase_id": "phase-a",
            "worker_sha256": "3" * 64,
        }
        remote_files = []
        remote_paths = {
            "a6000_phone_observer":
                "/srv/s40/a6000_phone_observer.py",
            "a6000_phone_route_control":
                "/srv/s40/a6000_phone_route_control.py",
            "adb": "/usr/bin/adb",
            "phone_gateway": "/srv/s40/phone_gateway.py",
            "python": "/usr/bin/python3",
            "readiness_v23": "/srv/s40/readiness_v23.py",
            "remote_config": "/srv/s40/routes.json",
        }
        for index, (role, path) in enumerate(sorted(remote_paths.items())):
            remote_files.append({
                "bytes": index + 1,
                "path": path,
                "role": role,
                "sha256": f"{index + 1:x}" * 64,
            })
        remote_package = {
            "a6000_identity": "4" * 64,
            "files": remote_files,
            "host_boot_id": "11111111-1111-4111-8111-111111111111",
            "schema": "s40-a6000-identity-package-v1",
        }
        value = {
            "identity_file_path": str(identity_file),
            "identity_public_key_bytes": public_key.stat().st_size,
            "identity_public_key_fingerprint": fingerprint,
            "identity_public_key_path": str(public_key),
            "identity_public_key_sha256": hashlib.sha256(
                public_key.read_bytes()
            ).hexdigest(),
            "known_hosts_bytes": known_hosts.stat().st_size,
            "known_hosts_path": str(known_hosts),
            "known_hosts_sha256": hashlib.sha256(
                known_hosts.read_bytes()
            ).hexdigest(),
            "remote_config_path": "/srv/s40/routes.json",
            "remote_identity_package": remote_package,
            "remote_identity_package_sha256": hashlib.sha256(
                canonical_bytes(remote_package)
            ).hexdigest(),
            "remote_python_path": "/usr/bin/python3",
            "remote_script_path": "/srv/s40/a6000_phone_route_control.py",
            "routes": {"model-a": self.route},
            "schema": "s40-a6000-ssh-control-v3",
            "ssh_argv": [
                str(ssh),
                "-F",
                "/dev/null",
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                f"UserKnownHostsFile={known_hosts}",
                "-o",
                "GlobalKnownHostsFile=/dev/null",
                "-o",
                "PasswordAuthentication=no",
                "-o",
                "KbdInteractiveAuthentication=no",
                "-o",
                "IdentityAgent=none",
                "-o",
                "LogLevel=ERROR",
                "-i",
                str(identity_file),
                "user@a6000",
            ],
            "ssh_env": {
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
            },
            "ssh_executable_bytes": ssh.stat().st_size,
            "ssh_executable_sha256": hashlib.sha256(
                ssh.read_bytes()
            ).hexdigest(),
            "ssh_keygen_bytes": ssh_keygen.stat().st_size,
            "ssh_keygen_path": str(ssh_keygen),
            "ssh_keygen_sha256": hashlib.sha256(
                ssh_keygen.read_bytes()
            ).hexdigest(),
        }
        self.value = value
        self.identity_file = identity_file
        self.known_hosts = known_hosts
        self.ssh_keygen = ssh_keygen
        self.config_path = root / "config.json"
        self.config_path.write_bytes(canonical_bytes(value))
        self.config = load_config(self.config_path)

    def tearDown(self):
        self.temporary.cleanup()

    def result(self, action, mutate=None):
        if action == "load":
            value = {
                **self.route,
                "model_id": "model-a",
                "route_observation": {
                    "direct_peer": {},
                    "model_id": "model-a",
                    "phones": {"op12": {}, "op15": {}},
                    "route_instance_id": "route-1",
                    "schema": "s40-phone-route-observation-v1",
                },
                "route_instance_id": "route-1",
                "schema": "s40-phone-route-load-v3",
                "success": True,
            }
        elif action == "unload":
            value = {
                "model_id": "model-a",
                "placements": {"op12": {}, "op15": {}},
                "route_instance_id": "route-1",
                "schema": "s40-phone-route-unload-v2",
                "success": True,
            }
        else:
            self.assertEqual(action, "rollback")
            value = {
                "model_id": "model-a",
                "route_instance_id": "route-1",
                "schema": "s40-phone-route-rollback-v1",
                "success": True,
            }
        if mutate is not None:
            mutate(value)
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=canonical_bytes(value),
            stderr=b"",
        )

    def test_exact_load_and_unload(self):
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            action = argv[argv.index("--action") + 1]
            return self.result(action)

        load = strict_json_loads(
            run_control(self.config, "load", "model-a", runner),
            "load",
        )
        unload = strict_json_loads(
            run_control(self.config, "unload", "model-a", runner),
            "unload",
        )
        self.assertEqual(load["route_instance_id"], "route-1")
        self.assertEqual(unload["route_instance_id"], "route-1")
        self.assertEqual(len(calls), 2)
        self.assertIn("StrictHostKeyChecking=yes", calls[0][0])
        self.assertEqual(calls[0][1]["stdin"], subprocess.DEVNULL)
        self.assertEqual(
            calls[0][1]["env"],
            {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(
            calls[0][0][-10:-6],
            ["/usr/bin/python3", "-B", "-s",
             "/srv/s40/a6000_phone_route_control.py"],
        )

    def test_stale_phone_boot_id_is_rejected(self):
        def runner(*_args, **_kwargs):
            return self.result(
                "load",
                lambda value: value.__setitem__("op15_boot_id", "stale"),
            )

        with self.assertRaisesRegex(Exception, "op15_boot_id mismatch"):
            run_control(self.config, "load", "model-a", runner)

    def test_ssh_stderr_is_rejected(self):
        def runner(*_args, **_kwargs):
            result = self.result("load")
            result.stderr = b"warning\n"
            return result

        with self.assertRaisesRegex(Exception, "wrote stderr"):
            run_control(self.config, "load", "model-a", runner)

    def test_mutated_known_hosts_is_rejected_before_ssh(self):
        self.known_hosts.write_text(
            "a6000 ssh-ed25519 CHANGED\n",
            encoding="ascii",
        )
        called = False

        def runner(*_args, **_kwargs):
            nonlocal called
            called = True
            return self.result("load")

        with self.assertRaisesRegex(Exception, "known_hosts"):
            run_control(self.config, "load", "model-a", runner)
        self.assertFalse(called)

    def test_missing_public_key_is_rejected(self):
        value = dict(self.value)
        Path(value["identity_public_key_path"]).unlink()
        path = self.config_path.with_name("missing-public-key.json")
        path.write_bytes(canonical_bytes(value))
        with self.assertRaisesRegex(Exception, "identity_public_key file"):
            load_config(path)

    def test_mutated_remote_package_is_rejected(self):
        value = dict(self.value)
        value["remote_identity_package"] = {
            **value["remote_identity_package"],
            "host_boot_id": "22222222-2222-4222-8222-222222222222",
        }
        path = self.config_path.with_name("bad-remote-package.json")
        path.write_bytes(canonical_bytes(value))
        with self.assertRaisesRegex(Exception, "package digest"):
            load_config(path)

    def test_private_key_must_match_pinned_public_key(self):
        replacement = self.identity_file.with_name("replacement")
        subprocess.run(
            [
                "/usr/bin/ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-f",
                str(replacement),
            ],
            check=True,
        )
        self.identity_file.write_bytes(replacement.read_bytes())
        self.identity_file.chmod(0o600)
        with self.assertRaisesRegex(Exception, "does not match"):
            run_control(
                self.config,
                "load",
                "model-a",
                lambda *_args, **_kwargs: self.result("load"),
            )

    def test_private_key_mode_and_symlink_are_rejected(self):
        self.identity_file.chmod(0o644)
        with self.assertRaisesRegex(Exception, "private file"):
            run_control(
                self.config,
                "load",
                "model-a",
                lambda *_args, **_kwargs: self.result("load"),
            )
        self.identity_file.chmod(0o600)
        target = self.identity_file.with_name("private-key-target")
        self.identity_file.rename(target)
        self.identity_file.symlink_to(target)
        with self.assertRaisesRegex(Exception, "private file"):
            run_control(
                self.config,
                "load",
                "model-a",
                lambda *_args, **_kwargs: self.result("load"),
            )

    def test_ssh_keygen_mutation_during_command_is_rejected(self):
        called = False

        def runner(*_args, **_kwargs):
            nonlocal called
            called = True
            self.ssh_keygen.write_bytes(b"changed\n")
            self.ssh_keygen.chmod(0o755)
            return self.result("load")

        with self.assertRaisesRegex(Exception, "ssh_keygen"):
            run_control(self.config, "load", "model-a", runner)
        self.assertTrue(called)


if __name__ == "__main__":
    unittest.main()
