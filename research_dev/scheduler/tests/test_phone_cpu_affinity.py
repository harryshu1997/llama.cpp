"""Explicit phone launch affinity is scoped and bound to its qualification."""

from dataclasses import replace
import os
from pathlib import Path
import shlex
from types import SimpleNamespace
import unittest

from research_dev.scheduler.adapters import DirectPhoneFfnSessionConfiguration, PhysicalAdapterError
from research_dev.scheduler.adapters.phone_session_ops.launch import _start_remote_command
from test_transport_profiles import identity


SHA = "sha256:" + "1" * 64


def configuration(mask=None, qualified_mask=None, qualified=True):
    proof = identity((SHA,))
    if qualified_mask is not None:
        proof = replace(proof, hardware_identity={**proof.hardware_identity,
            "phone_cpu_affinity": qualified_mask}, software_identity=dict(proof.software_identity))
    return DirectPhoneFfnSessionConfiguration(
        adb_path=Path("/usr/bin/true"), usb_close_path=Path("/usr/bin/true"),
        serial="synthetic-phone", adb_port=5037, session_script="/phone/session.sh",
        restore_script="/phone/restore.sh", session_root="/phone/run",
        worker_paths_by_artifact={SHA: "/phone/worker"}, model_paths_by_artifact={SHA: "/phone/model"},
        backend_by_device={"phone": "HTP0"}, minimum_usb_speed_mbps=5000,
        required_kernel_release="synthetic-kernel", cpu_affinity=mask,
        transport_qualification_identity=proof if qualified else None,
        transport_host_binary_path=Path("/usr/bin/true") if qualified else None,
        phone_boot_image_sha256=SHA if qualified else None,
    )


class PhoneCpuAffinityTests(unittest.TestCase):
    def test_default_does_not_change_launch(self):
        self.assertIsNone(configuration().cpu_affinity)
        self.assertNotIn("taskset", self.command(configuration()))

    def test_mask_requires_exact_qualification(self):
        for selected, measured in (("c0", None), ("c0", "80"), (None, "c0")):
            with self.subTest(selected=selected, measured=measured):
                with self.assertRaisesRegex(PhysicalAdapterError, "differs from qualification"):
                    configuration(selected, measured)
        self.assertEqual(configuration("c0", "c0").cpu_affinity, "c0")

    def test_missing_qualification_fails_closed(self):
        with self.assertRaisesRegex(PhysicalAdapterError, "identity is absent"):
            configuration("c0", qualified=False)

    def test_invalid_masks_fail_closed(self):
        for mask in ("", "0", "0x80", "C0", "80;false", "1" * 17, 128, True):
            with self.subTest(mask=mask):
                with self.assertRaisesRegex(PhysicalAdapterError, "affinity is invalid"):
                    configuration(mask)

    def test_launch_mask_is_subprocess_scoped(self):
        before = dict(os.environ)
        command = self.command(configuration("c0", "c0"))
        self.assertIn("env taskset c0 sh /phone/session.sh", command)
        self.assertEqual(dict(os.environ), before)

    @staticmethod
    def command(config):
        owner = SimpleNamespace(configuration=config,
            _worker_environment=lambda *args: (), _append_start_environment=lambda words: None)
        command = SimpleNamespace(ticket_id="ticket", adapter_parameters={"ffn_column_quantum": 32})
        execution = SimpleNamespace(columns=128, layers="1-2")
        *_, remote = _start_remote_command(owner, command, SimpleNamespace(artifact_sha256=SHA),
            execution, None, (), False, None, "/phone/worker", "/phone/model", "HTP0")
        return shlex.split(remote)[2]


if __name__ == "__main__":
    unittest.main()
